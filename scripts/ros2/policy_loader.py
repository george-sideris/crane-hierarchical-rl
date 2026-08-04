"""Load and run inference for all policy types (BC, BC->RL, SAC, heuristic).

Standalone module — no Isaac Lab dependency. Uses PyTorch for BC/RL models
and stable-baselines3 for SAC.
"""

import math
import numpy as np
import torch


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def decode_action(action: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray,
                  cossin: bool = True) -> tuple:
    """Decode raw policy output to (x, y, z, yaw) in base frame.

    Args:
        action: raw policy output (4D or 5D).
        bounds_min: [x_min, y_min, z_min] workspace bounds.
        bounds_max: [x_max, y_max, z_max] workspace bounds.
        cossin: True for 5D cos/sin yaw encoding, False for 4D direct yaw.

    Returns:
        (x, y, z, yaw) in base frame coordinates (meters, radians).
    """
    a = np.tanh(action)

    x = bounds_min[0] + (a[0] + 1.0) / 2.0 * (bounds_max[0] - bounds_min[0])
    y = bounds_min[1] + (a[1] + 1.0) / 2.0 * (bounds_max[1] - bounds_min[1])
    z = bounds_min[2] + (a[2] + 1.0) / 2.0 * (bounds_max[2] - bounds_min[2])

    if cossin and len(action) >= 5:
        yaw = math.atan2(a[4], a[3]) / 2.0
    else:
        yaw = a[3] * (math.pi / 2.0)

    return float(x), float(y), float(z), float(yaw)


class PolicyBase:
    """Base class for all policy types."""

    def __init__(self, bounds_min: np.ndarray, bounds_max: np.ndarray, cossin: bool = True):
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.cossin = cossin

    def get_target(self, obs: torch.Tensor) -> tuple:
        """Run inference and return decoded (x, y, z, yaw) in base frame."""
        raise NotImplementedError


class RslRlPolicy(PolicyBase):
    """BC or BC->RL policy loaded from RSL-RL .pt checkpoint."""

    def __init__(self, checkpoint_path: str, bounds_min: np.ndarray, bounds_max: np.ndarray,
                 cossin: bool = True, device: str = "auto"):
        super().__init__(bounds_min, bounds_max, cossin)
        self.device = torch.device(_resolve_device(device))

        # Load RSL-RL checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        model_state = ckpt.get("model_state_dict", ckpt)

        # Infer dimensions from state dict
        action_dim = None
        for key in model_state:
            if "actor" in key and "weight" in key:
                action_dim = model_state[key].shape[0]
        if action_dim is None:
            action_dim = 5 if cossin else 4

        # Try to load — caller may need to provide exact model config
        self.model = self._build_model(model_state, action_dim)
        self.model.eval()
        self.model.to(self.device)
        print(f"[PolicyLoader] Loaded RSL-RL policy from {checkpoint_path}")

    def _build_model(self, state_dict: dict, action_dim: int):
        """Reconstruct model from state dict keys."""
        # Detect if this is a PointNet model (has encoder keys) or MLP
        has_encoder = any("encoder" in k for k in state_dict)

        if has_encoder:
            PointNetActorCritic = self._load_pointnet_class()
            model = PointNetActorCritic(
                num_actor_obs=3072,   # 1024 * 3
                num_critic_obs=3072,
                num_actions=action_dim,
                num_points=1024,
                encoder_features=256,
                actor_hidden_dims=[128, 64],
                critic_hidden_dims=[128, 64],
                norm_type="batchnorm",
            )
        else:
            # Plain MLP — use RSL-RL's ActorCritic
            from rsl_rl.modules import ActorCritic
            model = ActorCritic(
                num_obs=128,  # pose-based
                num_actions=action_dim,
                actor_hidden_dims=[256, 128, 64],
                critic_hidden_dims=[256, 128, 64],
            )

        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            print(f"[PolicyLoader] Missing keys (OK for actor-only): {result.missing_keys[:5]}...")
        return model

    @staticmethod
    def _load_pointnet_class():
        """Import PointNetActorCritic directly by file path to avoid crane_testbed package deps."""
        import importlib.util, os
        _pn_path = os.path.join(os.path.dirname(__file__), "..", "..",
                                "source", "crane_testbed", "crane_testbed",
                                "agents", "pointnet_actor_critic.py")
        _pn_path = os.path.abspath(_pn_path)
        _spec = importlib.util.spec_from_file_location("pointnet_actor_critic", _pn_path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        return _mod.PointNetActorCritic

    @torch.no_grad()
    def get_target(self, obs: torch.Tensor) -> tuple:
        obs = obs.unsqueeze(0).to(self.device) if obs.dim() == 1 else obs.to(self.device)
        action = self.model.act_inference(obs)
        return decode_action(action[0].cpu().numpy(), self.bounds_min, self.bounds_max, self.cossin)


class SACPolicy(PolicyBase):
    """SAC policy loaded from SB3 .zip checkpoint."""

    def __init__(self, checkpoint_path: str, bounds_min: np.ndarray, bounds_max: np.ndarray,
                 cossin: bool = True, device: str = "auto"):
        super().__init__(bounds_min, bounds_max, cossin)
        from stable_baselines3 import SAC
        self.model = SAC.load(checkpoint_path, device=_resolve_device(device))
        print(f"[PolicyLoader] Loaded SAC policy from {checkpoint_path}")

    def get_target(self, obs: torch.Tensor) -> tuple:
        obs_np = obs.cpu().numpy()
        if obs_np.ndim == 1:
            obs_np = obs_np.reshape(1, -1)
        action, _ = self.model.predict(obs_np, deterministic=True)
        return decode_action(action[0], self.bounds_min, self.bounds_max, self.cossin)


class HeuristicPolicy(PolicyBase):
    """Heuristic: target the highest point in the point cloud."""

    def __init__(self, bounds_min: np.ndarray, bounds_max: np.ndarray, cossin: bool = True):
        super().__init__(bounds_min, bounds_max, cossin)

    def get_target(self, obs: torch.Tensor) -> tuple:
        """Select highest point in the point cloud as grasp target."""
        pc = obs.cpu().numpy().reshape(-1, 3)  # (1024, 3)
        # Filter out zero-padding
        valid = np.any(pc != 0, axis=1)
        if not valid.any():
            # Fallback: center of workspace
            cx = (self.bounds_min[0] + self.bounds_max[0]) / 2
            cy = (self.bounds_min[1] + self.bounds_max[1]) / 2
            cz = (self.bounds_min[2] + self.bounds_max[2]) / 2
            return cx, cy, cz, 0.0
        pc_valid = pc[valid]
        # Highest point (max Z in base frame)
        top_idx = np.argmax(pc_valid[:, 2])
        x, y, z = pc_valid[top_idx]
        # Yaw: align with local PCA of nearby points
        yaw = self._estimate_yaw(pc_valid, x, y, z)
        return float(x), float(y), float(z), float(yaw)

    @staticmethod
    def _estimate_yaw(points: np.ndarray, x: float, y: float, z: float,
                      radius: float = 0.3) -> float:
        """Estimate log yaw from PCA of nearby points."""
        dists = np.sqrt((points[:, 0] - x)**2 + (points[:, 1] - y)**2)
        nearby = points[dists < radius]
        if len(nearby) < 5:
            return 0.0
        # PCA on XY plane
        centered = nearby[:, :2] - nearby[:, :2].mean(axis=0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal = eigvecs[:, np.argmax(eigvals)]
        return float(math.atan2(principal[1], principal[0]))


def load_policy(policy_type: str, checkpoint_path,
                bounds_min: np.ndarray, bounds_max: np.ndarray,
                cossin: bool = True, device: str = "auto") -> PolicyBase:
    """Factory function to load any policy type.

    Args:
        policy_type: "bc", "bcrl", "rl", "sac", or "heuristic".
        checkpoint_path: path to model file (.pt for RSL-RL, .zip for SAC, None for heuristic).
        bounds_min: [x_min, y_min, z_min] workspace bounds.
        bounds_max: [x_max, y_max, z_max] workspace bounds.
        cossin: True for 5D cos/sin yaw encoding.
        device: "cuda" or "cpu".
    """
    if policy_type in ("bc", "bcrl", "rl"):
        return RslRlPolicy(checkpoint_path, bounds_min, bounds_max, cossin, device)
    elif policy_type == "sac":
        return SACPolicy(checkpoint_path, bounds_min, bounds_max, cossin, device)
    elif policy_type == "heuristic":
        return HeuristicPolicy(bounds_min, bounds_max, cossin)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
