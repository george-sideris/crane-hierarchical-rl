"""Load and run inference for all policy types (BC, BC->RL, SAC, heuristic).

Standalone module with no Isaac Lab dependency. Uses PyTorch for BC/RL
models and stable-baselines3 for SAC.
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
            # Plain MLP. Use RSL-RL's ActorCritic.
            from rsl_rl.modules import ActorCritic
            model = ActorCritic(
                num_obs=128,  # pose-based
                num_actions=action_dim,
                actor_hidden_dims=[256, 128, 64],
                critic_hidden_dims=[256, 128, 64],
            )

        # Drop checkpoint entries whose shape doesn't match the model (e.g. a vestigial `std`
        # sized to a stale action_dim from BC training). Deployment inference is deterministic
        # (act_inference = actor mean), so the action-noise `std` is unused; a size-mismatched
        # key would otherwise abort load_state_dict even under strict=False.
        model_sd = model.state_dict()
        filtered, dropped = {}, []
        for k, v in state_dict.items():
            if k in model_sd and model_sd[k].shape != v.shape:
                dropped.append(f"{k}{tuple(v.shape)}!=model{tuple(model_sd[k].shape)}")
            else:
                filtered[k] = v
        if dropped:
            print(f"[PolicyLoader] Dropped shape-mismatched keys (unused at inference): {dropped}")
        result = model.load_state_dict(filtered, strict=False)
        if result.missing_keys:
            print(f"[PolicyLoader] Missing keys (OK for actor-only): {result.missing_keys[:5]}...")
        return model

    @staticmethod
    def _load_pointnet_class():
        """Import PointNetActorCritic from the sibling file."""
        import importlib.util, os
        _pn_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "pointnet_actor_critic.py")
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


    def __init__(self, bounds_min: np.ndarray, bounds_max: np.ndarray, cossin: bool = True,
                 dig: float = 0.30):
        super().__init__(bounds_min, bounds_max, cossin)
        # Dig below the observed surface. 0.30 = real-tuned (2026-07-27: shallow closes rake
        # and miss; executed successes sat 0.2-0.4 below surface). 0.056 = the sim expert's
        # log-center convention, settable for sim-convention-on-real experiments.
        self._dig = float(dig)

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
        # Candidate set: the middle of the pile lengthwise. Logs lie along base x in
        # the rack, so restricting candidates to the central band of the x-range makes
        # the heuristic grasp log CENTERS directly; a global argmax usually lands on a
        # log end at the crop edge. Fall back to the full cloud if the band is sparse.
        x_lo = self.bounds_min[0] + 0.25 * (self.bounds_max[0] - self.bounds_min[0])
        x_hi = self.bounds_max[0] - 0.25 * (self.bounds_max[0] - self.bounds_min[0])
        in_band = (pc_valid[:, 0] >= x_lo) & (pc_valid[:, 0] <= x_hi)
        candidates = pc_valid[in_band] if in_band.sum() >= 20 else pc_valid
        # Highest SUPPORTED candidate: walk from the top down and take the first with
        # enough xy-neighbors, so an isolated depth flyer cannot become the target.
        order = np.argsort(candidates[:, 2])[::-1]
        top_idx = order[0]
        for idx in order[:20]:
            d = np.sqrt((candidates[:, 0] - candidates[idx, 0])**2
                        + (candidates[:, 1] - candidates[idx, 1])**2)
            if (d < 0.15).sum() >= 3:   # includes the candidate itself
                top_idx = idx
                break
        x, y, z = candidates[top_idx]
        # Log axis direction from local PCA (full cloud: the log line may extend
        # beyond the selection band)
        axis_yaw = self._estimate_yaw(pc_valid, x, y, z)
        x, y, z = self._center_along_log(pc_valid, x, y, z, axis_yaw)
        # Dig in: command the grasp BELOW the observed top surface. A surface-level z
        # only catches wood while rows protrude; once the pile flattens the closing
        # tongs sweep the top centimeters and miss (real run 2026-07-27: 4 good grasps,
        # then repeated misses on the flattened middle until the rakes lowered the pile).
        # Successful executed grasps sat 0.2-0.4 m below the local surface. Ground
        # protection is NOT applied here anymore: the old per-cloud p5+0.08 floor was
        # occlusion-biased (on a full pile the lowest visible points are pile, not bed;
        # 2026-07-29 run: over-clamped 29/50 cycles by up to 0.13 m) and, being inside
        # the heuristic only, made the baseline comparison asymmetric. The platform-level
        # bed floor in crane_policy_node (bed_z + bed_margin, all policy types) is the
        # single ground guard now.
        z = z - self._dig
        # Grapple convention: the commanded yaw that wraps the tongs around a log is
        # PERPENDICULAR to the log axis (executed real grasps on logs lying along x
        # all carry yaw ~ +-90 deg). Wrap to (-pi, pi].
        yaw = math.atan2(math.sin(axis_yaw + math.pi / 2.0),
                         math.cos(axis_yaw + math.pi / 2.0))
        return float(x), float(y), float(z), float(yaw)

    @staticmethod
    def _center_along_log(points: np.ndarray, x: float, y: float, z: float,
                          yaw: float, corridor: float = 0.2, z_band: float = 0.25):
        """Slide the pick along the log axis to the middle of the visible log run.

        The raw top pick is length-biased: a few cm of cloud tilt (lean or residual
        extrinsic pitch) consistently puts the height argmax at one END of the log
        (real runs: every pick at the front of the band). Height still chooses WHICH
        log; this centers WHERE along it: points within a corridor of the axis line
        and near the pick's height are projected onto the axis and the pick moves to
        the midpoint of their robust extent, so the tongs wrap the log middle."""
        a = np.array([math.cos(yaw), math.sin(yaw)])
        rel = points[:, :2] - np.array([x, y])
        t = rel @ a
        perp = np.abs(rel[:, 0] * a[1] - rel[:, 1] * a[0])
        m = (perp < corridor) & (np.abs(points[:, 2] - z) < z_band)
        if m.sum() < 5:
            return float(x), float(y), float(z)
        t_mid = 0.5 * (np.percentile(t[m], 5) + np.percentile(t[m], 95))
        nx = x + t_mid * a[0]
        ny = y + t_mid * a[1]
        near = m & (np.abs(t - t_mid) < 0.15)
        nz = float(np.percentile(points[near, 2], 90)) if near.sum() >= 3 else z
        return float(nx), float(ny), float(nz)

    @staticmethod
    def _refine_on_cluster(points: np.ndarray, x: float, y: float, z: float,
                           yaw: float, corridor: float = 0.2, t_max: float = 0.5,
                           z_band: float = 0.12) -> tuple:
        """Denoise the target with the local same-height cluster instead of trusting
        the single selected point.

        Cluster: points at the target's height within a corridor of the log axis and
        within t_max of the target along it (local only; no long-range sliding, so
        the target stays on observed wood). Position is the cluster mean and grasp
        depth its 80th-percentile z, so one noisy point sets neither."""
        u = np.array([math.cos(yaw), math.sin(yaw)])
        rel = points[:, :2] - np.array([x, y])
        t = rel[:, 0] * u[0] + rel[:, 1] * u[1]
        perp = rel[:, 1] * u[0] - rel[:, 0] * u[1]
        near = ((np.abs(perp) < corridor) & (np.abs(t) < t_max)
                & (np.abs(points[:, 2] - z) < z_band))
        if near.sum() < 5:
            return x, y, z
        t_mid = float(np.mean(t[near]))
        perp_mid = float(np.mean(perp[near]))
        z_row = float(np.percentile(points[near, 2], 80))
        return (x + t_mid * u[0] - perp_mid * u[1],
                y + t_mid * u[1] + perp_mid * u[0], z_row)

    @staticmethod
    def _estimate_yaw(points: np.ndarray, x: float, y: float, z: float,
                      radius: float = 0.6, z_band: float = 0.12,
                      min_elongation: float = 3.0, default_yaw: float = 0.0) -> float:
        """Estimate log yaw from PCA of points on the target log's top surface.

        Gating the neighborhood by height (|dz| < z_band) keeps the pile slope
        below the target out of the fit: when the target is a protruding log,
        the surviving points trace its axis as a line in xy. Without the gate,
        most neighbors belong to lower logs and the direction is noise (real
        capture 2026-07-16: 116 deg on a log lying along x).

        The PCA direction is only trusted when the neighborhood is clearly
        elongated (eigenvalue ratio >= min_elongation). On a flat pile area the
        same-height points form an isotropic patch whose principal direction is
        noise (real capture 2026-07-20: elongation 1.7, yaw 112 deg on logs
        along x); there the estimate falls back to default_yaw, the rack's
        storage orientation (logs lie along base x)."""
        dists = np.sqrt((points[:, 0] - x)**2 + (points[:, 1] - y)**2)
        nearby = points[(dists < radius) & (np.abs(points[:, 2] - z) < z_band)]
        if len(nearby) < 5:
            # Not enough same-height support; retry with the height gate widened
            # rather than silently returning an arbitrary yaw.
            nearby = points[(dists < radius) & (np.abs(points[:, 2] - z) < 2 * z_band)]
            if len(nearby) < 5:
                return default_yaw
        # PCA on XY plane
        centered = nearby[:, :2] - nearby[:, :2].mean(axis=0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        if eigvals.max() < min_elongation * max(eigvals.min(), 1e-9):
            return default_yaw
        principal = eigvecs[:, np.argmax(eigvals)]
        return float(math.atan2(principal[1], principal[0]))


class ScoringHeadPolicy(PolicyBase):
    """Per-point SCORING policy (arch 'scoring_head_v1', see scoring_head.py).

    Unlike the regression policies this one does NOT emit an action in the arctanh-encoded action
    box: it scores every observed point, takes the argmax, and reads the grasp off that point, so
    get_target() already has metres and bounds_min/max are not used to decode. Two consequences
    worth knowing at deployment:
      * the commanded xy is always ON observed material - aiming at empty space is unrepresentable
        (the deployed regression BC put 27% of its 2026-08-03 targets on zero observed points)
      * z is a residual from the chosen surface point, not an absolute coordinate, so it does not
        inherit the regression head's label-clamping behaviour
    The bed-floor clamp still applies downstream in _plan_from_target, as for every policy type.
    """

    def __init__(self, checkpoint_path: str, bounds_min: np.ndarray, bounds_max: np.ndarray,
                 cossin: bool = True, device: str = "auto", support_frac: float = 0.0):
        super().__init__(bounds_min, bounds_max, cossin)
        self.device = torch.device(_resolve_device(device))
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        arch = ckpt.get("arch")
        if arch != "scoring_head_v1":
            raise ValueError(f"{checkpoint_path}: expected arch 'scoring_head_v1', got {arch!r}")

        import importlib.util
        import os as _os
        _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "scoring_head.py")
        _spec = importlib.util.spec_from_file_location("scoring_head", _p)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)

        self.num_points = int(ckpt.get("num_points", 1024))
        self.model = _mod.ScoringGraspPolicy(num_points=self.num_points).to(self.device)
        # candidate mask: only points inside the ACTION box are selectable; margin-crop points
        # are context. The net sees training coords (shim), so the training-coords box applies.
        self.model.candidate_min = tuple(float(v) for v in bounds_min)
        self.model.candidate_max = tuple(float(v) for v in bounds_max)
        # support gate: OFF by default (0.0 = raw argmax); opt in with support_frac=0.25.
        # scoring_v1 (tight/1024) needs it under noise. The candidate mask above is NOT the
        # gate - reachability stays enforced regardless.
        self.model.support_frac = float(support_frac)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        print(f"[PolicyLoader] Loaded SCORING head from {checkpoint_path} "
              f"(num_points={self.num_points}, dig={ckpt.get('dig', 0.0)})")

    @torch.no_grad()
    def get_target(self, obs: torch.Tensor) -> tuple:
        obs = obs.to(self.device).reshape(1, -1, 3)
        if obs.shape[1] != self.num_points:
            raise ValueError(f"scoring head expects {self.num_points} points, got {obs.shape[1]}; "
                             f"the node's num_points must match the checkpoint")
        x, y, z, yaw = self.model.act(obs)[0].cpu().numpy().tolist()
        return float(x), float(y), float(z), float(yaw)


def load_policy(policy_type: str, checkpoint_path,
                bounds_min: np.ndarray, bounds_max: np.ndarray,
                cossin: bool = True, device: str = "auto",
                heuristic_dig: float = 0.30,
                support_frac: float = 0.0) -> PolicyBase:
    """Factory function to load any policy type.

    Args:
        policy_type: "bc", "bcrl", "rl", "sac", "scoring", or "heuristic".
                     "scoring" = the per-point argmax head; its checkpoint carries
                     arch='scoring_head_v1' and its own num_points, which the node must match.
        checkpoint_path: path to model file (.pt for RSL-RL, .zip for SAC, None for heuristic).
        bounds_min: [x_min, y_min, z_min] workspace bounds.
        bounds_max: [x_max, y_max, z_max] workspace bounds.
        cossin: True for 5D cos/sin yaw encoding.
        device: "cuda" or "cpu".
    """
    if policy_type in ("bc", "bcrl", "rl"):
        return RslRlPolicy(checkpoint_path, bounds_min, bounds_max, cossin, device)
    elif policy_type == "scoring":
        return ScoringHeadPolicy(checkpoint_path, bounds_min, bounds_max, cossin, device,
                                 support_frac=support_frac)
    elif policy_type == "sac":
        return SACPolicy(checkpoint_path, bounds_min, bounds_max, cossin, device)
    elif policy_type == "heuristic":
        return HeuristicPolicy(bounds_min, bounds_max, cossin, dig=heuristic_dig)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
