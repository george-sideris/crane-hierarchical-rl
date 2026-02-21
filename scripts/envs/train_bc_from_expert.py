#!/usr/bin/env python3
"""
Behavioral Cloning from Expert Heuristic (Full 4D)

Collects demonstrations from the expert heuristic and trains a BC policy.
Uses full 4D action space [x, y, z, yaw] for pretraining before RL fine-tuning.

Usage:
    ./isaaclab.sh -p crane_testbed/scripts/envs/train_bc_from_expert.py \
        --num_envs 8 --num_episodes 200 --epochs 100 --headless
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from datetime import datetime
import numpy as np
from pathlib import Path

# Add the envs path
crane_scripts_path = Path(__file__).resolve().parent
sys.path.insert(0, str(crane_scripts_path))

# Parse args before IsaacLab imports
parser = argparse.ArgumentParser(description="Train BC policy from expert heuristic")
parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments")
parser.add_argument("--num_episodes", type=int, default=200, help="Number of episodes to collect")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: logs/bc_policy)")
parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128, 64], help="Hidden layer dims")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
parser.add_argument("--device", type=str, default="cuda:0", help="Device")
parser.add_argument("--domain_randomization", action="store_true", help="Enable domain randomization")
parser.add_argument("--headless", action="store_true", help="Run without visualization")
# Expert randomization options
parser.add_argument("--top_k", type=int, default=1, help="Pick randomly from top K logs (1=always highest)")
parser.add_argument("--pos_noise", type=float, default=0.0, help="Gaussian noise std for target position (cm)")
parser.add_argument("--yaw_noise", type=float, default=0.0, help="Gaussian noise std for target yaw (degrees)")
# Collection and checkpointing options
parser.add_argument("--save_interval", type=int, default=10, help="Save checkpoint every N episodes")
parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint directory (e.g., logs/bc_policy/bc_20260117_143052)")
parser.add_argument("--collect_only", action="store_true", help="Only collect data, skip training")
args_cli, _ = parser.parse_known_args()

# IsaacLab imports
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

# Import the FULL 4D environment
import crane_rl_env_full
from crane_rl_env_full import CraneDirectEnvFull, CraneDirectEnvCfgFull


class BCPolicy(nn.Module):
    """MLP policy matching RSL-RL architecture."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list = [256, 128, 64]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, obs):
        return self.network(obs)


class Expert4D:
    """Expert that computes 4D actions [x, y, z, yaw] using env's existing methods."""

    def __init__(self, env, top_k: int = 1, pos_noise_cm: float = 0.0, yaw_noise_deg: float = 0.0):
        self.env = env
        self.top_k = max(1, top_k)
        self.pos_noise = pos_noise_cm / 100.0  # cm -> meters
        self.yaw_noise = np.radians(yaw_noise_deg)  # degrees -> radians

    def _compute_yaw_for_log(self, env_i: int, log_quat_w: torch.Tensor) -> float:
        """
        Compute optimal yaw joint angle to align grapple with specific log.

        This is similar to env._calc_optimal_yaw_feedback but uses the provided
        log quaternion instead of looking up the highest log.
        """
        import math
        env = self.env

        def wrap(a: float) -> float:
            return (a + math.pi) % (2.0 * math.pi) - math.pi

        def quat_conj_wxyz(q):
            return (q[0], -q[1], -q[2], -q[3])

        def quat_mul_wxyz(a, b):
            aw, ax, ay, az = a
            bw, bx, by, bz = b
            return (
                aw*bw - ax*bx - ay*by - az*bz,
                aw*bx + ax*bw + ay*bz - az*by,
                aw*by - ax*bz + ay*bw + az*bx,
                aw*bz + ax*by - ay*bx + az*bw,
            )

        def yaw_from_quat_wxyz(qw, qx, qy, qz) -> float:
            return math.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))

        # Get current yaw joint value
        yaw_joint_ids, _ = env.crane.find_joints(["lowerpassive_to_basegrapple"])
        if len(yaw_joint_ids) == 0:
            return 0.0
        yaw_joint_id = int(yaw_joint_ids[0])
        current_yaw_joint = float(env.crane.data.joint_pos[env_i, yaw_joint_id].item())

        # Get base orientation (world->base transform)
        base_quat_w = env.crane.data.root_pose_w[env_i, 3:7]
        q_base_w = (float(base_quat_w[0]), float(base_quat_w[1]),
                    float(base_quat_w[2]), float(base_quat_w[3]))
        q_base_inv = quat_conj_wxyz(q_base_w)

        # Get current basegrapple orientation
        bg_pose_w = env.crane.data.body_pose_w[env_i, env._basegrapple_body_id]
        bg_quat_w = bg_pose_w[3:7]
        q_bg_b = quat_mul_wxyz(q_base_inv, (float(bg_quat_w[0]), float(bg_quat_w[1]),
                                           float(bg_quat_w[2]), float(bg_quat_w[3])))
        current_bg_z_rotation = yaw_from_quat_wxyz(*q_bg_b)

        # Transform log orientation to base frame
        q_log_w = (float(log_quat_w[0]), float(log_quat_w[1]),
                   float(log_quat_w[2]), float(log_quat_w[3]))
        q_log_b = quat_mul_wxyz(q_base_inv, q_log_w)
        log_yaw_b = yaw_from_quat_wxyz(*q_log_b)

        # Calculate rotation diff (no offset - fingers align with log's long axis)
        rotation_diff = wrap(log_yaw_b - current_bg_z_rotation)

        # Apply to yaw joint with 180° flip option
        target_yaw1 = current_yaw_joint + rotation_diff
        target_yaw2 = current_yaw_joint + rotation_diff + math.pi

        # Choose minimal rotation
        yaw1 = wrap(target_yaw1)
        yaw2 = wrap(target_yaw2)
        diff1 = abs(wrap(yaw1 - current_yaw_joint))
        diff2 = abs(wrap(yaw2 - current_yaw_joint))

        return yaw1 if diff1 <= diff2 else yaw2

    def get_action(self, env_i: int) -> torch.Tensor:
        """
        Compute 4D expert action [x, y, z, yaw] for target log.
        Uses env's existing _target_top_log_center_b for position.
        Computes yaw directly from selected log's orientation.
        """
        env = self.env

        # Use env's existing method to get target position
        log_pos_b, selected_id, log_quat_w = env._target_top_log_center_b(env_i)

        if selected_id == -1:
            return torch.zeros(4, device=env.device)

        # Store target position
        env._target_log_pos_b[env_i] = log_pos_b
        env._current_target_log_id[env_i] = selected_id

        # Get selected log's quaternion and compute yaw
        _, quat_w = env._get_logs_root_pose_w()
        if quat_w is not None:
            log_quat_w = quat_w[selected_id]
            env._target_log_quat_w[env_i] = log_quat_w
            # Get target basegrapple yaw in basemast frame (state-independent)
            # Policy predicts this, then env converts to joint position at runtime
            yaw_target = env._get_target_grapple_yaw_b(env_i)
        else:
            yaw_target = 0.0

        # Extract position
        x_target = log_pos_b[0].item()
        y_target = log_pos_b[1].item()
        z_target = log_pos_b[2].item()

        # Add noise
        if self.pos_noise > 0:
            x_target += np.random.normal(0, self.pos_noise)
            y_target += np.random.normal(0, self.pos_noise)
            z_target += np.random.normal(0, self.pos_noise)
        if self.yaw_noise > 0:
            yaw_target += np.random.normal(0, self.yaw_noise)

        # Get action bounds
        if not env._action_bounds_valid[env_i]:
            env._compute_action_space_bounds()

        min_b = env._action_bounds_min[env_i]
        max_b = env._action_bounds_max[env_i]

        # Normalize to [-1, 1] then apply arctanh
        def normalize_and_arctanh(val, vmin, vmax):
            range_v = vmax - vmin
            if range_v <= 0:
                return 0.0
            norm = 2.0 * (val - vmin) / range_v - 1.0
            norm = max(-0.999, min(0.999, norm))  # Clamp for arctanh
            return np.arctanh(norm)

        x_action = normalize_and_arctanh(x_target, min_b[0].item(), max_b[0].item())
        y_action = normalize_and_arctanh(y_target, min_b[1].item(), max_b[1].item())
        z_action = normalize_and_arctanh(z_target, min_b[2].item(), max_b[2].item())
        # Yaw is now normalized to [-π/2, π/2] for log symmetry
        yaw_norm = max(-0.999, min(0.999, yaw_target / (np.pi / 2)))
        yaw_action = np.arctanh(yaw_norm)

        return torch.tensor([x_action, y_action, z_action, yaw_action],
                           device=env.device, dtype=torch.float32)


def save_collection_checkpoint(output_dir: str, observations: list, actions: list,
                                episode_rewards: list, metrics_state: dict):
    """Save collection checkpoint for resume capability."""
    os.makedirs(output_dir, exist_ok=True)

    # Save data arrays
    if observations:
        np.save(os.path.join(output_dir, "observations.npy"), np.stack(observations))
        np.save(os.path.join(output_dir, "actions.npy"), np.stack(actions))
    np.save(os.path.join(output_dir, "episode_rewards.npy"), np.array(episode_rewards))

    # Save metrics state for resume
    with open(os.path.join(output_dir, "checkpoint.json"), "w") as f:
        json.dump(metrics_state, f, indent=2)

    print(f"[BC] Checkpoint saved: {len(observations)} samples, {metrics_state['episodes_done']} episodes", flush=True)


def load_collection_checkpoint(checkpoint_dir: str):
    """Load collection checkpoint to resume."""
    obs_path = os.path.join(checkpoint_dir, "observations.npy")
    act_path = os.path.join(checkpoint_dir, "actions.npy")
    rew_path = os.path.join(checkpoint_dir, "episode_rewards.npy")
    state_path = os.path.join(checkpoint_dir, "checkpoint.json")

    if not os.path.exists(state_path):
        return None, None, None, None

    with open(state_path, "r") as f:
        metrics_state = json.load(f)

    observations = list(np.load(obs_path)) if os.path.exists(obs_path) else []
    actions = list(np.load(act_path)) if os.path.exists(act_path) else []
    episode_rewards = list(np.load(rew_path)) if os.path.exists(rew_path) else []

    return observations, actions, episode_rewards, metrics_state


def collect_demonstrations(env, num_episodes: int, output_dir: str, save_interval: int = 10,
                           top_k: int = 1, pos_noise_cm: float = 0.0, yaw_noise_deg: float = 0.0,
                           resume_from: str = None):
    """Collect (obs, action) pairs from expert with incremental saving.

    Args:
        env: The environment
        num_episodes: Total episodes to collect
        output_dir: Directory to save checkpoints
        save_interval: Save checkpoint every N episodes
        top_k: Pick from top K logs
        pos_noise_cm: Position noise in cm
        yaw_noise_deg: Yaw noise in degrees
        resume_from: Path to checkpoint directory to resume from
    """
    # Try to resume from checkpoint
    if resume_from and os.path.exists(resume_from):
        loaded = load_collection_checkpoint(resume_from)
        if loaded[0] is not None:
            observations, actions, episode_rewards, metrics_state = loaded
            episodes_done = metrics_state['episodes_done']
            successful_grasps = metrics_state['successful_grasps']
            failed_grasps = metrics_state['failed_grasps']
            total_logs_grasped = metrics_state['total_logs_grasped']
            total_alignment = metrics_state['total_alignment']
            piles_fully_cleared = metrics_state['piles_fully_cleared']
            grasps_to_clear = metrics_state['grasps_to_clear']
            clearing_percentages = metrics_state['clearing_percentages']
            logs_per_episode = metrics_state['logs_per_episode']
            print(f"[BC] Resumed from checkpoint: {episodes_done} episodes, {len(observations)} samples", flush=True)
        else:
            print(f"[BC] No valid checkpoint found at {resume_from}, starting fresh", flush=True)
            resume_from = None

    if not resume_from:
        observations = []
        actions = []
        episode_rewards = []
        episodes_done = 0
        successful_grasps, failed_grasps = 0, 0
        total_logs_grasped = 0
        total_alignment = 0.0
        piles_fully_cleared = 0
        grasps_to_clear = []
        clearing_percentages = []
        logs_per_episode = []

    current_reward = torch.zeros(env.num_envs, device=env.device)
    episode_grasps = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
    episode_logs_cleared = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)  # Track logs per episode
    episode_starting_logs = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)  # Track starting logs per episode

    expert = Expert4D(env, top_k=top_k, pos_noise_cm=pos_noise_cm, yaw_noise_deg=yaw_noise_deg)

    print(f"[BC] Collecting {num_episodes} episodes (4D actions: x, y, z, yaw)", flush=True)
    print(f"[BC] Saving checkpoints every {save_interval} episodes to {output_dir}", flush=True)
    print(f"[BC] Only keeping SUCCESSFUL grasps (reward > 0)", flush=True)
    if top_k > 1:
        print(f"[BC] Picking from top {top_k} logs", flush=True)
    if pos_noise_cm > 0 or yaw_noise_deg > 0:
        print(f"[BC] Noise: pos={pos_noise_cm:.1f}cm, yaw={yaw_noise_deg:.1f}deg", flush=True)

    env.reset()

    # Check log counts AFTER reset (when they're initialized)
    has_variable_logs = hasattr(env, '_per_env_log_counts') and env._per_env_log_counts is not None
    if has_variable_logs:
        log_counts = [int(env._per_env_log_counts[i].item()) for i in range(env.num_envs)]
        print(f"[BC] Logs per pile: {np.mean(log_counts):.0f} avg (range: {min(log_counts)}-{max(log_counts)})", flush=True)
        # Initialize starting logs for each episode
        for i in range(env.num_envs):
            episode_starting_logs[i] = int(env._per_env_log_counts[i].item())
    else:
        default_logs = int(env._per_env_target) if hasattr(env, '_per_env_target') and env._per_env_target > 0 else 200
        print(f"[BC] Logs per pile: {default_logs}", flush=True)
        episode_starting_logs[:] = default_logs

    while episodes_done < num_episodes and simulation_app.is_running():
        with torch.inference_mode():
            # BEFORE step: get current obs and expert actions (all envs at HOVER_UP)
            obs_batch = env._build_target_selection_obs().cpu().numpy()
            expert_actions = torch.stack([expert.get_action(i) for i in range(env.num_envs)])
            action_batch = expert_actions.cpu().numpy()

            # Step runs entire cycle and returns reward
            _, rew, terminated, truncated, _ = env.step(expert_actions)

            # AFTER step: check outcomes for each env
            for env_i in range(env.num_envs):
                episode_grasps[env_i] += 1
                logs_this_grasp = int(env._prev_logs_grasped[env_i].item())
                alignment_this_grasp = env._prev_grasp_alignment[env_i].item()

                # Track logs cleared this episode (before potential reset)
                episode_logs_cleared[env_i] += logs_this_grasp

                if logs_this_grasp > 0:  # Successful grasp
                    observations.append(obs_batch[env_i])
                    actions.append(action_batch[env_i])
                    successful_grasps += 1
                    total_logs_grasped += logs_this_grasp
                    total_alignment += alignment_this_grasp
                else:
                    failed_grasps += 1

            current_reward += rew
            done = terminated | truncated

            for env_i in range(env.num_envs):
                if done[env_i]:
                    episode_rewards.append(current_reward[env_i].item())

                    # Use TRACKED starting logs (not post-reset count which is already updated!)
                    total_logs_this_env = int(episode_starting_logs[env_i].item())
                    if total_logs_this_env <= 0:
                        total_logs_this_env = 200  # Fallback

                    # Track pile clearing progress using tracked logs
                    logs_cleared = int(episode_logs_cleared[env_i].item())
                    clear_pct = (logs_cleared / max(1, total_logs_this_env)) * 100
                    clearing_percentages.append(clear_pct)
                    logs_per_episode.append(total_logs_this_env)

                    # Check if pile was fully cleared (all logs removed)
                    if logs_cleared >= total_logs_this_env:
                        piles_fully_cleared += 1
                        grasps_to_clear.append(int(episode_grasps[env_i].item()))

                    # Reset per-episode tracking and update starting logs for next episode
                    current_reward[env_i] = 0.0
                    episode_grasps[env_i] = 0
                    episode_logs_cleared[env_i] = 0
                    # Update starting logs for the NEW episode (env already reset, so _per_env_log_counts has new value)
                    if has_variable_logs:
                        episode_starting_logs[env_i] = int(env._per_env_log_counts[env_i].item())
                    episodes_done += 1

                    # Print progress after every episode
                    grasp_rate = successful_grasps / max(1, successful_grasps + failed_grasps) * 100
                    avg_clear_pct = np.mean(clearing_percentages) if clearing_percentages else 0.0
                    avg_throughput = total_logs_grasped / max(1, successful_grasps)
                    avg_align = total_alignment / max(1, successful_grasps)

                    print(f"[BC] Ep {episodes_done}/{num_episodes} | "
                          f"Samples: {len(observations)} | "
                          f"Grasp: {grasp_rate:.0f}% | "
                          f"Clear: {avg_clear_pct:.1f}% | "
                          f"Thru: {avg_throughput:.1f} | "
                          f"Align: {avg_align:.2f}", flush=True)

                    # Save checkpoint periodically
                    if episodes_done % save_interval == 0:
                        metrics_state = {
                            'episodes_done': episodes_done,
                            'successful_grasps': successful_grasps,
                            'failed_grasps': failed_grasps,
                            'total_logs_grasped': total_logs_grasped,
                            'total_alignment': total_alignment,
                            'piles_fully_cleared': piles_fully_cleared,
                            'grasps_to_clear': grasps_to_clear,
                            'clearing_percentages': clearing_percentages,
                            'logs_per_episode': logs_per_episode,
                        }
                        save_collection_checkpoint(output_dir, observations, actions, episode_rewards, metrics_state)

                    if episodes_done >= num_episodes:
                        break

    # Final summary
    grasp_rate = successful_grasps / max(1, successful_grasps + failed_grasps) * 100
    avg_clear_pct = np.mean(clearing_percentages) if clearing_percentages else 0.0
    full_clear_rate = piles_fully_cleared / max(1, episodes_done) * 100
    avg_throughput = total_logs_grasped / max(1, successful_grasps)
    avg_align = total_alignment / max(1, successful_grasps)

    print(f"\n[BC] ====== COLLECTION SUMMARY ======", flush=True)
    print(f"[BC] Episodes: {episodes_done}", flush=True)
    print(f"[BC] Samples collected: {len(observations)}", flush=True)
    if has_variable_logs and logs_per_episode:
        print(f"[BC] Logs per pile: {np.mean(logs_per_episode):.1f} avg ({np.min(logs_per_episode)}-{np.max(logs_per_episode)} range)", flush=True)
    print(f"[BC] Grasp success rate: {grasp_rate:.1f}% ({successful_grasps}/{successful_grasps + failed_grasps})", flush=True)
    print(f"[BC] Avg pile cleared: {avg_clear_pct:.1f}% of logs per episode", flush=True)
    print(f"[BC] Full clears: {full_clear_rate:.1f}% ({piles_fully_cleared}/{episodes_done} episodes)", flush=True)
    print(f"[BC] Avg throughput: {avg_throughput:.2f} logs/grasp", flush=True)
    print(f"[BC] Avg alignment: {avg_align:.3f}", flush=True)
    print(f"[BC] ===================================", flush=True)

    # Build metrics dict
    metrics = {
        "collection": {
            "episodes": episodes_done,
            "samples": len(observations),
            "logs_per_pile_avg": float(np.mean(logs_per_episode)) if logs_per_episode else 0,
            "logs_per_pile_min": int(np.min(logs_per_episode)) if logs_per_episode else 0,
            "logs_per_pile_max": int(np.max(logs_per_episode)) if logs_per_episode else 0,
            "variable_log_counts": has_variable_logs,
            "top_k": top_k,
            "pos_noise_cm": pos_noise_cm,
            "yaw_noise_deg": yaw_noise_deg,
        },
        "grasp": {
            "success_rate": grasp_rate,
            "successful": successful_grasps,
            "failed": failed_grasps,
            "total": successful_grasps + failed_grasps,
        },
        "pile_clearing": {
            "avg_cleared_pct": avg_clear_pct,
            "full_clear_rate": full_clear_rate,
            "full_clears": piles_fully_cleared,
        },
        "performance": {
            "avg_throughput": avg_throughput,
            "avg_alignment": avg_align,
        },
        "episode_rewards": episode_rewards,
        "clearing_percentages": clearing_percentages,
    }

    return np.stack(observations), np.stack(actions), metrics


def train_bc_policy(obs: np.ndarray, actions: np.ndarray,
                    hidden_dims: list, batch_size: int, lr: float,
                    epochs: int, device: str):
    """Train BC policy via supervised learning."""
    print(f"\n[BC] Training: {obs.shape[1]} obs -> {hidden_dims} -> {actions.shape[1]} actions")

    obs_t = torch.from_numpy(obs).float().to(device)
    act_t = torch.from_numpy(actions).float().to(device)
    dataset = TensorDataset(obs_t, act_t)

    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    policy = BCPolicy(obs.shape[1], actions.shape[1], hidden_dims).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        policy.train()
        for b_obs, b_act in train_loader:
            optimizer.zero_grad()
            loss = criterion(policy(b_obs), b_act)
            loss.backward()
            optimizer.step()

        policy.eval()
        with torch.no_grad():
            val_loss = sum(criterion(policy(b_obs), b_act).item()
                          for b_obs, b_act in val_loader) / len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in policy.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[BC] Epoch {epoch:3d}/{epochs} | Val Loss: {val_loss:.6f}")

    policy.load_state_dict(best_state)
    print(f"[BC] Best validation loss: {best_val_loss:.6f}")
    return policy


def save_data(obs: np.ndarray, actions: np.ndarray, metrics: dict, output_dir: str):
    """Save raw demonstration data and metrics."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "observations.npy"), obs)
    np.save(os.path.join(output_dir, "actions.npy"), actions)
    np.save(os.path.join(output_dir, "episode_rewards.npy"), np.array(metrics["episode_rewards"]))

    # Save metrics to JSON
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[BC] Saved data: obs={obs.shape}, actions={actions.shape}")


def save_policy(policy: BCPolicy, output_dir: str, obs_dim: int, action_dim: int, hidden_dims: list):
    """Save policy in both standalone and RSL-RL format."""
    os.makedirs(output_dir, exist_ok=True)

    # Standalone checkpoint
    torch.save({
        'model_state_dict': policy.state_dict(),
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'hidden_dims': hidden_dims,
    }, os.path.join(output_dir, "bc_policy.pt"))

    # RSL-RL format - must match ActorCritic state dict structure
    # ActorCritic expects: actor.X.weight, actor.X.bias, critic.X.weight, critic.X.bias, std
    rsl_rl_state_dict = {}

    # Map BC network weights to actor weights
    # BC: network.0.weight -> RSL-RL: actor.0.weight
    for name, param in policy.network.named_parameters():
        actor_key = f"actor.{name}"
        rsl_rl_state_dict[actor_key] = param.data.cpu().clone()

    # Initialize critic with same architecture (random weights, same shapes)
    # This is needed for RSL-RL to load without errors
    for name, param in policy.network.named_parameters():
        critic_key = f"critic.{name}"
        # Random initialization for critic
        rsl_rl_state_dict[critic_key] = torch.randn_like(param.data.cpu())

    # Initialize action noise std (small value for fine-tuning)
    rsl_rl_state_dict["std"] = torch.ones(action_dim) * 0.3

    torch.save({
        'model_state_dict': rsl_rl_state_dict,
        'optimizer_state_dict': {},
        'iter': 0,
        'infos': {'bc_pretrained': True, 'obs_dim': obs_dim, 'action_dim': action_dim, 'hidden_dims': hidden_dims},
    }, os.path.join(output_dir, "bc_policy_rsl_rl.pt"))

    print(f"[BC] Saved policy to {output_dir}/")
    print(f"     - bc_policy.pt")
    print(f"     - bc_policy_rsl_rl.pt (for RL fine-tuning)")
    print(f"     - observations.npy, actions.npy (raw data)")


def main():
    # Handle resume mode - use existing directory
    if args_cli.resume:
        output_dir = args_cli.resume
        if not os.path.exists(output_dir):
            print(f"[BC] ERROR: Resume directory does not exist: {output_dir}", flush=True)
            return
        print(f"[BC] Resuming from: {output_dir}", flush=True)
        resume_from = output_dir
    else:
        # Determine output directory - use logs/bc_policy by default
        # Use same approach as RSL-RL: relative to current working directory
        if args_cli.output_dir is None:
            base_output_dir = os.path.abspath(os.path.join("logs", "bc_policy"))
        else:
            base_output_dir = args_cli.output_dir

        # Create timestamped output directory to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, f"bc_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        # Also create a "latest" symlink for convenience
        latest_link = os.path.join(base_output_dir, "latest")
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        try:
            os.symlink(f"bc_{timestamp}", latest_link)
        except OSError:
            pass

        print(f"[BC] Output directory: {output_dir}", flush=True)
        print(f"[BC] (also linked as {latest_link})", flush=True)
        resume_from = None

    cfg = CraneDirectEnvCfgFull()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.use_hierarchical_rl = True
    cfg.enable_domain_randomization = args_cli.domain_randomization
    cfg.sim.physx.solver_type = 1
    cfg.sim.physx.enable_stabilization = True

    print(f"[BC] Full 4D environment (x, y, z, yaw)", flush=True)
    print(f"[BC] Domain randomization: {args_cli.domain_randomization}", flush=True)

    env = CraneDirectEnvFull(cfg)
    print("[BC] Environment created", flush=True)

    # Save run config (only if not resuming)
    if not args_cli.resume:
        run_config = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "num_envs": args_cli.num_envs,
            "num_episodes": args_cli.num_episodes,
            "hidden_dims": args_cli.hidden_dims,
            "batch_size": args_cli.batch_size,
            "learning_rate": args_cli.lr,
            "epochs": args_cli.epochs,
            "device": args_cli.device,
            "domain_randomization": args_cli.domain_randomization,
            "top_k": args_cli.top_k,
            "pos_noise_cm": args_cli.pos_noise,
            "yaw_noise_deg": args_cli.yaw_noise,
            "save_interval": args_cli.save_interval,
        }
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(run_config, f, indent=2)

    # Collect demonstrations with incremental saving
    obs, actions, metrics = collect_demonstrations(
        env, args_cli.num_episodes,
        output_dir=output_dir,
        save_interval=args_cli.save_interval,
        top_k=args_cli.top_k,
        pos_noise_cm=args_cli.pos_noise,
        yaw_noise_deg=args_cli.yaw_noise,
        resume_from=resume_from
    )

    # Save final data and metrics
    save_data(obs, actions, metrics, output_dir)

    env.close()

    # Skip training if collect_only mode
    if args_cli.collect_only:
        print(f"\n[BC] ====== COLLECTION COMPLETE ======", flush=True)
        print(f"[BC] Data saved to: {output_dir}", flush=True)
        print(f"[BC] To train later, run:", flush=True)
        print(f"     python train_bc.py --data_dir {output_dir}", flush=True)
        return

    # Train policy
    policy = train_bc_policy(
        obs, actions,
        hidden_dims=args_cli.hidden_dims,
        batch_size=args_cli.batch_size,
        lr=args_cli.lr,
        epochs=args_cli.epochs,
        device=args_cli.device
    )

    save_policy(policy, output_dir, obs.shape[1], actions.shape[1], args_cli.hidden_dims)

    print(f"\n[BC] ====== RUN COMPLETE ======", flush=True)
    print(f"[BC] All outputs saved to: {output_dir}", flush=True)
    print(f"[BC] Files:", flush=True)
    print(f"     - config.json (run configuration)", flush=True)
    print(f"     - metrics.json (collection & training metrics)", flush=True)
    print(f"     - observations.npy, actions.npy (raw data)", flush=True)
    print(f"     - bc_policy.pt (standalone checkpoint)", flush=True)
    print(f"     - bc_policy_rsl_rl.pt (for RL fine-tuning)", flush=True)
    print(f"[BC]", flush=True)
    print(f"[BC] To fine-tune with RL:", flush=True)
    print(f"     ./isaaclab.sh -p train_rl.py --task Isaac-Crane-Full-v0 \\", flush=True)
    print(f"         --load_checkpoint {output_dir}/bc_policy_rsl_rl.pt", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
