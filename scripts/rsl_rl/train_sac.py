"""Script to train SAC agent with Stable-Baselines3 on crane Pose task.

Usage (inside Docker container):
    # One-time install:
    pip install stable-baselines3

    # Train:
    ./isaaclab.sh -p crane_testbed/scripts/rsl_rl/train_sac.py \
        --task Isaac-Crane-Full-CosSin-MR-v0 \
        --num_envs 64 --seed 42 --max_iterations 500 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import signal
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train SAC agent with Stable-Baselines3.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Crane-Full-CosSin-MR-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=10000, help="Training iterations (each = num_envs env steps).")
parser.add_argument("--run_name", type=str, default=None, help="Run name suffix for log directory.")
# SAC hyperparameters
parser.add_argument("--learning_rate", type=float, default=3e-4, help="SAC learning rate.")
parser.add_argument("--batch_size", type=int, default=256, help="SAC minibatch size.")
parser.add_argument("--buffer_size", type=int, default=100_000, help="Replay buffer size.")
parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient.")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
parser.add_argument("--ent_coef", type=str, default="auto", help="Entropy coefficient ('auto' for learned).")
parser.add_argument("--train_freq", type=int, default=1, help="Update policy every N steps.")
parser.add_argument("--gradient_steps", type=int, default=1, help="Gradient steps per update.")
parser.add_argument("--learning_starts", type=int, default=1000, help="Steps before learning starts.")
parser.add_argument("--net_arch", type=str, default="256,128,64", help="Hidden layer sizes (comma-separated).")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def cleanup_pbar(*args):
    import gc
    tqdm_objects = [obj for obj in gc.get_objects() if "tqdm" in type(obj).__name__]
    for tqdm_object in tqdm_objects:
        if "tqdm_rich" in type(tqdm_object).__name__:
            tqdm_object.close()
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, cleanup_pbar)


"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg, DirectMARLEnv, DirectMARLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

import isaaclab_tasks  # noqa: F401
import crane_testbed.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class EnvMetricsCallback(BaseCallback):
    """Log the env's Episode/* metrics to TensorBoard, matching RSL-RL's output.

    RSL-RL logs extras["episode"] dict every iteration. SB3 ignores it, so this
    callback reads _get_extras() from the unwrapped env and logs the same keys.
    """

    def _on_step(self) -> bool:
        # Access the unwrapped IsaacLab env through the SB3 wrapper.
        # For PointCloud tasks, the base env with _get_extras is on ._base_env.
        unwrapped = self.training_env.unwrapped
        base = getattr(unwrapped, "_base_env", unwrapped)
        extras = base._get_extras() if hasattr(base, "_get_extras") else {}
        ep_dict = extras.get("episode", {})
        for key, value in ep_dict.items():
            if "/" in key:
                self.logger.record(key, value)
            else:
                self.logger.record(f"Episode/{key}", value)
        return True


def main():
    """Train SAC agent."""
    # Load env config from registry
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # Log directory
    task_folder = args_cli.task.lower().replace("isaac-", "").replace("-", "_")
    log_root = os.path.abspath(os.path.join("logs", "sac", task_folder))
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args_cli.run_name:
        run_name += f"_{args_cli.run_name}"
    log_dir = os.path.join(log_root, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging experiment in directory: {log_dir}")

    # Create environment (PointCloud tasks need direct instantiation)
    if "PointCloud" in args_cli.task:
        import sys
        from pathlib import Path
        envs_dir = Path(__file__).parent.parent / "envs"
        if str(envs_dir) not in sys.path:
            sys.path.insert(0, str(envs_dir))
        from crane_pointcloud_direct_env import CranePointCloudDirectEnv
        env = CranePointCloudDirectEnv(env_cfg)
        env.log_dir = log_dir
    else:
        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

    # Wrap for SB3
    env = Sb3VecEnvWrapper(env)

    # Parse net_arch
    net_arch = [int(x) for x in args_cli.net_arch.split(",")]

    # Parse ent_coef
    ent_coef = args_cli.ent_coef
    if ent_coef != "auto":
        ent_coef = float(ent_coef)

    # Compute total timesteps from max_iterations
    # For SAC, one "iteration" = num_envs steps in the environment
    total_timesteps = args_cli.max_iterations * args_cli.num_envs

    # Detect point cloud tasks and configure PointNet features extractor
    is_pointcloud = "PointCloud" in args_cli.task
    policy_kwargs = {"net_arch": net_arch}
    if is_pointcloud:
        from crane_testbed.agents.pointnet_extractor_sb3 import PointNetFeaturesExtractor
        num_points = getattr(env_cfg, "num_points", 1024)
        policy_kwargs["features_extractor_class"] = PointNetFeaturesExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            "num_points": num_points,
            "output_dim": 256,
            "norm_type": "layernorm",
        }
        print(f"[INFO] PointCloud task detected — using PointNet extractor ({num_points} points, 256D output)")

    # Create SAC agent
    agent = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=args_cli.learning_rate,
        buffer_size=args_cli.buffer_size,
        batch_size=args_cli.batch_size,
        tau=args_cli.tau,
        gamma=args_cli.gamma,
        ent_coef=ent_coef,
        train_freq=args_cli.train_freq,
        gradient_steps=args_cli.gradient_steps,
        learning_starts=args_cli.learning_starts,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args_cli.seed,
        device="auto",
        tensorboard_log=log_dir,
    )

    print(f"[INFO] SAC agent created. Total timesteps: {total_timesteps}")
    print(f"[INFO] Policy architecture: {net_arch}")
    print(f"[INFO] Buffer size: {args_cli.buffer_size}, Batch size: {args_cli.batch_size}")
    print(f"[INFO] LR: {args_cli.learning_rate}, Gamma: {args_cli.gamma}, Tau: {args_cli.tau}")
    print(f"[INFO] Entropy coef: {ent_coef}")

    # Save config
    import yaml
    config = {
        "task": args_cli.task,
        "algorithm": "SAC",
        "num_envs": args_cli.num_envs,
        "seed": args_cli.seed,
        "total_timesteps": total_timesteps,
        "learning_rate": args_cli.learning_rate,
        "buffer_size": args_cli.buffer_size,
        "batch_size": args_cli.batch_size,
        "tau": args_cli.tau,
        "gamma": args_cli.gamma,
        "ent_coef": str(ent_coef),
        "train_freq": args_cli.train_freq,
        "gradient_steps": args_cli.gradient_steps,
        "learning_starts": args_cli.learning_starts,
        "net_arch": net_arch,
    }
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    with open(os.path.join(log_dir, "params", "agent.yaml"), "w") as f:
        yaml.dump(config, f)

    # Callbacks — save every 10 "iterations" worth of env steps to match PPO's save_interval=10.
    # PPO iteration = num_steps_per_env * num_envs env steps.  We use a comparable interval.
    # SB3 save_freq is in total env steps (num_timesteps units).
    save_freq_steps = 10 * args_cli.num_envs  # ~200 for 20 envs, ~320 for 32 envs
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_steps,
        save_path=log_dir,
        name_prefix="model",
        verbose=1,
    )

    # Train
    with contextlib.suppress(KeyboardInterrupt):
        agent.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, EnvMetricsCallback()],
            progress_bar=True,
            log_interval=1,
        )

    # Save final model
    final_path = os.path.join(log_dir, "model_final")
    agent.save(final_path)
    print(f"[INFO] Final model saved to: {final_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
