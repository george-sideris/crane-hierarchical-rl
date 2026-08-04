#!/usr/bin/env python3
"""
Timing harness for the heuristic trailer-unloading baseline.

Reports, for a batch of parallel environments:
  - simulated (physical) time to unload a trailer  = internal physics steps / 120 Hz
  - wall-clock compute time per trailer and per batch

Usage (inside container):
    /workspace/isaaclab/isaaclab.sh -p \
        /workspace/isaaclab/crane_testbed/scripts/envs/time_heuristic.py \
        --num_envs 20 --num_episodes 20 --seed 42 --headless
"""

import sys
import time
import argparse
import torch
from pathlib import Path

crane_scripts_path = Path(__file__).resolve().parent
sys.path.insert(0, str(crane_scripts_path))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Time heuristic baseline")
parser.add_argument("--num_envs", type=int, default=20)
parser.add_argument("--num_episodes", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--domain_randomization", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import crane_rl_env_full
from crane_rl_env_full import CraneDirectEnvFull, CraneDirectEnvCfgFull


def main():
    cfg = CraneDirectEnvCfgFull()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.use_hierarchical_rl = False
    cfg.enable_domain_randomization = args_cli.domain_randomization
    cfg.seed = args_cli.seed
    cfg.sim.physx.solver_type = 1
    cfg.sim.physx.enable_stabilization = True

    env = CraneDirectEnvFull(cfg)
    physics_hz = 1.0 / env.physics_dt
    print(f"[Time] {env.num_envs} envs | physics dt={env.physics_dt:.5f}s ({physics_hz:.1f} Hz)")

    # Count internal physics sub-steps by wrapping sim.step
    phys_steps = {"n": 0}
    _orig_sim_step = env.sim.step
    def _counting_step(render=True):
        phys_steps["n"] += 1
        return _orig_sim_step(render=render)
    env.sim.step = _counting_step

    env.reset()  # includes log-settling (excluded from timing below)

    starting_logs = torch.zeros(env.num_envs, dtype=torch.int32)
    if hasattr(env, "_per_env_log_counts") and env._per_env_log_counts is not None:
        for i in range(env.num_envs):
            starting_logs[i] = int(env._per_env_log_counts[i].item())
    else:
        default_logs = int(env._per_env_target) if getattr(env, "_per_env_target", 0) > 0 else 200
        starting_logs[:] = default_logs
    print(f"[Time] Logs/trailer: {starting_logs.float().mean():.0f} "
          f"(min {int(starting_logs.min())}, max {int(starting_logs.max())})")

    episode_logs_cleared = torch.zeros(env.num_envs, dtype=torch.int32)
    episodes_done = 0
    total_cycles = 0

    # Per-trailer accounting
    trailer_sim_seconds = []   # simulated physical seconds to clear each trailer
    trailer_cycles = []
    ep_phys_at_start = [phys_steps["n"]] * env.num_envs
    ep_sim_steps = [0] * env.num_envs

    phys_steps["n"] = 0
    for i in range(env.num_envs):
        ep_phys_at_start[i] = 0

    wall_start = time.perf_counter()
    phys_before_loop = phys_steps["n"]

    while episodes_done < args_cli.num_episodes and simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
            phys_before = phys_steps["n"]
            _, _, terminated, truncated, _ = env.step(actions)
            phys_this_cycle = phys_steps["n"] - phys_before  # shared across all envs in the batch
            total_cycles += 1

            for i in range(env.num_envs):
                episode_logs_cleared[i] += int(env._prev_logs_grasped[i].item())
                ep_sim_steps[i] += phys_this_cycle

            done = terminated | truncated
            for i in range(env.num_envs):
                if done[i]:
                    episodes_done += 1
                    sim_secs = ep_sim_steps[i] / physics_hz
                    cyc = ep_sim_steps[i]  # not used
                    trailer_sim_seconds.append(sim_secs)
                    ep_sim_steps[i] = 0
                    episode_logs_cleared[i] = 0
                    if episodes_done >= args_cli.num_episodes:
                        break

    wall_elapsed = time.perf_counter() - wall_start
    phys_total = phys_steps["n"] - phys_before_loop
    sim_total_seconds = phys_total / physics_hz

    def _mean(v):
        return sum(v) / len(v) if v else 0.0

    def _std(v, m):
        return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0

    mean_sim = _mean(trailer_sim_seconds)
    std_sim = _std(trailer_sim_seconds, mean_sim)

    print("\n" + "=" * 64)
    print(f"[Time] ===== TIMING RESULTS =====")
    print(f"[Time] Envs (parallel trailers): {env.num_envs}")
    print(f"[Time] Trailers fully processed:  {episodes_done}")
    print(f"[Time] Total grasp cycles (batch loop iters): {total_cycles}")
    print(f"[Time] Total internal physics steps: {phys_total}")
    print("-" * 64)
    print(f"[Time] REAL (simulated/physical) time per trailer:")
    print(f"[Time]   {mean_sim:.1f} +/- {std_sim:.1f} s  ({mean_sim/60:.2f} min)")
    print(f"[Time]   (= physics steps for that trailer / {physics_hz:.0f} Hz)")
    print("-" * 64)
    print(f"[Time] COMPUTE (wall-clock) time:")
    print(f"[Time]   Total wall-clock for batch loop: {wall_elapsed:.1f} s ({wall_elapsed/60:.2f} min)")
    print(f"[Time]   Wall-clock per trailer (amortized over {env.num_envs} parallel): "
          f"{wall_elapsed/max(1,episodes_done):.2f} s")
    print(f"[Time]   Real-time factor (sim/compute): {sim_total_seconds/max(1e-9,wall_elapsed):.2f}x")
    print(f"[Time]   (>1 = faster than real time, summed across {env.num_envs} envs)")
    print("=" * 64)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
