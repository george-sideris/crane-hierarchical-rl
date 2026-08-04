#!/usr/bin/env python3
"""Plot RL training reward curves comparing PPO runs with SAC.

Both PPO (RSL-RL) and SAC (SB3) track completed-episode returns via a
rolling deque(maxlen=100). We use:
  - PPO: Train/mean_reward (RSL-RL's rewbuffer)
  - SAC: rollout/ep_rew_mean (SB3's ep_info_buffer)

X-axis is unified to total environment steps (grasp cycles across all envs):
  - PPO: iteration * num_steps_per_env * num_envs  (read from params/)
  - SAC: already in total timesteps (SB3 increments by num_envs per step)

Output: reward_curves_with_sac.{pdf,png}
"""

import os
import re
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Paths ─────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

PPO_RUNS = {
    "PPO (Pose)": {
        "logdir": os.path.join(REPO_ROOT, "crane_testbed/results/RL_Pose/2026-02-23_20-19-49"),
        "tag": "Train/mean_reward",
        "color": "#1f77b4",
        "linestyle": "-",
    },
    "PPO (Seg PCD)": {
        "logdir": os.path.join(REPO_ROOT, "crane_testbed/results/RL_Seg_PCD/2026-02-22_11-02-21"),
        "tag": "Train/mean_reward",
        "color": "#ff7f0e",
        "linestyle": "-",
    },
    "PPO (Raw PCD)": {
        "logdir": os.path.join(REPO_ROOT, "crane_testbed/results/RL_Raw_PCD/2026-02-26_23-18-20"),
        "tag": "Train/mean_reward",
        "color": "#2ca02c",
        "linestyle": "-",
    },
    r"BC $\rightarrow$ RL (PPO)": {
        "logdir": os.path.join(REPO_ROOT, "crane_testbed/results/BCRL_Raw_PCD_Sigma_0.05/2026-02-27_04-21-30"),
        "tag": "Train/mean_reward",
        "color": "#d62728",
        "linestyle": "-",
    },
}

SAC_RUN = {
    "label": "SAC (Pose)",
    "logdir": os.path.join(REPO_ROOT, "logs/sac/crane_full_cossin_mr_v0/2026-03-23_05-36-45/SAC_1"),
    "tag": "rollout/ep_rew_mean",
    "color": "#9467bd",
    "linestyle": "--",
}

SMOOTH_WINDOW = 20
SAC_SMOOTH_WINDOW = 5
X_MAX_TRANSITIONS = 50_000


# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "text.usetex": False,
    "mathtext.fontset": "cm",
})


# ── Config loading ────────────────────────────────────────────────────
def _grep_yaml_int(path, key):
    """Extract an integer value from a YAML file by regex (avoids safe_load issues with Python types)."""
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*:\s*(\d+)", re.MULTILINE)
    with open(path) as f:
        m = pattern.search(f.read())
    if not m:
        raise ValueError(f"{key} not found in {path}")
    return int(m.group(1))


def load_ppo_x_multiplier(logdir):
    """Read num_steps_per_env and num_envs from a PPO run's saved params.

    Old PPO runs logged at iteration number (multiplier = num_steps_per_env * num_envs).
    New PPO runs using OnPolicyRunnerEnvSteps log at total env steps (multiplier = 1).
    We detect this by checking if Train/mean_reward x-values exceed max_iterations.
    """
    params_dir = os.path.join(logdir, "params")
    num_steps = _grep_yaml_int(os.path.join(params_dir, "agent.yaml"), "num_steps_per_env")
    num_envs = _grep_yaml_int(os.path.join(params_dir, "env.yaml"), "num_envs")
    multiplier = num_steps * num_envs

    # Auto-detect: if x-values are already large (> max_iterations), run used env-step logging
    ea = EventAccumulator(logdir)
    ea.Reload()
    events = ea.Scalars("Train/mean_reward")
    if events:
        max_step = max(e.step for e in events)
        max_iter = _grep_yaml_int(os.path.join(params_dir, "agent.yaml"), "max_iterations")
        if max_step > max_iter:
            # Already logged at total env steps
            print(f"  {os.path.basename(logdir)}: already in env steps (max_step={max_step})")
            return 1

    print(f"  {os.path.basename(logdir)}: num_steps_per_env={num_steps}, num_envs={num_envs}, multiplier={multiplier}")
    return multiplier


def load_sac_x_multiplier(logdir):
    """SAC x-axis is already total timesteps (SB3 increments by num_envs per step)."""
    return 1


# ── Data loading ──────────────────────────────────────────────────────
def load_scalar(logdir, tag):
    ea = EventAccumulator(logdir)
    ea.Reload()
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    # Deduplicate: keep last value per step
    _, idx = np.unique(steps[::-1], return_index=True)
    idx = len(steps) - 1 - idx
    return steps[idx], values[idx]


def smooth(values, window):
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# ── Plot ──────────────────────────────────────────────────────────────
def main():
    fig, ax = plt.subplots(figsize=(3.5, 2.4))  # single-column IEEE width

    # Plot PPO runs
    for label, cfg in PPO_RUNS.items():
        x_mult = load_ppo_x_multiplier(cfg["logdir"])
        steps, values = load_scalar(cfg["logdir"], cfg["tag"])
        steps = steps * x_mult
        mask = steps <= X_MAX_TRANSITIONS
        steps, values = steps[mask], values[mask]
        smoothed = smooth(values, SMOOTH_WINDOW)
        ax.plot(steps, values, color=cfg["color"], alpha=0.15, linewidth=0.3)
        ax.plot(steps, smoothed, color=cfg["color"], linewidth=0.8,
                linestyle=cfg["linestyle"], label=label)

    # Plot SAC
    sac = SAC_RUN
    x_mult = load_sac_x_multiplier(sac["logdir"])
    steps, values = load_scalar(sac["logdir"], sac["tag"])
    steps = steps * x_mult
    mask = steps <= X_MAX_TRANSITIONS
    steps, values = steps[mask], values[mask]
    smoothed = smooth(values, SAC_SMOOTH_WINDOW)
    ax.plot(steps, values, color=sac["color"], alpha=0.15, linewidth=0.3)
    ax.plot(steps, smoothed, color=sac["color"], linewidth=0.8,
            linestyle=sac["linestyle"], label=sac["label"])

    # Style (matching paper figure)
    ax.set_xlabel("Environment Steps (all envs)")
    ax.set_ylabel("Mean Episode Return")
    ax.set_xlim(0, X_MAX_TRANSITIONS)
    ax.grid(True, alpha=0.25, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="none",
              fontsize=6, handlelength=1.2, handletextpad=0.4,
              borderpad=0.3, labelspacing=0.25)

    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x > 0 else "0"))

    out = os.path.join(OUT_DIR, "reward_curves_with_sac")
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png")
    print(f"Saved {out}.{{pdf,png}}")
    plt.close(fig)


if __name__ == "__main__":
    print("Loading TensorBoard data...")
    main()
    print("Done.")
