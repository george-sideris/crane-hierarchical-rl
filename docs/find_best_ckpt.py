#!/usr/bin/env python3
"""Find the best checkpoint <= 800 iterations for each RL run.

Reads TensorBoard event files, computes smoothed (window=20) episode return,
and reports the peak iteration and corresponding model file for each run.
"""

import os
import glob
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

RUNS = {
    "RL (Pose)": {
        "logdir": os.path.join(REPO_ROOT, "crane_testbed/results/RL_Pose/2026-02-23_20-19-49"),
        "current_ckpt": 940,
    },
    "RL (Seg PCD)": {
        "logdir": os.path.join(REPO_ROOT, "crane_testbed/results/RL_Seg_PCD/2026-02-22_11-02-21"),
        "current_ckpt": 880,
    },
    "RL (Raw PCD)": {
        "logdir": os.path.join(REPO_ROOT, "crane_testbed/results/RL_Raw_PCD/2026-02-26_23-18-20"),
        "current_ckpt": 650,
    },
}

TAG = "Episode/episode_return"
SMOOTH_WINDOW = 20
MAX_ITER = 800


def load_scalar(logdir, tag):
    ea = EventAccumulator(logdir)
    ea.Reload()
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def smooth(values, window):
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def find_available_checkpoints(logdir):
    """Return sorted list of checkpoint iterations available on disk."""
    pattern = os.path.join(logdir, "model_*.pt")
    files = glob.glob(pattern)
    iters = []
    for f in files:
        basename = os.path.basename(f)
        num = basename.replace("model_", "").replace(".pt", "")
        try:
            iters.append(int(num))
        except ValueError:
            continue
    return sorted(iters)


def main():
    print(f"Finding best checkpoints with iter <= {MAX_ITER}")
    print(f"Smoothing window: {SMOOTH_WINDOW}")
    print("=" * 70)

    for name, cfg in RUNS.items():
        logdir = cfg["logdir"]
        current = cfg["current_ckpt"]

        steps, values = load_scalar(logdir, TAG)

        # Filter to <= MAX_ITER
        mask = steps <= MAX_ITER
        steps_capped = steps[mask]
        values_capped = values[mask]
        smoothed = smooth(values_capped, SMOOTH_WINDOW)

        # Find peak smoothed return
        best_idx = np.argmax(smoothed)
        best_step = int(steps_capped[best_idx])
        best_value = smoothed[best_idx]

        # Find nearest available checkpoint
        available = find_available_checkpoints(logdir)
        available_capped = [i for i in available if i <= MAX_ITER]

        # Find checkpoint closest to best_step
        if available_capped:
            nearest_ckpt = min(available_capped, key=lambda x: abs(x - best_step))
        else:
            nearest_ckpt = None

        # Also find the checkpoint with the highest smoothed return
        # (might differ from the one closest to the peak step)
        best_ckpt_value = -np.inf
        best_ckpt_iter = None
        for ckpt_iter in available_capped:
            idx = np.argmin(np.abs(steps_capped - ckpt_iter))
            val = smoothed[idx]
            if val > best_ckpt_value:
                best_ckpt_value = val
                best_ckpt_iter = ckpt_iter

        changed = "CHANGED" if best_ckpt_iter != current else "unchanged"

        print(f"\n{name}")
        print(f"  Current checkpoint:       model_{current}.pt")
        print(f"  Peak smoothed return:     {best_value:.2f} at iter {best_step}")
        print(f"  Nearest available ckpt:   model_{nearest_ckpt}.pt")
        print(f"  Best available ckpt:      model_{best_ckpt_iter}.pt "
              f"(smoothed return {best_ckpt_value:.2f})")
        print(f"  Status: {changed}")

        # Show top-5 checkpoints by smoothed return for reference
        ckpt_scores = []
        for ckpt_iter in available_capped:
            idx = np.argmin(np.abs(steps_capped - ckpt_iter))
            ckpt_scores.append((ckpt_iter, smoothed[idx]))
        ckpt_scores.sort(key=lambda x: x[1], reverse=True)
        print(f"  Top-5 checkpoints (by smoothed return):")
        for rank, (it, val) in enumerate(ckpt_scores[:5], 1):
            print(f"    {rank}. model_{it}.pt  ->  {val:.2f}")

    print("\n" + "=" * 70)
    print("Done. Use these checkpoints for eval.")


if __name__ == "__main__":
    main()
