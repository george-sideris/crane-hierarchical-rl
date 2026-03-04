#!/usr/bin/env python3
"""Regenerate paper-compact pipeline and progression figures from saved episode data.

This script is a thin wrapper around the paper-specific viz functions in
eval_viz.py.  It can be used in two modes:

  1. **From pickle** (offline, no sim needed):
     If a previous eval run saved episode data to a .pkl file, pass it here:

       python gen_paper_figures.py --data episode_001_data.pkl

  2. **Live** (requires sim):
     Re-run the eval with --paper_viz; compact figures are generated alongside
     the originals:

       # BC policy
       ./isaaclab.sh -p crane_testbed/scripts/envs/play_bc_pointcloud.py \\
           --checkpoint <ckpt> --num_envs 1 --num_episodes 1 \\
           --paper_viz --raw_pcd --headless

       # RL policy
       ./isaaclab.sh -p crane_testbed/scripts/rsl_rl/play.py \\
           --task <task> --num_envs 1 --paper_viz --headless

     The compact PNGs (episode_*_pipeline_paper.png,
     episode_*_progression_paper.png) appear alongside the originals in the
     paper_viz/ output directory.  Copy them into crane_testbed/docs/ and
     update the \\includegraphics paths in iros2026_draft_new.tex.

After regeneration, the expected filenames are:
  - episode_001_pipeline_paper.png   (2 rows x 4 cols)
  - episode_001_progression_paper.png (2 rows x 5 cols)
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "envs"))


def main():
    parser = argparse.ArgumentParser(description="Regenerate paper figures from saved episode data")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to pickled episode data (.pkl)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as data file)")
    parser.add_argument("--episode_idx", type=int, default=1,
                        help="Episode index for filename (default: 1)")
    args = parser.parse_args()

    from eval_viz import save_paper_pipeline_viz, save_paper_progression_viz

    with open(args.data, "rb") as f:
        episode_data = pickle.load(f)

    if isinstance(episode_data, dict):
        trimmed = episode_data.get("grasps", episode_data.get("episode_data", []))
        bounds_min = episode_data.get("bounds_min")
        bounds_max = episode_data.get("bounds_max")
    else:
        trimmed = episode_data
        bounds_min = trimmed[0].get("bounds_min") if trimmed else None
        bounds_max = trimmed[0].get("bounds_max") if trimmed else None

    out_dir = args.output_dir or os.path.dirname(args.data)
    os.makedirs(out_dir, exist_ok=True)

    n = len(trimmed)
    print(f"Loaded {n} grasps from {args.data}")

    # Pipeline: first + last grasp
    paper_rep = [trimmed[0]]
    if n > 1:
        paper_rep.append(trimmed[-1])
    save_paper_pipeline_viz(paper_rep, args.episode_idx, out_dir,
                            bounds_min=bounds_min, bounds_max=bounds_max)

    # Progression: 2 rows x 5 cols
    save_paper_progression_viz(trimmed, args.episode_idx, out_dir,
                               bounds_min=bounds_min, bounds_max=bounds_max)

    print(f"Done. Output in {out_dir}")


if __name__ == "__main__":
    main()
