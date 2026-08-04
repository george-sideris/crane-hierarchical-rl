#!/usr/bin/env python3
"""Does a support gate change the DEPLOYED heuristic's targets on the real failure clouds?

Replays the exact deployed stack over the raw clouds extracted from the 3 baseline-trial bags:
process_cloud (crop @ the run's recorded rack_y_shift, 3 pole boxes, FPS 1024, ROR) -> shim ->
HeuristicPolicy(dig 0.30). Then repeats with a support gate applied to the cloud (drop points
whose 0.5 m neighbourhood support is below frac * cloud max) and reports whether the target moved.

Context: the deployed heuristic ALREADY walks down to the highest SUPPORTED candidate (>=3
neighbours in 0.15 m) and the pipeline ALREADY runs ROR - and the trials still logged 16
structure/noise-target failures. Hypothesis under test: the gate fixes only sparse ghosts (air /
pole remnants), not dense structure (rail, end-board), because structure IS well-supported.

    python3 scripts/envs/heuristic_gate_check.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "fpi_crane_ros2/fpi_crane_rl/fpi_crane_rl")
from pointcloud_pipeline import process_cloud          # noqa: E402
from policy_loader import HeuristicPolicy              # noqa: E402

B_MIN = np.array([-5.364, -1.684, -1.30], dtype=np.float32)
B_MAX = np.array([-3.364, 5.316, 0.10], dtype=np.float32)
# node params as the trials ran them (rack_y_shift from each run's npz; these are the declares)
POLE = dict(pole_x_gt=-3.6, pole_y_lt=-1.35, pole2_x_gt=-3.9, pole2_y_gt=4.6,
            pole3_x_lt=-5.15, pole3_y_lt=-1.45)
ROR = dict(ror_radius=0.2, ror_min_neighbors=2)
RUNS = {"double": "run_20260803_155559", "jagged": "run_20260803_174017",
        "single": "run_20260803_204034"}
# ODS-verified failure cycles (1-based) with their notes
FAIL = {("double", 13): "noise: targeted rack", ("double", 17): "noise: targeted rack",
        ("jagged", 13): "noise: targeted pole", ("jagged", 18): "noise: targeted air",
        ("jagged", 20): "noise: rack", ("jagged", 21): "noise: air (poles in pcd)",
        ("single", 15): "grabbed rack", ("single", 16): "noise: rack",
        ("single", 17): "noise: rack", ("single", 18): "noise: rack"}


def gate_cloud(flat: torch.Tensor, frac: float = 0.25, radius: float = 0.5) -> torch.Tensor:
    """Zero out points whose xy support is below frac * (cloud max)."""
    p = flat.reshape(-1, 3).clone()
    valid = p.abs().sum(1) > 1e-6
    q = p[valid]
    cnt = (torch.cdist(q[:, :2], q[:, :2]) < radius).sum(1).float()
    drop = cnt < frac * cnt.max()
    idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    p[idx[drop]] = 0.0
    return p.reshape(-1)


def main():
    heur = HeuristicPolicy(B_MIN, B_MAX, cossin=True, dig=0.30)
    print(f"{'cloud':16s} {'shift':>6} | {'recorded y,z':>14} | {'replay y,z':>14} {'fid':>5} | "
          f"{'gated y,z':>14} {'moved':>6} | note")
    moved_fail, moved_ok, n_fail, n_ok = 0, 0, 0, 0
    for tag, run in RUNS.items():
        dec = {d["cycle"]: d for l in open(f"logs/crane_policy_debug/{run}/decisions.jsonl")
               if (d := json.loads(l))["kind"] == "gaze"}
        shift = float(np.load(f"logs/crane_policy_debug/{run}/policy_debug_001.npz")["rack_y_shift"])
        for f in sorted(glob.glob(f"logs/real_raw_clouds/{tag}_c*.npz")):
            cyc = int(os.path.basename(f)[len(tag) + 2:-4])
            if cyc not in dec:
                continue
            raw = np.load(f)["points"].astype(np.float32)
            crop_min = B_MIN.copy(); crop_min[1] += shift
            crop_max = B_MAX.copy(); crop_max[1] += shift
            pol = dict(POLE); pol["pole_y_lt"] += shift; pol["pole2_y_gt"] += shift
            pol["pole3_y_lt"] += shift
            flat = process_cloud(raw, np.zeros(3, np.float32),
                                 np.array([1, 0, 0, 0], np.float32),
                                 crop_min, crop_max, num_points=1024, **pol, **ROR)
            obs = flat.reshape(-1, 3).clone()
            obs[:, 1] -= shift * (obs.abs().sum(1) > 1e-6).float()   # shim (pads untouched)

            x0, y0, z0, _ = heur.get_target(obs.reshape(-1))
            xg, yg, zg, _ = heur.get_target(gate_cloud(obs.reshape(-1)))

            rec = dec[cyc]["target"]
            ry, rz = rec[1] - shift, rec[2]                          # to training coords
            fid = np.hypot(y0 - ry, z0 - rz)
            mv = np.hypot(yg - y0, zg - z0)
            note = FAIL.get((tag, cyc), "")
            is_fail = (tag, cyc) in FAIL
            if is_fail:
                n_fail += 1; moved_fail += (mv > 0.3)
            else:
                n_ok += 1; moved_ok += (mv > 0.3)
            print(f"{tag+'_c%02d' % cyc:16s} {shift:6.2f} | ({ry:5.2f},{rz:5.2f}) | "
                  f"({y0:5.2f},{z0:5.2f}) {fid:5.2f} | ({yg:5.2f},{zg:5.2f}) {mv:6.2f} | "
                  f"{'FAIL: ' + note if note else 'ok cycle'}")
    print(f"\ngate moved the target >0.3 m on {moved_fail}/{n_fail} covered FAILURE cycles "
          f"and {moved_ok}/{n_ok} ok cycles")
    print("(fid = |replay - recorded| in the y,z plane; large fid means the replay is not "
          "faithful for that cloud and its row should be read with caution)")


if __name__ == "__main__":
    main()
