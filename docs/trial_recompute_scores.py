"""Recompute per-point scores for a deployed run and verify they reproduce the logged target.

The saved npz holds the cloud in real (rack-shifted) coordinates. The node feeds the
network training coordinates (subtract rack_y_shift) and adds the shift back to the
decoded command, so we do the same here.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/george/IsaacLab/crane_testbed/scripts/envs")
from scoring_head import ScoringGraspPolicy  # noqa: E402

RUN = ("/home/george/IsaacLab/crane_testbed/logs/bc_pointcloud/"
       "scoring_margin05_2048_c/policy_debug/run_20260805_194008/")
CKPT = ("/home/george/IsaacLab/crane_testbed/logs/bc_pointcloud/"
        "scoring_margin05_2048_c/scoring_policy.pt")

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
npts = int(ck["num_points"])
model = ScoringGraspPolicy(num_points=npts)
model.load_state_dict(ck["model_state_dict"])
model.eval()


def scores_for(cycle):
    """Return (points_real, scores, target_logged, target_recomputed)."""
    d = np.load(RUN + f"policy_debug_{cycle:03d}.npz")
    p_real = d["points"].astype(np.float32)
    shift = float(d["rack_y_shift"])
    p_train = p_real.copy()
    p_train[:, 1] -= shift

    buf = np.zeros((npts, 3), dtype=np.float32)
    n = min(len(p_train), npts)
    buf[:n] = p_train[:n]
    t = torch.from_numpy(buf).unsqueeze(0)

    with torch.no_grad():
        score, dz, yaw, pts = model(t)
        act = model.act(t)[0].numpy()

    act_real = act.copy()
    act_real[1] += shift
    return p_real[:n], score[0, :n].numpy(), d["target"], act_real


if __name__ == "__main__":
    cycles = [int(a) for a in sys.argv[1:]] or list(range(1, 33))
    logged = {}
    for line in open(RUN + "decisions.jsonl"):
        r = json.loads(line)
        logged[r["cycle"]] = r

    worst = 0.0
    for c in cycles:
        try:
            p, s, tgt, rec = scores_for(c)
        except FileNotFoundError:
            continue
        err = float(np.linalg.norm(rec[:3] - np.asarray(tgt)[:3]))
        worst = max(worst, err)
        print(f"cycle {c:02d}  n={len(p):4d}  logged=({tgt[0]:.3f},{tgt[1]:.3f},{tgt[2]:.3f})"
              f"  recomputed=({rec[0]:.3f},{rec[1]:.3f},{rec[2]:.3f})  err={err*1000:.1f} mm"
              f"  score[min,max]=({s.min():.2f},{s.max():.2f})")
    print(f"\nworst position error over {len(cycles)} cycles: {worst*1000:.1f} mm")
