#!/usr/bin/env python3
"""How stable is each policy's grasp target under ZED sensor noise, on REAL clouds?

Motivation: on the real rack, all three policies sometimes pick the correct log and sometimes
the broken pole or the rack rail, on what is nominally the same scene. If that flip is driven
by sensor noise, then "which target it picks" is not a property of the policy alone and a
single-shot comparison is not evidence of anything. This measures it directly: hold the cloud
fixed, resample the noise K times, and look at where the target lands.

Three numbers per policy, all on the same clouds and the same noise draws:
  spread     mean distance of the K targets from their centroid [m]. Low = the decision is a
             property of the scene, not of the noise.
  modes      number of distinct target clusters (single-link, MODE_EPS). >1 means the policy is
             bistable on this cloud: it is flipping between candidate grasps.
  pole%      fraction of draws landing in the documented front-left pole/rail region
             (x > -3.6, y < -1.35), i.e. the failure that dominated the real trials.

The heuristic baseline is the top-of-pile rule (argmax z inside the action box), which is what
it does after the yaw fix; it is included because the interesting claim is not "BC beats the
expert on average" but "learning makes target selection ROBUST", and robustness needs a
non-learned reference.

    python3 scripts/envs/noise_sensitivity.py --draws 32 --out docs/figures/noise_sensitivity.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring_head import ScoringGraspPolicy  # noqa: E402

B_MIN = np.array([-5.364, -1.684, -1.30], dtype=np.float32)
B_MAX = np.array([-3.364, 5.316, 0.10], dtype=np.float32)
SH = -0.55                      # rack_y_shift applied to the real clouds
POLE_X, POLE_Y = -3.6, -1.35    # documented front-left pole/rail filter (deploy node)
MODE_EPS = 0.40                 # [m] single-link cluster radius for counting modes
SUPPORT_R = 0.50                # [m] grapple neighbourhood, matches the scoring support gate

RECORDED = "logs/bc_pointcloud/bc_aug1v2_dig25/policy_debug/run_20260803_142901"
RAW = "logs/real_raw_clouds"


class RegressionBC(nn.Module):
    """Standalone mirror of train_bc_pointcloud.PointNetEncoder + actor_mlp.

    Must match that file exactly (per-point Linear+BatchNorm over a FLATTENED (B*N, C) tensor,
    ELU throughout, latent 256, actor 256->128->64->action). The render_policy_comparison mirror
    is a different, older checkpoint's shape and does not load these weights.
    """

    def __init__(self, action_dim=5, latent=256):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.mlp1 = nn.Sequential(
            nn.Linear(3, 64), nn.BatchNorm1d(64), nn.ELU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ELU(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ELU())
        self.encoder.fc = nn.Sequential(
            nn.Linear(256, latent), nn.BatchNorm1d(latent), nn.ELU())
        self.actor_mlp = nn.Sequential(
            nn.Linear(latent, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, action_dim))

    def forward(self, x):
        b, n, _ = x.shape
        f = self.encoder.mlp1(x.reshape(b * n, 3)).view(b, n, -1).max(dim=1)[0]
        return self.actor_mlp(self.encoder.fc(f))


def dec5(a):
    xyz = B_MIN + (np.tanh(a[:3]) + 1.0) / 2.0 * (B_MAX - B_MIN)
    return np.array([xyz[0], xyz[1], xyz[2]], np.float32)


def fps(p, k, seed=0, pre=20000):
    if len(p) <= k:
        out = np.zeros((k, 3), np.float32); out[:len(p)] = p; return out
    rng = np.random.default_rng(seed)
    if len(p) > pre:
        p = p[rng.choice(len(p), pre, replace=False)]
    idx = np.empty(k, np.int64); idx[0] = rng.integers(len(p))
    d = np.linalg.norm(p - p[idx[0]], axis=1)
    for i in range(1, k):
        idx[i] = int(np.argmax(d))
        d = np.minimum(d, np.linalg.norm(p - p[idx[i]], axis=1))
    return p[idx].astype(np.float32)


def crop(p, g):
    m = ((p[:, 0] >= B_MIN[0] - g) & (p[:, 0] <= B_MAX[0] + g) &
         (p[:, 1] >= B_MIN[1] - g) & (p[:, 1] <= B_MAX[1] + g) &
         (p[:, 2] >= B_MIN[2] - g) & (p[:, 2] <= B_MAX[2]))
    return p[m]


def pad_to(p, k):
    out = np.zeros((k, 3), np.float32); n = min(len(p), k); out[:n] = p[:n]; return out


def zed_noise(p, cam, coeff, rng):
    """Axial noise along the camera ray, sigma = coeff * range^2 - the SAME model used for
    training augmentation and sim eval (train_bc_pointcloud.apply_zed_noise), so the sensitivity
    measured here is to the noise the policies were actually trained against."""
    ray = p - cam[None, :]
    r = np.linalg.norm(ray, axis=1, keepdims=True)
    dirn = ray / (r + 1e-6)
    return (p + dirn * (rng.standard_normal((len(p), 1)) * (coeff * r ** 2))).astype(np.float32)


def heuristic_target(cloud):
    """Top-of-pile: highest point inside the action box (the deployed rule after the yaw fix)."""
    m = np.all((cloud >= B_MIN) & (cloud <= B_MAX), axis=1)
    c = cloud[m]
    return c[int(np.argmax(c[:, 2]))].copy() if len(c) else np.full(3, np.nan, np.float32)


def n_modes(T, eps=MODE_EPS):
    """Single-link clusters among the K targets."""
    lab = [-1] * len(T)
    cur = 0
    for i in range(len(T)):
        if lab[i] != -1:
            continue
        stack, lab[i] = [i], cur
        while stack:
            j = stack.pop()
            for k in range(len(T)):
                if lab[k] == -1 and np.linalg.norm(T[j] - T[k]) <= eps:
                    lab[k] = cur; stack.append(k)
        cur += 1
    return cur


def stats(T, clean):
    """Stability AND correctness. Spread alone is misleading: a regression policy is stable
    BECAUSE it emits the average of the valid grasps, gaps included, so low spread can mean
    'consistently aiming at nothing'. `support` (points within the grapple neighbourhood of the
    target, measured on the CLEAN cloud) is what separates stable-and-right from stable-and-wrong.
    """
    T = np.asarray([t for t in T if np.all(np.isfinite(t))], np.float32)
    if len(T) < 2:
        return None
    c = T.mean(0)
    d = np.linalg.norm(clean[None, :, :] - T[:, None, :], axis=2)     # (K, M)
    return {"spread": float(np.linalg.norm(T - c, axis=1).mean()),
            "modes": int(n_modes(T)),
            "pole": float(np.mean((T[:, 0] > POLE_X) & (T[:, 1] < POLE_Y))),
            "nn": float(d.min(axis=1).mean()),                        # m to nearest real point
            "support": float((d <= SUPPORT_R).sum(axis=1).mean()),    # pts in grasp neighbourhood
            "n": int(len(T))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=32)
    ap.add_argument("--coeff", type=float, default=0.0014, help="matches P2c training (zed_coeff)")
    ap.add_argument("--mults", type=float, nargs="+", default=[1.0, 3.0, 8.0],
                    help="noise multipliers on --coeff. 1x is the level the policies were "
                         "trained against; 3x/8x probe how gracefully each degrades (same "
                         "ladder as the earlier bcrl noise sweep).")
    ap.add_argument("--cam", default="logs/bc_pointcloud/bc_margin05_2048_500/cam_pos_base.npy")
    ap.add_argument("--scoring", default="logs/bc_pointcloud/scoring_margin05_2048_c/scoring_policy.pt")
    ap.add_argument("--reg", default="logs/bc_pointcloud/bc_aug1v2_dig25/bc_pointcloud_policy.pt")
    ap.add_argument("--json_out", default="logs/noise_sensitivity.json")
    ap.add_argument("--limit", type=int, default=0, help="use at most N clouds per source (0=all)")
    ap.add_argument("--pre_fps", type=int, default=20000,
                    help="random-subsample the cloud to this many points before FPS. FPS is "
                         "O(k*n) in a Python loop, so this is the wall-clock knob.")
    a = ap.parse_args()

    cam = np.load(a.cam).astype(np.float32).reshape(3)
    print(f"cam_pos(base) = {np.round(cam, 3).tolist()}   coeff={a.coeff}   draws={a.draws}")

    sc = ScoringGraspPolicy(num_points=2048)
    sc.load_state_dict(torch.load(a.scoring, map_location="cpu", weights_only=False)["model_state_dict"])
    rg = RegressionBC()
    rg.load_state_dict(torch.load(a.reg, map_location="cpu", weights_only=False)["model_state_dict"])
    sc.eval(); rg.eval()

    rec = sorted(glob.glob(os.path.join(RECORDED, "policy_debug_*.npz")))
    bag = sorted(glob.glob(os.path.join(RAW, "*.npz")))
    if a.limit:
        # even stride, so a subset still spans the whole run rather than its first minutes
        rec = rec[::max(1, len(rec) // a.limit)][:a.limit]
        bag = bag[::max(1, len(bag) // a.limit)][:a.limit]
    files = [("recorded", f) for f in rec] + [("bag", f) for f in bag]
    if not files:
        raise SystemExit("no real clouds found")

    per_cloud = []
    for mult in a.mults:
      for kind, f in files:
        d = np.load(f)
        p = d["points"].astype(np.float32).copy()
        p[:, 1] -= float(d["rack_y_shift"]) if "rack_y_shift" in d.files else SH
        base = crop(p, 0.5)
        if len(base) < 200:
            continue
        rng = np.random.default_rng(0)
        clean = base if len(base) <= 4000 else base[np.random.default_rng(1).choice(len(base), 4000, replace=False)]
        T = {"heuristic": [], "bc_reg": [], "scoring": []}
        for _ in range(a.draws):
            q = zed_noise(base, cam, a.coeff * mult, rng)
            marg = pad_to(fps(q, 2048, seed=0, pre=a.pre_fps), 2048)
            with torch.no_grad():
                t = sc.act(torch.from_numpy(marg)[None])[0].numpy()
                T["scoring"].append(np.asarray(t[:3], np.float32))
                T["bc_reg"].append(dec5(rg(torch.from_numpy(marg)[None])[0].numpy()))
            T["heuristic"].append(heuristic_target(q))
        row = {"file": os.path.basename(f), "kind": kind, "mult": float(mult),
               **{k: stats(v, clean) for k, v in T.items()}}
        per_cloud.append(row)
        print(f"  {os.path.basename(f)[:34]:34} "
              + "  ".join(f"{k}: sp={row[k]['spread']:.3f} m={row[k]['modes']} p={row[k]['pole']:.2f}"
                          for k in ("heuristic", "bc_reg", "scoring") if row[k]))

    print(f"\n{'noise':>6} {'policy':12} {'clouds':>7} {'spread(m)':>10} {'bistable%':>10} "
          f"{'nn(m)':>7} {'support':>8} {'pole%':>7}")
    print("-" * 72)
    summary = {}
    for mult in a.mults:
        for k in ("heuristic", "bc_reg", "scoring"):
            rows = [r[k] for r in per_cloud if r.get(k) and r["mult"] == mult]
            if not rows:
                continue
            sp = float(np.mean([r["spread"] for r in rows]))
            bi = 100.0 * float(np.mean([r["modes"] > 1 for r in rows]))
            nn = float(np.mean([r["nn"] for r in rows]))
            su = float(np.mean([r["support"] for r in rows]))
            po = 100.0 * float(np.mean([r["pole"] for r in rows]))
            summary[f"{mult:g}x/{k}"] = {"clouds": len(rows), "spread": sp, "bistable_pct": bi,
                                         "nn": nn, "support": su, "pole_pct": po}
            print(f"{mult:5g}x {k:12} {len(rows):7d} {sp:10.3f} {bi:10.1f} {nn:7.3f} "
                  f"{su:8.1f} {po:7.1f}")
        print()
    os.makedirs(os.path.dirname(os.path.abspath(a.json_out)), exist_ok=True)
    with open(a.json_out, "w") as fh:
        json.dump({"summary": summary, "per_cloud": per_cloud,
                   "cfg": {"draws": a.draws, "coeff": a.coeff, "mults": a.mults}}, fh, indent=1)
    print(f"\nwrote {a.json_out}")
    print("spread/bistable = is the target a property of the SCENE or of the NOISE (lower better)\n"
          "nn/support      = is it a real graspable location (nn lower better, support HIGHER better)\n"
          "pole%           = the failure that dominated the real trials (lower better)\n"
          "Read spread and support TOGETHER: a regression policy is stable because it emits the\n"
          "average of valid grasps, so low spread with low support means consistently aiming at a gap.")


if __name__ == "__main__":
    main()
