#!/usr/bin/env python3
"""Measure the SETTLED height profile of a spawned pile from a recorded decision cloud.

Why: the spawn code carves a height profile (mound / two_mounds / ramp), but the logs then
settle under physics and the shape can relax. For the two-mound diagnostic the question is
whether the VALLEY between the peaks survives - a policy can only be caught aiming into a gap
if a gap exists. Spawn geometry predicts peaks ~3.1 m apart with a 0.24-0.38 m valley; the
default-friction pile measured 1.96 m / 0.05 m, i.e. essentially flat.

Profile = the 90th-percentile z in each y-bin (not max: a single stray log on top of the pile
would otherwise define the peak). Peaks are the two highest bins at least --min_sep apart, and
the valley is the lowest bin between them.

    python3 scripts/envs/measure_pile_profile.py logs/sim_eval/friction_probe/*/
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def profile_of(npz, cycle=0, nbins=24, min_pts=4):
    d = np.load(npz, allow_pickle=True)
    p = d["points"][cycle]
    p = p[np.abs(p).sum(-1) > 1e-6]
    # Restrict to the ACTION BOX. Margin-crop policies (--crop_margin) see rack rails, poles and
    # bed as context, and those sit ABOVE the pile top: profiling the raw margin cloud finds a
    # "peak" at z ~ -0.03 m that is structure, not logs, and reports a fake valley. Clipping to
    # the action box makes tight-crop and margin-crop runs measure the same thing.
    if "bounds_min" in d.files and "bounds_max" in d.files:
        lo, hi = d["bounds_min"][cycle], d["bounds_max"][cycle]
        p = p[np.all((p >= lo) & (p <= hi), axis=1)]
    if len(p) < 50:
        return None
    ys, zs = p[:, 1], p[:, 2]
    edges = np.linspace(ys.min(), ys.max(), nbins + 1)
    ctr, hgt = [], []
    for k in range(nbins):
        m = (ys >= edges[k]) & (ys < edges[k + 1])
        if m.sum() >= min_pts:
            ctr.append(0.5 * (edges[k] + edges[k + 1]))
            hgt.append(np.percentile(zs[m], 90))
    return np.array(ctr), np.array(hgt), len(p)


def analyse(row_dir, min_sep=1.5):
    f = sorted(glob.glob(os.path.join(row_dir, "decisions_*.npz")))
    if not f:
        return None
    r = profile_of(f[-1])
    if r is None:
        return None
    y, h, n = r
    order = np.argsort(h)[::-1]
    i1 = order[0]
    i2 = next((k for k in order if abs(y[k] - y[i1]) >= min_sep), None)
    out = {"n": n, "relief": float(h.max() - np.percentile(h, 10))}
    if i2 is None:
        return out
    lo, hi = sorted([i1, i2])
    valley = float(h[lo:hi + 1].min())
    out.update({"y1": float(y[i1]), "h1": float(h[i1]), "y2": float(y[i2]), "h2": float(h[i2]),
                "sep": float(abs(y[i1] - y[i2])), "valley": valley,
                "depth": float(min(h[i1], h[i2]) - valley)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--min_sep", type=float, default=1.5)
    a = ap.parse_args()
    print(f"{'run':16} {'pts':>5} {'relief':>7} {'sep':>6} {'peak1':>7} {'peak2':>7} "
          f"{'valley':>7} {'DEPTH':>7}")
    print("-" * 72)
    for d in a.dirs:
        d = d.rstrip("/")
        r = analyse(d, a.min_sep)
        name = os.path.basename(d)
        if r is None:
            print(f"{name:16} (no decisions yet)")
            continue
        if "sep" not in r:
            print(f"{name:16} {r['n']:5d} {r['relief']:7.2f}   (no second peak: not a double mound)")
            continue
        print(f"{name:16} {r['n']:5d} {r['relief']:7.2f} {r['sep']:6.2f} {r['h1']:7.2f} "
              f"{r['h2']:7.2f} {r['valley']:7.2f} {r['depth']:7.2f}")
    print("\nDEPTH = lower peak minus the valley between the peaks. Spawn geometry intends "
          "0.24-0.38 m;\nanything under ~0.10 m is a flat pile with no gap for a policy to aim into.")


if __name__ == "__main__":
    main()
