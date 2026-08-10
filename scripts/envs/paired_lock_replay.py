"""Paired targeting replay: scoring policy vs the deployed heuristic, same clouds.

For every gaze cycle of the three 2026-08-03 baseline (heuristic) trials, the
node saved the cloud it consumed and the target it issued. This script runs the
deployed scoring checkpoint on those same clouds and classifies both policies'
targets with one symmetric criterion, so the comparison is same-input,
same-classifier.

Classifier (validated against the heuristic's own deposit outcomes below):
a target is STRUCTURE/NOISE-SUSPECT if it lies within EDGE_D of the action-box
boundary in x or y (rails, end boards and pole columns all live at the box
periphery) OR its local support (points within SUPPORT_R horizontally) is
below SUPPORT_MIN (isolated noise points).

Limitation, stated up front: the baseline node saved the tight 1024-point
crop, while the deployed scoring policy trains on the margin crop at 2048
points; its inputs here lack the margin band. The full-fidelity version
re-extracts margin clouds from the trial bags (T9 drive).

Run on HOST (CPU is fine): python3 scripts/envs/paired_lock_replay.py
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring_head import ScoringGraspPolicy  # noqa: E402

ROOT = "/home/george/IsaacLab/crane_testbed"
RUNS = {  # run dir -> trial name (matched by deposit pattern and cycle count)
    "run_20260803_155559": "double mound",
    "run_20260803_174017": "jagged",
    "run_20260803_204034": "single mound",
}
CKPT = f"{ROOT}/logs/bc_pointcloud/scoring_margin05_2048_c/scoring_policy.pt"

EDGE_D = 0.30       # m to the action-box boundary in x or y
SUPPORT_R = 0.5     # m, the deployed support radius
SUPPORT_MIN = 15    # points


def d_edge(t, lo, hi):
    return min(t[0] - lo[0], hi[0] - t[0], t[1] - lo[1], hi[1] - t[1])


def support(t, pts):
    return int((np.linalg.norm(pts[:, :2] - np.asarray(t[:2]), axis=1) < SUPPORT_R).sum())


def suspect(t, pts, lo, hi):
    return d_edge(t, lo, hi) < EDGE_D or support(t, pts) < SUPPORT_MIN


def load_run(d):
    rows = [json.loads(l) for l in open(f"{d}/decisions.jsonl")]
    dep = {r["cycle"] for r in rows if r["kind"] == "deposit"}
    out = []
    for r in rows:
        if r["kind"] != "gaze":
            continue
        z = np.load(f"{d}/{r['npz']}")
        out.append({
            "cycle": r["cycle"], "pts": z["points"].astype(np.float32),
            "h_tgt": np.asarray(r["target"], np.float32),
            "lo": z["bounds_min"], "hi": z["bounds_max"],
            "shift": float(z["rack_y_shift"]), "deposited": r["cycle"] in dep,
        })
    return out


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    npts = int(ck["num_points"])
    model = ScoringGraspPolicy(num_points=npts)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    print("== classifier validation on the heuristic's own targets ==")
    all_cycles = []
    for d, name in RUNS.items():
        cyc = load_run(f"{ROOT}/logs/crane_policy_debug/{d}")
        for c in cyc:
            c["trial"] = name
        all_cycles += cyc
    dep_s = [suspect(c["h_tgt"], c["pts"], c["lo"], c["hi"]) for c in all_cycles if c["deposited"]]
    emp_s = [suspect(c["h_tgt"], c["pts"], c["lo"], c["hi"]) for c in all_cycles if not c["deposited"]]
    print(f"  deposited cycles flagged suspect: {sum(dep_s)}/{len(dep_s)}")
    print(f"  empty     cycles flagged suspect: {sum(emp_s)}/{len(emp_s)}")

    print("\n== paired replay ==")
    rows = []
    for c in all_cycles:
        p_train = c["pts"].copy()
        p_train[:, 1] -= c["shift"]
        buf = np.zeros((npts, 3), np.float32)
        n = min(len(p_train), npts)
        buf[:n] = p_train[:n]
        with torch.no_grad():
            s_tgt = model.act(torch.from_numpy(buf).unsqueeze(0))[0].numpy()
        s_tgt[1] += c["shift"]
        rows.append({
            "trial": c["trial"], "cycle": c["cycle"], "deposited": c["deposited"],
            "h_suspect": suspect(c["h_tgt"], c["pts"], c["lo"], c["hi"]),
            "s_suspect": suspect(s_tgt, c["pts"], c["lo"], c["hi"]),
            "h_tgt": c["h_tgt"].tolist(), "s_tgt": s_tgt.tolist(),
            "h_sup": support(c["h_tgt"], c["pts"]), "s_sup": support(s_tgt, c["pts"]),
            "displacement": float(np.linalg.norm(np.asarray(s_tgt[:2]) - c["h_tgt"][:2])),
        })

    n = len(rows)
    hS = sum(r["h_suspect"] for r in rows)
    sS = sum(r["s_suspect"] for r in rows)
    b = sum(1 for r in rows if r["h_suspect"] and not r["s_suspect"])
    c_ = sum(1 for r in rows if not r["h_suspect"] and r["s_suspect"])
    print(f"  cycles: {n}   heuristic suspect: {hS}   scoring suspect: {sS}")
    print(f"  discordant pairs: heuristic-only {b}, scoring-only {c_}")

    # collapse consecutive heuristic-suspect cycles within a trial into lock episodes
    episodes = []
    for name in RUNS.values():
        tr = [r for r in rows if r["trial"] == name]
        tr.sort(key=lambda r: r["cycle"])
        run = []
        for r in tr:
            if r["h_suspect"]:
                run.append(r)
            else:
                if run:
                    episodes.append(run)
                run = []
        if run:
            episodes.append(run)
    print(f"\n  heuristic lock episodes: {len(episodes)} "
          f"(lengths {[len(e) for e in episodes]})")
    esc = [all(not r["s_suspect"] for r in e) for e in episodes]
    print(f"  episodes where scoring targeted material on EVERY cloud: {sum(esc)}/{len(esc)}")

    from scipy import stats
    if b + c_ > 0:
        p_mcnemar = stats.binomtest(b, b + c_, 0.5).pvalue
        print(f"  cycle-level McNemar exact p = {p_mcnemar:.4g}  (b={b}, c={c_})")
    k = sum(esc)
    p_ep = stats.binomtest(k, len(esc), 0.5, alternative="greater").pvalue
    ci = stats.binomtest(k, len(esc)).proportion_ci(0.95)
    print(f"  episode-level: {k}/{len(esc)} escaped, exact binomial p (vs 0.5) = {p_ep:.4g},"
          f" 95% CI [{ci.low:.2f}, {ci.high:.2f}]")

    out = f"{ROOT}/results/analysis/paired_lock_replay.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"rows": rows, "episodes": [[r["cycle"] for r in e] for e in episodes],
               "escaped": esc, "params": {"EDGE_D": EDGE_D, "SUPPORT_R": SUPPORT_R,
               "SUPPORT_MIN": SUPPORT_MIN}}, open(out, "w"), indent=1)
    print(f"\n  written: {out}")


if __name__ == "__main__":
    main()
