#!/usr/bin/env python3
"""Commanded grasp depth below the observed pile surface, across the hardware campaign.

Backs the digging-depth convention with trial data: successful grasps occupy a depth
band bounded on both sides by observed failure modes. Too shallow (the 2026-07-27
surface-level sessions) the closing tongs rake the pile top and capture nothing; too
deep the grapple chokes and cannot close (the 'choked' cycles of the August trials).
The depth commanded by the learned policies, which was never tuned by hand, is read
off the same recordings.

Depth metric: commanded target z minus the local observed surface, where the local
surface is the highest cloud point within SURF_R of the target in the horizontal
plane, on the exact cloud the policy saw (policy_debug_*.npz). This is the quantity
the policy controls; it is internal to the cloud frame, so it is unaffected by the
absolute camera-height calibration state of a given day.

Outcome labels come from the manually scored ~/Documents/real_trials.ods (per-cycle
logs, alignment, failure note), NOT from deposit events. Cycles whose note marks
structure or noise targeting (rack / pole / air) are excluded from the depth band:
their 'local surface' is a rail or a phantom point, so depth below it is not a
statement about the pile. Runs are matched to spreadsheet blocks by trial shape and
cycle count; re-gazed cycles are deduplicated by cycle number (last record wins).
The BCRL double-mound trial is not in the spreadsheet and is reported without an
outcome split.

    python3 scripts/envs/analyze_grasp_depth.py [--ods ~/Documents/real_trials.ods]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

SURF_R = 0.20          # horizontal radius for the local surface [m]
LOG_RADIUS = 0.056     # sim expert convention: grasp z = surface - LOG_RADIUS

# Trial registry: spreadsheet block/column -> run directory.
# Baseline shapes verified against the first-cycle cloud profile (see check_shape).
TRIALS = [
    # (policy, shape, run_dir, labeled)
    ("Heuristic", "double", "logs/crane_policy_debug/run_20260803_155559", True),
    ("Heuristic", "flat",   "logs/crane_policy_debug/run_20260803_174017", True),
    ("Heuristic", "single", "logs/crane_policy_debug/run_20260803_204034", True),
    ("BC",        "double", "logs/bc_pointcloud/scoring_margin05_2048_c/policy_debug/run_20260805_194008", True),
    ("BC",        "single", "logs/bc_pointcloud/scoring_margin05_2048_c/policy_debug/run_20260805_153036", True),
    ("BC",        "flat",   "logs/bc_pointcloud/scoring_margin05_2048_c/policy_debug/run_20260812_134304", True),
    ("BCRL",      "flat",   "logs/deploy/policy_debug/run_20260812_191253", True),
    # BC+RL Trial 03 (double mound, 2026-08-13): labels come from the shared CSV
    # export rather than the ods. The mapping was verified cycle by cycle: the
    # CSV's 'targeted right rack base' cycles (19-21, 25) all land at the far-y
    # rack end with floor-clamped z, and its 'left side' failures (13-17, 26-30)
    # all land at y < -1 in the recording.
    ("BCRL",      "double", "logs/deploy/policy_debug/run_20260813_180430", "csv"),
]
JULY = [
    ("Jul 27 (surface-level)", "logs/crane_policy_debug/run_20260727_*"),
    ("Jul 30 (dig probe)",     "logs/crane_policy_debug/run_20260730_*"),
]

BLOCK_ROW = {"Heuristic": "BASELINE", "BC": "BC", "BCRL": "BCRL"}
SHAPE_COL = {"double": 0, "flat": 3, "single": 7}


def read_ods_rows(path):
    with zipfile.ZipFile(os.path.expanduser(path)) as z:
        root = ET.fromstring(z.read("content.xml"))
    tns = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"
    xns = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
    rows = []
    for tr in root.iter(tns + "table-row"):
        row = []
        for tc in tr.findall(tns + "table-cell"):
            rep = int(tc.get(tns + "number-columns-repeated", 1))
            txt = "".join(t.text or "" for t in tc.iter(xns + "p"))
            row.extend([txt] * min(rep, 30))
        rows.append(row)
    return rows


def parse_block(rows, policy, shape):
    """Per-cycle (logs, note) for one trial column, ending at the total row."""
    hdr = BLOCK_ROW[policy]
    col = SHAPE_COL[shape]
    start = next(i for i, r in enumerate(rows) if r and r[0:1] == [hdr] or
                 (r and len(r) > col and r[col] == hdr))
    out = []
    for r in rows[start + 1:]:
        cell = r[col] if len(r) > col else ""
        if not cell.strip():
            break
        try:
            logs = int(cell)
        except ValueError:
            break
        if logs > 60:          # inventory total row terminates the column
            break
        note = r[col + 2] if len(r) > col + 2 else ""
        out.append((logs, note.strip().lower()))
    return out


def load_cycles(run_dir):
    """(cycle, target, cloud) per executed cycle; re-gazes deduped, last wins."""
    dj = os.path.join(run_dir, "decisions.jsonl")
    recs = [json.loads(l) for l in open(dj)]
    recs = [r for r in recs if r.get("kind", "gaze") == "gaze"]
    by_cycle = {}
    for r in recs:
        by_cycle[int(r["cycle"])] = r
    out = []
    for c in sorted(by_cycle):
        r = by_cycle[c]
        npz = os.path.join(run_dir, r.get("npz", "policy_debug_%03d.npz" % c))
        if not os.path.exists(npz):
            continue
        z = np.load(npz, allow_pickle=True)
        pts = z["points"].astype(np.float32)
        pts = pts[np.abs(pts).sum(-1) > 1e-6]
        out.append((c, np.asarray(r["target"][:3], np.float32), pts))
    return out


def depth_below_surface(target, pts, r=SURF_R):
    d = np.linalg.norm(pts[:, :2] - target[None, :2], axis=1)
    near = pts[d < r]
    if not len(near):
        near = pts[np.argsort(d)[:10]]
    return float(target[2] - near[:, 2].max())


def check_shape(pts, shape):
    """Sanity-check the run/spreadsheet mapping from the first-cycle height profile."""
    # Interior of the rack only: the margin band and the rack ends carry poles and
    # end boards, which read as spurious ridges on a flat pile.
    core = pts[(pts[:, 0] > -5.2) & (pts[:, 0] < -3.55)
               & (pts[:, 1] > -1.0) & (pts[:, 1] < 4.7)]
    if len(core) < 100:
        core = pts
    ys = np.linspace(core[:, 1].min(), core[:, 1].max(), 24)
    prof = []
    for y in ys:
        near = core[np.abs(core[:, 1] - y) < 0.25]
        if len(near):
            prof.append(np.percentile(near[:, 2], 95))
    prof = np.array(prof)
    if len(prof) < 10:
        return "?"
    prof = prof[2:-2]      # rack-end bins: pile tails and end boards, not shape
    hi = prof > prof.min() + 0.6 * (prof.max() - prof.min())
    n_ridges = int(np.diff(hi.astype(int)).clip(min=0).sum()) + int(hi[0])
    span = float(prof.max() - prof.min())
    got = "flat" if span < 0.45 else ("double" if n_ridges >= 2 else "single")
    return "ok" if got == shape else "MISMATCH(%s, span %.2f, ridges %d)" % (got, span, n_ridges)


def categorize(logs, note):
    if re.search(r"rack|pole|noise|air|structure", note):
        return "structure"
    if re.search(r"chok", note):
        return "choke"
    if re.search(r"not deep enough|did not go as deep", note):
        return "shallow-exec"   # executed shallow from a calibration bias, command looked normal
    if logs > 0:
        return "success"
    return "other"


CSV_TRIALS = {
    # (policy, shape) -> CSV export path (one trial per file)
    ("BCRL", "double"): "~/Downloads/log_loader_ML_experimental_data(03_BC_RL).csv",
}


def parse_csv_trial(path):
    """Per-cycle (logs, note) from a shared one-trial CSV export."""
    import csv as _csv
    out = []
    with open(os.path.expanduser(path)) as f:
        for row in _csv.reader(f):
            if not row or not row[0].strip():
                continue
            try:
                int(row[0])
            except ValueError:
                continue
            logs = int(row[1]) if len(row) > 1 and row[1].strip().isdigit() else 0
            note = (row[4] if len(row) > 4 else "").strip().lower().replace("\n", " ")
            out.append((logs, note))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ods", default="~/Documents/real_trials.ods")
    ap.add_argument("--fig", default="")
    ap.add_argument("--fig2", default="", help="two-cycle qualitative slice figure")
    args = ap.parse_args()
    rows = read_ods_rows(args.ods)

    samples = []   # (policy, shape, cycle, depth, category, logs, note)
    for policy, shape, run, labeled in TRIALS:
        cyc = load_cycles(run)
        print("[%s %s] %s: %d cycles, shape check %s" %
              (policy, shape, os.path.basename(run), len(cyc), check_shape(cyc[0][2], shape)))
        if labeled == "csv":
            lab = parse_csv_trial(CSV_TRIALS[(policy, shape)])
        elif labeled:
            lab = parse_block(rows, policy, shape)
        else:
            lab = None
        if lab is not None:
            if abs(len(lab) - len(cyc)) > 2:
                print("  WARN cycle count mismatch: %d npz vs %d labeled" % (len(cyc), len(lab)))
            for (c, tgt, pts), (logs, note) in zip(cyc, lab):
                samples.append((policy, shape, c, depth_below_surface(tgt, pts),
                                categorize(logs, note), logs, note))
        else:
            for c, tgt, pts in cyc:
                samples.append((policy, shape, c, depth_below_surface(tgt, pts),
                                "unlabeled", -1, ""))

    july = {}
    for name, pat in JULY:
        ds = []
        for run in sorted(glob.glob(pat)):
            if not os.path.exists(os.path.join(run, "decisions.jsonl")):
                continue
            for c, tgt, pts in load_cycles(run):
                ds.append(depth_below_surface(tgt, pts))
        july[name] = np.array(ds)
        print("[%s] n=%d  mean %+.3f  p10 %+.3f  p90 %+.3f" %
              (name, len(ds), np.mean(ds), np.percentile(ds, 10), np.percentile(ds, 90)))

    def stat(sel, label):
        a = np.array([s[3] for s in sel])
        if len(a):
            print("%-38s n=%3d  mean %+.3f  median %+.3f  [p10 %+.3f, p90 %+.3f]" %
                  (label, len(a), a.mean(), np.median(a), np.percentile(a, 10), np.percentile(a, 90)))
        return a

    print("\n=== August trials: commanded depth below local surface [m] ===")
    for pol in ("Heuristic", "BC", "BCRL"):
        for cat in ("success", "choke", "shallow-exec", "other"):
            stat([s for s in samples if s[0] == pol and s[4] == cat], "%s / %s" % (pol, cat))
        n_struct = len([s for s in samples if s[0] == pol and s[4] == "structure"])
        print("%-38s n=%3d  (excluded from depth band)" % (pol + " / structure-noise", n_struct))
    stat([s for s in samples if s[4] == "unlabeled"], "BCRL double (unlabeled)")
    stat([s for s in samples if s[4] == "success"], "ALL successes")
    stat([s for s in samples if s[4] == "choke"], "ALL chokes")

    if args.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        groups = [
            ("Jul 27 tuning\n(raked, no capture)", july["Jul 27 (surface-level)"], "#c44e52"),
            ("Jul 30 dig probe", july["Jul 30 (dig probe)"], "#8172b2"),
            ("Heuristic successes", np.array([s[3] for s in samples if s[0] == "Heuristic" and s[4] == "success"]), "#55a868"),
            ("BC successes", np.array([s[3] for s in samples if s[0] == "BC" and s[4] == "success"]), "#55a868"),
            ("BC$\\to$RL successes", np.array([s[3] for s in samples if s[0] == "BCRL" and s[4] == "success"]), "#55a868"),
            ("Chokes (all policies)", np.array([s[3] for s in samples if s[4] == "choke"]), "#c44e52"),
        ]
        fig, ax = plt.subplots(figsize=(7.0, 3.4))
        rng = np.random.RandomState(0)
        for i, (name, a, color) in enumerate(groups):
            y = i + rng.uniform(-0.16, 0.16, len(a))
            ax.scatter(a, y, s=14, alpha=0.65, color=color, edgecolors="none", zorder=3)
            ax.scatter([np.median(a)], [i], marker="|", s=380, color="black", zorder=4, linewidths=1.8)
        ax.axvline(0.0, color="0.35", lw=0.9)
        ax.axvline(-LOG_RADIUS, color="0.35", lw=0.9, ls="--")
        ax.text(0.004, -0.62, "observed surface", fontsize=7.5, color="0.25", rotation=90, va="top")
        ax.text(-LOG_RADIUS + 0.004, -0.62, "sim expert convention", fontsize=7.5, color="0.25", rotation=90, va="top")
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels([g[0] for g in groups], fontsize=8.5)
        ax.set_xlabel("commanded grasp depth below the local observed surface [m]", fontsize=9)
        ax.invert_yaxis()
        ax.grid(axis="x", lw=0.4, alpha=0.4)
        fig.tight_layout()
        fig.savefig(args.fig, dpi=200)
        print("figure ->", args.fig)

    if args.fig2:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def pick_cycle(run_dir, want_cycle=None, depth_lo=None, depth_hi=None):
            for c, tgt, pts in load_cycles(run_dir):
                d = depth_below_surface(tgt, pts)
                if want_cycle is not None and c != want_cycle:
                    continue
                if depth_lo is not None and not (depth_lo <= d <= depth_hi):
                    continue
                return c, tgt, pts, d
            raise SystemExit("no matching cycle in " + run_dir)

        # (a) a Jul 27 surface-level command that raked; (b) a deep-bite success
        # from the BC double-mound trial (labeled success, 18 logs).
        rake_run = sorted(glob.glob("logs/crane_policy_debug/run_20260727_*"))
        rake_run = [r for r in rake_run if os.path.exists(r + "/decisions.jsonl")][-1]
        a = pick_cycle(rake_run, depth_lo=-0.05, depth_hi=0.05)
        b_run = "logs/bc_pointcloud/scoring_margin05_2048_c/policy_debug/run_20260805_194008"
        b = pick_cycle(b_run, depth_lo=-0.30, depth_hi=-0.20)

        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=False)
        for ax, (c, tgt, pts, d), title in zip(
                axes, (a, b),
                ("(a) surface-level command (Jul 27):\ntongs rake the top, capture nothing",
                 "(b) deployed scoring policy:\nlearned bite %.2f m below the surface" % -b[3])):
            slab = pts[np.abs(pts[:, 0] - tgt[0]) < 0.35]
            near = slab[np.abs(slab[:, 1] - tgt[1]) < 1.35]
            ax.scatter(near[:, 1], near[:, 2], s=5, color="0.55", alpha=0.65)
            dloc = np.linalg.norm(pts[:, :2] - tgt[None, :2], axis=1)
            surf = pts[dloc < SURF_R][:, 2].max()
            ax.plot([tgt[1] - SURF_R, tgt[1] + SURF_R], [surf, surf], color="#4c72b0", lw=1.8)
            ax.plot([tgt[1]], [tgt[2]], marker="x", color="#c44e52", ms=10, mew=2.4)
            ya = tgt[1] + 0.30
            ax.annotate("", xy=(ya, tgt[2]), xytext=(ya, surf),
                        arrowprops=dict(arrowstyle="<->", color="black", lw=1.0))
            ax.text(ya + 0.07, (tgt[2] + surf) / 2, "%.2f m" % (surf - tgt[2]),
                    fontsize=8, va="center")
            ax.set_xlim(tgt[1] - 1.35, tgt[1] + 1.35)
            ax.set_ylim(surf - 0.80, surf + 0.30)
            ax.set_title(title, fontsize=8.5)
            ax.set_xlabel("y [m]", fontsize=8)
            ax.set_ylabel("z [m]", fontsize=8)
            ax.tick_params(labelsize=7.5)
            ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(args.fig2, dpi=200)
        print("figure2 -> %s  (rake: %s cycle %d, bite: cycle %d)"
              % (args.fig2, os.path.basename(rake_run), a[0], b[0]))


if __name__ == "__main__":
    main()
