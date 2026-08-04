#!/usr/bin/env python3
"""Compare seed-paired policy evals: summary table, paired stats, decision-level figures.

Consumes, per policy dir, the newest eval_metrics_*.json and decisions_*.npz written by
play_bc_pointcloud.py (--save_metrics --save_decisions). All evals must share the seed so
episode k of env i is the same pile for every policy (paired design).

Outputs (to --out): summary.md/.csv, paired_stats.md, and figures:
  stability_paired.png      slope chart, one line per episode across the 2x2 policies
  clearing_progression.png  mean clearing %% vs cycle with std band, per policy
  dig_depth.png             violin of target z minus local pile surface, per policy
  target_heatmaps.png       2D histogram of grasp targets over the rack, per policy
  depletion_trends.png      success/stability/dig depth/logs-per-grasp vs pile remaining

Optional --replay: run every policy offline on the union of logged clouds (CPU) ->
decision divergence matrix + delta-z distributions between lineage pairs; with
--real_clouds also on real-crane clouds (the input-conditional correction test).

Usage:
  python3 compare_evals.py [--policies LABEL=DIR ...] [--out DIR] [--replay]
                           [--real_clouds <pointclouds.npy>]
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_POLICIES = [
    ("BC",          "logs/bc_pointcloud/bc_20260713_182049"),
    ("BCRL",        "logs/rsl_rl/crane_pointcloud_gaze_cossin_raw_mr_v0/2026-07-15_05-08-06"),
    ("BC+SFT",      "logs/bc_pointcloud/bc182049_sft_bagmix2"),
    ("BCRL+SFT",    "logs/bc_pointcloud/bcrl380_sft_bagmix2"),
    ("BCRL250+SFT", "logs/bc_pointcloud/bcrl250_sft_bagmix2"),
    ("bagmix2",     "logs/bc_pointcloud/bc_20260713_052741_bagmix2"),
]

# Planned paired contrasts (Holm-corrected per metric within this family)
CONTRASTS = [
    ("BCRL", "BC", "RL gain exists"),
    ("BCRL+SFT", "BC+SFT", "RL gain survives SFT"),
    ("BCRL", "BCRL+SFT", "SFT cost on RL lineage"),
    ("BC", "BC+SFT", "SFT cost on BC lineage"),
]
STAT_METRICS = ["stabilities", "success_rates", "clearing_pcts", "throughputs"]
CORE4 = ["BC", "BCRL", "BC+SFT", "BCRL+SFT"]

SUMMARY_KEYS = [
    ("pile_cleared_pct", "Clearing %"),
    ("grasp_success_pct", "Success %"),
    ("throughput", "Logs/grasp"),
    ("alignment", "Alignment"),
    ("stability", "Stability"),
    ("cycles_to_95pct", "Cycles to 95%"),
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="*", default=None,
                    help="LABEL=DIR overrides (default: the six overnight policies)")
    ap.add_argument("--out", default=None, help="output dir (default logs/eval_analysis)")
    ap.add_argument("--replay", action="store_true", help="counterfactual replay (needs torch)")
    ap.add_argument("--real_clouds", default=None,
                    help="pointclouds.npy of real clouds for the sim-vs-real delta-z test")
    ap.add_argument("--replay_cap", type=int, default=2000, help="max clouds for replay")
    return ap.parse_args()


def newest(pattern):
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return files[-1] if files else None


def load_policy_data(label, d):
    d = d if os.path.isabs(d) else os.path.join(REPO, d)
    mfile = newest(os.path.join(d, "eval_metrics_*.json"))
    dfile = newest(os.path.join(d, "decisions_*.npz"))
    if mfile is None:
        print(f"[warn] {label}: no eval_metrics in {d}, skipping")
        return None
    with open(mfile) as f:
        metrics = json.load(f)
    dec = np.load(dfile, allow_pickle=True) if dfile else None
    print(f"[load] {label}: {os.path.basename(mfile)}"
          + (f" + {os.path.basename(dfile)} ({len(dec['cycle'])} grasps)" if dec is not None else " (no decisions)"))
    return {"label": label, "dir": d, "metrics": metrics, "dec": dec,
            "ckpt": metrics["eval_config"].get("checkpoint")}


# -- local pile surface at the target ----------------------------------------

def local_surface_z(points, x, y, radius=0.4, min_pts=10, pctl=90):
    """Robust local surface height near (x, y): pctl of z among points within radius,
    expanding to the nearest 30 points if the disk is too empty."""
    p = points[np.any(points != 0.0, axis=1)]
    d2 = (p[:, 0] - x) ** 2 + (p[:, 1] - y) ** 2
    near = p[d2 <= radius * radius]
    if len(near) < min_pts:
        near = p[np.argsort(d2)[:30]]
    return float(np.percentile(near[:, 2], pctl)) if len(near) else float("nan")


def dig_depths(dec):
    """target z minus local surface, per grasp record."""
    out = np.empty(len(dec["cycle"]), dtype=np.float32)
    for i in range(len(out)):
        s = local_surface_z(dec["points"][i], dec["target"][i, 0], dec["target"][i, 1])
        out[i] = dec["target"][i, 2] - s
    return out


def remaining_fraction(dec):
    """logs_remaining / starting logs per record; NaN where not recorded."""
    rem = dec["logs_remaining"].astype(np.float32) if "logs_remaining" in dec.files else None
    if rem is None or (rem < 0).all():
        return np.full(len(dec["cycle"]), np.nan, dtype=np.float32)
    out = np.full(len(dec["cycle"]), np.nan, dtype=np.float32)
    key = dec["env"].astype(np.int64) * 100000 + dec["episode"].astype(np.int64)
    for k in np.unique(key):
        idx = np.where(key == k)[0]
        idx = idx[np.argsort(dec["cycle"][idx])]
        first = idx[0]
        start = rem[first] + dec["logs_grasped"][first] + (dec["knocked_off"][first] if "knocked_off" in dec.files else 0)
        if start > 0:
            out[idx] = rem[idx] / float(start)
    return out


# -- sections -----------------------------------------------------------------

def write_summary(policies, out_dir):
    lines = ["| Policy | " + " | ".join(t for _, t in SUMMARY_KEYS) + " |",
             "|" + "---|" * (len(SUMMARY_KEYS) + 1)]
    csv = ["policy," + ",".join(k for k, _ in SUMMARY_KEYS)]
    for p in policies:
        s = p["metrics"]["summary"]
        cells, csvc = [], []
        for k, _ in SUMMARY_KEYS:
            v = s.get(k)
            if isinstance(v, dict):
                cells.append(f"{v['mean']:.2f} +/- {v['std']:.2f}")
                csvc.append(f"{v['mean']:.4f}")
            else:
                cells.append("-" if v is None else f"{v:.2f}")
                csvc.append("" if v is None else f"{v:.4f}")
        lines.append(f"| {p['label']} | " + " | ".join(cells) + " |")
        csv.append(p["label"] + "," + ",".join(csvc))
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "summary.csv"), "w") as f:
        f.write("\n".join(csv) + "\n")
    print("\n".join(lines))


def paired_stats(policies, out_dir):
    by = {p["label"]: p for p in policies}
    lines = ["# Paired contrasts (Wilcoxon signed-rank, Holm-corrected per metric)", ""]
    for metric in STAT_METRICS:
        rows = []
        for a, b, desc in CONTRASTS:
            if a not in by or b not in by:
                continue
            va = np.asarray(by[a]["metrics"]["per_episode"][metric], dtype=float)
            vb = np.asarray(by[b]["metrics"]["per_episode"][metric], dtype=float)
            n = min(len(va), len(vb))
            if n < 5:
                continue
            diff = va[:n] - vb[:n]
            try:
                p = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            except ValueError:
                p = 1.0
            boots = [np.mean(np.random.default_rng(s).choice(diff, n)) for s in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append([f"{a} - {b}", desc, n, float(np.median(diff)),
                         float(np.mean(diff)), lo, hi, p])
        order = np.argsort([r[-1] for r in rows])
        m = len(rows)
        adj = {}
        prev = 0.0
        for rank, ri in enumerate(order):
            padj = min(1.0, max(prev, (m - rank) * rows[ri][-1]))
            adj[ri] = padj
            prev = padj
        lines.append(f"## {metric}")
        lines.append("| contrast | question | n | median diff | mean diff | 95% CI | p | p (Holm) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for ri, r in enumerate(rows):
            lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.3f} | {r[4]:.3f} | "
                         f"[{r[5]:.3f}, {r[6]:.3f}] | {r[7]:.4f} | {adj.get(ri, 1.0):.4f} |")
        lines.append("")
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "paired_stats.md"), "w") as f:
        f.write(text + "\n")
    print(text)


def fig_stability_paired(policies, out_dir):
    by = {p["label"]: p for p in policies}
    labels = [l for l in CORE4 if l in by]
    if len(labels) < 2:
        return
    vals = np.array([by[l]["metrics"]["per_episode"]["stabilities"] for l in labels])
    n = min(v.shape[0] if v.ndim else len(v) for v in vals)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    xs = np.arange(len(labels))
    for e in range(min(n, vals.shape[1])):
        ax.plot(xs, vals[:, e], color="gray", alpha=0.3, linewidth=0.6)
    ax.plot(xs, vals.mean(axis=1), color="#d62728", linewidth=2, marker="o", label="mean")
    ax.set_xticks(xs, labels)
    ax.set_ylabel("Per-episode stability")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stability_paired.png"), dpi=200)
    plt.close(fig)


def fig_clearing(policies, out_dir):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for p in policies:
        curves = p["metrics"]["per_episode"]["clearing_curves"]
        L = max(len(c) for c in curves)
        arr = np.array([c + [c[-1]] * (L - len(c)) for c in curves], dtype=float)
        x = np.arange(1, L + 1)
        m, s = arr.mean(axis=0), arr.std(axis=0)
        ax.plot(x, m, label=p["label"], linewidth=1.2)
        ax.fill_between(x, m - s, m + s, alpha=0.12)
    ax.set_xlabel("Grasp cycle")
    ax.set_ylabel("Pile cleared [%]")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "clearing_progression.png"), dpi=200)
    plt.close(fig)


def fig_dig_depth(policies, out_dir):
    data, labels = [], []
    for p in policies:
        if p["dec"] is None:
            continue
        dd = p.setdefault("dig", dig_depths(p["dec"]))
        data.append(dd[~np.isnan(dd)])
        labels.append(p["label"])
    if not data:
        return
    fig, ax = plt.subplots(figsize=(6, 3.2))
    vp = ax.violinplot(data, showmedians=True)
    ax.axhline(0.0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=20)
    ax.set_ylabel("Target z - local surface [m]")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dig_depth.png"), dpi=200)
    plt.close(fig)


def fig_target_heatmaps(policies, out_dir):
    with_dec = [p for p in policies if p["dec"] is not None]
    if not with_dec:
        return
    ncol = 3
    nrow = int(np.ceil(len(with_dec) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3.4 * nrow), squeeze=False)
    for ax, p in zip(axes.flat, with_dec):
        t = p["dec"]["target"]
        h = ax.hist2d(t[:, 0], t[:, 1], bins=[14, 28], cmap="magma")
        ax.set_title(p["label"], fontsize=9)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    for ax in axes.flat[len(with_dec):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "target_heatmaps.png"), dpi=200)
    plt.close(fig)


def fig_depletion(policies, out_dir):
    bins = [(1.0, 0.75), (0.75, 0.5), (0.5, 0.25), (0.25, 0.0)]
    blabels = ["100-75%", "75-50%", "50-25%", "<25%"]
    panels = [("grasp success [%]", lambda d, m: 100.0 * (d["logs_grasped"][m] > 0).mean()),
              ("stability (successful)", lambda d, m: float(d["stability"][m & (d["logs_grasped"] > 0)].mean())
               if (m & (d["logs_grasped"] > 0)).any() else np.nan),
              ("logs per grasp", lambda d, m: float(d["logs_grasped"][m].mean())),
              ("dig depth [m]", None)]  # filled from cached dig depths
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.4))
    for p in policies:
        if p["dec"] is None:
            continue
        frac = remaining_fraction(p["dec"])
        if np.isnan(frac).all():
            # fall back to cycle thirds when remaining counts are absent
            cyc = p["dec"]["cycle"].astype(float)
            frac = 1.0 - cyc / max(1.0, cyc.max())
        dd = p.setdefault("dig", dig_depths(p["dec"]))
        for ax, (title, fn) in zip(axes.flat, panels):
            ys = []
            for hi, lo in bins:
                m = (frac <= hi) & (frac > lo)
                if not m.any():
                    ys.append(np.nan)
                elif fn is None:
                    ys.append(float(np.nanmean(dd[m])))
                else:
                    ys.append(fn(p["dec"], m))
            ax.plot(np.arange(len(bins)), ys, marker="o", linewidth=1.1, label=p["label"])
            ax.set_title(title, fontsize=9)
            ax.set_xticks(np.arange(len(bins)), blabels, fontsize=7)
            ax.grid(True, alpha=0.25)
    axes.flat[0].legend(fontsize=6)
    fig.suptitle("Behavior vs pile remaining", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "depletion_trends.png"), dpi=200)
    plt.close(fig)


# -- counterfactual replay ----------------------------------------------------

def replay(policies, out_dir, cap, real_clouds_path):
    import torch
    import torch.nn as nn
    import sys
    sys.argv = ["compare_evals", "--train_only", "x"]
    sys.path.insert(0, os.path.join(REPO, "scripts", "envs"))
    import train_bc_pointcloud as tbc

    with_dec = [p for p in policies if p["dec"] is not None and p["ckpt"]]
    if len(with_dec) < 2:
        print("[replay] need >= 2 policies with decisions + checkpoints, skipping")
        return

    # union of logged sim clouds (subsampled) + their decode bounds
    clouds, bmin, bmax = [], None, None
    for p in with_dec:
        clouds.append(p["dec"]["points"])
        if "bounds_min" in p["dec"].files:
            bmin = p["dec"]["bounds_min"][0]
            bmax = p["dec"]["bounds_max"][0]
    clouds = np.concatenate(clouds)
    rng = np.random.default_rng(0)
    if len(clouds) > cap:
        clouds = clouds[rng.choice(len(clouds), cap, replace=False)]
    if bmin is None:
        bmin = np.array([-5.364, -1.684, -1.30], dtype=np.float32)
        bmax = np.array([-3.364, 5.316, 0.10], dtype=np.float32)
        print("[replay] no bounds in npz, using default rack crop bounds")

    def load_net(ckpt_path):
        state = tbc.load_init_state(ckpt_path if os.path.isabs(ckpt_path)
                                    else os.path.join(REPO, ckpt_path))
        net = tbc.BCPointNetPolicy(num_points=1024, action_dim=5)
        net.load_state_dict(state)
        net.eval()
        return net

    def decode(raw):
        xyz = bmin[:3] + (np.tanh(raw[:, :3]) + 1) / 2 * (bmax[:3] - bmin[:3])
        yaw = np.arctan2(np.tanh(raw[:, 4]), np.tanh(raw[:, 3])) / 2.0
        return np.concatenate([xyz, yaw[:, None]], axis=1)

    def run_all(cloud_arr):
        res = {}
        x = torch.from_numpy(cloud_arr.reshape(len(cloud_arr), -1)).float()
        for p in with_dec:
            ck = p["ckpt"]
            ck = ck.replace("/workspace/crane_testbed", REPO)
            net = load_net(ck)
            outs = []
            with torch.no_grad():
                for i in range(0, len(x), 256):
                    outs.append(net(x[i:i + 256]).numpy())
            res[p["label"]] = decode(np.concatenate(outs))
        return res

    sim_dec = run_all(clouds)
    labels = list(sim_dec.keys())

    # divergence matrices: mean planar distance and mean |dz| on identical clouds
    n = len(labels)
    dxy = np.zeros((n, n))
    dz = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a, b = sim_dec[labels[i]], sim_dec[labels[j]]
            dxy[i, j] = np.mean(np.hypot(a[:, 0] - b[:, 0], a[:, 1] - b[:, 1]))
            dz[i, j] = np.mean(a[:, 2] - b[:, 2])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, mat, title in [(axes[0], dxy, "mean planar distance [m]"),
                           (axes[1], dz, "mean signed delta z [m] (row - col)")]:
        im = ax.imshow(mat, cmap="coolwarm" if "delta" in title else "viridis")
        ax.set_xticks(range(n), labels, rotation=30, fontsize=7)
        ax.set_yticks(range(n), labels, fontsize=7)
        ax.set_title(title, fontsize=9)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "decision_divergence.png"), dpi=200)
    plt.close(fig)

    # input-conditional correction test: dz on sim clouds vs real clouds
    if real_clouds_path:
        real = np.load(real_clouds_path)
        if len(real) > cap:
            real = real[rng.choice(len(real), cap, replace=False)]
        real_dec = run_all(real)
        pairs = [(a, b) for (a, b) in [("BCRL+SFT", "BCRL"), ("BC+SFT", "BC")]
                 if a in sim_dec and b in sim_dec]
        if pairs:
            fig, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 3.2), squeeze=False)
            for ax, (a, b) in zip(axes.flat, pairs):
                ax.hist(sim_dec[a][:, 2] - sim_dec[b][:, 2], bins=40, alpha=0.6,
                        label="sim clouds", density=True)
                ax.hist(real_dec[a][:, 2] - real_dec[b][:, 2], bins=40, alpha=0.6,
                        label="real clouds", density=True)
                ax.axvline(0, color="gray", linewidth=0.6)
                ax.set_title(f"{a} minus {b}: delta z", fontsize=9)
                ax.set_xlabel("delta z [m]")
                ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "dz_sim_vs_real.png"), dpi=200)
            plt.close(fig)
    print(f"[replay] done on {len(clouds)} sim clouds"
          + (f" + real clouds" if real_clouds_path else ""))


def main():
    args = parse_args()
    pol_spec = DEFAULT_POLICIES if not args.policies else \
        [tuple(s.split("=", 1)) for s in args.policies]
    policies = [p for p in (load_policy_data(l, d) for l, d in pol_spec) if p]
    if not policies:
        raise SystemExit("no policy data found")
    out_dir = args.out or os.path.join(REPO, "logs", "eval_analysis")
    os.makedirs(out_dir, exist_ok=True)

    write_summary(policies, out_dir)
    paired_stats(policies, out_dir)
    fig_stability_paired(policies, out_dir)
    fig_clearing(policies, out_dir)
    fig_dig_depth(policies, out_dir)
    fig_target_heatmaps(policies, out_dir)
    fig_depletion(policies, out_dir)
    if args.replay:
        replay(policies, out_dir, args.replay_cap, args.real_clouds)
    print(f"\n[compare_evals] outputs in {out_dir}")


if __name__ == "__main__":
    main()
