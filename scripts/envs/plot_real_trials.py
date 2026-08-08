#!/usr/bin/env python3
"""Real-trial analysis plots from real_trials.ods + policy_debug decisions.jsonl.

Run from the crane_testbed root (HOST, needs matplotlib):
    python3 scripts/envs/plot_real_trials.py
Outputs PNGs + real_trials_analysis.pdf into docs/figures/real_trials/.
The Open3D 3D target renders come from render_trial_targets_3d.py and are
embedded as pages if already present.
"""

import zipfile, re, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = "/home/george/IsaacLab/crane_testbed/docs/figures/real_trials"
LOGS = "/home/george/IsaacLab/crane_testbed/logs"

# ---------------- parse ODS ----------------
z = zipfile.ZipFile("/home/george/Documents/real_trials.ods")
xml = z.read("content.xml").decode()
tbl = re.search(r'<table:table table:name="Sheet1".*?</table:table>', xml, re.S).group(0)
rows = re.findall(r'<table:table-row[^>]*>(.*?)</table:table-row>', tbl, re.S)

def parse_row(r):
    cells = []
    for m in re.finditer(r'<table:table-cell([^>]*)/>|<table:table-cell([^>]*)>(.*?)</table:table-cell>', r, re.S):
        attrs = m.group(1) if m.group(1) is not None else m.group(2)
        content = m.group(3) or ""
        rep = re.search(r'table:number-columns-repeated="(\d+)"', attrs)
        n = int(rep.group(1)) if rep else 1
        val = re.search(r'office:value="([^"]+)"', attrs)
        text = " ".join(re.findall(r'<text:p[^>]*>(.*?)</text:p>', content, re.S))
        text = re.sub(r'<[^>]+>', '', text).strip()
        v = val.group(1) if val else (text if text else None)
        cells.extend([v] * min(n, 30))
    return cells

grid = [parse_row(r) for r in rows]
maxc = max(len(r) for r in grid)
for r in grid:
    r.extend([None] * (maxc - len(r)))

def get_trial(col, r0, r1):
    out = []
    for i in range(r0, r1 + 1):
        logs = grid[i][col]
        if logs is None:
            break
        try:
            logs = int(float(logs))
        except (TypeError, ValueError):
            break
        align = grid[i][col + 1]
        try:
            align = float(align)
        except (TypeError, ValueError):
            align = None
        note = grid[i][col + 2] or (grid[i][col + 1] if align is None else None)
        note = note if isinstance(note, str) else None
        out.append({"logs": logs, "align": align, "note": note})
    return out

trials = {
    "Baseline double":  get_trial(0, 1, 18),
    "Baseline jagged":  get_trial(3, 1, 36),
    "Baseline single":  get_trial(7, 1, 19),
    "Scoring double":   get_trial(0, 41, 70),
    "Scoring single":   get_trial(7, 41, 63),
}
for name, t in trials.items():
    print(name, len(t), "cycles,", sum(c["logs"] for c in t), "logs")

def classify(c):
    n = (c["note"] or "").lower()
    if c["logs"] > 0:
        return "success"
    if "chok" in n:
        return "choke"
    if "targeted" in n or "noise" in n:
        return "structure/noise"
    if "yaw" in n or "misalign" in n:
        return "yaw/control"
    return "other"

CLS_COLORS = {"success": "#2a9d3a", "structure/noise": "#d62728",
              "choke": "#ff9d00", "yaw/control": "#7b52c9", "other": "#8c6d31"}

# ---------------- decisions ----------------
def load_gaze(p):
    recs = [json.loads(l) for l in open(p)]
    return [r for r in recs if r["kind"] == "gaze"]

D = {
 "Baseline double":  load_gaze(f"{LOGS}/crane_policy_debug/run_20260803_155559/decisions.jsonl"),
 "Baseline jagged":  load_gaze(f"{LOGS}/crane_policy_debug/run_20260803_174017/decisions.jsonl"),
 "Baseline single":  load_gaze(f"{LOGS}/crane_policy_debug/run_20260803_204034/decisions.jsonl"),
 "Scoring single":   load_gaze(f"{LOGS}/bc_pointcloud/scoring_margin05_2048_c/policy_debug/run_20260805_153036/decisions.jsonl"),
 "Scoring double":   load_gaze(f"{LOGS}/bc_pointcloud/scoring_margin05_2048_c/policy_debug/run_20260805_194008/decisions.jsonl"),
}
NPZ = {
 "Baseline single": f"{LOGS}/crane_policy_debug/run_20260803_204034/policy_debug_001.npz",
 "Scoring double":  f"{LOGS}/bc_pointcloud/scoring_margin05_2048_c/policy_debug/run_20260805_194008/policy_debug_001.npz",
}

pdf = PdfPages(f"{OUT}/real_trials_analysis.pdf")
def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=160, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

ORDER = ["Baseline double", "Baseline jagged", "Baseline single",
         "Scoring single", "Scoring double"]
TCOLORS = {"Baseline double": "#1f77b4", "Baseline jagged": "#17becf",
           "Baseline single": "#4c72b0", "Scoring single": "#e07b39",
           "Scoring double": "#d62728"}

# ---------------- F1 clearing curves ----------------
fig, ax = plt.subplots(figsize=(8, 5))
for name in ORDER:
    t = trials[name]
    cum = np.cumsum([c["logs"] for c in t])
    x = np.arange(1, len(t) + 1)
    ls = "--" if name.startswith("Baseline") else "-"
    ax.plot(x, cum, ls, color=TCOLORS[name], label=f"{name} ({cum[-1]} logs / {len(t)} cy)", lw=2)
    empty = [i + 1 for i, c in enumerate(t) if c["logs"] == 0]
    ax.plot(empty, cum[np.array(empty) - 1], "x", color=TCOLORS[name], ms=7, mew=2)
ax.set_xlabel("cycle"); ax.set_ylabel("cumulative logs moved")
ax.set_title("Clearing curves, real trials (x = empty cycle)")
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
save(fig, "clearing_curves_real")

# ---------------- F2 outcome timelines ----------------
fig, axes = plt.subplots(len(ORDER), 1, figsize=(9, 9), sharex=True)
for ax, name in zip(axes, ORDER):
    t = trials[name]
    for i, c in enumerate(t):
        cls = classify(c)
        h = c["logs"] if c["logs"] > 0 else -3
        ax.bar(i + 1, h, color=CLS_COLORS[cls], width=0.85)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel(name.replace(" ", "\n"), fontsize=8)
    ax.set_ylim(-4, 26); ax.grid(alpha=0.25, axis="y")
axes[-1].set_xlabel("cycle")
handles = [plt.Rectangle((0, 0), 1, 1, color=CLS_COLORS[k]) for k in CLS_COLORS]
axes[0].legend(handles, list(CLS_COLORS), fontsize=7, ncol=5, loc="upper right")
axes[0].set_title("Per-cycle outcomes (bar = logs grasped; below axis = empty cycle by class)")
save(fig, "outcome_timelines")

# ---------------- F3 empty-cycle taxonomy by trial progress ----------------
fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
for ax, (title, names) in zip(axes, [("Baseline (3 trials, 73 cycles)", ORDER[:3]),
                                     ("Scoring (2 trials, 53 cycles)", ORDER[3:])]):
    terc = {0: {}, 1: {}, 2: {}}
    for name in names:
        t = trials[name]
        for i, c in enumerate(t):
            if c["logs"] > 0:
                continue
            k = min(2, int(3 * i / len(t)))
            cls = classify(c)
            terc[k][cls] = terc[k].get(cls, 0) + 1
    bottoms = np.zeros(3)
    for cls in ["structure/noise", "choke", "yaw/control", "other"]:
        vals = [terc[k].get(cls, 0) for k in range(3)]
        ax.bar(range(3), vals, bottom=bottoms, color=CLS_COLORS[cls], label=cls)
        bottoms += vals
    ax.set_xticks(range(3)); ax.set_xticklabels(["early third", "mid third", "final third"])
    ax.set_title(title, fontsize=10); ax.grid(alpha=0.25, axis="y")
axes[0].set_ylabel("empty cycles"); axes[1].legend(fontsize=8)
fig.suptitle("Empty-cycle causes by trial progress", y=1.02)
save(fig, "failure_taxonomy_terciles")

# ---------------- F4 histograms ----------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
bl = [c["logs"] for n in ORDER[:3] for c in trials[n] if c["logs"] > 0]
sc = [c["logs"] for n in ORDER[3:] for c in trials[n] if c["logs"] > 0]
def grouped_hist(ax, a, b, bins, la, lb):
    ha, _ = np.histogram(a, bins=bins); hb, _ = np.histogram(b, bins=bins)
    ctr = 0.5 * (bins[:-1] + bins[1:]); w = 0.4 * (bins[1] - bins[0])
    ax.bar(ctr - w / 2, ha, width=w, color="#4c72b0", label=la)
    ax.bar(ctr + w / 2, hb, width=w, color="#d62728", label=lb)
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
grouped_hist(axes[0], bl, sc, np.arange(0.5, 25.5, 2),
             f"baseline (med {np.median(bl):.0f})", f"scoring (med {np.median(sc):.0f})")
axes[0].set_xlabel("logs per successful cycle"); axes[0].set_ylabel("cycles")
bla = [c["align"] for n in ORDER[:3] for c in trials[n] if c["align"] is not None]
sca = [c["align"] for n in ORDER[3:] for c in trials[n] if c["align"] is not None]
grouped_hist(axes[1], bla, sca, np.arange(0.5, 6.5, 1),
             f"baseline (mean {np.mean(bla):.2f})", f"scoring (mean {np.mean(sca):.2f})")
axes[1].set_xlabel("operator alignment score (1-5)")
fig.suptitle("Grasp quality distributions")
save(fig, "quality_histograms")

# ---------------- F5 support (scoring runs) ----------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, name in zip(axes, ["Scoring single", "Scoring double"]):
    g = D[name]
    cyc = [r["cycle"] for r in g]; sup = [r.get("support", np.nan) for r in g]
    ax.plot(cyc, sup, "o-", color=TCOLORS[name], ms=4)
    ax.axhline(20, color="r", ls="--", lw=1, label="support 20 (warn level)")
    ax.set_title(f"{name}: target support per gaze", fontsize=10)
    ax.set_xlabel("gaze cycle"); ax.set_ylabel("points within 0.5 m")
    ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_ylim(0, None)
save(fig, "support_per_cycle")

# ---------------- F6 target y map ----------------
fig, axes = plt.subplots(len(ORDER), 1, figsize=(9, 10), sharex=False)
for ax, name in zip(axes, ORDER):
    g = D[name]
    ys = [r["target"][1] for r in g]; cyc = [r["cycle"] for r in g]
    d0 = np.load({**{n: f"{LOGS}/crane_policy_debug/run_20260803_155559/policy_debug_001.npz" for n in [ORDER[0]]},
                  **{ORDER[1]: f"{LOGS}/crane_policy_debug/run_20260803_174017/policy_debug_001.npz",
                     ORDER[2]: f"{LOGS}/crane_policy_debug/run_20260803_204034/policy_debug_001.npz",
                     ORDER[3]: f"{LOGS}/bc_pointcloud/scoring_margin05_2048_c/policy_debug/run_20260805_153036/policy_debug_001.npz",
                     ORDER[4]: f"{LOGS}/bc_pointcloud/scoring_margin05_2048_c/policy_debug/run_20260805_194008/policy_debug_001.npz"}}[name])
    sh = float(d0["rack_y_shift"]); bmin = d0["bounds_min"]; bmax = d0["bounds_max"]
    ax.axhline(bmin[1], color="k", ls=":", lw=1); ax.axhline(bmax[1], color="k", ls=":", lw=1)
    ax.plot(cyc, ys, "o-", color=TCOLORS[name], ms=4, lw=1)
    ax.set_ylabel("target y [m]", fontsize=8)
    ax.set_title(f"{name} (rack_y_shift {sh:+.2f}; dotted = action box y bounds)", fontsize=9)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel("gaze cycle")
fig.suptitle("Commanded target y over each trial", y=1.005)
fig.tight_layout()
save(fig, "target_y_maps")

# ---------------- F7/F8 Open3D renders (rendered by render_trial_targets_3d.py) ----------------
import matplotlib.image as mpimg
for fn, cap in [("targets_3d.png", "Commanded targets over the recorded clouds (Open3D)"),
                ("targets_3d_phases.png", "First third vs final third: targets migrate to the rack end")]:
    path = f"{OUT}/{fn}"
    if not os.path.exists(path):
        continue
    img = mpimg.imread(path)
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(11, 11 * h / w))
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img); ax.axis("off")
    ax.set_title(cap, fontsize=9)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

pdf.close()
print("wrote", OUT)
