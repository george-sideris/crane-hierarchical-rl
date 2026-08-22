#!/usr/bin/env python3
"""Clearing progression for the six policies of the headline comparison.

Reads the per-episode clearing curves out of the deep-protocol eval JSONs
(20 envs x 100 episodes, seed 42, repaired environment). Every policy saw the
same 100 pile seeds, so the curves are paired episode by episode and not merely
drawn from a common distribution.

Panel (a) is the mean fraction of the pile cleared against grasp cycle: the
bulk phase, where the policies are hard to tell apart. Panel (b) is the
fraction of episodes finished by a given cycle, which is where they separate;
its right-hand end reproduces the full-clear column of the results table
exactly, so the two panels are the same measurement read two ways.

An episode's curve stops when the pile clears or when the 30-grasp budget runs
out. A cleared pile stays cleared, so curves are held forward at their last
value to the budget before averaging; otherwise the mean would silently drop
policies out of the average exactly when they succeed.

Writes docs/thesis/figures/fig_clearing_curves.pdf.
"""
import json
import os
import glob
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEEP = os.path.join(ROOT, "logs", "eval_archive", "deep")
OUT = os.path.join(ROOT, "docs", "thesis", "figures", "fig_clearing_curves.pdf")

HORIZON = 30  # grasp budget H

# Colour order validated for adjacent-pair CVD separation on a light surface.
# Every line is also direct-labelled and dash-coded, which is the secondary
# encoding the aqua/red pair and the low-contrast aqua need.
SERIES = [
    ("Privileged expert",   "expert_100ep_fixedenv",     "#4a3aa7", (0, (5, 2))),
    ("Geometric heuristic", "heuristic_100ep_fixedenv",  "#008300", (0, (5, 2))),
    ("BC, regression head", "regression_100ep_fixedenv", "#e34948", (0, (1, 1.6))),
    ("RL (scratch)",        "scc0_30_100ep_fixedenv",    "#1baf7a", (0, (4, 1.3, 1, 1.3))),
    ("BC, scoring head",    "bc_100ep_fixedenv",         "#eb6834", "solid"),
    (r"BC$\rightarrow$RL",  "g100_100ep_fixedenv",       "#2a78d6", "solid"),
]

INK = "#1a1a19"
MUTED = "#6b6a66"


def curves(tag):
    matches = glob.glob(os.path.join(DEEP, tag, "eval_metrics_*.json"))
    if not matches:
        raise SystemExit("no eval json for %s" % tag)
    blob = json.load(open(matches[0]))
    raw = blob["per_episode"]["clearing_curves"]

    held = []
    finish = []
    for c in raw:
        c = list(c)
        k = next((i for i, v in enumerate(c) if v >= 99.9999), None)
        finish.append(k)
        c = c + [c[-1]] * (HORIZON + 1 - len(c))
        held.append(c[: HORIZON + 1])

    cleared = [statistics.mean(col) for col in zip(*held)]
    n = len(raw)
    finished = [100.0 * sum(1 for k in finish if k is not None and k <= t) / n
                for t in range(HORIZON + 1)]
    # the completion curve must land on the reported full-clear rate
    assert abs(finished[-1] - blob["summary"]["full_clear_rate"]) < 1e-6, tag
    return cleared, finished, n


def main():
    cleared, finished = {}, {}
    n_eps = set()
    for label, tag, _, _ in SERIES:
        cleared[label], finished[label], n = curves(tag)
        n_eps.add(n)
    if len(n_eps) != 1:
        raise SystemExit("episode counts differ across policies: %s" % n_eps)

    # Sized so that \includegraphics[width=\linewidth] scales it by about 0.75;
    # the font sizes below are set to land near 8pt on the printed page.
    fig, (ax, axf) = plt.subplots(1, 2, figsize=(6.2, 2.75))
    x = list(range(HORIZON + 1))

    handles = []
    for label, _, colour, dash in SERIES:
        ax.plot(x, cleared[label], color=colour, linestyle=dash, linewidth=1.6,
                solid_capstyle="round", zorder=3)
        line, = axf.plot(x, finished[label], color=colour, linestyle=dash,
                         linewidth=1.6, solid_capstyle="round", zorder=3,
                         label=label)
        handles.append(line)

    for a in (ax, axf):
        a.set_xlim(0, HORIZON)
        a.set_ylim(0, 102)
        a.grid(True, color="#dcdbd6", linewidth=0.55, zorder=0)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color("#b8b7b2")
        a.tick_params(colors=MUTED, labelsize=9, length=3)
        a.set_xlabel("Grasp cycle", fontsize=10, color=INK)

    ax.set_ylabel("Pile cleared (\\%)", fontsize=10, color=INK)
    axf.set_ylabel("Episodes finished (\\%)", fontsize=10, color=INK)
    ax.set_title("(a) Bulk removal", fontsize=10.5, color=INK, loc="left")
    axf.set_title("(b) Completion", fontsize=10.5, color=INK, loc="left")

    # No direct labels: the completion curves all rise inside the same narrow
    # band of cycles, so any in-plot label crosses a neighbouring curve. Identity
    # comes from the legend, distinguishability from the dash coding.
    leg = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                     fontsize=9, handlelength=2.4, columnspacing=1.6,
                     handletextpad=0.6, bbox_to_anchor=(0.5, -0.015))
    for text in leg.get_texts():
        text.set_color(INK)

    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.365, top=0.9, wspace=0.3)
    fig.savefig(OUT)
    print("wrote", OUT)
    for label, _, _, _ in SERIES:
        print("  %-22s cleared %.2f  finished %.1f"
              % (label, cleared[label][HORIZON], finished[label][HORIZON]))


if __name__ == "__main__":
    main()
