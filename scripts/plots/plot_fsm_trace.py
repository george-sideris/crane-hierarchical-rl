#!/usr/bin/env python3
"""Task-space trace of one grasp cycle, phase by phase.

Chapter 3 describes the state machine in prose. This plots what it does: the grapple
reference point in the crane base frame and the grapple yaw, over one cycle, with the
phases shaded and the commanded target drawn as a dashed line. Reading the figure top to
bottom shows the hover rise, the yaw alignment at height, the descent onto the target,
the closing sweep and the lift.

    python3 scripts/plots/plot_fsm_trace.py [--trace logs/fsm_trace.npz] [--cycle 1]
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "legend.fontsize": 7,
    "mathtext.fontset": "cm",
})

# phase id -> (label, shading colour); ids from crane_rl_env_gaze.py
PHASES = {10: ("GAZE", "#dfe6ef"),
          0: ("HOVER", "#e8dfef"),
          1: ("ALIGN", "#efe6da"),
          2: ("DESCEND", "#dfefe2"),
          3: ("CLOSE", "#efdfdf"),
          4: ("LIFT", "#e4e4e4"),
          5: ("CARRY", "#dfe9ef"),
          6: ("ALIGN$_\\mathrm{h}$", "#efe9da"),
          7: ("LOWER", "#e2efdf"),
          8: ("OPEN", "#efdfe9"),
          9: ("SETTLE", "#e9e9e0")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="logs/fsm_trace.npz")
    ap.add_argument("--cycle", type=int, default=1, help="which recorded cycle to draw")
    ap.add_argument("--out", default="docs/thesis/figures/fsm_task_space.pdf")
    a = ap.parse_args()

    d = np.load(a.trace, allow_pickle=True)
    ph = d["phase"]
    # a cycle runs from one GAZE block to the next
    gaze = ph == 10
    starts = np.flatnonzero(gaze & ~np.r_[False, gaze[:-1]])
    if len(starts) > a.cycle:
        lo, hi = starts[a.cycle - 1], starts[a.cycle]
    else:
        lo, hi = 0, len(ph)
    sl = slice(lo, hi)
    rate = float(d["rate_hz"]) if "rate_hz" in d.files else 120.0
    t = (d["t"][sl] - d["t"][lo]) / rate          # physics rate -> seconds

    fig, axes = plt.subplots(5, 1, figsize=(6.4, 7.2), sharex=True)
    series = [("x", "$x$ [m]"), ("y", "$y$ [m]"), ("z", "$z$ [m]")]
    for ax, (k, lab) in zip(axes, series):
        ax.plot(t, d[k][sl], color="#22415f", lw=1.6, label="grapple")
        ax.plot(t, d["t" + k][sl], color="#c44e52", lw=1.1, ls="--", label="commanded target")
        ax.set_ylabel(lab)
    # Yaw is defined modulo 180 degrees: the policy predicts a doubled angle, and a log
    # rotated by half a turn is the same log. Both traces are wrapped into [-90, 90) so
    # that the alignment is read on the axis the policy actually controls.
    def wrap180(a_deg):
        return (a_deg + 90.0) % 180.0 - 90.0
    tgt = wrap180(np.degrees(d["tyaw"][sl]))
    # of the two equivalent representatives, draw the one nearest the commanded angle,
    # so the trace does not jump the wrap while the grapple turns smoothly
    rel = wrap180(np.degrees(d["yaw"][sl]) - np.degrees(d["tyaw"][sl]))
    axes[3].plot(t, tgt + rel, color="#22415f", lw=1.6)
    axes[3].plot(t, tgt, color="#c44e52", lw=1.1, ls="--")
    axes[3].set_ylabel(r"$\psi$ [deg]")
    # tong opening: what CLOSE actually does, and the grip held through LIFT
    axes[4].plot(t, np.degrees(d["tong"][sl]), color="#22415f", lw=1.6)
    axes[4].set_ylabel(r"tong joint [deg]")
    axes[4].set_xlabel("time within the cycle [s]")

    # shade the phases and label them once, along the top
    bounds = np.flatnonzero(np.diff(ph[sl])) + 1
    edges = np.r_[0, bounds, len(t)]
    drawn = []
    for s0, s1 in zip(edges[:-1], edges[1:]):
        pid = int(ph[sl][s0])
        if pid not in PHASES:
            continue
        name, col = PHASES[pid]
        for ax in axes:
            ax.axvspan(t[s0], t[min(s1, len(t) - 1)], color=col, lw=0, zorder=0)
        # alternate two rows so narrow phases keep readable labels
        row = 1.02 if (len(drawn) % 2 == 0) else 1.10
        axes[0].text(0.5 * (t[s0] + t[min(s1, len(t) - 1)]), row, name,
                     transform=axes[0].get_xaxis_transform(), ha="center", va="bottom",
                     fontsize=7.0, color="0.25")
        drawn.append(pid)
    for ax in axes:
        ax.grid(lw=0.4, alpha=0.35)
    axes[0].legend(frameon=False, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"wrote {a.out}  ({t[-1]:.1f} s of cycle {a.cycle}, "
          f"{len(set(ph[sl].tolist()))} phases)")


if __name__ == "__main__":
    main()
