#!/usr/bin/env python3
"""Interactive 3D episode viewer for eval decision logs (self-contained plotly HTML).

Reads a decisions_<ts>.npz written by play_bc_pointcloud.py --save_decisions and renders
one episode of one env as a rotatable 3D scene with a cycle slider: the policy-input
cloud at the selected cycle, all grasp targets up to that cycle (color = cycle order,
hover = cycle/logs/alignment/stability/reward), the current target highlighted, and a
short segment showing the commanded yaw.

Usage:
  python3 plot_episode_3d.py --decisions <path/decisions_*.npz> [--env 0] [--episode 0]
                             [--out <out.html>]
"""

import argparse
import os

import numpy as np
import plotly.graph_objects as go


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True, help="decisions_*.npz from --save_decisions")
    ap.add_argument("--env", type=int, default=0)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out", default=None, help="output HTML (default: alongside npz)")
    return ap.parse_args()


def yaw_segment(x, y, z, yaw, half_len=0.35):
    """Endpoints of a short horizontal segment through the target along the grasp yaw."""
    dx, dy = np.cos(yaw) * half_len, np.sin(yaw) * half_len
    return [x - dx, x + dx], [y - dy, y + dy], [z, z]


def main():
    args = parse_args()
    d = np.load(args.decisions, allow_pickle=True)
    sel = (d["env"] == args.env) & (d["episode"] == args.episode)
    if not sel.any():
        avail = sorted({(int(e), int(p)) for e, p in zip(d["env"], d["episode"])})
        raise SystemExit(f"no records for env={args.env} episode={args.episode}; available (env, episode): {avail}")

    order = np.argsort(d["cycle"][sel])
    pts = d["points"][sel][order]          # (C, N, 3)
    tgt = d["target"][sel][order]          # (C, 4) x y z yaw
    logs = d["logs_grasped"][sel][order]
    align = d["alignment"][sel][order]
    stab = d["stability"][sel][order]
    rew = d["reward"][sel][order]
    cycles = d["cycle"][sel][order]
    n_cyc = len(cycles)

    hover = [f"cycle {int(c)}: {int(l)} logs, align={a:.2f}, stab={s:.2f}, rew={r:.2f}"
             for c, l, a, s, r in zip(cycles, logs, align, stab, rew)]
    tgt_colors = np.arange(n_cyc)

    def cloud_trace(k):
        p = pts[k]
        p = p[np.any(p != 0.0, axis=1)]  # drop zero padding
        return go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                            marker=dict(size=1.5, color=p[:, 2], colorscale="Greys_r",
                                        opacity=0.6),
                            name="cloud", hoverinfo="skip")

    def targets_trace(k):
        return go.Scatter3d(x=tgt[:k + 1, 0], y=tgt[:k + 1, 1], z=tgt[:k + 1, 2],
                            mode="markers",
                            marker=dict(size=4, symbol="diamond", color=tgt_colors[:k + 1],
                                        colorscale="Turbo", cmin=0, cmax=max(1, n_cyc - 1)),
                            text=hover[:k + 1], hoverinfo="text", name="targets")

    def current_trace(k):
        sx, sy, sz = yaw_segment(*tgt[k])
        return go.Scatter3d(x=sx, y=sy, z=sz, mode="lines+markers",
                            line=dict(color="red", width=6),
                            marker=dict(size=6, color="red", symbol="x"),
                            text=[hover[k]] * 2, hoverinfo="text", name="current grasp")

    frames = [go.Frame(data=[cloud_trace(k), targets_trace(k), current_trace(k)],
                       name=str(int(cycles[k]))) for k in range(n_cyc)]

    fig = go.Figure(data=frames[0].data, frames=frames)
    steps = [dict(method="animate", label=str(int(cycles[k])),
                  args=[[str(int(cycles[k]))],
                        dict(mode="immediate", frame=dict(duration=0, redraw=True),
                             transition=dict(duration=0))])
             for k in range(n_cyc)]
    src = os.path.basename(os.path.dirname(os.path.abspath(args.decisions)))
    fig.update_layout(
        title=f"{src} | env {args.env} episode {args.episode} ({n_cyc} grasp cycles)",
        scene=dict(aspectmode="data",
                   xaxis_title="base x [m]", yaxis_title="base y [m]", zaxis_title="base z [m]"),
        sliders=[dict(steps=steps, currentvalue=dict(prefix="cycle "))],
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.decisions)),
        f"episode3d_env{args.env}_ep{args.episode}.html")
    fig.write_html(out, include_plotlyjs=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
