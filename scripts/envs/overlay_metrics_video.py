#!/usr/bin/env python3
"""Burn the per-cycle metric accounting onto a recorded episode video.

Purpose: let a human WATCH a grasp and check the counters against what the grapple actually
did. This is the visual counterpart to the correction in summarize_eval_rows.py, where
`logs_grasped` (proximity-derived) was found to over-count what leaves the rack by 24-26%.
The overlay shows both numbers side by side every cycle, and flags the disagreement:

    REPORTED  = logs_grasped          (what the metric counts as a success)
    MEASURED  = drop in logs_in_rack  (what actually left the rack)
    TRUE      = MEASURED - knocked_off  (removed by the grapple, not knocked out of bounds)

A cycle where REPORTED > 0 but TRUE == 0 is a phantom success, drawn in red. Those are exactly
the cycles that inflate the reported success rate.

Frame->cycle mapping comes from `video_cycle_bounds` in the decisions npz (frame index at the
end of each cycle), recorded by the env because cycle length varies with FSM duration and
cannot be recovered from fps.

    python3 scripts/envs/overlay_metrics_video.py \
        --video  logs/sim_eval/paired_video/media/tm_bc_reg_overview.mp4 \
        --decisions logs/sim_eval/paired_video/tm_bc_reg \
        --out    docs/figures/tm_bc_reg_annotated.mp4
"""

from __future__ import annotations

import argparse
import bisect
import glob
import os

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    raise SystemExit("needs pillow:  pip install pillow")
try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover
    raise SystemExit("needs imageio + imageio-ffmpeg:  pip install imageio imageio-ffmpeg")

_FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]


def _font(size):
    for p in _FONTS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def load_cycles(dec_path, env_id=0):
    """Per-cycle rows for one env, in cycle order, plus the frame bounds and fps."""
    if os.path.isdir(dec_path):
        f = sorted(glob.glob(os.path.join(dec_path, "decisions_*.npz")))
        if not f:
            raise SystemExit(f"no decisions_*.npz under {dec_path}")
        dec_path = f[-1]
    d = np.load(dec_path, allow_pickle=True)
    m = d["env"] == env_id
    order = np.argsort(d["cycle"][m])
    rows = []
    prev_rack = None
    for j in order:
        g = int(d["logs_grasped"][m][j])
        rack = int(d["logs_in_rack"][m][j])
        ko = int(d["knocked_off"][m][j])
        # cycle i removed (prev_rack - rack); knocked-off is not a grasp
        measured = None if prev_rack is None else (prev_rack - rack)
        true_rm = None if measured is None else max(0, measured - ko)
        rows.append({"cycle": int(d["cycle"][m][j]), "grasped": g, "rack": rack,
                     "knocked": ko, "measured": measured, "true": true_rm})
        prev_rack = rack
    bounds = d["video_cycle_bounds"].tolist() if "video_cycle_bounds" in d.files else []
    fps = int(d["video_fps"]) if "video_fps" in d.files else 30
    return rows, bounds, fps


def draw(frame, row, totals, cyc_idx, n_cyc):
    im = Image.fromarray(frame).convert("RGB")
    dr = ImageDraw.Draw(im, "RGBA")
    W = im.width
    fs = max(13, W // 62)
    f, fb = _font(fs), _font(int(fs * 1.15))

    phantom = row is not None and row["true"] is not None and row["grasped"] > 0 and row["true"] == 0
    lines = [
        ("CYCLE %d / %d" % (cyc_idx + 1, n_cyc), (255, 255, 255)),
    ]
    if row is None:
        lines.append(("(pre-first-cycle)", (200, 200, 200)))
    else:
        meas = "--" if row["measured"] is None else str(row["measured"])
        true = "--" if row["true"] is None else str(row["true"])
        lines += [
            ("REPORTED grasped : %d" % row["grasped"], (120, 200, 255)),
            ("MEASURED removed : %s" % meas, (255, 255, 255)),
            ("knocked off      : %d" % row["knocked"], (255, 190, 120)),
            ("TRUE removed     : %s" % true, (140, 255, 160)),
            ("logs in rack     : %d" % row["rack"], (220, 220, 220)),
        ]
        if phantom:
            lines.append(("PHANTOM SUCCESS (counted, nothing left rack)", (255, 90, 90)))
    lines.append(("", (0, 0, 0)))
    lines.append(("running  reported succ %d   TRUE succ %d"
                  % (totals["rep_s"], totals["true_s"]), (200, 200, 200)))
    lines.append(("         reported logs %d   TRUE logs %d"
                  % (totals["rep_g"], totals["true_g"]), (200, 200, 200)))

    pad, lh = int(fs * 0.7), int(fs * 1.45)
    bw = int(max(dr.textlength(t, font=fb) for t, _ in lines if t) + 2 * pad)
    bh = lh * len(lines) + 2 * pad
    dr.rectangle([0, 0, bw, bh], fill=(0, 0, 0, 165))
    if phantom:
        dr.rectangle([0, 0, bw, bh], outline=(255, 60, 60), width=3)
    y = pad
    for t, col in lines:
        if t:
            dr.text((pad, y), t, fill=col, font=fb if t.startswith("CYCLE") else f)
        y += lh
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--decisions", required=True, help="decisions npz, or the dir holding it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--env_id", type=int, default=0)
    a = ap.parse_args()

    rows, bounds, fps = load_cycles(a.decisions, a.env_id)
    if not bounds:
        raise SystemExit("decisions npz has no video_cycle_bounds - the run predates the "
                         "instrumentation, or was not recorded with --record_video")
    print(f"{len(rows)} cycles, {len(bounds)} frame bounds, fps={fps}")
    if len(bounds) < len(rows):
        print(f"  note: fewer bounds than cycles; trailing cycles map to the last segment")

    rd = imageio.get_reader(a.video)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    wr = imageio.get_writer(a.out, fps=fps, codec="libx264", quality=8,
                            macro_block_size=None)
    totals = {"rep_s": 0, "true_s": 0, "rep_g": 0, "true_g": 0}
    counted = -1
    n = 0
    for i, frame in enumerate(rd):
        # frames in [bounds[c-1], bounds[c]) belong to cycle c
        c = bisect.bisect_right(bounds, i)
        c = min(c, len(rows) - 1) if rows else 0
        # accumulate each cycle's outcome once, as soon as its segment starts
        while counted < c:
            counted += 1
            r = rows[counted]
            if r["true"] is not None:
                totals["rep_s"] += 1 if r["grasped"] > 0 else 0
                totals["true_s"] += 1 if r["true"] > 0 else 0
                totals["rep_g"] += r["grasped"]
                totals["true_g"] += r["true"]
        wr.append_data(draw(frame, rows[c] if rows else None, totals, c, len(rows)))
        n += 1
    rd.close(); wr.close()
    print(f"wrote {a.out} ({n} frames)")
    rep_g, true_g = totals["rep_g"], totals["true_g"]
    if true_g:
        print(f"episode: reported {rep_g} logs vs TRUE {true_g} "
              f"({100.0 * (rep_g - true_g) / true_g:+.0f}%)")


if __name__ == "__main__":
    main()
