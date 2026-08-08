#!/usr/bin/env python3
"""Contact-sheet of recorded episodes: one row per policy, one column per PILE STATE.

Columns are chosen by logs-remaining (200 / 150 / 100 / 50 / last), NOT by time, because the
policies take different numbers of cycles (17 vs 25 vs 30) and comparing them at equal
wall-clock would put them at different stages of the job. Aligning on pile state answers the
question you actually want: at the same point in clearing, what does each policy do?

Frames are cropped to a COMMON content box computed across every panel (saturation-based: the
floor and grid lines are grey, the crane is yellow and the logs are tan), so all panels stay at
one scale and the crop is comparable row to row.

    python3 scripts/envs/render_video_collage.py \
        --media logs/sim_eval/paired_video/media \
        --runs  logs/sim_eval/paired_video/tm_expert ... \
        --out   docs/figures/tm_collage.png
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import imageio.v2 as imageio

_FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]


def _font(sz):
    for p in _FONTS:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def cycles_of(run_dir, env_id=0):
    f = sorted(glob.glob(os.path.join(run_dir, "decisions_*.npz")))
    if not f:
        return None
    d = np.load(f[-1], allow_pickle=True)
    m = d["env"] == env_id
    o = np.argsort(d["cycle"][m])
    rack = d["logs_in_rack"][m][o].astype(int)
    grasped = d["logs_grasped"][m][o].astype(int)
    bounds = d["video_cycle_bounds"].tolist() if "video_cycle_bounds" in d.files else []
    return rack, grasped, bounds


def pick_frames(rack, bounds, targets):
    """For each target logs-remaining, the mid-frame of the first cycle at or below it."""
    picks = []
    for t in targets:
        if t is None:                                  # last cycle
            c = len(rack) - 1
        else:
            idx = np.where(rack <= t)[0]
            c = int(idx[0]) if len(idx) else len(rack) - 1
        c = min(c, len(bounds) - 1)
        lo = bounds[c - 1] if c > 0 else 0
        hi = bounds[c]
        picks.append((c, (lo + hi) // 2 if hi > lo else lo))
    return picks


def grab(video, frame_ids):
    """One sequential pass - seeking a 6k-frame mp4 per panel is far slower."""
    want = sorted(set(frame_ids))
    out, r = {}, imageio.get_reader(video)
    wi = 0
    for i, f in enumerate(r):
        while wi < len(want) and want[wi] == i:
            out[i] = np.asarray(f)
            wi += 1
        if wi >= len(want):
            break
    r.close()
    return out


def content_box(frames, pad=45):
    """Common bbox of saturated (non-floor) pixels across all panels."""
    y0, y1, x0, x1 = 10**9, -1, 10**9, -1
    for f in frames:
        a = f.astype(np.int16)
        sat = a.max(axis=2) - a.min(axis=2)
        ys, xs = np.where(sat > 28)
        if len(ys):
            y0, y1 = min(y0, ys.min()), max(y1, ys.max())
            x0, x1 = min(x0, xs.min()), max(x1, xs.max())
    if y1 < 0:
        return None
    H, W = frames[0].shape[:2]
    return (max(0, x0 - pad), max(0, y0 - pad),
            min(W, x1 + pad), min(H, y1 + pad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--suffix", default="_overview.mp4")
    ap.add_argument("--targets", nargs="+", default=["200", "150", "100", "50", "last"])
    ap.add_argument("--panel_w", type=int, default=560)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    targets = [None if t == "last" else int(t) for t in a.targets]
    rows = []
    for rd in a.runs:
        tag = os.path.basename(rd.rstrip("/"))
        info = cycles_of(rd)
        if info is None:
            print(f"skip {tag}: no decisions"); continue
        rack, grasped, bounds = info
        if not bounds:
            print(f"skip {tag}: no video_cycle_bounds"); continue
        vid = os.path.join(a.media, tag + a.suffix)
        if not os.path.exists(vid):
            print(f"skip {tag}: no {vid}"); continue
        picks = pick_frames(rack, bounds, targets)
        got = grab(vid, [f for _, f in picks])
        panels = []
        for (c, fi) in picks:
            img = got.get(fi)
            if img is None:
                img = np.full((720, 1280, 3), 255, np.uint8)
            panels.append((img, c, int(rack[c]), int(grasped[c])))
        rows.append((tag, panels, len(rack)))
        print(f"{tag}: cycles={len(rack)} picks={[(c, f) for c, f in picks]}")

    if not rows:
        raise SystemExit("nothing to render")

    box = content_box([p[0] for _, ps, _ in rows for p in ps])
    print("common crop box:", box)

    fs = max(15, a.panel_w // 34)
    f_lab, f_hdr = _font(fs), _font(int(fs * 1.25))
    lab_h, hdr_h, row_lab_w = int(fs * 1.9), int(fs * 2.2), int(a.panel_w * 0.30)

    tiles = []
    for tag, panels, ncyc in rows:
        strip = []
        for img, c, rk, g in panels:
            im = Image.fromarray(img)
            if box:
                im = im.crop(box)
            w = a.panel_w
            h = int(im.height * w / im.width)
            im = im.resize((w, h), Image.LANCZOS)
            can = Image.new("RGB", (w, h + lab_h), (255, 255, 255))
            can.paste(im, (0, lab_h))
            d = ImageDraw.Draw(can)
            d.text((6, 4), f"cycle {c + 1}/{ncyc}   in rack {rk}   grasped {g}",
                   fill=(20, 20, 20), font=f_lab)
            strip.append(np.asarray(can))
        tiles.append((tag, np.concatenate(strip, axis=1), ncyc))

    rw = max(t.shape[1] for _, t, _ in tiles)
    full = []
    for tag, t, ncyc in tiles:
        band = Image.new("RGB", (rw + row_lab_w, t.shape[0]), (245, 245, 245))
        band.paste(Image.fromarray(t), (row_lab_w, 0))
        d = ImageDraw.Draw(band)
        d.text((10, t.shape[0] // 2 - fs), tag.replace("tm_", ""), fill=(0, 0, 0), font=f_hdr)
        d.text((10, t.shape[0] // 2 + fs // 2), f"{ncyc} cycles", fill=(90, 90, 90), font=f_lab)
        full.append(np.asarray(band))

    hdr = Image.new("RGB", (rw + row_lab_w, hdr_h), (255, 255, 255))
    d = ImageDraw.Draw(hdr)
    d.text((10, 4), "columns = same PILE STATE (logs remaining), not same time",
           fill=(0, 0, 0), font=f_hdr)
    out = np.concatenate([np.asarray(hdr)] + full, axis=0)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    Image.fromarray(out).save(a.out)
    print(f"wrote {a.out}  ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
