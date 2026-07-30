#!/usr/bin/env python3
"""Stitch a crane_policy_node run into a single review video.

Input: a run directory produced by crane_policy_node's debug_save_dir (i.e.
<checkpoint_dir>/policy_debug/run_<stamp>/), containing:
    policy_debug_NNN.npz     per cycle: cropped policy-input cloud + chosen target
    basemast_run.mp4         (optional) the basemast RGB stream for the run

Output: <run_dir>/review.mp4
  - If the camera video is present: each frame = [camera | decision iso-view], with
    the decision panel switching at each cycle. NOTE: the node does not stamp video
    frames with decision times, so cycles are distributed EVENLY across the video
    timeline (a good-enough review alignment, not frame-exact).
  - If there is no camera video: a decisions-only slideshow (each decision held
    --hold seconds).

Runs on the HOST (needs imageio+ffmpeg, matplotlib, numpy, Pillow).

Usage:
    python3 render_run_video.py <run_dir> [--out review.mp4] [--fps 15] [--hold 2.0]
"""
import argparse
import bisect
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from PIL import Image, ImageDraw, ImageFont


# One shared Open3D offscreen renderer (filament engine init is expensive; reuse it).
_O3D = {"renderer": None, "ok": True}


def _o3d_panel(npz_path, height):
    """Pretty Open3D offscreen render of one cycle (same look as view_policy_debug.py):
    gray cloud, red target sphere + yaw line, blue crop box, base axes. Returns RGB array."""
    import open3d as o3d
    from open3d.visualization import rendering

    d = np.load(npz_path)
    pts = d["points"].astype(np.float64)
    tx, ty, tz, tyaw = [float(v) for v in d["target"]]
    bmin = d["bounds_min"].astype(np.float64)
    bmax = d["bounds_max"].astype(np.float64)
    if "max_z" in d.files:   # cap the overlay (box, dividers, framing) at the real fill max
        bmax[2] = float(d["max_z"][0])

    r = _O3D["renderer"]
    if r is None:
        r = rendering.OffscreenRenderer(800, 600)
        r.scene.set_background([1.0, 1.0, 1.0, 1.0])
        _O3D["renderer"] = r
    sc = r.scene
    sc.clear_geometry()

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.paint_uniform_color([0.55, 0.55, 0.6])
    sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.12)
    sph.translate([tx, ty, tz]); sph.paint_uniform_color([1, 0, 0]); sph.compute_vertex_normals()
    end = [tx + 0.7 * np.cos(tyaw), ty + 0.7 * np.sin(tyaw), tz]
    line = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector([[tx, ty, tz], end]),
                                lines=o3d.utility.Vector2iVector([[0, 1]]))
    line.paint_uniform_color([1, 0, 0])
    boxls = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        o3d.geometry.AxisAlignedBoundingBox(bmin, bmax))
    boxls.paint_uniform_color([0.1, 0.5, 1.0])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    mp = rendering.MaterialRecord(); mp.shader = "defaultUnlit"; mp.point_size = 4.0
    ml = rendering.MaterialRecord(); ml.shader = "unlitLine"; ml.line_width = 3.0
    mm = rendering.MaterialRecord(); mm.shader = "defaultLit"
    sc.add_geometry("pcd", pcd, mp); sc.add_geometry("sph", sph, mm)
    sc.add_geometry("line", line, ml); sc.add_geometry("box", boxls, ml)

    # bins-mode deposit overlay: draw the compartment dividers (orange rectangles at each pole y)
    # and outline the active bin (green), inside the trailer box. No-op for pile mode / policy panels.
    edges = d["bin_edges"].astype(np.float64) if "bin_edges" in d.files else np.zeros((0,))
    active = int(d["active_bin"][0]) if "active_bin" in d.files else -1
    if edges.size >= 2:
        dp, dl = [], []
        for ye in edges:
            b = len(dp)
            dp += [[bmin[0], ye, bmin[2]], [bmax[0], ye, bmin[2]],
                   [bmax[0], ye, bmax[2]], [bmin[0], ye, bmax[2]]]
            dl += [[b, b + 1], [b + 1, b + 2], [b + 2, b + 3], [b + 3, b]]
        divls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.array(dp, np.float64)),
            lines=o3d.utility.Vector2iVector(np.array(dl, np.int32)))
        divls.paint_uniform_color([1.0, 0.5, 0.0])
        sc.add_geometry("bins", divls, ml)
        if 0 <= active < edges.size - 1:
            abox = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
                o3d.geometry.AxisAlignedBoundingBox(
                    np.array([bmin[0], edges[active], bmin[2]]),
                    np.array([bmax[0], edges[active + 1], bmax[2]])))
            abox.paint_uniform_color([0.0, 0.8, 0.1])
            sc.add_geometry("active_bin", abox, ml)

    # frame the crop box (the scene that matters), not the point spread: fills the panel.
    # look at the pile plane (lower third of the box) so the content centers vertically
    c = 0.5 * (bmin + bmax)
    c[2] = bmin[2] + 0.3 * (bmax[2] - bmin[2])
    rad = max(float(np.linalg.norm(bmax - bmin)), 1.0)
    eye = c + np.array([1.0, -1.0, 0.6]) * (0.6 * rad)
    r.setup_camera(60.0, c, eye, [0.0, 0.0, 1.0])
    arr = np.asarray(r.render_to_image())
    # auto-crop the background margins so the inset is all content (keep a small pad);
    # background color taken from the corner (the renderer tone-maps pure white to light gray)
    bg = arr[2, 2, :3].astype(int)
    content = np.any(np.abs(arr[:, :, :3].astype(int) - bg) > 12, axis=2)
    ys, xs = np.where(content)
    if len(ys) > 100:
        pad = 12
        y0, y1 = max(0, ys.min() - pad), min(arr.shape[0], ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(arr.shape[1], xs.max() + pad)
        arr = arr[y0:y1, x0:x1]
    img = Image.fromarray(arr)

    m = re.search(r"(\d+)", os.path.basename(npz_path))
    cyc = m.group(1) if m else "?"
    try:
        from PIL import ImageDraw
        ImageDraw.Draw(img).text(
            (8, 6), f"cycle {cyc}  target ({tx:.2f}, {ty:.2f}, {tz:.2f}) yaw {np.degrees(tyaw):.0f} deg",
            fill=(0, 0, 0))
    except Exception:
        pass
    w = max(1, int(round(img.width * height / img.height)))
    return np.asarray(img.resize((w, height), Image.BILINEAR))


def _mpl_panel(npz_path, height):
    """Matplotlib fallback (used only if Open3D is unavailable / offscreen GL fails)."""
    d = np.load(npz_path)
    pts = d["points"]
    tx, ty, tz, tyaw = [float(v) for v in d["target"]]

    fig = plt.figure(figsize=(6.0, 4.5), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c=pts[:, 2], cmap="viridis")
    ax.scatter([tx], [ty], [tz], c="red", marker="*", s=300, edgecolor="k", zorder=5)
    ax.plot([tx, tx + 0.6 * np.cos(tyaw)], [ty, ty + 0.6 * np.sin(tyaw)], [tz, tz], "r-", lw=2)
    # bins-mode deposit overlay: orange compartment dividers + green active bin.
    if "bin_edges" in d.files and d["bin_edges"].size >= 2 and "bounds_min" in d.files:
        bmin, bmax, edges = d["bounds_min"], np.array(d["bounds_max"]), d["bin_edges"]
        if "max_z" in d.files:   # cap the overlay at the real fill max
            bmax[2] = float(d["max_z"][0])
        active = int(d["active_bin"][0]) if "active_bin" in d.files else -1
        for ye in edges:
            ax.plot([bmin[0], bmax[0], bmax[0], bmin[0], bmin[0]], [ye] * 5,
                    [bmin[2], bmin[2], bmax[2], bmax[2], bmin[2]], color="orange", lw=1)
        if 0 <= active < len(edges) - 1:
            y0, y1 = float(edges[active]), float(edges[active + 1])
            ax.plot([bmin[0], bmax[0], bmax[0], bmin[0], bmin[0]], [y0, y0, y1, y1, y0],
                    [bmin[2]] * 5, color="green", lw=2)
    ax.view_init(elev=28, azim=-60)
    try:
        ax.set_box_aspect((1, 1, 0.5))
    except Exception:
        pass
    ax.set_xlabel("base X"); ax.set_ylabel("base Y"); ax.set_zlabel("base Z")
    m = re.search(r"(\d+)", os.path.basename(npz_path))
    cyc = m.group(1) if m else "?"
    ax.set_title(f"cycle {cyc}: target ({tx:.2f}, {ty:.2f}, {tz:.2f}) yaw {np.degrees(tyaw):.0f} deg")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)

    img = Image.fromarray(buf)
    w = max(1, int(round(img.width * height / img.height)))
    return np.asarray(img.resize((w, height), Image.BILINEAR))


def render_panel(npz_path, height):
    """One cycle's cloud + target as an RGB array (resized to `height`). Pretty Open3D render,
    matplotlib fallback if offscreen GL isn't available."""
    if _O3D["ok"]:
        try:
            return _o3d_panel(npz_path, height)
        except Exception as e:  # noqa: BLE001
            print(f"[render_run_video] Open3D panel failed ({e}); using matplotlib panels")
            _O3D["ok"] = False
    return _mpl_panel(npz_path, height)


def overlay_panel(cam, panel, margin=10):
    """Paste `panel` (already sized) into the top-left of `cam` (a copy), with a dark border."""
    out = cam.copy()
    ph, pw = panel.shape[:2]
    y0, x0 = margin, margin
    y1, x1 = min(cam.shape[0], y0 + ph), min(cam.shape[1], x0 + pw)
    b = 2  # border thickness
    out[max(0, y0 - b):y1 + b, max(0, x0 - b):x1 + b] = 0   # dark frame around the inset
    out[y0:y1, x0:x1] = panel[: y1 - y0, : x1 - x0, :3]
    return out


def load_decision_frames(run_dir, n):
    """Return the ascending per-decision video_frame indices from decisions.jsonl (exact sync),
    or None to signal 'fall back to even distribution'."""
    path = os.path.join(run_dir, "decisions.jsonl")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            recs = [json.loads(line) for line in f if line.strip()]
    except Exception:
        return None
    frames = [r.get("video_frame") for r in recs]
    if len(recs) != n or any(v is None for v in frames):
        return None
    frames = [int(v) for v in frames]
    if any(frames[i] < frames[i - 1] for i in range(1, len(frames))):
        return None   # not monotonic -> can't trust; even distribution instead
    return frames


def load_events(run_dir):
    """Interleaved panel timeline from decisions.jsonl: gaze + deposit events in FILE order
    (chronological; a resumed run resets video_frame, so a global frame sort would scramble
    sessions), each pointing at its npz dump. Returns
    [{'frame': int, 'npz': abspath, 't_wall': float, 'cycle': int}, ...] or None if unusable."""
    path = os.path.join(run_dir, "decisions.jsonl")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            recs = [json.loads(line) for line in f if line.strip()]
    except Exception:
        return None
    ev = []
    for r in recs:
        npz, vf = r.get("npz"), r.get("video_frame")
        if not npz or vf is None:
            continue
        p = os.path.join(run_dir, npz)
        if os.path.exists(p):
            ev.append({"frame": int(vf), "npz": p,
                       "t_wall": r.get("t_wall"), "cycle": r.get("cycle")})
    return ev or None


def load_phase_records(run_dir):
    """phases.jsonl records in FILE order (chronological across resumed sessions):
    [{'frame': int, 'phase': str, 't_wall': float}, ...]."""
    path = os.path.join(run_dir, "phases.jsonl")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            recs = [json.loads(line) for line in f if line.strip()]
    except Exception:
        return []
    return [{"frame": int(r["video_frame"]), "phase": str(r.get("phase", "?")),
             "t_wall": r.get("t_wall")}
            for r in recs if r.get("video_frame") is not None]


def probe_segment(path):
    """Metadata for one basemast segment; falls back to <path>_fixed.mp4 (untrunc
    recovery) when the original is truncated/unreadable. None if neither opens."""
    import imageio
    for p in (path, path + "_fixed.mp4"):
        if not os.path.exists(p):
            continue
        try:
            r = imageio.get_reader(p)
            meta = r.get_meta_data()
            fps = float(meta.get("fps", 15.0) or 15.0)
            dur = float(meta.get("duration") or 0.0)
            r.close()
        except Exception:
            continue
        end = os.path.getmtime(p)
        return {"path": p, "fps": fps, "t0": end - dur, "t1": end}
    if os.path.exists(path):
        print(f"segment unreadable (no _fixed recovery), skipping: {path}")
    return None


def find_segments(run_dir):
    """All basemast video segments of a (possibly resumed) run, chronological:
    basemast_run_part1.mp4 .. partN.mp4 (preserved earlier sessions), then
    basemast_run.mp4 (latest). partN numbering is preservation order == chronology;
    mtime (preserved by the renames) is used as the authoritative sort key."""
    parts = []
    for p in glob.glob(os.path.join(run_dir, "basemast_run_part*.mp4")):
        m = re.search(r"part(\d+)\.mp4$", p)
        if m:
            parts.append((int(m.group(1)), p))
    paths = [p for _, p in sorted(parts)]
    paths.append(os.path.join(run_dir, "basemast_run.mp4"))
    segs = [s for s in (probe_segment(p) for p in paths) if s]
    segs.sort(key=lambda s: s["t1"])
    return segs


def split_sessions(recs):
    """Split chronological records into per-session groups at video_frame resets
    (a node restart re-counts frames from 0). Adds each session's t_wall span."""
    sessions = []
    for r in recs:
        if not sessions or r["frame"] < sessions[-1]["recs"][-1]["frame"]:
            sessions.append({"recs": []})
        sessions[-1]["recs"].append(r)
    for s in sessions:
        ts = [r["t_wall"] for r in s["recs"] if r.get("t_wall") is not None]
        s["t0"], s["t1"] = (min(ts), max(ts)) if ts else (None, None)
    return sessions


def match_sessions(segments, sessions, label, slack=180.0):
    """Chronological two-pointer match of record sessions onto video segments by
    wall-clock overlap (segment window = [mtime - duration, mtime] +- slack).
    Returns a list parallel to segments (session dict or None). Sessions whose
    video did not survive are reported and skipped."""
    out, si = [], 0
    for seg in segments:
        matched = None
        while si < len(sessions):
            s = sessions[si]
            if s["t1"] is not None and s["t1"] < seg["t0"] - slack:
                cyc = [r.get("cycle") for r in s["recs"] if r.get("cycle") is not None]
                print(f"warning: {label} session (cycles {min(cyc)}..{max(cyc)})"
                      if cyc else f"warning: a {label} session",
                      "has no surviving video segment; skipped")
                si += 1
                continue
            if s["t0"] is not None and s["t0"] > seg["t1"] + slack:
                break   # session starts after this segment ends -> segment has no records
            matched = s
            si += 1
            break
        out.append(matched)
    for s in sessions[si:]:
        cyc = [r.get("cycle") for r in s["recs"] if r.get("cycle") is not None]
        print(f"warning: {label} session (cycles {min(cyc)}..{max(cyc)})"
              if cyc else f"warning: a {label} session",
              "has no surviving video segment; skipped")
    return out


def load_phases(run_dir):
    """Phase timeline from phases.jsonl: (frame, phase_name) sorted by video_frame, for the overlay.
    Returns two parallel lists (frames, names) or (None, None)."""
    path = os.path.join(run_dir, "phases.jsonl")
    if not os.path.exists(path):
        return None, None
    try:
        with open(path) as f:
            recs = [json.loads(line) for line in f if line.strip()]
    except Exception:
        return None, None
    ev = [(int(r["video_frame"]), str(r.get("phase", "?"))) for r in recs
          if r.get("video_frame") is not None]
    if not ev:
        return None, None
    ev.sort(key=lambda e: e[0])
    return [e[0] for e in ev], [e[1] for e in ev]


def draw_phase(frame, text):
    """Overlay the active FSM phase as a labeled banner (bottom-left) on an RGB frame."""
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    tw, th = d.textbbox((0, 0), text, font=font)[2:]
    x0, y0 = 12, img.height - th - 20
    d.rectangle([x0 - 6, y0 - 6, x0 + tw + 6, y0 + th + 6], fill=(0, 0, 0))
    d.text((x0, y0), text, fill=(0, 255, 120), font=font)
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser(description="Stitch a policy run into a review video")
    ap.add_argument("run_dir", help="run dir with policy_debug_*.npz (+ optional basemast_run.mp4)")
    ap.add_argument("--out", default=None, help="output path (default <run_dir>/review.mp4)")
    ap.add_argument("--fps", type=float, default=15.0, help="fps for the decisions-only slideshow")
    ap.add_argument("--hold", type=float, default=2.0, help="seconds per decision when no camera video")
    ap.add_argument("--layout", choices=["overlay", "side"], default="overlay",
                    help="overlay = decision panel inset at top-left of the camera frame (default); "
                         "side = camera and panel side-by-side")
    ap.add_argument("--inset-scale", type=float, default=0.33,
                    help="overlay inset height as a fraction of the frame height (default 0.33)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speedup: keep every Nth camera frame (e.g. 5 -> 5x video)")
    ap.add_argument("--glob", default=None,
                    help="omit (default) to INTERLEAVE gaze + deposit panels from decisions.jsonl (the "
                         "panel switches to the trailer/deposit view during the deposit phase); or set "
                         "'policy_debug_*.npz' / 'deposit_debug_*.npz' for a single-kind review.")
    args = ap.parse_args()

    # Panel timeline. Default = interleave gaze+deposit events from decisions.jsonl (each event = a
    # video_frame + its npz), so the overlay switches to the deposit panel when the deposit happens.
    # An explicit --glob gives a single-kind review from those npz files.
    events = load_events(args.run_dir) if args.glob is None else None
    if events:
        npzs = [e["npz"] for e in events]
        event_frames = [e["frame"] for e in events]
    else:
        pat = args.glob or "policy_debug_*.npz"
        npzs = sorted(glob.glob(os.path.join(args.run_dir, pat)))
        if not npzs:
            raise SystemExit(f"no {pat} found in {args.run_dir}")
        event_frames = load_decision_frames(args.run_dir, len(npzs))
    out = args.out or os.path.join(
        args.run_dir, "deposit_review.mp4" if (args.glob or "").startswith("deposit") else "review.mp4")
    n = len(npzs)
    phase_frames, phase_names = load_phases(args.run_dir)   # for the FSM-phase banner overlay

    import imageio

    mp4 = os.path.join(args.run_dir, "basemast_run.mp4")
    if args.glob is None and events:
        segments = find_segments(args.run_dir)
    elif os.path.exists(mp4):
        segments = [s for s in [probe_segment(mp4)] if s]
    else:
        segments = []

    if segments and args.glob is None and events:
        # Segment-aware stitch: a resumed run has several video segments
        # (basemast_run_part1..N + basemast_run.mp4) and decisions whose
        # video_frame indices reset at every restart. Match record sessions to
        # segments by wall clock and sync each session's panels to ITS segment;
        # mixing them against one segment shows the wrong pile states.
        dmatch = match_sessions(segments, split_sessions(events), "decisions")
        pmatch = match_sessions(segments, split_sessions(load_phase_records(args.run_dir)),
                                "phases")
        writer = imageio.get_writer(out, fps=segments[-1]["fps"], macro_block_size=None)
        step = max(1, int(round(args.speed)))
        count, n_panels, blank = 0, 0, None
        for seg, dsess, psess in zip(segments, dmatch, pmatch):
            reader = imageio.get_reader(seg["path"])
            cam_h = reader.get_data(0).shape[0]
            panel_h = cam_h if args.layout == "side" else max(120, int(cam_h * args.inset_scale))
            evs = sorted(dsess["recs"], key=lambda e: e["frame"]) if dsess else []
            panels = [render_panel(e["npz"], panel_h) for e in evs]
            starts = [e["frame"] for e in evs]
            n_panels += len(panels)
            if panels:
                blank = np.zeros_like(panels[0])
            precs = sorted(psess["recs"], key=lambda e: e["frame"]) if psess else []
            pframes = [r["frame"] for r in precs]
            pnames = [r["phase"] for r in precs]
            for i, cam in enumerate(reader):
                if i % step:
                    continue   # --speed N: keep every Nth frame
                cam3 = np.asarray(cam)[..., :3]
                if pframes:
                    pj = bisect.bisect_right(pframes, i) - 1
                    if pj >= 0:
                        cam3 = draw_phase(cam3, pnames[pj])
                if panels:
                    ci = max(0, min(len(panels) - 1, bisect.bisect_right(starts, i) - 1))
                    cam3 = (np.hstack([cam3, panels[ci]]) if args.layout == "side"
                            else overlay_panel(cam3, panels[ci]))
                elif args.layout == "side" and blank is not None:
                    cam3 = np.hstack([cam3, blank])
                writer.append_data(cam3)
                count += 1
            reader.close()
        writer.close()
        print(f"wrote {out}: {count} frames | {n_panels} decisions | "
              f"{len(segments)} segment(s) | layout={args.layout} | "
              f"align=exact per segment (decisions.jsonl video_frame)")
    elif segments:
        seg = segments[0]
        reader = imageio.get_reader(seg["path"])
        fps = seg["fps"]
        try:
            n_frames = reader.count_frames()
        except Exception:
            n_frames = None
        cam_h = reader.get_data(0).shape[0]
        reader.close()

        # Render panels at full frame height for side-by-side, or scaled down for the overlay inset.
        panel_h = cam_h if args.layout == "side" else max(120, int(cam_h * args.inset_scale))
        panels = [render_panel(z, panel_h) for z in npzs]
        start_frames = event_frames   # exact sync (event video_frames) if available
        writer = imageio.get_writer(out, fps=fps, macro_block_size=None)
        reader = imageio.get_reader(seg["path"])
        step = max(1, int(round(args.speed)))
        count = 0
        for i, cam in enumerate(reader):
            if i % step:
                continue   # --speed N: keep every Nth frame
            if start_frames is not None:
                ci = max(0, min(n - 1, bisect.bisect_right(start_frames, i) - 1))
            else:
                frac = (i / n_frames) if n_frames else 0.0
                ci = min(n - 1, int(frac * n)) if n_frames else min(n - 1, i)
            cam3 = np.asarray(cam)[..., :3]
            if phase_frames is not None:
                pj = bisect.bisect_right(phase_frames, i) - 1
                if pj >= 0:
                    cam3 = draw_phase(cam3, phase_names[pj])
            if args.layout == "side":
                writer.append_data(np.hstack([cam3, panels[ci]]))
            else:
                writer.append_data(overlay_panel(cam3, panels[ci]))
            count += 1
        reader.close()
        writer.close()
        align = ("exact (decisions.jsonl video_frame)" if start_frames is not None
                 else "even distribution (no decisions.jsonl)")
        print(f"wrote {out}: {count} frames | {n} decisions | {fps:.0f} fps | "
              f"layout={args.layout} | align={align}")
    else:
        height = 720
        panels = [render_panel(z, height) for z in npzs]
        writer = imageio.get_writer(out, fps=args.fps, macro_block_size=None)
        hold = max(1, int(round(args.hold * args.fps)))
        for p in panels:
            for _ in range(hold):
                writer.append_data(p)
        writer.close()
        print(f"wrote {out}: {n} decisions x {args.hold}s each "
              f"(no basemast_run.mp4 found -> decisions-only slideshow)")


if __name__ == "__main__":
    main()
