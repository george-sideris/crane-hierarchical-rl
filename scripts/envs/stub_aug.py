#!/usr/bin/env python3
"""Stub augmentation for structure-inclusive BC clouds - NO recollection needed.

Two operations, applied per-epoch at train time (fresh draws each epoch):

  truncate_poles : the collected margin clouds contain full tall pole columns; randomly cut
                   each one to a stub of height h ~ U(hmin, hmax). Turns the tall-pole
                   negatives the policy already learned into the SHORT-stub negatives it
                   missed (real broken-pole base, observed 2026-08-05).
  inject_stub    : procedurally add a camera-facing half-cylinder of points (ZED-ish noise)
                   at a random position - breaks position dependence so "short vertical
                   thing != log" is learned as a class. KEEP-OUT radius around the label xy
                   so a stub can never steal the nearest-point classification target.

Both keep the point count constant (removals -> zero pads at the END; injections overwrite
pads first, else random non-label points), so tensors stay fixed-shape.
"""

from __future__ import annotations

import numpy as np


def find_pole_columns(pts: np.ndarray, top_z: float = -0.25, xy_radius: float = 0.18):
    """Columns of points reaching above top_z (only structure does; pile tops ~ -0.35).

    Returns list of (center_xy, column_mask)."""
    valid = np.abs(pts).sum(-1) > 1e-6
    hi = pts[valid & (pts[:, 2] > top_z)]
    if len(hi) == 0:
        return []
    # cluster the high points on a coarse xy grid
    key = np.round(hi[:, :2] / 0.3).astype(int)
    cols = []
    for k in np.unique(key, axis=0):
        c = hi[(key == k).all(1)][:, :2].mean(0)
        if any(np.linalg.norm(c - c0) < 0.3 for c0, _ in cols):
            continue
        mask = valid & (np.linalg.norm(pts[:, :2] - c[None, :], axis=1) < xy_radius)
        cols.append((c, mask))
    return cols


def truncate_poles(pts: np.ndarray, rng: np.random.Generator,
                   hmin: float = 0.08, hmax: float = 0.6):
    """Cut every pole column to a random stub height. Returns (new_pts, removed_mask)."""
    out = pts.copy()
    removed = np.zeros(len(pts), dtype=bool)
    for c, mask in find_pole_columns(pts):
        zbase = pts[mask][:, 2].min()
        h = rng.uniform(hmin, hmax)
        cut = mask & (pts[:, 2] > zbase + h)
        removed |= cut
    out[removed] = 0.0
    return out, removed


def inject_stub(pts: np.ndarray, cam_pos: np.ndarray, rng: np.random.Generator,
                label_xy: np.ndarray | None = None, keepout: float = 0.8,
                box_min=(-5.364, -1.684), box_max=(-3.364, 5.316)):
    """Add one synthetic camera-facing stub. Returns (new_pts, injected_mask)."""
    # Stumps are RACK FIXTURES: they occur on the perimeter (rail line / pole positions),
    # standing on exposed bed - never protruding from the middle of the pile (which would
    # also wrongly punish steeply-poking logs). Placement prior: perimeter band or a
    # jittered pole spot; accept only where the local content is LOW (stub stands on bed).
    # ONLY at pole positions (a broken pole IS a pole fixture) with modest jitter - the
    # earlier perimeter band leaked stubs among log ENDS at the box x-edges. Pole spots are
    # detected per cloud from the tall columns; if none survive detection or every jittered
    # placement fails the checks, we simply skip the injection (truncation already provides
    # pole-position stubs in most draws).
    valid = np.abs(pts).sum(-1) > 1e-6
    pole_spots = [c for c, _ in find_pole_columns(pts)]
    if not pole_spots:
        return pts, np.zeros(len(pts), dtype=bool)
    ok = False
    for _ in range(30):
        c = pole_spots[int(rng.integers(len(pole_spots)))]
        x0 = c[0] + rng.uniform(-0.25, 0.25)
        y0 = c[1] + rng.uniform(-0.25, 0.25)
        if label_xy is not None and np.hypot(x0 - label_xy[0], y0 - label_xy[1]) <= keepout:
            continue
        near = valid & (np.linalg.norm(pts[:, :2] - [x0, y0], axis=1) < 0.35)
        zloc = np.percentile(pts[near][:, 2], 5) if near.sum() > 8 else -1.30
        # strict low-content check: stub must stand on bed, never among log points
        if near.sum() <= 8 or np.percentile(pts[near][:, 2], 90) < zloc + 0.12:
            ok = True
            break
    if not ok:
        return pts, np.zeros(len(pts), dtype=bool)
    near = valid & (np.linalg.norm(pts[:, :2] - [x0, y0], axis=1) < 0.4)
    zloc = np.percentile(pts[near][:, 2], 5) if near.sum() > 8 else -1.30

    h = rng.uniform(0.15, 0.45)
    r = rng.uniform(0.05, 0.12)
    n = int(rng.integers(35, 90))
    base_ang = np.arctan2(cam_pos[1] - y0, cam_pos[0] - x0)
    th = base_ang + rng.uniform(-1.3, 1.3, n)             # camera-facing half only
    z = zloc + rng.uniform(0, h, n)
    stub = np.stack([x0 + r * np.cos(th), y0 + r * np.sin(th), z], axis=1)
    tilt = rng.uniform(-0.12, 0.12, 2)                    # slight lean
    stub[:, 0] += tilt[0] * (stub[:, 2] - zloc)
    stub[:, 1] += tilt[1] * (stub[:, 2] - zloc)
    stub += rng.normal(0, 0.008, stub.shape)              # sensor-ish jitter

    out = pts.copy()
    injected = np.zeros(len(pts), dtype=bool)
    pads = np.where(~valid)[0]
    take = pads[:n]
    if len(take) < n:                                     # overwrite random non-label points
        cand = np.where(valid)[0]
        if label_xy is not None:
            cand = cand[np.linalg.norm(pts[cand][:, :2] - label_xy[None, :], axis=1) > keepout]
        extra = rng.choice(cand, n - len(take), replace=False)
        take = np.concatenate([take, extra])
    out[take] = stub[:len(take)].astype(pts.dtype)
    injected[take] = True
    return out, injected


def stub_augment(pts, cam_pos, rng, label_xy=None, p_truncate=0.7, n_inject=(0, 3)):
    """The train-time composition: maybe truncate poles, inject 0-N stubs."""
    out = pts
    if rng.uniform() < p_truncate:
        out, _ = truncate_poles(out, rng)
    for _ in range(int(rng.integers(n_inject[0], n_inject[1] + 1))):
        out, _ = inject_stub(out, cam_pos, rng, label_xy=label_xy)
    return out
