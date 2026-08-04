#!/usr/bin/env python3
"""Per-point SCORING policy: pick a grasp by argmax over observed cloud points.

Motivation (measured 2026-08-03). The regression head predicts (x, y, z, cos2yaw, sin2yaw)
under an MSE loss. On a pile with two near-equal mounds the L2-optimal output is their MEAN,
which lands in the empty gap between them. Measured gap-aim rate on near-tie scenes:

    expert (privileged argmax)   0.0 %
    heuristic (argmax)           0.0 %
    BC regression head          33.9 %   (21 % of those grasp nothing)
    BCRL m100                   48.5 %

Both argmax controllers are immune BY CONSTRUCTION - they select among candidates instead of
averaging coordinates. On the crane this failure is absorbing: a grasp over a gap disturbs
nothing, so the next cloud is identical and the policy re-issues the same target (observed
2026-08-03, 5 consecutive cycles). In sim it costs ~3 cycles per pile (c95 19.0 vs 16.2) because
failed grasps still nudge the pile, letting the policy escape.

Design: score every input point, take the argmax, and read the grasp off that point.
  * mode-seeking by construction - cross-entropy cannot be minimised by "averaging" two mounds
  * the target is always ON observed material, so aiming at empty space becomes impossible
  * z is a small residual from the chosen (surface) point rather than an absolute coordinate,
    which also sidesteps the label-clamping problem that flattened the dig-0.25 labels
  * reuses the existing PointNet per-point features (currently discarded by the max-pool), so
    the encoder - and therefore the frozen-encoder BCRL recipe - carries over unchanged

Trains on the EXISTING dataset: the classification target is the cloud point nearest the
expert's labelled grasp; z/yaw are regressed only at that point.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetPerPoint(nn.Module):
    """PointNet trunk that returns BOTH per-point features and the pooled global feature.

    Identical layer sizes to the deployed PointNetEncoder so encoder weights remain
    load-compatible (mlp1.* / fc.*); the only change is that per-point features are also
    returned instead of being discarded at the max-pool.
    """

    def __init__(self, input_dim: int = 3, output_dim: int = 256):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ELU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ELU(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ELU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, output_dim), nn.BatchNorm1d(output_dim), nn.ELU(),
        )
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor):
        b, n, _ = x.shape
        f = self.mlp1(x.reshape(b * n, -1)).view(b, n, -1)      # (B, N, 256) per-point
        g = self.fc(f.max(dim=1).values)                        # (B, output_dim) global
        return f, g


class ScoringGraspPolicy(nn.Module):
    """Argmax-over-points grasp policy.

    forward() returns raw heads; act() returns the decoded grasp in METRES (base frame),
    which is what the deployment node and the offline checks consume.

    Heads (all per-point, conditioned on the global feature):
      score : (B, N)      grasp-quality logit for every observed point
      dz    : (B, N)      vertical offset from that point to the commanded grasp centre
      yaw   : (B, N, 2)   (cos 2*yaw, sin 2*yaw) at that point
    """

    def __init__(self, num_points: int = 1024, latent_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.num_points = num_points
        self.encoder = PointNetPerPoint(input_dim=3, output_dim=latent_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # per-point feature (256) + global (latent_dim) + raw xyz (3)
        in_dim = 256 + latent_dim + 3
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
        )
        self.score_out = nn.Linear(64, 1)
        self.dz_out = nn.Linear(64, 1)
        self.yaw_out = nn.Linear(64, 2)
        print(f"[ScoringHead] {num_points} points -> per-point score/dz/yaw (argmax select)")

    def forward(self, points: torch.Tensor):
        b = points.shape[0]
        if points.dim() == 2:
            points = points.view(b, self.num_points, 3)
        n = points.shape[1]

        per_pt, glob = self.encoder(points)
        glob_e = self.dropout(glob).unsqueeze(1).expand(-1, n, -1)
        h = self.head(torch.cat([per_pt, glob_e, points], dim=-1))

        score = self.score_out(h).squeeze(-1)                    # (B, N)
        # zero-padded slots must never win the argmax
        pad = (points.abs().sum(-1) < 1e-6)
        score = score.masked_fill(pad, -1e9)
        return score, self.dz_out(h).squeeze(-1), self.yaw_out(h), points

    # SUPPORT GATE (inference-time). Raw argmax is mode-seeking but brittle: ONE noise-displaced
    # point can win, which collapsed the sim eval at the characterised ZED noise level (93% -> 50%
    # grasp success; the empty cycles' issued targets had median support 3 vs 56 for successful
    # ones). Score smoothing does NOT fix this - an isolated outlier's neighbourhood mean is its
    # own score. Instead, points with local support below support_frac * (cloud max) are excluded
    # from the argmax: a target must sit on a supported cluster. Verified on the exact collapse
    # clouds: empty-cycle target support 3 -> 40, successful-cycle targets unmoved (median 0.00 m).
    # support_frac=0 disables (pre-2026-08-04 behaviour).
    support_frac: float = 0.25
    support_radius: float = 0.5

    @torch.no_grad()
    def act(self, points: torch.Tensor, support_frac: float | None = None) -> torch.Tensor:
        """Decoded grasp (B, 4) = (x, y, z, yaw) in metres/radians."""
        # the gate's (B, N, N) cdist is ~1 GB at B=64, N=2048 - chunk large batches so
        # validation inside an 8 GB budget cannot OOM (observed 2026-08-04)
        if points.dim() == 3 and points.shape[0] > 8:
            return torch.cat([self.act(points[i:i + 8], support_frac)
                              for i in range(0, points.shape[0], 8)], dim=0)
        score, dz, yaw, pts = self.forward(points)
        frac = self.support_frac if support_frac is None else support_frac
        valid = (pts.abs().sum(-1) > 1e-6)                       # (B, N)
        sel = score
        if frac and frac > 0:
            nb = (torch.cdist(pts[:, :, :2], pts[:, :, :2]) < self.support_radius)
            cnt = (nb & valid.unsqueeze(1)).sum(-1).float()      # (B, N) local support
            gate = valid & (cnt >= frac * cnt.max(dim=1, keepdim=True).values)
            # a fully-gated cloud (degenerate) falls back to the raw argmax
            gate = torch.where(gate.any(dim=1, keepdim=True), gate, valid)
            sel = score.masked_fill(~gate, -1e9)
        idx = sel.argmax(dim=1)                                  # (B,) chosen point
        b = torch.arange(pts.shape[0], device=pts.device)
        p = pts[b, idx]                                          # (B, 3)
        z = p[:, 2] + dz[b, idx]
        y2 = yaw[b, idx]
        ang = 0.5 * torch.atan2(y2[:, 1], y2[:, 0])
        return torch.stack([p[:, 0], p[:, 1], z, ang], dim=-1)


def scoring_loss(model: ScoringGraspPolicy, points: torch.Tensor, target: torch.Tensor,
                 label_smooth_radius: float = 0.12, w_dz: float = 1.0, w_yaw: float = 0.5):
    """Cross-entropy on the point nearest the labelled grasp + regressions at that point.

    Args:
        points: (B, N, 3) cloud, metres, base frame
        target: (B, 4) expert grasp (x, y, z, yaw) in metres/radians
    label_smooth_radius spreads probability over points within that horizontal distance of the
    label, so near-identical neighbours do not fight each other - it does NOT blur across the
    gap between mounds, which is the whole point of using a classification loss here.
    """
    score, dz, yaw, pts = model(points)
    b, n, _ = pts.shape

    d_xy = torch.linalg.norm(pts[:, :, :2] - target[:, None, :2], dim=-1)   # (B, N)
    pad = (pts.abs().sum(-1) < 1e-6)
    d_xy = d_xy.masked_fill(pad, 1e9)
    nearest = d_xy.argmin(dim=1)

    if label_smooth_radius > 0:
        soft = (d_xy <= label_smooth_radius).float()
        soft[torch.arange(b, device=pts.device), nearest] = 1.0            # always include argmin
        soft = soft / soft.sum(dim=1, keepdim=True).clamp_min(1e-6)
        loss_cls = -(soft * F.log_softmax(score, dim=1)).sum(dim=1).mean()
    else:
        loss_cls = F.cross_entropy(score, nearest)

    ar = torch.arange(b, device=pts.device)
    p_sel = pts[ar, nearest]
    loss_dz = F.mse_loss(dz[ar, nearest], target[:, 2] - p_sel[:, 2])
    yaw_t = torch.stack([torch.cos(2 * target[:, 3]), torch.sin(2 * target[:, 3])], dim=-1)
    loss_yaw = F.mse_loss(yaw[ar, nearest], yaw_t)

    total = loss_cls + w_dz * loss_dz + w_yaw * loss_yaw
    return total, {"cls": float(loss_cls), "dz": float(loss_dz), "yaw": float(loss_yaw)}


def decode_regression_labels(actions: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    """Existing dataset labels (arctanh-encoded 5D) -> (N, 4) metres/radians targets."""
    xyz = bmin[:3] + (np.tanh(actions[:, :3]) + 1.0) / 2.0 * (bmax[:3] - bmin[:3])
    yaw = 0.5 * np.arctan2(actions[:, 4], actions[:, 3]) if actions.shape[1] >= 5 \
        else actions[:, 3][:, None]
    return np.concatenate([xyz, np.asarray(yaw).reshape(-1, 1)], axis=1).astype(np.float32)
