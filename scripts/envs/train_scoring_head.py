#!/usr/bin/env python3
"""Train the per-point SCORING grasp policy (see scoring_head.py for the motivation).

Trains on an EXISTING BC dataset - no recollection. The regression labels are decoded back to
metres and the classification target becomes the cloud point nearest the expert grasp.

    python3 scripts/envs/train_scoring_head.py \
        --dataset logs/bc_pointcloud/bc_policy_aug1_v2 \
        --output_dir logs/bc_pointcloud/scoring_v1 \
        --epochs 100 --zed_noise --dig 0.25

Deliberately standalone (plain torch, no Isaac import) so it runs while the simulator or the
crane's ZED node holds the GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring_head import ScoringGraspPolicy, scoring_loss, decode_regression_labels  # noqa: E402

# action box (gaze env decode bounds; matches deployment crop)
B_MIN = np.array([-5.364, -1.684, -1.30], dtype=np.float32)
B_MAX = np.array([-3.364, 5.316, 0.10], dtype=np.float32)
BED_FLOOR = -1.20          # bed_z + margin, the executable envelope
LOG_RADIUS = 0.056         # expert labels are log CENTRES = local surface - LOG_RADIUS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="BC dataset dir (pointclouds.npy + actions.npy)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--label_radius", type=float, default=0.12,
                   help="soft-label radius [m]; spreads probability over near-identical "
                        "neighbours but never across a gap between mounds")
    p.add_argument("--dig", type=float, default=0.0,
                   help="deployment dig convention [m]: shifts the label grasp z to "
                        "(local surface - dig). 0 keeps the sim expert's log-centre convention. "
                        "Applied to the REGRESSION residual only, so unlike the regression head "
                        "it cannot collapse the classification signal.")
    p.add_argument("--zed_noise", action="store_true", help="per-epoch ZED axial noise aug")
    p.add_argument("--zed_coeff", type=float, default=0.0014)
    p.add_argument("--stub_aug", type=float, default=0.0,
                   help="per-sample probability of POLE VARIATION: each pole column in the "
                        "cloud is independently kept (25%%), cut to a random stub of height "
                        "U(0.08, 0.6) m (65%%), or removed entirely (10%%). Fresh draws every "
                        "epoch. Teaches the short-vertical-remnant negative the margin data "
                        "misses (real broken-pole base, 2026-08-05); train-time ZED noise "
                        "roughens the stub points like everything else. Pole masks are "
                        "precomputed once, so the per-epoch cost is boolean ops only.")
    p.add_argument("--stub_neg", type=float, default=0.0,
                   help="EXPLICIT negative-loss weight on known pole/stub points (softplus on "
                        "their raw scores). Cross-entropy alone gives structure points no more "
                        "downward pressure than any other non-label point, which is why stub "
                        "variation alone only nudged the real stump's score (99.1 -> 98.0 "
                        "percentile). Requires --stub_aug > 0 (uses its precomputed pole masks).")
    p.add_argument("--shift_aug_x", type=float, default=0.0,
                   help="random rigid translation of the whole scene, uniform in [-v, v] metres")
    p.add_argument("--shift_aug_y", type=float, default=0.0,
                   help="ditto in y. This is the calibration-independence knob: the classification "
                        "label is a POINT INDEX, which a rigid translation leaves unchanged, so the "
                        "augmentation is free supervision for translation invariance. (A regression "
                        "head has to relearn the moved coordinate instead, which is why the deployed "
                        "policies needed the rack_y_shift shim.) z is NOT augmented - the bed floor "
                        "and gravity are physical, not a calibration choice.")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def apply_zed_noise(pc: torch.Tensor, cam: torch.Tensor, coeff: float) -> torch.Tensor:
    """Displace each point along its camera ray by N(0, coeff*range^2); pads untouched."""
    ray = pc - cam
    rng = ray.norm(dim=-1, keepdim=True)
    noised = pc + (ray / (rng + 1e-6)) * (torch.randn_like(rng) * (coeff * rng ** 2))
    keep = (pc.abs().sum(-1, keepdim=True) > 1e-6)
    return torch.where(keep, noised, pc)


def main():
    a = parse_args()
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "[Scoring] FATAL: --device cuda requested but torch.cuda.is_available() is False.\n"
            "  The container's SYSTEM python3 ships a torch built for a newer CUDA than the\n"
            "  driver, so it silently falls back to CPU. Run with Isaac's interpreter:\n"
            "    /workspace/isaaclab/_isaac_sim/python.sh scripts/envs/train_scoring_head.py ...\n"
            "  (or pass --device cpu deliberately).")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.output_dir, exist_ok=True)

    pc = np.load(os.path.join(a.dataset, "pointclouds.npy"))
    ac = np.load(os.path.join(a.dataset, "actions.npy"))
    tgt = decode_regression_labels(ac, B_MIN, B_MAX)          # (N, 4) metres/radians

    if a.dig > 0:
        # expert label z is the top log's CENTRE = local surface - LOG_RADIUS, so aiming at
        # (surface - dig) is a constant shift; clamp to the bed floor since deeper is not
        # executable. Only the dz RESIDUAL is affected - which point to grasp is untouched.
        tgt[:, 2] = np.maximum(tgt[:, 2] + (LOG_RADIUS - a.dig), BED_FLOOR)
        at_floor = float(np.mean(np.abs(tgt[:, 2] - BED_FLOOR) < 1e-6) * 100)
        print(f"[Scoring] dig {a.dig:.2f} m -> label z = surface-{a.dig:.2f} "
              f"({at_floor:.0f}% clamped at the {BED_FLOOR} floor; affects dz only)")

    cam_path = os.path.join(a.dataset, "cam_pos_base.npy")
    cam = torch.from_numpy(np.load(cam_path)).float() if os.path.exists(cam_path) else None
    if a.zed_noise and cam is None:
        print("[Scoring] WARN --zed_noise set but cam_pos_base.npy missing -> DISABLED")

    pole_info = None
    if a.stub_aug > 0:
        from stub_aug import find_pole_columns
        print("[Scoring] precomputing pole columns for --stub_aug ...")
        pole_info = []
        for i in range(len(pc)):
            cols = [(float(pc[i][m][:, 2].min()), np.where(m)[0].astype(np.int32))
                    for _, m in find_pole_columns(pc[i])]
            pole_info.append(cols)
        n_with = sum(1 for c in pole_info if c)
        print(f"[Scoring] pole columns found in {n_with}/{len(pc)} samples "
              f"(mean {np.mean([len(c) for c in pole_info]):.1f}/cloud)")

    n = len(pc)
    idx = np.random.permutation(n)
    n_val = int(n * a.val_frac)
    tr, va = idx[n_val:], idx[:n_val]
    print(f"[Scoring] dataset {a.dataset}: {n} samples -> train {len(tr)}, val {len(va)}")

    dev = torch.device(a.device)
    Xtr = torch.from_numpy(pc[tr]).float()
    Ttr = torch.from_numpy(tgt[tr]).float()
    Xva = torch.from_numpy(pc[va]).float().to(dev)
    Tva = torch.from_numpy(tgt[va]).float().to(dev)  # sliced per batch below
    tr_t = torch.from_numpy(tr)                     # original dataset indices (for pole_info)
    loader = DataLoader(TensorDataset(Xtr, Ttr, tr_t), batch_size=a.batch_size, shuffle=True,
                        drop_last=True)

    model = ScoringGraspPolicy(num_points=pc.shape[1], dropout=a.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    cam_d = cam.to(dev) if cam is not None else None

    best = float("inf")
    for ep in range(a.epochs):
        model.train()
        agg = {"loss": 0.0, "cls": 0.0, "dz": 0.0, "yaw": 0.0, "neg": 0.0, "n": 0}
        for xb, tb, ib in loader:
            xb, tb = xb.to(dev), tb.to(dev)
            neg_mask = None
            if pole_info is not None:
                # pole variation: per sample (prob stub_aug), each pole kept / cut / removed;
                # whatever pole points REMAIN in the cloud become explicit negatives (--stub_neg)
                xb = xb.clone()
                neg_mask = torch.zeros(xb.shape[0], xb.shape[1], dtype=torch.bool, device=dev)
                for bi in range(len(ib)):
                    cols = pole_info[int(ib[bi])]
                    do_cut = np.random.rand() < a.stub_aug
                    for zbase, pidx in cols:
                        if do_cut:
                            u = np.random.rand()
                            if u >= 0.25:
                                h = 0.05 if u > 0.90 else np.random.uniform(0.08, 0.6)
                                col = xb[bi, pidx]
                                cut = col[:, 2] > (zbase + h)
                                if cut.any():
                                    xb[bi, pidx[cut.cpu().numpy() if cut.is_cuda else cut.numpy()]] = 0.0
                        keep = xb[bi, pidx].abs().sum(-1) > 1e-6
                        kn = keep.cpu().numpy() if keep.is_cuda else keep.numpy()
                        neg_mask[bi, pidx[kn]] = True
            if a.zed_noise and cam_d is not None:
                xb = apply_zed_noise(xb, cam_d, a.zed_coeff)   # before the shift: ray geometry is
                                                               # defined in the ORIGINAL camera frame
            if a.shift_aug_x > 0 or a.shift_aug_y > 0:
                # rigid x/y translation of cloud AND label together; pads stay at the origin
                d = torch.zeros(xb.shape[0], 1, 3, device=dev)
                for j, v in enumerate((a.shift_aug_x, a.shift_aug_y)):
                    if v > 0:
                        d[:, 0, j] = (torch.rand(xb.shape[0], device=dev) * 2 - 1) * v
                keep = (xb.abs().sum(-1, keepdim=True) > 1e-6)
                xb = torch.where(keep, xb + d, xb)
                tb = tb.clone()
                tb[:, :2] += d[:, 0, :2]
            loss, parts = scoring_loss(model, xb, tb, label_smooth_radius=a.label_radius,
                                       neg_mask=neg_mask, w_neg=a.stub_neg)
            opt.zero_grad(); loss.backward(); opt.step()
            agg["loss"] += float(loss); agg["n"] += 1
            for k in ("cls", "dz", "yaw", "neg"):
                agg[k] += parts.get(k, 0.0)

        model.eval()
        with torch.no_grad():
            # batched: per-point activations for the whole val split do not fit in 8 GB
            vl, errs = [], []
            for i in range(0, len(Xva), a.batch_size):
                xb, tb = Xva[i:i + a.batch_size], Tva[i:i + a.batch_size]
                l, _ = scoring_loss(model, xb, tb, label_smooth_radius=a.label_radius)
                vl.append(float(l) * len(xb))
                pr = model.act(xb)
                errs.append(torch.linalg.norm(pr[:, :2] - tb[:, :2], dim=-1))
            vloss = sum(vl) / len(Xva)
            err_xy = torch.cat(errs)
            hit = float((err_xy < 0.30).float().mean() * 100)
        sched.step(vloss)

        if ep % 10 == 0 or ep == a.epochs - 1:
            m = agg["n"]
            print(f"Epoch {ep:3d}/{a.epochs} | train {agg['loss']/m:.4f} "
                  f"(cls {agg['cls']/m:.3f} dz {agg['dz']/m:.4f} yaw {agg['yaw']/m:.3f} "
                  f"neg {agg['neg']/m:.3f}) | "
                  f"val {vloss:.4f} | xy-err {float(err_xy.median()):.3f} m | "
                  f"within-30cm {hit:.0f}% | lr {opt.param_groups[0]['lr']:.2e}")

        if vloss < best:
            best = vloss
            torch.save({"model_state_dict": model.state_dict(),
                        "num_points": int(pc.shape[1]), "arch": "scoring_head_v1",
                        "dig": a.dig, "bounds_min": B_MIN, "bounds_max": B_MAX,
                        "metadata": {"dataset": a.dataset, "epochs": a.epochs,
                                     "trained": datetime.now().isoformat()}},
                       os.path.join(a.output_dir, "scoring_policy.pt"))

    print(f"\n[Scoring] best val {best:.4f} -> {a.output_dir}/scoring_policy.pt")
    with open(os.path.join(a.output_dir, "config.json"), "w") as f:
        json.dump(vars(a), f, indent=2)


if __name__ == "__main__":
    main()
