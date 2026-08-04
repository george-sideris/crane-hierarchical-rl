#!/usr/bin/env python3
"""Build a self-distillation dataset: a copy of a collection dir whose actions are a
checkpoint's OWN outputs on the collection's clouds.

Motivation (2026-07-29): the BCRL's gains live entirely in the actor head (encoder frozen
during RL), and every SFT epoch trains that head toward expert-flavored labels, eroding the
RL deltas. Fine-tuning on {sim clouds -> the RL policy's own actions} anchors RL behavior on
the sim domain (learning-without-forgetting), while the mixed-in real frames and bimodal aug
carry corrective supervision only where the RL policy was never trained.

Usage:
  python3 self_label_dataset.py --data <collection dir> --checkpoint <BC or RSL-RL ckpt> \
                                --out <new dataset dir>
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="collection dir (pointclouds.npy + actions.npy)")
    ap.add_argument("--checkpoint", required=True, help="checkpoint whose outputs become the labels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    sys.argv = ["x", "--train_only", "d"]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import train_bc_pointcloud as t

    pc = np.load(os.path.join(args.data, "pointclouds.npy"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = t.BCPointNetPolicy(num_points=pc.shape[1], action_dim=5)
    net.load_state_dict(t.load_init_state(args.checkpoint))
    net.eval().to(dev)

    outs = []
    with torch.no_grad():
        for i in range(0, len(pc), args.batch):
            x = torch.from_numpy(pc[i:i + args.batch].reshape(len(pc[i:i + args.batch]), -1)).float().to(dev)
            outs.append(net(x).cpu().numpy())
    actions = np.concatenate(outs).astype(np.float32)

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "actions.npy"), actions)
    for f in ("pointclouds.npy", "episode_rewards.npy", "cam_pos_base.npy"):
        src = os.path.join(args.data, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out, f))
    with open(os.path.join(args.data, "metadata.json")) as f:
        meta = json.load(f)
    meta["self_labeled_from"] = args.checkpoint
    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"self-labeled dataset: {len(actions)} samples -> {args.out} (labels from {args.checkpoint})")


if __name__ == "__main__":
    main()
