#!/usr/bin/env python3
"""Sim2real sanity check: run the trained BC point-cloud policy on a REAL gaze-pose
ZED cloud (base frame) and visualize the predicted grasp target on the real pile.

No Isaac: pure torch forward pass. Preprocessing (crop to action bounds + FPS to 1024)
matches training exactly. 5D cossin action decoded to (x,y,z,yaw)."""
import argparse
import numpy as np
import torch
import torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="/workspace/isaaclab/logs/bc_pointcloud/bc_20260629_053119_cossin/bc_pointcloud_policy.pt")
ap.add_argument("--real", default="out/real_gaze_base.npy")
ap.add_argument("--interactive", action="store_true", help="open an interactive Open3D 3D view (needs X11) instead of saving a PNG")
ap.add_argument("--full", action="store_true", help="show the full cloud, not just the action-box crop")
ap.add_argument("--full_input", action="store_true",
                help="policy trained with --use_full_pcd: feed the UNCROPPED cloud (no action-box crop, no pole "
                     "strip), only the sim depth-range filter (1-10 m from the camera). --real should point to a "
                     "real_cycleN_full.npy; the camera origin is read from the matching real_cycleN_cam.npy")
args = ap.parse_args()
CKPT, REAL = args.checkpoint, args.real

# action bounds = the calibrated gaze action box (must match training)
RX, RY = -4.364, 1.816
BMIN = np.array([RX - 1.00, RY - 3.50, -1.30])
BMAX = np.array([RX + 1.00, RY + 3.50, 0.10])
NUM_POINTS = 1024


class PointNetEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=256, norm_type="batchnorm"):
        super().__init__()
        nc = nn.LayerNorm if norm_type == "layernorm" else nn.BatchNorm1d
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 64), nc(64), nn.ELU(),
            nn.Linear(64, 128), nc(128), nn.ELU(),
            nn.Linear(128, 256), nc(256), nn.ELU())
        self.fc = nn.Sequential(nn.Linear(256, output_dim), nc(output_dim), nn.ELU())

    def forward(self, x):
        b, n, _ = x.shape
        x = self.mlp1(x.reshape(b * n, -1)).reshape(b, n, -1)
        x = x.max(dim=1)[0]
        return self.fc(x)


class BCPointNetPolicy(nn.Module):
    def __init__(self, num_points=1024, action_dim=5, latent_dim=256, dropout=0.2, norm_type="batchnorm"):
        super().__init__()
        self.num_points = num_points
        self.encoder = PointNetEncoder(3, latent_dim, norm_type)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.actor_mlp = nn.Sequential(nn.Linear(latent_dim, 128), nn.ELU(),
                                       nn.Linear(128, 64), nn.ELU(), nn.Linear(64, action_dim))

    def forward(self, points):
        if points.dim() == 2:
            points = points.view(points.shape[0], self.num_points, 3)
        return self.actor_mlp(self.dropout(self.encoder(points)))


def fps(P, k, seed=0):
    rng = np.random.default_rng(seed)
    N = len(P)
    if N <= k:
        pad = np.zeros((k - N, 3), np.float32)
        return np.vstack([P, pad])
    idx = [int(rng.integers(N))]
    d = np.full(N, 1e18)
    for _ in range(k - 1):
        d = np.minimum(d, ((P - P[idx[-1]]) ** 2).sum(1))
        idx.append(int(d.argmax()))
    return P[idx]


def denorm(n, lo, hi):
    return lo + (n + 1.0) / 2.0 * (hi - lo)


def show_o3d(cloud, target, yaw):
    import open3d as o3d
    import matplotlib.cm as cm
    cloud = cloud[:, :3].astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    z = cloud[:, 2]; zn = (z - z.min()) / (np.ptp(z) + 1e-9)
    pcd.colors = o3d.utility.Vector3dVector(cm.viridis(zn)[:, :3])
    sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.12)
    sph.translate(target); sph.paint_uniform_color([1, 0, 0]); sph.compute_vertex_normals()
    end = np.asarray(target) + np.array([0.7*np.cos(yaw), 0.7*np.sin(yaw), 0.0])
    line = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector([target, end]),
                                lines=o3d.utility.Vector2iVector([[0, 1]]))
    line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)  # base_link origin
    print("opening Open3D window — mouse: rotate, scroll: zoom, shift+drag: pan. close window to exit.")
    o3d.visualization.draw_geometries([pcd, sph, line, frame],
                                      window_name="BC grasp target on real gaze cloud (red=target)")


def main():
    ck = torch.load(CKPT, map_location="cpu")
    sd = ck.get("model_state_dict", ck)
    action_dim = ck.get("action_dim", 5)
    num_points = ck.get("num_points", NUM_POINTS)
    norm_type = "layernorm" if any("encoder.fc.1.weight" in k and sd[k].dim() == 1 and "running" not in k for k in sd) else "batchnorm"
    # detect norm by presence of running_mean (batchnorm) keys
    norm_type = "batchnorm" if any("running_mean" in k for k in sd) else "layernorm"
    policy = BCPointNetPolicy(num_points=num_points, action_dim=action_dim, norm_type=norm_type)
    policy.load_state_dict(sd)
    policy.eval()
    print(f"loaded policy: action_dim={action_dim} num_points={num_points} norm={norm_type}")

    real = np.load(REAL).astype(np.float32)
    if args.full_input:
        # mirror the sim full-cloud pipeline: depth-range filter about the camera, nothing else
        cam_path = REAL.replace("_full.npy", "_cam.npy")
        cam = np.load(cam_path).astype(np.float32)
        dist = np.linalg.norm(real - cam, axis=1)
        m = (dist >= 1.0) & (dist <= 10.0)
        crop = real[m]
        # sim pre-samples uniformly to 6x target before FPS; mirror it (also keeps FPS fast)
        if len(crop) > num_points * 6:
            idx = np.random.default_rng(0).choice(len(crop), num_points * 6, replace=False)
            crop = crop[idx]
        print(f"real cloud {len(real)} -> in depth range 1-10 m of cam {cam.round(2)}: {len(crop)} -> FPS {num_points}")
    else:
        m = np.all((real >= BMIN) & (real <= BMAX), axis=1)
        # strip the front rack poles: static vertical columns clipping the box corners that the
        # sim training clouds never contain (they drag the PointNet max-pool features)
        pole = ((real[:, 0] > -3.6) & (real[:, 1] < -1.35)) \
             | ((real[:, 0] > -3.8) & (real[:, 1] > 5.1))
        n_pole = int((m & pole).sum())
        m &= ~pole
        crop = real[m]
        print(f"real cloud {len(real)} -> in action box {len(crop)} (pole points removed: {n_pole}) -> FPS {num_points}")
    pol_in = fps(crop, num_points)

    with torch.no_grad():
        a = policy(torch.from_numpy(pol_in[None]).float())[0].numpy()
    t = np.tanh(a)
    x = denorm(t[0], BMIN[0], BMAX[0]); y = denorm(t[1], BMIN[1], BMAX[1]); z = denorm(t[2], BMIN[2], BMAX[2])
    yaw = np.arctan2(t[4], t[3]) / 2.0 if action_dim == 5 else denorm(t[3], -np.pi/2, np.pi/2)
    print(f"\nPREDICTED grasp target (base frame): x={x:.3f} y={y:.3f} z={z:.3f}  yaw={np.degrees(yaw):.1f} deg")

    tgt = np.array([x, y, z])
    if args.interactive:
        return show_o3d(real if args.full else crop, tgt, yaw)

    # viz: top-down + side, real cloud + predicted target
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(crop[:, 0], crop[:, 1], s=2, c=crop[:, 2], cmap="viridis")
    ax.scatter([x], [y], c="red", marker="*", s=400, edgecolor="k", zorder=5, label="predicted target")
    ax.plot([x, x + 0.6*np.cos(yaw)], [y, y + 0.6*np.sin(yaw)], "r-", lw=2)
    ax.set_aspect("equal"); ax.set_title("top-down: real pile + predicted grasp"); ax.set_xlabel("base X"); ax.set_ylabel("base Y"); ax.legend()
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(crop[:, 1], crop[:, 2], s=2, c=crop[:, 2], cmap="viridis")
    ax2.scatter([y], [z], c="red", marker="*", s=400, edgecolor="k", zorder=5)
    ax2.set_title("side Y-Z: is the target on top of the pile?"); ax2.set_xlabel("base Y"); ax2.set_ylabel("base Z"); ax2.set_aspect("equal")
    plt.tight_layout(); plt.savefig("out/policy_on_real.png", dpi=95)
    print("saved out/policy_on_real.png")


if __name__ == "__main__":
    main()
