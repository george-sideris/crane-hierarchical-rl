"""Standalone point cloud pipeline for real ZED X depth camera.

Uses the EXACT same math as Isaac Lab's create_pointcloud_from_depth +
_world_to_base_frame from crane_pointcloud_direct_env.py.

Functions copied from:
  - isaaclab/utils/math.py: unproject_depth, transform_points, matrix_from_quat
  - crane_pointcloud_direct_env.py: _world_to_base_frame
"""

import numpy as np
import torch


# ── Copied from isaaclab/utils/math.py ───────────────────────────────

def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        Rotation matrices. The shape is (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def unproject_depth(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Un-project depth image into a pointcloud.

    Copied from isaaclab/utils/math.py.

    Args:
        depth: (H, W) depth image.
        intrinsics: (3, 3) camera intrinsic matrix.

    Returns:
        (P, 3) points in camera frame.
    """
    depth_batch = depth.clone()
    intrinsics_batch = intrinsics.clone()

    if depth_batch.dim() == 2:
        depth_batch = depth_batch[None]  # (H, W) -> (1, H, W)
    if depth_batch.dim() == 4 and depth_batch.shape[-1] == 1:
        depth_batch = depth_batch.squeeze(dim=3)
    if intrinsics_batch.dim() == 2:
        intrinsics_batch = intrinsics_batch[None]

    im_height, im_width = depth_batch.shape[1:]
    indices_u = torch.arange(im_width, device=depth.device, dtype=depth.dtype)
    indices_v = torch.arange(im_height, device=depth.device, dtype=depth.dtype)
    img_indices = torch.stack(torch.meshgrid([indices_u, indices_v], indexing="ij"), dim=0).reshape(2, -1)
    pixels = torch.nn.functional.pad(img_indices, (0, 0, 0, 1), mode="constant", value=1.0)
    pixels = pixels.unsqueeze(0)

    points = torch.matmul(torch.inverse(intrinsics_batch), pixels)
    points = points / points[:, -1, :].unsqueeze(1)
    depth_batch = depth_batch.transpose_(1, 2).reshape(depth_batch.shape[0], -1).unsqueeze(2)
    depth_batch = depth_batch.expand(-1, -1, 3)
    points_xyz = points.transpose_(1, 2) * depth_batch

    return points_xyz.squeeze(0)  # (P, 3)


def transform_points(points: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Transform points from source frame to target frame.

    Copied from isaaclab/utils/math.py.

    Args:
        points: (P, 3) points.
        pos: (3,) position of target frame.
        quat: (4,) quaternion (w,x,y,z) of target frame.

    Returns:
        (P, 3) transformed points.
    """
    points_batch = points.clone()
    if points_batch.dim() == 2:
        points_batch = points_batch[None]

    rot_mat = matrix_from_quat(quat)
    if rot_mat.dim() == 2:
        rot_mat = rot_mat[None]

    points_batch = torch.matmul(rot_mat, points_batch.transpose_(1, 2))
    points_batch = points_batch.transpose_(1, 2)

    if pos is not None:
        if pos.dim() == 1:
            pos = pos[None, None, :]
        points_batch += pos

    return points_batch.squeeze(0)


# ── Copied from crane_pointcloud_direct_env.py ───────────────────────

def world_to_base_frame(pc_world: torch.Tensor, base_pos: torch.Tensor,
                        base_quat: torch.Tensor) -> torch.Tensor:
    """Transform world-frame point cloud to crane base frame.

    Exact copy of CranePointCloudDirectEnv._world_to_base_frame.

    Args:
        pc_world: (N, 3) points in world frame.
        base_pos: (3,) crane base position in world frame.
        base_quat: (4,) crane base quaternion (wxyz) in world frame.

    Returns:
        (N, 3) points in base frame.
    """
    pc_translated = pc_world - base_pos

    w, x, y, z = base_quat[0], base_quat[1], base_quat[2], base_quat[3]
    R = torch.stack([
        torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y]),
        torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x]),
        torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]),
    ])

    pc_base = pc_translated @ R

    return pc_base


# ── FPS (same as before) ─────────────────────────────────────────────

def farthest_point_sampling(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Greedy farthest point sampling on torch tensor."""
    n = points.shape[0]
    if n == 0:
        return torch.zeros((num_samples, 3), dtype=points.dtype, device=points.device)
    if n <= num_samples:
        padding = torch.zeros((num_samples - n, 3), dtype=points.dtype, device=points.device)
        return torch.cat([points, padding], dim=0)

    indices = torch.zeros(num_samples, dtype=torch.long, device=points.device)
    distances = torch.full((n,), float('inf'), device=points.device)
    current = torch.randint(0, n, (1,), device=points.device).item()

    for i in range(num_samples):
        indices[i] = current
        dist = torch.norm(points - points[current:current+1], dim=1)
        distances = torch.minimum(distances, dist)
        current = torch.argmax(distances).item()

    return points[indices]


# ── Main pipeline ─────────────────────────────────────────────────────

_debug_printed = False  # reset on each restart

def process_depth(depth_np, intrinsics_np, cam_pos_np, cam_quat_wxyz_np,
                  base_pos_np, base_quat_wxyz_np,
                  num_points=1024, depth_range=(1.0, 10.0)):
    """Full pipeline: depth image -> flattened base-frame point cloud.

    Uses the EXACT same functions as Isaac Lab's sim pipeline.

    Args:
        depth_np: (H, W) float32 numpy depth in meters.
        intrinsics_np: (3, 3) float32 numpy intrinsic matrix.
        cam_pos_np: (3,) float32 numpy camera position in world frame.
        cam_quat_wxyz_np: (4,) float32 numpy camera quaternion (wxyz, ROS convention).
        base_pos_np: (3,) float32 numpy crane base position in world frame.
        base_quat_wxyz_np: (4,) float32 numpy crane base quaternion (wxyz).
        num_points: FPS target count.
        depth_range: (min, max) valid depth in meters.

    Returns:
        (num_points * 3,) float32 tensor — flattened point cloud in base frame.
    """
    global _debug_printed
    device = torch.device("cpu")

    # Convert to torch
    depth = torch.from_numpy(depth_np).float().to(device)
    intrinsics = torch.from_numpy(intrinsics_np).float().to(device)
    cam_pos = torch.from_numpy(cam_pos_np).float().to(device)
    cam_quat = torch.from_numpy(cam_quat_wxyz_np).float().to(device)
    base_pos = torch.from_numpy(base_pos_np).float().to(device)
    base_quat = torch.from_numpy(base_quat_wxyz_np).float().to(device)

    # Filter depth range
    min_d, max_d = depth_range
    depth_mask = (depth >= min_d) & (depth <= max_d) & (~torch.isinf(depth))
    filtered_depth = torch.where(depth_mask, depth, torch.tensor(float('inf'), device=device))

    # Unproject to camera frame (same as Isaac Lab)
    points_cam = unproject_depth(filtered_depth, intrinsics)

    # Debug: print stats at each step (once)
    if not _debug_printed:
        v = torch.all(torch.isfinite(points_cam), dim=1)
        pc = points_cam[v]
        if pc.shape[0] > 0:
            print(f"[PCD DEBUG] cam_frame: n={pc.shape[0]}, "
                  f"x=[{pc[:,0].min():.2f},{pc[:,0].max():.2f}] "
                  f"y=[{pc[:,1].min():.2f},{pc[:,1].max():.2f}] "
                  f"z=[{pc[:,2].min():.2f},{pc[:,2].max():.2f}]")

    # Camera frame -> world frame (same as Isaac Lab)
    points_world = transform_points(points_cam, cam_pos, cam_quat)

    if not _debug_printed:
        v = torch.all(torch.isfinite(points_world), dim=1)
        pc = points_world[v]
        if pc.shape[0] > 0:
            print(f"[PCD DEBUG] world_frame: n={pc.shape[0]}, "
                  f"x=[{pc[:,0].min():.2f},{pc[:,0].max():.2f}] "
                  f"y=[{pc[:,1].min():.2f},{pc[:,1].max():.2f}] "
                  f"z=[{pc[:,2].min():.2f},{pc[:,2].max():.2f}]")

    # Remove invalid points
    valid = torch.all(torch.isfinite(points_world), dim=1)
    points_world = points_world[valid]

    if points_world.shape[0] == 0:
        return torch.zeros(num_points * 3, dtype=torch.float32)

    # Limit points before base frame transform
    if points_world.shape[0] > 5000:
        idx = torch.randperm(points_world.shape[0])[:5000]
        points_world = points_world[idx]

    # World frame -> base frame (same as crane_pointcloud_direct_env.py)
    points_base = world_to_base_frame(points_world, base_pos, base_quat)

    if not _debug_printed:
        print(f"[PCD DEBUG] base_frame: n={points_base.shape[0]}, "
              f"x=[{points_base[:,0].min():.2f},{points_base[:,0].max():.2f}] "
              f"y=[{points_base[:,1].min():.2f},{points_base[:,1].max():.2f}] "
              f"z=[{points_base[:,2].min():.2f},{points_base[:,2].max():.2f}]")
        print(f"[PCD DEBUG] cam_pos={cam_pos_np}, cam_quat={cam_quat_wxyz_np}")
        print(f"[PCD DEBUG] base_pos={base_pos_np}, base_quat={base_quat_wxyz_np}")
        _debug_printed = True

    # FPS downsample
    points_sampled = farthest_point_sampling(points_base, num_points)

    return points_sampled.view(-1).float()
