"""Depth image to base-frame point cloud, matching the training pipeline.

Pipeline steps:
  1. unproject the depth image into camera-frame XYZ
  2. transform camera-frame XYZ into world-frame using the camera extrinsic
  3. transform world-frame into crane base frame using the base pose
  4. farthest-point-sample down to num_points
"""

import numpy as np
import torch


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


def world_to_base_frame(pc_world: torch.Tensor, base_pos: torch.Tensor,
                        base_quat: torch.Tensor) -> torch.Tensor:
    """Transform world-frame point cloud to crane base frame.

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


def process_cloud(points_np, base_from_cloud_pos_np, base_from_cloud_quat_wxyz_np,
                  bounds_min_np, bounds_max_np, num_points=1024,
                  pole_x_gt=None, pole_y_lt=None, pole2_x_gt=None, pole2_y_gt=None,
                  pole3_x_lt=None, pole3_y_lt=None,
                  ror_radius=None, ror_min_neighbors=None):
    """Registered ZED cloud -> cropped, FPS'd, flattened BASE-frame cloud.

    Mirrors training exactly (get_pointcloud_base -> crop_to_bounds -> FPS): the ZED publishes
    the cloud in its optical frame; tf gives base_link<-optical in ONE hop. No depth unprojection,
    no world pivot.

    Args:
        points_np: (N,3) float32, points in the cloud's own frame (msg.header.frame_id).
        base_from_cloud_pos_np: (3,) translation of tf base_link<-cloud_frame.
        base_from_cloud_quat_wxyz_np: (4,) quaternion (wxyz) of tf base_link<-cloud_frame.
        bounds_min_np / bounds_max_np: (3,) action-box crop bounds in base frame (MUST match training).
        num_points: FPS target count.
        pole_x_gt / pole_y_lt: if both set, drop points with x > pole_x_gt AND y < pole_y_lt (base
            frame) after the crop: the front-left rack pole clips the box corner, sim training
            clouds have no pole there, and its few points hijack the PointNet max-pool.
        pole2_x_gt / pole2_y_gt: same for the front-RIGHT pole (x > pole2_x_gt AND y > pole2_y_gt).
        ror_radius / ror_min_neighbors: if both set, radius-outlier removal on the sampled cloud -
            zero out any point with fewer than ror_min_neighbors other points within ror_radius (m).
            A lone isolated point (sensor speck, pole tip) otherwise wins the PointNet max-pool and
            rails the target to it; ROR removes it wherever it lands (unlike the fixed pole boxes,
            which the point drifts out of after a recal / with slew). Isolation-keyed, so it leaves
            dense pile tops untouched (validated 0mm top-drop on sim + real mound clouds).

    Returns:
        (num_points * 3,) float32 tensor of base-frame XYZ points, flat.
    """
    pts = torch.from_numpy(points_np).float()
    pos = torch.from_numpy(base_from_cloud_pos_np).float()
    quat = torch.from_numpy(base_from_cloud_quat_wxyz_np).float()
    pc_base = transform_points(pts, pos, quat)            # cloud frame -> base frame (one tf hop)

    bmin = torch.from_numpy(bounds_min_np).float()
    bmax = torch.from_numpy(bounds_max_np).float()
    m = ((pc_base >= bmin) & (pc_base <= bmax)).all(dim=1)   # crop to the action box
    if pole_x_gt is not None and pole_y_lt is not None:
        m &= ~((pc_base[:, 0] > pole_x_gt) & (pc_base[:, 1] < pole_y_lt))
    if pole2_x_gt is not None and pole2_y_gt is not None:
        m &= ~((pc_base[:, 0] > pole2_x_gt) & (pc_base[:, 1] > pole2_y_gt))
    if pole3_x_lt is not None and pole3_y_lt is not None:
        # BACK-left rack pole (far x side, so it escapes the front filters' x > conditions).
        # Measured 2026-07-29: x in [-5.36,-5.26], reaching the crop ceiling; as a tall thin
        # column it wins the max-pool and hijacked the harvest auto-label the same day.
        m &= ~((pc_base[:, 0] < pole3_x_lt) & (pc_base[:, 1] < pole3_y_lt))
    pc_base = pc_base[m]
    if pc_base.shape[0] == 0:
        return torch.zeros(num_points * 3, dtype=torch.float32)
    sampled = farthest_point_sampling(pc_base, num_points)   # (num_points, 3), zero-padded if sparse
    if ror_radius is not None and ror_min_neighbors is not None:
        nz = sampled.abs().sum(dim=1) > 0                     # skip zero-padding rows
        if int(nz.sum()) > ror_min_neighbors:
            pts = sampled[nz]
            cnt = (torch.cdist(pts, pts) <= ror_radius).sum(dim=1) - 1   # neighbors within radius (excl self)
            drop = cnt < ror_min_neighbors
            if bool(drop.any()):
                idx = torch.nonzero(nz, as_tuple=False).squeeze(1)
                sampled[idx[drop]] = 0.0                      # zero out isolated points (max-pool hijackers)
    return sampled.reshape(-1)


def process_depth(depth_np, intrinsics_np, cam_pos_np, cam_quat_wxyz_np,
                  base_pos_np, base_quat_wxyz_np,
                  num_points=1024, depth_range=(1.0, 10.0)):
    """Full pipeline: depth image to flattened base-frame point cloud.

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
        (num_points * 3,) float32 tensor of base-frame XYZ points, flat.
    """
    device = torch.device("cpu")

    depth = torch.from_numpy(depth_np).float().to(device)
    intrinsics = torch.from_numpy(intrinsics_np).float().to(device)
    cam_pos = torch.from_numpy(cam_pos_np).float().to(device)
    cam_quat = torch.from_numpy(cam_quat_wxyz_np).float().to(device)
    base_pos = torch.from_numpy(base_pos_np).float().to(device)
    base_quat = torch.from_numpy(base_quat_wxyz_np).float().to(device)

    min_d, max_d = depth_range
    depth_mask = (depth >= min_d) & (depth <= max_d) & (~torch.isinf(depth))
    filtered_depth = torch.where(depth_mask, depth, torch.tensor(float('inf'), device=device))

    points_cam = unproject_depth(filtered_depth, intrinsics)
    points_world = transform_points(points_cam, cam_pos, cam_quat)

    valid = torch.all(torch.isfinite(points_world), dim=1)
    points_world = points_world[valid]
    if points_world.shape[0] == 0:
        return torch.zeros(num_points * 3, dtype=torch.float32)

    # Cap point count before the base-frame transform for speed.
    if points_world.shape[0] > 5000:
        idx = torch.randperm(points_world.shape[0])[:5000]
        points_world = points_world[idx]

    points_base = world_to_base_frame(points_world, base_pos, base_quat)
    points_sampled = farthest_point_sampling(points_base, num_points)
    return points_sampled.view(-1).float()
