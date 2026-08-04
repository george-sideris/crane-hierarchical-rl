import numpy as np
from scipy.interpolate import CubicSpline


DEFAULT_RESOLUTION = 0.01 # meters


class PathPlanner:
    """"""
    def __init__(self, default_resolution=DEFAULT_RESOLUTION):
        self.default_resolution = default_resolution

    def plan_linear_path(self, waypoints: np.ndarray, resolution: float = None):
        """
        Generates a linear piecewise path through multiple waypoints in task space, including yaw and grapple opening.

        Args:
            waypoints (np.ndarray): [num_waypoints, 5] array of waypoints, each being [x, y, z, yaw, opening]
            resolution (float, optional): Interpolation resolution override (in meters).

        Returns:
            np.ndarray: [num_steps, 5] array of interpolated poses
        """
        waypoints = np.asarray(waypoints)
        assert waypoints.shape[1] == 5, "Each waypoint must be [x, y, z, yaw, opening]"
        resolution = resolution if resolution else self.default_resolution
        path = []
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]
            distance = np.linalg.norm(end[:3] - start[:3])
            steps = max(2, int(distance / resolution))
            segment = np.linspace(start, end, steps, endpoint=False)
            path.append(segment)
        path.append(waypoints[-1][None, :])
        return np.vstack(path)
    
    def plan_cubic_spline_path(self, waypoints: np.ndarray, resolution: float = None):
        """
        Generates a smooth cubic spline path through multiple task-space waypoints, including yaw and grapple opening.

        Args:
            waypoints (np.ndarray): [num_waypoints, 5] array of waypoints where each waypoint is [x, y, z, yaw, opening]
            resolution (float, optional): Interpolation resolution override (in meters).

        Returns:
            np.ndarray: [num_steps, 5] array of interpolated poses along the spline,
        """
        waypoints = np.asarray(waypoints)
        assert waypoints.shape[1] == 5, "Expected [x, y, z, yaw, opening] format"
        t = np.linspace(0, 1, len(waypoints))
        splines = [CubicSpline(t, waypoints[:, i]) for i in range(5)]
        total_length = np.sum(np.linalg.norm(np.diff(waypoints[:, :3], axis=0), axis=1))
        resolution = resolution if resolution else self.default_resolution
        steps = max(2, int(total_length / resolution))
        t_interp = np.linspace(0, 1, steps)
        path = np.stack([spline(t_interp) for spline in splines], axis=-1)
        return path
    