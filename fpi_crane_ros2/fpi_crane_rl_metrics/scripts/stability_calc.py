import numpy as np
from scipy.spatial.transform import Rotation


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)

    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Cannot normalize zero or invalid vector")

    return vector / norm


def gravity_from_acceleration(
    acceleration_imu: np.ndarray,
    R_base_imu: np.ndarray,
    acceleration_sign: float = 1.0,
) -> np.ndarray:
    """
    Estimate gravity-up in base_link using accelerometer data.

    acceleration_sign should be determined experimentally:
      +1 if normalized acceleration points up
      -1 if normalized acceleration points down
    """
    z_up_imu = acceleration_sign * normalize(acceleration_imu)
    return normalize(R_base_imu @ z_up_imu)


def gravity_from_orientation(
    quaternion_xyzw: np.ndarray,
    R_base_imu: np.ndarray,
) -> np.ndarray:
    """
    Estimate gravity-up in base_link using the fused IMU orientation.

    Assumes the quaternion represents IMU orientation in an
    Earth frame whose +z direction is up.
    """
    R_earth_imu = Rotation.from_quat(
        quaternion_xyzw
    ).as_matrix()

    z_up_earth = np.array([0.0, 0.0, 1.0])

    # Express Earth-up in the IMU frame.
    z_up_imu = R_earth_imu.T @ z_up_earth

    # Express it in base_link.
    return normalize(R_base_imu @ z_up_imu)

if __name__ == "__main__":
    # Test that the functions run without error.
    R_base_imu = np.eye(3)
    acceleration_imu = np.array([-0.11213520914316177, -0.07015224546194077, 9.828349113464355])
    quaternion_xyzw = np.array([-0.06682755798101425,0.1570015400648117,0.4094937741756439,0.896213948726654])

    z_up_accel = gravity_from_acceleration(acceleration_imu, R_base_imu)
    z_up_orientation = gravity_from_orientation(quaternion_xyzw, R_base_imu)

    # Rotation matrix from quaternion
    R_earth_imu = Rotation.from_quat(quaternion_xyzw).as_matrix()

    print("Gravity-up from acceleration:", z_up_accel)
    print("Gravity-up from orientation:", z_up_orientation)

    differnece_angle = np.arccos(np.clip(np.dot(z_up_accel, z_up_orientation), -1.0, 1.0))
    print("Angle difference between the two estimates (radians):", differnece_angle)
    print("Angle difference between the two estimates (degrees):", np.degrees(differnece_angle))

    print("Rotation matrix from quaternion:\n", R_earth_imu)
