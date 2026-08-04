import numpy as np
from scipy.spatial.transform import Rotation as R

z_up_imu = np.array([-0.33473503,  0.00869411,  0.94227219])  # gravity up in IMU frame
z_grapple_base = np.array([0.43184146809505114, -0.06953385368484172, 0.8988766136524025])  # grapple z-axis in base frame
q_imu_camera = np.array([0.001, 0.000, -0.005, 1.000])  # IMU to camera rotation quaternion
q_camera_base = np.array([-0.1643474567930718,-0.0017434173405340093,0.9679964641991942,-0.1896568460126983])  # camera to base rotation quaternion

if __name__ == "__main__":
    # Convert quaternions to rotation matrices
    R_imu_camera = R.from_quat(q_imu_camera).as_matrix()
    R_camera_base = R.from_quat(q_camera_base).as_matrix()

    # Transform z_up_imu to base frame
    z_up_base = R_camera_base @ (R_imu_camera @ z_up_imu)

    # Normalize the vectors
    z_up_base /= np.linalg.norm(z_up_base)
    z_grapple_base /= np.linalg.norm(z_grapple_base)

    # Compute the angle between the two vectors
    dot_product = np.clip(np.dot(z_up_base, z_grapple_base), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)

    print(f"Angle between IMU up and grapple z-axis in base frame: {angle_deg:.2f} degrees")
