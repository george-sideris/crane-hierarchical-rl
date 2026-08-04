import open3d as o3d
import numpy as np
import math
import copy
import os
import pandas as pd
import concurrent.futures
from datetime import datetime
from scipy.spatial import KDTree
from scipy.optimize import minimize
import argparse

# Load the gltf file
BASE_FOLDER = "/home/wilah/datasets/wilah_mtl_visit_grapple_data/20250708-0840"

LEFT_GRIPPER_PATH = "data/original_models/fpi_size_gltf/left_thong_without_mesh-good_orientation_good_scale.gltf"
RIGHT_GRIPPER_PATH = "data/original_models/fpi_size_gltf/right_thong_without_mesh-good_orientation_good_scale.gltf"
CENTER_PIECE_PATH = "data/original_models/fpi_size_gltf/fpi_scale_baselink_without_mesh-fpi_cap_plus_center.gltf"

UPPER_ROT = 32.54782793
LOWER_ROT = 10.04996516
BASE_ROT = 45.25767816
GRIPPER_ROT = -15
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
LOGGING_LEVEL = 1
DEBUG_LEVEL = 0
RAYTRACING = False
MAXFUN = 100000
MAXITER = 100000
FTOL = 1e-15
XTOL = 1e-15
EPS = 1e-4
VOXEL_DOWNSAMPLE = 0.001
MAX_WORKERS = 28

# TODO WLH: Déterminer les limites de mes axes
# TODO WLH: Avoir le code pour projeter a partir d'un point de vue
# TODO WLH: Avoir une métrique d'erreur (voir paper de Will D - Relative pose error (RPE)?)
# TODO WLH: Faire une version wish de ICP
# TODO WLH: Tracer les liens entre les matchs pour ICP
# TODO WLH: Faire une animation de l'ICP pour passer d'une itération à l'autre
# TODO WLH: Tester plusieurs solveurs
# TODO WLH: Faire scaler le point cloud

def parse_args():
    parser = argparse.ArgumentParser(description="Run model-based calibration")
    parser.add_argument(
        "folder",
        nargs="?",
        default=BASE_FOLDER,
        help=f"Base folder (default: {BASE_FOLDER})"
    )
    return parser.parse_args()


def print_with_timestamp(message):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp} : {message}")


def forward_kinematics_with_encoders(angles, encoder_angles):
    upper_rot, lower_rot, base_rot, gripper_rot, cam_roll, cam_pitch, cam_yaw, cam_x, cam_y, cam_z = angles
    upper_enc, lower_enc, base_enc, gripper_enc = encoder_angles
    cam_x = cam_x / 1000
    cam_y = cam_y / 1000
    cam_z = cam_z / 1000
    # Define the transformation matrices
    zedx_left_2_zedx_opt_left = np.eye(4)
    zedx_left_2_zedx_opt_left[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((math.radians(-90), math.radians(90), math.radians(0)))

    zedx_center_2_zedx_left = np.eye(4)
    zedx_center_2_zedx_left[0, 3] = -0.01
    zedx_center_2_zedx_left[1, 3] = 0.06

    zedx_base_2_zedx_center = np.eye(4)
    zedx_base_2_zedx_center[2, 3] = 0.016

    camera_2_zedx_base = np.eye(4)
    camera_2_zedx_base[0, 3] = cam_x
    camera_2_zedx_base[1, 3] = cam_y
    camera_2_zedx_base[2, 3] = cam_z
    camera_2_zedx_base[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((math.radians(cam_roll), math.radians(cam_pitch), math.radians(cam_yaw)))

    stick_2_camera = np.eye(4)
    stick_2_camera[1, 3] = 0.37
    stick_2_camera[2, 3] = 0.03

    stick_2_telescope = np.eye(4)
    stick_2_telescope[1, 3] = 0.255
    stick_2_telescope[2, 3] = 0.21

    telescope_2_upper = np.eye(4)
    telescope_2_upper[1, 3] = 2.037
    telescope_2_upper[2, 3] = -0.202
    telescope_2_upper[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((math.radians(upper_rot + upper_enc), math.radians(0), math.radians(0)))

    upper_2_lower = np.eye(4)
    upper_2_lower[2, 3] = -0.154
    upper_2_lower[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((math.radians(0), math.radians(lower_rot + lower_enc), math.radians(0)))

    lower_2_base_grapple = np.eye(4)
    lower_2_base_grapple[2, 3] = -0.261
    lower_2_base_grapple[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((math.radians(0), math.radians(0), math.radians(base_rot + base_enc)))

    base_grapple_2_left_gripper = np.eye(4)
    base_grapple_2_left_gripper[0, 3] = -0.275
    base_grapple_2_left_gripper[1, 3] = -0.019
    base_grapple_2_left_gripper[2, 3] = -0.202
    base_grapple_2_left_gripper[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((math.radians(0), math.radians(-gripper_rot), math.radians(0)))

    base_grapple_2_right_gripper = np.eye(4)
    base_grapple_2_right_gripper[0, 3] = 0.275
    base_grapple_2_right_gripper[1, 3] = -0.019
    base_grapple_2_right_gripper[2, 3] = -0.202
    base_grapple_2_right_gripper[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((math.radians(0), math.radians(gripper_rot), math.radians(0)))

    zedx_opt_left_2_zedx_left = np.linalg.inv(zedx_left_2_zedx_opt_left)
    zedx_left_2_zedx_center = np.linalg.inv(zedx_center_2_zedx_left)
    zedx_center_2_zedx_base = np.linalg.inv(zedx_base_2_zedx_center)
    zedx_base_2_camera = np.linalg.inv(camera_2_zedx_base)
    camera_2_stick = np.linalg.inv(stick_2_camera)
    
    transform_base = zedx_opt_left_2_zedx_left @ zedx_left_2_zedx_center @ zedx_center_2_zedx_base @ zedx_base_2_camera @ camera_2_stick @ stick_2_telescope @ telescope_2_upper @ upper_2_lower @ lower_2_base_grapple
    transform_left_gripper = transform_base @ base_grapple_2_left_gripper
    transform_right_gripper = transform_base @ base_grapple_2_right_gripper

    return transform_base, transform_left_gripper, transform_right_gripper

def render_grapple_pc(transform_base, transform_left_gripper, transform_right_gripper):
    # Load the gltf files
    left_gripper = o3d.io.read_triangle_mesh(LEFT_GRIPPER_PATH)
    right_gripper = o3d.io.read_triangle_mesh(RIGHT_GRIPPER_PATH)
    center_piece = o3d.io.read_triangle_mesh(CENTER_PIECE_PATH)

    # Rotate the grippers 90 deg around the x axis because the model is not oriented correctly TODO WLH
    left_gripper.rotate(left_gripper.get_rotation_matrix_from_xyz((math.radians(90), 0, 0)), center=(0, 0, 0))
    right_gripper.rotate(right_gripper.get_rotation_matrix_from_xyz((math.radians(90), 0, 0)), center=(0, 0, 0))
    center_piece.rotate(center_piece.get_rotation_matrix_from_xyz((math.radians(90), 0, 0)), center=(0, 0, 0))

    ref_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    left_gripper.transform(transform_left_gripper)
    right_gripper.transform(transform_right_gripper)
    center_piece.transform(transform_base)

    return ref_frame, left_gripper, right_gripper, center_piece

def mesh_to_pc(mesh):
    """Convert a mesh to a point cloud by sampling points uniformly."""
    pc = o3d.geometry.PointCloud()
    pc.points = mesh.sample_points_uniformly(number_of_points=5000).points
    return pc

def mesh_list_to_pc(mesh_list):
    pc = o3d.geometry.PointCloud()
    for mesh in mesh_list:
        pc += mesh.sample_points_uniformly(number_of_points=5000)
    return pc

def aggregate_point_clouds(point_clouds):
    """Aggregate multiple point clouds into a single point cloud with the colors."""
    combined_pc = o3d.geometry.PointCloud()
    for pc in point_clouds:
        combined_pc += pc
    return combined_pc

def nearest_neighbors(source, target):
    """Find the nearest neighbors of source points in the target cloud using a KDTree."""
    tree = KDTree(target)
    distances, indices = tree.query(source)
    return indices, distances

def _compute_error_single_target_dense(args):
    angles, source_data, target_data, enc, logging_level, debug_level = args
    pc_base_pts, pc_left_pts, pc_right_pts = source_data
    target_pts = target_data

    pc_base = o3d.geometry.PointCloud()
    pc_left_gripper = o3d.geometry.PointCloud()
    pc_right_gripper = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()

    pc_base.points = o3d.utility.Vector3dVector(pc_base_pts)
    pc_left_gripper.points = o3d.utility.Vector3dVector(pc_left_pts)
    pc_right_gripper.points = o3d.utility.Vector3dVector(pc_right_pts)
    target.points = o3d.utility.Vector3dVector(target_pts)
    transform_base, transform_left_gripper, transform_right_gripper = forward_kinematics_with_encoders(angles, enc)

    pc_base.transform(transform_base)
    pc_left_gripper.transform(transform_left_gripper)
    pc_right_gripper.transform(transform_right_gripper)

    total_pc = aggregate_point_clouds([pc_base, pc_left_gripper, pc_right_gripper])

    transformed_source = np.asarray(total_pc.points)
    indices, distances = nearest_neighbors(transformed_source, np.asarray(target.points))
    err = np.sum(distances ** 2)

    if transformed_source.size == 0:
        err = np.inf

    if logging_level > 1:
        print_with_timestamp(f"Angles: {angles}")
        print_with_timestamp(f"Distance: {err}")

    if debug_level > 1:
        total_pc.paint_uniform_color([0, 1, 0])  # Green
        ref_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        o3d.visualization.draw_geometries(
            [total_pc, target, ref_frame],
            lookat=[0, 0, 0],
            up=[0, -1, 0],
            front=[0, 0, -1],
            zoom=0.5
        )
    return err

def cost_function_dense(angles, sources, target_list, enc_list, logging_level=0, debug_level=0):
    sources_data = (
        np.asarray(sources[0].points),
        np.asarray(sources[1].points),
        np.asarray(sources[2].points)
    )

    args_list = []
    for target, enc in zip(target_list, enc_list):
        target_data = np.asarray(target.points)
        args_list.append((angles, sources_data, target_data, enc, logging_level, debug_level))

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        err_list = list(executor.map(_compute_error_single_target_dense, args_list))

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        err_list = list(executor.map(_compute_error_single_target_dense, args_list))

    err_total = np.sum(err_list)
    if logging_level > 0:
        print_with_timestamp(f"Total error: {err_total} over {len(err_list)} targets for an average of {err_total / len(err_list):.6f}")
    return err_total

def cost_function_raytracing(angles, sources_mesh, target_list, enc_list, logging_level=0, debug_level=0):
    err_list = []
    for target, enc in zip(target_list, enc_list):
        base_mesh, left_gripper_mesh, right_gripper_mesh = sources_mesh
        transform_base, transform_left_gripper, transform_right_gripper = forward_kinematics_with_encoders(angles, enc)
        base_mesh_tmp = copy.deepcopy(base_mesh)
        left_gripper_mesh_tmp = copy.deepcopy(left_gripper_mesh)
        right_gripper_mesh_tmp = copy.deepcopy(right_gripper_mesh)
        base_mesh_tmp.transform(transform_base)
        left_gripper_mesh_tmp.transform(transform_left_gripper)
        right_gripper_mesh_tmp.transform(transform_right_gripper)
        # Load the camera intrinsic matrix
        intrinsic_matrix = np.loadtxt(INTRINSIC_MATRIX_PATH)
        width, height = IMAGE_WIDTH, IMAGE_HEIGHT
        fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
        intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        camera_pose = np.eye(4)
        scene = o3d.t.geometry.RaycastingScene()
        # Ensure the meshes are valid before adding them to the scene
        base_mesh_tmp = base_mesh_tmp.compute_vertex_normals()
        left_gripper_mesh_tmp = left_gripper_mesh_tmp.compute_vertex_normals()
        right_gripper_mesh_tmp = right_gripper_mesh_tmp.compute_vertex_normals()

        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(base_mesh_tmp))
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(left_gripper_mesh_tmp))
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(right_gripper_mesh_tmp))

        intrinsic_tensor = o3d.core.Tensor(intrinsics.intrinsic_matrix, dtype=o3d.core.Dtype.Float32)
        extrinsic_tensor = o3d.core.Tensor(np.linalg.inv(camera_pose), dtype=o3d.core.Dtype.Float32)

        # Create an Open3D camera object
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_tensor, 
            extrinsic_tensor,
            IMAGE_WIDTH, 
            IMAGE_HEIGHT
        )

        # Perform raycasting to generate the depth map
        raycasting_results = scene.cast_rays(rays)
        depth_map = raycasting_results["t_hit"].numpy().reshape((IMAGE_HEIGHT, IMAGE_WIDTH))

        depth_o3d = o3d.geometry.Image((depth_map * 1000).astype(np.uint16))  # convert to mm for Open3D
        rgb_o3d = o3d.geometry.Image(np.zeros((height, width, 3), dtype=np.uint8))  # dummy image

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb_o3d,
            depth=depth_o3d,
            depth_scale=1000.0,
            depth_trunc=5.0,
            convert_rgb_to_intensity=False
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            intrinsics,
            extrinsic=camera_pose
        )

        transformed_source = np.asarray(pcd.points)
        indices, distances = nearest_neighbors(transformed_source, np.asarray(target.points))
        err = np.sum(distances ** 2)
        if transformed_source.size == 0:
            err = np.inf
        if logging_level > 1:
            print_with_timestamp(f"Angles: {angles}")
            print_with_timestamp(f"Distance: {err}")
        if debug_level > 1:
            pcd.paint_uniform_color([0, 1, 0])  # Green
            ref_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            o3d.visualization.draw_geometries(
                [pcd, target, ref_frame],
                lookat=[0, 0, 0],  # Center of the scene
                up=[0, -1, 0],      # Align with the z-axis
                front=[0, 0, -1],  # Align with the x-axis
                zoom=0.5           # Adjust zoom level
            )
        err_list.append(err)
    err_total = np.sum(err_list)
    if logging_level > 0:
        print_with_timestamp(f"Total error: {err_total} over {len(err_list)} targets for an average of {err_total / len(err_list):.6f}")
    return err_total

def compute_initial_costs(sources, target_list, enc_list, initial_angles, num_samples=1, num_candidates=1):
    """Generate num_samples perturbations and compute their errors then return the num_candidates best candidates."""
    candidates = []
    for i in range(num_samples):
        print_with_timestamp(f"Generating candidate {i}")
        perturbed_guess = np.copy(initial_angles)
        if RAYTRACING:
            cost = cost_function_raytracing(perturbed_guess, sources, target_list, enc_list, LOGGING_LEVEL, DEBUG_LEVEL)
        else:
            cost = cost_function_dense(perturbed_guess, sources, target_list, enc_list, LOGGING_LEVEL, DEBUG_LEVEL)
        candidates.append((cost, perturbed_guess))
    
    # Sort by lowest error and return the top num_candidates
    candidates.sort(key=lambda x: x[0])
    return [x[1] for x in candidates[:num_candidates]]

def optimize_joint_angles_dense(sources, target_list, enc_list, initial_angles):
    """Optimize joint angles to minimize the alignment error with tighter tolerances."""
    best_solutions = compute_initial_costs(sources, target_list, enc_list, initial_angles)
    best_result = None
    best_cost = np.inf

    for guess in best_solutions:
        print_with_timestamp(f"Starting optimization with initial guess {guess}")
        result = minimize(
            cost_function_dense,
            guess,
            args=(sources, target_list, enc_list, LOGGING_LEVEL, DEBUG_LEVEL),
            method='L-BFGS-B',
            options={'ftol': FTOL, 'xtol': XTOL, 'maxiter': MAXITER, 'maxfun': MAXFUN, 'eps': EPS}
        )
        print_with_timestamp(f"Optimization result: {result.x}, Cost: {result.fun:.6f}")
        
        if result.fun < best_cost:
            best_result = result.x
            best_cost = result.fun
    
    print_with_timestamp(f"Best solution found: {best_result} with average cost {best_cost/len(target_list):.6f}")
    return best_result, best_cost/len(target_list)

def optimize_joint_angles_raytracing(sources, target_list, enc_list, initial_angles):
    """Optimize joint angles to minimize the alignment error with tighter tolerances."""
    best_solutions = compute_initial_costs(sources, target_list, enc_list, initial_angles)
    best_result = None
    best_cost = np.inf

    for guess in best_solutions:
        print_with_timestamp(f"Starting optimization with initial guess {guess}")
        result = minimize(
            cost_function_raytracing,
            guess,
            args=(sources, target_list, enc_list, LOGGING_LEVEL, DEBUG_LEVEL),
            method='L-BFGS-B',
            options={'ftol': FTOL, 'xtol': XTOL, 'maxiter': MAXITER, 'maxfun': MAXFUN, 'eps': EPS}
        )
        print_with_timestamp(f"Optimization result: {result.x}, Cost: {result.fun:.6f}")
        
        if result.fun < best_cost:
            best_result = result.x
            best_cost = result.fun
    
    print_with_timestamp(f"Best solution found: {best_result} with average cost {best_cost/len(target_list):.6f}")
    return best_result, best_cost/len(target_list)

def fix_df(csv_path):
    df = pd.read_csv(csv_path)
    df['position'] = df['position'].apply(lambda x: list(map(float, x.strip('[]').split())))
    df['name'] = df['name'].apply(lambda x: list(map(str, x.strip('[]').split())))
    df['velocity'] = df['velocity'].apply(lambda x: list(map(float, x.strip('[]').split())))
    df['effort'] = df['effort'].apply(lambda x: list(map(float, x.strip('[]').split())))
    return df
    
def create_joint_dict(df, file_path=None):
    joint_dict = {}
    for index, row in df.iterrows():
        joint_dict[row['timestamp']] = {
            'upper_joint': math.degrees(row['position'][4]),
            'lower_joint': math.degrees(row['position'][5]),
            'base_joint': math.degrees(row['position'][6]),
            'telescope_joint': row['position'][3],
            'gripper_joint': 0,
            'basemast_to_mast': math.degrees(row['position'][0]),
            'mast_to_mainboom': math.degrees(row['position'][1]),
            'stick_to_telescope': math.degrees(row['position'][2]),
        }
    if file_path is not None:
        joint_df = pd.DataFrame.from_dict(joint_dict, orient='index')
        joint_df.index.name = 'timestamp'
        joint_df.reset_index(inplace=True)
        joint_df.to_csv(file_path, index=False)
    return joint_dict

def get_closest_joint_values(joint_dict, timestamp):
    closest_timestamp = min(joint_dict.keys(), key=lambda t: abs(t - timestamp))
    return joint_dict[closest_timestamp]

def main(base_folder):
    global BASE_FOLDER, INTRINSIC_MATRIX_PATH, DATA_FOLDER, OUTPUT_FOLDER_PATH, ROSBAG_CSV_PATH

    # Update global variables that depend on BASE_FOLDER
    BASE_FOLDER = base_folder
    INTRINSIC_MATRIX_PATH = "data/camera_matrices/cam_K_fpi_svo.txt"
    DATA_FOLDER = os.path.join(BASE_FOLDER, "point_cloud_depth_image")
    OUTPUT_FOLDER_PATH = os.path.join(BASE_FOLDER, "calibration_results")
    ROSBAG_CSV_PATH = os.path.join(BASE_FOLDER, "joint_states", "joint_states.csv")

    # Create a csv to store the results
    # Create the output directory with the current date
    if not os.path.exists(OUTPUT_FOLDER_PATH):
        os.makedirs(OUTPUT_FOLDER_PATH)
    # Create the output directory with the current date
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(OUTPUT_FOLDER_PATH, datetime_str)
    os.makedirs(output_dir, exist_ok=True)
    # Create the csv file
    csv_file_path = os.path.join(output_dir, "calib_results.csv")
    with open(csv_file_path, 'w') as f:
        f.write("timestamp,upper,lower,base,gripper,pitch,roll,yaw,x,y,z,average_cost\n")

    # List all .ply files in the DATA_FOLDER
    ply_files = [os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER) if f.endswith('.ply')]
    print_with_timestamp(f"Found {len(ply_files)} .ply files in {DATA_FOLDER}:")
    # Sort the files by name
    ply_files.sort()

    joint_dict = create_joint_dict(fix_df(ROSBAG_CSV_PATH), file_path=os.path.join(OUTPUT_FOLDER_PATH, "original_joint_values.csv"))

    def process_file_list(file_list):
        enc_list = []
        target_list = []
        for file in file_list:
            timestamp = float(os.path.splitext(os.path.basename(file))[0])
            closest_joint_values = get_closest_joint_values(joint_dict, timestamp)
            angles = [
                closest_joint_values['upper_joint'],
                closest_joint_values['lower_joint'],
                closest_joint_values['base_joint'],
                closest_joint_values['gripper_joint']
            ]
            print_with_timestamp(f"File: {file}, Closest Joint Values: {closest_joint_values}")
            enc_list.append(angles)
            point_cloud = o3d.io.read_point_cloud(file)
            if VOXEL_DOWNSAMPLE is not None:
                point_cloud = point_cloud.voxel_down_sample(VOXEL_DOWNSAMPLE)
            point_cloud.rotate(point_cloud.get_rotation_matrix_from_xyz((0, 0, math.radians(180))), center=(0, 0, 0))
            target_list.append(point_cloud)

        initial_center_piece = o3d.io.read_triangle_mesh(CENTER_PIECE_PATH)
        initial_left_gripper = o3d.io.read_triangle_mesh(LEFT_GRIPPER_PATH)
        initial_right_gripper = o3d.io.read_triangle_mesh(RIGHT_GRIPPER_PATH)
        initial_center_piece.rotate(initial_center_piece.get_rotation_matrix_from_xyz((math.radians(90), 0, 0)), center=(0, 0, 0))
        initial_left_gripper.rotate(initial_left_gripper.get_rotation_matrix_from_xyz((math.radians(90), 0, 0)), center=(0, 0, 0))
        initial_right_gripper.rotate(initial_right_gripper.get_rotation_matrix_from_xyz((math.radians(90), 0, 0)), center=(0, 0, 0))
        base_pc = mesh_to_pc(initial_center_piece)
        left_gripper_pc = mesh_to_pc(initial_left_gripper)
        right_gripper_pc = mesh_to_pc(initial_right_gripper)
        
        angles = [0, 0, 0, 0, -4, 0, 90, 40, -20, -120]

        if RAYTRACING:
            optimized_angles, cost = optimize_joint_angles_raytracing((initial_center_piece, initial_left_gripper, initial_right_gripper), target_list, enc_list, angles)
        else:
            optimized_angles, cost = optimize_joint_angles_dense((base_pc, left_gripper_pc, right_gripper_pc), target_list, enc_list, angles)

        # Write the results to the csv file
        with open(csv_file_path, 'a') as f:
            f.write(f"{0},{optimized_angles[0]},{optimized_angles[1]},{optimized_angles[2]},{optimized_angles[3]},{optimized_angles[4]},{optimized_angles[5]},{optimized_angles[6]},{optimized_angles[7]},{optimized_angles[8]},{optimized_angles[9]},{cost:.6f}\n")

    process_file_list(ply_files[::100])
    

if __name__ == "__main__":
    args = parse_args()
    main(args.folder)