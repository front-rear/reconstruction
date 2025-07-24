import os
import shutil
import numpy as np
import open3d as o3d

import constants
from util import PoseEstimator
from run_rodin import run_rodin, glb_to_obj
from run_oops import run_oops, adjust_scale, refine_pose, calculate_mask_error, visualize_pose
from run_SAM3D import run_SAM3D_render, run_SAM3D_inference, extract_SAM3D_results
from run_REART import segment_mesh, run_REART, extract_urdf
from record_affordance import record_affordance

def transform_mesh(mesh_path: str, mean: np.ndarray, scale: float, transform_matrix: np.ndarray):
    """
    Transform the mesh according to the given transform matrix and scale.
    """

    # Load mesh (real frame)
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    # Convert to canonical frame
    mesh.translate(-mean)
    mesh.scale(scale, center=np.zeros(3))

    # Transform
    mesh.transform(transform_matrix)

    # Convert to real frame
    mesh.scale(1.0 / scale, center=np.zeros(3))
    mesh.translate(mean)

    # Save mesh
    o3d.io.write_triangle_mesh(mesh_path, mesh)

def conduct_object_transform(dataset_folder: str, dst_frame: int = -1):
    REART_path = os.path.join(dataset_folder, "REART")
    data: np.ndarray = np.load(os.path.join(REART_path, "raw_output.pkl"), allow_pickle=True)["pred_pose_list"]  # (part_num, frame_num, 4, 4)

    # Load transform
    transform_data = np.load(os.path.join(REART_path, "pcd_transform.npz"), allow_pickle=True)
    mean: np.ndarray = transform_data["mean"]
    scale: float = transform_data["scale"]
    transform_data.close()

    # Copy original urdf
    new_urdf_path = os.path.join(REART_path, "urdf_adjusted")
    shutil.copytree(os.path.join(REART_path, "urdf"), new_urdf_path, dirs_exist_ok=True)

    # Transform mesh according to pose
    for part_idx in range(data.shape[1]):
        dst_pose = data[dst_frame, part_idx]
        # dst_pose = np.eye(4)

        for obj_name in os.listdir(os.path.join(new_urdf_path, "collision")):
            obj_path = os.path.join(new_urdf_path, "collision", obj_name)
            if int(obj_name.split("_")[1]) == part_idx:
                transform_mesh(obj_path, mean, scale, dst_pose)

        transform_mesh(os.path.join(new_urdf_path, "visual", f"part_{part_idx}", "mesh.obj"), mean, scale, dst_pose)

                

if __name__ == '__main__':

    # base_input_folder = "/home/rvsa/gary318/build_kinematic/input_rgbd"
    dataset_folder = "/home/rvsa/gary318/build_kinematic/input_rgbd/compound_011701"

    # datasets = [os.path.join(base_input_folder, "laptop_gray_01090%d") % i for i in range(1, 6)]
    #            [os.path.join(base_input_folder, "laptop_black_01090%d") % i for i in range(1, 6)]

    # datasets = [os.path.join(base_input_folder, "faucet_square_01090%d") % i for i in range(1, 6)]

    mesh_folder = os.path.join(dataset_folder, "mesh")
    render_folder = os.path.join(dataset_folder, "render")
    SAM3D_folder = os.path.join(dataset_folder, "SAM3D")

    # SAM & Affordance
    # os.chdir(constants.SAM_PATH)
    # os.system(f"python {constants.SAM_SCRIPT_PATH} " +
    #           f"--img_in_path={os.path.join(dataset_folder, 'raw_rgb')} " +
    #           f"--img_out_path={os.path.join(dataset_folder, 'mask')}"
    # )
    # record_affordance(dataset_folder)

    # Process mask
    # est = PoseEstimator()
    # est.set_dataset(dataset_folder)
    # est.generate_masked_rgbd()

    # Run Rodin
    # run_rodin(os.path.join(dataset_folder, "color_cropped"), mesh_folder)
    # glb_to_obj(mesh_folder, os.path.join(dataset_folder, "objs"))

    # # Run OOPS
    # run_oops(dataset_folder)
    # adjust_scale(dataset_folder, display_hist=False)
    # refine_pose(dataset_folder, display_icp=False)
    # calculate_mask_error(dataset_folder)
    visualize_pose(dataset_folder)

    # # SAM3D
    # run_SAM3D_render(mesh_folder, render_folder)
    # run_SAM3D_inference(render_folder, SAM3D_folder)
    # run_SAM3D_render(mesh_folder, render_folder, pick_filename=["state_3.glb"])
    # run_SAM3D_inference(render_folder, SAM3D_folder, pick_filename=["state_3"])

    # # Movable Part Segmentation
    # extract_SAM3D_results(SAM3D_folder)
    # while True:
    #     segment_mesh(dataset_folder, show_difference=False, show_final_pcd=True, canonical_idx=3)
    #     if input("Break? y / [n]").lower() == "y":
    #         break

    # # REART
    # while True:
    #     # run_REART(dataset_folder, ablation=False, canonical_idx=3)
    #     all_revolute_joints, all_connect_to_root, ret_dict = extract_urdf(dataset_folder)
    #     if all_connect_to_root or True:
    #         # if ret_dict["has_gt"] and ret_dict["axis_rotation_error"] > 7:
    #         #     print("Axis rotation error is too large, try again.")
    #         #     continue
    #         break

    # conduct_object_transform(dataset_folder)
