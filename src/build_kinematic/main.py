import os
import logging
import numpy as np

import constants
from util import PoseEstimator
from run_rodin import run_rodin, glb_to_obj
from run_oops import run_oops, adjust_scale, refine_pose, calculate_mask_error, visualize_pose
from run_SAM3D import run_SAM3D_render, run_SAM3D_inference, extract_SAM3D_results
from run_REART import segment_mesh, run_REART, extract_urdf
from record_affordance import record_affordance

if __name__ == '__main__':

    # base_input_folder = "/home/rvsa/gary318/build_kinematic/input_rgbd"
    dataset_folder = "/home/rvsa/gary318/build_kinematic/input_rgbd/fig1_011801"

    mesh_folder = os.path.join(dataset_folder, "mesh")
    render_folder = os.path.join(dataset_folder, "render")
    SAM3D_folder = os.path.join(dataset_folder, "SAM3D")

    # SAM & Affordance
    os.chdir(constants.SAM_PATH)
    os.system(f"python {constants.SAM_SCRIPT_PATH} " +
              f"--img_in_path={os.path.join(dataset_folder, 'raw_rgb')} " +
              f"--img_out_path={os.path.join(dataset_folder, 'mask')}"
    )
    record_affordance(dataset_folder)

    # Process mask
    est = PoseEstimator()
    est.set_dataset(dataset_folder)
    est.generate_masked_rgbd()

    # Run Rodin
    run_rodin(os.path.join(dataset_folder, "color_cropped"), mesh_folder)
    glb_to_obj(mesh_folder, os.path.join(dataset_folder, "objs"))

    # Run OOPS
    run_oops(dataset_folder)
    adjust_scale(dataset_folder, display_hist=False)
    refine_pose(dataset_folder, display_icp=False)
    calculate_mask_error(dataset_folder)
    # visualize_pose(dataset_folder)

    # SAM3D
    run_SAM3D_render(mesh_folder, render_folder)
    run_SAM3D_inference(render_folder, SAM3D_folder)

    # Movable Part Segmentation
    extract_SAM3D_results(SAM3D_folder)
    segment_mesh(dataset_folder, show_difference=False, show_final_pcd=False)

    # REART
    run_REART(dataset_folder, ablation=False)
    all_revolute_joints, all_connect_to_root, ret_dict = extract_urdf(dataset_folder)

