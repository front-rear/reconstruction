import os
import logging
import numpy as np
import open3d as o3d

import constants
from util import PoseEstimator
from run_rodin import run_rodin, glb_to_obj
from run_oops import run_oops, adjust_scale, refine_pose, calculate_mask_error, visualize_pose
from run_SAM3D import run_SAM3D_render, run_SAM3D_inference, extract_SAM3D_results
from run_REART import segment_mesh, run_REART, extract_urdf
from record_affordance import record_affordance

from typing import Dict, Any

def draw_pcd_and_axis(dataset_folder: str, ablation: bool):
    reart_path = os.path.join(dataset_folder, "REART")
    ablation_suffix = "_ablation" if ablation else ""

    pcd = o3d.io.read_point_cloud(os.path.join(reart_path, "cano_pc%s.ply" % ablation_suffix))
    pcd.estimate_normals()

    geoms = [pcd]
    connections: Dict[str, Any] = np.load(os.path.join(reart_path, "kinematic_result%s.npz" % ablation_suffix),
                                          allow_pickle=True)["connection_dict"].item()
    for connection in connections.values():
        SIZE = 0.01
        LENGTH = 0.3
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=SIZE,
            cone_radius=SIZE * 1.5,
            cylinder_height=LENGTH,
            cone_height=LENGTH * 0.3
        )
        arrow.compute_vertex_normals()
        if connection["type"] == "revolute":
            arrow.paint_uniform_color([1.0, 0, 1.0])
        elif connection["type"] == "prismatic":
            arrow.paint_uniform_color([0, 1.0, 1.0])

        axis = connection["axis"] / np.linalg.norm(connection["axis"])
        origin = np.cross(axis, connection["moment"])

        # Transform the arrow to the correct position
        T = np.eye(4)
        T[:3, 3] = origin
        T[:3, 2] = axis
        T[:3, 1] -= np.dot(T[:3, 2], T[:3, 1]) * T[:3, 2]
        T[:3, 1] /= np.linalg.norm(T[:3, 1])
        T[:3, 0] = np.cross(T[:3, 1], T[:3, 2])
        arrow.transform(T)

        # Add the arrow to the geometry
        geoms.append(arrow)

    o3d.visualization.draw_geometries(geoms)

if __name__ == '__main__':

    base_input_folder = "/home/rvsa/gary318/build_kinematic/input_rgbd"
    # datasets = [os.path.join(base_input_folder, "work", "laptop_gray_01100%d") % i for i in [1, 2, 3, 5]]
    # datasets += [os.path.join(base_input_folder, "work", "laptop_gold_01120%d") % i for i in range(1, 5+1)]

    # datasets = [os.path.join(base_input_folder, "cabinet_01100%d") % i for i in [1, 3, 4, 5]]
    # datasets += [os.path.join(base_input_folder, name) for name in ["cabinet_011202"]]
    # datasets += [os.path.join(base_input_folder, "fridge_01140%d") % i for i in range(1, 5+1)]

    # datasets += [os.path.join(base_input_folder, "fridge_01100%d") % i for i in range(1, 6+1)]
    # datasets = [os.path.join(base_input_folder, "work", "lamp_thin_01130%d") % i for i in [1, 2, 3, 4, 6]]
    # datasets += [os.path.join(base_input_folder, "work", "lamp_round_01110%d") % i for i in [2, 3, 5, 6, 7]]

    # datasets = [os.path.join(base_input_folder, "drawer_compact_01140%d") % i for i in [5, 6]]
    # datasets += [os.path.join(base_input_folder, "drawer_simple_01140%d") % i for i in [3, 4, 5]]

    # Fail cases
    datasets = [os.path.join(base_input_folder, "work", "laptop_gray_011004")]
    # datasets = [os.path.join(base_input_folder, "drawer_compact_011403")]

    # datasets = [os.path.join(base_input_folder, "drawer_compact_011402")]

    rotation_err = []
    translation_err = []
    for dataset_folder in datasets:
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
        # visualize_pose(dataset_folder)

        # # SAM3D
        # run_SAM3D_render(mesh_folder, render_folder)
        # run_SAM3D_inference(render_folder, SAM3D_folder)

        # # Movable Part Segmentation
        # extract_SAM3D_results(SAM3D_folder)
        # segment_mesh(dataset_folder, show_difference=True, show_final_pcd=True)

        # # REART
        # run_REART(dataset_folder, ablation=False)
        draw_pcd_and_axis(dataset_folder, ablation=False)
        # all_revolute_joints, all_connect_to_root, ret_dict = extract_urdf(dataset_folder, ablation=True)
        # print(f"Result of {dataset_folder}: {ret_dict}")
        # rotation_err.append(ret_dict["axis_rotation_error"])
        # translation_err.append(ret_dict["axis_translation_error"])

    # print(f"Rotation error: mean {np.mean(rotation_err)}, min {np.min(rotation_err)}, max {np.max(rotation_err)}")
    # print(f"Translation error: mean {np.mean(translation_err)}, min {np.min(translation_err)}, max {np.max(translation_err)}")