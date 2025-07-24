import os
import json
import shutil
import cv2
import numpy as np
import open3d as o3d
from typing import List, Tuple

import constants


def load_and_transform_mesh(file_path: str, scale: float = 1.0, transform: np.ndarray = np.eye(4)) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh = mesh.scale(constants.DEFAULT_SCALE * scale, np.array([0, 0, 0]))
    mesh.transform(constants.X_ROTATE_180)
    mesh.transform(transform)
    return mesh


def run_oops(dataset_path: str) -> None:
    dataset_path = os.path.abspath(dataset_path)
    os.makedirs(os.path.join(dataset_path, "oops"), exist_ok=True)

    SCRIPT_PATH = "/FoundationPose/runs/my_run.py"
    os.chdir(os.path.dirname(SCRIPT_PATH))

    for filename in os.listdir(os.path.join(dataset_path, "mesh")):
        if not filename.endswith(".glb"):
            continue

        basename = os.path.splitext(filename)[0]
        command = (
            f"python {SCRIPT_PATH} --inference "
            "--special_name test_mid_final_zerodepth_maskrgb "
            "--first_selection_num 25 --diff_render_iteration 300 "
            "--indix 1 --matching --debug 3 --zero_depth --mask_rgb "
            "--first_selection --mask_crop --dataset_dir {0} --state_name {1}".format(dataset_path, basename)
        )
        os.system(command)

        output_path = os.path.dirname(SCRIPT_PATH)
        oops_path = os.path.join(dataset_path, "oops")
        files_to_copy = [
            ("depth_best_u16.png", f"{basename}_render_depth.png"),
            ("pose_1_default.npy", f"{basename}_pose_raw.npy"),
            ("vis_score.png", f"{basename}_visualize.png"),
            ("vis_score_before_selection.png", f"{basename}_selection.png"),
            ("edge_best_with_input.png", f"{basename}_edge.png"),
        ]
        for src, dst in files_to_copy:
            shutil.copy(os.path.join(output_path, src), os.path.join(oops_path, dst))


def adjust_scale(dataset_path: str, display_hist: bool = False) -> None:
    oops_path = os.path.join(dataset_path, "oops")
    os.makedirs(oops_path, exist_ok=True)

    for filename in os.listdir(os.path.join(dataset_path, "mesh")):
        if not filename.endswith(".glb"):
            continue

        basename = os.path.splitext(filename)[0]
        raw_pose = np.load(os.path.join(oops_path, f"{basename}_pose_raw.npy"))

        rendered_depth = np.asarray(o3d.io.read_image(os.path.join(oops_path, f"{basename}_render_depth.png")))
        real_depth = np.asarray(o3d.io.read_image(os.path.join(dataset_path, "raw_depth", f"{basename}.png")))
        mask = cv2.imread(os.path.join(dataset_path, "mask", f"{basename}.png"), cv2.IMREAD_GRAYSCALE)

        mask = (mask & (real_depth > 0) & (real_depth < constants.OOPS_MAX_VALID_GT_DEPTH) &
                (rendered_depth > 0) & (rendered_depth < 2000)).astype(bool)
        depth_scale = real_depth[mask] / rendered_depth[mask]
        scale_mean, scale_std = np.mean(depth_scale), np.std(depth_scale)

        depth_scale = depth_scale[np.abs(depth_scale - scale_mean) < 3 * scale_std]
        scale_mean, scale_std = np.mean(depth_scale), np.std(depth_scale)

        if display_hist:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(19.2, 10.8))
            plt.imshow(real_depth, cmap='jet')
            plt.axis('off')
            plt.show()

            plt.figure(figsize=(19.2, 10.8))
            plt.imshow(rendered_depth, cmap='jet')
            plt.axis('off')
            plt.show()

            plt.hist(depth_scale, bins=100)
            plt.title(f"Depth scale distribution of {basename}")
            plt.show()

        depth_scale = np.mean(depth_scale)
        raw_pose[:3, 3] *= depth_scale
        print(f"Adjusted translation component of {basename} by {depth_scale:.2f}, std={scale_std:.2f}")

        np.savez(os.path.join(oops_path, f"{basename}_pose_scaled.npz"), pose=raw_pose, scale=depth_scale)


def refine_pose(dataset_path: str, display_icp: bool = False, canonical_idx: int = -1) -> None:
    mesh_path = os.path.join(dataset_path, "mesh")
    oops_path = os.path.join(dataset_path, "oops")
    state_nums = sorted([int(f.split('_')[-1].split('.')[0]) for f in os.listdir(mesh_path) if f.endswith(".glb")])

    canonical_idx = canonical_idx if canonical_idx >= 0 else len(state_nums) + canonical_idx
    reference_state = state_nums[canonical_idx]
    ref_data = np.load(os.path.join(oops_path, f"state_{reference_state}_pose_scaled.npz"))
    ref_transform, ref_scale = constants.CAM_UNDER_BASE @ ref_data["pose"], float(ref_data["scale"])
    ref_mesh = load_and_transform_mesh(os.path.join(mesh_path, f"state_{reference_state}.glb"), scale=ref_scale, transform=ref_transform)
    ref_pcd = ref_mesh.sample_points_uniformly(10000)

    for state_num in state_nums:
        save_path = os.path.join(oops_path, f"state_{state_num}_pose_adjusted.npz")
        if state_num == reference_state:
            np.savez(save_path, pose=ref_transform, scale=ref_scale)
            continue

        data = np.load(os.path.join(oops_path, f"state_{state_num}_pose_scaled.npz"))
        transform, scale = constants.CAM_UNDER_BASE @ data["pose"], float(data["scale"])
        mesh = load_and_transform_mesh(os.path.join(mesh_path, f"state_{state_num}.glb"), scale=scale, transform=transform)
        part_pcd = mesh.sample_points_uniformly(10000)

        reg_p2p = o3d.pipelines.registration.registration_icp(
            part_pcd, ref_pcd, constants.ICP_THRESHOLD, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        if display_icp:
            print("ICP result:", reg_p2p.transformation)
            print("Before ICP:")
            o3d.visualization.draw_geometries([part_pcd, ref_pcd])
            print("After ICP:")
            part_pcd.transform(reg_p2p.transformation)
            o3d.visualization.draw_geometries([part_pcd, ref_pcd])

        np.savez(save_path, pose=reg_p2p.transformation @ transform, scale=scale)


def visualize_pose(dataset_path: str) -> None:
    mesh_path = os.path.join(dataset_path, "mesh")
    oops_path = os.path.join(dataset_path, "oops")
    COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]

    meshes = []
    for filename in os.listdir(mesh_path):
        if not filename.endswith(".glb"):
            continue

        basename = os.path.splitext(filename)[0]
        state_number = int(basename.split('_')[-1])
        data = np.load(os.path.join(oops_path, f"{basename}_pose_adjusted.npz"))
        transform, scale = data["pose"], float(data["scale"])

        mesh = load_and_transform_mesh(os.path.join(mesh_path, filename), scale=scale, transform=transform)
        mesh.paint_uniform_color(COLORS[state_number % len(COLORS)])
        meshes.append(mesh)

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1 + 0.05 * state_number, origin=[0, 0, 0])
        frame.transform(transform)
        meshes.append(frame)

    meshes.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0]))
    o3d.visualization.draw_geometries(meshes)


def calculate_mask_error(dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
    gt_depth_path = os.path.join(dataset_path, "depth_filtered")
    oops_path = os.path.join(dataset_path, "oops")

    ious, depth_errors = [], []
    for filename in os.listdir(gt_depth_path):
        name_part = os.path.splitext(filename)[0]
        scale = np.load(os.path.join(oops_path, f"{name_part}_pose_adjusted.npz"))["scale"]

        gt_depth = np.asarray(o3d.io.read_image(os.path.join(gt_depth_path, filename)))
        rendered_depth = np.asarray(o3d.io.read_image(os.path.join(oops_path, f"{name_part}_render_depth.png"))) * scale
        gt_mask = gt_depth > 0
        rendered_mask = rendered_depth > 0

        iou = np.count_nonzero(gt_mask & rendered_mask) / np.count_nonzero(gt_mask | rendered_mask)
        ious.append(iou)

        gt_mask = gt_mask & (gt_depth < constants.OOPS_MAX_VALID_GT_DEPTH)

        avg_gt_depth = np.mean(gt_depth[gt_mask])
        depth_error = np.mean(np.abs(rendered_depth[gt_mask & rendered_mask] - gt_depth[gt_mask & rendered_mask]))
        depth_errors.append(depth_error / avg_gt_depth)

        error_data = {"iou": iou, "depth_error": depth_error, "depth_error_rel": depth_error / avg_gt_depth}
        json.dump(error_data, open(os.path.join(oops_path, f"{name_part}_mask_error.json"), "w"))

    return np.array(ious), np.array(depth_errors)


if __name__ == "__main__":
    dataset_path = "/build_kinematic/123101_drawer"
    run_oops(dataset_path)
    adjust_scale(dataset_path)
    refine_pose(dataset_path)
    visualize_pose(dataset_path)
    calculate_mask_error(dataset_path)
