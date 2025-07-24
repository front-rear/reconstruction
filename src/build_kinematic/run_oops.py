import os
import json
import shutil

import cv2
import numpy as np
import open3d as o3d

from typing import List, Tuple

import constants


def load_and_transform_mesh(file_path: str, *, scale: float = 1.0, transform: np.ndarray = np.eye(4)) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh = mesh.scale(constants.DEFAULT_SCALE * scale, np.array([0, 0, 0]))  # scale must equal to that in my_run.py
    mesh.transform(constants.X_ROTATE_180)
    mesh.transform(transform)
    return mesh


def run_oops(dataset_path: str) -> None:
    dataset_path = os.path.abspath(dataset_path)
    assert os.path.exists(dataset_path), f"Dataset path {dataset_path} does not exist"

    SCRIPT_PATH = "/home/rvsa/gary318/FoundationPose/runs/my_run.py"
    os.chdir(os.path.dirname(SCRIPT_PATH))

    for filename in os.listdir(os.path.join(dataset_path, "mesh")):
        suffix = os.path.splitext(filename)[1]
        if suffix != ".glb":
            continue

        # Run the script
        basename = os.path.splitext(filename)[0]
        retv = os.system(f"python {SCRIPT_PATH} " +
                         "--inference --special_name test_mid_final_zerodepth_maskrgb " +
                         "--first_selection_num 25 --diff_render_iteration 300 " +
                         "--indix 1 --matching --debug 3 --zero_depth --mask_rgb --first_selection --mask_crop " +
                         "--dataset_dir %s --state_name %s" % (dataset_path, basename))
        if retv != 0:
            raise RuntimeError("Oops failed")

        # Copy the output
        output_path = os.path.dirname(SCRIPT_PATH)
        dataset_save_path = os.path.join(dataset_path, "oops")
        if not os.path.exists(dataset_save_path):
            os.makedirs(dataset_save_path)

        shutil.copy(os.path.join(output_path, "depth_best_u16.png"), os.path.join(dataset_save_path, f"{basename}_render_depth.png"))
        shutil.copy(os.path.join(output_path, "pose_1_default.npy"), os.path.join(dataset_save_path, f"{basename}_pose_raw.npy"))
        shutil.copy(os.path.join(output_path, "vis_score.png"), os.path.join(dataset_save_path, f"{basename}_visualize.png"))
        shutil.copy(os.path.join(output_path, "vis_score_before_selection.png"), os.path.join(dataset_save_path, f"{basename}_selection.png"))
        shutil.copy(os.path.join(output_path, "edge_best_with_input.png"), os.path.join(dataset_save_path, f"{basename}_edge.png"))

def adjust_scale(dataset_path: str, display_hist: bool = False) -> None:
    oops_path = os.path.join(dataset_path, "oops")
    assert os.path.exists(oops_path), f"Oops path {oops_path} does not exist"

    for filename in os.listdir(os.path.join(dataset_path, "mesh")):
        suffix = os.path.splitext(filename)[1]
        basename = os.path.splitext(filename)[0]
        if suffix != ".glb":
            continue

        raw_pose_path = os.path.join(oops_path, f"{basename}_pose_raw.npy")
        raw_pose: np.ndarray = np.load(raw_pose_path)  # (4, 4)

        rendered_depth = np.asarray(o3d.io.read_image(os.path.join(oops_path, f"{basename}_render_depth.png")))
        real_depth = np.asarray(o3d.io.read_image(os.path.join(dataset_path, "raw_depth", f"{basename}.png")))
        mask = cv2.imread(os.path.join(dataset_path, "mask", f"{basename}.png"), cv2.IMREAD_GRAYSCALE)

        # Adjust the translation component
        mask &= ((real_depth > 0) & (real_depth < constants.OOPS_MAX_VALID_GT_DEPTH))
        mask &= ((rendered_depth > 0) & (rendered_depth < 2000))
        mask = mask.astype(bool)
        depth_scale = real_depth[mask] / rendered_depth[mask]
        scale_mean, scale_std = np.mean(depth_scale), np.std(depth_scale)

        depth_scale = depth_scale[np.abs(depth_scale - scale_mean) < 3 * scale_std]
        scale_mean, scale_std = np.mean(depth_scale), np.std(depth_scale)

        # Debug: visualize the distribution of depth scale
        if display_hist:
            import matplotlib.pyplot as plt
            from matplotlib import cm

            used_depth = real_depth.copy()
            used_depth[~mask] = 0
            plt.figure(figsize=(19.2, 10.8))
            plt.imshow(used_depth, cmap=cm.jet)
            plt.axis('off')
            plt.show(block=True)

            used_depth = rendered_depth.copy()
            used_depth[~mask] = 0
            plt.figure(figsize=(19.2, 10.8))
            plt.imshow(used_depth, cmap=cm.jet)
            plt.axis('off')
            plt.show(block=True)

            plt.hist(depth_scale, bins=100)
            plt.title(f"Depth scale distribution of {basename}")
            plt.show(block=True)

        depth_scale = np.mean(depth_scale)
        raw_pose[:3, 3] *= depth_scale
        print(f"Adjusted translation component of {basename} by {depth_scale:.2f}, std={scale_std:.2f}")

        # Save the adjusted pose
        np.savez(os.path.join(oops_path, f"{basename}_pose_scaled.npz"), pose=raw_pose, scale=depth_scale)

def refine_pose(dataset_path: str, display_icp: bool = False, canonical_idx: int = -1) -> None:
    mesh_path = os.path.join(dataset_path, "mesh")
    oops_path = os.path.join(dataset_path, "oops")
    assert os.path.exists(oops_path), f"Oops path {oops_path} does not exist"
    assert os.path.exists(mesh_path), f"Mesh path {mesh_path} does not exist"

    state_nums: List[int] = []
    for filename in os.listdir(mesh_path):
        suffix = os.path.splitext(filename)[1]
        basename = os.path.splitext(filename)[0]
        if suffix == ".glb":
            state_nums.append(int(basename[-1]))
    state_nums.sort()

    # Load the reference mesh
    canonical_idx = canonical_idx if canonical_idx >= 0 else len(state_nums) + canonical_idx
    reference_state = state_nums[canonical_idx]
    ref_data = np.load(os.path.join(oops_path, f"state_{reference_state}_pose_scaled.npz"))
    ref_transform, ref_scale = constants.CAM_UNDER_BASE @ ref_data["pose"], float(ref_data["scale"])  # Transform: obj under base
    ref_mesh = load_and_transform_mesh(os.path.join(mesh_path, "state_%d.glb" % reference_state), scale=ref_scale, transform=ref_transform)  # Why glb?
    ref_pcd = ref_mesh.sample_points_uniformly(10000)

    for state_num in state_nums:
        save_path = os.path.join(oops_path, f"state_{state_num}_pose_adjusted.npz")
        if state_num == reference_state:
            np.savez(save_path, pose=ref_transform, scale=ref_scale)
            continue

        data = np.load(os.path.join(oops_path, f"state_{state_num}_pose_scaled.npz"))
        transform, scale = constants.CAM_UNDER_BASE @ data["pose"], float(data["scale"])
        mesh = load_and_transform_mesh(os.path.join(mesh_path, "state_%d.glb" % state_num), scale=scale, transform=transform)
        part_pcd = mesh.sample_points_uniformly(10000)

        # Conduct ICP
        reg_p2p = o3d.pipelines.registration.registration_icp(
            part_pcd, ref_pcd, constants.ICP_THRESHOLD, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        # Visualize the result
        if display_icp:
            print("ICP result:", reg_p2p.transformation)
            print("Before ICP:")
            o3d.visualization.draw_geometries([part_pcd, ref_pcd])
            print("After ICP:")
            part_pcd.transform(reg_p2p.transformation)
            o3d.visualization.draw_geometries([part_pcd, ref_pcd])

        # Save the adjusted pose
        np.savez(save_path, pose=reg_p2p.transformation @ transform, scale=scale)

def visualize_pose(dataset_path: str) -> None:
    mesh_path = os.path.join(dataset_path, "mesh")
    oops_path = os.path.join(dataset_path, "oops")
    assert os.path.exists(oops_path), f"Oops path {oops_path} does not exist"
    assert os.path.exists(mesh_path), f"Mesh path {mesh_path} does not exist"

    COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]

    meshes = []
    for filename in os.listdir(mesh_path):
        suffix = os.path.splitext(filename)[1]
        if suffix != ".glb":
            continue

        basename = os.path.splitext(filename)[0]
        state_number = int(basename[-1])
        mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_path, filename))
        data = np.load(os.path.join(oops_path, f"{basename}_pose_adjusted.npz"))
        transform, scale = data["pose"], float(data["scale"]) # Transform: obj under base

        print("Transform:", transform)

        mesh = load_and_transform_mesh(os.path.join(mesh_path, filename), scale=scale, transform=transform)
        mesh.paint_uniform_color(COLORS[state_number])
        meshes.append(mesh)

        # Draw a coordinate frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1 + 0.05 * state_number, origin=[0, 0, 0])
        frame.transform(transform)
        meshes.append(frame)


    # Add world frame
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    meshes.append(world_frame)

    o3d.visualization.draw_geometries(meshes)

def calculate_mask_error(dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
    gt_depth_path = os.path.join(dataset_path, "depth_filtered")
    oops_path = os.path.join(dataset_path, "oops")
    assert os.path.exists(oops_path), f"Oops path {oops_path} does not exist"
    assert os.path.exists(gt_depth_path), f"GT depth path {gt_depth_path} does not exist"

    ious = []
    depth_errors = []
    for filename in os.listdir(gt_depth_path):
        name_part = os.path.splitext(filename)[0]

        scale: float = np.load(os.path.join(oops_path, f"{name_part}_pose_adjusted.npz"))["scale"]

        gt_depth = np.asarray(o3d.io.read_image(os.path.join(gt_depth_path, filename)))
        rendered_depth = np.asarray(o3d.io.read_image(os.path.join(oops_path, f"{name_part}_render_depth.png"))) * scale
        gt_mask = gt_depth > 0
        rendered_mask = rendered_depth > 0

        iou: float = np.count_nonzero(gt_mask & rendered_mask) / np.count_nonzero(gt_mask | rendered_mask)
        ious.append(iou)

        # Debug: visualize the mask
        # cv2.imshow("gt_mask", gt_mask.astype(np.uint8) * 255)
        # cv2.imshow("rendered_mask", rendered_mask.astype(np.uint8) * 255)
        # cv2.waitKey(0)

        # Filter out invalid depths before calculating the error
        gt_mask &= (gt_depth < constants.OOPS_MAX_VALID_GT_DEPTH)

        avg_gt_depth = np.mean(gt_depth[gt_mask])
        depth_error = np.mean(np.abs(rendered_depth[gt_mask & rendered_mask] - gt_depth[gt_mask & rendered_mask]))
        depth_error_rel = depth_error / avg_gt_depth
        depth_errors.append(depth_error_rel)

        print(f"{name_part}: iou={iou:.4f}, depth_error={depth_error:.4f}, depth_error_rel={depth_error_rel:.4f}")
        json.dump(
            {"iou": iou, "depth_error": depth_error, "depth_error_rel": depth_error_rel},
            open(os.path.join(oops_path, f"{name_part}_mask_error.json"), "w")
        )

    return np.array(ious), np.array(depth_errors)

if __name__ == "__main__":
    dataset_path = "/home/rvsa/gary318/build_kinematic/input_rgbd/123101_drawer"
    run_oops(dataset_path)

    adjust_scale(dataset_path)
    refine_pose(dataset_path)

    visualize_pose(dataset_path)

    calculate_mask_error(dataset_path)
