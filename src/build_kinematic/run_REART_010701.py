import os
import shutil
import trimesh
import numpy as np
import open3d as o3d

import constants

from typing import List

def load_and_transform_mesh(model_path: str, transform_path: str) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(model_path)
    data = np.load(transform_path)
    transform, scale = data["pose"], data["scale"]

    mesh = mesh.scale(constants.DEFAULT_SCALE * scale, np.array([0, 0, 0]))  # scale must equal to that in my_run.py
    mesh.transform(constants.X_ROTATE_180)
    mesh.transform(transform)
    return mesh

def load_SAM3D_mesh(model_path: str, transform_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(model_path)
    data = np.load(transform_path)
    transform, scale = data["pose"], data["scale"]

    mesh.apply_scale(constants.DEFAULT_SCALE * scale)
    mesh.apply_transform(constants.X_ROTATE_180)
    mesh.apply_transform(transform)
    return mesh

def segment_mesh(dataset_path: str) -> None:
    oops_path = os.path.join(dataset_path, "oops")
    mesh_path = os.path.join(dataset_path, "mesh")
    SAM3D_path = os.path.join(dataset_path, "SAM3D")
    assert os.path.exists(oops_path), f"Oops path {oops_path} does not exist"
    assert os.path.exists(mesh_path), f"Mesh path {mesh_path} does not exist"

    state_nums: List[int] = []
    for filename in os.listdir(mesh_path):
        suffix = os.path.splitext(filename)[1]
        basename = os.path.splitext(filename)[0]
        if suffix == ".glb":
            state_nums.append(int(basename[-1]))
    state_nums.sort()

    # Sample point clouds
    pcds = []
    SAM3D_labels = []
    for state_num in state_nums:
        mesh = load_SAM3D_mesh(os.path.join(SAM3D_path, f"state_{state_num}", "mesh.ply"), os.path.join(oops_path, f"state_{state_num}_pose_scaled.npz"))
        pts, face_idx = trimesh.sample.sample_surface_even(mesh, 20000)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color(constants.COLOR_MAP[0])
        pcds.append(pcd)

        label = np.load(os.path.join(SAM3D_path, f"state_{state_num}", "labels.npy"))
        SAM3D_labels.append(label[face_idx])

    # Tag differences between adjacent point clouds
    for i in range(len(pcds) - 1):
        tag = i + 1

        dist_12 = np.asarray(pcds[i].compute_point_cloud_distance(pcds[i+1]))
        color_12 = np.array(pcds[i].colors)
        printed = False
        mean_dist_12 = []
        for label in range(0, np.max(SAM3D_labels[i])):
            mean_dist = np.mean(dist_12[SAM3D_labels[i] == label])
            mean_dist_12.append(mean_dist)
            std_dist = np.std(dist_12[SAM3D_labels[i] == label])
            if mean_dist > 0.03:
                print(f"Distance between state {i} and {i+1} for label {label}: mean {mean_dist*100:.2f}, std {std_dist*100:.2f}")
            
            if mean_dist > constants.PCD_OVERLAP_THRESHOLD:
                printed = True
                color_12[SAM3D_labels[i] == label] = np.array(constants.COLOR_MAP[tag])
        if not printed:
            farthest_part_index = np.argmax(mean_dist_12)
            farthest_part_pcd = o3d.geometry.PointCloud()
            farthest_part_pcd.points = o3d.utility.Vector3dVector(np.array(pcds[i].points)[SAM3D_labels[i] == farthest_part_index])
            print(f"Printing farthest part {farthest_part_index} of state {i}")

            print_part = [j for j in range(0, np.max(SAM3D_labels[i])) if
                            np.array(
                                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(pcds[i].points)[SAM3D_labels[i] == j])) \
                                .compute_point_cloud_distance(farthest_part_pcd)
                            ).mean() < constants.PCD_ALLCLOSE_THRESHOLD
                         ]
            for label in print_part:
                color_12[SAM3D_labels[i] == label] = np.array(constants.COLOR_MAP[tag])
        pcds[i].colors = o3d.utility.Vector3dVector(color_12)

        dist_21 = np.asarray(pcds[i+1].compute_point_cloud_distance(pcds[i]))
        color_21 = np.array(pcds[i+1].colors)
        printed = False
        mean_dist_21 = []
        for label in range(0, np.max(SAM3D_labels[i+1])):
            mean_dist = np.mean(dist_21[SAM3D_labels[i+1] == label])
            mean_dist_21.append(mean_dist)
            std_dist = np.std(dist_21[SAM3D_labels[i+1] == label])
            if mean_dist > 0.03:
                print(f"Distance between state {i+1} and {i} for label {label}: mean {mean_dist*100:.2f}, std {std_dist*100:.2f}")

            if mean_dist > constants.PCD_OVERLAP_THRESHOLD:
                printed = True
                color_21[SAM3D_labels[i+1] == label] = np.array(constants.COLOR_MAP[tag])
        if not printed:
            farthest_part_index = np.argmax(mean_dist_21)
            farthest_part_pcd = o3d.geometry.PointCloud()
            farthest_part_pcd.points = o3d.utility.Vector3dVector(np.array(pcds[i+1].points)[SAM3D_labels[i+1] == farthest_part_index])
            print(f"Printing farthest part {farthest_part_index} of state {i+1}")

            print_part = [j for j in range(0, np.max(SAM3D_labels[i+1])) if
                            np.array(
                                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(pcds[i+1].points)[SAM3D_labels[i+1] == j])) \
                                .compute_point_cloud_distance(farthest_part_pcd)
                            ).mean() < constants.PCD_ALLCLOSE_THRESHOLD
                         ]
            for label in print_part:
                color_21[SAM3D_labels[i+1] == label] = np.array(constants.COLOR_MAP[tag])
        pcds[i+1].colors = o3d.utility.Vector3dVector(color_21)

        o3d.visualization.draw_geometries([pcds[i], pcds[i+1]])

    # Project the difference to last point cloud
    raise
    assert len(state_nums) <= 3
    pcd_last = pcds[-1]
    kd_tree = o3d.geometry.KDTreeFlann(pcd_last)
    for i in range(1, len(pcds) - 1):
        # Extract part i from color map
        color = np.array(pcds[i].colors)
        points_to_cast = np.array(pcds[i].points)[np.all(color == np.array(constants.COLOR_MAP[i]), axis=1)]

        # Refine points_to_cast
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(points_to_cast)
        dist_to_last = np.asarray(temp_pcd.compute_point_cloud_distance(pcd_last))
        points_to_cast = points_to_cast[dist_to_last < constants.PCD_CASTING_THRESHOLD]

        # Find closest point in last point cloud
        clostest_idx = np.array([kd_tree.search_knn_vector_3d(x, 1)[1][0] for x in points_to_cast])
        pcd_last_color = np.array(pcd_last.colors)
        pcd_last_color[clostest_idx] = np.array(constants.COLOR_MAP[i])
        pcd_last.colors = o3d.utility.Vector3dVector(pcd_last_color)

    o3d.visualization.draw_geometries([pcd_last])

    # Separate the last point cloud into objects and conduct statistical outlier removal
    final_points = []
    part_labels = []
    for part_id in range(len(state_nums)):
        part_points = np.array(pcd_last.points)[np.all(np.array(pcd_last.colors) == np.array(constants.COLOR_MAP[part_id]), axis=1)]
        part_pcd = o3d.geometry.PointCloud()
        part_pcd.points = o3d.utility.Vector3dVector(part_points)
        part_pcd, _ = part_pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.0)
        part_pcd.paint_uniform_color(constants.COLOR_MAP[part_id])

        final_points.append(np.array(part_pcd.points))
        part_labels.append(np.repeat(part_id, final_points[-1].shape[0]))

    # Visualize
    # o3d.visualization.draw_geometries(pcds)
    o3d.visualization.draw_geometries([pcd_last])

    # Save the result
    np.savez(os.path.join(oops_path, "result.npz"),
             pcd=np.concatenate(final_points),
             part_id=np.concatenate(part_labels))

if __name__ == "__main__":
    dataset_path = "/home/rvsa/gary318/build_kinematic/input_rgbd/010602_fridge"
    segment_mesh(dataset_path)
