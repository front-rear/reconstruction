import os
import shutil
from copy import deepcopy

import sapien.core as sapien
from sapien.utils import Viewer

import math
import trimesh
import numpy as np
import pandas as pd
import open3d as o3d
import xml.etree.ElementTree as ET

import constants

from typing import List, Tuple, Dict, Any

def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("Picked points:", vis.get_picked_points())
    return vis.get_picked_points()
def batch_closest_points(queries: np.ndarray, kd_tree: o3d.geometry.KDTreeFlann) -> Tuple[np.ndarray, np.ndarray]:
    results = [kd_tree.search_knn_vector_3d(x, 1) for x in queries]
    indices = np.array([x[1][0] for x in results])
    distances = np.array([x[2][0] for x in results])
    return indices, distances
def tag_difference(pcd_to_tag: o3d.geometry.PointCloud, labels: np.ndarray, pcd_ref: o3d.geometry.PointCloud, tag_id: int) -> None:
    points1 = np.array(pcd_to_tag.points)
    points2 = np.array(pcd_ref.points)
    kd_tree1 = o3d.geometry.KDTreeFlann(pcd_to_tag)
    kd_tree2 = o3d.geometry.KDTreeFlann(pcd_ref)

    # obj_size: float = np.linalg.norm(np.max(points1, axis=0) - np.min(points1, axis=0))

    invalid_labels = []
    for label in range(0, np.max(labels) + 1):
        if np.count_nonzero(labels == label) == 0:
            continue

        index_forward, distance_forward = batch_closest_points(points1[labels == label], kd_tree2)
        index_backward, distance_backward = batch_closest_points(points2[index_forward], kd_tree1)

        mean_dist_forward = np.median(distance_forward) * 100
        mean_dist_backward = np.median(distance_backward) * 100
        ratio = mean_dist_backward / mean_dist_forward

        # Debug
        # temp_pcd = deepcopy(pcd_to_tag)
        # temp_color = np.array(temp_pcd.colors)
        # temp_color[labels == label] = constants.COLOR_MAP[tag_id]
        # temp_pcd.colors = o3d.utility.Vector3dVector(temp_color)
        # pick_points(temp_pcd)

        if (ratio >= constants.PCD_DIST_RATIO_THRESHOLD and mean_dist_forward < constants.PCD_DIST_MEDIAN_THRESHOLD) and \
           (mean_dist_forward <= 0.7):
            invalid_labels.append(label)
            print(f"\tDistance for label {label}: median {mean_dist_forward} → {mean_dist_backward} ← ratio {mean_dist_backward/mean_dist_forward:.2f} [NOT MOVING]")
        else:
            print(f"\tDistance for label {label}: median {mean_dist_forward} → {mean_dist_backward} ← ratio {mean_dist_backward/mean_dist_forward:.2f}")
        


    valid_index = np.in1d(labels, invalid_labels, invert=True)
    colors = np.array(pcd_to_tag.colors)
    colors[valid_index] = constants.COLOR_MAP[tag_id]
    pcd_to_tag.colors = o3d.utility.Vector3dVector(colors)
def load_mesh(model_path: str, transform_path: str, extra_transform: np.ndarray = np.eye(4)) -> trimesh.Trimesh:
    suffix = os.path.splitext(model_path)[1]

    mesh = trimesh.load(model_path)
    if suffix == ".glb":
        mesh: trimesh.Trimesh = mesh.geometry["model"]
    data = np.load(transform_path)
    transform, scale = data["pose"], data["scale"]

    mesh.apply_scale(constants.DEFAULT_SCALE * scale)
    mesh.apply_transform(extra_transform)
    mesh.apply_transform(constants.X_ROTATE_180)
    mesh.apply_transform(transform)
    return mesh
def add_default_inerital(link: ET.Element) -> None:
    inertial = ET.SubElement(link, "inertial")
    mass = ET.SubElement(inertial, "mass", value="10")
    inertia = ET.SubElement(inertial, "inertia",
                            ixx="1", ixy="0", ixz="0",
                            iyy="1", iyz="0", izz="1")
def add_geometry(visual_or_collision: ET.Element, obj_file: str) -> None:
    geometry = ET.SubElement(visual_or_collision, "geometry")
    mesh = ET.SubElement(geometry, "mesh",
                          filename=obj_file)


def segment_mesh(dataset_path: str, show_difference: bool = False, show_final_pcd: bool = False) -> None:
    oops_path = os.path.join(dataset_path, "oops")
    mesh_path = os.path.join(dataset_path, "mesh")
    SAM3D_path = os.path.join(dataset_path, "SAM3D")
    REART_path = os.path.join(dataset_path, "REART")
    assert os.path.exists(oops_path), f"Oops path {oops_path} does not exist"
    assert os.path.exists(mesh_path), f"Mesh path {mesh_path} does not exist"
    if not os.path.exists(REART_path):
        os.makedirs(REART_path)

    state_nums: List[int] = []
    for filename in os.listdir(mesh_path):
        suffix = os.path.splitext(filename)[1]
        basename = os.path.splitext(filename)[0]
        if suffix == ".glb":
            state_nums.append(int(basename[-1]))
    state_nums.sort()

    # Sample point clouds
    pcds: List[o3d.geometry.PointCloud] = []
    kd_trees: List[o3d.geometry.KDTreeFlann] = []
    SAM3D_labels: List[np.ndarray] = []
    for i, state_num in enumerate(state_nums):
        mesh = load_mesh(os.path.join(SAM3D_path, f"state_{state_num}", "mesh.ply"), os.path.join(oops_path, f"state_{state_num}_pose_adjusted.npz"))
        pts, face_idx = trimesh.sample.sample_surface_even(mesh, constants.SAMPLE_SIZE)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color(constants.COLOR_MAP[0])
        pcds.append(pcd)
        kd_trees.append(o3d.geometry.KDTreeFlann(pcd))

        label = np.load(os.path.join(SAM3D_path, f"state_{state_num}", "labels.npy"))
        SAM3D_labels.append(label[face_idx])

        if i != len(state_nums) - 1:
            o3d.io.write_point_cloud(os.path.join(REART_path, f"state_{state_num}.xyz"), pcd)

    # Merge small labels
    for i in range(len(SAM3D_labels)):
        print(f"Merging small labels in state {state_nums[i]}")
        while True:

            # Find smallest label
            label_counts = np.bincount(SAM3D_labels[i])
            label_counts[label_counts == 0] = 1000000
            smallest_label = np.argmin(label_counts)
            if label_counts[smallest_label] >= SAM3D_labels[i].shape[0] * constants.CLUSTER_MERGING_THRESHOLD:
                print(f"\tNo small labels found. Label counts: {label_counts}")
                break

            # Assign all points in smallest label to the nearest label
            label_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
                np.array(pcds[i].points)[SAM3D_labels[i] == smallest_label]
            ))
            nearest_label_index: int = 0
            nearest_label_distance: float = 1e9
            for target_label in range(0, np.max(SAM3D_labels[i]) + 1):
                if target_label == smallest_label or np.count_nonzero(SAM3D_labels[i] == target_label) == 0:
                    continue
                target_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
                    np.array(pcds[i].points)[SAM3D_labels[i] == target_label]
                ))
                distance = np.median(label_pcd.compute_point_cloud_distance(target_pcd))
                if distance < nearest_label_distance:
                    nearest_label_index = target_label
                    nearest_label_distance = distance
            SAM3D_labels[i][SAM3D_labels[i] == smallest_label] = nearest_label_index
            print(f"\tMerged label {smallest_label} (size {label_counts[smallest_label]}) with label {nearest_label_index} (size {label_counts[nearest_label_index]})")

    # Tag difference between adjacent point clouds
    for i in range(len(pcds) - 1):
        print(f"Tag difference between state {state_nums[i]} and state {state_nums[i+1]}")
        tag_difference(pcds[i], SAM3D_labels[i], pcds[i+1], i+1)
        tag_difference(pcds[i+1], SAM3D_labels[i+1], pcds[i], i+1)

        if show_difference:
            print(f"State {i} vs State {i+1}")
            o3d.visualization.draw_geometries([pcds[i], pcds[i+1]])

    # Project the difference to last point cloud
    assert len(state_nums) <= 3
    pcd_last = pcds[-1]
    pcd_last_color = np.array(pcd_last.colors)
    for i in range(1, len(pcds) - 1):
        # Extract part i from color map
        color = np.array(pcds[i].colors)
        points_to_cast = np.array(pcds[i].points)[np.all(color == np.array(constants.COLOR_MAP[i]), axis=1)]
        points_to_cast_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_to_cast))

        # Find parts that are close enough in last point cloud
        for label in range(0, np.max(SAM3D_labels[-1]) + 1):
            part = np.array(pcd_last.points)[SAM3D_labels[-1] == label]
            part_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(part))
            if np.mean(part_pcd.compute_point_cloud_distance(points_to_cast_pcd)) < constants.PCD_CASTING_THRESHOLD:
                pcd_last_color[SAM3D_labels[-1] == label] = constants.COLOR_MAP[i]
                print(f"Casting part {i} from state {state_nums[i]} to state {state_nums[-1]}'s part {label}")
    pcd_last.colors = o3d.utility.Vector3dVector(pcd_last_color)

    # Separate the last point cloud into objects and conduct statistical outlier removal
    final_points = []
    part_labels = []
    for part_id in range(len(state_nums)):
        part_points = np.array(pcd_last.points)[np.all(np.array(pcd_last.colors) == np.array(constants.COLOR_MAP[part_id]), axis=1)]
        part_pcd = o3d.geometry.PointCloud()
        part_pcd.points = o3d.utility.Vector3dVector(part_points)
        part_pcd, _ = part_pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.2)
        part_pcd.paint_uniform_color(constants.COLOR_MAP[part_id])
        # o3d.visualization.draw_geometries([part_pcd])

        final_points.append(np.array(part_pcd.points))
        part_labels.append(np.repeat(part_id, final_points[-1].shape[0]))

    # Save the result
    final_points = np.concatenate(final_points)
    part_labels = np.concatenate(part_labels)
    final_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(final_points))
    kd_tree = o3d.geometry.KDTreeFlann(final_pcd)
    np.save(os.path.join(REART_path, f"state_{state_nums[-1]}.npy"), final_points)
    np.save(os.path.join(REART_path, f"state_{state_nums[-1]}_labels.npy"), part_labels)

    # Visualize
    final_pcd.colors = o3d.utility.Vector3dVector(np.array(constants.COLOR_MAP)[part_labels])
    o3d.io.write_point_cloud(os.path.join(REART_path, f"visualized_input.ply"), pcd_last)

    if show_final_pcd:
        print("Final point cloud for %s" % dataset_path)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([final_pcd, coord_frame])


    export_path = os.path.join(REART_path, "urdf")
    if os.path.exists(export_path):
        shutil.rmtree(export_path)
    os.makedirs(export_path, exist_ok=True)
    os.makedirs(os.path.join(export_path, "visual"), exist_ok=True)
    os.makedirs(os.path.join(export_path, "collision"), exist_ok=True)

    # Tag parts for original mesh which has texture
    texture_mesh: trimesh.Trimesh = load_mesh(os.path.join(mesh_path, f"state_{state_nums[-1]}.glb"),
                                              os.path.join(oops_path, f"state_{state_nums[-1]}_pose_adjusted.npz"))
    avg_face_coord = np.mean(texture_mesh.vertices[texture_mesh.faces], axis=1)
    
    face_label = part_labels[batch_closest_points(avg_face_coord, kd_tree)[0]]

    # Extract each part of the mesh
    for part_id in range(len(state_nums)):
        part_path = os.path.join(export_path, "visual", f"part_{part_id}")
        os.makedirs(part_path, exist_ok=True)

        face_ids = np.where(face_label == part_id)
        if np.count_nonzero(face_ids):
            part_mesh = texture_mesh.submesh(np.where(face_label == part_id))[0]
            part_mesh.export(os.path.join(part_path, "mesh.obj"))

            # Export collision meshes
            part_mesh = trimesh.load_mesh(os.path.join(part_path, "mesh.obj"))
            convex_part_kwargs = trimesh.decomposition.convex_decomposition(part_mesh, maxConvexHulls=constants.CONVEX_DECOMP_RESULT_COUNT)
            meshes = [trimesh.Trimesh(**kwargs) for kwargs in convex_part_kwargs]
            for i, mesh in enumerate(meshes):
                mesh.export(os.path.join(export_path, "collision", f"part_{part_id}_{i}.obj"))
        else:
            print(f"[Warning] No faces found for part {part_id}")


    # last_mesh.export(os.path.join(REART_path, f"repainted.ply"))

def run_REART(dataset_path: str, farthest_point_sampling: bool = True) -> None:
    mesh_path = os.path.join(dataset_path, "mesh")
    REART_path = os.path.join(dataset_path, "REART")
    assert os.path.exists(REART_path), f"REART path {REART_path} does not exist"
    assert os.path.exists(mesh_path), f"Mesh path {mesh_path} does not exist"

    state_nums: List[int] = []
    for filename in os.listdir(mesh_path):
        suffix = os.path.splitext(filename)[1]
        basename = os.path.splitext(filename)[0]
        if suffix == ".glb":
            state_nums.append(int(basename[-1]))
    state_nums.sort()

    # Generate valid input for REART

    pcds = []
    # Add point clouds other than the last one
    for state_num in state_nums:
        if state_num == state_nums[-1]:
            break

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.loadtxt(os.path.join(REART_path, "state_%d.xyz" % state_num))))
        if farthest_point_sampling:
            pcd = np.array(pcd.farthest_point_down_sample(constants.REART_POINTCLOUD_SIZE).points)
        else:
            pcd = np.array(pcd.points)[np.random.choice(range(len(pcd.points)), size=constants.REART_POINTCLOUD_SIZE)]
        pcds.append(pcd)

    # Add the last point cloud
    oringin_pcd = np.load(os.path.join(REART_path, "state_%d.npy" % state_nums[-1]))
    labels = np.load(os.path.join(REART_path, "state_%d_labels.npy" % state_nums[-1])) + 1
    if farthest_point_sampling:
        pcd = np.array(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(oringin_pcd)).farthest_point_down_sample(constants.REART_POINTCLOUD_SIZE).points)
        choice = np.array([np.argmin(np.linalg.norm(pcd[i] - oringin_pcd, axis=1)) for i in range(pcd.shape[0])])
    else:
        choice = np.random.choice(range(oringin_pcd.shape[0]), size=constants.REART_POINTCLOUD_SIZE)
        pcd = oringin_pcd[choice]

    labels = labels[choice]
    pcds.append(pcd)

    pc = np.stack(pcds, axis=0)
    mean = np.mean(pc, axis=(0, 1), keepdims=True)
    pc = pc - mean

    scale = 0.1 / np.std(pc)
    pc *= scale

    # Visualize last point cloud and its labels
    # pcd_last = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc[-1]))
    # pcd_last.colors = o3d.utility.Vector3dVector(np.array(constants.COLOR_MAP)[labels])
    # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([pcd_last, coord_frame])

    # Save the input for REART
    raw_data = np.load(constants.REART_DATA_TEMPLATE, allow_pickle=True)
    trans = dict([(k, v[:pc.shape[0]]) for k, v in raw_data["trans"].item().items()])
    np.savez(constants.REART_DATA_PATH, segm=np.tile(labels, (pc.shape[0], 1)), pc=pc, trans=trans)
    np.savez(os.path.join(REART_path, "pcd_transform.npz"), mean=mean[0, 0, :], scale=scale)
    # print()

    # Run REART Pass 1
    cano_idx = len(state_nums) - 1
    os.chdir(constants.REART_PATH)
    os.system(f"python run_sapien.py --sapien_idx=212 --save_root=exp --n_iter=2000 --cano_idx={cano_idx} --use_nproc --use_assign_loss")

    # Run REART Pass 2
    os.chdir(constants.REART_PATH)
    os.system(f"python run_sapien.py --sapien_idx=212 --save_root=exp --n_iter=200 --cano_idx={cano_idx} " +
              "--model=kinematic --use_nproc --use_assign_loss --assign_iter=0 --assign_gap=1 --snapshot_gap=10 --base_result_path=exp/sapien_212/result.pkl")

    # Extract the final result
    os.chdir(constants.REART_PATH)
    os.system(f"python extract_results.py --sapien_idx=212 --save_root=exp --n_iter=200 --cano_idx={cano_idx} " +
              "--model=kinematic --use_nproc --use_assign_loss --assign_iter=0 --assign_gap=1 --snapshot_gap=10 --base_result_path=exp/sapien_212/result.pkl " +
              "--export_path=%s" % os.path.join(REART_path, "kinematic_result.npz"))

def extract_urdf(dataset_path: str) -> Tuple[bool, bool, Dict[str, Any]]:
    dataset_name = os.path.basename(dataset_path)
    REART_path = os.path.join(dataset_path, "REART")
    assert os.path.exists(REART_path), f"REART path {REART_path} does not exist"

    # Create folder
    urdf_folder = os.path.join(REART_path, "urdf")
    os.makedirs(urdf_folder, exist_ok=True)

    # Load ground truth
    has_gt = False
    df = pd.read_csv(constants.GT_FILE_PATH, index_col=0)
    if dataset_name in df.index:
        has_gt = True
        gt_data = df.loc[dataset_name].to_dict()
        gt_origin = constants.parse_grid_coord(gt_data["axis0_origin"])
        gt_axis = np.array([float(x) for x in gt_data["axis0_direction"].split(" ")])
        gt_axis /= np.linalg.norm(gt_axis)
        print(f"Ground truth found for {dataset_name}: axis {gt_axis}, origin {gt_origin}")

    # Load transform
    data = np.load(os.path.join(REART_path, "pcd_transform.npz"), allow_pickle=True)
    mean: np.ndarray = data["mean"]
    scale: float = data["scale"]
    data.close()

    # Load kinematic result from REART
    data = np.load(os.path.join(REART_path, "kinematic_result.npz"), allow_pickle=True)
    root_part: int = 0  # int(data["root_part"]), Hard-coded for now
    connection_dict: Dict[str, Dict[str, Any]] = data["connection_dict"].item()
    data.close()
    print("Root part: %d" % root_part)
    print("\t", connection_dict)

    ret_dict = {"has_gt": has_gt}
    all_revolute_joints = all([connection["type"] == "revolute" for connection in connection_dict.values()])
    all_connect_to_root = all([str(root_part) in key for key in connection_dict.keys()])

    root = ET.Element("robot", name="object")
    tree = ET.ElementTree(root)

    # base link
    base_link = ET.SubElement(root, "link", name="part_%d" % root_part)
    add_default_inerital(base_link)
    for i in range(constants.CONVEX_DECOMP_RESULT_COUNT):
        visual = ET.SubElement(base_link, "visual")
        add_geometry(visual, "visual/part_%d/mesh.obj" % root_part)

        collision = ET.SubElement(base_link, "collision")
        add_geometry(collision, "collision/part_%d_%d.obj" % (root_part, i))

    origin_mapping = {root_part: [0, 0, 0]}
    for connection in sorted(connection_dict.values(), key=lambda x: x["parent"]):
        print("Connection: %s" % str(connection))

        axis = connection["axis"] / np.linalg.norm(connection["axis"])
        assert np.allclose(np.linalg.norm(axis), 1)

        absolute_origin = (np.cross(axis, connection["moment"]) / scale) + mean
        if connection["parent"] in origin_mapping:
            origin = absolute_origin - origin_mapping[connection["parent"]]
        else:
            origin = absolute_origin
            print("\t [Warning] Parent origin not found, using absolute origin")
        origin_mapping[connection["child"]] = absolute_origin
        print("\tOrigin: %s" % str(origin))

        # Calculate error
        if has_gt:
            axis_rotation_error = math.degrees(math.acos(np.dot(axis, gt_axis)))
            if axis_rotation_error > 90: axis_rotation_error = 180 - axis_rotation_error
            ret_dict["axis_rotation_error"] = axis_rotation_error
            print("\tAxis rotation error: %.3f°" % axis_rotation_error)

            if connection["type"] == "revolute":
                common_normal = np.cross(axis, gt_axis)
                axis_translation_error = np.abs(np.dot(common_normal, (absolute_origin - gt_origin))) / np.linalg.norm(common_normal) * 100
                ret_dict["axis_translation_error"] = axis_translation_error
                print("\tAxis translation error: %.3f" % axis_translation_error)
            else:
                ret_dict["axis_translation_error"] = 0.0

        # Create link
        link = ET.SubElement(root, "link", name="part_%d" % connection["child"])
        add_default_inerital(link)

        # Add visual and collision
        for i in range(constants.CONVEX_DECOMP_RESULT_COUNT):
            visual = ET.SubElement(link, "visual")
            add_geometry(visual, "visual/part_%d/mesh.obj" % connection["child"])
            ET.SubElement(visual, "origin", xyz="%f %f %f" % tuple(-origin))

            collision = ET.SubElement(link, "collision")
            add_geometry(collision, "collision/part_%d_%d.obj" % (connection["child"], i))
            ET.SubElement(collision, "origin", xyz="%f %f %f" % tuple(-origin))

        # Create joint
        joint = ET.SubElement(root, "joint", name="part_%d_joint" % connection["child"], type=connection["type"])
        ET.SubElement(joint, "origin", xyz="%f %f %f" % tuple(origin))
        ET.SubElement(joint, "axis", xyz="%f %f %f" % tuple(axis))
        ET.SubElement(joint, "parent", link="part_%d" % connection["parent"])
        ET.SubElement(joint, "child", link="part_%d" % connection["child"])
        ET.SubElement(joint, "limit", effort="1000", lower="-3.14", upper="3.14", velocity="1000")
        ET.SubElement(joint, "dynamics", damping="0.1", friction="0.1")

    tree.write(os.path.join(urdf_folder, "object.urdf"))
    return all_revolute_joints, all_connect_to_root, ret_dict

if __name__ == "__main__":
    dataset_path = "/home/rvsa/gary318/build_kinematic/input_rgbd/010602_fridge"

    # segment_mesh(dataset_path)
    extract_urdf(dataset_path)

    # for i in range(100):
    #     print("Iteration %d" % i)
    #     run_REART(dataset_path)
    #     all_revolute_joints, all_connect_to_root = extract_urdf(dataset_path)
    #     if all_revolute_joints and all_connect_to_root:
    #         break

    # # Load the urdf into sapien
    # engine = sapien.Engine()
    # renderer = sapien.SapienRenderer()
    # engine.set_renderer(renderer)

    # scene_config = sapien.SceneConfig()
    # scene = engine.create_scene(scene_config)
    # scene.set_timestep(1 / 60.0)
    # scene.add_ground(0)

    # scene.set_ambient_light([0.5, 0.5, 0.5])
    # scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    # viewer = Viewer(renderer)
    # viewer.set_scene(scene)
    # viewer.set_camera_xyz(x=-2, y=0, z=1)
    # viewer.set_camera_rpy(r=0, p=-0.3, y=0)

    # loader = scene.create_urdf_loader()
    # loader.fix_root_link = True
    # loader.scale = 1.0
    # art = loader.load(os.path.join(dataset_path, "REART/urdf/object.urdf"))

    # while not viewer.closed:  # Press key q to quit
    #     scene.step()  # Simulate the world
    #     scene.update_render()  # Update the world to the renderer
    #     viewer.render()