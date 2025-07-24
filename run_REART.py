import os
import shutil
from copy import deepcopy

import sapien.core as sapien
from sapien.utils import Viewer
import networkx as nx

import math
import trimesh
import numpy as np
import pandas as pd
import open3d as o3d
import xml.etree.ElementTree as ET

import constants

from typing import List, Tuple, Set, Dict, Any


def pick_points(pcd):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    return vis.get_picked_points()


def batch_closest_points(queries: np.ndarray, kd_tree: o3d.geometry.KDTreeFlann) -> Tuple[np.ndarray, np.ndarray]:
    results = [kd_tree.search_knn_vector_3d(x, 1) for x in queries]
    indices = np.array([x[1][0] for x in results])
    distances = np.array([x[2][0] for x in results])
    return indices, distances


def tag_pcd(pcd: o3d.geometry.PointCloud, mask: np.ndarray, tag_id: int) -> None:
    colors = np.array(pcd.colors)
    colors[mask] = constants.COLOR_MAP[tag_id]
    pcd.colors = o3d.utility.Vector3dVector(colors)


def tag_difference(pcd_to_tag: o3d.geometry.PointCloud, labels: np.ndarray, pcd_ref: o3d.geometry.PointCloud, tag_id: int) -> List[int]:
    points1 = np.array(pcd_to_tag.points)
    points2 = np.array(pcd_ref.points)
    kd_tree1 = o3d.geometry.KDTreeFlann(pcd_to_tag)
    kd_tree2 = o3d.geometry.KDTreeFlann(pcd_ref)

    moved_labels = []
    invalid_labels = []
    for label in range(0, np.max(labels) + 1):
        if np.count_nonzero(labels == label) == 0:
            continue

        index_forward, distance_forward = batch_closest_points(points1[labels == label], kd_tree2)
        index_backward, distance_backward = batch_closest_points(points2[index_forward], kd_tree1)

        mean_dist_forward = np.median(distance_forward) * 100
        mean_dist_backward = np.median(distance_backward) * 100
        ratio = mean_dist_backward / mean_dist_forward

        if (ratio >= constants.PCD_DIST_RATIO_THRESHOLD and mean_dist_forward < constants.PCD_DIST_MEDIAN_THRESHOLD) and mean_dist_forward <= 0.7:
            invalid_labels.append(label)
        else:
            moved_labels.append(label)

    valid_index = np.in1d(labels, invalid_labels, invert=True)
    colors = np.array(pcd_to_tag.colors)
    colors[valid_index] = constants.COLOR_MAP[tag_id]
    pcd_to_tag.colors = o3d.utility.Vector3dVector(colors)

    return moved_labels


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
    ET.SubElement(inertial, "mass", value="10")
    ET.SubElement(inertial, "inertia", ixx="1", ixy="0", ixz="0", iyy="1", iyz="0", izz="1")


def add_geometry(visual_or_collision: ET.Element, obj_file: str) -> None:
    geometry = ET.SubElement(visual_or_collision, "geometry")
    ET.SubElement(geometry, "mesh", filename=obj_file)


def segment_mesh(dataset_path: str, show_difference: bool = False, show_final_pcd: bool = False, canonical_idx: int = -1) -> None:
    oops_path = os.path.join(dataset_path, "oops")
    mesh_path = os.path.join(dataset_path, "mesh")
    SAM3D_path = os.path.join(dataset_path, "SAM3D")
    REART_path = os.path.join(dataset_path, "REART")
    os.makedirs(REART_path, exist_ok=True)

    state_nums = [int(os.path.splitext(f)[0][-1]) for f in os.listdir(mesh_path) if f.endswith(".glb")]
    state_nums.sort()

    canonical_idx = canonical_idx if canonical_idx >= 0 else len(state_nums) + canonical_idx
    state_nums = state_nums[:canonical_idx + 1]

    raw_meshes = []
    pcds = []
    kd_trees = []
    SAM3D_labels = []
    point2face = []
    for state_num in state_nums:
        mesh = load_mesh(os.path.join(SAM3D_path, f"state_{state_num}", "mesh.ply"), os.path.join(oops_path, f"state_{state_num}_pose_adjusted.npz"))
        pts, face_idx = trimesh.sample.sample_surface_even(mesh, constants.SAMPLE_SIZE)

        raw_meshes.append(mesh)
        point2face.append(face_idx)

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd.paint_uniform_color(constants.COLOR_MAP[0])
        pcds.append(pcd)
        kd_trees.append(o3d.geometry.KDTreeFlann(pcd))

        label = np.load(os.path.join(SAM3D_path, f"state_{state_num}", "labels.npy"))
        SAM3D_labels.append(label[face_idx])

    for i in range(len(state_nums)):
        label_counts = np.bincount(SAM3D_labels[i])
        label_counts[label_counts == 0] = 1000000
        smallest_label = np.argmin(label_counts)
        if label_counts[smallest_label] >= SAM3D_labels[i].shape[0] * constants.CLUSTER_MERGING_THRESHOLD:
            break

        label_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(pcds[i].points)[SAM3D_labels[i] == smallest_label]))
        nearest_label_index = 0
        nearest_label_distance = 1e9
        for target_label in range(0, np.max(SAM3D_labels[i]) + 1):
            if target_label == smallest_label or np.count_nonzero(SAM3D_labels[i] == target_label) == 0:
                continue
            target_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(pcds[i].points)[SAM3D_labels[i] == target_label]))
            distance = np.median(label_pcd.compute_point_cloud_distance(target_pcd))
            if distance < nearest_label_distance:
                nearest_label_index = target_label
                nearest_label_distance = distance
        SAM3D_labels[i][SAM3D_labels[i] == smallest_label] = nearest_label_index

    graphs = []
    for i in range(len(state_nums)):
        edges = []
        label_vertices = []
        for label in range(0, np.max(SAM3D_labels[i]) + 1):
            valid_points = np.where(SAM3D_labels[i] == label)[0]
            label_vertices.append(set(raw_meshes[i].faces[point2face[i][valid_points]].flatten()))

        for j in range(len(label_vertices)):
            for k in range(j + 1, len(label_vertices)):
                if len(label_vertices[j] & label_vertices[k]) > 10:
                    edges.append((j, k))
        graphs.append(nx.Graph(edges))

    affordance_path = os.path.join(dataset_path, "affordance.npy")
    if os.path.exists(affordance_path):
        affordance_pos = np.load(affordance_path)
        for state in range(1, len(state_nums)):
            closest_index = batch_closest_points(affordance_pos[state - 1:state], kd_trees[state])[0][0]
            root_label = SAM3D_labels[state][closest_index]
            graphs[state].graph["root"] = root_label

            visited = set()
            queue = [(root_label, 0)]
            while len(queue):
                current_label, distance = queue.pop(0)
                if current_label in visited:
                    continue
                visited.add(current_label)
                graphs[state].nodes[current_label]["distance"] = distance

                for neighbor in graphs[state].neighbors(current_label):
                    if neighbor not in visited:
                        queue.append((neighbor, distance + 1))

    for i in range(len(pcds) - 1):
        moved_labels = tag_difference(pcds[i + 1], SAM3D_labels[i + 1], pcds[i], i + 1)

        if not os.path.exists(affordance_path):
            continue

        g = graphs[i + 1]
        moved_labels.sort(key=lambda label: g.nodes[label]["distance"])
        if g.graph["root"] not in moved_labels:
            moved_labels.append(g.graph["root"])
            tag_pcd(pcds[i + 1], SAM3D_labels[i + 1] == g.graph["root"], i + 1)

        for label in moved_labels:
            dist = g.nodes[label]["distance"]
            if dist == 0:
                continue

            parent_labels = [node for node in g.neighbors(label) if g.nodes[node]["distance"] == dist - 1]
            if not any([parent in moved_labels for parent in parent_labels]):
                tag_pcd(pcds[i + 1], SAM3D_labels[i + 1] == label, 0)

    pcd_last = pcds[-1]
    pcd_last_color = np.array(pcd_last.colors)
    for i in range(1, len(pcds) - 1):
        color = np.array(pcds[i].colors)
        points_to_cast = np.array(pcds[i].points)[np.all(color == np.array(constants.COLOR_MAP[i]), axis=1)]
        points_to_cast_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_to_cast))

        for label in range(0, np.max(SAM3D_labels[-1]) + 1):
            part = np.array(pcd_last.points)[SAM3D_labels[-1] == label]
            part_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(part))
            if np.mean(part_pcd.compute_point_cloud_distance(points_to_cast_pcd)) < constants.PCD_CASTING_THRESHOLD:
                pcd_last_color[SAM3D_labels[-1] == label] = constants.COLOR_MAP[i]
    pcd_last.colors = o3d.utility.Vector3dVector(pcd_last_color)

    final_points = []
    part_labels = []
    for part_id in range(len(state_nums)):
        part_points = np.array(pcd_last.points)[np.all(np.array(pcd_last.colors) == np.array(constants.COLOR_MAP[part_id]), axis=1)]
        part_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(part_points))
        part_pcd, _ = part_pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.2)
        part_pcd.paint_uniform_color(constants.COLOR_MAP[part_id])

        final_points.append(np.array(part_pcd.points))
        part_labels.append(np.repeat(part_id, final_points[-1].shape[0]))

    final_points = np.concatenate(final_points)
    part_labels = np.concatenate(part_labels)
    final_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(final_points))
    kd_tree = o3d.geometry.KDTreeFlann(final_pcd)
    np.save(os.path.join(REART_path, f"state_{state_nums[-1]}.npy"), final_points)
    np.save(os.path.join(REART_path, f"state_{state_nums[-1]}_labels.npy"), part_labels)

    final_pcd.colors = o3d.utility.Vector3dVector(np.array(constants.COLOR_MAP)[part_labels])
    o3d.io.write_point_cloud(os.path.join(REART_path, f"visualized_input.ply"), final_pcd)

    if show_final_pcd:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([final_pcd, coord_frame])

    export_path = os.path.join(REART_path, "urdf")
    shutil.rmtree(export_path, ignore_errors=True)
    os.makedirs(export_path, exist_ok=True)
    os.makedirs(os.path.join(export_path, "visual"), exist_ok=True)
    os.makedirs(os.path.join(export_path, "collision"), exist_ok=True)

    texture_mesh = load_mesh(os.path.join(mesh_path, f"state_{state_nums[-1]}.glb"),
                             os.path.join(oops_path, f"state_{state_nums[-1]}_pose_adjusted.npz"),
                             np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.29], [0, 0, 0, 1]]))
    avg_face_coord = np.mean(texture_mesh.vertices[texture_mesh.faces], axis=1)

    face_label = part_labels[batch_closest_points(avg_face_coord, kd_tree)[0]]

    for part_id in range(len(state_nums)):
        part_path = os.path.join(export_path, "visual", f"part_{part_id}")
        os.makedirs(part_path, exist_ok=True)

        face_ids = np.where(face_label == part_id)
        if np.count_nonzero(face_ids):
            part_mesh = texture_mesh.submesh(np.where(face_label == part_id))[0]
            part_mesh.export(os.path.join(part_path, "mesh.obj"))

            convex_part_kwargs = trimesh.decomposition.convex_decomposition(part_mesh, maxConvexHulls=constants.CONVEX_DECOMP_RESULT_COUNT)
            meshes = [trimesh.Trimesh(**kwargs) for kwargs in convex_part_kwargs]
            for i, mesh in enumerate(meshes):
                mesh.export(os.path.join(export_path, "collision", f"part_{part_id}_{i}.obj"))


def run_REART(dataset_path: str, farthest_point_sampling: bool = True, *, ablation: bool = False, canonical_idx: int = -1) -> None:
    mesh_path = os.path.join(dataset_path, "mesh")
    REART_path = os.path.join(dataset_path, "REART")
    state_nums = [int(os.path.splitext(f)[0][-1]) for f in os.listdir(mesh_path) if f.endswith(".glb")]
    state_nums.sort()

    canonical_idx = canonical_idx if canonical_idx >= 0 else len(state_nums) + canonical_idx
    state_nums = state_nums[:canonical_idx + 1]

    pcds = []
    for state_num in state_nums:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.loadtxt(os.path.join(REART_path, f"state_{state_num}.xyz"))))
        if farthest_point_sampling:
            pcd = np.array(pcd.farthest_point_down_sample(constants.REART_POINTCLOUD_SIZE).points)
        else:
            pcd = np.array(pcd.points)[np.random.choice(range(len(pcd.points)), size=constants.REART_POINTCLOUD_SIZE)]
        pcds.append(pcd)

    oringin_pcd = np.load(os.path.join(REART_path, f"state_{state_nums[-1]}.npy"))
    labels = np.load(os.path.join(REART_path, f"state_{state_nums[-1]}_labels.npy")) + 1
    if farthest_point_sampling:
        pcd = np.array(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(oringin_pcd)).farthest_point_down_sample(constants.REART_POINTCLOUD_SIZE).points)
        choice = np.array([np.argmin(np.linalg.norm(pcd[i] - oringin_pcd, axis=1)) for i in range(pcd.shape[0])])
    else:
        choice = np.random.choice(range(oringin_pcd.shape[0]), size=constants.REART_POINTCLOUD_SIZE)
        pcd = oringin_pcd[choice]
    labels = labels[choice]
    pcds.insert(canonical_idx, pcd)

    pc = np.stack(pcds, axis=0)
    mean = np.mean(pc, axis=(0, 1), keepdims=True)
    pc = pc - mean
    scale = 0.1 / np.std(pc)
    pc *= scale

    raw_data = np.load(constants.REART_DATA_TEMPLATE, allow_pickle=True)
    trans = {k: v[:pc.shape[0]] for k, v in raw_data["trans"].item().items()}
    np.savez(constants.REART_DATA_PATH, segm=np.tile(labels, (pc.shape[0], 1)), pc=pc, trans=trans)
    np.savez(os.path.join(REART_path, "pcd_transform.npz"), mean=mean[0, 0, :], scale=scale)

    os.chdir(constants.REART_PATH)
    os.system(f"python run_sapien.py --sapien_idx=212 --save_root=exp --num_parts={len(state_nums)} --merge_it=0 --n_iter=2000 --cano_idx={canonical_idx} --use_nproc --use_assign_loss" + (" --ablation" if ablation else ""))
    os.system(f"python run_sapien.py --sapien_idx=212 --save_root=exp --num_parts={len(state_nums)} --merge_it=0 --n_iter=200 --cano_idx={canonical_idx} --model=kinematic --use_nproc --use_assign_loss --assign_iter=0 --assign_gap=1 --snapshot_gap=10 --base_result_path=exp/sapien_212/result.pkl" + (" --ablation" if ablation else ""))
    if not ablation:
        os.system(f"python run_sapien.py --sapien_idx=212 --save_root=exp --num_parts={len(state_nums)} --merge_it=0 --n_iter=200 --cano_idx={canonical_idx} --model=kinematic --use_nproc --use_assign_loss --assign_iter=0 --assign_gap=1 --snapshot_gap=10 --base_result_path=exp/sapien_212/result.pkl --check_revolute_joint_position")

    shutil.copyfile(os.path.join(constants.REART_PATH, "exp", "sapien_212", "result.pkl"), os.path.join(REART_path, "raw_output.pkl"))
    shutil.copyfile(os.path.join(constants.REART_PATH, "exp", "sapien_212", "cano_pc.ply"), os.path.join(REART_path, f"cano_pc{'_ablation' if ablation else ''}.ply"))
    os.system(f"python extract_results.py --sapien_idx=212 --save_root=exp --num_parts={len(state_nums)} --merge_it=0 --n_iter=200 --cano_idx={canonical_idx} --model=kinematic --use_nproc --use_assign_loss --assign_iter=0 --assign_gap=1 --snapshot_gap=10 --base_result_path=exp/sapien_212/result.pkl --export_path={os.path.join(REART_path, f'kinematic_result{'_ablation' if ablation else ''}.npz')}")


def extract_urdf(dataset_path: str, ablation: bool = False) -> Tuple[bool, bool, Dict[str, Any]]:
    dataset_name = os.path.basename(dataset_path)
    REART_path = os.path.join(dataset_path, "REART")
    urdf_folder = os.path.join(REART_path, "urdf")
    os.makedirs(urdf_folder, exist_ok=True)

    has_gt = False
    df = pd.read_csv(constants.GT_FILE_PATH, index_col=0)
    if dataset_name in df.index:
        has_gt = True
        gt_data = df.loc[dataset_name].to_dict()
        gt_origin = constants.parse_grid_coord(gt_data["axis0_origin"])
        gt_axis = np.array([float(x) for x in gt_data["axis0_direction"].split(" ")])
        gt_axis /= np.linalg.norm(gt_axis)

    data = np.load(os.path.join(REART_path, "pcd_transform.npz"), allow_pickle=True)
    mean = data["mean"]
    scale = data["scale"]
    data.close()

    data = np.load(os.path.join(REART_path, f"kinematic_result{'_ablation' if ablation else ''}.npz"), allow_pickle=True)
    root_part = 0
    connection_dict = data["connection_dict"].item()
    data.close()

    ret_dict = {"has_gt": has_gt, "joint_count": len(connection_dict)}
    all_revolute_joints = all([connection["type"] == "revolute" for connection in connection_dict.values()])
    all_connect_to_root = all([str(root_part) in key for key in connection_dict.keys()])

    root = ET.Element("robot", name="object")
    tree = ET.ElementTree(root)

    base_link = ET.SubElement(root, "link", name=f"part_{root_part}")
    add_default_inerital(base_link)
    visual = ET.SubElement(base_link, "visual")
    add_geometry(visual, f"visual/part_{root_part}/mesh.obj")
    for i in range(constants.CONVEX_DECOMP_RESULT_COUNT):
        collision = ET.SubElement(base_link, "collision")
        add_geometry(collision, f"collision/part_{root_part}_{i}.obj")

    origin_mapping = {root_part: [0, 0, 0]}
    for connection in sorted(connection_dict.values(), key=lambda x: x["parent"]):
        axis = connection["axis"] / np.linalg.norm(connection["axis"])
        absolute_origin = (np.cross(axis, connection["moment"]) / scale) + mean
        if connection["parent"] in origin_mapping:
            origin = absolute_origin - origin_mapping[connection["parent"]]
        else:
            origin = absolute_origin
        origin_mapping[connection["child"]] = absolute_origin

        if has_gt:
            axis_rotation_error = math.degrees(math.acos(np.dot(axis, gt_axis)))
            if axis_rotation_error > 90:
                axis_rotation_error = 180 - axis_rotation_error
            ret_dict["axis_rotation_error"] = axis_rotation_error

            if connection["type"] == "revolute":
                common_normal = np.cross(axis, gt_axis)
                axis_translation_error = np.abs(np.dot(common_normal, (absolute_origin - gt_origin))) / np.linalg.norm(common_normal) * 100
                ret_dict["axis_translation_error"] = axis_translation_error
            else:
                ret_dict["axis_translation_error"] = 0.0

        link = ET.SubElement(root, "link", name=f"part_{connection['child']}")
        add_default_inerital(link)
        visual = ET.SubElement(link, "visual")
        add_geometry(visual, f"visual/part_{connection['child']}/mesh.obj")
        ET.SubElement(visual, "origin", xyz=" ".join(map(str, -origin)))
        for i in range(constants.CONVEX_DECOMP_RESULT_COUNT):
            collision = ET.SubElement(link, "collision")
            add_geometry(collision, f"collision/part_{connection['child']}_{i}.obj")
            ET.SubElement(collision, "origin", xyz=" ".join(map(str, -origin)))

        joint = ET.SubElement(root, "joint", name=f"part_{connection['child']}_joint", type=connection["type"])
        ET.SubElement(joint, "origin", xyz=" ".join(map(str, origin)))
        ET.SubElement(joint, "axis", xyz=" ".join(map(str, axis)))
        ET.SubElement(joint, "parent", link=f"part_{connection['parent']}")
        ET.SubElement(joint, "child", link=f"part_{connection['child']}")
        ET.SubElement(joint, "limit", effort="1000", lower="-3.14", upper="3.14", velocity="1000")
        ET.SubElement(joint, "dynamics", damping="0.1", friction="0.1")

    tree.write(os.path.join(urdf_folder, f"object{'_ablation' if ablation else ''}.urdf"))
    return all_revolute_joints, all_connect_to_root, ret_dict


if __name__ == "__main__":
    dataset_path = "/build_kinematic/010602_fridge"

    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene_config = sapien.SceneConfig()
    scene = engine.create_scene(scene_config)
    scene.set_timestep(1 / 60.0)

    viewer = Viewer(renderer)
    viewer.set_scene(scene)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.scale = 1.0
    art = loader.load("/build_kinematic/fig1_011801/REART/changed_texture/object.urdf")

    while not viewer.closed:
        scene.step()
        scene.update_render()
        viewer.render()
