import argparse
import os, shutil
import xml.etree.ElementTree as ET
import functools
import random
import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
import pickle
import torch
from tqdm import tqdm
import open3d as o3d

from utils.chamfer import ChamferDistance # https://github.com/krrish94/chamferdist
from knn_cuda import KNN  # https://github.com/unlimblue/KNN_CUDA

from utils.viz_utils import vis_pc, vis_structure, vis_pc_seq
from utils.model_utils import compute_pc_transform, tau_cosine, compute_ass_err, get_src_permutation_idx, get_tgt_permutation_idx, parallel_lap, compute_align_trans
from utils.kinematic_utils import extract_kinematic, build_graph, edge_index2edges, compute_root_cost
from utils.graph_utils import denoise_seg_label, merging_wrapper, mst_wrapper, compute_screw_cost
from utils.eval_utils import eval_seg, compute_chamfer_list


from dataset.dataset_sapien import Sapien
from utils.sapien_utils import compute_full_flow, eval_flow, load_model, compute_flow_list, seg_propagation_list
from msync.models.full_net import feature_propagation

from networks.model import BaseModel, KinematicModel
from networks.loss import recon_loss, flow_loss
from networks.pointnet2_utils import farthest_point_sample, index_points
from networks.feature_extractor import get_extractor

COLORS = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
)


def add_default_inerital(link):
    inertial = ET.SubElement(link, "inertial")
    mass = ET.SubElement(inertial, "mass", value="10")
    inertia = ET.SubElement(inertial, "inertia",
                            ixx="1", ixy="0", ixz="0",
                            iyy="1", iyz="0", izz="1")

def add_geometry(visual_or_collision, obj_file):
    geometry = ET.SubElement(visual_or_collision, "geometry")
    mesh = ET.SubElement(geometry, "mesh",
                          filename=obj_file)

def main(args):
    # Initialize randoms seeds
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    dataset = Sapien(args.sapien_base_folder, cano_idx=args.cano_idx)
    exp_name = "sapien_{}".format(args.sapien_idx)
    save_dir = os.path.join(args.save_root, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")

    # chamfer_dist = ChamferDistance()
    sample = dataset[args.sapien_idx]
    cano_pc = torch.from_numpy(sample['cano_pc']).float().to(device)

    # complete_pc_list = torch.from_numpy(sample['complete_pc_list']).float().to(device)
    # pc_list = torch.from_numpy(sample['pc_list']).float().to(device)  # exclude cano frame
    # cano_idx = dataset.cano_idx
    
    # visualize input point cloud sequence
    # save_path = os.path.join(save_dir, f"input.gif")
    # vis_pc_seq(sample['complete_pc_list'], name="input", save_path=save_path)
    # print("save input pc vis to {}".format(save_path))

    # complete_gt_part_list = torch.from_numpy(sample['complete_gt_part_list']).long().to(device)
    # gt_full_flow = torch.from_numpy(sample['gt_full_flow']).to(device)

    # if args.use_flow_loss:
    #     # use multibody-sync predicted flow
    #     flow_model = load_model(config_path=args.flow_model_config_path, model_path=args.flow_model_path)
    #     flow_model.to(device)
    #     flow_model.eval()
    #     knn_flow = KNN(k=3, transpose_mode=True)
    #     complete_pc_list = torch.from_numpy(sample['complete_pc_list']).float().to(device)
    #     flow_ref_list, _ = compute_flow_list(flow_model, complete_pc_list)
    #     pc_ref_list = complete_pc_list[:-1]


    assert args.base_result_path is not None
    with open(os.path.join(args.base_result_path), 'rb') as f:
        result = pickle.load(f)
    print(f"load base result from {args.base_result_path}")
    assert args.cano_idx == result['cano_idx']
    seg_part = torch.from_numpy(result['pred_cano_part']).long().to(device)
    trans_list = torch.from_numpy(result['pred_pose_list']).float().to(device)

    joint_connection = torch.from_numpy(np.array(result['joint_connection'])).long().to(device)
    # new_trans_list: (T, 20, 4, 4), trans list of each part
    # new_connection: (E, 2), edge list
    new_seg, new_trans_list, new_connection = extract_kinematic(seg_part, trans_list, joint_connection)
    root_part = torch.mode(new_seg).values.item()
    root_trans = trans_list[:, root_part]
    align_trans_list = compute_align_trans(new_trans_list, root_trans)
    G, root_part, axis_list, moment_list, theta_list, distance_list, edge_index, joint_type_list = build_graph(
        new_connection, align_trans_list,
        verbose=False, revolute_only=False, root_part=root_part, return_joint_type=True, cano_points=cano_pc)

    run_folder = "/home/rvsa/gary318/reart/exp/sapien_%d" % args.sapien_idx

    # Create folder
    urdf_folder = os.path.join(run_folder, "urdf")
    if os.path.exists(urdf_folder):
        shutil.rmtree(urdf_folder)
    os.makedirs(urdf_folder)

    # Create meshes
    obj_folder = os.path.join(urdf_folder, "meshes")
    os.makedirs(obj_folder)
    meshes = []
    pcds = []
    min_link_id, max_link_id = seg_part.min().item(), seg_part.max().item()
    for i in range(min_link_id, max_link_id + 1):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cano_pc[seg_part == i, :].cpu().numpy())
        pcd.paint_uniform_color(COLORS[i])
        pcds.append(pcd)

        mesh, _ = pcd.compute_convex_hull()
        meshes.append(mesh)
        
        o3d.io.write_triangle_mesh(os.path.join(obj_folder, "part_%d.obj" % i), mesh)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries(pcds + [coord_frame])

    root = ET.Element("robot", name="object")
    tree = ET.ElementTree(root)

    # base link
    base_link = ET.SubElement(root, "link", name="part_%d" % root_part)
    add_default_inerital(base_link)
    visual = ET.SubElement(base_link, "visual")
    add_geometry(visual, "meshes/part_%d.obj" % root_part)

    connection_dict = dict(
        [
            ("%d_%d" % (new_connection[i, 1].item(), new_connection[i, 0].item()), {
                "parent": new_connection[i, 1].item(),
                "child": new_connection[i, 0].item(),
                "type": joint_type_list[i],
                "axis": axis_list[i],
                "moment": moment_list[i],
            }) for i in range(new_connection.shape[0])
        ]
    )
    print("Root part: %d" % root_part)
    print(connection_dict)

    for connection in connection_dict.values():
        # Create link
        link = ET.SubElement(root, "link", name="part_%d" % connection["child"])
        add_default_inerital(link)
        visual = ET.SubElement(link, "visual")
    
        origin = torch.cross(connection["axis"], connection["moment"])
        add_geometry(visual, "meshes/part_%d.obj" % connection["child"])
        ET.SubElement(visual, "origin", xyz="%f %f %f" % tuple(-origin))

        # Create joint
        joint = ET.SubElement(root, "joint", name="part_%d_joint" % connection["child"], type=connection["type"])
        ET.SubElement(joint, "origin", xyz="%f %f %f" % tuple(origin))
        ET.SubElement(joint, "axis", xyz="%f %f %f" % tuple(connection["axis"]))
        ET.SubElement(joint, "parent", link="part_%d" % connection["parent"])
        ET.SubElement(joint, "child", link="part_%d" % connection["child"])
        ET.SubElement(joint, "limit", effort="1000", lower="-1.57", upper="1.57", velocity="1000")

    # for connection in connection_dict:
    #     if connection["child"] == root_part:
    #         continue

    #     # Create link
    #     link = ET.SubElement(root, "link", name="part_%d" % connection["child"])
    #     add_default_inerital(link)

    #     visual = ET.SubElement(link, "visual")

    #     connection = connection_dict["%d_%d" % (root_part, connection["child"])]
    #     assert connection["type"] == "revolute"
    
    #     origin = torch.cross(connection["axis"], connection["moment"])
    #     add_geometry(visual, "meshes/part_%d.obj" % connection["child"])
    #     ET.SubElement(visual, "origin", xyz="%f %f %f" % tuple(-origin))

    #     # Create joint
    #     joint = ET.SubElement(root, "joint", name="part_%d_joint" % connection["child"], type="revolute")
    #     ET.SubElement(joint, "origin", xyz="%f %f %f" % tuple(origin))
    #     ET.SubElement(joint, "axis", xyz="%f %f %f" % tuple(connection_dict["%d_%d" % (root_part, connection["child"])]["axis"]))
    #     ET.SubElement(joint, "parent", link="part_%d" % root_part)
    #     ET.SubElement(joint, "child", link="part_%d" % connection["child"])
    #     ET.SubElement(joint, "limit", effort="1000", lower="-1.57", upper="1.57", velocity="1000")

    # # joints
    # for i, door_id in enumerate(range(1, 3)):
    #     joint = ET.SubElement(root, "joint", name="door_%d_joint" % door_id, type="revolute")
    #     ET.SubElement(joint, "origin",
    #                   xyz=list_to_space_sep_string(motion_json[i]["axis_position"]))
    #     ET.SubElement(joint, "axis", xyz=list_to_space_sep_string(motion_json[i]["axis_direction"]))
    #     ET.SubElement(joint, "parent", link="base")
    #     ET.SubElement(joint, "child", link="door_%d" % door_id)
    #     ET.SubElement(joint, "limit", effort="1000", lower="-1.57", upper="1.57", velocity="1000")

    # Write to file
    tree.write(os.path.join(urdf_folder, "object.urdf"))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sapien")
    # common
    parser.add_argument("--manual_seed", default=2, type=int, help="manual seed")
    parser.add_argument("--resume", type=str, nargs="+", metavar="PATH",
                        help="path to latest checkpoint (default: none)")
    parser.add_argument("--evaluate", dest="evaluate", action="store_true", help="evaluate mode")
    parser.add_argument("--snapshot_gap", default=100, type=int,
                        help="How often to take a snapshot vis of the training")
    parser.add_argument("--use_cuda", default=1, type=int, help="use GPU (default: True)")

    # dataset
    parser.add_argument("--cano_idx", default=0, type=int, help="cano frame idx")
    parser.add_argument("--seq_path", default="/home/shaowei3/datasets/articulation/robot/nao/test_1", type=str)
    
    # optimization
    parser.add_argument("--start_tau", default=1, type=float, help="gumbel softmax start temperature")
    parser.add_argument("--end_tau", default=1, type=float, help="gumbel softmax end temperature")
    parser.add_argument("--seg_lr", default=1e-3, type=float, help="seg MLP learning rate")
    parser.add_argument("--trans_lr", default=1e-2, type=float, help="seg MLP learning rate")
    parser.add_argument("--weight_decay", default=0, type=float)

    parser.add_argument("--n_iter", default=2000, type=int, help="number of optimization iterations")
    parser.add_argument("--assign_iter", default=1000, type=int, help="iteration apply assignment loss")

    # network
    parser.add_argument("--num_parts", default=10, type=int, help="seg MLP number of parts")
    parser.add_argument("--model", default="base", type=str, choices=['base', 'kinematic'], help="model type")
    parser.add_argument("--base_result_path", default=None, type=str, help="kinematic model initialization")

    # flow
    parser.add_argument("--use_flow_loss", action="store_true", help="use flow loss")

    # other constraints
    parser.add_argument("--use_assign_loss", action="store_true", help="use pc assignment loss")
    
    parser.add_argument("--use_nproc", action="store_true", help="use multi process to compute assignment loss")
    parser.add_argument("--downsample", default=1, type=int, help="downsample rate when computing assignment loss")
    parser.add_argument("--assign_gap", default=5, type=int, help="assignment loss gap")

    # loss weight
    parser.add_argument("--lambda_assign", default=3e-1, type=float, help="assignment loss weight")
    parser.add_argument("--lambda_flow", default=1, type=float, help="flow loss weight")
    parser.add_argument("--lambda_joint", default=1e-1, type=float, help="joint cost/loss weight")

    # structure_utils
    parser.add_argument("--cano_dist_thr", default=1e-2, type=float,
                        help="mst cano dist threshold (below consider an edge candidate)")
    parser.add_argument("--merge_thr", default=3e-2, type=float, help="graph geo merging threshold")
    parser.add_argument("--merge_it", default=3, type=int, help="graph geo merging iteration")

    # utils func
    parser.add_argument("--save_root", default="exp", type=str, help="results saving path")
   
    # sapien utils
    parser.add_argument("--sapien_base_folder", default="data/mbs-sapien", type=str, help="sapien dataset base folder")
    parser.add_argument("--sapien_idx", default=212, type=int, help="sapien dataset test index")
    parser.add_argument("--flow_model_config_path", type=str, default="msync/config/articulated-full.yaml")
    parser.add_argument("--flow_model_path", type=str, default="msync/ckpt/articulated-full/best.pth.tar")
    
    args = parser.parse_args()
    os.makedirs(args.save_root, exist_ok=True)
    main(args)