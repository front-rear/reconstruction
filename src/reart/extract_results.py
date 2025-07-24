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
from utils.kinematic_utils import extract_kinematic, build_graph


from dataset.dataset_sapien import Sapien

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
        verbose=False, revolute_only=False, root_part=root_part, return_joint_type=True, cano_points=cano_pc, seg_part=seg_part)

    new_connection = new_connection.cpu().numpy()
    axis_list = axis_list.cpu().numpy()
    moment_list = moment_list.cpu().numpy()
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

    np.savez(args.export_path, root_part=root_part, connection_dict=connection_dict)


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

    parser.add_argument("--export_path", type=str)

    args = parser.parse_args()
    os.makedirs(args.save_root, exist_ok=True)
    main(args)