import csv
import glob
import os
import copy

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics


import Utils
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import MaxPool2d

# from Depth_Anything_V2_main.metric_depth.depth_anything_v2.dpt import DepthAnythingV2 as DepthAnythingV2_metric

import pandas as pd





def objid_2_objname(dataset,scene_id=None,base_dir=None):
    if base_dir == None:
        if dataset == "clearpose":
            base_dir = "/mnt/disk0/dataset/clearpose/downsample"
        elif dataset == "transpose":
            base_dir = "/mnt/disk0/dataset/TRansPose/downsample"
    if dataset == "activezero":
        OBJECT_INFO = ["beer_can","camera","cellphone","champagne","coca_cola","coffee_cup","coke_bottle","gold_ball","hammer","jack_daniels","pepsi_bottle","rubik","sharpener","spellegrino","steel_ball","tennis_ball","voss"]

        objid_2_objname = {i: _ for i, _ in enumerate(OBJECT_INFO)}
    elif dataset == "glassmolder":
        csv_file = csv.reader(open("/mnt/disk0/dataset/transtouch_pc2_2/objects_v2.csv", "r"))
        OBJECT_NAME = []

        for line in csv_file:
            OBJ_NAME = line[0]
            OBJECT_NAME.append(OBJ_NAME)

        objid_2_objname = {(i-1): _ for i, _ in enumerate(OBJECT_NAME)}
    elif dataset == "linemodocc":

        # OBJECT_NAME = ["ape", "waterpour", "cat", "driller", "duck", "eggbox", "glue", "holepuncher"]
        OBJECT_NAME = [
            'ape',
            'benchvise',
            'bowl',
            'camera',
           'water_pour',
            'cat',
            'cup',
            'driller',
            'duck',
            'eggbox',
            'glue',
            'holepuncher',
            'iron',
            'lamp',
            'phone',
        ]
        objid_2_objname = {(i+1): _ for i, _ in enumerate(OBJECT_NAME)}
    elif dataset == "clearpose":
        csv_path = f"{base_dir}/ClearPose-main/data/objects.csv"
        csv_file = csv.reader(open(csv_path, "r"))
        OBJECT_NAME = []
        for line in csv_file:
            OBJECT_NAME.append(line[1])
        # OBJECT_NAME = ["ape", "waterpour", "cat", "driller", "duck", "eggbox", "glue", "holepuncher"]

        objid_2_objname = {(i+1): _ for i, _ in enumerate(OBJECT_NAME)}
    elif dataset == "housecat6d":
        obj_path = f"/mnt/disk0/dataset/housecat6d/test_scene/test_scene{scene_id}/sim_obj"
        obj_names_path = sorted(glob.glob(f"{obj_path}/*.obj"))
        obj_names = [obj_name.split('/')[-1].split('.')[0] for obj_name in obj_names_path if
                     obj_name.split('/')[-1].split('.')[0] != "mesh_processed"]
        objid_2_objname = {(i): _ for i, _ in enumerate(obj_names)}

    elif dataset == "transpose":

        json_path = f"{base_dir}/models/object_label.json"
        import json
        with open(json_path, "r") as f:
            objname2objid = json.load(f)

        objid_2_objname = {int(objname2objid[obj_name]): obj_name for obj_name in objname2objid.keys()}

    return objid_2_objname

def depth2normal(depth):
    dx = -cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
    dy = -cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
    dz = np.ones_like(depth)
    normal = np.stack([dx, dy, dz], axis=-1)
    normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)

    normal_vis = (normal + 1) / 2
    normal_vis = (normal_vis * 255).astype(np.uint8)

    return normal, normal_vis

class Sobel_torch(nn.Module):

    def __init__(self):
        super(Sobel_torch, self).__init__()
        kernel_v = [[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]]
        kernel_h = [[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0).cuda()
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0).cuda()
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    # def get_gray(self,x):
    #     '''
    #     Convert image to its gray one.
    #     '''
    #     gray_coeffs = [65.738, 129.057, 25.064]
    #     convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
    #     x_gray = x.mul(convert).sum(dim=1)
    #     return x_gray.unsqueeze(1)

    def forward(self, x):
        # x_list = []
        # for i in range(x.shape[1]):
        #     x_i = x[:, i]
        #     x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
        #     x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
        #     x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
        #     x_list.append(x_i)

        # x = torch.cat(x_list, dim=1)
        # if x.shape[1] == 3:
        #     x = self.get_gray(x)

        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x_v,x_h,x
def depth2normal_torch(depth):
    sobel = Sobel_torch()
    dx,dy,dz = sobel(depth)
    dz = torch.ones_like(depth)
    normal = torch.stack([dx, dy, dz], dim=-1)
    normal = normal / torch.norm(normal, dim=-1, keepdim=True)

    normal_vis = (normal + 1) / 2
    normal_vis = (normal_vis * 255)
    normal_vis = normal_vis.detach().squeeze(0).cpu().numpy().astype(np.uint8)
    # plt.imshow(normal_vis)
    # plt.show()

    return normal

def objname_2_objid(dataset,scene_id=None,base_dir = None):
    if dataset == "activezero":

        OBJECT_INFO = ["beer_can","camera","cellphone","champagne","coca_cola","coffee_cup","coke_bottle","gold_ball","hammer","jack_daniels","pepsi_bottle","rubik","sharpener","spellegrino","steel_ball","tennis_ball","voss"]
        objname_2_objid = {_:i  for i, _ in enumerate(OBJECT_INFO)}
    elif dataset == "glassmolder":
        csv_file = csv.reader(open("/mnt/disk0/dataset/transtouch_pc2_2/objects_v2.csv", "r"))
        OBJECT_NAME = []
        for line in csv_file:
            OBJ_NAME = line[0]
            OBJECT_NAME.append(OBJ_NAME)

        objname_2_objid = {_: (i - 1) for i, _ in enumerate(OBJECT_NAME)}
    elif dataset == "linemodocc":

        OBJECT_NAME = [
            'ape',
            'benchvise',
            'bowl',
            'camera',
            'water_pour',
            'cat',
            'cup',
            'driller',
            'duck',
            'eggbox',
            'glue',
            'holepuncher',
            'iron',
            'lamp',
            'phone',
        ]

        objname_2_objid = {_: (i+1) for i, _ in enumerate(OBJECT_NAME)}
    elif dataset == "clearpose":
        base_dir = "/mnt/disk0/dataset/clearpose/downsample"
        csv_path = f"/{base_dir}/ClearPose-main/data/objects.csv"
        csv_file = csv.reader(open(csv_path, "r"))
        OBJECT_NAME = []
        for line in csv_file:
            OBJECT_NAME.append(line[1])
        # OBJECT_NAME = ["ape", "waterpour", "cat", "driller", "duck", "eggbox", "glue", "holepuncher"]

        objname_2_objid = {_: (i+1) for i, _ in enumerate(OBJECT_NAME)}

    elif dataset == "housecat6d":
        obj_path = f"/mnt/disk0/dataset/housecat6d/test_scene/test_scene{scene_id}/sim_obj"
        obj_names_path = sorted(glob.glob(f"{obj_path}/*.obj"))
        obj_names = [obj_name.split('/')[-1].split('.')[0] for obj_name in obj_names_path if obj_name.split('/')[-1].split('.')[0] != "mesh_processed"]
        objname_2_objid = {_: (i) for i, _ in enumerate(obj_names)}

    elif dataset == "transpose":
        json_path = f"{base_dir}/models/object_label.json"
        import json
        with open(json_path, "r") as f:
            objname_2_objid = json.load(f)


    return objname_2_objid
def img_name_2_img_id(dataset):
    if dataset == "glassmolder":

        with open("/mnt/disk0/dataset/transtouch_pc2_hcp/split_file.txt", "r") as f:
            prefix = [line.strip() for line in f]
        img_name_2_img_id = {img_name: i for i, img_name in enumerate(prefix)}
        return img_name_2_img_id
    elif dataset == "activezero":
        with open("/mnt/disk0/dataset/rand_scenes/list_file_rand_second.txt", "r") as f:
            prefix = [line.strip() for line in f]
        img_name_2_img_id = {img_name: i for i, img_name in enumerate(prefix)}
        return img_name_2_img_id
def img_id_2_img_name(dataset):
    if dataset == "glassmolder":

        with open("/mnt/disk0/dataset/transtouch_pc2_hcp/split_file.txt", "r") as f:
            prefix = [line.strip() for line in f]
        img_id_2_img_name = {i: img_name for i, img_name in enumerate(prefix)}
        return img_id_2_img_name
    elif dataset == "activezero":
        with open("/mnt/disk0/dataset/rand_scenes/list_file_rand_second.txt", "r") as f:
            prefix = [line.strip() for line in f]
        img_id_2_img_name = {i: img_name for i, img_name in enumerate(prefix)}
        return img_id_2_img_name

def find_according_imid_objid_new(result,scene_id,image_id,obj_id):
    return result[scene_id][image_id][obj_id]["R"], result[scene_id][image_id][obj_id]["t"]

def find_according_imid_objid(result,image_id,obj_id):
    return result[image_id][obj_id]["R"], result[image_id][obj_id]["t"]

def found_gt_pose(dataset, scene_id, set_id, img_id, obj_id):
    # load the gt pose
    if dataset == "clearpose":
        result_gt_path = f"/home/rvsa/disk0/pzh/foundationpose/FoundationPose-main/{dataset}/debug/clearpose_gt_set{set_id}_scene{scene_id}.csv"
        result_gt_path_dict = f"/home/rvsa/disk0/pzh/foundationpose/FoundationPose-main/{dataset}/debug/clearpose_gt_set{set_id}_scene{scene_id}.npy"
        if os.path.exists(result_gt_path_dict):
            result_gt = np.load(result_gt_path_dict, allow_pickle=True).item()
        else:
            result_gt = pd.read_csv(result_gt_path)
            result_gt = csv_to_dict(result_gt)
            np.save(result_gt_path_dict, result_gt)

        result_gt = result_gt[img_id]
        obj_list = list(result_gt.keys())
        try:

            result_gt = result_gt[obj_id]
        except:
            print(f"{scene_id} {img_id} {obj_id} missing")
            print("in obj_list", obj_list)
            return None
        R_gt, t_gt = result_gt["R"], result_gt["t"]
        R_gt = R_transform(R_gt)
        t_gt = t_transform(t_gt)
        T_gt = np.eye(4)
        T_gt[:3, :3] = R_gt
        T_gt[:3, 3] = t_gt
        gt_pose = T_gt
    elif dataset == "transpose":
        pose_gt = np.load("/mnt/disk0/pzh/foundationpose/FoundationPose-main/transpose/debug/transpose_gt.npy",
                          allow_pickle=True).item()
        pose_gt = pose_gt[scene_id][img_id][obj_id]
        R = pose_gt["R"]
        t = pose_gt["t"]
        R = R_transform(R)
        t = t_transform(t)
        T_gt = np.eye(4)
        T_gt[:3, :3] = R
        T_gt[:3, 3] = t
        gt_pose = T_gt

    return gt_pose

def xyz_2_pts(xyz):
    '''
    Transfer xyz map to point cloud.

    input:
        :np.array (3,h,w), xyz map
    output:
        :o3d.geometry.Pointcloud
    '''

    pts = np.reshape(xyz,(3,-1)).T
    pts_o3d = o3d.geometry.PointCloud()
    pts_vec = o3d.utility.Vector3dVector(pts)
    pts_o3d.points = pts_vec
    pts = copy.deepcopy(pts_o3d)

    return pts
def sample_uniform_from_edge(self,sample_num,batch_size,edge_pred_deoccluded,optimizer_translation,optimizer_rotation,batch_pose_reconstruct,relative_depth_rgb,H,W):


    for b in range(batch_size):
      edge_pred_deoccluded_reshape = edge_pred_deoccluded[b].reshape(-1)
      idx = torch.argwhere(edge_pred_deoccluded_reshape > 0.1)
      try:

        idx_total = torch.randperm(idx.shape[0])
        idx_last_sample_num = idx_total[sample_num:]
        idx_last = idx[idx_last_sample_num]
        idx_last = idx_last.squeeze(1)
        edge_pred_deoccluded_reshape[idx_last] = 0
        edge_pred_deoccluded_reshape[edge_pred_deoccluded_reshape < 0.1] = 0
        edge_pred_deoccluded[b] = edge_pred_deoccluded_reshape.reshape(H, W)
      except Exception as e:
        print(e)
        print("error")
        print("idx:", idx)
        pass

    return edge_pred_deoccluded

def xyz_map_2_pts_torch(xyz_map):
    '''

    Transfer xyz map to point cloud.

    '''



    batch_num = xyz_map.shape[0]
    pts_num = xyz_map.shape[1] * xyz_map.shape[2]

    #xyz map: B x H X W X 3
    pts = xyz_map.reshape(batch_num,pts_num,4)
    pts = pts[:,:,:3]
    #

    pts_opencv = torch.stack((pts[:,:,0],-pts[:,:,1],-pts[:,:,2]),dim = 2)



    pts_opencv[pts_opencv[:,:,2] > 2] = torch.tensor([0,0,0],dtype = torch.float32).cuda()
    pts_opencv[pts_opencv[:,:,2] < 0] = torch.tensor([0,0,0],dtype = torch.float32).cuda()
    pts_z = pts_opencv[:, :, 2]
    num_points = torch.sum(pts_z > 0,dim = 1)
    sample_num = int(max(num_points))
    pts_final = torch.zeros((batch_num,sample_num,3)).cuda()
    print("max z ", torch.max(pts_z))
    for b in range(batch_num):
        pts_non_zero_depth = pts_opencv[b][pts_z[b] > 0]
        pts_num = num_points[b]
        try:
            if pts_num < sample_num:
                degree = int(sample_num / pts_num) + 1
                padding_num = sample_num - pts_num
                pts_num = degree * pts_num
                pts_non_zero_depth_repeat = pts_non_zero_depth.repeat(degree,1)
                idx_padding = torch.randperm(pts_num)
                pts_padding = pts_non_zero_depth_repeat[idx_padding][:padding_num]
                pts_final[b] = torch.cat((pts_non_zero_depth,pts_padding),dim = 0)
        except Exception as e:
            print(e)
            print("error")
            print("pts_num:",pts_num)
            print("sample_num:",sample_num)
            pass






    return pts

def xyz_map_2_pts_open3d(xyz_map,batch=0):
    '''

    Transfer xyz map to point cloud.

    '''



    batch_num = xyz_map.shape[0]
    pts_num = xyz_map.shape[1] * xyz_map.shape[2]

    #xyz map: B x H X W X 3
    pts = xyz_map.reshape(batch_num,pts_num,4)
    pts = pts[:,:,:3]
    #

    pts_opencv = torch.stack((pts[:,:,0],-pts[:,:,1],-pts[:,:,2]),dim = 2)



    pts_opencv[pts_opencv[:,:,2] > 2] = torch.tensor([0,0,0],dtype = torch.float32).cuda()
    pts_opencv[pts_opencv[:,:,2] < 0] = torch.tensor([0,0,0],dtype = torch.float32).cuda()
    pts_z = pts_opencv[:, :, 2]
    num_points = torch.sum(pts_z > 0,dim = 1)
    sample_num = int(max(num_points))
    pts_final = torch.zeros((batch_num,sample_num,3)).cuda()
    print("max z ", torch.max(pts_z))
    for b in range(batch_num):
        pts_non_zero_depth = pts_opencv[b][pts_z[b] > 0]
        pts_num = num_points[b]
        try:
            if pts_num < sample_num:
                degree = int(sample_num / pts_num) + 1
                padding_num = sample_num - pts_num
                pts_num = degree * pts_num
                pts_non_zero_depth_repeat = pts_non_zero_depth.repeat(degree,1)
                idx_padding = torch.randperm(pts_num)
                pts_padding = pts_non_zero_depth_repeat[idx_padding][:padding_num]
                pts_final[b] = torch.cat((pts_non_zero_depth,pts_padding),dim = 0)
        except Exception as e:
            print(e)
            print("error")
            print("pts_num:",pts_num)
            print("sample_num:",sample_num)
            pass

    pts_b = pts_final[batch]
    pts_b = pts_b.detach().cpu().numpy()
    pts_b = np.asarray(pts_b)
    pts_o3d = o3d.geometry.PointCloud()
    pts_vec = o3d.utility.Vector3dVector(pts_b)
    pts_o3d.points = pts_vec
    # o3d.visualization.draw_geometries([pts_o3d])

    # paint as green
    color = np.zeros((pts.shape[0], 3))
    color[:, 1] = 1
    pts_o3d.colors = o3d.utility.Vector3dVector(color)


    return pts_o3d


def depth_2_pts_torch(depth,K,sample_num = 500):
    depth = depth.T
    K = torch.tensor(K).cuda()
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x = torch.linspace(0,depth.shape[0]-1,depth.shape[0]).cuda()
    y = torch.linspace(0,depth.shape[1]-1,depth.shape[1]).cuda()
    meshy, meshx = torch.meshgrid(y,x)

    depth = depth.T

    X = (meshx - cx + 0.5) / fx * depth
    Y = (meshy - cy + 0.5) / fy * depth

    pts_x = X.reshape(-1)
    pts_y = Y.reshape(-1)
    pts_z = depth.reshape(-1)


    pts_x = torch.unsqueeze(pts_x,1)
    pts_y = torch.unsqueeze(pts_y,1)
    pts_z = torch.unsqueeze(pts_z,1)
    pts_new = []
    pts = torch.cat((pts_x,pts_y,pts_z),1)
    pts = pts[pts[:,2] > 0]

    # idx = torch.randperm(sample_num)
    # pts = pts[idx]

    return pts

def depth_2_pts_open3d(depth,K,sample_num = 500):
    depth = depth.T
    K = torch.tensor(K).cuda()
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x = torch.linspace(0,depth.shape[0]-1,depth.shape[0]).cuda()
    y = torch.linspace(0,depth.shape[1]-1,depth.shape[1]).cuda()
    meshy, meshx = torch.meshgrid(y,x)

    depth = depth.T

    X = (meshx - cx + 0.5) / fx * depth
    Y = (meshy - cy + 0.5) / fy * depth

    pts_x = X.reshape(-1)
    pts_y = Y.reshape(-1)
    pts_z = depth.reshape(-1)


    pts_x = torch.unsqueeze(pts_x,1)
    pts_y = torch.unsqueeze(pts_y,1)
    pts_z = torch.unsqueeze(pts_z,1)
    pts_new = []
    pts = torch.cat((pts_x,pts_y,pts_z),1)
    pts = pts[pts[:,2] > 0]

    # idx = torch.randperm(sample_num)
    # pts = pts[idx]

    pts = pts.detach().cpu().numpy()
    pts = np.asarray(pts)
    pts_o3d = o3d.geometry.PointCloud()
    pts_vec = o3d.utility.Vector3dVector(pts)
    pts_o3d.points = pts_vec

    #paint as blue
    color = np.zeros((pts.shape[0],3))
    color[:,0] = 1
    pts_o3d.colors = o3d.utility.Vector3dVector(color)



    return pts_o3d
def depth_2_pts(depth,K,normalize = False,open_3d = True,xyz = False,rgb = None):
    '''
    Project the depth to point cloud given depth map and camera intrinsic.

    Input:
        depth: np.array (h,w)
        K: np.array (3,3) , camera intrinsic,
        normalize: Normalize the points by the (farest distance / 2) between points if True.
        open_3d: Return the o3d.geometry.Pointcloud if True.
        xyz: Transfer xyz map to point cloud if "depth" is a xyz map.

    Output:
        pts: np.array (N,3) or o3d.geometry.Pointcloud , Point cloud according to depth and camera intrinsic.
    '''
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    depth = depth.T
    x = np.linspace(0,depth.shape[0]-1,depth.shape[0])
    y = np.linspace(0,depth.shape[1]-1,depth.shape[1])
    meshx, meshy = np.meshgrid(x,y)

    depth = depth.T

    X = (meshx - cx + 0.5) / fx * depth
    Y = (meshy - cy + 0.5) / fy * depth

    pts_x = X.reshape(-1)
    pts_y = Y.reshape(-1)
    pts_z = depth.reshape(-1)

    pts_x = np.expand_dims(pts_x,1)
    pts_y = np.expand_dims(pts_y,1)
    pts_z = np.expand_dims(pts_z,1)

    pts = np.concatenate((pts_x,pts_y,pts_z),1)
    color = np.zeros((pts.shape[0],3))
    if rgb is not None:
        color = rgb.reshape(-1,3) / 255

    pts_new = []
    color_new = []
    for i in range(pts.shape[0]):
        # print(pts[i,:])
        if pts[i,2] < 2:
            if pts[i, 0] != 0 or pts[i, 1] != 0 or pts[i, 2] != 0 :
                pts_new.append(pts[i, :])
                color_new.append(color[i,:])



    pts = np.array(copy.deepcopy(pts_new))

    if normalize:

        point_ref_max = np.array([-np.infty,-np.infty,-np.infty])
        point_ref_min = np.array([np.infty, np.infty, np.infty])

        #normalization
        for i in range(len(pts_new)):
            point = pts_new[i]
            point_ref_max = np.maximum(point_ref_max,point)
            point_ref_min = np.minimum(point_ref_min,point)
        diagnal = np.linalg.norm(point_ref_max- point_ref_min) / 2

        pts = np.asarray(pts_new)
        pts = (pts - np.mean(pts)) / diagnal

    if xyz:
        pts = np.reshape(depth,(3,-1))

    if open_3d:
        pts_o3d = o3d.geometry.PointCloud()
        pts_vec = o3d.utility.Vector3dVector(pts)
        pts_o3d.points = pts_vec
        if rgb is not None:
            pts_o3d.colors = o3d.utility.Vector3dVector(color_new)
        pts = copy.deepcopy(pts_o3d)



    return pts

def pts2depth(pts,K,w,h):
    focalx,focaly,ax,ay = K[0,0],K[1,1],K[0,2],K[1,2]
    points = np.asarray(pts.points) #(n,3)
    point_x = np.round(points[:,0] / points[:,2] * focalx + ax).astype(int)
    point_y = np.round(points[:,1] / points[:,2] * focaly + ay).astype(int)
    point_z = points[:,2]

    valid = np.bitwise_and(np.bitwise_and((point_x >= 0), (point_x < w)),
                           np.bitwise_and((point_y >= 0), (point_y < h)))
    point_x, point_y, point_z = point_x[valid], point_y[valid], point_z[valid]
    img_z = np.full((h, w), np.inf)
    for ui, vi, zi in zip(point_x, point_y, point_z):
        img_z[vi, ui] = min(img_z[vi, ui], zi)  # 近距离像素屏蔽远距离像素

    # 小洞和“透射”消除
    img_z_shift = np.array([img_z,
                            np.roll(img_z, 1, axis=0),
                            np.roll(img_z, -1, axis=0),
                            np.roll(img_z, 1, axis=1),
                            np.roll(img_z, -1, axis=1)])
    img_z = np.min(img_z_shift, axis=0)
    # cv2.imshow("name",img_z)
    # cv2.waitKey()

    # 保存重新投影生成的深度图dep_rot


    return img_z


def edge2D_2_edge3D(edge_inner):
    H, W = edge_inner.shape
    grid_width = np.linspace(0, W - 1, W)
    grid_height = np.linspace(0, H - 1, H)
    grid_x, grid_y = np.meshgrid(grid_width, grid_height)
    edge_inner_pts = np.stack([grid_x[edge_inner > 0], grid_y[edge_inner > 0]], axis=1)
    edge_inner_pts = np.concatenate([edge_inner_pts, np.zeros((edge_inner_pts.shape[0], 1))], axis=1)
    pts = o3d.geometry.PointCloud()
    pts.points = o3d.utility.Vector3dVector(edge_inner_pts)
    return pts


def find_deoccluded_edge(edge_inner, edge_outter, depth):
    edge_deoccluded = np.zeros_like(depth)
    edge_inner_pts = edge2D_2_edge3D(edge_inner)
    edge_outter_pts = edge2D_2_edge3D(edge_outter)
    tree_outter = o3d.geometry.KDTreeFlann(edge_outter_pts)
    for point in edge_inner_pts.points:
        [k, idx, _] = tree_outter.search_knn_vector_3d(point, 50)
        point_outter_total = 0
        depth_outter = 0
        for i in range(k):
            point_outter = edge_outter_pts.points[idx[i]]
            point_outter_total += edge_outter_pts.points[idx[i]]
            depth_outter += depth[int(point_outter[1]), int(point_outter[0])]
        depth_outter_mean = depth_outter / k
        depth_inner = depth[int(point[1]), int(point[0])]
        edge_deoccluded[int(point[1]), int(point[0])] = depth[int(point[1]), int(point[0])] - depth_outter_mean

    # thred = 0.005
    # edge_occluded[edge_occluded == 0] += thred
    # edge_occluded = edge_occluded < thred

    edge_deoccluded = edge_deoccluded < 0

    edge_deoccluded_pts = edge2D_2_edge3D(edge_deoccluded)
    tree_deoccluded = o3d.geometry.KDTreeFlann(edge_deoccluded_pts)

    try:

        for point in edge_deoccluded_pts.points:
            [k, idx, _] = tree_deoccluded.search_knn_vector_3d(point, 1)
            point_deoccluded = edge_deoccluded_pts.points[idx[0]]
            if np.linalg.norm(point - point_deoccluded) > 1:
                edge_deoccluded[int(point[1]), int(point[0])] = 0
    except Exception as e:
        print(e)
        print("error")
        print("point:", point)
        print("idx:", idx)
        pass
    return edge_deoccluded

'''
def find_edge_and_deocc_edge(dataset,mask_type,set_id=7, scene_id=4, img_id=240, obj_id=36, deocc=True):


    if dataset == "clearpose":
        rgb_input_path = f"/mnt/disk0/dataset/clearpose/downsample/set{set_id}/scene{scene_id}/{img_id:06d}-color.png"
        mask_input_path = f"/mnt/disk0/dataset/clearpose/downsample/set{set_id}/scene{scene_id}/{img_id:06d}_{obj_id:06d}_sam2_noise.png"
    elif dataset == "transpose":
        pose_gt_path = f"/mnt/disk0/dataset/TRansPose/downsample/seq_test_{scene_id:02d}/sequences/seq_test_{scene_id:02d}/cam_R/pose/{img_id:06d}.json"
        import json
        with open(pose_gt_path, "r") as f:
            pose_gt_json = json.load(f)
        for pose in pose_gt_json.keys():
            if obj_id == int(pose_gt_json[pose]["obj_id"]):
                obj_id_transpose = int(pose)
                break
        if mask_type == "sam2":
            mask_input_path = f"/mnt/disk0/dataset/TRansPose/downsample/seq_test_{scene_id:02d}/sequences/seq_test_{scene_id:02d}/cam_R/mask_sam2/{img_id:06d}_{obj_id_transpose:06d}.png"
        elif mask_type == "sam2_noise":
            mask_input_path = f"/mnt/disk0/dataset/TRansPose/downsample/seq_test_{scene_id:02d}/sequences/seq_test_{scene_id:02d}/cam_R/mask_sam2_noise/{img_id:06d}_{obj_id_transpose:06d}.png"
        elif mask_type == "gt":
            mask_input_path = f"/mnt/disk0/dataset/TRansPose/downsample/seq_test_{scene_id:02d}/sequences/seq_test_{scene_id:02d}/cam_R/mask/{img_id:06d}_{obj_id_transpose:06d}.png"


        rgb_input_path = f"/mnt/disk0/dataset/TRansPose/downsample/seq_test_{scene_id:02d}/sequences/seq_test_{scene_id:02d}/cam_R/rgb/{img_id:06d}.png"
    mask_input = cv2.imread(mask_input_path, cv2.IMREAD_UNCHANGED)
    rgb_input = cv2.imread(rgb_input_path, cv2.IMREAD_UNCHANGED)
    mask_input = torch.tensor(mask_input).unsqueeze(0).unsqueeze(0).float()
    MaxPool2d_ = MaxPool2d(kernel_size=3, stride=1, padding=1)
    MaxPool2d_outter_ = MaxPool2d(kernel_size=13, stride=1, padding=6)
    mask_input_erode = -MaxPool2d_(-mask_input)
    mask_input_dilate = MaxPool2d_outter_(mask_input)
    mask_input_erode = mask_input_erode.squeeze(0).numpy()
    mask_input_dilate = mask_input_dilate.squeeze(0).numpy()
    mask_input = mask_input.squeeze(0).numpy()

    edge_inner = mask_input - mask_input_erode
    edge_outter = mask_input_dilate - mask_input

    edge_inner = edge_inner.astype(np.uint8)[0]
    edge_outter = edge_outter.astype(np.uint8)[0]
    # cv2.imshow("mask_input", mask_input)
    # cv2.imshow("mask_input_erode", mask_input_erode)
    # cv2.imshow("mask_input_dilate", mask_input_dilate)
    # plt.subplot(2, 2, 1)
    # plt.imshow(edge_inner, cmap='gray')
    # plt.subplot(2, 2, 2)
    # plt.imshow(edge_outter, cmap='gray')

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }
    # dataset = 'hypersim'  # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 20  # 20 for indoor model, 80 for outdoor model
    encoder = 'vitl'  # or 'vits', 'vitb'
    depth_anything_metric = DepthAnythingV2_metric(**{**model_configs[encoder]}, max_depth=max_depth)
    depth_anything_metric.load_state_dict(
        torch.load(f'../depth_anything_v2/depth_anything_v2_metric_hypersim_vitl.pth',
                   map_location='cpu'))
    depth_anything_metric = depth_anything_metric.to(0).eval()

    relative_depth_rgb = depth_anything_metric.infer_image(rgb_input, 518)
    # clip the relative depth rgb to 0 to 2
    relative_depth_rgb = np.clip(relative_depth_rgb, 0, 2)
    # plt.subplot(2, 2, 3)
    # plt.imshow(relative_depth_rgb, cmap='gray')

    edge_deoccluded = find_deoccluded_edge(edge_inner, edge_outter, relative_depth_rgb)
    edge_deoccluded = edge_deoccluded.astype(np.uint8) * 255
    # plt.subplot(2, 2, 4)
    # plt.imshow(edge_deoccluded, cmap='gray')
    # # make subplot conpact
    # plt.tight_layout()
    # plt.show()
    if deocc:
        return edge_deoccluded
    else:
        return edge_inner
'''

def min_rotation_error(R_gt, R_pred, symmetries):
    angle_error_sym = []
    for sym in symmetries:
        R_sym = sym
        R_pred_sym = np.dot(R_pred, R_sym)
        angle_error_sym.append(np.arccos((np.trace(np.dot(R_gt.T, R_pred_sym)) - 1) / 2) * 180 / np.pi)

    # 返回最小误差
    return min(angle_error_sym)
def cal_rotation_error(dataset,obj_id, R_pred, R_gt):
    # add_s_all = add_s_all[:, 0]
    # auc, X, Y = Utils.compute_auc_sklearn(add_s_all[add_s_all < 2000], 3000, 1)
    # #
    # plt.xlabel("add_s_thredhold(mm)")
    # plt.ylabel("Precision")
    # plt.plot(X, Y, label=f"ALL(auc:{auc})")
    from scipy.spatial.transform import Rotation
    if dataset == "clearpose":

        if obj_id == 34 or obj_id == 36:
            error = np.arccos((np.trace(np.dot(R_pred.T, R_gt)) - 1) / 2) * 180 / np.pi
        elif obj_id in [2, 3, 4, 5, 8, 9, 10, 11, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 37, 38, 41, 42, 43, 44, 45, 46,
                        47, 48, 49, 50, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]:
            # 示例旋转矩阵

            R1 = R_gt
            R2 = R_pred

            # 对称性矩阵（例如绕z轴对称）
            symmetries = []
            for angle in range(0, 360, 1):
                # print(angle)
                symmetries.append(Rotation.from_euler('z', angle / 10, degrees=True).as_matrix())

            # 计算最小旋转误差
            error = min_rotation_error(R1, R2, symmetries)
        elif obj_id in [18, 26, 27, 28, 29, 30, 51]:
            # 示例旋转矩阵

            R1 = R_gt
            R2 = R_pred

            # 对称性矩阵（例如绕z轴对称）
            symmetries = []
            for angle in range(0, 360, 90):
                # print(angle)
                symmetries.append(Rotation.from_euler('z', angle, degrees=True).as_matrix())

            # 计算最小旋转误差
            error = min_rotation_error(R1, R2, symmetries)

        elif obj_id in [52]:
            # 示例旋转矩阵

            R1 = R_gt
            R2 = R_pred

            # 对称性矩阵（例如绕z轴对称）
            symmetries = []
            for angle in range(0, 360, 40):
                # print(angle)
                symmetries.append(Rotation.from_euler('z', angle, degrees=True).as_matrix())
            error = min_rotation_error(R1, R2, symmetries)
        elif obj_id in [13]:
            # 示例旋转矩阵

            R1 = R_gt
            R2 = R_pred

            # 对称性矩阵（例如绕z轴对称）
            symmetries = []
            for angle in range(0, 360, 1):
                # print(angle)
                symmetries.append(Rotation.from_euler('x', angle / 10, degrees=True).as_matrix())

            # 计算最小旋转误差
            error = min_rotation_error(R1, R2, symmetries)
        else:
            error = np.arccos((np.trace(np.dot(R_pred.T, R_gt)) - 1) / 2) * 180 / np.pi
    elif dataset == "transpose":
        if obj_id in [38, 41, 47, 1, 2, 3, 11, 12, 21, 22, 23, 24.25, 26, 27, 28, 29, 38, 39, 40, 42, 43, 44, 45, 46,
                      47, 59, 61, 62, 63, 64, 65, 66, 67, 68, 71, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
                      211, 212, 213, 214, 73, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 89, 90, 92, 93, 94, 96, 97]:
            # 示例旋转矩阵

            R1 = R_gt
            R2 = R_pred

            # 对称性矩阵（例如绕z轴对称）
            symmetries = []
            symmetries = []
            # print(objid_2_objname("transpose")[obj_id])
            for angle in range(0, 360, 1):
                # print(angle)
                # xyz y axis sysmetry
                angle = angle / 180 * np.pi
                R = np.array([[np.cos(angle), 0, np.sin(angle)],
                              [0, 1, 0],
                              [-np.sin(angle), 0, np.cos(angle)]])
                symmetries.append(R)

            # 计算最小旋转误差
            error = min_rotation_error(R1, R2, symmetries)
        elif obj_id in [37, 88]:
            # 示例旋转矩阵

            R1 = R_gt
            R2 = R_pred

            # 对称性矩阵（例如绕z轴对称）
            symmetries = []
            for angle in range(0, 360, 60):
                # print(angle)
                # xyz y axis sysmetry
                angle = angle / 180 * np.pi
                R = np.array([[np.cos(angle), 0, np.sin(angle)],
                              [0, 1, 0],
                              [-np.sin(angle), 0, np.cos(angle)]])
                symmetries.append(R)

            # 计算最小旋转误差
            error = min_rotation_error(R1, R2, symmetries)


        elif obj_id in [48, 49, 57, 58, 70, 72, 217, 218, 98, 99]:

            # 示例旋转矩阵
            R1 = R_gt
            R2 = R_pred

            # 对称性矩阵（例如绕z轴对称）

            symmetries = []
            for angle in range(0, 360, 180):
                angle = angle / 180 * np.pi
                R = np.array([[np.cos(angle), 0, np.sin(angle)],
                              [0, 1, 0],
                              [-np.sin(angle), 0, np.cos(angle)]])
                symmetries.append(R)
            # 计算最小旋转误差
            error = min_rotation_error(R1, R2, symmetries)
        elif obj_id in [50, 52, 52, 53, 54, 55, 56, 69, 74, 75, 79, 91, 95]:
            # 示例旋转矩阵
            R1 = R_gt
            R2 = R_pred
            # 对称性矩阵（例如绕z轴对称）
            symmetries = []
            for angle in range(0, 360, 90):
                # print(angle)
                # xyz y axis sysmetry
                angle = angle / 180 * np.pi
                R = np.array([[np.cos(angle), 0, np.sin(angle)],
                              [0, 1, 0],
                              [-np.sin(angle), 0, np.cos(angle)]])
                symmetries.append(R)
            # 计算最小旋转误差
            error = min_rotation_error(R1, R2, symmetries)

        elif obj_id in [60]:
            # 示例旋转矩阵
            R1 = R_gt
            R2 = R_pred

            # 对称性矩阵（例如绕z轴对称）
            symmetries = []
            for angle in range(0, 360, 30):
                # print(angle)
                # xyz y axis sysmetry
                angle = angle / 180 * np.pi
                R = np.array([[np.cos(angle), 0, np.sin(angle)],
                              [0, 1, 0],
                              [-np.sin(angle), 0, np.cos(angle)]])
                symmetries.append(R)

            # 计算最小旋转误差
            error = min_rotation_error(R1, R2, symmetries)
        else:
            error = np.arccos((np.trace(np.dot(R_pred.T, R_gt)) - 1) / 2) * 180 / np.pi

    return error
def cal_add(model,R_gt,t_gt,R_pred,t_pred,mesh=False):
    '''
    Calculate add s metrics for a model given pred pose and gt pose.

    Input:
        model: path of mesh
        R_gt: ny.array (3,3)
        R_pred: np.array (3,3)
        t_gt: np.array (3,)
        t_pred: np.array (3,)

    Output:
        add_s_num: float
    '''
    if mesh:
        mesh = o3d.io.read_triangle_mesh(model)
        # uniform sample point from mesh surface
        pts = mesh.sample_points_uniformly(2000)
    else:
        mesh = o3d.io.read_point_cloud(model)
        pts = mesh.uniform_down_sample(10)




    T_gt = np.eye(4)
    T_pred = np.eye(4)
    T_gt[:3,:3] = R_gt
    T_gt[:3,3] = t_gt
    T_pred[:3, :3] = R_pred
    T_pred[:3, 3] = t_pred
    pts_transform_gt = copy.deepcopy(pts).transform(T_gt)
    pts_transform_pred = copy.deepcopy(pts).transform(T_pred)

    pts_diff = np.asarray(pts_transform_gt.points) - np.asarray(pts_transform_pred.points)

    pts_distance = np.linalg.norm(pts_diff,axis = 1)
    add_s_num = np.mean(pts_distance)

    return add_s_num


def cal_add_s(model,R_gt,t_gt,R_pred,t_pred,mesh=False):
    '''
    Calculate add s metrics for a model given pred pose and gt pose.

    Input:
        model: path of mesh
        R_gt: ny.array (3,3)
        R_pred: np.array (3,3)
        t_gt: np.array (3,)
        t_pred: np.array (3,)

    Output:
        add_s_num: float
    '''

    if mesh:
        mesh = o3d.io.read_triangle_mesh(model)
        # uniform sample point from mesh surface
        pts = mesh.sample_points_uniformly(2000)
    else:
        mesh = o3d.io.read_point_cloud(model)
        pts = mesh.uniform_down_sample(10)

    T_gt = np.eye(4)
    T_pred = np.eye(4)
    T_gt[:3,:3] = R_gt
    T_gt[:3,3] = t_gt
    T_pred[:3, :3] = R_pred
    T_pred[:3, 3] = t_pred
    pts_transform_gt = copy.deepcopy(pts).transform(T_gt)
    pts_transform_pred = copy.deepcopy(pts).transform(T_pred)

    # depth = pts2depth(pts_transform_gt, K, w, h)
    # zeros = np.zeros((h,w))
    # depth_mask = np.where(mask[:,:,0] == 0, zeros,depth)
    # depth_mask = np.where(depth_mask < 0, zeros, depth_mask)
    # depth_mask = np.where(depth_mask > 5000, zeros, depth_mask)
    # pts_transform_gt_mask = depth_2_pts(depth_mask,K)
    #
    # depth2 = pts2depth(pts_transform_pred, K, w, h)
    # zeros = np.zeros((h, w))
    # depth_mask2 = np.where(mask[:, :, 0] == 0, zeros, depth2)
    # depth_mask2 = np.where(depth_mask2 < 0, zeros, depth_mask2)
    # depth_mask2 = np.where(depth_mask2 > 5000, zeros, depth_mask2)
    # pts_transform_pred_mask = depth_2_pts(depth_mask2, K)

    tree_pred = o3d.geometry.KDTreeFlann(pts_transform_pred)

    num_point = 0
    dis_sum = 0
    # print(f"number of points: {np.asarray(pts_transform_gt_mask.points).shape[0]}")
    for point in pts_transform_gt.points:

        point_nearest_idx = tree_pred.search_knn_vector_3d(point,1)[1]
        try:
            point_nearest = np.asarray(pts_transform_pred.points)[point_nearest_idx][0,:]
        except Exception as e:
            print(e)
            print(point_nearest_idx)
        dis_cal = np.linalg.norm(point_nearest-point)
        dis_sum += dis_cal
        num_point += 1

    try:
        add_s_num = dis_sum / num_point
    except Exception as e:
        print(e)
        print("num_point:",num_point)
        print("dis_sum:",dis_sum)
        add_s_num

    return add_s_num

def csv_to_dict(csv_file_panda):
    dict = {}
    img_Id_list = csv_file_panda["im_id"].unique()
    img_Id_list = np.sort(img_Id_list)

    for i in img_Id_list:
        dict[i] = {}
    for i in range(len(csv_file_panda)):
        dict_3 = {}
        dict_3["R"] = csv_file_panda["R"][i]
        dict_3["t"] = csv_file_panda["t"][i]
        # print("img_id" + str(imge_id))
        # print(obj_id)
        dict[csv_file_panda["im_id"][i]][csv_file_panda["obj_id"][i]] = dict_3

    # print(dict)
    return dict


from collections import defaultdict
def NestDict():
  return defaultdict(NestDict)
def csv_to_dict_new(csv_file_panda):

    from collections import defaultdict
    nested_dict = lambda: defaultdict(nested_dict)
    dict = nested_dict()



    for i in range(len(csv_file_panda)):
        dict_3 = {}
        dict_3["R"] = csv_file_panda["R"][i]
        dict_3["t"] = csv_file_panda["t"][i]
        # print("img_id" + str(imge_id))
        # print(obj_id)
        dict[csv_file_panda["scene_id"][i]][csv_file_panda["im_id"][i]][csv_file_panda["obj_id"][i]] = dict_3

    # print(dict)
    return dict



def R_transform(R):
    """
    Input: string "R[0] ...... R[8]"
    Output: np.array (3,3)
    """
    R_matric = np.eye(3)
    R_matric[0][0] = R.split(" ")[0]
    R_matric[0][1] = R.split(" ")[1]
    R_matric[0][2] = R.split(" ")[2]
    R_matric[1][0] = R.split(" ")[3]
    R_matric[1][1] = R.split(" ")[4]
    R_matric[1][2] = R.split(" ")[5]
    R_matric[2][0] = R.split(" ")[6]
    R_matric[2][1] = R.split(" ")[7]
    R_matric[2][2] = R.split(" ")[8]

    return R_matric

def t_transform(t):
    """
        Input: string "t[0] ...... t[2]"
        Output: np.array (3,)
    """
    t_vector = np.zeros((3,))

    t_vector[0] = t.split(" ")[0]
    t_vector[1] = t.split(" ")[1]
    t_vector[2] = t.split(" ")[2]

    return t_vector

def colorize_depth_image():
    """
    Visualize the depth.
    """
    save_path = "/mnt/disk0/dataset/BOP/lmo/test/000002/depth_vis"
    for img_id in range(1214):
        img = cv2.imread(f"/mnt/disk0/dataset/BOP/lmo/test/000002/depth/{img_id:06d}.png",cv2.IMREAD_UNCHANGED)
        plt.imsave(
            os.path.join(save_path, f"{img_id:06d}.png"),
            img,
            vmin=0.0,
            vmax=2000,
            cmap="gray",
        )

from Utils import erode_depth
from Utils import bilateral_filter_depth
import json
def visualize_pcd(result_gt, result_pred, image_id, obj_id, type, K):
    '''
    Visualize the pointcloud given img_id and obj_id,
    red is the gt pointcloud
    green is the input pointcloud
    blue is the pred pointcloud

    Input:
        :dict{img_id:{obj_id:{[R],[t]}}}, dict of the gt result containing pose indiced by img_id and obj_id
        :dict{img_id:{obj_id:{[R],[t]}}}, dict of the pred result containing pose indiced by img_id and obj_id
        :int
        :int
        :type of mask, "mask" or "mask_visib"
        :np.array (3,3), camera intrinsic

    Output:
        None


    '''

    pos = 0
    with open("/mnt/disk0/dataset/BOP/lmo/test/000002/scene_gt.json","r") as f:
        scene_gt = json.load(f)
    if scene_gt is not None:
        for k in scene_gt[str(image_id)]:
            if k['obj_id'] == obj_id:
                break
            pos += 1
        mask_file = f"/mnt/disk0/dataset/BOP/lmo/test/000002/{type}/{image_id:06d}_{pos:06d}.png"
        if not os.path.exists(mask_file):
            print(f'{mask_file} not found')
            return None

    mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
    depth_img = cv2.imread(f"/mnt/disk0/dataset/BOP/lmo/test/000002/depth/{image_id:06d}.png",cv2.IMREAD_UNCHANGED).astype(np.float16)
    depth_img[mask == 0] = 0
    pts_input = depth_2_pts(depth_img, K)

    R_gt, t_gt = find_according_imid_objid(result_gt, image_id, obj_id)
    R_gt = R_transform(R_gt)
    t_gt = t_transform(t_gt)
    R_pred, t_pred = find_according_imid_objid(result_pred, image_id, obj_id)
    R_pred = R_transform(R_pred)
    t_pred = t_transform(t_pred)

    R_gt_temp = np.eye(4)
    R_pred_temp = np.eye(4)
    R_gt_temp[:3, :3] = R_gt
    R_gt_temp[:3, 3] = t_gt
    R_pred_temp[:3, :3] = R_pred
    R_pred_temp[:3, 3] = t_pred

    model_path = f"/mnt/disk0/dataset/BOP/lmo/lmo_models/models/obj_{obj_id:06d}.ply"
    mesh = o3d.io.read_triangle_mesh(model_path)
    pts = mesh.sample_points_uniformly(number_of_points=3000)

    pts_transform_gt = copy.deepcopy(pts).transform(R_gt_temp)
    pts_transform_pred = copy.deepcopy(pts).transform(R_pred_temp)

    pts_transform_gt.paint_uniform_color([1, 0, 0])
    pts_transform_pred.paint_uniform_color([0, 0, 1])
    pts_input.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries([pts_transform_gt,pts_transform_pred,pts_input])

def visualize_pcd_activezero(result_pred, img_name, obj_id):

    mask_all = cv2.imread(f"/mnt/disk0/dataset/rand_scenes/{img_name}/label.png",
                          cv2.IMREAD_UNCHANGED)
    obj_list = np.unique(mask_all)
    obj_list = obj_list[obj_list != 18]
    obj_list = obj_list[obj_list != 19]
    obj_meshs = []
    for obj_id in obj_list:
        obj_mesh_path = os.path.abspath(f'/mnt/disk0/dataset/bbox_norm/models/{objid_2_objname()[obj_id]}/visual_mesh.obj')
        obj_mesh = o3d.io.read_triangle_mesh(obj_mesh_path)
        pts = obj_mesh.sample_points_uniformly(number_of_points=2000)
        depth = cv2.imread(f"/mnt/disk0/dataset/rand_scenes/{img_name}/1024_depth_real.png", cv2.IMREAD_UNCHANGED)
        depth = cv2.resize(depth, dsize=(640,360), fx=1, fy=1, interpolation=cv2.INTER_LINEAR_EXACT) / 1000
        depth[depth > 3] = 0

        path = f"/mnt/disk0/dataset/rand_scenes/{img_name}/meta.pkl"
        meta_dict = np.load(path, allow_pickle=True)
        # print(meta_dict)
        K = meta_dict["intrinsic"]
        K[:2,:] /= 3
        scene_pts = depth_2_pts(depth,K)
        print(pts.get_max_bound())
        R_pred, t_pred = find_according_imid_objid(result_pred, img_name_2_img_id()[img_name], obj_id)
        R_pred = R_transform(R_pred)
        t_pred = t_transform(t_pred)

        T_pred_temp = np.eye(4)
        T_pred_temp[:3, :3] = R_pred
        T_pred_temp[:3, 3] = t_pred / 1000
        obj_mesh.transform(T_pred_temp)
        obj_meshs.append(obj_mesh)

    obj_meshs.append(scene_pts)


    o3d.visualization.draw_geometries([obj for obj in obj_meshs])
    # o3d.visualization.draw_geometries([scene_pts])

import trimesh
def visualize_pcd_in_the_scene(result_gt, result_pred, image_id, obj_id, type, K):
    '''
    Visualize the pointcloud given img_id and obj_id under the scene
    red is the gt pointcloud
    green is the input pointcloud
    blue is the pred pointcloud

    Input:
        :dict{img_id:{obj_id:{[R],[t]}}}, dict of the gt result containing pose indiced by img_id and obj_id
        :dict{img_id:{obj_id:{[R],[t]}}}, dict of the pred result containing pose indiced by img_id and obj_id
        :int
        :int
        :type of mask, "mask" or "mask_visib"
        :np.array (3,3), camera intrinsic

    Output:
        None


    '''
    pos = 0
    with open("/mnt/disk0/dataset/BOP/lmo/test/000002/scene_gt.json","r") as f:
        scene_gt = json.load(f)
    if scene_gt is not None:
        for k in scene_gt[str(image_id)]:
            if k['obj_id'] == obj_id:
                break
            pos += 1
        mask_file = f"/mnt/disk0/dataset/BOP/lmo/test/000002/{type}/{image_id:06d}_{pos:06d}.png"
        if not os.path.exists(mask_file):
            print(f'{mask_file} not found')
            return None

    mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
    depth_img = cv2.imread(f"/mnt/disk0/dataset/BOP/lmo/test/000002/depth/{image_id:06d}.png",cv2.IMREAD_UNCHANGED).astype(np.float16)
    depth_img[mask == 0 ] = 0
    pts_input = depth_2_pts(depth_img, K)

    R_gt, t_gt = find_according_imid_objid(result_gt, image_id, obj_id)
    R_gt = R_transform(R_gt)
    t_gt = t_transform(t_gt)

    R_pred, t_pred = find_according_imid_objid(result_pred, image_id, obj_id)
    R_pred = R_transform(R_pred)
    t_pred = t_transform(t_pred)

    model_path = f"/mnt/disk0/dataset/BOP/lmo/lmo_models/models/obj_{obj_id:06d}.ply"
    R_gt_temp = np.eye(4)
    R_pred_temp = np.eye(4)
    R_gt_temp[:3,:3] = R_gt
    R_gt_temp[:3,3] = t_gt
    R_pred_temp[:3, :3] = R_pred
    R_pred_temp[:3, 3] = t_pred

    mesh = o3d.io.read_triangle_mesh(model_path)
    pts = mesh.sample_points_uniformly(number_of_points=3000)
    pts_transform_gt = copy.deepcopy(pts).transform(R_gt_temp)
    pts_transform_pred = copy.deepcopy(pts).transform(R_pred_temp)

    pts_input.paint_uniform_color([0, 1, 0])

    scene_pcd = o3d.io.read_point_cloud(f"/mnt/disk0/pzh/foundationpose/FoundationPose-main/debug/scene_complete_{image_id}.ply")

    print(scene_pcd.get_max_bound())
    print(scene_pcd.get_min_bound())

    pts_temp = np.array(scene_pcd.points) * 1000
    pts_temp2 = o3d.utility.Vector3dVector(pts_temp)
    scene_pcd.points = pts_temp2

    mesh = o3d.io.read_triangle_mesh("/mnt/disk0/pzh/foundationpose/FoundationPose-main/debug/model_tf_0.obj")
    mesh_pts = mesh.sample_points_uniformly(number_of_points=3000)

    mesh_pts_temp = np.array(mesh_pts.points) * 1000
    mesh_pts_temp2 = o3d.utility.Vector3dVector(mesh_pts_temp)
    mesh_pts.points = mesh_pts_temp2



    o3d.visualization.draw_geometries([pts_transform_gt,pts_transform_pred,pts_input,mesh_pts])
    # o3d.visualization.draw_geometries([mesh_gt, mesh_pred, pts_input, mesh_pts])
def generate_add_s_npy(result_gt,result_pred,save_path,K):
    '''
    Calculate and save add_s metrics of pred pose given gt pose and pred pose.

    Input:
        :dict{img_id:{obj_id:{[R],[t]}}}
        :dict{img_id:{obj_id:{[R],[t]}}}
        :string

    Output:
        None
    '''
    add_s_list_all = []
    add_s_list_indi = {}

    add_s_dict_all = {}
    for img_id in range(1214):
        add_s_dict_all[img_id] = {}

    for obj_id in [1, 5, 6, 8, 9, 10, 11, 12]:
        add_s_list_indi[obj_id] = []

    with open("/mnt/disk0/pzh/foundationpose/FoundationPose-main/debug/lmo_test/test_targets_bop19.json") as f:
        json_dict = json.load(f)
    # for image_id in tqdm(range(1214)):
    #     for obj_id in [1, 5, 6, 8, 9, 10, 11, 12]:
    #         try:
    for path in tqdm(json_dict):
        image_id = path["im_id"]
        obj_id = path["obj_id"]
        R_gt, t_gt = find_according_imid_objid(result_gt, image_id, obj_id)
        R_gt = R_transform(R_gt)
        t_gt = t_transform(t_gt)

        R_pred, t_pred = find_according_imid_objid(result_pred, image_id, obj_id)
        R_pred = R_transform(R_pred)
        t_pred = t_transform(t_pred)

        model_path = f"/mnt/disk0/dataset/BOP/lmo/lmo_models/models/obj_{obj_id:06d}.ply"
        mask = cv2.imread(f"/mnt/disk0/dataset/BOP/lmo/test/000002/mask_visib_correct_format/{image_id:06d}_{obj_id:06d}.png")

        add_s = cal_add_s(model_path, R_gt, t_gt, R_pred, t_pred,K,640,480,mask)
        add_s = [add_s,image_id,obj_id]
        add_s_list_all.append(add_s)
        add_s_list_indi[obj_id].append(add_s)

        add_s_dict_all[image_id][obj_id] = add_s

            # except Exception as e:
            #     print(e)
            #     print(f"img:{image_id}-obj:{obj_id} missed.")
    np.save(os.path.join(save_path, "add_s_all_list.npy"), add_s_list_all)
    np.save(os.path.join(save_path, "add_s_indi_dict.npy"), add_s_list_indi)
    np.save(os.path.join(save_path, "add_s_all_dict.npy"), add_s_dict_all)

def draw_roc_and_cal_auc(add_s_all,add_s_indi):

    # add_s_all = add_s_all[:, 0]
    # auc, X, Y = Utils.compute_auc_sklearn(add_s_all[add_s_all < 2000], 3000, 1)
    # #
    # plt.xlabel("add_s_thredhold(mm)")
    # plt.ylabel("Precision")
    # plt.plot(X, Y, label=f"ALL(auc:{auc})")

    plt.xlabel("add_s_thredhold(obj_diameter)")
    plt.ylabel("Precision")

    for obj_id in [1, 5, 6, 8, 9, 10, 11, 12]:
        model_path = f"/mnt/disk0/dataset/BOP/lmo/lmo_models/models/obj_{obj_id:06d}.ply"
        mesh = o3d.io.read_triangle_mesh(model_path)


        max_bound = mesh.get_max_bound()
        min_bound = mesh.get_min_bound()

        obj_diameter = np.linalg.norm(max_bound-min_bound)

        pts = mesh.sample_points_uniformly(number_of_points=2000)

        result = np.array(add_s_indi[obj_id])[:, 0] / (obj_diameter)
        auc, X, Y = Utils.compute_auc_sklearn(result[result < 0.1], 0.1, 0.001)
        plt.plot(X, Y, label=f"Obj {ob_id_to_names[obj_id]}(auc:{auc})")

    plt.legend()
    plt.show()

def max_k(add_s_all, k ,dataset,upper,lower,set_id = None,scene_id = None):
    '''
    Return the indix of max k metrics of the input list with the condition of visibility. Only visibility larger than the percent will be considered.
    input: list[[matrics,img_id,obj_id] * num_of_object], k, dataset, percent
    output: list[k]
    '''
    if dataset == "linemodocc":
        path_mask = "/mnt/disk0/dataset/BOP/lmo/test/000002/mask_correct_format"
        path_vis = "/mnt/disk0/dataset/BOP/lmo/test/000002/mask_visib_correct_format"
        add_s_all_num = np.array([i[0] for i in add_s_all])
        # print(max(add_s_all_num))
        # add_s_all_num[add_s_all_num > 2000] = 0
        img_id = np.array([i[1] for i in add_s_all])
        obj_id = np.array([i[2] for i in add_s_all])
        for i in range(len(add_s_all)):
            img_name = f"{img_id[i]:06d}_{obj_id[i]:06d}.png"
            vis = cv2.imread(os.path.join(path_vis, img_name), cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(os.path.join(path_mask, img_name), cv2.IMREAD_UNCHANGED)
            count_vis = np.count_nonzero(vis)
            count_mask = np.count_nonzero(mask)

            if count_mask == 0:
                percent_temp = 0
            else:
                percent_temp = count_vis / count_mask

            if percent_temp < lower or percent_temp > upper:
                # delete
                add_s_all_num[i] = 0
    elif dataset == "clearpose":

        add_s_all_num = np.array([i[0] for i in add_s_all])
        # print(max(add_s_all_num))
        # add_s_all_num[add_s_all_num > 2000] = 0
        img_id = np.array([i[1] for i in add_s_all])
        obj_id = np.array([i[2] for i in add_s_all])

        for i in range(len(add_s_all)):
            # img_name = f"{img_id[i]:06d}_{obj_id[i]:06d}.png"
            path_mask = f"/mnt/disk0/pzh/pc2-dataset-rendering_PZH-master/render_mask/{set_id}/{scene_id}/{obj_id[i]}/{obj_id[i]}_{img_id[i]}-mask.png"
            path_vis = f"/mnt/disk0/dataset/clearpose/downsample/set{set_id}/scene{scene_id}/{img_id[i]:06d}-label.png"
            vis = cv2.imread(path_vis, cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(path_mask, cv2.IMREAD_UNCHANGED)[...,0]
            count_vis = np.count_nonzero(vis == obj_id[i])
            count_mask = np.count_nonzero(mask)

            if count_mask == 0:
                percent_temp = 0
            else:
                percent_temp = count_vis / count_mask

            if percent_temp < lower or percent_temp > upper:
                # delete
                add_s_all_num[i] = 0
    else:
        add_s_all_num = np.array([i[0] for i in add_s_all])




    max_topk_id = np.argpartition(add_s_all_num,len(add_s_all_num) - k)[-k:]

    return max_topk_id

def min_k(add_s_all, k,dataset,upper, lower,set_id = None,scene_id = None):
    '''
    Return the indix of min k metrics of the input list
    input: list[[matrics,img_id,obj_id] * num_of_object], k
    output: list[k]
    '''
    add_s_all_num = np.array([i[0] for i in add_s_all])

    if dataset == "linemodocc":
        path_mask = "/mnt/disk0/dataset/BOP/lmo/test/000002/mask_correct_format"
        path_vis = "/mnt/disk0/dataset/BOP/lmo/test/000002/mask_visib_correct_format"
        add_s_all_num = np.array([i[0] for i in add_s_all])
        # print(max(add_s_all_num))
        # add_s_all_num[add_s_all_num > 2000] = 0
        img_id = np.array([i[1] for i in add_s_all])
        obj_id = np.array([i[2] for i in add_s_all])
        for i in range(len(add_s_all)):
            img_name = f"{img_id[i]:06d}_{obj_id[i]:06d}.png"
            vis = cv2.imread(os.path.join(path_vis, img_name), cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(os.path.join(path_mask, img_name), cv2.IMREAD_UNCHANGED)[...,0]
            count_vis = np.count_nonzero(vis)
            count_mask = np.count_nonzero(mask)

            if count_mask == 0:
                percent_temp = 0
            else:
                percent_temp = count_vis / count_mask

            if percent_temp < lower or percent_temp > upper:
                # delete
                add_s_all_num[i] = 1000000
    elif dataset == "clearpose":

        add_s_all_num = np.array([i[0] for i in add_s_all])
        # print(max(add_s_all_num))
        # add_s_all_num[add_s_all_num > 2000] = 0
        img_id = np.array([i[1] for i in add_s_all])
        obj_id = np.array([i[2] for i in add_s_all])
        for i in range(len(add_s_all)):
            # img_name = f"{img_id[i]:06d}_{obj_id[i]:06d}.png"
            path_mask = f"/mnt/disk0/pzh/pc2-dataset-rendering_PZH-master/render_mask/{set_id}/{scene_id}/{obj_id[i]}/{obj_id[i]}_{img_id[i]}-mask.png"
            path_vis = f"/mnt/disk0/dataset/clearpose/downsample/set{set_id}/scene{scene_id}/{img_id[i]:06d}-label.png"
            vis = cv2.imread(path_vis, cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(path_mask, cv2.IMREAD_UNCHANGED)[...,0]
            count_vis = np.count_nonzero(vis == obj_id[i])
            count_mask = np.count_nonzero(mask)

            if count_mask == 0:
                percent_temp = 0
            else:
                percent_temp = count_vis / count_mask

            if percent_temp < lower or percent_temp > upper:
                # delete
                add_s_all_num[i] = 1000000
    else:
        add_s_all_num = np.array([i[0] for i in add_s_all])


    # add_s_all_num[add_s_all_num > 2000] = 0
    min_topk_id = np.argpartition(add_s_all_num,k)[:k]

    return min_topk_id

def write_markdown(headers,indi_table,save_path,path_bbox,path_orig,path_render,max_or_min):
    table_markdown = ""
    lines = []
    line_ = "-----------"
    for i in range(len(headers)):
        lines.append(line_)
    for i, header in enumerate(headers):
        table_markdown += f"|{header}"
        if i == len(headers) - 1:
            table_markdown += "|\n"

    for i, line in enumerate(lines):
        table_markdown += f"|{line}"
        if i == len(headers) - 1:
            table_markdown += "|\n"

    for i, line in enumerate(indi_table):
        for j, item in enumerate(line):
            if j == 5:
                table_markdown += f"|<img src=\"{path_bbox[i]}\">"
            elif j == 6:
                table_markdown += f"|<img src=\"{path_orig[i]}\">"
            # elif j == 7:
            #     table_markdown += f"|<img src=\"{path_render[i]}\">"

            elif j == 0 or j == 1 or j == 2 or j == 7:
                table_markdown += f"|{item}"

            else:
                table_markdown += f"|{item:.3f}"
            if j == len(line) - 1:
                table_markdown += "|\n"

    with open(os.path.join(save_path,f"summary_{max_or_min}.md"),"w") as f:
        f.write(table_markdown)






from tabulate import tabulate
def metrics_indi(add_s_indi,k,save_path):
    '''
    Return summary of metrics according to add_s with the format of markdown.
    Generate max_k and min_K summary table.
    Generate max_k and min_k pictures and place them into the generated folders.

    Input:
        :add_s_indi
        :int, to specific max_k and min_K
        :string

    Output:
        None
    '''
    for obj_id in [1, 5, 6, 8, 9, 10, 11, 12]:
        model_path = f"/mnt/disk0/dataset/BOP/lmo/lmo_models/models/obj_{obj_id:06d}.ply"
        mesh = o3d.io.read_triangle_mesh(model_path)
        max_bound = mesh.get_max_bound()
        min_bound = mesh.get_min_bound()
        obj_diameter = np.linalg.norm(max_bound - min_bound)
        add_s_indi_list = add_s_indi[obj_id]
        max_topk_id = max_k(add_s_indi_list,k)
        print(f"Obj {ob_id_to_names[obj_id]} top {k}:")
        max_k_list = np.array([add_s_indi_list[i] for i in max_topk_id])
        #sort by add_s
        max_k_list = max_k_list[np.argsort(max_k_list[:,0],)[::-1]]
        max_k_list_dia = copy.deepcopy(max_k_list)
        max_k_list_dia[:,0] /= obj_diameter
        print(max_k_list)
        # print([add_s_indi_list[i] for i in max_topk_id])
        indi_table = []
        indi_image = []
        indi_image_orig = []
        indi_image_render = []
        headers = ["Obj","Top_i","Img_id","Error(mm)","Error(Obj_diameter)","Bbox","Image","Render"]
        for i in range(k):
            # indi_table.append([f"{ob_id_to_names[obj_id]}",i,max_k_list[i][0],max_k_list_dia[i][0],max_k_list_dia[i][1]])
            indi_table.append(
                [f"{ob_id_to_names[obj_id]}", i, int(max_k_list_dia[i][1]),max_k_list[i][0], max_k_list_dia[i][0], f"../../bbox/pred/obj_{obj_id}/{int(max_k_list_dia[i][1])}.png",f"../../../../../../../dataset/BOP/lmo/test/000002/rgb/{int(max_k_list_dia[i][1]):06d}.png",f"../../../../../../../dataset/BOP/lmo/test/000002/rgb_with_render_obj_visib/{int(max_k_list_dia[i][1]):06d}_{obj_id:06d}.png"])
            indi_image.append(cv2.imread(f"/mnt/disk0/pzh/foundationpose/FoundationPose-main/debug/lmo/bbox/pred/obj_{obj_id}/{int(max_k_list_dia[i][1])}.png"))
            indi_image_orig.append(cv2.imread(f"/mnt/disk0/dataset/BOP/lmo/test/000002/rgb/{int(max_k_list_dia[i][1]):06d}.png"))
            indi_image_render.append(
                cv2.imread(f"/mnt/disk0/dataset/BOP/lmo/test/000002/rgb_with_render_obj_visib/{int(max_k_list_dia[i][1]):06d}_{obj_id:06d}.png"))

        # s = tabulate(indi_table,headers,floatfmt=".3f",tablefmt="fancy_grid")

        save_path_indi = os.path.join(save_path,f"Obj_{ob_id_to_names[obj_id]}")
        if not os.path.exists(save_path_indi):
            os.mkdir(save_path_indi)

        for i in range(k):
            cv2.imwrite(os.path.join(save_path_indi,f"{ob_id_to_names[obj_id]}_max_{i}.png"),indi_image[i])
            cv2.imwrite(os.path.join(save_path_indi, f"{ob_id_to_names[obj_id]}_max_{i}_org.png"), indi_image_orig[i])
            cv2.imwrite(os.path.join(save_path_indi, f"{ob_id_to_names[obj_id]}_max_{i}_render.png"), indi_image_render[i])


        # with open(os.path.join(save_path_indi,f"{ob_id_to_names[obj_id]}_max{k}_summary.txt"),"w") as f:
        #     f.write(s)
        path_bbox = []
        path_orig = []
        path_render = []
        for i in range(k):
            path_bbox.append(f"{ob_id_to_names[obj_id]}_max_{i}.png")
            path_orig.append(f"{ob_id_to_names[obj_id]}_max_{i}_org.png")
            path_render.append(f"{ob_id_to_names[obj_id]}_max_{i}_render.png")
        write_markdown(headers,indi_table,save_path_indi,path_bbox,path_orig,path_render,"max")

        ##Minimize:
        min_topk_id = min_k(add_s_indi_list, k)
        min_k_list = np.array([add_s_indi_list[i] for i in min_topk_id])
        # sort by add_s
        min_k_list = min_k_list[np.argsort(min_k_list[:, 0], )]
        min_k_list_dia = copy.deepcopy(min_k_list)
        min_k_list_dia[:, 0] /= obj_diameter

        indi_table = []
        indi_image = []
        indi_image_orig = []
        indi_image_render = []



        # headers = ["Obj", "Top_i", "Img_id","Error(mm)", "Error(Obj_diameter)", "Img_id"]
        for i in range(k):

            indi_table.append(
                [f"{ob_id_to_names[obj_id]}", -i, int(min_k_list_dia[i][1]),min_k_list[i][0], min_k_list_dia[i][0], f"../../bbox/pred/obj_{obj_id}/{int(min_k_list_dia[i][1])}.png",f"../../../../../../../dataset/BOP/lmo/test/000002/rgb/{int(min_k_list_dia[i][1]):06d}.png",f"../../../../../../../dataset/BOP/lmo/test/000002/rgb_with_render_obj_visib/{int(min_k_list_dia[i][1]):06d}_{obj_id:06d}.png"])
            indi_image.append(cv2.imread(
                f"/mnt/disk0/pzh/foundationpose/FoundationPose-main/debug/lmo/bbox/pred/obj_{obj_id}/{int(min_k_list_dia[i][1])}.png"))
            indi_image_orig.append(
                cv2.imread(f"/mnt/disk0/dataset/BOP/lmo/test/000002/rgb/{int(min_k_list_dia[i][1]):06d}.png"))
            indi_image_render.append(
                cv2.imread(
                    f"/mnt/disk0/dataset/BOP/lmo/test/000002/rgb_with_render_obj_visib/{int(max_k_list_dia[i][1]):06d}_{obj_id:06d}.png"))

        # s = tabulate(indi_table, headers, floatfmt=".3f", tablefmt="fancy_grid")

        # save_path_indi = os.path.join(save_path, f"Obj_{ob_id_to_names[obj_id]}")
        if not os.path.exists(save_path_indi):
            os.mkdir(save_path_indi)


        for i in range(k):
            cv2.imwrite(os.path.join(save_path_indi, f"{ob_id_to_names[obj_id]}_min_{i}.png"), indi_image[i])
            cv2.imwrite(os.path.join(save_path_indi, f"{ob_id_to_names[obj_id]}_min_{i}_org.png"), indi_image_orig[i])
            cv2.imwrite(os.path.join(save_path_indi, f"{ob_id_to_names[obj_id]}_min_{i}_render.png"),
                        indi_image_render[i])

        # with open(os.path.join(save_path_indi, f"{ob_id_to_names[obj_id]}_min{k}_summary.txt"), "w") as f:
        #     f.write(s)
        path_bbox = []
        path_orig = []
        path_render = []
        for i in range(k):
            path_bbox.append(f"{ob_id_to_names[obj_id]}_min_{i}.png")
            path_orig.append(f"{ob_id_to_names[obj_id]}_min_{i}_org.png")
            path_render.append(f"{ob_id_to_names[obj_id]}_min_{i}_render.png")
        # headers = ["Obj", "Top_i", "Error(mm)", "Error(Obj_diameter)", "Bbox", "Image"]
        write_markdown(headers, indi_table, save_path_indi, path_bbox, path_orig,path_render,"min")

def cal_add_s_given_R_t_pred(R_pred,t_pred, img_id,obj_id,dataset,set_id,scene_id):
    '''
    Calculate the add_s metrics of pred pose given gt pose and pred pose.
    '''
    data_gt = f"/mnt/disk0/pzh/foundationpose/FoundationPose-main/debug/clearpose_gt_set{set_id}_scene{scene_id}.npy"
    if os.path.exists(data_gt):
        data_gt = np.load(data_gt,allow_pickle=True).item()
    else:
        data_gt_path = f"/mnt/disk0/pzh/foundationpose/FoundationPose-main/clearpose/debug/clearpose_gt_set{set_id}_scene{scene_id}.csv"
        data_gt = pd.read_csv(data_gt_path)
        data_gt = csv_to_dict(data_gt)
        np.save(f"/mnt/disk0/pzh/foundationpose/FoundationPose-main/debug/clearpose_gt_set{set_id}_scene{scene_id}.npy",data_gt)
        data_gt = np.load(f"/mnt/disk0/pzh/foundationpose/FoundationPose-main/debug/clearpose_gt_set{set_id}_scene{scene_id}.npy",allow_pickle=True).item()



    R_gt = data_gt[img_id][obj_id]["R"]
    t_gt = data_gt[img_id][obj_id]["t"]

    R_gt = R_transform(R_gt)
    t_gt = t_transform(t_gt)



    base_dir = "/mnt/disk0/dataset/clearpose"
    model_name = objid_2_objname("clearpose", base_dir=base_dir)[obj_id]
    model_path = f"/mnt/disk0/dataset/clearpose/model/{model_name}/{model_name}.ply"
    if obj_id == 34 or obj_id == 36:
        add_s = cal_add_s(model_path, R_gt, t_gt, R_pred, t_pred)
    elif obj_id in [2, 3, 4, 5, 8, 9, 10, 11, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 37, 38, 41, 42, 43, 44, 45, 46,
                    47, 48, 49, 50, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]:

        add_s = cal_add_s(model_path, R_gt, t_gt, R_pred, t_pred)

    elif obj_id in [18, 26, 27, 28, 29, 30, 51]:
        # 示例旋转矩阵
        add_s = cal_add_s(model_path, R_gt, t_gt, R_pred, t_pred)

    elif obj_id in [52]:

        add_s = cal_add_s(model_path, R_gt, t_gt, R_pred, t_pred)
    elif obj_id in [13]:
        # 示例旋转矩阵

        add_s = cal_add_s(model_path, R_gt, t_gt, R_pred, t_pred)
    else:
        add_s = cal_add(model_path, R_gt, t_gt, R_pred, t_pred)



    return add_s


if __name__ == "__main__":
    import pandas as pd
    #
    mesh_dir = "/mnt/disk0/dataset/clearpose/model/beaker_1/beaker_1.ply"
    R_gt = np.array([[1,0,0],
                        [0,1,0],
                        [0,0,1]])
    t_gt = np.array([0,0,0])
    R_pred = np.array([[1,0,0],
                        [0,1,0],
                        [0,0,1]])
    t_pred = np.array([1,1,0])








