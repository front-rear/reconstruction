import pandas as pd
import os
import copy

import open3d as o3d
import cv2
import numpy as np
from collections import defaultdict
def NestDict():
  return defaultdict(NestDict)




def csv_to_dict_transpose(csv_file_panda):
    dict = NestDict()

    print(len(csv_file_panda))
    for i in range(len(csv_file_panda)):
        dict3 = {}
        dict3["R"] = csv_file_panda["R"][i]
        dict3["t"] = csv_file_panda["t"][i]
        scene_id = csv_file_panda["scene_id"][i]
        obj_id = csv_file_panda["obj_id"][i]
        img_id = csv_file_panda["im_id"][i]
        print("scene_id" + str(scene_id))
        print("obj_id" + str(obj_id))
        print("img_id" + str(img_id))
        print(i)
        dict[int(scene_id)][int(img_id)][int(obj_id)] = dict3
    return dict


def csv_to_dict(csv_file_panda):
    dict = {}

    for i in range(len(csv_file_panda)):
        dict[i] = {}

    print(len(csv_file_panda))
    for i in range(len(csv_file_panda) ):
        dict_3 = {}
        dict_3["R"] = csv_file_panda["R"][i]
        dict_3["t"] = csv_file_panda["t"][i]
        # print("img_id" + str(imge_id))
        # print(obj_id)
        print(csv_file_panda["im_id"][i])
        print(csv_file_panda["obj_id"][i])
        print(i)
        dict[int(csv_file_panda["im_id"][i])][int(csv_file_panda["obj_id"][i])] = dict_3

    # print(dict)
    return dict
def cal_add_s(model,R_gt,t_gt,R_pred,t_pred,K= None,w = None,h= None,mask= None):
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

    mesh = o3d.io.read_triangle_mesh(model)
    pts = mesh.sample_points_uniformly(number_of_points=4000)

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
    dis_list = []
    # print(f"number of points: {np.asarray(pts_transform_gt_mask.points).shape[0]}")
    for point in pts_transform_gt.points:

        point_nearest_idx = tree_pred.search_knn_vector_3d(point,1)[1]
        try:
            point_nearest = np.asarray(pts_transform_pred.points)[point_nearest_idx][0,:]
        except Exception as e:
            print(e)
            print(point_nearest_idx)
        dis_cal = np.linalg.norm(point_nearest-point)
        dis_list.append(dis_cal)
        dis_sum += dis_cal
        num_point += 1

    add_s_num = dis_sum / num_point

    return add_s_num
from sklearn.neighbors import KDTree
def cal_add_s_v2(model,R_gt,t_gt,R_pred,t_pred,K= None,w = None,h= None,mask= None):
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

    mesh = o3d.io.read_triangle_mesh(model)
    pts = mesh.sample_points_uniformly(number_of_points=4000)



    T_gt = np.eye(4)
    T_pred = np.eye(4)
    T_gt[:3,:3] = R_gt
    T_gt[:3,3] = t_gt
    T_pred[:3, :3] = R_pred
    T_pred[:3, 3] = t_pred
    pts_transform_gt = copy.deepcopy(pts).transform(T_gt)
    pts_transform_pred = copy.deepcopy(pts).transform(T_pred)


    tree_pred = o3d.geometry.KDTreeFlann(pts_transform_pred)

    num_point = 0
    dis_sum = 0
    dis_list = []
    # print(f"number of points: {np.asarray(pts_transform_gt_mask.points).shape[0]}")
    for point in pts_transform_gt.points:

        point_nearest_idx = tree_pred.search_knn_vector_3d(point,1)[1]
        try:
            point_nearest = np.asarray(pts_transform_pred.points)[point_nearest_idx][0,:]
        except Exception as e:
            print(e)
            print(point_nearest_idx)
        dis_cal = np.linalg.norm(point_nearest-point)
        dis_list.append(dis_cal)
        dis_sum += dis_cal
        num_point += 1

    add_s_num = dis_sum / num_point

    pts3d = np.asarray(pts.points)

    pts_xformed_gt = R_gt @ pts3d.transpose() + t_gt[:,None]
    pts_xformed_pred = R_pred @ pts3d.transpose() + t_pred[:,None]
    kdt = KDTree(pts_xformed_gt.transpose(), metric='euclidean')
    distance, _ = kdt.query(pts_xformed_pred.transpose(), k=1)

    add_s_num = np.mean(distance)

    return add_s_num


def find_according_imid_objid(result,image_id,obj_id):
    return result[image_id][obj_id]["R"], result[image_id][obj_id]["t"]
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
def depth_2_pts(depth,K,normalize = False,open_3d = True,xyz = False):
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

    pts_new = []
    for i in range(pts.shape[0]):
        # print(pts[i,:])
        if pts[i, 0] != 0 or pts[i, 1] != 0 or pts[i, 2] != 0 :
            pts_new.append(pts[i, :])

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
        pts = copy.deepcopy(pts_o3d)



    return pts



from ADD_S_activezero import objid_2_objname,objname_2_objid,img_id_2_img_name,img_name_2_img_id

def visualize_obj_transpose(obj_index,obj_id,T,img_id,scene_id, gt = False,pred = True,base_dir = None):
    '''
    Visualize gt/pred/input pointcloud/mesh given image name and object id and pred pose for activezero dataset.

    GT pose has been created before and you can load
    from "/mnt/disk0/pzh/foundationpose/FoundationPose-main/debug/activezero_test/activezero_gt.csv", which
    generated from ADD_S_activezero.py.


    obj_id_list = list[]: obj id list
    T                   : pred pose
    image_name          : like "0-300100" or img_id_2_img_name[i]
    gt                  : whether show gt mesh
    pred                : whether show pred pointcloud
    input_pc            : whethere show input pointcloud
    '''

    result_gt_path = "./debug/transpose_gt.npy"
    result_gt = np.load(result_gt_path,allow_pickle=True).item()
    obj_meshs = []

    obj_mesh_path = os.path.abspath(f'{base_dir}/models/{objid_2_objname("transpose",base_dir=base_dir)[obj_id]}/{objid_2_objname("transpose",base_dir=base_dir)[obj_id]}.obj')
    obj_mesh = o3d.io.read_triangle_mesh(obj_mesh_path)


    T_pred_temp = np.array(T)
    obj_mesh.transform(T_pred_temp)
    if pred:
        obj_meshs.append(obj_mesh)

    if gt:

        R_gt, t_gt = result_gt[scene_id][img_id][obj_index][obj_id]["R"], result_gt[scene_id][img_id][obj_index][obj_id]["t"]
        R_gt = R_transform(R_gt)
        t_gt = t_transform(t_gt)
        T_gt_temp = np.eye(4)
        T_gt_temp[:3, :3] = R_gt
        T_gt_temp[:3, 3] = t_gt
        obj_mesh_path = os.path.abspath(
            f'{base_dir}/models/{objid_2_objname("transpose",base_dir=base_dir)[obj_id]}/{objid_2_objname("transpose",base_dir=base_dir)[obj_id]}.obj')
        obj_mesh_gt = o3d.io.read_triangle_mesh(obj_mesh_path)
        obj_mesh_gt.transform(T_gt_temp)





        obj_mesh_gt.paint_uniform_color([1, 0, 0])
        obj_meshs.append(obj_mesh_gt)

        # if scene:
        #     obj_name = objid_2_objname("glassmolder")[obj_id]
        #     scene_pts = o3d.io.read_point_cloud(f"/home/rvsa/disk0/pzh/foundationpose/FoundationPose-main/glassmolder/debug/glassmolder_purematching50_refine5/debug/9-20/flask5/scene_after_refined.ply")
        #     obj_meshs.append(scene_pts)
    # if input_pc:

        # obj_meshs.append(scene_pts)


    # o3d.visualization.draw_plotly([obj for obj in obj_meshs])
    o3d.visualization.draw_geometries([obj for obj in obj_meshs])
def cal_add_indi_clearpose(dataset,method_name,set_id,scene_id,OBJ_ID,IMG_ID,indix = 1,special_name_var = "default"):
    '''
    Calculate and print the add_s metrics given obj_id and img_id and pred pose in unit of Object diameter and mm.

    Input:
    T           : pred pose [4,4]
    obj_id      : obj_id of activezero
    img_id      : img_id of activezero


    Output:
        None

    '''


    result_gt_path = "./debug/clearpose_gt.csv"
    result_gt = pd.read_csv(result_gt_path)
    result_gt = csv_to_dict(result_gt)
    OBJ_NAME = objid_2_objname(dataset)[OBJ_ID]

    R_gt, t_gt =  result_gt[IMG_ID][OBJ_ID]["R"], result_gt[IMG_ID][OBJ_ID]["t"]
    R_gt = R_transform(R_gt)
    t_gt = t_transform(t_gt)
    T_gt = np.eye(4)
    T_gt[:3,:3] = R_gt
    T_gt[:3,3] = t_gt

    T_pred = np.load(f"/mnt/disk0/pzh/foundationpose/FoundationPose-main/clearpose/debug/{method_name}/debug/{set_id}/{scene_id}/{OBJ_NAME}/{IMG_ID}/pose_{indix}_{special_name_var}.npy")
    t_pred = T_pred[:3,3]
    R_pred = T_pred[:3,:3]


    obj_meshs = []
    obj_name = objid_2_objname(dataset)[OBJ_ID]
    model_path = os.path.abspath(
        f'/mnt/disk0/dataset/clearpose/model/{obj_name}/{obj_name}.obj')
    obj_mesh_gt = o3d.io.read_triangle_mesh(model_path)
    obj_mesh_gt.transform(T_gt)
    obj_mesh_gt.paint_uniform_color([1, 0, 0])
    obj_meshs.append(obj_mesh_gt)

    obj_mesh_pred = o3d.io.read_triangle_mesh(model_path)
    obj_mesh_pred.transform(T_pred)
    obj_mesh_pred.paint_uniform_color([0, 1, 0])
    obj_meshs.append(obj_mesh_pred)
    # o3d.visualization.draw_geometries([obj for obj in obj_meshs])
    add_s = cal_add_s(model_path, R_gt, t_gt, R_pred, t_pred)
    mesh = o3d.io.read_triangle_mesh(model_path)
    max_bound = mesh.get_max_bound()
    min_bound = mesh.get_min_bound()
    obj_diameter = np.linalg.norm(max_bound - min_bound)
    add_s_obj_dia = add_s / obj_diameter


    print(f"IMG_ID:{IMG_ID},ADD_S(mm):{add_s * 1000},ADD_S:(obj_diamter){add_s_obj_dia}")


def min_rotation_error(R_gt, R_pred, symmetries):

    angle_error_sym = []
    R_pred_sym_list = []
    for sym in symmetries:
        R_sym = sym
        R_pred_sym = R_pred @ R_sym
        R_pred_sym_list.append(R_pred_sym)
        angle_error_sym.append(np.arccos((np.trace(R_gt @ R_pred_sym.T) - 1) / 2) * 180 / np.pi)
    # for sym in symmetries:
    #     R_sym = sym
    #     R2_sym = np.dot(R_pred, R_sym)
    #     R_pred_sym_list.append(R2_sym)
    #     angle_error_sym.append( np.arccos((np.trace(np.dot(R_gt.T, R2_sym)) - 1) / 2) * 180 / np.pi)


    # 返回最小误差
    return min(angle_error_sym), R_pred_sym_list

def cal_add_indi(T,obj_index,obj_id,img_id,scene_id,base_dir):
    '''
    Calculate and print the add_s metrics given obj_id and img_id and pred pose in unit of Object diameter and mm.

    Input:
    T           : pred pose [4,4]
    obj_id      : obj_id of activezero
    img_id      : img_id of activezero


    Output:
        None

    '''


    result_gt_path = f"./debug/transpose_gt.npy"
    result_gt = np.load(result_gt_path,allow_pickle=True).item()

    R_gt, t_gt = result_gt[scene_id][img_id][obj_index][obj_id]["R"], result_gt[scene_id][img_id][obj_index][obj_id]["t"]
    R_gt = R_transform(R_gt)
    t_gt = t_transform(t_gt)
    T_gt = np.eye(4)
    T_gt[:3,:3] = R_gt
    T_gt[:3,3] = t_gt

    T_pred = np.array(T)
    # T_pred[:3,3] = T_pred[:3,3]
    R_pred = T_pred[:3,:3]
    t_pred = T_pred[:3,3]


    obj_meshs = []
    "/mnt/disk0/dataset/TRansPose/models/obj_C_01_01_G/obj_C_01_01_G.obj"
    model_path = os.path.abspath(
        f'{base_dir}/models/{objid_2_objname("transpose",base_dir=base_dir)[obj_id]}/{objid_2_objname("transpose",base_dir=base_dir)[obj_id]}.obj')
    obj_mesh_gt = o3d.io.read_triangle_mesh(model_path)
    obj_mesh_gt.transform(T_gt)
    obj_mesh_gt.paint_uniform_color([1, 0, 0])
    obj_meshs.append(obj_mesh_gt)

    obj_mesh_pred = o3d.io.read_triangle_mesh(model_path)
    obj_mesh_pred.transform(T_pred)
    obj_mesh_pred.paint_uniform_color([0, 1, 0])
    obj_meshs.append(obj_mesh_pred)
    # o3d.visualization.draw_geometries([obj for obj in obj_meshs])
    # t_gt /= 1000
    # t_pred /= 1000
    add_s = cal_add_s_v2(model_path, R_gt, t_gt, R_pred, t_pred)
    mesh = o3d.io.read_triangle_mesh(model_path)
    max_bound = mesh.get_max_bound()
    min_bound = mesh.get_min_bound()
    obj_diameter = np.linalg.norm(max_bound - min_bound)
    add_s_obj_dia = add_s / obj_diameter

    t_error = np.linalg.norm(t_gt - t_pred)
    R_error = np.arccos(0.5 * (np.trace(R_gt @ R_pred.T) - 1)) * 180 / np.pi
    print(f"ADD_S(mm):{add_s * 1000},ADD_S:(obj_diamter){add_s_obj_dia}")


    symmetries = []
    print(objid_2_objname("transpose",base_dir=base_dir)[obj_id])
    for angle in range(0, 361, 10):
        # print(angle)
        #xyz y axis sysmetry
        angle = angle / 180 * np.pi
        R = np.array([[np.cos(angle), 0, np.sin(angle)],
                      [0, 1, 0],
                      [-np.sin(angle), 0, np.cos(angle)]])
        symmetries.append(R)

    min_error, R_pred_list = min_rotation_error(R_gt, R_pred, symmetries)
    # for sys in symmetries:
    #     obj_mesh_pred = o3d.io.read_triangle_mesh(model_path)
    #     # T_pred = np.eye(4)
    #     # T_pred[:3, :3] = R_pred
    #     # T_pred[:3, 3] = t_pred
    #
    #     T_sym = np.eye(4)
    #     T_sym[:3, :3] = sys
    #
    #     T_pred = np.dot(T_pred, T_sym)
    #
    #     visualize_obj_transpose([obj_id], T_pred, img_id,scene_id, True, True)
    print(f"t_error:{t_error},R_error:{min_error}")
    # print(f"t_error:{t_error},R_error:{R_error}")

import scipy.io as sio
def read_meta(meta_path):
    meta = sio.loadmat(meta_path)
    return meta

from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    PointLights,
    DirectionalLights,
    AmbientLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
from scipy.spatial.transform import Rotation as R


def quaternion_distance(q1, q2):
    # 计算四元数之间的距离
    return np.arccos(2 * np.dot(q1, q2) ** 2 - 1) * 180 / np.pi



import torch
def vis_set_scene(set_id, scene_id,method_name,OBJ_NAME,IMG_ID,vis = True,special_name_var = "default",indix = 1):

    meta_path = '/mnt/disk0/dataset/clearpose/set3/scene1/metadata.mat'
    meta = read_meta(meta_path)

    # print(meta["000000"][0][0])
    camera_intrinsic = meta["000000"][0][0][3][:3,:3].astype(np.float32)
    camera2table = meta["000000"][0][0][5][:3,:4].astype(np.float32)
    camera2table = np.vstack([camera2table, np.array([0,0,0,1])])
    # print(camera2table)
    table2camera = np.linalg.inv(camera2table)
    # print(table2camera)
    OBJ_ID = objname_2_objid("clearpose")[OBJ_NAME]



    obj_list = meta[f"{IMG_ID:06d}"][0][0][0].flatten()
    ros2opencv = np.array([[1., 0., 0.,0],
                           [0., -1., 0., 0.],
                           [0., 0., -1, 0.],
                           [0., 0., 0., 1.]], dtype=np.float32)
    scene = []
    for i, obj in enumerate(obj_list):
        if obj == OBJ_ID:

            obj_id = obj
            obj_pose = meta[f"{IMG_ID:06d}"][0][0][4][:3, :4, i].astype(np.float32)
            obj2camera = obj_pose
            obj2camera = np.vstack([obj2camera, np.array([0, 0, 0, 1])])

            obj2camera =  ros2opencv @ obj2camera
            # print("obj2camera", obj2camera)
            # obj2camera = table2camera @ obj2table
            # print("obj2camera",obj2camera)
            if obj_id == 0:
                continue
            obj_name = objid_2_objname("clearpose")[obj_id]
            obj_mesh_path = os.path.abspath(f'/mnt/disk0/dataset/clearpose/model/{obj_name}/{obj_name}.obj')
            obj_mesh = o3d.io.read_triangle_mesh(obj_mesh_path)
            obj_mesh.transform(obj2camera)
            #color red
            obj_mesh.paint_uniform_color([1, 0, 0])
            scene.append(obj_mesh)
    predict_pose = ros2opencv @ np.load(f"/mnt/disk0/pzh/foundationpose/FoundationPose-main/clearpose/debug/{method_name}/debug/{set_id}/{scene_id}/{OBJ_NAME}/{IMG_ID}/pose_{indix}_{special_name_var}.npy")
    obj_mesh_path = os.path.abspath(f'/mnt/disk0/dataset/clearpose/model/{obj_name}/{obj_name}.obj')
    obj_mesh = o3d.io.read_triangle_mesh(obj_mesh_path)
    obj_mesh.transform(predict_pose)
    obj_mesh.paint_uniform_color([0, 1, 0])
    scene.append(obj_mesh)
    scene_raw = o3d.io.read_point_cloud(f"/mnt/disk0/pzh/foundationpose/FoundationPose-main/clearpose/debug/{method_name}/debug/{set_id}/{scene_id}/{OBJ_NAME}/{IMG_ID}/scene_raw.ply")
    scene_raw.transform(ros2opencv)
    scene.append(scene_raw)
    if vis:
        o3d.visualization.draw_geometries([obj for obj in scene])
    cal_add_indi_clearpose("clearpose", method_name, set_id, scene_id, OBJ_ID, IMG_ID)
    # o3d.visualization.draw_geometries([scene_raw, obj_mesh])
    # depth_anything_path = f"/mnt/disk0/pzh/foundationpose/FoundationPose-main/clearpose/debug/{method_name}/debug/{set_id}/{scene_id}/{OBJ_NAME}/{IMG_ID}/relative_depth_rgb.ply"
    # if os.path.exists(depth_anything_path):
    #     depth_anything = o3d.io.read_point_cloud(depth_anything_path)
    #     o3d.visualization.draw_geometries([depth_anything])
    # 对称性矩阵（例如绕z轴对称）
    symmetries = [R.from_euler('y', angle, degrees=True).as_matrix() for angle in [i for i in range(360)]]
    min_error = min_rotation_error(predict_pose[:3,:3], obj2camera[:3,:3], symmetries)
    print(f"Min rotation error: {min_error:.2f} degrees")

def cal_and_vis(IMG_ID,OBJ_INDEX,OBJ_ID,scene_id ,method_name = "clearpose_purematching30_refine0_zerodepth_testfoundationpose",base_dir = None):
    # for IMG_ID in [5,40]:
    dataset = "transpose"
    special_name_var = "default"
    OBJ_NAME = objid_2_objname("transpose",base_dir=base_dir)[OBJ_ID]

    indix = 1
    whether_vis = True
    "/mnt/disk0/pzh/foundationpose/FoundationPose-main/transpose/debug/transpose_purematching30_refine0_zerodepth_sample_test_deocc_set0_scene17_masksam2_noise_abnormal/debug/0/17/1/obj_T_02_19_P_8/pose_1_default.npy"
    predict_pose = np.load(
        f"./debug/{method_name}/debug/0/{scene_id}/{IMG_ID}/{OBJ_NAME}_{OBJ_INDEX}/pose_{indix}_{special_name_var}.npy")
    visualize_obj_transpose(OBJ_INDEX,OBJ_ID, predict_pose, IMG_ID,scene_id, True, True,base_dir)

    cal_add_indi(T = predict_pose, obj_id = OBJ_ID,img_id= IMG_ID,scene_id= scene_id,base_dir = base_dir,obj_index=OBJ_INDEX)

def return_gt(predict_pose, IMG_ID,OBJ_ID,scene_id ):


    cal_add_indi(T = predict_pose, obj_id = OBJ_ID,img_id= IMG_ID,scene_id= scene_id)

if __name__ == "__main__":

    # dataset = "clearpose"
    # img_id = 262
    # obj_id_list = [objname_2_objid("glassmolder")["flask5"]]
    # Tcameraz_adapt =  [[0.9777929782867432, 0.09821566939353943, 0.185133159160614, 0.1529649794101715], [0.19074273109436035, -0.7830328941345215, -0.5920106768608093, 0.12777772545814514], [0.08682069182395935, 0.6141766905784607, -0.7843782305717468, 0.8548299670219421], [0.0, 0.0, 0.0, 1.0]]
    # # visualize_obj_activezero(obj_id_list, Tcamera,  img_id_2_img_name()[img_id], True, True, True)
    # # cal_add_indi(T = Tcamera, obj_id = obj_id_list[0],img_id=img_id)
    # visualize_obj_glassmolder(obj_id_list, Tcameraz_adapt, img_id_2_img_name()[img_id], True, True, True)
    # cal_add_indi(T = Tcameraz_adapt, obj_id = obj_id_list[0],img_id= img_id)


    OBJ_NAME = "wine_cup_3"
    IMG_ID = 40
    cal_and_vis(IMG_ID, objname_2_objid("clearpose")[OBJ_NAME])
        # method_name = "clearpose_purematching30_refine0_zerodepth_double_test_woocc"
        # OBJ_NAME = "wine_cup_7"
        # # OBJ_NAME = objid_2_objname("clearpose")[16]
        # # IMG_ID = 85
        # vis_set_scene(3, 1, method_name, OBJ_NAME, IMG_ID,vis = whether_vis)
