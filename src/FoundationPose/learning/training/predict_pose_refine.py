# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import copy
import functools
import os,sys,kornia
import time

import cv2
from pytorch3d.transforms.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix
from pytorch3d.renderer import FoVPerspectiveCameras

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../../')
import numpy as np
import torch
from omegaconf import OmegaConf
from learning.models.refine_network import RefineNet
from learning.datasets.h5_dataset import *
from Utils import *
from datareader import *
from torch.nn import MaxPool2d

from pytorch3d.transforms import so3
from ChamferDistancePytorch.chamfer2D.dist_chamfer_2D import chamfer_2DFunction
# from ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DFunction
# from torchvision.transforms import GaussianBlur
from transpose.transpose_obj_given_pose_obj_scene import cal_and_vis, return_gt
from ADD_S import depth_2_pts_torch,xyz_map_2_pts_torch,xyz_map_2_pts_open3d,depth_2_pts_open3d,depth2normal,Sobel_torch,depth2normal_torch

import matplotlib.pyplot as plt

from knn_cuda import KNN

@torch.inference_mode()
def make_crop_data_batch(render_size, ob_in_cams, mesh, rgb, depth, K, crop_ratio, xyz_map, mask= None,normal_map=None, mesh_diameter=None, cfg=None, glctx=None, mesh_tensors=None, dataset:PoseRefinePairH5Dataset=None):
  logging.info("Welcome make_crop_data_batch")
  H,W = depth.shape[:2]
  args = []
  method = 'box_3d'
  ob_in_cams = ob_in_cams.float()
  tf_to_crops = compute_crop_window_tf_batch(pts=mesh.vertices, H=H, W=W, poses=ob_in_cams, K=K, crop_ratio=crop_ratio, out_size=(render_size[1], render_size[0]), method=method, mesh_diameter=mesh_diameter)

  # plt.imshow(copy.deepcopy(depth).cpu())
  # plt.show()

  logging.info("make tf_to_crops done")

  B = len(ob_in_cams)
  poseA = torch.as_tensor(ob_in_cams, dtype=torch.float, device='cuda')

  bs = 512
  rgb_rs = []
  depth_rs = []
  normal_rs = []
  xyz_map_rs = []

  bbox2d_crop = torch.as_tensor(np.array([0, 0, cfg['input_resize'][0]-1, cfg['input_resize'][1]-1]).reshape(2,2), device='cuda', dtype=torch.float)
  bbox2d_ori = transform_pts(bbox2d_crop, tf_to_crops.inverse()).reshape(-1,4)

  # time1 = time.perf_counter()
  for b in range(0,len(poseA),bs):
    extra = {}
    rgb_r, depth_r, normal_r = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=poseA[b:b+bs], context='cuda', get_normal=cfg['use_normal'], glctx=glctx, mesh_tensors=mesh_tensors, output_size=cfg['input_resize'], bbox2d=bbox2d_ori[b:b+bs], use_light=True, extra=extra)

    rgb_rs.append(rgb_r)
    depth_rs.append(depth_r[...,None])
    normal_rs.append(normal_r)
    xyz_map_rs.append(extra['xyz_map'])
  # time2 = time.perf_counter()
  # print(f"render time: {time2-time1}")
  rgb_rs = torch.cat(rgb_rs, dim=0).permute(0,3,1,2) * 255
  depth_rs = torch.cat(depth_rs, dim=0).permute(0,3,1,2)  #(B,1,H,W)
  xyz_map_rs = torch.cat(xyz_map_rs, dim=0).permute(0,3,1,2)  #(B,3,H,W)
  Ks = torch.as_tensor(K, device='cuda', dtype=torch.float).reshape(1,3,3)
  if cfg['use_normal']:
    normal_rs = torch.cat(normal_rs, dim=0).permute(0,3,1,2)  #(B,3,H,W)

  logging.info("render done")

  rgbBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(rgb, dtype=torch.float, device='cuda').permute(2,0,1)[None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
  # if mask is not None:

    # maskBs = kornia.geometry.transform.warp_perspective(
    # torch.as_tensor(mask, dtype=torch.float, device='cuda')[None,None].expand(B, -1, -1, -1), tf_to_crops,
    # dsize=render_size, mode='bilinear', align_corners=False)
  if rgb_rs.shape[-2:]!=cfg['input_resize']:
    rgbAs = kornia.geometry.transform.warp_perspective(rgb_rs, tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
  else:
    rgbAs = rgb_rs
  if xyz_map_rs.shape[-2:]!=cfg['input_resize']:
    xyz_mapAs = kornia.geometry.transform.warp_perspective(xyz_map_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  else:
    xyz_mapAs = xyz_map_rs
  # xyz_mapBs = kornia.geometry.transform.warp_perspective(xyz_map, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)  #(B,3,H,W)
  xyz_mapBs = kornia.geometry.transform.warp_perspective(
    xyz_map,
    tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  # xyz_mapBs_numpy = copy.deepcopy(xyz_mapBs.permute(0,2,3,1)).cpu().numpy()[0][...,2]
  # cv2.imshow("name",xyz_mapBs_numpy)
  # cv2.waitKey(100)

  if cfg['use_normal']:
    normalAs = kornia.geometry.transform.warp_perspective(normal_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
    normalBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(normal_map, dtype=torch.float, device='cuda').permute(2,0,1)[None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  else:
    normalAs = None
    normalBs = None

  logging.info("warp done")

  mesh_diameters = torch.ones((len(rgbAs)), dtype=torch.float, device='cuda')*mesh_diameter
  # if mask is not None:
  #   pose_data = BatchPoseData(rgbAs=rgbAs, rgbBs=rgbBs, depthAs=None, depthBs=None, normalAs=normalAs, normalBs=normalBs, poseA=poseA, poseB=None, xyz_mapAs=xyz_mapAs, xyz_mapBs=xyz_mapBs, tf_to_crops=tf_to_crops, Ks=Ks, mesh_diameters=mesh_diameters,maskBs=maskBs)
  # else:
  pose_data = BatchPoseData(rgbAs=rgbAs, rgbBs=rgbBs, depthAs=None, depthBs=None, normalAs=normalAs,
                              normalBs=normalBs, poseA=poseA, poseB=None, xyz_mapAs=xyz_mapAs, xyz_mapBs=xyz_mapBs,
                              tf_to_crops=tf_to_crops, Ks=Ks, mesh_diameters=mesh_diameters)
  pose_data = dataset.transform_batch(batch=pose_data, H_ori=H, W_ori=W, bound=1)

  logging.info("pose batch data done")

  return pose_data

from ADD_S import objid_2_objname


def pts2depth(pts, K, w, h):
  focalx, focaly, ax, ay = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
  points = np.asarray(pts.points)  # (n,3)
  point_x = np.round(points[:, 0] / points[:, 2] * focalx + ax).astype(int)
  point_y = np.round(points[:, 1] / points[:, 2] * focaly + ay).astype(int)
  point_z = points[:, 2]

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
  zeros = np.zeros_like(img_z)
  img_z = np.where(img_z == np.inf, zeros, img_z)
  # cv2.imshow("name", img_z)
  # cv2.waitKey()
  
  return img_z

  # 保存重新投影生成的深度图dep_rot


class PoseRefinePredictor:
  def __init__(self,obj_id,adapt_projection):
    logging.info("welcome")
    self.amp = True
    self.run_name = "2023-10-28-18-33-37"
    model_name = 'model_best.pth'
    code_dir = os.path.dirname(os.path.realpath(__file__))
    ckpt_dir = f'{code_dir}/../../weights/{self.run_name}/{model_name}'

    self.cfg = OmegaConf.load(f'{code_dir}/../../weights/{self.run_name}/config.yml')
    self.cfg['ckpt_dir'] = ckpt_dir
    self.cfg['enable_amp'] = True

    ########## Defaults, to be backward compatible
    if 'use_normal' not in self.cfg:
      self.cfg['use_normal'] = False
    if 'use_mask' not in self.cfg:
      self.cfg['use_mask'] = False
    if 'use_BN' not in self.cfg:
      self.cfg['use_BN'] = False
    if 'c_in' not in self.cfg:
      self.cfg['c_in'] = 4
    if 'crop_ratio' not in self.cfg or self.cfg['crop_ratio'] is None:
      self.cfg['crop_ratio'] = 1.2
    if 'n_view' not in self.cfg:
      self.cfg['n_view'] = 1
    if 'trans_rep' not in self.cfg:
      self.cfg['trans_rep'] = 'tracknet'
    if 'rot_rep' not in self.cfg:
      self.cfg['rot_rep'] = 'axis_angle'
    if 'zfar' not in self.cfg:
      self.cfg['zfar'] = 3
    if 'normalize_xyz' not in self.cfg:
      self.cfg['normalize_xyz'] = False
    if isinstance(self.cfg['zfar'], str) and 'inf' in self.cfg['zfar'].lower():
      self.cfg['zfar'] = np.inf
    if 'normal_uint8' not in self.cfg:
      self.cfg['normal_uint8'] = False
    logging.info(f"self.cfg: \n {OmegaConf.to_yaml(self.cfg)}")

    self.dataset = PoseRefinePairH5Dataset(cfg=self.cfg, h5_file='', mode='test')
    self.model = RefineNet(cfg=self.cfg, c_in=self.cfg['c_in']).cuda()
    self.adapt_projection = adapt_projection
    logging.info(f"Using pretrained model from {ckpt_dir}")
    ckpt = torch.load(ckpt_dir)
    if 'model' in ckpt:
      ckpt = ckpt['model']
    self.model.load_state_dict(ckpt)

    self.model.cuda().eval()
    logging.info("init done")
    self.last_trans_update = None
    self.last_rot_update = None
    self.obj_id = obj_id
    # print(obj_id)
    # print(objid_2_objname()[obj_id])
    dataset = "glassmolder"
    # if dataset == "activezero":
    #
    #   model_path = f'/mnt/disk0/dataset/bbox_norm/models/{objid_2_objname("activezero")[obj_id]}/visual_mesh.obj'
    #
    #   self.obj_model = o3d.io.read_triangle_mesh(model_path).sample_points_uniformly(3000)
    # elif dataset == "glassmolder":
    #   model_path = f'/mnt/disk0/dataset/transtouch_pc2_2/model_obj/{objid_2_objname("glassmolder")[obj_id]}.obj'
    #
    #   self.obj_model = o3d.io.read_triangle_mesh(model_path).sample_points_uniformly(3000)

  def compute_iou(self, mask_input, mask_pred):
    mask1 = np.uint8(mask_input > 0)
    mask2 = np.uint8(mask_pred > 0)
    # mask2 = mask_pred
    intersection = np.sum(mask1 * mask2, axis=(1, 2))
    union = np.sum(mask1, axis=(1, 2)) + np.sum(mask2, axis=(1, 2)) - intersection
    iou = intersection / union
    return iou

  def compute_chamfer(self, edge_pred, edge_inner):
    mask1 = np.uint8(mask_input > 0)
    mask2 = np.uint8(mask_pred > 0)
    # mask2 = mask_pred
    intersection = np.sum(mask1 * mask2, axis=(1, 2))
    union = np.sum(mask1, axis=(1, 2)) + np.sum(mask2, axis=(1, 2)) - intersection
    iou = intersection / union
    return iou
  def edge2D_2_edge3D(self,edge_inner):
    H, W = edge_inner.shape
    grid_width = np.linspace(0,W-1,W)
    grid_height = np.linspace(0,H-1,H)
    grid_x,grid_y = np.meshgrid(grid_width,grid_height)
    edge_inner_pts = np.stack([grid_x[edge_inner>0],grid_y[edge_inner>0]],axis=1)
    edge_inner_pts = np.concatenate([edge_inner_pts,np.zeros((edge_inner_pts.shape[0],1))],axis=1)
    pts = o3d.geometry.PointCloud()
    pts.points = o3d.utility.Vector3dVector(edge_inner_pts)
    return pts

  def edge2D_2_edge3D_torch(self,edge_inner):
    W, H = edge_inner.shape
    grid_width = torch.linspace(0,W-1,W)
    grid_height = torch.linspace(0,H-1,H)
    grid_x,grid_y = torch.meshgrid(grid_width,grid_height)

    edge_inner_pts = torch.stack([grid_x[edge_inner>0],grid_y[edge_inner>0]],dim=1)
    edge_inner_pts = torch.tensor(edge_inner_pts, dtype=torch.int).cuda()

    return edge_inner_pts
  def find_deoccluded_edge(self,edge_inner,edge_outter,depth):
    edge_deoccluded = np.zeros_like(depth)
    edge_inner_pts = self.edge2D_2_edge3D(edge_inner)
    edge_outter_pts = self.edge2D_2_edge3D(edge_outter)
    tree_outter = o3d.geometry.KDTreeFlann(edge_outter_pts)
    for point in edge_inner_pts.points:
      [k, idx, _] = tree_outter.search_knn_vector_3d(point, 50)
      point_outter_total = 0
      depth_outter = 0
      for i in range(k):
        point_outter = edge_outter_pts.points[idx[i]]
        point_outter_total += edge_outter_pts.points[idx[i]]
        depth_outter += depth[int(point_outter[1]),int(point_outter[0])]
      depth_outter_mean = depth_outter / k
      edge_deoccluded[int(point[1]),int(point[0])] = depth[int(point[1]),int(point[0])] - depth_outter_mean


    # thred = 0.005
    # edge_occluded[edge_occluded == 0] += thred
    # edge_occluded = edge_occluded < thred

    edge_deoccluded = edge_deoccluded < 0

    edge_deoccluded_pts = self.edge2D_2_edge3D(edge_deoccluded)
    tree_deoccluded = o3d.geometry.KDTreeFlann(edge_deoccluded_pts)

    try:

      for point in edge_deoccluded_pts.points:
          [k, idx, _] = tree_deoccluded.search_knn_vector_3d(point, 1)
          point_deoccluded = edge_deoccluded_pts.points[idx[0]]
          if np.linalg.norm(point - point_deoccluded) > 1:
              edge_deoccluded[int(point[1]),int(point[0])] = 0
    except Exception as e:
      print(e)
      print("error")
      print("point:",point)
      print("idx:",idx)
      pass
    return edge_deoccluded

  def find_deoccluded_edge_torch(self,edge_inner,edge_outter,depth):
     # (H,W) -> (W,H)
    edge_deoccluded = torch.zeros_like(depth)
    # edge_inner = torch.tensor(edge_inner, dtype=torch.int).cuda()
    # edge_outter = torch.tensor(edge_outter, dtype=torch.int).cuda()

    edge_inner_pts = self.edge2D_2_edge3D_torch(edge_inner) # (n,2)
    edge_outter_pts = self.edge2D_2_edge3D_torch(edge_outter) # (m,2)


    ref = edge_outter_pts.unsqueeze(0)
    query = edge_inner_pts.unsqueeze(0)
    k = 50
    time2 = time.perf_counter()
    knn = KNN(k=k, transpose_mode=True)
    dist,indx = knn(ref,query)# (1, n_query, 50)
    time3 = time.perf_counter()

    time2 = time.perf_counter()
    indx = indx.view(-1)
    point_outter = edge_outter_pts[indx] # (n_query * 50,2)

    depth_outter_mean =depth[point_outter[:,0],point_outter[:,1]] # (n_query * 50)
    depth_outter_mean = depth_outter_mean.view(-1,k).mean(dim=1)
    depth_outter_mean = depth_outter_mean.T # (n_query)
    time3 = time.perf_counter()

    time2 = time.perf_counter()
    depth_inner = depth[edge_inner_pts[:,0],edge_inner_pts[:,1]]

    edge_deoccluded[edge_inner_pts[:,0],edge_inner_pts[:,1]] = depth_inner - depth_outter_mean
    edge_deoccluded = edge_deoccluded < 0
    time3 = time.perf_counter()


    # edge_deoccluded_pts = self.edge2D_2_edge3D_torch(edge_deoccluded)


    # k = 1
    # ref = edge_deoccluded_pts.unsqueeze(0)
    # query = edge_deoccluded_pts.unsqueeze(0)
    # knn = KNN(k=k, transpose_mode=True)
    # try:
    #
    #   dist, indx = knn(ref, query)  # (1, n, 1)
    #
    #
    #   depend = torch.norm(edge_deoccluded_pts - edge_deoccluded_pts[indx[0,:,0]],dim=1) > 1
    #   edge_deoccluded_pts[depend] = 0
    #   edge_deoccluded =  torch.zeros_like(depth)
    #   edge_deoccluded[edge_deoccluded_pts[:,0],edge_deoccluded_pts[:,1]] = 1
    #
    # except Exception as e:
    #   print(e)
    #   print("error")
    #
    #   pass


    return edge_deoccluded

  def edge_pts_2_edge_map(self,W,H,edge_inner_pts,edge_inner_map):

    edge_inner = np.zeros((H,W))
    for p in range(edge_inner_pts.shape[0]):
        point = edge_inner_pts[p]
        x,y = point
        if edge_inner_map[p] == True:
          edge_inner[int(y),int(x)] = 1
    return edge_inner

  def edge_map_2_edge_pts(self, W, H, edge_input, edge_input_deoccluded,B):
    pts_batch_input = []
    grid_width = torch.linspace(0, W - 1, W).cuda()
    grid_height = torch.linspace(0, H - 1, H).cuda()
    for b in range(B):
      grid_x, grid_y = torch.meshgrid(grid_width, grid_height)
      grid_x = grid_x.permute(1, 0)
      grid_y = grid_y.permute(1, 0)
      edge_input_pts = torch.stack([grid_x[edge_input_deoccluded[0] != 0], grid_y[edge_input_deoccluded[0] != 0]], dim=1)
      # edge_input_pts = torch.cat([edge_input_pts,torch.zeros((edge_input_pts.shape[0],1)).cuda()],dim=1)
      pts_batch_input.append(edge_input_pts)

    edge_inner_pts_2D = torch.stack(pts_batch_input, dim=0).cuda()
    if edge_inner_pts_2D.shape[1] == 0:

      pts_batch_input = []
      for b in range(B):
        grid_x, grid_y = torch.meshgrid(grid_width, grid_height)
        grid_x = grid_x.permute(1, 0)
        grid_y = grid_y.permute(1, 0)
        edge_input_pts_x = grid_x[edge_input[0] != 0]
        edge_input_pts_y = grid_y[edge_input[0] != 0]
        edge_input_pts = torch.stack([edge_input_pts_x, edge_input_pts_y], dim=1)

        # edge_input_pts = torch.cat([edge_input_pts,torch.zeros((edge_input_pts.shape[0],1)).cuda()],dim=1)
        pts_batch_input.append(edge_input_pts)
      # edge_inner_deoccluded = torch.tensor(edge_inner, dtype=torch.int).cuda().unsqueeze(0)
      edge_inner_pts_2D = torch.stack(pts_batch_input, dim=0).cuda()

    return edge_inner_pts_2D

  def edge_map_2_edge_pts_differetial(self,  edge_pred_deoccluded, test_x_map, test_y_map, batch_size):
    edge_pred_deoccluded_temp = edge_pred_deoccluded.bool()
    pts_x = torch.masked_select(test_x_map, edge_pred_deoccluded_temp)
    pts_y = torch.masked_select(test_y_map, edge_pred_deoccluded_temp)
    del edge_pred_deoccluded_temp
    pts_x = pts_x.reshape(batch_size, -1)
    pts_y = pts_y.reshape(batch_size, -1)
    pts = torch.stack((pts_x, pts_y), dim=2)

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

  @torch.enable_grad()
  def fuse_process(self, K, glcam_in_cvcam, rgb_tensor, depth, mask_input_tensor_batch, mesh_tensors,batch_pose, obj_dia = 0,mask_matching =None, glctx=None, xyz_map_tensor=None,depth_anything = None,vis_plot = False,mesh=None,inner = False,depth_anything_score = None,only_edge = None, only_mask = None):
    time_total = time.perf_counter()
    time1 = time.perf_counter()
    #initializa of the pose and preparation for differential rendering
    translation_batch = batch_pose[:,:3,3].clone().detach().requires_grad_(True)
    rotation_batch = batch_pose[:,:3,:3].clone().detach()
    batch_size = batch_pose.shape[0]
    theta_z = torch.zeros(batch_size).cuda().requires_grad_(True)
    def rotation_matrix_z(theta):
      cos_theta = torch.cos(theta)
      sin_theta = torch.sin(theta)
      zeros = torch.zeros_like(theta)
      ones = torch.ones_like(theta)
      return torch.stack([cos_theta, -sin_theta, zeros, sin_theta, cos_theta, zeros, zeros, zeros, ones], dim=1).reshape(batch_size, 3, 3)
    rotation_matrices_z = rotation_matrix_z(theta_z)
    quaternion_batch = matrix_to_quaternion(rotation_batch).requires_grad_(True)
    lr_base = 0.01
    lr_falloff = 1.0
    optimizer_translation = torch.optim.Adam([translation_batch], betas=(0.9, 0.999), lr=lr_base)
    optimizer_rotation = torch.optim.Adam([theta_z], betas=(0.9, 0.999), lr=lr_base)


    time1 = time.perf_counter()
    if depth_anything:
      with torch.no_grad():
        relative_depth_rgb = depth_anything.infer_image(rgb_tensor, 518)
      time2 = time.perf_counter()
      # print(f"depth_anything time: {time2-time1}")
      #0.035s
        # depth_anything.to('cpu')
    else:
      relative_depth_rgb = None

    time2 = time.perf_counter()
    print(f"initial time: {time2 - time1}")


    # print(f"depth_anything time: {time2-time1}")

    if not mask_matching:

      rotation_batch_from_quaternion = torch.matmul(rotation_batch, rotation_matrices_z)
      batch_pose_reconstruct = torch.cat((rotation_batch_from_quaternion, translation_batch.unsqueeze(2)), dim=2)
      batch_pose_reconstruct = torch.cat((batch_pose_reconstruct,
                                          torch.tensor([0, 0, 0, 1]).unsqueeze(0).unsqueeze(0).expand(
                                            batch_pose_reconstruct.shape[0], -1, -1)), dim=1)

      H, W = depth.shape[:2]
      pos = mesh_tensors['pos']
      ob_in_glcams = torch.tensor(glcam_in_cvcam, device='cuda', dtype=torch.float)[None] @ batch_pose_reconstruct
      projection_mat = projection_matrix_from_intrinsics(K, height=H, width=W, znear=0.1, zfar=100)
      projection_mat = torch.as_tensor(projection_mat.reshape(4, 4), device='cuda', dtype=torch.float)
      mtx = projection_mat @ ob_in_glcams
      pos_homo = to_homo_torch(pos)
      pos_trans = (mtx[:, None] @ pos_homo[None, ..., None])[..., 0]
      pos_idx = mesh_tensors['faces']
      rast_out_com, _ = dr.rasterize(glctx, pos_trans, pos_idx, resolution=np.asarray(np.asarray([H, W])))
      # img_clone = rast_out_com.clone().detach().cpu()
      mask_color = torch.tensor([1], dtype=torch.float).cuda().unsqueeze(1)
      mask_zeros = torch.ones_like(pos_idx) - 1
      mask_map, _ = dr.interpolate(mask_color, rast_out_com, mask_zeros)
      mask_pred = dr.antialias(mask_map, rast_out_com, pos_trans, pos_idx)
      mask_pred = torch.flip(mask_pred, dims=[1])

    else:

      time1 = time.perf_counter()
      mask_origin = mask_input_tensor_batch
      MaxPool2d_ = MaxPool2d(kernel_size=3, stride=1, padding=1)
      MaxPool2d_outter_ = MaxPool2d(kernel_size=13, stride=1, padding=6)

      #650 * 480
      # MaxPool2d_ = MaxPool2d(kernel_size=13,stride=1, padding=6)
      mask_input = mask_input_tensor_batch[0].unsqueeze(0)
      mask_input_erode = -MaxPool2d_(-mask_input)
      mask_input_dilate = MaxPool2d_outter_(mask_input)
      mask_input_erode = mask_input_erode.squeeze(0)
      mask_input_dilate = mask_input_dilate.squeeze(0)
      mask_input = mask_input.squeeze(0)

      edge_inner = mask_input - mask_input_erode
      edge_outter = mask_input_dilate - mask_input


      if depth_anything:
        edge_deoccluded = self.find_deoccluded_edge_torch(edge_inner, edge_outter, relative_depth_rgb)
        time3 = time.perf_counter()

      else:
        edge_deoccluded = edge_inner
      # print("time for find deoccluded edge", time3 - time1)
      # remove the surrounding of the image's edge
      edge_deoccluded[0:10,:] = 0
      edge_deoccluded[-10:,:] = 0
      edge_deoccluded[:,0:10] = 0
      edge_deoccluded[:,-10:] = 0
      edge_inner[0:10,:] = 0
      edge_inner[-10:,:] = 0
      edge_inner[:,0:10] = 0
      edge_inner[:,-10:] = 0

      edge_inner_deoccluded = edge_deoccluded.unsqueeze(0)
      edge_inner = edge_inner.unsqueeze(0)

      B = translation_batch.shape[0]
      for it in range(mask_matching):
        time1_total = time.perf_counter()
        # Set learning rate.
        itf = 1.0 * it / mask_matching
        # itf = 1.0
        lr = lr_base * lr_falloff ** itf
        for param_group in optimizer_rotation.param_groups:
          param_group['lr'] = lr
        for param_group in optimizer_translation.param_groups:
          param_group['lr'] = lr

        rotation_batch_from_quaternion = quaternion_to_matrix(quaternion_batch)
        batch_pose_reconstruct = torch.cat((rotation_batch_from_quaternion, translation_batch.unsqueeze(2)), dim=2)
        batch_pose_reconstruct = torch.cat((batch_pose_reconstruct,
                                            torch.tensor([0, 0, 0, 1]).unsqueeze(0).unsqueeze(0).expand(
                                              batch_pose_reconstruct.shape[0], -1, -1)), dim=1)
        if torch.sum(edge_inner_deoccluded) == 0 or torch.sum(edge_inner) == 0:
          print("Leck of points")
          optimizer_translation.zero_grad()
          optimizer_rotation.zero_grad()
          return None, batch_pose_reconstruct, relative_depth_rgb


        H, W = depth.shape[:2]
        pos = mesh_tensors['pos']
        ob_in_glcams = torch.tensor(glcam_in_cvcam, device='cuda', dtype=torch.float)[None] @ batch_pose_reconstruct
        projection_mat = projection_matrix_from_intrinsics(K, height=H, width=W, znear=0.1, zfar=100)
        projection_mat = torch.as_tensor(projection_mat.reshape(4, 4), device='cuda', dtype=torch.float)
        mtx = projection_mat @ ob_in_glcams
        pos_homo = to_homo_torch(pos)
        pos_trans = (mtx[:, None] @ pos_homo[None, ..., None])[..., 0]
        pos_idx = mesh_tensors['faces']
        rast_out_com, _ = dr.rasterize(glctx, pos_trans, pos_idx, resolution=np.asarray(np.asarray([H, W])))
        mask_color = torch.tensor([1],dtype=torch.float).cuda().unsqueeze(1)
        mask_zeros = torch.ones_like(pos_idx) - 1
        mask_map,_ = dr.interpolate(mask_color,rast_out_com,mask_zeros)
        mask_pred  = dr.antialias(mask_map, rast_out_com, pos_trans, pos_idx)
        mask_pred = torch.flip(mask_pred,dims=[1])
        mask_pred = mask_pred.reshape(mask_pred.shape[0],mask_pred.shape[1],mask_pred.shape[2])
        mask_input = mask_input_tensor_batch
        time4 = time.perf_counter()
        # print(f"time for test1 ", time4 - time1_total)

        # print(f"time for mask rendering ", time_total)

        pos_idx = mesh_tensors['faces']
        pos_eyes = (ob_in_glcams[:, None] @ pos_homo[None, ..., None])[..., 0]
        xyz_map, _ = dr.interpolate(pos_eyes, rast_out_com, pos_idx)
        xyz_map = torch.flip(xyz_map, dims=[1])


        z_map = xyz_map.permute(0, 3, 1, 2)[:, 2, :, :]
        x_map = xyz_map.permute(0, 3, 1, 2)[:, 0, :, :]
        y_map = xyz_map.permute(0, 3, 1, 2)[:, 1, :, :]


        u_map = x_map / (z_map + 10 ** -12) * K[0, 0]
        v_map = y_map / (z_map + 10 ** -12) * K[1, 1]

        test_x_map = u_map
        test_y_map = v_map


        test_y_map += K[1, 2]
        test_x_map -= K[0, 2]
        test_x_map *= -1

        test_x_map[u_map == 0] = 0
        test_y_map[v_map == 0] = 0

        #generate the edge of predict mask
        batch_size = B
        edge_pred_deoccluded = mask_pred + MaxPool2d_(-mask_pred)

        num_points = [len(edge_pred_deoccluded[b].reshape(-1)[edge_pred_deoccluded[b].reshape(-1) > 0.1]) for b in
                      range(batch_size)]
        sample_num = int(min(num_points))
        if sample_num >500:
           sample_num = 500
        if sample_num == 0:
          print("Leck of points")
          optimizer_translation.zero_grad()
          optimizer_rotation.zero_grad()
          return None, batch_pose_reconstruct, relative_depth_rgb

        edge_pred_deoccluded = self.sample_uniform_from_edge(sample_num, batch_size, edge_pred_deoccluded, optimizer_translation, optimizer_rotation, batch_pose_reconstruct, relative_depth_rgb, H, W)

        edge_pred_pts_2D = self.edge_map_2_edge_pts_differetial(edge_pred_deoccluded, test_x_map, test_y_map, batch_size)

        B = translation_batch.shape[0]
        H, W = depth.shape[:2]
        edge_inner_deoccluded = self.sample_uniform_from_edge(sample_num, 1, edge_inner_deoccluded, optimizer_translation, optimizer_rotation, batch_pose_reconstruct, relative_depth_rgb, H, W)
        edge_inner_pts_2D = self.edge_map_2_edge_pts(W, H, edge_inner, edge_inner_deoccluded,B)

        time4 = time.perf_counter()
        # print(f"time for test ", time4 - time1_total)

        _, dist2, _, _ = chamfer_2DFunction.apply(edge_pred_pts_2D, edge_inner_pts_2D)
        batch_loss5 = torch.mean(dist2, 1)
        loss5 = torch.mean(dist2)
        mask_cross = mask_pred * mask_input
        loss2 = torch.abs(mask_cross - mask_input)
        batch_loss2 = torch.mean(loss2, (1,2))
        batch_loss = batch_loss2 + batch_loss5
        loss2 = torch.mean(loss2)
        loss = loss2 + loss5

        if only_mask:

          loss = loss2

        if only_edge:

          loss = loss5



        optimizer_translation.zero_grad()
        optimizer_rotation.zero_grad()
        loss.backward()
        optimizer_translation.step()
        optimizer_rotation.step()
        time4 = time.perf_counter()
        # print(f"time for optimize ", time4 - time1_total)

        if vis_plot :

          # time2 = time.perf_counter()
          # print("time for optimize:", time2 - time1)
          # dist2_temp = dist2.clone()
          # chamfer_min = torch.sum(dist2_temp, dim=1)
          # chamfer_min_id = torch.argmin(chamfer_min)
          mask_pred_cpu = mask_pred.detach().clone().cpu().numpy()
          mask_input_cpu = mask_input.detach().clone().cpu().numpy()
          iou_max = self.compute_iou(mask_input_cpu, mask_pred_cpu)
          max_iou_indice = np.argmax(iou_max)


          # plt.imshow(mask_pred_cpu[max_iou_indice])
          # plt.pause(0.1)
          # cv2.imshow("mask_pred", mask_pred_cpu[max_iou_indice] * 255)
          # cv2.imshow("mask_input", mask_input_cpu[max_iou_indice] * 255)

          edge_pred_deoccluded = mask_pred + MaxPool2d_(-mask_pred)
          edge_pred_cpu = edge_pred_deoccluded.detach().clone().cpu().numpy()
          edge_inner_cpu = edge_inner_deoccluded.detach().clone().cpu().numpy()
          edge_inner_cpu = mask_input + MaxPool2d_(-mask_input)
          edge_inner_cpu = edge_inner_cpu.detach().clone().cpu().numpy()
          # cv2.imwrite("/mnt/disk0/pzh/foundationpose/FoundationPose-main/plot/edge_pred.png",edge_pred_cpu[chamfer_min_id]*255)
          # cv2.imwrite("/mnt/disk0/pzh/foundationpose/FoundationPose-main/plot/mask_pred.png",mask_pred_cpu[chamfer_min_id]*255)
          # cv2.imshow("edge_pred", edge_pred_cpu[max_iou_indice] * 255)
          # cv2.imshow("edge_inner", (edge_inner_cpu[max_iou_indice] * 255).astype(np.uint8))
          edge_total = edge_pred_cpu[max_iou_indice] + edge_inner_cpu[max_iou_indice]
          edge_total[edge_total > 1] = 1
          cv2.imshow("edge_total", edge_total * 255)


          cv2.waitKey(1)
          # occ = "occ"
          # scene_id = "scene5 8200 22"
          # cv2.imwrite(f"/mnt/disk0/pzh/foundationpose/FoundationPose-main/plot/deocc/{scene_id}/{occ}/edge_pred{it}.png", edge_pred_cpu[chamfer_min_id] * 255)
          # cv2.imwrite(f"/mnt/disk0/pzh/foundationpose/FoundationPose-main/plot/deocc/{scene_id}/{occ}/edge_inner{it}.png", (edge_inner_cpu[0] * 255).astype(np.uint8))
          # cv2.imwrite(f"/mnt/disk0/pzh/foundationpose/FoundationPose-main/plot/deocc/{scene_id}/{occ}/mask_pred{it}.png", mask_pred_cpu[chamfer_min_id] * 255)
          # cv2.imwrite(f"/mnt/disk0/pzh/foundationpose/FoundationPose-main/plot/deocc/{scene_id}/{occ}/mask_input.png", mask_input_cpu[chamfer_min_id] * 255)

          # print("z" ,batch_pose_reconstruct[max_iou_indice,2,3])
          # print("loss",loss)
        torch.cuda.empty_cache()




    return xyz_map_tensor,batch_pose_reconstruct,relative_depth_rgb, batch_loss.detach().clone().cpu().numpy()

  def refine_process(self, K, glcam_in_cvcam, rgb_tensor, depth, mask_input_tensor_batch, mesh_tensors, batch_pose,
                   obj_dia=0, mask_matching=None, glctx=None, depth_anything=None,mesh = None,img_id = None,scene_id = None):

    # initializa of the pose and preparation for differential rendering
    translation_batch = batch_pose[:, :3, 3].clone().detach().requires_grad_(True)
    rotation_batch = batch_pose[:, :3, :3].clone().detach()
    quaternion_batch = matrix_to_quaternion(rotation_batch).requires_grad_(True)
    lr_base = 0.01
    lr_falloff = 1.0
    optimizer_translation = torch.optim.Adam([translation_batch], betas=(0.9, 0.999), lr=lr_base)
    optimizer_rotation = torch.optim.Adam([quaternion_batch], betas=(0.9, 0.999), lr=lr_base)

    mask_origin = mask_input_tensor_batch
    mask_origin = torch.tensor(mask_origin, dtype=torch.float)
    mask_input = torch.zeros_like(mask_origin)
    mask_input[mask_origin == 1] = 1
    # rgb_tensor_vis = rgb_tensor
    # plt.imshow(rgb_tensor_vis)2
    # plt.show()
    rgb_tensor = torch.tensor(rgb_tensor, dtype=torch.float).cuda()
    Batch_size = translation_batch.shape[0]



    for it in range(mask_matching):

      rotation_batch_from_quaternion = quaternion_to_matrix(quaternion_batch)
      batch_pose_reconstruct = torch.cat((rotation_batch_from_quaternion, translation_batch.unsqueeze(2)), dim=2)
      batch_pose_reconstruct = torch.cat((batch_pose_reconstruct,
                                          torch.tensor([0, 0, 0, 1]).unsqueeze(0).unsqueeze(0).expand(
                                            batch_pose_reconstruct.shape[0], -1, -1)), dim=1)

      itf = 1.0 * it / mask_matching

      lr = lr_base * lr_falloff ** itf
      for param_group in optimizer_rotation.param_groups:
        param_group['lr'] = lr
      for param_group in optimizer_translation.param_groups:
        param_group['lr'] = lr



      H, W = depth.shape[:2]
      pos = mesh_tensors['pos']
      ob_in_glcams = torch.tensor(glcam_in_cvcam, device='cuda', dtype=torch.float)[None] @ batch_pose_reconstruct
      projection_mat = projection_matrix_from_intrinsics(K, height=H, width=W, znear=0.1, zfar=100)
      projection_mat = torch.as_tensor(projection_mat.reshape(4, 4), device='cuda', dtype=torch.float)
      mtx = projection_mat @ ob_in_glcams
      pos_homo = to_homo_torch(pos)
      pos_trans = (mtx[:, None] @ pos_homo[None, ..., None])[..., 0]
      pos_idx = mesh_tensors['faces']
      rast_out_com, _ = dr.rasterize(glctx, pos_trans, pos_idx, resolution=np.asarray(np.asarray([H, W])))
      mask_color = torch.tensor([1], dtype=torch.float).cuda().unsqueeze(1)
      mask_zeros = torch.ones_like(pos_idx) - 1
      mask_map, _ = dr.interpolate(mask_color, rast_out_com, mask_zeros)
      mask_pred = dr.antialias(mask_map, rast_out_com, pos_trans, pos_idx)
      mask_pred = torch.flip(mask_pred, dims=[1])
      mask_rs = mask_pred.permute(0, 3, 1, 2)

      render_size = (160,160)
      crop_ratio = 1.2
      method = 'box_3d'
      tf_to_crops = compute_crop_window_tf_batch(pts=mesh.vertices, H=H, W=W, poses=batch_pose_reconstruct, K=K,
                                                 crop_ratio=crop_ratio, out_size=(render_size[1], render_size[0]),
                                                 method=method, mesh_diameter=obj_dia)

      rgbBs = kornia.geometry.transform.warp_perspective(
        rgb_tensor.permute(2, 0, 1)[None].expand(Batch_size, -1, -1, -1),
        tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
      maskBs = kornia.geometry.transform.warp_perspective(
        torch.as_tensor(mask_input[None][None].expand(Batch_size, -1, -1, -1), dtype=torch.float, device='cuda'), tf_to_crops,
        dsize=render_size, mode='nearest', align_corners=False)
      # rgbB_vis = rgbBs.detach().clone().cpu().numpy()
      # rgbB_vis = rgbB_vis[0].transpose(1, 2, 0).astype(np.uint8)
      # plt.imshow(rgbB_vis)
      # plt.show()
      pos = mesh_tensors['pos']
      vnormals = mesh_tensors['vnormals']
      pos_idx = mesh_tensors['faces']
      has_tex = 'tex' in mesh_tensors
      use_light = True
      light_dir = np.array([0, 0, 1])
      light_pos = np.array([0, 0, 0])
      w_ambient = 0.5
      w_diffuse = 0.5
      pts_cam = transform_pts(pos, batch_pose_reconstruct)
      get_normal = True
      light_color = None
      # if has_tex:
      #   texc, _ = dr.interpolate(mesh_tensors['uv'], rast_out_com, mesh_tensors['uv_idx'])
      #   color = dr.texture(mesh_tensors['tex'], texc, filter_mode='linear')linear
      # else:
      color, _ = dr.interpolate(mesh_tensors['vertex_color'], rast_out_com, pos_idx)

      color_vis = color.detach().clone().cpu().numpy()
      color_vis = color_vis[0]
      mask_vis = mask_pred.detach().clone().cpu().numpy()
      mask_vis = mask_vis[0]
      mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2RGB)
      fuse = cv2.addWeighted(color_vis, 0.5, mask_vis, 0.5, 0)
      # plt.imshow(fuse)
      # plt.show()


      if use_light:
        get_normal = True
      if get_normal:
        vnormals_cam = transform_dirs(vnormals, batch_pose_reconstruct)
        normal_map, _ = dr.interpolate(vnormals_cam, rast_out_com, pos_idx)
        normal_map = F.normalize(normal_map, dim=-1)
        normal_map = torch.flip(normal_map, dims=[1])
      else:
        normal_map = None

      if use_light:
        if light_dir is not None:
          light_dir_neg = -torch.as_tensor(light_dir, dtype=torch.float, device='cuda')
        else:
          light_dir_neg = torch.as_tensor(light_pos, dtype=torch.float, device='cuda').reshape(1, 1, 3) - pts_cam
        diffuse_intensity = \
        (F.normalize(vnormals_cam, dim=-1) * F.normalize(light_dir_neg, dim=-1)).sum(dim=-1).clip(0, 1)[..., None]
        diffuse_intensity_map, _ = dr.interpolate(diffuse_intensity, rast_out_com, pos_idx)  # (N_pose, H, W, 1)
        if light_color is None:
          light_color = color
        else:
          light_color = torch.as_tensor(light_color, device='cuda', dtype=torch.float)
        color = color * w_ambient + diffuse_intensity_map * light_color * w_diffuse

      color = color.clip(0, 1)
      color = color * torch.clamp(rast_out_com[..., -1:], 0, 1)  # Mask out background using alpha
      color = torch.flip(color, dims=[1])
      # color_vis = color.detach().clone().cpu().numpy()
      # color_vis = color_vis[0]
      # plt.imshow(color_vis)
      # plt.show()
      color = color.permute(0, 3, 1, 2)
      rgbAs = kornia.geometry.transform.warp_perspective(color, tf_to_crops, dsize=render_size, mode='bilinear',
                                                           align_corners=False)
      maskAs = kornia.geometry.transform.warp_perspective(mask_rs, tf_to_crops, dsize=render_size, mode='nearest',
                                                           align_corners=False)



      rgbAs_cpu = rgbAs.detach().clone().cpu().numpy()
      rgbBs_cpu = rgbBs.detach().clone().cpu().numpy()

      # rgbA_vis = rgbAs_cpu[0].transpose(1, 2, 0)
      # rgbB_vis = rgbBs_cpu[0].transpose(1, 2, 0).astype(np.uint8)
      # maskA_vis = maskAs.detach().clone().cpu().numpy()
      # maskB_vis = maskBs.detach().clone().cpu().numpy()
      # plt.imshow(rgbA_vis)
      # plt.imshow(maskA_vis[0][0])
      # plt.show()
      #
      # plt.imshow(rgbB_vis / 255.0)
      # plt.imshow(maskB_vis[0][0])
      # plt.show()




      if depth_anything:
        time1 = time.perf_counter()
        # depth_anything.train()
        A = rgbAs * maskAs
        B = rgbBs / 255.0

        A = A * maskBs
        B = B * maskBs
        rgbb_vis = B.detach().clone().cpu().numpy()
        maskb_vis = maskBs.detach().clone().cpu().numpy()
        # plt.imshow(rgbb_vis[0].transpose(1, 2, 0))
        # plt.show()
        # plt.imshow(maskb_vis[0].transpose(1, 2, 0))
        # plt.show()
        #
        outputAs = []
        with torch.cuda.amp.autocast(enabled=self.amp):
          # for i in range(B.shape[0]):
          #   outputA = depth_anything.infer_image(A[i].permute(1,2,0).detach().cpu().numpy(),
          #                                              input_size=560)
          #   outputA = torch.tensor(outputA, dtype=torch.float).cuda()
          #   outputAs.append(outputA)
          # outputB = depth_anything.infer_image(B[0].permute(1,2,0).detach().cpu().numpy(),
          #                                            input_size=560)
          # outputB = torch.tensor(outputB, dtype=torch.float).cuda()
          # outputA = torch.stack(outputAs, dim=0)

          for i in range(B.shape[0]):
            outputA = depth_anything.infer_image_torch(A[i].permute(1, 2, 0),
                                                       input_size=1120)
            outputAs.append(outputA)
          outputB = depth_anything.infer_image_torch(B[0].permute(1,2,0),
                                                     input_size=1120)
          outputB = outputB.unsqueeze(0)
          outputA = torch.stack(outputAs, dim=0)
          outputB = outputB.unsqueeze(1)
          outputA = outputA.unsqueeze(1)
          outputB = outputB.repeat(outputA.shape[0], 1, 1, 1)
          A = torch.nn.functional.interpolate(outputA, size=(160, 160), mode='bilinear', align_corners=False)
          B = torch.nn.functional.interpolate(outputB, size=(160, 160), mode='bilinear', align_corners=False)

          maskB_batch = maskBs
          maskA_batch = maskBs
          A = A * maskB_batch
          B = B * maskB_batch

          # A_vis = A.detach().clone().cpu().numpy()
          # B_vis = B.detach().clone().cpu().numpy()
          # plt.imshow(A_vis[0].transpose(1, 2, 0))
          # plt.show()
          # plt.imshow(B_vis[0].transpose(1, 2, 0))
          # plt.show()

          #
          A_non_zero_batch = A.reshape(A.shape[0], -1)
          B_non_zero_batch = B.reshape(B.shape[0], -1)
          A_max, _ = torch.max(A_non_zero_batch, dim=1, keepdim=True)
          B_max, _ = torch.max(B_non_zero_batch, dim=1, keepdim=True)
          A_non_zero_batch[A_non_zero_batch == 0.0] = 100000
          B_non_zero_batch[B_non_zero_batch == 0.0] = 100000
          B_min, _ = torch.min(B_non_zero_batch, dim=1, keepdim=True)
          A_min, _ = torch.min(A_non_zero_batch, dim=1, keepdim=True)

          A_min = A_min.unsqueeze(2).unsqueeze(3)
          A_max = A_max.unsqueeze(2).unsqueeze(3)
          B_min = B_min.unsqueeze(2).unsqueeze(3)
          B_max = B_max.unsqueeze(2).unsqueeze(3)
          A_masklike = torch.ones_like(A)
          A_masklike[A == 0] = 0
          A_masklike = A_masklike * A_min
          A = (A - A_masklike) / (A_max - A_min)

          B_masklike = torch.ones_like(B)
          B_masklike[B == 0] = 0
          B_masklike = B_masklike * B_min
          B = (B - B_masklike) / (B_max - B_min)
          A[A > 2] = 0.0
          B[B > 2] = 0.0

          A_new = 1 - A
          B_new = 1 - B
          A_new[A == 0.0] = 0.0
          B_new[B == 0.0] = 0.0

          forward_thred = 0.1
          A_masklike = torch.ones_like(A)
          A_masklike[A == 0.0] = 0.0
          A_masklike = A_masklike * forward_thred
          A_new += A_masklike
          B_masklike = torch.ones_like(B)
          B_masklike[B == 0.0] = 0.0
          B_masklike = B_masklike * forward_thred
          B_new += B_masklike

          # plt.imshow(A_new[0].detach().clone().cpu().numpy().transpose(1, 2, 0))
          # plt.show()
          # plt.imshow(B_new[0].detach().clone().cpu().numpy().transpose(1, 2, 0))
          # plt.show()


      #
      # A_new = A * maskBs
      # B_new = B * maskBs
      plt.imshow(A_new[0].detach().clone().cpu().numpy().transpose(1, 2, 0))
      plt.pause(0.01)
      plt.imshow(B_new[0].detach().clone().cpu().numpy().transpose(1, 2, 0))
      plt.pause(0.01)

      A_new_1 = A_new[:, 0, :-2, :]
      A_new_2 =  A_new[:, 0, 2:, :]
      A_grd_x = A_new_1 - A_new_2
      A_new_1 = A_new[:, 0, :, :-2]
      A_new_2 = A_new[:, 0, :, 2:]
      A_grd_y = A_new_1 - A_new_2

      B_new_1 = B_new[:, 0, :-2, :]
      B_new_2 = B_new[:, 0, 2:, :]
      B_grd_x = B_new_1 - B_new_2
      B_new_1 = B_new[:, 0, :, :-2]
      B_new_2 = B_new[:, 0, :, 2:]
      B_grd_y = B_new_1 - B_new_2

      # A_grad = torch.sqrt(A_grd_x[:,:,:-2,:] ** 2 + A_grd_y[:,:,:,:-2] ** 2)
      # B_grad = torch.sqrt(B_grd_x[:,:,:-2,:] ** 2 + B_grd_y[:,:,:,:-2] ** 2)

      loss5_x = torch.abs(A_grd_x) - torch.abs(B_grd_x)
      loss5_y = torch.abs(A_grd_y) - torch.abs(B_grd_y)
      loss5 = torch.mean(loss5_x) + torch.mean(loss5_y)
      # loss5_vis = loss5.detach().clone().cpu().numpy()
      # loss5 = loss5[loss5 < 0.3]
      # loss5 = torch.sum(loss5)
      #
      #
      loss =  loss5
      print(loss)
      # loss = 0.6 * loss2 + 0.4 * loss4

      # if loss < 1e-19:
      #   print("loss is too small")

      optimizer_translation.zero_grad()
      optimizer_rotation.zero_grad()
      loss.backward()
      optimizer_translation.step()
      optimizer_rotation.step()

      pose_cpu = batch_pose_reconstruct.detach().clone().cpu().numpy()[0]
      return_gt(pose_cpu,OBJ_ID=self.obj_id,scene_id=scene_id,IMG_ID=img_id)

      vis_plot = True
      # print("time for one iteration:", time4 - time3)
      # if vis_plot:
      #   time2 = time.perf_counter()
      #   print("time for optimize:", time2 - time1)
      #
      #
      #
      #
      #   # plt.imshow(mask_pred_cpu[max_iou_indice])
      #   # plt.pause(0.1)
      #   plt.subplot(1, 3, 1)
      #   plt.imshow()
      #
      #   edge_pred_cpu = edge_pred_deoccluded.detach().clone().cpu().numpy()
      #   edge_inner_cpu = edge_inner_deoccluded.detach().clone().cpu().numpy()
      #   edge_total = edge_pred_cpu + edge_inner_cpu
      #
      #   plt.subplot(1, 3, 2)
      #   plt.imshow(edge_pred_cpu[chamfer_min_id])
      #
      #   plt.subplot(1, 3, 3)
      #   plt.imshow(edge_inner_cpu[0])
      #
      #   # edge_inner_deoccluded_fil_cpu = edge_inner_deoccluded_fil.detach().clone().cpu().numpy()
      #   # mask_hollow_cpu = mask_hollow.detach().clone().cpu().numpy()
      #   # plt.subplot(2, 3, 5)
      #   # plt.imshow(edge_inner_deoccluded_fil_cpu[chamfer_min_id])
      #
      #   # plt.subplot(2, 3, 6)
      #   # plt.imshow(mask_hollow_cpu[0])
      #
      #   edge_inner_pts_2D_cpu = edge_inner_pts_2D.detach().clone().cpu().numpy()
      #   edge_inner_pts_2D_cpu = edge_inner_pts_2D_cpu[chamfer_min_id]
      #
      #   plt.pause(0.01)
      #
      #   print("z", batch_pose_reconstruct[max_iou_indice, 2, 3])
      #   print("loss", loss)
    #   torch.cuda.empty_cache()
    # time2 = time.perf_counter()
    # # print("time for optimize:", time2 - time1)
    # time_final = time.perf_counter()
    #
    # # print("time for whole process:", time_final - time_total)

    return  batch_pose_reconstruct

  def find_max_iou(self, K, glcam_in_cvcam, depth, mask_input_tensor_batch, mesh_tensors, batch_pose_reconstruct, glctx):

    H, W = depth.shape[:2]
    pos = mesh_tensors['pos']
    ob_in_glcams = torch.tensor(glcam_in_cvcam, device='cuda', dtype=torch.float)[None] @ batch_pose_reconstruct
    projection_mat = projection_matrix_from_intrinsics(K, height=H, width=W, znear=0.1, zfar=100)
    projection_mat = torch.as_tensor(projection_mat.reshape(4, 4), device='cuda', dtype=torch.float)
    mtx = projection_mat @ ob_in_glcams
    pos_homo = to_homo_torch(pos)
    pos_trans = (mtx[:, None] @ pos_homo[None, ..., None])[..., 0]
    pos_idx = mesh_tensors['faces']
    rast_out_com, _ = dr.rasterize(glctx, pos_trans, pos_idx, resolution=np.asarray(np.asarray([H, W])))
    # img_clone = rast_out_com.clone().detach().cpu()
    mask_color = torch.tensor([1], dtype=torch.float).cuda().unsqueeze(1)
    mask_zeros = torch.ones_like(pos_idx) - 1
    mask_map, _ = dr.interpolate(mask_color, rast_out_com, mask_zeros)
    mask_pred = dr.antialias(mask_map, rast_out_com, pos_trans, pos_idx)
    mask_pred = torch.flip(mask_pred, dims=[1])
    mask_pred = mask_pred.reshape(mask_pred.shape[0], mask_pred.shape[1], mask_pred.shape[2])
    mask_input = mask_input_tensor_batch

    mask_pred_cpu = mask_pred.detach().clone().cpu().numpy()
    mask_input_cpu = mask_input.detach().clone().cpu().numpy()

    iou_max = self.compute_iou(mask_input_cpu, mask_pred_cpu)
    max_iou_indice = np.argmax(iou_max)

    return max_iou_indice



  @torch.no_grad()
  def predict(self, rgb, depth, K, ob_in_cams, xyz_map=None, mask=None,normal_map=None, get_vis=True, mesh=None, mesh_tensors=None, glctx=None, mesh_diameter=None, iteration=5, diff_render_iteration=20,depth_anything = None,vis_plot = False,obj_dia = None,inner = False,depth_anything_score = None,only_edge= None,only_mask = None):
    '''
    @rgb: np array (H,W,3)
    @ob_in_cams: np array (N,4,4)
    '''
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    logging.info(f'ob_in_cams:{ob_in_cams.shape}')
    tf_to_center = np.eye(4)
    ob_centered_in_cams = ob_in_cams
    mesh_centered = mesh

    logging.info(f'self.cfg.use_normal:{self.cfg.use_normal}')
    if not self.cfg.use_normal:
      normal_map = None

    crop_ratio = self.cfg['crop_ratio']
    logging.info(f"trans_normalizer:{self.cfg['trans_normalizer']}, rot_normalizer:{self.cfg['rot_normalizer']}")
    bs = 1024

    B_in_cams = torch.tensor(ob_centered_in_cams, device='cuda', dtype=torch.float,requires_grad = True)


    if mesh_tensors is None:
      mesh_tensors = make_mesh_tensors(mesh_centered)

    B = len(ob_in_cams)

    rgb_tensor = torch.as_tensor(rgb, device='cuda', dtype=torch.float)
    depth_tensor = torch.as_tensor(depth, device='cuda', dtype=torch.float)
    xyz_map_tensor = torch.as_tensor(xyz_map, device='cuda', dtype=torch.float).permute(2,0,1)[None].expand(B,-1,-1,-1)
    # xyz_map_tensor = torch.as_tensor(xyz_map, device='cuda', dtype=torch.float)
    mask_tensor = torch.as_tensor(mask, device='cuda', dtype=torch.float)
    mask_input_tensor_batch = torch.as_tensor(mask, dtype=torch.float, device='cuda')[None].expand(B, -1, -1)

    trans_normalizer = self.cfg['trans_normalizer']
    if not isinstance(trans_normalizer, float):
      trans_normalizer = torch.as_tensor(list(trans_normalizer), device='cuda', dtype=torch.float).reshape(1,3)

    depth_tensor_max_iou_cpu = depth
    xyz_map_tensor_repeat = xyz_map_tensor[0]
    if self.adapt_projection:
      iteration_diff_rendering = diff_render_iteration
      time1 = time.perf_counter()
      xyz_map_tensor, B_in_cam,relative_depth_rgb, batch_losses = self.fuse_process(K, glcam_in_cvcam, rgb_tensor, depth,
                                                                                   mask_input_tensor_batch,
                                                                                   mesh_tensors, B_in_cams,
                                                                        mask_matching=iteration_diff_rendering,
                                                                                   glctx=glctx,
                                                                                   xyz_map_tensor=xyz_map_tensor,depth_anything= depth_anything,vis_plot = vis_plot,obj_dia = obj_dia,mesh = mesh,inner=inner,depth_anything_score=depth_anything_score,only_edge = only_edge,only_mask = only_mask)

      time2 = time.perf_counter()
      # print("time for fuse:",time2-time1)

      # plt.imshow(xyz_map_tensor_repeat.permute(1,2,0)[...,0].cpu().numpy())
      # plt.show()


    if iteration == 0:
      B_in_cams = []
      B_in_cams.append(B_in_cam)
      B_in_cams = torch.cat(B_in_cams, dim=0).reshape(len(ob_in_cams), 4, 4)
    else:
      relative_depth_rgb = None
      time2 = time.perf_counter()
      for iter_id, _ in enumerate(range(iteration)):
        logging.info("making cropped data")
        pose_data = make_crop_data_batch(self.cfg.input_resize, B_in_cams, mesh_centered, rgb_tensor, depth_tensor, K, crop_ratio=crop_ratio, normal_map=normal_map, xyz_map=xyz_map_tensor, cfg=self.cfg, glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset, mesh_diameter=mesh_diameter,mask = mask_tensor)


        B_in_cams = []
        for b in range(0, pose_data.rgbAs.shape[0], bs):
          A = torch.cat([pose_data.rgbAs[b:b+bs].cuda(), pose_data.xyz_mapAs[b:b+bs].cuda()], dim=1).float()
          B = torch.cat([pose_data.rgbBs[b:b+bs].cuda(), pose_data.xyz_mapBs[b:b+bs].cuda()], dim=1).float()
          logging.info("forward start")
          with torch.cuda.amp.autocast(enabled=self.amp):
            output = self.model(A,B)
          for k in output:
            output[k] = output[k].float()
          logging.info("forward done")
          if self.cfg['trans_rep']=='tracknet':
            if not self.cfg['normalize_xyz']:
              trans_delta = torch.tanh(output["trans"])*trans_normalizer
            else:
              trans_delta = output["trans"]

          elif self.cfg['trans_rep']=='deepim':
            def project_and_transform_to_crop(centers):
              uvs = (pose_data.Ks[b:b+bs]@centers.reshape(-1,3,1)).reshape(-1,3)
              uvs = uvs/uvs[:,2:3]
              uvs = (pose_data.tf_to_crops[b:b+bs]@uvs.reshape(-1,3,1)).reshape(-1,3)
              return uvs[:,:2]

            rot_delta = output["rot"]
            z_pred = output['trans'][:,2]*pose_data.poseA[b:b+bs][...,2,3]
            uvA_crop = project_and_transform_to_crop(pose_data.poseA[b:b+bs][...,:3,3])
            uv_pred_crop = uvA_crop + output['trans'][:,:2]*self.cfg['input_resize'][0]
            uv_pred = transform_pts(uv_pred_crop, pose_data.tf_to_crops[b:b+bs].inverse().cuda())
            center_pred = torch.cat([uv_pred, torch.ones((len(rot_delta),1), dtype=torch.float, device='cuda')], dim=-1)
            center_pred = (pose_data.Ks[b:b+bs].inverse().cuda()@center_pred.reshape(len(rot_delta),3,1)).reshape(len(rot_delta),3) * z_pred.reshape(len(rot_delta),1)
            trans_delta = center_pred-pose_data.poseA[b:b+bs][...,:3,3]


          else:
            trans_delta = output["trans"]

          if self.cfg['rot_rep']=='axis_angle':
            rot_mat_delta = torch.tanh(output["rot"])*self.cfg['rot_normalizer']
            rot_mat_delta = so3_exp_map(rot_mat_delta).permute(0,2,1)
          elif self.cfg['rot_rep']=='6d':
            rot_mat_delta = rotation_6d_to_matrix(output['rot']).permute(0,2,1)
          else:
            raise RuntimeError

          if self.cfg['normalize_xyz']:
            trans_delta *= (mesh_diameter/2)

          B_in_cam = egocentric_delta_pose_to_pose(pose_data.poseA[b:b+bs], trans_delta=trans_delta, rot_mat_delta=rot_mat_delta)
          B_in_cams.append(B_in_cam)

        B_in_cams = torch.cat(B_in_cams, dim=0).reshape(len(ob_in_cams),4,4)

    B_in_cams_out = B_in_cams@torch.tensor(tf_to_center[None], device='cuda', dtype=torch.float)
    torch.cuda.empty_cache()
    time1 = time.perf_counter()
    # print("refine time:",time1 - time2)
    if iteration == 0:
      pass
    else:
      self.last_trans_update = trans_delta
      self.last_rot_update = rot_mat_delta

    # depth_tensor_max_iou_cpu = depth_max_iou.clone().detach().cpu().numpy()




    return B_in_cams_out, None, depth_tensor_max_iou_cpu,relative_depth_rgb, batch_losses


  @torch.no_grad()
  def matching(self, rgb, depth, K, ob_in_cams, xyz_map, mask, normal_map=None, get_vis=True, mesh=None, mesh_tensors=None,
              glctx=None, mesh_diameter=None, iteration=5,iteration_diff_rendering=10,diff_render_iteration_batch=10):
    '''
    @rgb: np array (H,W,3)
    @ob_in_cams: np array (N,4,4)
    '''
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    logging.info(f'ob_in_cams:{ob_in_cams.shape}')
    tf_to_center = np.eye(4)
    mesh_centered = mesh
    ob_centered_in_cams = ob_in_cams
    B_in_cams = torch.tensor(ob_centered_in_cams, device='cuda', dtype=torch.float, requires_grad=True)

    logging.info(f'self.cfg.use_normal:{self.cfg.use_normal}')

    if mesh_tensors is None:
      mesh_tensors = make_mesh_tensors(mesh_centered)

    B = len(ob_in_cams)


    xyz_map_tensor = torch.as_tensor(xyz_map, device='cuda', dtype=torch.float).permute(2, 0, 1)[None].expand(B, -1, -1,
                                                                                                              -1)

    mask_input_tensor_batch = torch.as_tensor(mask, dtype=torch.float, device='cuda')[None].expand(B, -1, -1)



    time2 = time.perf_counter()
    xyz_map_tensor, B_in_cam, max_iou_indice, max_iou_indice = self.matching_process(K, glcam_in_cvcam, rgb, depth,
                                                                                 mask_input_tensor_batch,
                                                                                 mesh_tensors, B_in_cams,
                                                                            mask_matching=iteration_diff_rendering,
                                                                                 glctx=glctx,
                                                                                 xyz_map_tensor=xyz_map_tensor,mask_matching_batch= diff_render_iteration_batch)
    time1 = time.perf_counter()
    # print(f"Time taken for matching: {time1 - time2:.2f} seconds")
    xyz_map_tensor_repeat = xyz_map_tensor[max_iou_indice].detach().clone()
    # depth_tensor_max_iou_cpu = depth_max_iou.clone().detach().cpu().numpy()
    B_in_cams = []


    B_in_cams.append(B_in_cam)

    # B_in_cams = torch.cat(B_in_cams, dim=0).reshape(len(ob_in_cams), 4, 4)
    B_in_cams = torch.cat(B_in_cams, dim=0).reshape(1, 4, 4)
    B_in_cams_out = B_in_cams @ torch.tensor(tf_to_center[None], device='cuda', dtype=torch.float)
    torch.cuda.empty_cache()



    return B_in_cams_out
  @torch.enable_grad()
  def matching_process(self, K, glcam_in_cvcam, rgb, depth, mask_input_tensor_batch, mesh_tensors, batch_pose,
                   mask_matching=None, glctx=None, xyz_map_tensor=None,mask_matching_batch = 10):

    # initializa of the pose and preparation for differential rendering
    translation_batch = batch_pose[:, :3, 3].clone().detach().requires_grad_(True)
    rotation_batch = batch_pose[:, :3, :3].clone().detach()
    quaternion_batch = matrix_to_quaternion(rotation_batch).requires_grad_(True)
    lr_base = 0.01
    lr_falloff = 0.99
    optimizer_translation = torch.optim.Adam([translation_batch], betas=(0.9, 0.999), lr=lr_base)
    optimizer_rotation = torch.optim.Adam([quaternion_batch], betas=(0.9, 0.999), lr=lr_base)

    for it in range(mask_matching):
      time_for2 = time.perf_counter()
      # Set learning rate.
      itf = it
      lr = lr_base * lr_falloff ** itf
      # lr = 0.05
      for param_group in optimizer_rotation.param_groups:
        param_group['lr'] = lr
      for param_group in optimizer_translation.param_groups:
        param_group['lr'] = lr

      batch_iteration_threshold = mask_matching_batch

      # if it == batch_iteration_threshold:
      #   quaternion_batch = quaternion_batch[max_iou_indice].clone().detach().unsqueeze(0).requires_grad_(True)
      #   translation_batch = translation_batch[max_iou_indice].clone().detach().unsqueeze(0).requires_grad_(True)
      #   optimizer_translation = torch.optim.Adam([translation_batch], betas=(0.9, 0.999), lr=lr_base)
      #   optimizer_rotation = torch.optim.Adam([quaternion_batch], betas=(0.9, 0.999), lr=lr_base)
      rotation_batch_from_quaternion = quaternion_to_matrix(quaternion_batch)
      batch_pose_reconstruct = torch.cat((rotation_batch_from_quaternion, translation_batch.unsqueeze(2)), dim=2)
      batch_pose_reconstruct = torch.cat((batch_pose_reconstruct,
                                          torch.tensor([0, 0, 0, 1]).unsqueeze(0).unsqueeze(0).expand(
                                            batch_pose_reconstruct.shape[0], -1, -1)), dim=1)
      time_for1 = time.perf_counter()
      H, W = depth.shape[:2]
      pos = mesh_tensors['pos']
      ob_in_glcams = torch.tensor(glcam_in_cvcam, device='cuda', dtype=torch.float)[None] @ batch_pose_reconstruct
      projection_mat = projection_matrix_from_intrinsics(K, height=H, width=W, znear=0.1, zfar=100)
      projection_mat = torch.as_tensor(projection_mat.reshape(4, 4), device='cuda', dtype=torch.float)
      mtx = projection_mat @ ob_in_glcams
      pos_homo = to_homo_torch(pos)
      pos_trans = (mtx[:, None] @ pos_homo[None, ..., None])[..., 0]
      pos_idx = mesh_tensors['faces']
      rast_out_com, _ = dr.rasterize(glctx, pos_trans, pos_idx, resolution=np.asarray(np.asarray([H, W])))
      # img_clone = rast_out_com.clone().detach().cpu()
      mask_color = torch.tensor([1], dtype=torch.float).cuda().unsqueeze(1)
      mask_zeros = torch.ones_like(pos_idx) - 1
      mask_map, _ = dr.interpolate(mask_color, rast_out_com, mask_zeros)
      mask_pred = dr.antialias(mask_map, rast_out_com, pos_trans, pos_idx)
      mask_pred = torch.flip(mask_pred, dims=[1])
      mask_pred = mask_pred.reshape(mask_pred.shape[0], mask_pred.shape[1], mask_pred.shape[2])
      mask_input = mask_input_tensor_batch

      mask_cross =  mask_pred * mask_input
      if torch.sum(mask_input) > torch.sum(mask_pred):
        loss2 = torch.abs(mask_cross - mask_pred).sum()
      else:
        loss2 = torch.abs(mask_cross - mask_input).sum()

      edge_tolarence = 3
      mask_input_shifted_left = mask_input[:,:,edge_tolarence:,:]
      mask_input_shifted_right = mask_input[:,:,:-edge_tolarence,:]
      mask_input_diff = torch.abs(mask_input_shifted_left - mask_input_shifted_right)

      mask_pred_shifted_left = mask_pred[:, :, edge_tolarence:, :]
      mask_pred_shifted_right = mask_pred[:, :, :-edge_tolarence, :]
      mask_pred_diff = torch.abs(mask_pred_shifted_left - mask_pred_shifted_right)

      loss3 = torch.abs(mask_input_diff - mask_pred_diff).sum()


      loss = loss2 + loss3





      # loss_fn = nn.MSELoss(reduction='none')
      # if it < batch_iteration_threshold:
      #
      #
      #   loss = loss_fn(mask_pred, mask_input)
      #   loss = torch.mean(loss)
      # else:
      #   loss = loss_fn(mask_pred[0],mask_input[0])
      #   loss = torch.mean(loss)

      # print(loss)
      # if loss < 0.02:
      #   new_lr = 0.05 * (0.06 **(it/50))
      #   for param_group in optimizer_rotation.param_groups:
      #       param_group['lr'] = new_lr
      #   for param_group in optimizer_translation.param_groups:
      #       param_group['lr'] = new_lr


      time_zerograd2 = time.perf_counter()
      optimizer_translation.zero_grad()
      optimizer_rotation.zero_grad()
      time_zerograd1 = time.perf_counter()
      # print(f"Time taken for zero grad: {time_zerograd1 - time_zerograd2:.2f} seconds")
      time_bp2 = time.perf_counter()
      loss.backward()
      optimizer_translation.step()
      optimizer_rotation.step()
      time_for1 = time.perf_counter()
      time_for1 = time.perf_counter()
      # print(f"Time taken for for: {time_for1 - time_for2:.2f} seconds")
      # print(f"Time taken for backprop: {time_bp1 - time_bp2:.2f} seconds")
      if it == batch_iteration_threshold - 1:
        mask_pred_cpu = mask_pred.detach().clone().cpu().numpy()
        mask_input_cpu = mask_input.detach().clone().cpu().numpy()


        iou_max = self.compute_iou(mask_input_cpu, mask_pred_cpu)
        max_iou_indice = np.argmax(iou_max)
      # print(f"Time taken for for: {time_for1 - time_for2:.2f} seconds")







    return xyz_map_tensor, batch_pose_reconstruct, max_iou_indice, max_iou_indice