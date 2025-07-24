# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import functools
import os,sys,kornia
import time

import cv2
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../../../')
from learning.datasets.h5_dataset import *
from learning.models.score_network import *
from learning.datasets.pose_dataset import *
from Utils import *
from datareader import *

from torchvision.transforms import Compose
import matplotlib

class NormalizeImage(object):
  """Normlize image by given mean and std.
  """

  def __init__(self, mean, std):
    self.__mean = mean
    self.__std = std

  def __call__(self, sample):
    sample["image"] = (sample["image"] - self.__mean) / self.__std

    return sample

class PrepareForNet(object):
  """Prepare sample for usage as network input.
  """

  def __init__(self):
    pass

  def __call__(self, sample):
    image = np.transpose(sample["image"], (2, 0, 1))
    sample["image"] = np.ascontiguousarray(image).astype(np.float32)

    if "depth" in sample:
      depth = sample["depth"].astype(np.float32)
      sample["depth"] = np.ascontiguousarray(depth)

    if "mask" in sample:
      sample["mask"] = sample["mask"].astype(np.float32)
      sample["mask"] = np.ascontiguousarray(sample["mask"])

    return sample

class Resize(object):
  """Resize sample to given size (width, height).
  """

  def __init__(
          self,
          width,
          height,
          resize_target=True,
          keep_aspect_ratio=False,
          ensure_multiple_of=1,
          resize_method="lower_bound",
          image_interpolation_method=cv2.INTER_AREA,
  ):
    """Init.

    Args:
        width (int): desired output width
        height (int): desired output height
        resize_target (bool, optional):
            True: Resize the full sample (image, mask, target).
            False: Resize image only.
            Defaults to True.
        keep_aspect_ratio (bool, optional):
            True: Keep the aspect ratio of the input sample.
            Output sample might not have the given width and height, and
            resize behaviour depends on the parameter 'resize_method'.
            Defaults to False.
        ensure_multiple_of (int, optional):
            Output width and height is constrained to be multiple of this parameter.
            Defaults to 1.
        resize_method (str, optional):
            "lower_bound": Output will be at least as large as the given size.
            "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
            "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
            Defaults to "lower_bound".
    """
    self.__width = width
    self.__height = height

    self.__resize_target = resize_target
    self.__keep_aspect_ratio = keep_aspect_ratio
    self.__multiple_of = ensure_multiple_of
    self.__resize_method = resize_method
    self.__image_interpolation_method = image_interpolation_method

  def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
    y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

    if max_val is not None and y > max_val:
      y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

    if y < min_val:
      y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

    return y

  def get_size(self, width, height):
    # determine new height and width
    scale_height = self.__height / height
    scale_width = self.__width / width

    if self.__keep_aspect_ratio:
      if self.__resize_method == "lower_bound":
        # scale such that output size is lower bound
        if scale_width > scale_height:
          # fit width
          scale_height = scale_width
        else:
          # fit height
          scale_width = scale_height
      elif self.__resize_method == "upper_bound":
        # scale such that output size is upper bound
        if scale_width < scale_height:
          # fit width
          scale_height = scale_width
        else:
          # fit height
          scale_width = scale_height
      elif self.__resize_method == "minimal":
        # scale as least as possbile
        if abs(1 - scale_width) < abs(1 - scale_height):
          # fit width
          scale_height = scale_width
        else:
          # fit height
          scale_width = scale_height
      else:
        raise ValueError(f"resize_method {self.__resize_method} not implemented")

    if self.__resize_method == "lower_bound":
      new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.__height)
      new_width = self.constrain_to_multiple_of(scale_width * width, min_val=self.__width)
    elif self.__resize_method == "upper_bound":
      new_height = self.constrain_to_multiple_of(scale_height * height, max_val=self.__height)
      new_width = self.constrain_to_multiple_of(scale_width * width, max_val=self.__width)
    elif self.__resize_method == "minimal":
      new_height = self.constrain_to_multiple_of(scale_height * height)
      new_width = self.constrain_to_multiple_of(scale_width * width)
    else:
      raise ValueError(f"resize_method {self.__resize_method} not implemented")

    return (new_width, new_height)

  def __call__(self, sample):
    width, height = self.get_size(sample["image"].shape[1], sample["image"].shape[0])

    # resize sample
    sample["image"] = cv2.resize(sample["image"], (width, height), interpolation=self.__image_interpolation_method)

    if self.__resize_target:
      if "depth" in sample:
        sample["depth"] = cv2.resize(sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST)

      if "mask" in sample:
        sample["mask"] = cv2.resize(sample["mask"].astype(np.float32), (width, height),
                                    interpolation=cv2.INTER_NEAREST)

    return sample



def image_tensor_preprocess( raw_image, input_size=160):
  transform = Compose([

    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
  ])


  image = raw_image/ 255.0

  transpform_images = transform(image)

  return transpform_images


def vis_batch_data_scores(pose_data, ids, scores, pad_margin=5,mask_crop = False):
  assert len(scores)==len(ids)
  canvas = []
  for i,id in enumerate(ids):
    rgbA_vis = (pose_data.rgbAs[id]*255).permute(1,2,0).data.cpu().numpy()
    rgbB_vis = (pose_data.rgbBs[id]*255).permute(1,2,0).data.cpu().numpy()
    H,W = rgbA_vis.shape[:2]
    zmin = 0
    zmax = 2
    depthA_vis = depth_to_vis(pose_data.depthAs[id].data.cpu().numpy().reshape(H,W), zmin=zmin, zmax=zmax, inverse=False)
    depthB_vis = depth_to_vis(pose_data.depthBs[id].data.cpu().numpy().reshape(H,W), zmin=zmin, zmax=zmax, inverse=False)
    maskA_vis = torch.where(pose_data.rgbAs[id]>0, 1, 0).permute(1,2,0).data.cpu().numpy().astype(np.uint8)
    maskB_vis = torch.where(pose_data.maskBs[id]>0, 1, 0).permute(1,2,0).data.cpu().numpy().astype(np.uint8)
    maskAs_orig = torch.where(pose_data.maskAs_orig[id]>0, 1, 0).permute(1,2,0).data.cpu().numpy().astype(np.uint8)
    maskBs_orig = torch.where(pose_data.maskBs_orig[id]>0, 1, 0).permute(1,2,0).data.cpu().numpy().astype(np.uint8)
    maskAs_orig = maskAs_orig*255
    maskBs_orig = maskBs_orig*255


    mask_cross_orig = (0.5*maskAs_orig + 0.5*maskBs_orig).astype(np.uint8)
    mask_cross_orig = cv2.resize(mask_cross_orig, (W,H), interpolation=cv2.INTER_NEAREST)
    mask_cross_orig = mask_cross_orig[...,None]
    mask_cross_orig = np.concatenate([mask_cross_orig, mask_cross_orig, mask_cross_orig], axis=2)

    mask_cross = (0.5*maskA_vis + 0.5*maskB_vis).astype(np.uint8) * 255
    if i == 0:
      mask_cross_hollow = maskA_vis * maskB_vis
      score_mask_hollow = np.sum((maskA_vis - mask_cross_hollow)[...,0])
      score_mask_cross = np.sum((maskB_vis - mask_cross_hollow)[...,0])
      print(f"first_rank_input_mask_score: {score_mask_hollow}")
      print(f"first_rank_pred_mask_score: {score_mask_cross}")
    # mask_cross[mask_cross == 1] = 125
    # mask_cross[mask_cross == 2] = 255
    if pose_data.normalAs is not None:
      pass
    pad = np.ones((rgbA_vis.shape[0],pad_margin,3))*255
    if pose_data.normalAs is not None:
      pass
    else:
      if mask_crop:
        row = np.concatenate([rgbA_vis, pad, depthA_vis, pad, rgbB_vis, pad, depthB_vis,pad,mask_cross_orig], axis=1)
      else:
        row = np.concatenate([rgbA_vis, pad, depthA_vis, pad, rgbB_vis, pad, depthB_vis,pad,mask_cross], axis=1)
    s = 100/row.shape[0]
    row = cv2.resize(row, fx=s, fy=s, dsize=None)
    row = cv_draw_text(row, text=f'id:{id}, score:{scores[id]:.3f}', uv_top_left=(10,10), color=(0,255,0), fontScale=0.5)
    canvas.append(row)
    pad = np.ones((pad_margin, row.shape[1], 3))*255
    canvas.append(pad)
  canvas = np.concatenate(canvas, axis=0).astype(np.uint8)
  canvas_list = []
  canvas_list.append(canvas)
  best_id = ids[0]
  depthAs_orig_best = pose_data.depthAs_orig[best_id].data.cpu().numpy()
  maskAs_orig_best = pose_data.maskAs_orig[best_id].data.cpu().numpy()
  maskAs_orig_best = np.where(maskAs_orig_best>0, 1, 0)[0]
  maskAs_orig_best = maskAs_orig_best.astype(np.uint8)*255
  maskBs_orig_best = pose_data.maskBs_orig[best_id].data.cpu().numpy()
  maskBs_orig_best = np.where(maskBs_orig_best>0, 1, 0)[0]
  maskBs_orig_best = maskBs_orig_best.astype(np.uint8)*255
  maskAs_orig_erode = cv2.erode(maskAs_orig_best, np.ones((3,3),np.uint8), iterations = 1)
  edgeAs_orig_best = maskAs_orig_best - maskAs_orig_erode
  edgeAs_orig_best = edgeAs_orig_best*255
  edgeAs_orig_best = np.stack([np.zeros_like(edgeAs_orig_best), np.zeros_like(edgeAs_orig_best), edgeAs_orig_best], axis=2)
  maskBs_orig_erode = cv2.erode(maskBs_orig_best, np.ones((3,3),np.uint8), iterations = 1)
  edgeBs_orig_best = maskBs_orig_best - maskBs_orig_erode
  edgeBs_orig_best = edgeBs_orig_best*255
  edgeBs_orig_best = np.stack([np.zeros_like(edgeBs_orig_best), edgeBs_orig_best, np.zeros_like(edgeBs_orig_best)], axis=2)
  edge_mix = edgeBs_orig_best + edgeAs_orig_best

  best_list = [depthAs_orig_best, maskAs_orig_best, edge_mix]
  canvas_list.append(best_list)
  return canvas_list



@torch.no_grad()
def make_crop_data_batch(render_size, ob_in_cams, mesh, rgb, depth, K, crop_ratio, normal_map=None, mesh_diameter=None, glctx=None, mesh_tensors=None, dataset:TripletH5Dataset=None, cfg=None):
  logging.info("Welcome make_crop_data_batch")
  H,W = depth.shape[:2]

  args = []
  method = 'box_3d'
  tf_to_crops = compute_crop_window_tf_batch(pts=mesh.vertices, H=H, W=W, poses=ob_in_cams, K=K, crop_ratio=crop_ratio, out_size=(render_size[1], render_size[0]), method=method, mesh_diameter=mesh_diameter)
  logging.info("make tf_to_crops done")

  B = len(ob_in_cams)
  poseAs = torch.as_tensor(ob_in_cams, dtype=torch.float, device='cuda')

  bs = 512
  rgb_rs = []
  depth_rs = []
  xyz_map_rs = []

  bbox2d_crop = torch.as_tensor(np.array([0, 0, cfg['input_resize'][0]-1, cfg['input_resize'][1]-1]).reshape(2,2), device='cuda', dtype=torch.float)
  bbox2d_ori = transform_pts(bbox2d_crop, tf_to_crops.inverse()[:,None]).reshape(-1,4)

  for b in range(0,len(ob_in_cams),bs):
    extra = {}
    rgb_r, depth_r, normal_r = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=poseAs[b:b+bs], context='cuda', get_normal=cfg['use_normal'], glctx=glctx, mesh_tensors=mesh_tensors, output_size=cfg['input_resize'], bbox2d=bbox2d_ori[b:b+bs], use_light=True, extra=extra)
    rgb_rs.append(rgb_r)
    depth_rs.append(depth_r[...,None])
    xyz_map_rs.append(extra['xyz_map'])

  rgb_rs = torch.cat(rgb_rs, dim=0).permute(0,3,1,2) * 255
  depth_rs = torch.cat(depth_rs, dim=0).permute(0,3,1,2)
  xyz_map_rs = torch.cat(xyz_map_rs, dim=0).permute(0,3,1,2)  #(B,3,H,W)
  logging.info("render done")

  rgbBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(rgb, dtype=torch.float, device='cuda').permute(2,0,1)[None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
  depthBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(depth, dtype=torch.float, device='cuda')[None,None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  if rgb_rs.shape[-2:]!=cfg['input_resize']:
    rgbAs = kornia.geometry.transform.warp_perspective(rgb_rs, tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
    depthAs = kornia.geometry.transform.warp_perspective(depth_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  else:
    rgbAs = rgb_rs
    depthAs = depth_rs

  if xyz_map_rs.shape[-2:]!=cfg['input_resize']:
    xyz_mapAs = kornia.geometry.transform.warp_perspective(xyz_map_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  else:
    xyz_mapAs = xyz_map_rs

  normalAs = None
  normalBs = None

  Ks = torch.as_tensor(K, dtype=torch.float).reshape(1,3,3).expand(len(rgbAs),3,3)
  mesh_diameters = torch.ones((len(rgbAs)), dtype=torch.float, device='cuda')*mesh_diameter

  pose_data = BatchPoseData(rgbAs=rgbAs, rgbBs=rgbBs, depthAs=depthAs, depthBs=depthBs, normalAs=normalAs, normalBs=normalBs, poseA=poseAs, xyz_mapAs=xyz_mapAs, tf_to_crops=tf_to_crops, Ks=Ks, mesh_diameters=mesh_diameters)
  pose_data = dataset.transform_batch(pose_data, H_ori=H, W_ori=W, bound=1)

  logging.info("pose batch data done")

  return pose_data

@torch.no_grad()
# def make_crop_data_batch_score_version_rgb(render_size, ob_in_cams, mesh,  depth, K, crop_ratio, normal_map=None, mesh_diameter=None, glctx=None, mesh_tensors=None, dataset:TripletH5Dataset=None, cfg=None,mask=None,rgb = None, rgb_origin = None):
#   logging.info("Welcome make_crop_data_batch")
#   H,W = depth.shape[:2]
#
#   args = []
#   method = 'box_3d'
#   tf_to_crops = compute_crop_window_tf_batch(pts=mesh.vertices, H=H, W=W, poses=ob_in_cams, K=K, crop_ratio=crop_ratio, out_size=(render_size[1], render_size[0]), method=method, mesh_diameter=mesh_diameter)
#   logging.info("make tf_to_crops done")
#
#   B = len(ob_in_cams)
#   poseAs = torch.as_tensor(ob_in_cams, dtype=torch.float, device='cuda')
#
#   bs = 512
#   rgb_rs = []
#   depth_rs = []
#   xyz_map_rs = []
#
#   bbox2d_crop = torch.as_tensor(np.array([0, 0, cfg['input_resize'][0]-1, cfg['input_resize'][1]-1]).reshape(2,2), device='cuda', dtype=torch.float)
#   bbox2d_ori = transform_pts(bbox2d_crop, tf_to_crops.inverse()[:,None]).reshape(-1,4)
#
#   for b in range(0,len(ob_in_cams),bs):
#     extra = {}
#     rgb_r, depth_r, normal_r = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=poseAs[b:b+bs], context='cuda', get_normal=cfg['use_normal'], glctx=glctx, mesh_tensors=mesh_tensors, output_size=cfg['input_resize'], bbox2d=bbox2d_ori[b:b+bs], use_light=True, extra=extra,rgb_tensor = rgb_origin)
#     rgb_rs.append(rgb_r)
#     depth_rs.append(depth_r[...,None])
#     xyz_map_rs.append(extra['xyz_map'])
#
#   # rgb_r_cpu = rgb_rs[0].detach().cpu().numpy().astyep(np.uint8)
#   # plt.imshow(rgb_r_cpu)
#   # plt.show()
#
#   rgb_rs = torch.cat(rgb_rs, dim=0).permute(0,3,1,2) * 255
#   depth_rs = torch.cat(depth_rs, dim=0).permute(0,3,1,2)
#   # depth_rs = torch.zeros_like(depth_rs)
#   xyz_map_rs = torch.cat(xyz_map_rs, dim=0).permute(0,3,1,2)  #(B,3,H,W)
#   logging.info("render done")
#
#
#   rgbBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(rgb, dtype=torch.float, device='cuda').permute(2,0,1)[None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
#   depthBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(depth, dtype=torch.float, device='cuda')[None,None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
#   maskBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(mask, dtype=torch.float, device='cuda')[None,None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
#   if rgb_rs.shape[-2:]!=cfg['input_resize']:
#     rgbAs = kornia.geometry.transform.warp_perspective(rgb_rs, tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
#     depthAs = kornia.geometry.transform.warp_perspective(depth_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
#   else:
#     rgbAs = rgb_rs
#     depthAs = depth_rs
#
#   if xyz_map_rs.shape[-2:]!=cfg['input_resize']:
#     xyz_mapAs = kornia.geometry.transform.warp_perspective(xyz_map_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
#   else:
#     xyz_mapAs = xyz_map_rs
#
#   normalAs = None
#   normalBs = None
#
#   Ks = torch.as_tensor(K, dtype=torch.float).reshape(1,3,3).expand(len(rgbAs),3,3)
#   mesh_diameters = torch.ones((len(rgbAs)), dtype=torch.float, device='cuda')*mesh_diameter
#
#   #floor for rgbAs
#   rgbAs = torch.floor(rgbAs)
#
#   pose_data = BatchPoseData(rgbAs=rgbAs, rgbBs=rgbBs, depthAs=depthAs, depthBs=depthBs, normalAs=normalAs, normalBs=normalBs, poseA=poseAs, xyz_mapAs=xyz_mapAs, tf_to_crops=tf_to_crops, Ks=Ks, mesh_diameters=mesh_diameters,maskBs=maskBs)
#   pose_data = dataset.transform_batch(pose_data, H_ori=H, W_ori=W, bound=1)
#
#   logging.info("pose batch data done")
#
#   return pose_data


def make_crop_data_batch_score_version(render_size, ob_in_cams, mesh,  depth, K, crop_ratio, normal_map=None, mesh_diameter=None, glctx=None, mesh_tensors=None, dataset:TripletH5Dataset=None, cfg=None,mask=None,rgb = None, rgb_origin = None,zero_depth = False, mask_crop = True):
  logging.info("Welcome make_crop_data_batch")
  H,W = depth.shape[:2]

  args = []
  method = 'box_3d'

  def compute_tf_batch(left, right, top, bottom):
    B = len(left)
    left = left.round()
    right = right.round()
    top = top.round()
    bottom = bottom.round()

    tf = torch.eye(3)[None].expand(B,-1,-1).contiguous()
    tf[:,0,2] = -left
    tf[:,1,2] = -top
    new_tf = torch.eye(3)[None].expand(B,-1,-1).contiguous()
    new_tf[:,0,0] = render_size[1]/(right-left)
    new_tf[:,1,1] = render_size[0]/(bottom-top)
    tf = new_tf@tf
    return tf

  B = len(ob_in_cams)
  # find bounding box of the mask
  x, y = np.where(mask>0)
  padding = 85
  left = np.min(y)
  left = max(0, left-padding)
  right = np.max(y)
  right = min(W-1, right+padding)
  top = np.min(x)
  top = max(0, top-padding)
  bottom = np.max(x)
  bottom = min(H-1, bottom+padding)
  left = torch.ones(B, dtype=torch.float, device='cuda')*left
  right = torch.ones(B, dtype=torch.float, device='cuda')*right
  top = torch.ones(B, dtype=torch.float, device='cuda')*top
  bottom = torch.ones(B, dtype=torch.float, device='cuda')*bottom

  tf_to_crops = compute_tf_batch(left, right, top, bottom)
  tf_to_crops_new = tf_to_crops.cuda()

  tf_to_crops = compute_crop_window_tf_batch(pts=mesh.vertices, H=H, W=W, poses=ob_in_cams, K=K, crop_ratio=crop_ratio, out_size=(render_size[1], render_size[0]), method=method, mesh_diameter=mesh_diameter)
  logging.info("make tf_to_crops done")
  if mask_crop:
    pass
  else:
    tf_to_crops_new = tf_to_crops


  B = len(ob_in_cams)
  poseAs = torch.as_tensor(ob_in_cams, dtype=torch.float, device='cuda')

  bs = 512
  rgb_rs = []
  depth_rs = []
  xyz_map_rs = []
  mask_rs = []
  mask_pred_orig_list = []
  depth_pred_orig_list = []

  bbox2d_crop = torch.as_tensor(np.array([0, 0, cfg['input_resize'][0]-1, cfg['input_resize'][1]-1]).reshape(2,2), device='cuda', dtype=torch.float)
  bbox2d_ori = transform_pts(bbox2d_crop, tf_to_crops.inverse()[:,None]).reshape(-1,4)

  for b in range(0,len(ob_in_cams),bs):
    extra = {}
    rgb_r, depth_r, normal_r,mask_r,mask_pred_orig,depth_pred_orig = nvdiffrast_render_mask_version(K=K, H=H, W=W, ob_in_cams=poseAs[b:b+bs], context='cuda', get_normal=cfg['use_normal'], glctx=glctx, mesh_tensors=mesh_tensors, output_size=cfg['input_resize'], bbox2d=bbox2d_ori[b:b+bs], use_light=True, extra=extra,rgb_tensor = rgb_origin)
    rgb_rs.append(rgb_r)
    depth_rs.append(depth_r[...,None])
    xyz_map_rs.append(extra['xyz_map'])
    mask_rs.append(mask_r)
    mask_pred_orig_list.append(mask_pred_orig)
    depth_pred_orig_list.append(depth_pred_orig[...,None])


  rgb_rs = torch.cat(rgb_rs, dim=0).permute(0,3,1,2) * 255
  depth_rs = torch.cat(depth_rs, dim=0).permute(0,3,1,2)
  if zero_depth:
    depth_rs = torch.zeros_like(depth_rs)
  xyz_map_rs = torch.cat(xyz_map_rs, dim=0).permute(0,3,1,2)  #(B,3,H,W)
  mask_rs = torch.cat(mask_rs, dim=0).permute(0,3,1,2)
  mask_pred_orig = torch.cat(mask_pred_orig_list, dim=0).permute(0,3,1,2)
  depth_pred_orig = torch.cat(depth_pred_orig_list, dim=0).permute(0,3,1,2)
  logging.info("render done")

  maskAs_orig = mask_pred_orig
  depthAs_orig = depth_pred_orig
  maskBs_orig = torch.tensor(mask, dtype=torch.float, device='cuda')[None,None].expand(B,-1,-1,-1)
  depthBs_orig = torch.tensor(depth, dtype=torch.float, device='cuda')[None,None].expand(B,-1,-1,-1)
  rgbBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(rgb, dtype=torch.float, device='cuda').permute(2,0,1)[None].expand(B,-1,-1,-1), tf_to_crops_new, dsize=render_size, mode='bilinear', align_corners=False)
  depthBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(depth, dtype=torch.float, device='cuda')[None,None].expand(B,-1,-1,-1), tf_to_crops_new, dsize=render_size, mode='nearest', align_corners=False)
  maskBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(mask, dtype=torch.float, device='cuda')[None,None].expand(B,-1,-1,-1), tf_to_crops_new, dsize=render_size, mode='nearest', align_corners=False)
  if rgb_rs.shape[-2:]!=cfg['input_resize']:
    rgbAs = kornia.geometry.transform.warp_perspective(rgb_rs, tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
    depthAs = kornia.geometry.transform.warp_perspective(depth_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  else:
    rgbAs = rgb_rs
    depthAs = depth_rs

  if xyz_map_rs.shape[-2:]!=cfg['input_resize']:
    xyz_mapAs = kornia.geometry.transform.warp_perspective(xyz_map_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  else:
    xyz_mapAs = xyz_map_rs

  normalAs = None
  normalBs = None




  Ks = torch.as_tensor(K, dtype=torch.float).reshape(1,3,3).expand(len(rgbAs),3,3)
  mesh_diameters = torch.ones((len(rgbAs)), dtype=torch.float, device='cuda')*mesh_diameter

  #floor for rgbAs
  rgbAs = torch.floor(rgbAs)

  pose_data = BatchPoseData(rgbAs=rgbAs, rgbBs=rgbBs, depthAs=depthAs, depthBs=depthBs, normalAs=normalAs, normalBs=normalBs, poseA=poseAs, xyz_mapAs=xyz_mapAs, tf_to_crops=tf_to_crops, Ks=Ks, mesh_diameters=mesh_diameters,maskBs=maskBs,maskAs=mask_rs,maskAs_orig=maskAs_orig,depthAs_orig=depthAs_orig,maskBs_orig=maskBs_orig,depthBs_orig=depthBs_orig)
  pose_data = dataset.transform_batch(pose_data, H_ori=H, W_ori=W, bound=1)

  logging.info("pose batch data done")

  return pose_data

class ScorePredictor:
  def __init__(self, amp=True):
    self.amp = amp
    self.run_name = "2024-01-11-20-02-45"

    model_name = 'model_best.pth'
    code_dir = os.path.dirname(os.path.realpath(__file__))
    ckpt_dir = f'{code_dir}/../../weights/{self.run_name}/{model_name}'

    self.cfg = OmegaConf.load(f'{code_dir}/../../weights/{self.run_name}/config.yml')

    self.cfg['ckpt_dir'] = ckpt_dir
    self.cfg['enable_amp'] = True

    ########## Defaults, to be backward compatible
    if 'use_normal' not in self.cfg:
      self.cfg['use_normal'] = False
    if 'use_BN' not in self.cfg:
      self.cfg['use_BN'] = False
    if 'zfar' not in self.cfg:
      self.cfg['zfar'] = np.inf
    if 'c_in' not in self.cfg:
      self.cfg['c_in'] = 4
    if 'normalize_xyz' not in self.cfg:
      self.cfg['normalize_xyz'] = False
    if 'crop_ratio' not in self.cfg or self.cfg['crop_ratio'] is None:
      self.cfg['crop_ratio'] = 1.2

    logging.info(f"self.cfg: \n {OmegaConf.to_yaml(self.cfg)}")

    self.dataset = ScoreMultiPairH5Dataset(cfg=self.cfg, mode='test', h5_file=None, max_num_key=1)
    self.model = ScoreNetMultiPair(cfg=self.cfg, c_in=self.cfg['c_in']).cuda()

    logging.info(f"Using pretrained model from {ckpt_dir}")
    ckpt = torch.load(ckpt_dir)
    if 'model' in ckpt:
      ckpt = ckpt['model']
    self.model.load_state_dict(ckpt)

    self.model.cuda().eval()
    logging.info("init done")

  @torch.inference_mode()
  def first_selection(self, rgb, depth, K, ob_in_cams, normal_map=None, get_vis=True, mesh=None, mesh_tensors=None, glctx=None,
              mesh_diameter=None,mask = None):
    '''
    @rgb: np array (H,W,3)
    '''
    logging.info(f"ob_in_cams:{ob_in_cams.shape}")
    ob_in_cams = torch.as_tensor(ob_in_cams, dtype=torch.float, device='cuda')

    logging.info(f'self.cfg.use_normal:{self.cfg.use_normal}')
    if not self.cfg.use_normal:
      normal_map = None

    logging.info("making cropped data")

    if mesh_tensors is None:
      mesh_tensors = make_mesh_tensors(mesh)

    rgb = torch.as_tensor(rgb, device='cuda', dtype=torch.float)
    depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)

    pose_data = make_crop_data_batch_score_version(self.cfg.input_resize, ob_in_cams, mesh, rgb, depth, K,
                                                   crop_ratio=self.cfg['crop_ratio'], glctx=glctx,
                                                   mesh_tensors=mesh_tensors, dataset=self.dataset, cfg=self.cfg,
                                                   mesh_diameter=mesh_diameter)

    def find_best_among_pairs(pose_data: BatchPoseData):
      logging.info(f'pose_data.rgbAs.shape[0]: {pose_data.rgbAs.shape[0]}')
      ids = []
      scores = []
      bs = pose_data.rgbAs.shape[0]
      for b in range(0, pose_data.rgbAs.shape[0], bs):
        A = torch.cat([pose_data.rgbAs[b:b + bs].cuda(), pose_data.xyz_mapAs[b:b + bs].cuda()], dim=1).float()
        B = torch.cat([pose_data.rgbBs[b:b + bs].cuda(), pose_data.xyz_mapBs[b:b + bs].cuda()], dim=1).float()
        if pose_data.normalAs is not None:
          A = torch.cat([A, pose_data.normalAs.cuda().float()], dim=1)
          B = torch.cat([B, pose_data.normalBs.cuda().float()], dim=1)
        with torch.cuda.amp.autocast(enabled=self.amp):
          output = self.model(A, B, L=len(A))
        scores_cur = output["score_logit"].float().reshape(-1)
        ids.append(scores_cur.argmax() + b)
        scores.append(scores_cur)
      ids = torch.stack(ids, dim=0).reshape(-1)
      scores = torch.cat(scores, dim=0).reshape(-1)
      return ids, scores

    pose_data_iter = pose_data
    global_ids = torch.arange(len(ob_in_cams), device='cuda', dtype=torch.long)
    scores_global = torch.zeros((len(ob_in_cams)), dtype=torch.float, device='cuda')

    while 1:
      ids, scores = find_best_among_pairs(pose_data_iter)
      if len(ids) == 1:
        scores_global[global_ids] = scores + 100
        break
      global_ids = global_ids[ids]
      pose_data_iter = pose_data.select_by_indices(global_ids)

    scores = scores_global

    logging.info(f'forward done')
    torch.cuda.empty_cache()

    if get_vis:
      logging.info("get_vis...")
      canvas = []
      ids = scores.argsort(descending=True)
      canvas = vis_batch_data_scores(pose_data, ids=ids, scores=scores)
      return scores, canvas

    return scores, None

  @torch.inference_mode()
  def predict(self, rgb, depth, K, ob_in_cams, normal_map=None, get_vis=True, mesh=None, mesh_tensors=None, glctx=None, mesh_diameter=None,mask = None,obj_dia = None,rgb_origin = None,depth_anything = None,zero_depth = False,mask_crop = False, losses=None):
    '''
    @rgb: np array (H,W,3)
    '''
    logging.info(f"ob_in_cams:{ob_in_cams.shape}")
    ob_in_cams = torch.as_tensor(ob_in_cams, dtype=torch.float, device='cuda')

    logging.info(f'self.cfg.use_normal:{self.cfg.use_normal}')
    if not self.cfg.use_normal:
      normal_map = None

    logging.info("making cropped data")

    if mesh_tensors is None:
      mesh_tensors = make_mesh_tensors(mesh)
    rgb = torch.as_tensor(rgb, device='cuda', dtype=torch.float)

    depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)
    rgb_origin = torch.as_tensor(rgb_origin, device='cuda', dtype=torch.float)

    pose_data = make_crop_data_batch_score_version(self.cfg.input_resize, ob_in_cams, mesh,  depth, K, crop_ratio=self.cfg['crop_ratio'], glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset, cfg=self.cfg, mesh_diameter=mesh_diameter,mask=mask,rgb = rgb,rgb_origin = rgb_origin,zero_depth = zero_depth,mask_crop=mask_crop)

    def find_best_among_pairs(pose_data:BatchPoseData):
      logging.info(f'pose_data.rgbAs.shape[0]: {pose_data.rgbAs.shape[0]}')
      ids = []
      scores = []
      bs = pose_data.rgbAs.shape[0]
      for b in range(0, pose_data.rgbAs.shape[0], bs):
        A = torch.cat([pose_data.rgbAs[b:b+bs].cuda(), pose_data.xyz_mapAs[b:b+bs].cuda()], dim=1).float()
        B = torch.cat([pose_data.rgbBs[b:b+bs].cuda(), pose_data.xyz_mapBs[b:b+bs].cuda()], dim=1).float()
        if pose_data.normalAs is not None:
          A = torch.cat([A, pose_data.normalAs.cuda().float()], dim=1)
          B = torch.cat([B, pose_data.normalBs.cuda().float()], dim=1)
        with torch.cuda.amp.autocast(enabled=self.amp):
          output = self.model(A, B, L=len(A))
        scores_cur = output["score_logit"].float().reshape(-1)
        ids.append(scores_cur.argmax()+b)
        scores.append(scores_cur)
      ids = torch.stack(ids, dim=0).reshape(-1)
      scores = torch.cat(scores, dim=0).reshape(-1)
      return ids, scores

    bs = pose_data.rgbAs.shape[0]
    if depth_anything:
      time1 = time.perf_counter()
      for b in range(0, pose_data.rgbAs.shape[0], bs):
        A = pose_data.rgbAs[b:b + bs].cuda().float()
        B = pose_data.rgbBs[b:b + bs].cuda().float()
        #
        outputAs = []
        outputBs = []
        with torch.cuda.amp.autocast(enabled=self.amp):
          # for i in range(B.shape[0]):

          half = len(A) // 2
          # for i in range(len(A)):
          #
          #   outputA = depth_anything.infer_image_torch(A[i].permute(1,2,0), input_size=518)
          #   outputB = depth_anything.infer_image_torch(B[i].permute(1,2,0), input_size=518)
          #
          #   outputA = torch.tensor(outputA).cuda().float()
          #   outputB = torch.tensor(outputB).cuda().float()
          #   outputA = outputA.unsqueeze(0).unsqueeze(0)
          #   outputB = outputB.unsqueeze(0).unsqueeze(0)
          #   outputAs.append(outputA)
          #   outputBs.append(outputB)

          outputA1 = depth_anything.infer_image_torch_batch(A[:half],
                                                                input_size=560)
          outputA2 = depth_anything.infer_image_torch_batch(A[half:], input_size=560)
          outputAs.append(outputA1)
          outputAs.append(outputA2)

          outputB1 = depth_anything.infer_image_torch_batch(B[:half],
                                                            input_size=560)
          outputB2 = depth_anything.infer_image_torch_batch(B[half:], input_size=560)
          outputBs.append(outputB1)
          outputBs.append(outputB2)
        #     outputA = outputA.unsqueeze(0)
        #     outputAs.append(outputA)
        #   outputB = depth_anything.infer_image(B[0].permute(1, 2, 0).detach().cpu().numpy() * 255, input_size=518)
        #   # outputB = torch.tensor(outputB).cuda().float()
        # outputB = outputB.unsqueeze(0)
        # outputB = outputB.unsqueeze(1)
        outputA = torch.cat(outputAs, dim=0)
        outputB = torch.cat(outputBs, dim=0)
        # outputB = outputB.repeat(outputA.shape[0], 1, 1, 1)
        A = torch.nn.functional.interpolate(outputA, size=(160, 160), mode='bilinear', align_corners=False)
        B = torch.nn.functional.interpolate(outputB, size=(160, 160), mode='bilinear', align_corners=False)

        maskB_batch = pose_data.maskBs[b:b + bs].cuda().float()
        maskA_batch = pose_data.maskAs[b:b + bs].cuda().float()
        A = A * maskA_batch
        B = B * maskB_batch
        # maskB_batch = torch.ones_like(maskB_batch).cuda().float()
        # B = B * maskB_batch

        A_non_zero_batch = A.reshape(A.shape[0], -1).detach().clone()
        B_non_zero_batch = B.reshape(B.shape[0], -1).detach().clone()
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
        # A[A == 100000.0] = 0.0
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

        # A = torch.ones_like(A).cuda()
        pose_data.rgbAs[b:b + bs] = A_new
        pose_data.rgbBs[b:b + bs] = B_new
        pose_data.depthAs[b:b + bs] = A_new
        pose_data.depthBs[b:b + bs] = B_new
        time2 = time.perf_counter()
        print(f"depth_anything_time: {time2 - time1}")
        # depthA_cpu = A_new[0][0].detach().cpu().numpy()
        # depthB_cpu = B_new[0][0].detach().cpu().numpy()
        #
        # plt.imshow(depthA_cpu)
        # plt.show()
        # plt.imshow(depthB_cpu)
        # plt.show()
    pose_data_iter = pose_data
    global_ids = torch.arange(len(ob_in_cams), device='cuda', dtype=torch.long)
    scores_global = torch.zeros((len(ob_in_cams)), dtype=torch.float, device='cuda')

    while 1:
      ids, scores = find_best_among_pairs(pose_data_iter)
      if len(ids)==1:
        scores_global[global_ids] = scores + 100
        break
      global_ids = global_ids[ids]
      pose_data_iter = pose_data.select_by_indices(global_ids)

    scores = scores_global

    logging.info(f'forward done')
    torch.cuda.empty_cache()

    if get_vis :
      logging.info("get_vis...")
      canvas = []
      if losses is not None:
        ids = losses.argsort(descending=False)
      else:
        ids = scores.argsort(descending=True)
      canvas = vis_batch_data_scores(pose_data, ids=ids, scores=scores,mask_crop = mask_crop)
      return scores, canvas

    return scores, None

