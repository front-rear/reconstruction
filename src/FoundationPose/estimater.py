# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import os.path
import time

import matplotlib.pyplot as plt
import torch

from Utils import *
from datareader import *
import itertools
from learning.training.predict_score import *
from learning.training.predict_pose_refine import *
import yaml
import numpy as np

from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix
class FoundationPose:
  def __init__(self, model_pts, model_normals, symmetry_tfs=None, mesh=None, scorer:ScorePredictor=None, refiner:PoseRefinePredictor=None, glctx=None, debug=0, debug_dir='/home/bowen/debug/novel_pose_debug/',obj_id = None,img_id = None,scene_id = None,dataset = None):
    self.gt_pose = None
    self.ignore_normal_flip = True
    self.debug = debug
    self.dataset = dataset

    self.adapt_projection = None

    self.obj_id = obj_id
    self.img_id = img_id
    self.scene_id = scene_id

    self.reset_object(model_pts, model_normals, symmetry_tfs=symmetry_tfs, mesh=mesh)
    # self.make_rotation_grid(min_n_views=40, inplane_step=60)
    self.make_rotation_grid(min_n_views=40, inplane_step=60)
    self.glctx = glctx

    if scorer is not None:
      self.scorer = scorer
    else:
      self.scorer = ScorePredictor()

    if refiner is not None:
      self.refiner = refiner
    # else:
    #   self.refiner = PoseRefinePredictor(obj_id)

    self.pose_last = None   # Used for tracking; per the centered mesh

    self.debug_dir = None
  def get_debug_dir(self,debug_dir):
    self.debug_dir = debug_dir
    os.makedirs(debug_dir, exist_ok=True)


  def get_obj_id_and_init_refiner(self,obj_id,adapt_projection):
    self.obj_id = obj_id
    self.adapt_projection = adapt_projection
    self.refiner = PoseRefinePredictor(obj_id,adapt_projection)
  def reset_object(self, model_pts, model_normals, symmetry_tfs=None, mesh=None):
    max_xyz = mesh.vertices.max(axis=0)
    min_xyz = mesh.vertices.min(axis=0)
    self.model_center = (min_xyz+max_xyz)/2
    if mesh is not None:
      self.mesh_ori = mesh.copy()
      mesh = mesh.copy()
      mesh.vertices = mesh.vertices - self.model_center.reshape(1,3)

    model_pts = mesh.vertices
    self.diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)
    self.vox_size = max(self.diameter/20.0, 0.003)
    logging.info(f'self.diameter:{self.diameter}, vox_size:{self.vox_size}')
    self.dist_bin = self.vox_size/2
    self.angle_bin = 20  # Deg
    pcd = toOpen3dCloud(model_pts, normals=model_normals)
    pcd = pcd.voxel_down_sample(self.vox_size)
    self.max_xyz = np.asarray(pcd.points).max(axis=0)
    self.min_xyz = np.asarray(pcd.points).min(axis=0)
    self.pts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')
    self.normals = F.normalize(torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device='cuda'), dim=-1)
    logging.info(f'self.pts:{self.pts.shape}')
    self.mesh_path = None
    self.mesh = mesh
    if self.mesh is not None:
      self.mesh_path = f'/tmp/{uuid.uuid4()}.obj'
      self.mesh.export(self.mesh_path)
    self.mesh_tensors = make_mesh_tensors(self.mesh)

    if symmetry_tfs is None:
      self.symmetry_tfs = torch.eye(4).float().cuda()[None]
    else:
      self.symmetry_tfs = torch.as_tensor(symmetry_tfs, device='cuda', dtype=torch.float)

    logging.info("reset done")



  def get_tf_to_centered_mesh(self):
    tf_to_center = torch.eye(4, dtype=torch.float, device='cuda')
    tf_to_center[:3,3] = -torch.as_tensor(self.model_center, device='cuda', dtype=torch.float)
    return tf_to_center


  def to_device(self, s='cuda:0'):
    for k in self.__dict__:
      self.__dict__[k] = self.__dict__[k]
      if torch.is_tensor(self.__dict__[k]) or isinstance(self.__dict__[k], nn.Module):
        logging.info(f"Moving {k} to device {s}")
        self.__dict__[k] = self.__dict__[k].to(s)
    for k in self.mesh_tensors:
      logging.info(f"Moving {k} to device {s}")
      self.mesh_tensors[k] = self.mesh_tensors[k].to(s)
    if self.refiner is not None:
      self.refiner.model.to(s)
    if self.scorer is not None:
      self.scorer.model.to(s)
    if self.glctx is not None:
      self.glctx = dr.RasterizeCudaContext(s)



  def make_rotation_grid(self, min_n_views=40, inplane_step=60):
    cam_in_obs = sample_views_icosphere(n_views=min_n_views)
    logging.info(f'cam_in_obs:{cam_in_obs.shape}')
    rot_grid = []
    for i in range(len(cam_in_obs)):
      for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
        cam_in_ob = cam_in_obs[i]
        R_inplane = euler_matrix(0,0,inplane_rot)
        cam_in_ob = cam_in_ob@R_inplane
        ob_in_cam = np.linalg.inv(cam_in_ob)
        rot_grid.append(ob_in_cam)

    # for i in range(len(cam_in_obs)):
    #     cam_in_ob = cam_in_obs[i]
    #     ob_in_cam = np.linalg.inv(cam_in_ob)
    #     rot_grid.append(ob_in_cam)

    rot_grid = np.asarray(rot_grid)
    logging.info(f"rot_grid:{rot_grid.shape}")
    # rot_grid = mycpp.cluster_poses(30, 99999, rot_grid, self.symmetry_tfs.data.cpu().numpy())
    rot_grid = np.asarray(rot_grid)
    logging.info(f"after cluster, rot_grid:{rot_grid.shape}")
    self.rot_grid = torch.as_tensor(rot_grid, device='cuda', dtype=torch.float)
    logging.info(f"self.rot_grid: {self.rot_grid.shape}")


  def generate_random_pose_hypo(self, K, rgb, depth, mask, scene_pts=None):
    '''
    @scene_pts: torch tensor (N,3)
    '''
    ob_in_cams = self.rot_grid.clone()
    center = self.guess_translation(depth=depth, mask=mask, K=K)
    ob_in_cams[:,:3,3] = torch.tensor(center, device='cuda', dtype=torch.float).reshape(1,3)
    return ob_in_cams


  def axis_angle_to_rotation_matrix(self, axis, angle):
    from scipy.spatial.transform import Rotation as R
    rot = R.from_rotvec(angle * axis)
    return rot.as_matrix()

  def generate_sample_pose_around_gt_pose(self, pose_gt, n_samples=200, angle_error = 5, translation_error = 0.01):
    '''
    @scene_pts: torch tensor (N,3)
    '''
    from scipy.spatial.transform import Rotation as R
    random_axis = np.random.uniform(-1, 1, size=(n_samples, 3))
    random_axis = random_axis / np.linalg.norm(random_axis, axis=1, keepdims=True)
    perturbation_rotations = self.axis_angle_to_rotation_matrix(random_axis, angle_error * np.pi / 180)
    rotation_gt = pose_gt[:3, :3]
    perturbation_rotations = np.einsum('ijk,kl->ijl', perturbation_rotations, rotation_gt)
    random_axis = np.random.uniform(-1, 1, size=(n_samples, 3))
    random_axis = random_axis / np.linalg.norm(random_axis, axis=1, keepdims=True)
    delta_translation = random_axis * translation_error
    translation_gt = pose_gt[:3, 3]
    perturbation_translations = translation_gt + delta_translation
    ob_in_cams = np.zeros((n_samples, 4, 4))
    ob_in_cams[:, :3, :3] = perturbation_rotations
    ob_in_cams[:, :3, 3] = perturbation_translations
    ob_in_cams[:, 3, 3] = 1



    return ob_in_cams


  def guess_translation(self, depth, mask, K):
    vs,us = np.where(mask>0)
    if len(us)==0:
      logging.info(f'mask is all zero')
      return np.zeros((3))
    uc = (us.min()+us.max())/2.0
    vc = (vs.min()+vs.max())/2.0
    valid = mask.astype(bool) & (depth>=0.1)
    if not valid.any():
      logging.info(f"valid is empty")
      return np.zeros((3))

    zc = np.median(depth[valid])
    center = (np.linalg.inv(K)@np.asarray([uc,vc,1]).reshape(3,1)) * zc

    if self.debug>=2:
      pcd = toOpen3dCloud(center.reshape(1,3))
      o3d.io.write_point_cloud(f'{self.debug_dir}/init_center.ply', pcd)

    return center.reshape(3)

  def add_random_rotation_translation(self, poses, angle,dis):
    num_poses = poses.shape[0]
    for i in range(num_poses):
      #generate the random number between 0 and 2 pi
      random_seta = np.random.uniform(0, 2*np.pi)
      #generate the random number between -1 and 1
      random_z = np.random.uniform(-1, 1)
      random_aix = np.array([np.cos(random_seta)*np.sqrt(1-random_z**2), np.sin(random_seta)*np.sqrt(1-random_z**2), random_z])
      random_aix = random_aix * angle * np.pi / 180
      random_aix = torch.tensor(random_aix, dtype=torch.float32, device='cuda')
      matrix = axis_angle_to_matrix(random_aix)
      poses[i,:3,:3] = matrix @ poses[i,:3,:3]

      random_seta = np.random.uniform(0, 2 * np.pi)
      # generate the random number between -1 and 1
      random_z = np.random.uniform(-1, 1)
      random_aix = np.array(
        [np.cos(random_seta) * np.sqrt(1 - random_z ** 2), np.sin(random_seta) * np.sqrt(1 - random_z ** 2), random_z])
      translation_noise = random_aix * dis
      translation_noise = torch.tensor(translation_noise, dtype=torch.float32, device='cuda')
      poses[i,:3,3] = poses[i,:3,3] + translation_noise

    return   poses

  def register(self, K, rgb, depth, ob_mask, ob_id=None, glctx=None, iteration=5,diff_render_iteration = 19,zero_depth=False,depth_anything_metric = None,depth_anything_score= None,indix = 1,special_name_var = "default",first_selection= False,mask_rgb = False,pure_color = False,inner = False,img_id = None,scene_id = None,only_edge = False,only_mask = False,set_id = None,sample_around_gt = False,sample_translation_error = None,sample_rotation_error = None,mask_crop = False,first_selection_num = 25):
    '''Copmute pose from given pts to self.pcd
    @pts: (N,3) np array, downsampled scene points
    '''
    # set_seed(0)
    logging.info('Welcome')

    self.img_id = img_id
    self.scene_id = scene_id
    self.set_id = set_id

    if self.glctx is None:
      if glctx is None:
        self.glctx = dr.RasterizeCudaContext()
        # self.glctx = dr.RasterizeGLContext()
      else:
        self.glctx = glctx

    # depth = erode_depth(depth, radius=2, device='cuda')
    # depth = bilateral_filter_depth(depth, radius=2, device='cuda')
    #
    # plt.imshow(depth)
    # plt.show()

    # if self.debug>=2:
    #   xyz_map = depth2xyzmap(depth , K)
    #   valid = (xyz_map[...,2]>=0.1) & (xyz_map[...,2] < 1.5)
    #   pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
    #   o3d.io.write_point_cloud(f'{self.debug_dir}/scene_raw.ply',pcd)
    #   cv2.imwrite(f'{self.debug_dir}/ob_mask.png', (ob_mask*255.0).clip(0,255))

    normal_map = None
    valid = (depth>=0.1) & (ob_mask>0)

    if valid.sum()<4:
      logging.info(f'valid too small, return')
      pose = np.eye(4)
      pose[:3,3] = self.guess_translation(depth=depth, mask=ob_mask, K=K)
      return pose

    if self.debug>=2:
      imageio.imwrite(f'{self.debug_dir}/color.png', rgb)
    #   cv2.imwrite(f'{self.debug_dir}/depth.png', (depth*1000).astype(np.uint16))
    #   valid = xyz_map[...,2]>=0.1 and xyz_map[...,2] < 2
    #   pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
    #   o3d.io.write_point_cloud(f'{self.debug_dir}/scene_raw.ply',pcd)

    self.H, self.W = depth.shape[:2]
    self.K = K
    self.ob_id = ob_id
    self.ob_mask = ob_mask

    poses = self.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None)

    #### Temporary code
    # phis = np.linspace(math.radians(-120), math.radians(0), poses.shape[0])
    phis = np.linspace(math.radians(-100), math.radians(-60), 128)

    obj0_under_base = torch.tensor([
      [1, 0, 0, 0.8],
      [0, 1, 0, 0],
      [0, 0, 1, 0.3],
      [0, 0, 0, 1]
    ])
    objs_under_obj0 = torch.tensor([[
      [math.cos(phi), -math.sin(phi), 0, 0],
      [math.sin(phi), math.cos(phi), 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1]
    ] for phi in phis])
    cam_under_base = torch.tensor([
      [0.70512699, -0.21169334, 0.67674357, 0.14125227],
      [-0.7090684, -0.216209, 0.67117485, -0.4770995],
      [0.0042348, -0.95312098, -0.30255986, 0.54021694],
      [0, 0, 0, 1]
    ])
    poses = torch.matmul(torch.inverse(cam_under_base), torch.matmul(obj0_under_base.unsqueeze(0), objs_under_obj0))
    poses = poses.data.cpu()
  
    ####

    poses = poses.data.cpu().numpy()
    print('pose0', poses[0])
    logging.info(f'poses:{poses.shape}')
    center = self.guess_translation(depth=depth, mask=ob_mask, K=K)
    print('center', center)

    poses = torch.as_tensor(poses, device='cuda', dtype=torch.float)
    poses[:,:3,3] = torch.as_tensor(center.reshape(1,3), device='cuda')

    add_errs = self.compute_add_err_to_gt_pose(poses)
    logging.info(f"after viewpdoint, add_errs min:{add_errs.min()}")

    xyz_map = depth2xyzmap(depth, K)
    time1 = time.perf_counter()

    add_random = False

    if zero_depth:
      depth_score = torch.zeros_like(torch.tensor(depth))
    else:
      depth_score = depth



    if first_selection:
      if mask_rgb:
        rgb_mask = np.zeros_like(rgb)
        rgb_mask[ob_mask > 0] = rgb[ob_mask > 0]

      # turn the rgb_mask into a grey_mask
        if pure_color:
          rgb_mask[ob_mask > 0] = 0.7 * np.array([255, 255, 255])
        else:
          pass
      else:
        rgb_mask = rgb

      scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb_mask, depth=depth_score, K=K, ob_in_cams=poses.data.cpu().numpy(),
                                        normal_map=normal_map, mesh_tensors=self.mesh_tensors, glctx=self.glctx,
                                        mesh_diameter=self.diameter, get_vis=self.debug >= 2, mask = ob_mask,rgb_origin = rgb,depth_anything=depth_anything_score,zero_depth = zero_depth,mask_crop = mask_crop)
      if vis is not None:
        imageio.imwrite(f'{self.debug_dir}/vis_score_before_selection.png', vis[0])

      time2 = time.perf_counter()
      print(f"refine time: {time2 - time1}")
      add_errs = self.compute_add_err_to_gt_pose(poses)
      logging.info(f"final, add_errs min:{add_errs.min()}")

      ids = torch.as_tensor(scores).argsort(descending=True)
      logging.info(f'sort ids:{ids}')
      scores = scores[ids]
      poses = poses[ids]

      logging.info(f'sorted scores:{scores}')

      poses = poses[:first_selection_num]

    if self.debug>=5:
      vis_plot = True
    else:
      vis_plot = False
    diameter = self.diameter
    diameter = torch.tensor(diameter, dtype=torch.float32, device='cuda')
    poses, vis, depth_new,relative_depth_rgb, losses= self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors,
                                                                            rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, 
                                                                            xyz_map=xyz_map, glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration, get_vis=self.debug>=2,mask = ob_mask,diff_render_iteration=diff_render_iteration,
                                                                            depth_anything=depth_anything_metric,vis_plot=vis_plot,obj_dia = diameter,inner = inner,
                                                                            depth_anything_score = depth_anything_score,only_edge=only_edge,only_mask = only_mask)
    losses = torch.tensor(losses, device=scores.device, dtype=torch.float)
    
    time2 = time.perf_counter()

    print(f"predict time: {time2 - time1}")

    if add_random:
      angle = 1
      dis = 1 * 10 ** -2
      poses = self.add_random_rotation_translation(poses,angle,dis)


    #save_pose
    if self.debug>=2:

      np.save(f'{self.debug_dir}/poses_{indix}_{special_name_var}.npy', poses.data.cpu().numpy())

    if vis is not None:
      imageio.imwrite(f'{self.debug_dir}/vis_refiner.png', vis)

    if mask_rgb:
      rgb_mask = np.zeros_like(rgb)
      rgb_mask[ob_mask>0] = rgb[ob_mask>0]
      if pure_color:
      #turn the rgb_mask into a grey_mask
        rgb_mask[ob_mask>0] = 0.7 * np.array([255,255,255])
    else:
      rgb_mask = rgb

    if depth_anything_score:
      depth_anything_score = depth_anything_score.to(0).eval()
    # depth = torch.tensor(depth_new)
    scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb_mask, depth=depth_score, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, mesh_tensors=self.mesh_tensors, glctx=self.glctx,
                                       mesh_diameter=self.diameter, get_vis=self.debug>=2, mask = ob_mask,rgb_origin = rgb,depth_anything=depth_anything_score,zero_depth = zero_depth,mask_crop = mask_crop,  losses=losses,)
    if vis is not None:
      imageio.imwrite(f'{self.debug_dir}/vis_score.png', vis[0])
      depth_best = vis[1][0][0]
      depth_best_u16 = (depth_best * 1000).astype(np.uint16)
      dir_name = os.path.join(self.debug_dir, 'depth_best_u16.png')
      cv2.imwrite(dir_name, depth_best_u16)
      # depth_best = depth_best.cpu().numpy()
      depth_best_vis = (depth_best - depth_best.min()) / (depth_best.max() - depth_best.min()) * 255.0
      depth_best_vis = depth_best_vis.astype(np.uint8)
      depth_best_vis = Image.fromarray(depth_best_vis)
      dir_name = os.path.join(self.debug_dir, 'depth_best_vis.png')
      depth_best_vis.save(dir_name)
      mask_best =  vis[1][1]
      # mask_best = mask_best.cpu().numpy()
      dir_name = os.path.join(self.debug_dir, 'mask_best.png')
      cv2.imwrite(dir_name, (mask_best * 255.0).clip(0, 255))
      edge_best = vis[1][2]
      # mask_best = mask_best.cpu().numpy()
      dir_name = os.path.join(self.debug_dir, 'edge_best_with_input.png')
      #write text to indicate the color of the edge
      edge_best = (edge_best * 255.0).clip(0, 255)
      edge_best = cv2.putText(edge_best, 'red: predict edge', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
      edge_best = cv2.putText(edge_best, 'green: input edge', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
      cv2.imwrite(dir_name, edge_best)


    ids = torch.as_tensor(losses).argsort(descending=False)
    logging.info(f'sort ids:{ids}')
    scores = scores[ids]
    poses = poses[ids]
    losses = losses[ids]
    for i in range(len(losses)):
      logging.info(f'score:{scores[i]}, pose:{poses[i]}, loss:{losses[i]}')

    logging.info(f'sorted scores:{scores}')


    best_pose = poses[0]@self.get_tf_to_centered_mesh()
    self.pose_last = poses[0]
    self.best_id = ids[0]

    self.poses = poses
    self.scores = scores
    pose_toberefine = poses[0].unsqueeze(0)


    if self.debug>=2:
      np.save(f'{self.debug_dir}/pose_{indix}_{special_name_var}.npy', best_pose.data.cpu().numpy())
      import matplotlib
      cmap = matplotlib.colormaps.get_cmap('jet')
      if depth_anything_metric and relative_depth_rgb is not None:

        relative_depth_rgb *= 1000
        # relative_depth_rgb_save = relative_depth_rgb.astype(np.uint16)
        # dir_name = os.path.join(self.debug_dir, 'relative_depth_rgb_u16.png')
        # cv2.imwrite(dir_name, relative_depth_rgb_save)
        # relative_depth_rgb = (relative_depth_rgb - relative_depth_rgb.min()) / (
        #         relative_depth_rgb.max() - relative_depth_rgb.min()) * 255.0
        # relative_depth_rgb = relative_depth_rgb.astype(np.uint8)
        # colored_depth = (cmap(relative_depth_rgb)[:, :, :3] * 255).astype(np.uint8)
        #
        # colored_depth = Image.fromarray(colored_depth)
        # dir_name = os.path.join(self.debug_dir, 'relative_depth_rgb.png')
        # colored_depth.save(dir_name)
        # Generate mesh grid and calculate point cloud coordinates
        # width = relative_depth_rgb.shape[1]
        # height = relative_depth_rgb.shape[0]
        # focal_length_x = K[0, 0]
        # focal_length_y = K[1, 1]
        # x, y = np.meshgrid(np.arange(width), np.arange(height))
        # x = (x - width / 2) / focal_length_x
        # y = (y - height / 2) / focal_length_y
        # z = np.array(relative_depth_rgb)
        # points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
        #
        #
        #
        # colors = np.array(rgb).reshape(-1, 3) / 255.0
        #
        # Create the point cloud and save it to the output directory
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # pcd.colors = o3d.utility.Vector3dVector(colors)
        # o3d.io.write_point_cloud(os.path.join(self.debug_dir, 'relative_depth_rgb.ply'), pcd)

    return best_pose.data.cpu().numpy()


  def compute_add_err_to_gt_pose(self, poses):
    '''
    @poses: wrt. the centered mesh
    '''
    return -torch.ones(len(poses), device='cuda', dtype=torch.float)


  def track_one(self, rgb, depth, K, iteration, extra={}):
    if self.pose_last is None:
      logging.info("Please init pose by register first")
      raise RuntimeError
    logging.info("Welcome")

    depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)
    depth = erode_depth(depth, radius=2, device='cuda')
    depth = bilateral_filter_depth(depth, radius=2, device='cuda')
    logging.info("depth processing done")

    xyz_map = depth2xyzmap_batch(depth[None], torch.as_tensor(K, dtype=torch.float, device='cuda')[None], zfar=np.inf)[0]

    pose, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=self.pose_last.reshape(1,4,4).data.cpu().numpy(), normal_map=None, xyz_map=xyz_map, mesh_diameter=self.diameter, glctx=self.glctx, iteration=iteration, get_vis=self.debug>=2)
    logging.info("pose done")
    if self.debug>=2:
      extra['vis'] = vis
    self.pose_last = pose
    return (pose@self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4,4)

  def matching_one(self,rgb, mask, K, iteration= 0, depth_anything = True,diff_render_iteration = 30,extra={}):
    if self.pose_last is None:
      logging.info("Please init pose by register first")
      raise RuntimeError
    logging.info("Welcome")
    if self.debug>=5:
      vis_plot = True
    else:
      vis_plot = False

    depth = np.ones_like(rgb[:,:,0]) * self.pose_last[2,3].cpu().numpy()
    xyz_map = depth2xyzmap(depth, K)
    poses, vis, depth_new,relative_depth_rgb = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=self.pose_last.reshape(1,4,4).data.cpu().numpy(), normal_map=None,mesh_diameter=self.diameter, glctx=self.glctx, iteration=iteration, get_vis=self.debug>=2,mask = mask,diff_render_iteration=diff_render_iteration,depth_anything=depth_anything,obj_dia = self.diameter,xyz_map= xyz_map,vis_plot=vis_plot)

    logging.info("pose done")
    # if self.debug>=2:
    #   extra['vis'] = vis
    pose = poses[0]
    self.pose_last = poses[0]

    return (pose@self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4,4)


class MatchingPose(FoundationPose):

  def __init__(self, model_pts, model_normals, symmetry_tfs=None, mesh=None, scorer:ScorePredictor=None, refiner:PoseRefinePredictor=None, glctx=None, debug=0, debug_dir='/home/bowen/debug/novel_pose_debug/',obj_id = None,img_id = None):
      super(MatchingPose, self).__init__(model_pts, model_normals, symmetry_tfs=symmetry_tfs, mesh=mesh, scorer=scorer, refiner=refiner, glctx=glctx, debug=debug, debug_dir=debug_dir)
      self.img_id = img_id


  def register(self, K, rgb, depth, ob_mask, ob_id=None, glctx=None, refine_iteration=5,diff_render_iteration = 19,diff_render_iteration_batch = 10):
    '''Copmute pose from given pts to self.pcd
    @pts: (N,3) np array, downsampled scene points
    '''
    set_seed(0)
    logging.info('Welcome')

    if self.glctx is None:
      if glctx is None:
        self.glctx = dr.RasterizeCudaContext()
        # self.glctx = dr.RasterizeGLContext()
      else:
        self.glctx = glctx

    depth = erode_depth(depth, radius=2, device='cuda')
    depth = bilateral_filter_depth(depth, radius=2, device='cuda')


    if self.debug>=2:
      xyz_map = depth2xyzmap(depth, K)
      valid = (xyz_map[...,2]>=0.1) & (xyz_map[...,2] < 1.5)
      pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
      o3d.io.write_point_cloud(f'{self.debug_dir}/scene_raw.ply',pcd)
      cv2.imwrite(f'{self.debug_dir}/ob_mask.png', (ob_mask*255.0).clip(0,255))

    normal_map = None
    valid = (depth>=0.1) & (ob_mask>0)
    if valid.sum()<4:
      logging.info(f'valid too small, return')
      pose = np.eye(4)
      pose[:3,3] = self.guess_translation(depth=depth, mask=ob_mask, K=K)
      return pose

    # if self.debug>=2:
    #   imageio.imwrite(f'{self.debug_dir}/color.png', rgb)
    #   cv2.imwrite(f'{self.debug_dir}/depth.png', (depth*1000).astype(np.uint16))
    #   valid = xyz_map[...,2]>=0.1 and xyz_map[...,2] < 2
    #   pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
    #   o3d.io.write_point_cloud(f'{self.debug_dir}/scene_raw.ply',pcd)

    self.H, self.W = depth.shape[:2]
    self.K = K
    self.ob_id = ob_id
    self.ob_mask = ob_mask

    poses = self.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None)
    poses = poses.data.cpu().numpy()
    logging.info(f'poses:{poses.shape}')
    center = self.guess_translation(depth=depth, mask=ob_mask, K=K)

    poses = torch.as_tensor(poses, device='cuda', dtype=torch.float)
    poses[:,:3,3] = torch.as_tensor(center.reshape(1,3), device='cuda')

    add_errs = self.compute_add_err_to_gt_pose(poses)
    logging.info(f"after viewpoint, add_errs min:{add_errs.min()}")

    xyz_map = depth2xyzmap(depth, K)
    poses = self.refiner.matching(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, xyz_map=xyz_map, glctx=self.glctx, mesh_diameter=self.diameter, iteration=refine_iteration, get_vis=self.debug>=2,mask = ob_mask,iteration_diff_rendering=diff_render_iteration,diff_render_iteration_batch = diff_render_iteration_batch)
    # if vis is not None:
    #   imageio.imwrite(f'{self.debug_dir}/vis_refiner.png', vis)
    # if self.debug>=2:
    #   xyz_map_new = copy.deepcopy(xyz_map_new.permute(1,2,0).detach().clone().cpu().numpy())
    #   # plt.imshow(xyz_map_new)
    #   # plt.show()
    #   valid = (xyz_map_new[..., 2] >= 0.1) & (xyz_map_new[..., 2] < 1.5)
    #   pcd = toOpen3dCloud(xyz_map_new[valid], rgb[valid])
    #   o3d.io.write_point_cloud(f'{self.debug_dir}/scene_after_refined.ply',pcd)
    # plt.imshow(depth_new)
    # plt.show()
    # plt.imshow(xyz_map_new[...,2])
    # plt.show()

    # if vis is not None:
    #   imageio.imwrite(f'{self.debug_dir}/vis_score.png', vis)

    add_errs = self.compute_add_err_to_gt_pose(poses)
    logging.info(f"final, add_errs min:{add_errs.min()}")

    ids = 0
    logging.info(f'sort ids:{ids}')
    poses = poses[ids]


    best_pose = poses@self.get_tf_to_centered_mesh()


    self.poses = poses

    return best_pose.data.cpu().numpy()

