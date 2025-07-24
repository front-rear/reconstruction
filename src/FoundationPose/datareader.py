# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import cv2
import numpy as np

from Utils import *
import json,os,sys
from ADD_S_activezero import objid_2_objname,objname_2_objid

BOP_LIST = ['lmo','tless','ycbv','hb','tudl','icbin','itodd']
BOP_DIR = '/mnt/disk0/dataset/BOP'

def get_bop_reader(video_dir, zfar=np.inf):
  if 'ycbv' in video_dir or 'YCB' in video_dir:
    return YcbVideoReader(video_dir, zfar=zfar)
  if 'lmo' in video_dir or 'LINEMOD-O' in video_dir:
    return LinemodOcclusionReader(video_dir, zfar=zfar)
  if 'tless' in video_dir or 'TLESS' in video_dir:
    return TlessReader(video_dir, zfar=zfar)
  if 'hb' in video_dir:
    return HomebrewedReader(video_dir, zfar=zfar)
  if 'tudl' in video_dir:
    return TudlReader(video_dir, zfar=zfar)
  if 'icbin' in video_dir:
    return IcbinReader(video_dir, zfar=zfar)
  if 'itodd' in video_dir:
    return ItoddReader(video_dir, zfar=zfar)
  else:
    raise RuntimeError


def get_bop_video_dirs(dataset):
  if dataset=='ycbv':
    video_dirs = sorted(glob.glob(f'{BOP_DIR}/ycbv/test/*'))
  elif dataset=='lmo':
    video_dirs = sorted(glob.glob(f'{BOP_DIR}/lmo/lmo_test_bop19/test/*'))
  elif dataset=='tless':
    video_dirs = sorted(glob.glob(f'{BOP_DIR}/tless/tless_test_primesense_bop19/test_primesense/*'))
  elif dataset=='hb':
    video_dirs = sorted(glob.glob(f'{BOP_DIR}/hb/hb_test_primesense_bop19/test_primesense/*'))
  elif dataset=='tudl':
    video_dirs = sorted(glob.glob(f'{BOP_DIR}/tudl/tudl_test_bop19/test/*'))
  elif dataset=='icbin':
    video_dirs = sorted(glob.glob(f'{BOP_DIR}/icbin/icbin_test_bop19/test/*'))
  elif dataset=='itodd':
    video_dirs = sorted(glob.glob(f'{BOP_DIR}/itodd/itodd_test_bop19/test/*'))
  else:
    raise RuntimeError
  return video_dirs



class YcbineoatReader:
  def __init__(self,video_dir, downscale=1, shorter_side=None, zfar=np.inf):
    self.video_dir = video_dir
    self.downscale = downscale
    self.zfar = zfar
    self.color_files = sorted(glob.glob(f"{self.video_dir}/rgb/*.png"))
    self.K = np.loadtxt(f'{video_dir}/cam_K.txt').reshape(3,3)
    self.id_strs = []
    for color_file in self.color_files:
      id_str = os.path.basename(color_file).replace('.png','')
      self.id_strs.append(id_str)
    self.H,self.W = cv2.imread(self.color_files[0]).shape[:2]

    if shorter_side is not None:
      self.downscale = shorter_side/min(self.H, self.W)

    self.H = int(self.H*self.downscale)
    self.W = int(self.W*self.downscale)
    self.K[:2] *= self.downscale

    self.gt_pose_files = sorted(glob.glob(f'{self.video_dir}/annotated_poses/*'))

    self.videoname_to_object = {
      'bleach0': "021_bleach_cleanser",
      'bleach_hard_00_03_chaitanya': "021_bleach_cleanser",
      'cracker_box_reorient': '003_cracker_box',
      'cracker_box_yalehand0': '003_cracker_box',
      'mustard0': '006_mustard_bottle',
      'mustard_easy_00_02': '006_mustard_bottle',
      'sugar_box1': '004_sugar_box',
      'sugar_box_yalehand0': '004_sugar_box',
      'tomato_soup_can_yalehand0': '005_tomato_soup_can',
    }


  def get_video_name(self):
    return self.video_dir.split('/')[-1]

  def __len__(self):
    return len(self.color_files)

  def get_gt_pose(self,i):
    try:
      pose = np.loadtxt(self.gt_pose_files[i]).reshape(4,4)
      return pose
    except:
      logging.info("GT pose not found, return None")
      return None


  def get_color(self,i):
    color = imageio.imread(self.color_files[i])[...,:3]
    color = cv2.resize(color, (self.W,self.H), interpolation=cv2.INTER_NEAREST)
    return color

  def get_mask(self,i):
    mask = cv2.imread(self.color_files[i].replace('rgb','masks'),-1)
    if len(mask.shape)==3:
      for c in range(3):
        if mask[...,c].sum()>0:
          mask = mask[...,c]
          break
    mask = cv2.resize(mask, (self.W,self.H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
    return mask

  def get_depth(self,i):
    depth = cv2.imread(self.color_files[i].replace('rgb','depth'),-1)/1e3
    depth = cv2.resize(depth, (self.W,self.H), interpolation=cv2.INTER_NEAREST)
    depth[(depth<0.1) | (depth>=self.zfar)] = 0
    return depth


  def get_xyz_map(self,i):
    depth = self.get_depth(i)
    xyz_map = depth2xyzmap(depth, self.K)
    return xyz_map

  def get_occ_mask(self,i):
    hand_mask_file = self.color_files[i].replace('rgb','masks_hand')
    occ_mask = np.zeros((self.H,self.W), dtype=bool)
    if os.path.exists(hand_mask_file):
      occ_mask = occ_mask | (cv2.imread(hand_mask_file,-1)>0)

    right_hand_mask_file = self.color_files[i].replace('rgb','masks_hand_right')
    if os.path.exists(right_hand_mask_file):
      occ_mask = occ_mask | (cv2.imread(right_hand_mask_file,-1)>0)

    occ_mask = cv2.resize(occ_mask, (self.W,self.H), interpolation=cv2.INTER_NEAREST)

    return occ_mask.astype(np.uint8)

  def get_gt_mesh(self):
    ob_name = self.videoname_to_object[self.get_video_name()]
    YCB_VIDEO_DIR = os.getenv('YCB_VIDEO_DIR')
    mesh = trimesh.load(f'{YCB_VIDEO_DIR}/models/{ob_name}/textured_simple.obj')
    return mesh


class BopBaseReader:
  def __init__(self, base_dir, zfar=np.inf, resize=1):
    self.base_dir = base_dir
    self.resize = resize
    self.dataset_name = None
    self.color_files = sorted(glob.glob(f"{self.base_dir}/rgb/*"))
    if len(self.color_files)==0:
      self.color_files = sorted(glob.glob(f"{self.base_dir}/gray/*"))
    self.zfar = zfar

    self.K_table = {}
    with open(f'{self.base_dir}/scene_camera.json','r') as ff:
      info = json.load(ff)
    for k in info:
      self.K_table[f'{int(k):06d}'] = np.array(info[k]['cam_K']).reshape(3,3)
      self.bop_depth_scale = info[k]['depth_scale']

    if os.path.exists(f'{self.base_dir}/scene_gt.json'):
      with open(f'{self.base_dir}/scene_gt.json','r') as ff:
        self.scene_gt = json.load(ff)
      self.scene_gt = copy.deepcopy(self.scene_gt)   # Release file handle to be pickle-able by joblib
      assert len(self.scene_gt)==len(self.color_files)
    else:
      self.scene_gt = None

    self.make_id_strs()


  def make_scene_ob_ids_dict(self):
    with open(f'{BOP_DIR}/{self.dataset_name}/test_targets_bop19.json','r') as ff:
      self.scene_ob_ids_dict = {}
      data = json.load(ff)
      for d in data:
        if d['scene_id']==self.get_video_id():
          id_str = f"{d['im_id']:06d}"
          if id_str not in self.scene_ob_ids_dict:
            self.scene_ob_ids_dict[id_str] = []
          self.scene_ob_ids_dict[id_str] += [d['obj_id']]*d['inst_count']


  def get_K(self, i_frame):
    K = self.K_table[self.id_strs[i_frame]]
    if self.resize!=1:
      K[:2,:2] *= self.resize
    return K


  def get_video_dir(self):
    video_id = int(self.base_dir.rstrip('/').split('/')[-1])
    return video_id

  def make_id_strs(self):
    self.id_strs = []
    for i in range(len(self.color_files)):
      name = os.path.basename(self.color_files[i]).split('.')[0]
      self.id_strs.append(name)


  def get_instance_ids_in_image(self, i_frame:int):
    ob_ids = []
    if self.scene_gt is not None:
      name = int(os.path.basename(self.color_files[i_frame]).split('.')[0])
      for k in self.scene_gt[str(name)]:
        ob_ids.append(k['obj_id'])
    elif self.scene_ob_ids_dict is not None:
      return np.array(self.scene_ob_ids_dict[self.id_strs[i_frame]])
    else:
      mask_dir = os.path.dirname(self.color_files[0]).replace('rgb','mask_visib')
      id_str = self.id_strs[i_frame]
      mask_files = sorted(glob.glob(f'{mask_dir}/{id_str}_*.png'))
      ob_ids = []
      for mask_file in mask_files:
        ob_id = int(os.path.basename(mask_file).split('.')[0].split('_')[1])
        ob_ids.append(ob_id)
    ob_ids = np.asarray(ob_ids)
    return ob_ids


  def get_gt_mesh_file(self, ob_id):
    raise RuntimeError("You should override this")


  def get_color(self,i):
    color = imageio.imread(self.color_files[i])
    if len(color.shape)==2:
      color = np.tile(color[...,None], (1,1,3))  # Gray to RGB
    if self.resize!=1:
      color = cv2.resize(color, fx=self.resize, fy=self.resize, dsize=None)
    return color


  def get_depth(self,i, filled=False):
    if filled:
      depth_file = self.color_files[i].replace('rgb','depth_filled')
      depth_file = f'{os.path.dirname(depth_file)}/0{os.path.basename(depth_file)}'
      depth = cv2.imread(depth_file,-1)/1e3
    else:
      depth_file = self.color_files[i].replace('rgb','depth').replace('gray','depth')
      depth = cv2.imread(depth_file,-1)*1e-3*self.bop_depth_scale
    if self.resize!=1:
      depth = cv2.resize(depth, fx=self.resize, fy=self.resize, dsize=None, interpolation=cv2.INTER_NEAREST)
    depth[depth<0.1] = 0
    depth[depth>self.zfar] = 0
    return depth

  def get_xyz_map(self,i):
    depth = self.get_depth(i)
    xyz_map = depth2xyzmap(depth, self.get_K(i))
    return xyz_map


  def get_mask(self, i_frame:int, ob_id:int, type='mask_visib'):
    '''
    @type: mask_visib (only visible part) / mask (projected mask from whole model)
    '''
    pos = 0
    name = int(os.path.basename(self.color_files[i_frame]).split('.')[0])
    if self.scene_gt is not None:
      for k in self.scene_gt[str(name)]:
        if k['obj_id']==ob_id:
          break
        pos += 1
      mask_file = f'{self.base_dir}/{type}/{name:06d}_{pos:06d}.png'
      if not os.path.exists(mask_file):
        logging.info(f'{mask_file} not found')
        return None
    else:
      # mask_dir = os.path.dirname(self.color_files[0]).replace('rgb',type)
      # mask_file = f'{mask_dir}/{self.id_strs[i_frame]}_{ob_id:06d}.png'
      raise RuntimeError
    mask = cv2.imread(mask_file, -1)
    if self.resize!=1:
      mask = cv2.resize(mask, fx=self.resize, fy=self.resize, dsize=None, interpolation=cv2.INTER_NEAREST)
    return mask>0


  def get_gt_mesh(self, ob_id:int):
    mesh_file = self.get_gt_mesh_file(ob_id)
    mesh = trimesh.load(mesh_file)
    # mesh.show()
    mesh.vertices *= 1e-3
    return mesh


  def get_model_diameter(self, ob_id):
    dir = os.path.dirname(self.get_gt_mesh_file(self.ob_ids[0]))
    info_file = f'{dir}/models_info.json'
    with open(info_file,'r') as ff:
      info = json.load(ff)
    return info[str(ob_id)]['diameter']/1e3



  def get_gt_poses(self, i_frame, ob_id):
    gt_poses = []
    name = int(self.id_strs[i_frame])
    for i_k, k in enumerate(self.scene_gt[str(name)]):
      if k['obj_id']==ob_id:
        cur = np.eye(4)
        cur[:3,:3] = np.array(k['cam_R_m2c']).reshape(3,3)
        cur[:3,3] = np.array(k['cam_t_m2c'])/1e3
        gt_poses.append(cur)
    return np.asarray(gt_poses).reshape(-1,4,4)


  def get_gt_pose(self, i_frame:int, ob_id, mask=None, use_my_correction=False):
    ob_in_cam = np.eye(4)
    best_iou = -np.inf
    best_gt_mask = None
    name = int(self.id_strs[i_frame])
    for i_k, k in enumerate(self.scene_gt[str(name)]):
      if k['obj_id']==ob_id:
        cur = np.eye(4)
        cur[:3,:3] = np.array(k['cam_R_m2c']).reshape(3,3)
        cur[:3,3] = np.array(k['cam_t_m2c'])/1e3
        if mask is not None:  # When multi-instance exists, use mask to determine which one
          gt_mask = cv2.imread(f'{self.base_dir}/mask_visib/{self.id_strs[i_frame]}_{i_k:06d}.png', -1).astype(bool)
          intersect = (gt_mask*mask).astype(bool)
          union = (gt_mask+mask).astype(bool)
          iou = float(intersect.sum())/union.sum()
          if iou>best_iou:
            best_iou = iou
            best_gt_mask = gt_mask
            ob_in_cam = cur
        else:
          ob_in_cam = cur
          break


    if use_my_correction:
      if 'ycb' in self.base_dir.lower() and 'train_real' in self.color_files[i_frame]:
        video_id = self.get_video_id()
        if ob_id==1:
          if video_id in [12,13,14,17,24]:
            ob_in_cam = ob_in_cam@self.symmetry_tfs[ob_id][1]
    return ob_in_cam


  def load_symmetry_tfs(self):
    dir = "/mnt/disk0/dataset/BOP/ycbv/ycbv_models/models"
    print(self.get_gt_mesh_file(self.ob_ids[0]))
    info_file = f'{dir}/models_info.json'
    with open(info_file,'r') as ff:
      info = json.load(ff)
    self.symmetry_tfs = {}
    self.symmetry_info_table = {}
    for ob_id in self.ob_ids:
      self.symmetry_info_table[ob_id] = info[str(ob_id)]
      self.symmetry_tfs[ob_id] = symmetry_tfs_from_info(info[str(ob_id)], rot_angle_discrete=5)
    self.geometry_symmetry_info_table = copy.deepcopy(self.symmetry_info_table)


  def get_video_id(self):
    return int(self.base_dir.split('/')[-1])



class LinemodOcclusionReader(BopBaseReader):
  def __init__(self,base_dir='/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/LINEMOD-O/lmo_test_all/test/000002', zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'lmo'
    self.K = list(self.K_table.values())[0]
    self.ob_ids = [1,5,6,8,9,10,11,12]
    self.dataset = "linemodocc"
    self.ob_id_to_names = {
      1: 'ape',
      2: 'benchvise',
      3: 'bowl',
      4: 'camera',
      5: 'water_pour',
      6: 'cat',
      7: 'cup',
      8: 'driller',
      9: 'duck',
      10: 'eggbox',
      11: 'glue',
      12: 'holepuncher',
      13: 'iron',
      14: 'lamp',
      15: 'phone',
    }
    self.load_symmetry_tfs()

  def get_gt_mesh_file(self, ob_id):
    mesh_dir = f'{BOP_DIR}/{self.dataset_name}/{self.dataset_name}_models/models/obj_{ob_id:06d}.ply'
    return mesh_dir

class LinemodReader(LinemodOcclusionReader):
  def __init__(self, base_dir='/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/LINEMOD/lm_test_all/test/000001', zfar=np.inf, split=None):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'lm'
    if split is not None:  # train/test
      with open(f'/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/LINEMOD/Linemod_preprocessed/data/{self.get_video_id():02d}/{split}.txt','r') as ff:
        lines = ff.read().splitlines()
      self.color_files = []
      for line in lines:
        id = int(line)
        self.color_files.append(f'{self.base_dir}/rgb/{id:06d}.png')
      self.make_id_strs()

    self.ob_ids = np.setdiff1d(np.arange(1,16), np.array([7,3])).tolist()  # Exclude bowl and mug
    self.load_symmetry_tfs()


  def get_gt_mesh_file(self, ob_id):
    root = self.base_dir
    while 1:
      if os.path.exists(f'{root}/lm_models'):
        mesh_dir = f'{root}/lm_models/models/obj_{ob_id:06d}.ply'
        break
      else:
        root = os.path.abspath(f'{root}/../')
    return mesh_dir



  def get_reconstructed_mesh(self, ob_id, ref_view_dir):
    mesh = trimesh.load(os.path.abspath(f'{ref_view_dir}/ob_{ob_id:07d}/model/model.obj'))
    return mesh
import csv

class ActivezeroReader():
    def __init__(self,base_dir = "/mnt/disk0/dataset/rand_scenes"):
        self.base_dir = base_dir
        self.dataset_name = "activezero"
        self.K = None
        self.color_files_path = []
        with open("/mnt/disk0/dataset/rand_scenes/list_file_rand_second.txt", "r") as f:
            prefix = [line.strip() for line in f]
        self.img_name_2_img_id = {img_name:i for i,img_name in enumerate(prefix)}
        self.img_id_2_img_name = {i: img_name for i, img_name in enumerate(prefix)}
        # print(prefix)
        self.color_files_path = prefix
        self.ob_ids = [i for i in range(18)][1:]
        OBJECT_INFO = ["beer_can","camera","cellphone","champagne","coca_cola","coffee_cup","coke_bottle","gold_ball","hammer","jack_daniels","pepsi_bottle","rubik","sharpener","spellegrino","steel_ball","tennis_ball","voss"]


        self.objid_2_objname = {i: _ for i, _ in enumerate(OBJECT_INFO)}
        self.objname_2_objid = {_:i  for i, _ in enumerate(OBJECT_INFO)}
    def get_depth(self,img_name):
        print(f"{self.base_dir}/{img_name}/1024_depth_real.png")
        depth = cv2.imread(f"{self.base_dir}/{img_name}/1024_depth_real.png",cv2.IMREAD_UNCHANGED) / 10**3
        depth = cv2.resize(depth, dsize=(640, 360), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
        return depth

    def get_color(self,img_name):
        color = cv2.imread(f"{self.base_dir}/{img_name}/1024_rgb_real.png",cv2.IMREAD_UNCHANGED)
        color = cv2.resize(color, dsize=(640, 360), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
        color = cv2.cvtColor(color,cv2.COLOR_BGR2RGB)

        return color
    def get_gt_mesh_file(self, ob_id):

        mesh_file = os.path.abspath(f'/mnt/disk0/dataset/bbox_norm/models/{self.objid_2_objname[ob_id]}/visual_mesh.obj')
        # print(self.objid_2_objname[ob_id])

        return mesh_file
    def get_video_id(self):
        return 0

    def get_all_obj(self,img_name):
        mask_all = cv2.imread(f"{self.base_dir}/{img_name}/label.png",
                              cv2.IMREAD_UNCHANGED)
        obj_list = np.unique(mask_all)
        obj_list = obj_list[obj_list != 17]
        obj_list = obj_list[obj_list != 18]
        obj_list = obj_list[obj_list != 19]
        # print(obj_list)
        # print(img_name)
        # cv2.imshow("name",mask_all)
        # cv2.waitKey()
        return obj_list
    def get_mask(self,img_name,obj_id):
        mask_all = cv2.imread(f"{self.base_dir}/{img_name}/label.png",
                           cv2.IMREAD_UNCHANGED)
        mask_all =  cv2.resize(mask_all,dsize=(640,360),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)

        zeros = np.zeros_like(mask_all)
        ones = np.ones_like(mask_all)
        mask_indi = np.where(mask_all == obj_id, ones,zeros)

        return mask_indi
    def get_gt_mesh(self, obj_id):

        mesh_file = self.get_gt_mesh_file(obj_id)
        # mesh = o3d.io.read_triangle_mesh(mesh_fil?
        mesh = trimesh.load(mesh_file, process=False,force="mesh")
        # mesh.show()


        return mesh
    def get_K(self,img_name):

        path = f"{self.base_dir}/{img_name}/meta.pkl"
        meta_dict = np.load(path,allow_pickle=True)
        # print(meta_dict)
        K = meta_dict["intrinsic"]
        K[:2,:] /= 3
        K = K[:3,:3]
        # print(K)
        return K

from ADD_S import objid_2_objname, objname_2_objid
class GlassMolderReader():
  def __init__(self, base_dir="/mnt/disk0/dataset/transtouch_pc2_hcp/dataset_render_0_to_20_hcp"):
    self.base_dir = base_dir
    self.dataset_name = "glassmolder"
    self.K = None
    self.color_files_path = []
    with open("/mnt/disk0/dataset/transtouch_pc2_hcp/split_file.txt", "r") as f:
      prefix = [line.strip() for line in f]
    self.img_name_2_img_id = {img_name: i for i, img_name in enumerate(prefix)}
    self.img_id_2_img_name = {i: img_name for i, img_name in enumerate(prefix)}
    # print(prefix)
    self.color_files_path = prefix
    self.ob_ids = [i for i in range(24)][1:]
    self.objid_2_objname = objid_2_objname("glassmolder")
    self.objname_2_objid = objname_2_objid("glassmolder")

  def get_depth(self, img_name):
    print(f"{self.base_dir}/{img_name}/depth_rgb_emitter.png")
    depth = cv2.imread(f"{self.base_dir}/{img_name}/depth_rgb_emitter.png", cv2.IMREAD_UNCHANGED) / 10 ** 3
    depth = cv2.resize(depth, dsize=(640, 360), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    return depth

  def get_color(self, img_name):
    color = cv2.imread(f"{self.base_dir}/{img_name}/rgb.png", cv2.IMREAD_UNCHANGED)
    color = cv2.resize(color, dsize=(640, 360), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    return color

  def get_gt_mesh_file(self, ob_id):

    mesh_file = os.path.abspath(f'/mnt/disk0/dataset/transtouch_pc2_2/model_obj/obj_file/{self.objid_2_objname[ob_id]}.obj')
    # print(self.objid_2_objname[ob_id])

    return mesh_file

  def get_video_id(self):
    return 0

  def get_all_obj(self, img_name):
    mask_all = cv2.imread(f"{self.base_dir}/{img_name}/label_rgb.png",
                          cv2.IMREAD_UNCHANGED)
    obj_list = np.unique(mask_all)
    obj_list = obj_list[obj_list != 127]
    obj_list = obj_list[obj_list != 255]

    # print(obj_list)
    # print(img_name)
    # cv2.imshow("name",mask_all)
    # cv2.waitKey()
    return obj_list

  def get_mask(self, img_name, obj_id):
    mask_all = cv2.imread(f"{self.base_dir}/{img_name}/label_rgb.png",
                          cv2.IMREAD_UNCHANGED)
    mask_all = cv2.resize(mask_all, dsize=(640, 360), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    # cv2.imshow("name",mask_all)
    # cv2.waitKey()

    zeros = np.zeros_like(mask_all)
    ones = np.ones_like(mask_all)
    mask_indi = np.where(mask_all == obj_id, ones, zeros)

    return mask_indi

  def get_gt_mesh(self, obj_id):
    mesh_file = self.get_gt_mesh_file(obj_id)
    # mesh = o3d.io.read_triangle_mesh(mesh_fil?
    mesh = trimesh.load(mesh_file, process=False, force="mesh")
    # mesh.show()

    return mesh

  def get_K(self, img_name):
    path = f"{self.base_dir}/{img_name}/meta.pkl"
    meta_dict = np.load(path, allow_pickle=True)
    # print(meta_dict)
    K = meta_dict["intrinsic_rgb"][:3,:3].astype(np.float32)
    K[:2, :] /= 2
    # print(K)
    return K
import scipy.io as sio
class ClearposeReader():
  def __init__(self, base_dir="/mnt/disk0/dataset/clearpose/downsample",set_id=0,scene_id=0,mask_type = "mask_gt"):
    self.base_dir = base_dir
    self.dataset = "clearpose"
    self.K = None
    self.ob_ids = [i for i in range(64)][1:]
    self.objid_2_objname = objid_2_objname(self.dataset,base_dir=base_dir)
    self.objname_2_objid = objname_2_objid(self.dataset,base_dir=base_dir)
    self.set_id = set_id
    self.scene_id = scene_id
    self.color_files = sorted(glob.glob(f"{self.base_dir}/set{self.set_id}/scene{self.scene_id}/*-color.png"))
    self.color_files_id = [int(os.path.basename(color_file).split("-")[0]) for color_file in self.color_files]
    self.mask_type = mask_type

  def get_depth(self, img_id):
    # print(f"{self.base_dir}/set{set_id}/scene{scene_id}/{img_id:06d}-depth.png")
    depth = cv2.imread(f"{self.base_dir}/set{self.set_id}/scene{self.scene_id}/{img_id:06d}-depth.png", cv2.IMREAD_UNCHANGED) / 10 ** 3

    # depth = cv2.resize(depth, dsize=(640, 360), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    return depth

  def get_color(self, img_id):
    color = cv2.imread(f"{self.base_dir}/set{self.set_id}/scene{self.scene_id}/{img_id:06d}-color.png", cv2.IMREAD_UNCHANGED)
    # color = cv2.resize(color, dsize=(640, 360), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    return color

  def get_gt_mesh_file(self, ob_id):

    mesh_file = os.path.abspath(f'{self.base_dir}/model/{self.objid_2_objname[ob_id]}/{self.objid_2_objname[ob_id]}.obj')
    # print(self.objid_2_objname[ob_id])

    return mesh_file

  def get_scene_id(self):

    return self.scene_id

  def get_all_obj(self, img_name):
    mask_all = cv2.imread(f"{self.base_dir}/set{self.set_id}/scene{self.scene_id}/{img_id:06d}-label.png",
                          cv2.IMREAD_UNCHANGED)
    obj_list = np.unique(mask_all)

    return obj_list

  def get_mask(self, img_id, obj_id):
    if self.mask_type == "gt":
        mask_all = cv2.imread(f"{self.base_dir}/set{self.set_id}/scene{self.scene_id}/{img_id:06d}-{obj_id:06d}_gt.png",
                              cv2.IMREAD_UNCHANGED)
    elif self.mask_type == "sam2":
        mask_all = cv2.imread(f"{self.base_dir}/set{self.set_id}/scene{self.scene_id}/{img_id:06d}_{obj_id:06d}_sam2.png",
                                cv2.IMREAD_UNCHANGED)
    elif self.mask_type == "sam2_noise":
        mask_all = cv2.imread(f"{self.base_dir}/set{self.set_id}/scene{self.scene_id}/{img_id:06d}_{obj_id:06d}_sam2_noise.png",
                              cv2.IMREAD_UNCHANGED)
    else:
      mask_all = cv2.imread(
        f"{self.base_dir}/set{self.set_id}/scene{self.scene_id}/{img_id:06d}_{obj_id:06d}_{self.mask_type}.png",
        cv2.IMREAD_UNCHANGED)

    zeros = np.zeros_like(mask_all)
    ones = np.ones_like(mask_all)
    mask_indi = np.where(mask_all == 255, ones, zeros)
    if mask_indi.sum() == 0:
      return None
    return mask_indi

  def get_gt_mesh(self, obj_id):
    mesh_file = self.get_gt_mesh_file(obj_id)
    # mesh = o3d.io.read_triangle_mesh(mesh_fil?
    mesh = trimesh.load(mesh_file, process=False, force="mesh")
    # mesh.show()

    return mesh

  def get_K(self,  img_id):
    path = f"{self.base_dir}/set{self.set_id}/scene{self.scene_id}/metadata.mat"
    meta_dict = sio.loadmat(path)
    # print(meta_dict)
    img_id = f"{img_id:06d}"
    K = meta_dict[img_id][0][0][3][:3,:3].astype(np.float64)
    # K[:2, :] /= 2
    # print(K)
    return K

class TransposeReader():
  def __init__(self, base_dir="/mnt/disk0/dataset/TRansPose/test",set_id=0,scene_id=0,mask_type="mask_sam2"):
    self.base_dir = base_dir
    self.dataset = "transpose"
    self.K = None
    self.ob_ids = [i for i in range(122)][1:]
    self.objid_2_objname = objid_2_objname(self.dataset,base_dir=self.base_dir)
    self.objname_2_objid = objname_2_objid(self.dataset,base_dir=self.base_dir)
    self.set_id = set_id
    self.scene_id = scene_id
    self.color_files = sorted(glob.glob(f"{self.base_dir}/seq_test_{self.scene_id:02d}/sequences/seq_test_{self.scene_id:02d}/cam_R/rgb/*.png"))
    self.color_files_id = [int(os.path.basename(color_file).split(".")[0]) for color_file in self.color_files]
    self.mask_type = mask_type



  def get_depth(self, img_id):
    # print(f"{self.base_dir}/set{set_id}/scene{scene_id}/{img_id:06d}-depth.png")
    "/mnt/disk0/dataset/TRansPose/test/seq_test_01/sequences/seq_test_01/cam_R/depth/raw"
    depth = cv2.imread(f"{self.base_dir}/seq_test_{self.scene_id:02d}/sequences/seq_test_{self.scene_id:02d}/cam_R/depth/raw/{img_id:06d}.png", cv2.IMREAD_UNCHANGED) / 10 ** 3
    # depth = cv2.resize(depth, dsize=(640, 360), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    return depth

  def get_color(self, img_id):
    "/mnt/disk0/dataset/TRansPose/test/seq_test_01/sequences/seq_test_01/cam_L/rgb"
    color = cv2.imread(f"{self.base_dir}/seq_test_{self.scene_id:02d}/sequences/seq_test_{self.scene_id:02d}/cam_R/rgb/{img_id:06d}.png", cv2.IMREAD_UNCHANGED)
    # color = cv2.resize(color, dsize=(640, 360), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    return color

  def get_gt_mesh_file(self, ob_id):
    mesh_file = os.path.abspath(f'{self.base_dir}/models/{self.objid_2_objname[ob_id]}/{self.objid_2_objname[ob_id]}.obj')
    # print(self.objid_2_objname[ob_id])

    return mesh_file

  def get_video_id(self):

    return self.set_id

  def get_scene_id(self):

    return self.scene_id

  def get_all_obj(self, img_name):
    import json
    obj_all = f"{self.base_dir}/test/seq_test_{self.scene_id:02d}/sequences/seq_test_{self.scene_id:02d}/cam_R/camera_info.json"
    with open(obj_all, "r") as f:
        obj_all = json.load(f)

    obj_name_list = obj_all.keys()
    obj_id_list = np.array([obj_all[obj_name] for obj_name in obj_name_list])
    obj_id_list = np.unique(obj_id_list)
    obj_id_narry_sorted = np.sort(obj_id_list)


    return obj_id_narry_sorted

  def get_mask(self, img_id, obj_index):
    "/mnt/disk0/dataset/TRansPose/test/seq_test_01/sequences/seq_test_01/cam_L/rgb"

    pose_json = f"{self.base_dir}/seq_test_{self.scene_id:02d}/sequences/seq_test_{self.scene_id:02d}/cam_R/pose/{img_id:06d}.json"

    with open(pose_json, "r") as f:
        pose_dict = json.load(f)

    #find the key of a specific value
    obj_id = pose_dict[str(obj_index)]["obj_id"]
    obj_inddex_num = int(obj_index)
    if self.mask_type == "gt":

      mask_all = cv2.imread(f"{self.base_dir}/seq_test_{self.scene_id:02d}/sequences/seq_test_{self.scene_id:02d}/cam_R/mask/{img_id:06d}_{obj_inddex_num:06d}.png",
                            cv2.IMREAD_UNCHANGED)
    elif self.mask_type == "sam2":
        mask_all = cv2.imread(f"{self.base_dir}/seq_test_{self.scene_id:02d}/sequences/seq_test_{self.scene_id:02d}/cam_R/mask_{self.mask_type}/{img_id:06d}_{obj_inddex_num:06d}.png",
                                cv2.IMREAD_UNCHANGED)
    elif self.mask_type == "sam2_noise":
        mask_all = cv2.imread(f"{self.base_dir}/seq_test_{self.scene_id:02d}/sequences/seq_test_{self.scene_id:02d}/cam_R/mask_{self.mask_type}/{img_id:06d}_{obj_inddex_num:06d}.png",
                              cv2.IMREAD_UNCHANGED)
    elif self.mask_type == "sam2_woholes":
        mask_all = cv2.imread(f"{self.base_dir}/seq_test_{self.scene_id:02d}/sequences/seq_test_{self.scene_id:02d}/cam_R/mask_{self.mask_type}/{img_id:06d}_{obj_inddex_num:06d}.png",
                              cv2.IMREAD_UNCHANGED)
    elif self.mask_type == "sam2_noise_woholes":
        mask_all = cv2.imread(f"{self.base_dir}/seq_test_{self.scene_id:02d}/sequences/seq_test_{self.scene_id:02d}/cam_R/mask_{self.mask_type}/{img_id:06d}_{obj_inddex_num:06d}.png",
                              cv2.IMREAD_UNCHANGED)
    else:
        raise RuntimeError
    zeros = np.zeros_like(mask_all)
    ones = np.ones_like(mask_all)
    mask_indi = np.where(mask_all > 0, ones, zeros)
    if len(mask_indi.shape) == 2:
        mask_indi = mask_indi[:, :, None]
    mask_indi = mask_indi[:,:,0] * 255
    if mask_indi.sum() == 0:
      return None
    return mask_indi

  def get_gt_mesh(self, obj_id):
    mesh_file = self.get_gt_mesh_file(obj_id)
    # mesh = o3d.io.read_triangle_mesh(mesh_fil?
    mesh = trimesh.load(mesh_file, process=False, force="mesh")
    # mesh.show()

    return mesh

  def get_K(self,  img_id):
    "/mnt/disk0/dataset/TRansPose/test/seq_test_01/sequences/seq_test_01/cam_R/camera_info.json"
    path = f"{self.base_dir}/seq_test_{self.scene_id:02d}/sequences/seq_test_{self.scene_id:02d}/cam_R/camera_info.json"
    import json
    with open(path, "r") as f:
        K_json = json.load(f)
    "intrinsic: [603.7764783730764, 0.0, 329.259404556074, 0.0, 604.631625241128, 246.484798070861, 0.0, 0.0, 1.0]"
    K_json = np.array(K_json["intrinsic"])
    K = np.zeros((3, 3))
    K[0,0] = K_json[0]
    K[0,2] = K_json[2]
    K[1,1] = K_json[4]
    K[1,2] = K_json[5]
    K[2,2] = 1

    # print(meta_dict)
    # K[:2, :] /= 2
    # print(K)
    return K




class Housecat6dReader():
  def __init__(self, base_dir="/mnt/disk0/dataset/housecat6d/test_scene", scene_id=0):
    self.base_dir = base_dir
    self.dataset = "housecat6d"
    self.K = None
    self.ob_ids = [i for i in range(64)][1:]
    self.set_id = 0
    self.scene_id = scene_id

    self.meta_path =f'/mnt/disk0/dataset/housecat6d/test_scene/test_scene{scene_id}/meta.txt'
    with open(self.meta_path, 'r') as f:
      self.meta = f.readlines()
    self.obj_id_list = [int(obj_info.split(" ")[1]) for obj_info in self.meta]
    self.obj_name_list = [str(obj_info.split(" ")[2].replace('\n', '')) for obj_info in self.meta]
    self.seg_id_list = [i for i in range(len(self.obj_id_list))]

    self.color_files = sorted(glob.glob(f"{self.base_dir}/test_scene{self.scene_id}/rgb/*.png"))
    self.color_files_id = [int(os.path.basename(color_file).split('.')[0]) for color_file in self.color_files]

  def get_scene_id(self):
    return self.scene_id
  def get_gt_mesh(self, obj_name):
      obj_cat = obj_name.split("-")[0]
      obj_file_path = f"/mnt/disk0/dataset/housecat6d/obj_models/obj_models_small_size_final/{obj_cat}/{obj_name}.obj"

      mesh = trimesh.load(obj_file_path, process=False, force="mesh")
      return mesh

  def get_depth(self, img_id):
    # print(f"{self.base_dir}/set{set_id}/scene{scene_id}/{img_id:06d}-depth.png")
    depth = cv2.imread(f"{self.base_dir}/test_scene{self.scene_id}/depth/{img_id:06d}.png",
                       cv2.IMREAD_UNCHANGED)
    # depth = cv2.resize(depth, dsize=(640, 360), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    depth = depth[15:837,:].astype(np.float32)

    depth /= 1000

    # resize the image to 640,480
    depth = cv2.resize(depth, dsize=(640, 480), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
    return depth

  def get_color(self, img_id):
    color = cv2.imread(f"{self.base_dir}/test_scene{self.scene_id}/rgb/{img_id:06d}.png",
                       cv2.IMREAD_UNCHANGED)
    # color = cv2.resize(color, dsize=(640, 360), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    #cut the image(1096,852) to 1096,822 with the center of the image(1096,852)
    color = color[15:837,:,:]
    #resize the image to 640,480
    color = cv2.resize(color, dsize=(640, 480), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)


    return color

  def get_mask(self, img_id, obj_name,mask_type="mask_sam2_noise"):

    mask = cv2.imread(f"{self.base_dir}/test_scene{self.scene_id}/{mask_type}/{img_id:06d}_{obj_name}.png",
                       cv2.IMREAD_UNCHANGED)
    mask[mask>0] = 1
    mask = mask[15:837,:]
    # resize the image to 640,480
    mask = cv2.resize(mask, dsize=(640, 480), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)

    return mask

  def get_K(self, scene_id):
    path = f"{self.base_dir}/test_scene{scene_id}/intrinsics.txt"
    with open(path, "r") as f:
        lines = f.readlines()
        K = np.zeros((3, 3))
        for i, line in enumerate(lines):
            line = line.strip().split(" ")
            for j, value in enumerate(line):
                K[i, j] = float(value)

    K[1,2] -= 15
    K[:2,:3] *= 640 / 1096

    return K


class YcbVideoReader(BopBaseReader):
  def __init__(self, base_dir, zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'ycbv'
    self.K = list(self.K_table.values())[0]
    self.dataset = "ycbv"
    self.make_id_strs()

    self.ob_ids = np.arange(1,22).astype(int).tolist()
    YCB_VIDEO_DIR = os.getenv('YCB_VIDEO_DIR')
    names = sorted(os.listdir(f'/mnt/disk0/dataset/BOP/ycbv/ycbv_models/models/'))
    self.ob_id_to_names = {}
    self.name_to_ob_id = {}
    for i,ob_id in enumerate(self.ob_ids):
      self.ob_id_to_names[ob_id] = names[i]
      self.name_to_ob_id[names[i]] = ob_id

    if 'BOP' not in self.base_dir:
      with open(f'{self.base_dir}/../../keyframe.txt','r') as ff:
        self.keyframe_lines = ff.read().splitlines()



    self.load_symmetry_tfs()
    for ob_id in self.ob_ids:
      if ob_id in [1,4,6,18]:   # Cylinder
        self.geometry_symmetry_info_table[ob_id] = {
          'symmetries_continuous': [
            {'axis':[0,0,1], 'offset':[0,0,0]},
          ],
          'symmetries_discrete': euler_matrix(0, np.pi, 0).reshape(1,4,4).tolist(),
        }
      elif ob_id in [13]:
        self.geometry_symmetry_info_table[ob_id] = {
          'symmetries_continuous': [
            {'axis':[0,0,1], 'offset':[0,0,0]},
          ],
        }
      elif ob_id in [2,3,9,21]:   # Rectangle box
        tfs = []
        for rz in [0, np.pi]:
          for rx in [0,np.pi]:
            for ry in [0,np.pi]:
              tfs.append(euler_matrix(rx, ry, rz))
        self.geometry_symmetry_info_table[ob_id] = {
          'symmetries_discrete': np.asarray(tfs).reshape(-1,4,4).tolist(),
        }
      else:
        pass

  def get_gt_mesh_file(self, ob_id):
    if 'BOP' in self.base_dir:
      mesh_file = os.path.abspath(f'{self.base_dir}/../../ycbv_models/models/obj_{ob_id:06d}.ply')
    else:
      mesh_file = f'{self.base_dir}/....//../ycbv_models/models/obj_{ob_id:06d}.ply'
    return mesh_file


  def get_gt_mesh(self, ob_id:int, get_posecnn_version=False):
    if get_posecnn_version:
      YCB_VIDEO_DIR = "/mnt/disk0/dataset/BOP/ycbv/ycbv_models"
      mesh = trimesh.load(f'{YCB_VIDEO_DIR}/models/{self.ob_id_to_names[ob_id]}/textured_simple.obj')
      return mesh
    mesh_file = self.get_gt_mesh_file(ob_id)
    mesh = trimesh.load(mesh_file, process=False)
    mesh.vertices *= 1e-3
    tex_file = mesh_file.replace('.ply','.png')
    if os.path.exists(tex_file):
      from PIL import Image
      im = Image.open(tex_file)
      uv = mesh.visual.uv
      material = trimesh.visual.texture.SimpleMaterial(image=im)
      color_visuals = trimesh.visual.TextureVisuals(uv=uv, image=im, material=material)
      mesh.visual = color_visuals
    return mesh


  def get_reconstructed_mesh(self, ob_id, ref_view_dir):
    mesh = trimesh.load(os.path.abspath(f'{ref_view_dir}/ob_{ob_id:07d}/model/model.obj'))
    return mesh


  def get_transform_reconstructed_to_gt_model(self, ob_id):
    out = np.eye(4)
    return out


  def get_visible_cloud(self, ob_id):
    file = os.path.abspath(f'{self.base_dir}/../../models/{self.ob_id_to_names[ob_id]}/visible_cloud.ply')
    pcd = o3d.io.read_point_cloud(file)
    return pcd


  def is_keyframe(self, i):
    color_file = self.color_files[i]
    video_id = self.get_video_id()
    frame_id = int(os.path.basename(color_file).split('.')[0])
    key = f'{video_id:04d}/{frame_id:06d}'
    return (key in self.keyframe_lines)



class TlessReader(BopBaseReader):
  def __init__(self, base_dir, zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'tless'

    self.ob_ids = np.arange(1,31).astype(int).tolist()
    self.load_symmetry_tfs()


  def get_gt_mesh_file(self, ob_id):
    mesh_file = f'{self.base_dir}/../../../models_cad/obj_{ob_id:06d}.ply'
    return mesh_file


  def get_gt_mesh(self, ob_id):
    mesh = trimesh.load(self.get_gt_mesh_file(ob_id))
    mesh.vertices *= 1e-3
    mesh = trimesh_add_pure_colored_texture(mesh, color=np.ones((3))*200)
    return mesh


class HomebrewedReader(BopBaseReader):
  def __init__(self, base_dir, zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'hb'
    self.ob_ids = np.arange(1,34).astype(int).tolist()
    self.load_symmetry_tfs()
    self.make_scene_ob_ids_dict()


  def get_gt_mesh_file(self, ob_id):
    mesh_file = f'{self.base_dir}/../../../hb_models/models/obj_{ob_id:06d}.ply'
    return mesh_file


  def get_gt_pose(self, i_frame:int, ob_id, use_my_correction=False):
    logging.info("WARN HomeBrewed doesn't have GT pose")
    return np.eye(4)



class ItoddReader(BopBaseReader):
  def __init__(self, base_dir, zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'itodd'
    self.make_id_strs()

    self.ob_ids = np.arange(1,29).astype(int).tolist()
    self.load_symmetry_tfs()
    self.make_scene_ob_ids_dict()


  def get_gt_mesh_file(self, ob_id):
    mesh_file = f'{self.base_dir}/../../../itodd_models/models/obj_{ob_id:06d}.ply'
    return mesh_file


class IcbinReader(BopBaseReader):
  def __init__(self, base_dir, zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'icbin'
    self.ob_ids = np.arange(1,3).astype(int).tolist()
    self.load_symmetry_tfs()

  def get_gt_mesh_file(self, ob_id):
    mesh_file = f'{self.base_dir}/../../../icbin_models/models/obj_{ob_id:06d}.ply'
    return mesh_file


class TudlReader(BopBaseReader):
  def __init__(self, base_dir, zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'tudl'
    self.ob_ids = np.arange(1,4).astype(int).tolist()
    self.load_symmetry_tfs()

  def get_gt_mesh_file(self, ob_id):
    mesh_file = f'{self.base_dir}/../../../tudl_models/models/obj_{ob_id:06d}.ply'
    return mesh_file


