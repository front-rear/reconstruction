import os,sys

import cv2

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from estimater import *
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/mycpp/build')
# sys.path.append(f'{code_dir}/../Depth-Anything-V2-main')
logging.getLogger().setLevel(logging.CRITICAL)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# from Depth_Anything_V2_main.depth_anything_v2.dpt import DepthAnythingV2
# from Depth_Anything_V2_main.metric_depth.depth_anything_v2.dpt import DepthAnythingV2 as DepthAnythingV2_metric

def run_pose_estimation():
    dataset = "clearpose"
    wp.force_load(device='cuda')
    set_id = opt.set_id
    scene_id = opt.scene_id
    base_dir = opt.base_dir
    whether_normal = opt.inner
    if whether_normal:
        whether_normal = "normal"
    else:
        whether_normal = "abnormal"

    use_reconstructed_mesh = opt.use_reconstructed_mesh
    debug = opt.debug

    if opt.matching:
        if opt.zero_depth:

            debug_dir = f"./debug/{dataset}_purematching{opt.diff_render_iteration}_refine{opt.refine_iteration}_zerodepth_{opt.special_name}_set{set_id}_scene{scene_id}_mask{opt.mask_type}_{whether_normal}/debug"
        else:
            debug_dir = f"./debug/{dataset}_purematching{opt.diff_render_iteration}_refine{opt.refine_iteration}_inputdepth_{opt.special_name}_set{set_id}_scene{scene_id}_mask{opt.mask_type}_{whether_normal}/debug"
        # np.save(
        #     f"/home/rvsa/disk0/pzh/foundationpose/FoundationPose-main/glassmolder/debug/glassmolder_no_fused_refine{opt.refine_iteration}_",
    else:
        if opt.zero_depth:

            debug_dir = f"./debug/{dataset}_foundationpose_refine{opt.refine_iteration}_zerodepth_{opt.special_name}_set{set_id}_scene{scene_id}_mask{opt.mask_type}_{whether_normal}/debug"
        else:
            debug_dir = f"./debug/{dataset}_foundationpose_refine{opt.refine_iteration}_inputdepth_{opt.special_name}_set{set_id}_scene{scene_id}_mask{opt.mask_type}_{whether_normal}/debug"

    res = NestDict()
    glctx = dr.RasterizeCudaContext()
    mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)).to_mesh()
    est = FoundationPose(model_pts=mesh_tmp.vertices.copy(), model_normals=mesh_tmp.vertex_normals.copy(), symmetry_tfs=None, mesh=mesh_tmp, scorer=None, refiner=None, glctx=glctx, debug=debug,debug_dir=debug_dir,dataset=dataset)

    matching = opt.matching

    # find_all_obj_in_one_scene(reader_tmp)

    if opt.depth_anything_metric:
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

    else:
        depth_anything_metric = None

    if opt.depth_anything_score:
        encoder = 'vitl'
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        depth_anything_score = DepthAnythingV2(**model_configs[encoder])
        depth_anything_score.load_state_dict(
            torch.load(f'../depth_anything_v2/depth_anything_v2_vitl.pth', map_location='cpu'))
        # depth_anything_score = depth_anything_score.to(0).eval()
    else:
        depth_anything_score = None

    # print(f"ob_ids:{reader_tmp.ob_ids}")

    # for ob_id in [objname_2_objid(dataset)["pitcher_1"]]:
    # for ob_id in [22]:
    #     print(f"ob_id:{ob_id}")

    mesh_file = os.path.join(opt.dataset_dir, "objs", opt.state_name, "%s.obj" % opt.state_name)
    x_rotate_180 = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    mesh = trimesh.load(mesh_file, process=False, force="mesh").apply_scale(1 / 6.0).apply_transform(x_rotate_180)

    est.get_obj_id_and_init_refiner(0, matching)
    # reader = ClearposeReader(base_dir = base_dir,set_id=set_id, scene_id=scene_id,mask_type=opt.mask_type)
    est.reset_object(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(), mesh=mesh)

    outs = []
    out = run_pose_estimation_worker(None, est,  "cuda:0",depth_anything_metric,depth_anything_score,opt.indix,opt.special_name_varience)
    outs.append(out)


# noinspection PyPackageRequirements
def run_pose_estimation_worker(reader:ClearposeReader, est:FoundationPose=None,  device='cuda:0',depth_anything_metric = None,depth_anything_score=None,indix = 0,special_name_var = "default"):
    est.to_device(device)
    est.glctx = dr.RasterizeCudaContext(device=device)
    debug = opt.debug
    result = NestDict()
    # set_id = reader.set_id
    # scene_id = reader.get_scene_id()
    if opt.depth_anything_metric:
        depth_anything_metric = depth_anything_metric.to(0).eval()


    color = cv2.imread(os.path.join(opt.dataset_dir, "raw_rgb", "%s.png" % opt.state_name), cv2.IMREAD_UNCHANGED)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(os.path.join(opt.dataset_dir, "raw_depth", "%s.png" % opt.state_name),cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float64) / 1000
    ob_mask = cv2.imread(os.path.join(opt.dataset_dir, "mask", "%s.png" % opt.state_name),cv2.IMREAD_UNCHANGED)
    K_intrinsic = np.array([
        [906.461181640625, 0, 635.8511962890625],
        [0, 905.659912109375, 350.6916809082031],
        [0, 0, 1]
    ], dtype=np.float64)
    debug_dir ="./"
    est.get_debug_dir(debug_dir = debug_dir)
    mask_crop = opt.mask_crop
    pose = est.register(K=K_intrinsic, rgb=color, depth=depth, ob_mask=ob_mask, ob_id=0,zero_depth=opt.zero_depth,iteration=opt.refine_iteration,diff_render_iteration=opt.diff_render_iteration,depth_anything_metric = depth_anything_metric,depth_anything_score = depth_anything_score,indix = indix,special_name_var = special_name_var,mask_rgb = opt.mask_rgb,first_selection = opt.first_selection,pure_color = opt.pure_color,inner = opt.inner,sample_around_gt = opt.sample_around_gt,sample_translation_error = opt.sample_translation_error, sample_rotation_error = opt.sample_rotation_error,mask_crop = mask_crop,first_selection_num = opt.first_selection_num)

    return pose

'''
def find_all_obj_in_one_scene(reader):
    path = f"{reader.base_dir}/set{reader.set_id}/scene{reader.scene_id}/metadata.mat"
    meta_dict = sio.loadmat(path)
    obj_list = []
    for key in meta_dict:
        if key[0] != "_":
            obj_list_indi = meta_dict[key][0][0][0]
            for obj in obj_list_indi:
                obj_list.append(obj)

    obj_list_np = np.array(obj_list)
    obj_list_np = np.unique(obj_list_np)
    obj_list_np_sort = np.sort(obj_list_np)
    reader.ob_ids = obj_list_np_sort

    return obj_list_np_sort
'''

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--linemod_dir', type=str, default="/mnt/disk0/dataset/BOP/lmo", help="linemod root dir")
    parser.add_argument('--use_reconstructed_mesh', type=int, default=0)
    parser.add_argument('--ref_view_dir', type=str, default="/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/bowen_addon/ref_views_16")
    parser.add_argument('--debug', type=int, default=3)
    parser.add_argument('--debug_dir', type=str, default=f'/home/rvsa/disk0/pzh/foundationpose/FoundationPose-main/glassmolder/debug/glassmolder_fused_fused15_matching8/debug')
    parser.add_argument('--diff_render_iteration', type=int, default=300)
    parser.add_argument('--refine_iteration', type=int, default=0)
    parser.add_argument('--matching', action="store_true")
    parser.add_argument('--mask_crop', action="store_true")
    parser.add_argument('--inference', action="store_true")
    parser.add_argument('--inference_no_matching', action="store_true")
    parser.add_argument('--zero_depth', action="store_true")
    parser.add_argument('--depth_anything_metric', action="store_true")
    parser.add_argument('--depth_anything_score', action="store_true")
    parser.add_argument('--inner', action="store_true")
    parser.add_argument('--special_name', type=str, default="default")
    parser.add_argument('--indix', type=int, default=0)
    parser.add_argument('--special_name_varience', type=str, default="default")
    parser.add_argument('--mask_rgb', action="store_true")

    parser.add_argument('--only_edge', action="store_true")
    parser.add_argument('--only_mask', action="store_true")
    parser.add_argument('--first_selection', action="store_true")
    parser.add_argument('--sample_around_gt', action="store_true")
    parser.add_argument('--sample_translation_error', type=float, default=None)
    parser.add_argument('--sample_rotation_error', type=float, default=None)
    parser.add_argument('--pure_color', action="store_true")
    parser.add_argument('--mask_type', type=str, default="gt")
    parser.add_argument('--set_id', type=int, default=7)
    parser.add_argument('--scene_id', type=int, default=1)
    parser.add_argument('--first_selection_num', type=int, default=25)
    parser.add_argument('--base_dir', type=str, default="/mnt/disk0/dataset/clearpose/downsample")

    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--state_name', type=str)
    opt = parser.parse_args()
    # set_seed(0)

    detect_type = 'mask'   # mask / box / detected

    run_pose_estimation()
    # mesh = o3d.io.read_("/mnt/disk0/dataset/clearpose/model/container_5/container_5.obj")
    # o3d.visualization.draw_geometries([mesh])