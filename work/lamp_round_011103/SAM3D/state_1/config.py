weight = None
resume = False
evaluate = True
test_only = False
seed = 39622499
save_path = '/home/rvsa/gary318/build_kinematic/input_rgbd/lamp_round_011103/SAM3D/state_1'
num_worker = 1
batch_size = 1
batch_size_val = None
batch_size_test = None
epoch = 500
eval_epoch = 500
sync_bn = False
enable_amp = True
empty_cache = False
find_unused_parameters = False
mix_prob = 0
param_dicts = None
hooks = [
    dict(type='CheckpointLoader'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='CheckpointSaver', save_freq=None)
]
train = dict(type='DefaultTrainer')
test = dict(type='SemSegTester', verbose=True)
model = dict(
    type='SAMPart3D',
    backbone_dim=384,
    output_dim=384,
    pcd_feat_dim=9,
    freeze_backbone=True,
    max_grouping_scale=2,
    use_hierarchy_losses=True,
    backbone=dict(
        type='PTv3-obj',
        in_channels=9,
        order=['z', 'z-trans', 'hilbert', 'hilbert-trans'],
        stride=(),
        enc_depths=(3, 3, 3, 6, 16),
        enc_channels=(32, 64, 128, 256, 384),
        enc_num_head=(2, 4, 8, 16, 24),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        shuffle_orders=False,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False))
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=1e-08)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.0001],
    pct_start=0.1,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=10.0)
dataset_type = 'SAMPart3DDataset16Views'
data_root = '/home/rvsa/gary318/build_kinematic/input_rgbd/lamp_round_011103/render'
mesh_root = '/home/rvsa/gary318/build_kinematic/input_rgbd/lamp_round_011103/mesh'
backbone_weight_path = '/home/rvsa/gary318/SAMPart3D/ckpt/ptv3-object.pth'
val_scales_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
mesh_voting = True
data = dict(
    train=dict(
        type='SAMPart3DDataset16Views',
        split='train',
        data_root=
        '/home/rvsa/gary318/build_kinematic/input_rgbd/lamp_round_011103/render',
        mesh_root=
        '/home/rvsa/gary318/build_kinematic/input_rgbd/lamp_round_011103/mesh',
        sample_num=15000,
        pixels_per_image=256,
        batch_size=90,
        extent_scale=10.0,
        transform=[
            dict(type='NormalizeCoord'),
            dict(type='CenterShift', apply_z=True),
            dict(
                type='GridSample',
                grid_size=0.01,
                keys=('coord', 'color', 'normal', 'origin_coord',
                      'face_index'),
                hash_type='fnv',
                mode='train',
                return_grid_coord=True,
                return_inverse=True),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'inverse', 'origin_coord',
                      'face_index'),
                feat_keys=('coord', 'normal', 'color'))
        ],
        loop=1))
oid = 'state_1'
label = 'None'
