import numpy as np

from typing import Tuple, Literal

PRESET: Literal["laptop", "cabinet", "lamp_round", "lamp_thin", "drawer", "compound", None] = "cabinet"

# SAM
SAM_PATH = "/SAM"
SAM_SCRIPT_PATH = "sam_script.py"

# RODIN
RGB_CROP_PADDING = 30
RODIN_BASE_URL = "https://hyperhuman.deemos.com/api/v2"

# SAM3D
BLENDER_PATH = "/SAMPart3D/blender-4.0.0-linux-x64/blender"
SAM3D_PATH = "/SAMPart3D"
SAM3D_CONFIG_BASE = "/SAMPart3D/configs/dexSim2Real/fridge_config_base.py"
SAM3D_PYTHON_PATH = "/home/rvsa/miniconda3/envs/reart/bin/python"
SAM3D_RESULT_SCALE = "0.2"

ICP_THRESHOLD = 0.02
OOPS_MAX_VALID_GT_DEPTH = 1750

# Movable Part Segmentation
SAMPLE_SIZE = 20000
CLUSTER_MERGING_THRESHOLD = 0.05

PCD_DIST_RATIO_THRESHOLD = 0.15
PCD_DIST_MEDIAN_THRESHOLD = 0.15  # cm
PCD_ALLCLOSE_THRESHOLD = 0.01
PCD_CASTING_THRESHOLD = 0.04
MESH_CASTING_THRESHOLD = 0.01
CONVEX_DECOMP_RESULT_COUNT = 20

# REART
REART_PATH = "/reart"
REART_DATA_TEMPLATE = "/reart/data/mbs-sapien/data/000212 copy.npz"
REART_DATA_PATH = "/reart/data/mbs-sapien/data/000212.npz"
REART_POINTCLOUD_SIZE = int(512 * 3)  # Multiple of 512
REART_POINTCLOUD_COUNT = 4

GT_FILE_PATH = "/build_kinematic/gt.csv"

# Visualization
COLOR_MAP: Tuple[Tuple[float, float, float], ...] = (
    (0.375, 0.375, 0.375),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0.75, 0.75, 0),
    (0.75, 0, 0.75),
)
def get_color_id(color: Tuple[float, float, float]) -> int:
    for i, c in enumerate(COLOR_MAP):
        if c == color:
            return i
    raise ValueError(f"Color {color} not found in color map")


DEFAULT_SCALE = 1 / 6.0
X_ROTATE_180 = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
X_ROTATE_90 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
CAM_UNDER_BASE = np.array([
    [0.70512699, -0.21169334, 0.67674357, 0.14125227],
    [-0.7090684, -0.216209, 0.67117485, -0.4770995],
    [0.0042348, -0.95312098, -0.30255986, 0.54021694],
    [0, 0, 0, 1]
])
CAM_INTRINSIC = np.array([
    [906.461181640625, 0, 635.8511962890625],
    [0, 905.659912109375, 350.6916809082031],
    [0, 0, 1]
])
DEPTH_SCALE = 1000.0

GRID_SIZE = 0.025
ROBOT_BASE_GRID_COORD = (52, 10)
ROBOT_BASE_HEIGHT = 0.022

def parse_grid_coord(coord_str: str) -> np.ndarray:
    x, y, z = coord_str.split(" ")
    return np.array([
        (ROBOT_BASE_GRID_COORD[0] - float(x)) * GRID_SIZE,
        (ROBOT_BASE_GRID_COORD[1] - float(y)) * GRID_SIZE,
        float(z) - ROBOT_BASE_HEIGHT
    ])

if PRESET == "laptop":
    ICP_THRESHOLD = 0.02
    CLUSTER_MERGING_THRESHOLD = 0.075
    PCD_DIST_RATIO_THRESHOLD = 0.15
    PCD_DIST_MEDIAN_THRESHOLD = 0.12  # cm
    SAM3D_RESULT_SCALE = "0.2"
elif PRESET == "cabinet":
    ICP_THRESHOLD = 0.02
    CLUSTER_MERGING_THRESHOLD = 0.05

    PCD_DIST_RATIO_THRESHOLD = 0.08
    PCD_DIST_MEDIAN_THRESHOLD = 0.15  # cm
    SAM3D_RESULT_SCALE = "0.2"
elif PRESET == "lamp_round":
    ICP_THRESHOLD = 0.01
    CLUSTER_MERGING_THRESHOLD = 0.05

    PCD_DIST_RATIO_THRESHOLD = 0.15
    PCD_DIST_MEDIAN_THRESHOLD = 0.17  # cm
    SAM3D_RESULT_SCALE = "0.2"
elif PRESET == "lamp_thin":
    ICP_THRESHOLD = 0.01
    CLUSTER_MERGING_THRESHOLD = 0.05

    PCD_DIST_RATIO_THRESHOLD = 0.15
    PCD_DIST_MEDIAN_THRESHOLD = 0.3  # cm
    SAM3D_RESULT_SCALE = "0.2"

    REART_POINTCLOUD_SIZE = 512 * 3
elif PRESET == "drawer":
    ICP_THRESHOLD = 0.02
    CLUSTER_MERGING_THRESHOLD = 0.03

    PCD_DIST_RATIO_THRESHOLD = 0.15
    PCD_DIST_MEDIAN_THRESHOLD = 0.08  # cm
    SAM3D_RESULT_SCALE = "0.2"
elif PRESET == "compound":
    ICP_THRESHOLD = 0.02

    CLUSTER_MERGING_THRESHOLD = 0.01
    PCD_DIST_RATIO_THRESHOLD = 0.1
    PCD_DIST_MEDIAN_THRESHOLD = 0.20  # cm
    PCD_CASTING_THRESHOLD = 0.09
    SAM3D_RESULT_SCALE = "0.2"
