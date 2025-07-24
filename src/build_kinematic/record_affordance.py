import os
import cv2
import numpy as np

from constants import CAM_INTRINSIC, DEPTH_SCALE, CAM_UNDER_BASE

from typing import List

def click_callback(event: int, x: int, y: int, flags: int, param: List) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        param.clear()
        param.append((x, y))
        print(f"Clicked at ({x}, {y})")

def record_affordance(dataset_path: str) -> None:
    rgb_path = os.path.join(dataset_path, 'raw_rgb')
    depth_path = os.path.join(dataset_path, 'raw_depth')
    assert os.path.exists(rgb_path), f"Cannot find {rgb_path}"
    assert os.path.exists(depth_path), f"Cannot find {depth_path}"

    state_nums: List[int] = []
    for filename in os.listdir(depth_path):
        suffix = os.path.splitext(filename)[1]
        basename = os.path.splitext(filename)[0]
        if suffix == '.npy':
            state_nums.append(int(basename[-1]))
    state_nums.sort()

    affordance_3d_coords = []
    for i, state_num in enumerate(state_nums):
        if i == 0:
            continue

        rgb = cv2.imread(os.path.join(rgb_path, f'state_{state_num}.png'))
        depth = np.load(os.path.join(depth_path, f'state_{state_num}.npy'))
        temp_list = []

        print(f"Recording affordance for state {state_num}")
        cv2.namedWindow('RGB')
        cv2.setMouseCallback('RGB', click_callback, param=temp_list)
        while True:
            cv2.imshow('RGB', rgb)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break

        # Convert depth to 3D coordinates
        u, v = temp_list[0]
        z = depth[v, u] / DEPTH_SCALE
        cam_coord = np.linalg.inv(CAM_INTRINSIC).dot(np.array([u, v, 1])) * z
        world_coord = CAM_UNDER_BASE @ np.concatenate((cam_coord, [1]))
        affordance_3d_coords.append(world_coord[:3])

    cv2.destroyAllWindows()
    np.save(os.path.join(dataset_path, 'affordance.npy'), np.array(affordance_3d_coords))
