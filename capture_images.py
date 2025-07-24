## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

import pyrealsense2 as rs
import numpy as np
import cv2
import os

dataset_folder = '/work'

# Create pipelines and configs
pipeline_main = rs.pipeline()
pipeline_side = rs.pipeline()

config_main = rs.config()
config_main.enable_device("105422061051")
config_main.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
config_main.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

config_side = rs.config()
config_side.enable_device("847412063499")
config_side.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

# Start streaming
profile_main = pipeline_main.start(config_main)
profile_side = pipeline_side.start(config_side)

depth_sensor = profile_main.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Create folders
for folder_name in ["raw_rgb", "raw_depth"]:
    p = os.path.join(dataset_folder, folder_name)
    if not os.path.exists(p):
        os.makedirs(p)

align_to = rs.stream.color
align = rs.align(align_to)

hole_filter = rs.hole_filling_filter()
spat_filter = rs.spatial_filter()

state = 0
recording = False

try:
    while True:
        frames_main = pipeline_main.wait_for_frames()
        aligned_frames = align.process(frames_main)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image_main = np.asanyarray(color_frame.get_data())

        frames_side = pipeline_side.wait_for_frames()
        color_frame_side = np.asanyarray(frames_side.get_color_frame().get_data())

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image_main, depth_colormap, color_frame_side))

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        elif key & 0xFF == ord('r'):
            recording = not recording
            print("Recording is set to", recording)

        if recording or (key & 0xFF == ord('s')):
            state += 1
            cv2.imwrite(os.path.join(dataset_folder, "raw_rgb", f"state_{state}.png"), color_image_main)
            cv2.imwrite(os.path.join(dataset_folder, "raw_rgb", f"state_{state}_side.png"), color_frame_side)
            np.save(os.path.join(dataset_folder, "raw_depth", f"state_{state}.npy"), depth_image)
            print(f'Saved image #{state}')
            time.sleep(0.2)

finally:
    pipeline_main.stop()
    pipeline_side.stop()
