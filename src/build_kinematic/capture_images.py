## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
import os, time

dataset_folder = '/home/adrianjiang/gary318/123101'

# Create a pipeline
pipeline_main = rs.pipeline()
pipeline_side = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config_main = rs.config()
config_main.enable_device("105422061051")
config_main.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
config_main.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

config_side = rs.config()
config_side.enable_device("847412063499")
config_side.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)


# Get device product line for setting a supporting resolution
# pipeline_wrapper = rs.pipeline_wrapper(pipeline)
# pipeline_profile = config.resolve(pipeline_wrapper)
# device = pipeline_profile.get_device()
# device_product_line = str(device.get_info(rs.camera_info.product_line))


# Start streaming
profile_main = pipeline_main.start(config_main)
profile_side = pipeline_side.start(config_side)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile_main.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

p = os.path.join(dataset_folder, "raw_rgb")
if not os.path.exists(p):
    os.makedirs(p)

p = os.path.join(dataset_folder, "raw_depth")
if not os.path.exists(p):
    os.makedirs(p)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

state = 0
recording = False
# Streaming loop
try:
    #Declare filters
    dec_filter = rs.decimation_filter ()   # Decimation - reduces depth frame density. Value range [2-8]. Default is 2.
    spat_filter = rs.spatial_filter()      # Spatial    - edge-preserving spatial smoothing
    temp_filter = rs.temporal_filter()     # Temporal   - reduces temporal noise
    hole_filter = rs.hole_filling_filter() # Hole Filling

    while True:
        # Get frameset of color and depth (main camera)
        frames_main = pipeline_main.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames_main)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        aligned_depth_frame = hole_filter.process(aligned_depth_frame)
        aligned_depth_frame = spat_filter.process(aligned_depth_frame)
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image_main = np.asanyarray(color_frame.get_data())

        # Get side camera frame
        frames_side = pipeline_side.wait_for_frames()
        color_frame_side = np.asanyarray(frames_side.get_color_frame().get_data())

        # Render images:2
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image_main, depth_colormap, color_frame_side))

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        elif key & 0xFF == ord('r'):
            recording = not recording
            print("Recording is set to", recording)

        if recording or (key & 0xFF == ord('s')):
            state += 1
            cv2.imwrite(os.path.join(dataset_folder, "raw_rgb", "state_%d.png" % state), color_image_main)
            cv2.imwrite(os.path.join(dataset_folder, "raw_rgb", "state_%d_side.png" % state), color_frame_side)
            np.save(os.path.join(dataset_folder, "raw_depth", "state_%d.npy" % state), depth_image)
            print('Saved image #%d' % state)
            time.sleep(0.2)

finally:
    pipeline_main.stop()
    pipeline_side.stop()
