import pyrealsense2 as rs
import cv2
import numpy as np

import os
os.chdir(os.path.dirname(__file__))

pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipe.start(config)

recording = False

try:
  while True:
    frames = pipe.wait_for_frames()
    tstamp: float = frames.timestamp
    color = np.array(frames.get_color_frame().data)
    depth = np.array(frames.get_depth_frame().data)

    

    cv2.imshow("color", color)
    key = cv2.waitKey(1)
    if key == ord('q'):
       break
    if key == ord('r'):
       recording = not recording
    if recording or key == ord('c'):
      print(tstamp)
      cv2.imwrite("%f.jpg" % tstamp, color)
      np.save("%f.npy" % tstamp, depth)
finally:
    pipe.stop()