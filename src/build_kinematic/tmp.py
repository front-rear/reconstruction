import numpy as np

pth = "/home/rvsa/gary318/build_kinematic/input_rgbd/compound_011603/oops/state_3_pose_adjusted.npz"

data = np.load(pth)

np.savez(pth, pose=data['pose'], scale=2.094301357202575)