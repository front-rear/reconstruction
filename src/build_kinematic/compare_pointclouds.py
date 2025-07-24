import os
import numpy as np
import open3d as o3d

import constants

def tag_difference(pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud, tag: int) -> None:
    """
    Tag the difference between two point clouds INPLACE.
    """

    dist_12 = pcd1.compute_point_cloud_distance(pcd2)
    pcd1.colors = np.where(dist_12 < constants.PCD_OVERLAP_THRESHOLD,
                           np.repeat(np.array(constants.COLOR_MAP[tag]), len(dist_12)), pcd1.colors)

    dist_21 = pcd2.compute_point_cloud_distance(pcd1)
    pcd2.colors = np.where(dist_21 < constants.PCD_OVERLAP_THRESHOLD,
                           np.repeat(np.array(constants.COLOR_MAP[tag]), len(dist_21)), pcd2.colors)
