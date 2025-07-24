import cv2
import tqdm
import os
import open3d as o3d
import numpy as np

from constants import RGB_CROP_PADDING

class PoseEstimator:

    dataset_path: str
    """The path to the current working dataset."""
    
    def set_dataset(self, dataset_path: str) -> None:
        self.dataset_path = dataset_path

    def generate_masked_rgbd(self) -> None:
        """Generates masked RGBD images for the current working dataset."""

        raw_color_path = self.dataset_path + '/raw_rgb/'
        raw_depth_path = self.dataset_path + '/raw_depth/'
        mask_path = self.dataset_path + '/mask/'
        masked_color_path = self.dataset_path + '/color_segmented/'
        masked_depth_path = self.dataset_path + '/depth_filtered/'
        cropped_color_path = self.dataset_path + '/color_cropped/'

        if not os.path.exists(masked_color_path): os.makedirs(masked_color_path)
        if not os.path.exists(masked_depth_path): os.makedirs(masked_depth_path)
        if not os.path.exists(cropped_color_path): os.makedirs(cropped_color_path)

        img_names = sorted(os.listdir(raw_color_path))
        for img_name in img_names:
            img_color = cv2.imread(os.path.join(raw_color_path, img_name))
            img_mask = np.asarray(o3d.io.read_image(os.path.join(mask_path, img_name)))
            valid_y, valid_x = np.where(img_mask > 0)

            # Masking the image
            img_masked_color = img_color.copy()
            img_masked_color[img_mask == 0] = 0
            cv2.imwrite(os.path.join(masked_color_path, img_name), img_masked_color)

            # Add a 'A' channel
            img_masked_color = cv2.cvtColor(img_masked_color, cv2.COLOR_BGR2BGRA)
            img_masked_color[:, :, 3] = img_mask.astype(np.uint8)

            # Cropping the image
            ymin, ymax = max(0, valid_y.min() - RGB_CROP_PADDING), min(img_color.shape[0], valid_y.max() + RGB_CROP_PADDING)
            xmin, xmax = max(0, valid_x.min() - RGB_CROP_PADDING), min(img_color.shape[1], valid_x.max() + RGB_CROP_PADDING)
            img_cropped_color = img_masked_color[ymin:ymax, xmin:xmax, :]
            cv2.imwrite(os.path.join(cropped_color_path, img_name), img_cropped_color)

            depth_path = os.path.join(raw_depth_path, img_name.replace(".png", ".npy"))
            if os.path.exists(depth_path):
                depth = np.load(depth_path).astype(np.uint16)
                img_masked_depth = depth.copy()
                img_masked_depth[img_mask == 0] = 0
                cv2.imwrite(os.path.join(raw_depth_path, img_name), depth.astype(np.uint16))
                cv2.imwrite(os.path.join(masked_depth_path, img_name), img_masked_depth)

    def generate_pcd(self, intrinsic: np.ndarray, depth_scale: float = 1000.0) -> None:
        """Generates point clouds for the current working dataset."""

        color_path = self.dataset_path + '/color_segmented/'
        depth_path = self.dataset_path + '/depth_filtered/'
        pcd_path = self.dataset_path + '/pcd/'

        if not os.path.exists(pcd_path): os.makedirs(pcd_path)

        intrinsic = o3d.camera.PinholeCameraIntrinsic(1280, 720, intrinsic)
        # intrinsic.intrinsic_matrix = intrinsic

        img_names = sorted(os.listdir(color_path))
        for img_name in tqdm.tqdm(img_names):
            name_part = os.path.splitext(img_name)[0]
            img_color = o3d.io.read_image(os.path.join(color_path, img_name))
            img_depth = o3d.io.read_image(os.path.join(depth_path, img_name))

            source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth, depth_scale=depth_scale,
                                                                                   convert_rgb_to_intensity=False, depth_trunc=3.0)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, intrinsic)

            # # Plane Segmentation
            # plane_model, inliers = pcd.segment_plane(distance_threshold=0.02,
            #                                         ransac_n=3,
            #                                         num_iterations=1000)
            # [a, b, c, d] = plane_model
            # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
            # inlier_cloud = pcd.select_by_index(inliers)
            # inlier_cloud.paint_uniform_color([1.0, 0, 0])
            # outlier_cloud = pcd.select_by_index(inliers, invert=True)
            # # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
            # # o3d.visualization.draw_geometries([outlier_cloud])

            # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            #     labels = np.array(
            #         outlier_cloud.cluster_dbscan(eps=0.02, # Epsilon defines the distance between to neighbors in a cluster
            #                         min_points=500, # minimum number of points required to form a cluster
            #                         print_progress=True))

            # max_label = labels.max()
            # print(f"point cloud has {max_label + 1} clusters")

            # # clusters = labels
            # indexes = np.where(labels == 0)

            # # Extract Interest point clouds
            # interest_pcd = o3d.geometry.PointCloud()
            # interest_pcd.points = o3d.utility.Vector3dVector(np.asarray(outlier_cloud.points, np.float32)[indexes])
            # interest_pcd.colors = o3d.utility.Vector3dVector(np.asarray(outlier_cloud.colors, np.float32)[indexes])

            # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
            # colors[labels < 0] = 0
            # outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
            # # o3d.visualization.draw_geometries([interest_pcd])

            # y = np.asarray(outlier_cloud.points)[:, 1]
            # y_mean = np.mean(y)
            # # plt.plot(y)
            # # plt.show()
            # # idx = np.array([i for i in range(len(z))], dtype=np.int)
            # idx = np.where(y < 0.18)[0]
            # idx = np.asarray(idx, dtype=np.int)

            # interest_pcd = outlier_cloud.select_by_index(list(idx))
            # # o3d.visualization.draw_geometries([interest_pcd])
            # o3d.visualization.draw_geometries([pcd])

            # print("Statistical oulier removal")
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=1000, std_ratio=0.8)
            # o3d.visualization.draw_geometries([cl])

            # print("Radius oulier removal")
            # cl, ind = cl.remove_radius_outlier(nb_points=100, radius=0.01)
            # o3d.visualization.draw_geometries([cl])

            o3d.io.write_point_cloud(os.path.join(pcd_path, name_part + '.pcd'), cl)

    def generate_pose(self) -> None:
        pass

if __name__ == '__main__':
    est = PoseEstimator()
    est.set_dataset("/home/rvsa/SAM/data/122601")

    intrinsic_435 = np.array([
        [920.523681640625, 0, 648.9248657226562],
        [0, 920.2175903320312, 362.54510498046875],
        [0, 0, 1]
    ], dtype=np.float64)
    intrinsic_415 = np.array([
        [906.461181640625, 0, 635.8511962890625],
        [0, 905.659912109375, 350.6916809082031],
        [0, 0, 1]
    ], dtype=np.float64)


    est.generate_masked_rgbd()
    est.generate_pcd(intrinsic_415)
