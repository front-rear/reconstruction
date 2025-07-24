import cv2
import os
import open3d as o3d
import numpy as np

from constants import RGB_CROP_PADDING


class PoseEstimator:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def generate_masked_rgbd(self) -> None:
        raw_color_path = os.path.join(self.dataset_path, "raw_rgb")
        raw_depth_path = os.path.join(self.dataset_path, "raw_depth")
        mask_path = os.path.join(self.dataset_path, "mask")
        masked_color_path = os.path.join(self.dataset_path, "color_segmented")
        masked_depth_path = os.path.join(self.dataset_path, "depth_filtered")
        cropped_color_path = os.path.join(self.dataset_path, "color_cropped")

        os.makedirs(masked_color_path, exist_ok=True)
        os.makedirs(masked_depth_path, exist_ok=True)
        os.makedirs(cropped_color_path, exist_ok=True)

        for img_name in sorted(os.listdir(raw_color_path)):
            img_color = cv2.imread(os.path.join(raw_color_path, img_name))
            img_mask = np.array(o3d.io.read_image(os.path.join(mask_path, img_name)))

            img_masked_color = img_color.copy()
            img_masked_color[img_mask == 0] = 0
            cv2.imwrite(os.path.join(masked_color_path, img_name), img_masked_color)

            valid_y, valid_x = np.where(img_mask > 0)
            ymin, ymax = max(0, valid_y.min() - RGB_CROP_PADDING), min(img_color.shape[0], valid_y.max() + RGB_CROP_PADDING)
            xmin, xmax = max(0, valid_x.min() - RGB_CROP_PADDING), min(img_color.shape[1], valid_x.max() + RGB_CROP_PADDING)
            img_cropped_color = img_masked_color[ymin:ymax, xmin:xmax]
            cv2.imwrite(os.path.join(cropped_color_path, img_name), img_cropped_color)

            depth_path = os.path.join(raw_depth_path, img_name.replace(".png", ".npy"))
            if os.path.exists(depth_path):
                depth = np.load(depth_path).astype(np.uint16)
                img_masked_depth = depth.copy()
                img_masked_depth[img_mask == 0] = 0
                cv2.imwrite(os.path.join(masked_depth_path, img_name), img_masked_depth)

    def generate_pcd(self, intrinsic: np.ndarray, depth_scale: float = 1000.0) -> None:
        color_path = os.path.join(self.dataset_path, "color_segmented")
        depth_path = os.path.join(self.dataset_path, "depth_filtered")
        pcd_path = os.path.join(self.dataset_path, "pcd")
        os.makedirs(pcd_path, exist_ok=True)

        intrinsic = o3d.camera.PinholeCameraIntrinsic(1280, 720, intrinsic)
        for img_name in sorted(os.listdir(color_path)):
            name_part = os.path.splitext(img_name)[0]
            img_color = o3d.io.read_image(os.path.join(color_path, img_name))
            img_depth = o3d.io.read_image(os.path.join(depth_path, img_name))

            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                img_color, img_depth, depth_scale=depth_scale, convert_rgb_to_intensity=False, depth_trunc=3.0
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

            cl, _ = pcd.remove_statistical_outlier(nb_neighbors=1000, std_ratio=0.8)
            o3d.io.write_point_cloud(os.path.join(pcd_path, f"{name_part}.pcd"), cl)


if __name__ == "__main__":
    dataset_path = "/SAM/data/122601"
    intrinsic_415 = np.array([
        [906.461181640625, 0, 635.8511962890625],
        [0, 905.659912109375, 350.6916809082031],
        [0, 0, 1]
    ], dtype=np.float64)

    estimator = PoseEstimator(dataset_path)
    estimator.generate_masked_rgbd()
    estimator.generate_pcd(intrinsic_415)
