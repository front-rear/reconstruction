import numpy as np
import torch
import trimesh
import json
import cv2
import os
from os.path import join
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
import pointops
import sys
sys.path.append(os.path.abspath(".."))

from pointcept.datasets.sampart3d_util import *


def cal_mapping_2d_3d(render_dir, mesh_path):
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    samples, face_index, colors = sample_surface(mesh, 50000, sample_color=True)
    face_index = torch.from_numpy(face_index).int()
    face_index = torch.concat([face_index, torch.tensor([-1]).int()])

    meta_data = json.load(open(join(render_dir, "meta.json")))
    mesh_scale = meta_data["scaling_factor"]
    mesh_center_offset = meta_data["mesh_offset"]

    object_org_coord = samples
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]])
    object_org_coord = np.dot(object_org_coord, rotation_matrix)
    object_org_coord = object_org_coord * mesh_scale + mesh_center_offset
    object_org_coord = torch.from_numpy(object_org_coord).to("cuda").contiguous().float()
    obj_offset = torch.tensor(object_org_coord.shape[0]).to("cuda")

    mapping_list = []
    camera_angle_x = meta_data['camera_angle_x']
    for i, c2w_opengl in enumerate(meta_data["transforms"]):
        c2w_opengl = np.array(c2w_opengl)
        rgb_path = join(render_dir, f"render_{i:04d}.webp")
        img = np.array(Image.open(rgb_path))
        if img.shape[-1] == 4:
            mask_img = img[..., 3] == 0
            img[mask_img] = [255, 255, 255, 255]
            img = img[..., :3]
            img = Image.fromarray(img.astype('uint8'))

        # Calculate mapping
        depth_path = join(render_dir, f"depth_{i:04d}.exr")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth[..., 0]
        depth_valid = torch.tensor(depth < 65500.0)

        org_points = gen_pcd(depth, c2w_opengl, camera_angle_x)
        # points = torch.from_numpy(points).to(self.device).contiguous().float()
        org_points = torch.from_numpy(org_points)
        points_tensor = org_points.to("cuda").contiguous().float()
        offset = torch.tensor(points_tensor.shape[0]).to("cuda")
        indices, distances = pointops.knn_query(1, object_org_coord, obj_offset, points_tensor, offset)
        mapping = torch.zeros((depth.shape[0], depth.shape[1]), dtype=torch.int) - 1

        mask_dis = distances[..., 0] < 0.03
        indices[~mask_dis] = -1
        mapping[depth_valid] = face_index[indices.cpu().flatten()]

        mapping_list.append(mapping.cpu().numpy())
    return np.stack(mapping_list)


def highlight_parts_in_multi_views(render_dir, mesh_path, results_dir, save_dir, img_num=1):

    print(f"Processing {mesh_path}")
    obj_mapping = cal_mapping_2d_3d(render_dir, mesh_path)
    scale_list = ["0.0", "0.5", "1.0", "1.5", "2.0"]
    for scale in scale_list:
        ins_pred = np.load(join(results_dir, f"mesh_{scale}.npy"))
        # Get the number of images and the number of classes
        num_images = obj_mapping.shape[0]
        num_classes = np.max(ins_pred) + 1
        # Initialize an array to store the pixel count for each class in each image
        pixel_count = np.zeros((num_images, num_classes), dtype=np.int32)
        # Iterate over each image
        for i in range(num_images):
            # Get the group numbers for each pixel in the image
            valid_areas = obj_mapping[i] != -1
            groups = ins_pred[obj_mapping[i][valid_areas]]
            # Count the number of pixels for each group
            pixel_count[i], _ = np.histogram(groups, bins=np.arange(num_classes + 1) - 0.5)
        # Find the top 1 images for each class
        top_image_ids = np.argsort(-pixel_count, axis=0)[:img_num]
        # top_image_ids = np.stack([top_image_ids[0, :], top_image_ids[2, :], top_image_ids[4, :]])

        save_path = join(save_dir, scale)
        os.makedirs(save_path, exist_ok=True)
        for part_id in range(ins_pred.max()+1):
            img_id_list = top_image_ids[:, part_id]
            for topj, img_id in enumerate(img_id_list):
                image = np.array(Image.open(join(render_dir, f"render_{img_id:04d}.webp")))
                if image.shape[-1] == 4:
                    mask_img = image[..., 3] == 0
                    image[mask_img] = [255, 255, 255, 255]
                    image = image[..., :3]
                image = Image.fromarray(image)
                valid_areas = obj_mapping[img_id] != -1
                mask = np.zeros_like(obj_mapping[img_id], dtype=bool)
                mask[valid_areas] = (ins_pred[obj_mapping[img_id][valid_areas]] == part_id)

                # Find the edges of the mask
                edges = ndi.binary_dilation(mask, iterations=1) ^ mask
                # Draw a red circle around the edges
                draw = ImageDraw.Draw(image)
                for y, x in np.argwhere(edges):
                    draw.ellipse([x-2, y-2, x+2, y+2], fill='red')
                image.save(join(save_path, f"{part_id}-{topj}.png"))


if __name__ == '__main__':
    render_dir = ""
    mesh_path = ""
    results_dir = ""
    save_dir = ""
    highlight_parts_in_multi_views(render_dir, mesh_path, results_dir, save_dir)