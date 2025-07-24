import cv2
import trimesh
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from run_REART import load_mesh


def visualize_depth(depth_image_path: str, save_path: str) -> None:
    """
    Visualize the depth image by applying colormap to it.
    """

    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
    depth_image = depth_image.astype(np.float32) / 1000.0  # convert to meters

    plt.figure(figsize=(19.2, 10.8))
    plt.imshow(depth_image, cmap=cm.jet)
    plt.axis('off')
    
    # 设置保存图像的大小和dpi
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show(block=True)
    plt.close()

# def display_obj(obj_path: str) -> None:
#     """
#     Display the object mesh.
#     """
#     read_mesh = o3d.io.read_triangle_mesh(obj_path)

#     new_mesh = o3d.geometry.TriangleMesh()
#     new_mesh.vertices = read_mesh.vertices
#     new_mesh.triangles = read_mesh.triangles
#     new_mesh.triangle_uvs = read_mesh.triangle_uvs
#     new_mesh.triangle_material_ids = read_mesh.triangle_material_ids
#     new_mesh.vertex_colors = read_mesh.vertex_colors
#     new_mesh.textures = [read_mesh.textures[0]]

#     o3d.visualization.draw_geometries([new_mesh])

if __name__ == '__main__':
    # visualize_depth("/home/rvsa/gary318/build_kinematic/input_rgbd/cabinet_011201/depth_filtered/state_1.png", "visualize_depth.png")
    # display_obj("/home/rvsa/gary318/build_kinematic/input_rgbd/010602_fridge/SAM3D/state_2/vis_pcd/last/mesh_0.4.ply")

    # Save a coordinate frame mesh from open3d
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    # o3d.io.write_triangle_mesh("coordinate_frame.ply", mesh_frame)

    mesh = load_mesh("/home/rvsa/gary318/build_kinematic/input_rgbd/010602_fridge/objs/state_1/state_1.obj", "/home/rvsa/gary318/build_kinematic/input_rgbd/010602_fridge/oops/state_1_pose_adjusted.npz")
    
    mesh.export("state_1.glb")