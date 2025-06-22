import os
import shutil
import trimesh
import numpy as np


import constants

from typing import List

def run_SAM3D_render(input_glb_folder: str, output_folder: str, pick_filename: List[str] = []) -> None:
    assert os.path.exists(input_glb_folder), f"Input folder {input_glb_folder} does not exist"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    """
    blender-4.0.0-linux-x64/blender -b -P blender_render_16views.py mesh_root/knight.glb glb data_root/knight
    """
    for file in pick_filename if pick_filename else os.listdir(input_glb_folder):
        suffix = os.path.splitext(file)[1][1:]
        if suffix != "glb":
            continue
        file_name = os.path.splitext(file)[0].removesuffix("." + suffix)

        input_file = os.path.join(input_glb_folder, file)
        new_folder = os.path.join(output_folder, file_name)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

        os.system("%s -b -P %s %s glb %s" % (
            constants.BLENDER_PATH, os.path.join(constants.SAM3D_PATH, "tools", "blender_render_16views.py"),
            input_file, new_folder
        ))

def run_SAM3D_inference(render_folder: str, output_folder: str, pick_filename: List[str] = []) -> None:
    render_folder = os.path.abspath(render_folder)
    output_folder = os.path.abspath(output_folder)
    assert os.path.exists(render_folder), f"Render folder {render_folder} does not exist"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Base cfg file
    with open(constants.SAM3D_CONFIG_BASE, "r") as f:
        base_content = f.read()

    """
    sh scripts/train.sh -g 1 -d sampart3d -c sampart3d-trainmlp-render16views -n knight -o knight
    """
    for dir_name in pick_filename if pick_filename else os.listdir(render_folder):
        if not os.path.isdir(os.path.join(render_folder, dir_name)):
            continue

        # Generate the config file
        new_content = base_content.replace("!!!DATA_ROOT_PLACEHOLDER!!!", render_folder) \
                                  .replace("!!!MESH_ROOT_PLACEHOLDER!!!", os.path.abspath(os.path.join(render_folder, "..", "mesh")))
        new_config_file = os.path.join(os.path.dirname(constants.SAM3D_CONFIG_BASE), "build_kinematic_temp.py")
        with open(new_config_file, "w") as f:
            f.write(new_content)

        os.system("sh %s -g 1 -c %s -n %s -o %s -p %s" % (
            os.path.join(constants.SAM3D_PATH, "scripts", "simplified_train.sh"), new_config_file,
            os.path.join(output_folder, dir_name), dir_name,
            constants.SAM3D_PYTHON_PATH
        ))

def extract_SAM3D_results(SAM3D_output_folder: str) -> None:
    SAM3D_output_folder = os.path.abspath(SAM3D_output_folder)
    assert os.path.exists(SAM3D_output_folder), f"SAM3D output folder {SAM3D_output_folder} does not exist"

    X_ROTATE_90 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    for dir_name in os.listdir(SAM3D_output_folder):
        if not os.path.isdir(os.path.join(SAM3D_output_folder, dir_name)):
            continue
        # if dir_name != "state_2":
        #     continue

        # Extract the results
        dir_path = os.path.join(SAM3D_output_folder, dir_name)
        shutil.copy(os.path.join(dir_path, "vis_pcd", "last", "%s.ply" % constants.SAM3D_RESULT_SCALE),
                    os.path.join(dir_path, "pointcloud.ply"))
        shutil.copy(os.path.join(dir_path, "results", "last", "mesh_%s.npy" % constants.SAM3D_RESULT_SCALE),
                    os.path.join(dir_path, "labels.npy"))

        mesh = trimesh.load(os.path.join(dir_path, "vis_pcd", "last", "mesh_%s.ply" % constants.SAM3D_RESULT_SCALE))
        mesh.apply_transform(X_ROTATE_90)
        mesh.export(os.path.join(dir_path, "mesh.ply"))

        # Visualize the results
        # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([mesh, coord_frame])

if __name__ == "__main__":
    run_SAM3D_render("/home/rvsa/gary318/build_kinematic/input_rgbd/123001/mesh", "/home/rvsa/gary318/build_kinematic/input_rgbd/123001/render")
    run_SAM3D_inference("/home/rvsa/gary318/build_kinematic/input_rgbd/123001/render", "/home/rvsa/gary318/build_kinematic/input_rgbd/123001/SAM3D")
