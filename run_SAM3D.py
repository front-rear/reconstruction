import os
import shutil
import trimesh
import numpy as np

import constants

from typing import List


def run_SAM3D_render(input_glb_folder: str, output_folder: str, pick_filename: List[str] = None) -> None:
    assert os.path.exists(input_glb_folder), f"Input folder {input_glb_folder} does not exist"
    os.makedirs(output_folder, exist_ok=True)

    for file in pick_filename if pick_filename else os.listdir(input_glb_folder):
        if not file.endswith(".glb"):
            continue

        file_name = os.path.splitext(file)[0]
        input_file = os.path.join(input_glb_folder, file)
        new_folder = os.path.join(output_folder, file_name)
        os.makedirs(new_folder, exist_ok=True)

        os.system(f"{constants.BLENDER_PATH} -b -P {os.path.join(constants.SAM3D_PATH, 'tools', 'blender_render_16views.py')} {input_file} glb {new_folder}")


def run_SAM3D_inference(render_folder: str, output_folder: str, pick_filename: List[str] = None) -> None:
    render_folder = os.path.abspath(render_folder)
    output_folder = os.path.abspath(output_folder)
    assert os.path.exists(render_folder), f"Render folder {render_folder} does not exist"
    os.makedirs(output_folder, exist_ok=True)

    with open(constants.SAM3D_CONFIG_BASE, "r") as f:
        base_content = f.read()

    for dir_name in pick_filename if pick_filename else os.listdir(render_folder):
        dir_path = os.path.join(render_folder, dir_name)
        if not os.path.isdir(dir_path):
            continue

        new_content = base_content.replace("!!!DATA_ROOT_PLACEHOLDER!!!", render_folder) \
                                 .replace("!!!MESH_ROOT_PLACEHOLDER!!!", os.path.join(render_folder, "..", "mesh"))
        new_config_file = os.path.join(os.path.dirname(constants.SAM3D_CONFIG_BASE), "build_kinematic_temp.py")
        with open(new_config_file, "w") as f:
            f.write(new_content)

        os.system(f"sh {os.path.join(constants.SAM3D_PATH, 'scripts', 'simplified_train.sh')} -g 1 -c {new_config_file} -n {os.path.join(output_folder, dir_name)} -o {dir_name} -p {constants.SAM3D_PYTHON_PATH}")


def extract_SAM3D_results(SAM3D_output_folder: str) -> None:
    SAM3D_output_folder = os.path.abspath(SAM3D_output_folder)
    assert os.path.exists(SAM3D_output_folder), f"SAM3D output folder {SAM3D_output_folder} does not exist"

    X_ROTATE_90 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    for dir_name in os.listdir(SAM3D_output_folder):
        dir_path = os.path.join(SAM3D_output_folder, dir_name)
        if not os.path.isdir(dir_path):
            continue

        shutil.copy(os.path.join(dir_path, "vis_pcd", "last", f"{constants.SAM3D_RESULT_SCALE}.ply"),
                    os.path.join(dir_path, "pointcloud.ply"))
        shutil.copy(os.path.join(dir_path, "results", "last", f"mesh_{constants.SAM3D_RESULT_SCALE}.npy"),
                    os.path.join(dir_path, "labels.npy"))

        mesh = trimesh.load(os.path.join(dir_path, "vis_pcd", "last", f"mesh_{constants.SAM3D_RESULT_SCALE}.ply"))
        mesh.apply_transform(X_ROTATE_90)
        mesh.export(os.path.join(dir_path, "mesh.ply"))


if __name__ == "__main__":
    run_SAM3D_render("/home/rvsa/gary318/build_kinematic/input_rgbd/123001/mesh", "/home/rvsa/gary318/build_kinematic/input_rgbd/123001/render")
    run_SAM3D_inference("/home/rvsa/gary318/build_kinematic/input_rgbd/123001/render", "/home/rvsa/gary318/build_kinematic/input_rgbd/123001/SAM3D")
