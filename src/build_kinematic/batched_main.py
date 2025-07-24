import os
import time
import logging
import numpy as np

import constants
from util import PoseEstimator
from run_rodin import run_rodin, glb_to_obj
from run_oops import run_oops, adjust_scale, refine_pose, calculate_mask_error, visualize_pose
from run_SAM3D import run_SAM3D_render, run_SAM3D_inference, extract_SAM3D_results
from run_REART import segment_mesh, run_REART, extract_urdf
from record_affordance import record_affordance

if __name__ == '__main__':
    logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s %(message)s')

    mask_errors = []
    rotation_err = []
    translation_err = []

    base_input_folder = "/home/rvsa/gary318/build_kinematic/input_rgbd"
    # dataset_folder = "/home/rvsa/gary318/build_kinematic/input_rgbd/010502_laptop"

    # datasets = [os.path.join(base_input_folder, "laptop_gray_01100%d") % i for i in range(1, 5+1)]
    # datasets += [os.path.join(base_input_folder, "laptop_gold_01120%d") % i for i in range(1, 5+1)]

    # datasets = [os.path.join(base_input_folder, "cabinet_01100%d") % i for i in [1, 3, 4, 5]]
    # datasets += [os.path.join(base_input_folder, name) for name in ["cabinet_011202"]]
    # datasets += [os.path.join(base_input_folder, "fridge_01140%d") % i for i in range(1, 5+1)]

    # datasets += [os.path.join(base_input_folder, "fridge_01100%d") % i for i in range(1, 6+1)]
    # datasets = [os.path.join(base_input_folder, "lamp_thin_01130%d") % i for i in [1, 2, 3, 4, 6]]
    # datasets += [os.path.join(base_input_folder, "lamp_round_01110%d") % i for i in range(2, 7+1)]

    # datasets = [os.path.join(base_input_folder, "drawer_compact_01140%d") % i for i in [1, 2, 3, 5, 6]]
    # datasets += [os.path.join(base_input_folder, "drawer_simple_01140%d") % i for i in [1, 3, 4, 5, 6]]

    datasets = [os.path.join(base_input_folder, "fig1_011801")]

    # for dataset_folder in datasets:
    #     os.chdir(constants.SAM_PATH)
    #     os.system(f"python {constants.SAM_SCRIPT_PATH} " +
    #               f"--img_in_path={os.path.join(dataset_folder, 'raw_rgb')} " +
    #               f"--img_out_path={os.path.join(dataset_folder, 'mask')}"
    #     )
    #     record_affordance(dataset_folder)

    # raise

    for dataset_folder in datasets:
        logging.info(f"Processing {dataset_folder}")
        mesh_folder = os.path.join(dataset_folder, "mesh")
        render_folder = os.path.join(dataset_folder, "render")
        SAM3D_folder = os.path.join(dataset_folder, "SAM3D")

        # Process mask
        est = PoseEstimator()
        est.set_dataset(dataset_folder)
        est.generate_masked_rgbd()

        # Run Rodin
        # while True:
        #     try:
        #         run_rodin(os.path.join(dataset_folder, "color_cropped"), mesh_folder)
        #         logging.info(f"Run Rodin completed for {dataset_folder}")
        #         break
        #     except Exception as e:
        #         logging.error(f"Rodin failed for {dataset_folder}: {e}, Retrying...")
        #         time.sleep(10)
        # glb_to_obj(mesh_folder, os.path.join(dataset_folder, "objs"))

        # Run OOPS
        # try:
        #     run_oops(dataset_folder)
        #     adjust_scale(dataset_folder, display_hist=False)
        #     refine_pose(dataset_folder, display_icp=False)
        #     mask_errors.append(calculate_mask_error(dataset_folder))
        #     visualize_pose(dataset_folder)
        #     logging.info(f"OOPS completed for {dataset_folder}")
        # except:
        #     logging.error(f"OOPS failed for {dataset_folder}")
        #     continue

        # SAM3D
        # try:
        #     run_SAM3D_render(mesh_folder, render_folder)
        #     run_SAM3D_inference(render_folder, SAM3D_folder)
        #     logging.info(f"SAM3D completed for {dataset_folder}")
        # except Exception as e:
        #     logging.error(f"SAM3D failed for {dataset_folder}: {e}")
        #     continue

        # Movable Part Segmentation
        # try:
        #     extract_SAM3D_results(SAM3D_folder)
        #     segment_mesh(dataset_folder, show_difference=True, show_final_pcd=True)
        #     logging.info(f"Part Segmentation completed for {dataset_folder}")
        # except Exception as e:
        #     logging.error(f"Part Segmentation failed for {dataset_folder}: {e}")
        #     continue

        # REART
        try:
            while True:
                run_REART(dataset_folder)
                all_revolute_joints, all_connect_to_root, ret_dict = extract_urdf(dataset_folder)
                # rotation_err.append(ret_dict["axis_rotation_error"])
                # translation_err.append(ret_dict["axis_translation_error"])

                if all_connect_to_root:
                    break
                else:
                    logging.error(f"REART failed for {dataset_folder}, retrying...")
            logging.info(f"Reart gives {all_revolute_joints} and {all_connect_to_root}")
        except ValueError as e:
            logging.error(f"REART failed for {dataset_folder}: {e}")
            continue

        logging.info(f"Finished processing {dataset_folder}")

    # print(f"Rotation error: mean {np.mean(rotation_err)}, min {np.min(rotation_err)}, max {np.max(rotation_err)}")
    # print(f"Translation error: mean {np.mean(translation_err)}, min {np.min(translation_err)}, max {np.max(translation_err)}")
    # mask_errors = np.array(mask_errors)
    # print(f"Mask error: mean {np.mean(mask_errors, axis=(0, 2))}, min {np.min(mask_errors, axis=(0, 2))}, max {np.max(mask_errors, axis=(0, 2))}")