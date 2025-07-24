import os
import json
import copy
from tqdm import tqdm

import open3d as o3d
import numpy as np
import trimesh
import cv2
import matplotlib.pyplot as plt

import Utils
from ADD_S import csv_to_dict
from ADD_S import img_id_2_img_name,img_name_2_img_id,objname_2_objid,objid_2_objname
from ADD_S import max_k,min_k,find_according_imid_objid,t_transform,R_transform,cal_add_s,write_markdown,depth_2_pts


def draw_roc_and_cal_auc(add_s_indi):

    # add_s_all = add_s_all[:, 0]
    # auc, X, Y = Utils.compute_auc_sklearn(add_s_all[add_s_all < 2000], 3000, 1)
    # #
    # plt.xlabel("add_s_thredhold(mm)")
    # plt.ylabel("Precision")
    # plt.plot(X, Y, label=f"ALL(auc:{auc})")

    plt.xlabel("add_s_thredhold(obj_diameter)")
    plt.ylabel("Precision")

    for obj_id in range(46):
        try:
            obj_name = objid_2_objname("glassmolder")[obj_id]
            model_path = os.path.abspath(
                f'/mnt/disk0/dataset/transtouch_pc2_2/model_obj/obj_file/{obj_name}.obj')
            mesh = o3d.io.read_triangle_mesh(model_path)


            max_bound = mesh.get_max_bound()
            min_bound = mesh.get_min_bound()

            obj_diameter = np.linalg.norm(max_bound-min_bound)

            result = np.array(add_s_indi[obj_id])[:, 0] / (obj_diameter)
        except:
            print(f"{obj_name} missed.")
            continue
        auc, X, Y = Utils.compute_auc_sklearn(result, 1, 0.001)
        plt.plot(X, Y, label=f"Obj {obj_name}(auc:{auc})")

    plt.legend()
    plt.show()

def visualize_pcd_activezero(result_pred,result_gt ,img_name, gt = True):

    mask_all = cv2.imread(f"/mnt/disk0/dataset/rand_scenes/{img_name}/label.png",
                          cv2.IMREAD_UNCHANGED)
    obj_list = np.unique(mask_all)
    obj_list = obj_list[obj_list != 17]
    obj_list = obj_list[obj_list != 18]
    obj_list = obj_list[obj_list != 19]
    obj_meshs = []
    for obj_id in obj_list:
        obj_mesh_path = os.path.abspath(f'/mnt/disk0/dataset/bbox_norm/models/{objid_2_objname()[obj_id]}/visual_mesh.obj')
        obj_mesh = o3d.io.read_triangle_mesh(obj_mesh_path)
        pts = obj_mesh.sample_points_uniformly(number_of_points=2000)
        depth = cv2.imread(f"/mnt/disk0/dataset/rand_scenes/{img_name}/1024_depth_real.png", cv2.IMREAD_UNCHANGED)
        depth = cv2.resize(depth, dsize=(640,360), fx=1, fy=1, interpolation=cv2.INTER_LINEAR_EXACT) / 1000
        depth[depth > 3] = 0

        path = f"/mnt/disk0/dataset/rand_scenes/{img_name}/meta.pkl"
        meta_dict = np.load(path, allow_pickle=True)
        # print(meta_dict)
        K = meta_dict["intrinsic"]
        K[:2,:] /= 3
        scene_pts = depth_2_pts(depth,K)
        # print(pts.get_max_bound())
        R_pred, t_pred = find_according_imid_objid(result_pred, img_name_2_img_id()[img_name], obj_id)
        R_pred = R_transform(R_pred)
        t_pred = t_transform(t_pred)
        T_pred_temp = np.eye(4)
        T_pred_temp[:3, :3] = R_pred
        T_pred_temp[:3, 3] = t_pred / 1000
        obj_mesh.transform(T_pred_temp)
        obj_meshs.append(obj_mesh)

        if gt:

            R_gt, t_gt = find_according_imid_objid(result_gt, img_name_2_img_id()[img_name], obj_id)
            R_gt = R_transform(R_gt)
            t_gt = t_transform(t_gt)
            T_gt_temp = np.eye(4)
            T_gt_temp[:3, :3] = R_gt
            T_gt_temp[:3, 3] = t_gt / 1000
            obj_mesh_path = os.path.abspath(
                f'/mnt/disk0/dataset/bbox_norm/models/{objid_2_objname()[obj_id]}/visual_mesh.obj')
            obj_mesh = o3d.io.read_triangle_mesh(obj_mesh_path)
            obj_mesh.transform(T_gt_temp)
            obj_mesh.paint_uniform_color([1, 0, 0])
            obj_meshs.append(obj_mesh)

    obj_meshs.append(scene_pts)


    o3d.visualization.draw_geometries([obj for obj in obj_meshs])
    # o3d.visualization.draw_geometries([scene_pts])\
import open3d as o3d
def generate_bbox_glassmolder(result_pred,result_gt, add_s_all_dict,save_path):


    for obj_id in tqdm(range(46)):
        try:

            model_path = os.path.abspath(
                f'/mnt/disk0/dataset/transtouch_pc2_2/model_obj/obj_file/{objid_2_objname("glassmolder")[obj_id]}.obj')
            mesh = trimesh.load(model_path)

            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)

            to_origin[:3, 3] *= 1000
            bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3) * 1000
        except Exception as e:
            print(e)
            print(f"{objid_2_objname('glassmolder')[obj_id]} missed.")
            continue
        for img_id in tqdm(range(567)):
            try:
                path = f"/mnt/disk0/dataset/transtouch_pc2_hcp/dataset_render_0_to_20_hcp/{img_id_2_img_name()[img_id]}/meta.pkl"
                meta_dict = np.load(path, allow_pickle=True)
                # print(meta_dict)
                K = meta_dict["intrinsic_rgb"][:3,:3]
                # K[:2, :] /= 3
                # "/mnt/disk0/dataset/rand_scenes/0-300002-1/1024_rgb_real.png"
                color = cv2.imread(f"/mnt/disk0/dataset/transtouch_pc2_hcp/dataset_render_0_to_20_hcp/{img_id_2_img_name()[img_id]}/rgb.png",cv2.IMREAD_UNCHANGED)
                # print(result_pred[img_id][obj_id]["R"])

                R_gt = np.concatenate((R_transform(result_gt[img_id][obj_id]["R"]), np.zeros((1, 3))), axis=0)
                t_gt = np.concatenate((t_transform(result_gt[img_id][obj_id]["t"]), np.ones(1)), axis=0)
                pose_gt = np.concatenate((R_gt, np.expand_dims(t_gt, 1)), axis=1)
                center_pose_gt = pose_gt @ np.linalg.inv(to_origin)
                vis = Utils.draw_posed_3d_box(K, img=color, ob_in_cam=center_pose_gt, bbox=bbox,line_color=(0,0,255))

                R = np.concatenate((R_transform(result_pred[img_id][obj_id]["R"]),np.zeros((1,3))),axis=0)
                t = np.concatenate((t_transform(result_pred[img_id][obj_id]["t"]),np.ones(1)),axis=0)
                pose = np.concatenate((R,np.expand_dims(t,1)),axis=1)
                center_pose = pose @ np.linalg.inv(to_origin)
                vis = Utils.draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = Utils.draw_xyz_axis(vis, ob_in_cam=center_pose, scale=100, K=K, thickness=3, transparency=0,is_input_rgb=True)
                OBJ = objid_2_objname("glassmolder")[add_s_all_dict[img_id][obj_id][2]]
                cv2.putText(vis, f"OBJ: {OBJ}", (45, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 200), 2)
                img_name = img_id_2_img_name()[add_s_all_dict[img_id][obj_id][1]]
                cv2.putText(vis, f"Image name: {img_name}", (45, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 200), 2)
                cv2.putText(vis, f"ADD-s: {add_s_all_dict[img_id][obj_id][0] * 1000:.2f} mm", (45, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 200), 2)


                # cv2.imshow('1', vis)
                # cv2.waitKey()
                obj_name = objid_2_objname("glassmolder")[obj_id]
                save_path_obj = os.path.join(save_path,f"{obj_name}")
                if not os.path.exists(save_path_obj):
                    os.mkdir(save_path_obj)

                cv2.imwrite(os.path.join(save_path_obj,f"{img_id}.png"),vis)
                # cv2.waitKey(1000)
            except Exception as e:
                print(e)
                print(img_id)
                # vis = color
                # cv2.imshow('1', vis)
                # cv2.waitKey(1)
def generate_add_s_npy_activezero(result_gt,result_pred,save_path):
    '''
    Calculate and save add_s metrics of pred pose given gt pose and pred pose.

    Input:
        :dict{img_id:{obj_id:{[R],[t]}}}
        :dict{img_id:{obj_id:{[R],[t]}}}
        :string

    Output:
        None
    '''
    add_s_list_all = []
    add_s_list_indi = {}

    add_s_dict_all = {}
    for img_id in range(441):
        add_s_dict_all[img_id] = {}

    for obj_id in range(17):
        add_s_list_indi[obj_id] = []


    # for image_id in tqdm(range(1214)):
    #     for obj_id in [1, 5, 6, 8, 9, 10, 11, 12]:
    #         try:
    for img_id in range(411):
        for obj_id in range(17):
            try:

                R_gt, t_gt = find_according_imid_objid(result_gt, img_id, obj_id)
                R_gt = R_transform(R_gt)
                t_gt = t_transform(t_gt)


                R_pred, t_pred = find_according_imid_objid(result_pred, img_id, obj_id)
                R_pred = R_transform(R_pred)
                t_pred = t_transform(t_pred)

            except Exception as e:
                print(e)
                print(f"Img{img_id}_Obj{obj_id} missed.")
                continue

            model_path = os.path.abspath(
                f'/mnt/disk0/dataset/bbox_norm/models/{objid_2_objname()[obj_id]}/visual_mesh.obj')
            add_s = cal_add_s(model_path, R_gt, t_gt, R_pred, t_pred)
            add_s = [add_s,img_id,obj_id]
            add_s_list_all.append(add_s)
            add_s_list_indi[obj_id].append(add_s)

            add_s_dict_all[img_id][obj_id] = add_s

                # except Exception as e:
                #     print(e)
                #     print(f"img:{image_id}-obj:{obj_id} missed.")
    print(add_s_list_all)
    print(add_s_dict_all)
    np.save(os.path.join(save_path, "add_s_all_list.npy"), add_s_list_all)
    np.save(os.path.join(save_path, "add_s_indi_dict.npy"), add_s_list_indi)
    np.save(os.path.join(save_path, "add_s_all_dict.npy"), add_s_dict_all)


def generate_add_s_npy_glassmolder(result_gt, result_pred, save_path):
    '''
    Calculate and save add_s metrics of pred pose given gt pose and pred pose.

    Input:
        :dict{img_id:{obj_id:{[R],[t]}}}
        :dict{img_id:{obj_id:{[R],[t]}}}
        :string

    Output:
        None
    '''
    add_s_list_all = []
    add_s_list_indi = {}

    add_s_dict_all = {}
    for img_id in range(567):
        add_s_dict_all[img_id] = {}

    for obj_id in range(46):
        add_s_list_indi[obj_id] = []

    # for image_id in tqdm(range(1214)):
    #     for obj_id in [1, 5, 6, 8, 9, 10, 11, 12]:
    #         try:
    for img_id in range(567):
        for obj_id in range(46):
            try:

                R_gt, t_gt = find_according_imid_objid(result_gt, img_id, obj_id)
                R_gt = R_transform(R_gt)
                t_gt = t_transform(t_gt)

                R_pred, t_pred = find_according_imid_objid(result_pred, img_id, obj_id)
                R_pred = R_transform(R_pred)
                t_pred = t_transform(t_pred)

            except Exception as e:
                print(e)
                print(f"Img{img_id}_Obj{obj_id} missed.")
                continue

            model_path = os.path.abspath(
                f'/mnt/disk0/dataset/transtouch_pc2_2/model_obj/obj_file/{objid_2_objname("glassmolder")[obj_id]}.obj')
            t_gt /= 1000
            t_pred /= 1000
            add_s = cal_add_s(model_path, R_gt, t_gt, R_pred, t_pred)
            add_s = [add_s, img_id, obj_id]
            add_s_list_all.append(add_s)
            add_s_list_indi[obj_id].append(add_s)

            add_s_dict_all[img_id][obj_id] = add_s

            # except Exception as e:
            #     print(e)
            #     print(f"img:{image_id}-obj:{obj_id} missed.")
    print(add_s_list_all)
    print(add_s_dict_all)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    np.save(os.path.join(save_path, "add_s_all_list.npy"), add_s_list_all)
    np.save(os.path.join(save_path, "add_s_indi_dict.npy"), add_s_list_indi)
    np.save(os.path.join(save_path, "add_s_all_dict.npy"), add_s_dict_all)
def metrics_indi_activezero(add_s_indi,k,save_path):
    '''
    Return summary of metrics according to add_s with the format of markdown.
    Generate max_k and min_K summary table.
    Generate max_k and min_k pictures and place them into the generated folders.

    Input:
        :add_s_indi
        :int, to specific max_k and min_K
        :string

    Output:
        None
    '''
    for obj_id in range(17):
        model_path = os.path.abspath(
            f'/mnt/disk0/dataset/bbox_norm/models/{objid_2_objname()[obj_id]}/visual_mesh.obj')
        mesh = o3d.io.read_triangle_mesh(model_path)
        max_bound = mesh.get_max_bound()
        min_bound = mesh.get_min_bound()
        obj_diameter = np.linalg.norm(max_bound - min_bound) * 1000
        add_s_indi_list = add_s_indi[obj_id]
        try:

            max_topk_id = max_k(add_s_indi_list,k)

        except:
            print(f"{objid_2_objname()[obj_id]} omit.")
            continue
        print(f"Obj {objid_2_objname()[obj_id]} top {k}:")
        max_k_list = np.array([add_s_indi_list[i] for i in max_topk_id])
        #sort by add_s
        max_k_list = max_k_list[np.argsort(max_k_list[:,0],)[::-1]]
        max_k_list_dia = copy.deepcopy(max_k_list)
        max_k_list_dia[:,0] /= obj_diameter
        # print(max_k_list)
        # print([add_s_indi_list[i] for i in max_topk_id])
        indi_table = []
        indi_image = []
        indi_image_orig = []
        indi_image_render = []
        headers = ["Obj","Top_i","Img_id","Error(mm)","Error(Obj_diameter)","Bbox","Image","Render"]
        for i in range(k):
            # indi_table.append([f"{ob_id_to_names[obj_id]}",i,max_k_list[i][0],max_k_list_dia[i][0],max_k_list_dia[i][1]])
            indi_table.append(
                [f"{objid_2_objname()[obj_id]}", i, int(max_k_list_dia[i][1]),max_k_list[i][0], max_k_list_dia[i][0], f"../../../activezero_test/obj_{obj_id}/{int(max_k_list_dia[i][1])}.png",f"../../../../../../../dataset/rand_scenes/{img_id_2_img_name()[int(max_k_list_dia[i][1])]}/1024_rgb_real.png"])
            indi_image.append(cv2.imread(f"/mnt/disk0/pzh/foundationpose/FoundationPose-main/debug/activezero_test/obj_{obj_id}/{int(max_k_list_dia[i][1])}.png"))
            indi_image_orig.append(cv2.imread(f"/mnt/disk0/dataset/rand_scenes/{img_id_2_img_name()[int(max_k_list_dia[i][1])]}/1024_rgb_real.png"))
            # indi_image_render.append(
            #     cv2.imread(f"/mnt/disk0/dataset/BOP/lmo/test/000002/rgb_with_render_obj_visib/{int(max_k_list_dia[i][1]):06d}_{obj_id:06d}.png"))

        # s = tabulate(indi_table,headers,floatfmt=".3f",tablefmt="fancy_grid")

        save_path_indi = os.path.join(save_path,f"Obj_{objid_2_objname()[obj_id]}")
        if not os.path.exists(save_path_indi):
            os.mkdir(save_path_indi)
        save_path_indi2 = os.path.join(save_path_indi,"max")
        if not os.path.exists(save_path_indi2):
            os.mkdir(save_path_indi2)
        for i in range(k):
            cv2.imwrite(os.path.join(save_path_indi2,f"{objid_2_objname()[obj_id]}_max_{i}.png"),indi_image[i])
            cv2.imwrite(os.path.join(save_path_indi2, f"{objid_2_objname()[obj_id]}_max_{i}_org.png"), indi_image_orig[i])
            # cv2.imwrite(os.path.join(save_path_indi, f"{objid_2_objname()[obj_id]}_max_{i}_render.png"), indi_image_render[i])


        # with open(os.path.join(save_path_indi,f"{ob_id_to_names[obj_id]}_max{k}_summary.txt"),"w") as f:
        #     f.write(s)
        path_bbox = []
        path_orig = []
        path_render = []
        for i in range(k):
            path_bbox.append(f"max/{objid_2_objname()[obj_id]}_max_{i}.png")
            path_orig.append(f"max/{objid_2_objname()[obj_id]}_max_{i}_org.png")
            # path_render.append(f"max/{objid_2_objname()[obj_id]}_max_{i}_render.png")
        write_markdown(headers,indi_table,save_path_indi,path_bbox,path_orig,path_render,"max")

        ##Minimize:
        min_topk_id = min_k(add_s_indi_list, k)
        min_k_list = np.array([add_s_indi_list[i] for i in min_topk_id])
        # sort by add_s
        min_k_list = min_k_list[np.argsort(min_k_list[:, 0], )]
        min_k_list_dia = copy.deepcopy(min_k_list)
        min_k_list_dia[:, 0] /= obj_diameter

        indi_table = []
        indi_image = []
        indi_image_orig = []
        indi_image_render = []



        headers = ["Obj", "Top_i", "Img_id","Error(mm)", "Error(Obj_diameter)", "Img_id"]
        for i in range(k):
            indi_table.append(
                [f"{objid_2_objname()[obj_id]}", i, int(min_k_list_dia[i][1]), min_k_list[i][0], min_k_list_dia[i][0],
                 f"../../../activezero_test/obj_{obj_id}/{int(min_k_list_dia[i][1])}.png",
                 f"../../../../../../../dataset/rand_scenes/{img_id_2_img_name()[int(min_k_list_dia[i][1])]}/1024_rgb_real.png"])
            indi_image.append(cv2.imread(
                f"/mnt/disk0/pzh/foundationpose/FoundationPose-main/debug/activezero_test/obj_{obj_id}/{int(min_k_list_dia[i][1])}.png"))
            indi_image_orig.append(cv2.imread(
                f"/mnt/disk0/dataset/rand_scenes/{img_id_2_img_name()[int(min_k_list_dia[i][1])]}/1024_rgb_real.png"))
            # indi_image_render.append(
            #     cv2.imread(
            #         f"/mnt/disk0/dataset/BOP/lmo/test/000002/rgb_with_render_obj_visib/{int(max_k_list_dia[i][1]):06d}_{obj_id:06d}.png"))

        # s = tabulate(indi_table, headers, floatfmt=".3f", tablefmt="fancy_grid")

        save_path_indi = os.path.join(save_path, f"Obj_{objid_2_objname()[obj_id]}")
        if not os.path.exists(save_path_indi):
            os.mkdir(save_path_indi)
        save_path_indi2 = os.path.join(save_path_indi, "min")
        if not os.path.exists(save_path_indi2):
            os.mkdir(save_path_indi2)

        for i in range(k):
            cv2.imwrite(os.path.join(save_path_indi2, f"{objid_2_objname()[obj_id]}_min_{i}.png"), indi_image[i])
            cv2.imwrite(os.path.join(save_path_indi2, f"{objid_2_objname()[obj_id]}_min_{i}_org.png"), indi_image_orig[i])
            # cv2.imwrite(os.path.join(save_path_indi, f"{objid_2_objname()[obj_id]}_min_{i}_render.png"),
            #             indi_image_render[i])

        # with open(os.path.join(save_path_indi, f"{ob_id_to_names[obj_id]}_min{k}_summary.txt"), "w") as f:
        #     f.write(s)
        path_bbox = []
        path_orig = []
        path_render = []
        for i in range(k):
            path_bbox.append(f"min/{objid_2_objname()[obj_id]}_min_{i}.png")
            path_orig.append(f"min/{objid_2_objname()[obj_id]}_min_{i}_org.png")
            # path_render.append(f"{objid_2_objname()[obj_id]}_min_{i}_render.png")
        # headers = ["Obj", "Top_i", "Error(mm)", "Error(Obj_diameter)", "Bbox", "Image"]
        write_markdown(headers, indi_table, save_path_indi, path_bbox, path_orig,path_render,"min")

def metrics_indi_glassmolder(add_s_indi,k,save_path):
    '''
    Return summary of metrics according to add_s with the format of markdown.
    Generate max_k and min_K summary table.
    Generate max_k and min_k pictures and place them into the generated folders.

    Input:
        :add_s_indi
        :int, to specific max_k and min_K
        :string

    Output:
        None
    '''
    for obj_id in range(46):
        try:
            model_path = os.path.abspath(
                f'/mnt/disk0/dataset/transtouch_pc2_2/model_obj/obj_file/{objid_2_objname("glassmolder")[obj_id]}.obj')
            mesh = o3d.io.read_triangle_mesh(model_path)
            max_bound = mesh.get_max_bound()
            min_bound = mesh.get_min_bound()
            obj_diameter = np.linalg.norm(max_bound - min_bound)
            add_s_indi_list = add_s_indi[obj_id]
        except:
            print(f"{objid_2_objname('glassmolder')[obj_id]} omit.")
            continue
        try:

            max_topk_id = max_k(add_s_indi_list,k)

        except:
            obj_name = objid_2_objname("glassmolder")[obj_id]
            print(f"{obj_name} omit.")
            continue
        obj_name = objid_2_objname("glassmolder")[obj_id]
        print(f"Obj {obj_name} top {k}:")
        max_k_list = np.array([add_s_indi_list[i] for i in max_topk_id])
        #sort by add_s
        max_k_list = max_k_list[np.argsort(max_k_list[:,0],)[::-1]]
        max_k_list_dia = copy.deepcopy(max_k_list)
        max_k_list_dia[:,0] /= obj_diameter
        # print(max_k_list)
        # print([add_s_indi_list[i] for i in max_topk_id])
        indi_table = []
        indi_image = []
        indi_image_orig = []
        indi_image_render = []

        save_path2 = os.path.join(save_path, "top_k")
        if not os.path.exists(save_path2):
            os.mkdir(save_path2)
        obj_name = objid_2_objname("glassmolder")[obj_id]
        save_path_indi = os.path.join(save_path2,f"{obj_name}")
        if not os.path.exists(save_path_indi):
            os.mkdir(save_path_indi)
        save_path_indi2 = os.path.join(save_path_indi,"max")
        if not os.path.exists(save_path_indi2):
            os.mkdir(save_path_indi2)
        for i in range(k):
            img_id = int(max_k_list_dia[i][1])
            img_origin = cv2.imread(f"/mnt/disk0/dataset/transtouch_pc2_hcp/dataset_render_0_to_20_hcp/{img_id_2_img_name()[img_id]}/rgb.png")
            img_bbox = cv2.imread(f"{save_path}/{obj_name}/{img_id}.png")
            cv2.imwrite(os.path.join(save_path_indi2,f"{obj_name}_max_{i}.png"),img_bbox)
            cv2.imwrite(os.path.join(save_path_indi2, f"{obj_name}_max_{i}_org.png"), img_origin)

        path_bbox = []
        path_orig = []
        path_render = []
        for i in range(k):
            path_bbox.append(f"max/{obj_name}_max_{i}.png")
            path_orig.append(f"max/{obj_name}_max_{i}_org.png")
            # path_render.append(f"max/{objid_2_objname()[obj_id]}_max_{i}_render.png")

        headers = ["Obj", "Top_i", "Img_id", "Error(mm)", "Error(Obj_diameter)", "Bbox","Orig"]
        for i in range(k):

            indi_table.append(
                [f"{obj_name}", i, int(max_k_list_dia[i][1]),max_k_list[i][0], max_k_list_dia[i][0], path_bbox[i],path_orig[i]])



        write_markdown(headers,indi_table,save_path_indi,path_bbox,path_orig,path_render,"max")

        ##Minimize:
        min_topk_id = min_k(add_s_indi_list, k)
        min_k_list = np.array([add_s_indi_list[i] for i in min_topk_id])
        # sort by add_s
        min_k_list = min_k_list[np.argsort(min_k_list[:, 0], )]
        min_k_list_dia = copy.deepcopy(min_k_list)
        min_k_list_dia[:, 0] /= obj_diameter

        indi_table = []



        headers = ["Obj", "Top_i", "Img_id","Error(mm)", "Error(Obj_diameter)", "Img_id"]


        if not os.path.exists(save_path_indi):
            os.mkdir(save_path_indi)
        save_path_indi2 = os.path.join(save_path_indi, "min")
        if not os.path.exists(save_path_indi2):
            os.mkdir(save_path_indi2)

        for i in range(k):
            img_id = int(max_k_list_dia[i][1])
            img_origin = cv2.imread(
                f"/mnt/disk0/dataset/transtouch_pc2_hcp/dataset_render_0_to_20_hcp/{img_id_2_img_name()[img_id]}/rgb.png")
            img_bbox = cv2.imread(f"{save_path}/{obj_name}/{img_id}.png")
            cv2.imwrite(os.path.join(save_path_indi2, f"{obj_name}_min_{i}.png"), img_bbox)
            cv2.imwrite(os.path.join(save_path_indi2, f"{obj_name}_min_{i}_org.png"), img_origin)

        path_bbox = []
        path_orig = []
        path_render = []
        for i in range(k):
            path_bbox.append(f"min/{obj_name}_min_{i}.png")
            path_orig.append(f"min/{obj_name}_min_{i}_org.png")

        headers = ["Obj", "Top_i", "Img_id", "Error(mm)", "Error(Obj_diameter)", "Bbox", "Orig"]
        for i in range(k):
            indi_table.append(
                [f"{obj_name}", i, int(min_k_list_dia[i][1]), min_k_list[i][0], min_k_list_dia[i][0], path_bbox[i],
                 path_orig[i]])
        write_markdown(headers, indi_table, save_path_indi, path_bbox, path_orig,path_render,"min")
