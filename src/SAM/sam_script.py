import numpy as np
import cv2
import os
import copy
import argparse

from segment_anything import sam_model_registry, SamPredictor



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_in_path", type=str, default='/home/rvsa/gary318/build_kinematic/input_rgbd/010602_fridge/raw_rgb')
    parser.add_argument("--img_out_path", type=str, default='/home/rvsa/gary318/build_kinematic/input_rgbd/010602_fridge/mask')
    parser.add_argument("--sam_checkpoint", type=str, default='/home/rvsa/SAM/sam_vit_h_4b8939.pth')
    parser.add_argument("--model_type", type=str, default='vit_h')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--thickness_circle", type=int, default=1)
    parser.add_argument("--thickness_box", type=int, default=2)
    parser.add_argument("--mix_rate", type=float, default=0.5)
    parser.add_argument("--box_points_scale", type=float, default=0.25)
    parser.add_argument("--use_bbox", type=int, default=1)
    parser.add_argument("--use_points", type=int, default=0)
    opt = parser.parse_args()
    return opt



def draw_box(img, point1, point2, color, thickness = 1):
    point3 = (point1[0], point2[1])
    point4 = (point2[0], point1[1])
    img = cv2.line(img, point1, point3, color, thickness=thickness)
    img = cv2.line(img, point3, point2, color, thickness=thickness)
    img = cv2.line(img, point2, point4, color, thickness=thickness)
    img = cv2.line(img, point4, point1, color, thickness=thickness)
    return img


def call_back_box(event, x, y, flags, param):
    img = copy.deepcopy(param['image0'])
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(param['box_points']) < 2:
            param['box_points'].append((x, y))
        if len(param['box_points']) == 1:
            param['drawing'] = True
        else:
            param['drawing'] = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        param['box_points'].clear()
        param['drawing'] = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if param['drawing']:
            if len(param['box_points']) == 1:
                draw_box(img, param['box_points'][0], (x, y), (0, 0, 255), param['thickness'])
            else:
                param['drawing'] = False
    if len(param['box_points']) == 2:
        draw_box(img, param['box_points'][0], param['box_points'][1], (0, 255, 0), param['thickness'])
    param['image'] = img
    if event == cv2.EVENT_MBUTTONDOWN:
        param['checked'] = True


def draw_circle_cross(img, center, radius, color, thickness = 1, horizontal = True, vertical = True):
    img = cv2.circle(img, center, radius, color, thickness=thickness)
    radius_ = radius - 2
    if radius_ > 0:
        if horizontal:
            img = cv2.line(img, (center[0]-radius_, center[1]), (center[0]+radius_, center[1]), color, thickness=thickness)
        if vertical:
            img = cv2.line(img, (center[0], center[1]-radius_), (center[0], center[1]+radius_), color, thickness=thickness)
    return img


def call_back_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        del_radius = param['del_radius']
        to_del = []
        for i, p in enumerate(param['points_neg']):
            if (p[0] > x - del_radius) & (p[0] < x + del_radius) & (p[1] > y - del_radius) & (p[1] < y + del_radius):
                to_del.append(i)
        if len(to_del) > 0:
            for i in to_del[::-1]:
                del param['points_neg'][i]
        else:
            param['points_pos'].append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        del_radius = param['del_radius']
        to_del = []
        for i, p in enumerate(param['points_pos']):
            if (p[0] > x - del_radius) & (p[0] < x + del_radius) & (p[1] > y - del_radius) & (p[1] < y + del_radius):
                to_del.append(i)
        if len(to_del) > 0:
            for i in to_del[::-1]:
                del param['points_pos'][i]
        else:
            param['points_neg'].append((x, y))
    elif event == cv2.EVENT_MBUTTONDOWN:
        param['checked'] = True
    img = copy.deepcopy(param['image0'])
    for xy in param['points_pos']:
        draw_circle_cross(img, xy, param['radius'], (0, 255, 0), thickness = param['thickness'], horizontal = True, vertical = True)
    for xy in param['points_neg']:
        draw_circle_cross(img, xy, param['radius'], (0, 0, 255), thickness = param['thickness'], horizontal = True, vertical = False)
    param['image'] = img


def select_mask(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for i in range(4):
            if (x > param['range'][4*i]) & (x < param['range'][4*i+1]) & (y > param['range'][4*i+2]) & (y < param['range'][4*i+3]):
                param['mask_id'] = i

def main():
    opt = parse_args()
    thickness_circle = opt.thickness_circle
    thickness_box = opt.thickness_box
    radius = opt.radius
    img_in_path = opt.img_in_path
    img_out_path = opt.img_out_path
    mix_rate = opt.mix_rate
    sam_checkpoint = opt.sam_checkpoint
    model_type = opt.model_type
    device = opt.device
    box_points_scale = opt.box_points_scale
    if opt.use_bbox > 0:
        use_bbox = True
    else:
        use_bbox = False
    if opt.use_points > 0:
        use_points = True
    else:
        use_points = False

    del_radius = 20
    mask_show_scale = 0.5

    if box_points_scale <= 0:
        box_points_scale = 1
    if mask_show_scale <= 0:
        mask_show_scale = 1
 
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    if os.path.isdir(img_in_path):
        img_list = os.listdir(img_in_path)
        # img_list.sort(key=lambda x: x[0] + "%05d" % int(x[2:x.find('.')]))
    else:
        img_list = [img_in_path]

    if not os.path.exists(img_out_path):
        os.makedirs(img_out_path)

    curr_img_index = 0
    mask_bbox = None
    while curr_img_index < len(img_list):
        img_ext = os.path.splitext(img_list[curr_img_index])[1]
        img_name: str = os.path.split(img_list[curr_img_index])[1]
        img_name = img_name[:-len(img_ext)]
        if img_ext not in ('.jpg', '.png', '.jpeg') or img_name.endswith('_mask'):
            curr_img_index += 1
            continue

        img_path = os.path.join(img_in_path, img_list[curr_img_index])

        image = cv2.imread(img_path).astype(np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_sam = copy.deepcopy(image)
        x_size = int(image.shape[1] * box_points_scale)
        y_size = int(image.shape[0] * box_points_scale)
        x_rate = image.shape[1] / x_size
        y_rate = image.shape[0] / y_size
        image = cv2.resize(image, (x_size, y_size))
        image0 = copy.deepcopy(image)
        image_sam = cv2.cvtColor(image_sam, cv2.COLOR_BGR2RGB)  # ?

        param_box = dict()
        param_box['box_points'] = []
        param_box['thickness'] = thickness_box

        param_points = dict()
        param_points['points_pos'] = []
        param_points['points_neg'] = []
        param_points['thickness'] = thickness_circle
        param_points['radius'] = radius
        param_points['del_radius'] = del_radius

        loop = True
        first_pass = True
        re_select = False
        while loop:
            img_bbox = copy.deepcopy(image)
            if use_bbox:
                if mask_bbox is not None and first_pass:
                    bbox = mask_bbox
                else:
                    param_box['drawing'] = False
                    param_box['image0'] = image
                    param_box['image'] = copy.deepcopy(image)
                    param_box['checked'] = False
                    cv2.namedWindow(f"box")
                    cv2.setMouseCallback(f"box", call_back_box, param_box)
                    while (not param_box['checked']):
                        cv2.imshow(f"box", param_box['image'])
                        key = cv2.waitKey(5) & 0xFF
                        if key == ord('q'):
                            break
                    cv2.destroyAllWindows()

                    bbox = param_box['box_points']
                    if len(bbox) == 2:
                        img_bbox = draw_box(img_bbox, bbox[0], bbox[1], (255, 255, 0), thickness=thickness_box)
                        bbox = np.array(bbox, dtype=np.float32)
                        bbox[:, 0] *= x_rate
                        bbox[:, 1] *= y_rate
                    else:
                        bbox = None
            else:
                bbox = None

            if use_points or re_select:
                param_points['checked'] = False
                param_points['image0'] = img_bbox
                param_points['image'] = copy.deepcopy(img_bbox)
                param_points['bbox'] = bbox
                cv2.namedWindow(f"points")
                cv2.setMouseCallback(f"points", call_back_points, param_points)
                while (not param_points['checked']):
                    cv2.imshow(f"points", param_points['image'])
                    key = cv2.waitKey(5) & 0xFF
                    if key == ord('q'):
                        break
                cv2.destroyAllWindows()

                points_pos = param_points['points_pos']
                points_neg = param_points['points_neg']
                points_pos = np.asarray(points_pos, dtype=np.float32)
                points_neg = np.asarray(points_neg, dtype=np.float32)
                input_label_pos = np.ones(points_pos.shape[0])
                input_label_neg = np.zeros(points_neg.shape[0])
                if points_pos.shape[0] > 0:
                    if points_neg.shape[0] > 0:
                        input_points = np.concatenate([points_pos, points_neg], axis=0)
                        input_label = np.concatenate([input_label_pos, input_label_neg], axis=0)
                    else:
                        input_points = points_pos
                        input_label = input_label_pos
                else:
                    if points_neg.shape[0] > 0:
                        input_points = points_neg
                        input_label = input_label_neg
                    else:
                        input_points = None
                        input_label = None

                if input_points is not None:
                    input_points[:, 0] *= x_rate
                    input_points[:, 1] *= y_rate
            else:
                input_points = None
                input_label = None


            predictor.set_image(image_sam)
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_label,
                box = bbox,
                multimask_output=True,
            )

            mask_color = np.asarray([255, 0, 0])
            masks_show_all = []
            for mi in masks:
                mi_ = mi.astype(np.uint8)
                mi_ = cv2.resize(mi_, (x_size, y_size))
                masks_show = copy.deepcopy(image0)
                masks_show[mi_ > 0] = masks_show[mi_ > 0] * (1 - mix_rate) + mask_color * mix_rate
                masks_show_all.append(masks_show)

            h_show = int(image.shape[0] * mask_show_scale)
            w_show = int(image.shape[1] * mask_show_scale)
            mask_show_all_rescale = []
            for mi in masks_show_all:
                mask_show_all_rescale.append(cv2.resize(mi, (w_show, h_show)))

            to_show = np.zeros((h_show * 2, w_show * 2, 3), dtype = np.uint8)
            to_show[:h_show, :w_show, :] = mask_show_all_rescale[0]
            to_show[:h_show:, w_show:, :] = mask_show_all_rescale[1]
            to_show[h_show:, :w_show, :] = mask_show_all_rescale[2]
            range_0 = [10, w_show - 10, 10, h_show - 10]
            range_1 = [w_show + 10, w_show * 2 - 10, 10, h_show - 10]
            range_2 = [10, w_show - 10, h_show + 10, h_show * 2 - 10]
            range_3 = [w_show + 10, w_show * 2 - 10, h_show + 10, h_show * 2 - 10]
            range_all = range_0 + range_1 + range_2 + range_3

            param = dict()
            param['range'] = range_all
            param['mask_id'] = -1
            cv2.namedWindow(f"mask")
            cv2.setMouseCallback(f"mask", select_mask, param)
            while(param['mask_id'] < 0):
                cv2.imshow(f"mask", to_show)
                key = cv2.waitKey(5) & 0xFF
                if key == ord('q'):
                    break
            cv2.destroyAllWindows()

            if param['mask_id'] in (0, 1, 2):
                mask_slct = masks[param['mask_id']]
                mask_coord = np.where(mask_slct > 0)
                mask_bbox = np.array([[min(mask_coord[1]) - 70, min(mask_coord[0]) - 70],
                                      [max(mask_coord[1]) + 70, max(mask_coord[0]) + 70]])
                mask_slct = mask_slct.astype(np.uint8) * 255
                cv2.imwrite(os.path.join(img_out_path, img_name + '.png'), mask_slct)
                loop = False
                re_select = False
            else:
                image = masks_show_all[1]
                re_select = True
                print('return to start')
            
            first_pass = False
        curr_img_index += 1

    print('done')


if __name__ == '__main__':
    main()