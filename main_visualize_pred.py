import os
import numpy as np
import json
import cv2
from util import util_functions


if __name__ == "__main__":
    DATA_DIR = 'dataset/test_mpreal/'
    img_dir = os.path.join(DATA_DIR, 'depth_maps')
    ann_file = 'labels/test_mpreal/labels.json'
    res_file = 'predictions/test_mpreal/pop_results.json'
    our_dir = 'vis_pred'
    if not os.path.exists(our_dir):
        os.mkdir(our_dir)

    anno_dic = json.load(open(ann_file, 'r'))
    anno_ids = [key for key, value in anno_dic.items() if key != 'intrinsics']
    kp_connections = util_functions.kp_connections(util_functions.get_keypoints())

    res_data = json.load(open(res_file, 'r'))
    if 'pop' in res_file:
        pred_2d_set = res_data['human_pred_set_2d_aligned']
        pred_3d_set = res_data['human_pred_set_3d_aligned']
    else:
        pred_2d_set = res_data['human_pred_set_2d']
        pred_3d_set = res_data['human_pred_set_3d']

    cnt = 0
    for i in range(len(pred_2d_set)):
        print('{}/{}'.format(i, len(pred_2d_set)))

        image_id = anno_ids[i]
        single_img = np.load(os.path.join(img_dir, image_id)).astype(np.float)

        humans_2d = pred_2d_set[i]
        humans_3d = pred_3d_set[i]

        single_img[single_img <= 0] = 0
        single_img[single_img >= util_functions.depth_max] = util_functions.depth_max
        single_img /= util_functions.depth_max
        single_img *= 255
        single_img = cv2.cvtColor(single_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        single_img = util_functions.draw_humans(single_img,
                                                humans_2d,
                                                kp_connections,
                                                util_functions.jointColors)
        cv2.imwrite(os.path.join(our_dir, '{:06d}.jpg'.format(cnt)), single_img)
        cnt += 1

