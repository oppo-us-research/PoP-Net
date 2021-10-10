import os
import numpy as np
import json
import cv2
from util import util_functions


if __name__ == "__main__":
    DATA_DIR = 'dataset/train_val/'
    img_dir = os.path.join(DATA_DIR, 'depth_maps')
    seg_dir = os.path.join(DATA_DIR, 'seg_maps')
    ann_file = 'labels/train_val/labels_test.json'
    our_dir = 'vis_gt'
    if not os.path.exists(our_dir):
        os.mkdir(our_dir)

    anno_dic = json.load(open(ann_file, 'r'))
    anno_ids = [key for key, value in anno_dic.items() if key != 'intrinsics']
    kp_connections = util_functions.kp_connections(util_functions.get_keypoints())

    cnt = 0
    for i in range(len(anno_ids)):
        print('{}/{}'.format(i, len(anno_ids)))

        image_id = anno_ids[i]
        single_img = np.load(os.path.join(img_dir, image_id)).astype(np.float)

        humans_2d = []
        humans_3d = []
        for ann in anno_dic[image_id]:
            humans_2d.append(ann['2d_joints'])
            humans_3d.append(ann['3d_joints'])

        single_img[single_img <= 0] = 0
        single_img[single_img >= util_functions.depth_max] = util_functions.depth_max
        single_img /= util_functions.depth_max
        single_img *= 255
        single_img = cv2.cvtColor(single_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # superimpose mask for visualization if exists
        if os.path.exists(seg_dir):
            seg_map = np.load(os.path.join(seg_dir, image_id)).astype(np.float)
            single_img[:, :, 2] = single_img[:, :, 2] + seg_map * 50
            single_img[single_img > 255] = 255

        single_img = util_functions.draw_humans(single_img,
                                                humans_2d,
                                                kp_connections,
                                                util_functions.jointColors)
        cv2.imwrite(os.path.join(our_dir, '{:06d}.jpg'.format(cnt)), single_img)
        cnt += 1

