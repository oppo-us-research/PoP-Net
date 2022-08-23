import copy
import logging
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2


jointColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
               [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
               [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [85, 255, 85]]

root_joint = 'torso'

intrinsics = {'fx': 504.1189880371094, 'fy': 504.042724609375, 'cx': 231.7421875, 'cy': 320.62640380859375}

depth_mean = 3
depth_std = 2
depth_max = 6
joint2box_margin = 25


def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('torso'), keypoints.index('right_hip')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('torso'), keypoints.index('left_hip')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('torso'), keypoints.index('neck')],
        [keypoints.index('neck'), keypoints.index('right_shoulder')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('neck'), keypoints.index('left_shoulder')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('neck'), keypoints.index('head')]
    ]
    return kp_lines


def get_keypoints():
    """Get the itop keypoints"""
    keypoints = [
        'head',
        'neck',
        'right_shoulder',
        'left_shoulder',
        'right_elbow',
        'left_elbow',
        'right_wrist',
        'left_wrist',
        'torso',
        'right_hip',
        'left_hip',
        'right_knee',
        'left_knee',
        'right_ankle',
        'left_ankle']
    return keypoints


def draw_humans(img, humans, limbs, jointColors, visibilities=None):
    visibilities = visibilities or None
    for i, human in enumerate(humans):
        human_vis = np.array(human)
        for k, limb in enumerate(limbs):
            if visibilities is not None and visibilities[i][limb[0]] < 0.5:
                continue
            center1 = human_vis[limb[0], :2].astype(np.int)
            img = cv2.circle(img, tuple(center1), 3, [0, 0, 255], thickness=2, lineType=8, shift=0)

            if visibilities is not None and visibilities[i][limb[1]] < 0.5:
                continue
            center2 = human_vis[limb[1], :2].astype(np.int)
            img = cv2.line(img, tuple(center1), tuple(center2), jointColors[k], 2)
            img = cv2.circle(img, tuple(center2), 3, [0, 0, 255], thickness=2, lineType=8, shift=0)

    return img


if __name__ == "__main__":

    DATA_DIR = '/media/yuliang/DATA/Datasets/Kinect_Depth_Human3D/multiperson_test_v2'
    img_dir = os.path.join(DATA_DIR, 'depth_maps')
    ann_file = '{}/labels_test.json'.format(os.path.join(DATA_DIR, 'labels'))
    eval_file = '/media/yuliang/DATA/Projects/Realtime_MultiPerson_3D_Pose_Estimation/predictions/pop_net_kdh3d_mpaug_mpreal_0903/eval_data.json'
    our_dir = os.path.join(DATA_DIR, 'vis_pop')
    if not os.path.exists(our_dir):
        os.mkdir(our_dir)

    anno_dic = json.load(open(ann_file, 'r'))
    anno_ids = [key for key, value in anno_dic.items() if key != 'intrinsics']

    eval_data = json.load(open(eval_file, 'r'))
    pred_2d_set = eval_data['human_pred_set_2d']
    pred_3d_set = eval_data['human_pred_set_3d']

    cnt = 0
    for i in range(len(pred_2d_set)):
        print('{}/{}'.format(i, len(pred_2d_set)))

        image_id = anno_ids[i]
        single_img = np.load(os.path.join(img_dir, image_id)).astype(np.float)

        humans_2d = pred_2d_set[i]
        humans_3d = pred_3d_set[i]

        single_img[single_img <= 0] = 0
        single_img[single_img >= depth_max] = depth_max
        single_img /= depth_max
        single_img *= 255
        single_img = cv2.cvtColor(single_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        single_img = draw_humans(single_img,
                                 humans_2d,
                                 kp_connections(get_keypoints()),
                                 jointColors)
        cv2.imwrite(os.path.join(our_dir, '{:06d}.jpg'.format(cnt)), single_img)
        cnt += 1

