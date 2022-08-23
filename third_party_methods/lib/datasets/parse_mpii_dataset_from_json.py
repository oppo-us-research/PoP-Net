"""
@author: Yuliang Guo <33yuliangguo@gmail.com>

TODO: this code seems not include mp labels in valid data, no head rect saved either, so it is not very suitable for mp

Specifically, the weight of each sample is computed and saved.
The weight of a sample is based on the probability of the pose.
"""

import json
import pickle
import argparse
import os, shutil
import numpy as np
import copy
import cv2 as cv
from glob import glob
import os.path as ops

import random
import sys
sys.path.insert(0, '../..')
from lib.utils.common import *
from lib.datasets import datasets_kdh3d


# # depth truncation should not be applied in ground-truth images
# DEPTH_MAX = 5000  # in mm
# DEPTH_MIN = 0  # in mm

vis = True
joint_names = ['ANKLE_RIGHT', 'KNEE_RIGHT', 'HIP_RIGHT', 'HIP_LEFT', 'KNEE_LEFT', 'ANKLE_LEFT', 'PELVIS', 'THORAX',
               'UPPER_NECK', 'HEAD_TOP', 'WRIST_RIGHT', 'ELBOW_RIGHT', 'SHOULDER_RIGHT', 'SHOULDER_LEFT',
               'ELBOW_LEFT', 'WRIST_LEFT']
joint_ind_dic = {}
for i, name in enumerate(joint_names):
    joint_ind_dic[name] = i


def get_args():
    parser = argparse.ArgumentParser("Parsing human tof dataset")
    # parser.add_argument("--dataset_path", type=str, default="/media/yuliang/DATA/Datasets/Kinect_Depth_Human3D/multiperson_test_v2")
    parser.add_argument("--mpii_path", type=str, default="/media/yuliang/DATA/Datasets/MPII")
    parser.add_argument("--label_out_dir", type=str, default="labels")

    args = parser.parse_args()
    return args


def mkdir_if_missing(directory):
    if not ops.exists(directory):
        os.makedirs(directory)


def prepare_mpii_labels(in_train_file, istrain=True):
    img_id = 0
    annos = json.load(open(in_train_file, 'r'))
    image_label_dict = {}

    for i in range(len(annos)):

        save_name = annos[i]['image']
        if save_name not in image_label_dict:
            image_label_dict[save_name] = []

        # append label file
        if istrain:
            json_single_person = {
                '2d_joints': annos[i]['joints'],
                'visible_joints': annos[i]['joints_vis']
            }

            image_label_dict[save_name].append(json_single_person)

        # if vis:
        #     depth_vis = np.copy(depth_maps[i, :, :].astype(np.float32))
        #     depth_vis[depth_vis < 0] = 0
        #     depth_vis[depth_vis > 5] = 5
        #     depth_vis /= 5
        #     depth_vis *= 255
        #     depth_vis = cv2.cvtColor(depth_vis.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        #     depth_vis = draw_humans_visibility(depth_vis,
        #                                        joints_2d_cur_frame,
        #                                        limb_ids,
        #                                        datasets_kdh3d.jointColors)
        #
        #     cv2.imshow('skel vis', depth_vis)
        #     cv2.waitKey()

        img_id += 1

    print('Processed {} samples.'.format(img_id))

    return image_label_dict


def main(opt):

    """
        prepare data set
    """

    label_out_path = ops.join(opt.mpii_path, opt.label_out_dir)
    mkdir_if_missing(label_out_path)

    ################## Training Set ##############################
    in_file = ops.join(opt.mpii_path, 'annot/train.json')
    out_file = ops.join(label_out_path, 'labels_train.json')

    image_label_dict_json = prepare_mpii_labels(in_file)

    print('Parsed training set \n')
    # save json label file
    with open(out_file, 'w') as json_file:
        json.dump(image_label_dict_json, json_file, indent=4)

    ################## Validation Set #############################
    in_file = ops.join(opt.mpii_path, 'annot/valid.json')
    out_file = ops.join(label_out_path, 'labels_val.json')

    image_label_dict_json = prepare_mpii_labels(in_file)

    print('Parsed validation set \n')
    # save json label file
    with open(out_file, 'w') as json_file:
        json.dump(image_label_dict_json, json_file, indent=4)

    ################## Train+Val Set ##############################
    in_file = ops.join(opt.mpii_path, 'annot/trainval.json')
    out_file = ops.join(label_out_path, 'labels_trainval.json')

    image_label_dict_json = prepare_mpii_labels(in_file)

    print('Parsed trainval set \n')
    # save json label file
    with open(out_file, 'w') as json_file:
        json.dump(image_label_dict_json, json_file, indent=4)

    ################## Test Set ###################################
    in_file = ops.join(opt.mpii_path, 'annot/test.json')
    out_file = ops.join(label_out_path, 'labels_test.json')

    image_label_dict_json = prepare_mpii_labels(in_file, istrain=False)

    print('Parsed test set \n')
    # save json label file
    with open(out_file, 'w') as json_file:
        json.dump(image_label_dict_json, json_file, indent=4)


if __name__ == "__main__":
    opt = get_args()
    main(opt)
