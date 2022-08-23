"""
@author: Yuliang Guo <33yuliangguo@gmail.com>

Depth maps is converted to meters

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

vis = False
joint_names = ['HEAD', 'NECK', 'SHOULDER_RIGHT', 'SHOULDER_LEFT', 'ELBOW_RIGHT', 'ELBOW_LEFT', 'WRIST_RIGHT',
               'WRIST_LEFT', 'SPINE_NAVAL', 'HIP_RIGHT', 'HIP_LEFT', 'KNEE_RIGHT', 'KNEE_LEFT', 'ANKLE_RIGHT', 'ANKLE_LEFT']
joint_ind_dic = {}
for i, name in enumerate(joint_names):
    joint_ind_dic[name] = i

limb_ids = datasets_kdh3d.kp_connections(datasets_kdh3d.get_keypoints())


def get_args():
    parser = argparse.ArgumentParser("Parsing human tof dataset")
    parser.add_argument("--dataset_path", type=str, default="/media/yuliang/DATA/Datasets/Kinect_Depth_Human3D_small/multiperson_test_v2")
    # parser.add_argument("--dataset_path", type=str, default="/raid/yuliangguo/Datasets/Kinect_Depth_Human3D/multiperson_test")
    # parser.add_argument("--depth_out_dir", type=str, default="depth_maps")
    parser.add_argument("--label_in_dir", type=str, default="labels")
    parser.add_argument("--vis_ref_dir", type=str, default="vis_gt_refine")
    parser.add_argument("--label_out_dir", type=str, default="labels_refine")

    args = parser.parse_args()
    return args


def mkdir_if_missing(directory):
    if not ops.exists(directory):
        os.makedirs(directory)


def main(opt):

    # depth_data_files = sorted(glob('{}/*.npy'.format(opt.dataset_path)))

    """
        prepare data set
    """

    # depth_out_path = ops.join(opt.dataset_path, opt.depth_out_dir)
    label_in_path = ops.join(opt.dataset_path, opt.label_in_dir)
    vis_ref_path = ops.join(opt.dataset_path, opt.vis_ref_dir)
    label_out_path = ops.join(opt.dataset_path, opt.label_out_dir)

    in_label_file = ops.join(label_in_path, 'labels_test.json')
    mkdir_if_missing(label_out_path)
    out_label_file = ops.join(label_out_path, 'labels_test.json')

    in_label_dict_json = json.load(open(in_label_file, 'r'))
    out_label_dict_json = {}
    for key in in_label_dict_json:
        if ops.exists(ops.join(vis_ref_path, key[2:-4]+'.jpg')):
            out_label_dict_json[key] = in_label_dict_json[key]

    print('constructed data set \n')
    # save json label file
    with open(out_label_file, 'w') as json_file:
        json.dump(out_label_dict_json, json_file, indent=4)


if __name__ == "__main__":
    opt = get_args()
    main(opt)
