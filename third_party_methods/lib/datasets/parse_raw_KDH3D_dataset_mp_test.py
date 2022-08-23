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
    parser.add_argument("--dataset_path", type=str, default="/media/yuliang/DATA/Datasets/Kinect_Depth_Human3D_small/multiperson_test_raw")
    # parser.add_argument("--dataset_path", type=str, default="/raid/yuliangguo/Datasets/Kinect_Depth_Human3D/multiperson_test")
    parser.add_argument("--depth_out_dir", type=str, default="depth_maps")
    parser.add_argument("--label_out_dir", type=str, default="labels")

    args = parser.parse_args()
    return args


def mkdir_if_missing(directory):
    if not ops.exists(directory):
        os.makedirs(directory)


def prepare_dataset_from_files(depth_data_files, depth_out_path):
    """
        Currently, this program assumes each sample includes only one skeleton.
        Raw 3D unit in millimeters are changed to meters.
    Args:
        depth_data_files:
        depth_out_path:
        seg_out_path:
        img_id:
        depth_data_files_train:

    Returns:

    """
    image_label_dict = {}
    img_id = 0

    # save each depth map individually, rename, and save path in the new output
    for depth_file in depth_data_files:
        anno_file = '{}_label.json'.format(depth_file[:depth_file.rfind('.')])

        depth_maps = np.load(depth_file)
        depth_maps /= 1000  # convert to meter unit
        annos = json.load(open(anno_file, 'r'))

        joint_positions_3d = annos['3D_joint_positions']
        joint_positions_2d = annos['2D_joint_positions']
        bboxes = annos['bounding_boxes']
        valid_bodies = annos['valid']

        # retrieve the subset of joints from kinect raw labels
        kinect_joint_names = annos['joint_names']
        joint_subset_ids = []
        for jname in joint_names:
            joint_subset_ids.append(kinect_joint_names.index(jname))

        for i in range(depth_maps.shape[0]):
            if np.sum(np.array(valid_bodies[i])) < len(valid_bodies[i]):
                continue

            save_name = '{:08d}.npy'.format(img_id)
            np.save(os.path.join(depth_out_path, save_name),  depth_maps[i, :, :])

            image_label_dict[save_name] = []
            joints_2d_cur_frame = []
            bbox_cur_frame = []
            for j in range(len(joint_positions_3d[i])):
                joints_2d = np.array(joint_positions_2d[i][j])
                joints_3d = np.array(joint_positions_3d[i][j]) / 1000  # convert to meter unit

                joints_2d = joints_2d[joint_subset_ids, :]
                joints_3d = joints_3d[joint_subset_ids, :]
                # append label file
                json_single_person = {
                    '2d_joints': joints_2d.tolist(),
                    '3d_joints': joints_3d.tolist(),
                    "bbox": bboxes[i][j],
                }

                image_label_dict[save_name].append(json_single_person)
                joints_2d_cur_frame.append(joints_2d)
                bbox_cur_frame.append(bboxes[i][j])

            if vis:
                depth_vis = np.copy(depth_maps[i, :, :].astype(np.float32))
                depth_vis[depth_vis < 0] = 0
                depth_vis[depth_vis > 5] = 5
                depth_vis /= 5
                depth_vis *= 255
                depth_vis = cv2.cvtColor(depth_vis.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                # for bbox in bbox_cur_frame:
                #     depth_vis = cv2.rectangle(depth_vis, (bbox[0], bbox[1]),
                #                               (bbox[2], bbox[3]), color=(0, 0, 255), thickness=2)
                depth_vis = draw_humans_visibility(depth_vis,
                                                   joints_2d_cur_frame,
                                                   limb_ids,
                                                   datasets_kdh3d.jointColors)

                # cv2.imshow('bbox vis', depth_vis)
                # cv2.waitKey()
                save_name = '{:08d}.png'.format(img_id)
                cv2.imwrite(os.path.join(depth_out_path, save_name), depth_vis)

            img_id += 1

        print('Finished processing record {}.'.format(depth_file[:depth_file.rfind('.')]))
        print('Processed {} samples.'.format(img_id))

    image_label_dict['intrinsics'] = annos['intrinsics']

    return image_label_dict


def main(opt):

    depth_data_files = sorted(glob('{}/*.npy'.format(opt.dataset_path)))

    """
        prepare data set
    """

    depth_out_path = ops.join(opt.dataset_path, opt.depth_out_dir)
    label_out_path = ops.join(opt.dataset_path, opt.label_out_dir)
    if os.path.exists(depth_out_path):
        shutil.rmtree(depth_out_path)
    os.mkdir(depth_out_path)
    mkdir_if_missing(label_out_path)
    out_label_file = ops.join(label_out_path, 'labels_test.json')

    image_label_dict_json = prepare_dataset_from_files(depth_data_files, depth_out_path)

    print('constructed data set \n')
    # save json label file
    with open(out_label_file, 'w') as json_file:
        json.dump(image_label_dict_json, json_file, indent=4)


if __name__ == "__main__":
    opt = get_args()
    main(opt)
