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

vis = True
joint_names = ['HEAD', 'NECK', 'SHOULDER_RIGHT', 'SHOULDER_LEFT', 'ELBOW_RIGHT', 'ELBOW_LEFT', 'WRIST_RIGHT',
               'WRIST_LEFT', 'SPINE_NAVAL', 'HIP_RIGHT', 'HIP_LEFT', 'KNEE_RIGHT', 'KNEE_LEFT', 'ANKLE_RIGHT', 'ANKLE_LEFT']
joint_ind_dic = {}
for i, name in enumerate(joint_names):
    joint_ind_dic[name] = i

limb_ids = datasets_kdh3d.kp_connections(datasets_kdh3d.get_keypoints())


def get_args():
    parser = argparse.ArgumentParser("Parsing human tof dataset")
    parser.add_argument("--dataset_path", type=str, default="/media/yuliang/DATA/Datasets/Kinect_Depth_Human3D")
    # parser.add_argument("--dataset_path", type=str, default="/raid/yuliangguo/Datasets/Kinect_Depth_Human3D")
    parser.add_argument("--depth_out_dir", type=str, default="depth_maps")
    parser.add_argument("--seg_out_dir", type=str, default="seg_maps")
    parser.add_argument("--label_out_dir", type=str, default="labels")
    parser.add_argument("--train_list", type=str, default="train_list_short.txt")
    parser.add_argument("--val_list", type=str, default="test_list.txt")

    args = parser.parse_args()
    return args


def mkdir_if_missing(directory):
    if not ops.exists(directory):
        os.makedirs(directory)


def compute_pose_weights(depth_data_files):
    """
    Compute the weight for each pose from a set of annotation files.
    Currently, it assumes each frame includes only a singe human.

    Args:
        depth_data_files: a list of depth file names, label files are parsed from them

    Returns:

    """
    self_pose_set_3d = None

    for depth_file in depth_data_files:
        print('processing: ' + depth_file)
        anno_file = '{}_label.json'.format(depth_file[:depth_file.rfind('.')])
        annos = json.load(open(anno_file, 'r'))
        joint_positions_3d = np.array(annos['3D_joint_positions'])

        drop_file = '{}_drop.json'.format(depth_file[:depth_file.rfind('.')])
        drop_indices = json.load(open(drop_file, 'r'))['drop_list']
        keep_indices = [ind for ind in range(joint_positions_3d.shape[0]) if ind not in drop_indices]

        # remove dropped frames
        joint_positions_3d = joint_positions_3d[keep_indices]

        # retrieve the subset of joints from kinect raw labels
        kinect_joint_names = annos['joint_names']
        joint_subset_ids = []
        for jname in joint_names:
            joint_subset_ids.append(kinect_joint_names.index(jname))
        joint_positions_3d = joint_positions_3d[:, joint_subset_ids, :]

        self_pose_set_3d_cur_file = \
            joint_positions_3d - joint_positions_3d[:, joint_ind_dic['SPINE_NAVAL'], :].reshape(-1, 1, 3)

        # transform self-pose wrt current pelvis coordinate frame
        root_axis = approx_root_orientation(joint_positions_3d[:, joint_ind_dic['HIP_LEFT'], :],
                                            joint_positions_3d[:, joint_ind_dic['HIP_RIGHT'], :],
                                            joint_positions_3d[:, joint_ind_dic['NECK'], :])

        for ii in range(self_pose_set_3d_cur_file.shape[0]):
            self_pose_set_3d_cur_file[ii, :, :] = np.matmul(self_pose_set_3d_cur_file[ii, :, :], root_axis[ii, :, :])

        self_pose_set_3d_cur_file = np.delete(self_pose_set_3d_cur_file, joint_ind_dic['SPINE_NAVAL'], axis=1)
        if self_pose_set_3d is None:
            self_pose_set_3d = self_pose_set_3d_cur_file
        else:
            self_pose_set_3d = np.concatenate((self_pose_set_3d, self_pose_set_3d_cur_file), axis=0)

    self_poses_not_nan = self_pose_set_3d[~np.isnan(self_pose_set_3d).any(axis=2).any(axis=1)]
    self_poses_mean = np.mean(self_poses_not_nan, axis=0).reshape([1, -1, 3])
    self_poses_std = np.std(self_poses_not_nan, axis=0).reshape([1, -1, 3])

    # dists = np.sum(((self_poses - self_poses_mean) / self_poses_std) ** 2, axis=2)
    dists = np.sqrt(np.sum(((self_pose_set_3d - self_poses_mean) / self_poses_std) ** 2, axis=2))

    # a specific function to make close-range quadratic, and far-range linear
    dists[dists < 1] = dists[dists < 1] ** 2 / 2
    dists[dists >= 1] = dists[dists >= 1] - 0.5
    pose_weights = np.mean(dists, axis=1).astype(np.float32)

    return pose_weights, self_poses_mean, self_poses_std


def prepare_dataset_from_files(depth_data_files, depth_out_path, seg_out_path, depth_data_files_train):
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
    image_label_dict_train = {}
    image_label_dict_test = {}
    img_id = 0

    print('Computing pose weights...')
    pose_weights, self_poses_mean, self_poses_std = compute_pose_weights(depth_data_files)

    # save each depth map individually, rename, and save path in the new output
    for depth_file in depth_data_files:
        mask_file = '{}_mask.npy'.format(depth_file[:depth_file.rfind('.')])
        anno_file = '{}_label.json'.format(depth_file[:depth_file.rfind('.')])
        drop_file = '{}_drop.json'.format(depth_file[:depth_file.rfind('.')])
        is_train = depth_file in depth_data_files_train

        depth_maps = np.load(depth_file)
        depth_maps /= 1000  # convert to meter unit
        seg_maps = np.load(mask_file)
        annos = json.load(open(anno_file, 'r'))
        joint_positions_3d = np.array(annos['3D_joint_positions']) / 1000  # convert to meter unit
        joint_positions_2d = np.array(annos['2D_joint_positions'])
        bboxes = np.array(annos['bounding_boxes'])
        drop_indices = json.load(open(drop_file, 'r'))['drop_list']
        keep_indices = [ind for ind in range(depth_maps.shape[0]) if ind not in drop_indices]

        # remove dropped frames
        depth_maps = depth_maps[keep_indices]
        joint_positions_3d = joint_positions_3d[keep_indices]
        joint_positions_2d = joint_positions_2d[keep_indices]
        bboxes = bboxes[keep_indices]

        # retrieve the subset of joints from kinect raw labels
        kinect_joint_names = annos['joint_names']
        joint_subset_ids = []
        for jname in joint_names:
            joint_subset_ids.append(kinect_joint_names.index(jname))
        joint_positions_3d = joint_positions_3d[:, joint_subset_ids, :]
        joint_positions_2d = joint_positions_2d[:, joint_subset_ids, :]

        for i in range(depth_maps.shape[0]):
            save_name = '{:08d}.npy'.format(img_id)
            np.save(os.path.join(depth_out_path, save_name),  depth_maps[i, :, :])
            np.save(os.path.join(seg_out_path, save_name), seg_maps[i, :, :])

            # append label file
            json_single_person = {
                '2d_joints': joint_positions_2d[i, :, :].tolist(),
                '3d_joints': joint_positions_3d[i, :, :].tolist(),
                "bbox": bboxes[i, :].tolist(),
                'pose_weight': pose_weights[img_id].tolist()
            }

            image_label_dict[save_name] = [json_single_person]

            if is_train:
                image_label_dict_train[save_name] = [json_single_person]
            else:
                image_label_dict_test[save_name] = [json_single_person]

            if vis and img_id % 100 == 0 and is_train:
                depth_vis = np.copy(depth_maps[i, :, :].astype(np.float32))
                depth_vis[depth_vis < 0] = 0
                depth_vis[depth_vis > 5] = 5
                depth_vis /= 5
                depth_vis *= 255
                depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
                # depth_vis = cv2.rectangle(depth_vis, (bboxes[i, 0], bboxes[i, 1]),
                #                           (bboxes[i, 2], bboxes[i, 3]), color=(0, 0, 1), thickness=2)
                depth_vis = draw_humans_visibility(depth_vis,
                                                   [json_single_person['2d_joints']],
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
    image_label_dict_train['intrinsics'] = annos['intrinsics']
    image_label_dict_test['intrinsics'] = annos['intrinsics']

    return image_label_dict, image_label_dict_train, image_label_dict_test, self_poses_mean, self_poses_std


def main(opt):

    train_list_file = os.path.join(opt.dataset_path, opt.label_out_dir, opt.train_list)
    test_list_file = os.path.join(opt.dataset_path, opt.label_out_dir, opt.val_list)
    if os.path.exists(train_list_file) and os.path.exists(test_list_file):
        train_file = open(train_list_file, 'r')
        train_names = train_file.readlines()
        train_file.close()
        test_file = open(test_list_file, 'r')
        test_names = test_file.readlines()
        test_file.close()
        depth_data_files_train = [opt.dataset_path + "/" + fname[:-1] for fname in train_names if len(fname) > 1]
        depth_data_files_test = [opt.dataset_path + "/" + fname[:-1] for fname in test_names if len(fname) > 1]
        depth_data_files = depth_data_files_train + depth_data_files_test
    else:
        mask_data_files = sorted(glob('{}/*mask.npy'.format(opt.dataset_path)))
        # mask_data_files = mask_data_files[:2]
        depth_data_files = [f[:-9]+'.npy' for f in mask_data_files]

        # # shuffle the list of recordings and split the whole dataset into train and test video-wise
        # order = [ind for ind in range(len(depth_data_files))]
        # random.shuffle(order)
        # indices_train = order[: int(0.8*len(order))]
        # indices_val = order[int(0.8*len(order)):]
        # depth_data_files_train = [depth_data_files[ind] for ind in indices_train]
        # depth_data_files_val = [depth_data_files[ind] for ind in indices_val]

        # # TODO: some free styles existing violating the 4 sets
        num_train_files = int(0.8*len(depth_data_files) / 4) * 4
        depth_data_files_train = depth_data_files[:num_train_files]
        # depth_data_files_train = depth_data_files[:1]

    """
        prepare training set
    """

    depth_out_path = ops.join(opt.dataset_path, opt.depth_out_dir)
    seg_out_path = ops.join(opt.dataset_path, opt.seg_out_dir)
    label_out_path = ops.join(opt.dataset_path, opt.label_out_dir)
    if os.path.exists(depth_out_path):
        shutil.rmtree(depth_out_path)
    os.mkdir(depth_out_path)
    if os.path.exists(seg_out_path):
        shutil.rmtree(seg_out_path)
    os.mkdir(seg_out_path)
    mkdir_if_missing(label_out_path)
    out_label_file = ops.join(label_out_path, 'labels_all.json')
    out_label_file_train = ops.join(label_out_path, 'labels_train.json')
    out_label_file_test = ops.join(label_out_path, 'labels_test.json')

    image_label_dict_json, image_label_dict_train_json, image_label_dict_test_json, \
        self_pose_3d_mean, self_pose_3d_std = prepare_dataset_from_files(depth_data_files,
                                                                         depth_out_path,
                                                                         seg_out_path,
                                                                         depth_data_files_train)

    json_self_pose_distr = {}
    json_self_pose_distr['self pose mean 3d'] = self_pose_3d_mean.tolist()
    json_self_pose_distr['self pose std 3d'] = self_pose_3d_std.tolist()
    with open(ops.join(label_out_path, 'self_pose_distr.json'), 'w') as json_file:
        json.dump(json_self_pose_distr, json_file, indent=4)

    print('constructed data set \n')
    # num_train_samples = copy.deepcopy(img_id)
    # save json label file
    with open(out_label_file, 'w') as json_file:
        json.dump(image_label_dict_json, json_file, indent=4)

    with open(out_label_file_train, 'w') as json_file:
        json.dump(image_label_dict_train_json, json_file, indent=4)

    with open(out_label_file_test, 'w') as json_file:
        json.dump(image_label_dict_test_json, json_file, indent=4)


if __name__ == "__main__":
    opt = get_args()
    main(opt)
