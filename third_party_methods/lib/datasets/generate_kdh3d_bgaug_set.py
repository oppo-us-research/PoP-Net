import argparse
import time
import os, shutil
import numpy as np
import cv2
import torch
import pylab as plt
import json
import copy
from collections import OrderedDict
from glob import glob
import os.path as ops
import sys
sys.path.insert(0, '../..')
from lib.network.yolo_posenet import YoloPoseNet
from lib.datasets import datasets_kdh3d, data_augmentation_2d3d
from lib.utils.common import *
from lib.utils.prior_pose_align import parse_prior_pose
from evaluate.eval_pose_mp import eval_human_dataset_2d, eval_human_dataset_3d


"""
Generate a set of composed multi-person dataset for later testing

Author: Yuliang Guo (33yuliangguo@gmail.com)
Date: April 2020
"""


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DATA_DIR = '/raid/yuliangguo/Datasets/Kinect_Depth_Human3D_small'
# DATA_DIR = '/media/yuliang/DATA/Datasets/Kinect_Depth_Human3D'
ANNOTATIONS_VAL = '{}/labels_test.json'.format(os.path.join(DATA_DIR, 'labels'))
IMAGE_DIR = os.path.join(DATA_DIR, 'depth_maps')
BG_FILE = os.path.join(DATA_DIR, 'labels', 'labels_bg.json')
BG_DIR = os.path.join(DATA_DIR, 'bg_maps')
SEG_DIR = os.path.join(DATA_DIR, 'seg_maps')
anchors = np.array([(6., 3.), (12., 6.)])


def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--annotations', default=ANNOTATIONS_VAL)
    parser.add_argument('--image-dir', default=IMAGE_DIR)
    parser.add_argument('--bg-file', default=BG_FILE)
    parser.add_argument('--bg-dir', default=BG_DIR)
    parser.add_argument('--seg-dir', default=SEG_DIR)
    parser.add_argument('--loader-workers', default=4, type=int,
                        help='number of workers for data loading')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='batch size')
    parser.add_argument('--w-org', default=480, type=int,
                        help='original width of dataset image')
    parser.add_argument('--h-org', default=512, type=int,
                        help='original height of dataset image')
    parser.add_argument('--max-aug-ratio', default=1.7, type=float,
                        help='the max augment depth ratio')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--output-dir', type=str,
                        default=DATA_DIR+'/bgaug_test/')
    parser.add_argument("--depth_out_dir", type=str, default="depth_maps")
    parser.add_argument("--label_out_dir", type=str, default="labels")
    args = parser.parse_args()

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True

    return args


def mkdir_if_missing(directory):
    if not ops.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":
    args = cli()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    depth_out_path = ops.join(args.output_dir, args.depth_out_dir)
    label_out_path = ops.join(args.output_dir, args.label_out_dir)
    if os.path.exists(depth_out_path):
        shutil.rmtree(depth_out_path)
    os.mkdir(depth_out_path)
    mkdir_if_missing(label_out_path)
    out_label_file = ops.join(label_out_path, 'labels_test.json')
    intrinsics = datasets_kdh3d.intrinsics

    print("Loading dataset...")
    # load train data
    preprocess = data_augmentation_2d3d.Compose([
        data_augmentation_2d3d.Cvt2ndarray(),
        # data_augmentation_depth_3d.Rotate(cx=intrinsics['cx'], cy=intrinsics['cy']),  # 2D-3D relation still holds
        # data_augmentation_depth_3d.RenderDepth(cx=intrinsics['cx'], cy=intrinsics['cy'], max_ratio=args.max_aug_ratio),  # 2D-3D relation still holds
        data_augmentation_2d3d.Resize(args.w_org, args.h_org)  # 2D-3D relation broke, but easy to recover
    ])

    val_data = datasets_kdh3d.KDH3D_Keypoints(
        img_dir=args.image_dir,
        ann_file=args.annotations,
        preprocess=preprocess,
        input_x=args.w_org,
        input_y=args.h_org,
        anchors=anchors,
        bg_aug=True,
        bg_file=args.bg_file,
        bg_dir=args.bg_dir,
        seg_dir=args.seg_dir,
        is_train=False
    )

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True)

    image_label_dict = {}
    cnt = 0
    print("Processing dataset...")

    human_gt_set_2d = []
    human_gt_set_3d = []
    for batch_i, (imgs, indices) in enumerate(val_loader):

        humans_gt_2d_batch = []
        humans_gt_3d_batch = []
        gt_bbox_batch = []

        for id in indices.numpy():
            human_gt_batch = val_data.anno_dic[val_data.ids[id]]
            human_gt_2d = [human['2d_joints'] for human in human_gt_batch]
            humans_gt_2d_batch.append(human_gt_2d)
            human_gt_set_2d.append(human_gt_2d)

            human_gt_3d = [human['3d_joints'] for human in human_gt_batch]
            humans_gt_3d_batch.append(human_gt_3d)
            human_gt_set_3d.append(human_gt_3d)

            gt_bbox_batch.append([human['bbox'] for human in human_gt_batch])

        # unormalize images
        # imgs = imgs.cpu().data.numpy()
        imgs *= datasets_kdh3d.depth_std
        imgs += datasets_kdh3d.depth_mean

        for b in range(args.batch_size):
            save_name = '{:08d}.npy'.format(cnt)
            np.save(os.path.join(depth_out_path, save_name),  imgs[b, 0, :, :])

            image_label_dict[save_name] = []

            for j in range(len(humans_gt_2d_batch[b])):
                # append label file
                json_single_person = {
                    '2d_joints': humans_gt_2d_batch[b][j],
                    '3d_joints': humans_gt_3d_batch[b][j],
                    'bbox': gt_bbox_batch[b][j]
                }

                image_label_dict[save_name].append(json_single_person)
            print(cnt)
            cnt += 1

    print('constructed data set \n')
    # save json label file
    with open(out_label_file, 'w') as json_file:
        json.dump(image_label_dict, json_file, indent=4)


