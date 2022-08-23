"""
@author: Yuliang Guo <33yuliangguo@gmail.com>
"""

import json
import pickle
import argparse
import os, shutil
import copy
import numpy as np
import cv2 as cv
from glob import glob
import os.path as ops

import random

# depth truncation should not be applied in ground-truth images
DEPTH_MAX = 6
DEPTH_MIN = 0


def get_args():
    parser = argparse.ArgumentParser("Parsing background tof dataset")
    # parser.add_argument("--dataset_path", type=str, default="/media/yuliang/DATA/Datasets/Kinect_Depth_Human3D")
    parser.add_argument("--dataset_path", type=str, default="/raid/yuliangguo/Datasets/Kinect_Depth_Human3D")
    parser.add_argument("--bg_out_dir", type=str, default="bg_maps")
    parser.add_argument("--label_out_dir", type=str, default="labels")
    parser.add_argument("--sample_interval", type=int, default=2)
    args = parser.parse_args()
    return args


def main(opt):
    image_dict = {}
    img_id = 0
    lower_bound = 15

    # save each depth map individually, rename, and save path in the new output
    bg_out_path = ops.join(opt.dataset_path, opt.bg_out_dir)
    if os.path.exists(bg_out_path):
        shutil.rmtree(bg_out_path)
    os.mkdir(bg_out_path)
    label_out_path = ops.join(opt.dataset_path, opt.label_out_dir)
    if not os.path.exists(label_out_path):
        os.mkdir(label_out_path)
    out_label_file = ops.join(label_out_path, 'labels_bg.json')

    depth_data_files = glob('{}/bg*.npy'.format(opt.dataset_path))

    # construct train set
    for depth_file in depth_data_files:
        # depth_maps = np.load(depth_file)
        depth_maps = np.load(depth_file)
        depth_maps /= 1000  # convert to meter unit

        # normalize for visualization
        depth_maps_copy = copy.deepcopy(depth_maps)
        depth_maps_copy[depth_maps_copy < DEPTH_MIN] = DEPTH_MIN
        depth_maps_copy[depth_maps_copy > DEPTH_MAX] = DEPTH_MAX
        depth_maps_copy = depth_maps_copy / DEPTH_MAX * 255

        for i in range(0, depth_maps.shape[0], opt.sample_interval):
            if np.mean(depth_maps_copy[i, :, :]) < lower_bound:
                continue
            save_name = '{:08d}.png'.format(img_id)
            cv.imwrite(os.path.join(bg_out_path, save_name), depth_maps_copy[i, :, :])

            save_name = '{:08d}.npy'.format(img_id)
            np.save(os.path.join(bg_out_path, save_name), depth_maps[i, :, :])

            image_dict[img_id] = {"file_name": save_name}
            img_id += 1
        print('Finished processing record {}.'.format(depth_file[:depth_file.rfind('.')]))
        print('Processed {} samples.'.format(img_id))

    print('constructed bg set \n')
    with open(out_label_file, 'w') as json_file:
        json.dump(image_dict, json_file, indent=4)
    print('Processed {} samples in total.'.format(img_id))


if __name__ == "__main__":
    opt = get_args()
    main(opt)
