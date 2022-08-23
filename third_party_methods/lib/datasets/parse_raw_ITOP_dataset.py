import os
import os.path as ops
import numpy as np
import cv2
import h5py
import json
from lib.datasets import datasets_itop_rtpose
from lib.utils.common import approx_root_orientation

vis = 1
joint2box_margin = 30
seg2box_margin = 15
width = 320
height = 240

intrinsics = {'f': 1 / 0.0035, 'cx': 160, 'cy': 120}


def mkdir_if_missing(directory):
    if not ops.exists(directory):
        os.makedirs(directory)


def compute_bbox(segmentation, humans_gt_2d, joint2box_margin, height, width):
    """
    Segmentation is totally unreliable, not in use now!
    """
    xmin = width
    xmax = 0
    ymin = height
    ymax = 0
    seg_coords = np.nonzero(segmentation >= 0)
    if len(seg_coords[0]) > 0:
        xmin = min(xmin, np.min(seg_coords[1]) - seg2box_margin)
        ymin = min(ymin, np.min(seg_coords[0]) - seg2box_margin)
        xmax = max(xmax, np.max(seg_coords[1]) + seg2box_margin)
        ymax = max(ymax, np.max(seg_coords[0]) + seg2box_margin)

    human_2d = np.array(humans_gt_2d)
    new_xmin = min(xmin, np.min(human_2d[:, 0]) - joint2box_margin)
    new_ymin = min(ymin, np.min(human_2d[:, 1]) - joint2box_margin)
    new_xmax = max(xmax, np.max(human_2d[:, 0]) + joint2box_margin)
    new_ymax = max(ymax, np.max(human_2d[:, 1]) + joint2box_margin)

    new_xmin = int(max(0, min(width, new_xmin)))
    new_ymin = int(max(0, min(height, new_ymin)))
    new_xmax = int(max(0, min(width, new_xmax)))
    new_ymax = int(max(0, min(height, new_ymax)))
    boxes = [new_xmin, new_ymin, new_xmax, new_ymax]

    return boxes


def compute_pose_weight(labels):
    joint_names = datasets_itop_rtpose.get_keypoints()
    root_id = joint_names.index('torso')

    self_poses = labels['real_world_coordinates'] - labels['real_world_coordinates'][:, root_id, :].reshape(-1, 1, 3)

    # transform self-pose wrt current pelvis coordinate frame
    root_axis = approx_root_orientation(labels['real_world_coordinates'][:, joint_names.index('left_hip'), :],
                                          labels['real_world_coordinates'][:, joint_names.index('right_hip'), :],
                                          labels['real_world_coordinates'][:, joint_names.index('neck'), :])

    for i in range(self_poses.shape[0]):
        self_poses[i, :, :] = np.matmul(self_poses[i, :, :], root_axis[i, :, :])
    # self_poses = np.matmul(self_poses, root_axis)

    self_poses = np.delete(self_poses, root_id, axis=1)
    self_poses_not_nan = self_poses[~np.isnan(self_poses).any(axis=2).any(axis=1)]
    self_poses_mean = np.mean(self_poses_not_nan, axis=0).reshape([1, -1, 3])
    self_poses_std = np.std(self_poses_not_nan, axis=0).reshape([1, -1, 3])

    # dists = np.sum(((self_poses - self_poses_mean) / self_poses_std) ** 2, axis=2)
    dists = np.sqrt(np.sum(((self_poses - self_poses_mean) / self_poses_std) ** 2, axis=2))

    # a specific function to make close-range quadratic, and far-range linear
    dists[dists < 1] = dists[dists < 1]**2 / 2
    dists[dists >= 1] = dists[dists >= 1] - 0.5
    weights = np.mean(dists, axis=1).astype(np.float32)
    return weights


def process_itop_labels(labels, depth_maps, out_depth_dir, out_seg_dir, out_label_file):
    pose_weights = compute_pose_weight(labels)

    id = labels['id'][()]
    json_out = {}
    for i in range(id.shape[0]):
        # only keep valid samples as ground-truth
        if not labels['is_valid'][i]:
            continue

        fname = str(id[i])[2:-1]
        # save each depth and segmentation map
        depth_file = ops.join(out_depth_dir, fname) + '.npy'
        np.save(depth_file, depth_maps['data'][i, :, :])
        seg_file = ops.join(out_seg_dir, fname) + '.npy'
        np.save(seg_file, labels['segmentation'][i, :, :])

        # seg_i = labels['segmentation'][i, :, :]
        # bbox = compute_bbox(seg_i, labels['image_coordinates'][i, :, :], joint2box_margin, height, width)
        #
        # if vis:
        #     depth_vis = np.copy(depth_maps['data'][i, :, :].astype(np.float32))
        #     depth_vis[depth_vis < 0] = 0
        #     depth_vis[depth_vis > 5] = 5
        #     depth_vis /= 5
        #     depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
        #     depth_vis = cv2.rectangle(depth_vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 1), thickness=2)
        #     cv2.imshow('bbox vis', depth_vis)
        #     cv2.waitKey()
        human_3d = labels['real_world_coordinates'][i, :, :]
        x_2d = human_3d[:, 0] * intrinsics['f'] / human_3d[:, 2] + intrinsics['cx']
        y_2d = -human_3d[:, 1] * intrinsics['f'] / human_3d[:, 2] + intrinsics['cy']

        # append label file
        json_single_person = {
            '2d_joints': np.vstack([x_2d, y_2d]).T.tolist(),
            '3d_joints': labels['real_world_coordinates'][i, :, :].tolist(),
            'visible_joints': labels['visible_joints'][i, :].tolist(),
            'pose_weight': pose_weights[i].tolist()
        }

        # treated as a list for considering multiple humans
        json_out[fname] = [json_single_person]
        print('processed {}/{} samples'.format(i+1, id.shape[0]))

    # save json label file
    with open(out_label_file, 'w') as json_file:
        json.dump(json_out, json_file, indent=4)

    print('Finish preparation. Pose weight min: {0}, max: {1}'.format(np.min(pose_weights[~np.isnan(pose_weights)]),
                                                                      np.max(pose_weights[~np.isnan(pose_weights)])))


if __name__ == "__main__":
    # base_folder = '/media/yuliang/DATA/Datasets/ITOP/'
    base_folder = 'D:/Datasets/ITOP/'
    depth_file_train = ops.join(base_folder, 'ITOP_side_train_depth_map.h5')
    depth_file_test = ops.join(base_folder, 'ITOP_side_test_depth_map.h5')
    label_file_train = ops.join(base_folder, 'ITOP_side_train_labels.h5')
    label_file_test = ops.join(base_folder, 'ITOP_side_test_labels.h5')

    out_depth_train_dir = ops.join(base_folder, 'ITOP_side_train_depth_map')
    out_depth_test_dir = ops.join(base_folder, 'ITOP_side_test_depth_map')
    out_seg_train_dir = ops.join(base_folder, 'ITOP_side_train_seg_map')
    out_seg_test_dir = ops.join(base_folder, 'ITOP_side_test_seg_map')
    annotation_folder = ops.join(base_folder, 'annotations')
    out_label_train_file = ops.join(annotation_folder, 'ITOP_side_train_labels.json')
    out_label_test_file = ops.join(annotation_folder, 'ITOP_side_test_labels.json')

    mkdir_if_missing(out_depth_train_dir)
    mkdir_if_missing(out_depth_test_dir)
    mkdir_if_missing(out_seg_train_dir)
    mkdir_if_missing(out_seg_test_dir)
    mkdir_if_missing(annotation_folder)

    # process data
    labels_train = h5py.File(label_file_train, 'r')
    depth_maps_train = h5py.File(depth_file_train, 'r')
    labels_test = h5py.File(label_file_test, 'r')
    depth_maps_test = h5py.File(depth_file_test, 'r')

    print('processing training set')
    process_itop_labels(labels_train, depth_maps_train, out_depth_train_dir, out_seg_train_dir, out_label_train_file)
    print('processing testing set')
    process_itop_labels(labels_test, depth_maps_test, out_depth_test_dir, out_seg_test_dir, out_label_test_file)
