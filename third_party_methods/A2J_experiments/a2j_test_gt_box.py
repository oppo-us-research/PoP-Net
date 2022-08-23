import copy
import logging
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
import torch.utils.data
import torchvision
import torch.optim.lr_scheduler as lr_scheduler
import logging
import time
import datetime
import random
from PIL import Image
from tqdm import tqdm
from glob import glob
from random import uniform
import sys

sys.path.insert(0, '..')
from A2J_experiments import model, anchor, resnet, random_erasing

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# DataHyperParms
keypointsNumber = 15
imgWidth = 480
imgHeight = 512
cropWidth = 288
cropHeight = 288
batch_size = 8
intrinsics = {'fx': 504.1189880371094, 'fy': 504.042724609375, 'cx': 231.7421875, 'cy': 320.62640380859375}


def pixel2world(x, y, z):
    worldX = (x - intrinsics['cx']) * z / intrinsics['fx']
    worldY = (y - intrinsics['cy']) * z / intrinsics['fy']
    return worldX, worldY


save_dir = './bgaug_result_gtbox'

try:
    os.makedirs(save_dir)
except OSError:
    pass


testingImageDir = 'C:/Users/USS00019/Dataset' \
                  '/Kinect_Depth_Human3D_small/bgaug_test/depth_maps/'
model_dir = 'C:/Users/USS00019/pop_net_multiperson_3d_pose_estimation/A2J_experiments' \
            '/bg_result/net_34_wetD_0.0001_depFact_0.5_RegFact_3_rndShft_5.pth'

MEAN = 3
STD = 2
depth_max = 6

with open('C:/Users/USS00019/Dataset' \
                  '/Kinect_Depth_Human3D_small/bgaug_test/labels/labels_test.json') as f:
    ANNOTATIONS_TEST = json.load(f)

keypointsUVD_test = []
keypoints2d_test = []
bndbox_test = []
test_image_ids = []
keypoints2d_test_set = []
keypointsUVD_test_set = []

for key, value in ANNOTATIONS_TEST.items():
    if key == 'intrinsics':
        continue
    for frame in value:
        test_image_ids.append(key)
        keypointsUVD_test.append([frame['3d_joints']])
        keypoints2d_test.append([frame['2d_joints']])
        bndbox_test.append(frame['bbox'])

keypointsUVD_test = np.asarray(keypointsUVD_test, dtype=np.float32)
keypoints2d_test = np.asarray(keypoints2d_test, dtype=np.float32)
bndbox_test = np.asarray(bndbox_test, dtype=np.float32)



joint_id_to_name = {
  0: 'Head',
  1: 'Neck',
  2: 'RShoulder',
  3: 'LShoulder',
  4: 'RElbow',
  5: 'LElbow',
  6: 'RHand',
  7: 'LHand',
  8: 'Torso',
  9: 'RHip',
  10: 'LHip',
  11: 'RKnee',
  12: 'LKnee',
  13: 'RFoot',
  14: 'LFoot',
}


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

jointColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
               [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
               [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [85, 255, 85]]

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

def transform(img, label, matrix):
    '''
    img: [H, W]  label, [N,2]
    '''
    img_out = cv2.warpAffine(img,matrix,(cropWidth,cropHeight))
    label_out = np.ones((keypointsNumber, 3))
    label_out[:,:2] = label[:,:2].copy()
    label_out = np.matmul(matrix, label_out.transpose())
    label_out = label_out.transpose()

    return img_out, label_out

def draw_humans_visibility(img, humans, limbs, jointColors, visibilities=None):
    visibilities = visibilities or None
    for i, human in enumerate(humans):
        human_vis = np.array(human)
        for k, limb in enumerate(limbs):
            if visibilities is not None and visibilities[i][limb[0]] < 0.5:
                color = [0, 0, 0]
            else:
                color = [0, 0, 255]
            center1 = human_vis[limb[0], :2].astype(np.int)
            img = cv2.circle(img, tuple(center1), 3, color, thickness=2, lineType=8, shift=0)

            if visibilities is not None and visibilities[i][limb[1]] < 0.5:
                color = [0, 0, 0]
            else:
                color = [0, 0, 255]
            center2 = human_vis[limb[1], :2].astype(np.int)
            img = cv2.line(img, tuple(center1), tuple(center2), jointColors[k], 2)
            img = cv2.circle(img, tuple(center2), 3, color, thickness=2, lineType=8, shift=0)

    return img


def eval_human_dataset_2d(humans_pred_set, humans_gt_set, num_joints=15, dist_th=10.0, iou_th=0.5, human_gt_set_visibility=None):
    """
    Evaluation of the full dataset by matching predicted humans to ground-truth humans.
    This evaluation considers multi-to-multi matching, although itop dataset only includes single person.
    For each ground-truth human, find its max overlapping predicted human vis bbox IOUs.
    For each pair of matched poses, compute distance per joint. If the distance is < dist_th, the gt joint is
    considered detected.

    Overall metric: average Keypoint Correct Percentage (KCP) over all the joints and over the whole test set

    ATTENTION: those missed detection due to occlusion are not treated separately. If the guess is wrong, it's punished.


    :param humans_pred_set: N x list of humans for N images, each human is a K x 2 matrix recording x, y image coordinates
    :param humans_gt_set: N x list of humans for N images, each human is a K x 2 matrix recording x, y image coordinates
    :param num_joints: number of joints per person
    :param dist_th: the distance threshold in pixels
    :param iou_th: the iou threshold considering two human poses overlapping
    :return:
        joint_avg_dist: average distance per joint for those matched ground-truth joints only
        joint_KCP: KCP per joint over the test set
    """

    assert len(humans_gt_set) == len(humans_pred_set)
    human_gt_set_visibility_all = []
    samples_cnt = 0  # number of humans, not number of images
    joint_dists_set = []
    for i in range(len(humans_gt_set)):
        # print('evaluate {}/{}'.format(i, len(humans_gt_set)))
        humans_gt = humans_gt_set[i]
        humans_pred = humans_pred_set[i]
        samples_cnt += len(humans_gt)

        if len(humans_gt) == 0:
            continue

        joint_dists = match_humans_2d(humans_pred, humans_gt, iou_th)

        if human_gt_set_visibility is not None:
            for j, human_gt_visibility in enumerate(human_gt_set_visibility[i]):
                human_gt_set_visibility_all.append(human_gt_visibility)
                joint_dists[j][np.array(human_gt_visibility) == 0] = -1

        joint_dists_set += joint_dists

    human_gt_set_visibility_all = np.array(human_gt_set_visibility_all)

    joint_dists_set = np.array(joint_dists_set)
    joint_avg_dist = []
    joint_KCP = []
    for k in range(num_joints):
        single_joint_dists = joint_dists_set[:, k]
        joint_avg_dist.append(np.average(single_joint_dists[np.where(single_joint_dists >= 0)]))
        hit_cnt = np.sum(np.logical_and(single_joint_dists >= 0, single_joint_dists < dist_th))
        # A option to only consider gt visible parts
        if human_gt_set_visibility_all.shape[0] is not 0:
            joint_KCP.append(hit_cnt / np.sum(human_gt_set_visibility_all[:, k]))
        else:
            joint_KCP.append(hit_cnt / samples_cnt)

    return joint_avg_dist, joint_KCP

def match_humans_2d(humans_pred, humans_gt, iou_th=0.5):
    """
    For each ground-truth human, find its max overlapping predicted human vis bbox IOUs.
    For each pair of matched poses, compute distance per joint.
    If the whole ground-truth person got no match, its corresponding joint dists are assigned to be -1.

    ATTENTION:
    1. Assume gt list is not empty, but pred list can be empty.
    2. Those invalid joint due to occlusion lead to invalid distance -1

    :param humans_pred: list of list x, y pos
    :param humans_gt: list of list x, y pos
    :return: a list of K-length array recording joint distances per ground-truth human
    """
    joint_dists = []
    # if prediction is empty, return -1 distance for each gt joint
    if len(humans_pred) == 0:
        for human_gt in humans_gt:
            joint_dists.append(np.ones(len(human_gt))*(-1))
        return joint_dists

    # compute bbox
    bboxes_gt = compute_bbox_from_humans(humans_gt)
    bboxes_pred = compute_bbox_from_humans(humans_pred)

    # compute ious
    ious = bbox_ious(bboxes_gt, bboxes_pred)

    # compute matched joint distance per ground-truth human
    for i, human_gt in enumerate(humans_gt):
        if np.max(ious[i, :]) < iou_th:
            joint_dists.append(np.ones(len(human_gt))*(-1))
            continue

        human_pred = humans_pred[np.argmax(ious[i, :])]
        human_gt = np.array(human_gt)
        human_pred = np.array(human_pred)
        joint_dist = np.sqrt(np.sum((human_gt-human_pred)**2, axis=1))
        # invalid detected joint leads to invalid distance -1
        joint_dist[np.logical_and(human_pred[:, 0] == -1, human_pred[:, 1] == -1)] = -1
        # if np.sum(np.logical_and(human_pred[:, 0] == -1, human_pred[:, 1] == -1)) > 0:
        #     print('find invalid joint')
        joint_dists.append(joint_dist)

    return joint_dists


def eval_human_dataset_3d(humans_pred_set_2d, humans_gt_set_2d, humans_pred_set_3d, humans_gt_set_3d, num_joints=15, dist_th=0.1, iou_th=0.5, human_gt_set_visibility=None):
    """
    Evaluation of the full dataset by matching predicted humans to ground-truth humans.
    This evaluation considers multi-to-multi matching, although itop dataset only includes single person.
    For each ground-truth human, find its max overlapping predicted human vis bbox IOUs.
    For each pair of matched poses, compute distance per joint. If the distance is < dist_th, the gt joint is
    considered detected.

    Overall metric: average Keypoint Correct Percentage (KCP) over all the joints and over the whole test set

    ATTENTION: those missed detection due to occlusion are not treated separately. If the guess is wrong, it's punished.


    :param humans_pred_set_2d: N x list of humans for N images, each human is a K x 2 matrix recording x, y image coordinates
    :param humans_gt_set_2d: N x list of humans for N images, each human is a K x 2 matrix recording x, y image coordinates
    :param num_joints: number of joints per person
    :param dist_th: the distance threshold in meters
    :param iou_th: the iou threshold considering two human poses overlapping
    :return:
        joint_avg_dist: average distance per joint for those matched ground-truth joints only
        joint_KCP: KCP per joint over the test set
    """

    assert len(humans_gt_set_2d) == len(humans_pred_set_2d)

    samples_cnt = 0  # number of humans, not number of images
    joint_dists_set = []
    human_gt_set_visibility_all = []
    for i in range(len(humans_gt_set_2d)):
        # print('evaluate {}/{}'.format(i, len(humans_gt_set_2d)))
        humans_gt_2d = humans_gt_set_2d[i]
        humans_pred_2d = humans_pred_set_2d[i]
        humans_gt_3d = humans_gt_set_3d[i]
        humans_pred_3d = humans_pred_set_3d[i]
        samples_cnt += len(humans_gt_2d)

        if len(humans_gt_2d) == 0:
            continue

        joint_dists = match_humans_3d(humans_pred_2d, humans_gt_2d, humans_pred_3d, humans_gt_3d, iou_th)

        if human_gt_set_visibility is not None:
            for j, human_gt_visibility in enumerate(human_gt_set_visibility[i]):
                human_gt_set_visibility_all.append(human_gt_visibility)
                joint_dists[j][np.array(human_gt_visibility) == 0] = -1

        joint_dists_set += joint_dists
    human_gt_set_visibility_all = np.array(human_gt_set_visibility_all)

    joint_dists_set = np.array(joint_dists_set)
    joint_avg_dist = []
    joint_KCP = []
    for k in range(num_joints):
        single_joint_dists = joint_dists_set[:, k]
        joint_avg_dist.append(np.average(single_joint_dists[np.where(single_joint_dists >= 0)]))
        hit_cnt = np.sum(np.logical_and(single_joint_dists >= 0, single_joint_dists < dist_th))
        if human_gt_set_visibility is not None:
            joint_KCP.append(hit_cnt / np.sum(human_gt_set_visibility_all[:, k]))
        else:
            joint_KCP.append(hit_cnt / samples_cnt)

    return joint_avg_dist, joint_KCP

def match_humans_3d(humans_pred_2d, humans_gt_2d, humans_pred_3d, humans_gt_3d, iou_th=0.5):
    """
    For each ground-truth human, find its max overlapping predicted human vis bbox IOUs.
    For each pair of matched poses, compute distance per joint.
    If the whole ground-truth person got no match, its corresponding joint dists are assigned to be -1.

    ATTENTION:
    1. Assume gt list is not empty, but pred list can be empty.
    2. Those invalid joint due to occlusion lead to invalid distance -1

    :param humans_pred_2d: list of list x, y pos in image plane
    :param humans_gt_2d: list of list x, y pos in image plane
    :param humans_pred_3d: list of list x, y, z pos in camera coordinate frame
    :param humans_gt_3d: list of list x, y, x pos in camera coordinate frame
    :return: a list of K-length array recording joint distances per ground-truth human
    """
    joint_dists = []
    # if prediction is empty, return -1 distance for each gt joint
    if len(humans_pred_2d) == 0:
        for human_gt in humans_gt_2d:
            joint_dists.append(np.ones(len(human_gt))*(-1))
        return joint_dists

    # compute bbox in 2D
    bboxes_gt = compute_bbox_from_humans(humans_gt_2d)
    bboxes_pred = compute_bbox_from_humans(humans_pred_2d)

    # compute ious
    ious = bbox_ious(bboxes_gt, bboxes_pred)

    # compute matched joint 3D distance per ground-truth human
    for i, human_gt in enumerate(humans_gt_3d):
        if np.max(ious[i, :]) < iou_th:
            joint_dists.append(np.ones(len(human_gt))*(-1))
            continue

        human_pred = humans_pred_3d[np.argmax(ious[i, :])]
        human_gt = np.array(human_gt)
        human_pred = np.array(human_pred)
        joint_dist = np.sqrt(np.sum((human_gt-human_pred)**2, axis=1))

        # invalid detected joint leads to invalid distance -1
        human_pred2d = humans_pred_2d[np.argmax(ious[i, :])]
        human_pred2d = np.array(human_pred2d)
        joint_dist[np.logical_and(human_pred2d[:, 0] == -1, human_pred2d[:, 1] == -1)] = -1

        # TODO: invisible gt does not count distance (if to consider all GT, do not update 2D gt pos to (-1, -1))
        human_gt2d = humans_gt_2d[i]
        human_gt2d = np.array(human_gt2d)
        joint_dist[np.logical_and(human_gt2d[:, 0] == -1, human_gt2d[:, 1] == -1)] = -1

        joint_dists.append(joint_dist)

    return joint_dists

def compute_bbox_from_humans(humans):
    """
        ATTENTION: only those valid joints are used to calculate bbox
    :param humans: pure list of list
    :return:
    """
    bboxes = []
    for human in humans:
        valid_joints = human
        valid_joints = np.array(valid_joints)
        #valid_joints = np.array([joint for joint in human if joint != [-1, -1]])
        if len(valid_joints) == 0:
            return np.array([])
        xmin = np.min(valid_joints[:, 0])
        ymin = np.min(valid_joints[:, 1])
        xmax = np.max(valid_joints[:, 0])
        ymax = np.max(valid_joints[:, 1])
        bboxes.append([xmin, ymin, xmax, ymax])
    return np.array(bboxes)


def bbox_ious(boxes1, boxes2):
    """

    :param boxes1: N1 X 4, [xmin, ymin, xmax, ymax]
    :param boxes2: N2 X 4, [xmin, ymin, xmax, ymax]
    :return: N1 X N2
    """

    if len(boxes2) == 0:
        return np.ones([len(boxes1), 1]) * (-1)
    b1x1, b1y1 = np.split(boxes1[:, :2], 2, axis=1)
    b1x2, b1y2 = np.split(boxes1[:, 2:4], 2, axis=1)
    b2x1, b2y1 = np.split(boxes2[:, :2], 2, axis=1)
    b2x2, b2y2 = np.split(boxes2[:, 2:4], 2, axis=1)

    dx = np.maximum(np.minimum(b1x2, np.transpose(b2x2)) - np.maximum(b1x1, np.transpose(b2x1)), 0)
    dy = np.maximum(np.minimum(b1y2, np.transpose(b2y2)) - np.maximum(b1y1, np.transpose(b2y1)), 0)
    intersections = dx * dy

    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = (areas1 + np.transpose(areas2)) - intersections

    return intersections / unions

def dataPreprocess(index, imgDir, keypointsPixel, keypointsWorld, bndbox, mode, augment = False):
    depth_img = np.load(os.path.join(imgDir, test_image_ids[index])).astype(np.float)

    imageOutputs = np.ones((cropHeight, cropWidth, 1), dtype='float32')
    labelOutputs = np.ones((keypointsNumber, 3), dtype = 'float32')

    new_Xmin = max(bndbox[index][0], 0)
    new_Ymin = max(bndbox[index][1], 0)
    new_Xmax = min(bndbox[index][2], depth_img.shape[1] - 1)
    new_Ymax = min(bndbox[index][3], depth_img.shape[0] - 1)

    imCrop = depth_img.copy()[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)]


    if bndbox[index][0] < 0 or bndbox[index][1] < 0 or bndbox[index][2] > imgWidth or bndbox[index][3] > imgHeight:
        img_padded = np.zeros((int(bndbox[index][3] - bndbox[index][1]), int(bndbox[index][2] - bndbox[index][0])))
        start1 = 0
        start2 = 0
        end1 = img_padded.shape[0]
        end2 = img_padded.shape[1]
        if bndbox[index][1] < 0:
            start1 = 0 - bndbox[index][1]
        if bndbox[index][0] < 0:
            start2 = 0 - bndbox[index][0]
        if bndbox[index][3] > imgHeight:
            end1 = imCrop.shape[0] + start1
        if bndbox[index][2] > imgWidth:
            end2 = imCrop.shape[1] + start2
        for i in range(img_padded.shape[0]):
            for j in range(img_padded.shape[1]):
                if start1 < i < end1 and start2 < j < end2 and int(i - start1) < imCrop.shape[0] and int(j - start2) < imCrop.shape[1]:
                    img_padded[i][j] = \
                        imCrop[int(i - start1)][int(j - start2)]
        imCrop = img_padded

    imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)
    imgResize = np.asarray(imgResize,dtype = 'float32')  # H*W*C
    imgResize = (imgResize - MEAN) / STD

    imageOutputs[:, :, 0] = imgResize

    imageOutputs = np.asarray(imageOutputs)
    imageNCHWOut = imageOutputs.transpose(2, 0, 1)  # [H, W, C] --->>>  [C, H, W]
    imageNCHWOut = np.asarray(imageNCHWOut)
    labelOutputs = np.asarray(labelOutputs)

    data = torch.from_numpy(imageNCHWOut)

    return data, depth_img, index



######################   Pytorch dataloader   #################
class my_dataloader(torch.utils.data.Dataset):

    def __init__(self, imageDir, bndbox, keypointsPixel, keypointsWorld, mode, augment = True):
        self.imageDir = imageDir
        self.mean = MEAN
        self.std = STD
        self.bndbox = bndbox
        self.keypointsPixel = keypointsPixel
        self.keypointsWorld = keypointsWorld
        self.mode = mode
        self.augment = augment


    def __getitem__(self, index):
        data, orig_img, index = dataPreprocess(index, self.imageDir, self.keypointsPixel, self.keypointsWorld,
                                     self.bndbox, self.mode, self.augment)
        return data, orig_img, index

    def __len__(self):
        return len(self.bndbox)

test_image_datasets = my_dataloader(testingImageDir, bndbox_test, keypoints2d_test, keypointsUVD_test,
                                     mode = 'TEST', augment=False)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size = batch_size,
                                             shuffle = False, num_workers = 8)


def main():
    net = model.A2J_model(num_classes = keypointsNumber)
    net.load_state_dict(torch.load(model_dir))
    net = net.cuda()
    net.eval()

    post_precess = anchor.post_process(shape=[cropHeight // 16, cropWidth // 16], stride=16, P_h=None, P_w=None)
    output = torch.FloatTensor()

    for i, (img, orig_img, index) in tqdm(enumerate(test_dataloaders)):
        with torch.no_grad():
            img = img.cuda()

            heads = net(img)
            pred_keypoints = post_precess(heads, voting=False)
            output = torch.cat([output, pred_keypoints.data.cpu()], 0)

    result = output.cpu().data.numpy()
    np.save('C:/Users/USS00019/pop_net_multiperson_3d_pose_estimation/A2J_experiments' \
            '/bgaug_result_gtbox/result.npy', result)

    result = np.load('C:/Users/USS00019/pop_net_multiperson_3d_pose_estimation/A2J_experiments' \
                     '/bgaug_result_gtbox/result.npy')
    Test1_ = result.copy()
    Test1_[:, :, 0] = result[:, :, 1]
    Test1_[:, :, 1] = result[:, :, 0]
    Test1 = Test1_  # [x, y, z]

    for i in range(len(Test1_)):
        Test1[i, :, 0] = Test1_[i, :, 0] * (bndbox_test[i, 2] - bndbox_test[i, 0]) / cropWidth + bndbox_test[
            i, 0]  # x
        Test1[i, :, 1] = Test1_[i, :, 1] * (bndbox_test[i, 3] - bndbox_test[i, 1]) / cropHeight + bndbox_test[
            i, 1]  # y
        Test1[i, :, 2] = Test1_[i, :, 2]

    Test2d = Test1.copy()[:, :, 0:2]

    TestWorld = np.ones((len(Test1), keypointsNumber, 3))
    TestWorld_tuple = pixel2world(Test1[:, :, 0], Test1[:, :, 1], Test1[:, :, 2])

    TestWorld[:, :, 0] = TestWorld_tuple[0]
    TestWorld[:, :, 1] = TestWorld_tuple[1]
    TestWorld[:, :, 2] = Test1[:, :, 2]
    single_img = np.zeros((480, 512))

    pred2d_set = []
    pred3d_set = []
    pred2d_single = []
    pred3d_single = []

    count = 0
    for i in tqdm(range(len(test_image_ids))):
        single_img = np.load(os.path.join(testingImageDir, test_image_ids[i])).astype(np.float)
        depth_max = 6
        single_img[single_img >= depth_max] = depth_max
        single_img /= depth_max
        single_img *= 255
        if i != 0 and test_image_ids[i] != test_image_ids[i - 1]:
            single_img = np.load(os.path.join(testingImageDir, test_image_ids[i])).astype(np.float)
            depth_max = 6
            single_img[single_img >= depth_max] = depth_max
            single_img /= depth_max
            single_img *= 255

        single_img = single_img.astype(np.uint8)
        if single_img.shape == (512, 480):
            single_img = cv2.cvtColor(single_img, cv2.COLOR_GRAY2BGR)
        pred2d_single.append(Test2d[i].tolist())
        pred3d_single.append(TestWorld[i].tolist())
        single_img = draw_humans_visibility(single_img,
                                            [Test2d[i]],
                                            #[keypoints2d_test[i][0]],
                                            kp_connections(get_keypoints()),
                                            jointColors)
        if i == len(test_image_ids) - 1:
            count += 1
            pred2d_set.append(pred2d_single)
            pred3d_set.append(pred3d_single)
            cv2.imwrite(save_dir + '/images/testing/' + 'testing' + str(count) + '.png', single_img)
        if i != len(test_image_ids) - 1 and test_image_ids[i] != test_image_ids[i + 1]:
            pred2d_set.append(pred2d_single)
            pred3d_set.append(pred3d_single)
            pred2d_single = []
            pred3d_single = []
            cv2.imwrite(save_dir + '/images/testing/' + 'testing' + str(count) + '.png', single_img)
            count += 1


    eval_data = {'human_pred_set_2d': pred2d_set,
                 'human_pred_set_3d': pred3d_set,
                 'human_gt_set_2d': keypoints2d_test,
                 'human_gt_set_3d': keypointsUVD_test}


    w_org = imgHeight
    h_org = imgWidth
    dist_th_2d = 0.02 * np.sqrt(w_org ** 2 + h_org ** 2)

    print('\nevaluating in 2D...')
    dist_th_2d = 0.02 * np.sqrt(w_org ** 2 + h_org ** 2)
    joint_avg_dist, joint_KCP = eval_human_dataset_2d(eval_data['human_pred_set_2d'],
                                                      eval_data['human_gt_set_2d'],
                                                      num_joints=keypointsNumber,
                                                      dist_th=dist_th_2d,
                                                      iou_th=0.5)
    joint_names = get_keypoints()
    print('     2D threshold: {:03f}'.format(dist_th_2d))
    for i, name in enumerate(joint_names):
        print('     joint: {},  PCK: {:03f}, avg 2D error: {:03f}'.format(name,
                                                                          joint_KCP[i],
                                                                          joint_avg_dist[i]))

    print('\n     Overall: PCK: {:03f}, avg 2D error: {:03f} \n'.format(np.average(joint_KCP),
                                                                        np.average(joint_avg_dist)))

    #####################################################################################
    print('\nevaluating in 3D...')
    joint_avg_dist, joint_KCP = eval_human_dataset_3d(eval_data['human_pred_set_2d'],
                                                      eval_data['human_gt_set_2d'],
                                                      eval_data['human_pred_set_3d'],
                                                      eval_data['human_gt_set_3d'],
                                                      num_joints=keypointsNumber,
                                                      dist_th=0.1,
                                                      iou_th=0.5)
    joint_names = get_keypoints()
    print('     3D threshold: {:03f} meter'.format(0.1))
    for i, name in enumerate(joint_names):
        print('     joint: {},  PCK: {:03f}, avg 3D error: {:03f}'.format(name,
                                                                          joint_KCP[i],
                                                                          joint_avg_dist[i]))

    print('\n     Overall: PCK: {:03f}, avg 3D error: {:03f} \n'.format(np.average(joint_KCP),
                                                                        np.average(joint_avg_dist)))



if __name__ == '__main__':
    main()
