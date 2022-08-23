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
from evaluate.eval_pose_mp import *
from evaluate.eval_ap_mpii import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# DataHyperParms
keypointsNumber = 15
imgWidth = 480
imgHeight = 512
cropWidth = 288
cropHeight = 288
batch_size = 1
intrinsics = {'fx': 504.1189880371094, 'fy': 504.042724609375, 'cx': 231.7421875, 'cy': 320.62640380859375}


def pixel2world(x, y, z):
    worldX = (x - intrinsics['cx']) * z / intrinsics['fx']
    worldY = (y - intrinsics['cy']) * z / intrinsics['fy']
    return worldX, worldY


# DATA_DIR = '/raid/yuliangguo/Datasets/Kinect_Depth_Human3D/multiperson_test_v2'
DATA_DIR = '/media/yuliang/DATA/Datasets/Kinect_Depth_Human3D_small/multiperson_test_v2'
bbox_dir = 'mp_result/yolo_bbox_mpreal_refine'
save_dir = 'mp_result/result_mpreal_refine_predbox'

with open(DATA_DIR + '/labels_refine/labels_test.json') as f:
    ANNOTATIONS_TEST = json.load(f)

try:
    os.makedirs(save_dir)
except OSError:
    pass


testingImageDir = DATA_DIR + '/depth_maps/'
model_dir = 'mp_result/net_47_wetD_0.0001_depFact_0.5_RegFact_3_rndShft_5.pth'

MEAN = 3
STD = 2
depth_max = 6

# keypointsUVD_test = []
# keypoints2d_test = []
test_image_ids = []

keypoints2d_test_set = []
keypointsUVD_test_set = []
test_image_ids_set = []

# make sure that bndbox and image ids are generated and loaded correctly
bndbox_test = np.load(bbox_dir + '/generated_bndbox_real.npy')
test_images_indices = np.load(bbox_dir + '/generated_indices_real.npy')
image_names = list(ANNOTATIONS_TEST.keys())
for i in test_images_indices:
    test_image_ids.append(image_names[i])


for key, value in ANNOTATIONS_TEST.items():
    if key == 'intrinsics':
        continue

    test_image_ids_set.append(key)
    keypoints2d_frame = []
    keypointsUVD_frame = []

    # if len(value) == 0:
    #     test_image_ids.append(key)
    #     keypointsUVD_test.append([])
    #     keypoints2d_test.append([])
    #     bndbox_test.append([])
    #     keypoints2d_test_set.append([])
    #     keypointsUVD_test_set.append([])
    #     continue

    for frame in value:
        # test_image_ids.append(key)
        # keypoints2d_test.append([frame['2d_joints']])
        # keypointsUVD_test.append([frame['3d_joints']])
        # bndbox_test.append(frame['bbox'])

        keypoints2d_frame.append(frame['2d_joints'])
        keypointsUVD_frame.append(frame['3d_joints'])

    keypoints2d_test_set.append(keypoints2d_frame)
    keypointsUVD_test_set.append(keypointsUVD_frame)

# keypointsUVD_test = np.asarray(keypointsUVD_test, dtype=np.float32)
# keypoints2d_test = np.asarray(keypoints2d_test, dtype=np.float32)
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
            center1 = human_vis[limb[0], :2]
            if np.sum(np.isinf(center1)) > 0:
                continue
            center1 = center1.astype(np.int)
            img = cv2.circle(img, tuple(center1), 3, color, thickness=2, lineType=8, shift=0)

            if visibilities is not None and visibilities[i][limb[1]] < 0.5:
                color = [0, 0, 0]
            else:
                color = [0, 0, 255]
            center2 = human_vis[limb[1], :2]
            if np.sum(np.isinf(center2)) > 0:
                continue
            center2 = center2.astype(np.int)
            img = cv2.line(img, tuple(center1), tuple(center2), jointColors[k], 2)
            img = cv2.circle(img, tuple(center2), 3, color, thickness=2, lineType=8, shift=0)

    return img


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


def dataPreprocess(index, imgDir, bndbox, mode, augment = False):
    depth_img = np.load(os.path.join(imgDir, test_image_ids[index])).astype(np.float)
    imageOutputs = np.zeros((cropHeight, cropWidth, 1), dtype='float32')

    # do not prepare data for invalide data
    if bndbox[index][4] > 0.01:
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

    data = torch.from_numpy(imageNCHWOut)

    return data, depth_img, index


######################   Pytorch dataloader   #################
class my_dataloader(torch.utils.data.Dataset):

    def __init__(self, imageDir, bndbox, mode, augment = True):
        self.imageDir = imageDir
        self.mean = MEAN
        self.std = STD
        self.bndbox = bndbox
        self.mode = mode
        self.augment = augment

    def __getitem__(self, index):
        data, orig_img, index = dataPreprocess(index, self.imageDir, self.bndbox, self.mode, self.augment)
        return data, orig_img, index

    def __len__(self):
        return len(self.bndbox)


test_image_datasets = my_dataloader(testingImageDir, bndbox_test, mode = 'TEST', augment=False)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=batch_size,
                                               shuffle=False, num_workers=8)


def main():
    vis = True
    from thop import profile, clever_format

    net = model.A2J_model(num_classes = keypointsNumber)
    net.load_state_dict(torch.load(model_dir))
    net = net.cuda()
    net.eval()

    post_precess = anchor.post_process(shape=[cropHeight // 16, cropWidth // 16], stride=16, P_h=None, P_w=None)
    output = torch.FloatTensor()

    for i, (img, orig_img, index) in tqdm(enumerate(test_dataloaders)):
        with torch.no_grad():
            img = img.cuda()

            # end = time.time()
            heads = net(img)
            # # measure elapsed time
            # process_time = (time.time() - end)
            # end = time.time()
            # print(' process time: {:3f}/image\t'.format(process_time / batch_size))

            # macs, params = profile(net, inputs=(img,))
            # print("Params(M): {:.3f}, MACs(G): {:.3f}".format(params / 10 ** 6, macs / 10 ** 9))

            pred_keypoints = post_precess(heads, voting=False)
            output = torch.cat([output, pred_keypoints.data.cpu()], 0)

    result = output.cpu().data.numpy()
    np.save(save_dir + '/result.npy', result)

    result = np.load(save_dir + '/result.npy')
    Test1_ = result.copy()
    Test1_[:, :, 0] = result[:, :, 1]
    Test1_[:, :, 1] = result[:, :, 0]
    Test1 = Test1_  # [x, y, z]
    part_conf_all_vec = np.zeros((len(Test1_), keypointsNumber))

    for i in range(len(Test1_)):
        Test1[i, :, 0] = Test1_[i, :, 0] * (bndbox_test[i, 2] - bndbox_test[i, 0]) / cropWidth + bndbox_test[
            i, 0]  # x
        Test1[i, :, 1] = Test1_[i, :, 1] * (bndbox_test[i, 3] - bndbox_test[i, 1]) / cropHeight + bndbox_test[
            i, 1]  # y
        Test1[i, :, 2] = Test1_[i, :, 2]
        part_conf_all_vec[i, :] = bndbox_test[i, 4].astype(np.float)

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
    human_pred_set_part_conf = []
    pred_part_conf_single = []

    # convert human-wise output into frame-wise output
    for i in range(len(test_image_ids)):
        pred2d_single.append(Test2d[i].tolist())
        pred3d_single.append(TestWorld[i].tolist())
        pred_part_conf_single.append(part_conf_all_vec[i].tolist())

        if i == len(test_image_ids) - 1:
            # count += 1
            pred2d_set.append(pred2d_single)
            pred3d_set.append(pred3d_single)
            human_pred_set_part_conf.append(pred_part_conf_single)
        if i != len(test_image_ids) - 1 and test_image_ids[i] != test_image_ids[i + 1]:
            pred2d_set.append(pred2d_single)
            pred3d_set.append(pred3d_single)
            human_pred_set_part_conf.append(pred_part_conf_single)
            pred2d_single = []
            pred3d_single = []
            pred_part_conf_single = []

    # visualization
    if vis:
        print('writing predictions ...')
        count = 0
        for i in tqdm(range(len(test_image_ids_set))):
            single_img = np.load(os.path.join(testingImageDir, test_image_ids_set[i])).astype(np.float)
            depth_max = 6
            single_img[single_img >= depth_max] = depth_max
            single_img /= depth_max
            single_img *= 255
            single_img = single_img.astype(np.uint8)
            if single_img.shape == (512, 480):
                single_img = cv2.cvtColor(single_img, cv2.COLOR_GRAY2BGR)

            single_img = draw_humans_visibility(single_img,
                                                pred2d_set[i],
                                                #[keypoints2d_test[i][0]],
                                                kp_connections(get_keypoints()),
                                                jointColors)

            cv2.imwrite(save_dir + '/{:06d}.jpg'.format(count), single_img)
            count += 1

    eval_data = {'human_pred_set_2d': pred2d_set,
                 'human_pred_set_3d': pred3d_set,
                 'human_gt_set_2d': keypoints2d_test_set,
                 'human_gt_set_3d': keypointsUVD_test_set,
                 'human_pred_set_part_conf': human_pred_set_part_conf}

    # evaluation
    # eval_data = json.load(open(os.path.join(save_dir, 'eval_data.json'), 'r'))
    print('\nevaluating in 2D PCKh-0.5...')
    # dist_th_2d = 0.02 * np.sqrt(args.w_org ** 2 + args.h_org ** 2)
    joint_avg_dist, joint_KCP = eval_human_dataset_2d_PCKh(eval_data['human_pred_set_2d'],
                                                           eval_data['human_gt_set_2d'],
                                                           num_joints=keypointsNumber,
                                                           ind1=0,
                                                           ind2=1,
                                                           iou_th=0.5)
    joint_names = datasets_kdh3d_mpreal.get_keypoints()
    # print('     2D threshold: {:03f}'.format(dist_th_2d))
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

    #####################################################################################
    AP_2D = eval_ap_mpii_v2(eval_data['human_pred_set_2d'], eval_data['human_pred_set_part_conf'],
                            eval_data['human_gt_set_2d'], gt_visibility_set=[],
                            head_id=0, neck_id=1, joint_names=joint_names, thresh=0.5)

    #####################################################################################
    AP_3D = eval_ap_3D(eval_data['human_pred_set_3d'], eval_data['human_pred_set_part_conf'],
                       eval_data['human_gt_set_3d'], gt_visibility_set=[], joint_names=joint_names, thresh=0.1)


if __name__ == '__main__':
    main()
