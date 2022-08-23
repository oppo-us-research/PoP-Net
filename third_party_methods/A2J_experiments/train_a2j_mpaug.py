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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from glob import glob
from random import uniform
import sys
sys.path.insert(0, '..')
from A2J_experiments import model, anchor, resnet, random_erasing


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# DataHyperParms
BgImgFrames = 8680
keypointsNumber = 15
imgWidth = 480
imgHeight = 512
cropWidth = 288
cropHeight = 288
batch_size = 64
learning_rate = 0.00035
Weight_Decay = 1e-4
nepoch = 35
RegLossFactor = 3
spatialFactor = 0.5
RandCropShift = 5
RandshiftDepth = 0.01
RandRotate = 10
depth_factor = 50
intrinsics = {'fx': 504.1189880371094, 'fy': 504.042724609375, 'cx': 231.7421875, 'cy': 320.62640380859375}

def pixel2world(x, y, z):
    worldX = (x - intrinsics['cx']) * z / intrinsics['fx']
    worldY = (y - intrinsics['cy']) * z / intrinsics['fy']
    return worldX, worldY


save_dir = './mp_result'

try:
    os.makedirs(save_dir)
except OSError:
    pass



#directories on the server:
trainingImageDir = '/raid/yuliangguo/Datasets/Kinect_Depth_Human3D/depth_maps/'
testingImageDir = '/raid/yuliangguo/Datasets/Kinect_Depth_Human3D/depth_maps/'
maskDir = '/raid/yuliangguo/Datasets/Kinect_Depth_Human3D/seg_maps/'
bgDir = '/raid/yuliangguo/Datasets/Kinect_Depth_Human3D/bg_maps/'
bg_anno = '/raid/yuliangguo/Datasets/Kinect_Depth_Human3D/labels/labels_bg.json'
bg_list = list(json.load(open(bg_anno, 'r')).values())

MEAN = 3
STD = 2
depth_max = 6
aug_mods = [[0, 3], [1, 2], [0, 1], [2, 3], [4]]

ANNOTATIONS_TRAIN = sorted(glob('{}/labels_train_*.json'.format('/raid/yuliangguo/Datasets/Kinect_Depth_Human3D/labels')))

annotations_train = []
keypointsUVD_train = []
keypoints2d_train = []
bndbox_train = []
train_image_ids = []
TrainImgFrames = 0
for file in ANNOTATIONS_TRAIN:
    with open(file) as f:
        dict = json.load(f)
        annotations_train.append(dict)
    keypointsUVD_train_single = []
    keypoints2d_train_single = []
    bndbox_train_single = []
    for key, value in dict.items():
        if key != 'intrinsics':
            temp = value[0]
            keypointsUVD_train_single.append(temp['3d_joints'])
            keypoints2d_train_single.append(temp['2d_joints'])
            bndbox_train_single.append(temp['bbox'])
    keypointsUVD_train.append(keypointsUVD_train_single)
    keypoints2d_train.append(keypoints2d_train_single)
    bndbox_train.append(bndbox_train_single)
    train_image_ids.append(list(dict.keys()))
    TrainImgFrames += len(keypoints2d_train_single)


ANNOTATIONS_TEST = sorted(glob('{}/labels_test_*.json'.format('/raid/yuliangguo/Datasets/Kinect_Depth_Human3D/labels')))

annotations_test = []
keypointsUVD_test = []
keypoints2d_test = []
bndbox_test = []
test_image_ids = []
for file in ANNOTATIONS_TEST:
    with open(file) as f:
        dict = json.load(f)
        annotations_test.append(dict)
    for key, value in dict.items():
        if key != 'intrinsics':
            temp = value[0]
            keypointsUVD_test.append(temp['3d_joints'])
            keypoints2d_test.append(temp['2d_joints'])
            bndbox_test.append(temp['bbox'])
    test_image_ids.append(list(dict.keys()))
bndbox_test = np.asarray(bndbox_test)
keypoints2d_test = np.asarray(keypoints2d_test)
keypointsUVD_test = np.asarray(keypointsUVD_test)
TestImgFrames = len(keypoints2d_test)



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

def evaluation2D_perJoint(source, target, dist_th_2d):
    assert np.shape(source) == np.shape(target), "source has different shape with target"
    count = 0
    acc_vec = []

    for j in range(keypointsNumber):
        for i in range(len(source)):
            if np.square(source[i, j, 0] - target[i, j, 0]) + np.square(
                    source[i, j, 1] - target[i, j, 1]) < np.square(dist_th_2d):
                count = count + 1

        accuracy = count / (len(source))
        print('joint_', j, joint_id_to_name[j], ', accuracy: ', accuracy)
        acc_vec.append(accuracy)
        accuracy = 0
        count = 0


def evaluation2D(source, target, dist_th_2d):
    assert np.shape(source) == np.shape(target), "source has different shape with target"
    count = 0
    for i in range(len(source)):
        for j in range(keypointsNumber):
            if np.square(source[i, j, 0] - target[i, j, 0]) + np.square(
                    source[i, j, 1] - target[i, j, 1]) < np.square(dist_th_2d):
                count = count + 1
    accuracy = count / (len(source) * keypointsNumber)
    return accuracy

def dataPreprocess(image, anns, mode, augment = True):
    # if mode == 'TEST':
    #     depth_img = np.load(os.path.join(imgDir, test_image_ids[index])).astype(np.float)
    #     mask = np.load(os.path.join(maskDir, test_image_ids[index])).astype(np.float)
    # else:
    #     depth_img = np.load(os.path.join(imgDir, train_image_ids[index])).astype(np.float)
    #     mask = np.load(os.path.join(maskDir, train_image_ids[index])).astype(np.float)
    #
    # # TODO: add background augmentations here
    # bg_id = index % BgImgFrames
    # bg_image_path = os.path.join(bgDir, bg_list[bg_id]['file_name'])
    # bg_image = np.load(bg_image_path)
    # image = depth_img * mask + bg_image * (np.ones_like(mask) - mask)

    imageOutputs_list = []
    labelOutputs_list = []
    imageOutputs = np.ones((cropHeight, cropWidth, 1), dtype='float32')
    labelOutputs = np.ones((keypointsNumber, 3), dtype = 'float32')
    if augment:
        RandomOffset_1 = np.random.randint(-1 * RandCropShift, RandCropShift)
        RandomOffset_2 = np.random.randint(-1 * RandCropShift, RandCropShift)
        RandomOffset_3 = np.random.randint(-1 * RandCropShift, RandCropShift)
        RandomOffset_4 = np.random.randint(-1 * RandCropShift, RandCropShift)
        RandomOffsetDepth = np.random.normal(0, RandshiftDepth, cropHeight * cropWidth).reshape(cropHeight, cropWidth)
        RandomOffsetDepth[np.where(RandomOffsetDepth < RandshiftDepth)] = 0
        RandomRotate = np.random.randint(-1 * RandRotate, RandRotate)
        RandomScale = np.random.randint(700, 1700)
        RandomScale /= 1000
        #RandomScale = 1
        matrix = cv2.getRotationMatrix2D((cropWidth / 2, cropHeight / 2), RandomRotate, RandomScale)
    else:
        RandomOffset_1, RandomOffset_2, RandomOffset_3, RandomOffset_4 = 0, 0, 0, 0
        RandomRotate = 0
        RandomScale = 1
        RandomOffsetDepth = 0
        matrix = cv2.getRotationMatrix2D((cropWidth / 2, cropHeight / 2), RandomRotate, RandomScale)

    img_list = []
    keypoints_list = []
    imCrop = []
    img_padded = []
    for ann in anns:
        if ann == []:
            break
        new_Xmin = max(ann['bbox'][0], 0)
        new_Ymin = max(ann['bbox'][1], 0)
        new_Xmax = min(ann['bbox'][2], image.shape[1] - 1)
        new_Ymax = min(ann['bbox'][3], image.shape[0] - 1)

        imCrop = image.copy()[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)]
        img_padded = np.zeros((int(ann['bbox'][3] - ann['bbox'][1]), int(ann['bbox'][2] - ann['bbox'][0])))
        start1 = 0
        start2 = 0
        end1 = img_padded.shape[0]
        end2 = img_padded.shape[1]
        if ann['bbox'][0] < 0 or ann['bbox'][1] < 0 or ann['bbox'][2] > imgWidth or ann['bbox'][3] > imgHeight:
            if ann['bbox'][1] < 0:
                start1 = 0 - ann['bbox'][1]
            if ann['bbox'][0] < 0:
                start2 = 0 - ann['bbox'][0]
            if ann['bbox'][3] > imgHeight:
                end1 = imCrop.shape[0] + start1
            if ann['bbox'][2] > imgWidth:
                end2 = imCrop.shape[1] + start2

            for i in range(img_padded.shape[0]):
                for j in range(img_padded.shape[1]):
                    if start1 < i < end1 and start2 < j < end2 and int(i - start1) < imCrop.shape[0] and int(
                            j - start2) < \
                            imCrop.shape[1]:
                        img_padded[i][j] = \
                            imCrop[int(i - start1)][int(j - start2)]
            imCrop = img_padded.copy()


        imgResize = cv2.resize(imCrop.copy(), (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)
        imgResize = np.asarray(imgResize, dtype='float32')  # H*W*C
        imgResize = (imgResize - MEAN) / STD
        img_list.append(imgResize)

        label_xy = np.ones((keypointsNumber, 2), dtype='float32')
        keypoints2d = np.asarray(ann['2d_joints'])
        label_xy[:, 0] = \
            (keypoints2d[:, 0] - new_Xmin + start2) * cropWidth / img_padded.shape[1]
        label_xy[:, 1] = (keypoints2d[:, 1] - new_Ymin + start1) * cropHeight / img_padded.shape[0]  # y
        keypoints_list.append(label_xy)


    if augment:
        for i in range(len(img_list)):
            img_list[i], keypoints_list[i] = transform(img_list[i], keypoints_list[i], matrix)

    for img in img_list:
        imageOutputs[:, :, 0] = img
        imageOutputs = np.asarray(imageOutputs)
        imageNCHWOut = imageOutputs.transpose(2, 0, 1)  # [H, W, C] --->>>  [C, H, W]
        imageNCHWOut = np.asarray(imageNCHWOut)
        imageOutputs_list.append(imageNCHWOut)

    for label in keypoints_list:
        labelOutputs[:, 1] = label[:, 0]
        labelOutputs[:, 0] = label[:, 1]
        keypoints_world = np.asarray(ann['3d_joints'])
        labelOutputs[:, 2] = (keypoints_world[:, 2])
        labelOutputs_list.append(labelOutputs)

    imageOutputs_list = np.asarray(imageOutputs_list)
    labelOutputs_list = np.asarray(labelOutputs_list)

    index_r = np.random.randint(len(imageOutputs_list))
    bndbox = anns[index_r]['bbox']

    data, label = torch.from_numpy(imageOutputs_list[index_r]), torch.from_numpy(labelOutputs_list[index_r])

    return data, label, bndbox


######################   Pytorch dataloader   #################
class my_dataloader(torch.utils.data.Dataset):

    def __init__(self, imageDir, mode, augment = True):
        self.imageDir = imageDir
        self.mean = MEAN
        self.std = STD
        self.mode = mode
        if self.mode == 'TRAINING':
            self.ids_list = train_image_ids
            self.anno_dict_list = annotations_train
        else:
            self.ids_list = test_image_ids
            self.anno_dict_list = annotations_test
        self.augment = augment


    def __getitem__(self, index):
        if self.mode == 'TRAIN':
            ids_list = train_image_ids
            anno_dict_list = annotations_train
        else:
            ids_list = test_image_ids
            anno_dict_list = annotations_test
        image = np.ones([imgHeight, imgWidth]) * 2 * depth_max
        fg_union = np.zeros([imgHeight, imgWidth])
        anns = []
        mod_id = random.randint(0, len(aug_mods)-1)
        for ii in aug_mods[mod_id]:
            if uniform(0, 1) > 0.8:
                continue
            image_id = ids_list[ii][index % len(ids_list[ii])]
            anns += copy.deepcopy(anno_dict_list[ii][image_id])

            image_ii = np.load(os.path.join(self.imageDir, image_id)).astype(np.float)
            fg_mask_ii = np.load(os.path.join(maskDir, image_id)).astype(np.float)

            # z-buffer composition of fg depth maps
            image[fg_mask_ii > 0] = np.minimum((image_ii * fg_mask_ii)[fg_mask_ii > 0], image[fg_mask_ii > 0])
            fg_union = np.maximum(fg_union, fg_mask_ii)

        #make sure at least one example is added
        if len(anns) == 0:
            ii = random.randint(0, len(ids_list)-1)
            image_id = ids_list[ii][index % len(ids_list[ii])]
            anns += copy.deepcopy(anno_dict_list[ii][image_id])

            image_ii = np.load(os.path.join(self.imageDir, image_id)).astype(np.float)
            fg_mask_ii = np.load(os.path.join(maskDir, image_id)).astype(np.float)

            # z-buffer composition of fg depth maps
            image[fg_mask_ii > 0] = np.minimum((image_ii * fg_mask_ii)[fg_mask_ii > 0], image[fg_mask_ii > 0])
            fg_union = np.maximum(fg_union, fg_mask_ii)

        bg_id = index % BgImgFrames
        bg_image_path = os.path.join(bgDir, bg_list[bg_id]["file_name"])
        bg_image = np.load(bg_image_path)

        # compose fg bg depth map using fg mask
        image = image * fg_union + bg_image * (np.ones_like(fg_union) - fg_union)

        data, label, bndbox = dataPreprocess(image, anns, self.mode, self.augment)
        return data, label, bndbox

    def __len__(self):
        if self.mode == 'TRAIN':
            return TrainImgFrames
        else:
            return TestImgFrames

train_image_datasets = my_dataloader(trainingImageDir, mode = 'TRAIN', augment=True)
train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size = batch_size,
                                             shuffle = True, num_workers = 8)



def train():
    net = model.A2J_model(num_classes=keypointsNumber)
    net = net.cuda()

    post_precess = anchor.post_process(shape=[cropWidth // 16, cropHeight// 16], stride=16, P_h=None, P_w=None)
    criterion = anchor.A2J_loss(shape=[cropWidth// 16, cropHeight // 16], thres=[16.0, 32.0], stride=16, \
                                spatialFactor=spatialFactor, img_shape=[cropWidth, cropHeight], P_h=None, P_w=None)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=Weight_Decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                        filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
    logging.info('======================================================')

    for epoch in range(nepoch):
        net = net.train()
        train_loss_add = 0.0
        Cls_loss_add = 0.0
        Reg_loss_add = 0.0
        timer = time.time()
        # Training loop
        for i, (img, label, bndbox) in enumerate(train_dataloaders):
            torch.cuda.synchronize()

            img, label = img.cuda(), label.cuda()
            heads = net(img)
            optimizer.zero_grad()

            depth_max = 6
            depth_mean = MEAN
            depth_std = STD
            img, label = img.cpu(), label.cpu()
            img_ = img.numpy().copy()
            img_ = img_[0, :, :]
            label = label.numpy()
            single_label = label[0]
            label_new = np.copy(single_label)
            label_new[:, 0] = single_label[:, 1]
            label_new[:, 1] = single_label[:, 0]

            human_2d = [label_new]

            single_img = np.copy(img_)
            single_img = single_img[0, :, :]
            single_img *= depth_std
            single_img += depth_mean
            single_img[single_img >= depth_max] = depth_max
            single_img /= depth_max
            single_img *= 255
            single_img = single_img.astype(np.uint8)
            single_img = cv2.cvtColor(single_img, cv2.COLOR_GRAY2BGR)
            single_img = draw_humans_visibility(single_img,
                                                human_2d,
                                                kp_connections(get_keypoints()),
                                                jointColors)
            single_img = single_img.astype(np.uint8)
            cv2.imwrite(save_dir + '/images/input/input' + str(i) + '.png', single_img)
            label = torch.from_numpy(label).cuda()

            Cls_loss, Reg_loss = criterion(heads, label)

            loss = 1 * Cls_loss + Reg_loss * RegLossFactor
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()

            train_loss_add = train_loss_add + (loss.item()) * len(img)
            Cls_loss_add = Cls_loss_add + (Cls_loss.item()) * len(img)
            Reg_loss_add = Reg_loss_add + (Reg_loss.item()) * len(img)

            # printing loss info
            if i % 10 == 0:
                print('epoch: ', epoch, ' step: ', i, 'Cls_loss ', Cls_loss.item(), 'Reg_loss ', Reg_loss.item(),
                      ' total loss ', loss.item())

        scheduler.step(epoch)

        # time taken
        torch.cuda.synchronize()
        timer = time.time() - timer
        timer = timer / TrainImgFrames
        print('==> time to learn 1 sample = %f (ms)' % (timer * 1000))

        train_loss_add = train_loss_add / TrainImgFrames
        Cls_loss_add = Cls_loss_add / TrainImgFrames
        Reg_loss_add = Reg_loss_add / TrainImgFrames
        print('mean train_loss_add of 1 sample: %f, #train_indexes = %d' % (train_loss_add, TrainImgFrames))
        print('mean Cls_loss_add of 1 sample: %f, #train_indexes = %d' % (Cls_loss_add, TrainImgFrames))
        print('mean Reg_loss_add of 1 sample: %f, #train_indexes = %d' % (Reg_loss_add, TrainImgFrames))

        if (epoch % 1 == 0):
                              
            net = net.eval()
            output = torch.FloatTensor()
            outputTrain = torch.FloatTensor()

            saveNamePrefix = '%s/net_%d_wetD_' % (save_dir, epoch) + str(Weight_Decay) + '_depFact_' + str(
                spatialFactor) + '_RegFact_' + str(RegLossFactor) + '_rndShft_' + str(RandCropShift)
            torch.save(net.state_dict(), saveNamePrefix + '.pth')

        logging.info('Epoch#%d: total loss=%.4f, Cls_loss=%.4f, Reg_loss=%.4f, lr = %.6f'
                     % (epoch, train_loss_add, Cls_loss_add, Reg_loss_add, scheduler.get_lr()[0]))


if __name__ == '__main__':
    train()
