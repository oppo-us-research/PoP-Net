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
import sys
sys.path.insert(0, '..')
from A2J_experiments import model, anchor, resnet, random_erasing


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# DataHyperParms
TrainImgFrames = 14674
TestImgFrames = 29331
keypointsNumber = 15
imgWidth = 480
imgHeight = 512
cropWidth = 288
cropHeight = 288
batch_size = 8
learning_rate = 0.00035
Weight_Decay = 1e-4
nepoch = 35
RegLossFactor = 3
spatialFactor = 0.5
RandCropShift = 5
RandshiftDepth = 0.01
RandRotate = 10
RandScale = (1.0, 0.5)
depth_factor = 50
intrinsics = {'fx': 504.1189880371094, 'fy': 504.042724609375, 'cx': 231.7421875, 'cy': 320.62640380859375}

def pixel2world(x, y, z):
    worldX = (x - intrinsics['cx']) * z / intrinsics['fx']
    worldY = (y - intrinsics['cy']) * z / intrinsics['fy']
    return worldX, worldY


save_dir = './result'

try:
    os.makedirs(save_dir)
except OSError:
    pass



#local directories:

testingImageDir = 'C:/Users/USS00019/pop_net_multiperson_3d_pose_estimation/A2J_experiments' \
                  '/Kinect_Depth_Human3D_small/depth_maps/testing'

model_dir = 'C:/Users/USS00019/pop_net_multiperson_3d_pose_estimation/A2J_experiments' \
                  '/result/net_34_wetD_0.0001_depFact_0.5_RegFact_3_rndShft_5.pth'

# #directories on the server:
# trainingImageDir = '/raid/xuenan/data/ITOP/ITOP_side_train_depth_map/'
# testingImageDir = '/raid/xuenan/data/ITOP/ITOP_side_test_depth_map/'

MEAN = 3
STD = 2

with open('C:/Users/USS00019/pop_net_multiperson_3d_pose_estimation/A2J_experiments'
          '/Kinect_Depth_Human3D_small/labels/labels_test.json') as f:
    dict = json.load(f)
    annotations_test = dict

keypointsUVD_test = []
keypoints2d_test = []
bndbox_test = []
for key,value in dict.items():
    if key != 'intrinsics':
        temp = value[0]
        keypointsUVD_test.append(temp['3d_joints'])
        keypoints2d_test.append(temp['2d_joints'])
        bndbox_test.append(temp['bbox'])

keypointsUVD_test = np.asarray(keypointsUVD_test, dtype=np.float32)
keypoints2d_test = np.asarray(keypoints2d_test, dtype=np.float32)
bndbox_test = np.asarray(bndbox_test, dtype=np.float32)

test_image_ids = list(annotations_test.keys())

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
    c = 0
    for i in range(len(source)):
        for j in range(keypointsNumber):
            if np.square(source[i, j, 0] - target[i, j, 0]) + np.square(
                    source[i, j, 1] - target[i, j, 1]) < np.square(dist_th_2d):
                count = count + 1
            else:
                a = source[i, j]
                b = target[i, j]
                c += 1
    accuracy = count / (len(source) * keypointsNumber)
    return accuracy




def dataPreprocess(index, imgDir, keypointsPixel, keypointsWorld, bndbox, mode, augment = True):
    if mode == 'TEST':
        depth_img = np.load(os.path.join(imgDir, test_image_ids[index])).astype(np.float)
    else:
        depth_img = np.load(os.path.join(imgDir, train_image_ids[index])).astype(np.float)

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
        #RandomScale = np.random.rand() * RandScale[0] + RandScale[1]
        RandomScale = np.random.randint(900, 1100)
        RandomScale /= 1000
        #RandomScale = 1
        matrix = cv2.getRotationMatrix2D((cropWidth / 2, cropHeight / 2), RandomRotate, RandomScale)
    else:
        RandomOffset_1, RandomOffset_2, RandomOffset_3, RandomOffset_4 = 0, 0, 0, 0
        RandomRotate = 0
        RandomScale = 1
        RandomOffsetDepth = 0
        matrix = cv2.getRotationMatrix2D((cropWidth / 2, cropHeight / 2), RandomRotate, RandomScale)

    new_Xmin = max(bndbox[index][0], 0)
    new_Ymin = max(bndbox[index][1], 0)
    new_Xmax = min(bndbox[index][2], depth_img.shape[1] - 1)
    new_Ymax = min(bndbox[index][3], depth_img.shape[0] - 1)

    imCrop = depth_img.copy()[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)]
    imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)
    imgResize = np.asarray(imgResize,dtype = 'float32')  # H*W*C
    imgResize = (imgResize - MEAN) / STD

    label_xy = np.ones((keypointsNumber, 2), dtype = 'float32')
    label_xy[:,0] = (keypointsPixel[index,:,0] - new_Xmin)*cropWidth/(new_Xmax - new_Xmin)
    label_xy[:,1] = (keypointsPixel[index,:,1] - new_Ymin)*cropHeight/(new_Ymax - new_Ymin) # y

    if augment:
        imgResize, label_xy = transform(imgResize, label_xy, matrix)

    imageOutputs[:, :, 0] = imgResize

    labelOutputs[:, 1] = label_xy[:, 0]
    labelOutputs[:, 0] = label_xy[:, 1]
    labelOutputs[:, 2] = (keypointsWorld.copy()[index, :, 2]) / depth_factor

    imageOutputs = np.asarray(imageOutputs)
    imageNCHWOut = imageOutputs.transpose(2, 0, 1)  # [H, W, C] --->>>  [C, H, W]
    imageNCHWOut = np.asarray(imageNCHWOut)
    labelOutputs = np.asarray(labelOutputs)

    data, label = torch.from_numpy(imageNCHWOut), torch.from_numpy(labelOutputs)

    return data, label, depth_img, index


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
        data, label, orig_img, index = dataPreprocess(index, self.imageDir, self.keypointsPixel, self.keypointsWorld,
                                     self.bndbox, self.mode, self.augment)
        return data, label, orig_img, index

    def __len__(self):
        return len(self.bndbox)

test_image_datasets = my_dataloader(testingImageDir, bndbox_test, keypoints2d_test, keypointsUVD_test,
                                     mode = 'TEST', augment=False)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size = batch_size,
                                             shuffle = False, num_workers = 8)

writer = SummaryWriter()



def test():
    net = model.A2J_model(num_classes = keypointsNumber)
    net.load_state_dict(torch.load(model_dir))
    net = net.cuda()
    net.eval()

    post_precess = anchor.post_process(shape=[cropHeight // 16, cropWidth // 16], stride=16, P_h=None, P_w=None)
    output = torch.FloatTensor()

    # for i, (img, label, orig_img, index) in tqdm(enumerate(test_dataloaders)):
    #     with torch.no_grad():
    #         img, label = img.cuda(), label.cuda()
    #
    #         heads = net(img)
    #         pred_keypoints = post_precess(heads, voting=False)
    #         output = torch.cat([output, pred_keypoints.data.cpu()], 0)
    #
    #
    #         #visualize results. Denormalize each pixel value and make it fit the values for colors (0, 255)
    #         # single_bndbox = bndbox_test[index]
    #         # depth_max = 6
    #         # depth_mean = MEAN
    #         # depth_std = STD
    #         # img, label = img.cpu(), label.cpu()
    #         # img_ = img.numpy().copy()
    #         # img_ = img_[0, 0, :, :]
    #         # orig_img_ = orig_img.numpy().copy()
    #         # orig_img_ = orig_img_[0]
    #         # orig_img_[orig_img_ >= depth_max] = depth_max
    #         # orig_img_ /= depth_max
    #         # orig_img_ *= 255
    #         # single_bndbox = single_bndbox[0]
    #         # pred_keypoints = pred_keypoints.cpu().numpy()
    #         # pred = pred_keypoints[0]
    #         # pred_new = pred.copy()
    #         # pred_new[:, 0] = pred[:, 1]
    #         # pred_new[:, 1] = pred[:, 0]
    #         # pred_new = [pred_new]
    #         #
    #         # single_img = np.copy(img_)
    #         # single_img *= depth_std
    #         # single_img += depth_mean
    #         #
    #         # single_img[single_img >= depth_max] = depth_max
    #         # single_img /= depth_max
    #         # single_img *= 255
    #         # single_img = single_img.astype(np.uint8)
    #         # single_img = cv2.cvtColor(single_img, cv2.COLOR_GRAY2BGR)
    #         # single_img = draw_humans_visibility(single_img,
    #         #                                     pred_new,
    #         #                                     kp_connections(get_keypoints()),
    #         #                                     jointColors)
    #         # single_img = cv2.resize(single_img, (int(single_bndbox[2]) - int(single_bndbox[0]),
    #         #                                      int(single_bndbox[3]) - int(single_bndbox[1])),
    #         #                         interpolation=cv2.INTER_NEAREST)
    #         #
    #         # #conbine original image with processed image with skeleton drawn inside the bounding box area
    #         # reconstructed_img = np.zeros((imgHeight, imgWidth))
    #         #
    #         # for k in range(imgHeight):
    #         #     for j in range(imgWidth):
    #         #         if k > int(single_bndbox[1]) and k < int(single_bndbox[3]) and j > int(
    #         #                 single_bndbox[0]) and j < int(single_bndbox[2]):
    #         #             reconstructed_img[k][j] = single_img[k - int(single_bndbox[1])][j - int(single_bndbox[0])]
    #         #         else:
    #         #             reconstructed_img[k][j] = orig_img_[k][j]
    #         #
    #         # plt.imshow(reconstructed_img)
    #         # plt.imsave(save_dir + '/images/testing/'+ 'testing' + str(i) + '.png', reconstructed_img)
    #
    # result = output.cpu().data.numpy()
    w_org = imgHeight
    h_org = imgWidth
    dist_th_2d = 0.02 * np.sqrt(w_org ** 2 + h_org ** 2)
    #
    # Test1_ = result.copy()
    # Test1_[:, :, 0] = result[:, :, 1]
    # Test1_[:, :, 1] = result[:, :, 0]
    # Test1 = Test1_  # [x, y, z]
    #
    # for i in range(len(Test1_)):
    #     Test1[i, :, 0] = Test1_[i, :, 0] * (bndbox_test[i, 2] - bndbox_test[i, 0]) / cropWidth + bndbox_test[
    #         i, 0]  # x
    #     Test1[i, :, 1] = Test1_[i, :, 1] * (bndbox_test[i, 3] - bndbox_test[i, 1]) / cropHeight + bndbox_test[
    #         i, 1]  # y
    #     Test1[i, :, 2] = Test1_[i, :, 2] * depth_factor
    #
    # Test2d = Test1.copy()[:, :, 0:2]
    # np.save('C:/Users/USS00019/pop_net_multiperson_3d_pose_estimation/A2J_experiments' \
    #               '/result/Test2d.npy', Test2d)



    Test2d = np.load('C:/Users/USS00019/pop_net_multiperson_3d_pose_estimation/A2J_experiments' \
                   '/result/Test2d.npy')

    for i in range(len(test_image_ids) - 1):
        single_img = np.load(os.path.join(testingImageDir, test_image_ids[i])).astype(np.float)
        depth_max = 6
        depth_mean = MEAN
        depth_std = STD
        single_img *= depth_std
        single_img += depth_mean
        single_img[single_img >= depth_max] = depth_max
        single_img /= depth_max
        single_img *= 255
        single_img = single_img.astype(np.uint8)
        single_img = cv2.cvtColor(single_img, cv2.COLOR_GRAY2BGR)
        single_img = draw_humans_visibility(single_img,
                                           [keypoints2d_test[i]],
                                           kp_connections(get_keypoints()),
                                           jointColors)
        plt.imsave(save_dir + '/images/gt/'+ 'gt' + str(i) + '.png', single_img)


    accuracy_2d = evaluation2D(Test2d, keypoints2d_test, dist_th_2d)
    print("Accuracy 2d:", accuracy_2d)
    evaluation2D_perJoint(Test2d, keypoints2d_test, dist_th_2d)
    # TestWorld = np.ones((len(Test1), keypointsNumber, 3))
    # TestWorld_tuple = pixel2world(Test1[:, :, 0], Test1[:, :, 1], Test1[:, :, 2])

    # TestWorld[:, :, 0] = TestWorld_tuple[0]
    # TestWorld[:, :, 1] = TestWorld_tuple[1]
    # TestWorld[:, :, 2] = Test1[:, :, 2]
    # np.save('C:/Users/USS00019/pop_net_multiperson_3d_pose_estimation/A2J_experiments' \
    #                 '/result/TestWorld.npy', TestWorld)
    TestWorld = np.load('C:/Users/USS00019/pop_net_multiperson_3d_pose_estimation/A2J_experiments' \
                     '/result/TestWorld.npy')
    Accuracy_test = evaluation10CMRule(TestWorld, keypointsUVD_test, bndbox_test)
    print('Accuracy:', Accuracy_test)
    evaluation10CMRule_perJoint(TestWorld, keypointsUVD_test, bndbox_test)

def evaluation10CMRule(source, target, Bndbox):
    assert np.shape(source) == np.shape(target), "source has different shape with target"

    count = 0
    for i in range(len(source)):
        for j in range(keypointsNumber):
            if np.square(source[i, j, 0] - target[i, j, 0]) + np.square(
                    source[i, j, 1] - target[i, j, 1]) + np.square(source[i, j, 2] - target[i, j, 2]) < np.square(
                    0.1):  # 10cm
                count = count + 1
    accuracy = count / (len(source) * keypointsNumber)
    return accuracy

def evaluation10CMRule_perJoint(source, target, Bndbox):
    assert np.shape(source) == np.shape(target), "source has different shape with target"
    count = 0
    accuracy = 0
    for j in range(keypointsNumber):
        for i in range(len(source)):
            if np.square(source[i, j, 0] - target[i, j, 0]) + np.square(
                    source[i, j, 1] - target[i, j, 1]) + np.square(source[i, j, 2] - target[i, j, 2]) < np.square(
                    0.1):  # 10cm
                count = count + 1

        accuracy = count / (len(source))
        print('joint_', j, joint_id_to_name[j], ', accuracy: ', accuracy)
        accuracy = 0
        count = 0

def errorCompute(source, target, Bndbox):
    assert np.shape(source) == np.shape(target), "source has different shape with target"

    Test1_ = source.copy()
    target_ = target.copy()
    Test1_[:, :, 0] = source[:, :, 1]
    Test1_[:, :, 1] = source[:, :, 0]
    Test1 = Test1_  # [x, y, z]

    for i in range(len(Test1_)):
        Test1[i, :, 0] = Test1_[i, :, 0] * (Bndbox[i, 2] - Bndbox[i, 0]) / cropWidth + Bndbox[i, 0]  # x
        Test1[i, :, 1] = Test1_[i, :, 1] * (Bndbox[i, 3] - Bndbox[i, 1]) / cropHeight + Bndbox[i, 1]  # y
        Test1[i, :, 2] = Test1_[i, :, 2] / depth_factor

    TestWorld = np.ones((len(Test1), keypointsNumber, 3))
    TestWorld_tuple = pixel2world(Test1[:, :, 0], Test1[:, :, 1], Test1[:, :, 2])

    TestWorld[:, :, 1] = TestWorld_tuple[0]
    TestWorld[:, :, 0] = TestWorld_tuple[1]
    TestWorld[:, :, 2] = Test1[:, :, 2]


    errors = np.sqrt(np.sum((TestWorld - target_) ** 2, axis=2))

    return np.mean(errors)


if __name__ == '__main__':
    test()

