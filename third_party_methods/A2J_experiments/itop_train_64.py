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
from lib.datasets.heatmap import putGaussianMaps
from lib.datasets.paf import putVecMaps
from lib.datasets.posemap import putLimbZ, putJointZ, putJointDxDy
from lib.datasets import data_augmentation_2d3d
from lib.utils.common import draw_humans_visibility, pos_3d_from_2d_and_depth
from ITOP import datasets_itop as ip
from A2J import model, anchor, resnet, random_erasing
from tqdm import tqdm
from matplotlib.patches import Circle, ConnectionPatch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

fx = 286
fy = -286
u0 = 160
v0 = 120

# DataHyperParms
TrainImgFrames = 17991 #TODO: change this number to the number of frames in training dataset
TestImgFrames = 4863 #TODO: change this number
keypointsNumber = 15 #These 3 numbers are copied from itop_side.py, the testing src code
cropWidth = 288
cropHeight = 288
batch_size = 64  #TODO: try 16, 32, and 64
learning_rate = 0.00035
Weight_Decay = 1e-4
nepoch = 35
RegLossFactor = 3
spatialFactor = 0.5
RandCropShift = 5
RandshiftDepth = 1
RandRotate = 10
RandScale = (1.0, 0.5)
xy_thres = 120
depth_thres = 0.4

def pixel2world(x, fx, fy, ux, uy):
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x


def world2pixel(x, fx, fy, ux, uy):
    x[:, :, 0] = x[:, :, 0] * fx / x[:, :, 2] + ux
    x[:, :, 1] = x[:, :, 1] * fy / x[:, :, 2] + uy
    return x

save_dir = './result/itop64'

try:
    os.makedirs(save_dir)
except OSError:
    pass



# #local directories:
# trainingImageDir = 'C:/Users/USS00019/realtime_multiperson_3d_pose_estimation/ITOP/ITOP_side_train_depth_map/'
# testingImageDir = 'C:/Users/USS00019/realtime_multiperson_3d_pose_estimation/ITOP/'  # mat images
# MEAN = np.load('C:/Users/USS00019/realtime_multiperson_3d_pose_estimation/ITOP/itop_side_mean.npy')
# STD = np.load('C:/Users/USS00019/realtime_multiperson_3d_pose_estimation/ITOP/itop_side_std.npy')
# modelDir = ('C:/Users/USS00019/realtime_multiperson_3d_pose_estimation/'
#             'result/itop16/net_10_wetD_0.0001_depFact_0.5_RegFact_3_rndShft_5.pth')


#directories on the server:
trainingImageDir = '/raid/xuenan/data/ITOP/ITOP_side_train_depth_map/'
testingImageDir = '/raid/xuenan/data/ITOP/ITOP_side_test_depth_map/'
MEAN = np.load('/raid/xuenan/data/ITOP/itop_side_mean.npy')
STD = np.load('/raid/xuenan/data/ITOP/itop_side_std.npy')

annotations_train = dict()
#local directory:
#with open('C:/Users/USS00019/realtime_multiperson_3d_pose_estimation/ITOP/annotations/ITOP_side_train_labels.json') as f:
with open('/raid/xuenan/data/ITOP/annotations/ITOP_side_train_labels.json') as f:
    dict = json.load(f)
    annotations_train = dict
    #key: each frame is a key; value: ['2d_joints', '3d_joints']

keypointsUVD_train = []
for key,value in dict.items():
    temp = value[0]
    keypointsUVD_train.append(temp['3d_joints'])

keypointsUVD_train = np.asarray(keypointsUVD_train, dtype=np.float32)


#with open('C:/Users/USS00019/realtime_multiperson_3d_pose_estimation/ITOP/annotations/ITOP_side_test_labels.json') as f:
with open('/raid/xuenan/data/ITOP/annotations/ITOP_side_test_labels.json') as f:
    dict = json.load(f)
    annotations_test = dict
    #key: each frame is a key; value: ['2d_joints', '3d_joints']

keypointsUVD_test = []
for key,value in dict.items():
    temp = value[0]
    keypointsUVD_test.append(temp['3d_joints'])

keypointsUVD_test = np.asarray(keypointsUVD_test, dtype=np.float32)


center_train = []
#local directory:
#with open('C:/Users/USS00019/A2J/data/itop_side/center_train.txt') as f:
#directory on the server:
with open('/raid/xuenan/data/ITOP/center_train.txt') as f:
    lines = []
    for line in f:
        lines.append(line)
    for line in lines:
        split_line = line.split()
        invalid = False
        for i in range(len(split_line)):
            if split_line[i] == "invalid":
                invalid = True
        if invalid == False:
            inter = []
            inter.append(split_line)
            center_train.append(inter)

center_train = np.array(center_train, dtype=np.float32)

centre_train_world = pixel2world(center_train.copy(), fx, fy, u0, v0)

center_test = []
#local directories:
#with open('C:/Users/USS00019/A2J/data/itop_side/center_test.txt') as f:
with open('/raid/xuenan/data/ITOP/center_test.txt') as f:
    lines = []
    for line in f:
        lines.append(line)
    for line in lines:
        split_line = line.split()
        invalid = False
        for i in range(len(split_line)):
            if split_line[i] == "invalid":
                invalid = True
        if invalid == False:
            inter = []
            inter.append(split_line)
            center_test.append(inter)

center_test = np.array(center_test, dtype=np.float32)

centre_test_world = pixel2world(center_test.copy(), fx, fy, u0, v0)


centerlefttop_train = centre_train_world.copy()
centerlefttop_train[:, 0, 0] = centerlefttop_train[:, 0, 0] - xy_thres
centerlefttop_train[:, 0, 1] = centerlefttop_train[:, 0, 1] + xy_thres

centerrightbottom_train = centre_train_world.copy()
centerrightbottom_train[:, 0, 0] = centerrightbottom_train[:, 0, 0] + xy_thres
centerrightbottom_train[:, 0, 1] = centerrightbottom_train[:, 0, 1] - xy_thres

train_lefttop_pixel = world2pixel(centerlefttop_train, fx, fy, u0, v0)
train_rightbottom_pixel = world2pixel(centerrightbottom_train, fx, fy, u0, v0)


centerlefttop_test = centre_test_world.copy()
centerlefttop_test[:, 0, 0] = centerlefttop_test[:, 0, 0] - xy_thres
centerlefttop_test[:, 0, 1] = centerlefttop_test[:, 0, 1] + xy_thres

centerrightbottom_test = centre_test_world.copy()
centerrightbottom_test[:, 0, 0] = centerrightbottom_test[:, 0, 0] + xy_thres
centerrightbottom_test[:, 0, 1] = centerrightbottom_test[:, 0, 1] - xy_thres

test_lefttop_pixel = world2pixel(centerlefttop_test, fx, fy, u0, v0)
test_rightbottom_pixel = world2pixel(centerrightbottom_test, fx, fy, u0, v0)



#Copied from itop_side.py, the testing src code
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

def dataPreprocess(index, img, keypointsUVD, center, mean, std, lefttop_pixel, rightbottom_pixel, xy_thres=90,
                   depth_thres=0.4, augment=True):
    #keypointsUVD = world2pixel(keypointsUVD, fx, fy, u0, v0)
    imageOutputs = np.ones((cropHeight, cropWidth, 1), dtype='float32')
    labelOutputs = np.ones((keypointsNumber, 3), dtype='float32')

    #visualize(index, img, keypointsUVD)

    if augment:
        RandomOffset_1 = np.random.randint(-1 * RandCropShift, RandCropShift)
        RandomOffset_2 = np.random.randint(-1 * RandCropShift, RandCropShift)
        RandomOffset_3 = np.random.randint(-1 * RandCropShift, RandCropShift)
        RandomOffset_4 = np.random.randint(-1 * RandCropShift, RandCropShift)
        RandomOffsetDepth = np.random.normal(0, RandshiftDepth, cropHeight * cropWidth).reshape(cropHeight, cropWidth)
        RandomOffsetDepth[np.where(RandomOffsetDepth < RandshiftDepth)] = 0
        RandomRotate = np.random.randint(-1 * RandRotate, RandRotate)
        RandomScale = np.random.rand() * RandScale[0] + RandScale[1]
        matrix = cv2.getRotationMatrix2D((cropWidth / 2, cropHeight / 2), RandomRotate, RandomScale)
    else:
        RandomOffset_1, RandomOffset_2, RandomOffset_3, RandomOffset_4 = 0, 0, 0, 0
        RandomRotate = 0
        RandomScale = 1
        RandomOffsetDepth = 0
        matrix = cv2.getRotationMatrix2D((cropWidth / 2, cropHeight / 2), RandomRotate, RandomScale)

    new_Xmin = max(lefttop_pixel[index, 0, 0] + RandomOffset_1, 0)
    new_Ymin = max(lefttop_pixel[index, 0, 1] + RandomOffset_2, 0)
    new_Xmax = min(rightbottom_pixel[index, 0, 0] + RandomOffset_3, img.shape[1] - 1)
    new_Ymax = min(rightbottom_pixel[index, 0, 1] + RandomOffset_4, img.shape[0] - 1)

    imCrop = img[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)].copy()

    imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)

    imgResize = np.asarray(imgResize, dtype='float32')  # H*W*C

    imgResize[np.where(imgResize >= center[index][0][2] + depth_thres)] = center[index][0][2]
    imgResize[np.where(imgResize <= center[index][0][2] - depth_thres)] = center[index][0][2]
    imgResize = (imgResize - center[index][0][2]) * RandomScale

    imgResize = (imgResize - mean[-1]) / std[-1]

    ## label
    label_xy = np.ones((keypointsNumber, 2), dtype='float32')
    label_xy[:, 1] = (keypointsUVD[index, :, 0].copy() - new_Xmin) * cropWidth / (new_Xmax - new_Xmin)  # x
    label_xy[:, 0] = (keypointsUVD[index, :, 1].copy() - new_Ymin) * cropHeight / (new_Ymax - new_Ymin)  # y

    if augment:
        imgResize, label_xy = transform(imgResize, label_xy, matrix)  ## rotation, scale

    imageOutputs[:, :, 0] = imgResize


    labelOutputs[:, 0] = label_xy[:, 0]
    labelOutputs[:, 1] = label_xy[:, 1]
    labelOutputs[:, 2] = (keypointsUVD[index, :, 2] - center[index][0][2]) * RandomScale  # Z

    imageOutputs = np.asarray(imageOutputs)
    imageNCHWOut = imageOutputs.transpose(2, 0, 1)  # [H, W, C] --->>>  [C, H, W]
    imageNCHWOut = np.asarray(imageNCHWOut)
    labelOutputs = np.asarray(labelOutputs)

    data, label = torch.from_numpy(imageNCHWOut), torch.from_numpy(labelOutputs)


    # data_vis = data[0, :, :]
    #
    # fig, ax = plt.subplots(1)
    # ax.imshow(data_vis)
    # for i in range(keypointsNumber):
    #     circ = Circle((label[i][0], label[i][1]), 3)
    #     ax.add_patch(circ)
    #
    # plt.show()

    return data, label


######################   Pytorch dataloader   #################
class my_dataloader(torch.utils.data.Dataset):

    def __init__(self, ImgDir, center, lefttop_pixel, rightbottom_pixel, keypointsUVD, mode,augment=True):
        self.ImgDir = ImgDir
        self.mean = MEAN
        self.std = STD
        self.center = center
        self.lefttop_pixel = lefttop_pixel
        self.rightbottom_pixel = rightbottom_pixel
        self.keypointsUVD = keypointsUVD
        self.xy_thres = xy_thres
        self.depth_thres = depth_thres
        self.mode = mode
        self.augment = augment
        self.randomErase = random_erasing.RandomErasing(probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0])



    def __getitem__(self, index):
        if self.mode == 'TRAIN':
            image_ids = list(annotations_train.keys())
        else:
            image_ids = list(annotations_test.keys())
        depthTemp = np.load(os.path.join(self.ImgDir, image_ids[index] + '.npy')).astype(np.float)

        data, label = dataPreprocess(index, depthTemp, self.keypointsUVD, self.center, self.mean, self.std, \
            self.lefttop_pixel, self.rightbottom_pixel, self.xy_thres, self.depth_thres, self.augment)

        return data, label

    def __len__(self):
        return len(self.center)

train_image_datasets = my_dataloader(trainingImageDir, center_train, train_lefttop_pixel,
                                     train_rightbottom_pixel, keypointsUVD_train, mode = 'TRAIN', augment=True)
train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size = batch_size,
                                             shuffle = True, num_workers = 8)

test_image_datasets = my_dataloader(testingImageDir, center_test, test_lefttop_pixel,
                                    test_rightbottom_pixel, keypointsUVD_test, mode = 'TEST', augment=True)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size = batch_size,
                                             shuffle = True, num_workers = 8)

writer = SummaryWriter()


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
        for i, (img, label) in enumerate(train_dataloaders):
            torch.cuda.synchronize()

            img, label = img.cuda(), label.cuda()
            #print img and label

            heads = net(img)
            # print(regression)
            optimizer.zero_grad()

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

        Error_test = 0
        Error_train = 0
        Error_test_wrist = 0

        if (epoch % 1 == 0):
            net = net.eval()
            output = torch.FloatTensor()
            outputTrain = torch.FloatTensor()

            for i, (img, label) in tqdm(enumerate(test_dataloaders)):
                with torch.no_grad():
                    img, label = img.cuda(), label.cuda()
                    heads = net(img)
                    pred_keypoints = post_precess(heads, voting=False)
                    output = torch.cat([output, pred_keypoints.data.cpu()], 0)

            result = output.cpu().data.numpy()
            Error_test = errorCompute(result, keypointsUVD_test, center_test)
            print('epoch: ', epoch, 'Test error:', Error_test)
            saveNamePrefix = '%s/net_%d_wetD_' % (save_dir, epoch) + str(Weight_Decay) + '_depFact_' + str(
                spatialFactor) + '_RegFact_' + str(RegLossFactor) + '_rndShft_' + str(RandCropShift)
            torch.save(net.state_dict(), saveNamePrefix + '.pth')

        # log
        writer.add_scalar('Loss/train', train_loss_add, epoch)
        writer.add_scalar('Loss/test', Error_test, epoch)

        logging.info('Epoch#%d: total loss=%.4f, Cls_loss=%.4f, Reg_loss=%.4f, Err_test=%.4f, lr = %.6f'
                     % (epoch, train_loss_add, Cls_loss_add, Reg_loss_add, Error_test, scheduler.get_lr()[0]))


def test():
    net = model.A2J_model(num_classes=keypointsNumber)
    net.load_state_dict(torch.load(model_dir))
    net = net.cuda()
    net.eval()
    MODE = 'TEST'

    post_precess = anchor.post_process(shape=[cropWidth// 16, cropHeight // 16], stride=16, P_h=None, P_w=None)

    output = torch.FloatTensor()
    torch.cuda.synchronize()
    for i, (img, label) in tqdm(enumerate(test_dataloaders)):
        with torch.no_grad():
            img, label = img.cuda(), label.cuda()
            heads = net(img)
            pred_keypoints = post_precess(heads, voting=False)
            output = torch.cat([output, pred_keypoints.data.cpu()], 0)

    torch.cuda.synchronize()

    result = output.cpu().data.numpy()
    writeTxt(result, center_test)
    error = errorCompute(result, keypointsUVD_test, center_test)
    print('Error:', error)


def errorCompute(source, target, center):
    assert np.shape(source) == np.shape(target), "source has different shape with target"

    Test1_ = source.copy()
    target_ = target.copy()
    Test1_[:, :, 0] = source[:, :, 1]
    Test1_[:, :, 1] = source[:, :, 0]
    Test1 = Test1_  # [x, y, z]

    center_pixel = center.copy()
    centre_world = pixel2world(center.copy(), fx, fy, u0, v0)

    centerlefttop = centre_world.copy()
    centerlefttop[:, 0, 0] = centerlefttop[:, 0, 0]
    centerlefttop[:, 0, 1] = centerlefttop[:, 0, 1]

    centerrightbottom = centre_world.copy()
    centerrightbottom[:, 0, 0] = centerrightbottom[:, 0, 0]
    centerrightbottom[:, 0, 1] = centerrightbottom[:, 0, 1]

    lefttop_pixel = world2pixel(centerlefttop, fx, fy, u0, v0)
    rightbottom_pixel = world2pixel(centerrightbottom, fx, fy, u0, v0)

    for i in range(len(Test1_)):
        Xmin = max(lefttop_pixel[i, 0, 0], 0)
        Ymin = max(lefttop_pixel[i, 0, 1], 0)
        Xmax = min(rightbottom_pixel[i, 0, 0], 320 * 2 - 1)
        Ymax = min(rightbottom_pixel[i, 0, 1], 240 * 2 - 1)

        Test1[i, :, 0] = Test1_[i, :, 0] * (Xmax - Xmin) / cropWidth + Xmin  # x
        Test1[i, :, 1] = Test1_[i, :, 1] * (Ymax - Ymin) / cropHeight+ Ymin  # y
        Test1[i, :, 2] = source[i, :, 2] + center[i][0][2]

    labels = pixel2world(target_, fx, fy, u0, v0)
    outputs = pixel2world(Test1.copy(), fx, fy, u0, v0)

    errors = np.sqrt(np.sum((labels - outputs) ** 2, axis=2))

    return np.mean(errors)


def writeTxt(result, center):
    resultUVD_ = result.copy()
    resultUVD_[:, :, 0] = result[:, :, 1]
    resultUVD_[:, :, 1] = result[:, :, 0]
    resultUVD = resultUVD_  # [x, y, z]

    center_pixel = center.copy()
    centre_world = pixel2world(center.copy(), fx, fy, u0, v0)

    centerlefttop = centre_world.copy()
    centerlefttop[:, 0, 0] = centerlefttop[:, 0, 0] - xy_thres
    centerlefttop[:, 0, 1] = centerlefttop[:, 0, 1] + xy_thres

    centerrightbottom = centre_world.copy()
    centerrightbottom[:, 0, 0] = centerrightbottom[:, 0, 0] + xy_thres
    centerrightbottom[:, 0, 1] = centerrightbottom[:, 0, 1] - xy_thres

    lefttop_pixel = world2pixel(centerlefttop, fx, fy, u0, v0)
    rightbottom_pixel = world2pixel(centerrightbottom, fx, fy, u0, v0)

    for i in range(len(result)):
        Xmin = max(lefttop_pixel[i, 0, 0], 0)
        Ymin = max(lefttop_pixel[i, 0, 1], 0)
        Xmax = min(rightbottom_pixel[i, 0, 0], 320 * 2 - 1)
        Ymax = min(rightbottom_pixel[i, 0, 1], 240 * 2 - 1)

        resultUVD[i, :, 0] = resultUVD_[i, :, 0] * (Xmax - Xmin) / cropWidth + Xmin  # x
        resultUVD[i, :, 1] = resultUVD_[i, :, 1] * (Ymax - Ymin) / cropHeight+ Ymin  # y
        resultUVD[i, :, 2] = result[i, :, 2] + center[i][0][2]

    resultReshape = resultUVD.reshape(len(result), -1)

    with open(os.path.join(save_dir, result_file), 'w') as f:
        for i in range(len(resultReshape)):
            for j in range(keypointsNumber * 3):
                f.write(str(resultReshape[i, j]) + ' ')
            f.write('\n')

    f.close()


if __name__ == '__main__':
    train()
    test()

