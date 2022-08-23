import argparse
import time
import os
import numpy as np
import cv2
import torch
import pylab as plt
import json
import copy
from collections import OrderedDict
import sys
sys.path.insert(0, '..')
from lib.network.yolo_posenet import YoloPoseNet
from lib.datasets import datasets_itop, data_augmentation_2d3d
from lib.utils.common import *
from lib.utils.prior_pose_align import parse_prior_pose, universe_align_map
from evaluate.eval_pose_mp import eval_human_dataset_2d, eval_human_dataset_3d
anchors = np.array([(6., 3.), (12., 6.)])


"""
A generic pose network with new 3D branch and 2D align branch :
Modification from rtpose:
    1. preprocess trained_model uses residual modules rather than vgg. The resolution reduction is tuned to be 1/8
    2. only uses two stages in the prediction
    3. New 3D branch is developed and integrated
    4. New 2D align branch is developed and integrated
    5. A new pose prior subnetwork is connected

Ablation evaluation includes:
    1. direct read from depth input
    2. depth read from pose depth estimation
    3. read pose depth given perfect 2D location
    4. read pose depth given perfect 2D location, focusing only on visible joints

Author: Yuliang Guo (33yuliangguo@gmail.com)
Date: April 2020
"""


# DATA_DIR = 'D:/Datasets/ITOP'
DATA_DIR = '/media/yuliang/DATA/Datasets/ITOP'
ANNOTATIONS_VAL = os.path.join(DATA_DIR, 'annotations', 'ITOP_side_test_labels.json')
IMAGE_DIR_TRAIN = os.path.join(DATA_DIR, 'ITOP_side_train_depth_map')
IMAGE_DIR_VAL = os.path.join(DATA_DIR, 'ITOP_side_test_depth_map')


def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--val-annotations', default=ANNOTATIONS_VAL)
    parser.add_argument('--val-image-dir', default=IMAGE_DIR_VAL)
    parser.add_argument('--loader-workers', default=0, type=int,
                        help='number of workers for data loading')
    parser.add_argument('--batch-size', default=15, type=int,
                        help='batch size')
    parser.add_argument('--input-size', default=224, type=int,
                        help='input size of image for network')
    parser.add_argument('--w-org', default=320, type=int,
                        help='original width of dataset image')
    parser.add_argument('--h-org', default=240, type=int,
                        help='original height of dataset image')
    parser.add_argument('--ht-reduction', default=8, type=float,
                        help='dimension reduction for the heatmap')
    parser.add_argument('--num-parts', default=15, type=int,
                        help='number of body parts')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--weight', type=str,
                        default='../trained_model/yolo_posenet_itop/best_pose.pth')
    parser.add_argument('--output-dir', type=str,
                        default='../predictions/yolo_posenet_itop/')
    args = parser.parse_args()

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True

    return args


if __name__ == "__main__":
    vis = True
    args = cli()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    joint_names = datasets_itop.get_keypoints()
    limb_ids = datasets_itop.kp_connections(joint_names)
    # joint2chn = datasets_itop_new.get_joint2chn()
    joint2chn = np.array(range(args.num_parts))

    print("Loading dataset...")
    # load train data
    preprocess = data_augmentation_2d3d.Compose([
        data_augmentation_2d3d.Cvt2ndarray(),
        data_augmentation_2d3d.Resize(args.input_size)
    ])

    val_data = datasets_itop.ItopKeypoints(
        img_dir=args.val_image_dir,
        ann_file=args.val_annotations,
        is_train=False,
        preprocess=preprocess,
        input_x=args.input_size,
        input_y=args.input_size
    )

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True)

    # model
    model = YoloPoseNet(args.num_parts, input_dim=1, anchors=anchors)
    state_dict = torch.load(args.weight)
    # modify for pytorch version compatibility
    state_dict = OrderedDict([(key[key.find('.') + 1:], value) for key, value in state_dict.items()])
    model.load_state_dict(state_dict)

    # evaluate on validation set
    model = torch.nn.DataParallel(model).cuda()
    model.float()
    model.eval()

    end = time.time()
    cnt = 0
    print("Processing dataset...")

    human_pred_set_2d = []
    human_pred_set_3d = []
    human_gt_set_2d = []
    human_gt_set_3d = []
    for batch_i, (img, indices) in enumerate(val_loader):
        humans_gt_2d_batch = []
        for id in indices.numpy():
            human_gt_batch = val_data.anno_dic[val_data.ids[id]]
            human_gt_2d = [human['2d_joints'] for human in human_gt_batch]
            humans_gt_2d_batch.append(human_gt_2d)
            human_gt_set_2d.append(human_gt_2d)
            human_gt_3d = [human['3d_joints'] for human in human_gt_batch]
            human_gt_set_3d.append(human_gt_3d)

        img = img.cuda()

        # compute output
        with torch.no_grad():
            prior_posemap = model(img)

        # unormalize images
        img = img.cpu().data.numpy()
        img *= datasets_itop.depth_std
        img += datasets_itop.depth_mean

        # evaluate the and visualize direct prior results
        bboxes, humans_prior, visibilities = parse_prior_pose(prior_posemap,
                                                              model.module.anchors,
                                                              args.num_parts,
                                                              args.input_size,
                                                              args.input_size,
                                                              datasets_itop.depth_mean,
                                                              datasets_itop.depth_std,
                                                              conf_threshold=0.1)

        for b in range(args.batch_size):
            # refine pose detection from prior sunbetwork via local alignment
            # TODO: HARD-CODED for single human
            if len(humans_prior[b]) > 0:
                humans_2d = [humans_prior[b][0][:, :2]]
                humans_depth = [humans_prior[b][0][:, 2]]
            else:
                humans_2d = [np.ones([args.num_parts, 2])*(-1)]
                humans_depth = [np.ones(args.num_parts) * (-1)]

            # convert detected joints to original image scale
            for i, human in enumerate(humans_2d):
                human = np.array(human)
                human[:, 0] = human[:, 0] / args.input_size * args.w_org
                human[:, 1] = human[:, 1] / args.input_size * args.h_org
                humans_2d[i] = human

            # convert prediction to 3D
            humans_3d = []
            humans_3d_raw = []
            humans_3d_aligned = []
            for i, human in enumerate(humans_2d):
                """
                human_3d read from Z map
                """
                human_3d = pos_3d_from_2d_and_depth(human[:, 0],
                                                    human[:, 1],
                                                    humans_depth[i],
                                                    datasets_itop.intrinsics['cx'],
                                                    datasets_itop.intrinsics['cy'],
                                                    datasets_itop.intrinsics['f'],
                                                    datasets_itop.intrinsics['f'])
                human_3d[:, 1] *= (-1)  # ATTENTION: ITOP requires to 3D Y
                humans_3d.append(human_3d.tolist())
                humans_2d[i] = human.tolist()

            # append full set results
            human_pred_set_2d.append(humans_2d)
            human_pred_set_3d.append(humans_3d)

            if vis:
                # visualization
                visImg = cv2.resize(img[b, 0, :, :], (args.w_org, args.h_org))
                visImg[visImg > datasets_itop.depth_max] = datasets_itop.depth_max
                visImg[visImg < 0] = 0
                visImg /= datasets_itop.depth_max
                visImg *= 255
                visImg = visImg.astype(np.uint8)
                visImg = cv2.cvtColor(visImg, cv2.COLOR_GRAY2BGR)
                visImg2 = np.copy(visImg)

                visImg = draw_humans_visibility(visImg,
                                                humans_2d,
                                                limb_ids,
                                                datasets_itop.jointColors)
                cv2.imwrite(os.path.join(args.output_dir, '{:06d}.jpg'.format(cnt)), visImg)

            cnt += 1

        # measure elapsed time
        process_time = (time.time() - end)
        end = time.time()
        print('batch:[{}/{}], process time: {:3f}/image\t'.format(batch_i, len(val_loader), process_time/args.batch_size))

    # human_pred_set_2d_read_raw_depth = human_pred_set_2d
    # human_pred_set_2d_perfect_2d = human_gt_set_2d

    # save data
    eval_data = {'human_pred_set_2d': human_pred_set_2d,
                 'human_pred_set_3d': human_pred_set_3d,
                 'human_gt_set_2d': human_gt_set_2d,
                 'human_gt_set_3d': human_gt_set_3d}
    with open(os.path.join(args.output_dir, 'eval_data.json'), 'w') as json_file:
        json.dump(eval_data, json_file, indent=4)

    # evaluation
    print('\nevaluating in 2D...')
    dist_th_2d = 0.02 * np.sqrt(args.w_org ** 2 + args.h_org ** 2)
    joint_avg_dist, joint_KCP = eval_human_dataset_2d(eval_data['human_pred_set_2d'],
                                                      eval_data['human_gt_set_2d'],
                                                      num_joints=args.num_parts,
                                                      dist_th=dist_th_2d,
                                                      iou_th=0.5)
    joint_names = datasets_itop.get_keypoints()
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
                                                      num_joints=args.num_parts,
                                                      dist_th=0.1,
                                                      iou_th=0.5)
    joint_names = datasets_itop.get_keypoints()
    print('     3D threshold: {:03f} meter'.format(0.1))
    for i, name in enumerate(joint_names):
        print('     joint: {},  PCK: {:03f}, avg 3D error: {:03f}'.format(name,
                                                                          joint_KCP[i],
                                                                          joint_avg_dist[i]))

    print('\n     Overall: PCK: {:03f}, avg 3D error: {:03f} \n'.format(np.average(joint_KCP),
                                                                        np.average(joint_avg_dist)))

