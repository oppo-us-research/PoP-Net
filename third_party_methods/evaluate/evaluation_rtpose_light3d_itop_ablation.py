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

from lib.network.rtpose_light3d import rtpose_light3d
from lib.datasets import datasets_itop_rtpose, data_augmentation_2d3d
from lib.utils.common import *
from lib.config import cfg
from lib.utils.paf_to_pose import paf_to_pose
from evaluate.eval_pose_mp import eval_human_dataset_2d, eval_human_dataset_3d


"""
A light version of openpose or rtpose network with 3D branch:
Modification from rtpose:
    1. preprocess trained_model uses residual modules rather than vgg. The resolution reduction is tuned to be 1/8
    2. only uses two stages in the prediction
    3. 3D branch is developed and integrated


Ablation evaluation includes:
    1. direct read from depth input
    2. depth read from pose depth estimation
    3. read pose depth given perfect 2D location
    4. read pose depth given perfect 2D location, focusing only on visible joints

Author: Yuliang Guo (33yuliangguo@gmail.com)
Date: April 2020
"""

DATA_DIR = '/media/yuliang/DATA/Datasets/ITOP'
# DATA_DIR = 'D:/Datasets/ITOP'
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
    parser.add_argument('--num-parts', default=15, type=int,
                        help='number of body parts')
    parser.add_argument('--num-limbs', default=14, type=int,
                        help='number of body parts')
    parser.add_argument('--num-stages', default=2, type=int,
                        help='number of stages in prediction trained_model')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--weight', type=str,
                        default='./trained_model/rtpose_light3d_itop/best_pose.pth')
    parser.add_argument('--output-dir', type=str,
                        default='./predictions/rtpose_light3d_itop/')
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
    vis_feat = False
    args = cli()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    joint_names = datasets_itop_rtpose.get_keypoints()
    limb_ids = datasets_itop_rtpose.kp_connections(joint_names)
    # joint2chn = datasets_itop_rtpose.get_joint2chn()
    joint2chn = np.array(range(args.num_parts))

    # TODO: replace cfg with args
    cfg.defrost()
    # cfg.merge_from_file(args.cfg)
    # cfg.merge_from_list(args.opts)
    cfg.MODEL.DOWNSAMPLE = 8
    cfg.MODEL.NUM_KEYPOINTS = args.num_parts
    cfg.MODEL.NUM_LIMBS = args.num_limbs
    cfg.MODEL.NUM_STAGES = args.num_stages
    cfg.MODEL.IMAGE_SIZE = [args.input_size, args.input_size]
    cfg.freeze()

    print("Loading dataset...")
    # load train data
    preprocess = data_augmentation_2d3d.Compose([
        data_augmentation_2d3d.Cvt2ndarray(),
        data_augmentation_2d3d.Resize(args.input_size)
    ])

    val_data = datasets_itop_rtpose.ItopKeypoints(
        img_dir=args.val_image_dir,
        annFile=args.val_annotations,
        preprocess=preprocess,
        # image_transform=transforms_itop.image_transform_train,
        input_x=args.input_size,
        input_y=args.input_size
    )

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True)

    # model
    model = rtpose_light3d(args.num_parts, args.num_limbs, args.num_stages, input_dim=1)
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
    human_pred_set_3d_read_raw_depth = []
    human_pred_set_3d_perfect_2d = []
    human_pred_set_3d_perfect_2d_read_raw_depth = []
    human_gt_set_2d_visible = []

    visibility_pred_set = []
    human_gt_set_2d = []
    human_gt_set_3d = []
    human_gt_set_visibility = []
    for batch_i, (img, _, _, _, _, _, _, _, indices) in enumerate(val_loader):
        humans_gt_2d_batch = []
        humans_gt_visibility_batch = []
        for id in indices.numpy():
            human_gt_batch = val_data.anno_dic[val_data.ids[id]]
            human_gt_2d = [human['2d_joints'] for human in human_gt_batch]
            humans_gt_2d_batch.append(human_gt_2d)
            human_gt_set_2d.append(human_gt_2d)
            human_gt_3d = [human['3d_joints'] for human in human_gt_batch]
            human_gt_set_3d.append(human_gt_3d)
            human_gt_visibility = [human['visible_joints'] for human in human_gt_batch]
            humans_gt_visibility_batch.append(human_gt_visibility)
            human_gt_set_visibility.append(human_gt_visibility)

        img = img.cuda()
        # compute output
        with torch.no_grad():
            predicted_outputs, _ = model(img)
        output1, output2, output3 = predicted_outputs[-3], predicted_outputs[-2], predicted_outputs[-1]
        heatmap = output2.cpu().data.numpy().transpose(0, 2, 3, 1)
        paf = output1.cpu().data.numpy().transpose(0, 2, 3, 1)
        posedepth = output3.cpu().data.numpy().transpose(0, 2, 3, 1)
        posedepth *= datasets_itop_rtpose.depth_std
        posedepth += datasets_itop_rtpose.depth_mean

        # unormalize images
        img = img.cpu().data.numpy()
        img *= datasets_itop_rtpose.depth_std
        img += datasets_itop_rtpose.depth_mean

        for b in range(args.batch_size):
            # ensemble human bodies using paf
            humans = paf_to_pose(heatmap[b], paf[b], cfg)

            # convert paf results to a list of humans
            humans_2d, visibility = paf_to_human_list(humans[0], humans[1])

            # TODO: HARD-CODED for debugging
            if len(humans_2d) > 0:
                humans_2d = [humans_2d[0]]
                visibility = [visibility[0]]

            # DO NOT use upsampled posedepth map to check
            # posedepth_upsamp = cv2.resize(
            #     posedepth[b], None, fx=cfg.MODEL.DOWNSAMPLE, fy=cfg.MODEL.DOWNSAMPLE, interpolation=cv2.INTER_CUBIC)

            # read depth prediction from posedepth map network output
            humans_depth = []
            humans_depth_raw = []
            for i, human in enumerate(humans_2d):
                human_depth = np.ones(len(joint_names)) * -1
                human_depth_raw = np.ones(len(joint_names)) * -1
                # human_depth_perfect_2d = np.ones(len(joint_names)) * -1
                for j, joint in enumerate(human):
                    if visibility[i][j] > 0.5:
                        # human_depth[j] = posedepth_upsamp[int(joint[1]), int(joint[0]), joint2chn[j]]
                        #  ATTENTION: why using 'round' leads to worse result than 'int'?
                        #  because x \in (0, 1) should belong to index 0
                        # human_depth[j] = posedepth[b][int(joint[1] / cfg.MODEL.DOWNSAMPLE),
                        #                               int(joint[0] / cfg.MODEL.DOWNSAMPLE),
                        #                               joint2chn[j]]
                        human_depth[j] = retrieve_depth_heat_weighted(
                            [int(joint[0] / cfg.MODEL.DOWNSAMPLE), int(joint[1] / cfg.MODEL.DOWNSAMPLE)],
                            posedepth[b, :, :, joint2chn[j]],
                            heatmap[b, :, :, joint2chn[j]], radius=1)
                        human_depth_raw[j] = img[b][0][int(joint[1]), int(joint[0])]
                humans_depth.append(human_depth)
                humans_depth_raw.append(human_depth_raw)

            humans_depth_perfect_2d = []
            humans_depth_perfect_2d_raw = []
            for i, human in enumerate(humans_gt_2d_batch[b]):
                human_depth_perfect_2d = np.ones(len(joint_names)) * -1
                human_depth_perfect_2d_raw = np.ones(len(joint_names)) * -1
                for j, joint in enumerate(human):
                    # if visibility[i][j] > 0.5:
                    x2d = int(
                        humans_gt_2d_batch[b][i][j][0] / args.w_org * args.input_size / cfg.MODEL.DOWNSAMPLE)
                    y2d = int(
                        humans_gt_2d_batch[b][i][j][1] / args.h_org * args.input_size / cfg.MODEL.DOWNSAMPLE)
                    x2d = min(max(x2d, 0), int(args.input_size / cfg.MODEL.DOWNSAMPLE) - 1)
                    y2d = min(max(y2d, 0), int(args.input_size / cfg.MODEL.DOWNSAMPLE) - 1)
                    human_depth_perfect_2d[j] = posedepth[b][y2d,
                                                             x2d,
                                                             joint2chn[j]]

                    # depth read from perfect 2d and raw depth
                    x2d = min(max(int(joint[0] / args.w_org * args.input_size), 0), int(args.input_size)-1)
                    y2d = min(max(int(joint[1] / args.h_org * args.input_size), 0), int(args.input_size)-1)
                    human_depth_perfect_2d_raw[j] = img[b][0][y2d, x2d]
                humans_depth_perfect_2d.append(human_depth_perfect_2d)
                humans_depth_perfect_2d_raw.append(human_depth_perfect_2d_raw)

            # convert detected joints to original image scale
            for i, human in enumerate(humans_2d):
                human = np.array(human)
                human[np.where(visibility[i]), 0] = human[np.where(visibility[i]), 0] / args.input_size * args.w_org
                human[np.where(visibility[i]), 1] = human[np.where(visibility[i]), 1] / args.input_size * args.h_org
                humans_2d[i] = human

            # convert prediction to 3D
            humans_3d = []
            humans_3d_raw = []
            for i, human in enumerate(humans_2d):
                """
                human_3d read from pose depth
                """
                human_3d_x = (human[:, 0] - datasets_itop_rtpose.intrinsics['cx']) * humans_depth[i] / datasets_itop_rtpose.intrinsics[
                    'f']
                #  ATTENTION: ITOP flips 3D Y
                human_3d_y = -(human[:, 1] - datasets_itop_rtpose.intrinsics['cy']) * humans_depth[i] / datasets_itop_rtpose.intrinsics[
                    'f']
                human_3d = np.vstack([human_3d_x, human_3d_y, humans_depth[i]]).T
                humans_3d.append(human_3d.tolist())

                """
                human_3d read from raw depth
                """
                human_3d_x = (human[:, 0] - datasets_itop_rtpose.intrinsics['cx']) * humans_depth_raw[i] / datasets_itop_rtpose.intrinsics[
                    'f']
                #  ATTENTION: ITOP flips 3D Y
                human_3d_y = -(human[:, 1] - datasets_itop_rtpose.intrinsics['cy']) * humans_depth_raw[i] / datasets_itop_rtpose.intrinsics[
                    'f']
                human_3d_raw = np.vstack([human_3d_x, human_3d_y, humans_depth_raw[i]]).T
                humans_3d_raw.append(human_3d_raw.tolist())
                humans_2d[i] = human.tolist()

            humans_gt_2d_new = []
            humans_3d_perfect_2d = []
            humans_3d_perfect_2d_raw = []
            for i, human_gt in enumerate(humans_gt_2d_batch[b]):
                """
                human_3d given perfect 2d, read from pose depth
                """
                human_gt = np.array(human_gt)
                human_3d_x = (human_gt[:, 0] - datasets_itop_rtpose.intrinsics['cx']) * humans_depth_perfect_2d[i] / datasets_itop_rtpose.intrinsics[
                    'f']
                #  ATTENTION: ITOP flips 3D Y
                human_3d_y = -(human_gt[:, 1] - datasets_itop_rtpose.intrinsics['cy']) * humans_depth_perfect_2d[i] / datasets_itop_rtpose.intrinsics[
                    'f']
                human_3d_perfect_2d = np.vstack([human_3d_x, human_3d_y, humans_depth_perfect_2d[i]]).T
                humans_3d_perfect_2d.append(human_3d_perfect_2d.tolist())

                """
                human_3d given perfect 2d, read from raw depth
                """
                human_3d_x = (human_gt[:, 0] - datasets_itop_rtpose.intrinsics['cx']) * humans_depth_perfect_2d_raw[i] / datasets_itop_rtpose.intrinsics[
                    'f']
                #  ATTENTION: ITOP flips 3D Y
                human_3d_y = -(human_gt[:, 1] - datasets_itop_rtpose.intrinsics['cy']) * humans_depth_perfect_2d_raw[i] / datasets_itop_rtpose.intrinsics[
                    'f']
                human_3d_perfect_2d_raw = np.vstack([human_3d_x, human_3d_y, humans_depth_perfect_2d_raw[i]]).T
                humans_3d_perfect_2d_raw.append(human_3d_perfect_2d_raw.tolist())

                """
                update humens_gt with invisible parts --> enable to calculate avg error only on visible parts
                """
                human_gt_new = copy.deepcopy(human_gt)
                human_gt_new[np.array(humans_gt_visibility_batch[b][i]) == 0, 0] = -1
                human_gt_new[np.array(humans_gt_visibility_batch[b][i]) == 0, 1] = -1
                humans_gt_2d_new.append(human_gt_new.tolist())

            # append full set results
            human_pred_set_2d.append(humans_2d)
            human_pred_set_3d.append(humans_3d)
            human_pred_set_3d_read_raw_depth.append(humans_3d_raw)
            human_pred_set_3d_perfect_2d.append(humans_3d_perfect_2d)
            human_pred_set_3d_perfect_2d_read_raw_depth.append(humans_3d_perfect_2d_raw)
            human_gt_set_2d_visible.append(humans_gt_2d_new)

            visibility_pred_set.append(visibility)

            if vis:
                # visualization
                visImg = cv2.resize(img[b, 0, :, :], (args.w_org, args.h_org))
                visImg[visImg > datasets_itop_rtpose.depth_max] = datasets_itop_rtpose.depth_max
                visImg[visImg < 0] = 0
                visImg /= datasets_itop_rtpose.depth_max
                visImg *= 255
                visImg = visImg.astype(np.uint8)
                visImg = cv2.cvtColor(visImg, cv2.COLOR_GRAY2BGR)

                visImg = draw_humans(visImg,
                                     humans_2d,
                                     limb_ids,
                                     datasets_itop_rtpose.jointColors,
                                     visibility)
                cv2.imwrite(os.path.join(args.output_dir, '{:06d}.jpg'.format(cnt)), visImg)

                if vis_feat:
                    fig1 = plt.figure()

                    ax1 = fig1.add_subplot(231)
                    ax1.imshow(np.flip(visImg, 2))

                    # visualize heatmap
                    ht_max = np.max(heatmap[b][:, :, :-1], axis=2)
                    ax2 = fig1.add_subplot(232)
                    ax2.imshow(ht_max)
                    ax2.set_title('ht max')

                    # visualize paf map
                    paf_x = paf[b][:, :, 0:2 * cfg.MODEL.NUM_LIMBS:2]
                    paf_x = np.sum(paf_x, axis=2)
                    paf_y = paf[b][:, :, 1:2 * cfg.MODEL.NUM_LIMBS:2]
                    paf_y = np.sum(paf_y, axis=2)
                    ax3 = fig1.add_subplot(235)
                    ax3.imshow(paf_x)
                    ax3.set_title('paf x')
                    ax4 = fig1.add_subplot(236)
                    ax4.imshow(paf_y)
                    ax4.set_title('paf y')

                    # visualize min posedepth (not accurately representing each channel)
                    pose_depth_min = np.min(posedepth[b], axis=2)
                    ax5 = fig1.add_subplot(233)
                    ax5.imshow(pose_depth_min)
                    ax5.set_title('PDepth min')

                    # draw skeletons on posedepth
                    pose_depth_min = cv2.resize(pose_depth_min, (args.w_org, args.h_org), interpolation=cv2.INTER_CUBIC)
                    pose_depth_min[pose_depth_min > datasets_itop_rtpose.depth_max] = datasets_itop_rtpose.depth_max
                    pose_depth_min[pose_depth_min < 0] = 0
                    pose_depth_min /= datasets_itop_rtpose.depth_max
                    pose_depth_min *= 255
                    pose_depth_min = pose_depth_min.astype(np.uint8)
                    pose_depth_min = cv2.cvtColor(pose_depth_min, cv2.COLOR_GRAY2BGR)

                    pose_depth_min = draw_humans(pose_depth_min,
                                                 humans_2d,
                                                 limb_ids,
                                                 datasets_itop_rtpose.jointColors,
                                                 visibility)
                    ax5 = fig1.add_subplot(234)
                    ax5.imshow(np.flip(pose_depth_min, 2))
                    ax5.set_title('PDepth upsampl skel')

                    fig1.show()
                    plt.waitforbuttonpress()
                    plt.pause(2)
                    plt.close(fig1)
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
                 'human_pred_set_3d_read_raw_depth': human_pred_set_3d_read_raw_depth,
                 'human_pred_set_3d_perfect_2d': human_pred_set_3d_perfect_2d,
                 'human_pred_set_3d_perfect_2d_read_raw_depth': human_pred_set_3d_perfect_2d_read_raw_depth,
                 'visibility_pred_set': visibility_pred_set,
                 'human_gt_set_2d': human_gt_set_2d,
                 'human_gt_set_2d_visible': human_gt_set_2d_visible,
                 'human_gt_set_3d': human_gt_set_3d,
                 'human_gt_set_visibility': human_gt_set_visibility}
    with open(os.path.join(args.output_dir, 'eval_data.json'), 'w') as json_file:
        json.dump(eval_data, json_file, indent=4)

    # evaluation
    print('\nevaluating in 2D...')
    dist_th_2d = 0.02 * np.sqrt(args.w_org ** 2 + args.h_org ** 2)
    joint_avg_dist, joint_KCP = eval_human_dataset_2d(eval_data['human_pred_set_2d'],
                                                      eval_data['human_gt_set_2d'],
                                                      num_joints=15,
                                                      dist_th=dist_th_2d,
                                                      iou_th=0.5)
    joint_names = datasets_itop_rtpose.get_keypoints()
    print('     2D threshold: {:03f}'.format(dist_th_2d))
    for i, name in enumerate(joint_names):
        print('     joint: {},  PCK: {:03f}, avg 2D error: {:03f}'.format(name,
                                                                          joint_KCP[i],
                                                                          joint_avg_dist[i]))

    print('\n     Overall: PCK: {:03f}, avg 2D error: {:03f} \n'.format(np.average(joint_KCP),
                                                                         np.average(joint_avg_dist)))

    # # save json result
    # json_out_file = os.path.join(args.output_dir, 'eval_res2d.json')
    # json_eval_res = {'dist_th_2d': dist_th_2d,
    #                  'joint_names': joint_names,
    #                  'joint_avg_2d_error': joint_avg_dist,
    #                  'joint_KCP': joint_KCP,
    #                  'overall_avg_2d_error': np.average(joint_avg_dist),
    #                  'overall_KCP': np.average(joint_KCP)}
    # with open(json_out_file, 'w') as json_file:
    #     json.dump(json_eval_res, json_file, indent=4)

    #####################################################################################
    print('\nevaluating in 3D read from pose depth')
    joint_avg_dist, joint_KCP = eval_human_dataset_3d(eval_data['human_pred_set_2d'],
                                                      eval_data['human_gt_set_2d'],
                                                      eval_data['human_pred_set_3d'],
                                                      eval_data['human_gt_set_3d'],
                                                      num_joints=15,
                                                      dist_th=0.1,
                                                      iou_th=0.5)
    joint_names = datasets_itop_rtpose.get_keypoints()
    print('     3D threshold: {:03f} meter'.format(0.1))
    for i, name in enumerate(joint_names):
        print('     joint: {},  PCK: {:03f}, avg 3D error: {:03f}'.format(name,
                                                                          joint_KCP[i],
                                                                          joint_avg_dist[i]))

    print('\n     Overall: PCK: {:03f}, avg 3D error: {:03f} \n'.format(np.average(joint_KCP),
                                                                        np.average(joint_avg_dist)))

    #####################################################################################
    print('\nevaluating in 3D read from pose depth given perfect 2d')
    joint_avg_dist, joint_KCP = eval_human_dataset_3d(eval_data['human_gt_set_2d'],
                                                      eval_data['human_gt_set_2d'],
                                                      eval_data['human_pred_set_3d_perfect_2d'],
                                                      eval_data['human_gt_set_3d'],
                                                      num_joints=15,
                                                      dist_th=0.1,
                                                      iou_th=0.5)
    joint_names = datasets_itop_rtpose.get_keypoints()
    print('     3D threshold: {:03f} meter'.format(0.1))
    for i, name in enumerate(joint_names):
        print('     joint: {},  PCK: {:03f}, avg 3D error: {:03f}'.format(name,
                                                                          joint_KCP[i],
                                                                          joint_avg_dist[i]))

    print('\n     Overall: PCK: {:03f}, avg 3D error: {:03f} \n'.format(np.average(joint_KCP),
                                                                        np.average(joint_avg_dist)))

    #####################################################################################
    print('\nevaluating in 3D read from pose depth given perfect 2d, focusing on visible gt parts')
    joint_avg_dist, joint_KCP = eval_human_dataset_3d(eval_data['human_gt_set_2d_visible'],
                                                      eval_data['human_gt_set_2d'],
                                                      eval_data['human_pred_set_3d_perfect_2d'],
                                                      eval_data['human_gt_set_3d'],
                                                      num_joints=15,
                                                      dist_th=0.1,
                                                      iou_th=0.5)
    joint_names = datasets_itop_rtpose.get_keypoints()
    print('     3D threshold: {:03f} meter'.format(0.1))
    for i, name in enumerate(joint_names):
        print('     joint: {},  PCK: {:03f}, avg 3D error: {:03f}'.format(name,
                                                                          joint_KCP[i],
                                                                          joint_avg_dist[i]))

    print('\n     Overall: PCK: {:03f}, avg 3D error: {:03f} \n'.format(np.average(joint_KCP),
                                                                        np.average(joint_avg_dist)))

    #####################################################################################
    print('\nevaluating in 3D read from raw depth')
    joint_avg_dist, joint_KCP = eval_human_dataset_3d(eval_data['human_pred_set_2d'],
                                                      eval_data['human_gt_set_2d'],
                                                      eval_data['human_pred_set_3d_read_raw_depth'],
                                                      eval_data['human_gt_set_3d'],
                                                      num_joints=15,
                                                      dist_th=0.1,
                                                      iou_th=0.5)
    joint_names = datasets_itop_rtpose.get_keypoints()
    print('     3D threshold: {:03f} meter'.format(0.1))
    for i, name in enumerate(joint_names):
        print('     joint: {},  PCK: {:03f}, avg 3D error: {:03f}'.format(name,
                                                                          joint_KCP[i],
                                                                          joint_avg_dist[i]))

    print('\n     Overall: PCK: {:03f}, avg 3D error: {:03f} \n'.format(np.average(joint_KCP),
                                                                        np.average(joint_avg_dist)))

    #####################################################################################
    print('\nevaluating in 3D read from raw depth given perfect 2d')
    joint_avg_dist, joint_KCP = eval_human_dataset_3d(eval_data['human_gt_set_2d'],
                                                      eval_data['human_gt_set_2d'],
                                                      eval_data['human_pred_set_3d_perfect_2d_read_raw_depth'],
                                                      eval_data['human_gt_set_3d'],
                                                      num_joints=15,
                                                      dist_th=0.1,
                                                      iou_th=0.5)
    joint_names = datasets_itop_rtpose.get_keypoints()
    print('     3D threshold: {:03f} meter'.format(0.1))
    for i, name in enumerate(joint_names):
        print('     joint: {},  PCK: {:03f}, avg 3D error: {:03f}'.format(name,
                                                                          joint_KCP[i],
                                                                          joint_avg_dist[i]))

    print('\n     Overall: PCK: {:03f}, avg 3D error: {:03f} \n'.format(np.average(joint_KCP),
                                                                        np.average(joint_avg_dist)))

    #####################################################################################
    print('\nevaluating in 3D read from raw depth given perfect 2d, focusing on visible gt parts')
    joint_avg_dist, joint_KCP = eval_human_dataset_3d(eval_data['human_gt_set_2d_visible'],
                                                      eval_data['human_gt_set_2d'],
                                                      eval_data['human_pred_set_3d_perfect_2d_read_raw_depth'],
                                                      eval_data['human_gt_set_3d'],
                                                      num_joints=15,
                                                      dist_th=0.1,
                                                      iou_th=0.5)
    joint_names = datasets_itop_rtpose.get_keypoints()
    print('     3D threshold: {:03f} meter'.format(0.1))
    for i, name in enumerate(joint_names):
        print('     joint: {},  PCK: {:03f}, avg 3D error: {:03f}'.format(name,
                                                                          joint_KCP[i],
                                                                          joint_avg_dist[i]))

    print('\n     Overall: PCK: {:03f}, avg 3D error: {:03f} \n'.format(np.average(joint_KCP),
                                                                        np.average(joint_avg_dist)))

    # # save json result
    # json_out_file = os.path.join(args.output_dir, 'eval_res3d.json')
    # json_eval_res = {'dist_th_3d': 0.1,
    #                  'joint_names': joint_names,
    #                  'joint_avg_3d_error': joint_avg_dist,
    #                  'joint_KCP': joint_KCP,
    #                  'overall_avg_3d_error': np.average(joint_avg_dist),
    #                  'overall_KCP': np.average(joint_KCP)}
    # with open(json_out_file, 'w') as json_file:
    #     json.dump(json_eval_res, json_file, indent=4)
