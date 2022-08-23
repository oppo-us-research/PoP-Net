"""
Data Preparation for rt-pose network with 3D align extension on ITOP dataset
"""
import copy
import logging
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
import torch.utils.data
import torchvision
from PIL import Image
from lib.datasets.heatmap import putGaussianMaps
from lib.datasets.paf import putVecMaps
from lib.datasets.posemap import putJointZ, putJointDxDy
from lib.datasets import data_augmentation_2d3d
from lib.utils.common import draw_humans, pos_3d_from_2d_and_depth
plt.rcParams['figure.figsize'] = (6, 3)

# joint_id_to_name = {
#   0: 'Head',        8: 'Torso',
#   1: 'Neck',        9: 'R Hip',
#   2: 'R Shoulder',  10: 'L Hip',
#   3: 'L Shoulder',  11: 'R Knee',
#   4: 'R Elbow',     12: 'L Knee',
#   5: 'L Elbow',     13: 'R Foot',
#   6: 'R Hand',      14: 'L Foot',
#   7: 'L Hand',
# }

intrinsics = {'f': 1 / 0.0035, 'cx': 160, 'cy': 120}

jointColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
               [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
               [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [85, 255, 85]]

root_joint = 'torso'

depth_mean = 3
depth_std = 2
depth_max = 5


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


def get_joint2chn():
    """
    compute the corresponding chn in posedepth map for each joint
    """
    joint_names = get_keypoints()
    limb_ids = kp_connections(joint_names)
    root_id = joint_names.index(root_joint)
    joint2chn = np.zeros(len(joint_names)).astype(np.int)
    joint2chn[root_id] = 0
    for k, limb in enumerate(limb_ids):
        joint2chn[limb[1]] = k+1
    return joint2chn


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


def get_swap_part_indices():
    keypoints = get_keypoints()
    swap_indices = []
    for keypoint in keypoints:
        if keypoint == 'right_shoulder':
            swap_indices.append(keypoints.index('left_shoulder'))
            continue

        if keypoint == 'left_shoulder':
            swap_indices.append(keypoints.index('right_shoulder'))
            continue

        if keypoint == 'right_elbow':
            swap_indices.append(keypoints.index('left_elbow'))
            continue

        if keypoint == 'left_elbow':
            swap_indices.append(keypoints.index('right_elbow'))
            continue

        if keypoint == 'right_wrist':
            swap_indices.append(keypoints.index('left_wrist'))
            continue

        if keypoint == 'left_wrist':
            swap_indices.append(keypoints.index('right_wrist'))
            continue

        if keypoint == 'right_hip':
            swap_indices.append(keypoints.index('left_hip'))
            continue

        if keypoint == 'left_hip':
            swap_indices.append(keypoints.index('right_hip'))
            continue

        if keypoint == 'right_knee':
            swap_indices.append(keypoints.index('left_knee'))
            continue

        if keypoint == 'left_knee':
            swap_indices.append(keypoints.index('right_knee'))
            continue

        if keypoint == 'right_ankle':
            swap_indices.append(keypoints.index('left_ankle'))
            continue

        if keypoint == 'left_ankle':
            swap_indices.append(keypoints.index('right_ankle'))
            continue

        swap_indices.append(keypoints.index(keypoint))

    return swap_indices


class ItopKeypoints(torch.utils.data.Dataset):
    """ITOP Dataset.

    Caches preprocessing.

    Args:
        img_dir (string): Root directory where images are saved.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, img_dir, annFile, preprocess=None, input_x=224, input_y=224, stride=8, PoseAlign=False, z_radius=2, AlignRadius=2):
        self.img_dir = img_dir
        # load all the ids and joint annotations
        self.anno_dic = json.load(open(annFile, 'r'))
        self.ids = [key for key, value in self.anno_dic.items()]
        # if n_images:
        #     self.ids = self.ids[:n_images]
        print('Images: {}'.format(len(self.ids)))

        # preprocess for data augmentation, at original image values
        self.preprocess = preprocess

        # standard totensor and normalize, apply after data augmentation
        self.image_transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean=[depth_mean], std=[depth_std])])

        self.JOINT_NAMES = get_keypoints()
        self.LIMB_IDS = kp_connections(self.JOINT_NAMES)
        self.ROOT_ID = self.JOINT_NAMES.index(root_joint)
        self.HEATMAP_COUNT = len(self.JOINT_NAMES)
        # self.joint2chn = get_joint2chn()
        self.input_x = input_x
        self.input_y = input_y
        self.stride = stride
        self.PoseAlign = PoseAlign
        self.z_radius = z_radius
        self.AlignRadius = AlignRadius
        self.log = logging.getLogger(self.__class__.__name__)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``itop.loadAnns``.
        """
        image_id = self.ids[index]
        anns = self.anno_dic[image_id]
        anns = copy.deepcopy(anns)

        image = np.load(os.path.join(self.img_dir, image_id + '.npy')).astype(np.float)

        # data augmentation
        image, anns = self.preprocess((image, anns))

        # data normalization
        image[image < 0] = 0
        image[image > depth_max] = depth_max
        depth_resize = cv2.resize(image,
                                  (np.int(self.input_x/self.stride), np.int(self.input_y/self.stride)))
        image = self.image_transform(image)

        # ground-truth maps preparation
        heatmaps, pafs, zmaps, fg_masks_z, align_maps, fg_masks_align = self.single_image_processing(anns, depth_resize)

        return image, heatmaps, pafs, zmaps, fg_masks_z, align_maps, fg_masks_align, anns[0]['2d_joints'], index

    def single_image_processing(self, anns, depth_resize):
        """

        :param anns: 2d and 3d annotations
        :param depth_resize: it is a copy of image, but in numpy format for convenience
        :return:
        """
        # prepare target maps
        heatmaps, pafs, zmaps, fg_masks_z, align_maps, fg_masks_align = self.get_ground_truth(anns, depth_resize)
        
        heatmaps = torch.from_numpy(heatmaps.transpose((2, 0, 1)).astype(np.float32))
        pafs = torch.from_numpy(pafs.transpose((2, 0, 1)).astype(np.float32))
        zmaps = torch.from_numpy(zmaps.transpose((2, 0, 1)).astype(np.float32))
        fg_masks_z = torch.from_numpy(fg_masks_z.transpose((2, 0, 1)).astype(np.float32))
        align_maps = torch.from_numpy(align_maps.transpose((2, 0, 1)).astype(np.float32))
        fg_masks_align = torch.from_numpy(fg_masks_align.transpose((2, 0, 1)).astype(np.float32))

        return heatmaps, pafs, zmaps, fg_masks_z, align_maps, fg_masks_align

    def remove_illegal_joint(self, keypoints_2d, visibility=None):

        if visibility is None:
            visibility = np.ones([keypoints_2d.shape[0], keypoints_2d.shape[1]])
        mask = np.logical_or.reduce((keypoints_2d[:, :, 0] >= self.input_x,
                                     keypoints_2d[:, :,  0] < 0,
                                     keypoints_2d[:, :, 1] >= self.input_y,
                                     keypoints_2d[:, :, 1] < 0))
        visibility[mask] = 0
        return visibility

    # This is the key function to prepare a set of target maps
    def get_ground_truth(self, anns, depth_input):
        grid_y = int(self.input_y / self.stride)
        grid_x = int(self.input_x / self.stride)
        channels_heat = (self.HEATMAP_COUNT + 1)
        channels_paf = 2 * len(self.LIMB_IDS)
        heatmaps = np.zeros((int(grid_y), int(grid_x), channels_heat))
        pafs = np.zeros((int(grid_y), int(grid_x), channels_paf))

        # ATTENTION: initialize with max and only work on foreground. At last update background with the union fg mask
        # initialize pose depth with input depth map
        zmaps_org = np.repeat(depth_input[:, :, np.newaxis], len(self.joint_names), axis=2)
        zmaps = np.ones_like(zmaps_org) * 2*depth_max
        fg_masks_z = np.zeros((int(grid_y), int(grid_x), len(self.JOINT_NAMES)))
        align_maps = np.zeros((int(grid_y), int(grid_x), 2*len(self.JOINT_NAMES)))
        fg_masks_align = np.zeros((int(grid_y), int(grid_x), 2*len(self.JOINT_NAMES)))
        max_dist = 2 * (self.align_radius + 0.5)
        dist_maps = np.ones((int(grid_y), int(grid_x), len(self.joint_names))).astype(np.float32) * max_dist

        # This considers multiple annotations per frame later
        keypoints_2d = []
        keypoints_3d = []
        visibilities = []
        for ann in anns:
            keypoints_2d.append(np.array(ann['2d_joints']).reshape(-1, 2))
            keypoints_3d.append(np.array(ann['3d_joints']).reshape(-1, 3))
            visibilities.append(ann['visible_joints'])
        keypoints_2d = np.array(keypoints_2d)
        keypoints_3d = np.array(keypoints_3d)
        visibilities = np.array(visibilities)  # indicate part occlusion
        inbounds = self.remove_illegal_joint(keypoints_2d)  # indicate inbound of image

        """
            Notes:
                Truncation and occlusion should be considered separately
                Occluded part needs to be prepared in regression maps so that they can be predicted in its channel.
                This is ambiguity in multi-person overlapping of the same joint type.
                So it require to label the occlusion caused by same joint type or others.
        """

        # confidence maps for body parts
        for i in range(self.HEATMAP_COUNT):
            joints = keypoints_2d[:, i, :]
            for j, joint in enumerate(joints):
                # consider visible and within-range joint
                if inbounds[j, i] > 0.5:
                    root_center = joint[:2]
                    gaussian_map = heatmaps[:, :, i]
                    heatmaps[:, :, i] = putGaussianMaps(
                        root_center, gaussian_map,
                        7.0, grid_y, grid_x, self.stride)
        # background: the last dimension of the heatmap prepared for background. Default for cross-entropy loss
        heatmaps[:, :, -1] = np.maximum(
            1 - np.max(heatmaps[:, :, :self.HEATMAP_COUNT], axis=2),
            0.
        )

        # pafs for limbs
        for i, (k1, k2) in enumerate(self.LIMB_IDS):
            # limb
            count = np.zeros((int(grid_y), int(grid_x)), dtype=np.uint32)
            for j, joints in enumerate(keypoints_2d):
                if inbounds[j, k1] > 0.5 and inbounds[j, k2] > 0.5:
                    centerA = joints[k1, :2]
                    centerB = joints[k2, :2]
                    vec_map = pafs[:, :, 2 * i:2 * (i + 1)]

                    pafs[:, :, 2 * i:2 * (i + 1)], count = putVecMaps(
                        centerA=centerA,
                        centerB=centerB,
                        accumulate_vec_map=vec_map,
                        count=count, grid_y=grid_y, grid_x=grid_x, stride=self.stride
                    )

        # pose maps
        # Z maps
        for j, joints in enumerate(keypoints_2d):
            for k, joint in enumerate(joints):
                if inbounds[j, k] < 0.5:
                    continue
                joint_center = joint[:2]
                joint_depth = keypoints_3d[j, k, 2]
                map_z = zmaps[:, :, k]
                fg_mask_z = fg_masks_z[:, :, k]
                zmaps[:, :, k], fg_masks_z[:, :, k] = putJointZ(
                    center=joint_center,
                    depth=joint_depth,
                    accumulate_map=map_z,
                    accumulate_mask=fg_mask_z,
                    grid_y=grid_y, grid_x=grid_x, stride=self.stride, max_depth=depth_max, radius=self.z_radius
                )

        # align maps
        for j, joints in enumerate(keypoints_2d):
            for k, joint in enumerate(joints):
                if inbounds[j, k] < 0.5:
                    continue
                joint_center = joint[:2]
                joint_depth = keypoints_3d[j, k, 2]
                fg_mask_align = fg_masks_align[:, :, 2 * k: 2 * (k + 1)]
                mapDxDy = align_maps[:, :, 2 * k: 2 * (k + 1)]

                align_maps[:, :, 2 * k: 2 * (k + 1)],\
                fg_masks_align[:, :, 2 * k: 2 * (k + 1)],\
                dist_maps[:, :, k] = putJointDxDy(
                    center=joint_center,
                    depth=joint_depth,
                    accumulate_map=mapDxDy,
                    accumulate_mask=fg_mask_align,
                    accumulate_dist=dist_maps[:, :, k],
                    z_map=zmaps[:, :, k],
                    grid_y=grid_y, grid_x=grid_x, stride=self.strideA, radius=self.align_radius, max_dist=max_dist
                )

        zmaps[fg_masks_z == 0] = zmaps_org[fg_masks_z == 0]

        # normalize Z map
        zmaps[zmaps < 0] = 0
        zmaps[zmaps > depth_max] = depth_max
        zmaps -= depth_mean
        zmaps /= depth_std

        return heatmaps, pafs, zmaps, fg_masks_z, align_maps, fg_masks_align

    def __len__(self):
        return len(self.ids)


# unit test for dataloader
if __name__ == "__main__":

    vis = True
    num_joints = 15
    num_bones = 14
    PoseAlign = True
    AlignRadius = 2
    # joint2chn = get_joint2chn()
    joint2chn = np.array(range(num_joints))
    DATA_DIR = 'D:/Datasets/ITOP'
    ANNOTATIONS_TRAIN = [os.path.join(DATA_DIR, 'annotations', item) for item in ['ITOP_side_train_labels.json']]
    preprocess = data_augmentation_2d3d.Compose([
        data_augmentation_2d3d.Cvt2ndarray(),
        # data_augmentation_itop_3d.Rotate(),
        # data_augmentation_itop_3d.RenderDepth(),
        # data_augmentation_itop_3d.Hflip(swap_indices=get_swap_part_indices()),
        # data_augmentation_itop_3d.Crop(),  # it may violate the 3D-2D geometry
        data_augmentation_2d3d.Resize(224)
    ])

    train_data = ItopKeypoints(
        img_dir=os.path.join(DATA_DIR, 'ITOP_side_train_depth_map'),
        annFile=ANNOTATIONS_TRAIN[0],
        preprocess=preprocess,
        input_x=224,
        input_y=224,
        PoseAlign=PoseAlign
    )
    # train_data = torch.utils.data.ConcatDataset(train_datas)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=20, shuffle=True,
        pin_memory=True, num_workers=0, drop_last=True)  # set num_wrkers=1 for debugging

    dists_3d_perfect_2d = []
    dists_3d_rounded_2d = []
    dists_3d_aligned_2d = []

    for i, (img, heatmap_target, paf_target, zmap_target, fg_masks_z, alignmap_target, fg_masks_align, anns, index) in enumerate(train_loader):
        # verify 2D pos + Z branch recovers 3D pose,
        # TODO: verification is currently not compatible with augmentation
        zmap_target = zmap_target.cpu().data.numpy().transpose(0, 2, 3, 1)
        zmap_target[:, :, :, :num_joints] *= depth_std
        zmap_target[:, :, :, :num_joints] += depth_mean
        alignmap_target = alignmap_target.cpu().data.numpy().transpose(0, 2, 3, 1)

        humans_gt_2d_batch = []
        humans_gt_3d_batch = []
        humans_gt_vis_batch = []
        for id in index.numpy():
            human_gt_batch = train_data.anno_dic[train_data.ids[id]]
            human_gt_2d = [human['2d_joints'] for human in human_gt_batch]
            humans_gt_2d_batch.append(human_gt_2d)
            human_gt_3d = [human['3d_joints'] for human in human_gt_batch]
            humans_gt_3d_batch.append(human_gt_3d)
            human_gt_vis = [human['visible_joints'] for human in human_gt_batch]
            humans_gt_vis_batch.append(human_gt_vis)

        for j in range(20):
            for ii, human in enumerate(humans_gt_2d_batch[j]):
                human_depth = np.ones(num_joints) * (-1)
                human_rounded = np.zeros([num_joints, 2])
                human_aligned = np.zeros([num_joints, 2])
                for k, joint in enumerate(human):
                    if humans_gt_vis_batch[j][ii][k] < 0.5:
                        continue
                    x_2d_resize = int(float(joint[0]) / 320 * 224 / 8)
                    y_2d_resize = int(float(joint[1]) / 240 * 224 / 8)
                    human_depth[k] = (zmap_target[j, y_2d_resize, x_2d_resize, joint2chn[k]])
                    human_rounded[k, 0] = float(x_2d_resize) * 8 / 224 * 320
                    human_rounded[k, 1] = float(y_2d_resize) * 8 / 224 * 240
                    if PoseAlign:
                        dx = alignmap_target[j, y_2d_resize, x_2d_resize, 2 * joint2chn[k]] * (0.5 + AlignRadius) + 0.5
                        dy = alignmap_target[j, y_2d_resize, x_2d_resize, 2 * joint2chn[k] + 1] * (0.5 + AlignRadius) + 0.5
                        human_aligned[k, 0] = (float(x_2d_resize) + dx) * 8 / 224 * 320
                        human_aligned[k, 1] = (float(y_2d_resize) + dy) * 8 / 224 * 240

                human_3d_target = np.array(humans_gt_3d_batch[j][ii])
                # evaluate using perfect 2D
                # TODO: where the 3D error is from by using float 2D in computation?
                #  float truncation in 2D GT when projected from 3D?
                human_3d_perfect_2d = pos_3d_from_2d_and_depth(np.array(human, dtype=np.float)[:, 0],
                                                               np.array(human, dtype=np.float)[:, 1],
                                                               human_depth,
                                                               intrinsics['cx'],
                                                               intrinsics['cy'],
                                                               intrinsics['f'],
                                                               intrinsics['f'])
                human_3d_perfect_2d[:, 1] *= (-1)  # ATTENTION: ITOP requires to 3D Y
                dist_3d_perfect_2d = np.sqrt(np.sum((human_3d_perfect_2d - human_3d_target)**2, axis=1))
                dist_3d_perfect_2d[np.array(humans_gt_vis_batch[j][ii]) < 0.5] = -1
                dists_3d_perfect_2d.append(dist_3d_perfect_2d)

                # evaluate using rounded 2D
                human_3d_round_2d = pos_3d_from_2d_and_depth(np.array(human_rounded, dtype=np.float)[:, 0],
                                                             np.array(human_rounded, dtype=np.float)[:, 1],
                                                             human_depth,
                                                             intrinsics['cx'],
                                                             intrinsics['cy'],
                                                             intrinsics['f'],
                                                             intrinsics['f'])
                human_3d_round_2d[:, 1] *= (-1)  # ATTENTION: ITOP requires to 3D Y
                dist_3d_rounded_2d = np.sqrt(np.sum((human_3d_round_2d - human_3d_target)**2, axis=1))
                dist_3d_rounded_2d[np.array(humans_gt_vis_batch[j][ii]) < 0.5] = -1
                dists_3d_rounded_2d.append(dist_3d_rounded_2d)

                # evaluate using aligned 2D
                if PoseAlign:
                    human_3d_aligned_2d = pos_3d_from_2d_and_depth(np.array(human_aligned, dtype=np.float)[:, 0],
                                                                   np.array(human_aligned, dtype=np.float)[:, 1],
                                                                   human_depth,
                                                                   intrinsics['cx'],
                                                                   intrinsics['cy'],
                                                                   intrinsics['f'],
                                                                   intrinsics['f'])
                    human_3d_aligned_2d[:, 1] *= (-1)  # ATTENTION: ITOP requires to 3D Y
                    dist_3d_aligned_2d = np.sqrt(np.sum((human_3d_aligned_2d - human_3d_target) ** 2, axis=1))
                    dist_3d_aligned_2d[np.array(humans_gt_vis_batch[j][ii]) < 0.5] = -1
                    dists_3d_aligned_2d.append(dist_3d_aligned_2d)

        if vis is True:
            for j in range(20):
                single_img = img[j, 0, :, :].numpy()
                single_heatmap = heatmap_target[j, ...].numpy()
                single_paf = paf_target[j, ...].numpy()
                single_zmap = zmap_target[j, ...].transpose(2, 0, 1)
                single_fg_mask_z = fg_masks_z[j, ...].numpy()
                single_alignmap = alignmap_target[j, ...].transpose(2, 0, 1)
                single_fg_mask_align = fg_masks_align[j, ...].numpy()

                single_anno = [anns[j, ...].numpy()]  # ATTENTION: this anno goes through data augmentation

                # normalize depth image for visualization
                single_img *= 2
                single_img += 3
                single_img[single_img <= 0] = 0
                single_img[single_img >= 5] = 5
                single_img /= 5

                fig1 = plt.figure()
                ax1 = fig1.add_subplot(241)
                ax1.imshow(single_img)
                ax1.set_title('depth input')

                # visualize heatmap
                ht_max = np.max(single_heatmap[:-1, :, :], axis=0)
                ax2 = fig1.add_subplot(242)
                ax2.imshow(ht_max)
                ax2.set_title('ht max')

                # visualize paf map
                paf_x = single_paf[0:2*num_bones:2, ...]
                paf_x = np.sum(paf_x, axis=0)
                paf_y = single_paf[1:2*num_bones:2, ...]
                paf_y = np.sum(paf_y, axis=0)
                ax3 = fig1.add_subplot(243)
                ax3.imshow(paf_x)
                ax3.set_title('paf x')
                ax4 = fig1.add_subplot(247)
                ax4.imshow(paf_y)
                ax4.set_title('paf y')

                # visualize zmap
                # ATTENTION: simply visualizing min may keep the noisy part.
                zmap_vis = np.min(single_zmap, axis=0)
                zmap_vis[zmap_vis <= 0] = 0
                zmap_vis[zmap_vis >= depth_max] = depth_max
                zmap_vis /= depth_max
                ax5 = fig1.add_subplot(245)
                ax5.imshow(zmap_vis)
                ax5.set_title('poseD head')

                # draw humans on the image
                single_img *= 255
                single_img = cv2.cvtColor(single_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                single_img = draw_humans(single_img,
                                         single_anno,
                                         kp_connections(get_keypoints()),
                                         jointColors)
                single_img = np.flip(single_img, 2)
                ax6 = fig1.add_subplot(246)
                ax6.imshow(single_img)
                ax6.set_title('skeleton')

                if PoseAlign:
                    # visualize dx dy field maps, using sum for simplicity, but not correct for occlusion
                    dx_field = single_alignmap[0: 2 * num_joints:2, ...]
                    dx_field = np.sum(dx_field, axis=0)
                    dy_field = single_alignmap[1: 2 * num_joints:2, ...]
                    dy_field = np.sum(dy_field, axis=0)
                    ax7 = fig1.add_subplot(244)
                    ax7.imshow(dx_field)
                    ax7.set_title('dx field')
                    ax8 = fig1.add_subplot(248)
                    ax8.imshow(dy_field)
                    ax8.set_title('dy field')

                fig1.subplots_adjust(wspace=0.1, hspace=0.01)
                fig1.show()
                plt.waitforbuttonpress()
                plt.pause(1)
                plt.close(fig1)

        print('iter: [{}/{}]\t'.format(i, len(train_loader)))

    ##################################################################################################
    print('\nGround-truth preparation error using perfect 2D joint:')
    dists_3d_perfect_2d = np.array(dists_3d_perfect_2d)
    joint_avg_dist = []
    for k in range(num_joints):
        single_joint_dists = dists_3d_perfect_2d[:, k]
        joint_avg_dist.append(np.average(single_joint_dists[np.where(single_joint_dists >= 0)]))
    for i, name in enumerate(get_keypoints()):
        print('     joint: {},  avg 3D error: {:03f}'.format(name, joint_avg_dist[i]))

    print('\n     Overall: avg 3D error: {:03f} \n'.format(np.average(np.average(joint_avg_dist))))

    ##################################################################################################
    print('\nGround-truth preparation error using rounded 2D joint:')
    dists_3d_rounded_2d = np.array(dists_3d_rounded_2d)
    joint_avg_dist = []
    for k in range(num_joints):
        single_joint_dists = dists_3d_rounded_2d[:, k]
        joint_avg_dist.append(np.average(single_joint_dists[np.where(single_joint_dists >= 0)]))
    for i, name in enumerate(get_keypoints()):
        print('     joint: {},  avg 3D error: {:03f}'.format(name, joint_avg_dist[i]))

    print('\n     Overall: avg 3D error: {:03f} \n'.format(np.average(np.average(joint_avg_dist))))

    if PoseAlign:
        ##################################################################################################
        print('\nGround-truth preparation error using aligned 2D joint:')
        dists_3d_aligned_2d = np.array(dists_3d_aligned_2d)
        joint_avg_dist = []
        for k in range(num_joints):
            single_joint_dists = dists_3d_aligned_2d[:, k]
            joint_avg_dist.append(np.average(single_joint_dists[np.where(single_joint_dists >= 0)]))
        for i, name in enumerate(get_keypoints()):
            print('     joint: {},  avg 3D error: {:03f}'.format(name, joint_avg_dist[i]))

        print('\n     Overall: avg 3D error: {:03f} \n'.format(np.average(np.average(joint_avg_dist))))

