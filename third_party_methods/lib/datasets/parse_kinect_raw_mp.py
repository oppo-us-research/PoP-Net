
'''
    Given calibration, construct human body dataset at syncronized timestamps
    output depth maps in a single file (reno sensors),
    human skeleton 3D joints in a single file (in reno 3D coordinate).
    human skeleton 2D joints in a single file (in reno image coordinates).
'''

import numpy as np
import cv2 as cv
import json
import shutil
import argparse
import copy
import os
import sys
sys.path.insert(0, '../../')
from lib.utils.common import *


"""
Predefined parameters for our application
human joint names
the bones list connecting joints
predefined joint sizes serving to compute 2D bounding box from projecting 3D bounding boxes of each joints
Known resolutions for both reno phone and kinect
truncation range of depth sensor inputs which are used for visualization
"""

WIDTH_1 = 320
HEIGHT_1 = 240
WIDTH_2 = 640
HEIGHT_2 = 576
DEPTH_MAX = 5000  # in mm
DEPTH_MIN = 0  # in mm
offset_x = 0
offset_y = 0
offset_z = 0


# old version
crop_x = 100
crop_y = 32
img_width = 480
img_height = 512


#new version
# crop_x = 91
# crop_y = 128
# img_width = 480
# img_height = 448


#  PELVIS = 0,  SPINE_NAVAL = 1,  SPINE_CHEST = 2, NECK = 3, CLAVICLE_LEFT = 4, SHOULDER_LEFT = 5,  ELBOW_LEFT = 6
#  WRIST_LEFT = 7, HAND_LEFT = 8, HANDTIP_LEFT = 9, THUMB_LEFT = 10, CLAVICLE_RIGHT = 11, SHOULDER_RIGHT = 12
#  ELBOW_RIGHT = 13, WRIST_RIGHT = 14, HAND_RIGHT = 15, HANDTIP_RIGHT = 16, THUMB_RIGHT = 17, HIP_LEFT = 18, KNEE_LEFT = 19
#  ANKLE_LEFT = 20, FOOT_LEFT = 21, HIP_RIGHT = 22, KNEE_RIGHT = 23, ANKLE_RIGHT = 24, FOOT_RIGHT = 25, HEAD = 26, NOSE =27
#  EYE_LEFT = 28, EAR_LEFT = 29, EYE_RIGHT= 30, EAR_RIGHT = 31


joint_names = ['PELVIS', 'SPINE_NAVAL', 'SPINE_CHEST', 'NECK', 'CLAVICLE_LEFT', 'SHOULDER_LEFT', 'ELBOW_LEFT',
               'WRIST_LEFT', 'HAND_LEFT', 'HANDTIP_LEFT', 'THUMB_LEFT', 'CLAVICLE_RIGHT', 'SHOULDER_RIGHT',
               'ELBOW_RIGHT', 'WRIST_RIGHT','HAND_RIGHT','HANDTIP_RIGHT','THUMB_RIGHT','HIP_LEFT','KNEE_LEFT',
               'ANKLE_LEFT','FOOT_LEFT','HIP_RIGHT','KNEE_RIGHT','ANKLE_RIGHT','FOOT_RIGHT','HEAD','NOSE',
               'EYE_LEFT','EAR_LEFT','EYE_RIGHT','EAR_RIGHT']

bone_list = [[26,3], [3,2],
             [2,11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [14, 17],
             [2,4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [7, 10],
             [2,1], [1, 0], [0,22], [22, 23], [23, 24], [24, 25], [0, 18], [18, 19],[19, 20], [20, 21]]

joint_ind_dic = {}
for i, name in enumerate(joint_names):
    joint_ind_dic[name] = i

N = len(joint_names)

joint_sz_3d = np.ones(N) * 200  # in mm
for i, name in enumerate(joint_names):
    if name is 'HEAD':
        joint_sz_3d[i] = 300
    if name is 'NOSE' or name is 'EAR_LEFT' or name is 'EAR_RIGHT' or name is 'EYE_LEFT' or name is 'EYE_RIGHT':
        joint_sz_3d[i] = 50
    if name is 'HANDTIP_LEFT' or name is 'HANDTIP_RIGHT' or name is 'THUMB_RIGHT' or name is 'THUMB_LEFT':
        joint_sz_3d[i] = 50
    if name is 'FOOT_LEFT' or name is 'FOOT_RIGHT':
        joint_sz_3d[i] = 50


out_joint_names = ['HEAD', 'NECK', 'SHOULDER_RIGHT', 'SHOULDER_LEFT', 'ELBOW_RIGHT', 'ELBOW_LEFT', 'WRIST_RIGHT',
               'WRIST_LEFT', 'SPINE_NAVAL', 'HIP_RIGHT', 'HIP_LEFT', 'KNEE_RIGHT', 'KNEE_LEFT', 'ANKLE_RIGHT', 'ANKLE_LEFT']

out_bone_list_names = [['SPINE_NAVAL', 'NECK'], ['NECK', 'HEAD'],
             ['NECK', 'SHOULDER_LEFT'], ['SHOULDER_LEFT', 'ELBOW_LEFT'], ['ELBOW_LEFT', 'WRIST_LEFT'],
             ['NECK', 'SHOULDER_RIGHT'], ['SHOULDER_RIGHT', 'ELBOW_RIGHT'], ['ELBOW_RIGHT', 'WRIST_RIGHT'],
             ['SPINE_NAVAL', 'HIP_LEFT'], ['HIP_LEFT', 'KNEE_LEFT'], ['KNEE_LEFT', 'ANKLE_LEFT'],
             ['SPINE_NAVAL', 'HIP_RIGHT'], ['HIP_RIGHT', 'KNEE_RIGHT'], ['KNEE_RIGHT', 'ANKLE_RIGHT']]

out_joint_ind = [joint_ind_dic[j_name] for j_name in out_joint_names]

joint_ind_dic_out = {}
for i, name in enumerate(out_joint_names):
    joint_ind_dic_out[name] = i
out_bone_list = []
for bone in out_bone_list_names:
    out_bone_list.append([joint_ind_dic_out[bone[0]], joint_ind_dic_out[bone[1]]])


# collate hard-coded for the above bone_list
def get_color(num):
    if num == 0:
        return (106, 106, 155)
    elif num == 1:
        return (71,130,255)
    elif num == 2:
        return (209, 206, 0)
    elif num == 3:
        return (204, 209, 72)
    elif num == 4:
        return (255, 255, 0)
    elif num == 5:
        return (28, 28, 28)
    elif num == 6:
        return (54,54,54)
    elif num == 7:
        return (79, 79, 79)
    elif num == 8:
        return (19, 69, 139)
    elif num == 9:
        return (45, 82, 160)
    elif num == 10:
        return (63, 133, 205)
    elif num == 11:
        return (204, 50, 153)
    elif num == 12:
        return (211, 0, 148)
    else:
        return (226, 43, 138)


def draw_bone_list(img, joint_2d_proj, bone_list, color=None):
    for k in range(0, len(bone_list)):
        pt_start = tuple(joint_2d_proj[bone_list[k][0]].astype(np.int))
        pt_end = tuple(joint_2d_proj[bone_list[k][1]].astype(np.int))
        if color is None:
            point_color = get_color(k)
        else:
            point_color = color
        img = cv.line(img, pt_start, pt_end, point_color, 2, 8)
    return img


def compute_2d_Bbox_from_3d_joints(joint_3d_positions, joint_sz_3d, K):
    """
    Computed the 3D BB for each joint's 3D position based on predefined joint_sz_3d, assuming z is fixed. Project these
    3D rectangles into 2D and computed the 2D BB for the whole body.
    """
    xmin_pts = copy.deepcopy(joint_3d_positions)
    xmin_pts[:, 0] = xmin_pts[:, 0] - joint_sz_3d
    xmax_pts = copy.deepcopy(joint_3d_positions)
    xmax_pts[:, 0] = xmax_pts[:, 0] + joint_sz_3d
    ymin_pts = copy.deepcopy(joint_3d_positions)
    ymin_pts[:, 1] = ymin_pts[:, 1] - joint_sz_3d
    ymax_pts = copy.deepcopy(joint_3d_positions)
    ymax_pts[:, 1] = ymax_pts[:, 1] + joint_sz_3d

    xmin_pts_2D = projective_transform(xmin_pts, K, np.eye(3), np.zeros((3, 1)))
    xmin = np.min(xmin_pts_2D[:, 0])
    xmax_pts_2D = projective_transform(xmax_pts, K, np.eye(3), np.zeros((3, 1)))
    xmax = np.max(xmax_pts_2D[:, 0])
    ymin_pts_2D = projective_transform(ymin_pts, K, np.eye(3), np.zeros((3, 1)))
    ymin = np.min(ymin_pts_2D[:, 1])
    ymax_pts_2D = projective_transform(ymax_pts, K, np.eye(3), np.zeros((3, 1)))
    ymax = np.max(ymax_pts_2D[:, 1])

    return np.array([xmin, ymin, xmax, ymax]).astype(np.int)


def main():


    # fourcc = cv.VideoWriter_fourcc(*'DIVX')
    #
    # videoWriter = cv.VideoWriter('data/'+ fname + "_kinect.avi", fourcc, 20, (480, 512), True)
    parser = argparse.ArgumentParser(description='Crop the image and get the bonelist information from kinect data.')
    parser.add_argument('--load_dir', help='The position of input file', default='/media/yuliang/DATA/Datasets/Kinect_Depth_Human3D_small/multiperson_test_raw/kinect_raw')
    parser.add_argument('--out_dir', help='The position of the output file', default='/media/yuliang/DATA/Datasets/Kinect_Depth_Human3D_small/multiperson_test_raw')
    parser.add_argument('--fname', help='The file name', default='trial_04')
    parser.add_argument('--vis', action = "store_true", help = 'If you want to visualize the images, add this parameter', default=False)

    args = parser.parse_args()

    fname = args.fname
    dir_name = args.load_dir
    out_dir = args.out_dir
    vis = args.vis

    depth_data_set2 = np.fromfile(dir_name + "/" + fname + ".bin", '<H')
    depth_data_set2 = depth_data_set2.reshape([-1, HEIGHT_2, WIDTH_2])
    kinect_skeleton_file = dir_name + "/" + fname + ".json"

    if out_dir and not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    # else:
    #     shutil.rmtree(out_dir)

    # load skeleton data
    with open(kinect_skeleton_file) as f:
        skeleton_data_set2 = json.load(f)

    # name_mask = np.where(np.in1d(skeleton_data_set2["joint_names"], joint_names))[0]
    src_names = np.array(skeleton_data_set2["joint_names"])
    for name in src_names:
        print(name)


    # load calibration matrices for Kinect
    K2 = np.eye(3)
    K2[0, 0] = skeleton_data_set2["intrinsics"]['fx']
    K2[1, 1] = skeleton_data_set2["intrinsics"]['fy']
    K2[0, 2] = skeleton_data_set2["intrinsics"]['cx']
    K2[1, 2] = skeleton_data_set2["intrinsics"]['cy']

    # distort coefs in order "(k1,k2,p1,p2[,k3[,k4,k5,k6)"
    distCoefs = np.array([skeleton_data_set2["intrinsics"]['k1'], skeleton_data_set2["intrinsics"]['k2'],
                          skeleton_data_set2["intrinsics"]['p1'], skeleton_data_set2["intrinsics"]['p2'],
                          skeleton_data_set2["intrinsics"]['k3'], skeleton_data_set2["intrinsics"]['k4'],
                          skeleton_data_set2["intrinsics"]['k5'], skeleton_data_set2["intrinsics"]['k6']])


    output_skeletons_2D = {}
    output_skeletons_3D = {}
    label = {}
    bodies = {}
    bodies["2D_joint_positions"] = []
    bodies["3D_joint_positions"] = []
    bodies["bounding_boxes"] = []

    label["joint_names"] = out_joint_names
    label["bone_list"] = out_bone_list
    label["2D_joint_positions"] = []
    label["bounding_boxes"] = []
    label["3D_joint_positions"] = []
    label["intrinsics"] = K2.tolist()
    label["distCoefs"] = distCoefs.tolist()
    label["frame_id"] = []
    label["bodies"] = []
    label["valid"] = []


    output_skeletons_2D["joint_names"] = joint_names
    output_skeletons_2D["bone_list"] = bone_list
    output_skeletons_2D["joint_positions"] = []
    output_skeletons_2D["bounding_boxes"] = []

    output_skeletons_3D["joint_names"] = joint_names
    output_skeletons_3D["bone_list"] = bone_list
    output_skeletons_3D["joint_positions"] = []
    output_skeletons_3D["intrinsics"] = K2.tolist()
    output_skeletons_3D["distCoefs"] = distCoefs.tolist()
    output_skeletons_3D["rot_axis_2"] = []

    skip_id = []

    src_names = np.array(skeleton_data_set2["joint_names"])
    # joint_subset_indices = []
    # for name in joint_names:
    #     joint_subset_indices.append(np.where(src_names == name)[0][0])

    saved_depth_data_set2 = np.empty((depth_data_set2.shape[0],img_height,img_width))
    for i in range(0,depth_data_set2.shape[0]):
        print('image {} out of {}'.format(i, depth_data_set2.shape[0]))
        # if bodies aren't detected, skip the frame
        if len(skeleton_data_set2['frames'][i]['bodies']) == 0:
            # skip_id.append(i)
            label["frame_id"].append(skeleton_data_set2['frames'][i]["frame_id"])
            label["2D_joint_positions"].append([])
            label["bounding_boxes"].append([])
            label["3D_joint_positions"].append([])
            label["valid"].append([])
            # continue
        else:
            for m in range(0,len(skeleton_data_set2['frames'][i]['bodies'])):
                joint_pos_3d_set2 = np.array(skeleton_data_set2['frames'][i]['bodies'][m]['joint_positions'])
                joint_pos_2d_set2 = projective_camera(joint_pos_3d_set2, K2)

                for j in range(0, joint_pos_2d_set2.shape[0]):
                    joint_pos_2d_set2[j][0] = joint_pos_2d_set2[j][0] - crop_x
                    joint_pos_2d_set2[j][1] = joint_pos_2d_set2[j][1] - crop_y

                # # compute 2D Bbox from 3D Bbox
                Bbox_2d = compute_2d_Bbox_from_3d_joints(joint_pos_3d_set2, joint_sz_3d, K2)
                Bbox_2d[0] -= crop_x
                Bbox_2d[1] -= crop_y
                Bbox_2d[2] -= crop_x
                Bbox_2d[3] -= crop_y

                if m == 0:
                    result_2D = joint_pos_2d_set2
                    result_3D = joint_pos_3d_set2
                    result_BB = Bbox_2d
                else:
                    result_2D = np.vstack((result_2D,joint_pos_2d_set2))
                    result_3D = np.vstack((result_3D, joint_pos_3d_set2))
                    result_BB = np.vstack((result_BB,Bbox_2d))

            vaild_set = np.ones(len(skeleton_data_set2['frames'][i]['bodies'])).astype('int')
            result_2D = result_2D.reshape(len(skeleton_data_set2['frames'][i]['bodies']),N,2)
            result_3D = result_3D.reshape(len(skeleton_data_set2['frames'][i]['bodies']), N, 3)
            result_BB = result_BB.reshape(len(skeleton_data_set2['frames'][i]['bodies']),4)

            # extract output joints positions
            result_2D = result_2D[:, out_joint_ind, :]
            result_3D = result_3D[:, out_joint_ind, :]

            label["frame_id"].append(skeleton_data_set2['frames'][i]["frame_id"])
            label["2D_joint_positions"].append(result_2D.tolist())
            label["bounding_boxes"].append(result_BB.tolist())
            label["3D_joint_positions"].append(result_3D.tolist())
            label["valid"].append(vaild_set.tolist())

        depth_data_set2[i, :, :] = cv.undistort(depth_data_set2[i, :, :], K2, distCoefs)
        saved_depth_data_set2[i, :, :] = depth_data_set2[i, crop_y: crop_y + img_height, crop_x: crop_x + img_width]

        if vis:
            # visualize on kinect image
            depth_img2 = copy.deepcopy(saved_depth_data_set2[i, :, :])
            print('num of bodies: {}'.format(result_2D.shape[0]))
            depth_img2[depth_img2 > DEPTH_MAX] = DEPTH_MAX
            depth_img2[depth_img2 < DEPTH_MIN] = DEPTH_MIN
            depth_img2 = depth_img2 / DEPTH_MAX * 255

            depth_img2 = cv.cvtColor(depth_img2.astype(np.uint8), cv.COLOR_GRAY2BGR)
            # draw skeleton
            for n in range(0,result_2D.shape[0]):
                for j in range(0, result_2D[n].shape[0]):
                    depth_img2 = cv.circle(depth_img2, tuple(result_2D[n, j, :].astype(np.int)), 3, [0, 0, 255], 2)
                # draw orientation of Pelvis

                depth_img2 = cv.rectangle(depth_img2, (result_BB[n][0] , result_BB[n][1] ), (result_BB[n][2] , result_BB[n][3] ), (0, 255, 0), 2)
                draw_bone_list(depth_img2, result_2D[n], out_bone_list, (19, 69, 139))

            cv.imshow('kinect skeleton', depth_img2)
            #videoWriter.write(depth_img2)
            cv.waitKey(200)

    saved_depth_data_set2 = np.delete(saved_depth_data_set2, skip_id, 0)
    print('number of depth maps to save: {}'.format(saved_depth_data_set2.shape[0]))

    print('number of labels to save: {}'.format(len(label["frame_id"])))
    # videoWriter.release()

    depth_out_file = out_dir + "/" + fname + ".npy"
    np.save(depth_out_file, saved_depth_data_set2)

    out_file = out_dir + "/" + fname + "_label.json"
    with open(out_file, 'w') as json_file:
        json.dump(label, json_file, indent=4)


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()