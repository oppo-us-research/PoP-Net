"""
@author: Yuliang Guo <33yuliangguo@gmail.com>

Specifically, the weight of each sample is computed and saved.
The weight of a sample is based on the probability of the pose.
"""


import os
from scipy.io import loadmat
import json
import argparse
import numpy as np
import sys
import os.path as ops
sys.path.insert(0, '../..')


num_joints = 16


def get_args():
    parser = argparse.ArgumentParser("Parsing human tof dataset")
    # parser.add_argument("--dataset_path", type=str, default="/media/yuliang/DATA/Datasets/Kinect_Depth_Human3D/multiperson_test_v2")
    parser.add_argument("--mpii_path", type=str, default="/media/yuliang/DATA/Datasets/MPII")
    parser.add_argument("--label_out_dir", type=str, default="labels")

    args = parser.parse_args()
    return args


def mkdir_if_missing(directory):
    if not ops.exists(directory):
        os.makedirs(directory)


def fix_wrong_joints(joint):
    if '12' in joint and '13' in joint and '2' in joint and '3' in joint:
        if ((joint['12'][0] < joint['13'][0]) and
                (joint['3'][0] < joint['2'][0])):
            joint['2'], joint['3'] = joint['3'], joint['2']
        if ((joint['12'][0] > joint['13'][0]) and
                (joint['3'][0] > joint['2'][0])):
            joint['2'], joint['3'] = joint['3'], joint['2']

    return joint


def save_joints(input_file):
    """
    Convert annotations mat file to json and save on disk.
    Only persons with annotations of all 16 joints will be written in the json.
    """

    mat = loadmat(input_file)

    data_set = []
    # train_set = []
    # test_set = []

    for i, (anno, train_flag) in enumerate(
        zip(mat['RELEASE']['annolist'][0, 0][0],
            mat['RELEASE']['img_train'][0, 0][0])):

        img_fn = anno['image']['name'][0, 0][0]
        train_flag = int(train_flag)

        if 'annopoints' in str(anno['annorect'].dtype):
            annopoints = anno['annorect']['annopoints'][0]
            head_x1s = anno['annorect']['x1'][0]
            head_y1s = anno['annorect']['y1'][0]
            head_x2s = anno['annorect']['x2'][0]
            head_y2s = anno['annorect']['y2'][0]
            for annopoint, head_x1, head_y1, head_x2, head_y2 in \
                    zip(annopoints, head_x1s, head_y1s, head_x2s, head_y2s):
                if len(annopoint) > 0:
                    head_rect = [float(head_x1[0, 0]),
                                 float(head_y1[0, 0]),
                                 float(head_x2[0, 0]),
                                 float(head_y2[0, 0])]

                    # joint coordinates
                    annopoint = annopoint['point'][0, 0]
                    j_id = [str(j_i[0, 0]) for j_i in annopoint['id'][0]]
                    x = [x[0, 0] for x in annopoint['x'][0]]
                    y = [y[0, 0] for y in annopoint['y'][0]]
                    joint_pos = {}
                    for _j_id, (_x, _y) in zip(j_id, zip(x, y)):
                        joint_pos[str(_j_id)] = [float(_x), float(_y)]
                    # joint_pos = fix_wrong_joints(joint_pos)

                    # visiblity list
                    if 'is_visible' in str(annopoint.dtype):
                        vis = [v[0] if len(v) > 0 else np.array([-1])
                               for v in annopoint['is_visible'][0]]
                        vis = dict([(k, int(v[0])) if len(v) > 0 else v
                                    for k, v in zip(j_id, vis)])
                    else:
                        vis = None
                        continue

                    if len(vis) != len(joint_pos):
                        continue

                    data = {
                        'filename': img_fn,
                        'train': train_flag,
                        'head_rect': head_rect,
                        'is_visible': vis,
                        'joint_pos': joint_pos
                    }
                    data_set.append(data)

    # with open(joint_data_fn, 'w') as json_file:
    #     json.dump(data_set, json_file, indent=4)

    # split train and test
    N = len(data_set)
    N_test = int(N * 0.1)
    N_train = N - N_test

    print('N:{}'.format(N))
    print('N_train:{}'.format(N_train))
    print('N_test:{}'.format(N_test))

    np.random.seed(1701)
    perm = np.random.permutation(N)
    test_indices = perm[:N_test]
    train_indices = perm[N_test:]
    train_set = [data_set[ind] for ind in train_indices]
    test_set = [data_set[ind] for ind in test_indices]
    # with open(joint_data_train, 'w') as json_file:
    #     json.dump(train_set, json_file, indent=4)
    #
    # with open(joint_data_test, 'w') as json_file:
    #     json.dump(test_set, json_file, indent=4)
    return data_set, train_set, test_set


def prepare_mpii_labels(annos, istrain=True):
    img_id = 0
    # annos = json.load(open(in_train_file, 'r'))
    # annos = open(in_train_file).readlines()
    image_label_dict = {}

    for i in range(len(annos)):

        save_name = annos[i]['filename']
        if save_name not in image_label_dict:
            image_label_dict[save_name] = []

        # append label file
        joints_2d = np.ones((num_joints, 2)) * (-1)
        visible_joints = np.zeros(num_joints)
        for j in range(num_joints):
            if str(j) in annos[i]['joint_pos'].keys():
                joints_2d[j, 0] = annos[i]['joint_pos'][str(j)][0]
                joints_2d[j, 1] = annos[i]['joint_pos'][str(j)][1]
                visible_joints[j] = (annos[i]['is_visible'][str(j)])
        # visible_joints = (visible_joints != 0).astype(np.int)

        if istrain:
            json_single_person = {
                '2d_joints': joints_2d.tolist(),
                'head_rect': annos[i]['head_rect'],
                'visible_joints': visible_joints.tolist()
            }
            image_label_dict[save_name].append(json_single_person)

        img_id += 1

    print('Processed {} samples.'.format(img_id))

    return image_label_dict


def main(opt):

    """
        prepare data set
    """

    # parse the origin .mat file int json
    MPII_DATA_DIR = ops.join(opt.mpii_path, 'annot_origin')
    input_file = os.path.join(MPII_DATA_DIR, 'mpii_human_pose_v1_u12_1.mat')
    data_set, train_set, test_set = save_joints(input_file)

    label_out_path = ops.join(opt.mpii_path, opt.label_out_dir)
    mkdir_if_missing(label_out_path)

    ################## Training Set ##############################
    out_file = ops.join(label_out_path, 'labels_train.json')

    image_label_dict_json = prepare_mpii_labels(train_set)

    print('Parsed training set \n')
    # save json label file
    with open(out_file, 'w') as json_file:
        json.dump(image_label_dict_json, json_file, indent=4)

    ################## Testing Set #############################
    out_file = ops.join(label_out_path, 'labels_test.json')

    image_label_dict_json = prepare_mpii_labels(test_set)

    print('Parsed testing set \n')
    # save json label file
    with open(out_file, 'w') as json_file:
        json.dump(image_label_dict_json, json_file, indent=4)

    ################## Train+Val Set ##############################
    out_file = ops.join(label_out_path, 'labels_traintest.json')

    image_label_dict_json = prepare_mpii_labels(data_set)

    print('Parsed all set \n')
    # save json label file
    with open(out_file, 'w') as json_file:
        json.dump(image_label_dict_json, json_file, indent=4)


if __name__ == '__main__':
    opt = get_args()
    main(opt)
