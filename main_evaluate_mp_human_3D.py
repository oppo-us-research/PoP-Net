"""
    The main function to evaluate multiple human predictions against ground-truth.
    Methods are evaluated under mAP and PCK, 2D and 3D.

    Author:
        Yuliang Guo (yuliang.guo@oppo.com)

    Date:
        Sep, 2020

"""

import json
import sys
from util import util_functions
from util.eval_mAP import *
from util.eval_pck import *


def parse_gt_labels(anno_dic):
    human_gt_set_2d = []
    human_gt_set_3d = []

    anno_ids = [key for key, value in anno_dic.items() if key != 'intrinsics']

    for i in range(len(anno_ids)):
        image_id = anno_ids[i]

        humans_2d = []
        humans_3d = []
        for ann in anno_dic[image_id]:
            humans_2d.append(ann['2d_joints'])
            humans_3d.append(ann['3d_joints'])
        human_gt_set_2d.append(humans_2d)
        human_gt_set_3d.append(humans_3d)

    return human_gt_set_2d, human_gt_set_3d


if __name__ is "__main__":
    joint_names = util_functions.get_keypoints()
    gt_file = 'labels/test_bgaug/labels.json'
    res_file = 'predictions/test_bgaug/pop_results.json'
    res_data = json.load(open(res_file, 'r'))
    if 'pop' in res_file:
        human_pred_set_2d = res_data['human_pred_set_2d_aligned']
        human_pred_set_3d = res_data['human_pred_set_3d_aligned']
    else:
        human_pred_set_2d = res_data['human_pred_set_2d']
        human_pred_set_3d = res_data['human_pred_set_3d']

    # parse gt labels into lists
    anno_dic = json.load(open(gt_file, 'r'))
    human_gt_set_2d, human_gt_set_3d = parse_gt_labels(anno_dic)

    #####################################################################################
    print('2d PCKh-0.5')
    joint_avg_dist, joint_KCP = eval_human_dataset_2d_PCKh(human_pred_set_2d,
                                                           human_gt_set_2d,
                                                           num_joints=util_functions.num_parts,
                                                           head_id=0, neck_id=1,
                                                           iou_th=0.5)
    for i, name in enumerate(joint_names):
        print('     joint: {},  PCK: {:03f}, avg 2D error: {:03f}'.format(name,
                                                                          joint_KCP[i],
                                                                          joint_avg_dist[i]))

    print('\n     Overall: PCK: {:03f}, avg 2D error: {:03f} \n'.format(np.average(joint_KCP),
                                                                        np.average(joint_avg_dist)))

    #####################################################################################
    print('3d PCK')
    joint_avg_dist, joint_KCP = eval_human_dataset_3d(human_pred_set_2d,
                                                      human_gt_set_2d,
                                                      human_pred_set_3d,
                                                      human_gt_set_3d,
                                                      num_joints=util_functions.num_parts,
                                                      dist_th=0.1,
                                                      iou_th=0.5)

    print('     3D threshold: {:03f} meter'.format(0.1))
    for i, name in enumerate(joint_names):
        print('     joint: {},  PCK: {:03f}, avg 3D error: {:03f}'.format(name,
                                                                          joint_KCP[i],
                                                                          joint_avg_dist[i]))

    print('\n     Overall: PCK: {:03f}, avg 3D error: {:03f} \n'.format(np.average(joint_KCP),
                                                                        np.average(joint_avg_dist)))

    #####################################################################################
    print('2d mAP')
    AP_2D = eval_ap_mpii_v2(human_pred_set_2d, res_data['human_pred_set_part_conf'],
                            human_gt_set_2d, gt_visibility_set=[],
                            head_id=0, neck_id=1, joint_names=joint_names, thresh=0.5)

    #####################################################################################
    print('3d mAP')
    AP_3D = eval_ap_3D(human_pred_set_3d, res_data['human_pred_set_part_conf'],
                       human_gt_set_3d, gt_visibility_set=[], joint_names=joint_names, thresh=0.1)

