import numpy as np
import json
from lib.datasets import datasets_itop_rtpose


def eval_human_dataset_2d(humans_pred_set, humans_gt_set, num_joints=15, dist_th=10.0, iou_th=0.5, human_gt_set_visibility=None):
    """
    Evaluation of the full dataset by matching predicted humans to ground-truth humans.
    This evaluation considers multi-to-multi matching, although itop dataset only includes single person.
    For each ground-truth human, find its max overlapping predicted human vis bbox IOUs.
    For each pair of matched poses, compute distance per joint. If the distance is < dist_th, the gt joint is
    considered detected.

    Overall metric: average Keypoint Correct Percentage (KCP) over all the joints and over the whole test set

    ATTENTION: those missed detection due to occlusion are not treated separately. If the guess is wrong, it's punished.


    :param humans_pred_set: N x list of humans for N images, each human is a K x 2 matrix recording x, y image coordinates
    :param humans_gt_set: N x list of humans for N images, each human is a K x 2 matrix recording x, y image coordinates
    :param num_joints: number of joints per person
    :param dist_th: the distance threshold in pixels
    :param iou_th: the iou threshold considering two human poses overlapping
    :return:
        joint_avg_dist: average distance per joint for those matched ground-truth joints only
        joint_KCP: KCP per joint over the test set
    """

    assert len(humans_gt_set) == len(humans_pred_set)
    human_gt_set_visibility_all = []
    samples_cnt = 0  # number of humans, not number of images
    joint_dists_set = []
    for i in range(len(humans_gt_set)):
        # print('evaluate {}/{}'.format(i, len(humans_gt_set)))
        humans_gt = humans_gt_set[i]
        humans_pred = humans_pred_set[i]
        samples_cnt += len(humans_gt)

        if len(humans_gt) == 0:
            continue

        joint_dists = match_humans_2d(humans_pred, humans_gt, iou_th)

        if human_gt_set_visibility is not None:
            for j, human_gt_visibility in enumerate(human_gt_set_visibility[i]):
                human_gt_set_visibility_all.append(human_gt_visibility)
                joint_dists[j][np.array(human_gt_visibility) == 0] = -1

        joint_dists_set += joint_dists

    human_gt_set_visibility_all = np.array(human_gt_set_visibility_all)

    joint_dists_set = np.array(joint_dists_set)
    joint_avg_dist = []
    joint_KCP = []
    for k in range(num_joints):
        single_joint_dists = joint_dists_set[:, k]
        joint_avg_dist.append(np.average(single_joint_dists[np.where(single_joint_dists >= 0)]))
        hit_cnt = np.sum(np.logical_and(single_joint_dists >= 0, single_joint_dists < dist_th))
        # A option to only consider gt visible parts
        if human_gt_set_visibility_all.shape[0] is not 0:
            joint_KCP.append(hit_cnt / np.sum(human_gt_set_visibility_all[:, k]))
        else:
            joint_KCP.append(hit_cnt / samples_cnt)

    return joint_avg_dist, joint_KCP


def eval_human_dataset_2d_PCKh(humans_pred_set, humans_gt_set, ind1, ind2, num_joints=15, h_th=0.5, iou_th=0.5, human_gt_set_visibility=None):
    """
    Evaluation of the full dataset by matching predicted humans to ground-truth humans.
    This evaluation considers multi-to-multi matching, although itop dataset only includes single person.
    For each ground-truth human, find its max overlapping predicted human vis bbox IOUs.
    For each pair of matched poses, compute distance per joint. If the distance is < dist_th, the gt joint is
    considered detected.

    Overall metric: average Keypoint Correct Percentage (KCP) over all the joints and over the whole test set

    ATTENTION: those missed detection due to occlusion are not treated separately. If the guess is wrong, it's punished.


    :param humans_pred_set: N x list of humans for N images, each human is a K x 2 matrix recording x, y image coordinates
    :param humans_gt_set: N x list of humans for N images, each human is a K x 2 matrix recording x, y image coordinates
    :param num_joints: number of joints per person
    :param dist_th: the distance threshold in pixels
    :param iou_th: the iou threshold considering two human poses overlapping
    :return:
        joint_avg_dist: average distance per joint for those matched ground-truth joints only
        joint_KCP: KCP per joint over the test set
    """

    assert len(humans_gt_set) == len(humans_pred_set)

    if human_gt_set_visibility is None:
        human_gt_set_visibility = []
        for i in range(len(humans_gt_set)):
            visibilites = np.ones((len(humans_gt_set[i]), num_joints))
            human_gt_set_visibility.append(visibilites.tolist())

    human_gt_set_visibility_all = []
    samples_cnt = 0  # number of humans, not number of images
    joint_dists_set = []
    hit_vec = []
    for i in range(len(humans_gt_set)):
        # print('evaluate {}/{}'.format(i, len(humans_gt_set)))
        humans_gt = humans_gt_set[i]
        humans_pred = humans_pred_set[i]
        samples_cnt += len(humans_gt)

        if len(humans_gt) == 0:
            continue

        joint_dists = match_humans_2d(humans_pred, humans_gt, iou_th)
        hsz_vec = compute_head_size(humans_gt, ind1, ind2)

        if human_gt_set_visibility is not None:
            for j, human_gt_visibility in enumerate(human_gt_set_visibility[i]):
                human_gt_set_visibility_all.append(human_gt_visibility)
                joint_dists[j][np.array(human_gt_visibility) == 0] = -1
                hit_vec.append(np.logical_and(joint_dists[j] >= 0, joint_dists[j] < hsz_vec[j]*h_th))
        joint_dists_set += joint_dists

    human_gt_set_visibility_all = np.array(human_gt_set_visibility_all)

    joint_dists_set = np.array(joint_dists_set)
    hit_vec = np.array(hit_vec)
    joint_avg_dist = []
    joint_KCP = []
    for k in range(num_joints):
        single_joint_dists = joint_dists_set[:, k]
        joint_avg_dist.append(np.average(single_joint_dists[np.where(single_joint_dists >= 0)]))

        # hit_cnt = np.sum(np.logical_and(single_joint_dists >= 0, single_joint_dists < dist_th))
        hit_cnt = np.sum(hit_vec[:, k])
        # A option to only consider gt visible parts
        if human_gt_set_visibility_all.shape[0] is not 0:
            joint_KCP.append(hit_cnt / np.sum(human_gt_set_visibility_all[:, k]))
        else:
            joint_KCP.append(hit_cnt / samples_cnt)

    return joint_avg_dist, joint_KCP


def eval_human_dataset_2d_PCKh_rect(humans_pred_set, humans_gt_set, head_sz_set, num_joints=15, h_th=0.5, iou_th=0.5, human_gt_set_visibility=None):
    """
    Evaluation of the full dataset by matching predicted humans to ground-truth humans.
    This evaluation considers multi-to-multi matching, although itop dataset only includes single person.
    For each ground-truth human, find its max overlapping predicted human vis bbox IOUs.
    For each pair of matched poses, compute distance per joint. If the distance is < dist_th, the gt joint is
    considered detected.

    Overall metric: average Keypoint Correct Percentage (KCP) over all the joints and over the whole test set

    ATTENTION: those missed detection due to occlusion are not treated separately. If the guess is wrong, it's punished.


    :param humans_pred_set: N x list of humans for N images, each human is a K x 2 matrix recording x, y image coordinates
    :param humans_gt_set: N x list of humans for N images, each human is a K x 2 matrix recording x, y image coordinates
    :param num_joints: number of joints per person
    :param dist_th: the distance threshold in pixels
    :param iou_th: the iou threshold considering two human poses overlapping
    :return:
        joint_avg_dist: average distance per joint for those matched ground-truth joints only
        joint_KCP: KCP per joint over the test set
    """

    assert len(humans_gt_set) == len(humans_pred_set)

    if human_gt_set_visibility is None:
        human_gt_set_visibility = []
        for i in range(len(humans_gt_set)):
            visibilites = np.ones((len(humans_gt_set[i]), num_joints))
            human_gt_set_visibility.append(visibilites.tolist())

    human_gt_set_visibility_all = []
    samples_cnt = 0  # number of humans, not number of images
    joint_dists_set = []
    hit_vec = []
    for i in range(len(humans_gt_set)):
        # print('evaluate {}/{}'.format(i, len(humans_gt_set)))
        humans_gt = humans_gt_set[i]
        humans_pred = humans_pred_set[i]
        samples_cnt += len(humans_gt)

        if len(humans_gt) == 0:
            continue

        joint_dists = match_humans_2d(humans_pred, humans_gt, iou_th)
        hsz_vec = compute_head_size_from_rect(head_sz_set[i])

        if human_gt_set_visibility is not None:
            for j, human_gt_visibility in enumerate(human_gt_set_visibility[i]):
                human_gt_set_visibility_all.append(human_gt_visibility)
                joint_dists[j][np.array(human_gt_visibility) == 0] = -1
                hit_vec.append(np.logical_and(joint_dists[j] >= 0, joint_dists[j] < hsz_vec[j]*h_th))
        joint_dists_set += joint_dists

    human_gt_set_visibility_all = np.array(human_gt_set_visibility_all)

    joint_dists_set = np.array(joint_dists_set)
    hit_vec = np.array(hit_vec)
    joint_avg_dist = []
    joint_KCP = []
    for k in range(num_joints):
        single_joint_dists = joint_dists_set[:, k]
        joint_avg_dist.append(np.average(single_joint_dists[np.where(single_joint_dists >= 0)]))

        # hit_cnt = np.sum(np.logical_and(single_joint_dists >= 0, single_joint_dists < dist_th))
        hit_cnt = np.sum(hit_vec[:, k])
        # A option to only consider gt visible parts
        if human_gt_set_visibility_all.shape[0] is not 0:
            joint_KCP.append(hit_cnt / np.sum(human_gt_set_visibility_all[:, k]))
        else:
            joint_KCP.append(hit_cnt / samples_cnt)

    return joint_avg_dist, joint_KCP


def compute_head_size(humans, ind1, ind2):
    """
        use the 2x euclidean distance between head center and neck as head size
    Args:
        humans:
        ind1: head top index
        ind2: neck index

    Returns:

    """
    hsz_vec = []
    for human in humans:
        hsz_vec.append(2*np.sqrt((human[ind1][0] -  human[ind2][0])**2 + (human[ind1][1] -  human[ind2][1])**2))
    return hsz_vec


def compute_head_size_from_rect(head_rects, SC_BIAS=0.6):
    """
        use the euclidean distance between head top and neck as head size
    Args:
        humans:
        ind1: head top index
        ind2: neck index

    Returns:

    """
    hsz_vec = []
    for rect in head_rects:
        hsz_vec.append(np.sqrt((rect[2] - rect[0])**2 + (rect[3] - rect[1])**2)*SC_BIAS)
    return hsz_vec


def match_humans_2d(humans_pred, humans_gt, iou_th=0.5):
    """
    For each ground-truth human, find its max overlapping predicted human vis bbox IOUs.
    For each pair of matched poses, compute distance per joint.
    If the whole ground-truth person got no match, its corresponding joint dists are assigned to be -1.

    ATTENTION:
    1. Assume gt list is not empty, but pred list can be empty.
    2. Those invalid joint due to occlusion lead to invalid distance -1

    :param humans_pred: list of list x, y pos
    :param humans_gt: list of list x, y pos
    :return: a list of K-length array recording joint distances per ground-truth human
    """
    joint_dists = []
    # if prediction is empty, return -1 distance for each gt joint
    if len(humans_pred) == 0:
        for human_gt in humans_gt:
            joint_dists.append(np.ones(len(human_gt))*(-1))
        return joint_dists

    # compute bbox
    bboxes_gt = compute_bbox_from_humans(humans_gt)
    bboxes_pred = compute_bbox_from_humans(humans_pred)

    # compute ious
    ious = bbox_ious(bboxes_gt, bboxes_pred)

    # compute matched joint distance per ground-truth human
    for i, human_gt in enumerate(humans_gt):
        if np.max(ious[i, :]) < iou_th:
            joint_dists.append(np.ones(len(human_gt))*(-1))
            continue

        human_pred = humans_pred[np.argmax(ious[i, :])]
        human_gt = np.array(human_gt)
        human_pred = np.array(human_pred)
        joint_dist = np.sqrt(np.sum((human_gt-human_pred)**2, axis=1))
        # invalid detected joint leads to invalid distance -1
        joint_dist[np.logical_and(human_pred[:, 0] == -1, human_pred[:, 1] == -1)] = -1
        # if np.sum(np.logical_and(human_pred[:, 0] == -1, human_pred[:, 1] == -1)) > 0:
        #     print('find invalid joint')
        joint_dists.append(joint_dist)

    return joint_dists


def eval_human_dataset_3d(humans_pred_set_2d, humans_gt_set_2d, humans_pred_set_3d, humans_gt_set_3d, num_joints=15, dist_th=0.1, iou_th=0.5, human_gt_set_visibility=None):
    """
    Evaluation of the full dataset by matching predicted humans to ground-truth humans.
    This evaluation considers multi-to-multi matching, although itop dataset only includes single person.
    For each ground-truth human, find its max overlapping predicted human vis bbox IOUs.
    For each pair of matched poses, compute distance per joint. If the distance is < dist_th, the gt joint is
    considered detected.

    Overall metric: average Keypoint Correct Percentage (KCP) over all the joints and over the whole test set

    ATTENTION: those missed detection due to occlusion are not treated separately. If the guess is wrong, it's punished.


    :param humans_pred_set_2d: N x list of humans for N images, each human is a K x 2 matrix recording x, y image coordinates
    :param humans_gt_set_2d: N x list of humans for N images, each human is a K x 2 matrix recording x, y image coordinates
    :param num_joints: number of joints per person
    :param dist_th: the distance threshold in meters
    :param iou_th: the iou threshold considering two human poses overlapping
    :return:
        joint_avg_dist: average distance per joint for those matched ground-truth joints only
        joint_KCP: KCP per joint over the test set
    """

    assert len(humans_gt_set_2d) == len(humans_pred_set_2d)

    samples_cnt = 0  # number of humans, not number of images
    joint_dists_set = []
    human_gt_set_visibility_all = []
    for i in range(len(humans_gt_set_2d)):
        # print('evaluate {}/{}'.format(i, len(humans_gt_set_2d)))
        humans_gt_2d = humans_gt_set_2d[i]
        humans_pred_2d = humans_pred_set_2d[i]
        humans_gt_3d = humans_gt_set_3d[i]
        humans_pred_3d = humans_pred_set_3d[i]
        samples_cnt += len(humans_gt_2d)

        if len(humans_gt_2d) == 0:
            continue

        joint_dists = match_humans_3d(humans_pred_2d, humans_gt_2d, humans_pred_3d, humans_gt_3d, iou_th)

        if human_gt_set_visibility is not None:
            for j, human_gt_visibility in enumerate(human_gt_set_visibility[i]):
                human_gt_set_visibility_all.append(human_gt_visibility)
                joint_dists[j][np.array(human_gt_visibility) == 0] = -1

        joint_dists_set += joint_dists
    human_gt_set_visibility_all = np.array(human_gt_set_visibility_all)

    joint_dists_set = np.array(joint_dists_set)
    joint_avg_dist = []
    joint_KCP = []
    for k in range(num_joints):
        single_joint_dists = joint_dists_set[:, k]
        joint_avg_dist.append(np.average(single_joint_dists[np.where(single_joint_dists >= 0)]))
        hit_cnt = np.sum(np.logical_and(single_joint_dists >= 0, single_joint_dists < dist_th))
        if human_gt_set_visibility is not None:
            joint_KCP.append(hit_cnt / np.sum(human_gt_set_visibility_all[:, k]))
        else:
            joint_KCP.append(hit_cnt / samples_cnt)

    return joint_avg_dist, joint_KCP


def match_humans_3d(humans_pred_2d, humans_gt_2d, humans_pred_3d, humans_gt_3d, iou_th=0.5):
    """
    For each ground-truth human, find its max overlapping predicted human vis bbox IOUs.
    For each pair of matched poses, compute distance per joint.
    If the whole ground-truth person got no match, its corresponding joint dists are assigned to be -1.

    ATTENTION:
    1. Assume gt list is not empty, but pred list can be empty.
    2. Those invalid joint due to occlusion lead to invalid distance -1

    :param humans_pred_2d: list of list x, y pos in image plane
    :param humans_gt_2d: list of list x, y pos in image plane
    :param humans_pred_3d: list of list x, y, z pos in camera coordinate frame
    :param humans_gt_3d: list of list x, y, x pos in camera coordinate frame
    :return: a list of K-length array recording joint distances per ground-truth human
    """
    joint_dists = []
    # if prediction is empty, return -1 distance for each gt joint
    if len(humans_pred_2d) == 0:
        for human_gt in humans_gt_2d:
            joint_dists.append(np.ones(len(human_gt))*(-1))
        return joint_dists

    # compute bbox in 2D
    bboxes_gt = compute_bbox_from_humans(humans_gt_2d)
    bboxes_pred = compute_bbox_from_humans(humans_pred_2d)

    # compute ious
    ious = bbox_ious(bboxes_gt, bboxes_pred)

    # compute matched joint 3D distance per ground-truth human
    for i, human_gt in enumerate(humans_gt_3d):
        if np.max(ious[i, :]) < iou_th:
            joint_dists.append(np.ones(len(human_gt))*(-1))
            continue

        human_pred = humans_pred_3d[np.argmax(ious[i, :])]
        human_gt = np.array(human_gt)
        human_pred = np.array(human_pred)
        joint_dist = np.sqrt(np.sum((human_gt-human_pred)**2, axis=1))

        # invalid detected joint leads to invalid distance -1
        human_pred2d = humans_pred_2d[np.argmax(ious[i, :])]
        human_pred2d = np.array(human_pred2d)
        joint_dist[np.logical_and(human_pred2d[:, 0] == -1, human_pred2d[:, 1] == -1)] = -1

        # TODO: invisible gt does not count distance (if to consider all GT, do not update 2D gt pos to (-1, -1))
        human_gt2d = humans_gt_2d[i]
        human_gt2d = np.array(human_gt2d)
        joint_dist[np.logical_and(human_gt2d[:, 0] == -1, human_gt2d[:, 1] == -1)] = -1

        joint_dists.append(joint_dist)

    return joint_dists


def compute_bbox_from_humans(humans):
    """
        ATTENTION: only those valid joints are used to calculate bbox
    :param humans: pure list of list
    :return:
    """
    bboxes = []
    for human in humans:
        valid_joints = np.array([joint for joint in human if joint != [-1, -1]])
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


if __name__ is "__main__":
    eval_file = '../predictions/pop_net_kdh3d_mpaug_vis_mpreal/eval_data.json'
    eval_data = json.load(open(eval_file, 'r'))
    ablation = True
    multiperson = False
    w_org = 480
    h_org = 512
    start_index = 0

    for key, value in eval_data.items():
        eval_data[key] = [eval_data[key][ind] for ind in range(start_index, len(eval_data[key]))]

    if not ablation:
        #####################################################################################
        print('\nevaluating in 2D...')
        dist_th_2d = 0.02*np.sqrt(h_org**2 + w_org**2)
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

        # save json result
        json_out_file = 'predictions/rtpose_light3d_itop/eval_res2d.json'
        json_eval_res = {'dist_th_2d': dist_th_2d,
                         'joint_names': joint_names,
                         'joint_avg_2d_error': joint_avg_dist,
                         'joint_KCP': joint_KCP,
                         'overall_avg_2d_error': np.average(joint_avg_dist),
                         'overall_KCP': np.average(joint_KCP)}
        with open(json_out_file, 'w') as json_file:
            json.dump(json_eval_res, json_file, indent=4)

        #####################################################################################
        print('\nevaluating in 3D...')
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

        # save json result
        json_out_file = 'predictions/rtpose_light3d_itop/eval_res3d.json'
        json_eval_res = {'dist_th_3d': 0.1,
                         'joint_names': joint_names,
                         'joint_avg_3d_error': joint_avg_dist,
                         'joint_KCP': joint_KCP,
                         'overall_avg_3d_error': np.average(joint_avg_dist),
                         'overall_KCP': np.average(joint_KCP)}
        with open(json_out_file, 'w') as json_file:
            json.dump(json_eval_res, json_file, indent=4)

    else:

        """
        Ablation study
        """

        print('\nevaluating in 2D...')
        dist_th_2d = 0.02 * np.sqrt(w_org ** 2 + h_org ** 2)
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

        #####################################################################################
        print('\nevaluating in 2D aligned...')
        joint_avg_dist, joint_KCP = eval_human_dataset_2d(eval_data['human_pred_set_2d_aligned'],
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


        #####################################################################################
        print('\nevaluating in 3D read from Z map')
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
        print('\nevaluating in 3D aligned result read from Z map')
        joint_avg_dist, joint_KCP = eval_human_dataset_3d(eval_data['human_pred_set_2d_aligned'],
                                                          eval_data['human_gt_set_2d'],
                                                          eval_data['human_pred_set_3d_aligned'],
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
        if not multiperson:
            print('\nevaluating in 3D read from Z map given perfect 2d')
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

        if not multiperson:
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

