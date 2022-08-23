import numpy as np
import sys
import json
sys.path.insert(0, '..')
from lib.datasets import datasets_kdh3d_mpreal


def compute_head_size_from_two_joints(humans, ind1, ind2):
    """
        use the 2 x euclidean distance between head top and neck as head size
    Args:
        humans:
        ind1: head top index
        ind2: neck index

    Returns:

    """
    hsz_vec = []
    for human in humans:
        hsz_vec.append(2*np.sqrt((human[ind1][0] - human[ind2][0])**2 + (human[ind1][1] - human[ind2][1])**2))
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


def assignGTmulti(humans_pred_set, conf_pred_set, humans_gt_set, gt_visibility_set, ref_dist_set, num_joints, thresh):
    """
    predicted human set and GT human set from a dataset are matched to calculate scoresAll, labelsAll, and nGTall
        Dimensions of scoresAll, labelsAll: number_joints x num_images x num_pred on each image (in list format)
        Dimensions of nGTall: num_joints x number_images (in ndarray)

    Args:
        humans_pred_set:
        humans_gt_set:
        ref_dist_set:
        num_joints:
        h_th:

    Returns:

    """
    # part detections scores
    scoresAll = [[[] for i in range(len(humans_gt_set))] for j in range(num_joints)]
    # positive / negative labels
    labelsAll = [[[] for i in range(len(humans_gt_set))] for j in range(num_joints)]
    # number of annotated GT joints per image
    nGTall = np.zeros((num_joints, len(humans_gt_set)))

    for imgidx in range(len(humans_gt_set)):
        # distance between predicted and GT joints
        dist = np.ones((len(humans_pred_set[imgidx]), len(humans_gt_set[imgidx]), num_joints)) * np.inf
        # score of the predicted joint
        score = np.zeros((len(humans_pred_set[imgidx]), num_joints))
        # body joint prediction exisit
        hasPred = np.zeros((len(humans_pred_set[imgidx]), num_joints))
        # body joint is annotataed
        hasGT = np.zeros((len(humans_gt_set[imgidx]), num_joints))

        if len(humans_pred_set[imgidx]) > 0:
            # iterate over predicted poses
            for ridxPred in range(len(humans_pred_set[imgidx])):
                # predicted poses
                pointsPred = humans_pred_set[imgidx][ridxPred]
                # iterate over GT poses
                for ridxGT in range(len(humans_gt_set[imgidx])):
                    # GT poses
                    pointsGT = humans_gt_set[imgidx][ridxGT]
                    # reference of head size
                    refDist = ref_dist_set[imgidx][ridxGT]
                    # iterate over all possible body joints
                    for i in range(num_joints):
                        # check pred point exist
                        if len(pointsPred[i]) > 0:
                            hasPred[ridxPred, i] = 1
                            score[ridxPred, i] = conf_pred_set[imgidx][ridxPred][i]
                        ppPred = np.array(pointsPred[i])
                        if len(pointsGT[i]) > 0 and gt_visibility_set[imgidx][ridxGT][i] > 0:
                            hasGT[ridxGT, i] = 1
                        ppGT = np.array(pointsGT[i])
                        # compute distance between predicted and GT joint locations
                        if hasPred[ridxPred, i] and hasGT[ridxGT, i]:
                            dist[ridxPred, ridxGT, i] = np.linalg.norm(ppPred - ppGT) / refDist

            # number of annotated joints
            nGT = np.repeat(np.sum(hasGT, 1).reshape([1, -1]), hasPred.shape[0], axis=0)
            # compute PCKh
            match = (dist <= thresh).astype(np.int)
            pck = np.sum(match, 2) / nGT
            # preserve best GT match only
            idx = np.argmax(pck, 1)
            for ridxPred in range(len(idx)):
                pck[ridxPred, np.array(range(pck.shape[1]))!=idx[ridxPred]] = 0
            val = np.max(pck, 0)
            predToGT = np.argmax(pck, 0)
            predToGT[val == 0] = -1  # assign invalid index for unqualified best matches

            # assign predicted poses to GT poses
            for ridxPred in range(len(humans_pred_set[imgidx])):
                if ridxPred in predToGT:  # pose matches to GT
                    # GT pose that matches the predicted
                    ridxGT = np.where(predToGT == ridxPred)[0][0]
                    s = score[ridxPred, :]
                    m = match[ridxPred, ridxGT, :]
                    hp = hasPred[ridxPred, :]
                    idxs = np.where(hp)[0]
                    for i in range(len(idxs)):
                        scoresAll[idxs[i]][imgidx].append(s[idxs[i]])
                        labelsAll[idxs[i]][imgidx].append(m[idxs[i]])
                else:  # no matching to GT
                    s = score[ridxPred, :]
                    m = np.zeros(match.shape[2])
                    hp = hasPred[ridxPred, :]
                    idxs = np.where(hp)[0]
                    for i in range(len(idxs)):
                        scoresAll[idxs[i]][imgidx].append(s[idxs[i]])
                        labelsAll[idxs[i]][imgidx].append(m[idxs[i]])

        # save number of GT joints
        for ridxGT in range(len(humans_gt_set[imgidx])):
            hg = hasGT[ridxGT, :]
            nGTall[:, imgidx] += hg

    return scoresAll, labelsAll, nGTall


def getRPC(class_margin, true_labels, totalpos):
    """
        For a part, compute precision and recall from all the confidence of prediction from the whole dataset, and the
        binary indicator of found matches, given a scalar of total number of positive samples.

    Args:
        class_margin: N-vector, N is the total number of predicted instances from the dataset
        true_labels: N-vector,
        totalpos: scalar

    Returns:

    """
    class_margin = np.array(class_margin)
    true_labels = np.array(true_labels)
    N = true_labels.shape[0]
    ndet = N
    npos = 0

    sortidx = np.flip(np.argsort(class_margin))
    sorted_labels = true_labels[sortidx]

    recall = np.zeros(ndet)
    precision = np.zeros(ndet)

    for ridx in range(ndet):
        if sorted_labels[ridx] == 1:
            npos = npos + 1
        precision[ridx] = npos / (ridx+1)
        recall[ridx] = npos / totalpos

    return precision, recall


def VOCap(recall, precision):
    vecN = len(recall) + 2
    mrec = np.zeros(vecN)
    mrec[1:-1] = recall
    mrec[-1] = 1.
    mpre = np.zeros(vecN)
    mpre[1:-1] = precision
    mpre[-1] = 0.

    for i in range(vecN-2, -1, -1):
        mpre[i] = np.maximum(mpre[i], mpre[i+1])
    indices = np.where((mrec[1:] - mrec[:-1]) > 0)[0] + 1
    ap = np.sum((mrec[indices] - mrec[indices-1])*mpre[indices])
    return ap


def eval_ap_mpii(humans_pred_set, conf_pred_set, humans_gt_set, gt_visibility_set, head_sz_set, joint_names, thresh=0.5):
    """
    the main evaluation function calculating ap for each joint.
    Args:
        humans_pred_set:
        humans_gt_set:
        head_sz_set:
        num_joints:
        thresh:

    Returns:

    """
    print('2D evaluation in AP evaluation under PCKh-{:01f} rule ...'.format(thresh))

    assert len(humans_gt_set) == len(humans_pred_set)
    num_joints = len(joint_names)

    # compute the reference dist set
    ref_dist_set = []
    for i in range(len(humans_gt_set)):
        hsz_vec = compute_head_size_from_rect(head_sz_set[i])
        ref_dist_set.append(hsz_vec)

    # consider all gt joints visible when no input is specified
    if len(gt_visibility_set) == 0:
        for i in range(len(humans_gt_set)):
            visibilites = np.ones((len(humans_gt_set[i]), num_joints))
            gt_visibility_set.append(visibilites.tolist())

    # if no predicted confidence provided, consider them all equal
    if len(conf_pred_set) == 0:
        for i in range(len(humans_pred_set)):
            conf_vec = np.ones((len(humans_pred_set[i]), num_joints))
            conf_pred_set.append(conf_vec.tolist())

    # assign predicted poses to GT poses
    scoresAll, labelsAll, nGTall = assignGTmulti(humans_pred_set, conf_pred_set,
                                                 humans_gt_set, gt_visibility_set,
                                                 ref_dist_set, num_joints, thresh)

    # compute average precision (AP) per part
    ap = np.zeros(num_joints + 1)
    for j in range(num_joints):
        scores = []
        labels = []
        for i in range(len(humans_pred_set)):
            scores += (scoresAll[j][i])
            labels += (labelsAll[j][i])
        # compute precision/recall
        precision, recall = getRPC(scores, labels, np.sum(nGTall[j, :]))
        # compute AP
        ap[j] = VOCap(recall, precision)*100
    ap[-1] = np.mean(ap[:-1])

    for j, name in enumerate(joint_names):
        print('    {},  AP: {:03f}'.format(name, ap[j]))
    print('\n     Overall: AP: {:03f}\n'.format(ap[-1]))

    return ap


def eval_ap_mpii_v2(humans_pred_set, conf_pred_set, humans_gt_set, gt_visibility_set, head_id, neck_id, joint_names, thresh=0.5):
    """
    the main evaluation function calculating ap for each joint.
    this version uses the 2 x distance between neck and head center as reference distance
    Args:
        humans_pred_set:
        humans_gt_set:
        head_sz_set:
        num_joints:
        thresh:

    Returns:

    """
    print('2D evaluation in AP evaluation under PCKh-{:01f} rule ...'.format(thresh))

    assert len(humans_gt_set) == len(humans_pred_set)
    num_joints = len(joint_names)

    # compute the reference dist set
    ref_dist_set = []
    for i in range(len(humans_gt_set)):
        hsz_vec = compute_head_size_from_two_joints(humans_gt_set[i], head_id, neck_id)
        ref_dist_set.append(hsz_vec)

    # consider all gt joints visible when no input is specified
    if len(gt_visibility_set) == 0:
        for i in range(len(humans_gt_set)):
            visibilites = np.ones((len(humans_gt_set[i]), num_joints))
            gt_visibility_set.append(visibilites.tolist())

    # if no predicted confidence provided, consider them all equal
    if len(conf_pred_set) == 0:
        for i in range(len(humans_pred_set)):
            conf_vec = np.ones((len(humans_pred_set[i]), num_joints))
            conf_pred_set.append(conf_vec.tolist())

    # assign predicted poses to GT poses
    scoresAll, labelsAll, nGTall = assignGTmulti(humans_pred_set, conf_pred_set,
                                                 humans_gt_set, gt_visibility_set,
                                                 ref_dist_set, num_joints, thresh)

    # compute average precision (AP) per part
    ap = np.zeros(num_joints + 1)
    for j in range(num_joints):
        scores = []
        labels = []
        for i in range(len(humans_pred_set)):
            scores += (scoresAll[j][i])
            labels += (labelsAll[j][i])
        # compute precision/recall
        precision, recall = getRPC(scores, labels, np.sum(nGTall[j, :]))
        # compute AP
        ap[j] = VOCap(recall, precision)*100
    ap[-1] = np.mean(ap[:-1])

    for j, name in enumerate(joint_names):
        print('    {},  AP: {:03f}'.format(name, ap[j]))
    print('\n     Overall: AP: {:03f}\n'.format(ap[-1]))

    return ap


def eval_ap_3D(humans_pred_set, conf_pred_set, humans_gt_set, gt_visibility_set, joint_names, thresh=0.1):
    """
    the main evaluation function calculating ap for each joint, 3D version only needs slight change
    Args:
        humans_pred_set:
        conf_pred_set:
        humans_gt_set:
        gt_visibility_set:
        num_joints:
        thresh: 0.1 meter

    Returns:

    """
    print('3D evaluation in AP under {:01f} meter rule ...'.format(thresh))

    assert len(humans_gt_set) == len(humans_pred_set)
    num_joints = len(joint_names)

    # assign reference dist set as ones for 3D case
    ref_dist_set = []
    for i in range(len(humans_gt_set)):
        hsz_vec = np.ones(len(humans_gt_set[i]))
        ref_dist_set.append(hsz_vec.tolist())

    # consider all gt joints visible when no input is specified
    if len(gt_visibility_set) == 0:
        for i in range(len(humans_gt_set)):
            visibilites = np.ones((len(humans_gt_set[i]), num_joints))
            gt_visibility_set.append(visibilites.tolist())

    # if no predicted confidence provided, consider them all equal
    if len(conf_pred_set) == 0:
        for i in range(len(humans_pred_set)):
            conf_vec = np.ones((len(humans_pred_set[i]), num_joints))
            conf_pred_set.append(conf_vec.tolist())

    # assign predicted poses to GT poses
    scoresAll, labelsAll, nGTall = assignGTmulti(humans_pred_set, conf_pred_set,
                                                 humans_gt_set, gt_visibility_set,
                                                 ref_dist_set, num_joints, thresh)

    # compute average precision (AP) per part
    ap = np.zeros(num_joints + 1)
    for j in range(num_joints):
        scores = []
        labels = []
        for i in range(len(humans_pred_set)):
            scores += (scoresAll[j][i])
            labels += (labelsAll[j][i])
        # compute precision/recall
        precision, recall = getRPC(scores, labels, np.sum(nGTall[j, :]))
        # compute AP
        ap[j] = VOCap(recall, precision)*100
    ap[-1] = np.mean(ap[:-1])

    for j, name in enumerate(joint_names):
        print('    {},  AP: {:03f}'.format(name, ap[j]))
    print('\n     Overall: AP: {:03f}\n'.format(ap[-1]))

    return ap


if __name__ is "__main__":
    joint_names = datasets_kdh3d_mpreal.get_keypoints()
    eval_file = '../predictions/pop_net_kdh3d_mpaug_mpreal_0903/eval_data.json'
    eval_data = json.load(open(eval_file, 'r'))

    AP_2D = eval_ap_mpii_v2(eval_data['human_pred_set_2d_aligned'], eval_data['human_pred_set_part_conf'],
                            eval_data['human_gt_set_2d'], gt_visibility_set=[],
                            head_id=0, neck_id=1, joint_names=joint_names, thresh=0.5)

    AP_3D = eval_ap_3D(eval_data['human_pred_set_3d_aligned'], eval_data['human_pred_set_part_conf'],
                       eval_data['human_gt_set_3d'], gt_visibility_set=[], joint_names=joint_names, thresh=0.1)
