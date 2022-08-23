import numpy as np
import json

"""
The evaluation method in this file is limited to single person per image
"""

keypointsNumber = 15
cropWidth = 288
cropHeight = 288
depthFactor = 50

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


def pixel2world(x, y, z):
    worldX = (x - 160.0) * z * 0.0035
    worldY = (120.0 - y) * z * 0.0035
    return worldX, worldY


def world2pixel(x, y, z):
    pixelX = 160.0 + x / (0.0035 * z)
    pixelY = 120.0 - y / (0.0035 * z)
    return pixelX, pixelY


def evaluation10CMRule(source, target):
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


def evaluation10CMRule_perJoint(source, target):
    assert np.shape(source) == np.shape(target), "source has different shape with target"
    count = 0
    acc_vec = []

    for j in range(keypointsNumber):
        for i in range(len(source)):
            if np.square(source[i, j, 0] - target[i, j, 0]) + np.square(
                    source[i, j, 1] - target[i, j, 1]) + np.square(source[i, j, 2] - target[i, j, 2]) < np.square(
                    0.1):  # 10cm
                count = count + 1

        accuracy = count / (len(source))
        print('joint_', j, joint_id_to_name[j], ', accuracy: ', accuracy)
        acc_vec.append(accuracy)
        accuracy = 0
        count = 0
    # print('avg:', np.mean(np.array(acc_vec)))


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
    for i in range(len(source)):
        for j in range(keypointsNumber):
            if np.square(source[i, j, 0] - target[i, j, 0]) + np.square(
                    source[i, j, 1] - target[i, j, 1]) < np.square(dist_th_2d):
                count = count + 1
    accuracy = count / (len(source) * keypointsNumber)
    return accuracy


if __name__ is "__main__":
    w_org = 320
    h_org = 240
    dist_th_2d = 0.02 * np.sqrt(w_org ** 2 + h_org ** 2)

    eval_file = '../predictions/pop_net_roi_v1_1_3_1_itop/eval_data.json'
    eval_data = json.load(open(eval_file, 'r'))

    print('Evaluate 3d')
    source = np.array(eval_data['human_pred_set_3d_aligned'])
    target = np.array(eval_data['human_gt_set_3d'])
    source = np.squeeze(source, axis=1)
    target = np.squeeze(target, axis=1)

    evaluation10CMRule_perJoint(source, target)
    Accuracy_test = evaluation10CMRule(source, target)
    print('Accuracy:', Accuracy_test)
    print('\n')

    print('Evaluate 2d')
    source = np.array(eval_data['human_pred_set_2d_aligned'])
    target = np.array(eval_data['human_gt_set_2d'])
    source = np.squeeze(source, axis=1)
    target = np.squeeze(target, axis=1)

    evaluation2D_perJoint(source, target, dist_th_2d)
    Accuracy_test = evaluation2D(source, target, dist_th_2d)
    print('Accuracy:', Accuracy_test)