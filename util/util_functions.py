import numpy as np
import cv2

intrinsics = {'fx': 504.1189880371094, 'fy': 504.042724609375, 'cx': 231.7421875, 'cy': 320.62640380859375}

jointColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
               [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
               [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [85, 255, 85]]

root_joint = 'torso'
depth_mean = 3
depth_std = 2
depth_max = 6
num_parts = 15


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


def draw_humans(img, humans, limbs, jointColors, visibilities=None):
    visibilities = visibilities or None
    for i, human in enumerate(humans):
        human_vis = np.array(human)
        for k, limb in enumerate(limbs):
            if visibilities is not None and visibilities[i][limb[0]] < 0.5:
                continue
            center1 = human_vis[limb[0], :2].astype(np.int)
            img = cv2.circle(img, tuple(center1), 3, [0, 0, 255], thickness=2, lineType=8, shift=0)

            if visibilities is not None and visibilities[i][limb[1]] < 0.5:
                continue
            center2 = human_vis[limb[1], :2].astype(np.int)
            img = cv2.line(img, tuple(center1), tuple(center2), jointColors[k], 2)
            img = cv2.circle(img, tuple(center2), 3, [0, 0, 255], thickness=2, lineType=8, shift=0)

    return img


def draw_humans_visibility(img, humans, limbs, jointColors, visibilities=None):
    visibilities = visibilities or None
    for i, human in enumerate(humans):
        human_vis = np.array(human)
        for k, limb in enumerate(limbs):
            if visibilities is not None and visibilities[i][limb[0]] < 0.5:
                color = [0, 0, 0]
            else:
                color = [0, 0, 255]
            center1 = human_vis[limb[0], :2].astype(np.int)
            img = cv2.circle(img, tuple(center1), 3, color, thickness=2, lineType=8, shift=0)

            if visibilities is not None and visibilities[i][limb[1]] < 0.5:
                color = [0, 0, 0]
            else:
                color = [0, 0, 255]
            center2 = human_vis[limb[1], :2].astype(np.int)
            img = cv2.line(img, tuple(center1), tuple(center2), jointColors[k], 2)
            img = cv2.circle(img, tuple(center2), 3, color, thickness=2, lineType=8, shift=0)

    return img


def homographic_transform(M, x, y):
    ones = np.ones_like(y)
    pos = np.vstack([x, y, ones])
    trans = np.matmul(M, pos)

    x_val = trans[0, :] / trans[2, :]
    y_val = trans[1, :] / trans[2, :]

    return x_val, y_val


def pos_3d_from_2d_and_depth(x_2d, y_2d, Z, cx, cy, fx, fy):
    """
        x_2d: numpy array
        y_2d: numpy array
        Z: numpy array of 3d depth
    """
    X = (x_2d - cx) / fx * Z
    Y = (y_2d - cy) / fy * Z
    return np.vstack([X, Y, Z]).T


def approx_root_orientation(hip_left_pt, hip_right_pt, neck_pt):
    """

        This function approximates the axis of coordinate frame associated with pelvis
        X-axis: vector_L2R
        Y-axis: vector_R2L X vector_L2N
        Z-axis: X-axis X y-axis

        return: a rotation matrix [X-axis, Y-axis, Z-axis]
    """

    # consider input as N X 3 matrices
    hip_left_pt = hip_left_pt.reshape([-1, 3])
    hip_right_pt = hip_right_pt.reshape([-1, 3])
    neck_pt = neck_pt.reshape([-1, 3])

    x_axis = hip_right_pt - hip_left_pt
    x_axis /= (np.linalg.norm(x_axis, axis=1).reshape([-1, 1]) + 0.000000001)
    y_axis = np.cross(-x_axis, neck_pt - hip_left_pt)
    y_axis /= (np.linalg.norm(y_axis, axis=1).reshape([-1, 1]) + 0.000000001)
    z_axis = np.cross(x_axis, y_axis)

    R = np.concatenate([x_axis.reshape([-1, 3, 1]),
                        y_axis.reshape([-1, 3, 1]),
                        z_axis.reshape([-1, 3, 1])], axis=2)

    # R = np.array([x_axis, y_axis, z_axis]).T
    return R


def inbound_visibility(keypoints_2d, input_x, input_y, visibility=None):
    """
    For a batch of humans 2d joints, return the each visility based on image range
    """

    if visibility is None:
        visibility = np.ones([keypoints_2d.shape[0], keypoints_2d.shape[1]])
    mask = np.logical_or.reduce((keypoints_2d[:, :, 0] >= input_x,
                                 keypoints_2d[:, :,  0] < 0,
                                 keypoints_2d[:, :, 1] >= input_y,
                                 keypoints_2d[:, :, 1] < 0))
    visibility[mask] = 0
    return visibility


def bbox_from_human(humans_gt_2d_batch, joint2box_margin, height, width):
    # compute ROI from 2D human
    bboxes = []
    for humans_gt_2d in humans_gt_2d_batch:
        rnd_id = np.random.randint(len(humans_gt_2d), size=1)

        human_2d = np.array(humans_gt_2d[rnd_id[0]])
        xmin = np.min(human_2d[:, 0]) - joint2box_margin
        ymin = np.min(human_2d[:, 1]) - joint2box_margin
        xmax = np.max(human_2d[:, 0]) + joint2box_margin
        ymax = np.max(human_2d[:, 1]) + joint2box_margin

        new_xmin = int(max(0, min(width, xmin)))
        new_ymin = int(max(0, min(height, ymin)))
        new_xmax = int(max(0, min(width, xmax)))
        new_ymax = int(max(0, min(height, ymax)))
        bboxes.append([new_xmin, new_ymin, new_xmax, new_ymax])

    return bboxes


def bbox_from_human_v2(humans_gt_2d_batch, margin_ratio_x, margin_ratio_y, height, width):
    # compute ROI from 2D human
    bboxes = []
    for humans_gt_2d in humans_gt_2d_batch:
        rnd_id = np.random.randint(len(humans_gt_2d), size=1)

        human_2d = np.array(humans_gt_2d[rnd_id[0]])
        xmin = np.min(human_2d[:, 0])
        xmax = np.max(human_2d[:, 0])
        ymin = np.min(human_2d[:, 1])
        ymax = np.max(human_2d[:, 1])
        xcenter = (xmin + xmax) / 2
        box_w = xmax - xmin
        ycenter = (ymin + ymax) / 2
        box_h = ymax - ymin

        new_xmin = xcenter - box_w / 2 * margin_ratio_x
        new_xmax = xcenter + box_w / 2 * margin_ratio_x
        new_ymin = ycenter - box_h / 2 * margin_ratio_y
        new_ymax = ycenter + box_h / 2 * margin_ratio_y

        new_xmin = int(max(0, min(width, new_xmin)))
        new_xmax = int(max(0, min(width, new_xmax)))
        new_ymin = int(max(0, min(height, new_ymin)))
        new_ymax = int(max(0, min(height, new_ymax)))

        bboxes.append([new_xmin, new_ymin, new_xmax, new_ymax])

    return bboxes


def projective_transform(points_3d, K, R, T):
    """
    :param points_3d: N X 3 ndarray
    :param K: 3 X 3 ndarray, intrinsic parameters
    :param R: 3 X 3 ndarray, rotation matrix
    :param T: 3 vector, translation vector
    :return: N X 2 ndarray of projected 2D points
    """
    points_3d = np.concatenate((points_3d, np.ones((points_3d.shape[0], 1)).astype(np.float32)), axis=1).T
    P = np.matmul(K, np.concatenate((R, T.reshape(3, 1)), axis=1))
    points_2d = np.matmul(P, points_3d)
    points_2d = points_2d[0:2, :] / points_2d[2, :]
    return points_2d.T


def projective_camera(points_3d, K):
    """
    :param points_3d: N X 3 ndarray
    :param K: 3 X 3 ndarray, intrinsic parameters
    :return: N X 2 ndarray of projected 2D points
    """
    points_3d = points_3d.reshape(-1, 3)
    points_2d = np.matmul(K, points_3d.T)
    points_2d = points_2d[0:2, :] / points_2d[2, :]
    return points_2d.T


def transform_3d(points_3d, R, T):
    """
    :param points_3d: N X 3 ndarray
    :param R: 3 X 3 ndarray, rotation matrix
    :param T: 3 vector, translation vector
    :return: N X 2 ndarray of projected 2D points
    """
    points_3d = np.concatenate((points_3d, np.ones((points_3d.shape[0], 1)).astype(np.float32)), axis=1).T
    P = np.concatenate((R, T.reshape(3, 1)), axis=1)
    points_3d = np.matmul(P, points_3d)
    return points_3d.T