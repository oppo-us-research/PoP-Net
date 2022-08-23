import numpy as np
import cv2


def paf_to_human_list(joint_list, person_to_joint_assoc):
    """
    Unfold the output of paf to human list
    :param joint_list: detected joints in image
    :param person_to_joint_assoc: the association of joints to each human
    :return: a list of human, each is a K x 2 array. A list indicators for visibility 0/1
    """
    humans = []
    visibility = []
    conf_vec = []
    for human in person_to_joint_assoc:
        joint_indices = human[:-2].astype(np.int)

        # assign the location of those missed parts as [-1, -1], and return a visibility vector
        joints = []
        conf = []
        for ind in joint_indices:
            if ind < 0:
                joints.append([-1, -1])
                conf.append(0)
            else:
                joints.append(joint_list[ind, :2].tolist())
                conf.append(joint_list[ind, 2].astype(np.float))
        humans.append(joints)
        visibility.append((joint_indices >= 0).astype(np.int).tolist())
        conf_vec.append(conf)

    return humans, visibility, conf_vec


def draw_humans_from_paf(img, joint_list, person_to_joint_assoc, limbs, jointColors):
    for human in person_to_joint_assoc:
        joint_indices = human[:-2].astype(np.int)

        for k, limb in enumerate(limbs):
            if joint_indices[limb[0]] < 0:
                continue
            center1 = joint_list[joint_indices[limb[0]], :2].astype(np.int)
            img = cv2.circle(img, tuple(center1), 3, [0, 0, 255], thickness=2, lineType=8, shift=0)

            if joint_indices[limb[1]] < 0:
                continue
            center2 = joint_list[joint_indices[limb[1]], :2].astype(np.int)
            img = cv2.line(img, tuple(center1), tuple(center2), jointColors[k], 2)
            img = cv2.circle(img, tuple(center2), 3, [0, 0, 255], thickness=2, lineType=8, shift=0)

    return img


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


def retrieve_offsets_direct(center, AlignField):
    """
        Read dx dy offset with weighted aggregation
        center: (x, y)
        AlignField: H X W X 2 of dx, dy field
    """

    dx = AlignField[center[1], center[0], 0] + 0.5
    dy = AlignField[center[1], center[0], 1] + 0.5

    return dx, dy


def retrieve_offsets_weighted(center, AlignField, radius=1):
    """
        Read dx dy offset with weighted aggregation
        center: (x, y)
        AlignField: H X W X 2 of dx, dy field
    """
    grid_x = AlignField.shape[1]
    grid_y = AlignField.shape[0]
    min_x = min(max(int(center[0] - radius), 0), grid_x-1)
    max_x = max(min(int(center[0] + radius), grid_x-1), 0)
    min_y = min(max(int(center[1] - radius), 0), grid_y-1)
    max_y = max(min(int(center[1] + radius), grid_y-1), 0)

    range_x = list(range(int(min_x), int(max_x)+1, 1))
    range_y = list(range(int(min_y), int(max_y)+1, 1))
    xx, yy = np.meshgrid(range_x, range_y)

    dx_map = (xx - center[0])
    dy_map = (yy - center[1])
    # w_vec = np.exp(-(dx_map**2 + dy_map**2))

    dx_vec = AlignField[yy, xx, 0] + dx_map + 0.5
    dy_vec = AlignField[yy, xx, 1] + dy_map + 0.5

    # return np.sum(dx_vec * w_vec) / np.sum(w_vec), np.sum(dy_vec * w_vec) / np.sum(w_vec)
    return np.mean(dx_vec), np.mean(dy_vec)


def retrieve_offsets_heat_weighted(center, AlignField, heatmap, radius=1):
    """
        Read dx dy offset with weighted aggregation
        center: (x, y)
        AlignField: H X W X 2 of dx, dy field
    """
    heatmap[heatmap < 0] = 0

    grid_x = AlignField.shape[1]
    grid_y = AlignField.shape[0]
    min_x = min(max(int(center[0] - radius), 0), grid_x-1)
    max_x = max(min(int(center[0] + radius), grid_x-1), 0)
    min_y = min(max(int(center[1] - radius), 0), grid_y-1)
    max_y = max(min(int(center[1] + radius), grid_y-1), 0)

    range_x = list(range(int(min_x), int(max_x)+1, 1))
    range_y = list(range(int(min_y), int(max_y)+1, 1))
    xx, yy = np.meshgrid(range_x, range_y)

    dx_map = (xx - center[0])
    dy_map = (yy - center[1])
    w_vec = heatmap[yy, xx] + 0.000000001

    dx_vec = AlignField[yy, xx, 0] + dx_map + 0.5
    dy_vec = AlignField[yy, xx, 1] + dy_map + 0.5

    return np.sum(dx_vec * w_vec) / np.sum(w_vec), np.sum(dy_vec * w_vec) / np.sum(w_vec)


def retrieve_offsets_heat_max(center, AlignField, heatmap, radius=1):
    """
        Read dx dy offset with weighted aggregation
        center: (x, y)
        AlignField: H X W X 2 of dx, dy field
    """
    heatmap[heatmap < 0] = 0

    grid_x = AlignField.shape[1]
    grid_y = AlignField.shape[0]
    min_x = min(max(int(center[0] - radius), 0), grid_x-1)
    max_x = max(min(int(center[0] + radius), grid_x-1), 0)
    min_y = min(max(int(center[1] - radius), 0), grid_y-1)
    max_y = max(min(int(center[1] + radius), grid_y-1), 0)

    range_x = list(range(int(min_x), int(max_x)+1, 1))
    range_y = list(range(int(min_y), int(max_y)+1, 1))
    xx, yy = np.meshgrid(range_x, range_y)

    dx_map = (xx - center[0])
    dy_map = (yy - center[1])
    w_vec = heatmap[yy, xx]

    dx_vec = AlignField[yy, xx, 0] + dx_map + 0.5
    dy_vec = AlignField[yy, xx, 1] + dy_map + 0.5

    dx_out = dx_vec.ravel()[np.argmax(w_vec)]
    dy_out = dy_vec.ravel()[np.argmax(w_vec)]

    return dx_out, dy_out


def retrieve_offsets_nn(center, AlignField, radius=1):
    """
        Read dx dy offset with weighted aggregation
        center: (x, y)
        AlignField: H X W X 2 of dx, dy field
    """
    grid_x = AlignField.shape[1]
    grid_y = AlignField.shape[0]
    min_x = max(int(int(center[0] - radius)), 0)
    max_x = min(int(int(center[0] + radius)), grid_x-1)
    min_y = max(int(int(center[1] - radius)), 0)
    max_y = min(int(int(center[1] + radius)), grid_y-1)

    range_x = list(range(int(min_x), int(max_x)+1, 1))
    range_y = list(range(int(min_y), int(max_y)+1, 1))
    xx, yy = np.meshgrid(range_x, range_y)

    dx_map = (xx - center[0]) + 0.5
    dy_map = (yy - center[1]) + 0.5

    dx_vec = AlignField[yy, xx, 0]
    dy_vec = AlignField[yy, xx, 1]

    dx_vec = dx_vec.flatten()
    dy_vec = dy_vec.flatten()
    dx_map = dx_map.flatten()
    dy_map = dy_map.flatten()
    min_ind = np.argmin(dx_vec**2 + dy_vec**2)
    return dx_vec[min_ind] + dx_map[min_ind], dy_vec[min_ind] + dy_map[min_ind]


def retrieve_depth_weighted(center, depthmap, radius=1):
    """
        Read dx dy offset with weighted aggregation
        center: (x, y)
    """
    grid_x = depthmap.shape[1]
    grid_y = depthmap.shape[0]
    min_x = min(max(int(center[0] - radius), 0), grid_x-1)
    max_x = max(min(int(center[0] + radius), grid_x-1), 0)
    min_y = min(max(int(center[1] - radius), 0), grid_y-1)
    max_y = max(min(int(center[1] + radius), grid_y-1), 0)

    range_x = list(range(int(min_x), int(max_x)+1, 1))
    range_y = list(range(int(min_y), int(max_y)+1, 1))
    xx, yy = np.meshgrid(range_x, range_y)

    depth_vec = depthmap[yy, xx]

    return np.mean(depth_vec)


def retrieve_depth_heat_weighted(center, depthmap, heatmap, radius=1):
    """
        Read dx dy offset with weighted aggregation
        center: (x, y)
    """
    heatmap[heatmap < 0] = 0

    grid_x = depthmap.shape[1]
    grid_y = depthmap.shape[0]
    min_x = min(max(int(center[0] - radius), 0), grid_x-1)
    max_x = max(min(int(center[0] + radius), grid_x-1), 0)
    min_y = min(max(int(center[1] - radius), 0), grid_y-1)
    max_y = max(min(int(center[1] + radius), grid_y-1), 0)

    range_x = list(range(int(min_x), int(max_x)+1, 1))
    range_y = list(range(int(min_y), int(max_y)+1, 1))
    xx, yy = np.meshgrid(range_x, range_y)

    w_vec = heatmap[yy, xx] + 0.000000001
    depth_vec = depthmap[yy, xx]

    return np.sum(depth_vec * w_vec) / np.sum(w_vec)


def retrieve_depth_heat_max(center, depthmap, heatmap, radius=1):
    """
        Read dx dy offset with weighted aggregation
        center: (x, y)
    """
    heatmap[heatmap < 0] = 0

    grid_x = depthmap.shape[1]
    grid_y = depthmap.shape[0]
    min_x = min(max(int(center[0] - radius), 0), grid_x-1)
    max_x = max(min(int(center[0] + radius), grid_x-1), 0)
    min_y = min(max(int(center[1] - radius), 0), grid_y-1)
    max_y = max(min(int(center[1] + radius), grid_y-1), 0)

    range_x = list(range(int(min_x), int(max_x)+1, 1))
    range_y = list(range(int(min_y), int(max_y)+1, 1))
    xx, yy = np.meshgrid(range_x, range_y)

    w_vec = heatmap[yy, xx]
    depth_vec = depthmap[yy, xx].ravel()

    depth_out = depth_vec[np.argmax(w_vec)]
    return depth_out


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


def superimpose_colormap_on_img(img, colormap, color_weight=2):
    """
    A function to superimpose a colormap on img

    Args:
        img: H x W x 3 matrix
        colormap: H x W x 3 matrix
        color_weight: the color weight for weighted avg of img and color

    Returns:

    """
    h, w, _ = img.shape
    colormap = cv2.resize(colormap, (w, h))
    img_out = img + colormap.astype(np.float) * color_weight
    img_out = (img_out / (color_weight + 1)).astype(np.uint8)

    return img_out


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