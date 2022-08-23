import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure
import torch
from torch.autograd import Variable


# perform NMS to output bounding box results and associated skeleton, all scales are kept the save as input logits
def parse_prior_pose(posemaps, anchors, num_joints, w_out, h_out, depth_mean, depth_std, conf_threshold=0.35, nms_threshold=0.5, pred_vis=False, vis_margin=0):
    """
    posemaps: needs to be in tensor
    others: numpy
    """

    num_anchors = len(anchors)
    anchors = torch.Tensor(anchors)
    if isinstance(posemaps, Variable):
        posemaps = posemaps.data

    if posemaps.dim() == 3:
        posemaps.unsqueeze_(0)

    batch = posemaps.size(0)
    h = posemaps.size(2)
    w = posemaps.size(3)

    # Compute xc,yc, w,h, box_score on Tensor
    lin_x = torch.linspace(0, w - 1, w).repeat(h, 1).view(h * w)
    lin_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().contiguous().view(h * w)
    anchor_w = anchors[:, 0].contiguous().view(1, num_anchors, 1)
    anchor_h = anchors[:, 1].contiguous().view(1, num_anchors, 1)
    if torch.cuda.is_available():
        lin_x = lin_x.cuda()
        lin_y = lin_y.cuda()
        anchor_w = anchor_w.cuda()
        anchor_h = anchor_h.cuda()

    posemaps = posemaps.view(batch, num_anchors, -1, h * w)

    # convert network output [dx dy, w/anchor_w, h/anchor_h] to absolute layer coord [cx, cy, w, h]
    posemaps[:, :, 0, :].add_(lin_x).div_(w)
    posemaps[:, :, 1, :].add_(lin_y).div_(h)
    posemaps[:, :, 2, :].mul_(anchor_w).div_(w)
    posemaps[:, :, 3, :].mul_(anchor_h).div_(h)

    # un-normalize joint locations
    anchor_w = anchor_w.view(1, num_anchors, 1, 1)
    anchor_h = anchor_h.view(1, num_anchors, 1, 1)
    posemaps[:, :, 5:5 + num_joints, :].mul_(anchor_w / 2.0).add_(lin_x).div_(w)
    posemaps[:, :, 5 + num_joints:5 + 2 * num_joints, :].mul_(anchor_h / 2.0).add_(lin_y).div_(h)
    posemaps[:, :, 5 + 2 * num_joints:5 + 3 * num_joints, :].mul_(depth_std).add_(depth_mean)

    score_thresh = posemaps[:, :, 4, :] > conf_threshold
    score_thresh_flat = score_thresh.view(-1)
    if score_thresh.sum() == 0:
        predicted_boxes = []
        for i in range(batch):
            predicted_boxes.append(torch.Tensor([]))
    else:
        detections = posemaps.transpose(2, 3)
        if pred_vis:
            detections = detections[score_thresh[..., None].expand_as(detections)].view(-1, 5+4*num_joints)
        else:
            detections = detections[score_thresh[..., None].expand_as(detections)].view(-1, 5+3*num_joints)

        max_det_per_batch = num_anchors * h * w
        slices = [slice(max_det_per_batch * i, max_det_per_batch * (i + 1)) for i in range(batch)]
        det_per_batch = torch.IntTensor([score_thresh_flat[s].int().sum() for s in slices])
        split_idx = torch.cumsum(det_per_batch, dim=0)

        # Group detections per image of batch
        predicted_boxes = []
        start = 0
        for end in split_idx:
            predicted_boxes.append(detections[start: end])
            start = end

    selected_boxes = []
    for boxes in predicted_boxes:
        if boxes.numel() == 0:
            selected_boxes.append(boxes)
            continue

        a = boxes[:, :2]
        b = boxes[:, 2:4]
        # confirmed network output [dx dy, w/anchor_w, h/anchor_h]
        bboxes = torch.cat([a - b / 2, a + b / 2], 1)
        scores = boxes[:, 4]

        # Sort coordinates by descending score
        scores, order = scores.sort(0, descending=True)
        x1, y1, x2, y2 = bboxes[order].split(1, 1)

        # Compute dx and dy between each pair of boxes (these mat contain every pair twice...)
        dx = (x2.min(x2.t()) - x1.max(x1.t())).clamp(min=0)
        dy = (y2.min(y2.t()) - y1.max(y1.t())).clamp(min=0)

        # Compute iou
        intersections = dx * dy
        areas = (x2 - x1) * (y2 - y1)
        unions = (areas + areas.t()) - intersections
        ious = intersections / unions

        # Filter based on iou (and class)
        conflicting = (ious > nms_threshold).type(torch.IntTensor).triu(1)

        keep = conflicting.sum(0).int()
        keep = keep.cpu()
        conflicting = conflicting.cpu()

        keep_len = len(keep) - 1
        for i in range(1, keep_len):
            if keep[i] > 0:
                keep -= conflicting[i].int()
        if torch.cuda.is_available():
            keep = keep.cuda()

        keep = (keep == 0)
        if pred_vis:
            selected_boxes.append(boxes[order][keep[:, None].expand_as(boxes)].view(-1, 5 + 4*num_joints).contiguous())
        else:
            selected_boxes.append(boxes[order][keep[:, None].expand_as(boxes)].view(-1, 5 + 3*num_joints).contiguous())

    humans_prior = []
    visibility = []
    bboxes_out = []
    for boxes in selected_boxes:
        if boxes.shape[0] == 0:
            bboxes_out.append([])
            humans_prior.append([])
            visibility.append([])
        else:
            boxes[:, 0] *= w_out
            boxes[:, 2] *= w_out
            boxes[:, 1] *= h_out
            boxes[:, 3] *= h_out
            # convert center to the tof-left corner, the last two stay as w, h
            boxes[:, 0] -= boxes[:, 2] / 2
            boxes[:, 1] -= boxes[:, 3] / 2
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
            boxes[:, 5:5 + num_joints] *= w_out
            boxes[:, 5 + num_joints: 5 + 2*num_joints] *= h_out
            if torch.cuda.is_available():
                boxes = boxes.cpu().data
            bboxes_out.append([box[:5].numpy() for box in boxes])
            humans_b = []
            visibility_b = []
            for box in boxes:
                human = box[5:5 + 3*num_joints].numpy().reshape(3, -1).T
                humans_b.append(human)
                if pred_vis:
                    visibility_b.append(
                        np.logical_and(np.logical_and(human[:, 0] >= 0 + vis_margin, human[:, 0] <= w_out - 1 - vis_margin),
                                       np.logical_and(human[:, 1] >= 0 + vis_margin, human[:, 1] <= h_out - 1 - vis_margin)) *
                            box[5 + 3*num_joints:].numpy())
                    # visibility_b.append(box[5 + 3*num_joints:].numpy())
                else:
                    visibility_b.append(np.logical_and(np.logical_and(human[:, 0] >= 0 + vis_margin, human[:, 0] <= w_out - 1 - vis_margin),
                                                       np.logical_and(human[:, 1] >= 0 + vis_margin, human[:, 1] <= h_out - 1 - vis_margin)))
            humans_prior.append(humans_b)
            visibility.append(visibility_b)

        if len(bboxes_out[-1]) != len(humans_prior[-1]):
            print('debug')

    return bboxes_out, humans_prior, visibility


# perform NMS to output bounding box results and associated skeleton, all scales are kept the save as input logits
def parse_prior_pose_rgb(posemaps, anchors, num_joints, w_out, h_out, conf_threshold=0.35, nms_threshold=0.5, vis_margin=0):
    """
    posemaps: needs to be in tensor
    others: numpy
    """

    num_anchors = len(anchors)
    anchors = torch.Tensor(anchors)
    if isinstance(posemaps, Variable):
        posemaps = posemaps.data

    if posemaps.dim() == 3:
        posemaps.unsqueeze_(0)

    batch = posemaps.size(0)
    h = posemaps.size(2)
    w = posemaps.size(3)

    # Compute xc,yc, w,h, box_score on Tensor
    lin_x = torch.linspace(0, w - 1, w).repeat(h, 1).view(h * w)
    lin_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().contiguous().view(h * w)
    anchor_w = anchors[:, 0].contiguous().view(1, num_anchors, 1)
    anchor_h = anchors[:, 1].contiguous().view(1, num_anchors, 1)
    if torch.cuda.is_available():
        lin_x = lin_x.cuda()
        lin_y = lin_y.cuda()
        anchor_w = anchor_w.cuda()
        anchor_h = anchor_h.cuda()

    posemaps = posemaps.view(batch, num_anchors, -1, h * w)

    # convert network output [dx dy, w/anchor_w, h/anchor_h] to absolute layer coord [cx, cy, w, h]
    posemaps[:, :, 0, :].add_(lin_x).div_(w)
    posemaps[:, :, 1, :].add_(lin_y).div_(h)
    posemaps[:, :, 2, :].mul_(anchor_w).div_(w)
    posemaps[:, :, 3, :].mul_(anchor_h).div_(h)

    # un-normalize joint locations
    anchor_w = anchor_w.view(1, num_anchors, 1, 1)
    anchor_h = anchor_h.view(1, num_anchors, 1, 1)
    posemaps[:, :, 5:5 + num_joints, :].mul_(anchor_w / 2.0).add_(lin_x).div_(w)
    posemaps[:, :, 5 + num_joints:5 + 2 * num_joints, :].mul_(anchor_h / 2.0).add_(lin_y).div_(h)

    score_thresh = posemaps[:, :, 4, :] > conf_threshold
    score_thresh_flat = score_thresh.view(-1)
    if score_thresh.sum() == 0:
        predicted_boxes = []
        for i in range(batch):
            predicted_boxes.append(torch.Tensor([]))
    else:
        detections = posemaps.transpose(2, 3)
        detections = detections[score_thresh[..., None].expand_as(detections)].view(-1, 5+3*num_joints)

        max_det_per_batch = num_anchors * h * w
        slices = [slice(max_det_per_batch * i, max_det_per_batch * (i + 1)) for i in range(batch)]
        det_per_batch = torch.IntTensor([score_thresh_flat[s].int().sum() for s in slices])
        split_idx = torch.cumsum(det_per_batch, dim=0)

        # Group detections per image of batch
        predicted_boxes = []
        start = 0
        for end in split_idx:
            predicted_boxes.append(detections[start: end])
            start = end

    selected_boxes = []
    for boxes in predicted_boxes:
        if boxes.numel() == 0:
            selected_boxes.append(boxes)
            continue

        a = boxes[:, :2]
        b = boxes[:, 2:4]
        # confirmed network output [dx dy, w/anchor_w, h/anchor_h]
        bboxes = torch.cat([a - b / 2, a + b / 2], 1)
        scores = boxes[:, 4]

        # Sort coordinates by descending score
        scores, order = scores.sort(0, descending=True)
        x1, y1, x2, y2 = bboxes[order].split(1, 1)

        # Compute dx and dy between each pair of boxes (these mat contain every pair twice...)
        dx = (x2.min(x2.t()) - x1.max(x1.t())).clamp(min=0)
        dy = (y2.min(y2.t()) - y1.max(y1.t())).clamp(min=0)

        # Compute iou
        intersections = dx * dy
        areas = (x2 - x1) * (y2 - y1)
        unions = (areas + areas.t()) - intersections
        ious = intersections / unions

        # Filter based on iou (and class)
        conflicting = (ious > nms_threshold).type(torch.IntTensor).triu(1)

        keep = conflicting.sum(0).int()
        keep = keep.cpu()
        conflicting = conflicting.cpu()

        keep_len = len(keep) - 1
        for i in range(1, keep_len):
            if keep[i] > 0:
                keep -= conflicting[i].int()
        if torch.cuda.is_available():
            keep = keep.cuda()

        keep = (keep == 0)
        selected_boxes.append(boxes[order][keep[:, None].expand_as(boxes)].view(-1, 5 + 3*num_joints).contiguous())

    humans_prior = []
    visibility = []
    bboxes_out = []
    for boxes in selected_boxes:
        if boxes.shape[0] == 0:
            bboxes_out.append([])
            humans_prior.append([])
            visibility.append([])
        else:
            boxes[:, 0] *= w_out
            boxes[:, 2] *= w_out
            boxes[:, 1] *= h_out
            boxes[:, 3] *= h_out
            # convert center to the tof-left corner, the last two stay as w, h
            boxes[:, 0] -= boxes[:, 2] / 2
            boxes[:, 1] -= boxes[:, 3] / 2
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
            boxes[:, 5:5 + num_joints] *= w_out
            boxes[:, 5 + num_joints: 5 + 2*num_joints] *= h_out
            if torch.cuda.is_available():
                boxes = boxes.cpu().data
            bboxes_out.append([box[:5].numpy() for box in boxes])
            humans_b = []
            visibility_b = []
            for box in boxes:
                human = box[5:5 + 2*num_joints].numpy().reshape(2, -1).T
                humans_b.append(human)
                visibility_b.append(
                    np.logical_and(np.logical_and(human[:, 0] >= 0 + vis_margin, human[:, 0] <= w_out - 1 - vis_margin),
                                   np.logical_and(human[:, 1] >= 0 + vis_margin, human[:, 1] <= h_out - 1 - vis_margin)) *
                        box[5 + 2*num_joints:].numpy())
            humans_prior.append(humans_b)
            visibility.append(visibility_b)

        if len(bboxes_out[-1]) != len(humans_prior[-1]):
            print('debug')

    return bboxes_out, humans_prior, visibility


def universe_align_map(heatmaps, alignmaps, num_joints, align_radius, ht_thresh=0.5, top_n=None, visibility=None):
    """
        Unifies close-range reifned distance field and long-range distance field.
        Compute long-range distance map from NMS of heatmaps.
        Combine distance maps from two ranges based on the radius in close-range map construction.

        ht_thesh: needs to be high. Only overwrite with long-range alignmap when local detection is very sure

        TODO: serious problem here when some parts heatmap peaks are missing. THe missed one with be draged to the other confident

        TODO: visibility now assumes single target
    """

    h = heatmaps.shape[0]
    w = heatmaps.shape[1]
    range_x = list(range(0, w, 1))
    range_y = list(range(0, h, 1))
    xx, yy = np.meshgrid(range_x, range_y)

    uni_alignmaps = np.copy(alignmaps)
    for j in range(num_joints):
        map_orig = heatmaps[:, :, j]
        peak_coords = find_peaks(ht_thresh, map_orig, top_n)
        if len(peak_coords) == 0 or (visibility is not None and visibility[j] < 0.5):
            continue
        dx_maps = []
        dy_maps = []
        dist_maps = []
        fg_mask = np.zeros([h, w]).astype(np.int)
        # consider multi-peaks per joint for multi-person
        for i, peak in enumerate(peak_coords):
            dx_map = peak[0] - np.copy(xx)
            dy_map = peak[1] - np.copy(yy)
            dx_maps.append(dx_map)
            dy_maps.append(dy_map)
            dist_maps.append(dx_map**2 + dy_map**2)

            # update fg masks
            x_min, y_min = np.maximum(0, peak - align_radius)
            x_max, y_max = np.minimum(
                np.array(map_orig.T.shape) - 1, peak + align_radius)
            fg_mask[y_min:y_max+1, x_min:x_max+1] = 1

        # fuse distance maps
        dx_maps = np.array(dx_maps)
        dy_maps = np.array(dy_maps)
        dist_maps = np.array(dist_maps)
        min_ind_map = np.argmin(dist_maps, axis=0)
        far_range_alignmap_x = dx_maps[min_ind_map, yy, xx]
        far_range_alignmap_y = dy_maps[min_ind_map, yy, xx]
        yy_fg, xx_fg = np.where(fg_mask == 0)
        uni_alignmaps[yy_fg, xx_fg, 2*j] = far_range_alignmap_x[fg_mask == 0]
        uni_alignmaps[yy_fg, xx_fg, 2*j + 1] = far_range_alignmap_y[fg_mask == 0]

    return uni_alignmaps


def find_peaks(param, img, top_n=None):
    """
    Given a (grayscale) image, find local maxima whose value is above a given
    threshold (param['thre1'])
    :param img: Input image (2d array) where we want to find peaks
    : top_n: impose the number of
    :return: 2d np.array containing the [x,y] coordinates of each peak found
    in the image
    """

    peaks_binary = (maximum_filter(img, footprint=generate_binary_structure(
        2, 1)) == img) * (img > param)
    # Note reverse ([::-1]): we return [[x y], [x y]...] instead of [[y x], [y
    # x]...]
    peaks = np.array(np.nonzero(peaks_binary)[::-1]).T
    if top_n and len(peaks) > top_n:
        confs = img[peaks[:, 1], peaks[:, 0]]
        sort_idx = np.argsort(confs)
        sort_idx = sort_idx[::-1]
        return peaks[sort_idx[:top_n]]
    return peaks

