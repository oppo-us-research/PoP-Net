from collections import OrderedDict
import math
import torch
import torch.nn as nn


def WeightedMSELoss(input, target, weights):
    out = (input - target)**2
    out = out * weights
    loss = out.mean()
    return loss


def WeightedMSELossV2(input, target, weights):
    out = (input - target)**2
    out = out * weights
    loss = out.sum() / (weights.sum() + 0.000001)
    return loss


def WeightedSSELoss(input, target, weights):
    out = (input - target)**2
    out = out * weights
    loss = out.sum()
    return loss


def rtpose_light3d_loss(saved_for_loss, heat_gt, vec_temp, posedepth_temp, num_stages, names):

    # names = build_names(num_stages)
    saved_for_log = OrderedDict()
    criterion = nn.MSELoss(reduction='mean').cuda()
    total_loss = 0

    for j in range(num_stages):
        pred1 = saved_for_loss[3 * j]
        pred2 = saved_for_loss[3 * j + 1]
        pred3 = saved_for_loss[3 * j + 2]

        # Compute losses
        loss1 = criterion(pred1, vec_temp)
        loss2 = criterion(pred2, heat_gt)
        loss3 = criterion(pred3, posedepth_temp)

        total_loss += loss1
        total_loss += loss2
        total_loss += loss3
        # print(total_loss)

        # Get value from Variable and save for log
        saved_for_log[names[3 * j]] = loss1.item()
        saved_for_log[names[3 * j + 1]] = loss2.item()
        saved_for_log[names[3 * j + 2]] = loss3.item()

    saved_for_log['max_ht'] = torch.max(
        saved_for_loss[-3].data[:, 0:-1, :, :]).item()
    saved_for_log['min_ht'] = torch.min(
        saved_for_loss[-3].data[:, 0:-1, :, :]).item()
    saved_for_log['max_paf'] = torch.max(saved_for_loss[-2].data).item()
    saved_for_log['min_paf'] = torch.min(saved_for_loss[-2].data).item()

    return total_loss, saved_for_log


def rtpose_light3d_loss_fgweight(saved_for_loss, heat_gt, vec_temp, posedepth_temp, fg_mask, num_stages, names):
    """
        this version uses foreground weighted losses, hard and soft for different maps
    """
    # names = build_names(num_stages)
    saved_for_log = OrderedDict()
    criterion = nn.MSELoss(reduction='mean').cuda()
    total_loss = 0
    # Hard-coded weight for fg and bg
    weight = torch.ones_like(fg_mask) * 0.1
    weight = weight + fg_mask * 0.9

    for j in range(num_stages):
        pred1 = saved_for_loss[3 * j]
        pred2 = saved_for_loss[3 * j + 1]
        pred3 = saved_for_loss[3 * j + 2]

        # Compute losses
        loss1 = criterion(pred1, vec_temp)
        loss2 = criterion(pred2, heat_gt)
        loss3 = WeightedMSELoss(pred3, posedepth_temp, weight)

        total_loss += loss1
        total_loss += loss2
        total_loss += loss3
        # print(total_loss)

        # Get value from Variable and save for log
        saved_for_log[names[3 * j]] = loss1.item()
        saved_for_log[names[3 * j + 1]] = loss2.item()
        saved_for_log[names[3 * j + 2]] = loss3.item()

    saved_for_log['max_ht'] = torch.max(
        saved_for_loss[-2].data[:, 0:-1, :, :]).item()
    saved_for_log['min_ht'] = torch.min(
        saved_for_loss[-2].data[:, 0:-1, :, :]).item()
    saved_for_log['max_paf'] = torch.max(saved_for_loss[-3].data).item()
    saved_for_log['min_paf'] = torch.min(saved_for_loss[-3].data).item()
    saved_for_log['max_z'] = torch.max(saved_for_loss[-1].data).item()
    saved_for_log['min_z'] = torch.min(saved_for_loss[-1].data).item()

    return total_loss, saved_for_log


def pop_net_loss_fgweight_roi(saved_for_loss, heat_gt, zmap_gt, fg_mask_z, alignmap_temp, fg_mask_align, obj_label, num_stages, num_joints, names, prior_subnet_only=False, pred_vis=False):
    """
        this version uses foreground weighted losses, hard and soft for different maps
        this loss focues with the roi version model
    """
    # names = build_names(num_stages)
    saved_for_log = OrderedDict()
    criterion1 = nn.MSELoss(reduction='mean').cuda()
    total_loss = 0
    # Hard-coded weight for fg and bg, scale needs to be tuned
    weight_z = torch.ones_like(fg_mask_z) * 0.1
    weight_z = weight_z + fg_mask_z * 0.9
    weight_align = fg_mask_align
    b, c, h, w = weight_z.size()
    # pay attention to the weight for the extended dimension
    weight_ht = torch.cat([weight_z, torch.ones(b, 1, h, w).cuda()], 1)
    obj_label = obj_label.view(b, -1)

    for j in range(num_stages):
        pred1 = saved_for_loss[3 * j]  # heatmap
        pred2 = saved_for_loss[3 * j + 1]  # zmap
        pred3 = saved_for_loss[3 * j + 2]  # alignmap

        # Compute losses
        # loss1 = criterion1(pred1, heat_gt)
        loss1 = WeightedMSELoss(pred1, heat_gt, weight_ht)
        loss2 = WeightedMSELoss(pred2, zmap_gt, weight_z)  # zmap loss uses high fg weight
        loss3 = WeightedMSELoss(pred3, alignmap_temp, weight_align)  # align loss focuses strictly on fg

        if not prior_subnet_only:
            total_loss += loss1
            total_loss += loss2
            total_loss += loss3

        # Get value from Variable and save for log
        saved_for_log[names[3 * j]] = loss1.item()
        saved_for_log[names[3 * j + 1]] = loss2.item()
        saved_for_log[names[3 * j + 2]] = loss3.item()

    # pose prior loss
    if pred_vis:
        loss_prior = criterion1(saved_for_loss[-1], obj_label) * 4 * num_joints
    else:
        loss_prior = criterion1(saved_for_loss[-1], obj_label) * 3 * num_joints
    total_loss += loss_prior
    saved_for_log['loss_prior'] = loss_prior.data.item()

    saved_for_log['max_ht'] = torch.max(
        saved_for_loss[-4].data[:, 0:-1, :, :]).item()
    saved_for_log['min_ht'] = torch.min(
        saved_for_loss[-4].data[:, 0:-1, :, :]).item()
    saved_for_log['max_z'] = torch.max(saved_for_loss[-3].data).item()
    saved_for_log['min_z'] = torch.min(saved_for_loss[-3].data).item()
    saved_for_log['max_alignf'] = torch.max((saved_for_loss[-2] * fg_mask_align).data).item()
    saved_for_log['min_alignf'] = torch.min((saved_for_loss[-2] * fg_mask_align).data).item()

    return total_loss, saved_for_log


def pop_net_loss_fgweight_roi_poseweight(saved_for_loss, heat_gt, zmap_gt, fg_mask_z, alignmap_temp, fg_mask_align, obj_label, num_stages, num_joints, names, pose_weights, prior_subnet_only=False, pred_vis=False):
    """
        this version uses foreground weighted losses, hard and soft for different maps
        this loss focues with the roi version model
        this version use pose rarity to re-weight the prior loss
    """
    # names = build_names(num_stages)
    saved_for_log = OrderedDict()
    criterion1 = nn.MSELoss(reduction='mean').cuda()
    total_loss = 0
    # Hard-coded weight for fg and bg, scale needs to be tuned
    weight_z = torch.ones_like(fg_mask_z) * 0.1
    weight_z = weight_z + fg_mask_z * 0.9
    weight_align = fg_mask_align
    b, c, h, w = weight_z.size()
    # pay attention to the weight for the extended dimension
    weight_ht = torch.cat([weight_z, torch.ones(b, 1, h, w).cuda()], 1)
    tweight = torch.zeros(b)
    for bi in range(b):
        tweight[bi] = pose_weights[bi].mean()
    tweight = tweight.view(b, 1, 1, 1).cuda()

    for j in range(num_stages):
        pred1 = saved_for_loss[3 * j]  # heatmap
        pred2 = saved_for_loss[3 * j + 1]  # zmap
        pred3 = saved_for_loss[3 * j + 2]  # alignmap

        # Compute losses
        # loss1 = criterion1(pred1, heat_gt)
        loss1 = WeightedMSELoss(pred1, heat_gt, weight_ht)
        loss2 = WeightedMSELoss(pred2, zmap_gt, weight_z)  # zmap loss uses high fg weight
        loss3 = WeightedMSELoss(pred3, alignmap_temp, weight_align)  # align loss focuses strictly on fg

        # loss1 = WeightedMSELoss(pred1 * weight_ht, heat_gt * weight_ht, tweight)
        # loss2 = WeightedMSELoss(pred2 * weight_z, zmap_gt * weight_z, tweight)  # zmap loss uses high fg weight
        # loss3 = WeightedMSELoss(pred3 * weight_align, alignmap_temp * weight_align,
        #                         tweight)  # align loss focuses strictly on fg

        if not prior_subnet_only:
            total_loss += loss1
            total_loss += loss2
            total_loss += loss3

        # Get value from Variable and save for log
        saved_for_log[names[3 * j]] = loss1.item()
        saved_for_log[names[3 * j + 1]] = loss2.item()
        saved_for_log[names[3 * j + 2]] = loss3.item()

    # pose prior loss
    # loss_prior = criterion1(saved_for_loss[-1], obj_label.view(b, -1))
    if pred_vis:
        loss_prior = WeightedMSELoss(saved_for_loss[-1], obj_label.view(b, -1), tweight.view(b, -1)) * 4 * num_joints
    else:
        loss_prior = WeightedMSELoss(saved_for_loss[-1], obj_label.view(b, -1), tweight.view(b, -1)) * 3 * num_joints
    total_loss += loss_prior
    saved_for_log['loss_prior'] = loss_prior.data.item()

    saved_for_log['max_ht'] = torch.max(
        saved_for_loss[-4].data[:, 0:-1, :, :]).item()
    saved_for_log['min_ht'] = torch.min(
        saved_for_loss[-4].data[:, 0:-1, :, :]).item()
    saved_for_log['max_z'] = torch.max(saved_for_loss[-3].data).item()
    saved_for_log['min_z'] = torch.min(saved_for_loss[-3].data).item()
    saved_for_log['max_alignf'] = torch.max((saved_for_loss[-2] * fg_mask_align).data).item()
    saved_for_log['min_alignf'] = torch.min((saved_for_loss[-2] * fg_mask_align).data).item()

    return total_loss, saved_for_log


def pop_net_loss_fgweight(saved_for_loss, heat_gt, zmap_gt, fg_mask_z, alignmap_gt, fg_mask_align, prior_map_gt, prior_mask_conf, prior_mask_coord, num_stages, names, num_joints, num_anchors, prior_subnet_only=False, pred_vis=False):

    # names = build_names(num_stages)
    saved_for_log = OrderedDict()
    criterion1 = nn.MSELoss(reduction='mean').cuda()
    total_loss = 0
    # TODO: Hard-coded weight for fg and bg, scale needs to be tuned
    weight_z = torch.ones_like(fg_mask_z) * 0.1
    weight_z = weight_z + fg_mask_z * 0.9
    weight_align = fg_mask_align
    b, c, h, w = weight_z.size()
    # pay attention to the weight for the extended dimension
    weight_ht = torch.cat([weight_z, torch.ones(b, 1, h, w).cuda()], 1)

    for j in range(num_stages):
        pred1 = saved_for_loss[3 * j]  # heatmap
        pred2 = saved_for_loss[3 * j + 1]  # zmap
        pred3 = saved_for_loss[3 * j + 2]  # alignmap

        # Compute losses
        # loss1 = criterion1(pred1, heat_gt)
        loss1 = WeightedMSELoss(pred1, heat_gt, weight_ht)
        loss2 = WeightedMSELoss(pred2, zmap_gt, weight_z)  # zmap loss uses high fg weight
        loss3 = WeightedMSELoss(pred3, alignmap_gt, weight_align)  # align loss focuses strictly on fg

        if not prior_subnet_only:
            total_loss += loss1
            total_loss += loss2
            total_loss += loss3

        # Get value from Variable and save for log
        saved_for_log[names[3 * j]] = loss1.item()
        saved_for_log[names[3 * j + 1]] = loss2.item()
        saved_for_log[names[3 * j + 2]] = loss3.item()

    # pose prior loss
    _, _, h_prior, w_prior = saved_for_loss[-1].size()
    prior_map_pred = saved_for_loss[-1].view(b, num_anchors, -1, h_prior*w_prior).permute(0, 3, 1, 2).contiguous()
    prior_map_gt = prior_map_gt.view(b, num_anchors, -1, h_prior*w_prior).permute(0, 3, 1, 2).contiguous()
    prior_mask_coord = prior_mask_coord.view(b, num_anchors, h_prior*w_prior, 1).permute(0, 2, 1, 3).contiguous()
    prior_mask_conf = prior_mask_conf.view(b, num_anchors, h_prior*w_prior).permute(0, 2, 1).contiguous()

    coords_pred = prior_map_pred[:, :, :, :4]
    conf_pred = prior_map_pred[:, :, :, 4]
    joints_pred = prior_map_pred[:, :, :, 5:]
    coords_gt = prior_map_gt[:, :, :, :4]
    conf_gt = prior_map_gt[:, :, :, 4]
    joints_gt = prior_map_gt[:, :, :, 5:]

    loss_coord = WeightedMSELoss(coords_pred, coords_gt, prior_mask_coord) * 4
    loss_obj = WeightedMSELoss(conf_pred, conf_gt, prior_mask_conf)
    if pred_vis:
        loss_selfpose = WeightedMSELoss(joints_pred, joints_gt, prior_mask_coord) * 4 * num_joints
    else:
        loss_selfpose = WeightedMSELoss(joints_pred, joints_gt, prior_mask_coord) * 3 * num_joints
    loss_prior = loss_coord + loss_obj + loss_selfpose

    total_loss += loss_prior
    saved_for_log['loss_prior'] = loss_prior.data.item()
    saved_for_log['loss_bbox'] = loss_coord.data.item()
    saved_for_log['loss_obj'] = loss_obj.data.item()
    saved_for_log['loss_selfpose'] = loss_selfpose.data.item()

    saved_for_log['max_ht'] = torch.max(
        saved_for_loss[-4].data[:, 0:-1, :, :]).item()
    saved_for_log['min_ht'] = torch.min(
        saved_for_loss[-4].data[:, 0:-1, :, :]).item()
    saved_for_log['max_z'] = torch.max(saved_for_loss[-3].data).item()
    saved_for_log['min_z'] = torch.min(saved_for_loss[-3].data).item()
    saved_for_log['max_alignf'] = torch.max((saved_for_loss[-2] * fg_mask_align).data).item()
    saved_for_log['min_alignf'] = torch.min((saved_for_loss[-2] * fg_mask_align).data).item()

    return total_loss, saved_for_log


def pop_net_loss_fgweight_poseweight(saved_for_loss, heat_gt, zmap_gt, fg_mask_z, alignmap_temp, fg_mask_align, prior_map_gt, prior_mask_conf, prior_mask_coord, prior_weight_map, num_stages, names, num_joints, num_anchors, prior_subnet_only=False, pred_vis=False):
    """
    this version use pose rarity to re-weight the prior loss
    """

    # all the labels are casted to [-1, 1] range for better learning
    # heat_gt = (heat_gt - 0.5) * 2

    # names = build_names(num_stages)
    saved_for_log = OrderedDict()
    criterion1 = nn.MSELoss(reduction='mean').cuda()
    total_loss = 0
    # TODO: Hard-coded weight for fg and bg, scale needs to be tuned
    weight_z = torch.ones_like(fg_mask_z) * 0.1
    weight_z = weight_z + fg_mask_z * 0.9
    weight_align = fg_mask_align
    b, c, h, w = weight_z.size()
    # pay attention to the weight for the extended dimension
    weight_ht = torch.cat([weight_z, torch.ones(b, 1, h, w).cuda()], 1)

    for j in range(num_stages):
        # saved_for_loss[3 * j] = saved_for_loss[3 * j].sigmoid()
        pred1 = saved_for_loss[3 * j]  # heatmap
        pred2 = saved_for_loss[3 * j + 1]  # zmap
        pred3 = saved_for_loss[3 * j + 2]  # alignmap

        # Compute losses
        # loss1 = criterion1(pred1, heat_gt)
        loss1 = WeightedMSELoss(pred1, heat_gt, weight_ht)
        loss2 = WeightedMSELoss(pred2, zmap_gt, weight_z)  # zmap loss uses high fg weight
        loss3 = WeightedMSELoss(pred3, alignmap_temp, weight_align)  # align loss focuses strictly on fg

        if not prior_subnet_only:
            total_loss += loss1
            total_loss += loss2
            total_loss += loss3

        # Get value from Variable and save for log
        saved_for_log[names[3 * j]] = loss1.item()
        saved_for_log[names[3 * j + 1]] = loss2.item()
        saved_for_log[names[3 * j + 2]] = loss3.item()

    # pose prior loss
    _, _, h_prior, w_prior = saved_for_loss[-1].size()
    prior_map_pred = saved_for_loss[-1].view(b, num_anchors, -1, h_prior*w_prior).permute(0, 3, 1, 2).contiguous()
    prior_map_gt = prior_map_gt.view(b, num_anchors, -1, h_prior*w_prior).permute(0, 3, 1, 2).contiguous()
    prior_mask_coord = prior_mask_coord.view(b, num_anchors, h_prior*w_prior, 1).permute(0, 2, 1, 3).contiguous()
    prior_mask_conf = prior_mask_conf.view(b, num_anchors, h_prior*w_prior).permute(0, 2, 1).contiguous()
    prior_weight_map = prior_weight_map.view(b, num_anchors, h_prior*w_prior, 1).permute(0, 2, 1, 3).contiguous()

    coords_pred = prior_map_pred[:, :, :, :4]
    conf_pred = prior_map_pred[:, :, :, 4]
    joints_pred = prior_map_pred[:, :, :, 5:]
    coords_gt = prior_map_gt[:, :, :, :4]
    conf_gt = prior_map_gt[:, :, :, 4]
    joints_gt = prior_map_gt[:, :, :, 5:]

    loss_coord = WeightedMSELoss(coords_pred*prior_mask_coord, coords_gt*prior_mask_coord, prior_weight_map) * 4
    loss_obj = WeightedMSELoss(conf_pred*prior_mask_conf, conf_gt*prior_mask_conf, prior_weight_map[:, :, :, 0])
    if pred_vis:
        loss_selfpose = WeightedMSELoss(joints_pred * prior_mask_coord, joints_gt * prior_mask_coord,
                                        prior_weight_map) * 4 * num_joints
    else:
        loss_selfpose = WeightedMSELoss(joints_pred*prior_mask_coord, joints_gt*prior_mask_coord,
                                        prior_weight_map) * 3 * num_joints
    loss_prior = loss_coord + loss_obj + loss_selfpose

    total_loss += loss_prior
    saved_for_log['loss_prior'] = loss_prior.data.item()
    saved_for_log['loss_bbox'] = loss_coord.data.item()
    saved_for_log['loss_obj'] = loss_obj.data.item()
    saved_for_log['loss_selfpose'] = loss_selfpose.data.item()

    saved_for_log['max_ht'] = torch.max(
        saved_for_loss[-4].data[:, 0:-1, :, :]).item()
    saved_for_log['min_ht'] = torch.min(
        saved_for_loss[-4].data[:, 0:-1, :, :]).item()
    saved_for_log['max_z'] = torch.max(saved_for_loss[-3].data).item()
    saved_for_log['min_z'] = torch.min(saved_for_loss[-3].data).item()
    saved_for_log['max_alignf'] = torch.max((saved_for_loss[-2] * fg_mask_align).data).item()
    saved_for_log['min_alignf'] = torch.min((saved_for_loss[-2] * fg_mask_align).data).item()

    return total_loss, saved_for_log


def yolo_loss_fgweight(pred, prior_map_gt, prior_mask_conf, prior_mask_coord, num_joints, num_anchors):

    saved_for_log = OrderedDict()
    total_loss = 0

    # pose prior loss
    b, _, h_prior, w_prior = pred.size()
    prior_map_pred = pred.view(b, num_anchors, -1, h_prior * w_prior).permute(0, 3, 1, 2).contiguous()
    prior_map_gt = prior_map_gt.view(b, num_anchors, -1, h_prior*w_prior).permute(0, 3, 1, 2).contiguous()
    prior_mask_coord = prior_mask_coord.view(b, num_anchors, h_prior*w_prior, 1).permute(0, 2, 1, 3).contiguous()
    prior_mask_conf = prior_mask_conf.view(b, num_anchors, h_prior*w_prior).permute(0, 2, 1).contiguous()

    coords_pred = prior_map_pred[:, :, :, :4]
    conf_pred = prior_map_pred[:, :, :, 4]
    joints_pred = prior_map_pred[:, :, :, 5:]
    coords_gt = prior_map_gt[:, :, :, :4]
    conf_gt = prior_map_gt[:, :, :, 4]
    joints_gt = prior_map_gt[:, :, :, 5:]

    loss_coord = WeightedMSELoss(coords_pred, coords_gt, prior_mask_coord) * 4
    loss_obj = WeightedMSELoss(conf_pred, conf_gt, prior_mask_conf)
    loss_selfpose = WeightedMSELoss(joints_pred, joints_gt, prior_mask_coord) * 3 * num_joints
    loss_prior = loss_coord + loss_obj + loss_selfpose

    total_loss += loss_prior
    saved_for_log['loss_prior'] = loss_prior.data.item()
    saved_for_log['loss_bbox'] = loss_coord.data.item()
    saved_for_log['loss_obj'] = loss_obj.data.item()
    saved_for_log['loss_selfpose'] = loss_selfpose.data.item()

    return total_loss


def yolo_loss_fgweight_poseweight(pred, prior_map_gt, prior_mask_conf, prior_mask_coord, prior_weight_map, num_joints, num_anchors):
    """
    this version use pose rarity to re-weight the prior loss
    """

    saved_for_log = OrderedDict()
    total_loss = 0

    # pose prior loss
    b, _, h_prior, w_prior = pred.size()
    prior_map_pred = pred.view(b, num_anchors, -1, h_prior * w_prior).permute(0, 3, 1, 2).contiguous()
    prior_map_gt = prior_map_gt.view(b, num_anchors, -1, h_prior*w_prior).permute(0, 3, 1, 2).contiguous()
    prior_mask_coord = prior_mask_coord.view(b, num_anchors, h_prior*w_prior, 1).permute(0, 2, 1, 3).contiguous()
    prior_mask_conf = prior_mask_conf.view(b, num_anchors, h_prior*w_prior).permute(0, 2, 1).contiguous()
    prior_weight_map = prior_weight_map.view(b, num_anchors, h_prior*w_prior, 1).permute(0, 2, 1, 3).contiguous()

    coords_pred = prior_map_pred[:, :, :, :4]
    conf_pred = prior_map_pred[:, :, :, 4]
    joints_pred = prior_map_pred[:, :, :, 5:]
    coords_gt = prior_map_gt[:, :, :, :4]
    conf_gt = prior_map_gt[:, :, :, 4]
    joints_gt = prior_map_gt[:, :, :, 5:]

    loss_coord = WeightedMSELoss(coords_pred*prior_mask_coord, coords_gt*prior_mask_coord, prior_weight_map) * 4
    loss_obj = WeightedMSELoss(conf_pred*prior_mask_conf, conf_gt*prior_mask_conf, prior_weight_map[:, :, :, 0])
    loss_selfpose = WeightedMSELoss(joints_pred*prior_mask_coord, joints_gt*prior_mask_coord,
                                    prior_weight_map) * 3 * num_joints
    loss_prior = loss_coord + loss_obj + loss_selfpose

    total_loss += loss_prior
    saved_for_log['loss_prior'] = loss_prior.data.item()
    saved_for_log['loss_bbox'] = loss_coord.data.item()
    saved_for_log['loss_obj'] = loss_obj.data.item()
    saved_for_log['loss_selfpose'] = loss_selfpose.data.item()

    return total_loss, saved_for_log


def pop_net_rgb_loss_fgweight(saved_for_loss, heat_gt, alignmap_gt, fg_mask_align, prior_map_gt, prior_mask_conf, prior_mask_coord, num_stages, names, num_joints, num_anchors, prior_subnet_only=False):

    # names = build_names(num_stages)
    saved_for_log = OrderedDict()
    criterion1 = nn.MSELoss(reduction='mean').cuda()
    total_loss = 0
    weight_align = fg_mask_align
    b, c, h, w = weight_align.size()
    # pay attention to the weight for the extended dimension
    fg_mask = weight_align[:, :num_joints, :, :]
    weight_fg = torch.ones_like(fg_mask) * 0.1
    weight_fg = weight_fg + fg_mask * 0.9
    weight_ht = torch.cat([weight_fg, torch.ones(b, 1, h, w).cuda()], 1)

    for j in range(num_stages):
        pred1 = saved_for_loss[2 * j]  # heatmap
        pred2 = saved_for_loss[2 * j + 1]  # alignmap

        # Compute losses
        # loss1 = criterion1(pred1, heat_gt)
        loss1 = WeightedMSELoss(pred1, heat_gt, weight_ht)
        loss2 = WeightedMSELoss(pred2, alignmap_gt, weight_align)  # align loss focuses strictly on fg

        if not prior_subnet_only:
            total_loss += loss1
            total_loss += loss2

        # Get value from Variable and save for log
        saved_for_log[names[2 * j]] = loss1.item()
        saved_for_log[names[2 * j + 1]] = loss2.item()

    # pose prior loss
    _, _, h_prior, w_prior = saved_for_loss[-1].size()
    prior_map_pred = saved_for_loss[-1].view(b, num_anchors, -1, h_prior*w_prior).permute(0, 3, 1, 2).contiguous()
    prior_map_gt = prior_map_gt.view(b, num_anchors, -1, h_prior*w_prior).permute(0, 3, 1, 2).contiguous()
    prior_mask_coord = prior_mask_coord.view(b, num_anchors, h_prior*w_prior, 1).permute(0, 2, 1, 3).contiguous()
    prior_mask_conf = prior_mask_conf.view(b, num_anchors, h_prior*w_prior).permute(0, 2, 1).contiguous()

    coords_pred = prior_map_pred[:, :, :, :4]
    conf_pred = prior_map_pred[:, :, :, 4]
    joints_pred = prior_map_pred[:, :, :, 5:]
    coords_gt = prior_map_gt[:, :, :, :4]
    conf_gt = prior_map_gt[:, :, :, 4]
    joints_gt = prior_map_gt[:, :, :, 5:]

    loss_coord = WeightedMSELoss(coords_pred, coords_gt, prior_mask_coord) * 4
    loss_obj = WeightedMSELoss(conf_pred, conf_gt, prior_mask_conf)
    # loss_selfpose = WeightedMSELoss(joints_pred, joints_gt, prior_mask_coord) * 3 * num_joints
    """
        Use gt visibility, such that joint position loss focus on visible joints
        just use the fg mask for training visibility attributes
    """
    selfpose_mask = prior_mask_coord.repeat(1, 1, 1, num_joints)
    selfpose_mask = selfpose_mask.mul(joints_gt[:, :, :, 2*num_joints:])
    selfpose_mask = torch.cat([selfpose_mask.repeat(1, 1, 1, 2), prior_mask_coord.repeat(1, 1, 1, num_joints)], 3)
    loss_selfpose = WeightedMSELoss(joints_pred, joints_gt, selfpose_mask) * 3 * num_joints
    loss_prior = loss_coord + loss_obj + loss_selfpose

    total_loss += loss_prior
    saved_for_log['loss_prior'] = loss_prior.data.item()
    saved_for_log['loss_bbox'] = loss_coord.data.item()
    saved_for_log['loss_obj'] = loss_obj.data.item()
    saved_for_log['loss_selfpose'] = loss_selfpose.data.item()

    saved_for_log['max_ht'] = torch.max(
        saved_for_loss[-3].data[:, 0:-1, :, :]).item()
    saved_for_log['min_ht'] = torch.min(
        saved_for_loss[-3].data[:, 0:-1, :, :]).item()
    saved_for_log['max_alignf'] = torch.max((saved_for_loss[-2] * fg_mask_align).data).item()
    saved_for_log['min_alignf'] = torch.min((saved_for_loss[-2] * fg_mask_align).data).item()

    return total_loss, saved_for_log