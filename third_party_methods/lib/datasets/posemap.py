# coding=utf-8
"""Implement Pose Regulated Depth maps
:param centerA: int with shape (2,), centerA will pointed by centerB.
:param centerB: int with shape (2,), centerB will point to centerA.
:param accumulate_vec_map: one channel of paf.
:param count: store how many pafs overlaped in one coordinate of accumulate_vec_map.
:param params_transform: store the value of stride and crop_szie_y, crop_size_x                 
"""
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc, ndimage


def putJointDxDy(center, depth, accumulate_map, accumulate_mask, accumulate_dist, z_map, grid_y, grid_x, stride, radius=2, max_dist=None):
    """

    Args:
        center:
        depth:
        accumulate_map:
        accumulate_mask:
        accumulate_dist: for multiple instance, the field with the minimum dists are updated. Initialized as max
        z_map: for the exact overlap, use prepared zmap and current depth to determine updating
        grid_y:
        grid_x:
        stride:
        radius:

    Returns:

    """
    if max_dist is None:
        max_dist = 2 * (radius + 0.5)

    # radius determines the distance field range. the larger the radius, the more possible conflict from multiple-people
    center = center.astype(float)
    center = center / stride

    # compute range considering image border, ATTENTION, need to use 'int' not 'round'
    min_x = max(int(int(center[0] - radius)), 0)
    max_x = min(int(int(center[0] + radius)), grid_x-1)
    min_y = max(int(int(center[1] - radius)), 0)
    max_y = min(int(int(center[1] + radius)), grid_y-1)

    range_x = list(range(int(min_x), int(max_x)+1, 1))
    range_y = list(range(int(min_y), int(max_y)+1, 1))
    xx, yy = np.meshgrid(range_x, range_y)

    # # compute full-range distance field
    # range_x = list(range(0, grid_x))
    # range_y = list(range(0, grid_y))
    # xx, yy = np.meshgrid(range_x, range_y)
    dx_map = -(xx + 0.5 - center[0])
    dy_map = -(yy + 0.5 - center[1])

    # truncate
    dx_map[dx_map < (-radius-0.5)] = (-radius-0.5)
    dx_map[dx_map > (radius+0.5)] = (radius+0.5)
    dy_map[dy_map < (-radius-0.5)] = (-radius-0.5)
    dy_map[dy_map > (radius+0.5)] = (radius+0.5)
    # normalize
    dx_map /= (0.5+radius)  # the vector from (x,y) to center normalized by radius
    dy_map /= (0.5+radius)  # the vector from (x,y) to center normalized by radius
    # assign
    posemap_dxdy = np.ones_like(accumulate_map).astype(np.float32)*max_dist
    posemap_dxdy[yy, xx, 0] = dx_map
    posemap_dxdy[yy, xx, 1] = dy_map
    dist_map = np.sqrt(posemap_dxdy[:, :, 0]**2 + posemap_dxdy[:, :, 1]**2)

    # update accumulate map and mask
    update_indices = np.logical_and(np.abs(posemap_dxdy) < max_dist, np.repeat((dist_map < accumulate_dist)[:, :, np.newaxis], 2, axis=2))
    # update_indices = np.logical_and(update_indices,  np.repeat((z_map >= depth)[:, :, np.newaxis], 2, axis=2))
    accumulate_map[update_indices] = posemap_dxdy[update_indices]
    accumulate_dist[update_indices[:, :, 0]] = dist_map[update_indices[:, :, 0]]
    accumulate_mask[yy, xx, :] = 1
    return accumulate_map, accumulate_mask, accumulate_dist


def putJointZ(center, depth, accumulate_map, accumulate_mask, grid_y, grid_x, stride, radius=1, max_depth=10.0):
    # radius = 1  # part radius
    center = center.astype(float)
    center = center / stride

    # compute range considering image border
    min_x = max(int(int(center[0] - radius)), 0)
    max_x = min(int(int(center[0] + radius)), grid_x-1)
    min_y = max(int(int(center[1] - radius)), 0)
    max_y = min(int(int(center[1] + radius)), grid_y-1)

    range_x = list(range(int(min_x), int(max_x)+1, 1))
    range_y = list(range(int(min_y), int(max_y)+1, 1))
    xx, yy = np.meshgrid(range_x, range_y)

    posemap_Z = np.ones_like(accumulate_map) * max_depth
    posemap_Z[yy, xx] = depth

    # update accumulate map and mask
    accumulate_map = np.minimum(posemap_Z, accumulate_map)
    new_mask = np.logical_and(posemap_Z < max_depth, accumulate_mask == 0)
    accumulate_map[new_mask] = posemap_Z[new_mask]
    accumulate_mask = np.logical_or(accumulate_mask, new_mask)
    return accumulate_map, accumulate_mask

