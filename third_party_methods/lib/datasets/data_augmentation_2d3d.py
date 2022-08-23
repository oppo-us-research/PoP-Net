import io
import PIL
import numpy as np
import scipy
from random import uniform
import cv2
import copy
import torch
import torchvision
from lib.utils.common import homographic_transform


# normalize_depth = torchvision.transforms.Normalize(  # pylint: disable=invalid-name
#     mean=[3],
#     std=[2]
# )
#
#
# image_transform = torchvision.transforms.Compose([  # pylint: disable=invalid-name
#     torchvision.transforms.ToTensor(),
#     normalize_depth])


"""
think of the usage of these augmentation for depth data 
"""


def jpeg_compression_augmentation(im):
    f = io.BytesIO()
    im.save(f, 'jpeg', quality=50)
    return PIL.Image.open(f)


def blur_augmentation(im, max_sigma=5.0):
    im_np = np.asarray(im)
    sigma = max_sigma * float(torch.rand(1).item())
    im_np = scipy.ndimage.filters.gaussian_filter(im_np, sigma=(sigma, sigma, 0))
    return PIL.Image.fromarray(im_np)


# image_transform_train = torchvision.transforms.Compose([  # pylint: disable=invalid-name
#     torchvision.transforms.ColorJitter(brightness=0.1,
#                                        contrast=0.1,
#                                        saturation=0.1,
#                                        hue=0.1),
#     torchvision.transforms.RandomApply([
#         # maybe not relevant for COCO, but good for other datasets:
#         torchvision.transforms.Lambda(jpeg_compression_augmentation),
#     ], p=0.1),
#     torchvision.transforms.RandomGrayscale(p=0.01),
#     torchvision.transforms.ToTensor(),
#     normalize_depth,
# ])

"""###############################################################################################################"""


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for function_ in self.transforms:
            data = function_(data)
        return data


class Cvt2ndarray(object):
    def __init__(self, num_joints=15, max_crop=0.1):
        super().__init__()
        self.max_crop = max_crop
        self.num_joints = num_joints

    def __call__(self, data):
        image, label = data

        new_label = []
        for lb in label:
            single_new_label = copy.deepcopy(lb)
            single_new_label['2d_joints'] = np.array(single_new_label['2d_joints']).reshape([self.num_joints, 2]).astype(np.float32)
            if 'visible_joints' in single_new_label.keys():
                single_new_label['visible_joints'] = np.array(single_new_label['visible_joints'])
            if 'bbox' in single_new_label.keys():
                single_new_label['bbox'] = np.array(single_new_label['bbox'])

            new_label.append(single_new_label)
        return image.astype(np.float32), new_label


# TODO: The change of aspect ratio conflicts with fixed intrinsics. Should not be used in testing.
#  But in training, does it affect Z estimation? OR only relative X,Y?
class Crop(object):

    def __init__(self, max_crop=0.1):
        super().__init__()
        self.max_crop = max_crop

    def __call__(self, data):
        image, label = data
        height, width = image.shape[:2]
        xmin = width
        ymin = height
        xmax = 0
        ymax = 0

        crop_left = uniform(0, self.max_crop)
        crop_right = uniform(0, self.max_crop)
        crop_top = uniform(0, self.max_crop)
        crop_bottom = uniform(0, self.max_crop)

        new_xmin = int(min(crop_left*width, xmin))
        new_ymin = int(min(crop_top*height, ymin))
        new_xmax = int(max(width - 1 - crop_right*width, xmax))
        new_ymax = int(max(height - 1 - crop_bottom*height, ymax))

        image = image[new_ymin:new_ymax, new_xmin:new_xmax]
        new_label = []
        for lb in label:
            single_new_label = copy.deepcopy(lb)
            single_new_label['2d_joints'][:, 0] -= new_xmin
            single_new_label['2d_joints'][:, 1] -= new_ymin
            if 'bbox' in single_new_label.keys():
                single_new_label['bbox'][0:4:2] -= new_xmin
                single_new_label['bbox'][1:4:2] -= new_ymin
            new_label.append(single_new_label)
        return image, new_label


class CropPoseRoi(object):
    """
        compute ROI from a random chosen human
        output ROI image and its corresponding pose
    """
    def __init__(self, joint2box_margin=20):
        super().__init__()
        self.joint2box_margin = joint2box_margin

    def __call__(self, data):
        image, label = data
        height, width = image.shape[:2]

        # choose only one random body if multiple people in a single image
        rnd_id = np.random.randint(len(label), size=1)

        xmin = np.min(label[rnd_id[0]]['2d_joints'][:, 0]) - self.joint2box_margin
        ymin = np.min(label[rnd_id[0]]['2d_joints'][:, 1]) - self.joint2box_margin
        xmax = np.max(label[rnd_id[0]]['2d_joints'][:, 0]) + self.joint2box_margin
        ymax = np.max(label[rnd_id[0]]['2d_joints'][:, 1]) + self.joint2box_margin

        new_xmin = int(max(0, min(width, xmin)))
        new_ymin = int(max(0, min(height, ymin)))
        new_xmax = int(max(0, min(width, xmax)))
        new_ymax = int(max(0, min(height, ymax)))

        image = image[new_ymin:new_ymax, new_xmin:new_xmax]
        new_label = []
        single_new_label = copy.deepcopy(label[rnd_id[0]])
        single_new_label['2d_joints'][:, 0] -= new_xmin
        single_new_label['2d_joints'][:, 1] -= new_ymin
        if 'bbox' in single_new_label.keys():
            single_new_label['bbox'][0:4:2] -= new_xmin
            single_new_label['bbox'][1:4:2] -= new_ymin
        new_label.append(single_new_label)

        # ATTENTION: how can the roi used here be consistent with the post-process to recover origin pose?
        #  --> testing should not have other augmentations in previous.
        return image, new_label


class CropPoseRoiJitter(object):
    """
        compute ROI from a random chosen human
        output ROI image and its corresponding pose
    """
    def __init__(self, joint2box_margin=20, max_c_jitter=10, max_aspect_jitter=0.2):
        super().__init__()
        self.joint2box_margin = joint2box_margin
        self.max_c_jitter = max_c_jitter
        self.max_aspect_jitter = max_aspect_jitter

    def __call__(self, data):
        image, label = data
        height, width = image.shape[:2]

        # cx_jitter = uniform(-self.max_c_jitter, self.max_c_jitter)
        cx_jitter = 0
        # cy_jitter = uniform(-self.max_c_jitter, 0)
        cy_jitter = 0
        w_jitter = uniform(1 - self.max_aspect_jitter, 1)
        h_jitter = uniform(1 - self.max_aspect_jitter, 1)

        # choose only one random body if multiple people in a single image
        rnd_id = np.random.randint(len(label), size=1)

        xmin = np.min(label[rnd_id[0]]['2d_joints'][:, 0]) - self.joint2box_margin
        ymin = np.min(label[rnd_id[0]]['2d_joints'][:, 1]) - self.joint2box_margin
        xmax = np.max(label[rnd_id[0]]['2d_joints'][:, 0]) + self.joint2box_margin
        ymax = np.max(label[rnd_id[0]]['2d_joints'][:, 1]) + self.joint2box_margin

        cx = (xmin + xmax) / 2 + cx_jitter
        cy = (ymin + ymax) / 2 + cy_jitter
        crop_w = (xmax - xmin) * w_jitter
        crop_h = (ymax - ymin) * h_jitter
        new_xmin = cx - crop_w / 2
        new_ymin = cy - crop_h / 2
        new_xmax = cx + crop_w / 2
        new_ymax = cy + crop_h / 2

        new_xmin = int(max(0, min(width, new_xmin)))
        new_ymin = int(max(0, min(height, new_ymin)))
        new_xmax = int(max(0, min(width, new_xmax)))
        new_ymax = int(max(0, min(height, new_ymax)))

        image = image[new_ymin:new_ymax, new_xmin:new_xmax]
        new_label = []
        single_new_label = copy.deepcopy(label[rnd_id[0]])
        single_new_label['2d_joints'][:, 0] -= new_xmin
        single_new_label['2d_joints'][:, 1] -= new_ymin
        if 'bbox' in single_new_label.keys():
            single_new_label['bbox'][0:4:2] -= new_xmin
            single_new_label['bbox'][1:4:2] -= new_ymin
        new_label.append(single_new_label)

        # ATTENTION: how can the roi used here be consistent with the post-process to recover origin pose?
        #  --> testing should not have other augmentations in previous.
        return image, new_label


class CropPoseRoiV2(object):
    """
        compute ROI from a random chosen human
        output ROI image and its corresponding pose
    """
    def __init__(self, margin_ratio_x=2, margin_ratio_y=1.5):
        super().__init__()
        self.margin_ratio_x = margin_ratio_x
        self.margin_ratio_y = margin_ratio_y

    def __call__(self, data):
        image, label = data
        height, width = image.shape[:2]

        # choose only one random body if multiple people in a single image
        rnd_id = np.random.randint(len(label), size=1)

        xmin = np.min(label[rnd_id[0]]['2d_joints'][:, 0])
        xmax = np.max(label[rnd_id[0]]['2d_joints'][:, 0])
        ymin = np.min(label[rnd_id[0]]['2d_joints'][:, 1])
        ymax = np.max(label[rnd_id[0]]['2d_joints'][:, 1])
        xcenter = (xmin + xmax) / 2
        box_w = xmax - xmin
        ycenter = (ymin + ymax) / 2
        box_h = ymax - ymin

        new_xmin = xcenter - box_w / 2 * self.margin_ratio_x
        new_xmax = xcenter + box_w / 2 * self.margin_ratio_x
        new_ymin = ycenter - box_h / 2 * self.margin_ratio_y
        new_ymax = ycenter + box_h / 2 * self.margin_ratio_y

        new_xmin = int(max(0, min(width, new_xmin)))
        new_xmax = int(max(0, min(width, new_xmax)))
        new_ymin = int(max(0, min(height, new_ymin)))
        new_ymax = int(max(0, min(height, new_ymax)))

        image = image[new_ymin:new_ymax, new_xmin:new_xmax]
        new_label = []
        single_new_label = copy.deepcopy(label[rnd_id[0]])
        single_new_label['2d_joints'][:, 0] -= new_xmin
        single_new_label['2d_joints'][:, 1] -= new_ymin
        if 'bbox' in single_new_label.keys():
            single_new_label['bbox'][0:4:2] -= new_xmin
            single_new_label['bbox'][1:4:2] -= new_ymin
        new_label.append(single_new_label)

        # ATTENTION: how can the roi used here be consistent with the post-process to recover origin pose?
        #  --> testing should not have other augmentations in previous.
        return image, new_label


# simulate an depth image captured at a position by moving camera along current principal axis
class RenderDepth(object):
    """
        ATTENTION: the rounding error associated in cropping 2D image is inevitable, and increases with ratio deviation from 1.0
    """
    def __init__(self, cx=None, cy=None, min_ratio=0.7, max_ratio=1.2):
        super().__init__()
        self.cx = cx or None
        self.cy = cy or None
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, data):
        a = uniform(self.min_ratio, self.max_ratio)

        image, label = data
        chn = 1
        if len(image.shape) == 2:
            height, width = image.shape
        else:
            height, width, chn = image.shape
        xmin = float(0)
        ymin = float(0)
        xmax = float(width)
        ymax = float(height)

        if self.cx is None:
            self.cx = width/2
        if self.cy is None:
            self.cy = height/2

        new_xmin = int(a * (xmin - self.cx) + self.cx)
        new_ymin = int(a * (ymin - self.cy) + self.cy)
        new_xmax = int(a * (xmax - self.cx) + self.cx)
        new_ymax = int(a * (ymax - self.cy) + self.cy)
        # recompute 'a' using rounded 2d coordinate to reduce rounding error
        ax = (new_xmin - self.cx) / (xmin - self.cx)
        ay = (new_ymin - self.cy) / (ymin - self.cy)
        a = (ax + ay) / 2

        new_width = new_xmax - new_xmin + 1
        new_height = new_ymax - new_ymin + 1
        if a <= 1:
            new_image = image[new_ymin:new_ymax, new_xmin:new_xmax]
        else:
            # new_image = np.ones((new_height, new_width)).astype(np.float32)*self.max_depth
            dx = int(xmin - new_xmin)
            dy = int(ymin - new_ymin)
            if chn > 1:
                new_image = np.zeros((new_height, new_width, chn)).astype(np.float32)
                new_image[dy:dy + height, dx:dx + width, :] = image
            else:
                new_image = np.zeros((new_height, new_width)).astype(np.float32)
                new_image[dy:dy + height, dx:dx + width] = image

        new_label = []
        for lb in label:
            single_new_label = copy.deepcopy(lb)
            # moving camera along principle axis changes 2D x, y and 3D Z
            single_new_label['2d_joints'][:, 0] -= new_xmin
            single_new_label['2d_joints'][:, 1] -= new_ymin
            single_new_label['3d_joints'][:, 2] *= a
            if 'bbox' in single_new_label.keys():
                single_new_label['bbox'][0:4:2] -= new_xmin
                single_new_label['bbox'][1:4:2] -= new_ymin
            new_label.append(single_new_label)

        new_image *= a
        return new_image, new_label


# simulate an rgb image captions at a different camera distance
class RandomSacleRGB(object):
    """
        ATTENTION: the rounding error associated in cropping 2D image is inevitable, and increases with ratio deviation from 1.0
    """
    def __init__(self, min_ratio=0.7, max_ratio=1.3):
        super().__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, data):
        a = uniform(self.min_ratio, self.max_ratio)

        image, label = data
        height, width, chn = image.shape
        xmin = float(0)
        ymin = float(0)
        xmax = float(width)
        ymax = float(height)

        self.cx = width/2
        self.cy = height/2

        new_xmin = int(a * (xmin - self.cx) + self.cx)
        new_ymin = int(a * (ymin - self.cy) + self.cy)
        new_xmax = int(a * (xmax - self.cx) + self.cx)
        new_ymax = int(a * (ymax - self.cy) + self.cy)
        # recompute 'a' using rounded 2d coordinate to reduce rounding error
        ax = (new_xmin - self.cx) / (xmin - self.cx)
        ay = (new_ymin - self.cy) / (ymin - self.cy)
        a = (ax + ay) / 2

        new_width = new_xmax - new_xmin + 1
        new_height = new_ymax - new_ymin + 1
        if a <= 1:
            new_image = image[new_ymin:new_ymax, new_xmin:new_xmax]
        else:
            # new_image = np.ones((new_height, new_width)).astype(np.float32)*self.max_depth
            dx = int(xmin - new_xmin)
            dy = int(ymin - new_ymin)
            new_image = np.zeros((new_height, new_width, chn)).astype(np.float32)
            new_image[dy:dy + height, dx:dx + width, :] = image

        new_label = []
        for lb in label:
            single_new_label = copy.deepcopy(lb)
            # moving camera along principle axis changes 2D x, y and 3D Z
            single_new_label['2d_joints'][:, 0] -= new_xmin
            single_new_label['2d_joints'][:, 1] -= new_ymin
            if 'bbox' in single_new_label.keys():
                single_new_label['bbox'][0:4:2] -= new_xmin
                single_new_label['bbox'][1:4:2] -= new_ymin
            new_label.append(single_new_label)

        return new_image, new_label


# rotate around (cx, cy), both 2D x y and 3D X Y rotate wrt principle axis
class Rotate(object):
    """
        Attention: use the the same bounding box given small range of degrees to rotate
    """
    def __init__(self, cx=None, cy=None, is_3d=False):
        super().__init__()
        self.cx = cx or None
        self.cy = cy or None
        self.is_3d = is_3d

    def __call__(self, data):
        image, label = data
        height, width = image.shape[:2]

        rot = uniform(-10, 10)
        if self.cx is not None and self.cy is not None:
            center_x = self.cx
            center_y = self.cy
        else:
            center_x = width/2
            center_y = height/2
        rot_mat = cv2.getRotationMatrix2D((center_x, center_y), rot, 1.0)
        img_rot = cv2.warpAffine(image, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
        rot_mat = np.vstack([rot_mat, [0, 0, 1]])

        rot_mat3d = cv2.getRotationMatrix2D((0, 0), rot, 1.0)
        rot_mat3d = np.vstack([rot_mat3d, [0, 0, 1]])

        new_label = []
        for lb in label:
            single_new_label = copy.deepcopy(lb)
            single_new_label['2d_joints'][:, 0], single_new_label['2d_joints'][:, 1] = \
                homographic_transform(rot_mat, single_new_label['2d_joints'][:, 0], single_new_label['2d_joints'][:, 1])
            if self.is_3d:
                single_new_label['3d_joints'][:, 0], single_new_label['3d_joints'][:, 1] = \
                    homographic_transform(rot_mat3d, single_new_label['3d_joints'][:, 0], single_new_label['3d_joints'][:, 1])
            new_label.append(single_new_label)
        return img_rot, new_label


# horizontal flip of image data
class Hflip(object):
    """
        TODO: In fact, image should flip wrt. x = cx. Now it assumes cx = width/2. If not, need to cut the image.
        Swap the indices of left/right parts.
        2D x update
        3D X flip
    """
    def __init__(self, swap_indices, is_3d=False):
        super().__init__()
        self.swap_indices = swap_indices
        self.is_3d = is_3d

    def __call__(self, data):
        image, label = data

        if uniform(0, 1) >= 0.5:
            image = np.flip(image, axis=1)
            height, width = image.shape[:2]

            new_label = []
            for lb in label:
                single_new_label = copy.deepcopy(lb)
                # update 2D x and 3D X
                single_new_label['2d_joints'][:, 0] = - single_new_label['2d_joints'][:, 0] + width
                if self.is_3d:
                    single_new_label['3d_joints'][:, 0] *= -1
                # swap left and right parts
                single_new_label['2d_joints'] = single_new_label['2d_joints'][self.swap_indices, :]
                if self.is_3d:
                    single_new_label['3d_joints'] = single_new_label['3d_joints'][self.swap_indices, :]
                if 'visible_joints' in single_new_label.keys():
                    single_new_label['visible_joints'] = single_new_label['visible_joints'][self.swap_indices]
                if 'bbox' in single_new_label.keys():
                    xmin = -single_new_label['bbox'][2] + width
                    xmax = -single_new_label['bbox'][0] + width
                    single_new_label['bbox'][0] = xmin
                    single_new_label['bbox'][2] = xmax

                new_label.append(single_new_label)

            label = new_label
        return image, label


# resize image input target size, this only affects 2D labels
class Resize(object):

    def __init__(self, target_w, target_h=None):
        super().__init__()
        self.target_w = target_w
        if target_h is None:
            self.target_h = target_w
        else:
            self.target_h = target_h

    def __call__(self, data):
        image, label = data
        height, width = image.shape[:2]
        image = cv2.resize(image, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        width_ratio = float(self.target_w) / width
        height_ratio = float(self.target_h) / height
        new_label = []
        for lb in label:
            single_new_label = copy.deepcopy(lb)
            single_new_label['2d_joints'][:, 0] *= width_ratio
            single_new_label['2d_joints'][:, 1] *= height_ratio
            if 'bbox' in single_new_label.keys():
                single_new_label['bbox'][0:4:2] = single_new_label['bbox'][0:4:2].astype(np.float)*width_ratio
                single_new_label['bbox'][1:4:2] = single_new_label['bbox'][1:4:2].astype(np.float)*height_ratio
            new_label.append(single_new_label)
        return image, new_label


# given any sized image, pad it to make it square, keep the center portion active
class SquarePadRGB(object):

    def __init__(self):
        super().__init__()

    def __call__(self, data):

        image, label = data
        height, width, chn = image.shape

        square_edge = np.maximum(height, width)
        image_new = np.zeros((square_edge, square_edge, chn)).astype(np.float32)

        new_xmin = int((square_edge - width)/2)
        new_ymin = int((square_edge - height)/2)

        image_new[new_ymin:new_ymin+height, new_xmin:new_xmin+width, :] = image
        new_label = []
        for lb in label:
            single_new_label = copy.deepcopy(lb)
            single_new_label['2d_joints'][:, 0] += new_xmin
            single_new_label['2d_joints'][:, 1] += new_ymin
            if 'bbox' in single_new_label.keys():
                single_new_label['bbox'][0:4:2] += new_xmin
                single_new_label['bbox'][1:4:2] += new_ymin
            new_label.append(single_new_label)
        return image_new, new_label
