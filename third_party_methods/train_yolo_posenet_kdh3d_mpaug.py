import argparse
import time
import os
from glob import glob
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter

from lib.network.yolo_posenet import YoloPoseNet
from lib.network.losses import *
from lib.datasets import data_augmentation_2d3d, datasets_kdh3d_mpaug

"""
A generic pose network with new 3D branch and 2D align branch :
Modification from rtpose:
    1. preprocess trained_model uses residual modules rather than vgg. The resolution reduction is tuned to be 1/8
    2. only uses two stages in the prediction
    3. New 3D branch is developed and integrated
    4. New 2D align branch is developed and integrated
    5. A new pose prior subnetwork is connected

Author: Yuliang Guo (33yuliangguo@gmail.com)
Date: April 2020
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DATA_DIR = '/raid/yuliangguo/Datasets/Kinect_Depth_Human3D'
# DATA_DIR = '/media/yuliang/DATA/Datasets/Kinect_Depth_Human3D'
if not os.path.exists('./trained_model'):
    os.mkdir('./trained_model')
save_folder = './trained_model/yolo_posenet_kdh3d_mpaug'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

ANNOTATIONS_TRAIN = sorted(glob('{}/labels_train_*.json'.format(os.path.join(DATA_DIR, 'labels'))))
ANNOTATIONS_VAL = sorted(glob('{}/labels_test_*.json'.format(os.path.join(DATA_DIR, 'labels'))))
IMAGE_DIR = os.path.join(DATA_DIR, 'depth_maps')
BG_FILE = os.path.join(DATA_DIR, 'labels', 'labels_bg.json')
BG_DIR = os.path.join(DATA_DIR, 'bg_maps')
SEG_DIR = os.path.join(DATA_DIR, 'seg_maps')
anchors = np.array([(6., 3.), (12., 6.)])


def train_cli(parser):
    group = parser.add_argument_group('dataset and loader')
    group.add_argument('--train-annotations', default=ANNOTATIONS_TRAIN)
    group.add_argument('--val-annotations', default=ANNOTATIONS_VAL)
    group.add_argument('--image-dir', default=IMAGE_DIR)
    group.add_argument('--bg-file', default=BG_FILE)
    group.add_argument('--bg-dir', default=BG_DIR)
    group.add_argument('--seg-dir', default=SEG_DIR)
    group.add_argument('--bg-aug', default=True)
    group.add_argument('--loader-workers', default=16, type=int,
                       help='number of workers for data loading')
    group.add_argument('--batch-size', default=30, type=int,
                       help='batch size')
    group.add_argument('--lr', '--learning-rate', default=1., type=float,
                       metavar='LR', help='initial learning rate')
    group.add_argument('--momentum', default=0.9, type=float, metavar='M',
                       help='momentum')
    group.add_argument('--weight-decay', '--wd', default=0.000, type=float,
                       metavar='W', help='weight decay (default: 1e-4)')
    group.add_argument('--nesterov', dest='nesterov', default=True, type=bool)
    group.add_argument('--print_freq', default=20, type=int, metavar='N',
                       help='number of iterations to print the training statistics')


def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_cli(parser)
    parser.add_argument('--epochs', default=200, type=int,
                        help='number of epochs to train')
    parser.add_argument('--square-edge', default=224, type=int,
                        help='square edge of input images')
    parser.add_argument('--num-parts', default=15, type=int,
                        help='number of body parts')
    parser.add_argument('--rarity-weight', default=1, type=int,
                        help='if use rarity weighted loss for training')
    parser.add_argument('--reduction', default=8, type=float,
                        help='dimension reduction for the parts stage')
    parser.add_argument('--max-aug-ratio', default=1.7, type=float,
                        help='the max augment depth ratio')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    args = parser.parse_args()

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True

    return args


def train_factory(args, preprocess):
    train_data = datasets_kdh3d_mpaug.KDH3D_Keypoints(
        img_dir=args.image_dir,
        ann_file_list=args.train_annotations,
        preprocess=preprocess,
        input_x=args.square_edge,
        input_y=args.square_edge,
        stride=args.reduction,
        anchors=anchors,
        bg_file=args.bg_file,
        bg_dir=args.bg_dir,
        seg_dir=args.seg_dir
    )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True)

    val_data = datasets_kdh3d_mpaug.KDH3D_Keypoints(
        img_dir=args.image_dir,
        ann_file_list=args.val_annotations,
        preprocess=preprocess,
        input_x=args.square_edge,
        input_y=args.square_edge,
        stride=args.reduction,
        anchors=anchors,
        bg_file=args.bg_file,
        bg_dir=args.bg_dir,
        seg_dir=args.seg_dir
    )

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True)

    return train_loader, val_loader, train_data, val_data


def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    meter_dict = {}
    meter_dict['loss_prior'] = AverageMeter()
    meter_dict['loss_bbox'] = AverageMeter()
    meter_dict['loss_obj'] = AverageMeter()
    meter_dict['loss_selfpose'] = AverageMeter()
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, (img, _, _, _, _, _, _, prior_map, prior_mask_conf, prior_mask_coord, prior_weight_map, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        img = img.cuda()
        prior_map = prior_map.cuda()
        prior_mask_conf = prior_mask_conf.cuda()
        prior_mask_coord = prior_mask_coord.cuda()
        prior_weight_map = prior_weight_map.cuda()

        # compute output
        pred = model(img)

        if not args.rarity_weight:
            total_loss, saved_for_log = yolo_loss_fgweight(pred,
                                                           prior_map,
                                                           prior_mask_conf,
                                                           prior_mask_coord,
                                                           model.module.num_parts,
                                                           len(anchors))
        else:
            total_loss, saved_for_log = yolo_loss_fgweight_poseweight(pred,
                                                                      prior_map,
                                                                      prior_mask_conf,
                                                                      prior_mask_coord,
                                                                      prior_weight_map,
                                                                      model.module.num_parts,
                                                                      len(anchors))

        for name, _ in meter_dict.items():
            meter_dict[name].update(saved_for_log[name], img.size(0))
        losses.update(total_loss, img.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print_string = 'Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(train_loader))
            print_string += 'Data time {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(data_time=data_time)
            print_string += 'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses)

            for name, value in meter_dict.items():
                print_string += '{name}: {loss.val:.4f} ({loss.avg:.4f})\t'.format(name=name, loss=value)
            print(print_string)
    return losses.avg  
        
        
def validate(val_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    meter_dict = {}

    # switch to train mode
    model.eval()

    end = time.time()
    for i, (img, _, _, _, _, _, _, prior_map, prior_mask_conf, prior_mask_coord, prior_weight_map, _) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        img = img.cuda()
        prior_map = prior_map.cuda()
        prior_mask_conf = prior_mask_conf.cuda()
        prior_mask_coord = prior_mask_coord.cuda()
        prior_weight_map = prior_weight_map.cuda()

        # compute output
        pred = model(img)

        if not args.rarity_weight:
            total_loss, saved_for_log = yolo_loss_fgweight(pred,
                                                           prior_map,
                                                           prior_mask_conf,
                                                           prior_mask_coord,
                                                           model.module.num_parts,
                                                           len(anchors))
        else:
            total_loss, saved_for_log = yolo_loss_fgweight_poseweight(pred,
                                                                      prior_map,
                                                                      prior_mask_conf,
                                                                      prior_mask_coord,
                                                                      prior_weight_map,
                                                                      model.module.num_parts,
                                                                      len(anchors))

        losses.update(total_loss.item(), img.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()  
        if i % args.print_freq == 0:
            print_string = 'Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(val_loader))
            print_string += 'Data time {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(data_time=data_time)
            print_string += 'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses)

            for name, value in meter_dict.items():
                print_string += '{name}: {loss.val:.4f} ({loss.avg:.4f})\t'.format(name=name, loss=value)
            print(print_string)
                
    return losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    args = cli()

    model_save_filename = os.path.join(save_folder, 'best_pose.pth')
    intrinsics = datasets_kdh3d_mpaug.intrinsics

    print("Loading dataset...")
    # load train data
    preprocess = data_augmentation_2d3d.Compose([
        data_augmentation_2d3d.Cvt2ndarray(),
        data_augmentation_2d3d.Rotate(cx=intrinsics['cx'], cy=intrinsics['cy']),
        data_augmentation_2d3d.RenderDepth(cx=intrinsics['cx'], cy=intrinsics['cy'], max_ratio=args.max_aug_ratio),
        # data_augmentation_depth_3d.Hflip(swap_indices=datasets_kdh3d_gen.get_swap_part_indices()),
        data_augmentation_2d3d.Crop(),  # it may violate the 3D-2D geometry
        data_augmentation_2d3d.Resize(args.square_edge)
    ])
    train_loader, val_loader, train_data, val_data = train_factory(args, preprocess)

    # model
    model = YoloPoseNet(args.num_parts, input_dim=1, anchors=anchors)
    if os.path.exists(model_save_filename):
        state_dict = torch.load(model_save_filename)
        # modify for cuda version and cpu version compatibility
        state_dict = OrderedDict([(key[key.find('.') + 1:], value) for key, value in state_dict.items()])
        model.load_state_dict(state_dict)

    model = torch.nn.DataParallel(model).cuda()
    trainable_vars = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(trainable_vars, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True, threshold=0.0001,
                                     threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-08)

    # Tensorboard writer
    writer = SummaryWriter(os.path.join(save_folder, 'Tensorboard/'))

    best_val_loss = np.inf
    for epoch in range(args.epochs):

        # train for one epoch
        train_loss = train(train_loader, model, optimizer, epoch)

        # evaluate on validation set
        val_loss = validate(val_loader, model, epoch)

        lr_scheduler.step(val_loss)

        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        if is_best:
            torch.save(model.state_dict(), model_save_filename)
            print('best model updated at epoch: {}'.format(epoch))

        writer.add_scalar('Train/loss', train_loss, epoch)
        writer.add_scalar('Val/loss', val_loss, epoch)
    writer.close()
