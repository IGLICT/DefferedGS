#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import jittor as jt
# import torch.nn.functional as F
# from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return jt.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = jt.array([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

## jittor implementation of create_window
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    
    _2D_window = _1D_window.matmul(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = jt.array(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

## jittor implementation of ssim
def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.shape[-3]
    window = create_window(window_size, channel)
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)

## jittor implementation of _ssim

def _ssim(img1, img2, window, window_size, channel, size_average=True):

    # when not require grad, use cudnn_conv
    if jt.flags.no_grad:
        # breakpoint()
        conv2d = lambda x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1: \
            jt.cudnn.ops.cudnn_conv(x, weight, *stride, *padding, *dilation, groups)
    else:
        conv2d = jt.nn.conv2d
    if len(img1.shape) == 3:
        img1 = img1.unsqueeze(0)
    if len(img2.shape) == 3:
        img2 = img2.unsqueeze(0)
    
    mu1 = jt.nn.conv2d(img1, window, padding=window_size // 2, groups=channel)
    img2 = img2.float32()
    mu2 = jt.nn.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    mu1_sq.sync()
    mu2_sq.sync()
    mu1_mu2.sync()
    
    
    sigma1_sq = jt.nn.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = jt.nn.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = jt.nn.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_map.sync()
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    

def total_variation_loss(image):
    # Calculate total variation loss.
    # a = jt.mean(jt.abs(image[:, :, :, 1:]-image[:, :, :, :-1]))
    # b = jt.mean(jt.abs(image[:, :, 1:, :]-image[:, :, :-1, :]))
    a = jt.mean(jt.abs(image[:, :, 1:]-image[:, :, :-1]))
    b = jt.mean(jt.abs(image[:, 1:, :]-image[:, :-1, :]))
    return a+b

