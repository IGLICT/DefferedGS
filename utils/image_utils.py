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
import matplotlib.pyplot as plt
# import torch.nn.functional as F

def log10(x):
    return jt.log(x) / jt.log(10.0)
    
def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * log10(1.0 / jt.sqrt(mse))

def gradient_map(image):
    sobel_x = jt.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    sobel_y = jt.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    
    grad_x = jt.cat([jt.nn.conv2d(image[i].unsqueeze(0), sobel_x, padding=1) for i in range(image.shape[0])])
    grad_y = jt.cat([jt.nn.conv2d(image[i].unsqueeze(0), sobel_y, padding=1) for i in range(image.shape[0])])
    magnitude = jt.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = magnitude.norm(dim=0, keepdim=True)

    return magnitude

def colormap(map, cmap="turbo"):
    colors = jt.array(plt.cm.get_cmap(cmap).colors)
    map = (map - map.min()) / (map.max() - map.min())
    map = (map * 255).round().long().squeeze()
    map = colors[map].permute(2,0,1)
    return map

def render_net_image(render_pkg, render_items, render_mode, camera):
    output = render_items[render_mode].lower()
    if output == 'alpha':
        net_image = render_pkg["rend_alpha"]
    elif output == 'normal':
        net_image = render_pkg["rend_normal"]
        net_image = (net_image+1)/2
    elif output == 'depth':
        net_image = render_pkg["surf_depth"]
    elif output == 'edge':
        net_image = gradient_map(render_pkg["render"])
    elif output == 'curvature':
        net_image = render_pkg["rend_normal"]
        net_image = (net_image+1)/2
        net_image = gradient_map(net_image)
    else:
        net_image = render_pkg["render"]

    if net_image.shape[0]==1:
        net_image = colormap(net_image)
    return net_image

def linear_to_srgb(linear, eps=1e-10):
    eps =jt.finfo("float32").eps
    srgb0 = 323 / 25 * linear
    # srgb1 = (
    #     211 * jt.fmax(jt.full_like(linear, eps), linear) ** (5 / 12) - 11
    # ) / 200
    srgb1 = (
        211 * jt.where(jt.full_like(linear, eps)>linear,jt.full_like(linear, eps), linear) ** (5 / 12) - 11
    ) / 200
    
    return jt.where(linear <= 0.0031308, srgb0, srgb1)

def srgb_to_linear(srgb):
    gamma = ((srgb + 0.055) / 1.055)**2.4
    scale = srgb / 12.92
    return jt.where(srgb <= 0.0404482362771082, scale, gamma)

def linear_to_srgb_np(linear, eps=1e-10):
    eps = np.finfo(np.float32).eps
    srgb0 = 323 / 25 * linear
    srgb1 = (
        211 * np.fmax(np.full_like(linear, eps), linear) ** (5 / 12) - 11
    ) / 200
    return np.where(linear <= 0.0031308, srgb0, srgb1)

def srgb_to_linear_np(srgb):
    gamma = ((srgb + 0.055) / 1.055)**2.4
    scale = srgb / 12.92
    return np.where(srgb <= 0.0404482362771082, scale, gamma)