import jittor as jt
from jittor import nn

import numpy as np
import os, cv2
import matplotlib.pyplot as plt
import math

def depths_to_points(view, depthmap):
    c2w = jt.linalg.inv((view.world_view_transform.transpose(0,1)))
    W, H = view.image_width, view.image_height
    ndc2pix = jt.array([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().transpose(0,1)
    projection_matrix = c2w.transpose(0,1) @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3,:3].transpose(0,1)
    
    # grid_x, grid_y = jt.meshgrid(jt.arange(W).float(), jt.arange(H).float())
    x = jt.arange(W).float()  # shape: (W,)
    y = jt.arange(H).float()  # shape: (H,)
    grid_x = x.unsqueeze(0).expand(H, W)    # shape: (H, W)
    grid_y = y.unsqueeze(1).expand(H, W)    # shape: (H, W)
    # grid_x, grid_y = jt.meshgrid(jt.arange(W).float(), jt.arange(H).float(), indexing='xy')
    points = jt.stack([grid_x, grid_y, jt.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ (jt.linalg.inv(intrins).transpose(0,1)) @ (c2w[:3,:3].transpose(0,1))
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o

    return points
# def depths_to_points(view, depthmap):
#     c2w = (view.world_view_transform.transpose(0,1)).inverse()
#     W, H = view.image_width, view.image_height
#     ndc2pix = jt.array([
#         [W / 2, 0, 0, (W) / 2],
#         [0, H / 2, 0, (H) / 2],
#         [0, 0, 0, 1]]).float().T
#     projection_matrix = c2w.T @ view.full_proj_transform
#     intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
#     grid_x, grid_y = jt.meshgrid(jt.arange(W).float(), jt.arange(H).float(), indexing='xy')
#     points = jt.stack([grid_x, grid_y, jt.ones_like(grid_x)], dim=-1).reshape(-1, 3)
#     rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
#     rays_o = c2w[:3,3]
#     points = depthmap.reshape(-1, 1) * rays_d + rays_o
#     return points

def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    
    output = jt.zeros_like(points)
    dx = jt.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = jt.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = jt.normalize(jt.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output