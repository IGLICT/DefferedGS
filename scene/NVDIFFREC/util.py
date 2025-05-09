# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import numpy as np
# import torch
import jittor as jt
# import nvdiffrast.torch as dr
import imageio
from . import texture as dr

#----------------------------------------------------------------------------
# Vector operations
#----------------------------------------------------------------------------

def dot(x: jt.Var, y: jt.Var) -> jt.Var:
    return jt.sum(x*y, -1, keepdim=True)

def reflect(x: jt.Var, n: jt.Var) -> jt.Var:
    return 2*dot(x, n)*n - x

def length(x: jt.Var, eps: float =1e-20) -> jt.Var:
    return jt.sqrt(jt.clamp(dot(x,x), min_v=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: jt.Var, eps: float =1e-20) -> jt.Var:
    return x / length(x, eps)

def to_hvec(x: jt.Var, w: float) -> jt.Var:
    return jt.nn.pad(x, pad=(0,1), mode='constant', value=w)

def avg_pool_nhwc(x  : jt.Var, size) -> jt.Var:
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    y = jt.nn.avg_pool2d(y, size)
    
    return y.permute(0, 2, 3, 1) # NCHW -> NHWC

def cube_to_dir(s, x, y):
    if s == 0:   rx, ry, rz = jt.ones_like(x), -y, -x
    elif s == 1: rx, ry, rz = -jt.ones_like(x), -y, x
    elif s == 2: rx, ry, rz = x, jt.ones_like(x), y
    elif s == 3: rx, ry, rz = x, -jt.ones_like(x), -y
    elif s == 4: rx, ry, rz = x, -y, jt.ones_like(x)
    elif s == 5: rx, ry, rz = -x, -y, -jt.ones_like(x)
    return jt.stack((rx, ry, rz), dim=-1)

#----------------------------------------------------------------------------
# Image save/load helper.
#----------------------------------------------------------------------------

def save_image(fn, x : np.ndarray):
    try:
        if os.path.splitext(fn)[1] == ".png":
            imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8), compress_level=3) # Low compression for faster saving
        else:
            imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8))
    except:
        print("WARNING: FAILED to save image %s" % fn)

# def save_image_raw(fn, x : np.ndarray):
#     try:
#         imageio.imwrite(fn, x)
#     except:
#         print("WARNING: FAILED to save image %s" % fn)
def save_image_raw(fn, x : np.ndarray):
    imageio.imwrite(fn, x)



def load_image_raw(fn) -> np.ndarray:

    return imageio.imread(fn)

def load_image(fn) -> np.ndarray:
    img = load_image_raw(fn)
    if img.dtype == np.float32: # HDR image
        return img
    else: # LDR image
        return img.astype(np.float32) / 255

def latlong_to_cubemap(latlong_map, res):
    cubemap = jt.zeros(6, res[0], res[1], latlong_map.shape[-1])
    for s in range(6):
        # gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'), 
        #                         torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'), 
        #                         )
        gy, gx = jt.meshgrid(jt.linspace(-1.0, 1.0, res[0]), 
                                jt.linspace(-1.0, 1.0, res[1]), 
                                )
        v = safe_normalize(cube_to_dir(s, gx, gy))

        tu = jt.atan2(v[..., 0:1], v[..., 1:2]) / (2 * np.pi) + 0.25
        tv = jt.acos(jt.clamp(v[..., 2:3], min_v=-1, max_v=1)) / np.pi
        texcoord = jt.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode='linear')
    return cubemap 
def cubemap_to_latlong(cubemap, res):
    gy, gx = jt.meshgrid(jt.linspace( 0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0]), 
                            jt.linspace(1.5 - 1.0 / res[1], -0.5 + 1.0 / res[1], res[1]),
                            # indexing='ij')
                            )
    
    sintheta, costheta = jt.sin(gy*np.pi), jt.cos(gy*np.pi)
    sinphi, cosphi     = jt.sin(gx*np.pi), jt.cos(gx*np.pi)
    
    reflvec = jt.stack((
        sintheta*sinphi, 
        -sintheta*cosphi,
        costheta, 
        ), dim=-1)
    return dr.texture(cubemap[None, ...], reflvec[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
#-----------------------------------------------------------------------------------
# Metrics (taken from jaxNerf source code, in order to replicate their measurements)
#
# https://github.com/google-research/google-research/blob/301451a62102b046bbeebff49a760ebeec9707b8/jaxnerf/nerf/utils.py#L266
#
#-----------------------------------------------------------------------------------

def mse_to_psnr(mse):
  """Compute PSNR given an MSE (we assume the maximum pixel value is 1)."""
  return -10. / np.log(10.) * np.log(mse)

def psnr_to_mse(psnr):
  """Compute MSE given a PSNR (we assume the maximum pixel value is 1)."""
  return np.exp(-0.1 * np.log(10.) * psnr)


#----------------------------------------------------------------------------
# Displacement texture lookup
#----------------------------------------------------------------------------

def get_miplevels(texture: np.ndarray) -> float:
    minDim = min(texture.shape[0], texture.shape[1])
    return np.floor(np.log2(minDim))

def tex_2d(tex_map, coords, filter='nearest'):
    tex_map = tex_map[None, ...]    # Add batch dimension
    tex_map = tex_map.permute(0, 3, 1, 2) # NHWC -> NCHW
    tex = jt.nn.grid_sample(tex_map, coords[None, None, ...] * 2 - 1, mode=filter, align_corners=False)
    tex = tex.permute(0, 2, 3, 1) # NCHW -> NHWC
    return tex[0, 0, ...]