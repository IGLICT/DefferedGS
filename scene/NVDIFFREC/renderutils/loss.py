# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import jittor as jt

#----------------------------------------------------------------------------
# HDR image losses
#----------------------------------------------------------------------------

def _tonemap_srgb(f):
    return jt.where(f > 0.0031308, jt.pow(jt.clamp(f, min=0.0031308), 1.0/2.4)*1.055 - 0.055, 12.92*f)

def _SMAPE(img, target, eps=0.01):
    nom = jt.abs(img - target)
    denom = jt.abs(img) + jt.abs(target) + 0.01
    return jt.mean(nom / denom)

def _RELMSE(img, target, eps=0.1):
    nom = (img - target) * (img - target)
    denom = img * img + target * target + 0.1 
    return jt.mean(nom / denom)

def image_loss_fn(img, target, loss, tonemapper):
    if tonemapper == 'log_srgb':
        img    = _tonemap_srgb(jt.log(jt.clamp(img, min=0, max=65535) + 1))
        target = _tonemap_srgb(jt.log(jt.clamp(target, min=0, max=65535) + 1))

    if loss == 'mse':
        return jt.nn.mse_loss(img, target)
    elif loss == 'smape':
        return _SMAPE(img, target)
    elif loss == 'relmse':
        return _RELMSE(img, target)
    else:
        return jt.nn.l1_loss(img, target)
