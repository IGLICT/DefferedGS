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
import jittor as jt
# import torch
# import nvdiffrast.torch as dr
from . import texture as dr
from . import util
from . import renderutils as ru

from icecream import ic
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

######################################################################################
# Utility functions
######################################################################################

# https://github.com/StanfordORB/Stanford-ORB/blob/962ea6d2cced6c9ea076fea4dc33464589036552/orb/utils/env_map.py
def env_map_to_physg(env_map: np.ndarray):
    # change convention from ours (Left +Z, Up +Y) to physg (Left -Z, Up +Y)
    H, W = env_map.shape[:2]
    env_map = np.roll(env_map, W // 2, axis=1)
    return env_map


def env_map_to_cam_to_world_by_convention(envmap: np.ndarray, c2w, convention):
    R = c2w[:3,:3]
    H, W = envmap.shape[:2]
    theta, phi = np.meshgrid(np.linspace(2*np.pi, -0*np.pi, W), np.linspace(0., np.pi, H))
    viewdirs = np.stack([-np.cos(theta) * np.sin(phi), -np.sin(theta) * np.sin(phi), np.cos(phi)],
                           axis=-1).reshape(H*W, 3)    # [H, W, 3]
    viewdirs = (R.T @ viewdirs.T).T.reshape(H, W, 3)
    viewdirs = viewdirs.reshape(H, W, 3)
    # This is correspond to the convention of +Z at left, +Y at top
    # -np.cos(theta) * np.sin(phi), np.cos(phi), -np.sin(theta) * np.sin(phi)
    coord_y = ((np.arccos(viewdirs[..., 1])/np.pi*(H-1)+H)%H).astype(np.float32)
    coord_x = (((np.arctan2(viewdirs[...,0], -viewdirs[...,2])+np.pi)/2/np.pi*(W-1)+W)%W).astype(np.float32)
    envmap_remapped = cv2.remap(envmap, coord_x, coord_y, cv2.INTER_LINEAR)
    if convention == 'ours':
        return envmap_remapped
    if convention == 'physg':
        # change convention from ours (Left +Z, Up +Y) to physg (Left -Z, Up +Y)
        envmap_remapped_physg = np.roll(envmap_remapped, W//2, axis=1)
        return envmap_remapped_physg
    if convention == 'nerd':
        # change convention from ours (Left +Z-X, Up +Y) to nerd (Left +Z+X, Up +Y)
        envmap_remapped_nerd = envmap_remapped[:,::-1,:]
        return envmap_remapped_nerd

    if convention == 'invrender':
        assert convention == 'invrender', convention
        # change convention from ours (Left +Z-X, Up +Y) to invrender (Left -X+Y, Up +Z)
        theta, phi = np.meshgrid(np.linspace(1.0 * np.pi, -1.0 * np.pi, W), np.linspace(0., np.pi, H))
        viewdirs = np.stack([np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(phi)], axis=-1)    # [H, W, 3]
        # viewdirs = np.stack([-viewdirs[...,0], viewdirs[...,2], viewdirs[...,1]], axis=-1)
        coord_y = ((np.arccos(viewdirs[..., 1])/np.pi*(H-1)+H)%H).astype(np.float32)
        coord_x = (((np.arctan2(viewdirs[...,0], -viewdirs[...,2])+np.pi)/2/np.pi*(W-1)+W)%W).astype(np.float32)
        envmap_remapped_Inv = cv2.remap(envmap_remapped, coord_x, coord_y, cv2.INTER_LINEAR)
        return envmap_remapped_Inv

    if convention == 'new':
        R = c2w[:3,:3]
        H, W = envmap.shape[:2]
        theta, phi = np.meshgrid(np.linspace(2*np.pi, -0*np.pi, W), np.linspace(0., np.pi, H))
        viewdirs = np.stack([-np.cos(theta) * np.sin(phi), np.cos(phi), -np.sin(theta) * np.sin(phi)], axis=-1).reshape(H*W, 3) # [H, W, 3]
        viewdirs = (R.T @ viewdirs.T).T.reshape(H, W, 3)
        viewdirs = viewdirs.reshape(H, W, 3)
        # This is correspond to the convention of +Z at left, +Y at top
        # -np.cos(theta) * np.sin(phi), np.cos(phi), -np.sin(theta) * np.sin(phi)
        coord_y = ((np.arccos(viewdirs[..., 1])/np.pi*(H-1)+H)%H).astype(np.float32)
        coord_x = (((np.arctan2(viewdirs[...,0], -viewdirs[...,2])+np.pi)/2/np.pi*(W-1)+W)%W).astype(np.float32)
        envmap_remapped = cv2.remap(envmap, coord_x, coord_y, cv2.INTER_LINEAR)
        return envmap_remapped

# class cubemap_mip(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, cubemap):
#         return util.avg_pool_nhwc(cubemap, (2,2))

#     @staticmethod
#     def backward(ctx, dout):
#         res = dout.shape[1] * 2
#         out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device="cuda")
#         for s in range(6):
#             gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"), 
#                                     torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
#                                     )
#                                     # indexing='ij')
#             v = util.safe_normalize(util.cube_to_dir(s, gx, gy))
#             out[s, ...] = dr.texture(dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
#         return out
class cubemap_mip(jt.Function):

    def execute(self, cubemap):

        return util.avg_pool_nhwc(cubemap, (2,2))
    
    def grad(self, dout):

        res = dout.shape[1] * 2
        out = jt.zeros(6, res, res, dout.shape[-1])
        for s in range(6):
            gy, gx = jt.meshgrid(jt.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res), 
                                    jt.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res),
                                    )
                                    # indexing='ij')
            v = util.safe_normalize(util.cube_to_dir(s, gx, gy))
            out[s, ...] = dr.texture(dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
        return out

######################################################################################
# Split-sum environment map light source with automatic mipmap generation
######################################################################################

class EnvironmentLight(jt.nn.Module):
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(self, base):
        super(EnvironmentLight, self).__init__()
        # super().__init__()
        self.mtx = None      
        # self.base = base.clone()
        self.base = base.clone().detach()
        # self.register_parameter('env_base', self.base)

    def xfm(self, mtx):
        self.mtx = mtx

    def clone(self):
        return EnvironmentLight(self.base.clone().detach())

    def clamp_(self, min=None, max=None):
        self.base.clamp_(min, max)

    def get_mip(self, roughness):
        return jt.where(roughness < self.MAX_ROUGHNESS
                        , (jt.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS) - self.MIN_ROUGHNESS) / (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) * (len(self.specular) - 2)
                        , (jt.clamp(roughness, self.MAX_ROUGHNESS, 1.0) - self.MAX_ROUGHNESS) / (1.0 - self.MAX_ROUGHNESS) + len(self.specular) - 2)
        
    # def build_mips(self, cutoff=0.99):
    #     # print("light.py",self.base)
    #     self.specular = [self.base]
        
    #     while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
    #         # print("out",cubemap_mip.apply(self.specular[-1]).shape)
    #         self.specular += [cubemap_mip.apply(self.specular[-1])]

    #     # print(self.specular[-1])
    #     self.diffuse = ru.diffuse_cubemap(self.specular[-1])
    #     # print(self.diffuse)
    #     for idx in range(len(self.specular) - 1):
    #         roughness = (idx / (len(self.specular) - 2)) * (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) + self.MIN_ROUGHNESS
    #         self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff) 
            
    #     # problem here!!!!
    #     self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)
    def build_mips(self, cutoff=0.99):
        # ic(self.base.isnan().nonzero())
        # ic(self.base)
        nan_mask = jt.isnan(self.base)
        if(jt.all(jt.isfinite(self.base))==False):
            print("change")
            self.base[nan_mask] = self.previous_base.mean()
        else:
            self.previous_base = self.base.detach()
        self.specular = [self.base]
        
        while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
            # print("out",cubemap_mip.apply(self.specular[-1]).shape)
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        # ic(self.specular[-1].shape)
        self.diffuse = ru.diffuse_cubemap(self.specular[-1])
        # print(self.diffuse)
        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) + self.MIN_ROUGHNESS
            self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff) 
            
        # problem here!!!!
        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)

        
        # print(self.specular[-1] )
    def regularizer(self):
        white = (self.base[..., 0:1] + self.base[..., 1:2] + self.base[..., 2:3]) / 3.0
        return jt.mean(jt.abs(self.base - white))
    
    # 20231228 modified by wutong 
    def shade(self, gb_pos, gb_normal, kd, ks, kr, view_pos, specular=True):
        # (H, W, N, 3)
        wo = util.safe_normalize(view_pos - gb_pos)

        if specular:
            diff_col  = kd
            roughness = kr
            spec_col  = ks
        else:
            diff_col = kd

        reflvec = util.safe_normalize(util.reflect(wo, gb_normal))
        nrmvec = gb_normal
        if self.mtx is not None: # Rotate lookup
            # mtx = torch.as_tensor(self.mtx, dtype=torch.float32, device='cuda')
            mtx = jt.array(self.mtx)
            reflvec = ru.xfm_vectors(reflvec.view(reflvec.shape[0], reflvec.shape[1] * reflvec.shape[2], reflvec.shape[3]), mtx).view(*reflvec.shape)
            nrmvec  = ru.xfm_vectors(nrmvec.view(nrmvec.shape[0], nrmvec.shape[1] * nrmvec.shape[2], nrmvec.shape[3]), mtx).view(*nrmvec.shape)
        param1 = self.diffuse[None, ...]
        diffuse = dr.texture(param1, nrmvec.contiguous(), filter_mode='linear', boundary_mode='cube')
        # ic(jt.grad(diffuse,param1).nonzero())
        shaded_diffuse_col = diffuse * diff_col
        extras = {"diffuse": shaded_diffuse_col}

        if specular:
            # Lookup FG term from lookup texture
            NdotV = jt.clamp(util.dot(wo, gb_normal), min_v=1e-4)
            fg_uv = jt.cat((NdotV, roughness), dim=-1)
            if not hasattr(self, '_FG_LUT'):
                # self._FG_LUT = torch.as_tensor(np.fromfile('scene/NVDIFFREC/irrmaps/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device='cuda')
                self._FG_LUT = jt.array(np.fromfile('scene/NVDIFFREC/irrmaps/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2))
            fg_lookup = dr.texture(self._FG_LUT, fg_uv, filter_mode='linear', boundary_mode='clamp')
            
            # Roughness adjusted specular env lookup
            miplevel = self.get_mip(roughness)
            param1 = self.specular[0][None, ...]
            param2  = reflvec.contiguous()
            param3 = list(m[None, ...] for m in self.specular[1:])
            param4 = miplevel[..., 0]
            # spec = dr.texture(self.specular[0][None, ...], reflvec.contiguous(), mip=list(m[None, ...] for m in self.specular[1:]), mip_level_bias=miplevel[..., 0], filter_mode='linear-mipmap-linear', boundary_mode='cube')
            spec = dr.texture(param1, param2, mip=param3, mip_level_bias=param4, filter_mode='linear-mipmap-linear', boundary_mode='cube')
            # ic(jt.grad(spec,param1).nonzero())
            # ic(jt.grad(spec,param2).nonzero())
            # ic(jt.grad(spec,param3[0]).nonzero())
            # ic(jt.grad(spec,param3[1]).nonzero())
            # ic(jt.grad(spec,param4).nonzero())
            # Compute aggregate lighting
            reflectance = spec_col * fg_lookup[...,0:1] + fg_lookup[...,1:2]
            shaded_spec_col = spec * reflectance
            extras["specular"] = shaded_spec_col
        # shaded_col = shaded_diffuse_col + shaded_spec_col
        shaded_col = shaded_diffuse_col + shaded_spec_col
        return shaded_col, extras # Modulate by hemisphere visibility
    
    # GaussianShader orignal
    # def shade1(self, gb_pos, gb_normal, kd, ks, kr, view_pos, specular=True):
    #     # (H, W, N, C)
    #     wo = util.safe_normalize(view_pos - gb_pos)

    #     if specular:
    #         diffuse_raw = kd
    #         roughness = kr
    #         spec_col  = ks
    #         diff_col  = 1.0 - ks
    #     else:
    #         raise NotImplementedError

    #     reflvec = util.safe_normalize(util.reflect(wo, gb_normal))
    #     nrmvec = gb_normal
    #     if self.mtx is not None: # Rotate lookup
    #         mtx = torch.as_tensor(self.mtx, dtype=torch.float32, device='cuda')
    #         reflvec = ru.xfm_vectors(reflvec.view(reflvec.shape[0], reflvec.shape[1] * reflvec.shape[2], reflvec.shape[3]), mtx).view(*reflvec.shape)
    #         nrmvec  = ru.xfm_vectors(nrmvec.view(nrmvec.shape[0], nrmvec.shape[1] * nrmvec.shape[2], nrmvec.shape[3]), mtx).view(*nrmvec.shape)

    #     ambient = dr.texture(self.diffuse[None, ...], nrmvec.contiguous(), filter_mode='linear', boundary_mode='cube')
    #     # specular_linear = ambient * specular_tint
    #     specular_linear = ambient * diff_col

    #     if specular:
    #         # Lookup FG term from lookup texture
    #         NdotV = torch.clamp(util.dot(wo, gb_normal), min=1e-4)
    #         fg_uv = torch.cat((NdotV, roughness), dim=-1)
    #         if not hasattr(self, '_FG_LUT'):
    #             self._FG_LUT = torch.as_tensor(np.fromfile('scene/NVDIFFREC/irrmaps/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device='cuda')
    #         fg_lookup = dr.texture(self._FG_LUT, fg_uv, filter_mode='linear', boundary_mode='clamp')

    #         # Roughness adjusted specular env lookup
    #         miplevel = self.get_mip(roughness)
    #         spec = dr.texture(self.specular[0][None, ...], reflvec.contiguous(), mip=list(m[None, ...] for m in self.specular[1:]), mip_level_bias=miplevel[..., 0], filter_mode='linear-mipmap-linear', boundary_mode='cube')

    #         # Compute aggregate lighting
    #         # reflectance = specular_tint * fg_lookup[...,0:1] + fg_lookup[...,1:2]
    #         reflectance = spec_col * fg_lookup[...,0:1] + fg_lookup[...,1:2]
    #         specular_linear += spec * reflectance
    #     extras = {"specular": specular_linear}

    #     diffuse_linear = torch.sigmoid(diffuse_raw - np.log(3.0))
    #     extras["diffuse"] = diffuse_linear

    #     rgb = specular_linear + diffuse_linear

    #     return rgb, extras

######################################################################################
# Load and store
######################################################################################

# Load from latlong .HDR file
def _load_env_hdr(fn, scale=1.0):
    latlong_img = jt.array(np.array(util.load_image(fn)))*scale
    # latlong_img = jt.array(util.load_image(fn), dtype=torch.float32, device='cuda')*scale
    # latlong_img = latlong_img * 255

    ic(latlong_img.max(), latlong_img.min(), latlong_img.shape, latlong_img.dtype)
    
    cubemap = util.latlong_to_cubemap(latlong_img, [512, 512])

    l = EnvironmentLight(cubemap)
    l.build_mips()

    return l

def _load_env_exr(fn, scale=1.0, c2w=None, convention='ours'):
    import cv2
    # img = cv2.imread(fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(fn)
    # print(img.max(), img.min(), img.shape, img.dtype)
    # if img.dtype == np.float32: # HDR image
    #     print('hdr')
    #     ic(img.max(), img.min())
    #     img = img.clip(0)
    #     ic(img.max(), img.min())
    #     img = img ** (1/2.2)
    #     ic(img.max(), img.min())
    #     img = img.clip(0, 1)
    #     ic(img.max(), img.min())
    #     pass
    # else: # LDR image
    #     print('ldr')
    #     img = img.astype(np.float32) / 255.0
        
    
    with open(fn, 'rb') as h:
        buffer_ = np.frombuffer(h.read(), np.uint8)
    ic(buffer_.max(), buffer_.min())
    bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
    ic(bgr.max(), bgr.min())
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ic(img.max(), img.min())
    
    if c2w is not None:
        img = env_map_to_cam_to_world_by_convention(img, c2w=c2w, convention=convention)
        cv2.imwrite('envmap_{}.hdr'.format(os.path.basename(fn).split('.')[0]), img)
        # raise
        # latlong_img = torch.tensor(img, dtype=torch.float32, device='cuda')*scale
        latlong_img = jt.array(img)*scale
        cubemap = util.latlong_to_cubemap(latlong_img, [512, 512])
    else:
        # latlong_img = torch.tensor(img, dtype=torch.float32, device='cuda')*scale
        latlong_img = jt.array(img)*scale
        cubemap = util.latlong_to_cubemap(latlong_img, [512, 512])

    l = EnvironmentLight(cubemap)
    l.build_mips()

    return l

def load_env(fn, scale=1.0, c2w=None, convention='ours'):
    if os.path.splitext(fn)[1].lower() == ".hdr":
        return _load_env_hdr(fn, scale)
    elif os.path.splitext(fn)[1].lower() == ".exr":
        return _load_env_exr(fn, scale, c2w=c2w, convention=convention)
    else:
        assert False, "Unknown envlight extension %s" % os.path.splitext(fn)[1]

def save_env_map(fn, light):
    assert isinstance(light, EnvironmentLight), "Can only save EnvironmentLight currently"
    if isinstance(light, EnvironmentLight):
        color = util.cubemap_to_latlong(light.base, [512, 1024])
    util.save_image_raw(fn, color.numpy())

######################################################################################
# Create trainable env map with random initialization
######################################################################################

def create_trainable_env_rnd(base_res, scale=0.5, bias=0.25):
    # base = torch.rand(6, base_res, base_res, 3, dtype=torch.float32, device='cuda') * scale + bias
    base = jt.rand((6, base_res, base_res, 3)) * scale + bias
    return EnvironmentLight(base)

# def extract_env_map(light, resolution=[512, 1024]):
#     assert isinstance(light, EnvironmentLight), "Can only save EnvironmentLight currently"
#     color = util.cubemap_to_latlong(light.base, resolution)
#     return color
