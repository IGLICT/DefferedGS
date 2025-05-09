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

# import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
# from torch import nn
import jittor as jt
from jittor import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
# from simple_knn._C import distCUDA2
from scene.simple_knn import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

from scene.NVDIFFREC import create_trainable_env_rnd, load_env

from icecream import ic

def nan_to_num(input, nan=0.0, posinf=None, neginf=None):
    # 处理 NaN
    output = jt.where(jt.isnan(input), jt.array(nan, dtype=input.dtype), input)
    
    # 处理正无穷 (如果没有指定 posinf，则用最大的有限值代替)
    if posinf is None:
        posinf = jt.finfo(input.dtype).max
    output = jt.where(jt.isposinf(output), jt.array(posinf, dtype=input.dtype), output)
    
    # 处理负无穷 (如果没有指定 neginf，则用最小的有限值代替)
    if neginf is None:
        neginf = jt.finfo(input.dtype).min
    output = jt.where(jt.isneginf(output), jt.array(neginf, dtype=input.dtype), output)
    return output

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(jt.concat([scaling * scaling_modifier, jt.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = jt.zeros((center.shape[0], 4, 4), dtype="float")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = jt.exp
        self.scaling_inverse_activation = jt.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = jt.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = jt.normalize
        
        self.diffuse_activation = jt.sigmoid
        self.specular_activation = jt.sigmoid
        self.roughness_activation = jt.sigmoid
        self.specular_1D = False

    def __init__(self, sh_degree : int, brdf_dim : int, brdf_mode : str, brdf_envmap_res : int, default_roughness):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = jt.empty(0)
        self._features_dc = jt.empty(0)
        self._features_rest = jt.empty(0)
        self._scaling = jt.empty(0)
        self._rotation = jt.empty(0)
        self._opacity = jt.empty(0)
        self.max_radii2D = jt.empty(0)
        self.xyz_gradient_accum = jt.empty(0)
        self.denom = jt.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        
        # brdf setting
        self.brdf_dim = brdf_dim  
        self.brdf_mode = brdf_mode  
        self.brdf_envmap_res = brdf_envmap_res
        self._specular = jt.empty(0)
        self._roughness = jt.empty(0)
        self.default_roughness = default_roughness
        self.brdf_mlp = create_trainable_env_rnd(self.brdf_envmap_res, scale=0.0, bias=0.8)
        self.setup_functions()

        # used for texture editing
        self.BVH = None
        self.mesh_3d = None
        self.mesh_2d = None
        self.texture = None
        self.edit_mode = None
        self.edit_mode_2_buffer_dict = {
            'diffuse': 'diffuse_color',
            'specular': 'specular',
            'roughness': 'roughness',
        }

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) #.clamp(max=1)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return jt.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_diffuse(self):
        return self.diffuse_activation(self._features_dc)

    @property
    def get_specular(self):
        original_specular = self.specular_activation(self._specular)
        if self.specular_1D:
            original_specular = original_specular[:, :1].repeat(1, 3)
        return original_specular

    @property
    def get_roughness(self):
        original_roughness = self.roughness_activation(self._roughness)
        return original_roughness
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = jt.array(np.asarray(pcd.points)).float()
        fused_color = RGB2SH(jt.array(np.asarray(pcd.colors)).float())
        # features = jt.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        # features[:, :3, 0 ] = fused_color
        # features[:, 3:, 1:] = 0.0
        
        fused_color = jt.array(np.asarray(pcd.colors)).float()
        # features = jt.zeros((fused_color.shape[0], self.brdf_dim + 3)).float()
        features = jt.zeros((fused_color.shape[0], self.brdf_dim + 3)).float()
        features[:, :3 ] = fused_color
        features[:, 3: ] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = jt.clamp(distCUDA2(jt.array(np.asarray(pcd.points)).float()), min_v=0.0000001)
        scales = jt.log(jt.sqrt(dist2))[...,None].repeat(1, 2)
        rots = jt.rand((fused_point_cloud.shape[0], 4))
        # rots = jt.ones((fused_point_cloud.shape[0], 4))
        opacities = self.inverse_opacity_activation(0.1 * jt.ones((fused_point_cloud.shape[0], 1), dtype="float"))

        self._xyz = fused_point_cloud.clone()
        
        self._features_dc = features[:,:3].contiguous().clone()
        self._features_rest = features[:,3:].contiguous().clone()
        self._specular = jt.zeros((fused_point_cloud.shape[0], 3)).clone()
        self._roughness = self.default_roughness*jt.ones((fused_point_cloud.shape[0], 1)).clone()
        # self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().clone())
        # self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().clone())
        
        self._scaling = scales.clone()
        self._rotation = rots.clone()
        self._opacity = opacities.clone()
        self.max_radii2D = jt.zeros((self.get_xyz.shape[0]))


    def training_setup(self, training_args):
        self.fix_brdf_lr = training_args.fix_brdf_lr
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = jt.zeros((self.get_xyz.shape[0], 1))
        self.denom = jt.zeros((self.get_xyz.shape[0], 1))

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        brdf_list = []
        for name,param in self.brdf_mlp.named_parameters():
            if name == "base":
                self.base = param
                self.previous_base = jt.zeros_like(self.base)
                brdf_list.append(param)
                # list(self.brdf_mlp.parameters())
        l.extend([
                {'params': brdf_list, 'lr': training_args.brdf_mlp_lr_init, "name": "brdf_mlp"},
                {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
                {'params': [self._specular], 'lr': training_args.specular_lr, "name": "specular"}
            ])
        
        self.optimizer = jt.nn.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        self.brdf_mlp_scheduler_args = get_expon_lr_func(lr_init=training_args.brdf_mlp_lr_init,
                                        lr_final=training_args.brdf_mlp_lr_final,
                                        lr_delay_mult=training_args.brdf_mlp_lr_delay_mult,
                                        max_steps=training_args.brdf_mlp_lr_max_steps)

    def reset_viewspace_point(self):
        self.screenspace_points = jt.zeros_like(self.get_xyz, dtype=self.get_xyz.dtype) + 0
        pg = self.optimizer.param_groups[-1]
        if pg["name"] == "screenspace_points":
            self.optimizer.param_groups.pop()
        self.optimizer.add_param_group(
            {'params': [self.screenspace_points], 'lr':0., "name": "screenspace_points"}
        )
    def get_viewspace_point_grad(self):
        pg = self.optimizer.param_groups[-1]
        if pg["name"] == "screenspace_points":
            # breakpoint()
            return pg["grads"][0]
        else:
            assert False, "No viewspace_point_grad found"
    def _update_learning_rate(self, iteration, param):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == param:
                try:
                    lr = getattr(self, f"{param}_scheduler_args", self.brdf_mlp_scheduler_args)(iteration)
                    param_group['lr'] = lr
                    return lr
                except AttributeError:
                    pass
                
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        self._update_learning_rate(iteration, "xyz")
        if not self.fix_brdf_lr:
            for param in ["brdf_mlp","roughness","specular","f_dc"]:
                lr = self._update_learning_rate(iteration, param)

    def construct_list_of_attributes(self, viewer_fmt=False):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
            
        if viewer_fmt:
            features_rest_len = 45
        elif (self.brdf_mode=="envmap" and self.brdf_dim==0):
            features_rest_len = self._features_rest.shape[1]
        for i in range(features_rest_len):
            l.append('f_rest_{}'.format(i))
                
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
            
        if not viewer_fmt:
            l.append('roughness')
            for i in range(self._specular.shape[1]):
                l.append('specular{}'.format(i))
        return l

    def save_ply(self, path, viewer_fmt=False):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().numpy()
        normals = np.zeros_like(xyz)
        # f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        
        f_dc = self._features_dc.detach().numpy()
        f_rest = self._features_rest.detach().numpy()
        if viewer_fmt:
            f_rest = np.zeros((f_rest.shape[0], 45))
            
        opacities = self._opacity.detach().numpy()
        scale = self._scaling.detach().numpy()
        rotation = self._rotation.detach().numpy()
        roughness = self._roughness.detach().numpy()
        specular = self._specular.detach().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(viewer_fmt)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        if not viewer_fmt:
            ic(xyz.shape, normals.shape, f_dc.shape, f_rest.shape, opacities.shape, scale.shape, rotation.shape, roughness.shape, specular.shape)
            if len(f_rest.shape) == 3:
                f_rest = f_rest.reshape((f_rest.shape[0], f_rest.shape[1]*f_rest.shape[2]))
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, roughness, specular), axis=1)
        else:
            ic(xyz.shape, normals.shape, f_dc.shape, f_rest.shape, opacities.shape, scale.shape, rotation.shape)
            if len(f_rest.shape) == 3:
                f_rest = f_rest.reshape((f_rest.shape[0], f_rest.shape[1]*f_rest.shape[2]))
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
            
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        def min(a,b):
            return jt.where(a<b,a,b)
        opacities_new = self.inverse_opacity_activation(min(self.get_opacity, jt.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        jt.sync()
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # features_dc = np.zeros((xyz.shape[0], 3, 1))
        # features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        # features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        # features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        features_dc = np.zeros((xyz.shape[0], 3))
        features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        
        # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        # features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        # for idx, attr_name in enumerate(extra_f_names):
        #     features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        # features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        
        if self.brdf_mode=="envmap":
            features_extra = np.zeros((xyz.shape[0], 3*(self.brdf_dim + 1) ** 2 ))
            if len(extra_f_names)==3*(self.brdf_dim + 1) ** 2:
                for idx, attr_name in enumerate(extra_f_names):
                    features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
                features_extra = features_extra.reshape((features_extra.shape[0], (self.brdf_dim + 1) ** 2, 3))
                features_extra = features_extra.swapaxes(1,2)
            else:
                print(f"NO INITIAL SH FEATURES FOUND!!! USE ZERO SH AS INITIALIZE.")
                features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.brdf_dim + 1) ** 2))
        
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = jt.array(xyz, dtype=jt.float).clone()
        # self._features_dc = nn.Parameter(jt.array(features_dc, dtype=jt.float).transpose(1, 2).contiguous().clone())
        # self._features_rest = nn.Parameter(jt.array(features_extra, dtype=jt.float).transpose(1, 2).contiguous().clone())
        self._features_dc = jt.array(features_dc, dtype=jt.float).contiguous().clone()
        self._features_rest = jt.array(features_extra, dtype=jt.float).clone()
        
        self._opacity = jt.array(opacities, dtype=jt.float).clone()
        self._scaling = jt.array(scales, dtype=jt.float).clone()
        self._rotation = jt.array(rots, dtype=jt.float).clone()
        self.screenspace_points = jt.zeros_like(self.get_xyz, dtype=self.get_xyz.dtype) + 0

        roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]
        specular_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("specular")]
        specular = np.zeros((xyz.shape[0], len(specular_names)))
        for idx, attr_name in enumerate(specular_names):
            specular[:, idx] = np.asarray(plydata.elements[0][attr_name])
        self._roughness = jt.array(roughness).clone()
        self._specular = jt.array(specular).clone()
            
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == 'screenspace_points': 
                continue
            if group["name"] == "brdf_mlp":
                continue
            if group["name"] == name:
                with jt.enable_grad():
                    group["params"][0] = tensor.copy()
                group["m"][0] = jt.zeros_like(tensor)
                group["values"][0] = jt.zeros_like(tensor)
                optimizable_tensors[group["name"]] = group["params"][0]
            # if group["name"] == name:
            #     stored_state = self.optimizer.state.get(group['params'][0], None)
            #     stored_state["exp_avg"] = jt.zeros_like(tensor)
            #     stored_state["exp_avg_sq"] = jt.zeros_like(tensor)

            #     del self.optimizer.state[group['params'][0]]
            #     group["params"][0] = tensor.clone()
            #     self.optimizer.state[group['params'][0]] = stored_state

            #     optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == 'screenspace_points': 
                continue
            if group["name"] == "brdf_mlp":
                continue
            if group['params'][0] is not None:

                group['m'][0].update(group['m'][0][mask])
                group['values'][0].update(group['values'][0][mask])
                with jt.enable_grad():
                    old = group["params"].pop()
                    group["params"].append(old[mask])
                    del old
                optimizable_tensors[group["name"]] = group["params"][0]
            # stored_state = self.optimizer.state.get(group['params'][0], None)
            # if stored_state is not None:
            #     stored_state["exp_avg"] = stored_state["exp_avg"][mask]
            #     stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

            #     del self.optimizer.state[group['params'][0]]
            #     group["params"][0] = (group["params"][0][mask].clone())
            #     self.optimizer.state[group['params'][0]] = stored_state

            #     optimizable_tensors[group["name"]] = group["params"][0]
            # else:
            #     group["params"][0] = group["params"][0][mask].clone()
            #     optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = mask.logical_not()
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        self._roughness = optimizable_tensors["roughness"]
        self._specular = optimizable_tensors["specular"]
            
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "brdf_mlp":
                continue
            if group["name"] == 'screenspace_points': 
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            group["m"][0] = jt.concat((group["m"][0], jt.zeros_like(extension_tensor)), dim=0)
            group["values"][0] = jt.concat((group["values"][0], jt.zeros_like(extension_tensor)), dim=0)
            old_tensor = group["params"].pop()
            with jt.enable_grad():
                group["params"].append(jt.concat((old_tensor, extension_tensor), dim=0))
                del old_tensor
            optimizable_tensors[group["name"]] = group["params"][0]
            # stored_state = self.optimizer.state.get(group['params'][0], None)
            # if stored_state is not None:

            #     stored_state["exp_avg"] = jt.cat((stored_state["exp_avg"], jt.zeros_like(extension_tensor)), dim=0)
            #     stored_state["exp_avg_sq"] = jt.cat((stored_state["exp_avg_sq"], jt.zeros_like(extension_tensor)), dim=0)

            #     del self.optimizer.state[group['params'][0]]
            #     group["params"][0] = jt.cat((group["params"][0], extension_tensor), dim=0).clone()
            #     self.optimizer.state[group['params'][0]] = stored_state

            #     optimizable_tensors[group["name"]] = group["params"][0]
            # else:
            #     group["params"][0] = jt.cat((group["params"][0], extension_tensor), dim=0).clone()
            #     optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_roughness, new_specular):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        d.update({
            "roughness": new_roughness,
            "specular" : new_specular,})

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._roughness = optimizable_tensors["roughness"]
        self._specular = optimizable_tensors["specular"]
            
        self.xyz_gradient_accum = jt.zeros((self.get_xyz.shape[0], 1))
        self.denom = jt.zeros((self.get_xyz.shape[0], 1)).int()
        self.max_radii2D = jt.zeros((self.get_xyz.shape[0]))

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = jt.zeros((n_init_points))
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = jt.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = jt.logical_and(selected_pts_mask,
                                              jt.max(self.get_scaling, dim=1) > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = jt.cat([stds, 0 * jt.ones_like(stds[:,:1])], dim=-1)
        means = jt.zeros_like(stds)
        samples = jt.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = jt.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        
        new_roughness = self._roughness[selected_pts_mask].repeat(N,1)
        new_specular = self._specular[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_roughness, new_specular)

        prune_filter = jt.concat([selected_pts_mask, jt.zeros((N * selected_pts_mask.sum().item(),), dtype=bool)],dim=0)
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = jt.where(jt.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = jt.logical_and(selected_pts_mask,
                                              jt.max(self.get_scaling, dim=1) <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_roughness = self._roughness[selected_pts_mask]
        new_specular = self._specular[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_roughness, new_specular)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1) > 0.1 * extent
            prune_mask = jt.logical_or(jt.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        jt.gc()
        # jt.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor_grad, update_filter):

        # ic(self.denom[update_filter].shape)
        # ic(jt.norm(viewspace_point_tensor_grad[update_filter,:2], dim=-1, keepdim=True).shape)
        # ic(self.xyz_gradient_accum[update_filter].shape)
        
        tmp = jt.norm(viewspace_point_tensor_grad[update_filter,:2], dim=-1, keepdim=True)
        # self.xyz_gradient_accum.sync()
        # tmp.sync()
        self.xyz_gradient_accum[update_filter] = tmp + self.xyz_gradient_accum[update_filter]
        # self.denom.sync()
        self.denom[update_filter] = 1 + self.denom[update_filter]
        # self.denom.sync()
        # self.xyz_gradient_accum.sync()
    def reset_env_from_nan(self):
        pass
        # nan_mask = jt.isnan(self.base)
        # if(jt.all(jt.isfinite(self.base))==False):
        #     print("change")
        #     self.base[nan_mask] = self.previous_base[nan_mask]
        # else:
        #     print((self.base-self.previous_base).nonzero().shape)
        #     self.previous_base = self.base.detach()

    # def densify_and_split_in_texture_edit(self, selected_pts_mask, N=4):
    #     if torch.sum(selected_pts_mask) == 0:
    #         return

    #     stds = self.get_scaling[selected_pts_mask].repeat(N,1)
    #     means =torch.zeros((stds.size(0), 3),device="cuda")
    #     samples = torch.normal(mean=means, std=stds)
    #     rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
    #     new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
    #     new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
    #     new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
    #     new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1) if not self.brdf else self._features_dc[selected_pts_mask].repeat(N,1)
    #     new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
    #     new_opacities = self._opacity[selected_pts_mask].repeat(N,1)
    #     new_roughness = self._roughness[selected_pts_mask].repeat(N,1) if self.brdf else None
    #     new_specular = self._specular[selected_pts_mask].repeat(N,1) if self.brdf else None
    #     new_normal = self._normal[selected_pts_mask].repeat(N,1) if self.brdf else None
    #     new_normal2 = self._normal2[selected_pts_mask].repeat(N,1) if (self.brdf) else None
    #     self.densification_postfix_in_texture_edit(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, 
    #                                new_roughness, new_specular, new_normal, new_normal2)

    # def densification_postfix_in_texture_edit(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, \
    #                           new_roughness, new_specular, new_normal, new_normal2):
    #     d = {"xyz": new_xyz,
    #     "f_dc": new_features_dc,
    #     "f_rest": new_features_rest,
    #     "opacity": new_opacities,
    #     "scaling" : new_scaling,
    #     "rotation" : new_rotation}
    #     if self.brdf:
    #         d.update({
    #             "roughness": new_roughness,
    #             "specular" : new_specular,
    #             "normal" : new_normal,
    #             "normal2" : new_normal2,
    #         })

    #     self._xyz = nn.Parameter(torch.cat((self._xyz, d['xyz']), dim=0).requires_grad_(True))
    #     self._features_dc = nn.Parameter(torch.cat((self._features_dc, d['f_dc']), dim=0).requires_grad_(True))
    #     self._features_rest = nn.Parameter(torch.cat((self._features_rest, d['f_rest']), dim=0).requires_grad_(True))
    #     self._opacity = nn.Parameter(torch.cat((self._opacity, d['opacity']), dim=0).requires_grad_(True))
    #     self._scaling = nn.Parameter(torch.cat((self._scaling, d['scaling']), dim=0).requires_grad_(True))
    #     self._rotation = nn.Parameter(torch.cat((self._rotation, d['rotation']), dim=0).requires_grad_(True))
        
    #     if self.brdf:
    #         self._roughness = nn.Parameter(torch.cat((self._roughness, d['roughness']), dim=0).requires_grad_(True))
    #         self._specular = nn.Parameter(torch.cat((self._specular, d['specular']), dim=0).requires_grad_(True))
    #         self._normal = nn.Parameter(torch.cat((self._normal, d['normal']), dim=0).requires_grad_(True))
    #         self._normal2 = nn.Parameter(torch.cat((self._normal2, d['normal2']), dim=0).requires_grad_(True))
    #     self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
    #     self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
    #     self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    # def prune_points_in_texture_edit(self, mask):
    #     valid_points_mask = ~mask
    #     d = {"xyz": self.get_xyz,
    #     "f_dc": self._features_dc,
    #     "f_rest": self._features_rest,
    #     "opacity": self._opacity,
    #     "scaling" : self.scaling_inverse_activation(self.get_scaling),
    #     "rotation" : self._rotation}
    #     if self.brdf:
    #         d.update({
    #             "roughness": self._roughness,
    #             "specular" : self._specular,
    #             "normal" : self._normal,
    #             "normal2" : self._normal2,
    #         })

    #     self._xyz = nn.Parameter(d['xyz'][valid_points_mask].requires_grad_(True))
    #     self._features_dc = nn.Parameter(d['f_dc'][valid_points_mask].requires_grad_(True))
    #     self._features_rest = nn.Parameter(d["f_rest"][valid_points_mask].requires_grad_(True))
    #     self._opacity = nn.Parameter(d["opacity"][valid_points_mask].requires_grad_(True))
    #     self._scaling = nn.Parameter(d["scaling"][valid_points_mask].requires_grad_(True))
    #     self._rotation = nn.Parameter(d["rotation"][valid_points_mask].requires_grad_(True))
    #     if self.brdf:
    #         self._roughness = nn.Parameter(d["roughness"][valid_points_mask].requires_grad_(True))
    #         self._specular = nn.Parameter(d["specular"][valid_points_mask].requires_grad_(True))
    #         self._normal = nn.Parameter(d["normal"][valid_points_mask].requires_grad_(True))
    #         self._normal2 = nn.Parameter(d["normal2"][valid_points_mask].requires_grad_(True))

    #     self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

    #     self.denom = self.denom[valid_points_mask]
    #     self.max_radii2D = self.max_radii2D[valid_points_mask]
    
    # def prune_in_texture_edit(self, min_opacity, extent, max_screen_size):
    #     prune_mask = (self.get_opacity < min_opacity).squeeze()
    #     if max_screen_size:
    #         big_points_vs = self.max_radii2D > max_screen_size
    #         big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
    #         prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
    #     self.prune_points_in_texture_edit(prune_mask)

    #     torch.cuda.empty_cache()