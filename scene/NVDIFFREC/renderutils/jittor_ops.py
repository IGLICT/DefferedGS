

import numpy as np
import os
import sys
import jittor as jt

from .bsdf import *
from .loss import *

cuda_header = """
#include "bsdf.h"
#include "common.h"
#include "cubemap.h"
#include "mesh.h"
#include "normal.h"
#include "tensor.h"
#include "vec3f.h"
#include "vec4f.h"
#include <cuda_runtime.h>
#include "var.h"
#define BLOCK_X 8
#define BLOCK_Y 8



Tensor make_cuda_tensor(jittor::Var*& val)
{
    Tensor res;
    for (int i = 0; i < val->shape.size(); ++i)
    {
        res.dims[i] = val->shape[i];
    }
    res.strides[val->shape.size() - 1] = 1;
    for (int i = val->shape.size() - 2; i >= 0; --i) {
        res.strides[i] = res.strides[i + 1] * (val->shape[i + 1]);
    }
    // res.fp16 = val.scalar_type() == torch::kBFloat16;
    res.fp16 = !val->is_float();
    res.val = res.fp16 ? val->ptr<float>() : val->ptr<float>();
    res.d_val = nullptr;
    return res;
}

Tensor make_cuda_tensor(jittor::Var*& val, dim3 outDims)
{
    Tensor res;
    for (int i = 0; i < val->shape.size(); ++i)
    {
        res.dims[i] = val->shape[i];
        
    }
    res.strides[val->shape.size() - 1] = 1;
    for (int i = val->shape.size() - 2; i >= 0; --i) {
        res.strides[i] = res.strides[i + 1] * (val->shape[i + 1]);
    }
    
    if (val->shape.size() == 4)
        res._dims[0] = outDims.z, res._dims[1] = outDims.y, res._dims[2] = outDims.x, res._dims[3] = val->shape[3];
    else
        res._dims[0] = outDims.z, res._dims[1] = outDims.x, res._dims[2] = val->shape[2], res._dims[3] = 1; // Add a trailing one for indexing math to work out
    res.fp16 = !val->is_float();
    res.val = res.fp16 ? val->ptr<float>() : val->ptr<float>();
    res.d_val = nullptr;
    return res;
}

void update_grid(dim3 &gridSize, jittor::Var*& x)
{
    gridSize.x = std::max(gridSize.x, (uint32_t)x->shape[2]);
    gridSize.y = std::max(gridSize.y, (uint32_t)x->shape[1]);
    gridSize.z = std::max(gridSize.z, (uint32_t)x->shape[0]);
}

template<typename... Ts>
void update_grid(dim3& gridSize, jittor::Var*& x, Ts&&... vs)
{
    gridSize.x = std::max(gridSize.x, (uint32_t)x->shape[2]);
    gridSize.y = std::max(gridSize.y, (uint32_t)x->shape[1]);
    gridSize.z = std::max(gridSize.z, (uint32_t)x->shape[0]);
    update_grid(gridSize, std::forward<Ts>(vs)...);
}
"""


header_path = os.path.join(os.path.dirname(__file__), 'c_src')
proj_options = {f'FLAGS: -I{header_path} -l"RenderUtilsJittor" -L"{os.path.dirname(__file__)}"':1}
#----------------------------------------------------------------------------
# cubemap filter with filtering across edges

class _diffuse_cubemap_func(jt.Function):
    def save_for_backward(self,*args):
        self.saved_tensors = args

    def execute(self, cubemap):
        out = diffuse_cubemap_fwd(cubemap)
        self.save_for_backward(cubemap)
        return out


    def grad(self, dout):
        cubemap, = self.saved_tensors
        cubemap_grad = diffuse_cubemap_bwd(cubemap, dout)
        return cubemap_grad

def diffuse_cubemap(cubemap, use_python=False):
    if use_python:
        assert False
    else:
        out = _diffuse_cubemap_func.apply(cubemap)
        # out[jt.isfinite(out).logical_not()] = 0.0
        if(jt.all(jt.isfinite(out))==False):
            print(jt.all(jt.isfinite(cubemap)))
            assert jt.all(jt.isfinite(out)), "Output of diffuse_cubemap contains inf or NaN"
    return out

class _specular_cubemap(jt.Function):
    def save_for_backward(self,*args):
        self.saved_tensors = args

    def execute(self, cubemap, roughness, costheta_cutoff, bounds):
        out = specular_cubemap_fwd(cubemap, bounds, roughness, costheta_cutoff)
        self.save_for_backward(cubemap, bounds)
        self.roughness, self.theta_cutoff = roughness, costheta_cutoff
        return out

    def grad(self, dout):
        cubemap, bounds = self.saved_tensors
        cubemap_grad = specular_cubemap_bwd(cubemap, bounds, dout, self.roughness, self.theta_cutoff)
        return cubemap_grad, None, None, None

# Compute the bounds of the GGX NDF lobe to retain "cutoff" percent of the energy
def __ndfBounds(res, roughness, cutoff):
    def ndfGGX(alphaSqr, costheta):
        costheta = np.clip(costheta, 0.0, 1.0)
        d = (costheta * alphaSqr - costheta) * costheta + 1.0
        return alphaSqr / (d * d * np.pi)
    # Sample out cutoff angle
    nSamples = 1000000
    costheta = np.cos(np.linspace(0, np.pi/2.0, nSamples))
    D = np.cumsum(ndfGGX(roughness**4, costheta))
    idx = np.argmax(D >= D[..., -1] * cutoff)
    # Brute force compute lookup table with bounds
    bounds = specular_bounds(res, costheta[idx])
    return costheta[idx], bounds

__ndfBoundsDict = {}

def specular_cubemap(cubemap, roughness, cutoff=0.99, use_python=False):
    assert cubemap.shape[0] == 6 and cubemap.shape[1] == cubemap.shape[2], "Bad shape for cubemap tensor: %s" % str(cubemap.shape)

    if use_python:
        assert False
    else:
        key = (cubemap.shape[1], roughness, cutoff)
        if key not in __ndfBoundsDict:
            __ndfBoundsDict[key] = __ndfBounds(*key)
        out = _specular_cubemap.apply(cubemap, roughness, *__ndfBoundsDict[key])
        # out[jt.isfinite(out)] = 0.0
        # print("ss",out[..., 0:3])
        # print("ss",out[..., 3:])
        # assert jt.all(jt.isfinite(out)), "Output of specular_cubemap contains inf or NaN"
    ans = out[..., 0:3] / out[..., 3:]
    if(jt.all(jt.isfinite(out))==False):
        print("specular")
        print(jt.all(jt.isfinite(cubemap)))
        assert jt.all(jt.isfinite(cubemap)), "Output of diffuse_cubemap contains inf or NaN"
    # ans[jt.isfinite(ans).logical_not()] = 0.0
    return ans



#----------------------------------------------------------------------------
# Transform points function

class _xfm_func(jt.Function):
    def save_for_backward(self,*args):
        self.saved_tensors = args    
    def forward(self, points, matrix, isPoints):
        self.save_for_backward(points, matrix)
        self.isPoints = isPoints
        return xfm_fwd(points, matrix, isPoints, False)

    def backward(self, dout):
        points, matrix = self.saved_tensors
        return (xfm_bwd(points, matrix, dout, self.isPoints),) + (None, None, None)

def xfm_points(points, matrix, use_python=False):
    '''Transform points.
    Args:
        points: Tensor containing 3D points with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use Pyjt's jt.matmul (for validation)
    Returns:
        Transformed points in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    '''    
    if use_python:
        out = jt.matmul(jt.nn.pad(points, pad=(0,1), mode='constant', value=1.0), jt.transpose(matrix, 1, 2))
    else:
        out = _xfm_func.apply(points, matrix, True)
    assert jt.all(jt.isfinite(out)), "Output of xfm_points contains inf or NaN"
    return out

def xfm_vectors(vectors, matrix, use_python=False):
    '''Transform vectors.
    Args:
        vectors: Tensor containing 3D vectors with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use Pyjt's jt.matmul (for validation)

    Returns:
        Transformed vectors in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    '''    

    if use_python:
        out = jt.matmul(jt.functional.pad(vectors, pad=(0,1), mode='constant', value=0.0), jt.transpose(matrix, 1, 2))[..., 0:3].contiguous()
    else:
        out = _xfm_func.apply(vectors, matrix, False)
    assert jt.all(jt.isfinite(out)), "Output of xfm_vectors contains inf or NaN"
    return out


import numpy as np

class DiffuseCubemapKernelParams:
    def __init__(self, cubemap=jt.array([]), out=jt.array([]), gridSize=[0,0,0]):
        self.cubemap = cubemap  # 假设是 numpy 数组
        self.out = out          # 假设是 numpy 数组
        self.gridSize = gridSize  # 使用元组表示 dim3

class SpecularCubemapKernelParams:
    def __init__(self, cubemap=jt.array([]), bounds=jt.array([]), out=jt.array([]), gridSize=[0,0,0], costheta_cutoff=0.0, roughness=0.0):
        self.cubemap = cubemap  # 假设是 numpy 数组
        self.bounds = bounds    # 假设是 numpy 数组
        self.out = out          # 假设是 numpy 数组
        self.gridSize = gridSize  # 使用元组表示 dim3
        self.costheta_cutoff = costheta_cutoff  # float
        self.roughness = roughness  # float

class SpecularBoundsKernelParams:
    def __init__(self, costheta_cutoff=0.0, out=jt.array([]), gridSize=[0,0,0]):
        self.costheta_cutoff = costheta_cutoff  # float
        self.out = out          # 假设是 numpy 数组
        self.gridSize = gridSize  # 使用元组表示 dim3


def update_grid(gridSize,x):
    gridSize = [0,0,0]
    gridSize[0] = max(gridSize[0],x.shape[2])
    gridSize[1] = max(gridSize[1],x.shape[1])
    gridSize[2] = max(gridSize[2],x.shape[0])
    return gridSize

def update_grid_variadic(gridSize, *tensors):
    for x in tensors:
        gridSize = update_grid(gridSize, x)
    return gridSize


def getLaunchBlockSize(max_width, max_height, dims):
    max_threads = max_width * max_height
    if max_threads <= 1 or (dims[0] * dims[1]) <= 1:
        return (1, 1, 1)  # Degenerate case
    # Start from max size
    bw = max_width
    bh = max_height

    # Optimizations for weirdly sized buffers
    if dims[0] < bw:
        # Decrease block width to smallest power of two that covers the buffer width
        while (bw >> 1) >= dims[0]:
            bw >>= 1
        # Maximize height
        bh = max_threads // bw
        if bh > dims[1]:
            bh = dims[1]
    elif dims[1] < bh:
        # Halve height and double width until fits completely inside buffer vertically
        while bh > dims[1]:
            bh >>= 1
            if bw < dims[0]:
                bw <<= 1
    # Done
    return (bw, bh, 1)

def getWarpSize(block_size):
    # 计算 warp 大小
    warp_x = min(block_size[0], 32)
    warp_y = min(max(32 // block_size[0], 1), min(32, block_size[1]))
    warp_z = min(max(32 // (block_size[0] * block_size[1]), 1), min(32, block_size[2]))
    
    return (warp_x, warp_y, warp_z)

def getLaunchGridSize(block_size, dims):
    # 计算网格大小
    grid_x = (dims[0] - 1) // block_size[0] + 1
    grid_y = (dims[1] - 1) // block_size[1] + 1
    grid_z = (dims[2] - 1) // block_size[2] + 1
    return (grid_x, grid_y, grid_z)

def diffuse_cubemap_fwd(incubemap):

    p = DiffuseCubemapKernelParams()
    p.gridSize = update_grid(p.gridSize,incubemap)
    # print(incubemap)
    out_jittor = jt.ones((p.gridSize[2],p.gridSize[1],p.gridSize[0],3),dtype = "float32")
    # blockSize = getLaunchBlockSize(8, 8, p.gridSize)
    # gridSize = getLaunchGridSize(blockSize, p.gridSize)
    jt.flags.use_cuda=1
    with jt.flag_scope(compile_options=proj_options):
        # print("in",out_jittor.shape)
        out_jittor,=jt.code(
                outputs=[out_jittor],
                inputs=[incubemap],
                cuda_header=cuda_header,
                cuda_src='''
            @alias(incubemap, in0)
            @alias(out_jittor, out0)
            DiffuseCubemapKernelParams p;
            update_grid(p.gridSize, incubemap);
            dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
            dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);
            p.cubemap = make_cuda_tensor(in0, p.gridSize);
            #undef out
            p.out = make_cuda_tensor(out0, p.gridSize);
            void* args[] = { &p };
            cudaStream_t stream;
            cudaStreamCreate(&stream);
            cudaLaunchKernel((const void*)DiffuseCubemapFwdKernel, gridSize, blockSize, args, 0, stream);

            // DiffuseCubemapFwdKernel<<<gridSize, blockSize, 0, 0>>>(p);
            // TestKernel<<<1, 1>>>();
            // cudaStreamSynchronize(stream);
        ''')
        # print("out",out_jittor)
    return out_jittor

def diffuse_cubemap_bwd(incubemap,ingrad):
    p = DiffuseCubemapKernelParams()
    p.gridSize = update_grid(p.gridSize,incubemap)
    out_jittor = jt.zeros((p.gridSize[2],p.gridSize[1],p.gridSize[0],incubemap.shape[3]),dtype = "float32")

    # blockSize = getLaunchBlockSize(8, 8, p.gridSize)
    # gridSize = getLaunchGridSize(blockSize, p.gridSize)
    with jt.flag_scope(compile_options=proj_options):
        out_jittor=jt.code(
                outputs=[out_jittor],
                inputs=[incubemap,ingrad],
                cuda_header=cuda_header,
                cuda_src='''
            @alias(incubemap, in0)
            @alias(ingrad, in1)
            @alias(out_jittor, out0)
            DiffuseCubemapKernelParams p;
            update_grid(p.gridSize, incubemap);
            dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
            dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);
            #undef out
            p.cubemap = make_cuda_tensor(in0, p.gridSize);
            p.out = make_cuda_tensor(in1, p.gridSize);
            p.cubemap.d_val = out0_p;

            void* args[] = { &p };
            cudaLaunchKernel((const void*)DiffuseCubemapBwdKernel, gridSize, blockSize, args, 0, 0);

        ''')
    return out_jittor[0]

def specular_bounds(resolution, costheta_cutoff):

    out_jittor = jt.zeros((6,resolution,resolution,24),dtype = "float32")
    input = jt.zeros((6,resolution,resolution,24))
    # blockSize = getLaunchBlockSize(8, 8, p.gridSize)
    # gridSize = getLaunchGridSize(blockSize, p.gridSize)
    with jt.flag_scope(compile_options=proj_options):
        out_jittor=jt.code(
                outputs=[out_jittor],
                inputs=[input],
                data = {
                    'resolution':int(resolution),
                    'costhetacutoff':float(costheta_cutoff),
                },
                cuda_header=cuda_header,
                cuda_src='''
                @alias(input, in0)
                @alias(out_jittor, out0)
                SpecularBoundsKernelParams p;
                float costheta_cutoff = data["costhetacutoff"];
                int resolution = data["resolution"];
                p.costheta_cutoff = costheta_cutoff;
                p.gridSize = dim3(resolution, resolution, 6);
                dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
                dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);
                // Setup tensors
                #undef out
                p.out = make_cuda_tensor(out0, p.gridSize);
                // Launch CUDA kernel.
                void* args[] = { &p };
                cudaLaunchKernel((const void*)SpecularBoundsKernel, gridSize, blockSize, args, 0, 0);

        ''')
    return out_jittor[0]

def specular_cubemap_fwd(cubemap,bounds,roughness,costheta_cutoff):

    out_jittor = jt.zeros((cubemap.shape[0],cubemap.shape[1],cubemap.shape[2],4),dtype = "float32")
    with jt.flag_scope(compile_options=proj_options):
        out_jittor=jt.code(
                outputs=[out_jittor],
                inputs=[cubemap,bounds],
                data = {
                    'roughness':float(roughness),
                    'costheta_cutoff':float(costheta_cutoff),
                },
                cuda_header=cuda_header,
                cuda_src='''

                @alias(out_jittor, out0)
                float roughness = data["roughness"];
                float costheta_cutoff = data["costheta_cutoff"];
                SpecularCubemapKernelParams p;
                p.roughness = roughness;
                p.costheta_cutoff = costheta_cutoff;
                update_grid(p.gridSize, in0);

                // Choose launch parameters.
                dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
                dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

                // Setup tensors
                #undef out
                p.cubemap = make_cuda_tensor(in0, p.gridSize);
                p.bounds = make_cuda_tensor(in1, p.gridSize);
                p.out = make_cuda_tensor(out0, p.gridSize);

                // Launch CUDA kernel.
                void* args[] = { &p };
                cudaLaunchKernel((const void*)SpecularCubemapFwdKernel, gridSize, blockSize, args, 0, 0);

        ''')
    return out_jittor[0]

def specular_cubemap_bwd(cubemap,bounds,grad,roughness:float,costheta_cutoff:float):
    out_jittor = jt.zeros((cubemap.shape[0],cubemap.shape[1],cubemap.shape[2],cubemap.shape[3]),dtype = "float32")
    with jt.flag_scope(compile_options=proj_options):
        out_jittor=jt.code(
                outputs=[out_jittor],
                inputs=[cubemap,bounds,grad],
                data = {
                    'roughness':float(roughness),
                    'costheta_cutoff':float(costheta_cutoff),
                },
                cuda_header=cuda_header,
                cuda_src='''
                @alias(grad, in2)
                @alias(out_jittor, out0)
                float roughness = data["roughness"];
                float costheta_cutoff = data["costheta_cutoff"];
                SpecularCubemapKernelParams p;
                p.roughness = roughness;
                p.costheta_cutoff = costheta_cutoff;
                update_grid(p.gridSize, in0);

                // Choose launch parameters.
                dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
                dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

                // Setup tensors
                #undef out
                p.cubemap = make_cuda_tensor(in0, p.gridSize);
                p.bounds = make_cuda_tensor(in1, p.gridSize);
                p.out = make_cuda_tensor(in2, p.gridSize);
                p.cubemap.d_val = (void*)out0_p;

                // Launch CUDA kernel.
                void* args[] = { &p };
                cudaLaunchKernel((const void*)SpecularCubemapBwdKernel, gridSize, blockSize, args, 0, 0);

        ''')
    return out_jittor[0]


def xfm_fwd(points, matrix, isPoints, fp16):
    out_jittor = jt.zeros((matrix.shape[0],points.shape[1],4),dtype = "float32")
    with jt.flag_scope(compile_options=proj_options):
        out_jittor=jt.code(
                outputs=[out_jittor],
                inputs=[points, matrix],
                data = {
                    'isPoints':int(isPoints),
                    'fp16':int(fp16),
                },
                cuda_header=cuda_header,
                cuda_src='''
                @alias(out_jittor, out0)
                bool isPoints = data["isPoints"];
                bool fp16 = data["fp16"];
                // Extract input parameters.
                XfmKernelParams p;
                #undef out
                p.out.fp16 = fp16;
                p.isPoints = isPoints;
                p.gridSize.x = points_shape1;
                p.gridSize.y = 1;
                p.gridSize.z = std::max(matrix_shape0, points_shape0);

                // Choose launch parameters.
                dim3 blockSize(BLOCK_X * BLOCK_Y, 1, 1);
                dim3 warpSize = getWarpSize(blockSize);
                dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

                p.points = make_cuda_tensor(in0, p.gridSize);
                p.matrix = make_cuda_tensor(in1, p.gridSize);
                p.out = make_cuda_tensor(out0, p.gridSize);

                // Launch CUDA kernel.
                void* args[] = { &p };
                cudaLaunchKernel((const void*)xfmPointsFwdKernel, gridSize, blockSize, args, 0, 0);

        ''')
    return out_jittor[0]

def xfm_bwd(points, matrix, grad, isPoints):
    if(points.dim()==4):
        out_jittor = jt.zeros((max(matrix.shape[0],points.shape[0]),1,points.shape[1],points.shape[3]),dtype = "float32")
    else:
        out_jittor = jt.zeros((max(matrix.shape[0],points.shape[0]),points.shape[1],points.shape[2]),dtype = "float32")
    with jt.flag_scope(compile_options=proj_options):
        out_jittor=jt.code(
                outputs=[out_jittor],
                inputs=[points, matrix,grad],
                data = {
                    'isPoints':int(isPoints),
                },
                cuda_header=cuda_header,
                cuda_src='''
                @alias(out_jittor, out0)
                #undef out
                bool isPoints = data["isPoints"];
                // Extract input parameters.
                XfmKernelParams p;
                p.isPoints = isPoints;
                p.gridSize.x = points_shape1;
                p.gridSize.y = 1;
                p.gridSize.z = std::max(matrix_shape0, points_shape0);

                // Choose launch parameters.
                dim3 blockSize(BLOCK_X * BLOCK_Y, 1, 1);
                dim3 warpSize = getWarpSize(blockSize);
                dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);
                
                p.points = make_cuda_tensor(in0, p.gridSize);
                p.points.d_val = out0_p;
                p.matrix = make_cuda_tensor(in1, p.gridSize);
                p.out = make_cuda_tensor(in2, p.gridSize);

                // Launch CUDA kernel.
                void* args[] = { &p };
                cudaLaunchKernel((const void*)xfmPointsBwdKernel, gridSize, blockSize, args, 0, 0);

        ''')
    return out_jittor[0]








    
