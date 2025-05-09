import jittor as jt
from jittor import nn
import os
import pathlib
import numpy as np
from typing import List, Optional, Tuple






def texture(tex, uv, uv_da=None, mip_level_bias=None, mip=None, filter_mode='auto', boundary_mode='wrap', max_mip_level=None):
    """Perform texture sampling.

    All input tensors must be contiguous and reside in GPU memory. The output tensor
    will be contiguous and reside in GPU memory.

    Args:
        tex: Texture tensor with dtype `torch.float32`. For 2D textures, must have shape
             [minibatch_size, tex_height, tex_width, tex_channels]. For cube map textures,
             must have shape [minibatch_size, 6, tex_height, tex_width, tex_channels] where
             tex_width and tex_height are equal. Note that `boundary_mode` must also be set
             to 'cube' to enable cube map mode. Broadcasting is supported along the minibatch axis.
        uv: Tensor containing per-pixel texture coordinates. When sampling a 2D texture,
            must have shape [minibatch_size, height, width, 2]. When sampling a cube map
            texture, must have shape [minibatch_size, height, width, 3].
        uv_da: (Optional) Tensor containing image-space derivatives of texture coordinates.
               Must have same shape as `uv` except for the last dimension that is to be twice
               as long.
        mip_level_bias: (Optional) Per-pixel bias for mip level selection. If `uv_da` is omitted,
                        determines mip level directly. Must have shape [minibatch_size, height, width].
        mip: (Optional) Preconstructed mipmap stack from a `texture_construct_mip()` call, or a list
                        of tensors specifying a custom mipmap stack. When specifying a custom mipmap stack,
                        the tensors in the list must follow the same format as `tex` except for width and
                        height that must follow the usual rules for mipmap sizes. The base level texture
                        is still supplied in `tex` and must not be included in the list. Gradients of a
                        custom mipmap stack are not automatically propagated to base texture but the mipmap
                        tensors will receive gradients of their own. If a mipmap stack is not specified
                        but the chosen filter mode requires it, the mipmap stack is constructed internally
                        and discarded afterwards.
        filter_mode: Texture filtering mode to be used. Valid values are 'auto', 'nearest',
                     'linear', 'linear-mipmap-nearest', and 'linear-mipmap-linear'. Mode 'auto'
                     selects 'linear' if neither `uv_da` or `mip_level_bias` is specified, and
                     'linear-mipmap-linear' when at least one of them is specified, these being
                     the highest-quality modes possible depending on the availability of the
                     image-space derivatives of the texture coordinates or direct mip level information.
        boundary_mode: Valid values are 'wrap', 'clamp', 'zero', and 'cube'. If `tex` defines a
                       cube map, this must be set to 'cube'. The default mode 'wrap' takes fractional
                       part of texture coordinates. Mode 'clamp' clamps texture coordinates to the
                       centers of the boundary texels. Mode 'zero' virtually extends the texture with
                       all-zero values in all directions.
        max_mip_level: If specified, limits the number of mipmaps constructed and used in mipmap-based
                       filter modes.

    Returns:
        A tensor containing the results of the texture sampling with shape
        [minibatch_size, height, width, tex_channels]. Cube map fetches with invalid uv coordinates
        (e.g., zero vectors) output all zeros and do not propagate gradients.
    """

    # Default filter mode.
    if filter_mode == 'auto':
        filter_mode = 'linear-mipmap-linear' if (uv_da is not None or mip_level_bias is not None) else 'linear'

    # Sanitize inputs.
    if max_mip_level is None:
        max_mip_level = -1
    else:
        max_mip_level = int(max_mip_level)
        assert max_mip_level >= 0

    # Check inputs.
    # assert isinstance(tex, torch.Tensor) and isinstance(uv, torch.Tensor)
    # if 'mipmap' in filter_mode:
    #     assert isinstance(uv_da, torch.Tensor) or isinstance(mip_level_bias, torch.Tensor)

    # If mipping disabled via max level=0, we may as well use simpler filtering internally.
    if max_mip_level == 0 and filter_mode in ['linear-mipmap-nearest', 'linear-mipmap-linear']:
        filter_mode = 'linear'

    # Convert filter mode to internal enumeration.
    filter_mode_dict = {'nearest': 0, 'linear': 1, 'linear-mipmap-nearest': 2, 'linear-mipmap-linear': 3}
    filter_mode_enum = filter_mode_dict[filter_mode]

    # Convert boundary mode to internal enumeration.
    boundary_mode_dict = {'cube': 0, 'wrap': 1, 'clamp': 2, 'zero': 3}
    boundary_mode_enum = boundary_mode_dict[boundary_mode]


    # Construct a mipmap if necessary.
    num_stack = 0
    if 'mipmap' in filter_mode:
        mip_wrapper, mip_stack = None, []
        mip_stack1,mip_stack2,mip_stack3,mip_stack4,mip_stack5,mip_stack6,mip_stack7,mip_stack8,mip_stack9,mip_stack10,mip_stack11,mip_stack12 = None, None, None, None, None, None, None, None, None, None, None, None
        if mip is not None:
            if isinstance(mip, list):
                mip_stack = mip
                if(len(mip_stack)>12):
                    print("sorry, not support above 12!!!")
                if(len(mip_stack)>=1):
                    mip_stack1 = mip_stack[0]
                if(len(mip_stack)>=2):
                    mip_stack2 = mip_stack[1]
                if(len(mip_stack)>=3):
                    mip_stack3 = mip_stack[2]
                if(len(mip_stack)>=4):
                    mip_stack4 = mip_stack[3]
                if(len(mip_stack)>=5):
                    mip_stack5 = mip_stack[4]
                if(len(mip_stack)>=6):
                    mip_stack6 = mip_stack[5]
                if(len(mip_stack)>=7):
                    mip_stack7 = mip_stack[6]
                if(len(mip_stack)>=8):
                    mip_stack8 = mip_stack[7]
                if(len(mip_stack)>=9):
                    mip_stack9 = mip_stack[8]
                if(len(mip_stack)>=10):
                    mip_stack10 = mip_stack[9]
                if(len(mip_stack)>=11):
                    mip_stack11 = mip_stack[10]
                if(len(mip_stack)>=12):
                    mip_stack12 = mip_stack[11]
            else:
                mip_wrapper = mip
        else:
            mip_wrapper = texture_construct_mip(tex, max_mip_level, boundary_mode == 'cube')

    # Choose stub.
    if filter_mode == 'linear-mipmap-linear' or filter_mode == 'linear-mipmap-nearest':
    
        return _texture_func_mip.apply(filter_mode, tex, uv, uv_da, mip_level_bias, mip_wrapper, filter_mode_enum, boundary_mode_enum, mip_stack1, mip_stack2, mip_stack3, mip_stack4,
                                       mip_stack5, mip_stack6, mip_stack7, mip_stack8, mip_stack9, mip_stack10, mip_stack11, mip_stack12)[0]
    else:
        
        return _texture_func.apply(filter_mode, tex, uv, filter_mode_enum, boundary_mode_enum)[0]


class _texture_func_mip(jt.Function):
    def save_for_backward(self,*args):
        self.saved_tensors = args
    def execute(self, filter_mode, tex, uv, uv_da, mip_level_bias, mip_wrapper, filter_mode_enum, boundary_mode_enum, mip_stack1=None,mip_stack2=None,mip_stack3=None,mip_stack4=None,
                mip_stack5=None,mip_stack6=None,mip_stack7=None,mip_stack8=None,mip_stack9=None,mip_stack10=None,mip_stack11=None,mip_stack12=None):
        mip_stack = []
        empty = jt.array([])
        if uv_da is None:
            uv_da = empty
        if mip_level_bias is None:
            mip_level_bias = empty
        if mip_wrapper is None:
            mip_wrapper = TextureMipWrapper()
        if mip_stack1 is not None:
            mip_stack.append(mip_stack1)
        if mip_stack2 is not None:
            mip_stack.append(mip_stack2)
        if mip_stack3 is not None:
            mip_stack.append(mip_stack3)
        if mip_stack4 is not None:
            mip_stack.append(mip_stack4)
        if mip_stack5 is not None:
            mip_stack.append(mip_stack5)
        if mip_stack6 is not None:
            mip_stack.append(mip_stack6)
        if mip_stack7 is not None:
            mip_stack.append(mip_stack7)
        if mip_stack8 is not None:
            mip_stack.append(mip_stack8)
        if mip_stack9 is not None:
            mip_stack.append(mip_stack9)
        if mip_stack10 is not None:
            mip_stack.append(mip_stack10)
        if mip_stack11 is not None:
            mip_stack.append(mip_stack11)
        if mip_stack12 is not None:
            mip_stack.append(mip_stack12)
        out = texture_fwd_mip(tex, uv, uv_da, mip_level_bias, mip_wrapper, mip_stack, filter_mode_enum, boundary_mode_enum)
        self.save_for_backward(tex, uv, uv_da, mip_level_bias, mip_stack)
        self.saved_misc = filter_mode, mip_wrapper, filter_mode_enum, boundary_mode_enum
        return out
    
    def grad(self, dy):
        tex, uv, uv_da, mip_level_bias, mip_stack = self.saved_tensors
        filter_mode, mip_wrapper, filter_mode_enum, boundary_mode_enum = self.saved_misc
        if filter_mode == 'linear-mipmap-linear':
            (g_tex, g_uv, g_uv_da, g_mip_level_bias, g_mip_stack1,g_mip_stack2,g_mip_stack3,g_mip_stack4,g_mip_stack5,g_mip_stack6,g_mip_stack7,g_mip_stack8,
            g_mip_stack9,g_mip_stack10,g_mip_stack11,g_mip_stack12)= texture_grad_linear_mipmap_linear(tex, uv, dy, uv_da, mip_level_bias, mip_wrapper, mip_stack, filter_mode_enum, boundary_mode_enum)
            if(g_uv_da.shape[0]==0):
                g_uv_da = None
            if(g_mip_level_bias.shape[0]==0):
                g_mip_level_bias = None
            if(g_mip_stack1.shape[0]==0):
                g_mip_stack1 = None
            if(g_mip_stack2.shape[0]==0):
                g_mip_stack2 = None
            if(g_mip_stack3.shape[0]==0):
                g_mip_stack3 = None
            if(g_mip_stack4.shape[0]==0):
                g_mip_stack4 = None
            if(g_mip_stack5.shape[0]==0):
                g_mip_stack5 = None
            if(g_mip_stack6.shape[0]==0):
                g_mip_stack6 = None
            if(g_mip_stack7.shape[0]==0):
                g_mip_stack7 = None
            if(g_mip_stack8.shape[0]==0):
                g_mip_stack8 = None
            if(g_mip_stack9.shape[0]==0):
                g_mip_stack9 = None
            if(g_mip_stack10.shape[0]==0):
                g_mip_stack10 = None
            if(g_mip_stack11.shape[0]==0):
                g_mip_stack11 = None
            if(g_mip_stack12.shape[0]==0):
                g_mip_stack12 = None
            # return (None, g_tex, g_uv, g_uv_da, g_mip_level_bias, None, None, None) + tuple(g_mip_stack)

            return (None, g_tex, g_uv, g_uv_da, g_mip_level_bias, None, None, None, g_mip_stack1, g_mip_stack2, g_mip_stack3, g_mip_stack4, g_mip_stack5, g_mip_stack6, g_mip_stack7, g_mip_stack8
                    , g_mip_stack9, g_mip_stack10, g_mip_stack11, g_mip_stack12)
        else: # linear-mipmap-nearest
            g_tex, g_uv, g_mip_stack1,g_mip_stack2,g_mip_stack3,g_mip_stack4 = texture_grad_linear_mipmap_nearest(tex, uv, dy, uv_da, mip_level_bias, mip_wrapper, mip_stack, filter_mode_enum, boundary_mode_enum)
            if(g_mip_stack1.shape[0]==0):
                g_mip_stack1 = None
            if(g_mip_stack2.shape[0]==0):
                g_mip_stack2 = None
            if(g_mip_stack3.shape[0]==0):
                g_mip_stack3 = None
            if(g_mip_stack4.shape[0]==0):
                g_mip_stack4 = None
            if(g_mip_stack5.shape[0]==0):
                g_mip_stack5 = None
            if(g_mip_stack6.shape[0]==0):
                g_mip_stack6 = None
            if(g_mip_stack7.shape[0]==0):
                g_mip_stack7 = None
            if(g_mip_stack8.shape[0]==0):
                g_mip_stack8 = None
            if(g_mip_stack9.shape[0]==0):
                g_mip_stack9 = None
            if(g_mip_stack10.shape[0]==0):
                g_mip_stack10 = None
            if(g_mip_stack11.shape[0]==0):
                g_mip_stack11 = None
            if(g_mip_stack12.shape[0]==0):
                g_mip_stack12 = None

            return (None, g_tex, g_uv, None, None, None, None, None, g_mip_stack1, g_mip_stack2, g_mip_stack3, g_mip_stack4, g_mip_stack5, g_mip_stack6, g_mip_stack7, g_mip_stack8
                    , g_mip_stack9, g_mip_stack10, g_mip_stack11, g_mip_stack12)

class _texture_func(jt.Function):
    def save_for_backward(self,*args):
        self.saved_tensors = args
    def execute(self, filter_mode, tex, uv, filter_mode_enum, boundary_mode_enum):
        out = texture_fwd(tex, uv, filter_mode_enum, boundary_mode_enum)
        self.save_for_backward(tex, uv)
        self.saved_misc = filter_mode, filter_mode_enum, boundary_mode_enum

        return out
    def grad(self, dy):
        tex, uv = self.saved_tensors
        filter_mode, filter_mode_enum, boundary_mode_enum = self.saved_misc
        if filter_mode == 'linear':
            g_tex, g_uv = texture_grad_linear(tex, uv, dy, filter_mode_enum, boundary_mode_enum)

            return None, g_tex, g_uv, None, None
        else: # nearest
            g_tex = texture_grad_nearest(tex, uv, dy, filter_mode_enum, boundary_mode_enum)
            return None, g_tex, None, None, None
        
class TextureMipWrapper:
    def __init__(
        self,
        mip = jt.array([]),
        max_mip_level = 0,
        texture_size = [],
        cube_mode = False,
    ):
        self.mip = mip
        self.max_mip_level = max_mip_level
        self.texture_size = texture_size
        self.cube_mode = cube_mode

cuda_header = """
#include "common.h"
#include "texture.cuh"
#include <cuda_runtime.h>
#include <stdint.h>

//------------------------------------------------------------------------
// Kernel prototypes.



//------------------------------------------------------------------------
// Modeselektor.

static void set_modes(TextureKernelParams& p, int filter_mode, int boundary_mode, int max_mip_level)
{
    // Mip and filter modes.
    p.filterMode = filter_mode;
    //NVDR_CHECK(p.filterMode >= 0 && p.filterMode < TEX_MODE_COUNT, "filter_mode unsupported");
    p.enableMip = (p.filterMode == TEX_MODE_LINEAR_MIPMAP_NEAREST || p.filterMode == TEX_MODE_LINEAR_MIPMAP_LINEAR);

    // Mip level clamp.
    if (p.enableMip)
    {
        p.mipLevelLimit = max_mip_level;
        //NVDR_CHECK(p.mipLevelLimit >= -1, "invalid max_mip_level");
    }

    // Boundary mode.
    p.boundaryMode = boundary_mode;
    //NVDR_CHECK(p.boundaryMode >= 0 && p.boundaryMode < TEX_BOUNDARY_MODE_COUNT, "boundary_mode unsupported");
}

"""

header_path = os.path.join(os.path.dirname(__file__), 'common')
# glm_path = os.path.join(os.path.dirname(__file__),'third_party','glm')
proj_options = {f'FLAGS: -I{header_path} -l"TextureJittor" -L"{os.path.dirname(__file__)}"':1}

def texture_construct_mip(texin:jt.Var, max_mip_level=None, cube_mode=False):
    if max_mip_level is None:
        max_mip_level = -1
    else:
        max_mip_level = int(max_mip_level)
    cube_mode = int(cube_mode)   
    mip_total = jt.zeros([1])
    if(cube_mode):
        channels = texin.shape[4]
    else:
        channels = texin.shape[3]
    with jt.flag_scope(compile_options=proj_options):
        mip_total=jt.code(
                outputs=[mip_total],
                inputs=[texin],
                data = {
                    'max_mip_level':max_mip_level,
                    'cube_mode':cube_mode,
                    'channels': channels,
                },
                cpu_header=cuda_header,
                cpu_src='''
            @alias(texin, in0)
            @alias(mip_total,out0)
            int max_mip_level = data["max_mip_level"];
            bool cube_mode = data["cube_mode"];
            TextureKernelParams p = {};
            p.mipLevelLimit = max_mip_level;
            p.boundaryMode = cube_mode ? TEX_BOUNDARY_MODE_CUBE : TEX_BOUNDARY_MODE_WRAP;
            p.texDepth  = in0_shape0;
            if(cube_mode){
            p.texHeight = in0_shape2;
            p.texWidth  = in0_shape3;
            p.channels  = data["channels"];
            }
            else{
            p.texHeight = in0_shape1;
            p.texWidth  = in0_shape2;
            p.channels  = data["channels"];
            }
            int mipOffsets[TEX_MAX_MIP_LEVEL];
            int mipTotal = calculateMipInfo(p, mipOffsets);
            @out0(0) = mipTotal;
        ''')
        mip = jt.empty((int(mip_total[0])))
        
        mip=jt.code(
                outputs=[mip],
                inputs=[texin],
                data = {
                    'max_mip_level':max_mip_level,
                    'cube_mode':cube_mode,
                    'channels': channels,
                },
                cuda_header=cuda_header,
                cuda_src='''
            @alias(texin, in0)
            @alias(mip,out0)
            int max_mip_level = data["max_mip_level"];
            bool cube_mode = data["cube_mode"];
            TextureKernelParams p = {};
            p.mipLevelLimit = max_mip_level;
            p.boundaryMode = cube_mode ? TEX_BOUNDARY_MODE_CUBE : TEX_BOUNDARY_MODE_WRAP;
            p.texDepth  = in0_shape0;
            if(cube_mode){
            p.texHeight = in0_shape2;
            p.texWidth  = in0_shape3;
            p.channels  = data["channels"];
            }
            else{
            p.texHeight = in0_shape1;
            p.texWidth  = in0_shape2;
            p.channels  = data["channels"];
            }
            p.tex[0] = in0_p;
            int mipOffsets[TEX_MAX_MIP_LEVEL];
            int mipTotal = calculateMipInfo(p, mipOffsets);
            // float* pmip = mip_p;
            for (int i=1; i <= p.mipLevelMax; i++)
            p.tex[i] = mip_p + mipOffsets[i]; // Pointers to mip levels.
            void* args[] = {&p};
            int channel_div_idx = 0;
            if (!(p.channels & 3))
                channel_div_idx = 2;  // Channel count divisible by 4.
            else if (!(p.channels & 1))
                channel_div_idx = 1;  // Channel count divisible by 2.
            // Build mip levels.
            for (int i=1; i <= p.mipLevelMax; i++)
            {
                int2 ms = mipLevelSize(p, i);
                int3 sz = make_int3(ms.x, ms.y, p.texDepth);
                dim3 blockSize = getLaunchBlockSize(TEX_FWD_MAX_MIP_KERNEL_BLOCK_WIDTH, TEX_FWD_MAX_MIP_KERNEL_BLOCK_HEIGHT, sz.x, sz.y);
                dim3 gridSize  = getLaunchGridSize(blockSize, sz.x, sz.y, sz.z * (cube_mode ? 6 : 1));
                p.mipLevelOut = i;
                void* build_func_tbl[3] = { (void*)MipBuildKernel1, (void*)MipBuildKernel2, (void*)MipBuildKernel4 };
                cudaLaunchKernel(build_func_tbl[channel_div_idx], gridSize, blockSize, args, 0, 0);

    }
        ''')
        mip_wrapper=TextureMipWrapper(mip[0],max_mip_level,list(texin.numpy().shape),cube_mode)
    # out.compile_options = proj_options
    return mip_wrapper

def texture_fwd_mip(tex_in:jt.Var, uv_in:jt.Var, uv_da_in:jt.Var, mip_level_bias:jt.Var, 
    mip_wrapper, mip_stack, filter_mode, boundary_mode):
    
    if(tex_in.dim()==4):
        channels = tex_in.shape[3]
    else:
        channels = tex_in.shape[4]
    has_mip_stack = (len(mip_stack) > 0)
    mip_w = mip_wrapper.mip
    has_mip_stack = len(mip_stack) > 0 if mip_stack is not None else False
    max_mip_level = len(mip_stack) if has_mip_stack else mip_wrapper.max_mip_level
    has_uv_da = uv_da_in is not None and uv_da_in.shape[0]!=0
    has_mip_level_bias = mip_level_bias is not None and mip_level_bias.shape[0]!=0

    if(len(mip_stack)>12):
        print("now the len of stack is above 4!!")
        return
    

    # mip_stack4 = jt.array([])
    # mip_stack3 = jt.array([])
    # mip_stack2 = jt.array([])
    # mip_stack1 = jt.array([])
    
    # if(len(mip_stack)>=4):
    #     mip_stack4 = mip_stack[3]
    # if(len(mip_stack)>=3):
    #     mip_stack3 = mip_stack[2]
    # if(len(mip_stack)>=2):
    #     mip_stack2 = mip_stack[1]
    # if(len(mip_stack)>=1):
    #     mip_stack1 = mip_stack[0]

    mip_stacks = [jt.array([]) for _ in range(12)]

    # 填充实际存在的元素
    for i in range(min(12, len(mip_stack))):
        mip_stacks[i] = mip_stack[i]

    # 也可以解包为单独的变量（如果需要）
    (mip_stack1, mip_stack2, mip_stack3, mip_stack4, 
    mip_stack5, mip_stack6, mip_stack7, mip_stack8,
    mip_stack9, mip_stack10, mip_stack11, mip_stack12) = mip_stacks

    ans = jt.ones([uv_in.shape[0],uv_in.shape[1],uv_in.shape[2],channels],dtype = "float32")

    with jt.flag_scope(compile_options=proj_options):
        ans,=jt.code(
                outputs=[ans],
                inputs=[tex_in,uv_in,uv_da_in,mip_level_bias,mip_w,mip_stack1,mip_stack2,mip_stack3,mip_stack4,mip_stack5, mip_stack6, mip_stack7, mip_stack8,
    mip_stack9, mip_stack10, mip_stack11, mip_stack12],
                data = {
                    'filter_mode':filter_mode,
                    'boundary_mode':boundary_mode,
                    'max_mip_level':max_mip_level,
                    'has_uv_da': int(has_uv_da),
                    'has_mip_level_bias': int(has_mip_level_bias),
                    'channels': channels,
                    'has_mip_stack': int(has_mip_stack),
                },
                cuda_header=cuda_header,
                cuda_src='''
            @alias(tex_in, in0)
            @alias(uv_in, in1)
            @alias(uv_da_in, in2)
            @alias(mip_level_bias, in3)
            @alias(mip_w, in4)
            @alias(mip_stack1, in5)
            @alias(mip_stack2, in6)
            @alias(mip_stack3, in7)
            @alias(mip_stack4, in8)
            @alias(mip_stack1, in5)
            @alias(mip_stack2, in6)
            @alias(mip_stack3, in7)
            @alias(mip_stack4, in8)
            @alias(mip_stack5, in9)
            @alias(mip_stack6, in10)
            @alias(mip_stack7, in11)
            @alias(mip_stack8, in12)
            @alias(mip_stack9, in13)
            @alias(mip_stack10, in14)
            @alias(mip_stack11, in15)
            @alias(mip_stack12, in16)
            @alias(ans,out0)
            TextureKernelParams p = {};
            int max_mip_level = data["max_mip_level"];
            int boundary_mode = data["boundary_mode"];
            int filter_mode = data["filter_mode"];
            set_modes(p, filter_mode, boundary_mode, max_mip_level);
            // See if we have these tensors or not.
            bool has_uv_da = data["has_uv_da"];
            bool has_mip_level_bias = data["has_mip_level_bias"];
            bool cube_mode = (boundary_mode == TEX_BOUNDARY_MODE_CUBE);
            if(!cube_mode){
            p.texHeight = in0_shape1;
            p.texWidth  = in0_shape2;
            p.channels  = data["channels"];
            }
            else{
            p.texHeight = in0_shape2;
            p.texWidth  = in0_shape3;
            p.channels  = data["channels"];
            }
            p.n         = in1_shape0;
            p.imgHeight = in1_shape1;
            p.imgWidth  = in1_shape2;
            p.texDepth  = in0_shape0;

            p.tex[0] = in0_p;
            p.uv = in1_p;
            p.uvDA = (p.enableMip && has_uv_da) ? uv_da_in_p : NULL;
            p.mipLevelBias = (p.enableMip && has_mip_level_bias) ? mip_level_bias_p: NULL;

            #undef out
            p.out = out0_p;

            void* args[] = {&p};
            int channel_div_idx = 0;
            if (!(p.channels & 3))
                channel_div_idx = 2;  // Channel count divisible by 4.
            else if (!(p.channels & 1))
                channel_div_idx = 1;  // Channel count divisible by 2.

                
            float* pmip = nullptr;
            if (p.enableMip)
                {
                    if (data["has_mip_stack"])
                    {
                        // Custom mip stack supplied. Check that sizes match and assign.
                        p.mipLevelMax = max_mip_level;
                        for (int i=1; i <= p.mipLevelMax; i++)
                        {
                            //torch::Tensor& t = mip_stack[i-1];
                            int2 sz = mipLevelSize(p, i);
                            if(i==1) p.tex[i] = mip_stack1_p;
                            else if(i==2) p.tex[i] = mip_stack2_p;
                            else if(i==3) p.tex[i] = mip_stack3_p;
                            else if(i==4) p.tex[i] = mip_stack4_p;
                            else if(i==5) p.tex[i] = mip_stack5_p;
                            else if(i==6) p.tex[i] = mip_stack6_p;
                            else if(i==7) p.tex[i] = mip_stack7_p;
                            else if(i==8) p.tex[i] = mip_stack8_p;
                            else if(i==9) p.tex[i] = mip_stack9_p;
                            else if(i==10) p.tex[i] = mip_stack10_p;
                            else if(i==11) p.tex[i] = mip_stack11_p;
                            else if(i==12) p.tex[i] = mip_stack12_p;
                        }
                        
                    }
                    else
                    {
                        // Generate mip offsets, check mipmap size, and set mip data pointer.
                        int mipOffsets[TEX_MAX_MIP_LEVEL];
                        int mipTotal = calculateMipInfo(p, mipOffsets);
                        pmip = mip_w_p;
                        for (int i=1; i <= p.mipLevelMax; i++)
                            p.tex[i] = pmip + mipOffsets[i]; // Pointers to mip levels.
                        
                    }
                }
            dim3 blockSize = getLaunchBlockSize(TEX_FWD_MAX_KERNEL_BLOCK_WIDTH, TEX_FWD_MAX_KERNEL_BLOCK_HEIGHT, p.imgWidth, p.imgHeight);
            dim3 gridSize  = getLaunchGridSize(blockSize, p.imgWidth, p.imgHeight, p.n);

            void* func_tbl[TEX_MODE_COUNT * 2 * 2 * 3] = {
                        (void*)TextureFwdKernelNearest1,
                        (void*)TextureFwdKernelNearest2,
                        (void*)TextureFwdKernelNearest4,
                        (void*)TextureFwdKernelLinear1,
                        (void*)TextureFwdKernelLinear2,
                        (void*)TextureFwdKernelLinear4,
                        (void*)TextureFwdKernelLinearMipmapNearest1,
                        (void*)TextureFwdKernelLinearMipmapNearest2,
                        (void*)TextureFwdKernelLinearMipmapNearest4,
                        (void*)TextureFwdKernelLinearMipmapLinear1,
                        (void*)TextureFwdKernelLinearMipmapLinear2,
                        (void*)TextureFwdKernelLinearMipmapLinear4,
                        (void*)TextureFwdKernelCubeNearest1,
                        (void*)TextureFwdKernelCubeNearest2,
                        (void*)TextureFwdKernelCubeNearest4,
                        (void*)TextureFwdKernelCubeLinear1,
                        (void*)TextureFwdKernelCubeLinear2,
                        (void*)TextureFwdKernelCubeLinear4,
                        (void*)TextureFwdKernelCubeLinearMipmapNearest1,
                        (void*)TextureFwdKernelCubeLinearMipmapNearest2,
                        (void*)TextureFwdKernelCubeLinearMipmapNearest4,
                        (void*)TextureFwdKernelCubeLinearMipmapLinear1,
                        (void*)TextureFwdKernelCubeLinearMipmapLinear2,
                        (void*)TextureFwdKernelCubeLinearMipmapLinear4,
                        NULL,
                        NULL,
                        NULL,
                        NULL,
                        NULL,
                        NULL,
                        (void*)TextureFwdKernelLinearMipmapNearestBO1,
                        (void*)TextureFwdKernelLinearMipmapNearestBO2,
                        (void*)TextureFwdKernelLinearMipmapNearestBO4,
                        (void*)TextureFwdKernelLinearMipmapLinearBO1,
                        (void*)TextureFwdKernelLinearMipmapLinearBO2,
                        (void*)TextureFwdKernelLinearMipmapLinearBO4,
                        NULL,
                        NULL,
                        NULL,
                        NULL,
                        NULL,
                        NULL,
                        (void*)TextureFwdKernelCubeLinearMipmapNearestBO1,
                        (void*)TextureFwdKernelCubeLinearMipmapNearestBO2,
                        (void*)TextureFwdKernelCubeLinearMipmapNearestBO4,
                        (void*)TextureFwdKernelCubeLinearMipmapLinearBO1,
                        (void*)TextureFwdKernelCubeLinearMipmapLinearBO2,
                        (void*)TextureFwdKernelCubeLinearMipmapLinearBO4,
                    };
            int func_idx = p.filterMode;
            if (cube_mode)
                func_idx += TEX_MODE_COUNT; // Cube variant.
            if (p.enableMip && !has_uv_da)
                func_idx += TEX_MODE_COUNT * 2; // Bias-only variant.
            func_idx = func_idx * 3 + channel_div_idx; // Choose vector size.
            cudaLaunchKernel(func_tbl[func_idx], gridSize, blockSize, args, 0, 0);

            
        ''' )
    return ans

def texture_fwd(tex,uv,filter_mode,boundary_mode):
    empty_tensor = jt.array([],dtype = "float32")
    empty_vector = jt.array([],dtype = "float32")
    return texture_fwd_mip(tex, uv, empty_tensor, empty_tensor, TextureMipWrapper(), empty_vector, filter_mode, boundary_mode)

def texture_grad_linear_mipmap_linear(tex_in, uv_in, dy_in, uv_da_in, mip_level_bias, mip_wrapper, mip_stack, filter_mode, boundary_mode):
    
    if(tex_in.dim()==4):
        channels = tex_in.shape[3]
    else:
        channels = tex_in.shape[4]
    has_mip_stack = (len(mip_stack) > 0)
    mip_w = mip_wrapper.mip
    has_mip_stack = len(mip_stack) > 0 if mip_stack is not None else False
    max_mip_level = len(mip_stack) if has_mip_stack else mip_wrapper.max_mip_level
    
    has_uv_da = uv_da_in.shape[0]>0 if uv_da_in is not None else False
    has_mip_level_bias = mip_level_bias.shape[0]>0 if mip_level_bias is not None else False

    if(len(mip_stack)>12):
        print("now the len of stack is above 12!!")
        return
    
    mip_stacks = [jt.array([], dtype="float32") for _ in range(12)]
    grad_mip_stacks = [jt.zeros_like(stack) for stack in mip_stacks]

    # 填充数据
    for i in range(min(12, len(mip_stack))):
        mip_stacks[i] = mip_stack[i]
        grad_mip_stacks[i] = jt.zeros_like(mip_stacks[i])

    # 解包为独立变量（如果需要）
    (mip_stack1, mip_stack2, mip_stack3, mip_stack4,
    mip_stack5, mip_stack6, mip_stack7, mip_stack8,
    mip_stack9, mip_stack10, mip_stack11, mip_stack12) = mip_stacks

    (grad_mip_stack1, grad_mip_stack2, grad_mip_stack3, grad_mip_stack4,
    grad_mip_stack5, grad_mip_stack6, grad_mip_stack7, grad_mip_stack8,
    grad_mip_stack9, grad_mip_stack10, grad_mip_stack11, grad_mip_stack12) = grad_mip_stacks

    # mip_stack3 = jt.array([],dtype="float32")
    # mip_stack2 = jt.array([],dtype="float32")
    # mip_stack1 = jt.array([],dtype="float32")
    # mip_stack4 = jt.array([],dtype="float32")
    
    # if(len(mip_stack)>=4):
    #     mip_stack4 = mip_stack[3]
    # if(len(mip_stack)>=3):
    #     mip_stack3 = mip_stack[2]
    # if(len(mip_stack)>=2):
    #     mip_stack2 = mip_stack[1]
    # if(len(mip_stack)>=1):
    #     mip_stack1 = mip_stack[0]

    # grad_mip_stack1 = jt.zeros_like(mip_stack1)
    # grad_mip_stack2 = jt.zeros_like(mip_stack2)
    # grad_mip_stack3 = jt.zeros_like(mip_stack3)
    # grad_mip_stack4 = jt.zeros_like(mip_stack4)
    
    ans = jt.zeros_like(tex_in)
    grad_uv = jt.rand_like(uv_in)
    grad_uv_da = jt.rand_like(uv_da_in)
    grad_mip_level_bias = jt.rand_like(mip_level_bias)
    grad_mip = jt.array([])
    grad_mip_stack = []
    # for i in [tex_in,dy_in,uv_in,uv_da_in,mip_level_bias,mip_w,mip_stack1,mip_stack2,mip_stack3,mip_stack4]:
    #     print(i.shape)
    with jt.flag_scope(compile_options=proj_options):
        ans,grad_uv,grad_uv_da,grad_mip_level_bias,grad_mip,grad_mip_stack1,grad_mip_stack2,grad_mip_stack3,grad_mip_stack4,grad_mip_stack5, grad_mip_stack6, grad_mip_stack7, grad_mip_stack8, grad_mip_stack9, grad_mip_stack10, grad_mip_stack11, grad_mip_stack12=jt.code(
                outputs=[ans,grad_uv,grad_uv_da,grad_mip_level_bias,grad_mip,grad_mip_stack1,grad_mip_stack2,grad_mip_stack3,grad_mip_stack4,grad_mip_stack5, grad_mip_stack6, grad_mip_stack7, grad_mip_stack8,
    grad_mip_stack9, grad_mip_stack10, grad_mip_stack11, grad_mip_stack12],
                inputs=[tex_in,dy_in,uv_in,uv_da_in,mip_level_bias,mip_w,mip_stack1,mip_stack2,mip_stack3,mip_stack4,mip_stack5, mip_stack6, mip_stack7, mip_stack8,
    mip_stack9, mip_stack10, mip_stack11, mip_stack12],
                data = {
                    'filter_mode':filter_mode,
                    'boundary_mode':boundary_mode,
                    'max_mip_level':max_mip_level,
                    'has_uv_da': int(has_uv_da),
                    'has_mip_level_bias': int(has_mip_level_bias),
                    'channels': channels,
                    'has_mip_stack': int(has_mip_stack),
                },
                cuda_header=cuda_header,
                cuda_src='''
            @alias(tex_in, in0)
            @alias(dy_in, in1)
            @alias(uv_in, in2)
            @alias(uv_da_in, in3)
            @alias(mip_level_bias, in4)
            @alias(mip_w, in5)
            @alias(mip_stack1, in6)
            @alias(mip_stack2, in7)
            @alias(mip_stack3, in8)
            @alias(mip_stack4, in9)
            @alias(mip_stack5, in10)
            @alias(mip_stack6, in11)
            @alias(mip_stack7, in12)
            @alias(mip_stack8, in13)
            @alias(mip_stack9, in14)
            @alias(mip_stack10, in15)
            @alias(mip_stack11, in16)
            @alias(mip_stack12, in17)
            @alias(ans,out0)
            @alias(grad_uv,out1)
            @alias(grad_uv_da,out2)
            @alias(grad_mip_level_bias,out3)
            @alias(grad_mip,out4)
            @alias(grad_mip_stack1, out5)
            @alias(grad_mip_stack2, out6)
            @alias(grad_mip_stack3, out7)
            @alias(grad_mip_stack4, out8)
            @alias(grad_mip_stack5, out9)
            @alias(grad_mip_stack6, out10)
            @alias(grad_mip_stack7, out11)
            @alias(grad_mip_stack8, out12)
            @alias(grad_mip_stack9, out13)
            @alias(grad_mip_stack10, out14)
            @alias(grad_mip_stack11, out15)
            @alias(grad_mip_stack12, out16)
            TextureKernelParams p = {};
            int max_mip_level = data["max_mip_level"];
            int boundary_mode = data["boundary_mode"];
            int filter_mode = data["filter_mode"];
            set_modes(p, filter_mode, boundary_mode, max_mip_level);
            // See if we have these tensors or not.
            bool has_uv_da = data["has_uv_da"];
            bool has_mip_level_bias = data["has_mip_level_bias"];
            bool cube_mode = (boundary_mode == TEX_BOUNDARY_MODE_CUBE);
            if(!cube_mode){
            p.texHeight = in0_shape1;
            p.texWidth  = in0_shape2;
            p.channels  = data["channels"];
            }
            else{
            p.texHeight = in0_shape2;
            p.texWidth  = in0_shape3;
            p.channels  = data["channels"];
            }
            p.n         = in2_shape0;
            p.imgHeight = in2_shape1;
            p.imgWidth  = in2_shape2;
            p.texDepth  = in0_shape0;

            p.tex[0] = in0_p;
            p.uv = in2_p;
            p.dy = dy_in_p;
            p.uvDA = (p.enableMip && has_uv_da) ? uv_da_in_p : NULL;
            p.mipLevelBias = (p.enableMip && has_mip_level_bias) ? mip_level_bias_p: NULL;

            p.gradTex[0] = ans_p;

            if (p.filterMode != TEX_MODE_NEAREST)
            {
                
                p.gradUV = grad_uv_p;
                // Gradients for things affecting mip level.
                if (p.filterMode == TEX_MODE_LINEAR_MIPMAP_LINEAR)
                {
                    // Allocate output tensor for uv_da gradient.
                    if (has_uv_da)
                    {
                        p.gradUVDA = grad_uv_da_p;
                    }

                    // Allocate output tensor for mip_level_bias gradient.
                    if (has_mip_level_bias)
                    {
                        p.gradMipLevelBias = grad_mip_level_bias_p;
                    }
                }
            }

            // Choose kernel variants based on channel count.
            int channel_div_idx = 0;
            if (!(p.channels & 3))
                channel_div_idx = 2;  // Channel count divisible by 4.
            else if (!(p.channels & 1))
                channel_div_idx = 1;  // Channel count divisible by 2.

                
            float* pmip = nullptr;
            float* pgradMip = nullptr;
            if (p.enableMip)
                {
                    if (data["has_mip_stack"])
                    {
                        // Custom mip stack supplied. Check that sizes match and assign.
                        p.mipLevelMax = max_mip_level;
                        for (int i=1; i <= p.mipLevelMax; i++)
                        {
                            //torch::Tensor& t = mip_stack[i-1];
                            int2 sz = mipLevelSize(p, i);
                            if(i == 1) {
                                p.tex[i] = mip_stack1_p;
                                p.gradTex[i] = grad_mip_stack1_p;
                            }
                            else if(i == 2) {
                                p.tex[i] = mip_stack2_p;
                                p.gradTex[i] = grad_mip_stack2_p;
                            }
                            else if(i == 3) {
                                p.tex[i] = mip_stack3_p;
                                p.gradTex[i] = grad_mip_stack3_p;
                            }
                            else if(i == 4) {
                                p.tex[i] = mip_stack4_p;
                                p.gradTex[i] = grad_mip_stack4_p;
                            }
                            else if(i == 5) {
                                p.tex[i] = mip_stack5_p;
                                p.gradTex[i] = grad_mip_stack5_p;
                            }
                            else if(i == 6) {
                                p.tex[i] = mip_stack6_p;
                                p.gradTex[i] = grad_mip_stack6_p;
                            }
                            else if(i == 7) {
                                p.tex[i] = mip_stack7_p;
                                p.gradTex[i] = grad_mip_stack7_p;
                            }
                            else if(i == 8) {
                                p.tex[i] = mip_stack8_p;
                                p.gradTex[i] = grad_mip_stack8_p;
                            }
                            else if(i == 9) {
                                p.tex[i] = mip_stack9_p;
                                p.gradTex[i] = grad_mip_stack9_p;
                            }
                            else if(i == 10) {
                                p.tex[i] = mip_stack10_p;
                                p.gradTex[i] = grad_mip_stack10_p;
                            }
                            else if(i == 11) {
                                p.tex[i] = mip_stack11_p;
                                p.gradTex[i] = grad_mip_stack11_p;
                            }
                            else if(i == 12) {
                                p.tex[i] = mip_stack12_p;
                                p.gradTex[i] = grad_mip_stack12_p;
                            }
                        }
                    }
                    else
                    {
                        // Generate mip offsets, check mipmap size, and set mip data pointer.
                        int mipOffsets[TEX_MAX_MIP_LEVEL];
                        int mipTotal = calculateMipInfo(p, mipOffsets);
                        pmip = mip_w_p;
                        pgradMip = grad_mip_p;
                        for (int i=1; i <= p.mipLevelMax; i++){
                            p.tex[i] = pmip + mipOffsets[i]; // Pointers to mip levels.
                            p.gradTex[i] = pgradMip + mipOffsets[i];}
                    }
                }
            void* args[] = {&p};
            dim3 blockSize = getLaunchBlockSize(TEX_FWD_MAX_KERNEL_BLOCK_WIDTH, TEX_FWD_MAX_KERNEL_BLOCK_HEIGHT, p.imgWidth, p.imgHeight);
            dim3 gridSize  = getLaunchGridSize(blockSize, p.imgWidth, p.imgHeight, p.n);
            void* func_tbl[TEX_MODE_COUNT * 2 * 2] = {
                (void*)TextureGradKernelNearest,
                (void*)TextureGradKernelLinear,
                (void*)TextureGradKernelLinearMipmapNearest,
                (void*)TextureGradKernelLinearMipmapLinear,
                (void*)TextureGradKernelCubeNearest,
                (void*)TextureGradKernelCubeLinear,
                (void*)TextureGradKernelCubeLinearMipmapNearest,
                (void*)TextureGradKernelCubeLinearMipmapLinear,
                NULL,
                NULL,
                (void*)TextureGradKernelLinearMipmapNearestBO,
                (void*)TextureGradKernelLinearMipmapLinearBO,
                NULL,
                NULL,
                (void*)TextureGradKernelCubeLinearMipmapNearestBO,
                (void*)TextureGradKernelCubeLinearMipmapLinearBO,
            };

            int func_idx = p.filterMode;
            if (cube_mode)
                func_idx += TEX_MODE_COUNT; // Cube variant.
            if (p.enableMip && !has_uv_da)
                func_idx += TEX_MODE_COUNT * 2; // Bias-only variant
            
            cudaLaunchKernel(func_tbl[func_idx], gridSize, blockSize, args, 0, 0);

            
            if (p.enableMip && !data["has_mip_stack"])
                {
                    dim3 blockSize = getLaunchBlockSize(TEX_GRAD_MAX_MIP_KERNEL_BLOCK_WIDTH, TEX_GRAD_MAX_MIP_KERNEL_BLOCK_HEIGHT, p.texWidth, p.texHeight);
                    dim3 gridSize  = getLaunchGridSize(blockSize, p.texWidth, p.texHeight, p.texDepth * (cube_mode ? 6 : 1));
                    
                    int sharedBytes = blockSize.x * blockSize.y * p.channels * sizeof(float);
                    std::cout<<sharedBytes;
                    void* mip_grad_func_tbl[3] = { (void*)MipGradKernel1, (void*)MipGradKernel2, (void*)MipGradKernel4 };
                    cudaLaunchKernel(mip_grad_func_tbl[channel_div_idx], gridSize, blockSize, args, sharedBytes, 0);

                
                    
                    //NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(mip_grad_func_tbl[channel_div_idx], gridSize, blockSize, args, sharedBytes, stream));
                }

        ''' 
        )

        # if has_mip_stack and max_mip_level<=4:
        #     for i in range(max_mip_level):
        #         if(i==1):
        #             grad_mip_stack.append(grad_mip_stack1)
        #         elif(i==2):
        #             grad_mip_stack.append(grad_mip_stack2)
        #         elif(i==3):
        #             grad_mip_stack.append(grad_mip_stack3)
        #         elif(i==4):
        #             grad_mip_stack.append(grad_mip_stack4)    
    return (ans, grad_uv, grad_uv_da, grad_mip_level_bias, grad_mip_stack1, grad_mip_stack2, grad_mip_stack3, grad_mip_stack4,
    grad_mip_stack5, grad_mip_stack6, grad_mip_stack7, grad_mip_stack8,
    grad_mip_stack9, grad_mip_stack10, grad_mip_stack11, grad_mip_stack12)

def texture_grad_nearest(tex, uv, dy, filter_mode, boundary_mode):
    empty_tensor = jt.array([])
    empty_vector = [ ]
    result = texture_grad_linear_mipmap_linear(tex, uv, dy, empty_tensor, empty_tensor, TextureMipWrapper(), empty_vector, filter_mode=filter_mode, boundary_mode=boundary_mode)
    return result[0]

def texture_grad_linear(tex, uv, dy, filter_mode, boundary_mode):
    empty_tensor = jt.array([])
    empty_vector = [ ]
    result = texture_grad_linear_mipmap_linear(tex, uv, dy, empty_tensor, empty_tensor, TextureMipWrapper(), empty_vector, filter_mode=filter_mode, boundary_mode=boundary_mode)
    return result[0],result[1]

def texture_grad_linear_mipmap_nearest(tex, uv, dy, uv_da, mip_level_bias, mip_wrapper, mip_stack, filter_mode, boundary_mode):
    
    result = texture_grad_linear_mipmap_linear(tex, uv, dy, uv_da, mip_level_bias, mip_wrapper, mip_stack, filter_mode, boundary_mode)
    return result[0], result[1], result[4], result[5], result[6], result[7],result[8], result[9], result[10], result[11],result[12], result[13], result[14], result[15]


if __name__ == "__main__":
    pass


