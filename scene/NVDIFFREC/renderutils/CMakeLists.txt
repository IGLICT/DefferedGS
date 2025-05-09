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

cmake_minimum_required(VERSION 3.20)

project(Texture LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(BUILD_SHARED_LIBS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_library(RenderUtilsJittor
	c_src/vec3f.h
	c_src/vec4f.h
	c_src/mesh.h
    c_src/loss.h
    c_src/bsdf.h
    c_src/normal.h
    c_src/cubemap.h
    c_src/common.h
    c_src/common.cpp
)

set_target_properties(RenderUtilsJittor PROPERTIES CUDA_ARCHITECTURES "70;75;86")
target_include_directories(RenderUtilsJittor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/common)
target_include_directories(RenderUtilsJittor PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} )


