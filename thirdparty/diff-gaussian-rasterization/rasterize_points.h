/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/torch.h>
#include <cstdio>
#include <tuple>
#include <string>
	
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& viewmatrix_next,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
    const torch::Tensor& flowimg,
    torch::Tensor& flowconf,
    const bool prefiltered,
	const bool usecolor,
    const bool useflow,
	const bool trainpose,
	const bool updateconf);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	// torch::Tensor& stable_count,
	// torch::Tensor& stable_status,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& viewmatrix_next,
    const torch::Tensor& projmatrix,
	const torch::Tensor& flowimg,
	const torch::Tensor& flowconf,
	const torch::Tensor& gsflow,
	const float tan_fovx, 
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_silh,
	const torch::Tensor& dL_dout_flowraw,
	const torch::Tensor& dL_dout_flow,
	const torch::Tensor& dL_dout_aux,
	const torch::Tensor& dL_dout_aux2,
	const torch::Tensor& dL_dout_aux3,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool usecolor,
    const bool useflow,
	const bool trainpose,
	const bool useflowrawgrad);

// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, // 5
// torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,  // 5 
// torch::Tensor, torch::Tensor, torch::Tensor>
// RasterizeGaussiansJacobianHessian(
// 	const torch::Tensor& background,
// 	const torch::Tensor& means3D,
// 	const torch::Tensor& radii,
//     const torch::Tensor& colors,
// 	const torch::Tensor& scales,
// 	const torch::Tensor& rotations,
// 	const float scale_modifier,
// 	const torch::Tensor& cov3D_precomp,
// 	const torch::Tensor& viewmatrix,
// 	const torch::Tensor& viewmatrix_next,
//     const torch::Tensor& projmatrix,
// 	const torch::Tensor& flowimg,
// 	const torch::Tensor& flowconf,
// 	const torch::Tensor& gsflow,
// 	const float tan_fovx,
// 	const float tan_fovy,
// 	const torch::Tensor& sh,
// 	const int degree,
// 	const torch::Tensor& campos,
// 	const torch::Tensor& geomBuffer,
// 	const int R,
// 	const torch::Tensor& binningBuffer,
// 	const torch::Tensor& imageBuffer,
// 	const bool usecolor,
//     const bool useflow);
		
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);
