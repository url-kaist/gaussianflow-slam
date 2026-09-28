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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int P, int W, int H,
		const float* bg_color,
		const float2* means2D,
		const float2* means2D_next,
		const float4* conic_opacity,
		const float4* covmap_next,
		// const uint2* rect_next_min,
		// const uint2* rect_next_max,
		// const bool* stable_status,
		const int* radii,
		const float* colors,
		const float* final_Ts,
		const uint32_t* n_contrib,
		const float* flowimg,
		const float* flowconf,
		const float* gsflow,
		const float* dL_dpixels,
		const float* dL_dsilh,
		const float* dL_dflowraw,
		const float* dL_dflowcost,
		const float* dL_daux,
		const float* dL_daux2,
		const float* dL_daux3,
		// const float* tile_loss_sum,
		// const float* tile_loss_sigma,
		float3* dL_dmean2D,
		float3* dL_dmean2D_next,
		float4* dL_dconic2D,
		float4* dL_dcovmap_next,
		float* dL_dopacity,
		float* dL_dcolors,
		float* dL_derror,
		float* dL_derror2,
		float* dL_derror3,
		const bool usecolor,
    	const bool useflow,
		const bool trainpose,
		const bool useflowrawgrad);

	void render_grad(
		const dim3 grid, const dim3 block,
		const uint2* ranges,					// == imgState.ranges
		const uint32_t* point_list,				// == binningState.point_list
		const float2* means2D,					// == geomState.means2D,
		int P, int W, int H,							
		const float4* conic_opacity,			// == geomState.conic_opacity,
		const int* radii,
		const float* final_Ts,					// == imgState.accum_alpha
		const uint32_t* n_contrib,				// == imgState.n_contrib
		float* dL_dmean3D,		// (output)
		float* out_grad_blending);

	void preprocess(
		int P, int D, int M,
		const float3* means,
		// int* stable_count,
		// bool* stable_status,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const float4* conic_opacity,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* cov3Ds,
		const float* view,
		const float* view_next,
		const float* relpose,
		const float* proj,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const glm::vec3* campos,
		const float3* dL_dmean2D,
		const float3* dL_dmean2D_next,
		const float* dL_dconics,
		const float* dL_dcovmap_next,
		const float* dL_derror,
		glm::vec3* dL_dmeans,
		float* dL_dcolor,
		float* dL_dcov3D,
		float* dL_dsh,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot,
		float* dL_dtau,
		float* dL_dtau_next,
		const bool usecolor,
    	const bool useflow,
		const bool trainpose);
}

#endif