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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* viewmatrix_next,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
            const float* flowimg,
            float* flowconf,
			const bool prefiltered,
			const bool usecolor,
            const bool useflow,
			const bool trainpose,
			const bool updateconf,
			float* out_color,
            int* radii = nullptr,
			int* n_touched = nullptr,
			int* n_found = nullptr,
			float* weights_sum = nullptr,
            float* out_depth = nullptr,
            float* out_silh = nullptr,
			float* out_gsflow = nullptr,
			float* out_flowcost = nullptr);

		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* means3D,
			// int* stable_count,
			// bool* stable_status,
			const float* shs,
			const float* colors_precomp,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* viewmatrix_next,
			const float* relpose,
			const float* projmatrix,
			const float* campos,
			const float* flowimg,
			const float* flowconf,
			const float* gsflow,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			const float* dL_dsilh,
			const float* dL_dflowraw,
			const float* dL_dflow,
			const float* dL_daux,
			const float* dL_daux2,
			const float* dL_daux3,
			float* dL_derror,
			float* dL_derror2,
			float* dL_derror3,
			float* dL_dmean2D,
			float* dL_dmean2D_next,
			float* dL_dconic,
			float* dL_dcovmap_next,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			float* dL_dtau,
			float* dL_dtau_next,
			// float* out_grad_blending,
			const bool usecolor,
            const bool useflow,
			const bool trainpose,
			const bool useflowrawgrad);
	};
};

#endif
