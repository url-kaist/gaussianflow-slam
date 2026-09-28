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

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{
		size_t scan_size;
		float* depths;
		char* scanning_space;
		bool* clamped;			// if false, the color is negative, and not will be used.
		int* internal_radii;
		// uint2* rect_next_min;
		// uint2* rect_next_max;
		float2* means2D;
		float2* means2D_next;
		float* cov3D;
		float4* conic_opacity;
		float4* covmap_next;
		float* rgb;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint2* ranges;
		uint32_t* n_contrib;
		float* accum_alpha;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	
	template<typename T> 
	size_t required(size_t P)
	{
		// since 'size pointer' is started from nullptr,
		// size (== chunck) after T::fromChunk becomes the memory size to obtian P numbers of types
		// size: how many bytes needs to calculated P points.
		
		// For example of GeometryState::fromChunk()
		// depths(P), clamped(P*3), internal_radii(P), means2D(P), cov3D(P*6), conic_opacity(P), 
		// rgb(P*3), tiles_touched(P), scanning_space(same size of tiles_touched, P), points_offset(P)
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};