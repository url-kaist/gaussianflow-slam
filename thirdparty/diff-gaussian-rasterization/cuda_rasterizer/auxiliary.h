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
/* 
	Copyright:           © YEARS (e.g. 2018-2020) by Stevens Institute of Technology
	Contributors:        Dr. Enrique Dunn
	Affiliation:         Stevens Institute of Technology
	URL:                 https://www.stevens.edu/directory/office-innovation-and-entrepreneurship
	Citation:            
	Min, Zhixiang, Yiding Yang, and Enrique Dunn. "VOLDOR: Visual Odometry From Log-Logistic Dense Optical Flow Residuals." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
	Min, Zhixiang, and Enrique Dunn. "VOLDOR-SLAM: For the Times When Feature-Based or Direct Methods Are Not Good Enough." arXiv preprint arXiv:2104.06800 (2021).
 
	All rights reserved.
 */


#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

/* For calculating weight of optical flow */
#define EST_RF 1.0 // 반으로 downsampling한 경우 optical flow도 반으로 줄임.
#define FISK_A1 0.01f
#define FISK_A2 0.09f
#define FISK_B1 1.0f
#define FISK_B2 -0.0022f
#define MIN_OBS_FMAG 2.f
#define MAX_OBS_FMAG 100.f
#define LAMBDA 0.15f //0.15f
#define ZDE FLT_EPSILON
// #define ZDE 0.0000000000000001f
#define L2_NORM(x,y) ( sqrtf( (x)*(x) + (y)*(y) ) )
/*******************************************/

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__forceinline__ __device__ float3& operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__forceinline__ __device__ float2& operator+=(float2& a, const float2& b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}

__forceinline__ __device__ float4& operator+=(float4& a, const float4& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}

__forceinline__ __device__ float fast_powf(float base, float exponent) {
    return __expf(exponent * __logf(base + 1e-6f));  
}

__forceinline__ __device__ float pow1p5(float x) {
    return x * sqrtf(x);  // pow(x, 1.5f)
}

__forceinline__ __device__ float pow3(float x) {
    float x2 = x * x;
    return x * x2;
}


__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

// viewmatrix를 받을때 transpose해서 받으므로
// 이 함수는 그냥 행렬곱처럼 보여도 transpose를 곱해주는게 맞음 (... 왜 transpose해서 input으로 넣어주는지는 모르겠음)
__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / fmaxf(sqrt(sum2 * sum2 * sum2),ZDE);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / fmaxf(sqrt(sum2 * sum2 * sum2),ZDE);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / fmaxf(sqrt(sum2 * sum2 * sum2),ZDE);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	// float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	// float p_w = 1.0f / (p_hom.w + 0.0000001f);
	// float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	// p_view = transformPoint4x3(p_orig, viewmatrix);

	p_view = transformPoint4x3(p_orig, viewmatrix);
	float4 p_hom = transformPoint4x4(p_view, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	if (p_view.z <= 0.01f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* viewmatrix_next, 
	const float* projmatrix,
	bool prefiltered,
	float3& p_view,
	float3& p_view_next)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	// float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	// float p_w = 1.0f / (p_hom.w + 0.0000001f);
	// float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	// p_view = transformPoint4x3(p_orig, viewmatrix);

	p_view = transformPoint4x3(p_orig, viewmatrix);
	p_view_next = transformPoint4x3(p_orig, viewmatrix_next);
	float4 p_hom = transformPoint4x4(p_view, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// 250317
	// if (p_view.z <= 0.2f || p_view_next.z <= 0.2f) //  || p_view_next.z <= 0.2f
	if (p_view.z <= 0.01f) //  || p_view_next.z <= 0.2f
	// if (p_view.z <= 0.2f || p_view_next.z <= 0.2f || ((p_proj.x < -1.0 || p_proj.x > 1.0 || p_proj.y < -1.0 || p_proj.y > 1.0)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

/* For calculating weight of optical flow */

// a/(x+b)+(1-a/b)
__device__ __forceinline__ static float fun_fmag_c(float fmag) {
	fmag = fminf(fmaxf(fmag*EST_RF, MIN_OBS_FMAG), MAX_OBS_FMAG);
	return FISK_B1 + FISK_B2 * fmag;
}

// a*exp(b*x)
__device__ __forceinline__ static float fun_fmag_scale(float fmag) {
	fmag = fminf(fmaxf(fmag*EST_RF, MIN_OBS_FMAG), MAX_OBS_FMAG);
	return FISK_A1 * expf(FISK_A2 * fmag);
}


// c*(x/s)^(-c-1)*(1+(x/s).^(-c))^(-2)/s
// c => beta, scale => alpha
__device__ __forceinline__ static float fisk_dist_pdf(float x, float c, float scale) {
	x = fmaxf(x*EST_RF, ZDE);
	// beta * (x^2 / alpha)^(-beta - 1) / (1 + (x^2 / alpha)^(-beta))^2 / alpha
	// 약분하면 같은식이긴 한데 굳이 이렇게 한 이유..? x^y에서 y가 0에 덜가깝게 하려고?
	return (c * fast_powf((x*x) / scale, -c - 1.f) * fast_powf(1 + fast_powf((x*x) / scale, -c), -2.f)) / scale;
}


__device__ __forceinline__ static float fun_rigidness(float dx1, float dy1, float dx2, float dy2) {
	const float obs_fmag = L2_NORM(dx2, dy2);
	const float diff_fmag = L2_NORM(dx1 - dx2, dy1 - dy2);
	const float c = fun_fmag_c(obs_fmag); // beta
	const float s = fun_fmag_scale(obs_fmag); // alpha
	const float fisk_prob = fisk_dist_pdf(diff_fmag, c, s);
	const float mu = fisk_dist_pdf(LAMBDA*obs_fmag, c, s);
	return fisk_prob / (fisk_prob + mu);
}

__device__ __forceinline__ static void fun_cost(float dx1, float dy1, float dx2, float dy2, float in_weight,
float& out_cost, float& out_weight, const bool update_conf)
{
	float prob = fun_rigidness(dx1, dy1, dx2, dy2);
	prob = fmaxf(prob, 1e-6);
	out_cost = -in_weight * logf(prob);
	// if (update_conf)
	// 	out_weight = 0.01 * prob + 0.99 * in_weight;
	if (update_conf)
		out_weight = prob;
	
	// out_weight = prob;
	// else
		// out_weight = 0.1 * prob + 0.9 * in_weight;
		// out_weight = 0.1 * prob + 0.9 * in_weight;
	// if (update_conf)
	// out_cost = -weight * logf(fun_rigidness(dx1, dy1, dx2, dy2));
}

/*******************************************/



/******************** Speedy-splat *********************/

__device__ inline float2 computeEllipseIntersection(
    const float4 con_o, const float disc, const float t, const float2 p,
    const bool isY, const float coord)
{
    float p_u = isY ? p.y : p.x;
    float p_v = isY ? p.x : p.y;
    float coeff = isY ? con_o.x : con_o.z;

    float h = coord - p_u;  // h = y - p.y for y, x - p.x for x
    float sqrt_term = sqrt(disc * h * h + t * coeff);

    return {
      (-con_o.y * h - sqrt_term) / coeff + p_v,
      (-con_o.y * h + sqrt_term) / coeff + p_v
    };
}

__device__ inline uint32_t processTiles(
    const float4 con_o, const float disc, const float t, const float2 p,
    float2 bbox_min, float2 bbox_max,
    float2 bbox_argmin, float2 bbox_argmax,
    int2 rect_min, int2 rect_max,
    const dim3 grid, const bool isY,
    uint32_t idx, uint32_t off, float depth,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted
    )
{

    // ---- AccuTile Code ---- //

    // Set variables based on the isY flag
    float BLOCK_U = isY ? BLOCK_Y : BLOCK_X;
    float BLOCK_V = isY ? BLOCK_X : BLOCK_Y;

    if (isY) {
      rect_min = {rect_min.y, rect_min.x};
      rect_max = {rect_max.y, rect_max.x};

      bbox_min = {bbox_min.y, bbox_min.x};
      bbox_max = {bbox_max.y, bbox_max.x};

      bbox_argmin = {bbox_argmin.y, bbox_argmin.x};
      bbox_argmax = {bbox_argmax.y, bbox_argmax.x};
    }

    uint32_t tiles_count = 0;
    float2 intersect_min_line, intersect_max_line;
    float ellipse_min, ellipse_max;
    float min_line, max_line;

    // Initialize max line
    // Just need the min to be >= all points on the ellipse
    // and  max to be <= all points on the ellipse
    intersect_max_line = {bbox_max.y, bbox_min.y};

    min_line = rect_min.x * BLOCK_U;
    // Initialize min line intersections.
    if (bbox_min.x <= min_line) {
      // Boundary case
      intersect_min_line = computeEllipseIntersection(
                con_o, disc, t, p, isY, rect_min.x * BLOCK_U);

    } else {
      // Same as max line
      intersect_min_line = intersect_max_line;
    }


    // Loop over either y slices or x slices based on the `isY` flag.
    for (int u = rect_min.x; u < rect_max.x; ++u)
    {
        // Starting from the bottom or left, we will only need to compute
        // intersections at the next line.
        max_line = min_line + BLOCK_U;
        if (max_line <= bbox_max.x) {
          intersect_max_line = computeEllipseIntersection(
                    con_o, disc, t, p, isY, max_line);
        }

        // If the bbox min is in this slice, then it is the minimum
        // ellipse point in this slice. Otherwise, the minimum ellipse
        // point will be the minimum of the intersections of the min/max lines.
        if (min_line <= bbox_argmin.y && bbox_argmin.y < max_line) {
          ellipse_min = bbox_min.y;
        } else {
          ellipse_min = min(intersect_min_line.x, intersect_max_line.x);
        }

        // If the bbox max is in this slice, then it is the maximum
        // ellipse point in this slice. Otherwise, the maximum ellipse
        // point will be the maximum of the intersections of the min/max lines.
        if (min_line <= bbox_argmax.y && bbox_argmax.y < max_line) {
          ellipse_max = bbox_max.y;
        } else {
          ellipse_max = max(intersect_min_line.y, intersect_max_line.y);
        }

        // Convert ellipse_min/ellipse_max to tiles touched
        // First map back to tile coordinates, then subtract.
        int min_tile_v = max(rect_min.y,
            min(rect_max.y, (int)(ellipse_min / BLOCK_V))
            );
        int max_tile_v = min(rect_max.y,
            max(rect_min.y, (int)(ellipse_max / BLOCK_V + 1))
            );

        tiles_count += max_tile_v - min_tile_v;
        // Only update keys array if it exists.
        if (gaussian_keys_unsorted != nullptr) {
          // Loop over tiles and add to keys array
          for (int v = min_tile_v; v < max_tile_v; v++)
          {
            // For each tile that the Gaussian overlaps, emit a
            // key/value pair. The key is |  tile ID  |      depth      |,
            // and the value is the ID of the Gaussian. Sorting the values
            // with this key yields Gaussian IDs in a list, such that they
            // are first sorted by tile and then by depth.
            uint64_t key = isY ?  (u * grid.x + v) : (v * grid.x + u);
            key <<= 32;
            key |= *((uint32_t*)&depth);
            gaussian_keys_unsorted[off] = key;
            gaussian_values_unsorted[off] = idx;
            off++;
          }
        }
        // Max line of this tile slice will be min lin of next tile slice
        intersect_min_line = intersect_max_line;
        min_line = max_line;
    }
    return tiles_count;
}

__device__ inline uint32_t duplicateToTilesTouched(
    const float2 p, const float4 con_o, const dim3 grid,
    uint32_t idx, uint32_t off, float depth,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted
    )
{

    //  ---- SNUGBOX Code ---- //
	bool zero_tiles = false;

    // Calculate discriminant
    float disc = con_o.y * con_o.y - con_o.x * con_o.z;

    // If ill-formed ellipse, return 0
    if (gaussian_keys_unsorted == nullptr && (con_o.x <= 0 || con_o.z <= 0 || disc >= 0)) {
        return 0;
		// zero_tiles = true;
    }

    // Threshold: opacity * Gaussian = 1 / 255
    float t = 2.0f * log(con_o.w * 255.0f);
	// FastGS
	// t = 0.5f * t;

    float x_term = sqrt(-(con_o.y * con_o.y * t) / (disc * con_o.x));
    x_term = (con_o.y < 0) ? x_term : -x_term;
    float y_term = sqrt(-(con_o.y * con_o.y * t) / (disc * con_o.z));
    y_term = (con_o.y < 0) ? y_term : -y_term;

    float2 bbox_argmin = { p.y - y_term, p.x - x_term };
    float2 bbox_argmax = { p.y + y_term, p.x + x_term };

    float2 bbox_min = {
      computeEllipseIntersection(con_o, disc, t, p, true, bbox_argmin.x).x,
      computeEllipseIntersection(con_o, disc, t, p, false, bbox_argmin.y).x
    };
    float2 bbox_max = {
      computeEllipseIntersection(con_o, disc, t, p, true, bbox_argmax.x).y,
      computeEllipseIntersection(con_o, disc, t, p, false, bbox_argmax.y).y
    };

    // Rectangular tile extent of ellipse
    int2 rect_min = {
        max(0, min((int)grid.x, (int)(bbox_min.x / BLOCK_X))),
        max(0, min((int)grid.y, (int)(bbox_min.y / BLOCK_Y)))
    };
    int2 rect_max = {
        max(0, min((int)grid.x, (int)(bbox_max.x / BLOCK_X + 1))),
        max(0, min((int)grid.y, (int)(bbox_max.y / BLOCK_Y + 1)))
    };

    int y_span = rect_max.y - rect_min.y;
    int x_span = rect_max.x - rect_min.x;

    // If no tiles are touched, return 0
    if (gaussian_keys_unsorted == nullptr && y_span * x_span == 0) {
        return 0;
		// zero_tiles = true;
    }

    // If fewer y tiles, loop over y slices else loop over x slices
	bool isY = y_span < x_span;
    return processTiles(
        con_o, disc, t, p,
        bbox_min, bbox_max,
        bbox_argmin, bbox_argmax,
        rect_min, rect_max,
        grid, isY,
        idx, off, depth,
        gaussian_keys_unsorted,
        gaussian_values_unsorted
    );

    // bool isY = y_span < x_span;
	// uint32_t nonzero_tiles = 0;
	// if (gaussian_keys_unsorted == nullptr && zero_tiles)
	// 	return 0;
	
	// nonzero_tiles = processTiles(
	// 	con_o, disc, t, p,
	// 	bbox_min, bbox_max,
	// 	bbox_argmin, bbox_argmax,
	// 	rect_min, rect_max,
	// 	grid, isY,
	// 	idx, off, depth,
	// 	gaussian_keys_unsorted,
	// 	gaussian_values_unsorted
	// );

	// if (zero_tiles)
	// 	return 0;
	// else
	// 	return nonzero_tiles;
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

/*******************************************************/


#endif