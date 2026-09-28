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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& viewspace_mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.

	// 3D point in "camera coordinate" 
	// float3 t = transformPoint4x3(mean, viewmatrix);
	float3 t = viewspace_mean;

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;

	// clipping if t.x, t.y exceed the range
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	// Jacobian (affine approximation) of the perspective projection 
	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

__device__ void computeCov3D_safe(const glm::vec3& scale_in, float mod_in, const glm::vec4& rot_in, float* cov3D) {
    const float eps_scale = 1e-6f;
    const float eps_mod = 1e-6f;
    const float eps_sigma = 1e-6f;

    // 1. Clamp scale to avoid zero or negative values
    glm::vec3 scale = glm::max(scale_in, glm::vec3(eps_scale));

    // 2. Clamp mod to avoid degenerate scaling
    float mod = fmaxf(mod_in, eps_mod);

    // 3. Create scaling matrix
    glm::mat3 S(1.0f);
    S[0][0] = mod * scale.x;
    S[1][1] = mod * scale.y;
    S[2][2] = mod * scale.z;

    // 4. Normalize quaternion (careful about order)
    glm::vec4 q = rot_in;
    float norm_q = glm::length(q);
    if (norm_q < 1e-6f) {
        q = glm::vec4(1, 0, 0, 0);  // fallback identity rotation
    } else {
        q /= norm_q;
    }

    float r = q.x, x = q.y, y = q.z, z = q.w;

    // 5. Rotation matrix from quaternion
    glm::mat3 R(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z),     2.f * (x * z + r * y),
        2.f * (x * y + r * z),     1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y),     2.f * (y * z + r * x),     1.f - 2.f * (x * x + y * y)
    );

    // 6. Compute final transform
    glm::mat3 M = S * R;

    // 7. 3D covariance
    glm::mat3 Sigma = glm::transpose(M) * M;

    // 8. Fix invalid diagonal elements
    for (int i = 0; i < 3; ++i) {
        if (Sigma[i][i] <= 0.f || isnan(Sigma[i][i]) || isinf(Sigma[i][i])) {
            Sigma[i][i] = eps_sigma;
        }
    }

    // 9. Write upper-triangular part (column-major order for Gaussian renderer)
    cov3D[0] = Sigma[0][0];
    cov3D[1] = Sigma[0][1];
    cov3D[2] = Sigma[0][2];
    cov3D[3] = Sigma[1][1];
    cov3D[4] = Sigma[1][2];
    cov3D[5] = Sigma[2][2];
}


// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	// ORDER of quaternion changed...!!!!!!!! (because of casting float* => glm:vec4(x,y,z,w))
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Reference: Charles-Alban Deledalle, Loic Denis, Sonia Tabti, Florence Tupin. Closed-form expressions of the
// eigen decomposition of 2 x 2 and 3 x 3 Hermitian matrices. [Research Report] Université de Lyon. 2017.
__device__ void computeSqrtCov2DInv(float3 cov2D, glm::mat2& sqrtcov2D, glm::mat2& sqrtcov2Dinv)
{
	float delta = sqrt(4*cov2D.y*cov2D.y + (cov2D.x - cov2D.z)*(cov2D.x - cov2D.z));
	float lambda1 = (cov2D.x + cov2D.z - delta) / 2.0f;
	float lambda2 = (cov2D.x + cov2D.z + delta) / 2.0f;

	float2 v1 = { 1.0f, 0.0f };
	float2 v2 = { 0.0f, 1.0f };

	if (abs(cov2D.y) > ZDE)
	{
		float v11 = (lambda1 - cov2D.z) / (cov2D.y);
		float v21 = (lambda2 - cov2D.z) / (cov2D.y);
		float v1_norm = sqrt(1 + v11 * v11);
		float v2_norm = sqrt(1 + v21 * v21);
		v1 = { v11 / v1_norm, 1.0f / v1_norm };
		v2 = { v21 / v2_norm, 1.0f / v2_norm };
	}

	if (lambda1 > ZDE && lambda2 > ZDE)
	{
		glm::mat2 Q = glm::mat2(v1.x, v1.y, v2.x, v2.y);
		glm::mat2 S_sqrt = glm::mat2(sqrt(lambda1), 0, 0, sqrt(lambda2));
		glm::mat2 S_sqrt_inv = glm::mat2(1.0f / sqrt(lambda1), 0, 0, 1.0f / sqrt(lambda2));

		sqrtcov2D = Q * S_sqrt * glm::transpose(Q);
		sqrtcov2Dinv = Q * S_sqrt_inv * glm::transpose(Q);
	}

	// sqrtcov2D.x = result1[0][0];
	// sqrtcov2D.y = result1[0][1];
	// sqrtcov2D.z = result1[1][1];

	// sqrtcov2Dinv.x = result2[0][0];
	// sqrtcov2Dinv.y = result2[0][1];
	// sqrtcov2Dinv.z = result2[1][1];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<bool UseColor, bool UseFlow, int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points, 		// == means3D
	const glm::vec3* scales,  		
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,					// == geomState.clamped
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,		// == transpose of Tcw
	const float* viewmatrix_next,		// == transpose of Tcw
	const float* projmatrix,		// == transpose of Tiw	
	const glm::vec3* cam_pos,		// float* => glm::vec3*
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	// uint2* rect_next_min,
	// uint2* rect_next_max,
	float2* points_xy_image,		// == geomState.means2D
	float2* points_xy_image_next,		// == geomState.means2D_next
	float* depths,					// == geomState.depths
	float* cov3Ds,					// == geomState.cov3D
	float* rgb,						// == geomState.rgb
	float4* conic_opacity,			// == geomState.conic_opacity
	float4* covmap_next,			// == geomState.covmap_next
	const dim3 grid,				// == tile_grid: number of blocks in 2D
	uint32_t* tiles_touched,		// == geomState.tiles_touched
	// bool usecolor,
	// bool useflow,
	bool prefiltered)				// == false
{
	// auto idx = cg::this_grid().thread_rank();
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view, p_view_next;

	// p_view: 3D points (mean) in "camera coordinate", so z value becomes depth
	if constexpr (UseColor)
	{
		if (!in_frustum(idx, orig_points, viewmatrix, viewmatrix_next, projmatrix, prefiltered, p_view, p_view_next))
			return;
	}
	else if constexpr (UseFlow)
	{
		// 250317
		if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
			return;
	}

	// Transform point by projecting
	// float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	// float4 p_hom = transformPoint4x4(p_orig, projmatrix); // p_hom: 3D points in image coordinate (not camera!!!)
	float4 p_hom = transformPoint4x4(p_view, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	float4 p_hom_next, p_proj_next;
	float p_w_next;

	if constexpr (UseFlow)
	{
		p_hom_next = transformPoint4x4(p_view_next, projmatrix);
		p_w_next = 1.0f / (p_hom_next.w + 0.0000001f);
		p_proj_next = { p_hom_next.x * p_w_next, p_hom_next.y * p_w_next, p_hom_next.z * p_w_next };
	}
	// Get projected point in the next frame

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		// computeCov3D_safe(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	// float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);
	float3 cov = computeCov2D(p_view, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);
	float3 cov_next = {0.0f, 0.0f, 0.0f};
	if constexpr (UseFlow)
		cov_next = computeCov2D(p_view_next, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix_next);

	// float3 Lcov_inv = { 1.f / sqrt(cov.x), -cov.y / (sqrt(cov.x * cov.x * cov.z - cov.x * cov.y * cov.y)), 
	// 	sqrt(cov.x) / sqrt(cov.x * cov.z - cov.y * cov.y) };
	// float3 Lcov_next = { sqrt(cov_next.x), cov_next.y / sqrt(cov_next.x), 
	// 	sqrt(cov_next.x * cov_next.z - cov_next.y * cov_next.y) / sqrt(cov_next.x) };

	glm::mat2 sqrtcov2D      = glm::mat2(0.0f);
	glm::mat2 sqrtcov2Dinv   = glm::mat2(0.0f);
	glm::mat2 sqrtcov2D_next = glm::mat2(0.0f);
	glm::mat2 sqrtcov2Dinv_next = glm::mat2(0.0f);
	glm::mat2 covmap         = glm::mat2(0.0f);

	if constexpr (UseFlow)
	{
		computeSqrtCov2DInv(cov, sqrtcov2D, sqrtcov2Dinv);
		computeSqrtCov2DInv(cov_next, sqrtcov2D_next, sqrtcov2Dinv_next);
		covmap = sqrtcov2D_next * sqrtcov2Dinv;
	}
	// Invert covariance (EWA algorithm)
	// cov.x: cov[0][0]
	// cov.y: cov[0][1]
	// cov.z: cov[1][1]

	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	// conic: inverse of covariance
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// float det_next = (cov_next.x * cov_next.z - cov_next.y * cov_next.y);
	// if (det_next == 0.0f)
	// 	  return;
	// float det_inv_next = 1.f / det_next;
	// conic: inverse of covariance
	// float3 conic_next = { cov_next.z * det_inv_next, -cov_next.y * det_inv_next, cov_next.x * det_inv_next };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z); // trace of 2D matrix 
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det)); // eigenvalue_1
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det)); // eigenvalue_2
	// float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2))); // 99.7% => 3sigma, ceiling for tile rendering
	int my_radius = max(int(ceil(3.f * sqrt(max(lambda1, lambda2)))), 1); // 99.7% => 3sigma, ceiling for tile rendering
	// if (!isfinite(my_radius))
	// 	return;

	// ndc coordinate => pixel coordinate 
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };

	float2 point_image_next = {0.0f, 0.0f};
	if constexpr (UseFlow)
		point_image_next = { ndc2Pix(p_proj_next.x, W), ndc2Pix(p_proj_next.y, H) };

	uint2 rect_min = {0, 0};
	uint2 rect_max = {0, 0};
	// if (useflow)
	// {
	// 	my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	// }

	// **IMPORTANT**
	// rect_min, rect_max => the range of "blocks" where the radius can be involved

	/******************** Original code ********************/
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	/*******************************************************/

	/******************** Speedy-splat *********************/
	// float4 con_o = { conic.x, conic.y, conic.z, opacities[idx] };
	// uint32_t tiles_count = duplicateToTilesTouched(
	// 	point_image, con_o, grid,
	// 	0, 0, 0,
	// 	nullptr, nullptr);
	// if (tiles_count == 0)
	//   	return;
	/*******************************************************/

	// 250317
	// if (useflow)
	// {
	// 	float det_next = (cov_next.x * cov_next.z - cov_next.y * cov_next.y);
	// 	// if (det_next == 0.0f)
	// 	// 	  return;
	// 	if (det_next != 0.0f)
	// 	{
	// 		float det_inv_next = 1.f / det_next;
	// 		float3 conic_next = { cov_next.z * det_inv_next, -cov_next.y * det_inv_next, cov_next.x * det_inv_next };

	// 		float mid_next = 0.5f * (cov_next.x + cov_next.z); // trace of 2D matrix
	// 		float lambda1_next = mid_next + sqrt(max(0.1f, mid_next * mid_next - det_next)); // eigenvalue_1
	// 		float lambda2_next = mid_next - sqrt(max(0.1f, mid_next * mid_next - det_next)); // eigenvalue_2
	// 		// float my_radius_next = ceil(3.f * sqrt(max(lambda1_next, lambda2_next))); // 99.7% => 3sigma, ceiling for tile rendering
	// 		int my_radius_next = max(int(ceil(3.f * sqrt(max(lambda1_next, lambda2_next)))),1); // 99.7% => 3sigma, ceiling for tile rendering
	// 		if (!isfinite(my_radius_next))
	// 			return;

	// 		uint2 rect_min_next = {0, 0};
	// 		uint2 rect_max_next = {0, 0};
	// 		getRect(point_image_next, my_radius_next, rect_min_next, rect_max_next, grid);
	// 		// if ((rect_max_next.x - rect_min_next.x) * (rect_max_next.y - rect_min_next.y) == 0)
	// 		// 	return;
	// 		// if ((rect_max_next.x - rect_min_next.x) * (rect_max_next.y - rect_min_next.y) != 0)
	// 		// {
	// 		// 	rect_next_min[idx] = rect_min_next;
	// 		// 	rect_next_max[idx] = rect_max_next;
	// 		// }
	// 	}
	// }
	
	/******************** Original code ********************/
	if ( !((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0) )
	{	
	/*******************************************************/
		// If colors have been precomputed, use them, otherwise convert
		// spherical harmonics coefficients to RGB color.
		if constexpr (UseColor)
		{
			if (colors_precomp == nullptr)
			{
				// SH => RGB
				// clamped: if RGB value of each pixel is clamped, true
				glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
				rgb[idx * C + 0] = result.x;
				rgb[idx * C + 1] = result.y;
				rgb[idx * C + 2] = result.z;
			}
		}

		// Store some useful helper data for the next steps.
		// glmmat[col][row]
		depths[idx] = p_view.z; // z value of mean3D in camera coordinate
		radii[idx] = my_radius; // 3 * max eigenvalue for each GS
		points_xy_image[idx] = point_image; // 2D pixel coordinate of each GS
		// Inverse 2D covariance and opacity neatly pack into one float4

		/******************** Original code ********************/
		conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
		tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x); // area of a sqaure representing range
		/*******************************************************/
		// covmap_next[idx] = { Lcov_next.x * Lcov_inv.x, 0, Lcov_next.y * Lcov_inv.y + Lcov_next.z * Lcov_inv.y,
		// 	Lcov_next.z * Lcov_inv.z };

		/******************** Speedy-splat *********************/
		// conic_opacity[idx] = con_o;
  		// tiles_touched[idx] = tiles_count;
		/*******************************************************/

		if constexpr (UseFlow)
		{
			points_xy_image_next[idx] = point_image_next; // 2D pixel coordinate of each GS
			covmap_next[idx] = { covmap[0][0], covmap[1][0], covmap[0][1], covmap[1][1] };
		}
	/******************** Original code ********************/
	}
	/*******************************************************/
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <bool UseColor, bool UseFlow, uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,  				// == imgState.ranges (closest index / farthest) index of binningState.point_list_keys in each tile
	const uint32_t* __restrict__ point_list,		// == binningState.point_list. All sorted GS index by tile->depth	
	int W, int H,							
	const float2* __restrict__ points_xy_image,		// == geomState.means2D
	const float2* __restrict__ points_xy_image_next,// == geomState.means2D
	const float* __restrict__ points_depth_image,	// == geomState.depths
	const float* __restrict__ features,				// == geomState.rgb (rgb from SH, seen by viewmatrx)
	const float4* __restrict__ conic_opacity,		// == geomState.conic_opacity		
	const float4* __restrict__ covmap_next,	// == geomState.conic_opacity		
	// const uint2* __restrict__ rect_next_min,
	// const uint2* __restrict__ rect_next_max,
	float* __restrict__ final_T,					// == imgState.accum_alpha (not calculated yet)
	uint32_t* __restrict__ n_contrib,				// == imgState.n_contrib   (not calculated yet)		
	const float* __restrict__ bg_color,
	const float* __restrict__ flowimg,
	float* __restrict__ flowconf,
	// const bool usecolor,
	// const bool useflow,
	// const bool trainpose,
	const bool updateconf,
	float* __restrict__ out_color,
	float* __restrict__ out_depth,
	float* __restrict__ out_silh,
	float* __restrict__ out_gsflow,
	float* __restrict__ out_flowcost,
	int * __restrict__ n_touched,
	int * __restrict__ n_found,
	float* __restrict__ weights_sum)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();    // current tile(block) index
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X; // max horizontal number of blocks
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y }; 	// miminum pixel-x / pixel-y of this tile
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };			// maximum pixel-x / pixel-y of this tile
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y }; // current pixel position
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };
	float2 pixf_next = {0.0f, 0.0f};

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// 250317

	if constexpr (UseFlow)
	{
		if (inside)
		{
			pixf_next = { pixf.x + flowimg[pix_id], pixf.y + flowimg[H * W + pix_id] };
		}
	}

	// 250317 done으로 해도되나?
	// bool done_flow = false;
	// // if (useflow)
	// // {
	// // 	bool inside_next = pixf_next.x >= 0 && pixf_next.x < W && pixf_next.y >= 0 && pixf_next.y < H;
	// // 	inside = inside && inside_next;
	// // 	done = !inside;
	// // }
	// if (useflow && inside)
	// {
	// 	bool inside_next_flow = pixf_next.x >= 0 && pixf_next.x < W && pixf_next.y >= 0 && pixf_next.y < H;
	// 	inside_next_flow = inside && inside_next_flow;
	// 	done_flow = !inside_next_flow;
	// }

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE); // total rounds needed for per-block_size calculation
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	// ***** shared memory ***** much faster!!
	// this meory is shared between threads in the block
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float2 collected_xy_next[BLOCK_SIZE];
	__shared__ float collected_depth[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float4 collected_covmap_next[BLOCK_SIZE];
	// __shared__ uint2 collected_rect_next_min[BLOCK_SIZE];
	// __shared__ uint2 collected_rect_next_max[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	uint32_t pixel_contributor = 0;
	float C[CHANNELS] = { 0 };
	float Flow[2] = { 0 };
	float D = 0;
	float Silh = 0;

	/***** DEBUG *****/
	// uint32_t first3_count = 0;
	// float top_weight = 0;
	// float top_alpha = 0;
	// float second_alpha = 0;
	// float first3_weights[3] = { 0 };
	/*****************/
	
	// Iterate over batches until all done or range is complete
	// For each pixel(thread) in this tile(block), contributing Gaussians are used in the calculations 
	// by iterating through each block_size.
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		// get the threads that is done (not inside)
		// if there are 'done' pixels (not in the image or all )
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_depth[block.thread_rank()] = points_depth_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			
			if constexpr (UseFlow)
			{
				collected_xy_next[block.thread_rank()] = points_xy_image_next[coll_id];
				collected_covmap_next[block.thread_rank()] = covmap_next[coll_id];
				// collected_rect_next_min[block.thread_rank()] = rect_next_min[coll_id];
				// collected_rect_next_max[block.thread_rank()] = rect_next_max[coll_id];
			}
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			// a1 + (1-a1)a2 + (1-a1)(1-a2)a3 + ... => T becomes accumulated alphas
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}
			
			/***** DEBUG *****/
			// if (test_T > 0.01f)
			// {
			// 	pixel_contributor++;
			// 	if (first3_count < 3)
			// 		first3_weights[first3_count] = alpha * T;
			// 	if (alpha * T > top_weight)
			// 		top_weight = alpha * T;
			// 	if (alpha > top_alpha)
			// 		top_alpha = alpha;
			// 	if (alpha > second_alpha && alpha < top_alpha)
			// 		second_alpha = alpha;
			// 	first3_count++;
			// }
			/*****************/

			// Eq. (3) from 3D Gaussian splatting paper.
			// if (usecolor)
			if constexpr (UseColor)
			{
				for (int ch = 0; ch < CHANNELS; ch++)
					C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
			}

			D += collected_depth[j] * alpha * T;
			if (test_T > 0.5f) 
			{
				atomicAdd(&(n_touched[collected_id[j]]), 1);
			}
			
			// 각 Gaussian이 ray상에서 visible하게 보인 pixel수
			if (test_T > 0.0001f)
				atomicAdd(&(n_found[collected_id[j]]), 1);
			

			// For 2nd-order optimization / Gaussian의 importance의 합
			atomicAdd(&(weights_sum[collected_id[j]]), alpha * T);

			// Silh += alpha * T;

			// 250317
			// if (useflow && con_o.w > 0.5f)
			// if (useflow) // && !done_flow
			if constexpr (UseFlow)
			{
				// if ((collected_rect_next_min[j].x * BLOCK_X < pixf_next.x) && (collected_rect_next_max[j].x * BLOCK_X > pixf_next.x)
				// 	&& (collected_rect_next_min[j].y * BLOCK_Y < pixf_next.y) && (pixf_next.y < collected_rect_next_max[j].y * BLOCK_Y))
				{
					float f1 = (collected_covmap_next[j].x * -d.x + collected_covmap_next[j].y * -d.y 
						+ collected_xy_next[j].x - pixf.x);
					float f2 = (collected_covmap_next[j].z * -d.x + collected_covmap_next[j].w * -d.y 
						+ collected_xy_next[j].y - pixf.y);
					Flow[0] += f1 * alpha * T;
					Flow[1] += f2 * alpha * T;
				}
				// Flow[0] += f1 * alpha * T;
				// Flow[1] += f2 * alpha * T;
			}
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;  // represents how many gaussian splattings contributed to render this pixel

		/***** DEBUG *****/
		// if (pix_id % 100000 == 0 && false)
		// {
		// 	printf("pixel_contributor [pix_id: %d]: %d\n", pix_id, pixel_contributor);
		// 	printf("top_weight [pix_id: %d]: %f\n", pix_id, top_weight);
		// 	printf("top_alpha [pix_id: %d]: %f\n", pix_id, top_alpha);
		// 	printf("second_alpha [pix_id: %d]: %f\n", pix_id, second_alpha);
		// 	printf("first3_weights [pix_id: %d]: %f, %f, %f\n", pix_id, first3_weights[0], first3_weights[1], first3_weights[2]);
		// }
		/*****************/
		
		if constexpr (UseColor)
		{
			for (int ch = 0; ch < CHANNELS; ch++)
				// out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch]; // 250418 bg코드 삭제
				out_color[ch * H * W + pix_id] = C[ch];
		}
		
		out_depth[pix_id] = D;
		out_silh[pix_id] = 1-T;
		if constexpr (UseFlow)
		{
			// GaussianFlow normalization: Flow = Σ(α·T)·f is the raw alpha-composited flow; divide by the
			// accumulated opacity A = (1−T) = out_silh to get the EXPECTED per-surface flow (unbiased where
			// coverage <1). fun_cost is computed on the NORMALIZED flow G too, so out_flowcost AND the in-kernel
			// backward cost gradient (computeFunCostCUDA reads gsflow=G) are consistent (forward==backward).
			float _A3 = 1.0f - T;
			float _iA3 = (_A3 > 1e-6f) ? (1.0f / _A3) : 0.0f;
			float Gx3 = Flow[0] * _iA3;
			float Gy3 = Flow[1] * _iA3;
			out_gsflow[0 * H * W + pix_id] = Gx3;
			out_gsflow[1 * H * W + pix_id] = Gy3;

			float dx1 = Gx3;                       // cost on the NORMALIZED flow (matches backward dx1=gsflow=G)
			float dx2 = flowimg[pix_id];
			float dy1 = Gy3;
			float dy2 = flowimg[H * W + pix_id];
			float cost = 0.f;
			float in_weight = flowconf[pix_id];
			float dummy_weight;
			// if (!updateconf)
			// 	fun_cost(dx1, dy1, dx2, dy2, in_weight, cost, dummy_weight, updateconf);
			// else
			fun_cost(dx1, dy1, dx2, dy2, in_weight, cost, flowconf[pix_id], updateconf);
			out_flowcost[pix_id] = cost;
		}
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block, 	// total blocks / threads per block
	const uint2* ranges,			// == imgState.ranges (closest / farthest) GS index of each tile
	const uint32_t* point_list,		// == binningState.point_list. All sorted GS index by tile->depth
	int W, int H,					
	const float2* means2D,			// == geomState.means2D
	const float2* means2D_next,		// == geomState.means2D_next
	const float* depths,		    // == geomState.depths
	const float* colors,			// == geomState.rgb
	const float4* conic_opacity,	// == geomState.conic_opacity
	const float4* covmap_next,	// == geomState.conic_opacity
	// const uint2* rect_next_min,
	// const uint2* rect_next_max,
	float* final_T,					// == imgState.accum_alpha (not calculated yet)
	uint32_t* n_contrib,			// == imgState.n_contrib   (not calculated yet)
	const float* bg_color,
	const float* flowimg,
	float* flowconf,
	const bool usecolor,
	const bool useflow,
	const bool trainpose,
	const bool updateconf,
	float* out_color,
	float* out_depth,
	float* out_silh,
	float* out_gsflow,
	float* out_flowcost,
	int* n_touched,
	int* n_found,
	float* weights_sum)
{
	// cudaError_t syncErr0f = cudaGetLastError();
	// cudaError_t asyncErr0f = cudaDeviceSynchronize();
	// if (syncErr0f != cudaSuccess) printf("Error0fs: %s\n", cudaGetErrorString(syncErr0f));
	// if (asyncErr0f != cudaSuccess) printf("Error0fa: %s\n", cudaGetErrorString(asyncErr0f));

	if (usecolor && useflow)
		renderCUDA<true, true, NUM_CHANNELS> << <grid, block >> > (
			ranges,
			point_list,
			W, H,
			means2D,
			means2D_next,
			depths,
			colors,
			conic_opacity,
			covmap_next,
			// rect_next_min,
			// rect_next_max,
			final_T,
			n_contrib,
			bg_color,
			flowimg,
			flowconf,
			// usecolor,
			// useflow,
			// trainpose,
			updateconf,
			out_color,
			out_depth,
			out_silh,
			out_gsflow,
			out_flowcost,
			n_touched,
			n_found,
			weights_sum);
	else if (usecolor && !useflow)
		renderCUDA<true, false, NUM_CHANNELS> << <grid, block >> > (
			ranges,
			point_list,
			W, H,
			means2D,
			means2D_next,
			depths,
			colors,
			conic_opacity,
			covmap_next,
			// rect_next_min,
			// rect_next_max,
			final_T,
			n_contrib,
			bg_color,
			flowimg,
			flowconf,
			// usecolor,
			// useflow,
			// trainpose,
			updateconf,
			out_color,
			out_depth,
			out_silh,
			out_gsflow,
			out_flowcost,
			n_touched,
			n_found,
			weights_sum);
	else if (!usecolor && useflow)
		renderCUDA<false, true, NUM_CHANNELS> << <grid, block >> > (
			ranges,
			point_list,
			W, H,
			means2D,
			means2D_next,
			depths,
			colors,
			conic_opacity,
			covmap_next,
			// rect_next_min,
			// rect_next_max,
			final_T,
			n_contrib,
			bg_color,
			flowimg,
			flowconf,
			// usecolor,
			// useflow,
			// trainpose,
			updateconf,
			out_color,
			out_depth,
			out_silh,
			out_gsflow,
			out_flowcost,
			n_touched,
			n_found,
			weights_sum);
	else
		renderCUDA<false, false, NUM_CHANNELS> << <grid, block >> > (
			ranges,
			point_list,
			W, H,
			means2D,
			means2D_next,
			depths,
			colors,
			conic_opacity,
			covmap_next,
			// rect_next_min,
			// rect_next_max,
			final_T,
			n_contrib,
			bg_color,
			flowimg,
			flowconf,
			// usecolor,
			// useflow,
			// trainpose,
			updateconf,
			out_color,
			out_depth,
			out_silh,
			out_gsflow,
			out_flowcost,
			n_touched,
			n_found,
			weights_sum);

	// cudaError_t syncErr1f = cudaGetLastError();
	// cudaError_t asyncErr1f = cudaDeviceSynchronize();
	// if (syncErr1f != cudaSuccess) printf("Error01s: %s\n", cudaGetErrorString(syncErr1f));
	// if (asyncErr1f != cudaSuccess) printf("Error01a: %s\n", cudaGetErrorString(asyncErr1f));
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* viewmatrix_next,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	// uint2* rect_next_min,
	// uint2* rect_next_max,
	float2* means2D,
	float2* means2D_next,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	float4* covmap_next,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool usecolor,
	bool useflow,
	bool prefiltered)
{
	// cudaError_t syncErr2f = cudaGetLastError();
	// cudaError_t asyncErr2f = cudaDeviceSynchronize();
	// if (syncErr2f != cudaSuccess) printf("Error2fs: %s\n", cudaGetErrorString(syncErr2f));
	// if (asyncErr2f != cudaSuccess) printf("Error2fa: %s\n", cudaGetErrorString(asyncErr2f));

	// Total blocks: (P+255) / 256
	// Threads per block: 256
	// NUM_CHANNELS: template T
	if (usecolor && useflow)
		preprocessCUDA<true, true, NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
			P, D, M,
			means3D,
			scales,
			scale_modifier,
			rotations,
			opacities,
			shs,
			clamped,
			cov3D_precomp,
			colors_precomp,
			viewmatrix,
			viewmatrix_next, 
			projmatrix,
			cam_pos,
			W, H,
			tan_fovx, tan_fovy,
			focal_x, focal_y,
			radii,
			// rect_next_min,
			// rect_next_max,
			means2D,
			means2D_next,
			depths,
			cov3Ds,
			rgb,
			conic_opacity,
			covmap_next,
			grid,
			tiles_touched,
			// usecolor,
			// useflow,
			prefiltered
			);
	else if (usecolor && !useflow)
		preprocessCUDA<true, false, NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
			P, D, M,
			means3D,
			scales,
			scale_modifier,
			rotations,
			opacities,
			shs,
			clamped,
			cov3D_precomp,
			colors_precomp,
			viewmatrix,
			viewmatrix_next, 
			projmatrix,
			cam_pos,
			W, H,
			tan_fovx, tan_fovy,
			focal_x, focal_y,
			radii,
			// rect_next_min,
			// rect_next_max,
			means2D,
			means2D_next,
			depths,
			cov3Ds,
			rgb,
			conic_opacity,
			covmap_next,
			grid,
			tiles_touched,
			// usecolor,
			// useflow,
			prefiltered
			);
	else if (!usecolor && useflow)
		preprocessCUDA<false, true, NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
			P, D, M,
			means3D,
			scales,
			scale_modifier,
			rotations,
			opacities,
			shs,
			clamped,
			cov3D_precomp,
			colors_precomp,
			viewmatrix,
			viewmatrix_next, 
			projmatrix,
			cam_pos,
			W, H,
			tan_fovx, tan_fovy,
			focal_x, focal_y,
			radii,
			// rect_next_min,
			// rect_next_max,
			means2D,
			means2D_next,
			depths,
			cov3Ds,
			rgb,
			conic_opacity,
			covmap_next,
			grid,
			tiles_touched,
			// usecolor,
			// useflow,
			prefiltered
			);
	else
		preprocessCUDA<false, false, NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
			P, D, M,
			means3D,
			scales,
			scale_modifier,
			rotations,
			opacities,
			shs,
			clamped,
			cov3D_precomp,
			colors_precomp,
			viewmatrix,
			viewmatrix_next, 
			projmatrix,
			cam_pos,
			W, H,
			tan_fovx, tan_fovy,
			focal_x, focal_y,
			radii,
			// rect_next_min,
			// rect_next_max,
			means2D,
			means2D_next,
			depths,
			cov3Ds,
			rgb,
			conic_opacity,
			covmap_next,
			grid,
			tiles_touched,
			// usecolor,
			// useflow,
			prefiltered
			);

	// cudaError_t syncErr3f = cudaGetLastError();
	// cudaError_t asyncErr3f = cudaDeviceSynchronize();
	// if (syncErr3f != cudaSuccess) printf("Error0fs: %s\n", cudaGetErrorString(syncErr3f));
	// if (asyncErr3f != cudaSuccess) printf("Error0fa: %s\n", cudaGetErrorString(asyncErr3f));
}