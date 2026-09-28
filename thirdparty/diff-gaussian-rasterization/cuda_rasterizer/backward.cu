/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and e	valuation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


#define ERROR_SUM_THRES 0.2f 
#define ERROR_NORM_THRES 0.0005f
#define STABLE_COUNT_THRES 10
#define UNSTABLE_COUNT_THRES 0
#define COUNT_MIN -40
#define COUNT_MAX 20
#define UNSTABLE_OPA_THRES 0.1f
#define SKIP_BLOCK_SUM_THRES 0.01f * BLOCK_X * BLOCK_Y // * 3 // (3 channel)
#define SKIP_BLOCK_SIGMA_THRES 0.01f

#define CHECK_PTR(name) if ((name) == nullptr) { \
    printf("X " #name " is NULL at idx %d\n", idx); return; }


/* For training pose */
__forceinline__ __device__ glm::vec3 cross(glm::vec3 a, glm::vec3 b) 
{
	// 실제수식: (a2b3 - a3b2, a3b1-a1b3, a1b2 - a2b1)
	// 코드: 	(-a2b3 + a3b2, a1b3 - a3b1, -a1b2 + a2b1)
	// 즉: 실제 cross product에 음수가 붙은 형태임.
	return glm::vec3{
		-a.y * b.z + a.z * b.y,
		a.x * b.z - a.z * b.x,
		-a.x * b.y + a.y * b.x
	};
}

__forceinline__ __device__ glm::mat3 skew_x(float3 v)
{
	return glm::mat3{
		0, v.z, -v.y, // column 1
		-v.z, 0, v.x, // column 2
		v.y, -v.x, 0  // column 3
	};
}

template <typename T>
__device__ void inline reduce_helper(int lane, int i, T *data) {
  if (lane < i) {
    data[lane] += data[lane + i];
  }
}

template <typename group_t, typename... Lists>
__device__ void render_cuda_reduce_sum(group_t g, Lists... lists) {
  int lane = g.thread_rank();
  const int blockSize = g.size();
  g.sync();

  for (int i = g.size() / 2; i > 0; i /= 2) {
    (...,
     reduce_helper(
         lane, i, lists)); // Fold expression: apply reduce_helper for each list
    g.sync();
  }
	// for (int stride = blockSize >> 1; stride > 0; stride >>= 1) {
	// 	if (lane < stride) {
	// 		(void)std::initializer_list<int>{
	// 			(reduce_helper(lane, stride, lists), 0)...
	// 		};
	// 	}
	// 	g.sync();
	// }
}


// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
// deg == D, max_coeffs == M, 
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	// dir은 camera에서 GS를 보는 방향 (SH의 특성)
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos; // vector campos -> GSmean
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	// forward식에서 단순히 미분하면 가능함.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	// RGB는 서로 independent하므로 요소별로 곱하되, 한번에 계산 가능. == 즉 dot product
	// dL_dx,y,z = dL_dRGB * dRGB * dx,y,z
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	// dv_norm = dvnorm_dv * dv
	// dvnorm_dv를 jacobian으로 계산하여 구한 후, dv_norm을 구함.
	// forward에서 sh 계산할때 normalized로 구했으므로, 마찬가지로 normalize상태에 변환에 대해서도 계산하는 것.
	// dL_dmean = dL_ddir * ddir_dmean인데, dL_ddir은 알고있고, ddir_dmean이 바로 여기서 계산하는 jacobian이 됨.
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	return glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

__device__ void computeSqrtCov2DInv(glm::mat3& cov2D, glm::mat3& cov2D_next, glm::mat2& dL_dcovmap,
		float3& dL_dcov2D, float3& dL_dcov2D_next)
{
	float3 cov = {cov2D[0][0], cov2D[0][1], cov2D[1][1]};
	float3 cov_next = {cov2D_next[0][0], cov2D_next[0][1], cov2D_next[1][1]};
	glm::mat2 sqrtcov2D, sqrtcov2Dinv;
	glm::mat2 sqrtcov2D_next, sqrtcov2Dinv_next;
	glm::mat2 covmap;

	float delta = sqrt(4*cov.y*cov.y + (cov.x - cov.z)*(cov.x - cov.z));
	float lambda1 = (cov.x + cov.z - delta) / 2.0f;
	float lambda2 = (cov.x + cov.z + delta) / 2.0f;
	lambda1 = fmaxf(lambda1, ZDE);
	lambda2 = fmaxf(lambda2, ZDE);

	float delta_next = sqrt(4*cov_next.y*cov_next.y + (cov_next.x - cov_next.z)*(cov_next.x - cov_next.z));
	float lambda1_next = (cov_next.x + cov_next.z - delta_next) / 2.0f;
	float lambda2_next = (cov_next.x + cov_next.z + delta_next) / 2.0f;
	lambda1_next = fmaxf(lambda1_next, ZDE);
	lambda2_next = fmaxf(lambda2_next, ZDE);

	bool covy_valid = (abs(cov.y) > ZDE);
	bool covnexty_valid = (abs(cov_next.y) > ZDE);

	float v11 = covy_valid ? (lambda1 - cov.z) / (cov.y) : 1.0f;
	float v21 = covy_valid ? (lambda2 - cov.z) / (cov.y) : 0.0f;
	float v11_next = covnexty_valid ? (lambda1_next - cov_next.z) / (cov_next.y) : 1.0f;
	float v21_next = covnexty_valid ? (lambda2_next - cov_next.z) / (cov_next.y) : 0.0f;

	float v1_norm = sqrt(1 + v11 * v11);
	float v2_norm = sqrt(1 + v21 * v21);
	float2 v1 = { v11 / v1_norm, 1.0f / v1_norm };
	float2 v2 = { v21 / v2_norm, 1.0f / v2_norm };

	float v1_norm_next = sqrt(1 + v11_next * v11_next);
	float v2_norm_next = sqrt(1 + v21_next * v21_next);
	float2 v1_next = { v11_next / v1_norm_next, 1.0f / v1_norm_next };
	float2 v2_next = { v21_next / v2_norm_next, 1.0f / v2_norm_next };

	// Calculate eigen decomposition for sqrt / sqrtinv of cov2Ds
	if (lambda1 > ZDE && lambda2 > ZDE && lambda1_next > ZDE && lambda2_next > ZDE)
	{
		glm::mat2 Q = glm::mat2(v1.x, v1.y, v2.x, v2.y);
		// glm::mat2 S_sqrt = glm::mat2(sqrt(lambda1), 0, 0, sqrt(lambda2));
		// glm::mat2 S_sqrt_inv = glm::mat2(1.0f / sqrt(lambda1), 0, 0, 1.0f / sqrt(lambda2));
		glm::mat2 S_sqrt = glm::mat2(sqrt(lambda1), 0.0f,
                             0.0f, sqrt(lambda2));
		glm::mat2 S_sqrt_inv = glm::mat2(1.0f / sqrt(lambda1), 0.0f,
										0.0f, 1.0f / sqrt(lambda2));
		sqrtcov2D = Q * S_sqrt * glm::transpose(Q);
		sqrtcov2Dinv = Q * S_sqrt_inv * glm::transpose(Q);

		glm::mat2 Q_next = glm::mat2(v1_next.x, v1_next.y, v2_next.x, v2_next.y);
		// glm::mat2 S_sqrt_next = glm::mat2(sqrt(lambda1_next), 0, 0, sqrt(lambda2_next));
		// glm::mat2 S_sqrt_inv_next = glm::mat2((1.0f / sqrt(lambda1_next)), 0, 0, 1.0f / (sqrt(lambda2_next)));
		glm::mat2 S_sqrt_next = glm::mat2(sqrt(lambda1_next), 0.0f,
                                  0.0f, sqrt(lambda2_next));
		glm::mat2 S_sqrt_inv_next = glm::mat2(1.0f / sqrt(lambda1_next), 0.0f,
											0.0f, 1.0f / sqrt(lambda2_next));

		sqrtcov2D_next = Q_next * S_sqrt_next * glm::transpose(Q_next);
		// sqrtcov2Dinv_next = Q_next * S_sqrt_inv_next * glm::transpose(Q_next);
		
		// Using Daleckii–Krein formula
		if (abs(cov.y) > ZDE)
		{
			glm::mat2 dL_dsqrtcov2Dinv = sqrtcov2D_next * dL_dcovmap;
			// float l1 = lambda1;
			// float l2 = lambda2;
		
			// lambda1, lambda2는 위에서 이미 fmax(lambda, ZDE)로 clamp 되어 있다고 가정
			// f(t) = t^{-1/2}
			float f1_prime = -0.5f / (lambda1 * sqrtf(lambda1)); // f'(λ1) = -1/(2 λ1^{3/2})
			float f2_prime = -0.5f / (lambda2 * sqrtf(lambda2)); // f'(λ2)
		
			float f_l1 = 1.0f / sqrtf(lambda1); // f(λ1)
			float f_l2 = 1.0f / sqrtf(lambda2); // f(λ2)
		
			float denom = lambda1 - lambda2;
		
			// Loewner off-diagonal: (f(λ1) - f(λ2)) / (λ1 - λ2)
			float off;
			if (fabsf(denom) > ZDE) {
				off = (f_l1 - f_l2) / denom;
			} else {
				// λ1 ≈ λ2 일 때는 극한값 f'(λ1)을 쓰는게 자연스러움
				off = f1_prime;
			}
		
			// F = [ f'(λ1)   off
			//       off      f'(λ2) ]
			glm::mat2 F;
			F[0][0] = f1_prime;
			F[1][1] = f2_prime;
			F[0][1] = off;
			F[1][0] = off;
		
			// Geig = Q^T * (dL/dX) * Q
			glm::mat2 Geig = glm::transpose(Q) * dL_dsqrtcov2Dinv * Q;
		
			// H = F ⊙ Geig (elementwise product)
			glm::mat2 H;
			H[0][0] = F[0][0] * Geig[0][0];
			H[0][1] = F[0][1] * Geig[0][1];
			H[1][0] = F[1][0] * Geig[1][0];
			H[1][1] = F[1][1] * Geig[1][1];
		
			// dL/dB = Q * H * Q^T  (B = cov2D)
			glm::mat2 dL_dcov2D_mat = Q * H * glm::transpose(Q);
		
			// cov2D는 [ [x, y],
			//           [y, z] ] 구조라고 했으니, tri 형태로 저장
			dL_dcov2D.x = dL_dcov2D_mat[0][0]; // dL/dx
			dL_dcov2D.y = dL_dcov2D_mat[0][1] + dL_dcov2D_mat[1][0]; // dL/dy  (== [1][0])
			dL_dcov2D.z = dL_dcov2D_mat[1][1]; // dL/dz
		}

		if (abs(cov_next.y) > ZDE)
		{
			glm::mat2 dL_sqrtcov2D_next = dL_dcovmap * glm::transpose(sqrtcov2Dinv);
			// float l1n = lambda1_next;
			// float l2n = lambda2_next;

			// g(t) = t^{1/2}
			float g1_prime = 0.5f / sqrtf(lambda1_next); // g'(λ1) = 1/(2√λ1)
			float g2_prime = 0.5f / sqrtf(lambda2_next); // g'(λ2)

			float g_l1 = sqrtf(lambda1_next); // g(λ1)
			float g_l2 = sqrtf(lambda2_next); // g(λ2)

			float denom_n = lambda1_next - lambda2_next;

			// Loewner off-diagonal: (g(λ1) - g(λ2)) / (λ1 - λ2)
			float off_n;
			if (fabsf(denom_n) > ZDE) {
				off_n = (g_l1 - g_l2) / denom_n;
			} else {
				// λ1 ≈ λ2 인 경우 극한값 g'(λ1)
				off_n = g1_prime;
			}

			glm::mat2 F_n;
			F_n[0][0] = g1_prime;
			F_n[1][1] = g2_prime;
			F_n[0][1] = off_n;
			F_n[1][0] = off_n;

			glm::mat2 Geig_n = glm::transpose(Q_next) * dL_sqrtcov2D_next * Q_next;

			glm::mat2 H_n;
			H_n[0][0] = F_n[0][0] * Geig_n[0][0];
			H_n[0][1] = F_n[0][1] * Geig_n[0][1];
			H_n[1][0] = F_n[1][0] * Geig_n[1][0];
			H_n[1][1] = F_n[1][1] * Geig_n[1][1];

			glm::mat2 dL_dcov2D_next_mat = Q_next * H_n * glm::transpose(Q_next);

			dL_dcov2D_next.x = dL_dcov2D_next_mat[0][0];
			dL_dcov2D_next.y = dL_dcov2D_next_mat[0][1] + dL_dcov2D_next_mat[1][0];
			dL_dcov2D_next.z = dL_dcov2D_next_mat[1][1];
		}

		// Using scalar chain rule
		// covmap = sqrtcov2D_next * sqrtcov2Dinv
		/************************ dL_dcov2D 계산 ************************/
		// dL_dsqrtcov2Dinv, dL_dQ
		/* @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ */
		// if (abs(cov.y) > ZDE) //  && abs(delta) > ZDE
		// {
		// 	// float dotforgrad_x = glm::dot(dL_dcovmap[0], sqrtcov2D_next[0]);
		// 	// float dotforgrad_y = glm::dot(dL_dcovmap[1], sqrtcov2D_next[0]);
		// 	// float dotforgrad_z = glm::dot(dL_dcovmap[0], sqrtcov2D_next[1]);
		// 	// float dotforgrad_w = glm::dot(dL_dcovmap[1], sqrtcov2D_next[1]);
		// 	// glm::mat2 dL_dsqrtcov2Dinv = glm::mat2(dotforgrad_x, dotforgrad_z, dotforgrad_y, dotforgrad_w);
		// 	glm::mat2 dL_dsqrtcov2Dinv = glm::transpose(sqrtcov2D_next) * dL_dcovmap;

		// 	// sqrtcov2Dinv = Q * S_sqrt_inv * Q^T
		// 	glm::mat2 dL_dQ = (dL_dsqrtcov2Dinv + glm::transpose(dL_dsqrtcov2Dinv)) * Q * S_sqrt_inv;

		// 	// dL_dsqrtinv, dL_dv11, dL_dv21
		// 	glm::mat2 dL_dSsqrtinv = glm::transpose(Q) * dL_dsqrtcov2Dinv * Q;
		// 	// dL_dSsqrtinv[0][1] = 0.0f;
		// 	// dL_dSsqrtinv[1][0] = 0.0f;
		// 	float2 dL_dv1 = {dL_dQ[0][0], dL_dQ[0][1]};
		// 	float2 dL_dv2 = {dL_dQ[1][0], dL_dQ[1][1]};
		// 	float dL_dv11 = dL_dv1.x * (1.0f / pow3(v1_norm)) + dL_dv1.y * (-v11 / pow3(v1_norm));
		// 	float dL_dv21 = dL_dv2.x * (1.0f / pow3(v2_norm)) + dL_dv2.y * (-v21 / pow3(v2_norm));
	
		// // dL_dlambda1, dL_dlambda2, dL_dcov2D
		// // if (abs(cov.y) > ZDE && abs(delta) > ZDE)
		// // if (abs(cov.y) > ZDE)
		// // {
		// 	float dL_dlambda1 = dL_dv11 / cov.y + dL_dSsqrtinv[0][0] * (-1.0f / (2 * pow1p5(lambda1)) );
		// 	float dL_dlambda2 = dL_dv21 / cov.y + dL_dSsqrtinv[1][1] * (-1.0f / (2 * pow1p5(lambda2)) );
		// 	dL_dcov2D.x = dL_dlambda1 * (1.f / 2.f) 
		// 		+ dL_dlambda1 * (-1.f / 2.f) * (cov.x - cov.z) / delta
		// 		+ dL_dlambda2 * (1.f / 2.f)
		// 		+ dL_dlambda2 * (1.f / 2.f) * (cov.x - cov.z) / delta;
		// 	dL_dcov2D.y = dL_dlambda1 * (-1.f / 2.f) * 4.f * cov.y / delta
		// 		+ dL_dlambda2 * (1.f / 2.f) * 4.f * cov.y / delta
		// 		+ dL_dv11 * (cov.z - lambda1) / (cov.y * cov.y)
		// 		+ dL_dv21 * (cov.z - lambda2) / (cov.y * cov.y);
		// 	dL_dcov2D.z = dL_dlambda1 * (1.f / 2.f) 
		// 		+ dL_dlambda1 * (1.f / 2.f) * (cov.x - cov.z) / delta
		// 		+ dL_dlambda2 * (1.f / 2.f)
		// 		+ dL_dlambda2 * (-1.f / 2.f) * (cov.x - cov.z) / delta
		// 		+ dL_dv11 * (-1.f / cov.y)
		// 		+ dL_dv21 * (-1.f / cov.y);	
		// }
	
		// /****************************************************************/

		// /************************ dL_dcov2D_next 계산 ************************/
		// // dL_dsqrtcov2D_next, dL_dQ_next
		// if (abs(cov_next.y) > ZDE) //  && abs(delta_next) > ZDE
		// {
		// 	glm::mat2 dL_sqrtcov2D_next = dL_dcovmap * glm::transpose(sqrtcov2Dinv);
		// 	// sqrtcov2D_next = Q_next * S_sqrt_next * Q_next^T
		// 	glm::mat2 dL_dQ_next = (dL_sqrtcov2D_next + glm::transpose(dL_sqrtcov2D_next)) * Q_next * S_sqrt_next;

		// 	// dL_dSsqrt_next, dL_dv11_next, dL_dv21_next
		// 	glm::mat2 dL_dSsqrt_next = glm::transpose(Q_next) * dL_sqrtcov2D_next * Q_next;
		// 	// dL_dSsqrt_next[0][1] = 0.0f;
		// 	// dL_dSsqrt_next[1][0] = 0.0f;
		// 	float2 dL_dv1_next = {dL_dQ_next[0][0], dL_dQ_next[0][1]};
		// 	float2 dL_dv2_next = {dL_dQ_next[1][0], dL_dQ_next[1][1]};
		// 	float dL_dv11_next = dL_dv1_next.x * (1.0f / pow3(v1_norm_next)) + dL_dv1_next.y * (-v11_next / pow3(v1_norm_next));
		// 	float dL_dv21_next = dL_dv2_next.x * (1.0f / pow3(v2_norm_next)) + dL_dv2_next.y * (-v21_next / pow3(v2_norm_next));

		// // dL_dlambda1_next, dL_dlambda2_next, dL_dcov2D_next
		// // if (abs(cov_next.y) > ZDE && abs(delta_next) > ZDE)
		// // if (abs(cov_next.y) > ZDE)
		// // {
		// 	float dL_dlambda1_next = dL_dv11_next / cov_next.y + dL_dSsqrt_next[0][0] * (1.f / (2.f * sqrt(lambda1_next) + ZDE));
		// 	float dL_dlambda2_next = dL_dv21_next / cov_next.y + dL_dSsqrt_next[1][1] * (1.f / (2.f * sqrt(lambda2_next) + ZDE));
		// 	dL_dcov2D_next.x = dL_dlambda1_next * (1.f / 2.f) 
		// 		+ dL_dlambda1_next * (-1.f / 2.f) * (cov_next.x - cov_next.z) / delta_next
		// 		+ dL_dlambda2_next * (1.f / 2.f)
		// 		+ dL_dlambda2_next * (1.f / 2.f) * (cov_next.x - cov_next.z) / delta_next;
		// 	dL_dcov2D_next.y = dL_dlambda1_next * (-1.f / 2.f) * 4.f * cov_next.y / delta_next
		// 		+ dL_dlambda2_next * (1.f / 2.f) * 4.f * cov_next.y / delta_next
		// 		+ dL_dv11_next * (cov_next.z - lambda1_next) / (cov_next.y * cov_next.y)
		// 		+ dL_dv21_next * (cov_next.z - lambda2_next) / (cov_next.y * cov_next.y);
		// 	dL_dcov2D_next.z = dL_dlambda1_next * (1.f / 2.f) 
		// 		+ dL_dlambda1_next * (1.f / 2.f) * (cov_next.x - cov_next.z) / delta_next
		// 		+ dL_dlambda2_next * (1.f / 2.f)
		// 		+ dL_dlambda2_next * (-1.f / 2.f) * (cov_next.x - cov_next.z) / delta_next
		// 		+ dL_dv11_next * (-1.f / cov_next.y)
		// 		+ dL_dv21_next * (-1.f / cov_next.y);
		// }
		/* @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ */
	}
	/****************************************************************/
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
template <bool UseFlow, bool TrainPose>
__global__ void computeCov2DCUDA(int P,
	const float3* means,	// == means3D
	// const bool* stable_status,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,	// == focal_x, focal_y
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* view_matrix_next,
	const float* relpose,
	const float* dL_dconics,	// (input)
	const float* dL_dcovmap_next,	// (input)
	float3* dL_dmeans,			// (output)
	float* dL_dcov,				// (output)
	float* dL_dtau,				// (output)
	float* dL_dtau_next)				// (output)
	// const bool usecolor,
	// const bool useflow,
	// const bool trainpose)				
{
	// auto idx = cg::this_grid().thread_rank();	// "global" thread idx in the grid
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= P || !(radii[idx] > 0))
		return;
	
	// Stable한 GS이면서 trainpose 하지 않는 경우 계산하지 않는다．
	// if (stable_status[idx] && !trainpose)
	// 	return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.

	// forward에서 computeCov2D와 같은 계산 진행
	float3 mean = means[idx];
	// dL_dconic.x, dL_dconic.y, dL_dconic.w
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix); // t = Tcw * Pw = Pc (camera coorindate에서의 GS의 3D 좌표)
	float3 t_next = transformPoint4x3(mean, view_matrix_next);

	t.z = max(t.z, ZDE);
	t_next.z = max(t_next.z, ZDE);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;	// 범위 제한
	t.y = min(limy, max(-limy, tytz)) * t.z;	// 범위 제한

	const float txtz_next = t_next.x / t_next.z;
	const float tytz_next = t_next.y / t_next.z;
	if constexpr (UseFlow)
	{
		t_next.x = min(limx, max(-limx, txtz_next)) * t_next.z;	// 범위 제한
		t_next.y = min(limy, max(-limy, tytz_next)) * t_next.z;	// 범위 제한
	}
	
	// 범위 벗어나면 x_grad_mul를 0으로 설정
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;
	const float x_grad_mul_next = txtz_next < -limx || txtz_next > limx ? 0 : 1;
	const float y_grad_mul_next = tytz_next < -limy || tytz_next > limy ? 0 : 1;

	// J, W, Vrk, T, cov2D까지 forward에서와 동일하게 계산함.
	/////////////////////////////////////////////////////////////////////
	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;
	/////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////
	glm::mat3 J_next(0.0f), W_next(0.0f), T_next(0.0f), cov2D_next(0.0f);

	if constexpr (UseFlow)
	{
		J_next = glm::mat3(h_x / t_next.z, 0.0f, -(h_x * t_next.x) / (t_next.z * t_next.z),
		0.0f, h_y / t_next.z, -(h_y * t_next.y) / (t_next.z * t_next.z),
		0, 0, 0);

		W_next = glm::mat3(
			view_matrix_next[0], view_matrix_next[4], view_matrix_next[8],
			view_matrix_next[1], view_matrix_next[5], view_matrix_next[9],
			view_matrix_next[2], view_matrix_next[6], view_matrix_next[10]);

		T_next = W_next * J_next;
		cov2D_next = glm::transpose(T_next) * glm::transpose(Vrk) * T_next;
	}
	/////////////////////////////////////////////////////////////////////

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	float a_next = cov2D_next[0][0] += 0.3f;
	float b_next = cov2D_next[0][1];
	float c_next = cov2D_next[1][1] += 0.3f;
	float dL_da_next = 0, dL_db_next = 0, dL_dc_next = 0;

	float denom_next = a_next * c_next - b_next * b_next;
	float denom2inv_next = 1.0f / ((denom_next * denom_next) + 0.0000001f);
	
	float3 dL_dcov2Dflow = {0.f}; 
	float3 dL_dcov2Dflow_next = {0.f}; 
	if constexpr (UseFlow)
	{
		glm::mat2 dL_dcovmap = glm::mat2(dL_dcovmap_next[4*idx], dL_dcovmap_next[4*idx + 2], 
			dL_dcovmap_next[4*idx + 1], dL_dcovmap_next[4*idx + 3]);
		computeSqrtCov2DInv(cov2D, cov2D_next, dL_dcovmap, dL_dcov2Dflow, dL_dcov2Dflow_next);
	}
	for (int i = 0; i < 6; i++)
		dL_dcov[6 * idx + i] = 0;

	if (denom2inv != 0 && denom != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		// 다음 chain rule에 의해 계산..! (https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)
		// dL/da = \Sigma_{k=1}\Sigma_{l=1} dL/dconic_{kl} * dconic_{kl}/da
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);
	}
	if constexpr (UseFlow)
	{
		if (denom2inv_next!=0 && denom_next!=0)
		{
			dL_da += dL_dcov2Dflow.x;
			dL_db += dL_dcov2Dflow.y;
			dL_dc += dL_dcov2Dflow.z;

			dL_da_next = dL_dcov2Dflow_next.x;
			dL_db_next = dL_dcov2Dflow_next.y;
			dL_dc_next = dL_dcov2Dflow_next.z;
		}
	}
	// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
	// given gradients w.r.t. 2D covariance matrix (diagonal).
	// cov2D = transpose(T) * transpose(Vrk) * T;
	// cov2D = T^T * cov3D * T를 전개한다음에 
	// dL_dCov3D_00 = \Sigma_{k=1} \Sigma_{l=1} dL_dCov2Dkl * dCov2Dkl_dCov3D_00 와 같이 계산하면 됨.
	// dCov2Dkl_dCov3D_00은 cov2D = T^Tcov3DT를 다 전개한 후 구하면 됨.
	// 주의..!: glm::Mat은 column-major order이므로, T[1][0] == T_{01}임. 즉 T[column idx][row idx]임
	dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
	dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
	dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

	// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
	// given gradients w.r.t. 2D covariance matrix (off-diagonal).
	// Off-diagonal elements appear twice --> double the gradient.
	// cov2D = transpose(T) * transpose(Vrk) * T;
	// back-diangonal에 대해서는 2배가 되는것을 확인할 수 있음..!!
	dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
	dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
	dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;

	if constexpr (UseFlow)
	{
		dL_dcov[6 * idx + 0] += T_next[0][0] * T_next[0][0] * dL_da_next + T_next[0][0] * T_next[1][0] * dL_db_next + T_next[1][0] * T_next[1][0] * dL_dc_next;
		dL_dcov[6 * idx + 3] += T_next[0][1] * T_next[0][1] * dL_da_next + T_next[0][1] * T_next[1][1] * dL_db_next + T_next[1][1] * T_next[1][1] * dL_dc_next;
		dL_dcov[6 * idx + 5] += T_next[0][2] * T_next[0][2] * dL_da_next + T_next[0][2] * T_next[1][2] * dL_db_next + T_next[1][2] * T_next[1][2] * dL_dc_next;
		
		dL_dcov[6 * idx + 1] += 2 * T_next[0][0] * T_next[0][1] * dL_da_next + (T_next[0][0] * T_next[1][1] + T_next[0][1] * T_next[1][0]) * dL_db_next + 2 * T_next[1][0] * T_next[1][1] * dL_dc_next;
		dL_dcov[6 * idx + 2] += 2 * T_next[0][0] * T_next[0][2] * dL_da_next + (T_next[0][0] * T_next[1][2] + T_next[0][2] * T_next[1][0]) * dL_db_next + 2 * T_next[1][0] * T_next[1][2] * dL_dc_next;
		dL_dcov[6 * idx + 4] += 2 * T_next[0][2] * T_next[0][1] * dL_da_next + (T_next[0][1] * T_next[1][2] + T_next[0][2] * T_next[1][1]) * dL_db_next + 2 * T_next[1][1] * T_next[1][2] * dL_dc_next;
	}
	// else
	// {
	// 	for (int i = 0; i < 6; i++)
	// 		dL_dcov[6 * idx + i] = 0;
	// }

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	// dcov2D_dT = 2(Vrk)T

	// == dL_dT[0,0], dL_dT[1,0], dL_dT[2,0] ...
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix (column-major이므로 3x2이고 실제로는 2x3)
	// T = W * J
	// dL_dJ = dL_dT * dT_dJ
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	// float tz = 1.f / t.z;
	float tz = 1.0f / max(t.z, ZDE);
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	// mean3D in image_coordinate = (fx * t0/t2, fx * t1/t2, ||t||)
	// dL_dtx = \Sigma_{k=1} \Sigma_{l=1} dL_dJkl * dJkl_dtx 같은 방식으로 2x3부분에 대해서만 계산하면 됨!
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	if constexpr (TrainPose)
	{
		glm::vec3 dL_drho = {dL_dtx, dL_dty, dL_dtz};
		glm::vec3 dL_dtheta = cross({dL_dtx, dL_dty, dL_dtz}, {t.x, t.y, t.z});

		glm::mat3 dL_dW = {
			dL_dT00 * J[0][0], dL_dT01 * J[0][0], dL_dT02 * J[0][0],
			dL_dT10 * J[1][1], dL_dT11 * J[1][1], dL_dT12 * J[1][1],
			dL_dT00 * J[0][2] + dL_dT10 * J[1][2], dL_dT01 * J[0][2] + dL_dT11 * J[1][2], dL_dT02 * J[0][2] + dL_dT12 * J[1][2]
		};

		dL_dtheta.x += glm::dot(dL_dW[1], -W[2]) + glm::dot(dL_dW[2], W[1]);
		dL_dtheta.y += glm::dot(dL_dW[0], W[2]) + glm::dot(dL_dW[2], -W[0]);
		dL_dtheta.z += glm::dot(dL_dW[0], -W[1]) + glm::dot(dL_dW[1], W[0]);

		dL_dtau[6 * idx + 0] = dL_drho.x;
		dL_dtau[6 * idx + 1] = dL_drho.y;
		dL_dtau[6 * idx + 2] = dL_drho.z;
		dL_dtau[6 * idx + 3] = dL_dtheta.x;
		dL_dtau[6 * idx + 4] = dL_dtheta.y;
		dL_dtau[6 * idx + 5] = dL_dtheta.z;

		/// TODO: curr/next 따로 dL_dtau 분리하고 usecolor 없애야함
		// if (usecolor)
		// {
		// 	dL_dtau[6 * idx + 0] = dL_drho.x;
		// 	dL_dtau[6 * idx + 1] = dL_drho.y;
		// 	dL_dtau[6 * idx + 2] = dL_drho.z;
		// 	dL_dtau[6 * idx + 3] = dL_dtheta.x;
		// 	dL_dtau[6 * idx + 4] = dL_dtheta.y;
		// 	dL_dtau[6 * idx + 5] = dL_dtheta.z;
		// }
		// else /// TODO: 이거는 없어져야함. (잘못된 코드)
		// {
		// 	/// 여기서 next_pose기준으로 변환하고 더해줘야함.
		// 	// == (R^c_cnext)^T * dL_drho
		// 	// float3 dL_drho_next = transformVec4x3Transpose({dL_drho.x, dL_drho.y, dL_drho.z}, relpose);
		// 	// float3 t_rel = {relpose[12], relpose[13], relpose[14]};
		// 	// glm::mat3 t_x = skew_x(t_rel);
		// 	// glm::vec3 dL_drho_tx = glm::transpose(t_x) * dL_drho; // == dL_drho * t_x
		// 	// float3 dL_dtheta_next = transformVec4x3Transpose({dL_drho_tx.x, dL_drho_tx.y, dL_drho_tx.z}, relpose)
		// 	// 	+ transformVec4x3Transpose({dL_dtheta.x, dL_dtheta.y, dL_dtheta.z}, relpose);

		// 	// dL_dtau[6 * idx + 0] = dL_drho_next.x;
		// 	// dL_dtau[6 * idx + 1] = dL_drho_next.y;
		// 	// dL_dtau[6 * idx + 2] = dL_drho_next.z;
		// 	// dL_dtau[6 * idx + 3] = dL_dtheta_next.x;
		// 	// dL_dtau[6 * idx + 4] = dL_dtheta_next.y;
		// 	// dL_dtau[6 * idx + 5] = dL_dtheta_next.z;
		// }
	}

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	// Rwc * {dL_dt in camera coordinate} => tW
	// 해당 함수에서 rotation부분만 곱하고 있고, transpose해서 곱하므로 Rwc를 곱하는것..!!!
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;

	if constexpr (UseFlow)
	{
		float dL_dT00_next = 2 * (T_next[0][0] * Vrk[0][0] + T_next[0][1] * Vrk[0][1] + T_next[0][2] * Vrk[0][2]) * dL_da_next +
			(T_next[1][0] * Vrk[0][0] + T_next[1][1] * Vrk[0][1] + T_next[1][2] * Vrk[0][2]) * dL_db_next;
		float dL_dT01_next = 2 * (T_next[0][0] * Vrk[1][0] + T_next[0][1] * Vrk[1][1] + T_next[0][2] * Vrk[1][2]) * dL_da_next +
			(T_next[1][0] * Vrk[1][0] + T_next[1][1] * Vrk[1][1] + T_next[1][2] * Vrk[1][2]) * dL_db_next;
		float dL_dT02_next = 2 * (T_next[0][0] * Vrk[2][0] + T_next[0][1] * Vrk[2][1] + T_next[0][2] * Vrk[2][2]) * dL_da_next +
			(T_next[1][0] * Vrk[2][0] + T_next[1][1] * Vrk[2][1] + T_next[1][2] * Vrk[2][2]) * dL_db_next;
		float dL_dT10_next = 2 * (T_next[1][0] * Vrk[0][0] + T_next[1][1] * Vrk[0][1] + T_next[1][2] * Vrk[0][2]) * dL_dc_next +
			(T_next[0][0] * Vrk[0][0] + T_next[0][1] * Vrk[0][1] + T_next[0][2] * Vrk[0][2]) * dL_db_next;
		float dL_dT11_next = 2 * (T_next[1][0] * Vrk[1][0] + T_next[1][1] * Vrk[1][1] + T_next[1][2] * Vrk[1][2]) * dL_dc_next +
			(T_next[0][0] * Vrk[1][0] + T_next[0][1] * Vrk[1][1] + T_next[0][2] * Vrk[1][2]) * dL_db_next;
		float dL_dT12_next = 2 * (T_next[1][0] * Vrk[2][0] + T_next[1][1] * Vrk[2][1] + T_next[1][2] * Vrk[2][2]) * dL_dc_next +
			(T_next[0][0] * Vrk[2][0] + T_next[0][1] * Vrk[2][1] + T_next[0][2] * Vrk[2][2]) * dL_db_next;

		float dL_dJ00_next = W_next[0][0] * dL_dT00_next + W_next[0][1] * dL_dT01_next + W_next[0][2] * dL_dT02_next;
		float dL_dJ02_next = W_next[2][0] * dL_dT00_next + W_next[2][1] * dL_dT01_next + W_next[2][2] * dL_dT02_next;
		float dL_dJ11_next = W_next[1][0] * dL_dT10_next + W_next[1][1] * dL_dT11_next + W_next[1][2] * dL_dT12_next;
		float dL_dJ12_next = W_next[2][0] * dL_dT10_next + W_next[2][1] * dL_dT11_next + W_next[2][2] * dL_dT12_next;

		// float tz_next = 1.f / t_next.z;
		float tz_next = 1.0f / max(t_next.z, ZDE);
		float tz2_next = tz_next * tz_next;
		float tz3_next = tz2_next * tz_next;

		float dL_dtx_next = x_grad_mul_next * -h_x * tz2_next * dL_dJ02_next;
		float dL_dty_next = y_grad_mul_next * -h_y * tz2_next * dL_dJ12_next;
		float dL_dtz_next = -h_x * tz2_next * dL_dJ00_next - h_y * tz2_next * dL_dJ11_next + (2 * h_x * t_next.x) * tz3_next * dL_dJ02_next + (2 * h_y * t_next.y) * tz3_next * dL_dJ12_next;

		// if (trainpose) // if (trainpose) // if (false)
		if constexpr (TrainPose)
		{
			glm::vec3 dL_drho_next = {dL_dtx_next, dL_dty_next, dL_dtz_next};
			glm::vec3 dL_dtheta_next = cross({dL_dtx_next, dL_dty_next, dL_dtz_next}, {t_next.x, t_next.y, t_next.z});
			glm::mat3 dL_dW_next = {
				dL_dT00_next * J_next[0][0], dL_dT01_next * J_next[0][0], dL_dT02_next * J_next[0][0],
				dL_dT10_next * J_next[1][1], dL_dT11_next * J_next[1][1], dL_dT12_next * J_next[1][1],
				dL_dT00_next * J_next[0][2] + dL_dT10_next * J_next[1][2], dL_dT01_next * J_next[0][2] + dL_dT11_next * J_next[1][2], dL_dT02_next * J_next[0][2] + dL_dT12_next * J_next[1][2]
			};

			dL_dtheta_next.x += glm::dot(dL_dW_next[1], -W_next[2]) + glm::dot(dL_dW_next[2], W_next[1]);
			dL_dtheta_next.y += glm::dot(dL_dW_next[0], W_next[2]) + glm::dot(dL_dW_next[2], -W_next[0]);
			dL_dtheta_next.z += glm::dot(dL_dW_next[0], -W_next[1]) + glm::dot(dL_dW_next[1], W_next[0]);

			dL_dtau_next[6 * idx + 0] += dL_drho_next.x;
			dL_dtau_next[6 * idx + 1] += dL_drho_next.y;
			dL_dtau_next[6 * idx + 2] += dL_drho_next.z;
			dL_dtau_next[6 * idx + 3] += dL_dtheta_next.x;
			dL_dtau_next[6 * idx + 4] += dL_dtheta_next.y;
			dL_dtau_next[6 * idx + 5] += dL_dtheta_next.z;

			/// TODO: dL_dtau curr/next로 나누고 flow쓸때는 이거 적용하면됨.
			// if (!usecolor)
			// {
			// 	dL_dtau[6 * idx + 0] += dL_drho_next.x;
			// 	dL_dtau[6 * idx + 1] += dL_drho_next.y;
			// 	dL_dtau[6 * idx + 2] += dL_drho_next.z;
			// 	dL_dtau[6 * idx + 3] += dL_dtheta_next.x;
			// 	dL_dtau[6 * idx + 4] += dL_dtheta_next.y;
			// 	dL_dtau[6 * idx + 5] += dL_dtheta_next.z;
			// }
			// else /// TODO: 이거는 없어져야함. (잘못된 코드)
			// {
			// 	/// 만약 color & flow 다 사용하고 trainpose이면, current 기준으로 변환하고 더해줘야함.
			// 	// == (R^cnext_c)^T * dL_drho_next == dL_drho_next * R^cnext_c
			// 	// float3 dL_drho = transformVec4x3Transpose({dL_drho_next.x, dL_drho_next.y, dL_drho_next.z}, relpose);
			// 	// float3 t_rel = {relpose[12], relpose[13], relpose[14]};
			// 	// glm::mat3 t_x = skew_x(t_rel);
			// 	// glm::vec3 dL_drho_next_tx = glm::transpose(t_x) * dL_drho_next;
			// 	// float3 dL_dtheta = transformVec4x3Transpose({dL_drho_next_tx.x, dL_drho_next_tx.y, dL_drho_next_tx.z}, relpose)
			// 	// 	+ transformVec4x3Transpose({dL_dtheta_next.x, dL_dtheta_next.y, dL_dtheta_next.z}, relpose);

			// 	// dL_dtau[6 * idx + 0] += dL_drho.x;
			// 	// dL_dtau[6 * idx + 1] += dL_drho.y;
			// 	// dL_dtau[6 * idx + 2] += dL_drho.z;
			// 	// dL_dtau[6 * idx + 3] += dL_dtheta.x;
			// 	// dL_dtau[6 * idx + 4] += dL_dtheta.y;
			// 	// dL_dtau[6 * idx + 5] += dL_dtheta.z;
			// }
		}

		float3 dL_dmean_next = transformVec4x3Transpose({ dL_dtx_next, dL_dty_next, dL_dtz_next }, view_matrix_next);
		dL_dmeans[idx].x += dL_dmean_next.x;
		dL_dmeans[idx].y += dL_dmean_next.y;
		dL_dmeans[idx].z += dL_dmean_next.z;
	}

}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	// forward()에서와 마찬가지로 q, R, S, M 계산함.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	// back-diangonal elements는 변수가 중복되어 2배로 계산했었으므로, 0.5로 나누어주어야함.
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// Sigma = cov3D = M^T * M이므로
	// dSigma_dM = 2 * M
	// dL_dM = dL_dSigma * dSigma_dM
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R); // 하단에서 scale계산을 위해 transpose함.
	// dL_dM은 Sigma = M^TM에 대해 계산한 것이므로, 
	// 정상적으로 계산한 것이지만 밑에서 dL_dscale계산을 위해 transpose함.
	glm::mat3 dL_dMt = glm::transpose(dL_dM); 

	// Gradients of loss w.r.t. scale
	// dL_dscale = dL_dM * dM_dscale
	// dM_dscale = Rt임.
	// dL_dM * dM_dscale에서 scale은 대각선 elements만 존재하므로
	// (dL_dM의 row == dL_dM의 column) @ (Rt의 column)을 계산하면 됨.
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]); // Rt[0] (first column) @ dL_dMt[0] (first column)
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	// M = S * R인데 
	// S에 대각선 성분만 있으므로
	// 이를 M^T에 반영하면 Mt의 열에만 곱해주면 됨.
	// 단순 scalar곱이므로 S의 영향을 여기서 제외하고
	// dL_dq.x = \Sigma{k=1}\Sigma{l=1}dL_dMkl * dMkl_dq.x 를 계산가능하게함.
	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	// dL_dq.x = \Sigma{k=1}\Sigma{l=1}dL_dMtkl * dMtkl_dq.x
	// dL_dMt는 알고 있으므로, dMtkl_dq를 계산해야 하는데, scale 값을 고려하지 않아도 되므로
	// R의 각 성분에 대해 미분하고 합하면 됨.
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<bool UseColor, bool UseFlow, bool TrainPose, int C>
__global__ void preprocessCUDA(
	int P, int D, int M, 	// P: num_points, D: sh_degree, M: sh channels (total color channels)
	const float3* means,	// == means3D
	// int* stable_count,
	// bool* stable_status,
	const float* dL_derror,
	const int* radii,	
	const float* shs,	
	const bool* clamped,	// color value clamped or not. 1: RGB < 0, 0: RGB >= 0
	const float4* conic_opacity,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* viewmatrix,
	const float* viewmatrix_next,
	const float* relpose,
	const float* proj,			// projection matrix transpose of Tiw
	const glm::vec3* campos,	// translation of Twc 
	const float3* dL_dmean2D,	// == dL_dmean2D 	(input)
	const float3* dL_dmean2D_next,	// == dL_dmean2D 	(input)
	glm::vec3* dL_dmeans,		// == dL_dmeans3D	(input/output)
	float* dL_dcolor,			// (input)
	float* dL_dcov3D,			// (output)
	float* dL_dsh,				// (output)
	glm::vec3* dL_dscale,		// (output)
	glm::vec4* dL_drot,			// (output)
	float* dL_dtau,				// (output)
	float* dL_dtau_next)				// (output)
	// const bool usecolor,
	// const bool useflow,
	// const bool trainpose,
	// const bool updatestatus)			
{
	// auto idx = cg::this_grid().thread_rank();	// global thread idx
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= P || !(radii[idx] > 0))
		return;

	// float E_k = dL_derror[idx];
	// bool stable_prev = stable_status[idx];
	// if (!trainpose && updatestatus)
	// {
	// 	float opa = conic_opacity[idx].w;
	// 	// if (useflow)
	// 	// if (false)
	// 	// {
	// 	// 	if (dL_derror[idx] > opa * ERROR_SUM_THRES * 2 || (dL_derror[idx] / (float)(radii[idx] * radii[idx])) > opa * ERROR_NORM_THRES * 2)
	// 	// 	{
	// 	// 		if (stable_count[idx] > COUNT_MIN)
	// 	// 			stable_count[idx] -= 1;
	// 	// 	}
	// 	// 	else
	// 	// 	{
	// 	// 		if (stable_count[idx] < COUNT_MAX)
	// 	// 			stable_count[idx] += 1;
	// 	// 	}
	// 	// }
	// 	// else
	// 	// {
	// 	// 	if (dL_derror[idx] > ERROR_SUM_THRES || (dL_derror[idx] / (float)(radii[idx] * radii[idx])) > ERROR_NORM_THRES)
	// 	// 	{
	// 	// 		if (stable_count[idx] > COUNT_MIN)
	// 	// 			stable_count[idx] -= 1;
	// 	// 	}
	// 	// 	else
	// 	// 	{
	// 	// 		if (stable_count[idx] < COUNT_MAX)
	// 	// 			stable_count[idx] += 1;
	// 	// 	}
	// 	// }

	// 	// 만약 stable한 GS의 count가 UNSTABLE_COUNT_THRES보다 작으면, unstable로 변경.
	// 	if (stable_status[idx] && stable_count[idx] < UNSTABLE_COUNT_THRES)
	// 		stable_status[idx] = false;
	// 	// 만약 unstable한 GS의 count가 STABLE_COUNT_THRES보다 크면, stable로 변경.
	// 	else if (!stable_status[idx] && stable_count[idx] > STABLE_COUNT_THRES)
	// 		stable_status[idx] = true;
	// }

	glm::vec3 dL_dd = glm::vec3(0.f, 0.f, 0.f);
	if constexpr (UseColor)
	{
		if (shs)
		{
			dL_dd = computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dsh);
		}
	}

	// if (stable_prev && !trainpose)
	// 	return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	// float4 m_hom = transformPoint4x4(m, proj);	// m_hom을 world coordinate에서 image coordinate로 변환
	float3 v = transformPoint4x3(m, viewmatrix);	// m_hom을 world coordinate에서 camera coordinate로 변환
	float4 m_hom = transformPoint4x4(v, proj);		// m_hom을 camera coordinate에서 image coordinate로 변환
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	float3 v_next = { 0.f };
	float4 m_hom_next = { 0.f };
	float m_w_next = 0.f;
	if constexpr (UseFlow)
	{
		v_next = transformPoint4x3(m, viewmatrix_next);
		m_hom_next = transformPoint4x4(v_next, proj);
		m_w_next = 1.0f / (m_hom_next.w + 0.0000001f);
	}

	// 이전의 computeCov2DCUDA에서는 covariance(J,W)의 변경으로 인한 dL_dmean3D를 계산한것.
	// 여기서는 dL_dmean2D로 인한 dL_dmean3D를 계산함.
	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	// dL_dmean(3D)i = \Sigma_{k=1}\Sigma_{l=1} dL_dmean2Dkl * dmean2Dkl_dmean(3D)i
	// = dL_dmean2D * proj
	// proj: row-major로 잘 들어온듯?
	// x_2D.x = mul1 / w에 대해, w = proj[3] * m.x + proj[7] * m.y + proj[11] * m.z + proj[15]이므로
	// m.x와 연관되어 있기 때문에, 몫의 미분을 한 결과.
	// dL_dx3D.x = dL_dx2D.x * dx2D.x_x3D.x + dL_dx2D.y * dx2D.y_x3D.x

	/****** DEPRECATED ******/
	// glm::vec3 dL_dmean;
	// float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	// float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	// dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	// dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	// dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;
	/****** DEPRECATED ******/

	float3 dL_dm2 = dL_dmean2D[idx]; 
	float4 dL_dp = {
		dL_dm2.x * m_w,
		dL_dm2.y * m_w,
		0, // z coordinate of projected point is never used in forward pass
		-(dL_dm2.x * m_hom.x + dL_dm2.y * m_hom.y) * m_w * m_w
	};
	float3 dL_dv = {
		dL_dp.x * proj[0],
		dL_dp.y * proj[5],
		// add gradient on the depth (.z) here because it was not projected in forward
		// TODO: multiply .z with projection matrix entry?
		dL_dp.x * proj[8] + dL_dp.y * proj[9] + dL_dp.w * proj[11] + dL_dm2.z
	};
	float3 dL_dm = transformVec4x3Transpose(dL_dv, viewmatrix);
	glm::vec3 dL_dmean = {dL_dm.x, dL_dm.y, dL_dm.z};

	// Compute gradient updates due to computing colors from SHs
	// glm::vec3 dL_dd = glm::vec3(0.f, 0.f, 0.f);
	// if (shs && usecolor)
	// {
	// 	dL_dd = computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dsh);
	// 	dL_dmean += dL_dd;
	// }
	if constexpr (UseColor)
	{
		if (shs)
		{
			dL_dmean += dL_dd;
		}
	}

	if constexpr (TrainPose)
	{
		float3 t = {-viewmatrix[12], -viewmatrix[13], -viewmatrix[14]};

		glm::vec3 dL_drho = glm::vec3(dL_dv.x, dL_dv.y, dL_dv.z);
		glm::vec3 dL_dtheta = cross({dL_dv.x, dL_dv.y, dL_dv.z}, {v.x, v.y, v.z});

		if constexpr (UseColor)
		{	
			// float3 drho = transformVec4x3({dL_dd.x, dL_dd.y, dL_dd.z}, viewmatrix);
			// dL_drho += glm::vec3{drho.x, drho.y, drho.z};
			// glm::vec3 dxt = cross(dL_dd, {-t.x, -t.y, -t.z}); // = -dL_dd * [-t_c]_\times (cross함수에서 음수 붙음)
			// float3 dtheta = transformVec4x3({dxt.x, dxt.y, dxt.z}, viewmatrix); // = R^c_w * dxt
			// dL_dtheta += glm::vec3{dtheta.x, dtheta.y, dtheta.z};

			// tau가 Tcw에 대한거임!! 이게맞음 250619
			float3 minus_drho = transformVec4x3Transpose({dL_dd.x, dL_dd.y, dL_dd.z}, viewmatrix);
			glm::vec3 dL_drho_color = -glm::vec3{minus_drho.x, minus_drho.y, minus_drho.z};
			glm::vec3 dxt = cross(dL_drho_color, {-t.x, -t.y, -t.z}); // = -dL_drho_color * [t_c]_\times (cross함수에서 음수 붙음)
			glm::vec3 dL_dtheta_color = glm::vec3{dxt.x, dxt.y, dxt.z};
			dL_drho += dL_drho_color;
			dL_dtheta += dL_dtheta_color;
		}

		dL_dtau[6 * idx + 0] += dL_drho.x;
		dL_dtau[6 * idx + 1] += dL_drho.y;
		dL_dtau[6 * idx + 2] += dL_drho.z;
		dL_dtau[6 * idx + 3] += dL_dtheta.x;
		dL_dtau[6 * idx + 4] += dL_dtheta.y;
		dL_dtau[6 * idx + 5] += dL_dtheta.z;

		/// TODO: curr/next 따로 dL_dtau 분리하고 usecolor 없애야함
		// if (usecolor)
		// {
		// 	dL_dtau[6 * idx + 0] += dL_drho.x;
		// 	dL_dtau[6 * idx + 1] += dL_drho.y;
		// 	dL_dtau[6 * idx + 2] += dL_drho.z;
		// 	dL_dtau[6 * idx + 3] += dL_dtheta.x;
		// 	dL_dtau[6 * idx + 4] += dL_dtheta.y;
		// 	dL_dtau[6 * idx + 5] += dL_dtheta.z;
		// }
		// else /// TODO: 이거는 없어져야함. (잘못된 코드)
		// {
		// 	// dL_drho * relpose == (relpose)^T * dL_drho
		// 	// float3 dL_drho_next = transformVec4x3Transpose({dL_drho.x, dL_drho.y, dL_drho.z}, relpose);
		// 	// float3 t_rel = {relpose[12], relpose[13], relpose[14]};
		// 	// glm::mat3 t_x = skew_x(t_rel);
		// 	// // dL_drho * t_x == t_x^T * dL_drho
		// 	// glm::vec3 dL_drho_tx = glm::transpose(t_x) * dL_drho;
		// 	// // dL_drho_tx * relpose == (relpose)^T * dL_drho_tx
		// 	// float3 dL_dtheta_next = transformVec4x3Transpose({dL_drho_tx.x, dL_drho_tx.y, dL_drho_tx.z}, relpose)
		// 	// 	+ transformVec4x3Transpose({dL_dtheta.x, dL_dtheta.y, dL_dtheta.z}, relpose);

		// 	// dL_dtau[6 * idx + 0] += dL_drho_next.x;
		// 	// dL_dtau[6 * idx + 1] += dL_drho_next.y;
		// 	// dL_dtau[6 * idx + 2] += dL_drho_next.z;
		// 	// dL_dtau[6 * idx + 3] += dL_dtheta_next.x;
		// 	// dL_dtau[6 * idx + 4] += dL_dtheta_next.y;
		// 	// dL_dtau[6 * idx + 5] += dL_dtheta_next.z;
		// }
	}

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	if constexpr (UseFlow)
	{

		float3 dL_dm2_next = dL_dmean2D_next[idx]; 
		float4 dL_dp_next = {
			dL_dm2_next.x * m_w_next,
			dL_dm2_next.y * m_w_next,
			0, // z coordinate of projected point is never used in forward pass
			-(dL_dm2_next.x * m_hom_next.x + dL_dm2_next.y * m_hom_next.y) * m_w_next * m_w_next
		};
		float3 dL_dv_next = {
			dL_dp_next.x * proj[0],
			dL_dp_next.y * proj[5],
			// add gradient on the depth (.z) here because it was not projected in forward
			// TODO: multiply .z with projection matrix entry?
			dL_dp_next.x * proj[8] + dL_dp_next.y * proj[9] + dL_dp_next.w * proj[11] + dL_dm2_next.z
		};
		float3 dL_dm_next = transformVec4x3Transpose(dL_dv_next, viewmatrix_next);
		glm::vec3 dL_dmean_next = {dL_dm_next.x, dL_dm_next.y, dL_dm_next.z};

		// if (trainpose) // if (trainpose) // if (false)
		if constexpr (TrainPose)
		{
			float3 t_next = {-viewmatrix_next[12], -viewmatrix_next[13], -viewmatrix_next[14]};
			glm::vec3 dL_drho_next = glm::vec3(dL_dv_next.x, dL_dv_next.y, dL_dv_next.z);
			glm::vec3 dL_dtheta_next = cross({dL_dv_next.x, dL_dv_next.y, dL_dv_next.z}, {v_next.x, v_next.y, v_next.z});

			dL_dtau_next[6 * idx + 0] += dL_drho_next.x;
			dL_dtau_next[6 * idx + 1] += dL_drho_next.y;
			dL_dtau_next[6 * idx + 2] += dL_drho_next.z;
			dL_dtau_next[6 * idx + 3] += dL_dtheta_next.x;
			dL_dtau_next[6 * idx + 4] += dL_dtheta_next.y;
			dL_dtau_next[6 * idx + 5] += dL_dtheta_next.z;

			/// TODO: dL_dtau curr/next로 나누고 flow쓸때는 이거 적용하면됨.
			// if (!usecolor)
			// {
			// 	dL_dtau[6 * idx + 0] += dL_drho_next.x;
			// 	dL_dtau[6 * idx + 1] += dL_drho_next.y;
			// 	dL_dtau[6 * idx + 2] += dL_drho_next.z;
			// 	dL_dtau[6 * idx + 3] += dL_dtheta_next.x;
			// 	dL_dtau[6 * idx + 4] += dL_dtheta_next.y;
			// 	dL_dtau[6 * idx + 5] += dL_dtheta_next.z;
			// }
			// else /// TODO: 이거는 없어져야함. (잘못된 코드)
			// {
			// 	/// 만약 color & flow 다 사용하고 trainpose이면, current 기준으로 변환하고 더해줘야함.
			// 	// == (R^cnext_c)^T * dL_drho_next == dL_drho_next * R^cnext_c
			// 	float3 dL_drho = transformVec4x3Transpose({dL_drho_next.x, dL_drho_next.y, dL_drho_next.z}, relpose);
			// 	float3 t_rel = {relpose[12], relpose[13], relpose[14]};
			// 	glm::mat3 t_x = skew_x(t_rel);
			// 	glm::vec3 dL_drho_next_tx = glm::transpose(t_x) * dL_drho_next;
			// 	float3 dL_dtheta = transformVec4x3Transpose({dL_drho_next_tx.x, dL_drho_next_tx.y, dL_drho_next_tx.z}, relpose)
			// 		+ transformVec4x3Transpose({dL_dtheta_next.x, dL_dtheta_next.y, dL_dtheta_next.z}, relpose);

			// 	dL_dtau[6 * idx + 0] += dL_drho.x;
			// 	dL_dtau[6 * idx + 1] += dL_drho.y;
			// 	dL_dtau[6 * idx + 2] += dL_drho.z;
			// 	dL_dtau[6 * idx + 3] += dL_dtheta.x;
			// 	dL_dtau[6 * idx + 4] += dL_dtheta.y;
			// 	dL_dtau[6 * idx + 5] += dL_dtheta.z;
			// }
		}

		dL_dmeans[idx] += dL_dmean_next;
	}

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}


__device__ void computeFunCostCUDA(const float dx1, const float dy1, const float dx2, const float dy2,
								   const float in_weight, const float dL_dfcost, float2& dL_dflows)
{
	const float obs_fmag = L2_NORM(dx2, dy2);
	const float diff_fmag = L2_NORM(dx1 - dx2, dy1 - dy2);
	const float c = fun_fmag_c(obs_fmag);
	const float s = fmaxf(fun_fmag_scale(obs_fmag), ZDE);
	const float fisk_prob = fisk_dist_pdf(diff_fmag, c, s);
	const float mu = fisk_dist_pdf(LAMBDA*obs_fmag, c, s);

	const float out_weight = fisk_prob / fmaxf((fisk_prob + mu), ZDE);

	float dclamp1_ddiff_fmag = 0.0f;
	// if ((diff_fmag * EST_RF) < ZDE)
	//     dclamp1_ddiff_fmag = ZDE;
	// else
	//     dclamp1_ddiff_fmag = EST_RF;
	dclamp1_ddiff_fmag = ((diff_fmag * EST_RF) < ZDE) ? ZDE : EST_RF;

	float x_calc = diff_fmag * EST_RF;
	float x2byspowc = fast_powf(x_calc * x_calc / s, c);

	float dfisk_prob_dclamp1 = 
			-2 * c * x2byspowc * (x2byspowc + c * (x2byspowc - 1) + 1) 
			/ (fmaxf(pow3(x_calc) * pow3((x2byspowc + 1)), ZDE));

	float dL_ddiff_fmag = dL_dfcost * -in_weight * (1.0 / fmaxf(out_weight, ZDE)) 
				* (mu / fmaxf((fisk_prob + mu) * (fisk_prob + mu), ZDE))
				* dfisk_prob_dclamp1 * dclamp1_ddiff_fmag;

	float dL_dx1 = dL_ddiff_fmag * (dx1 - dx2) / fmaxf(diff_fmag, ZDE);
	float dL_dy1 = dL_ddiff_fmag * (dy1 - dy2) / fmaxf(diff_fmag, ZDE);

	dL_dflows.x = dL_dx1;
	dL_dflows.y = dL_dy1;
}

template <bool UseColor, bool UseFlow, uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA2(
	const uint2* __restrict__ ranges,			// == imgState.ranges
	const uint32_t* __restrict__ point_list,	// == binningState.point_list
	int P, int W, int H,
	const float* __restrict__ bg_color,			// background color (0,0,0)
	const float2* __restrict__ points_xy_image,	// == geomState.means2D,
	const float2* __restrict__ points_xy_image_next,	// == geomState.means2D_next,
	const float4* __restrict__ conic_opacity,	// == geomState.conic_opacity,
	const float4* __restrict__ covmap_next,		// == geomState.covmap_next,
	// const uint2* __restrict__ rect_next_min,
	// const uint2* __restrict__ rect_next_max,
	// const bool* __restrict__ stable_status,
	const int* __restrict__ radii,				// == geomState.radii
	const float* __restrict__ colors,			// == geomState.rgb
	const float* __restrict__ final_Ts,			// == imgState.accum_alpha
	const uint32_t* __restrict__ n_contrib,		// == imgState.n_contrib
	const float* __restrict__ flowimg,			// 
	const float* __restrict__ flowconf,			// 
	const float* __restrict__ gsflow,
	const float* __restrict__ dL_dpixels,		// == dL_dpix == grad_out_color
	const float* __restrict__ dL_dsilh,
	const float* __restrict__ dL_dflowraw,		// == dL_dflowraw == grad_out_flowraw
	const float* __restrict__ dL_dflowcost,		// == dL_dflow == grad_out_flow
	const float* __restrict__ dL_daux,
	const float* __restrict__ dL_daux2,
	const float* __restrict__ dL_daux3,
	// const float* __restrict__ tile_loss_sum,
	// const float* __restrict__ tile_loss_sigma,
	float3* __restrict__ dL_dmean2D,
	float3* __restrict__ dL_dmean2D_next,
	float4* __restrict__ dL_dconic2D,
	float4* __restrict__ dL_dcovmap_next,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_derror,
	float* __restrict__ dL_derror2,
	float* __restrict__ dL_derror3,
	// const bool usecolor,
	// const bool useflow,							// colors_precomp에 대한 loss.
	const bool trainpose,
	const bool useflowrawgrad)
{
	auto block = cg::this_thread_block();
	auto tid = block.thread_rank();
    
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };
	float2 pixf_next = {0.0f, 0.0f};

	bool inside = pix.x < W && pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	if constexpr (UseFlow)
	{
		if (inside)
		{
			pixf_next = { pixf.x + flowimg[pix_id], pixf.y + flowimg[H * W + pix_id] };
		}
	}

	// if (useflow && inside)
	// {
	// 	pixf_next = { pixf.x + flowimg[pix_id], pixf.y + flowimg[H * W + pix_id] };
	// }

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	bool done = !inside;
	int toDo = range.y - range.x;

	// bool done_flow = false;
	// if (useflow && inside)
	// {
	// 	// bool inside_next = pixf_next.x >= 0 && pixf_next.x < W && pixf_next.y >= 0 && pixf_next.y < H;
	// 	// inside = inside && inside_next;
	// 	// done = !inside;
	// 	bool inside_next_flow = pixf_next.x >= 0 && pixf_next.x < W && pixf_next.y >= 0 && pixf_next.y < H;
	// 	inside_next_flow = inside && inside_next_flow;
	// 	done_flow = !inside_next_flow;
	// }

	// bool done_flow = false;
	// if constexpr (UseFlow) 
	// {
	// 	if (inside)
	// 	{
	// 		// const float2 pixf_next = { pixf.x + flowimg[pix_id], pixf.y + flowimg[H * W + pix_id] };
	// 		bool inside_next_flow = pixf_next.x >= 0 && pixf_next.x < W && pixf_next.y >= 0 && pixf_next.y < H;
	// 		inside_next_flow = inside && inside_next_flow;
	// 		done_flow = !inside_next_flow;
	// 	}
	// }

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float2 dL_dmean2D_shared[BLOCK_SIZE];
	__shared__ float3 dL_dcolors_shared[BLOCK_SIZE];
	__shared__ float dL_dopacity_shared[BLOCK_SIZE];
	__shared__ float4 dL_dconic2D_shared[BLOCK_SIZE];

	// donguk
	__shared__ float2 collected_xy_next[BLOCK_SIZE];
	__shared__ float4 collected_covmap_next[BLOCK_SIZE];
	// __shared__ uint2 collected_rect_next_min[BLOCK_SIZE];
	// __shared__ uint2 collected_rect_next_max[BLOCK_SIZE];
	__shared__ float2 dL_dmean2D_flow_shared[BLOCK_SIZE];
	__shared__ float2 dL_dmean2D_next_flow_shared[BLOCK_SIZE];
	__shared__ float4 dL_dcovmap_next_shared[BLOCK_SIZE];
	__shared__ float dL_derror_shared[BLOCK_SIZE];
	__shared__ float dL_derror2_shared[BLOCK_SIZE];
	__shared__ float dL_derror3_shared[BLOCK_SIZE];

	dL_dmean2D_shared[tid]        = make_float2(0.f, 0.f);
    dL_dconic2D_shared[tid]       = make_float4(0.f, 0.f, 0.f, 0.f);
    dL_dopacity_shared[tid]       = 0.f;
    dL_dcolors_shared[tid]        = make_float3(0.f,0.f,0.f);
    dL_dmean2D_flow_shared[tid]   = make_float2(0.f, 0.f);
    dL_dmean2D_next_flow_shared[tid]=make_float2(0.f, 0.f);
    dL_dcovmap_next_shared[tid]   = make_float4(0.f, 0.f, 0.f, 0.f);
    dL_derror_shared[tid]         = 0.f;
    dL_derror2_shared[tid]        = 0.f;
    dL_derror3_shared[tid]        = 0.f;

	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C] = { 0 };

	if constexpr (UseColor)
	{
		if (inside) {
			#pragma unroll
			for (int i = 0; i < C; i++) {
				dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
			}
		}
	}
	// if (inside && usecolor) {
	// 	#pragma unroll
	// 	for (int i = 0; i < C; i++) {
	// 		dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
	// 	}
	// }

	float accum_rec_flow[2] = { 0 };
	const float dx1 = inside ? gsflow[pix_id] : 0.f;
	const float dy1 = inside ? gsflow[H * W + pix_id] : 0.f;
	const float dx2 = inside ? flowimg[pix_id] : 0.f;
	const float dy2 = inside ? flowimg[H * W + pix_id] : 0.f;
	float dL_dfcost = 0.0f;
	float2 dL_dflows = { 0.0f, 0.0f };

	const float loss_pix = inside ? dL_daux[pix_id] : 0.f;
	const float loss_pix2 = inside ? dL_daux2[pix_id] : 0.f;
	const float loss_pix3 = inside ? dL_daux3[pix_id] : 0.f;

	if constexpr (UseFlow)
	{
		if (inside) // && !done_flow
		{
			if (useflowrawgrad)
			{
				dL_dflows.x = dL_dflowraw[pix_id];
				dL_dflows.y = dL_dflowraw[H * W + pix_id];
			}
			else
			{
				dL_dfcost = dL_dflowcost[pix_id];
				computeFunCostCUDA(dx1, dy1, dx2, dy2, flowconf[pix_id], dL_dfcost, dL_dflows);
			}
		}
	}

	// if (inside && useflow) // && !done_flow
	// {
	// 	dL_dfcost = dL_dflowcost[pix_id];
	// 	computeFunCostCUDA(dx1, dy1, dx2, dy2, flowconf[pix_id], dL_dfcost, dL_dflows);
	// }
	// GaussianFlow normalized G = Flow/A (A=1-T_final): scale dL/dG by 1/A so the geometry chain
	// (dL_dflows*alpha*T) becomes the correct dL/df_k; the alpha block adds the -G*(1-accum_rec_silh) term.
	// (dx1,dy1 already hold the NORMALIZED forward flow G; dL_dflows=0 when !UseFlow so this is a no-op there.)
	const float A3 = 1.0f - T_final;
	const float invA3 = (A3 > 1e-6f) ? (1.0f / A3) : 0.0f;
	dL_dflows.x *= invA3;
	dL_dflows.y *= invA3;
	float dL_dsilh_pix = inside ? dL_dsilh[pix_id] : 0.f;

	float last_alpha = 0.f;
	float last_color[C] = { 0.f };
	float last_flow[2] = { 0 }; // donguk
	float last_silh = 0.f; 

	float accum_rec_silh = 0.f; 

	const float ddelx_dx = 0.5f * W;
	const float ddely_dy = 0.5f * H;
	__shared__ int skip_counter;

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		const int progress = i * BLOCK_SIZE + tid;
		int access_idx = range.y - progress - 1;

		if (range.x + progress < range.y)
		// if (progress < (range.y - range.x) && access_idx >= 0)
		{
			// const int coll_id = point_list[range.y - progress - 1];
			const int coll_id = point_list[access_idx];
			// if (coll_id < P)
			// {
				collected_id[tid] = coll_id;
				collected_xy[tid] = points_xy_image[coll_id];
				collected_conic_opacity[tid] = conic_opacity[coll_id];
				#pragma unroll
				for (int i = 0; i < C; i++) {
					collected_colors[i * BLOCK_SIZE + tid] = colors[coll_id * C + i];
					
				}
				if constexpr (UseFlow)
				{
					// collected_xy_next[block.thread_rank()] = points_xy_image_next[coll_id];
					// collected_covmap_next[block.thread_rank()] = covmap_next[coll_id];
					collected_xy_next[tid] = points_xy_image_next[coll_id];
					collected_covmap_next[tid] = covmap_next[coll_id];

					// collected_rect_next_min[block.thread_rank()] = rect_next_min[coll_id];
					// collected_rect_next_max[block.thread_rank()] = rect_next_max[coll_id];
				}
			// }
			// else
			// {
			// 	printf("coll_id : %d is out of bound!!!!", coll_id);
			// }
		}

		for (int j = 0; j < min(BLOCK_SIZE, toDo); j++) 
		{
			block.sync();
			if (tid == 0) {
				skip_counter = 0;
			}
			block.sync();
			bool skip = done;
			contributor = done ? contributor : contributor - 1;
			skip |= contributor >= last_contributor;

			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			skip |= power > 0.0f;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			skip |= alpha < 1.0f / 255.0f;

			if (skip) {
				atomicAdd(&skip_counter, 1);
			}
			block.sync();
			if (skip_counter == BLOCK_SIZE) {
				continue;
			}

			T = skip ? T : T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T; // weight

			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			// if (global_id >= P)
			// {
			// 	printf("global_id : %d is out of bound!!!!", global_id);
			// 	skip = true;
			// }

			// atomicAdd(&(dL_derror[global_id]), loss_pix * dchannel_dcolor);
			// atomicAdd(&(dL_derror2[global_id]), loss_pix2 * dchannel_dcolor);
			// atomicAdd(&(dL_derror3[global_id]), loss_pix3 * dchannel_dcolor);
			dL_derror_shared[tid] = skip ? 0.0f : loss_pix * dchannel_dcolor;
			dL_derror2_shared[tid] = skip ? 0.0f : loss_pix2 * dchannel_dcolor;
			dL_derror3_shared[tid] = skip ? 0.0f : loss_pix3 * dchannel_dcolor;

			float local_dL_dcolors[3];
			accum_rec_silh = skip ? accum_rec_silh : last_alpha * last_silh + (1.f - last_alpha) * accum_rec_silh;
			// silh's per-gaussian "colour" is 1, so on a NON-skip contributor last_silh=1; on a skip it must be
			// KEPT (mirror the colour channel's `last_color = skip ? last_color : c`). The old `skip ? 0 : 1`
			// zeroed it on interior skips -> the next contributor's accum_rec_silh lost the last_alpha*last_silh
			// term -> accum_rec_silh underestimated A_behind -> (1-accum_rec_silh) overestimated dL_dalpha (3-9x).
			last_silh = skip ? last_silh : 1.0f;
			// silh (accumulated opacity A=Σα·T) is a channel whose per-gaussian value is 1, so
			// ∂A/∂α_k = T_k·(1 − A_behind) = T_k·(1 − accum_rec_silh). The T_k is supplied by the
			// `dL_dalpha *= T` at the end of this iter (same as the colour channel), so this term must
			// NOT carry the extra dchannel_dcolor=α·T_k weight (that double-counts α·T_k -> the silh
			// backward was off by α·T_k, breaking any silh-routed gradient incl. the gsflow /A normalizer).
			dL_dalpha += (1.0f - accum_rec_silh) * dL_dsilh_pix; // skip ? 0.0f :

			if constexpr (UseColor)
			{
				#pragma unroll
				for (int ch = 0; ch < C; ch++)
				{
					const float c = collected_colors[ch * BLOCK_SIZE + j];
					// Update last color (to be used in the next iteration)
					accum_rec[ch] = skip ? accum_rec[ch] : last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
					last_color[ch] = skip ? last_color[ch] : c;

					const float dL_dchannel = dL_dpixel[ch];
					dL_dalpha += (c - accum_rec[ch]) * dL_dchannel; // skip ? 0.0f : 
					local_dL_dcolors[ch] = skip ? 0.0f : dchannel_dcolor * dL_dchannel;
				}
				dL_dcolors_shared[tid].x = local_dL_dcolors[0];
				dL_dcolors_shared[tid].y = local_dL_dcolors[1];
				dL_dcolors_shared[tid].z = local_dL_dcolors[2];
			}

			if constexpr (UseFlow) // && !done_flow
			{					
				// if (!done_flow)
				// {
					float f1 = (collected_covmap_next[j].x * -d.x + collected_covmap_next[j].y * -d.y 
						+ collected_xy_next[j].x - pixf.x);
					float f2 = (collected_covmap_next[j].z * -d.x + collected_covmap_next[j].w * -d.y 
								+ collected_xy_next[j].y - pixf.y);

					accum_rec_flow[0] = skip ? accum_rec_flow[0] : last_alpha * last_flow[0] + (1.f - last_alpha) * accum_rec_flow[0];
					accum_rec_flow[1] = skip ? accum_rec_flow[1] : last_alpha * last_flow[1] + (1.f - last_alpha) * accum_rec_flow[1];

					last_flow[0] = skip ? last_flow[0] : f1;
					last_flow[1] = skip ? last_flow[1] : f2;

					// normalized G=Flow/A: subtract G*(1-accum_rec_silh) (the denominator's alpha-dependence). dx1,dy1
					// = normalized forward flow G; accum_rec_silh = the alpha suffix. dL_dflows already /A (Term1).
					dL_dalpha += ((f1 - accum_rec_flow[0]) - dx1 * (1.0f - accum_rec_silh)) * dL_dflows.x;
					dL_dalpha += ((f2 - accum_rec_flow[1]) - dy1 * (1.0f - accum_rec_silh)) * dL_dflows.y;

					dL_dcovmap_next_shared[tid].x = skip ? 0.0f : -d.x * dchannel_dcolor * dL_dflows.x;
					dL_dcovmap_next_shared[tid].y = skip ? 0.0f : -d.y * dchannel_dcolor * dL_dflows.x;
					dL_dcovmap_next_shared[tid].z = skip ? 0.0f : -d.x * dchannel_dcolor * dL_dflows.y;
					dL_dcovmap_next_shared[tid].w = skip ? 0.0f : -d.y * dchannel_dcolor * dL_dflows.y;
					// dL_dcovmap_next_shared[tid].x = 0.0f;
					// dL_dcovmap_next_shared[tid].y = 0.0f;
					// dL_dcovmap_next_shared[tid].z = 0.0f;
					// dL_dcovmap_next_shared[tid].w = 0.0f;

					dL_dmean2D_flow_shared[tid].x = skip ? 0.0f : dchannel_dcolor * (dL_dflows.x * collected_covmap_next[j].x
												+ dL_dflows.y * collected_covmap_next[j].z) * -1.f * ddelx_dx;
					dL_dmean2D_flow_shared[tid].y = skip ? 0.0f : dchannel_dcolor * (dL_dflows.x * collected_covmap_next[j].y
												+ dL_dflows.y * collected_covmap_next[j].w) * -1.f * ddely_dy;
					dL_dmean2D_next_flow_shared[tid].x = skip ? 0.0f : dchannel_dcolor * dL_dflows.x * ddelx_dx;
					dL_dmean2D_next_flow_shared[tid].y = skip ? 0.0f : dchannel_dcolor * dL_dflows.y * ddely_dy;
				// }
			}

			dL_dalpha *= T; // skip ? 1.0f : 
			// Update last alpha (to be used in the next iteration)
			last_alpha = skip ? last_alpha : alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			// float bg_dot_dpixel = 0.f;
			// #pragma unroll
			// for (int i = 0; i < C; i++) {
			// 	bg_dot_dpixel +=  bg_color[i] * dL_dpixel[i];
			// }
			// dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			dL_dmean2D_shared[tid].x = skip ? 0.f : dL_dG * dG_ddelx * ddelx_dx;
			dL_dmean2D_shared[tid].y = skip ? 0.f : dL_dG * dG_ddely * ddely_dy;
			dL_dconic2D_shared[tid].x = skip ? 0.f : -0.5f * gdx * d.x * dL_dG;
			dL_dconic2D_shared[tid].y = skip ? 0.f : -0.5f * gdx * d.y * dL_dG;
			dL_dconic2D_shared[tid].w = skip ? 0.f : -0.5f * gdy * d.y * dL_dG;
			dL_dopacity_shared[tid] = skip ? 0.f : G * dL_dalpha;

			render_cuda_reduce_sum(block, 
				dL_dmean2D_shared,
				dL_dconic2D_shared,
				dL_dopacity_shared,
				dL_dcolors_shared, 
				dL_derror_shared, 		// donguk
				dL_derror2_shared,
				dL_derror3_shared,
				dL_dcovmap_next_shared,	// donguk
				dL_dmean2D_flow_shared,	// donguk
				dL_dmean2D_next_flow_shared	// donguk
			);	
			
			// if (useflow)
			// {
			// 	render_cuda_reduce_sum(block, 
			// 		dL_dcovmap_next_shared,	// donguk
			// 		dL_dmean2D_flow_shared,	// donguk
			// 		dL_dmean2D_next_flow_shared	// donguk
			// 	);
			// }

			if (tid == 0) 
			{
				float2 dL_dmean2D_acc = dL_dmean2D_shared[0];
				float4 dL_dconic2D_acc = dL_dconic2D_shared[0];
				float dL_dopacity_acc = dL_dopacity_shared[0];
				float3 dL_dcolors_acc = dL_dcolors_shared[0];

				atomicAdd(&dL_dmean2D[global_id].x, dL_dmean2D_acc.x);
				atomicAdd(&dL_dmean2D[global_id].y, dL_dmean2D_acc.y);
				atomicAdd(&dL_dconic2D[global_id].x, dL_dconic2D_acc.x);
				atomicAdd(&dL_dconic2D[global_id].y, dL_dconic2D_acc.y);
				atomicAdd(&dL_dconic2D[global_id].w, dL_dconic2D_acc.w);
				atomicAdd(&dL_dopacity[global_id], dL_dopacity_acc);
				atomicAdd(&dL_dcolors[global_id * C + 0], dL_dcolors_acc.x);
				atomicAdd(&dL_dcolors[global_id * C + 1], dL_dcolors_acc.y);
				atomicAdd(&dL_dcolors[global_id * C + 2], dL_dcolors_acc.z);

				// donguk
				atomicAdd(&dL_derror[global_id], dL_derror_shared[0]);
				atomicAdd(&dL_derror2[global_id], dL_derror2_shared[0]);
				atomicAdd(&dL_derror3[global_id], dL_derror3_shared[0]);

				// donguk
				if constexpr (UseFlow)
				{
					float2 dL_dmean2D_flow_acc = dL_dmean2D_flow_shared[0];
					float2 dL_dmean2D_next_flow_acc = dL_dmean2D_next_flow_shared[0];
					float4 dL_dcovmap_next_acc = dL_dcovmap_next_shared[0];

					// donguk
					atomicAdd(&dL_dmean2D[global_id].x, dL_dmean2D_flow_acc.x);
					atomicAdd(&dL_dmean2D[global_id].y, dL_dmean2D_flow_acc.y);
					atomicAdd(&dL_dmean2D_next[global_id].x, dL_dmean2D_next_flow_acc.x);
					atomicAdd(&dL_dmean2D_next[global_id].y, dL_dmean2D_next_flow_acc.y);
					atomicAdd(&dL_dcovmap_next[global_id].x, dL_dcovmap_next_acc.x);
					atomicAdd(&dL_dcovmap_next[global_id].y, dL_dcovmap_next_acc.y);
					atomicAdd(&dL_dcovmap_next[global_id].z, dL_dcovmap_next_acc.z);
					atomicAdd(&dL_dcovmap_next[global_id].w, dL_dcovmap_next_acc.w);
				}
			}
		}

	}
}


/////// ******** error_per_gs2, error_per_gs3 미구현 ********
// Backward version of the rendering procedure.
// dL_dpixels가 주어졌다.
// dL_dmean2D, dL_dconic2D, dL_dopacity, dL_dcolors를 계산하자...!
// dL_dpixels를 이용해 어떻게 연결시킬 것인가가 핵심..!
// dL_dcolors는 SUM{alpha * T * color이므로} 
template <bool UseColor, bool UseFlow, uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,			// == imgState.ranges
	const uint32_t* __restrict__ point_list,	// == binningState.point_list
	int P, int W, int H,
	const float* __restrict__ bg_color,			// background color (0,0,0)
	const float2* __restrict__ points_xy_image,	// == geomState.means2D,
	const float2* __restrict__ points_xy_image_next,	// == geomState.means2D_next,
	const float4* __restrict__ conic_opacity,	// == geomState.conic_opacity,
	const float4* __restrict__ covmap_next,		// == geomState.covmap_next,
	// const uint2* __restrict__ rect_next_min,
	// const uint2* __restrict__ rect_next_max,
	// const bool* __restrict__ stable_status,
	const int* __restrict__ radii,				// == geomState.radii
	const float* __restrict__ colors,			// == geomState.rgb
	const float* __restrict__ final_Ts,			// == imgState.accum_alpha
	const uint32_t* __restrict__ n_contrib,		// == imgState.n_contrib
	const float* __restrict__ flowimg,			// 
	const float* __restrict__ flowconf,			// 
	const float* __restrict__ gsflow,
	const float* __restrict__ dL_dpixels,		// == dL_dpix == grad_out_color
	const float* __restrict__ dL_dsilh,
	const float* __restrict__ dL_dflowraw,		// == dL_dflowraw == grad_out_flowraw
	const float* __restrict__ dL_dflowcost,		// == dL_dflow == grad_out_flow
	const float* __restrict__ dL_daux,
	const float* __restrict__ dL_daux2,
	const float* __restrict__ dL_daux3,
	// const float* __restrict__ tile_loss_sum,
	// const float* __restrict__ tile_loss_sigma,
	float3* __restrict__ dL_dmean2D,
	float3* __restrict__ dL_dmean2D_next,
	float4* __restrict__ dL_dconic2D,
	float4* __restrict__ dL_dcovmap_next,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_derror,
	float* __restrict__ dL_derror2,
	float* __restrict__ dL_derror3,
	// const bool usecolor,
	// const bool useflow,							// colors_precomp에 대한 loss.
	const bool trainpose)
{
	int blockId1D = blockIdx.x + blockIdx.y * gridDim.x;
	// if (tile_loss_sum[blockId1D] < BLOCK_SUM_THRES)
	// if (useflow)
	
	// if (false)
	// {
	// 	if (tile_loss_sum[blockId1D] < SKIP_BLOCK_SUM_THRES * 2 && tile_loss_sigma[blockId1D] < SKIP_BLOCK_SIGMA_THRES)
	// 		return;
	// }
	// else
	// {
	// 	if (tile_loss_sum[blockId1D] < SKIP_BLOCK_SUM_THRES && tile_loss_sigma[blockId1D] < SKIP_BLOCK_SIGMA_THRES)
	// 		return;
	// }

	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();				// 지금 block id(x,y)
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;	// horizontal block 개수
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };	// 현재 block의 min pixel
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };		// 현재 block의 max pixel
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };	// 현재 thread의 pixel
	const uint32_t pix_id = W * pix.y + pix.x;		// serialized pixel id
	const float2 pixf = { (float)pix.x, (float)pix.y };
	float2 pixf_next = {0.0f, 0.0f};

	bool inside = pix.x < W && pix.y < H;
	// ranges == [closest index / farthest(는 미포함)) index of binningState.point_list_keys in each tile
	// range == [closest(or first) index / farthest(or last)) index of point_list in this tile
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	if constexpr (UseFlow)
	{
		if (inside)
		{
			pixf_next = { pixf.x + flowimg[pix_id], pixf.y + flowimg[H * W + pix_id] };
		}
	}
	// if (useflow && inside)
	// {
	// 	pixf_next = { pixf.x + flowimg[pix_id], pixf.y + flowimg[H * W + pix_id] };
	// }

	// total rounds needed for per-block_size calculation
	// total amount of Gaussian Splattings  / BLOCK_SIZE (==BLOCK_X * BLOCK_Y)
	// 이 Gaussian Splattings를 처리하기 위해 필요한 총 block의 개수
	// 즉, 한 tile에서 처리해야할 point_list에서의 Gaussian Splatting index를 전부 가져와서, 이걸 또 block단위로 나누어서 진행함.
	// 예를 들어 이 함수가 호출하는 pixel(thread)이 속한 tile(block)에 GS가 1000개 필요하고, BLOCK_SIZE가 256이면,
	// rounds = 4가 된다.
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	bool done = !inside;
	int toDo = range.y - range.x;

	// 250317
	// bool done_flow = false;
	// if (useflow)
	// {
	// 	bool inside_next = pixf_next.x >= 0 && pixf_next.x < W && pixf_next.y >= 0 && pixf_next.y < H;
	// 	inside = inside && inside_next;
	// 	done = !inside;
	// }
	// bool done_flow = false;
	// if constexpr (UseFlow) 
	// {
	// 	if (inside)
	// 	{
	// 		// const float2 pixf_next = { pixf.x + flowimg[pix_id], pixf.y + flowimg[H * W + pix_id] };
	// 		bool inside_next_flow = pixf_next.x >= 0 && pixf_next.x < W && pixf_next.y >= 0 && pixf_next.y < H;
	// 		inside_next_flow = inside && inside_next_flow;
	// 		done_flow = !inside_next_flow;
	// 	}
	// }

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float2 collected_xy_next[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float4 collected_covmap_next[BLOCK_SIZE];
	// __shared__ int collected_radii[BLOCK_SIZE];
	// __shared__ uint2 collected_rect_next_min[BLOCK_SIZE];
	// __shared__ uint2 collected_rect_next_max[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;

	// n_contrib[pix_id]: how many gaussian splattings contributed to render this pixel
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec_color[C] = { 0 };
	float accum_rec_flow[2] = { 0 };
	float dL_dpixel[C] = { 0 }; 
	// float dL_dflow[2] = { 0 };

	const float dx1 = inside ? gsflow[pix_id] : 0.f;
	const float dy1 = inside ? gsflow[H * W + pix_id] : 0.f;
	const float dx2 = inside ? flowimg[pix_id] : 0.f;
	const float dy2 = inside ? flowimg[H * W + pix_id] : 0.f;
	float dL_dfcost = 0.0f;
	float2 dL_dflows = { 0.0f, 0.0f };

	const float loss_pix = inside ? dL_daux[pix_id] : 0.f;
	const float loss_pix2 = inside ? dL_daux2[pix_id] : 0.f;
	const float loss_pix3 = inside ? dL_daux3[pix_id] : 0.f;
	float dL_dsilh_pix = inside ? dL_dsilh[pix_id] : 0.f;

	if (inside)
	{
		if constexpr (UseColor)
		{
			for (int i = 0; i < C; i++)
				dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];  // == grad_out_color. RGB마다 값을 가지고 있음.
		}

		if constexpr (UseFlow) // && !done_flow
		{
			// if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
			// 	printf("dL_dflowcost[pix_id]: %.9f\n", dL_dflowcost[pix_id]);
			// }

			dL_dfcost = dL_dflowcost[pix_id];
			computeFunCostCUDA(dx1, dy1, dx2, dy2, flowconf[pix_id], dL_dfcost, dL_dflows);
			// GaussianFlow normalized G=Flow/A: scale dL/dG by 1/A (A=1-T_final) so the geometry chain is
			// auto-correct; the alpha block adds the -G*(1-accum_rec_silh) term. (render_grad/pose path.)
			{ const float A3 = 1.0f - T_final; const float invA3 = (A3 > 1e-6f) ? (1.0f / A3) : 0.0f;
			  dL_dflows.x *= invA3; dL_dflows.y *= invA3; }
			// if (trainpose)
			// {
			// 	dL_dflows.x = dL_dflowraw[pix_id];
			// 	dL_dflows.y = dL_dflowraw[H * W + pix_id];
			// }
			// else
			// {
			// 	dL_dfcost = dL_dflowcost[pix_id];
			// 	computeFunCostCUDA(dx1, dy1, dx2, dy2, flowconf[pix_id], dL_dfcost, dL_dflows);
			// }
		}
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };
	float last_flow[2] = { 0 };

	float last_silh = 0.f; 
	float accum_rec_silh = 0.f; 

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5f * W;
	const float ddely_dy = 0.5f * H;
	const float ddelx_ddely = W / H;
	const float ddely_ddelx = H / W;

	// Traverse all Gaussians
	// point_list: All sorted GS index by tile->depth	
	// progress는 처리해야할 첫 GS부터 몇번째 GS인지를 나타낸다.
	// for문 한번마다 BLOCK_SIZE만큼의 처리해야할 GS를 가져온다.
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		// (0부터 시작) i번째 block에서의 현재 thread idx (Gaussian Splatting index)를 구함.
		// thread_rank()는 block안에서 몇번째 thread인지 나타냄.
		// 어차피 thread_rank()는 0~BLOCK_SIZE-1까지의 값을 가지므로, 
		// i * BLOCK_SIZE + block.thread_rank()는 한 BLOCK_SIZE 안에서의 처리해야할 GS index중 하나가 됨.
		// 이것이 실제 pixel단위의 block, thread index는 아니므로 주의!!!!!!!!!!!!!
		const int progress = i * BLOCK_SIZE + block.thread_rank(); 

		// range.y 에 해당하는 index가 아닌 경우에만 진행
		// collected_id, collected_xy, collected_conic_opacity, collected_colors에는 BLOCK내에서 GS의 정보가 담긴다.
		// 이렇게 저장함으로써, 실제 하드웨어에서 block내의 서로 다른 thread들이 정보를 공유한다.
		// 각 thread는 block내에서의 pixel 1개를 다루는데, block내에서는 어차피 collected_id~collected_colors정보가 동일하다.
		// 각 thread마다 이렇게 한 index마다의 정보를 저장한 후 sync한다.
		// 그리고 아래에서 for (int j; ~~~) 를 진행하면서 각자 저장해두었던 정보를 함께 사용한다.
		if (range.x + progress < range.y)
		{
			// 뒤에서부터 계산함.
			const int coll_id = point_list[range.y - progress - 1];
			// if (coll_id < P)
			// {
				collected_id[block.thread_rank()] = coll_id;							// GS의 global index
				collected_xy[block.thread_rank()] = points_xy_image[coll_id];			// GS의 2D mean
				collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];	// GS의 conic + opacity
				// collected_radii[block.thread_rank()] = radii[coll_id];					// GS의 radius

				for (int i = 0; i < C; i++)
					collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
				if constexpr (UseFlow)
				{
					collected_xy_next[block.thread_rank()] = points_xy_image_next[coll_id];
					collected_covmap_next[block.thread_rank()] = covmap_next[coll_id];
					// collected_rect_next_min[block.thread_rank()] = rect_next_min[coll_id];
					// collected_rect_next_max[block.thread_rank()] = rect_next_max[coll_id];
				}
			// }
			// else
			// {
			// 	printf("coll_id : %d is out of bound!!!!", coll_id);
			// }
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			// tile에서 모든 GS를 가져왔으므로, 각 pixel에 대해 정말로 기여한 GS만 사용하기 위한 코드
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			// alpha를 곱해 나간 결과가 T이므로, 이제는 나눠가면서 진행함. 뒤에서 id가 저장되어 있으므로, 뒤에서부터 줄여나감.
			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			// if (global_id >= P)
			// {
			// 	printf("global_id : %d is out of bound!!!!", global_id);
			// 	continue;
			// }
			// const int global_radii = collected_radii[j];

			atomicAdd(&(dL_derror[global_id]), loss_pix * dchannel_dcolor);
			atomicAdd(&(dL_derror2[global_id]), loss_pix2 * dchannel_dcolor);
			atomicAdd(&(dL_derror3[global_id]), loss_pix3 * dchannel_dcolor);

			accum_rec_silh = last_alpha * last_silh + (1.f - last_alpha) * accum_rec_silh;
			last_silh = 1.0f;
			dL_dalpha += (1.0f - accum_rec_silh) * dL_dsilh_pix; // silh value=1, T_k supplied by *T below (drop dchannel_dcolor; renderCUDA has no skip)

			// Stable한 GS이면서 trainpose 하지 않는 경우 계산하지 않는다．
			// if (stable_status[global_id] && !trainpose)
			// 	continue;

			// C == NUM_CHANNELS. RGB에 대해 계산함
			if constexpr (UseColor)
			{
				for (int ch = 0; ch < C; ch++)
				{
					const float c = collected_colors[ch * BLOCK_SIZE + j];
					// Update last color (to be used in the next iteration)
					accum_rec_color[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec_color[ch];
					last_color[ch] = c;

					const float dL_dchannel = dL_dpixel[ch];
					// dL_dalpha = dL_dchannel * dchannel_dalpha이고
					// dL_dchannel은 아니까 dchannel_dalpha를 구하자.

					// 각 channel은 독립적이고, C = c_1 * alpha_1 + c_2 * alpha_2 (1-alpha_1) + ... 이다.
					// 만약 alpha2를 예시로 들면
					// dchannel_dalpha2 = (1-alpha_1) * (c2 - (c_3 *alpha_3 + (1-alpha_3) * {c_4 * alpha_4 + (1-alpha_4) * {c_5 * alpha_5 + ...}}}
					// ...
					// = (1-alpha_1)(1-alpha_2)...(1-alpha_(k-1)) * (c - accum_rec[ch]) 인 것을 확인할 수 있다.
					// (1-alpha_1)(1-alpha_2)...(1-alpha_(k-1))는 T이므로 밑에서 곱해준다.
					dL_dalpha += (c - accum_rec_color[ch]) * dL_dchannel;
					// Update the gradients w.r.t. color of the Gaussian. 
					// Atomic, since this pixel is just one of potentially
					// many that were affected by this Gaussian.

					// pixel의 color C는 c_i * alpha * T의 합이므로, dL_dcolors = atomicAdd로
					// chain rule을 통해 더해준다. dL_dcolors += dL_dchannel * dchannel_dcolor
					atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
				}
			}

			// [dL_dalpha with flow 계산]
			// == dL_dx1 * dx1_dalpha + dL_dy1 * dy1_dalpha
			// if (useflow && con_o.w > 0.5f)
			if constexpr (UseFlow) //  && !done_flow
			{
				// if ((collected_rect_next_min[j].x * BLOCK_X < pixf_next.x) && (collected_rect_next_max[j].x * BLOCK_X > pixf_next.x)
				// 	&& (collected_rect_next_min[j].y * BLOCK_Y < pixf_next.y) && (pixf_next.y < collected_rect_next_max[j].y * BLOCK_Y))
				{
					float f1 = (collected_covmap_next[j].x * -d.x + collected_covmap_next[j].y * -d.y 
								+ collected_xy_next[j].x - pixf.x);
					float f2 = (collected_covmap_next[j].z * -d.x + collected_covmap_next[j].w * -d.y 
								+ collected_xy_next[j].y - pixf.y);

					accum_rec_flow[0] = last_alpha * last_flow[0] + (1.f - last_alpha) * accum_rec_flow[0];
					accum_rec_flow[1] = last_alpha * last_flow[1] + (1.f - last_alpha) * accum_rec_flow[1];

					last_flow[0] = f1;
					last_flow[1] = f2;

					// normalized G=Flow/A: subtract G*(1-accum_rec_silh) (dx1,dy1 = normalized gsflow; dL_dflows /A).
					dL_dalpha += ((f1 - accum_rec_flow[0]) - dx1 * (1.0f - accum_rec_silh)) * dL_dflows.x;
					dL_dalpha += ((f2 - accum_rec_flow[1]) - dy1 * (1.0f - accum_rec_silh)) * dL_dflows.y;

					atomicAdd(&(dL_dcovmap_next[global_id].x), -d.x * dchannel_dcolor * dL_dflows.x);
					atomicAdd(&(dL_dcovmap_next[global_id].y), -d.y * dchannel_dcolor * dL_dflows.x);
					atomicAdd(&(dL_dcovmap_next[global_id].z), -d.x * dchannel_dcolor * dL_dflows.y);
					atomicAdd(&(dL_dcovmap_next[global_id].w), -d.y * dchannel_dcolor * dL_dflows.y);

					// atomicAdd(&(dL_dmean2D[global_id].x), dchannel_dcolor * (dL_dflows.x * collected_covmap_next[j].x
					// 			+ dL_dflows.y * collected_covmap_next[j].z * ddely_ddelx) * -1.f * ddelx_dx);
					// atomicAdd(&(dL_dmean2D[global_id].y), dchannel_dcolor * (dL_dflows.x * collected_covmap_next[j].y * ddelx_ddely
					// 			+ dL_dflows.y * collected_covmap_next[j].w) * -1.f * ddely_dy);
					// atomicAdd(&(dL_dmean2D_next[global_id].x), dchannel_dcolor * dL_dflows.x * ddelx_dx);
					// atomicAdd(&(dL_dmean2D_next[global_id].y), dchannel_dcolor * dL_dflows.y * ddely_dy);

					atomicAdd(&(dL_dmean2D[global_id].x), dchannel_dcolor * (dL_dflows.x * collected_covmap_next[j].x
								+ dL_dflows.y * collected_covmap_next[j].z) * -1.f * ddelx_dx);
					atomicAdd(&(dL_dmean2D[global_id].y), dchannel_dcolor * (dL_dflows.x * collected_covmap_next[j].y 
								+ dL_dflows.y * collected_covmap_next[j].w) * -1.f * ddely_dy);
					atomicAdd(&(dL_dmean2D_next[global_id].x), dchannel_dcolor * dL_dflows.x * ddelx_dx);
					atomicAdd(&(dL_dmean2D_next[global_id].y), dchannel_dcolor * dL_dflows.y * ddely_dy);
				}
			}
		
			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend

			// 250418 bg코드 삭제
			{
			// float bg_dot_dpixel = 0;
			// for (int i = 0; i < C; i++)
			// 	bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			// forward()에서 마지막에 background color * T_final 더해줌.
			// 따라서, dL_dalpha = dL_dchannel * dchannel_dalpha 이고,
			// dchannel_dalpha = bg_color * T_final / (1-alpha) * -1 이다.
			// bg_dot_dpixel = bg_color * dL_dchannel이므로 성립.
			// dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;
			}
		
			// Helpful reusable temporary variables
			// alpha는 G, opacity와 연관되어 있다. 따라서 dL_dG와 dL_dopacity를 구할 때 더해주어야 한다.
			// dL_dalpha를 구했으므로 이제 dL_dG에도 적용하자.
			// dL_dG = dL_dalpha * dalpha_dG
			// alpha = con_o.w * G 이므로
			// dalpha_dG = con_o.w
			const float dL_dG = con_o.w * dL_dalpha;

			// d.x, d.y = (x - mean.x), (y - mean.y)이다.
			// G = exp(-0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y) 이므로
			// dG_ddelx = -0.5f * con_o.x * 2 * d.x * G - con_o.y * d.y * G
			// dg_ddely = -0.5f * con_o.z * 2 * d.y * G = con_o.x * d.x * G
			// ddelx_dx: delx는 0 ~ W coordinate이고, x는 -1~1 범위임.
			// 따라서 두 변수간의 관계는 delx = 0.5 W * (x + 1) 
			// ddelx_dx = 0.5W 이다. (ddely_dy도 마찬가지)
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// 지금까지 계산한것을 그대로 chain rules 적용
			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}

__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
rendergradCUDA(
	const uint2* __restrict__ ranges,			// == imgState.ranges
	const uint32_t* __restrict__ point_list,	// == binningState.point_list
	int P, int W, int H,
	const float2* __restrict__ points_xy_image,	// == geomState.means2D,
	const float4* __restrict__ conic_opacity,	// == geomState.conic_opacity,
	const int* __restrict__ radii,				// == geomState.radii
	const float* __restrict__ final_Ts,			// == imgState.accum_alpha
	const uint32_t* __restrict__ n_contrib,		// == imgState.n_contrib
	float* __restrict__ dL_dmeans,		// == dL_dmeans3D	(input/output)
	float* __restrict__ out_grad_blending)
{
	auto block = cg::this_thread_block();
	auto tid = block.thread_rank();

	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	bool inside = pix.x < W && pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float grad_blending = 0.0f;

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank(); 
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;							// GS의 global index
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];			// GS의 2D mean
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];	// GS의 conic + opacity
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			contributor--;
			if (contributor >= last_contributor)
				continue;

			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;
			float dL_dmeans_x = dL_dmeans[collected_id[j] * 3];
			float dL_dmeans_y = dL_dmeans[collected_id[j] * 3 + 1];
			float dL_dmeans_z = dL_dmeans[collected_id[j] * 3 + 2];
			grad_blending += sqrtf(dL_dmeans_x * dL_dmeans_x + dL_dmeans_y * dL_dmeans_y + dL_dmeans_z * dL_dmeans_z) * dchannel_dcolor;
		}
	}

	if (inside)
	{
		out_grad_blending[pix_id] = grad_blending;
	}	
}

void BACKWARD::preprocess(
	int P, int D, int M, 	// P: num_points, D: sh_degree, M: sh channels (total color channels)
	const float3* means3D,
	// int* stable_count,
	// bool* stable_status,
	const int* radii,
	const float* shs,
	const bool* clamped,	// if false, the color of the GS should not be used
	const float4* conic_opacity,	// == geomState.conic_opacity,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,	// transpose of Tcw	
	const float* viewmatrix_next,	// transpose of Tcw
	const float* relpose,		// transpose of (Tcw_next * Tcw^-1)
	const float* projmatrix,	// transpose of Tiw
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,	// translation of Twc
	const float3* dL_dmean2D,	// (input) 
	const float3* dL_dmean2D_next,	// (input) 
	const float* dL_dconic,		// (input)
	const float* dL_dcovmap_next,		// (input)
	const float* dL_derror,
	glm::vec3* dL_dmean3D,		// (output)
	float* dL_dcolor,			// (input)
	float* dL_dcov3D,			// (output)
	float* dL_dsh,				// (output)
	glm::vec3* dL_dscale,		// (output)
	glm::vec4* dL_drot,			// (output)
	float* dL_dtau,				// (output)
	float* dL_dtau_next,		// (output)
	const bool usecolor,
    const bool useflow,
	const bool trainpose)
{
	// cudaError_t syncErr0 = cudaGetLastError();
	// cudaError_t asyncErr0 = cudaDeviceSynchronize();
	// if (syncErr0 != cudaSuccess) printf("Error0s: %s\n", cudaGetErrorString(syncErr0));
	// if (asyncErr0 != cudaSuccess) printf("Error0a: %s\n", cudaGetErrorString(asyncErr0));

	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	if (useflow && trainpose)
		computeCov2DCUDA <true, true> <<<(P + 255) / 256, 256 >>> (
			P,			// num_points
			means3D,
			// stable_status,
			radii,
			cov3Ds,
			focal_x,
			focal_y,
			tan_fovx,
			tan_fovy,
			viewmatrix,
			viewmatrix_next,
			relpose,
			dL_dconic,				// (input)
			dL_dcovmap_next,
			(float3*)dL_dmean3D,	// (output)
			dL_dcov3D,
			dL_dtau,
			dL_dtau_next);				// (output)
	else if (!useflow && trainpose)
		computeCov2DCUDA <false, true> <<<(P + 255) / 256, 256 >>> (
			P,			// num_points
			means3D,
			// stable_status,
			radii,
			cov3Ds,
			focal_x,
			focal_y,
			tan_fovx,
			tan_fovy,
			viewmatrix,
			viewmatrix_next,
			relpose,
			dL_dconic,				// (input)
			dL_dcovmap_next,
			(float3*)dL_dmean3D,	// (output)
			dL_dcov3D,
			dL_dtau,
			dL_dtau_next);				// (output)
	else if (useflow && !trainpose)
		computeCov2DCUDA <true, false> <<<(P + 255) / 256, 256 >>> (
			P,			// num_points
			means3D,
			// stable_status,
			radii,
			cov3Ds,
			focal_x,
			focal_y,
			tan_fovx,
			tan_fovy,
			viewmatrix,
			viewmatrix_next,
			relpose,
			dL_dconic,				// (input)
			dL_dcovmap_next,
			(float3*)dL_dmean3D,	// (output)
			dL_dcov3D,
			dL_dtau,
			dL_dtau_next);				// (output)
	else
		computeCov2DCUDA <false, false> <<<(P + 255) / 256, 256 >>> (
			P,			// num_points
			means3D,
			// stable_status,
			radii,
			cov3Ds,
			focal_x,
			focal_y,
			tan_fovx,
			tan_fovy,
			viewmatrix,
			viewmatrix_next,
			relpose,
			dL_dconic,				// (input)
			dL_dcovmap_next,
			(float3*)dL_dmean3D,	// (output)
			dL_dcov3D,
			dL_dtau,
			dL_dtau_next);				// (output)

	// cudaError_t syncErr1 = cudaGetLastError();
	// cudaError_t asyncErr1 = cudaDeviceSynchronize();
	// if (syncErr1 != cudaSuccess) printf("Error1s: %s\n", cudaGetErrorString(syncErr1));
	// if (asyncErr1 != cudaSuccess) printf("Error1a: %s\n", cudaGetErrorString(asyncErr1));

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	if (usecolor && useflow && trainpose)
		preprocessCUDA<true, true, true, NUM_CHANNELS> <<< (P + 255) / 256, 256 >>> (
			P, D, M,
			(float3*)means3D,
			// stable_count,
			// stable_status,
			dL_derror,
			radii,
			shs,
			clamped,
			conic_opacity,
			(glm::vec3*)scales,
			(glm::vec4*)rotations,
			scale_modifier,
			viewmatrix,
			viewmatrix_next,
			relpose,
			projmatrix,
			campos,
			(float3*)dL_dmean2D, 	// (input) 
			(float3*)dL_dmean2D_next,	// (input)
			(glm::vec3*)dL_dmean3D,	// (input from computeCov2DCUDA, output) 
			dL_dcolor,				// (input) 
			dL_dcov3D,				// (output) 
			dL_dsh,					// (output) 
			dL_dscale,				// (output) 
			dL_drot,
			dL_dtau,
			dL_dtau_next);
		// usecolor,
		// useflow,
		// trainpose,
		// updatestatus);	
	else if (usecolor && useflow && !trainpose)
		preprocessCUDA<true, true, false, NUM_CHANNELS> <<< (P + 255) / 256, 256 >>> (
			P, D, M,
			(float3*)means3D,
			// stable_count,
			// stable_status,
			dL_derror,
			radii,
			shs,
			clamped,
			conic_opacity,
			(glm::vec3*)scales,
			(glm::vec4*)rotations,
			scale_modifier,
			viewmatrix,
			viewmatrix_next,
			relpose,
			projmatrix,
			campos,
			(float3*)dL_dmean2D, 	// (input) 
			(float3*)dL_dmean2D_next,	// (input)
			(glm::vec3*)dL_dmean3D,	// (input from computeCov2DCUDA, output) 
			dL_dcolor,				// (input) 
			dL_dcov3D,				// (output) 
			dL_dsh,					// (output) 
			dL_dscale,				// (output) 
			dL_drot,
			dL_dtau,
			dL_dtau_next);
	else if (usecolor && !useflow && trainpose)
		preprocessCUDA<true, false, true, NUM_CHANNELS> <<< (P + 255) / 256, 256 >>> (
			P, D, M,
			(float3*)means3D,
			// stable_count,
			// stable_status,
			dL_derror,
			radii,
			shs,
			clamped,
			conic_opacity,
			(glm::vec3*)scales,
			(glm::vec4*)rotations,
			scale_modifier,
			viewmatrix,
			viewmatrix_next,
			relpose,
			projmatrix,
			campos,
			(float3*)dL_dmean2D, 	// (input) 
			(float3*)dL_dmean2D_next,	// (input)
			(glm::vec3*)dL_dmean3D,	// (input from computeCov2DCUDA, output) 
			dL_dcolor,				// (input) 
			dL_dcov3D,				// (output) 
			dL_dsh,					// (output) 
			dL_dscale,				// (output) 
			dL_drot,
			dL_dtau,
			dL_dtau_next);
	else if (usecolor && !useflow && !trainpose)
		preprocessCUDA<true, false, false, NUM_CHANNELS> <<< (P + 255) / 256, 256 >>> (
			P, D, M,
			(float3*)means3D,
			// stable_count,
			// stable_status,
			dL_derror,
			radii,
			shs,
			clamped,
			conic_opacity,
			(glm::vec3*)scales,
			(glm::vec4*)rotations,
			scale_modifier,
			viewmatrix,
			viewmatrix_next,
			relpose,
			projmatrix,
			campos,
			(float3*)dL_dmean2D, 	// (input) 
			(float3*)dL_dmean2D_next,	// (input)
			(glm::vec3*)dL_dmean3D,	// (input from computeCov2DCUDA, output) 
			dL_dcolor,				// (input) 
			dL_dcov3D,				// (output) 
			dL_dsh,					// (output) 
			dL_dscale,				// (output) 
			dL_drot,
			dL_dtau,
			dL_dtau_next);
	else if (!usecolor && useflow && trainpose)
		preprocessCUDA<false, true, true, NUM_CHANNELS> <<< (P + 255) / 256, 256 >>> (
			P, D, M,
			(float3*)means3D,
			// stable_count,
			// stable_status,
			dL_derror,
			radii,
			shs,
			clamped,
			conic_opacity,
			(glm::vec3*)scales,
			(glm::vec4*)rotations,
			scale_modifier,
			viewmatrix,
			viewmatrix_next,
			relpose,
			projmatrix,
			campos,
			(float3*)dL_dmean2D, 	// (input) 
			(float3*)dL_dmean2D_next,	// (input)
			(glm::vec3*)dL_dmean3D,	// (input from computeCov2DCUDA, output) 
			dL_dcolor,				// (input) 
			dL_dcov3D,				// (output) 
			dL_dsh,					// (output) 
			dL_dscale,				// (output) 
			dL_drot,
			dL_dtau,
			dL_dtau_next);
	else if (!usecolor && useflow && !trainpose)
		preprocessCUDA<false, true, false, NUM_CHANNELS> <<< (P + 255) / 256, 256 >>> (
			P, D, M,
			(float3*)means3D,
			// stable_count,
			// stable_status,
			dL_derror,
			radii,
			shs,
			clamped,
			conic_opacity,
			(glm::vec3*)scales,
			(glm::vec4*)rotations,
			scale_modifier,
			viewmatrix,
			viewmatrix_next,
			relpose,
			projmatrix,
			campos,
			(float3*)dL_dmean2D, 	// (input) 
			(float3*)dL_dmean2D_next,	// (input)
			(glm::vec3*)dL_dmean3D,	// (input from computeCov2DCUDA, output) 
			dL_dcolor,				// (input) 
			dL_dcov3D,				// (output) 
			dL_dsh,					// (output) 
			dL_dscale,				// (output) 
			dL_drot,
			dL_dtau,
			dL_dtau_next);
	
	// cudaError_t syncErr2 = cudaGetLastError();
	// cudaError_t asyncErr2 = cudaDeviceSynchronize();
	// if (syncErr2 != cudaSuccess) printf("Error2s: %s\n", cudaGetErrorString(syncErr2));
	// if (asyncErr2 != cudaSuccess) printf("Error2a: %s\n", cudaGetErrorString(asyncErr2));
}

void BACKWARD::render_grad(
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
	float* out_grad_blending
)
{
	rendergradCUDA << <grid, block >> >(
		ranges,
		point_list,
		P, W, H,
		means2D,
		conic_opacity,
		radii,
		final_Ts,
		n_contrib,
		dL_dmean3D,
		out_grad_blending
		);
}


void BACKWARD::render(
	const dim3 grid, const dim3 block,		// grid: 1 grid 당 blocks의 좌우 개수, block: 1 block 당 threads의 좌우 개수
	const uint2* ranges,					// == imgState.ranges
	const uint32_t* point_list,				// == binningState.point_list
	int P, int W, int H,							
	const float* bg_color,					// background color (0,0,0)
	const float2* means2D,					// == geomState.means2D,
	const float2* means2D_next,				// == geomState.means2D_next,
	const float4* conic_opacity,			// == geomState.conic_opacity,
	const float4* covmap_next,		// == geomState.conic_opacity_next,
	// const uint2* rect_next_min,
	// const uint2* rect_next_max,
	// const bool* stable_status,
	const int* radii,
	const float* colors,					// == geomState.rgb
	const float* final_Ts,					// == imgState.accum_alpha
	const uint32_t* n_contrib,				// == imgState.n_contrib
	const float* flowimg,
	const float* flowconf,
	const float* gsflow,
	const float* dL_dpixels,				// == dL_dpix == grad_out_color
	const float* dL_dsilh,
	const float* dL_dflowraw,					// == dL_dflowraw == grad_out_flowraw
	const float* dL_dflowcost,					// == dL_dflow == grad_out_flow
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
	const bool useflowrawgrad)
{
	// cudaError_t syncErr3 = cudaGetLastError();
	// cudaError_t asyncErr3 = cudaDeviceSynchronize();
	// if (syncErr3 != cudaSuccess) printf("Error3s: %s\n", cudaGetErrorString(syncErr3));
	// if (asyncErr3 != cudaSuccess) printf("Error3a: %s\n", cudaGetErrorString(asyncErr3));

	if (usecolor, useflow)
		renderCUDA2<true, true, NUM_CHANNELS> << <grid, block >> >(
			ranges,
			point_list,
			P, W, H,
			bg_color,
			means2D,
			means2D_next,
			conic_opacity,
			covmap_next,
			// rect_next_min,
			// rect_next_max,
			// stable_status,
			radii,
			colors,
			final_Ts,
			n_contrib,
			flowimg,
			flowconf,
			gsflow,
			dL_dpixels,
			dL_dsilh,
			dL_dflowraw,
			dL_dflowcost,
			dL_daux,
			dL_daux2,
			dL_daux3,
			// tile_loss_sum,
			// tile_loss_sigma,
			dL_dmean2D,
			dL_dmean2D_next,
			dL_dconic2D,
			dL_dcovmap_next,
			dL_dopacity,
			dL_dcolors,
			dL_derror,
			dL_derror2,
			dL_derror3,
			// usecolor,
			// useflow,
			trainpose,
			useflowrawgrad
			);
	else if (usecolor && !useflow)
		renderCUDA2<true, false, NUM_CHANNELS> << <grid, block >> >(
			ranges,
			point_list,
			P, W, H,
			bg_color,
			means2D,
			means2D_next,
			conic_opacity,
			covmap_next,
			// rect_next_min,
			// rect_next_max,
			// stable_status,
			radii,
			colors,
			final_Ts,
			n_contrib,
			flowimg,
			flowconf,
			gsflow,
			dL_dpixels,
			dL_dsilh,
			dL_dflowraw,
			dL_dflowcost,
			dL_daux,
			dL_daux2,
			dL_daux3,
			// tile_loss_sum,
			// tile_loss_sigma,
			dL_dmean2D,
			dL_dmean2D_next,
			dL_dconic2D,
			dL_dcovmap_next,
			dL_dopacity,
			dL_dcolors,
			dL_derror,
			dL_derror2,
			dL_derror3,
			// usecolor,
			// useflow,
			trainpose,
			useflowrawgrad
			);
	else if (!usecolor && useflow)
		renderCUDA2<false, true, NUM_CHANNELS> << <grid, block >> >(
			ranges,
			point_list,
			P, W, H,
			bg_color,
			means2D,
			means2D_next,
			conic_opacity,
			covmap_next,
			// rect_next_min,
			// rect_next_max,
			// stable_status,
			radii,
			colors,
			final_Ts,
			n_contrib,
			flowimg,
			flowconf,
			gsflow,
			dL_dpixels,
			dL_dsilh,
			dL_dflowraw,
			dL_dflowcost,
			dL_daux,
			dL_daux2,
			dL_daux3,
			// tile_loss_sum,
			// tile_loss_sigma,
			dL_dmean2D,
			dL_dmean2D_next,
			dL_dconic2D,
			dL_dcovmap_next,
			dL_dopacity,
			dL_dcolors,
			dL_derror,
			dL_derror2,
			dL_derror3,
			// usecolor,
			// useflow,
			trainpose,
			useflowrawgrad
			);
	else
		renderCUDA2<false, false, NUM_CHANNELS> << <grid, block >> >(
			ranges,
			point_list,
			P, W, H,
			bg_color,
			means2D,
			means2D_next,
			conic_opacity,
			covmap_next,
			// rect_next_min,
			// rect_next_max,
			// stable_status,
			radii,
			colors,
			final_Ts,
			n_contrib,
			flowimg,
			flowconf,
			gsflow,
			dL_dpixels,
			dL_dsilh,
			dL_dflowraw,
			dL_dflowcost,
			dL_daux,
			dL_daux2,
			dL_daux3,
			// tile_loss_sum,
			// tile_loss_sigma,
			dL_dmean2D,
			dL_dmean2D_next,
			dL_dconic2D,
			dL_dcovmap_next,
			dL_dopacity,
			dL_dcolors,
			dL_derror,
			dL_derror2,
			dL_derror3,
			// usecolor,
			// useflow,
			trainpose,
			useflowrawgrad
			);

	// cudaError_t syncErr4 = cudaGetLastError();
	// cudaError_t asyncErr4 = cudaDeviceSynchronize();
	// if (syncErr4 != cudaSuccess) printf("Error4s: %s\n", cudaGetErrorString(syncErr4));
	// if (asyncErr4 != cudaSuccess) printf("Error4a: %s\n", cudaGetErrorString(asyncErr4));
}