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

#include <math.h>
#include <torch/torch.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "rasterize_points.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

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
	const bool updateconf)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0); // num_points
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor n_touched = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor n_found = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_silh = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_gsflow = torch::full({2, H, W}, 0.0, float_opts);
  torch::Tensor out_flowcost = torch::full({1, H, W}, 0.0, float_opts);

  // 2nd order
  torch::Tensor weights_sum = torch::full({P}, 0.0, float_opts);
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));

  // function of buffer. size of (torch::kByte, 1byte) * N will be created
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

      // Move to CudaRasterizer::Rasterizer::forward() of [rasterizer_impl.cu]
	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data_ptr<float>(),
		W, H,
		means3D.contiguous().data_ptr<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data_ptr<float>(), 
		opacity.contiguous().data_ptr<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data_ptr<float>(), 
		viewmatrix.contiguous().data_ptr<float>(), 
		viewmatrix_next.contiguous().data_ptr<float>(),
		projmatrix.contiguous().data_ptr<float>(),
		campos.contiguous().data_ptr<float>(),
		tan_fovx,
		tan_fovy,
        flowimg.contiguous().data_ptr<float>(),
        flowconf.contiguous().data_ptr<float>(),
		prefiltered,
		usecolor,
        useflow,
		trainpose,
		updateconf,
		out_color.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<int>(),
		n_touched.contiguous().data<int>(),
		n_found.contiguous().data<int>(),
		weights_sum.contiguous().data_ptr<float>(),
		// rect_next_min.contiguous().data_ptr<int>(),
		// rect_next_max.contiguous().data_ptr<int>(),
        out_depth.contiguous().data_ptr<float>(),
        out_silh.contiguous().data_ptr<float>(),
		out_gsflow.contiguous().data_ptr<float>(),
		out_flowcost.contiguous().data_ptr<float>());
  }

  // rendered image /
  // this returns: num_rendered / rendered_image(color) / radii of each GS / geomBuffer / binningBuffer / imgBuffer
  return std::make_tuple(rendered, out_color, out_depth, out_silh, out_gsflow, out_flowcost, radii, geomBuffer, binningBuffer, imgBuffer, n_touched, n_found, weights_sum);
}

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
// 	const torch::Tensor& flowimg, // (2, H, W)
// 	const torch::Tensor& flowconf, // (1, H, W) => log-logistic, (2, H, W) => L1, mahalanobis
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
//     const bool useflow)
// {

// 	const int P = means3D.size(0);		// number of GS points
// 	const int H = flowimg.size(1);
// 	const int W = flowimg.size(2);

// 	torch::Tensor J_flow_means3D = torch::zeros({P, 2, 3}, means3D.options());	// dim: 2x3
// 	torch::Tensor J_flow_means2D = torch::zeros({P, 2, 2}, means3D.options());	// dim: 2x2
// 	torch::Tensor J_flow_means2D_next = torch::zeros({P, 2, 2}, means3D.options());	// dim: 2x2

// 	torch::Tensor J_flow_conic2D =  torch::zeros({P, 2, 4}, means3D.options());	// dim: 2x4 (xyzw)
// 	torch::Tensor J_flow_covmap_next = torch::zeros({P, 2, 4}, means3D.options());	// dim: 2x4 (xyzw)
// 	torch::Tensor J_flow_opacity = torch::zeros({P, 2, 1}, means3D.options());	// dim: 2x1
// 	torch::Tensor J_flow_cov3D = torch::zeros({P, 2, 6}, means3D.options());	// dim: 2x6
// 	torch::Tensor J_flow_scales = torch::zeros({P, 2, 3}, means3D.options());	// dim: 2x3
// 	torch::Tensor J_flow_rotations = torch::zeros({P, 2, 4}, means3D.options());	// dim: 2x4

// 	torch::Tensor J_flow_pix_position = torch::zeros({H, W, 2, 3}, means3D.options());	// dim: HxWx2x3
// 	torch::Tensor J_flow_pix_scales = torch::zeros({H, W, 2, 3}, means3D.options());	// dim: HxWx2x3

// }

// Output: dL_dmeans3D, dL_dmeans2D, dL_dsh, dL_dcolors, 
// dL_dopacity, dL_dcov3D, dL_dscales, dL_drotations
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, // 5
torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,  // 5 
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
    const torch::Tensor& dL_dout_color, // == grad_out_color
	const torch::Tensor& dL_dout_silh, // == grad_out_silh
	const torch::Tensor& dL_dout_flowraw, // == grad_out_flowraw
	const torch::Tensor& dL_dout_flow, // == grad_out_flow
	const torch::Tensor& dL_dout_aux, // == grad_out_aux
	const torch::Tensor& dL_dout_aux2, // == grad_out_aux
	const torch::Tensor& dL_dout_aux3, // == grad_out_aux
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
	const bool useflowrawgrad)
{
  const int P = means3D.size(0);		// number of GS points
  const int H = dL_dout_color.size(1);	// grad_out_color는 rendered_image torch에 대한 pixel별 gradient임.
  const int W = dL_dout_color.size(2);	
  
  int M = 0;
  if(sh.size(0) != 0)	// shs = cat(features_dc_, features_rest_) 이므로 M = channels of shs
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());	// dim: 3 (x,y,z)
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());	// dim: 3 (x,y,1)
  torch::Tensor dL_dmeans2D_next = torch::zeros({P, 3}, means3D.options());	// dim: 3 (x,y,1)

  torch::Tensor dL_derror = torch::zeros({P, 1}, means3D.options());	// dim: 1
  torch::Tensor dL_derror2 = torch::zeros({P, 1}, means3D.options());	// dim: 1
  torch::Tensor dL_derror3 = torch::zeros({P, 1}, means3D.options());	// dim: 1

  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());	// dim: 3 (r,g,b)
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());	// dim: 2x2				
  torch::Tensor dL_dcovmap_next = torch::zeros({P, 2, 2}, means3D.options());	// dim: 2x2
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());	// dim: 1
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());	// dim: 6
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());	// dim: Mx3 (r,g,b)
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());	// dim: 3 (x,y,z)
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options()); 	// dim: 4 (x,y,z,w)
  torch::Tensor dL_dtau = torch::zeros({P, 6}, means3D.options());		// dim: 6 (rho, theta)
  torch::Tensor dL_dtau_next = torch::zeros({P, 6}, means3D.options());		// dim: 6 (rho_next, theta_next)
  
  torch::Tensor relpose = torch::zeros({4, 4}, means3D.options());	// dim: 4x4
/// DEPRECATED
  // Current pose 기준으로 구하는 경우
  // if relpose ==  Twc^T * Tcw_next^T = (Tcw_next * Twc)^T = Tcnext_c^T
//   if (usecolor)
// 	relpose = torch::matmul(at::linalg_inv(viewmatrix), viewmatrix_next);
//   // Next pose 기준으로 구하는 경우
//   // if relpose ==  Twc_next^T * Tcw^T = (Tcw * Twc_next)^T = (Tc_cnext)^T
//   else
//   	relpose = torch::matmul(at::linalg_inv(viewmatrix_next), viewmatrix);

  // 2nd-order test code
//   torch::Tensor out_grad_blending = torch::full({1, H, W}, 0.0, means3D.options());

  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data_ptr<float>(),
	  W, H, 
	  means3D.contiguous().data_ptr<float>(),
	//   stable_count.contiguous().data_ptr<int>(),
	//   stable_status.contiguous().data_ptr<bool>(),
	  sh.contiguous().data_ptr<float>(),
	  colors.contiguous().data_ptr<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data_ptr<float>(),
	  viewmatrix.contiguous().data_ptr<float>(),
	  viewmatrix_next.contiguous().data_ptr<float>(),
	  relpose.contiguous().data_ptr<float>(),
	  projmatrix.contiguous().data_ptr<float>(),
	  campos.contiguous().data_ptr<float>(),
	  flowimg.contiguous().data_ptr<float>(),
	  flowconf.contiguous().data_ptr<float>(),
	  gsflow.contiguous().data_ptr<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data_ptr<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data_ptr<float>(),
	  dL_dout_silh.contiguous().data_ptr<float>(),
	  dL_dout_flowraw.contiguous().data_ptr<float>(),
	  dL_dout_flow.contiguous().data_ptr<float>(),
	  dL_dout_aux.contiguous().data_ptr<float>(),
	  dL_dout_aux2.contiguous().data_ptr<float>(),
	  dL_dout_aux3.contiguous().data_ptr<float>(),
	  dL_derror.contiguous().data_ptr<float>(),
	  dL_derror2.contiguous().data_ptr<float>(),
	  dL_derror3.contiguous().data_ptr<float>(),
	  dL_dmeans2D.contiguous().data_ptr<float>(),
	  dL_dmeans2D_next.contiguous().data_ptr<float>(),
	  dL_dconic.contiguous().data_ptr<float>(),  
	  dL_dcovmap_next.contiguous().data_ptr<float>(),  
	  dL_dopacity.contiguous().data_ptr<float>(),
	  dL_dcolors.contiguous().data_ptr<float>(),
	  dL_dmeans3D.contiguous().data_ptr<float>(),
	  dL_dcov3D.contiguous().data_ptr<float>(),
	  dL_dsh.contiguous().data_ptr<float>(),
	  dL_dscales.contiguous().data_ptr<float>(),
	  dL_drotations.contiguous().data_ptr<float>(),
	  dL_dtau.contiguous().data_ptr<float>(),
	  dL_dtau_next.contiguous().data_ptr<float>(),
	//   out_grad_blending.contiguous().data_ptr<float>(), // 2nd-order test
	  usecolor,
	  useflow,
	  trainpose,
	  useflowrawgrad);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, 
	dL_dsh, dL_dscales, dL_drotations, dL_dtau, dL_dtau_next, 
	dL_derror, dL_derror2, dL_derror3);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data_ptr<float>(),
		viewmatrix.contiguous().data_ptr<float>(),
		projmatrix.contiguous().data_ptr<float>(),
		present.contiguous().data_ptr<bool>());
  }
  
  return present;
}


