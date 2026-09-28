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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

__global__ void printKernel(const float* d_out_sum, const float* d_out_sigma, int length)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < length) {
        printf("d_out_sum[%d] = %f \n", idx, d_out_sum[idx]);
		printf("d_out_sigma[%d] = %f \n", idx, d_out_sigma[idx]);
    }
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	// auto idx = cg::this_grid().thread_rank();
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).

/******************** Original code ********************/
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,					// == geomState.depths
	const uint32_t* offsets,  				// == geomState.point_offsets)
	uint64_t* gaussian_keys_unsorted,		// == binningState.point_list_keys_unsorted
	uint32_t* gaussian_values_unsorted,		// == binningState.point_list_unsorted
	int* radii,
	dim3 grid)
{
	// auto idx = cg::this_grid().thread_rank();
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		// "offsets" stores the number of tiles touched by each Gaussian.
		// so "off" can represent the all of tiles touched by total gaussians
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		// Get the min/max range of tiles
		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 

		// for each tiles touched by a gaussian,
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x; // get block index (tile ID)
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;  // this final "keys" can be used to sort by depth in the same tile.
				gaussian_values_unsorted[off] = idx;  // save the gaussian ID for the tile
				// each "off" represents the tile_idx corresponding to the key<<=32
				off++;
			}
		}
	}
}
/*******************************************************/

/******************** Speedy-splat *********************/
// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
// __global__ void duplicateWithKeys(
// 	int P,
// 	const float2* points_xy,
// 	const float* depths,
// 	const uint32_t* offsets,
// 	uint64_t* gaussian_keys_unsorted,
// 	uint32_t* gaussian_values_unsorted,
// 	float4* con_o,
//   uint32_t* tiles_touched,
// 	dim3 grid)
// {
// 	int idx = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (idx >= P)
// 		return;
		
// 	// Generate no key/value pair for invisible Gaussians
// 	if (tiles_touched[idx] > 0)
// 	{
// 		// Find this Gaussian's offset in buffer for writing keys/values.
// 		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
// 		// Update unsorted arrays with Gaussian idx for every tile that
// 		// Gaussian touches
// 		duplicateToTilesTouched(
// 			points_xy[idx], con_o[idx], grid,
// 			idx, off, depths[idx],
// 			gaussian_keys_unsorted,
// 			gaussian_values_unsorted);
// 	}
// }
/*******************************************************/

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	// auto idx = cg::this_grid().thread_rank();
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		// if tile ID is different, set this GS as last GS of previous tile
		// and first GS of current tile
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

__global__ void tileReductionKernelForLoss(const float* __restrict__ dL_daux, 
											float* __restrict__ d_out_sum,
											float* __restrict__ d_out_sigma,
											int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ float sdata[BLOCK_X * BLOCK_Y];
	__shared__ float sdata_sq[BLOCK_X * BLOCK_Y];

	int localId = threadIdx.y * blockDim.x + threadIdx.x; // pixel index in a block

	float val = 0.0f;
    if (x < width && y < height) {
        int idx = y * width + x;  
        val = dL_daux[idx];
    }

	// if (val > 1.0f) {
	// 	printf("Out-of-range val: %f, x=%d/%d, y=%d/%d\n", val, x, width, y, height);
	// }

	sdata[localId] = val;
	sdata_sq[localId] = val * val;
    __syncthreads();

	// parallel reduction
	// stride => 128 => 64 => 32 => 16 => 8 => ...
	for (int stride = (BLOCK_X * BLOCK_Y) / 2; stride > 0; stride >>= 1) {
        if (localId < stride) {
            sdata[localId] += sdata[localId + stride];
			sdata_sq[localId] += sdata_sq[localId + stride];
        }
        __syncthreads();
    }

	if (localId == 0) {
		int count = BLOCK_X * BLOCK_Y;
		float sum = sdata[0];
		float mean = sum / count;
		float sum_sq = sdata_sq[0];
        float variance = sum_sq / count - mean * mean;
        float sigma = sqrtf(variance);

        int blockIdx1D = blockIdx.y * gridDim.x + blockIdx.x;
        // d_out_sum[blockIdx1D] = sdata[0];
		d_out_sum[blockIdx1D] = sum;
		d_out_sigma[blockIdx1D] = sigma;
    }
}


// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	// Obtian sizeof(geom.depths) aligned by 128bytes. 
	// geom.depths: ptr to access geo_depth aligend address by 128bytes.
	// chunk: pointer position after obtained
	obtain(chunk, geom.depths, P, 128);          
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	// obtain(chunk, geom.rect_next_min, P, 128);
	// obtain(chunk, geom.rect_next_max, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.means2D_next, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.covmap_next, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	// cub::DeviceScan::InclusiveSum: if nullptr, **only calculate required size** to geom.scan_size (no work done)
	// geom.scan_size: needed size is automatically calculated
	// geom.tiles_touched (input) : where the function want to caluclate accumulated sum
	// geom.tiles_touched (output): where accumulated sum will be saved
	// P: number of data to proceed 
	// *****Conclusion: P * size of type(geom.tiles_touched) will be calculated to geom.scan_size(bytes)****
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);

	// geom.scanning_space-> char* => so memory of geom.scan_size is allocated aligned by 128 bytes.
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M, 		// num_points, sh_degree, sh_channels (sh.size(1))
	const float* background,     	// background color; black(0,0,0)
	const int width, int height, 	
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,		// transpose of Tcw
	const float* viewmatrix_next,		// transpose of Tcw
	const float* projmatrix,		// transpose of Tiw	
	const float* cam_pos,			// translation of Twc
	const float tan_fovx, float tan_fovy,  // (W/2) / fx, (H/2) / fy
    const float* flowimg,
    float* flowconf,
	const bool prefiltered,
	const bool usecolor,
    const bool useflow,
	const bool trainpose,
	const bool updateconf,
	float* out_color,				// (output) rendered image
    int* radii,                     // (output) radii of each GS
	int* n_touched,
	int* n_found,
	float* weights_sum,
    float* out_depth,
    float* out_silh,
	float* out_gsflow,
	float* out_flowcost)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P); // get required bytes for num_points 
	char* chunkptr = geometryBuffer(chunk_size);    // allocated memory for GeometryState
	// allocated each memory chunk to each elements of GeometryState
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);   

	if (radii == nullptr) // it is not nullptr!! because created in RasterizeGaussiansCUDA() function.
	{
		radii = geomState.internal_radii;
	}

	// tile_grid: number of blocks in 2D
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

	// block: number of threads in a block
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	// allocated memory for each pixel of accum_alpha(N), n_contrib(N), ranges(N)
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)

	// The following variables are calculated in FORWARD:preprocess():
	// depths[idx] = p_view.z; // z value of mean3D in camera coordinate
	// radii[idx] = my_radius; // 3 * max eigenvalue for each GS
	// points_xy_image[idx] = point_image; // 2D pixel coordinate of each GS
	// // Inverse 2D covariance and opacity neatly pack into one float4
	// conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	// tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x); // area of a sqaure representing range

	// preprocessCUDA()함수로 바로 연결됨. 
	// ** 각 GS가 thread마다 주어짐. **
	// 1. GS들이 in_frustum인지 확인
	// 2. GS들의 mean3D와 cov3D를 현재 이미지의 좌표계에서인 p_view로 변경. cov2D도 계산함.
	// 3. 이미지 좌표계가 ndc coordinate (-1~1)인데 이걸 pixel coordinate (0~H,W)로 바꿈.
	// 4. depths[idx]에 p_view.z저장, radii[idx]에 my_radius (2D covariance의 3*eigenvalue)
	// 5. points_xy_image[idx]에 point_image 저장 (pixel coordinate에서의 2D x,y지점)
	// 6. conic_opacity[idx] (2D에서 conic과 해당 GS의 opacity저장)
	// 7. tiles_touched[idx]에는 이 GS가 속하는 tile의 넓이(개수) 저장
	FORWARD::preprocess(
		P, D, M,  // num_points, sh_degree, sh_channels (sh.size(1))
		means3D,   
		(glm::vec3*)scales,
		scale_modifier,  // 1.0
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, viewmatrix_next, projmatrix,    // transpose of Tcw, transpose of Tiw	
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		// rect_next_min,
		// rect_next_max,
		// geomState.rect_next_min,
		// geomState.rect_next_max,
		geomState.means2D,
		geomState.means2D_next,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		geomState.covmap_next,
		tile_grid,                  // grid for multi-threading of CUDA
		geomState.tiles_touched,
		usecolor,
		useflow,
		prefiltered
	);

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]

	// geomState.point_offsets: accumulate the block(tile) nums 
	// 예를들어 GS 5개가 2, 3, 0, 2, 1개의 block에 걸쳐있다면, 2, 5, 5, 7, 8로 point_offsets에 저장함.
	cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size,
		geomState.tiles_touched, geomState.point_offsets, P);

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	// num_rendered: output
	int num_rendered;
	// num_rendered: represents total "tiles" covered by all gaussian splatting (same as total GS numbers)
	// geomState.point_offsets + P - 1주소 (마지막 주소)에 있는 값이 총 render에 사용될 tile 개수(num_rendered)임.
	cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted

	// Each image is split in to 16x16 tiles
	// point_list_keys_unsorted와 point_list_unsorted에 각 GS가 가진 point_offsets을 인덱스로 하여
	// tile index(y*grid.x + x)와 GS의 index를 저장함.
	// 예를들어 위와 같은 2,5,5,7,8 예시에서 세번째 GS인 경우 이 GS는 2개의 for문을 돌게 되고
	// point_list_keys_unsorted[5] = tile index | gs의 depth (mean3D의)
	// point_list_unsorted[5] = gs index
	// point_list_keys_unsorted[6] = tile index | gs의 depth (mean3D의)
	// point_list_unsorted[6] = gs index
	// 이런식으로 저장된다.

	/******************** Original code ********************/
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid
		);
	/*******************************************************/

	/******************** Speedy-splat *********************/
	// duplicateWithKeys << <(P + 255) / 256, 256 >> > (
	// 	P,
	// 	geomState.means2D,
	// 	geomState.depths,
	// 	geomState.point_offsets,
	// 	binningState.point_list_keys_unsorted,
	// 	binningState.point_list_unsorted,
	// 	geomState.conic_opacity,
    // 	geomState.tiles_touched,
	// 	tile_grid);
	
	/*******************************************************/

	// find MSB of the input
	// if tile_grid.x * tile_grid.y = 256
	// bit becomes 9
	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	// ***Sorted by tile ID firstly, sorted by depth secondly.***
	// output(binningState.point_list) will have sorted gaussian splatting idx
	// 0 ~ 32+bit will be used for sorting
	// 0 ~ 32 : depths (낮은 values)
	// 33 ~ 32 + bit: for tile IDs (높은 values)
	// tile기준으로 sorting => 그다음에 depth기준으로 sorting된다.
	// 기준에 따라서 point_list_keys에는 해당 key들이 들어있게 되고, point_list에는 해당 GS index들이 들어있게 된다.
	cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size, // size of list_sorting_space
		binningState.point_list_keys_unsorted, binningState.point_list_keys, 	// (input)keys unsorted => (out)keys sorted
		binningState.point_list_unsorted, binningState.point_list,				// (input)values unsorted => (out) values sorted by keys
		num_rendered, 0, 32 + bit);

	// uint2: uint32_t * 2
	// set 0 as range value(start/end) for each tile
	cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2));

	// Identify start and end of per-tile workloads in sorted list
	// imgState.ranges: closest index / farthest index of binningState.point_list_keys in each tile
	// point_list_keys에는 tile별 정렬 => depth별 정렬 순으로 들어있음.
	// GPU의 각 thread마다 point_list_keys[thread_idx -1] >> 32과 point_list_keys[thread_idx] >> 32를 비교하여
	// tile index가 바뀌는 지점을 체크한다. 바뀌는 지점에서 ranges[prev tile id].y = thread idx, ranges[cur tile id].x = thread idx
	// 이렇게 지정하면 ranges.y에는 가장 depth가 먼 GS의 index+1가 range.x에는 가장 depth가 가까운 GS의 index가 저장된다.
	// 즉 imgState.ranges는 각 tile마다 가장 가까운 GS의 index, 먼 GS의 index가 저장되어있다.
	// **주의** 여기서 말하는 GS의 index는 단순히 thread순서의 index이므로, depth와 호환되는 GS 고유의 index가 아니다.
	// 즉 range.y - range.x에는 GS가 해당 tile에 몇개들어있는지를 나타내는 것이고, 실제 GS고유의 index가 아니다.
	// 실제 고유의 index를 얻으려면 point_list를 사용해야한다.
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges
			);

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	FORWARD::render(
		tile_grid, block, // total blocks / threads per block
		imgState.ranges,  // 
		binningState.point_list,
		width, height,
		geomState.means2D,
		geomState.means2D_next,
		geomState.depths,
		feature_ptr,
		geomState.conic_opacity,
		geomState.covmap_next,
		// geomState.rect_next_min,
		// geomState.rect_next_max,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		flowimg,
    	flowconf,
		usecolor,
		useflow,
		trainpose,
		updateconf,
		out_color,
		out_depth,
		out_silh,
		out_gsflow,
		out_flowcost,
		n_touched,
		n_found,
		weights_sum);

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,	// P: num_points, D: sh_degree, M: sh channels (total color channels), R: num_tiles (total touched "tiles" covered by all gaussian splatting)
	const float* background,			// background image (0,0,0) 
	const int width, int height,		
	const float* means3D,
	// int* stable_count,
	// bool* stable_status,
	const float* shs,
	const float* colors_precomp,
	const float* scales,			 	// scales of each GS
	const float scale_modifier,
	const float* rotations,				// rotations(quaternions) of each GS
	const float* cov3D_precomp,
	const float* viewmatrix,			// transpose of Tcw
	const float* viewmatrix_next,		// transpose of Tcw
	const float* relpose,				// transpose of Tcnext_ccurr
	const float* projmatrix,			// transpose of Tic
	const float* campos,				// translation of Twc
	const float* flowimg,
	const float* flowconf,
	const float* gsflow,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer, 					// GeometryState
	char* binning_buffer,				// BinningState
	char* img_buffer,					// ImageState
	const float* dL_dpix,			    // == grad_out_color (dL_dout_color, WxH에서 pixel별 color비교에 대한 gradient)
	const float* dL_dsilh,
	const float* dL_dflowraw,			    // == grad_out_flowraw
	const float* dL_dflow,			    // == grad_out_flow
	const float* dL_daux,				// == grad_out_aux
	const float* dL_daux2,				// == grad_out_aux2
	const float* dL_daux3,				// == grad_out_aux3
	float* dL_derror,				// (output) gradient of error per GS (== E_k)
	float* dL_derror2,
	float* dL_derror3,
	float* dL_dmean2D,					// (output) gradient of 2D mean position
	float* dL_dmean2D_next,				// (output) gradient of 2D mean position
	float* dL_dconic,					// (output) gradient of conic matrix
	float* dL_dcovmap_next,					// (output) gradient of conic matrix
	float* dL_dopacity,					// (output) gradient of opacity
	float* dL_dcolor,					// (output) gradient of colors of Gaussian Splatting
	float* dL_dmean3D,					// (output) gradient of 3D mean position
	float* dL_dcov3D,					// (output) gradient of 3D covariance
	float* dL_dsh,						// (output) gradient of shs
	float* dL_dscale,					// (output) gradient of scales
	float* dL_drot,						// (output) gradient of rotations
	float* dL_dtau,
	float* dL_dtau_next,
	// float* out_grad_blending,
	const bool usecolor,
    const bool useflow,
	const bool trainpose,
	const bool useflowrawgrad)
{

	/// Load saved data (forward에서 저장해둔 데이터 불러옴) ///

	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	// tile_grid: 이미지 전체를 16(BLOCK_X) x 16(BLOCK_Y)의 사이즈를 가지는 타일들로 나눔. tile_grid는 이 타일들의 index를 나타냄.
	// 즉, tile 1개(== block)는 16x16 pixel을 담당함.
	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// cudaEvent_t start0, stop0;
	// cudaEventCreate(&start0);
	// cudaEventCreate(&stop0);
	// cudaEventRecord(start0, 0);
	
	// int numBlocks = tile_grid.x * tile_grid.y;
	// float* tile_loss_sum = nullptr;
	// float* tile_loss_sigma = nullptr;
    // cudaMalloc((void**)&tile_loss_sum, sizeof(float) * numBlocks);
	// cudaMalloc((void**)&tile_loss_sigma, sizeof(float) * numBlocks);
	// tileReductionKernelForLoss<<<tile_grid, block>>>(dL_daux, tile_loss_sum, tile_loss_sigma, width, height);

	// cudaEventRecord(stop0, 0);
	// cudaEventSynchronize(stop0);
	// float elapsedTime0;
	// cudaEventElapsedTime(&elapsedTime0, start0, stop0);
	// printf("Reduction Kernel Execution Time: %.3f ms\n", elapsedTime0);

	// int threadsPerBlock = 256; 
	// int blocksPerGrid   = (numBlocks + threadsPerBlock - 1) / threadsPerBlock;
	// printKernel<<< blocksPerGrid, threadsPerBlock>>>(tile_loss_sum, tile_loss_sigma, numBlocks);
	// cudaDeviceSynchronize();

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	// forward와 다르게 render를 먼저함.
	// BACKWARD::render(renderCUDA)에서는 각 pixel이 각 thread가 되며, 이 pixel을 touch한 GS들을 전부 훑어본다.
	// 따라서, alpha값으로 인해 stable하더라도 해당 pixel을 touch한 경우 계산에 참여해야하며,
	// stable한 경우 gradient계산만 하지 않으면 되므로, continue하는 것이 최선임.
	// 추가한다면, 해당 block의 loss_pixel의 합 (dL_daux의 합)이 너무 작으면, 해당 block은 id의 경우 그냥 건너 뛰는 것도 좋을듯

	// cudaEvent_t start1, stop1;
	// cudaEventCreate(&start1);
	// cudaEventCreate(&stop1);
	// cudaEventRecord(start1, 0);

	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		P, width, height,
		background,
		geomState.means2D,
		geomState.means2D_next,
		geomState.conic_opacity,
		geomState.covmap_next,
		// geomState.rect_next_min,
		// geomState.rect_next_max,
		// stable_status,
		radii,
		color_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		flowimg,
		flowconf,
		gsflow,
		dL_dpix,
		dL_dsilh,
		dL_dflowraw,
		dL_dflow,
		dL_daux,
		dL_daux2,
		dL_daux3,
		// tile_loss_sum,
		// tile_loss_sigma,
		(float3*)dL_dmean2D,
		(float3*)dL_dmean2D_next,
		(float4*)dL_dconic,
		(float4*)dL_dcovmap_next,
		dL_dopacity,
		dL_dcolor,
		dL_derror,
		dL_derror2,
		dL_derror3,
		usecolor,
		useflow,
		trainpose,
		useflowrawgrad);

	// cudaEventRecord(stop1, 0);
	// cudaEventSynchronize(stop1);
	// float elapsedTime1;
	// cudaEventElapsedTime(&elapsedTime1, start1, stop1);
	// printf("First Kernel Execution Time: %.3f ms\n", elapsedTime1);

	// BACKWARD::preprocess에서는 256개의 GS마다 P / 256만큼 계산을 진행한다.
	// 물론 기존 코드에서도 radii[idx]가 대부분 0이 아니기 때문에 건너띄어지겠지만, stable한것도 계산에서 제외하고자 한다.

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	// BACKWARD::render에서 계산한 dL_dmean2D, dL_dconic, dL_dcolor를 이용한다.

	// cudaEvent_t start2, stop2;
	// cudaEventCreate(&start2);
	// cudaEventCreate(&stop2);
	// cudaEventRecord(start2, 0);

	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		// stable_count,
		// stable_status,
		radii,
		shs,
		geomState.clamped,
		geomState.conic_opacity,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix, viewmatrix_next, relpose,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		(float3*)dL_dmean2D_next,
		dL_dconic,
		dL_dcovmap_next,
		dL_derror,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		dL_dtau,
		dL_dtau_next,
		usecolor,
		useflow,
		trainpose);

	// BACKWARD::render_grad(
	// 	tile_grid,
	// 	block,
	// 	imgState.ranges,
	// 	binningState.point_list,
	// 	geomState.means2D,
	// 	P, width, height,
	// 	geomState.conic_opacity,
	// 	radii,
	// 	imgState.accum_alpha,
	// 	imgState.n_contrib,
	// 	dL_dmean3D,
	// 	out_grad_blending);

	// cudaEventRecord(stop2, 0);
	// cudaEventSynchronize(stop2);
	// float elapsedTime2;
	// cudaEventElapsedTime(&elapsedTime2, start2, stop2);
	// printf("Second Kernel Execution Time: %.3f ms\n", elapsedTime2);
}
