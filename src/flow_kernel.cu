#include <cuda_runtime.h>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <torch/extension.h>

#define gpuErrchk  { cudaError_t code = cudaGetLastError(); if (code!= cudaSuccess) {printf("GPUassert : %s\n%s at line %d\n", cudaGetErrorString(code), __FILE__, __LINE__); return code;} }


__global__ void estimate_depthpoints_kernel(const float* flow_x, const float* flow_y, float* depth, 
    const float* b, const float* KRKinv,
    int rows, int cols,
    float min_depth, float max_depth)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols)
    {
        int x = idx % cols;
        int y = idx / cols;

        float delta_x = flow_x[idx];
        float delta_y = flow_y[idx];

        float P[3] = {x * KRKinv[0] + y * KRKinv[1] + KRKinv[2],
        x * KRKinv[3] + y * KRKinv[4] + KRKinv[5],
        x * KRKinv[6] + y * KRKinv[7] + KRKinv[8]};

        float w1 = P[0], w2 = P[1], w3 = P[2];
        float a1 = x + delta_x, a2 = y + delta_y;

        float z_nume = (a1 * b[2] - b[0]) * (w1 - a1 * w3) + (a2 * b[2] - b[1]) * (w2 - a2 * w3);
        float z_deno = (w1 - a1 * w3) * (w1 - a1 * w3) + (w2 - a2 * w3) * (w2 - a2 * w3);
        float d = fminf(fmaxf(fabsf(z_nume / z_deno), min_depth), max_depth);
        depth[idx] = d;
    }
}

void estimate_flow_depth_cuda(const torch::Tensor flow, const torch::Tensor K,
                                const torch::Tensor Rc2c1, const torch::Tensor tc2c1,
                                torch::Tensor depth,
                                float min_depth, float max_depth)
{
    TORCH_CHECK(flow.device() == K.device() && K.device() == Rc2c1.device() && Rc2c1.device() == tc2c1.device(),
    "All tensors must be on the same device");

    int rows = flow.size(0);
    int cols = flow.size(1);

    // Convert camera calibration matrix K, Rc2c1, tc2c1 to float
    torch::Tensor b = torch::matmul(K, tc2c1);
    torch::Tensor KRKinv = torch::matmul(K, Rc2c1);
    KRKinv = torch::matmul(KRKinv, K.inverse());

    torch::Tensor flow_x = flow.slice(2, 0, 1);
    torch::Tensor flow_y = flow.slice(2, 1, 2);

    // Launch the CUDA kernel (assuming this kernel is written for CUDA)
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (rows * cols + threadsPerBlock - 1) / threadsPerBlock;

    // Use the kernel to estimate depth points (passing flattened arrays)
    estimate_depthpoints_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        flow_x.contiguous().data_ptr<float>(),
        flow_y.contiguous().data_ptr<float>(),
        depth.contiguous().data_ptr<float>(),
        b.contiguous().data_ptr<float>(),
        KRKinv.contiguous().data_ptr<float>(),
        rows, cols, min_depth, max_depth
    );

    // Synchronize the device
    cudaDeviceSynchronize();
}


// int launch_estimate_depthpoints_kernel(const float* flow_x, const float* flow_y,
//     float* depth, float3* depth_points,
//     const float* b, const float* KRKinv,
//     int rows, int cols,
//     float min_depth, float max_depth)
// {
//     // 16x16 threads per block
//     dim3 threadsPerBlock(16, 16);
//     // how many blocks in x-axis ((cols + remaining) / threadsPerBlock.x), how may blocks in y-axis
//     // **Ceiling** will be applied
//     dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
//     (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

//     // Perform kerenl for <blocks per grid, threads per block>
//     estimate_depthpoints_kernel
//     <<<blocksPerGrid, threadsPerBlock>>>(flow_x, flow_y, depth, depth_points, b, KRKinv, 
//     rows, cols, min_depth, max_depth);

//     // NOTE: after gpuErrchk, *** cudaDeviceSynchronize() should be specified ***
//     gpuErrchk;
//     cudaDeviceSynchronize();
//     return cudaSuccess;
// }


