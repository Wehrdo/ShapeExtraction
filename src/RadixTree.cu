#include "RadixTree.hpp"
#include "CudaCommon.cuh"
#include "libmorton/include/morton.h"
#include "cub/device/device_reduce.cuh"
#include "cub/device/device_radix_sort.cuh"

#include <array>
#include <algorithm>
#include <limits>

using namespace RT;
using cub::DeviceReduce;
using cub::DeviceRadixSort;

template <typename T>
__global__ void makeCodes(const T minCoord,
                          const T maxCoord,
                          const T* __restrict__ x_vals,
                          const T* __restrict__ y_vals,
                          const T* __restrict__ z_vals,
                          Node* nodes,
                          const size_t N) {
    // only supports 1-dimension blocks and grids
    assert(threadIdx.y == threadIdx.z == 1);
    assert(blockIdx.y == blockIdx.z == 1);

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        T range = (maxCoord - minCoord);
        // We can only encode 21 bits (21 <bits> * 3 <dimensions> = 63 <bits>)
        const uint32_t bitscale = 0xFFFFFFFFu >> (32 - 21);
        uint32_t x_coord = bitscale * ((x_vals[idx] - minCoord) / range);
        uint32_t y_coord = bitscale * ((y_vals[idx] - minCoord) / range);
        uint32_t z_coord = bitscale * ((z_vals[idx] - minCoord) / range);
        nodes[idx].mortonCode = morton3D_64_encode(x_coord, y_coord, z_coord);
        // if (idx == 0) {
        //     // printf("min = %f, max = %f\n", minCoord, maxCoord);
        //     printf("%u, %u, %u\n", x_coord, y_coord, z_coord);
        //     printf("%f, %f, %f = %x\n", x_vals[idx], y_vals[idx], z_vals[idx], nodes[idx].mortonCode);

        //     uint_fast32_t dec_raw_x, dec_raw_y, dec_raw_z;
        //     morton3D_64_decode(nodes[idx].mortonCode, dec_raw_x, dec_raw_y, dec_raw_z);
        //     float dec_x = ((float)dec_raw_x / bitscale) * range + minCoord;
        //     float dec_y = ((float)dec_raw_y / bitscale) * range + minCoord;
        //     float dec_z = ((float)dec_raw_z / bitscale) * range + minCoord;
        //     printf("decoded = %f, %f, %f\n", dec_x, dec_y, dec_z);
        // }
    }
}

__global__ void fillCodes(const Node* nodes, decltype(nodes[0].mortonCode)* codes, const size_t N) {
    assert(threadIdx.y == threadIdx.z == 1);
    assert(blockIdx.y == blockIdx.z == 1);

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        codes[idx] = nodes[idx].mortonCode;
    }
}

RadixTree::RadixTree(const PointCloud<float>& cloud) {
    size_t n_pts = cloud.x_vals.size();
    // Allocate for tree
    size_t tree_size = n_pts * sizeof(Node);
    CudaCheckCall(cudaMalloc(&d_tree, tree_size));
    // Allocate for raw data points
    size_t data_size = n_pts * sizeof(cloud.x_vals[0]);
    CudaCheckCall(cudaMalloc(&d_data_x, data_size));
    CudaCheckCall(cudaMalloc(&d_data_y, data_size));
    CudaCheckCall(cudaMalloc(&d_data_z, data_size));
    // Copy points to GPU
    CudaCheckCall(cudaMemcpyAsync(d_data_x, &cloud.x_vals[0], data_size, cudaMemcpyHostToDevice));
    CudaCheckCall(cudaMemcpyAsync(d_data_y, &cloud.y_vals[0], data_size, cudaMemcpyHostToDevice));
    CudaCheckCall(cudaMemcpyAsync(d_data_z, &cloud.z_vals[0], data_size, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();

    // Find maximum and minumum values in data
    std::array<float, 3> mins, maxes;
    float *d_mins, *d_maxes;
    CudaCheckCall(g_allocator.DeviceAllocate((void**)&d_mins, sizeof(float) * 3));
    CudaCheckCall(g_allocator.DeviceAllocate((void**)&d_maxes, sizeof(float) * 3));

    size_t temp_storage_reqd = 0;
    void* d_temp_storage = nullptr;
    // get amount of required memory
    DeviceReduce::Max(d_temp_storage, temp_storage_reqd, d_data_x, &maxes[0], n_pts);
    // allocate temporary storage
    CudaCheckCall(g_allocator.DeviceAllocate((void**)&d_temp_storage,  temp_storage_reqd));
    // Find maximum
    DeviceReduce::Max(d_temp_storage, temp_storage_reqd, d_data_x, &d_maxes[0], n_pts);
    DeviceReduce::Max(d_temp_storage, temp_storage_reqd, d_data_y, &d_maxes[1], n_pts);
    DeviceReduce::Max(d_temp_storage, temp_storage_reqd, d_data_z, &d_maxes[2], n_pts);
    DeviceReduce::Min(d_temp_storage, temp_storage_reqd, d_data_x, &d_mins[0], n_pts);
    DeviceReduce::Min(d_temp_storage, temp_storage_reqd, d_data_y, &d_mins[1], n_pts);
    DeviceReduce::Min(d_temp_storage, temp_storage_reqd, d_data_z, &d_mins[2], n_pts);
    cudaDeviceSynchronize();

    cudaMemcpy(&mins[0], d_mins, sizeof(float) * mins.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxes[0], d_maxes, sizeof(float) * maxes.size(), cudaMemcpyDeviceToHost);
    g_allocator.DeviceFree(d_mins);
    g_allocator.DeviceFree(d_maxes);
    g_allocator.DeviceFree(d_temp_storage);
    cudaDeviceSynchronize();
    float max_val = *std::max_element(maxes.begin(), maxes.end());
    float min_val = *std::min_element(mins.begin(), mins.end());
    // std::cout << "range = [" << min_val << ", " << max_val << "]" << std::endl;

    int tpb = 256;
    int blocks = (n_pts + tpb - 1) / tpb;
    makeCodes<<<blocks, tpb>>>(min_val, max_val, d_data_x, d_data_y, d_data_z, d_tree, n_pts);
    cudaDeviceSynchronize();
    CudaCheckError();

    // Now that codes created, raw values not needed
    CudaCheckCall(cudaFree(d_data_x));
    CudaCheckCall(cudaFree(d_data_y));
    CudaCheckCall(cudaFree(d_data_z));

    // Sort in ascending order
    d_temp_storage = nullptr;
    uint64_t *d_keys, *d_keys_sorted;
    Node* d_tree_sorted;
    CudaCheckCall(cudaMalloc(&d_tree_sorted, tree_size));
    CudaCheckCall(cudaMalloc(&d_keys, sizeof(*d_keys) * n_pts));
    CudaCheckCall(cudaMalloc(&d_keys_sorted, sizeof(*d_keys_sorted) * n_pts));
    fillCodes<<<blocks, tpb>>>(d_tree, d_keys, n_pts);
    cudaDeviceSynchronize();
    CudaCheckError();
    CudaCheckCall(
        DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_reqd,
                                   d_keys, d_keys_sorted,
                                   d_tree, d_tree_sorted,
                                   n_pts)
    );
    CudaCheckCall(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_reqd));
    CudaCheckCall(
        DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_reqd,
                                   d_keys, d_keys_sorted,
                                   d_tree, d_tree_sorted,
                                   n_pts)
    );
}

RadixTree::~RadixTree() {
    CudaCheckCall(cudaFree(d_tree));
}
