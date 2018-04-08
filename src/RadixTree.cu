#include "RadixTree.hpp"
#include "CudaCommon.cuh"
#include "libmorton/include/morton.h"
#include "cub/device/device_reduce.cuh"
#include "cub/device/device_radix_sort.cuh"

#include <array>
#include <algorithm>
#include <limits>
#include <cstdint>
#include <type_traits>

#include <math.h>

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

__global__ void fillCodes(const Node* nodes, Code_t* codes, const size_t N) {
    assert(threadIdx.y == threadIdx.z == 1);
    assert(blockIdx.y == blockIdx.z == 1);

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        codes[idx] = nodes[idx].mortonCode;
    }
}

// computes ceil(a / b)
template<typename T>
__device__ inline T ceil_div(T a, T b) {
    // If a + b might overflow, do the following instead? (untested):
    //     1 + ((x - 1) / y); // if x != 0
    assert(!std::is_signed<decltype(a)>() || a >= 0);
    assert(!std::is_signed<decltype(b)>() || b >= 0);
    return (a + b - 1) / b;
}

// delta(a, b) is the length of the longest prefix between codes a and b
__device__ inline uint_fast8_t delta(const Code_t a, const Code_t b) {
    // Assuming first bit is 0. Asserts check that.
    // Not necessary, so if want to store info in that bit in the future, requires a change
    Code_t bit1_mask = (Code_t)1 << (sizeof(a) * 8 - 1);
    assert(a & bit1_mask == 0);
    assert(b & bit1_mask == 0);
    return __clzll(a ^ b) - 1;
}

__global__ void constructTree(Node* nodes, const size_t N) {
    assert(threadIdx.y == threadIdx.z == 1);
    assert(blockIdx.y == blockIdx.z == 1);

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        auto code_i = nodes[i].mortonCode;
        // Determine direction of the range (+1 or -1)
        // TODO: This will break when i = 0 or i = n-1
        auto delta_diff_right = delta(code_i, nodes[i+1].mortonCode);
        auto delta_diff_left = delta(code_i, nodes[i-1].mortonCode);
        int_fast8_t direction_difference = delta_diff_right - delta_diff_left;
        int_fast8_t d = (direction_difference > 0) - (direction_difference < 0);
        assert(d == -1 || d == 1);

        // Compute upper bound for the length of the range
        auto delta_min = delta(code_i, nodes[i - d].mortonCode);
        Code_t l_max = 2;
        // Cast to ptrdiff_t so in case the result is negative (since d is +/- 1), we can catch it and not index out of bounds
        while (static_cast<ptrdiff_t>(i) + static_cast<ptrdiff_t>(l_max)*d >= 0 &&
               i + l_max*d < N &&
               delta(code_i, nodes[i + l_max * d].mortonCode) > delta_min) {
            l_max *= 2;
        }
        // Find the other end using binary search
        Code_t l = 0;
        uint_fast8_t divisor;
        size_t t;
        for (t = l_max / 2, divisor = 2; t >= 1; divisor *= 2, t = l_max / divisor) {
            if (delta(code_i, nodes[i + (l + t)*d].mortonCode) > delta_min) {
                l += t;
            }
        }
        size_t j = i + l*d;
        // Find the split position using binary search
        auto delta_node = delta(nodes[i].mortonCode, nodes[j].mortonCode);
        size_t s = 0;
        for (t = ceil_div<Code_t>(l, 2), divisor = 2; t >= 1; divisor *= 2, t = ceil_div<Code_t>(l, divisor)) {
            if (delta(code_i, nodes[i + (s + t)*d].mortonCode) > delta_node) {
                s += t;
            }
        }

        // Split position
        size_t gamma = i + s*d + min(d, 0);
        nodes[i].leftChild = gamma;
        nodes[i].hasLeafLeft = (min(i, j) == gamma);
        nodes[i].hasLeafRight = (max(i, j) == gamma+1);
        // Set parents of left and right children, if they aren't leaves
        // can't set this node as parent of its leaves, because the
        // leaf also represents an internal node with a differnent parent
        if (!nodes[i].hasLeafLeft) {
            nodes[gamma].parent = i;
        }
        if (!nodes[i].hasLeafRight) {
            nodes[gamma + 1].parent = i;
        }
    }
}

 std::tuple<int, int> makeLaunchParams(size_t n, int tpb = 512) {
    // int tpb = 256;
    int blocks = (n + tpb - 1) / tpb;
    return std::make_tuple(blocks, tpb);
 }

void RadixTree::encodePoints(const PointCloud<float>& cloud) {
    // Check that the cast is okay
    assert(cloud.x_vals.size() <= std::numeric_limits<decltype(n_pts)>::max());
    n_pts = static_cast<decltype(n_pts)>(cloud.x_vals.size());

    // Allocate for tree
    size_t tree_size = n_pts * sizeof(Node);
    CudaCheckCall(cudaMalloc(&d_tree, tree_size));
    // Allocate for raw data points
    size_t data_size = n_pts * sizeof(cloud.x_vals[0]);
    float *d_data_x, *d_data_y, *d_data_z;
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
    CudaCheckError();

    cudaMemcpy(&mins[0], d_mins, sizeof(float) * mins.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxes[0], d_maxes, sizeof(float) * maxes.size(), cudaMemcpyDeviceToHost);
    g_allocator.DeviceFree(d_mins);
    g_allocator.DeviceFree(d_maxes);
    g_allocator.DeviceFree(d_temp_storage);
    cudaDeviceSynchronize();
    float max_val = *std::max_element(maxes.begin(), maxes.end());
    float min_val = *std::min_element(mins.begin(), mins.end());
    // std::cout << "range = [" << min_val << ", " << max_val << "]" << std::endl;

    int blocks, tpb;
    std::tie(blocks, tpb) = makeLaunchParams(n_pts);
    makeCodes<<<blocks, tpb>>>(min_val, max_val, d_data_x, d_data_y, d_data_z, d_tree, n_pts);
    cudaDeviceSynchronize();
    CudaCheckError();

    // Now that codes created, raw values not needed
    CudaCheckCall(cudaFree(d_data_x));
    CudaCheckCall(cudaFree(d_data_y));
    CudaCheckCall(cudaFree(d_data_z));
}

RadixTree::RadixTree(const PointCloud<float>& cloud) {
    // fill up mortonCode in d_tree
    encodePoints(cloud);

    // Sort in ascending order
    // Just the Morton codes from the nodes
    Code_t *d_keys;
    CudaCheckCall(cudaMalloc(&d_keys, sizeof(*d_keys) * n_pts));
    int blocks, tpb;
    std::tie(blocks, tpb) = makeLaunchParams(n_pts);
    fillCodes<<<blocks, tpb>>>(d_tree, d_keys, n_pts);
    cudaDeviceSynchronize();
    CudaCheckError();
    void* d_temp_storage = nullptr;
    size_t temp_storage_reqd = 0;
    CudaCheckCall(
        // get storage requirements
        DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_reqd,
                                   d_keys, d_keys,
                                   d_tree, d_tree,
                                   n_pts)
    );
    CudaCheckCall(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_reqd));
    // sort key-value pairs, where key is morton code (d_keys), and values are tree nodes
    CudaCheckCall(
        DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_reqd,
                                   d_keys, d_keys,
                                   d_tree, d_tree,
                                   n_pts)
    );
    cudaDeviceSynchronize();
    CudaCheckError();
    g_allocator.DeviceFree(d_temp_storage);

    // Make tree

}

RadixTree::~RadixTree() {
    CudaCheckCall(cudaFree(d_tree));
}
