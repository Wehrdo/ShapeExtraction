#include "RadixTree.hpp"
#include "CudaCommon.hpp"
#include "cub/device/device_reduce.cuh"
#include "cub/device/device_radix_sort.cuh"

#include <memory>
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
__global__ void makeCodes(
    const T min_coord,
    const T range,
    const T* __restrict__ x_vals,
    const T* __restrict__ y_vals,
    const T* __restrict__ z_vals,
    Code_t* codes,
    const size_t N) {
    // only supports 1-dimension blocks and grids
    assert(threadIdx.y == threadIdx.z == 1);
    assert(blockIdx.y == blockIdx.z == 1);

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // printf("Raw point %d = (%f, %f, %f)\n", idx, x_vals[idx], y_vals[idx], z_vals[idx]);
        codes[idx] = pointToCode(x_vals[idx], y_vals[idx], z_vals[idx], min_coord, range);
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

__device__ inline int log2_ceil(Code_t x) {
    static_assert(sizeof(x) == sizeof(long long int), "__clzll(x) is for long long int");
    // Counting from LSB to MSB, number of bits before last '1'
    // This is floor(log(x))
    int n_lower_bits = (8 * sizeof(x)) - __clzll(x) - 1;
    // Add 1 if 2^n_lower_bits is less than x
    //     (i.e. we rounded down because x was not a power of 2)
    return n_lower_bits + (x > (1 << n_lower_bits));
}

// delta(a, b) is the length of the longest prefix between codes a and b
__device__ inline int_fast8_t delta(const Code_t a, const Code_t b) {
    // Assuming first bit is 0. Asserts check that.
    // Not necessary, so if want to store info in that bit in the future, requires a change
    Code_t bit1_mask = (Code_t)1 << (sizeof(a) * 8 - 1);
    assert((a & bit1_mask) == 0);
    assert((b & bit1_mask) == 0);
    return __clzll(a ^ b) - 1;
}

__global__ void constructTree(
    const Code_t* codes,
    bool* hasLeafLeft,
    bool* hasLeafRight,
    int* leftChild,
    int* parent,
    uint8_t* prefixN,
    const size_t N) {
    assert(threadIdx.y == threadIdx.z == 1);
    assert(blockIdx.y == blockIdx.z == 1);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        auto code_i = codes[i];
        // Determine direction of the range (+1 or -1)
        int d;
        if (i == 0) {
            d = 1;
        }
        else {
            auto delta_diff_right = delta(code_i, codes[i+1]);
            auto delta_diff_left = delta(code_i, codes[i-1]);
            int direction_difference = delta_diff_right - delta_diff_left;
            d = (direction_difference > 0) - (direction_difference < 0);
        }

        // Compute upper bound for the length of the range
        
        Code_t l = 0;
        if (i == 0) {
            // First node is root, covering whole tree
            l = N - 1;
        }
        else {
            auto delta_min = delta(code_i, codes[i - d]);
            Code_t l_max = 2;
            // Cast to ptrdiff_t so in case the result is negative (since d is +/- 1), we can catch it and not index out of bounds
            while (i + static_cast<ptrdiff_t>(l_max)*d >= 0 &&
                i + l_max*d < N &&
                delta(code_i, codes[i + l_max * d]) > delta_min) {
                l_max *= 2;
            }
            int t;
            int divisor;
            // Find the other end using binary search
            for (t = l_max / 2, divisor = 2; t >= 1; divisor *= 2, t = l_max / divisor) {
                if (i + (l + t)*d >= 0 &&
                    i + (l + t)*d < N &&
                    delta(code_i, codes[i + (l + t)*d]) > delta_min) {
                    l += t;
                }
            }
        }
        int j = i + l*d;
        // Find the split position using binary search
        auto delta_node = delta(codes[i], codes[j]);
        prefixN[i] = delta_node;
        int s = 0;
        int t;
        int max_divisor = 1 << log2_ceil(l);
        int divisor = 2;
        for (t = ceil_div<Code_t>(l, 2); divisor <= max_divisor; divisor <<= 1, t = ceil_div<Code_t>(l, divisor)) {
        // for (t = ceil_div<Code_t>(l, 2), divisor = 2; t >= 1; divisor *= 2, t = ceil_div<Code_t>(l, divisor)) {
            if (delta(code_i, codes[i + (s + t)*d]) > delta_node) {
                s += t;
            }
        }

        // Split position
        int gamma = i + s*d + min(d, 0);
        leftChild[i] = gamma;
        hasLeafLeft[i] = (min(i, j) == gamma);
        hasLeafRight[i] = (max(i, j) == gamma+1);
        // Set parents of left and right children, if they aren't leaves
        // can't set this node as parent of its leaves, because the
        // leaf also represents an internal node with a differnent parent
        if (!hasLeafLeft[i]) {
            parent[gamma] = i;
        }
        if (!hasLeafRight[i]) {
            parent[gamma + 1] = i;
        }
    }
}

void RadixTree::encodePoints(const PointCloud<float>& cloud) {
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
    // std::array<float, 3> mins, maxes;
    float *mins, *maxes;
    CudaCheckCall(cudaMallocManaged(&mins, sizeof(float) * 3));
    CudaCheckCall(cudaMallocManaged(&maxes, sizeof(float) * 3));
    // float *d_mins, *d_maxes;
    // CudaCheckCall(g_allocator.DeviceAllocate((void**)&d_mins, sizeof(float) * 3));
    // CudaCheckCall(g_allocator.DeviceAllocate((void**)&d_maxes, sizeof(float) * 3));

    size_t temp_storage_reqd = 0;
    void* d_temp_storage = nullptr;
    // get amount of required memory
    DeviceReduce::Max(d_temp_storage, temp_storage_reqd, d_data_x, &maxes[0], n_pts);
    // allocate temporary storage
    CudaCheckCall(g_allocator.DeviceAllocate((void**)&d_temp_storage,  temp_storage_reqd));
    // Find maximum
    DeviceReduce::Max(d_temp_storage, temp_storage_reqd, d_data_x, &maxes[0], n_pts);
    DeviceReduce::Max(d_temp_storage, temp_storage_reqd, d_data_y, &maxes[1], n_pts);
    DeviceReduce::Max(d_temp_storage, temp_storage_reqd, d_data_z, &maxes[2], n_pts);
    DeviceReduce::Min(d_temp_storage, temp_storage_reqd, d_data_x, &mins[0], n_pts);
    DeviceReduce::Min(d_temp_storage, temp_storage_reqd, d_data_y, &mins[1], n_pts);
    DeviceReduce::Min(d_temp_storage, temp_storage_reqd, d_data_z, &mins[2], n_pts);
    cudaDeviceSynchronize();
    CudaCheckError();

    // cudaMemcpy(&mins[0], d_mins, sizeof(float) * mins.size(), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&maxes[0], d_maxes, sizeof(float) * maxes.size(), cudaMemcpyDeviceToHost);
    // g_allocator.DeviceFree(d_mins);
    // g_allocator.DeviceFree(d_maxes);
    g_allocator.DeviceFree(d_temp_storage);
    cudaDeviceSynchronize();
    max_coord = *std::max_element(&maxes[0], &maxes[3]);
    min_coord = *std::min_element(&mins[0], &mins[3]);
    // std::cout << "range = [" << min_val << ", " << max_val << "]" << std::endl;

    int blocks, tpb;
    std::tie(blocks, tpb) = makeLaunchParams(n_pts);
    makeCodes<<<blocks, tpb>>>(min_coord, max_coord - min_coord, d_data_x, d_data_y, d_data_z, d_tree.mortonCode, n_pts);
    cudaDeviceSynchronize();
    CudaCheckError();

    // Now that codes created, raw values not needed
    CudaCheckCall(cudaFree(d_data_x));
    CudaCheckCall(cudaFree(d_data_y));
    CudaCheckCall(cudaFree(d_data_z));
}

RadixTree::RadixTree(const PointCloud<float>& cloud) {
    // Check that the cast is okay
    assert(cloud.x_vals.size() <= std::numeric_limits<decltype(n_pts)>::max());
    n_pts = static_cast<decltype(n_pts)>(cloud.x_vals.size());
    // allocate memory for tree
    CudaCheckCall(cudaMallocManaged(&d_tree.mortonCode, sizeof(*d_tree.mortonCode) * n_pts));
    CudaCheckCall(cudaMallocManaged(&d_tree.hasLeafLeft, sizeof(*d_tree.hasLeafRight) * n_pts));
    CudaCheckCall(cudaMallocManaged(&d_tree.hasLeafRight, sizeof(*d_tree.hasLeafRight) * n_pts));
    CudaCheckCall(cudaMallocManaged(&d_tree.prefixN, sizeof(*d_tree.prefixN) * n_pts));
    CudaCheckCall(cudaMallocManaged(&d_tree.leftChild, sizeof(*d_tree.leftChild) * n_pts));
    CudaCheckCall(cudaMallocManaged(&d_tree.parent, sizeof(*d_tree.parent) * n_pts));

    // fill up mortonCode in d_tree
    encodePoints(cloud);

    // Sort in ascending order
    // Just the Morton codes from the nodes
    Code_t* d_codes_sorted;
    CudaCheckCall(cudaMallocManaged(&d_codes_sorted, sizeof(*d_codes_sorted) * n_pts));
    void* d_temp_storage = nullptr;
    size_t temp_storage_reqd = 0;
    CudaCheckCall(
        // get storage requirements
        DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_reqd,
                                  d_tree.mortonCode, d_codes_sorted,
                                  n_pts)
    );
    CudaCheckCall(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_reqd));
    // sort key-value pairs, where key is morton code (d_keys), and values are tree nodes
    CudaCheckCall(
        DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_reqd,
                                   d_tree.mortonCode, d_codes_sorted,
                                   n_pts)
    );
    cudaDeviceSynchronize();
    CudaCheckError();
    g_allocator.DeviceFree(d_temp_storage);
    // TODO: Remove duplicates
    n_nodes = n_pts - 1;

    // Swap out keys for sorted keys
    // Okay to do at this point, because nothing else in d_tree has been filled
    CudaCheckCall(cudaFree(d_tree.mortonCode));
    d_tree.mortonCode = d_codes_sorted;

    // Make tree
    int blocks, tpb;
    std::tie(blocks, tpb) = makeLaunchParams(n_nodes);
    constructTree<<<blocks, tpb>>>(d_tree.mortonCode,
                                   d_tree.hasLeafLeft,
                                   d_tree.hasLeafRight,
                                   d_tree.leftChild,
                                   d_tree.parent,
                                   d_tree.prefixN,
                                   n_nodes);
	cudaDeviceSynchronize();
    CudaCheckError();

    // for (int i = 0; i < n_nodes; ++i) {
    //     printf("idx = %d, code = %llx, prefixN = %d, left = %d, parent = %d, leftLeaf=%d, rightLeft=%d\n",
    //             i, d_tree.mortonCode[i], (int)d_tree.prefixN[i], d_tree.leftChild[i], d_tree.parent[i], (int)d_tree.hasLeafLeft[i], (int)d_tree.hasLeafRight[i]);
    // }
}

RadixTree::~RadixTree() {
    CudaCheckCall(cudaFree(d_tree.mortonCode));
    CudaCheckCall(cudaFree(d_tree.hasLeafLeft));
    CudaCheckCall(cudaFree(d_tree.hasLeafRight));
    CudaCheckCall(cudaFree(d_tree.prefixN));
    CudaCheckCall(cudaFree(d_tree.leftChild));
    CudaCheckCall(cudaFree(d_tree.parent));
}
