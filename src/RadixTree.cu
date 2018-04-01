#include <cuda_runtime_api.h>
#include <cuda.h>

#include "RadixTree.hpp"
#include "CudaCommon.cuh"
#include "libmorton/include/morton.h"

using namespace RT;

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
        uint32_t x_coord = (x_vals[idx] / range) + minCoord;
        uint32_t y_coord = (y_vals[idx] / range) + minCoord;
        uint32_t z_coord = (z_vals[idx] / range) + minCoord;
        nodes[idx].mortonCode = morton3D_64_encode(x_coord, y_coord, z_coord);
    }
}

RadixTree::RadixTree(const PointCloud<float>& cloud) {
    size_t tree_size = cloud.x_vals.size() * sizeof(Node);
    std::cout << "allocating " << tree_size << " bytes" << std::endl;
    CudaCheckCall(cudaMalloc(&d_tree, tree_size));
}

RadixTree::~RadixTree() {
    CudaCheckCall(cudaFree(d_tree));
}