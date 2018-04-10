#include "Octree.hpp"
#include "CudaCommon.hpp"
#include "cub/device/device_scan.cuh"

#include <type_traits>

using namespace RT;
using cub::DeviceScan;

__global__ void calcEdgeNodes(
    const std::remove_pointer<decltype(Nodes::prefixN)>::type *prefixN,
    const std::remove_pointer<decltype(Nodes::parent)>::type *parents,
    edgeNodeCnt_t *edgeNodes,
    const size_t N) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < N-1) {
        int my_depth = prefixN[i] / 3;
        int parent_depth = prefixN[parents[i]] / 3;
        edgeNodes[i] = my_depth - parent_depth;
    }
}

Octree::Octree(const RadixTree& radix_tree) {
    // Copy a "1" to the first element to account for the root
    
    CudaCheckCall(cudaMalloc(&d_edgeNodes, sizeof(*d_edgeNodes) * radix_tree.n_nodes));
    edgeNodeCnt_t edgeNode_1 = 1;
    CudaCheckCall(cudaMemcpyAsync(d_edgeNodes, &edgeNode_1, sizeof(edgeNode_1), cudaMemcpyHostToDevice));
    int blocks, tpb;
    std::tie(blocks, tpb) = makeLaunchParams(radix_tree.n_nodes);
    calcEdgeNodes<<<blocks, tpb>>>(radix_tree.d_tree.prefixN, radix_tree.d_tree.parent, d_edgeNodes, radix_tree.n_nodes);
	cudaDeviceSynchronize();
    CudaCheckError();

    // Inclusive prefix sum to find location of each octree node
    edgeNodeCnt_t *oc_node_offsets;
    CudaCheckCall(cudaMalloc(&oc_node_offsets, radix_tree.n_nodes * sizeof(*oc_node_offsets)));

    void* d_temp_storage = nullptr;
    size_t temp_storage_reqd = 0;
    CudaCheckCall(
        DeviceScan::InclusiveSum(d_temp_storage, temp_storage_reqd,
                d_edgeNodes, oc_node_offsets,
                radix_tree.n_nodes)
    );
    CudaCheckCall(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_reqd));
    CudaCheckCall(
        DeviceScan::InclusiveSum(d_temp_storage, temp_storage_reqd,
                d_edgeNodes, oc_node_offsets,
                radix_tree.n_nodes)
    );
	cudaDeviceSynchronize();
    CudaCheckError();
    g_allocator.DeviceFree(d_temp_storage);
}

Octree::~Octree() {
    CudaCheckCall(cudaFree(d_edgeNodes));
}