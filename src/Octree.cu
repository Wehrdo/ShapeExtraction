#include "Octree.hpp"
#include "CudaCommon.hpp"
#include "cub/device/device_scan.cuh"

#include <type_traits>

using namespace OT;
using cub::DeviceScan;

__device__ void OTNode::setChild(size_t child, int my_child_idx) {
    children[my_child_idx] = child;
    atomicOr(&child_mask, 1 << my_child_idx);
    // child_mask |= (1 << my_child_idx);
}

__global__ void calcEdgeNodes(
    const std::remove_pointer<decltype(RT::Nodes::prefixN)>::type *prefixN,
    const std::remove_pointer<decltype(RT::Nodes::parent)>::type *parents,
    int* edgeNodes,
    const size_t N) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    // root has no parent, so don't do for index 0
    if (i > 0 && i < N) {
        int my_depth = prefixN[i] / 3;
        int parent_depth = prefixN[parents[i]] / 3;
        edgeNodes[i] = my_depth - parent_depth;
    }
}

__global__ void initializeOTNodes(
    OTNode* nodes,
    const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        nodes[i].child_mask = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            nodes[i].children[i] = -1;
        }
    }
}

__global__ void makeNodes(
    const int* node_offsets,
    const RT::Code_t* codes,
    const int* rt_node_counts,
    const uint8_t* prefixN,
    const int* rt_parents,
    OTNode* nodes,
    const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: What to do with node 0?
    if (i > 0 && i < N) {
        int oct_idx = node_offsets[i];
        // int n_new_nodes = node_offsets[i] - node_offsets[i - 1];
        int n_new_nodes = rt_node_counts[i];
        for (int j = 0; j < n_new_nodes - 1; ++j) {
            // int out_idx = i + j; // TODO: optimize. can loop over indices directly
            // int level = prefixN[i]/3 - (n_new_nodes - 1 - j);
            // RT::Code_t node_prefix = codes[i] >> (RT::codeLen - (3 * level));
            // if (j != 0) {
            //     size_t parent_idx = out_idx - 1;
            //     nodes[this_idx].parent = parent_idx;
            //     size_t child_code = node_prefix & 0b111;
            //     nodes[parent_idx].children[child_code] = out_idx;
            // }


            // int level = prefixN[i]/3 - (n_new_nodes - 1 - j);
            int level = prefixN[i]/3 - j;
            RT::Code_t node_prefix = codes[i] >> (RT::codeLen - (3 * level));
            int child_idx = node_prefix & 0b111;
            int parent = oct_idx + 1;
            // nodes[parent].children[child_idx] = oct_idx;
            // nodes[parent].child_mask |= 1 << child_idx;
            nodes[parent].setChild(oct_idx, child_idx);
            oct_idx = parent;
        }
        if (n_new_nodes > 0) {
            int rt_parent = rt_parents[i];
            while (rt_node_counts[rt_parent] == 0) {
                rt_parent = rt_parents[rt_parent];
            }
            int oct_parent = node_offsets[rt_parent];
            // int top_level = prefixN[i]/3;
            int top_level = prefixN[i]/3 - n_new_nodes + 1;
            RT::Code_t top_node_prefix = codes[i] >> (RT::codeLen - (3 * top_level));
            int child_idx = top_node_prefix & 0b111;
            nodes[oct_parent].setChild(oct_idx, child_idx);
        }
    }
}

Octree::Octree(const RT::RadixTree& radix_tree) {
    // Copy a "1" to the first element to account for the root
    
    CudaCheckCall(cudaMallocManaged(&edgeNodes, sizeof(*edgeNodes) * radix_tree.n_nodes));
    edgeNodes[0] = 1;
    int blocks, tpb;
    std::tie(blocks, tpb) = makeLaunchParams(radix_tree.n_nodes);
    calcEdgeNodes<<<blocks, tpb>>>(radix_tree.d_tree.prefixN, radix_tree.d_tree.parent, edgeNodes, radix_tree.n_nodes);
	cudaDeviceSynchronize();
    CudaCheckError();

    // Inclusive prefix sum to find location of each octree node
    int* oc_node_offsets;
    CudaCheckCall(cudaMallocManaged(&oc_node_offsets, (1 + radix_tree.n_nodes) * sizeof(*oc_node_offsets)));
    oc_node_offsets[0] = 0;

    void* d_temp_storage = nullptr;
    size_t temp_storage_reqd = 0;
    CudaCheckCall(
        DeviceScan::InclusiveSum(d_temp_storage, temp_storage_reqd,
                edgeNodes, oc_node_offsets + 1,
                radix_tree.n_nodes)
    );
    CudaCheckCall(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_reqd));
    CudaCheckCall(
        DeviceScan::InclusiveSum(d_temp_storage, temp_storage_reqd,
                edgeNodes, oc_node_offsets + 1,
                radix_tree.n_nodes)
    );
	cudaDeviceSynchronize();
    CudaCheckError();
    g_allocator.DeviceFree(d_temp_storage);

    auto totalNodes = oc_node_offsets[radix_tree.n_nodes];
    printf("total nodes: %d\n", totalNodes);

    CudaCheckCall(cudaMallocManaged(&nodes, totalNodes * sizeof(*nodes)));

    // setup initial values of octree node objects
    initializeOTNodes<<<blocks, tpb>>>(nodes, totalNodes);

    makeNodes<<<blocks, tpb>>>(oc_node_offsets,
                               radix_tree.d_tree.mortonCode,
                               edgeNodes,
                               radix_tree.d_tree.prefixN,
                               radix_tree.d_tree.parent,
                               nodes,
                               radix_tree.n_nodes);
    cudaDeviceSynchronize();
    CudaCheckError();

    for (int i = 0; i < totalNodes; ++i) {
        printf("Node %d:\n\tparent: %d\n\tchild_mask: %x\n\tchildren:\n", i, nodes[i].parent, nodes[i].child_mask);
        for (int j = 0; j < 8; ++j) {
            if (nodes[i].child_mask & (1 << j)) {
                printf("\t\t%d: %d\n", j, nodes[i].children[j]);
            }
        }
    }
}

Octree::~Octree() {
    CudaCheckCall(cudaFree(edgeNodes));
}