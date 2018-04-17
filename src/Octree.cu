#include "Octree.hpp"
#include "dequeue.hpp"
#include "MortonUtils.hpp"
#include "CudaCommon.hpp"
#include "cub/device/device_scan.cuh"

#include <vector>
#include <tuple>
#include <type_traits>

using namespace OT;
using cub::DeviceScan;

__global__ void decodePoints(
    Point* points,
    const Code_t* codes,
    const float min_coord,
    const float range,
    const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        points[i] = codeToPoint(codes[i], min_coord, range);
    }
}

__device__ void OTNode::setChild(const int child, const int my_child_idx) {
    children[my_child_idx] = child;
    // atomic version of child_mask |= (1 << my_child_idx);
    atomicOr(&child_node_mask, 1 << my_child_idx);
}

__device__ void OTNode::setLeaf(const int leaf, const int my_child_idx) {
    children[my_child_idx] = leaf;
    // atomic version of child_mask &= ~(1 << my_child_idx);
    atomicOr(&child_leaf_mask, 1 << my_child_idx);
}

__global__ void calcEdgeCounts(
    const std::remove_pointer<decltype(RT::Nodes::prefixN)>::type *prefixN,
    const std::remove_pointer<decltype(RT::Nodes::parent)>::type *parents,
    int* rt_edge_counts,
    const size_t N) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    // root has no parent, so don't do for index 0
    if (i > 0 && i < N) {
        int my_depth = prefixN[i] / 3;
        int parent_depth = prefixN[parents[i]] / 3;
        rt_edge_counts[i] = my_depth - parent_depth;
    }
}

// __global__ void initializeOTNodes(
//     OTNode* nodes,
//     const int N) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < N) {
//         nodes[i].child_mask = 0;
//         #pragma unroll
//         for (int child = 0; child < 8; ++child) {
//             nodes[i].children[child] = -1;
//         }
//     }
// }

__global__ void linkLeafNodes(
    OTNode* nodes,
    const int* node_offsets,
    const int* rt_node_counts,
    const Code_t* codes,
    const bool* rt_hasLeafLeft,
    const bool* rt_hasLeafRight,
    const uint8_t* rt_prefixN,
    const int* rt_leftChild,
    const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // int n_new_nodes = rt_node_counts[i];
        // if (n_new_nodes > 0) {
            // link leaves if possible
            if (rt_hasLeafLeft[i]) {
                int leaf_idx = rt_leftChild[i];
                int leaf_level = rt_prefixN[i]/3 + 1;
                Code_t leaf_prefix = codes[leaf_idx] >> (CODE_LEN - (3 * leaf_level));
                int child_idx = leaf_prefix & 0b111;
                // link leaf to bottom octree node in string
                int bottom_oct = node_offsets[i];
                nodes[bottom_oct].setLeaf(leaf_idx, child_idx);
            }
            if (rt_hasLeafRight[i]) {
                int leaf_idx = rt_leftChild[i] + 1;
                int leaf_level = rt_prefixN[i]/3 + 1;
                Code_t leaf_prefix = codes[leaf_idx] >> (CODE_LEN - (3 * leaf_level));
                int child_idx = leaf_prefix & 0b111;
                int bottom_oct = node_offsets[i];
                nodes[bottom_oct].setLeaf(leaf_idx, child_idx);
            }
        // }
    }

}

__global__ void makeNodes(
    OTNode* nodes,
    const int* node_offsets,
    const int* rt_node_counts,
    const Code_t* codes,
    const uint8_t* rt_prefixN,
    const int* rt_parents,
    const float min_coord,
    const float range,
    const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < N) {
        // the root doesn't represent level 0 of the "entire" octree
        int root_level = rt_prefixN[0]/3;
        int oct_idx = node_offsets[i];
        // int n_new_nodes = node_offsets[i] - node_offsets[i - 1];
        int n_new_nodes = rt_node_counts[i];
        for (int j = 0; j < n_new_nodes - 1; ++j) {
            int level = rt_prefixN[i]/3 - j;
            Code_t node_prefix = codes[i] >> (CODE_LEN - (3 * level));
            int child_idx = node_prefix & 0b111;
            int parent = oct_idx + 1;
            nodes[parent].setChild(oct_idx, child_idx);
            // calculate corner point
            //   (less significant bits have already been shifted off)
            nodes[oct_idx].corner = codeToPoint(node_prefix << (CODE_LEN - (3 * level)), min_coord, range);
            // each cell is half the size of 
            nodes[oct_idx].cell_size = range / static_cast<float>(1 << (level - root_level));
            oct_idx = parent;
        }
        if (n_new_nodes > 0) {
            int rt_parent = rt_parents[i];
            while (rt_node_counts[rt_parent] == 0) {
                rt_parent = rt_parents[rt_parent];
            }
            int oct_parent = node_offsets[rt_parent];
            int top_level = rt_prefixN[i]/3 - n_new_nodes + 1;
            Code_t top_node_prefix = codes[i] >> (CODE_LEN - (3 * top_level));
            int child_idx = top_node_prefix & 0b111;
            nodes[oct_parent].setChild(oct_idx, child_idx);
            // set corner point
            nodes[oct_idx].corner = codeToPoint(top_node_prefix << (CODE_LEN - (3 * top_level)), min_coord, range);
            nodes[oct_idx].cell_size = range / static_cast<float>(1 << (top_level - root_level));
        }
    }
}

Octree::Octree(const RT::RadixTree& radix_tree) {
    // Number of octree nodes between a node and its parent
    int* rt_edge_counts;
    CudaCheckCall(cudaMallocManaged(&rt_edge_counts, sizeof(*rt_edge_counts) * radix_tree.n_nodes));
    // Copy a "1" to the first element to account for the root
    rt_edge_counts[0] = 1;
    int blocks, tpb;
    std::tie(blocks, tpb) = makeLaunchParams(radix_tree.n_nodes);
    calcEdgeCounts<<<blocks, tpb>>>(radix_tree.d_tree.prefixN, radix_tree.d_tree.parent, rt_edge_counts, radix_tree.n_nodes);
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
                rt_edge_counts, oc_node_offsets + 1,
                radix_tree.n_nodes)
    );
    CudaCheckCall(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_reqd));
    CudaCheckCall(
        DeviceScan::InclusiveSum(d_temp_storage, temp_storage_reqd,
                rt_edge_counts, oc_node_offsets + 1,
                radix_tree.n_nodes)
    );
	cudaDeviceSynchronize();
    CudaCheckError();
    g_allocator.DeviceFree(d_temp_storage);

    auto n_oct_nodes = oc_node_offsets[radix_tree.n_nodes];
    printf("total nodes: %d\n", n_oct_nodes);

    CudaCheckCall(cudaMallocManaged(&nodes, n_oct_nodes * sizeof(*nodes)));

    // setup initial values of octree node objects
    CudaCheckCall(cudaMemset(nodes, 0, n_oct_nodes * sizeof(*nodes)));
    // std::tie(blocks, tpb) = makeLaunchParams(n_oct_nodes);
    // initializeOTNodes<<<blocks, tpb>>>(nodes, n_oct_nodes);

    std::tie(blocks, tpb) = makeLaunchParams(radix_tree.n_nodes);
    makeNodes<<<blocks, tpb>>>(nodes,
                               oc_node_offsets,
                               rt_edge_counts,
                               radix_tree.d_tree.mortonCode,
                               radix_tree.d_tree.prefixN,
                               radix_tree.d_tree.parent,
                               radix_tree.min_coord,
                               radix_tree.max_coord - radix_tree.min_coord,
                               radix_tree.n_nodes);

    linkLeafNodes<<<blocks, tpb>>>(nodes,
                                   oc_node_offsets,
                                   rt_edge_counts,
                                   radix_tree.d_tree.mortonCode,
                                   radix_tree.d_tree.hasLeafLeft,
                                   radix_tree.d_tree.hasLeafRight,
                                   radix_tree.d_tree.prefixN,
                                   radix_tree.d_tree.leftChild,
                                   radix_tree.n_nodes);

    // cudaDeviceSynchronize();
    // for (int i = 0; i < n_oct_nodes; ++i) {
    //     printf("Node %d:\n\tparent: %d\n\tchildren:\n", i, nodes[i].parent);
    //     for (int j = 0; j < 8; ++j) {
    //         if (nodes[i].child_node_mask & (1 << j)) {
    //             printf("\t\tNode %d: %d\n", j, nodes[i].children[j]);
    //         }
    //         if (nodes[i].child_leaf_mask & (1 << j)) {
    //             printf("\t\tLeaf %d: %d\n", j, nodes[i].children[j]);
    //         }
    //     }
    // }

    // free temporary memory from construction
    CudaCheckCall(cudaFree(rt_edge_counts));
    CudaCheckCall(cudaFree(oc_node_offsets));

    // decode points for use later
    CudaCheckCall(cudaMallocManaged(&u_points, radix_tree.n_pts * sizeof(*u_points)));
    std::tie(blocks, tpb) = makeLaunchParams(radix_tree.n_pts);
    decodePoints<<<blocks, tpb>>>(u_points,
                                   radix_tree.d_tree.mortonCode,
                                   radix_tree.min_coord,
                                   radix_tree.max_coord - radix_tree.min_coord,
                                   radix_tree.n_pts);
    cudaDeviceSynchronize();
    CudaCheckError();
    // copy them to host memory, too
    h_points = new Point[radix_tree.n_pts];
    CudaCheckCall(cudaMemcpy(h_points, u_points, radix_tree.n_pts * sizeof(*h_points), cudaMemcpyDeviceToHost));
}

Octree::~Octree() {
    CudaCheckCall(cudaFree(nodes));
    CudaCheckCall(cudaFree(u_points));
    delete h_points;
}

// return index of best point and distance to it
__device__ std::tuple<int, float> OT::nodePointDistance(
    const Point& query_pt,
    const OTNode& node,
    const Point* points) {
    // best distance (squared) so far (guaranteed to find a point closer, so will not return infinity)
    float nearest_dist2 = INFINITY;
    int best_pt_idx = -1;
    // for the following loop, last bit is whether this child is a leaf
    int cur_leaf_mask = node.child_leaf_mask;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (cur_leaf_mask & 1) {
            const Point& leaf_pt = points[node.children[i]];
            const float p_dist2 = Point::distance2(query_pt, leaf_pt);
            if (p_dist2 < nearest_dist2) {
                nearest_dist2 = p_dist2;
                best_pt_idx = node.children[i];
            }
        }
        // Shift over now that this child has been checked
        cur_leaf_mask >>= 1;
    }
    return std::make_tuple(best_pt_idx, nearest_dist2);
}

__device__ float OT::nodeBoxDistance(
    const Point& query_pt,
    const OTNode& node,
    const Point* points) {
    // if node has a point as a child (leaf children)
    // if (node.child_leaf_mask) {
    //     return std::get<1>(nodePointDistance(query_pt, node, points));
    // }
    // no children, so use nearest corner on node bounding box
    // else {
        // boundaries of octree cell
        float x_bounds1 = node.corner.x;
        float x_bounds2 = node.corner.x + node.cell_size;
        float y_bounds1 = node.corner.y;
        float y_bounds2 = node.corner.y + node.cell_size;
        float z_bounds1 = node.corner.z;
        float z_bounds2 = node.corner.z + node.cell_size;
        // minimum distance to boundaries in each dimension
        float min_x = fminf(fabsf(query_pt.x - x_bounds1), fabsf(query_pt.x - x_bounds2));
        float min_y = fminf(fabsf(query_pt.y - y_bounds1), fabsf(query_pt.y - y_bounds2));
        float min_z = fminf(fabsf(query_pt.z - z_bounds1), fabsf(query_pt.z - z_bounds2));
        // total distance to nearest corner
        return min_x*min_x + min_y*min_y + min_z*min_z;
    // }
}
