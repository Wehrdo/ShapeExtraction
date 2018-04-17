#pragma once

#include "RadixTree.hpp"
#include "PointCloud.hpp"
#include "CudaCommon.hpp"
#include "dequeue.hpp"

#include <array>
#include <vector>
#include <tuple>

// #define SEARCH_Q_SIZE (32)
namespace OT {
constexpr static int SEARCH_Q_SIZE = 32;

struct OTNode {
    int parent;
    // TODO: This is overkill number of pointers
    int children[8];

    // bounding box of cell is defined by minimum (x, y, z) coordinate of cell and depth in tree
    Point corner;
    float cell_size;

    // For bit position i (from the right):
    //     If 1, children[i] is the index of a child octree node
    //     If 0, the ith child is either absent, or children[i] is the index of a leaf.
    int child_node_mask;
    // For bit position i (from the right):
    //     If 1, children[i] is the index of a leaf (in the corresponding points array)
    //     If 0, the ith child is either absent, or an octree node.
    int child_leaf_mask;

    // Set a child
    //     child: index of octree node that will become the child
    //     my_child_idx: which of my children it will be [0-7]
    __device__ void setChild(const int child, const int my_child_idx);
    // Set a leaf child
    //     leaf: index of point that will become the leaf child
    //     my_child_idx; which of my children it will be [0-7]
    __device__ void setLeaf(const int leaf, const int my_child_idx);
};

class Octree {
public:
    Octree(const RT::RadixTree& radix_tree);
    ~Octree();
    template <int k>
    std::vector<std::array<int, k>> knnSearch(const std::vector<Point>& points);

    // points in host memory
    Point* h_points;
private:
    // caching device allocator for CUB temporary storage
    cub::CachingDeviceAllocator g_allocator;

    // the octree
    OTNode* nodes;

    // points, as points converted back from morton codes (in unified memory)
    Point* u_points;
    // prefix for the root node
    Code_t root_prefix;
};

/*
 // --- Templated kernels ---
*/
// pre-declarations of functions for kernel
__device__ float nodeBoxDistance(
    const Point& query_pt,
    const OTNode& node);
__device__ std::tuple<int, float> nodePointDistance(
    const Point& query_pt,
    const OTNode& node,
    const Point* points);

template <int k>
__global__ void knnSearchKernel(
    const OTNode* octree,
    const Point* all_pts,
    const Point* query_pts,
    int* result_pts,
    const float eps,
    const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int k_found = 0;
        int best_pt = -1;
        const Point query = query_pts[idx];
        PriorityQueue<OTNode, SEARCH_Q_SIZE> queue;
        // start with root in queue
        queue.insert(octree, 0);
        // distance explored so far
        float r = 0;
        // best distance so far
        float d = INFINITY;
        while (d >= (1 + eps) * r && queue.size) {
            auto queue_top = queue.removeMin();
            const OTNode& node = *queue_top.data;
            r = queue_top.priority;
            // if this cell contains a point that's better than best found so far, choose it
            if (node.child_leaf_mask) {
                // TODO: Check all contained points for knn
                //       Idea: Make a PriorityQueue of size k, and try to add all contained points
                auto close_pt = nodePointDistance(query, node, all_pts);
                float candidate_dist = std::get<1>(close_pt);
                if (candidate_dist < d) {
                    best_pt = std::get<0>(close_pt);
                    d = candidate_dist;
                }
            }
            // now add all non-leaf children of this node, with priority as their closest possible distance
            for (int i = 0; i < 8; ++i) {
                if (node.child_node_mask & (1 << i)) {
                    const OTNode& candidate_node = octree[node.children[i]];
                    queue.insert(&candidate_node,
                                 nodeBoxDistance(query, candidate_node));
                }
            }
        }

        result_pts[idx * k] = best_pt;

    }
}


/*
  // --- Class template implementations ---
*/
template <int k>
std::vector<std::array<int, k>> Octree::knnSearch(const std::vector<Point>& points) {
    const int n = static_cast<int>(points.size());
    assert(points.size() <= std::numeric_limits<decltype(n)>::max());
    std::vector<std::array<int, k>> results;
    // results in unified memory. each array of k elements is stored back-to-back
    int* d_results;
    // query points;
    Point* d_query;
    // allocate device storage 
    CudaCheckCall(cudaMallocManaged(&d_results, n * k * sizeof(*d_results)));
    CudaCheckCall(cudaMalloc(&d_query, n * sizeof(*d_query)));
    // transfer query points to device memory
    CudaCheckCall(cudaMemcpy(d_query, &points[0], n * sizeof(Point), cudaMemcpyHostToDevice));

    int blocks, tpb;
    std::tie(blocks, tpb) = makeLaunchParams(n);
    knnSearchKernel<k><<<blocks, tpb>>>(nodes,
                                  u_points,
                                  d_query,
                                  d_results,
                                  0.0f,
                                  n);
    cudaDeviceSynchronize();
    CudaCheckError();

    results.resize(points.size());
    CudaCheckCall(cudaMemcpy(&results[0], d_results, n * k * sizeof(*d_results), cudaMemcpyDeviceToHost));


    CudaCheckCall(cudaFree(d_query));
    CudaCheckCall(cudaFree(d_results));

    return results;
}

}
