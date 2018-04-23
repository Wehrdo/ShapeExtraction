#pragma once

#include "RadixTree.hpp"
#include "PointCloud.hpp"
#include "CudaCommon.hpp"
#include "dequeue.hpp"

#include <mpi.h>

#include <array>
#include <vector>
#include <tuple>
#include <memory>

// #define SEARCH_Q_SIZE (32)
namespace OT {
constexpr static int SEARCH_Q_SIZE = 32;

struct OTNode {
    // int parent;
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

    static MPI_Datatype getMpiDatatype() {
        static_assert(std::is_standard_layout<OTNode>::value, "OTNode is not standard layout");

        static bool mpi_datatype_initialized = 0;
        static MPI_Datatype mpi_datatype;

        if (!mpi_datatype_initialized) {
            constexpr int n_elems = 5;
            const int block_lengths[n_elems] = {8, 1, 1, 1, 1};
            const MPI_Aint displacements[n_elems] = {offsetof(OTNode, children),
                                                     offsetof(OTNode, corner),
                                                     offsetof(OTNode, cell_size),
                                                     offsetof(OTNode, child_node_mask),
                                                     offsetof(OTNode, child_leaf_mask),
                                                     };
            const MPI_Datatype types[n_elems] = {MPI_INT,
                                                 Point::getMpiDatatype(),
                                                 MPI_FLOAT,
                                                 MPI_INT,
                                                 MPI_INT};
            MPI_Type_create_struct(
                n_elems,
                block_lengths,
                displacements,
                types,
                &mpi_datatype
            );
            mpi_datatype_initialized = 1;
        }
        return mpi_datatype;
    }
};

class Octree {
public:
    Octree() {};
    // Constructor for building octree from radix tree
    Octree(const RT::RadixTree& radix_tree);
    // Constructor for making octree from given host data (For passing Octree via MPI)
    Octree(std::shared_ptr<std::vector<OTNode>> _nodes, const int _n_nodes, std::shared_ptr<std::vector<Point>> _points, const int _n_pts);
    ~Octree();
    // move assignment operator
    Octree& operator=(Octree&& other);

    template <int k>
    std::vector<std::array<int, k>> knnSearch(const std::vector<Point>& points, const float eps=0.01) const;

    // Performs knn search and stores the k nearest neighbors in pre-allocated device memory d_nn_indices
    template <int k>
    void deviceKnnSearch(const Point* d_query, int* d_nn_indices, const int N, const float eps) const;

    int n_pts;
    // points, as points converted back from morton codes (in unified memory)
    Point* u_points = nullptr;

    // points in host memory
    std::shared_ptr<std::vector<Point>> h_points;

    // numer of ofctree nodes
    int n_nodes;
    // the octree
    OTNode* u_nodes = nullptr;
private:
    // caching device allocator for CUB temporary storage
    cub::CachingDeviceAllocator g_allocator;

    // prefix for the root node
    // Code_t root_prefix;
};

/*
 // --- Templated kernels ---
*/
// pre-declarations of functions for kernel
__device__ float nodeBoxDistance(
    const Point& query_pt,
    const OTNode& node);

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
        const Point query = query_pts[idx];
        PriorityQueue<const OTNode*, SEARCH_Q_SIZE> search_q;
        PriorityQueue<int, k> result_q;
        // start with root in queue
        search_q.insert(octree, 0);
        // distance explored so far
        float r = 0;
        // best distance so far
        float d = INFINITY;
        while (d >= (1 + eps) * r && search_q.size) {
            auto queue_top = search_q.removeMin();
            const OTNode& node = *queue_top.data;
            r = queue_top.priority;
            // if this cell contains a point that's better than best found so far, choose it
            if (node.child_leaf_mask) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    if (node.child_leaf_mask & (1 << i)) {
                        const Point& leaf_pt = all_pts[node.children[i]];
                        const float p_dist2 = Point::distance2(query, leaf_pt);
                        result_q.insert(node.children[i], p_dist2);
                    }
                }
            }
            // now add all non-leaf children of this node, with priority as their closest possible distance
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (node.child_node_mask & (1 << i)) {
                    const OTNode& candidate_node = octree[node.children[i]];
                    search_q.insert(&candidate_node,
                                 nodeBoxDistance(query, candidate_node));
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < k; ++i) {
            // if not enough nearest points found, just repeat the closest
            if (result_q.size == 0) {
                result_pts[idx*k + i] = result_pts[idx*k];
            }
            else {
                auto top_result = result_q.removeMin();
                result_pts[idx*k + i] = top_result.data;
            }
        }
    }
}


/*
  // --- Class template implementations ---
*/
template <int k>
void Octree::deviceKnnSearch(const Point* d_query, int* d_nn_indices, const int N, const float eps) const {
    assert(u_nodes != nullptr && u_points != nullptr);
    int blocks, tpb;
    std::tie(blocks, tpb) = makeLaunchParams(N);
    knnSearchKernel<k><<<blocks, tpb>>>(u_nodes,
                                  u_points,
                                  d_query,
                                  d_nn_indices,
                                  eps,
                                  N);
    cudaDeviceSynchronize();
    CudaCheckError();
}

template <int k>
std::vector<std::array<int, k>> Octree::knnSearch(const std::vector<Point>& points, const float eps) const {
    assert(u_nodes != nullptr && u_points != nullptr);
    const int n = static_cast<int>(points.size());
    assert(points.size() <= std::numeric_limits<decltype(n)>::max());

    // query points;
    Point* d_query;
    // allocate device storage 
    CudaCheckCall(cudaMalloc(&d_query, n * sizeof(*d_query)));
    // transfer query points to device memory
    CudaCheckCall(cudaMemcpy(d_query, &points[0], n * sizeof(Point), cudaMemcpyHostToDevice));

    std::vector<std::array<int, k>> results;
    // results in unified memory. each array of k elements is stored back-to-back
    int* d_results;
    CudaCheckCall(cudaMallocManaged(&d_results, n * k * sizeof(*d_results)));

    deviceKnnSearch<k>(d_query, d_results, eps, n);

    results.resize(points.size());
    // std::vector and std::array are contiguous, so we can just memcpy
    CudaCheckCall(cudaMemcpy(&results[0], d_results, n * k * sizeof(*d_results), cudaMemcpyDeviceToHost));

    CudaCheckCall(cudaFree(d_results));
    CudaCheckCall(cudaFree(d_query));

    return results;
}

}
