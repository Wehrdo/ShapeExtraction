#pragma once

#include "RadixTree.hpp"
#include "PointCloud.hpp"

namespace OT {

struct OTNode {
    int parent;
    // TODO: This is overkill number of pointers
    int children[8];


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
private:
    // caching device allocator for CUB temporary storage
    cub::CachingDeviceAllocator g_allocator;

    // the octree
    OTNode* nodes;

    // points, as points converted back from morton codes (in unified memory)
    Point* u_points;
    // points in host memory
    Point* h_points;
    // prefix for the root node
    Code_t root_prefix;
};
}