#pragma once

#include "RadixTree.hpp"

namespace OT {

struct OTNode {
    int parent;
    // TODO: This is overkill number of pointers
    int children[8];
    // For bit position i (from the right):
    //     If 1, children[i] is the index of a child octree node
    //     If 0, the ith child is either absent, or children[i] is the index of a leaf.
    //           If children[i] is -1, then child is absent
    int child_mask;

    __device__ void setChild(size_t child, int my_child_idx);
};

class Octree {
public:
    Octree(const RT::RadixTree& radix_tree);
    ~Octree();
private:
    // caching device allocator for CUB temporary storage
    cub::CachingDeviceAllocator g_allocator;

    // Number of octree nodes between a node and its parent
    int* edgeNodes;

    OTNode* nodes;
    // prefix for the root node
    RT::Code_t root_prefix;
};
}