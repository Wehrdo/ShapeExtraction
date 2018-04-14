#pragma once

#include "RadixTree.hpp"

namespace OT {

struct OTNode {
    int parent;
    // TODO: This is overkill number of pointers
    int children[8];
    // whether each child exists. 8 bits for 8 children
    uint8_t child_mask;

    __device__ void setChild(size_t child, int my_child_idx) {
        children[my_child_idx] = child;
        child_mask |= (1 << my_child_idx);
    }
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
};
}