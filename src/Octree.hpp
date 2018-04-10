#pragma once

#include "RadixTree.hpp"

typedef uint8_t edgeNodeCnt_t;
class Octree {
public:
    Octree(const RT::RadixTree& radix_tree);
    ~Octree();
private:
    // caching device allocator for CUB temporary storage
    cub::CachingDeviceAllocator g_allocator;

    // Number of octree nodes between a node and its parent
    edgeNodeCnt_t* d_edgeNodes;
};