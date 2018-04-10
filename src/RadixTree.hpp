#pragma once

#include "PointCloud.hpp"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "cub/util_allocator.cuh"

#include <cstdint>

/*
 * Implementation of the algorithm described by Karras in
 * "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees"
 */

namespace RT {
typedef uint64_t Code_t;
struct Nodes {
    // 63-bit morton code, packed to the right
    Code_t* mortonCode;

    // Flag determining whether this is a leaf node or not
    // TODO: Store "leaf" flag in first bit of mortonCode
    bool* hasLeafLeft;
    bool* hasLeafRight;

    // The number of bits in the mortonCode this node represents
    // Corresponds to delta_node in [Karras]
    uint8_t* prefixN;
    
    // Index of left child of this node
    // Right child is leftChild + 1
    size_t* leftChild;
    // Index of parent
    size_t* parent;
};

class RadixTree {
public:
    RadixTree(const PointCloud<float>& cloud);
    ~RadixTree();
    // radix tree on GPU
    struct Nodes d_tree;
    // Number of tree nodes
    int n_nodes;
private:
    // Encodes point cloud into mortonCode array of d_tree
    void encodePoints(const PointCloud<float>& cloud);

    // caching device allocator for CUB temporary storage
    cub::CachingDeviceAllocator g_allocator;

	// n_pts is an int only to match CUB type definitions.
    int n_pts; // number of points
};

}