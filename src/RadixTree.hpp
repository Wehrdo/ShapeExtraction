#pragma once

#include "PointCloud.hpp"
#include "MortonUtils.hpp"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "cub/util_allocator.cuh"

#include <cstdint>

/*
 * Implementation of the algorithm described by Karras in
 * "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees"
 */

namespace RT {
struct Nodes {
    // 63-bit morton code, packed to the right
    Code_t* mortonCode;

    // Flags determining whether the left and right children are leaves
    bool* hasLeafLeft;
    bool* hasLeafRight;

    // The number of bits in the mortonCode this node represents
    // Corresponds to delta_node in [Karras]
    uint8_t* prefixN;
    
    // Index of left child of this node
    // Right child is leftChild + 1
    int* leftChild;
    // Index of parent
    int* parent;
};

class RadixTree {
public:
    RadixTree(const PointCloud<float>& cloud);
    ~RadixTree();
    // radix tree on GPU
    struct Nodes d_tree;
	// n_pts is an int only to match CUB type definitions.
    int n_pts; // number of points

    // Number of tree nodes (n_pts - 1)
    int n_nodes;

    // minimum and maximum coordinate in points.
    // Represents the scaling factor for the morton codes
    float min_coord, max_coord;
private:
    // Encodes point cloud into mortonCode array of d_tree
    void encodePoints(const PointCloud<float>& cloud);

    // caching device allocator for CUB temporary storage
    cub::CachingDeviceAllocator g_allocator;
};

}