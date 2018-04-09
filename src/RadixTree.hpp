#include "PointCloud.hpp"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "cub/util_allocator.cuh"

#include <cstdint>
#include <memory>

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

    // Number of octree nodes between this node and its parent
    uint8_t* edgeNode;
};

class RadixTree {
public:
    RadixTree(const PointCloud<float>& cloud);
    ~RadixTree();
private:
    void encodePoints(const PointCloud<float>& cloud);

    cub::CachingDeviceAllocator g_allocator;

    // device tree
    struct Nodes d_tree;
	// n_pts is an int only to match CUB type definitions.
    int n_pts; // number of points
    // Node* d_tree_internal;
    // float* d_data_x;
    // float* d_data_y;
    // float* d_data_z;

};

}