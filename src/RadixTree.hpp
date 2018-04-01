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
class Node {
    // 63-bit morton code, packed to the right
    uint64_t mortonCode;

    // Flag determining whether this is a leaf node or not
    // TODO: Store "leaf" flag in first bit of mortonCode
    bool leaf;

    // The number of bits in the mortonCode this node represents
    // Corresponds to delta_node in [Karras]
    uint8_t prefixN;
    
    // Children of this node
    Node* leftChild;
    Node* rightChild;
};

class RadixTree {
public:
    RadixTree(const PointCloud<float>& cloud);
    ~RadixTree();
private:
    cub::CachingDeviceAllocator g_allocator;
    std::unique_ptr<Node> h_tree;
    Node* d_tree;

};

}