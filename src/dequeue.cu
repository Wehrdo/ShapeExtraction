#include "dequeue.hpp"

__host__ __device__ int int_log2(int x) {
    int ret = 0;
    while (x >>= 1) ++ret;
    return ret;
}