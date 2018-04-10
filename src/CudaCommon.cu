#include "CudaCommon.hpp"


 std::tuple<int, int> makeLaunchParams(size_t n, int tpb) {
    // int tpb = 256;
    int blocks = (n + tpb - 1) / tpb;
    return std::make_tuple(blocks, tpb);
 }