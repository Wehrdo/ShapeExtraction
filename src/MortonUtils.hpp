#pragma once

#include <cstdint>

#include "PointCloud.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

typedef uint64_t Code_t;
constexpr int CODE_LEN = 63;

__host__ __device__ Code_t pointToCode(
    const float x,
    const float y,
    const float z,
    const float min_coord,
    const float range);

// converts a code to the original 3D point
__host__ __device__ Point codeToPoint(const Code_t code, const float min_coord, const float range);