#pragma once

#include <cstdint>

#include "PointCloud.hpp"

typedef uint64_t Code_t;
constexpr int CODE_LEN = 63;

__device__ Code_t pointToCode(
    const float x,
    const float y,
    const float z,
    const float min_coord,
    const float range);

__device__ Point codeToPoint(const Code_t code, const float min_coord, const float range);