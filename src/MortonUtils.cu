#include "MortonUtils.hpp"

#include "libmorton/include/morton.h"

#include <cuda.h>

__host__ __device__ Code_t pointToCode(
    const float x,
    const float y,
    const float z,
    const float min_coord,
    const float range) {
    const uint32_t bitscale = 0xFFFFFFFFu >> (32 - (CODE_LEN / 3));
    const uint32_t x_coord = static_cast<uint32_t>(bitscale * ((x - min_coord) / range));
    const uint32_t y_coord = static_cast<uint32_t>(bitscale * ((y - min_coord) / range));
    const uint32_t z_coord = static_cast<uint32_t>(bitscale * ((z - min_coord) / range));
    // printf("Point %lu = (%u, %u, %u)\n", (unsigned long)idx, (unsigned int)x_coord, (unsigned int)y_coord, (unsigned int)z_coord);
    return morton3D_64_encode(x_coord, y_coord, z_coord);
}

__host__ __device__ Point codeToPoint(const Code_t code, const float min_coord, const float range) {
        const uint32_t bitscale = 0xFFFFFFFFu >> (32 - (CODE_LEN / 3));
        uint32_t dec_raw_x, dec_raw_y, dec_raw_z;
        morton3D_64_decode(code, dec_raw_x, dec_raw_y, dec_raw_z);
        float dec_x = ((float)dec_raw_x / bitscale) * range + min_coord;
        float dec_y = ((float)dec_raw_y / bitscale) * range + min_coord;
        float dec_z = ((float)dec_raw_z / bitscale) * range + min_coord;
        return Point(dec_x, dec_y, dec_z);
}