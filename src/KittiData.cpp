#include <iostream>

#include "KittiData.hpp"


PointCloud<float> KittiData::load(std::string filepath) {
    PointCloud<float> cloud;

    size_t num = 1000000;
    float* data = (float*)malloc(num * sizeof(float));

    auto stream = std::fopen(filepath.c_str(), "rb");
    num = fread(data, sizeof(float), num, stream) / 4;
    float* px = data + 0;
    float* py = data + 1;
    float* pz = data + 2;
    // r is reflectance
    float* pr = data + 3;
    for (size_t i = 0; i < num; ++i) {
        cloud.x_vals.push_back(*px);
        cloud.y_vals.push_back(*py);
        cloud.z_vals.push_back(*pz);
        px += 4; py += 4; pz += 4; pr += 4;
    }
    fclose(stream);

    return cloud;
}