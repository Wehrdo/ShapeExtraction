#include "KittiData.hpp"

#include <iostream>
#include <memory>

PointCloud<float> KittiData::load(std::string filepath) {
    PointCloud<float> cloud;

    size_t num = 1000000;
    // float* data = (float*)malloc(num * sizeof(float));
    std::unique_ptr<float> data(new float[num]);

    auto stream = std::fopen(filepath.c_str(), "rb");
    num = fread(data.get(), sizeof(float), num, stream) / 4;
    // num = 10;
    float* px = data.get() + 0;
    float* py = data.get() + 1;
    float* pz = data.get() + 2;
    // r is reflectance
    float* pr = data.get() + 3;
    for (size_t i = 0; i < num; ++i) {
        cloud.x_vals.push_back(*px);
        cloud.y_vals.push_back(*py);
        cloud.z_vals.push_back(*pz);
        px += 4; py += 4; pz += 4; pr += 4;
    }
    fclose(stream);

    return cloud;
}