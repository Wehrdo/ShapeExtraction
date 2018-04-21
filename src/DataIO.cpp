#include "DataIO.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>

PointCloud<float> DataIO::loadKitti(const std::string& filepath) {
    PointCloud<float> cloud;

    size_t num = 1000000;
    // float* data = (float*)malloc(num * sizeof(float));
    std::unique_ptr<float> data(new float[num]);

    auto stream = std::fopen(filepath.c_str(), "rb");
    num = fread(data.get(), sizeof(float), num, stream) / 4;
    // num = 10000;
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

PointCloud<float> DataIO::loadObj(const std::string& filepath) {
    PointCloud<float> cloud;

    std::ifstream file(filepath);
    std::string line;
    while (std::getline(file, line)) {
        if (line.length() > 1 && line[0] == 'v' && line[1] == ' ') {
            float x, y, z;
            std::string v;
            std::istringstream iss(line);
            if (!(iss >> v >> x >> y >> z)) {
                std::cout << "Failed to parse vertex line " << line << std::endl;
            }
            else {
                cloud.x_vals.push_back(x);
                cloud.y_vals.push_back(y);
                cloud.z_vals.push_back(z);
            }
        }
    }

    return cloud;
}