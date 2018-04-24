#include "DataIO.hpp"

#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <ios>

#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

PointCloud<float> DataIO::loadFile(const std::string& filepath) {
    auto endsWith = [&] (const std::string ext) {
        return filepath.rfind(ext) == filepath.length() - ext.length();
    };

    if (endsWith(".obj")) {
        return loadObj(filepath);
    }
    if (endsWith(".bin")) {
        // get size of file
        fs::path p{filepath};
        p = fs::canonical(p);
        size_t length = fs::file_size(p);
        return loadKitti(filepath, length);
    }
    if (endsWith(".txt")) {
        return loadSemantic3D(filepath);
    }
}

PointCloud<float> DataIO::loadKitti(const std::string& filepath, size_t num) {
    PointCloud<float> cloud;

    // size_t num = 1000000;
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

void DataIO::saveKitti(const PointCloud<float> cloud, const std::string& filepath) {
    std::ofstream stream(filepath, std::ios::out | std::ios::binary);

    // assert(4 * cloud.x_vals.size() <= std::numeric_limits<int32_t>::max());
    // int32_t cloud_size = static_cast<int32_t>(cloud.x_vals.size()) * 4;
    // stream.write((char*) &cloud_size, sizeof(cloud_size));


    float reflectance = 255;
    for (int i = 0; i < cloud.x_vals.size(); ++i) {
        stream.write((char*) &cloud.x_vals[i], sizeof(float));
        stream.write((char*) &cloud.y_vals[i], sizeof(float));
        stream.write((char*) &cloud.z_vals[i], sizeof(float));
        stream.write((char*) &reflectance, sizeof(float));
    }

    stream.close();
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

PointCloud<float> DataIO::loadSemantic3D(const std::string& filepath) {
    PointCloud<float> cloud;

    std::ifstream file(filepath);
    std::string line;
    while (std::getline(file, line)) {
        float x, y, z, r, g, b;
        std::istringstream iss(line);
        if (!(iss >> x >> y >> z >> r >> g >> b)) {
            std::cout << "Failed to parse line " << line << std::endl;
        }
        else {
            cloud.x_vals.push_back(x);
            cloud.y_vals.push_back(y);
            cloud.z_vals.push_back(z);
        }
    }

    return cloud;
}