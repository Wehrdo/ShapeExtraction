#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>


#ifdef __CUDACC__
#define POINT_CONST_PREFIX __host__ __device__
#else
#define POINT_CONST_PREFIX
#endif


struct Point {
    POINT_CONST_PREFIX Point(float x, float y, float z) : x(x), y(y), z(z) {}
    float x, y, z;
};

template <typename T>
class PointCloud {
public:
    std::vector<T> x_vals;
    std::vector<T> y_vals;
    std::vector<T> z_vals;

    bool saveAsPly(std::string filename);
private:
};

template <typename T>
bool PointCloud<T>::saveAsPly(std::string filename) {
    assert(x_vals.size() == y_vals.size());
    assert(x_vals.size() == z_vals.size());

    std::ofstream of;
    of.open(filename);
    if (!of) { return false; }

    of << "ply" << std::endl;
    of << "format ascii 1.0" << std::endl;

    of << "element vertex " << x_vals.size() << std::endl;
    of << "property float x" << std::endl;
    of << "property float y" << std::endl;
    of << "property float z" << std::endl;
    of << "end_header" << std::endl;
    for (size_t i = 0; i < x_vals.size(); ++i) {
        of << x_vals[i] << " " << y_vals[i] << " " << z_vals[i] << std::endl;
    }

    of.close();
    return true;
}