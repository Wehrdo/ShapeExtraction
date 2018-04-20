#include "PointCloud.hpp"

template <>
void PointCloud<float>::setNormals(std::vector<Point> normals) {
    assert(normals.size() == x_vals.size());
    x_normals.resize(normals.size());
    y_normals.resize(normals.size());
    z_normals.resize(normals.size());
    for (size_t i = 0; i < normals.size(); ++i) {
        x_normals[i] = normals[i].x;
        y_normals[i] = normals[i].y;
        z_normals[i] = normals[i].z;
    }
}
