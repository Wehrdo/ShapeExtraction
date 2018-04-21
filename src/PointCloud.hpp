#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>


// code location prefix. When compiling as CUDA, makes available on host and device
#ifdef __CUDACC__
#define POINT_LOC_PREFIX __host__ __device__
#else
#define POINT_LOC_PREFIX
#endif

struct Point {
    POINT_LOC_PREFIX Point(float x, float y, float z) : x(x), y(y), z(z) {}
    POINT_LOC_PREFIX Point() : x(0), y(0), z(0) {}

    // members
    float x, y, z;

    // the square distance between two points
    POINT_LOC_PREFIX static float distance2(const Point& p1, const Point& p2) {
        float x_diff = p1.x - p2.x;
        float y_diff = p1.y - p2.y;
        float z_diff = p1.z - p2.z;
        return x_diff*x_diff + y_diff*y_diff + z_diff*z_diff;
    }
    
    POINT_LOC_PREFIX float dot(const Point& other) const {
        return x*other.x + y*other.y + z*other.z;
    }

    POINT_LOC_PREFIX Point& operator+=(const Point& rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }

    POINT_LOC_PREFIX friend Point operator+(Point lhs, const Point& rhs) {lhs += rhs; return lhs;}

    POINT_LOC_PREFIX Point& operator/=(const float rhs) {
        x /= rhs;
        y /= rhs;
        z /= rhs;
        return *this;
    }

    POINT_LOC_PREFIX Point& operator-=(const Point& rhs) {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        return *this;
    }

    POINT_LOC_PREFIX friend Point operator-(Point lhs, const Point& rhs) {lhs -= rhs; return lhs;}
};

template <typename T>
class PointCloud {
public:
    std::vector<T> x_vals;
    std::vector<T> y_vals;
    std::vector<T> z_vals;

    std::vector<T> x_normals;
    std::vector<T> y_normals;
    std::vector<T> z_normals;

    bool saveAsPly(std::string filename);
    void setNormals(std::vector<Point> normals);
private:
};

template <typename T>
bool PointCloud<T>::saveAsPly(std::string filename) {
    assert(x_vals.size() == y_vals.size());
    assert(x_vals.size() == z_vals.size());

    bool has_normals = x_normals.size() != 0;
    if (has_normals) {
        assert(x_vals.size() == x_normals.size());
        assert(x_vals.size() == y_normals.size());
        assert(x_vals.size() == z_normals.size());
    }

    std::ofstream of;
    of.open(filename);
    if (!of) { return false; }

    of << "ply" << std::endl;
    of << "format ascii 1.0" << std::endl;

    of << "element vertex " << x_vals.size() << std::endl;
    of << "property float x" << std::endl;
    of << "property float y" << std::endl;
    of << "property float z" << std::endl;
    if (has_normals) {
        of << "property float nx" << std::endl;
        of << "property float ny" << std::endl;
        of << "property float nz" << std::endl;
    }
    of << "end_header" << std::endl;
    for (size_t i = 0; i < x_vals.size(); ++i) {
        of << x_vals[i] << " " << y_vals[i] << " " << z_vals[i];
        if (has_normals) {
            of << " " << x_normals[i] << " " << y_normals[i] << " " << z_normals[i];
        }
        of << std::endl;
    }

    of.close();
    return true;
}

template <>
void PointCloud<float>::setNormals(std::vector<Point> normals);

template <typename T>
void PointCloud<T>::setNormals(std::vector<Point> normals) {
    assert(false); // not implemented for non-floats, since Point object holds floats
}
