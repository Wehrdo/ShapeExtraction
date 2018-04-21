#include <string>

#include "PointCloud.hpp"

class DataIO {
public:
    // loads binary Kitti point clouds
    static PointCloud<float> loadKitti(const std::string& filepath);

    // loads vertices from a standard ASCII obj file
    static PointCloud<float> loadObj(const std::string& filepath);
};