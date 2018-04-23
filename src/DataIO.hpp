#include <string>

#include "PointCloud.hpp"

class DataIO {
public:
    // loads binary Kitti point clouds num is maximum number of floats to read (4 * number of points)
    static PointCloud<float> loadKitti(const std::string& filepath, size_t num);
    // saves a point cloud as Kitti format. Convenient, because it's binary
    static void saveKitti(const PointCloud<float> cloud, const std::string& filepath);

    // loads vertices from a standard ASCII obj file
    static PointCloud<float> loadObj(const std::string& filepath);

    // loads vertices from Semantic3D data files
    static PointCloud<float> loadSemantic3D(const std::string& filepath);
};