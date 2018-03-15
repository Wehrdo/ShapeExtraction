#include <string>

#include "PointCloud.hpp"

class KittiData {
public:
    static PointCloud<float> load(std::string filename);
};