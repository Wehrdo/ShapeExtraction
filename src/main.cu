#include <iostream>

#include "PointCloud.hpp"
#include "KittiData.hpp"
#include "RadixTree.hpp"
#include "Octree.hpp"

using std::cout;
using std::endl;

int main() {
    auto cloud = KittiData::load("../../data/kitti/2011_09_26/2011_09_26_drive_0002_sync/velodyne_points/data/0000000000.bin");

    RT::RadixTree radix_tree(cloud);
    OT::Octree octree(radix_tree);

    std::vector<Point> search_pts({
        Point(50.7, 0.5, 1.9),
        Point(52, 0, 3),
        Point(50.583, 1.29, 1.92)
    });

    auto results = octree.knnSearch<1>(search_pts);
    for (const auto& pt_idxs : results) {
        const Point& pt = octree.h_points[pt_idxs[0]];
        std::cout << pt.x << ", " << pt.y << ", " << pt.z << std::endl;
    }

    // cloud.saveAsPly("kitti0.ply");

    // size_t n_pts = cloud.x_vals.size();
    // for (size_t i = 0; i < 50 && i < n_pts; ++i) {
    //     cout << cloud.x_vals[i] << ", " << cloud.y_vals[i] << ", " << cloud.z_vals[i] << endl;
    // }
    return 0;
}