#include <iostream>

#include "KittiData.hpp"
#include "RadixTree.hpp"

using std::cout;
using std::endl;

int main() {
    auto cloud = KittiData::load("/home/dawehr/Development/pc_ransac/data/kitti/2011_09_26/2011_09_26_drive_0002_sync/velodyne_points/data/0000000000.bin");

    RT::RadixTree tree(cloud);

    // cloud.saveAsPly("kitti0.ply");

    // size_t n_pts = cloud.x_vals.size();
    // for (size_t i = 0; i < 50 && i < n_pts; ++i) {
    //     cout << cloud.x_vals[i] << ", " << cloud.y_vals[i] << ", " << cloud.z_vals[i] << endl;
    // }
}