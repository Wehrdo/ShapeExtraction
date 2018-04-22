#include <iostream>
#include <chrono>

#include "PointCloud.hpp"
#include "DataIO.hpp"
#include "RadixTree.hpp"
#include "Octree.hpp"
#include "NormalEstimation.hpp"

using std::cout;
using std::endl;

int main() {
    auto input_cloud = DataIO::loadKitti("../../data/kitti/2011_09_26/2011_09_26_drive_0002_sync/velodyne_points/data/0000000000.bin");
    // auto input_cloud = DataIO::loadObj("../../data/test_sphere_small.obj");
    std::cout << "Input data has " << input_cloud.x_vals.size() << " points" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    RT::RadixTree radix_tree(input_cloud);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Radix tree construction took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    OT::Octree octree(radix_tree);
    end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Octree construction took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms" << std::endl;

    std::vector<Point> search_pts({
        // Point(50.7, 0.5, 1.9),
        // Point(52, 0, 3),
        // Point(50.583, 1.29, 1.92)
        Point(1, 2, 3),
        Point(-9, -1.3, -1.5),
        Point(30, 5, 16)
    });

    // auto results = octree.knnSearch<2>(search_pts);
    // for (const auto& pt_idxs : results) {
    //     for (int pt_idx : pt_idxs) {
    //         const Point& pt = octree.h_points[pt_idx];
    //         std::cout << pt.x << ", " << pt.y << ", " << pt.z << std::endl;
    //     }
    //     std::cout << std::endl;
    // }

    start_time = std::chrono::high_resolution_clock::now();
    auto normals = NormalEstimation::estimateNormals<8>(octree);
    end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Normal estimation took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms" << std::endl;

    
    PointCloud<float> output_cloud(octree.h_points, normals);
    output_cloud.saveAsPly("cloud.ply");

    return 0;
}
