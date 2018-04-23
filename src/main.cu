#include "PointCloud.hpp"
#include "DataIO.hpp"
#include "RadixTree.hpp"
#include "Octree.hpp"
#include "NormalEstimation.hpp"
#include "CudaCommon.hpp"

#include <mpi.h>

#include <memory>
#include <iostream>
#include <chrono>

using std::cout;
using std::endl;

void bcastOctree(const int rank, OT::Octree& octree) {
    // broadcast number of points and nodes
    int n_pts = octree.n_pts;
    int n_nodes = octree.n_nodes;
    MPI_Bcast(&n_pts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // allocate points and nodes arrays on host
    auto h_points = std::make_shared<std::vector<Point>>(n_pts);
    auto h_nodes = std::make_shared<std::vector<OT::OTNode>>(n_nodes);
    // copy nodes from device to host
    if (rank == 0) {
        CudaCheckCall(cudaMemcpy(&(*h_nodes)[0], octree.u_nodes, n_nodes * sizeof(*octree.u_nodes), cudaMemcpyDeviceToHost));
        h_points = octree.h_points;
    }
    MPI_Bcast(&(*h_nodes)[0], n_nodes, OT::OTNode::getMpiDatatype(), 0, MPI_COMM_WORLD);
    MPI_Bcast(&(*h_points)[0], n_pts, Point::getMpiDatatype(), 0, MPI_COMM_WORLD);

    octree = OT::Octree(h_nodes, n_nodes, h_points, n_pts);

    // necessary memory will be freed automatically, since we used shared pointers
}

int main() {
    MPI_Init(NULL, NULL);
    int n_nodes, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    OT::Octree octree;
    // Only one node needs to compute the octree
    if (rank == 0) {
        // auto input_cloud = DataIO::loadKitti("../../data/kitti/2011_09_26/2011_09_26_drive_0002_sync/velodyne_points/data/0000000000.bin");
        auto input_cloud = DataIO::loadObj("../../data/test_sphere.obj");
        std::cout << "Input data has " << input_cloud.x_vals.size() << " points" << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();
        RT::RadixTree radix_tree(input_cloud);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::cout << "Radix tree construction took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms" << std::endl;

        start_time = std::chrono::high_resolution_clock::now();
        octree = OT::Octree(radix_tree);
        end_time = std::chrono::high_resolution_clock::now();
        std::cout << "Octree construction took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms" << std::endl;

        // std::vector<Point> search_pts({
        //     // Point(50.7, 0.5, 1.9),
        //     // Point(52, 0, 3),
        //     // Point(50.583, 1.29, 1.92)
        //     Point(1, 2, 3),
        //     Point(-9, -1.3, -1.5),
        //     Point(30, 5, 16)
        // });
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    auto normals = NormalEstimation::estimateNormals<8>(octree);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Normal estimation took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms" << std::endl;

    
    PointCloud<float> output_cloud(octree.h_points, normals);
    output_cloud.saveAsPly("cloud.ply");

    return MPI_Finalize();
}
