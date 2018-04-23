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
    std::shared_ptr<std::vector<Point>> h_points;
    auto h_nodes = std::make_shared<std::vector<OT::OTNode>>(n_nodes);
    // copy nodes from device to host
    if (rank == 0) {
        CudaCheckCall(cudaMemcpy(&(*h_nodes)[0], octree.u_nodes, n_nodes * sizeof(*octree.u_nodes), cudaMemcpyDeviceToHost));
        h_points = octree.h_points;
    }
    else {
        // only allocate new space of not rank 0, because rank 0 already has this allocated
        h_points = std::make_shared<std::vector<Point>>(n_pts);
    }
    MPI_Bcast(&(*h_nodes)[0], n_nodes, OT::OTNode::getMpiDatatype(), 0, MPI_COMM_WORLD);
    MPI_Bcast(&(*h_points)[0], n_pts, Point::getMpiDatatype(), 0, MPI_COMM_WORLD);

    if (rank != 0) {
        octree = OT::Octree(h_nodes, n_nodes, h_points, n_pts);
    }

    // unneeded memory will be freed automatically, since we used shared pointers
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
    }

    // Share constructed octree
    bcastOctree(rank, octree);

    auto start_time = std::chrono::high_resolution_clock::now();
    int total_pts = octree.n_pts;
    std::vector<int> dist_counts(n_nodes);
    int start_idx = 0;
    int n_extra = total_pts % n_nodes;
    for (int i = 0; i < n_nodes; ++i) {
        dist_counts[i] = total_pts / n_nodes + (rank < n_extra);
        if (rank > i) {
            start_idx += dist_counts[i];
        }
    }
    std::cout << "Node " << rank << " start = " << start_idx << ", n = " << dist_counts[rank] << std::endl;
    auto local_normals = NormalEstimation::estimateNormals<8>(octree, start_idx, dist_counts[rank]);
    auto end_time = std::chrono::high_resolution_clock::now();

    std::vector<Point> all_normals(rank == 0 ? total_pts : 0);
    MPI_Gather(&local_normals[0], dist_counts[rank], Point::getMpiDatatype(), rank == 0 ? &all_normals[0] : nullptr, total_pts, Point::getMpiDatatype(), 0, MPI_COMM_WORLD);

    auto total_time = std::chrono::high_resolution_clock::now() - start_time;
    std::cout << "Node " << rank << ": " << "Normal estimation took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms" << std::endl;

    if (rank == 0) {
        std::cout << "Total normal estimation time took " << std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count() << "ms" << std::endl;

        PointCloud<float> output_cloud(octree.h_points, all_normals);
        output_cloud.saveAsPly("cloud.ply");
    }

    return MPI_Finalize();
}
