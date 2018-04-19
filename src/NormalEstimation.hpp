#pragma once

#include "PointCloud.hpp"
#include "Octree.hpp"
#include "CudaCommon.hpp"

#include <tuple>

class NormalEstimation {
public:
    template <int k>
    static std::vector<Point> estimateNormals(const OT::Octree& octree, const std::vector<Point>& points);
};

template <int k>
__global__ void estimateNormalsKernel(
    Point* normals,
    const Point* points,
    const int* nn_indices,
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) { return; }
    
    const int nn_idx = idx * k;

    // find centroid of points
    Point centroid;
    for (int i = 0; i < k; ++i) {
        centroid += points[nn_indices[nn_idx + i]];
    }
    centroid /= k;

    normals[idx] = centroid;
}

/*
  // --- Class template implementations ---
*/
template <int k>
std::vector<Point> NormalEstimation::estimateNormals(const OT::Octree& octree, const std::vector<Point>& points) {
    // first find nearest neighbors
    const int n = static_cast<int>(points.size());
    assert(points.size() <= std::numeric_limits<decltype(n)>::max());

    // nearest neighbors in unified memory
    int* u_nn;
    CudaCheckCall(cudaMallocManaged(&u_nn, n * k * sizeof(*u_nn)));
    // normals in unified memory
    Point* u_normals;
    CudaCheckCall(cudaMallocManaged(&u_normals, n * k * sizeof(*u_normals)));

    octree.deviceKnnSearch<k>(points, u_nn, 0.01f);

    int blocks, tpb;
    std::tie(blocks, tpb) = makeLaunchParams(n);

    estimateNormalsKernel<k><<<blocks, tpb>>>(u_normals, octree.u_points, u_nn, n);

    cudaDeviceSynchronize();
    CudaCheckError();

    std::vector<Point> results(n);
    CudaCheckCall(cudaMemcpy(&results[0], u_normals, n * sizeof(*u_normals), cudaMemcpyDeviceToHost));
    return results;
}
