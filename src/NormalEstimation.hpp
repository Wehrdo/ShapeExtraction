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

// to save typing
__device__ inline float sqr(float val) {
    return val * val;
}

/*
 The following matrix functions work with symmetrical 3x3 matrices,
 since it's only for handling 3D covariances. Each matrix is passed
 as an array of the upper elements, e.g. for matrix A
     | a d e |
 A = | d b f |
     | e f c |
 data = [a, d, e, b, f, c]
*/


// Computes determinant of symmetrical matrix
__device__ inline float calcDeterminant(const float M_u[6]) {
    const float a, b, c, d, e, f;
    a = M_u[0]; d = M_u[1]; e = M_u[2];
                b = M_u[3]; f = M_u[4];
                            c = M_u[5];
    return a*(b*c - f*f)
         - d*(d*c - e*f)
         + e*(d*f - e*b);
}

// algorithm is from https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3%C3%973_matrices
// call input matrix A
__device__ inline void eigenValues(float eVals[3], const float cov_upper[6]) {
    float p1 = sqr(cov_upper[1]) + sqr(cov_upper[2]) + sqr(cov_upper[4]);
    // TODO: Is this check necessary? Will the algorithm still compute correctly when diagonal? Are there potentially numerical stability issues when almost diagonal?
    if (p1 == 0) {
        // matrix is diagonal, so no covariance, and diagonals are eigenvalues!
        eVals[0] = cov_upper[0];
        eVals[1] = cov_upper[3];
        eVals[2] = cov_upper[5];
        return;
    }
    float trace = cov_upper[0] + cov_upper[3] + cov_upper[5];
    float q = trace / 3;

    // float p2 = sqr(cov_upper[0] - q) + sqr(cov_upper[3] - q) + sqr(cov_upper[5] - q) + 2 * p1;
    // float p = sqrt(p2);
    // specialized CUDA function for this
    float p = norm4df(cov_upper[0] - q, cov_upper[3] - q, cov_upper[5] - q, 2 * p1);

    // B = A - q*I (where I is identity)
    float B_fac = 1 / p;
    float B[6] = {B_fac*(cov_upper[0]-q), B_fac*cov_upper[1],     B_fac*cov_upper[2],
                                          B_fac*(cov_upper[3]-q), B_fac*cov_upper[4],
                                                                  B_fac*(cov_upper[5] - q)};
    float r = calcDeterminant(B);
    
    float pi = M_PI; // be sure to use float
    float phi;
    if (r <= -1) {
        phi = pi / 3;
    }
    else if (r >= 1) {
        phi = 0;
    }
    else {
        phi = acosf(r) / 3;
    }

    // the eigenvalues satisfy eig3 <= eig2 <= eig1
    eVals[0] = q + 2 * p * cosf(phi);
    eVals[2] = q + 2 * p * cosf(phi + (2*pi / 3));
    eVals[1] = 3 * q - eVals[0] - eVals[2]; // Because trace(A) = eig1 + eig2 + eig3
}

// points is all points
// pt_indices is the nearest neighbor indices for this point (points[pt_indices[0]]) is closest point
template <int k>
__device__ inline void calculateCovariance(float cov_upper[6], const Point* points, const int pt_indices[k]) {
    // find centroid of points
    Point centroid;
    for (int i = 0; i < k; ++i) {
        centroid += points[pt_indices[i]];
    }
    centroid /= k;
    
    for (int i = 0; i < k; ++i) {
        const Point& pt = points[pt_indices[i]];
        const Point pt_m = pt - centroid; // mean-shifted point
        cov_upper[0] += pt_m.x * pt_m.x; // xx covariance (x variance): element a
        cov_upper[1] += pt_m.x * pt_m.y; // xy covariance: element d
        cov_upper[2] += pt_m.x * pt_m.z; // xz covariance: element e
        cov_upper[3] += pt_m.y * pt_m.y; // yy covariance (y variance): element b
        cov_upper[4] += pt_m.y * pt_m.z; // yz covariance: element f
        cov_upper[5] += pt_m.z * pt_m.z; // zz covariance (z variance): element d
    }
    // scale summed covariance
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        cov_upper[i] /= k; // TODO: Why doesn't covariance calculation divide by k-1, but variance doeos?
    }
}

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
    float cov_upper[6] = {0};
    calculateCovariance(cov_upper, points, &nn_indices[nn_idx]);


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

    // TODO: It might be more efficient to actually copy the points instead of passing the indices,
    // to increase memory access cohesion. It's only 3x the memory
    estimateNormalsKernel<k><<<blocks, tpb>>>(u_normals, octree.u_points, u_nn, n);

    cudaDeviceSynchronize();
    CudaCheckError();

    std::vector<Point> results(n);
    CudaCheckCall(cudaMemcpy(&results[0], u_normals, n * sizeof(*u_normals), cudaMemcpyDeviceToHost));
    return results;
}
