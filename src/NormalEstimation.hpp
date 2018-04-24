#pragma once

#include "PointCloud.hpp"
#include "Octree.hpp"
#include "CudaCommon.hpp"

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <tuple>
#include <cmath>
#include <chrono>

class NormalEstimation {
public:
    template <int k>
    static std::vector<Point> estimateNormals(const OT::Octree& octree);
    template <int k>
    static std::vector<Point> estimateNormals(const OT::Octree& octree, const int start_idx, const int n);
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


// // Computes determinant of symmetrical matrix
// __device__ inline float calcDeterminant(const float M_u[6]) {
//     const float a, b, c, d, e, f;
//     a = M_u[0]; d = M_u[1]; e = M_u[2];
//                 b = M_u[3]; f = M_u[4];
//                             c = M_u[5];
//     return a*(b*c - f*f)
//          - d*(d*c - e*f)
//          + e*(d*f - e*b);
// }

// // algorithm is from https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3%C3%973_matrices
// // call input matrix A
// __device__ inline void eigenValues(float eVals[3], const float cov_upper[6]) {
//     float p1 = sqr(cov_upper[1]) + sqr(cov_upper[2]) + sqr(cov_upper[4]);
//     // TODO: Is this check necessary? Will the algorithm still compute correctly when diagonal? Are there potentially numerical stability issues when almost diagonal?
//     if (p1 == 0) {
//         // matrix is diagonal, so no covariance, and diagonals are eigenvalues!
//         eVals[0] = cov_upper[0];
//         eVals[1] = cov_upper[3];
//         eVals[2] = cov_upper[5];
//         return;
//     }
//     float trace = cov_upper[0] + cov_upper[3] + cov_upper[5];
//     float q = trace / 3;

//     // float p2 = sqr(cov_upper[0] - q) + sqr(cov_upper[3] - q) + sqr(cov_upper[5] - q) + 2 * p1;
//     // float p = sqrt(p2);
//     // specialized CUDA function for this
//     float p = norm4df(cov_upper[0] - q, cov_upper[3] - q, cov_upper[5] - q, 2 * p1);

//     // B = A - q*I (where I is identity)
//     float B_fac = 1 / p;
//     float B[6] = {B_fac*(cov_upper[0]-q), B_fac*cov_upper[1],     B_fac*cov_upper[2],
//                                           B_fac*(cov_upper[3]-q), B_fac*cov_upper[4],
//                                                                   B_fac*(cov_upper[5] - q)};
//     float r = calcDeterminant(B);
    
//     float pi = M_PI; // be sure to use float
//     float phi;
//     if (r <= -1) {
//         phi = pi / 3;
//     }
//     else if (r >= 1) {
//         phi = 0;
//     }
//     else {
//         phi = acosf(r) / 3;
//     }

//     // the eigenvalues satisfy eig3 <= eig2 <= eig1
//     eVals[0] = q + 2 * p * cosf(phi);
//     eVals[2] = q + 2 * p * cosf(phi + (2*pi / 3));
//     eVals[1] = 3 * q - eVals[0] - eVals[2]; // Because trace(A) = eig1 + eig2 + eig3
// }

// points is all points
// pt_indices is the nearest neighbor indices for this point (points[pt_indices[0]]) is closest point
template <int k>
__device__ inline void calculateCovariance(float cov_upper[6], const Point* points, const int pt_indices[k]) {
    // find centroid of points
    Point centroid(0, 0, 0);
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
        cov_upper[i] /= (k - 1);
    }
}

// populates covars with N 3x3 covariance matrices, using the nearest neighbors
template <int k>
__global__ void findCovariancesKernel(
    float* covars,
    const Point* points,
    const int* nn_indices,
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) { return; }
    
    const int nn_idx = idx * k;
    // calculate in local memory
    float cov_upper[6] = {0};
    calculateCovariance<k>(cov_upper, points, &nn_indices[nn_idx]);

    // copy local memory into global array

    // start of this matrix
    float* cov_M = &covars[3 * 3 * idx];

    // TODO: It might not be necessary to populate entire 3x3. check documentation for parameter A of cuSolver call. Probably can't memcpy unless we store cov_upper in column-major, I think (need to verify this)
    // cov_M is technically in column-major, but because of symmetry, doesn't matter
    cov_M[0] = cov_upper[0];
    cov_M[1] = cov_upper[1];
    cov_M[2] = cov_upper[2];

    cov_M[3] = cov_upper[1];
    cov_M[4] = cov_upper[3];
    cov_M[5] = cov_upper[4];

    cov_M[6] = cov_upper[2];
    cov_M[7] = cov_upper[4];
    cov_M[8] = cov_upper[5];
} 

// takes cross product of two largest eigenvectors (representing the plane) to compute normal
// assuming eigen is an array of packed 3x3 matrices in column-major order, sorted by largest eigenvectors
__global__ void normalsFromEigenVectors(
    Point* normals,
    const float* eigen,
    const Point* points,
    const Point viewpoint,
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) { return; }

    // // second-largest eigenvector
    // const float* vec2 = &eigen[idx * 3 * 3 + 3];
    // Point norm;
    // norm.x = vec1[1]*vec2[2] - vec1[2]*vec2[1];
    // norm.y = vec1[2]*vec2[0] - vec1[0]*vec2[2];
    // norm.z = vec1[0]*vec2[1] - vec1[1]*vec2[0];

    // smallest eigenvector
    const float* vec = &eigen[idx * 3 * 3];
    Point norm(vec[0], vec[1], vec[2]);

    // normalize vector
    // norm /= norm3df(norm.x, norm.y, norm.z);

    Point point = points[idx];
    if ((viewpoint - point).dot(norm) < 0) {
        norm.x *= -1;
        norm.y *= -1;
        norm.z *= -1;
    }
    
    normals[idx] = norm;
}
/*
// --- Class template implementations ---
*/
template <int k>
std::vector<Point> NormalEstimation::estimateNormals(const OT::Octree& octree) {
    return estimateNormals<k>(octree, 0, octree.n_pts);
}

template <int k>
std::vector<Point> NormalEstimation::estimateNormals(const OT::Octree& octree, int start_idx, int n) {
    // the points to search
    Point* search_pts = octree.u_points + start_idx;

    // nearest neighbors in unified memory
    int* u_nn;
    CudaCheckCall(cudaMallocManaged(&u_nn, n * k * sizeof(*u_nn)));

    auto start_time = std::chrono::high_resolution_clock::now();
    octree.deviceKnnSearch<k>(search_pts, u_nn, n, 0.01f);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "For start_idx " << start_idx << ", knnSearch took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();

    // matrix for covariances
    const int lda = 3; // leading dimension size
    const int m = 3; // other dimension size
    const int batch_size = n;
    float* u_C;
    CudaCheckCall(cudaMallocManaged(&u_C, lda * m * batch_size * sizeof(*u_C)));

    int blocks, tpb;
    std::tie(blocks, tpb) = makeLaunchParams(n);

    // TODO: It might be more efficient to actually copy the points instead of passing the indices,
    // to increase memory access cohesion. It's only 3x the memory
    findCovariancesKernel<k><<<blocks, tpb>>>(u_C, octree.u_points, u_nn, n);

    cudaDeviceSynchronize();
    CudaCheckError();
    CudaCheckCall(cudaFree(u_nn));

    // calculate eigenvalues and eigenvectors using cuSolver
    cusolverDnHandle_t cusolverH = NULL;
    // cudaStream_t stream = 0;
    syevjInfo_t syevj_params = NULL;

    // holds resulting eigenvalues
    float* u_W;
    CudaCheckCall(cudaMallocManaged(&u_W, m * batch_size * sizeof(*u_W)));
    // see documentation for more info, but if solve_info[i] == 0, then solving matrix i was successful
    int* u_solve_info;
    CudaCheckCall(cudaMallocManaged(&u_solve_info, batch_size * sizeof(*u_solve_info)));
    // temporary workspace memory for cuSolver
    int temp_work_size = 0;
    float* d_temp_work = NULL;

    // parameters of syevj
    const float tol = 1.e-7; // tolerance I guess?
    const int max_sweeps = 15;
    const int sort_eig = 1; // sort the eigenvalues in ascending order
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // Can be CUSOLVER_EIG_MODE_VECTOR or CUSOLVER_EIG_MODE_NOVECTOR. We want vectors reported
    const cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER; // I think that means just the uppper-triangular values need to be there

    // setup cusolver with parameters
    CusolverCheckCall(cusolverDnCreate(&cusolverH));
    // CudaCheckCall(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); // why do we have to make a new stream?
    // CusolverCheckCall(cusolverDnSetStream(cusolverH, stream));
    // get default params
    CusolverCheckCall(cusolverDnCreateSyevjInfo(&syevj_params));
    // set custom params
    CusolverCheckCall(cusolverDnXsyevjSetTolerance(syevj_params, tol));
    CusolverCheckCall(cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps));
    CusolverCheckCall(cusolverDnXsyevjSetSortEig(syevj_params, sort_eig));

    // get amount of temp memory required
    CusolverCheckCall(cusolverDnSsyevjBatched_bufferSize(
        cusolverH,
        jobz,
        uplo,
        m,
        u_C,
        lda,
        u_W,
        &temp_work_size,
        syevj_params,
        batch_size
    ));

    // allocate temp memory
    CudaCheckCall(cudaMalloc(&d_temp_work, temp_work_size * sizeof(*d_temp_work)));

    // run solver
    CusolverCheckCall(cusolverDnSsyevjBatched(
        cusolverH,
        jobz,
        uplo,
        m,
        u_C,
        lda,
        u_W,
        d_temp_work,
        temp_work_size,
        u_solve_info,
        syevj_params,
        batch_size
    ));
    cudaDeviceSynchronize();
    CudaCheckError();

    // for (int i = 0; i < n; ++i) {
    //     if (u_solve_info[i] > 0) {
    //         printf("Matrix %d does not converge: %d\n", i, u_solve_info[i]);
    //     }
    //     if (u_solve_info[i] < 0) {
    //         printf("Matrix %d has wrong parameter %d\n", i, -u_solve_info[i]);
    //     }
    // }
    CudaCheckCall(cudaFree(u_solve_info));
    CudaCheckCall(cudaFree(u_W));
    CudaCheckCall(cudaFree(d_temp_work));
    end_time = std::chrono::high_resolution_clock::now();
    std::cout << "For start_idx " << start_idx << ", eigenvector calculation took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms" << std::endl;
    
    start_time = std::chrono::high_resolution_clock::now();
    // the location of the viewpoint points were captured from, for orienting normals
    Point viewpoint(0, 0, 0);
    // normals in unified memory
    Point* u_normals;
    CudaCheckCall(cudaMallocManaged(&u_normals, n * sizeof(*u_normals)));
    normalsFromEigenVectors<<<blocks, tpb>>>(u_normals, u_C, search_pts, viewpoint, n);
    cudaDeviceSynchronize();
    CudaCheckError();

    CudaCheckCall(cudaFree(u_C));

    std::vector<Point> results(n);
    CudaCheckCall(cudaMemcpy(&results[0], u_normals, n * sizeof(*u_normals), cudaMemcpyDeviceToHost));
    CudaCheckCall(cudaFree(u_normals));
    end_time = std::chrono::high_resolution_clock::now();
    std::cout << "For start_idx " << start_idx << ", normal kernel took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms" << std::endl;
    return results;
}
