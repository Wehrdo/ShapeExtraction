#pragma once

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <utility>
#include <tuple>
#include <string>

/*
 * From https://codeyarns.com/2011/03/02/how-to-do-error-checking-in-cuda/
 */

// #ifndef NDEBUG
// #define CudaChk(err) {cudaError_t res = err; if (res) printf("CudaErr == %d\n", res); assert(res == cudaSuccess);}
// #else
// #define CudaChk(err) ((void)0)
// #endif

#define CudaCheckCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CusolverCheckCall( err ) __cusolverSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifndef NDEBUG
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        // exit( -1 );
    }
#endif

    return;
}

std::string getCusolverErrorString(const cusolverStatus_t err);

inline void __cusolverSafeCall( cusolverStatus_t err, const char *file, const int line )
{
#ifndef NDEBUG
    if ( CUSOLVER_STATUS_SUCCESS != err )
    {
        fprintf( stderr, "CusolverCheckCall() failed at %s:%i : %s\n",
                 file, line, getCusolverErrorString( err ).c_str() );
        // exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifndef NDEBUG
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        // exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    // err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        // exit( -1 );
    }
#endif

    return;
}

 std::tuple<int, int> makeLaunchParams(int n, int tpb = 512);
