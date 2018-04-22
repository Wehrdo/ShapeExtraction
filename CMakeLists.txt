cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(shapes LANGUAGES CXX CUDA)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_modules)

file(GLOB shapes_sources ./src/*.cpp ./src/*.cu ./src/*.hpp)
list(REMOVE_ITEM shapes_sources "main.cu")

add_executable(shapes ${shapes_sources})
target_include_directories(shapes PRIVATE libmorton)
target_include_directories(shapes PRIVATE cub)

# Request that shapes be built with -std=c++14
# As this is a public compile feature anything that links to particles
# will also build with -std=c++14
target_compile_features(shapes PUBLIC cxx_std_14)

# set_target_properties(shapes PROPERTIES POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_FLAGS "-arch=sm_61 --expt-relaxed-constexpr")
# Device debug info in debug mode
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties( shapes
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(APPLE)
  # We need to add the default path to the driver (libcuda.dylib) as an rpath,
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET shapes PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

# OpenMP required for cuSolver
find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(CUDALibs REQUIRED)
# Link cuSolver and cuBLAS
# target_link_libraries(shapes ${CUDA_cusolver_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})
# message(STATUS, ${CUDA_cusolver_LIBRARY})
# message(STATUS, ${CUDA_CUBLAS_LIBRARIES})
# find_library(CUBLAS_LIB cublas HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
# message(STATUS, ${CUBLAS_LIB})
# target_link_libraries(shapes, cublas)
# cuda_add_cublas_to_target(shapes)
# target_link_libraries(shapes ${CUDA_cublas_LIBRARY} ${CUDA_cublas_device_LIBRARY})
target_link_libraries(shapes ${CUDA_cublas_LIBRARY})
target_link_libraries(shapes ${CUDA_cusolver_LIBRARY})

list(APPEND CUDA_NVCC_FLAGS -lcublas_device)
