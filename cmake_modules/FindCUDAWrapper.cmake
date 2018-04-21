#.rst:
# FindCUDAWrapper
# --------
#
# Wrapper which can be used to port the following commands over
# to the cmake cuda native language support:
# - cuda_add_executable
# - cuda_add_library
# - cuda_add_cufft_to_target
# - cuda_add_cublas_to_target
#
# The following variables affect the behavior of the macros in the
# script (in alphabetical order).
#
#   CUDA_SEPARABLE_COMPILATION (Default OFF)
#   -- If set this will enable separable compilation for all CUDA runtime object
#      files.
#
# The following variables have been provided to help transition existing
# code bases over to 'modern' target usage requirements
#
#
#
# The following functions have been provided to help bridge the differences
# between FindCUDA and native language support
# - convert_nvcc_flags
#


#Very important the first step is to enable the CUDA language.
enable_language(CUDA)

option(CUDA_SEPARABLE_COMPILATION "Compile CUDA objects with separable compilation enabled.  Requires CUDA 5.0+" OFF)

# Find the CUDA_INCLUDE_DIRS and CUDA_TOOLKIT_INCLUDE like FindCUDA does
find_path(CUDA_TOOLKIT_INCLUDE
  device_functions.h # Header included in toolkit
  PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
  PATH_SUFFIXES include ../include
  NO_DEFAULT_PATH
  )
set(CUDA_TOOLKIT_TARGET_DIR_INTERNAL "${CUDA_TOOLKIT_TARGET_DIR}" CACHE INTERNAL
  "This is the value of the last time CUDA_TOOLKIT_TARGET_DIR was set successfully." FORCE)

# Find the CUDA_NVCC_EXECUTABLE
find_program(CUDA_NVCC_EXECUTABLE
  NAMES nvcc
  PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
  PATH_SUFFIXES ../bin ../bin64
  NO_DEFAULT_PATH
  )

# Setup CUDA_INCLUDE_DIRS
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_INCLUDE})

# Setup CUDA_LIBRARIES.
#
# The CUDA_LIBRARIES should only contain libraries that start with
# 'cuda*'' from CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES
function(cuda_setup_cuda_libraries)
  set(CUDA_LIBRARIES )
  foreach(lib IN LISTS CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES )
    get_filename_component(lib_name "${lib}" NAME_WE)
    string(REGEX REPLACE "^lib" "" lib_name "${lib_name}")
    if(lib_name MATCHES "^cuda")

      #keep the cmake cache clean of these find calls
      unset(cuda_find_lib_lib_path CACHE)
      find_library(cuda_find_lib_lib_path
        NAMES "${lib_name}"
        PATHS "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}"
        NO_DEFAULT_PATH
        )
      if(cuda_find_lib_lib_path)
          list(APPEND CUDA_LIBRARIES "${cuda_find_lib_lib_path}")
      endif()
      unset(cuda_find_lib_lib_path CACHE)
    endif()
  endforeach()

  if(APPLE)
    # We need to add the default path to the driver (libcuda.dylib) as an rpath, so that
    # the static cuda runtime can find it at runtime.
    list(APPEND CUDA_LIBRARIES -Wl,-rpath,/usr/local/cuda/lib)
  endif()

  set(CUDA_LIBRARIES "${CUDA_LIBRARIES}" PARENT_SCOPE)
endfunction()

set(CUDA_LIBRARIES )
cuda_setup_cuda_libraries()

# wrapper for cuda_add_library
# Issues:
#
function(cuda_add_library)
  add_library(${ARGV})
  target_include_directories(${ARGV0} PUBLIC
                             ${CUDA_INCLUDE_DIRS})
  target_link_libraries(${ARGV0} ${CUDA_LIBRARIES})

  if(CUDA_SEPARABLE_COMPILATION)
    set_target_properties(${ARGV0} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  endif()
endfunction()


# wrapper for cuda_add_library
# Issues:
#
function(cuda_add_executable)
  add_executable(${ARGV})
  target_include_directories(${ARGV0} PUBLIC
                             ${CUDA_INCLUDE_DIRS})
  target_link_libraries(${ARGV0} ${CUDA_LIBRARIES})

  if(CUDA_SEPARABLE_COMPILATION)
    set_target_properties(${ARGV0} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  endif()
endfunction()

# wrapper for cuda_add_cufft_to_target
# Issues:
#
function(cuda_add_cufft_to_target target)
  target_link_libraries(${target} ${CUDA_cufft_LIBRARY})
endfunction()

# wrapper for cuda_add_cublas_to_target
# Issues:
#
function(cuda_add_cublas_to_target target)
  target_link_libraries(${target} ${CUDA_cublas_LIBRARY} ${CUDA_cublas_device_LIBRARY})
endfunction()

# Will take the exiting CUDA_NVCC_FLAGS and apply them to CMAKE_CUDA_FLAGS
# Doing the proper conversion from a list to a single string that is required
#
# Note: This will append the flags to the back of the exiting CMAKE_CUDA_FLAGS
function(convert_nvcc_flags )
    # Native CUDA support requires CMAKE_CUDA_FLAGS to be a single string
    # just like CMAKE_CXX_FLAGS and CMAKE_C_FLAGS
    string(REPLACE ";" " " temp_flags "${CUDA_NVCC_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${_temp_flags}" PARENT_SCOPE)
endfunction()

find_package(CUDALibs)

set(CUDAWRAPPERS_FOUND TRUE)
set(CUDA_FOUND TRUE)
