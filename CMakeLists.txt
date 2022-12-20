##############################################################################
# Build file for guanaco project
##############################################################################

cmake_minimum_required(VERSION 3.17.0)

# Set the project name
project(guanaco CXX CUDA)

# Set the build type to release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Set the cuda arch
if(NOT CMAKE_CUDA_ARCHITECTURES)
  if ($ENV{CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES $ENV{CMAKE_CUDA_ARCHITECTURES})
  else()
    set(CMAKE_CUDA_ARCHITECTURES OFF)
  endif()
endif()

# Set the cmake module path
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Find CUDA
find_package(CUDAToolkit REQUIRED)
find_package(FFTW REQUIRED)

# Add pybind sub directory
add_subdirectory(pybind11)

# Add the automatically determined parts of the RPATH which point to directories
# outside the build tree to the install RPATH. Required for submission to
# clusters which may not allow export of LD_LIBRARY_PATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH True)

# Set position independent code (-fPIC)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Enable LTO
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION True)

# Use C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Allow Lambdas to be called from device
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")


##############################################################################
# Build the guanaco library
##############################################################################

add_library(guanaco STATIC
  src/libguanaco/correct.cpp
  src/libguanaco/correct.cu
  src/libguanaco/fft.cpp
  src/libguanaco/fft.cu
  src/libguanaco/filter.cpp
  src/libguanaco/filter.cu
  src/libguanaco/reconstructor.cpp
  src/libguanaco/reconstructor.cu)

# Not specifying CUDA architecture throws and error.
# Setting this option does not pass arch flag to compiler
set_property(TARGET guanaco PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

# Enable LTO
set_property(TARGET guanaco PROPERTY INTERPROCEDURAL_OPTIMIZATION True)

# Set the include directory
target_include_directories(guanaco PUBLIC include src ${FFTW_INCLUDES})

##############################################################################
# Build the guanaco library
##############################################################################

# Add a C/C++ extension
pybind11_add_module(guanaco_ext 
  src/guanaco/detail/ext/guanaco_ext.cpp)

# Set the include directory
target_include_directories(guanaco_ext PUBLIC include src)

# Set the link libraries
target_link_libraries(guanaco_ext PUBLIC
  guanaco
  CUDA::cudart
  CUDA::cuda_driver
  CUDA::cufft
  ${FFTW_LIBRARIES})

# Install the python extension
install(TARGETS guanaco_ext LIBRARY DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
