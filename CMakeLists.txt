##############################################################################
# Build file for guanaco-ctf project
##############################################################################

cmake_minimum_required(VERSION 3.12.0)

# Set the project name
project(guanaco-ctf CXX CUDA)

# Set the cmake module path
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Find CUDA
find_package(FFTW REQUIRED)
find_package(CUDA REQUIRED)

# Add pybind sub directory
add_subdirectory(pybind11)

# Add the automatically determined parts of the RPATH which point to directories
# outside the build tree to the install RPATH. Required for submission to
# clusters which may not allow export of LD_LIBRARY_PATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH True)

##############################################################################
# Build the guanaco-ctf library
##############################################################################

# add_library(guanaco SHARED ${ASTRA_SOURCE})
# set_target_properties(guanaco PROPERTIES CXX_VISIBILITY_PRESET "default")
# set_target_properties(guanaco PROPERTIES CUDA_VISIBILITY_PRESET "default")
# target_compile_definitions(guanaco PUBLIC ASTRA_CUDA)
# target_compile_definitions(guanaco PUBLIC DLL_EXPORTS)
# target_include_directories(guanaco PUBLIC 
  # include
  # include/astra
  # /home/upc86896/Software/astra-toolbox/lib/include
  # ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
# target_compile_features(guanaco PUBLIC cxx_std_14)
# set_property(TARGET guanaco PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
# target_link_libraries(guanaco PUBLIC
#   ${CUDA_LIBRARIES} 
#   ${CUDA_CUFFT_LIBRARIES})

# Add a C/C++ extension
pybind11_add_module(guanaco_ext 
  src/guanaco/detail/ext/reconstructor.cpp
  src/guanaco/detail/ext/reconstructor_gpu.cu
  src/guanaco/detail/ext/guanaco_ext.cpp)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# Not specifying CUDA architecture throws and error.
# Setting this option does not pass arch flag to compiler
set_property(TARGET guanaco_ext PROPERTY CUDA_ARCHITECTURES OFF)


target_compile_definitions(guanaco_ext PUBLIC ASTRA_CUDA)





# Ensure we are using C++11
target_compile_features(guanaco_ext PUBLIC cxx_std_14)

# Enable LTO
set_property(TARGET guanaco_ext PROPERTY INTERPROCEDURAL_OPTIMIZATION True)

# Set the include directory
target_include_directories(guanaco_ext PUBLIC include src include/astra ${FFTW_INCLUDES})

# Set the link libraries
target_link_libraries(guanaco_ext PUBLIC #guanaco
  ${CUDA_LIBRARIES} 
  ${CUDA_CUFFT_LIBRARIES}
  ${FFTW_LIBRARIES})

# Install the python extension
install(TARGETS guanaco_ext LIBRARY DESTINATION src/guanaco/detail)
