##############################################################################
# Build file for guanaco-ctf project
##############################################################################

cmake_minimum_required(VERSION 3.12.0)

# Set the project name
project(guanaco-ctf CXX)

# Set the cmake module path
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Add pybind sub directory
add_subdirectory(pybind11)

# Add the automatically determined parts of the RPATH which point to directories
# outside the build tree to the install RPATH. Required for submission to
# clusters which may not allow export of LD_LIBRARY_PATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH True)

##############################################################################
# Build the guanaco-ctf library
##############################################################################

# Add a C/C++ extension
pybind11_add_module(guanaco_ext src/guanaco/ext/guanaco_ext.cpp)

# Ensure we are using C++11
target_compile_features(guanaco_ext PUBLIC cxx_std_14)

# Enable LTO
set_property(TARGET guanaco_ext PROPERTY INTERPROCEDURAL_OPTIMIZATION True)

# Set the include directory
target_include_directories(guanaco_ext PUBLIC src)

# Install the python extension
install(TARGETS guanaco_ext LIBRARY DESTINATION src/guanaco)
