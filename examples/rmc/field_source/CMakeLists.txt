cmake_minimum_required(VERSION 3.16)
project(rmc_field_source CXX)

# Builds a compiled-source plugin that samples the frozen field. It must link
# against the same libopenmc that loads it: point CMAKE_PREFIX_PATH at the
# OpenMC install made from your build (cmake --install <build> --prefix ...).
find_package(OpenMC REQUIRED)
add_library(rmc_field_source SHARED field_source.cpp)
target_link_libraries(rmc_field_source OpenMC::libopenmc)
