cmake_minimum_required(VERSION 3.25)
project(pointcloud_icp LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

find_package(Eigen3 3.4...5 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

# The ICP pipeline as a library, so the benchmark app (and later the Vulkan
# stages and tests) can link the same code. PUBLIC settings pass on to anything
# that links icp_core.
add_library(icp_core STATIC
  src/cloud.cpp
  src/nearest.cpp
  src/kdtree.cpp
  src/alignment.cpp
  src/icp.cpp
)
target_include_directories(icp_core PUBLIC src)
target_link_libraries(icp_core PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX)
target_compile_features(icp_core PUBLIC cxx_std_20)
target_compile_options(icp_core PUBLIC -Wall -Wextra -Wpedantic -march=native)

add_executable(icp src/main.cpp)
target_link_libraries(icp PRIVATE icp_core)
