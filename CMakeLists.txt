cmake_minimum_required(VERSION 3.25)
project(pointcloud_icp LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

add_executable(icp src/main.cpp)
target_compile_features(icp PRIVATE cxx_std_20)
target_compile_options(icp PRIVATE -Wall -Wextra -Wpedantic -march=native)
