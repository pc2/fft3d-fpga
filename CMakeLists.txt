# Author: Arjun Ramaswami
cmake_minimum_required (VERSION 3.10)
project(fft VERSION 2.0
          DESCRIPTION "OpenCL based FFT library for Intel FPGAs"
          LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# build external dependencies
message("-- Building external dependencies")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/extDep.cmake)

## find Intel FPGA SDK for OpenCL
find_package(IntelFPGAOpenCL REQUIRED)
message("-- IntelFPGAOpenCL found")

## find FFTW
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(FFTW REQUIRED)
message("-- FFTW found")

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to 'Debug' as none was specified.")
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "RelWithDebInfo")
endif()

# sub directories
add_subdirectory(api)
add_subdirectory(kernels)
add_subdirectory(examples)

# build tests
message("-- Building tests")
add_subdirectory(tests)
enable_testing()