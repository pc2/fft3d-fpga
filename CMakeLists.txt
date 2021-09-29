# Author: Arjun Ramaswami
cmake_minimum_required (VERSION 3.10.3)
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

# Link argparse as static library
add_subdirectory(${CMAKE_SOURCE_DIR}/extern/argparse)

# Add sub directories
add_subdirectory(api)
add_subdirectory(kernels)
add_subdirectory(examples)

# build tests
message("-- Building tests")
add_subdirectory(tests)
enable_testing()