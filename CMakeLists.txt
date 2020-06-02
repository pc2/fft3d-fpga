# Arjun Ramaswami
cmake_minimum_required(VERSION 3.10)

project(fft-fpga VERSION 0.2
            DESCRIPTION "Library for FFT using FPGAs"
            LANGUAGES C CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Setup submodules that are required
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

# Include hlslib in CMake module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/extern/hlslib/cmake)

# Find hlslib Intel OpenCL kernels
find_package(IntelFPGAOpenCL REQUIRED)

# OpenCL host target generation
add_subdirectory(api/)

# GTest Setup
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  option(PACKAGE_TESTS "Build the tests" ON)
endif()


#if(PACKAGE_TESTS)
#    enable_testing()
#    include(GoogleTest)
#    add_subdirectory(tests)
#endif()

# OpenCL kernel targets generation
## setup cmake variables to generate header file
set(LOG_POINTS 3 CACHE STRING "Log of per sample data points")
math(EXPR POINTS "1 << ${LOG_POINTS}")

set(LOG_FFT_SIZE 6 CACHE STRING "Log of points of FFT")
math(EXPR FFT_SIZE "1 << ${LOG_FFT_SIZE}")
message("-- FFT size is ${FFT_SIZE}")

configure_file(
  "${CMAKE_SOURCE_DIR}/kernels/common/fft_config.h.in"
  "${CMAKE_BINARY_DIR}/kernels/common/fft_config.h"
)

##
# Get FPGA Board Name from env variable or use default
# Set it to CMakeCache
##
set(BOARD_NAME $ENV{FPGA_BOARD_NAME})
if(BOARD_NAME)
  set(FPGA_BOARD_NAME ${BOARD_NAME} CACHE STRING "Target Board")
else()
  set(FPGA_BOARD_NAME p520_hpc_sg280l CACHE STRING "Target Board")
endif()

## Flags for different target options
set(AOC_FLAGS "-g -v -fpc -fp-relaxed -cl-single-precision-constant -no-interleaving=default" CACHE STRING "AOC compiler flags")
separate_arguments(AOC_FLAGS)
set(EMU_FLAGS "-legacy-emulator -march=emulator" CACHE STRING "AOC emulation flags")
separate_arguments(EMU_FLAGS)
set(REP_FLAGS "-report -rtl" CACHE STRING "AOC report flags")
separate_arguments(REP_FLAGS)
set(PROF_FLAGS "-profile=all" CACHE STRING "AOC profile flags")
separate_arguments(PROF_FLAGS)

if (INTELFPGAOPENCL_FOUND)
  add_subdirectory(kernels/fft1d)
  add_subdirectory(kernels/fft2d)
  add_subdirectory(kernels/fft3d)
else()
  message(WARNING, "Intel FPGA OpenCL SDK not found!")
endif()
