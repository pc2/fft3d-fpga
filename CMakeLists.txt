cmake_minimum_required(VERSION 3.15)

project(fft-fpga VERSION 0.1
            DESCRIPTION "Library for FFT using FPGAs"
            LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -fsanitize=leak -ggdb3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -fsanitize=leak -ggdb3")

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  enable_testing()
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/extern/hlslib/cmake)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

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

find_package(IntelFPGAOpenCL)

add_subdirectory(src/host)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    add_subdirectory(tests)
endif()

# FPGA

set(LOG_FFT_SIZE 6 CACHE STRING "Log of points of FFT")
math(EXPR FFT_SIZE "1 << ${LOG_FFT_SIZE}" OUTPUT_FORMAT DECIMAL)
#message(" FFT size is ${FFT_SIZE}")

configure_file(
  "${CMAKE_SOURCE_DIR}/src/kernels/common/fft_config.h.in"
  "${CMAKE_SOURCE_DIR}/src/kernels/common/fft_config.h"
)

## TODO: enumerate options to choose from
set(FPGA_BOARD_NAME p520_hpc_sg280l CACHE STRING "Target Board")

## Flags for different target options
set(AOC_FLAGS "-g -v -fpc -fp-relaxed -cl-single-precision-constant -no-interleaving=default" CACHE STRING "AOC compiler flags")
separate_arguments(AOC_FLAGS)
set(EMU_FLAGS "-legacy-emulator -march=emulator" CACHE STRING "AOC emulation flags")
separate_arguments(EMU_FLAGS)
set(REP_FLAGS "-report -rtl" CACHE STRING "AOC report flags")
set(PROF_FLAGS "-profile=all" CACHE STRING "AOC profile flags")
separate_arguments(REP_FLAGS)
separate_arguments(PROF_FLAGS)

if (INTELFPGAOPENCL_FOUND)
  add_subdirectory(src/kernels/fft1d_kernels)
  add_subdirectory(src/kernels/fft2d_kernels)
else()
  message("Intel FPGA OpenCL SDK not found!")
endif()
