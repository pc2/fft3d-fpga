# Author: Arjun Ramaswami
cmake_minimum_required (VERSION 3.10.3)

project(fft)

set(CMAKE_C_STANDARD 11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

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

# Find FFTW
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/extern/findFFTW)
find_package(FFTW REQUIRED)

# Link argparse as static library
add_subdirectory(${CMAKE_SOURCE_DIR}/extern/argparse)

# Add sub directories
add_subdirectory(api)
add_subdirectory(kernels)
add_subdirectory(examples)

# GTest Setup
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  option(PACKAGE_TESTS "Build the tests" ON)
endif()

if(PACKAGE_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(tests)
endif()

# TODO: shift to external project instead of submodules as suggested by GTest