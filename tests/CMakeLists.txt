#  Author: Arjun Ramaswami

cmake_minimum_required(VERSION 3.15)

set(This test_fft_fpga)

add_subdirectory(../extern/gtest ${CMAKE_CURRENT_BINARY_DIR}/lib)

include_directories(
  ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/extern/argparse 
  ${CMAKE_SOURCE_DIR}/src/host/include
  ${IntelFPGAOpenCL_INCLUDE_DIRS}
)

set(Sources_dir ${CMAKE_SOURCE_DIR}/src/host)

set(Sources 
  ${CMAKE_SOURCE_DIR}/extern/argparse/argparse.c
#  ${Sources_dir}/main.c
  ${Sources_dir}/src/fftfpga.c 
  ${Sources_dir}/src/helper.c
  ${Sources_dir}/src/opencl_utils.c
)

set(TestSources
  ${CMAKE_SOURCE_DIR}/tests/test_opencl_utils.cpp
  ${CMAKE_SOURCE_DIR}/tests/test_fft_fpga.cpp
  ${CMAKE_SOURCE_DIR}/tests/test_helper.cpp
)

add_executable(${This} ${Sources} ${TestSources})

target_link_libraries(${This} PUBLIC 
  gtest_main
  gtest
  gmock
  ${IntelFPGAOpenCL_LIBRARIES}
)

add_test(
  NAME ${This}
  COMMAND ${This}
)