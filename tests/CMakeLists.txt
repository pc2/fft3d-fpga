#  Author: Arjun Ramaswami

cmake_minimum_required(VERSION 3.10)

add_subdirectory(../extern/gtest ${CMAKE_CURRENT_BINARY_DIR}/lib)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "../extern/findFFTW")

add_executable(test_fftfpga
  #${CMAKE_CURRENT_SOURCE_DIR}/test_fft_fpga.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_opencl_utils.cpp
)

target_include_directories(test_fftfpga
  PUBLIC ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR}
        "${CMAKE_SOURCE_DIR}/extern/argparse"
        "${CMAKE_SOURCE_DIR}/api/src"
        ${IntelFPGAOpenCL_INCLUDE_DIRS}
)

target_link_libraries(test_fftfpga PUBLIC 
  gtest_main gtest gmock ${IntelFPGAOpenCL_LIBRARIES} argparse fftfpga
)

if(FFTW_FOUND)
  target_compile_definitions(test_fftfpga PRIVATE USE_FFTW)
  target_link_libraries(test_fftfpga PUBLIC fftw3 fftw3f m)
else()
  message(WARNING, "FFTW library not found. Cannot perform correctness tests!")
endif()

add_test(
  NAME test 
  COMMAND test
)