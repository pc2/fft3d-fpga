#  Author: Arjun Ramaswami
cmake_minimum_required(VERSION 3.10)
project(testfftfpga VERSION 2.0
            DESCRIPTION "Tests for FFT using FPGAs"
            LANGUAGES C CXX)

add_executable(test_fftfpga
      test_fft_setup.cpp
      test_fft1d_fpga.cpp
      test_fft2d_fpga.cpp
      test_fft3d_fpga.cpp
      test_opencl_utils.cpp
)

target_include_directories(test_fftfpga
  PUBLIC  ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR}
          ${IntelFPGAOpenCL_INCLUDE_DIRS}
          ${examplesfftfpga_SOURCE_DIR}
          ${CMAKE_SOURCE_DIR}/api/src
)

target_link_libraries(test_fftfpga PUBLIC 
  gtest_main gtest gmock ${IntelFPGAOpenCL_LIBRARIES} fftfpga m
)

if(FFTW_FOUND)
  target_compile_definitions(test_fftfpga PRIVATE USE_FFTW)
  target_link_libraries(test_fftfpga PUBLIC fftw3 fftw3f m)
else()
  message(WARNING, "FFTW library not found. Cannot perform correctness tests!")
endif()

add_dependencies(test_fftfpga fft3d_bram_emulate)
add_dependencies(test_fftfpga fft3d_ddr_emulate)
add_dependencies(test_fftfpga fft2d_bram_emulate)
add_dependencies(test_fftfpga fft2d_ddr_emulate)
add_dependencies(test_fftfpga fft1d_emulate)

add_test(
  NAME test 
  COMMAND test
)