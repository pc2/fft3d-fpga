#  Author: Arjun Ramaswami

cmake_minimum_required(VERSION 3.10)

add_subdirectory(../extern/gtest ${CMAKE_CURRENT_BINARY_DIR}/lib)

add_executable(test_fftfpga
  #${CMAKE_CURRENT_SOURCE_DIR}/test_fft_fpga.cpp
      test_fft_setup.cpp
      test_fft1d_fpga.cpp
      test_fft2d_fpga.cpp
      test_fft3d_fpga.cpp
      test_opencl_utils.cpp
      test_misc.cpp
    ${examplesfftfpga_SOURCE_DIR}/common/helper.c
    ${examplesfftfpga_SOURCE_DIR}/common/verify_fftw.c
)

target_include_directories(test_fftfpga
  PUBLIC ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR}
        ${IntelFPGAOpenCL_INCLUDE_DIRS}
        ${examplesfftfpga_SOURCE_DIR}/common
        ${CMAKE_SOURCE_DIR}/api/src
        ${CMAKE_SOURCE_DIR}/extern/argparse
)

target_link_libraries(test_fftfpga PUBLIC 
  gtest_main gtest gmock ${IntelFPGAOpenCL_LIBRARIES} argparse fftfpga m
)

if(FFTW_FOUND)
  target_compile_definitions(test_fftfpga PRIVATE USE_FFTW)
  target_link_libraries(test_fftfpga PUBLIC fftw3 fftw3f m)
else()
  message(WARNING, "FFTW library not found. Cannot perform correctness tests!")
endif()

add_dependencies(test_fftfpga fft3d_bram_emu)
add_dependencies(test_fftfpga fft3d_ddr_triv_emu)
add_dependencies(test_fftfpga fft2d_bram_emu)
add_dependencies(test_fftfpga fft2d_ddr_emu)
add_dependencies(test_fftfpga fft1d_emu)

add_test(
  NAME test 
  COMMAND test
)