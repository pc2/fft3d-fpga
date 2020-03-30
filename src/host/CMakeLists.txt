cmake_minimum_required(VERSION 3.15)

set(This fft_fpga)

include_directories(
  ${CMAKE_SOURCE_DIR}/extern/argparse
  ${CMAKE_SOURCE_DIR}/src/host/include
  ${IntelFPGAOpenCL_INCLUDE_DIRS}
)

set(Sources_dir ${CMAKE_SOURCE_DIR}/src/host)
set(Sources 
      ${CMAKE_SOURCE_DIR}/extern/argparse/argparse.c
      ${Sources_dir}/main.c
      ${Sources_dir}/src/fftfpga.c 
      ${Sources_dir}/src/helper.c
      ${Sources_dir}/src/opencl_utils.c
)
add_executable(${This} ${Sources})

target_link_libraries(${This} PUBLIC ${IntelFPGAOpenCL_LIBRARIES})