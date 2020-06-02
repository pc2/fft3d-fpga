# Author: Arjun Ramaswami
cmake_minimum_required(VERSION 3.10)

## 
# Call function to create custom build commands
# Generates targets:
#   - ${kernel_name}_emu: to generate emulation binary
#   - ${kernel_name}_rep: to generate report
#   - ${kernel_name}_syn: to generate synthesis binary
##
set(CL_PATH "${CMAKE_SOURCE_DIR}/kernels/fft3d")

include(${CMAKE_SOURCE_DIR}/cmake/genKernelTargets.cmake)

set(kernels fft3d_bram fft3d_ddr fft3d_ddr_triv)

if (INTELFPGAOPENCL_FOUND)
  gen_fft_targets(${kernels})
endif()
