cmake_minimum_required(VERSION 3.15)

set(This fft1d)

set(bitstream_emulate ${This}_emulate.aocx)

set(AOC_FLAGS "-g -v -fpc -fp-relaxed -cl-single-precision-constant -no-interleaving=default" CACHE STRING "AOC compiler flags")
set(EMU_FLAGS "-march=emulator -legacy-emulator" CACHE STRING "AOC emulation flags")
separate_arguments(AOC_FLAGS)
separate_arguments(EMU_FLAGS)
mark_as_advanced(AOC_FLAGS EMU_FLAGS)

set(KERNEL_DIR ${CMAKE_SOURCE_DIR}/src/kernels/fft1d_kernels)

set(CL_FILES 
      ${KERNEL_DIR}/fft1d.cl 
      ${CMAKE_SOURCE_DIR}/src/kernels/fft_config.h
)

# sub-project level output directory

# BOARD

add_custom_command(
  OUTPUT ${bitstream_emulate}
  COMMAND ${IntelFPGAOpenCL_AOC} ${CL_FILES} ${AOC_FLAGS} ${EMU_FLAGS} -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${bitstream_emulate}
  MAIN_DEPENDENCY ${CL_FILES}
  COMMENT "Building ${This} for emulation to folder ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  VERBATIM
)

add_custom_target(
  ${This}_emulate DEPENDS ${bitstream_emulate}
)

#if (INTELFPGAOPENCL_FOUND)