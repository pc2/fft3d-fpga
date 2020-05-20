# Author: Arjun Ramaswami

cmake_minimum_required(VERSION 3.10)

set(This fft3d)

# OpenCL and related files
set(CL_FILES_BRAM
  ${CMAKE_SOURCE_DIR}/src/kernels/fft3d_kernels/fft3d_bram.cl 
)
set(CL_FILES_DDR
  ${CMAKE_SOURCE_DIR}/src/kernels/fft3d_kernels/fft3d_ddr.cl 
)

set(CL_INCLUDES
 "-I${CMAKE_SOURCE_DIR}/src/kernels/common/fft_config.h"
)

# Command Output Names
set(bst_name ${FFT_SIZE}pt_${This})
set(bitstream_ddr_emulate ${bst_name}_ddr_emulate.aocx)
set(bitstream_bram_emulate ${bst_name}_bram_emulate.aocx)
set(bitstream_report ${bst_name}_report.aocr)
set(bitstream_profile ${bst_name}_profile.aocx)
set(bitstream_syn ${bst_name}.aocx)

# Emulation
add_custom_command(
  OUTPUT ${bitstream_bram_emulate}
  COMMAND 
    ${IntelFPGAOpenCL_AOC} ${CL_FILES_BRAM} ${CL_INCLUDES} ${AOC_FLAGS} ${EMU_FLAGS} -board=${FPGA_BOARD_NAME} -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${bitstream_bram_emulate}
  MAIN_DEPENDENCY ${CL_FILES}
  COMMENT 
    "Building ${This} for emulation to folder ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  VERBATIM
)

add_custom_target(
  ${This}_bram_emulate DEPENDS ${bitstream_bram_emulate}
)

# Emulation DDR
add_custom_command(
  OUTPUT ${bitstream_ddr_emulate}
  COMMAND 
    ${IntelFPGAOpenCL_AOC} ${CL_FILES_DDR} ${CL_INCLUDES} ${AOC_FLAGS} ${EMU_FLAGS} -board=${FPGA_BOARD_NAME} -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${bitstream_ddr_emulate}
  MAIN_DEPENDENCY ${CL_FILES_DDR}
  COMMENT 
    "Building ${This} using DDR for emulation to folder ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  VERBATIM
)

add_custom_target(
  ${This}_ddr_emulate DEPENDS ${bitstream_ddr_emulate}
)

# Report generation
add_custom_command(
  OUTPUT ${bitstream_report}
  COMMAND 
    ${IntelFPGAOpenCL_AOC} ${CL_FILES} ${CL_INCLUDES} ${AOC_FLAGS} ${REP_FLAGS} -board=${FPGA_BOARD_NAME} -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${bitstream_report}
  MAIN_DEPENDENCY ${CL_FILES}
  COMMENT 
    "Generating report for ${This} in folder ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  VERBATIM
)

add_custom_target(
  ${This}_report DEPENDS ${bitstream_report}
)

# Profiling
add_custom_command(
  OUTPUT ${bitstream_profile}
  COMMAND 
    ${IntelFPGAOpenCL_AOC} ${CL_FILES} ${CL_INCLUDES} ${AOC_FLAGS} ${PROF_FLAGS} -board=${FPGA_BOARD_NAME} -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${bitstream_profile}
  MAIN_DEPENDENCY ${CL_FILES}
  COMMENT 
    "Building ${This} for profiling to folder ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  VERBATIM
)

add_custom_target(
  ${This}_profile DEPENDS ${bitstream_report}
)

# Synthesis
add_custom_command(
  OUTPUT ${bitstream_syn}
  COMMAND 
    ${IntelFPGAOpenCL_AOC} ${CL_FILES} ${CL_INCLUDES} ${AOC_FLAGS} -board=${FPGA_BOARD_NAME} -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${bitstream_syn}
  MAIN_DEPENDENCY ${CL_FILES}
  COMMENT 
    "Building ${This} for synthesis to folder ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  VERBATIM
)

add_custom_target(
  ${This}_syn DEPENDS ${bitstream_syn}
)

# TODO
## Debug compilation flag