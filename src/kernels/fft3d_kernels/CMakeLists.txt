# Author: Arjun Ramaswami

cmake_minimum_required(VERSION 3.10)

set(This fft3d)

# OpenCL and related files
set(CL_FILES_BRAM
  ${CMAKE_SOURCE_DIR}/src/kernels/fft3d_kernels/fft3d_bram.cl 
)
set(CL_FILES_DDR
  ${CMAKE_SOURCE_DIR}/src/kernels/fft3d_kernels/fft3d_ddr_triv.cl 
)

set(CL_INCLUDES
 "-I${CMAKE_SOURCE_DIR}/src/kernels/common/fft_config.h"
)

# Command Output Names
set(bst_name ${FFT_SIZE}pt_${This})
set(bitstream_ddr_emulate ${bst_name}_ddr_emulate.aocx)
set(bitstream_bram_emulate ${bst_name}_bram_emulate.aocx)

set(bitstream_ddr_report ${bst_name}_ddr_report.aocr)
set(bitstream_bram_report ${bst_name}_bram_report.aocr)

set(bitstream_profile ${bst_name}_profile.aocx)

set(bitstream_ddr_syn ${bst_name}_ddr_syn.aocx)
set(bitstream_bram_syn ${bst_name}_bram_syn.aocx)

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

# Report generation BRAM
add_custom_command(
  OUTPUT ${bitstream_bram_report}
  COMMAND 
    ${IntelFPGAOpenCL_AOC} ${CL_FILES_BRAM} ${CL_INCLUDES} ${AOC_FLAGS} ${REP_FLAGS} -board=${FPGA_BOARD_NAME} -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${bitstream_bram_report}
  MAIN_DEPENDENCY ${CL_FILES_BRAM}
  COMMENT 
    "Generating report for ${This} in folder ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  VERBATIM
)

add_custom_target(
  ${This}_bram_report DEPENDS ${bitstream_bram_report}
)

# Report generation DDR
add_custom_command(
  OUTPUT ${bitstream_ddr_report}
  COMMAND 
    ${IntelFPGAOpenCL_AOC} ${CL_FILES_DDR} ${CL_INCLUDES} ${AOC_FLAGS} ${REP_FLAGS} -board=${FPGA_BOARD_NAME} -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${bitstream_ddr_report}
  MAIN_DEPENDENCY ${CL_FILES_DDR}
  COMMENT 
    "Generating report for ${This} in folder ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  VERBATIM
)

add_custom_target(
  ${This}_ddr_report DEPENDS ${bitstream_ddr_report}
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

# Synthesis BRAM
add_custom_command(
  OUTPUT ${bitstream_bram_syn}
  COMMAND 
    ${IntelFPGAOpenCL_AOC} ${CL_FILES_BRAM} ${CL_INCLUDES} ${AOC_FLAGS} -board=${FPGA_BOARD_NAME} -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${bitstream_bram_syn}
  MAIN_DEPENDENCY ${CL_FILES_BRAM}
  COMMENT 
    "Building ${This} for synthesis to folder ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  VERBATIM
)

add_custom_target(
  ${This}_bram_syn DEPENDS ${bitstream_bram_syn}
)

# Synthesis DDR
add_custom_command(
  OUTPUT ${bitstream_ddr_syn}
  COMMAND 
    ${IntelFPGAOpenCL_AOC} ${CL_FILES_DDR} ${CL_INCLUDES} ${AOC_FLAGS} -board=${FPGA_BOARD_NAME} -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${bitstream_ddr_syn}
  MAIN_DEPENDENCY ${CL_FILES_DDR}
  COMMENT 
    "Building ${This} for synthesis to folder ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  VERBATIM
)

add_custom_target(
  ${This}_ddr_syn DEPENDS ${bitstream_ddr_syn}
)
# TODO
## Debug compilation flag