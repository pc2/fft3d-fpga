# Arjun Ramaswami
cmake_minimum_required(VERSION 3.10)

##
# Generate host executable that is required to call OpenCL kernel bitstreams
# Target: host
##
add_executable(host 
    main.c
    src/fftfpga.c 
    src/helper.c
    src/opencl_utils.c
    ${CMAKE_SOURCE_DIR}/extern/argparse/argparse.c)

target_compile_options(host
    PRIVATE -Wall -Werror)
    
target_include_directories(host
    PRIVATE include
        ${IntelFPGAOpenCL_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/extern/argparse)
  
target_link_libraries(host
    PUBLIC ${IntelFPGAOpenCL_LIBRARIES} m)
