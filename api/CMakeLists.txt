# Arjun Ramaswami
cmake_minimum_required(VERSION 3.10)
project(fftfpga VERSION 0.3
            DESCRIPTION "APIs for FFT using FPGAs"
            LANGUAGES C CXX)

##
# Generate host executable that is required to call OpenCL kernel bitstreams
# Target: host
##
add_library(${PROJECT_NAME} STATIC 
              ${PROJECT_SOURCE_DIR}/src/fftfpga.c 
              ${PROJECT_SOURCE_DIR}/src/opencl_utils.c
              ${PROJECT_SOURCE_DIR}/src/misc.c)

target_compile_options(${PROJECT_NAME}
    PRIVATE -Wall -Werror)
    
if(USE_DEBUG)
  target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG)
endif()

target_include_directories(${PROJECT_NAME}
    PRIVATE src 
    PUBLIC ${IntelFPGAOpenCL_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
  
target_link_libraries(${PROJECT_NAME}
    PUBLIC ${IntelFPGAOpenCL_LIBRARIES} m)
