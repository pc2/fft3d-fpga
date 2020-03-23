
include_directories(${CMAKE_SOURCE_DIR}/extern/argparse)
include_directories(${CMAKE_SOURCE_DIR}/src/host/include)
include_directories(${IntelFPGAOpenCL_INCLUDE_DIRS})

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -fsanitize=leak -ggdb")

file(GLOB_RECURSE SOURCES "*.c")
set(SOURCES ${SOURCES} ${CMAKE_SOURCE_DIR}/extern/argparse/argparse.c)
#target_compile_definitions(../../bin/host_debug PRIVATE -DDEBUG)

add_executable(fft_fpga ${SOURCES})

target_link_libraries(fft_fpga PUBLIC ${IntelFPGAOpenCL_LIBRARIES})