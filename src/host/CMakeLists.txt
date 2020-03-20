
include_directories(../../extern/argparse include)
include_directories(${IntelFPGAOpenCL_INCLUDE_DIRS})

file(GLOB_RECURSE SOURCES "*.c")

add_compile_options(--std=c99)

add_executable(../../bin/host ${SOURCES})
target_link_libraries(../../bin/host ${IntelFPGAOpenCL_LIBRARIES})