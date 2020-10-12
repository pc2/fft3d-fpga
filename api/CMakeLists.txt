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
              ${PROJECT_SOURCE_DIR}/src/fft3d.c
              ${PROJECT_SOURCE_DIR}/src/fft2d.c
              ${PROJECT_SOURCE_DIR}/src/fft1d.c
              ${PROJECT_SOURCE_DIR}/src/svm.c
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

##
# Doxygen Build
##
option(BUILD_DOC "Build documentation" OFF)

if(BUILD_DOC) 

  find_package(Doxygen)
  if(DOXYGEN_FOUND)
    
    # Doxygen Options
    set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/docs)
    set(DOXYGEN_PROJECT_NAME "FFTFPGA")
    set(DOXYGEN_PROJECT_BRIEF "OpenCL based FFT library for FPGAs")
    set(DOXYGEN_GENERATE_LATEX YES)
    set(DOXYGEN_OPTIMIZE_OUTPUT_FOR_C YES)
    set(DOXYGEN_SHOW_FILES YES)

    doxygen_add_docs(doc_doxygen
      ${PROJECT_SOURCE_DIR}/include/fftfpga/fftfpga.h 
      COMMENT "Generate library documentation"
    )

  else (DOXYGEN_FOUND)
    message(WARNING, "Doxygen need to be installed to generate the doxygen documentation")
  endif (DOXYGEN_FOUND)
endif()