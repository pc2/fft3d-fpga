cmake_minimum_required(VERSION 3.10)
project(examplesfftfpga VERSION 2.0
            DESCRIPTION "Sample Code that uses libfftfpga"
            LANGUAGES C CXX)

set(examples fft)

# create a target for each of the example 
foreach(example ${examples})
  add_executable(${example} 
                  ${example}.cpp helper.cpp)

  target_compile_options(${example} PRIVATE -Wall -Werror)
      
  target_include_directories(${example}
    PRIVATE ${PROJECT_SOURCE_DIR} 
            ${IntelFPGAOpenCL_INCLUDE_DIRS} 
            ${CMAKE_BINARY_DIR}/include 
            ${FFTW_INCLUDE_DIRS})
    
  target_compile_definitions(${example} PRIVATE USE_FFTW)

  target_link_libraries(${example}
    PRIVATE cxxopts fftfpga fftw3 fftw3f
            ${IntelFPGAOpenCL_LIBRARIES})
endforeach()