# Arjun Ramaswami
cmake_minimum_required(VERSION 3.10)
project(examplesfftfpga VERSION 0.1
            DESCRIPTION "Example Code that uses libfftfpga"
            LANGUAGES C CXX)

set(examples fft3d fft2d fft1d)

# create a target for each of the example 
foreach(example ${examples})

  add_executable(${example} ${example}.c
                  common/helper.c)  

  target_compile_options(${example}
      PRIVATE -Wall -Werror)
      
  if(USE_DEBUG)
    target_compile_definitions(${example} PRIVATE DEBUG)
  endif()

  target_include_directories(${example}
      PRIVATE  ${IntelFPGAOpenCL_INCLUDE_DIRS}
                "${CMAKE_SOURCE_DIR}/extern/argparse"
                common)
    
  target_link_libraries(${example}
      PRIVATE ${IntelFPGAOpenCL_LIBRARIES} fftfpga argparse m)

endforeach()


# GTest Setup
#if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
#  option(PACKAGE_TESTS "Build the tests" ON)
#endif()


#if(PACKAGE_TESTS)
#    enable_testing()
#    include(GoogleTest)
#    add_subdirectory(tests)
#endif()

#${CMAKE_SOURCE_DIR}/extern/argparse/argparse.c
# "${CMAKE_SOURCE_DIR}/extern/argparse"