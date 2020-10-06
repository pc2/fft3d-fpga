# Arjun Ramaswami
cmake_minimum_required(VERSION 3.10)
project(examplesfftfpga VERSION 0.1
            DESCRIPTION "Example Code that uses libfftfpga"
            LANGUAGES C CXX)

set(examples fft3d_ddr fft3d_bram fft3d_ddr_svm fft3d_ddr_batch fft3d_ddr_svm_batch fft2d fft1d fft1d_svm)

# create a target for each of the example 
foreach(example ${examples})

  add_executable(${example} ${example}.c
                  common/helper.c
                  common/verify_fftw.c)

  target_compile_options(${example}
      PRIVATE -Wall -Werror)
      
  target_include_directories(${example}
      PRIVATE  ${IntelFPGAOpenCL_INCLUDE_DIRS}
                "${CMAKE_SOURCE_DIR}/extern/argparse"
                common)
    
  target_link_libraries(${example}
      PRIVATE ${IntelFPGAOpenCL_LIBRARIES} fftfpga argparse m)

if(FFTW_FOUND)
  target_compile_definitions(${example} PRIVATE USE_FFTW)
  target_link_libraries(${example} PUBLIC fftw3 fftw3f)
else()
  message(WARNING, "FFTW library not found. Cannot perform verification")
endif()

endforeach()