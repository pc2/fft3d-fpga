//  Author: Arjun Ramaswami

#include <iostream>
#include <math.h>
#include <fftw3.h>

#include "gtest/gtest.h" 
extern "C" {
  #include "CL/opencl.h"
  #include "fftfpga/fftfpga.h"
}

/**
 * \brief fpga_initialize()
 */
TEST(fftFPGASetupTest, ValidInit){
  // empty path argument
  EXPECT_EQ(fpga_initialize("Intel(R) FPGA", "", false), -1);

  // wrong platform name
  EXPECT_EQ(fpga_initialize("TEST", "fft1d_emulate.aocx", false), -2);

  // wrong path argument
  const char* platform_name = "intel(r) fpga sdk for opencl(tm)";
  EXPECT_EQ(fpga_initialize(platform_name, "TEST", false), -4);

  // right path and platform names
  const char* path = "p520_hpc_sg280l/emulation/fft3d_bram_64_nointer/fft3d_bram.aocx";
  EXPECT_EQ(fpga_initialize(platform_name, path, false), 0);
  fpga_final();
}

/**
 * \brief fftfpga_complex_malloc()
 */
TEST(fftFPGASetupTest, ValidDpMalloc){
  // request zero size
  EXPECT_EQ(fftfpga_complex_malloc(0), nullptr);
}

/**
 * \brief fftfpgaf_complex_malloc()
 */
TEST(fftFPGASetupTest, ValidSpMalloc){
  // request zero size
  EXPECT_EQ(fftfpgaf_complex_malloc(0), nullptr);
}