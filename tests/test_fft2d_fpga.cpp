//  Author: Arjun Ramaswami

#include <iostream>
#include "gtest/gtest.h" 
#include <math.h>
#include <fftw3.h>
#include "helper.hpp"

extern "C" {
  #include "CL/opencl.h"
  #include "fftfpga/fftfpga.h"
}

/**
 * \brief fftfpgaf_c2c_2d_bram()
 */
TEST(fft2dFPGATest, InputValidityBRAM){
  const unsigned N = 64;

  size_t sz = sizeof(float2) * N * N;
  float2 *test = (float2*)malloc(sz);
  fpga_t fft_time = {0.0, 0.0, 0.0, 0};

  // null inp ptr input
  fft_time = fftfpgaf_c2c_2d_bram(64, NULL, test, 0, 0, 1);
  EXPECT_EQ(fft_time.valid, 0);

  // null out ptr input
  fft_time = fftfpgaf_c2c_2d_bram(64, test, NULL, 0, 0, 1);
  EXPECT_EQ(fft_time.valid, 0);

  // if N not a power of 2
  fft_time = fftfpgaf_c2c_2d_bram(63, test, test, 0, 0, 1);
  EXPECT_EQ(fft_time.valid, 0);

  free(test);
}

/**
 * \brief fftfpgaf_c2c_2d_ddr()
 */
TEST(fft2dFPGATest, InputValidityDDR){
  const unsigned N = 64;

  size_t sz = sizeof(float2) * N * N;
  float2 *test = (float2*)malloc(sz);
  fpga_t fft_time = {0.0, 0.0, 0.0, 0};

  // null inp ptr input
  fft_time = fftfpgaf_c2c_2d_ddr(64, NULL, test, 0);
  EXPECT_EQ(fft_time.valid, 0);

  // null out ptr input
  fft_time = fftfpgaf_c2c_2d_ddr(64, test, NULL, 0);
  EXPECT_EQ(fft_time.valid, 0);

  // if N not a power of 2
  fft_time = fftfpgaf_c2c_2d_ddr(63, test, test, 0);
  EXPECT_EQ(fft_time.valid, 0);

  free(test);
}