//  Author: Arjun Ramaswami

#include "gtest/gtest.h"  // finds this because gtest is linked
#include <iostream>
#include <fftw3.h>

extern "C" {
  #include "CL/opencl.h"
  #include "fftfpga/fftfpga.h"
}
#include "helper.hpp"

/**
 * \brief fftfpgaf_c2c_1d()
 */
TEST(fft1dFPGATest, InputValidity){
  const unsigned N = (1 << 6);
  const size_t sz = sizeof(float2) * N;

  float2 *test = (float2*)malloc(sz);
  fpga_t fft_time = {0.0, 0.0, 0.0, 0};

  // null inp ptr input
  fft_time = fftfpgaf_c2c_1d(N, NULL, test, false, 1);
  EXPECT_EQ(fft_time.valid, 0);

  // null out ptr input
  fft_time = fftfpgaf_c2c_1d(N, test, NULL, false, 1);
  EXPECT_EQ(fft_time.valid, 0);

  // if N not a power of 2
  fft_time = fftfpgaf_c2c_1d(N-1, test, test, false, 1);
  EXPECT_EQ(fft_time.valid, 0);

  free(test);
}

/**
 * \brief fftfpgaf_c2c_1d()
 */
TEST(fft1dFPGATest, InputValiditySVM){
  const unsigned N = (1 << 6);

  size_t sz = sizeof(float2) * N;
  float2 *test = (float2*)malloc(sz);
  fpga_t fft_time = {0.0, 0.0, 0.0, 0};

  // svm not enabled
  fft_time = fftfpgaf_c2c_1d_svm(N, test, test, false, 1);
  EXPECT_EQ(fft_time.valid, 0);

  int isInit = fpga_initialize("intel(r) fpga sdk for opencl(tm)", "p520_hpc_sg280l/emulation/fft1d_64_nointer/fft1d.aocx", true);
  ASSERT_EQ(isInit, 0);

  // null inp ptr input
  fft_time = fftfpgaf_c2c_1d_svm(N, NULL, test, false, 1);
  EXPECT_EQ(fft_time.valid, 0);

  // null out ptr input
  fft_time = fftfpgaf_c2c_1d_svm(N, test, NULL, false, 1);
  EXPECT_EQ(fft_time.valid, 0);

  // if N not a power of 2
  fft_time = fftfpgaf_c2c_1d_svm(N-1, test, test, false, 1);
  EXPECT_EQ(fft_time.valid, 0);

  free(test);

  fpga_final();
}