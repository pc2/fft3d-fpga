//  Author: Arjun Ramaswami

#include <iostream>
#include "gtest/gtest.h" 
#include <fftw3.h>
#include "helper.hpp"

extern "C" {
  #include "CL/opencl.h"
  #include "fftfpga/fftfpga.h"
}

/**
 * \brief fftfpgaf_c2c_3d_bram()
 */
TEST(fft3dFPGATest, InputValidityBRAM){
  const unsigned N = 64;
  
  const size_t sz = sizeof(float2) * N * N * N;
  float2 *test = (float2*)malloc(sz);
  fpga_t fft_time = {0.0, 0.0, 0.0, 0};

  // null inp ptr input
  fft_time = fftfpgaf_c2c_3d_bram(64, NULL, test, 0, 0);
  EXPECT_EQ(fft_time.valid, 0);

  // null out ptr input
  fft_time = fftfpgaf_c2c_3d_bram(64, test, NULL, 0, 0);
  EXPECT_EQ(fft_time.valid, 0);

  // if N not a power of 2
  fft_time = fftfpgaf_c2c_3d_bram(63, test, test, 0, 0);
  EXPECT_EQ(fft_time.valid, 0);

  free(test);
}

/**
 * \brief fftfpgaf_c2c_3d_ddr()
 */
TEST(fft3dFPGATest, InputValidityDDR){
  const unsigned N = 64;
  const size_t sz = sizeof(float2) * N * N * N;

  float2 *test = (float2*)malloc(sz);
  fpga_t fft_time = {0.0, 0.0, 0.0, 0};

  // null inp ptr input
  fft_time = fftfpgaf_c2c_3d_ddr(64, NULL, test, 0);
  EXPECT_EQ(fft_time.valid, 0);

  // null out ptr input
  fft_time = fftfpgaf_c2c_3d_ddr(64, test, NULL, 0);
  EXPECT_EQ(fft_time.valid, 0);

  // if N not a power of 2
  fft_time = fftfpgaf_c2c_3d_ddr(63, test, test, 0);
  EXPECT_EQ(fft_time.valid, 0);

  free(test);
}

/**
 * \brief fftfpgaf_c2c_3d_ddr_svm_batch()
 */
TEST(fft3dFPGATest, InputValidityDDRSVMBatch){
  const unsigned N = 64;
  const size_t sz = sizeof(float2) * N * N * N* 2;

  float2 *test = (float2*)malloc(sz);
  fpga_t fft_time = {0.0, 0.0, 0.0, 0};

  // null inp ptr input
  fft_time = fftfpgaf_c2c_3d_ddr_svm_batch(64, NULL, test, 0, 2);
  EXPECT_EQ(fft_time.valid, 0);

  // null out ptr input
  fft_time = fftfpgaf_c2c_3d_ddr_svm_batch(64, test, NULL, 0, 2);
  EXPECT_EQ(fft_time.valid, 0);

  // if N not a power of 2
  fft_time = fftfpgaf_c2c_3d_ddr_svm_batch(63, test, test, 0, 2);
  EXPECT_EQ(fft_time.valid, 0);

  // howmany is 0 
  fft_time = fftfpgaf_c2c_3d_ddr_svm_batch(63, test, test, 0, 0);
  EXPECT_EQ(fft_time.valid, 0);

  // howmany is negative
  fft_time = fftfpgaf_c2c_3d_ddr_svm_batch(63, test, test, 0, -1);
  EXPECT_EQ(fft_time.valid, 0);

  free(test);
}
