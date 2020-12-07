//  Author: Arjun Ramaswami

#include "gtest/gtest.h"  // finds this because gtest is linked
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

extern "C" {
  #include "CL/opencl.h"
  #include "fftfpga/fftfpga.h"
  #include "helper.h"
  #include "verify_fftw.h"

#ifdef USE_FFTW
  #include <fftw3.h>
#endif
}

/**
 * \brief fftfpgaf_c2c_1d()
 */
TEST(fft1dFPGATest, InputValidity){
  const int N = (1 << 6);

  size_t sz = sizeof(float2) * N;
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

TEST(fft1dFPGATest, CorrectnessSp){
  // check correctness of output
#ifdef USE_FFTW
  const int logN = 6;
  int N = (1 << logN);

  size_t sz = sizeof(float2) * N;
  float2 *inp = (float2*)fftfpgaf_complex_malloc(sz);
  float2 *out = (float2*)fftfpgaf_complex_malloc(sz);

  // malloc data to input
  fftf_create_data(inp, N);

  int isInit= fpga_initialize("Intel(R) FPGA", "emu_64_fft1d/fft1d.aocx", false);
  ASSERT_EQ(isInit, 0);

  fpga_t fft_time = fftfpgaf_c2c_1d(64, inp, out, 0, 1);

  bool result = verify_fftwf(out, inp, N, 1, false, 1);
  EXPECT_TRUE(result);

  free(inp);
  free(out);

  fpga_final();
#endif
}

/**
 * \brief fftfpgaf_c2c_1d()
 */
TEST(fft1dFPGATest, InputValiditySVM){
  const int N = (1 << 6);

  size_t sz = sizeof(float2) * N;
  float2 *test = (float2*)malloc(sz);
  fpga_t fft_time = {0.0, 0.0, 0.0, 0};

  // svm not enabled
  fft_time = fftfpgaf_c2c_1d_svm(N, test, test, false, 1);
  EXPECT_EQ(fft_time.valid, 0);

  int isInit = fpga_initialize("Intel(R) FPGA", "emu_64_fft1d/ff1d.aocx", true);
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