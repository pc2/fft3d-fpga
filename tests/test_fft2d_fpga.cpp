//  Author: Arjun Ramaswami

#include "gtest/gtest.h"  // finds this because gtest is linked
#include <stdlib.h>   // malloc, free
#include <math.h>
#ifdef USE_FFTW
  #include <fftw3.h>
#endif

extern "C" {
  #include "CL/opencl.h"
  #include "fftfpga/fftfpga.h"
  #include "helper.h"
  #include "verify_fftw.h"
}

/**
 * \brief fftfpgaf_c2c_2d_bram()
 */
TEST(fft2dFPGATest, InputValidityBRAM){
  const int N = 64;

  size_t sz = sizeof(float2) * N * N;
  float2 *test = (float2*)malloc(sz);
  fpga_t fft_time = {0.0, 0.0, 0.0, 0};

  // null inp ptr input
  fft_time = fftfpgaf_c2c_2d_bram(64, NULL, test, 0, 0);
  EXPECT_EQ(fft_time.valid, 0);

  // null out ptr input
  fft_time = fftfpgaf_c2c_2d_bram(64, test, NULL, 0, 0);
  EXPECT_EQ(fft_time.valid, 0);

  // if N not a power of 2
  fft_time = fftfpgaf_c2c_2d_bram(63, test, test, 0, 0);
  EXPECT_EQ(fft_time.valid, 0);

  free(test);
}

TEST(fft2dFPGATest, CorrectnessBRAM){
  // check correctness of output
#ifdef USE_FFTW
  // malloc data to input
  const int N = (1 << 6);
  fpga_t fft_time = {0.0, 0.0, 0.0, 0};

  int isInit = fpga_initialize("Intel(R) FPGA", "emu_64_fft2d_bram/fft2d_bram.aocx", 0);
  EXPECT_EQ(isInit, 0);

  size_t sz = sizeof(float2) * N * N;
  float2 *inp = (float2*)fftfpgaf_complex_malloc(sz, 0);
  float2 *out = (float2*)fftfpgaf_complex_malloc(sz, 0);

  fftf_create_data(inp, N * N);

  fft_time = fftfpgaf_c2c_2d_bram(N, inp, out, 0, 0);

  int result = verify_sp_fft2d_fftw(out, inp, N, 0);

  EXPECT_EQ(result, 1);

  free(inp);
  free(out);

  fpga_final();
#endif
}

/**
 * \brief fftfpgaf_c2c_2d_ddr()
 */
TEST(fft2dFPGATest, InputValidityDDR){
  const int N = 64;

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

/**
 * \brief fftfpgaf_c2c_2d_ddr()
 */
TEST(fftFPGATest, ValidSp2dFFTDDR){
   // check correctness of output
#ifdef USE_FFTW
  // malloc data to input
  const int N = (1 << 6);
  fpga_t fft_time = {0.0, 0.0, 0.0, 0};

  int isInit = fpga_initialize("Intel(R) FPGA", "emu_64_fft2d_ddr/fft2d_ddr.aocx", 0);
  ASSERT_EQ(isInit, 0);

  size_t sz = sizeof(float2) * N * N;
  float2 *inp = (float2*)fftfpgaf_complex_malloc(sz, 0);
  float2 *out = (float2*)fftfpgaf_complex_malloc(sz, 0);

  fftf_create_data(inp, N * N);

  fft_time = fftfpgaf_c2c_2d_ddr(N, inp, out, 0);

  int result = verify_sp_fft2d_fftw(out, inp, N, 0);

  EXPECT_EQ(result, 1);

  free(inp);
  free(out);

  fpga_final();
#endif
}