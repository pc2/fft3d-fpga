/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

#include "gtest/gtest.h"  // finds this because gtest is linked

extern "C" {
  #include "CL/opencl.h"
  #include "../src/host/include/fftfpga.h"
  #include "../src/host/include/helper.h"
#ifdef USE_FFTW
  #include <fftw3.h>
#endif
}

class fftFPGATest : public :: testing :: Test {

  void SetUp(){}
  void TearDown(){}

  protected:
};

/**
 * \brief fpga_initialize()
 */
TEST_F(fftFPGATest, ValidInit){
  // empty path argument
  EXPECT_EQ(fpga_initialize("Intel(R) FPGA", "", 0, 1), 1);
  // wrong path argument
  EXPECT_EQ(fpga_initialize("Intel(R) FPGA", "TEST", 0, 1), 1);
  // wrong platform name
  EXPECT_EQ(fpga_initialize("TEST", "fft1d_emulate.aocx", 0, 1), 1);
  // right path and platform names
  //EXPECT_EQ(fpga_initialize("Intel(R) FPGA", "64pt_fft1d_emulate.aocx", 0, 1), 0);
}

/**
 * \brief fftfpga_complex_malloc()
 */
TEST_F(fftFPGATest, ValidDpMalloc){
  // request zero size
  EXPECT_EQ(fftfpga_complex_malloc(0, 0), nullptr);
  // TODO: do not support svm
  EXPECT_EQ(fftfpga_complex_malloc(0, 1), nullptr);
}

/**
 * \brief fftfpgaf_complex_malloc()
 */
TEST_F(fftFPGATest, ValidSpMalloc){
  // request zero size
  EXPECT_EQ(fftfpgaf_complex_malloc(0, 0), nullptr);
  // TODO: do not support svm
  EXPECT_EQ(fftfpgaf_complex_malloc(0, 1), nullptr);
}

/**
 * \brief fftfpga_c2c_1d()
 */
TEST_F(fftFPGATest, ValidDp1dFFT){
  int N = 64;
  size_t sz = sizeof(double2) * N;
  double2 *inp = (double2*)fftfpga_complex_malloc(sz, 0);
  double2 *out = (double2*)fftfpga_complex_malloc(sz, 0);

  // null inp ptr input
  fpga_t fft_time = fftfpga_c2c_1d(64, NULL, out, 0, 1);
  EXPECT_EQ(fft_time.valid, 0);

  // null out ptr input
  fft_time = fftfpga_c2c_1d(64, inp, NULL, 0, 1);
  EXPECT_EQ(fft_time.valid, 0);

  // if N not a power of 2
  fft_time = fftfpga_c2c_1d(63, inp, out, 0, 1);
  EXPECT_EQ(fft_time.valid, 0);

  // check correctness of output
#ifdef USE_FFTW
  // malloc data to input
  fft_time = fftfpga_c2c_1d(64, inp, out, 0, 1);

  fftw_complex *fftw_dp_data = (fftw_complex*)fftw_malloc(sz);
  fftw_plan plan = fftw_plan_dft_1d( N, &fftw_dp_data[0], &fftw_dp_data[0], FFTW_FORWARD, FFTW_ESTIMATE);

  // get same data for fftw
  fftw_execute(plan);

  // verification

  fftw_free(fftw_dp_data);
  fftw_destroy_plan(plan);
#endif

  free(inp);
  free(out);
}

/*
TEST_F(fftFPGATest, ValidSp1dFFT){
  int N = 64, iter = 1, inv = 0;
  fpga_t timing = {0.0, 0.0, 0.0};

  float2 *inp = (float2 *)alignedMalloc(sizeof(float2) * N * iter);
  float2 *out = (float2 *)alignedMalloc(sizeof(float2) * N * iter);

  ASSERT_EQ(fpga_initialize("Intel(R) FPGA", "fft1d.aocx", 0, 1), 1);

  fftf_create_data(inp, N * iter);

  timing = fftfpgaf_c2c_1d(N, inp, out, inv, iter);
  printf("timing = %lf\n", timing.pcie_read_t);

  free(inp);
  free(out);
}
*/