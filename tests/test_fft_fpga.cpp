/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

#include "gtest/gtest.h"  // finds this because gtest is linked

extern "C" {
  #include "CL/opencl.h"
  #include "../src/host/include/fftfpga.h"
  #include "../src/host/include/helper.h"
}

class fftFPGATest : public :: testing :: Test {

  void SetUp(){}
  void TearDown(){}

  protected:
};

TEST_F(fftFPGATest, ValidInit){
  EXPECT_EQ(fpga_initialize("Intel(R) FPGA", ""), 0);
  EXPECT_EQ(fpga_initialize("Intel(R) FPGA", "TEST"), 0);
  EXPECT_EQ(fpga_initialize("TEST", "fft1d.aocx"), 0);
  EXPECT_EQ(fpga_initialize("Intel(R) FPGA", "../../src/bin/fft1d.aocx"), 1);
}

TEST_F(fftFPGATest, ValidSp1dFFT){
  int N = 64, iter = 1, inv = 0;
  fpga_t timing = {0.0, 0.0, 0.0};

  float2 *inp = (float2 *)alignedMalloc(sizeof(float2) * N * iter);
  float2 *out = (float2 *)alignedMalloc(sizeof(float2) * N * iter);

  ASSERT_EQ(fpga_initialize("Intel(R) FPGA", "../../src/bin/fft1d.aocx"), 1);

  fftf_create_data(inp, N * iter);

  timing = fftfpgaf_c2c_1d(N, inp, out, inv, iter);
  printf("timing = %lf\n", timing.pcie_read_t);

  free(inp);
  free(out);
}