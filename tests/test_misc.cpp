// Author: Arjun Ramaswami

#include "gtest/gtest.h"  // finds this because gtest is linked
#include <stdbool.h>

extern "C" {
  #include "CL/opencl.h"
  #include "helper.h"
}

/**
 *  \brief fftf_create_data
 */
TEST(HelperTest, CreateValidRandomSpData){
  int N = 8;
  size_t sz = sizeof(float2) * N;
  float2 *inp = (float2*)fftfpgaf_complex_malloc(sz);

  // sz 0
  EXPECT_FALSE(fftf_create_data(0, 1));

  // good input
  EXPECT_TRUE(fftf_create_data(inp, N));

  free(inp);
}

/**
 *  \brief fft_create_data
 */
TEST(HelperTest, CreateValidRandomDpData){
  int N = 8;
  size_t sz = sizeof(double2) * N;
  double2 *inp = (double2*)fftfpga_complex_malloc(sz);

  // sz 0
  EXPECT_FALSE(fft_create_data(0, 1));

  // good input
  EXPECT_TRUE(fft_create_data(inp, N));

  free(inp);
}