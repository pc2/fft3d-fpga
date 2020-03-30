/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

#include "gtest/gtest.h"  // finds this because gtest is linked
#include <stdlib.h>

extern "C" {
  #include "CL/opencl.h"
  #include "../src/host/include/fftfpga.h"
  #include "../src/host/include/helper.h"
}

/**
 *  \brief Tests whether valid data is created
 */
TEST(HelperTest, CreateValidData){
  int N = 8;
  float2 *inp = (float2*)malloc(sizeof(float2) * N);

  int status = fftf_create_data(inp, 1);
  EXPECT_NE(status, -1);
  EXPECT_NE(status, -2);
  EXPECT_NE(status, 0);

  free(inp);
}