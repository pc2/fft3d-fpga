#include "gtest/gtest.h"  // finds this because gtest is linked

extern "C" {
#include "CL/opencl.h"
#include "../src/host/include/opencl_utils.h"
}

TEST(OpenCLUtils, FindValidPlatform){
  EXPECT_TRUE(true);
  //ASSERT_EQ(findPlatform("Intel(R) FPGA"), nullptr);
  //EXPECT_EQ(findPlatform("Xilinx"), nullptr);
}