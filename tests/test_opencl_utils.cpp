#include "gtest/gtest.h"  // finds this because gtest is linked

extern "C" {
  #include "CL/opencl.h"
  #include "../src/host/include/opencl_utils.h"
}

class OpenCLUtilsTest : public :: testing :: Test {

  void SetUp(){}
  void TearDown() override {
  }

  protected:
    static cl_platform_id pl_id;
    static cl_device_id device;
};

cl_platform_id OpenCLUtilsTest :: pl_id = NULL;
cl_device_id OpenCLUtilsTest :: device = NULL;

TEST_F(OpenCLUtilsTest, FindValidPlatform){
  pl_id = findPlatform("Intel(R) FPGA");
  ASSERT_NE(pl_id, nullptr);
}

TEST_F(OpenCLUtilsTest, FindValidDevice){
  cl_uint count = 0;
  cl_device_id *devices;
  devices = getDevices(pl_id, CL_DEVICE_TYPE_ALL, &count); 
  device = devices[0];

  EXPECT_NE(device, nullptr);
  EXPECT_NE(count, 0);
}

TEST_F(OpenCLUtilsTest, FindValidProgram){
  cl_int status = 0;
  
  cl_context context = clCreateContext(NULL, 1, &device, &openCLContextCallBackFxn, NULL, &status);
  checkError(status, "Context Fail");
  
  ASSERT_EQ(getProgramWithBinary(context, &device, 1, NULL), nullptr);
}

TEST_F(OpenCLUtilsTest, CreateValidAlloc){
  float *ptr; 
  ptr = (float *)alignedMalloc(sizeof(float));
  EXPECT_NE(ptr, nullptr);
  free(ptr);
}