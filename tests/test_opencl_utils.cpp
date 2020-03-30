/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

#include "gtest/gtest.h"  // finds this because gtest is linked

extern "C" {
  #include "CL/opencl.h"
  #include "../src/host/include/opencl_utils.h"
}

class OpenCLUtilsTest : public :: testing :: Test {

  void SetUp(){}
  void TearDown() override {
    queue_cleanup();
    fpga_final();
  }

  protected:
    static cl_platform_id pl_id;
    static cl_device_id device;
};

cl_platform_id OpenCLUtilsTest :: pl_id = NULL;
cl_device_id OpenCLUtilsTest :: device = NULL;

/**
 *  \brief Tests whether a valid platform is found by testing if the platform id returned is NULL
 */
TEST_F(OpenCLUtilsTest, FindValidPlatform){
  pl_id = findPlatform("Intel(R) FPGA");
  ASSERT_NE(pl_id, nullptr);
}

/**
 *  \brief Tests whether a valid device is found by testing if the device id returned is NULL and if the number of devices found is zero.
 */
TEST_F(OpenCLUtilsTest, FindValidDevice){
  cl_uint count = 0;
  cl_device_id *devices;
  devices = getDevices(pl_id, CL_DEVICE_TYPE_ALL, &count); 
  EXPECT_NE(count, 0);
  ASSERT_NE(devices, nullptr);

  device = devices[0];
  free(devices);
}

/**
 *  \brief Tests whether a valid program is created by testing if the program returned is NULL.
 */
TEST_F(OpenCLUtilsTest, CreateValidProgram){
  cl_int status = 0;
  
  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
  checkError(status, "Context Fail");
  
  EXPECT_EQ(getProgramWithBinary(context, &device, 1, NULL), nullptr);
}

/**
 *  \brief Tests whether a valid allocation of data is created by testing whether the pointer to the buffer returned is NULL.
 */
TEST_F(OpenCLUtilsTest, CreateValidAlloc){
  float *ptr; 
  ptr = (float *)alignedMalloc(sizeof(float));
  EXPECT_NE(ptr, nullptr);
  free(ptr);
}