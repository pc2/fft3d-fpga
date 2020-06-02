//  Author: Arjun Ramaswami

#include "gtest/gtest.h"  // finds this because gtest is linked

extern "C" {
  #include "CL/opencl.h"
  #include "opencl_utils.h"
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
 *  \brief findPlatform 
 */
TEST_F(OpenCLUtilsTest, FindValidPlatform){
  // empty platform name
  EXPECT_EQ(findPlatform(""), nullptr);

  // bad platform name
  EXPECT_EQ(findPlatform("test"), nullptr);

  // correct platform name
  pl_id = findPlatform("Intel(R) FPGA");
  ASSERT_NE(pl_id, nullptr);
}

/**
 *  \brief getDevices
 */
TEST_F(OpenCLUtilsTest, FindValidDevice){
  cl_uint num_devices;
  cl_device_id *devices;

  // bad platform id
  EXPECT_EQ(getDevices(NULL, CL_DEVICE_TYPE_ALL, &num_devices), nullptr);

  // correct input
  devices = getDevices(pl_id, CL_DEVICE_TYPE_ALL, &num_devices); 

  // test for number of devices greater than 0
  EXPECT_NE(num_devices, 0);

  // test for correct device
  ASSERT_NE(devices, nullptr);

  device = devices[0];
  free(devices);
}

/**
 *  \brief getProgramWithBinary()
 */
TEST_F(OpenCLUtilsTest, CreateValidProgram){
  cl_int status = 0;
  const char *path = "64pt_fft1d_emulate.aocx";

  // bad context
  cl_context bad_context;
  EXPECT_EQ(getProgramWithBinary(bad_context, &device, 1, path), nullptr);

  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
  checkError(status, "Context Fail");

  // bad device_id
  cl_device_id bad_device;
  EXPECT_EQ(getProgramWithBinary(context, &bad_device, 1, path), nullptr);

  // 0 devices
  EXPECT_EQ(getProgramWithBinary(context, &device, 0, path), nullptr);

  // wrong path
  EXPECT_EQ(getProgramWithBinary(context, &device, 1, NULL), nullptr);

  // right path
  //EXPECT_NE(getProgramWithBinary(context, &device, 1, path), NULL);
}