// Author: Arjun Ramaswami

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#define CL_VERSION_2_0
#include <CL/cl_ext_intelfpga.h> // to disable interleaving & transfer data to specific banks - CL_CHANNEL_1_INTELFPGA
#include "CL/opencl.h"

#include "fpga_state.h"
#include "fftfpga/fftfpga.h"
#include "svm.h"
#include "opencl_utils.h"
#include "misc.h"

cl_platform_id platform = NULL;
cl_device_id *devices;
cl_device_id device = NULL;
cl_context context = NULL;
cl_program program = NULL;

cl_command_queue queue1 = NULL, queue2 = NULL, queue3 = NULL;
cl_command_queue queue4 = NULL, queue5 = NULL, queue6 = NULL;
cl_command_queue queue7 = NULL, queue8 = NULL;

//static int svm_handle;
bool svm_enabled = false;

/** 
 * @brief Allocate memory of double precision complex floating points
 * @param sz  : size_t - size to allocate
 * @param svm : 1 if svm
 * @return void ptr or NULL
 */
void* fftfpga_complex_malloc(size_t sz){
  if(sz == 0){
    return NULL;
  }
  else{
    return ((double2 *)alignedMalloc(sz));
  }
}

/** 
 * @brief Allocate memory of single precision complex floating points
 * @param sz  : size_t : size to allocate
 * @param svm : 1 if svm
 * @return void ptr or NULL
 */
void* fftfpgaf_complex_malloc(size_t sz){

  if(sz == 0){
    return NULL;
  }
  return ((float2 *)alignedMalloc(sz));
}

/** 
 * @brief Initialize FPGA
 * @param platform name: string - name of the OpenCL platform
 * @param path         : string - path to binary
 * @param use_svm      : 1 if true 0 otherwise
 * @return 0 if successful 
          -1 Path to binary missing
          -2 Unable to find platform passed as argument
          -3 Unable to find devices for given OpenCL platform
          -4 Failed to create program, file not found in path
          -5 Device does not support required SVM
*/
int fpga_initialize(const char *platform_name, const char *path, bool use_svm){
  cl_int status = 0;

  printf("-- Initializing FPGA ...\n");
  // Path to binary missing
  if(path == NULL || strlen(path) == 0){
    return -1;
  }

  // Check if this has to be sent as a pointer or value
  // Get the OpenCL platform.
  platform = findPlatform(platform_name);
  // Unable to find given OpenCL platform
  if(platform == NULL){
    return -2;
  }
  // Query the available OpenCL devices.
  cl_uint num_devices;
  devices = getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices);
  // Unable to find device for the OpenCL platform
  printf("\n-- %u devices found\n", num_devices);
  if(devices == NULL){
    return -3;
  }

  // use the first device.
  device = devices[0];
  printf("\tChoosing first device by default\n");

  if(use_svm){
    if(!check_valid_svm_device(device)){
      return -5;
    }
    else{
      printf("-- Device supports SVM \n");
      svm_enabled = true;
    }
  }

  // Create the context.
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
  checkError(status, "Failed to create context");

  printf("\n-- Getting program binary from path: %s\n", path);
  // Create the program.
  program = getProgramWithBinary(context, &device, 1, path);
  if(program == NULL) {
    fprintf(stderr, "Failed to create program\n");
    fpga_final();
    return -4;
  }

  printf("-- Building the program\n\n");
  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  return 0;
}

/** 
 * @brief Release FPGA Resources
 */
void fpga_final(){
  printf("-- Cleaning up FPGA resources ...\n");
  if(program) 
    clReleaseProgram(program);
  if(context)
    clReleaseContext(context);
  free(devices);
}

/**
 * \brief Create a command queue for each kernel
 */
void queue_setup(){
  cl_int status = 0;
  // Create one command queue for each kernel.
  queue1 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue1");
  queue2 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue2");
  queue3 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue3");
  queue4 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue4");
  queue5 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue5");
  queue6 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue6");
  queue7 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue6");
  queue8 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue6");
}

/**
 * \brief Release all command queues
 */
void queue_cleanup() {
  if(queue1) 
    clReleaseCommandQueue(queue1);
  if(queue2) 
    clReleaseCommandQueue(queue2);
  if(queue3) 
    clReleaseCommandQueue(queue3);
  if(queue4) 
    clReleaseCommandQueue(queue4);
  if(queue5) 
    clReleaseCommandQueue(queue5);
  if(queue6) 
    clReleaseCommandQueue(queue6);
  if(queue7) 
    clReleaseCommandQueue(queue7);
  if(queue8) 
    clReleaseCommandQueue(queue8);
}