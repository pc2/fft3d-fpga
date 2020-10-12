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

#ifdef VERBOSE
  printf("\tInitializing FPGA ...\n");
#endif

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
  if(devices == NULL){
    return -3;
  }

  // use the first device.
  device = devices[0];

  if(use_svm){
    if(!check_valid_svm_device(device)){
      return -5;
    }
    else{
      printf("Supports SVM \n");
      svm_enabled = true;
    }
  }

  // Create the context.
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
  checkError(status, "Failed to create context");

#ifdef VERBOSE
  printf("\tGetting program binary from path %s ...\n", path);
#endif
  // Create the program.
  program = getProgramWithBinary(context, &device, 1, path);
  if(program == NULL) {
    fprintf(stderr, "Failed to create program\n");
    fpga_final();
    return -4;
  }

#ifdef VERBOSE
  printf("\tBuilding program ...\n");
#endif
  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  return 0;
}

/** 
 * @brief Release FPGA Resources
 */
void fpga_final(){

#ifdef VERBOSE
  printf("\tCleaning up FPGA resources ...\n");
#endif
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

/**
 * \brief compute an batched out-of-place single precision complex 3D-FFT using the DDR of the FPGA for 3D Transpose
 * \param N    : integer pointer addressing the size of FFT3d  
 * \param inp  : float2 pointer to input data of size [N * N * N]
 * \param out  : float2 pointer to output data of size [N * N * N]
 * \param inv  : int toggle to activate backward FFT
 * \param interleaving : enable burst interleaved global memory buffers
 * \param how_many : number of batched computations
 * \return fpga_t : time taken in milliseconds for data transfers and execution
 */
/*
fpga_t fftfpgaf_c2c_3d_ddr_batch(int N, float2 *inp, float2 *out, bool inv, bool interleaving, int how_many) {
  fpga_t fft_time = {0.0, 0.0, 0.0, 0};
  cl_int status = 0;
  int num_pts = N * N * N;
  
  // if N is not a power of 2
  if(inp == NULL || out == NULL || ( (N & (N-1)) !=0) || (how_many <= 1)){
    return fft_time;
  }

#ifdef VERBOSE
  printf("Launching%s 3d FFT transform in DDR for Batched execution\n", inv ? " inverse":"");
#endif

  // Can't pass bool to device, so convert it to int
  int inverse_int = (int)inv;

  // Setup kernels
  cl_kernel fetch1_kernel = clCreateKernel(program, "fetchBitrev1", &status);
  checkError(status, "Failed to create fetch1 kernel");
  cl_kernel ffta_kernel = clCreateKernel(program, "fft3da", &status);
  checkError(status, "Failed to create fft3da kernel");
  cl_kernel transpose_kernel = clCreateKernel(program, "transpose", &status);
  checkError(status, "Failed to create transpose kernel");
  cl_kernel fftb_kernel = clCreateKernel(program, "fft3db", &status);
  checkError(status, "Failed to create fft3db kernel");
  cl_kernel store1_kernel = clCreateKernel(program, "transposeStore1", &status);
  checkError(status, "Failed to create store1 kernel");

  cl_kernel fetch2_kernel = clCreateKernel(program, "fetchBitrev2", &status);
  checkError(status, "Failed to create fetch2 kernel");
  cl_kernel fftc_kernel = clCreateKernel(program, "fft3dc", &status);
  checkError(status, "Failed to create fft3dc kernel");
  cl_kernel store2_kernel = clCreateKernel(program, "transposeStore2", &status);
  checkError(status, "Failed to create store2 kernel");

  // Setup Queues to the kernels
  queue_setup();

  // Device memory buffers: using 1st and 2nd banks
  // Double Buffers, using 3rd and 4th banks
  cl_mem d_inData1, d_inData2, d_outData1, d_outData2;
  d_inData1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_CHANNEL_1_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");

  d_inData2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_CHANNEL_3_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");

  d_outData1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_CHANNEL_1_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  d_outData2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_CHANNEL_3_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  cl_mem d_transpose;
  d_transpose = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  // Default Kernel Arguments
  status=clSetKernelArg(fetch1_kernel, 0, sizeof(cl_mem), (void *)&d_inData1);
  checkError(status, "Failed to set fetch1 kernel arg");
  status=clSetKernelArg(ffta_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set ffta kernel arg");
  status=clSetKernelArg(fftb_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set fftb kernel arg");
  status=clSetKernelArg(store1_kernel, 0, sizeof(cl_mem), (void *)&d_transpose);
  checkError(status, "Failed to set store1 kernel arg");

  status=clSetKernelArg(fetch2_kernel, 0, sizeof(cl_mem), (void *)&d_transpose);
  checkError(status, "Failed to set fetch2 kernel arg");
  status=clSetKernelArg(fftc_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set fftc kernel arg");
  status=clSetKernelArg(store2_kernel, 0, sizeof(cl_mem), (void *)&d_outData1);
  checkError(status, "Failed to set store2 kernel arg");

  fft_time.exec_t = getTimeinMilliSec();

  // First Phase 
  // Write to DDR first buffer
  status = clEnqueueWriteBuffer(queue1, d_inData1, CL_TRUE, 0, sizeof(float2) * num_pts, inp, 0, NULL, NULL);

  status = clFinish(queue1);
  checkError(status, "failed to finish");

  // Second Phase
  // Unblocking write to DDR second buffer from index num_pts
  cl_event write_event[2];
  status = clEnqueueWriteBuffer(queue6, d_inData2, CL_FALSE, 0, sizeof(float2) * num_pts, (void*)&inp[num_pts], 0, NULL, &write_event[0]);
  checkError(status, "Failed to write to DDR buffer");

  // Compute First FFT already transferred
  status = clEnqueueTask(queue1, fetch1_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fetch kernel");

  status = clEnqueueTask(queue2, ffta_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fft kernel");

  status = clEnqueueTask(queue3, transpose_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch transpose kernel");

  status = clEnqueueTask(queue4, fftb_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch second fft kernel");

  status = clEnqueueTask(queue5, store1_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch second transpose kernel");

  status = clEnqueueTask(queue5, fetch2_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fetch kernel");

  status = clEnqueueTask(queue4, fftc_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fft kernel");

  status = clEnqueueTask(queue3, store2_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch transpose kernel");

  // Check finish of transfer and computations
  status = clFinish(queue6);
  checkError(status, "failed to finish");
  status = clFinish(queue5);
  checkError(status, "failed to finish");
  status = clFinish(queue4);
  checkError(status, "failed to finish");
  status = clFinish(queue3);
  checkError(status, "failed to finish");
  status = clFinish(queue2);
  checkError(status, "failed to finish");
  status = clFinish(queue1);
  checkError(status, "failed to finish");

  clWaitForEvents(1, &write_event[0]);
  clReleaseEvent(write_event[0]);

  // Loop over the 3 stages
  for(size_t i = 2; i < how_many; i++){

    // Unblocking transfers between DDR and host 
    if( (i % 2) == 0){
      status = clEnqueueReadBuffer(queue6, d_outData1, CL_FALSE, 0, sizeof(float2) * num_pts, &out[((i - 2) * num_pts)], 0, NULL, &write_event[0]);
      checkError(status, "Failed to read from DDR buffer");

      status = clEnqueueWriteBuffer(queue7, d_inData1, CL_FALSE, 0, sizeof(float2) * num_pts, &inp[(i * num_pts)], 0, NULL, &write_event[1]);
      checkError(status, "Failed to write to DDR buffer");

      status=clSetKernelArg(fetch1_kernel, 0, sizeof(cl_mem), (void *)&d_inData2);
      checkError(status, "Failed to set fetch1 kernel arg");
    
      status=clSetKernelArg(store2_kernel, 0, sizeof(cl_mem), (void *)&d_outData2);
      checkError(status, "Failed to set store2 kernel arg");
    }
    else{
      status = clEnqueueReadBuffer(queue6, d_outData2, CL_FALSE, 0, sizeof(float2) * num_pts, &out[((i - 2) * num_pts)], 0, NULL, &write_event[0]);
      checkError(status, "Failed to read from DDR buffer");

      status = clEnqueueWriteBuffer(queue7, d_inData2, CL_FALSE, 0, sizeof(float2) * num_pts, &inp[(i * num_pts)], 0, NULL, &write_event[1]);
      checkError(status, "Failed to write to DDR buffer");

      status=clSetKernelArg(fetch1_kernel, 0, sizeof(cl_mem), (void *)&d_inData1);
      checkError(status, "Failed to set fetch1 kernel arg");
    
      status=clSetKernelArg(store2_kernel, 0, sizeof(cl_mem), (void *)&d_outData1);
      checkError(status, "Failed to set store2 kernel arg");
    }

    // Set Kernel Arguments before execution
    status = clEnqueueTask(queue1, fetch1_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch fetch kernel");

    status = clEnqueueTask(queue2, ffta_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch fft kernel");

    status = clEnqueueTask(queue3, transpose_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch transpose kernel");

    status = clEnqueueTask(queue4, fftb_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch second fft kernel");

    status = clEnqueueTask(queue5, store1_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch second transpose kernel");

    status = clEnqueueTask(queue5, fetch2_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch fetch kernel");

    status = clEnqueueTask(queue4, fftc_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch fft kernel");

    status = clEnqueueTask(queue3, store2_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch transpose kernel");

    status = clFinish(queue7);
    checkError(status, "failed to finish");
    status = clFinish(queue6);
    checkError(status, "failed to finish");
    status = clFinish(queue5);
    checkError(status, "failed to finish");
    status = clFinish(queue4);
    checkError(status, "failed to finish");
    status = clFinish(queue3);
    checkError(status, "failed to finish");
    status = clFinish(queue2);
    checkError(status, "failed to finish");
    status = clFinish(queue1);
    checkError(status, "failed to finish");

    clWaitForEvents(2, write_event);
    clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);
  }

  if( (how_many % 2) == 0){
    status = clEnqueueReadBuffer(queue6, d_outData1, CL_FALSE, 0, sizeof(float2) * num_pts, &out[(how_many - 2) * num_pts], 0, NULL, &write_event[0]);
    checkError(status, "Failed to read from DDR buffer");

    status=clSetKernelArg(fetch1_kernel, 0, sizeof(cl_mem), (void *)&d_inData2);
    checkError(status, "Failed to set fetch1 kernel arg");
  
    status=clSetKernelArg(store2_kernel, 0, sizeof(cl_mem), (void *)&d_outData2);
    checkError(status, "Failed to set store2 kernel arg");
  }
  else{
    status = clEnqueueReadBuffer(queue6, d_outData2, CL_FALSE, 0, sizeof(float2) * num_pts, &out[(how_many - 2) * num_pts], 0, NULL, &write_event[0]);
    checkError(status, "Failed to read from DDR buffer");

    status=clSetKernelArg(fetch1_kernel, 0, sizeof(cl_mem), (void *)&d_inData1);
    checkError(status, "Failed to set fetch1 kernel arg");
  
    status=clSetKernelArg(store2_kernel, 0, sizeof(cl_mem), (void *)&d_outData1);
    checkError(status, "Failed to set store2 kernel arg");
  }

  status = clEnqueueTask(queue1, fetch1_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fetch kernel");

  status = clEnqueueTask(queue2, ffta_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fft kernel");

  status = clEnqueueTask(queue3, transpose_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch transpose kernel");

  status = clEnqueueTask(queue4, fftb_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch second fft kernel");

  status = clEnqueueTask(queue5, store1_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch second transpose kernel");

  status = clEnqueueTask(queue5, fetch2_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fetch kernel");

  status = clEnqueueTask(queue4, fftc_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fft kernel");

  status = clEnqueueTask(queue3, store2_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch transpose kernel");

  status = clFinish(queue6);
  checkError(status, "failed to finish");
  status = clFinish(queue5);
  checkError(status, "failed to finish");
  status = clFinish(queue4);
  checkError(status, "failed to finish");
  status = clFinish(queue3);
  checkError(status, "failed to finish");
  status = clFinish(queue2);
  checkError(status, "failed to finish");
  status = clFinish(queue1);
  checkError(status, "failed to finish");

  clWaitForEvents(1, &write_event[0]);
  clReleaseEvent(write_event[0]);

  if( (how_many % 2) == 0){
    status = clEnqueueReadBuffer(queue6, d_outData2, CL_FALSE, 0, sizeof(float2) * num_pts, &out[(how_many - 1) * num_pts], 0, NULL, &write_event[0]);
    checkError(status, "Failed to read from DDR buffer");
  }
  else{
    status = clEnqueueReadBuffer(queue6, d_outData1, CL_FALSE, 0, sizeof(float2) * num_pts, &out[(how_many - 1) * num_pts], 0, NULL, &write_event[0]);
    checkError(status, "Failed to read from DDR buffer");
  }

  status = clFinish(queue6);
  checkError(status, "failed to finish reading DDR using PCIe");

  clWaitForEvents(1, &write_event[0]);
  clReleaseEvent(write_event[0]);

  fft_time.exec_t = getTimeinMilliSec() - fft_time.exec_t;
  checkError(status, "Failed to copy data from device");
  
  queue_cleanup();

  if (d_inData1)
    clReleaseMemObject(d_inData1);
  if (d_inData2)
    clReleaseMemObject(d_inData2);

  if (d_outData2) 
    clReleaseMemObject(d_outData2);
  if (d_outData2) 
    clReleaseMemObject(d_outData2);

  if (d_transpose) 
    clReleaseMemObject(d_transpose);

  if(fetch1_kernel) 
    clReleaseKernel(fetch1_kernel);  
  if(fetch2_kernel) 
    clReleaseKernel(fetch2_kernel);  

  if(ffta_kernel) 
    clReleaseKernel(ffta_kernel);  
  if(fftb_kernel) 
    clReleaseKernel(fftb_kernel);  
  if(fftc_kernel) 
    clReleaseKernel(fftc_kernel);  

  if(transpose_kernel) 
    clReleaseKernel(transpose_kernel);  

  if(store1_kernel) 
    clReleaseKernel(store1_kernel);  
  if(store2_kernel) 
    clReleaseKernel(store2_kernel);  

  fft_time.valid = 1;
  return fft_time;
}
*/