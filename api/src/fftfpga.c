// Author: Arjun Ramaswami

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define CL_VERSION_2_0
#include <CL/cl_ext_intelfpga.h> // to disable interleaving & transfer data to specific banks - CL_CHANNEL_1_INTELFPGA
#include "CL/opencl.h"

#include "fftfpga/fftfpga.h"
#include "opencl_utils.h"
#include "misc.h"

#ifndef KERNEL_VARS
#define KERNEL_VARS
static cl_platform_id platform = NULL;
static cl_device_id *devices;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_program program = NULL;
static cl_command_queue queue1 = NULL, queue2 = NULL, queue3 = NULL;
static cl_command_queue queue4 = NULL, queue5 = NULL, queue6 = NULL;

//static int svm_handle;
//static int svm_enabled = 0;
#endif

static void queue_setup();
void queue_cleanup();

/** 
 * @brief Allocate memory of double precision complex floating points
 * @param sz  : size_t - size to allocate
 * @param svm : 1 if svm
 * @return void ptr or NULL
 */
void* fftfpga_complex_malloc(size_t sz, int svm){
  if(svm == 1){
    fprintf(stderr, "Working in progress\n");
    return NULL;
    // return aocl_mmd_shared_mem_alloc(svm_handle, sizeof(double2) * sz, inData, device_ptr);
  }
  else if(sz == 0){
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
void* fftfpgaf_complex_malloc(size_t sz, int svm){
  if(svm == 1){
    fprintf(stderr, "Working in progress\n");
    return NULL;
    // return aocl_mmd_shared_mem_alloc(svm_handle, sizeof(double2) * sz, inData, device_ptr);
  }
  else if(sz == 0){
    return NULL;
  }
  else{
    return ((float2 *)alignedMalloc(sz));
  }
}

/** 
 * @brief Initialize FPGA
 * @param platform name: string - name of the OpenCL platform
 * @param path         : string - path to binary
 * @param use_svm      : 1 if true 0 otherwise
 * @param use_emulator : 1 if true 0 otherwise
 * @return 0 if successful 
 */
int fpga_initialize(const char *platform_name, const char *path, int use_svm, int use_emulator){
  cl_int status = 0;

#ifdef VERBOSE
  printf("\tInitializing FPGA ...\n");
#endif

  if(path == NULL || strlen(path) == 0){
    fprintf(stderr, "Path to binary missing\n");
    return 1;
  }

  // Check if this has to be sent as a pointer or value
  // Get the OpenCL platform.
  platform = findPlatform(platform_name);
  if(platform == NULL){
    fprintf(stderr,"ERROR: Unable to find %s OpenCL platform\n", platform_name);
    return 1;
  }
  // Query the available OpenCL devices.
  cl_uint num_devices;
  devices = getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices);
  if(devices == NULL){
    fprintf(stderr, "ERROR: Unable to find devices for %s OpenCL platform\n", platform_name);
    return 1;
  }

  // use the first device.
  device = devices[0];

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
    return 1;
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
 * \brief  compute an out-of-place double precision complex 1D-FFT on the FPGA
 * \param  N    : integer pointer to size of FFT3d  
 * \param  inp  : double2 pointer to input data of size N
 * \param  out  : double2 pointer to output data of size N
 * \param  inv  : int toggle to activate backward FFT
 * \param  iter : int toggle to activate backward FFT
 * \return fpga_t : time taken in milliseconds for data transfers and execution
 */
fpga_t fftfpga_c2c_1d(int N, double2 *inp, double2 *out, int inv, int iter){
  fpga_t fft_time = {0.0, 0.0, 0.0, 0};
  cl_kernel kernel1 = NULL, kernel2 = NULL;
  cl_int status = 0;

  // if N is not a power of 2
  if(inp == NULL || out == NULL || ((N & (N-1)) !=0)){
    return fft_time;
  }

#ifdef VERBOSE
  printf("Launching%s FFT transform for %d iter \n", inv ? " inverse":"", iter);
#endif

  queue_setup();

  cl_mem d_inData, d_outData;
  printf("Launching%s FFT transform for %d iter \n", inv ? " inverse":"", iter);

  // Create device buffers - assign the buffers in different banks for more efficient memory access 
  d_inData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double2) * N * iter, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");

  // TODO: check CL_CHANNEL_2_INTELFPGA
  d_outData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(double2) * N * iter, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  // Copy data from host to device
  fft_time.pcie_write_t = getTimeinMilliSec();

  status = clEnqueueWriteBuffer(queue1, d_inData, CL_TRUE, 0, sizeof(double2) * N * iter, inp, 0, NULL, NULL);

  fft_time.pcie_write_t = getTimeinMilliSec() - fft_time.pcie_write_t;
  checkError(status, "Failed to copy data to device");

  // Can't pass bool to device, so convert it to int
  int inverse_int = inv;

  // Create Kernels - names must match the kernel name in the original CL file
  kernel1 = clCreateKernel(program, "fetch", &status);
  checkError(status, "Failed to create fetch kernel");

  kernel2 = clCreateKernel(program, "fft1d", &status);
  checkError(status, "Failed to create fft1d kernel");
  // Set the kernel arguments
  // from here
  status = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *)&d_inData);
  checkError(status, "Failed to set kernel1 arg 0");
  status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void *)&d_outData);
  checkError(status, "Failed to set kernel arg 0");
  status = clSetKernelArg(kernel2, 1, sizeof(cl_int), (void*)&iter);
  checkError(status, "Failed to set kernel arg 1");
  status = clSetKernelArg(kernel2, 2, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set kernel arg 2");

  printf(inverse_int ? "\tInverse FFT" : "\tFFT");
  printf(" kernel initialization is complete.\n");

  size_t ls = N/8;
  size_t gs = iter * ls;

  // Measure execution time
  fft_time.exec_t = getTimeinMilliSec();

  // Launch the kernel - we launch a single work item hence enqueue a task
  // FFT1d kernel is the SWI kernel
  status = clEnqueueTask(queue1, kernel2, 0, NULL, NULL);
  checkError(status, "Failed to launch fft1d kernel");

  status = clEnqueueNDRangeKernel(queue2, kernel1, 1, NULL, &gs, &ls, 0, NULL, NULL);
  checkError(status, "Failed to launch fetch kernel");
  
  // Wait for command queue to complete pending events
  status = clFinish(queue1);
  checkError(status, "Failed to finish queue1");
  status = clFinish(queue2);
  checkError(status, "Failed to finish queue2");
  
  // Record execution time
  fft_time.exec_t = getTimeinMilliSec() - fft_time.exec_t;

  // Copy results from device to host
  fft_time.pcie_read_t = getTimeinMilliSec();
  status = clEnqueueReadBuffer(queue1, d_outData, CL_TRUE, 0, sizeof(float2) * N * iter, out, 0, NULL, NULL);
  fft_time.pcie_read_t = getTimeinMilliSec() - fft_time.pcie_read_t;
  checkError(status, "Failed to copy data from device");

  // Cleanup
  if (d_inData)
  	clReleaseMemObject(d_inData);
  if (d_outData) 
	  clReleaseMemObject(d_outData);
  if(kernel1)
    clReleaseKernel(kernel1);
  if(kernel2)
    clReleaseKernel(kernel2);
  queue_cleanup();
  fft_time.valid = 1;
  return fft_time;
}

/**
 * \brief  compute an out-of-place single precision complex 1D-FFT on the FPGA
 * \param  N    : integer pointer to size of FFT3d  
 * \param  inp  : float2 pointer to input data of size N
 * \param  out  : float2 pointer to output data of size N
 * \param  inv  : int toggle to activate backward FFT
 * \param  iter : int toggle to activate backward FFT
 * \return fpga_t : time taken in milliseconds for data transfers and execution
 */
fpga_t fftfpgaf_c2c_1d(int N, float2 *inp, float2 *out, int inv, int iter){

  fpga_t fft_time = {0.0, 0.0, 0.0, 0};
  cl_kernel kernel1 = NULL, kernel2 = NULL;
  cl_int status = 0;

  // if N is not a power of 2
  if(inp == NULL || out == NULL || ( (N & (N-1)) !=0)){
    return fft_time;
  }

#ifdef VERBOSE
  printf("Launching%s FFT transform for %d iter \n", inv ? " inverse":"", iter);
#endif

  queue_setup();

  cl_mem d_inData, d_outData;
  printf("Launching%s FFT transform for %d iter \n", inv ? " inverse":"", iter);

  // Create device buffers - assign the buffers in different banks for more efficient memory access 
  d_inData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float2) * N * iter, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");

  // TODO: check CL_CHANNEL_2_INTELFPGA
  d_outData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * N * iter, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  // Copy data from host to device
  fft_time.pcie_write_t = getTimeinMilliSec();

  status = clEnqueueWriteBuffer(queue1, d_inData, CL_TRUE, 0, sizeof(float2) * N * iter, inp, 0, NULL, NULL);

  fft_time.pcie_write_t = getTimeinMilliSec() - fft_time.pcie_write_t;
  checkError(status, "Failed to copy data to device");

  // Can't pass bool to device, so convert it to int
  int inverse_int = inv;

  // Create Kernels - names must match the kernel name in the original CL file
  kernel1 = clCreateKernel(program, "fetch", &status);
  checkError(status, "Failed to create fetch kernel");

  kernel2 = clCreateKernel(program, "fft1d", &status);
  checkError(status, "Failed to create fft1d kernel");
  // Set the kernel arguments
  // from here
  status = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *)&d_inData);
  checkError(status, "Failed to set kernel1 arg 0");
  status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void *)&d_outData);
  checkError(status, "Failed to set kernel arg 0");
  status = clSetKernelArg(kernel2, 1, sizeof(cl_int), (void*)&iter);
  checkError(status, "Failed to set kernel arg 1");
  status = clSetKernelArg(kernel2, 2, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set kernel arg 2");

  printf(inverse_int ? "\tInverse FFT" : "\tFFT");
  printf(" kernel initialization is complete.\n");

  size_t ls = N/8;
  size_t gs = iter * ls;

  // Measure execution time
  fft_time.exec_t = getTimeinMilliSec();

  // Launch the kernel - we launch a single work item hence enqueue a task
  // FFT1d kernel is the SWI kernel
  status = clEnqueueTask(queue1, kernel2, 0, NULL, NULL);
  checkError(status, "Failed to launch fft1d kernel");

  status = clEnqueueNDRangeKernel(queue2, kernel1, 1, NULL, &gs, &ls, 0, NULL, NULL);
  checkError(status, "Failed to launch fetch kernel");
  
  // Wait for command queue to complete pending events
  status = clFinish(queue1);
  checkError(status, "Failed to finish queue1");
  status = clFinish(queue2);
  checkError(status, "Failed to finish queue2");
  
  // Record execution time
  fft_time.exec_t = getTimeinMilliSec() - fft_time.exec_t;

  // Copy results from device to host
  fft_time.pcie_read_t = getTimeinMilliSec();
  status = clEnqueueReadBuffer(queue1, d_outData, CL_TRUE, 0, sizeof(float2) * N * iter, out, 0, NULL, NULL);
  fft_time.pcie_read_t = getTimeinMilliSec() - fft_time.pcie_read_t;
  checkError(status, "Failed to copy data from device");

  // Cleanup
  if (d_inData)
  	clReleaseMemObject(d_inData);
  if (d_outData) 
	  clReleaseMemObject(d_outData);
  if(kernel1)
    clReleaseKernel(kernel1);
  if(kernel2)
    clReleaseKernel(kernel2);
  queue_cleanup();

  fft_time.valid = 1;
  return fft_time;
}

/**
 * \brief  compute an out-of-place single precision complex 2D-FFT using the DDR of the FPGA
 * \param  N    : integer pointer to size of FFT2d  
 * \param  inp  : float2 pointer to input data of size N
 * \param  out  : float2 pointer to output data of size N
 * \param  inv  : int toggle to activate backward FFT
 * \param  iter : int toggle to activate backward FFT
 * \return fpga_t : time taken in milliseconds for data transfers and execution
 */
fpga_t fftfpgaf_c2c_2d_ddr(int N, float2 *inp, float2 *out, int inv){
  fpga_t fft_time = {0.0, 0.0, 0.0, 0};
  cl_kernel fetch_kernel = NULL, fft_kernel = NULL, transpose_kernel = NULL;
  cl_int status = 0;
  int mangle_int = 0;

  // if N is not a power of 2
  if(inp == NULL || out == NULL || ( (N & (N-1)) !=0)){
    return fft_time;
  }

#ifdef VERBOSE
  printf("Launching%s 2d FFT transform \n", inv ? " inverse":"");
#endif

  queue_setup();

  cl_mem d_inData, d_outData, d_tmp;
  //printf("Launching%s FFT transform for %d iter \n", inv ? " inverse":"", iter);

  d_inData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float2) * N * N, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");
  d_outData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float2) * N * N, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");
  d_tmp = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * N * N, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  // Copy data from host to device
  fft_time.pcie_write_t = getTimeinMilliSec();

  status = clEnqueueWriteBuffer(queue1, d_inData, CL_TRUE, 0, sizeof(float2) * N * N, inp, 0, NULL, NULL);

  fft_time.pcie_write_t = getTimeinMilliSec() - fft_time.pcie_write_t;
  checkError(status, "Failed to copy data to device");

  // Can't pass bool to device, so convert it to int
  int inverse_int = inv;

  // Create Kernels - names must match the kernel name in the original CL file
  fft_kernel = clCreateKernel(program, "fft2d", &status);
  checkError(status, "Failed to create kernel");
  fetch_kernel = clCreateKernel(program, "fetch", &status);
  checkError(status, "Failed to create kernel");
  transpose_kernel = clCreateKernel(program, "transpose", &status);
  checkError(status, "Failed to create kernel");

  // Record execution time
  fft_time.exec_t = getTimeinMilliSec();

  // Loop twice over the kernels
  for (size_t i = 0; i < 2; i++) {

    // Set the kernel arguments
    status = clSetKernelArg(fetch_kernel, 0, sizeof(cl_mem), i == 0 ? (void *)&d_inData : (void *)&d_tmp);
    checkError(status, "Failed to set kernel arg 0");
    status = clSetKernelArg(fetch_kernel, 1, sizeof(cl_int), (void*)&mangle_int);
    checkError(status, "Failed to set kernel arg 1");
    size_t lws_fetch[] = {N};
    size_t gws_fetch[] = {N * N / 8};
    status = clEnqueueNDRangeKernel(queue1, fetch_kernel, 1, 0, gws_fetch, lws_fetch, 0, NULL, NULL);
    checkError(status, "Failed to launch kernel");

    // Launch the fft kernel - we launch a single work item hence enqueue a task
    status = clSetKernelArg(fft_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
    checkError(status, "Failed to set kernel arg 0");
    status = clEnqueueTask(queue2, fft_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch kernel");

    // Set the kernel arguments
    status = clSetKernelArg(transpose_kernel, 0, sizeof(cl_mem), i == 0 ? (void *)&d_tmp : (void *)&d_outData);
    checkError(status, "Failed to set kernel arg 0");

    status = clSetKernelArg(transpose_kernel, 1, sizeof(cl_int), (void*)&mangle_int);
    checkError(status, "Failed to set kernel arg 1");

    size_t lws_transpose[] = {N};
    size_t gws_transpose[] = {N * N / 8};
    status = clEnqueueNDRangeKernel(queue3, transpose_kernel, 1, 0, gws_transpose, lws_transpose, 0, NULL, NULL);
    checkError(status, "Failed to launch kernel");

    // Wait for all command queues to complete pending events
    status = clFinish(queue1);
    checkError(status, "failed to finish");
    status = clFinish(queue2);
    checkError(status, "failed to finish");
    status = clFinish(queue3);
    checkError(status, "failed to finish");
  }

  fft_time.exec_t = getTimeinMilliSec() - fft_time.exec_t;

  // Copy results from device to host
  fft_time.pcie_read_t = getTimeinMilliSec();
  status = clEnqueueReadBuffer(queue1, d_outData, CL_TRUE, 0, sizeof(float2) * N * N, out, 0, NULL, NULL);
  fft_time.pcie_read_t = getTimeinMilliSec() - fft_time.pcie_read_t;
  checkError(status, "Failed to copy data from device");

  // Cleanup
  if (d_inData)
  	clReleaseMemObject(d_inData);
  if (d_outData) 
	  clReleaseMemObject(d_outData);
  if (d_tmp)
	  clReleaseMemObject(d_tmp);
  if(fft_kernel)
    clReleaseKernel(fft_kernel);
  if(fetch_kernel)
    clReleaseKernel(fetch_kernel);
  if(transpose_kernel)
    clReleaseKernel(transpose_kernel);
  queue_cleanup();

  fft_time.valid = 1;
  return fft_time;
}

/**
 * \brief  compute an out-of-place single precision complex 2D-FFT using the BRAM of the FPGA
 * \param  N    : integer pointer to size of FFT2d  
 * \param  inp  : float2 pointer to input data of size N
 * \param  out  : float2 pointer to output data of size N
 * \param  inv  : int toggle to activate backward FFT
 * \return fpga_t : time taken in milliseconds for data transfers and execution
 */
fpga_t fftfpgaf_c2c_2d_bram(int N, float2 *inp, float2 *out, int inv){
  fpga_t fft_time = {0.0, 0.0, 0.0, 0};
  cl_kernel ffta_kernel = NULL, fftb_kernel = NULL;
  cl_kernel fetch_kernel = NULL, transpose_kernel = NULL, store_kernel = NULL;

  cl_int status = 0;
  int num_pts = N * N;

  // if N is not a power of 2
  if(inp == NULL || out == NULL || ( (N & (N-1)) !=0)){
    return fft_time;
  }

#ifdef VERBOSE
  printf("Launching%s 3d FFT transform in DDR \n", inv ? " inverse":"");
#endif

  queue_setup();

  // Device memory buffers
  cl_mem d_inData, d_outData;
  d_inData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");

  d_outData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

 // Copy data from host to device
  fft_time.pcie_write_t = getTimeinMilliSec();

  status = clEnqueueWriteBuffer(queue1, d_inData, CL_TRUE, 0, sizeof(float2) * num_pts, inp, 0, NULL, NULL);

  status = clFinish(queue1);
  checkError(status, "failed to finish");

  fft_time.pcie_write_t = getTimeinMilliSec() - fft_time.pcie_write_t;
  checkError(status, "Failed to copy data to device");

  // Can't pass bool to device, so convert it to int
  int inverse_int = inv;

  ffta_kernel = clCreateKernel(program, "fft2da", &status);
  checkError(status, "Failed to create fft3da kernel");
  fftb_kernel = clCreateKernel(program, "fft2db", &status);
  checkError(status, "Failed to create fft3db kernel");

  fetch_kernel = clCreateKernel(program, "fetch", &status);
  checkError(status, "Failed to create fetch kernel");

  transpose_kernel = clCreateKernel(program, "transpose", &status);
  checkError(status, "Failed to create transpose kernel");

  store_kernel = clCreateKernel(program, "store", &status);
  checkError(status, "Failed to create store kernel");

  status = clSetKernelArg(fetch_kernel, 0, sizeof(cl_mem), (void *)&d_inData);
  checkError(status, "Failed to set fetch kernel arg");
  status = clSetKernelArg(ffta_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set ffta kernel arg");
  status = clSetKernelArg(fftb_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set fftb kernel arg");
  status = clSetKernelArg(store_kernel, 0, sizeof(cl_mem), (void *)&d_outData);
  checkError(status, "Failed to set store kernel arg");

  fft_time.exec_t = getTimeinMilliSec();
  status = clEnqueueTask(queue1, fetch_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fetch kernel");

  status = clEnqueueTask(queue2, ffta_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fft kernel");

  status = clEnqueueTask(queue3, transpose_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch transpose kernel");

  status = clEnqueueTask(queue4, fftb_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch second fft kernel");

  status = clEnqueueTask(queue5, store_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch store kernel");

  // Wait for all command queues to complete pending events
  status = clFinish(queue1);
  checkError(status, "failed to finish");
  status = clFinish(queue2);
  checkError(status, "failed to finish");
  status = clFinish(queue3);
  checkError(status, "failed to finish");
  status = clFinish(queue4);
  checkError(status, "failed to finish");
  status = clFinish(queue5);
  checkError(status, "failed to finish");
  fft_time.exec_t = getTimeinMilliSec() - fft_time.exec_t;

  // Copy results from device to host
  fft_time.pcie_read_t = getTimeinMilliSec();
  status = clEnqueueReadBuffer(queue1, d_outData, CL_TRUE, 0, sizeof(float2) * num_pts, out, 0, NULL, NULL);
  fft_time.pcie_read_t = getTimeinMilliSec() - fft_time.pcie_read_t;
  checkError(status, "Failed to copy data from device");

  queue_cleanup();

  if (d_inData)
  	clReleaseMemObject(d_inData);
  if (d_outData) 
	  clReleaseMemObject(d_outData);

  if(fetch_kernel) 
    clReleaseKernel(fetch_kernel);  

  if(ffta_kernel) 
    clReleaseKernel(ffta_kernel);  
  if(fftb_kernel) 
    clReleaseKernel(fftb_kernel);  

  if(transpose_kernel) 
    clReleaseKernel(transpose_kernel);  

  if(store_kernel) 
    clReleaseKernel(store_kernel);  

  fft_time.valid = 1;
  return fft_time;
}


fpga_t fftfpgaf_c2c_3d_bram(int N, float2 *inp, float2 *out, int inv) {
  fpga_t fft_time = {0.0, 0.0, 0.0, 0};
  cl_kernel fft_kernel = NULL, fft_kernel_2 = NULL;
  cl_kernel fetch_kernel = NULL, transpose_kernel = NULL, transpose_kernel_2 = NULL;
  cl_int status = 0;

  // if N is not a power of 2
  if(inp == NULL || out == NULL || ( (N & (N-1)) !=0)){
    return fft_time;
  }

#ifdef VERBOSE
  printf("Launching%s 3d FFT transform \n", inv ? " inverse":"");
#endif

  queue_setup();

  // Device memory buffers
  cl_mem d_inData, d_outData;
  d_inData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_1_INTELFPGA, sizeof(float2) * N * N * N, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");
  d_outData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * N * N * N, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

 // Copy data from host to device
  fft_time.pcie_write_t = getTimeinMilliSec();

  status = clEnqueueWriteBuffer(queue1, d_inData, CL_TRUE, 0, sizeof(float2) * N * N * N, inp, 0, NULL, NULL);

  status = clFinish(queue1);
  checkError(status, "failed to finish");

  fft_time.pcie_write_t = getTimeinMilliSec() - fft_time.pcie_write_t;
  checkError(status, "Failed to copy data to device");

  // Can't pass bool to device, so convert it to int
  int inverse_int = inv;

  // Create the kernel - name passed in here must match kernel name in the
  // original CL file, that was compiled into an AOCX file using the AOC tool
  fft_kernel = clCreateKernel(program, "fft3da", &status);
  checkError(status, "Failed to create fft3da kernel");
  fft_kernel_2 = clCreateKernel(program, "fft3db", &status);
  checkError(status, "Failed to create fft3db kernel");
  fetch_kernel = clCreateKernel(program, "fetch", &status);
  checkError(status, "Failed to create fetch kernel");
  transpose_kernel = clCreateKernel(program, "transpose", &status);
  checkError(status, "Failed to create transpose kernel");
  transpose_kernel_2 = clCreateKernel(program, "transpose3d", &status);
  checkError(status, "Failed to create transpose3d kernel");

  status = clSetKernelArg(fetch_kernel, 0, sizeof(cl_mem), (void *)&d_inData);
  checkError(status, "Failed to set kernel arg 0");
  status = clSetKernelArg(fft_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set kernel arg 1");
  status = clSetKernelArg(transpose_kernel, 0, sizeof(cl_mem), (void *)&d_outData);
  checkError(status, "Failed to set kernel arg 2");
  status = clSetKernelArg(fft_kernel_2, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set kernel arg 3");

  fft_time.exec_t = getTimeinMilliSec();
  status = clEnqueueTask(queue1, fetch_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fetch kernel");

  // Launch the fft kernel - we launch a single work item hence enqueue a task
  status = clEnqueueTask(queue2, fft_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fft kernel");

  status = clEnqueueTask(queue3, transpose_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch transpose kernel");

  status = clEnqueueTask(queue4, fft_kernel_2, 0, NULL, NULL);
  checkError(status, "Failed to launch second fft kernel");

  status = clEnqueueTask(queue5, transpose_kernel_2, 0, NULL, NULL);
  checkError(status, "Failed to launch second transpose kernel");

  // Wait for all command queues to complete pending events
  status = clFinish(queue1);
  checkError(status, "failed to finish");
  status = clFinish(queue2);
  checkError(status, "failed to finish");
  status = clFinish(queue3);
  checkError(status, "failed to finish");
  status = clFinish(queue4);
  checkError(status, "failed to finish");
  status = clFinish(queue5);
  checkError(status, "failed to finish");

  fft_time.exec_t = getTimeinMilliSec() - fft_time.exec_t;

  // Copy results from device to host
  fft_time.pcie_read_t = getTimeinMilliSec();
  status = clEnqueueReadBuffer(queue1, d_outData, CL_TRUE, 0, sizeof(float2) * N * N * N, out, 0, NULL, NULL);
  fft_time.pcie_read_t = getTimeinMilliSec() - fft_time.pcie_read_t;
  checkError(status, "Failed to copy data from device");

  queue_cleanup();

  if (d_inData)
  	clReleaseMemObject(d_inData);
  if (d_outData) 
	  clReleaseMemObject(d_outData);

  if(fetch_kernel) 
    clReleaseKernel(fetch_kernel);  
  if(fft_kernel) 
    clReleaseKernel(fft_kernel);  
  if(fft_kernel_2) 
    clReleaseKernel(fft_kernel_2);  
  if(transpose_kernel) 
    clReleaseKernel(transpose_kernel);  
  if(transpose_kernel_2) 
    clReleaseKernel(transpose_kernel_2); 

  fft_time.valid = 1;
  return fft_time;
}

fpga_t fftfpgaf_c2c_3d_ddr(int N, float2 *inp, float2 *out, int inv) {
  fpga_t fft_time = {0.0, 0.0, 0.0, 0};
  cl_kernel ffta_kernel = NULL, fftb_kernel = NULL, fftc_kernel = NULL;
  cl_kernel fetch1_kernel = NULL, fetch2_kernel = NULL; 
  cl_kernel transpose_kernel = NULL;
  cl_kernel store1_kernel = NULL, store2_kernel = NULL; 

  cl_int status = 0;
  int num_pts = N * N * N;

  // if N is not a power of 2
  if(inp == NULL || out == NULL || ( (N & (N-1)) !=0)){
    return fft_time;
  }

#ifdef VERBOSE
  printf("Launching%s 3d FFT transform in DDR \n", inv ? " inverse":"");
#endif

  queue_setup();

  // Device memory buffers
  cl_mem d_inData, d_outData;
  d_inData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_1_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");
  d_outData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

 // Copy data from host to device
  fft_time.pcie_write_t = getTimeinMilliSec();

  status = clEnqueueWriteBuffer(queue1, d_inData, CL_TRUE, 0, sizeof(float2) * num_pts, inp, 0, NULL, NULL);

  status = clFinish(queue1);
  checkError(status, "failed to finish");

  fft_time.pcie_write_t = getTimeinMilliSec() - fft_time.pcie_write_t;
  checkError(status, "Failed to copy data to device");

  // Can't pass bool to device, so convert it to int
  int inverse_int = inv;

  fetch1_kernel = clCreateKernel(program, "fetch1", &status);
  checkError(status, "Failed to create fetch1 kernel");
  ffta_kernel = clCreateKernel(program, "fft3da", &status);
  checkError(status, "Failed to create fft3da kernel");
  transpose_kernel = clCreateKernel(program, "transpose", &status);
  checkError(status, "Failed to create transpose kernel");
  fftb_kernel = clCreateKernel(program, "fft3db", &status);
  checkError(status, "Failed to create fft3db kernel");
  store1_kernel = clCreateKernel(program, "store1", &status);
  checkError(status, "Failed to create store1 kernel");

  fetch2_kernel = clCreateKernel(program, "fetch2", &status);
  checkError(status, "Failed to create fetch2 kernel");
  fftc_kernel = clCreateKernel(program, "fft3dc", &status);
  checkError(status, "Failed to create fft3dc kernel");
  store2_kernel = clCreateKernel(program, "store2", &status);
  checkError(status, "Failed to create store2 kernel");

  status = clSetKernelArg(fetch1_kernel, 0, sizeof(cl_mem), (void *)&d_inData);
  checkError(status, "Failed to set fetch1 kernel arg");
  status = clSetKernelArg(ffta_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set ffta kernel arg");
  status = clSetKernelArg(fftb_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set fftb kernel arg");
  status = clSetKernelArg(store1_kernel, 0, sizeof(cl_mem), (void *)&d_outData);
  checkError(status, "Failed to set store1 kernel arg");

  status = clSetKernelArg(fetch2_kernel, 0, sizeof(cl_mem), (void *)&d_outData);
  checkError(status, "Failed to set fetch2 kernel arg");
  status = clSetKernelArg(fftc_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set fftc kernel arg");
  status = clSetKernelArg(store2_kernel, 0, sizeof(cl_mem), (void *)&d_inData);
  checkError(status, "Failed to set store2 kernel arg");

  fft_time.exec_t = getTimeinMilliSec();
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

  // Wait for all command queues to complete pending events
  status = clFinish(queue1);
  checkError(status, "failed to finish");
  status = clFinish(queue2);
  checkError(status, "failed to finish");
  status = clFinish(queue3);
  checkError(status, "failed to finish");
  status = clFinish(queue4);
  checkError(status, "failed to finish");
  status = clFinish(queue5);
  checkError(status, "failed to finish");

  status = clEnqueueTask(queue1, fetch2_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fetch kernel");

  status = clEnqueueTask(queue2, fftc_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fft kernel");

  status = clEnqueueTask(queue3, store2_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch transpose kernel");

  status = clFinish(queue1);
  checkError(status, "failed to finish");
  status = clFinish(queue2);
  checkError(status, "failed to finish");
  status = clFinish(queue3);
  checkError(status, "failed to finish");
  fft_time.exec_t = getTimeinMilliSec() - fft_time.exec_t;

  // Copy results from device to host
  fft_time.pcie_read_t = getTimeinMilliSec();
  status = clEnqueueReadBuffer(queue1, d_inData, CL_TRUE, 0, sizeof(float2) * num_pts, out, 0, NULL, NULL);
  fft_time.pcie_read_t = getTimeinMilliSec() - fft_time.pcie_read_t;
  checkError(status, "Failed to copy data from device");

  queue_cleanup();

  if (d_inData)
  	clReleaseMemObject(d_inData);
  if (d_outData) 
	  clReleaseMemObject(d_outData);

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
}
