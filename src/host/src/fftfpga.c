// Author: Arjun Ramaswami

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define CL_VERSION_2_0
#include <CL/cl_ext_intelfpga.h> // to disable interleaving & transfer data to specific banks - CL_CHANNEL_1_INTELFPGA
#include "CL/opencl.h"
#include "aocl_mmd.h"

#include "fftfpga.h"
#include "opencl_utils.h"
#include "helper.h"

// host variables
#ifndef KERNEL_VARS
#define KERNEL_VARS
static cl_platform_id platform = NULL;
static cl_device_id *devices;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_program program = NULL;
static cl_command_queue queue1 = NULL, queue2 = NULL, queue3 = NULL;
static cl_command_queue queue4 = NULL, queue5 = NULL, queue6 = NULL;

static int svm_handle;
static int svm_enabled = 0;
#endif

// Function prototypes
static void queue_setup();
void queue_cleanup();
//static double fftfpga_run_3d(int inverse, int N[3], cmplx *c_in);

/*
int replace(){
  const char* board_name;
  aocl_mmd_offline_info_t info_id;
  aocl_mmd_get_offline_info(info_id, );

  return aocl_mmd_open(board_name);
}
*/

/** 
 * @brief Check if device support svm 
 * @param device
 * @return 1 if supported and 0 if not
 */
static int check_valid_svm_device(cl_device_id device){
  cl_device_svm_capabilities caps = 0;
  cl_int status;

  status = clGetDeviceInfo(
    device,
    CL_DEVICE_SVM_CAPABILITIES,
    sizeof(cl_device_svm_capabilities),
    &caps,
    0
  );
  checkError(status, "Failed to get device info");

  if(caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER){
    fprintf(stderr, "Found CL_DEVICE_SVM_FINE_GRAIN_BUFFER. API support in progress\n");
    return 0;
  }
  else if(caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM){
    fprintf(stderr, "Found CL_DEVICE_SVM_FINE_GRAIN_SYSTEM. API support in progress\n");
    return 0;
  }
  else if(caps & CL_DEVICE_SVM_ATOMICS){
    fprintf(stderr, "Found CL_DEVICE_SVM_ATOMICS. API support in progress\n");
    return 0;
  }
  else if (caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER){
    return 1;
  }
  else{
    fprintf(stderr, "No SVM Support found!");
    return 0;
  }
  return 0;
}

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

  if(path == NULL){
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

  // return if SVM enabled but no device supported
  if(use_svm){
    // TODO: emulation and svm
    if (check_valid_svm_device(device)){
      svm_enabled = 1;
    }
    else{
      fpga_final();
      return 1;
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
  if(inp == NULL || out == NULL || (N & (N-1) !=0)){
    return fft_time;
  }

#ifdef VERBOSE
  printf("Launching%s FFT transform for %d iter \n", inv ? " inverse":"", iter);
#endif
  double2 *h_inData, *h_outData;
  queue_setup();

  if(svm_enabled){
    // Transfer Data to Global Memory or allocate SVM buffer
    /*
    size_t buf_size = sizeof(double2) * N * iter;
    (double2 *)aocl_mmd_shared_mem_alloc(, buf_size, )
    h_inData = (double2 *)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(double2) * N * iter, 0);

    h_outData = (float2 *)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(double2) * N * iter, 0);

    status = clEnqueueSVMMap(queue1, CL_TRUE, CL_MAP_WRITE,
        (void *)h_inData, sizeof(double2) * N * iter, 0, NULL, NULL);
    checkError(status, "Failed to map input data");

    // Copy data from input file to SVM allocated memory.
    for (int i = 0; i < N * iter; i++) {
      h_inData[i].x = inp[i].x;
      h_inData[i].y = inp[i].y;
    }

    status = clEnqueueSVMUnmap(queue1, (void *)h_inData, 0, NULL, NULL);
    checkError(status, "Failed to unmap input data");

    // Can't pass bool to device, so convert it to int
    int inverse_int = inv;

    // Create Kernels - names must match the kernel name in the original CL file
    kernel1 = clCreateKernel(program, "fetch", &status);
    checkError(status, "Failed to create fetch kernel");

    kernel2 = clCreateKernel(program, "fft1d", &status);
    checkError(status, "Failed to create fft1d kernel");
    // Set the kernel arguments

    status = clSetKernelArgSVMPointer(kernel1, 0, (void *)h_inData);
    checkError(status, "Failed to set kernel1 arg 0");

    status = clSetKernelArgSVMPointer(kernel2, 0, (void *)h_outData);
    checkError(status, "Failed to set kernel1 arg 0");
    status = clSetKernelArgSVMPointer(kernel1, 0, (void *)h_inData);
    checkError(status, "Failed to set kernel1 arg 0");

    status = clSetKernelArgSVMPointer(kernel2, 0, (void *)h_outData);

    checkError(status, "Failed to set kernel arg 0");
    status = clSetKernelArg(kernel2, 1, sizeof(cl_int), (void*)&iter);
    checkError(status, "Failed to set kernel arg 1");
    status = clSetKernelArg(kernel2, 2, sizeof(cl_int), (void*)&inverse_int);
    checkError(status, "Failed to set kernel arg 2");

    printf(inverse_int ? "\tInverse FFT" : "\tFFT");
    printf(" kernel initialization is complete.\n");

    // Get the itertamp to evaluate performance
    fft_time.exec_t = getTimeinMilliSec();

    // Launch the kernel - we launch a single work item hence enqueue a task
    status = clEnqueueTask(queue1, kernel1, 0, NULL, NULL);
    checkError(status, "Failed to launch kernel");

    size_t ls = N/8;
    size_t gs = iter * ls;
    status = clEnqueueNDRangeKernel(queue1, kernel2, 1, NULL, &gs, &ls, 0, NULL, NULL);
    checkError(status, "Failed to launch fetch kernel");
    
    // Wait for command queue to complete pending events
    status = clFinish(queue1);
    checkError(status, "Failed to finish");
    status = clFinish(queue2);
    checkError(status, "Failed to finish queue1");
    
    // Record execution time
    fft_time.exec_t = getTimeinMilliSec() - fft_time.exec_t;
    status = clEnqueueSVMMap(queue1, CL_TRUE, CL_MAP_READ,
        (void *)h_outData, sizeof(float2) * N * iter, 0, NULL, NULL);
    checkError(status, "Failed to map out data");

      // Copy data from input file to SVM allocated memory.
    for (int i = 0; i < N * iter; i++) {
      out[i].x = h_outData[i].x;
      out[i].y = h_outData[i].y;
    }

    status = clEnqueueSVMUnmap(queue1, (void *)h_outData, 0, NULL, NULL);
    checkError(status, "Failed to unmap input data");

    // Cleanup
    if(kernel1)
      clReleaseKernel(kernel1);
    if(kernel2)
      clReleaseKernel(kernel2);
    queue_cleanup();
    if (h_inData)
      clSVMFree(context, h_inData);
    if (h_outData)
      clSVMFree(context, h_outData);
    }
    */
  }
  else{
    cl_mem d_inData, d_outData;
    printf("Launching%s FFT transform for %d iter \n", inv ? " inverse":"", iter);
    queue_setup();

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

    //printf("PCIe Write Transfer Time of %lfms for %d points of %lu bytes\n", fft_time.pcie_write_t * 1E3, N*iter, sizeof(float2) * N * iter);

    //printf("PCIe Read Transfer Time of %lfms for %d points of %lu bytes\n", fft_time.pcie_read_t * 1E3, N*iter, sizeof(float2) * N * iter);
  }

  // Cleanup
  if(kernel1)
    clReleaseKernel(kernel1);
  if(kernel2)
    clReleaseKernel(kernel2);
  queue_cleanup();

  if (h_inData)
    clSVMFree(context, h_inData);
  if (h_outData)
    clSVMFree(context, h_outData);

  return fft_time;
}

/**
 * \brief  compute an out-of-place float2 precision complex 1D-FFT on the FPGA
 * \param  N    : integer pointer to size of FFT3d  
 * \param  inp  : float2 pointer to input data of size N
 * \param  out  : float2 pointer to output data of size N
 * \param  inv  : int toggle to activate backward FFT
 * \param  iter : int toggle to activate backward FFT
 * \return fpga_t : time taken in milliseconds for data transfers and execution
 */
fpga_t fftfpgaf_c2c_1d(int N, float2 *inp, float2 *out, int inv, int iter){
  fpga_t fft_time = {0.0, 0.0, 0.0};
  cl_kernel kernel1 = NULL, kernel2 = NULL;
  cl_int status = 0;

#if SVM_API == 1
  float2 *h_inData, *h_outData;
#else
  cl_mem d_inData, d_outData;
#endif

  printf("Launching%s FFT transform for %d iter \n", inv ? " inverse":"", iter);
  queue_setup();

  // Transfer Data to Global Memory or allocate SVM buffer
#if SVM_API == 1
  h_inData = (float2 *)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(float2) * N * iter, 0);

  h_outData = (float2 *)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(float2) * N * iter, 0);

  status = clEnqueueSVMMap(queue1, CL_TRUE, CL_MAP_WRITE,
      (void *)h_inData, sizeof(float2) * N * iter, 0, NULL, NULL);
  checkError(status, "Failed to map input data");

  // Copy data from input file to SVM allocated memory.
  for (int i = 0; i < N * iter; i++) {
    h_inData[i].x = inp[i].x;
    h_inData[i].y = inp[i].y;
  }

  status = clEnqueueSVMUnmap(queue1, (void *)h_inData, 0, NULL, NULL);
  checkError(status, "Failed to unmap input data");
#else
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
#endif /* SVM_API == 1 */

  // Can't pass bool to device, so convert it to int
  int inverse_int = inv;

  // Create Kernels - names must match the kernel name in the original CL file
  kernel1 = clCreateKernel(program, "fetch", &status);
  checkError(status, "Failed to create fetch kernel");

  kernel2 = clCreateKernel(program, "fft1d", &status);
  checkError(status, "Failed to create fft1d kernel");
  // Set the kernel arguments

#if SVM_API == 1
  status = clSetKernelArgSVMPointer(kernel1, 0, (void *)h_inData);
  checkError(status, "Failed to set kernel1 arg 0");

  status = clSetKernelArgSVMPointer(kernel2, 0, (void *)h_outData);
#else
  status = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *)&d_inData);
  checkError(status, "Failed to set kernel1 arg 0");
  
  status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void *)&d_outData);
#endif /* SVM_API == 1 */
  checkError(status, "Failed to set kernel arg 0");
  status = clSetKernelArg(kernel2, 1, sizeof(cl_int), (void*)&iter);
  checkError(status, "Failed to set kernel arg 1");
  status = clSetKernelArg(kernel2, 2, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set kernel arg 2");

  printf(inverse_int ? "\tInverse FFT" : "\tFFT");
  printf(" kernel initialization is complete.\n");

  // Get the itertamp to evaluate performance
  fft_time.exec_t = getTimeinMilliSec();

  // Launch the FFT kernel - SingleWorkItem kernel
  status = clEnqueueTask(queue1, kernel2, 0, NULL, NULL);
  checkError(status, "Failed to launch kernel");

  size_t ls = N/8;
  size_t gs = iter * ls;
  status = clEnqueueNDRangeKernel(queue2, kernel1, 1, NULL, &gs, &ls, 0, NULL, NULL);
  checkError(status, "Failed to launch fetch kernel");
  
  // Wait for command queue to complete pending events
  status = clFinish(queue1);
  checkError(status, "Failed to finish FFT kernel");
  status = clFinish(queue2);
  checkError(status, "Failed to finish Fetch kernel");
  
  // Record execution time
  fft_time.exec_t = getTimeinMilliSec() - fft_time.exec_t;

#if SVM_API == 0
  // Copy results from device to host
  fft_time.pcie_read_t = getTimeinMilliSec();
  status = clEnqueueReadBuffer(queue1, d_outData, CL_TRUE, 0, sizeof(float2) * N * iter, out, 0, NULL, NULL);
  fft_time.pcie_read_t = getTimeinMilliSec() - fft_time.pcie_read_t;
  checkError(status, "Failed to copy data from device");

  //printf("PCIe Write Transfer Time of %lfms for %d points of %lu bytes\n", fft_time.pcie_write_t * 1E3, N*iter, sizeof(float2) * N * iter);

  //printf("PCIe Read Transfer Time of %lfms for %d points of %lu bytes\n", fft_time.pcie_read_t * 1E3, N*iter, sizeof(float2) * N * iter);
#else
  status = clEnqueueSVMMap(queue1, CL_TRUE, CL_MAP_READ,
      (void *)h_outData, sizeof(float2) * N * iter, 0, NULL, NULL);
  checkError(status, "Failed to map out data");

    // Copy data from input file to SVM allocated memory.
  for (int i = 0; i < N * iter; i++) {
    out[i].x = h_outData[i].x;
    out[i].y = h_outData[i].y;
  }

  status = clEnqueueSVMUnmap(queue1, (void *)h_outData, 0, NULL, NULL);
  checkError(status, "Failed to unmap input data");

#endif /* SVM_API == 0 */

  // Cleanup
  if(kernel1)
    clReleaseKernel(kernel1);
  if(kernel2)
    clReleaseKernel(kernel2);
  queue_cleanup();

#if USE_SVM_API == 0
  if (d_inData)
	  clReleaseMemObject(d_inData);
  if (d_outData) 
	  clReleaseMemObject(d_outData);
#else
  if (h_inData)
    clSVMFree(context, h_inData);
  if (h_outData)
    clSVMFree(context, h_outData);
#endif /* USE_SVM_API == 0 */

  return fft_time; 
}

/******************************************************************************
 * \brief   compute an in-place single precision complex 2D-FFT on the FPGA
 * \param   N   : integer pointer to size of FFT3d  
 * \param   inp : double2 pointer to input data of size N^2
 * \param   inv : int toggle to activate backward FFT
 * \retval fpga_t : time taken in milliseconds for data transfers and execution
 *****************************************************************************/
/*
fpga_t fftfpga_c2c_2d(int N, double2 *inp, int inv){

}
*/

/******************************************************************************
 * \brief   compute an in-place single precision complex 2D-FFT on the FPGA
 * \param   N   : integer pointer to size of FFT3d  
 * \param   inp : float2 pointer to input data of size N^2
 * \param   inv : int toggle to activate backward FFT
 * \retval fpga_t : time taken in milliseconds for data transfers and execution
 *****************************************************************************/
/*
fpga_t fftfpgaf_c2c_2d(int N, float2 *inp, int inv){

}
*/

/******************************************************************************
 * \brief   compute an in-place double precision complex 3D-FFT on the FPGA
 * \param   N   : integer pointer to size of FFT3d  
 * \param   inp : double2 pointer to input data of size N^3
 * \param   inv : int toggle to activate backward FFT
 * \retval fpga_t : time taken in milliseconds for data transfers and execution
 *****************************************************************************/
/*
fpga_t fftfpga_c2c_3d(int N, double2 *inp, int inv){

}
*/

/******************************************************************************
 * \brief   compute an in-place single precision complex 3D-FFT on the FPGA
 * \param   N   : integer pointer to size of FFT3d  
 * \param   inp : float2 pointer to input data of size N^3
 * \param   inv : int toggle to activate backward FFT
 * \retval fpga_t : time taken in milliseconds for data transfers and execution
 *****************************************************************************/
/*
fpga_t fftfpgaf_c2c_3d(int N, float2 *inp, int inv){

}
*/

/******************************************************************************
 * \brief   Execute a single precision complex FFT3d
 * \param   inverse : int
 * \param   N       : integer pointer to size of FFT3d  
 * \param   din     : complex input/output single precision data pointer 
 * \retval double : time taken for FFT3d compute 
 *****************************************************************************/
/*
static double fftfpga_run_3d(int inverse, int N[3], cmplx *c_in) {
  cl_int status = 0;
  int inverse_int = inverse;
  cl_kernel fft_kernel = NULL, fft_kernel_2 = NULL;
  cl_kernel fetch_kernel = NULL, transpose_kernel = NULL, transpose_kernel_2 = NULL;
  
 // Device memory buffers
  cl_mem d_inData, d_outData;

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

  d_inData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cmplx) * N[0] * N[1] * N[2], NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");
  d_outData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cmplx) * N[0] * N[1] * N[2], NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  cmplx *h_inData = (cmplx *)alignedMalloc(sizeof(cmplx) * N[0] * N[1] * N[2]);
  if (h_inData == NULL){
    printf("Unable to allocate host memory\n");
    exit(1);
  }
  cmplx *h_outData = (cmplx *)alignedMalloc(sizeof(cmplx) * N[0] * N[1] * N[2]);
  if (h_outData == NULL){
    printf("Unable to allocate host memory\n");
    exit(1);
  }

  memcpy(h_inData, c_in, sizeof(cmplx) * N[0] * N[1] * N[2]);

  queue_setup();

  // Copy data from host to device
  double pcie_wr_start = getTimeinMilliSec();
  status = clEnqueueWriteBuffer(queue6, d_inData, CL_TRUE, 0, sizeof(cmplx) * N[0] * N[1] * N[2], h_inData, 0, NULL, NULL);
  checkError(status, "Failed to copy data to device");
  double pcie_wr_time = getTimeinMilliSec() - pcie_wr_start;
  status = clFinish(queue6);
  checkError(status, "failed to finish");
  printf("PCIE_Write Time of %d points totalling % bytes %lf\n", pcie_wr_time, N[0] * N[1] * N[2], sizeof(cmplx) * N[0] * N[1] * N[2]);

  status = clSetKernelArg(fetch_kernel, 0, sizeof(cl_mem), (void *)&d_inData);
  checkError(status, "Failed to set kernel arg 0");
  status = clSetKernelArg(fft_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set kernel arg 1");
  status = clSetKernelArg(transpose_kernel, 0, sizeof(cl_mem), (void *)&d_outData);
  checkError(status, "Failed to set kernel arg 2");
  status = clSetKernelArg(fft_kernel_2, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set kernel arg 3");

  double start = getTimeinMilliSec();
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

  double stop = getTimeinMilliSec();
  double fpga_runtime = stop - start;
   
  // Copy results from device to host
  double pcie_rd_start = getTimeinMilliSec();
  status = clEnqueueReadBuffer(queue3, d_outData, CL_TRUE, 0, sizeof(cmplx) * N[0] * N[1] * N[2], h_outData, 0, NULL, NULL);
  checkError(status, "Failed to read data from device");
  double pcie_rd_time = getTimeinMilliSec() - pcie_rd_start;

  printf("PCIE_Read Time of %d points totalling % bytes %lf\n", pcie_rd_time, N[0] * N[1] * N[2], sizeof(cmplx) * N[0] * N[1] * N[2]);

  memcpy(c_in, h_outData, sizeof(cmplx) * N[0] * N[1] * N[2] );

  queue_cleanup();

  if (h_outData)
	  free(h_outData);
  if (h_inData)
	  free(h_inData);

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

  return fpga_runtime;
}
*/
/******************************************************************************
 * \brief   Create a command queue for each kernel
 *****************************************************************************/
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

/******************************************************************************
 * \brief   Release all command queues
 *****************************************************************************/
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

