/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

// global dependencies
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fftw3.h>

// common dependencies
#include "CL/opencl.h"
#include "../common/opencl_utils.h"
#include "fft_api.h"
#include "helper.h"
#include <CL/cl_ext_intelfpga.h> // to disable interleaving & transfer data to specific banks - CL_CHANNEL_1_INTELFPGA

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
#endif

// Function prototypes
int init();
void cleanup();
static void cleanup_program();
static void init_program(int N[3], char *data_path);
static void queue_setup();
void queue_cleanup();
static double fftfpga_run_3d(int inverse, int N[3], cmplx *c_in);

// --- CODE -------------------------------------------------------------------

int fpga_initialize(const char *platform_name){
  cl_int status = 0;

  // Check if this has to be sent as a pointer or value
  // Get the OpenCL platform.
  platform = findPlatform(platform_name);
  if(platform == NULL) {
    printf("ERROR: Unable to find %s OpenCL platform\n", platform_name);
    return 0;
  }
  // Query the available OpenCL devices.
  cl_uint num_devices;
  devices = getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices);

  return 1;
}

void fpga_final_(){
   cleanup();
}

/******************************************************************************
 * \brief  check whether FFT3d can be computed on the FPGA or not. This depends 
 *         on the availability of bitstreams whose sizes are for now listed here 
 *         If the fft sizes are found and the FPGA is not setup before, it is done
 * \param  data_path - path to the data directory
 * \param  N - integer pointer to the size of the FFT3d
 * \retval true if fft3d size supported
 *****************************************************************************/
int fpga_bitstream(char *bitstream_path){
    int status = init_program(bitstream_path);
    return status;
}

/******************************************************************************
 * \brief   compute an in-place double precision complex 1D-FFT on the FPGA
 * \param   N   : integer pointer to size of FFT3d  
 * \param   inp : double2 pointer to input data of size N
 * \param   inv : int toggle to activate backward FFT
 * \retval fpga_t : time taken in milliseconds for data transfers and execution
 *****************************************************************************/
fpga_t fftfpga_c2c_1d(int N, double2 *inp, int inv, int iter){
  fpga_t fft_time = {0.0, 0.0, 0.0};

  printf("Launching");
  if (inv) 
	printf(" inverse");
  printf(" FFT transform for %d iterations\n", iterations);

#if SVM_API == 1
  status = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE,
      (void *)inp, sizeof(double2) * N * iterations, 0, NULL, NULL);
  checkError(status, "Failed to map input data");
#endif /* SVM_API == 1 */

#if SVM_API == 1
  status = clEnqueueSVMUnmap(queue, (void *)inp, 0, NULL, NULL);
  checkError(status, "Failed to unmap input data");
#else
  // Create device buffers - assign the buffers in different banks for more efficient memory access 
  double pcie_wr_time = 0.0, pcie_rd_time = 0.0;

  d_inData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double2) * N * iterations, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");

  // TODO: check CL_CHANNEL_2_INTELFPGA
  d_outData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(double2) * N * iterations, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  // Copy data from host to device
  pcie_wr_time = getCurrentTimestamp();

  status = clEnqueueWriteBuffer(queue1, d_inData, CL_TRUE, 0, sizeof(double2) * N * iterations, inp, 0, NULL, NULL);

  fft_time.pcie_write_t = getCurrentTimestamp() - pcie_wr_time;
  checkError(status, "Failed to copy data to device");
#endif /* SVM_API == 1 */

  // Can't pass bool to device, so convert it to int
  int inverse_int = inv;

  // Set the kernel arguments

#if SVM_API == 1
  status = clSetKernelArgSVMPointer(kernel1, 0, (void *)inp);
  checkError(status, "Failed to set kernel1 arg 0");

  status = clSetKernelArgSVMPointer(kernel, 0, (void *)out);
#else
  status = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *)&d_inData);
  checkError(status, "Failed to set kernel1 arg 0");
  
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_outData);
#endif /* SVM_API == 1 */
  checkError(status, "Failed to set kernel arg 0");
  status = clSetKernelArg(kernel, 1, sizeof(cl_int), (void*)&iterations);
  checkError(status, "Failed to set kernel arg 1");
  status = clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set kernel arg 2");

  printf(inverse ? "\tInverse FFT" : "\tFFT");
  printf(" kernel initialization is complete.\n");

  // Get the iterationstamp to evaluate performance
  double time = getCurrentTimestamp();

  // Launch the kernel - we launch a single work item hence enqueue a task
  status = clEnqueueTask(queue, kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch kernel");

  size_t ls = N/8;
  size_t gs = iterations * ls;
  status = clEnqueueNDRangeKernel(queue1, kernel1, 1, NULL, &gs, &ls, 0, NULL, NULL);
  checkError(status, "Failed to launch fetch kernel");
  
  // Wait for command queue to complete pending events
  status = clFinish(queue);
  checkError(status, "Failed to finish");
  status = clFinish(queue1);
  checkError(status, "Failed to finish queue1");
  
  // Record execution time
  time = getCurrentTimestamp() - time;

#if SVM_API == 0
  // Copy results from device to host
  pcie_rd_start = getCurrentTimestamp();
  status = clEnqueueReadBuffer(queue, d_outData, CL_TRUE, 0, sizeof(float2) * N * iterations, h_outData, 0, NULL, NULL);
  pcie_rd_time = getCurrentTimestamp() - pcie_rd_start;
  checkError(status, "Failed to copy data from device");

  printf("PCIe Write Transfer Time of %lfms for %d points of %lu bytes\n", pcie_wr_time * 1E3, N*iterations, sizeof(float2) * N * iterations);

  printf("PCIe Read Transfer Time of %lfms for %d points of %lu bytes\n", pcie_rd_time * 1E3, N*iterations, sizeof(float2) * N * iterations);
#else
  status = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ,
      (void *)h_outData, sizeof(float2) * N * iterations, 0, NULL, NULL);
  checkError(status, "Failed to map out data");
#endif /* SVM_API == 0 */

  return fft_time;
}

/******************************************************************************
 * \brief   compute an in-place single precision complex 1D-FFT on the FPGA
 * \param   N   : integer pointer to size of FFT3d  
 * \param   inp : float2 pointer to input data of size N
 * \param   inv : int toggle to activate backward FFT
 * \retval fpga_t : time taken in milliseconds for data transfers and execution
 *****************************************************************************/
fpga_t fftfpgaf_c2c_1d(int N, float2 *inp, int inv, int iter){
  fpga_t fft_time = {0.0, 0.0, 0.0};

  printf("Launching");
  if (inv) 
	printf(" inverse");
  printf(" FFT transform for %d iterations\n", iterations);

#if SVM_API == 1
  status = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE,
      (void *)inp, sizeof(float2) * N * iterations, 0, NULL, NULL);
  checkError(status, "Failed to map input data");
#endif /* SVM_API == 1 */

#if SVM_API == 1
  status = clEnqueueSVMUnmap(queue, (void *)inp, 0, NULL, NULL);
  checkError(status, "Failed to unmap input data");
#else
  // Create device buffers - assign the buffers in different banks for more efficient
  // memory access 
  double pcie_wr_time = 0.0, pcie_rd_time = 0.0;

  d_inData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float2) * N * iterations, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");

  d_outData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * N * iterations, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  // Copy data from host to device
  pcie_wr_time = getCurrentTimestamp();
  status = clEnqueueWriteBuffer(queue1, d_inData, CL_TRUE, 0, sizeof(float2) * N * iterations, h_inData, 0, NULL, NULL);
  fft_time.pcie_write_t = getCurrentTimestamp() - pcie_wr_time;
  checkError(status, "Failed to copy data to device");
#endif /* SVM_API == 1 */

  // Can't pass bool to device, so convert it to int
  int inverse_int = inv;

  // Set the kernel arguments
#if SVM_API == 0
  status = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *)&d_inData);
  checkError(status, "Failed to set kernel1 arg 0");
  
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_outData);
#else
  status = clSetKernelArgSVMPointer(kernel1, 0, (void *)h_inData);
  checkError(status, "Failed to set kernel1 arg 0");

  status = clSetKernelArgSVMPointer(kernel, 0, (void *)h_outData);
#endif /* SVM_API == 0 */
  checkError(status, "Failed to set kernel arg 0");
  status = clSetKernelArg(kernel, 1, sizeof(cl_int), (void*)&iterations);
  checkError(status, "Failed to set kernel arg 1");
  status = clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set kernel arg 2");

  printf(inverse ? "\tInverse FFT" : "\tFFT");
  printf(" kernel initialization is complete.\n");

  // Get the iterationstamp to evaluate performance
  double time = getCurrentTimestamp();

  // Launch the kernel - we launch a single work item hence enqueue a task
  status = clEnqueueTask(queue, kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch kernel");

  size_t ls = N/8;
  size_t gs = iterations * ls;
  status = clEnqueueNDRangeKernel(queue1, kernel1, 1, NULL, &gs, &ls, 0, NULL, NULL);
  checkError(status, "Failed to launch fetch kernel");
  
  // Wait for command queue to complete pending events
  status = clFinish(queue);
  checkError(status, "Failed to finish");
  status = clFinish(queue1);
  checkError(status, "Failed to finish queue1");
  
  // Record execution time
  time = getCurrentTimestamp() - time;

#if SVM_API == 0
  // Copy results from device to host
  pcie_rd_start = getCurrentTimestamp();
  status = clEnqueueReadBuffer(queue, d_outData, CL_TRUE, 0, sizeof(float2) * N * iterations, h_outData, 0, NULL, NULL);
  pcie_rd_time = getCurrentTimestamp() - pcie_rd_start;
  checkError(status, "Failed to copy data from device");

  printf("PCIe Write Transfer Time of %lfms for %d points of %lu bytes\n", pcie_wr_time * 1E3, N*iterations, sizeof(float2) * N * iterations);

  printf("PCIe Read Transfer Time of %lfms for %d points of %lu bytes\n", pcie_rd_time * 1E3, N*iterations, sizeof(float2) * N * iterations);
#else
  status = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ,
      (void *)h_outData, sizeof(float2) * N * iterations, 0, NULL, NULL);
  checkError(status, "Failed to map out data");
#endif /* SVM_API == 0 */

  return fft_time;
}

/******************************************************************************
 * \brief   compute an in-place single precision complex 2D-FFT on the FPGA
 * \param   N   : integer pointer to size of FFT3d  
 * \param   inp : double2 pointer to input data of size N^2
 * \param   inv : int toggle to activate backward FFT
 * \retval fpga_t : time taken in milliseconds for data transfers and execution
 *****************************************************************************/
fpga_t fftfpga_c2c_2d(int N, double2 *inp, int inv){

}

/******************************************************************************
 * \brief   compute an in-place single precision complex 2D-FFT on the FPGA
 * \param   N   : integer pointer to size of FFT3d  
 * \param   inp : float2 pointer to input data of size N^2
 * \param   inv : int toggle to activate backward FFT
 * \retval fpga_t : time taken in milliseconds for data transfers and execution
 *****************************************************************************/
fpga_t fftfpgaf_c2c_2d(int N, float2 *inp, int inv){

}

/******************************************************************************
 * \brief   compute an in-place double precision complex 3D-FFT on the FPGA
 * \param   N   : integer pointer to size of FFT3d  
 * \param   inp : double2 pointer to input data of size N^3
 * \param   inv : int toggle to activate backward FFT
 * \retval fpga_t : time taken in milliseconds for data transfers and execution
 *****************************************************************************/
fpga_t fftfpga_c2c_3d(int N, double2 *inp, int inv){

}

/******************************************************************************
 * \brief   compute an in-place single precision complex 3D-FFT on the FPGA
 * \param   N   : integer pointer to size of FFT3d  
 * \param   inp : float2 pointer to input data of size N^3
 * \param   inv : int toggle to activate backward FFT
 * \retval fpga_t : time taken in milliseconds for data transfers and execution
 *****************************************************************************/
fpga_t fftfpgaf_c2c_3d(int N, float2 *inp, int inv){

}

/******************************************************************************
 * \brief   compute an in-place single precision complex 3D-FFT on the FPGA
 * \param   direction : direction - 1/forward, otherwise/backward FFT3d
 * \param   N   : integer pointer to size of FFT3d  
 * \param   din : complex input/output single precision data pointer 
 * \retval double : time taken for FFT3d compute 
 *****************************************************************************/
double fpga_fft3d_sp_(int direction, int N[3], cmplx *din) {
  // setup device specific constructs 
  if(direction == 1){
    return fftfpga_run_3d(0, N, din);
  }
  else{
    return fftfpga_run_3d(1, N, din);
  }
}

/******************************************************************************
 * \brief   compute an in-place double precision complex 3D-FFT on the FPGA
 * \param   direction : direction - 1/forward, otherwise/backward FFT3d
 * \param   N   : integer pointer to size of FFT3d  
 * \param   din : complex input/output single precision data pointer 
 * \retval double : time taken for FFT3d compute 
 *****************************************************************************/
double fpga_fft3d_dp_(int direction, int N[3], cmplx *din) {
  // setup device specific constructs 
  if(direction == 1){
    return fftfpga_run_3d(0, N, din);
  }
  else{
    return fftfpga_run_3d(1, N, din);
  }
}

/******************************************************************************
 * \brief   Execute a single precision complex FFT3d
 * \param   inverse : int
 * \param   N       : integer pointer to size of FFT3d  
 * \param   din     : complex input/output single precision data pointer 
 * \retval double : time taken for FFT3d compute 
 *****************************************************************************/
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


/******************************************************************************
 * \brief   Initialize the program - select device, create context and program
 *****************************************************************************/
void init_program(int N[3], char *data_path){
  cl_int status = 0;

  // use the first device.
  device = devices[0];

  // Create the context.
  context = clCreateContext(NULL, 1, &device, &openCLContextCallBackFxn, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the program.
  program = getProgramWithBinary(context, &device, 1, N, data_path);
  if(program == NULL) {
    printf("Failed to create program");
    exit(1);
  }
  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

}

/******************************************************************************
 * \brief   Free resources allocated during program initialization
 *****************************************************************************/
void cleanup_program(){
  if(program) 
    clReleaseProgram(program);
  if(context)
    clReleaseContext(context);
}

/******************************************************************************
 * \brief   Initialize the OpenCL FPGA environment - platform and devices
 * \retval  true if error in initialization
 *****************************************************************************/
int init() {
  cl_int status = 0;

  // Get the OpenCL platform.
  platform = findPlatform("Intel(R) FPGA");
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform\n");
    return 1;
  }
  // Query the available OpenCL devices.
  cl_uint num_devices;
  devices = getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices);

  return 0;
}

/******************************************************************************
 * \brief   Free resources allocated during initialization - devices
 *****************************************************************************/
void cleanup(){
  cleanup_program();
  free(devices);
}

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
