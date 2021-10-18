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

/**
 * \brief  compute an out-of-place double precision complex 1D-FFT on the FPGA
 * \param  N    : unsigned integer to the number of points in 1D FFT
 * \param  inp  : double2 pointer to input data of size N
 * \param  out  : double2 pointer to output data of size N
 * \param  inv  : int toggle to activate backward FFT
 * \param  batch : number of batched executions of 1D FFT
 * \return fpga_t : time taken in milliseconds for data transfers and execution
 */
fpga_t fftfpga_c2c_1d(const unsigned N, const double2 *inp, double2 *out, const bool inv, const unsigned batch){
  fpga_t fft_time = {0.0, 0.0, 0.0, 0};
  cl_kernel fetch_kernel = NULL, fft_kernel = NULL;
  cl_int status = 0;

  // if N is not a power of 2
  if(inp == NULL || out == NULL || ((N & (N-1)) !=0)){
    return fft_time;
  }

  queue_setup();

  cl_mem d_inData, d_outData;
  d_inData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double2) * N * batch, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");

  d_outData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(double2) * N * batch, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  printf("-- Copying data from host to device\n");
  // Copy data from host to device
  cl_event writeBuf_event;
  status = clEnqueueWriteBuffer(queue1, d_inData, CL_TRUE, 0, sizeof(double2) * N * batch, inp, 0, NULL, &writeBuf_event);
  checkError(status, "Failed to copy data to device");

  status = clFinish(queue1);
  checkError(status, "failed to finish writing buffer using PCIe");

  cl_ulong writeBuf_start = 0.0, writeBuf_end = 0.0;
  clGetEventProfilingInfo(writeBuf_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &writeBuf_start, NULL);
  clGetEventProfilingInfo(writeBuf_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &writeBuf_end, NULL);

  fft_time.pcie_write_t = (cl_double)(writeBuf_end - writeBuf_start) * (cl_double)(1e-06);

  // Can't pass bool to device, so convert it to int
  int inverse_int = (int)inv;

  // Create Kernels - names must match the kernel name in the original CL file
  fetch_kernel = clCreateKernel(program, "fetch", &status);
  checkError(status, "Failed to create fetch kernel");

  fft_kernel = clCreateKernel(program, "fft1d", &status);
  checkError(status, "Failed to create fft1d kernel");
  // Set the kernel arguments
  // from here
  status = clSetKernelArg(fetch_kernel, 0, sizeof(cl_mem), (void *)&d_inData);
  checkError(status, "Failed to set fetch_kernel arg 0");
  status = clSetKernelArg(fft_kernel, 0, sizeof(cl_mem), (void *)&d_outData);
  checkError(status, "Failed to set fft_kernel arg 0");
  status = clSetKernelArg(fft_kernel, 1, sizeof(cl_int), (void*)&batch);
  checkError(status, "Failed to set fft_kernel arg 1");
  status = clSetKernelArg(fft_kernel, 2, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set fft_kernel arg 2");

  printf(inverse_int ? "\tInverse FFT" : "\tFFT");
  printf(" kernel initialization is complete.\n");

  size_t ls = N/8;
  size_t gs = batch * ls;

  printf("-- Executing kernels\n");
  // Measure execution time
  cl_event exec_event;
  // FFT1d kernel is the SWI kernel
  status = clEnqueueTask(queue1, fft_kernel, 0, NULL, &exec_event);
  checkError(status, "Failed to launch fft1d kernel");

  status = clEnqueueNDRangeKernel(queue2, fetch_kernel, 1, NULL, &gs, &ls, 0, NULL, NULL);
  checkError(status, "Failed to launch fetch kernel");
  
  // Wait for command queue to complete pending events
  status = clFinish(queue1);
  checkError(status, "Failed to finish queue1");
  status = clFinish(queue2);
  checkError(status, "Failed to finish queue2");
  
  // Record execution time
  cl_ulong kernel_start = 0, kernel_end = 0;
  clGetEventProfilingInfo(exec_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel_start, NULL);
  clGetEventProfilingInfo(exec_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel_end, NULL);

  fft_time.exec_t = (cl_double)(kernel_end - kernel_start) * (cl_double)(1e-06); 

  // Copy results from device to host
  printf("-- Transfering results back to host\n");
  cl_event readBuf_event;
  status = clEnqueueReadBuffer(queue1, d_outData, CL_TRUE, 0, sizeof(float2) * N * batch, out, 0, NULL, &readBuf_event);
  checkError(status, "Failed to copy data from device");

  status = clFinish(queue1);
  checkError(status, "failed to finish reading buffer using PCIe");

  cl_ulong readBuf_start = 0, readBuf_end = 0;
  clGetEventProfilingInfo(readBuf_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &readBuf_start, NULL);
  clGetEventProfilingInfo(readBuf_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &readBuf_end, NULL);

  fft_time.pcie_read_t = (cl_double)(readBuf_end - readBuf_start) * (cl_double)(1e-06); 

  // Cleanup
  if (d_inData)
  	clReleaseMemObject(d_inData);
  if (d_outData) 
	  clReleaseMemObject(d_outData);
  if(fetch_kernel)
    clReleaseKernel(fetch_kernel);
  if(fft_kernel)
    clReleaseKernel(fft_kernel);
  queue_cleanup();
  fft_time.valid = 1;
  return fft_time;
}

/**
 * \brief  compute an out-of-place single precision complex 1D-FFT on the FPGA
 * \param  N    : unsigned integer to the number of points in FFT1d  
 * \param  inp  : float2 pointer to input data of size N
 * \param  out  : float2 pointer to output data of size N
 * \param  inv  : toggle for backward transforms
 * \param  batch : number of batched executions of 1D FFT
 * \return fpga_t : time taken in milliseconds for data transfers and execution
 */
fpga_t fftfpgaf_c2c_1d(const unsigned N, const float2 *inp, float2 *out, const bool inv, const unsigned batch){

  fpga_t fft_time = {0.0, 0.0, 0.0, 0};
  cl_kernel kernel1 = NULL, kernel2 = NULL;
  cl_int status = 0;

  // if N is not a power of 2
  if(inp == NULL || out == NULL || ( (N & (N-1)) !=0)){
    return fft_time;
  }

  printf("-- Launching%s 1D FFT of %d batches \n", inv ? " inverse":"", batch);

  queue_setup();

  cl_mem d_inData, d_outData;
  printf("Launching%s FFT transform for %d batch \n", inv ? " inverse":"", batch);

  // Create device buffers - assign the buffers in different banks for more efficient memory access 
  d_inData = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float2) * N * batch, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");

  d_outData = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * N * batch, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  printf("-- Copying data from host to device\n");
  // Copy data from host to device
  status = clEnqueueWriteBuffer(queue1, d_inData, CL_TRUE, 0, sizeof(float2) * N * batch, inp, 0, NULL, NULL);
  checkError(status, "Failed to copy data to device");

  status = clFinish(queue1);
  checkError(status, "failed to finish writing buffer using PCIe");

  // Can't pass bool to device, so convert it to int
  int inverse_int = (int)inv;

  // Create Kernels - names must match the kernel name in the original CL file
  kernel1 = clCreateKernel(program, "fetch", &status);
  checkError(status, "Failed to create fetch kernel");

  kernel2 = clCreateKernel(program, "fft1d", &status);
  checkError(status, "Failed to create fft1d kernel");
  // Set the kernel arguments
  status = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *)&d_inData);
  checkError(status, "Failed to set kernel1 arg 0");
  status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void *)&d_outData);
  checkError(status, "Failed to set kernel arg 0");
  status = clSetKernelArg(kernel2, 1, sizeof(cl_int), (void*)&batch);
  checkError(status, "Failed to set kernel arg 1");
  status = clSetKernelArg(kernel2, 2, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set kernel arg 2");

  size_t ls = N/8;
  size_t gs = batch * ls;

  printf("-- Executing kernels\n");
  cl_event startExec_event, endExec_event;
  // Measure execution time
  // Launch the kernel - we launch a single work item hence enqueue a task
  // FFT1d kernel is the SWI kernel
  status = clEnqueueTask(queue1, kernel2, 0, NULL, &endExec_event);
  checkError(status, "Failed to launch fft1d kernel");

  status = clEnqueueNDRangeKernel(queue2, kernel1, 1, NULL, &gs, &ls, 0, NULL, &startExec_event);
  checkError(status, "Failed to launch fetch kernel");
  
  // Wait for command queue to complete pending events
  status = clFinish(queue1);
  checkError(status, "Failed to finish queue1");
  status = clFinish(queue2);
  checkError(status, "Failed to finish queue2");
  
  // Record execution time
  cl_ulong kernel_start = 0, kernel_end = 0;
  clGetEventProfilingInfo(startExec_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel_start, NULL);
  clGetEventProfilingInfo(endExec_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel_end, NULL);
  fft_time.exec_t = (cl_double)(kernel_end - kernel_start) * (cl_double)(1e-06);

  // Copy results from device to host
  printf("-- Transfering results back to host\n");
  status = clEnqueueReadBuffer(queue1, d_outData, CL_TRUE, 0, sizeof(float2) * N * batch, out, 0, NULL, NULL);
  checkError(status, "Failed to copy data from device");

  status = clFinish(queue1);
  checkError(status, "failed to finish reading buffer using PCIe");

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
 * \brief  compute an out-of-place single precision complex 1D-FFT on the FPGA using Shared Virtual Memory for data transfers between host's main memory and FPGA
 * \param  N    : unsigned integer to the number of points in 1D FFT  
 * \param  inp  : float2 pointer to input data of size N
 * \param  out  : float2 pointer to output data of size N
 * \param  inv  : toggle to activate backward FFT
 * \param  batch : number of batched executions of 1D FFT
 * \return fpga_t : time taken in milliseconds for data transfers and execution
 */
fpga_t fftfpgaf_c2c_1d_svm(const unsigned N, const float2 *inp, float2 *out, const bool inv, const unsigned batch){
  fpga_t fft_time = {0.0, 0.0, 0.0, 0};
  cl_int status = 0;
  unsigned num_pts = N * batch;
  
  // if N is not a power of 2
  if(inp == NULL || out == NULL || ( (N & (N-1)) !=0) || !(svm_enabled)){
    return fft_time;
  }

  // Can't pass bool to device, so convert it to int
  int inverse_int = (int)inv;

  // Setup kernels
  cl_kernel fetch_kernel = clCreateKernel(program, "fetch", &status);
  checkError(status, "Failed to create fetch1 kernel");
  cl_kernel fft_kernel = clCreateKernel(program, "fft1d", &status);
  checkError(status, "Failed to create fft3da kernel");

  // Setup Queues to the kernels
  queue_setup();

  // allocate SVM buffers
  float2 *h_inData, *h_outData;
  h_inData = (float2 *)clSVMAlloc(context, CL_MEM_READ_ONLY, sizeof(float2) * num_pts, 0);
  h_outData = (float2 *)clSVMAlloc(context, CL_MEM_WRITE_ONLY, sizeof(float2) * num_pts, 0);

  // copy data into h_inData
  status = clEnqueueSVMMap(queue1, CL_TRUE, CL_MAP_WRITE, (void *)h_inData, sizeof(float2) * num_pts, 0, NULL, NULL);
  checkError(status, "Failed to map input data");

  for(size_t i = 0; i < num_pts; i++){
    h_inData[i].x = inp[i].x;
    h_inData[i].y = inp[i].y;
  }

  status = clEnqueueSVMUnmap(queue1, (void *)h_inData, 0, NULL, NULL);
  checkError(status, "Failed to unmap input data");

  // initialize h_outData with zeroes
  status = clEnqueueSVMMap(queue1, CL_TRUE, CL_MAP_WRITE, (void *)h_outData, sizeof(float2) * num_pts, 0, NULL, NULL);
  checkError(status, "Failed to map input data");

  for(size_t i = 0; i < num_pts; i++){
    h_outData[i].x = 0.0;
    h_outData[i].y = 0.0;
  }

  status = clEnqueueSVMUnmap(queue1, (void *)h_outData, 0, NULL, NULL);
  checkError(status, "Failed to unmap input data");

  // write to fetch kernel using SVM based PCIe
  status = clSetKernelArgSVMPointer(fetch_kernel, 0, (void *)h_inData);
  checkError(status, "Failed to set fetch kernel arg");

  // kernel transforms and stores to DDR memory
  status = clSetKernelArgSVMPointer(fft_kernel, 0, (void *)h_outData);
  checkError(status, "Failed to set store2 kernel arg");

  status=clSetKernelArg(fft_kernel, 1, sizeof(cl_int), (void*)&batch);
  checkError(status, "Failed to set fft kernel arg");

  status=clSetKernelArg(fft_kernel, 2, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set fft kernel arg");

  size_t ls = N/8;
  size_t gs = batch * ls;

  printf("-- Executing\n");
  cl_event startExec_event, endExec_event;

  status = clEnqueueTask(queue1, fft_kernel, 0, NULL, &endExec_event);
  checkError(status, "Failed to launch fetch kernel");

  status = clEnqueueNDRangeKernel(queue2, fetch_kernel, 1, NULL, &gs, &ls, 0, NULL, &startExec_event);
  checkError(status, "Failed to launch fetch kernel");

  status = clFinish(queue2);
  checkError(status, "failed to finish");
  status = clFinish(queue1);
  checkError(status, "failed to finish");

  cl_ulong kernel_start = 0, kernel_end = 0;

  clGetEventProfilingInfo(startExec_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel_start, NULL);
  clGetEventProfilingInfo(endExec_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel_end, NULL);

  fft_time.exec_t = (cl_double)(kernel_end - kernel_start) * (cl_double)(1e-06);

  status = clEnqueueSVMMap(queue1, CL_TRUE, CL_MAP_READ,
    (void *)h_outData, sizeof(float2) * num_pts, 0, NULL, NULL);
  checkError(status, "Failed to map out data");

  for(size_t i = 0; i < num_pts; i++){
    out[i].x = h_outData[i].x;
    out[i].y = h_outData[i].y;
  }

  status = clEnqueueSVMUnmap(queue1, (void *)h_outData, 0, NULL, NULL);
  checkError(status, "Failed to unmap out data");

  if (h_inData)
    clSVMFree(context, h_inData);
  if (h_outData)
    clSVMFree(context, h_outData);

  queue_cleanup();

  if(fetch_kernel) 
    clReleaseKernel(fetch_kernel);  
  if(fft_kernel) 
    clReleaseKernel(fft_kernel);  

  fft_time.valid = 1;
  return fft_time;
}
