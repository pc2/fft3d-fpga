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
 * \brief  compute an out-of-place single precision complex 2D-FFT using the DDR of the FPGA
 * \param  N    : integer pointer to size of FFT2d  
 * \param  inp  : float2 pointer to input data of size [N * N]
 * \param  out  : float2 pointer to output data of size [N * N]
 * \param  inv  : int toggle to activate backward FFT
 * \param  iter : int toggle to activate backward FFT
 * \return fpga_t : time taken in milliseconds for data transfers and execution
 */
fpga_t fftfpgaf_c2c_2d_ddr(const unsigned N, const float2 *inp, float2 *out, const bool inv){
  fpga_t fft_time = {0.0, 0.0, 0.0, 0.0, 0};
  cl_kernel fetch_kernel = NULL, fft_kernel = NULL, transpose_kernel = NULL;
  cl_int status = 0;
  int mangle_int = 0;

  // if N is not a power of 2
  if(inp == NULL || out == NULL || ( (N & (N-1)) !=0)){
    return fft_time;
  }

  queue_setup();

  cl_mem d_inData, d_outData, d_tmp;

  d_inData = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float2) * N * N, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");
  d_outData = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float2) * N * N, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");
  d_tmp = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * N * N, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  // Copy data from host to device
  cl_event writeBuf_event;
  status = clEnqueueWriteBuffer(queue1, d_inData, CL_TRUE, 0, sizeof(float2) * N * N, inp, 0, NULL, &writeBuf_event);
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
  fft_kernel = clCreateKernel(program, "fft2d", &status);
  checkError(status, "Failed to create kernel");
  fetch_kernel = clCreateKernel(program, "fetch", &status);
  checkError(status, "Failed to create kernel");
  transpose_kernel = clCreateKernel(program, "transpose", &status);
  checkError(status, "Failed to create kernel");

  cl_event startExec_event[2], endExec_event[2];
  // Loop twice over the kernels
  for (size_t i = 0; i < 2; i++) {

    // Set the kernel arguments
    status = clSetKernelArg(fetch_kernel, 0, sizeof(cl_mem), i == 0 ? (void *)&d_inData : (void *)&d_tmp);
    checkError(status, "Failed to set kernel arg 0");
    status = clSetKernelArg(fetch_kernel, 1, sizeof(cl_int), (void*)&mangle_int);
    checkError(status, "Failed to set kernel arg 1");
    size_t lws_fetch[] = {N};
    size_t gws_fetch[] = {N * N / 8};
    status = clEnqueueNDRangeKernel(queue1, fetch_kernel, 1, 0, gws_fetch, lws_fetch, 0, NULL, (i % 2) == 0 ? &startExec_event[0] : &startExec_event[1]);
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
    status = clEnqueueNDRangeKernel(queue3, transpose_kernel, 1, 0, gws_transpose, lws_transpose, 0, NULL, (i % 2) == 0 ? &endExec_event[0] : &endExec_event[1]);
    checkError(status, "Failed to launch kernel");

    // Wait for all command queues to complete pending events
    status = clFinish(queue1);
    checkError(status, "failed to finish");
    status = clFinish(queue2);
    checkError(status, "failed to finish");
    status = clFinish(queue3);
    checkError(status, "failed to finish");
  }

  cl_ulong kernel_start = 0, kernel_end = 0;

  clGetEventProfilingInfo(startExec_event[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel_start, NULL);
  clGetEventProfilingInfo(endExec_event[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel_end, NULL);

  fft_time.exec_t = (cl_double)(kernel_end - kernel_start) * (cl_double)(1e-06); 

  clGetEventProfilingInfo(startExec_event[1], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel_start, NULL);
  clGetEventProfilingInfo(endExec_event[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel_end, NULL);

  fft_time.exec_t += (cl_double)(kernel_end - kernel_start) * (cl_double)(1e-06); 

  // Copy results from device to host
  cl_event readBuf_event;
  status = clEnqueueReadBuffer(queue1, d_outData, CL_TRUE, 0, sizeof(float2) * N * N, out, 0, NULL, &readBuf_event);
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
 * \param  inp  : float2 pointer to input data of size [N * N]
 * \param  out  : float2 pointer to output data of size [N * N]
 * \param  inv  : int toggle to activate backward FFT
 * \param  interleaving : enable interleaved global memory buffers
 * \return fpga_t : time taken in milliseconds for data transfers and execution
 */
fpga_t fftfpgaf_c2c_2d_bram(const unsigned N, const float2 *inp, float2 *out, const bool inv, const bool interleaving, const unsigned how_many){
  fpga_t fft_time = {0.0, 0.0, 0.0, 0};
  cl_kernel ffta_kernel = NULL, fftb_kernel = NULL;
  cl_kernel fetch_kernel = NULL, store_kernel = NULL;
  cl_kernel transpose_kernel = NULL;

  cl_int status = 0;
  unsigned num_pts = how_many * N * N;

  // if N is not a power of 2
  if(inp == NULL || out == NULL || ( (N & (N-1)) !=0)){
    return fft_time;
  }

  queue_setup();

  cl_mem_flags flagbuf1, flagbuf2;
  if(interleaving == 1){
    flagbuf1 = CL_MEM_READ_WRITE;
    flagbuf2 = CL_MEM_READ_WRITE;
  }
  else{
    flagbuf1 = CL_MEM_READ_ONLY | CL_CHANNEL_1_INTELFPGA;
    flagbuf2 = CL_MEM_WRITE_ONLY | CL_CHANNEL_2_INTELFPGA;
  }
  
  // Device memory buffers
  cl_mem d_inData, d_outData;
  d_inData = clCreateBuffer(context, flagbuf1, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");

  d_outData = clCreateBuffer(context, flagbuf2, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

 // Copy data from host to device
  cl_event writeBuf_event;
  status = clEnqueueWriteBuffer(queue1, d_inData, CL_TRUE, 0, sizeof(float2) * num_pts, inp, 0, NULL, &writeBuf_event);
  checkError(status, "Failed to copy data to device");

  status = clFinish(queue1);
  checkError(status, "failed to finish");

  cl_ulong writeBuf_start = 0.0, writeBuf_end = 0.0;

  clGetEventProfilingInfo(writeBuf_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &writeBuf_start, NULL);
  clGetEventProfilingInfo(writeBuf_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &writeBuf_end, NULL);

  fft_time.pcie_write_t = (cl_double)(writeBuf_end - writeBuf_start) * (cl_double)(1e-06);

  // Can't pass bool to device, so convert it to int
  int inverse_int = (int)inv;

  ffta_kernel = clCreateKernel(program, "fft2da", &status);
  checkError(status, "Failed to create fft2da kernel");

  fftb_kernel = clCreateKernel(program, "fft2db", &status);
  checkError(status, "Failed to create fft2db kernel");

  fetch_kernel = clCreateKernel(program, "fetchBitrev", &status);
  checkError(status, "Failed to create fetch kernel");

  transpose_kernel = clCreateKernel(program, "transpose", &status);
  checkError(status, "Failed to create transpose1 kernel");

  store_kernel = clCreateKernel(program, "transposeStore", &status);
  checkError(status, "Failed to create store kernel");

  status = clSetKernelArg(fetch_kernel, 0, sizeof(cl_mem), (void *)&d_inData);
  checkError(status, "Failed to set fetch kernel arg 0");

  status = clSetKernelArg(fetch_kernel, 1, sizeof(cl_int), (void *)&how_many);
  checkError(status, "Failed to set fetch kernel arg 1");

  status = clSetKernelArg(ffta_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set ffta kernel arg 0");

  status = clSetKernelArg(ffta_kernel, 1, sizeof(cl_int), (void*)&how_many);
  checkError(status, "Failed to set ffta kernel arg 1");

  status = clSetKernelArg(transpose_kernel, 0, sizeof(cl_int), (void*)&how_many);
  checkError(status, "Failed to set transpose kernel arg 0");

  status = clSetKernelArg(fftb_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set fftb kernel arg 0");

  status = clSetKernelArg(fftb_kernel, 1, sizeof(cl_int), (void*)&how_many);
  checkError(status, "Failed to set fftb kernel arg 1");

  status = clSetKernelArg(store_kernel, 0, sizeof(cl_mem), (void *)&d_outData);
  checkError(status, "Failed to set store kernel arg");

  status = clSetKernelArg(store_kernel, 1, sizeof(cl_int), (void *)&how_many);
  checkError(status, "Failed to set store kernel arg");

  // Kernel Execution
  cl_event startExec_event, endExec_event;
  status = clEnqueueTask(queue1, fetch_kernel, 0, NULL, &startExec_event);
  checkError(status, "Failed to launch fetch kernel");

  status = clEnqueueTask(queue2, ffta_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fft kernel");

  status = clEnqueueTask(queue3, transpose_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch transpose1 kernel");

  status = clEnqueueTask(queue4, fftb_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch second fft kernel");

  status = clEnqueueTask(queue5, store_kernel, 0, NULL, &endExec_event);
  checkError(status, "Failed to launch store kernel");

  // Wait for all command queues to complete pending events
  status = clFinish(queue1);
  checkError(status, "failed to finish queue1");
  status = clFinish(queue2);
  checkError(status, "failed to finish queue2");
  status = clFinish(queue3);
  checkError(status, "failed to finish queue3");
  status = clFinish(queue4);
  checkError(status, "failed to finish queue4");
  status = clFinish(queue5);
  checkError(status, "failed to finish queue5");

  cl_ulong kernel_start = 0, kernel_end = 0;

  clGetEventProfilingInfo(startExec_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel_start, NULL);
  clGetEventProfilingInfo(endExec_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel_end, NULL);

  fft_time.exec_t = (cl_double)(kernel_end - kernel_start) * (cl_double)(1e-06); 

  // Copy results from device to host
  cl_event readBuf_event;

  status = clEnqueueReadBuffer(queue1, d_outData, CL_TRUE, 0, sizeof(float2) * num_pts, out, 0, NULL, &readBuf_event);
  checkError(status, "Failed to copy data from device");

  status = clFinish(queue1);
  checkError(status, "failed to finish reading buffer using PCIe");

  cl_ulong readBuf_start = 0, readBuf_end = 0;
  clGetEventProfilingInfo(readBuf_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &readBuf_start, NULL);
  clGetEventProfilingInfo(readBuf_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &readBuf_end, NULL);

  fft_time.pcie_read_t = (cl_double)(readBuf_end - readBuf_start) * (cl_double)(1e-06);

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

/**
 * \brief  compute an out-of-place single precision complex 2DFFT using the BRAM of the FPGA and Shared Virtual Memory for Host to Device Communication
 * \param  N    : integer pointer to size of FFT2d  
 * \param  inp  : float2 pointer to input data of size [N * N]
 * \param  out  : float2 pointer to output data of size [N * N]
 * \param  inv  : int toggle to activate backward FFT
 * \return fpga_t : time taken in milliseconds for data transfers and execution
 */
fpga_t fftfpgaf_c2c_2d_bram_svm(const unsigned N, const float2 *inp, float2 *out, const bool inv, const unsigned how_many){
  fpga_t fft_time = {0.0, 0.0, 0.0, 0};
  cl_int status = 0;
  unsigned num_pts = how_many * N * N;

  cl_kernel ffta_kernel = NULL, fftb_kernel = NULL;
  cl_kernel fetch_kernel = NULL, store_kernel = NULL;
  cl_kernel transpose_kernel = NULL;

  // if N is not a power of 2
  if(inp == NULL || out == NULL || ( (N & (N-1)) !=0) || (!svm_enabled))
    return fft_time;

  queue_setup();

  // allocate SVM buffers
  float2 *h_inData, *h_outData;
  h_inData = (float2 *)clSVMAlloc(context, CL_MEM_READ_ONLY, sizeof(float2) * num_pts, 0);
  h_outData = (float2 *)clSVMAlloc(context, CL_MEM_WRITE_ONLY, sizeof(float2) * num_pts, 0);

  status = clEnqueueSVMMap(queue1, CL_TRUE, CL_MAP_WRITE, (void *)h_inData, sizeof(float2) * num_pts, 0, NULL, NULL);
  checkError(status, "Failed to map input data");

  // copy data into h_inData
  for(size_t i = 0; i < num_pts; i++){
    h_inData[i].x = inp[i].x;
    h_inData[i].y = inp[i].y;
  }

  status = clEnqueueSVMUnmap(queue1, (void *)h_inData, 0, NULL, NULL);
  checkError(status, "Failed to unmap input data");

  status = clEnqueueSVMMap(queue1, CL_TRUE, CL_MAP_WRITE, (void *)h_outData, sizeof(float2) * num_pts, 0, NULL, NULL);
  checkError(status, "Failed to map input data");

  // copy data into h_inData
  for(size_t i = 0; i < num_pts; i++){
    h_outData[i].x = 0.0;
    h_outData[i].y = 0.0;
  }

  status = clEnqueueSVMUnmap(queue1, (void *)h_outData, 0, NULL, NULL);
  checkError(status, "Failed to unmap input data");

  // Can't pass bool to device, so convert it to int
  int inverse_int = (int)inv;

  ffta_kernel = clCreateKernel(program, "fft2da", &status);
  checkError(status, "Failed to create fft2da kernel");

  fftb_kernel = clCreateKernel(program, "fft2db", &status);
  checkError(status, "Failed to create fft2db kernel");

  fetch_kernel = clCreateKernel(program, "fetchBitrev", &status);
  checkError(status, "Failed to create fetch kernel");

  transpose_kernel = clCreateKernel(program, "transpose", &status);
  checkError(status, "Failed to create transpose1 kernel");

  store_kernel = clCreateKernel(program, "transposeStore", &status);
  checkError(status, "Failed to create store kernel");

  // write to fetch kernel using SVM based PCIe
  status = clSetKernelArgSVMPointer(fetch_kernel, 0, (void *)h_inData);
  checkError(status, "Failed to set fetch1 kernel arg");

  status = clSetKernelArg(fetch_kernel, 1, sizeof(cl_int), (void *)&how_many);
  checkError(status, "Failed to set fetch kernel arg 1");

  status = clSetKernelArg(ffta_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set ffta kernel arg 0");

  status = clSetKernelArg(ffta_kernel, 1, sizeof(cl_int), (void*)&how_many);
  checkError(status, "Failed to set ffta kernel arg 1");

  status = clSetKernelArg(transpose_kernel, 0, sizeof(cl_int), (void*)&how_many);
  checkError(status, "Failed to set transpose kernel arg 0");

  status = clSetKernelArg(fftb_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set fftb kernel arg 0");

  status = clSetKernelArg(fftb_kernel, 1, sizeof(cl_int), (void*)&how_many);
  checkError(status, "Failed to set fftb kernel arg 1");

  // kernel stores using SVM based PCIe to host
  status = clSetKernelArgSVMPointer(store_kernel, 0, (void*)h_outData);
  checkError(status, "Failed to set store2 kernel arg");

  status = clSetKernelArg(store_kernel, 1, sizeof(cl_int), (void *)&how_many);
  checkError(status, "Failed to set store kernel arg");

  cl_event startExec_event, endExec_event;
  status = clEnqueueTask(queue1, fetch_kernel, 0, NULL, &startExec_event);
  checkError(status, "Failed to launch fetch kernel");

  status = clEnqueueTask(queue2, ffta_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fft kernel");

  status = clEnqueueTask(queue3, transpose_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch transpose1 kernel");

  status = clEnqueueTask(queue4, fftb_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch second fft kernel");

  status = clEnqueueTask(queue5, store_kernel, 0, NULL, &endExec_event);
  checkError(status, "Failed to launch store kernel");

  // Wait for all command queues to complete pending events
  status = clFinish(queue5);
  checkError(status, "failed to finish queue5");
  status = clFinish(queue1);
  checkError(status, "failed to finish queue1");
  status = clFinish(queue2);
  checkError(status, "failed to finish queue2");
  status = clFinish(queue3);
  checkError(status, "failed to finish queue3");
  status = clFinish(queue4);
  checkError(status, "failed to finish queue4");

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
