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
#include "opencl_utils.h"
#include "misc.h"
#include "svm.h"

#define WR_GLOBALMEM 0
#define RD_GLOBALMEM 1
#define BATCH 2

/**
 * \brief  compute an out-of-place single precision complex 3D FFT using the DDR for 3D Transpose where the data access between the host and the FPGA is using Shared Virtual Memory (SVM)
 * \param  N    : unsigned integer denoting  the size of FFT3d  
 * \param  inp  : float2 pointer to input data of size [N * N * N]
 * \param  out  : float2 pointer to output data of size [N * N * N]
 * \param  inv  : toggle to activate backward FFT
 * \param  interleaving : toggle to use  burst interleaved global memory buffers
 * \return fpga_t : time taken in milliseconds for data transfers and execution
 */
fpga_t fftfpgaf_c2c_3d_ddr_svm(const unsigned N, const float2 *inp, float2 *out, const bool inv, const bool interleaving) {
  fpga_t fft_time = {0.0, 0.0, 0.0, 0.0, 0.0, false};
  cl_int status = 0;
  unsigned num_pts = N * N * N;

  // 0 - WR_GLOBALMEM, 1 - RD_GLOBALMEM, 2 - BATCH
  int mode = WR_GLOBALMEM;

  // if N is not a power of 2
  if(inp == NULL || out == NULL || ( (N & (N-1)) !=0) || !(svm_enabled)){
    return fft_time;
  }

  // Can't pass bool to device, so convert it to int
  int inverse_int = (int)inv;

  // Setup kernels
  cl_kernel fetch_kernel = clCreateKernel(program, "fetch", &status);
  checkError(status, "Failed to create fetch kernel");
  cl_kernel ffta_kernel = clCreateKernel(program, "fft3da", &status);
  checkError(status, "Failed to create fft3da kernel");
  cl_kernel transpose_kernel = clCreateKernel(program, "transpose", &status);
  checkError(status, "Failed to create transpose kernel");
  cl_kernel fftb_kernel = clCreateKernel(program, "fft3db", &status);
  checkError(status, "Failed to create fft3db kernel");
  cl_kernel transpose3D_kernel= clCreateKernel(program, "transpose3D", &status);
  checkError(status, "Failed to create transpose3D kernel");

  cl_kernel fftc_kernel = clCreateKernel(program, "fft3dc", &status);
  checkError(status, "Failed to create fft3dc kernel");
  cl_kernel store_kernel = clCreateKernel(program, "store", &status);
  checkError(status, "Failed to create store kernel");

  // Setup Queues to the kernels
  queue_setup();

  // Device memory buffers
  cl_mem d_inOutData;
  if(!interleaving){
    d_inOutData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_1_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
    checkError(status, "Failed to allocate output device buffer\n");
  }
  else{
    d_inOutData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float2) * num_pts, NULL, &status);
    checkError(status, "Failed to allocate output device buffer\n");
  }

  // allocate SVM buffers
  float2 *h_inData, *h_outData;
  h_inData = (float2 *)clSVMAlloc(context, CL_MEM_READ_ONLY, sizeof(float2) * num_pts, 0);
  h_outData = (float2 *)clSVMAlloc(context, CL_MEM_WRITE_ONLY, sizeof(float2) * num_pts, 0);

  size_t num_bytes = num_pts * sizeof(float2);
  double svm_copyin_t = getTimeinMilliSec();

  status = clEnqueueSVMMap(queue1, CL_TRUE, CL_MAP_WRITE, (void *)h_inData, sizeof(float2) * num_pts, 0, NULL, NULL);
  checkError(status, "Failed to map input data");

  // copy data into h_inData
  memcpy(h_inData, inp, num_bytes);

  status = clEnqueueSVMUnmap(queue1, (void *)h_inData, 0, NULL, NULL);
  checkError(status, "Failed to unmap input data");
  fft_time.svm_copyin_t += getTimeinMilliSec() - svm_copyin_t;

  status = clEnqueueSVMMap(queue1, CL_TRUE, CL_MAP_WRITE, (void *)h_outData, sizeof(float2) * num_pts, 0, NULL, NULL);
  checkError(status, "Failed to map input data");

  // copy data into h_inData
  memset(&h_outData[0], 0, num_bytes);

  status = clEnqueueSVMUnmap(queue1, (void *)h_outData, 0, NULL, NULL);
  checkError(status, "Failed to unmap input data");

  /*
  * kernel arguments
  */
  // write to fetch kernel using SVM based PCIe
  status = clSetKernelArgSVMPointer(fetch_kernel, 0, (void *)h_inData);
  checkError(status, "Failed to set fetch kernel arg");

  status=clSetKernelArg(ffta_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set ffta kernel arg");
  status=clSetKernelArg(fftb_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set fftb kernel arg");

  // kernel stores to DDR memory
  status=clSetKernelArg(transpose3D_kernel, 0, sizeof(cl_mem), (void*)&d_inOutData);
  checkError(status, "Failed to set transpose3D kernel arg");

  // kernel fetches from DDR memory
  status=clSetKernelArg(transpose3D_kernel, 1, sizeof(cl_mem), (void*)&d_inOutData);
  checkError(status, "Failed to set transpose3D kernel arg");

  mode = WR_GLOBALMEM; 

  status=clSetKernelArg(transpose3D_kernel, 2, sizeof(cl_int), (void*)&mode);
  checkError(status, "Failed to set transpose3D kernel arg 2");

  status=clSetKernelArg(fftc_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set fftc kernel arg");

  // kernel stores using SVM based PCIe to host
  status = clSetKernelArgSVMPointer(store_kernel, 0, (void*)h_outData);
  checkError(status, "Failed to set store kernel arg");

  cl_event startExec_event, endExec_event;
  status = clEnqueueTask(queue7, store_kernel, 0, NULL, &endExec_event);
  checkError(status, "Failed to launch transpose kernel");

  status = clEnqueueTask(queue6, fftc_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fft kernel");

  status = clEnqueueTask(queue5, transpose3D_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fetch kernel");

  mode = RD_GLOBALMEM;

  status = clSetKernelArg(transpose3D_kernel, 2, sizeof(cl_int), (void*)&mode);
  checkError(status, "Failed to set transpose3D kernel arg 2");

  status = clEnqueueTask(queue5, transpose3D_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fetch kernel");

  status = clEnqueueTask(queue4, fftb_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch second fft kernel");

  status = clEnqueueTask(queue3, transpose_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch transpose kernel");

  status = clEnqueueTask(queue2, ffta_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fft kernel");

  status = clEnqueueTask(queue1, fetch_kernel, 0, NULL,  &startExec_event);
  checkError(status, "Failed to launch fetch kernel");

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

  cl_ulong kernel_start = 0, kernel_end = 0;

  clGetEventProfilingInfo(startExec_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel_start, NULL);
  clGetEventProfilingInfo(endExec_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel_end, NULL);

  fft_time.exec_t = (cl_double)(kernel_end - kernel_start) * (cl_double)(1e-06);

  double svm_copyout_t = 0.0;
  svm_copyout_t = getTimeinMilliSec();
  status = clEnqueueSVMMap(queue1, CL_TRUE, CL_MAP_READ,
    (void *)h_outData, sizeof(float2) * num_pts, 0, NULL, NULL);
  checkError(status, "Failed to map out data");

  memcpy(out, h_outData, num_bytes);

  status = clEnqueueSVMUnmap(queue1, (void *)h_outData, 0, NULL, NULL);
  checkError(status, "Failed to unmap out data");
  fft_time.svm_copyout_t += getTimeinMilliSec() - svm_copyout_t;

  if (h_inData)
    clSVMFree(context, h_inData);
  if (h_outData)
    clSVMFree(context, h_outData);

  queue_cleanup();

  if (d_inOutData)
    clReleaseMemObject(d_inOutData);

  if(fetch_kernel) 
    clReleaseKernel(fetch_kernel);  

  if(ffta_kernel) 
    clReleaseKernel(ffta_kernel);  
  if(fftb_kernel) 
    clReleaseKernel(fftb_kernel);  
  if(fftc_kernel) 
    clReleaseKernel(fftc_kernel);  

  if(transpose_kernel) 
    clReleaseKernel(transpose_kernel);  

  if(transpose3D_kernel) 
    clReleaseKernel(transpose3D_kernel);  

  if(store_kernel) 
    clReleaseKernel(store_kernel);  

  fft_time.valid = true;
  return fft_time;
}


/**
 * \brief compute a batched out-of-place single precision complex 3D-FFT using the DDR of the FPGA for 3D Transpose and for data transfers between host's main memory and FPGA using Shared Virtual Memory 
 * \param N    : unsigned integer denoting the size of FFT3d  
 * \param inp  : float2 pointer to input data of size [N * N * N]
 * \param out  : float2 pointer to output data of size [N * N * N]
 * \param inv  : toggle to activate backward FFT
 * \param how_many : number of batched computations
 * \return fpga_t : time taken in milliseconds for data transfers and execution
 */
fpga_t fftfpgaf_c2c_3d_ddr_svm_batch(const unsigned N, const float2 *inp, float2 *out, const bool inv, const unsigned how_many) {
  fpga_t fft_time = {0.0, 0.0, 0.0, 0.0, 0.0, false};
  cl_int status = 0;
  // 0 - WR_GLOBALMEM, 1 - RD_GLOBALMEM, 2 - BATCH
  int mode_transpose = WR_GLOBALMEM;

  // if N is not a power of 2
  if(inp == NULL || out == NULL || ( (N & (N-1)) !=0) || (how_many <= 0) || !svm_enabled){
    return fft_time;
  }

  // Can't pass bool to device, so convert it to int
  int inverse_int = (int)inv;

  // Setup kernels
  cl_kernel fetch_kernel = clCreateKernel(program, "fetch", &status);
  checkError(status, "Failed to create fetch kernel");
  cl_kernel ffta_kernel = clCreateKernel(program, "fft3da", &status);
  checkError(status, "Failed to create fft3da kernel");
  cl_kernel transpose_kernel = clCreateKernel(program, "transpose", &status);
  checkError(status, "Failed to create transpose kernel");
  cl_kernel fftb_kernel = clCreateKernel(program, "fft3db", &status);
  checkError(status, "Failed to create fft3db kernel");
  cl_kernel transpose3D_kernel = clCreateKernel(program, "transpose3D", &status);
  checkError(status, "Failed to create transpose3D kernel");

  cl_kernel fftc_kernel = clCreateKernel(program, "fft3dc", &status);
  checkError(status, "Failed to create fft3dc kernel");
  cl_kernel store_kernel = clCreateKernel(program, "store", &status);
  checkError(status, "Failed to create store kernel");

  // Setup Queues to the kernels
  queue_setup();

  // Device memory buffers: double buffers
  unsigned num_pts = N * N * N;

  cl_mem d_inOutData_0 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_1_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  cl_mem d_inOutData_1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  // allocate and initialize SVM buffers
  double svm_copyin_t = 0.0;
  float2 *h_inData[how_many], *h_outData[how_many];
  for(size_t i = 0; i < how_many; i++){
    
    h_inData[i] = (float2 *)clSVMAlloc(context, CL_MEM_READ_ONLY, sizeof(float2) * num_pts, 0);
    h_outData[i] = (float2 *)clSVMAlloc(context, CL_MEM_WRITE_ONLY, sizeof(float2) * num_pts, 0);

    size_t num_bytes = num_pts * sizeof(float2);

    svm_copyin_t = getTimeinMilliSec();
    status = clEnqueueSVMMap(queue1, CL_TRUE, CL_MAP_WRITE, (void *)h_inData[i], sizeof(float2) * num_pts, 0, NULL, NULL);
    checkError(status, "Failed to map input data");

    // copy data into h_inData
    memcpy(&h_inData[i][0], &inp[i*num_pts], num_bytes);

    status = clEnqueueSVMUnmap(queue1, (void *)h_inData[i], 0, NULL, NULL);
    checkError(status, "Failed to unmap input data");
    fft_time.svm_copyin_t += getTimeinMilliSec() - svm_copyin_t;

    status = clEnqueueSVMMap(queue1, CL_TRUE, CL_MAP_WRITE, (void *)h_outData[i], sizeof(float2) * num_pts, 0, NULL, NULL);
    checkError(status, "Failed to map input data");

    // set h_outData to 0
    memset(&h_outData[i][0], 0, num_bytes);

    status = clEnqueueSVMUnmap(queue1, (void *)h_outData[i], 0, NULL, NULL);
    checkError(status, "Failed to unmap input data");
  }

  /*
  * kernel arguments
  */
  // write to fetch kernel using SVM based PCIe
  status = clSetKernelArgSVMPointer(fetch_kernel, 0, (void *)h_inData[0]);
  checkError(status, "Failed to set fetch1 kernel arg");

  status=clSetKernelArg(ffta_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set ffta kernel arg");
  // transpose() has no arguments
  status=clSetKernelArg(fftb_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set fftb kernel arg");

  // kernel stores to DDR memory
  status=clSetKernelArg(transpose3D_kernel, 0, sizeof(cl_mem), (void*)&d_inOutData_1);
  checkError(status, "Failed to set transpose3D kernel arg");

  status=clSetKernelArg(transpose3D_kernel, 1, sizeof(cl_mem), (void*)&d_inOutData_0);
  checkError(status, "Failed to set transpose3D kernel arg");

  mode_transpose = WR_GLOBALMEM;
  status=clSetKernelArg(transpose3D_kernel, 2, sizeof(cl_int), (void*)&mode_transpose);
  checkError(status, "Failed to set transpose3D kernel arg");

  status=clSetKernelArg(fftc_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set fftc kernel arg");

  cl_event startExec_event, endExec_event;
  /*
  *  First batch write phase
  */
  fft_time.exec_t = getTimeinMilliSec();
  status = clEnqueueTask(queue5, transpose3D_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch second transpose kernel");

  status = clEnqueueTask(queue4, fftb_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch second fft kernel");

  status = clEnqueueTask(queue3, transpose_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch transpose kernel");

  status = clEnqueueTask(queue2, ffta_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fft kernel");

  status = clEnqueueTask(queue1, fetch_kernel, 0, NULL, &startExec_event);
  checkError(status, "Failed to launch fetch kernel");

  status = clFinish(queue1);
  checkError(status, "Failed to finish queue1");
  status = clFinish(queue2);
  checkError(status, "Failed to finish queue2");
  status = clFinish(queue3);
  checkError(status, "Failed to finish queue3");
  status = clFinish(queue4);
  checkError(status, "Failed to finish queue4");
  status = clFinish(queue5);
  checkError(status, "Failed to finish queue5");

  for(size_t i = 1; i < how_many; i++){

    status = clSetKernelArgSVMPointer(fetch_kernel, 0, (void *)h_inData[i]);
    checkError(status, "Failed to set fetch kernel arg");

    status = clSetKernelArg(transpose3D_kernel, 0, sizeof(cl_mem), ((i % 2) == 1) ? (void*)&d_inOutData_0 : (void*)&d_inOutData_1);
    checkError(status, "Failed to set transpose3D kernel arg 0");

    status = clSetKernelArg(transpose3D_kernel, 1, sizeof(cl_mem), ((i % 2) == 1) ? (void*)&d_inOutData_1 : (void*)&d_inOutData_0);
    checkError(status, "Failed to set transpose3D kernel arg 1");

    mode_transpose = BATCH;
    status=clSetKernelArg(transpose3D_kernel, 2, sizeof(cl_int), (void*)&mode_transpose);
    checkError(status, "Failed to set transpose3D kernel arg 2");

    status = clSetKernelArgSVMPointer(store_kernel, 0, (void *)h_outData[i-1]);
    checkError(status, "Failed to set store kernel arg");

    // Enqueue Tasks
    status = clEnqueueTask(queue5, transpose3D_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch transpose3D kernel");

    status = clEnqueueTask(queue1, fetch_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch fetch kernel");

    status = clEnqueueTask(queue2, ffta_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch fft kernel");

    status = clEnqueueTask(queue3, transpose_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch transpose kernel");

    status = clEnqueueTask(queue4, fftb_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch second fft kernel");

    status = clEnqueueTask(queue6, fftc_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch fft kernel");

    status = clEnqueueTask(queue7, store_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch store kernel");

    status = clFinish(queue1);
    checkError(status, "Failed to finish queue1");
    status = clFinish(queue2);
    checkError(status, "Failed to finish queue2");
    status = clFinish(queue3);
    checkError(status, "Failed to finish queue3");
    status = clFinish(queue4);
    checkError(status, "Failed to finish queue4");
    status = clFinish(queue5);
    checkError(status, "Failed to finish queue5");
    status = clFinish(queue6);
    checkError(status, "Failed to finish queue6");
    status = clFinish(queue7);
    checkError(status, "Failed to finish queue7");
  }
  
  status = clSetKernelArg(transpose3D_kernel, 0, sizeof(cl_mem), ((how_many % 2) == 0) ? (void*)&d_inOutData_1 : (void*)&d_inOutData_0);
  checkError(status, "Failed to set transpose3D kernel arg 0");

  status = clSetKernelArg(transpose3D_kernel, 1, sizeof(cl_mem), ((how_many % 2) == 0) ? (void*)&d_inOutData_0 : (void*)&d_inOutData_1);
  checkError(status, "Failed to set transpose3D kernel arg 1");

  mode_transpose = RD_GLOBALMEM;
  status=clSetKernelArg(transpose3D_kernel, 2, sizeof(cl_int), (void*)&mode_transpose);
  checkError(status, "Failed to set transpose3D kernel arg 2");

  status = clEnqueueTask(queue5, transpose3D_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch transpose3D kernel");

  status = clEnqueueTask(queue6, fftc_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fft kernel");

  status = clSetKernelArgSVMPointer(store_kernel, 0, (void *)h_outData[how_many - 1]);
  checkError(status, "Failed to set store kernel arg");
  status = clEnqueueTask(queue7, store_kernel, 0, NULL, &endExec_event);
  checkError(status, "Failed to launch store kernel");
  
  status = clFinish(queue5);
  checkError(status, "Failed to finish queue5");
  status = clFinish(queue6);
  checkError(status, "Failed to finish queue6");
  status = clFinish(queue7);
  checkError(status, "Failed to finish queue7");

  fft_time.exec_t = getTimeinMilliSec() - fft_time.exec_t;

  cl_ulong kernel_start = 0, kernel_end = 0;

  clGetEventProfilingInfo(startExec_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel_start, NULL);
  clGetEventProfilingInfo(endExec_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel_end, NULL);

  fft_time.exec_t = (cl_double)(kernel_end - kernel_start) * (cl_double)(1e-06);

  double svm_copyout_t = 0.0;
  for(size_t i = 0; i < how_many; i++){

    // copy data into h_outData
    size_t num_bytes = num_pts * sizeof(float2);
    svm_copyout_t = getTimeinMilliSec();

    status = clEnqueueSVMMap(queue1, CL_TRUE, CL_MAP_READ,
      (void *)h_outData[i], sizeof(float2) * num_pts, 0, NULL, NULL);
    checkError(status, "Failed to map out data");

    memcpy(&out[i*num_pts], &h_outData[i][0], num_bytes);

    status = clEnqueueSVMUnmap(queue1, (void *)h_outData[i], 0, NULL, NULL);
    checkError(status, "Failed to unmap out data");
    fft_time.svm_copyout_t += getTimeinMilliSec() - svm_copyout_t;
  }

  for(size_t i = 0; i < how_many; i++){
    clSVMFree(context, h_inData[i]);
    clSVMFree(context, h_outData[i]);
  }

  queue_cleanup();

  if (d_inOutData_0) 
    clReleaseMemObject(d_inOutData_0);
  if (d_inOutData_1) 
    clReleaseMemObject(d_inOutData_1);

  if(fetch_kernel) 
    clReleaseKernel(fetch_kernel);  

  if(ffta_kernel) 
    clReleaseKernel(ffta_kernel);  
  if(fftb_kernel) 
    clReleaseKernel(fftb_kernel);  
  if(fftc_kernel) 
    clReleaseKernel(fftc_kernel);  

  if(transpose_kernel) 
    clReleaseKernel(transpose_kernel);  

  if(transpose3D_kernel) 
    clReleaseKernel(transpose3D_kernel);  

  if(store_kernel) 
    clReleaseKernel(store_kernel);  

  fft_time.valid = true;
  return fft_time;
}