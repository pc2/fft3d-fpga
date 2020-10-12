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
 * \brief  compute an out-of-place single precision complex 3D-FFT using the BRAM of the FPGA
 * \param  N    : integer pointer addressing the size of FFT3d  
 * \param  inp  : float2 pointer to input data of size [N * N * N]
 * \param  out  : float2 pointer to output data of size [N * N * N]
 * \param  inv  : int toggle to activate backward FFT
 * \param  interleaving : 1 if using burst interleaved global memory buffers
 * \return fpga_t : time taken in milliseconds for data transfers and execution
 */
fpga_t fftfpgaf_c2c_3d_bram(int N, float2 *inp, float2 *out, bool inv, bool interleaving) {
  fpga_t fft_time = {0.0, 0.0, 0.0, 0};
  cl_int status = 0;

  cl_kernel fft3da_kernel = NULL, fft3db_kernel = NULL, fft3dc_kernel = NULL;
  cl_kernel fetch_kernel = NULL, store_kernel = NULL;
  cl_kernel transpose_kernel = NULL, transpose3d_kernel = NULL;

  // if N is not a power of 2
  if(inp == NULL || out == NULL || ( (N & (N-1)) !=0)){
    return fft_time;
  }

#ifdef VERBOSE
  printf("Launching%s 3d FFT transform \n", inv ? " inverse":"");
#endif

  queue_setup();

  cl_mem_flags flagbuf1, flagbuf2;
  if(interleaving){
    flagbuf1 = CL_MEM_READ_WRITE;
    flagbuf2 = CL_MEM_READ_WRITE;
  }
  else{
    flagbuf1 = CL_MEM_READ_ONLY | CL_CHANNEL_1_INTELFPGA;
    flagbuf2 = CL_MEM_WRITE_ONLY | CL_CHANNEL_2_INTELFPGA;
  }

  // Device memory buffers
  cl_mem d_inData, d_outData;
  d_inData = clCreateBuffer(context, flagbuf1, sizeof(float2) * N * N * N, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");
  d_outData = clCreateBuffer(context, flagbuf2, sizeof(float2) * N * N * N, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

 // Copy data from host to device
  fft_time.pcie_write_t = getTimeinMilliSec();

  status = clEnqueueWriteBuffer(queue1, d_inData, CL_TRUE, 0, sizeof(float2) * N * N * N, inp, 0, NULL, NULL);

  status = clFinish(queue1);
  checkError(status, "failed to finish");

  fft_time.pcie_write_t = getTimeinMilliSec() - fft_time.pcie_write_t;
  checkError(status, "Failed to copy data to device");

  // Can't pass bool to device, so convert it to int
  int inverse_int = (int)inv;

  // Create the kernel - name passed in here must match kernel name in the
  // original CL file, that was compiled into an AOCX file using the AOC tool
  fetch_kernel = clCreateKernel(program, "fetch", &status);
  checkError(status, "Failed to create fetch kernel");

  fft3da_kernel = clCreateKernel(program, "fft3da", &status);
  checkError(status, "Failed to create fft3da kernel");

  transpose_kernel = clCreateKernel(program, "transpose2d", &status);
  checkError(status, "Failed to create transpose kernel");

  fft3db_kernel = clCreateKernel(program, "fft3db", &status);
  checkError(status, "Failed to create fft3db kernel");

  transpose3d_kernel = clCreateKernel(program, "transpose3D", &status);
  checkError(status, "Failed to create transpose3D kernel");

  fft3dc_kernel = clCreateKernel(program, "fft3dc", &status);
  checkError(status, "Failed to create fft3dc kernel");

  store_kernel = clCreateKernel(program, "store", &status);
  checkError(status, "Failed to create store kernel");

  status = clSetKernelArg(fetch_kernel, 0, sizeof(cl_mem), (void *)&d_inData);
  checkError(status, "Failed to set fetch kernel arg 0");
  status = clSetKernelArg(fft3da_kernel, 0, sizeof(cl_int),(void*)&inverse_int);
  checkError(status, "Failed to set fft3da kernel arg 0");
  status = clSetKernelArg(fft3db_kernel, 0, sizeof(cl_int),(void*)&inverse_int);
  checkError(status, "Failed to set fft3db_kernel arg 0");
  status = clSetKernelArg(fft3dc_kernel, 0, sizeof(cl_int),(void*)&inverse_int);
  checkError(status, "Failed to set fft3dc_kernel arg 0");
  status = clSetKernelArg(store_kernel, 0, sizeof(cl_mem), (void *)&d_outData);
  checkError(status, "Failed to set store kernel arg 0");

  fft_time.exec_t = getTimeinMilliSec();
  status = clEnqueueTask(queue1, fetch_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fetch kernel");

  status = clEnqueueTask(queue2, fft3da_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fft kernel");

  status = clEnqueueTask(queue3, transpose_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch transpose kernel");

  status = clEnqueueTask(queue4, fft3db_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch second fft kernel");

  status = clEnqueueTask(queue5, transpose3d_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch second transpose kernel");

  status = clEnqueueTask(queue6, fft3dc_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch third fft kernel");

  status = clEnqueueTask(queue7, store_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch store transpose kernel");

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
  status = clFinish(queue6);
  checkError(status, "failed to finish");
  status = clFinish(queue7);
  checkError(status, "failed to finish");

  fft_time.exec_t = getTimeinMilliSec() - fft_time.exec_t;

  // Copy results from device to host
  fft_time.pcie_read_t = getTimeinMilliSec();
  status = clEnqueueReadBuffer(queue1, d_outData, CL_TRUE, 0, sizeof(float2) * N * N * N, out, 0, NULL, NULL);

  status = clFinish(queue1);
  checkError(status, "failed to finish reading buffer using PCIe");

  fft_time.pcie_read_t = getTimeinMilliSec() - fft_time.pcie_read_t;
  checkError(status, "Failed to copy data from device");

  queue_cleanup();

  if (d_inData)
  	clReleaseMemObject(d_inData);
  if (d_outData) 
	  clReleaseMemObject(d_outData);

  if(fetch_kernel) 
    clReleaseKernel(fetch_kernel);  
  if(fft3da_kernel) 
    clReleaseKernel(fft3da_kernel);  
  if(fft3db_kernel) 
    clReleaseKernel(fft3db_kernel);  
  if(fft3dc_kernel) 
    clReleaseKernel(fft3dc_kernel);  
  if(transpose_kernel) 
    clReleaseKernel(transpose_kernel);  
  if(transpose3d_kernel) 
    clReleaseKernel(transpose3d_kernel); 
  if(store_kernel) 
    clReleaseKernel(store_kernel);  

  fft_time.valid = 1;
  return fft_time;
}

/**
 * \brief  compute an out-of-place single precision complex 3D FFT using the DDR for 3D Transpose where the data access between the host and the FPGA is using Shared Virtual Memory (SVM)
 * \param  N    : integer pointer addressing the size of FFT3d  
 * \param  inp  : float2 pointer to input data of size [N * N * N]
 * \param  out  : float2 pointer to output data of size [N * N * N]
 * \param  inv  : int toggle to activate backward FFT
 * \param  interleaving : 1 if using burst interleaved global memory buffers
 * \return fpga_t : time taken in milliseconds for data transfers and execution
 */
fpga_t fftfpgaf_c2c_3d_ddr_svm(int N, float2 *inp, float2 *out, bool inv) {
  fpga_t fft_time = {0.0, 0.0, 0.0, 0};
  cl_int status = 0;
  int num_pts = N * N * N;
  
  /*
  const char* board_name;
  int *bytes;
  aocl_mmd_offline_info_t info_id;
  info_id = AOCL_MMD_BOARD_NAMES;
  aocl_mmd_get_offline_info(info_id, sizeof(char*), &board_name, size_t(int));

  svm_handle = aocl_mmd_open(board_name);
  if(svm_handle < 0 ){
    return NULL;
  }
  return aocl_mmd_shared_mem_alloc(svm_handle, sz, inData, device_ptr);
  */

  // if N is not a power of 2
  if(inp == NULL || out == NULL || ( (N & (N-1)) !=0) || !(svm_enabled)){
    return fft_time;
  }

#ifdef VERBOSE
  printf("Launching%s 3d FFT transform in DDR \n", inv ? " inverse":"");
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

  // Device memory buffers
  cl_mem d_inOutData;
  d_inOutData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_1_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

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

  // write to fetch kernel using SVM based PCIe
  status = clSetKernelArgSVMPointer(fetch1_kernel, 0, (void *)h_inData);
  checkError(status, "Failed to set fetch1 kernel arg");

  status=clSetKernelArg(ffta_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set ffta kernel arg");
  status=clSetKernelArg(fftb_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set fftb kernel arg");

  // kernel stores to DDR memory
  status=clSetKernelArg(store1_kernel, 0, sizeof(cl_mem), (void*)&d_inOutData);
  checkError(status, "Failed to set store1 kernel arg");

  // kernel fetches from DDR memory
  status=clSetKernelArg(fetch2_kernel, 0, sizeof(cl_mem), (void*)&d_inOutData);
  checkError(status, "Failed to set fetch2 kernel arg");
  status=clSetKernelArg(fftc_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set fftc kernel arg");

  // kernel stores using SVM based PCIe to host
  status = clSetKernelArgSVMPointer(store2_kernel, 0, (void*)h_outData);
  checkError(status, "Failed to set store2 kernel arg");

  fft_time.exec_t = getTimeinMilliSec();
  double first_half = getTimeinMilliSec();
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
  first_half = getTimeinMilliSec() - first_half;

  double second_half = getTimeinMilliSec();
  // enqueue fetch to same queue as the store kernel due to data dependency
  status = clEnqueueTask(queue5, fetch2_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fetch kernel");

  status = clEnqueueTask(queue4, fftc_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fft kernel");

  status = clEnqueueTask(queue3, store2_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch transpose kernel");

  status = clFinish(queue5);
  checkError(status, "failed to finish");
  status = clFinish(queue4);
  checkError(status, "failed to finish");
  status = clFinish(queue3);
  checkError(status, "failed to finish");
  /*
  status = clFinish(queue2);
  checkError(status, "failed to finish");
  status = clFinish(queue1);
  checkError(status, "failed to finish");
  */

  second_half = getTimeinMilliSec() - second_half;
  fft_time.exec_t = getTimeinMilliSec() - fft_time.exec_t;
  
  printf("First half: %lf Second half: %lf\n\n", first_half, second_half);

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

  if (d_inOutData)
    clReleaseMemObject(d_inOutData);

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
 * \brief  compute an out-of-place single precision complex 3D-FFT using the DDR of the FPGA for 3D Transpose
 * \param  N    : integer pointer addressing the size of FFT3d  
 * \param  inp  : float2 pointer to input data of size [N * N * N]
 * \param  out  : float2 pointer to output data of size [N * N * N]
 * \param  inv  : int toggle to activate backward FFT
 * \param  interleaving : 1 if using burst interleaved global memory buffers
 * \return fpga_t : time taken in milliseconds for data transfers and execution
 */
fpga_t fftfpgaf_c2c_3d_ddr(int N, float2 *inp, float2 *out, bool inv) {
  fpga_t fft_time = {0.0, 0.0, 0.0, 0};
  cl_int status = 0;
  int num_pts = N * N * N;
  
  // if N is not a power of 2
  if(inp == NULL || out == NULL || ( (N & (N-1)) !=0)){
    return fft_time;
  }

#ifdef VERBOSE
  printf("Launching%s 3d FFT transform in DDR \n", inv ? " inverse":"");
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

  // Device memory buffers
  cl_mem d_inData, d_transpose, d_outData;
  d_inData = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_CHANNEL_1_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");

  d_transpose = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  d_outData = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_CHANNEL_1_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  // Copy data from host to device
  fft_time.pcie_write_t = getTimeinMilliSec();

  status = clEnqueueWriteBuffer(queue1, d_inData, CL_TRUE, 0, sizeof(float2) * num_pts, inp, 0, NULL, NULL);

  status = clFinish(queue1);
  checkError(status, "failed to finish");

  fft_time.pcie_write_t = getTimeinMilliSec() - fft_time.pcie_write_t;
  checkError(status, "Failed to copy data to device");

  status=clSetKernelArg(fetch1_kernel, 0, sizeof(cl_mem), (void *)&d_inData);
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
  status=clSetKernelArg(store2_kernel, 0, sizeof(cl_mem), (void *)&d_outData);
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

  // enqueue fetch to same queue as the store kernel due to data dependency
  status = clEnqueueTask(queue5, fetch2_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fetch kernel");

  status = clEnqueueTask(queue4, fftc_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fft kernel");

  status = clEnqueueTask(queue3, store2_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch transpose kernel");

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

  fft_time.exec_t = getTimeinMilliSec() - fft_time.exec_t;

  // Copy results from device to host
  fft_time.pcie_read_t = getTimeinMilliSec();
  status = clEnqueueReadBuffer(queue1, d_outData, CL_TRUE, 0, sizeof(float2) * num_pts, out, 0, NULL, NULL);
  
  status = clFinish(queue1);
  checkError(status, "failed to finish reading DDR using PCIe");

  fft_time.pcie_read_t = getTimeinMilliSec() - fft_time.pcie_read_t;
  checkError(status, "Failed to copy data from device");

  queue_cleanup();

  if (d_inData)
    clReleaseMemObject(d_inData);
  if (d_outData) 
    clReleaseMemObject(d_outData);
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
  // a and b are double buffers
  cl_mem d_inData1, d_inData2, d_inData3, d_inData4;
  cl_mem d_outData1, d_outData2, d_outData3, d_outData4;

  d_inData1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_CHANNEL_1_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");

  d_inData2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");

  d_inData3 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_CHANNEL_3_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");

  d_inData4 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_CHANNEL_4_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");

  d_outData1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_CHANNEL_1_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  d_outData2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  d_outData3 = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_CHANNEL_3_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  d_outData4 = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_CHANNEL_4_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  cl_mem d_transpose1, d_transpose2, d_transpose3, d_transpose4;
  d_transpose1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_1_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  d_transpose2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  d_transpose3 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_3_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  d_transpose4 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_4_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  // Default Kernel Arguments
  status=clSetKernelArg(fetch1_kernel, 0, sizeof(cl_mem), (void *)&d_inData1);
  checkError(status, "Failed to set fetch1 kernel arg");
  status=clSetKernelArg(ffta_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set ffta kernel arg");
  status=clSetKernelArg(fftb_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set fftb kernel arg");
  status=clSetKernelArg(store1_kernel, 0, sizeof(cl_mem), (void *)&d_transpose3);
  checkError(status, "Failed to set store1 kernel arg");

  status=clSetKernelArg(fetch2_kernel, 0, sizeof(cl_mem), (void *)&d_transpose3);
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
  //status = clEnqueueWriteBuffer(queue6, d_inData2, CL_TRUE, 0, sizeof(float2) * num_pts, (void*)&inp[num_pts], 0, NULL, NULL);
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
  for(size_t i = 0; i < how_many-2; i++){

    // Unblocking transfers between DDR and host 
    if( (i % 4) == 0){
      status = clEnqueueWriteBuffer(queue7, d_inData3, CL_FALSE, 0, sizeof(float2) * num_pts, &inp[( (i+2) * num_pts)], 0, NULL, &write_event[1]);
      checkError(status, "Failed to write to DDR buffer");

      status = clEnqueueReadBuffer(queue6, d_outData1, CL_FALSE, 0, sizeof(float2) * num_pts, &out[(i * num_pts)], 0, NULL, &write_event[0]);
      checkError(status, "Failed to read from DDR buffer");

      status=clSetKernelArg(fetch1_kernel, 0, sizeof(cl_mem), (void *)&d_inData2);
      checkError(status, "Failed to set fetch1 kernel arg");
    
      status=clSetKernelArg(store1_kernel, 0, sizeof(cl_mem), (void *)&d_transpose4);
      checkError(status, "Failed to set store1 kernel arg");

      status=clSetKernelArg(fetch2_kernel, 0, sizeof(cl_mem), (void *)&d_transpose4);

      status=clSetKernelArg(store2_kernel, 0, sizeof(cl_mem), (void *)&d_outData2);
      checkError(status, "Failed to set store2 kernel arg");
    }
    else if( (i % 4) == 1){
      status = clEnqueueWriteBuffer(queue7, d_inData4, CL_FALSE, 0, sizeof(float2) * num_pts, &inp[((i + 2) * num_pts)], 0, NULL, &write_event[1]);
      checkError(status, "Failed to write to DDR buffer");

      status = clEnqueueReadBuffer(queue6, d_outData2, CL_FALSE, 0, sizeof(float2) * num_pts, &out[(i * num_pts)], 0, NULL, &write_event[0]);
      checkError(status, "Failed to read from DDR buffer");

      status=clSetKernelArg(fetch1_kernel, 0, sizeof(cl_mem), (void *)&d_inData3);
      checkError(status, "Failed to set fetch1 kernel arg");
    
      status=clSetKernelArg(store1_kernel, 0, sizeof(cl_mem), (void *)&d_transpose1);
      checkError(status, "Failed to set store1 kernel arg");

      status=clSetKernelArg(fetch2_kernel, 0, sizeof(cl_mem), (void *)&d_transpose1);

      status=clSetKernelArg(store2_kernel, 0, sizeof(cl_mem), (void *)&d_outData3);
      checkError(status, "Failed to set store2 kernel arg");
    }
    else if( (i % 4) == 2){
      status = clEnqueueWriteBuffer(queue7, d_inData1, CL_FALSE, 0, sizeof(float2) * num_pts, &inp[( (i + 2) * num_pts)], 0, NULL, &write_event[1]);
      checkError(status, "Failed to write to DDR buffer");

      status = clEnqueueReadBuffer(queue6, d_outData3, CL_FALSE, 0, sizeof(float2) * num_pts, &out[(i * num_pts)], 0, NULL, &write_event[0]);
      checkError(status, "Failed to read from DDR buffer");

      status=clSetKernelArg(fetch1_kernel, 0, sizeof(cl_mem), (void *)&d_inData4);
      checkError(status, "Failed to set fetch1 kernel arg");
    
      status=clSetKernelArg(store1_kernel, 0, sizeof(cl_mem), (void *)&d_transpose2);
      checkError(status, "Failed to set store1 kernel arg");

      status=clSetKernelArg(fetch2_kernel, 0, sizeof(cl_mem), (void *)&d_transpose2);

      status=clSetKernelArg(store2_kernel, 0, sizeof(cl_mem), (void *)&d_outData4);
      checkError(status, "Failed to set store2 kernel arg");
    }
    else{
      status = clEnqueueWriteBuffer(queue7, d_inData2, CL_FALSE, 0, sizeof(float2) * num_pts, &inp[( (i+2) * num_pts)], 0, NULL, &write_event[1]);
      checkError(status, "Failed to write to DDR buffer");

      status = clEnqueueReadBuffer(queue6, d_outData4, CL_FALSE, 0, sizeof(float2) * num_pts, &out[(i * num_pts)], 0, NULL, &write_event[0]);
      checkError(status, "Failed to read from DDR buffer");

      status=clSetKernelArg(fetch1_kernel, 0, sizeof(cl_mem), (void *)&d_inData1);
      checkError(status, "Failed to set fetch1 kernel arg");
    
      status=clSetKernelArg(store1_kernel, 0, sizeof(cl_mem), (void *)&d_transpose3);
      checkError(status, "Failed to set store1 kernel arg");

      status=clSetKernelArg(fetch2_kernel, 0, sizeof(cl_mem), (void *)&d_transpose3);
      checkError(status, "Failed to set store1 kernel arg");

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

  if( (how_many % 4) == 0){
    status = clEnqueueReadBuffer(queue6, d_outData3, CL_FALSE, 0, sizeof(float2) * num_pts, &out[(how_many - 2) * num_pts], 0, NULL, &write_event[0]);
    checkError(status, "Failed to read from DDR buffer");

    status=clSetKernelArg(fetch1_kernel, 0, sizeof(cl_mem), (void *)&d_inData4);
    checkError(status, "Failed to set fetch1 kernel arg");
  
    status=clSetKernelArg(store1_kernel, 0, sizeof(cl_mem), (void *)&d_transpose2);
    checkError(status, "Failed to set store1 kernel arg");

    status=clSetKernelArg(fetch2_kernel, 0, sizeof(cl_mem), (void *)&d_transpose2);
    checkError(status, "Failed to set store1 kernel arg");
    
    status=clSetKernelArg(store2_kernel, 0, sizeof(cl_mem), (void *)&d_outData4);
    checkError(status, "Failed to set store2 kernel arg");
  }
  else if((how_many % 4) == 1){
    status = clEnqueueReadBuffer(queue6, d_outData4, CL_FALSE, 0, sizeof(float2) * num_pts, &out[(how_many - 2) * num_pts], 0, NULL, &write_event[0]);
    checkError(status, "Failed to read from DDR buffer");

    status=clSetKernelArg(fetch1_kernel, 0, sizeof(cl_mem), (void *)&d_inData1);
    checkError(status, "Failed to set fetch1 kernel arg");
  
    status=clSetKernelArg(store1_kernel, 0, sizeof(cl_mem), (void *)&d_transpose3);
    checkError(status, "Failed to set store1 kernel arg");

    status=clSetKernelArg(fetch2_kernel, 0, sizeof(cl_mem), (void *)&d_transpose3);
    checkError(status, "Failed to set store1 kernel arg");
    
    status=clSetKernelArg(store2_kernel, 0, sizeof(cl_mem), (void *)&d_outData1);
    checkError(status, "Failed to set store2 kernel arg");
  }
  else if((how_many % 4) == 2){
    status = clEnqueueReadBuffer(queue6, d_outData1, CL_FALSE, 0, sizeof(float2) * num_pts, &out[(how_many - 2) * num_pts], 0, NULL, &write_event[0]);
    checkError(status, "Failed to read from DDR buffer");

    status=clSetKernelArg(fetch1_kernel, 0, sizeof(cl_mem), (void *)&d_inData2);
    checkError(status, "Failed to set fetch1 kernel arg");
  
    status=clSetKernelArg(store1_kernel, 0, sizeof(cl_mem), (void *)&d_transpose4);
    checkError(status, "Failed to set store1 kernel arg");

    status=clSetKernelArg(fetch2_kernel, 0, sizeof(cl_mem), (void *)&d_transpose4);
    checkError(status, "Failed to set store1 kernel arg");
    
    status=clSetKernelArg(store2_kernel, 0, sizeof(cl_mem), (void *)&d_outData2);
    checkError(status, "Failed to set store2 kernel arg");
  }
  else{
    status = clEnqueueReadBuffer(queue6, d_outData2, CL_FALSE, 0, sizeof(float2) * num_pts, &out[(how_many - 2) * num_pts], 0, NULL, &write_event[0]);
    checkError(status, "Failed to read from DDR buffer");

    status=clSetKernelArg(fetch1_kernel, 0, sizeof(cl_mem), (void *)&d_inData3);
    checkError(status, "Failed to set fetch1 kernel arg");
  
    status=clSetKernelArg(store1_kernel, 0, sizeof(cl_mem), (void *)&d_transpose1);
    checkError(status, "Failed to set store1 kernel arg");

    status=clSetKernelArg(fetch2_kernel, 0, sizeof(cl_mem), (void *)&d_transpose1);
    checkError(status, "Failed to set store1 kernel arg");
    
    status=clSetKernelArg(store2_kernel, 0, sizeof(cl_mem), (void *)&d_outData3);
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

  if( (how_many % 4) == 0){
    status = clEnqueueReadBuffer(queue6, d_outData4, CL_FALSE, 0, sizeof(float2) * num_pts, &out[(how_many - 1) * num_pts], 0, NULL, &write_event[0]);
    checkError(status, "Failed to read from DDR buffer");
  }
  else if((how_many % 4) == 1){
    status = clEnqueueReadBuffer(queue6, d_outData1, CL_FALSE, 0, sizeof(float2) * num_pts, &out[(how_many - 1) * num_pts], 0, NULL, &write_event[0]);
    checkError(status, "Failed to read from DDR buffer");
  }
  else if((how_many % 4) == 2){
    status = clEnqueueReadBuffer(queue6, d_outData2, CL_FALSE, 0, sizeof(float2) * num_pts, &out[(how_many - 1) * num_pts], 0, NULL, &write_event[0]);
    checkError(status, "Failed to read from DDR buffer");
  }
  else{
    status = clEnqueueReadBuffer(queue6, d_outData3, CL_FALSE, 0, sizeof(float2) * num_pts, &out[(how_many - 1) * num_pts], 0, NULL, &write_event[0]);
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
  if (d_inData3)
    clReleaseMemObject(d_inData3);
  if (d_inData4)
    clReleaseMemObject(d_inData4);

  if (d_outData2) 
    clReleaseMemObject(d_outData2);
  if (d_outData2) 
    clReleaseMemObject(d_outData2);
  if (d_outData3) 
    clReleaseMemObject(d_outData3);
  if (d_outData4) 
    clReleaseMemObject(d_outData4);

  if (d_transpose1) 
    clReleaseMemObject(d_transpose1);
  if (d_transpose2) 
    clReleaseMemObject(d_transpose2);
  if (d_transpose3) 
    clReleaseMemObject(d_transpose3);
  if (d_transpose4) 
    clReleaseMemObject(d_transpose4);

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
 * \brief compute an batched out-of-place single precision complex 3D-FFT using the DDR of the FPGA for 3D Transpose and for data transfers between host's main memory and FPGA using Shared Virtual Memory 
 * \param N    : integer pointer addressing the size of FFT3d  
 * \param inp  : float2 pointer to input data of size [N * N * N]
 * \param out  : float2 pointer to output data of size [N * N * N]
 * \param inv  : int toggle to activate backward FFT
 * \param how_many : number of batched computations
 * \return fpga_t : time taken in milliseconds for data transfers and execution
 */
fpga_t fftfpgaf_c2c_3d_ddr_svm_batch(int N, float2 *inp, float2 *out, bool inv, int how_many) {
  fpga_t fft_time = {0.0, 0.0, 0.0, 0};
  cl_int status = 0;
  int num_pts = N * N * N;

  // if N is not a power of 2
  if(inp == NULL || out == NULL || ( (N & (N-1)) !=0) || (how_many <= 0)){
    return fft_time;
  }

  if(!svm_enabled){
    return fft_time;
  }

#ifdef VERBOSE
  printf("Launching%s 3d FFT transform in DDR \n", inv ? " inverse":"");
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

  // Device memory buffers: double buffers
  cl_mem d_outData_0;
  d_outData_0 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_1_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  cl_mem d_outData_1;
  d_outData_1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  // allocate and initialize SVM buffers

  float2 *h_inData[how_many], *h_outData[how_many];
  for(size_t i = 0; i < how_many; i++){
    h_inData[i] = (float2 *)clSVMAlloc(context, CL_MEM_READ_ONLY, sizeof(float2) * num_pts, 0);
    h_outData[i] = (float2 *)clSVMAlloc(context, CL_MEM_WRITE_ONLY, sizeof(float2) * num_pts, 0);

    status = clEnqueueSVMMap(queue1, CL_TRUE, CL_MAP_WRITE, (void *)h_inData[i], sizeof(float2) * num_pts, 0, NULL, NULL);
    checkError(status, "Failed to map input data");

    // copy data into h_inData
    size_t stride = i * num_pts; 
    for(size_t j = 0; j < num_pts; j++){
      h_inData[i][j].x = inp[stride + j].x;
      h_inData[i][j].y = inp[stride + j].y;
    }

    status = clEnqueueSVMUnmap(queue1, (void *)h_inData[i], 0, NULL, NULL);
    checkError(status, "Failed to unmap input data");

    status = clEnqueueSVMMap(queue1, CL_TRUE, CL_MAP_WRITE, (void *)h_outData[i], sizeof(float2) * num_pts, 0, NULL, NULL);
    checkError(status, "Failed to map input data");

    // copy data into h_inData
    for(size_t j = 0; j < num_pts; j++){
      h_outData[i][j].x = 0.0;
      h_outData[i][j].y = 0.0;
    }

    status = clEnqueueSVMUnmap(queue1, (void *)h_outData[i], 0, NULL, NULL);
    checkError(status, "Failed to unmap input data");
  }

  /*
  * kernel arguments
  */
  // write to fetch kernel using SVM based PCIe
  status = clSetKernelArgSVMPointer(fetch1_kernel, 0, (void *)h_inData[0]);
  checkError(status, "Failed to set fetch1 kernel arg");

  status=clSetKernelArg(ffta_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set ffta kernel arg");
  status=clSetKernelArg(fftb_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set fftb kernel arg");

  // kernel stores to DDR memory
  status=clSetKernelArg(store1_kernel, 0, sizeof(cl_mem), (void *)&d_outData_0);
  checkError(status, "Failed to set store1 kernel arg");

  /*
  *  First batch write phase
  */
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

  for(size_t i = 1; i < how_many; i++){

    /*
    *  Read phase of previous iteration
    */
    // kernel fetches from DDR memory
    // kernel stores using SVM based PCIe to host
    if( (i % 2) == 1){
      // if odd number of batches
      status=clSetKernelArg(fetch2_kernel, 0, sizeof(cl_mem), (void *)&d_outData_0);
      checkError(status, "Failed to set fetch2 kernel arg");

      // Start fetch2 phase with same queue as store1
      status = clEnqueueTask(queue5, fetch2_kernel, 0, NULL, NULL);
      checkError(status, "Failed to launch fetch kernel");
    }
    else{
      status=clSetKernelArg(fetch2_kernel, 0, sizeof(cl_mem), (void *)&d_outData_1);
      checkError(status, "Failed to set fetch2 kernel arg");

      // Start fetch2 phase with same queue as store1
      status = clEnqueueTask(queue8, fetch2_kernel, 0, NULL, NULL);
      checkError(status, "Failed to launch fetch kernel");
    }
    status=clSetKernelArg(fftc_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
    checkError(status, "Failed to set fftc kernel arg");
    status = clSetKernelArgSVMPointer(store2_kernel, 0, (void *)h_outData[i-1]);
    checkError(status, "Failed to set store2 kernel arg");

    status = clEnqueueTask(queue6, fftc_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch fft kernel");

    status = clEnqueueTask(queue7, store2_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch transpose kernel");

    /*
     *  write phase of current iteration
     */
    // change write phase host and ddr ptrs
    status = clSetKernelArgSVMPointer(fetch1_kernel, 0, (void *)h_inData[i]);
    checkError(status, "Failed to set fetch1 kernel arg");
    if(i % 2 == 1){
      status=clSetKernelArg(store1_kernel, 0, sizeof(cl_mem), (void *)&d_outData_1);
      checkError(status, "Failed to set store1 kernel arg");
    }
    else{
      status=clSetKernelArg(store1_kernel, 0, sizeof(cl_mem), (void *)&d_outData_0);
      checkError(status, "Failed to set store1 kernel arg");
    }

    // Start write phase of current iteration
    status = clEnqueueTask(queue1, fetch1_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch fetch kernel");

    status = clEnqueueTask(queue2, ffta_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch fft kernel");

    status = clEnqueueTask(queue3, transpose_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch transpose kernel");

    status = clEnqueueTask(queue4, fftb_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch second fft kernel");

    if(i % 2 == 1){
      status = clEnqueueTask(queue8, store1_kernel, 0, NULL, NULL);
      checkError(status, "Failed to launch second transpose kernel");
    }
    else{
      status = clEnqueueTask(queue5, store1_kernel, 0, NULL, NULL);
      checkError(status, "Failed to launch second transpose kernel");
    }
  }
  
  if(how_many % 2 == 1){
    status=clSetKernelArg(fetch2_kernel, 0, sizeof(cl_mem), (void *)&d_outData_0);
    checkError(status, "Failed to set fetch2 kernel arg");

    status = clEnqueueTask(queue5, fetch2_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch fetch kernel");
  }
  else{
    status=clSetKernelArg(fetch2_kernel, 0, sizeof(cl_mem), (void *)&d_outData_1);
    checkError(status, "Failed to set fetch2 kernel arg");
    status = clEnqueueTask(queue8, fetch2_kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch fetch kernel");
  }
  status=clSetKernelArg(fftc_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set fftc kernel arg");
  status = clSetKernelArgSVMPointer(store2_kernel, 0, (void *)h_outData[how_many-1]);
  checkError(status, "Failed to set store2 kernel arg");

  status = clEnqueueTask(queue6, fftc_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fft kernel");

  status = clEnqueueTask(queue7, store2_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch transpose kernel");

  status = clFinish(queue1);
  checkError(status, "Failed to finish queue1");
  status = clFinish(queue2);
  checkError(status, "Failed to finish queue2");
  status = clFinish(queue3);
  checkError(status, "Failed to finish queue1");
  status = clFinish(queue4);
  checkError(status, "Failed to finish queue2");
  status = clFinish(queue5);
  checkError(status, "Failed to finish queue1");
  status = clFinish(queue6);
  checkError(status, "Failed to finish queue2");
  status = clFinish(queue7);
  checkError(status, "Failed to finish queue1");
  status = clFinish(queue8);
  checkError(status, "Failed to finish queue2");

  fft_time.exec_t = getTimeinMilliSec() - fft_time.exec_t;

  for(size_t i = 0; i < how_many; i++){

    status = clEnqueueSVMMap(queue2, CL_TRUE, CL_MAP_READ,
      (void *)h_outData[i], sizeof(float2) * num_pts, 0, NULL, NULL);
    checkError(status, "Failed to map out data");

    size_t stride = i * num_pts;
    for(size_t j = 0; j < num_pts; j++){
      out[stride + j].x = h_outData[i][j].x;
      out[stride + j].y = h_outData[i][j].y;
    }

    status = clEnqueueSVMUnmap(queue2, (void *)h_outData[i], 0, NULL, NULL);
    checkError(status, "Failed to unmap out data");
  }

  for(size_t i = 0; i < how_many; i++){
    clSVMFree(context, h_inData[i]);
    clSVMFree(context, h_outData[i]);
  }

  queue_cleanup();

  if (d_outData_0) 
    clReleaseMemObject(d_outData_0);
  if (d_outData_1) 
    clReleaseMemObject(d_outData_1);

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
