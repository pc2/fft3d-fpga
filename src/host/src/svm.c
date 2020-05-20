#include "aocl_mmd.h"

int replace(){
  const char* board_name;
  aocl_mmd_offline_info_t info_id;
  aocl_mmd_get_offline_info(info_id, );

  return aocl_mmd_open(board_name);
}


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


  // return if SVM enabled but no device supported
  /*
  if(use_svm){
    // TODO: emulation and svm
    if (check_valid_svm_device(device)){
      svm_enabled = 1;
    }
    else{
      fpga_final();
      return 1;
    }
    return 1;
  }
  */