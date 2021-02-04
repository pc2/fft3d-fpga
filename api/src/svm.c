#define CL_VERSION_2_0
#include <stdio.h>
#include <stdbool.h>
#include "CL/opencl.h"
#include "svm.h"
#include "opencl_utils.h"

/** 
 * @brief Check if device support svm 
 * @param device
 * @return true if supported and false if not
 */
bool check_valid_svm_device(cl_device_id device){
  cl_device_svm_capabilities caps = 0;
  cl_int status;
  size_t sz_return;

  status = clGetDeviceInfo(
    device,
    CL_DEVICE_SVM_CAPABILITIES,
    sizeof(cl_device_svm_capabilities),
    &caps,
    &sz_return
  );
  checkError(status, "Failed to get device info");
 
  if (caps && CL_DEVICE_SVM_COARSE_GRAIN_BUFFER){
    printf(" -- Found Coarse Grained Buffer SVM capability\n");
    return true;
  }
  else if(caps && CL_DEVICE_SVM_FINE_GRAIN_BUFFER){
    fprintf(stderr, "Found CL_DEVICE_SVM_FINE_GRAIN_BUFFER. API support in progress\n");
    return true;
  }
  else if((caps && CL_DEVICE_SVM_FINE_GRAIN_BUFFER) && (caps &&CL_DEVICE_SVM_ATOMICS)){
    fprintf(stderr, "Found CL_DEVICE_SVM_FINE_GRAIN_BUFFER with support for CL_DEVICE_SVM_ATOMICS. API support in progress\n");
    return true;
  }
  else if(caps && CL_DEVICE_SVM_FINE_GRAIN_SYSTEM){
    fprintf(stderr, "Found CL_DEVICE_SVM_FINE_GRAIN_SYSTEM. API support in progress\n");
    return false;
  }
  else if((caps && CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) && (caps &&CL_DEVICE_SVM_ATOMICS)){
    fprintf(stderr, "Found CL_DEVICE_SVM_FINE_GRAIN_SYSTEM with support for CL_DEVICE_SVM_ATOMICS. API support in progress\n");
    return false;
  }
  else{
    fprintf(stderr, "No SVM Support found!");
    return false;
  }
  return false;
}