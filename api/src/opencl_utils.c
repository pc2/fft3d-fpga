// Author: Arjun Ramaswami

#include <stdio.h>  
#include <stdlib.h> // size_t, EXIT_FAILURE, NULL, EXIT_SUCCESS
#include <string.h> // strlen, strstr
#include <stdarg.h> // valist, va_start, va_end
#include <unistd.h> // access in fileExists()
#include <ctype.h>  // tolower
#include <stdbool.h> // true, false

#include "CL/opencl.h"
#include "opencl_utils.h"
#include "fftfpga/fftfpga.h"

// function prototype
static void tolowercase(const char *p, char *q);
static size_t loadBinary(const char *binary_path, char **buf);

/**
 * \brief  returns the first platform id with the name passed as argument
 * \param  platform_name: string to search for platform of particular name
 * \return platform_id (is a typedef struct*) or NULL if not found 
 */
cl_platform_id findPlatform(const char *platform_name){
  cl_uint status;

  if(platform_name == NULL || strlen(platform_name) == 0){
    return NULL;
  }

  // Check if there are any platforms available
  cl_uint num_platforms;
  status = clGetPlatformIDs(0, NULL, &num_platforms);
  if (status != CL_SUCCESS){
    fprintf(stderr, "Query for number of platforms failed\n");
    return NULL;
  }

  // Get ids of platforms available
  cl_platform_id *pids = (cl_platform_id*) malloc(sizeof(cl_platform_id) * num_platforms);
  status = clGetPlatformIDs(num_platforms, pids, NULL);
  if (status != CL_SUCCESS){
    fprintf(stderr, "Query for platform ids failed\n");
    free(pids);
    return NULL;
  }

  // Convert argument string to lowercase to compare platform names
  size_t pl_len = strlen(platform_name);  // needs boundary check?
  char name_search[pl_len + 1];   // VLA
  tolowercase(platform_name, name_search);

#ifndef NDEBUG
  printf("Num of Platforms found - %d\n", num_platforms);
#endif

  // Search the platforms for the platform name passed as argument
  for(int i = 0; i < num_platforms; i++){
    // Get the size of the platform name referred to by the id
    size_t sz;
		status = clGetPlatformInfo(pids[i], CL_PLATFORM_NAME, 0, NULL, &sz);
    if (status != CL_SUCCESS){
      fprintf(stderr, "Query for platform info failed\n");
      free(pids);
      return NULL;
    }

    // Store the name of string size
    char plat_name[sz], plat_name_lc[sz];
	  status = clGetPlatformInfo(pids[i], CL_PLATFORM_NAME, sz, plat_name, NULL);
    if (status != CL_SUCCESS){
      fprintf(stderr, "Query for platform info failed\n");
      free(pids);
      return NULL;
    }

    tolowercase(plat_name, plat_name_lc);
#ifndef NDEBUG
    printf("  %d - %s \n", i, plat_name_lc);
#endif
    if( strstr(plat_name_lc, name_search)){
      cl_platform_id pid = pids[i];
      free(pids);
      return pid;
    }
  }
  free(pids);
  return NULL;
}

/**
 * \brief  gets the list of all devices for the specfic platform
 * \param  platform_id: id of the platform to search for devices
 * \param  specific type of device to search for
 * \param  total number of devices found for the given platform
 * \return array of device ids or NULL if not found
 */
cl_device_id* getDevices(cl_platform_id pid, cl_device_type device_type, cl_uint *num_devices) {
  cl_int status;

  if(pid == NULL || device_type == 0){
    fprintf(stderr, "Bad arguments passed\n");
    return NULL;
  }

  // Query for number of devices
  status = clGetDeviceIDs(pid, device_type, 0, NULL, num_devices);
  if(status != CL_SUCCESS){
    fprintf(stderr, "Query for number of devices failed\n");
    return NULL;
  }

  //  Based on the number of devices get their device ids
  cl_device_id *dev_ids = (cl_device_id*) malloc(sizeof(cl_device_id) * (*num_devices));
  status = clGetDeviceIDs(pid, device_type, *num_devices, dev_ids, NULL);
  if(status != CL_SUCCESS){
    fprintf(stderr, "Query for device ids failed\n");
    free(dev_ids);
    return NULL;
  }
  return dev_ids;
}

/**
 * \brief  checks if file exists with given filename
 * \param  filename: string
 * \retval true if successful and false otherwise
 */
static bool fileExists(const char* filename){
  if( access( filename, R_OK ) != -1 ) {
    return true;
  } else {
    return false;
  }
}

/**
 * \brief  returns the program created from the binary found in the path.
 * \param  context: context created using device
 * \param  devices: array of devices
 * \param  num_devices: number of devices to load binaries into
 * \param  path: path to binary
 * \retval created program or NULL if unsuccessful
 */
cl_program getProgramWithBinary(cl_context context, cl_device_id *devices, cl_uint num_devices, const char *path){
  char *binary, *binaries[num_devices];
  cl_int bin_status, status;

  if(num_devices == 0)
    return NULL;

  if (!fileExists(path)){
    fprintf(stderr, "File not found in path %s\n", path);
    return NULL;
  }

  // Load binary to character array
  size_t bin_size = loadBinary(path, &binary);
  if(bin_size == 0){
    fprintf(stderr, "Could not load binary\n");
    return NULL;
  }

  binaries[0] = binary;

  // Create the program.
  cl_program program = clCreateProgramWithBinary(context, 1, devices, &bin_size, (const unsigned char **) binaries, &bin_status, &status);
  if (status != CL_SUCCESS){
    fprintf(stderr, "Query to create program with binary failed\n");
    free(binary);
    return NULL;
  }
  free(binary);
  return program;
}

static size_t loadBinary(const char *binary_path, char **buf){
  FILE *fp;

  // Open file and check if it exists
  fp = fopen(binary_path, "rb");
  if(fp == 0){
    return 0;
  }

  // Find the size of the file
  fseek(fp, 0L, SEEK_END);
  size_t bin_size = ftell(fp);

  *buf = (char *)malloc(bin_size);
  rewind(fp);    // point to beginning of file

  if(fread((void *)*buf, bin_size, 1, fp) == 0) {
    free(*buf);
    fclose(fp);
    return 0;
  }

  fclose(fp);
  return bin_size;
}

/**
 * \brief  Allocate host side buffers to be 64-byte aligned to make use of DMA 
 *         transfer between host and global memory
 * \param  size in bytes : allocate size bytes multiples of 64
 * \return pointer to allocated memory on successful allocation otherwise NULL
 */
void* alignedMalloc(size_t size){
  size_t OPENCL_ALIGNMENT = 64;
  void *memptr = NULL;
  int ret = posix_memalign(&memptr, OPENCL_ALIGNMENT, size);
  if (ret != 0){
    return NULL;
  }
  return memptr;
}

static void printError(cl_int error) {

  switch(error){
    case -1:
      printf("CL_DEVICE_NOT_FOUND ");
      break;
    case -2:
      printf("CL_DEVICE_NOT_AVAILABLE ");
      break;
    case -3:
      printf("CL_COMPILER_NOT_AVAILABLE ");
      break;
    case -4:
      printf("CL_MEM_OBJECT_ALLOCATION_FAILURE ");
      break;
    case -5:
      printf("CL_OUT_OF_RESOURCES ");
      break;
    case -6:
      printf("CL_OUT_OF_HOST_MEMORY ");
      break;
    case -7:
      printf("CL_PROFILING_INFO_NOT_AVAILABLE ");
      break;
    case -8:
      printf("CL_MEM_COPY_OVERLAP ");
      break;
    case -9:
      printf("CL_IMAGE_FORMAT_MISMATCH ");
      break;
    case -10:
      printf("CL_IMAGE_FORMAT_NOT_SUPPORTED ");
      break;
    case -11:
      printf("CL_BUILD_PROGRAM_FAILURE ");
      break;
    case -12:
      printf("CL_MAP_FAILURE ");
      break;
    case -13:
      printf("CL_MISALIGNED_SUB_BUFFER_OFFSET ");
      break;
    case -14:
      printf("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST ");
      break;
    case -15:
      printf("CL_COMPILE_PROGRAM_FAILURE ");
      break;
    case -16:
      printf("CL_LINKER_NOT_AVAILABLE ");
      break;
    case -17:
      printf("CL_LINK_PROGRAM_FAILURE ");
      break;
    case -18:
      printf("CL_DEVICE_PARTITION_FAILED ");
      break;
    case -19:
      printf("CL_KERNEL_ARG_INFO_NOT_AVAILABLE ");
      break;

    case -30:
      printf("CL_INVALID_VALUE ");
      break;
    case -31:
      printf("CL_INVALID_DEVICE_TYPE ");
      break;
    case -32:
      printf("CL_INVALID_PLATFORM ");
      break;
    case -33:
      printf("CL_INVALID_DEVICE ");
      break;
    case -34:
      printf("CL_INVALID_CONTEXT ");
      break;
    case -35:
      printf("CL_INVALID_QUEUE_PROPERTIES ");
      break;
    case -36:
      printf("CL_INVALID_COMMAND_QUEUE ");
      break;
    case -37:
      printf("CL_INVALID_HOST_PTR ");
      break;
    case -38:
      printf("CL_INVALID_MEM_OBJECT ");
      break;
    case -39:
      printf("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR ");
      break;
    case -40:
      printf("CL_INVALID_IMAGE_SIZE ");
      break;
    case -41:
      printf("CL_INVALID_SAMPLER ");
      break;
    case -42:
      printf("CL_INVALID_BINARY ");
      break;
    case -43:
      printf("CL_INVALID_BUILD_OPTIONS ");
      break;
    case -44:
      printf("CL_INVALID_PROGRAM ");
      break;
    case -45:
      printf("CL_INVALID_PROGRAM_EXECUTABLE ");
      break;
    case -46:
      printf("CL_INVALID_KERNEL_NAME ");
      break;
    case -47:
      printf("CL_INVALID_KERNEL_DEFINITION ");
      break;
    case -48:
      printf("CL_INVALID_KERNEL ");
      break;
    case -49:
      printf("CL_INVALID_ARG_INDEX ");
      break;
    case -50:
      printf("CL_INVALID_ARG_VALUE ");
      break;
    case -51:
      printf("CL_INVALID_ARG_SIZE ");
      break;
    case -52:
      printf("CL_INVALID_KERNEL_ARGS ");
      break;
    case -53:
      printf("CL_INVALID_WORK_DIMENSION ");
      break;
    case -54:
      printf("CL_INVALID_WORK_GROUP_SIZE ");
      break;
    case -55:
      printf("CL_INVALID_WORK_ITEM_SIZE ");
      break;
    case -56:
      printf("CL_INVALID_GLOBAL_OFFSET ");
      break;
    case -57:
      printf("CL_INVALID_EVENT_WAIT_LIST ");
      break;
    case -58:
      printf("CL_INVALID_EVENT ");
      break;
    case -59:
      printf("CL_INVALID_OPERATION ");
      break;
    case -60:
      printf("CL_INVALID_GL_OBJECT ");
      break;
    case -61:
      printf("CL_INVALID_BUFFER_SIZE ");
      break;
    case -62:
      printf("CL_INVALID_MIP_LEVEL ");
      break;
    case -63:
      printf("CL_INVALID_GLOBAL_WORK_SIZE ");
      break;
    case -64:
      printf("CL_INVALID_PROPERTY ");
      break;
    case -65:
      printf("CL_INVALID_IMAGE_DESCRIPTOR ");
      break;
    case -66:
      printf("CL_INVALID_COMPILER_OPTIONS ");
      break;
    case -67:
      printf("CL_INVALID_LINKER_OPTIONS ");
      break;
    case -68:
      printf("CL_INVALID_DEVICE_PARTITION_COUNT ");
      break;
    case -69:
      printf("CL_INVALID_PIPE_SIZE ");
      break;
    case -70:
      printf("CL_INVALID_DEVICE_QUEUE ");
      break;

    case -1001:
      printf("CL_PLATFORM_NOT_FOUND_KHR ");
      break;

    case -1094:
      printf("CL_INVALID_ACCELERATOR_INTEL ");
      break;
    case -1095:
      printf("CL_INVALID_ACCELERATOR_TYPE_INTEL ");
      break;
    case -1096:
      printf("CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL ");
      break;
    case -1097:
      printf("CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL ");
      break;
    default:
      printf("UNRECOGNIZED ERROR CODE (%d)", error);
  } 

}

void _checkError(const char *file, int line, const char *func, cl_int err, const char *msg, ...){

  if(err != CL_SUCCESS){
    printf("ERROR: ");
    printError(err);
    printf("\nError Location: %s:%d:%s\n", file, line, func);

    // custom message 
    va_list vl;
    va_start(vl, msg);
    vprintf(msg, vl);
    printf("\n");
    va_end(vl);

    queue_cleanup();
    fpga_final();
    exit(err);
  }
}

/**
 * \brief  converts a given null-terminated string to lowercase and stores in q
 * \param  p : null-terminated string
 * \param  q : string in lowercase with (strlen(p)+1) length 
 */
static void tolowercase(const char *p, char *q){
  char a;
  for(int i=0; i < strlen(p);i++){
    a = tolower(p[i]);
    q[i] = a;
  }
  q[strlen(p)] = '\0';
}