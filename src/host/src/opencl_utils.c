// Author: Arjun Ramaswami

#include <stdio.h>  
#include <stdlib.h> // size_t, EXIT_FAILURE, NULL, EXIT_SUCCESS
#include <string.h> // strlen, strstr
#include <stdarg.h> // valist, va_start, va_end
#include <unistd.h> // access in fileExists()
#include <ctype.h>  // tolower

#include "CL/opencl.h"
#include "../include/opencl_utils.h"
#include "../include/fftfpga.h"

// function prototype
static void tolowercase(const char *p, char *q);
static size_t loadBinary(const char *binary_path, char **buf);
//void fpga_final();
//void queue_cleanup();

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
 * \retval 0 if successful and 1 otherwise
 */
static int fileExists(const char* filename){
  if( access( filename, R_OK ) != -1 ) {
    return 0;
  } else {
    return 1;
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

  if (fileExists(path)){
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
  const unsigned OPENCL_ALIGNMENT = 64;
  void *memptr = NULL;
  int ret = posix_memalign(&memptr, OPENCL_ALIGNMENT, size);
  if (ret != 0){
    return NULL;
  }
  return memptr;
}

static void printError(cl_int error) {

  switch(error)
  {
    case CL_INVALID_PLATFORM:
      fprintf(stderr, "CL_PLATFORM NOT FOUND OR INVALID ");
      break;
    case CL_INVALID_DEVICE:
      fprintf(stderr, "CL_DEVICE NOT FOUND OR INVALID OR DOESN'T MATCH THE PLATFORM ");
      break;
    case CL_INVALID_CONTEXT:
      fprintf(stderr, "CL_CONTEXT INVALID ");
      break;
    case CL_OUT_OF_HOST_MEMORY:
      fprintf(stderr, "FAILURE TO ALLOCATE RESOURCES BY OPENCL");
      break;
    case CL_DEVICE_NOT_AVAILABLE:
      fprintf(stderr, "CL_DEVICE NOT AVAILABLE ALTHOUGH FOUND");
      break;
    case CL_INVALID_QUEUE_PROPERTIES:
      fprintf(stderr, "CL_QUEUE PROPERTIES INVALID");
      break;
    case CL_INVALID_PROGRAM:
      fprintf(stderr, "CL_PROGRAM INVALID");
      break;
    case CL_INVALID_BINARY:
      fprintf(stderr, "CL_BINARY INVALID");
      break;
    case CL_INVALID_KERNEL_NAME:
      fprintf(stderr, "CL_KERNEL_NAME INVALID");
      break;
    case CL_INVALID_KERNEL_DEFINITION:
      fprintf(stderr, "CL_KERNEL_DEFN INVALID");
      break;
    case CL_INVALID_VALUE:
      fprintf(stderr, "CL_VALUE INVALID");
      break;
    case CL_INVALID_BUFFER_SIZE:
      fprintf(stderr, "CL_BUFFER_SIZE INVALID");
      break;
    case CL_INVALID_HOST_PTR:
      fprintf(stderr, "CL_HOST_PTR INVALID");
      break;
    case CL_INVALID_COMMAND_QUEUE:
      fprintf(stderr, "CL_COMMAND_QUEUE INVALID");
      break;
    case CL_INVALID_MEM_OBJECT:
      fprintf(stderr, "CL_MEM_OBJECT INVALID");
      break;
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      fprintf(stderr, "CL_MEM_OBJECT_ALLOCATION INVALID");
      break;
    case CL_INVALID_ARG_INDEX:
      fprintf(stderr, "CL_ARG_INDEX INVALID");
      break;
    case CL_INVALID_ARG_VALUE:
      fprintf(stderr, "CL_ARG_VALUE INVALID");
      break;
    case CL_INVALID_ARG_SIZE:
      fprintf(stderr, "CL_ARG_SIZE INVALID");
      break;
    case CL_INVALID_PROGRAM_EXECUTABLE:
      fprintf(stderr, "CL_PROGRAM_EXEC INVALID");
      break;
    case CL_INVALID_KERNEL:
      fprintf(stderr, "CL_KERNEL INVALID");
      break;
    case CL_INVALID_KERNEL_ARGS:
      fprintf(stderr, "CL_KERNEL_ARG INVALID");
      break;
    case CL_INVALID_WORK_GROUP_SIZE:
      fprintf(stderr, "CL_WORK_GROUP_SIZE INVALID");
      break;

    default:
      fprintf(stderr, "UNKNOWN ERROR %d\n", error);
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