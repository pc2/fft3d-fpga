/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

#ifndef OPENCL_UTILS_H
#define OPENCL_UTILS_H

extern void queue_cleanup();
extern void fpga_final();

// Search for a platform that contains the search string
// Returns platform id if found
// Return NULL if none found
cl_platform_id findPlatform(const char *platform_name);

// Search for a device based on the platform
// Return array of device ids
cl_device_id* getDevices(cl_platform_id pid, cl_device_type device_type, cl_uint *num_devices);

// OpenCL program created for all the devices of the context with the same binary
cl_program getProgramWithBinary(cl_context context, cl_device_id *devices, cl_uint num_devices, const char *data_path);

void* alignedMalloc(size_t size);

void _checkError(const char *file, int line, const char *func, cl_int err, const char *msg, ...);

#define checkError(status, ...) _checkError(__FILE__, __LINE__, __FUNCTION__, status, __VA_ARGS__)

#endif // OPENCL_UTILS_H