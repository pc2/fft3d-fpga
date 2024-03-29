// Author: Arjun Ramaswami

/**
 * @file fftfpga.h
 * @brief Header file that provides APIs for OpenCL Host code
 */

#ifndef FFTFPGA_H
#define FFTFPGA_H

#include <stdbool.h>

/**
 * Single Precision Complex Floating Point Data Structure
 */
typedef struct {
  float x; /**< real value */
  float y; /**< imaginary value */
} float2;

/**
 * Double Precision Complex Floating Point Data Structure
 */
typedef struct {
  double x; /**< real value */
  double y; /**< imaginary value */
} double2;

/**
 * Record time in milliseconds of different FPGA runtime stages
 */
typedef struct fpga_timing {
  double pcie_read_t;     /**< Time to read from DDR to host using PCIe bus  */ 
  double pcie_write_t;    /**< Time to write from DDR to host using PCIe bus */ 
  double exec_t;          /**< Kernel execution time */
  double svm_copyin_t;    /**< Time to copy in data to SVM */
  double svm_copyout_t;   /**< Time to copy data out of SVM */ 
  bool valid;             /**< Represents true signifying valid execution */
} fpga_t;

#ifdef __cplusplus
extern "C" {
#endif
/** 
 * @brief Initialize FPGA
 * @param platform_name: name of the OpenCL platform
 * @param path         : path to binary
 * @param use_svm      : 1 if true 0 otherwise
 * @return 0 if successful 
          -1 Path to binary missing
          -2 Unable to find platform passed as argument
          -3 Unable to find devices for given OpenCL platform
          -4 Failed to create program, file not found in path
          -5 Device does not support required SVM
 */
extern int fpga_initialize(const char *platform_name, const char *path, const bool use_svm);

/** 
 * @brief Release FPGA Resources
 */
extern void fpga_final();

/** 
 * @brief Allocate memory of double precision complex floating points
 * @param sz  : size_t - size to allocate
 * @return void ptr or NULL
 */
extern void* fftfpga_complex_malloc(const size_t sz);

/** 
 * @brief Allocate memory of single precision complex floating points
 * @param sz  : size_t : size to allocate
 * @return void ptr or NULL
 */
extern void* fftfpgaf_complex_malloc(const size_t sz);

/**
 * @brief  compute an out-of-place double precision complex 1D-FFT on the FPGA
 * @param  N    : integer pointer to size of FFT3d  
 * @param  inp  : double2 pointer to input data of size N
 * @param  out  : double2 pointer to output data of size N
 * @param  inv  : int toggle to activate backward FFT
 * @param  iter : number of iterations of the N point FFT
 * @return fpga_t : time taken in milliseconds for data transfers and execution
 */
extern fpga_t fftfpga_c2c_1d(const unsigned N, const double2 *inp, double2 *out, const bool inv, const unsigned iter);

/**
 * @brief  compute an out-of-place single precision complex 1D-FFT on the FPGA
 * @param  N    : integer pointer to size of FFT3d  
 * @param  inp  : float2 pointer to input data of size N
 * @param  out  : float2 pointer to output data of size N
 * @param  inv  : int toggle to activate backward FFT
 * @param  iter : number of iterations of the N point FFT
 * @return fpga_t : time taken in milliseconds for data transfers and execution
 */
extern fpga_t fftfpgaf_c2c_1d(const unsigned N, const float2 *inp, float2 *out, const bool inv, const unsigned iter);

/**
 * @brief  compute an out-of-place single precision complex 1D-FFT on the FPGA
 * @param  N    : integer pointer to size of FFT3d  
 * @param  inp  : float2 pointer to input data of size N
 * @param  out  : float2 pointer to output data of size N
 * @param  inv  : int toggle to activate backward FFT
 * @param  iter : number of iterations of the N point FFT
 * @return fpga_t : time taken in milliseconds for data transfers and execution
 */
extern fpga_t fftfpgaf_c2c_1d_svm(const unsigned N, const float2 *inp, float2 *out, const bool inv, const unsigned batch);

/**
 * @brief  compute an out-of-place single precision complex 2D-FFT using the BRAM of the FPGA
 * @param  N    : integer pointer to size of FFT2d  
 * @param  inp  : float2 pointer to input data of size [N * N]
 * @param  out  : float2 pointer to output data of size [N * N]
 * @param  inv  : int toggle to activate backward FFT
 * @param  interleaving : enable interleaved global memory buffers
 * @param  how_many : number of 2D FFTs to computer, default 1
 * @return fpga_t : time taken in milliseconds for data transfers and execution
 */
extern fpga_t fftfpgaf_c2c_2d_bram(const unsigned N, const float2 *inp, float2 *out, const bool inv, const bool interleaving, const unsigned how_many);

/**
 * @brief  compute an out-of-place single precision complex 2DFFT using the BRAM of the FPGA and Shared Virtual Memory for Host to Device Communication
 * @param  N    : integer pointer to size of FFT2d  
 * @param  inp  : float2 pointer to input data of size [N * N]
 * @param  out  : float2 pointer to output data of size [N * N]
 * @param  inv  : int toggle to activate backward FFT
 * @param  how_many : number of 2D FFTs to computer, default 1
 * @return fpga_t : time taken in milliseconds for data transfers and execution
 */
extern fpga_t fftfpgaf_c2c_2d_bram_svm(const unsigned N, const float2 *inp, float2 *out, const bool inv, const unsigned how_many);

/**
 * @brief  compute an out-of-place single precision complex 2D-FFT using the DDR of the FPGA
 * @param  N    : integer pointer to size of FFT2d  
 * @param  inp  : float2 pointer to input data of size [N * N]
 * @param  out  : float2 pointer to output data of size [N * N]
 * @param  inv  : int toggle to activate backward FFT
 * @return fpga_t : time taken in milliseconds for data transfers and execution
 */
extern fpga_t fftfpgaf_c2c_2d_ddr(const unsigned N, const float2 *inp, float2 *out, const bool inv);

/**
 * @brief  compute an out-of-place single precision complex 3D-FFT using the BRAM of the FPGA
 * @param  N    : integer pointer addressing the size of FFT3d  
 * @param  inp  : float2 pointer to input data of size [N * N * N]
 * @param  out  : float2 pointer to output data of size [N * N * N]
 * @param  inv  : int toggle to activate backward FFT
 * @param  interleaving : enable burst interleaved global memory buffers
 * @return fpga_t : time taken in milliseconds for data transfers and execution
 */
extern fpga_t fftfpgaf_c2c_3d_bram(const unsigned N, const float2 *inp, float2 *out, const bool inv, const bool interleaving);

/**
 * @brief  compute an out-of-place single precision complex 3D-FFT using the DDR of the FPGA
 * @param  N    : integer pointer addressing the size of FFT3d  
 * @param  inp  : float2 pointer to input data of size [N * N * N]
 * @param  out  : float2 pointer to output data of size [N * N * N]
 * @param  inv  : int toggle to activate backward FFT
 * @return fpga_t : time taken in milliseconds for data transfers and execution
 */
extern fpga_t fftfpgaf_c2c_3d_ddr(const unsigned N, const float2 *inp, float2 *out, const bool inv);

extern fpga_t fftfpgaf_c2c_3d_ddr_batch(const unsigned N, const float2 *inp, float2 *out, const bool inv, const bool interleaving, const unsigned how_many);

/**
 * @brief  compute an out-of-place single precision complex 3D-FFT using the DDR of the FPGA and Shared Virtual Memory for Host to Device Communication
 * @param  N    : unsigned integer size of FFT3d  
 * @param  inp  : float2 pointer to input data of size [N * N * N]
 * @param  out  : float2 pointer to output data of size [N * N * N]
 * @param  inv  : toggle to activate backward FFT
 * @param  interleaving  : toggle interleaved device memory
 * @return fpga_t : time taken in milliseconds for data transfers and execution
 */
extern fpga_t fftfpgaf_c2c_3d_ddr_svm(const unsigned N, const float2 *inp, float2 *out, const bool inv, const bool interleaving);

/**
 * @brief  compute an out-of-place single precision complex 3D-FFT using the DDR of the FPGA and Shared Virtual Memory for Host to Device Communication
 * @param  N    : unsigned integer size of FFT3d  
 * @param  inp  : float2 pointer to input data of size [N * N * N]
 * @param  out  : float2 pointer to output data of size [N * N * N]
 * @param  inv  : toggle to activate backward FFT
 * @param  interleaving  : toggle interleaved device memory
 * @return fpga_t : time taken in milliseconds for data transfers and execution
 */
extern fpga_t fftfpgaf_c2c_3d_ddr_svm_batch(const unsigned N, const float2 *inp, float2 *out, const bool inv, const unsigned how_many);

#ifdef __cplusplus
}
#endif

#endif
