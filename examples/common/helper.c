// Author: Arjun Ramaswami

#define _POSIX_C_SOURCE 199309L  
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "helper.h"
#include <time.h>
#include <sys/time.h>
#include <math.h>
#define _USE_MATH_DEFINES

/**
 * \brief  create random single precision complex floating point values  
 * \param  inp : pointer to float2 data of size N 
 * \param  N   : number of points in the array
 * \return true if successful
 */
bool fftf_create_data(float2 *inp, unsigned num_pts){

  if(inp == NULL || num_pts <= 0){
    return false;
  }

  for(size_t i = 0; i < num_pts; i++){
    inp[i].x = (float)((float)rand() / (float)RAND_MAX);
    inp[i].y = (float)((float)rand() / (float)RAND_MAX);
  }

  return true;
}

/**
 * \brief  create random double precision complex floating point values  
 * \param  inp : pointer to double2 data of size inp_sz 
 * \param  inp_sz : number of points in the array
 * \return true if successful
 */
bool fft_create_data(double2 *inp, unsigned num_pts){

  if(inp == NULL || num_pts <= 0){
    return false;
  }

  for(size_t i = 0; i < num_pts; i++){
    inp[i].x = (double)((double)rand() / (double)RAND_MAX);
    inp[i].y = (double)((double)rand() / (double)RAND_MAX);
  }

  return true;
}

/**
 * \brief  print configuration chosen to execute on FPGA
 * \param  N: fft size
 * \param  dim: number of dimensions of size
 * \param  iter: number of iterations of each transformation (if BATCH mode)
 * \param  inv: true for backward transform
 * \param  sp: true for single precision floating point transformation
 * \param  use_bram: true if transpose uses BRAM, not DDR (valid for 2d and 3d FFT)
 * \param  interleaving: true if data should be interleaved amongst the banks in DDR memory
 */
void print_config(int N, int dim, int iter, bool inv, bool sp, int batch, bool use_bram, bool interleaving){
  printf("\n------------------------------------------\n");
  printf("FFT Configuration: \n");
  printf("--------------------------------------------\n");
  printf("Type               = Complex to Complex\n");
  printf("Points             = %d%s \n", N, dim == 1 ? "" : dim == 2 ? "^2" : "^3");
  printf("Precision          = %s \n", sp ? "Single": "Double");
  printf("Direction          = %s \n", inv ? "Backward":"Forward");
  printf("Placement          = In Place    \n");
  printf("Batch              = %d \n", batch);
  printf("Iterations         = %d \n", iter);
  printf("Transpose          = %s \n", use_bram ? "BRAM":"DDR");
  printf("Interleaving       = %s \n", interleaving ? "Yes":"No");
  printf("--------------------------------------------\n\n");
}

/**
 * \brief  print time taken for fpga and fftw runs to a file
 * \param  total_api_time: time taken to call iter times the host code
 * \param  timing: kernel execution and pcie transfer timing 
 * \param  N: fft size
 * \param  dim: number of dimensions of size
 * \param  iter: number of iterations of each transformation (if BATCH mode)
 * \param  inv: true if backward transform
 * \param  single precision floating point transformation
 */
void display_measures(double total_api_time, double pcie_rd, double pcie_wr, double exec_t, double hw_pcie_rd, double hw_pcie_wr, double hw_exec, int N, int dim, int iter, int batch, bool inv, bool sp){

  double avg_api_time = 0.0;

  if (total_api_time != 0.0){
    avg_api_time = total_api_time / iter;
  }

  double pcie_read = pcie_rd / iter;
  double pcie_write = pcie_wr / iter;
  double exec = exec_t / iter;

  double hw_pcie_read = hw_pcie_rd / iter;
  double hw_pcie_write = hw_pcie_wr / iter;
  double hw_execution = hw_exec / iter;

  double gpoints_per_sec = (batch * pow(N, dim))  / (hw_execution * 1e-3 * 1024 * 1024 * 1024);
  double gBytes_per_sec = 0.0;

  if(sp){
    gBytes_per_sec =  gpoints_per_sec * 8; // bytes
  }
  else{
    gBytes_per_sec *=  gpoints_per_sec * 16;
  }

  double gflops = batch * dim * 5 * pow(N, dim) * (log((double)N)/log((double)2))/(exec * 1e-3 * 1024*1024*1024); 

  printf("\n\n------------------------------------------\n");
  printf("Measurements \n");
  printf("--------------------------------------------\n");
  printf("Points               = %d%s \n", N, dim == 1 ? "" : dim == 2 ? "^2" : "^3");
  printf("Precision            = %s\n",  sp ? "Single": "Double");
  printf("Direction            = %s\n", inv ? "Backward":"Forward");
  printf("Iterations           = %d\n", iter);
  printf("Batch                = %d\n", batch);

  printf("%s", iter>1 ? "Average Measurements of iterations\n":"");
  printf("PCIe Write           = %.4lfms\n", pcie_write);
  printf("Kernel Execution     = %.4lfms\n", exec);
  printf("PCIe Read            = %.4lfms\n", pcie_read);
  printf("Total                = %.4lfms\n", pcie_read + exec + pcie_write);
  printf("HW PCIe Write        = %.4lfms\n", hw_pcie_write);
  printf("HW Kernel Execution  = %.4lfms\n", hw_execution);
  printf("HW PCIe Read         = %.4lfms\n", hw_pcie_read);
  printf("Hw Total             = %.4lfms\n", hw_pcie_write + hw_execution + hw_pcie_read);
  printf("Throughput           = %.4lfGFLOPS/s | %.4lf GB/s\n", gflops, gBytes_per_sec);
  printf("API runtime          = %.4lfms\n", avg_api_time);

}

/**
 * \brief  compute walltime in milliseconds
 * \return time in milliseconds
 */
double getTimeinMilliseconds(){
   struct timespec a;
   if(clock_gettime(CLOCK_MONOTONIC, &a) != 0){
     fprintf(stderr, "Error in getting wall clock time \n");
     exit(EXIT_FAILURE);
   }
   return (double)(a.tv_nsec) * 1.0e-6 + (double)(a.tv_sec) * 1.0E3;
}