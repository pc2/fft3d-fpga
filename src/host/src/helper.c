// Author: Arjun Ramaswami

#define _POSIX_C_SOURCE 199309L  
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#define _USE_MATH_DEFINES

#include "../include/fftfpga.h"
#include "../include/helper.h"

#ifdef USE_FFTW
  #include <fftw3.h>
#endif

/**
 * \brief  create random single precision complex floating point values  
 * \param  inp : pointer to float2 data of size N 
 * \param  N   : number of points in the array
 * \return 0 if successful 1 if not
 */
int fftf_create_data(float2 *inp, int N){

  if(inp == NULL || N <= 0 || N > 1024){
    return 1;
  }

  for(int i = 0; i < N; i++){
    inp[i].x = (float)((float)rand() / (float)RAND_MAX);
    inp[i].y = (float)((float)rand() / (float)RAND_MAX);
  }

#ifdef DEBUG          
    FILE *fptr = fopen("input_data.txt", "w"); 
    for(int i = 0; i < N; i++){
      if (fptr != NULL){
        fprintf(fptr, "%d : fft[%d] = (%f, %f) \n", i, i, inp[i].x, inp[i].y);
      }
    }
    fclose(fptr); 
#endif

  return 0;
}

/**
 * \brief  create random double precision complex floating point values  
 * \param  inp : pointer to double2 data of size N 
 * \param  N   : number of points in the array
 * \return 0 if successful 1 if not
 */
int fft_create_data(double2 *inp, int N){

  if(inp == NULL || N <= 0 || N > 1024){
    return 1;
  }

  for(int i = 0; i < N; i++){
    inp[i].x = (double)((double)rand() / (double)RAND_MAX);
    inp[i].y = (double)((double)rand() / (double)RAND_MAX);
  }

#ifdef DEBUG          
    FILE *fptr = fopen("input_data.txt", "w"); 
    for(int i = 0; i < N; i++){
      if (fptr != NULL){
        fprintf(fptr, "%d : fft[%d] = (%lf, %lf) \n", i, i, inp[i].x, inp[i].y);
      }
    }
    fclose(fptr); 
#endif

  return 0;
}

/**
 * \brief  compute walltime in milliseconds
 * \return time in milliseconds
 */
double getTimeinMilliSec(){
   struct timespec a;
   if(clock_gettime(CLOCK_MONOTONIC, &a) != 0){
     fprintf(stderr, "Error in getting wall clock time \n");
     exit(EXIT_FAILURE);
   }
   return (double)(a.tv_nsec) * 1.0e-6 + (double)(a.tv_sec) * 1.0E3;
}

/******************************************************************************
 * \brief  compute the offset in the matrix based on the indices of dim given
 * \param  i, j, k : indices of different dimensions used to find the 
 *         coordinate in the matrix 
 * \param  N : fft size
 * \retval linear offset in the flattened 3d matrix
 *****************************************************************************/
/*
unsigned coord(unsigned N[3], unsigned i, unsigned j, unsigned k) {
  // TODO : works only for uniform dims
  return i * N[0] * N[1] + j * N[2] + k;
}
*/

/******************************************************************************
 * \brief  verify computed fft3d with FFTW fft3d
 * \param  fft_data  : pointer to fft3d sized allocation of sp complex data for fpga
 * \param  fftw_data : pointer to fft3d sized allocation of sp complex data for fftw cpu computation
 * \param  N : 3 element integer array containing the size of FFT3d  
 *****************************************************************************/
/*
void verify_sp_fft(float2 *fft_data, fftwf_complex *fftw_data, int N[3]){
  unsigned where, i, j, k;
  float mag_sum = 0, noise_sum = 0, magnitude, noise;

  for (i = 0; i < N[0]; i++) {
    for (j = 0; j < N[1]; j++) {
      for ( k = 0; k < N[2]; k++) {
        where = coord(N, i, j, k);
        float magnitude = fftw_data[where][0] * fftw_data[where][0] + \
                          fftw_data[where][1] * fftw_data[where][1];
        float noise = (fftw_data[where][0] - fft_data[where].x) \
            * (fftw_data[where][0] - fft_data[where].x) + 
            (fftw_data[where][1] - fft_data[where].y) * (fftw_data[where][1] - fft_data[where].y);

        mag_sum += magnitude;
        noise_sum += noise;
#ifdef DEBUG
        printf("%d : fpga - (%e %e) cpu - (%e %e)\n", where, fft_data[where].x, fft_data[where].y, fftw_data[where][0], fftw_data[where][1]);
#endif            
      }
    }
  }

  float db = 10 * log(mag_sum / noise_sum) / log(10.0);
  printf("-> Signal to noise ratio on output sample: %f --> %s\n\n", db, db > 120 ? "PASSED" : "FAILED");
}
*/
/******************************************************************************
 * \brief  verify computed fft3d with FFTW fft3d
 * \param  fft_data  : pointer to fft3d sized allocation of dp complex data for fpga
 * \param  fftw_data : pointer to fft3d sized allocation of dp complex data for fftw cpu computation
 * \param  N : 3 element integer array containing the size of FFT3d  
 *****************************************************************************/
#ifdef USE_FFTW
/*
int verify_dp_fft(double2 *fft_data, fftw_complex *fftw_data, int N){
  double mag_sum = 0, noise_sum = 0, magnitude, noise;

  for (size_t i = 0; i < N; i++) {
    double magnitude = fftw_data[i][0] * fftw_data[i][0] + \
                      fftw_data[i][1] * fftw_data[i][1];
    double noise = (fftw_data[i][0] - fft_data[i].x) \
        * (fftw_data[i][0] - fft_data[i].x) + 
        (fftw_data[i][1] - fft_data[i].y) * (fftw_data[i][1] - fft_data[i].y);

    mag_sum += magnitude;
    noise_sum += noise;
#ifdef DEBUG
    printf("%d : fpga - (%e %e)  cpu - (%e %e)\n", i, fft_data[i].x, fft_data[i].y, fftw_data[i][0], fftw_data[i][1]);
#endif
  }

  double db = 10 * log(mag_sum / noise_sum) / log(10.0);
  if( db > 120)
    return 0;
  else
    return 1;
  
  //printf("-> Signal to noise ratio on output sample: %lf --> %s\n\n", db, db > 120 ? "PASSED" : "FAILED");
}
*/
#endif

/******************************************************************************
 * \brief  print time taken for fpga and fftw runs to a file
 * \param  fftw_time, fpga_time: double
 * \param  iter - number of iterations of each
 * \param  fname - filename given through cmd line arg
 * \param  N - fft size
 *****************************************************************************/
/*
void compute_metrics( double fpga_runtime, double fpga_computetime, double fftw_runtime, unsigned iter, int N[3]){
  char filename[] = "../outputfiles/output.csv";
  printf("Printing metrics to %s\n", filename);

  FILE *fp = fopen(filename,"r");
  if(fp == NULL){
    fp = fopen(filename,"w");
    if(fp == NULL){
      printf("Unable to create file - %s\n", filename);
      exit(1);
    }
    fprintf(fp,"device, N, runtime, computetime, throughput\n");
  }
  else{
    fp = fopen(filename,"a");
  }


  printf("\nNumber of runs: %d\n\n", iter);
  printf("\tFFT Size\tRuntime(ms)\tComputetime(ms)\tThroughput(GFLOPS/sec)\t\n");
  printf("fpga:");
  fprintf(fp, "fpga,");

  if(fpga_runtime != 0.0 || fpga_computetime != 0.0){
    fpga_runtime = fpga_runtime / iter;
    fpga_computetime = fpga_computetime / iter;
    double gpoints_per_sec = ( N[0] * N[1] * N[2] / (fpga_computetime * 1e-3)) * 1e-9;
    double gflops = 3 * 5 * N[0] * N[1] * N[2] * (log((double)N[0])/log((double)2))/(fpga_computetime * 1e-3 * 1E9);
    printf("\t  %d³ \t\t %.4f \t %.4f \t  %.4f \n", N[0], fpga_runtime, fpga_computetime, gflops);
    fprintf(fp, "%d,%.4f,%.4f,%.4f\n", N[0], fpga_runtime, fpga_computetime, gflops);
  }
  else{
    printf("ERROR in FFT3d \n");
  }

  printf("fftw:"); 
  fprintf(fp, "fftw,"); 
  if(fftw_runtime != 0.0){
    fftw_runtime = fftw_runtime / iter;
    double gpoints_per_sec = ( N[0] * N[1] * N[2] / (fftw_runtime * 1E-3)) * 1E-9;
    double gflops = 3 * 5 * N[0] * N[1] * N[2]* (log((double)N[0])/log((double)2))/(fftw_runtime * 1E-3 * 1E9);
    printf("\t  %d³ \t\t\t\t %.4f \t  %.4f \t\n", N[0], fftw_runtime, gflops);
    fprintf(fp, "%d,%.4f,%.4f\n", N[0], fftw_runtime, gflops);
  }
  else{
    printf("ERROR in FFT3d\n");
  }

  fclose(fp);
}
*/