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

  if(inp == NULL || N <= 0){
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