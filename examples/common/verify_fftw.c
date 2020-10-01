// Author: Arjun Ramaswami
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include "fftfpga/fftfpga.h"

#ifdef USE_FFTW
#include <fftw3.h>

/**
 * \brief Verify FFT computed in FPGA with FFTW 
 * \param fpga_out: pointer to FPGA computation for sp complex data 
 * \param fftw_data: pointer to FFT sized allocation of sp complex data for fftw cpu computation
 * \param N: number of points per dimension of FFT3d
 * \param dim: number of dimensions of points
 * \param inverse: true if backward FFT
 * \param how_many: default is 1 
 * \return true if verification passed
 */
bool verify_fftwf(float2 *fpgaout, float2 *verify, int N, int dim, bool inverse, int how_many){

  // Copy inp data to verify using FFTW
  // requires allocating data specifically for FFTW computation
  size_t num_pts = how_many * pow(N, dim);
  fftwf_complex *fftw_data = fftwf_alloc_complex(num_pts);

  for(size_t i = 0; i < num_pts; i++){
    fftw_data[i][0] = verify[i].x;
    fftw_data[i][1] = verify[i].y;
  }

  int *n = (int*)calloc(N * dim , sizeof(int));
  for(size_t i = 0; i < dim; i++){
    n[i] = N;
  }

  // Compute 3d FFT using FFTW
  // Create Plan using simple heuristic and in place FFT
  fftwf_plan plan;
  //const int n[] = {N, N, N};
  //int idist = N*N*N, odist = N*N*N;
  int idist = pow(N, dim);
  int odist = pow(N, dim);
  int istride = 1, ostride = 1; // contiguous in memory

  if(inverse){
    plan = fftwf_plan_many_dft(dim, n, how_many, &fftw_data[0], NULL, istride, idist, fftw_data, NULL, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);
    //plan = fftwf_plan_dft_3d( N, N, N, &fftw_data[0], &fftw_data[0], FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  else{
    plan = fftwf_plan_many_dft(dim, n, how_many, &fftw_data[0], NULL, istride, idist, fftw_data, NULL, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);
    //plan = fftwf_plan_dft_3d( N, N, N, &fftw_data[0], &fftw_data[0], FFTW_FORWARD, FFTW_ESTIMATE);
  }

  // Execute in place FFTW based on plan created
  fftwf_execute(plan);

  // verify by calculating signal-to-noise ratio (SNR)
  float mag_sum = 0, noise_sum = 0, magnitude, noise;

  for (size_t i = 0; i < num_pts; i++) {

    magnitude = fftw_data[i][0] * fftw_data[i][0] + \
                      fftw_data[i][1] * fftw_data[i][1];
    noise = (fftw_data[i][0] - fpgaout[i].x) \
        * (fftw_data[i][0] - fpgaout[i].x) + 
        (fftw_data[i][1] - fpgaout[i].y) * (fftw_data[i][1] - fpgaout[i].y);

    mag_sum += magnitude;
    noise_sum += noise;
#ifndef NDEBUG
    printf("%zu : fpga - (%e %e) cpu - (%e %e)\n", i, fpgaout[i].x, fpgaout[i].y, fftw_data[i][0], fftw_data[i][1]);
#endif            
  }

  // Calculate SNR
  float db = 10 * log(mag_sum / noise_sum) / log(10.0);
  
  // Free FFTW data
  fftwf_free(fftw_data);

  // destroy plan
  fftwf_destroy_plan(plan);

  // if SNR greater than 120, verification passes
  if(db > 120){
    return true;
  }
  else{
    printf("\tSignal to noise ratio on output sample: %f --> %s\n\n", db, "FAILED");
    return false;
  }
}

#endif // USE_FFTW