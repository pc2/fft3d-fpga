// Author: Arjun Ramaswami
#include <stdio.h>
#include <math.h>
#include "fftfpga/fftfpga.h"

#ifdef USE_FFTW
#include <fftw3.h>
/**
 * \brief verify FPGA fft3d with FFTW fft3d
 * \param fpga_out: pointer to fpga computation of fft3d for sp complex data 
 * \param fftw_data: pointer to fft3d sized allocation of sp complex data for fftw cpu computation
 * \param N: number of points per dimension of FFT3d
 * \param inverse: 1 if inverse
 * \return 0 if verification passed, 1 failed
 */
int verify_sp_fft3d_fftw(float2 *fpgaout, float2 *verify, int N, int inverse){

  // Copy inp data to verify using FFTW
  // requires allocating data specifically for FFTW computation
  size_t num_pts = N * N * N;
  fftwf_complex *fftw_data = fftwf_alloc_complex(num_pts);

  for(size_t i = 0; i < num_pts; i++){
    fftw_data[i][0] = verify[i].x;
    fftw_data[i][1] = verify[i].y;
  }

  // Compute 3d FFT using FFTW
  // Create Plan using simple heuristic and in place FFT
  fftwf_plan plan;

  if(inverse){
    plan = fftwf_plan_dft_3d( N, N, N, &fftw_data[0], &fftw_data[0], FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  else{
    plan = fftwf_plan_dft_3d( N, N, N, &fftw_data[0], &fftw_data[0], FFTW_FORWARD, FFTW_ESTIMATE);
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
#ifdef DEBUG
    printf("%d : fpga - (%e %e) cpu - (%e %e)\n", where, fpgaout[i].x, fpgaout[i].y, fftw_data[i][0], fftw_data[i][1]);
#endif            
  }

  // Calculate SNR
  float db = 10 * log(mag_sum / noise_sum) / log(10.0);
  
  // Free FFTW data
  fftwf_free(fftw_data);

  // destroy plan
  fftwf_destroy_plan(plan);

  // if SNR greater than 120, verification passes
  printf("\tSignal to noise ratio on output sample: %f --> %s\n\n", db, db > 120 ? "PASSED" : "FAILED");

  if(db > 120){
    return 1;
  }
  else{
    return 0;
  }

}

/**
 * \brief verify FPGA fft2d with FFTW fft2d
 * \param fpga_out: pointer to fpga computation of fft2d for sp complex data 
 * \param fftw_data: pointer to fft2d sized allocation of sp complex data for fftw cpu computation
 * \param N: number of points per dimension of FFT2d
 * \param inverse: 1 if inverse
 * \return 0 if verification passed, 1 failed
 */
int verify_sp_fft2d_fftw(float2 *fpgaout, float2 *verify, int N, int inverse){

  // Copy inp data to verify using FFTW
  // requires allocating data specifically for FFTW computation
  size_t num_pts = N * N;
  fftwf_complex *fftw_data = fftwf_alloc_complex(num_pts);

  for(size_t i = 0; i < num_pts; i++){
    fftw_data[i][0] = verify[i].x;
    fftw_data[i][1] = verify[i].y;
  }

  // Compute 3d FFT using FFTW
  // Create Plan using simple heuristic and in place FFT
  fftwf_plan plan;

  if(inverse){
    plan = fftwf_plan_dft_2d( N, N, &fftw_data[0], &fftw_data[0], FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  else{
    plan = fftwf_plan_dft_2d( N, N, &fftw_data[0], &fftw_data[0], FFTW_FORWARD, FFTW_ESTIMATE);
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
#ifdef DEBUG
    printf("%d : fpga - (%e %e) cpu - (%e %e)\n", where, fpgaout[i].x, fpgaout[i].y, fftw_data[i][0], fftw_data[i][1]);
#endif            
  }

  // Calculate SNR
  float db = 10 * log(mag_sum / noise_sum) / log(10.0);
  
  // Free FFTW data
  fftwf_free(fftw_data);

  // destroy plan
  fftwf_destroy_plan(plan);

  // if SNR greater than 120, verification passes
  printf("\tSignal to noise ratio on output sample: %f --> %s\n\n", db, db > 120 ? "PASSED" : "FAILED");

  if(db > 120){
    return 1;
  }
  else{
    return 0;
  }

}

#endif // USE_FFTW