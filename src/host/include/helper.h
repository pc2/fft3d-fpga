/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

#ifndef HELPER_H
#define HELPER_H

int fftf_create_data(float2 *inp, int N);

int fft_create_data(double2 *inp, int N);

/*
double compute_sp_fftw(float2 *fftw_data, int N[3], int inverse);

double compute_dp_fftw(double2 *fftw_data, int N[3], int inverse);

void verify_sp_fft(float2 *fft_data, float2 *fftw_data, int N[3]);

void verify_dp_fft(double2 *fft_data, double2 *fftw_data, int N[3]);
*/

double getTimeinMilliSec();

/*
unsigned coord(unsigned N[3], unsigned i, unsigned j, unsigned k);

void compute_metrics( double fpga_runtime, double fpga_computetime, double fftw_runtime, unsigned iter, int N[3]);
*/

#endif // HELPER_H
