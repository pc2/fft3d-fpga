/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

#ifndef HELPER_H
#define HELPER_H

void get_sp_input_data(float2 *fft_data, fftwf_complex* fftw_data, unsigned N[3], char* fname);

void get_dp_input_data(double2 *fft_data, fftw_complex* fftw_data, unsigned N[3], char* fname);

double compute_sp_fftw(fftwf_complex *fftw_data, int N[3], int inverse);

double compute_dp_fftw(fftw_complex *fftw_data, int N[3], int inverse);

void verify_sp_fft(float2 *fft_data, fftwf_complex *fftw_data, int N[3]);

void verify_dp_fft(double2 *fft_data, fftw_complex *fftw_data, int N[3]);

double getTimeinMilliSec();

unsigned coord(unsigned N[3], unsigned i, unsigned j, unsigned k);

void compute_metrics( double fpga_runtime, double fpga_computetime, double fftw_runtime, unsigned iter, int N[3]);

#endif // HELPER_H
