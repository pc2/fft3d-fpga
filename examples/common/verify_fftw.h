// Author: Arjun Ramaswami

#ifndef FFT3D_FFTW_H
#define FFT3D_FFTW_H

int verify_sp_fft2d_fftw(float2 *fpgaout, float2 *verify, int N, int inverse);

int verify_sp_fft3d_fftw(float2 *fpgaout, float2 *verify, int N, int inverse, int how_many);

#endif // FFT3D_FFTW_H