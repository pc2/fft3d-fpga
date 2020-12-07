// Author: Arjun Ramaswami

#ifndef FFT3D_FFTW_H
#define FFT3D_FFTW_H

#include<stdbool.h>

bool verify_fftwf(float2 *fpgaout, const float2 *verify, int N, int dim, bool inverse, int how_many);

#endif // FFT3D_FFTW_H