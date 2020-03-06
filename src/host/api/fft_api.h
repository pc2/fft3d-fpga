/******************************************************************************
 *  Authors: Arjun Ramaswami
 *****************************************************************************/

#ifndef FFT_API_H
#define FFT_API_H

typedef struct {
  double x;
  double y;
} double2;

typedef struct {
  float x;
  float y;
} float2;

#ifdef __FPGA_SP
    typedef float2 cmplx;
#else
    typedef double2 cmplx;
#endif

#endif