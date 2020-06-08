//  Author: Arjun Ramaswami

#ifndef HELPER_H
#define HELPER_H

#include "fftfpga/fftfpga.h"

int fftf_create_data(float2 *inp, int N);

int fft_create_data(double2 *inp, int N);

void print_config(int N, int dim, int iter, int inv, int sp, int use_bram);

void display_measures(fpga_t timing, int N, int dim, int iter, int inv, int sp);

#endif // HELPER_H
