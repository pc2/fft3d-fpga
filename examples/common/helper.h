//  Author: Arjun Ramaswami

#ifndef HELPER_H
#define HELPER_H

#include <stdbool.h>
#include "fftfpga/fftfpga.h"

bool fftf_create_data(float2 *inp, unsigned N);

bool fft_create_data(double2 *inp, unsigned N);

void print_config(int N, int dim, int iter, bool inv, bool sp, int batch, bool use_bram, bool interleaving);

void display_measures(double total_api_time, double pcie_rd, double pcie_wr, double exec, double avg_hw_pcie_rd, double avg_hw_pcie_wr, double avg_hw_exec, int N, int dim, int iter, int batch, bool inv, bool sp);

double getTimeinMilliseconds();
#endif // HELPER_H
