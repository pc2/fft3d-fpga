#ifndef HELPER_HPP
#define HELPER_HPP

#include <iostream>
#include "fftfpga/fftfpga.h"

struct CONFIG{
  unsigned num; 
  unsigned dim; 
  bool inv;
  unsigned iter;
  std::string path;
  bool noverify;
  unsigned batch;
  bool burst;
  bool use_bram;
  bool emulate;
  bool use_usm;
};

void parse_args(int argc, char* argv[], CONFIG &config);

void print_config(const CONFIG config);

double getTimeinMilliSec();

void create_data(float2 *inp, const unsigned num);

bool verify_fftwf(const float2 *verify, float2 *fpgaout, const CONFIG config);

void perf_measures(const CONFIG config, fpga_t *runtime);

#endif // HELPER_HPP