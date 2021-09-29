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
};

void parse_args(int argc, char* argv[], CONFIG &config);

void print_config(CONFIG config);

double getTimeinMilliSec();

void create_data(float2 *inp, const unsigned num);

#endif // HELPER_HPP