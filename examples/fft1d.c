//  Author: Arjun Ramaswami

#include <stdio.h>
#include <stdlib.h> // EXIT_FAILURE
#include <math.h>
#include <stdbool.h>

#include "CL/opencl.h"
#include "fftfpga/fftfpga.h"

#include "argparse.h"
#include "helper.h"

static const char *const usage[] = {
    "bin/host [options]",
    NULL,
};

int main(int argc, const char **argv) {
  int N = 64, dim = 1, iter = 1, inv = 0, sp = 0, use_bram;
  char *path = "fft1d_emulate.aocx";
  const char *platform = "Intel(R) FPGA";
  fpga_t timing = {0.0, 0.0, 0.0, 0};
  int use_svm = 0, use_emulator = 0;
  bool status = true;

  struct argparse_option options[] = {
    OPT_HELP(),
    OPT_GROUP("Basic Options"),
    OPT_INTEGER('n',"n", &N, "FFT Points"),
    OPT_BOOLEAN('s',"sp", &sp, "Single Precision"),
    OPT_INTEGER('i',"iter", &iter, "Iterations"),
    OPT_BOOLEAN('b',"back", &inv, "Backward FFT"),
    OPT_BOOLEAN('v',"svm", &use_svm, "Use SVM"),
    OPT_BOOLEAN('m',"bram", &use_bram, "Use BRAM"),
    OPT_STRING('p', "path", &path, "Path to bitstream"),
    OPT_END(),
  };

  struct argparse argparse;
  argparse_init(&argparse, options, usage, 0);
  argparse_describe(&argparse, "Computing FFT using FPGA", "FFT size and dimensions are mandatory, default dimension and number of iterations are 1");
  argc = argparse_parse(&argparse, argc, argv);

  // Print to console the configuration chosen to execute during runtime
  print_config(N, dim, iter, inv, sp, use_bram);

  int isInit = fpga_initialize(platform, path, use_svm, use_emulator);
  if(isInit != 0){
    return EXIT_FAILURE;
  }

  // Select based on dimensions and precisions different functions
  if(sp == 0){
    printf("Not implemented. Work in Progress\n");
    return EXIT_SUCCESS;
  } 
  else{
    size_t inp_sz = sizeof(float2) * N * iter;

    float2 *inp = (float2*)fftfpgaf_complex_malloc(inp_sz, use_svm);
    float2 *out = (float2*)fftfpgaf_complex_malloc(inp_sz, use_svm);

    status = fftf_create_data(inp, N * iter);
    if(!status){
      free(inp);
      free(out);
      return EXIT_FAILURE;
    }

    timing = fftfpgaf_c2c_1d(N, inp, out, inv, iter);

    free(inp);
    free(out);
  }

  // destroy data
  fpga_final();

  if(timing.valid == 1){

    if(timing.exec_t == 0.0){
      fprintf(stderr, "Invalid measurement. Execute kernel did not run\n");
      return EXIT_FAILURE;
    }

    display_measures(0.0, timing.pcie_read_t, timing.pcie_write_t, timing.exec_t, N, dim, iter, inv, sp);
  }
  else{
    fprintf(stderr, "Invalid timing measurement. Function returned prematurely\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}