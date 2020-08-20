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
  int N = 64, dim = 1, iter = 1, inv = 0, sp = 0, batch = 1, use_bram;
  char *path = "fft1d_emulate.aocx";
  const char *platform = "Intel(R) FPGA";
  fpga_t timing = {0.0, 0.0, 0.0, 0};
  int use_svm = 0;
  double avg_rd = 0.0, avg_wr = 0.0, avg_exec = 0.0;
  double temp_timer = 0.0, total_api_time = 0.0;
  bool status = true, use_emulator = false;

  struct argparse_option options[] = {
    OPT_HELP(),
    OPT_GROUP("Basic Options"),
    OPT_INTEGER('n',"n", &N, "FFT Points"),
    OPT_BOOLEAN('s',"sp", &sp, "Single Precision"),
    OPT_INTEGER('i',"iter", &iter, "Iterations"),
    OPT_BOOLEAN('b',"back", &inv, "Backward FFT"),
    OPT_BOOLEAN('v',"svm", &use_svm, "Use SVM"),
    OPT_INTEGER('c',"batch", &batch, "Batch"),
    OPT_BOOLEAN('m',"bram", &use_bram, "Use BRAM"),
    OPT_STRING('p', "path", &path, "Path to bitstream"),
    OPT_BOOLEAN('e', "emu", &use_emulator, "Use emulator"),
    OPT_END(),
  };

  struct argparse argparse;
  argparse_init(&argparse, options, usage, 0);
  argparse_describe(&argparse, "Computing FFT using FPGA", "FFT size and dimensions are mandatory, default dimension and number of iterations are 1");
  argc = argparse_parse(&argparse, argc, argv);

  // Print to console the configuration chosen to execute during runtime
  print_config(N, dim, iter, inv, sp, batch, use_bram);
  
  if(use_emulator){
    platform = "Intel(R) FPGA Emulation Platform for OpenCL(TM)";
  }
  else{
    platform = "Intel(R) FPGA SDK for OpenCL(TM)";
  }

  int isInit = fpga_initialize(platform, path, use_svm);
  if(isInit != 0){
    return EXIT_FAILURE;
  }

  // Select based on dimensions and precisions different functions
  if(sp == 0){
    printf("Not implemented. Work in Progress\n");
    return EXIT_SUCCESS;
  } 
  else{

    // find the average of iterations of batched 1D FFTs
    // random data every iteration and every batch
    for(size_t i = 0; i < iter; i++){

      size_t inp_sz = sizeof(float2) * N * batch;

      float2 *inp = (float2*)fftfpgaf_complex_malloc(inp_sz);
      float2 *out = (float2*)fftfpgaf_complex_malloc(inp_sz);

      status = fftf_create_data(inp, N * batch);
      if(!status){
        fprintf(stderr, "Error in Data Creation \n");
        free(inp);
        free(out);
        return EXIT_FAILURE;
      }

      temp_timer = getTimeinMilliseconds();
      timing = fftfpgaf_c2c_1d(N, inp, out, inv, batch);
      total_api_time += getTimeinMilliseconds() - temp_timer;

      // TODO: Verification of bit reversed output
      if(timing.valid == 0){
        fprintf(stderr, "Invalid execution, timing found to be 0");
        free(inp);
        free(out);
        return EXIT_FAILURE;
      }
      avg_rd += timing.pcie_read_t;
      avg_wr += timing.pcie_write_t;
      avg_exec += timing.exec_t;

      printf("Iter: %lu\n", i);
      printf("\tPCIe Rd: %lfms\n", timing.pcie_read_t);
      printf("\tKernel: %lfms\n", timing.exec_t);
      printf("\tPCIe Wr: %lfms\n\n", timing.pcie_write_t);
             
      // destroy FFT input and output
      free(inp);
      free(out);
    }
  }

  // destroy data
  fpga_final();


  display_measures(total_api_time, avg_rd, avg_wr, avg_exec, N, dim, iter, batch, inv, sp);

  return EXIT_SUCCESS;
}