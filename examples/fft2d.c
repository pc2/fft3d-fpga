//  Author: Arjun Ramaswami

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "CL/opencl.h"
#include "fftfpga/fftfpga.h"

#include "argparse.h"
#include "helper.h"
#include "verify_fftw.h"

static const char *const usage[] = {
    "bin/host [options]",
    NULL,
};

int main(int argc, const char **argv) {
  int N = 64, dim = 2, iter = 1, inv = 0, sp = 0, use_bram = 0;

  char *path = "fft2d_emulate.aocx";
  const char *platform = "Intel(R) FPGA";
  fpga_t timing = {0.0, 0.0, 0.0, 0};
  int use_svm = 0, use_emulator = 0;
  double avg_rd = 0.0, avg_wr = 0.0, avg_exec = 0.0;

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

  if(fpga_initialize(platform, path, use_svm, use_emulator)){
    return 1;
  }

  if(sp == 0){
    fprintf(stderr, "Not implemented. Work in Progress\n");
    return 0;
  } 
  else{

    size_t inp_sz = sizeof(float2) * N * N;
    float2 *inp = (float2*)fftfpgaf_complex_malloc(inp_sz, use_svm);
    float2 *out = (float2*)fftfpgaf_complex_malloc(inp_sz, use_svm);

    fftf_create_data(inp, N * N);

    for(size_t i = 0; i < iter; i++){
      if(use_bram == 1){
        timing = fftfpgaf_c2c_2d_bram(N, inp, out, inv);
      }
      else{
        timing = fftfpgaf_c2c_2d_ddr(N, inp, out, inv);
      }

  #ifdef USE_FFTW
      if(verify_sp_fft2d_fftw(out, inp, N, inv)){
        printf("2d FFT Verification Passed \n");
      }
      else{
        printf("2d FFT Verification Failed \n");
      }
  #endif

      if(timing.valid != 0){
        avg_rd += timing.pcie_read_t;
        avg_wr += timing.pcie_write_t;
        avg_exec += timing.exec_t;
      }
      else{
        break;
      } 
    }

    free(inp);
    free(out);
  }

  // destroy data
  fpga_final();

  if(timing.valid == 1){

    if(avg_exec == 0.0){
      fprintf(stderr, "Invalid measurement. Execute kernel did not run\n");
      return 1;
    }

    display_measures(avg_rd, avg_wr, avg_exec, N, dim, iter, inv, sp);
  }
  else{
    fprintf(stderr, "Invalid timing measurement. Function returned prematurely\n");
    return 1;
  }

  return 0;
}