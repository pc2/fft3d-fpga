//  Author: Arjun Ramaswami

#include <stdio.h>
#include <stdlib.h> // EXIT_FAILURE
#include <math.h>
#include <stdbool.h>

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
  int N = 64, dim = 3, iter = 1, batch = 1;

  bool interleaving = false, use_bram = true, sp = true;
  bool use_svm = false, inv = false;
  bool status = true, use_emulator = false;

  char *path = "fft3d_emulate.aocx";
  const char *platform;

  fpga_t timing = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0};
  double avg_rd = 0.0, avg_wr = 0.0, avg_exec = 0.0;
  double avg_hw_rd = 0.0, avg_hw_wr = 0.0, avg_hw_exec = 0.0;
  double temp_timer = 0.0, total_api_time = 0.0;

  struct argparse_option options[] = {
    OPT_HELP(),
    OPT_GROUP("Basic Options"),
    OPT_INTEGER('n',"n", &N, "FFT Points"),
    OPT_BOOLEAN('s',"sp", &sp, "Single Precision"),
    OPT_INTEGER('i',"iter", &iter, "Iterations"),
    OPT_BOOLEAN('b',"back", &inv, "Backward FFT"),
    OPT_BOOLEAN('t',"interleaving", &interleaving, "Use burst interleaving in case of BRAM designs"),
    OPT_STRING('p', "path", &path, "Path to bitstream"),
    OPT_BOOLEAN('e', "emu", &use_emulator, "Use emulator"),
    OPT_END(),
  };

  struct argparse argparse;
  argparse_init(&argparse, options, usage, 0);
  argparse_describe(&argparse, "Computing FFT using FPGA", "FFT size and dimensions are mandatory, default dimension and number of iterations are 1");
  argc = argparse_parse(&argparse, argc, argv);

  // Print to console the configuration chosen to execute during runtime
  print_config(N, dim, iter, inv, sp, batch, use_bram, interleaving);

  if(use_emulator){
    platform = "Intel(R) FPGA Emulation Platform for OpenCL(TM)";
    //platform = "Intel(R) FPGA";
  }
  else{
    platform = "Intel(R) FPGA SDK for OpenCL(TM)";
    //platform = "Intel(R) FPGA";
  }
  
  int isInit = fpga_initialize(platform, path, use_svm);
  if(isInit != 0){
    fprintf(stderr, "FPGA initialization error\n");
    return EXIT_FAILURE;
  }

  size_t inp_sz = sizeof(float2) * N * N * N;
  float2 *inp = (float2*)fftfpgaf_complex_malloc(inp_sz);
  float2 *out = (float2*)fftfpgaf_complex_malloc(inp_sz);

  for(size_t i = 0; i < iter; i++){

    // create and destroy data every iteration
    status = fftf_create_data(inp, N * N * N);
    if(!status){
      fprintf(stderr, "Error in Data Creation \n");
      free(inp);
      free(out);
      return EXIT_FAILURE;
    }

    // use bram for 3d Transpose
    temp_timer = getTimeinMilliseconds();
    timing = fftfpgaf_c2c_3d_bram(N, inp, out, inv, interleaving);
    total_api_time += getTimeinMilliseconds() - temp_timer;

#ifdef USE_FFTW
    if(!verify_fftwf(out, inp, N, 3, inv, 1)){
      fprintf(stderr, "3d FFT Verification Failed \n");
      free(inp);
      free(out);
      return EXIT_FAILURE;
    }
#endif
    if(timing.valid == 0){
      fprintf(stderr, "Invalid execution, timing found to be 0");
      free(inp);
      free(out);
      return EXIT_FAILURE;
    }

    avg_rd += timing.pcie_read_t;
    avg_wr += timing.pcie_write_t;
    avg_exec += timing.exec_t;
    avg_hw_rd += timing.hw_pcie_read_t;
    avg_hw_wr += timing.hw_pcie_write_t;
    avg_hw_exec += timing.hw_exec_t;

    printf("Iter: %lu\n", i);
    printf("\tPCIe Rd: %lfms\n", timing.pcie_read_t);
    printf("\tKernel: %lfms\n", timing.exec_t);
    printf("\tPCIe Wr: %lfms\n\n", timing.pcie_write_t);
            
    printf("Hw Counters: \n");
    printf("\tHW PCIe Rd: %lfms\n", timing.hw_pcie_read_t);
    printf("\tHW Kernel: %lfms\n", timing.hw_exec_t);
    printf("\tHW PCIe Wr: %lfms\n\n", timing.hw_pcie_write_t);
  }  // iter
  // destroy FFT input and output
  free(inp);
  free(out);

  // destroy fpga state
  fpga_final();

  // display performance measures
  display_measures(total_api_time, avg_rd, avg_wr, avg_exec, avg_hw_rd, avg_hw_wr, avg_hw_exec, N, dim, iter, batch, inv, sp);

  return EXIT_SUCCESS;
}
