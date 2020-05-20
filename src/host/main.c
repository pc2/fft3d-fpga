//  Author: Arjun Ramaswami

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "CL/opencl.h"

#include "argparse.h"
#include "include/fftfpga.h"
#include "include/helper.h"

static const char *const usage[] = {
    "bin/host [options]",
    NULL,
};

static void print_config(int N, int dim, int iter, int inv, int sp);
static void display_measures(fpga_t timing, int N, int dim, int iter, int inv, int sp);

int main(int argc, const char **argv) {
  int N = 64, dim = 1, iter = 1, inv = 0, sp = 0, use_bram = 1;
  char *path = "64pt_fft1d_emulate.aocx";
  const char *platform = "Intel(R) FPGA";
  fpga_t timing = {0.0, 0.0, 0.0};
  int use_svm = 0, use_emulator = 0;

  struct argparse_option options[] = {
    OPT_HELP(),
    OPT_GROUP("Basic Options"),
    OPT_INTEGER('n',"n", &N, "FFT Points"),
    OPT_INTEGER('d',"dim", &dim, "Dimensions"),
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
  print_config(N, dim, iter, inv, sp);

  if(fpga_initialize(platform, path, use_svm, use_emulator)){
    return 1;
  }

  // Select based on dimensions and precisions different functions
  switch(dim){
    case 1:
      if(sp == 0){
        fprintf(stderr, "Not implemented. Work in Progress\n");
        return 0;
      } 
      else{
        size_t inp_sz = sizeof(float2) * N * iter;

        float2 *inp = (float2*)fftfpgaf_complex_malloc(inp_sz, use_svm);
        float2 *out = (float2*)fftfpgaf_complex_malloc(inp_sz, use_svm);

        fftf_create_data(inp, N * iter);

        timing = fftfpgaf_c2c_1d(N, inp, out, inv, iter);

        free(inp);
        free(out);
      }
      break;
    case 2:
      if(sp == 0){
        fprintf(stderr, "Not implemented. Work in Progress\n");
        return 0;
      } 
      else{

        size_t inp_sz = sizeof(float2) * N * N;
        float2 *inp = (float2*)fftfpgaf_complex_malloc(inp_sz, use_svm);
        float2 *out = (float2*)fftfpgaf_complex_malloc(inp_sz, use_svm);

        fftf_create_data(inp, N * N);

        if(use_bram == 1){
          timing = fftfpgaf_c2c_2d_bram(N, inp, out, inv);
        }
        else{
          timing = fftfpgaf_c2c_2d_ddr(N, inp, out, inv);
        }

        free(inp);
        free(out);
      }
      break;
    case 3:
      if(sp == 0){
        fprintf(stderr, "Not implemented. Work in Progress\n");
        return 0;
      } 
      else{
        size_t inp_sz = sizeof(float2) * N * N * N;
        float2 *inp = (float2*)fftfpgaf_complex_malloc(inp_sz, use_svm);
        float2 *out = (float2*)fftfpgaf_complex_malloc(inp_sz, use_svm);

        fftf_create_data(inp, N * N * N);

        timing = fftfpgaf_c2c_3d_ddr(N, inp, out, inv);
        free(inp);
        free(out);
      }
      break;

    default:
      fprintf(stderr, "No dimension entered \n");
      return 0;
  }

  // destroy data
  fpga_final();

  if(timing.valid == 1){

    if(timing.exec_t == 0.0){
      fprintf(stderr, "Measurement invalid\n");
      return 1;
    }

    display_measures(timing, N, dim, iter, inv, sp);
  }

  return 0;
}

void print_config(int N, int dim, int iter, int inv, int sp){
  printf("\n------------------------------------------\n");
  printf("FFT Configuration: \n");
  printf("--------------------------------------------\n");
  printf("Type               = Complex to Complex\n");
  printf("Points             = %d%s \n", N, dim == 1 ? "" : dim == 2 ? "^2" : "^3");
  printf("Precision          = %s \n",  sp==1 ? "Single": "Double");
  printf("Direction          = %s \n", inv ? "Backward":"Forward");
  printf("Placement          = In Place    \n");
  printf("Iterations         = %d \n", iter);
  printf("--------------------------------------------\n\n");
}

/**
 * \brief  print time taken for fpga and fftw runs to a file
 * \param  timing: kernel execution and pcie transfer timing 
 * \param  N: fft size
 * \param  dim: number of dimensions of size
 * \param  iter: number of iterations of each transformation (if BATCH mode)
 * \param  inv: 1 if backward transform
 * \param  single precision floating point transformation
 */
void display_measures(fpga_t timing, int N, int dim, int iter, int inv, int sp){

  double exec = timing.exec_t / iter;
  double gpoints_per_sec = (pow(N, dim)  / (exec * 1e-3)) * 1e-9;
  double gBytes_per_sec = 0.0;

  if(sp == 1){
    gBytes_per_sec =  gpoints_per_sec * 8; // bytes
  }
  else{
    gBytes_per_sec *=  gpoints_per_sec * 16;
  }

  double gflops = dim * 5 * pow(N, dim) * (log((double)N)/log((double)2))/(exec * 1e-3 * 1E9); 

  printf("\n------------------------------------------\n");
  printf("Measurements \n");
  printf("--------------------------------------------\n");
  printf("Points             = %d%s \n", N, dim == 1 ? "" : dim == 2 ? "^2" : "^3");
  printf("Precision          = %s\n",  sp==1 ? "Single": "Double");
  printf("Direction          = %s\n", inv ? "Backward":"Forward");
  printf("PCIe Write         = %.2lfms\n", timing.pcie_write_t);
  printf("Kernel Execution   = %.2lfms\n", exec);
  printf("PCIe Write         = %.2lfms\n", timing.pcie_read_t);
  printf("Throughput         = %.2lfGFLOPS/s | %.2lf GB/s\n", gflops, gBytes_per_sec);
}