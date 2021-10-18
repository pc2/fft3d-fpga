#include <iostream>
#include <math.h>
#include "fftfpga/fftfpga.h"
#include "helper.hpp"

using namespace std;

int main(int argc, char* argv[]){

  CONFIG config;
  parse_args(argc, argv, config);
  print_config(config);

  const char* platform;
  if(config.emulate)
    platform = "Intel(R) FPGA Emulation Platform for OpenCL(TM)";
  else
    platform = "Intel(R) FPGA SDK for OpenCL(TM)";
  
  int isInit = fpga_initialize(platform, config.path.data(), config.use_usm);
  if(isInit != 0){
    cerr << "FPGA initialization error\n";
    return EXIT_FAILURE;
  }

  const unsigned num = config.num;
  const unsigned sz = config.batch * pow(num, config.dim);
  float2 *inp = new float2[sz]();
  float2 *out = new float2[sz]();
  fpga_t runtime[config.iter];

  try{
    create_data(inp, sz);
  
    const unsigned inv = config.inv;
    const bool burst = config.burst;

    for(unsigned i = 0; i < config.iter; i++){
      cout << i << ": Calculating FFT - " << endl;
      switch(config.dim) {
        case 1: {
          if(config.use_usm)
            runtime[i] = fftfpgaf_c2c_1d_svm(num, inp, out, inv, config.batch);
          else
            runtime[i] = fftfpgaf_c2c_1d(num, inp, out, inv, config.batch);
          break;
        }
        case 2: {
          if(config.use_bram && config.use_usm)
            runtime[i] = fftfpgaf_c2c_2d_bram_svm(num, inp, out, inv, config.batch);
          else if(config.use_bram && !config.use_usm)
            runtime[i] = fftfpgaf_c2c_2d_bram(num, inp, out, inv, burst, config.batch);
          else
            runtime[i] = fftfpgaf_c2c_2d_ddr(num, inp, out, inv); 
          break;
        }
        case 3:{
          if(config.use_bram)
            runtime[i] = fftfpgaf_c2c_3d_bram(num, inp, out, inv, burst);
          else if(!config.use_bram && (!config.use_usm) && (config.batch > 1))
            runtime[i] = fftfpgaf_c2c_3d_ddr_batch(num, inp, out, inv, burst, config.batch);
          else if(config.use_usm){
            if(config.batch > 1)
              runtime[i] = fftfpgaf_c2c_3d_ddr_svm_batch(num, inp, out, inv, config.batch);
            else 
              runtime[i] = fftfpgaf_c2c_3d_ddr_svm(num, inp, out, inv, burst);
            break;
          }
          else
            runtime[i] = fftfpgaf_c2c_3d_ddr(num, inp, out, inv);
          break;
        }
        default:
          break;
      }

      if(!config.noverify){
        if(!verify_fftwf(inp, out, config)){
          char excp[80];
          snprintf(excp, 80, "Iter %u: FPGA result incorrect in comparison to FFTW\n", i);
          throw runtime_error(excp);
        }
      }
    }
  }
  catch(const char* msg){
    cerr << msg << endl;
    fpga_final();
    delete inp;
    delete out;
    return EXIT_FAILURE;
  }

  // destroy fpga state
  fpga_final();

  perf_measures(config, runtime);

  delete inp;
  delete out;
  return EXIT_SUCCESS;
}