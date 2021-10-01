#include <iostream>
#include <math.h>
#include <fftw3.h>
#include "cxxopts.hpp"
#include "helper.hpp"
#include "fftfpga/fftfpga.h"

using namespace std;

/**
 * \brief  create random single precision complex floating point values  
 * \param  inp : pointer to float2 data of size N 
 * \param  N   : number of points in the array
 * \return true if successful
 */
void create_data(float2 *inp, const unsigned num){

  if(inp == NULL || num < 4){ throw "Bad args in create data function";}

  for(unsigned i = 0; i < num; i++){
    inp[i].x = (float)((float)rand() / (float)RAND_MAX);
    inp[i].y = (float)((float)rand() / (float)RAND_MAX);
  }
}

/**
 * \brief  using cxxopts to parse cmd line args to the executable
 * \param  argc, argv
 * \param  config: custom structure of variables storing config values
 */
void parse_args(int argc, char* argv[], CONFIG &config){

  try{
    cxxopts::Options options("./fft<..>", "Offloading FFT on FPGA");
    options.add_options()
      ("n, num", "Number of sample points in a dimension", cxxopts::value<unsigned>()->default_value("64"))
      ("d, dim", "Number of dimensions", cxxopts::value<unsigned>()->default_value("3"))
      ("b, back", "Toggle Backward FFT", cxxopts::value<bool>()->default_value("false") )
      ("i, iter", "Number of iterations", cxxopts::value<unsigned>()->default_value("1"))
      ("p, path", "Path to FPGA bitstream", cxxopts::value<string>())
      ("y, noverify", "Toggle to not verify with FFTW", cxxopts::value<bool>()->default_value("false") )
      ("c, batch", "Number of batches of FFT calculations in FPGA", cxxopts::value<unsigned>()->default_value("1") )
      ("t, burst", "Toggle to use burst interleaved global memory accesses  in FPGA", cxxopts::value<bool>()->default_value("false") )
      ("m, use_bram", "Toggle to use BRAM instead of DDR for 3D Transpose  ", cxxopts::value<bool>()->default_value("false") )
      ("e, emulate", "Toggle to enable emulation ", cxxopts::value<bool>()->default_value("false") )
      ("h,help", "Print usage");
    auto opt = options.parse(argc, argv);

    // print help
    if (opt.count("help")){
      cout << options.help() << endl;
      exit(0);
    }

    config.num = opt["num"].as<unsigned>();
    config.dim = opt["dim"].as<unsigned>();
    config.inv = opt["back"].as<bool>();
    config.iter = opt["iter"].as<unsigned>();
    config.noverify = opt["noverify"].as<bool>();
    config.batch = opt["batch"].as<unsigned>();
    config.burst = opt["burst"].as<bool>();
    config.use_bram = opt["use_bram"].as<bool>();
    config.emulate = opt["emulate"].as<bool>();

    if(opt.count("path")){
      config.path = opt["path"].as<string>();
    }
    else{
      throw "please input path to bitstream. Exiting! \n";
    }
  }
  catch(const char *msg){
    cerr << "Error parsing options: " << msg << endl;
    exit(1);
  }
}

void print_config(CONFIG config){
  printf("\n------------------------------------------\n");
  printf("FFT CONFIGURATION: \n");
  printf("--------------------------------------------\n");
  printf("Type               : Complex to Complex\n");
  printf("Points             : %d%s \n", config.num, config.dim == 1 ? "" : config.dim == 2 ? "^2" : "^3");
  printf("Direction          : %s \n", config.inv ? "Backward":"Forward");
  printf("Placement          : In Place    \n");
  printf("Batch              : %d \n", config.batch);
  printf("Iterations         : %d \n", config.iter);
  printf("Transpose3D        : %s \n", config.use_bram ? "BRAM":"DDR");
  printf("Burst Interleaving : %s \n", config.burst ? "Yes":"No");
  printf("Emulation          : %s \n", config.emulate ? "Yes":"No");
  printf("--------------------------------------------\n\n");
}

/**
 * \brief Verify by comparing FFT computed in FPGA with FFTW 
 * \param verify: float2 pointer for fftw cpu computation
 * \param fpga_out: float2 pointer output from FPGA computation to verify 
 * \param config: struct of program state 
 * \return true if verification passed
 */
bool verify_fftwf(float2 *verify, float2 *fpgaout, const CONFIG config){

  unsigned sz = pow(config.num, config.dim);
  unsigned total_sz = config.batch * sz;

  fftwf_complex *fftw_data = fftwf_alloc_complex(sz);

  for(size_t i = 0; i < total_sz; i++){
    fftw_data[i][0] = verify[i].x;
    fftw_data[i][1] = verify[i].y;
  }

  //const int n[] = {N, N, N};
  int *n = (int*)calloc(config.num * config.dim , sizeof(int));
  for(unsigned i = 0; i < config.dim; i++){
    n[i] = config.num;
  }
  int idist = sz, odist = sz;
  int istride = 1, ostride = 1; // contiguous in memory

  fftwf_plan plan;
  if(config.inv){
    plan = fftwf_plan_many_dft(config.dim, n, config.batch, &fftw_data[0], NULL, istride, idist, fftw_data, NULL, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  else{
    plan = fftwf_plan_many_dft(config.dim, n, config.batch, &fftw_data[0], NULL, istride, idist, fftw_data, NULL, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);
  }

  fftwf_execute(plan);

  // Verification using SNR
  float mag_sum = 0, noise_sum = 0, magnitude, noise;
  for (size_t i = 0; i < total_sz; i++) {
    magnitude = fftw_data[i][0] * fftw_data[i][0] + \
                      fftw_data[i][1] * fftw_data[i][1];
    noise = (fftw_data[i][0] - fpgaout[i].x) \
        * (fftw_data[i][0] - fpgaout[i].x) + 
        (fftw_data[i][1] - fpgaout[i].y) * (fftw_data[i][1] - fpgaout[i].y);

    mag_sum += magnitude;
    noise_sum += noise;
  }

#ifndef NDEBUG
  printf("\nFFTW and FFTFPGA results comparison: \n");
  for(unsigned i = 0; i < total_sz; i++){
    printf("%u : fpga - (%e %e) cpu - (%e %e)\n", i, fpgaout[i].x, fpgaout[i].y, fftw_data[i][0], fftw_data[i][1]);
  }
  printf("\n\n");
#endif            

  float db = 10 * log(mag_sum / noise_sum) / log(10.0);
  
  fftwf_free(fftw_data);
  free(n);

  fftwf_destroy_plan(plan);

  // if SNR greater than 120, verification passes
  if(db > 120)
    return true;
  else{
    printf("\tSignal to noise ratio on output sample: %f --> %s\n\n", db, "FAILED");
    return false;
  }
}

/**
 * \brief print time taken for fpga and fftw runs
 * \param config: custom structure of variables storing config values 
 * \param runtime: iteration number of fpga timing measurements
 * \param total_api_time: time taken to call iter times the host code
 */

void perf_measures(const CONFIG config, fpga_t *runtime){

  fpga_t avg_runtime = {0.0, 0.0, 0.0, 0.0, 0.0, 0};
  for(unsigned i = 0; i < config.iter; i++){
    avg_runtime.exec_t += runtime[i].exec_t;
    avg_runtime.pcie_read_t += runtime[i].pcie_read_t;
    avg_runtime.pcie_write_t += runtime[i].pcie_write_t;
  }
  avg_runtime.exec_t = avg_runtime.exec_t / config.iter;
  avg_runtime.pcie_read_t = avg_runtime.pcie_read_t / config.iter;
  avg_runtime.pcie_write_t = avg_runtime.pcie_write_t / config.iter;

  fpga_t variance = {0.0, 0.0, 0.0, 0.0, 0.0, 0};
  fpga_t sd = {0.0, 0.0, 0.0, 0.0, 0.0, 0};
  for(unsigned i = 0; i < config.iter; i++){
    variance.exec_t += pow(runtime[i].exec_t - avg_runtime.exec_t, 2);
    variance.pcie_read_t += pow(runtime[i].pcie_read_t - avg_runtime.pcie_read_t, 2);
    variance.pcie_write_t += pow(runtime[i].pcie_write_t - avg_runtime.pcie_write_t, 2);
  }
  sd.exec_t = variance.exec_t / config.iter;
  sd.pcie_read_t = variance.pcie_read_t / config.iter;
  sd.pcie_write_t = variance.pcie_write_t / config.iter;

  double avg_total_runtime = avg_runtime.exec_t + avg_runtime.pcie_write_t + avg_runtime.pcie_read_t;

  double gpoints_per_sec = (config.batch * pow(config.num, config.dim)) / (avg_runtime.exec_t * 1e-3 * 1024 * 1024);

  double gBytes_per_sec = gpoints_per_sec * 8; // bytes

  double gflops = config.batch * config.dim * 5 * pow(config.num, config.dim) * (log((double)config.num)/log((double)2))/(avg_runtime.exec_t * 1e-3 * 1024*1024*1024); 

  printf("\n\n------------------------------------------\n");
  printf("Measurements \n");
  printf("--------------------------------------------\n");
  printf("%s", config.iter>1 ? "Average Measurements of iterations\n":"");
  printf("PCIe Write          = %.4lfms\n", avg_runtime.pcie_write_t);
  printf("Kernel Execution    = %.4lfms\n", avg_runtime.exec_t);
  printf("Kernel Exec/Batch   = %.4lfms\n", avg_runtime.exec_t / config.batch);
  printf("PCIe Read           = %.4lfms\n", avg_runtime.pcie_read_t);
  printf("Total               = %.4lfms\n", avg_total_runtime);
  printf("Throughput          = %.4lfGFLOPS/s | %.4lf GB/s\n", gflops, gBytes_per_sec);
  if(config.iter > 1){
    printf("\n");
    printf("%s", config.iter>1 ? "Standard Deviations of iterations\n":"");
    printf("PCIe Write          = %.4lfms\n", sd.pcie_write_t);
    printf("Kernel Execution    = %.4lfms\n", sd.exec_t);
    printf("PCIe Read           = %.4lfms\n", sd.pcie_read_t);
  }
}