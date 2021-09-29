#include <iostream>
#include "cxxopts.hpp"
#include "helper.hpp"

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
  printf("--------------------------------------------\n\n");
}