#include <iostream>
#include "helper.hpp"

using namespace std;

int main(int argc, char* argv[]){

  CONFIG config;
  parse_args(argc, argv, config);

  print_config(config);
  return EXIT_SUCCESS;
}