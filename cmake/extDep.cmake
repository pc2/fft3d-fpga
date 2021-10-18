cmake_minimum_required(VERSION 3.10.3)

message("-- Fetching HLSLib")
# External Dependencies
## HLSLib - finds Intel OpenCL for FPGA installation
include(FetchContent)
FetchContent_Declare(
  hlslib
  GIT_REPOSITORY      https://github.com/definelicht/hlslib.git
  GIT_TAG             66462501a8779694f8b64b2c32cc59552cd59429
)
FetchContent_MakeAvailable(hlslib)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${hlslib_SOURCE_DIR}/cmake)

message("-- Fetching Cxxopts")
## CxxOpts - for command line argument parsing
FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts
  GIT_TAG        v2.2.1
)
FetchContent_MakeAvailable(cxxopts)

message("-- Fetching gTest")
## googleTest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.10.0
)
FetchContent_MakeAvailable(googletest)
