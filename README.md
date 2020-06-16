# FFTFPGA

FFTFPGA is an OpenCL based library for Fast Fourier Transformations for FPGAs.
This repository provides OpenCL host code in the form of FFTW like APIs, which can be used to offload existing FFT routines to FPGAs with minimal effort. It also provides OpenCL kernels that can be synthesized to bitstreams, which the APIs can utilize.

## Features

- 1D, 2D and 3D Transforms
- Input sizes of powers of 2
- Single Precision (32 bit floating point)
- C2C: Complex input to complex output
- Out-of-place transforms

## Supported FPGAs

The library has been tested on the following FPGAs:

- Intel Stratix 10 GX 2800
- Intel Arria 10

## Who is using FFTFPGA?

- [CP2K](https://github.com/cp2k/cp2k):  the quantum chemistry software package has an interface to offload 3d FFTs to Intel FPGAs that uses the OpenCL kernel designs of FFTFPGA.

## Getting Started


### Dependencies

- [CMake](https://cmake.org/) >= 3.10
- C Compiler with C11 support
- Intel OpenCL FPGA SDK

Additional submodules used:

- [argparse](https://github.com/cofyc/argparse.git) for command line argument parsing
- [hlslib](https://github.com/definelicht/hlslib) for CMake Intel FPGA OpenCL find packages
- [findFFTW](https://github.com/egpbos/findFFTW.git) for CMake FFTW find package
- [gtest](https://github.com/google/googletest.git) for unit tests

### Structure

The repository consists of the following:

- `api`     : host code to setup and execute FPGA bitstreams. Compiled to static library that can be linked to your application
- `kernels` : OpenCL kernel code for 1d, 2d and 3d FFT
- `examples`: Sample code that makes use of the api
- `extern` : external packages as submodules required to run the project
- `cmake`  : cmake modules used by the build system
- `scripts`: convenience slurm scripts
- `docs`   : describes models regarding performance and resource utilization
- `data`   : evaluation results and measurements

### Setup

FFTFPGA has a CMake build script that can be used to build the project. This consists of two steps:

1. Building the API that can be linked to your application
2. Building OpenCL Kernel Designs that are used by the API

#### API

```bash
mkdir build && cd build  # Directory to store build outputs
cmake ..
make
```

This generates the following:

- `fftfpga` static library to link such as `-lfftfpga`
- `fftfpga/fftfpga.h` header file

The sample programs given in the `example` directory are also compiled to binaries of their respective names, which makes use of the files given previously.

#### OpenCL Kernel Designs

FFTFPGA provides OpenCL designs that can be compiled for different options:

- Emulation

```bash
make <kernel_name>_emu
make fft1d_emulate
```

- Report Generation

```bash
make <kernel_name>_rep
make fft1d_rep
```

- Synthesis

```bash
make <kernel_name>_syn
make fft1d_syn
```

Paths to these bitstreams should be provided as parameters to certain API calls to execute the design.

### Examples

#### Additional Dependency

- FFTW3

#### Execution

```bash
./fft3d -n 64 -m -s -p emu_64_fft3d_bram/fft3d_bram.aocx
```

Prepend the command with `CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1`for emulation.

#### Compile Definitions

- `LOG_SIZE`: set the log of the length of the matrix. Example: `-DLOG_SIZE=6`.

#### Runtime Input Parameters

```bash
    -h, --help        show this help message and exit

Basic Options
    -n, --n=<int>     FFT Points
    -s, --sp          Single Precision
    -i, --iter=<int>  Iterations
    -b, --back        Backward FFT
    -v, --svm         Use SVM
    -m, --bram        Use BRAM
    -p, --path=<str>  Path to bitstreamm
```

#### Output

The examples measure and output relevant performance metrics that are shown below:

```bash
------------------------------------------
FFT Configuration:
--------------------------------------------
Type               = Complex to Complex
Points             = 64
Precision          = Single
Direction          = Forward
Placement          = In Place
Iterations         = 1
--------------------------------------------

        Initializing FPGA ...
        Getting program binary from path emu_64_fft3d_bram/fft3d_bram.aocx ...
        Building program ...
        FFT kernel initialization is complete.
        Cleaning up FPGA resources ...

------------------------------------------
Measurements
--------------------------------------------
Points             = 64
Precision          = Single
Direction          = Forward
PCIe Write         = 0.03ms
Kernel Execution   = 0.48ms
PCIe Read          = 0.02ms
Throughput         = 0.00GFLOPS/s | 0.00 GB/s
```

- `PCIe Write` and `PCIe Read` the time taken in milliseconds for transfer of data from host to global memory through PCIe bus.

- `Kernel Execution` represents the time taken in milliseconds for the execution of the OpenCL implementation that includes the global memory accesses.

## Publications

FFTFPGA has been cited in the following publications:

1. CP2K: An electronic structure and molecular dynamics software package - Quickstep: Efficient and accurate electronic structure calculations: https://doi.org/10.1063/5.0007045

## Contact

- [Arjun Ramaswami](https://github.com/arjunramaswami)
- [Tobias Kenter](https://www.uni-paderborn.de/person/3145/)
- [Thomas D. Kühne](https://chemie.uni-paderborn.de/arbeitskreise/theoretische-chemie/kuehne/)
- [Christian Plessl](https://github.com/plessl)

## Acknowledgements

- [Marius Meyer](https://pc2.uni-paderborn.de/about-pc2/staff-board/staff/person/?tx_upbperson_personsite%5BpersonId%5D=40778&tx_upbperson_personsite%5Bcontroller%5D=Person&cHash=867dec7cae43afd76c85cd503d8da47b) for code reviews, testing and discussions.
