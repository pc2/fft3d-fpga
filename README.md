# FFTFPGA

FFTFPGA is an OpenCL based library for Fast Fourier Transformations for FPGAs.
This repository provides OpenCL host code in the form of FFTW like APIs, which can be used to offload existing FFT routines to FPGAs with minimal effort. It also provides OpenCL kernels that can be synthesized to bitstreams, which the APIs can utilize.

FFTFPGA has been tested on Intel FPGAs namely, Stratix 10 GX 2800. This version of FFTFPGA supports the following features:

- 1D, 2D and 3D Transforms
- Input sizes of powers of 2
- Single Precision (32 bit floating point)
- C2C: Complex input to complex output
- Out-of-place transforms

## Build

### Dependencies

- CMake >= 3.15
- C Compiler with C99 support
- Intel OpenCL FPGA SDK

Additional submodules used:

- [argparse](https://github.com/cofyc/argparse.git) for command line argument parsing
- [hlslib](https://github.com/definelicht/hlslib) for CMake Intel FPGA OpenCL find packages
- [findFFTW](https://github.com/egpbos/findFFTW.git) for CMake FFTW find package
- [gtest](https://github.com/google/googletest.git) for unit tests

Build the host application by the following:

```bash
mkdir build && cd build
cmake ..
make
```

Building OpenCL kernels within the build directory:

- For emulation

```bash
make fft1d_emulate
make fft2d_emulate
make fft3d_emulate
```

- For synthesis

```bash
make fft1d_syn
make fft2d_syn
make fft3d_syn
```

## Execution

Execute:

```bash
./fft_fpga -n 64 -d 3 -s -p 64pt_fft1d_syn.aocx
```

Input Parameters

```bash
    -h, --help        show this help message and exit

Basic Options
    -n, --n=<int>     FFT Points
    -d, --dim=<int>   Dimensions
    -s, --sp          Single Precision
    -i, --iter=<int>  Iterations
    -b, --back        Backward FFT
    -v, --svm         Use SVM
    -p, --path=<str>  Path to bitstream
```

- Emulation

```bash
env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./fft_fpga
```

## Output

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
        Getting program binary from path 64pt_fft1d_emulate.aocx ...
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
PCIe Write         = 0.02ms
Throughput         = 0.00GFLOPS/s | 0.00 GB/s
```
