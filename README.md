# FFTFPGA

[![GitHub license](https://img.shields.io/github/license/pc2/fft3d-fpga)](https://github.com/pc2/fft3d-fpga/blob/master/LICENSE)
[![GitHub license](https://img.shields.io/github/v/release/pc2/fft3d-fpga)](https://github.com/pc2/fft3d-fpga/releases/)

FFTFPGA is an OpenCL based library for Fast Fourier Transformations for FPGAs.
This repository provides OpenCL host code in the form of FFTW like APIs, which can be used to offload existing FFT routines to FPGAs with minimal effort. It also provides OpenCL kernels that can be synthesized to bitstreams, which the APIs can utilize.

## Features

- 1D, 2D and 3D Transforms
- Input sizes of powers of 2
- Single Precision (32 bit floating point)
- C2C: Complex input to complex output
- Out-of-place transforms
- Batched 3D transforms
- OpenCL Shared Virtual Memory (SVM) extensions for data transfers

## Supported FPGAs

This library has been tested using the following FPGAs present in the [Noctua](https://pc2.uni-paderborn.de/hpc-services/available-systems/noctua1/) cluster of the Paderborn Center for Parallel Computing (PC2) at Paderborn University:

- [Bittware 520N](https://www.bittware.com/fpga/520n/) card with Intel Stratix 10 GX 2800 FPGA
- [Intel FPGA PAC D5005](https://www.intel.com/content/www/us/en/programmable/products/boards_and_kits/dev-kits/altera/intel-fpga-pac-d5005/overview.html) card with Intel Stratix 10 SX 2800 FPGA

## Who is using FFTFPGA?

- [CP2K](https://github.com/cp2k/cp2k):  the quantum chemistry software package has an interface to offload 3d FFTs to Intel FPGAs that uses the OpenCL kernel designs of FFTFPGA.

## Quick Setup

Firstly, *dependencies* for building the system
- [CMake](https://cmake.org/) >= 3.10
- C++ compiler with C++11 support (GCC 4.9.0+)
- Intel FPGA SDK for OpenCL
- FFTW3

Once you have this covered, execute the following:

```bash
mkdir build && cd build  # Directory to store build outputs
cmake ..
make
```

You have built the *API* i.e., the OpenCL host code that invokes different transformations correctly are packed into a static library. This must be linked to an application.

A sample application that helps invoke the APIs.

*Strictly said*, you have built the following:

- `fftfpga` static library, linked such as `-lfftfpga`
- `fftfpga/fftfpga.h` header file
- `fft` - a sample application which links and includes the above two.

Now, for the real deal, synthesizing the OpenCL FFT kernels. These can be synthesized to run on software emulation or on hardware as bitstreams.

- Emulation

```bash
make <kernel_name>_emu
make fft3d_ddr_emulate
```

- Hardware Bitstream

```bash
make <kernel_name>_syn
make fft3d_ddr_syn
```

Putting them all together, set the path to the synthesized bitstream along with other correct configurations as command line parameters to the sample application generated, to execute the transformation.

```bash
./fft --num=64 --dim=3 --path=fft3d_ddr_128.aocx
```

*Tip*: for emulation, use the `--emulate` command line parameter.

For explanations regarding the command line options and the OpenCL kernels, check out x and y in the advanced guide.  

## Publications

FFTFPGA has been cited in the following publications:

1. Evaluating the Design Space for Offloading 3D FFT Calculations to an FPGA for High-Performance Computing : https://doi.org/10.1007/978-3-030-79025-7_21

2. CP2K: An electronic structure and molecular dynamics software package - Quickstep: Efficient and accurate electronic structure calculations: https://doi.org/10.1063/5.0007045

3. Efficient Ab-Initio Molecular Dynamic Simulations by Offloading Fast Fourier Transformations to FPGAs : https://doi.org/10.1109/FPL50879.2020.00065

## Related Repositories

- ConvFPGA - an OpenCL based library for FFT-based convolution on FPGAs
- FFTFPGA-eval

## Contact

- [Arjun Ramaswami](https://github.com/arjunramaswami)
- [Tobias Kenter](https://www.uni-paderborn.de/person/3145/)
- [Thomas D. Kühne](https://chemie.uni-paderborn.de/arbeitskreise/theoretische-chemie/kuehne/)
- [Christian Plessl](https://github.com/plessl)

## Acknowledgements

- [Marius Meyer](https://pc2.uni-paderborn.de/about-pc2/staff-board/staff/person/?tx_upbperson_personsite%5BpersonId%5D=40778&tx_upbperson_personsite%5Bcontroller%5D=Person&cHash=867dec7cae43afd76c85cd503d8da47b) for code reviews, testing and discussions.
