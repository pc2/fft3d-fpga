# User Guide

## Repository Structure

- `api`     : host code to setup and execute FPGA bitstreams. Compiled to static library that can be linked to your application
- `kernels` : OpenCL kernel code for 1d, 2d and 3d FFT
- `examples`: Sample code that makes use of the api
- `cmake`  : cmake modules used by the build system
- `scripts`: convenience slurm scripts
- `docs`   : describes models regarding performance and resource utilization

## Build System

### External Libraries

These additional libraries that are automatically fetched during system configuration:

- [cxxopts](https://github.com/jarro2783/cxxopts) for command line argument parsing
- [hlslib](https://github.com/definelicht/hlslib) for CMake Intel FPGA OpenCL find packages
- [findFFTW](https://github.com/egpbos/findFFTW.git) for CMake FFTW find package
- [gtest](https://github.com/google/googletest.git) for unit tests

### List of Kernels

|     | Kernel Name | Description                         |
| :-- | :---------- | :---------------------------------- |
| 1D  | fft1d       | OpenCL design provided by Intel     |
| 2D  | fft2d\_ddr  | DDR memory is used for 2D Transpose |
|     | fft2d\_bram | BRAM is used for 2D Transpose       |
| 3D  | fft3d\_ddr  | DDR memory is used for 3D Transpose |
|     | fft3d\_bram | BRAM is used for 3D Transpose       |

These kernels can be synthesized by appending `_emulate` or `_syn` to its suffix such as `fft1d_emulate`.

### Additional Kernel Builds

Generation of aocl reports

```bash
make <kernel_name>_report
make fft1d_report
```

## Compile Definitions

Using ccmake or by setting it using -D 

- `LOG_SIZE`: set the log of the length of the matrix. Example: `-DLOG_SIZE=6`.

## Enabling Shared Virtual Memory Extensions (SVM)

Currently tested for pacd5005 board. The board specification required setting the following attributes to global memory accesses, hence it has been set automatically. Otherwise, it can be set under the variable names.



## Runtime Input Parameters

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

## Output Interpretation

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


