# User Guide

## Repository Structure

- `api`     : host code to setup and execute FPGA bitstreams. Compiled to static library that can be linked to your application
- `kernels` : OpenCL kernel code for 1d, 2d and 3d FFT
- `examples`: Sample code that makes use of the api
- `cmake`  : cmake modules used by the build system
- `scripts`: convenience slurm scripts
- `docs`   : describes models regarding performance and resource utilization

## CMake Build Setup

### External Libraries

These additional libraries are automatically fetched during system configuration:

- [cxxopts](https://github.com/jarro2783/cxxopts) for command line argument parsing
- [hlslib](https://github.com/definelicht/hlslib) for CMake Intel FPGA OpenCL find packages
- [findFFTW](https://github.com/egpbos/findFFTW.git) for CMake FFTW find package
- [gtest](https://github.com/google/googletest.git) for unit tests

### Configuration Options

The following compile options can be set when creating a CMake build directory either using the `-D` parameter or by using the cmake-gui such as:

`cmake -DCMAKE_BUILD_TYPE=Release ..`

`ccmake ..`

| ** Name                  ** | ** Description                                                                                                                              **     | ** Default Values                 ** | ** Alternate Values        ** |
| :-------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------- | :---------------------------- |
|  `AOC\_FLAGS*`            * | Intel offline compiler flags used for kernel compilation                                                                                           | `-g -v -no-interleaving=default`     |                               |
| `EMU\_FLAGS`                | Compiler flags used for emulation, with fast emulation as default                                                                                  | `-march=emulator`                    |                               |
| `FPGA\_BOARD\_NAME`         | Name of the target FPGA board                                                                                                                      | `p520\_hpc\_sg280l`                  | `pac\_s10\_usm`               |
| `LOG\_FFT\_SIZE`            | Currently supported log2 number of points along each FFT dimension                                                                                 | 6                                    | 5, 7, 8, 9                    |
|  `BURST\_INTERLEAVING*`    |  Toggle to enable burst interleaved global memory accesses  <br>  Sets the `-no-interleaving=` to the `AOC\_FLAGS*` *parameter*                   | NO                                   | YES                           |
| `DDR\_BUFFER\_LOCATION`     |  Name of the global memory interface found in the `board\_spec.xml`  <br>  `DDR` :`p520\_hpc\_sg280l`, `device` : `pac\_s10\_usm` board            | `DDR`                                | `device`                      |
| `SVM\_BUFFER\_LOCATION`     |  Name of the SVM global memory interface found in the `board\_spec.xml*` * <br>  "" : `p520\_hpc\_sg280l`, `host`: `pac\_s10\_usm`                 |                                      | `host`                        |
| `CMAKE\_BUILD\_TYPE`        | Specify the build type                                                                                                                             | `Debug`                              | `Release`, `RelWithDebInfo`   |

### Additional Kernel Builds

Generation of Intel OpenCL Offline Compiler reports

```bash
make <kernel_name>_report
make fft1d_report
```

### Using the GNU Debugger (gdb) with the Debug builds

CMake debug builds lets you step through code using gdb.

`gdb --args ./fft -n 64 -d 2 -p <path-to-bitstream>`

## Runtime Input Parameters

```bash
Offloading FFT on FPGA
Usage:
  ./fft<..> [OPTION...]

  -n, --num arg    Number of sample points in a dimension (default: 64)
  -d, --dim arg    Number of dimensions (default: 3)
  -b, --back       Toggle Backward FFT
  -i, --iter arg   Number of iterations (default: 1)
  -p, --path arg   Path to FPGA bitstream
  -y, --noverify   Toggle to not verify with FFTW
  -c, --batch arg  Number of batches of FFT calculations in FPGA (default: 1)
  -t, --burst      Toggle to use burst interleaved global memory accesses  in
                   FPGA
  -m, --use_bram   Toggle to use BRAM instead of DDR for 3D Transpose  
  -s, --use_usm    Toggle to use Unified Shared Memory features for data
                   transfers between host and device
  -e, --emulate    Toggle to enable emulation 
  -h, --help       Print usage
```

## Output Interpretation

The examples measure and output relevant performance metrics that are shown below:

```bash
------------------------------------------
FFT CONFIGURATION: 
--------------------------------------------
Type               : Complex to Complex
Points             : 64 
Direction          : Forward 
Placement          : In Place    
Batch              : 1 
Iterations         : 1 
Transpose3D        : DDR 
Burst Interleaving : No 
Emulation          : Yes 
USM Feature        : No 
--------------------------------------------
-- Initializing FPGA ...
-- 1 platforms found
	0: intel(r) fpga emulation platform for opencl(tm)
-- 1 devices found
	Choosing first device by default
-- Getting program binary from path: p520_hpc_sg280l/emulation/fft1d_64_nointer/fft1d.aocx
-- Building the program
0: Calculating FFT - 
-- Launching 1D FFT of 1 batches 
Launching FFT transform for 1 batch 
-- Copying data from host to device
-- Executing kernels
-- Transfering results back to host
-- Cleaning up FPGA resources ...
------------------------------------------
Measurements 
--------------------------------------------
PCIe Write          = 0.0000ms
Kernel Execution    = 0.0182ms
Kernel Exec/Batch   = 0.0182ms
PCIe Read           = 0.0000ms
Total               = 0.0182ms
Throughput          = 0.0982GFLOPS/s | 26.8213 GB/s
```

- `PCIe Write`: time taken in milliseconds to transfer data from host memory of the CPU to the global memory of the FPGA.

- `PCIe Read` : the time taken in milliseconds to transfer data from global memory of the FPGA to the host memory of the CPU.

- `Kernel Execution` : the time taken in milliseconds for the execution of the required kernels, which includes the global memory accesses.

- `Total` : `PCIe Write` + `Kernel Execution` + `PCIe Read`

- `Throughput` : $$ \frac{dim * 5 * N^{dim} * log_2 N}{runtime}$$