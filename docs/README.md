# Modelling the performance of FFT3d

Performance of FFT3d is dependent on the performance of its building blocks - FFT1d kernels.

## Modelling Latency of FFT1d Kernel

The FFT1d kernel provided in the Intel Design Samples processes 8 complex points per cycle, which matches the data width of a single bank of DDR memory. Therefore, the kernel is memory bandwidth bound.

| FFT Size | Expected Latency (microsec)  |   Measured Latency (microsec)   |
|:--------:|:------------------:|:------------------:|
|    32    |      0.013         |                    |  
|    64    |      0.027         |     0.032          |
|    128   |      0.053         |    0.063           |
|    256   |      0.107         |    0.126           |

The empty value for the size 32 of the measured latency is because the code cannot be synthesized out-of-the-box for FFT values below 64 and needs refactoring.

### Measurement Details

- Latency measured as an average of 100000 iterations for every given FFT size.
- Kernel code is synthesized using Intel OpenCL SDK version 19.3 and the Nallatech BSP version 19.2.0_hpc.
- Host code is compiled using gcc 8.3.0.

## Modelling Throughput of FFT1d Kernel

FFT1d kernel modelled here can be found in the Intel OpenCL Design Samples. The design follows the radix 2<sup>2</sup> FFT architecture, which consists of the following:

1. logN radix-2 butterflies
2. trivial rotations at every even stage
3. non-trivial rotations at every odd stage. This is the twiddle factor multiplication computed after the stage's butterfly.
4. shuffling using shift registers

In order to calculate an N-point FFT, the design inputs 8 complex points per cycle in a bit reversed order. This requires `N / 8` cycles to store the N points into a shift register. Each point requires `logN` stages to complete the transformation. After a delay of `N / 8 - 1` cycles, 8 distinct complex points are output per cycle for `N / 8` cycles for the N transformed points.

### Butterfly

- FFT kernel performs radix-2 butterflies. Considering 8 points of input, 4 parallel butterflies are performed at every stage.

- There are `logN stages` i.e. `logN * 4` butterflies.

#### Radix-2 butterfly

- 2 complex floating points as input and output.

- Floating point addition of the complex points for the first output point and floating subtraction for the second. Considering these are complex points, this makes for 2 additions and 2 subtractions per butterfly.

Therefore, each stage has `4 * 2` floating point additions and `4 * 2` floating point subtractions, with a total of:

        logN * 8 floating point additions 
        logN * 8 floating point subtractions

### Complex Rotations

As mentioned above, every odd stage until `logN -1` stages perform a complex rotation. This involves multiplying the points with the twiddle factors. The twiddle factors are pre-calculated and stored; every stage looks up the value based on the stage and the index values. The number of complex rotations in a N-point FFT can be expressed using this formula:

![\lfloor((logN-1)/2)\rfloor](https://latex.codecogs.com/svg.latex?\lfloor((logN-1)/2)\rfloor)

In a complex rotation of 8 input points, every point except Point 0 and Point 4 are multiplied with distinct twiddle factors, since these points are multiplied by 1. Complex multiplication follows the formula:

![(x+yi)(a+bi)=(xa-yb)+(xb+ya)i](https://latex.codecogs.com/svg.latex?(x&plus;yi)(a&plus;bi)=(xa-yb)&plus;(xb&plus;ya)i)

Each complex multiplication comprises of 4 floating point multiplications, 1 floating point subtraction and 1 floating point addition as described in the code sample below.

        float2 comp_mult(float2 a, float2 b) {
                float2 res;
                res.x = a.x * b.x - a.y * b.y;
                res.y = a.x * b.y + a.y * b.x;
                return res;
        }

The number of floating point operations:

        floor((logN - 1) / 2) complex rot * 6 mult per rot * 6 flops per multiplication 
![\lfloor((logN-1)/2)*36\rfloor](https://latex.codecogs.com/svg.latex?\lfloor((logN-1)/2)*36\rfloor)

This can be mapped to two dot product computations of size 2. For the 6 complex multiplications in a complex rotation, this is a total of 12 dot products (of size 2). Each dot product is implemented by a specific hardened floating point dot product DSP. Considering the size of dot product is 2, 2 DSPs are required. The total number of DSPs required for all complex rotations of N point FFT can be mapped by the formula:

        floor((logN - 1) / 2) rot * 6 mult * 2 dot products * 2 size of a dot product

![\lfloor((logN-1)/2)*24\rfloor](https://latex.codecogs.com/svg.latex?\lfloor((logN-1)/2)*24\rfloor)
### DSP Usage

Total number of DSPs required by N-point FFT:

![(logN * 16)+\lfloor((logN - 1) / 2)\rfloor*24](https://latex.codecogs.com/svg.latex?(logN&space;*&space;16)&plus;\lfloor((logN&space;-&space;1)&space;/&space;2)\rfloor*24)

Estimating DSP required for different FFT sizes:

| FFT Size | # DSPs |
|:--------:|:------:|
|    16    |   88   |
|    32    |   128  |
|    64    |   144  |
|    128   |   184  |
|    256   |   200  |

### Throughput

Total number of floating point operations is a total of the number of butterflies and the number of complex rotations:

        num_flops = ((logN * 16) + floor((logN - 1) / 2) * 36) 

Considering logN stages are pipelined, for maximum frequency:

        throughput = num_flops * clock_freq
                  = ((logN * 16) + floor((logN - 1) / 2) * 6)  * clock_freq

Modelled for clock frequency of 467 MHz assuming hyperflex is turned on.

![Throughput for different FFT1d Sizes](common/fft1d_throughput.png)

**Note**: The FFT1d building block kernel used to implement FFT3d is the one provided in the Intel's design samples. The input and the output to the FFT1d kernel are both in bit-reversed order, the latter entails the need to perform another bit reversal to obtain the standard FFT output.

## Modelling Latency of FFT3d Kernel

Developing an OpenCL 3d FFT kernel design requires transferring N$^3$ points from the host CPU to the DDR (global) memory via the PCIe bus, transforming the data and finally, transferring the results back to the host CPU. Each stage mentioned incurs latency, therefore, these can be categorized into:

1. PCIe Latency to transfer data between Host CPU to DDR Memory.
2. DDR memory access and transfer latency
3. Kernel execution latency

### PCIe Latency

This can be modelled as (incomplete):

![\frac{datasize}{datarate} + PCIe_{latency}](https://latex.codecogs.com/svg.latex?\frac{datasize}{datarate}&plus;PCIe_{latency})

### DDR Memory access and transfer latency

Latency to acces DDR4 operating at 2400 MT/s incurs a latency of 240 cycles. (incomplete).

### Kernel Execution latency

The following figure illustrates the kernels of the 3d FFT. The expressions within the kernels denote the size of the buffers required to perform the operations specified by the name of the kernel. The arrows are connections between the kernels, which are created using channels. Buffering adds latency to the pipeline, which is indicated as number of cycles required across every kernel. In this model, only a single lane of the design is considered i.e., a kernel reads data from a bank of memory to stores into another forming a single lane of the pipeline.

<img src="common/fft3d_singlebank_latency.png" alt="FFT3d single lane model of latency"	title="FFT3d single lane model of latency" width="200" height="500" />

The 2d transposition requires $\frac{N*N}{8}$ cycles to buffer N$^2$ points and another $\frac{N*N}{8}$ to output them. This is because the pipeline processes 8 complex single precision floating points in every stage of the pipeline. This stage, however, adds only a latency of $\frac{N*N}{8}$; besides the first write to the buffer every other stage is overlapping with others.

The 3d transpose is not pipelined therefore has 2 distinct read and write phases, denoted by the dashes. Additional 2d transpositions are required as intermediate buffers before storing and loading from the 3d buffer that adds further $\frac{N*N}{8}$ cycles of latency each.

### Total Latency

The total latency in cycles, can therefore, be expressed using the following equations:

![L_{total}=L_{pcie\_load}+L_{kernel}+ L_{pcie\_store}](https://latex.codecogs.com/svg.latex?L_{total}=L_{pcie\_load}&plus;L_{kernel}&plus;L_{pcie\_store})

![L_{kernel}=L_{ddr\_fetch}+L_{pipeline}+L_{ddr\_store}](https://latex.codecogs.com/svg.latex?L_{kernel}=L_{ddr\_fetch}&plus;L_{pipeline}&plus;L_{ddr\_store})

![L_{pipeline}\approx\frac{2*N+2*N^{2}+N^{3} -12}{4}](https://latex.codecogs.com/svg.latex?L_{pipeline}\approx\frac{2*N&plus;2*N^{2}&plus;N^{3}&space;-12}{4})

This is a simplification of the delay contributed by the buffers as given in the figure. Assuming a clock frequency of 300 MHz, the pipeline latency can be estimated to be approximately:

|  N$^3$  | Latency$_{singlelane}$(ms) |
|:-----:|:-----------------------:|
| 32^3  | 0.029                   |
| 64^3  | 0.226                   |
| 128^3 | 1.776                   |
| 256^3 | 14.09                   |
