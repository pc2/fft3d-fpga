# Modelling the performance of FFT3d

Performance of FFT3d is dependent on the performance of its building block the FFT1d kernels.

## Modelling Throughput of FFT1d Kernel

Floating point operations in the kernel:

1. Butterfly

    8 complex floating points as input.
    Performs 8 floating point additions and 8 floating point subtractions.
    For an N point FFT, there are logN stages that the N points require for an FFT. Each stage performs an 8 point butterfly. Therefore, for N points, there are **logN** butterflies. This makes

        logN * 16 flops for butterflies


2. Complex Rotations or multiplications with twiddle factors

    8 complex floating points as input.
    Performs 4 dot products per multiplication with twiddle factor.
    There are a total of 6 multiplications because point 0 and N / 2 multiply by 1. There are **logN - 1 / 2** complex rotates. This makes

        floor(logN - 1 / 2) * 6

Total number of floating point operations performed by N point FFT

        num_flops = (logN * 16) + floor((logN - 1) / 2) * 6 

Considering these stages are pipelined, for a given frequency

        throughput = num_flops * clock_freq
                  = ((logN * 16) + floor((logN - 1) / 2) * 6)  * clock_freq

#### Number of DSPs used for different FFT sizes.

| FFT Size | # DSPs |
|:--------:|:------:|
|    16    |   88   |
|    32    |   128  |
|    64    |   144  |
|    128   |   184  |
|    256   |   200  |

#### Throughput

Modelled for clock frequency of 467 MHz assuming hyperflex is turned on. 

![Throughput for different FFT1d Sizes](common/fft1d_throughput.png)

TODO: Is a dot product considered as 3 flops?

**Note**: The FFT1d building block kernel used to implement FFT3d is the one provided in the Intel's design samples. The input and the output to the FFT1d kernel are both in bit-reversed order, the latter entails the need to perform another bit reversal to obtain the standard FFT output.
