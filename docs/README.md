# Modelling the performance of FFT3d

Performance of FFT3d is dependent on the performance of its building block the FFT1d kernels.

## Modelling FFT1d

The FFT1d building block kernel used to implement FFT3d is the one provided in the Intel's design samples. The input and the output to the FFT1d kernel are both in bit-reversed order, the latter entails the need to perform another bit reversal to obtain the standard FFT output.
