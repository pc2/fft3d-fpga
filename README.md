# FFT3d for FPGAs

This repository contains the OpenCL implementation of FFT3d for Intel FPGAs. Currently tested only for Intel Arria 10 and Stratix 10 FPGAs.

Performance model for the application can be found [here](docs/3dfft_model.md).

```
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake -DCMAKE_BUILD_TYPE=Release ..

make
make fft1d_emulate
make fft2d_emulate
make test
```

batch mode
