// Author: Arjun Ramaswami

#ifndef FFTFPGA_H
#define FFTFPGA_H

typedef struct {
  float x;
  float y;
} float2;

typedef struct {
  double x;
  double y;
} double2;

typedef struct fpga_timing {
  double pcie_read_t;
  double pcie_write_t;
  double exec_t;
  int valid;
} fpga_t;

// Initialize FPGA
extern int fpga_initialize(const char *platform_name, const char *path, int use_svm, int use_emulator);

// Finalize FPGA
extern void fpga_final();

// Double precision complex memory allocation
extern void* fftfpga_complex_malloc(size_t sz, int svm);

// Single precision complex memory allocation
extern void* fftfpgaf_complex_malloc(size_t sz, int svm);

// Double Precision 1d FFT
extern fpga_t fftfpga_c2c_1d(int N, double2 *inp, double2 *out, int inv, int iter);

// Single Precision 1d FFT
extern fpga_t fftfpgaf_c2c_1d(int N, float2 *inp, float2 *out, int inv, int iter);

// Single Precision 2d FFT using BRAM
extern fpga_t fftfpgaf_c2c_2d_bram(int N, float2 *inp, float2 *out, int inv);

// Single Precision 2d FFT using DDR
extern fpga_t fftfpgaf_c2c_2d_ddr(int N, float2 *inp, float2 *out, int inv);

// Single Precision in BRAM 3d FFT
extern fpga_t fftfpgaf_c2c_3d_bram(int N, float2 *inp, float2 *out, int inv);

// Single Precision in DDR 3d FFT
extern fpga_t fftfpgaf_c2c_3d_ddr(int N, float2 *inp, float2 *out, int inv);

#endif
