/******************************************************************************
 *  Authors: Arjun Ramaswami
 *****************************************************************************/

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
} fpga_t;

// Initialize FPGA
extern int fpga_initialize(const char *platform_name, const char *path);

// Finalize FPGA
extern void fpga_final();

// Double Precision 1d FFT
extern fpga_t fftfpga_c2c_1d(int N, double2 *inp, double2 *out, int inv, int iter);

// Single Precision 1d FFT
extern fpga_t fftfpgaf_c2c_1d(int N, float2 *inp, float2 *out, int inv, int iter);

// Double Precision 2d FFT
extern fpga_t fftfpga_c2c_2d(int N, double2 *inp, double2 *out, int inv, int iter);

// Single Precision 2d FFT
extern fpga_t fftfpgaf_c2c_2d(int N, float2 *inp, double2 *out, int inv, int iter);

// Double Precision 3d FFT
extern fpga_t fftfpga_c2c_3d(int N, double2 *inp, double2 *out, int inv, int iter);

// Single Precision 3d FFT
extern fpga_t fftfpgaf_c2c_3d(int N, float2 *inp, double2 *out, int inv, int iter);

// Allocate host side buffers to be 64-byte aligned
extern void* alignedMalloc(size_t size);
#endif
