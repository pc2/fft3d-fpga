/******************************************************************************
 *  Authors: Arjun Ramaswami
 *****************************************************************************/

#ifndef FFTFPGA_H
#define FFTFPGA_H

typedef struct {
  double x;
  double y;
} float2;

typedef struct {
  float x;
  float y;
} double2;

typedef struct fpga_timing {
  double pcie_read_t;
  double pcie_write_t;
  double exec_t;
} fpga_t;

// Initialize FPGA
int fpga_initialize();

// Finalize FPGA
void fpga_final();

// Check fpga bitstream present in directory
int fpga_check_bitstream(char *data_path, int N[3]);

// Double Precision 1d FFT
fpga_t fftfpga_c2c_1d(int N, double2 *inp, int inv, int iter);

// Single Precision 1d FFT
fpga_t fftfpgaf_c2c_1d(int N, float2 *inp, int inv, int iter);

// Double Precision 2d FFT
fpga_t int fftfpga_c2c_2d(int N, double2 *inp, int inv);

// Single Precision 2d FFT
fpga_t int fftfpgaf_c2c_2d(int N, float2 *inp, int inv);

// Double Precision 3d FFT
fpga_t int fftfpga_c2c_3d(int N, double2 *inp, int inv);

// Single Precision 3d FFT
fpga_t int fftfpgaf_c2c_3d(int N, float2 *inp, int inv);

#endif
