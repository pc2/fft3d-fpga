/******************************************************************************
 *  Authors: Arjun Ramaswami
 *****************************************************************************/

#ifndef FFT_FPGA_H
#define FFT_FPGA_H

// Initialize FPGA
int fpga_initialize_();

// Finalize FPGA
void fpga_final_();

// Single precision FFT3d procedure
double fpga_fft3d_sp_(int direction, int N[3], cmplx *din);

// Double precision FFT3d procedure
double fpga_fft3d_dp_(int direction, int N[3], cmplx *din);

// Check fpga bitstream present in directory
int fpga_check_bitstream_(char *data_path, int N[3]);
#endif
