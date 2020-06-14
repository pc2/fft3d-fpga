//  Author: Arjun Ramaswami

#include "gtest/gtest.h"  // finds this because gtest is linked
#include <math.h>
#ifdef USE_FFTW
  #include <fftw3.h>
#endif

extern "C" {
  #include "CL/opencl.h"
  #include "fftfpga/fftfpga.h"
}

/**
 * \brief fpga_initialize()
 */
TEST(fftFPGASetupTest, ValidInit){
  // empty path argument
  EXPECT_EQ(fpga_initialize("Intel(R) FPGA", "", 0, 1), -1);

  // wrong platform name
  EXPECT_EQ(fpga_initialize("TEST", "fft1d_emulate.aocx", 0, 1), -2);

  // wrong path argument
  EXPECT_EQ(fpga_initialize("Intel(R) FPGA", "TEST", 0, 1), -4);

  // right path and platform names
  EXPECT_EQ(fpga_initialize("Intel(R) FPGA", "emu_64_fft3d_bram/fft3d_bram.aocx", 0, 1), 0);
  fpga_final();
}

/**
 * \brief fftfpga_complex_malloc()
 */
TEST(fftFPGASetupTest, ValidDpMalloc){
  // request zero size
  EXPECT_EQ(fftfpga_complex_malloc(0, 0), nullptr);
  // TODO: do not support svm
  EXPECT_EQ(fftfpga_complex_malloc(0, 1), nullptr);
}

/**
 * \brief fftfpgaf_complex_malloc()
 */
TEST(fftFPGASetupTest, ValidSpMalloc){
  // request zero size
  EXPECT_EQ(fftfpgaf_complex_malloc(0, 0), nullptr);
  // TODO: do not support svm
  EXPECT_EQ(fftfpgaf_complex_malloc(0, 1), nullptr);
}