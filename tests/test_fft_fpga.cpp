/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

#include "gtest/gtest.h"  // finds this because gtest is linked

extern "C" {
#ifdef USE_FFTW
  #include <fftw3.h>
#endif
  #include "CL/opencl.h"
  #include "../src/host/include/fftfpga.h"
  #include "../src/host/include/helper.h"
  #include <math.h>
}

class fftFPGATest : public :: testing :: Test {

  void SetUp(){}
  void TearDown(){
    //fpga_final();
  }

  protected:
};

/**
 * \brief fpga_initialize()
 */
TEST_F(fftFPGATest, ValidInit){
  // empty path argument
  EXPECT_EQ(fpga_initialize("Intel(R) FPGA", "", 0, 1), 1);
  // wrong path argument
  EXPECT_EQ(fpga_initialize("Intel(R) FPGA", "TEST", 0, 1), 1);
  // wrong platform name
  EXPECT_EQ(fpga_initialize("TEST", "fft1d_emulate.aocx", 0, 1), 1);
  // right path and platform names
  EXPECT_EQ(fpga_initialize("Intel(R) FPGA", "64pt_fft1d_emulate.aocx", 0, 1), 0);
  fpga_final();
}

/**
 * \brief fftfpga_complex_malloc()
 */
TEST_F(fftFPGATest, ValidDpMalloc){
  // request zero size
  EXPECT_EQ(fftfpga_complex_malloc(0, 0), nullptr);
  // TODO: do not support svm
  EXPECT_EQ(fftfpga_complex_malloc(0, 1), nullptr);
}

/**
 * \brief fftfpgaf_complex_malloc()
 */
TEST_F(fftFPGATest, ValidSpMalloc){
  // request zero size
  EXPECT_EQ(fftfpgaf_complex_malloc(0, 0), nullptr);
  // TODO: do not support svm
  EXPECT_EQ(fftfpgaf_complex_malloc(0, 1), nullptr);
}

/**
 * \brief fftfpgaf_c2c_1d()
 */
TEST_F(fftFPGATest, ValidSp1dFFT){
  int logN = 6;
  int N = (1 << 6);

  size_t sz = sizeof(float2) * N;
  float2 *inp = (float2*)fftfpgaf_complex_malloc(sz, 0);
  float2 *out = (float2*)fftfpgaf_complex_malloc(sz, 0);
  // null inp ptr input
  fpga_t fft_time = fftfpgaf_c2c_1d(64, NULL, out, 0, 1);
  EXPECT_EQ(fft_time.valid, 0);

  // null out ptr input
  fft_time = fftfpgaf_c2c_1d(64, inp, NULL, 0, 1);
  EXPECT_EQ(fft_time.valid, 0);

  // if N not a power of 2
  fft_time = fftfpgaf_c2c_1d(63, inp, out, 0, 1);
  EXPECT_EQ(fft_time.valid, 0);

  // check correctness of output
#ifdef USE_FFTW
  // malloc data to input
  fftf_create_data(inp, N);

  fpga_initialize("Intel(R) FPGA", "64pt_fft1d_emulate.aocx", 0, 1);
  fft_time = fftfpgaf_c2c_1d(64, inp, out, 0, 1);

  fftwf_complex* fftw_inp = (fftwf_complex*)fftwf_alloc_complex(sz);
  fftwf_complex* fftw_out = (fftwf_complex*)fftwf_alloc_complex(sz);
  fftwf_plan plan = fftwf_plan_dft_1d( N, &fftw_inp[0], &fftw_out[0], FFTW_FORWARD, FFTW_ESTIMATE);

  float2 *temp = (float2 *)fftfpgaf_complex_malloc(sz, 0);

  for (int i = 0; i < N; i++){
    temp[i] = out[i];
  }
  for (int i = 0; i < N; i++) {
    int fwd = i;
    int bit_rev = 0;
    for (int j = 0; j < logN; j++) {
        bit_rev <<= 1;
        bit_rev |= fwd & 1;
        fwd >>= 1;
    }
    out[i] = temp[bit_rev];
  }

  for(int i = 0; i < N; i++){
    fftw_inp[i][0] = inp[i].x;
    fftw_inp[i][1] = inp[i].y;
  }

  fftwf_execute(plan);

  double mag_sum = 0, noise_sum = 0;

  for (int i = 0; i < N; i++) {
    double magnitude = fftw_out[i][0] * fftw_out[i][0] + \
                      fftw_out[i][1] * fftw_out[i][1];
    double noise = (fftw_out[i][0] - out[i].x) \
        * (fftw_out[i][0] - out[i].x) + 
        (fftw_out[i][1] - out[i].y) * (fftw_out[i][1] - out[i].y);

    mag_sum += magnitude;
    noise_sum += noise;
  }
  double db = 10 * log(mag_sum / noise_sum) / log(10.0);
  ASSERT_GT(db, 120);
  EXPECT_GT(fft_time.exec_t, 0.0);
  EXPECT_EQ(fft_time.valid, 1);

  fftwf_free(fftw_inp);
  fftwf_free(fftw_out);
  fftwf_destroy_plan(plan);
  free(temp);
  fpga_final();
#endif

  free(inp);
  free(out);
}

/**
 * \brief fftfpgaf_c2c_2d_bram()
 */
TEST_F(fftFPGATest, ValidSp2dFFTBRAM){
  int N = (1 << 6);

  size_t sz = sizeof(float2) * N * N;
  float2 *inp = (float2*)fftfpgaf_complex_malloc(sz, 0);
  float2 *out = (float2*)fftfpgaf_complex_malloc(sz, 0);
  // null inp ptr input
  fpga_t fft_time = fftfpgaf_c2c_2d_bram(64, NULL, out, 0);
  EXPECT_EQ(fft_time.valid, 0);

  // null out ptr input
  fft_time = fftfpgaf_c2c_2d_bram(64, inp, NULL, 0);
  EXPECT_EQ(fft_time.valid, 0);

  // if N not a power of 2
  fft_time = fftfpgaf_c2c_2d_bram(63, inp, out, 0);
  EXPECT_EQ(fft_time.valid, 0);

  // check correctness of output
#ifdef USE_FFTW
  // malloc data to input
  fftf_create_data(inp, N * N);

  int test = fpga_initialize("Intel(R) FPGA", "64pt_fft2d_bram_emulate.aocx", 0, 1);
  ASSERT_NE(test, 1);

  fft_time = fftfpgaf_c2c_2d_bram(64, inp, out, 0);

  fftwf_complex* fftw_inp = (fftwf_complex*)fftwf_alloc_complex(sz);
  fftwf_complex* fftw_out = (fftwf_complex*)fftwf_alloc_complex(sz);

  fftwf_plan plan = fftwf_plan_dft_2d(N, N, &fftw_inp[0], &fftw_out[0], FFTW_FORWARD, FFTW_ESTIMATE);

  for(size_t i = 0; i < N * N; i++){
    fftw_inp[i][0] = inp[i].x;
    fftw_inp[i][1] = inp[i].y;
  }

  fftwf_execute(plan);

  double mag_sum = 0, noise_sum = 0;

  for (size_t i = 0; i < N * N; i++) {
    double magnitude = fftw_out[i][0] * fftw_out[i][0] + \
                      fftw_out[i][1] * fftw_out[i][1];
    double noise = (fftw_out[i][0] - out[i].x) \
        * (fftw_out[i][0] - out[i].x) + 
        (fftw_out[i][1] - out[i].y) * (fftw_out[i][1] - out[i].y);

    mag_sum += magnitude;
    noise_sum += noise;
  }
  double db = 10 * log(mag_sum / noise_sum) / log(10.0);
  ASSERT_GT(db, 120);
  EXPECT_GT(fft_time.exec_t, 0.0);
  EXPECT_EQ(fft_time.valid, 1);

  fftwf_free(fftw_inp);
  fftwf_free(fftw_out);
  fftwf_destroy_plan(plan);
  fpga_final();
#endif

  free(inp);
  free(out);
}

/**
 * \brief fftfpgaf_c2c_2d_ddr()
 */
TEST_F(fftFPGATest, ValidSp2dFFTDDR){
  int N = (1 << 6);

  size_t sz = sizeof(float2) * N * N;
  float2 *inp = (float2*)fftfpgaf_complex_malloc(sz, 0);
  float2 *out = (float2*)fftfpgaf_complex_malloc(sz, 0);
  // null inp ptr input
  fpga_t fft_time = fftfpgaf_c2c_2d_ddr(64, NULL, out, 0);
  EXPECT_EQ(fft_time.valid, 0);

  // null out ptr input
  fft_time = fftfpgaf_c2c_2d_ddr(64, inp, NULL, 0);
  EXPECT_EQ(fft_time.valid, 0);

  // if N not a power of 2
  fft_time = fftfpgaf_c2c_2d_ddr(63, inp, out, 0);
  EXPECT_EQ(fft_time.valid, 0);

  // check correctness of output
#ifdef USE_FFTW
  // malloc data to input
  fftf_create_data(inp, N * N);

  int test = fpga_initialize("Intel(R) FPGA", "64pt_fft2d_ddr_emulate.aocx", 0, 1);
  ASSERT_NE(test, 1);

  fft_time = fftfpgaf_c2c_2d_ddr(64, inp, out, 0);

  fftwf_complex* fftw_inp = (fftwf_complex*)fftwf_alloc_complex(sz);
  fftwf_complex* fftw_out = (fftwf_complex*)fftwf_alloc_complex(sz);

  fftwf_plan plan = fftwf_plan_dft_2d(N, N, &fftw_inp[0], &fftw_out[0], FFTW_FORWARD, FFTW_ESTIMATE);

  for(size_t i = 0; i < N * N; i++){
    fftw_inp[i][0] = inp[i].x;
    fftw_inp[i][1] = inp[i].y;
  }

  fftwf_execute(plan);

  double mag_sum = 0, noise_sum = 0;

  for (size_t i = 0; i < N * N; i++) {
    double magnitude = fftw_out[i][0] * fftw_out[i][0] + \
                      fftw_out[i][1] * fftw_out[i][1];
    double noise = (fftw_out[i][0] - out[i].x) \
        * (fftw_out[i][0] - out[i].x) + 
        (fftw_out[i][1] - out[i].y) * (fftw_out[i][1] - out[i].y);

    mag_sum += magnitude;
    noise_sum += noise;
  }
  double db = 10 * log(mag_sum / noise_sum) / log(10.0);
  ASSERT_GT(db, 120);
  EXPECT_GT(fft_time.exec_t, 0.0);
  EXPECT_EQ(fft_time.valid, 1);

  fftwf_free(fftw_inp);
  fftwf_free(fftw_out);
  fftwf_destroy_plan(plan);
  fpga_final();
#endif

  free(inp);
  free(out);
}

/**
 * \brief fftfpgaf_c2c_3d_bram()
 */
TEST_F(fftFPGATest, ValidSp3dBRAMFFT){
  int N = (1 << 6);

  size_t sz = sizeof(float2) * N * N * N;
  float2 *inp = (float2*)fftfpgaf_complex_malloc(sz, 0);
  float2 *out = (float2*)fftfpgaf_complex_malloc(sz, 0);
  // null inp ptr input
  fpga_t fft_time = fftfpgaf_c2c_3d_bram(64, NULL, out, 0);
  EXPECT_EQ(fft_time.valid, 0);

  // null out ptr input
  fft_time = fftfpgaf_c2c_3d_bram(64, inp, NULL, 0);
  EXPECT_EQ(fft_time.valid, 0);

  // if N not a power of 2
  fft_time = fftfpgaf_c2c_3d_bram(63, inp, out, 0);
  EXPECT_EQ(fft_time.valid, 0);

  // check correctness of output
#ifdef USE_FFTW
  // malloc data to input
  fftf_create_data(inp, N * N * N);

  int test = fpga_initialize("Intel(R) FPGA", "64pt_fft3d_bram_emulate.aocx", 0, 1);
  ASSERT_NE(test, 1);

  fft_time = fftfpgaf_c2c_3d_bram(64, inp, out, 0);

  fftwf_complex* fftw_inp = (fftwf_complex*)fftwf_alloc_complex(sz);
  fftwf_complex* fftw_out = (fftwf_complex*)fftwf_alloc_complex(sz);

  fftwf_plan plan = fftwf_plan_dft_3d(N, N, N, &fftw_inp[0], &fftw_out[0], FFTW_FORWARD, FFTW_ESTIMATE);

  for(int i = 0; i < (N * N * N); i++){
    fftw_inp[i][0] = inp[i].x;
    fftw_inp[i][1] = inp[i].y;
  }

  fftwf_execute(plan);

  double mag_sum = 0, noise_sum = 0;

  for (int i = 0; i < (N * N * N); i++) {
    double magnitude = fftw_out[i][0] * fftw_out[i][0] + \
                      fftw_out[i][1] * fftw_out[i][1];
    double noise = (fftw_out[i][0] - out[i].x) \
        * (fftw_out[i][0] - out[i].x) + 
        (fftw_out[i][1] - out[i].y) * (fftw_out[i][1] - out[i].y);

    mag_sum += magnitude;
    noise_sum += noise;
  }
  double db = 10 * log(mag_sum / noise_sum) / log(10.0);
  ASSERT_GT(db, 120);
  EXPECT_GT(fft_time.exec_t, 0.0);
  EXPECT_EQ(fft_time.valid, 1);

  fftwf_free(fftw_inp);
  fftwf_free(fftw_out);
  fftwf_destroy_plan(plan);
  fpga_final();
#endif

  free(inp);
  free(out);
}
/**
 * \brief fftfpgaf_c2c_3d_ddr()
 */
TEST_F(fftFPGATest, ValidSp3dFFTDDR){
  int N = (1 << 6);

  size_t sz = sizeof(float2) * N * N * N;
  float2 *inp = (float2*)fftfpgaf_complex_malloc(sz, 0);
  float2 *out = (float2*)fftfpgaf_complex_malloc(sz, 0);
  // null inp ptr input
  fpga_t fft_time = fftfpgaf_c2c_3d_ddr(64, NULL, out, 0);
  EXPECT_EQ(fft_time.valid, 0);

  // null out ptr input
  fft_time = fftfpgaf_c2c_3d_ddr(64, inp, NULL, 0);
  EXPECT_EQ(fft_time.valid, 0);

  // if N not a power of 2
  fft_time = fftfpgaf_c2c_3d_ddr(63, inp, out, 0);
  EXPECT_EQ(fft_time.valid, 0);

  // check correctness of output
#ifdef USE_FFTW
  // malloc data to input
  fftf_create_data(inp, N * N * N);

  int test = fpga_initialize("Intel(R) FPGA", "64pt_fft3d_ddr_emulate.aocx", 0, 1);
  ASSERT_NE(test, 1);

  fft_time = fftfpgaf_c2c_3d_ddr(64, inp, out, 0);

  fftwf_complex* fftw_inp = (fftwf_complex*)fftwf_alloc_complex(sz);
  fftwf_complex* fftw_out = (fftwf_complex*)fftwf_alloc_complex(sz);

  fftwf_plan plan = fftwf_plan_dft_3d(N, N, N, &fftw_inp[0], &fftw_out[0], FFTW_FORWARD, FFTW_ESTIMATE);

  for(int i = 0; i < (N * N * N); i++){
    fftw_inp[i][0] = inp[i].x;
    fftw_inp[i][1] = inp[i].y;
  }

  fftwf_execute(plan);

  double mag_sum = 0, noise_sum = 0;

  for (int i = 0; i < (N * N * N); i++) {
    double magnitude = fftw_out[i][0] * fftw_out[i][0] + \
                      fftw_out[i][1] * fftw_out[i][1];
    double noise = (fftw_out[i][0] - out[i].x) \
        * (fftw_out[i][0] - out[i].x) + 
        (fftw_out[i][1] - out[i].y) * (fftw_out[i][1] - out[i].y);

    //printf("%d : fftw[%d] = (%lf, %lf) fpga = (%lf, %lf) \n", i, i, fftw_out[i][0], fftw_out[i][1], out[i].x, out[i].y);
    mag_sum += magnitude;
    noise_sum += noise;
  }
  double db = 10 * log(mag_sum / noise_sum) / log(10.0);
  ASSERT_GT(db, 120);
  EXPECT_GT(fft_time.exec_t, 0.0);
  EXPECT_EQ(fft_time.valid, 1);

  fftwf_free(fftw_inp);
  fftwf_free(fftw_out);
  fftwf_destroy_plan(plan);
  fpga_final();
#endif

  free(inp);
  free(out);
}