//  Author: Arjun Ramaswami

#include "gtest/gtest.h"  // finds this because gtest is linked

extern "C" {
  #include "CL/opencl.h"
  #include "fftfpga/fftfpga.h"
  #include "helper.h"
  #include <math.h>

#ifdef USE_FFTW
  #include <fftw3.h>
#endif
}

/**
 * \brief fftfpgaf_c2c_1d()
 */
TEST(fft1dFPGATest, InputValidity){
  const int N = (1 << 6);

  size_t sz = sizeof(float2) * N;
  float2 *test = (float2*)malloc(sz);
  fpga_t fft_time = {0.0, 0.0, 0.0, 0};

  // null inp ptr input
  fft_time = fftfpgaf_c2c_1d(64, NULL, test, 0, 1);
  EXPECT_EQ(fft_time.valid, 0);

  // null out ptr input
  fft_time = fftfpgaf_c2c_1d(64, test, NULL, 0, 1);
  EXPECT_EQ(fft_time.valid, 0);

  // if N not a power of 2
  fft_time = fftfpgaf_c2c_1d(63, test, test, 0, 1);
  EXPECT_EQ(fft_time.valid, 0);

  free(test);
}

TEST(fft1dFPGATest, CorrectnessSp){
  // check correctness of output
#ifdef USE_FFTW
  const int logN = 6;
  int N = (1 << logN);

  size_t sz = sizeof(float2) * N;
  float2 *inp = (float2*)fftfpgaf_complex_malloc(sz, 0);
  float2 *out = (float2*)fftfpgaf_complex_malloc(sz, 0);

  // malloc data to input
  fftf_create_data(inp, N);

  int isInit= fpga_initialize("Intel(R) FPGA", "emu_64_fft1d/fft1d.aocx", 0, 1);
  ASSERT_EQ(isInit, 1);

  fpga_t fft_time = fftfpgaf_c2c_1d(64, inp, out, 0, 1);

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