//  Author: Arjun Ramaswami

#include "fft_8.cl" 

// Macros for the 8 point 1d FFT
#define LOGPOINTS 3
#define POINTS (1 << LOGPOINTS)

// Source the log(size) (log(1k) = 10) from a header shared with the host code
#include "../common/fft_config.h"

#pragma OPENCL EXTENSION cl_intel_channels : enable
channel float2 chaninfft[8] __attribute__((depth(8)));
channel float2 chanoutfft[8] __attribute__((depth(8)));
channel float2 chaninfft2[8] __attribute__((depth(8)));
channel float2 chanoutfft2[8] __attribute__((depth(8)));
channel float2 chaninfetch[8] __attribute__((depth(8)));


int bit_reversed(int x, int bits) {
  int y = 0;
  #pragma unroll 
  for (int i = 0; i < bits; i++) {
    y <<= 1;
    y |= x & 1;
    x >>= 1;
  }
  return y;
}

void sendTofft(float2 *buffer, const unsigned N, unsigned j){
  write_channel_intel(chaninfft[0], buffer[j]);               // 0
  write_channel_intel(chaninfft[1], buffer[4 * N / 8 + j]);   // 32
  write_channel_intel(chaninfft[2], buffer[2 * N / 8 + j]);   // 16
  write_channel_intel(chaninfft[3], buffer[6 * N / 8 + j]);   // 48
  write_channel_intel(chaninfft[4], buffer[N / 8 + j]);       // 8
  write_channel_intel(chaninfft[5], buffer[5 * N / 8 + j]);   // 40
  write_channel_intel(chaninfft[6], buffer[3 * N / 8 + j]);   // 24
  write_channel_intel(chaninfft[7], buffer[7 * N / 8 + j]);   // 54
}

// Kernel that fetches data from global memory 
kernel void fetch(global volatile float2 * restrict src) {
  const unsigned N = (1 << LOGN);

  for(unsigned k = 0; k < (1 << (LOGN + LOGN)); k++){ 

    float2 buf[N];
    #pragma unroll 8
    for(unsigned i = 0; i < N; i++){
      buf[i & ((1<<LOGN)-1)] = src[(k << LOGN) + i];    
    }

    for(unsigned j = 0; j < (N / 8); j++){
      sendTofft(&buf[0], N, j);
    }
  }

  for(unsigned k = 0; k < (1 << (LOGN+LOGN)); k++){ 

    float2 buf[N];
    for(unsigned i = 0; i < (N / 8); i++){

        #pragma unroll 8
        for(unsigned u = 0; u < 8; u++){
          buf[((i << LOGPOINTS) & ((1 << LOGN)-1)) + u] = read_channel_intel(chaninfetch[u]);
        }
    }

    for(unsigned j = 0; j < (N / 8); j++){
      sendTofft(&buf[0], N, j);
    }
  }
}

/* This single work-item task wraps the FFT engine
 * 'inverse' toggles between the direct and the inverse transform
 */

kernel void fft3da(int inverse) {
  const int N = (1 << LOGN);

  /* The FFT engine requires a sliding window for data reordering; data stored
   * in this array is carried across loop iterations and shifted by 1 element
   * every iteration; all loop dependencies derived from the uses of this 
   * array are simple transfers between adjacent array elements
   */

  float2 fft_delay_elements[N + POINTS * (LOGN - 2)];
  for( int j = 0; j < N * 2; j++){

      // needs to run "N / 8 - 1" additional iterations to drain the last outputs
      for (unsigned i = 0; i < N * (N / POINTS) + N / POINTS - 1; i++) {
        float2x8 data;

        // Read data from channels
        if (i < N * (N / POINTS)) {
          data.i0 = read_channel_intel(chaninfft[0]);
          data.i1 = read_channel_intel(chaninfft[1]);
          data.i2 = read_channel_intel(chaninfft[2]);
          data.i3 = read_channel_intel(chaninfft[3]);
          data.i4 = read_channel_intel(chaninfft[4]);
          data.i5 = read_channel_intel(chaninfft[5]);
          data.i6 = read_channel_intel(chaninfft[6]);
          data.i7 = read_channel_intel(chaninfft[7]);

        } else {
          data.i0 = data.i1 = data.i2 = data.i3 = 
                    data.i4 = data.i5 = data.i6 = data.i7 = 0;
        }

        // Perform one FFT step
        data = fft_step(data, i % (N / POINTS), fft_delay_elements, inverse, LOGN);

        // Write result to channels
        if (i >= N / POINTS - 1) {
          write_channel_intel(chanoutfft[0], data.i0);
          write_channel_intel(chanoutfft[1], data.i1);
          write_channel_intel(chanoutfft[2], data.i2);
          write_channel_intel(chanoutfft[3], data.i3);
          write_channel_intel(chanoutfft[4], data.i4);
          write_channel_intel(chanoutfft[5], data.i5);
          write_channel_intel(chanoutfft[6], data.i6);
          write_channel_intel(chanoutfft[7], data.i7);
        }
      }
   }
}

// Transposes fetched data; stores them to global memory
kernel void transpose(global float2 * restrict dest) {

  const unsigned N = (1 << LOGN);
  unsigned revcolt, where_read, where_write, where;

  local float2 buf[N * N];

  // Perform N times N*N transpositions and transfers
  for(unsigned p = 0; p < N; p++){

    for(unsigned i = 0; i < N; i++){
      for(unsigned k = 0; k < (N / 8); k++){
        where_read = ((i << LOGN) + (k << LOGPOINTS));

        #pragma unroll 8
        for( unsigned u = 0; u < 8; u++){
          buf[where_read + u] = read_channel_intel(chanoutfft[u]);
        }
      }
    }

    for(unsigned i = 0; i < N; i++){
      revcolt = bit_reversed(i, LOGN);

      for(unsigned k = 0; k < (N / 8); k++){
        where_write = ((k * N) + revcolt);

        write_channel_intel(chaninfft2[0], buf[where_write]);               // 0
        write_channel_intel(chaninfft2[1], buf[where_write + 4 * (N / 8) * N]);   // 32
        write_channel_intel(chaninfft2[2], buf[where_write + 2 * (N / 8) * N]);   // 16
        write_channel_intel(chaninfft2[3], buf[where_write + 6 * (N / 8) * N]);   // 48
        write_channel_intel(chaninfft2[4], buf[where_write + (N / 8) * N]);       // 8
        write_channel_intel(chaninfft2[5], buf[where_write + 5 * (N / 8) * N]);   // 40
        write_channel_intel(chaninfft2[6], buf[where_write + 3 * (N / 8) * N]);   // 24
        write_channel_intel(chaninfft2[7], buf[where_write + 7 * (N / 8) * N]);   // 54
      }
    }
  }

  for(unsigned p = 0; p < N; p++){

    for(unsigned i = 0; i < N; i++){
      for(unsigned j = 0; j < (N / 8); j++){
        where = ((i << LOGN) + (j << LOGPOINTS));
        
        #pragma unroll 8
        for(unsigned u = 0; u < 8; u++){
          buf[where + u] = read_channel_intel(chanoutfft[u]);
        }
      }
    }

    for(unsigned i = 0; i < N; i++){
      revcolt = bit_reversed(i, LOGN);
      where = ( (i << (LOGN + LOGN)) + (p << LOGN));

      #pragma unroll 8
      for( unsigned q = 0; q < N; q++){
        dest[where + q] = buf[(q << LOGN) + revcolt];
      }
    }

  }

}

kernel void fft3db(int inverse) {
  const int N = (1 << LOGN);

  /* The FFT engine requires a sliding window for data reordering; data stored
   * in this array is carried across loop iterations and shifted by 1 element
   * every iteration; all loop dependencies derived from the uses of this 
   * array are simple transfers between adjacent array elements
   */

  float2 fft_delay_elements[N + POINTS * (LOGN - 2)];
  for( int j = 0; j < N; j++){

      for (unsigned i = 0; i < N * (N / POINTS) + N / POINTS - 1; i++) {
        float2x8 data;

        // Read data from channels
        if (i < N * (N / POINTS)) {
          data.i0 = read_channel_intel(chaninfft2[0]);
          data.i1 = read_channel_intel(chaninfft2[1]);
          data.i2 = read_channel_intel(chaninfft2[2]);
          data.i3 = read_channel_intel(chaninfft2[3]);
          data.i4 = read_channel_intel(chaninfft2[4]);
          data.i5 = read_channel_intel(chaninfft2[5]);
          data.i6 = read_channel_intel(chaninfft2[6]);
          data.i7 = read_channel_intel(chaninfft2[7]);
        } else {
          data.i0 = data.i1 = data.i2 = data.i3 = 
                    data.i4 = data.i5 = data.i6 = data.i7 = 0;
        }

        // Perform one FFT step
        data = fft_step(data, i % (N / POINTS), fft_delay_elements, inverse, LOGN);

        // Write result to channels
        if (i >= N / POINTS - 1) {
          write_channel_intel(chanoutfft2[0], data.i0);
          write_channel_intel(chanoutfft2[1], data.i1);
          write_channel_intel(chanoutfft2[2], data.i2);
          write_channel_intel(chanoutfft2[3], data.i3);
          write_channel_intel(chanoutfft2[4], data.i4);
          write_channel_intel(chanoutfft2[5], data.i5);
          write_channel_intel(chanoutfft2[6], data.i6);
          write_channel_intel(chanoutfft2[7], data.i7);
        }
      }

   }
}

// Stores data for 3rd dim FFT 
kernel void transpose3d(){
  const unsigned N = (1 << LOGN);
  unsigned revcolt, where;
  unsigned where_test;

  local float2 buf_3d[N * N * N];
  local float2 buf[N * N];

  // perform N*N*N writes to buffer
  for(unsigned m = 0; m < N; m++){

    for(unsigned i = 0; i < N; i++){
      for(unsigned j = 0; j < (N / 8); j++){
        where = ((i << LOGN) + (j << LOGPOINTS));
        
        #pragma unroll 8
        for(unsigned u = 0; u < 8; u++){
          buf[where + u] = read_channel_intel(chanoutfft2[u]);
        }
      }
    }

    for(unsigned i = 0; i < N; i++){
      revcolt = bit_reversed(i, LOGN);
      where = (i << LOGN) + (m << (LOGN + LOGN));
      
      #pragma unroll 8
      for( unsigned u = 0; u < N; u++){
        buf_3d[where + u] = buf[(u << LOGN) + revcolt];
      }
    }

  }

  // Flush entire 3d buffer transposed through channels
  for(unsigned m = 0; m < N; m++){

    for(unsigned i = 0; i < N; i++){
      where = ((i << (LOGN + LOGN)) + ( m << LOGN));

      #pragma unroll 8
      for(unsigned u = 0; u < N; u++){
        buf[(i << LOGN) + u] = buf_3d[where + u];
      }
    }

    for( unsigned i = 0; i < N; i++){
      for( unsigned j = 0; j < (N / 8); j++){
        where = (j * N * 8) + i;
  
        #pragma unroll 8
        for(unsigned u = 0; u < 8; u++){
          write_channel_intel(chaninfetch[u], buf[where + (u << LOGN)]);
        }
      }
    }

  }

}