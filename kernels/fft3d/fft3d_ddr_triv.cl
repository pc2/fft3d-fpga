// Author: Arjun Ramaswami

#include "fft_8.cl" 

// Source the log(size) (log(1k) = 10) from a header shared with the host code
#include "fft_config.h"

#pragma OPENCL EXTENSION cl_intel_channels : enable
channel float2 chaninfft1[8] __attribute__((depth(8)));
channel float2 chanoutfft1[8] __attribute__((depth(8)));

channel float2 chaninfft2[8] __attribute__((depth(8)));
channel float2 chanoutfft2[8] __attribute__((depth(8)));

channel float2 chaninfft3[8] __attribute__((depth(8)));
channel float2 chanoutfft3[8] __attribute__((depth(8)));

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

// Kernel that fetches data from global memory 
kernel void fetch1(global volatile float2 * restrict src1) {
  const unsigned N = (1 << LOGN);

  for(unsigned k = 0; k < (N * N); k++){ 
    float2 buf[N];

    #pragma unroll 8
    for(unsigned i = 0; i < N; i++){
      buf[i & ((1<<LOGN)-1)] = src1[(k << LOGN) + i];    
    }

    for(unsigned j = 0; j < (N / 8); j++){
      write_channel_intel(chaninfft1[0], buf[j]);               // 0
      write_channel_intel(chaninfft1[1], buf[4 * N / 8 + j]);   // 32
      write_channel_intel(chaninfft1[2], buf[2 * N / 8 + j]);   // 16
      write_channel_intel(chaninfft1[3], buf[6 * N / 8 + j]);   // 48
      write_channel_intel(chaninfft1[4], buf[N / 8 + j]);       // 8
      write_channel_intel(chaninfft1[5], buf[5 * N / 8 + j]);   // 40
      write_channel_intel(chaninfft1[6], buf[3 * N / 8 + j]);   // 24
      write_channel_intel(chaninfft1[7], buf[7 * N / 8 + j]);   // 54
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
  for( int j = 0; j < N; j++){

      // needs to run "N / 8 - 1" additional iterations to drain the last outputs
      for (unsigned i = 0; i < N * (N / POINTS) + N / POINTS - 1; i++) {
        float2x8 data;

        // Read data from channels
        if (i < N * (N / POINTS)) {
          data.i0 = read_channel_intel(chaninfft1[0]);
          data.i1 = read_channel_intel(chaninfft1[1]);
          data.i2 = read_channel_intel(chaninfft1[2]);
          data.i3 = read_channel_intel(chaninfft1[3]);
          data.i4 = read_channel_intel(chaninfft1[4]);
          data.i5 = read_channel_intel(chaninfft1[5]);
          data.i6 = read_channel_intel(chaninfft1[6]);
          data.i7 = read_channel_intel(chaninfft1[7]);
        } 
        else {
          data.i0 = data.i1 = data.i2 = data.i3 = 
                    data.i4 = data.i5 = data.i6 = data.i7 = 0;
        }

        // Perform one FFT step
        data = fft_step(data, i % (N / POINTS), fft_delay_elements, inverse, LOGN);

        // Write result to channels
        if (i >= N / POINTS - 1) {
          write_channel_intel(chanoutfft1[0], data.i0);
          write_channel_intel(chanoutfft1[1], data.i1);
          write_channel_intel(chanoutfft1[2], data.i2);
          write_channel_intel(chanoutfft1[3], data.i3);
          write_channel_intel(chanoutfft1[4], data.i4);
          write_channel_intel(chanoutfft1[5], data.i5);
          write_channel_intel(chanoutfft1[6], data.i6);
          write_channel_intel(chanoutfft1[7], data.i7);
        }
      }
   }
}

// Transposes fetched data; stores them to global memory
kernel void transpose(){

  const unsigned N = (1 << LOGN);
  unsigned revcolt, where, where_write;

  local float2 buf[N * N];

  // Perform N times N*N transpositions and transfers
  for(unsigned p = 0; p < N; p++){

    for(unsigned i = 0; i < N; i++){
      for(unsigned k = 0; k < (N / 8); k++){
        where = ((i << LOGN) + (k << LOGPOINTS));

        buf[where + 0] = read_channel_intel(chanoutfft1[0]);
        buf[where + 1] = read_channel_intel(chanoutfft1[1]);
        buf[where + 2] = read_channel_intel(chanoutfft1[2]);
        buf[where + 3] = read_channel_intel(chanoutfft1[3]);
        buf[where + 4] = read_channel_intel(chanoutfft1[4]);
        buf[where + 5] = read_channel_intel(chanoutfft1[5]);
        buf[where + 6] = read_channel_intel(chanoutfft1[6]);
        buf[where + 7] = read_channel_intel(chanoutfft1[7]);
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

/*
 * Input through channels in bit reversed format
 */
__kernel 
void store1(__global __attribute__((buffer_location(BUFFER_LOCATION))) volatile float2 * restrict dest1){ 
            
  const unsigned N = (1 << LOGN);
  local float2 buf[N * N];

  for(unsigned zdim = 0; zdim < N; zdim++){

    //  Store yx plane in buffer, ydim in bit reversed format
    for(unsigned xdim = 0; xdim < N; xdim++){
      for(unsigned ydim = 0; ydim < (N / 8); ydim++){
        unsigned where = ((xdim * N) + (ydim * POINTS));
        
        buf[where + 0] = read_channel_intel(chanoutfft2[0]);
        buf[where + 1] = read_channel_intel(chanoutfft2[1]);
        buf[where + 2] = read_channel_intel(chanoutfft2[2]);
        buf[where + 3] = read_channel_intel(chanoutfft2[3]);
        buf[where + 4] = read_channel_intel(chanoutfft2[4]);
        buf[where + 5] = read_channel_intel(chanoutfft2[5]);
        buf[where + 6] = read_channel_intel(chanoutfft2[6]);
        buf[where + 7] = read_channel_intel(chanoutfft2[7]);
      }
    } // stored yx plane in buffer

    for(unsigned ydim = 0; ydim < N; ydim++){
      // bit reverse rows / ydim to get back normal order
      unsigned revcolt = bit_reversed(ydim, LOGN);

      unsigned ddr_loc = (zdim * N * N) + (ydim * N);
      
      #pragma unroll 8
      for( unsigned xdim = 0; xdim < N; xdim++){
        dest1[ddr_loc + xdim] = buf[(xdim * N) + revcolt];
      }
    }
  } // stored N*N*N points in DDR
}

// Kernel that fetches data from global memory 
__kernel
void fetch2(__global __attribute__((buffer_location(BUFFER_LOCATION))) volatile float2 * restrict src2){
     
  const unsigned N = (1 << LOGN);
  local float2 buf[N * N];

  for(unsigned ydim = 0; ydim < N; ydim++){
    /*
     * Store xz plane in the buffer
     */
    for(unsigned i = 0; i < N; i++){
      unsigned ddr_loc = ( (i * N * N) + (ydim * N) );

      #pragma unroll 8
      for(unsigned xdim = 0; xdim < N; xdim++){
        buf[(i * N) + xdim] = src2[ddr_loc + xdim];
      }
    }

    /* Transpose xz plane i.e. zx
     * Transfer bit reverse input to FFT
     */
    for(unsigned i = 0; i < N; i++){

      for(unsigned k = 0; k < (N / 8); k++){
        unsigned where = i + (k * N);

        write_channel_intel(chaninfft3[0], buf[where]);               // 0
        write_channel_intel(chaninfft3[1], buf[where + 4 * (N / 8) * N]); // 32
        write_channel_intel(chaninfft3[2], buf[where + 2 * (N / 8) * N]); // 16
        write_channel_intel(chaninfft3[3], buf[where + 6 * (N / 8) * N]); // 48
        write_channel_intel(chaninfft3[4], buf[where + (N / 8) * N]);     // 8
        write_channel_intel(chaninfft3[5], buf[where + 5 * (N / 8) * N]); // 40
        write_channel_intel(chaninfft3[6], buf[where + 3 * (N / 8) * N]); // 24
        write_channel_intel(chaninfft3[7], buf[where + 7 * (N / 8) * N]); // 54
      }
    }
  } // y axis
}

/*
 * Input and output data in bit-reversed format
 */
kernel void fft3dc(int inverse) {
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
          data.i0 = read_channel_intel(chaninfft3[0]);
          data.i1 = read_channel_intel(chaninfft3[1]);
          data.i2 = read_channel_intel(chaninfft3[2]);
          data.i3 = read_channel_intel(chaninfft3[3]);
          data.i4 = read_channel_intel(chaninfft3[4]);
          data.i5 = read_channel_intel(chaninfft3[5]);
          data.i6 = read_channel_intel(chaninfft3[6]);
          data.i7 = read_channel_intel(chaninfft3[7]);
        } else {
          data.i0 = data.i1 = data.i2 = data.i3 = 
                    data.i4 = data.i5 = data.i6 = data.i7 = 0;
        }

        // Perform one FFT step
        data = fft_step(data, i % (N / POINTS), fft_delay_elements, inverse, LOGN);

        // Write result to channels
        if (i >= N / POINTS - 1) {
          write_channel_intel(chanoutfft3[0], data.i0);
          write_channel_intel(chanoutfft3[1], data.i1);
          write_channel_intel(chanoutfft3[2], data.i2);
          write_channel_intel(chanoutfft3[3], data.i3);
          write_channel_intel(chanoutfft3[4], data.i4);
          write_channel_intel(chanoutfft3[5], data.i5);
          write_channel_intel(chanoutfft3[6], data.i6);
          write_channel_intel(chanoutfft3[7], data.i7);
        }
      }
   }
}

/*
 * input through channels: transformed zx planes
 *  - values in the z axis is in bitreversed format
 */ 
kernel void store2(global float2 * restrict dest2){

  const unsigned N = (1 << LOGN);

  local float2 buf[N * N];

  for(unsigned ydim = 0; ydim < N; ydim++){

    /*
     * Store zx plane in 2d buffer in bit reversed format
     *  - outer loop iterates rows
     *  - inner loop stores elements of each row / zdim in bursts of POINTS (8)
     */
    for(unsigned xdim = 0; xdim < N; xdim++){
      for(unsigned zdim = 0; zdim < (N / 8); zdim++){

        // xdim * N iterates through the 2nd dim, here x
        unsigned where = ((xdim * N) + (zdim * POINTS));
        
        buf[where + 0] = read_channel_intel(chanoutfft3[0]);
        buf[where + 1] = read_channel_intel(chanoutfft3[1]);
        buf[where + 2] = read_channel_intel(chanoutfft3[2]);
        buf[where + 3] = read_channel_intel(chanoutfft3[3]);
        buf[where + 4] = read_channel_intel(chanoutfft3[4]);
        buf[where + 5] = read_channel_intel(chanoutfft3[5]);
        buf[where + 6] = read_channel_intel(chanoutfft3[6]);
        buf[where + 7] = read_channel_intel(chanoutfft3[7]);

      }
    } // zx plane stored in buffer

    /*
     * Transpose and bitreverse the zx plane in 2d buffer to xz,
     *  then store in global memory
     *  - outer loop iterates through the rows / zdim  
     *  - inner loop iterates through each column
     *    - selects elements based from bit reversed indices
     */
    for(unsigned zdim = 0; zdim < N; zdim++){

      // write to ddr in planes of xz
      unsigned ddr_loc = ( (ydim * N) + (zdim * N * N) );

      /*
       * Read column-wise in buffer as a transpose of zx to xz plane
       * store in ddr row-wise (xdim) then zdim
       *  1. bit reverse z axis - revcolt(z)
       *  2. transpose zx to xz - xdim * N
       *  : combine both to read the bitreversed column directly - buf_loc
       */
      unsigned revcolt = bit_reversed(zdim, LOGN);

      #pragma unroll 8
      for(unsigned xdim = 0; xdim < N; xdim++){
        unsigned buf_loc = revcolt + (xdim * N);
        dest2[ddr_loc + xdim] = buf[buf_loc];
      }
    } // stored 2d buffer to ddr

  } // stored entire 3d points to ddr
}
