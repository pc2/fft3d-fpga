//  Author: Arjun Ramaswami

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
kernel void fetch1(global volatile float2 * restrict src) {
  const unsigned N = (1 << LOGN);

  for(unsigned k = 0; k < (N * N); k++){ 
    float2 buf[N];

    #pragma unroll 8
    for(unsigned i = 0; i < N; i++){
      buf[i & ((1<<LOGN)-1)] = src[(k << LOGN) + i];    
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

__attribute__((max_global_work_dim(0)))
kernel void transpose() {
  const unsigned N = (1 << LOGN);
  const unsigned DEPTH = (1 << (LOGN + LOGN - LOGPOINTS));

  // iterate over N 2d matrices
  for(unsigned k = 0 ; k < N; k++){

    // Buffer with width - 8 points, depth - (N*N / 8), banked column-wise
    float2 buf[DEPTH][POINTS];
      
    // iterate within a 2d matrix
    for(unsigned row = 0; row < N; row++){

      // Temporary buffer to rotate before filling the matrix
      //float2 rotate_in[POINTS];
      float2 buf_bitrev[N];

      // bit-reversed ordered input stored in normal order
      for(unsigned j = 0; j < (N / 8); j++){
        buf_bitrev[j] = read_channel_intel(chanoutfft1[0]);               // 0
        buf_bitrev[4 * N / 8 + j] = read_channel_intel(chanoutfft1[1]);   // 32
        buf_bitrev[2 * N / 8 + j] = read_channel_intel(chanoutfft1[2]);   // 16
        buf_bitrev[6 * N / 8 + j] = read_channel_intel(chanoutfft1[3]);   // 48
        buf_bitrev[N / 8 + j] = read_channel_intel(chanoutfft1[4]);       // 8
        buf_bitrev[5 * N / 8 + j] = read_channel_intel(chanoutfft1[5]);   // 40
        buf_bitrev[3 * N / 8 + j] = read_channel_intel(chanoutfft1[6]);   // 24
        buf_bitrev[7 * N / 8 + j] = read_channel_intel(chanoutfft1[7]);   // 54
      }

      /* For each outer loop iteration, N data items are processed.
       * These N data items should reside in N/8 rows in buf.
       * Each of this N/8 rows are rotated by 1
       * Considering BRAM is POINTS wide, rotations should wrap around at POINTS
       * row & (POINTS - 1)
       */
      unsigned rot = row & (POINTS - 1);

      // fill the POINTS wide row of the buffer each iteration
      // N/8 rows filled with the same rotation
      for(unsigned j = 0; j < N / 8; j++){
 
        // Bitreverse read from rotate_in
        float2 rotate_in[POINTS];

        #pragma unroll 8
        for(unsigned i = 0; i < 8; i++){
          rotate_in[i] = buf_bitrev[(j * POINTS) + i];
        }

        // Rotate write into buffer
        #pragma unroll 8
        for(unsigned i = 0; i < 8; i++){
            unsigned where = ((i + POINTS) - rot) & (POINTS - 1);
            unsigned buf_row = (row * (N / 8)) + j;
            buf[buf_row][i] = rotate_in[where];
        }
      }
    }

    for(unsigned row = 0; row < N; row++){

      float2 rotate_out[N];
      unsigned offset = 0;            

      #pragma unroll 8
      for(unsigned j = 0; j < N; j++){
        unsigned rot = (DEPTH + j - row) << (LOGN - LOGPOINTS) & (DEPTH -1);
        unsigned offset = row >> LOGPOINTS;
        unsigned row_rotate = offset + rot;
        unsigned col_rotate = j & (POINTS - 1);

        rotate_out[j] = buf[row_rotate][col_rotate];
      }

      for(unsigned j = 0; j < N / 8; j++){
        //unsigned rev = bit_reversed((j * POINTS), LOGN);
        unsigned rev = j;
        unsigned rot_out = row & (N - 1);
        
        unsigned chan0 = (rot_out + rev) & (N - 1);                 // 0
        unsigned chan1 = ((4 * N / 8) + rot_out + rev) & (N - 1);  // 32
        unsigned chan2 = ((2 * N / 8) + rot_out + rev) & (N - 1);  // 16
        unsigned chan3 = ((6 * N / 8) + rot_out + rev) & (N - 1);  // 48
        unsigned chan4 = ((N / 8) + rot_out + rev) & (N - 1);       // 8
        unsigned chan5 = ((5 * N / 8) + rot_out + rev) & (N - 1);  // 40
        unsigned chan6 = ((3 * N / 8) + rot_out + rev) & (N - 1);  // 24
        unsigned chan7 = ((7 * N / 8) + rot_out + rev) & (N - 1);  // 56

        write_channel_intel(chaninfft2[0], rotate_out[chan0]);    // 0
        write_channel_intel(chaninfft2[1], rotate_out[chan1]);   // 32
        write_channel_intel(chaninfft2[2], rotate_out[chan2]);   // 16
        write_channel_intel(chaninfft2[3], rotate_out[chan3]);   // 48
        write_channel_intel(chaninfft2[4], rotate_out[chan4]);   // 8
        write_channel_intel(chaninfft2[5], rotate_out[chan5]);   // 40
        write_channel_intel(chaninfft2[6], rotate_out[chan6]);   // 24
        write_channel_intel(chaninfft2[7], rotate_out[chan7]);   // 54
      }
    } // row

  } // iter matrices
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
kernel void store1(global volatile float2 * restrict dest){
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
        dest[ddr_loc + xdim] = buf[(xdim * N) + revcolt];
      }
    }
  } // stored N*N*N points in DDR
}

// Kernel that fetches data from global memory 
kernel void fetch2(global volatile float2 * restrict src) {
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
        buf[(i * N) + xdim] = src[ddr_loc + xdim];
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
kernel void store2(global float2 * restrict dest){

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
        dest[ddr_loc + xdim] = buf[buf_loc];
      }
    } // stored 2d buffer to ddr

  } // stored entire 3d points to ddr
}
