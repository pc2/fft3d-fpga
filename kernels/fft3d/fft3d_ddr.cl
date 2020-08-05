// Author: Arjun Ramaswami

#include "fft_config.h"
#include "fft_8.cl" 
#include "../matrixTranspose/diagonal_bitrev.cl"

#pragma OPENCL EXTENSION cl_intel_channels : enable

channel float2 chaninfft3da[POINTS] __attribute__((depth(POINTS)));
channel float2 chaninfft3db[POINTS] __attribute__((depth(POINTS)));
channel float2 chaninfft3dc[POINTS] __attribute__((depth(POINTS)));

channel float2 chaninTranspose[POINTS] __attribute__((depth(POINTS)));
channel float2 chaninTranStore1[POINTS] __attribute__((depth(POINTS)));
channel float2 chaninTranStore2[POINTS] __attribute__((depth(POINTS)));

// Kernel that fetches data from global memory 
kernel void fetchBitrev1(global volatile float2 * restrict src) {

  for(unsigned k = 0; k < (N * N); k++){ 
    float2 buf[N];

    #pragma unroll 8
    for(unsigned i = 0; i < N; i++){
      buf[i & ((1<<LOGN)-1)] = src[(k << LOGN) + i];    
    }

    for(unsigned j = 0; j < (N / 8); j++){
      write_channel_intel(chaninfft3da[0], buf[j]);               // 0
      write_channel_intel(chaninfft3da[1], buf[4 * N / 8 + j]);   // 32
      write_channel_intel(chaninfft3da[2], buf[2 * N / 8 + j]);   // 16
      write_channel_intel(chaninfft3da[3], buf[6 * N / 8 + j]);   // 48
      write_channel_intel(chaninfft3da[4], buf[N / 8 + j]);       // 8
      write_channel_intel(chaninfft3da[5], buf[5 * N / 8 + j]);   // 40
      write_channel_intel(chaninfft3da[6], buf[3 * N / 8 + j]);   // 24
      write_channel_intel(chaninfft3da[7], buf[7 * N / 8 + j]);   // 54
    }
  }
}

/* This single work-item task wraps the FFT engine
 * 'inverse' toggles between the direct and the inverse transform
 */

kernel void fft3da(int inverse) {

  /* The FFT engine requires a sliding window for data reordering; data stored
   * in this array is carried across loop iterations and shifted by 1 element
   * every iteration; all loop dependencies derived from the uses of this 
   * array are simple transfers between adjacent array elements
   */

  float2 fft_delay_elements[N + POINTS * (LOGN - 2)];

  #pragma loop_coalesce
  for(unsigned j = 0; j < N; j++){
    for (unsigned i = 0; i < N * (N / POINTS) + N / POINTS - 1; i++) {
      float2x8 data;

      if (i < N * (N / POINTS)) {
        data.i0 = read_channel_intel(chaninfft3da[0]);
        data.i1 = read_channel_intel(chaninfft3da[1]);
        data.i2 = read_channel_intel(chaninfft3da[2]);
        data.i3 = read_channel_intel(chaninfft3da[3]);
        data.i4 = read_channel_intel(chaninfft3da[4]);
        data.i5 = read_channel_intel(chaninfft3da[5]);
        data.i6 = read_channel_intel(chaninfft3da[6]);
        data.i7 = read_channel_intel(chaninfft3da[7]);
      } 
      else {
        data.i0 = data.i1 = data.i2 = data.i3 = 
                  data.i4 = data.i5 = data.i6 = data.i7 = 0;
      }

      data = fft_step(data, i % (N / POINTS), fft_delay_elements, inverse, LOGN);

      // Write result to channels
      if (i >= N / POINTS - 1) {
        write_channel_intel(chaninTranspose[0], data.i0);
        write_channel_intel(chaninTranspose[1], data.i1);
        write_channel_intel(chaninTranspose[2], data.i2);
        write_channel_intel(chaninTranspose[3], data.i3);
        write_channel_intel(chaninTranspose[4], data.i4);
        write_channel_intel(chaninTranspose[5], data.i5);
        write_channel_intel(chaninTranspose[6], data.i6);
        write_channel_intel(chaninTranspose[7], data.i7);
      }
    }
  }
}

__attribute__((max_global_work_dim(0)))
kernel void transpose() {
  const int DELAY = (1 << (LOGN - LOGPOINTS)); // N / 8
  bool is_bufA = false, is_bitrevA = false;

  float2 buf[2][DEPTH][POINTS];
  float2 bitrev_in[2][N], bitrev_out[2][N] ;
  //float2 bitrev_in[2][N] __attribute__((memory("MLAB")));
  
  int initial_delay = DELAY + DELAY; // for each of the bitrev buffer

  // additional iterations to fill the buffers
  for(int step = -initial_delay; step < ((N * DEPTH) + DEPTH); step++){

    float2x8 data, data_out;
    if (step < ((N * DEPTH) - initial_delay)) {
      data.i0 = read_channel_intel(chaninTranspose[0]);
      data.i1 = read_channel_intel(chaninTranspose[1]);
      data.i2 = read_channel_intel(chaninTranspose[2]);
      data.i3 = read_channel_intel(chaninTranspose[3]);
      data.i4 = read_channel_intel(chaninTranspose[4]);
      data.i5 = read_channel_intel(chaninTranspose[5]);
      data.i6 = read_channel_intel(chaninTranspose[6]);
      data.i7 = read_channel_intel(chaninTranspose[7]);
    } else {
      data.i0 = data.i1 = data.i2 = data.i3 = 
                data.i4 = data.i5 = data.i6 = data.i7 = 0;
    }

    // Swap buffers every N*N/8 iterations 
    // starting from the additional delay of N/8 iterations
    is_bufA = (( (step + DELAY) & (DEPTH - 1)) == 0) ? !is_bufA: is_bufA;

    // Swap bitrev buffers every N/8 iterations
    is_bitrevA = ( (step & ((N / 8) - 1)) == 0) ? !is_bitrevA: is_bitrevA;

    unsigned row = step & (DEPTH - 1);
    data = bitreverse_in(data,
      is_bitrevA ? bitrev_in[0] : bitrev_in[1], 
      is_bitrevA ? bitrev_in[1] : bitrev_in[0], 
      row);

    writeBuf(data,
      is_bufA ? buf[0] : buf[1],
      step, DELAY);

    data_out = readBuf(
      is_bufA ? buf[1] : buf[0], 
      step);

    unsigned start_row = (step + DELAY) & (DEPTH -1);
    data_out = bitreverse_out(
      is_bitrevA ? bitrev_out[0] : bitrev_out[1],
      is_bitrevA ? bitrev_out[1] : bitrev_out[0],
      data_out, start_row);


    if (step >= (DEPTH)) {
      write_channel_intel(chaninfft3db[0], data_out.i0);
      write_channel_intel(chaninfft3db[1], data_out.i1);
      write_channel_intel(chaninfft3db[2], data_out.i2);
      write_channel_intel(chaninfft3db[3], data_out.i3);
      write_channel_intel(chaninfft3db[4], data_out.i4);
      write_channel_intel(chaninfft3db[5], data_out.i5);
      write_channel_intel(chaninfft3db[6], data_out.i6);
      write_channel_intel(chaninfft3db[7], data_out.i7);
    }
  }
}

kernel void fft3db(int inverse) {

  /* The FFT engine requires a sliding window for data reordering; data stored
   * in this array is carried across loop iterations and shifted by 1 element
   * every iteration; all loop dependencies derived from the uses of this 
   * array are simple transfers between adjacent array elements
   */

  float2 fft_delay_elements[N + POINTS * (LOGN - 2)];

  #pragma loop_coalesce
  for(unsigned j = 0; j < N; j++){
    for (unsigned i = 0; i < N * (N / POINTS) + N / POINTS - 1; i++) {
      float2x8 data;

      if (i < N * (N / POINTS)) {
        data.i0 = read_channel_intel(chaninfft3db[0]);
        data.i1 = read_channel_intel(chaninfft3db[1]);
        data.i2 = read_channel_intel(chaninfft3db[2]);
        data.i3 = read_channel_intel(chaninfft3db[3]);
        data.i4 = read_channel_intel(chaninfft3db[4]);
        data.i5 = read_channel_intel(chaninfft3db[5]);
        data.i6 = read_channel_intel(chaninfft3db[6]);
        data.i7 = read_channel_intel(chaninfft3db[7]);
      } else {
        data.i0 = data.i1 = data.i2 = data.i3 = 
                  data.i4 = data.i5 = data.i6 = data.i7 = 0;
      }

      data = fft_step(data, i % (N / POINTS), fft_delay_elements, inverse, LOGN);

      if (i >= N / POINTS - 1) {
        write_channel_intel(chaninTranStore1[0], data.i0);
        write_channel_intel(chaninTranStore1[1], data.i1);
        write_channel_intel(chaninTranStore1[2], data.i2);
        write_channel_intel(chaninTranStore1[3], data.i3);
        write_channel_intel(chaninTranStore1[4], data.i4);
        write_channel_intel(chaninTranStore1[5], data.i5);
        write_channel_intel(chaninTranStore1[6], data.i6);
        write_channel_intel(chaninTranStore1[7], data.i7);
      }
    }
  }
}

__attribute__((max_global_work_dim(0)))
kernel void transposeStore1(global float2 * restrict dest) {

  const int DELAY = (1 << (LOGN - LOGPOINTS)); // N / 8
  bool is_bufA = false, is_bitrevA = false;

  float2 buf[2][DEPTH][POINTS];
  float2 __attribute__((memory, numbanks(8))) bitrev_in[2][N];
  
  int initial_delay = DELAY; // for each of the bitrev buffer
  // additional iterations to fill the buffers
  for(int step = -initial_delay; step < ((N * DEPTH) + DEPTH); step++){

    float2x8 data, data_out;
    if (step < ((N * DEPTH) - initial_delay)) {
      data.i0 = read_channel_intel(chaninTranStore1[0]);
      data.i1 = read_channel_intel(chaninTranStore1[1]);
      data.i2 = read_channel_intel(chaninTranStore1[2]);
      data.i3 = read_channel_intel(chaninTranStore1[3]);
      data.i4 = read_channel_intel(chaninTranStore1[4]);
      data.i5 = read_channel_intel(chaninTranStore1[5]);
      data.i6 = read_channel_intel(chaninTranStore1[6]);
      data.i7 = read_channel_intel(chaninTranStore1[7]);
    } else {
      data.i0 = data.i1 = data.i2 = data.i3 = 
                data.i4 = data.i5 = data.i6 = data.i7 = 0;
    }
    // Swap buffers every N*N/8 iterations 
    // starting from the additional delay of N/8 iterations
    is_bufA = (( step & (DEPTH - 1)) == 0) ? !is_bufA: is_bufA;

    // Swap bitrev buffers every N/8 iterations
    is_bitrevA = ( (step & ((N / 8) - 1)) == 0) ? !is_bitrevA: is_bitrevA;

    unsigned row = step & (DEPTH - 1);
    data = bitreverse_in(data,
      is_bitrevA ? bitrev_in[0] : bitrev_in[1], 
      is_bitrevA ? bitrev_in[1] : bitrev_in[0], 
      row);

    writeBuf(data,
      is_bufA ? buf[0] : buf[1],
      step, 0);

    data_out = readBuf_store(
      is_bufA ? buf[1] : buf[0], 
      step);

    if (step >= (DEPTH)) {
      unsigned index = (step - DEPTH) * 8;

      dest[index + 0] = data_out.i0;
      dest[index + 1] = data_out.i1;
      dest[index + 2] = data_out.i2;
      dest[index + 3] = data_out.i3;
      dest[index + 4] = data_out.i4;
      dest[index + 5] = data_out.i5;
      dest[index + 6] = data_out.i6;
      dest[index + 7] = data_out.i7;
    }
  }
}

// Kernel that fetches data from global memory 
kernel void fetchBitrev2(global volatile float2 * restrict src) {
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

        write_channel_intel(chaninfft3dc[0], buf[where]); 
        write_channel_intel(chaninfft3dc[1], buf[where + 4 * (N / 8) * N]);
        write_channel_intel(chaninfft3dc[2], buf[where + 2 * (N / 8) * N]);
        write_channel_intel(chaninfft3dc[3], buf[where + 6 * (N / 8) * N]);
        write_channel_intel(chaninfft3dc[4], buf[where + (N / 8) * N]); 
        write_channel_intel(chaninfft3dc[5], buf[where + 5 * (N / 8) * N]);
        write_channel_intel(chaninfft3dc[6], buf[where + 3 * (N / 8) * N]);
        write_channel_intel(chaninfft3dc[7], buf[where + 7 * (N / 8) * N]);
      }
    }
  } // y axis
}

/*
 * Input and output data in bit-reversed format
 */
kernel void fft3dc(int inverse) {

  /* The FFT engine requires a sliding window for data reordering; data stored
   * in this array is carried across loop iterations and shifted by 1 element
   * every iteration; all loop dependencies derived from the uses of this 
   * array are simple transfers between adjacent array elements
   */

  float2 fft_delay_elements[N + POINTS * (LOGN - 2)];

  #pragma loop_coalesce
  for(unsigned j = 0; j < N; j++){

    for (unsigned i = 0; i < N * (N / POINTS) + N / POINTS - 1; i++) {
      float2x8 data;

      if (i < N * (N / POINTS)) {
        data.i0 = read_channel_intel(chaninfft3dc[0]);
        data.i1 = read_channel_intel(chaninfft3dc[1]);
        data.i2 = read_channel_intel(chaninfft3dc[2]);
        data.i3 = read_channel_intel(chaninfft3dc[3]);
        data.i4 = read_channel_intel(chaninfft3dc[4]);
        data.i5 = read_channel_intel(chaninfft3dc[5]);
        data.i6 = read_channel_intel(chaninfft3dc[6]);
        data.i7 = read_channel_intel(chaninfft3dc[7]);
      } else {
        data.i0 = data.i1 = data.i2 = data.i3 = 
                  data.i4 = data.i5 = data.i6 = data.i7 = 0;
      }

      // Perform one FFT step
      data = fft_step(data, i % (N / POINTS), fft_delay_elements, inverse, LOGN);

      // Write result to channels
      if (i >= N / POINTS - 1) {
        write_channel_intel(chaninTranStore2[0], data.i0);
        write_channel_intel(chaninTranStore2[1], data.i1);
        write_channel_intel(chaninTranStore2[2], data.i2);
        write_channel_intel(chaninTranStore2[3], data.i3);
        write_channel_intel(chaninTranStore2[4], data.i4);
        write_channel_intel(chaninTranStore2[5], data.i5);
        write_channel_intel(chaninTranStore2[6], data.i6);
        write_channel_intel(chaninTranStore2[7], data.i7);
      }
    }
  }
}

__attribute__((max_global_work_dim(0)))
kernel void transposeStore2(global float2 * restrict dest) {

  const int DELAY = (1 << (LOGN - LOGPOINTS)); // N / 8
  bool is_bufA = false, is_bitrevA = false;

  float2 buf[2][DEPTH][POINTS];
  float2 __attribute__((memory, numbanks(8))) bitrev_in[2][N];
  
  int initial_delay = DELAY; // for each of the bitrev buffer
  // additional iterations to fill the buffers
  for(int step = -initial_delay; step < ((N * DEPTH) + DEPTH); step++){

    float2x8 data, data_out;
    if (step < ((N * DEPTH) - initial_delay)) {
      data.i0 = read_channel_intel(chaninTranStore2[0]);
      data.i1 = read_channel_intel(chaninTranStore2[1]);
      data.i2 = read_channel_intel(chaninTranStore2[2]);
      data.i3 = read_channel_intel(chaninTranStore2[3]);
      data.i4 = read_channel_intel(chaninTranStore2[4]);
      data.i5 = read_channel_intel(chaninTranStore2[5]);
      data.i6 = read_channel_intel(chaninTranStore2[6]);
      data.i7 = read_channel_intel(chaninTranStore2[7]);
    } else {
      data.i0 = data.i1 = data.i2 = data.i3 = 
                data.i4 = data.i5 = data.i6 = data.i7 = 0;
    }
    // Swap buffers every N*N/8 iterations 
    // starting from the additional delay of N/8 iterations
    is_bufA = (( step & (DEPTH - 1)) == 0) ? !is_bufA: is_bufA;

    // Swap bitrev buffers every N/8 iterations
    is_bitrevA = ( (step & ((N / 8) - 1)) == 0) ? !is_bitrevA: is_bitrevA;

    unsigned row = step & (DEPTH - 1);
    data = bitreverse_in(data,
      is_bitrevA ? bitrev_in[0] : bitrev_in[1], 
      is_bitrevA ? bitrev_in[1] : bitrev_in[0], 
      row);

    writeBuf(data,
      is_bufA ? buf[0] : buf[1],
      step, 0);

    data_out = readBuf_store(
      is_bufA ? buf[1] : buf[0], 
      step);

    if (step >= (DEPTH)) {
      unsigned start_index = (step - DEPTH);
      // increment z by 1 every N/8 steps until (N*N/ 8)
      unsigned zdim = (start_index >> (LOGN - LOGPOINTS)) & (N - 1); 

      // increment y by 1 every N*N/8 points until N
      unsigned ydim = (start_index >> (LOGN + LOGN - LOGPOINTS)) & (N - 1);

      // incremenet by 8 until N / 8
      unsigned xdim = (start_index * 8) & ( N - 1);
      //unsigned index = (step - DEPTH) * 8;

      // increment by N*N*N
      unsigned cube = LOGN + LOGN + LOGN - LOGPOINTS;

      // increment by 1 every N*N*N / 8 steps
      unsigned batch_index = (start_index >> cube);
      //unsigned batch_index = 0;

      unsigned index = (batch_index * N * N * N) + (zdim * N * N) + (ydim * N) + xdim; 

      //printf("start_index: %u, batch: %u, zim: %u, ydim: %u, xdim: %u, index: %u \n", start_index, batch_index, zdim, ydim, xdim, index);

      dest[index + 0] = data_out.i0;
      dest[index + 1] = data_out.i1;
      dest[index + 2] = data_out.i2;
      dest[index + 3] = data_out.i3;
      dest[index + 4] = data_out.i4;
      dest[index + 5] = data_out.i5;
      dest[index + 6] = data_out.i6;
      dest[index + 7] = data_out.i7;
    }
  }
}