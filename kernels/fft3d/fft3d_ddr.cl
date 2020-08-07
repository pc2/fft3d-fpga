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
  unsigned delay = (1 << (LOGN - LOGPOINTS)); // N / 8
  bool is_bitrevA = false;

  float2 __attribute__((memory, numbanks(8))) buf[2][N];
  
  // additional iterations to fill the buffers
  for(unsigned step = 0; step < (N * DEPTH) + delay; step++){

    unsigned where = (step & ((N * DEPTH) - 1)) * 8; 

    float2x8 data;
    if (step < (N * DEPTH)) {
      data.i0 = src[where + 0];
      data.i1 = src[where + 1];
      data.i2 = src[where + 2];
      data.i3 = src[where + 3];
      data.i4 = src[where + 4];
      data.i5 = src[where + 5];
      data.i6 = src[where + 6];
      data.i7 = src[where + 7];
    } else {
      data.i0 = data.i1 = data.i2 = data.i3 = 
                data.i4 = data.i5 = data.i6 = data.i7 = 0;
    }

    is_bitrevA = ( (step & ((N / 8) - 1)) == 0) ? !is_bitrevA: is_bitrevA;

    unsigned row = step & (DEPTH - 1);
    data = bitreverse_fetch(data,
      is_bitrevA ? buf[0] : buf[1], 
      is_bitrevA ? buf[1] : buf[0], 
      row);

    if (step >= delay) {
      write_channel_intel(chaninfft3da[0], data.i0);
      write_channel_intel(chaninfft3da[1], data.i1);
      write_channel_intel(chaninfft3da[2], data.i2);
      write_channel_intel(chaninfft3da[3], data.i3);
      write_channel_intel(chaninfft3da[4], data.i4);
      write_channel_intel(chaninfft3da[5], data.i5);
      write_channel_intel(chaninfft3da[6], data.i6);
      write_channel_intel(chaninfft3da[7], data.i7);
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
  //float2 bitrev_in[2][N], bitrev_out[2][N];
  //float2 __attribute__((memory, numbanks(8))) bitrev_in[2][N];
  float2 bitrev_in[2][N];
  float2 __attribute__((memory, numbanks(8))) bitrev_out[2][N];
  
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
kernel void transposeStore1(__global __attribute__((buffer_location(BUFFER_LOCATION))) volatile float2 * restrict dest) {

  const int DELAY = (1 << (LOGN - LOGPOINTS)); // N / 8
  bool is_bufA = false, is_bitrevA = false;

  float2 buf[2][DEPTH][POINTS];
  //float2 __attribute__((memory, numbanks(8))) bitrev_in[2][N];
  float2 bitrev_in[2][N];
  
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
__attribute__((max_global_work_dim(0)))
kernel void fetchBitrev2(__global __attribute__((buffer_location(BUFFER_LOCATION))) volatile float2 * restrict src) {
  unsigned delay = (1 << (LOGN - LOGPOINTS)); // N / 8

  bool is_bufA = false, is_bitrevA = false;
  float2 __attribute__((memory, numbanks(8))) bitrev_out[2][N];
  float2 buf[2][DEPTH][POINTS];
  
  // additional iterations to fill the buffers
  for(unsigned step = 0; step < (N * DEPTH) + DEPTH + delay; step++){
    // increment z by 1 every N/8 steps until (N*N/ 8)
    unsigned start_index = step + delay;
    unsigned zdim = (step >> (LOGN - LOGPOINTS)) & (N - 1); 

    // increment y by 1 every N*N/8 points until N
    unsigned ydim = (step >> (LOGN + LOGN - LOGPOINTS)) & (N - 1);

    // increment by 8 until N / 8
    unsigned xdim = (step * 8) & (N - 1);

    // increment by 1 every N*N*N / 8 steps
    unsigned batch_index = (step >> (LOGN + LOGN + LOGN - LOGPOINTS));

    unsigned index = (batch_index * N * N * N) + (zdim * N * N) + (ydim * N) + xdim; 

    float2x8 data, data_out;
    if (step < (N * DEPTH)) {
      data.i0 = src[index + 0];
      data.i1 = src[index + 1];
      data.i2 = src[index + 2];
      data.i3 = src[index + 3];
      data.i4 = src[index + 4];
      data.i5 = src[index + 5];
      data.i6 = src[index + 6];
      data.i7 = src[index + 7];
    } else {
      data.i0 = data.i1 = data.i2 = data.i3 = 
                data.i4 = data.i5 = data.i6 = data.i7 = 0;
    }
  
    is_bufA = (( step & (DEPTH - 1)) == 0) ? !is_bufA: is_bufA;

    // Swap bitrev buffers every N/8 iterations
    is_bitrevA = ( (step & ((N / 8) - 1)) == 0) ? !is_bitrevA: is_bitrevA;

    writeBuf(data,
      is_bufA ? buf[0] : buf[1],
      step, 0);

    data_out = readBuf_fetch(
      is_bufA ? buf[1] : buf[0], 
      step, 0);

    unsigned start_row = step & (DEPTH -1);
    data_out = bitreverse_out(
      is_bitrevA ? bitrev_out[0] : bitrev_out[1],
      is_bitrevA ? bitrev_out[1] : bitrev_out[0],
      data_out, start_row);

    if (step >= (DEPTH + delay)) {

      write_channel_intel(chaninfft3dc[0], data_out.i0);
      write_channel_intel(chaninfft3dc[1], data_out.i1);
      write_channel_intel(chaninfft3dc[2], data_out.i2);
      write_channel_intel(chaninfft3dc[3], data_out.i3);
      write_channel_intel(chaninfft3dc[4], data_out.i4);
      write_channel_intel(chaninfft3dc[5], data_out.i5);
      write_channel_intel(chaninfft3dc[6], data_out.i6);
      write_channel_intel(chaninfft3dc[7], data_out.i7);
    }
  }
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
  float2 bitrev_in[2][N];
  //float2 __attribute__((memory, numbanks(8))) bitrev_in[2][N];
  
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