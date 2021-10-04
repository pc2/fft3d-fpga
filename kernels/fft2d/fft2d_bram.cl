//  Author: Arjun Ramaswami

#include "fft_config.h"
#include "fft_8.cl" 
#include "../matrixTranspose/diagonal_bitrev.cl"

#pragma OPENCL EXTENSION cl_intel_channels : enable
channel float2 chaninfft2da[POINTS] __attribute__((depth(POINTS)));
channel float2 chaninfft2db[POINTS] __attribute__((depth(POINTS)));

channel float2 chaninTranspose[POINTS] __attribute__((depth(POINTS)));
channel float2 chaninTransStore[POINTS] __attribute__((depth(POINTS)));

kernel void fetchBitrev(global volatile float2 * restrict src, int how_many) {
  unsigned delay = (1 << (LOGN - LOGPOINTS)); // N / 8
  bool is_bitrevA = false;

  float2 __attribute__((memory, numbanks(8))) buf[2][N];
  
  // additional iterations to fill the buffers
  for(unsigned step = 0; step < (how_many * DEPTH) + delay; step++){

    unsigned where = (step & ((N * DEPTH) - 1)) * 8; 

    float2x8 data;
    if (step < (how_many * DEPTH)) {
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
      write_channel_intel(chaninfft2da[0], data.i0);
      write_channel_intel(chaninfft2da[1], data.i1);
      write_channel_intel(chaninfft2da[2], data.i2);
      write_channel_intel(chaninfft2da[3], data.i3);
      write_channel_intel(chaninfft2da[4], data.i4);
      write_channel_intel(chaninfft2da[5], data.i5);
      write_channel_intel(chaninfft2da[6], data.i6);
      write_channel_intel(chaninfft2da[7], data.i7);
    }
  }
}

kernel void fft2da(int inverse, int how_many) {

  /* The FFT engine requires a sliding window for data reordering; data stored
   * in this array is carried across loop iterations and shifted by 1 element
   * every iteration; all loop dependencies derived from the uses of this 
   * array are simple transfers between adjacent array elements
   */

  float2 fft_delay_elements[N + POINTS * (LOGN - 2)];

  // needs to run "N / 8 - 1" additional iterations to drain the last outputs
  #pragma loop_coalesce
  for(unsigned j = 0; j < how_many; j++){
    for (unsigned i = 0; i < N * (N / POINTS) + N / POINTS - 1; i++) {
      float2x8 data;

      // Read data from channels
      if (i < N * (N / POINTS)) {
        data.i0 = read_channel_intel(chaninfft2da[0]);
        data.i1 = read_channel_intel(chaninfft2da[1]);
        data.i2 = read_channel_intel(chaninfft2da[2]);
        data.i3 = read_channel_intel(chaninfft2da[3]);
        data.i4 = read_channel_intel(chaninfft2da[4]);
        data.i5 = read_channel_intel(chaninfft2da[5]);
        data.i6 = read_channel_intel(chaninfft2da[6]);
        data.i7 = read_channel_intel(chaninfft2da[7]);
      } 
      else {
        data.i0 = data.i1 = data.i2 = data.i3 = 
                  data.i4 = data.i5 = data.i6 = data.i7 = 0;
      }

      // Perform one FFT step
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

kernel void transpose(int how_many) {
  const int DELAY = (1 << (LOGN - LOGPOINTS)); // N / 8
  bool is_bufA = false, is_bitrevA = false;

  float2 buf[2][DEPTH][POINTS];
  float2 bitrev_in[2][N];
  float2 __attribute__((memory, numbanks(8))) bitrev_out[2][N];
  
  int initial_delay = DELAY + DELAY; // for each of the bitrev buffer

  // additional iterations to fill the buffers
  for(int step = -initial_delay; step < ((how_many * DEPTH) + DEPTH); step++){

    float2x8 data, data_out;
    if (step < ((how_many * DEPTH) - initial_delay)) {
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
      write_channel_intel(chaninfft2db[0], data_out.i0);
      write_channel_intel(chaninfft2db[1], data_out.i1);
      write_channel_intel(chaninfft2db[2], data_out.i2);
      write_channel_intel(chaninfft2db[3], data_out.i3);
      write_channel_intel(chaninfft2db[4], data_out.i4);
      write_channel_intel(chaninfft2db[5], data_out.i5);
      write_channel_intel(chaninfft2db[6], data_out.i6);
      write_channel_intel(chaninfft2db[7], data_out.i7);
    }
  }
}

kernel void fft2db(int inverse, int how_many) {

  /* The FFT engine requires a sliding window for data reordering; data stored
   * in this array is carried across loop iterations and shifted by 1 element
   * every iteration; all loop dependencies derived from the uses of this 
   * array are simple transfers between adjacent array elements
   */

  float2 fft_delay_elements[N + POINTS * (LOGN - 2)];

  #pragma loop_coalesce
  for(unsigned j = 0; j < 1; j++){
    for (unsigned i = 0; i < N * (N / POINTS) + N / POINTS - 1; i++) {
      float2x8 data;

      // Read data from channels
      if (i < N * (N / POINTS)) {
        data.i0 = read_channel_intel(chaninfft2db[0]);
        data.i1 = read_channel_intel(chaninfft2db[1]);
        data.i2 = read_channel_intel(chaninfft2db[2]);
        data.i3 = read_channel_intel(chaninfft2db[3]);
        data.i4 = read_channel_intel(chaninfft2db[4]);
        data.i5 = read_channel_intel(chaninfft2db[5]);
        data.i6 = read_channel_intel(chaninfft2db[6]);
        data.i7 = read_channel_intel(chaninfft2db[7]);
      } else {
        data.i0 = data.i1 = data.i2 = data.i3 = 
                  data.i4 = data.i5 = data.i6 = data.i7 = 0;
      }

      // Perform one FFT step
      data = fft_step(data, i % (N / POINTS), fft_delay_elements, inverse, LOGN);

      // Write result to channels
      if (i >= N / POINTS - 1) {
        write_channel_intel(chaninTransStore[0], data.i0);
        write_channel_intel(chaninTransStore[1], data.i1);
        write_channel_intel(chaninTransStore[2], data.i2);
        write_channel_intel(chaninTransStore[3], data.i3);
        write_channel_intel(chaninTransStore[4], data.i4);
        write_channel_intel(chaninTransStore[5], data.i5);
        write_channel_intel(chaninTransStore[6], data.i6);
        write_channel_intel(chaninTransStore[7], data.i7);
      }
    }
  }
}

kernel void transposeStore(global volatile float2 * restrict dest, int how_many) {

  const int DELAY = (1 << (LOGN - LOGPOINTS)); // N / 8
  bool is_bufA = false, is_bitrevA = false;

  float2 buf[2][DEPTH][POINTS];
  float2 bitrev_in[2][N];
  
  int initial_delay = DELAY; // for each of the bitrev buffer
  // additional iterations to fill the buffers
  for(int step = -initial_delay; step < ((how_many * DEPTH) + DEPTH); step++){

    float2x8 data, data_out;
    if (step < ((how_many * DEPTH) - initial_delay)) {
      data.i0 = read_channel_intel(chaninTransStore[0]);
      data.i1 = read_channel_intel(chaninTransStore[1]);
      data.i2 = read_channel_intel(chaninTransStore[2]);
      data.i3 = read_channel_intel(chaninTransStore[3]);
      data.i4 = read_channel_intel(chaninTransStore[4]);
      data.i5 = read_channel_intel(chaninTransStore[5]);
      data.i6 = read_channel_intel(chaninTransStore[6]);
      data.i7 = read_channel_intel(chaninTransStore[7]);
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