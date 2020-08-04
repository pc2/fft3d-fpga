//  Author: Arjun Ramaswami

#include "fft_config.h"
#include "fft_8.cl" 
#include "../matrixTranspose/diagonal_bitrev.cl"

#pragma OPENCL EXTENSION cl_intel_channels : enable
channel float2 chaninfft2da[POINTS] __attribute__((depth(POINTS)));
channel float2 chaninfft2db[POINTS] __attribute__((depth(POINTS)));

channel float2 chaninTranspose1[POINTS] __attribute__((depth(POINTS)));
channel float2 chaninTranspose2[POINTS] __attribute__((depth(POINTS)));
channel float2 chaninStore[POINTS] __attribute__((depth(POINTS)));

// Kernel that fetches data from global memory 
kernel void fetchBitrev1(global volatile float2 * restrict src) {

  for(unsigned k = 0; k < N; k++){ 
    float2 buf[N];

    #pragma unroll 8
    for(unsigned i = 0; i < N; i++){
      buf[i & ((1<<LOGN)-1)] = src[(k << LOGN) + i];    
    }

    for(unsigned j = 0; j < (N / 8); j++){
      write_channel_intel(chaninfft2da[0], buf[j]);               // 0
      write_channel_intel(chaninfft2da[1], buf[4 * N / 8 + j]);   // 32
      write_channel_intel(chaninfft2da[2], buf[2 * N / 8 + j]);   // 16
      write_channel_intel(chaninfft2da[3], buf[6 * N / 8 + j]);   // 48
      write_channel_intel(chaninfft2da[4], buf[N / 8 + j]);       // 8
      write_channel_intel(chaninfft2da[5], buf[5 * N / 8 + j]);   // 40
      write_channel_intel(chaninfft2da[6], buf[3 * N / 8 + j]);   // 24
      write_channel_intel(chaninfft2da[7], buf[7 * N / 8 + j]);   // 54
    }
  }
}

/* This single work-item task wraps the FFT engine
 * 'inverse' toggles between the direct and the inverse transform
 */

kernel void fft2da(int inverse) {

  /* The FFT engine requires a sliding window for data reordering; data stored
   * in this array is carried across loop iterations and shifted by 1 element
   * every iteration; all loop dependencies derived from the uses of this 
   * array are simple transfers between adjacent array elements
   */

  float2 fft_delay_elements[N + POINTS * (LOGN - 2)];

  // needs to run "N / 8 - 1" additional iterations to drain the last outputs
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
      write_channel_intel(chaninTranspose1[0], data.i0);
      write_channel_intel(chaninTranspose1[1], data.i1);
      write_channel_intel(chaninTranspose1[2], data.i2);
      write_channel_intel(chaninTranspose1[3], data.i3);
      write_channel_intel(chaninTranspose1[4], data.i4);
      write_channel_intel(chaninTranspose1[5], data.i5);
      write_channel_intel(chaninTranspose1[6], data.i6);
      write_channel_intel(chaninTranspose1[7], data.i7);
    }
  }
}

__attribute__((max_global_work_dim(0)))
kernel void transpose1() {
  const unsigned DELAY = (1 << (LOGN - LOGPOINTS)); // N / 8
  bool is_bufA = false, is_bitrevA = false;

  float2 buf[2][DEPTH][POINTS];
  float2 bitrev_in[2][N], bitrev_out[2][N] ;
  //float2 bitrev_in[2][N] __attribute__((memory("MLAB")));
  
  int initial_delay = DELAY + DELAY; // for each of the bitrev buffer

  // additional iterations to fill the buffers
  for(int step = -initial_delay; step < ((DEPTH) + DEPTH); step++){
    float2x8 data, data_out;
    if (step < ((DEPTH) - initial_delay)) {
      data.i0 = read_channel_intel(chaninTranspose1[0]);
      data.i1 = read_channel_intel(chaninTranspose1[1]);
      data.i2 = read_channel_intel(chaninTranspose1[2]);
      data.i3 = read_channel_intel(chaninTranspose1[3]);
      data.i4 = read_channel_intel(chaninTranspose1[4]);
      data.i5 = read_channel_intel(chaninTranspose1[5]);
      data.i6 = read_channel_intel(chaninTranspose1[6]);
      data.i7 = read_channel_intel(chaninTranspose1[7]);
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

kernel void fft2db(int inverse) {

  /* The FFT engine requires a sliding window for data reordering; data stored
   * in this array is carried across loop iterations and shifted by 1 element
   * every iteration; all loop dependencies derived from the uses of this 
   * array are simple transfers between adjacent array elements
   */

  float2 fft_delay_elements[N + POINTS * (LOGN - 2)];

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
      write_channel_intel(chaninTranspose2[0], data.i0);
      write_channel_intel(chaninTranspose2[1], data.i1);
      write_channel_intel(chaninTranspose2[2], data.i2);
      write_channel_intel(chaninTranspose2[3], data.i3);
      write_channel_intel(chaninTranspose2[4], data.i4);
      write_channel_intel(chaninTranspose2[5], data.i5);
      write_channel_intel(chaninTranspose2[6], data.i6);
      write_channel_intel(chaninTranspose2[7], data.i7);
    }
  }
}

__attribute__((max_global_work_dim(0)))
kernel void transpose2() {
  const int DELAY = (1 << (LOGN - LOGPOINTS)); // N / 8
  bool is_bufA = false, is_bitrevA = false;

  float2 buf[2][DEPTH][POINTS];
  float2 __attribute__((memory, numbanks(8))) bitrev_in[2][N];
  
  int initial_delay = DELAY; // for each of the bitrev buffer
  // additional iterations to fill the buffers
  for(int step = -initial_delay; step < ((DEPTH) + DEPTH); step++){
    float2x8 data, data_out;
    if (step < ((DEPTH) - initial_delay)) {
      data.i0 = read_channel_intel(chaninTranspose2[0]);
      data.i1 = read_channel_intel(chaninTranspose2[1]);
      data.i2 = read_channel_intel(chaninTranspose2[2]);
      data.i3 = read_channel_intel(chaninTranspose2[3]);
      data.i4 = read_channel_intel(chaninTranspose2[4]);
      data.i5 = read_channel_intel(chaninTranspose2[5]);
      data.i6 = read_channel_intel(chaninTranspose2[6]);
      data.i7 = read_channel_intel(chaninTranspose2[7]);
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
      write_channel_intel(chaninStore[0], data_out.i0);
      write_channel_intel(chaninStore[1], data_out.i1);
      write_channel_intel(chaninStore[2], data_out.i2);
      write_channel_intel(chaninStore[3], data_out.i3);
      write_channel_intel(chaninStore[4], data_out.i4);
      write_channel_intel(chaninStore[5], data_out.i5);
      write_channel_intel(chaninStore[6], data_out.i6);
      write_channel_intel(chaninStore[7], data_out.i7);
    }
  }
}

kernel void store(global volatile float2 * restrict dest){

  // perform N*N writes to buffer
  for(unsigned i = 0; i < N; i++){
    for(unsigned j = 0; j < (N / 8); j++){
      unsigned where = ((i << LOGN) + (j << LOGPOINTS));
      
      #pragma unroll 8
      for(unsigned u = 0; u < 8; u++){
        dest[where + u] = read_channel_intel(chaninStore[u]);
      }
    }
  }
}
/*
kernel void store(global volatile float2 * restrict dest){
  unsigned revcolt, where;

  local float2 buf[N * N];

  // perform N*N writes to buffer
  for(unsigned i = 0; i < N; i++){
    for(unsigned j = 0; j < (N / 8); j++){
      where = ((i << LOGN) + (j << LOGPOINTS));
      
      #pragma unroll 8
      for(unsigned u = 0; u < 8; u++){
        buf[where + u] = read_channel_intel(chaninStore[u]);
      }
    }
  }

  for(unsigned i = 0; i < N; i++){
    revcolt = bit_reversed(i, LOGN);
    where = (i << LOGN);
    
    #pragma unroll 8
    for( unsigned u = 0; u < N; u++){
      dest[where + u] = buf[(u << LOGN) + revcolt];
    }
  }
}
*/
/*
kernel void store(global volatile float2 * restrict dest){
  const int DELAY = (1 << (LOGN - LOGPOINTS)); // N / 8
  bool is_bufA = false, is_bitrevA = false;

  float2 buf[2][DEPTH][POINTS];
  float2 bitrev_in[2][N], bitrev_out[2][N] ;
  //float2 bitrev_in[2][N] __attribute__((memory("MLAB")));
  
  int initial_delay = DELAY + DELAY; // for each of the bitrev buffer

  // additional iterations to fill the buffers
  for(int step = -initial_delay; step < ((DEPTH) + DEPTH); step++){

    float2x8 data, data_out;
    if (step < ((DEPTH) - initial_delay)) {
      data.i0 = read_channel_intel(chaninStore[0]);
      data.i1 = read_channel_intel(chaninStore[1]);
      data.i2 = read_channel_intel(chaninStore[2]);
      data.i3 = read_channel_intel(chaninStore[3]);
      data.i4 = read_channel_intel(chaninStore[4]);
      data.i5 = read_channel_intel(chaninStore[5]);
      data.i6 = read_channel_intel(chaninStore[6]);
      data.i7 = read_channel_intel(chaninStore[7]);
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
      step);

    data_out = readBuf(
      is_bufA ? buf[1] : buf[0], 
      step);

    if (step >= (DEPTH)) {
      unsigned index = (step - DEPTH) * 8;
      printf("Store index - %d step : %d \n", index, step);
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
  printf("Store Completed\n");
}
*/