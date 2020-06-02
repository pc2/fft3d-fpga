/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

#ifndef LOGN
#define LOGN 6
#endif

// Macros for the 8 point 1d FFT
#define LOGPOINTS 3
#define POINTS (1 << LOGPOINTS)

// Log of the number of replications of the pipeline
#define LOGREPL 2            // 4 replications 
#define REPL (1 << LOGREPL)  // 4 replications 
#define UNROLL_FACTOR 8 

#pragma OPENCL EXTENSION cl_intel_channels : enable
channel float2 chaninTranspose[8] __attribute__((depth(8)));
channel float2 chanoutTranspose[8] __attribute__((depth(8)));

// --- CODE -------------------------------------------------------------------
__attribute__((max_global_work_dim(0)))
kernel void fetch(global const volatile float2 * restrict src, int iter) {
  const unsigned N = (1 << LOGN);

  for(unsigned i = 0; i < iter * N * N / 8; i++){

    write_channel_intel(chaninTranspose[0], src[(i * 8) + 0]);
    write_channel_intel(chaninTranspose[1], src[(i * 8) + 1]);
    write_channel_intel(chaninTranspose[2], src[(i * 8) + 2]);
    write_channel_intel(chaninTranspose[3], src[(i * 8) + 3]);
    write_channel_intel(chaninTranspose[4], src[(i * 8) + 4]);
    write_channel_intel(chaninTranspose[5], src[(i * 8) + 5]);
    write_channel_intel(chaninTranspose[6], src[(i * 8) + 6]);
    write_channel_intel(chaninTranspose[7], src[(i * 8) + 7]);
  }
}

__attribute__((max_global_work_dim(0)))
kernel void transpose(int iter) {
  const unsigned N = (1 << LOGN);
  const unsigned DEPTH = (1 << (LOGN + LOGN - LOGPOINTS));

  /* Input: Create M20ks banked column-wise, fill with rotated data
   *  - width : 512 bits or 8 complex or 8 banks, since input sample size
   *  - depth : N * N / 8, eg. 64 * 64 / 8 deep, to fill the M20k maximum
   *
   *  Example for 64^2 matrix:
   *  -------------------------------------------------
   *  |   0 |   1 |   2 |   3 |   4 |   5 |   6 |   7 |  - Depth 0
   *  |   8 |   9 |  10 |  11 |  12 |  13 |  14 |  15 |  - Depth 1
   *  |  16 |  17 |  18 |  19 |  20 |  21 |  22 |  23 |  - Depth 2
   *  |  ....  ....  ....  .                          |
   *  |  71 |  64 |  65 |  66 |  67 |  68 |  69 |  70 |  - Depth 8 (N / 8)
   *  |  79 |  72 |  73 |  74 |  75 |  76 |  77 |  78 |  - Depth 9
   *  ..
   *  | 133 | 134 | 128 | 129 | 130 | 131 | 131 | 132 |  - Depth 16 (2 *(N / 8))
   *  ..
   *
   *  REASONING:
   *  Note that every 8th row (N / 8), data is filled with a stride of 1.
   *
   *  This creates traditional column wise data to be in different banks at
   *  strides of (N / 8). Therefore, accessing transposed data in different bank
   *
   *  This requires an additional buffer to rotate data before filling.
   */

  // iterate over 2d matrices
  for(unsigned k = 0 ; k < iter; k++){

    // Buffer with width - 8 points, depth - (N*N / 8), banked column-wise
    float2 buf[DEPTH][POINTS];
      
    // iterate within a 2d matrix
    for(unsigned row = 0; row < DEPTH; row++){

      // Temporary buffer to rotate before filling the matrix
      float2 rotate_in[POINTS];

      // store data in a temp buffer
      rotate_in[0] = read_channel_intel(chaninTranspose[0]);
      rotate_in[1] = read_channel_intel(chaninTranspose[1]);
      rotate_in[2] = read_channel_intel(chaninTranspose[2]);
      rotate_in[3] = read_channel_intel(chaninTranspose[3]);
      rotate_in[4] = read_channel_intel(chaninTranspose[4]);
      rotate_in[5] = read_channel_intel(chaninTranspose[5]);
      rotate_in[6] = read_channel_intel(chaninTranspose[6]);
      rotate_in[7] = read_channel_intel(chaninTranspose[7]);

      /*  Computing whether and how much to rotate.
       *  Idea: Rotate every N / 8 rows, wrap around after 8 rotations
       *
       *  Every (N / 8 rows): (row >> LOGN - LOGPOINTS)
       *  Every 8 rotations wrap around the banks, rot = (...) & 7
       *
       *  Values : [0, 1, .. , 7]
      */
      unsigned rot = row >> (LOGN - LOGPOINTS) & (POINTS - 1);

      /* Idea: access temp buffer with rotation
       * rotate_in : buffer of 8 points
       * offset_val : [0, 1,.. 7]
       *
       * row = 0, rot = 0, offset_val = {0,1,..7} -> (i + 8 - 0) & 7
       *    output - 0, 1, 2 .. 7
       *
       * row = 1, rot = 0, offset_val = {0,1,..7} -> (i + 8 - 0) & 7
       *    output - 8, 9, 10 .. 15
       * ...
       * row = 8, rot = 1, offset_val = {1,2,..7,0} -> (i + 8 - 1) & 7
       *    rotate_in - {64, 65, 66, .. 71}
       *    output -  { 71, 64, 65 .. 70}
       *
       */
      #pragma unroll 8
      for(unsigned i = 0; i < POINTS; i++){
          buf[row][i] = rotate_in[((i + POINTS) - rot) & (POINTS -1)];
      }

    }


    for(unsigned row = 0; row < DEPTH; row++){

      float2 rotate_out[POINTS];

    /* Idea: Fetch transposed data that is already rotated
     *
     *  base : {0, N, 2N, .. 7N},
     *    each base fetches 8 points between [0, N] with strides of N / 8
     *
     *  offset : {0, 1, 2 .. }
     *    every N * 8 data item fetches, changes the offset by 1
     *
     *  rot :
     *    ((POINTS + i - (row / (N / 8))) * 8) & 63
     *
     *    row = 0, i = 0, rot =  (8 * 8) & 63 = 0
     *    row = 0, i = 1, rot =  (9 * 8) & 63 = 8
     *    row = 0, i = 2, rot = (10 * 8) & 63 = 16
     *    row = 0, ..
     *    row = 0, i = 7, rot = (15 * 8) & 63 = 56
     *
     *    row = 1, i = 0, rot = (8 * 8) & 63 = 0
     *    row = 1, i = 1, rot = (9 * 8) & 63 = 8
     *    ...
     *    row = 8, i = 0, rot = ((8 + 0 - 1) * 8) & 63 = (7 * 8) & 63 = 56
     *    row = 8, i = 1, rot = ((8 + 1 - 1) * 8) & 63 = (8 * 8) & 63 = 0
     *    row = 8, i = 2, rot = ((8 + 2 - 1) * 8) & 63 = (9 * 8) & 63 = 8
     *    row = 8,
     *    ...
     *    row = 9, i = 0, rot = (9 + 0 - 1) * 8) & 63 = (8 * 8) & 63 = 0
     *    row = 10, i = 0, rot = (10 + 0 - 1) * 8) & 63 = (9 * 8) & 63 = 8
     *    ...
     *    row = 16, i = 0, rot = 16 + 0 - 1) * 8) & 63 = 15 * 8) & 63 = 48
     *    row = 16, i = 1, rot = 16 + 1 - 1) * 8) & 63 = (8 * 8) & 63 = 56
     *    row = 16, i = 2, rot = 16 + 2 - 1) * 8) & 63 = (9 * 8) & 63 = 0
     *    ...
     *    row = 64, i=0, rot = 0
     *    row = 64, i=1, rot = 8
     *    ...
     *
     *   ---------------------------------------------------
     *
     *  row = 0,
     *    base : 0, offset : 0, rot : {0, 8, 16, .. 56},
     *       A[0][0], A[8][1], A[16][2] .. A[56][7]
     *  e.g. rotate_out = {0,64,128,..}
     *
     *  row = 1,
     *    base : N, offset : 0, rot : {0, 8, .. 48, 56},
     *       A[64][0], A[64 + 8][1], A[64 + 16][2]
     *  e.g. rotate_out = {512, 576,..}
     *
     *  row = 2,
     *    base : 2N, offset : 0, rot : {0, 8, .. 48, 56},
     *       A[128][0], A[128+8][1], A[128+16][2]
     *  E.g. rotate_out = {1024, 1088,..}
     *
     *  ... gets one column equivalent of data that is rotated every row
     *  ...
     *
     *
     *  row = 8,
     *    base : 0, offset : 0, rot : {56, 0, 8, .. 48},
     *       A[56][0], A[0][1], A[8][2]
     *  e.g. rotate_out = {449, 1, 65, 129.. }
     *
     *  row = 9,
     *    base : N, offset : 0, rot : {56, 0, 8, .. 48}
     *       A[64+56][0], A[64][1], A[64+8][2]
     *  e.g. rotate_out = {961, 513, 577, ..}
     *
     *  ...
     *  ... gets second column equivalent of data that is rotated every row
     *  ...
     *
     *  row = 64,
     *      base : 0, offset : 1, rot : {0, 8, 16, .. 56}
     *      A[1][0], A[9][1], A[17][2]
     *  row = 65,
     *      base : N, offset : 1, rot : {0, 8, 16, .. 56}
     *

     *  for 64^2 matrix,
     *  row_rotate = 0 + 0 + {0, 8, .. 56}
     *  row_rotate = 64 + 0 + {56, 0, .. }
     *  ...
     */

      unsigned base = (row & (N / POINTS - 1)) << LOGN; // 0, N, 2N, ...
      unsigned offset = row >> LOGN;                    // 0, .. N / POINTS

      // store data into temp buffer
      #pragma unroll 8
      for(unsigned i = 0; i < POINTS; i++){
        unsigned rot = ((POINTS + i - (row >> (LOGN - LOGPOINTS))) << (LOGN - LOGPOINTS)) & (N - 1);
        unsigned row_rotate  = base + offset + rot;
        rotate_out[i] = buf[row_rotate][i];
      }

      /*  Computing whether and how much to rotate.
       *  Idea: Rotate every N / 8 rows, wrap around after 8 rotations
       *
       *  Every (N / 8 rows): (row >> LOGN - LOGPOINTS)
       *  Every 8 rotations wrap around the banks, rot = (...) & 7
       *
       *  Values : [0, 1, .. , 7]
      */
      unsigned rot_out = row >> (LOGN - LOGPOINTS) & (POINTS - 1);

      write_channel_intel(chanoutTranspose[0], rotate_out[(0 + rot_out) & (POINTS - 1)]);
      write_channel_intel(chanoutTranspose[1], rotate_out[(1 + rot_out) & (POINTS - 1)]);
      write_channel_intel(chanoutTranspose[2], rotate_out[(2 + rot_out) & (POINTS - 1)]);
      write_channel_intel(chanoutTranspose[3], rotate_out[(3 + rot_out) & (POINTS - 1)]);
      write_channel_intel(chanoutTranspose[4], rotate_out[(4 + rot_out) & (POINTS - 1)]);
      write_channel_intel(chanoutTranspose[5], rotate_out[(5 + rot_out) & (POINTS - 1)]);
      write_channel_intel(chanoutTranspose[6], rotate_out[(6 + rot_out) & (POINTS - 1)]);
      write_channel_intel(chanoutTranspose[7], rotate_out[(7 + rot_out) & (POINTS - 1)]);

    }
  }

}

__attribute__((max_global_work_dim(0)))
kernel void store(global float2 * restrict dest, int iter) {
  const int N = (1 << LOGN);

  for(unsigned i = 0; i < iter * N * N / 8; i++){
    dest[(i * 8) + 0] = read_channel_intel(chanoutTranspose[0]);
    dest[(i * 8) + 1] = read_channel_intel(chanoutTranspose[1]);
    dest[(i * 8) + 2] = read_channel_intel(chanoutTranspose[2]);
    dest[(i * 8) + 3] = read_channel_intel(chanoutTranspose[3]);
    dest[(i * 8) + 4] = read_channel_intel(chanoutTranspose[4]);
    dest[(i * 8) + 5] = read_channel_intel(chanoutTranspose[5]);
    dest[(i * 8) + 6] = read_channel_intel(chanoutTranspose[6]);
    dest[(i * 8) + 7] = read_channel_intel(chanoutTranspose[7]);
  }

}