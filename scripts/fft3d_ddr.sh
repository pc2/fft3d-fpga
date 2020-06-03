#!/bin/bash
#SBATCH -A pc2-mitarbeiter
#SBATCH -J 3dfftsyn
#SBATCH -p fpgasyn
#SBATCH --mem=90000MB 
#SBATCH --time=24:00:00

module load intelFPGA_pro/20.1.0 nalla_pcie/19.4.0_hpc
module load numlib/FFTW

cd ../build

cmake -DLOG_FFT_SIZE=5 ..
make
make fft3d_ddr_triv_syn
