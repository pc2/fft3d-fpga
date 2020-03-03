# PCIe Latency

In the infrastructure of our choice, there is a 8 lane Gen 3 PCIe bus connecting the CPU and the 4 banks of external DDR4 memory of the Nallatech 520N board. The PCIe is connected to the FPGA's PCIe Hard IP. Each bank of the memory is 8GB per bank, with a total of 32 GB. Each lane of the bus gives a bandwidth of **985 MB/s** with a total of 7880 MB/s for 8 lanes or 7.69 GB/s.

The transfer rate depends on the block size used to transfer the data. Using the tool `aocl diagnose`, one can diagnose a top write speed of 6313.0 MB/s for writing 262144 KBs of memory using the 1 block of the same size, however this changes with different block sizes or average 3179.63 MB/s for writing 512 MB blocks. For read data, it also follows the same trend of 6412.3 MB/s and 3333 MB/s for 512 MB block sizes.

How is the FPGA Hard IP connected to DDR? Is it that we don't have the IP core that supports 16 lanes?

IP core contains Avalon MM for encoding/decoding of TLP and Avalon MM DMA for read write to DDR. 
