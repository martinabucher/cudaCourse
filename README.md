# cudaCourse

Makefile

cudaEnhancedHelloWorld.cu

   Launches 10 threads on the GPU and prints Hello world with thread number from each

cudaEnhancedHelloWorldBis.cu

   Same as above but with error checking improved to reduce clutter.

mbVecAdd.cu

   Example program to add two vectors.

matmulNaive_driver.cu
 
   Example program to multiply two large matrices (as timing)
   Naive implementation with large trafic from global memory

matmulTiled_driver.cu
 
   Same as above except tiling with shared memory is used to reduce trafic from global memory and thus achieve speedup.

my_reduce.cu

   Example performing "reduction."  A sum of many elements is calculated in parallel. 

