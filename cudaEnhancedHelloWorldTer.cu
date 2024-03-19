#include <stdlib.h>
#include <stdio.h>

__global__ void helloWorldKernel();

int main(){

  cudaError_t err;
  
  printf("Calling kernel function on device.\n"); 
  helloWorldKernel<<<1,10>>>();
  printf("Returning from kernel function on device.\n"); 
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if ( err != cudaSuccess ){
     printf("CUDA Error: %s\n", cudaGetErrorString(err));       
     exit(0); 
  }

  return 0;
}

__global__ void helloWorldKernel(){
  for(int i=0;i<10;i++){
    if (threadIdx.x==i )
      printf("Hello world from device, block= %d, thread=%d \n", blockIdx.x, threadIdx.x); 
    __syncthreads(); 
  }
}

