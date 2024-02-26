#define N 2048

#include <stdio.h>
#include <stdlib.h>

__global__ void scanFloat(float a[], const int n){

  /* The input array a[i] i=0..(n-1) is overwritten 
  so that upon return a[i] is replaced with 

            a[0]+a[1]+...+a[i] 

  where the latter expression in terms of the iput array. */

  int i=threadIdx.x;
  int m=1; 
  while ( n > m ){
    if ( i%(2*m) >= m ){
      index=i-(i%(2*m))+m-1;
      if (index >= 0)
        a[i]+=a[index];
    }
    m*=2;
    sync_threads();
  }
}

// We need to check that n does not exceed the allowed number of threads per block.

int main(){
  
  int n=N;
  float *a_h; 
  a_h=(float*) malloc(n*sizeof(float));
  for(int i=0;i<n;i++)
    a_h[i]=1.;
  float *a_d;

  cudaMalloc((void**) &a_d,n*sizeof(float));
  cudaMemcpy(a_d,a_h,n*sizeof(float), cudaMemcpyHostToDevice);

  scanFloat<<<1,n>>>(a_d, n); 

  cudaMemcpy(a_h,a_d,n*sizeof(float), cudaMemcpyDeviceToHost);
  for(int i=0;i<100;i++)
    printf("i= %d sum= %f \n",i,a_h[i]);
  free(a_h); 
  return 0; 
} 

