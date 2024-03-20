
#include <stdio.h>

__device__ void reduce_sum(float *s, const int n, float *sum);
__global__ void testReduceSum(float* s, const int n, float *result);

__device__ void reduce_sum(float *s, const int n, float *sum){
  if (n == 1){   // eliminate trivial case 
     *sum = s[0]; 
  }
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int step=1;
  while(1){
    if (2*step >= n ) break;
    step*=2;
  }
  if (index == 0 )
     printf("step %d\n",step); 
  int n_max=n; 
  while(1){
    if ( index + step < n_max )
      s[index]+=s[index+step]; 
    __syncthreads();
    n_max=step;
    if ( step==1 ) break;
    step/=2;
  }
  *sum=s[0];
}

__global__ void testReduceSum(float* s, const int n, float *result){
 reduce_sum(s, n, result);
}

int main(void){
  const int thread_dim=10;
  const int block_dim =10;
  int n=100; 
  size_t size=n*sizeof(float);
  float* s_d; 
  float *result_d;
  float result_h;
  cudaMalloc((void **) &s_d, size);
  cudaMalloc((void **) &result_d,sizeof(float));
  float s_h[size];
  float local_sum=0.; 
  for(int j=0;j<n;j++){
    s_h[j]=(float) (j+1); 
    local_sum+=s_h[j];}
  printf("Expected sum is equal to %f \n", local_sum);
  cudaMemcpy(s_d, s_h, size, cudaMemcpyHostToDevice);
  testReduceSum<<<block_dim,thread_dim>>>(s_d,n,result_d); 
  cudaMemcpy(&result_h, result_d, sizeof(float), cudaMemcpyDeviceToHost);
  printf("CUDA sum is equal to %f \n", result_h);
}

