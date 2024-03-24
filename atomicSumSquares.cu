
#include <stdio.h>

#define errCheck(command)       errCheck2((command),#command,__FILE__,__LINE__)

inline void errCheck2(int command, const char *commandString, const char *file, int line){
    int value=command;
    if ( value != cudaSuccess ){
      printf("%s  in file %s at line %d \n", commandString, file, line);
      printf("Error: program aborting.\n");
      exit(-1);
    }
}

__global__ void sum_squares(float* acc_d);
__device__    void incrementFun(float *accumulator, const float increment);

int main(){
  const int n=10;
  float accumulator_h=0.;
  float *accumulator_d;
  errCheck(cudaMalloc((void**) &accumulator_d,sizeof(float)));
  errCheck(cudaMemcpy(accumulator_d, &accumulator_h, sizeof(float), cudaMemcpyHostToDevice));
  sum_squares<<<1,n>>>(accumulator_d);
  cudaError_t err = cudaGetLastError();
  if ( err != cudaSuccess ){
     printf("CUDA Error: %s\n", cudaGetErrorString(err));
     exit(-1);
  }
  errCheck(cudaMemcpy(&accumulator_h, accumulator_d, sizeof(float), cudaMemcpyDeviceToHost));
  printf("Sum of squares = %f \n", accumulator_h);
  // Check
  float mySum=0.;
  for (int j=0;j<n;j++)
    mySum+=j*j;
  printf("Sum of squares = %f \n", mySum);
}

__global__ void sum_squares(float *acc_d){
  int i=threadIdx.x;
  float increment= (float) i*i;
  incrementFun(acc_d, increment);
}

__device__ void incrementFun(float *accumulator, const float increment){
  float old=*accumulator;
  while(true){
     float value=old+increment;
     int* value_int_ptr=(int*) &value;
     int value_int=*value_int_ptr;
     int* old_int_ptr=(int*) &old;
     int old_int=*old_int_ptr;
     int new_int=atomicCAS((int*) accumulator, old_int, value_int);
     if (new_int == old_int) break;
     float* new_float_ptr= (float*) &new_int;
     float newF=*new_float_ptr;
     old=newF;
  }
}
