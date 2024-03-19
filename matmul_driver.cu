#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//#define DIM (4*32)     
#define DIM (64*32)     

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// The following is a placeholder:
 
#define errCheck(command)       errCheck2((command),#command,__FILE__,__LINE__)

inline void errCheck2(int command, const char *commandString, const char *file, int line){
    int value=command; 
    if ( value != cudaSuccess ){
      printf("%s  in file %s at line %d \n", commandString, file, line); 
      printf("Error: program aborting.\n");
      exit(0); 
    }
}

__global__ void matmul_naive(int N, const float *A, const float *B, float *AB);

int main(){
  
  dim3 gridDim (CEIL_DIV(DIM,32), CEIL_DIV(DIM, 32),1);
  dim3 blockDim(32,               32,           1);

  printf("Grid  dim = ( %d , %d )\n", gridDim.x, gridDim.y );
  printf("Block dim = ( %d , %d )\n", blockDim.x, blockDim.y );

  // Create data (and vector to copy back result) 
  // -------------------------------------------

  // We dynamically allocate memory on the host (CPU) using malloc and free (as defined in stdlib.h of the C Standard Library)  
  // Upon error (for example if there is not enough memory available, malloc returns the null pointer. 
  // [For documentation, see for examle https://en.cppreference.com/w/c/memory/malloc

  float *a_h, *b_h, *ab_h;

  a_h =(float*) malloc(DIM*DIM*sizeof(float)); 
  b_h =(float*) malloc(DIM*DIM*sizeof(float)); 
  ab_h=(float*) malloc(DIM*DIM*sizeof(float)); 

  if ( a_h == NULL || b_h == NULL || ab_h == NULL ){
    fprintf(stderr,"Error: malloc failed. Exiting.\n"); 
    exit(-1);  //  'exit' is defined in stdlib.h --- the program is terminated with return status 0 here, meaning successful completion. 
  }

  // Now we use the random number generator of the C Standard Library to generte random input data. 

  srand(676);    //

  for(int i=0; i<DIM*DIM; i++){
    a_h[i]=((float) rand())/((float) RAND_MAX);
    b_h[i]=((float) rand())/((float) RAND_MAX);
  } 

 // The rand() random number generator produces a sequence of integer pseudo-random numbers from 0 to RAND_MAX (inclusive). 
 // This is not guaranteed to be a good random number generator and in many implementations is not. The seed is set by
 // by the call void srand( unsigned seed ), so each time this program is run the same sequence of pseudo-random numbers will result.
 // Seee https://en.cppreference.com/w/c/numeric/random/rand

  // Allocate global memory on device and transfer data from host to device
  // ----------------------------------------------------------------------

  float *a_d, *b_d, *ab_d;

  errCheck(cudaMalloc((void**) &a_d, DIM*DIM*sizeof(float)));
  errCheck(cudaMalloc((void**) &b_d, DIM*DIM*sizeof(float)));
  errCheck(cudaMalloc((void**) &ab_d,DIM*DIM*sizeof(float)));

  /* cudaMalloc is much like malloc. malloc cannot allocate memory on the device but only on the host.
     Another difference is that cudaMalloc (like other cuda.... functions) returns an error code rather
     than a pointer to space allocated. The latter is passed by address in the first argument, which must
     be cast to a pointer to a void pointer type. */ 

  errCheck(cudaMemcpy(a_d, a_h, DIM*DIM*sizeof(float), cudaMemcpyHostToDevice));
  errCheck(cudaMemcpy(b_d, b_h, DIM*DIM*sizeof(float), cudaMemcpyHostToDevice));
  
  // Here is the prototype: __host__ cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind );
  // [See https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html]

  // Launch grid of processes on device to carry out computation 
  // -----------------------------------------------------------

  cudaEvent_t     startBis;
  cudaEvent_t     stopBis;
  float time_ms;

  errCheck(cudaEventCreate(&startBis));
  errCheck(cudaEventCreate(&stopBis));

  //clock_t start=clock();
  errCheck(cudaEventRecord(startBis, 0));
  int threadsPerBlock=1024;
  int numBlocks=ceil(DIM/threadsPerBlock);
  matmul_naive<<<gridDim,blockDim>>>(DIM, a_d, b_d, ab_d); 
     // ERROR CHECKING
  cudaError_t err = cudaGetLastError();
  if ( err != cudaSuccess ){
     fprintf(stderr,"CUDA Error: %s\n", cudaGetErrorString(err));   
     exit(-1); 
  }  
  errCheck(cudaDeviceSynchronize());
  //clock_t end=clock();

  errCheck(cudaEventRecord(stopBis, 0));
  errCheck(cudaEventSynchronize(stopBis)); 
  errCheck(cudaEventElapsedTime(&time_ms, startBis, stopBis));
  printf("Kernel timing (ms) = %g\n", time_ms);

  //float timing= ( (float) (end-start) )/( (float) CLOCKS_PER_SEC );
  //printf("Kernel function time= %e secs. \n",timing);
  //printf("CLOCKS_PER_SEC= %d\n", CLOCKS_PER_SEC);

  // Copy data back from device to host


  errCheck(cudaDeviceSynchronize());
  errCheck(cudaMemcpy(ab_h, ab_d, DIM*DIM*sizeof(float), cudaMemcpyDeviceToHost));
     // same syntax as above and last argument indicates direction of transfer 
  errCheck(cudaDeviceSynchronize());

  // Compare result with host function 

  clock_t start=clock();
  float ab_h2[DIM*DIM];
  for(int i=0; i<DIM; i++)
    for(int j=0; j<DIM; j++){
      float sum=0.0;
      for(int k=0; k<DIM; k++)
         sum+=a_h[i*DIM+k]*b_h[k*DIM+j]; 
      ab_h2[i*DIM+j]=sum;
    }
  clock_t end=clock();
  float timing= ( (float) (end-start) )/( (float) CLOCKS_PER_SEC );
  timing/=1.e-3; 
  printf("Host function time= %e millisecs. \n",timing);
  printf("CLOCKS_PER_SEC= %e\n", (float) CLOCKS_PER_SEC);
  float speedup_factor=timing/time_ms;
  printf("Speedup factor = %e\n", speedup_factor);

  float eps=1.e-3;
  bool same=true;
  for(int i=0; i<DIM; i++)
    for(int j=0; j<DIM; j++){
      if ( fabs( ab_h[i*DIM +j] - ab_h2[i*DIM+j] ) > eps )
        same = false; 
    }
  if (same == true) 
    printf("The host and device calculations agree.\n");
  else 
    printf("The host and device calculations do not agree.\n"); 
  return 0; 
} 


__global__ void matmul_naive(int N, const float *A, const float *B, float *AB) {
  // compute position in AB that this thread is responsible for

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for N is not a multiples of 32

  if (x < N && y < N) {
    float sum = 0.0;
    for (int i = 0; i < N; ++i) 
      sum += A[x * N + i] * B[i * N + y];
    AB[x * N + y] = sum;
  }
}

