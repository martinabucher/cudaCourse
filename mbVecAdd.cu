/* Parallel Implementation of Vector Addition 
   ------------------------------------------

This is a very simple implementation of parallel program
involving data created on the host (CPU), transferred to
the GPU. The data is then processed in parallel on the device
by a grid of instances of the kernel program and finally 
transferred back to the host. 

In practice, it is doubtful that any time would be saved
in carry out vector addition in this way. There are overheads
in transferring that the data back and forth between the 
host and kernel, and the work to be done per thread is so
little it is likely that no time will be saved, and perhaps
the program will take longer to run.

The object here is to illustrate CUDA programming concepts.
Only in our next example Matrix Multiplication can we expect
to be able to achieve significant speedup.

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define DIM (64*512*1024)     // 128 MB vector (32 M elements) 

__global__ void addVectorsKernel(float *c_d, float *a_d,float *b_d, int sz);

// The following is a placeholder:
 
#define errCheck(command)       errCheck2((command),#command,__FILE__,__LINE__)

inline void errCheck2(int command, const char *commandString, const char *file, int line){
    int value=command; 
    if ( value != cudaSuccess ){
      printf("%s  in file %s at line %d \n", commandString, file, line); 
      printf("Error: program aborting.\n");
      exit(-1); 
    }
}

int main(){
  

  // Create data (and vector to copy back result) 
  // -------------------------------------------

  // We dynamically allocate memory on the host (CPU) using malloc and free (as defined in stdlib.h of the C Standard Library)  
  // Upon error (for example if there is not enough memory available, malloc returns the null pointer. 
  // [For documentation, see for examle https://en.cppreference.com/w/c/memory/malloc

  float *a_h, *b_h, *c_h;

  a_h=(float*) malloc(DIM*sizeof(float)); 
  b_h=(float*) malloc(DIM*sizeof(float)); 
  c_h=(float*) malloc(DIM*sizeof(float)); 

  if ( a_h == NULL || b_h == NULL || c_h == NULL ){
    printf("Error: malloc failed. Exiting.\n"); 
    exit(-1);  //  'exit' is defined in stdlib.h --- the program is terminated with return status -1 here, meaning unsuccessful completion. 
  }

  // Now we use the random number generator of the C Standard Library to generte random input data. 

  srand(676);    //

  for(int i=0; i<DIM; i++){
    a_h[i]=((float) rand())/((float) RAND_MAX);
    b_h[i]=((float) rand())/((float) RAND_MAX);
  } 

 // The rand() random number generator produces a sequence of integer pseudo-random numbers from 0 to RAND_MAX (inclusive). 
 // This is not guaranteed to be a good random number generator and in many implementations is not. The seed is set by
 // by the call void srand( unsigned seed ), so each time this program is run the same sequence of pseudo-random numbers will result.
 // Seee https://en.cppreference.com/w/c/numeric/random/rand

  // Allocate global memory on device and transfer data from host to device
  // ----------------------------------------------------------------------

  float *a_d, *b_d, *c_d;

  errCheck(cudaMalloc((void**) &a_d,DIM*sizeof(float)));
  errCheck(cudaMalloc((void**) &b_d,DIM*sizeof(float)));
  errCheck(cudaMalloc((void**) &c_d,DIM*sizeof(float)));

  /* cudaMalloc is much like malloc. malloc cannot allocate memory on the device but only on the host.
     Another difference is that cudaMalloc (like other cuda.... functions) returns an error code rather
     than a pointer to space allocated. The latter is passed by address in the first argument, which must
     be cast to a pointer to a void pointer type. */ 

  errCheck(cudaMemcpy(a_d, a_h, DIM*sizeof(float), cudaMemcpyHostToDevice));
  errCheck(cudaMemcpy(b_d, b_h, DIM*sizeof(float), cudaMemcpyHostToDevice));
  
  // Here is the prototype: __host__ cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind );
  // [See https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html]

  // Launch grid of processes on device to carry out computation 
  // -----------------------------------------------------------

  float time_ms;
  cudaEvent_t     startBis,  stopBis;
  errCheck(cudaEventCreate(&startBis));
  errCheck(cudaEventCreate(&stopBis));

  errCheck(cudaEventRecord(startBis, 0));
  int sz=DIM;
  int threadsPerBlock=1024;
  int numBlocks=ceil(DIM/threadsPerBlock);
  addVectorsKernel<<<numBlocks,threadsPerBlock>>>(c_d,a_d,b_d, sz);
     // ERROR CHECKING
  cudaError_t err = cudaGetLastError();
  if ( err != cudaSuccess ){
     printf("CUDA Error: %s\n", cudaGetErrorString(err));   
     exit(-1); 
  }  
  errCheck(cudaEventRecord(stopBis, 0));
  errCheck(cudaEventSynchronize(stopBis));
  cudaEventElapsedTime(&time_ms, startBis, stopBis);
  printf("Kernel function time (ms) = %g\n", time_ms);

  // Copy data back from device to host
  errCheck(cudaMemcpy(c_h, c_d, DIM*sizeof(float), cudaMemcpyDeviceToHost));
     // same syntax as above and last argument indicates direction of transfer 

  // Compare result with host function 

  double c_h2[DIM];
  clock_t tStart=clock();
  for(int i=0; i<DIM; i++)
    c_h2[i]=a_h[i]+b_h[i]; 
  clock_t tEnd=clock();
  float timeH_ms=(1000.*(tEnd-tStart))/((float) CLOCKS_PER_SEC );
  printf("Host time (ms) = %e\n",timeH_ms);
  float eps=1.e-8;
  bool same=true;
  for(int i=0; i<DIM; i++)
    if ( fabs( c_h[i] - c_h2[i] ) > eps )
      same = false; 
  if (same == true) 
    printf("The host and device calculations agree.\n");
  else 
    printf("The host and device calculations do not agree.\n"); 
  return 0; 
} 

__global__ void addVectorsKernel(float *c_d, float *a_d,float *b_d, int sz){
  // Here the built-in variables (which are structures of three integers x, y, z) are used to compute a unique index for each thread.
  int index=blockDim.x*blockIdx.x+threadIdx.x; 
  c_d[index]= a_d[index]+ b_d[index];
}

