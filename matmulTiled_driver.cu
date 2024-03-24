#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define DIM_TILE   32
#define DIM_GRID   64
#define DIM    (DIM_TILE*DIM_GRID)

// The following is a placeholder:
 
#define errCheck(command)       errCheck2((command),#command,__FILE__,__LINE__)

inline void errCheck2(int command, const char *commandString, const char *file, int line){
    int value=command; 
    if ( value != cudaSuccess ){
      fprintf(stderr,"%s  in file %s at line %d \n", commandString, file, line); 
      fprintf(stderr,"Error: program aborting.\n");
      exit(-1); 
    }
}

__global__ void matmul_naive(int N, const float *A, const float *B, float *AB);
__global__ void matmul_tiled(int N, const float *A, const float *B, float *AB);

int main(){
  
  dim3 gridDim (DIM_GRID, DIM_GRID,   1);
  dim3 blockDim(DIM_TILE, DIM_TILE,  1);

  printf("Grid  dim = ( %d , %d )\n", gridDim.x, gridDim.y );
  printf("Block dim = ( %d , %d )\n", blockDim.x, blockDim.y );

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

  float *a_d, *b_d, *ab_d;

  errCheck(cudaMalloc((void**) &a_d, DIM*DIM*sizeof(float)));
  errCheck(cudaMalloc((void**) &b_d, DIM*DIM*sizeof(float)));
  errCheck(cudaMalloc((void**) &ab_d,DIM*DIM*sizeof(float)));

  errCheck(cudaMemcpy(a_d, a_h, DIM*DIM*sizeof(float), cudaMemcpyHostToDevice));
  errCheck(cudaMemcpy(b_d, b_h, DIM*DIM*sizeof(float), cudaMemcpyHostToDevice));
  
  float time_ms; cudaEvent_t startBis, stopBis;
  errCheck(cudaEventCreate(&startBis));
  errCheck(cudaEventCreate(&stopBis));

  errCheck(cudaEventRecord(startBis, 0));
  matmul_tiled<<<gridDim,blockDim>>>(DIM, a_d, b_d, ab_d); 
  cudaError_t err = cudaGetLastError(); // ERROR CHECKING
  errCheck(cudaEventRecord(stopBis, 0));
  if ( err != cudaSuccess ){
     fprintf(stderr,"CUDA Error: %s\n", cudaGetErrorString(err));   
     exit(-1); 
  }  
  errCheck(cudaEventSynchronize(stopBis)); 

  errCheck(cudaEventElapsedTime(&time_ms, startBis, stopBis));
  printf("Device timing (in milliseconds) = %g\n", time_ms);

  errCheck(cudaDeviceSynchronize());
  errCheck(cudaMemcpy(ab_h, ab_d, DIM*DIM*sizeof(float), cudaMemcpyDeviceToHost));
     // same syntax as above and last argument indicates direction of transfer 

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

  // Compare results
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


// below we assume that the matrix dimension is divisible by DIM_TILE
__global__ void matmul_tiled(int N, const float *A, const float *B, float *AB) {

  __shared__ float tile_a[DIM_TILE*DIM_TILE], tile_b[DIM_TILE*DIM_TILE];

  int i=threadIdx.x, j=threadIdx.y;                             // indices within tile
  int i_block=DIM_TILE*blockIdx.x, j_block=DIM_TILE*blockIdx.y; // tile offsets
  float sum=0.;
  for(int n=0; n<DIM_GRID; n++){                                // sum over blocks
    int n_block=DIM_TILE*n;                                         // tile offset
    // copy tiles
    tile_a[i*DIM_TILE+j]=A[DIM*(i+i_block)+(j+n_block)];
    tile_b[i*DIM_TILE+j]=B[DIM*(i+n_block)+(j+j_block)];
    __syncthreads();  // sync needed as each thread copies one element but elements copied by other threads are needed below
    for(int k=0;k<DIM_TILE;k++)   // accumulate a_tile@b_tile[i,j] to sum 
      sum+=tile_a[i*DIM_TILE+k]*tile_b[k*DIM_TILE+j];
    __syncthreads();  // make sure all threads are done before proceeding to copy new tiles
  }
  AB[(i+i_block)*DIM_TILE*DIM_GRID+(j+j_block)]=sum; 
} 

