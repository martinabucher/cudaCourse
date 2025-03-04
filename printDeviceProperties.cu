#include <stdlib.h>
#include <stdio.h>

#include "helper.cuh"

int main(){
  cudaDeviceProp prop;
  int count;

  errCheck(cudaGetDeviceCount (&count));
  if(count==0){
    printf("Error: No CUDA enabled devices found.\n");
    exit(-1);
  }
  int device=0;
  errCheck(cudaGetDeviceProperties (&prop, device));

  printf("clockRate                   = %d (%g)\n", prop.clockRate                  , (double) prop.clockRate                  ); 
  printf("l2CacheSize                 = %d (%g)\n", prop.l2CacheSize                , (double) prop.l2CacheSize                ); 
  printf("major                       = %d (%g)\n", prop.major                      , (double) prop.major                      ); 
  printf("minor                       = %d (%g)\n", prop.minor                      , (double) prop.minor                      ); 
  printf("maxBlocksPerMultiProcessor  = %d (%g)\n", prop.maxBlocksPerMultiProcessor , (double) prop.maxBlocksPerMultiProcessor ); 
  printf("maxGridSize[0]              = %d (%g)\n", prop.maxGridSize[0]             , (double) prop.maxGridSize[0]             ); 
  printf("maxGridSize[1]              = %d (%g)\n", prop.maxGridSize[1]             , (double) prop.maxGridSize[1]             ); 
  printf("maxGridSize[2]              = %d (%g)\n", prop.maxGridSize[2]             , (double) prop.maxGridSize[2]             ); 
  printf("maxThreadsDim[0]            = %d (%g)\n", prop.maxThreadsDim[0]           , (double) prop.maxThreadsDim[0]           ); 
  printf("maxThreadsDim[1]            = %d (%g)\n", prop.maxThreadsDim[1]           , (double) prop.maxThreadsDim[1]           ); 
  printf("maxThreadsDim[2]            = %d (%g)\n", prop.maxThreadsDim[2]           , (double) prop.maxThreadsDim[2]           ); 
  printf("maxThreadsPerBlock          = %d (%g)\n", prop.maxThreadsPerBlock         , (double) prop.maxThreadsPerBlock         ); 
  printf("maxThreadsPerMultiProcessor = %d (%g)\n", prop.maxThreadsPerMultiProcessor, (double) prop.maxThreadsPerMultiProcessor);
  printf("memPitch                    = %ld (%g)\n", prop.memPitch                   , (double) prop.memPitch                   ); 
  printf("memoryBusWidth              = %d (%g)\n", prop.memoryBusWidth             , (double) prop.memoryBusWidth             ); 
  printf("memoryClockRate             = %d (%g)\n", prop.memoryClockRate            , (double) prop.memoryClockRate            ); 
  printf("multiProcessorCount         = %d (%g)\n", prop.multiProcessorCount        , (double) prop.multiProcessorCount        ); 
  printf("name[256]                   = %s \n", prop.name                ); 
  printf("persistingL2CacheMaxSize    = %d (%g)\n", prop.persistingL2CacheMaxSize  , (double) prop.persistingL2CacheMaxSize  ); 
  printf("regsPerBlock                = %d (%g)\n", prop.regsPerBlock              , (double) prop.regsPerBlock              ); 
  printf("regsPerMultiprocessor       = %d (%g)\n", prop.regsPerMultiprocessor     , (double) prop.regsPerMultiprocessor     ); 
  printf("sharedMemPerBlock           = %ld (%e)\n", prop.sharedMemPerBlock         , (double) prop.sharedMemPerBlock         ); 
  printf("sharedMemPerBlockOptin      = %ld (%e)\n", prop.sharedMemPerBlockOptin    , (double) prop.sharedMemPerBlockOptin    ); 
  printf("sharedMemPerMultiprocessor  = %ld (%e)\n", prop.sharedMemPerMultiprocessor, (double) prop.sharedMemPerMultiprocessor); 
  printf("totalConstMem               = %ld (%e)\n", prop.totalConstMem             , (double) prop.totalConstMem             ); 
  printf("totalGlobalMem              = %ld (%e) \n", (long int) prop.totalGlobalMem, (double) prop.totalGlobalMem); 
  printf("warpSize                    = %d\n", prop.warpSize                  ); 

  return 0;
}

// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
// https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp

/* These are the elements that seem of most interest:

* int     clockRate                    
* int     l2CacheSize                   
* int     major                       
* int     minor                       
* int     maxBlocksPerMultiProcessor  
* int     maxGridSize[3]             
* int     maxThreadsDim[3]             
* int     maxThreadsPerBlock              
* int     maxThreadsPerMultiProcessor     
* size_t  memPitch                      
* int     memoryBusWidth                
* int     memoryClockRate                 
* int     multiProcessorCount             
* char    name[256]                     
* int     persistingL2CacheMaxSize        
* int     regsPerBlock                    
* int     regsPerMultiprocessor              
* size_t  sharedMemPerBlock                  
* size_t  sharedMemPerBlockOptin           
* size_t  sharedMemPerMultiprocessor     
* size_t  totalConstMem                 
* size_t  totalGlobalMem               
* int     warpSize                             

*/ 


