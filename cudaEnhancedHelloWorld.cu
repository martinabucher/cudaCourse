/* cudaEnhancedHelloWorld.cu

Based on the simplest example C program from the classic exposition of C,
Kernighan and Ritchie, The C Programming Language, this is an enhancement
of their hello world program. First we first "Hello World" from the host 
processor, and then from each of the device programs from the GPU
from a grid of threads that we set up.

Additionally this program calls a number of cuda functions from the host
probing the available GPU hardware and its capabilities. One way to find
out about the details of the available hardware is to ask someone what
has been installed on a particular computer that one is using, but another
option is probe the hardware using cuda functions as has been done here.

Here are the function prototypes of the CUDA functions used here, whose 
documnentation can be found at https://docs.nvidia.com/cuda/cuda-runtime-api/ :

__host__ __device__ cudaError_t cudaGetDeviceCount ( int* count );

__host__ cuda Error_t cudaGetDeviceProperties ( cudaDeviceProp* prop, int  device);

__host__ cudaError_t cudaSetDevice (int  device); 

*/

#include <stdlib.h>
#include <stdio.h>

__global__ void helloWorldKernel();

int main(){
  cudaError_t err;
  cudaDeviceProp prop;
  int device=0;
  int count;
  int major, minor;

  err=cudaGetDeviceCount(&count);
  if (err != cudaSuccess ){
    printf("Error: cudaGetDeviceCount call failed.\n");
    exit(-1);
  }
  if (count==0){
    printf("Error: No CUDA enabled devices found.\n");
    exit(-1);
  }
  printf("Found %d CUDA enabled devices.\n",count);
  for(device=0;device<count;device++){
    err=cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess){
      printf("Error: unable to probe device %d.\n",device);
      exit(-1);
    } 
    printf("Device number %d has compute capability %d.%d.\n",device, prop.major, prop.minor);
  }
  
  // Choose available device with the highest compute capability.

  int best_device=0;
  int best_major, best_minor;
  if (count == 1 ){
      best_device=0;
  } else {
      best_device=0;
      err=cudaGetDeviceProperties (&prop, best_device);
      if (err != cudaSuccess){
        printf("Error: unable to probe device %d.\n", best_device);
        exit(-1);}
      best_major=prop.major;
      best_minor=prop.minor;
      for(device=1;device<count;device++){
         err=cudaGetDeviceProperties (&prop, device);
         if (err != cudaSuccess){
           printf("Error: unable to probe device %d.\n", best_device);
           exit(-1);}
         major=prop.major; minor=prop.minor;
         bool better=false;
         if ( major>best_major )
           better = true;
         else if (major == best_major)
           if ( minor > best_minor)
             better=true;
         if (better){
           best_device=device;
           err=cudaGetDeviceProperties (&prop, best_device);
           if (err != cudaSuccess){
             printf("Error: unable to probe device %d.\n", best_device);
             exit(-1);}
           best_major=prop.major, best_minor=prop.minor;
         }
      }
  }
  printf("Best device = %d.\n",best_device); 

  err=cudaSetDevice(best_device); 
  if (err != cudaSuccess){
    printf("Error: cudaSetDevice failed.\n");
    exit(-1);
  }

  printf("Hello world. This is the host.\n"); 
  
  printf("Calling kernel function on device.\n"); 
  helloWorldKernel<<<1,10>>>();
  printf("Returning from kernel function on device.\n"); 
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if ( err != cudaSuccess ){
     printf("CUDA Error: %s\n", cudaGetErrorString(err));       
     exit(-1); 
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

/* Comments on the above kernel function call.

The keyword __global__ indicates to the nvcc compiler that this is a function to be called from the host 
program to launch a grid on the device. 

The syntax for defining a kernel function is as follows:

__global__ void kernelFunction(....){

}

A kernel function is always void. It cannot return results in the usual way. Results must be written to the device "global
memory" and copied back using cudaMemCpy, as we shall explain in later sample programs. 

The calling syntax within the host (or CPU) program is 

     kernelFunction<<numOfBlocks,threadsPerBlock>>(....); 

Here a grid of numOfBlocks blocks each containing threadsPerBlock. There are launched on the device numOfBlocks*threadsPerBlock
independent instances of the program, all running the same code as in the kernel function defined above. 

The different instances are distinguished by the variables
threadIdx.x and blockIdx.x   

** Here the grid is one dimensional, hence the structure member integer variable .x. As we shall later see, a grid, subdivded
into blocks, which are in turn subdivided into threads, can have a higher dimension, up to three dimensions. 

Here there is just one block composed of 10 threads running simultaneously. Without the for loop and the synchronisation barrier,
there is no guarantee as to the order of execution, and the output of more than one printf command can overlap, leading to
garbled output. Here the for loop ensures that each of the 10 threads speaks in turn, and the so-called barrier command

__syncthreads();

causes the thread execution is halt until all threads have reached this point before proceeding. 

Care must be taken to place __syncthreads() calls such that all threads call __syncthreads().
For example, __syncthreads() should not be placed in an if block, where in principle only a subset
of the threads reach the __syncthreads() call. In this case, according to the CUDA documentation that
behavior is "undefined". In this case it is likely that the program will simply hang and never finishing
because it is waiting for an event to happen that never occurs. 

Care must be taken to prevent such deadlock situations. 

*/ 

/* Additional comments

Above we have only probed two of the available device properties, but there are many more properties
available. The declaration in the above program

cudaDeviceProp prop;

defines a C structure defined in one of the CUDA header files, and here we cut and paste its definition
from the documentation in https://docs.nvidia.com/cuda/cuda-runtime-api/. Its many member variables provide
a host of invaluable information. 

Writing efficient (as opposed to simply formally correct) CUDA programs requires knowledge of the device
being used and its capabilities. And if one wants to write general purpose software that will run efficiently
on a wide range of CUDA devices, the algorithms and their parameters must be tuned according to the CUDA device
being used. Thus information obtained from this structure would be used to match the algorithm to the available 
hardware. 

Such a procedure is not unique to CUDA. Linear algebra on CPU computers is generally carried out using 
libraries such as the BLAS libraries or the Intel Math Kernel Library, which probe the hardware choosing
routines matched to the hardware for maximum speed. What these routines do is relatively trivial, and a
short functions would be written to carry out the specified calculation correctly. But one would have great
difficulty writing a routine on one's own that achieves the same performance.

    struct cudaDeviceProp {
              char name[256];
              cudaUUID_t uuid;
              size_t totalGlobalMem;
              size_t sharedMemPerBlock;
              int regsPerBlock;
              int warpSize;
              size_t memPitch;
              int maxThreadsPerBlock;
              int maxThreadsDim[3];
              int maxGridSize[3];
              int clockRate;
              size_t totalConstMem;
              int major;
              int minor;
              size_t textureAlignment;
              size_t texturePitchAlignment;
              int deviceOverlap;
              int multiProcessorCount;
              int kernelExecTimeoutEnabled;
              int integrated;
              int canMapHostMemory;
              int computeMode;
              int maxTexture1D;
              int maxTexture1DMipmap;
              int maxTexture1DLinear;
              int maxTexture2D[2];
              int maxTexture2DMipmap[2];
              int maxTexture2DLinear[3];
              int maxTexture2DGather[2];
              int maxTexture3D[3];
              int maxTexture3DAlt[3];
              int maxTextureCubemap;
              int maxTexture1DLayered[2];
              int maxTexture2DLayered[3];
              int maxTextureCubemapLayered[2];
              int maxSurface1D;
              int maxSurface2D[2];
              int maxSurface3D[3];
              int maxSurface1DLayered[2];
              int maxSurface2DLayered[3];
              int maxSurfaceCubemap;
              int maxSurfaceCubemapLayered[2];
              size_t surfaceAlignment;
              int concurrentKernels;
              int ECCEnabled;
              int pciBusID;
              int pciDeviceID;
              int pciDomainID;
              int tccDriver;
              int asyncEngineCount;
              int unifiedAddressing;
              int memoryClockRate;
              int memoryBusWidth;
              int l2CacheSize;
              int persistingL2CacheMaxSize;
              int maxThreadsPerMultiProcessor;
              int streamPrioritiesSupported;
              int globalL1CacheSupported;
              int localL1CacheSupported;
              size_t sharedMemPerMultiprocessor;
              int regsPerMultiprocessor;
              int managedMemory;
              int isMultiGpuBoard;
              int multiGpuBoardGroupID;
              int singleToDoublePrecisionPerfRatio;
              int pageableMemoryAccess;
              int concurrentManagedAccess;
              int computePreemptionSupported;
              int canUseHostPointerForRegisteredMem;
              int cooperativeLaunch;
              int cooperativeMultiDeviceLaunch;
              int pageableMemoryAccessUsesHostPageTables;
              int directManagedMemAccessFromHost;
              int accessPolicyMaxWindowSize;
          }

*/ 


