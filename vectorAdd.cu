//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>

#include "helper.cuh"

cudaError_t addWithCudaLambdaError(int* c, const int* a, const int* b, unsigned int size);
cudaError_t outputDeviceProperties(int best_device);


// this is the kernel that will run in paralel on the GPU
__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main(){

    int best_device=get_best_device(); 
    errCheck(cudaSetDevice(best_device)); 

    // read and print the properties of the CUDA device to the console
    errCheck(outputDeviceProperties(best_device));

    // set up our data that we want to use on the GPU for the parallel calculation
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel on the GPU using the helper function
    errCheck(addWithCudaLambdaError(c, a, b, arraySize));

    // output the results of the operation to the console
    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    errCheck(cudaDeviceReset());

    return 0;
}

cudaError_t outputDeviceProperties(int best_device){
    // Read device properties and print to console
    cudaDeviceProp prop;
    errCheck(cudaGetDeviceProperties_v2(&prop, best_device));

    printf("Global memory size: %zu\n", prop.totalGlobalMem);
    printf("L2 cache size: %d\n", prop.l2CacheSize);
    printf("Clock rate: %d\n", prop.clockRate);
    printf("Number of multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Blocks per multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Warp size: %d\n", prop.warpSize);

    return cudaSuccess;
}

cudaError_t addWithCudaLambdaError(int* c, const int* a, const int* b, unsigned int size){

    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Allocate GPU buffers for three vectors (two input, one output)    .
    errCheck(cudaMalloc((void**)&dev_c, size * sizeof(int)));

    errCheck(cudaMalloc((void**)&dev_a, size * sizeof(int)));
    errCheck(cudaMalloc((void**)&dev_b, size * sizeof(int)));

    // Copy input vectors from host memory to GPU buffers.
    errCheck(cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice));
    errCheck(cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice));

    // Launch a kernel on the GPU with one thread for each element.
    addKernel << <1, size >> > (dev_c, dev_a, dev_b);
    errCheck( cudaGetLastError()); // Check for any errors launching the kernel
    // cudaDeviceSynchronize waits for the kernel to finish and returns any errors encountered during the launch.
    errCheck(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
    errCheck(cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost));

    return cudaStatus; 
}


