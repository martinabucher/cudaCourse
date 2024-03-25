//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>

cudaError_t addWithCudaLambdaError(int* c, const int* a, const int* b, unsigned int size);
cudaError_t outputDeviceProperties();


// this is the kernel that will run in paralel on the GPU
__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    // error variable to catch anything reported by cuda functions
    cudaError_t cudaStatus;

    // read and print the properties of the CUDA device to the console
    cudaStatus = outputDeviceProperties();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed reading device properties!");
        return 1;
    }

    // set up our data that we want to use on the GPU for the parallel calculation
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel on the GPU using the helper function
    cudaStatus = addWithCudaLambdaError(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // output the results of the operation to the console
    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t outputDeviceProperties()
{
    cudaError_t cudaStatus;
    // read the device properties and print them to console
    cudaDeviceProp prop;
    //cudaStatus = cudaGetDeviceProperties_v2(&prop, 1);
    cudaStatus = cudaGetDeviceProperties_v2(&prop, 0);
    //cudaStatus = cudaGetDeviceProperties(&prop,0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return cudaStatus;
    }
    printf("Global memory size: %zu\n", prop.totalGlobalMem);
    printf("L2 cache size: %d\n", prop.l2CacheSize);
    printf("Clock rate: %d\n", prop.clockRate);
    printf("Number of multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Blocks per multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Warp size: %d\n", prop.warpSize);
    return cudaStatus;

}

cudaError_t addWithCudaLambdaError(int* c, const int* a, const int* b, unsigned int size)
{

    // lambda expression to clear memory and return exit message
    auto exitWithStatus = [](int* a, int* b, int* c, cudaError_t errstate, std::string errmsg = "") {
        cudaFree(c);
        cudaFree(a);
        cudaFree(b);
        if (errmsg != "") { fprintf(stderr, errmsg.c_str()); };
        return errstate;
        };

    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        return exitWithStatus(dev_a, dev_b, dev_c, cudaStatus, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        return exitWithStatus(dev_a, dev_b, dev_c, cudaStatus, "cudaMalloc failed!");
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        return exitWithStatus(dev_a, dev_b, dev_c, cudaStatus, "cudaMalloc failed!");
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        return exitWithStatus(dev_a, dev_b, dev_c, cudaStatus, "cudaMalloc failed!");
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        return exitWithStatus(dev_a, dev_b, dev_c, cudaStatus, "cudaMemcpy failed!");
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        return exitWithStatus(dev_a, dev_b, dev_c, cudaStatus, "cudaMemcpy failed!");
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel << <1, size >> > (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        return exitWithStatus(dev_a, dev_b, dev_c, cudaStatus, std::string("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus)));
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        return exitWithStatus(dev_a, dev_b, dev_c, cudaStatus, std::string("cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus));
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        return exitWithStatus(dev_a, dev_b, dev_c, cudaStatus, "cudaMemcpy failed!");
    }

    return exitWithStatus(dev_a, dev_b, dev_c, cudaStatus);
}


