
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <fstream>

#include "helper.cuh"

/* this is the kernel that will run in paralel on the GPU
* we use the blocks this time to paralellise since there are only 1024 threads in a block
* and the overlap in the brick sort will be a problem on odd sorts larger than 1024!
*/
__global__ void oddevenRefactor(int* x, int I, int n)
{
    int index = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + I;
    int secondPos = index + 1;
    if ((secondPos) < n) {
        if (x[index] > x[secondPos]) {
            int X = x[index];
            x[index] = x[secondPos];
            x[secondPos] = X;
        }
    }
}


/*
* perform the search and time it
*/
float sortAndTime(int* inputData, int* outputData, int size, int threadsPerBlock) {
    // start and stop variables
    cudaEvent_t     startBis;
    cudaEvent_t     stopBis;
    float time_ms;

    // pointer to data on device
    int* dData;

    // create the start and stop timing events
    errCheck(cudaEventCreate(&startBis));
    errCheck(cudaEventCreate(&stopBis));

    // allocate memory on the device for the input array
    errCheck(cudaMalloc((void**)&dData, size * sizeof(int)));

    // copy the input array from host memory to the memory location allocated on the device
    errCheck(cudaMemcpy(dData, inputData, size * sizeof(int), cudaMemcpyHostToDevice));

    // start timing the sort algorithm
    errCheck(cudaEventRecord(startBis, 0));
    // in host space, loop through n times as for brick sort the first value could need to be the last value in the array
    for (int i = 0; i < size; i++) {
        // call the kernal witn n/2 blocks with one thread each and pass the device memory location, iteration and size
        oddevenRefactor << <size / (2 * threadsPerBlock), threadsPerBlock >> > (dData, i % 2, size);
        // check if there were any errors in the kernel call and report
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "kernel call failed");
            cudaDeviceReset();
            exit(-1);
        }
        errCheck(cudaDeviceSynchronize());
    }
    // stop timing the sort algorithm
    errCheck(cudaEventRecord(stopBis, 0));
    errCheck(cudaEventSynchronize(stopBis));
    errCheck(cudaEventElapsedTime(&time_ms, startBis, stopBis));

    printf("sorted with %d threads per block \n", threadsPerBlock);

    // copy the sorted array from device back to the output array on the host
    errCheck(cudaMemcpy(outputData, dData, size * sizeof(int), cudaMemcpyDeviceToHost));

    // sync the device to ensure the copy is complete
    errCheck(cudaDeviceSynchronize());

    return time_ms;
}

/*
* perform the search and time it on the CPU
*/
float sortAndTimeCPU(int* inputData, int* outputData, int size) {
    // start and stop variables
    cudaEvent_t     startBis;
    cudaEvent_t     stopBis;
    float time_ms;

    // copy the input data to the output array
    // memcpy(outputData, inputData, size * sizeof(int));
    std::copy(inputData, inputData + size, outputData);

    // create the start and stop timing events
    errCheck(cudaEventCreate(&startBis));
    errCheck(cudaEventCreate(&stopBis));

    // start timing the sort algorithm
    errCheck(cudaEventRecord(startBis, 0));
    // in host space, loop through n times as for brick sort the first value could need to be the last value in the array
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size - 1; j++) {
            int index = j;
            int secondPos = index + 1;
            if ((secondPos) < size) {
                if (outputData[index] > outputData[secondPos]) {
                    int X = outputData[index];
                    outputData[index] = outputData[secondPos];
                    outputData[secondPos] = X;
                }
            }
        }
    }
    // stop timing the sort algorithm
    errCheck(cudaEventRecord(stopBis, 0));
    errCheck(cudaEventSynchronize(stopBis));
    errCheck(cudaEventElapsedTime(&time_ms, startBis, stopBis));

    printf("sorted on the cpu\n");

    // sync the device to ensure the copy is complete
    errCheck(cudaDeviceSynchronize());

    return time_ms;
}



int main(){

  int best_device=get_best_device(); 
  errCheck(cudaSetDevice(best_device)); 
	

    printf("starting experiment...\n");

    // run the experiment up till 135168
    int expSize = 33;

    for (int iter = 1; iter < expSize; iter++) {
        int n = iter * 4096;
        printf("starting experiment with %d size arrays...", n);

        // c++ standard way of getting random numbers in a uniform distribution using the Mercine Twister engine
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distr(1, n * 2);

        // create a new input array to be sorted
        int* a = new int[n];
        // assign random values to the input array at each index
        for (int i = 0; i < n; i++)
        {
            a[i] = distr(gen);
        }

        // output arrays
        int* aOut = new int[n];
        int* bOut = new int[n];
        int* cOut = new int[n];
        int* dOut = new int[n];
        int* eOut = new int[n];
        int* fOut = new int[n];
        int* gOut = new int[n];
        int* hOut = new int[n];

        // run the sort and time it with different thread sizes
        float firstSortTime = sortAndTime(a, aOut, n, 1);
        float secondSortTime = sortAndTime(a, bOut, n, 32);
        float thirdSortTime = sortAndTime(a, cOut, n, 64);
        float fourthSortTime = sortAndTime(a, dOut, n, 128);
        float fifthSortTime = sortAndTime(a, eOut, n, 256);
        float sixthSortTime = sortAndTime(a, fOut, n, 512);
        float seventhSortTime = sortAndTime(a, gOut, n, 1024);
        float cpuSortTime = sortAndTimeCPU(a, hOut, n);

        std::ofstream recordFile;
        recordFile.open("timingsExperiment.csv", std::ofstream::out | std::ofstream::app);
        recordFile << n << ", ";
        recordFile << firstSortTime << ", ";
        recordFile << secondSortTime << ", ";
        recordFile << thirdSortTime << ", ";
        recordFile << fourthSortTime << ", ";
        recordFile << fifthSortTime << ", ";
        recordFile << sixthSortTime << ", ";
        recordFile << seventhSortTime << ", ";
        recordFile << cpuSortTime << "\n";
        recordFile.close();
    }

    // reset the device to exit cleanly
    errCheck(cudaDeviceReset());

    return 0;
}
