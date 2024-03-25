
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <fstream>

#include "helper.cuh" 


/* This is the kernel that will run in paralel on the GPU
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

cudaError_t outputDeviceProperties(int best_device)
{
    //cudaError_t cudaStatus;
    // read the device properties and print them to console
    cudaDeviceProp prop;
    //cudaStatus = cudaGetDeviceProperties_v2(&prop, 1);
    errCheck(
    cudaGetDeviceProperties_v2(&prop, best_device));
    //cudaStatus = cudaGetDeviceProperties_v2(&prop, 0);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return cudaStatus;
    //}
    printf("Global memory size: %zu\n", prop.totalGlobalMem);
    printf("L2 cache size: %d\n", prop.l2CacheSize);
    printf("Clock rate: %d\n", prop.clockRate);
    printf("Number of multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Blocks per multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Warp size: %d\n", prop.warpSize);
    //return cudaStatus;
    return cudaSuccess;

}

/*
* perform the search and time it
*/
float sortAndTime(int* inputData,int* outputData, int size, int threadsPerBlock) {
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
        oddevenRefactor << <size / (2* threadsPerBlock), threadsPerBlock >> > (dData, i % 2, size);
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



int main()
{
    int best_device=get_best_device(); 
    errCheck(cudaSetDevice(best_device)); 

    // maximum size of array that we will allow in this program
    const int MAX_ARRAY_SIZE = 262144;
    // input arrays, size
    int* a= new int[MAX_ARRAY_SIZE];
    int n;
    // read from console how many elements the input array should have and bound check it
    printf("Enter how many elements of input array (max %d):", MAX_ARRAY_SIZE);
    scanf("%d", &n);
    n = std::min(n, MAX_ARRAY_SIZE);

    // c++ standard way of getting random numbers in a uniform distribution using the Mercine Twister engine
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(1, MAX_ARRAY_SIZE * 2);

    // assign random values to the input array at each index
    for (int i = 0; i < n; i++)
    {
        a[i] = distr(gen);
    }

    errCheck(outputDeviceProperties(best_device));
    
    // output arrays
    int* aOut = new int[MAX_ARRAY_SIZE];
    int* bOut = new int[MAX_ARRAY_SIZE];
    int* cOut = new int[MAX_ARRAY_SIZE];
    int* dOut = new int[MAX_ARRAY_SIZE];
    int* eOut = new int[MAX_ARRAY_SIZE];
    int* fOut = new int[MAX_ARRAY_SIZE];
    int* gOut = new int[MAX_ARRAY_SIZE];
    int* hOut = new int[MAX_ARRAY_SIZE];

    // run the sort and time it with different thread sizes
    float firstSortTime = sortAndTime(a, aOut, n, 1);
    float secondSortTime = sortAndTime(a, bOut, n, 32);
    float thirdSortTime = sortAndTime(a, cOut, n, 64);
    float fourthSortTime = sortAndTime(a, dOut, n, 128);
    float fifthSortTime = sortAndTime(a, eOut, n, 256);
    float sixthSortTime = sortAndTime(a, fOut, n, 512);
    float seventhSortTime = sortAndTime(a, gOut, n, 1024);
    float cpuSortTime = sortAndTimeCPU(a, hOut, n);
    
    // output the timing results for each sort
    printf("Sort time with %d blocks with 1 thread each (in milliseconds) = %g\n", n/2, firstSortTime);
    printf("Sort time with %d blocks with %d threads each (in milliseconds) = %g\n", n/(2*32), 32, secondSortTime);
    printf("Sort time with %d blocks with %d threads each (in milliseconds) = %g\n", n / (2*64), 64, thirdSortTime);
    printf("Sort time with %d blocks with %d threads each (in milliseconds) = %g\n", n / (2*128), 128, fourthSortTime);
    printf("Sort time with %d blocks with %d threads each (in milliseconds) = %g\n", n / (2*256), 256, fifthSortTime);
    printf("Sort time with %d blocks with %d threads each (in milliseconds) = %g\n", n / (2*512), 512, sixthSortTime);
    printf("Sort time with %d blocks with %d threads each (in milliseconds) = %g\n", n / (2*1024), 1024, seventhSortTime);
    printf("Sort time with the cpu (in milliseconds) = %g\n", cpuSortTime);

    // compare the output arrays to make sure the sort worked correctly for all variations
    int sizeOfOne = sizeof(aOut) / sizeof(*aOut);
    int sizeOfTwo = sizeof(bOut) / sizeof(*bOut);
    int sizeOfThree = sizeof(cOut) / sizeof(*cOut);
    int sizeOfFour = sizeof(dOut) / sizeof(*dOut);
    int sizeOfFive = sizeof(eOut) / sizeof(*eOut);
    int sizeOfSix = sizeof(fOut) / sizeof(*fOut);
    int sizeOfSeven = sizeof(gOut) / sizeof(*gOut);
    int sizeOfCpu = sizeof(hOut) / sizeof(*hOut);
    if (sizeOfOne == sizeOfTwo && sizeOfTwo == sizeOfThree && sizeOfThree == sizeOfFour && sizeOfFour == sizeOfFive && sizeOfFive == sizeOfSix && sizeOfSix == sizeOfSeven && sizeOfSeven == sizeOfCpu &&
        std::equal(aOut, aOut + sizeOfOne, bOut) &&
        std::equal(bOut, bOut + sizeOfTwo, cOut) &&
        std::equal(cOut, cOut + sizeOfThree, dOut) &&
        std::equal(dOut, dOut + sizeOfFour, eOut) &&
        std::equal(eOut, eOut + sizeOfFive, fOut) &&
        std::equal(fOut, fOut + sizeOfSix, gOut) &&
        std::equal(gOut, gOut + sizeOfSeven, hOut)) {
        printf("arrays are all equal!");
        std::ofstream recordFile;
        recordFile.open("timings.csv", std::ofstream::out | std::ofstream::app);
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
    else {
        printf("arrays are not all equal!");
        // output the results of the sort operation to file to debug. Only first 4096 rows are written
        std::ofstream outputFile("errorOutputFile.csv");
        outputFile << "input, 1 thread, 32 threads, 64 threads, 128 threads, 256 threads, 512 threads, 1024 threads, CPU \n";
        for (int i = 0; i < std::min(n, 4096); i++)
        {
            printf("%d \t %d \t %d \t %d\t %d \t %d \t %d \t %d \t %d \n", a[i], aOut[i], bOut[i], cOut[i], dOut[i], eOut[i], fOut[i], gOut[i], hOut[i]);
            outputFile << a[i] << ", ";
            outputFile << aOut[i] << ", ";
            outputFile << bOut[i] << ", ";
            outputFile << cOut[i] << ", ";
            outputFile << dOut[i] << ", ";
            outputFile << eOut[i] << ", ";
            outputFile << fOut[i] << ", ";
            outputFile << gOut[i] << ", ";
            outputFile << hOut[i] << "\n";
        }
        outputFile.close();
    }

    // reset the device to exit cleanly
    errCheck(cudaDeviceReset());

    return 0;
}
