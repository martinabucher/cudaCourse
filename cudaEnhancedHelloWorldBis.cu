/* CUDA HelloWorld++Bis.cu

Improved Error Handling With Less Clutter
-----------------------------------------

Here we revist the previous version of this program modifying the error handling.
We have stripped all the commentary, and new commentary has been added only to
explain the error handling using the C++PPP (C++ preprocessor) and the C++ inline 
function functionality. 

In the previous version a large fraction of the lines of code dealt with error handling,
and in the normal course of execution these lines of code should never be reached. One 
may be tempted to skip over handling errors that could occur, but code of this sort,
while easier to write is likely to take longer to debug. No one wants to use a code 
written by someone else to find that the code crashes for example with a segmentation
fault leaving the using to try to uncover from the file and source code what went wrong.
It is much better for the program to end gracefully with a text error message as to
what failed. 

The C++ language (much like Python) provides an error handling facility whereby 
when errors occur they can "throw exceptions", which can be "caught" and "handled"...

Even though CUDA is an extension of C/C++ CUDA error handling is more in the style
of old-fashioned C programs. Functions return an integer error code, which should
be checked after the function call by appropriate calls. 

A cudaError_t is introduced ....

Here we copy and explain the error handling as implemented in the CUDA Sample Codes
available on github.

ETC

The basic idea is in the code to wrap the CUDA functions that may result in error
using the syntax 

checkError(.....);

and put all of the error code in the definition of 'checkError'. This results
is more compact and easier to read code. There is no to write an if block
at each place an error may occur. However there is some loss in flexibility.

#define checkError(function_call) check(function_call, #function_call, __FILE__, __LINE__)

inline check 

A CUDA kernel function (called from the host to set up a grid of device processes)
is of the type void and thus does not return an error code. Nevertheless its error
code may be obtained


     CUDA kernel function call;

     cudaError_t err = cudaGetLastError();
     if ( err != cudaSuccess )
     {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));       

        // Possibly: exit(-1) if program cannot continue....
     }

*/

#include <stdlib.h>
#include <stdio.h>

__global__ void helloWorldKernel();

#define checkError(command)       errCheck2((command),#command,__FILE__,__LINE__)

inline void errCheck2(int command, const char *commandString, const char *file, int line){
    int value=command;
    if ( value != cudaSuccess ){
      printf("%s  in file %s at line %d \n", commandString, file, line);
      printf("Error: program aborting.\n");
      exit(0);
    }
}

int main(){
  cudaDeviceProp prop;
  int count;

  checkError(cudaGetDeviceCount (&count));
  if(count==0){
    printf("Error: No CUDA enabled devices found.\n");
    exit(-1);
  }
  for(int device=0;device<count;device++){
    checkError(cudaGetDeviceProperties (&prop, device));
    printf("Device number %d has compute capability %d.%d.",device, prop.major, prop.minor);
  }
  
  // Choose available device with the highest compute capability.

  int best_device=0;
  int major, minor, best_major, best_minor;
  if (count == 1 ){
      best_device=0;
  } else {
      best_device=0;
      checkError(cudaGetDeviceProperties (&prop, best_device));
      best_major=prop.major, best_minor=prop.minor;
      for(int device=1;device<count;device++){
         checkError(cudaGetDeviceProperties (&prop, device));
         major=prop.major; minor=prop.minor;
         bool better=false;
         if ( major>best_major )
           better = true;
         else if (major == best_major)
           if (minor > best_minor)
             better=true;
         if (better){
           best_device=device;
           checkError(cudaGetDeviceProperties (&prop, best_device));
           best_major=prop.major, best_minor=prop.minor;
         }
      }
  }
  printf("Best device = %d.\n",best_device); 

  checkError(cudaSetDevice(best_device)); 

  printf("Hello world. This is the host.\n"); 
  
  printf("Calling kernel function on device.\n"); 
  helloWorldKernel<<<1,10>>>();
  printf("Returning from kernel function on device.\n"); 
  checkError(cudaDeviceSynchronize());

  return 0;
}

__global__ void helloWorldKernel(){
  for(int i=0 ;i<10;i++)
    if (threadIdx.x==i )
      printf("Hello world from device, block= %d, thread=%d \n", blockIdx.x, threadIdx.x); 
    __syncthreads(); 
}



