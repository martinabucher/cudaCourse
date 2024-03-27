#ifndef HELPER_GUARD
#define HELPER_GUARD

#define errCheck(command)       errCheck2((command),#command,__FILE__,__LINE__)

inline void errCheck2(int command, const char *commandString, const char *file, int line){
    int value=command;
    if ( value != cudaSuccess ){
      printf("%s  in file %s at line %d \n", commandString, file, line);
      printf("Error: program aborting.\n");
      exit(-1);
    }
}

inline int get_best_device(){

  cudaDeviceProp prop;
  int count;

  errCheck(cudaGetDeviceCount (&count));
  if(count==0){
    fprintf(stderr,"Error: No CUDA enabled devices found.\n");
    exit(-1);
  }
  for(int device=0;device<count;device++){
    errCheck(cudaGetDeviceProperties (&prop, device));
    printf("Device number %d has compute capability %d.%d.\n",device, prop.major, prop.minor);
  }

  // Choose available device with the highest compute capability.

  int best_device=0;
  int major, minor, best_major, best_minor;
  if (count == 1 ){
      best_device=0;
      best_major=prop.major, best_minor=prop.minor;
  } else {
      best_device=0;
      errCheck(cudaGetDeviceProperties (&prop, best_device));
      best_major=prop.major, best_minor=prop.minor;
      for(int device=1;device<count;device++){
         errCheck(cudaGetDeviceProperties (&prop, device));
         major=prop.major; minor=prop.minor;
         bool better=false;
         if ( major>best_major )
           better = true;
         else if (major == best_major)
           if (minor > best_minor)
             better=true;
         if (better){
           best_device=device;
           errCheck(cudaGetDeviceProperties (&prop, best_device));
           best_major=prop.major, best_minor=prop.minor;
         }
      }
  }
  printf("Best device = %d.\n",best_device);
  printf("Compute capability = %d.%d.\n", best_major, best_minor);
  return best_device;

} 

#endif 
