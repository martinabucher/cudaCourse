
NVCC=/usr/local/cuda-12/bin/nvcc

SRC=$(shell ls *.cu)

EXECUTABLES=${SRC:.cu=}

all: ${EXECUTABLES}

#NVCCFLAGS= -O3 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75  -std=c++11 
NVCCFLAGS= -O3 -gencode arch=compute_80,code=sm_80 -std=c++11 

%:%.cu helper.cuh
	$(NVCC) -o $@ $(NVCCFLAGS) $<

clean:
	rm ${EXECUTABLES}
