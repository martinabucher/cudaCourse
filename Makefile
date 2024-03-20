
# /usr/local/cuda-12/bin/nvcc -o my_reduce -g -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 scan.cu

NVCC=/usr/local/cuda-12/bin/nvcc

all: cudaEnhancedHelloWorld cudaEnhancedHelloWorldBis mbVecAdd scan cudaEnhancedHelloWorldTer matmul_driver matmulTiled_driver my_reduce

# NVCCFLAGS= -g -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75
NVCCFLAGS= -O3 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75

%:%.cu 
	$(NVCC) -o $@ $(NVCCFLAGS) $<

