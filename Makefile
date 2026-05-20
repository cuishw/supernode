NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17

all: gpu_mem_bw_cuda

gpu_mem_bw_cuda: gpu_mem_bw_cuda.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f gpu_mem_bw_cuda
