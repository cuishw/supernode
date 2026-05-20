NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17
MXCC ?= nvcc
MXCCFLAGS ?= -O3 -std=c++17

all: nvidia muxi

nvidia: gpu_mem_bw_cuda
muxi: gpu_mem_bw_mxgpu_mc

gpu_mem_bw_cuda: gpu_mem_bw_cuda.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ $<

gpu_mem_bw_mxgpu_mc: gpu_mem_bw_mxgpu_mc.cu
	$(MXCC) $(MXCCFLAGS) -o $@ $<

clean:
	rm -f gpu_mem_bw_cuda gpu_mem_bw_mxgpu_mc
