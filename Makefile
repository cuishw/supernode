NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17
MXCC ?= nvcc
MXCCFLAGS ?= -O3 -std=c++17

all: nvidia muxi

nvidia: gpu_mem_bw_cuda
muxi: gpu_mem_bw_mxgpu_mc

gpu_mem_bw_cuda: gpu_mem_bw_unified.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

gpu_mem_bw_mxgpu_mc: gpu_mem_bw_unified.cu
	$(MXCC) $(MXCCFLAGS) -DUSE_MXGPU -o $@ $<

clean:
	rm -f gpu_mem_bw_cuda gpu_mem_bw_mxgpu_mc
