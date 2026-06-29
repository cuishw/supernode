NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17
MXCC ?= nvcc
MXCCFLAGS ?= -O3 -std=c++17

all: nvidia muxi register_mm

nvidia: gpu_mem_bw_cuda
muxi: gpu_mem_bw_mxgpu_mc register_mm

gpu_mem_bw_cuda: gpu_mem_bw_unified.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

gpu_mem_bw_mxgpu_mc: gpu_mem_bw_unified.cu
	$(MXCC) $(MXCCFLAGS) -DUSE_MXGPU -o $@ $<

register_mm: physmap_char/register_mm.cu
	$(MXCC) $(MXCCFLAGS) -o $@ $<

clean:
	rm -f gpu_mem_bw_cuda gpu_mem_bw_mxgpu_mc register_mm
