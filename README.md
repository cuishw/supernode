# GPU 内存读写带宽测试

包含两个程序，**逻辑保持一致**：

- `gpu_mem_bw_cuda.cpp`：NVIDIA CUDA 版本
- `gpu_mem_bw_mxgpu_mc.cu`：沐曦 MC Runtime 版本（把 NVIDIA 接口替换为沐曦接口）

两者都测试：
- 匿名 `mmap` 内存
- 字符设备 `mmap` 内存（可选）
- 输出 H2D / D2H / D2D 带宽

## NVIDIA

```bash
make nvidia
./gpu_mem_bw_cuda --bytes 268435456 --iters 30 --device 0
./gpu_mem_bw_cuda --bytes 268435456 --iters 30 --device 0 --char-dev /dev/your_char_device --offset 0
```

## 沐曦

```bash
make muxi
./gpu_mem_bw_mxgpu_mc --bytes 268435456 --iters 30 --device 0
./gpu_mem_bw_mxgpu_mc --bytes 268435456 --iters 30 --device 0 --char-dev /dev/your_char_device --offset 0
```
