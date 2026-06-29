# GPU 内存读写带宽测试（统一源码）

现在只保留一个源码文件：`gpu_mem_bw_unified.cu`。

- 默认按 **NVIDIA 接口** 编译/调用。
- 在沐曦上通过 `-DUSE_MXGPU` 宏把接口重定向到 `mc_runtime` 对应接口。

## 编译

```bash
make nvidia   # 生成 gpu_mem_bw_cuda
make muxi     # 生成 gpu_mem_bw_mxgpu_mc（启用 -DUSE_MXGPU）
```

## 运行

```bash
./gpu_mem_bw_cuda --bytes 268435456 --iters 30 --device 0
./gpu_mem_bw_mxgpu_mc --bytes 268435456 --iters 30 --device 0
```

可选字符设备 mmap：

```bash
--char-dev /dev/your_char_device --offset 0
```

## 注册 flag

- 匿名 mmap：IoMemory flag
- 字符设备 mmap：Default flag

## 沐曦注册系统/映射内存保活测试

`physmap_char/register_mm.cu` 用于在沐曦 GPU 上调用 `mcHostRegister` 注册一段主机侧内存，并保持进程不退出，便于观察驱动或设备状态。可通过 `SIGINT`/`SIGTERM` 触发反注册并退出。

```bash
make register_mm
./register_mm --device 0 --bytes 1073741824 --mem system
./register_mm --device 0 --bytes 268435456 --mem mmap
./register_mm --device 0 --bytes 4096 --mem mmap --mmap-path /dev/your_mmio_device --offset 0 --mmio
```

- `--mem system` 使用页对齐系统内存并以 `mcHostRegisterDefault` 注册。
- `--mem mmap` 可匿名映射普通内存，也可用 `--mmap-path` 映射文件或字符设备。
- `--mmio` 仅用于 `--mem mmap`，使用 `mcHostRegisterIoMemory` 注册 MMIO 映射。
