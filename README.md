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
