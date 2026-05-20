# NVIDIA GPU 内存读写带宽测试（CUDA）

这是一个**先在 NVIDIA GPU 上运行**的测试程序，用于比较两种 `mmap` 来源内存的带宽：

1. 匿名 `mmap`（`MAP_ANONYMOUS`）
2. 字符设备 `mmap`（`--char-dev /dev/xxx`）

输出三项带宽：

- H2D（Host -> Device）
- D2H（Device -> Host）
- D2D（Device -> Device）

## 编译

```bash
make
```

## 运行示例

### 仅测试匿名 mmap

```bash
./gpu_mem_bw_cuda --bytes 268435456 --iters 30 --device 0
```

### 同时测试字符设备 mmap

```bash
./gpu_mem_bw_cuda \
  --bytes 268435456 \
  --iters 30 \
  --device 0 \
  --char-dev /dev/your_char_device \
  --offset 0
```

## 参数说明

- `--bytes`：每轮传输字节数（默认 `256 MiB`）
- `--iters`：迭代次数（默认 `30`）
- `--device`：NVIDIA GPU 索引（默认 `0`）
- `--char-dev`：字符设备路径（不传则跳过字符设备测试）
- `--offset`：字符设备映射偏移（默认 `0`）

## 备注

- 该版本仅面向 NVIDIA/CUDA 环境。
- 字符设备测试要求设备支持 `mmap`，并满足驱动对长度/偏移（通常页对齐）的要求。
