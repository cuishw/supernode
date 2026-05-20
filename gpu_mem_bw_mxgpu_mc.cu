#include <mc_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

struct Options {
  size_t bytes = 256ull * 1024 * 1024;
  int iters = 30;
  int device = 0;
  std::string char_dev;
  size_t char_dev_offset = 0;
};

static void check(mcError_t err, const char* what) {
  if (err != mcSuccess) {
    throw std::runtime_error(std::string(what) + " failed: " + mcGetErrorString(err));
  }
}

static double gbps(size_t bytes, float ms) {
  return static_cast<double>(bytes) / (ms / 1000.0) / 1e9;
}

static void* map_anon(size_t bytes) {
  void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) throw std::runtime_error("mmap anon failed");
  return p;
}

struct DevMap {
  int fd = -1;
  void* ptr = nullptr;
  size_t bytes = 0;
};

static DevMap map_chardev(const std::string& path, size_t bytes, size_t off) {
  DevMap m;
  m.fd = open(path.c_str(), O_RDWR | O_SYNC);
  if (m.fd < 0) throw std::runtime_error("open char dev failed: " + path);
  m.ptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, m.fd, off);
  if (m.ptr == MAP_FAILED) {
    close(m.fd);
    throw std::runtime_error("mmap char dev failed");
  }
  m.bytes = bytes;
  return m;
}

static float run_h2d(void* dptr, const void* hptr, size_t bytes, int iters) {
  mcEvent_t s, e;
  check(mcEventCreate(&s), "mcEventCreate start");
  check(mcEventCreate(&e), "mcEventCreate end");
  float total_ms = 0;
  for (int i = 0; i < iters; ++i) {
    check(mcEventRecord(s), "mcEventRecord start");
    check(mcMemcpy(dptr, hptr, bytes, mcMemcpyHostToDevice), "mcMemcpy H2D");
    check(mcEventRecord(e), "mcEventRecord end");
    check(mcEventSynchronize(e), "mcEventSynchronize");
    float ms = 0;
    check(mcEventElapsedTime(&ms, s, e), "mcEventElapsedTime");
    total_ms += ms;
  }
  mcEventDestroy(s);
  mcEventDestroy(e);
  return total_ms / iters;
}

static float run_d2h(void* hptr, const void* dptr, size_t bytes, int iters) {
  mcEvent_t s, e;
  check(mcEventCreate(&s), "mcEventCreate start");
  check(mcEventCreate(&e), "mcEventCreate end");
  float total_ms = 0;
  for (int i = 0; i < iters; ++i) {
    check(mcEventRecord(s), "mcEventRecord start");
    check(mcMemcpy(hptr, dptr, bytes, mcMemcpyDeviceToHost), "mcMemcpy D2H");
    check(mcEventRecord(e), "mcEventRecord end");
    check(mcEventSynchronize(e), "mcEventSynchronize");
    float ms = 0;
    check(mcEventElapsedTime(&ms, s, e), "mcEventElapsedTime");
    total_ms += ms;
  }
  mcEventDestroy(s);
  mcEventDestroy(e);
  return total_ms / iters;
}

static float run_d2d(void* dst, const void* src, size_t bytes, int iters) {
  mcEvent_t s, e;
  check(mcEventCreate(&s), "mcEventCreate start");
  check(mcEventCreate(&e), "mcEventCreate end");
  float total_ms = 0;
  for (int i = 0; i < iters; ++i) {
    check(mcEventRecord(s), "mcEventRecord start");
    check(mcMemcpy(dst, src, bytes, mcMemcpyDeviceToDevice), "mcMemcpy D2D");
    check(mcEventRecord(e), "mcEventRecord end");
    check(mcEventSynchronize(e), "mcEventSynchronize");
    float ms = 0;
    check(mcEventElapsedTime(&ms, s, e), "mcEventElapsedTime");
    total_ms += ms;
  }
  mcEventDestroy(s);
  mcEventDestroy(e);
  return total_ms / iters;
}

static Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* name) {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return std::string(argv[++i]);
    };
    if (a == "--bytes") o.bytes = std::stoull(need("--bytes"));
    else if (a == "--iters") o.iters = std::stoi(need("--iters"));
    else if (a == "--device") o.device = std::stoi(need("--device"));
    else if (a == "--char-dev") o.char_dev = need("--char-dev");
    else if (a == "--offset") o.char_dev_offset = std::stoull(need("--offset"));
    else throw std::runtime_error("unknown arg: " + a);
  }
  return o;
}

static void run_one(const char* tag, void* host_ptr, size_t bytes, int iters, void* d0, void* d1) {
  std::memset(host_ptr, 0x5A, bytes);
  float h2d_ms = run_h2d(d0, host_ptr, bytes, iters);
  float d2h_ms = run_d2h(host_ptr, d0, bytes, iters);
  float d2d_ms = run_d2d(d1, d0, bytes, iters);

  std::cout << "[" << tag << "]\n";
  std::cout << "  H2D: " << std::fixed << std::setprecision(2) << gbps(bytes, h2d_ms) << " GB/s\n";
  std::cout << "  D2H: " << std::fixed << std::setprecision(2) << gbps(bytes, d2h_ms) << " GB/s\n";
  std::cout << "  D2D: " << std::fixed << std::setprecision(2) << gbps(bytes, d2d_ms) << " GB/s\n";
}

int main(int argc, char** argv) {
  try {
    Options o = parse(argc, argv);
    check(mcInit(0), "mcInit");
    check(mcSetDevice(o.device), "mcSetDevice");

    void *d0 = nullptr, *d1 = nullptr;
    check(mcMalloc(&d0, o.bytes), "mcMalloc d0");
    check(mcMalloc(&d1, o.bytes), "mcMalloc d1");

    void* anon = map_anon(o.bytes);
    run_one("mmap-anonymous", anon, o.bytes, o.iters, d0, d1);

    if (!o.char_dev.empty()) {
      DevMap dm = map_chardev(o.char_dev, o.bytes, o.char_dev_offset);
      run_one("mmap-chardev", dm.ptr, o.bytes, o.iters, d0, d1);
      munmap(dm.ptr, dm.bytes);
      close(dm.fd);
    } else {
      std::cout << "[mmap-chardev] skipped (set --char-dev /dev/xxx)\n";
    }

    munmap(anon, o.bytes);
    mcFree(d0);
    mcFree(d1);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
