#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#if defined(USE_MXGPU)
#include <mc_runtime.h>
typedef mcError_t GpuError;
typedef mcEvent_t GpuEvent;
#define gpuSuccess mcSuccess
#define gpuGetErrorString mcGetErrorString
#define gpuInit mcInit
#define gpuSetDevice mcSetDevice
#define gpuMalloc mcMalloc
#define gpuFree mcFree
#define gpuMemcpy mcMemcpy
#define gpuMemcpyHostToDevice mcMemcpyHostToDevice
#define gpuMemcpyDeviceToHost mcMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice mcMemcpyDeviceToDevice
#define gpuEventCreate mcEventCreate
#define gpuEventDestroy mcEventDestroy
#define gpuEventRecord mcEventRecord
#define gpuEventSynchronize mcEventSynchronize
#define gpuEventElapsedTime mcEventElapsedTime
#define gpuHostRegister mcHostRegister
#define gpuHostUnregister mcHostUnregister
#define gpuHostRegisterDefault mcHostRegisterDefault
#define gpuHostRegisterIoMemory mcHostRegisterIoMemory
#else
#include <cuda_runtime.h>
typedef cudaError_t GpuError;
typedef cudaEvent_t GpuEvent;
#define gpuSuccess cudaSuccess
#define gpuGetErrorString cudaGetErrorString
#define gpuInit(flags) cudaSuccess
#define gpuSetDevice cudaSetDevice
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuEventCreate cudaEventCreate
#define gpuEventDestroy cudaEventDestroy
#define gpuEventRecord(ev) cudaEventRecord(ev)
#define gpuEventSynchronize cudaEventSynchronize
#define gpuEventElapsedTime cudaEventElapsedTime
#define gpuHostRegister cudaHostRegister
#define gpuHostUnregister cudaHostUnregister
#define gpuHostRegisterDefault cudaHostRegisterDefault
#define gpuHostRegisterIoMemory cudaHostRegisterIoMemory
#endif

struct Options { size_t bytes = 256ull * 1024 * 1024; int iters = 30; int device = 0; std::string char_dev; size_t char_dev_offset = 0; };

static void check(GpuError err, const char* what) { if (err != gpuSuccess) throw std::runtime_error(std::string(what) + " failed: " + gpuGetErrorString(err)); }
static double gbps(size_t bytes, float ms) { return static_cast<double>(bytes) / (ms / 1000.0) / 1e9; }
static void* map_anon(size_t bytes) { void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0); if (p == MAP_FAILED) throw std::runtime_error("mmap anon failed"); return p; }
struct DevMap { int fd = -1; void* ptr = nullptr; size_t bytes = 0; };
static DevMap map_chardev(const std::string& path, size_t bytes, size_t off) { DevMap m; m.fd = open(path.c_str(), O_RDWR | O_SYNC); if (m.fd < 0) throw std::runtime_error("open char dev failed: " + path); m.ptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, m.fd, off); if (m.ptr == MAP_FAILED) { close(m.fd); throw std::runtime_error("mmap char dev failed"); } m.bytes = bytes; return m; }

static float run_copy(void* a, const void* b, size_t bytes, int iters, int kind) {
  GpuEvent s, e; check(gpuEventCreate(&s), "event create s"); check(gpuEventCreate(&e), "event create e");
  float total_ms = 0;
  for (int i = 0; i < iters; ++i) {
    check(gpuEventRecord(s), "event record s");
    check(gpuMemcpy(a, b, bytes, kind), "memcpy");
    check(gpuEventRecord(e), "event record e");
    check(gpuEventSynchronize(e), "event sync e");
    float ms = 0; check(gpuEventElapsedTime(&ms, s, e), "event elapsed"); total_ms += ms;
  }
  gpuEventDestroy(s); gpuEventDestroy(e); return total_ms / iters;
}

static Options parse(int argc, char** argv){ Options o; for(int i=1;i<argc;++i){ std::string a=argv[i]; auto need=[&](const char* n){ if(i+1>=argc) throw std::runtime_error(std::string("missing value for ")+n); return std::string(argv[++i]);}; if(a=="--bytes") o.bytes=std::stoull(need("--bytes")); else if(a=="--iters") o.iters=std::stoi(need("--iters")); else if(a=="--device") o.device=std::stoi(need("--device")); else if(a=="--char-dev") o.char_dev=need("--char-dev"); else if(a=="--offset") o.char_dev_offset=std::stoull(need("--offset")); else throw std::runtime_error("unknown arg: "+a);} return o; }

static void run_one(const char* tag, void* h, size_t bytes, int iters, void* d0, void* d1, unsigned int reg_flag){
  check(gpuHostRegister(h, bytes, reg_flag), "host register");
  std::memset(h, 0x5A, bytes);
  float h2d = run_copy(d0, h, bytes, iters, gpuMemcpyHostToDevice);
  float d2h = run_copy(h, d0, bytes, iters, gpuMemcpyDeviceToHost);
  float d2d = run_copy(d1, d0, bytes, iters, gpuMemcpyDeviceToDevice);
  std::cout << "["<<tag<<"]\n";
  std::cout << "  H2D: " << std::fixed << std::setprecision(2) << gbps(bytes, h2d) << " GB/s\n";
  std::cout << "  D2H: " << std::fixed << std::setprecision(2) << gbps(bytes, d2h) << " GB/s\n";
  std::cout << "  D2D: " << std::fixed << std::setprecision(2) << gbps(bytes, d2d) << " GB/s\n";
  check(gpuHostUnregister(h), "host unregister");
}

int main(int argc, char** argv){
  try {
    Options o=parse(argc,argv);
    check(gpuInit(0), "gpuInit");
    check(gpuSetDevice(o.device), "gpuSetDevice");
    void *d0=nullptr,*d1=nullptr; check(gpuMalloc(&d0,o.bytes), "gpuMalloc d0"); check(gpuMalloc(&d1,o.bytes), "gpuMalloc d1");
    void* anon=map_anon(o.bytes);
    run_one("mmap-anonymous", anon, o.bytes, o.iters, d0, d1, gpuHostRegisterIoMemory);
    if(!o.char_dev.empty()){ DevMap dm=map_chardev(o.char_dev,o.bytes,o.char_dev_offset); run_one("mmap-chardev", dm.ptr, o.bytes, o.iters, d0, d1, gpuHostRegisterDefault); munmap(dm.ptr, dm.bytes); close(dm.fd);} else std::cout << "[mmap-chardev] skipped (set --char-dev /dev/xxx)\n";
    munmap(anon, o.bytes); gpuFree(d0); gpuFree(d1); return 0;
  } catch(const std::exception& e){ std::cerr << "error: "<<e.what()<<"\n"; return 1; }
}
