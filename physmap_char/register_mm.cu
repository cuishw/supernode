#include <fcntl.h>
#include <mc_runtime.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <chrono>

namespace {

struct Options {
  int device = 0;
  size_t bytes = 256ull * 1024 * 1024;
  std::string mem_type = "system";  // system, mmap
  std::string mmap_path;
  size_t mmap_offset = 0;
  bool mmio = false;
  bool touch = true;
};

struct Mapping {
  void* ptr = nullptr;
  size_t bytes = 0;
  int fd = -1;
  bool is_mmap = false;
};

volatile sig_atomic_t g_stop = 0;

void on_signal(int) { g_stop = 1; }

std::string errno_msg(const std::string& what) {
  return what + ": " + std::strerror(errno);
}

void check(mcError_t err, const char* what) {
  if (err != mcSuccess) {
    throw std::runtime_error(std::string(what) + " failed: " + mcGetErrorString(err));
  }
}

std::string need_value(int& i, int argc, char** argv, const char* name) {
  if (i + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + name);
  }
  return argv[++i];
}

void usage(const char* prog) {
  std::cout
      << "Usage: " << prog << " [options]\n\n"
      << "Options:\n"
      << "  --device N           Muxi GPU device id (default: 0)\n"
      << "  --bytes N            memory size to register (default: 268435456)\n"
      << "  --mem system|mmap    allocate with malloc/aligned_alloc or mmap (default: system)\n"
      << "  --mmap-path PATH     mmap this file/device with MAP_SHARED; omit for anonymous mmap\n"
      << "  --offset N           mmap offset for --mmap-path (default: 0)\n"
      << "  --mmio               register mmap memory with mcHostRegisterIoMemory\n"
      << "  --no-touch           do not memset the registered memory after mapping\n"
      << "  --help               show this help\n\n"
      << "The program registers memory and then stays alive until SIGINT/SIGTERM.\n"
      << "Examples:\n"
      << "  " << prog << " --device 0 --bytes 1073741824 --mem system\n"
      << "  " << prog << " --device 0 --bytes 268435456 --mem mmap\n"
      << "  " << prog << " --device 0 --bytes 4096 --mem mmap --mmap-path /dev/xxx --mmio\n";
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--device") {
      o.device = std::stoi(need_value(i, argc, argv, "--device"));
    } else if (a == "--bytes") {
      o.bytes = std::stoull(need_value(i, argc, argv, "--bytes"));
    } else if (a == "--mem") {
      o.mem_type = need_value(i, argc, argv, "--mem");
    } else if (a == "--mmap-path") {
      o.mmap_path = need_value(i, argc, argv, "--mmap-path");
    } else if (a == "--offset") {
      o.mmap_offset = std::stoull(need_value(i, argc, argv, "--offset"));
    } else if (a == "--mmio") {
      o.mmio = true;
    } else if (a == "--no-touch") {
      o.touch = false;
    } else if (a == "--help" || a == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown arg: " + a);
    }
  }

  if (o.bytes == 0) {
    throw std::runtime_error("--bytes must be greater than 0");
  }
  if (o.mem_type != "system" && o.mem_type != "mmap") {
    throw std::runtime_error("--mem must be system or mmap");
  }
  if (o.mmio && o.mem_type != "mmap") {
    throw std::runtime_error("--mmio is only valid with --mem mmap");
  }
  return o;
}

Mapping allocate_memory(const Options& o) {
  Mapping m;
  m.bytes = o.bytes;

  if (o.mem_type == "system") {
    const size_t alignment = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    if (posix_memalign(&m.ptr, alignment, o.bytes) != 0) {
      throw std::runtime_error("posix_memalign failed");
    }
    return m;
  }

  m.is_mmap = true;
  int flags = MAP_SHARED;
  if (o.mmap_path.empty()) {
    flags |= MAP_ANONYMOUS;
    m.fd = -1;
  } else {
    m.fd = open(o.mmap_path.c_str(), O_RDWR | O_SYNC);
    if (m.fd < 0) {
      throw std::runtime_error(errno_msg("open " + o.mmap_path));
    }
  }

  m.ptr = mmap(nullptr, o.bytes, PROT_READ | PROT_WRITE, flags, m.fd, o.mmap_offset);
  if (m.ptr == MAP_FAILED) {
    if (m.fd >= 0) {
      close(m.fd);
    }
    throw std::runtime_error(errno_msg("mmap"));
  }
  return m;
}

void free_memory(Mapping& m) {
  if (!m.ptr) {
    return;
  }
  if (m.is_mmap) {
    munmap(m.ptr, m.bytes);
    if (m.fd >= 0) {
      close(m.fd);
    }
  } else {
    free(m.ptr);
  }
  m.ptr = nullptr;
}

}  // namespace

int main(int argc, char** argv) {
  Mapping mapping;
  bool registered = false;

  try {
    Options o = parse(argc, argv);
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    check(mcInit(0), "mcInit");
    check(mcSetDevice(o.device), "mcSetDevice");

    mapping = allocate_memory(o);
    if (o.touch) {
      std::memset(mapping.ptr, 0, mapping.bytes);
    }

    unsigned int flags = o.mmio ? mcHostRegisterIoMemory : mcHostRegisterDefault;
    check(mcHostRegister(mapping.ptr, mapping.bytes, flags), "mcHostRegister");
    registered = true;

    std::cout << "registered " << mapping.bytes << " bytes at " << mapping.ptr
              << " on Muxi GPU device " << o.device
              << " using " << (o.mmio ? "mcHostRegisterIoMemory" : "mcHostRegisterDefault")
              << " (mem=" << o.mem_type;
    if (!o.mmap_path.empty()) {
      std::cout << ", path=" << o.mmap_path << ", offset=" << o.mmap_offset;
    }
    std::cout << ")\nPress Ctrl+C or send SIGTERM to unregister and exit.\n";

    while (!g_stop) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    check(mcHostUnregister(mapping.ptr), "mcHostUnregister");
    registered = false;
    free_memory(mapping);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    if (registered) {
      mcHostUnregister(mapping.ptr);
    }
    free_memory(mapping);
    return 1;
  }
}
