// chip_probe.cpp — three-axis micro-probe for Apple Silicon (M1 Max vs M5 Max).
//
// Isolates the three things that actually move generation-to-generation and
// that dominate low-latency / data-heavy code:
//   1. MEMORY LATENCY   - dependent pointer-chase across working-set sizes.
//                         This is your hash-table / quote-slot lookup cost.
//   2. MEMORY BANDWIDTH  - STREAM-style triad over buffers larger than cache.
//                         This is market-data replay / analytics scan speed.
//   3. COMPUTE THROUGHPUT- in-register integer mixing (splitmix64), 4 ILP lanes.
//                         This is per-message work / hashing, clock x IPC bound.
//
// No dependencies. Self-labels the machine (brand, P-cores, RAM) so you just
// run it on each laptop and diff the two outputs.
//
// BUILD (run on each Mac):
//     clang++ -std=c++17 -O3 -mcpu=native chip_probe.cpp -o chip_probe && ./chip_probe
// Optional working-set scale (default 1.0): ./chip_probe 0.25   (smaller buffers)
//
// Note: an online compiler (Compiler Explorer etc.) runs on a cloud x86 box,
// so it tells you nothing about your hardware. The whole point is to run the
// SAME binary natively on each laptop. The file is single-TU and dependency-
// free, so the one-liner above is all you need.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#if defined(__APPLE__)
  #include <sys/sysctl.h>
  #include <pthread/qos.h>
#endif

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Clock = std::chrono::steady_clock;

template <class T> static inline void sink(T const& v) {
    asm volatile("" : : "r,m"(v) : "memory");   // keep the compiler honest
}

static double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

static void* xaligned(std::size_t bytes, std::size_t align = 128) {
    void* p = nullptr;
    bytes = (bytes + align - 1) & ~(align - 1);
    if (posix_memalign(&p, align, bytes) != 0) { std::perror("alloc"); std::exit(1); }
    std::memset(p, 0, bytes);                    // fault the pages in up front
    return p;
}

// Bias work onto performance cores and reduce scheduler noise.
static void pin_to_fast_core() {
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

static void print_machine() {
    char brand[256] = "unknown CPU";
    long pcores = 0, mem = 0;
#if defined(__APPLE__)
    size_t n = sizeof(brand);
    sysctlbyname("machdep.cpu.brand_string", brand, &n, nullptr, 0);
    n = sizeof(pcores); sysctlbyname("hw.perflevel0.physicalcpu", &pcores, &n, nullptr, 0);
    n = sizeof(mem);    sysctlbyname("hw.memsize", &mem, &n, nullptr, 0);
#endif
    std::printf("machine : %s\n", brand);
    if (pcores) std::printf("P-cores : %ld\n", pcores);
    if (mem)    std::printf("RAM     : %.0f GB\n", mem / 1e9);
    std::printf("--------------------------------------------------\n");
}

// ---- 1. memory latency: single-cycle pointer chase -------------------------
static double chase_ns(std::size_t elems, u64 hops) {
    std::vector<u32> next(elems);
    for (std::size_t i = 0; i < elems; ++i) next[i] = static_cast<u32>(i);
    // Sattolo's algorithm -> one big cycle, so the chase can't be predicted.
    std::mt19937_64 rng(0xC0FFEE);
    for (std::size_t i = elems - 1; i > 0; --i) {
        std::size_t j = rng() % i;               // j in [0, i)
        std::swap(next[i], next[j]);
    }
    u32 idx = 0;
    for (u64 w = 0; w < elems; ++w) idx = next[idx];   // warm
    idx = 0;
    auto t0 = Clock::now();
    for (u64 h = 0; h < hops; ++h) idx = next[idx];    // each load depends on prior
    auto t1 = Clock::now();
    sink(idx);
    return secs(t0, t1) * 1e9 / double(hops);
}

static void latency_curve(double scale) {
    std::printf("[1] memory latency  (dependent load, ns/access)\n");
    struct { const char* label; std::size_t bytes; } pts[] = {
        {"16 KB  (L1)",      16ull << 10},
        {"128 KB (L2)",     128ull << 10},
        {"1 MB   (L2/SLC)",   1ull << 20},
        {"8 MB   (SLC)",      8ull << 20},
        {"64 MB  (DRAM)",    64ull << 20},
        {"256 MB (DRAM)",   256ull << 20},
    };
    for (auto& p : pts) {
        std::size_t bytes = std::size_t(p.bytes * scale);
        std::size_t elems = std::max<std::size_t>(bytes / sizeof(u32), 1024);
        u64 hops = elems < (1u << 20) ? 200'000'000ull : 40'000'000ull;
        double ns = chase_ns(elems, hops);
        std::printf("    %-16s %7.2f ns\n", p.label, ns);
    }
}

// ---- 2. memory bandwidth: STREAM triad -------------------------------------
static void bandwidth(double scale) {
    std::printf("[2] memory bandwidth  (triad a = b + s*c)\n");
    std::size_t n = std::size_t((256ull << 20) / sizeof(double) * scale);  // ~256MB/arr
    auto* a = static_cast<double*>(xaligned(n * sizeof(double)));
    auto* b = static_cast<double*>(xaligned(n * sizeof(double)));
    auto* c = static_cast<double*>(xaligned(n * sizeof(double)));
    for (std::size_t i = 0; i < n; ++i) { b[i] = 1.0; c[i] = 2.0; }
    const double s = 3.0;
    for (int w = 0; w < 2; ++w)                              // warm
        for (std::size_t i = 0; i < n; ++i) a[i] = b[i] + s * c[i];

    const int reps = 8;
    auto t0 = Clock::now();
    for (int r = 0; r < reps; ++r)
        for (std::size_t i = 0; i < n; ++i) a[i] = b[i] + s * c[i];
    auto t1 = Clock::now();
    sink(a[n - 1]);
    double bytes = double(reps) * n * 3 * sizeof(double);    // STREAM convention: 3 arrays
    std::printf("    %.1f GB/s\n", bytes / secs(t0, t1) / 1e9);
    std::free(a); std::free(b); std::free(c);
}

// ---- 3. compute throughput: in-register integer mixing ---------------------
static inline u64 splitmix(u64 x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
static void compute(double scale) {
    std::printf("[3] integer compute  (splitmix64, 4 ILP lanes)\n");
    const u64 iters = u64(800'000'000ull * scale);
    u64 a = 1, b = 2, c = 3, d = 4;                          // independent chains
    auto t0 = Clock::now();
    for (u64 i = 0; i < iters; ++i) {
        a = splitmix(a); b = splitmix(b); c = splitmix(c); d = splitmix(d);
    }
    auto t1 = Clock::now();
    sink(a); sink(b); sink(c); sink(d);
    double ops = double(iters) * 4;
    std::printf("    %.2f G mixes/s\n", ops / secs(t0, t1) / 1e9);
}

int main(int argc, char** argv) {
    double scale = (argc > 1) ? std::atof(argv[1]) : 1.0;
    pin_to_fast_core();
    print_machine();
    latency_curve(scale);
    bandwidth(scale);
    compute(scale);
    return 0;
}
