// chip_probe_mt.cpp — multicore scaling probe for Apple Silicon.
//
// Answers: with a 3-tier core design (super / performance / efficiency), how
// does aggregate throughput actually scale, and where does memory bandwidth
// saturate? This is the number that matters for sharded matching engines or a
// many-agent market simulation, where you care about total big-core throughput
// rather than one hot thread.
//
// BUILD (each Mac):
//   clang++ -std=c++17 -O3 -mcpu=native chip_probe_mt.cpp -o chip_probe_mt && ./chip_probe_mt
//   optional buffer scale (default 1.0):  ./chip_probe_mt 0.5

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#if defined(__APPLE__)
  #include <sys/sysctl.h>
  #include <pthread/qos.h>
#endif

using u64 = std::uint64_t;
using Clock = std::chrono::steady_clock;
template <class T> static inline void sink(T const& v){ asm volatile("" : : "r,m"(v) : "memory"); }
static double secs(Clock::time_point a, Clock::time_point b){ return std::chrono::duration<double>(b-a).count(); }

static void* xaligned(std::size_t bytes, std::size_t align=128){
    void* p=nullptr; bytes=(bytes+align-1)&~(align-1);
    if(posix_memalign(&p,align,bytes)!=0){ std::perror("alloc"); std::exit(1);} std::memset(p,0,bytes); return p;
}
static void prefer_fast(){
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

static int topology(){
    int total = (int)std::thread::hardware_concurrency();
#if defined(__APPLE__)
    long nlevels=0; size_t n=sizeof(nlevels);
    sysctlbyname("hw.nperflevels", &nlevels, &n, nullptr, 0);
    char brand[256]="unknown"; n=sizeof(brand);
    sysctlbyname("machdep.cpu.brand_string", brand, &n, nullptr, 0);
    std::printf("machine : %s\n", brand);
    for(long i=0;i<nlevels;++i){
        char nm[64], key[64]; long phys=0, logi=0;
        std::snprintf(key,sizeof key,"hw.perflevel%ld.name",i); n=sizeof(nm);
        if(sysctlbyname(key,nm,&n,nullptr,0)!=0) std::strcpy(nm,"tier");
        std::snprintf(key,sizeof key,"hw.perflevel%ld.physicalcpu",i); n=sizeof(phys);
        sysctlbyname(key,&phys,&n,nullptr,0);
        std::snprintf(key,sizeof key,"hw.perflevel%ld.logicalcpu",i); n=sizeof(logi);
        sysctlbyname(key,&logi,&n,nullptr,0);
        std::printf("tier %ld  : %-12s %ld physical / %ld logical\n", i, nm, phys, logi);
    }
    long lc=0; n=sizeof(lc); if(sysctlbyname("hw.logicalcpu",&lc,&n,nullptr,0)==0 && lc) total=(int)lc;
#endif
    std::printf("total   : %d logical\n", total);
    std::printf("--------------------------------------------------\n");
    return total;
}

static inline u64 splitmix(u64 x){
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x>>30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x>>27)) * 0x94D049BB133111EBull;
    return x ^ (x>>31);
}

// Aggregate integer throughput across `t` threads (each 4 ILP lanes).
static double compute_scale(int t, u64 iters){
    std::atomic<bool> go{false};
    std::atomic<int> ready{0};
    std::vector<std::thread> th; th.reserve(t);
    std::vector<u64> out(t,0);
    for(int k=0;k<t;++k) th.emplace_back([&,k]{
        prefer_fast();
        u64 a=k+1,b=k+2,c=k+3,d=k+4;
        ready.fetch_add(1);
        while(!go.load(std::memory_order_acquire)) {}
        for(u64 i=0;i<iters;++i){ a=splitmix(a); b=splitmix(b); c=splitmix(c); d=splitmix(d);}
        out[k]=a^b^c^d;
    });
    while(ready.load()<t) {}
    auto t0=Clock::now(); go.store(true,std::memory_order_release);
    for(auto& x:th) x.join();
    auto t1=Clock::now();
    u64 s=0; for(auto v:out) s^=v; sink(s);
    return double(iters)*4*t / secs(t0,t1) / 1e9;   // aggregate G mixes/s
}

// Aggregate triad bandwidth across `t` threads over a shared large buffer.
static double bw_scale(int t, std::size_t n, double* a, double* b, double* c){
    std::atomic<bool> go{false}; std::atomic<int> ready{0};
    std::vector<std::thread> th; th.reserve(t);
    const std::size_t chunk=(n+t-1)/t; const double s=3.0; const int reps=6;
    for(int k=0;k<t;++k) th.emplace_back([&,k]{
        prefer_fast();
        std::size_t lo=std::min(n,(std::size_t)k*chunk), hi=std::min(n,lo+chunk);
        ready.fetch_add(1);
        while(!go.load(std::memory_order_acquire)) {}
        for(int r=0;r<reps;++r) for(std::size_t i=lo;i<hi;++i) a[i]=b[i]+s*c[i];
    });
    while(ready.load()<t) {}
    auto t0=Clock::now(); go.store(true,std::memory_order_release);
    for(auto& x:th) x.join();
    auto t1=Clock::now();
    sink(a[n-1]);
    return double(reps)*n*3*sizeof(double) / secs(t0,t1) / 1e9;  // aggregate GB/s
}

int main(int argc, char** argv){
    double scale = (argc>1)? std::atof(argv[1]) : 1.0;
    prefer_fast();
    int total = topology();

    std::vector<int> sweep;
    for(int c: {1,2,4,6,8,12,16,18,24,32}) if(c<=total) sweep.push_back(c);
    if(sweep.empty() || sweep.back()!=total) sweep.push_back(total);

    std::printf("[compute scaling]  splitmix64 aggregate\n");
    const u64 base_iters = (u64)(150'000'000ull * scale);
    double one=0;
    for(int t: sweep){
        double g = compute_scale(t, base_iters);
        if(t==1) one=g;
        std::printf("    %2d thr : %7.2f G/s   (%.2fx, %3.0f%% eff)\n",
            t, g, g/one, 100.0*g/(one*t));
    }

    std::printf("[bandwidth scaling]  triad aggregate\n");
    std::size_t n = std::size_t((512ull<<20)/sizeof(double)*scale);   // ~512MB/arr
    auto* a=(double*)xaligned(n*sizeof(double));
    auto* b=(double*)xaligned(n*sizeof(double));
    auto* c=(double*)xaligned(n*sizeof(double));
    for(std::size_t i=0;i<n;++i){ b[i]=1.0; c[i]=2.0; }
    double peak=0;
    for(int t: sweep){ double gb=bw_scale(t,n,a,b,c); if(gb>peak)peak=gb;
        std::printf("    %2d thr : %7.1f GB/s\n", t, gb); }
    std::printf("peak    : %.1f GB/s\n", peak);
    std::free(a); std::free(b); std::free(c);
    return 0;
}
