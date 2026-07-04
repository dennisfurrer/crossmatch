// chip_scan.cpp — Apple Silicon "intelligence scan" for low-latency systems work.
//
// Every metric maps to a real workload; derived metrics are labelled [D].
// Timing uses mach_absolute_time on Apple (steady_clock elsewhere).
//
// NOTE on core pinning: thread_affinity_policy is advisory-only on Apple
// Silicon (the kernel ignores it). We instead bias to the fastest tier via QoS
// and DERIVE per-tier per-core throughput from the scaling curve: with the fast
// hint the scheduler fills tier-0 cores first, so agg(n0)/n0 is per-tier0-core
// and (agg(n0+n1)-agg(n0))/n1 is per-tier1-core.
//
// BUILD (each Mac):
//   clang++ -std=c++17 -O3 -mcpu=native chip_scan.cpp -o chip_scan && ./chip_scan
//   optional scale (default 1.0):  ./chip_scan 0.5

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>
#include <algorithm>

#if defined(__APPLE__)
  #include <sys/sysctl.h>
  #include <pthread/qos.h>
  #include <mach/mach_time.h>
#endif
#if defined(__aarch64__)
  #include <arm_neon.h>
#endif

using u32 = std::uint32_t; using u64 = std::uint64_t;
using Clock = std::chrono::steady_clock;

template <class T> static inline void sink(T const& v){ asm volatile("" : : "r,m"(v) : "memory"); }
static double secs(Clock::time_point a, Clock::time_point b){ return std::chrono::duration<double>(b-a).count(); }

#if defined(__clang__)
  #define NT_STORE(p,v) __builtin_nontemporal_store((v),(p))
#else
  #define NT_STORE(p,v) (*(p)=(v))
#endif

static void prefer_fast(){
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}
static void* xaligned(std::size_t bytes, std::size_t align=128){
    void* p=nullptr; bytes=(bytes+align-1)&~(align-1);
    if(posix_memalign(&p,align,bytes)!=0){ std::perror("alloc"); std::exit(1);} std::memset(p,0,bytes); return p;
}

struct Tier { char name[32]; long phys, logi; };
static std::vector<Tier> g_tiers;
static int topology(){
    int total=(int)std::thread::hardware_concurrency();
#if defined(__APPLE__)
    char brand[256]="unknown"; size_t n=sizeof(brand);
    sysctlbyname("machdep.cpu.brand_string",brand,&n,nullptr,0);
    std::printf("machine : %s\n",brand);
    long nl=0; n=sizeof(nl); sysctlbyname("hw.nperflevels",&nl,&n,nullptr,0);
    for(long i=0;i<nl;++i){ Tier t{}; char key[64]; long v=0;
        std::snprintf(key,sizeof key,"hw.perflevel%ld.name",i); n=sizeof(t.name);
        if(sysctlbyname(key,t.name,&n,nullptr,0)!=0) std::strcpy(t.name,"tier");
        std::snprintf(key,sizeof key,"hw.perflevel%ld.physicalcpu",i); n=sizeof(v);
        sysctlbyname(key,&v,&n,nullptr,0); t.phys=v;
        std::snprintf(key,sizeof key,"hw.perflevel%ld.logicalcpu",i); n=sizeof(v);
        sysctlbyname(key,&v,&n,nullptr,0); t.logi=v;
        g_tiers.push_back(t);
        std::printf("tier %ld  : %-12s %ld physical / %ld logical\n",i,t.name,t.phys,t.logi);
    }
    long lc=0; n=sizeof(lc); if(sysctlbyname("hw.logicalcpu",&lc,&n,nullptr,0)==0 && lc) total=(int)lc;
#endif
    std::printf("total   : %d logical\n",total);
    std::printf("==================================================\n");
    return total;
}

// spin-start parallel harness: returns wall seconds
template <class F>
static double parallel(int t, F&& body){
    std::atomic<bool> go{false}; std::atomic<int> ready{0};
    std::vector<std::thread> th; th.reserve(t);
    for(int k=0;k<t;++k) th.emplace_back([&,k]{
        prefer_fast(); ready.fetch_add(1);
        while(!go.load(std::memory_order_acquire)){}
        body(k);
    });
    while(ready.load()<t){}
    auto t0=Clock::now(); go.store(true,std::memory_order_release);
    for(auto& x:th) x.join();
    auto t1=Clock::now();
    return secs(t0,t1);
}

// ---- 1. latency curve ------------------------------------------------------
static double chase_ns(std::size_t elems,u64 hops){
    std::vector<u32> nx(elems); for(std::size_t i=0;i<elems;++i) nx[i]=(u32)i;
    std::mt19937_64 rng(0xC0FFEE);
    for(std::size_t i=elems-1;i>0;--i){ std::size_t j=rng()%i; std::swap(nx[i],nx[j]); }
    u32 idx=0; for(std::size_t w=0;w<elems;++w) idx=nx[idx]; idx=0;
    auto t0=Clock::now(); for(u64 h=0;h<hops;++h) idx=nx[idx]; auto t1=Clock::now();
    sink(idx); return secs(t0,t1)*1e9/double(hops);
}
static void latency_curve(double s){
    std::printf("[1] memory latency (ns/access)\n");
    struct{const char* l; std::size_t b;} pts[]={{"16 KB",16ull<<10},{"128 KB",128ull<<10},
        {"1 MB",1ull<<20},{"8 MB",8ull<<20},{"64 MB",64ull<<20},{"256 MB",256ull<<20}};
    for(auto&p:pts){ std::size_t e=std::max<std::size_t>(std::size_t(p.b*s)/4,1024);
        u64 h=e<(1u<<20)?150000000ull:30000000ull;
        std::printf("    %-8s %7.2f\n",p.l,chase_ns(e,h)); }
}

// ---- 2. bandwidth variants (single thread) ---------------------------------
static void bandwidth_variants(double s){
    std::printf("[2] bandwidth single-thread (GB/s, counted)\n");
    std::size_t n=std::size_t((256ull<<20)/8*s);
    auto* a=(double*)xaligned(n*8); auto* b=(double*)xaligned(n*8); auto* c=(double*)xaligned(n*8);
    for(std::size_t i=0;i<n;++i){ b[i]=1.0; c[i]=2.0; }
    auto time=[&](auto f){ f(); f(); auto t0=Clock::now(); int R=6; for(int r=0;r<R;++r) f(); auto t1=Clock::now(); return secs(t0,t1)/R; };
    double rd = n*8.0 / time([&]{ double s0=0,s1=0; for(std::size_t i=0;i<n;i+=2){s0+=b[i];s1+=b[i+1];} sink(s0+s1); }) /1e9;
    double cp = n*2*8.0 / time([&]{ for(std::size_t i=0;i<n;++i) a[i]=b[i]; sink(a[n-1]); }) /1e9;
    double tr = n*3*8.0 / time([&]{ for(std::size_t i=0;i<n;++i) a[i]=b[i]+3.0*c[i]; sink(a[n-1]); }) /1e9;
    double nt = n*3*8.0 / time([&]{ for(std::size_t i=0;i<n;++i){ double v=b[i]+3.0*c[i]; NT_STORE(&a[i],v);} sink(a[n-1]); }) /1e9;
    // random gather throughput (independent, MLP) vs sequential read
    std::size_t m=n; std::vector<u32> idx(m); std::mt19937_64 rng(1);
    for(std::size_t i=0;i<m;++i) idx[i]=(u32)(rng()%m);
    double rg = m*8.0 / time([&]{ double s0=0; for(std::size_t i=0;i<m;++i) s0+=b[idx[i]]; sink(s0); }) /1e9;
    std::printf("    read-only  %7.1f\n    copy       %7.1f\n    triad(RFO) %7.1f\n    triad(NT)  %7.1f\n    rand-gather%7.1f\n",rd,cp,tr,nt,rg);
    std::printf("    [D] RFO overhead    %5.1f%%\n",100.0*(nt-tr)/tr);
    std::printf("    [D] seq/random      %5.1fx\n",rd/rg);
    std::free(a); std::free(b); std::free(c);
}

// ---- 3. core-to-core latency (min one-way ns) ------------------------------
static void c2c_latency(double sc,int total){
    std::printf("[3] core-to-core latency (one-way ns)\n");
    if(total<2){ std::printf("    skipped (needs >=2 cores)\n"); return; }
    struct Line { alignas(128) std::atomic<u64> v{0}; };
    static Line line;
    const u64 iters=std::max<u64>((u64)(200000*sc),2000); double best=1e18;
    for(int trial=0;trial<3;++trial){
        line.v.store(0);
        std::atomic<bool> go{false}; std::atomic<int> ready{0};
        std::thread A([&]{ prefer_fast(); ready.fetch_add(1); while(!go.load()){}
            for(u64 i=0;i<iters;++i){ line.v.store(2*i+1,std::memory_order_release);
                while(line.v.load(std::memory_order_acquire)<2*i+2){} } });
        std::thread B([&]{ prefer_fast(); ready.fetch_add(1); while(!go.load()){}
            for(u64 i=0;i<iters;++i){ while(line.v.load(std::memory_order_acquire)<2*i+1){}
                line.v.store(2*i+2,std::memory_order_release); } });
        while(ready.load()<2){}
        auto t0=Clock::now(); go.store(true); A.join(); B.join(); auto t1=Clock::now();
        double oneway=secs(t0,t1)*1e9/double(iters)/2.0;
        if(oneway<best) best=oneway;
    }
    std::printf("    min one-way %7.1f\n",best);
}

// ---- 4. atomics: uncontended vs contended ----------------------------------
static void atomics(int total){
    std::printf("[4] atomics fetch_add (M ops/s)\n");
    static std::atomic<u64> shared{0};
    const u64 iters=std::max<u64>((u64)(50000000*1),1000000);
    shared.store(0);
    double t1=parallel(1,[&](int){ for(u64 i=0;i<iters;++i) shared.fetch_add(1,std::memory_order_relaxed); });
    double unc=iters/t1/1e6;
    std::printf("    uncontended %7.1f\n",unc);
    if(total<2){ std::printf("    contended: skipped (needs >=2 cores)\n"); return; }
    int t=std::min(total,8); shared.store(0);
    double tc=parallel(t,[&](int){ for(u64 i=0;i<iters;++i) shared.fetch_add(1,std::memory_order_relaxed); });
    double con=(double)iters*t/tc/1e6;
    std::printf("    contended%2d %7.1f\n",t,con);
    std::printf("    [D] collapse %6.1fx\n",unc/con);
}

// ---- 5. compute scaling + per-tier derivation ------------------------------
static inline u64 splitmix(u64 x){ x+=0x9E3779B97F4A7C15ull; x=(x^(x>>30))*0xBF58476D1CE4E5B9ull;
    x=(x^(x>>27))*0x94D049BB133111EBull; return x^(x>>31); }
static void compute_scaling(int total,double s){
    std::printf("[5] compute scaling (G mixes/s)\n");
    const u64 it=(u64)(120000000ull*s);
    std::vector<int> sweep; long n0=g_tiers.empty()?0:g_tiers[0].phys;
    for(int c:{1,2,4,6,8,12,16,18,24,32}) if(c<=total) sweep.push_back(c);
    if(n0 && std::find(sweep.begin(),sweep.end(),(int)n0)==sweep.end()) sweep.push_back((int)n0);
    if(std::find(sweep.begin(),sweep.end(),total)==sweep.end()) sweep.push_back(total);
    std::sort(sweep.begin(),sweep.end());
    std::vector<std::pair<int,double>> res;
    double one=0;
    for(int t:sweep){ std::vector<u64> out(t);
        double sec=parallel(t,[&](int k){ u64 a=k+1,b=k+2,c=k+3,d=k+4;
            for(u64 i=0;i<it;++i){a=splitmix(a);b=splitmix(b);c=splitmix(c);d=splitmix(d);} out[k]=a^b^c^d; });
        u64 x=0; for(auto v:out)x^=v; sink(x);
        double g=double(it)*4*t/sec/1e9; if(t==1)one=g;
        res.push_back({t,g});
        std::printf("    %2d thr %7.2f  (%.0f%% eff)\n",t,g,100.0*g/(one*t));
    }
    auto at=[&](int t){ for(auto&p:res) if(p.first==t) return p.second; return 0.0; };
    if(g_tiers.size()>=1){ long a=g_tiers[0].phys; double gA=at((int)a);
        if(gA>0) std::printf("    [D] per-%s core %6.3f G/s\n",g_tiers[0].name,gA/a);
        if(g_tiers.size()>=2){ long b=g_tiers[1].phys; double gAB=at((int)(a+b));
            if(gAB>0 && b>0) std::printf("    [D] per-%s core %6.3f G/s\n",g_tiers[1].name,(gAB-gA)/b); } }
}

// ---- 6. NEON FP throughput -------------------------------------------------
static double fp_kernel(u64 iters){
    // 16 independent FMA chains; portable, auto-vectorized at -O3.
    float acc[16]; for(int k=0;k<16;++k) acc[k]=1.0f+0.01f*k;
    const float x=1.0000001f, y=0.9999999f;
    for(u64 i=0;i<iters;++i) for(int k=0;k<16;++k) acc[k]=acc[k]*x+y;
    float s=0; for(int k=0;k<16;++k) s+=acc[k]; sink(s);
    return double(iters)*16*2; // flops
}
static void fp_throughput(int total,double s){
    std::printf("[6] NEON FP throughput (GFLOP/s)\n");
    const u64 it=(u64)(80000000ull*s);
    auto s0=Clock::now(); double f=fp_kernel(it); auto s1=Clock::now();
    double g1=f/secs(s0,s1)/1e9;
    double tt=parallel(total,[&](int){ fp_kernel(it); });
    double gN=double(it)*16*2*total/tt/1e9;
    std::printf("    1 thr    %7.1f\n    %2d thr   %7.1f\n",g1,total,gN);
}

// ---- 7. timestamp cost -----------------------------------------------------
static void timestamp_cost(){
    std::printf("[7] timestamp cost (ns/call)\n");
    const u64 iters=20000000; u64 acc=0;
#if defined(__APPLE__)
    auto t0=Clock::now(); for(u64 i=0;i<iters;++i) acc+=mach_absolute_time(); auto t1=Clock::now();
    sink(acc); std::printf("    mach_absolute_time %6.2f\n",secs(t0,t1)*1e9/iters);
#else
    auto t0=Clock::now(); for(u64 i=0;i<iters;++i) acc+=Clock::now().time_since_epoch().count(); auto t1=Clock::now();
    sink(acc); std::printf("    steady_clock::now  %6.2f\n",secs(t0,t1)*1e9/iters);
#endif
}

int main(int argc,char** argv){
    double s=(argc>1)?std::atof(argv[1]):1.0;
    prefer_fast();
    int total=topology();
    latency_curve(s);
    bandwidth_variants(s);
    c2c_latency(s,total);
    atomics(total);
    compute_scaling(total,s);
    fp_throughput(total,s);
    timestamp_cost();
    return 0;
}
