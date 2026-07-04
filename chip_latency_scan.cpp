// chip_scan.cpp v2 — Apple Silicon intelligence scan, NEON hot kernels.
//
// Changes vs v1:
//   * [2] read bandwidth rewritten: 8 independent NEON uint64 accumulators so
//     it saturates load ports / memory instead of bottlenecking on add latency.
//     Adds an all-core aggregate read + triad ceiling.
//   * [3] core-to-core LATENCY DISTRIBUTION: Apple Silicon gives no per-core
//     affinity (thread_affinity_policy is ignored), so we sample the scheduler's
//     placement N times and histogram it — the bands are the topology. Where an
//     efficiency tier exists we force a cross-tier pair via QoS (INTERACTIVE vs
//     BACKGROUND) for a labelled fast<->slow number.
//   * [6] FP rewritten in NEON (16x float32x4 FMA accumulators) for true peak.
//
// BUILD (each Mac):
//   clang++ -std=c++17 -O3 -mcpu=native chip_scan.cpp -o chip_scan && ./chip_scan
//   optional scale (default 1.0):  ./chip_scan 0.5

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#if defined(__APPLE__)
  #include <sys/sysctl.h>
  #include <pthread/qos.h>
  #include <mach/mach_time.h>
#endif
#if defined(__aarch64__)
  #include <arm_neon.h>
  #define HAVE_NEON 1
#else
  #define HAVE_NEON 0
#endif

using u32=std::uint32_t; using u64=std::uint64_t;
using Clock=std::chrono::steady_clock;
template<class T> static inline void sink(T const& v){ asm volatile("" : : "r,m"(v) : "memory"); }
static double secs(Clock::time_point a,Clock::time_point b){ return std::chrono::duration<double>(b-a).count(); }

static void* xaligned(std::size_t bytes,std::size_t align=128){
    void* p=nullptr; bytes=(bytes+align-1)&~(align-1);
    if(posix_memalign(&p,align,bytes)!=0){ std::perror("alloc"); std::exit(1);} std::memset(p,0,bytes); return p;
}

enum QTier { FAST, SLOW };
static void set_qos(QTier t){
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(t==FAST?QOS_CLASS_USER_INTERACTIVE:QOS_CLASS_BACKGROUND, 0);
#else
    (void)t;
#endif
}
static void prefer_fast(){ set_qos(FAST); }

struct Tier { char name[32]; long phys, logi; };
static std::vector<Tier> g_tiers;
static bool has_efficiency(){ for(auto&t:g_tiers) if(std::strstr(t.name,"Effic")) return true; return false; }
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
        std::snprintf(key,sizeof key,"hw.perflevel%ld.physicalcpu",i); n=sizeof(v); sysctlbyname(key,&v,&n,nullptr,0); t.phys=v;
        std::snprintf(key,sizeof key,"hw.perflevel%ld.logicalcpu",i); n=sizeof(v); sysctlbyname(key,&v,&n,nullptr,0); t.logi=v;
        g_tiers.push_back(t); std::printf("tier %ld  : %-12s %ld physical / %ld logical\n",i,t.name,t.phys,t.logi);
    }
    long lc=0; n=sizeof(lc); if(sysctlbyname("hw.logicalcpu",&lc,&n,nullptr,0)==0&&lc) total=(int)lc;
#endif
    std::printf("total   : %d logical\n==================================================\n",total);
    return total;
}

template<class F> static double parallel(int t,F&& body){
    std::atomic<bool> go{false}; std::atomic<int> ready{0};
    std::vector<std::thread> th; th.reserve(t);
    for(int k=0;k<t;++k) th.emplace_back([&,k]{ prefer_fast(); ready.fetch_add(1);
        while(!go.load(std::memory_order_acquire)){} body(k); });
    while(ready.load()<t){}
    auto t0=Clock::now(); go.store(true,std::memory_order_release);
    for(auto& x:th) x.join();
    auto t1=Clock::now(); return secs(t0,t1);
}

// ---- NEON kernels ----------------------------------------------------------
static u64 read_sum(const double* p,std::size_t n){
    const u64* q=reinterpret_cast<const u64*>(p);
#if HAVE_NEON
    uint64x2_t a0=vdupq_n_u64(0),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
    std::size_t i=0,lim=n&~std::size_t(15);
    for(;i<lim;i+=16){
        a0=vaddq_u64(a0,vld1q_u64(q+i));    a1=vaddq_u64(a1,vld1q_u64(q+i+2));
        a2=vaddq_u64(a2,vld1q_u64(q+i+4));  a3=vaddq_u64(a3,vld1q_u64(q+i+6));
        a4=vaddq_u64(a4,vld1q_u64(q+i+8));  a5=vaddq_u64(a5,vld1q_u64(q+i+10));
        a6=vaddq_u64(a6,vld1q_u64(q+i+12)); a7=vaddq_u64(a7,vld1q_u64(q+i+14));
    }
    uint64x2_t s=vaddq_u64(vaddq_u64(vaddq_u64(a0,a1),vaddq_u64(a2,a3)),vaddq_u64(vaddq_u64(a4,a5),vaddq_u64(a6,a7)));
    u64 sum=vgetq_lane_u64(s,0)+vgetq_lane_u64(s,1);
    for(;i<n;++i) sum+=q[i]; return sum;
#else
    u64 s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0; std::size_t i=0,lim=n&~std::size_t(7);
    for(;i<lim;i+=8){ s0+=q[i];s1+=q[i+1];s2+=q[i+2];s3+=q[i+3];s4+=q[i+4];s5+=q[i+5];s6+=q[i+6];s7+=q[i+7]; }
    u64 s=s0+s1+s2+s3+s4+s5+s6+s7; for(;i<n;++i) s+=q[i]; return s;
#endif
}
static double fp_flops(u64 iters){
#if HAVE_NEON
    float32x4_t acc[16]; for(int k=0;k<16;++k) acc[k]=vdupq_n_f32(1.0f+0.01f*k);
    const float32x4_t x=vdupq_n_f32(1.0000001f), y=vdupq_n_f32(0.9999999f);
    for(u64 i=0;i<iters;++i){
        acc[0]=vfmaq_f32(y,acc[0],x);  acc[1]=vfmaq_f32(y,acc[1],x);
        acc[2]=vfmaq_f32(y,acc[2],x);  acc[3]=vfmaq_f32(y,acc[3],x);
        acc[4]=vfmaq_f32(y,acc[4],x);  acc[5]=vfmaq_f32(y,acc[5],x);
        acc[6]=vfmaq_f32(y,acc[6],x);  acc[7]=vfmaq_f32(y,acc[7],x);
        acc[8]=vfmaq_f32(y,acc[8],x);  acc[9]=vfmaq_f32(y,acc[9],x);
        acc[10]=vfmaq_f32(y,acc[10],x);acc[11]=vfmaq_f32(y,acc[11],x);
        acc[12]=vfmaq_f32(y,acc[12],x);acc[13]=vfmaq_f32(y,acc[13],x);
        acc[14]=vfmaq_f32(y,acc[14],x);acc[15]=vfmaq_f32(y,acc[15],x);
    }
    float s=0; for(int k=0;k<16;++k) s+=vaddvq_f32(acc[k]); sink(s);
    return double(iters)*16*4*2;
#else
    float acc[16]; for(int k=0;k<16;++k) acc[k]=1.0f+0.01f*k;
    const float x=1.0000001f,y=0.9999999f;
    for(u64 i=0;i<iters;++i) for(int k=0;k<16;++k) acc[k]=acc[k]*x+y;
    float s=0; for(int k=0;k<16;++k) s+=acc[k]; sink(s);
    return double(iters)*16*2;
#endif
}

// ---- 1. latency curve ------------------------------------------------------
static double chase_ns(std::size_t e,u64 hops){
    std::vector<u32> nx(e); for(std::size_t i=0;i<e;++i) nx[i]=(u32)i;
    std::mt19937_64 rng(0xC0FFEE); for(std::size_t i=e-1;i>0;--i){ std::size_t j=rng()%i; std::swap(nx[i],nx[j]); }
    u32 idx=0; for(std::size_t w=0;w<e;++w) idx=nx[idx]; idx=0;
    auto t0=Clock::now(); for(u64 h=0;h<hops;++h) idx=nx[idx]; auto t1=Clock::now();
    sink(idx); return secs(t0,t1)*1e9/double(hops);
}
static void latency_curve(double s){
    std::printf("[1] memory latency (ns/access)\n");
    struct{const char* l;std::size_t b;} pts[]={{"16 KB",16ull<<10},{"128 KB",128ull<<10},
        {"1 MB",1ull<<20},{"8 MB",8ull<<20},{"64 MB",64ull<<20},{"256 MB",256ull<<20}};
    for(auto&p:pts){ std::size_t e=std::max<std::size_t>(std::size_t(p.b*s)/4,1024);
        u64 h=e<(1u<<20)?150000000ull:30000000ull; std::printf("    %-8s %7.2f\n",p.l,chase_ns(e,h)); }
}

// ---- 2. bandwidth (single + aggregate) -------------------------------------
static void bandwidth(int total,double s){
    std::printf("[2] bandwidth (GB/s, counted)\n");
    std::size_t n=std::size_t((256ull<<20)/8*s);
    auto* a=(double*)xaligned(n*8); auto* b=(double*)xaligned(n*8); auto* c=(double*)xaligned(n*8);
    for(std::size_t i=0;i<n;++i){ b[i]=1.0; c[i]=2.0; }
    auto t1=[&](auto f){ f(); f(); auto s0=Clock::now(); int R=8; for(int r=0;r<R;++r) f(); auto s1=Clock::now(); return secs(s0,s1)/R; };
    double rd=n*8.0/t1([&]{ sink(read_sum(b,n)); })/1e9;
    double cp=n*2*8.0/t1([&]{ for(std::size_t i=0;i<n;++i) a[i]=b[i]; sink(a[n-1]); })/1e9;
    double tr=n*3*8.0/t1([&]{ for(std::size_t i=0;i<n;++i) a[i]=b[i]+3.0*c[i]; sink(a[n-1]); })/1e9;
    std::printf("    read(1t)   %7.1f\n    copy(1t)   %7.1f\n    triad(1t)  %7.1f\n",rd,cp,tr);
    // aggregate read + triad ceiling
    int R=6;
    double ar=double(R)*n*8.0/parallel(total,[&](int k){ std::size_t chunk=(n+total-1)/total,lo=std::min(n,(std::size_t)k*chunk),hi=std::min(n,lo+chunk);
        u64 acc=0; for(int r=0;r<R;++r) acc+=read_sum(b+lo,hi-lo); sink(acc); })/1e9;
    double at=double(R)*n*3*8.0/parallel(total,[&](int k){ std::size_t chunk=(n+total-1)/total,lo=std::min(n,(std::size_t)k*chunk),hi=std::min(n,lo+chunk);
        for(int r=0;r<R;++r){ for(std::size_t i=lo;i<hi;++i) a[i]=b[i]+3.0*c[i]; } sink(a[n?n-1:0]); })/1e9;
    std::printf("    read(%dt)  %7.1f   [D] true read ceiling\n    triad(%dt) %7.1f   [D] memory bandwidth ceiling\n",total,ar,total,at);
    std::free(a); std::free(b); std::free(c);
}

// ---- 3. core-to-core latency distribution ----------------------------------
struct alignas(128) PingLine { std::atomic<u64> seq{0}; char pad[128-sizeof(std::atomic<u64>)]; };
static PingLine g_line;
static double c2c_once(u64 iters,QTier qa,QTier qb){
    g_line.seq.store(0);
    std::atomic<bool> go{false}; std::atomic<int> ready{0};
    std::thread A([&]{ set_qos(qa); ready.fetch_add(1); while(!go.load()){}
        for(u64 i=0;i<iters;++i){ g_line.seq.store(2*i+1,std::memory_order_release);
            while(g_line.seq.load(std::memory_order_acquire)<2*i+2){} } });
    std::thread B([&]{ set_qos(qb); ready.fetch_add(1); while(!go.load()){}
        for(u64 i=0;i<iters;++i){ while(g_line.seq.load(std::memory_order_acquire)<2*i+1){}
            g_line.seq.store(2*i+2,std::memory_order_release); } });
    while(ready.load()<2){}
    auto t0=Clock::now(); go.store(true); A.join(); B.join(); auto t1=Clock::now();
    return secs(t0,t1)*1e9/double(iters)/2.0;
}
static void hist(std::vector<double>& v,const char* tag){
    std::sort(v.begin(),v.end());
    auto pct=[&](double p){ return v[std::min(v.size()-1,(std::size_t)(p*v.size())) ]; };
    double lo=v.front(),hi=v.back();
    std::printf("    %-11s min %5.1f  p50 %5.1f  p95 %5.1f  max %5.1f ns\n",tag,lo,pct(0.5),pct(0.95),hi);
    int B=10; if(hi<=lo) hi=lo+1;
    int cnt[10]={0}; for(double x:v){ int bi=(int)((x-lo)/(hi-lo)*B); if(bi<0)bi=0; if(bi>=B)bi=B-1; cnt[bi]++; }
    int mx=1; for(int i=0;i<B;++i) mx=std::max(mx,cnt[i]);
    for(int i=0;i<B;++i){ double c0=lo+(hi-lo)*i/B; int bars=cnt[i]*24/mx;
        std::printf("      %5.1f |%.*s %d\n",c0,bars,"########################",cnt[i]); }
}
static void c2c_matrix(double sc,int total){
    std::printf("[3] core-to-core latency distribution (one-way ns)\n");
    if(total<2){ std::printf("    skipped (needs >=2 cores)\n"); return; }
    u64 iters=std::max<u64>((u64)(30000*sc),3000); int trials=200;
    std::vector<double> same; same.reserve(trials);
    for(int t=0;t<trials;++t) same.push_back(c2c_once(iters,FAST,FAST));
    hist(same,"same-tier");
    if(has_efficiency()){
        std::vector<double> cross; cross.reserve(trials);
        for(int t=0;t<trials;++t) cross.push_back(c2c_once(iters,FAST,SLOW));
        hist(cross,"cross-tier");
    }
}

// ---- 4. atomics ------------------------------------------------------------
static void atomics(int total){
    std::printf("[4] atomics fetch_add (M ops/s)\n");
    static std::atomic<u64> shared{0}; const u64 iters=50000000; shared.store(0);
    double t1=parallel(1,[&](int){ for(u64 i=0;i<iters;++i) shared.fetch_add(1,std::memory_order_relaxed); });
    double unc=iters/t1/1e6; std::printf("    uncontended %7.1f\n",unc);
    if(total<2){ std::printf("    contended: skipped\n"); return; }
    int t=std::min(total,8); shared.store(0);
    double tc=parallel(t,[&](int){ for(u64 i=0;i<iters;++i) shared.fetch_add(1,std::memory_order_relaxed); });
    double con=(double)iters*t/tc/1e6;
    std::printf("    contended%2d %7.1f\n    [D] collapse %6.1fx\n",t,con,unc/con);
}

// ---- 5. compute scaling + per-tier -----------------------------------------
static inline u64 splitmix(u64 x){ x+=0x9E3779B97F4A7C15ull; x=(x^(x>>30))*0xBF58476D1CE4E5B9ull;
    x=(x^(x>>27))*0x94D049BB133111EBull; return x^(x>>31); }
static void compute_scaling(int total,double s){
    std::printf("[5] compute scaling (G mixes/s)\n");
    const u64 it=(u64)(120000000ull*s);
    std::vector<int> sw; long n0=g_tiers.empty()?0:g_tiers[0].phys;
    for(int c:{1,2,4,6,8,12,16,18,24,32}) if(c<=total) sw.push_back(c);
    if(n0&&std::find(sw.begin(),sw.end(),(int)n0)==sw.end()) sw.push_back((int)n0);
    if(std::find(sw.begin(),sw.end(),total)==sw.end()) sw.push_back(total);
    std::sort(sw.begin(),sw.end());
    std::vector<std::pair<int,double>> res; double one=0;
    for(int t:sw){ std::vector<u64> out(t);
        double sec=parallel(t,[&](int k){ u64 a=k+1,b=k+2,c=k+3,d=k+4;
            for(u64 i=0;i<it;++i){a=splitmix(a);b=splitmix(b);c=splitmix(c);d=splitmix(d);} out[k]=a^b^c^d; });
        u64 x=0; for(auto v:out) x^=v; sink(x);
        double g=double(it)*4*t/sec/1e9; if(t==1)one=g; res.push_back({t,g});
        std::printf("    %2d thr %7.2f  (%.0f%% eff)\n",t,g,100.0*g/(one*t)); }
    auto at=[&](int t){ for(auto&p:res) if(p.first==t) return p.second; return 0.0; };
    if(!g_tiers.empty()){ long a=g_tiers[0].phys; double gA=at((int)a);
        if(gA>0) std::printf("    [D] per-%s core %6.3f G/s\n",g_tiers[0].name,gA/a);
        if(g_tiers.size()>=2){ long b=g_tiers[1].phys; double gAB=at((int)(a+b));
            if(gAB>0&&b>0) std::printf("    [D] per-%s core %6.3f G/s\n",g_tiers[1].name,(gAB-gA)/b); } }
}

// ---- 6. NEON FP ------------------------------------------------------------
static void fp_throughput(int total,double s){
    std::printf("[6] NEON FP throughput (GFLOP/s)\n");
    const u64 it=(u64)(60000000ull*s);
    auto s0=Clock::now(); double f=fp_flops(it); auto s1=Clock::now();
    double g1=f/secs(s0,s1)/1e9;
    double tt=parallel(total,[&](int){ fp_flops(it); });
    double flops_per=f; double gN=flops_per*total/tt/1e9;
    std::printf("    1 thr    %7.1f\n    %2d thr   %7.1f\n",g1,total,gN);
}

// ---- 7. timestamp ----------------------------------------------------------
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
    prefer_fast(); int total=topology();
    latency_curve(s); bandwidth(total,s); c2c_matrix(s,total);
    atomics(total); compute_scaling(total,s); fp_throughput(total,s); timestamp_cost();
    return 0;
}
