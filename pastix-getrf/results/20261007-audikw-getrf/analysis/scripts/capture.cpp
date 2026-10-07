// LD_PRELOAD companion for the opt-in core metadata audit. It interposes the
// exact current ordinary CDLS entry, preserving stream order and full lda*n.
#include <cdls/getrf.h>
#include <cdls/internal/common.h>
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <vector>
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr,"capture failed line %d\n",__LINE__); std::abort(); } } while (0)
extern "C" cdlsStatus_t cdlsDgetrf(cdlsHandle_t h,long m,long n,double *a,long lda,void *work,int *counter) {
    using Fn = decltype(&cdlsDgetrf);
    static Fn real = [] {
        auto p = dlopen(std::getenv("GETRF_REAL_CDLS"),RTLD_NOW|RTLD_LOCAL);
        REQUIRE(p); auto f = reinterpret_cast<Fn>(dlsym(p,"cdlsDgetrf")); REQUIRE(f); return f;
    }();
    static unsigned long long call = 0;
    static FILE *log = [] {
        auto f=std::fopen(std::getenv("GETRF_STREAM_CSV"),"wx"); REQUIRE(f);
        std::fprintf(f,"call_id,n,lda,backend,stream\n");return f;
    }();
    static std::set<std::pair<long,long>> selected = [] {
        std::set<std::pair<long,long>> pairs;
        const char *path=std::getenv("GETRF_CAPTURE_SELECTION");
        if(path) { std::ifstream f(path);REQUIRE(f);long n,ld;while(f>>n>>ld)pairs.emplace(n,ld); }
        return pairs;
    }();
    const auto id=call++;
    const auto stream=static_cast<hipStream_t>(h->stream_work);
    std::fprintf(log,"%llu,%ld,%ld,cdls,%p\n",id,n,lda,static_cast<void*>(stream));std::fflush(log);
    auto found=selected.find({n,lda});
    if(found!=selected.end()) {
        std::vector<double> host(size_t(n)*lda);
        REQUIRE(hipMemcpyAsync(host.data(),a,host.size()*sizeof(double),hipMemcpyDeviceToHost,stream)==hipSuccess);
        REQUIRE(hipStreamSynchronize(stream)==hipSuccess);
        std::string prefix=std::string(std::getenv("GETRF_CAPTURE_DIR"))+"/call"+std::to_string(id)+"-n"+std::to_string(n)+"-lda"+std::to_string(lda);
        auto f=std::fopen((prefix+".bin").c_str(),"wx");REQUIRE(f);
        REQUIRE(std::fwrite(host.data(),sizeof(double),host.size(),f)==host.size());REQUIRE(!std::fclose(f));
        std::ofstream meta(prefix+".json");REQUIRE(meta);
        meta<<"{\"call_id\":"<<id<<",\"n\":"<<n<<",\"lda\":"<<lda<<",\"bytes\":"<<host.size()*sizeof(double)<<",\"dtype\":\"float64\",\"layout\":\"column-major\",\"source\":\"after predecessor Schur updates and diagonal GEADD, before cdlsDgetrf\",\"padding\":\"all original rows n..lda-1 preserved\"}\n";
        selected.erase(found);
    }
    return real(h,m,n,a,lda,work,counter);
}
