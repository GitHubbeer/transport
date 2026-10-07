// Real diagonal input replay using the existing CDLS runner's event protocol
// and reconstruction helpers. Build timing and profiling as separate binaries.
#include "validation.hpp"
#include <cstring>
#include <fstream>
#ifdef GETRF_PROFILE
#include <rocprofiler-sdk-roctx/roctx.h>
#endif

struct MappedInfo {
    int *host = nullptr, *device = nullptr;
    MappedInfo() {
        check(hipHostMalloc(reinterpret_cast<void**>(&host),sizeof(int),hipHostMallocMapped));
        check(hipHostGetDevicePointer(reinterpret_cast<void**>(&device),host,0));
    }
    ~MappedInfo() { if(host) (void)hipHostFree(host); }
};

int main(int argc,char **argv) try {
#ifdef GETRF_PROFILE
    if(argc!=5) throw std::runtime_error("profile INPUT N LDA cdls|rocsolver");
    if(roctxProfilerPause(0)) throw std::runtime_error("profiler pause failed");
    const std::string backend=argv[4];
    if(backend!="cdls" && backend!="rocsolver") throw std::runtime_error("invalid backend");
#else
    if(argc!=4) throw std::runtime_error("replay INPUT N LDA");
#endif
    const int n=std::stoi(argv[2]),lda=std::stoi(argv[3]);
    if(n<1 || lda<n) throw std::runtime_error("invalid shape");
    std::vector<double> host(size_t(n)*lda);
    std::ifstream input(argv[1],std::ios::binary);
    input.read(reinterpret_cast<char*>(host.data()),host.size()*sizeof(double));
    if(!input || input.peek()!=EOF) throw std::runtime_error("input byte extent mismatch");
    hipDeviceProp_t prop{};check(hipGetDeviceProperties(&prop,0));
    int runtime=0;check(hipRuntimeGetVersion(&runtime));
    char solver_version[256]{},blas_version[256]{};
    check(rocsolver_get_version_string(solver_version,sizeof(solver_version)));
    check(rocblas_get_version_string(blas_version,sizeof(blas_version)));
    std::cout<<"device="<<prop.name<<" arch="<<prop.gcnArchName<<" HIP="<<runtime
             <<" CUs="<<prop.multiProcessorCount<<" rocSOLVER="<<solver_version<<" rocBLAS="<<blas_version
             <<" n="<<n<<" lda="<<lda<<" input="<<argv[1]<<" stream_mode=nonblocking fp64 npvt warmup=5 repeats=30\n";
    Context ctx;
    const size_t bytes=host.size()*sizeof(double);
    Buffer<double> original(host.size()),ours(host.size()),ref(host.size());
    MappedInfo roc_info;
    size_t workspace_bytes=0;check(cdlsDgetrf_bufferSize(n,n,&workspace_bytes));
    Buffer<unsigned char> work(workspace_bytes);
    check(rocblas_start_device_memory_size_query(ctx.blas));
    auto query=rocsolver_dgetrf_npvt(ctx.blas,n,n,ref.p,lda,roc_info.device);
    if(query!=rocblas_status_success && query!=rocblas_status_size_increased && query!=rocblas_status_size_unchanged)check(query);
    size_t roc_bytes=0;check(rocblas_stop_device_memory_size_query(ctx.blas,&roc_bytes));
    Buffer<unsigned char> roc_work(roc_bytes);
    check(rocblas_set_workspace(ctx.blas,roc_work.p,roc_bytes));
    check(hipMemcpyAsync(original.p,host.data(),bytes,hipMemcpyHostToDevice,ctx.stream));
    check(hipStreamSynchronize(ctx.stream));
    auto restore=[&](bool cdls) {
        check(hipMemcpyAsync(cdls?ours.p:ref.p,original.p,bytes,hipMemcpyDeviceToDevice,ctx.stream));
        check(hipStreamSynchronize(ctx.stream));
    };
    auto factor=[&](bool cdls) {
        if(cdls)check(cdlsDgetrf(ctx.cdls,n,n,ours.p,lda,work.p,nullptr));
        else check(rocsolver_dgetrf_npvt(ctx.blas,n,n,ref.p,lda,roc_info.device));
    };
    auto numerical_info=[&](bool cdls) {
        int value=-1;
        if(cdls) {
            // Current source: internal_info = workspace + workspace_size(n),
            // buffer_bytes(n)=workspace_size(n)+sizeof(int). Ordinary public API
            // has no info output. This is an explicitly version-bound diagnostic.
            check(hipMemcpy(&value,work.p+workspace_bytes-sizeof(int),sizeof(int),hipMemcpyDeviceToHost));
        } else value=*roc_info.host;
        if(value!=0)throw std::runtime_error("nonzero numerical info "+std::to_string(value));
        return value;
    };
    for(int i=0;i<5;++i) {
#ifdef GETRF_PROFILE
        const bool is_cdls=backend=="cdls";restore(is_cdls);factor(is_cdls);check(hipStreamSynchronize(ctx.stream));numerical_info(is_cdls);
#else
        for(bool is_cdls:{true,false}) {restore(is_cdls);factor(is_cdls);check(hipStreamSynchronize(ctx.stream));numerical_info(is_cdls);}
#endif
    }
#ifdef GETRF_PROFILE
    for(int call=0;call<3;++call) {
        const bool is_cdls=backend=="cdls";restore(is_cdls);
        const std::string marker="GETRF|backend="+backend+"|call_id="+std::to_string(call)+"|n="+std::to_string(n)+"|lda="+std::to_string(lda);
        if(roctxProfilerResume(0))throw std::runtime_error("profiler resume failed");
        roctxRangePushA(marker.c_str());factor(is_cdls);check(hipStreamSynchronize(ctx.stream));roctxRangePop();
        if(roctxProfilerPause(0))throw std::runtime_error("profiler pause failed");
        numerical_info(is_cdls);
    }
    std::cout<<"profile_complete backend="<<backend<<" calls=3\n";
#else
    for(int iteration=0;iteration<30;++iteration) {
        double gpu[2]{},submit[2]{},wall[2]{};
        for(int step=0;step<2;++step) {
            const bool cdls=(iteration%2==0)?step==0:step==1;
            const int idx=cdls?0:1;restore(cdls);
            const auto start=std::chrono::steady_clock::now();
            check(hipEventRecord(ctx.begin,ctx.stream));
            const auto submit_start=std::chrono::steady_clock::now();factor(cdls);
            const auto submit_end=std::chrono::steady_clock::now();
            check(hipEventRecord(ctx.end,ctx.stream));check(hipEventSynchronize(ctx.end));
            const auto end=std::chrono::steady_clock::now();
            float ms=0;check(hipEventElapsedTime(&ms,ctx.begin,ctx.end));gpu[idx]=ms;
            submit[idx]=std::chrono::duration<double,std::milli>(submit_end-submit_start).count();
            wall[idx]=std::chrono::duration<double,std::milli>(end-start).count();
            numerical_info(cdls); // Outside all measured clocks.
        }
        std::cout<<std::fixed<<std::setprecision(9)<<"sample n="<<n<<" lda="<<lda<<" iteration="<<iteration
                 <<" first="<<(iteration%2?"rocsolver":"cdls")<<" cdls_ms="<<gpu[0]<<" rocsolver_ms="<<gpu[1]
                 <<" cdls_submit_ms="<<submit[0]<<" roc_submit_ms="<<submit[1]<<" cdls_wall_ms="<<wall[0]<<" roc_wall_ms="<<wall[1]<<'\n';
    }
    bool pass=true;
    for(bool cdls:{true,false}) {
        const auto a=cdls?ours.p:ref.p;
        std::vector<double> factors(host.size());check(hipMemcpy(factors.data(),a,bytes,hipMemcpyDeviceToHost));
        bool padding=true,finite=true;
        for(int j=0;j<n;++j)for(int i=0;i<lda;++i) {
            const size_t idx=size_t(j)*lda+i;
            if(i>=n)padding=padding && !std::memcmp(&factors[idx],&host[idx],sizeof(double));
            else finite=finite && std::isfinite(factors[idx]);
        }
        const double res=residual(ctx,n,lda,a,host);
        const int info=numerical_info(cdls);
        const bool ok=finite && padding && res<=100;
        pass=pass && ok;
        std::cout<<std::scientific<<std::setprecision(12)<<"validation backend="<<(cdls?"cdls":"rocsolver")
                 <<" n="<<n<<" lda="<<lda<<" info="<<info<<" info_source="<<(cdls?"internal_workspace_diagnostic":"mapped_pinned_public_info")
                 <<" padding_bitwise="<<padding<<" finite="<<finite<<" residual_scaled="<<res
                 <<" residual_relative="<<res*n*eps<<" status="<<(ok?"PASS":"FAIL")<<'\n';
    }
    return pass?0:1;
#endif
    return 0;
} catch(const std::exception& e) {std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
