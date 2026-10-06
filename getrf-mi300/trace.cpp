// Isolated, warmed factorization for launch-count / GPU-busy-time attribution.
#include <cdls/getrf.h>
#include <cdls/internal/common.h>
#include <hip/hip_runtime.h>
#include <rocsolver/rocsolver.h>
#include <rocprofiler-sdk-roctx/roctx.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "failed line %d\n", __LINE__); std::abort(); } } while (0)
int main(int argc, char** argv) {
  REQUIRE(argc==3); const int n=std::atoi(argv[1]);
  const std::string implementation=argv[2];
  REQUIRE(implementation=="cdls" || implementation=="rocsolver" || implementation=="static");
  const bool ours=implementation!="rocsolver", pivot=implementation=="static";
  REQUIRE(roctxProfilerPause(0)==0);
  cdlsHandle_t handle; REQUIRE(cdlsCreate(&handle)==CDLS_STATUS_SUCCESS);
  auto stream=static_cast<hipStream_t>(handle->stream_work);
  rocblas_handle blas; REQUIRE(rocblas_create_handle(&blas)==rocblas_status_success);
  REQUIRE(rocblas_set_stream(blas,stream)==rocblas_status_success);
  size_t bytes; REQUIRE(cdlsDgetrf_bufferSize(n,n,&bytes)==CDLS_STATUS_SUCCESS);
  void* workspace; double *a,*backup; int* info;
  REQUIRE(hipMalloc(&workspace,bytes)==hipSuccess); REQUIRE(hipMalloc(&info,sizeof(int))==hipSuccess);
  const size_t cells=static_cast<size_t>(n)*n;
  REQUIRE(hipMalloc(&a,cells*sizeof(double))==hipSuccess); REQUIRE(hipMalloc(&backup,cells*sizeof(double))==hipSuccess);
  std::vector<double> host(cells);
  for(int col=0;col<n;++col) for(int row=0;row<n;++row)
    host[row+static_cast<size_t>(col)*n]=row==col?n+1.0:((row*37+col*13)%97-48)/100.0;
  REQUIRE(hipMemcpy(backup,host.data(),cells*sizeof(double),hipMemcpyHostToDevice)==hipSuccess);
  auto factor=[&] {
    if(pivot) {
      REQUIRE(hipMemsetAsync(info,0,sizeof(int),stream)==hipSuccess);
      REQUIRE(cdlsDgetrf_static_pivoting(handle,n,n,a,n,workspace,info,0.0)==CDLS_STATUS_SUCCESS);
    }
    else if(ours) REQUIRE(cdlsDgetrf(handle,n,n,a,n,workspace,info)==CDLS_STATUS_SUCCESS);
    else REQUIRE(rocsolver_dgetrf_npvt(blas,n,n,a,n,info)==rocblas_status_success);
  };
  for(int i=0;i<5;++i) {
    REQUIRE(hipMemcpyAsync(a,backup,cells*sizeof(double),hipMemcpyDeviceToDevice,stream)==hipSuccess);
    factor(); REQUIRE(hipStreamSynchronize(stream)==hipSuccess);
  }
  REQUIRE(hipMemcpyAsync(a,backup,cells*sizeof(double),hipMemcpyDeviceToDevice,stream)==hipSuccess);
  REQUIRE(hipStreamSynchronize(stream)==hipSuccess);
  REQUIRE(roctxProfilerResume(0)==0); roctxRangePushA("GETRF_FACTORIZATION");
  factor(); REQUIRE(hipStreamSynchronize(stream)==hipSuccess);
  roctxRangePop(); REQUIRE(roctxProfilerPause(0)==0);
  std::printf("profile n=%d implementation=%s\n",n,implementation.c_str());
  (void)hipFree(a);(void)hipFree(backup);(void)hipFree(workspace);(void)hipFree(info);
  (void)rocblas_destroy_handle(blas); REQUIRE(cdlsDestroy(handle)==CDLS_STATUS_SUCCESS);
}
