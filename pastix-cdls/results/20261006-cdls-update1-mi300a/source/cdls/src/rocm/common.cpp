#include <cdls/common.h>
#include <hip/hip_runtime_api.h>

#include <new>

#include "handle.h"

extern "C" cdlsStatus_t cdlsCreate(cdlsHandle_t* handle) {
  if (handle == nullptr) return CDLS_STATUS_INVALID_VALUE;
  *handle = nullptr;
  auto* rocm_handle = new (std::nothrow) RocmHandle{};
  if (!rocm_handle) return CDLS_STATUS_ALLOC_FAILED;

  hipStream_t work = nullptr, memset_stream = nullptr;
  if (hipStreamCreateWithFlags(&work, hipStreamNonBlocking) != hipSuccess) {
    delete rocm_handle;
    return CDLS_STATUS_EXECUTION_FAILED;
  }
  if (hipStreamCreateWithFlags(&memset_stream, hipStreamNonBlocking) !=
      hipSuccess) {
    (void)hipStreamDestroy(work);
    delete rocm_handle;
    return CDLS_STATUS_EXECUTION_FAILED;
  }
  const auto status = rocblas_create_handle(&rocm_handle->blas);
  if (status != rocblas_status_success) {
    (void)hipStreamDestroy(work);
    (void)hipStreamDestroy(memset_stream);
    delete rocm_handle;
    return status == rocblas_status_memory_error ? CDLS_STATUS_ALLOC_FAILED
                                                 : CDLS_STATUS_NOT_INITIALIZED;
  }
  rocm_handle->common.stream_work = work;
  rocm_handle->common.stream_memset = memset_stream;
  *handle = &rocm_handle->common;
  return CDLS_STATUS_SUCCESS;
}

extern "C" cdlsStatus_t cdlsDestroy(cdlsHandle_t handle) {
  if (handle == nullptr || handle->stream_work == nullptr ||
      handle->stream_memset == nullptr)
    return CDLS_STATUS_INVALID_VALUE;

  auto* rocm_handle = reinterpret_cast<RocmHandle*>(handle);
  const auto blas_status = rocblas_destroy_handle(rocm_handle->blas);
  const auto work_status =
      hipStreamDestroy(static_cast<hipStream_t>(handle->stream_work));
  const auto memset_status =
      hipStreamDestroy(static_cast<hipStream_t>(handle->stream_memset));
  delete rocm_handle;
  return blas_status == rocblas_status_success && work_status == hipSuccess &&
                 memset_status == hipSuccess
             ? CDLS_STATUS_SUCCESS
             : CDLS_STATUS_EXECUTION_FAILED;
}
