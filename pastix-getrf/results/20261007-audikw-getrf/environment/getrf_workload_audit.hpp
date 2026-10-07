#ifndef PASTIX_GETRF_WORKLOAD_AUDIT_HPP
#define PASTIX_GETRF_WORKLOAD_AUDIT_HPP
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace pastix_device {
// Opt-in submission metadata only. Capture and timing live in separate runners.
class GetrfWorkloadAudit {
    FILE *file_ = nullptr;
    unsigned long long call_ = 0;
public:
    GetrfWorkloadAudit() {
        const char *path = std::getenv("PASTIX_GETRF_WORKLOAD_CSV");
        if (path && *path) {
            file_ = std::fopen(path, "wx");
            if (!file_) throw std::runtime_error("Cannot create GETRF workload CSV");
            std::fprintf(file_, "call_id,cblk,n,lda,backend,queue\n");
        }
    }
    ~GetrfWorkloadAudit() { if (file_) std::fclose(file_); }
    void record(long long cblk, long long n, long long lda, const char *backend, unsigned queue) {
        if (!file_) return;
        std::fprintf(file_, "%llu,%lld,%lld,%lld,%s,%u\n", call_++, cblk, n, lda, backend, queue);
        std::fflush(file_);
    }
};
}
#endif
