#pragma once

#include <cdls/common.h>

#include <algorithm>
#include <cstdint>

namespace cdls::detail::getrf {

// Prefix factorization produces L21/U12 while leaving A22 untouched.
// Backends provide launch/update callbacks and measured panel/tail policies.
template <class Index, class Width, class Factor, class Update>
cdlsStatus_t factorize_hybrid(Index n, double* matrix, Index lda, int tail,
                              int tile, Width width_for, Factor factor,
                              Update update) {
  Index offset = 0;
  while (n - offset > tail) {
    const Index remaining = n - offset;
    const Index width = width_for(remaining);
    auto* base = matrix + offset + offset * lda;
    auto status = factor(remaining, base, static_cast<int>(width / tile));
    if (status != CDLS_STATUS_SUCCESS) return status;
    status = update(base, static_cast<int>(remaining - width),
                    static_cast<int>(width));
    if (status != CDLS_STATUS_SUCCESS) return status;
    offset += width;
  }
  return factor(n - offset, matrix + offset + offset * lda, 0);
}

}  // namespace cdls::detail::getrf
