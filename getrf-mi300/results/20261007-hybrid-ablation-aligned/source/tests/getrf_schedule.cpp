#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../src/common/getrf/numeric.h"
#include "../src/common/getrf/schedule.h"

int main() {
  using cdls::detail::getrf::reciprocal_safe;
  if (!reciprocal_safe(0x1p-500) || !reciprocal_safe(-0x1p500) ||
      reciprocal_safe(0) || reciprocal_safe(0x1p-600) ||
      reciprocal_safe(0x1p600) || reciprocal_safe(INFINITY) ||
      reciprocal_safe(NAN))
    throw std::runtime_error("Reciprocal guards changed at a boundary");
  // Near the maximum signed-int task count, exercise products that would
  // overflow 32-bit intermediate arithmetic even though the result fits.
  constexpr int large = 46340;
  for (int task : {0, 1, large, large * large / 2, large * large - 1}) {
    const auto tile = cdls::detail::getrf::index_to_coord(task, large);
    const long long panel = std::min(tile.row, tile.col);
    const long long offset = tile.row > tile.col
                                 ? 2LL * (tile.row - tile.col) - 1
                                 : 2LL * (tile.col - tile.row);
    if (panel * (2LL * large - panel) + offset != task)
      throw std::runtime_error("Large task index mapping overflowed");
  }
  for (int n : {1, 2, 3, 7, 16, 65, 129, 512}) {
    std::vector<int> order(n * n, -1);
    for (int t = 0; t < n * n; ++t) {
      const auto p = cdls::detail::getrf::task_tile(t, n);
      if (p.row < 0 || p.row >= n || p.col < 0 || p.col >= n ||
          order[p.row + p.col * n] != -1)
        throw std::runtime_error("Task mapping is not a bijection");
      const int panel = std::min(p.row, p.col);
      for (int k = 0; k < panel; ++k)
        if (order[p.row + k * n] < 0 || order[k + p.col * n] < 0)
          throw std::runtime_error(
              "GEMM dependency appears after its consumer");
      if (p.row != p.col && order[panel + panel * n] < 0)
        throw std::runtime_error("TRSM diagonal appears after its consumer");
      order[p.row + p.col * n] = t;
    }
    for (int panels : {1, std::max(1, n / 2), n}) {
      const int tasks = panels * (2 * n - panels);
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
          if ((order[i + j * n] < tasks) != (std::min(i, j) < panels))
            throw std::runtime_error(
                "Prefix omits a panel tile or touches trailing A22");
    }
  }
  for (int n : {1, 2, 3, 4, 5, 7, 16, 17, 40, 65, 129, 512}) {
    std::vector<int> order(n * n, -1);
    for (int t = 0; t < n * n; ++t) {
      const auto p = cdls::detail::getrf::task_tile_banded(t, n);
      if (p.row < 0 || p.row >= n || p.col < 0 || p.col >= n ||
          order[p.row + p.col * n] != -1)
        throw std::runtime_error("Banded mapping is not a bijection");
      const int panel = std::min(p.row, p.col);
      for (int k = 0; k < panel; ++k)
        if (order[p.row + k * n] < 0 || order[k + p.col * n] < 0)
          throw std::runtime_error(
              "Banded GEMM prerequisite appears after its consumer");
      if (p.row != p.col && order[panel + panel * n] < 0)
        throw std::runtime_error(
            "Banded TRSM diagonal appears after its consumer");
      order[p.row + p.col * n] = t;
    }
    for (int panels = 4; panels <= n; panels += 4) {
      const int tasks = panels * (2 * n - panels);
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
          if ((order[i + j * n] < tasks) != (std::min(i, j) < panels))
            throw std::runtime_error(
                "Banded prefix touches A22 or omits an RHS");
    }
  }
  for (int n : {1, 2, 3, 7, 16, 65, 129, 512}) {
    for (int panels : {1, 2, 4, 8, 16, 32, std::max(1, n / 2), n}) {
      if (panels > n) continue;
      std::vector<int> order(n * n, -1);
      const int tasks = panels * (2 * n - panels);
      for (int t = 0; t < tasks; ++t) {
        const auto p =
            cdls::detail::getrf::task_tile_diagonal_first(t, n, panels);
        if (p.row < 0 || p.row >= n || p.col < 0 || p.col >= n ||
            std::min(p.row, p.col) >= panels || order[p.row + p.col * n] != -1)
          throw std::runtime_error(
              "Diagonal-first mapping is not a prefix bijection");
        const int kmax = std::min(p.row, p.col);
        for (int k = 0; k < kmax; ++k)
          if (order[p.row + k * n] < 0 || order[k + p.col * n] < 0)
            throw std::runtime_error(
                "Diagonal-first GEMM prerequisite is unlaunched");
        if (p.row != p.col && order[kmax + kmax * n] < 0)
          throw std::runtime_error(
              "Diagonal-first TRSM prerequisite is unlaunched");
        order[p.row + p.col * n] = t;
      }
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
          if ((order[i + j * n] >= 0) != (std::min(i, j) < panels))
            throw std::runtime_error(
                "Diagonal-first prefix touches A22 or omits an RHS");
    }
  }
  std::cout << "Task mapping: complete, unique, topological; prefix covers "
               "exactly LU/TRSM tiles\n";
}
