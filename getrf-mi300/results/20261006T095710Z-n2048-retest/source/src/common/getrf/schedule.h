#pragma once
#include "portability.h"

namespace cdls::detail::getrf {
template <class Index>
struct TileCoord {
  Index row, col;
};
using Tile = TileCoord<int>;

// Diagonal, then alternating lower/upper tiles of that panel. Every dependency
// has a smaller task number. Use 64-bit products even though task IDs are int.
template <class Index>
CDLS_GETRF_HD TileCoord<Index> index_to_coord(Index task, Index tiles) {
  Index lo = 0, hi = tiles;
  while (lo + 1 < hi) {
    const Index mid = lo + (hi - lo) / 2;
    if (static_cast<long long>(mid) * (2LL * tiles - mid) <= task)
      lo = mid;
    else
      hi = mid;
  }
  const Index offset = task - static_cast<long long>(lo) * (2LL * tiles - lo);
  return (offset & 1) ? TileCoord<Index>{lo + (offset + 1) / 2, lo}
                      : TileCoord<Index>{lo, lo + offset / 2};
}

// Complete the leading panels x panels square first, in expanding square
// rings. Each ring's two strips precede its diagonal. The remaining rows and
// columns are independent once that square is factored. This keeps the next
// diagonal from sitting behind all distant RHS tiles in a wide matrix.
CDLS_GETRF_HD Tile task_tile_diagonal_first(int task, int tiles, int panels) {
  (void)tiles;  // The caller launches exactly panels * (2 * tiles - panels)
                // tasks.
  const long long square = static_cast<long long>(panels) * panels;
  if (task < square) {
    int lo = 0, hi = panels;
    while (lo + 1 < hi) {
      const int mid = lo + (hi - lo) / 2;
      if (static_cast<long long>(mid) * mid <= task)
        lo = mid;
      else
        hi = mid;
    }
    const int offset = task - static_cast<long long>(lo) * lo;
    if (offset == 2 * lo) return Tile{lo, lo};
    return (offset & 1) ? Tile{offset / 2, lo} : Tile{lo, offset / 2};
  }
  const int offset = task - square;
  const int line = panels + offset / (2 * panels);
  const int panel = (offset % (2 * panels)) / 2;
  return (offset & 1) ? Tile{panel, line} : Tile{line, panel};
}

// CDLS CUDA's bounded diagonal lookahead: finish a four-panel square before
// its distant RHS strips, then advance to the next group. Group boundaries
// coincide with the original panel-border prefix counts.
CDLS_GETRF_HD Tile task_tile_banded(int task, int tiles) {
  constexpr int Band = 4;
  const auto original = index_to_coord(task, tiles);
  const int first =
      (original.row < original.col ? original.row : original.col) / Band * Band;
  const int width = tiles - first < Band ? tiles - first : Band;
  const int offset =
      task - static_cast<long long>(first) * (2LL * tiles - first);
  if (offset < width * width) {
    const auto local = task_tile_diagonal_first(offset, width, width);
    return {first + local.row, first + local.col};
  }
  const int rest = offset - width * width;
  const int line = first + width + rest / (2 * width);
  const int panel = first + (rest % (2 * width)) / 2;
  return (rest & 1) ? Tile{panel, line} : Tile{line, panel};
}
CDLS_GETRF_HD Tile task_tile(int task, int tiles) {
  return index_to_coord(task, tiles);
}
}  // namespace cdls::detail::getrf
