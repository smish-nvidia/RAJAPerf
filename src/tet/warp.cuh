#pragma once

// The warp-level helpers of the direct method's kernels (INTSC_HEXHEX-CudaProjection.cpp,
// INTSC_HEXRECT-CudaProjection.cpp): a copy of the helper section of tet_intersection's
// src/gpu_common.cuh.

#include "tet/hex.hpp"

namespace tet::gpu {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Helpers of the direct method's kernels (intersect_hex_hex.cu, intersect_hex_rect.cu). They share a
// pattern: a warp sets up the triangles of its hexes in shared memory, box-tests every (T, S) item
// (a triangle of A against a triangle of B, or a cell) and queues those that pass, then takes the
// queue 32 items at a time, a lane per item, clipping and integrating it (projection.hpp).

// Per-lane scratch space for the direct method's polygon, in shared memory: in local memory, it
// made its kernels wait on L2 (on an A100, 7.5 -> 3.4 ms for 44k hex pairs, 14.8 -> 6.0 ms for 837k
// hex-cell pairs). Shared memory is what limits these kernels' occupancy, hence one warp per block,
// polygons that carry only (y, z), and clipping them in place. An odd number of doubles per lane
// keeps the lanes' accesses free of bank conflicts.
using ProjScratch = detail::ProjPoly;
static_assert(sizeof(ProjScratch) % 16 == 8, "ProjScratch must span an odd number of doubles");

// b as floats rounded outward (lo down, hi up), for box tests: they then let through every pair
// that the doubles would, and the comparisons stay off the FP64 pipe (on an A100 at 1200 MHz,
// INTSC_HEXHEX subzones 1.51 -> 1.41 ms at a jitter of 0.3).
inline __device__ void outward_floats(const Box& b, float* f) {
  f[0] = __double2float_rd(b.lo.x), f[1] = __double2float_rd(b.lo.y), f[2] = __double2float_rd(b.lo.z);
  f[3] = __double2float_ru(b.hi.x), f[4] = __double2float_ru(b.hi.y), f[5] = __double2float_ru(b.hi.z);
}

// Adds each lane's m to acc[key] (nothing where key < 0), the keys in nondecreasing order over the
// lanes (any lanes with key -1 last): an inclusive scan within each key, whose last lane then holds
// the key's sum. The result does not depend on timing, as it would with atomics.
inline __device__ void add_by_key(int key, const Moments& m, double (*acc)[4], int lane) {
  double v[4] = {m.volume, m.first.x, m.first.y, m.first.z};
  for (int d = 1; d < 32; d *= 2) {
    int same = __shfl_up_sync(~0u, key, d) == key && lane >= d;
    for (int i = 0; i < 4; ++i) {
      double u = __shfl_up_sync(~0u, v[i], d);
      if (same) v[i] += u;
    }
  }
  int next = __shfl_down_sync(~0u, key, 1);
  if (key >= 0 && (lane == 31 || next != key))
    for (int i = 0; i < 4; ++i) acc[key][i] += v[i];
  __syncwarp();
}

// A triangle's 3 corners as 9 doubles, and back.
inline __device__ void store_triangle(const Vec3* x, double* t) {
  for (int k = 0; k < 3; ++k) t[3 * k] = x[k].x, t[3 * k + 1] = x[k].y, t[3 * k + 2] = x[k].z;
}
inline __device__ void load_triangle(const double* t, Vec3* x) {
  for (int k = 0; k < 3; ++k) x[k] = {t[3 * k], t[3 * k + 1], t[3 * k + 2]};
}

// Makes the box of a triangle parallel to x (no projected area, so no terms) fail every box test
// below, as T (is_s false) or as S: such a triangle would only hold a lane that has nothing to do
// while the others clip (on an A100 at 1200 MHz, 4.6 -> 3.4 ms for RAJAPerf's INTSC_HEXRECT, where
// 61% of the items were such triangles).
inline __device__ void never_queued(Box& b, bool is_s) {
  if (is_s)
    b.lo.x = INFINITY;
  else
    b.hi.x = -INFINITY;
}

// The box of triangle x for the box tests, as T or as S (as outward_floats into f).
inline __device__ void queue_box(const Vec3* x, bool is_s, float* f) {
  Box b = box_of(x, 3);
  if (detail::parallel_to_x(cross(x[1] - x[0], x[2] - x[0]))) never_queued(b, is_s);
  outward_floats(b, f);
}

// The box test, on boxes as outward_floats (lo, hi): whether T's (t) can reach S's (s, or a cell's):
// T's top above S's bottom (T must be above S somewhere), and the two overlapping in y and z. The
// comparisons are combined without branches, so that a warp's lanes stay together.
inline __device__ bool may_overlap(const float* t, const float* s) {
  return (t[3] > s[0]) & (t[1] <= s[4]) & (s[1] <= t[4]) & (t[2] <= s[5]) & (s[2] <= t[5]);
}

// Appends `item` to the queue at `count`, for each lane where `keep`, in order of lane; returns the
// new count, the same in every lane.
inline __device__ int warp_append(bool keep, int item, unsigned short* queue, int count, int lane) {
  unsigned ballot = __ballot_sync(~0u, keep);
  if (keep) queue[count + __popc(ballot & ((1u << lane) - 1))] = item;
  return count + __popc(ballot);
}

// The sum of v over the warp, in every lane.
inline __device__ double warp_sum(double v) {
  for (int d = 16; d > 0; d /= 2) v += __shfl_xor_sync(~0u, v, d);
  return v;
}
inline __device__ Moments warp_sum(const Moments& m) {
  return {warp_sum(m.volume), {warp_sum(m.first.x), warp_sum(m.first.y), warp_sum(m.first.z)}};
}

}  // namespace tet::gpu
