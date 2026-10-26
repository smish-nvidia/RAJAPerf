//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// INTSC_HEXHEX by the projection ("direct") method: the Base_CUDA tuning "projection". Each subzone
// pair's overlap comes from the two subzones' boundary triangles directly (tet/projection.hpp):
// HexSplit::diagonal, the 12 triangles that bound the suite's 6 tets around the diagonal from
// corner 0 to 7, and a warp handles 4 pairs at a time. A copy of hex_projection_moments_multi from
// tet_intersection's src/intersect_hex_hex.cu, reading the suite's corners (dsubz, tsubz) and
// writing its records (vv_out) in one kernel, with no fixup pass.

#include "INTSC_HEXHEX.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include "tet/warp.cuh"

#include <iostream>

namespace rajaperf
{
namespace apps
{

namespace
{

using namespace tet;
using namespace tet::gpu;

// The suite numbers a subzone's corners x fastest: corner l at (l & 1, l >> 1 & 1, l >> 2 & 1).
// hex.hpp numbers them counterclockwise around the bottom face, then the top (as VTK does): hex
// corner c is the suite's corner c with bit 0 flipped where bit 1 is set (2 <-> 3, 6 <-> 7).
__device__ inline int suite_corner(int c) { return c ^ (c >> 1 & 1); }

// What a warp working on P subzone pairs keeps in shared memory, and the steps of its work, in
// order (intsc_hexhex_projection). Pair q's subzones are h[q][0] (the donor) and h[q][1] (the
// target); item k = nn q + nt t + s is triangle t of the donor (T) against triangle s of the target
// (S). Triangles are rebuilt from the corners when needed, rather than stored. Every member is
// called by all the warp's lanes.
template <int P>
struct PairsWarp {
  static constexpr HexSplit split = HexSplit::diagonal;
  static constexpr int nt = hex_triangles(split), nn = nt * nt;  // triangles per hex, items per pair

  Hex            h[P][2];              // the pairs' subzones
  double         o[P][3];              // their local origins, for conditioning: the donors' centres
  double         slope[P][2 * nt][2];  // the triangles' planes' slopes (PlaneX::ry, rz), T's then S's
  signed char    sx[P][2 * nt];        // and the signs of their normals' x
  double         acc[P][4];            // the pairs' moments
  unsigned short todo[P * nn];         // the queue of items
  float          margin[P];            // for separated_yz
  // The boxes and float corners are done with before the polygons are needed, so they share memory.
  union {
    struct {
      float box[P][2 * nt][6];  // the triangles' boxes, T's then S's (queue_box)
      float hf[P][2][8][2];     // the corners' y and z relative to o, as floats, for drop_empty
    };
    ProjScratch scratch[32];  // a polygon per lane, for add_terms
  };

  __device__ Vec3 origin(int q) const { return {o[q][0], o[q][1], o[q][2]}; }

  // Triangle i of pair q's subzone on `side` (0: donor, 1: target), relative to the pair's origin,
  // and its plane.
  __device__ void triangle(int q, int side, int i, Vec3* x) const { h[q][side].template triangle<split>(i, origin(q), x); }
  __device__ tet::detail::PlaneX plane(int q, int side, int i, const Vec3* x) const {
    const double* r = slope[q][side * nt + i];
    return {x[0], r[0], r[1]};
  }

  // The corners' y and z, as floats, of triangle i of pair q's subzone on `side`.
  __device__ void float_corners(int q, int side, int i, float (*f)[2]) const {
    int c[3];
    hex_diagonal_triangle(i, c);
    for (int k = 0; k < 3; ++k) f[k][0] = hf[q][side][c[k]][0], f[k][1] = hf[q][side][c[k]][1];
  }

  // 1. The np pairs' corners from the suite's arrays (24 doubles per subzone: x, y, z of its 8
  // corners), a corner per lane; then each pair's origin.
  __device__ void load(Real_ptr dsubz, Real_ptr tsubz, Index_type p0, int np, int lane) {
    for (int k = lane; k < 16 * np; k += 32) {  // corner k % 16 of pair k / 16: the donor's, then the target's
      int           q = k / 16, c = k % 16;
      const double* z = (c < 8 ? dsubz : tsubz) + 24 * (p0 + q) + suite_corner(c % 8);
      h[q][c / 8].v[c % 8] = {z[0], z[8], z[16]};
    }
    __syncwarp();
    if (lane < np) {
      Vec3 c = h[lane][0].center();
      o[lane][0] = c.x, o[lane][1] = c.y, o[lane][2] = c.z;
      for (int i = 0; i < 4; ++i) acc[lane][i] = 0;
    }
    __syncwarp();
  }

  // 2. A triangle per lane: its box (never queued if parallel to x) and its plane's slopes.
  __device__ void set_up_triangles(int np, int lane) {
    for (int k = lane; k < 2 * nt * np; k += 32) {  // triangle i of pair q: the donor's, then the target's
      int            q = k / (2 * nt), i = k % (2 * nt), side = i / nt;
      Vec3           x[3], nrm;
      tet::detail::PlaneX pl;
      h[q][side].template triangle<split>(i % nt, origin(q), x);
      Box b = box_of(x, 3);
      if (!tet::detail::plane_x(x, nrm, pl)) never_queued(b, side);
      slope[q][i][0] = pl.ry, slope[q][i][1] = pl.rz, sx[q][i] = nrm.x > 0 ? 1 : -1;
      outward_floats(b, box[q][i]);
    }
    __syncwarp();
  }

  // 3. The box test of all np nn items, an item per lane, queueing those that pass (in order of pair).
  // Each lane advances its item's indices from round to round rather than dividing them out of k.
  // Returns the queue's length.
  __device__ int queue_box_overlaps(int np, int lane) {
    int count = 0;
    for (int k0 = 0, q = lane / nn, t = lane % nn / nt, s = lane % nt; k0 < np * nn; k0 += 32) {
      int k = k0 + lane;  // = nn q + nt t + s
      count = warp_append(k < np * nn && may_overlap(box[q][t], box[q][nt + s]), k, todo, count, lane);
      s += 32 % nt, t += 32 / nt % nt + (s >= nt), s -= s >= nt ? nt : 0;  // k += 32
      q += 32 / nn + (t >= nt), t -= t >= nt ? nt : 0;
    }
    __syncwarp();
    return count;
  }

  // 4. Drops from the queue of `count` the items that separated_yz finds add nothing, so that they
  // take no lane in add_terms. The float corners and margins are set up here, for warps whose queue
  // is long enough to call this. In place: each round writes only entries it has read. Returns the
  // new length.
  __device__ int drop_empty(int count, int np, int lane) {
    for (int k0 = 0; k0 < 16 * np; k0 += 32) {  // a corner per lane, as floats; each 16 lanes find their pair's margin
      int   k = k0 + lane, q = k / 16, c = k % 16;
      float l = 0;
      if (k < 16 * np) {
        const Vec3& v = h[q][c / 8].v[c % 8];
        float       y = __double2float_rn(v.y - o[q][1]), z = __double2float_rn(v.z - o[q][2]);
        hf[q][c / 8][c % 8][0] = y, hf[q][c / 8][c % 8][1] = z, l = fmaxf(fabsf(y), fabsf(z));
      }
      for (int d = 8; d > 0; d /= 2) l = fmaxf(l, __shfl_xor_sync(~0u, l, d));
      if (c == 0 && k < 16 * np) margin[q] = 0x1p-18f * l * l * 1.0001f;
    }
    __syncwarp();
    int kept = 0;
    for (int r0 = 0; r0 < count; r0 += 32) {
      int  k    = r0 + lane < count ? todo[r0 + lane] : 0;
      bool keep = false;
      if (r0 + lane < count) {
        int   q = k / nn, t = k % nn / nt, s = k % nt;
        float tf[3][2], sf[3][2];
        float_corners(q, 0, t, tf);
        float_corners(q, 1, s, sf);
        keep = !separated_yz(tf, sf, margin[q]);
      }
      kept = warp_append(keep, k, todo, kept, lane);
    }
    __syncwarp();
    return kept;
  }

  // 5. The terms of the queue's items, a lane per item, summed per pair into acc.
  __device__ void add_terms(int count, int lane) {
    for (int r0 = 0; r0 < count; r0 += 32) {
      int     q = -1;  // the item's pair
      Moments m;
      if (r0 + lane < count) {
        int k = todo[r0 + lane], t = k % nn / nt, s = k % nt;
        q     = k / nn;
        Vec3 x[3], y[3];
        triangle(q, 0, t, x);
        triangle(q, 1, s, y);
        add_projected_pair(x, y, 3, plane(q, 0, t, x), plane(q, 1, s, y), sx[q][nt + s], m, scratch[lane]);
      }
      add_by_key(q, m, acc, lane);
    }
  }

  // 6. Lane q < np writes pair q's record: its volume and first moments, however small (the suite
  // keeps every overlap).
  __device__ void write(int np, int lane, Real_ptr out) const {
    if (lane >= np) return;
    Vec3   c = origin(lane);
    double v = acc[lane][0];
    out[4 * lane]     = v;
    out[4 * lane + 1] = acc[lane][1] + v * c.x;
    out[4 * lane + 2] = acc[lane][2] + v * c.y;
    out[4 * lane + 3] = acc[lane][3] + v * c.z;
  }
};

constexpr int pairs_per_warp = 4;

// A warp (block) per 4 subzone pairs, so that pairs with few items (8 per pair when the subzones
// are boxes) share their rounds. When the queue is longer than a round, drop_empty first drops the
// items that add nothing; with a round or less, that would only cost.
__global__ void __launch_bounds__(32) intsc_hexhex_projection(Real_ptr dsubz, Real_ptr tsubz, Index_type n,
                                                              Real_ptr vv_pair) {
  __shared__ PairsWarp<pairs_per_warp> w;
  const int        lane = threadIdx.x;
  const Index_type p0   = blockIdx.x * Index_type(pairs_per_warp);
  const int        np   = n - p0 < pairs_per_warp ? n - p0 : pairs_per_warp;
  w.load(dsubz, tsubz, p0, np, lane);
  w.set_up_triangles(np, lane);
  int count = w.queue_box_overlaps(np, lane);
  if (count > 32) count = w.drop_empty(count, np, lane);
  w.add_terms(count, lane);
  w.write(np, lane, vv_pair + 4 * p0);
}

}  // namespace


void INTSC_HEXHEX::runCudaVariantProjection(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getCudaResource()};

  if ( vid == Base_CUDA ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      const Size_type grid_size = RAJA_DIVIDE_CEILING_INT(m_n_subz_intsc, pairs_per_warp);
      constexpr Size_type shmem = 0;

      RP_CALI_SUBKERNEL_BEGIN("INTSC_HEXHEX_PROJECTION");
      RPlaunchCudaKernel( intsc_hexhex_projection,
                          grid_size, 32,
                          shmem, res.get_stream(),
                          m_dsubz, m_tsubz, m_n_subz_intsc, m_vv_out ) ;
      RP_CALI_SUBKERNEL_END("INTSC_HEXHEX_PROJECTION");

    }
    stopTimer();

  } else {
     getCout() << "\n  INTSC_HEXHEX : Unknown Cuda variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
