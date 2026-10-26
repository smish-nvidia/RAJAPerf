//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// INTSC_HEXRECT by the projection ("direct") method: the Base_CUDA tuning "projection". Each
// record's overlap comes from the donor hex's 24 boundary triangles and the target zone's box
// directly (tet/projection.hpp). A warp handles a donor zone, setting up its triangles once, then
// its records 16 at a time. A copy of cell_projection_moments from tet_intersection's
// src/intersect_hex_rect.cu, reading the suite's donor mesh, target planes and records.

#include "INTSC_HEXRECT.hpp"

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

// The suite numbers a zone's nodes x fastest: node l at (l & 1, l >> 1 & 1, l >> 2 & 1). hex.hpp
// numbers them counterclockwise around the bottom face, then the top (as VTK does): hex corner c
// is the suite's node c with bit 0 flipped where bit 1 is set (2 <-> 3, 6 <-> 7).
__device__ inline int suite_corner(int c) { return c ^ (c >> 1 & 1); }

// offsets[d]: the first record of donor zone d, so that its records are offsets[d] .. offsets[d + 1]
// (offsets[ndzones] = nrecords), a thread per record. ponytail: needs the records sorted by donor
// zone, as setupIntscPairs makes them; records in any order would need a sort first.
__global__ void donor_offsets(Int_ptr intsc_d, Index_type nrecords, Index_type ndzones, Int_ptr offsets) {
  Index_type r = blockIdx.x * Index_type(blockDim.x) + threadIdx.x;
  if (r > nrecords) return;
  Index_type lo = r == 0 ? -1 : intsc_d[r - 1], hi = r == nrecords ? ndzones : intsc_d[r];
  for (Index_type d = lo + 1; d <= hi; ++d) offsets[d] = r;
}

// What a warp working on a donor zone against its target zones keeps in shared memory, and the
// steps of its work (intsc_hexrect_projection). The target zones come a chunk of kCells at a time;
// item k = 24 c + t is the chunk's zone c against the donor's triangle t. Everything is relative to
// the donor's centre o, for conditioning. Every member is called by all the warp's lanes.
struct CellsWarp {
  static constexpr int kCells = 16;  // target zones per chunk

  double         tri[24][9];          // the donor's triangles (3 corners)
  float          tbox[24][6];         // and their boxes (queue_box)
  double         cbox[kCells][6];     // the chunk's target zones: lo, hi
  float          cboxf[kCells][6];    // and as outward_floats, for the box test
  double         acc[kCells][4];      // the chunk's records' moments
  unsigned short todo[kCells * 24];   // the queue of items
  ProjScratch    scratch[32];         // a polygon per lane, for add_terms

  // 1. Triangle `lane` < 24 of hex a: stored, with its box.
  __device__ void set_up_triangles(const Hex& a, Vec3 o, int lane) {
    if (lane < 24) {
      Vec3 x[3];
      a.triangle(lane, o, x);
      store_triangle(x, tri[lane]);
      queue_box(x, false, tbox[lane]);
    }
  }

  // 2. The target zones of the nc records from r0 on, a zone per lane, from the suite's planes;
  // their moments zeroed.
  __device__ void load_cells(const INTSC_HEXRECT::Plane& pl, Int_ptr intsc_t, Index_type r0, int nc, Vec3 o, int lane) {
    if (lane < nc) {
      const int     nx = pl.ncord[2], ny = pl.ncord[1], tz = intsc_t[r0 + lane];
      const int     jz = tz / (nx * ny), jy = tz / nx % ny, jx = tz % nx;
      const double *xp = pl.planes[2], *yp = pl.planes[1], *zp = pl.planes[0];
      double*       cb = cbox[lane];
      cb[0] = xp[jx] - o.x, cb[1] = yp[jy] - o.y, cb[2] = zp[jz] - o.z;
      cb[3] = xp[jx + 1] - o.x, cb[4] = yp[jy + 1] - o.y, cb[5] = zp[jz + 1] - o.z;
      outward_floats({{cb[0], cb[1], cb[2]}, {cb[3], cb[4], cb[5]}}, cboxf[lane]);
      for (int k = 0; k < 4; ++k) acc[lane][k] = 0;
    }
    __syncwarp();
  }

  // 3. The box test of the chunk's 24 nc items, an item per lane, queueing those that pass (in order
  // of zone). Each lane advances its item's indices from round to round rather than dividing them
  // out of k. Returns the queue's length.
  __device__ int queue_box_overlaps(int nc, int lane) {
    int count = 0;
    for (int k0 = 0, c = lane / 24, t = lane % 24; k0 < nc * 24; k0 += 32) {
      int  k    = k0 + lane;  // = 24 c + t
      bool keep = k < nc * 24 && may_overlap(tbox[t], cboxf[c]);
      t += 8, c += 1 + (t >= 24), t -= t >= 24 ? 24 : 0;  // k += 32
      count = warp_append(keep, k, todo, count, lane);
    }
    __syncwarp();
    return count;
  }

  // 4. The terms of the queue's items, a lane per item, summed per zone into acc.
  __device__ void add_terms(int count, int lane) {
    for (int r0 = 0; r0 < count; r0 += 32) {
      int     c = -1;  // the item's zone in the chunk
      Moments m;
      if (r0 + lane < count) {
        int k = todo[r0 + lane];
        c     = k / 24;
        Vec3          x[3];
        const double* cb = cbox[c];
        load_triangle(tri[k % 24], x);
        add_projected_box(x, {cb[0], cb[1], cb[2]}, {cb[3], cb[4], cb[5]}, m, scratch[lane]);
      }
      add_by_key(c, m, acc, lane);
    }
  }

  // 5. Lane c < nc writes the record of the chunk's zone c: its volume and first moments, however
  // small (the suite keeps every overlap).
  __device__ void write(int nc, int lane, Vec3 o, Real_ptr out) {
    if (lane < nc) {
      double v = acc[lane][0];
      out[4 * lane]     = v;
      out[4 * lane + 1] = acc[lane][1] + v * o.x;
      out[4 * lane + 2] = acc[lane][2] + v * o.y;
      out[4 * lane + 3] = acc[lane][3] + v * o.z;
    }
    __syncwarp();
  }
};

// A warp (block) per donor zone, over its records (offsets[d] ...) a chunk at a time. Each round's
// terms are summed per zone with a segmented scan over the lanes (add_by_key), so the result does
// not depend on timing (as it would with atomics).
__global__ void __launch_bounds__(32) intsc_hexrect_projection(Real_ptr xdnode, Real_ptr ydnode, Real_ptr zdnode,
                                                               Int_ptr znlist, Char_ptr ncord, Int_ptr intsc_t,
                                                               Int_ptr offsets, Real_ptr records) {
  __shared__ CellsWarp w;
  const int        d = blockIdx.x, lane = threadIdx.x;
  const Index_type first = offsets[d], nrec = offsets[d + 1] - first;
  Hex              a;
  for (int c = 0; c < 8; ++c) {
    int node = znlist[8 * d + suite_corner(c)];
    a.v[c]   = {xdnode[node], ydnode[node], zdnode[node]};
  }
  const Vec3 o = a.center();
  w.set_up_triangles(a, o, lane);
  const INTSC_HEXRECT::Plane& pl = *reinterpret_cast<const INTSC_HEXRECT::Plane*>(ncord);
  for (Index_type c0 = 0; c0 < nrec; c0 += CellsWarp::kCells) {
    const int nc = nrec - c0 < CellsWarp::kCells ? nrec - c0 : CellsWarp::kCells;
    w.load_cells(pl, intsc_t, first + c0, nc, o, lane);
    int count = w.queue_box_overlaps(nc, lane);
    w.add_terms(count, lane);
    w.write(nc, lane, o, records + 4 * (first + c0));
  }
}

}  // namespace


void INTSC_HEXRECT::runCudaVariantProjection(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getCudaResource()};

  // The first record of each donor zone (and one past the last).
  Int_ptr offsets;
  allocData(offsets, m_ndzones + 1, vid);

  if ( vid == Base_CUDA ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      constexpr Size_type offsets_block_size = 256;
      const Size_type offsets_grid_size = RAJA_DIVIDE_CEILING_INT(m_nrecords + 1, offsets_block_size);
      constexpr Size_type shmem = 0;

      RP_CALI_SUBKERNEL_BEGIN("INTSC_HEXRECT_PROJECTION_OFFSETS");
      RPlaunchCudaKernel( donor_offsets,
                          offsets_grid_size, offsets_block_size,
                          shmem, res.get_stream(),
                          m_intsc_d, m_nrecords, m_ndzones, offsets ) ;
      RP_CALI_SUBKERNEL_END("INTSC_HEXRECT_PROJECTION_OFFSETS");

      RP_CALI_SUBKERNEL_BEGIN("INTSC_HEXRECT_PROJECTION");
      RPlaunchCudaKernel( intsc_hexrect_projection,
                          m_ndzones, 32,
                          shmem, res.get_stream(),
                          m_xdnode, m_ydnode, m_zdnode, m_znlist,
                          m_ncord, m_intsc_t, offsets, m_records ) ;
      RP_CALI_SUBKERNEL_END("INTSC_HEXRECT_PROJECTION");

    }
    stopTimer();

  } else {
     getCout() << "\n  INTSC_HEXRECT : Unknown Cuda variant id = " << vid << std::endl;
  }

  deallocData(offsets, vid);
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
