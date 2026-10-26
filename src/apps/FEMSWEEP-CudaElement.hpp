//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// The per-element work of the experimental FEMSWEEP CUDA kernels: gather the
// upwind face terms into the right-hand side, then solve the 8x8 system
// (A + sigma*M0) x = b. It is shared by the per-(angle, group) kernel
// (FEMSWEEP-CudaExpCoop.cpp) and the DAG schedulers (FEMSWEEP-CudaExpDag.cpp),
// which differ only in how they pick which element a group solves next.
//
// A "group" here is LPE consecutive lanes of a warp that solve one element
// together; lane l of the group owns rows l, l + LPE, l + 2*LPE, ... of the
// system. With LPE = 4 a warp solves 8 elements (or one element for 8 energy
// groups) at once, and each lane holds 2 rows.
//

#ifndef RAJAPerf_Apps_FEMSWEEP_CudaElement_HPP
#define RAJAPerf_Apps_FEMSWEEP_CudaElement_HPP

#include "FEMSWEEP.hpp"

#if defined(RAJA_ENABLE_CUDA)

namespace rajaperf
{
namespace apps
{

namespace {

// one row of an 8x8 element matrix, aligned so it loads as four 16-byte loads
struct alignas(16) vec8 {
  __host__ __device__ Real_type & operator[](Index_type i) { return values[i]; }
  __host__ __device__ const Real_type & operator[](Index_type i) const { return values[i]; }

  Real_type values[ND];
};
static_assert(ND == 8, "vec8 assumes ND == 8");

}

// A group's shared memory slice of face terms, [face][element dof], padded
// from 48 to 48 + LPE doubles: the 16 / LPE groups of a half warp then start
// 2 * LPE banks apart, so the clear and the row reads (LPE lanes on
// consecutive doubles) are free of bank conflicts. (A stride of 56, right
// for LPE = 8, made half of LPE = 4's shared memory accesses conflicts.)
template < Index_type LPE >
constexpr Index_type femsweep_face_b_stride = NLF * ND + LPE;

// 1 / x without the IEEE division's slow-path check: the hardware estimate
// (rcp.approx.ftz.f64) refined by two Newton steps. Only for normal,
// well-scaled x, like the pivots of the diagonally dominant element systems
// (see FEMSWEEP.hpp); the result is within an ulp of 1.0 / x but not always
// correctly rounded.
__device__ __forceinline__
Real_type femsweepRcp(const Real_type x) {
  Real_type r;
  asm("rcp.approx.ftz.f64 %0, %1;" : "=d"(r) : "d"(x));
  Real_type err = fma(-x, r, 1.0);
  r = fma(r, err, r);
  err = fma(-x, r, 1.0);
  return fma(r, err, r);
}

// The LPE lanes in `mask` solve element e for angle a and group g; `lane` is
// this thread's index among them.
//
//   face_b       the group's femsweep_face_b_stride<LPE> doubles of shared memory
//   store        false: solve without writing X, so a group with no element
//                of its own can still take part in warp-wide shuffles
//   group_inner  X is laid out [a][e][dof][g] and sigma [e][g] (group
//                innermost) instead of Xdat's [a][g][e][dof] and Sgdat's [g][e]
//
// Expects d.faces = femsweepFaceTable(...).
template < Index_type LPE, bool group_inner = false >
__device__ __forceinline__
void femsweepSolveElement(const FemsweepData & d,
                          const Index_type a,
                          const Index_type e,
                          const Index_type g,
                          const Index_type lane,
                          const unsigned mask,
                          Real_type * face_b,
                          const bool store = true) {
  static_assert(ND % LPE == 0 && 32 % LPE == 0, "LPE must divide ND and the warp size");
  constexpr Index_type RPL = ND / LPE;   // rows per lane
  constexpr Index_type nfj = NLF * FDS;  // (face, face dof) pairs per element

  const Real_ptr Bdat  = d.Bdat;
  const Real_ptr Adat  = d.Adat;
  const Real_ptr Fdat  = d.Fdat;
  const Real_ptr Xdat  = d.Xdat;
  const Real_ptr Sgdat = d.Sgdat;
  const Real_ptr M0dat = d.M0dat;
  const Index_type ne  = d.ne;
  const Index_type ng  = d.ng;
  const Index_type sharedinteriorfaces = d.sharedinteriorfaces;
  const Real_type Ffactor = femsweep_Ffactor;   // upwind coupling scale, see FEMSWEEP.hpp

  // X index of dof index `dof` (= element * ND + element dof) for angle a, group g
  auto Xidx = [&](const Index_type dof) {
    return group_inner ? (a * ne * ND + dof) * ng + g : dof + g * ND * ne + a * ng * ND * ne;
  };

  //////// 1. upwind face terms, into face_b ////////

  // Each of the element's 6 faces has 4 dofs; lane `lane` computes the terms
  // of face dofs j = lane, lane + LPE, ... on every face (24 / LPE per lane).
  // A face term lands on one dof of this element, so it is written to
  // face_b[face][dof] and summed into the row owner's b in step 2. Inactive
  // (outgoing or boundary) faces leave their slots at zero.

  RAJA_UNROLL
  for (Index_type slot = lane; slot < NLF * ND; slot += LPE) {
    face_b[slot] = 0.0;
  }
  __syncwarp(mask);

  RAJA_UNROLL
  for (Index_type q = lane; q < nfj; q += LPE) {
    const Index_type face = q / FDS;
    const Index_type j    = q % FDS;

    // one 32-byte entry per (element, face); see femsweepFaceTable
    const int4 * entry = reinterpret_cast<const int4 *>(d.faces) + 2 * (e * NLF + face);
    const int4 head = entry[0];   // Fdat block offset (-1: boundary), packed target dofs
    const int4 dis  = entry[1];   // upwind dof index of face dof i = 0..3

    if ((d.AngleElem2FaceType[face + e * NLF + a * ne * NLF] != 0) && (head.x >= 0)) {
      const Index_type dof = (head.y >> (3 * j)) & 7;   // element dof this term lands on
      const Real_type * Frow = Fdat + head.x + j * FDS + a * sharedinteriorfaces * 2 * FDS * FDS;
      Real_type F = 0.0;
      F += Ffactor * Frow[0] * Xdat[Xidx(dis.x)];
      F += Ffactor * Frow[1] * Xdat[Xidx(dis.y)];
      F += Ffactor * Frow[2] * Xdat[Xidx(dis.z)];
      F += Ffactor * Frow[3] * Xdat[Xidx(dis.w)];
      face_b[face * ND + dof] = F;
    }
  }
  __syncwarp(mask);

  //////// 2. this lane's rows of b and of (A + sigma*M0) ////////

  const Real_type sigma = Sgdat[group_inner ? e * ng + g : e + g * ne];
  const vec8 * Arows  = reinterpret_cast<const vec8 *>(&Adat[e * ND * ND + a * ne * ND * ND]);
  const vec8 * M0rows = reinterpret_cast<const vec8 *>(&M0dat[e * ND * ND]);
  Real_type b[RPL];
  Real_type row[RPL][ND];
  RAJA_UNROLL
  for (Index_type m = 0; m < RPL; ++m) {
    const Index_type i = lane + LPE * m;
    b[m] = Bdat[i + e * ND + a * ne * ND];
    RAJA_UNROLL
    for (Index_type face = 0; face < NLF; ++face) {
      b[m] -= face_b[face * ND + i];
    }
    const vec8 arow = Arows[i];
    const vec8 mrow = M0rows[i];
    RAJA_UNROLL
    for (Index_type jj = 0; jj < ND; ++jj) {
      row[m][jj] = arow[jj] + sigma * mrow[jj];
    }
  }

  //////// 3. Doolittle LU without pivoting ////////

  // Row kk lives in lane kk % LPE, slot kk / LPE (compile-time constants in
  // the unrolled loops). Step kk broadcasts the pivot row with shuffles, and
  // every lane eliminates column kk from its rows below it. The pivot's owner
  // keeps 1 / pivot in place of the diagonal, which only the back
  // substitution reads again, so that substitution multiplies instead of
  // dividing.

  RAJA_UNROLL
  for (Index_type kk = 0; kk < ND; ++kk) {
    Real_type pivot_row[ND];
    RAJA_UNROLL
    for (Index_type jj = kk; jj < ND; ++jj) {
      pivot_row[jj] = __shfl_sync(mask, row[kk / LPE][jj], kk % LPE, LPE);
    }
    const Real_type inv_pivot = femsweepRcp(pivot_row[kk]);
    if (lane == kk % LPE) {
      row[kk / LPE][kk] = inv_pivot;
    }
    RAJA_UNROLL
    for (Index_type m = 0; m < RPL; ++m) {
      if (lane + LPE * m > kk) {
        row[m][kk] *= inv_pivot;
        RAJA_UNROLL
        for (Index_type jj = kk + 1; jj < ND; ++jj) {
          row[m][jj] -= row[m][kk] * pivot_row[jj];
        }
      }
    }
  }

  //////// 4. forward and back substitution, column by column ////////

  RAJA_UNROLL
  for (Index_type jj = 0; jj < ND - 1; ++jj) {
    const Real_type y = __shfl_sync(mask, b[jj / LPE], jj % LPE, LPE);
    RAJA_UNROLL
    for (Index_type m = 0; m < RPL; ++m) {
      if (lane + LPE * m > jj) {
        b[m] -= row[m][jj] * y;
      }
    }
  }

  RAJA_UNROLL
  for (Index_type jj = ND - 1; jj >= 0; --jj) {
    if (lane == jj % LPE) {
      b[jj / LPE] *= row[jj / LPE][jj];
    }
    const Real_type x = __shfl_sync(mask, b[jj / LPE], jj % LPE, LPE);
    RAJA_UNROLL
    for (Index_type m = 0; m < RPL; ++m) {
      if (lane + LPE * m < jj) {
        b[m] -= row[m][jj] * x;
      }
    }
  }

  if (store) {
    RAJA_UNROLL
    for (Index_type m = 0; m < RPL; ++m) {
      Xdat[Xidx(e * ND + lane + LPE * m)] = b[m];
    }
  }

  // the group must be done reading face_b before its next element clears it
  __syncwarp(mask);
}

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_CUDA

#endif // closing endif for header file include guard
