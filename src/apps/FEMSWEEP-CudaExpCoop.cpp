//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Experimental FEMSWEEP CUDA kernel "coop": the original kernel's schedule
// (one block per (angle, group), walking its hyperplanes with a barrier
// between them) with a cooperative element solve: 4 lanes per element
// instead of 1 thread (femsweepSolveElement, FEMSWEEP-CudaElement.hpp).
//
// Also defines femsweepFaceTable, the per-(element, face) connectivity table
// that all the experimental kernels read.
//
//   raja-perf.exe -k FEMSWEEP -v Base_CUDA -t coop_256 unroll_64
//

#include "FEMSWEEP-CudaLaunch.hpp"
#include "FEMSWEEP-CudaElement.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

namespace rajaperf
{
namespace apps
{

// The original face loop walks a chain of dependent loads before it can
// touch Fdat and Xdat:
//   elem_to_faces -> F_g2l -> idx1 (which side) -> idx1/idx2 (dof indices)
// None of that depends on the angle or energy group, so it is resolved once
// per (element, local face) into one entry of 8 ints (two 16-byte loads):
//   [0]    offset of the face's 4x4 block in Fdat for angle 0,
//          (s^1)*FDS*FDS + f*2*FDS*FDS, or -1 for a boundary face
//   [1]    element dof that face dof j's term lands on, 3 bits each (j = 0..3)
//   [2..3] unused
//   [4..7] upwind dof index (dis) of face dof i = 0..3
// ponytail: 32-bit entries, so f*2*FDS*FDS must stay below 2^31 (~67M
// interior faces) and dof indices below 2^31; widen the entry past that.
__global__ void FEMSweepBuildFaceTable(int * faces,
                                       const Index_type ne,
                                       const Index_ptr elem_to_faces,
                                       const Index_ptr F_g2l,
                                       const Index_ptr idx1,
                                       const Index_ptr idx2) {
  const Index_type t = Index_type(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= ne * NLF) return;
  const Index_type e    = t / NLF;
  const Index_type face = t % NLF;

  int4 head = {-1, 0, 0, 0};
  int4 dis  = {0, 0, 0, 0};
  const Index_type sf_gl = F_g2l[elem_to_faces[NLF * e + face]];
  if (sf_gl >= 0) {
    const Index_type f = sf_gl;
    const Index_type s = (e == idx1[f * FDS] / ND) ? 0 : 1;
    const Index_ptr target = s == 0 ? idx1 : idx2;   // djs
    const Index_ptr source = s == 0 ? idx2 : idx1;   // dis
    head.x = int((s ^ 1) * FDS * FDS + f * 2 * FDS * FDS);
    for (Index_type j = 0; j < FDS; ++j) {
      head.y |= int(target[f * FDS + j] % ND) << (3 * j);
    }
    dis = {int(source[f * FDS + 0]), int(source[f * FDS + 1]),
           int(source[f * FDS + 2]), int(source[f * FDS + 3])};
  }
  reinterpret_cast<int4 *>(faces)[2 * t]     = head;
  reinterpret_cast<int4 *>(faces)[2 * t + 1] = dis;
}

int * femsweepFaceTable(const FemsweepData & d, cudaStream_t stream) {
  const Index_type n = d.ne * NLF;
  int * faces = static_cast<int *>(detail::allocCudaDeviceData(n * 8 * sizeof(int)));
  constexpr unsigned threads = 256;
  RPlaunchCudaKernel(FEMSweepBuildFaceTable, dim3(unsigned((n + threads - 1) / threads)), threads, 0, stream,
                     faces, d.ne, d.elem_to_faces, d.F_g2l, d.idx1, d.idx2);
  return faces;
}

// One block per (angle a, group g), as the original kernel: each group of
// LPE lanes takes the next element of the current hyperplane, and a block
// barrier separates hyperplanes.
template < size_t block_size, Index_type LPE >
__launch_bounds__(block_size)
__global__ void FEMSweep3DExpCoop(const FemsweepData d) {
  static_assert(block_size % 32 == 0, "element groups must not straddle warps");
  constexpr Index_type groups_per_block = block_size / LPE;

  __shared__ Real_type face_b[groups_per_block][femsweep_face_b_stride<LPE>];

  const Index_type lane = threadIdx.x % LPE;   // owns rows lane + LPE * m
  const Index_type grp  = threadIdx.x / LPE;
  // The groups of a warp can run out of elements at different times, so each
  // shuffles with its own lanes only. (A mask known only at run time costs
  // extra instructions per shuffle; the DAG kernels avoid it, see there.)
  const unsigned mask   = ((1u << LPE) - 1u) << ((threadIdx.x % 32) - lane);

  const Index_type a = blockIdx.y;
  const Index_type g = blockIdx.x;

  const Index_type nhp = d.nhpaa_r[a];
  const Index_type ohp = d.ohpaa_r[a];

  Index_type nehp_pos = 0;
  for (Index_type hp = 0; hp < nhp; ++hp) {
    const Index_type nehp = d.phpaa_r[ohp + hp];

    // one element per group; the trip count is uniform within a group
    for (Index_type k = grp; k < nehp; k += groups_per_block) {
      const Index_type e = d.order_r[k + nehp_pos + a * d.ne];
      femsweepSolveElement<LPE>(d, a, e, g, lane, mask, face_b[grp]);
    }
    __syncthreads();
    nehp_pos += nehp;
  }
}

template < size_t block_size >
void femsweepLaunchCoop(const FemsweepData & d, cudaStream_t stream) {
  RPlaunchCudaKernel( (FEMSweep3DExpCoop<block_size, 4>),
                      dim3(d.ng, d.na), block_size,
                      0, stream, d );
}

template void femsweepLaunchCoop<256>(const FemsweepData &, cudaStream_t);

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_CUDA
