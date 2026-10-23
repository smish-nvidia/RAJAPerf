//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// One full FEMSWEEP sweep per call, for each CUDA implementation. Used by the
// RAJAPerf tunings (FEMSWEEP::runCudaLaunch, FEMSWEEP::runCudaDag) and by
// femsweep-bench.
//

#ifndef RAJAPerf_Apps_FEMSWEEP_CudaLaunch_HPP
#define RAJAPerf_Apps_FEMSWEEP_CudaLaunch_HPP

#include "FEMSWEEP.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <vector>

namespace rajaperf
{
namespace apps
{

// original kernel, FEMSWEEP-Cuda.cpp
template < size_t block_size, bool unroll >
void femsweepLaunchOriginal(const FemsweepData & d, cudaStream_t stream);
// original RAJA::launch kernel (RAJA_CUDA variant), FEMSWEEP-Cuda.cpp
template < size_t block_size, bool unroll, size_t min_blocks_per_sm >
void femsweepLaunchRaja(const FemsweepData & d, cudaStream_t stream);

// per-(element, face) connectivity table that the experimental kernels read
// (FEMSWEEP-CudaExpCoop.cpp); free with detail::deallocCudaDeviceData
int * femsweepFaceTable(const FemsweepData & d, cudaStream_t stream);

// original schedule with a cooperative element solve (FEMSWEEP-CudaExpCoop.cpp);
// expects d.faces = femsweepFaceTable(...)
template < size_t block_size >
void femsweepLaunchCoop(const FemsweepData & d, cudaStream_t stream);

// All angles and groups as one DAG (FEMSWEEP-CudaExpDag.cpp): warps solve
// (angle, element) for 8 groups at a time, scheduled by hyperplane launches
// (levels), the same launches from a CUDA graph (graph), or a persistent
// kernel pulling ready (angle, element) from a global queue (queue).
//
// With group_inner the sweep keeps X in its own group-innermost buffer, which
// unpack() copies into d.Xdat. The constructor does the untimed setup: level
// lists, face table, dependency counts, transposed sigma, graph capture.
class FemsweepDag {
public:
  static constexpr size_t block_size = 256;

  FemsweepDag(const FemsweepData & d, cudaStream_t stream);
  ~FemsweepDag();

  void sweep(FemsweepDagMode mode, bool group_inner, cudaStream_t stream);
  void unpack(cudaStream_t stream);            // group-innermost X -> d.Xdat
  void clearGroupInner(cudaStream_t stream);   // zero the group-innermost X

private:
  void launchLevels(bool group_inner, cudaStream_t stream);

  FemsweepData d;                        // d.faces: our own femsweepFaceTable
  FemsweepData dg;                       // d with group-innermost Xdat and Sgdat
  Index_type nq;                         // group octets
  std::vector<Index_type> level_begin;   // hyperplane h is items[level_begin[h], level_begin[h + 1])
  int * items;                           // a * ne + e, by hyperplane
  cudaGraphExec_t graph[2];              // by group_inner

  // queue
  int * indeg;                           // upwind faces of each (a, e)
  int * count;                           // upwind (a, e) done
  int * queue;
  unsigned * ends;                       // slots taken, slots pushed
  int queue_blocks[2];                   // by group_inner
};

// The face table is built before the timed region: mesh setup that an
// application would do once, not part of the sweep.
template < size_t block_size, void (*launch)(const FemsweepData &, cudaStream_t) >
void FEMSWEEP::runCudaLaunch(VariantID vid) {
  if (vid != Base_CUDA) {
    getCout() << "\n FEMSWEEP : Unknown CUDA launch variant id = " << vid << std::endl;
    return;
  }
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  auto res{getCudaResource()};
  FemsweepData d = data();
  d.faces = femsweepFaceTable(d, res.get_stream());

  startTimer();
  for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {
    RP_CALI_SUBKERNEL_BEGIN("FEMSWEEP_1");
    launch(d, res.get_stream());
    RP_CALI_SUBKERNEL_END("FEMSWEEP_1");
  }
  stopTimer();

  detail::deallocCudaDeviceData(const_cast<int *>(d.faces));
}

// group_inner: the timed sweep includes unpack(), so X ends in Xdat's layout
// like every other variant
template < FemsweepDagMode mode, bool group_inner >
void FEMSWEEP::runCudaDag(VariantID vid) {
  if (vid != Base_CUDA) {
    getCout() << "\n FEMSWEEP : Unknown CUDA DAG variant id = " << vid << std::endl;
    return;
  }
  setBlockSize(FemsweepDag::block_size);

  const Index_type run_reps = getRunReps();
  auto res{getCudaResource()};
  FemsweepDag dag(data(), res.get_stream());

  startTimer();
  for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {
    RP_CALI_SUBKERNEL_BEGIN("FEMSWEEP_1");
    dag.sweep(mode, group_inner, res.get_stream());
    if constexpr (group_inner) { dag.unpack(res.get_stream()); }
    RP_CALI_SUBKERNEL_END("FEMSWEEP_1");
  }
  stopTimer();
}

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_CUDA

#endif // closing endif for header file include guard
