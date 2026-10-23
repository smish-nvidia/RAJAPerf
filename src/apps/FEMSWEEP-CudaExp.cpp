//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Registers the experimental FEMSWEEP CUDA kernels as Base_CUDA tunings, so
// they run, are checksummed, and are timed alongside the reference
// implementations:
//
//   raja-perf.exe -k FEMSWEEP -v Base_CUDA RAJA_CUDA \
//     -t unroll_64 unroll_64_min_blocks_per_sm_5 coop_256 daggraph_256
//
//   coop_256         original schedule, cooperative element solve (FEMSWEEP-CudaExpCoop.cpp)
//   daglevels_256    all angles and groups as one DAG, one launch per hyperplane
//   daggraph_256     the same launches, replayed from a CUDA graph
//   dagqueue_256     one persistent launch pulling ready elements from a queue
//   daggraphgi_256   daggraph with X kept group innermost during the sweep
//   dagqueuegi_256   dagqueue with X kept group innermost during the sweep
//                    (DAG kernels: FEMSWEEP-CudaExpDag.cpp)
//

#include "FEMSWEEP-CudaLaunch.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

namespace rajaperf
{
namespace apps
{

void FEMSWEEP::defineCudaExperimentalTunings()
{
  addVariantTuning<&FEMSWEEP::runCudaLaunch<256, femsweepLaunchCoop<256>>>(Base_CUDA, "coop_256");

  addVariantTuning<&FEMSWEEP::runCudaDag<FemsweepDagMode::levels>>(Base_CUDA, "daglevels_256");
  addVariantTuning<&FEMSWEEP::runCudaDag<FemsweepDagMode::graph>>(Base_CUDA, "daggraph_256");
  addVariantTuning<&FEMSWEEP::runCudaDag<FemsweepDagMode::queue>>(Base_CUDA, "dagqueue_256");
  addVariantTuning<&FEMSWEEP::runCudaDag<FemsweepDagMode::graph, true>>(Base_CUDA, "daggraphgi_256");
  addVariantTuning<&FEMSWEEP::runCudaDag<FemsweepDagMode::queue, true>>(Base_CUDA, "dagqueuegi_256");
}

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_CUDA
