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
// femsweep-bench: every FEMSWEEP CUDA implementation (original and
// experimental) on one set of input data, checked against the original kernel
// and timed with CUDA events.
//
// Takes the RAJAPerf problem options, e.g.
//
//   femsweep-bench.exe --femsweep_mesh_dims 32 32 32 --femsweep_groups 16
//   femsweep-bench.exe --repfact 5        (50 timed sweeps per implementation)
//
// Unlike raja-perf.exe, the input data is generated once and nothing is
// checksummed on the host, so large meshes take seconds rather than minutes.
//

#include "apps/FEMSWEEP-CudaLaunch.hpp"
#include "common/RunParams.hpp"

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using namespace rajaperf;
using namespace rajaperf::apps;

// one implementation: run() is one sweep and is what gets timed; before() and
// after() bracket a batch of sweeps, untimed (e.g. an X layout conversion)
struct Impl {
  std::string origin;
  std::string name;
  std::function< void(cudaStream_t) > run;
  std::function< void(cudaStream_t) > before = [](cudaStream_t) {};
  std::function< void(cudaStream_t) > after  = [](cudaStream_t) {};
};

struct Result {
  const Impl * impl;
  double ms;      // per sweep
  double error;   // max |x - x_ref| / max |x_ref|, x_ref from the original kernel
};

static void check(cudaError_t status, const char * what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

// max |x - ref| over all entries, relative to max |ref|. A non-finite entry
// in x makes the error infinite.
static double maxRelError(const Real_type * x, const Real_type * ref, Index_type n) {
  auto it = thrust::counting_iterator< Index_type >(0);
  const double diff = thrust::transform_reduce(thrust::device, it, it + n,
      [=] __device__ (Index_type i) -> double { return isfinite(x[i]) ? fabs(x[i] - ref[i]) : INFINITY; },
      0.0, thrust::maximum< double >());
  const double scale = thrust::transform_reduce(thrust::device, it, it + n,
      [=] __device__ (Index_type i) -> double { return fabs(ref[i]); },
      0.0, thrust::maximum< double >());
  return diff / scale;
}

int main(int argc, char ** argv) {
  RunParams params(argc, argv);
  if (params.getInputState() == RunParams::InfoRequest) { return EXIT_SUCCESS; }
  if (params.getInputState() == RunParams::BadInput) { return EXIT_FAILURE; }

  const Index_type reps = params.getReps(10);

  // same mesh and input data raja-perf.exe would use
  FEMSWEEP kernel(params);
  kernel.setUp(Base_CUDA, 0);
  const FemsweepData d = kernel.data();

  cudaStream_t stream;
  check(cudaStreamCreate(&stream), "cudaStreamCreate");

  auto launch = [&d](void (*f)(const FemsweepData &, cudaStream_t)) {
    return [&d, f](cudaStream_t s) { f(d, s); };
  };

  FemsweepData dC = d;
  dC.faces = femsweepFaceTable(d, stream);
  FemsweepDag dag(d, stream);

  const std::vector< Impl > impls = {
    {"original",  "block_64",             launch(femsweepLaunchOriginal<64, false>)},
    {"original",  "unroll_64",            launch(femsweepLaunchOriginal<64, true>)},
    {"original",  "raja_unroll_64_min_blocks_per_sm_5", launch(femsweepLaunchRaja<64, true, 5>)},
    // face connectivity table built once at setup (untimed)
    {"ours",      "coop_256",             [&](cudaStream_t s) { femsweepLaunchCoop<256>(dC, s); }},
    // all angles and groups as one DAG; level lists, face table, graph built once at setup (untimed)
    {"ours",      "daglevels_256",        [&](cudaStream_t s) { dag.sweep(FemsweepDagMode::levels, false, s); }},
    {"ours",      "daggraph_256",         [&](cudaStream_t s) { dag.sweep(FemsweepDagMode::graph, false, s); }},
    {"ours",      "dagqueue_256",         [&](cudaStream_t s) { dag.sweep(FemsweepDagMode::queue, false, s); }},
    // X kept group innermost during the sweep, then unpacked into Xdat;
    // before() zeroes that buffer so the checked sweep can't read a previous
    // sweep's values
    {"ours",      "daggraphgi_256",
     [&](cudaStream_t s) { dag.sweep(FemsweepDagMode::graph, true, s); dag.unpack(s); },
     [&](cudaStream_t s) { dag.clearGroupInner(s); }},
    {"ours",      "dagqueuegi_256",
     [&](cudaStream_t s) { dag.sweep(FemsweepDagMode::queue, true, s); dag.unpack(s); },
     [&](cudaStream_t s) { dag.clearGroupInner(s); }},
    // the same sweeps without the unpack, as an application keeping X group
    // innermost would run them (X is unpacked once, untimed, for the check)
    {"ours",      "daggraphgi_256 (sweep only)",
     [&](cudaStream_t s) { dag.sweep(FemsweepDagMode::graph, true, s); },
     [&](cudaStream_t s) { dag.clearGroupInner(s); },
     [&](cudaStream_t s) { dag.unpack(s); }},
    {"ours",      "dagqueuegi_256 (sweep only)",
     [&](cudaStream_t s) { dag.sweep(FemsweepDagMode::queue, true, s); },
     [&](cudaStream_t s) { dag.clearGroupInner(s); },
     [&](cudaStream_t s) { dag.unpack(s); }},
  };

  const Size_type xbytes = Size_type(d.Xlen) * sizeof(Real_type);
  Real_ptr x_ref;
  check(cudaMalloc(&x_ref, xbytes), "cudaMalloc(x_ref)");

  cudaEvent_t start, stop;
  check(cudaEventCreate(&start), "cudaEventCreate");
  check(cudaEventCreate(&stop), "cudaEventCreate");

  // reference solution: the original unrolled kernel
  check(cudaMemsetAsync(d.Xdat, 0, xbytes, stream), "cudaMemsetAsync");
  femsweepLaunchOriginal<64, true>(d, stream);
  check(cudaMemcpyAsync(x_ref, d.Xdat, xbytes, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync(x_ref)");

  const double x_max = thrust::transform_reduce(thrust::device, x_ref, x_ref + d.Xlen,
      [] __device__ (Real_type x) -> double { return fabs(x); }, 0.0, thrust::maximum< double >());

  // relative difference from the original kernel above which a result is
  // rejected. The element systems are diagonally dominant (see FEMSWEEP.hpp),
  // so correct LU orderings agree to round-off, while a face-coupling bug
  // shows up at the size of the coupling (~Ffactor).
  constexpr double tolerance = 1.0e-12;

  std::vector< Result > results;
  for (const Impl & impl : impls) {
    // timed sweeps, after one warm-up
    impl.before(stream);
    impl.run(stream);
    check(cudaEventRecord(start, stream), "cudaEventRecord");
    for (Index_type r = 0; r < reps; ++r) { impl.run(stream); }
    check(cudaEventRecord(stop, stream), "cudaEventRecord");
    impl.after(stream);
    check(cudaStreamSynchronize(stream), impl.name.c_str());

    float ms;
    check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");

    // checked sweep, after the timed ones so state an implementation keeps
    // between sweeps is checked too: X starts zeroed, so anything left
    // unwritten, or read before it is written, is caught. (Each incoming
    // face's upwind element lies in an earlier hyperplane, so a correct sweep
    // doesn't depend on the starting X.)
    check(cudaMemsetAsync(d.Xdat, 0, xbytes, stream), "cudaMemsetAsync");
    impl.before(stream);
    impl.run(stream);
    impl.after(stream);
    check(cudaStreamSynchronize(stream), impl.name.c_str());
    const double error = maxRelError(d.Xdat, x_ref, d.Xlen);
    results.push_back({&impl, ms / reps, error});
  }

  const double ms_ref = std::find_if(results.begin(), results.end(),
      [](const Result & r) { return r.impl->name == "unroll_64"; })->ms;   // original unroll_64
  std::sort(results.begin(), results.end(), [](const Result & a, const Result & b) { return a.ms < b.ms; });

  cudaDeviceProp prop;
  check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
  std::printf("%s: %ld elements, %ld hyperplanes, %ld angles, %ld groups, %ld timed sweeps each\n",
              prop.name, long(d.ne), long(d.hplanes), long(d.na), long(d.ng), long(reps));
  std::printf("max |x| = %.3g, valid = max |x - x_original| / max |x_original| <= %.0e\n\n", x_max, tolerance);
  std::printf("| %-9s | %-34s | %10s | %10s | %9s | %-5s |\n", "origin", "implementation", "ms/sweep", "vs unroll", "rel error", "valid");
  std::printf("|:----------|:-----------------------------------|-----------:|-----------:|----------:|:------|\n");
  bool all_valid = true;
  for (const Result & r : results) {
    const bool valid = r.error <= tolerance;
    all_valid = all_valid && valid;
    std::printf("| %-9s | %-34s | %10.3f | %9.2fx | %9.1e | %-5s |\n",
                r.impl->origin.c_str(), r.impl->name.c_str(), r.ms, ms_ref / r.ms, r.error, valid ? "yes" : "NO");
  }

  check(cudaEventDestroy(start), "cudaEventDestroy");
  check(cudaEventDestroy(stop), "cudaEventDestroy");
  check(cudaFree(x_ref), "cudaFree(x_ref)");
  detail::deallocCudaDeviceData(const_cast<int *>(dC.faces));
  check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  kernel.tearDown(Base_CUDA, 0);

  return all_valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
