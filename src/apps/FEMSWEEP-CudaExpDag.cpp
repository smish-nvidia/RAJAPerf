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
// Experimental FEMSWEEP CUDA schedulers that treat the sweeps of all angles
// and groups as one DAG, instead of one block per (angle, group) walking its
// own hyperplanes. The element solve is the same as coop's (4 lanes per
// element, femsweepSolveElement in FEMSWEEP-CudaElement.hpp).
//
// Work item: element e of angle a, solved by one warp for 8 consecutive
// groups at a time. The 8 groups share e and a, so the loads of A, B, F and
// the face table hit the same addresses across the warp.
//
//   levels  hyperplane h of every angle is one launch, one warp per
//           (a, e, group octet)
//   graph   the same launches, replayed from a CUDA graph
//   queue   one persistent launch. Warps take ready (a, e) from a global
//           queue and solve all its group octets; finishing one counts it
//           into each downwind element, and whichever increment completes an
//           element's count pushes it. (Queueing (a, e, octet) instead put an
//           element's octets on different SMs: twice the L2 traffic for A, B,
//           F and the face table, and 10% slower at 32^3.)
//
//   group_inner (with any of the above): X kept group innermost,
//           [a][e][dof][g], so a warp's 8 groups gather and store contiguous
//           X; sigma likewise [e][g]. unpack() converts X back to Xdat's
//           layout.
//
//   raja-perf.exe -k FEMSWEEP -v Base_CUDA -t daglevels_256 daggraph_256 dagqueue_256 coop_256
//

#include "FEMSWEEP-CudaLaunch.hpp"
#include "FEMSWEEP-CudaElement.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <cuda/atomic>

#include <algorithm>

namespace rajaperf
{
namespace apps
{

namespace {

constexpr Index_type dag_lpe = 4;              // lanes per element solve
constexpr Index_type dag_gpw = 32 / dag_lpe;   // groups per warp, i.e. per work item
constexpr size_t dag_block_size = FemsweepDag::block_size;

using atomic_int  = cuda::atomic_ref< int, cuda::thread_scope_device >;
using atomic_uint = cuda::atomic_ref< unsigned, cuda::thread_scope_device >;

}

// warp-wide: solve element e = ae % ne of angle a = ae / ne for the groups of
// group octet gq. Every lane takes part, so the shuffles use the full warp
// mask: with each element's own lane mask (known only at run time) the
// compiler wraps every shuffle in a MATCH/REDUX/VOTEU/BRA sequence, which
// made daggraph 30% slower at 32^3. Groups past ng redo group ng - 1 without
// storing it. group_inner: X and sigma laid out group innermost, see
// femsweepSolveElement.
template < bool group_inner >
__device__ __forceinline__
void femsweepDagItem(const FemsweepData & d,
                     const Index_type ae,
                     const Index_type gq,
                     Real_type (*face_b)[femsweep_face_b_stride<dag_lpe>]) {
  const Index_type lane = threadIdx.x % dag_lpe;
  const Index_type grp  = threadIdx.x / dag_lpe;
  const Index_type g    = gq * dag_gpw + grp % dag_gpw;
  femsweepSolveElement<dag_lpe, group_inner>(d, ae / d.ne, ae % d.ne, g < d.ng ? g : d.ng - 1,
                                             lane, ~0u, face_b[grp], g < d.ng);
}

// one hyperplane: items[w / nq] for group octet w % nq, one warp each
template < bool group_inner >
__launch_bounds__(dag_block_size)
__global__ void FEMSweep3DDagLevel(const FemsweepData d,
                                   const int * items,
                                   const Index_type nwork,
                                   const Index_type nq) {
  __shared__ Real_type face_b[dag_block_size / dag_lpe][femsweep_face_b_stride<dag_lpe>];

  const Index_type w = (Index_type(blockIdx.x) * dag_block_size + threadIdx.x) / 32;
  if (w < nwork) {
    femsweepDagItem<group_inner>(d, items[w / nq], w % nq, face_b);
  }
}

// The persistent queue kernel.
//
// Queue: ready items, item = a * ne + e. Every item is pushed exactly once
// per sweep, so the queue is a plain array of n slots with no wraparound:
// ends[0] counts slots taken, ends[1] slots pushed. A slot is -1 until its
// item is written, so a warp that takes a slot ahead of the pushes waits on
// that slot alone.
//
// No deadlock: a waiting warp holds no item, and the items in pushed slots
// are all held by running warps, whose completion pushes whatever they make
// ready (the graph is acyclic).
//
// Memory ordering: a warp's X writes, __syncwarp, then its lanes' acq_rel
// increments of the downwind counts. The increment that completes a count
// has acquired every upwind release and passes them on with the release
// store of the queue slot; the consumer's acquire load of the slot then
// makes all the upwind X it reads visible.
//
// Occupancy: 3 blocks per SM. Past 80 registers only 2 fit, which costs
// ~25%; capping at 64 for 4 spills and is slower too.
template < bool group_inner >
__launch_bounds__(dag_block_size, 3)
__global__ void FEMSweep3DDagQueue(const FemsweepData d,
                                   const Index_type nq,
                                   const int * indeg,
                                   int * count,
                                   int * queue,
                                   unsigned * ends) {
  __shared__ Real_type face_b[dag_block_size / dag_lpe][femsweep_face_b_stride<dag_lpe>];

  const Index_type n = d.na * d.ne;
  const unsigned lane = threadIdx.x % 32;
  atomic_uint taken(ends[0]);
  atomic_uint pushed(ends[1]);

  while (true) {
    Index_type slot = 0;
    int item = 0;
    if (lane == 0) {
      slot = taken.fetch_add(1, cuda::memory_order_relaxed);
      if (slot < n) {
        atomic_int q(queue[slot]);
        while ((item = q.load(cuda::memory_order_acquire)) < 0) {
          __nanosleep(100);
        }
        // reset for the next sweep: nothing else touches either this sweep
        q.store(-1, cuda::memory_order_relaxed);
        count[item] = 0;
      }
    }
    slot = __shfl_sync(~0u, slot, 0);
    if (slot >= n) {
      return;
    }
    item = __shfl_sync(~0u, item, 0);
    __syncwarp();   // lane 0's acquire before every lane's upwind X reads

    const Index_type ae = item;

    // one lane per local face; outgoing interior faces lead downwind. Looked
    // up before the solve, so these loads overlap it.
    int down = -1;
    if (lane < NLF) {
      const int4 * entry = reinterpret_cast<const int4 *>(d.faces) + 2 * ((ae % d.ne) * NLF + lane);
      if ((entry[0].x >= 0) && (d.AngleElem2FaceType[NLF * ae + lane] == 0)) {
        down = int(ae - ae % d.ne + entry[1].x / ND);   // same angle, element across the face
      }
    }

    for (Index_type gq = 0; gq < nq; ++gq) {
      femsweepDagItem<group_inner>(d, ae, gq, face_b);
    }

    __syncwarp();   // every lane's X writes before the releases below

    if (down >= 0) {
      if (atomic_int(count[down]).fetch_add(1, cuda::memory_order_acq_rel) + 1 == indeg[down]) {
        atomic_int(queue[pushed.fetch_add(1, cuda::memory_order_relaxed)]).store(down, cuda::memory_order_release);
      }
    }
  }
}

__global__ void FEMSweepDagSeed(int * queue,
                                unsigned * ends,
                                const int * seeds,
                                const Index_type nseeds) {
  const Index_type t = Index_type(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t < nseeds) {
    queue[t] = seeds[t];
  }
  if (t == 0) {
    ends[0] = 0;
    ends[1] = unsigned(nseeds);
  }
}

// upwind (incoming interior) faces of element e = ae % ne for angle ae / ne
__global__ void FEMSweepDagIndeg(int * indeg, const FemsweepData d) {
  const Index_type ae = Index_type(blockIdx.x) * blockDim.x + threadIdx.x;
  if (ae >= d.na * d.ne) return;
  const Index_type e = ae % d.ne;
  int n = 0;
  for (Index_type face = 0; face < NLF; ++face) {
    n += (d.AngleElem2FaceType[NLF * ae + face] != 0) && (d.F_g2l[d.elem_to_faces[NLF * e + face]] >= 0);
  }
  indeg[ae] = n;
}

// sigma [g][e] -> [e][g], t = e * ng + g
__global__ void FEMSweepDagTransposeSg(Real_ptr Sgt, const Real_ptr Sg, const Index_type ne, const Index_type ng) {
  const Index_type t = Index_type(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= ne * ng) return;
  Sgt[t] = Sg[t / ng + (t % ng) * ne];
}

// X [a][e][dof][g] -> [a][g][e][dof], t = a * ne * ND + e * ND + dof: reads
// one dof's ng groups, writes ng rows (coalesced across t)
__global__ void FEMSweepDagUnpackX(Real_ptr X, const Real_ptr Xg, const Index_type ne, const Index_type ng, const Index_type na) {
  const Index_type t = Index_type(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= na * ne * ND) return;
  const Index_type a   = t / (ne * ND);
  const Index_type dof = t % (ne * ND);
  for (Index_type g = 0; g < ng; ++g) {
    X[dof + g * ND * ne + a * ng * ND * ne] = Xg[t * ng + g];
  }
}

template < typename T >
static T * dagAlloc(Size_type count) {
  return static_cast<T *>(detail::allocCudaDeviceData(count * sizeof(T)));
}

template < typename T >
static void dagCopy(T * dst, const T * src, Size_type count, cudaMemcpyKind kind, cudaStream_t stream) {
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMemcpyAsync, dst, src, count * sizeof(T), kind, stream);
}

// ponytail: items are 32-bit, so na * ne must stay below 2^31; widen items
// and queue past that.
FemsweepDag::FemsweepDag(const FemsweepData & d_in, cudaStream_t stream)
  : d(d_in), nq((d_in.ng + dag_gpw - 1) / dag_gpw)
{
  const Index_type na = d.na;
  const Index_type ne = d.ne;
  d.faces = femsweepFaceTable(d, stream);

  //////// every angle's elements of hyperplane h, as a * ne + e, h = 0, 1, ... ////////

  std::vector<Index_type> nhp(na), ohp(na), order(na * ne);
  dagCopy(nhp.data(), d.nhpaa_r, na, cudaMemcpyDeviceToHost, stream);
  dagCopy(ohp.data(), d.ohpaa_r, na, cudaMemcpyDeviceToHost, stream);
  dagCopy(order.data(), d.order_r, na * ne, cudaMemcpyDeviceToHost, stream);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaStreamSynchronize, stream);

  Index_type nlevels = 0, php_count = 0;
  for (Index_type a = 0; a < na; ++a) {
    nlevels   = std::max(nlevels, nhp[a]);
    php_count = std::max(php_count, ohp[a] + nhp[a]);
  }
  std::vector<Index_type> php(php_count);
  dagCopy(php.data(), d.phpaa_r, php_count, cudaMemcpyDeviceToHost, stream);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaStreamSynchronize, stream);

  std::vector<int> h_items;
  std::vector<Index_type> pos(na, 0);   // per angle, elements in earlier hyperplanes
  level_begin = {0};
  for (Index_type h = 0; h < nlevels; ++h) {
    for (Index_type a = 0; a < na; ++a) {
      if (h >= nhp[a]) continue;
      const Index_type nehp = php[ohp[a] + h];
      for (Index_type k = 0; k < nehp; ++k) {
        h_items.push_back(int(a * ne + order[a * ne + pos[a] + k]));
      }
      pos[a] += nehp;
    }
    level_begin.push_back(Index_type(h_items.size()));
  }

  items = dagAlloc<int>(h_items.size());
  dagCopy(items, h_items.data(), h_items.size(), cudaMemcpyHostToDevice, stream);

  //////// queue state; count and queue reset themselves during each sweep ////////

  const Size_type n = Size_type(na) * ne;
  indeg = dagAlloc<int>(n);
  count = dagAlloc<int>(n);
  queue = dagAlloc<int>(n);
  ends  = dagAlloc<unsigned>(2);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMemsetAsync, count, 0, n * sizeof(int), stream);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMemsetAsync, queue, 0xff, n * sizeof(int), stream);   // -1: empty
  RPlaunchCudaKernel(FEMSweepDagIndeg, dim3(unsigned((na * ne + 255) / 256)), 256, 0, stream, indeg, d);

  // persistent: as many blocks as fit on the GPU at once
  int device, sms, blocks_per_sm;
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaGetDevice, &device);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceGetAttribute, &sms, cudaDevAttrMultiProcessorCount, device);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor,
                                 &blocks_per_sm, FEMSweep3DDagQueue<false>, int(dag_block_size), 0);
  queue_blocks[0] = sms * blocks_per_sm;
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor,
                                 &blocks_per_sm, FEMSweep3DDagQueue<true>, int(dag_block_size), 0);
  queue_blocks[1] = sms * blocks_per_sm;

  //////// group-innermost X and sigma ////////

  dg = d;
  dg.Xdat  = dagAlloc<Real_type>(d.Xlen);
  dg.Sgdat = dagAlloc<Real_type>(ne * d.ng);
  RPlaunchCudaKernel(FEMSweepDagTransposeSg, dim3(unsigned((ne * d.ng + 255) / 256)), 256, 0, stream,
                     dg.Sgdat, (const Real_ptr)d.Sgdat, ne, d.ng);

  //////// the level launches, captured once into a graph per X layout ////////

  for (bool group_inner : {false, true}) {
    cudaStream_t capture;
    cudaGraph_t g;
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaStreamCreateWithFlags, &capture, cudaStreamNonBlocking);
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaStreamBeginCapture, capture, cudaStreamCaptureModeThreadLocal);
    launchLevels(group_inner, capture);
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaStreamEndCapture, capture, &g);
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaGraphInstantiate, &graph[group_inner], g, 0);
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaGraphDestroy, g);
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaStreamDestroy, capture);
  }

  // h_items is about to go out of scope
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaStreamSynchronize, stream);
}

FemsweepDag::~FemsweepDag()
{
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaGraphExecDestroy, graph[0]);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaGraphExecDestroy, graph[1]);
  detail::deallocCudaDeviceData(dg.Sgdat);
  detail::deallocCudaDeviceData(dg.Xdat);
  detail::deallocCudaDeviceData(ends);
  detail::deallocCudaDeviceData(queue);
  detail::deallocCudaDeviceData(count);
  detail::deallocCudaDeviceData(indeg);
  detail::deallocCudaDeviceData(items);
  detail::deallocCudaDeviceData(const_cast<int *>(d.faces));
}

void FemsweepDag::launchLevels(bool group_inner, cudaStream_t stream)
{
  constexpr Index_type warps_per_block = dag_block_size / 32;
  for (Size_type h = 0; h + 1 < level_begin.size(); ++h) {
    const Index_type nwork = (level_begin[h + 1] - level_begin[h]) * nq;
    RPlaunchCudaKernel(group_inner ? FEMSweep3DDagLevel<true> : FEMSweep3DDagLevel<false>,
                       dim3(unsigned((nwork + warps_per_block - 1) / warps_per_block)), dag_block_size,
                       0, stream, group_inner ? dg : d, (const int *)(items + level_begin[h]), nwork, nq);
  }
}

void FemsweepDag::sweep(FemsweepDagMode mode, bool group_inner, cudaStream_t stream)
{
  switch (mode) {
    case FemsweepDagMode::levels:
      launchLevels(group_inner, stream);
      break;
    case FemsweepDagMode::graph:
      CAMP_CUDA_API_INVOKE_AND_CHECK(cudaGraphLaunch, graph[group_inner], stream);
      break;
    case FemsweepDagMode::queue:
      // hyperplane 0, the elements with no upwind faces
      RPlaunchCudaKernel(FEMSweepDagSeed, dim3(unsigned((level_begin[1] + 255) / 256)), 256, 0, stream,
                         queue, ends, (const int *)items, level_begin[1]);
      RPlaunchCudaKernel(group_inner ? FEMSweep3DDagQueue<true> : FEMSweep3DDagQueue<false>,
                         dim3(unsigned(queue_blocks[group_inner])), dag_block_size, 0, stream,
                         group_inner ? dg : d, nq, (const int *)indeg, count, queue, ends);
      break;
  }
}

void FemsweepDag::unpack(cudaStream_t stream)
{
  RPlaunchCudaKernel(FEMSweepDagUnpackX, dim3(unsigned((d.na * d.ne * ND + 255) / 256)), 256, 0, stream,
                     d.Xdat, (const Real_ptr)dg.Xdat, d.ne, d.ng, d.na);
}

void FemsweepDag::clearGroupInner(cudaStream_t stream)
{
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMemsetAsync, dg.Xdat, 0, Size_type(d.Xlen) * sizeof(Real_type), stream);
}

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_CUDA
