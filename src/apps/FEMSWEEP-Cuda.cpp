//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FEMSWEEP-CudaLaunch.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

template < size_t block_size, bool unroll >
__launch_bounds__(block_size)
__global__ void FEMSweep3D( const Real_ptr Bdat,
                            const Real_ptr Adat,
                            const Real_ptr Fdat,
                            Real_ptr Xdat,
                            const Real_ptr Sgdat,
                            const Real_ptr M0dat,
                            const Index_type ne,
                            const Index_type ng,
                            const Index_type sharedinteriorfaces,
                            const Index_ptr nhpaa_r,
                            const Index_ptr ohpaa_r,
                            const Index_ptr phpaa_r,
                            const Index_ptr order_r,
                            const Index_ptr AngleElem2FaceType,
                            const Index_ptr elem_to_faces,
                            const Index_ptr F_g2l,
                            const Index_ptr idx1,
                            const Index_ptr idx2 )
{
  const Index_type a = blockIdx.y;
  const Index_type g = blockIdx.x;
  FEMSWEEP_KERNEL_SETUP;
  Index_type nehp_pos = 0;
  for (Index_type hp = 0; hp < nhp; ++hp)
  {
    const Index_type nehp = phpaa_r[ohp + hp];
    for (Index_type k = threadIdx.x; k < nehp; k += block_size)
    {
      femsweepHyperplaneElement<unroll>(
          Bdat, Adat, Fdat, Xdat, Sgdat, M0dat,
          ne, ng, sharedinteriorfaces, order_r,
          AngleElem2FaceType, elem_to_faces, F_g2l, idx1, idx2,
          a, g, k, nehp_pos, Ffactor);
    }
    __syncthreads();
    nehp_pos += nehp;
  }
}

// One sweep with the reference kernel, for femsweep-bench (the tunings below
// launch it themselves).
template < size_t block_size, bool unroll >
void femsweepLaunchOriginal(const FemsweepData & d, cudaStream_t stream)
{
  RPlaunchCudaKernel( (FEMSweep3D<block_size, unroll>),
                      dim3(d.ng, d.na), block_size,
                      0, stream,
                      d.Bdat, d.Adat, d.Fdat, d.Xdat, d.Sgdat, d.M0dat,
                      d.ne, d.ng, d.sharedinteriorfaces,
                      d.nhpaa_r, d.ohpaa_r, d.phpaa_r, d.order_r,
                      d.AngleElem2FaceType, d.elem_to_faces, d.F_g2l, d.idx1, d.idx2 );
}

template void femsweepLaunchOriginal<64, false>(const FemsweepData &, cudaStream_t);
template void femsweepLaunchOriginal<64, true>(const FemsweepData &, cudaStream_t);

// One sweep with the RAJA_CUDA variant's RAJA::launch, for its tunings and
// femsweep-bench.
template < size_t block_size, bool unroll, size_t min_blocks_per_sm >
void femsweepLaunchRaja(const FemsweepData & d, cudaStream_t stream)
{
  const Real_ptr Bdat  = d.Bdat;
  const Real_ptr Adat  = d.Adat;
  const Real_ptr Fdat  = d.Fdat;
  const Real_ptr Xdat  = d.Xdat;
  const Real_ptr Sgdat = d.Sgdat;
  const Real_ptr M0dat = d.M0dat;
  const Index_type ne  = d.ne;
  const Index_type na  = d.na;
  const Index_type ng  = d.ng;
  const Index_type sharedinteriorfaces = d.sharedinteriorfaces;
  const Index_ptr nhpaa_r            = d.nhpaa_r;
  const Index_ptr ohpaa_r            = d.ohpaa_r;
  const Index_ptr phpaa_r            = d.phpaa_r;
  const Index_ptr order_r            = d.order_r;
  const Index_ptr AngleElem2FaceType = d.AngleElem2FaceType;
  const Index_ptr elem_to_faces      = d.elem_to_faces;
  const Index_ptr F_g2l              = d.F_g2l;
  const Index_ptr idx1               = d.idx1;
  const Index_ptr idx2               = d.idx2;

  constexpr bool async = true;

  using launch_policy =
      RAJA::LaunchPolicy<RAJA::cuda_launch_explicit_t<async, block_size, min_blocks_per_sm>>;

  using outer_y =
      RAJA::LoopPolicy<RAJA::cuda_block_y_direct_unchecked>;

  using outer_x =
      RAJA::LoopPolicy<RAJA::cuda_block_x_direct_unchecked>;

  using inner_x =
      RAJA::LoopPolicy<RAJA::cuda_thread_size_x_loop<block_size>>;

  auto res = camp::resources::Cuda::CudaFromStream(stream);

  RAJA::launch<launch_policy>( res,
      RAJA::LaunchParams(RAJA::Teams(ng, na),
                         RAJA::Threads(block_size)),
      [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
    RAJA::loop<outer_y>(ctx, RAJA::RangeSegment(0, na),
        [&](Index_type a) {
      RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, ng),
          [&](Index_type g) {
        FEMSWEEP_KERNEL_SETUP;
        Index_type nehp_pos = 0;
        for (Index_type hp = 0; hp < nhp; ++hp)
        {
          const Index_type nehp = phpaa_r[ohp + hp];
          RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, nehp),
              [&](Index_type k) {
            femsweepHyperplaneElement<unroll>(
                Bdat, Adat, Fdat, Xdat, Sgdat, M0dat,
                ne, ng, sharedinteriorfaces, order_r,
                AngleElem2FaceType, elem_to_faces, F_g2l, idx1, idx2,
                a, g, k, nehp_pos, Ffactor);
          });
          ctx.teamSync();
          nehp_pos += nehp;
        }
      });
    });
  });
}

// RAJA_CUDA's fastest tuning, unroll_64_min_blocks_per_sm_5
template void femsweepLaunchRaja<64, true, 5>(const FemsweepData &, cudaStream_t);

template < size_t block_size, bool unroll, size_t min_blocks_per_sm >
void FEMSWEEP::runCudaVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getCudaResource()};

  FEMSWEEP_DATA_SETUP;

  switch ( vid ) {

    case Base_CUDA : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

         RP_CALI_SUBKERNEL_BEGIN("FEMSWEEP_1");
         const dim3 grid_size(ng, na);
         constexpr size_t shmem = 0;

         RPlaunchCudaKernel( (FEMSweep3D<block_size, unroll>),
                             grid_size, block_size,
                             shmem, res.get_stream(),
                             Bdat,
                             Adat,
                             Fdat,
                             Xdat,
                             Sgdat,
                             M0dat,
                             ne,
                             ng,
                             sharedinteriorfaces,
                             nhpaa_r,
                             ohpaa_r,
                             phpaa_r,
                             order_r,
                             AngleElem2FaceType,
                             elem_to_faces,
                             F_g2l,
                             idx1,
                             idx2 );
         RP_CALI_SUBKERNEL_END("FEMSWEEP_1");

      }
      stopTimer();

      break;
    }

    case RAJA_CUDA : {

      const FemsweepData d = data();

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

         RP_CALI_SUBKERNEL_BEGIN("FEMSWEEP_1");
         femsweepLaunchRaja<block_size, unroll, min_blocks_per_sm>(d, res.get_stream());
         RP_CALI_SUBKERNEL_END("FEMSWEEP_1");

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n FEMSWEEP : Unknown CUDA variant id = " << vid << std::endl;
    }

  }

}

void FEMSWEEP::defineCudaVariantTunings()
{
  defineCudaExperimentalTunings();

  for (VariantID vid : {Base_CUDA, RAJA_CUDA}) {
    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {
        if (block_size == 0u) {
          addVariantTuning<&FEMSWEEP::runCudaVariantImpl<
              block_size, false, RAJA::policy::cuda::MIN_BLOCKS_PER_SM>>(
              vid, "block_auto");
        } else {
          addVariantTuning<&FEMSWEEP::runCudaVariantImpl<
              block_size, false, RAJA::policy::cuda::MIN_BLOCKS_PER_SM>>(
              vid, "block_" + std::to_string(block_size));
        }

        addVariantTuning<&FEMSWEEP::runCudaVariantImpl<
            block_size, true, RAJA::policy::cuda::MIN_BLOCKS_PER_SM>>(
            vid, std::string(FEMSWEEP_UNROLL_TUNING_NAME) + "_" +
            std::to_string(block_size));

      }
    });

    // Optimal unrolled RAJA CUDA tuning with a 5-block-per-SM launch bound, and 64
    // threads per block. These tuning parameters were determined empirically on 
    // the H100, reducing per-thread register use and increasing occupancy.
    if (vid == RAJA_CUDA) {
      constexpr size_t unroll_min_blocks_per_sm_tuning = 5;
      constexpr size_t unroll_block_size_tuning = 64; 
      addVariantTuning<&FEMSWEEP::runCudaVariantImpl<
          unroll_block_size_tuning, true, unroll_min_blocks_per_sm_tuning>>(
          vid, std::string(FEMSWEEP_UNROLL_TUNING_NAME) + "_" +
          std::to_string(unroll_block_size_tuning) + "_min_blocks_per_sm_" +
          std::to_string(unroll_min_blocks_per_sm_tuning));
    }
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_CUDA
