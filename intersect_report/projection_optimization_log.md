# Projection kernels: optimization log

A copy of tet_intersection's `docs/projection_optimization_log.md`. Unless they are links, file names below are
tet_intersection's.

A record of every performance idea tried on the GPU kernels of the direct (projection) method in
`intersect_hex_hex.cu`, `intersect_hex_rect.cu`,
`gpu_common.cuh` and [`projection.hpp`](../src/tet/projection.hpp) (the
kernels were all in `gpu_intersect.cu` up to commit 3333fd1), kept or not, so that none is tried twice. The method itself is described in
[`projection_moments.md`](projection_moments.md).

## How things are measured

- **GPU:** an A100 (PCIe, 40 GB) with its SM clock locked at 1200 MHz, profiled with `ncu
  --clock-control none` so that the clock stays there.
- **Time:** a kernel's own GPU time, `gpu__time_duration.sum` from ncu, the least of the 10 launches
  a benchmark makes. The benchmarks' event timers include the time a busy host spends launching, and
  this machine is shared: they swung by up to 3x during this work.
- **Benchmarks**, with the kernel each one times:

  | name | command | kernel |
  |---|---|---|
  | subzones | `raja_comparison --hexhex --jitter j` (125,000 pairs of 12-triangle hexes) | `hex_projection_moments_multi` |
  | hex | `mesh_compare --hex 20 17` (44,469 pairs of 24-triangle hexes) | `hex_projection_moments` |
  | cart | `mesh_compare --cart 40 60` (837,070 hex-cell pairs) | `cell_projection_moments` |
  | hexrect | `raja_comparison --hexrect --jitter j` (1M hex-cell pairs, RAJAPerf's INTSC_HEXRECT) | `cell_projection_moments` |

- **Correctness**, for every change kept: `hex_tests`; `mesh_compare --hex` and `--cart` (GPU and
  CPU agree); `raja_comparison` at jitters 0, 0.3, 0.45 (all rows pass). (Up to commit 570c3d9 the
  library also had tet-tet overlaps and the clipping, closed-form and bilinear methods, and these
  checks included the tet tests and agreement with clipping.)
- **Profiles** are in `profiles/` (not in git).

## Current times (ms)

| commit | subzones j=0 | j=0.3 | j=0.45 | hex | cart | hexrect | hexrect j=0.3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| e078a8c | 0.634 | 1.635 | 1.686 | 2.142 | 3.610 | 2.347 | 4.802 |
| 357177f | 0.602 | 1.540 | 1.595 | 1.983 | 3.308 | 2.176 | 4.389 |
| a5af1c7 | 0.562 | 1.410 | 1.455 | 1.874 | 3.270 | 2.156 | 4.346 |
| 4e47a51 | 0.563 | 1.299 | 1.353 | 1.873 | 3.271 | 2.156 | 4.348 |
| 703e09d | 0.529 | 1.260 | 1.317 | 1.877 | 3.266 | 2.156 | 4.349 |
| aa21805 | 0.433 | 1.252 | 1.315 | 1.870 | 3.270 | 2.155 | 4.346 |
| 279a377 | 0.401 | 1.213 | 1.276 | 1.870 | 3.174 | 2.083 | 4.231 |
| 3333fd1 | 0.389 | 1.129 | 1.191 | 1.633 | 2.911 | 1.949 | 3.888 |

RAJAPerf's own kernels, on the same data: 2.8 ms (subzones, any jitter), 4.8 ms (hexrect).

## Kept

Times are kernel times before -> after, in the order the changes were made.

| change | effect | commit |
|---|---|---|
| Polygon scratch in shared memory instead of local memory | hex 7.5 -> 3.4 ms, cart 14.8 -> 6.0 ms (before the clock was locked) | 229a759 and earlier |
| Polygon vertices carry only (y, z); heights come from the triangle's plane | hex 3.0 -> 2.6, cart 6.0 -> 5.0 | earlier |
| Divide by constants as multiplications by their inverses (`* (1.0 / 6)`) | hexrect 4.6 -> 4.5 (with the next) | earlier |
| Triangles parallel to x never queued (box trick: `b.hi.x = -INFINITY`) | hexrect 4.6 -> 3.4 | earlier |
| 12-triangle subzone split (`HexSplit::diagonal`), as RAJAPerf's INTSC_HEXHEX | subzones j=0.3 5.15 -> 2.50 (half the items) | a94472e |
| Clip polygons in place, one buffer per lane (unread vertices shift up when rounding makes more than 2 edges cross) | 15-17% everywhere: subzones 1.48 -> 1.23 (j=0), 2.49 -> 2.05 (j=0.3); hex 2.87 -> 2.38; cart 4.63 -> 3.94; hexrect 3.31 -> 2.80. Warps per SM 11 -> 17, registers 108 -> 96 | a94472e |
| 2 subzone pairs per warp, one queue, sums per pair by segmented scan (`hex_projection_moments_multi`, `add_by_key`) | subzones 1.23 -> 0.76 (j=0), 2.05 -> 2.04 (j=0.3) | a94472e |
| Plane slopes once per triangle in setup (`add_projected_pair` overload taking them), hex volumes summed with shuffles, `plane_x` with one reciprocal, `parallel_to_x` comparing squares | subzones 0.757 -> 0.686, 2.036 -> 1.924; hex 2.37 -> 2.28; hexrect 2.80 -> 2.56 | 9cf4396 |
| `__drcp_rn` reciprocal at edge crossings (device only) | subzones 0.686 -> 0.667, 1.924 -> 1.839; hex 2.28 -> 2.18; cart 3.87 -> 3.71 | 9cf4396 |
| Overrun check only where an edge writes 2 vertices | subzones 0.667 -> 0.636, 1.839 -> 1.758; hexrect 2.52 -> 2.35 | 9cf4396 |
| Reciprocal at crossings as `rcp.approx.ftz.f64` and 3 FMAs (`detail::crossing`), no final rounding correction | subzones 0.634 -> 0.624, 1.631 -> 1.590; hex 2.14 -> 2.07; cart 3.61 -> 3.46; hexrect 2.35 -> 2.30, 4.81 -> 4.60 (j=0.3) | 357177f |
| First clip of a triangle from registers (`clip_proj` overload for 3 vertices): unrolled, independent distances; for S's first edge, and for the box case's first line | subzones 0.624 -> 0.602, 1.590 -> 1.540; hex 2.07 -> 1.98; cart 3.46 -> 3.31; hexrect 2.30 -> 2.18, 4.60 -> 4.39 | 357177f |
| Filter test: orientation folded into the edges as +-1 (exact), x_T - x_S at the corners as one linear function from T's corner | subzones 1.544 -> 1.511 (j=0.3), 1.593 -> 1.567 (j=0.45), j=0 unchanged; same items dropped | a5af1c7 |
| Box tests on floats rounded outward (`outward_floats`), all three kernels: off the FP64 pipe, half the shared-memory reads | subzones 0.601 -> 0.564, 1.511 -> 1.411, 1.567 -> 1.455; hex 1.98 -> 1.87; cart 3.31 -> 3.27; hexrect 2.18 -> 2.16 | a5af1c7 |
| Filter's separating-axis test in floats (`separated_yz`): corners relative to o as floats, set up only when the warp filters; decisions only beyond a margin of 2^-18 L^2 (above the rounding error) | subzones 1.411 -> 1.332 (j=0.3), 1.462 -> 1.368 (j=0.45) | 4e47a51 |
| ... and the corner-sign test (T below S, doubles) left out of the kernel's filter | 1.332 -> 1.298, 1.368 -> 1.354 | 4e47a51 |
| Subzone kernel: boxes and float corners in a union with the polygon scratch (dead before clipping): 8.5 -> 7.1 KB per block, 17 -> 20 blocks per SM | subzones 0.562 -> 0.528, 1.297 -> 1.260, 1.352 -> 1.317 | 703e09d |
| 4 subzone pairs per warp (16 blocks per SM) instead of 2 (20), now that the polygons share memory with the boxes | subzones 0.529 -> 0.434, 1.260 -> 1.255, 1.317 -> 1.322 (P = 3: 0.478, 1.257, 1.314) | aa21805 |
| Box-test loops (subzone and hex-cell kernels): comparisons combined with `&` rather than `&&`, item indices advanced per round rather than divided out | subzones 0.434 -> 0.402, 1.256 -> 1.211, 1.318 -> 1.277; cart 3.27 -> 3.16; hexrect 2.16 -> 2.08, 4.35 -> 4.24 | 279a377 |
| 24-triangle hex kernel: branch-free box test | hex 1.875 -> 1.771 | 3333fd1 |
| The plane cut (x_T - x_S) as one linear function from T's corner (alone; before, only tried with hoisted edge directions) | subzones 1.215 -> 1.196, 1.283 -> 1.260; hex 1.771 -> 1.749 | 3333fd1 |
| Clip loop with its last edge (back to the first vertex) peeled: no wrap-around select in the loop | subzones 0.402 -> 0.389, 1.197 -> 1.129, 1.259 -> 1.191; hex 1.745 -> 1.633; cart 3.16 -> 2.91; hexrect 2.08 -> 1.95, 4.23 -> 3.89 | 3333fd1 |
| Filter pass: drop queued items `projected_pair_empty` finds add nothing (2D separating axis, T below S at corners), branch-free, only when the queue is longer than a round | subzones 1.759 -> 1.635 (j=0.3), 1.882 -> 1.686 (j=0.45), j=0 unchanged | e078a8c |

## Tried and reverted

| idea | result | why |
|---|---|---|
| Skip clips that a triangle's vertices already decide | no faster | added divergence |
| Merge the last clip into the integration | slower | doubled the FP64 work |
| Edge crossings computed unconditionally (branch-free) | no faster | (old two-buffer kernel; not retried since) |
| Two-pass integration: clip, leave polygons in shared memory, then the warp integrates all fan triangles 32 at a time | hexrect 4.47 -> 4.94 ms | active lanes 12.8 -> 14.1, but 14% more instructions and a block fewer per SM |
| Exact triangle-rectangle test at queue time (cell kernel) | hexrect neutral (3.18, 6.12 ms); CPU cart 57 -> 61 ms | items it drops empty out early anyway |
| More subzone pairs per warp: 3, 4 | P=4: subzones 0.69 / 2.27 (j=0 / 0.3) vs P=2 0.76 / 2.04; P=3 0.68 / 2.12 | more shared memory per warp, fewer warps; P=2 is the best balance |
| Several pairs per warp for 24-triangle hexes | hex 2.38 -> 2.65 | those pairs already fill a warp, and rebuilding face-centre triangles costs |
| Hoist edge directions (`sign * (b - a)`) and fold the plane cut into one linear function | subzones 1.758 -> 1.799 (+2%) | 3% more instructions: the compiler had already folded those terms |
| Filter pass, first version: always on, early-exit tests | j=0 0.636 -> 0.713 (+13%), j=0.3 -> 1.691 | a lone round cannot get shorter; early exits diverge |
| Filter with only the separating-axis test (gated, branch-free) | 1.637 / 1.708 vs 1.635 / 1.686 for both tests | the corner test pays for itself at higher jitter |
| Filter with 4 pairs per warp | 0.637 / 1.873 / 1.936 | occupancy |
| Corners stored shifted by the local origin, cone volumes from the origin (no `Hex::center()` per triangle, no subtractions when rebuilding triangles) | 0.639 / 1.646 / 1.700 vs 0.637 / 1.634 / 1.685 | instructions -0.4% / +0.8%: that setup work was never on the critical path (it overlaps other warps' stalls) |
| Per-lane accumulators per pair, reduced once at the end, instead of `add_by_key` after every round | 0.679 / 1.640 / 1.694 | the final 4P-value butterfly costs more than one round's scan unjittered |
| Float corners and margins set up for every warp, the margins by 2 lanes serially | j=0 0.562 -> 0.589 | unjittered warps never filter; now only set up when filtering, margins by shuffles |
| With P = 4: the hexes' volumes in shared memory after summing, to free registers | registers stay 124, a block fewer per SM; 0.428 / 1.261 / 1.321 | the registers are taken elsewhere |
| ... and the box test with S's boxes in registers (24 lanes, 12 steps, T's boxes broadcast) | 0.666 / 1.683 / 1.741 | fewer instructions, but slower: T-major queue order, fewer full rounds |

## Candidates not yet tried

Latest profile (aa21805, before the last two commits): unjittered, the box test (22%), corner loads
(14%, global-load stalls) and setup (12%) are as big as the clipping (35%); at a jitter of 0.3 the
clipping is 58%, at about 18 of 32 lanes.

- Hide the corner loads' latency: persistent warps prefetching their next pairs (registers are at 124
  of 128 with 4 pairs per warp, so probably via `cp.async` into a second buffer of corners, which
  costs shared memory).

Lessons so far: only the clip loop's dependency chain, and how many lanes run it, set the time
while the FP64 pipe has room; work elsewhere overlaps other warps' stalls. Now that the subzone
kernel keeps the FP64 pipe 70% busy when jittered (math-pipe throttle stalls rising), FP64
instructions anywhere count: moving work to FP32 or integer pipes pays.

- Persistent warps that prefetch their next pairs (hides the corner loads, 10% unjittered).
- Cheaper filter test from per-triangle edge lines precomputed in setup (costs ~3.5 KB of shared
  memory per warp).
- Corner-sign test (T below S) in floats too, with margins from the planes' magnitudes: needs float
  plane data per triangle (~0.5 KB of shared memory, which would cost a block per SM).
- Occupancy with P = 4: 16 blocks per SM, limited by both registers (124) and shared memory (9.4 KB).
  20 would need <= 96 registers and <= 7.3 KB (e.g. polygon capacity 8 rather than 9, and queue
  indices or slopes packed tighter).
