# FEMSWEEP on the GPU: experimental CUDA kernels

This report describes the experimental CUDA implementations of the FEMSWEEP kernel that sit next to the original ones in `src/apps/`. It explains the main choices behind them, how they were measured and validated, and which ideas did not pay off.

**Bottom line:** the fastest variant, `daggraph_256`, solves exactly the same problem as the original kernel 3.7× to 4.7× faster than the RAJA_CUDA version on an A100. It gets there with two changes:

1. **Cooperative element solves.** 4 GPU threads ("lanes") solve each 8×8 element system together, instead of one thread solving it alone.
2. **One combined DAG.** The sweeps of all angles and energy groups are scheduled as a single dependency graph. The original runs one independent sweep per (angle, group).

Everything else is tuning, found by profiling one bottleneck at a time.

![Speedup over the original RAJA_CUDA kernel](speedup.svg)

| ms per sweep (A100, min of 3) | 5³ mesh | 16³ mesh | 32³ mesh |
|---|---:|---:|---:|
| original `RAJA_CUDA` `unroll_64_min_blocks_per_sm_5` | 0.484 | 12.62 | 100.5 |
| original `Base_CUDA` `unroll_64` | 0.413 | 10.25 | 82.0 |
| `coop_256`: original schedule, cooperative solve | 0.254 | 4.02 | 31.0 |
| `dagqueue_256`: persistent kernel with a work queue | 0.208 | 3.12 | 24.7 |
| `daglevels_256`: one launch per combined hyperplane | 0.153 | 3.00 | 21.7 |
| **`daggraph_256`: the same launches, from a CUDA graph** | **0.130** | **2.91** | **21.5** |
| `daggraphgi_256`: X kept group-innermost, converted back each sweep | 0.163 | 3.06 | 22.6 |
| `daggraphgi_256` (sweep only): X left group-innermost ([§7](#7-memory-layout-of-x-group-innermost)) | 0.124 | 2.58 | 18.9 |

All variants agree with the original kernel to within 1e-15 relative error ([§9](#9-validation)).

## Contents

1. [Benchmark setup and how to run](#1-benchmark-setup-and-how-to-run)
2. [Where the original kernel spends its time](#2-where-the-original-kernel-spends-its-time)
3. [Cooperative element solve](#3-cooperative-element-solve)
4. [All angles and groups as one DAG](#4-all-angles-and-groups-as-one-dag)
5. [Three schedulers for the DAG](#5-three-schedulers-for-the-dag)
6. [Optimizations found by profiling](#6-optimizations-found-by-profiling)
7. [Memory layout of X: group-innermost](#7-memory-layout-of-x-group-innermost)
8. [What did not work](#8-what-did-not-work)
9. [Validation](#9-validation)
10. [Where the time goes now](#10-where-the-time-goes-now)
11. [Code map](#11-code-map)

---

## 1. Benchmark setup and how to run

- **Hardware:** NVIDIA A100-PCIE-40GB (108 SMs), CUDA 13.3, compiled for `sm_80`. GPU clocks were not locked.
- **Problem:** 72 angles (3 polar × 3 azimuthal × 8 octants), 16 energy groups, and three cube meshes:

  | Mesh | Elements | Hyperplanes per angle |
  |---|---:|---:|
  | 5³ | 125 | 13 |
  | 16³ | 4,096 | 46 |
  | 32³ | 32,768 | 94 |

- **The face coupling is real in this tree.** The original `Ffactor` expression always evaluated to 0. Here it is `femsweep_Ffactor = 0.1`, and the diagonal of A is shifted to keep the sweep bounded (see the comment in `FEMSWEEP.hpp`). So the face terms and the sweep order genuinely affect the results, and the checks below would catch an ordering error.
- **Timing:** each figure is the minimum over 3 runs of 10 back-to-back sweeps. The test machine had intermittent outside load, which occasionally made every variant slower in the same run.

Two ways to run the comparison:

```bash
# every CUDA implementation on the same input, checked against the original kernel
./build/bin/femsweep-bench.exe --femsweep_mesh_dims 32 32 32

# inside the suite, with the RAJA variant as the speedup reference
./build/bin/raja-perf.exe -k FEMSWEEP -v Base_CUDA RAJA_CUDA \
  -t unroll_64 unroll_64_min_blocks_per_sm_5 coop_256 daglevels_256 daggraph_256 dagqueue_256 \
  -rv RAJA_CUDA --femsweep_mesh_dims 32 32 32 -od femsweep-cmp
```

`femsweep-bench` prints a table sorted by time. `raja-perf` writes its CSV reports to `femsweep-cmp/`.

## 2. Where the original kernel spends its time

![The original schedule](original-schedule.svg)

The original CUDA kernel launches one thread block per (angle, group), 1,152 blocks in total. Each block walks its angle's hyperplanes in order. Each thread solves one element's whole 8×8 system, and a `__syncthreads()` separates consecutive hyperplanes.

On small meshes most threads have nothing to do. That was the starting motivation for this work, but at 32³ it turned out not to be the main cost. Nsight Compute on the unrolled kernel at 32³ showed:

| Measurement | Value | What it means |
|---|---:|---|
| L1 cache throughput | 91% of peak | L1 is the limiting unit |
| Global memory sectors per load request | 30 | each thread loads its own element's 8×8 rows, so a warp's loads are scattered across memory |
| Registers per thread | 168 | only 6 blocks of 64 threads fit per SM: 16.5% achieved occupancy |
| Cycles a warp waits per instruction issued | 50 | the GPU issues instructions in under 5% of cycles |
| Waves of blocks | 1.78 | the last partial wave leaves part of the GPU idle |

In short, the kernel is not limited by arithmetic. One thread per element means scattered memory access and a very large register footprint, and those, not idle threads, are the main costs.

## 3. Cooperative element solve

All experimental kernels share one element routine, `femsweepSolveElement` in [`FEMSWEEP-CudaElement.hpp`](../src/apps/FEMSWEEP-CudaElement.hpp). It runs four steps: gather the upwind face terms, build the rows of A + σ·M0, LU-factorize, then substitute.

### 3.1 Four lanes per element

![How a warp is divided among element solves](warp-mapping.svg)

A warp (32 lanes) is split into 8 groups of 4 lanes, and each group solves one element. Lane *l* of a group owns rows *l* and *l*+4: 16 matrix entries plus 2 entries of b, all held in registers. Row *kk* lives in lane *kk* % 4, slot *kk* / 4. Both are compile-time constants inside the unrolled loops, so this distribution costs nothing in indexing.

**Why 4 lanes.** The number of lanes per element trades registers per lane against shuffle work:

| Lanes per element | What happens | Measured (32³, at the start of this round) |
|---|---|---:|
| 1 (original) | 168 registers per thread, uncoalesced loads | 82.2 ms |
| 8 | 4 elements per warp; every shuffle serves only 4 elements | 39.4 ms (`coopconn_256`) |
| 4 | 8 elements per warp; the same number of shuffles serves twice as many elements | 36.4 ms (`cooprowsconn4_256`, now `coop_256`) |

With 2 lanes per element, each lane would hold 4 rows, and that version was slower again.

### 3.2 LU through shuffles

![One step of the cooperative LU](lu-step.svg)

The factorization is Doolittle without pivoting, which is safe here because `setUp` makes every element system strictly diagonally dominant.

- **Broadcast:** at step *kk*, the owner of row *kk* shares it with the other three lanes using `__shfl_sync`, a register-to-register exchange within the warp that never touches memory.
- **Eliminate:** each lane updates its own rows below *kk*.
- **Store the reciprocal:** the pivot's owner overwrites the diagonal entry with 1/pivot, since only the back substitution reads it again. The back substitution can then multiply instead of divide, which removes 8 divisions per element ([§6.2](#62-reuse-the-pivot-reciprocal)).
- **Cheaper reciprocal:** each reciprocal uses a hardware estimate plus two Newton steps (`femsweepRcp`) instead of a full IEEE division ([§6.5](#65-branch-free-reciprocal)).

### 3.3 Face terms and the face table

![Gathering the upwind face terms](face-gather.svg)

- **Face table.** The original face loop follows a chain of four dependent loads before it can touch F or X: `elem_to_faces → F_g2l → idx1 (which side) → idx1/idx2`. None of that depends on the angle or group. So `femsweepFaceTable` resolves it once at setup into one 32-byte entry per (element, face), and the face loop makes a single load instead. That change alone took the 4-lane kernel from 43.2 ms to 36.4 ms at 32³.
- **Computing the terms.** Lane *l* computes the term of face dof *j* = *l* on all 6 faces and writes it into a small shared-memory array, `face_b[face][element dof]`. Inactive faces (outgoing or boundary) leave their slots at zero.
- **Summing into b.** The owner of row *d* then sums column *d* into b. The sum runs in face order, as in the original, so these kernels produce bitwise-identical results to one another.

## 4. All angles and groups as one DAG

Every (angle, group) sweep has the same structure: each element depends on its upwind neighbors. Nothing forces each sweep to run in its own block, so the experimental schedulers treat all 1,152 sweeps as one dependency graph with 72 × ne nodes. Each node is one (angle, element), solved for all 16 groups.

![Work per combined hyperplane](levels.svg)

**The work item: one (angle, element) for 8 groups.** A warp solves one element of one angle for 8 consecutive energy groups. All 8 groups read the same A and B rows, M0 rows, F blocks and face-table entries, so a single load serves the whole warp. The original layout, where each block handles a different (angle, group), can't share those loads.

**Hyperplane width.** Combining all angles and groups makes each hyperplane wide: at 32³ the widest has 110,592 warps of work, against 3,456 warps the A100 can hold at once. Only 12 of the 94 combined hyperplanes are narrower than one full wave. An earlier profile of the per-hyperplane version measured the cost: at that point only 3 blocks fit per SM, so 10 hyperplanes were narrower than a wave. Those launches took 0.18 ms of a 32.9 ms sweep, and 96% of the time ran at the steady-state rate of the widest hyperplanes.

So narrow hyperplanes are not the main cost at 32³. The combined view wins mainly through shared loads and full warps. At 5³, where the original kernel leaves most threads idle, the combined view also removes that idle time.

## 5. Three schedulers for the DAG

![Three schedules](schedules.svg)

All three schedulers run the same element routine and differ only in how they enforce the dependencies.

### 5.1 `daglevels`: one launch per combined hyperplane

Hyperplane *h* of every angle becomes one kernel launch, with one warp per (angle, element, group octet). The boundary between kernel launches acts as the barrier. Level lists are built once at setup from `nhpaa_r`, `ohpaa_r`, `phpaa_r` and `order_r`.

### 5.2 `daggraph`: the same launches from a CUDA graph

The launch sequence is captured into a CUDA graph once at setup and replayed for each sweep. That removes about 1.7–1.9 µs of launch overhead per hyperplane:

| Mesh | Launches | Graph saves |
|---|---:|---:|
| 5³ | 13 | 0.02 ms (10%) |
| 16³ | 46 | 0.08 ms (1.7%) |
| 32³ | 94 | 0.2 ms (0.6%) |

These are back-to-back sweeps, so the CPU queues launches ahead of the GPU. An application that synchronizes after every sweep would likely gain more from the graph, but that was not measured.

### 5.3 `dagqueue`: a persistent kernel with a work queue

![The persistent work queue](work-queue.svg)

One launch, with as many blocks as fit on the GPU, processes the whole sweep. A warp works in a loop:

1. **Take** the next slot of a global queue.
2. **Wait** for an item to appear in that slot.
3. **Solve** that (angle, element) for all 16 groups.
4. **Release** its downwind neighbors: atomically increment each neighbor's count. When a count reaches the neighbor's number of upwind faces (`indeg`), push that neighbor onto the queue.

Design points:

- **Write-once queue.** Every (angle, element) is pushed exactly once per sweep, so the queue is a plain array of 72 × ne slots with no wraparound. It needs two counters: slots taken and slots pushed. An unwritten slot holds −1, so a warp that took a slot ahead of the pushes waits on that slot alone.
- **No deadlock.** A waiting warp holds no work, and every pushed item is held by a running warp. When that warp finishes, it pushes whatever its completion makes ready. The graph is acyclic, so everything is eventually pushed.
- **Memory ordering.** A warp writes X, then `__syncwarp`, then increments each downstream count with an acquire-release atomic (`cuda::atomic_ref`). The increment that completes a count has therefore acquired every upwind warp's writes, and it publishes them with a release store into the queue slot. The consuming warp loads the slot with acquire semantics, so every upwind X value it reads is visible to it.
- **State resets itself.** The warp that takes an item clears its slot and its count, so the next sweep needs no reset pass. A small seed kernel just writes hyperplane 0 into the queue.
- **Item size matters.** The first version queued (angle, element, group octet), and the two octets of an element usually ran on different SMs. That doubled L2 traffic for A, B, F and the face table, and the L1 hit rate fell from 71% to 43%. Queueing whole elements cut the queue's time by 17% (36.3 ms to 30.2 ms at 32³, at the time).

**Why the queue trails the graph.** The queue avoids all global barriers, but each (angle, element) pays for a slot claim, up to three atomic count updates and a push. That accounts for roughly 10–20% of the queue kernel's stall samples. The barriers it avoids are cheap at 32³: wide hyperplanes and 96% steady-state efficiency mean there is little idle time to recover. So `daggraph` wins at every size measured. The queue's advantage would grow on meshes or quadratures with less uniform, narrower hyperplanes.

## 6. Optimizations found by profiling

Each step below came from one Nsight Compute profile and was kept only if it won on the minimum of three runs.

![Time per sweep through the optimization steps](progress.svg)

| Step | coop | daggraph | dagqueue |
|---|---:|---:|---:|
| first DAG version (whole-element queue items) | 34.7 | 32.5 | 30.2 |
| 6.1 full-warp shuffle mask | 34.7 | 24.8 | 27.7 |
| 6.2 reuse the pivot reciprocal | 32.9 | 23.45 | 27.0 |
| 6.3 queue: look up downstream neighbors early | 32.9 | 23.45 | 26.1 |
| 6.4 conflict-free `face_b` padding | 31.4 | 22.3 | 25.4 |
| 6.5 branch-free reciprocal (final) | 31.0 | 21.5 | 24.7 |

All times are ms per sweep at 32³.

### 6.1 Full-warp shuffle mask

`__shfl_sync` takes a mask of the participating lanes. With one mask per group of 4, computed at run time, the compiler cannot prove the mask is uniform, so it wraps every shuffle in a `MATCH`/`REDUX`/`VOTEU`/`BRA` sequence. That came to about 8% of all instructions, and it kept the ADU pipe 48% busy.

In the DAG kernels, all 32 lanes always run the element routine together. Groups beyond the last energy group repeat the last group's solve without storing it. So the shuffles can use the constant full-warp mask. Result: `daggraph` 24% faster, `dagqueue` 9% faster.

`coop` keeps per-group masks, because its groups can run out of elements at different times.

### 6.2 Reuse the pivot reciprocal

Back substitution used to divide by each diagonal entry, which is 8 double-precision divisions per element, each with a slow-path branch. Storing 1/pivot in place of the diagonal during the LU turns those divisions into multiplies (see [§3.2](#32-lu-through-shuffles)).

This cost the queue kernel 2 extra registers. 82 registers round up to 88 in the hardware's allocation steps, which fits only 2 blocks per SM instead of 3, and the queue got 25% slower. `__launch_bounds__(256, 3)` brings it back to 80 registers.

### 6.3 Queue: look up downstream neighbors early

The loads that find an element's downwind neighbors (face table plus face type) now happen before the solve instead of after it, so their latency overlaps the solve: 3.5% faster.

### 6.4 Conflict-free `face_b` padding

![Shared memory banks](bank-conflicts.svg)

Shared memory is split into 32 banks, and two lanes hitting the same bank in one access are served one after the other (a "bank conflict").

Each group's slice of `face_b` was padded to 56 doubles, which is the right stride for 8 lanes per element. With 4 lanes, half a warp covers 4 groups. At 56 doubles, groups 0 and 2 (and 1 and 3) land on the same banks, so half of all shared-memory accesses were conflict replays.

The stride is now `48 + LPE` doubles, which puts the groups exactly `2 × LPE` banks apart for any lanes-per-element value. Bank conflicts fell from 5.98M to 0.45M per launch of the widest hyperplane, and `daggraph` got 4.9% faster. Results stay bitwise identical.

### 6.5 Branch-free reciprocal

`1.0 / pivot` compiles to a reciprocal plus refinement steps and a slow-path check, and it was still 12% of all instructions. `femsweepRcp` uses the hardware estimate `rcp.approx.ftz.f64` with two Newton steps and no check. The diagonal-dominance construction rules out zero, denormal or infinite pivots, so the check is never needed. The result is within an ulp of `1.0 / x`.

This made `daggraph` 3.5% faster. It also dropped the level kernel from 78 to 62 registers, so 4 blocks fit per SM instead of 3, raising achieved occupancy from 36% to 48%.

## 7. Memory layout of X: group-innermost

![X layouts](x-layout.svg)

In the DAG kernels, a warp's 8 groups read the same upwind dof for 8 different groups. In the original X layout, `[a][g][e][dof]`, those 8 values are 256 KB apart at 32³, so every gather instruction touched up to 8 cache lines. That gather made up 54% of all global memory traffic and was the top stall location.

The `*gi` variants keep X as `[a][e][dof][g]` during the sweep, with σ transposed the same way once at setup. Each gather then reads 64 contiguous bytes. Results at 32³:

| | ms per sweep |
|---|---:|
| `daggraph_256` (Xdat layout) | 21.5 |
| `daggraphgi_256` (sweep only, X left group-innermost) | **18.9** (−12%) |
| `daggraphgi_256` (plus converting X back each sweep) | 22.6 |

Converting back to the benchmark's X layout moves 4.8 GB and takes about 3.7 ms at 32³. That is already close to memory bandwidth, so it costs more than the faster gather saves. Writing both layouts during the sweep (scattered stores, plus extra registers that lose the fourth block per SM) was slower too.

**The group-innermost layout pays off only if the application keeps X that way between sweeps.** "(sweep only)" in the tables means exactly that. The full calculation runs, and only the final conversion is left out of the timing; the result is converted and checked after the timed sweeps.

## 8. What did not work

These were tried, measured, and reverted. They are listed so they don't need to be retried blindly.

| Idea | Result at 32³ | Why |
|---|---|---|
| queue items of (angle, element, group octet) | 36.3 vs 30.2 ms | an element's octets ran on different SMs: twice the L2 traffic, L1 hit rate 43% instead of 71% |
| load the A/M0 rows before the face gather, to overlap latency | 27.1 / 26.2 vs 23.5 ms | the extra loads queue in front of the gather, which is the critical path; the L1 data pipe was already ~80% busy |
| sum face terms in registers instead of `face_b` | 27.4 vs 23.5 ms | removed all shared memory, but working out which face dof lands on each row lengthened the load chain, and global loads rose 27% |
| cap registers at 64 for 4 blocks per SM (queue) | 32.8 vs 30.2 ms; later 25.4 vs 24.9 ms | spills to local memory cost more than the extra occupancy gains |
| explicit fences instead of acq_rel atomics (queue) | 25.99 vs 26.08 ms | within noise, and the ordering argument is more subtle |
| claim 2 queue slots per atomic (queue) | 26.4 vs 26.1 ms; 5³ 40% slower | a claimed item waits while its warp finishes another, which delays the critical path |
| write X in both layouts | 22.7 vs 21.5 ms | scattered extra stores, and 70 registers lose the fourth block per SM |

Earlier explorations, now deleted, are in git history:
- a shared-memory variant of the 1-thread-per-element kernel
- several 8-lane variants with different gather and F layouts
- the NakulFork kernels, including `pairpad`, a per-hyperplane launch with a group-innermost X

## 9. Validation

- **Reference.** Every variant is compared against the original unrolled kernel on the same input. It must agree to `max |x − x_ref| / max |x_ref| ≤ 1e-12`. The experimental kernels agree to about 1e-15: the reciprocal changes (§6.2, §6.5) alter last-bit rounding but nothing larger.
- **Catching ordering bugs.** The checked sweep runs after the timed sweeps, on X zeroed beforehand. So a missed dependency, or state that one sweep leaves behind for the next (the queue's self-resetting counters, for example), shows up as a wrong answer. The group-innermost variants also zero their internal X buffer first. Otherwise a missed dependency could silently read the previous sweep's correct values.
- **Coverage.** Tested on cube meshes (5³, 16³, 32³) and a 12×7×9 mesh, with 1, 5, 16 and 20 groups. Counts that are not a multiple of 8 leave the last group octet partly empty.
- **Inside the suite.** The raja-perf checksums, computed after several repetitions, match the `Base_Seq` reference for all experimental tunings.

## 10. Where the time goes now

The last full profiles, taken just before the padding and branch-free reciprocal changes, show:

- **Latency-bound.** The schedulers issue an instruction on only ~30% of cycles. A test that varied the number of resident warps showed time still falling steeply with more warps (8 → 16 → 24 warps per SM gave 66 → 40 → 30 ms for the queue at 32³).
- **The L1 data pipe is the next limit.** It was 76–84% busy. Shared memory (`face_b` plus shuffles) accounted for about a third of that, and the padding fix removed most of its conflict replays.
- **Top stall locations:**
  - the upwind X gather (~20%), a chain of dependent loads (face table entry, then X)
  - the LU's dependency chains: DFMA ~25%, shuffles ~7%
  - for the queue, its atomics and push (~10–20%)
- **Occupancy** is limited by registers: 62 per thread for the level kernel (4 blocks per SM) and 80 for the queue (3 blocks per SM).

Directions that remain open:
- a solve with fewer live registers, to allow more resident warps
- an application-level decision to keep X group-innermost
- for the queue, larger work items per atomic without delaying the critical path

## 11. Code map

| File | Contents |
|---|---|
| [`FEMSWEEP.hpp`](../src/apps/FEMSWEEP.hpp) | reference element routine and `FemsweepData`, the struct of device pointers every launcher takes |
| [`FEMSWEEP-Cuda.cpp`](../src/apps/FEMSWEEP-Cuda.cpp) | original Base_CUDA and RAJA_CUDA kernels; `femsweepLaunchOriginal` and `femsweepLaunchRaja` let the bench run them |
| [`FEMSWEEP-CudaElement.hpp`](../src/apps/FEMSWEEP-CudaElement.hpp) | `femsweepSolveElement`, the cooperative element solve (§3); `femsweepRcp`; the `face_b` stride |
| [`FEMSWEEP-CudaExpCoop.cpp`](../src/apps/FEMSWEEP-CudaExpCoop.cpp) | `femsweepFaceTable` (§3.3) and `coop_256`, the original schedule with the cooperative solve |
| [`FEMSWEEP-CudaExpDag.cpp`](../src/apps/FEMSWEEP-CudaExpDag.cpp) | `FemsweepDag`: setup, `daglevels`, `daggraph`, `dagqueue`, and the group-innermost X option (§4–§7) |
| [`FEMSWEEP-CudaLaunch.hpp`](../src/apps/FEMSWEEP-CudaLaunch.hpp) | declarations of all launchers; `runCudaLaunch` and `runCudaDag`, which time them inside raja-perf |
| [`FEMSWEEP-CudaExp.cpp`](../src/apps/FEMSWEEP-CudaExp.cpp) | registers the experimental tunings with raja-perf |
| [`femsweep-bench.cpp`](../src/femsweep-bench.cpp) | standalone driver: runs every implementation, checks it, and prints a timing table |

The figures in this folder are generated by [`make_figures.py`](make_figures.py) (`python3 make_figures.py .`), so their data and layout can be edited in one place.
