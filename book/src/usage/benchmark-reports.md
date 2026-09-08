# Reading benchmark reports

Akita profile CI measures complete opening proofs at several public statement
shapes. Pull request runs compare the branch with its merge base on the same
runner, using the same feature set and interleaved executions.

The workflow at `.github/workflows/profile-bench.yml` defines the current cases.
The report scripts turn those runs into a compact pull request comment and a
full downloadable artifact.

## Current statement matrix

| Case | Polynomial groups | Setup path |
| --- | --- | --- |
| `dense_fp32` | 1 at nv30 | Direct |
| `onehot_fp32` | 1 at nv34 | Direct |
| `dense_fp64` | 1 at nv29 | Direct |
| `onehot_fp64` | 1 at nv35 | Direct |
| `dense_fp128` | 1 at nv28 | Direct |
| `onehot_fp128` | 1 at nv36 | Direct and recursive |
| `onehot_fp128_multi_group` | 1 at nv16 + 1 at nv16 + 2 at nv34 | Direct |
| `onehot_fp128_multi_group_recursive` | 1 at nv16 + 1 at nv16 + 2 at nv34 | Recursive |
| `onehot_fp128_multi_group_recursive_multi_chunk_w8r2` | 1 at nv16 + 1 at nv16 + 2 at nv32 | Recursive |
| `onehot_fp128_multi_chunk_w2r2` | 1 at nv32 | Direct |
| `onehot_fp128_multi_chunk_w4r2` | 1 at nv32 | Direct |
| `onehot_fp128_multi_chunk_w8r2` | 1 at nv32 | Direct |

The workflow splits these cases into narrow feature groups so each runner
compiles only the profile modes it measures. Each runner loads the matching
checked-in `.aks` schedule artifact at runtime.

## What each statement means

The dense cases commit to one arbitrary multilinear table and open it at one
point. The current fp32, fp64, and fp128 cases use 30, 29, and 28 variables.

The one hot source places one selected value in every 256 entry chunk. The
public statement still checks commitment and opening consistency. The source
representation changes prover work, not the meaning of the opening claim.

The multi group cases contain four polynomials in three ordered groups. Two
earlier groups each contain one 16 variable polynomial and use separate points.
The final group contains two polynomials that share one point. It uses 34
variables in the direct and standard recursive cases. The W8R2 case uses 32
variables.

Direct and recursive rows prove the same opening statement. The recursive setup
row carries the large public setup contribution through a Stage 3 sumcheck.
Direct mode evaluates that contribution during Stage 2.

W2R2, W4R2, and W8R2 divide the witness relation into 2, 4, or 8 chunks during
the first folds. The generated schedule may choose different A, B, and D ring
dimensions at each level.

## Work performed by one sample

Every measured process performs these operations:

1. Generate deterministic source data, opening points, and expected values.
2. Build setup and prepare retained NTT requirements.
3. Commit to every polynomial group.
4. Produce and serialize one complete opening proof.
5. Check the reported proof size.
6. Build verifier setup.
7. Verify with the configured worker pool.
8. Verify the same proof again with one worker.

Each sample runs in a fresh process. Setup is built in memory, and profile CI
does not enable disk persistence.

## How the comparison works

Pull request runs compare with the merge base, not the current tip of `main`.
The workflow builds both revisions and runs them on the same machine. It
interleaves the two binaries so changes in runner load affect them as evenly as
possible.

Each case has one discarded warmup process and three measured processes. The
report uses the median time and the largest peak resident memory. A negative
delta means that the branch used less time, memory, or proof data.

The x86 runners compile for the fixed `x86-64-v3` target. This keeps the
instruction set stable across machines while retaining AVX2. CPU fingerprints
are recorded with the artifacts.

## Compact pull request comment

The compact comment begins with the public statements. It then separates:

- Phase time.
- Peak memory and setup size.
- Proof size.
- The resolved A, B, and D dimension sequence.

Consecutive repeated dimension tuples are collapsed. This keeps the comment
readable while still showing a schedule change.

Failed cases remain visible and name the phase that failed. A missing or
unsupported merge base mode is reported instead of being compared with a
different workload.

## Full report artifact

`scripts/profile_bench_report.py` writes these files:

| File | Use |
| --- | --- |
| `summary.json` | Complete machine readable result |
| `summary.csv` | Flat data for analysis tools |
| `report.md` | Detailed human readable report |
| Pull request comment | Compact branch versus merge base summary |

The detailed report shows every fold. Each fold records matrix geometry,
decomposition, challenge parameters, witness or setup input, relation geometry,
and proof components. Multi group rows repeat these blocks for each group
instead of joining unrelated values.

The report also names the opening method for each group. A coefficient packing
row includes its challenge subring dimension, packing factor, partial width,
and packed field width. These fields explain a time or setup difference that a
single ring dimension cannot explain.

## Interpreting setup and cache numbers

`Setup and preparation` includes exact NTT prewarming for the selected commit
and prove schedule. The retention policy skips large ring switch requirements
that the runtime streams.

The harness checks that retained shared matrix cache entries do not grow during
commit or prove. As a result, these phases measure warm retained entries plus
intentional streamed work.

`Prepared NTT cache size` reports the retained shared matrix footprint at the
prewarm boundary. Compression cache entries are built by their operations and
are not included in that line. `CpuPreparedSetup::ntt_cache_bytes` reports the
complete resident cache when an application needs that number.

Dense statement preparation appears in traces but is excluded from protocol
phase comparisons. It includes deterministic evaluation generation and the
independent expected opening calculation.

## Keep reports tied to public claims

A performance number is useful only with its complete statement, schedule,
feature set, thread policy, and Git revision. The Akita report preserves these
facts so a proof size or timing change can be traced to the protocol choice that
caused it.

Use [Profiling a workload](./profiling.md) to reproduce one case locally. Use
[Arithmetic microbenchmarks](./arithmetic-benchmarks.md) when the report points
to NTT public matrix multiplication as the next target.
