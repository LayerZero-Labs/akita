# JL x86 kernel measurements — September 2026

This snapshot records the measurements behind the AVX2 and AVX-512 dispatch
choices. It is not a complete PCS benchmark or a schedule-wide security claim.
The baseline is `db08f40b0f8f6de7aa735600c634f3b3342d33f0`; the integrated
kernel and full-field test snapshot is `d62bea6ea`.

## Measurement conditions

- AMD EPYC 9554P (Zen 4), Linux VM with two visible CPUs.
- Rust 1.95.0, LLVM 22.1.2, Cargo release profile.
- Portable build flags; individual kernels use runtime-checked target features.
- Five trials per comparison. AES uses 1 GiB of output per trial and size.
- Projection uses 256 rows and 4K, 16K, and 64K columns.
- Field contraction uses 256 rows and 16K columns, with 50 iterations per
  production field per trial. All extension coefficients are sampled with a
  fixed RNG seed, not embedded from the base field.
- Builds and timing runs are serialized. VM scheduling still causes noise;
  small differences do not justify processor-specific dispatch tables.

## AES expansion

Representative throughput over 128 KiB, 1 MiB, and 8 MiB output buffers:

| Backend | GB/s |
| --- | ---: |
| RustCrypto CTR baseline | 7.6 |
| AES-NI, eight blocks | 10.4–10.5 |
| VAES256, eight vectors | 21.3–21.8 |
| VAES512, four vectors | 22.45–22.74 |

Setup-inclusive results closely track steady-state results. These numbers
measure AES filling, not matrix allocation or the complete expansion API.
The retained priority is VAES512, VAES256, AES-NI, then stock CTR. Each choice
checks its complete instruction-set requirements. Buffers below 256 bytes and
non-x86 targets retain stock CTR. Round keys stay in bounded inline storage.

Direct block generation lost 4–6% at large sizes. Reusing a stock CTR key
schedule was neutral. VAES512 with eight state vectors was no faster than four;
sixteen was slower. The manual AES benchmark retains those batch comparisons.

## Integer projection

At 256 × 16K, median packed AVX-512 kernel times were:

| Input | Old selector decode, µs | New decode, µs | Speedup |
| --- | ---: | ---: | ---: |
| i8 | 189.05 | 133.33 | 1.42× |
| i16 | 187.98 | 134.03 | 1.40× |
| i32 | 424.74 | 273.34 | 1.55× |
| i64 | 417.16 | 264.45 | 1.58× |

This compares selector kernels, not the old public dispatch for every input
type: the baseline public i32 path used dense arithmetic, and i16 had a 64K
packed threshold. The old wide selector comparator is generalized to i32 for
this experiment. Similar selector-kernel gains held at 4K and 64K columns.

AVX2 packed versus dense-first-call medians at 16K were 280 versus 1,324 µs for
i8, 277 versus 1,266 µs for i16, 1,138 versus 1,403 µs for i32, and 1,028 versus
1,778 µs for i64. Dense kernels won on reuse. Consequently, AVX2 uses a packed
first call, switches to the dense cache thereafter, and prepares dense directly
for multiple blocks. AVX-512 retains packed kernels. Here, cold means an empty
lazy compute cache, not cold DRAM.

An AVX2 permutation-based wide lookup lost about 8% to gathers at 64K and was
removed.

## Matrix field contraction

Full-field medians, including lookup-table construction and the complete matrix
scan, in microseconds per contraction:

| Field | Scalar | AVX2 | AVX-512 | Speedup, AVX2 / AVX-512 |
| --- | ---: | ---: | ---: | ---: |
| Fp32Ext | 3,332 | 2,420 | 2,359 | 1.38× / 1.41× |
| Fp64Ext | 3,928 | 2,289 | 2,067 | 1.72× / 1.90× |
| Fp128 | 3,131 | 2,317 | 2,407 | 1.35× / 1.30× |

SIMD decodes selectors into the same 81-entry lookup table; field values retain
their generic representation and arithmetic. No field-layout cast or new heap
buffer is needed. Equality-table construction remains separate from these
contraction timings. Earlier measurements restricted to embedded base-field
values are superseded by this table.

## Validation scope

The integrated snapshot passed the x86 JL suites with and without parallelism
(41 and 40 tests respectively; ignored timing tests run separately), both
corresponding algebra/challenges Clippy configurations, and a portable x86-64
verifier build with warnings denied. Forced-backend tests cover AES counter
wrap and tails, exhaustive packed selectors, full extension-field coefficients,
multiple vector batches, odd dimensions, and parallel panel composition.
Independent reviews of the SIMD implementations found no actionable issue.

All three repository CI Clippy configurations and documentation/style gates
passed on the development Mac. Schedule regeneration produced no artifact
changes. One full-suite shard passed 952 tests before the final column-selector
integration; the second shard's rebuild was stopped to close out the work.
This snapshot therefore does not claim a completed full-suite test run on the
final tree.

## Column weights

The column-weight kernel vectorizes the actual 4 × 4 bit transpose, then maps
the resulting nibbles to the same 81-entry field lookup tables. It uses only
fixed stack scratch and scalar tails. Full-field medians at 256 × 16K are:

| Field | Scalar, ms | AVX2, ms | AVX-512, ms | Speedup, AVX2 / AVX-512 |
| --- | ---: | ---: | ---: | ---: |
| Fp32Ext | 2.547 | 1.622 | 2.442 | 1.57× / 1.04× |
| Fp64Ext | 3.517 | 2.794 | 2.685 | 1.26× / 1.31× |
| Fp128 | 3.228 | 1.431 | 1.458 | 2.26× / 2.22× |

AVX2 is the dispatch preference because it improves all three fields
consistently. AVX-512 is independently implemented and force-tested. An earlier
experiment that merely applied target-feature annotations produced identical
scalar code and was removed; these results measure explicit SIMD operations.

## Reproduce and interpret

### Public API before and after

A separate Criterion run compares the baseline with the integrated snapshot
using the public APIs, 256 × 16K matrices, one Rayon thread, CPU 0 affinity,
300 ms warmup, and one-second measurement windows. These are point estimates
from short VM runs, not the five-trial kernel medians above.

| Operation | Baseline, ms | Updated, ms | Speedup |
| --- | ---: | ---: | ---: |
| Matrix expansion | 0.1551 | 0.0666 | 2.33× |
| Expansion + cold i32 projection | 1.5292 | 0.3235 | 4.73× |
| Matrix MLE, Fp32Ext, including equality tables | 3.1125 | 2.9940 | 1.04× |
| Matrix MLE, Fp64Ext, including equality tables | 3.1911 | 2.3003 | 1.39× |
| Matrix MLE, Fp128, including equality tables | 3.0172 | 2.5732 | 1.17× |
| Column weights, Fp32Ext, from point | 2.5709 | 2.3435 | 1.10× |
| Column weights, Fp64Ext, from point | 3.8620 | 3.4713 | 1.11× |
| Column weights, Fp128, from point | 3.4841 | 2.3403 | 1.49× |

Equality-table construction itself was unchanged; its measured variation was
within roughly 5%. Complete API gains are smaller than isolated kernel gains
for several fields. In particular, do not present the 1.57× Fp32Ext column
kernel improvement as a 1.57× public-API improvement.

```bash
RAYON_NUM_THREADS=1 AKITA_JL_BENCH_LOG_WIDTHS=14 taskset -c 0 \
  cargo bench -p akita-challenges --bench ternary_jl --features parallel -- \
  --warm-up-time 0.3 --measurement-time 1 --noplot \
  '(balanced_ternary_jl/(expand|expand_project_i32_cold)/|balanced_ternary_jl_(mle_e2e|column_weights_e2e|eq_tables)/fp(32_ext4|64_ext2|128)/)'
```

### Forced-backend measurements

Run these ignored microbenchmarks on a host supporting the named backends:

```bash
RAYON_NUM_THREADS=1 cargo test --release -p akita-challenges --lib \
  x86_aes_backend_microbench -- --ignored --nocapture --test-threads=1
RAYON_NUM_THREADS=1 cargo test --release -p akita-algebra --lib \
  x86_projection_microbench -- --ignored --nocapture --test-threads=1
RAYON_NUM_THREADS=1 cargo test --release -p akita-algebra --lib \
  matrix_mle_microbenchmark -- --ignored --nocapture --test-threads=1
RAYON_NUM_THREADS=1 cargo test --release -p akita-algebra --lib --features parallel \
  column_weight_kernel_microbenchmark -- --ignored --nocapture --test-threads=1
```

The projection benchmark reports its own five-trial medians. Repeat the AES and
MLE commands five times for the comparisons above. Unsupported backends skip;
building with `target-cpu=native` alone is not evidence of AVX2 execution on an
AVX-512 host. See the [microbenchmark guide](../book/src/usage/arithmetic-benchmarks.md)
for public-API Criterion cases and cold-versus-cached labels.
