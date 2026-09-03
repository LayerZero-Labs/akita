# Profiling a workload

Akita ships one profile harness for complete PCS measurements. It runs the
public statement preparation, setup, commitment, proof generation, encoding,
and verification needed by a real opening proof. It also records the generated
schedule and proof size breakdown.

## Run the canonical profile

Run this command from `crates/akita-pcs`:

```bash
AKITA_MODE=onehot_fp128 AKITA_NUM_VARS=32 \
  cargo run --release --no-default-features \
  --features parallel,profile-onehot-fp128,transcript-blake2b \
  --example profile
```

This build contains only the adaptive fp128 one hot profile and its required
protocol features. Schedule rows are loaded from the external artifact, while a
narrow build keeps unrelated profile-mode monomorphizations out of the binary.

The harness requires `--release`. Set `AKITA_ALLOW_DEBUG_PROFILE=1` only when
debugging the harness itself.

## What the default statement proves

`onehot_fp128` commits to one multilinear polynomial with $2^{32}$ entries and
opens it at one 32 coordinate point. The source stores one selected position in
each 256 entry chunk.

The public statement proves commitment and opening consistency. One hot is the
prover representation that lets Akita skip zero work. A host constraint system
can prove the one hot property separately when its application needs that
claim.

The admitted row from the external schedule artifact chooses the ring
dimensions for the root and every later fold. The report prints the selected
sequence. The application does not choose a fixed ring dimension for this run.

## Read the main phases

The profile output separates these phases:

| Phase | Work included |
| --- | --- |
| Statement preparation | Deterministic witness, opening point, and expected value generation |
| Setup and preparation | Public setup construction and exact retained NTT prewarming |
| Commit | Commitment to the source polynomial groups |
| Prove | Complete opening proof generation |
| Encode | Compressed proof serialization and size accounting |
| Verify | Replay with the configured multi threaded verifier pool |
| Verify with one thread | Replay of the same proof with one verifier worker |

Statement preparation belongs to the harness, not the PCS. Use the named setup,
commit, prove, encode, and verify phases for protocol comparisons.

## Choose the workload

| Variable | Default | Meaning |
| --- | --- | --- |
| `AKITA_MODE` | `onehot_fp128` | Configuration and source representation |
| `AKITA_NUM_VARS` | `32` | Number of multilinear variables |
| `AKITA_NUM_POLYS` | `1` | Number of polynomials in the final group |
| `AKITA_PROFILE_PROVE_THREADS` | Rayon default | Prover worker count |
| `AKITA_PROFILE_VERIFY_THREADS` | Prover count | Verifier worker count |
| `RAYON_NUM_THREADS` | Rayon default | Fallback worker count |

The compiled binary accepts only modes enabled by its feature set. An unknown
mode prints the available names.

The repository also provides dense, small field, grouped, recursive setup, and
multi chunk modes. Use the narrow feature that owns the selected mode when
collecting numbers for comparison.

## Record a Perfetto trace

Tracing is enabled by default for the profile harness.

| Variable | Default | Purpose |
| --- | --- | --- |
| `AKITA_PROFILE_TRACE` | `1` | Write a Chrome and Perfetto trace |
| `AKITA_PROFILE_MONITOR` | `1` when tracing | Record process CPU, host CPU, and memory counters |
| `AKITA_PROFILE_MONITOR_INTERVAL_MS` | `100` | Resource sampling interval in milliseconds |
| `AKITA_PROFILE_LOG` | `trace` | Console tracing filter |
| `AKITA_PROFILE_ANSI` | `1` | Colored terminal logs |
| `AKITA_PROFILE_SPAN_CLOSES` | `1` | Print span close events |

The trace contains counters for effective CPU cores, CPU percentage, resident
memory, virtual memory, and logical CPU count. One effective core means one core
was busy for the complete sampling interval.

After writing the trace, the harness writes a sibling `*.summary.json`. The
summary records the run identity, Git revision, exact peak resident memory,
root wall time, CPU pool use, counter statistics, and time for every span label.

Inspect it without opening Perfetto:

```bash
jq '{run, root, peak_rss_gib, cpu_utilization}' \
  profile_traces/akita_nv32_onehot_fp128_*.summary.json
jq '.spans["AkitaCommitmentScheme::batched_prove"]' \
  profile_traces/akita_nv32_onehot_fp128_*.summary.json
```

The sampled memory line shows when memory changes. The exact process high water
mark records the largest resident set even when a short peak falls between
samples.

## Compare thread policies

Set prover and verifier counts before the process starts. Rayon fixes its global
pool during initialization.

A prover value of zero lets Rayon choose. A verifier value of zero reuses the
resolved prover count. A positive verifier value creates a separate pool when
the counts differ. The harness prints both resolved counts.

Build without `parallel` to measure the same workload sequentially:

```bash
AKITA_MODE=onehot_fp128 AKITA_NUM_VARS=32 \
  cargo run --release --no-default-features \
  --features profile-onehot-fp128,transcript-blake2b \
  --example profile
```

## Measure response model data

Normal profile runs measure the production prover path. Response model
calibration adds full witness scans, so enable it explicitly for those runs:

```bash
AKITA_MODE=onehot_fp128 AKITA_NUM_VARS=32 \
  cargo run --release --no-default-features \
  --features parallel,profile-onehot-fp128,transcript-blake2b,response-model-diagnostics \
  --example profile
```

This mode records exact source and response energies, planned caps, accepted
nonces, and attempt counts. Those additional scans change prover work, so keep
their results separate from normal performance measurements.

## Continue the analysis

- [Reading benchmark reports](./benchmark-reports.md) explains the CI matrix
  and pull request comparison artifacts.
- [Arithmetic microbenchmarks](./arithmetic-benchmarks.md) isolates NTT public
  matrix multiplication and SIMD paths.
- [Troubleshooting](./troubleshooting.md) explains profile startup, schedule,
  allocation, cache, and thread errors.
