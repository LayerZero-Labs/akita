# z_first verifier profile (nv=36, onehot fp128 D=64)

Recorded **2026-06-25** on the local dev machine while evaluating whether to
hard-code `z_first = true` and remove the adaptive `ring_column_z_first` flag.
Uses temporary env-gated instrumentation in the profile example and verifier
(`AKITA_FORCE_Z_FIRST`, `AKITA_SPAN_AGG`, `verify_L{n}` parent spans). Remove
that scaffolding before the final z_first-removal PR.

## Workload

| Parameter | Value |
|-----------|-------|
| Mode | `onehot_fp128_d64` |
| `AKITA_NUM_VARS` | 36 |
| `AKITA_NUM_POLYS` | 1 |
| Fold levels | 8 (L0 root … L7 terminal) |
| `setup_contribution_mode` | Direct |
| Verify repeats | 30 per configuration |

## Reproduce

From `crates/akita-pcs/` (release build; first compile is slow):

```bash
# z_first = TRUE (forced)
AKITA_FORCE_Z_FIRST=1 AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=36 AKITA_NUM_POLYS=1 \
AKITA_SPAN_AGG=1 AKITA_VERIFY_REPEAT=30 \
AKITA_PROFILE_TRACE=0 AKITA_PROFILE_SPAN_CLOSES=0 AKITA_PROFILE_LOG=info AKITA_PROFILE_ANSI=0 \
cargo run --release --example profile

# z_first = FALSE (forced): same with AKITA_FORCE_Z_FIRST=0
```

**Gotcha:** `AKITA_PROFILE_LOG` must be `info` or finer; at `warn` the span
aggregator sees no INFO spans and the breakdown is empty.

Read the `=== verify span breakdown AVERAGE over 30 iters ===` and
`=== per-level row-eval breakdown ===` blocks at the end of the output.

Raw logs from the recording run:

- `/tmp/akita_zfirst_true_30_level.txt`
- `/tmp/akita_zfirst_false_30_level.txt`

(Copy elsewhere if `/tmp` may be cleared.)

## Total verify (30-iter average)

| Config | `batched_verify` | Δ vs false |
|--------|-----------------:|-----------:|
| `z_first = true` | **70.94 ms** | −4.25 ms (−5.4%) |
| `z_first = false` | **75.19 ms** | — |

Earlier 30-iter runs without per-level parent spans (aggregated only):

| Config | `batched_verify` |
|--------|-----------------:|
| `z_first = true` | 70.47 ms |
| `z_first = false` | 74.51 ms |

## Aggregated row-eval components (30-iter average)

Spans under each fold level's `verify_L{n}` parent, summed across L0–L7.

| Component | z_first=true | z_first=false | Δ (true − false) |
|-----------|-------------:|--------------:|-----------------:|
| `setup_contribution` | 58.16 ms | 57.63 ms | +0.54 ms (noise) |
| `z_structured` | **2.74 ms** | **7.49 ms** | **−4.76 ms** |
| `e_structured` | 0.008 ms | 0.008 ms | 0 |
| `t_structured` | 0.025 ms | 0.025 ms | 0 |
| `r_dense` (+ `r_structured`) | 0.51 ms | 0.65 ms | −0.14 ms |

**Conclusion:** the entire meaningful verify gap is in `z_structured`.
`setup_contribution` is ordering-independent; `e` and `t` are sub-0.03 ms total
and unchanged.

## Per-level row-eval breakdown (ms, 30-iter average)

`block_len` from the planned schedule logged at prove time. `r` combines
`r_dense` and `r_structured` (only one is active per level).

### z_first = true

| L | block_len | setup | z | e | t | r | row_total |
|---|----------:|------:|--:|--:|--:|--:|----------:|
| L0 | 262144 | 50.17 | 0.83 | 0.002 | 0.006 | 0.007 | 51.01 |
| L1 | 14339 | 4.01 | 0.90 | 0.001 | 0.006 | 0.005 | 4.93 |
| L2 | 3337 | 1.19 | 0.35 | 0.001 | 0.004 | 0.004 | 1.54 |
| L3 | 1384 | 0.80 | 0.19 | 0.001 | 0.002 | 0.121 | 1.11 |
| L4 | 585 | 0.55 | 0.13 | 0.001 | 0.002 | 0.103 | 0.78 |
| L5 | 503 | 0.55 | 0.13 | 0.001 | 0.002 | 0.093 | 0.77 |
| L6 | 342 | 0.50 | 0.12 | 0.001 | 0.002 | 0.087 | 0.71 |
| L7 | 281 | 0.41 | 0.10 | 0.001 | 0.002 | 0.089 | 0.60 |
| **sum** | | **58.16** | **2.74** | **0.008** | **0.025** | **0.51** | **61.44** |

### z_first = false

| L | block_len | setup | z | e | t | r | row_total |
|---|----------:|------:|--:|--:|--:|--:|----------:|
| L0 | 262144 | 49.36 | 0.82 | 0.002 | 0.006 | 0.007 | 50.19 |
| L1 | 14339 | 3.93 | **4.28** | 0.001 | 0.005 | 0.004 | **8.21** |
| L2 | 3337 | 1.25 | **1.25** | 0.002 | 0.004 | 0.003 | **2.50** |
| L3 | 1384 | 0.97 | 0.46 | 0.001 | 0.002 | 0.151 | 1.58 |
| L4 | 585 | 0.65 | 0.18 | 0.001 | 0.003 | 0.134 | 0.96 |
| L5 | 503 | 0.50 | 0.19 | 0.001 | 0.002 | 0.125 | 0.81 |
| L6 | 342 | 0.52 | 0.18 | 0.001 | 0.002 | 0.123 | 0.83 |
| L7 | 281 | 0.46 | 0.15 | 0.001 | 0.002 | 0.102 | 0.71 |
| **sum** | | **57.63** | **7.49** | **0.008** | **0.025** | **0.65** | **65.80** |

### Δ z per level (true − false)

| L | Δ setup | **Δ z** | Δ e | Δ t | Δ r |
|---|--------:|--------:|----:|----:|----:|
| L0 | +0.81 | +0.01 | 0 | 0 | 0 |
| L1 | +0.08 | **−3.37** | 0 | 0 | 0 |
| L2 | −0.07 | **−0.90** | 0 | 0 | 0 |
| L3 | −0.17 | −0.27 | 0 | 0 | −0.03 |
| L4 | −0.10 | −0.05 | 0 | 0 | −0.03 |
| L5 | +0.05 | −0.06 | 0 | 0 | −0.03 |
| L6 | −0.03 | −0.06 | 0 | 0 | −0.04 |
| L7 | −0.05 | −0.05 | 0 | 0 | −0.01 |
| **sum** | +0.54 | **−4.76** | 0 | 0 | −0.14 |

~**74%** of the total z savings is at **L1** (`block_len = 14339`, `log_basis = 2`).
~**19%** at **L2**. L3–L7 contribute ~0.05–0.27 ms each.

## Per-level verify wall time (includes sumchecks, not just row-eval)

| L | z_first=true | z_first=false | Δ |
|---|-------------:|--------------:|--:|
| L0 | 55.16 ms | 54.32 ms | +0.84 ms |
| L1 | 5.53 ms | 8.80 ms | −3.27 ms |
| L2 | 1.82 ms | 2.77 ms | −0.95 ms |
| L3 | 1.28 ms | 1.75 ms | −0.47 ms |
| L4 | 0.94 ms | 1.12 ms | −0.18 ms |
| L5 | 0.92 ms | 0.95 ms | −0.03 ms |
| L6 | 0.85 ms | 0.97 ms | −0.12 ms |
| L7 | 1.74 ms | 1.84 ms | −0.10 ms |

## Interpretation

- Hard-coding `z_first = true` saves ~**4–5 ms (~5%)** on total verify at
  nv=36 onehot D=64 with no cost to `e`, `t`, or `setup_contribution`.
- L0 dominates setup (~50 ms / ~58 ms summed setup); ordering does not matter
  there.
- Proof tail sizes are identical between runs (`tail_z_first` 1 vs 0); only
  column traversal order changes.

## Instrumentation touchpoints (temporary)

| Location | Purpose |
|----------|---------|
| `crates/akita-types/src/proof/ring_relation.rs` | `AKITA_FORCE_Z_FIRST` |
| `crates/akita-pcs/examples/profile/span_agg.rs` | `AKITA_SPAN_AGG=1` per-span aggregator |
| `crates/akita-pcs/examples/profile/workload.rs` | `AKITA_VERIFY_REPEAT` |
| `crates/akita-verifier/src/protocol/core.rs` | `verify_L{n}` parent spans |
| `crates/akita-verifier/src/protocol/ring_switch.rs` | `z/e/t/setup_contribution` child spans |
| `crates/akita-pcs/examples/profile/report.rs` | per-level breakdown printer |
