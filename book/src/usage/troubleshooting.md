# Troubleshooting

Common failure modes and fixes for building, profiling, and running Akita, plus
debugging the recursion guest. Each entry is symptom → cause → fix.

## Debug-profile guard

**Symptom:** the profile binary refuses to run in a debug build.

**Cause:** the profile harness refuses non-release builds to avoid misleading
timings (`crates/akita-pcs/examples/profile/main.rs:35-38`).

**Fix:** build with `--release`, or override with
`AKITA_ALLOW_DEBUG_PROFILE=1` for a correctness-only run.

## Eq-table OOM at high `num_vars`

**Symptom:** `EqPolynomial` allocation failure, or the prover tripping the
materialized eq-table budget at large `num_vars`.

**Cause:** some paths materialize full `2^num_vars` eq tables; the budget is a
real ceiling for `num_vars ≳ 41` on small-field one-hot profiles
(`crates/akita-algebra/src/eq_poly.rs`, `check_element_budget`).

**Fix:** lower `AKITA_NUM_VARS`. CI benches small-field presets at `nv=28`
under the eq-table memory budget (see the notes in
[quickstart.md](./quickstart.md)). The prover-side cap is separate from the
verifier-only allocation ceiling; the streamed-prover roadmap relaxes the
prover dependence on it
([`specs/eor-streamed-prover.md`](../../../specs/eor-streamed-prover.md)).

## Setup-cache invalidation

**Symptom:** a cached setup fails to deserialize after an upgrade.

**Cause:** setup cache layout is versioned; the cache file stores the expanded
setup followed by setup-prefix slots, and caches written before setup-prefix
persistence are rejected (`crates/akita-setup/src/lib.rs:1-5`).

**Fix:** delete the cache and regenerate. There is no compatibility wrapper —
regenerate the setup for the upgraded code.

## Rayon threads

**Symptom:** prove is slow or the machine is oversubscribed on many-core hosts.

**Cause:** Rayon parallelizes across all available cores by default; deep prover
paths can also hit stack overflows on the default worker stacks.

**Fix:** bound parallelism with `RAYON_NUM_THREADS`. A dedicated large-stack
Rayon pool for scheme compute is under review upstream
([PR #326](https://github.com/LayerZero-Labs/akita/pull/326)).

## Recursion-guest panics

**Symptom:** the Jolt recursion host returns a guest panic or a nonzero status.

**Fix:** the guest enables `jolt/stdout` so panic messages reach the host. To
get a symbolic backtrace, the `#[jolt::provable]` attribute defaults to
`backtrace = "off"` (faster traces); flip it to `backtrace = "dwarf"` for one
diagnostic iteration and re-run
(`profile/akita-recursion/README.md:94-113`):

```bash
ZEROOS_GUEST_RUSTFLAGS=-Zunstable-options \
    JOLT_BACKTRACE=full AKITA_RECURSION_LOG=info \
    ./target/release/akita-recursion-host --trace-only \
    --input target/akita_recursion_inputs_nv32.bin
```

To force a clean guest rebuild:

```bash
rm -rf /tmp/akita-recursion-targets /tmp/jolt-guest-targets
```

Malformed proofs produce a **proved nonzero result** rather than a guest panic
(status code `1`), because the guest decodes and verifies in-guest
(`profile/akita-recursion/README.md:87-92`).

## Environment-variable quick reference

Profile harness (`crates/akita-pcs/examples/profile/main.rs`):

| Variable | Default | Effect |
|----------|---------|--------|
| `AKITA_MODE` | `onehot_fp128_d64` | Profile preset (`onehot_*` / `dense_*`). |
| `AKITA_NUM_VARS` | `32` | Polynomial arity for the prover. |
| `AKITA_NUM_POLYS` | `1` | Number of committed polynomials. |
| `AKITA_PROFILE_LOG` | `trace` | `tracing-subscriber` filter. |
| `AKITA_ALLOW_DEBUG_PROFILE` | unset | `1` ⇒ bypass the `--release` guard. |

Recursion workspace (`profile/akita-recursion/README.md:115-124`):

| Variable | Default | Effect |
|----------|---------|--------|
| `AKITA_RECURSION_BLOB` | `target/akita_recursion_inputs.bin` | Output path for the artifact blob. |
| `AKITA_RECURSION_LOG` | `info` | `tracing-subscriber` filter (host). |
| `ZEROOS_GUEST_RUSTFLAGS` | unset | Pass `-Zunstable-options` when Rust requires it for Jolt's `riscv64imac-zero-linux-musl` target. |
| `JOLT_BACKTRACE` | unset | `full` ⇒ symbolic guest backtraces. |
