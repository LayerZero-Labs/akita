# Troubleshooting

Common failure modes and fixes for building, profiling, and running Akita, plus
debugging the recursion guest. Each entry is symptom → cause → fix.

## Debug-profile guard

**Symptom:** the profile binary refuses to run in a debug build.

**Cause:** the profile harness refuses non-release builds to avoid misleading
timings (`crates/akita-pcs/examples/profile/main.rs`).

**Fix:** build with `--release`, or override with
`AKITA_ALLOW_DEBUG_PROFILE=1` for a correctness-only run.

## Eq-table OOM at high `num_vars`

**Symptom:** `EqPolynomial` allocation failure, or the prover tripping the
materialized eq-table budget at large `num_vars`.

**Cause:** some paths materialize large `2^num_vars`-scale eq tables against the
1 GiB `MAX_MATERIALIZED_EQ_TABLE_BYTES` ceiling
(`crates/akita-algebra/src/eq_poly.rs`, `check_element_budget`). On the
small-field one-hot profiles the profile-bench matrix observes the prover's
eq-evaluation table exceeding that ceiling at `num_vars ≥ 30`; the raw full
`2^num_vars` table bound is lower, around `num_vars ≥ 27–29` depending on field
element size (fp128/fp64/fp32).

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
persistence are rejected (`crates/akita-setup/src/lib.rs`).

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

**Symptom:** the Jolt recursion host returns a guest panic.

**Cause:** the guest enables `jolt/stdout` so panic messages reach the host, but
`#[jolt::provable]` defaults to `backtrace = "off"` (measured faster), so a bare
panic message gives no source location.

**Fix:** flip it to `backtrace = "dwarf"` for one diagnostic iteration and re-run
(`profile/akita-recursion/README.md`). The invocation reads
`target/akita_recursion_inputs_nv32.bin`, which the artifact step must generate
first:

```bash
# 0. Run from the repository root; this cd takes you into the recursion
#    workspace, which the referenced README commands assume.
cd profile/akita-recursion

# 1. Build the binaries first (README quick-start step 1); otherwise
#    ./target/release/akita-recursion-* does not exist yet.
cargo build --release

# 2. Generate the verifier-input blob (required before the host run).
AKITA_NUM_VARS=32 \
    AKITA_RECURSION_BLOB=target/akita_recursion_inputs_nv32.bin \
    ./target/release/akita-recursion-artifact

# 3. Reproduce the panic with a symbolic backtrace.
ZEROOS_GUEST_RUSTFLAGS=-Zunstable-options \
    JOLT_BACKTRACE=full AKITA_RECURSION_LOG=info \
    ./target/release/akita-recursion-host --trace-only \
    --input target/akita_recursion_inputs_nv32.bin
```

To force a clean guest rebuild:

```bash
rm -rf /tmp/akita-recursion-targets /tmp/jolt-guest-targets
```

### Malformed proofs (nonzero status without a panic)

**Symptom:** the host reports a proved nonzero result even though the input
looks like a valid proof.

**Cause:** this is expected behavior — the guest decodes and verifies in-guest,
so a malformed proof yields guest status `1` (input decoding failed) or `2`
(verifier rejected the proof), not a guest panic
(`profile/akita-recursion/README.md`).

## Environment-variable quick reference

Profile harness (`crates/akita-pcs/examples/profile/main.rs`):

| Variable | Default | Effect |
|----------|---------|--------|
| `AKITA_MODE` | `onehot_fp128_d64` | Profile preset (`onehot_*` / `dense_*`). |
| `AKITA_NUM_VARS` | `25` | Polynomial arity for the prover. |
| `AKITA_NUM_POLYS` | `1` | Number of committed polynomials. |
| `AKITA_PROFILE_LOG` | `trace` | `tracing-subscriber` filter. |
| `AKITA_ALLOW_DEBUG_PROFILE` | unset | `1` ⇒ bypass the `--release` guard. |

Recursion workspace (`profile/akita-recursion/README.md`):

| Variable | Default | Effect |
|----------|---------|--------|
| `AKITA_RECURSION_BLOB` | `target/akita_recursion_inputs.bin` | Output path for the artifact blob. |
| `AKITA_RECURSION_LOG` | `info` | `tracing-subscriber` filter (host). |
| `ZEROOS_GUEST_RUSTFLAGS` | unset | Pass `-Zunstable-options` when Rust requires it for Jolt's `riscv64imac-zero-linux-musl` target. |
| `JOLT_BACKTRACE` | unset | `full` ⇒ symbolic guest backtraces. |
