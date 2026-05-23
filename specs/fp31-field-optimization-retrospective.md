# Spec: fp31 Field Optimization Retrospective

| Field       | Value                  |
|-------------|------------------------|
| Author(s)   | Quang Dao              |
| Created     | 2026-05-22             |
| Status      | implemented            |
| PR          | #99 (`akita-fp31`)     |

## Summary

This PR specializes Akita's 31-bit prime-field arithmetic and moves the
workspace to Rust 1.95. The runtime goal is to make fp31 a more efficient
field choice than the existing fp32 path where the modulus permits cheaper
addition, subtraction, reduction, and packed multiplication, while preserving
canonical field representation and the existing prover/verifier behavior.

## Intent

### Goal

Make fp31 arithmetic faster in `akita-field` without changing proof formats,
transcript bytes, schedules, setup artifacts, or public PCS APIs.

### Invariants

- `Fp32<P>` values remain canonical representatives in `[0, P)`.
- Packed fp31 add, sub, and mul must match scalar arithmetic lane-by-lane.
- Extension-field packed arithmetic built on fp31 packings must match scalar
  extension arithmetic for edge-lane values near `0`, `1`, and `P - 1`.
- The verifier no-panic contract is unchanged; this PR does not add any new
  verifier-facing unchecked decoding or shape assumptions.
- Rust 1.95 is the only toolchain baseline for the root workspace and the
  standalone recursion package after this PR.

### Non-Goals

- No full end-to-end switch to an fp31 commitment profile.
- No new proof layout, schedule search policy, transcript binding, or
  serialization format.
- No Montgomery representation cutover. Plonky3's Monty31 ideas were used only
  when they translated cleanly to Akita's Solinas/canonical representation.
- No compatibility shim for older Rust versions.

## Evaluation

### Acceptance Criteria

- [x] `cargo +1.95 fmt -q`
- [x] `cargo +1.95 clippy --all --message-format=short -q -- -D warnings`
- [x] `cargo +1.95 test`
- [x] `cargo +1.95 test -p akita-field`
- [x] `cargo +1.95 test -p akita-pcs --benches --no-run`
- [x] Canonical profile run verifies successfully:
  `AKITA_PROFILE_TRACE=0 AKITA_PROFILE_LOG=error AKITA_MODE=onehot AKITA_NUM_VARS=32 cargo +1.95 run --release --example profile`
- [x] x86 AVX512 and forced-AVX2 packed Mersenne31 edge tests pass on `leopard`.

### Testing Strategy

The field tests cover scalar fp31 edge cases, random fp31 arithmetic against
integer modular arithmetic, `u128` reduction, packed base-field edge lanes, and
packed fp4 edge lanes for named and generic sub-32-bit fields. The CI suite
then exercises the full workspace, portability checks, fuzz targets, and
profile benchmark workflow.

### Performance

The retained optimizations were benchmark-gated:

- ARM/NEON Mersenne31 packed mul latency improved from `1.7309` to
  `1.2286 ns/lane`; throughput improved from `0.5632` to `0.2885 ns/lane`.
- Leopard AVX512 Mersenne31 packed mul latency improved from `338.34` to
  `240.48 ps/lane`.
- Leopard forced-AVX2 Mersenne31 packed mul latency improved from `587.25` to
  `481.35 ps/lane`.
- Scalar `half` on ARM improved from `0.9670` to `0.6724 ns/op`.

Plonky3-inspired scalar add/sub variants were rejected after local ARM
benchmarks showed regressions for Akita's representation.

## Design

### Architecture

`crates/akita-field` owns all runtime changes. `Fp32<P>` gets a shared
`2^64 mod p` associated constant and a specialized `HalvingField::half`
implementation. Packed NEON, AVX2, and AVX512 backends get guarded
Mersenne31 multiply fast paths. Existing packed fp31 Solinas add/sub behavior
remains the general path.

The Rust 1.95 cutover updates root workspace crate metadata, the root
toolchain, and the standalone recursion sub-workspace toolchain/config.

### Alternatives Considered

- **Use Plonky3 scalar add/sub forms.** Rejected: they regressed on ARM under
  Akita's canonical Solinas representation.
- **Adopt Montgomery representation.** Rejected: it would be a representation
  migration, not a local optimization.
- **Keep temporary scalar-half benchmark rows.** Rejected before review: the
  runtime specialization is covered by field tests, and the extra Criterion
  rows widened the benchmark surface more than this PR needs.
- **Keep a 31-bit NEON carry-reduction helper.** Rejected: the carry reducer is
  unreachable for fp31 dot products because that path returns through the
  non-carry reducer.

## Documentation

This retrospective spec is the single PR-specific spec artifact. Existing
historical specs should not be edited merely to satisfy spec-tracking; they
should retain the context of the PR they originally documented unless this PR
is intentionally amending that historical design.

## References

- Plonky3 `monty-31` and `mersenne-31` implementations in the sibling checkout.
- PR #99: `https://github.com/LayerZero-Labs/akita/pull/99`
- Local worklog: `WORKLOG-NEVER-COMMIT.md` (not committed)
