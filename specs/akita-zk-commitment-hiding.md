# Spec: Akita ZK Commitment Hiding HidingMode

| Field       | Value                  |
|-------------|------------------------|
| Author(s)   | Amirhossein Khajehpour |
| Created     | 2026-05-05             |
| Status      | implemented            |
| PR          | TBD                    |

## Summary

This PR adds a feature-gated commitment hiding mode to Akita. The existing
commitment scheme remains transparent and deterministic by default, while the
new `zk` feature exposes a `ZK` mode that samples fresh Leftover Hash Lemma
(LHL) blinding material for the outer Ajtai commitment. The change threads the
mode through setup sizing, schedule/proof-size planning, commitment hints,
recursive witness construction, ring-switch prover/verifier replay, and
end-to-end scheme APIs so hidden commitments can be produced and verified
without changing the public proof format for transparent users.

## Intent

### Goal

Introduce a compile-time commitment masking mode for Akita so callers can choose
between the existing transparent commitment behavior and a ZK commitment-hiding
behavior that re-randomizes commitments with fresh outer B-matrix blinding.

The key surfaces modified are:

- `akita-types`: `HidingMode`, `Transparent`, feature-gated `ZK`, mode-aware witness
  sizing formulas, and ZK-aware commitment hints.
- `akita-config` and `akita-planner`: mode-aware schedule lookup, planner
  fallback, proof-size formulas, and setup matrix sizing.
- `akita-setup`: setup construction parameterized by commitment mode.
- `akita-prover`: fresh blinding sampling, commitment kernels, recursive hint
  caches, quadratic-equation assembly, ring-switch witness construction, and
  recursive `w` commitments.
- `akita-verifier`: mode-aware recursive witness lengths and verifier-side
  M-table evaluation for the extra blinding segment.
- `akita-scheme`: `AkitaCommitmentScheme<D, Cfg, M = Transparent>` and
  mode-threaded prover/verifier trait implementations.
- `akita-pcs`: public `zk` feature wiring, tests, examples, and benchmarks
  updated to select transparent layouts explicitly where needed.

### Invariants

1. Transparent mode remains the default API and preserves deterministic
   commitment behavior, schedule selection, proof verification, and existing
   tests except for explicit type-parameter updates.
2. `ZK` mode is available only when the `zk` Cargo feature is enabled. Builds
   without `zk` must not require OS randomness or ZK-only hint fields.
3. Every prover and verifier path that derives a recursive witness length or
   M-table width must use the same `HidingMode` type. Otherwise stage-1/stage-2
   verifier replay would evaluate a different witness shape than the prover.
4. The LHL blinding count must be derived from the output ring length,
   ring dimension, field modulus bit width, and a 128-bit statistical security
   target. Each committed group gets fresh blinding material.
5. Setup sizing must reserve enough shared-matrix stride for the original
   outer input plus the mode-specific blinding columns at every fold level used
   by the selected schedule.
6. Blinding digits are prover hint material only. They must not be serialized
   into public proofs or absorbed directly into the transcript; verifier replay
   checks their effect through the public commitment and ring-switch relation.
7. Prover and verifier must agree on the recursive witness segment order:
   `w_hat`, `t_hat`, ZK blinding, `z_hat`, and `r_hat`, with the existing
   `z_first` root-vs-recursive ordering rule preserved.
8. Root-direct verification remains a transparent preservation path that
   recomputes commitments from directly revealed field witnesses. ZK mode must
   not rely on this path to claim commitment hiding.
9. Generated transparent schedule tables remain valid for transparent mode.
   ZK schedules must account for extra blinding width and therefore cannot
   blindly reuse transparent generated rows.

### Non-Goals

1. Implementing the full Whiteout/zero-knowledge proof layer for all protocol
   messages. This PR hides commitments; it does not make every prover message
   simulatable.
2. Changing the Fiat-Shamir domain labels, transcript absorption order, or
   public proof encoding for transparent mode.
3. Generating dedicated ZK schedule tables in this PR. ZK mode may use planner
   fallback until audited generated tables are added.
4. Preserving old non-generic helper signatures such as
   `Cfg::commitment_layout(num_vars)`. The repo has no backward-compatibility
   guarantee, and mode-aware callers should pass the mode explicitly.
5. Adding runtime switching between transparent and ZK behavior. HidingMode selection
   is a type-level decision.
6. Publishing or stabilizing the `zk` feature as a final public API before the
   full zero-knowledge proof story is specified.

## Evaluation

### Acceptance Criteria

- [x] `akita-types` defines `HidingMode`, default `Transparent`, feature-gated `ZK`,
      and a 128-bit LHL blinding-count formula based on `CanonicalField`
      modulus bit width.
- [x] `CanonicalField` exposes modulus bit width for all concrete base fields
      used by Akita configs.
- [x] `AkitaCommitmentScheme` is generic over `M: HidingMode` with
      `Transparent` as the default type parameter.
- [x] `CommitmentProver` and `CommitmentVerifier` expose an associated `HidingMode`
      so dummy implementations and real scheme implementations make masking
      behavior explicit.
- [x] `CommitmentConfig`, `ScheduleProvider`, setup sizing, root commitment
      layout selection, prove schedule selection, and batched root layout
      selection are parameterized by `M: HidingMode`.
- [x] Generated fp128 schedules are used for transparent mode and deliberately
      not reused for ZK mode.
- [x] Planner proof-size and runtime generated-schedule validation include the
      mode-specific blinding segment in planned recursive witness sizes.
- [x] Prover setup capacity includes the extra ZK B-input columns in
      `max_stride`.
- [x] Root and recursive commitment kernels sample fresh blinding digits in
      `ZK` mode, append them to the outer B input, and preserve them in
      `AkitaCommitmentHint`.
- [x] Recursive commitment hint caches preserve and reconstruct ZK blinding
      material across runtime-D dispatch boundaries.
- [x] Quadratic-equation construction, ring-switch witness construction, and
      verifier M-evaluation include the blinding segment with matching offsets.
- [x] Direct transparent commitments remain deterministic for the same
      polynomial and setup.
- [x] ZK commitments to the same polynomial and setup re-randomize.
- [x] ZK dense end-to-end commit/prove/serialize/deserialize/verify tests pass
      for D=32, D=64, and D=128 fp128 full-field configs.
- [x] CI runs the ZK commitment hiding integration test explicitly.

### Testing Strategy

Existing tests must continue passing under the default transparent build:

- `cargo nextest run --all-features`
- `cargo test`
- `cargo clippy --all --message-format=short -q -- -D warnings`

New and updated focused checks:

- `cargo test -p akita-pcs --features zk --test zk_commitment_hiding`
- `cargo test -p akita-pcs --features zk --test zk_commitment_hiding zk_dense_d32_commitments_rerandomize_and_verify -- --exact`
- `cargo test -p akita-pcs --features zk --test zk_commitment_hiding zk_dense_d64_commitments_rerandomize_and_verify -- --exact`
- `cargo test -p akita-pcs --features zk --test zk_commitment_hiding zk_dense_d128_commitments_rerandomize_and_verify -- --exact`
- `cargo test -p akita-pcs --features zk --test zk_commitment_hiding direct_d32_commitments_are_transparent_and_verify -- --exact`
- `cargo test -p akita-pcs --features zk --test zk_commitment_hiding direct_d64_commitments_are_transparent_and_verify -- --exact`
- `cargo test -p akita-pcs --features zk --test zk_commitment_hiding direct_d128_commitments_are_transparent_and_verify -- --exact`
- `cargo doc -q --no-deps --all-features` to ensure the feature-gated public
  mode API documents cleanly.

### Performance

Transparent mode should have no intentional performance regression beyond the
extra type parameters and monomorphization. ZK mode intentionally increases
commitment, recursive witness, setup-stride, and proving work by adding
mode-specific B-blinding columns.

Relevant smoke checks:

- Existing benchmark targets must compile after mode threading.
- Existing profile example must compile after selecting transparent layouts.
- ZK proof-size/schedule changes should be measured against the planned witness
  size formulas before dedicated generated ZK schedule tables are added.

## Design

### Architecture

The mode abstraction lives in `akita-types` so every crate that derives proof,
schedule, setup, prover, or verifier shapes can use one shared contract:

```rust
pub trait HidingMode: Clone + Copy + Send + Sync + 'static {
    const ZK_ENABLED: bool;

    fn blind_ring_count<F: CanonicalField>(
        output_ring_len: usize,
        ring_dimension: usize,
    ) -> usize;

    fn blind_column_count<F: CanonicalField>(
        output_ring_len: usize,
        ring_dimension: usize,
        num_digits_open: usize,
    ) -> usize;
}
```

`Transparent` returns zero blinding. `ZK` computes enough fresh ring elements to
cover the outer output entropy plus LHL slack:

- one ring element contributes approximately `D * log2(q)` bits;
- the target is output entropy plus `2 * 128 - 2` bits;
- decomposed blinding columns are `blind_ring_count * num_digits_open`.

Commitment kernels append the sampled blinding digit planes after the ordinary
per-polynomial `t_hat` planes before multiplying by the B matrix. The resulting
public commitment stays a `RingCommitment`; the extra random source is retained
only in the prover hint so the folded witness can prove the commitment relation.

The recursive witness and verifier M-table gain a matching blinding segment.
For each commitment group, the segment occupies the B-input columns immediately
after that group's ordinary message columns. Prover `compute_m_evals_x`,
verifier `prepare_m_eval` / `PreparedMEval::eval_at_point`, and
`build_w_coeffs` must use identical lengths and offsets.

Schedule and setup policy become mode-aware rather than hardcoding transparent
sizes. Transparent generated tables remain authoritative. For ZK mode, fp128
configs return no generated plan so planner fallback can choose schedules using
the larger hidden-commitment witness sizes.

### Alternatives Considered

1. Runtime enum mode selection.
   This was rejected because all affected schedule, setup, prover, and verifier
   paths are already strongly typed by field/config/ring dimension. A type-level
   mode keeps generated code explicit and avoids runtime branching in hot paths.

2. Always enable blinding and treat transparent commitments as zero masks.
   This was rejected because transparent mode should preserve current proof
   sizes, schedules, setup sizes, and deterministic tests.

3. Reuse transparent generated schedules for ZK mode.
   This was rejected because ZK mode changes recursive witness lengths and setup
   stride through additional B-input columns. Reusing transparent rows would
   undercount proof sizes and could select invalid recursion shapes.

4. Serialize blinding material in proofs.
   This was rejected because blinding is secret prover witness data. The public
   proof should verify its effect through the commitment relation and sumchecks,
   not reveal the randomizer.

5. Make root-direct hidden commitments verify by re-randomized recomputation.
   This was rejected for this PR. The current root-direct path reveals the
   direct witness and preserves transparent recomputation behavior; hidden
   commitments should be opened through folded proofs until the direct path is
   redesigned.

## Documentation

This spec documents the branch-level design. Follow-up documentation should
update public API docs once the `zk` feature is ready to advertise beyond
internal testing, especially:

- README feature-flag notes for `zk`;
- a short protocol note explaining that this is commitment hiding, not full
  proof zero-knowledge;
- profiler/benchmark notes for comparing transparent and ZK schedules.

## Execution

Implemented order:

1. Add mode types and field modulus bit-width support.
2. Thread `M: HidingMode` through scheme traits, configs, setup construction, planner
   search, proof-size formulas, and schedule materialization.
3. Add prover-side masking sampling and append sampled digits to root and
   recursive B-matrix commitment inputs.
4. Extend `AkitaCommitmentHint` and `RecursiveCommitmentHintCache` to carry
   outer blinding digits under the `zk` feature.
5. Extend quadratic-equation and ring-switch prover logic to include the
   blinding segment in the recursive witness.
6. Extend verifier ring-switch M-evaluation and recursive witness-length checks
   with the same mode-dependent blinding segment.
7. Update scheme orchestration, examples, benches, and existing tests to use
   explicit transparent layouts when they inspect config policy directly.
8. Add ZK commitment hiding integration tests and wire them into CI.

## References

- `specs/TEMPLATE.md`
- `specs/akita-crate-followup-jolt-integration.md`
- `crates/akita-types/src/mode.rs`
- `crates/akita-prover/src/protocol/masking.rs`
- `crates/akita-prover/src/api/commitment.rs`
- `crates/akita-prover/src/protocol/ring_switch.rs`
- `crates/akita-verifier/src/protocol/ring_switch.rs`
- `crates/akita-pcs/tests/zk_commitment_hiding.rs`
