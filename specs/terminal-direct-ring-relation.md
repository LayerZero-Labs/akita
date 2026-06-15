# Spec: Terminal Direct Ring Relation Mode

| Field       | Value                                                        |
|-------------|--------------------------------------------------------------|
| Author(s)   | Quang Dao                                                    |
| Created     | 2026-05-31                                                   |
| Status      | proposed, refreshed after #190 segment-typed terminal tails  |
| PR          | #141                                                         |

## Summary

Akita's transparent folded terminal tail is now segment-typed.
After #190, non-zk terminal folds serialize `CleartextWitnessProof::SegmentTyped` with a Golomb-Rice `z` segment and raw-field `e`, `t`, and `r` segments.
That cutover changed the wire encoding, but it did not change the terminal relation proof.
The prover still computes the terminal quotient `r`, sends it as the raw `r` segment, samples terminal ring-switch aggregation challenges, and proves the terminal relation with a relation-only stage-2 sumcheck.

This spec defines the next tail slice: a terminal-only direct relation mode.
In direct mode, the terminal proof omits the terminal `r` segment and omits the terminal stage-2 sumcheck.
The verifier decodes the segment-typed terminal witness, recomposes the non-quotient terminal row inputs, and checks the reduced ring equations directly in `F[X] / (X^D + 1)` for every terminal row.

The mode is intentionally terminal-only.
Intermediate recursive folds still use committed next witnesses, stage 1, quotient rows, and stage 2 because their stage-2 challenges become the next recursive opening point.

This spec is S1 of the tail-wire umbrella in `specs/tail-wire-encoding.md`.
It does not implement the later terminal `t`-state / `u`-elision slice.

## Intent

### Goal

Add an explicit, transcript-bound terminal relation proof mode:

```rust
pub enum TerminalProofMode {
    RingSwitchSumcheck,
    DirectRingRelations,
}
```

`RingSwitchSumcheck` is the current terminal path:

- terminal witness encoding is `SegmentTyped(z, e, t, r)` in non-zk builds,
- terminal ring-switch `alpha` and `tau1` are squeezed after the terminal witness remainder,
- terminal stage 2 proves the relation-only sumcheck,
- and the verifier expands the terminal witness to the legacy digit stream for stage-2 replay.

`DirectRingRelations` is the new path:

- terminal witness encoding is `SegmentTyped(z, e, t)`,
- the terminal `r` segment is absent and its layout count is zero,
- terminal ring-switch `alpha`, terminal `tau1`, and terminal sumcheck rounds are not squeezed,
- terminal stage 2 is absent from the proof shape and proof bytes,
- and the verifier checks the reduced terminal row equations directly.

The mode must be selected through the same config and descriptor path as other verifier-reachable protocol policy.
The proof shape must also carry the mode, because Akita proof serialization is shape-driven and headerless.

### Non-Goals

- Do not change intermediate recursive folds.
- Do not change the root-direct fast path that sends `CleartextWitnessProof::FieldElements`.
- Do not add a compatibility decoder for old terminal proof bytes.
- Do not overload `MRowLayout::WithoutDBlock` to mean quotient omission.
- Do not implement terminal `t`-state / `u`-elision in this slice.
- Do not support direct terminal mode under `feature = "zk"` in this slice.
- Do not claim a prover, verifier, or Jolt cycle speedup until direct-mode profiling exists.

## Current Baseline After #190

### Terminal Tail Encoding

The current non-zk terminal witness is `CleartextWitnessProof::SegmentTyped`.
Its public layout is `TailSegmentLayout`:

```rust
pub struct TailSegmentLayout {
    pub ring_dimension: usize,
    pub log_basis: u32,
    pub z_first: bool,
    pub z_coords: usize,
    pub e_field_elems: usize,
    pub t_field_elems: usize,
    pub r_field_elems: usize,
    pub logical_num_elems: usize,
}
```

The current realized witness is:

```rust
pub struct SegmentTypedWitness<F: FieldCore> {
    pub layout: TailSegmentLayout,
    pub z_payload: Vec<u8>,
    pub e_fields: FlatRingVec<F>,
    pub t_fields: FlatRingVec<F>,
    pub r_fields: FlatRingVec<F>,
}
```

The segment meanings are:

| Segment | Current encoding | Role |
|---------|------------------|------|
| `z` | Golomb-Rice payload, length-prefixed on the wire | centered folded response |
| `e` | raw field coefficients | terminal `e_folded`, one ring element per opened block |
| `t` | raw field coefficients | recomposed inner rows from the relation witness |
| `r` | raw field coefficients | quotient witness for stage-2 representation |

The direct mode does not introduce a new witness representation.
It reuses `SegmentTypedWitness` with `layout.r_field_elems = 0` and an empty `r_fields`.
This keeps the segment decoder, `CleartextWitnessShape::admits_realized`, and schedule descriptor binding on the same shape family that #190 landed.

### Current Prover Flow

The current folded terminal prover path:

1. Builds the terminal relation instance under `MRowLayout::WithoutDBlock`.
2. Absorbs the terminal `e` segment before sparse-challenge sampling.
3. Samples the sparse challenge and derives fold challenges.
4. Calls `ring_switch_build_w`.
5. Inside `ring_switch_build_w`, calls `compute_relation_quotient`.
6. Builds `SegmentTyped(z, e, t, r)` from retained terminal artifacts.
7. Absorbs the terminal witness remainder.
8. Samples terminal `CHALLENGE_RING_SWITCH` and terminal `CHALLENGE_TAU1`.
9. Runs relation-only stage 2.
10. Serializes `TerminalLevelProof` with terminal stage-2 proof and final witness.

### Current Terminal Proof Body

`TerminalLevelProof` on the wire is headerless and carries only:

- optional extension-opening reduction,
- terminal stage-2 sumcheck (non-zk) or masked sumcheck (zk),
- and `final_witness` (`SegmentTyped` in non-zk folded builds).

Public terminal `y` is not serialized in that payload.
Since #154 (`specs/y-ring-trace-internalization.md`), on-wire `y_ring` / `y_rings` was dropped at every fold level, including the terminal.
The verifier recomputes the public-output row targets from the committed terminal witness (and opening incidence) when checking the relation.
Proof-size accounting matches this split: `level_proof_bytes(..., MRowLayout::WithoutDBlock)` prices only the terminal stage-2 body, and `direct_witness_bytes` prices `final_witness` on the terminal schedule step.

The `r` segment exists only because the terminal relation is still proved through the same quotient-to-sumcheck machinery as intermediate folds.
It is not part of the terminal statement once the verifier checks rows directly.

### Current Verifier Flow

The current folded terminal verifier path:

1. Deserializes the terminal proof using `TerminalLevelProofShape`.
2. Decodes the terminal witness using the scheduled `CleartextWitnessShape`.
3. Splits the terminal witness into transcript `e` bytes and remainder bytes.
4. Absorbs `e`, sparse-challenge context, the sparse challenge, fold challenges, and the remainder.
5. Calls `ring_switch_verifier_terminal`.
6. `ring_switch_verifier_terminal` samples terminal `alpha` and terminal `tau1`.
7. The verifier expands `SegmentTyped` back to the legacy logical digit stream.
8. The terminal stage-2 verifier checks the relation-only sumcheck.

Direct mode replaces steps 5 through 8 with deterministic row checks.

## Direct Mode Semantics

### Ring Relation

The current quotient relation is:

```text
M * z = y + (X^D + 1) * r.
```

That quotient is needed when the row relation is converted into a multilinear sumcheck over coefficient tables.
At the terminal, the verifier receives the clear terminal witness.
It can check the reduced relation directly:

```text
M_terminal * z_terminal == y_terminal in F[X] / (X^D + 1).
```

Direct mode therefore:

- uses the same terminal row semantics as `MRowLayout::WithoutDBlock`,
- sends no `r`,
- samples no terminal ring-switch aggregation challenges,
- and sends no terminal stage-2 proof.

The row layout remains:

- one consistency row,
- one public row per terminal `y` output,
- no D rows,
- the current commitment-row block,
- the current inner-B rows,
- and A rows.

This S1 spec keeps the current terminal statement.
The terminal commitment rows and inner-B rows remain checked in direct mode.
The later S2 terminal `t`-state cutover may remove the terminal commitment/B statement by changing the public terminal state from outer `u` to inner `t`; that is out of scope here.

### Terminal Surfaces

Direct mode must support both terminal proof surfaces:

- suffix-terminal verification, where the last recursive fold usually has `num_points = 1`,
- terminal-root verification, where the root itself is the only fold and can have multiple public rows.

The direct checker must take explicit row-shape data.
It must not infer singleton semantics from `MRowLayout::WithoutDBlock`.
The checker inputs should include:

- `LevelParams`,
- the active ring dimension,
- the decoded `SegmentTypedWitness`,
- recomputed terminal public-output row targets `y` (derived from witness and incidence, not read from the proof body),
- row coefficients,
- opening point and ring-multiplier point,
- commitment rows,
- `OpeningBatch` incidence data,
- `num_public_rows`,
- `num_commitment_groups`,
- and setup matrix views or prepared verifier matrix views.

### Segment-Typed Witness In Direct Mode

Direct mode keeps the #190 segment models:

| Segment | Direct-mode encoding | Notes |
|---------|----------------------|-------|
| `z` | Golomb-Rice | Same `beta_inf -> k` rule and same `z_payload` prefix |
| `e` | `RawField` | Same `e_folded` bytes absorbed before sparse seed |
| `t` | `RawField` | Same recomposed inner rows as current terminal tail |
| `r` | absent | `r_field_elems = 0`, no bytes, no decode |

The shape remains headerless.
The scheduled terminal shape carries `SegmentTypedWitnessShape { layout, z_payload_bytes }`.
For direct mode, the derived `layout.r_field_elems` must be zero.
For sumcheck mode, the derived `layout.r_field_elems` must match the current `m_row_count_for(..., MRowLayout::WithoutDBlock) * D`.

The verifier must reject if the descriptor mode, schedule shape, proof shape, and realized witness disagree about `r_field_elems`.

### Transcript Schedule

The common terminal prefix is unchanged:

```text
descriptor bind
commitment and opening context absorb
terminal e absorb
sparse-challenge context absorb
CHALLENGE_SPARSE_CHALLENGE squeeze
fold challenges
terminal witness remainder absorb
```

In `RingSwitchSumcheck` mode, the continuation is:

```text
CHALLENGE_RING_SWITCH squeeze
CHALLENGE_TAU1 squeezes
terminal stage-2 sumcheck round squeezes
```

In `DirectRingRelations` mode, the continuation is:

```text
direct reduced row checks
```

There is no terminal `CHALLENGE_RING_SWITCH`, terminal `CHALLENGE_TAU1`, or terminal `CHALLENGE_SUMCHECK_ROUND` in direct mode.
There is still no terminal `CHALLENGE_TAU0` in either terminal mode.

The mode must be bound in `AkitaInstanceDescriptor` because the transcript schedules diverge after the terminal witness remainder.
The mode must also be reflected in proof shape because the terminal stage-2 bytes are present in one mode and absent in the other.

## Invariants

- **Terminal-only.** Direct relation mode applies only to terminal folded proofs.
  Intermediate folds keep the quotient and stage-2 path.

- **No quotient segment.** Direct terminal `SegmentTyped` witnesses have `layout.r_field_elems = 0` and empty `r_fields`.
  Sumcheck terminal witnesses keep the nonzero `r` segment.

- **No terminal stage 2.** Direct terminal proof shape and proof bytes contain no terminal stage-2 payload.
  This must be represented by an explicit relation-proof enum, not by an empty sumcheck shape.

- **Mode-bound transcript.** `TerminalProofMode` is included in canonical descriptor bytes.
  A proof produced under one terminal mode must not verify under the other.

- **Shape-driven decoding.** Headerless proof decoding must know from shape whether terminal stage-2 bytes are present and whether the `r` segment is expected.

- **Shared segment layout.** Direct mode should generalize the existing segment-typed layout derivation rather than introduce a parallel decoder.
  `MRowLayout::WithoutDBlock` means D rows are absent; it does not imply quotient omission.

- **Verifier no-panic boundary.** Malformed direct terminal shapes, witness payloads, segment lengths, row counts, or setup layouts reject with `AkitaError` or `SerializationError`.
  New verifier code must avoid `unwrap`, `expect`, unchecked indexing, unchecked slicing, and overflow-prone arithmetic on verifier-reachable inputs.

- **Transparent-only first slice.** Under `feature = "zk"`, selecting direct terminal mode rejects with `InvalidSetup`.
  A masked direct relation design is a separate protocol.

## Design

### Type And Descriptor Changes

Add `TerminalProofMode` to `akita-types`.
A likely home is near `ProtocolFeatureSet` or proof shape definitions, because the mode affects descriptor bytes, schedule digest, proof shape, and verifier replay.

Extend `ProtocolFeatureSet` or add an adjacent protocol-policy section:

```rust
pub struct ProtocolFeatureSet {
    pub zk: bool,
    pub terminal_proof_mode: TerminalProofMode,
}
```

If the repo prefers to keep compile-time features and runtime protocol policy separate, add:

```rust
pub struct ProtocolPolicySet {
    pub terminal_proof_mode: TerminalProofMode,
}
```

Either way, canonical descriptor bytes must bind the terminal mode with a stable tag:

```text
0 = RingSwitchSumcheck
1 = DirectRingRelations
```

The descriptor version should bump if the serialized descriptor schema changes.

The config surface should expose a single terminal-mode hook.
The default production presets stay `RingSwitchSumcheck` until direct-mode proof sizing, tests, and regenerated tables are accepted.
A test-only or experimental preset can select `DirectRingRelations`.

### Proof Shape Changes

Replace the terminal stage-2 shape field with an explicit relation-proof shape:

```rust
pub enum TerminalRelationProofShape {
    RingSwitchSumcheck(SumcheckProofShape),
    DirectRingRelations,
}
```

`TerminalLevelProofShape` then carries:

```rust
pub struct TerminalLevelProofShape {
    pub extension_opening_reduction: Option<ExtensionOpeningReductionProofShape>,
    pub relation: TerminalRelationProofShape,
    pub final_witness: CleartextWitnessShape,
}
```

Do not reintroduce a `y_rings` shape field.
On-wire `y` was removed in #154 and is not part of `TerminalLevelProof` today.

The proof body mirrors it:

```rust
pub enum TerminalRelationProof<L> {
    RingSwitchSumcheck(SumcheckProof<L>),
    DirectRingRelations,
}
```

Under `feature = "zk"`, use a masked sumcheck variant for `RingSwitchSumcheck` and reject `DirectRingRelations`.
Do not encode direct mode as `stage2_sumcheck = []`.

### Terminal Witness Layout Changes

Generalize `tail_segment_layout` with an explicit quotient policy:

```rust
pub enum TerminalQuotientMode {
    IncludeR,
    OmitR,
}
```

Then derive:

```text
r_field_elems =
  IncludeR => m_row_count_for(num_commitment_groups, 0, WithoutDBlock) * D
  OmitR    => 0
```

Likewise:

```text
r_plane_rings =
  IncludeR => m_row_count_for(num_commitment_groups, 0, WithoutDBlock)
              * compute_num_digits_full_field(field_bits, log_basis)
  OmitR    => 0
```

`logical_num_elems` must match the digit-stream length used by verifier row checking.
For direct mode, it excludes the full-field `r` digit planes.
For sumcheck mode, it stays exactly as #190 computes it.

`terminal_direct_witness_shape`, `terminal_direct_witness_shape_for_key`, and `schedule_terminal_direct_witness_shape` should derive the quotient policy from the terminal proof mode.
They should not inspect feature flags or shape variants to infer the mode.

### Prover Changes

Split the terminal witness builder before quotient construction.
The current `ring_switch_build_w` always calls `compute_relation_quotient`.
Direct mode should avoid that call entirely.

Add a terminal-specific build path, for example:

```rust
pub fn ring_switch_build_terminal_witness<F, B, const D: usize>(
    instance: &RingRelationInstance<F, D>,
    witness: RingRelationWitness<F, D>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    lp: &LevelParams,
    mode: TerminalProofMode,
) -> Result<TerminalWitnessBuildOutput<F, D>, AkitaError>
```

In `RingSwitchSumcheck` mode:

1. call `compute_relation_quotient`,
2. retain `r`,
3. build `SegmentTyped(z, e, t, r)`,
4. expand or retain legacy logical digits for stage-2 prover input,
5. run terminal stage 2.

In `DirectRingRelations` mode:

1. do not call `compute_relation_quotient`,
2. build `SegmentTyped(z, e, t)` with zero `r_field_elems`,
3. absorb the same terminal `e` bytes and remainder bytes,
4. do not call `ring_switch_finalize_terminal`,
5. do not sample terminal `alpha` or terminal `tau1`,
6. do not run terminal stage 2,
7. emit a terminal relation proof body of `DirectRingRelations`.

The direct-mode builder should still compute the data needed by direct row verification:

- `e_folded`,
- recomposed inner rows for `t`,
- centered `z_folded` coefficients,
- recomputed public-output row targets `y` for row-equality checks,
- row coefficients and row rings already bound by the relation instance.

Do not reconstruct `r` and then drop it.
The point of the mode is to remove quotient work as well as quotient bytes.

### Verifier Changes

Add a verifier-side direct row checker.
A likely module is:

```text
crates/akita-verifier/src/protocol/ring_switch/terminal_direct.rs
```

The checker should accept already-decoded, mode-checked terminal data and return `Result<(), AkitaError>`.
Its responsibilities:

1. Validate the `SegmentTyped` layout against the descriptor-bound terminal mode and schedule shape.
2. Decode `z` through the same Golomb-Rice public parameters used by #190.
3. Decode `e` and `t` raw field segments.
4. Reconstruct the row-local ring inputs in the same order as the current terminal relation builder.
5. Evaluate every reduced row equation in `F[X] / (X^D + 1)`.
6. Compare row outputs against the terminal row targets.

The row targets are:

- consistency row equals zero,
- public rows equal the recomputed `y` targets from witness and incidence,
- commitment rows equal the current commitment-row semantics,
- inner-B rows equal the current inner-B row semantics,
- A rows equal zero,
- D rows are absent.

The checker must support suffix-terminal and terminal-root surfaces.
It should use explicit incidence and row-count inputs rather than deriving special cases from `num_points == 1`.

Prefer reusing existing verifier helpers when they already validate bounds and row routing.
If a helper currently lives under a root-direct-only module but has the right terminal row semantics, move it into a shared verifier module.
Do not duplicate unchecked row-index arithmetic.

### Proof Size And Planner Changes

Direct mode removes:

```text
terminal_stage2_sumcheck_bytes
  + serialized_raw_field_bytes(r)
```

Under #190, the `r` saving is:

```text
r_field_elems * field_bytes(F::modulus_bits())
```

not:

```text
ceil(r_digit_count * bits_per_elem / 8)
```

The latter was the pre-#190 packed-digit estimate and is now stale for non-zk tails.

Terminal level bytes follow the same split as today: proof body (`level_proof_bytes`) plus scheduled witness (`direct_witness_bytes`).
Neither term includes on-wire `y`; that was removed in #154.

The direct terminal level byte formula becomes:

```text
optional_extension_opening_reduction_bytes
  + direct_witness_bytes(SegmentTyped(z, e, t))
```

The sumcheck terminal formula stays:

```text
optional_extension_opening_reduction_bytes
  + terminal_stage2_sumcheck_bytes
  + direct_witness_bytes(SegmentTyped(z, e, t, r))
```

Here `terminal_stage2_sumcheck_bytes` is what `level_proof_bytes(..., MRowLayout::WithoutDBlock)` already prices.
`direct_witness_bytes` is the terminal schedule step's witness accounting, matching `segment_typed_witness_upper_bound_bytes` for the bound shape.

Planner and schedule materialization must derive the same terminal witness shape as the prover.
If direct mode is exposed by production configs, generated schedule tables must be regenerated under the direct-mode policy and protected by the drift guard.
Until then, direct mode should be test-only or runtime-DP-only with explicit proof-size assertions.

### ZK Policy

Direct terminal mode is transparent-only in this spec.
Under `feature = "zk"`, selecting `DirectRingRelations` must reject with `InvalidSetup` before proof construction or verification.

This is a protocol boundary, not merely missing code.
The current ZK terminal path relies on:

- `stage2_sumcheck_proof_masked`,
- verifier-side relation-claim masking,
- and B-side blinding carried through the terminal stage-2 row-evaluation path.

On-wire `y_ring` masking was removed with #154; the fused trace term in stage 2 binds the public opening instead.

Removing terminal stage 2 removes that masking mechanism.
A future ZK direct mode needs a separate masked direct-row relation argument.

## Evaluation

### Acceptance Criteria

- [ ] `TerminalProofMode` exists in a verifier-reachable shared crate.
- [ ] `AkitaInstanceDescriptor::canonical_bytes` binds the terminal proof mode.
- [ ] `CommitmentConfig` or the config-derived policy selects the terminal proof mode.
- [ ] `TerminalLevelProofShape` and `TerminalLevelProof` use an explicit terminal relation proof enum.
- [ ] Direct terminal proof bytes contain no terminal stage-2 sumcheck.
- [ ] Direct terminal `SegmentTyped` shape has `r_field_elems = 0`.
- [ ] Direct terminal prover does not call `compute_relation_quotient`.
- [ ] Direct terminal prover does not call terminal ring-switch finalization.
- [ ] Direct terminal transcript contains no terminal `CHALLENGE_RING_SWITCH`, `CHALLENGE_TAU1`, or terminal sumcheck round challenge.
- [ ] Direct terminal verifier checks every reduced terminal row.
- [ ] Direct terminal verifier supports suffix-terminal and terminal-root surfaces.
- [ ] Cross-mode descriptor or proof-shape mismatch rejects before row checking.
- [ ] Direct mode under `feature = "zk"` rejects with `InvalidSetup`.
- [ ] Proof-size accounting reports exact serialized proof bytes for direct and sumcheck terminal modes.
- [ ] Existing sumcheck terminal mode remains tested until production configs intentionally cut over.

### Tests

Add or update `akita-types` tests:

- `TerminalProofMode` descriptor tags are stable.
- `TerminalLevelProofShape` round-trips with `RingSwitchSumcheck`.
- `TerminalLevelProofShape` round-trips with `DirectRingRelations`.
- Sumcheck terminal bytes fail to deserialize under direct terminal shape.
- Direct terminal bytes fail to deserialize under sumcheck terminal shape.
- Direct `SegmentTypedWitnessShape` rejects nonzero `r_field_elems` when descriptor mode is direct.
- Sumcheck `SegmentTypedWitnessShape` rejects zero `r_field_elems` when descriptor mode is sumcheck.

Add or update `akita-prover` / `akita-verifier` tests:

- Direct row checker accepts a deterministic small suffix-terminal instance produced by prover code.
- Direct row checker accepts a deterministic terminal-root instance with multiple public rows.
- Tampering `z` rejects.
- Tampering `e` rejects, including when public-output row targets disagree with opening incidence.
- Tampering `t` rejects.
- Tampering commitment-row input rejects.
- Truncated `z_payload` rejects without panic.
- Malformed raw field segment lengths reject without panic.

Add or update `akita-pcs` e2e tests:

- Direct terminal proves and verifies for one dense profile.
- Direct terminal proves and verifies for one one-hot profile.
- Direct terminal proves and verifies for one terminal-root batched profile if that surface is enabled.
- Direct terminal proof shape has no terminal stage-2 proof.
- Direct terminal proof size equals planner exact bytes.
- Sumcheck terminal mode still proves and verifies under the default preset.

Add transcript tests:

- Direct terminal event order has the common terminal prefix and stops after the witness remainder.
- Direct terminal event stream contains no terminal `CHALLENGE_RING_SWITCH`.
- Direct terminal event stream contains no terminal `CHALLENGE_TAU1`.
- Direct terminal event stream contains no terminal `CHALLENGE_SUMCHECK_ROUND`.
- Sumcheck terminal event order remains unchanged.

Add ZK tests:

- A direct terminal config under `feature = "zk"` rejects with `InvalidSetup`.
- Existing ZK sumcheck terminal tests remain green.

Minimum local validation:

```bash
cargo fmt -q
cargo test -p akita-types
cargo test -p akita-pcs ring_switch
cargo test -p akita-pcs akita_e2e
cargo test -p akita-pcs transcript_hardening
cargo test -p akita-pcs zk --features zk
```

If production tables are regenerated:

```bash
cargo test -p akita-config generated_tables
cargo test -p akita-config runtime_fallback
```

## Performance

Direct mode has two proof-byte savings after #190:

```text
stage2_sumcheck_bytes
  + r_field_elems * field_bytes(F::modulus_bits())
```

This supersedes older packed-digit estimates.
Before #190, the terminal `r_hat` bytes were priced as balanced digit planes.
After #190, non-zk terminal `r` is raw field data, so direct mode's r-drop may save more bytes than the old `ceil(r_digits * bits_per_elem / 8)` estimate.

Expected runtime effects:

- The prover skips terminal quotient construction.
- The prover skips terminal stage 2.
- The verifier skips terminal ring-switch aggregation challenge preparation.
- The verifier skips terminal stage 2.
- The verifier adds deterministic all-row reduced ring checks.

The net verifier and Jolt cycle effect is empirical.
Direct row checks are deterministic and all-row, while the current path verifies a randomized sumcheck.
The implementation must add profile breakdown fields before claiming speedup.

Profile reporting should include:

- terminal proof mode,
- terminal witness encoding,
- `z_payload_bytes`,
- `e_bytes`,
- `t_bytes`,
- `r_bytes` or `0`,
- terminal stage-2 bytes or `0`,
- direct row-check time,
- and terminal witness decode time.

## Implementation Strategy

### Slice 0: Spec And Cross-Links

1. Land this refreshed spec as the S1 direct-relation contract.
2. Keep `specs/tail-wire-encoding.md` as the umbrella.
3. Update `specs/terminal-fold-cutover.md` to say #190 changed the terminal tail from `PackedDigits` to segment-typed encoding in non-zk builds.
4. Update `specs/transcript-immediate-fixes.md` to use `e` / `e_folded`, `z_folded`, and segment-typed terminology instead of stale `w_hat` / `z_pre` wording where it refers to the transparent tail.

### Slice 1: Mode And Shape Plumbing

1. Add `TerminalProofMode`.
2. Bind it in descriptor bytes.
3. Add terminal relation proof shape and proof body enums.
4. Thread mode through config policy.
5. Add cross-mode deserialize and descriptor mismatch tests.

This slice should not change prover behavior yet.
It should keep all production configs on `RingSwitchSumcheck`.

### Slice 2: Direct Segment Layout

1. Add `TerminalQuotientMode::{IncludeR, OmitR}`.
2. Generalize `tail_segment_layout` and `segment_typed_witness_shape`.
3. Derive `OmitR` from `TerminalProofMode::DirectRingRelations`.
4. Add tests for `r_field_elems = 0` and `logical_num_elems` in direct mode.
5. Update proof-size helpers to price both modes.

This slice should still be able to build sumcheck terminal witnesses unchanged.

### Slice 3: Prover Direct Terminal Path

1. Split terminal witness construction away from `ring_switch_build_w`.
2. In direct mode, build `SegmentTyped(z, e, t)` without computing `r`.
3. Absorb terminal transcript bytes through the same `terminal_transcript_parts` path.
4. Emit `TerminalRelationProof::DirectRingRelations`.
5. Reject direct mode under `feature = "zk"`.
6. Add prover-side shape and proof-size assertions.

### Slice 4: Verifier Direct Row Checker

1. Add `terminal_direct` verifier module.
2. Validate mode, shape, layout, and incidence before decoding hot row data.
3. Decode `z`, `e`, and `t`.
4. Reconstruct reduced terminal row inputs.
5. Check every row target directly in the cyclotomic ring.
6. Wire suffix-terminal verification through it.
7. Wire terminal-root verification through it.
8. Add tamper and malformed-input tests.

### Slice 5: E2E, Planner, And Profiling

1. Add a test-only direct terminal config.
2. Run e2e dense, one-hot, and terminal-root coverage.
3. Assert exact proof-size accounting against serialized proof bytes.
4. Add profile report fields for direct mode.
5. Measure verifier and Jolt decode/check costs.
6. Regenerate production tables only if production configs select direct mode.

### Cutover Decision

Keep production presets on `RingSwitchSumcheck` until all direct-mode tests and profile checks pass.
Once direct mode is accepted, cut production presets over in one pass.
Do not keep old and new terminal modes in a hidden fallback path.
If both modes remain exposed, they should be explicit config policy choices with separate tests and descriptor tags.

## Alternatives Considered

**Drop only terminal stage 2, keep `r`.**
This preserves witness shape but sends quotient data that direct row checking does not use.
It also keeps the terminal witness larger for no soundness benefit.

**Drop `r`, but keep terminal `alpha` and `tau1`.**
This preserves part of the current transcript schedule but leaves unused challenges in the proof transcript.
Direct mode should remove the whole terminal quotient aggregation layer.

**Use `MRowLayout::WithoutDBlock` as the direct-mode marker.**
Rejected.
`WithoutDBlock` already means D rows are absent.
Sumcheck terminal mode also uses `WithoutDBlock` and still needs `r`.

**Implement terminal `t`-state / `u`-elision at the same time.**
Rejected for this slice.
The direct relation mode is already a coherent S1 improvement on top of #190.
The `t`-state cutover changes the terminal public statement and should be implemented as S2 with its own soundness tests.

**Re-land the archived implementation branch.**
Rejected.
The archive predates #190's segment-typed tail and multiple prover/verifier flow refactors.
It is useful as design evidence only.
The implementation should be fresh and based on current `SegmentTyped` surfaces.

## References

- `specs/tail-wire-encoding.md`, umbrella tail-wire spec and #190 implementation scope.
- `specs/terminal-fold-cutover.md`, implemented terminal witness binding and D-block drop.
- `specs/transcript-immediate-fixes.md`, terminal transcript ordering.
- `crates/akita-types/src/proof/direct_witness.rs`, `CleartextWitnessProof` and terminal witness shape selection.
- `crates/akita-types/src/proof/tail_segments.rs`, `TailSegmentLayout`, `SegmentTypedWitness`, and segment expansion.
- `crates/akita-types/src/proof/levels.rs`, `TerminalLevelProof`.
- `crates/akita-types/src/proof/shapes.rs`, `TerminalLevelProofShape`.
- `crates/akita-types/src/proof/wire.rs`, proof serialization and shape-driven terminal deserialization.
- `crates/akita-types/src/schedule.rs`, `DirectStep` and schedule descriptor bytes.
- `crates/akita-types/src/instance_descriptor.rs`, `ProtocolFeatureSet` and descriptor binding.
- `crates/akita-prover/src/protocol/ring_switch/coeffs.rs`, current terminal quotient and segment-typed artifact construction.
- `crates/akita-prover/src/protocol/flow/suffix.rs`, current terminal stage-2 prover path.
- `crates/akita-prover/src/protocol/flow/root_fold.rs`, terminal-root prover path.
- `crates/akita-verifier/src/protocol/ring_switch.rs`, current terminal verifier replay and challenge sampling.
- `crates/akita-verifier/src/protocol/levels.rs`, terminal verifier dispatch.
- `crates/akita-verifier/src/stages/stage2.rs`, current terminal stage-2 verifier over decoded direct witness.
- `crates/akita-pcs/tests/transcript_hardening.rs`, terminal transcript and tamper coverage.
