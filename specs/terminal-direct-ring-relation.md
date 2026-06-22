# Spec: Terminal Direct Ring Relation Mode

| Field       | Value                                                        |
|-------------|--------------------------------------------------------------|
| Author(s)   | Quang Dao                                                    |
| Created     | 2026-05-31                                                   |
| Status      | accepted for implementation; base = `main`, no PR-#165/#166 dependency |
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
Proof-size accounting matches this split: `level_proof_bytes(..., MRowLayout::WithoutDBlock)` prices the terminal `fold_grind_nonce` wire field plus the terminal stage-2 body, and `direct_witness_bytes` prices `final_witness` on the terminal schedule step.
Direct mode still runs fold grinding before the witness is built, so its level-byte total keeps the same `fold_grind_nonce` term and drops only the terminal stage-2 sumcheck bytes.

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

#### Public-opening binding (post-#154)

Direct mode must not rely on `y_terminal` alone to bind the public opening claim.
Since #154 (`specs/y-ring-trace-internalization.md`), `generate_y` zeros the public-output rows of `y` and the public opening claim is bound through the fused trace term inside the terminal stage-2 sumcheck, not through `y` (`crates/akita-types/src/proof/relation.rs:24-60`, `crates/akita-types/src/trace_weight/stage2.rs`).
Direct mode removes terminal stage 2, so it also removes that trace binding.
Checking only `M_terminal * z_terminal == y_terminal` would therefore leave the public opening claim unbound, which is unsound.

Direct mode replaces the stage-2 trace term with an explicit opening-consistency check:
the verifier evaluates the revealed terminal witness multilinear at the bound opening point and compares it to the gamma-weighted public claim (`batched_eval_target_from_opening_batch`).
This reuses the cleartext-opening evaluator already used by the zero-fold / root-direct path (`crates/akita-verifier/src/proof/direct.rs::cleartext_witness_opening_matches`).
So the direct terminal statement is two checks, not one:

```text
1. M_terminal * z_terminal == y_terminal in F[X] / (X^D + 1)   (consistency, commitment, inner-B, A rows)
2. eval(witness, opening_point) == sum_i gamma_i * claim_i      (public opening binding, replacing the stage-2 trace term)
```

Both checks bind the same revealed witness whose bytes were absorbed before any post-witness challenge, so the opening point and the gamma weights are transcript-fixed before the witness is read.

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

```rust
pub enum TerminalProofMode {
    RingSwitchSumcheck,
    DirectRingRelations,
}
```

The mode is **runtime config policy**, not a compile-time feature.
`ProtocolFeatureSet` on `main` holds only `zk` and is built from `ProtocolFeatureSet::current()`, which reads `cfg!(feature = "zk")` (`crates/akita-types/src/instance_descriptor/mod.rs:135-150`).
The terminal mode is chosen by `CommitmentConfig`, so it must not flow through `current()`.
Add it as a runtime field on `SetupSection` (directly or via a small `ProtocolPolicySet`):

```rust
pub struct SetupSection {
    pub decomposition: DecompositionParams,
    pub sis_modulus_family: SisModulusFamily,
    pub setup_seed_digest: DescriptorDigest,
    pub protocol_features: ProtocolFeatureSet,
    pub fold_linf: FoldLinfProtocolBinding,
    pub terminal_proof_mode: TerminalProofMode,   // new
}
```

Canonical descriptor bytes bind the terminal mode with a stable tag:

```text
0 = RingSwitchSumcheck
1 = DirectRingRelations
```

Because the serialized descriptor schema changes, bump `AKITA_INSTANCE_DESCRIPTOR_VERSION` from `1` to `2` (`crates/akita-types/src/instance_descriptor/mod.rs:31`).
`SetupSection::from_parts` gains a `terminal_proof_mode` argument; its single caller `bind_transcript_instance_descriptor` (`crates/akita-config/src/transcript_binding.rs:35-65`) sources it from a new `Cfg::terminal_proof_mode()` hook, and the descriptor round-trip tests (`crates/akita-types/src/instance_descriptor/tests.rs`) update accordingly.

The config surface exposes a single terminal-mode hook on `CommitmentConfig` (`crates/akita-config/src/lib.rs`), defaulting to `RingSwitchSumcheck`:

```rust
fn terminal_proof_mode() -> TerminalProofMode {
    TerminalProofMode::RingSwitchSumcheck
}
```

Default production presets stay `RingSwitchSumcheck` until direct-mode proof sizing, tests, and regenerated tables are accepted.
A test-only or experimental preset overrides the hook to `DirectRingRelations`.

### Proof Shape Changes

The current terminal proof body nests the terminal witness inside the stage-2 payload.
On `main` the terminal body is `AkitaStage2Proof::Terminal(AkitaTerminalStage2Proof { sumcheck_proof, final_witness })` (`crates/akita-types/src/proof/levels.rs:72-98`), wrapped by `TerminalLevelProof { extension_opening_reduction, fold_grind_nonce, stage2 }` (`levels.rs:725-732`).
`final_witness` is therefore a field of the terminal stage-2 struct, not a sibling of the sumcheck.

This spec keeps that nesting (the narrowest cutover) and replaces only the `sumcheck_proof` field with an explicit relation enum:

```rust
pub enum TerminalRelationProof<L> {
    RingSwitchSumcheck(
        #[cfg(not(feature = "zk"))] SumcheckProof<L>,
        #[cfg(feature = "zk")] SumcheckProofMasked<L>,
    ),
    DirectRingRelations,
}

pub struct AkitaTerminalStage2Proof<F, L> {
    pub relation: TerminalRelationProof<L>,
    pub final_witness: CleartextWitnessProof<F>,
}
```

`AkitaStage2Proof::Terminal(AkitaTerminalStage2Proof)` and the existing `final_witness()` / `final_witness_mut()` accessors stay; only the inner sumcheck field becomes the relation enum.
This keeps direct mode an explicit enum variant, not an empty sumcheck (`stage2_sumcheck = []`).

The proof shape mirrors the body:

```rust
pub enum TerminalRelationProofShape {
    RingSwitchSumcheck(SumcheckProofShape),
    DirectRingRelations,
}

pub struct TerminalLevelProofShape {
    pub extension_opening_reduction: Option<ExtensionOpeningReductionShape>,
    pub relation: TerminalRelationProofShape,
    pub final_witness: CleartextWitnessShape,
}
```

This replaces the current `TerminalLevelProofShape { extension_opening_reduction, stage2_sumcheck, final_witness }` (`crates/akita-types/src/proof/shapes.rs:77-84`); the headerless terminal deserializer in `crates/akita-types/src/proof/wire.rs` branches on the relation discriminant to know whether stage-2 sumcheck bytes are present.

Do not reintroduce a `y_rings` shape field.
On-wire `y` was removed in #154 and is not part of `TerminalLevelProof` today.

Under `feature = "zk"`, the `RingSwitchSumcheck` arm carries the masked sumcheck and `DirectRingRelations` is rejected at the config / setup boundary before construction (see ZK Policy).

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
2. build `SegmentTyped(z, e, t)` with zero `r_field_elems` (tiered layouts still emit the hidden `û_concat` digit planes from `build_w_coeffs`; witness sizing follows `w_ring_element_count_with_counts_for_layout_bits`, which adds `u_concat_count` when `tier_split > 1`),
3. absorb the same terminal `e` bytes and remainder bytes,
4. do not call `ring_switch_finalize` (there is no `ring_switch_finalize_terminal`; the terminal sumcheck path calls `ring_switch_finalize` with `MRowLayout::WithoutDBlock`, `crates/akita-prover/src/protocol/ring_switch/finalize.rs:19`),
5. do not sample terminal `alpha` or terminal `tau1`,
6. do not run terminal stage 2,
7. emit a terminal relation proof body of `DirectRingRelations`.

The direct-mode builder should still compute the data needed by direct row verification:

- `e_folded`,
- recomposed inner rows for `t`,
- centered `z_folded` coefficients,
- the commitment-row relation target `y` (consistency / commitment / inner-B / A rows; public-output rows of `y` stay zero per #154),
- row coefficients and row rings already bound by the relation instance.

The prover does not send `y` (removed in #154) and does not send an opening evaluation; the verifier recomputes the relation target and evaluates the revealed witness at the opening point itself.

Do not reconstruct `r` and then drop it.
The point of the mode is to remove quotient work as well as quotient bytes.

### Verifier Changes

Add a verifier-side direct terminal checker.
A likely module is a new sibling of the shared per-fold engine:

```text
crates/akita-verifier/src/protocol/core/terminal_direct.rs
```

(The verifier ring-switch code is the single file `crates/akita-verifier/src/protocol/ring_switch.rs`, not a directory; the terminal dispatch lives in `crates/akita-verifier/src/protocol/core/{verify,root_fold,suffix,fold}.rs`.)

The checker should accept already-decoded, mode-checked terminal data and return `Result<(), AkitaError>`.
Its responsibilities:

1. Validate the `SegmentTyped` layout against the descriptor-bound terminal mode and schedule shape (`r_field_elems == 0`).
2. Decode `z` through the same Golomb-Rice public parameters used by #190.
3. Decode `e` and `t` raw field segments.
4. Reconstruct the row-local ring inputs in the same order as the prover's terminal relation builder (`compute_relation_quotient`, `crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs:518-587`, which is the authoritative reference for row roles and block routing).
5. Evaluate every reduced row equation in `F[X] / (X^D + 1)` and compare to the relation target `y` (check 1 below).
6. Evaluate the revealed witness multilinear at the opening point and compare to the gamma-weighted public claim (check 2 below).

The terminal statement is two checks:

```text
check 1 (ring relation): M_terminal * z_terminal == y_terminal
  - consistency row equals zero,
  - commitment rows equal the current commitment-row semantics
    (y from generate_y + flatten_batched_commitment_rows),
  - inner-B rows equal the current inner-B row semantics,
  - A rows equal zero,
  - public-output rows of y are zero (post-#154; bound by check 2, not by y),
  - D rows are absent.

check 2 (public opening): eval(witness, opening_point) == sum_i gamma_i * claim_i
```

There is no row-routing reuse from the current terminal path: terminal verification is sumcheck-only today (deferred row MLE via `RingSwitchDeferredRowEval::eval_at_point`, `crates/akita-verifier/src/protocol/ring_switch.rs:605-846`), so the all-row direct check is new code modeled on the prover's `compute_relation_quotient` loop.
Reusable primitives are the root-direct cyclotomic helpers (`mat_vec_mul_i8_plain`, `direct_decomposed_inner_rows`, `crates/akita-verifier/src/protocol/core/verify.rs:178-348`) and the cleartext opening evaluator (`crates/akita-verifier/src/proof/direct.rs::cleartext_witness_opening_matches`) for check 2.

The checker must support suffix-terminal and terminal-root surfaces.
It should use explicit incidence and row-count inputs rather than deriving special cases from `num_points == 1`.
Do not duplicate unchecked row-index arithmetic; gate all decode bounds, segment lengths, and row counts before hot arithmetic.

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
  + fold_grind_nonce_bytes
  + direct_witness_bytes(SegmentTyped(z, e, t))
```

`fold_grind_nonce_bytes` is the fixed 4-byte wire field priced by `FOLD_GRIND_NONCE_BYTES` / `level_proof_bytes(..., MRowLayout::WithoutDBlock)` on every fold level, including the terminal.
Direct mode removes the terminal stage-2 sumcheck from that helper, not the grind nonce.

The sumcheck terminal formula stays:

```text
optional_extension_opening_reduction_bytes
  + fold_grind_nonce_bytes
  + terminal_stage2_sumcheck_bytes
  + direct_witness_bytes(SegmentTyped(z, e, t, r))
```

Here `terminal_stage2_sumcheck_bytes` is the stage-2 portion of `level_proof_bytes(..., MRowLayout::WithoutDBlock)` (everything after the grind nonce).
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
- [ ] Direct terminal verifier binds the public opening claim by evaluating the revealed witness at the opening point (replacing the removed stage-2 trace term); tampering the opening claim or the witness rejects.
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

1. Add `TerminalProofMode` to `akita-types`.
2. Add `terminal_proof_mode` to `SetupSection`, bump `AKITA_INSTANCE_DESCRIPTOR_VERSION` to `2`, extend `SetupSection::from_parts`, and thread `Cfg::terminal_proof_mode()` through `bind_transcript_instance_descriptor`.
3. Add `TerminalRelationProof` / `TerminalRelationProofShape`; replace `AkitaTerminalStage2Proof.sumcheck_proof` with `relation` and `TerminalLevelProofShape.stage2_sumcheck` with `relation`; update `wire.rs` headerless terminal decode and `TerminalLevelProof::shape()`; update the verifier expected-shape build in `core/verify.rs`.
4. Add the defaulted `CommitmentConfig::terminal_proof_mode()` hook.
5. Add cross-mode deserialize, descriptor round-trip (v2), and descriptor-mismatch tests.

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

### Slice 4: Verifier Direct Terminal Checker

1. Add the `protocol/core/terminal_direct.rs` verifier module.
2. Validate mode, shape, layout (`r_field_elems == 0`), and incidence before decoding hot row data.
3. Decode `z`, `e`, and `t`.
4. Reconstruct reduced terminal row inputs in the same order as the prover's `compute_relation_quotient`.
5. Check 1: every reduced row in `F[X]/(X^D+1)` equals the relation target `y`.
6. Check 2: evaluate the revealed witness at the opening point and compare to the gamma-weighted public claim (reuse `cleartext_witness_opening_matches`).
7. Wire suffix-terminal verification through it (`core/suffix.rs`).
8. Wire terminal-root verification through it (`core/root_fold.rs`, `core/verify.rs`).
9. Add tamper (`z`, `e`, `t`, commitment-row, opening-claim) and malformed-input no-panic tests.

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
- `crates/akita-types/src/instance_descriptor/mod.rs`, `ProtocolFeatureSet`, `SetupSection`, `AKITA_INSTANCE_DESCRIPTOR_VERSION`, and descriptor binding.
- `crates/akita-config/src/transcript_binding.rs`, `bind_transcript_instance_descriptor` (sole `SetupSection::from_parts` caller).
- `crates/akita-config/src/lib.rs`, `CommitmentConfig` (home of the new `terminal_proof_mode()` hook).
- `crates/akita-prover/src/protocol/ring_switch/coeffs.rs`, `ring_switch_build_w` / `build_w_coeffs` and `RingSwitchTerminalArtifacts`.
- `crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs`, `compute_relation_quotient` (authoritative row roles / block routing for the direct checker).
- `crates/akita-prover/src/protocol/ring_switch/finalize.rs`, `ring_switch_finalize` (alpha / tau1 sampling, skipped in direct mode).
- `crates/akita-prover/src/protocol/core/fold.rs`, `prove_fold` (shared terminal engine), `bind_next_witness_for_ring_switch`, `prove_stage2`.
- `crates/akita-prover/src/protocol/core/root_fold.rs`, `prove_terminal_root_fold_with_params`.
- `crates/akita-prover/src/protocol/core/suffix.rs`, `prove_suffix` (recursive-terminal last level).
- `crates/akita-verifier/src/protocol/core/fold.rs`, `verify_fold` (shared terminal engine), `verify_stage2`.
- `crates/akita-verifier/src/protocol/core/{verify,root_fold,suffix}.rs`, terminal verifier dispatch and expected-shape gating.
- `crates/akita-verifier/src/protocol/ring_switch.rs`, `ring_switch_verifier_terminal`, `RingSwitchDeferredRowEval` (current sumcheck-only terminal replay).
- `crates/akita-verifier/src/protocol/core/verify.rs`, `mat_vec_mul_i8_plain` / `direct_decomposed_inner_rows` (reusable cyclotomic primitives).
- `crates/akita-verifier/src/proof/direct.rs`, `cleartext_witness_opening_matches` (reused for the direct-mode public opening check).
- `crates/akita-verifier/src/stages/stage2.rs`, current terminal stage-2 verifier over decoded direct witness.
- `crates/akita-pcs/tests/transcript_hardening.rs`, terminal transcript and tamper coverage.
