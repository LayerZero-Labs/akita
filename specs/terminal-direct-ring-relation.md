# Spec: Terminal Direct Ring Relation Mode

Author(s): Quang Dao

Created: 2026-05-31

Status: approved

PR: #141

## Rebase onto main (2026-06-05)

This spec was preserved from PR #141 branch tip `8599084d` and refreshed against current `main` (`f7bb582a`) before re-implementation.

Identifier updates applied in this refresh:

- Core protocol naming (#143): `RingRelationProver`, `compute_relation_quotient`, `MRowLayout::{WithDBlock,WithoutDBlock}`, `CleartextWitnessProof`.
- Witness notation (#150): `e_hat`, `z_folded_rings` (no `w_hat` / `z_pre` in `crates/`).
- Planner (#139): `CommitmentConfig::runtime_schedule` / `get_schedule`; drift guard `generated_schedule_tables_match_find_schedule`.
- Proof-size SSOT: `crates/akita-types/src/proof_size.rs`.
- Fp16 proof-size artifact row removed (#149).

Implementation reference (never commit): worktree `preservation-NEVER-COMMIT/pr-141-terminal-direct/`.

## Summary

Akita's terminal recursive fold already ships the final recursive witness in cleartext and omits the D-block rows (`v = D * e_hat`) from the terminal relation.
It still computes the per-row ring-switch quotient `r`, appends decomposed `r_hat` digits to the terminal `final_witness`, samples terminal ring-switch aggregation challenges, and proves the terminal relation with a relation-only stage-2 sumcheck.

This spec proposes a terminal-only direct relation mode.
In this mode the terminal proof omits both the stage-2 sumcheck and the `r_hat` quotient digits.
The verifier decodes the cleartext terminal witness, recomposes its non-quotient segments, and checks the terminal ring equations directly in `F[X] / (X^D + 1)` for every terminal row.

The mode is intentionally terminal-only.
Intermediate recursive folds continue to use the committed next witness, `v`, stage-1, ring-switch quotient digits, and stage-2 sumcheck because their stage-2 challenges are the next recursive opening point.

## Intent

### Goal

Add a transcript-bound terminal proof mode that replaces terminal ring-switch quotient transmission and terminal stage-2 sumcheck with direct verifier checks of the terminal ring relations.

The implementation must touch these protocol surfaces:

- `CommitmentConfig`, or an equivalent config-derived protocol policy, gains a terminal proof mode.
- `AkitaInstanceDescriptor` binds the mode so prover and verifier cannot replay different transcript schedules.
- `TerminalLevelProof` and `TerminalLevelProofShape` represent either the existing relation-only sumcheck terminal payload or the new direct-relation terminal payload.
- Terminal witness sizing excludes `r_hat` in direct mode.
- Transparent terminal prover construction emits `z_folded_rings`, `e_hat`, and `t_hat` segments only.
- Terminal verifier construction decodes the non-quotient final witness and checks the terminal rows directly across both suffix-terminal and terminal-root verification entrypoints.
- Proof-size and planner accounting use the selected terminal mode.

### Current Terminal Protocol

The current terminal recursive fold is implemented in `crates/akita-prover/src/protocol/flow/recursive.rs`.

Current prover flow:

1. `RingRelationProver::new_recursive_multipoint` builds terminal `e_hat`, skips `v`, absorbs terminal `e_hat`, samples fold challenges, computes `z_folded_rings`, and builds terminal `y`.
2. `prove_terminal_fold_level_from_ring_relation` computes `terminal_witness_segment_layout`.
3. It calls `ring_switch_build_w`, which calls `compute_relation_quotient`.
4. In transparent builds, `build_w_coeffs` emits `z_folded_rings`, `e_hat`, `t_hat`, and finally `r_hat`.
   Under `feature = "zk"`, it also emits terminal B-side blinding planes and keeps D-side blinding empty because terminal layout drops D rows.
5. The full `logical_w` is packed into `CleartextWitnessProof::PackedDigits`.
6. `ring_switch_finalize_terminal` absorbs the final-witness remainder and samples `alpha` and `tau1`.
7. The prover runs stage-2 in relation-only mode with `batching_coeff = 0`, `s_claim = 0`, and dummy `stage1_point`.
8. `TerminalLevelProof` serializes `y_rings`, optional extension-opening reduction, terminal stage-2 sumcheck, and `final_witness`.

Current verifier flow:

1. `verify_one_level_inner` checks the opening claim and y-ring relation.
2. It computes terminal `w_len` from the scheduled terminal direct step.
3. It splits terminal `final_witness` into `e_hat` and remainder transcript parts.
4. It derives fold challenges with `MRowLayout::WithoutDBlock`.
5. It calls `ring_switch_verifier_terminal`, which absorbs the remainder, samples `alpha` and `tau1`, and prepares deferred row evaluation.
6. It builds an `AkitaStage2Verifier` over the direct final witness.
7. It verifies the terminal stage-2 sumcheck.

The current proof shape makes terminal stage-2 mandatory.
`TerminalLevelProof` stores `stage2_sumcheck` or `stage2_sumcheck_proof_masked`, and `TerminalLevelProofShape` stores a `stage2_sumcheck` shape.
The wire format serializes terminal `y_rings`, optional extension-opening reduction, stage-2 sumcheck, and `final_witness` in that order.

### Current Terminal ZK Path

Current `feature = "zk"` terminal proofs do not hide the terminal tail by omitting it.
They keep a cleartext packed `final_witness` and add masking in three places:

1. The proof carries a top-level `ZkHidingProof` whose `u_blind`, `hiding_witness`, and short `b_blinding_digits` commit to all one-time pads used by the masked opening protocol.
2. Terminal public `y_rings` are masked on the wire as `y_rings_masked = y_rings + y_garbage`.
3. Terminal stage-2 is still present and is serialized as `stage2_sumcheck_proof_masked`.
   The verifier reconstructs `y` masks and a `relation_claim_mask` from the same hiding witness and verifies the masked relation-only stage-2 proof.

The terminal tail layout under `feature = "zk"` is therefore:

- if `z_first`, `z_folded_rings | e_hat | t_hat | b_blinding | r_hat`,
- otherwise, `e_hat | t_hat | b_blinding | z_folded_rings | r_hat`.

Terminal `d_blinding` is absent because `MRowLayout::WithoutDBlock` drops D rows and terminal `v`.
The transcript still binds logical terminal `e_hat` first and then the remainder, so the remainder contains `z_folded_rings`, `t_hat`, terminal `b_blinding`, and `r_hat` in the chosen segment order.

### Proposed Terminal Direct Protocol

In direct terminal mode, the terminal fold keeps the same public y-ring opening check and the same fold-challenge derivation, but removes the quotient and aggregation layer.

Prover flow:

1. Build terminal `e_hat` and absorb it before sampling fold challenges, as today.
2. Sample fold challenges, compute `z_folded_rings`, and build terminal `y`, as today.
3. Build a transparent terminal direct witness with the same segment ordering as the current transparent terminal witness, but omit the `r_hat` suffix.
4. Pack that non-quotient witness into `CleartextWitnessProof::PackedDigits`.
5. Absorb the final-witness remainder.
6. Do not call `compute_relation_quotient`.
7. Do not call `ring_switch_finalize_terminal`.
8. Do not sample `CHALLENGE_RING_SWITCH`, `CHALLENGE_TAU1`, or `CHALLENGE_SUMCHECK_ROUND` for terminal stage-2.
9. Emit a terminal proof payload with no stage-2 sumcheck field.

Verifier flow:

1. Decode and validate the terminal `final_witness`.
2. Split it into the same `e_hat` and remainder transcript parts.
3. Absorb `e_hat`, derive fold challenges, and absorb the remainder in the same relative positions as the prover.
4. Do not sample terminal ring-switch or terminal sumcheck challenges.
5. Recompose the non-quotient witness segments into terminal row inputs using a full terminal segment layout, not only the transcript `e_hat` slice.
6. For each terminal row, evaluate the row directly in the cyclotomic ring.
7. Compare the direct row output to the terminal `y` row layout:
   consistency row is zero,
   public rows are `y_rings`,
   B rows follow the current singleton or multipoint commitment-row semantics,
   A rows are zero,
   and D rows are absent.

### Terminal Verification Surfaces

Akita has two verifier entrypoints that consume `TerminalLevelProof`:

- the recursive suffix-terminal path, which currently expects exactly one terminal `y_ring`,
- the terminal-root path for the 1-fold case, which can have `num_points > 1` and `num_public_rows != num_claims`.

Direct mode must support both surfaces.
The direct checker API therefore needs explicit `num_claims`, `num_points`, `num_public_rows`, `num_polys_per_point`, `claim_to_point`, `claim_poly_indices`, and flattened commitment-row inputs.
It must not infer singleton semantics from `MRowLayout::WithoutDBlock`.
The existing root-direct proof path that bypasses recursive folding remains out of scope.

### Ring Relation

The existing quotient relation can be stated over unreduced products as:

```text
M * z = y + (X^D + 1) * r.
```

The direct terminal verifier checks the reduced relation instead:

```text
M_terminal * z_terminal == y_terminal in F[X] / (X^D + 1).
```

The quotient `r` is only needed when the proof system converts this row relation into a multilinear sumcheck over coefficient tables.
At the terminal, the verifier has the cleartext digit witness, the public commitment rows, the opening point, the ring-multiplier point, the setup matrices, and the public y rows.
It can therefore check the reduced ring equation directly and does not need the quotient witness.

The direct checker must use `MRowLayout::WithoutDBlock` and the same terminal row semantics as the current quotient row builder in `crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs`.
That layout contains:

- one consistency row,
- one public row per terminal y output,
- no D rows,
- `n_b * num_points` B rows,
- `n_a` A rows.

For the current recursive terminal suffix, `num_points = 1` and `num_public_rows = num_claims = 1`.
The implementation should still use explicit counts, not hard-code singleton assumptions, because batched-root terminal handling already has separate shape logic.

The direct checker is normative if and only if it mirrors the current terminal row construction up to reduction modulo `X^D + 1`.
For each row index in terminal layout it must:

1. decode the packed witness with a full terminal segment layout,
2. reconstruct the row-local `z_folded_rings`, `e_hat`, `t_hat`, and any future explicit blinding segments in the same block ordering used by `build_w_coeffs`,
3. evaluate the reduced cyclotomic row polynomial, and
4. compare it to the expected row target.

Those row targets are:

- consistency row equals zero,
- public rows equal `y_rings`,
- B rows follow the current commitment-row semantics, not a simplified `commitment_u` shortcut,
- A rows equal zero,
- D rows are absent.

The current quotient builder has a real singleton-vs-multipoint B-row fork.
If `commitment_row_count == n_b` and `num_points == 1`, the direct checker may use the current singleton relation-B-row path.
Otherwise it must mirror the current `repeated_b_commitment_rows` semantics and compare against flattened commitment rows in the same incidence order used by terminal-root verification.

`TerminalWitnessSegmentLayout` (`crates/akita-types/src/proof/terminal_witness.rs:10`) is not enough for this checker by itself because it only describes the transcript `e_hat` slice.
Direct row checking needs the full segment layout already used on the verifier side, `RingSwitchSegmentLayout` in `crates/akita-verifier/src/protocol/ring_switch.rs:140`, produced by the prepared-layout `segment_layout()` method at `crates/akita-verifier/src/protocol/ring_switch.rs:605`.
That type already pins `offset_w`, `offset_t`, `offset_z`, `offset_r`, the `z_first` ordering, and (under `feature = "zk"`) the blinding offsets.
Direct mode should generalize it so that `offset_r` is absent in `OmitRHat` mode rather than introducing a parallel layout type, keeping a single source of truth for `z_folded_rings`, `e_hat`, `t_hat`, blinding, and the omitted-`r_hat` policy.

### Invariants

- **Terminal-only.**
  Direct relation mode applies only to `TerminalLevelProof` and terminal root payloads.
  Intermediate `AkitaLevelProof` remains unchanged.

- **No quotient digits.**
  Direct terminal `final_witness` contains no `r_hat` suffix.
  The scheduled terminal direct witness shape and runtime witness length must agree with that omission.

- **No terminal stage-2 transcript.**
  Direct terminal mode must not serialize, deserialize, prove, verify, or transcript-replay the terminal stage-2 sumcheck.
  It must not squeeze terminal `CHALLENGE_RING_SWITCH`, `CHALLENGE_TAU1`, or `CHALLENGE_SUMCHECK_ROUND`.

- **Witness binding remains before challenge use.**
  The logical terminal `e_hat` segment is absorbed before fold-challenge sampling.
  The final-witness remainder is absorbed before direct row verification starts.

- **Descriptor binding.**
  The terminal proof mode is included in canonical instance descriptor bytes.
  A proof generated in sumcheck-terminal mode must not verify under direct-terminal mode, and the reverse must also fail.

- **Shape-driven deserialization stays unambiguous.**
  Headerless proof decoding must know whether a terminal stage-2 payload is present from the proof shape.
  An empty sumcheck shape must not be used as an implicit mode marker.

- **Two terminal verifier surfaces stay aligned.**
  Direct mode must specify how suffix-terminal verification and terminal-root verification share the same row-check contract while still passing explicit `num_points`, `num_public_rows`, and commitment-row routing data.

- **Verifier no-panic boundary.**
  Malformed terminal direct shapes, witness lengths, segment ranges, row counts, and packed digit payloads must return `AkitaError` or `SerializationError`.
  New verifier paths must avoid `unwrap`, `expect`, unchecked slicing, and overflow-prone arithmetic.

- **No compatibility shim.**
  The repo allows breaking protocol changes.
  The mode should be explicit and fully wired, not hidden behind deprecated aliases or auto-detection.

- **ZK is not silently supported.**
  If direct terminal mode is implemented for non-ZK first, `feature = "zk"` must reject it with `InvalidSetup`.
  A ZK implementation needs a separate masked direct-relation design.

- **Planner rollout is explicit.**
  Direct terminal mode remains test-only until schedule tables and exact proof-byte accounting have been updated together.

### Non-Goals

- Do not change intermediate recursive folds.
- Do not remove extension-opening reduction sumchecks.
- Do not change the root-direct proof path that bypasses recursive folding entirely.
- Do not alter SIS parameter derivation except for mode-aware proof-size and terminal witness-size accounting.
- Do not add backward-compatibility decoding for old terminal proof bytes.
- Do not claim a prover or verifier wall-time speedup until direct row-check timings are measured after implementation.

## Evaluation

### Acceptance Criteria

- `TerminalProofMode` or equivalent policy exists and is selected from the shared config path.
- The mode is bound in `AkitaInstanceDescriptor::canonical_bytes`.
- The mode has an explicit terminal relation discriminant in both `TerminalLevelProofShape` and `TerminalLevelProof`.
- Direct terminal proof serialization contains no terminal stage-2 sumcheck bytes.
- Direct terminal final witness shape excludes `r_hat`.
- The direct checker supports both suffix-terminal and terminal-root verification.
- Prover and verifier agree on terminal `final_w_len` in direct mode.
- Verifier rejects a direct terminal proof if any non-quotient witness segment is tampered.
- Verifier rejects a direct terminal proof if any terminal `y_rings` element is tampered.
- Verifier rejects a direct terminal proof if the descriptor or proof shape says sumcheck-terminal mode.
- Existing sumcheck-terminal mode remains testable unless the implementation intentionally makes direct mode the only terminal mode.
- The proof-size estimator reports exact serialized proof bytes for both modes.
- Direct mode is gated to a test-only config until generated tables and exact proof-byte accounting are refreshed together.
- Under `feature = "zk"`, selecting direct terminal mode rejects with `InvalidSetup`.

### Testing Strategy

Add tests in `crates/akita-pcs/tests/ring_switch.rs`:

- Terminal-layout direct row relation matches `MRowLayout::WithoutDBlock` public rows for a small deterministic instance.
- Terminal-root direct row relation matches the current multipoint B-row semantics for a small deterministic batched instance.
- Direct row checker rejects a modified `e_hat` digit.
- Direct row checker rejects a modified `t_hat` digit.
- Direct row checker rejects a modified `z_folded_rings` digit.

Add tests in `crates/akita-pcs/tests/akita_e2e.rs`:

- Direct terminal mode proves and verifies for one dense profile and one one-hot profile.
- Direct terminal mode proves and verifies for one terminal-root batched profile.
- Direct terminal proof shape has no terminal stage-2 sumcheck.
- Proof size equals planner `exact_proof_bytes`, asserted under the test-only direct config so the check does not depend on refreshed production schedule tables.
- Tampering `final_witness` rejects.
- Tampering terminal `y_rings` rejects.

Add tests in `crates/akita-types/src/proof/tests.rs`:

- `TerminalLevelProofShape` round-trips direct terminal mode.
- Deserializing a sumcheck-terminal proof with direct-terminal shape rejects.
- Deserializing a direct-terminal proof with sumcheck-terminal shape rejects.
- `TerminalWitnessSegmentLayout` remains correct when the witness has no `r_hat` suffix.

Add tests in `crates/akita-types/src/schedule.rs`:

- Terminal direct witness bytes exclude `r_count`.
- Terminal direct level bytes exclude stage-2 bytes.
- Planned terminal direct payload matches serialized proof payload.

Add transcript and tamper tests in `crates/akita-pcs/tests/transcript_hardening.rs` and helper coverage in `crates/akita-pcs/tests/common/mod.rs`:

- Direct terminal event order includes descriptor binding, commitment, evaluation claims, terminal `e_hat`, fold challenges, and terminal remainder.
- Direct terminal event order excludes `CHALLENGE_RING_SWITCH`, `CHALLENGE_TAU1`, and terminal `CHALLENGE_SUMCHECK_ROUND`.
- Sumcheck-terminal event order remains unchanged when that mode is selected.
- Direct terminal descriptor mismatch rejects before row checking begins.
- Direct terminal tamper coverage includes `e_hat`, `t_hat`, `z_folded_rings`, packed payload shape, and `y_rings`.

Add ZK rejection tests in `crates/akita-pcs/tests/zk.rs`:

- Selecting direct terminal mode under `feature = "zk"` rejects with `InvalidSetup`.

Feature combinations:

- Default `parallel` feature must pass.
- `no-default-features` must pass if direct terminal mode is available there.
- `zk` must pass with direct terminal mode disabled and with a regression test that the selection rejects cleanly.

Minimum acceptance commands:

- `cargo test -p akita-types`
- `cargo test -p akita-pcs ring_switch`
- `cargo test -p akita-pcs akita_e2e`
- `cargo test -p akita-pcs transcript_hardening`
- `cargo test -p akita-pcs zk --features zk`
- `cargo test -p akita-planner generated_schedule_tables_match_find_schedule`, if production tables are refreshed

### Performance

The proof-size savings are exact once the terminal level shape and witness length are known.
They are the sum of:

```text
stage2_sumcheck_bytes
  + ceil(r_ring_elements * D * tail_bits_per_elem / 8).
```

For recursive terminal levels in the current transparent benchmark matrix:

```text
r_ring_elements =
  next_w_ring
  - num_blocks * num_digits_open
  - num_blocks * n_a * num_digits_open
  - block_len * num_digits_commit * num_digits_fold.
```

The formula matches current transparent witness construction because the current transparent terminal witness is:

```text
z_folded_rings | e_hat | t_hat | r_hat
```

where `r_hat` has `r_ring_elements` ring elements, each with `D` packed digits.
The concrete segment order is `z_first`-dependent in `build_w_coeffs` (`crates/akita-prover/src/protocol/ring_switch/coeffs.rs:288-366`); the `z_folded_rings | e_hat | t_hat | r_hat` form above is the `z_first` layout.
The savings formula is order-independent because it only counts the `r_hat` ring elements, regardless of where the segment sits.
Under `feature = "zk"`, current terminal tails also carry B-side blinding planes and are out of scope for these artifact-backed savings estimates.

Artifact source:

- Workflow: `Akita Profile Benchmarks`.
- Artifact name: `profile-bench-data`.
- Downloaded open-PR runs:
  - PR 140, run `26715552897`, branch `quang/crt-ntt-prime-profiles`.
  - PR 136, run `26702497503`, branch `taghi/perf/eor-sc`.
  - PR 137, run `26673769674`, branch `setup-inner-product-oracle`.
  - PR 111, run `26716208244`, branch `taghi/perf/simd-subfield-fp8`.
  - PR 109, run `26666323936`, branch `quang/polyops-cutover-spec`.
  - PR 138, run `26648236754`, branch `setup-prefix-ladder`.

The downloaded artifacts all agree on proof-shape sizes for the benchmark matrix.
The PR 140 artifact gives the following direct-mode savings estimates:


- `fp32-dense-nv26-np1-d32`:
  proof `36624` bytes,
  terminal rows `10`,
  non-quotient terminal ring elements `2070`,
  `r_hat` ring elements `160`,
  tail bits per element `2`,
  `r_hat` bytes `1280`,
  terminal stage-2 bytes `816`,
  total saved `2096` bytes,
  estimated proof `34528` bytes,
  proof reduction `5.72%`.

- `fp32-onehot-nv32-np1-d32`:
  proof `38480` bytes,
  terminal rows `12`,
  non-quotient terminal ring elements `969`,
  `r_hat` ring elements `96`,
  tail bits per element `4`,
  `r_hat` bytes `1536`,
  terminal stage-2 bytes `768`,
  total saved `2304` bytes,
  estimated proof `36176` bytes,
  proof reduction `5.99%`.

- `fp64-onehot-nv32-np1-d32`:
  proof `44768` bytes,
  terminal rows `6`,
  non-quotient terminal ring elements `1748`,
  `r_hat` ring elements `132`,
  tail bits per element `3`,
  `r_hat` bytes `1584`,
  terminal stage-2 bytes `768`,
  total saved `2352` bytes,
  estimated proof `42416` bytes,
  proof reduction `5.25%`.

- `fp128-onehot-nv32-np1-d32`:
  proof `63172` bytes,
  terminal rows `6`,
  non-quotient terminal ring elements `1233`,
  `r_hat` ring elements `156`,
  tail bits per element `5`,
  `r_hat` bytes `3120`,
  terminal stage-2 bytes `768`,
  total saved `3888` bytes,
  estimated proof `59284` bytes,
  proof reduction `6.15%`.

- `fp128-onehot-batched-nv30-np4-d32`:
  not separately tabulated here.
  An earlier draft listed figures byte-identical to the `fp128-onehot-nv32-np1-d32` row; those were a transcription duplicate, not a measured `np4` terminal-root result, and have been removed.
  The terminal-root level for `num_points = 4` has more public rows and a different total proof size than the `np1` case, so its savings must be re-extracted from the artifact during implementation and gated by the planner `exact_proof_bytes` check rather than asserted up front.

Across the six downloaded open-PR artifacts, the savings range is stable for each single-point (`np1`) case:

- Dense profiles save `1728` to `2096` bytes.
- One-hot profiles save `2304` to `3888` bytes.
- Overall proof-size reduction for these `np1` profiles is `5.25%` to `6.15%`.

These ranges cover the single-point profiles only; the batched terminal-root (`np4`) savings are measured during implementation, not asserted here.

Expected runtime effects:

- Prover should save the terminal `compute_relation_quotient` call and terminal stage-2 prover work.
- Verifier should save terminal stage-2 verifier work and terminal ring-switch aggregation preparation.
- Verifier adds direct all-row terminal checks.
- The net verifier time is an empirical question because direct row checks are deterministic and all-row, while the current verifier checks a randomized sumcheck final relation.
- The benchmark artifacts do not include terminal subspan timings, so this spec treats wall-time speedup as unmeasured until implementation adds direct-mode profiling.

## Design

### Architecture

Add a new terminal proof mode enum in a shared crate.
A likely home is `akita-types` because the mode affects proof shape, schedule descriptors, and transcript descriptors.

```rust
pub enum TerminalProofMode {
    RingSwitchSumcheck,
    DirectRingRelations,
}
```

Expose the selected mode through `CommitmentConfig`.
The default production preset stays `RingSwitchSumcheck`; a dedicated experimental or test-only config selects `DirectRingRelations`.
Both enum variants are retained permanently because both terminal modes are legitimate coexisting runtime choices, not a migration step.
This is a runtime mode selector, not a deprecated alias or compatibility shim, so the repo's no-backward-compat rule does not require removing either variant: intermediate folds always use the sumcheck path, and the terminal mode is chosen per config.

Extend `ProtocolFeatureSet` or the descriptor setup section with the terminal proof mode.
This is required because direct mode changes transcript behavior.
It removes terminal ring-switch and terminal sumcheck squeezes.
`CommitmentConfig` chooses the mode, the descriptor binds it, the schedule materializes the matching witness shape, and the proof shape disambiguates headerless terminal bytes.
The verifier must check that all four agree.

Update proof shapes explicitly.
One clean shape is:

```rust
pub enum TerminalRelationProofShape {
    RingSwitchSumcheck(SumcheckProofShape),
    DirectRingRelations,
}
```

Then `TerminalLevelProofShape` stores `relation: TerminalRelationProofShape`.
The proof body mirrors that shape:

```rust
pub enum TerminalRelationProof<L> {
    RingSwitchSumcheck(SumcheckProof<L>),
    DirectRingRelations,
}
```

Under `feature = "zk"`, use a parallel masked enum or reject `DirectRingRelations` before proof construction.
Do not encode direct mode as `stage2_sumcheck = []`.
The proof format is headerless and must remain unambiguous from the shape alone.

### Descriptor And Wire Binding

The mode must be encoded twice:

- in canonical descriptor bytes, because transcript schedules diverge after the terminal remainder absorb,
- and in terminal proof shape and wire data, because terminal proof decoding is headerless.

`ProtocolFeatureSet` currently only binds `zk`.
Direct mode must extend descriptor fields explicitly rather than relying on schedule shape alone.
The verifier-reachable source of truth is:

1. `CommitmentConfig` chooses the terminal mode,
2. `AkitaInstanceDescriptor` binds it,
3. scheduled witness shape and proof-size accounting materialize it, and
4. `TerminalLevelProofShape` and `TerminalLevelProof` serialize it unambiguously.

Any disagreement across those layers must reject before hot arithmetic paths run.

Add mode-aware terminal witness sizing.
The existing `w_ring_element_count_with_counts_for_layout_bits` always includes:

```text
e_hat_count + t_hat_count + z_folded_rings_count + r_count
```

Direct terminal mode needs:

```text
e_hat_count + t_hat_count + z_folded_rings_count
```

The function should be generalized rather than duplicated.
A small enum such as `TerminalWitnessQuotient::{IncludeRHat, OmitRHat}` is clearer than overloading `MRowLayout`.
`MRowLayout::WithoutDBlock` already means D-block omitted from rows.
It should not silently mean quotient omitted.

Add a no-r terminal witness builder.
It should share the segment emission helpers used by `build_w_coeffs`:

- `emit_z_folded_rings_block_inner`,
- `emit_planes_block_inner`,
- ZK blinding emission if later supported.

It should not call `compute_relation_quotient`.
It should not allocate `r_planes`.
In transparent mode it emits only `z_folded_rings`, `e_hat`, and `t_hat`.
It should return `RecursiveWitnessFlat` so packing and transcript slicing can reuse existing `CleartextWitnessProof` helpers.

Add a direct terminal row checker in the verifier crate.
A likely module is `crates/akita-verifier/src/protocol/ring_switch/terminal_direct.rs`.
The checker should accept:

- terminal `final_witness`,
- the transcript-facing `TerminalWitnessSegmentLayout` plus a shared full segment layout for row decoding,
- `LevelParams`,
- prepared recursive opening point and ring multiplier point,
- flattened commitment rows,
- `y_rings`,
- setup matrix views,
- fold challenges,
- explicit `num_claims`, `num_points`, `num_public_rows`, `num_polys_per_point`, `claim_to_point`, and `claim_poly_indices`.

The checker should decode the packed witness and split it into:

- `z_folded_rings` digit planes,
- `e_hat` digit planes,
- `t_hat` digit planes,
- optional blinding planes only if a future ZK direct mode is specified.

Then it should recompose rows and check:

- consistency row equals zero,
- public rows equal `y_rings`,
- B rows equal the current commitment-row semantics for the active singleton or multipoint terminal surface,
- A rows equal zero.

The direct checker should prefer existing verifier helpers where they already implement the same arithmetic safely, but it must not blindly reuse root-direct witness semantics.
Useful nearby helpers include packed witness decoding, the `RingSwitchSegmentLayout` decoder (`crates/akita-verifier/src/protocol/ring_switch.rs:140,605`), and matrix multiplication helpers that already preserve verifier-only arithmetic conventions.
If a helper currently lives under root-direct verifier code but has the right verifier-only semantics, move it to a shared verifier module instead of duplicating logic.

### Transcript Schedule

Current terminal schedule:

```text
descriptor bind
commitment absorb
opening point and y absorb
terminal e_hat absorb
fold challenges
terminal witness remainder absorb
alpha squeeze
tau1 squeezes
stage2 sumcheck round squeezes
```

Direct terminal schedule:

```text
descriptor bind
commitment absorb
opening point and y absorb
terminal e_hat absorb
fold challenges
terminal witness remainder absorb
direct row checks
```

The descriptor mode bit is mandatory because all bytes after the terminal remainder diverge.
In direct mode there is no terminal `alpha`, no terminal `tau1`, and no terminal stage-2 challenge stream.
Extension-opening reduction, when present, keeps its existing pre-terminal transcript schedule.

### Proof Size Accounting

Current terminal level bytes are:

```text
y_bytes + extension_opening_reduction_bytes + stage2_sumcheck_bytes
```

Current terminal tail bytes are:

```text
packed_digits_bytes(z_folded_rings | e_hat | t_hat | r_hat).
```

Direct terminal level bytes become:

```text
y_bytes + extension_opening_reduction_bytes.
```

Direct terminal tail bytes become:

```text
packed_digits_bytes(z_folded_rings | e_hat | t_hat).
```

When the removed `r_hat` digits were not the maximum absolute digit in the current witness, the byte saving is exactly:

```text
ceil(r_hat_digits * bits_per_elem / 8).
```

If removing `r_hat` lowers `PackedDigits::required_bits_per_elem`, savings can be larger.
The artifact estimates use the conservative exact-current-bits calculation.

### Soundness Argument

The current terminal stage-2 sumcheck proves a randomized aggregation of:

```text
W(x, y) * alpha(y) * M_tau1_alpha(x).
```

That randomized aggregation is needed when the verifier cannot inspect every row and every witness segment directly.
In terminal direct mode, the verifier receives the cleartext non-quotient witness and checks every reduced ring row directly.
The check is stronger with respect to row coverage because no random row aggregation is needed.

The direct verifier must still enforce digit range through `PackedDigits` validation.
It must still enforce the y-ring opening relation before row checking.
It must still bind the exact final witness bytes before any challenge derived from that witness is consumed.

The quotient `r` is not part of the terminal statement.
It is an auxiliary witness for proving unreduced polynomial divisibility in the sumcheck representation.
Once the verifier checks the reduced cyclotomic row equations directly, `r` is unnecessary.

### ZK Policy For Direct Mode

Direct terminal mode is transparent-only in this spec.
If `feature = "zk"` is enabled, selecting `DirectRingRelations` must reject with `InvalidSetup` before proof construction or verification.

This is not just an implementation gap.
Current terminal ZK soundness depends on:

- masked `y_rings` on the wire,
- `stage2_sumcheck_proof_masked`,
- verifier-side `relation_claim_mask`, and
- the B-side blinding contribution carried through the terminal stage-2 row-evaluation path.

Removing terminal stage-2 removes that masking mechanism.
A future ZK direct mode therefore needs a separate spec.

If such a mode is added later, it should:

- keep masked `y_rings` on the wire,
- keep B-side blinding in the cleartext terminal tail,
- keep D-side blinding absent under `MRowLayout::WithoutDBlock`,
- omit `r_hat`, and
- replace `stage2_sumcheck_proof_masked` with a masked direct-row relation argument.

### Alternatives Considered

**Drop only stage-2 sumcheck, keep `r_hat`.**
This is simpler because it preserves terminal witness length.
It leaves a large part of the desired saving unused and still sends quotient data that direct row checking does not need.

**Keep terminal ring-switch challenges and check only at one random row.**
This saves proof bytes but does not replace the sumcheck soundness argument cleanly.
It also leaves transcript complexity in place without need.

**Reuse `MRowLayout::WithoutDBlock` to mean no quotient.**
This is too implicit.
`MRowLayout::WithoutDBlock` already means D rows are omitted from the row layout.
The quotient emission policy should be a separate mode because sumcheck-terminal mode still needs `r_hat`.

**Extend to intermediate recursive folds.**
Intermediate folds do not reveal the next witness.
Their stage-2 sumcheck supplies the challenge vector used as the next recursive opening point.
A direct check there would need a different handoff mechanism and is out of scope.

**Support ZK direct mode immediately.**
The current ZK terminal stage-2 path records masked sumcheck relations.
Direct all-row verification would need a masked row-equation design.
This should be a separate spec unless direct mode is intended to be transparent-only.

## Documentation

Update or add:

- `specs/terminal-fold-cutover.md`, add a note that the existing terminal cutover still uses relation-only stage-2 and `r_hat`.
- `specs/transcript-immediate-fixes.md`, document direct terminal transcript order if this mode ships.
- `specs/profile-bench-coverage-matrix.md`, include direct terminal mode in the proof-size benchmark matrix.
- Profile example docs, add direct terminal proof breakdown fields once implemented.

## Execution

Recommended implementation order:

1. Add `TerminalProofMode` and descriptor binding.
2. Make terminal proof shape and wire format mode-aware.
3. Add direct terminal witness sizing that excludes `r_hat`.
4. Add no-r terminal witness construction in the prover.
5. Add verifier packed-witness segment decoding for no-r terminal witnesses.
6. Add direct terminal row checker.
7. Branch terminal prover and verifier by mode.
8. Update proof-size formulas and schedule planning.
9. Add E2E and tamper tests.
10. Refresh generated schedule tables if production configs expose direct mode.
11. Add benchmark reporting for direct mode and compare against this spec's estimates.

Important implementation risks:

- The no-r witness segment layout must remain byte-for-byte aligned between prover packing, verifier transcript slicing, and verifier row decoding.
- Direct row checks must use reduced cyclotomic multiplication, not the cyclic high-half quotient arithmetic used to compute `r`.
- Proof-shape validation must reject mismatched terminal modes before verifier code reaches hot arithmetic paths.
- If direct mode is selectable by config, table-generated schedules must encode or derive terminal direct witness shapes consistently.
- If direct mode changes proof size enough to affect planner choices, generated schedule tables must be regenerated under the same policy.

## References

- `crates/akita-prover/src/protocol/flow/recursive.rs`, terminal prover flow.
- `crates/akita-prover/src/protocol/ring_switch/coeffs.rs`, current `r_hat` append in `build_w_coeffs`.
- `crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs`, quotient relation and `generate_y` row layout.
- `crates/akita-verifier/src/protocol/levels/recursive.rs`, terminal verifier flow.
- `crates/akita-verifier/src/protocol/levels.rs`, terminal-root verifier flow.
- `crates/akita-verifier/src/protocol/ring_switch.rs`, terminal ring-switch challenge replay.
- `crates/akita-verifier/src/protocol/levels/zk.rs`, verifier-side ZK terminal masking and hiding-commitment checks.
- `crates/akita-verifier/src/stages/stage2.rs`, relation-only terminal stage-2 verifier oracle.
- `crates/akita-types/src/proof/levels.rs`, `TerminalLevelProof`.
- `crates/akita-types/src/proof/shapes.rs`, terminal proof shape.
- `crates/akita-types/src/proof/tests.rs`, terminal proof round-trip coverage.
- `crates/akita-types/src/proof/wire.rs`, terminal serialization.
- `crates/akita-types/src/schedule.rs`, terminal witness width accounting.
- `crates/akita-types/src/proof_size.rs`, terminal proof byte accounting.
- `crates/akita-pcs/tests/transcript_hardening.rs`, current terminal transcript order and terminal witness tamper coverage.
- `crates/akita-pcs/tests/common/mod.rs`, current terminal transcript-order helper.
- `crates/akita-pcs/tests/zk.rs`, current masked-opening regression coverage.
- `crates/akita-pcs/examples/profile/report.rs`, proof breakdown fields used by the savings analysis.
- `.github/workflows/profile-bench.yml`, benchmark artifact workflow.
- `specs/terminal-fold-cutover.md`, previous terminal witness binding and D-block cutover.
- `specs/transcript-immediate-fixes.md`, terminal transcript schedule.
