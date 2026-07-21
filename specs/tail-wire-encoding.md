# Spec: Tail Wire Encoding (commitment elision + segment-typed entropy coding)

> **Pre-zk-strip historical.** This umbrella spec predates the zk-strip
> ([`akita-zk-strip-for-audit.md`](akita-zk-strip-for-audit.md)). References to
> `feature = "zk"` or `PackedDigits` describe removed code preserved on `zk-wip`.

| Field       | Value                                                     |
|-------------|-----------------------------------------------------------|
| Author(s)   | Quang Dao                                                 |
| Created     | 2026-06-13                                                |
| Status      | implemented                                               |
| PR          | #190, #209, #311                                          |
| Book-chapter | how/recursion.md                                         |

## Summary

The Akita proof tail is the only place a folded witness is sent in cleartext
rather than as a commitment. Every supported schedule contains a root fold and
at least one suffix fold before this terminal; zero-fold and root-terminal
proof families are unsupported. The canonical transparent witness is the sole
segment-typed `(z, e, t)` representation: Golomb-Rice
bytes for the norm-bounded `z` response and raw canonical field bytes for
`e_folded` and `t`. Terminal quotient `r`, terminal stage 2, and the old uniform
`PackedDigits` representation are absent.

Historically, #190 replaced one `PackedDigits` blob containing `z`, opening
digits, inner-commitment digits, and quotient digits with typed segments while
the elision work remained deferred. The folded-only cutover subsequently
removed the raw-field zero-fold representation entirely.
That historical encoding paid the worst-case width on every coordinate even
though `z` is norm-bounded while `e_folded`, `t`, and the former quotient `r`
are full-entropy field data.

This spec defines a comprehensive tail wire encoding built on one principle: **encode each wire object according to its actual distribution and role.**
It composes four levers, three of which are commitment/quotient *elision* (do not send objects whose information the verifier can reconstruct or check directly) and one of which is *entropy coding* (send norm-bounded witness segments at their true entropy, not at a worst-case fixed width).
The verifier reconstructs every elided or de-decomposed quantity deterministically, so soundness is unchanged and the wire shrinks to the realized information content.

**Current scope.** The transparent cutover includes segment typing, Golomb-Rice
`z`, terminal `r`/stage-2 elision, and the predecessor-bound terminal `t`
handoff. Ordinary recursive edges still ship outer `u`; the final edge always
binds inner `t` and the terminal has no B block. Historical #190/#209 staging
and remaining Jolt measurement notes are recorded in Execution.

## Intent

### Goal

Use a segment-typed, entropy-coded cleartext tail whose per-segment models are
derived from public, transcript-bound parameters with zero model side
information on the wire. Elide terminal quotient artifacts and, on a recursive
handoff into the terminal, replace outer `u` with canonical inner `t`.

The feature introduces or modifies:

- A **segment-typed tail witness** representation replacing the historical single-width `PackedDigits` blob: Golomb-Rice for `z` only and raw canonical field coefficients for `e`/`t`. Boundaries derive from the schedule/shape (headerless wire). Wire `rice_low_bits` is never on the wire; both sides derive it from the fold `‖z‖_inf` cap (`min(β_inf, t*)` or `β_inf` alone) for `z` only.
- A **canonical, total Golomb-Rice codec** (`akita-types`, verifier-reachable, no-panic) with zigzag sign mapping and standard unary+remainder Rice encoding only. Decode rejects unary runs longer than [`golomb_rice_max_quotient_for_cap`](../crates/akita-types/src/golomb_rice.rs). **#190 applies it to `z` only.**
- A per-level **fold `‖z‖_inf` cap accessor** via [`LevelParams::fold_witness_linf_cap_for_claims`](../crates/akita-types/src/layout/params.rs) and the deterministic cap→wire low-bits rules in [`tail_golomb_rice_low_bits`](../crates/akita-types/src/tail_golomb_rice_low_bits.rs) for the folded-response `z` segment.
- **Terminal `t`-state cutover**: the predecessor fold stops sending outer
  `u = B * decompose(t)` and binds canonical inner `t` rings as the terminal's
  public state; the terminal later checks their challenge-folded A relation.
- **`r`-elision and terminal-stage-2 elision** via the
  [terminal direct ring relations cutover](terminal-direct-ring-relations-cutover.md).
- **Descriptor binding** through the canonical schedule/instance descriptor:
  structural `folds + terminal` topology fixes ordinary outer-`u` edges and
  the final inner-`t` handoff.

### Invariants

1. **Lossless and soundness-preserving.** Entropy-coded segments decode bit-exactly; auxiliary `v`/`r` are replaced by direct checks; and the final outer `u` is replaced by transcript-bound inner `t` plus its A check. The digit/norm bounds the extractor relies on are unchanged.
2. **Canonical encoding.** Each integer vector has exactly one valid byte encoding under a fixed `(model, rice_low_bits)`; a non-canonical or malformed encoding is rejected with `AkitaError`/`SerializationError`, never decoded ambiguously. Protected by: a canonicality unit test (encode-decode-encode fixpoint) and a malformed-bytes rejection test.
3. **Total, bounded, no-panic decode.** The Golomb-Rice decoder terminates on every byte string, rejects unary quotients above the cap-derived maximum, allocates only the schedule-declared element count, rejects non-minimal trailing byte padding, and never panics, unwraps, or indexes unchecked. Protected by: a fuzz/edge unit test over random byte strings and the verifier no-panic audit.
4. **Public models, zero side info.** Segment boundaries and Golomb wire low bits for `z` (`cap`, `wire_rice_low_bits`) are derivable by the verifier from `LevelParams` + transcript before decode; `e`/`t` carry no separate model tag. Protected by: a prover/verifier cap/wire low-bits agreement test for `z` and a `LoggingTranscript` event-stream equality test.
5. **Terminal `t`-state preserves weak binding.** The last recursive transition changes the terminal input state from outer `u = B * decompose(t)` to canonical inner `t`. Soundness does **not** come from deleting B while keeping `u`; it comes from binding `t` before dependent challenges and checking `A_g z_g = Σ_i c_{g,i} t_{g,i}` together with the global consistency and opening-trace equations.
6. **Descriptor binding fixes topology.** The instance descriptor binds the complete structural schedule. Every accepted schedule has at least two folds, and its final edge selects the terminal-inner handoff. A proof-body binding variant inconsistent with that schedule rejects.
7. **Transparent-only.** Entropy coding applies only to the transparent tail. Historical `feature = "zk"` / `PackedDigits` notes refer to the removed pre-zk-strip implementation.

### Non-Goals

- **No general intermediate-level change.** Non-terminal recursive folds still commit their next witness with the usual outer `u` commitment. The only exception is the **last recursive transition into the transparent terminal**, whose next-state payload is `t` instead of `u`.
- **No degenerate proof fallback.** Inputs that cannot support at least two
  shrinking folds return `UnsupportedSchedule`; they do not select another
  witness representation.
- **No tiered multipoint tail.** Tiered commitment layouts require `num_points == 1` today; multipoint tiered terminal encoding is out of scope until that restriction is lifted.
- **No ZK tail encoding.** Masked/blinded witnesses are near-uniform and do not compress; a ZK direct/entropy tail needs a separate masked-relation design (same boundary PR #141 draws).
- **No change to the norm bound or the decomposition basis.** This spec changes how the realized witness is *encoded*, not the bound `K`/`t*` the verifier enforces (PR #174) nor the SIS rank pricing.
- **No new commitment scheme.** The t-reveal reuses the existing A-row check; it does not introduce a new matrix or assumption.
- **No entropy coding for `e`.** The `e` segment carries `e_folded` (partial-evaluation ring elements, `e_i = ⟨a, f_i⟩` before digit decomposition). Witness sparsity (one-hot, etc.) does not make `e` small on the wire; Golomb–Rice applies only to norm-bounded `z`.
- **No intermediate r-drop.** Terminal `r` and terminal stage 2 are gone;
  ordinary intermediate folds retain their quotient tail and sumchecks.

## Evaluation

### Acceptance Criteria

Encoding slice (#190):

- [x] A `GolombRice` codec in `akita-types` is canonical (encode-decode-encode fixpoint, minimal byte length), total (terminates and is no-panic on malformed/truncated bytes), and bijective on `[-cap, cap]` at wire low bits, verified by cap-range round-trip tests.
- [x] `fold_witness_linf_cap_for_claims` and the deterministic cap→low-bits rules (`cap_rice_low_bits`, `wire_rice_low_bits`) return integers pinned against reference calculations; prover and verifier derive the same wire low bits for `z`.
- [x] The transparent terminal `final_witness` serializes as segment-typed
  payloads (`z` length-prefixed Golomb bytes, then raw `e`/`t` fields);
  `SegmentTypedWitnessShape::admits_realized` accepts exact `z` payloads up to the
  schedule upper bound.
- [x] Non-zk shipped schedule tables regenerated under segment-typed terminal sizing; `generated_schedule_tables_match_find_schedule` passes for affected families.
- [x] Net `z` entropy win on cited `onehot_fp128_d64` cells (Golomb at `wire_rice_low_bits(cap)` beats legacy `PackedDigits`); profile emits structured tail breakdown (`proof tail summary`, Golomb vs packed `z` stats).

Current umbrella items:

- [x] No terminal `r` or terminal stage-2 sumcheck. The final recursive
  transition sends no parent `u`; the terminal relation has no B block.
- [x] Prover and verifier bind the same canonical terminal `t`, and direct A
  rows check the revealed witness against it. Existing tamper suites cover
  `z`, `e`, and `t`; exact two-fold coverage remains a recommended hardening
  pin.
- [x] Schedule descriptor bytes bind the topology from which the outgoing
  binding policy and terminal row layout are derived.
- [ ] In-guest decode cost measured via `final_witness_decode` in `profile/akita-recursion`; net `akita_verify` cycles do not regress versus the PR #141 direct-mode baseline.

### Testing Strategy

Must keep passing: all `akita-types` proof/shape/serialization tests, the terminal direct-relation row tests (PR #141), the schedule drift guards, e2e batched/multipoint/recursive suites, transcript and zk suites.

New tests:

- `akita-types`: Golomb-Rice round-trip, canonicality (minimal byte length), malformed/unary-above-cap edge cases, no-panic over random bytes; cap/wire low-bits reference pins; segment layout decode boundaries; `TerminalLevelProofShape` round-trips the segment-typed tail and rejects cross-policy shapes.
- `akita-prover`/`akita-verifier`: prover/verifier cap/wire low-bits agreement; terminal `t`-state row check (send `t`, check revealed witness maps to `t`, no B block); verifier no-panic on malformed tail bytes.
- e2e: tamper tests for `z`, `e`, `t`, segment shape, and `y_rings`; descriptor cross-policy reject; ZK rejection; post-decode norm-bound violation reject.
- `LoggingTranscript`: event-stream equality across the new tail schedule.

Feature combinations: default, `--no-default-features`, `--features zk` (policy rejects), `--features logging-transcript`.

### Performance

Expected direction: smaller proofs, no material prover slowdown, bounded verifier decode.

Per-lever, at the affected tail modes (verified by `AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 cargo run --release --example profile` and the planner `exact_proof_bytes`):

- **r-drop + terminal stage-2 drop**: implemented for transparent terminals.
- **u-drop (terminal `t`-state)**: implemented on the final recursive edge.
  The removed payload is one
  `n_b`-ring-element outgoing commitment on affected schedules.
- **z entropy coding** *(landed in #190; cap-aligned on fold-linf #189; wire tightened in #209)*: `z` coordinate cost drops from fixed `depth_fold * log_basis` bits/coord (packed digits) to `rice_low_bits + O(1)` bits/coord on wire at `wire_rice_low_bits(cap) = cap_rice_low_bits(cap) - 2` (~10 bits/coord on fp128 D64 vs 15 packed when `cap = β_inf`). Realized random witnesses average ~9.7–10 bits/coord. Planner budgets use `cap_rice_low_bits(cap) + 2` bits/coord ([`TAIL_Z_PLANNER_CAP_LOW_BITS_PLUS_TWO`]). Both wire and planner reference the same fold `‖z‖_inf` cap as `num_digits_fold` and grind acceptance, not the level variance envelope `isqrt_ceil(β_inf² · T_level · ρ²)` (which would imply `cap_rice_low_bits ≈ 22`).
- **`e`/`t` as `RawField`**: wire carries full ring/base-field coefficients
  for partial-evaluation `e_folded` and inner state `t`. Terminal quotient `r`
  no longer exists. No entropy coding is applied to either raw-field segment.

A-role rank, setup size, and L2 pricing are unchanged.

## Design

### Architecture

The unifying classification: every tail wire object is in exactly one bucket, and each bucket has one correct encoding.

| object | nature | correct encoding | lever |
|---|---|---|---|
| `v = D * e_hat` | auxiliary D image | do not send; reveal `e` at the tail | D-drop (done, PR #88) |
| terminal input | final recursive witness state | bind inner `t = A * w_terminal`; do not send outer `u` | terminal `t`-state |
| `z` (folded response) | norm-bounded per coefficient (`cap`) | Golomb-Rice keyed by `wire_rice_low_bits(cap)` | this spec (S4) |
| `e` (`e_folded`, partial evaluation) | full ring element per block (not sparse) | raw field coefficients (`RawField`) | **#190** |
| `t` (inner commitment image) | full field entropy | raw field coefficients (`RawField`) | **#190** |
| `r` quotient | auxiliary, not in the statement | do not send; direct ring relations | implemented terminal cutover |

Affected surfaces:

- `akita-types`: the codec, the `SegmentTypedWitness` representation and its shape, `direct_witness_bytes` / `proof_size.rs` tail accounting, the `cap`/low-bits accessors on `LevelParams` (via `fold_witness_linf_cap_for_claims` and `tail_golomb_rice_low_bits`), and the descriptor policy fields.
- `akita-prover`: emit the tail as typed segments (Golomb `z` on centered
  fold-response integers; `RawField` for `e_folded`/`t`), send terminal `t` as
  the next-state payload at the last recursive transition, and skip parent `u`.
- `akita-verifier`: decode typed segments (no-panic), expand `z` to digit planes and decompose `e`/`t` field segments to digit planes for row checks, verify terminal A rows against transcript-bound state.
- `akita-planner`/`akita-config`: codec-aware tail byte accounting, regenerate shipped tables under the new tail policy, bind the policy in the descriptor.

The transparent tail always uses `SegmentTypedWitness`. References below to a
retained ZK `PackedDigits` arm are historical.

### Terminal `t`-state / u-elision

Intermediate relations use `consistency | A(n_a) | B(n_b) | D(n_d)`
(`crates/akita-types/src/layout/params.rs`). Public openings bind through the
fused trace term in stage-2 sumcheck, not through M public rows. Every terminal
reveals `e_folded` in cleartext, so the D-role commitment `v = D * e_hat` and
its rows are gone. Because the terminal state is predecessor-bound `t`, the B
block is gone as well. The sole terminal layout is
`WithoutCommitmentBlocks = consistency | A`.

The correct S2 cutover is **not** "delete B rows while keeping `u` as the terminal statement."
That would be unsound: if the verifier still accepted a public commitment `u` but no longer checked `B * t_hat = u`, the terminal proof would no longer depend on `u`.

Instead, the last recursive transition changes the terminal input state and
binds the terminal inner rings

```text
t_{g,i}
```

as raw ring elements, in the transcript slot where the next-state payload is bound.
It does **not** send the outer image

```text
u = B * decompose(t).
```

The terminal direct relation then has no commitment/B block. Its A block is no
longer a zero-RHS quotient row whose `t` contribution lives only inside the
witness MLE. After sampling the fold coefficients, the verifier checks

```text
A_g z_g = Σ_i c_{g,i} t_{g,i}
Σ_g,i c_{g,i} e_{g,i} = Σ_g multiplier_g · G · z_g
weighted_opening_eval(e, row_coefficients, EOR_scales) = trace_eval_target.
```

Soundness sketch.
The terminal state is `t`, not `u`.
The prover must bind the canonical bytes of `t` before any terminal challenge depending on the terminal state is squeezed.
The terminal verifier decodes the clear witness, checks the public norm/digit
bounds, and enforces the three equations above. Thus an accepting proof gives
the challenge-folded short A preimage tied to the predecessor-bound `t`, while
the consistency and trace equations tie that response to the folded evaluation
and public opening.
No terminal B binding is required because `u` is no longer part of that
statement. A terminal without a predecessor is not an accepted proof topology.

This is why the cutover is sound: it replaces the outer-`u` statement with the
challenge-folded A/consistency/trace statement above at the transparent tail,
where `z`, `e`, and `t` are revealed and `z` is range-checked.
It is implemented as the sole `WithoutCommitmentBlocks` terminal row layout.
The execution schedule is the single source of truth for selecting it.

### The segment-typed tail encoder

The transparent tail witness becomes an ordered list of typed segments, replacing the single `PackedDigits` blob:

```rust
pub enum TailSegmentModel {
    /// Norm-bounded folded-response integers; Golomb-Rice with wire `rice_low_bits = wire_rice_low_bits(cap)`.
    Gaussian { rice_low_bits: u32 },
    /// Full field/base coefficients for one ring element (`e_folded` or `t`).
    RawField,
}
```

Only `z` uses `Gaussian`. The `e` segment is **`e_folded`**: one cyclotomic ring element per witness block, the partial evaluation `⟨a, f_i⟩` at the opening point. It is **not** `e_hat` (balanced opening digits) and is **not** compressible by witness sparsity: even one-hot witnesses yield full field values in `e_folded`.

The wire carries only concatenated segment payloads, with no per-segment header beyond a fixed `usize` length prefix on the variable-length Golomb `z` segment (so decoders can use the schedule's `z_payload_bytes` upper bound while reading the exact encoded payload).
Segment boundaries and emission order are derived by both sides from `LevelParams` + incidence.
For the final `t`-state tail, derive segment **coordinate counts** from the same public layout inputs as `terminal_witness_segment_layout` (`crates/akita-types/src/proof/terminal_witness.rs:204-229`).
Let `D = ring_dimension`.
The legacy `RingRelationInstance::segment_layout` plane counts (`crates/akita-types/src/proof/ring_relation.rs:226-237`) count digit **planes** for the packed-digit layout. The segment-typed wire uses different units per segment:

```text
z_coords       = num_z_segments * num_positions_per_block * num_digits_inner * D   (Golomb integers; one per base-field slot in folded z)
e_field_elems  = num_live_blocks * num_w_vectors * D                       (RawField; one ring element per block → D coeffs)
t_field_elems  = n_a * num_live_blocks * num_t_vectors * D                   (RawField)
```

`z_coords` is the Golomb element count. `e_field_elems` and `t_field_elems` are base-field coefficient counts for `RawField` serialization.
The historical `PackedDigits` layout in `build_w_coeffs` was the reference for
the original byte-neutral framing. The canonical transparent terminal has no
legacy `t_hat`, `û_concat`, or `r_hat` planes.
Segments appear in wire order `z ‖ e ‖ t`. Terminal quotient rows and their digit planes do not exist.
Multipoint layouts scale `z_coords` with `num_z_segments`.

This mirrors the existing headerless, shape-driven decode (the shape supplies
counts and the `z` payload upper bound). `SegmentTypedWitnessShape` and
`SegmentTypedWitness` descend from the typed representation shipped in #190.

The verifier decodes `z` via Golomb–Rice into centered ring elements and
decodes `e`/`t` as raw field coefficients. It always checks consistency and A
directly. The terminal has no B block. The wire never carries `e_hat` digit
planes for `e`; it carries
`e_folded` field elements.

### Golomb-Rice for the Gaussian z segment

For a signed integer `n` admitted by the `z` segment's public bound, zigzag to a non-negative `u = (n << 1) ^ (n >> (W-1))` where `W` is the signed-integer bit width from the fold cap (`golomb_rice_zigzag_width`). Rice-code `u` with low-bit width `rice_low_bits` as standard Golomb: unary quotient prefix + stop + remainder. Decode rejects unary length above `golomb_rice_max_quotient_for_cap(cap, rice_low_bits, W)`.

Wire low bits for the `z` segment are derived deterministically from the public per-coefficient fold `‖z‖_inf` cap:

```text
cap              = fold_witness_linf_cap_for_claims(num_claims)   (= min(β_inf, t*) or β_inf alone)
cap_rice_low_bits = rice_low_bits_for_cap(cap)                   (= max(0, floor(log2(cap))), pinned by test)
wire_rice_low_bits = cap_rice_low_bits - WIRE_RICE_LOW_BITS_DELTA  (= cap low bits - 2 today)
```

`cap` is the same bound already used by [`fold_witness_digit_plan`](../crates/akita-types/src/sis/norm_bound.rs) and grind acceptance. It is **not** the level variance envelope `isqrt_ceil(β_inf² · T_level · ρ²)` from PR #174's `t*` analysis: that quantity aggregates coordinates and is far too loose for per-coordinate Golomb-Rice parameterization (it would imply `cap_rice_low_bits ≈ 22` on fp128 D64 where `cap_rice_low_bits = 12` suffices).

`rice_low_bits_for_cap` is the cap-derived Rice low-bit width covering every admitted `z` coefficient magnitude at the planner reference. `wire_rice_low_bits` is what prover and verifier use on the wire (#209). The codec is canonical because standard Rice is bijective for a fixed `rice_low_bits` on `[-cap, cap]`, and wire payloads must use the minimal byte length (partial-byte zero padding only). Decode and grind use the cap-derived maximum quotient; no alternate wire shape exists.

### RawField segments (`e`, `t`)

Serialize `RingVec` coefficients in canonical field form (`field_bytes` each).
No Golomb and no per-segment model tag. The direct checker consumes the raw
segments without rebuilding a terminal stage-2 digit stream.

### Descriptor and wire binding

Tail encoding uses three layers:

| Layer | Source | Carries |
|-------|--------|---------|
| **Policy** | `AkitaInstanceDescriptor` schedule and level descriptors | codec/cap rules, terminal topology, row and witness parameters |
| **Layout** | `SegmentTypedWitnessShape` / `TerminalLevelProofShape` (S3) | per-segment element counts (and byte length once entropy-coded) |
| **Derived** | both sides at runtime | `cap`, `wire_rice_low_bits`, segment order (`z ‖ e ‖ t`), decode bounds and outgoing binding |

The tail-encoding policy is bound in the canonical instance descriptor:

- codec identity (Golomb-Rice variant id for `z`, cap-derived max-quotient rule, zigzag width rule `W_z`),
- the cap→wire low-bits rule identity for the `z` segment only,
- per-segment model assignment: Golomb-Rice for `z`, raw canonical fields for
  `e`/`t`;
- the complete structural `folds + terminal` schedule, whose final edge binds
  terminal inner `t` and carries no `u`/B state.

The proof shape carries segment **counts** (and, for variable-length entropy segments, the realized payload byte length). Model tags and wire low bits are **not** duplicated on the shape; the verifier checks `shape.counts == derive_counts(policy, level_params, incidence)` before decode.
Prover, verifier, schedule digest, and descriptor must agree; any disagreement rejects before hot arithmetic.
Transcript binding (S3 acceptance): absorb **canonical encoded segment bytes** in the same logical order as today's `ABSORB_TERMINAL_E_HAT` / `ABSORB_TERMINAL_W_REMAINDER` events (`terminal_witness.rs`), updated for segment boundaries.

### Proof-size accounting

`direct_witness_bytes` (`crates/akita-types/src/layout/proof_size.rs:32-41`) gains a segment-typed arm: runtime sizing sums exact encoded segment sizes; **planner** sizing uses a conservative upper bound on entropy segments (`~num_coords * (cap_rice_low_bits + 2)` bits for Gaussian `z` under [`TAIL_Z_PLANNER_CAP_LOW_BITS_PLUS_TWO`]), consistent with the repo's `actual <= planned + ACCEPTED_PLANNER_PROOF_SIZE_OVERCOUNT_BYTES` profile gate.
`level_proof_bytes` drops `next_commit_bytes` for the last recursive transition.
The terminal direct witness owns the sole raw `t` payload. Its only layout
drops `v`/stage 1 and B.
Shipped schedule tables are regenerated under the new tail policy in S5; the drift guard `generated_schedule_tables_match_find_schedule` must pass after regen.

### Alternatives Considered

- **rANS / range coding against the discrete-Gaussian CDF.** Within ~0.01 bit of entropy vs Golomb-Rice's ~0.1-0.5 bit, but the decoder is a heavier state machine to make canonical, total, and cheap in-guest. Golomb-Rice captures nearly all the win at a fraction of the verifier/Jolt complexity; rANS is a future opt-in if the residual bits justify it.
- **Keep `PackedDigits`, only add per-segment widths.** Still leaves the `z` worst-case-width slack; segment-typed `RawField` for `e` is the correct model (partial evaluation is full field), not digit-plane packing at `z`'s width.
- **Transmit a small histogram/model.** Unnecessary: `β_inf` is public/transcript-derivable, so a parametric model costs zero wire bytes.
- **Single fixed Rice low-bit width across all modes.** Simpler but loses per-level `β_inf` matching for `z`; the `cap -> wire_rice_low_bits` rule is descriptor-bound for the `z` segment only.
- **BoundedSmall / Golomb for `e`.** Rejected: `e` is `e_folded` partial evaluation (full ring elements). Witness sparsity does not shrink `e` on the wire.
- **Apply entropy coding at intermediate levels.** Impossible: intermediate witnesses are committed, not revealed; only the tail is cleartext.

## Documentation

- `specs/terminal-direct-ring-relations-cutover.md` records the direct
  relation/r-drop layer; `specs/terminal-fold-cutover.md` records the D-drop.
- Update PR #174 / `specs/fold-linf-rejection.md` cross-link: Golomb `z` sizing uses the same fold `‖z‖_inf` cap as `num_digits_fold` (distinct from the level variance envelope).
- Profile reports attribute segment-typed `z`/`e`/`t` bytes and the removed
  terminal quotient/stage-2/outgoing-`u` payloads separately.

## Execution

Historical umbrella dependency order, with current status:

```text
S1  Terminal r-drop + direct ring relations.                       implemented
S2  Suffix-terminal t-state / u-elision.                           implemented
S3  Segment-typed tail framing (replace single PackedDigits with    landed in #190
    typed segments; raw e/t + Golomb z; length-prefixed z)
S4  Golomb-Rice codec + cap/k for z segment only                  landed (#190 + #189 cap align)
S5  Descriptor binding + Jolt decode measurement + full S5 polish   partially landed (#190 + #209)
    (planner regen, profile tail accounting, wire low-bits rule in `FoldLinfProtocolBinding`,
    verifier `decode_terminal_z_golomb_payload` admissibility, grind planner budget gate)
```

**#190** was the first implementation PR under this umbrella. S1/S2 are now
part of the canonical transparent protocol rather than deferred alternatives.

Remaining audit/measurement items:

- Exact two-fold tamper tests should continue pinning predecessor `t`/A binding
  without a terminal B block.
- S3/S4 (landed in #190) must keep prover packing, verifier transcript slicing, and verifier row decoding byte-for-byte aligned across segment boundaries; `segment_typed_expand_matches_logical_w` guards this.
- S5 descriptor binding (wire low-bits rule id + δ in [`FoldLinfProtocolBinding`](../crates/akita-types/src/instance_descriptor/fold_linf_binding.rs); `AKITA_INSTANCE_DESCRIPTOR_VERSION = 2`) is implemented; only Jolt cycle measurement remains open.

## References

- PR #311 / [`terminal-direct-ring-relations-cutover.md`](terminal-direct-ring-relations-cutover.md): terminal r-drop and direct ring relations.
- PR #174 / `specs/fold-linf-rejection.md` (closed spec PR; implementation #189): the `t*` threshold and fold `‖z‖_inf` cap; Golomb `z` sizing uses the same cap as `num_digits_fold`, not the level variance envelope.
- `specs/terminal-fold-cutover.md` (PR #88): the D-role drop whose transcript-binding discipline the terminal `t`-state cutover reuses.
- `specs/weak-binding-norm-fix.md`: the weak-binding object the tail extraction recovers.
- `crates/akita-types/src/proof/direct_witness.rs` (`SegmentTypedWitness`),
  `crates/akita-types/src/proof/tail_segments.rs` (segment layout and transcript
  slicing), and `crates/akita-types/src/proof/levels.rs` (`TerminalLevelProof`).
- `crates/akita-types/src/proof_size.rs:72-106` and `crates/akita-types/src/layout/proof_size.rs` (proof-byte and witness accounting).
- `crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs` (intermediate A/B/D row roles), `crates/akita-prover/src/protocol/ring_switch/commit.rs` (ordinary outer `u` versus terminal inner `t`).
- Profile: `AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 cargo run --release --example profile`.
