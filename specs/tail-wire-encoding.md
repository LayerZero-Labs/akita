# Spec: Tail Wire Encoding (commitment elision + segment-typed entropy coding)

> **Pre-zk-strip historical.** This umbrella spec predates the zk-strip
> ([`akita-zk-strip-for-audit.md`](akita-zk-strip-for-audit.md)). References to
> `feature = "zk"` or `PackedDigits` describe removed code preserved on `zk-wip`.

| Field       | Value                                                     |
|-------------|-----------------------------------------------------------|
| Author(s)   | Quang Dao                                                 |
| Created     | 2026-06-13                                                |
| Status      | partially implemented (#190: non-zk encoding slice; umbrella elision deferred) |
| PR          | #190 (spec + encoding implementation, targets `main`; spec-only #187 superseded) |

## Summary

The Akita proof tail (the terminal recursive fold and the terminal-root 1-fold case) is the only place a folded witness is sent in cleartext rather than as a commitment.
Today that cleartext witness is one `PackedDigits` blob assembled in `crates/akita-prover/src/protocol/ring_switch/coeffs.rs` (`build_w_coeffs`) and packed via `PackedDigits::from_i8_digits_with_min_bits` (`crates/akita-types/src/proof/direct_witness.rs:165-169`): a single uniform `bits_per_elem` (at least `final_log_basis`, widened per witness if needed, capped at `log_basis <= 6`), with the folded response `z`, opening digits `e_hat`, inner-commitment digits `t_hat`, optional tiered `û_concat`, and quotient digits `r_hat` all concatenated and bit-packed at that one width.
The zero-fold fast path uses `CleartextWitnessProof::FieldElements` (raw polynomial coefficients), not this segment layout; see Non-Goals.
This is fixed-width to a worst-case bound and pays the worst-case width on *every* coordinate of *every* segment, even though the segments have very different distributions: `z` (folded response) is norm-bounded and sub-Gaussian; `e` (`e_folded`, partial evaluation at the opening point) is a full ring element per block and is **not** sparse or near-binary even when the witness polynomial is one-hot; inner commitment state `t` and quotient `r` are full-entropy field data; optional tiered `û_concat` is bounded.

This spec defines a comprehensive tail wire encoding built on one principle: **encode each wire object according to its actual distribution and role.**
It composes four levers, three of which are commitment/quotient *elision* (do not send objects whose information the verifier can reconstruct or check directly) and one of which is *entropy coding* (send norm-bounded witness segments at their true entropy, not at a worst-case fixed width).
The verifier reconstructs every elided or de-decomposed quantity deterministically, so soundness is unchanged and the wire shrinks to the realized information content.

**Implementation scope (#190).** This PR lands the **transparent non-zk encoding slice** only: `CleartextWitnessProof::SegmentTyped` with Golomb–Rice `z` (cap-derived `wire_rice_low_bits`), raw field segments for `e`/`t`/`r`, a length-prefixed `z` payload, planner upper-bound sizing, and non-zk production cutover. Wire tightening (`wire = cap - 2`) and planner `cap + 2` model ship in #209. Commitment elision (`r`-drop, terminal `t`-state / `u`-elision), descriptor binding, zk tail cutover, and Jolt decode measurement remain deferred per Execution below.

## Intent

### Goal

Replace the single fixed-width `PackedDigits` terminal witness with a segment-typed, entropy-coded cleartext tail whose per-segment models are derived from public/transcript-bound parameters with zero side information on the wire. The full umbrella also elides commitments and quotient segments the verifier can reconstruct; **#190 does not ship those elision levers yet** (see Execution).

The feature introduces or modifies:

- A **segment-typed tail witness** representation replacing the single-width `PackedDigits` blob on the transparent recursive terminal tail: **`Gaussian{rice_low_bits}` Golomb–Rice for `z` only**; **`RawField` for `e`/`t`/`r`** (full base-field coefficients per ring element). Boundaries derive from the schedule/shape (headerless wire). Wire `rice_low_bits` is never on the wire; both sides derive it from the fold `‖z‖_inf` cap (`min(β_inf, t*)` or `β_inf` alone) for `z` only.
- A **canonical, total Golomb-Rice codec** (`akita-types`, verifier-reachable, no-panic) with zigzag sign mapping and standard unary+remainder Rice encoding only. Decode rejects unary runs longer than [`golomb_rice_max_quotient_for_cap`](crates/akita-types/src/golomb_rice.rs). **#190 applies it to `z` only.**
- A per-level **fold `‖z‖_inf` cap accessor** via [`LevelParams::fold_witness_linf_cap_for_claims`](crates/akita-types/src/layout/params.rs) and the deterministic cap→wire low-bits rules in [`tail_golomb_rice_low_bits`](crates/akita-types/src/tail_golomb_rice_low_bits.rs) for the folded-response `z` segment.
- **Terminal `t`-state cutover** *(umbrella S2, deferred)*: the penultimate fold stops sending outer `u = B * t_hat` and binds inner `t = A * w_terminal` as the terminal public state.
- **`r`-elision and terminal-stage-2 elision** *(umbrella S1, deferred)* via PR #141 direct-terminal mode. **#190 still carries `r` on the wire** as a raw field segment.
- **Descriptor binding** *(umbrella S5, deferred)* of the tail-encoding policy in `AkitaInstanceDescriptor`. **#190** updates planner proof-size accounting and regenerates non-zk shipped tables under segment-typed sizing.

### Invariants

1. **Lossless and soundness-neutral.** Every elided object (`u`, `v`, `r`) is reconstructed or checked directly by the verifier; every entropy-coded segment decodes bit-exactly to the same integer vector a fixed-width encoding would carry. The terminal relation `M_terminal * z == y_terminal` in `F[X]/(X^D+1)` and the digit-range / norm bounds the extractor relies on are unchanged. Protected by: existing terminal direct-relation row tests (PR #141), new round-trip codec tests, and an e2e tamper test that a witness violating the digit/norm bound cannot produce an accepting transcript.
2. **Canonical encoding.** Each integer vector has exactly one valid byte encoding under a fixed `(model, rice_low_bits)`; a non-canonical or malformed encoding is rejected with `AkitaError`/`SerializationError`, never decoded ambiguously. Protected by: a canonicality unit test (encode-decode-encode fixpoint) and a malformed-bytes rejection test.
3. **Total, bounded, no-panic decode.** The Golomb-Rice decoder terminates on every byte string, rejects unary quotients above the cap-derived maximum, allocates only the schedule-declared element count, rejects non-minimal trailing byte padding, and never panics, unwraps, or indexes unchecked. Protected by: a fuzz/edge unit test over random byte strings and the verifier no-panic audit.
4. **Public models, zero side info.** Segment boundaries and Golomb wire low bits for `z` (`cap`, `wire_rice_low_bits`) are derivable by the verifier from `LevelParams` + transcript before decode; `e`/`t`/`r` carry no separate model tag. Protected by: a prover/verifier cap/wire low-bits agreement test for `z` and a `LoggingTranscript` event-stream equality test.
5. **Terminal `t`-state preserves weak binding.** The last recursive transition changes the terminal input state from the outer image `u = B * t_hat` to the inner image `t = A * w_terminal`. Soundness does **not** come from simply deleting B rows while keeping `u` as the statement; that would be unsound. It comes from making `t` the transcript-bound public terminal state and checking, in the direct terminal relation, that the revealed clear witness maps to that exact `t` under the A rows. Protected by: the soundness paragraph in Design, plus terminal-root / suffix-terminal row tests extended to the `t`-state layout.
6. **Descriptor binding distinguishes the policy.** A proof produced under one tail-encoding policy (codec, models, terminal-state mode, r-drop flags) must not verify under another. Protected by: a pinned descriptor-bytes test and a cross-policy verify-fails test.
7. **Transparent-only.** Entropy coding applies only to the transparent tail. **#190:** non-zk builds emit `SegmentTyped`; `feature = "zk"` keeps `PackedDigits` via compile-time gating (`#[cfg(feature = "zk")]`). Umbrella acceptance of `InvalidSetup` on zk policy selection is deferred to the zk tail slice.

### Non-Goals

- **No general intermediate-level change.** Non-terminal recursive folds still commit their next witness with the usual outer `u` commitment. The only exception is the **last recursive transition into the transparent terminal**, whose next-state payload is `t` instead of `u`.
- **No zero-fold tail encoding.** Zero-fold / root-direct schedules keep `CleartextWitnessProof::FieldElements`; they do not use the `z`/`e`/`t` segment layout.
- **No tiered multipoint tail.** Tiered commitment layouts require `num_points == 1` today; multipoint tiered terminal encoding is out of scope until that restriction is lifted.
- **No ZK tail encoding.** Masked/blinded witnesses are near-uniform and do not compress; a ZK direct/entropy tail needs a separate masked-relation design (same boundary PR #141 draws).
- **No change to the norm bound or the decomposition basis.** This spec changes how the realized witness is *encoded*, not the bound `K`/`t*` the verifier enforces (PR #174) nor the SIS rank pricing.
- **No new commitment scheme.** The t-reveal reuses the existing A-row check; it does not introduce a new matrix or assumption.
- **No entropy coding for `e`.** The `e` segment carries `e_folded` (partial-evaluation ring elements, `e_i = ⟨a, f_i⟩` before digit decomposition). Witness sparsity (one-hot, etc.) does not make `e` small on the wire; Golomb–Rice applies only to norm-bounded `z`.
- **Not the r-drop itself.** Terminal `r_hat` and terminal stage-2 elision are PR #141; this spec depends on it and only adds the encoding and terminal `t`-state cutover on top.

## Evaluation

### Acceptance Criteria

Encoding slice (#190):

- [x] A `GolombRice` codec in `akita-types` is canonical (encode-decode-encode fixpoint, minimal byte length), total (terminates and is no-panic on malformed/truncated bytes), and bijective on `[-cap, cap]` at wire low bits, verified by cap-range round-trip tests.
- [x] `fold_witness_linf_cap_for_claims` and the deterministic cap→low-bits rules (`cap_rice_low_bits`, `wire_rice_low_bits`) return integers pinned against reference calculations; prover and verifier derive the same wire low bits for `z`.
- [x] The transparent non-zk terminal `final_witness` serializes as segment-typed payloads (`z` length-prefixed Golomb bytes, then raw `e`/`t`/`r` field segments); `CleartextWitnessShape::admits_realized` accepts exact `z` payloads up to the schedule upper bound; `segment_typed_expand_matches_logical_w` proves expand matches legacy digit layout.
- [x] Non-zk shipped schedule tables regenerated under segment-typed terminal sizing; `generated_schedule_tables_match_find_schedule` passes for affected families.
- [x] Net `z` entropy win on cited `onehot_fp128_d64` cells (Golomb at `wire_rice_low_bits(cap)` beats legacy `PackedDigits`); profile emits structured tail breakdown (`proof tail summary`, Golomb vs packed `z` stats).

Umbrella items (deferred past #190):

- [ ] Under PR #141 direct terminal mode: no `r_hat` on the wire and no terminal stage-2 sumcheck bytes; under terminal `t`-state mode, the last recursive transition sends no parent `u = B * t_hat` commitment and the terminal relation has no B/COMMIT block.
- [ ] Terminal `t`-state soundness: tampering `t`, `e`, or `z` is rejected; the terminal A-row check enforces that the revealed witness maps to the transcript-bound `t` state.
- [ ] Descriptor bytes bind the tail-encoding policy and cross-policy verify fails.
- [ ] Under `feature = "zk"`, selecting the segment-typed tail policy rejects with `InvalidSetup` (or an equivalent explicit zk tail policy).
- [ ] In-guest decode cost measured via `final_witness_decode` in `profile/akita-recursion`; net `akita_verify` cycles do not regress versus the PR #141 direct-mode baseline.

### Testing Strategy

Must keep passing: all `akita-types` proof/shape/serialization tests, the terminal direct-relation row tests (PR #141), the schedule drift guards, e2e batched/multipoint/recursive suites, transcript and zk suites.

New tests:

- `akita-types`: Golomb-Rice round-trip, canonicality (minimal byte length), malformed/unary-above-cap edge cases, no-panic over random bytes; cap/wire low-bits reference pins; segment layout decode boundaries; `TerminalLevelProofShape` round-trips the segment-typed tail and rejects cross-policy shapes.
- `akita-prover`/`akita-verifier`: prover/verifier cap/wire low-bits agreement; terminal `t`-state row check (send `t`, check revealed witness maps to `t`, no terminal B/COMMIT block) for suffix-terminal and terminal-root; verifier no-panic on malformed tail bytes; tiered terminal layouts reject or route through the same `t`-state path without `û_concat`.
- e2e: tamper tests for `z`, `e`, `t`, segment shape, and `y_rings`; descriptor cross-policy reject; ZK rejection; post-decode norm-bound violation reject.
- `LoggingTranscript`: event-stream equality across the new tail schedule.

Feature combinations: default, `--no-default-features`, `--features zk` (policy rejects), `--features logging-transcript`.

### Performance

Expected direction: smaller proofs, no material prover slowdown, bounded verifier decode.

Per-lever, at the affected tail modes (verified by `AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 cargo run --release --example profile` and the planner `exact_proof_bytes`):

- **r-drop + terminal stage-2 drop** (PR #141 direct mode): **deferred**; projected **5.25%–6.15%** proof reduction on np1 profiles (secondary citation from PR #141 / JL analysis).
- **u-drop (terminal `t`-state)**: **deferred**; projected ~1 KB savings on cited fp128 D64 profiles (one `n_b`-ring-element `next_w_commitment` elided).
- **z entropy coding** *(landed in #190; cap-aligned on fold-linf #189; wire tightened in #209)*: `z` coordinate cost drops from fixed `depth_fold * log_basis` bits/coord (packed digits) to `rice_low_bits + O(1)` bits/coord on wire at `wire_rice_low_bits(cap) = cap_rice_low_bits(cap) - 2` (~10 bits/coord on fp128 D64 vs 15 packed when `cap = β_inf`). Realized random witnesses average ~9.7–10 bits/coord. Planner budgets use `cap_rice_low_bits(cap) + 2` bits/coord ([`TAIL_Z_PLANNER_CAP_LOW_BITS_PLUS_TWO`]). Both wire and planner reference the same fold `‖z‖_inf` cap as `num_digits_fold` and grind acceptance, not the level variance envelope `isqrt_ceil(β_inf² · T_level · ρ²)` (which would imply `cap_rice_low_bits ≈ 22`).
- **`e`/`t`/`r` as `RawField`** *(landed in #190)*: wire carries full ring/base-field coefficients for partial-evaluation `e_folded`, inner state `t`, and quotient `r`. No entropy coding; planner charges `field_bytes` per coefficient. Legacy `PackedDigits` unfairly priced `e_hat` digit planes at the same width as `z`.

A-role rank, setup size, and L2 pricing are unchanged.

## Design

### Architecture

The unifying classification: every tail wire object is in exactly one bucket, and each bucket has one correct encoding.

| object | nature | correct encoding | lever |
|---|---|---|---|
| `v = D * e_hat` | auxiliary D image | do not send; reveal `e` at the tail | D-drop (done, PR #88) |
| terminal input `u = B * t_hat` | outer commitment to terminal witness | do not send `u`; send inner state `t = A * w_terminal` and check it directly | terminal `t`-state (this spec, S2) |
| `z` (folded response) | norm-bounded per coefficient (`cap`) | Golomb-Rice keyed by `wire_rice_low_bits(cap)` | this spec (S4) |
| `e` (`e_folded`, partial evaluation) | full ring element per block (not sparse) | raw field coefficients (`RawField`) | **#190** |
| `t` (inner commitment image) | full field entropy | raw field coefficients (`RawField`) | **#190** |
| tiered `û_concat` | outer-commitment artifact | absent under terminal `t`-state | umbrella S2 (deferred) |
| `r` quotient | auxiliary, not in the statement | do not send; direct ring relations | PR #141 S1 (deferred; **#190 still sends `r` as `RawField`**) |

Affected surfaces:

- `akita-types`: the codec, the segment-typed `CleartextWitnessProof` variant and its shape, `direct_witness_bytes` / `proof_size.rs` tail accounting, the `cap`/low-bits accessors on `LevelParams` (via `fold_witness_linf_cap_for_claims` and `tail_golomb_rice_low_bits`), and the descriptor policy fields.
- `akita-prover`: emit the tail as typed segments (Golomb `z` on centered fold-response integers; `RawField` for `e_folded`/`t`/`r`), send terminal `t` as the next-state payload at the last recursive transition *(umbrella S2)*, and skip the parent `u = B * t_hat`.
- `akita-verifier`: decode typed segments (no-panic), expand `z` to digit planes and decompose `e`/`t` field segments to digit planes for row checks, verify terminal A rows against transcript-bound state.
- `akita-planner`/`akita-config`: codec-aware tail byte accounting, regenerate shipped tables under the new tail policy, bind the policy in the descriptor.

The current single-width path (`PackedDigits`) is retained for the **zk** tail (`#[cfg(feature = "zk")]`). The transparent **non-zk** tail routes through `CleartextWitnessProof::SegmentTyped` (#190).

### Terminal `t`-state / u-elision

The current relation row layout is `consistency | A(n_a) | B(n_b) | D(n_d)` (`crates/akita-types/src/layout/params.rs`). Public openings bind through the fused trace term in stage-2 sumcheck, not through M public rows. Tiered commitment (`B_inner`, second-tier `F`) was removed in PR #257.
The terminal already sets `n_d_active = 0` (`WithoutDBlock`) by revealing `e_folded` in cleartext: the D-role commitment `v = D * e_hat` and its rows are gone (`specs/terminal-fold-cutover.md`).

The correct S2 cutover is **not** "delete B rows while keeping `u` as the terminal statement."
That would be unsound: if the verifier still accepted a public commitment `u` but no longer checked `B * t_hat = u`, the terminal proof would no longer depend on `u`.

Instead, S2 changes the terminal input state.
At the last recursive transition, the proof sends the terminal inner image

```text
t = A * w_terminal
```

as raw ring elements, in the transcript slot where the next-state payload is bound.
It does **not** send the outer image

```text
u = B * decompose(t).
```

The terminal direct relation then has no `COMMIT`/B block.
Its A block is no longer a zero-RHS quotient row whose `t` contribution lives only inside the witness MLE; it is a direct check that the revealed terminal witness maps to the public `t` state.
Equivalently, the verifier computes the same folded A-row value from the revealed witness and checks equality to the transcript-bound `t` rows in `F[X]/(X^D+1)`.

Soundness sketch.
The terminal state is now `t`, not `u`.
The prover must bind the canonical bytes of `t` before any terminal challenge depending on the terminal state is squeezed.
The terminal verifier decodes the clear witness, checks the public norm/digit bounds, recomputes the A-image of that witness under the terminal fold challenges, and rejects unless it equals the bound `t`.
Thus an accepting proof gives an explicit short preimage of the public `t` under `A`; by the same Module-SIS weak-binding argument already used for the A-role, two different accepting terminal witnesses for the same `t` extract a short kernel vector of `A`.
No `B` binding is required because `u` is no longer part of the terminal statement.

This is why the cutover is sound: it replaces the statement `u = B * decompose(A * w_terminal)` with the statement `t = A * w_terminal` at the transparent tail, where `w_terminal` is revealed and range-checked.
It must be implemented as a distinct terminal row layout (for example, `WithoutDAndCommitBlock`) rather than as a local omission of `relation_n_b` from the current `WithoutDBlock` layout.
Tiered layouts follow the same rule: the terminal state is the inner A-image `t`, so neither the second-tier `F` rows nor the tiered `B_inner` / `û_concat` consistency rows are terminal statement rows.
If the implementation cannot route a tiered terminal through this `t`-state layout in the first slice, it must reject that policy rather than silently keeping `û_concat` in the transparent tail.

### The segment-typed tail encoder

The transparent tail witness becomes an ordered list of typed segments, replacing the single `PackedDigits` blob:

```rust
pub enum TailSegmentModel {
    /// Norm-bounded folded-response integers; Golomb-Rice with wire `rice_low_bits = wire_rice_low_bits(cap)`.
    Gaussian { rice_low_bits: u32 },
    /// Full field/base coefficients for one ring element (e_folded, t, r).
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
z_coords       = num_z_segments * block_len * num_digits_commit * D   (Golomb integers; one per base-field slot in folded z)
e_field_elems  = num_blocks * num_w_vectors * D                       (RawField; one ring element per block → D coeffs)
t_field_elems  = n_a * num_blocks * num_t_vectors * D                   (RawField)
r_field_elems  = relation_matrix_row_count_for(WithoutDBlock) * D                   (RawField; until PR #141 r-drop)
```

`z_coords` is the Golomb element count. `e_field_elems`, `t_field_elems`, and `r_field_elems` are base-field coefficient counts for `RawField` serialization.
The legacy `PackedDigits` layout in `build_w_coeffs` (`coeffs.rs`) is the source for S3's byte-neutral framing, but S2 removes the legacy `t_hat`, `û_concat`, and `r_hat` planes from the final terminal policy.
Segments appear in wire order `z ‖ e ‖ t ‖ r`; `r_hat` planes are absent under PR #141 direct mode.
Multipoint layouts scale `z_coords` with `num_z_segments`.

This mirrors the existing headerless, shape-driven decode (the shape supplies counts and the `z` payload upper bound). `CleartextWitnessShape::SegmentTyped` and `CleartextWitnessProof::SegmentTyped` ship in #190.

The verifier decodes `z` via Golomb–Rice into centered integers and expands to balanced digit planes; decodes `e`/`t`/`r` as raw field coefficients and **decomposes** `e`/`t` to digit planes on the verifier side for the existing row layout (`expand_segment_typed_to_i8_digits`). The wire never carries `e_hat` digit planes for `e`; it carries `e_folded` field elements.

### Golomb-Rice for the Gaussian z segment

For a signed integer `n` admitted by the `z` segment's public bound, zigzag to a non-negative `u = (n << 1) ^ (n >> (W-1))` where `W` is the signed-integer bit width from the fold cap (`golomb_rice_zigzag_width`). Rice-code `u` with low-bit width `rice_low_bits` as standard Golomb: unary quotient prefix + stop + remainder. Decode rejects unary length above `golomb_rice_max_quotient_for_cap(cap, rice_low_bits, W)`.

Wire low bits for the `z` segment are derived deterministically from the public per-coefficient fold `‖z‖_inf` cap:

```text
cap              = fold_witness_linf_cap_for_claims(num_claims)   (= min(β_inf, t*) or β_inf alone)
cap_rice_low_bits = rice_low_bits_for_cap(cap)                   (= max(0, floor(log2(cap))), pinned by test)
wire_rice_low_bits = cap_rice_low_bits - WIRE_RICE_LOW_BITS_DELTA  (= cap low bits - 2 today)
```

`cap` is the same bound already used by [`fold_witness_digit_plan`](crates/akita-types/src/sis/norm_bound.rs) and grind acceptance. It is **not** the level variance envelope `isqrt_ceil(β_inf² · T_level · ρ²)` from PR #174's `t*` analysis: that quantity aggregates coordinates and is far too loose for per-coordinate Golomb-Rice parameterization (it would imply `cap_rice_low_bits ≈ 22` on fp128 D64 where `cap_rice_low_bits = 12` suffices).

`rice_low_bits_for_cap` is the cap-derived Rice low-bit width covering every admitted `z` coefficient magnitude at the planner reference. `wire_rice_low_bits` is what prover and verifier use on the wire (#209). The codec is canonical because standard Rice is bijective for a fixed `rice_low_bits` on `[-cap, cap]`, and wire payloads must use the minimal byte length (partial-byte zero padding only). Decode and grind use the cap-derived maximum quotient; no alternate wire shape exists.

### RawField segments (`e`, `t`, `r`)

Serialize `FlatRingVec` coefficients in field canonical form (`field_bytes` each). No Golomb, no per-segment model tag. The verifier decomposes `e` and `t` to balanced digit planes after decode when rebuilding the legacy digit stream for row arithmetic.

### Descriptor and wire binding

Tail encoding uses three layers:

| Layer | Source | Carries |
|-------|--------|---------|
| **Policy** | `AkitaInstanceDescriptor` (new tail section + version bump) | codec id, cap→wire low-bits rule id, per-role segment models, terminal-state mode, r-drop flag |
| **Layout** | `CleartextWitnessShape` / `TerminalLevelProofShape` (S3) | per-segment element counts (and byte length once entropy-coded) |
| **Derived** | both sides at runtime | `cap`, `wire_rice_low_bits`, segment order (`z ‖ e ‖ t ‖ r`), decode bounds |

The tail-encoding policy is bound in `AkitaInstanceDescriptor` (same pattern as PR #141's terminal proof mode and PR #174's threshold policy):

- codec identity (Golomb-Rice variant id for `z`, cap-derived max-quotient rule, zigzag width rule `W_z`),
- the cap→wire low-bits rule identity for the `z` segment only,
- per-segment model assignment: `Gaussian` for `z`, `RawField` for `e`/`t`/`r`,
- the terminal-state mode (`OuterCommitmentU` legacy vs `InnerImageT` tail policy) and r-drop flags.

The proof shape carries segment **counts** (and, for variable-length entropy segments, the realized payload byte length). Model tags and wire low bits are **not** duplicated on the shape; the verifier checks `shape.counts == derive_counts(policy, level_params, incidence)` before decode.
Prover, verifier, schedule digest, and descriptor must agree; any disagreement rejects before hot arithmetic.
Transcript binding (S3 acceptance): absorb **canonical encoded segment bytes** in the same logical order as today's `ABSORB_TERMINAL_E_HAT` / `ABSORB_TERMINAL_W_REMAINDER` events (`terminal_witness.rs`), updated for segment boundaries.

### Proof-size accounting

`direct_witness_bytes` (`crates/akita-types/src/layout/proof_size.rs:32-41`) gains a segment-typed arm: runtime sizing sums exact encoded segment sizes; **planner** sizing uses a conservative upper bound on entropy segments (`~num_coords * (cap_rice_low_bits + 2)` bits for Gaussian `z` under [`TAIL_Z_PLANNER_CAP_LOW_BITS_PLUS_TWO`]), consistent with the repo's `actual <= planned + ACCEPTED_PLANNER_PROOF_SIZE_OVERCOUNT_BYTES` profile gate.
`level_proof_bytes` (`crates/akita-types/src/proof_size.rs:72-106`) drops `next_commit_bytes` for the last recursive transition and charges the raw `t` payload on the terminal direct witness instead (the terminal layout already drops `v_bytes`/stage-1 at `WithoutDBlock`).
Shipped schedule tables are regenerated under the new tail policy in S5; the drift guard `generated_schedule_tables_match_find_schedule` must pass after regen.

### Alternatives Considered

- **rANS / range coding against the discrete-Gaussian CDF.** Within ~0.01 bit of entropy vs Golomb-Rice's ~0.1-0.5 bit, but the decoder is a heavier state machine to make canonical, total, and cheap in-guest. Golomb-Rice captures nearly all the win at a fraction of the verifier/Jolt complexity; rANS is a future opt-in if the residual bits justify it.
- **Keep `PackedDigits`, only add per-segment widths.** Still leaves the `z` worst-case-width slack; segment-typed `RawField` for `e` is the correct model (partial evaluation is full field), not digit-plane packing at `z`'s width.
- **Transmit a small histogram/model.** Unnecessary: `β_inf` is public/transcript-derivable, so a parametric model costs zero wire bytes.
- **Single fixed Rice low-bit width across all modes.** Simpler but loses per-level `β_inf` matching for `z`; the `cap -> wire_rice_low_bits` rule is descriptor-bound for the `z` segment only.
- **BoundedSmall / Golomb for `e`.** Rejected: `e` is `e_folded` partial evaluation (full ring elements). Witness sparsity does not shrink `e` on the wire.
- **Apply entropy coding at intermediate levels.** Impossible: intermediate witnesses are committed, not revealed; only the tail is cleartext.

## Documentation

- Update PR #141 branch `specs/terminal-direct-ring-relation.md` and `specs/terminal-fold-cutover.md` (PR #88) to cross-link this spec as the encoding layer on top of the r-drop and D-drop, and to record the terminal `t`-state cutover as the replacement for the old terminal `u` opening.
- Update PR #174 / `specs/fold-linf-rejection.md` cross-link: Golomb `z` sizing uses the same fold `‖z‖_inf` cap as `num_digits_fold` (distinct from the level variance envelope).
- Profile example / bench reports: structured tail witness reporting is implemented in `crates/akita-pcs/examples/profile/report.rs` (`emit_proof_tail_report`) and `scripts/profile_bench_report.py`. The profile binary (non-zk) emits `proof tail summary` with `final_w_encoding` / `final_w_policy` and, for `segment_typed`, per-segment wire bytes plus Golomb-vs-`PackedDigits` `z` stats. Encoding variants on `CleartextWitnessProof`: `segment_typed` (non-zk folded terminal default), `packed_digits` (`feature = "zk"` fallback), `field_elements` (root-direct cleartext witness), and `none` (root-direct zero-fold; `tail_bytes = 0`).

## Execution

Umbrella dependency order (full tail wire encoding):

```text
S1  Land PR #141 (r-drop + terminal direct ring relations).        deferred
S2  Terminal t-state / u-elision (+ soundness paragraph).          deferred (depends on S1)
S3  Segment-typed tail framing (replace single PackedDigits with    landed in #190 (non-zk)
    typed segments; raw e/t/r + Golomb z; length-prefixed z)
S4  Golomb-Rice codec + cap/k for z segment only                  landed (#190 + #189 cap align)
S5  Descriptor binding + Jolt decode measurement + full S5 polish   partially landed (#190 + #209)
    (planner regen, profile tail accounting, wire low-bits rule in `FoldLinfProtocolBinding`,
    verifier `decode_terminal_z_golomb_payload` admissibility, grind planner budget gate)
```

**#190** is the first implementation PR under this umbrella. It is independently shippable: measurable `z` entropy win on non-zk folded terminals without waiting for S1/S2 elision.

Risks for remaining umbrella slices:

- The S2 soundness paragraph must state the terminal statement cutover precisely: `u` is no longer the terminal state, `t` is transcript-bound instead, and the terminal A rows check the revealed witness against `t`.
- S3/S4 (landed in #190) must keep prover packing, verifier transcript slicing, and verifier row decoding byte-for-byte aligned across segment boundaries; `segment_typed_expand_matches_logical_w` guards this.
- S5 descriptor binding (wire low-bits rule id + δ in [`FoldLinfProtocolBinding`](../../crates/akita-types/src/instance_descriptor/fold_linf_binding.rs); `AKITA_INSTANCE_DESCRIPTOR_VERSION = 2`) and Jolt cycle measurement remain partially open.

## References

- PR #141 branch `specs/terminal-direct-ring-relation.md`: terminal r-drop and direct ring relations; S1 dependency.
- PR #174 / `specs/fold-linf-rejection.md` (closed spec PR; implementation #189): the `t*` threshold and fold `‖z‖_inf` cap; Golomb `z` sizing uses the same cap as `num_digits_fold`, not the level variance envelope.
- `specs/terminal-fold-cutover.md` (PR #88): the D-role drop whose transcript-binding discipline the terminal `t`-state cutover reuses.
- `specs/weak-binding-norm-fix.md`: the weak-binding object the tail extraction recovers.
- `crates/akita-types/src/proof/direct_witness.rs` (`PackedDigits`, `CleartextWitnessProof`), `crates/akita-types/src/proof/terminal_witness.rs` (`terminal_witness_segment_layout`, transcript slicing), `crates/akita-types/src/proof/levels.rs:704-731` (`TerminalLevelProof`), `crates/akita-types/src/proof/ring_relation.rs:226-267` (segment plane counts).
- `crates/akita-types/src/proof_size.rs:72-106` and `crates/akita-types/src/layout/proof_size.rs` (proof-byte and witness accounting).
- `crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs:351-374` (A/B/D row roles), `crates/akita-prover/src/protocol/ring_switch/coeffs.rs` (`build_w_coeffs` segment order), `crates/akita-prover/src/protocol/ring_switch/commit.rs` (`u = B * t_hat`).
- Profile: `AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 cargo run --release --example profile`.
