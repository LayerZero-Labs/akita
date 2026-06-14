# Spec: Tail Wire Encoding (commitment elision + segment-typed entropy coding)

| Field       | Value                                                     |
|-------------|-----------------------------------------------------------|
| Author(s)   | Quang Dao                                                 |
| Created     | 2026-06-13                                                |
| Status      | proposed                                                  |
| PR          | #187 (umbrella spec; implementation stack in Execution) |

## Summary

The Akita proof tail (the terminal recursive fold and the terminal-root 1-fold case) is the only place a folded witness is sent in cleartext rather than as a commitment.
Today that cleartext witness is one `PackedDigits` blob assembled in `crates/akita-prover/src/protocol/ring_switch/coeffs.rs` (`build_w_coeffs`) and packed via `PackedDigits::from_i8_digits_with_min_bits` (`crates/akita-types/src/proof/direct_witness.rs:165-169`): a single uniform `bits_per_elem` (at least `final_log_basis`, widened per witness if needed, capped at `log_basis <= 6`), with the folded response `z`, opening digits `e_hat`, inner-commitment digits `t_hat`, optional tiered `û_concat`, and quotient digits `r_hat` all concatenated and bit-packed at that one width.
The zero-fold fast path uses `CleartextWitnessProof::FieldElements` (raw polynomial coefficients), not this segment layout; see Non-Goals.
This is fixed-width to a worst-case bound and pays the worst-case width on *every* coordinate of *every* segment, even though the segments have very different distributions: `z` is sub-Gaussian, one-hot opening digits `e` are near-binary, inner commitment digits `t_hat` (today) are full-entropy, optional tiered `û_concat` is bounded, and quotient digits `r_hat` are not part of the terminal statement at all.

This spec defines a comprehensive tail wire encoding built on one principle: **encode each wire object according to its actual distribution and role.**
It composes four levers, three of which are commitment/quotient *elision* (do not send objects whose information the verifier can reconstruct or check directly) and one of which is *entropy coding* (send norm-bounded witness segments at their true entropy, not at a worst-case fixed width).
The verifier reconstructs every elided or de-decomposed quantity deterministically, so soundness is unchanged and the wire shrinks to the realized information content.

## Intent

### Goal

Replace the single fixed-width `PackedDigits` terminal witness with a segment-typed, entropy-coded cleartext tail that carries no commitments and no quotient, and whose per-segment models are derived from public/transcript-bound parameters with zero side information on the wire.

The feature introduces or modifies:

- A **segment-typed tail witness** representation replacing the single-width `PackedDigits` blob on the transparent recursive terminal tail: a sequence of typed segments (`Gaussian{k}`, `BoundedSmall{k}`, `RawField`), each carrying only its payload bytes, with boundaries derived from the schedule/shape and models from the descriptor policy (headerless, like the rest of the wire format). `k` is never on the wire; both sides derive it from public `sigma` via the descriptor-bound `sigma -> k` rule.
- A **canonical, total Golomb-Rice codec** (`akita-types`, verifier-reachable, no-panic) with zigzag sign mapping and a bounded-unary escape, parameterized by the derived integer Rice parameter `k`.
- A per-level **`sigma` accessor** `LevelParams::fold_response_sigma(num_claims, num_public_rows)` deriving the folded-witness standard-deviation proxy from the same variance envelope PR #174 uses for `t*`: `V = sigma_inf^2 * T_level * rho2`, `sigma = isqrt_ceil(V)` (smallest integer `s` with `s^2 >= V`; see Golomb-Rice section), and the deterministic `sigma -> k` rule.
- **`t`-reveal at the penultimate committed tier** (the B-role mirror of the existing D-role drop): reveal the inner commitment value `t` as `RawField` ring elements in place of the parent commitment `u = B * t_hat` on the *next* level's wire, dropping the `n_b` B-rows from that fold's relation and the `u` commitment bytes from the proof.
- **`r`-elision and terminal-stage-2 elision** via the spec-approved terminal direct ring-relation mode ([PR #141](https://github.com/LayerZero-Labs/akita/pull/141), branch `quang/terminal-direct-ring-relation-spec`), on which this spec depends and which it does not re-specify. These apply only under the direct terminal proof mode that #141 introduces; the current shipped terminal still carries `r_hat` and relation-only stage-2.
- **Descriptor binding** of the active tail-encoding policy (codec identity, per-segment model identities, the `sigma -> k` rule, the B-drop and r-drop flags) in `AkitaInstanceDescriptor`, and **proof-size/planner accounting** updated to the new tail shape.

### Invariants

1. **Lossless and soundness-neutral.** Every elided object (`u`, `v`, `r`) is reconstructed or checked directly by the verifier; every entropy-coded segment decodes bit-exactly to the same integer vector a fixed-width encoding would carry. The terminal relation `M_terminal * z == y_terminal` in `F[X]/(X^D+1)` and the digit-range / norm bounds the extractor relies on are unchanged. Protected by: existing terminal direct-relation row tests (PR #141), new round-trip codec tests, and an e2e tamper test that a witness violating the digit/norm bound cannot produce an accepting transcript.
2. **Canonical encoding.** Each integer vector has exactly one valid byte encoding under a fixed `(model, k)`; a non-canonical or malformed encoding is rejected with `AkitaError`/`SerializationError`, never decoded ambiguously. Protected by: a canonicality unit test (encode-decode-encode fixpoint) and a malformed-bytes rejection test.
3. **Total, bounded, no-panic decode.** The Golomb-Rice decoder terminates on every byte string (bounded unary via the escape), allocates only the schedule-declared element count, and never panics, unwraps, or indexes unchecked. Protected by: a fuzz/edge unit test over random byte strings and the verifier no-panic audit.
4. **Public models, zero side info.** Every per-segment model parameter (`sigma`, `k`, segment presence, segment boundaries) is derivable by the verifier from `LevelParams` + descriptor + transcript before the segment is decoded; no model, histogram, or width is transmitted. Protected by: a prover/verifier `sigma`/`k` agreement test (mirroring the `beta_linf_fold_bound` / `num_digits_fold` mirror invariant of PR #174) and a `LoggingTranscript` event-stream equality test.
5. **B-role drop preserves weak binding (pending S2 proof).** Revealing `t` and checking `t = A * e` (A-rows) in place of committing `u = B * t_hat` (B-rows) is *intended* to preserve the weak-binding object the extractor recovers, by the same argument as the existing D-role drop (`specs/terminal-fold-cutover.md`, PR #88). This invariant is **not satisfied until** the full S2 soundness paragraph is written and reviewed. Protected by: that paragraph (S2 gate), plus terminal-root / suffix-terminal row tests extended to the B-dropped layout.
6. **Descriptor binding distinguishes the policy.** A proof produced under one tail-encoding policy (codec, models, B-drop/r-drop flags) must not verify under another. Protected by: a pinned descriptor-bytes test and a cross-policy verify-fails test.
7. **Transparent-only.** Entropy coding and the t-reveal apply only to the transparent tail; under `feature = "zk"` the masked tail keeps the existing representation and the new policy rejects with `InvalidSetup`. Protected by: a zk-rejection regression test.

### Non-Goals

- **No intermediate-level change.** Intermediate folds commit their witness; their `u`/`v` commitments stay full field elements and their digits stay committed. This spec touches only the transparent recursive terminal tail (suffix-terminal and terminal-root 1-fold).
- **No zero-fold tail encoding.** Zero-fold / root-direct schedules keep `CleartextWitnessProof::FieldElements`; they do not use the `z`/`e`/`t` segment layout.
- **No tiered multipoint tail.** Tiered commitment layouts require `num_points == 1` today; multipoint tiered terminal encoding is out of scope until that restriction is lifted.
- **No ZK tail encoding.** Masked/blinded witnesses are near-uniform and do not compress; a ZK direct/entropy tail needs a separate masked-relation design (same boundary PR #141 draws).
- **No change to the norm bound or the decomposition basis.** This spec changes how the realized witness is *encoded*, not the bound `K`/`t*` the verifier enforces (PR #174) nor the SIS rank pricing.
- **No new commitment scheme.** The t-reveal reuses the existing A-row check; it does not introduce a new matrix or assumption.
- **Not the r-drop itself.** Terminal `r_hat` and terminal stage-2 elision are PR #141; this spec depends on it and only adds the encoding and the B-drop on top.

## Evaluation

### Acceptance Criteria

- [ ] A `GolombRice` codec in `akita-types` is canonical (encode-decode-encode fixpoint), total (terminates and is no-panic on arbitrary bytes), and bijective on the integer range it admits, verified by unit tests including the escape path.
- [ ] `LevelParams::fold_response_sigma(num_claims, num_public_rows)` and the deterministic `sigma -> k` rule return integers pinned against a reference calculation, and prover/verifier read the identical value.
- [ ] The transparent terminal `final_witness` serializes as segment-typed payloads with no per-segment header bytes; runtime `direct_witness_bytes` matches the exact serialized tail size, and the profile gate `actual_proof_bytes <= planned_proof_bytes + ACCEPTED_PLANNER_PROOF_SIZE_OVERCOUNT_BYTES` passes (planner uses a documented conservative upper bound on entropy-coded segments, not the realized witness length).
- [ ] Under PR #141 direct terminal mode, the transparent terminal proof contains no `r_hat` and no terminal stage-2 sumcheck bytes; under B-drop, the penultimate committed tier has no parent `u = B * t_hat` commitment and that fold's relation drops the B-rows.
- [ ] A terminal proof that reveals `t` verifies, and tampering `t`, `e`, or `z` is rejected; the A-row check `t = A * e` is enforced on recomposed ring elements.
- [ ] Descriptor bytes change intentionally and are pinned; a proof under the new policy fails to verify under the legacy `PackedDigits` policy and vice versa.
- [ ] Under `feature = "zk"`, selecting the entropy/t-reveal tail policy rejects with `InvalidSetup`; the masked tail is unchanged.
- [ ] Net proof-size reduction reported by the profile command at the affected modes, with the per-lever breakdown (r-drop, u-drop, z entropy, one-hot `e` width recovery) recorded.
- [ ] In-guest decode cost measured: add a `final_witness_decode` cycle marker in `profile/akita-recursion`; entropy-coded tail decode adds bounded cycles and net `akita_verify` cycles do not regress versus the PR #141 direct-mode baseline at the cited profile cell.
- [ ] `generated_schedule_tables_match_find_schedule` passes after S5 table regen under the tail policy.

### Testing Strategy

Must keep passing: all `akita-types` proof/shape/serialization tests, the terminal direct-relation row tests (PR #141), the schedule drift guards, e2e batched/multipoint/recursive suites, transcript and zk suites.

New tests:

- `akita-types`: Golomb-Rice round-trip, canonicality, malformed/escape edge cases, no-panic over random bytes; `sigma`/`k` reference pins; segment layout decode boundaries; `TerminalLevelProofShape` round-trips the segment-typed tail and rejects cross-policy shapes.
- `akita-prover`/`akita-verifier`: prover/verifier `sigma`/`k` agreement; B-dropped terminal row check (reveal `t`, check `t = A * e`, drop B-rows) for suffix-terminal and terminal-root; verifier no-panic on malformed tail bytes; tiered `û_concat` segment round-trip when `u_planes > 0`.
- e2e: tamper tests for `z`, `e`, `t`, segment shape, and `y_rings`; descriptor cross-policy reject; ZK rejection; post-decode norm-bound violation reject.
- `LoggingTranscript`: event-stream equality across the new tail schedule.

Feature combinations: default, `--no-default-features`, `--features zk` (policy rejects), `--features logging-transcript`.

### Performance

Expected direction: smaller proofs, no material prover slowdown, bounded verifier decode.

Per-lever, at the affected tail modes (verified by `AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 cargo run --release --example profile` and the planner `exact_proof_bytes`; numbers below are **projected** until the stack lands):

- **r-drop + terminal stage-2 drop** (PR #141 direct mode): measured **5.25%–6.15%** proof reduction on np1 profiles (secondary citation from PR #141 / JL analysis; pin primary profile table in S5).
- **u-drop (t-reveal at penultimate tier)**: removes one `n_b`-ring-element `next_w_commitment` (~1 KB on cited fp128 D64 profiles at steady-state `n_b`), with `t` already carried on the terminal tail as `RawField` (replacing today's digit-packed `t_hat` segment at comparable or smaller byte cost).
- **z entropy coding**: `z` coordinate cost drops from `~log2(2 * bound)` to `~log2(sigma) + 2.05` bits; ~`1.2` bits/coordinate vs the post-#174 `t*` bound, more vs `beta_inf`.
- **per-segment width recovery**: one-hot `e` segments stop paying `z`/`t`'s global `bits_per_elem`; their near-binary digits collapse toward their own entropy.

A-role rank, setup size, and L2 pricing are unchanged.

## Design

### Architecture

The unifying classification: every tail wire object is in exactly one bucket, and each bucket has one correct encoding.

| object | nature | correct encoding | lever |
|---|---|---|---|
| `u = B * t_hat`, `v = D * e_hat` | full entropy mod q | do not send; reveal preimage at the tail | D-drop (done, PR #88), B-drop (this spec, S2) |
| `z` (folded response) | sub-Gaussian, public `sigma` | Golomb-Rice keyed by derived `k` | this spec (S4) |
| one-hot `e` | sparse / near-binary | bounded-small entropy code | this spec (S4) |
| `t = A * e` | full entropy mod q | raw field elements (`RawField`) | this spec (S3; replaces digit-packed `t_hat`) |
| tiered `û_concat` | bounded opening digits | bounded-small entropy code (tiered only) | this spec (S4) |
| `r` quotient | auxiliary, not in the statement | do not send; direct ring relations | PR #141 (S1) |

Affected surfaces:

- `akita-types`: the codec, the segment-typed `CleartextWitnessProof` variant and its shape, `direct_witness_bytes` / `proof_size.rs` tail accounting, the `sigma`/`k` accessors on `LevelParams`, and the descriptor policy fields.
- `akita-prover`: emit the tail as typed segments (recomposed integers for `z`/`e`, raw field for `t`), reveal `t` and skip the parent `u = B * t_hat`, emit no B-rows for that fold.
- `akita-verifier`: decode typed segments (no-panic), re-decompose where needed for the row check, check `t = A * e` (A-rows), drop B-rows, no-panic shape validation.
- `akita-planner`/`akita-config`: codec-aware tail byte accounting, regenerate shipped tables under the new tail policy, bind the policy in the descriptor.

The current single-width path (`PackedDigits`, `direct_witness.rs:106-163`) is retained only for the ZK tail; the transparent tail routes through the segment-typed variant.

### The reveal-t / B-role drop

The relation row layout is `consistency | public | D(n_d) | B(n_b) | A(n_a)` (`crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs:351-374`), where `n_d = d_key.row_len()`, `n_b = b_key.row_len()`, `n_a = a_key.row_len()`.
The terminal already sets `n_d_active = 0` (`WithoutDBlock`) by revealing opening digits `e` in cleartext: the D-role commitment `v = D * e_hat` and its rows are gone (`specs/terminal-fold-cutover.md`).

The B-role is the exact mirror of the D-drop already shipped in PR #88 (`specs/terminal-fold-cutover.md`).
The parent commitment is `u = B * t_hat` on the **penultimate** committed tier (`level_proof_bytes` `next_commit_bytes` via `LevelParams::effective_commit_rows()`, `crates/akita-types/src/proof_size.rs:94-99`; `crates/akita-prover/src/protocol/ring_switch/commit.rs`).
The inner commitment value `t = A * e` (A-rows) is revealed on the terminal tail as `RawField` coefficients, replacing today's digit-packed `t_hat` segment.
So at the penultimate committed tier we drop the parent `u = B * t_hat` commitment and the `n_b` B-rows, exactly as revealing `e` dropped `v` and the D-rows at the terminal.

Soundness sketch (**S2 gate; must be written in full before coding B-drop**).
Weak binding at the tail rests on: (a) the revealed witness segments are bound in the transcript before any challenge derived from them is squeezed, (b) the verifier checks the reduced ring relation directly over `F[X]/(X^D+1)`, and (c) the A-row check `t = A * e` ties the revealed `t` to the bound opening digits recovered from the entropy-coded `e` segment.
Dropping the B-commitment removes a binding of `t_hat` via `B`, but `t` is now revealed and bound directly, and `t = A * e` is checked on recomposed ring elements, so the object the extractor recovers (the opening, via `e`) is unchanged in intent.
This is structurally identical to the D-drop argument; the full version must state the extraction order (bind revealed segment bytes, check A-rows, conclude), cover tiered `B_inner` / `û_concat` layouts, and confirm no remaining soundness term referenced the `B` commitment of `t_hat`. Coordinate with the JL R-A pinning / extraction-order lemma (`specs/akita-jl-norm-check-resolutions.md` §8).

### The segment-typed tail encoder

The transparent tail witness becomes an ordered list of typed segments, replacing the single `PackedDigits` blob:

```rust
pub enum TailSegmentModel {
    /// Sub-Gaussian integers; Golomb-Rice keyed by a transcript-derived k.
    Gaussian { k: u32 },
    /// Small bounded integers (e.g. one-hot opening digits); Golomb-Rice with a small k.
    BoundedSmall { k: u32 },
    /// Full-entropy field elements (commitment value t); raw field bytes.
    RawField,
}
```

The wire carries only concatenated segment payloads, with no per-segment header.
Segment boundaries (digit-plane counts, each plane spanning `D` ring coefficients) and emission order are derived by both sides from `LevelParams` + incidence, matching `RingRelationInstance::segment_layout` (`crates/akita-types/src/proof/ring_relation.rs:226-267`) and `build_w_coeffs` (`coeffs.rs`):

```text
z_planes   = num_digits_fold * num_digits_commit * num_public_rows * block_len
e_planes   = num_digits_open * num_blocks * num_w_vectors
t_elems    = n_a * D                                    (RawField; not digit planes)
u_planes   = u_concat_ring_len_per_group()                (tiered only; 0 otherwise)
```

Segments appear in `z_first`-dependent order (`crates/akita-prover/src/protocol/ring_switch/coeffs.rs`); `r_hat` planes are absent under PR #141 direct mode.
Multipoint layouts scale `z_planes` with `num_public_rows`; tiered layouts add `u_planes` and require `num_points == 1`.

This mirrors the existing headerless, shape-driven decode (the shape supplies counts; the descriptor supplies models). **S3** adds `CleartextWitnessShape::SegmentTyped(...)` and the matching proof variant; they do not exist in the codebase today.

The verifier decodes each segment to its integer or field vector, then re-decomposes the `z`, `e`, and tiered `û_concat` integers into balanced digits *on the verifier side* for the digit-range and row checks (wire carries recomposed integers, not digit planes).
`t` is carried as `RawField` base-field coefficients (`n_a * D`); the verifier uses them directly in A-row checks after binding the segment bytes in the transcript.
Post-decode, every coordinate must lie within the public digit/norm bound (`t*` from PR #174); out-of-range decoded integers are rejected before row arithmetic.

### Golomb-Rice for the Gaussian z segment

For a signed integer `n`, zigzag to a non-negative `u = (n << 1) ^ (n >> (W-1))` with `W` pinned to the base-field limb width used for digit storage (`0, -1, 1, -2, 2 -> 0, 1, 2, 3, 4`).
Rice-code `u` with parameter `k`: quotient `q = u >> k` in unary (`q` ones then a zero), remainder `u & (2^k - 1)` in `k` bits.
Bounded-unary escape: if `q >= Q_MAX`, emit `Q_MAX` ones then a fixed `base_field_bits`-bit literal of `u` (the base prime field modulus bit width from setup); this caps decode work and keeps the decoder total. Normal Rice and escape ranges must be disjoint; encoders must use escape exactly when `q >= Q_MAX`.

`k` is derived deterministically from the public `sigma`:

```text
V       = sigma_inf^2 * T_level * rho2          (PR #174 branch spec; variance envelope)
sigma   = isqrt_ceil(V)                          (smallest integer s with s^2 >= V)
k       = optimal_rice_k(sigma)                  (deterministic integer rule, pinned by test)
```

`isqrt_ceil` is the integer square root rounded up (equivalently `ceil(sqrt(V))` for non-negative `V`). This is the single canonical `sigma` rule; do not use the factored form `ceil(sqrt(T_level * rho2)) * sigma_inf`, which can disagree for `sigma_inf > 1`.

`optimal_rice_k` is the integer minimizing the expected Rice length for a discrete half-normal of scale `sigma` (the folded zigzag distribution), computed in integer/fixed-point arithmetic with a pinned reference; a robust closed form is `k = max(0, floor(log2(sigma)))` adjusted by a small tabulated correction, but the spec fixes whichever rule the reference test pins.
The codec is canonical because Rice coding is bijective for a fixed `k` and the escape branch is bijective on its range; the encoder must always choose the escape exactly when `q >= Q_MAX`.

One-hot `e` uses `BoundedSmall { k }` with a small `k` (often `0`, near-unary), which collapses the near-binary digits; dense `e` uses the same family with `k` from its own bound.

### Descriptor and wire binding

Tail encoding uses three layers:

| Layer | Source | Carries |
|-------|--------|---------|
| **Policy** | `AkitaInstanceDescriptor` (new tail section + version bump) | codec id, `sigma -> k` rule id, per-role segment models, B-drop / r-drop flags |
| **Layout** | `CleartextWitnessShape` / `TerminalLevelProofShape` (S3) | per-segment element counts (and byte length once entropy-coded) |
| **Derived** | both sides at runtime | `sigma`, `k`, segment order (`z_first`), decode bounds |

The tail-encoding policy is bound in `AkitaInstanceDescriptor` (same pattern as PR #141's terminal proof mode and PR #174's threshold policy):

- codec identity (Golomb-Rice variant id, zigzag width `W`, `Q_MAX`, escape `base_field_bits`),
- the `sigma -> k` rule identity and per-segment model assignment by role (`z`, `e`, `u_concat`, `t`),
- the B-drop and r-drop flags.

The proof shape carries segment **counts** (and, for variable-length entropy segments, the realized payload byte length). Model tags and `k` are **not** duplicated on the shape; the verifier checks `shape.counts == derive_counts(policy, level_params, incidence)` before decode.
Prover, verifier, schedule digest, and descriptor must agree; any disagreement rejects before hot arithmetic.
Transcript binding (S3 acceptance): absorb **canonical encoded segment bytes** in the same logical order as today's `ABSORB_TERMINAL_E_HAT` / `ABSORB_TERMINAL_W_REMAINDER` events (`terminal_witness.rs`), updated for segment boundaries.

### Proof-size accounting

`direct_witness_bytes` (`crates/akita-types/src/layout/proof_size.rs:32-41`) gains a segment-typed arm: runtime sizing sums exact encoded segment sizes; **planner** sizing uses a conservative upper bound on entropy segments (expected bits `~num_coords * (log2(sigma) + 2.05)` for Gaussian, plus margin), consistent with the repo's `actual <= planned + ACCEPTED_PLANNER_PROOF_SIZE_OVERCOUNT_BYTES` profile gate.
`level_proof_bytes` (`crates/akita-types/src/proof_size.rs:72-106`) drops `next_commit_bytes` for the t-revealed penultimate tier (and already drops `v_bytes`/stage-1 at `WithoutDBlock`).
Shipped schedule tables are regenerated under the new tail policy in S5; the drift guard `generated_schedule_tables_match_find_schedule` must pass after regen.

### Alternatives Considered

- **rANS / range coding against the discrete-Gaussian CDF.** Within ~0.01 bit of entropy vs Golomb-Rice's ~0.1-0.5 bit, but the decoder is a heavier state machine to make canonical, total, and cheap in-guest. Golomb-Rice captures nearly all the win at a fraction of the verifier/Jolt complexity; rANS is a future opt-in if the residual bits justify it.
- **Keep `PackedDigits`, only add per-segment widths.** Recovers the one-hot `e_hat` width waste but leaves the `z` worst-case-width slack on the table. The segment-typed framing subsumes it at no extra interface cost.
- **Transmit a small histogram/model.** Unnecessary: `sigma` is public/transcript-derivable, so a parametric model costs zero wire bytes.
- **Single fixed Rice `k` across all modes.** Simpler but loses the per-level `sigma` matching; the `sigma -> k` rule is one integer per segment and is descriptor-bound anyway.
- **Apply entropy coding at intermediate levels.** Impossible: intermediate witnesses are committed, not revealed; only the tail is cleartext.

## Documentation

- Update PR #141 branch `specs/terminal-direct-ring-relation.md` and `specs/terminal-fold-cutover.md` (PR #88) to cross-link this spec as the encoding layer on top of the r-drop and D-drop, and to record the B-role drop as the mirror of the D-role drop.
- Update PR #174 branch `specs/fold-linf-rejection.md` to note that the `sigma` envelope used for `t*` is reused as the entropy-coding model parameter (`sigma = isqrt_ceil(V)`).
- Profile example / bench reports: add per-segment tail breakdown fields (S5).
- **Extension point:** a future revealed JL projection image `p` at the tail is itself sub-Gaussian and should become another `Gaussian{k}` segment; coordinate with `specs/akita-jl-norm-check-resolutions.md` §8 when that path is specified.

## Execution

Recommended as a **stack of small PRs under this umbrella spec**, not one cross-cutting PR (rationale below).
Dependency order:

```text
S1  Land PR #141 (r-drop + terminal direct ring relations).        [spec-approved; not merged]
S2  B-role drop / reveal-t (+ full soundness paragraph).           depends on S1; S2 proof is merge gate
S3  Segment-typed tail framing (replace single PackedDigits with    depends on S1
    typed segments; fixed-width payloads first).                    byte-neutral refactor; transcript bind
S4  Golomb-Rice codec + sigma/k accessors + Gaussian/BoundedSmall   depends on S3
    models for z, e, and tiered u_concat.                           the measurable entropy win
S5  Planner regen + descriptor binding + proof-size accounting +    depends on S2,S3,S4
    Jolt-decode cost measurement + drift guard.
```

Each slice is independently shippable with its own measurable proof-size delta and its own narrow review surface.

Risks to resolve first:

- The S2 soundness paragraph (reveal `t`, drop B-rows, preserve weak binding) is the only non-mechanical item; write and review it before coding S2.
- The S4 codec must be canonical, total, and no-panic on the verifier path, and cheap to decode in-guest; pin canonicality and the `sigma -> k` rule against reference tests, and measure Jolt cycles.
- S3 must keep prover packing, verifier transcript slicing, and verifier row decoding byte-for-byte aligned across the segment boundaries (the same alignment risk PR #141 names).
- S5 must regenerate shipped tables under the same policy and keep the drift guard green.

## References

- PR #141 branch `specs/terminal-direct-ring-relation.md`: terminal r-drop and direct ring relations; S1 dependency.
- PR #174 branch `specs/fold-linf-rejection.md`: the `t*` threshold and `V = sigma_inf^2 * T_level * rho2` envelope reused for entropy coding.
- `specs/terminal-fold-cutover.md` (PR #88): the D-role drop whose argument the B-role drop mirrors.
- `specs/weak-binding-norm-fix.md`: the weak-binding object the tail extraction recovers.
- `crates/akita-types/src/proof/direct_witness.rs` (`PackedDigits`, `CleartextWitnessProof`), `crates/akita-types/src/proof/terminal_witness.rs` (`terminal_witness_segment_layout`, transcript slicing), `crates/akita-types/src/proof/levels.rs:704-731` (`TerminalLevelProof`), `crates/akita-types/src/proof/ring_relation.rs:226-267` (segment plane counts).
- `crates/akita-types/src/proof_size.rs:72-106` and `crates/akita-types/src/layout/proof_size.rs` (proof-byte and witness accounting).
- `crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs:351-374` (A/B/D row roles), `crates/akita-prover/src/protocol/ring_switch/coeffs.rs` (`build_w_coeffs` segment order), `crates/akita-prover/src/protocol/ring_switch/commit.rs` (`u = B * t_hat`).
- Profile: `AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 cargo run --release --example profile`.
