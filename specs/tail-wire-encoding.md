# Spec: Tail Wire Encoding (commitment elision + segment-typed entropy coding)

| Field       | Value                                                     |
|-------------|-----------------------------------------------------------|
| Author(s)   | Quang Dao                                                 |
| Created     | 2026-06-13                                                |
| Status      | proposed                                                  |
| PR          | umbrella spec; implementation lands as a stack (see Execution) |

## Summary

The Akita proof tail (the terminal fold level, the terminal-root 1-fold case, and the zero-fold fast path) is the only place a witness is sent in cleartext rather than as a commitment.
Today that cleartext witness is one `PackedDigits` blob (`crates/akita-types/src/proof/direct_witness.rs:106-163`): a single uniform `bits_per_elem` (the max balanced digit over the *entire* witness, capped at `log_basis <= 6`), with the folded response `z`, opening digits `e_hat`, inner-commitment digits `t_hat`, and quotient digits `r_hat` all concatenated and bit-packed at that one width.
This is fixed-width to a worst-case bound and pays the worst-case width on *every* coordinate of *every* segment, even though the segments have very different distributions: `z` is sub-Gaussian, one-hot `e_hat` is near-binary, `t_hat` is a full-entropy commitment value, and `r_hat` is not part of the terminal statement at all.

This spec defines a comprehensive tail wire encoding built on one principle: **encode each wire object according to its actual distribution and role.**
It composes four levers, three of which are commitment/quotient *elision* (do not send objects whose information the verifier can reconstruct or check directly) and one of which is *entropy coding* (send norm-bounded witness segments at their true entropy, not at a worst-case fixed width).
The verifier reconstructs every elided or de-decomposed quantity deterministically, so soundness is unchanged and the wire shrinks to the realized information content.

## Intent

### Goal

Replace the single fixed-width `PackedDigits` terminal witness with a segment-typed, entropy-coded cleartext tail that carries no commitments and no quotient, and whose per-segment models are derived from public/transcript-bound parameters with zero side information on the wire.

The feature introduces or modifies:

- A **segment-typed tail witness** representation replacing the single-width `PackedDigits` blob on the transparent tail: a sequence of typed segments (`Gaussian{sigma}`, `BoundedSmall`, `RawField`), each carrying only its payload bytes, with boundaries and models derived from the schedule/descriptor (headerless, like the rest of the wire format).
- A **canonical, total Golomb-Rice codec** (`akita-types`, verifier-reachable, no-panic) with zigzag sign mapping and a bounded-unary escape, parameterized by a transcript-derived integer Rice parameter `k`.
- A per-level **`sigma` accessor** `LevelParams::fold_response_sigma(...)` deriving the folded-witness standard-deviation proxy `sigma = ceil(sqrt(T_level * rho2)) * sigma_inf` from the same inputs PR #174 uses for `t*` (`specs/fold-linf-rejection.md:315`), and the deterministic `sigma -> k` rule.
- **`t`-reveal at the final committed tier** (the B-role mirror of the existing D-role drop): reveal the inner commitment value `t` (already on the wire as `t_hat`) in place of the parent commitment `u = B * t_hat`, dropping the `n_b` B-rows from that fold's relation and the `u` commitment from the wire.
- **`r`-elision and terminal-stage-2 elision** via the already-approved terminal direct ring-relation mode (PR #141, `specs/terminal-direct-ring-relation.md`), on which this spec depends and which it does not re-specify.
- **Descriptor binding** of the active tail-encoding policy (codec identity, per-segment model identities, the `sigma -> k` rule, the B-drop and r-drop flags) in `AkitaInstanceDescriptor`, and **proof-size/planner accounting** updated to the new tail shape.

### Invariants

1. **Lossless and soundness-neutral.** Every elided object (`u`, `v`, `r`) is reconstructed or checked directly by the verifier; every entropy-coded segment decodes bit-exactly to the same integer vector a fixed-width encoding would carry. The terminal relation `M_terminal * z == y_terminal` in `F[X]/(X^D+1)` and the digit-range / norm bounds the extractor relies on are unchanged. Protected by: existing terminal direct-relation row tests (PR #141), new round-trip codec tests, and an e2e tamper test that a witness violating the digit/norm bound cannot produce an accepting transcript.
2. **Canonical encoding.** Each integer vector has exactly one valid byte encoding under a fixed `(model, k)`; a non-canonical or malformed encoding is rejected with `AkitaError`/`SerializationError`, never decoded ambiguously. Protected by: a canonicality unit test (encode-decode-encode fixpoint) and a malformed-bytes rejection test.
3. **Total, bounded, no-panic decode.** The Golomb-Rice decoder terminates on every byte string (bounded unary via the escape), allocates only the schedule-declared element count, and never panics, unwraps, or indexes unchecked. Protected by: a fuzz/edge unit test over random byte strings and the verifier no-panic audit.
4. **Public models, zero side info.** Every per-segment model parameter (`sigma`, `k`, segment presence, segment boundaries) is derivable by the verifier from `LevelParams` + descriptor + transcript before the segment is decoded; no model, histogram, or width is transmitted. Protected by: a prover/verifier `sigma`/`k` agreement test (mirroring the `beta_linf_fold_bound` / `num_digits_fold` mirror invariant of PR #174) and a `LoggingTranscript` event-stream equality test.
5. **B-role drop preserves weak binding.** Revealing `t` and checking `t = A * e_hat` (A-rows) in place of committing `u = B * t_hat` (B-rows) preserves the weak-binding object the extractor recovers, by the same argument as the existing D-role drop (`specs/terminal-fold-cutover.md`). Protected by: the soundness sketch in Design (to be written in full before the B-drop slice) and the terminal-root / suffix-terminal row tests extended to the B-dropped layout.
6. **Descriptor binding distinguishes the policy.** A proof produced under one tail-encoding policy (codec, models, B-drop/r-drop flags) must not verify under another. Protected by: a pinned descriptor-bytes test and a cross-policy verify-fails test.
7. **Transparent-only.** Entropy coding and the t-reveal apply only to the transparent tail; under `feature = "zk"` the masked tail keeps the existing representation and the new policy rejects with `InvalidSetup`. Protected by: a zk-rejection regression test.

### Non-Goals

- **No intermediate-level change.** Intermediate folds commit their witness; their `u`/`v` commitments stay full field elements and their digits stay committed. This spec touches only the cleartext tail (terminal level, terminal-root, zero-fold).
- **No ZK tail encoding.** Masked/blinded witnesses are near-uniform and do not compress; a ZK direct/entropy tail needs a separate masked-relation design (same boundary PR #141 draws).
- **No change to the norm bound or the decomposition basis.** This spec changes how the realized witness is *encoded*, not the bound `K`/`t*` the verifier enforces (PR #174) nor the SIS rank pricing.
- **No new commitment scheme.** The t-reveal reuses the existing A-row check; it does not introduce a new matrix or assumption.
- **Not the r-drop itself.** Terminal `r_hat` and terminal stage-2 elision are PR #141; this spec depends on it and only adds the encoding and the B-drop on top.

## Evaluation

### Acceptance Criteria

- [ ] A `GolombRice` codec in `akita-types` is canonical (encode-decode-encode fixpoint), total (terminates and is no-panic on arbitrary bytes), and bijective on the integer range it admits, verified by unit tests including the escape path.
- [ ] `LevelParams::fold_response_sigma(num_claims, num_fold_points)` and the deterministic `sigma -> k` rule return integers pinned against a reference calculation, and prover/verifier read the identical value.
- [ ] The transparent terminal `final_witness` serializes as segment-typed payloads with no per-segment header bytes; `direct_witness_bytes` and `level_proof_bytes` report the exact serialized size and the e2e `proof.size() == plan.exact_proof_bytes` gate passes.
- [ ] The transparent terminal proof contains no `r_hat` and no terminal stage-2 bytes (inherited from PR #141), and no parent `u = B * t_hat` commitment for the t-revealed tier; the B-rows are absent from that fold's relation.
- [ ] A terminal proof that reveals `t` verifies, and tampering `t`, `e_hat`, or `z` is rejected; the A-row check `t = A * e_hat` is enforced.
- [ ] Descriptor bytes change intentionally and are pinned; a proof under the new policy fails to verify under the legacy `PackedDigits` policy and vice versa.
- [ ] Under `feature = "zk"`, selecting the entropy/t-reveal tail policy rejects with `InvalidSetup`; the masked tail is unchanged.
- [ ] Net proof-size reduction reported by the profile command at the affected modes, with the per-lever breakdown (r-drop, u-drop, z entropy, one-hot `e_hat` width recovery) recorded.
- [ ] In-guest decode cost measured: the Jolt-recursion verifier decode of the entropy-coded tail adds bounded, small cycles relative to the deleted stage-2/ring-switch work.

### Testing Strategy

Must keep passing: all `akita-types` proof/shape/serialization tests, the terminal direct-relation row tests (PR #141), the schedule drift guards, e2e batched/multipoint/recursive suites, transcript and zk suites.

New tests:

- `akita-types`: Golomb-Rice round-trip, canonicality, malformed/escape edge cases, no-panic over random bytes; `sigma`/`k` reference pins; segment layout decode boundaries; `TerminalLevelProofShape` round-trips the segment-typed tail and rejects cross-policy shapes.
- `akita-prover`/`akita-verifier`: prover/verifier `sigma`/`k` agreement; B-dropped terminal row check (reveal `t`, check `t = A * e_hat`, drop B-rows) for suffix-terminal and terminal-root; verifier no-panic on malformed tail bytes.
- e2e: tamper tests for `z`, `e_hat`, `t`, segment shape, and `y_rings`; descriptor cross-policy reject; ZK rejection.
- `LoggingTranscript`: event-stream equality across the new tail schedule.

Feature combinations: default, `--no-default-features`, `--features zk` (policy rejects), `--features logging-transcript`.

### Performance

Expected direction: smaller proofs, no material prover slowdown, bounded verifier decode.

Per-lever, at the affected tail modes (verified by `AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 cargo run --release --example profile` and the planner `exact_proof_bytes`):

- **r-drop + terminal stage-2 drop** (PR #141): measured `5.25%`-`6.15%` proof reduction on np1 profiles.
- **u-drop (t-reveal)**: removes one `n_b`-ring-element commitment (~1 KB on the cited profiles), for ~no added bytes since `t_hat` is already on the tail wire.
- **z entropy coding**: `z` coordinate cost drops from `~log2(2 * bound)` to `~log2(sigma) + 2.05` bits; ~`1.2` bits/coordinate vs the post-#174 `t*` bound, more vs `beta_inf`.
- **per-segment width recovery**: one-hot `e_hat` segments stop paying `z`/`t`'s global `bits_per_elem`; their near-binary digits collapse toward their own entropy.

A-role rank, setup size, and L2 pricing are unchanged.

## Design

### Architecture

The unifying classification: every tail wire object is in exactly one bucket, and each bucket has one correct encoding.

| object | nature | correct encoding | lever |
|---|---|---|---|
| `u = B * t_hat`, `v = D * e_hat` | full entropy mod q | do not send; reveal preimage at the tail | D-drop (done), B-drop (this spec) |
| `z` (folded response) | sub-Gaussian, public `sigma` | Golomb-Rice keyed by `sigma` | this spec |
| one-hot `e` | sparse / near-binary | bounded-small entropy code | this spec |
| `t = A * e_hat` | full entropy mod q | raw field elements (already minimal) | this spec (de-hat, byte-neutral) |
| `r` quotient | auxiliary, not in the statement | do not send; direct ring relations | PR #141 |

Affected surfaces:

- `akita-types`: the codec, the segment-typed `CleartextWitnessProof` variant and its shape, `direct_witness_bytes` / `proof_size.rs` tail accounting, the `sigma`/`k` accessors on `LevelParams`, and the descriptor policy fields.
- `akita-prover`: emit the tail as typed segments (recomposed integers for `z`/`e`, raw field for `t`), reveal `t` and skip the parent `u = B * t_hat`, emit no B-rows for that fold.
- `akita-verifier`: decode typed segments (no-panic), re-decompose where needed for the row check, check `t = A * e_hat` (A-rows), drop B-rows, no-panic shape validation.
- `akita-planner`/`akita-config`: codec-aware tail byte accounting, regenerate shipped tables under the new tail policy, bind the policy in the descriptor.

The current single-width path (`PackedDigits`, `direct_witness.rs:106-163`) is retained only for the ZK tail; the transparent tail routes through the segment-typed variant.

### The reveal-t / B-role drop

The relation row layout is `consistency | public | D(n_d) | B(n_b) | A(n_a)` (`crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs:351-374`), where `n_d = d_key.row_len()`, `n_b = b_key.row_len()`, `n_a = a_key.row_len()`.
The terminal already sets `n_d_active = 0` (`WithoutDBlock`) by revealing `e_hat` in cleartext: the D-role commitment `v = D * e_hat` and its rows are gone (`specs/terminal-fold-cutover.md`).

The B-role is the exact mirror.
The parent commitment is `u = B * t_hat` (`level_proof_bytes` `next_commit_bytes`, `crates/akita-types/src/proof_size.rs:95-99`; `crates/akita-prover/src/protocol/ring_switch/commit.rs` where `u.len() == b_key.row_len()`).
The inner commitment value `t = A * e_hat` (the A-rows) is already revealed at the tail as the `t_hat` segment of `final_witness`.
So at the final committed tier we reveal `t` (equivalently `t_hat`, the same bytes) and drop the parent `u = B * t_hat` commitment and the `n_b` B-rows, exactly as revealing `e_hat` dropped `v` and the D-rows.

Soundness sketch (to be written in full before the B-drop slice).
Weak binding at the tail rests on: (a) the revealed witness segments are bound in the transcript before any challenge derived from them is squeezed, (b) the verifier checks the reduced ring relation directly over `F[X]/(X^D+1)`, and (c) the A-row check `t = A * e_hat` ties the revealed `t` to the bound `e_hat`.
Dropping the B-commitment removes a binding of `t_hat` via `B`, but `t` is now revealed and bound directly, and `t = A * e_hat` is checked directly, so the object the extractor recovers (`e_hat`, hence the underlying opening) is unchanged.
This is structurally identical to the D-drop argument; the full version must state the extraction order (bind revealed segments, check A-rows, conclude) and confirm no remaining soundness term referenced the `B` commitment of `t_hat`.

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
Segment boundaries (element counts) and models are derived by both sides from `LevelParams` + descriptor: `z` has `num_fold_points * inner_width * D` coordinates, `e_hat` has `num_w_vectors * num_blocks * num_digits_open * D`, `t` has `n_a * D` field elements, in the existing `z_first`-dependent order (`crates/akita-prover/src/protocol/ring_switch/coeffs.rs`).
This mirrors the existing headerless, shape-driven decode (the shape supplies counts; the descriptor supplies models).

The verifier decodes each segment to its integer/field vector, then re-decomposes the `z` and `e` integers into balanced digits *on the verifier side* for the digit-range and row checks (the "stop sending hats" move: digits are a verifier-side deterministic function, not wire data).
`t` is used directly as field elements (its digits are uniform, so neither de-hatting nor entropy coding changes its size; `RawField` is the floor).

### Golomb-Rice for the Gaussian z segment

For a signed integer `n`, zigzag to a non-negative `u = (n << 1) ^ (n >> (W-1))` (`0, -1, 1, -2, 2 -> 0, 1, 2, 3, 4`).
Rice-code `u` with parameter `k`: quotient `q = u >> k` in unary (`q` ones then a zero), remainder `u & (2^k - 1)` in `k` bits.
Bounded-unary escape: if `q >= Q_MAX`, emit `Q_MAX` ones then a fixed `field_bits`-bit literal of `u`; this caps decode work and keeps the decoder total.

`k` is derived deterministically from the public `sigma`:

```text
V       = sigma_inf^2 * T_level * rho2          (PR #174, specs/fold-linf-rejection.md:292,315)
sigma   = isqrt_ceil(V)                          (integer, transcript-derivable)
k       = optimal_rice_k(sigma)                  (deterministic integer rule, pinned by test)
```

`optimal_rice_k` is the integer minimizing the expected Rice length for a discrete half-normal of scale `sigma` (the folded zigzag distribution), computed in integer/fixed-point arithmetic with a pinned reference; a robust closed form is `k = max(0, floor(log2(sigma)))` adjusted by a small tabulated correction, but the spec fixes whichever rule the reference test pins.
The codec is canonical because Rice coding is bijective for a fixed `k` and the escape branch is bijective on its range; the encoder must always choose the escape exactly when `q >= Q_MAX`.

One-hot `e` uses `BoundedSmall { k }` with a small `k` (often `0`, near-unary), which collapses the near-binary digits; dense `e` uses the same family with `k` from its own bound.

### Descriptor and wire binding

The tail-encoding policy is bound in `AkitaInstanceDescriptor` (the same mechanism PR #141 uses for `TerminalProofMode` and PR #174 for the threshold policy):

- codec identity (Golomb-Rice variant id, zigzag, `Q_MAX`),
- the `sigma -> k` rule identity and per-segment model assignment,
- the B-drop and r-drop flags.

The proof shape (`TerminalLevelProofShape`, `CleartextWitnessShape`) carries the segment list (counts + model tags) so headerless decode is unambiguous, exactly as `PackedDigits((num_elems, bits_per_elem))` does today.
Prover, verifier, schedule, and descriptor must agree; any disagreement rejects before hot arithmetic.

### Proof-size accounting

`direct_witness_bytes` (`crates/akita-types/src/layout/proof_size.rs:32-41`) gains a segment-typed arm returning the sum of per-segment encoded sizes; for the entropy segments the planner uses the expected-entropy estimate (`~num_coords * (log2(sigma) + 2.05)` bits for Gaussian), with the realized size gated by the e2e `proof.size() == plan.exact_proof_bytes` check.
`level_proof_bytes` (`crates/akita-types/src/proof_size.rs:72-106`) drops `next_commit_bytes` for the t-revealed tier (and already drops `v_bytes`/stage-1 at `WithoutDBlock`).
Shipped schedule tables are regenerated under the new tail policy; the drift guard `generated_schedule_tables_match_find_schedule` must pass after regen.

### Alternatives Considered

- **rANS / range coding against the discrete-Gaussian CDF.** Within ~0.01 bit of entropy vs Golomb-Rice's ~0.1-0.5 bit, but the decoder is a heavier state machine to make canonical, total, and cheap in-guest. Golomb-Rice captures nearly all the win at a fraction of the verifier/Jolt complexity; rANS is a future opt-in if the residual bits justify it.
- **Keep `PackedDigits`, only add per-segment widths.** Recovers the one-hot `e_hat` width waste but leaves the `z` worst-case-width slack on the table. The segment-typed framing subsumes it at no extra interface cost.
- **Transmit a small histogram/model.** Unnecessary: `sigma` is public/transcript-derivable, so a parametric model costs zero wire bytes.
- **Single fixed Rice `k` across all modes.** Simpler but loses the per-level `sigma` matching; the `sigma -> k` rule is one integer per segment and is descriptor-bound anyway.
- **Apply entropy coding at intermediate levels.** Impossible: intermediate witnesses are committed, not revealed; only the tail is cleartext.

## Documentation

- Update `specs/terminal-direct-ring-relation.md` and `specs/terminal-fold-cutover.md` to cross-link this spec as the encoding layer on top of the r-drop and D-drop, and to record the B-role drop as the mirror of the D-role drop.
- Update `specs/fold-linf-rejection.md` to note that the `sigma` it computes for `t*` is reused as the entropy-coding model parameter.
- Public security-model docs (`book/src/how/security/*`): note that the tail witness is entropy-coded losslessly and that the bound enforced is unchanged.
- Profile example docs: add the per-segment tail breakdown fields.

## Execution

Recommended as a **stack of small PRs under this umbrella spec**, not one cross-cutting PR (rationale below).
Dependency order:

```text
S1  Land PR #141 (r-drop + terminal direct ring relations).        [already approved; re-land per its own notes]
S2  B-role drop / reveal-t (+ full soundness paragraph).           depends on S1
S3  Segment-typed tail framing (replace single PackedDigits with    depends on S1
    typed segments; RawField for t, fixed-width for the rest).      byte-neutral refactor
S4  Golomb-Rice codec + sigma/k accessors + Gaussian/BoundedSmall   depends on S3
    models for z and one-hot e.                                     the measurable entropy win
S5  Planner regen + descriptor binding + proof-size accounting +    depends on S2,S3,S4
    Jolt-decode cost measurement.
```

Each slice is independently shippable with its own measurable proof-size delta and its own narrow review surface.

Risks to resolve first:

- The S2 soundness paragraph (reveal `t`, drop B-rows, preserve weak binding) is the only non-mechanical item; write and review it before coding S2.
- The S4 codec must be canonical, total, and no-panic on the verifier path, and cheap to decode in-guest; pin canonicality and the `sigma -> k` rule against reference tests, and measure Jolt cycles.
- S3 must keep prover packing, verifier transcript slicing, and verifier row decoding byte-for-byte aligned across the segment boundaries (the same alignment risk PR #141 names).
- S5 must regenerate shipped tables under the same policy and keep the drift guard green.

## References

- `specs/terminal-direct-ring-relation.md` (PR #141): terminal r-drop and direct ring relations; the dependency for S1.
- `specs/fold-linf-rejection.md` (PR #174): the `t*` threshold and `rho2`/`sigma` derivation reused as the entropy model.
- `specs/terminal-fold-cutover.md`: the D-role drop whose argument the B-role drop mirrors.
- `specs/weak-binding-norm-fix.md`: the weak-binding object the tail extraction recovers.
- `crates/akita-types/src/proof/direct_witness.rs` (`PackedDigits`, `CleartextWitnessProof`), `crates/akita-types/src/proof/terminal_witness.rs` (segment layout), `crates/akita-types/src/proof/levels.rs:276-301` (`TerminalLevelProof`).
- `crates/akita-types/src/proof_size.rs:72-106` and `crates/akita-types/src/layout/proof_size.rs` (proof-byte and witness accounting).
- `crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs:351-374` (A/B/D row roles), `crates/akita-prover/src/protocol/ring_switch/commit.rs` (`u = B * t_hat`).
- Profile: `AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 cargo run --release --example profile`.
