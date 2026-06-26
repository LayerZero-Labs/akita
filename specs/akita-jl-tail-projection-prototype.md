# Spec: Unstructured JL Projection Prototype (tail reveal, standalone)

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao, Cursor agent (model: Claude Opus 4.8) |
| Created     | 2026-06-15                     |
| Status      | **prototype landed** on branch `quang/jl-projection-tail` ([PR #191](https://github.com/LayerZero-Labs/akita/pull/191)): PR1 projection, PR1b MLE eval, PR2 consistency sumcheck (split across `akita-types::jl`, `akita-prover::protocol::jl`, `akita-verifier::protocol::jl`). Fusion roadmap D1–D8 and security accounting remain open. |
| PR          | [#191](https://github.com/LayerZero-Labs/akita/pull/191) |
| Related     | Full cutover protocol design: [`akita-jl-projection-protocol.md`](akita-jl-projection-protocol.md) (Slot 2/3, stage-2 fusion, SIS repricing; not implemented here). |

## Summary

Akita is a lattice-based recursive polynomial commitment scheme.
Each recursive level commits a short witness, folds it against a challenge, and proves enough shortness for the weak-binding extractor to recover a norm-bounded opening.
Today that certificate is a stage-1 infinity-norm range-check sumcheck on the decomposed fold response, and the recursion ends by sending the final folded witness in cleartext (the terminal direct step).
Recent fixed-width planner snapshots for the fp32 one-hot families show the terminal cleartext witness is a large part of the proof at the shipped folding sizes, and each intermediate level also pays for its stage-1 range tree.
Those byte numbers are calibration, not a security premise, and they should be re-measured after the active tail-encoding work lands.

This spec defines a minimal, self-contained prototype of an alternative tail-level shortness mechanism: an unstructured (dense, field-granular) Johnson-Lindenstrauss random projection.
The verifier samples a dense binary-sign projection matrix from the Fiat-Shamir transcript; the prover projects the witness to an integer image `p`, reveals `p`, and the verifier checks a Euclidean norm bound on `p` over the integers.
A sumcheck checks projection consistency against the witness multilinear extension: JL rows are batched with `eq` weights on `row_bits = log2(n_rows)` challenges (not Vandermonde `rho^j`), the public weight is the joint binary-sign MLE `J_tilde(r_J, r_w)` partially evaluated in `r_J`, and the standalone prototype exercises that degree-2 product relation before fusion.
The wired protocol would replace the stage-1 infinity-norm range sumcheck at selected JL levels with an image-norm check plus a projection-consistency row.
This prototype does not make that replacement in the recursive flow.

The prototype is built as standalone, well-tested library code that is not wired into the recursive prove/verify flow.
The goal is to land reusable JL primitives, exercise the consistency-sumcheck mechanics across representative fields and ring dimensions, and document the fusion roadmap so later protocol-integration work can measure any proof-size or rank impact on the real flow.

### Implementation status (PR #191, 2026-06-22)

| Slice | Crate / path | Landed |
|-------|----------------|--------|
| PR1 — matrix sample, integer projection, norm helpers | `akita-challenges::jl` | yes |
| PR1 — fast kernels (column-panel `i8`/`i32`, runtime SIMD) | `akita-challenges::jl::kernels` | yes |
| PR1b — `build_jl_row_weights`, `eval_jl_mle_at` | `akita-challenges::jl::mle` | yes (production MLE path is **LUT-amortized** per packed byte; see **Joint MLE evaluation**) |
| PR2 — layout, wire validation, transcript replay, image claim | `akita-types::jl` | yes |
| PR2 — consistency prove harness | `akita-prover::protocol::jl` | yes |
| PR2 — consistency verify harness | `akita-verifier::protocol::jl` | yes |
| Transcript labels | `akita-transcript::labels` (`ABSORB_JL_PROJECTION`, `CHALLENGE_JL_SEED`, `ABSORB_JL_IMAGE`, `CHALLENGE_JL_ROW`) | yes |
| Benches | `benches/jl_projection.rs`, `benches/jl_mle.rs` | yes |
| Fusion D1–D8 | — | not started |

Tests (local): 28 in `akita-challenges` (`jl` filter), 4 in `akita-types` (`jl`), 2 in `akita-prover` (`jl`), 6 in `akita-verifier` (`jl`). Shared consistency harness helpers live in `akita-types::jl::fixtures` (`jl-test-fixtures` feature, dev-deps only). Nothing in `prove_batched` / `verify_batched` calls the new modules.

### Revision target: binary signs instead of ternary entries

The landed PR #191 mechanics use the LaBRADOR-style ternary alphabet `{-1,0,+1}` and two-bit packing. The next cutover target is to replace that alphabet everywhere with independent Rademacher signs `{-1,+1}`.

The reason is both theoretical and practical:

- **Theory precedent.** Achlioptas 2003, "Database-friendly random projections: Johnson-Lindenstrauss with binary coins", Theorem 1.1 explicitly proves Johnson-Lindenstrauss embeddings for independent `+1/-1` entries, scaled by `1/sqrt(k)`. So binary signs are a standard JL distribution, not a heuristic variant of ternary JL.
- **Crypto implementation precedent.** Greyhound §6 states that its implementation deviates from LaBRADOR by sampling JL matrices with coefficients `±1` rather than `-1,0,1`, and §6.2 describes the resulting fast Four-Russians projection using the 16 signed sums of four coefficients.
- **Akita implementation win.** A binary-sign row needs one bit per coordinate instead of two. Projection expansion and matrix memory traffic halve; the MLE packed-byte alphabet drops from ternary's collapsed 81-pattern four-column table to a direct 256-pattern eight-column table; and the projection hot loop becomes signed add/sub without a zero branch or zero-lane table.

This revision is a **full cutover**, not a compatibility mode. Once implemented, `JlProjectionMatrix` means binary-sign JL. There should be no ternary/binary enum, no old packing decoder, and no proof/wire format that accepts both alphabets. PR #191 remains the mechanical prototype; the security writeup and implementation follow-up should target binary signs only.

## Motivation: why JL projection (and what it does *not* do)

The proof is dominated by the cleartext terminal witness.
For `onehot_fp128_d64` at `nv=32` the tail is 80,032 B of a 112,016 B proof (71%); the eight fold levels together are only ~32 KB (PR #189 bench artifact).
It is natural to hope JL shrinks that tail by lifting the low-basis requirement of the inf-norm range check.
After working it through, that specific hope is **not** substantiated by the bytes evidence below; this section records why, so future work targets JL's real wins (committed-level rank, protocol simplicity, prover-time/recursion-depth) rather than the tail.

**The stage-1 inf-norm range check grows linearly in `lb`, not exponentially.**
Each fold level pays a stage-1 range-check sumcheck that certifies every digit lies in `[-b/2, b/2)`.
It is **not** a single degree-`b` product: the `s = w(w+1)` trick halves the degree-`b` range product, and `stage1_tree_stage_arities` then GKR-decomposes the result into `~(lb-1)/2` product stages of constant arity (2 or 4) plus a quartic leaf (`crates/akita-types/src/proof/stage1.rs:103-151`).
So the transcript grows **linearly in `lb`** (one stage per ~2 binary levels); the interstage-claim count only reaches `~b/8` asymptotically and is negligible at the operating `lb <= 5` (bench `stage1_interstage_claims_bytes = 0-32`).
(An earlier draft of this section read the `b <= 8` base case at `stage1.rs:130` (degree `b/2`) and wrongly generalized it to a `2^lb` tax — that is corrected here.)

**What JL changes per fold is therefore modest, and a tail-byte win is not established.**
JL replaces the range check with a global Euclidean-norm certificate (reveal `p = J w`, check `||p||_2`), but that image plus its consistency row has its own per-level cost, and the range check it removes is only linear in `lb`.
So JL is roughly **per-fold cost-neutral**: there is no exponential tax to remove.
The hope that "cheaper folds shift the fold-vs-cleartext crossover and contract the terminal" is **not established** and must not be claimed without a JL-priced DP run; with the range check only linear, the effect is likely small.

**The basis itself is byte-neutral; do not claim otherwise.**
It is tempting to say "JL unlocks a larger basis, which shrinks the tail." That is *not* the mechanism, and the main-worktree specs are right to push back on the bytes claim.
Under fixed-width packing the cleartext bytes are magnitude-locked (`delta * lb ~ magnitude_bits`, `akita-jl-norm-check-resolutions.md` 3.1), and under entropy coding (the live `tail-wire-encoding.md` plan) the basis is byte-*irrelevant* because the wire carries entropy, not digit-planes, and re-decomposes verifier-side (`akita-jl-norm-check-resolutions.md:250`).
Re-basing a *fixed* terminal witness does not change its bytes. The only thing that would shrink it is contracting to a *smaller* terminal (fewer ring elements / lower entropy), which requires more fold levels to be worthwhile — and per the per-fold analysis above, JL does not clearly make them worthwhile.

**What the basis is genuinely good for: recursion depth and prover time.**
A larger basis means fewer digit planes per level, hence fewer sumcheck variables per level and fewer recursion levels.
That directly reduces prover time and the verifier-circuit recursion depth (fewer cycles for the Jolt-embedded verifier), which is a real win on objectives *other* than proof bytes.
The per-term-basis freedom belongs here: because JL certifies only the global L2 norm and not per-digit ranges, there is no shared range-check structure forcing one common basis, so each term (`e_hat`, `t_hat`, `z_hat`, `r_hat`) can pick its own basis to minimize the next-level witness length and drop a variable.
This shrinks variables/depth, not entropy-coded bytes.

**SIS rank scaling (why a high basis is affordable when wanted).**
Raising `lb` raises the per-digit L-infinity collision (`2^lb - 1` for B/D roles, `~b/2` for dense A) and the L2 collision bound, which raises the A-matrix module rank `n_a`.
But the generated SIS floor table makes `n_a ~ (ln(W * beta^2) / c)^2`: quadratic in the *log* of (width x squared-collision).
For fixed `(family, d, rank)` the supported width is `~ 1/collision^2`, and each added module rank multiplies the supported width by `~370x` (Q32) to `~10^4-10^5x` (Q64/Q128) (`crates/akita-types/src/sis/generated_sis_table/`), so `n_a` grows only `~0.2` ranks per `+1` of `lb`.
Rank growth is far slower than the basis rise, so the depth/prover-time basis lever is not throttled by SIS; the remaining ceiling on the basis is the i8-packing datatype (`MAX_I8_LOG_BASIS = 6`), which JL does not lift on its own (widening to i16+ and re-deriving the CRT-NTT safe widths in `crates/akita-prover/src/kernels/linear/capacity.rs` is the knock-on engineering cost).

**The separate committed-level SIS win.**
Independently of the tail, JL tightens the A-role weak-binding price by replacing the deterministic envelope `beta_inf` with a realized norm (the `30-200x` slack, about 1-2 module ranks; see D4).
This shrinks committed-level commitment bytes and is basis-independent.

**Status of the prior DP "basis is byte-neutral" retraction (it stands).**
The DP measurement in `akita-jl-norm-check-resolutions.md` 3.2 (cap lifted 6 -> 16, byte-flat at `nv >= 28`) is essentially correct for proof size.
A previous version of this section objected that the DP "did not price a JL-cheapened fold (it priced a `2^lb` range tree)"; with the range check now understood to be only **linear** in `lb`, that objection is weak — replacing a linear range check with a JL image of comparable cost does not materially move the per-fold price, so the DP would not see a large fold-count shift either.
Combined with the `delta * lb ~ magnitude_bits` packing identity and the entropy-coding argument (the wire carries entropy, not digit-planes), the basis is **not** a proof-size lever, and removing the range check via JL does not obviously make it one.
A JL-priced DP re-run (with per-term bases) is still the right experiment to settle the magnitude, but the expected direction is "small", not the large contraction an earlier draft claimed.

**Open question for the tail.**
The original motivation for JL was to shrink the 71% tail by lifting the basis cap.
On the bytes evidence above (packing + entropy + linear range check) I cannot substantiate a tail-byte win from JL, and the honest position is that JL's proof-size value is the committed-level rank below, not the tail.
If there is a tail mechanism beyond fold economics (e.g. a structural change to what the terminal encodes), it is not captured here and should be added explicitly rather than asserted.

## Background and rationale

This section makes the spec self-contained. It states the protocol facts the prototype depends on and the design choice it implements.

### The recursive shape and the shortness obligation

One recursive level, in the order objects are bound to the transcript:

```
u_l        commits the current witness w_l (a short, decomposed object)
v          a derived commitment row (folds the opening witness e_hat and other roles)
c          fold challenge, sampled after v
z          fold response z = sum_i c_i * s_i, where s_i are the decomposed
           blocks (columns) of w_l
u'         commits the next witness w_next (which embeds z and the other roles)
... stage-2 sumcheck ties everything to the opening claim ...
```

Soundness needs the extractor to recover a witness whose Euclidean norm is bounded.
The current protocol certifies the fold response `z` with a range check; the JL direction certifies the realized norm of the committed next witness `w_next` statistically (the flat table includes `z_hat` as a segment), replacing the loose deterministic envelope on `z`.
The operator-norm weak-binding price `Gamma_bar` in `eta_A = 2 * Gamma_bar * beta_bar_2` is unchanged, as in LaBRADOR (see D4).
Today that certificate is **stage 1**: the balanced base-`2^lb` digits of `z` are committed and a range-check sumcheck of degree `2^lb` proves each digit lies in the digit set, giving an infinity-norm bound `||z||_inf <= beta_inf`, converted to Euclidean via `||z||_2 <= sqrt(d) * beta_inf`.
The deterministic envelope `beta_inf = num_claims * 2^r_vars * min(||c||_inf*||s||_1, ||c||_1*||s||_inf)` is what the SIS accounting prices today (see the A-role collision below).
Prior calibration notes report that this deterministic envelope is often 30 to 200 times larger than the realized `||z||_2`.
That motivates replacing the envelope, but calibration alone is not a soundness argument.

### The Johnson-Lindenstrauss alternative

A random projection can certify Euclidean shortness directly once its lower-tail failure probability and transcript-grinding budget are accounted for.
Fix a vector `s` before sampling a random matrix `J` with entries in `{-1,+1}` and `n_rows` rows.
With high probability over `J`, a vector fixed before `J` is sampled cannot have a large Euclidean norm while its image has a small Euclidean norm.
The real-valued concentration is standard binary JL (Achlioptas 2003). The modular statement needed by Akita must be re-derived from the LaBRADOR/GHL proof strategy with binary signs, because the published LaBRADOR Lemma 4.2 is stated for ternary entries with `Pr[0]=1/2`.
The intended wired verifier therefore samples `J` from the transcript (so it is independent of the pinned object), receives the image `p = J s`, and checks `||p||_2` over the integers.

The integer image check avoids proving an exact squared-sum identity modulo `q`, which is the small-field obstacle for exact `l2` certificates.
It does **not** remove every modular issue: the consistency sumcheck is still a field identity, so an accepted image coordinate must have an injective signed representation in the base field.
The wired verifier must enforce a coordinate encoding or bound such as `|p_j| < q/2` for every accepted coordinate, otherwise a prover could exploit congruent integer representatives with different Euclidean norms.
This is a protocol condition, not just an implementation detail.

### Why dense and field-granular is correct at the tail (the fold-commutation law)

A projection placed inside the recursion must commute with the fold to be checkable through the fold, i.e. `J * coeff(sum_i c_i s_i) = sum_i Rot(c_i) * J * coeff(s_i)` must hold.
This forces a structured, ring-granular `J = J_0 (x) I_D` (entries that are constant polynomials) whenever the projected image is committed and then folded again.

At a **tail reveal** level the image itself is not committed as a recursive witness segment and then folded through a later level.
The consistency is checked directly against the projected witness, and the image leaves the recursion as revealed data.
For that reveal variant the commutation constraint does not force a structured matrix.
The matrix can be a plain dense field-granular `{-1,+1}` matrix, provided the protocol is only using it in this direct-consistency form.
This is the "unstructured" projection this spec prototypes, and it is the simplest variant: dense matrix, reveal the image, check the norm over `Z`, prove consistency with one sumcheck.
(The structured ring-granular committed-image variant for mid levels is explicitly out of scope.)

### Binary modular-JL lemma target

The statement Akita should prove and cite internally is the following binary analogue of LaBRADOR Lemma 4.2.

**Target lemma (single-shot binary modular JL, Akita form).** Let `q` be an odd modulus, `J ∈ {-1,+1}^{m×d}` have independent Rademacher entries, and let `w ∈ [-q/2,q/2]^d` be fixed before `J` is sampled. For concrete constants `a_m`, `B_m`, and `Q_m`, if `||w||_2 >= b` and `b <= q / Q_m`, then

```text
Pr_J[ ||Jw mod q||_2 < sqrt(a_m) * b ] <= 2^-lambda.
```

For the PR #191 default, the intended row count is `m = 256` and `lambda = 128`. The proof task is to choose the largest usable `a_256` (for tight slack) and the smallest safe `Q_256` (for small-field headroom), while keeping the upper-tail completeness threshold `B_256` explicit:

```text
Pr_J[ ||Jw mod q||_2 > sqrt(B_256) * ||w||_2 ] <= eps_hi
```

The verifier then checks `||p||_2 <= T_p`. If the accepted image is consistent and `T_p = sqrt(a_m) * T_w`, soundness gives `||w||_2 <= T_w` except with the above failure probability. Completeness uses a larger honest threshold, or nonce regrind, sized from `B_m`.

**Do not reuse LaBRADOR constants blindly.** Ternary LaBRADOR uses `E[C^2]=1/2`, so one projection row has variance `||w||_2^2/2` and `E||Jw||_2^2 = 128||w||_2^2` at `m=256`. Binary signs have variance `||w||_2^2` per row and `E||Jw||_2^2 = 256||w||_2^2`. A binary version can be normalized either by:

```text
J_bin_raw ∈ {-1,+1}^{m×d};        E||J_bin_raw w||_2^2 = m ||w||_2^2
```

or by comparing against the ternary scale with `J_bin_scaled = J_bin_raw / sqrt(2)`. Implementation should keep raw integer signs and adjust thresholds/constants, not introduce irrational scaling into the protocol.

### Binary proof strategy

The proof should adapt LaBRADOR Appendix A, with the real binary tail from Achlioptas/GHL as the concentration input. The structure is:

1. **Real lower tail, no modulus.** Prove for fixed nonzero `w` that `Pr[||Jw||_2 < sqrt(a) ||w||_2]` is at most `2^-lambda` for binary `J`. The fastest conservative proof is Paley-Zygmund/small-ball per row plus Chernoff over `m` rows; the tight proof should use Achlioptas-style moment comparison or a numerical dynamic program over the extremal Rademacher sum to maximize `Pr[|<pi,w>| < t||w||]`. This replaces LaBRADOR Lemma 4.1.
2. **Case 1: small norm, wrap is rare.** If `||w||_2 < q/10`, then a small modular image implies either the real image is small or some row wrapped close to a nonzero multiple of `q`. Bound the first event by step 1. Bound the wrap event by the binary upper tail `Pr[|<pi,w>| > c q]`, unioned over `m` rows. Binary signs are subgaussian with parameter `||w||_2`, so this part is at least as clean as the ternary proof.
3. **Case 2: one huge coordinate.** If `||w||_∞ >= q/C_inf`, fix all row signs except the largest coordinate. For binary signs, toggling that one sign gives two values separated by `2|w_i|`; at most one can lie in a window of radius below `|w_i|`. Thus a single row has constant probability of escaping the small window, and Chernoff over `m` rows gives the `2^-lambda` lower-tail bound. This is the ternary proof's "large coordinate" case without the zero-sign complication.
4. **Case 3: spread-out large norm.** If `||w||_2 >= q/10` but `||w||_∞` is small, truncate to a subvector `v` with norm in a fixed interval below `q/10` and disjoint residual `w-v`. Apply Berry-Esseen to `<pi,v> = sum_i epsilon_i v_i`, where each summand has variance `v_i^2` and third moment `|v_i|^3`. The Berry-Esseen error is proportional to `||v||_∞ / ||v||_2`, and is smaller than the ternary case after matching thresholds because binary signs have twice the row variance. Condition on the residual `<pi,w-v> mod q`; the bad set is an interval of length `2 sqrt(a) b`, and anti-concentration of the approximating normal bounds one-row failure away from one. Raise to `m`.

The constants should be produced by a small checked script, committed under `scripts/` or `crates/akita-challenges/benches/` only if it is part of the reproducible security accounting. Inputs:

- `m` row count, initially 256.
- target `lambda`, initially 128.
- candidate lower threshold `a`.
- modular precondition constant `Q`.
- large-coordinate cutoff `C_inf`.
- Berry-Esseen constant, using the explicit `0.56` or better published constant chosen by the writeup.

Outputs:

- `a_m`: lower-tail threshold used for soundness.
- `B_m`: upper-tail threshold / honest acceptance bucket used for completeness and regrind.
- `Q_m`: modulus precondition `b <= q/Q_m`.
- individual failure terms for the three cases, so the final lemma is auditable rather than a black-box simulation.

The first pass may be conservative; the final pass should optimize `a_m` and `Q_m` jointly. Greyhound's binary `±1` implementation is precedent for the distribution, not a proof of these constants; Grand Danois inherits the ternary LaBRADOR constants and should not be cited for binary constants.

### The three corrections that frame the value

Prior internal analysis flagged three things that constrain how this prototype is framed:

1. **JL at the tail does not replace the terminal cleartext send.** The terminal direct opening is the PCS base case: something must still verify the final evaluation claim. The reveal projection may delete stage 1 at a tail level and may shrink the last one or two committed levels (replace a full-witness reveal with a small image plus birth-certification of `w_next`), but it does not eliminate the terminal witness. Frame this as "delete stage 1 plus reveal a small image," not "shrink the terminal 3-4x."
2. **JL is not a decomposition-basis byte win.** Lifting the decomposition basis was hypothesized to shrink the tail under JL; a planner sweep showed total proof size changes by at most 1 percent at small sizes and 0 percent at `num_vars >= 28`, because the digit-packing identity `delta * lb ~ field_bits` magnitude-locks every cleartext segment while the module rank `n_a(lb)` grows. So the candidate byte case for JL rests on deleting stage 1, possible tail-payload changes, and any proven SIS-rank repricing, not on a basis lever.
3. **JL does not drop the operator-norm factor from the A-role; it supplies a realized `beta_bar_2`.** The weak-binding collision shape stays `eta_A = 2 * Gamma_bar * beta_bar_2` (LaBRADOR Thm 5.1; akita `lem:batched-weak-binding`). `Gamma_bar` is structural (cross-multiplied fold challenges) and is **not** removed. JL replaces the **input** `beta_bar_2`: today that term is priced through the loose deterministic envelope `beta_inf` (`||z||_2 <= sqrt(d) * beta_inf`, often 30-200x over realized). The reveal prototype projects the flat next witness `w_next = (e_hat, t_hat, z_hat, r_hat)`; `z_hat` is a segment of that table, so `||z||_2 <= ||w_next||_2`, and a JL norm certificate on `w_next` (via `p = J w_next`, `||p||_2 <= T_p`, modular-JL slack) bounds `beta_bar_2` directly. D4 is finishing that constant map and writing it into `norm_bound.rs`; there is no separate "anchored extraction" fork.

### Reusable prior code

The retired `labrador-backup` branch contains a working dense reveal projection at `labrador-backup:src/protocol/labrador/johnson_lindenstrauss.rs`: a 256-row ternary matrix packed 2 bits per entry, deterministic per-row expansion from a 32-byte transcript seed via SHAKE, an integer projection path that centers base-field coefficients and rejects overflow, a nonce-regrind (retry-until-the-image-is-short) completeness loop, and a `collapse` (dot with a coefficient vector) helper.
Only the transcript hosting, centering, overflow discipline, nonce-regrind shape, and image/norm API ideas are reusable. The matrix alphabet, packing, decode tables, MLE LUTs, and constants should be replaced by the binary-sign design.
The transcript hosting, row count, coordinate encoding, overflow policy, and security constants must be reworked for current Akita rather than ported as-is.

## Intent

### Goal

Land a standalone, field-generic and ring-dimension-generic JL projection prototype that samples a dense binary-sign projection matrix from a transcript seed, projects a witness to an integer image, checks the image Euclidean norm over the integers, and exercises projection-consistency with one degree-2 sumcheck in the intended fused-row oracle form, without wiring it into the recursive prove/verify flow.

Concretely the prototype introduces:

- `akita-challenges::jl` (new module): the dense projection matrix `JlProjectionMatrix`, deterministic transcript-seeded expansion (`sample`), integer projection (`project`) over centered witness coefficients, signed-coordinate encoding checks, and Euclidean-norm helpers. The projection acts on the flat integer coefficient vector of base-field elements, so the public API takes a flat `&[F]` coefficient slice (`F: FieldCore + CanonicalField`) and the caller flattens any ring layout. Ring structure (`const D`) is irrelevant to the projection and reappears only in the consistency sumcheck (the `akita-prover` module below), so it is not a parameter of this crate; this keeps `akita-challenges` at its field + transcript dependency layer.
- `akita-challenges::jl::mle` (new submodule, PR1b): optimized joint-matrix MLE evaluation `eval_jl_mle_at` and the prover-side row-weight builder `build_jl_row_weights`. This is the verifier-critical bottleneck and reuses the packed binary-sign row format plus Dao-Thaler split-eq contraction (see **Joint MLE evaluation** below). Adds a dependency on `akita-algebra` for `SplitEqEvals` only in this submodule.
- `akita-types::jl` (new module, PR2): shared consistency shapes — `JlWitnessLayout`, verifier-wire image embedding/norm checks, transcript absorb/sample helpers, and the batched image input claim. Optional `jl::fixtures` (`jl-test-fixtures` feature) holds dev-only sumcheck harness helpers shared by prover/verifier tests. Mirrors the `trace_weight` split: layout and wire contracts live in `akita-types`, not in `akita-challenges`.
- `akita-prover::protocol::jl` (new module, PR2 prove side, not called by the flow): degree-2 product sumcheck prover (`prove_jl_consistency`, `JlConsistencyProver`). Builds padded witness/row-weight tables from `build_jl_row_weights`.
- `akita-verifier::protocol::jl` (new module, PR2 verify side, not called by the flow): degree-2 product sumcheck verifier (`verify_jl_consistency`, `JlConsistencyVerifier`). Standalone tests use a `w_eval_hook` instead of a commitment opening; full cryptographic binding is deferred to fusion.
- Cross-field, cross-dimension tests that exercise representative non-degenerate `(field, ring dim)` combinations the workspace ships.

### Invariants

- **Determinism / replayability.** For a fixed transcript state and fixed `(n_rows, cols)`, `JlProjectionMatrix::sample` is a pure function of the transcript-derived seed; prover and verifier reconstruct the identical matrix. Protected by a determinism test (two independent transcripts in the same state produce equal matrices and equal projections).
- **Projection correctness.** `project(w)` equals the exact integer matrix-vector product `J * centered_coeffs(w)` with no modular reduction; the image lives over the integers (balanced representatives of `w`). Protected by a test against a naive reference projection and by the consistency sumcheck harness.
- **Coordinate injectivity.** Any coordinate that is later embedded into the field for consistency must be checked to lie in the chosen signed encoding window, for example `|p_j| < q/2`. This prevents modulo-`q` aliases from passing the field consistency check with a different integer norm. Protected by boundary tests around the signed encoding limit.
- **No overflow on the integer path.** Centering uses balanced representatives in `[-(q-1)/2, (q-1)/2]`. The production fast path centers to `i32` digits and enforces the balanced-digit bound `|d| <= MAX_JL_DIGIT` (`= 32`, i.e. `lb <= 6`) at the boundary (`validate_digit_witness`), so every row sum `sum_i +-d_i` fits `i32` for any supported column count (`cols <= i32::MAX / MAX_JL_DIGIT`); accumulation is then unchecked `i32` on the hot path (including in the SIMD kernels, whose per-lane partials are bounded by the same argument). A non-digit input whose centered magnitude exceeds the digit bound (e.g. a full-magnitude fp128 element) is rejected at the boundary rather than wrapped or saturated, which is correct: it is not a JL witness. The checked-`i64` reference projection (`project_digits_reference`, test/bench only) is the correctness oracle and is the one place wider accumulation guards against overflow. The squared-norm reduction accumulates in `u128` (`l2_norm_sq_checked`); it is `O(n_rows)` and off the hot projection path. Protected by small-digit, oversized-rejection, digit-bound, fast-vs-`i64`-reference differential, and norm-bound tests.
- **Norm check is over the integers.** Shortness is accepted from `||p||_2^2 <= T_p^2` over the integers, never from a squared-sum identity modulo `q`. This avoids the exact-`l2` no-wrap gate, but it still relies on the coordinate-injectivity check above. Protected by a completeness test (honest witness passes under a generous prototype bound) and a soundness-direction test (an over-norm image is rejected).
- **Consistency claim has the intended fused-row form.** Rows are batched with `eq` weights on `row_bits = log2(n_rows)` variables (default `n_rows = 256`, `row_bits = 8`), matching the relation-sumcheck batching idiom rather than a Vandermonde `rho^j` fold. The public witness weight is `g(i) = sum_j eq(r_J, j) J[j, i]` on coefficient corners; equivalently `g_tilde(r_w) = J_tilde(r_J, r_w)` for the joint MLE `J_tilde` of the binary-sign matrix on the product hypercube `{0,1}^{row_bits} x {0,1}^{k_w}`. The proved identity is `sum_j eq(r_J, j) p[j] = sum_{x,y} g(x,y) w(x,y)`, a degree-2 product sumcheck `w_tilde * g_tilde` with input claim `sum_j eq(r_J, j) embed(p[j])`. This is the intended stage-2 fusion shape, but final drop-in compatibility still has to be checked against the current `w(x, y)` layout and stage-2 verifier API. Protected by a prove/verify round-trip test and a Schwartz-Zippel soundness-direction test (a wrong image fails except with the usual `r_J` / sumcheck collision probability).
- **No-panic on verifier-reachable paths.** Every shape mismatch (matrix dimensions, image length, point dimension, coefficient overflow) returns `AkitaError`, never panics, matching the verifier no-panic contract because the consistency check is intended to become verifier-reachable. Protected by malformed-input tests.
- **No protocol-flow regression.** Nothing in `prove_batched` / `verify_batched` calls the new modules, so all existing prover/verifier/integration tests pass unchanged and no serialized proof type changes. Protected by the full existing test suite.
- **Joint MLE evaluation correctness.** `eval_jl_mle_at(J, r_J, r_w)` equals the naive double sum `sum_{j,i} eq(r_J,j) eq(r_w,i) J[j,i]` for every packed matrix and challenge point; `build_jl_row_weights(J, r_J)[i]` equals `sum_j eq(r_J,j) J[j,i]`. The `#[doc(hidden)]` bench hooks `eval_jl_mle_at_from_eq_tables` / `eval_jl_mle_at_scalar_from_eq_tables` validate `e_j` / `e_w` lengths against the padded hypercube before slicing (verifier-reachable once fused). Protected by differential tests against a reference implementation, cross-checks that `eval_jl_mle_at` matches `eval_mle_from_weights(build_jl_row_weights(...), r_w)`, and an image-claim vs row-weight-dot-witness identity test in `jl::mle`.

### Non-Goals

These are the deferred items; each is investigated in the "Deferred work" section so the follow-up is fully scoped.

- **No wiring into the recursive flow.** No new `Step` variant, no `Schedule`/planner change, no serialized proof-type change, no `prove_batched` / `verify_batched` change.
- **No structured / ring-granular committed-image (mid-level) projection.** Only the dense reveal projection is built.
- **No weak-binding repricing write-up and no SIS repricing.** The realized-norm A-role price (operator factor retained, D4) is a security-accounting task, not needed to prototype mechanics.
- **No ZK masking of the revealed image.** The revealed image leaks `n_rows` linear functionals; the prototype targets non-ZK builds.
- **No finalized binary modular-JL constants.** `n_rows` (default 256) and the norm bound are configurable parameters until D4/D6 produce `a_m`, `B_m`, and `Q_m`.
- **No removal of the terminal cleartext base case.**

## Evaluation

### Acceptance Criteria

- [x] `akita-challenges::jl` compiles and exposes `JlProjectionMatrix` with transcript-seeded `sample`, a flat `project(&[F])`, signed-coordinate validation, and checked norm helpers, generic over `F: FieldCore + CanonicalField` (no `const D` in this crate).
- [x] `akita-challenges::jl::mle` exposes `eval_jl_mle_at` (fused verifier path) and `build_jl_row_weights` (prover path); fast kernels match reference on fp32/fp64/fp128 extension fields used by shipped configs; tail-geometry bench documents throughput at `n_rows = 256` and `cols` in the shipped tail range.
- [x] Determinism test: two transcripts in identical state yield byte-identical matrices and equal projections.
- [x] Projection-vs-reference test: `project` matches a naive integer reference for random witnesses across fields and dims.
- [x] fp128 digit test: small balanced digits project correctly over an fp128 base field; a non-digit, full-magnitude fp128 coefficient (centered value past `i64`) is rejected without panic.
- [x] Signed-coordinate tests: accepted coordinates embed injectively into the base field, and boundary aliases are rejected.
- [x] `akita-prover::protocol::jl` proves JL consistency for honest `(w, p)` with `w_eval_hook`-free prover tables.
- [x] `akita-verifier::protocol::jl` verifies JL consistency round-trips for honest `(w, p)` across fp32/fp64/fp128 base fields, using public test witness data or an explicit `w_tilde(r)` evaluation hook.
- [x] Soundness-direction tests: an image inconsistent with `w` is rejected by the consistency sumcheck for all but a negligible fraction of `r_J`; an over-norm image is rejected by the norm check.
- [x] Malformed-input tests: wrong matrix shape, wrong image length, wrong point dimension all return `AkitaError`, never panic.
- [x] All pre-existing workspace tests pass unchanged.
- [x] `cargo fmt -q`, `cargo clippy --all -- -D warnings`, and the relevant test passes are green.

### Testing Strategy

New tests live alongside the new modules.

- `akita-challenges`: unit tests for `sample` determinism, one-bit packed-matrix round-trip (`0 -> -1`, `1 -> +1`), `project` correctness vs reference, signed-coordinate injectivity, checked integer norm computation, fp128 small-digit projection, and oversized non-digit rejection. Port the analogous API tests from `labrador-backup:src/protocol/labrador/johnson_lindenstrauss.rs`, but do not inherit its ternary packing, fixed row count, or dual `i64`/`i128` width split.
- `akita-challenges::jl::mle`: differential tests of `eval_jl_mle_at` and `build_jl_row_weights` against a naive `Θ(n_rows · cols)` reference; identity `eval_jl_mle_at(J,r_J,r_w) == eval_mle_from_weights(build_jl_row_weights(J,r_J), r_w)`; image-claim vs row-weight-dot-witness identity; short `eq` table rejection; SIMD vs scalar kernel parity; malformed shape errors return `AkitaError`.
- `akita-types::jl`: layout pinning (`JlWitnessLayout`), wire validation (`embed_jl_image_coords`), layout/MLE geometry checks (`validate_layout_for_matrix_mle`). `jl::fixtures` (feature `jl-test-fixtures`) supplies shared witness-table helpers for downstream consistency tests.
- `akita-prover`: prove-side rejection tests (image norm bound, malformed layout).
- `akita-verifier`: round-trip, tampered-image, and malformed-layout tests across fp32/fp64/fp128 base fields.

Feature combinations: tests pass with and without `parallel` if both feature sets compile in the touched crates. ZK is out of scope (the reveal leaks), so any reveal-path test that assumes public image data is gated to non-`zk` builds or clearly marked as a non-ZK mechanics test.

Existing tests that must keep passing: the entire `akita-pcs` / `akita-prover` / `akita-verifier` suite, unchanged, since the flow is untouched.

### Performance

Standalone prototype: no proof-size or prover-time effect on shipped paths (nothing is wired).

The measurement this enables (in the fusion follow-up) is whether replacing one tail level's stage-1 work with a JL reveal (delete stage 1, send `n_rows` signed integer coordinates plus the consistency row) reduces total bytes under the then-current tail encoding.
Prior analysis suggests the basis-unlock argument is not a byte win; any byte claim should therefore be tied to deleting stage-1 work, changing the terminal payload, or repricing SIS ranks after the extraction lemma is settled.
Cost shape (updated after optimization investigation):

| Workload | Who | Target | Asymptotic |
|----------|-----|--------|------------|
| Integer projection `p = J c` | Prover | PR1 (landed) | `Θ(n_rows · cols)` i32 add/sub |
| Row weights `g[i] = Σ_j eq(r_J,j) J[j,i]` | Prover (sumcheck oracle table) | PR1b `build_jl_row_weights` | `Θ(n_rows · cols)`; byte-column bit sums reuse `Σ_j eq(r_J,j)` |
| Point eval `J̃(r_J,r_w)` | Verifier (fusion final check) | PR1b `eval_jl_mle_at` | `Θ(n_rows · cols)`; **LUT-amortized** one-shot eval over byte quads |
| Consistency sumcheck | Prover + verifier | PR2 | `k_w` witness rounds; verifier JL work is dominated by `eval_jl_mle_at` |

There is no closed-form shortcut for a dense random `J` (unlike `TraceWeight`). Constant-factor wins come from split-eq nesting, byte-wide binary-sign decode, deferred extension-field reduction, column-panel / outer-tile parallelism, and SIMD. Binary signs remove ternary zeros, halve matrix bytes, and let two tiny 16-entry nibble LUTs cover one eight-column packed byte. At tail scale (`n_rows = 256`, `cols ~ 2^{17}`) the verifier eval is intentionally `Θ(n_rows · cols)` but must be engineered to the same standard as PR1 projection kernels.

## Design

### Architecture

The prototype splits along the existing crate dependency graph.

**`akita-challenges::jl` (matrix sampling, projection, norm).**
This crate owns Fiat-Shamir challenge sampling and depends only on `akita-field` and `akita-transcript`, the right layer for the projection primitives (field-level math, transcript-seeded, no sumcheck).
It already exposes a SHAKE256-backed streaming `XofCursor` (with bias-free draws and a `next_sign` helper) and the transcript seed-derivation pattern used by the sparse-challenge sampler, which the JL module reuses.

- `JlProjectionMatrix { n_rows, cols, row_bytes, packed_rows: Vec<u8> }`: a dense binary-sign matrix with entries `{-1,+1}` packed 1 bit per entry (`0 -> -1`, `1 -> +1`) in a single contiguous row-major buffer, drawn from the transcript-derived XOF stream.
- `sample<F, T>(transcript, n_rows, cols) -> Result<Self, AkitaError>`: absorbs a context buffer (label, `n_rows`, `cols`, and a version/domain tag), draws a 32-byte seed, and expands rows deterministically. Per-row derivation, as in the backup, keeps generation parallel-safe; a single streaming `XofCursor` is also acceptable if tests lock the exact transcript behavior and no parallel row generation is required. `n_rows` and `cols` are parameters (default test `n_rows = 256`), not hardcoded.
- `project<F>(&self, coeffs: &[F]) -> Result<JlImage, AkitaError>` (`F: FieldCore + CanonicalField`): centers each coefficient to its balanced `i32` representative (`center_coefficients`) and projects through the fast kernel to `i32` image coordinates. JL is only ever applied to small balanced digits (`|d| <= MAX_JL_DIGIT = 32`), so `i32` holds both the centered digits and every row sum with room to spare; there is no `i64`/`i128` on the hot path. The modulus is recovered for `CanonicalField` as `(-F::one()).to_canonical_u128() + 1`; if a clearer modulus helper is added later, use it instead. A non-digit input whose centered magnitude exceeds the digit bound (e.g. a full-magnitude fp128 element) is rejected at the boundary rather than wrapped or saturated. Returns an error on a digit-bound violation or a `coeffs.len() != cols` shape mismatch. `project_digits(&[i32])` is the pre-centered entry point; `project_digits_reference` is a checked-`i64` oracle for tests/benches.
- `JlImage` (or an equivalent explicit type) stores signed integer coordinates, exposes checked embedding into `F` only when the configured signed window is injective, and exposes checked norm helpers such as `l2_norm_sq_checked` / `check_l2`.

**`akita-challenges::jl::mle` (joint matrix MLE, PR1b).**
Depends on `akita-algebra` for `SplitEqEvals`. Shares packed-row decode (`BINARY_SIGNS_FOR_BYTE` or direct bit masks) and SIMD dispatch conventions with `project`.

- `eval_jl_mle_at<L>(matrix, r_J, r_w) -> Result<L, AkitaError>`: fused verifier contraction (see **Joint MLE evaluation**).
- `build_jl_row_weights<L>(matrix, r_J) -> Result<Vec<L>, AkitaError>`: prover row-weight table `g`.
- Scalar reference + LUT production path (`mle/lut.rs`); row-weight builder uses row-eq accumulation in `mle/common.rs`. Runtime SIMD dispatch on projection kernels only (MLE is LUT + scalar/parallel panels).
- For binary row weights, process byte-aligned column panels by bit sums rather than row scatter:

```text
total = Σ_j eq(r_J,j)
ones_l = Σ_{j : bit_l(J[j, byte]) = 1} eq(r_J,j)
g[byte*8 + l] = 2 * ones_l - total
```

This is algebraically the same as `Σ_j eq(r_J,j) sign(bit_l)` with `0 -> -1`, `1 -> +1`, but reduces field additions and memory writes for the materialized prover table.

The matrix sampling and projection geometry (`n_rows`, seed domain) are intended to be bound into the instance descriptor when fusion lands; the prototype binds them through the transcript context buffer only and records descriptor binding as a fusion task.

**`akita-types::jl` (consistency layout and verifier-wire shapes, PR2).**
Depends on `akita-algebra` (eq tables for the image claim) and `akita-transcript` (absorb/sample labels). Mirrors `trace_weight`: shared contracts consumed by both prover and verifier.

- `JlWitnessLayout`: flat witness hypercube `w[x * 2^ring_bits + y]` with power-of-two padding.
- `embed_jl_image_coords`, `jl_image_claim`, `absorb_jl_image`, `sample_jl_row_point`, `padded_live_table`, `validate_layout_for_matrix_mle`.

**`akita-prover::protocol::jl` (consistency prove, PR2).**
Depends on `akita-challenges`, `akita-sumcheck`, `akita-types::jl`, and `akita-transcript`. Not referenced by `flow.rs` or any prove entry point.

- `prove_jl_consistency`, `JlConsistencyProver`: degree-2 product sumcheck prover over padded witness and row-weight tables.

**`akita-verifier::protocol::jl` (consistency verify, PR2).**
Depends on `akita-challenges`, `akita-sumcheck`, `akita-types::jl`, and `akita-transcript`. Exported as `verify_jl_consistency` from `akita-verifier`. Not referenced by `verify_batched`.

- `verify_jl_consistency`, `JlConsistencyVerifier`: replays transcript absorb/sample, checks the image claim, and verifies the sumcheck with `eval_jl_mle_at` at the final point.

See **Projection-consistency sumcheck** below for the default relation, batching, and verifier evaluation model.

- `build_jl_row_weights(matrix, r_J) -> Vec<L>` (PR1b, in `akita-challenges::jl::mle`): for each witness coefficient index `i`, compute `g[i] = sum_j eq(r_J, j) J[j, i]`.
- `eval_jl_mle_at(matrix, r_J, r_w) -> L` (PR1b): fused verifier path; equals `sum_{j,i} eq(r_J,j) eq(r_w,i) J[j,i]` without materializing the full `g` table.
- `prove_jl_consistency` / `verify_jl_consistency`: absorb the image, sample `r_J` from the transcript after the witness is bound (wire-before-squeeze), check coordinate injectivity and the integer norm bound, then run the sumcheck. The verifier obtains `w_tilde(r_w)` from the sumcheck output (standalone tests supply a witness hook).

**Why the consistency is in fused-row form now.**
The current stage-2 fused sumcheck batches subclaims by powers of `gamma`: `gamma^0 * relation + gamma^1 * range + gamma^2 * trace`, all sharing one witness scan over `w(x, y)`.
At JL levels the range term is deleted and replaced by a JL row `gamma^k * w(x,y) * JlWeight(x,y)` with public table `JlWeight = g`.
The witness MLE `w_tilde(r_w)` at the final point is shared across relation, trace, and JL; JL-specific verifier work is `J_tilde(r_J, r_w)`.
The exact `gamma` power `k`, the `w(x, y)` column layout, and the final-point evaluation path remain D2 work.

### Projection-consistency sumcheck

This subsection is the canonical design for PR2 (`akita-types::jl`, `akita-prover::protocol::jl`, `akita-verifier::protocol::jl`) and for stage-2 fusion (D2).

#### Objects and indexing

- `J ∈ {-1,+1}^{n_rows × cols}`: dense binary-sign matrix from `JlProjectionMatrix::sample` (packed one bit per entry).
- `n_rows` is a power of two in production (`256` default, `row_bits = 8`). If not, pad with zero rows and zero image coordinates.
- Witness coefficients `c_i` (centered digits) index corners of `{0,1}^{k_w}` with `k_w = col_bits + ring_bits`, matching stage-2 table `w(x, y)` under the same flatten map `i = index(x, y)` used by the prover layout.
- Row index `j ∈ {0,1}^{row_bits}` (little-endian, same convention as `EqPolynomial` elsewhere).
- Integer image `p_j = sum_i J_{j,i} c_i` (exact `ℤ`; revealed for the tail prototype).

#### Joint binary-sign MLE of `J`

Define the truth function on the product hypercube:

```text
J_hat(j, i) = J[j, i]  in {-1, +1}
```

Its multilinear extension over challenge points `(r_J, r_w)` with `r_J ∈ L^{row_bits}` and `r_w ∈ L^{k_w}` is:

```text
J_tilde(r_J, r_w)
  = sum_{j, i} eq(r_J, j) * eq(r_w, i) * J[j, i]
```

`J_hat` is **dense** on the hypercube (random `J`) and binary at every corner: values lie in `{-1,+1}`, and the implementation uses the same one-bit packed-row format as integer projection.

Partial evaluation in `r_J` yields the public witness weight table:

```text
g[i] = sum_j eq(r_J, j) * J[j, i]
g_tilde(r_w) = J_tilde(r_J, r_w)
```

The second equality is the key identity: the verifier target is one joint MLE evaluation, not an ad hoc "fold rows then MLE" pipeline.

#### Row batching with `eq` (not Vandermonde)

Sample `row_bits` extension-field challenges `r_J ∈ L^{row_bits}` from the transcript **after** the projected witness is bound and **after** the image `p` is absorbed (wire-before-squeeze). Batch rows with:

```text
eq(r_J, j) = prod_t ( j_t * r_{J,t} + (1 - j_t) * (1 - r_{J,t}) )
```

The batched consistency relation on integer corners is:

```text
sum_j eq(r_J, j) * p_j  =  sum_i g[i] * c_i  =  sum_{x,y} g(x,y) * w(x,y)
```

Field-side input claim (after injective embed of each `p_j`):

```text
claim_JL = sum_j eq(r_J, j) * embed(p_j)
```

Soundness: if `p ≠ p'`, then `sum_j eq(r_J, j) (p_j - p'_j)` is a nonzero multilinear polynomial in `r_J`; a random `r_J` falsifies it with Schwartz–Zippel error `O(row_bits / |L|)` (comparable to a degree-`(n_rows-1)` Vandermonde batch in one scalar `rho`).

`r_J` is **not** a sumcheck variable; it is a public batching point fixed before the round, like opening-point `eq` weights in the relation term.

#### Standalone degree-2 sumcheck

Sumcheck runs over **witness** variables only (`k_w` rounds). Per boolean corner:

```text
oracle(x, y) = w(x, y) * JlWeight(x, y)    where JlWeight(x,y) = g[index(x,y)]
```

```text
input_claim = claim_JL
```

At the final point `r_w = (r_x, r_y)` from the sumcheck:

```text
w_tilde(r_w) * g_tilde(r_w)  =  claim_JL
```

The verifier cannot close this from `g` and `p` alone: it needs `w_tilde(r_w)` at the sumcheck final point (from the shared sumcheck output in fusion; from public witness data or an explicit evaluation hook in the standalone prototype).

The prover builds `JlWeight` once from `J` and `r_J` via `build_jl_row_weights` (`Θ(n_rows · cols)` binary-sign work, shared kernels with projection and MLE). The witness scan reuses the stage-2 pattern (public weight table times witness MLE).

#### Fusion with stage-2

Fused per-corner oracle (JL level, stage 1 deleted):

```text
gamma^0 * w * alpha * m_tau1(x)
+ gamma^k * w * JlWeight(x,y)          [JL consistency; k TBD in D2]
+ gamma^2 * w * TraceWeight(x,y)       [trace; unchanged convention]
```

Fused input claim adds `gamma^k * claim_JL` alongside relation and trace contributions.

At the final point, the verifier already has `w_tilde(r_w)` from the shared sumcheck. JL-specific work:

1. Reconstruct `J` from the transcript seed (same as prover).
2. Evaluate `g_tilde(r_w) = J_tilde(r_J, r_w)` with `eval_jl_mle_at` (fused split-eq contraction; **do not** materialize the full `g` truth table on the verifier).
3. Check `gamma^k * w_tilde(r_w) * g_tilde(r_w)` against the batched input claim.

There is no separate witness opening for JL: `w_tilde(r_w)` is the same value used by relation and trace.

#### Verifier cost note

Dense random `J` has no `TraceWeight`-style closed form. Every matrix entry must be read once (`Θ(n_rows · cols)` memory traffic). Split-eq and tiling improve constant factors and parallelism but not the exponent. The verifier uses the **fused** point evaluator `eval_jl_mle_at` rather than `build_jl_row_weights` followed by a separate `eq(r_w, ·)` pass, avoiding a `2^{k_w}`-sized weight buffer. At tail scale (`n_rows = 256`, `cols ~ 2^{17}`) this is intentional; PR1b targets verifier-grade kernel engineering on that path.

#### API sketch (PR2)

| Symbol | Role |
|--------|------|
| `row_bits` | `log2(n_rows)` |
| `r_J` | `L^{row_bits}` row batching point |
| `build_jl_row_weights(J, r_J)` | prover row-weight table `g` (PR1b) |
| `eval_jl_mle_at(J, r_J, r_w)` | verifier fused `J̃(r_J,r_w)` (PR1b) |
| `JlWeight` | public table `g(x,y)` for prover scan |
| `claim_JL` | `sum_j eq(r_J,j) embed(p_j)` |

### Joint MLE evaluation (optimized kernels, PR1b)

Standalone slice between PR1 (projection) and PR2 (consistency sumcheck). Implements the verifier bottleneck and the prover's row-weight table builder. Canonical reference: Dao–Thaler split-eq (ePrint 2024/1210), already used as `SplitEqEvals` in `akita-algebra/src/eq_poly.rs` and nested contraction in `akita-types/src/extension_opening_reduction.rs` (`tensor_column_partials_split_fold`).

#### Two workloads

| API | Consumer | Output | Materialize `g` on `2^{k_w}` corners? |
|-----|----------|--------|--------------------------------------|
| `eval_jl_mle_at(J, r_J, r_w)` | Verifier (fusion final check) | one field element `J̃(r_J,r_w)` | **No** |
| `build_jl_row_weights(J, r_J)` | Prover (sumcheck oracle `w · JlWeight`) | vector `g` of length `cols = 2^{k_w}` | **Yes** |

Both have the same asymptotic cost (`Θ(n_rows · cols)` matrix touches) but different memory behavior. The verifier path is the performance-critical one and must not allocate the weight hypercube.

#### Production algorithm: LUT-amortized byte eval (binary target)

The shipped PR #191 `eval_jl_mle_at` path does **not** use split-eq nesting; it matches the projection kernels' byte geometry. Under the binary-sign cutover the same idea remains, but each packed byte uses two 16-entry nibble LUTs rather than the old 81-pattern ternary table over four signs:

1. Precompute `eq(r_J, ·)` and `eq(r_w, ·)` once (`EqPolynomial::evals`).
2. Validate table lengths against the padded row/column hypercube (`validate_eq_tables`); bench hooks return `AkitaError` on short buffers instead of silent truncation.
3. Partition witness columns into byte-aligned 8-column windows.
4. For each packed byte: build two 16-entry sign-weight LUTs from the eight `eq(r_w, ·)` weights.
5. Scan every matrix row: two LUT lookups plus one field add per packed byte into per-row accumulators `row_acc[j]`.
6. Finish with `Σ_j eq(r_J,j) · row_acc[j]`.

On aarch64 fp128 at tail-scale column counts the existing ternary LUT path is ~3× faster than a row-major scalar baseline and beats deferred-reduction wide variants tried during PR1b (`benches/jl_mle.rs`, `scalar` vs `lut`). Binary should improve the same path through half the matrix bytes and much smaller LUT construction.

`build_jl_row_weights` uses direct row-eq accumulation over packed matrix bytes (`mle/common.rs`), not the LUT (prover materializes the full `g` vector once).

#### Alternative considered: tensor split-eq

A split-eq nested contraction (Dao–Thaler; same template as `tensor_column_partials_split_fold`) was specced as the initial PR1b target. It remains a valid fallback if LUT amortization regresses on a target arch, but it is **not** the landed production path. Sketch:

```text
J̃(r_J, r_w) = Σ_{j_o,j_i,i_o,i_i} e_J_out[j_o] · e_J_in[j_i] · e_w_out[i_o] · e_w_in[i_i] · J[j,i]
```

with `W[j_i, i_i] = e_J_in[j_i] · e_w_in[i_i]` precomputed and outer tiles parallel over `(j_o, i_o)`.

#### Inner hot loop: binary sign × eq weight

Matrix entries are uniform random bits (`0 -> -1`, `1 -> +1`), so every entry contributes exactly one signed add/sub.

Per matrix entry the update is `acc ± W`, not a general field multiply by `J`.

**Byte-wide processing (primary micro-kernel):** replace PR1's ternary `SIGNS_FOR_BYTE` table with a binary sign table (`256 × 8` `i8`, 2 KiB, L1-resident) or with direct bit masks. For each packed row byte covering eight consecutive columns, load eight eq weights and eight signs; accumulate into a deferred extension-field accumulator (`MulBaseUnreduced` / `ProductAccum`, same contract as `partials_out_contribution`). For the MLE path, use two 16-entry nibble LUTs per packed byte; direct 256-entry byte-table construction was considered but table setup can dominate.

**SIMD:** mirror PR1 layout (NEON `vmlal_s16` over sign×weight products; x86 `madd_epi16` / AVX-512 widening). Here the "digits" are extension-field eq weights (or their lifted `i16` limb patterns for base fields), not witness `i8` digits. Runtime arch dispatch like PR1. Binary signs also enable XOR/sign-mask variants for integer projection; benchmark against widened multiply-add before replacing the simple sign-table path.

**Deferred reduction:** inner face sums many `±W` into `ProductAccum`; reduce once per inner face; multiply by the outer `e_J_out · e_w_out` and add to the global accumulator.

#### Optimization investigation (what helps, what does not)

**Helps (spec adopts):**

1. **Fused verifier eval** — saves a `2^{k_w}` field-element write and a second pass; largest win for verifier memory bandwidth.
2. **LUT-amortized byte scan** — per 8-column packed byte, two 16-entry sign-weight LUTs reused across all rows.
3. **Binary `SIGNS_FOR_BYTE` or direct bit masks + byte-wide scan** — amortizes sign decode over eight columns.
4. **Column panels / outer-tile parallelism** — witness read once in projection; MLE parallel path fans over quad windows when `parallel` is enabled.
5. **Branchless binary sign LUT** — `sign · W` with `sign ∈ {-1,+1}`.

**Does not help enough (spec rejects as primary strategies):**

1. **Zero-skip optimizations inherited from ternary JL.** Binary JL has no zero entries, so there is nothing to skip. The hot loop should focus on compact one-bit decode and signed add/sub.
2. **Four-Russians / large pattern tables over `k` binary columns.** Classic 4R amortizes `2^k` partial-sum tables across many queries on the same block. The verifier runs **one** `eval_jl_mle_at` per proof; table build dominates once `k` grows past a byte. Stage-1/stage-2 prefix LUTs (`STAGE1_B4_PREFIX_LOOKUP_TABLE`, etc.) work because the table is reused across millions of sumcheck rounds. Keep the per-proof JL table at byte size.
3. **Sparse / run-length `J` storage.** Would sacrifice the uniform dense XOF expansion and complicate matrix generation for uncertain gain on random data.
4. **Materializing full `g` on the verifier** then `Σ_i eq(r_w,i) g[i]` — correct but strictly worse than `eval_jl_mle_at` for memory.

**Benchmark later (optional, not spec blockers):** sign-mask add/sub kernels on AVX2/AVX-512; `vdotq_s32` on AArch64 when toolchain stabilizes; tuning split bit-widths (`4+4` vs `3+5` on rows) for cache.

#### Crate placement and dependencies

- Submodule `akita-challenges::jl::mle` owns `eval_jl_mle_at`, `build_jl_row_weights`, scalar reference, SIMD kernels, and benches (`benches/jl_mle.rs`).
- Adds `akita-algebra` dependency for `SplitEqEvals` / `EqPolynomial` (acceptable: only this submodule needs sumcheck-adjacent eq tables; keeps PR2 thin).
- Extension field `L` is generic (`L: MulBaseUnreduced<F> + …`); tests sweep fp32/fp64/fp128 extension types used in stage 2.

#### Flattening convention

Bit order for `(j, i) →` flat column index must match `build_w_coeffs` / stage-2 `w(x, y)` layout before PR2 builds `JlWeight`. Pin in tests with a small layout fixture; document the chosen little-endian convention in module docs.

**Field and ring-dimension genericity.**
The projection acts on integer coefficients of base-field ring elements, so it should be field-generic across base fields that implement `CanonicalField`.
Centering uses the canonical representative in `[0, q)` and the balanced interval around zero.
The matrix and projection are generic over `const D`.
The consistency sumcheck operates over the claim/extension field `L` for the witness multilinear extension in the same style as stage 2, but the prototype must explicitly bridge base-field signed image coordinates into `L`.
Tests sweep representative fp32/fp64/fp128 and supported `D` values, excluding degenerate configs that do not exercise a recursive witness.

### Alternatives Considered

- **Vandermonde `rho^j` row batching.** A single challenge `rho` with weights `rho^j` is equivalent in intent but mismatched to the relation-sumcheck `eq` batching idiom and to the joint `(r_J, r_w)` MLE view. Rejected in favor of `row_bits` challenges and `eq(r_J, j)` weights.
- **Verifier builds full `g` table then evaluates `eq(r_w, ·)`.** Algebraically identical to `eval_jl_mle_at` but allocates and writes `2^{k_w}` field elements on the verifier. Rejected in favor of fused split-eq point evaluation.
- **Four-Russians tables over large `k` column blocks.** No cross-query amortization on the verifier's one-shot eval; per-block setup with eq-dependent weights dominates. Rejected beyond nibble/byte LUTs; binary `SIGNS_FOR_BYTE` or direct bit masks are the right static decode size.
- **Branching on sign value.** Binary signs are dense, so branches only add misprediction risk. Rejected; use a branchless sign LUT or bit-mask add/sub.
- **Sparse / compressed `J` on disk.** Breaks uniform XOF expansion; marginal on random data. Rejected for prototype.
- **Standalone separate sumcheck (not fused-row).** Simpler to write but throwaway; it would not match the fusion target and would be rewritten. Rejected in favor of building the fused-row weight now.
- **Hand-rolled textbook sumcheck instead of the `akita-sumcheck` driver.** Duplicates the engine and is less drop-in for fusion. Rejected; reuse the existing driver via a small instance type.
- **Ring-granular structured projection (committed image, mid level).** Required when the projected image is committed as recursive witness data and folded through a later level (the commutation law). The reveal prototype checks the projection directly and does not fold the image, so dense matrix mechanics are the right standalone target. Out of scope; covered in Deferred work.
- **Replace the terminal cleartext with JL.** The PCS base case must still verify the evaluation claim; cannot be fully replaced. Not attempted.
- **Host everything in one crate.** Crate layering forbids a sumcheck in `akita-challenges`; splitting sampling (low) from consistency (prover layer) respects the graph.

## Deferred work (fusion roadmap)

Each deferred item below is investigated to the point where the follow-up can start without re-discovery. Suggested ordering is given at the end.

### D1. Instance-descriptor binding of the projection geometry

**What.** The transcript preamble (`AkitaInstanceDescriptor`, in `akita-types/src/instance_descriptor.rs`) binds the algebra, setup identity, effective plan, and per-call shape so a proof under one configuration cannot verify under another.
Fusion must bind the JL geometry: `n_rows`, the seed domain separator, the signed-coordinate encoding window, the per-level norm bound `T_p`, the variant flag (reveal vs committed), and which levels are JL levels.
The natural home is an extension of `PlanSection` (the effective per-level schedule), since the JL choice is per-level and schedule-driven, or a dedicated `JlSection`.

**Why deferred.** The prototype binds geometry through a transcript context buffer in `sample` only, which is enough for standalone determinism but not for cross-proof domain separation.

**Open questions.** Whether to fold the geometry into the existing per-level `LevelParams`/`PlanSection` digest or add a sibling section; how to keep the descriptor round-trippable and panic-free on deserialization (verifier no-panic contract); confirming a JL-level proof cannot replay as a non-JL-level proof or under a different signed-coordinate window.

### D2. The `Step` variant, proof payload, and stage-2 fusion

**What.** Three coordinated changes:

1. `akita-types/src/schedule.rs`: add a third `Step` variant (today `Fold(FoldStep)` and `Direct(DirectStep)`), e.g. `JlFold(JlFoldStep)`, carrying `n_rows`, `T_p`, and reusing `LevelParams`. The planner emits it for tail levels; `Schedule::fold_steps` and the num-levels helpers must account for it.
2. `akita-types/src/proof/levels.rs`: a JL level proof payload carrying the revealed image `p` (a `Vec` of `n_rows` signed integer coordinates, or an explicitly injective field encoding with the same signed window) in place of `stage1` and either `next_w_commitment` (intermediate JL) or the cleartext witness. The consistency proof rides in stage 2.
3. `akita-prover/src/protocol/sumcheck/akita_stage2`: add the `omega_JL` addend `gamma^k * w(x,y) * JlWeight(x,y)` to the fused batch and delete the `gamma^1 * range` term at JL levels. The verifier obtains `g_tilde(r_w)` via `eval_jl_mle_at(J, r_J, r_w)` at the stage-2 final point. `r_J` (`row_bits` challenges) is sampled after `u'` binds `w_next` and after `p` is absorbed (wire-before-squeeze); `gamma` already exists.

**Why deferred.** This is the actual protocol cutover and touches serialized types, the planner, and both prover and verifier; the prototype is explicitly standalone.

**Open questions.** The exact `gamma` power for the JL row; image serialization (signed-integer codec vs injective field encoding, which also feeds the tail entropy-coding work); whether intermediate JL levels (birth-certify `w_next`, then delete the next level's stage 1) and terminal-adjacent JL levels need distinct payloads; where the verifier obtains `w_tilde(r)` for the JL row in each case.

### D3. Stage-1 deletion and SIS repricing

**What.** Deleting stage 1 removes the degree-`2^lb` range tree, the carried `s_claim`, and the stage-1 transcript at JL levels.
It also reprices the SIS roles in `akita-types/src/sis/norm_bound.rs`:

- A-role (`rounded_up_collision_norm_s`) today is `committed_fold_collision_l2_sq` with `collision_linf = 8 * omega * beta_inf * nu`. Under a JL realized-norm certificate this keeps the operator-norm factor and becomes `2 * Gamma_bar * beta_bar_2` with `beta_bar_2` the JL-certified realized response (not the deterministic `sqrt(d)*beta_inf`); see D4. The rank lever is the 30-200x realized-vs-envelope tightening of `beta_inf`, not removal of the operator norm.
- B-role (`rounded_up_collision_norm_t`, the `t_hat` opening digits) and D-role (`rounded_up_collision_norm_w`) are digit-range collisions `2^lb - 1`; they are unchanged by JL but must be re-audited once stage 1 no longer certifies "digits are range-checked" (a comment in `norm_bound.rs` and several consumers assume this).
- The revealed image norm bound `T_p` and the coordinate-injectivity window must be added to pricing/sizing.

**Why deferred.** Repricing is only meaningful with the weak-binding price (D4) settled, and it regenerates the SIS floor tables and planner schedules.

**Open questions.** Which `norm_bound.rs` consumers implicitly rely on the stage-1 range guarantee (audit needed); the exact `T_p` to bucket; table regen scope.

### D4. Realized `beta_bar_2` from the JL image (operator norm unchanged)

**What.** Finish the SIS accounting identity that JL licenses. The A-role collision **shape does not change**:

```text
eta_A = 2 * Gamma_bar * beta_bar_2
```

`Gamma_bar` is the operator norm of fold-challenge differences in the cross-multiplied weak-binding kernel (`lem:batched-weak-binding`; code path `committed_fold_collision_l2_sq` today uses `8 * omega * beta_inf * nu` then converts to L2). JL does **not** remove `Gamma_bar`. It replaces how `beta_bar_2` is bounded.

**Today (loose).** Stage 1 certifies digit ranges, which yields a deterministic infinity-norm envelope `beta_inf` on the fold response. Pricing uses `||z||_2 <= sqrt(d) * beta_inf`. Calibration shows realized `||z||_2` is often **30-200x smaller** than that envelope. MSIS rank is sized for the envelope, not the realized norm.

**With JL (realized).** Slot-3 reveal (this prototype): after `u'` binds `w_next`, sample `J`, set `p = J * w_next` on the flat coefficient table, absorb `p`, check `||p||_2 <= T_p` over the integers, and prove consistency with the sumcheck. The witness table is the whole next recursive witness:

```text
w_next = (e_hat, t_hat, z_hat, r_hat)   // flat digits; z_hat is a segment, not a separate object
```

So `||z_hat||_2 <= ||w_next||_2` (Euclidean norm on the concatenated digit vector). The binary modular-JL lemma above relates the accepted image threshold to a witness bound: if `T_p = sqrt(a_m) * T_w`, then accepted consistency plus `||p||_2 <= T_p` implies `||w_next||_2 <= T_w` except with the selected statistical failure probability. The honest threshold/regrind policy uses the separate upper-tail constant `B_m`. For weak binding, `beta_resp` bounds the response difference `||z^{(i)} - z^{(0)}||_2 <= 2 * ||z_hat||_2 <= 2 * T_w`. Plug that realized `beta_bar_2` into `eta_A` instead of the `sqrt(d) * beta_inf` pipeline.

**What D4 work actually is.** Not a protocol fork and not an "extraction re-architecture" item:

1. Prove and document the binary constants `(a_m, B_m, Q_m)` for the selected `m` and `lambda` (D6).
2. Pick and document the slack map `T_p -> T_w -> beta_bar_2`, with `T_p = sqrt(a_m) * T_w`, honest buckets from `B_m`, regrind policy D5, and signed-coordinate window.
3. Size `n_rows` so the JL statistical term matches the per-level `2^{-128}` budget (D6).
4. One short writeup lemma: accepted `(p, J, w_next)` implies the `beta_bar_2` used in `norm_bound.rs`.
5. D3: swap the `beta_inf` input for the JL-derived `beta_bar_2` and regen SIS tables.

**Global projection.** One dense `J`, one image `p`, one norm check on the concatenated `w_next` coefficients (prototype default `n_rows = 256`). No per-block projection union.

**Why deferred.** Constants and writeup only; prototype mechanics already match the bound path.

**Open questions.** Tight binary constants `(a_m, B_m, Q_m)` under Akita's modulus and threshold policy; coordinate injectivity `|p_j| < q/2` in the JL-to-field consistency step; whether any `norm_bound.rs` consumer still assumes stage-1 digit-range pricing after JL levels delete stage 1 (D3 audit).

### D5. Completeness: nonce regrind and the norm-threshold policy

**What.** An honest witness occasionally projects to an over-threshold image (the JL window has an upper tail).
The backup handles this with a nonce-regrind loop: the prover searches a small nonce on a cloned transcript, only commits the accepted nonce, and the verifier absorbs that one nonce.
LaBRADOR's check-and-retry analysis gives a small constant in its setting; the Akita threshold and slack must be restated with the chosen `n_rows` and signed-coordinate window (single global projection, so no union model; see D6).
Fusion needs: a bounded regrind nonce in the proof/transcript, a schedule-fixed honest bucket `T_p` sized from calibrated RMS image norms, and a liveness cap (no-panic on exhaustion).

**Why deferred.** The standalone prototype can pick a single transcript draw and a generous `T_p` for tests; regrind is a completeness optimization for the wired path.

**Open questions.** Nonce search budget; how the regrind nonce binds in the descriptor/transcript; interaction with the per-draw 128-bit entropy floor.

### D6. Proving binary constants and sizing `n_rows` for the single global projection

**What.** With a single global projection (D4), the JL failure probability is the **single-shot** lower tail of the modular-JL lemma, not a union over blocks. For binary signs the required deliverable is the target lemma above:

```text
J ∈ {-1,+1}^{m×N},  w ∈ [+-q/2]^N,  ||w||_2 >= b,  b <= q/Q_m
    => Pr[||Jw mod q||_2 < sqrt(a_m) b] <= 2^-lambda.
```

Because the whole concatenated object is projected once and a single global Euclidean norm is certified, there is **no** `#objects` union and no `n_rows ~ 256 + 2*log2(#objects)` inflation. That earlier framing was an artifact of a per-block design the spec no longer pursues.

The proof work is:

1. Reproduce the LaBRADOR three-case modular proof in a standalone note or script comments.
2. Replace the ternary row distribution with binary Rademacher signs.
3. Compute explicit constants for all three cases, including the real lower tail, the wrap upper tail, the large-coordinate cutoff, and the Berry-Esseen spread case.
4. Emit a machine-checkable constants table for `(lambda, m) = (128, 256)` and any other row counts the planner considers.
5. Add a small Monte Carlo sanity bench only as a regression signal; the lemma constants must come from analytic bounds, not simulation.

**Why deferred.** The prototype uses a configurable `n_rows` (default 256) and a generous bound; the exact binary constants feed D3/D4 pricing and D5 completeness.

**Open questions.** Whether Akita keeps `m = 256` or uses a smaller/larger binary row count after constants are known; how aggressive `a_m` can be while keeping `Q_m` friendly to fp32; whether the upper-tail threshold `B_m` gives acceptable nonce-regrind rates; whether root-level JL is used at all (the algebraic range check has no JL statistical failure mode and the live dense-verifier cost is largest at the root).

### D7. ZK masking of the revealed image

**What.** A revealed image `p` leaks `n_rows` linear functionals of `w_next`, so the reveal variant is non-ZK unless masked.
Options: add `n_rows` blinding evaluations (a small deferred-mask family in the existing ZK accounting, alongside the stage-2 masks in `akita-prover/src/protocol/masking.rs` / `zk_hiding_commit.rs`), or restrict the reveal variant to non-ZK builds and terminal-adjacent levels where the witness is about to be sent in the clear anyway.

**Why deferred.** The prototype targets non-ZK builds; the leak is not a blocker for mechanics tests, but it must be resolved before any ZK path uses reveal projection.

**Open questions.** Mask family size and where it slots into the proof-level hiding witness cursor; whether the committed-image (Slot-2) variant is the better ZK path instead.

### D8. The structured ring-granular (committed-image, mid-level) variant

**What.** For mid levels (not the reveal tail), the image is committed in `v` and folded again, which the commutation law forces to a ring-granular `J_0 (x) I_D`.
This is a separate, more complex variant (no reveal, no leak, smaller image overhead via nesting), and it shares the consistency machinery with the reveal variant but checks the image norm via a committed micro-range or exact-`l2`-on-image rather than over `Z`.

**Why deferred.** The user scoped this prototype to the unstructured tail variant; the structured variant is a separate, larger piece.

**Open questions.** All of the committed-image enforcement menu (micro-range vs carry-lifted exact-l2), the nested-projection constants, and the same-level image-norm enforcement that prevents per-level slack from compounding.

### D9. Planner, proof-size, and larger-digit implications

**What.** Add a JL-aware planner scorer before any production schedule cutover.
The goal is to measure the proof-size optimum under the corrected A-role price:

```text
eta_A = 2 * Gamma_bar * beta_bar_2
```

The experiment must not assume the old clean `2 * T_s` anchored price.
It must use a schedule-fixed map `T_p -> T_w -> beta_bar_2` from D4 and keep `Gamma_bar` in the collision bucket.

**Current planner shape.** The production DP in `crates/akita-planner/src/schedule_params.rs` memoizes suffixes by:

```text
(level, current_witness_len, current_witness_len_terminal, current_lb)
```

The phase-0 ladder DP in `crates/akita-planner/src/ladder_byte_model.rs` adds `current_d` to that state:

```text
(level, current_d, current_witness_len, current_witness_len_terminal, current_lb)
```

Both scorers call `level_proof_bytes` in `crates/akita-types/src/proof_size.rs`.
For an intermediate fold, that formula prices:

```text
v_bytes
+ fold_grind_nonce
+ stage1_proof_bytes
+ stage2_sumcheck
+ next_commitment
+ next_eval
```

A JL reveal level changes exactly that local price:

```text
jl_reveal_level_bytes =
  range_level_bytes
  - stage1_proof_bytes
  + image_bytes(n_rows, coordinate_codec)
  + jl_consistency_delta
```

For the intended fused stage-2 form, `jl_consistency_delta` should be zero or a degree-change adjustment, not a second standalone sumcheck.
The consistency row rides in the existing stage-2 batch.
The first measurement should keep the existing stage-2 byte count unless the fusion implementation proves a degree drop and the serialized proof type reflects it.

**New schedule fields.** A JL-aware schedule needs a per-level shortness paradigm, not just `Fold` vs `Direct`.
The minimal production shape is one of:

```text
Step::Fold(FoldStep { shortness: Range, ... })
Step::JlRevealFold(JlRevealFoldStep { n_rows, T_p, coordinate_window, ... })
```

or a `shortness` field inside `FoldStep`.
Use the variant only for levels where the proof payload changes.
The instance descriptor must bind the chosen levels, `n_rows`, `T_p`, coordinate window, seed domain, and codec.

The DP state must include enough information to avoid an off-by-one mistake in birth certification.
If implementation treats the reveal as certifying `w_next` for the next level, the memo key needs a small state bit that says how the current witness is certified:

```text
current_shortness in {Range, JlBirthCertified}
```

If implementation instead stores the paradigm on the level whose stage-1 payload is removed, then the step variant alone is enough.
Do not mix the two models.
The root should stay `Range` until the root statistical budget is re-derived and explicitly accepted.

**JL pricing inputs.** Each candidate JL level must carry:

- `n_rows`, initially 256 for tail reveal experiments, or the D6 row-count rule once re-derived.
- `T_p`, the public image threshold bucket.
- `T_w`, the witness bound implied by `T_p` and the modular-JL slack.
- `beta_bar_2`, the value passed into A-role sizing with `Gamma_bar` retained.
- `coordinate_window`, used both for signed-coordinate injectivity and image wire sizing.
- `coordinate_codec`, initially a fixed signed-int upper bound, later a `Gaussian{k}` segment from `tail-wire-encoding.md`.

**Larger digits.** Do not lift `lb > 6` in the first JL fusion.
The prototype and current prover kernels are still i8-bound:

- `crates/akita-prover/src/validation.rs` caps `MAX_I8_LOG_BASIS = 6`.
- `crates/akita-prover/src/kernels/linear/capacity.rs` derives CRT safe widths using `BALANCED_DIGIT_RHS_MAX_ABS = 1 << (MAX_I8_LOG_BASIS - 1)`.
- `akita-challenges::jl` caps projection digits at `MAX_JL_DIGIT = 32`.
- The optimized JL projection path uses `i32` row sums for i8-sized digits and validates `cols <= i32::MAX / MAX_JL_DIGIT`.

Lifting the basis is an atomic engineering cutover.
It requires an i16 or wider witness path, new CRT capacity profiles with `rhs_abs_bound = 1 << (lb - 1)`, updated digit LUTs and matvec kernels, widened JL digit validation and projection kernels, and proof/tail serialization that can represent those digits.
The planner may run exploratory `lb > 6` simulations, but it must not emit production schedules above 6 until those prover and verifier paths exist.

**Recommended measurement path.**

1. Add a non-production `JlRevealPricing` scorer beside `ladder_byte_model.rs`.
2. Keep root levels as `Range`; allow the last one to three intermediate levels to choose `JlReveal`.
3. Price image bytes first with a conservative fixed signed-coordinate width, then with the tail `Gaussian{k}` codec.
4. Plug in D4's realized `beta_bar_2` and recompute `n_a`.
5. Sweep `fp128_d64_onehot`, `fp128_d64_full`, and fp32 one-hot families at `num_vars = 28` through `32`.
6. Report total bytes, fold bytes, terminal bytes, chosen JL levels, `n_a` deltas, stage-1 bytes removed, image bytes added, and ring-dimension ladder if mixed `D` is enabled.

**Expected outcome.** The proof-size optimum is not known yet.
The likely first-order win is the `n_a` drop from realized `beta_bar_2`, not a large basis or tail contraction.
Stage-1 deletion at tail levels is a smaller local trade: it removes a linear-in-`lb` range tree and adds roughly `n_rows` signed image coordinates.

### Full implementation cutover map

This is the concrete code cutover from the landed ternary prototype to the binary-sign design. It is intentionally one-way: remove ternary packing and update every caller/test/spec surface in the same branch.

**0. Security constants artifact.**

- Add a reproducible constants script before wiring binary JL into production schedules. Suggested path: `scripts/derive_binary_jl_constants.py` or a Rust `xtask` if the repo gets one.
- Inputs: `m`, `lambda`, `a`, `B`, `Q`, Berry-Esseen constant, large-coordinate cutoff.
- Outputs: a checked table for `(m, lambda)`, initially `(256, 128)`, with `(a_m, B_m, Q_m)` and a per-case failure ledger.
- Add the proof note to this spec or a companion `specs/binary-jl-modular-lemma.md`; cite Achlioptas for real binary JL, Greyhound for implementation precedent, and LaBRADOR only as the modular proof template.

**1. Matrix representation in `akita-challenges::jl`.**

- Change `row_bytes_for(cols)` from `ceil(2*cols/8)` to `ceil(cols/8)`.
- Replace the pair decoder with `bit_to_sign(bit) = if bit { +1 } else { -1 }`.
- Keep `JL_SAMPLE_DOMAIN_VERSION` unchanged while this PR branch is in development, per maintainer request. Revisit before any release/wire-stability point; every sampled matrix changes from the packing change even without a version bump.
- Keep `JlProjectionMatrix` as the only type. Do not add an alphabet enum or a ternary compatibility constructor.
- Update `from_sign_rows` test helper to pack one bit per sign and reject zero signs.

**2. Projection kernels.**

- Replace `SIGN_LUT = [-1,0,0,1]` and `SIGNS_FOR_BYTE: [[i8;4];256]` with either direct bit decode or `BINARY_SIGNS_FOR_BYTE: [[i8;8];256]`.
- Update scalar, NEON, AVX2, and AVX-512 kernels to process eight coefficients per row byte.
- Keep the checked `i64` reference path and differential tests.
- Re-benchmark sign-table multiply-add versus bitmask add/sub. Prefer the simpler sign-table path unless the bitmask variant wins clearly across Apple Silicon and x86.

**3. MLE kernels.**

- Delete `BYTE_TO_TERNARY4`, `pair_to_ternary_digit`, and the 81-pattern DP.
- Add a 16-entry nibble LUT builder:

```text
lut16[bits] = sum_{lane=0..3} sign(bit_l) * weight_l
```

- For each packed byte, use one low-nibble and one high-nibble lookup. Direct 256-entry byte-table construction is a benchmarked alternative, not the default.
- Update `accumulate_row_weight_range` and `scatter_row_weight_range` to decode binary signs.
- Keep verifier final eval fused; do not materialize `g` on the verifier.

**4. Wire/layout/protocol tests.**

- Update all test sign matrices to use only `±1`.
- Replace packed round-trip tests with one-bit packing tests.
- Keep all malformed-shape and no-panic tests.
- Add distribution sanity tests: sampled matrix bytes should be deterministic for fixed transcript, and sign counts should be close enough to 50/50 under a fixed large sample for a non-cryptographic smoke test.

**5. Spec/docs and public API names.**

- Rename comments/docs from "ternary", "sparse-ternary", and "zero pairs" to "binary sign" / "Rademacher".
- Keep API names generic (`JlProjectionMatrix`, `eval_jl_mle_at`) so the cutover does not leak representation names into downstream crates.
- Update benches labels from `ternary` to `binary` only where labels encode the distribution.

**6. Fusion integration after binary mechanics are green.**

- D1: bind binary JL geometry and `JL_SAMPLE_DOMAIN_VERSION` in the instance descriptor.
- D2: add the JL proof payload and fused stage-2 row against the binary `eval_jl_mle_at`.
- D3/D4: switch A-role pricing to binary `beta_bar_2` using `(a_m, B_m, Q_m)`.
- D5: add bounded nonce regrind with the accepted nonce absorbed before `CHALLENGE_JL_ROW`.
- D9: enable planner experiments only after the binary constants table exists.

**Verification gate.**

Run at minimum:

```bash
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test -p akita-challenges -- jl
cargo test -p akita-types -- jl
cargo test -p akita-prover -- jl
cargo test -p akita-verifier -- jl
cargo test
```

For a spec-only patch, no Rust tests are required. For the binary code cutover, the targeted JL tests should fail before updating tests if any ternary decoder remains reachable; that is the desired tripwire.

### Suggested ordering for fusion

D6 and D4 (security: prove binary constants, size `n_rows` for the single projection, settle the `T_p -> T_w -> beta_bar_2` map with the operator factor retained, and settle coordinate injectivity) settle pricing inputs; the binary mechanics cutover above makes PR #191's standalone code match the intended distribution; D9 gives the planner experiment; D3 (delete stage 1, reprice, regen tables) and D1 (descriptor) are the type/accounting changes; D2 (Step, payload, stage-2 fusion) is the protocol cutover; D5 (regrind) and D7 (ZK) are completeness/ZK work; D8 is a separate mid-level project.

## Documentation

- This spec is the primary design record for the prototype and the fusion roadmap.
- Crate-level module docs (`//!`) landed on `akita-challenges::jl`, `akita-challenges::jl::mle`, `akita-types::jl`, `akita-prover::protocol::jl`, `akita-verifier::protocol::jl`, and the kernel/MLE submodules.
- No public book or security-doc changes until fusion (the prototype changes no shipped behavior). When fusion lands, the security-model and norm-bounds pages need the JL paradigm-schedule and the weak-binding realized-norm pricing (operator factor retained).

## Execution

Prototype phases (all complete on PR #191):

1. [x] `akita-challenges::jl`: packed ternary matrix in PR #191, transcript-seeded expansion, `project`, norm helpers, signed-coordinate embedding. The binary cutover above replaces only the matrix alphabet/packing and its dependent kernels.
2. [x] Signed-coordinate encoding and checked norm helpers (before consistency work).
3. [x] `akita-challenges::jl::mle` (PR1b): `eval_jl_mle_at` + `build_jl_row_weights`; LUT production path; differential tests; `benches/jl_mle.rs`.
4. [x] `akita-types::jl` + `akita-prover::protocol::jl` + `akita-verifier::protocol::jl` (PR2): degree-2 product sumcheck via `akita-sumcheck`; `JlWitnessLayout` pins flat order `w[x * 2^ring_bits + y]`; prove on prover, verify on verifier with `w_eval_hook`; round-trip, tampered-image, norm-bound, and layout tests.
5. [x] Lint and targeted test sweep (`cargo test -p akita-challenges -- jl`, `cargo test -p akita-types -- jl`, `cargo test -p akita-prover -- jl`, `cargo test -p akita-verifier -- jl`).

Open before fusion (D2):

- Confirm `JlWitnessLayout::flat_index` matches stage-2 `w(x, y)` / `build_w_coeffs` on real prover layouts (prototype tests use synthetic small layouts only).
- Confirm gamma power and fused stage-2 oracle wiring against `akita_stage2` when cutting over.

## References

- `labrador-backup:src/protocol/labrador/johnson_lindenstrauss.rs`: reusable dense reveal projection implementation ideas (packed ternary, SHAKE seed expansion, centering, projection, nonce regrind). Port contracts, not constants or overflow return types, without re-auditing them.
- `akita-algebra/src/eq_poly.rs` (`SplitEqEvals`), `akita-types/src/extension_opening_reduction.rs` (`tensor_column_partials_split_fold`): split-eq contraction template for PR1b MLE eval.
- Dao & Thaler, ePrint 2024/1210 (`DaoThaler_SplitEq_SumCheck_2024_1210.pdf` in paper index): nested eq tables and iterated sums; cited in `akita-algebra/src/split_eq.rs`.
- Achlioptas, "Database-friendly random projections: Johnson-Lindenstrauss with binary coins" (JCSS 2003, https://doi.org/10.1016/S0022-0000(03)00025-4): real-valued JL concentration for independent binary `±1` entries, scaled by `1/sqrt(k)`.
- Gentry-Halevi-Lyubashevsky, "Practical Non-Interactive Publicly Verifiable Secret Sharing with Thousands of Parties" (EUROCRYPT 2022 / ePrint 2021/1397, https://eprint.iacr.org/2021/1397): upstream modular-JL predecessor cited by LaBRADOR.
- LaBRADOR (CRYPTO 2023, DOI https://doi.org/10.1007/978-3-031-38554-4_17): modular-JL lemma (Lemma 4.1/4.2), the 256-row single-projection setting, the weak-binding A-price (Thm 5.1, where the JL term is `4*T*sqrt(128/30)*beta` -- operator norm retained), and the check-and-retry threshold policy. The weak-binding argument is the template (D4): JL certifies the realized output norm; the operator factor in the A-binding is structural and does not drop. Akita's relation being linear in the blocks does **not** remove it.
- Greyhound, ePrint 2024/1293 (https://eprint.iacr.org/2024/1293): implementation precedent for switching LaBRADOR-style JL matrices to binary `±1` entries and using 16 signed sums of four coefficients in the projection kernel. Use as an implementation/data-point citation, not as the proof of binary constants.
- Grand Danois (eprint 2026/1196): structured-projection-in-relation and the nested lever (for the deferred mid-level variant); constants must be re-derived.
- `akita-types/src/sis/norm_bound.rs`: A/B/D-role collision pricing the realized-norm repricing (D3, D4) targets.
- `akita-types/src/instance_descriptor.rs`: the transcript preamble that must bind JL geometry (D1).
- `akita-types/src/schedule.rs`, `akita-types/src/proof/levels.rs`, `akita-prover/src/protocol/sumcheck/akita_stage2`: the `Step`, proof payload, and fused stage-2 surfaces fusion touches (D2).
- Profiling (for the fusion measurement): `AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 cargo run --release --example profile`.
