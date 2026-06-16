# Spec: Unstructured JL Projection Prototype (tail reveal, standalone)

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao, Cursor agent (model: Claude Opus 4.8) |
| Created     | 2026-06-15                     |
| Status      | proposed prototype; security accounting unresolved |
| PR          |                                |

## Summary

Akita is a lattice-based recursive polynomial commitment scheme.
Each recursive level commits a short witness, folds it against a challenge, and proves enough shortness for the weak-binding extractor to recover a norm-bounded opening.
Today that certificate is a stage-1 infinity-norm range-check sumcheck on the decomposed fold response, and the recursion ends by sending the final folded witness in cleartext (the terminal direct step).
Recent fixed-width planner snapshots for the fp32 one-hot families show the terminal cleartext witness is a large part of the proof at the shipped folding sizes, and each intermediate level also pays for its stage-1 range tree.
Those byte numbers are calibration, not a security premise, and they should be re-measured after the active tail-encoding work lands.

This spec defines a minimal, self-contained prototype of an alternative tail-level shortness mechanism: an unstructured (dense, field-granular) Johnson-Lindenstrauss random projection.
The verifier samples a dense ternary projection matrix from the Fiat-Shamir transcript; the prover projects the witness to an integer image `p`, reveals `p`, and the verifier checks a Euclidean norm bound on `p` over the integers.
A sumcheck checks projection consistency against the witness multilinear extension: JL rows are batched with `eq` weights on `row_bits = log2(n_rows)` challenges (not Vandermonde `rho^j`), the public weight is the joint sparse-ternary MLE `J_tilde(r_J, r_w)` partially evaluated in `r_J`, and the standalone prototype exercises that degree-2 product relation before fusion.
The wired protocol would replace the stage-1 infinity-norm range sumcheck at selected JL levels with an image-norm check plus a projection-consistency row.
This prototype does not make that replacement in the recursive flow.

The prototype is built as standalone, well-tested library code that is not wired into the recursive prove/verify flow.
The goal is to land reusable JL primitives, exercise the consistency-sumcheck mechanics across representative fields and ring dimensions, and document the fusion roadmap so later protocol-integration work can measure any proof-size or rank impact on the real flow.

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
The current protocol certifies the fold response `z` with a range check; the JL direction certifies the realized norm of the committed blocks statistically, replacing the loose deterministic envelope.
JL certifies only objects fixed before `J` is sampled (the committed blocks, pinned by their commitment), not the post-fold quotient the extractor naturally produces; the operator-norm weak-binding price is unchanged, as in LaBRADOR (see D4).
Today that certificate is **stage 1**: the balanced base-`2^lb` digits of `z` are committed and a range-check sumcheck of degree `2^lb` proves each digit lies in the digit set, giving an infinity-norm bound `||z||_inf <= beta_inf`, converted to Euclidean via `||z||_2 <= sqrt(d) * beta_inf`.
The deterministic envelope `beta_inf = num_claims * 2^r_vars * min(||c||_inf*||s||_1, ||c||_1*||s||_inf)` is what the SIS accounting prices today (see the A-role collision below).
Prior calibration notes report that this deterministic envelope is often 30 to 200 times larger than the realized `||z||_2`.
That motivates replacing the envelope, but calibration alone is not a soundness argument.

### The Johnson-Lindenstrauss alternative

A random projection can certify Euclidean shortness directly once its lower-tail failure probability and transcript-grinding budget are accounted for.
Fix a vector `s` before sampling a random matrix `J` with entries in `{-1, 0, +1}` and `n_rows` rows.
With high probability over `J`, a vector fixed before `J` is sampled cannot have a large Euclidean norm while its image has a small Euclidean norm.
The relevant lower-tail constant comes from the LaBRADOR modular-JL analysis, but the exact `c(n_rows)` and slack used by Akita must be re-derived rather than copied from the 256-row backup.
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
The matrix can be a plain dense field-granular `{-1, 0, +1}` matrix, provided the protocol is only using it in this direct-consistency form.
This is the "unstructured" projection this spec prototypes, and it is the simplest variant: dense matrix, reveal the image, check the norm over `Z`, prove consistency with one sumcheck.
(The structured ring-granular committed-image variant for mid levels is explicitly out of scope.)

### The three corrections that frame the value

Prior internal analysis flagged three things that constrain how this prototype is framed:

1. **JL at the tail does not replace the terminal cleartext send.** The terminal direct opening is the PCS base case: something must still verify the final evaluation claim. The reveal projection may delete stage 1 at a tail level and may shrink the last one or two committed levels (replace a full-witness reveal with a small image plus birth-certification of `w_next`), but it does not eliminate the terminal witness. Frame this as "delete stage 1 plus reveal a small image," not "shrink the terminal 3-4x."
2. **JL is not a decomposition-basis byte win.** Lifting the decomposition basis was hypothesized to shrink the tail under JL; a planner sweep showed total proof size changes by at most 1 percent at small sizes and 0 percent at `num_vars >= 28`, because the digit-packing identity `delta * lb ~ field_bits` magnitude-locks every cleartext segment while the module rank `n_a(lb)` grows. So the candidate byte case for JL rests on deleting stage 1, possible tail-payload changes, and any proven SIS-rank repricing, not on a basis lever.
3. **JL does not drop the operator-norm factor from the A-role; it tightens the certified norm.** An earlier framing hoped JL could replace `8*omega*beta_inf*nu` with a clean `2*T_s` by certifying the extracted block `s_i` short and colliding two openings via the bare kernel `s_i - s'_i`. That is **circular** (see D4): bounding `s_i - s'_i` needs `s_i, s'_i` themselves JL-certified short, which requires them fixed before `J` is sampled (Lemma 4.2), which requires A-binding to pin them, and the only non-circular A-binding is the operator-norm/response collision LaBRADOR and akita already use. What JL actually buys, exactly as in LaBRADOR (Thm 5.1, whose A-binding keeps the `T` factor as `4*T*sqrt(128/30)*beta`), is a **realized-norm certificate**: it bounds the realized recursive-witness norm (and through it `||z||_2`), replacing the deterministic envelope `beta_inf` that calibration shows is 30-200x loose. The A-role binding keeps its operator-norm factor; the win is `beta_inf -> realized`, the same realized-norm tightening the exact-l2 certificate targets, but with no no-wrap gate. The prototype does not touch SIS pricing.

### Reusable prior code

The retired `labrador-backup` branch contains a working dense reveal projection at `labrador-backup:src/protocol/labrador/johnson_lindenstrauss.rs`: a 256-row ternary matrix packed 2 bits per entry, deterministic per-row expansion from a 32-byte transcript seed via SHAKE, an integer projection path that centers base-field coefficients and rejects overflow, a nonce-regrind (retry-until-the-image-is-short) completeness loop, and a `collapse` (dot with a coefficient vector) helper.
The implementation techniques are reusable.
The transcript hosting, row count, coordinate encoding, overflow policy, and security constants must be reworked for current Akita rather than ported as-is.

## Intent

### Goal

Land a standalone, field-generic and ring-dimension-generic JL projection prototype that samples a dense ternary projection matrix from a transcript seed, projects a witness to an integer image, checks the image Euclidean norm over the integers, and exercises projection-consistency with one degree-2 sumcheck in the intended fused-row oracle form, without wiring it into the recursive prove/verify flow.

Concretely the prototype introduces:

- `akita-challenges::jl` (new module): the dense projection matrix `JlProjectionMatrix`, deterministic transcript-seeded expansion (`sample`), integer projection (`project`) over centered witness coefficients, signed-coordinate encoding checks, and Euclidean-norm helpers. The projection acts on the flat integer coefficient vector of base-field elements, so the public API takes a flat `&[F]` coefficient slice (`F: FieldCore + CanonicalField`) and the caller flattens any ring layout. Ring structure (`const D`) is irrelevant to the projection and reappears only in the consistency sumcheck (the `akita-prover` module below), so it is not a parameter of this crate; this keeps `akita-challenges` at its field + transcript dependency layer.
- `akita-challenges::jl::mle` (new submodule, PR1b): optimized joint-matrix MLE evaluation `eval_jl_mle_at` and the prover-side row-weight builder `build_jl_row_weights`. This is the verifier-critical bottleneck and reuses the packed-ternary row format plus Dao–Thaler split-eq contraction (see **Joint MLE evaluation** below). Adds a dependency on `akita-algebra` for `SplitEqEvals` only in this submodule.
- `akita-prover::protocol::jl` (new module, PR2, not called by the flow): the projection-consistency claim builder that batches JL rows with `eq` weights on the row hypercube, wires `eval_jl_mle_at` / `build_jl_row_weights` into the sumcheck oracle, and runs a prototype degree-2 product sumcheck. Since the standalone module is not connected to a commitment verifier, its verifier side must either take the witness as public test data or take an external trusted `w_tilde(r)` evaluation hook. Full cryptographic binding is deferred to fusion.
- Cross-field, cross-dimension tests that exercise representative non-degenerate `(field, ring dim)` combinations the workspace ships.

### Invariants

- **Determinism / replayability.** For a fixed transcript state and fixed `(n_rows, cols)`, `JlProjectionMatrix::sample` is a pure function of the transcript-derived seed; prover and verifier reconstruct the identical matrix. Protected by a determinism test (two independent transcripts in the same state produce equal matrices and equal projections).
- **Projection correctness.** `project(w)` equals the exact integer matrix-vector product `J * centered_coeffs(w)` with no modular reduction; the image lives over the integers (balanced representatives of `w`). Protected by a test against a naive reference projection and by the consistency sumcheck harness.
- **Coordinate injectivity.** Any coordinate that is later embedded into the field for consistency must be checked to lie in the chosen signed encoding window, for example `|p_j| < q/2`. This prevents modulo-`q` aliases from passing the field consistency check with a different integer norm. Protected by boundary tests around the signed encoding limit.
- **No overflow on the integer path.** Centering uses balanced representatives in `[-(q-1)/2, (q-1)/2]`. The production fast path centers to `i32` digits and enforces the balanced-digit bound `|d| <= MAX_JL_DIGIT` (`= 32`, i.e. `lb <= 6`) at the boundary (`validate_digit_witness`), so every row sum `sum_i +-d_i` fits `i32` for any supported column count (`cols <= i32::MAX / MAX_JL_DIGIT`); accumulation is then unchecked `i32` on the hot path (including in the SIMD kernels, whose per-lane partials are bounded by the same argument). A non-digit input whose centered magnitude exceeds the digit bound (e.g. a full-magnitude fp128 element) is rejected at the boundary rather than wrapped or saturated, which is correct: it is not a JL witness. The checked-`i64` reference projection (`project_digits_reference`, test/bench only) is the correctness oracle and is the one place wider accumulation guards against overflow. The squared-norm reduction accumulates in `u128` (`l2_norm_sq_checked`); it is `O(n_rows)` and off the hot projection path. Protected by small-digit, oversized-rejection, digit-bound, fast-vs-`i64`-reference differential, and norm-bound tests.
- **Norm check is over the integers.** Shortness is accepted from `||p||_2^2 <= T_p^2` over the integers, never from a squared-sum identity modulo `q`. This avoids the exact-`l2` no-wrap gate, but it still relies on the coordinate-injectivity check above. Protected by a completeness test (honest witness passes under a generous prototype bound) and a soundness-direction test (an over-norm image is rejected).
- **Consistency claim has the intended fused-row form.** Rows are batched with `eq` weights on `row_bits = log2(n_rows)` variables (default `n_rows = 256`, `row_bits = 8`), matching the relation-sumcheck batching idiom rather than a Vandermonde `rho^j` fold. The public witness weight is `g(i) = sum_j eq(r_J, j) J[j, i]` on coefficient corners; equivalently `g_tilde(r_w) = J_tilde(r_J, r_w)` for the joint MLE `J_tilde` of the sparse-ternary matrix on the product hypercube `{0,1}^{row_bits} x {0,1}^{k_w}`. The proved identity is `sum_j eq(r_J, j) p[j] = sum_{x,y} g(x,y) w(x,y)`, a degree-2 product sumcheck `w_tilde * g_tilde` with input claim `sum_j eq(r_J, j) embed(p[j])`. This is the intended stage-2 fusion shape, but final drop-in compatibility still has to be checked against the current `w(x, y)` layout and stage-2 verifier API. Protected by a prove/verify round-trip test and a Schwartz-Zippel soundness-direction test (a wrong image fails except with the usual `r_J` / sumcheck collision probability).
- **No-panic on verifier-reachable paths.** Every shape mismatch (matrix dimensions, image length, point dimension, coefficient overflow) returns `AkitaError`, never panics, matching the verifier no-panic contract because the consistency check is intended to become verifier-reachable. Protected by malformed-input tests.
- **No protocol-flow regression.** Nothing in `prove_batched` / `verify_batched` calls the new modules, so all existing prover/verifier/integration tests pass unchanged and no serialized proof type changes. Protected by the full existing test suite.
- **Joint MLE evaluation correctness.** `eval_jl_mle_at(J, r_J, r_w)` equals the naive double sum `sum_{j,i} eq(r_J,j) eq(r_w,i) J[j,i]` for every packed matrix and challenge point; `build_jl_row_weights(J, r_J)[i]` equals `sum_j eq(r_J,j) J[j,i]`. Protected by differential tests against a reference implementation and cross-checks that `eval_jl_mle_at` matches `eval_mle_from_weights(build_jl_row_weights(...), r_w)`.

### Non-Goals

These are the deferred items; each is investigated in the "Deferred work" section so the follow-up is fully scoped.

- **No wiring into the recursive flow.** No new `Step` variant, no `Schedule`/planner change, no serialized proof-type change, no `prove_batched` / `verify_batched` change.
- **No structured / ring-granular committed-image (mid-level) projection.** Only the dense reveal projection is built.
- **No weak-binding repricing write-up and no SIS repricing.** The realized-norm A-role price (operator factor retained, D4) is a security-accounting task, not needed to prototype mechanics.
- **No ZK masking of the revealed image.** The revealed image leaks `n_rows` linear functionals; the prototype targets non-ZK builds.
- **No exact `n_J` / `c(n_rows)` derivation.** `n_rows` (default 256) and the norm bound are configurable parameters.
- **No removal of the terminal cleartext base case.**

## Evaluation

### Acceptance Criteria

- [ ] `akita-challenges::jl` compiles and exposes `JlProjectionMatrix` with transcript-seeded `sample`, a flat `project(&[F])`, signed-coordinate validation, and checked norm helpers, generic over `F: FieldCore + CanonicalField` (no `const D` in this crate).
- [ ] `akita-challenges::jl::mle` exposes `eval_jl_mle_at` (fused verifier path) and `build_jl_row_weights` (prover path); fast kernels match reference on fp32/fp64/fp128 extension fields used by shipped configs; tail-geometry bench documents throughput at `n_rows = 256` and `cols` in the shipped tail range.
- [ ] Determinism test: two transcripts in identical state yield byte-identical matrices and equal projections.
- [ ] Projection-vs-reference test: `project` matches a naive integer reference for random witnesses across fields and dims.
- [ ] fp128 digit test: small balanced digits project correctly over an fp128 base field; a non-digit, full-magnitude fp128 coefficient (centered value past `i64`) is rejected without panic.
- [ ] Signed-coordinate tests: accepted coordinates embed injectively into the base field, and boundary aliases are rejected.
- [ ] `akita-prover::protocol::jl` consistency prove/verify round-trips for honest `(w, p)` across representative non-degenerate `(field, ring dim)` combinations, using public test witness data or an explicit `w_tilde(r)` evaluation hook.
- [ ] Soundness-direction tests: an image inconsistent with `w` is rejected by the consistency sumcheck for all but a negligible fraction of `r_J`; an over-norm image is rejected by the norm check.
- [ ] Malformed-input tests: wrong matrix shape, wrong image length, wrong point dimension all return `AkitaError`, never panic.
- [ ] All pre-existing workspace tests pass unchanged.
- [ ] `cargo fmt -q`, `cargo clippy --all -- -D warnings`, and the relevant test passes are green.

### Testing Strategy

New tests live alongside the new modules.

- `akita-challenges`: unit tests for `sample` determinism, packed-matrix round-trip (`00 -> -1`, `01/10 -> 0`, `11 -> +1`), `project` correctness vs reference, signed-coordinate injectivity, checked integer norm computation, fp128 small-digit projection, and oversized non-digit rejection. Port the analogous implementation tests from `labrador-backup:src/protocol/labrador/johnson_lindenstrauss.rs`, but do not inherit its fixed row count or its dual `i64`/`i128` width split.
- `akita-challenges::jl::mle`: differential tests of `eval_jl_mle_at` and `build_jl_row_weights` against a naive `Θ(n_rows · cols)` reference; identity `eval_jl_mle_at(J,r_J,r_w) == eval_mle_from_weights(build_jl_row_weights(J,r_J), r_w)`; SIMD vs scalar kernel parity; malformed shape errors return `AkitaError`.
- `akita-prover`: a `protocol::jl` test module that builds a random witness, samples `J` and the row batching point `r_J`, computes `p`, and round-trips the consistency sumcheck through the real `akita-sumcheck` driver and the transcript. Since this is standalone, the test verifier supplies the witness evaluations directly or through an explicit final-evaluation hook. The sweep covers representative non-degenerate `(field, ring dim)` pairs: fp32, fp64, fp128 base fields and the supported `D` values used by shipped configs. Include the soundness-direction and malformed-input tests.

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
| Row weights `g[i] = Σ_j eq(r_J,j) J[j,i]` | Prover (sumcheck oracle table) | PR1b `build_jl_row_weights` | `Θ(n_rows · cols)`; must touch every matrix entry |
| Point eval `J̃(r_J,r_w)` | Verifier (fusion final check) | PR1b `eval_jl_mle_at` | `Θ(n_rows · cols)`; **fused** one-shot, no `2^{k_w}` weight table |
| Consistency sumcheck | Prover + verifier | PR2 | `k_w` witness rounds; verifier JL work is dominated by `eval_jl_mle_at` |

There is no closed-form shortcut for a dense random `J` (unlike `TraceWeight`). Constant-factor wins come from split-eq nesting, byte-wide ternary decode, deferred extension-field reduction, column-panel / outer-tile parallelism, and SIMD. Ternary sparsity (`P(J=0) = 1/2`) reduces ALU in the inner loop but **not** memory traffic. At tail scale (`n_rows = 256`, `cols ~ 2^{17}`) the verifier eval is intentionally `Θ(n_rows · cols)` but must be engineered to the same standard as PR1 projection kernels.

## Design

### Architecture

The prototype splits along the existing crate dependency graph.

**`akita-challenges::jl` (matrix sampling, projection, norm).**
This crate owns Fiat-Shamir challenge sampling and depends only on `akita-field` and `akita-transcript`, the right layer for the projection primitives (field-level math, transcript-seeded, no sumcheck).
It already exposes a SHAKE256-backed streaming `XofCursor` (with bias-free draws and a `next_sign` helper) and the transcript seed-derivation pattern used by the sparse-challenge sampler, which the JL module reuses.

- `JlProjectionMatrix { n_rows, cols, row_bytes, packed_rows: Vec<u8> }`: a dense ternary matrix with entries `{-1, 0, +1}` packed 2 bits per entry (`00 -> -1`, `01/10 -> 0`, `11 -> +1`) in a single contiguous row-major buffer, drawn from the transcript-derived XOF stream.
- `sample<F, T>(transcript, n_rows, cols) -> Result<Self, AkitaError>`: absorbs a context buffer (label, `n_rows`, `cols`, and a version/domain tag), draws a 32-byte seed, and expands rows deterministically. Per-row derivation, as in the backup, keeps generation parallel-safe; a single streaming `XofCursor` is also acceptable if tests lock the exact transcript behavior and no parallel row generation is required. `n_rows` and `cols` are parameters (default test `n_rows = 256`), not hardcoded.
- `project<F>(&self, coeffs: &[F]) -> Result<JlImage, AkitaError>` (`F: FieldCore + CanonicalField`): centers each coefficient to its balanced `i32` representative (`center_coefficients`) and projects through the fast kernel to `i32` image coordinates. JL is only ever applied to small balanced digits (`|d| <= MAX_JL_DIGIT = 32`), so `i32` holds both the centered digits and every row sum with room to spare; there is no `i64`/`i128` on the hot path. The modulus is recovered for `CanonicalField` as `(-F::one()).to_canonical_u128() + 1`; if a clearer modulus helper is added later, use it instead. A non-digit input whose centered magnitude exceeds the digit bound (e.g. a full-magnitude fp128 element) is rejected at the boundary rather than wrapped or saturated. Returns an error on a digit-bound violation or a `coeffs.len() != cols` shape mismatch. `project_digits(&[i32])` is the pre-centered entry point; `project_digits_reference` is a checked-`i64` oracle for tests/benches.
- `JlImage` (or an equivalent explicit type) stores signed integer coordinates, exposes checked embedding into `F` only when the configured signed window is injective, and exposes checked norm helpers such as `l2_norm_sq_checked` / `check_l2`.

**`akita-challenges::jl::mle` (joint matrix MLE, PR1b).**
Depends on `akita-algebra` for `SplitEqEvals`. Shares packed-row decode (`SIGNS_FOR_BYTE`) and SIMD dispatch conventions with `project`.

- `eval_jl_mle_at<L>(matrix, r_J, r_w) -> Result<L, AkitaError>`: fused verifier contraction (see **Joint MLE evaluation**).
- `build_jl_row_weights<L>(matrix, r_J) -> Result<Vec<L>, AkitaError>`: prover row-weight table `g`.
- Scalar reference + arch-specific tile kernels (`kernels/mle_scalar.rs`, `mle_neon.rs`, `mle_x86.rs`); runtime dispatch like PR1.

The matrix sampling and projection geometry (`n_rows`, seed domain) are intended to be bound into the instance descriptor when fusion lands; the prototype binds them through the transcript context buffer only and records descriptor binding as a fusion task.

**`akita-prover::protocol::jl` (consistency claim, prototype prove/verify).**
`akita-prover` depends on `akita-challenges`, `akita-sumcheck`, `akita-witness`, `akita-transcript`, and `akita-algebra`, so it is the natural home for the consistency sumcheck and the place fusion will happen. The module is not referenced by `flow.rs` or any prove/verify entry point.

See **Projection-consistency sumcheck** below for the default relation, batching, and verifier evaluation model.

- `build_jl_row_weights(matrix, r_J) -> Vec<L>` (PR1b, in `akita-challenges::jl::mle`): same as the former `batch_jl_rows`; for each witness coefficient index `i`, compute `g[i] = sum_j eq(r_J, j) J[j, i]`. Column-panel / split-eq implementation; reference name kept in prose as "row weights".
- `eval_jl_mle_at(matrix, r_J, r_w) -> L` (PR1b): fused verifier path; equals `sum_{j,i} eq(r_J,j) eq(r_w,i) J[j,i]` without materializing the full `g` table.
- `build_jl_weight_table(layout, g) -> JlWeightTable`: map `g[i]` to `JlWeight(x, y)` on the same `(x, y)` hypercube as stage-2 witness table `w` (flattening convention must match `center_coefficients` / `build_w_coeffs`).
- `JlConsistencyInstance`: degree-2 product sumcheck via `akita-sumcheck` (`oracle = w_tilde * g_tilde`, `input_claim = sum_j eq(r_J, j) embed(p[j])`).
- `prove_jl_consistency` / `verify_jl_consistency`: absorb the image, sample `r_J` from the transcript after the witness is bound (wire-before-squeeze), check coordinate injectivity and the integer norm bound, build `g` / `JlWeight`, then run the sumcheck. The verifier obtains `w_tilde(r_w)` from the sumcheck output (standalone tests may supply a witness hook).

**Why the consistency is in fused-row form now.**
The current stage-2 fused sumcheck batches subclaims by powers of `gamma`: `gamma^0 * relation + gamma^1 * range + gamma^2 * trace`, all sharing one witness scan over `w(x, y)`.
At JL levels the range term is deleted and replaced by a JL row `gamma^k * w(x,y) * JlWeight(x,y)` with public table `JlWeight = g`.
The witness MLE `w_tilde(r_w)` at the final point is shared across relation, trace, and JL; JL-specific verifier work is `J_tilde(r_J, r_w)`.
The exact `gamma` power `k`, the `w(x, y)` column layout, and the final-point evaluation path remain D2 work.

### Projection-consistency sumcheck

This subsection is the canonical design for PR2 (`akita-prover::protocol::jl`) and for stage-2 fusion (D2).

#### Objects and indexing

- `J ∈ {-1,0,+1}^{n_rows × cols}`: dense ternary matrix from `JlProjectionMatrix::sample` (packed two bits per entry).
- `n_rows` is a power of two in production (`256` default, `row_bits = 8`). If not, pad with zero rows and zero image coordinates.
- Witness coefficients `c_i` (centered digits) index corners of `{0,1}^{k_w}` with `k_w = col_bits + ring_bits`, matching stage-2 table `w(x, y)` under the same flatten map `i = index(x, y)` used by the prover layout.
- Row index `j ∈ {0,1}^{row_bits}` (little-endian, same convention as `EqPolynomial` elsewhere).
- Integer image `p_j = sum_i J_{j,i} c_i` (exact `ℤ`; revealed for the tail prototype).

#### Joint sparse-ternary MLE of `J`

Define the truth function on the product hypercube:

```text
J_hat(j, i) = J[j, i]  in {-1, 0, +1}
```

Its multilinear extension over challenge points `(r_J, r_w)` with `r_J ∈ L^{row_bits}` and `r_w ∈ L^{k_w}` is:

```text
J_tilde(r_J, r_w)
  = sum_{j, i} eq(r_J, j) * eq(r_w, i) * J[j, i]
```

`J_hat` is **dense** on the hypercube (random `J`), but **ternary** at every corner: values lie in `{-1,0,+1}`, and the implementation uses the same packed-row format as integer projection.

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

The prover builds `JlWeight` once from `J` and `r_J` via `build_jl_row_weights` (`Θ(n_rows · cols)` ternary work, shared kernels with projection and MLE). The witness scan reuses the stage-2 pattern (public weight table times witness MLE).

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

#### Default algorithm: tensor split-eq

Factor the product MLE by axes (not an arbitrary bisection of the concatenated challenge):

```text
J̃(r_J, r_w)
  = Σ_{j_o,j_i,i_o,i_i}
      e_J_out[j_o] · e_J_in[j_i] · e_w_out[i_o] · e_w_in[i_i] · J[j,i]
```

with `j = (j_o, j_i)`, `i = (i_o, i_i)` (little-endian splits; `row_bits` and `k_w` each split near half, e.g. `8 → 4+4`, `17 → 9+8`).

**Precompute** (once per eval, size `2^{m_Ji + m_wi}`, L2-resident at tail scale):

```text
W[j_i, i_i] = e_J_in[j_i] · e_w_in[i_i]
```

**Outer loop** (parallel over `(j_o, i_o)` tiles): for each tile, accumulate an inner sum over `(j_i, i_i)` by scanning the corresponding `J` submatrix in row-major packed form, then add `e_J_out[j_o] · e_w_out[i_o] · inner` to the running total.

This matches `tensor_column_partials_split_fold`: outer `e_out` multiplies a deferred-reduction inner face sum. Use `GruenSplitEq` only inside incremental sumcheck rounds; this slice uses one-shot `SplitEqEvals`.

**Loop order for cache:** within a tile, fix `(j_o, j_i)` and scan `i_o · 2^{m_wi} + i_i` along contiguous row bytes (same geometry as PR1 column-panel projection).

#### Inner hot loop: ternary × eq weight

Matrix entries are uniform random 2-bit pairs (`00 → -1`, `01/10 → 0`, `11 → +1`), so `P(J=0) = 1/2` and `P(J=±1) = 1/4` each.

Per matrix entry the update is `acc ± W` (add, subtract, or skip), not a general field multiply by `J`.

**Byte-wide processing (primary micro-kernel):** reuse PR1's `SIGNS_FOR_BYTE` LUT (`256 × 4` `i8`, 1 KiB, L1-resident). For each packed row byte covering four consecutive columns, load four eq weights `W[j_i, i_i..i_i+3]` and four signs from the LUT; accumulate into a deferred extension-field accumulator (`MulBaseUnreduced` / `ProductAccum`, same contract as `partials_out_contribution`).

**SIMD:** mirror PR1 layout (NEON `vmlal_s16` over sign×weight products; x86 `madd_epi16` / AVX-512 widening). Here the "digits" are extension-field eq weights (or their lifted `i16` limb patterns for base fields), not witness `i8` digits. Runtime arch dispatch like PR1.

**Deferred reduction:** inner face sums many `±W` into `ProductAccum`; reduce once per inner face; multiply by the outer `e_J_out · e_w_out` and add to the global accumulator.

#### Optimization investigation (what helps, what does not)

**Helps (spec adopts):**

1. **Fused verifier eval** — saves a `2^{k_w}` field-element write and a second pass; largest win for verifier memory bandwidth.
2. **Tensor split-eq + precomputed `W`** — `O(2^{m_Ji+m_wi})` setup, nested loops with `2√{2^{row_bits+k_w}}`-scale eq tables instead of materializing `eq` on the full product hypercube.
3. **`SIGNS_FOR_BYTE` + byte-wide scan** — correct memoization granularity; amortizes decode over four columns (already proven in PR1).
4. **Column panels / outer-tile parallelism** — witness read once in projection; here, eq-weight blocks and row chunks read once per tile; `parallel` over outer `(j_o, i_o)` when above a threshold.
5. **Branchless ternary via sign LUT** — `sign · W` with `sign ∈ {-1,0,+1}`; zeros cost a multiply-by-zero or masked add, not a unpredictable branch.

**Does not help enough (spec rejects as primary strategies):**

1. **Skipping zero entries with branches.** Zeros are pseudorandom (`P=1/2`); unpredictable branches hurt more than they save. Must still **read** every packed 2-bit pair to discover the sign. Sparsity is an **ALU** (~2×) win via LUT, not a memory-traffic win.
2. **Four-Russians / large pattern tables over `k` ternary columns.** Classic 4R amortizes `3^k` (or `4^k` packed-byte) partial-sum tables across many queries on the same block. The verifier runs **one** `eval_jl_mle_at` per proof; table build would dominate. Stage-1/stage-2 prefix LUTs (`STAGE1_B4_PREFIX_LOOKUP_TABLE`, etc.) work because the **digit alphabet is tiny and fixed** and the table is reused across millions of sumcheck rounds, not because ternary JL entries resemble binary digits. Do not build per-proof `3^k` tables over eq-weighted column blocks.
3. **Sparse / run-length `J` storage.** Would sacrifice the uniform dense XOF expansion and complicate matrix generation for uncertain gain on random data.
4. **Materializing full `g` on the verifier** then `Σ_i eq(r_w,i) g[i]` — correct but strictly worse than `eval_jl_mle_at` for memory.

**Benchmark later (optional, not spec blockers):** AVX-512 `vpcompress`-style masked accumulation for nonzero lanes; `vdotq_s32` on AArch64 when toolchain stabilizes; tuning split bit-widths (`4+4` vs `3+5` on rows) for cache.

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
- **Four-Russians tables over `k` ternary column blocks.** No cross-query amortization on the verifier's one-shot eval; per-block setup with eq-dependent weights dominates. Rejected; `SIGNS_FOR_BYTE` is the right static LUT size (256 bytes worth of signs per packed byte).
- **Branching on `J=0` to skip half the updates.** Pseudorandom zeros; mispredictions likely outweigh savings. Rejected; use branchless sign LUT like PR1.
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

### D4. The weak-binding price under JL (operator norm retained; realized-norm certificate)

**What.** Determine the A-role collision price under a JL norm certificate, and the soundness argument that licenses it. The conclusion is the LaBRADOR weak-binding argument: JL certifies the realized recursive-witness norm, but the operator-norm factor in the A-binding stays.

**The weak-binding collision (LaBRADOR Thm 5.1; akita `lem:batched-weak-binding`).** Two distinct weak openings of the same inner commitment collide. With `t_hat = t_hat'` (B-binding), `A(s_i - s'_i) = 0` for the block where they differ. The reduction outputs the kernel
`z_A = c_bar' (c_bar s_i) - c_bar (c_bar' s'_i)`, with `A z_A = 0` and
`||z_A||_2 <= 2 * Gamma_bar * beta_resp`,
where `beta_resp` bounds the response difference `||c_bar s_i||_2 = ||z^{(i)} - z^0||_2`. The operator-norm factor `Gamma_bar` enters because the kernel cross-multiplies the two openings' challenge differences; `beta_resp` is available per transcript from the `||z|| <= gamma` check, with no pinning required. This is the only non-circular A-binding.

**Why the clean `2 * T_s` (drop the operator norm) is circular and does not work.** The tempting argument: certify `||s_i||_2 <= T_s` for the extracted block, then bound the bare kernel `s_i - s'_i` (the `A(s-s')=0` identity is already there) by `2 T_s`, with no operator norm. The flaw: to JL-certify `||s_i||_2`, Lemma 4.2 requires `s_i` fixed **before** `J` is sampled. The extracted `s_i = (z^{(i)} - z^0)/c_bar` is recovered from transcripts whose fold challenges are sampled **after** `J`, so a priori it is a function of `J`. Pinning it to the pre-`J` inner commitment requires A-binding (uniqueness of the weak opening). If that A-binding is the very `2 T_s` we are proving, the argument assumes its conclusion. `prop:committed-fold-price` is the structural restatement: the extracted quotient need not equal any range-checked committed block, so the committed witness's pinning never transfers to it. Confirmation from LaBRADOR: even its JL term in the rank-`kappa` binding is `4 * T * sqrt(128/30) * beta` (Thm 5.1) -- operator norm `T` **times** the JL-certified norm, never the bare JL norm.

**What JL actually buys (the realized-norm certificate).** JL replaces the deterministic response envelope `beta_inf` (which `sqrt(d)*beta_inf` over-prices `||z||_2` by a calibrated 30-200x) with a realized bound, so the A-role becomes
`eta_A = 2 * Gamma_bar * beta_bar_2`
with `beta_bar_2` the JL-certified realized response, keeping the operator factor. This is the same realized-norm tightening the exact-l2 certificate targets; JL's advantage over exact-l2 is the absence of a no-wrap gate (it works at the root), not a smaller `eta_A`. The certified norm is also the recursion's next-level input bound `beta'` -- otherwise unbounded, since the extracted quotient has no norm bound without it. This is exactly LaBRADOR's order: the response-based binding pins the witness (operator factor retained), then JL certifies its realized norm for the next level.

**Project the committed blocks, not the fold response.** JL can certify only objects fixed before `J`. The committed witness blocks are pre-`J` (pinned by their commitment); the fold response `z` is post-fold-challenge, hence post-`J`, so it cannot be the projected object for a Lemma-4.2 argument. Reaching `||z||_2` from the certified block norm costs a second operator factor `Gamma_fold` (`||z||_2 <= Gamma_fold * ||s||_2`). Whether the A-binding then carries one or two operator factors depends on akita's CWSS extraction details (whether `beta_resp` is bounded by a separate direct `||z||` check or only through the block norm) and is an open item; the load-bearing correction is that it carries **at least one** -- the clean zero-factor `2 T_s` is unavailable.

**Global projection (no per-block, no union).** A single JL projection of the concatenated committed object (one `n_rows x N_coeff` matrix, one image `p`, one norm check) certifies one global Euclidean norm, exactly as LaBRADOR projects the whole folded witness once. There is **no** per-block projection and **no** union over blocks: `beta_resp` / the next-level input norm is a single global bound, and a single block's norm is at most the whole vector's. `n_rows` is sized by the single-projection lower tail (LaBRADOR's 256-row `2^{-128}` constant), not by a `#blocks` union (see D6).

**Why deferred.** It is the soundness argument, not code; the prototype's mechanics (project, reveal, check norm, consistency sumcheck) are identical regardless of the final operator-factor count.

**Open questions.** Whether akita's CWSS extraction yields one or two operator factors in `eta_A`; whether the response bound `beta_resp` is supplied by a retained direct `||z||` check or derived from the JL block-norm; the precise uniqueness-bootstrap order in the recursive-tree extraction (response-priced A-binding pins, then Lemma 4.2 certifies); how the signed-coordinate injectivity condition enters the JL-to-field consistency argument.

### D5. Completeness: nonce regrind and the norm-threshold policy

**What.** An honest witness occasionally projects to an over-threshold image (the JL window has an upper tail).
The backup handles this with a nonce-regrind loop: the prover searches a small nonce on a cloned transcript, only commits the accepted nonce, and the verifier absorbs that one nonce.
LaBRADOR's check-and-retry analysis gives a small constant in its setting; the Akita threshold and slack must be restated with the chosen `n_rows` and signed-coordinate window (single global projection, so no union model; see D6).
Fusion needs: a bounded regrind nonce in the proof/transcript, a schedule-fixed honest bucket `T_p` sized from calibrated RMS image norms, and a liveness cap (no-panic on exhaustion).

**Why deferred.** The standalone prototype can pick a single transcript draw and a generous `T_p` for tests; regrind is a completeness optimization for the wired path.

**Open questions.** Nonce search budget; how the regrind nonce binds in the descriptor/transcript; interaction with the per-draw 128-bit entropy floor.

### D6. Sizing `n_rows` for the single global projection (no union)

**What.** With a single global projection (D4), the JL failure probability is the **single-shot** lower tail of the modular-JL lemma, not a union over blocks. LaBRADOR's Lemma 4.2 gives `Pr_J[||J s mod q||_2 < sqrt(30) * b] <= 2^{-128}` at `n_rows = 256` for one vector `s in [+-q/2]^N` with `||s||_2 >= b <= q/125`. The `q/125` precondition is engineered for `q ~ 2^32` (fp32), so it transplants directly. Because the whole concatenated object is projected once and a single global Euclidean norm is certified, there is **no** `#objects` union and no `n_rows ~ 256 + 2*log2(#objects)` inflation -- that earlier framing was an artifact of a per-block design the spec no longer pursues.

**Why deferred.** The prototype uses a configurable `n_rows` (default 256) and a generous bound; the exact slack and constant feed D3/D4 pricing.

**Open questions.** Whether akita keeps `n_rows = 256` with LaBRADOR's `2^{-128}` constant or re-derives the lower-tail exponent at a different `n_rows` (LaBRADOR states the bound only for 256 rows -- the binding lower tail is GHL21 Cor 3.2, not proven for general `n_rows` there); the `sqrt(128/30) ~ 2.07` slack vs a tighter threshold with nonce regrind (D5); whether root-level JL is used at all (the algebraic range check has no JL statistical failure mode and the live dense-verifier cost is largest at the root).

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

### Suggested ordering for fusion

D6 and D4 (security: size `n_rows` for the single projection, settle the weak-binding price with the operator factor retained, and settle coordinate injectivity) settle pricing inputs; D3 (delete stage 1, reprice, regen tables) and D1 (descriptor) are the type/accounting changes; D2 (Step, payload, stage-2 fusion) is the protocol cutover; D5 (regrind) and D7 (ZK) are completeness/ZK work; D8 is a separate mid-level project.

## Documentation

- This spec is the primary design record for the prototype and the fusion roadmap.
- New crate-level module docs (`//!`) on `akita-challenges::jl`, `akita-challenges::jl::mle`, and `akita-prover::protocol::jl`.
- No public book or security-doc changes until fusion (the prototype changes no shipped behavior). When fusion lands, the security-model and norm-bounds pages need the JL paradigm-schedule and the weak-binding realized-norm pricing (operator factor retained).

## Execution

Suggested order for the prototype:

1. `akita-challenges::jl`: port the packed-ternary matrix, seed expansion, `project`, and norm helpers from the backup; re-host the seed on the current transcript; make `n_rows`/`cols` parameters; add unit tests (determinism, round-trip, projection-vs-reference, fp128, norm bound).
2. Add signed-coordinate encoding and checked norm helpers before any consistency sumcheck work. The field embedding of `p` must be injective for accepted coordinates.
3. `akita-challenges::jl::mle` (PR1b): `eval_jl_mle_at` + `build_jl_row_weights` with split-eq orchestration and SIMD ternary–eq tile kernels; differential tests vs naive reference; tail-geometry bench (`benches/jl_mle.rs`).
4. `akita-prover::protocol::jl` (PR2): `JlConsistencyInstance` (degree-2 product); wire PR1b evaluators into prove/verify; standalone harness with public witness data or an explicit `w_tilde(r)` hook; cross-field/cross-dim round-trip, soundness-direction, and malformed-input tests.
5. Lint and full test sweep (`parallel` on and off, if both feature combinations are supported by the touched crates).

Risks to resolve first:

- Confirm the witness flatten map `index(x, y)` matches `center_coefficients` and stage-2 `w(x, y)` layout before building `JlWeight`.
- Confirm the standalone verifier interface: it must not pretend to verify a hidden committed witness unless it has an external `w_tilde(r)` value from a real opening path.
- Confirm `CanonicalField` plus current base-field types provide enough information for centered conversion and signed-coordinate injectivity across fp32/fp64/fp128.
- Confirm accepted coordinate and norm bounds fit the selected integer type; reject overflow rather than saturating or silently reducing.

## References

- `labrador-backup:src/protocol/labrador/johnson_lindenstrauss.rs`: reusable dense reveal projection implementation ideas (packed ternary, SHAKE seed expansion, centering, projection, nonce regrind). Port contracts, not constants or overflow return types, without re-auditing them.
- `akita-algebra/src/eq_poly.rs` (`SplitEqEvals`), `akita-types/src/extension_opening_reduction.rs` (`tensor_column_partials_split_fold`): split-eq contraction template for PR1b MLE eval.
- Dao & Thaler, ePrint 2024/1210 (`DaoThaler_SplitEq_SumCheck_2024_1210.pdf` in paper index): nested eq tables and iterated sums; cited in `akita-algebra/src/split_eq.rs`.
- LaBRADOR (eprint 2023/1729): modular-JL lemma (Lemma 4.1/4.2), the 256-row single-projection setting, the weak-binding A-price (Thm 5.1, where the JL term is `4*T*sqrt(128/30)*beta` -- operator norm retained), and the check-and-retry threshold policy. The weak-binding argument is the template (D4): JL certifies the realized output norm; the operator factor in the A-binding is structural and does not drop. Akita's relation being linear in the blocks does **not** remove it (the circularity in D4).
- Grand Danois (eprint 2026/1196): structured-projection-in-relation and the nested lever (for the deferred mid-level variant); constants must be re-derived.
- `akita-types/src/sis/norm_bound.rs`: A/B/D-role collision pricing the realized-norm repricing (D3, D4) targets.
- `akita-types/src/instance_descriptor.rs`: the transcript preamble that must bind JL geometry (D1).
- `akita-types/src/schedule.rs`, `akita-types/src/proof/levels.rs`, `akita-prover/src/protocol/sumcheck/akita_stage2`: the `Step`, proof payload, and fused stage-2 surfaces fusion touches (D2).
- Profiling (for the fusion measurement): `AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 cargo run --release --example profile`.
