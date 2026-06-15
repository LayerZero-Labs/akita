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
A sumcheck checks projection consistency against the witness multilinear extension, but in the standalone prototype this is only a mechanics test unless the verifier is also given a trusted way to evaluate the witness at the final sumcheck point.
The wired protocol would replace the stage-1 infinity-norm range sumcheck at selected JL levels with an image-norm check plus a projection-consistency row.
This prototype does not make that replacement in the recursive flow.

The prototype is built as standalone, well-tested library code that is not wired into the recursive prove/verify flow.
The goal is to land reusable JL primitives, exercise the consistency-sumcheck mechanics across representative fields and ring dimensions, and document the fusion roadmap so later protocol-integration work can measure any proof-size or rank impact on the real flow.

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
The current protocol certifies the fold response `z`; the proposed JL direction wants, if the anchored extraction argument works, to certify the committed blocks `s_i` directly.
Those are not automatically equivalent for soundness, because the existing CWSS extraction naturally produces fold-response quotients.
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
3. **The clean A-role repricing is not established by this prototype.** Replacing `8*omega*beta_inf*nu` with `2*T_s` requires a proof that the extracted A-object is the committed block pinned before `J`, not the post-fold quotient. That is the "R-A" item below. Without it, only a weaker fallback price may be available. This is the gating soundness item and is deferred; the prototype does not touch SIS pricing.

### Reusable prior code

The retired `labrador-backup` branch contains a working dense reveal projection at `labrador-backup:src/protocol/labrador/johnson_lindenstrauss.rs`: a 256-row ternary matrix packed 2 bits per entry, deterministic per-row expansion from a 32-byte transcript seed via SHAKE, an integer projection path that centers base-field coefficients and rejects overflow, a nonce-regrind (retry-until-the-image-is-short) completeness loop, and a `collapse` (dot with a coefficient vector) helper.
The implementation techniques are reusable.
The transcript hosting, row count, coordinate encoding, overflow policy, and security constants must be reworked for current Akita rather than ported as-is.

## Intent

### Goal

Land a standalone, field-generic and ring-dimension-generic JL projection prototype that samples a dense ternary projection matrix from a transcript seed, projects a witness to an integer image, checks the image Euclidean norm over the integers, and exercises projection-consistency with one degree-2 sumcheck in the intended fused-row oracle form, without wiring it into the recursive prove/verify flow.

Concretely the prototype introduces:

- `akita-challenges::jl` (new module): the dense projection matrix `JlProjectionMatrix`, deterministic transcript-seeded expansion (`sample`), integer projection (`project`) over centered witness coefficients, signed-coordinate encoding checks, and Euclidean-norm helpers. Generic over `F: FieldCore + CanonicalField` and `const D: usize`.
- `akita-prover::protocol::jl` (new module, not called by the flow): the projection-consistency claim builder that folds the matrix rows by powers of a transcript challenge `rho` into the public folded weight `g(x) = sum_j rho^j * J_tilde(j, x)`, plus a prototype sumcheck harness for `<rho-powers, p> = sum_x g(x) * w_tilde(x)`. Since the standalone module is not connected to a commitment verifier, its verifier side must either take the witness as public test data or take an external trusted `w_tilde(r)` evaluation hook. Full cryptographic binding is deferred to fusion.
- Cross-field, cross-dimension tests that exercise representative non-degenerate `(field, ring dim)` combinations the workspace ships.

### Invariants

- **Determinism / replayability.** For a fixed transcript state and fixed `(n_rows, cols)`, `JlProjectionMatrix::sample` is a pure function of the transcript-derived seed; prover and verifier reconstruct the identical matrix. Protected by a determinism test (two independent transcripts in the same state produce equal matrices and equal projections).
- **Projection correctness.** `project(w)` equals the exact integer matrix-vector product `J * centered_coeffs(w)` with no modular reduction; the image lives over the integers (balanced representatives of `w`). Protected by a test against a naive reference projection and by the consistency sumcheck harness.
- **Coordinate injectivity.** Any coordinate that is later embedded into the field for consistency must be checked to lie in the chosen signed encoding window, for example `|p_j| < q/2`. This prevents modulo-`q` aliases from passing the field consistency check with a different integer norm. Protected by boundary tests around the signed encoding limit.
- **No overflow on the integer path.** Centering uses balanced representatives in `[-(q-1)/2, (q-1)/2]`; accumulation uses `i128` and returns an error if a coordinate or norm computation would overflow the supported integer type. Do not silently saturate. fp128 witnesses with centered magnitudes exceeding `i64` must be handled, but arbitrary fp128-sized dense sums may still be rejected if they exceed `i128`. Protected by fp128 large-coefficient and overflow-rejection tests.
- **Norm check is over the integers.** Shortness is accepted from `||p||_2^2 <= T_p^2` over the integers, never from a squared-sum identity modulo `q`. This avoids the exact-`l2` no-wrap gate, but it still relies on the coordinate-injectivity check above. Protected by a completeness test (honest witness passes under a generous prototype bound) and a soundness-direction test (an over-norm image is rejected).
- **Consistency claim has the intended fused-row form.** The public folded weight is `g(x) = sum_{j<n_rows} rho^j * J_tilde(j, x)` and the proved identity is `sum_j rho^j * p[j] = sum_x g(x) * w_tilde(x)`, a degree-2 product sumcheck. This is the intended stage-2 fusion shape, but final drop-in compatibility still has to be checked against the current `w(x, y)` layout and stage-2 verifier API. Protected by a prove/verify round-trip test and a Schwartz-Zippel soundness-direction test (a wrong image fails except with the usual `rho` polynomial-collision probability).
- **No-panic on verifier-reachable paths.** Every shape mismatch (matrix dimensions, image length, point dimension, coefficient overflow) returns `AkitaError`, never panics, matching the verifier no-panic contract because the consistency check is intended to become verifier-reachable. Protected by malformed-input tests.
- **No protocol-flow regression.** Nothing in `prove_batched` / `verify_batched` calls the new modules, so all existing prover/verifier/integration tests pass unchanged and no serialized proof type changes. Protected by the full existing test suite.

### Non-Goals

These are the deferred items; each is investigated in the "Deferred work" section so the follow-up is fully scoped.

- **No wiring into the recursive flow.** No new `Step` variant, no `Schedule`/planner change, no serialized proof-type change, no `prove_batched` / `verify_batched` change.
- **No structured / ring-granular committed-image (mid-level) projection.** Only the dense reveal projection is built.
- **No anchored-extraction soundness lemma and no SIS repricing.** The clean A-role price requires the extraction re-architecture; not needed to prototype mechanics.
- **No ZK masking of the revealed image.** The revealed image leaks `n_rows` linear functionals; the prototype targets non-ZK builds.
- **No exact `n_J` / `c(n_rows)` derivation.** `n_rows` (default 256) and the norm bound are configurable parameters.
- **No removal of the terminal cleartext base case.**

## Evaluation

### Acceptance Criteria

- [ ] `akita-challenges::jl` compiles and exposes `JlProjectionMatrix` with transcript-seeded `sample`, `project`, signed-coordinate validation, and checked norm helpers, generic over `F: FieldCore + CanonicalField` and `const D: usize`.
- [ ] Determinism test: two transcripts in identical state yield byte-identical matrices and equal projections.
- [ ] Projection-vs-reference test: `project` matches a naive integer reference for random witnesses across fields and dims.
- [ ] fp128 large-coefficient test: centered magnitudes exceeding `i64` are handled without panic, and coordinates or norm sums exceeding the supported integer range are rejected.
- [ ] Signed-coordinate tests: accepted coordinates embed injectively into the base field, and boundary aliases are rejected.
- [ ] `akita-prover::protocol::jl` consistency prove/verify round-trips for honest `(w, p)` across representative non-degenerate `(field, ring dim)` combinations, using public test witness data or an explicit `w_tilde(r)` evaluation hook.
- [ ] Soundness-direction tests: an image inconsistent with `w` is rejected by the consistency sumcheck for all but a negligible fraction of `rho`; an over-norm image is rejected by the norm check.
- [ ] Malformed-input tests: wrong matrix shape, wrong image length, wrong point dimension all return `AkitaError`, never panic.
- [ ] All pre-existing workspace tests pass unchanged.
- [ ] `cargo fmt -q`, `cargo clippy --all -- -D warnings`, and the relevant test passes are green.

### Testing Strategy

New tests live alongside the new modules.

- `akita-challenges`: unit tests for `sample` determinism, packed-matrix round-trip (`00 -> -1`, `01/10 -> 0`, `11 -> +1`), `project` correctness vs reference, signed-coordinate injectivity, checked integer norm computation, and the fp128 large-coefficient path. Port the analogous implementation tests from `labrador-backup:src/protocol/labrador/johnson_lindenstrauss.rs`, but do not inherit its fixed row count or overflow-return type.
- `akita-prover`: a `protocol::jl` test module that builds a random witness, samples `J` and `rho`, computes `p`, and round-trips the consistency sumcheck through the real `akita-sumcheck` driver and the transcript. Since this is standalone, the test verifier supplies the witness evaluations directly or through an explicit final-evaluation hook. The sweep covers representative non-degenerate `(field, ring dim)` pairs: fp32, fp64, fp128 base fields and the supported `D` values used by shipped configs. Include the soundness-direction and malformed-input tests.

Feature combinations: tests pass with and without `parallel` if both feature sets compile in the touched crates. ZK is out of scope (the reveal leaks), so any reveal-path test that assumes public image data is gated to non-`zk` builds or clearly marked as a non-ZK mechanics test.

Existing tests that must keep passing: the entire `akita-pcs` / `akita-prover` / `akita-verifier` suite, unchanged, since the flow is untouched.

### Performance

Standalone prototype: no proof-size or prover-time effect on shipped paths (nothing is wired).

The measurement this enables (in the fusion follow-up) is whether replacing one tail level's stage-1 work with a JL reveal (delete stage 1, send `n_rows` signed integer coordinates plus the consistency row) reduces total bytes under the then-current tail encoding.
Prior analysis suggests the basis-unlock argument is not a byte win; any byte claim should therefore be tied to deleting stage-1 work, changing the terminal payload, or repricing SIS ranks after the extraction lemma is settled.
Cost shape: the prover projection is `O(n_rows * N_coeff)` signed additions; the consistency sumcheck is one degree-2 instance over `log2(N_coeff)` rounds; the verifier-side evaluation of the dense folded weight has no closed-form shortcut for a random matrix and costs `O(n_rows * cols)` field operations if done live. That is affordable only at small tail levels and is why this prototype is tail-focused.

## Design

### Architecture

The prototype splits along the existing crate dependency graph.

**`akita-challenges::jl` (matrix sampling, projection, norm).**
This crate owns Fiat-Shamir challenge sampling and depends only on `akita-field` and `akita-transcript`, the right layer for the projection primitives (field-level math, transcript-seeded, no sumcheck).
It already exposes a SHAKE256-backed streaming `XofCursor` (with bias-free draws and a `next_sign` helper) and the transcript seed-derivation pattern used by the sparse-challenge sampler, which the JL module reuses.

- `JlProjectionMatrix { n_rows, cols, row_bytes, packed_rows: Vec<Vec<u8>> }`: a dense ternary matrix with entries `{-1, 0, +1}` packed 2 bits per entry (`00 -> -1`, `01/10 -> 0`, `11 -> +1`), drawn from the transcript-derived XOF stream.
- `sample<F, T>(transcript, n_rows, cols) -> Result<Self, AkitaError>`: absorbs a context buffer (label, `n_rows`, `cols`, and a version/domain tag), draws a 32-byte seed, and expands rows deterministically. Per-row derivation, as in the backup, keeps generation parallel-safe; a single streaming `XofCursor` is also acceptable if tests lock the exact transcript behavior and no parallel row generation is required. `n_rows` and `cols` are parameters (default test `n_rows = 256`), not hardcoded.
- `project<const D: usize>(&self, witness: &[CyclotomicRing<F, D>]) -> Result<JlImage, AkitaError>`: centers each coefficient to its balanced representative and computes the exact integer product using `i128` coordinates. The modulus can be recovered for `CanonicalField` as `(-F::one()).to_canonical_u128() + 1`; if a clearer modulus helper is added later, use it instead. Return an error on coordinate overflow or shape mismatch.
- `JlImage` (or an equivalent explicit type) stores signed integer coordinates, exposes checked embedding into `F` only when the configured signed window is injective, and exposes checked norm helpers such as `l2_norm_sq_checked` / `check_l2`.

The matrix sampling and projection geometry (`n_rows`, seed domain) are intended to be bound into the instance descriptor when fusion lands; the prototype binds them through the transcript context buffer only and records descriptor binding as a fusion task.

**`akita-prover::protocol::jl` (consistency claim, prototype prove/verify).**
`akita-prover` depends on `akita-challenges`, `akita-sumcheck`, `akita-witness`, `akita-transcript`, and `akita-algebra`, so it is the natural home for the consistency sumcheck and the place fusion will happen. The module is not referenced by `flow.rs` or any prove/verify entry point.

- `fold_matrix_rows(matrix, rho) -> Vec<L>`: builds the public folded weight column vector `g_i = sum_{j<n_rows} rho^j * J[j, i]` over the cols (`cols = N_coeff`). This is the coefficient table of `g(x)`; its multilinear extension `g_tilde` is what the verifier evaluates.
- The proved identity is `sum_j rho^j * p[j] = sum_x g(x) * w_tilde(x)`, equivalently `<rho-powers, p> = <g, coeffs(w)>`. Implement it as a degree-2 product-of-two-multilinears sumcheck via the existing `akita-sumcheck` `SumcheckInstanceProver` / `SumcheckInstanceVerifier` driver (a small `JlConsistencyInstance` implementing those traits): oracle `g_tilde(x) * w_tilde(x)`, input claim `sum_j rho^j p[j]`.
- The verifier cannot check this identity from `g` and `p` alone. Its `expected_output_claim` also needs `w_tilde(r)` at the sumcheck challenge point. The standalone prototype should make that dependency explicit: tests may pass the witness table directly, while the fusion path will get the value from the normal commitment/opening machinery.
- `prove_jl_consistency` / `verify_jl_consistency`: a thin prototype harness that absorbs the image, samples `rho` from the transcript (respecting wire-before-squeeze), checks coordinate injectivity and the integer norm bound, then runs the sumcheck. These mirror the eventual fused-row insertion point without claiming standalone PCS soundness.

**Why the consistency is in fused-row form now.**
The current stage-2 fused sumcheck batches three terms by powers of a challenge `gamma`: `gamma^0 * relation + gamma^1 * range + gamma^2 * trace`, all sharing one witness scan over the multilinear table `w(x, y)`.
Building `g(x) = sum_j rho^j J_tilde(j, x)` and proving `<rho-powers, p> = sum_x g(x) w_tilde(x)` is the intended shape that, in fusion, becomes a new `gamma^k * omega_JL` addend in that batch (with the stage-1 `gamma^1 * range` term deleted at JL levels).
The exact `gamma` power, the `w(x, y)` column layout, and the final-point evaluation path remain D2 work.
Prototyping the row in this shape reduces throwaway work, but does not by itself prove it drops into the current stage-2 code unchanged.

**Field and ring-dimension genericity.**
The projection acts on integer coefficients of base-field ring elements, so it should be field-generic across base fields that implement `CanonicalField`.
Centering uses the canonical representative in `[0, q)` and the balanced interval around zero.
The matrix and projection are generic over `const D`.
The consistency sumcheck operates over the claim/extension field `L` for the witness multilinear extension in the same style as stage 2, but the prototype must explicitly bridge base-field signed image coordinates into `L`.
Tests sweep representative fp32/fp64/fp128 and supported `D` values, excluding degenerate configs that do not exercise a recursive witness.

### Alternatives Considered

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
3. `akita-prover/src/protocol/sumcheck/akita_stage2`: add the `omega_JL` addend `gamma^k * g_tilde(x) * w_tilde(x)` to the fused batch and delete the `gamma^1 * range` term at JL levels. The verifier evaluates `g_tilde` at the stage-2 final point. `rho` is sampled after `u'` binds `w_next` (wire-before-squeeze); `gamma` already exists.

**Why deferred.** This is the actual protocol cutover and touches serialized types, the planner, and both prover and verifier; the prototype is explicitly standalone.

**Open questions.** The exact `gamma` power for the JL row; image serialization (signed-integer codec vs injective field encoding, which also feeds the tail entropy-coding work); whether intermediate JL levels (birth-certify `w_next`, then delete the next level's stage 1) and terminal-adjacent JL levels need distinct payloads; where the verifier obtains `w_tilde(r)` for the JL row in each case.

### D3. Stage-1 deletion and SIS repricing

**What.** Deleting stage 1 removes the degree-`2^lb` range tree, the carried `s_claim`, and the stage-1 transcript at JL levels.
It also reprices the SIS roles in `akita-types/src/sis/norm_bound.rs`:

- A-role (`rounded_up_collision_norm_s`) today is `committed_fold_collision_l2_sq` with `collision_linf = 8 * omega * beta_inf * nu`. Under anchored extraction this becomes `2 * T_s` (clean, requires D4) or a fallback (also D4). This is the main candidate rank lever.
- B-role (`rounded_up_collision_norm_t`, the `t_hat` opening digits) and D-role (`rounded_up_collision_norm_w`) are digit-range collisions `2^lb - 1`; they are unchanged by JL but must be re-audited once stage 1 no longer certifies "digits are range-checked" (a comment in `norm_bound.rs` and several consumers assume this).
- The revealed image norm bound `T_p` and the coordinate-injectivity window must be added to pricing/sizing.

**Why deferred.** Repricing is only meaningful with the anchored-extraction lemma (D4) decided, and it regenerates the SIS floor tables and planner schedules.

**Open questions.** Which `norm_bound.rs` consumers implicitly rely on the stage-1 range guarantee (audit needed); the exact `T_p` to bucket; table regen scope.

### D4. The anchored-extraction lemma (the gating soundness item)

**What.** The clean A-role price `eta_A = 2 * T_s` holds only if the object the CWSS extractor recovers is the witness block `s_i` that was pinned in `u_l` before `J` was sampled.
The conversion from image bound `T_p` to source bound `T_s` depends on the final Akita JL lower-tail constant (for example a `sqrt(30)` denominator in the LaBRADOR-style 256-row statement), and must be re-derived with D6.
The natural CWSS extractor instead produces a fold-response quotient `s_i^ext = (z-difference) / c_i`, and division by the ring unit `c_i` is not norm-preserving, so a bound on the image of the quotient does not transfer to `||s_i^ext||`.
Two resolutions:

- **R-A (clean).** Re-architect the projection consistency to bind the committed blocks directly (`p_i = J s_i` against `w_l`'s opening), not the post-fold response, so the extracted A-object is the pinned block. If proved, this delivers `eta_A = 2 T_s`. This is the gating write-up item.
- **R-B (fallback).** Keep the collision on the fold quotient; JL may still tighten the fold-response bound to a realized-style `||z||_2 <= Gamma_fold * T_s`, giving a price of the form `eta_A = 2 * Gamma_bar * 2 * Gamma_fold * T_s`. This still carries challenge-mass factors and must be checked against the actual CWSS extraction and SIS tables before being advertised as a win.

The pinning order itself is not circular: the blocks `s_i` are committed in `u_l` before `J` is sampled.
What remains unproved is that the consistency row and extraction route make the pinned block, rather than a post-fold quotient or an adaptively switched opening, the object whose norm enters the A-role collision.

**Why deferred.** It is a soundness proof, not code; it decides clean vs fallback pricing for D3; the prototype's mechanics are independent of which resolution is chosen.

**Open questions.** Whether R-A's "consistency against the committed blocks, not the fold response" is compatible with akita's fused stage-2 trace structure without an extra commitment; the precise statement of the uniqueness-bootstrap order in the recursive-tree extraction; how the signed-coordinate injectivity condition enters the JL-to-field consistency argument.

### D5. Completeness: nonce regrind and the norm-threshold policy

**What.** An honest witness occasionally projects to an over-threshold image (the JL window has an upper tail).
The backup handles this with a nonce-regrind loop: the prover searches a small nonce on a cloned transcript, only commits the accepted nonce, and the verifier absorbs that one nonce.
LaBRADOR's check-and-retry analysis gives a small constant in its setting; the Akita threshold and slack must be restated with the chosen `n_rows`, signed-coordinate window, and union-bound model.
Fusion needs: a bounded regrind nonce in the proof/transcript, a schedule-fixed honest bucket `T_p` sized from calibrated RMS image norms, and a liveness cap (no-panic on exhaustion).

**Why deferred.** The standalone prototype can pick a single transcript draw and a generous `T_p` for tests; regrind is a completeness optimization for the wired path.

**Open questions.** Nonce search budget; how the regrind nonce binds in the descriptor/transcript; interaction with the per-draw 128-bit entropy floor.

### D6. The union bound and sizing `n_rows` (`n_J`) per level

**What.** The JL failure probability is a union over the projected objects: `kappa_JL = (#objects) * 2^{-c(n_rows)}`, with the 256-row LaBRADOR statement often summarized as about 128 bits for its exact setting.
The local resolution note suggests that, under Akita's current convention of targeting a per-level `2^{-128}` term and leaving Fiat-Shamir query count symbolic, a linear extrapolation would give `n_rows ~ 256 + 2*log2(#objects)`.
That extrapolation is not yet a theorem.
At tail levels `#objects` is expected to be small, so `n_rows` near 256 may be enough; root-level JL remains disfavored because both the union and live dense-verifier cost are largest there, and because the algebraic range check has no JL statistical failure mode.
The constant `c(n_rows)`, the exact object count, and any Fiat-Shamir grinding multiplier must be re-derived explicitly.

**Why deferred.** The prototype uses a configurable `n_rows` (default 256) and a generous bound; exact sizing is a security-accounting task that feeds D3 pricing.

**Open questions.** The exact `c(n_rows)`; whether the union is over blocks, coefficient slices, whole revealed images, or extraction branches in the final proof; whether to size per level from that count or fix one conservative `n_rows` for all JL levels.

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

D6 and D4 (security: size `n_rows`, prove the extraction route, and settle coordinate injectivity) settle pricing inputs; D3 (delete stage 1, reprice, regen tables) and D1 (descriptor) are the type/accounting changes; D2 (Step, payload, stage-2 fusion) is the protocol cutover; D5 (regrind) and D7 (ZK) are completeness/ZK work; D8 is a separate mid-level project.

## Documentation

- This spec is the primary design record for the prototype and the fusion roadmap.
- New crate-level module docs (`//!`) on `akita-challenges::jl` and `akita-prover::protocol::jl`.
- No public book or security-doc changes until fusion (the prototype changes no shipped behavior). When fusion lands, the security-model and norm-bounds pages need the JL paradigm-schedule and the anchored-extraction pricing.

## Execution

Suggested order for the prototype:

1. `akita-challenges::jl`: port the packed-ternary matrix, seed expansion, `project`, and norm helpers from the backup; re-host the seed on the current transcript; make `n_rows`/`cols` parameters; add unit tests (determinism, round-trip, projection-vs-reference, fp128, norm bound).
2. Add signed-coordinate encoding and checked norm helpers before any consistency sumcheck work. The field embedding of `p` must be injective for accepted coordinates.
3. `akita-prover::protocol::jl`: `fold_matrix_rows` plus the degree-2 product consistency sumcheck instance (a small `JlConsistencyInstance` implementing the `akita-sumcheck` prover/verifier traits); standalone `prove_jl_consistency` / `verify_jl_consistency` harness with public witness data or an explicit final-evaluation hook; cross-field/cross-dim round-trip, soundness-direction, and malformed-input tests.
4. Lint and full test sweep (`parallel` on and off, if both feature combinations are supported by the touched crates).

Risks to resolve first:

- Confirm the witness multilinear-extension vocabulary (`akita-witness::PolynomialView`) and the `akita-sumcheck` driver entry are ergonomic for a two-multilinear product oracle; if the driver expects a specific instance trait, implement the small `JlConsistencyInstance` rather than forcing a generic path.
- Confirm the standalone verifier interface: it must not pretend to verify a hidden committed witness unless it has an external `w_tilde(r)` value from a real opening path.
- Confirm `CanonicalField` plus current base-field types provide enough information for centered conversion and signed-coordinate injectivity across fp32/fp64/fp128.
- Confirm accepted coordinate and norm bounds fit the selected integer type; reject overflow rather than saturating or silently reducing.

## References

- `labrador-backup:src/protocol/labrador/johnson_lindenstrauss.rs`: reusable dense reveal projection implementation ideas (packed ternary, SHAKE seed expansion, centering, projection, nonce regrind). Port contracts, not constants or overflow return types, without re-auditing them.
- LaBRADOR (eprint 2023/1729): modular-JL lemma, the 256-row setting, and the check-and-retry threshold policy. Re-derive Akita's `c(n_rows)`, slack, union bound, and coordinate-injectivity conditions before using them for soundness.
- Grand Danois (eprint 2026/1196): structured-projection-in-relation and the nested lever (for the deferred mid-level variant); constants must be re-derived.
- `akita-types/src/sis/norm_bound.rs`: A/B/D-role collision pricing the anchored repricing (D3, D4) targets.
- `akita-types/src/instance_descriptor.rs`: the transcript preamble that must bind JL geometry (D1).
- `akita-types/src/schedule.rs`, `akita-types/src/proof/levels.rs`, `akita-prover/src/protocol/sumcheck/akita_stage2`: the `Step`, proof payload, and fused stage-2 surfaces fusion touches (D2).
- Profiling (for the fusion measurement): `AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 cargo run --release --example profile`.
