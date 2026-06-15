# Spec: Unstructured JL Projection Prototype (tail reveal, standalone)

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao, Cursor agent (model: Claude Opus 4.8) |
| Created     | 2026-06-15                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

Akita is a lattice-based recursive polynomial commitment scheme.
Each recursive level commits a short witness, folds it against a challenge, and proves the fold response is short so that an extractor can recover a norm-bounded opening.
Today shortness is proved by a stage-1 infinity-norm range-check sumcheck on the decomposed fold response, and the recursion ends by sending the final folded witness in cleartext (the terminal direct step).
That terminal cleartext witness dominates the proof: it is roughly 60 to 80 percent of total bytes at every problem size that folds (`num_vars <= 30`), and every intermediate level pays for its stage-1 range tree on top.

This spec defines a minimal, self-contained prototype of an alternative shortness proof for the tail levels: an unstructured (dense, field-granular) Johnson-Lindenstrauss random projection.
The verifier samples a dense ternary projection matrix from the Fiat-Shamir transcript; the prover projects the witness to a small integer image `p`, reveals `p`, and the verifier accepts shortness from a Euclidean norm check on `p` computed over the integers.
A single sumcheck proves the projection was computed honestly against the witness multilinear extension.
This replaces the infinity-norm range sumcheck with JL projection checks only (a norm check on the revealed image plus a projection-consistency sumcheck); there is no infinity-norm sumcheck.

The prototype is built as standalone, well-tested library code that is not wired into the recursive prove/verify flow.
The goal is to land the reusable JL primitives, prove out the consistency-sumcheck mechanics across every field and ring dimension, and document a concrete fusion roadmap, ahead of the protocol-integration work where the real proof-size win is realized.

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

Soundness needs the extractor to recover a witness whose Euclidean norm is bounded, which requires a certificate that the fold response `z` (equivalently the blocks `s_i`) is short.
Today that certificate is **stage 1**: the balanced base-`2^lb` digits of `z` are committed and a range-check sumcheck of degree `2^lb` proves each digit lies in the digit set, giving an infinity-norm bound `||z||_inf <= beta_inf`, converted to Euclidean via `||z||_2 <= sqrt(d) * beta_inf`.
The deterministic envelope `beta_inf = num_claims * 2^r_vars * min(||c||_inf*||s||_1, ||c||_1*||s||_inf)` is what the SIS accounting prices today (see the A-role collision below).
Calibration data shows this deterministic envelope is 30 to 200 times larger than the realized `||z||_2`, so the A-role SIS rank is paying for slack it does not need.

### The Johnson-Lindenstrauss alternative

A random projection certifies Euclidean shortness directly.
Fix a vector `s` before sampling a random matrix `J` with entries in `{-1, 0, +1}` and `n_rows` rows.
With high probability over `J`, the image norm `||J s||_2` is within a known window of `sqrt(n_rows/2) * ||s||_2`; concretely the modular-JL lemma (LaBRADOR, Lemma 2.5 / 4.2) gives a lower-tail bound so that observing a small `||J s||` certifies a small `||s||`.
The verifier therefore samples `J` from the transcript (so it is independent of the pinned `s`), the prover sends the short image `p = J s`, and the verifier checks `||p||_2` over the integers.
The certified slack is roughly `sqrt(337/30) ~ 3.35` for a single projection (LaBRADOR window), versus the 30 to 200 times slack of the deterministic infinity-norm envelope.

Because the image is revealed and checked over the integers (balanced representatives), there is **no modular wraparound** to gate: the verifier sums squares over `Z` directly.
This is the key small-field advantage over an exact mod-`q` Euclidean identity, which would need `q ~ 2^50` to avoid wrap and is infeasible on the fp32 prime.

### Why dense and field-granular is correct at the tail (the fold-commutation law)

A projection placed inside the recursion must commute with the fold to be checkable through the fold, i.e. `J * coeff(sum_i c_i s_i) = sum_i Rot(c_i) * J * coeff(s_i)` must hold.
This forces a structured, ring-granular `J = J_0 (x) I_D` (entries that are constant polynomials) whenever the projected image is committed and then folded again.

At a **tail** level the projection is the last thing that happens to that witness: nothing is folded after it, so the commutation constraint does not apply.
The matrix can therefore be a plain dense field-granular `{-1, 0, +1}` matrix.
This is the "unstructured" projection this spec prototypes, and it is the simplest variant: dense matrix, reveal the image, check the norm over `Z`, prove consistency with one sumcheck.
(The structured ring-granular committed-image variant for mid levels is explicitly out of scope.)

### The three corrections that frame the value

Prior internal analysis flagged three things that constrain how this prototype is framed:

1. **JL at the tail does not replace the terminal cleartext send.** The terminal direct opening is the PCS base case: something must still verify the final evaluation claim. The reveal projection deletes stage 1 at a tail level and can shrink the last one or two committed levels (replace a full-witness reveal with a small image plus birth-certification of `w_next`), but it does not eliminate the terminal witness. Frame this as "delete stage 1 plus reveal a small image," not "shrink the terminal 3-4x."
2. **JL is not a decomposition-basis byte win.** Lifting the decomposition basis was hypothesized to shrink the tail under JL; a planner sweep showed total proof size changes by at most 1 percent at small sizes and 0 percent at `num_vars >= 28`, because the digit-packing identity `delta * lb ~ field_bits` magnitude-locks every cleartext segment while the module rank `n_a(lb)` grows. So the byte case for JL rests on deleting stage 1 and the no-fallback simplicity, not on a basis lever.
3. **The clean A-role repricing needs an extraction re-architecture.** The big rank win (replace `8*omega*beta_inf*nu` with `2*T_s`) is sound but requires routing the projection consistency against the committed blocks rather than the post-fold response (the "R-A" item below). Without it, a weaker but still real fallback price applies. This is the gating soundness item and is deferred; the prototype does not touch SIS pricing.

### Reusable prior code

The retired `labrador-backup` branch contains a working dense reveal projection at `labrador-backup:src/protocol/labrador/johnson_lindenstrauss.rs`: a 256-row ternary matrix packed 2 bits per entry, deterministic per-row expansion from a 32-byte transcript seed via SHAKE, an integer `project` with i64 and i128 streaming paths and overflow-checked accumulation, a nonce-regrind (retry-until-the-image-is-short) completeness loop, and a `collapse` (dot with a coefficient vector) helper.
The math ports directly; the seeding must be re-hosted on the current spongefish-backed transcript, and the fixed 256 rows generalized to a parameter.

## Intent

### Goal

Land a standalone, field-generic and ring-dimension-generic JL projection prototype that samples a dense ternary projection matrix from a transcript seed, projects a witness to an integer image, checks the image Euclidean norm over the integers, and proves projection-consistency with one degree-2 sumcheck in the fused-row oracle form, without wiring it into the recursive prove/verify flow.

Concretely the prototype introduces:

- `akita-challenges::jl` (new module): the dense projection matrix `JlProjectionMatrix`, deterministic transcript-seeded expansion (`sample`), integer projection (`project`) over centered witness coefficients, and Euclidean-norm helpers. Generic over `F: FieldCore + CanonicalField` and `const D: usize`.
- `akita-prover::protocol::jl` (new module, not called by the flow): the projection-consistency claim builder that folds the matrix rows by powers of a transcript challenge `rho` into the public folded weight `g(x) = sum_j rho^j * J_tilde(j, x)`, plus a prototype prove/verify pair built on the existing `akita-sumcheck` driver that proves `<rho-powers, p> = sum_x g(x) * w_tilde(x)`. This is the same degree-2 product-of-two-multilinears row that fusion will add to the stage-2 batch, structured so it folds in unchanged.
- Cross-field, cross-dimension tests that exercise every non-degenerate `(field, ring dim)` combination the workspace ships.

### Invariants

- **Determinism / replayability.** For a fixed transcript state and fixed `(n_rows, cols)`, `JlProjectionMatrix::sample` is a pure function of the transcript-derived seed; prover and verifier reconstruct the identical matrix. Protected by a determinism test (two independent transcripts in the same state produce equal matrices and equal projections).
- **Projection correctness.** `project(w)` equals the exact integer matrix-vector product `J * centered_coeffs(w)` with no modular reduction; the image lives over the integers (balanced representatives of `w`). Protected by a test against a naive reference projection and by the consistency sumcheck.
- **No overflow on the integer path.** Centering uses balanced representatives in `[-(q-1)/2, (q-1)/2]`; accumulation uses `i128` with checked fallback when the per-row magnitude bound can exceed `i128`. fp128 witnesses must be handled (centered magnitudes can exceed `i64`). Protected by an fp128 large-coefficient test.
- **Norm check is over the integers, wrap-free.** Shortness is accepted from `||p||_2^2 <= T_p^2` over the integers, never modulo `q`; there is no no-wrap gate. Protected by a completeness test (honest witness passes) and a soundness-direction test (an over-norm image is rejected).
- **Consistency claim equals the fused-row form.** The public folded weight is exactly `g(x) = sum_{j<n_rows} rho^j * J_tilde(j, x)` and the proved identity is `sum_j rho^j * p[j] = sum_x g(x) * w_tilde(x)`, a degree-2 product sumcheck. This matches the stage-2 fusion target so the row drops in unchanged. Protected by a prove/verify round-trip test and a Schwartz-Zippel soundness-direction test (a wrong image fails for all but a negligible fraction of `rho`).
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

- [ ] `akita-challenges::jl` compiles and exposes `JlProjectionMatrix` with transcript-seeded `sample`, `project`, and an `l2_norm_sq` helper, generic over `F: FieldCore + CanonicalField` and `const D: usize`.
- [ ] Determinism test: two transcripts in identical state yield byte-identical matrices and equal projections.
- [ ] Projection-vs-reference test: `project` matches a naive integer reference for random witnesses across fields and dims.
- [ ] fp128 large-coefficient test: centered magnitudes exceeding `i64` are handled without overflow or panic.
- [ ] `akita-prover::protocol::jl` consistency prove/verify round-trips for honest `(w, p)` across every non-degenerate `(field, ring dim)` combination.
- [ ] Soundness-direction tests: an image inconsistent with `w` is rejected by the consistency sumcheck for all but a negligible fraction of `rho`; an over-norm image is rejected by the norm check.
- [ ] Malformed-input tests: wrong matrix shape, wrong image length, wrong point dimension all return `AkitaError`, never panic.
- [ ] All pre-existing workspace tests pass unchanged.
- [ ] `cargo fmt -q`, `cargo clippy --all -- -D warnings`, and the relevant test passes are green.

### Testing Strategy

New tests live alongside the new modules.

- `akita-challenges`: unit tests for `sample` determinism, packed-matrix round-trip, `project` correctness vs reference, integer norm computation, and the fp128 large-coefficient path. Port the analogous tests from `labrador-backup:src/protocol/labrador/johnson_lindenstrauss.rs`.
- `akita-prover`: a `protocol::jl` test module that builds a random witness, samples `J` and `rho`, computes `p`, and round-trips the consistency sumcheck through the real `akita-sumcheck` driver and the transcript. The sweep covers every non-degenerate `(field, ring dim)` pair: fp32, fp64, fp128 base fields and `D in {32, 64, 128, 256}` as supported by each field. Include the soundness-direction and malformed-input tests.

Feature combinations: tests pass with and without `parallel`. ZK is out of scope (the reveal is non-ZK), so the prototype's reveal-path tests are gated to non-`zk` builds.

Existing tests that must keep passing: the entire `akita-pcs` / `akita-prover` / `akita-verifier` suite, unchanged, since the flow is untouched.

### Performance

Standalone prototype: no proof-size or prover-time effect on shipped paths (nothing is wired).

The measurement this enables (in the fusion follow-up) is whether replacing one tail level's stage-1 plus full-witness reveal with a JL reveal (delete stage 1, send `n_rows` integer coordinates plus the consistency row) reduces total bytes.
Prior analysis predicts deleting roughly 2.4 KB/level of stage-1 at `lb=5` for roughly 1 KB of revealed image, and that it is not a basis-unlock win.
Cost shape: the projection is `O(n_rows * N_coeff)` sign-additions (zero multiplications); the consistency sumcheck is one degree-2 instance over `log2(N_coeff)` rounds; the verifier evaluates the dense folded weight `g` at the challenge point at `O(n_rows * cols)` additions (no closed-form multilinear extension for a random matrix), which is affordable only at small tail levels and is exactly why this is a tail-only lever.

## Design

### Architecture

The prototype splits along the existing crate dependency graph.

**`akita-challenges::jl` (matrix sampling, projection, norm).**
This crate owns Fiat-Shamir challenge sampling and depends only on `akita-field` and `akita-transcript`, the right layer for the projection primitives (field-level math, transcript-seeded, no sumcheck).
It already exposes a SHAKE256-backed streaming `XofCursor` (with bias-free draws and a `next_sign` helper) and the transcript seed-derivation pattern used by the sparse-challenge sampler, which the JL module reuses.

- `JlProjectionMatrix { n_rows, cols, row_bytes, packed_rows: Vec<Vec<u8>> }`: a dense ternary matrix with entries `{-1, 0, +1}` packed 2 bits per entry (`00 -> -1`, `01 -> 0`, `11 -> +1`), drawn from the transcript-derived XOF stream.
- `sample<F, T>(transcript, n_rows, cols) -> Result<Self, AkitaError>`: absorbs a context buffer (label, `n_rows`, `cols`), draws a 32-byte seed, and expands per-row via the `XofCursor` PRG. Per-row derivation keeps generation parallel-safe. `n_rows` and `cols` are parameters (default test `n_rows = 256`), not hardcoded.
- `project<const D: usize>(&self, witness: &[CyclotomicRing<F, D>]) -> Result<Vec<i64>, AkitaError>`: centers each coefficient to its balanced representative, then computes the exact integer product, with an i64 fast path, an i128 path for fp128, and checked accumulation when the per-row magnitude bound can exceed i128 (ported from the backup's `project_streaming`).
- `l2_norm_sq(image: &[i64]) -> u128` and `check_l2(image, bound_sq) -> bool`: integer Euclidean norm and threshold check, wrap-free.

The matrix sampling and projection geometry (`n_rows`, seed domain) are intended to be bound into the instance descriptor when fusion lands; the prototype binds them through the transcript context buffer only and records descriptor binding as a fusion task.

**`akita-prover::protocol::jl` (consistency claim, prototype prove/verify).**
`akita-prover` depends on `akita-challenges`, `akita-sumcheck`, `akita-witness`, `akita-transcript`, and `akita-algebra`, so it is the natural home for the consistency sumcheck and the place fusion will happen. The module is not referenced by `flow.rs` or any prove/verify entry point.

- `fold_matrix_rows(matrix, rho) -> Vec<L>`: builds the public folded weight column vector `g_i = sum_{j<n_rows} rho^j * J[j, i]` over the cols (`cols = N_coeff`). This is the coefficient table of `g(x)`; its multilinear extension `g_tilde` is what the verifier evaluates.
- The proved identity is `sum_j rho^j * p[j] = sum_x g(x) * w_tilde(x)`, equivalently `<rho-powers, p> = <g, coeffs(w)>`. Implemented as a degree-2 product-of-two-multilinears sumcheck via the existing `akita-sumcheck` `SumcheckInstanceProver` / `SumcheckInstanceVerifier` driver (a small `JlConsistencyInstance` implementing those traits): oracle `g_tilde(x) * w_tilde(x)`, input claim `sum_j rho^j p[j]`. The verifier's `expected_output_claim` evaluates `g_tilde` (dense, `O(n_rows * cols)` additions) and `w_tilde` at the challenge point and multiplies.
- `prove_jl_consistency` / `verify_jl_consistency`: a thin prototype harness that samples `rho` from the transcript after the image is absorbed (respecting wire-before-squeeze), runs the sumcheck, and checks the norm. These mirror the eventual fused-row insertion point.

**Why the consistency is in fused-row form now.**
The current stage-2 fused sumcheck batches three terms by powers of a challenge `gamma`: `gamma^0 * relation + gamma^1 * range + gamma^2 * trace`, all sharing one witness scan over the multilinear table `w(x, y)`.
Building `g(x) = sum_j rho^j J_tilde(j, x)` and proving `<rho-powers, p> = sum_x g(x) w_tilde(x)` is exactly the shape that, in fusion, becomes a new `gamma^k * omega_JL` addend in that batch (with the stage-1 `gamma^1 * range` term deleted at JL levels), evaluated at the same final point with no new Fiat-Shamir challenge beyond `rho`.
Prototyping it in this shape avoids a throwaway separate-sumcheck design.

**Field and ring-dimension genericity.**
The projection acts on integer coefficients, so it is field-generic by construction; centering is via `CanonicalField` (modulus from `-F::one()`, balanced representative). The matrix and projection are generic over `const D`. The consistency sumcheck operates over the claim/extension field `L` for the witness multilinear extension exactly as stage-2 does. Tests sweep fp32/fp64/fp128 and `D in {32, 64, 128, 256}`, excluding only the degenerate edge configs that default to a direct singleton send for insufficient SIS security.

### Alternatives Considered

- **Standalone separate sumcheck (not fused-row).** Simpler to write but throwaway; it would not match the fusion target and would be rewritten. Rejected in favor of building the fused-row weight now.
- **Hand-rolled textbook sumcheck instead of the `akita-sumcheck` driver.** Duplicates the engine and is less drop-in for fusion. Rejected; reuse the existing driver via a small instance type.
- **Ring-granular structured projection (committed image, mid level).** Required only when the image is folded after projection (the commutation law). The tail reveal does not fold after projecting, so the dense matrix is correct and simpler. Out of scope; covered in Deferred work.
- **Replace the terminal cleartext with JL.** The PCS base case must still verify the evaluation claim; cannot be fully replaced. Not attempted.
- **Host everything in one crate.** Crate layering forbids a sumcheck in `akita-challenges`; splitting sampling (low) from consistency (prover layer) respects the graph.

## Deferred work (fusion roadmap)

Each deferred item below is investigated to the point where the follow-up can start without re-discovery. Suggested ordering is given at the end.

### D1. Instance-descriptor binding of the projection geometry

**What.** The transcript preamble (`AkitaInstanceDescriptor`, in `akita-types/src/instance_descriptor.rs`) binds the algebra, setup identity, effective plan, and per-call shape so a proof under one configuration cannot verify under another.
Fusion must bind the JL geometry: `n_rows`, the seed domain separator, the per-level norm bound `T_p`, the variant flag (reveal vs committed), and which levels are JL levels.
The natural home is an extension of `PlanSection` (the effective per-level schedule), since the JL choice is per-level and schedule-driven, or a dedicated `JlSection`.

**Why deferred.** The prototype binds geometry through a transcript context buffer in `sample` only, which is enough for standalone determinism but not for cross-proof domain separation.

**Open questions.** Whether to fold the geometry into the existing per-level `LevelParams`/`PlanSection` digest or add a sibling section; how to keep the descriptor round-trippable and panic-free on deserialization (verifier no-panic contract); confirming a JL-level proof cannot replay as a non-JL-level proof.

### D2. The `Step` variant, proof payload, and stage-2 fusion

**What.** Three coordinated changes:

1. `akita-types/src/schedule.rs`: add a third `Step` variant (today `Fold(FoldStep)` and `Direct(DirectStep)`), e.g. `JlFold(JlFoldStep)`, carrying `n_rows`, `T_p`, and reusing `LevelParams`. The planner emits it for tail levels; `Schedule::fold_steps` and the num-levels helpers must account for it.
2. `akita-types/src/proof/levels.rs`: a JL level proof payload carrying the revealed image `p` (a `Vec` of `n_rows` integer coordinates, serialized as balanced field elements or signed integers) in place of `stage1` and either `next_w_commitment` (intermediate JL) or the cleartext witness. The consistency proof rides in stage 2.
3. `akita-prover/src/protocol/sumcheck/akita_stage2`: add the `omega_JL` addend `gamma^k * g_tilde(x) * w_tilde(x)` to the fused batch and delete the `gamma^1 * range` term at JL levels. The verifier evaluates `g_tilde` at the stage-2 final point. `rho` is sampled after `u'` binds `w_next` (wire-before-squeeze); `gamma` already exists.

**Why deferred.** This is the actual protocol cutover and touches serialized types, the planner, and both prover and verifier; the prototype is explicitly standalone.

**Open questions.** The exact `gamma` power for the JL row; image serialization (balanced field elements vs a signed-integer codec, which also feeds the tail entropy-coding work); whether intermediate JL levels (birth-certify `w_next`, then delete the next level's stage 1) and terminal-adjacent JL levels need distinct payloads.

### D3. Stage-1 deletion and SIS repricing

**What.** Deleting stage 1 removes the degree-`2^lb` range tree, the carried `s_claim`, and the stage-1 transcript at JL levels.
It also reprices the SIS roles in `akita-types/src/sis/norm_bound.rs`:

- A-role (`rounded_up_collision_norm_s`) today is `committed_fold_collision_l2_sq` with `collision_linf = 8 * omega * beta_inf * nu`. Under anchored extraction this becomes `2 * T_s` (clean, requires D4) or a fallback (also D4). This is the one real rank lever.
- B-role (`rounded_up_collision_norm_t`, the `t_hat` opening digits) and D-role (`rounded_up_collision_norm_w`) are digit-range collisions `2^lb - 1`; they are unchanged by JL but must be re-audited once stage 1 no longer certifies "digits are range-checked" (a comment in `norm_bound.rs` and several consumers assume this).
- The revealed image norm bound `T_p` must be added to pricing/sizing.

**Why deferred.** Repricing is only meaningful with the anchored-extraction lemma (D4) decided, and it regenerates the SIS floor tables and planner schedules.

**Open questions.** Which `norm_bound.rs` consumers implicitly rely on the stage-1 range guarantee (audit needed); the exact `T_p` to bucket; table regen scope.

### D4. The anchored-extraction lemma (the gating soundness item)

**What.** The clean A-role price `eta_A = 2 * T_s` (with `T_s = T_p / sqrt(30)` per block) holds only if the object the CWSS extractor recovers is the witness block `s_i` that was pinned in `u_l` before `J` was sampled.
The natural CWSS extractor instead produces a fold-response quotient `s_i^ext = (z-difference) / c_i`, and division by the ring unit `c_i` is not norm-preserving, so a bound on the image of the quotient does not transfer to `||s_i^ext||`.
Two resolutions:

- **R-A (clean).** Re-architect the projection consistency to bind the committed blocks directly (`p_i = J s_i` against `w_l`'s opening), not the post-fold response, so the extracted A-object is the pinned block. Delivers `eta_A = 2 T_s`. This is the gating write-up item.
- **R-B (fallback).** Keep the collision on the fold quotient; JL still tightens the fold-response bound to the realized `||z||_2 <= Gamma_fold * T_s`, giving `eta_A = 2 * Gamma_bar * 2 * Gamma_fold * T_s`. Larger than clean by roughly the squared challenge mass (1 to 2 module ranks) but still a large win over today's loose envelope, and it does not require re-architecting.

The pinning itself is sound and not circular: the blocks `s_i` are committed in `u_l` before `J` is sampled, so modular JL legitimately certifies their norm.

**Why deferred.** It is a soundness proof, not code; it decides clean vs fallback pricing for D3; the prototype's mechanics are independent of which resolution is chosen.

**Open questions.** Whether R-A's "consistency against the committed blocks, not the fold response" is compatible with akita's fused stage-2 trace structure without an extra commitment; the precise statement of the uniqueness-bootstrap order in the recursive-tree extraction.

### D5. Completeness: nonce regrind and the norm-threshold policy

**What.** An honest witness occasionally projects to an over-threshold image (the JL window has an upper tail).
The backup handles this with a nonce-regrind loop: the prover searches a small nonce on a cloned transcript, only commits the accepted nonce, and the verifier absorbs that one nonce.
The threshold policy (median-plus-regrind) gives a flat per-level slack near 2.07 (LaBRADOR's check-and-retry).
Fusion needs: a bounded regrind nonce in the proof/transcript, a schedule-fixed honest bucket `T_p` sized from calibrated RMS image norms, and a liveness cap (no-panic on exhaustion).

**Why deferred.** The standalone prototype can pick a single transcript draw and a generous `T_p` for tests; regrind is a completeness optimization for the wired path.

**Open questions.** Nonce search budget; how the regrind nonce binds in the descriptor/transcript; interaction with the per-draw 128-bit entropy floor.

### D6. The union bound and sizing `n_rows` (`n_J`) per level

**What.** The JL failure probability is a union over projected blocks: `kappa_JL = (#blocks) * 2^{-c(n_rows)}`, with `c(256) = 128` from the LaBRADOR 256-row constant.
Matching akita's shipped convention (drive `kappa_JL` to `2^{-128}` like the other per-level terms) gives `n_rows ~ 256 + 2*log2(#blocks)`.
At tail levels `#blocks` is small (a late level has `N_coeff ~ 2^17`, `m0 ~ 2^12`, so a few dozen blocks), so `n_rows ~ 256 to 270` suffices; the root would need `~285 to 295` and is excluded from JL on purpose (the algebraic range check stays at the root, where the union bound is largest and the projection is most expensive).
The constant `c(n_rows)` is currently an `n_rows/2` extrapolation of the LaBRADOR 256-row bound and must be re-derived exactly.

**Why deferred.** The prototype uses a configurable `n_rows` (default 256) and a generous bound; exact sizing is a security-accounting task that feeds D3 pricing.

**Open questions.** The exact `c(n_rows)`; whether to size per level from `#blocks` or fix one conservative `n_rows` for all JL levels.

### D7. ZK masking of the revealed image

**What.** A revealed image `p` leaks `n_rows` linear functionals of `w_next`, so the reveal variant is non-ZK unless masked.
Options: add `n_rows` blinding evaluations (a small deferred-mask family in the existing ZK accounting, alongside the stage-2 masks in `akita-prover/src/protocol/masking.rs` / `zk_hiding_commit.rs`), or restrict the reveal variant to non-ZK builds and terminal-adjacent levels where the witness is about to be sent in the clear anyway.

**Why deferred.** The prototype targets non-ZK builds; the leak is irrelevant until ZK fusion.

**Open questions.** Mask family size and where it slots into the proof-level hiding witness cursor; whether the committed-image (Slot-2) variant is the better ZK path instead.

### D8. The structured ring-granular (committed-image, mid-level) variant

**What.** For mid levels (not the tail), the image is committed in `v` and folded again, which the commutation law forces to a ring-granular `J_0 (x) I_D`.
This is the higher-value but more complex variant (no reveal, no leak, smaller image overhead via nesting), and it shares the consistency machinery with the reveal variant but checks the image norm via a committed micro-range or exact-`l2`-on-image rather than over `Z`.

**Why deferred.** The user scoped this prototype to the unstructured tail variant; the structured variant is a separate, larger piece.

**Open questions.** All of the committed-image enforcement menu (micro-range vs carry-lifted exact-l2), the nested-projection constants, and the same-level image-norm enforcement that prevents per-level slack from compounding.

### Suggested ordering for fusion

D6 and D4 (security: size `n_rows`, decide R-A vs R-B) settle pricing inputs; D3 (delete stage 1, reprice, regen tables) and D1 (descriptor) are the type/accounting changes; D2 (Step, payload, stage-2 fusion) is the protocol cutover; D5 (regrind) and D7 (ZK) are completeness/ZK polish; D8 is a separate mid-level project.

## Documentation

- This spec is the primary design record for the prototype and the fusion roadmap.
- New crate-level module docs (`//!`) on `akita-challenges::jl` and `akita-prover::protocol::jl`.
- No public book or security-doc changes until fusion (the prototype changes no shipped behavior). When fusion lands, the security-model and norm-bounds pages need the JL paradigm-schedule and the anchored-extraction pricing.

## Execution

Suggested order for the prototype:

1. `akita-challenges::jl`: port the packed-ternary matrix, seed expansion, `project`, and norm helpers from the backup; re-host the seed on the current transcript; make `n_rows`/`cols` parameters; add unit tests (determinism, round-trip, projection-vs-reference, fp128, norm bound).
2. `akita-prover::protocol::jl`: `fold_matrix_rows` plus the degree-2 product consistency sumcheck instance (a small `JlConsistencyInstance` implementing the `akita-sumcheck` prover/verifier traits); standalone `prove_jl_consistency` / `verify_jl_consistency` harness; cross-field/cross-dim round-trip, soundness-direction, and malformed-input tests.
3. Lint and full test sweep (`parallel` on and off).

Risks to resolve first:

- Confirm the witness multilinear-extension vocabulary (`akita-witness::PolynomialView`) and the `akita-sumcheck` driver entry are ergonomic for a two-multilinear product oracle; if the driver expects a specific instance trait, implement the small `JlConsistencyInstance` rather than forcing a generic path.
- Confirm `CanonicalField` exposes modulus and centered conversion uniformly across fp32/fp64/fp128.

## References

- `labrador-backup:src/protocol/labrador/johnson_lindenstrauss.rs`: the reusable dense reveal projection (packed ternary, SHAKE seed expansion, `project`, nonce regrind, i64/i128 paths) to port from.
- LaBRADOR (eprint 2023/1729): modular-JL lemma and the 256-row constant; the check-and-retry threshold policy.
- Grand Danois (eprint 2026/1196): structured-projection-in-relation and the nested lever (for the deferred mid-level variant); constants must be re-derived.
- `akita-types/src/sis/norm_bound.rs`: A/B/D-role collision pricing the anchored repricing (D3, D4) targets.
- `akita-types/src/instance_descriptor.rs`: the transcript preamble that must bind JL geometry (D1).
- `akita-types/src/schedule.rs`, `akita-types/src/proof/levels.rs`, `akita-prover/src/protocol/sumcheck/akita_stage2`: the `Step`, proof payload, and fused stage-2 surfaces fusion touches (D2).
- Profiling (for the fusion measurement): `AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 cargo run --release --example profile`.
