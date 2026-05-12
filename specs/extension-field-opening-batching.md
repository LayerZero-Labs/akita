# Spec: PR #71 Part 2 - Extension-Field Opening Completion And Frobenius Cutover

| Field | Value |
| --- | --- |
| Author(s) | Quang Dao |
| Created | 2026-05-06 (originally as the umbrella for PRs #69, #71, and this completion work) |
| Status | baseline extension path, generic multipoint incidence packaging, and the dense Frobenius route have landed in the worktree; remaining work is true-tower E2E coverage, one-hot extension coverage, planner tuning, and full CI validation |
| PR | #71 (`quang/general-field-final`) |
| Companion spec | `specs/extension-field-trace-cutover.md` (#71 first slice) |
| Earlier slices | `specs/general-field-support.md` (#60), `specs/extension-claim-incidence-cutover.md` (#69) |

## Summary

This spec tracks the second implementation slice inside PR #71. The codebase is
now the source of truth: the straight-line extension-field opening path has
landed end to end for the supported shapes, including proof-payload `F, L`
types, `L`-valued proof scalars, explicit ring-subfield materialization, root
and recursive field-reduction boundaries, compact recursive terminal witnesses,
field-family SIS sizing, and honest profile accounting.

Several concrete pieces have landed in this PR:

- Root folded proofs now sample batching coefficients in `L`, and stage-2
  already samples `batching_coeff` and round challenges in `L`. Public-opening
  batching is represented by explicit public rows rather than by an overloaded
  global claim vector, so same-point batching and same-commitment multipoint
  openings use the same incidence package.
- The root folded path commits to base coefficients after lifting/packing them
  through the ring-subfield encoding; it no longer treats arbitrary extension
  coordinates as raw base rows.
- Recursive folded witness openings use the same explicit Akita field-reduction
  boundary as the root path. The physical recursive W length stays separate
  from the serialized terminal witness shape, so the terminal direct proof now
  carries compact packed digits instead of a full extension-field row payload.
- Valid extension openings outside the packed-inner folded-root shape use the
  root-direct fallback instead of failing the public API.
- Folded small-field verification now keeps row-evaluation ordering aligned
  across prover and verifier, including the all-features `zk` path.
- The profile example has been split into modules, and small-field profile
  modes now cover explicit fp32/fp64 dense and one-hot candidate dimensions.
- SIS floors are now selected by `SisModulusFamily::{Q32,Q64,Q128}` and the
  generated registry covers the larger small-field D ladders.
- Sparse challenge sampling has stack-backed tiers through D=512; supported
  larger-D profiles no longer route through a heap-backed fallback.
- The prover/verifier dispatch now uses the same incidence summary to choose
  between folded root proofs and the sound root-direct fallback. Extension
  shapes that the folded path cannot yet price or materialize no longer fail
  by default; they route through direct witnesses until the Frobenius path can
  keep them folded.
- The profile harness now routes dense base-field workloads through the
  Frobenius-conjugate transform when the configured extension field supports
  the canonical split. The straightforward lifted baseline remains the
  conceptual control case: for `E > F`, base-field evaluations are lifted into
  `E`, psi-packed into base-field ring material, and opened at a transformed
  point with `original_num_vars + log2([E:F])` protocol variables.

The main architectural change is the field tower convention:

```text
F ⊆ E ⊆ L
```

- `F` is the base ring coefficient field, currently `Cfg::Field`.
- `E` is the public opening field, currently `Cfg::ClaimField`.
- `L` is the Fiat-Shamir / proof scalar field, currently `Cfg::ChallengeField`.

Generic code should use `F, E, L` for these roles. Avoid using `L` for
lengths, levels, or layout values in any scope that also names the challenge
field. Avoid using `K` for a field type: in this repository `K` means an
extension degree in APIs such as `SubfieldParams<D, K>`.

Folding the deferred work into PR #71 means the PR is split into:

- **Part 1: proof-scalar payload cutover.** Landed in this PR: proof structs
  carry `F` ring material and `L` proof-scalar material, stage-1/stage-2
  sumchecks run over `L`, recursive verifier/prover state stores `L` claims,
  and `AkitaCommitmentScheme` exposes `AkitaBatchedProof<F,
  Cfg::ChallengeField>`.
- **Part 2: extension-opening completion.** This spec. Most of this has now
  landed in #71, including the dense Frobenius route. The main remaining work
  is to harden the tests, tune the planner/profile choices for small fields,
  and decide which non-smoke profiles should become CI benches.

Part 2 has therefore split into completed baseline work and next optimization
work:

- **Phase 4 completion.** Landed: sample `gamma` in `L`, make root and recursive
  opening materialization work for true `E`/`L` points rather than degree-one
  projections, and remove the remaining degree-one bridges.
- **Phase 5A: generic multipoint incidence packaging.** Landed in the
  worktree: opening points, claims, and public rows are separate protocol
  objects, with row-local batching coefficients. This is the common substrate
  for existing same-point many-polynomial batching and new same-commitment
  multipoint openings.
- **Phase 5B: Frobenius-conjugate base/ext optimization.** Landed for dense
  base-coefficient workloads: the code supports split parameter `t`, base
  slices `f_h`, transformed tail polynomial `g`, conjugate tail openings, and
  Moore-system reconstruction/binding checks. Remaining work is hardening and
  planner tuning, not inventing a separate proof path.
- **Phase 6: planner / proof-size and SIS accounting.** Partly landed:
  field-family SIS floors, wider D candidates, compact terminal witness sizing,
  and serialized proof-size assertions are in place. Remaining Frobenius
  accounting must price split parameter `t`, base-field bytes versus
  extension-field bytes, and shared-group versus per-point/per-edge costs.
- **Phase 7: E2E and CI hardening.** Partly landed: dense extension E2E,
  root-direct incidence fallback tests, profile verification, and small-prime
  benchmark CI coverage exist. Remaining tests belong to Frobenius and a true
  `F < E < L` tower E2E.

## Intent

### Goal

Finish the extension-field opening cutover so that:

1. The live prover/verifier path has no degree-one bridges. `K > 1` configs are
   first-class, not only accepted by helper-level trace tests.
2. The proof payload type reflects the field tower: ring material over `F`,
   public openings over `E`, and proof scalars over `L`.
3. The current generic route is honest: base coefficients are lifted and Akita
   packed before commitment, proof scalars live in `L`, and terminal witnesses
   are serialized in the compact field-reduced shape rather than as full
   extension rows.
4. The Frobenius-conjugate route is available for the common Akita/Jolt shape:
   base-field-valued committed tables opened at extension-field sumcheck
   points.
5. The planner/proof-size layer can price the generic route and the
   Frobenius-conjugate route without pretending all scalars are fp128 base
   elements.

### Scope Boundary

- The proof-payload reshape is already part of PR #71 Part 1. The structs
  `AkitaStage2Proof`, `AkitaLevelProof`, `AkitaBatchedProof`,
  `AkitaBatchedRootProof`, and `AkitaProofStep` now carry an `L` type
  parameter for proof scalars; `AkitaStage1Proof<L>` and
  `AkitaStage1StageProof<L>` are scalar-typed directly by `L`.
- The public config associated type names may remain `ClaimField` and
  `ChallengeField` in this PR, but implementation generics and docs should use
  `F, E, L` for the tower. A later cosmetic rename to `OpeningField` can be
  mechanical if desired.
- The Frobenius-conjugate route ships as a selectable optimization on top of
  the incidence model, not as a separate public opening API.
- Until Frobenius lands, valid extension-field openings that do not fit the
  packed-inner folded-root materialization use the existing root-direct proof
  path. This is intentionally sound and complete for E2E behavior, but not
  proof-size optimal.
- The current folded baseline is intentionally straightforward: embed base
  coefficients into `E`/`L`, pack through the ring-subfield encoding into
  cyclotomic rings, and commit to the resulting base-field ring material. It is
  not the final proof-size target.
- Planner work extends the existing aggregate `(num_claims, num_groups,
  num_points)` shape with field-degree and split-parameter inputs. Do not
  rewrite the planner unless the existing model cannot express the required
  costs.

### Invariants

- Ring commitments, setup matrices, recursive witnesses, digit decomposition,
  CRT/NTT work, and SIS bounds remain over `F`.
- Public opening points and claimed evaluations are over `E`.
- Fiat-Shamir scalar challenges that need extension-field soundness are sampled
  over `L`; base-ring transcript absorption still uses transcript field `F`.
- `L: ExtField<F> + ExtField<E>` and `E: ExtField<F>`.
- The fp128 production path remains the degree-one specialization
  `F = E = L`; no compatibility wrappers or parallel legacy APIs remain.
- SIS security floors are keyed by the active base-field modulus family, not
  only by `(D, collision_inf, width)`. fp32/fp64 must not inherit the fp128 SIS
  width registry.
- One incidence representation covers same-point batching, same-commitment
  multipoint openings, arbitrary point/group routing, and Frobenius-conjugate
  openings.
- Public quotient rows are a packaging of claims, not the same object as
  opening points or claims. Without the later row-count optimization, each
  public row has exactly one opening point; several claims may share that row
  only when they share the same opening point.
- Each public row owns its own batching coefficients. Do not reuse one global
  `gamma_c` vector across unrelated rows: row-local coefficients make the
  soundness argument and transcript binding local to the claims actually
  combined in that row.
- Transcript binding absorbs the full incidence shape: points, groups,
  commitments, claim routing, public-row packaging, row-local batching
  coefficients, and claimed evaluations.
- Wrong claims, wrong conjugate points, invalid Moore systems, redistribution
  attempts, and transcript reordering are rejected.

### Non-Goals

- Do not introduce separate public APIs for each batching special case.
- Do not keep base-field-only aliases after the cutover.
- Do not implement the unsound literal base-field optimization based on
  unbound extension-valued partial evaluations.
- Do not add a separate ring-switching sumcheck for base/ext mismatch; the
  intended trade is wider same-commitment opening versus fewer transformed
  variables.
- Do not generate or fossilize production fp32/fp64 schedule tables until the
  field-family-specific SIS floor registry is in place and validated.

## Implementation Plan

### Phase 4A: Audit And Type-Parameter Cut

Status: landed in PR #71 Part 1.

Audit every place where a proof scalar is currently typed as `F`.

Classify as:

- `F`: ring coefficient, digit, commitment, setup, or base transcript material.
- `E`: public opening point or claimed evaluation.
- `L`: Fiat-Shamir challenge, batching coefficient, sumcheck scalar, or
  recursive claimed evaluation.

Then cut the proof data model:

- `AkitaStage1StageProof<F>` becomes `AkitaStage1StageProof<L>`.
- `AkitaStage1Proof<F>` becomes `AkitaStage1Proof<L>`, unless a base-field
  member is introduced; then use `AkitaStage1Proof<F, L>`.
- `AkitaStage2Proof<F>` becomes `AkitaStage2Proof<F, L>` because it carries an
  `F`-typed next-witness commitment and an `L`-typed sumcheck/evaluation.
- `AkitaLevelProof<F>` becomes `AkitaLevelProof<F, L>`.
- `AkitaBatchedFoldRoot<F>` becomes `AkitaBatchedFoldRoot<F, L>`.
- `AkitaBatchedRootProof<F>` becomes `AkitaBatchedRootProof<F, L>`.
- `AkitaProofStep<F>` becomes `AkitaProofStep<F, L>`.
- `AkitaBatchedProof<F>` becomes `AkitaBatchedProof<F, L>`.

Use a single full cutover. Do not add type aliases like
`type AkitaBatchedProofF<F> = ...`.

Serialization and validation changes:

- [x] Update `AkitaSerialize`, `Valid`, and deserialize-with-shape implementations
  for every reshaped proof type.
- [x] Shape descriptors remain field-agnostic unless the encoded shape truly
  changes.
- [ ] Add roundtrip tests for one extension pair such as `(fp32, Fp2<fp32>)`
  or `(fp32, TowerBasisFp4<fp32>)`. The current all-target check exercises the
  degree-one specialization.

### Phase 4B: Prover And Verifier Scalar Flow

Root prover:

- [x] Sample root same-point batching `gamma_i` in `L` for folded roots.
- [x] Sum per-point openings in `L` at the trace boundary for folded roots.
- [x] Feed `gamma: &[L]` into root ring-switch relation evaluation.
- [x] Use packed-inner ring-subfield materialization for folded extension
  roots when the shape is supported.
- [x] Fall back to root-direct for valid extension openings that need outer
  variables or same-point extension batching before Frobenius optimization.

Root verifier:

- [x] Sample `gamma_i` in `L` using the same transcript labels.
- [x] Sum public openings in `L`.
- [x] Convert the `E`/`L`-valued trace target into `F` coordinates only at the
  `dispatch_trace_inner_product_check` boundary. If `L != E`, explicitly
  project or require the trace target to land in `E`; do not silently truncate
  `L` to `E`.
- [x] Keep the trace-check helper over base coordinates in `F`.

Stage 1:

- [x] Stage-1 sumcheck payloads and interstage batching challenges are over `L`.
- [x] Stage-1 relations that read ring/digit witnesses continue to lift `F` values
  into `L`.

Stage 2:

- [x] Sample `batching_coeff` and stage-2 round challenges in `L`.
- [x] `AkitaStage2Verifier` and the prover-side stage-2 sumcheck are generic
  over `L`; relation row evaluations lift base-ring material into `L`.
- [x] `s_claim`, `next_w_eval`, and sumcheck final claims are `L`.

Recursive suffix:

- [x] `RecursiveVerifierState<'a, F>` becomes
  `RecursiveVerifierState<'a, F, L>`.
- [x] Prover recursive state carries `Vec<L>` sumcheck challenges.
- [x] `opening_point` and `opening` become `Vec<L>` and `L` in verifier state.
- [x] Replace the current degree-one projection used to materialize recursive
  ring opening points with the same explicit field-reduction boundary as the
  root path.

Scheme/config:

- [x] `AkitaCommitmentScheme` instantiates proof types as
  `AkitaBatchedProof<F, Cfg::ChallengeField>`.
- [x] Prover traits return `AkitaBatchedProof<F, L>` where
  `L = ChallengeField`.
- [x] Verifier traits consume the same proof type.

### Phase 4C: Remove Bridges And Add Early Validation

Bridge status:

- [x] `DegreeOneChallengeSampler` is removed. Root `gamma` is now sampled in
  `L`; no sampled challenge is projected through a degree-one bridge.
- [x] `claim_points_to_base`
- [x] `require_degree_one_ext`
- [x] `degree_one_ext_scalar_to_base`

The remaining folded-root guards now select root-direct for valid extension
openings outside the packed-inner folded shape. Removing those guards from the
folded path itself belongs to the generic multipoint incidence packaging and
Frobenius optimization work, not to the E2E correctness boundary.

Add early validation at setup/scheme entrypoints:

- [x] `E::EXT_DEGREE` is supported by `dispatch_trace_inner_product_check`.
- [x] `SubfieldParams<D, E::EXT_DEGREE>::new()` succeeds:
  - `D` is a nonzero power of two.
  - `E::EXT_DEGREE` divides `D / 2`.
  - `gcd(4 * E::EXT_DEGREE + 1, 2 * D) == 1`.
- [x] `L: ExtField<E>` is already a trait-level invariant; scheme entrypoints
  now also check that the declared absolute degrees match the relative tower
  degree before proving or verifying.

### Phase 4D: Documentation And Direct Tests

Rustdoc:

- [x] Add proof field-role docs near the proof structs:
  `F` is ring material, `L` is proof-scalar material.
- [x] Add field-reduction norm docs:
  - `k = 1`: no subfield-basis blowup; the trace shortcut is scalar equality.
  - `k > 1`: Akita uses the fixed subfield basis and pays the documented
    embedding/trace blowup.

Tests:

- [ ] Proof-payload roundtrip tests for representative `(F, L)` pairs.
- [x] Extension challenge replay tests cover transcript helper behavior; live
  root/stage paths now sample `gamma`, `batching_coeff`, and stage-2 round
  challenges as `L`.
- [x] Live verifier-orchestration trace tests for extension-valued openings, not
  only helper-level `field_reduction` tests.
- [ ] Add a true `F < E < L` tower E2E. Current small-prime production presets
  use `E = L`, while trait support for `F ⊆ Fp2 ⊆ Fp4` exists.

### Phase 5A: Generic Multipoint Incidence Packaging

Status: landed in the PR #71 worktree; focused hardening and CI validation are
still pending.

The folded root path should use one incidence model for:

- the existing same-point, many-polynomial batching;
- same-commitment multipoint openings;
- arbitrary point/group/poly routing;
- Frobenius-conjugate internal openings.

The normalized model has three layers:

```text
Opening point p:
  a_p in E^n

Claim c:
  group_idx(c)
  poly_idx(c)
  point_idx(c)
  claimed_value y_c in E

Public row r:
  point_idx(r)
  terms(r) = [(claim_idx, gamma_{r,claim_idx}), ...]
```

The row invariant for the no-row-count-optimization baseline is:

```text
point_idx(c) = point_idx(r) for every (c, gamma_{r,c}) in terms(r).
```

That is, a public row may batch many claims, but only when all those claims
are opened at the same point. Batching claims at different points into one
public row would require a new row object whose point multiplier is a linear
combination of point multipliers; that optimization is intentionally out of
scope for this slice.

Each public row proves one ring equation:

```text
sum_{(c, gamma_{r,c}) in terms(r)}
  gamma_{r,c} * B(a_{point_idx(r)}) * W_c
=
Y_r
```

where:

```text
Y_r = iota(sum_{(c, gamma_{r,c}) in terms(r)} gamma_{r,c} * y_c)
```

and `iota` is the ring-subfield/ring embedding into `R_F`. For singleton
rows, `gamma_{r,c} = 1`.

This makes the existing same-point batching a specialization:

```text
points = [a]
claims = [P_0(a)=y_0, P_1(a)=y_1, P_2(a)=y_2]
rows = [
  point 0: [(claim 0, gamma_{0,0}),
            (claim 1, gamma_{0,1}),
            (claim 2, gamma_{0,2})]
]
```

and generic multipoint a different packaging:

```text
points = [a_0, a_1, a_2, a_3]
claims = [g(a_0)=y_0, g(a_1)=y_1, g(a_2)=y_2, g(a_3)=y_3]
rows = [
  point 0: [(claim 0, 1)]
  point 1: [(claim 1, 1)]
  point 2: [(claim 2, 1)]
  point 3: [(claim 3, 1)]
]
```

Row-local batching coefficients must be sampled independently per row that
contains more than one term. Do not reuse a single global claim-indexed gamma
vector across all rows. The transcript order should be:

1. absorb the normalized incidence shape, including public-row packaging;
2. absorb commitments, public opening points, and claimed values;
3. for every public row, sample the row's local batching coefficients in `L`
   after the row terms are fixed.

The current `gamma: Vec<L>` API should therefore be cut over to explicit row
terms. Temporary interpretations such as "gamma is per claim, unless the claim
is a singleton multipoint row" are not acceptable as a long-term API.

Implementation shape:

```rust
struct OpeningClaim {
    point_idx: usize,
    group_idx: usize,
    poly_idx: usize,
}

struct PublicRowTerm<L> {
    claim_idx: usize,
    coeff: L,
}

struct PublicOpeningRow<L> {
    point_idx: usize,
    terms: Vec<PublicRowTerm<L>>,
}

struct OpeningIncidence<L> {
    claims: Vec<OpeningClaim>,
    public_rows: Vec<PublicOpeningRow<L>>,
    group_poly_counts: Vec<usize>,
}
```

Derived views may still exist for hot loops:

```text
claim_to_point[c]
claim_to_row[c]
row_to_point[r]
claim_to_group[c]
claim_poly_indices[c]
```

but those views must be derived from the canonical incidence package, not
hand-maintained as separate protocol truth.

Folded witness impact:

- `w_folded` remains claim-indexed:

  ```text
  W_c = partial_fold(P_c, a_{point_idx(c)})
  w_folded = [W_0, W_1, ..., W_{num_claims-1}]
  ```

- `w_hat` remains the digit decomposition of `w_folded`; it grows with the
  number of claims, not directly with the number of public rows.
- `t_hat` / recomposed inner rows remain commitment-hint material, grouped by
  committed polynomial group. They should not be semantically duplicated just
  because the same polynomial is opened at multiple points.
- `z_pre` / centered `z_hat` material remains the same decomposition-fold
  witness object. It is computed with the claim-indexed sparse challenges and
  grouped by opening point where the existing `decompose_fold_batched` path can
  aggregate claims at the same point.
- Quotient `r` rows still have the same layout:

  ```text
  consistency | public rows | D rows | B rows | A rows
  ```

  The only intended change is that the public-row block is driven by
  `public_rows.len()` and row-local terms rather than by an overloaded
  `num_points` / `claim_to_point` / `gamma` interpretation.

Current code state:

- `ClaimIncidenceSummary` carries explicit public-row counts and
  claim-to-public-row routing.
- `PublicOpeningRow` / row-local term data drive prover and verifier replay.
- Same-point many-polynomial batching emits one row with several row-local
  coefficients.
- Same-commitment multipoint openings emit singleton rows at distinct points.
- `combine_root_y_rings`, `QuadraticEquation::new_prover`,
  `compute_r_split_eq`, prover `compute_m_evals_x`, verifier
  `RingSwitchVerifier`, and relation-claim replay consume the same row-package
  semantics.
- The current `tau1` row combination remains unchanged:

  ```text
  relation_claim = sum_row eq_tau1(row) * eval_alpha(M_row)
  ```

  This is the generic late batching of all relation rows and is separate from
  row-local claim batching.

Remaining hardening:

- Add or strengthen tests for transcript reordering of row terms or rows.
- Keep row-local coefficient tests close to the incidence package so future
  planner/prover changes cannot accidentally recreate global-gamma semantics.

### Phase 5B: Frobenius-Conjugate Base/Ext Optimization

Status: landed for dense base-coefficient workloads and recursive folded
witness packing; proof-size/planner tuning and test hardening remain.

The current code has an honest but intentionally non-final baseline:

- `run_dense_for` in the profile example calls
  `lift_dense_evals_to_psi_packed_poly::<F, E, D>` and
  `transform_extension_opening_point::<F, E, D>`.
- That path increases the protocol variable count by
  `log2([E:F])`, because it first embeds base coefficients into extension
  slots and then exposes the ring-subfield packing dimensions to the root
  opening.
- Folded root support is guarded by
  `folded_root_supports_opening_shape::<F, E, L, D>`. Unsupported extension
  shapes fall back to root-direct in both prover and verifier dispatch.
- `validate_field_roles_for_ring` already checks the representability of `E`
  and `L` in the ring-subfield boundary and checks the tower degree relation
  `F ⊆ E ⊆ L`.

The Frobenius optimization should therefore be implemented as a replacement
commit/open transformation for base-coefficient polynomials, not as another
escape hatch around the existing verifier. The optimized path should still
commit through the production Akita pipeline: base coefficients are mixed into
extension-field packed coefficients, those coefficients are embedded through
the same psi/subfield boundary, and all proof recursion remains over `F` ring
  material with `L` proof scalars. Concretely, the extension-domain table `g`
  has `ell - t` Boolean variables, while the current `F`-coefficient Akita
  ring representation exposes `ell - t + log2([E:F])` protocol variables.
  The optimization removes `t` variables from the current lifted baseline; it
  does not remove the extension-coordinate slots needed for the injective
  subfield packing.

Add a small explicit representation for the split parameter:

```text
0 <= t <= log2([E : F])
P = 2^t
```

For base-field polynomial coefficients opened at `E` points:

1. Split variables into `X_head` of length `t` and `X_tail`.
2. Slice the base polynomial:

   ```text
   f(X_head, X_tail) = sum_h lambda_h(X_head) f_h(X_tail)
   ```

3. Choose deterministic `theta_h in E` whose Moore-type matrix below is
   nonsingular. For the `RingSubfieldFp4<F>` small-field path, use the
   canonical ring-subfield basis `[1, e1, e2, e3]` first; this is the basis
   that preserves the intended coefficient packing. Other extension families
   should use their canonical `ExtField::from_base_slice` basis unless a later
   measured reason forces a specialized theta family.
4. Build:

   ```text
   g(X_tail) = sum_h theta_h f_h(X_tail)
   ```

5. Open the same committed transformed polynomial at Frobenius-conjugate tail
   points:

   ```text
   x_tail^(q^j) = (x_{t+1}^{q^j}, ..., x_ell^{q^j})
   s_j = g(x_tail^(q^j))
   ```

6. Verify the Moore system:

   ```text
   s_j^(q^-j) = sum_h theta_h^(q^-j) * f_h(x_tail)
   ```

7. Reconstruct the original claim:

   ```text
   y = sum_h lambda_h(x_head) * f_h(x_tail)
   ```

The Moore coefficient matrix is:

```text
M_t(theta)_{j,h} = theta_h^(q^-j), 0 <= j,h < P.
```

For `P = [E:F]`, this is a classical Moore matrix up to row permutation. For
partial splits `P < [E:F]`, it is a Moore-type submatrix; do not rely on
`F`-linear independence alone as a hidden contract. The implementation must
select deterministic `theta_h` and explicitly validate that `M_t(theta)` is
nonsingular for every supported `(E, t)` pair.

Current code state:

- **Field algebra.** `akita-field` exposes explicit Frobenius support next to
  `ExtField`, including `frobenius_pow`, `frobenius_inv_pow`, canonical theta
  selection, Moore-type validation, and solve helpers for the supported
  degree-one/quadratic/quartic shapes.
- **Canonical theta basis.** `RingSubfieldFp4<F>` uses the canonical
  ring-subfield packing basis. This keeps the Frobenius transform aligned with
  the coefficient packing rather than choosing an unrelated basis that happens
  to be invertible.
- **Transformed polynomial construction.**
  `crates/akita-prover/src/backend/frobenius.rs` owns
  `dense_frobenius_transform`, which constructs
  `g(X_tail) = sum_h theta_h f_h(X_tail)` and then feeds the transformed
  polynomial into the existing commitment path.
- **Opening incidence expansion.** One public opening at
  `x = (x_head, x_tail)` expands into `P` ordinary internal openings of the
  same transformed commitment at
  `x_tail, x_tail^q, ..., x_tail^(q^(P-1))`. The proof path reuses the generic
  same-commitment multipoint incidence machinery from Phase 5A.
- **Claim payload binding.** The internal values
  `s_j = g(x_tail^(q^j))` are ordinary claimed openings of the transformed
  commitment. The public claim remains the original `E`-valued `y`.
- **Verifier reconstruction.** The verifier computes `r_j = s_j^(q^-j)`,
  solves `M_t(theta) z = r`, and checks
  `y == sum_h lambda_h(x_head) z_h`.
- **Recursive folded witness packing.** Recursive levels use the same
  Frobenius lift/pack boundary instead of reverting to the earlier large
  extension-row witness encoding.

Remaining TODOs for this branch:

- Strengthen Frobenius negative tests beyond the current wrong-conjugate,
  duplicate-theta, and public-opening-preserving redistribution coverage.
- Add more true `F < E < L` coverage. The code has the trait tower
  `F ⊆ Fp2 ⊆ Fp4`; production small-field presets currently use `E = L`, so a
  full prover/verifier E2E with strict inclusions is still a separate
  hardening target.
- Tune planner/profile inputs for the selected split `t`. For the Frobenius
  route, the transformed base workload should have:

  ```text
  extension-domain variables = ell - t
  protocol variables = ell - t + log2([E:F])
  internal opening width = 2^t
  ```

  This is different from the current lifted baseline, which has:

  ```text
  transformed variables = ell + log2([E:F])
  internal opening width = 1
  ```

  Profile output should make this distinction explicit so proof-size reports
  cannot accidentally compare the optimized route against stale baseline
  planner estimates.
- **Fallback boundary.** Keep root-direct fallback as the sound default for
  extension shapes outside the optimized route, but do not let the Frobenius
  implementation become a fallback path itself. If `(E, t)` has no validated
  Frobenius/Moore data, reject the optimized schedule selection and let the
  existing incidence dispatch choose root-direct or the generic folded
  baseline according to the schedule.

Negative tests:

- Wrong original claim fails.
- Wrong conjugate tail point fails.
- Degenerate or duplicate `theta_h` fails.
- Redistribution attack fails: changing the slice evaluations while preserving
  only the final linear combination must not verify.

Positive tests:

- Tiny hand-checkable `K = 2, t = 1` case:

  ```text
  g = f_0 + theta f_1
  s_0 = z_0 + theta z_1
  s_1 = z_0^q + theta z_1^q
  ```

  and verifier reconstruction recovers `z_0, z_1`.
- fp64 dense E2E with `t = 1`.
- fp32 dense E2E with `t = 2`.
- one-hot E2E on a small-field config.
- Multipoint same-commitment E2E where the internal points are exactly the
  Frobenius-conjugate tail points.

### Phase 6A: Field-Family SIS Floor Registry

Status: landed in PR #71 Part 2.

The current SIS floor registry is calibrated for an fp128 representative
modulus and is keyed only by `(D, collision_inf, width)`. That is not a valid
abstraction once `F` can be fp32 or fp64: the maximum secure width for a fixed
rank and collision bound depends on `q`. Reusing the fp128 table for small
fields can under-size ranks and overstate binding security.

Introduce an explicit SIS modulus-family policy:

```rust
pub enum SisModulusFamily {
    Q32,
    Q64,
    Q128,
}
```

The family names are security-table names, not CRT implementation details:

- `Q32`: table generated for the fp32 small-field family. For the current
  scaffold this should use the concrete prime `q = 2^32 - 99`, or a documented
  lower family representative if additional fp32 primes are admitted. This
  family must include larger ring dimensions because fp32 may need more ring
  dimension to recover the same SIS margin.
- `Q64`: table generated for the fp64 small-field family. For the current
  scaffold this should use the concrete prime `q = 2^64 - 59`, or a documented
  lower family representative if additional fp64 primes are admitted. This
  family should include at least one larger ring dimension than the current
  fp128 defaults.
- `Q128`: table generated for the fp128 production family. Use the conservative
  family representative

  ```text
  q_128 = 2^128 - (2^32 - 22537)
        = 2^128 - 0xffffa7f7
  ```

  because this is the smallest current production-style fp128 modulus in the
  supported pseudo-Mersenne family. Do not silently reuse the older
  `q = 2^128 - 275` table once this policy lands; that modulus is larger and
  therefore less conservative.

Implementation requirements:

- [x] Add a config hook such as `CommitmentConfig::sis_modulus_family()` and mirror
  it through `PlannerConfig`.
- [x] Change SIS lookup APIs from:

  ```rust
  min_rank_for_secure_width(d, collision_inf, width)
  ceil_supported_collision(d, collision_inf)
  ```

  to field-family-aware forms:

  ```rust
  min_rank_for_secure_width(family, d, collision_inf, width)
  ceil_supported_collision(family, d, collision_inf)
  ```

- [x] Move the generated SIS floor table shape from one global registry to
  per-family registries keyed by `(family, D, collision_inf)`.
- [x] Update `AjtaiKeyParams` validation and `sis_derivation` so every security
  check receives the config-selected family explicitly.
- [x] Update `scripts/gen_sis_table.py` to accept `--family {q32,q64,q128}` or an
  explicit `--q`, and emit the representative modulus in the generated Rust
  comments.
- [ ] Generated schedule tables must record or be generated under the same family
  used by the config that consumes them.

Tests:

- Unit tests prove fp32/fp64 configs select `Q32`/`Q64`, and fp128 presets
  select `Q128`.
- A regression test fails if a small-field config can validate against the
  fp128 SIS table.
- Generated `sis_floor` comments include the representative `q` for each
  family.
- Existing fp128 generated schedules continue to validate against the new
  `Q128` table after regeneration or an explicitly documented transition.

### Phase 6B: Ring-Dimension Family Presets

Status: landed for dynamic/static profile presets; generated production
schedule-table families remain deferred until defaults are selected from
profile data.

The old production schedule families were centered on fp128 and only generated
presets for `D = 32, 64, 128`. Once SIS tables are modulus-family-specific,
small fields need a wider ring-dimension ladder. As a rule of thumb, keeping
similar SIS room when the base modulus halves in bit width pushes the useful
ring dimension up by roughly one doubling:

```text
fp128 at D=32  ~  fp64 at D=64  ~  fp32 at D=128
```

That rule is only a sizing intuition. The planner and profiles must measure the
actual proof-size and runtime tradeoff, especially because larger `D` changes
several costs at once:

- It increases per-ring operation size, NTT work, cache footprint, and proof
  bytes for every emitted ring element.
- It can reduce required SIS rows and may improve commitment width/security
  feasibility.
- It reduces variable count by `alpha = log2(D)` inside root layout derivation,
  which can reduce some folding costs.
- For fp32 specifically, `D > 64` currently dispatches through the conservative
  Q64 CRT/NTT parameter family rather than the i16 Q32 fast path, so runtime
  effects must be profiled instead of inferred from byte counts alone.

Required candidate ladders:

- `Q128`: keep the existing production ladder `D in {32, 64, 128}`. Consider
  `D=256` only if the regenerated Q128 SIS table shows a real schedule win.
- `Q64`: add generated/configurable candidates for at least
  `D in {64, 128, 256}`.
- `Q32`: add generated/configurable candidates for at least
  `D in {128, 256, 512}`.
- Smaller cross-over dimensions may still be useful for dense profiles:
  Q64/D32 and Q32/D64 should be kept only if they appear in measured final
  dense schedules. They must not be treated as viable one-hot defaults unless
  the root layout is SIS-secure at the target non-smoke sizes.

Implementation requirements:

- [x] Add proof-optimized preset structs for the new
  small-field ring dimensions, without disturbing the existing fp128 preset
  names.
- [ ] Extend the schedule-table generator so family specs are not hardcoded to
  fp128 `D32/D64/D128`.
- [x] Extend SIS generation for Q32/Q64 to cover the larger `D` buckets above.
- [x] Sparse challenge samplers must have explicit fast paths for each supported
  candidate ring dimension. In particular, D=256 and D=512 must not route
  through a heap-backed "large D" fallback on the proof hot path.
- [x] Keep runtime profile mode names explicit, for example
  `onehot_fp32_d128`, `onehot_fp32_d256`, and `onehot_fp32_d512`, so profile
  output makes the selected ring dimension unambiguous.
- Selection helpers may choose the best generated schedule by proof bytes, but
  the profile report must still print timings for each candidate family before
  we bless a default.

Performance validation:

- Run non-smoke one-hot and dense profiles across the candidate ladders, at
  least:

  ```bash
  AKITA_MODE=onehot_fp32_d128 AKITA_NUM_VARS=32 cargo run --release --example profile
  AKITA_MODE=onehot_fp32_d256 AKITA_NUM_VARS=32 cargo run --release --example profile
  AKITA_MODE=onehot_fp32_d512 AKITA_NUM_VARS=32 cargo run --release --example profile
  AKITA_MODE=onehot_fp64_d64  AKITA_NUM_VARS=32 cargo run --release --example profile
  AKITA_MODE=onehot_fp64_d128 AKITA_NUM_VARS=32 cargo run --release --example profile
  AKITA_MODE=onehot_fp64_d256 AKITA_NUM_VARS=32 cargo run --release --example profile
  ```

- Include dense non-smoke cases, e.g. `dense nv26`, for the same candidate
  families. Also measure dense cross-over candidates:

  ```bash
  AKITA_MODE=dense_fp32_d64 AKITA_NUM_VARS=26 cargo run --release --example profile
  AKITA_MODE=dense_fp64_d32 AKITA_NUM_VARS=26 cargo run --release --example profile
  ```

- As of this PR slice, `onehot_fp32_d64 nv32` and `onehot_fp64_d32 nv32` are
  not final schedule candidates: the root layout cannot find a secure B-row
  rank within the current generated SIS `MAX_RANK=4` table. The dense
  cross-over candidates do verify and should be compared by proof-size/runtime
  objective before blessing defaults.
- Current non-smoke profile observations:

  | Mode | Size | Prove | Verify | Note |
  | --- | ---: | ---: | ---: | --- |
  | `onehot_fp32_d128 nv32` | 44,036 B | 0.667s | 0.039s | verifies |
  | `onehot_fp32_d256 nv32` | 75,560 B | 0.666s | 0.028s | verifies |
  | `onehot_fp32_d512 nv32` | 144,496 B | 0.921s | 0.016s | verifies with D512 stack sampler |
  | `onehot_fp64_d128 nv32` | 80,528 B | 1.160s | 0.072s | verifies |
  | `dense_fp32_d64 nv26` | 31,416 B | 0.777s | 0.026s | dense cross-over candidate |
  | `dense_fp32_d128 nv26` | 40,860 B | 0.263s | 0.007s | verifies |
  | `dense_fp64_d32 nv26` | 34,952 B | 0.623s | 0.018s | dense cross-over candidate |
  | `dense_fp64_d128 nv26` | 76,112 B | 0.441s | 0.008s | verifies |
- Report setup, commit, prove, verify, proof bytes, fold bytes, tail bytes, and
  selected SIS ranks.
- Do not assume adding larger `D` degrades or improves performance globally.
  Larger `D` is a candidate in the planner/search space; if it is not selected,
  it should only cost offline generation/search time. Runtime cost changes only
  for the selected family.

Tests:

- Generated tables cover the declared candidate ladders for Q32/Q64.
- Planner selection can compare candidate dimensions without mixing SIS
  modulus families.
- Regression tests ensure a Q32 schedule never consumes a Q64 or Q128 SIS row,
  and a Q64 schedule never consumes a Q128 SIS row.

### Phase 6C: Planner And Proof-Size Accounting

Extend planner/proof-size inputs with:

- base field byte width, from `F`;
- SIS modulus family, from `F`/`Cfg`;
- ring-dimension candidate family, from `Cfg`;
- opening field extension degree `[E : F]`;
- proof scalar field extension degree `[L : F]`;
- split parameter `t`;
- aggregate incidence shape `(num_claims, num_groups, num_points)`.

Cost model requirements:

- Ring, digit, SIS, commitment, and setup material are priced in base-field
  bytes.
- SIS ranks are selected from the active `SisModulusFamily`, not from a global
  fp128 table.
- Public opening points, claimed values, and proof scalar messages are priced
  in extension-field bytes according to their role (`E` or `L`).
- Shared group material is separated from per-point and per-edge material.
- For the Frobenius route:

  ```text
  extension-domain variables = ell - t
  protocol variables = ell - t + log2([E:F])
  opening width = 2^t
  ```

- `t = 0` is the no-split Frobenius transform, useful as an algebra/control
  case but not the current lifted baseline.
- `t = log2([E : F])` is the full Frobenius-conjugate route.
- The current lifted baseline is priced separately as
  `transformed variables = ell + log2([E:F])` with opening width `1`; do not
  reuse that formula for Frobenius profile estimates.

Tests:

- Golden SIS-rank lookups for Q32, Q64, and Q128 at representative
  `(D, collision_inf, width)` cells.
- Golden outputs for at least one fp32 or fp64 profile across `t`.
- Assertions that larger `t` reduces transformed variables and increases
  same-commitment opening width.
- Regression tests that generated schedule tables and runtime planner fallback
  agree on witness lengths for representative extension configurations.
- Regression tests that compact recursive terminal witness sizing agrees with
  serialized proof bytes after field reduction.

### Phase 7: E2E And CI Hardening

Add positive tests:

- [x] fp32 dense extension-point E2E through packed-inner folded root.
- [x] fp32 dense outer-variable extension-point E2E through root-direct
  fallback.
- [x] fp64 dense extension-point E2E through root-direct fallback.
- [ ] one-hot extension-point E2E. Current profile modes exercise one-hot
  small-field proving/verification; add a focused test before calling this
  complete.
- [x] same-point many-polynomial incidence E2E through root-direct fallback.
- [x] one-group many-point incidence E2E through root-direct fallback, with a
  wrong-claim rejection check.
- [ ] arbitrary incidence E2E.
- [x] Frobenius route E2E with at least one nonzero split.

Add negative tests:

- [ ] transcript reordering fails;
- [x] wrong claim fails for the packed-inner folded and outer-variable
  root-direct extension E2Es;
- [x] wrong conjugate point fails;
- [x] degenerate Moore matrix fails;
- [x] internal Frobenius claim perturbation changes the reconstructed public
  claim.
- [x] full redistribution attack fails: a nonzero internal-claim perturbation
  that preserves the reconstructed public opening is still rejected by the
  same-commitment multipoint proof.

Current profile sanity:

- The profile harness asserts `proof.size()` equals actual uncompressed
  serialization length.
- Runtime proof bytes must match planner `exact_proof_bytes` when a generated
  plan is available.
- Profile verification failures panic instead of becoming log-only warnings.
- `dense_fp32_d128 nv26` has been run through the release profile path with a
  folded proof and compact terminal witness; recent observed size was
  approximately 157 KB rather than the earlier root-direct 256 MiB failure mode.

Required handoff checks:

```bash
cargo fmt -q
cargo clippy --all --all-targets --all-features -- -D warnings
cargo test
RUSTDOCFLAGS="-D warnings" cargo doc -q --no-deps --all-features
```

CI note: the latest completed failing CI run before this update was the
all-features `CI / Test` job at commit `c214e37`, failing
`fp32_ring_subfield_root_fold_roundtrip_uses_extension_gamma` with
`InvalidProof`. The root cause was test-only setup sizing that omitted zk
B-blinding columns, so the prover's matrix multiply did not bind the same
outer columns that verifier replay priced. Test configs that hand-roll
`max_setup_matrix_size` must reserve the same zk outer width as production
configs.

## Primary Live Files

- `crates/akita-types/src/field_reduction.rs` owns the ring-subfield
  encoding, compact digit extraction, and trace dispatch boundary.
- `crates/akita-types/src/proof/{batch,mod,relation}.rs` owns the `F, L` proof
  payload shape and serialized sizes.
- `crates/akita-types/src/schedule.rs` owns the physical recursive W length
  versus terminal witness shape distinction.
- `crates/akita-prover/src/protocol/flow.rs` owns root/recursive materialization
  and terminal witness compaction.
- `crates/akita-verifier/src/{protocol/levels,protocol/ring_switch,stages/stage2}.rs`
  owns replay of `L`-valued proof scalars and base-ring field-reduction checks.
- `crates/akita-scheme/src/lib.rs` validates supported field roles and the
  `F ⊆ E ⊆ L` tower before proving/verifying.
- `crates/akita-config/src/{lib,proof_optimized,sis_policy}.rs`,
  `crates/akita-planner/src/*`, and `crates/akita-types/src/generated/sis_floor.rs`
  own field-family SIS sizing and profile candidate dimensions.
- `crates/akita-pcs/examples/profile/*` owns honest profile timing and proof-size
  reporting.
- `crates/akita-scheme/src/tests.rs`, `crates/akita-pcs/tests/ring_switch.rs`,
  and `crates/akita-pcs/tests/transcript.rs` are the focused regression suites
  for this PR's extension path.

## Review Checklist

- [ ] Generic naming follows `F, E, L` everywhere the field tower is visible.
- [x] No public compatibility aliases preserve the old `AkitaBatchedProof<F>`
      proof type.
- [x] No caller remains for degree-one bridge helpers.
- [x] `gamma`, `batching_coeff`, stage-2 round challenges, `s_claim`, and
      `next_w_eval` are all `L`.
- [x] Ring material remains `F`.
- [x] Public openings remain `E`.
- [x] `F = E = L` fp128 proofs remain semantically unchanged.
- [x] Small-field production presets use non-degree-one public claims and
      challenges (`fp32: E=L=RingSubfieldFp4<F>`, `fp64: E=L=Ext2<F>`).
- [x] Config/unit tests exercise a real `F < E < L` tower.
- [ ] Full prover/verifier E2E exercises a real `F < E < L` tower.
- [ ] CI is green on the final PR head.

## References

- Earliest predecessor (field-role split): `specs/general-field-support.md`
- Predecessor (claim incidence + ClaimField API + extension arithmetic in flow):
  `specs/extension-claim-incidence-cutover.md`
- Companion #71 trace primitive spec: `specs/extension-field-trace-cutover.md`
- Akita field-reduction helpers: `crates/akita-types/src/field_reduction.rs`
- Current verifier claim API: `crates/akita-types/src/proof/scheme.rs`
- Current batch helpers: `crates/akita-types/src/proof/batch.rs`
- Current prover flow: `crates/akita-prover/src/protocol/flow.rs`
- Current verifier orchestration: `crates/akita-verifier/src/protocol/levels.rs`
