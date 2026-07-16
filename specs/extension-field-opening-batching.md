# Spec: PR #71 Part 2 - Extension-Field Opening Completion And Opening-Reduction Cutover

| Field | Value |
| --- | --- |
| Author(s) | Quang Dao |
| Created | 2026-05-06 (originally as the umbrella for PRs #69, #71, and this completion work) |
| Status | baseline extension path, generic multipoint incidence packaging, recursive and root-level tensor-algebra extension-opening reduction, field-family SIS accounting, generated small-field schedule tables, and first small-field prover optimizations have landed in the worktree; the old dense/one-hot Frobenius multipoint route has been removed from the live implementation; root tensor projection now moves the transformed witness to the root commitment boundary for same-width `E = L` small-field roots; remaining work is true-tower E2E coverage, deeper planner tuning, profile reruns, and full CI validation |
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
- Extension root openings with a supported tensor split now commit to the
  tensor-packed root witness and carry a root extension-opening reduction
  proof. Tiny or unsupported root shapes still use the generic direct path
  rather than pretending the transformed opening is available.
- Folded small-field verification now keeps row-evaluation ordering aligned
  across prover and verifier, including the all-features `zk` path.
- The profile example has been split into modules, and small-field profile
  modes now cover explicit fp32/fp64 dense and one-hot candidate dimensions.
- SIS floors are now selected by `SisModulusProfileId::{Q32,Q64,Q128}` and the
  generated registry covers the larger small-field D ladders.
- Sparse challenge sampling has stack-backed tiers through D=512; supported
  larger-D profiles no longer route through a heap-backed fallback.
- The prover/verifier dispatch now uses the same incidence summary to choose
  between folded root proofs and the sound root-direct fallback. Extension
  shapes that the folded path cannot yet price or materialize no longer fail
  by default; root shapes route through direct witnesses when they cannot be
  folded soundly.
- Recursive folded small-field openings now use an extension-opening reduction
  layer. This layer follows the FRI-Binius tensor-algebra formulation of
  Diamond-Posen ring switching, but Akita does not call it "ring switching":
  that name is already reserved for Hachi's lattice/cyclotomic folding
  machinery. The reduction turns a logical extension-field opening claim into
  a single ordinary opening of the committed transformed witness by proving a
  degree-two sumcheck relation against a transparent extension-opening equality
  factor. Frobenius-orbit/multipoint openings are the product-coordinate
  representation of the same tensor object, not the live protocol shape.
- Root-level extension openings now use the same tensor-algebra reduction when
  the claim and challenge fields have the same supported extension width. The
  commitment API transforms the root witness before committing, and the folded
  root proof opens that transformed polynomial at the reduced `rho` point.

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
  landed in #71, including tensor-algebra extension-opening reduction for
  recursive small-field openings. The main remaining work is to harden the
  tests, tune the planner/profile choices for small fields, and decide which
  non-smoke profiles should become CI benches.

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
- **Phase 5B: Frobenius-conjugate base/ext optimization.** Superseded and
  removed from the live implementation. It remains useful as the Hashcaster
  product-coordinate dictionary for understanding the tensor object, but it is
  not a protocol branch, public API, or profile path.
- **Phase 5C: extension-opening reduction.** Live recursive target: keep the
  transformed witness/packing discipline, but replace Hashcaster-style
  Frobenius-orbit carry rows with the FRI-Binius tensor-algebra reduction.
  Bind the logical claim using the tensor object's column view, transpose to
  its row view, then prove one transparent equality-factor sumcheck that
  reduces each logical extension opening to one ordinary opening of the
  transformed committed witness.
- **Phase 6: planner / proof-size and SIS accounting.** Partly landed:
  field-family SIS floors, wider D candidates, compact terminal witness sizing,
  serialized proof-size assertions, grouped incidence keys, challenge-extension
  proof-byte accounting, and profile-time schedule selection are in place.
  Remaining work is selecting and generating production schedule tables rather
  than relying on dynamic/profile search.
- **Phase 7: E2E and CI hardening.** Partly landed: dense extension E2E,
  one-hot small-field profile verification, root-direct incidence fallback
  tests, tensor-reduction negative tests, profile verification, and
  small-prime benchmark CI coverage exist. Remaining tests belong to arbitrary
  incidence hardening, root tensor projection, and a true `F < E < L` tower
  E2E.

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
4. Extension-field openings of base-field-valued committed tables use the
   FRI-Binius tensor-algebra extension-opening reduction before the ordinary
   Hachi fold/opening step, rather than carrying several Frobenius-conjugate
   openings through recursive levels.
5. The planner/proof-size layer can price the tensor-reduced route without
   pretending all scalars are fp128 base elements.

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
- The old Frobenius-conjugate route was a baseline optimization on top of the
  incidence model and has been removed from the live code. Extension-opening
  reduction is the recursive optimized path; neither route should surface as a
  separate public opening API.
- Valid extension-field root openings that fit the tensor-projection boundary
  use the folded root reduction path. Shapes outside that boundary, including
  tiny direct-only roots and future true-tower cases whose width split differs,
  remain on the existing direct path until they get their own explicit design.
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
  multipoint openings, arbitrary point/group routing, and reduced extension
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
- Do not introduce another protocol component named "ring switch",
  "ring-switching", or similar for the base/ext opening mismatch. In Akita,
  "ring switching" means the existing Hachi lattice/cyclotomic relation layer.
  The new layer is an extension-opening reduction.
- Do not implement extension-opening reduction as "batch the Frobenius
  multipoint openings and then open the batch." The intended construction is
  the FRI-Binius tensor-algebra relation for the committed transformed witness:
  reduce the logical extension opening first, then discharge one ordinary
  opening. Frobenius multipoints may appear in explanatory dictionaries or
  legacy tests, but the live cutover should not be organized around them.
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
  consistency | A rows | B rows | D rows
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
  `compute_r_split_eq`, prover `compute_relation_matrix_col_evals`, verifier
  `RingSwitchVerifier`, and relation-claim replay consume the same row-package
  semantics.
- The current `tau1` row combination remains unchanged:

  ```text
  relation_claim = sum_row eq_tau1(row) * eval_alpha(M_row)
  ```

  This is the generic late batching of all relation rows and is separate from
  row-local claim batching.

Remaining hardening:

- [x] Strengthen tests for transcript reordering of row terms or rows.
- [x] Keep row-local coefficient tests close to the incidence package so future
  planner/prover changes cannot accidentally recreate global-gamma semantics.

### Phase 5B: Frobenius-Conjugate Base/Ext Optimization

Status: landed for dense and one-hot base-coefficient workloads and recursive
folded witness packing; generated schedule selection and true-tower E2E remain.
This phase is now the correctness and performance baseline for the optimized
small-field path, but Phase 5C is the intended replacement for recursive
extension-opening payloads.

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

- The Hashcaster/Frobenius transform implementation has been removed from the
  live prover surface. There is no dense or one-hot transform branch, no
  Frobenius opening plan, and no Moore reconstruction helper in the protocol
  path.
- The surviving backend helper is the tensor packer for recursive digit
  witnesses, plus the ring-subfield opening-point adapter used by folded
  extension openings.
- The tensor-algebra extension-opening reduction now owns recursive
  small-field claim reduction: column-view partials are sent, row-view
  batching is transcript-derived, a degree-two sumcheck reduces to `g(rho)`,
  and the ordinary Hachi opening layer discharges that single reduced claim.
- Root-level extension-opening reduction is intentionally not live yet. The
  folded root verifier rejects a root reduction payload until the root path can
  construct, commit, and verify the transformed root witness instead of the
  original public polynomial. Current root behavior is packed-inner folded
  materialization when supported, otherwise root-direct fallback.

Remaining TODOs for this branch:

- Add more true `F < E < L` coverage. The code has the trait tower
  `F ⊆ Fp2 ⊆ Fp4`; production small-field presets currently use `E = L`, so a
  full prover/verifier E2E with strict inclusions is still a separate
  hardening target. Current blocker: the prover/verifier requires both `E` and
  `L` to implement `RingSubfieldEncoding<F>`. The strict tower available today
  uses tower-basis `Fp4` for `L`, and that basis is intentionally not a
  ring-subfield encoding. Completing this requires an explicit basis/encoding
  decision, not only a test.
- Freeze generated planner/profile inputs for the tensor-reduced route. The
  transformed recursive workload has:

  ```text
  extension-domain variables = ell - kappa
  protocol variables = ell - kappa + log2([E:F])
  carried opening width = 1
  tensor partial count = [E:F]
  ```

  Planner and profile output report tensor partial bytes, reduction sumcheck
  bytes, and the single reduced carried row separately from the ordinary Hachi
  fold payload. Generated schedule materialization includes the extension
  opening-reduction proof bytes in `exact_proof_bytes`.
- **Fallback boundary.** Keep root-direct fallback as the sound default for
  extension root shapes outside the optimized folded route, but do not
  introduce a second Frobenius fallback path. The latest native small-field
  profiles show this boundary is now the dominant proof-size blocker.
- **Generated schedules.** Production generated tables are now family-bound by
  SIS modulus and reject mismatches. Small-field generated coverage is baked
  for separate full-field and one-hot configs: fp32 `D = 64,128,256,512` and
  fp64 `D = 32,64,128,256`, each with non-ZK and ZK variants through
  `nuposition_bits <= 32` for singleton and same-point `np=4` incidence shapes. fp32
  `D=32` remains tuning-only because strict generated-table validation exposed
  recursive terminal layouts that cannot be materialized under the current
  rank-4 SIS floors. SIS A-role collision pricing includes the `psi`
  ring-subfield embedding norm bound: factor `1` for the base-field `K=1`
  coefficient-packing path and factor `2` for the current small-field `K>1`
  embeddings.

Negative tests:

- Wrong original claim fails.
- Wrong tensor partial fails.
- Wrong reduction sumcheck final oracle fails.
- Degree-one proofs reject unexpected extension-opening reduction payloads.

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

### Phase 5C: Extension-Opening Reduction

Status: first plumbing slices landed in the worktree. Folded root and recursive
level proofs now have direct extension-opening-reduction payload slots, profile
proof-byte reporting accounts for them, and the degree-two
extension-opening-reduction sumcheck helper exists in `akita-sumcheck`.
Transparent reduction factors currently have a dense table/evaluation API, and
verifier-side round replay can return the final `rho`/claim without requiring
witness evaluations. The current Frobenius/Moore helper is a bridge from the
landed Phase 5B baseline, not the final protocol contract. The cutover target
is the FRI-Binius tensor-algebra formulation below.

This phase replaces the recursive use of Frobenius-conjugate multipoint
openings with a direct extension-opening reduction. It follows the
Diamond-Posen FRI-Binius reduction at the protocol-object level, while keeping
Akita terminology distinct:

- **Hachi ring switching** remains the existing lattice/cyclotomic fold
  relation, implemented by the stage-1/stage-2 machinery and quotient rows.
- **Extension-opening reduction** is the new layer that handles the
  base-field-committed / extension-field-opened mismatch before ordinary Hachi
  folding.
- **Frobenius multipoint openings** are the Hashcaster/product-coordinate
  representation of the same tensor-algebra object. They are useful as a
  dictionary and as the landed baseline, but they should not define the live
  cutover API.

#### Tensor-Algebra Model

For the mathematical reduction, specialize Diamond-Posen notation as:

```text
K = F                         ground coefficient field
M = E                         public opening / packed-value field
[M : K] = m = 2^kappa
```

The proof-scalar field `L` is used for Fiat-Shamir challenges and sumcheck
messages. When `L != E`, the same identities are evaluated after scalar
extension along `E -> L`; do not change the underlying packing map.

Fix a deterministic `F`-basis `(beta_v)_{v in B_kappa}` of `E`. For a
base-field polynomial `f(X_head, X_tail)` with `|X_head| = kappa`, define the
packed transformed polynomial:

```text
g(X_tail) = sum_{v in B_kappa} f(v, X_tail) * beta_v.
```

The tensor algebra is:

```text
A_E = E tensor_F E.
```

It has two canonical embeddings:

```text
phi0(a) = a tensor 1
phi1(a) = 1 tensor a
```

and each element has two useful representations:

```text
column view: S = sum_v S_v tensor beta_v
row view:    S = sum_u beta_u tensor T_u
```

For one logical opening claim:

```text
f(r_head, r_tail) = y,       r_head in E^kappa, r_tail in E^ell'
```

the prover's short message is the tensor partial object:

```text
S = phi1(g)(phi0(r_tail)) in A_E.
```

In implementation, the prover sends the column representation:

```text
partials[v] = S_v = f(v, r_tail),  v in B_kappa.
```

The verifier first checks the logical claim by multilinear recombination:

```text
y == sum_v eq(v, r_head) * partials[v].
```

This check is the reason to stay on the tensor side. The lossy strawman would
instead check only `sum_v beta_v * partials[v] == g(r_tail)`, but
`(beta_v)` is an `F`-basis, not an `E`-basis; combining `E`-valued partials
against `beta_v` is not injective.

The verifier then derives the row representation:

```text
S = sum_u beta_u tensor row_partials[u].
```

For the chosen basis this is a deterministic linear transpose of the sent
column representation. This should be implemented as an explicit tensor
transpose/basis operation, matching Binius64's `TensorAlgebra::transpose()`,
not as a Frobenius-orbit opening step.

#### Reduction Sumcheck

For each Boolean tail index `w`, decompose the tail equality factor:

```text
eq(r_tail, w) = sum_u A_u(w) * beta_u.
```

After the verifier samples row-batching challenges `eta in L^kappa`, define:

```text
eta_u = eq(u, eta)
A_eta(w) = sum_u eta_u * A_u(w)
c_eta = sum_u eta_u * row_partials[u].
```

The reduction sumcheck proves:

```text
sum_{w in B_ell'} g(w) * A_eta(w) = c_eta.
```

At the end of the sumcheck, with challenge point `rho`, the verifier checks:

```text
sumcheck_final = g(rho) * A_eta(rho).
```

`g(rho)` is not a separate proof system. It is discharged through the ordinary
single-point Hachi folded opening at `rho`, using the existing Hachi
ring-switch relation machinery after this reduction has collapsed the logical
extension claim.

The transparent factor `A_eta` should have two implementations:

- **Reference implementation:** materialize/evaluate dense `A_eta` tables using
  the existing `ExtensionOpeningReductionFactor` boundary.
- **Optimized implementation:** compute the tensor equality indicator directly
  from `phi0(r_tail)`, `phi1(rho)`, and `eta`, following the FRI-Binius/Binius64
  pattern. This avoids materializing one equality table per Frobenius point.

#### Frobenius Dictionary

When `E/F` is finite Galois, there is a canonical product-side isomorphism:

```text
E tensor_F E  ~=  product_{sigma in Gal(E/F)} E
a tensor b    |-> (sigma(a) * b)_sigma.
```

For finite fields, `Gal(E/F)` is generated by Frobenius. Under this
isomorphism:

```text
phi0(r_tail)        |-> (sigma^j(r_tail))_j
phi1(g)             |-> g in every component
S                   |-> (g(sigma^j(r_tail)))_j
```

up to the chosen Frobenius direction convention.

Therefore the Phase 5B Frobenius multipoint vector:

```text
s_j = g(r_tail^(q^j))
```

is not a different algebraic idea. It is the product-coordinate view of the
same tensor object `S`. Hashcaster operates on this product side and then
batches the orbit openings. FRI-Binius stays on the tensor side because:

- the logical claim check lives naturally in the column view;
- the sumcheck claims live naturally in the row view;
- the verifier avoids an explicit Frobenius/Galois matrix transform;
- the prover avoids constructing equality tensors for every orbit point.

The Frobenius/Moore helper has been removed from the live implementation.
Future diagnostics may compare against the product-coordinate dictionary, but
protocol code should stay on the tensor partial / tensor transpose
formulation.

#### Proof Payload

The proof payload should use one named optional object, not several loose
fields and not an object-free generic "reduction" wrapper. The option is
absent for degree-one/base-field openings and for paths that have not opted
into extension-opening reduction, so those proof wires pay zero bytes:

```rust
extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>

struct ExtensionOpeningReductionProof<L> {
    /// Column-view tensor partials `S_v = f(v, r_tail)`, lifted to `L`.
    partials: Vec<L>,
    /// Sumcheck proof for `sum_w g(w) * A_eta(w) = c_eta`.
    sumcheck: SumcheckProof<L>,
}
```

Serialization should be headerless like the existing sumcheck payloads: no
presence tag or vector length is sent inside the proof. The verifier's expected
field/schedule/shape determines whether the optional payload is present and how
many partials and sumcheck rounds to read. Exact names may change with the
codebase, but they should identify the polynomial/opening being reduced. Avoid
object-free reduction names, and avoid names containing `ring_switch`.

#### Recursive-Layer Cutover

1. The recursive state still carries the logical extension opening point and
   claim.
2. The prover sends the column-view tensor partials `S_v = f(v, r_tail)`.
3. The verifier checks those partials against the logical claim, absorbs them
   into the transcript, derives the row-view partials, then samples `eta`.
4. Prover and verifier run the degree-two sumcheck for
   `g(w) * A_eta(w)`.
5. The verifier replays the reduction sumcheck rounds to recover `rho` and the
   final claim, evaluates the transparent factor `A_eta(rho)`, and checks it
   against the recovered single value `g(rho)`.
6. Recursive public-row accounting uses one carried opening row, not
   `2^kappa` Frobenius rows.

#### Root-Layer Cutover

Status: implemented for the same-width small-field tower `F < E = L` when the
root has enough variables to preserve the physical root arity after packing.
Public verifier inputs remain logical `(poly/group, point, value)` claims. The
commitment boundary now transforms each root witness into the tensor-packed
tail polynomial `g` before committing whenever the selected proof schedule uses
a folded root and the tensor split is supported. The root prover binds the
logical claim through the column-view tensor partials, runs the degree-two
extension-opening reduction, then opens the committed transformed root at the
packed `rho` point inside the ordinary Hachi root fold.

The important soundness boundary is that the final single opening is not an
opening of the original root commitment. The reduction sumcheck speaks about
the tensor-packed tail witness `g`, while the original base-field witness is
`f`. Therefore the transformed root witness must be the committed witness used
by `QuadraticEquation::new_prover`; merely attaching a reduction payload to an
old root commitment is not sound.

1. Same-point and same-group batching should combine transparent tensor
   equality factors with row-local coefficients, so the root folded path still
   proves ordinary Hachi rows after reduction.
2. For a row with terms `(claim c, coeff gamma_c)`, the transparent factor is
   the corresponding row-local linear combination:

   ```text
  A_row(rho) = sum_c gamma_c * A_{eta_c}(rho)
  ```

   and the Hachi opening layer proves the matching row-combined
   `g_c(rho)` relation using the existing claim/group routing.

#### Planner And Proof-Size Accounting

- Add separate counters for logical claims, tensor partials, reduction
  sumcheck rounds, reduced single-point openings, public rows, and recursive
  carry rows.
- The previous Frobenius multipoint estimate treated the reduced path as
  `2^kappa` carried openings plus no reduction proof. Phase 5C should instead
  price one carried opening plus the tensor partials and degree-two sumcheck.
- Schedule keys must distinguish the number of reduced opening rows from the
  number of logical claims and from the number of committed groups.
- For `F = E = L`, extension-opening reduction is disabled; the degree-one
  path remains the ordinary single-point Hachi opening.

#### Soundness Requirements

- The extension basis / packing map used by the tensor object must be explicit
  and deterministic for each supported field family.
- The sent tensor partials are transcript-bound before `eta` is sampled.
- The verifier must not accept arbitrary extension-valued partials that are not
  checked against the logical opening claim.
- The tensor transpose from column view to row view must be deterministic and
  covered by tests for each supported extension family.
- Reordering logical claims, tensor partials, public rows, or row-local terms
  must change the transcript.
- Wrong logical claim, wrong extension point, wrong tensor partials, wrong
  transparent factor, and wrong final single-point opening must all fail.

### Phase 6A: Field-Family SIS Floor Registry

Status: landed in PR #71 Part 2.

The current SIS floor registry is calibrated for an fp128 representative
modulus and is keyed only by `(D, collision_inf, width)`. That is not a valid
abstraction once `F` can be fp32 or fp64: the maximum secure width for a fixed
rank and collision bound depends on `q`. Reusing the fp128 table for small
fields can under-size ranks and overstate binding security.

Introduce an explicit SIS modulus-family policy:

```rust
pub enum SisModulusProfileId {
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

- [x] Add a config hook such as `CommitmentConfig::sis_modulus_profile()` and mirror
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
- [x] Generated schedule tables must record or be generated under the same family
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

Status: landed for generated production small-field tables, with separate
full-field and one-hot config families where the one-hot configs use
`log_commit_bound = 1` and keep the opening bound at the underlying field
width.

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

- `Q128`: keep generated schedules only for `D in {32, 64}`.
- `Q64`: keep generated schedules only for `D in {32, 64}`.
- `Q32`: keep generated schedules for `D in {32, 64}`. `D=32` is the
  proof-size default after extending the Q32/D32 SIS floor through rank 20;
  `D=64` remains an explicit runtime comparison path.
- D128-or-larger generated schedules are intentionally removed across prime
  families because measured proof sizes are worse than the smaller-D tables.

Implementation requirements:

- [x] Add proof-optimized preset structs for the new
  small-field ring dimensions, without disturbing the existing fp128 preset
  names.
- [x] Extend the schedule-table generator so family specs are not hardcoded to
  fp128 `D32/D64/D128`.
- [x] Extend SIS generation for Q32/Q64 to cover the supported generated
  schedule buckets above. Q32/D32 is generated through rank 20 so folded fp32
  D32 schedules materialize at the canonical dense and one-hot sizes.
- [x] Sparse challenge samplers must have explicit fast paths for each supported
  generated ring dimension.
- [x] Keep runtime profile mode names explicit, for example
  `onehot_fp32`, `onehot_fp32_d32`, and `onehot_fp32_d64`, so profile output
  makes the selected ring dimension unambiguous. The unsuffixed
  `onehot_fp32` mode aliases the D32 proof-size default.
- Selection helpers may choose the best generated schedule by proof bytes, but
  the profile report must still print timings for each candidate family before
  we bless a default.

Performance validation:

- Run non-smoke one-hot and dense profiles across the candidate ladders, at
  least:

  ```bash
  AKITA_MODE=onehot_fp32      AKITA_NUM_VARS=32 AKITA_NUM_POLYS=4 cargo run --release --example profile
  AKITA_MODE=onehot_fp32_d64  AKITA_NUM_VARS=32 AKITA_NUM_POLYS=4 cargo run --release --example profile
  AKITA_MODE=onehot_fp64_d32  AKITA_NUM_VARS=32 cargo run --release --example profile
  AKITA_MODE=onehot_fp64_d64  AKITA_NUM_VARS=32 cargo run --release --example profile
  ```

- Include dense non-smoke cases, e.g. `dense nv26`, for the same candidate
  families. Also measure dense cross-over candidates:

  ```bash
  AKITA_MODE=dense_fp32_d32 AKITA_NUM_VARS=26 cargo run --release --example profile
  AKITA_MODE=dense_fp32_d64 AKITA_NUM_VARS=26 cargo run --release --example profile
  AKITA_MODE=dense_fp64_d32 AKITA_NUM_VARS=26 cargo run --release --example profile
  AKITA_MODE=dense_fp64_d64 AKITA_NUM_VARS=26 cargo run --release --example profile
  ```

- After splitting full-field and one-hot small-field configs, the lower-D
  one-hot candidates materialize under the generated SIS floor table. The
  profiler confirms the selected schedule byte estimates exactly for the
  measured shapes below.
- Current release profile observations after extending Q32/D32 SIS floors and
  regenerating fp32 D32 schedules:

  | Mode | Setup | Commit | Prove | Verify | Proof bytes | Fold bytes | Tail bytes |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | `dense_fp32_d32 nv26 np1` | 0.171s | 6.933s | 7.026s | 0.026s | 39,968 | 20,896 | 19,072 |
  | `dense_fp32_d64 nv26 np1` | 0.072s | 1.369s | 6.453s | 0.021s | 44,128 | 21,952 | 22,176 |
  | `onehot_fp32 nv32 np4` | 4.346s | 4.911s | 56.145s | 0.094s | 41,824 | 21,872 | 19,952 |
  | `onehot_fp32_d64 nv32 np4` | 1.561s | 4.472s | 39.633s | 0.077s | 47,264 | 23,408 | 23,856 |

  D32 is therefore the fp32 proof-size default, while D64 remains faster on
  these runtime profiles. The generic small-field prover-opening optimization
  work is tracked separately and should apply to both D32 and D64 rather than
  being reimplemented in this schedule PR.
- Report setup, commit, prove, verify, proof bytes, fold bytes, tail bytes, and
  selected SIS ranks.
- Do not assume adding larger `D` degrades or improves performance globally.
  Larger `D` is a candidate in the planner/search space; if it is not selected,
  it should only cost offline generation/search time. Runtime cost changes only
  for the selected family.

Tests:

- Generated tables cover the declared candidate ladders for Q32/Q64, including
  Q32 D32/D64 full and one-hot tables.
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
- SIS ranks are selected from the active `SisModulusProfileId`, not from a global
  fp128 table.
- SIS A-role collision bounds include the `psi` ring-subfield embedding norm
  bound. For `K=1`, `psi` is coefficient packing and the factor is `1`; for
  the current small-field ring-subfield embeddings, the conservative bound is
  `2`.
- Public opening points, claimed values, and proof scalar messages are priced
  in extension-field bytes according to their role (`E` or `L`).
- Shared group material is separated from per-point and per-edge material.
- Historical/product-coordinate Frobenius estimates may still be useful for
  comparison, but they are no longer a protocol branch:

  ```text
  extension-domain variables = ell - t
  protocol variables = ell - t + log2([E:F])
  opening width = 2^t
  ```

- `t = 0` is the no-split product-coordinate transform, useful as an
  algebra/control case but not the current lifted baseline.
- `t = log2([E : F])` is the full product-coordinate/Frobenius dictionary
  case.
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
- [x] fp32 dense outer-variable extension-point E2E through root tensor
  projection.
- [x] fp64 dense extension-point E2E coverage; profile rerun after root
  projection remains pending.
- [x] one-hot extension-point E2E coverage; profile rerun after root
  projection remains pending.
- [x] same-point many-polynomial incidence E2E through root tensor projection.
- [x] one-group many-point incidence E2E through root tensor projection, with
  a wrong-claim rejection check.
- [ ] arbitrary incidence E2E.
- [x] Recursive tensor extension-opening reduction E2E/prover-boundary
  coverage with non-power-of-two logical witness padding.

Add negative tests:

- [x] transcript row/term reordering fails by transcript divergence;
- [x] wrong claim fails for packed-inner folded and root tensor-projection
  extension E2Es;
- [x] wrong tensor partial / reduction sumcheck / final oracle fails;
- [x] unexpected root or degree-one extension-opening reduction payload fails.

Current profile sanity:

- The profile harness asserts `proof.size()` equals actual uncompressed
  serialization length.
- Latest native small-field release profiles before root tensor projection
  showed unsupported public root shapes falling back to direct witnesses:

  | Mode | Setup | Commit | Prove | Verify | Proof bytes | Notes |
  | --- | ---: | ---: | ---: | ---: | ---: | --- |
  | D128+ small-field candidates | n/a | n/a | n/a | n/a | worse than D32/D64 | generated schedules removed |

  These are retained only as the pre-root-projection failure baseline. The
  current implementation has cut root projection over for supported same-width
  small-field roots; release profiles need to be rerun before replacing the
  PR's recorded benchmark table.
- Runtime proof bytes must match planner `exact_proof_bytes` when a generated
  plan is available.
- Profile verification failures panic instead of becoming log-only warnings.
- Recent release profiles showed the D128+ small-field candidates losing on
  proof size, so the generated schedule surface is now limited to D32/D64.

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
- `crates/akita-sumcheck/src/extension_opening_reduction.rs` owns the
  reference degree-two extension-opening reduction sumcheck boundary.
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
- Diamond and Posen, `Diamond_Posen_FRI_Binius_2024_504.pdf`, ePrint
  2024/504: FRI-Binius tensor-algebra ring-switching compiler and Hashcaster
  comparison.
- Akita field-reduction helpers: `crates/akita-types/src/field_reduction.rs`
- Current verifier claim API: `crates/akita-types/src/proof/scheme.rs`
- Current batch helpers: `crates/akita-types/src/proof/batch.rs`
- Current prover flow: `crates/akita-prover/src/protocol/flow.rs`
- Current verifier orchestration: `crates/akita-verifier/src/protocol/levels.rs`
