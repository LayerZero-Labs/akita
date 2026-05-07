# Spec: Extension-Field Openings and Batched Claim Incidence

| Field | Value |
| --- | --- |
| Author(s) | Quang Dao |
| Created | 2026-05-06 |
| Status | living target |
| PR | follow-up to #60 |
| Depends on | `specs/general-field-support.md` |

## Summary

Akita should support native small-field commitments opened at extension-field
points. The motivating case is a polynomial with coefficients in `Cfg::Field`
evaluated at a point in `Cfg::ClaimField = F_{q^k}` for sumcheck soundness.
The current base-field-only public claim shape cannot express the optimized
route cleanly, because the optimization opens one committed transformed
polynomial at many Frobenius-conjugate points. This feature migrates opening
points and claimed evaluations to `Cfg::ClaimField`, introduces one canonical
point/group/claim incidence model for batching, implements the real Hachi
`k > 1` field embedding and trace relation, and adds the optimized
base-coefficient / extension-point opening path.

This spec is the target architecture for the native small-field commit/opening
sequence.
It also tracks already-landed scaffolding and branch-local implementation
progress so the remaining cutover stays grounded in the code.

## Intent

### Goal

Implement extension-field opening claims for Akita by migrating the existing
prove/verify surfaces to `Cfg::ClaimField` and replacing the nested batching
input with a point/group/claim incidence model that supports same-point
batching, same-commitment multipoint openings, and Frobenius-conjugate
base/ext openings through one public representation.

### Current State

Completed scaffolding from `specs/general-field-support.md`:

- `CommitmentConfig` already has distinct `Field`, `ClaimField`, and
  `ChallengeField` roles.
- Config helpers already centralize claim-field transcript absorption and
  challenge-field sampling through extension-aware transcript helpers.
- `akita-transcript` already has coordinate-wise extension absorption and
  extension challenge sampling.
- `akita-types::field_reduction` already has reference `SubfieldParams`,
  `trace_h`, and `psi_pack` helpers, but these are not yet production proof-path
  embedding.
- Planner and schedule helpers already accept field-width inputs for fp32/fp64
  scaffolding, while production fp128 remains the degree-one path.

Progress from the small-field proof worktree:

- `Fp2` is now a transparent `[F; 2]` coefficient container with `c0()` and
  `c1()` accessors, and `Fp2Config::mul_non_residue` gives scalar and packed
  lanes the same non-residue hook.
- Quartic fields have been split into explicit `TowerBasisFp4` and
  `PowerBasisFp4` representations.
- Power-basis and tower-basis quartic arithmetic, packing, transcript limb
  order, conversion tests, and packed multiplication kernels have been added.
- `ExtField`, `LiftBase`, and `MulBase` cover the base field, `Fp2`,
  `TowerBasisFp4`, and `PowerBasisFp4`.
- The canonical extension-limb order is univariate coefficient order.
  `TowerBasisFp4::from_base_slice([c0, c1, c2, c3])` stores internally as
  `(c0, c2), (c1, c3)`, but `ExtField::to_base_vec` returns
  `[c0, c1, c2, c3]` for transcript and serialization callers.
- The old `Pow2Offset*Field` naming and implicit per-width offset table have
  been replaced by registered `(bits, offset)` prime specs and explicit
  `Prime*Offset*` aliases, including the full-word `Prime64Offset59` path.
- The field arithmetic benchmark has been split into focused modules covering
  base, wide, `Fp2`, tower quartic, power quartic, packed, and parallel
  throughput cases.
- Sparse challenges, ring evaluation, relation helpers, and ring-switch
  prover/verifier internals have been generalized over a mixed base field
  `F` and extension field `E`.
- The live prove/verify orchestration still instantiates those generic
  folded root internals with degree-one claim scalars.
  Public `AkitaCommitmentScheme` prover/verifier claims are now cut over to
  `Cfg::ClaimField`, and true extension-valued folded roots are rejected until
  the stage-2 relation and ring-switch path are wired end-to-end over `E`.

Current main-line API facts that this spec must change:

- `OpeningPoints<'a, F>`, `CommittedOpenings<'a, F, C>`,
  `VerifierClaims<'a, F, C>`, and `ProverClaims<'a, F, P, C, H>` are still
  generic over one public claim scalar, and the concrete scheme now instantiates
  that scalar as `Cfg::ClaimField`.
- Base-field profile examples, setup tests, and current E2E benches explicitly
  constrain their configs to the degree-one `ClaimField = Field` specialization
  because those harnesses still construct base-field points and openings.
- Root batching now consumes `ClaimIncidenceSummary` directly; the old
  `MultiPointBatchShape` adapter has been removed from crate-level APIs.
- Root opening preparation and ring-native opening points are still over base
  field scalars.
- The extension-valued relation helper exists as scaffolding, but the live
  stage-2 relation still uses the base-field relation.

### Invariants

- Ring commitments, setup matrices, recursive witnesses, digit decomposition,
  CRT/NTT work, and SIS bounds remain over `Cfg::Field`.
- Public opening points and claimed evaluations are over `Cfg::ClaimField`.
- Fiat-Shamir scalar challenges that need extension-field soundness are sampled
  over `Cfg::ChallengeField`; base-ring transcript absorption still uses the
  base transcript field `Cfg::Field`.
- The fp128 production path remains the degree-one specialization
  `Field = ClaimField = ChallengeField`; no compatibility wrappers or parallel
  legacy APIs survive the cutover.
- One public claim representation covers:
  - one point, many committed polynomials;
  - one committed group, many points;
  - many points and many groups with arbitrary matching;
  - Frobenius-conjugate openings used by the base/ext optimization.
- The incidence model never forces callers to duplicate commitments, prover
  hints, or polynomial slices solely because one group is opened at multiple
  points.
- Transcript binding absorbs the full incidence shape: points, groups,
  commitments, claim routing, and claimed evaluations. Reordering points,
  groups, or claims changes the transcript unless the normalized representation
  explicitly canonicalizes that ordering.
- Prover and verifier derive identical flattened schedule quantities from the
  incidence graph: point count, group count, total claim count, claim-to-point
  routing, claim-to-group routing, per-group polynomial counts, and per-point
  claim counts.
- The generic extension-valued opening path uses the real Hachi fixed-subfield
  embedding for `k > 1`; the existing coefficient embedding remains the `k = 1`
  degeneration.
- The optimized base-coefficient / extension-point path does not use the
  literal Hachi partial-evaluation optimization that checks only one fixed
  linear relation among prover-supplied extension-valued partial evaluations.
- Wrong claimed values, wrong conjugate points, invalid Moore systems, and
  redistribution attempts are rejected.
- Existing same-point batching and current multipoint behavior remain correct
  after they are expressed as derived views of the incidence model.

New invariant tests should live close to the layer they protect:

- claim-shape validation and transcript binding in `akita-types` or
  `akita-pcs/tests/transcript.rs`;
- Hachi embedding and trace algebra in `akita-types/src/field_reduction.rs`;
- prover/verifier extension-opening consistency in `akita-pcs/tests`;
- planner tradeoff checks in `akita-planner` or `akita-config` tests.

### Non-Goals

- This does not introduce separate public APIs for each batching special case.
- This does not keep base-field-only public aliases after the cutover; this repo
  is full-cutover only.
- This does not require every execution-sharing optimization to land in the
  first implementation. The incidence model must support those optimizations
  without additional public API churn.
- This does not tune production fp32/fp64 schedule tables unless explicitly
  included in the implementation PR.
- This does not replace the default fp128 security parameters.
- This does not implement the unsound literal Hachi base-field optimization
  based on unbound extension-valued partial evaluations.
- This does not add a separate ring-switching sumcheck for the base/ext
  optimization; the intended trade is wider same-commitment multipoint opening
  versus fewer transformed variables.

## Evaluation

### Acceptance Criteria

Completed groundwork already available before the final cutover:

- [x] `CommitmentConfig` exposes distinct `Field`, `ClaimField`, and
  `ChallengeField` roles.
- [x] Config helpers can append claim-field values and sample challenge-field
  values through extension-aware transcript helpers.
- [x] Extension transcript helpers preserve degree-one transcript behavior and
  use coordinate-wise limb labels for higher-degree fields.
- [x] Reference Hachi field-reduction helpers exist for subgroup validation,
  `trace_h`, and coefficient-placement `psi_pack`.
- [x] Field-width-aware planner and schedule scaffolding exists for fp32/fp64
  experiments.
- [x] Explicit quartic representations, packed extension kernels, and
  mixed-field ring-switch scaffolding are present on this worktree.
- [x] `Fp2` uses transparent coefficient-array storage and both quadratic
  non-residue configs route scalar and packed multiplication through a shared
  `mul_non_residue` hook.
- [x] `TowerBasisFp4` and `PowerBasisFp4` have separate scalar and packed
  representations, with conversion and multiplication-agreement tests.
- [x] `ExtField::from_base_slice` and `ExtField::to_base_vec` define one
  canonical univariate limb order for base, quadratic, tower quartic, and power
  quartic extension elements.
- [x] `MulBase` lets mixed-field hot paths scale extension elements by base
  scalars without materializing a lifted base-field element first.
- [x] Packed field hooks cover `Fp2`, tower-basis quartic, and power-basis
  quartic multiplication, including NEON-backed small-field lanes.
- [x] Pseudo-Mersenne small-field presets use explicit registered
  `(bits, offset)` specs and `Prime*Offset*` aliases rather than an implicit
  power-of-two offset table.
- [x] Public prover and verifier traits expose an associated `ClaimField`
  separate from the base transcript/commitment field.
- [x] Shared batched-claim validation accepts opening-point coordinates from a
  field distinct from the base setup field.
- [x] Verifier claim preparation and root-direct witness checks accept
  extension-valued opening points and claimed evaluations.
- [x] Prover claim preparation accepts extension-valued opening points while
  keeping base-field commitments and hints.
- [x] Prover-side incidence group scaffolding attaches polynomial slices and
  hints to verifier-visible group metadata.
- [x] Normalized incidence summaries replace the legacy point-local
  batch-shape adapter in root batching.
- [x] Verifier claim inputs normalize into canonical incidence graphs while
  preserving the current grouped batch layout.
- [x] Verifier claim preparation now emits incidence summaries directly.
- [x] Prover claim preparation now emits incidence summaries directly.
- [x] Prepared prover/verifier claim views carry `ClaimIncidenceSummary`, and
  schedule lookup derives aggregate `(K, G, P)` counts directly from incidence
  rather than from the temporary legacy batch-shape adapter.
- [x] Root-direct verifier opening checks consume `ClaimIncidenceSummary`
  routing directly.
- [x] Folded-root prover and verifier trace/ring-switch routing use
  `ClaimIncidenceSummary::claim_to_point`; quadratic and M-table rows use
  incidence-derived group counts and claim-to-group/poly routing.

Public API and claim model:

- [x] Public prover claim inputs accept opening points as
  `&[Cfg::ClaimField]`.
- [x] Public verifier claim inputs accept opening points as
  `&[Cfg::ClaimField]`.
- [x] Public claimed evaluations use `Cfg::ClaimField`.
- [x] Commitment, setup, and ring proof objects remain over `Cfg::Field`.
- [ ] The old base-field-only public claim aliases are removed in the cutover.
- [x] A normalized incidence model represents distinct points, distinct
  committed groups, and individual claims.
- [x] The incidence model supports one committed group opened at multiple
  points without duplicating commitment or hint input.
- [ ] Same-point batching is represented as the special case with one point and
  many claims.
- [x] Existing multipoint batching is represented as the same incidence graph,
  and `MultiPointBatchShape` has been removed from public/protocol-facing claim
  flow.
- [x] Claim-shape validation rejects empty point sets, empty group sets, invalid
  point indices, invalid group indices, invalid polynomial indices, dimension
  mismatches, and setup-capacity overflows.

Transcript and serialization:

- [x] Transcript absorption includes the normalized claim incidence shape as a
  migration bridge, not as proof payload.
- [ ] Remove the separate incidence-shape transcript append once canonical
  public claim absorption binds the same point/group/claim routing.
- [x] Folded-root transcript absorption uses extension-aware claim-field
  helpers for public opening points and claimed evaluations in the current
  degree-one path.
- [ ] Full config-backed public claim absorption binds the same point/group/claim
  routing without the separate incidence-shape migration bridge.
- [x] Reordering claim-edge routing transcript-diverges unless the
  implementation explicitly canonicalizes that order first.
- [ ] Add end-to-end transcript tests covering point and group ordering once
  incidence drives the live root batching flow.
- [ ] Degree-one claim fields preserve current transcript behavior for fp128.
- [ ] Serialization/proof structs remain unambiguous about whether field
  elements are base-field or claim-field elements.

Generic extension-valued openings:

- [ ] Live prove/verify orchestration instantiates ring-switch internals with
  `Cfg::ChallengeField` where extension-field soundness is required, instead of
  collapsing the generic path to `E = F`.
- [ ] The live stage-2 relation uses the extension-valued relation when
  `Cfg::ChallengeField != Cfg::Field`.
- [ ] The proof path implements the Hachi fixed-subfield embedding for
  `k > 1`.
- [ ] The `k = 1` path remains the existing coefficient embedding shortcut.
- [ ] The trace relation verifies the packed inner product for extension-valued
  claims.
- [ ] Norm accounting documents and tests the `k = 1` no-blowup case and the
  `k > 1` Hachi subfield-basis blowup used by the implementation.
- [ ] Invalid extension degrees, invalid ring dimensions, and invalid subgroup
  parameters are rejected before proving or verification.

Optimized base-coefficient / extension-point openings:

- [ ] Base-field polynomial backends can be opened at
  `Cfg::ClaimField` points.
- [ ] The implementation supports a split parameter `t`.
- [ ] For split `t`, the prover forms base slices `f_h` and the extension-valued
  tail polynomial `g = sum_h theta_h f_h`.
- [ ] The prover opens the same committed transformed polynomial at
  Frobenius-conjugate tail points.
- [ ] The verifier checks the Moore-system binding of slice evaluations.
- [ ] The verifier checks the original claimed value
  `sum_h lambda_h(x_head) f_h(x_tail)`.
- [ ] The implementation rejects non-independent `theta_h` choices or any
  degenerate Moore matrix.
- [ ] The optimized path does not introduce an extra ring-switching sumcheck.

Tests and regressions:

- [ ] fp32 base-field dense polynomial opened at an `Fp2`, `TowerBasisFp4`, or
  `PowerBasisFp4` point passes commit/prove/verify.
- [ ] fp64 base-field dense polynomial opened at an extension point passes
  commit/prove/verify.
- [ ] One-hot polynomial opened at an extension point passes commit/prove/verify.
- [ ] Same-point many-polynomial batching still passes.
- [ ] One-committed-group many-point opening passes without duplicating group
  inputs.
- [ ] Arbitrary incidence graph with multiple points and multiple groups passes.
- [ ] Wrong claimed value rejection test fails verification.
- [ ] Wrong Frobenius-conjugate point rejection test fails verification.
- [ ] Redistribution-attack regression fails verification.
- [ ] Transcript-reordering regression fails verification.
- [ ] Planner/proof-size tests cover the split-parameter tradeoff.
- [x] Unit tests cover extension-field degrees, tower/power quartic conversion,
  canonical transcript limb order, extension array layout, `MulBase`
  equivalence, and packed extension arithmetic.
- [x] Registry and primality tests cover the explicit `Prime*Offset*` field
  aliases used by the fp32/fp64 scaffolds.
- [x] Ring-switch, direct-opening, and transcript tests have been updated to
  build through `ClaimIncidenceSummary` rather than `MultiPointBatchShape`.

Compatibility and CI:

- [ ] Existing fp128 E2E tests pass.
- [ ] Existing batched and multipoint E2E tests pass after the incidence cutover.
- [ ] `cargo fmt -q` passes.
- [ ] `cargo clippy --all --all-targets --all-features -- -D warnings` passes.
- [ ] `cargo test` passes.
- [ ] GitHub CI is green.

### Testing Strategy

Existing tests that must continue passing:

- all fp128 tests in `crates/akita-pcs/tests/akita_e2e.rs`;
- same-point batched tests in `crates/akita-pcs/tests`;
- multipoint batched tests in `crates/akita-pcs/tests`;
- extension arithmetic unit tests in `crates/akita-field/src/fields/ext.rs`,
  `crates/akita-field/src/fields/lift.rs`, and
  `crates/akita-field/src/fields/packed_ext.rs`;
- prime-offset registry tests in `crates/akita-pcs/tests/algebra.rs` and
  `crates/akita-pcs/tests/primality.rs`;
- setup-capacity tests in `crates/akita-pcs/tests/setup.rs`;
- transcript tests in `crates/akita-pcs/tests/transcript.rs`;
- field-reduction unit tests in `crates/akita-types/src/field_reduction.rs`;
- planner/proof-size tests in `crates/akita-planner` and `akita-config`.

New test groups:

- **Incidence normalization tests.** Construct point/group/claim graphs by hand
  and assert derived quantities match expected flattened schedule inputs.
- **Incidence validation tests.** Reject malformed indices, mismatched
  dimensions, empty inputs, duplicate ambiguities, and setup overflows.
- **Transcript binding tests.** Reorder point, group, and claim edges and assert
  transcript divergence.
- **Generic extension embedding tests.** Check `k = 1` degenerates to the
  existing coefficient embedding; check `k > 1` trace relation against direct
  extension-field inner products for small rings.
- **Base/ext optimized opening tests.** Use fp32 or fp64 base fields with
  `Fp2`, `TowerBasisFp4`, and `PowerBasisFp4` claim fields; cover dense and
  one-hot polynomial backends.
- **Soundness regressions.** Wrong claim, wrong conjugate point, degenerate
  Moore matrix, and redistribution attack must all fail verification.
- **Planner tradeoff tests.** For fixed `(ell, k)`, compare split choices and
  assert the expected monotonic direction: larger `t` decreases transformed
  variables and increases same-commitment opening width.

Recommended local verification before PR handoff:

```bash
cargo fmt -q
cargo clippy --all --all-targets --all-features -- -D warnings
cargo test
RUSTFLAGS='-C debuginfo=0' cargo test field_reduction --lib
RUSTFLAGS='-C debuginfo=0' cargo test transcript --test transcript
RUSTFLAGS='-C debuginfo=0' cargo test akita_e2e --test akita_e2e
```

### Performance

The fp128 degree-one path should have no material performance regression. The
incidence model may add normalization overhead at the API boundary, but it
should not add per-round overhead in the prover/verifier hot loops after
flattening.

Expected proof-size/planner behavior:

- Generic extension-valued opening pays the Hachi `ell - alpha + kappa`
  transformed-variable shape.
- Optimized base-coefficient / extension-point opening with split `t` pays
  fewer transformed variables:
  `ell - alpha + kappa - t`.
- The same optimization pays wider same-commitment opening width:
  `P = 2^t`.
- At full split, `t = log_2(k)`, transformed variables reduce by `log_2(k)` and
  opening width is `k`.
- Proof-size accounting separates base-field bytes for ring/digit/SIS material
  from extension-field bytes for public scalar claims and sumcheck messages.

The implementation PR should include a planner/proof-size test or script output
showing the split-parameter tradeoff for at least one fp32 or fp64 profile.

## Design

### Architecture

This target cuts across seven layers:

1. **Field/config layer.** `akita-config` already has
   `Field`, `ClaimField`, and `ChallengeField`.
   The cutover consumes those roles in public claim types and transcript code.
2. **Extension representation layer.** `akita-field` owns the concrete
   `Fp2`, `TowerBasisFp4`, and `PowerBasisFp4` representations, the
   `ExtField`/`LiftBase`/`MulBase` contract, packed extension kernels, and the
   explicit prime-offset registry used by small-field configs.
   Higher layers should depend on this contract rather than reaching into
   representation-specific coefficient layouts.
3. **Claim-shape layer.** `akita-types` should own the normalized incidence
   graph and derived flattened views so prover and verifier cannot disagree on
   routing.
4. **Prover API layer.** `akita-prover` should expose committed groups once and
   point/claim edges separately, instead of requiring nested duplication.
5. **Verifier API layer.** `akita-types::CommitmentVerifier` and
   `akita-verifier` should verify the same normalized claim shape without
   importing prover-only polynomial or hint types.
6. **Field-reduction layer.** `akita-types::field_reduction` already has
   reference `psi_pack`/`trace_h` helpers.
   The cutover should graduate them into production-ready embedding helpers for
   `k > 1`.
7. **Planner/proof-size layer.** `akita-planner`, `akita-config`, and schedule
   helpers should account for base-field bytes, claim-field bytes, point count,
   group count, claim count, and split parameter `t`.
   The incidence-only cutover should reuse the existing aggregate root shape
   pricing by deriving `(K, G, P)` from the incidence summary.
   A new pricing model is only needed if later multipoint-opening
   optimizations change the cost per claim edge or introduce new shared
   algebraic work beyond distinct points, distinct groups, and flattened
   claims.

### Claim Incidence Model

The normalized model is a bipartite graph plus claim edges:

```text
points[p]  ---- claim c ----  groups[g]
    |                         |
    |                         +-- commitment
    |                         +-- prover hint
    |                         +-- polynomial slice [poly_idx]
    |
    +-- opening point in Cfg::ClaimField^ell

claim c:
  point_idx
  group_idx
  poly_idx_within_group
  claimed_eval in Cfg::ClaimField
```

Suggested data ownership:

- `akita-types` owns verifier-safe normalized structs:
  - points;
  - group commitments;
  - claim edges;
  - derived shape summaries.
- `akita-prover` owns prover-only extensions that attach polynomial slices and
  hints to group indices.
- `akita-verifier` consumes the verifier-safe form only.

The legacy `MultiPointBatchShape` adapter has been removed.
Incidence summaries are the live root-batching abstraction because they
represent claim-to-group routing independently from claim-to-point routing.

For schedule selection, the normalized incidence graph should first collapse to
the existing aggregate witness shape:

```text
K = number of claim edges
G = number of distinct committed groups
P = number of distinct opening points
```

This means arbitrary incidence does require scheduler plumbing, but it does not
require repricing the root proof by the full incidence matrix.
The current planner already prices the root witness by these three aggregate
counts.
The implemented cutover makes `ClaimIncidenceSummary` the source of those
counts.
If we later implement a same-polynomial multipoint optimization that reduces
the per-edge work, or adds a different shared witness object, that optimization
should introduce explicit new planner inputs and proof-size formulas.

### Extension Representation Contract

Extension representation is part of the proof plumbing contract, not just an
arithmetic micro-optimization.
The public contract is:

```text
Fp2              : coeffs [c0, c1] in basis [1, u]
PowerBasisFp4    : coeffs [c0, c1, c2, c3] in basis [1, v, v^2, v^3]
TowerBasisFp4    : internal [(c0, c2), (c1, c3)] over Fp2
to_base_vec      : always [c0, c1, c2, c3] for quartic fields
```

The tower representation is chosen for tower arithmetic, but transcript
absorption, serialization-facing helpers, and cross-basis tests use the
univariate coefficient order.
This keeps tower and power quartics interchangeable at the API boundary.

`MulBase<F>` is the preferred mixed-field scaling operation for protocol hot
paths.
It avoids the cost and ambiguity of lifting `F` into `E` solely to multiply by a
base scalar.
`SparseChallenge::eval_at_pows`, extension-valued relation helpers, direct
witness checks, and ring-switch internals should use `mul_base` where the scalar
is known to be in the base field.

Packed extension arithmetic is routed through `PackedField` hooks for
quadratic, tower quartic, and power quartic multiplication.
Scalar defaults remain available, while NEON and small-field implementations can
fuse product sums and reductions.

Small-field presets are explicit registered primes.
Docs and tests should refer to names such as `Prime32Offset99` and
`Prime64Offset59`, not to an implicit `Pow2Offset` family.
The full-word `2^64 - offset` case is part of the supported arithmetic surface.

### Extension-Field API Cutover

The current conceptual shape:

```text
ProverClaims<'a, F, P, C, H>
VerifierClaims<'a, F, C>
OpeningPoints<'a, F> = &'a [F]
CommittedOpenings<'a, F, C> { openings: &'a [F], ... }
```

should become config-driven:

```text
base field      : Cfg::Field
claim field     : Cfg::ClaimField
challenge field : Cfg::ChallengeField

opening points  : &[Cfg::ClaimField]
claimed evals   : Cfg::ClaimField
commitments     : RingCommitment<Cfg::Field, D>
setup/rings     : Cfg::Field
```

Implementation should mutate existing APIs rather than introducing
`*_ext`, `*_claim`, or legacy wrapper variants.

### Generic Extension-Valued Transform

For extension-valued claims, the sound fallback is Hachi's generic transform:

1. Treat the claim as an inner product over `F_{q^k}`.
2. Embed each extension element into the fixed multiplicative subfield
   `R_q^H ~= F_{q^k}`.
3. Pack `(R_q^H)^{D/k}` into `R_q` using the Hachi `psi` map.
4. Verify the scalar inner product through
   `Tr_H(Y * sigma_-1(v)) = (D/k) * y`.
5. Preserve the existing coefficient embedding as the `k = 1` specialization.

This path is the correctness baseline for arbitrary extension-valued
polynomials and extension-valued points.

### Optimized Base-Coefficient / Extension-Point Path

For base-field polynomial coefficients and extension-field points, Akita should
use the Frobenius-conjugate optimization:

```text
f(X_head, X_tail) = sum_h lambda_h(X_head) f_h(X_tail)
g(X_tail) = sum_h theta_h f_h(X_tail)
```

where the `theta_h` are `F_q`-linearly independent in `Cfg::ClaimField`.

The prover commits to the transformed `g` and opens the same committed object
at conjugate tail points:

```text
x_tail^(q^j) = (x_{t+1}^{q^j}, ..., x_ell^{q^j})
s_j = g(x_tail^(q^j))
```

Since each `f_h` has base-field coefficients:

```text
f_h(x_tail^(q^j)) = f_h(x_tail)^(q^j)
s_j^(q^-j) = sum_h theta_h^(q^-j) * f_h(x_tail)
```

The Moore matrix `(theta_h^(q^-j))_{j,h}` binds the slice evaluations when the
`theta_h` are `F_q`-linearly independent. The verifier then checks:

```text
y = sum_h lambda_h(x_head) * f_h(x_tail)
```

This is the optimized path for the common sumcheck setting where committed
tables are base-field-valued and challenges live in an extension field.

### Planner Model

For extension degree `k` and split `t`:

```text
P = 2^t
ring variables = ell - alpha + kappa - t
opening width = P
```

The planner should model the tradeoff rather than hard-code full split:

- `t = 0`: generic base-as-extension route, one opening, more transformed
  variables.
- `0 < t < log_2(k)`: intermediate tradeoff.
- `t = log_2(k)`: full base/ext optimization, `k` conjugate openings, fewer
  transformed variables.

The incidence model should expose enough shape data for proof-size accounting
to separate shared group material from per-point and per-edge material.

### Alternatives Considered

**Keep the current nested claim shape.**
Rejected because one committed group opened at many points requires duplicating
the group under each point, which confuses ownership and makes sharing
optimizations accidental.

**Add separate APIs for one-poly-many-points and one-point-many-polys.**
Rejected because the general case is a matching between points and groups.
Separate APIs would multiply protocol paths and make future batching
optimizations brittle.

**Use the literal Hachi base-field optimization.**
Rejected because prover-supplied extension-valued partial evaluations are not
uniquely bound by a single fixed extension-field linear relation; a malicious
prover can redistribute error among them while preserving the checked
combination.

**Use FRI-Binius-style ring-switching sumcheck for base/ext mismatch.**
Rejected for this setting because the Frobenius-conjugate route can avoid an
extra ring-switching sumcheck and instead pay a wider same-commitment opening.
It remains useful prior art for reasoning about base/ext mismatch.

**Only implement the generic extension-valued transform.**
Rejected as the final target because the common Akita/Jolt setting has
base-field-valued committed tables with extension-field sumcheck points. The
optimized path should be available once the claim model can express it.

## Documentation

Required documentation changes:

- Update `specs/general-field-support.md` if the follow-up changes any boundary
  assumptions from PR #60.
- Keep the "Current State" section in this spec synchronized with the active
  implementation branch so the spec remains a live map rather than a stale
  aspirational document.
- Keep this spec updated with the actual incidence model names once
  implementation starts.
- Add crate docs or README notes for the new public claim input shape.
- Add developer documentation for choosing the base/ext split parameter `t`.
- Update profile/planner documentation if proof-size planning exposes
  base/ext split choices.
- Keep field-arithmetic benchmark notes synchronized with the modular
  `field_arith` bench layout, especially the separate `Fp2`, tower quartic,
  power quartic, packed, wide, and parallel throughput cases.
- Update any shared research notes or paper writeups if the implemented
  optimization diverges from the Hachi/Akita design described here.

## Execution

### Phase 0: Fold In Completed Groundwork

- [x] Keep field-role split from `specs/general-field-support.md` as the
  baseline.
- [x] Keep extension transcript helpers as the canonical way to absorb and
  sample extension elements over a base-field transcript.
- [x] Keep reference `SubfieldParams`, `trace_h`, and `psi_pack` helpers as the
  algebra tests for the later production embedding.
- [x] Keep field-width-aware proof-size and schedule scaffolding.
- [x] Keep the worktree's explicit `Fp2`, `TowerBasisFp4`, and `PowerBasisFp4`
  representation work.
- [x] Keep packed extension kernels and representation tests.
- [x] Keep the canonical univariate limb-order tests for both tower and power
  quartics.
- [x] Keep the explicit `Prime*Offset*` registry and associated primality /
  consistency tests.
- [x] Keep the modular `field_arith` benchmark split and remove obsolete
  one-off codegen and fp64 reduction probes from the public bench surface.
- [x] Keep mixed-field `ExtField`/`LiftBase`/`MulBase` plumbing for sparse
  challenges, ring evaluation, relation helpers, and ring-switch internals.
- [ ] Remove temporary branch-local aliases or compatibility names during the
  final cutover.

### Phase 1: Claim Incidence Model

- [x] Define verifier-safe point/group/claim structs in `akita-types`.
- [x] Define prover-side group structs in `akita-prover` that attach polynomial
  slices and hints by group index.
- [x] Add normalization from ergonomic caller input to canonical incidence
  graph.
- [x] Add validation for dimensions, indices, empty inputs, and setup capacity.
- [x] Route current root batching directly from the incidence graph.
- [x] Preserve `ClaimIncidenceSummary` in prepared prover and verifier claim
  views.
- [x] Route root-direct witness/opening checks from `ClaimIncidenceSummary`
  rather than from the legacy batch shape.
- [x] Route folded-root per-claim point lookups from
  `ClaimIncidenceSummary::claim_to_point`.
- [x] Cut over root batching to consume incidence summaries directly and remove
  the legacy batch-shape adapter.
- [x] Add temporary transcript absorption for normalized incidence shape.
- [ ] Remove incidence-shape transcript absorption after public claim
  absorption canonicalizes and binds the same routing.
- [x] Add unit tests for validation and routing.
- [x] Add unit tests for transcript binding.

### Phase 2: API Cutover To ClaimField

- [x] Add public `ClaimField` associated types to prover and verifier traits.
- [x] Generalize shared batched input validation over the public claim scalar.
- [x] Generalize verifier claim preparation over the public claim scalar.
- [x] Generalize root-direct witness checks over extension-valued verifier
  claims.
- [x] Generalize prover claim preparation over extension-valued opening points.
- [x] Instantiate public opening-point type aliases with `Cfg::ClaimField` in
  the concrete `AkitaCommitmentScheme` prover/verifier implementations.
- [x] Instantiate public claimed-evaluation types with `Cfg::ClaimField` in the
  concrete `AkitaCommitmentScheme` prover/verifier implementations.
- [x] Set `AkitaCommitmentScheme::ClaimField = Cfg::ClaimField` once the live
  prover/verifier flow accepts extension-valued claim inputs.
- [x] Keep commitments, setup, and ring proof payloads over `Cfg::Field`.
- [x] Update prover input preparation to use the incidence model.
- [x] Update verifier claim preparation to use the incidence model.
- [x] Preserve normalized incidence summaries in prepared prover and verifier
  claim views.
- [ ] Remove base-field-only compatibility aliases.
- [x] Update base-field call sites and tests to either use `Cfg::ClaimField`
  through the scheme or explicitly constrain degree-one harnesses to
  `ClaimField = Field`.

### Phase 3: Extension Arithmetic In Prover/Verifier Flow

- [x] Identify every scalar sumcheck/opening value that must move from
  `Cfg::Field` to `Cfg::ClaimField` or `Cfg::ChallengeField`.
  The public opening coordinates and claimed evaluations are `Cfg::ClaimField`.
  The root same-point batching challenges `CHALLENGE_EVAL_BATCH`, the stage-2
  batching challenge `CHALLENGE_SUMCHECK_BATCH`, and stage-2 round challenges
  `CHALLENGE_SUMCHECK_ROUND` are `Cfg::ChallengeField` at the folded-root
  boundary.
  Ring-switch `alpha`, `tau0`, and `tau1` already use the generic
  ring-switch scalar type.
  The stage-2 verifier and deferred M-eval source are generic over a proof
  scalar independent of the base ring field.
  Stage-1 tree interstage claims, stage-1 round challenges, recursive suffix
  opening points, recursive suffix openings, and stage-1/stage-2 proof payloads
  remain base-field until `AkitaStage1Proof`, `AkitaStage2Proof`, and
  `AkitaLevelProof` become proof-scalar generic.
- [ ] Wire folded root proving and verifying through the already-generic
  `E`-parameterized helpers instead of instantiating
  `prepare_batched_prove_inputs`, `ring_switch_verifier`, and the stage-2
  relation with `E = F`.
- [x] Update folded-root transcript absorption for public claim-field values in
  the degree-one path.
- [x] Route folded-root root-batching and stage-2 challenge sampling through
  `Cfg::ChallengeField` in the degree-one bridge.
- [ ] Update random/challenge sampling to true extension-valued stage-1/stage-2
  proof payloads once those payloads are proof-scalar generic.
- [ ] Ensure base-ring commitments and digit decomposition stay over
  `Cfg::Field`.
- [ ] Add degree-one tests proving fp128 transcript/proof behavior is unchanged
  where expected.

### Phase 4: Hachi `k > 1` Embedding

- [ ] Implement production extension-to-subfield basis embedding.
- [ ] Implement production `psi` packing for `(R_q^H)^{D/k} -> R_q`.
- [ ] Implement trace-scaling handling for `(D/k)`.
- [ ] Document norm behavior for `k = 1` and `k > 1`.
- [ ] Add direct algebra tests against extension-field inner products.
- [ ] Reject invalid ring/extension parameter combinations early.

### Phase 5: Frobenius-Conjugate Base/Ext Optimization

- [ ] Add representation for split parameter `t`.
- [ ] Implement base polynomial slicing into `f_h`.
- [ ] Choose or derive `F_q`-linearly independent `theta_h`.
- [ ] Build the transformed extension-valued tail polynomial `g`.
- [ ] Open `g` at Frobenius-conjugate tail points through the incidence model.
- [ ] Verify Moore-system binding of slice evaluations.
- [ ] Verify reconstruction of the original claim.
- [ ] Add wrong-claim, wrong-conjugate, and redistribution-attack tests.

### Phase 6: Planner And Proof-Size Accounting

- [x] Derive planner witness shape `(K, G, P)` from
  `ClaimIncidenceSummary::{num_claims, num_groups, num_points}`.
- [x] Use incidence-derived `(K, G, P)` for schedule lookup and DP fallback
  instead of deriving shape from the legacy batch-shape adapter.
- [x] Keep the incidence-only cutover on the existing aggregate root pricing
  model unless proof construction changes the cost of an edge.
- [ ] Add planner inputs for claim-field extension degree.
- [ ] Add planner inputs for split parameter `t`.
- [ ] Account for base-field bytes and claim-field bytes separately.
- [ ] Account for shared group material versus per-point/per-edge material.
- [ ] Add separate pricing only for later same-polynomial multipoint
  optimizations that change per-edge work or introduce new shared algebraic
  witness material.
- [ ] Add tests or golden outputs for split choices.
- [ ] Document recommended split selection.

### Phase 7: E2E And CI Hardening

- [ ] Add fp32 dense extension-point E2E.
- [ ] Add fp64 dense extension-point E2E.
- [ ] Add one-hot extension-point E2E.
- [ ] Add same-point many-polynomial incidence E2E.
- [ ] Add one-group many-point incidence E2E.
- [ ] Add arbitrary incidence E2E.
- [ ] Run `cargo fmt -q`.
- [ ] Run `cargo clippy --all --all-targets --all-features -- -D warnings`.
- [ ] Run `cargo test`.
- [ ] Confirm GitHub CI green.

### Primary Files To Touch

- `crates/akita-field/src/fields/ext.rs`
- `crates/akita-field/src/fields/lift.rs`
- `crates/akita-field/src/fields/packed_ext.rs`
- `crates/akita-field/src/fields/packed.rs`
- `crates/akita-field/src/fields/packed_neon.rs`
- `crates/akita-field/src/fields/pseudo_mersenne.rs`
- `crates/akita-field/src/fields/fp64.rs`
- `crates/akita-field/src/fields/mod.rs`
- `crates/akita-field/src/lib.rs`
- `crates/akita-algebra/src/ring/eval.rs`
- `crates/akita-challenges/src/challenge.rs`
- `crates/akita-types/src/proof/scheme.rs`
- `crates/akita-types/src/proof/batch.rs`
- `crates/akita-types/src/proof/incidence.rs`
- `crates/akita-types/src/proof/relation.rs`
- `crates/akita-types/src/field_reduction.rs`
- `crates/akita-prover/src/lib.rs`
- `crates/akita-prover/src/api/scheme.rs`
- `crates/akita-prover/src/protocol/flow.rs`
- `crates/akita-prover/src/protocol/ring_switch.rs`
- `crates/akita-prover/src/protocol/quadratic_equation.rs`
- `crates/akita-verifier/src/proof/claims.rs`
- `crates/akita-verifier/src/protocol/batched.rs`
- `crates/akita-verifier/src/protocol/levels.rs`
- `crates/akita-verifier/src/protocol/ring_switch.rs`
- `crates/akita-config/src/lib.rs`
- `crates/akita-config/src/proof_optimized.rs`
- `crates/akita-pcs/benches/field_arith.rs`
- `crates/akita-pcs/benches/field_arith/`
- `crates/akita-planner/src/proof_size.rs`
- `crates/akita-planner/src/search.rs`
- `crates/akita-pcs/tests/akita_e2e.rs`
- `crates/akita-pcs/tests/batched_aggregated_e2e.rs`
- `crates/akita-pcs/tests/multipoint_batched_e2e.rs`
- `crates/akita-pcs/tests/transcript.rs`

## References

- Current scaffolding spec: `specs/general-field-support.md`
- Hachi field-reduction helpers:
  `crates/akita-types/src/field_reduction.rs`
- Current verifier claim API:
  `crates/akita-types/src/proof/scheme.rs`
- Current batch-shape helpers:
  `crates/akita-types/src/proof/batch.rs`
- Current prover claim API:
  `crates/akita-prover/src/lib.rs`
- Current prover flow:
  `crates/akita-prover/src/protocol/flow.rs`
- Current verifier normalization:
  `crates/akita-verifier/src/proof/claims.rs`
- Akita/Hachi extension-field design notes discussed during PR planning.
- Related Akita/Hachi paper writeup describing the base/ext optimization.
