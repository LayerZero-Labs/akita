# Spec: Extension Claim Incidence Cutover

| Field | Value |
| --- | --- |
| Author(s) | Quang Dao |
| Created | 2026-05-08 (retrospective) |
| Status | landed |
| PR | #69 (`quang/small-field-proofs`, merge commit `37a7de1`) |
| Predecessor | `specs/general-field-support.md` (#60) |
| Successor | `specs/extension-field-trace-cutover.md` (#71) |

## Summary

Phases 1-3 of Akita's extension-field opening cutover, captured retroactively as a dedicated per-PR spec. This PR moves the prover/verifier public claim shape from base field to `Cfg::ClaimField`, replaces the legacy nested batched-claim adapter with a normalized point/group/claim incidence model, and threads challenge sampling through `Cfg::ChallengeField` via an explicit degree-one bridge. It also lands the field-representation work (`Fp2`, `TowerBasisFp4`, `PowerBasisFp4`, packed extension kernels, `Prime*Offset*` registry) that the later embedding cutover depends on.

The proof scalar boundary (recursive suffix opening points, stage-1/stage-2 proof payloads, root same-point batching coefficient) stays base-field. Removing those final degree-one bridges belongs to Phase 4 and ships with the production `k > 1` embedding (`specs/extension-field-trace-cutover.md` for the first slice, `specs/extension-field-opening-batching.md` for the rest).

## Intent

### Goal

Make extension-valued claims expressible at the public prover/verifier boundary without changing commitment, setup, or recursive proof payload representations, and without inventing parallel APIs for each batching variant. After this PR, callers can hand the scheme arbitrary point/group/claim incidence with `Cfg::ClaimField`-valued opening points and evaluations; the implementation routes those through one normalized graph that drives both prover and verifier.

### Scope Boundary

- The public commit/prove/verify input shape moves to `Cfg::ClaimField`. Internal commitments, recursive witnesses, ring proof payloads, and digit decomposition stay over `Cfg::Field`.
- Challenge sampling at the folded-root boundary routes through `Cfg::ChallengeField` via the explicit `DegreeOneChallengeSampler` bridge. True `Cfg::ChallengeField`-valued sampling is deferred to Phase 4 once recursive suffix opening points and proof-payload scalars become extension-valued together.
- The verifier-side stage-2 relation and deferred M-eval source become generic over a proof scalar `E`, with the live folded root still instantiating `E = F`.
- Live proof orchestration rejects true extension-valued folded roots until the Phase 4 embedding lands. The K=1 specialization is exercised end-to-end on fp128.

### Invariants

- Existing fp128 transcript/proof behavior is unchanged.
- One canonical incidence model represents same-point batching, same-commitment multipoint openings, and arbitrary point/group matchings; reordering edges binds the transcript.
- Prover and verifier derive identical flattened schedule quantities (`K`, `G`, `P`, claim-to-point/group routing) from the same incidence summary.
- The `MultiPointBatchShape` adapter is fully removed from public/protocol-facing claim flow.
- `Fp2`, `TowerBasisFp4`, `PowerBasisFp4` share a single canonical univariate limb order at API boundaries (`from_base_slice` and `to_base_vec`); tower internal storage `[(c0, c2), (c1, c3)]` is an arithmetic detail.
- `MulBase<F>` is the preferred mixed-field scaling op for protocol hot paths; sparse challenges, ring evaluation, relation helpers, and ring-switch internals use it where the scalar is base-field.
- Small-field presets are explicit registered primes (`Prime32Offset99`, `Prime64Offset59`, etc.), not an implicit power-of-two offset family.

### Non-Goals

- This does not implement the production Hachi `k > 1` embedding (`embed_subfield`, `psi_embed`, trace-scaling). Reference helpers from PR #60 remain.
- This does not lift root same-point batching `gamma`, stage-2 batching `batching_coeff`, or recursive suffix opening points to `Cfg::ChallengeField`.
- This does not make `AkitaStage1Proof`, `AkitaStage2Proof`, `AkitaLevelProof`, or `AkitaBatchedProof` proof-scalar generic.
- This does not implement the Frobenius-conjugate base/ext optimization.
- This does not regenerate fp32/fp64 production schedule tables.

## Evaluation

### Acceptance Criteria

Public API and claim model:

- [x] Public prover claim inputs accept opening points as `&[Cfg::ClaimField]`.
- [x] Public verifier claim inputs accept opening points as `&[Cfg::ClaimField]`.
- [x] Public claimed evaluations use `Cfg::ClaimField`.
- [x] Commitment, setup, and ring proof objects remain over `Cfg::Field`.
- [x] A normalized incidence model represents distinct points, distinct committed groups, and individual claims.
- [x] The incidence model supports one committed group opened at multiple points without duplicating commitment or hint input.
- [x] Existing multipoint batching is represented as the same incidence graph; `MultiPointBatchShape` is removed from public/protocol-facing claim flow.
- [x] Claim-shape validation rejects empty point sets, empty group sets, invalid point/group/poly indices, dimension mismatches, and setup-capacity overflows.

Extension representation:

- [x] `Fp2` uses transparent coefficient-array storage with `c0()` / `c1()` accessors; both quadratic non-residue configs route scalar and packed multiplication through a shared `mul_non_residue` hook.
- [x] `TowerBasisFp4` and `PowerBasisFp4` have separate scalar and packed representations, with conversion and multiplication-agreement tests.
- [x] `ExtField::from_base_slice` and `ExtField::to_base_vec` define one canonical univariate limb order for base, quadratic, tower quartic, and power quartic.
- [x] `MulBase<F>` covers base-field, `Fp2`, `TowerBasisFp4`, and `PowerBasisFp4`.
- [x] Packed field hooks cover `Fp2`, tower-basis quartic, and power-basis quartic multiplication, including NEON-backed small-field lanes.
- [x] Pseudo-Mersenne small-field presets use explicit `(bits, offset)` registered specs and `Prime*Offset*` aliases (incl. full-word `Prime64Offset59`).

Transcript and serialization:

- [x] Transcript absorption includes the normalized claim incidence shape as a migration bridge (not as proof payload).
- [x] Folded-root transcript absorption uses extension-aware claim-field helpers for public opening points and claimed evaluations in the current degree-one path.
- [x] Reordering claim-edge routing transcript-diverges unless the implementation explicitly canonicalizes that order first.

Extension arithmetic in flow:

- [x] Sparse challenges, ring evaluation, relation helpers, and ring-switch prover/verifier internals are generalized over a mixed base field `F` and extension field `E`.
- [x] Stage-2 verifier and deferred M-eval source are generic over a proof scalar.
- [x] Live folded-root prove/verify still instantiates the generic internals with degree-one claim scalars; true extension-valued folded roots are rejected until Phase 4.
- [x] Folded-root challenge sampling routes through `Cfg::ChallengeField` via the `DegreeOneChallengeSampler` bridge.

Tests:

- [x] Unit tests cover extension-field degrees, tower/power quartic conversion, canonical transcript limb order, extension array layout, `MulBase` equivalence, and packed extension arithmetic.
- [x] Registry and primality tests cover the explicit `Prime*Offset*` field aliases used by fp32/fp64 scaffolds.
- [x] Ring-switch, direct-opening, and transcript tests build through `ClaimIncidenceSummary` rather than `MultiPointBatchShape`.
- [x] Degree-one tests prove fp128 transcript/proof behavior is unchanged.
- [x] fp128 batched proof roundtrip stays stable.

Compatibility and CI:

- [x] Existing fp128 E2E tests pass.
- [x] Existing batched and multipoint E2E tests pass after the incidence cutover.
- [x] `cargo fmt`, `cargo clippy --all -- -D warnings`, `cargo test` all pass.
- [x] GitHub CI green on PR head.

### Testing Strategy

Targeted commands run on the PR head:

- `cargo test -p akita-field`
- `cargo test -p akita-types incidence`
- `cargo test -p akita-types degree_one_challenge_bridge_matches_base_sampling`
- `cargo test -p akita-verifier extension`
- `cargo test -p akita-prover prover_claim_preparation_accepts_extension_points`
- `cargo test -p akita-scheme fp128_degree_one_batched_proof_roundtrip_is_stable`
- `cargo test -p akita-scheme folded_payload_commitments_and_digits_stay_base_field`

## Design

The implementation realizes the canonical incidence graph:

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

Data ownership:

- `akita-types` owns verifier-safe normalized structs (points, group commitments, claim edges, derived shape summaries).
- `akita-prover` owns prover-only extensions that attach polynomial slices and hints to group indices.
- `akita-verifier` consumes the verifier-safe form only.

For schedule selection, the incidence graph collapses to aggregate `(K, G, P)` (claim count, group count, point count). The current planner already prices the root witness by these three counts; `ClaimIncidenceSummary` is the source.

The degree-one challenge bridge is intentional. `DegreeOneChallengeSampler<F, E>` rejects true `E::EXT_DEGREE != 1` at construction and projects sampled extension challenges back to base. This makes the remaining bridge sites explicit and easy to grep, and keeps fp128 behavior bit-identical. Removing it is the Phase 4 embedding's job.

(See the predecessor `specs/general-field-support.md` for the field-role split and extension transcript semantics. Detailed design rationale for the incidence model, extension representation contract, and extension API cutover lives in PR #69's diff against `4b0b86a` for `specs/extension-field-opening-batching.md`, preserved in git history.)

## Execution

### Phase 1: Claim Incidence Model

- [x] Define verifier-safe point/group/claim structs in `akita-types`.
- [x] Define prover-side group structs in `akita-prover` that attach polynomial slices and hints by group index.
- [x] Add normalization from ergonomic caller input to canonical incidence graph.
- [x] Add validation for dimensions, indices, empty inputs, and setup capacity.
- [x] Route current root batching directly from the incidence graph.
- [x] Preserve `ClaimIncidenceSummary` in prepared prover and verifier claim views.
- [x] Route root-direct witness/opening checks from `ClaimIncidenceSummary` rather than from the legacy batch shape.
- [x] Route folded-root per-claim point lookups from `ClaimIncidenceSummary::claim_to_point`.
- [x] Cut over root batching to consume incidence summaries directly and remove the legacy batch-shape adapter.
- [x] Add temporary transcript absorption for normalized incidence shape.
- [x] Add unit tests for validation and routing.
- [x] Add unit tests for transcript binding.

Deferred to a later spec:

- [ ] Remove incidence-shape transcript absorption after public claim absorption canonicalizes and binds the same routing.

### Phase 2: API Cutover To ClaimField

- [x] Add public `ClaimField` associated types to prover and verifier traits.
- [x] Generalize shared batched input validation over the public claim scalar.
- [x] Generalize verifier claim preparation over the public claim scalar.
- [x] Generalize root-direct witness checks over extension-valued verifier claims.
- [x] Generalize prover claim preparation over extension-valued opening points.
- [x] Instantiate public opening-point type aliases with `Cfg::ClaimField` in `AkitaCommitmentScheme`.
- [x] Instantiate public claimed-evaluation types with `Cfg::ClaimField` in `AkitaCommitmentScheme`.
- [x] Set `AkitaCommitmentScheme::ClaimField = Cfg::ClaimField`.
- [x] Keep commitments, setup, and ring proof payloads over `Cfg::Field`.
- [x] Update prover input preparation to use the incidence model.
- [x] Update verifier claim preparation to use the incidence model.
- [x] Preserve normalized incidence summaries in prepared prover and verifier claim views.
- [x] Remove base-field-only compatibility aliases.
- [x] Update base-field call sites and tests to either use `Cfg::ClaimField` through the scheme or explicitly constrain degree-one harnesses to `ClaimField = Field`.

### Phase 3: Extension Arithmetic In Prover/Verifier Flow

- [x] Identify every scalar sumcheck/opening value that must move from `Cfg::Field` to `Cfg::ClaimField` or `Cfg::ChallengeField`. Public opening coordinates and claimed evaluations are `Cfg::ClaimField`. Root same-point batching `CHALLENGE_EVAL_BATCH`, stage-2 batching `CHALLENGE_SUMCHECK_BATCH`, and stage-2 round challenges `CHALLENGE_SUMCHECK_ROUND` are `Cfg::ChallengeField` at the folded-root boundary. Stage-1 tree interstage claims, stage-1 round challenges, recursive suffix opening points, recursive suffix openings, and stage-1/stage-2 proof payloads remain base-field until `AkitaStage1Proof`, `AkitaStage2Proof`, and `AkitaLevelProof` become proof-scalar generic.
- [x] Wire the verifier-side stage-2 relation and deferred M-eval source through the already-generic `E`-parameterized helpers. Root proving and verifying still require the degree-one bridge at the public challenge boundary because true extension-valued stage-2 challenges become recursive suffix opening points; removing that bridge belongs with the Phase 4 `k > 1` embedding cutover.
- [x] Update folded-root transcript absorption for public claim-field values in the degree-one path.
- [x] Route folded-root root-batching and stage-2 challenge sampling through `Cfg::ChallengeField` in the degree-one bridge.
- [x] Keep true extension-valued stage-1/stage-2 proof payload sampling deferred to the Phase 4 embedding cutover, where recursive suffix openings can also be made extension-valued coherently.
- [x] Ensure base-ring commitments and digit decomposition stay over `Cfg::Field`.
- [x] Add degree-one tests proving fp128 transcript/proof behavior is unchanged where expected.

### Files Modified In This PR

(From `git diff 37a7de1^..37a7de1 --stat`. Trimmed to non-trivial protocol/types/field changes.)

- `crates/akita-field/src/fields/{ext,lift,packed_ext,packed,packed_neon,pseudo_mersenne,fp64,mod}.rs`
- `crates/akita-algebra/src/ring/eval.rs`
- `crates/akita-challenges/src/challenge.rs`
- `crates/akita-types/src/proof/{scheme,batch,incidence,relation}.rs`
- `crates/akita-prover/src/{lib,api/scheme,protocol/flow,protocol/ring_switch,protocol/quadratic_equation}.rs`
- `crates/akita-verifier/src/{proof/claims,protocol/batched,protocol/levels,protocol/ring_switch}.rs`
- `crates/akita-config/src/{lib,proof_optimized}.rs`
- `crates/akita-pcs/tests/{akita_e2e,batched_aggregated_e2e,multipoint_batched_e2e,transcript}.rs`
- `specs/extension-field-opening-batching.md` (umbrella update; the umbrella is later shrunk into per-PR specs by the retrospective scoping pass)

## References

- Predecessor spec (field-role split): `specs/general-field-support.md`
- Successor spec (production trace embedding): `specs/extension-field-trace-cutover.md`
- Remaining-work spec (Phase 4 payload reshape, Phase 5 Frobenius, Phase 6/7): `specs/extension-field-opening-batching.md`
- PR #69 merge commit: `37a7de1`
- Hachi field-reduction helpers: `crates/akita-types/src/field_reduction.rs`
- Incidence types: `crates/akita-types/src/proof/incidence.rs`
- Public claim API: `crates/akita-types/src/proof/scheme.rs`
- Folded-root verifier: `crates/akita-verifier/src/protocol/levels.rs`
- Folded-root prover: `crates/akita-prover/src/protocol/flow.rs`
