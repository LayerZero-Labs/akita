# Spec: Extension-Field Opening Completion And Frobenius Optimization

| Field | Value |
| --- | --- |
| Author(s) | Quang Dao |
| Created | 2026-05-06 (originally as the umbrella for PRs #69, #71, and the work below) |
| Status | proposal |
| PR | follow-up to #71 |
| Predecessor | `specs/extension-field-trace-cutover.md` (#71) |
| Earlier slices | `specs/general-field-support.md` (#60), `specs/extension-claim-incidence-cutover.md` (#69) |

## Summary

Final slice of Akita's extension-field opening cutover, plus the Frobenius-conjugate base/ext optimization, plus planner/proof-size accounting that knows about claim-field extension degree, plus extension-point E2E coverage.

This spec covers everything left after PR #71:

- **Phase 4 completion.** Promote stage-1/stage-2 proof payloads and recursive suffix opening points to a proof scalar so that `gamma` (root same-point batching), `batching_coeff` (stage-2), `s_claim`, `next_w_eval`, and recursive suffix opening points can live in `Cfg::ChallengeField` without serialization wedges. This removes the last remaining degree-one bridge (`DegreeOneChallengeSampler`) and the `claim_points_to_base` projection. Add the missing norm documentation and direct ext-inner-product tests, and reject invalid ring/extension parameter combinations early at the scheme/setup boundary.
- **Phase 5: Frobenius-conjugate base/ext optimization.** Implement the optimized path for base-field polynomial coefficients opened at extension-field points: split parameter `t`, base slicing, `F_q`-linearly independent `theta_h`, transformed extension-valued tail polynomial `g`, conjugate-tail openings, Moore-system binding, and the redistribution-attack regression suite.
- **Phase 6 completion: planner / proof-size for the new dimensions.** Account for claim-field extension degree, split parameter `t`, base-field bytes vs claim-field bytes, and the shared-group vs per-edge cost split. Add tests or golden outputs.
- **Phase 7: E2E and CI hardening.** fp32/fp64 dense + one-hot extension-point E2E, same-point many-poly incidence E2E, one-group many-point incidence E2E, arbitrary incidence E2E, and full CI on the result.

This spec was originally the umbrella for the entire extension-field-opening flow, including the work that landed in PRs #69 and #71. After the retrospective scoping pass on `quang/general-field-final`, the umbrella was shrunk to cover only the remaining work; the earlier slices have their own per-PR specs (see "Earlier slices" in the header).

## Intent

### Goal

Finish the extension-field opening cutover so that:

1. The verifier surface has no degree-one bridges left. `gamma`, stage-2 challenges, recursive suffix opening points, and proof payloads all live in their natural fields. `K > 1` configs are first-class on the live prove/verify path, not just on the trace-check helper.
2. The Frobenius-conjugate base/ext route is implemented and selectable for the common Akita/Jolt setting (base-field-valued committed tables with extension-field sumcheck points).
3. The planner can price the resulting tradeoff space (extension degree `k`, split `t`, group/point/claim shape) without faking fp128 byte accounting.
4. fp32 and fp64 base-field configs have full E2E commit/prove/verify coverage at extension-field points across the incidence shapes the model supports.

### Scope Boundary

- The proof-payload reshape is the architectural prerequisite for the rest. `AkitaStage1Proof<F>`, `AkitaStage2Proof<F>`, `AkitaLevelProof<F>`, `AkitaBatchedProof<F>`, `AkitaBatchedRootProof<F>`, and `AkitaProofStep<F>` gain a second type parameter for the proof scalar. All serialization, all callsites, and the `RecursiveVerifierState` carry that scalar.
- The Frobenius-conjugate route ships as a planner-selectable optimization on top of the generic extension-valued path, not as a parallel public API.
- Planner work limits itself to extending the existing aggregate `(K, G, P)` pricing model with explicit extension-degree and split-parameter inputs. A wholesale rewrite of the planner cost model is out of scope unless a measured profile demands it.

### Invariants

- Ring commitments, setup matrices, recursive witnesses, digit decomposition, CRT/NTT work, and SIS bounds remain over `Cfg::Field`.
- Public opening points and claimed evaluations are over `Cfg::ClaimField`.
- Fiat-Shamir scalar challenges that need extension-field soundness are sampled over `Cfg::ChallengeField`; base-ring transcript absorption still uses the base transcript field `Cfg::Field`.
- The fp128 production path remains the degree-one specialization `Field = ClaimField = ChallengeField`; no compatibility wrappers or parallel legacy APIs survive.
- One public claim representation covers same-point batching, same-commitment multipoint openings, arbitrary point/group matchings, and Frobenius-conjugate openings used by the base/ext optimization.
- Transcript binding absorbs the full incidence shape: points, groups, commitments, claim routing, and claimed evaluations.
- Wrong claimed values, wrong conjugate points, invalid Moore systems, and redistribution attempts are rejected.
- Existing same-point batching and current multipoint behavior remain correct after they are expressed as derived views of the incidence model.

### Non-Goals

- This does not introduce separate public APIs for each batching special case.
- This does not keep base-field-only public aliases after the cutover; this repo is full-cutover only.
- This does not require every execution-sharing optimization to land in the first implementation. The incidence model must support those optimizations without additional public API churn.
- This does not tune production fp32/fp64 schedule tables unless explicitly included in the implementation PR.
- This does not replace the default fp128 security parameters.
- This does not implement the unsound literal Hachi base-field optimization based on unbound extension-valued partial evaluations.
- This does not add a separate ring-switching sumcheck for the base/ext optimization; the intended trade is wider same-commitment multipoint opening versus fewer transformed variables.

## Evaluation

### Acceptance Criteria

Phase 4 completion (proof-scalar payload reshape + cleanup):

- [ ] `AkitaStage1Proof`, `AkitaStage2Proof`, `AkitaLevelProof`, `AkitaBatchedProof`, `AkitaBatchedRootProof`, `AkitaProofStep` are generic over a proof scalar `S` in addition to the base field `F`.
- [ ] Serialization (`CanonicalSerialize`, `CanonicalDeserialize`, `Valid`) is updated for all proof structs to handle the proof scalar.
- [ ] `RecursiveVerifierState` carries the proof scalar; recursive suffix opening points are extension-valued.
- [ ] Root same-point batching `gamma` is sampled in `Cfg::ChallengeField`.
- [ ] Stage-2 batching `batching_coeff` is sampled in `Cfg::ChallengeField`.
- [ ] Stage-2 round challenges are sampled in `Cfg::ChallengeField`.
- [ ] `s_claim`, `next_w_eval`, and stage-1/stage-2 sumcheck round payloads carry the proof scalar.
- [ ] `DegreeOneChallengeSampler` is removed from `akita-types` and `claim_points_to_base` (the remaining bridge for opening points) is removed.
- [ ] The live prover and verifier accept true extension-valued folded roots; the explicit "reject extension folded roots" guard is gone.
- [ ] Norm behavior is documented for `k = 1` (no blowup) and `k > 1` (Hachi subfield-basis blowup) in `crates/akita-types/src/field_reduction.rs` rustdoc and either inline in this spec's References section or in a sibling design note.
- [ ] Direct algebra tests exercise `Tr_H` and `psi_embed` against extension-field inner products at the verifier-orchestration level, not just the helper level.
- [ ] Invalid ring/extension parameter combinations (e.g. `K` not dividing `D / 2`, `gcd(4K + 1, 2D) != 1`, unsupported `K` for the runtime dispatcher) are rejected at scheme/setup construction, not only at the trace-check call site.

Phase 5 (Frobenius-conjugate base/ext optimization):

- [ ] Base-field polynomial backends can be opened at `Cfg::ClaimField` points.
- [ ] The implementation supports a split parameter `t`.
- [ ] For split `t`, the prover forms base slices `f_h` and the extension-valued tail polynomial `g = sum_h theta_h f_h`.
- [ ] The prover opens the same committed transformed polynomial at Frobenius-conjugate tail points.
- [ ] The verifier checks the Moore-system binding of slice evaluations.
- [ ] The verifier checks the original claimed value `sum_h lambda_h(x_head) f_h(x_tail)`.
- [ ] The implementation rejects non-independent `theta_h` choices or any degenerate Moore matrix.
- [ ] The optimized path does not introduce an extra ring-switching sumcheck.
- [ ] Wrong-claim, wrong-conjugate, redistribution-attack, and degenerate-Moore-matrix regression tests fail verification.

Phase 6 (planner and proof-size accounting):

- [ ] Add planner inputs for claim-field extension degree.
- [ ] Add planner inputs for split parameter `t`.
- [ ] Account for base-field bytes and claim-field bytes separately.
- [ ] Account for shared group material versus per-point/per-edge material.
- [ ] Add separate pricing only for later same-polynomial multipoint optimizations that change per-edge work or introduce new shared algebraic witness material.
- [ ] Add tests or golden outputs for split choices.
- [ ] Document recommended split selection.

Phase 7 (E2E and CI):

- [ ] fp32 dense extension-point E2E.
- [ ] fp64 dense extension-point E2E.
- [ ] One-hot extension-point E2E.
- [ ] Same-point many-polynomial incidence E2E.
- [ ] One-group many-point incidence E2E.
- [ ] Arbitrary incidence E2E.
- [ ] Transcript-reordering regression fails verification.
- [ ] `cargo fmt`, `cargo clippy --all --all-targets --all-features -- -D warnings`, `cargo test` all pass.
- [ ] GitHub CI green.

### Testing Strategy

Existing tests that must continue passing:

- All fp128 tests in `crates/akita-pcs/tests/akita_e2e.rs`.
- Same-point batched tests in `crates/akita-pcs/tests`.
- Multipoint batched tests in `crates/akita-pcs/tests`.
- Extension arithmetic unit tests in `crates/akita-field/src/fields/{ext,lift,packed_ext}.rs`.
- Prime-offset registry tests in `crates/akita-pcs/tests/{algebra,primality}.rs`.
- Setup-capacity tests in `crates/akita-pcs/tests/setup.rs`.
- Transcript tests in `crates/akita-pcs/tests/transcript.rs`.
- Field-reduction unit tests in `crates/akita-types/src/field_reduction.rs`.
- Planner / proof-size tests in `crates/akita-planner` and `akita-config`.
- The 572 workspace tests passing on `quang/general-field-final`'s head as the regression baseline.

New test groups:

- **Proof-payload roundtrip tests.** Construct `AkitaBatchedProof<F, S>` for representative `(F, S)` pairs, serialize, deserialize, assert structural equality, and verify the resulting proof.
- **Extension challenge replay tests.** Verify that `gamma`, `batching_coeff`, and stage-2 round challenges sampled in `Cfg::ChallengeField` reproduce identical values on prover and verifier replay.
- **Generic extension embedding tests.** Check `k = 1` degenerates to the existing coefficient embedding; check `k > 1` trace relation against direct extension-field inner products for small rings, exercised through the live verifier rather than only through the helper.
- **Base/ext optimized opening tests.** Use fp32 or fp64 base fields with `Fp2`, `TowerBasisFp4`, and `PowerBasisFp4` claim fields; cover dense and one-hot polynomial backends.
- **Soundness regressions.** Wrong claim, wrong conjugate point, degenerate Moore matrix, and redistribution attack must all fail verification.
- **Planner tradeoff tests.** For fixed `(ell, k)`, compare split choices and assert the expected monotonic direction: larger `t` decreases transformed variables and increases same-commitment opening width.

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

The fp128 degree-one path should have no material performance regression. The proof-payload reshape is mostly mechanical (an extra type parameter); per-call overhead is zero once monomorphized. The reshape does add bytes to serialized payloads when `Cfg::ChallengeField != Cfg::Field`; that is the intended cost for extension-field soundness on small base fields.

Expected proof-size / planner behavior:

- Generic extension-valued opening pays the Hachi `ell - alpha + kappa` transformed-variable shape.
- Optimized base-coefficient / extension-point opening with split `t` pays fewer transformed variables: `ell - alpha + kappa - t`.
- The same optimization pays wider same-commitment opening width: `P = 2^t`.
- At full split, `t = log_2(k)`, transformed variables reduce by `log_2(k)` and opening width is `k`.
- Proof-size accounting separates base-field bytes for ring/digit/SIS material from extension-field bytes for public scalar claims and sumcheck messages.

The implementation PR should include a planner / proof-size test or script output showing the split-parameter tradeoff for at least one fp32 or fp64 profile.

## Design

### Proof-Scalar Payload Reshape

The pre-reshape proof payload structs are F-typed:

```text
AkitaStage1Proof<F> { stages: Vec<Stage1StageProof<F>>, s_claim: F, ... }
AkitaStage2Proof<F> { sumcheck: SumcheckProof<F>, next_w_eval: F, ... }
AkitaLevelProof<F>  { y_ring: FlatRingVec<F>, v: ..., stage1, stage2, ... }
AkitaBatchedProof<F> { root, fold_steps: Vec<AkitaProofStep<F>>, ... }
```

After the reshape:

```text
AkitaStage1Proof<F, S> { ..., s_claim: S, ... }
AkitaStage2Proof<F, S> { sumcheck: SumcheckProof<S>, next_w_eval: S, ... }
AkitaLevelProof<F, S>  { y_ring: FlatRingVec<F>, v: ..., stage1: ...<F, S>, stage2: ...<F, S>, ... }
AkitaBatchedProof<F, S> { root, fold_steps: Vec<AkitaProofStep<F, S>>, ... }
```

`F` continues to govern ring elements, commitments, and digit decomposition. `S` governs sumcheck scalars, batching coefficients, and per-level "claimed eval" values that flow from root sampling through to recursive levels. The scheme instantiates `S = Cfg::ChallengeField`; for fp128 this collapses to `S = F` and the serialized layout is bit-identical.

`RecursiveVerifierState` becomes `RecursiveVerifierState<'a, F, S>` because `state.opening` is the recursive suffix opening point, which is now extension-valued.

### Removing The Last Degree-One Bridges

Two helpers in `akita-types::proof::batch` survive PR #71:

- `claim_points_to_base` projects opening point coordinates from `E` to `F`. After the reshape the prover and verifier carry typed coordinates throughout, so this projection has no remaining caller.
- `DegreeOneChallengeSampler<F, E>` is the explicit "reject `E::EXT_DEGREE != 1`" wrapper around `sample_ext_challenge`. After `gamma` and `batching_coeff` flow as `Cfg::ChallengeField`, the wrapper has no caller and is removed.

`require_degree_one_ext` and `degree_one_ext_scalar_to_base` are then dead and can be removed at the same time.

### Frobenius-Conjugate Base/Ext Optimization

For base-field polynomial coefficients and extension-field points, Akita uses the Frobenius-conjugate optimization:

```text
f(X_head, X_tail) = sum_h lambda_h(X_head) f_h(X_tail)
g(X_tail) = sum_h theta_h f_h(X_tail)
```

where the `theta_h` are `F_q`-linearly independent in `Cfg::ClaimField`.

The prover commits to the transformed `g` and opens the same committed object at conjugate tail points:

```text
x_tail^(q^j) = (x_{t+1}^{q^j}, ..., x_ell^{q^j})
s_j = g(x_tail^(q^j))
```

Since each `f_h` has base-field coefficients:

```text
f_h(x_tail^(q^j)) = f_h(x_tail)^(q^j)
s_j^(q^-j) = sum_h theta_h^(q^-j) * f_h(x_tail)
```

The Moore matrix `(theta_h^(q^-j))_{j,h}` binds the slice evaluations when the `theta_h` are `F_q`-linearly independent. The verifier then checks:

```text
y = sum_h lambda_h(x_head) * f_h(x_tail)
```

This is the optimized path for the common sumcheck setting where committed tables are base-field-valued and challenges live in an extension field.

### Planner Model

For extension degree `k` and split `t`:

```text
P = 2^t
ring variables = ell - alpha + kappa - t
opening width = P
```

The planner should model the tradeoff rather than hard-code full split:

- `t = 0`: generic base-as-extension route, one opening, more transformed variables.
- `0 < t < log_2(k)`: intermediate tradeoff.
- `t = log_2(k)`: full base/ext optimization, `k` conjugate openings, fewer transformed variables.

The incidence model already exposes enough shape data for proof-size accounting to separate shared group material from per-point and per-edge material; the work is to extend the existing planner cost helpers to consume `(k, t)` alongside `(K, G, P)`.

### Alternatives Considered

**Land Phase 4 completion as part of #71.**
Rejected. The proof-payload reshape touches every proof struct, every serializer, every callsite, the verifier replay, and the prover construction. Bundling it with the trace-check work would have made #71 several thousand lines and harder to review. The trace-check work and surface tightening are coherent on their own; the payload reshape gets a dedicated PR.

**Use the literal Hachi base-field optimization.**
Rejected because prover-supplied extension-valued partial evaluations are not uniquely bound by a single fixed extension-field linear relation; a malicious prover can redistribute error among them while preserving the checked combination.

**Use FRI-Binius-style ring-switching sumcheck for base/ext mismatch.**
Rejected for this setting because the Frobenius-conjugate route can avoid an extra ring-switching sumcheck and instead pay a wider same-commitment opening. It remains useful prior art for reasoning about base/ext mismatch.

**Only implement the generic extension-valued transform.**
Rejected as the final target because the common Akita/Jolt setting has base-field-valued committed tables with extension-field sumcheck points. The optimized path should be available once the claim model can express it.

**Do Phase 5 (Frobenius) before Phase 4 completion.**
Rejected because Phase 5 needs `Cfg::ChallengeField` to flow through the proof scalar to express conjugate-tail openings cleanly. Doing it in F-typed payloads would require its own degree-one bridge that would then have to be removed.

## Documentation

Required documentation changes:

- Add or update rustdoc on `crates/akita-types/src/proof/mod.rs` to describe the proof-scalar parameter and the `(F, S)` invariants.
- Add a norm-accounting note in `crates/akita-types/src/field_reduction.rs` describing the `k = 1` no-blowup case and the `k > 1` Hachi subfield-basis blowup used by the implementation.
- Add a developer documentation note for choosing the base/ext split parameter `t`, either in this spec's References section or in a sibling design note.
- Update profile / planner documentation if proof-size planning exposes base/ext split choices.
- Keep field-arithmetic benchmark notes synchronized with the modular `field_arith` bench layout, especially the separate `Fp2`, tower quartic, power quartic, packed, wide, and parallel throughput cases.
- Update any shared research notes or paper writeups if the implemented optimization diverges from the Hachi/Akita design described here.

## Execution

### Phase 4 Completion: Proof-Scalar Payload Reshape

- [ ] Audit every site where `Cfg::Field` is passed as a proof-scalar (prover-side and verifier-side) and classify as base-field vs proof-scalar.
- [ ] Add `S` type parameter to `AkitaStage1Proof`, `AkitaStage2Proof`, `AkitaLevelProof`, `AkitaBatchedProof`, `AkitaBatchedRootProof`, `AkitaProofStep`, plus any helper structs they own.
- [ ] Update all `CanonicalSerialize`, `CanonicalDeserialize`, and `Valid` implementations for the proof structs.
- [ ] Add `S` type parameter to `RecursiveVerifierState`; recursive suffix opening points become `S`-valued.
- [ ] Cut `verify_root_level` to sample `gamma` in `Cfg::ChallengeField`; update the per-point opening sum to live in `S`.
- [ ] Cut `verify_one_level` to consume `S`-valued openings.
- [ ] Cut `verify_fold_batched_proof`, `verify_batched_recursive_suffix`, `verify_batched_with_policy`, and `verify_batched_proof_with_schedule` to thread `S` through.
- [ ] Cut prover-side `crates/akita-prover/src/protocol/flow.rs` to construct `S`-valued payloads.
- [ ] Cut `crates/akita-scheme/src/lib.rs` to instantiate `S = Cfg::ChallengeField`.
- [ ] Remove `DegreeOneChallengeSampler`, `claim_points_to_base`, `require_degree_one_ext`, and `degree_one_ext_scalar_to_base` from `akita-types`.
- [ ] Remove the "reject extension folded roots" guard from the live prover/verifier orchestration.
- [ ] Add proof-payload roundtrip tests for representative `(F, S)` pairs.
- [ ] Add extension challenge replay tests.
- [ ] Add direct extension-inner-product tests at the verifier-orchestration level.
- [ ] Document norm behavior for `k = 1` and `k > 1`.
- [ ] Add early rejection of invalid ring/extension parameter combinations at the scheme/setup boundary.

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

- [ ] Add planner inputs for claim-field extension degree.
- [ ] Add planner inputs for split parameter `t`.
- [ ] Account for base-field bytes and claim-field bytes separately.
- [ ] Account for shared group material versus per-point/per-edge material.
- [ ] Add separate pricing only for later same-polynomial multipoint optimizations that change per-edge work or introduce new shared algebraic witness material.
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

- `crates/akita-types/src/proof/mod.rs` (proof structs gain `S` type parameter)
- `crates/akita-types/src/proof/batch.rs` (drop degree-one bridge helpers)
- `crates/akita-types/src/proof/scheme.rs`
- `crates/akita-types/src/proof/relation.rs`
- `crates/akita-types/src/field_reduction.rs` (norm doc, possibly more `K` arms in the dispatcher)
- `crates/akita-prover/src/lib.rs`
- `crates/akita-prover/src/api/scheme.rs`
- `crates/akita-prover/src/protocol/flow.rs`
- `crates/akita-prover/src/protocol/ring_switch.rs`
- `crates/akita-prover/src/protocol/quadratic_equation.rs`
- `crates/akita-verifier/src/proof/claims.rs`
- `crates/akita-verifier/src/protocol/batched.rs`
- `crates/akita-verifier/src/protocol/levels.rs`
- `crates/akita-verifier/src/protocol/ring_switch.rs`
- `crates/akita-verifier/src/stages/stage1.rs`
- `crates/akita-verifier/src/stages/stage2.rs`
- `crates/akita-scheme/src/lib.rs`
- `crates/akita-config/src/lib.rs`
- `crates/akita-config/src/proof_optimized.rs`
- `crates/akita-planner/src/proof_size.rs`
- `crates/akita-planner/src/search.rs`
- `crates/akita-pcs/tests/akita_e2e.rs`
- `crates/akita-pcs/tests/batched_aggregated_e2e.rs`
- `crates/akita-pcs/tests/multipoint_batched_e2e.rs`
- `crates/akita-pcs/tests/transcript.rs`
- `crates/akita-pcs/benches/field_arith*`

## References

- Earliest predecessor (field-role split): `specs/general-field-support.md`
- Predecessor (claim incidence + ClaimField API + extension arithmetic in flow): `specs/extension-claim-incidence-cutover.md`
- Direct predecessor (production trace primitives + verifier surface tightening): `specs/extension-field-trace-cutover.md`
- Hachi field-reduction helpers: `crates/akita-types/src/field_reduction.rs`
- Current verifier claim API: `crates/akita-types/src/proof/scheme.rs`
- Current batch-shape helpers: `crates/akita-types/src/proof/batch.rs`
- Current prover claim API: `crates/akita-prover/src/lib.rs`
- Current prover flow: `crates/akita-prover/src/protocol/flow.rs`
- Current verifier orchestration: `crates/akita-verifier/src/protocol/levels.rs`
- Akita/Hachi extension-field design notes discussed during PR planning.
