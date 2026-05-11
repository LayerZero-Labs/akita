# Spec: PR #71 Part 2 - Extension-Field Opening Completion And Frobenius Optimization

| Field | Value |
| --- | --- |
| Author(s) | Quang Dao |
| Created | 2026-05-06 (originally as the umbrella for PRs #69, #71, and this completion work) |
| Status | implementation plan for PR #71 part 2 |
| PR | #71 (`quang/general-field-final`) |
| Companion spec | `specs/extension-field-trace-cutover.md` (#71 first slice) |
| Earlier slices | `specs/general-field-support.md` (#60), `specs/extension-claim-incidence-cutover.md` (#69) |

## Summary

This spec is now the implementation plan for the second implementation slice
inside PR #71. The first implementation slice after the trace cutover has
landed the proof-payload `F, L` type shape and the stage-1/stage-2 proof-scalar
plumbing. This part-2 plan covers the remaining design-sensitive work: true
extension opening materialization, `gamma` over `L`, bridge removal, Frobenius
compression, field-family-aware SIS sizing, and planner accounting.

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
- **Part 2: extension-opening completion.** This spec. Finish the algebra that
  Part 1 deliberately did not decide.

Part 2 must include:

- **Phase 4 completion.** Sample `gamma` in `L`, make root and recursive
  opening materialization work for true `E`/`L` points rather than degree-one
  projections, and remove the remaining degree-one bridges.
- **Phase 5: Frobenius-conjugate base/ext optimization.** Support base-field
  polynomial coefficients opened at extension-field points through split
  parameter `t`, base slices `f_h`, transformed tail polynomial `g`, conjugate
  tail openings, and Moore-system binding checks.
- **Phase 6: planner / proof-size and SIS accounting.** Price `E` and `L`
  extension degrees, split parameter `t`, base-field bytes versus
  extension-field bytes, shared-group versus per-point/per-edge costs, and
  field-family-specific SIS floors.
- **Phase 7: E2E and CI hardening.** Add extension-point dense/one-hot and
  incidence-shape tests, plus negative transcript/conjugate/Moore tests.

## Intent

### Goal

Finish the extension-field opening cutover so that:

1. The live prover/verifier path has no degree-one bridges. `K > 1` configs are
   first-class, not only accepted by helper-level trace tests.
2. The proof payload type reflects the field tower: ring material over `F`,
   public openings over `E`, and proof scalars over `L`.
3. The Frobenius-conjugate route is available for the common Akita/Jolt shape:
   base-field-valued committed tables opened at extension-field sumcheck
   points.
4. The planner/proof-size layer can price the generic route and the
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
- Until that optimization lands, valid extension-field openings that do not
  fit the packed-inner folded-root materialization use the existing
  root-direct proof path. This is intentionally sound and complete for E2E
  behavior, but not proof-size optimal.
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
- Transcript binding absorbs the full incidence shape: points, groups,
  commitments, claim routing, and claimed evaluations.
- Wrong claims, wrong conjugate points, invalid Moore systems, redistribution
  attempts, and transcript reordering are rejected.

### Non-Goals

- Do not introduce separate public APIs for each batching special case.
- Do not keep base-field-only aliases after the cutover.
- Do not implement the unsound literal Hachi base-field optimization based on
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
- [x] Use packed-inner Hachi subfield materialization for folded extension
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
- [ ] Replace the current degree-one projection used to materialize recursive
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

- [x] `DegreeOneChallengeSampler` is removed. Root `gamma` stays explicitly
  `F`-sampled in Part 1 instead of being sampled through `L` and projected
  through a degree-one bridge.
- [ ] `claim_points_to_base`
- [ ] `require_degree_one_ext`
- [ ] `degree_one_ext_scalar_to_base`

The remaining folded-root guards now select root-direct for valid extension
openings outside the packed-inner folded shape. Removing those guards from the
folded path itself belongs to the Frobenius/coordinate-expanded optimization
work, not to the E2E correctness boundary.

Add early validation at setup/scheme entrypoints:

- `E::EXT_DEGREE` is supported by `dispatch_trace_inner_product_check`.
- `SubfieldParams<D, E::EXT_DEGREE>::new()` succeeds:
  - `D` is a nonzero power of two.
  - `E::EXT_DEGREE` divides `D / 2`.
  - `gcd(4 * E::EXT_DEGREE + 1, 2 * D) == 1`.
- `L: ExtField<E>` is already a trait-level invariant; add tests that a config
  cannot omit this tower relation.

### Phase 4D: Documentation And Direct Tests

Rustdoc:

- Add proof field-role docs near the proof structs:
  `F` is ring material, `L` is proof-scalar material.
- Add field-reduction norm docs:
  - `k = 1`: no subfield-basis blowup; the trace shortcut is scalar equality.
  - `k > 1`: Hachi uses the fixed subfield basis and pays the documented
    embedding/trace blowup.

Tests:

- Proof-payload roundtrip tests for representative `(F, L)` pairs.
- Extension challenge replay tests for `gamma`, `batching_coeff`, and stage-2
  round challenges.
- Live verifier-orchestration trace tests for extension-valued openings, not
  only helper-level `field_reduction` tests.

### Phase 5: Frobenius-Conjugate Base/Ext Optimization

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

3. Choose deterministic `F`-linearly independent `theta_h in E`.
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

Implementation requirements:

- The transformed `g` opening uses the existing incidence model.
- The optimized path does not introduce another ring-switching sumcheck.
- `theta_h` selection is deterministic, documented, and checked for
  `F`-linear independence.
- The verifier rejects degenerate Moore matrices.

Negative tests:

- Wrong original claim fails.
- Wrong conjugate tail point fails.
- Degenerate or duplicate `theta_h` fails.
- Redistribution attack fails: changing the slice evaluations while preserving
  only the final linear combination must not verify.

### Phase 6A: Field-Family SIS Floor Registry

Status: required before generated fp32/fp64 schedule tables.

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

- Add a config hook such as `CommitmentConfig::sis_modulus_family()` and mirror
  it through `PlannerConfig`.
- Change SIS lookup APIs from:

  ```rust
  min_rank_for_secure_width(d, collision_inf, width)
  ceil_supported_collision(d, collision_inf)
  ```

  to field-family-aware forms:

  ```rust
  min_rank_for_secure_width(family, d, collision_inf, width)
  ceil_supported_collision(family, d, collision_inf)
  ```

- Move the generated SIS floor table shape from one global registry to
  per-family registries keyed by `(family, D, collision_inf)`.
- Update `AjtaiKeyParams` validation and `sis_derivation` so every security
  check receives the config-selected family explicitly.
- Update `scripts/gen_sis_table.py` to accept `--family {q32,q64,q128}` or an
  explicit `--q`, and emit the representative modulus in the generated Rust
  comments.
- Generated schedule tables must record or be generated under the same family
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

Status: required for realistic fp32/fp64 planning.

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

- Add proof-optimized preset structs and generated-family names for the new
  small-field ring dimensions, without disturbing the existing fp128 preset
  names.
- Extend the schedule-table generator so family specs are not hardcoded to
  fp128 `D32/D64/D128`.
- Extend SIS generation for Q32/Q64 to cover the larger `D` buckets above.
- Sparse challenge samplers must have explicit fast paths for each supported
  candidate ring dimension. In particular, D=256 and D=512 must not route
  through a heap-backed "large D" fallback on the proof hot path.
- Keep runtime profile mode names explicit, for example
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
  transformed variables = ell - alpha + kappa - t
  opening width = 2^t
  ```

- `t = 0` is the generic route.
- `t = log2([E : F])` is the full Frobenius-conjugate route.

Tests:

- Golden SIS-rank lookups for Q32, Q64, and Q128 at representative
  `(D, collision_inf, width)` cells.
- Golden outputs for at least one fp32 or fp64 profile across `t`.
- Assertions that larger `t` reduces transformed variables and increases
  same-commitment opening width.
- Regression tests that generated schedule tables and runtime planner fallback
  agree on witness lengths for representative extension configurations.

### Phase 7: E2E And CI Hardening

Add positive tests:

- [x] fp32 dense extension-point E2E through packed-inner folded root.
- [x] fp32 dense outer-variable extension-point E2E through root-direct
  fallback.
- fp64 dense extension-point E2E.
- one-hot extension-point E2E.
- [x] same-point many-polynomial incidence E2E through root-direct fallback.
- [x] one-group many-point incidence E2E through root-direct fallback.
- arbitrary incidence E2E.
- Frobenius route E2E with at least one nonzero split.

Add negative tests:

- transcript reordering fails;
- [x] wrong claim fails for the packed-inner folded and outer-variable
  root-direct extension E2Es;
- wrong conjugate point fails;
- degenerate Moore matrix fails;
- redistribution attack fails.

Required handoff checks:

```bash
cargo fmt -q
cargo clippy --all --all-targets --all-features -- -D warnings
cargo test
RUSTDOCFLAGS="-D warnings" cargo doc -q --no-deps --all-features
```

## Primary Files To Touch

- `crates/akita-types/src/proof/mod.rs`
- `crates/akita-types/src/proof/batch.rs`
- `crates/akita-types/src/proof/scheme.rs`
- `crates/akita-types/src/proof/relation.rs`
- `crates/akita-types/src/field_reduction.rs`
- `crates/akita-prover/src/lib.rs`
- `crates/akita-prover/src/api/scheme.rs`
- `crates/akita-prover/src/protocol/flow.rs`
- `crates/akita-prover/src/protocol/quadratic_equation.rs`
- `crates/akita-prover/src/protocol/ring_switch.rs`
- `crates/akita-prover/src/protocol/sumcheck/akita_stage1_tree.rs`
- `crates/akita-prover/src/protocol/sumcheck/akita_stage2.rs`
- `crates/akita-verifier/src/proof/claims.rs`
- `crates/akita-verifier/src/protocol/batched.rs`
- `crates/akita-verifier/src/protocol/levels.rs`
- `crates/akita-verifier/src/protocol/ring_switch.rs`
- `crates/akita-verifier/src/stages/stage1.rs`
- `crates/akita-verifier/src/stages/stage2.rs`
- `crates/akita-scheme/src/lib.rs`
- `crates/akita-config/src/lib.rs`
- `crates/akita-config/src/proof_optimized.rs`
- `crates/akita-planner/src/schedule_params.rs`
- `crates/akita-types/src/layout/proof_size.rs`
- `crates/akita-pcs/tests/akita_e2e.rs`
- `crates/akita-pcs/tests/batched_aggregated_e2e.rs`
- `crates/akita-pcs/tests/multipoint_batched_e2e.rs`
- `crates/akita-pcs/tests/transcript.rs`

## Review Checklist

- [ ] Generic naming follows `F, E, L` everywhere the field tower is visible.
- [ ] No public compatibility aliases preserve the old `AkitaBatchedProof<F>`
      proof type.
- [ ] No caller remains for degree-one bridge helpers.
- [ ] `gamma`, `batching_coeff`, stage-2 round challenges, `s_claim`, and
      `next_w_eval` are all `L`.
- [ ] Ring material remains `F`.
- [ ] Public openings remain `E`.
- [ ] `F = E = L` fp128 proofs remain semantically unchanged.
- [ ] Extension-field tests exercise a real `F < E <= L` tower.
- [ ] CI is green on the final PR head.

## References

- Earliest predecessor (field-role split): `specs/general-field-support.md`
- Predecessor (claim incidence + ClaimField API + extension arithmetic in flow):
  `specs/extension-claim-incidence-cutover.md`
- Companion #71 trace primitive spec: `specs/extension-field-trace-cutover.md`
- Hachi field-reduction helpers: `crates/akita-types/src/field_reduction.rs`
- Current verifier claim API: `crates/akita-types/src/proof/scheme.rs`
- Current batch helpers: `crates/akita-types/src/proof/batch.rs`
- Current prover flow: `crates/akita-prover/src/protocol/flow.rs`
- Current verifier orchestration: `crates/akita-verifier/src/protocol/levels.rs`
