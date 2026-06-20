# Spec: Mixed Row Ring Dimensions

| Field         | Value                         |
|---------------|-------------------------------|
| Author(s)     | Quang Dao                     |
| Created       | 2026-06-21                    |
| Status        | proposed                      |
| PR            |                               |
| Supersedes    |                               |
| Superseded-by |                               |
| Book-chapter  |                               |

## Summary

Akita currently uses one ring dimension for all rows in one fold level.
This spec proposes a mixed row dimension protocol where the A rows can use a larger ring dimension than the B and D rows in the same fold.
The goal is to get the security and kernel benefits of larger A side challenge rings without paying the full proof-size and setup-cache cost of moving the whole level to that dimension.

The design is not a tower lowering of every row to one smaller ring.
Each row group keeps its own relation over its own quotient ring.
The shared object is a canonical flat coefficient witness.
A rows interpret the relevant flat coefficients as elements of `R_DA = F[X] / (X^DA + 1)`.
B and D rows interpret their coefficients as elements of `R_DBD = F[X] / (X^DBD + 1)`.
The prover computes one quotient per row group using that row group's denominator.

The main risk is no longer algebraic soundness.
The main risk is engineering cost and memory cost.
The current prover prepares one full shared NTT cache for one const `D`.
A naive mixed D implementation would prepare several full caches for the same expanded setup.
That can erase the proof-size and speed gains, especially when a larger A dimension appears in the first fold levels.
The implementation must therefore make setup preparation aware of row groups, active prefixes, and memory budgets.

## Intent

### Goal

Support a mixed row dimension fold where A rows may use `D_A` in `{64, 128, 256}` while B and D rows remain at `D_BD = 64`, with a planner that chooses `D_A` only when proof size, security, runtime, and prepared setup memory all improve or stay within policy.

Key surfaces:

- `LevelParams` gains a row dimension shape instead of one `ring_dimension` for every row.
- The ring relation becomes a row-group relation with one quotient denominator per group.
- Setup generation uses a maximum generation dimension `D_gen` that every runtime row dimension divides.
- Prover setup preparation materializes NTT caches by dimension and active prefix, not only by dimension.
- Planner scoring accounts for A side SIS sizing, folded witness digit counts, proof bytes, setup prefix length, and prepared cache bytes.
- Verifier code validates mixed row dimensions at schedule and proof boundaries without panics.

### Algebraic Model

Let the flat folded witness be a vector of coefficients:

```text
z_flat in F^N
```

For a row group with ring dimension `D_g`, the prover chunks the relevant prefix of `z_flat` into elements of:

```text
R_Dg = F[X] / (X^Dg + 1)
```

For each row `i` in that group, the row relation is:

```text
<M_i, z>_cyclic = y_i + (X^Dg + 1) q_i
```

The quotient `q_i` is computed with the same `D_g` as the row.
There is no requirement that an A row quotient and a B row quotient live in the same ring.
The proof object already treats rows as typed ring rows at the current level.
The mixed design generalizes this by giving each row group its own ring type.

The statement remains one statement because the transcript binds:

- the full mixed level parameter object;
- the canonical flat witness length;
- the row layout and row order;
- the setup seed and generation dimension;
- every row group's runtime dimension;
- every challenge shape used for SIS sizing and folding.

### Invariants

- **The flat witness is canonical.** Prover and verifier must agree on one flat coefficient vector for the folded witness. Ring views are derived from this flat vector.
- **Every runtime dimension divides the generation dimension.** If setup data is generated at `D_gen`, then each row dimension must divide `D_gen`. A D64 view of a D256 generated setup is valid. A D128 view of a D64 generated setup is invalid.
- **Row groups own their quotient denominator.** An A row at D128 uses `X^128 + 1`. A B row at D64 uses `X^64 + 1`. The prover must never compute a D128 quotient and reinterpret it as D64 quotient data.
- **Row order remains canonical.** The public row layout stays in the current order unless this spec explicitly changes it. The mixed dimension metadata must attach to row groups, not to ad hoc call sites.
- **SIS sizing is group-specific.** A rows use `D_A` and the A challenge shape when deriving `n_a`. B and D rows use `D_BD` and their challenge shape when deriving `n_b` and `n_d`.
- **Fold digit counts are priced with the selected A dimension.** Larger A dimensions can change the folded `z` infinity bound and `num_digits_fold`. The planner must include that effect.
- **Proof-size estimates must match serialized proofs.** Any mixed proof byte formula must be backed by serialization tests after implementation.
- **Setup preparation is bounded.** The prover must not prepare full shared setup NTT caches for every supported dimension by default.
- **Verifier code stays inside the no-panic boundary.** Bad mixed parameters, bad setup shape, bad proof row counts, and mismatched dimensions return `AkitaError` or serialization errors.
- **No backward compatibility shim is required.** This is a new protocol shape. Old homogeneous schedules do not need aliases or migration layers beyond staying valid as the special case `D_A = D_BD`.

### Non-Goals

- No arbitrary runtime dimension set. The first supported set is `D_BD = 64` and `D_A` in `{64, 128, 256}`.
- No mixed dimensions for B versus D in the first implementation.
- No tower-lowered protocol that rewrites every row into the smallest ring.
- No dynamic per-row dimension inside one row group. Dimensions are per role group.
- No GPU or Metal backend work in this spec.
- No attempt to make recursive setup offload, setup-prefix commitments, and mixed D all land in one PR unless the implementation shows this is smaller than separating them.

## Evaluation

### Planner Preview

A scratch dynamic programming preview was run on the branch `quang/mixed-row-ring-dimensions`.
The preview fixed `D_BD = 64`, allowed `D_A` in `{64, 128, 256}`, and let the planner rechoose `log_basis` and the `(m, r)` split at every fold.
It also priced A side SIS ranks with the selected `D_A` and included the effect on `num_digits_fold` and the next witness length.

The preview found consistent proof-size reductions for fp128 dense singleton shapes:

| num_vars | D64 baseline bytes | mixed D preview bytes | delta bytes | selected A dimensions |
|----------|--------------------|-----------------------|-------------|-----------------------|
| 20       | 110708             | 105352                | -5356       | 64, 128, 128, 64, 128, 64 |
| 22       | 112716             | 107416                | -5300       | 64, 64, 128, 64, 64, 64 |
| 24       | 114056             | 110636                | -3420       | 128, 256, 128, 128, 256, 128, 64 |
| 26       | 114872             | 111068                | -3804       | 64, 64, 128, 128, 64, 128, 64 |
| 28       | 116328             | 111884                | -4444       | 128, 64, 128, 128, 64, 64, 64 |

These numbers are not final proof sizes.
They are schedule estimates from a scratch model.
They show that the idea is worth implementing only if the setup preparation cost is also controlled.

### Acceptance Criteria

- [ ] A homogeneous schedule with `D_A = D_BD` produces the same rows, proof bytes, and verifier behavior as the current single D path.
- [ ] A mixed schedule can set `D_A = 128` or `D_A = 256` while keeping B and D rows at D64.
- [ ] The prover computes A quotients with `X^D_A + 1` and B/D quotients with `X^64 + 1`.
- [ ] The verifier binds mixed row dimensions in the instance descriptor and rejects proofs whose dimensions do not match the schedule.
- [ ] Setup generation at `D_gen = max(D_A, D_BD)` can serve every row group view used by the schedule.
- [ ] Prepared setup memory is measured and bounded. A run that uses both D64 and D128 must not automatically build full shared caches for both dimensions if only a small prefix needs the second dimension.
- [ ] The planner can reject or penalize candidates whose extra prepared cache footprint exceeds a policy limit.
- [ ] Proof-size formula tests compare mixed formula output against serialized mixed proofs.
- [ ] End-to-end prove and verify tests cover at least one D64 only schedule and one mixed schedule.
- [ ] `cargo fmt -q`, `cargo clippy --all --message-format=short -q -- -D warnings`, `cargo test`, and `./scripts/check-doc-guardrails.sh` pass before the implementation PR is merged.

### Testing Strategy

Unit tests:

- `akita-types` tests for mixed level parameter validation.
- `akita-types` tests for setup generation dimension divisibility.
- `akita-types` tests that homogeneous mixed params serialize to the same logical schedule as the current shape, or to a clearly documented new shape with equivalent meaning.
- `akita-planner` tests that `n_a` changes with `D_A` while `n_b` and `n_d` stay tied to D64.
- `akita-prover` quotient tests that compare row-specific quotient results against direct schoolbook calculations for D64, D128, and D256.
- `akita-prover` prepared setup tests that request a small D128 prefix and assert that the reported cache bytes match that prefix, not the full setup length.

End-to-end tests:

- Dense singleton mixed proof at a small variable count.
- One-hot mixed proof at a small variable count if the A side path shares the same mixed quotient logic.
- Homogeneous D64 proof through the mixed parameter path.
- Bad setup generation dimension rejection.
- Bad proof dimension metadata rejection.

Benchmarks and diagnostics:

- Profile dense fp128 D64 versus D128 for whole-level homogeneous schedules.
- Profile mixed A dimensions in the first fold and in a later fold.
- Report proof bytes, prover time, verifier time, prepared cache bytes, and setup preparation time.
- Add profiler output for every prepared NTT slot: dimension, natural prefix length, padded prefix length if any, and cache bytes.

### Performance Policy

The planner should not use proof bytes alone.
A candidate should be scored with at least these values:

- estimated proof bytes;
- estimated next witness length;
- A side flat output size `n_a * D_A`;
- `num_digits_fold`;
- extra prepared setup cache bytes;
- setup preparation time, if measured data exists;
- expected kernel speedup from the larger A dimension, if measured data exists.

The first policy can be conservative.
It can reject mixed candidates that introduce a full extra cache in fold 0 or fold 1.
It can allow larger A dimensions in later folds where the active setup prefix is small.
This matches the expected memory shape: an early fold touches a large setup prefix, while a later fold touches only a small envelope of A rows.

## Design

### Level Parameter Shape

The current `LevelParams` has one `ring_dimension`.
The mixed design needs a role dimension shape:

```text
RowRingDimensions {
    d_a: usize,
    d_bd: usize,
}
```

The homogeneous case is:

```text
d_a = d_bd = ring_dimension
```

The first implementation should keep B and D together because their rows are more self-contained and because the current relation code fuses them.
The field should be named by role, not by implementation detail.
For example, prefer `row_dimensions.d_a` and `row_dimensions.d_bd` over names such as `large_d` and `small_d`.

Validation rules:

- `d_a` and `d_bd` must be supported CRT NTT dimensions.
- `d_a` and `d_bd` must divide `D_gen`.
- `d_bd` must be 64 in the first implementation.
- `d_a` must be 64, 128, or 256 in the first implementation.
- every ring count and field count multiplication must be checked.

### Setup Generation

`AkitaSetupSeed` currently stores `gen_ring_dim`.
That field should become the generation dimension for the flat setup vector, not necessarily the current level dimension.
For mixed D, `gen_ring_dim` is a physical setup property.
The schedule chooses runtime views into that physical setup.

The raw expanded setup remains a flat matrix.
This is already the right storage model.
The mixed path should not duplicate raw setup data per dimension.

The setup envelope must be computed in field coefficients or in generated D slots with a clear conversion.
If the largest runtime dimension is D256, then storing `max_setup_len` at `D_gen = 256` can serve D128 and D64 views by splitting coefficients.
The implementation must be careful not to compare a D64 runtime ring count directly with a D256 generated ring count without converting.

Disk persistence needs an explicit cache identity update.
A cached setup generated at D256 can serve D64 and D128 runtime views.
A cached setup generated at D64 cannot serve D128.
The cache key should therefore bind:

- field family;
- setup seed;
- generation dimension;
- max variable count;
- max batch count;
- setup envelope in generation slots;
- ZK setup envelopes if the `zk` feature is enabled.

### Prepared Setup Caches

The current CPU prepared setup owns one full shared `NttSlotCache<D>`.
That shape is too coarse for mixed D.

The mixed prepared setup should be a cache registry keyed by:

```text
PreparedSetupCacheKey {
    role: SetupRole,
    d: usize,
    natural_ring_len: usize,
    cyclic: bool,
    negacyclic: bool,
}
```

The exact Rust shape may differ, but the key must contain the dimension and the prefix length.
The role should be explicit because A, B, D, ZK B, and ZK D may need different backing matrices.

The implementation should avoid building cyclic and negacyclic views when only one is needed.
If the kernel still needs both views for a fused operation, the cache can store both.
The API should not force both for every future use.

Cache policy:

- Build caches lazily where possible.
- Reuse exact prefix caches.
- Allow a larger prefix cache to serve a smaller prefix only if the indexing contract is clear and tested.
- Report cache bytes by cache key.
- Enforce a configurable maximum prepared cache byte budget in profile and benchmark paths.

This is the most important implementation guardrail.
Without it, mixed D can increase memory by preparing D64, D128, and D256 full shared caches for the same setup.

### Ring Relation Prover

The current fused relation path assumes one const `D`.
Mixed D needs the relation to split work by row group.

For the first implementation:

- keep B and D rows in the D64 relation path;
- move A quotient rows into an A dimension path;
- combine the resulting row values only at the typed proof assembly boundary;
- keep row order the same as today.

The quotient API should express the row group it computes.
A sketch:

```text
compute_a_quotient_rows<D_A>(...)
compute_bd_cyclic_rows<D_BD>(...)
```

This is clearer than one generic `compute_relation_quotient` that hides mixed dimensions internally.
The old homogeneous path can become the `D_A = D_BD` specialization once tests show the outputs match.

The important point is that A rows and B/D rows no longer share one prepared `NttSlotCache<D>`.
They request the cache that matches their row dimension and active setup prefix.

### Commit Paths

Root dense and recursive witness commit paths already operate on flat witness coefficients at their boundary.
The mixed design should preserve that boundary.
The commit path chooses the A row dimension for A side rows.

The implementation must audit every use of:

- `ring_dimension`;
- `total_ring_elements_at::<D>()`;
- `ring_view::<D>()`;
- `PreparedSetup<D>`;
- `NttSlotCache<D>`;
- `num_digits_fold`;
- `level_proof_bytes`;
- `stage3_setup_product_bytes`.

Any use that refers to A rows must use `D_A`.
Any use that refers to B or D rows must use D64.
Any use that refers to a flat setup prefix must be explicit about whether the unit is field coefficients, generation-dimension ring slots, or runtime-dimension ring slots.

### Verifier

The verifier does not need prepared NTT setup caches, but it must validate the same shape.
Verifier-visible parameters must bind:

- `D_gen`;
- `D_A`;
- `D_BD`;
- row layout;
- row counts;
- challenge shapes;
- proof byte shape.

The verifier must reject a proof if any row group length is not consistent with its dimension.
It must also reject a setup whose generation dimension cannot serve the schedule.

### Planner

The planner should search over `D_A` and keep `D_BD` fixed at 64 in this phase.
For every candidate level it should derive:

- `n_a` using `D_A`;
- `n_b` using D64;
- `n_d` using D64;
- A side challenge norm and fold witness infinity cap using `D_A`;
- `num_digits_fold` using the selected fold challenge shape;
- next witness length using the mixed row dimensions;
- proof bytes using the mixed row dimensions;
- prepared setup cache cost for every new cache required by the level.

The heuristic should not simply prefer the largest dimension that does not increase `n_a * D_A`.
That rule is useful as a local tie-breaker, but it misses recursive effects.
The dynamic program should minimize total cost over the full fold sequence.
The local tie-breaker can still be:

```text
prefer smaller n_a * D_A;
then prefer fewer fold digits;
then prefer smaller next witness length;
then prefer larger D_A only if cache cost and proof bytes tie.
```

### Proof Size

The current proof-size formula assumes one level dimension.
Mixed proof-size accounting must identify which serialized objects are priced with `D_A`, which are priced with D64, and which are flat field elements.

The first implementation should not rely on estimates alone.
For each supported mixed shape, build a dummy proof body and compare:

```text
formula bytes == AkitaSerialize::serialized_size()
```

The stage-3 setup-product proof also needs attention.
Its round count includes ring-coordinate bits and setup-ring-index bits.
If setup offload stays fixed at D64, the mixed A dimension does not directly change that stage.
If later work allows setup-product proofs at `D_A`, the formula must become group-specific.

## Alternatives Considered

### Homogeneous D128 or D256 Levels

The simplest implementation is to move the whole level to D128 or D256.
This keeps the relation code simple.
It also increases proof bytes and setup cache size for B and D rows that do not need the larger ring.
The preview suggests that the useful part of the change often comes from A rows, so homogeneous larger D is too blunt.

### Tower Lowering to D64

Another option is to lower D128 rows into a D64 module representation.
This gives one quotient denominator for the whole level.
It also removes the main reason to use D128 as a row ring and adds representation complexity.
The row-specific quotient model is cleaner because each row uses its natural ring.

### Planner-Only Mixed D

A planner-only experiment can estimate proof bytes.
It cannot prove that the implementation is useful because prepared setup caches and relation kernels dominate the risk.
The preview already showed promising proof bytes.
The next design step must include cache and runtime cost.

### Full Per-Row Dynamic Dimensions

The most general design assigns a dimension to every row.
That is not needed now.
A role-group design captures the target lever and keeps row layout, transcript binding, and code review manageable.

## Documentation

This spec is the design record for the first mixed row dimension PR.
If the design ships, the durable book content should explain:

- why the setup is generated at a maximum physical dimension;
- why each row group can use its own quotient denominator;
- how the transcript binds mixed row dimensions;
- how prover setup caches are bounded.

The likely book owner is the protocol or setup chapter once mixed D becomes implemented.
Until then, this spec stays in `specs/`.

## Execution

### Suggested PR Slices

1. Add mixed parameter types and validation.
2. Update setup generation and disk cache identity so `D_gen` is physical, not the same as every runtime row dimension.
3. Add prepared setup cache reporting and prefix-aware cache APIs.
4. Split ring relation prover APIs into A quotient work and B/D cyclic work.
5. Add homogeneous tests through the new mixed shape.
6. Add one mixed D proof path with `D_A = 128`, `D_BD = 64`.
7. Add planner search over `D_A`.
8. Add cache-aware planner policy and profile output.
9. Add D256 only after D128 is correct and measured.

### Code Surface

Expected crates and modules:

- `akita-types`
  - `layout::params`
  - `proof::ring_relation`
  - `proof_size`
  - `proof::setup`
  - `proof::setup_prefix`
- `akita-config`
  - `CommitmentConfig`
  - proof optimized presets
  - generated table hooks
  - the scratch planner preview should either become a checked-in dev tool or be removed
- `akita-planner`
  - candidate derivation
  - schedule scoring
  - table expansion
- `akita-setup`
  - setup generation
  - disk persistence
  - setup-prefix population
- `akita-prover`
  - compute backend traits
  - CPU prepared setup
  - CRT NTT slot construction
  - dense and recursive commit rows
  - ring-switch relation quotient code
  - setup-product prover if mixed setup offload is enabled
- `akita-verifier`
  - setup validation
  - ring relation verification
  - setup contribution evaluation
- `akita-pcs`
  - end-to-end tests
  - profile example reporting

### Open Questions

- Should D256 be enabled in the first implementation or only after D128 lands?
- Should the cache budget be a hard rejection in the planner or a profile-only warning at first?
- Should prepared setup cache keys use exact prefixes only, or can a larger prepared prefix serve smaller requests?
- Should setup-prefix offload stay fixed at D64 while A rows use D128 and D256?
- Should generated schedule tables include mixed D immediately, or should mixed D use planner DP only until measurements settle?

The conservative answers are:

- land D128 first;
- use a hard cache budget in planner experiments and a diagnostic warning in production until benchmarks stabilize;
- use exact prefix cache keys first;
- keep setup-prefix offload at D64 first;
- use planner DP first, then generate tables after the protocol and cache policy stop changing.

### Risks

- Early fold mixed D can allocate too much prepared setup memory.
- The relation code can accidentally compute a quotient at the wrong dimension if APIs keep one generic `D`.
- Setup envelope comparisons can mix field coefficients, generation slots, and runtime slots.
- Proof-size formulas can undercount if they assume one dimension for every row.
- D256 can look good in proof bytes while losing to D128 after cache and kernel costs are measured.

## References

- `crates/akita-types/src/layout/flat_matrix.rs` for the existing flat setup storage and `ring_view` model.
- `crates/akita-prover/src/compute/cpu.rs` for the current full shared `NttSlotCache<D>` prepared setup model.
- `crates/akita-prover/src/kernels/crt_ntt.rs` for NTT cache byte reporting.
- `crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs` for the current homogeneous quotient path.
- `crates/akita-types/src/proof_size.rs` for level proof byte formulas.
- `crates/akita-types/src/sis/norm_bound.rs` for D-specific fold challenge and digit sizing.
- `specs/setup-prefix-ladder.md` for setup-prefix cache and prefix identity context.
- Scratch preview: `crates/akita-config/src/bin/mixed_d_preview.rs` on branch `quang/mixed-row-ring-dimensions`.
