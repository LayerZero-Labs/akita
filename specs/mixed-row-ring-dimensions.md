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
A rows interpret the relevant flat coefficients as elements of `R_da = F[X] / (X^d_a + 1)`.
B and D rows interpret their coefficients as elements of `R_dbd = F[X] / (X^d_bd + 1)`.
The prover computes one quotient per row group using that row group's denominator.

The main risk is no longer algebraic soundness.
The main risk is engineering cost and memory cost.
The current prover prepares one full shared NTT cache for one const `D`.
A naive mixed D implementation would prepare several full caches for the same expanded setup.
That can erase the proof-size and speed gains, especially when a larger A dimension appears in the first fold levels.
The implementation must therefore make setup preparation aware of row groups, active prefixes, and memory budgets.

## Intent

### Goal

Support a mixed row dimension fold where A rows may use a larger ring dimension than B and D rows within the same fold level.
B and D rows stay at the profile floor dimension `D_floor`.
A rows may lift to `d_a` in the profile allowed A-lift set.
The planner chooses `d_a` only when proof size, security, runtime, and prepared setup memory all improve or stay within policy for that profile.

Key surfaces:

- `LevelParams` gains a row dimension shape instead of one `ring_dimension` for every row.
- The ring relation becomes a row-group relation with one quotient denominator per group.
- Setup generation uses a maximum generation dimension `D_gen` that every runtime row dimension divides.
- Prover setup preparation materializes NTT caches by dimension and active prefix, not only by dimension.
- Planner scoring is profile-relative: it uses `PlannerPolicy` / `Cfg::D` as `D_floor`, the profile SIS family, and `ring_challenge_config(d)` at each runtime dimension.
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
- **SIS sizing is group-specific.** A rows use `d_a` and `ring_challenge_config(d_a)` when deriving `n_a`. B and D rows use `d_bd = D_floor` and `ring_challenge_config(d_bd)` when deriving `n_b` and `n_d`.
- **Fold digit counts are priced with the selected A dimension.** Larger `d_a` can change the folded `z` infinity bound and `num_digits_fold`. The planner must include that effect using the profile challenge family at `d_a`.
- **Proof-size estimates must match serialized proofs.** Any mixed proof byte formula must be backed by serialization tests after implementation.
- **Setup preparation is bounded.** The prover must not prepare full shared setup NTT caches for every supported dimension by default.
- **Verifier code stays inside the no-panic boundary.** Bad mixed parameters, bad setup shape, bad proof row counts, and mismatched dimensions return `AkitaError` or serialization errors.
- **No backward compatibility shim is required.** This is a new protocol shape. Old homogeneous schedules do not need aliases or migration layers beyond staying valid as the special case `d_a = d_bd = D_floor`.

### Non-Goals

- No arbitrary runtime dimension set. Supported CRT dimensions remain `{32, 64, 128, 256}`.
- No mixed dimensions for B versus D in the first implementation.
- No tower-lowered protocol that rewrites every row into the smallest ring.
- No dynamic per-row dimension inside one row group. Dimensions are per role group.
- No targeting non-securable homogeneous floors (for example fp32/fp64 D32/D64 under honest SIS pricing) as shipping mixed-D profiles in v1.
- No blocking D=32 floor support in types or planner APIs. D=32 is a plausible future `D_floor` once mixed A is measured; see the Planner Profile Model section.
- No GPU or Metal backend work in this spec.
- No attempt to make recursive setup offload, setup-prefix commitments, and mixed D all land in one PR unless the implementation shows this is smaller than separating them.

## Evaluation

### Planner Preview (fp128 D64 example)

A scratch dynamic programming preview was run on the branch `quang/mixed-row-ring-dimensions`.
The preview fixed `d_bd = D_floor = 64` for an fp128 `D64Full` profile, allowed `d_a` in `{64, 128, 256}`, and let the planner rechoose `log_basis` and the `(m, r)` split at every fold.
It also priced A side SIS ranks with the selected `d_a` and included the effect on `num_digits_fold` and the next witness length.

This table is one profile-specific data point, not the general mixed-D definition.
Packet 1 must rerun the preview across the profile matrix in the Planner Profile Model section.

The fp128 D64 preview found consistent proof-size reductions for dense singleton shapes:

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

### Required Cache-Aware Preview Gate

Before protocol implementation starts, the scratch preview must become cache-aware enough to answer whether the proof-size win survives a prepared setup budget.
This gate is required because prepared NTT caches can be much larger than the raw setup matrix.

The preview must compare three schedules per profile:

- homogeneous baseline at `D_floor`;
- mixed D with no prepared cache cap;
- mixed D with the prepared cache cap below.

The first cap is profile-relative:

```text
mixed_prepared_cache_bytes(profile) <= min(
    baseline_prepared_cache_bytes(profile) * 5 / 4,
    baseline_prepared_cache_bytes(profile) + 256 MiB
)
```

where `baseline_prepared_cache_bytes(profile)` is the homogeneous schedule at that profile's `D_floor`.

This cap is intentionally conservative.
It should reject schedules that need a full extra early lift shared setup cache at a dimension well above `D_floor`.
It should still allow later mixed A dimensions when the active setup prefix is small.

The preview must count unique cache keys across the whole schedule.
It must not charge the same cache once per fold if the prepared setup would reuse it.
The cache key for the preview is:

```text
CacheKey {
    role,
    d,
    natural_ring_len,
    stores_negacyclic,
    stores_cyclic,
}
```

The preview should initially use exact prefix keys only.
A larger prepared prefix must not serve a smaller request in the first model.
That reuse rule can be added later only after the implementation has a tested indexing contract.

The byte model should mirror `NttSlotCache`.
For a cache with `n` ring elements at dimension `D`, it stores one negacyclic vector if `stores_negacyclic` is true and one cyclic vector if `stores_cyclic` is true.
The per-element byte size must use the selected CRT profile for the field and `D`.
If the scratch preview cannot call the exact Rust type size, it must print that the cache bytes are modeled and state the assumed bytes per cached ring element.

### Required Proof-Size Attribution

Every preview run must explain where the proof-size delta comes from.
It must split the total delta into:

- non-terminal fold proof bytes;
- terminal tail bytes;
- recursive setup-product bytes, if that mode is included;
- any other bytes that do not fit the first three groups.

For each `num_vars`, the report must print:

```text
baseline_total
mixed_total
total_delta
fold_delta
tail_delta
setup_product_delta
other_delta
baseline_cache_bytes
mixed_cache_bytes
cache_delta
```

For each fold level, the report must print:

```text
level
d_a
d_bd
log_basis
r_vars
block_len
n_a
n_b
n_d
n_a_times_d_a
num_digits_fold
next_w_len
fold_proof_bytes
tail_bytes_if_terminal
new_cache_bytes
cache_key_count
```

The report must make the following identity check explicit:

```text
total_delta == fold_delta + tail_delta + setup_product_delta + other_delta
```

If the mixed schedule still wins after the cache cap, the report should state whether the win mostly comes from smaller fold payloads, a smaller tail, or both.
If the win comes mostly from the tail, the implementation should be treated as lower priority.
If the win reduces intermediate next witness lengths and compounds over several folds, the implementation is higher priority.

### Acceptance Criteria

- [ ] A homogeneous schedule with `d_a = d_bd = D_floor` produces the same rows, proof bytes, and verifier behavior as the current single D path for that profile.
- [ ] A mixed schedule can lift `d_a` above `D_floor` while keeping B and D rows at `d_bd = D_floor`.
- [ ] The prover computes A quotients with `X^d_a + 1` and B/D quotients with `X^d_bd + 1`.
- [ ] The verifier binds mixed row dimensions in the instance descriptor and rejects proofs whose dimensions do not match the schedule.
- [ ] Setup generation at `D_gen = max(runtime dimensions used by the schedule)` can serve every row group view used by that profile.
- [ ] Prepared setup memory is measured and bounded. A run that uses both `D_floor` and a larger `d_a` must not automatically build full shared caches for both dimensions if only a small prefix needs the larger dimension.
- [ ] The planner can reject or penalize candidates whose extra prepared cache footprint exceeds a policy limit.
- [ ] The cache-aware preview prints fold versus tail proof-size attribution and passes the delta identity check above.
- [ ] The implementation work starts with one reference profile before generalizing. The first protocol slice may land on fp128 `D64Full`, but types and planner APIs must be profile-parameterized from the start.
- [ ] Larger `d_a` values above the first lift step are enabled only after the first lift is correct, measured, and still leaves a reason to test the next dimension.
- [ ] Proof-size formula tests compare mixed formula output against serialized mixed proofs.
- [ ] End-to-end prove and verify tests cover at least one homogeneous schedule at `D_floor` and one mixed schedule with `d_a > D_floor`.
- [ ] `cargo fmt -q`, `cargo clippy --all --message-format=short -q -- -D warnings`, `cargo test`, and `./scripts/check-doc-guardrails.sh` pass before the implementation PR is merged.

### Testing Strategy

Unit tests:

- `akita-types` tests for mixed level parameter validation.
- `akita-types` tests for setup generation dimension divisibility.
- `akita-types` tests that homogeneous mixed params serialize to the same logical schedule as the current shape, or to a clearly documented new shape with equivalent meaning.
- `akita-planner` tests that `n_a` changes with `d_a` while `n_b` and `n_d` stay tied to `d_bd = D_floor`.
- `akita-prover` quotient tests that compare row-specific quotient results against direct schoolbook calculations at `D_floor` and at least one lifted `d_a`.
- `akita-prover` prepared setup tests that request a small lifted-dimension prefix and assert that reported cache bytes match that prefix, not the full setup length.

End-to-end tests:

- Dense singleton mixed proof at a small variable count.
- One-hot mixed proof at a small variable count if the A side path shares the same mixed quotient logic.
- Homogeneous proof at `D_floor` through the mixed parameter path.
- Bad setup generation dimension rejection.
- Bad proof dimension metadata rejection.

Benchmarks and diagnostics:

- Profile dense schedules at `D_floor` versus a homogeneous schedule at the next supported dimension.
- Profile mixed A lifts in the first fold and in a later fold for each target profile.
- Report proof bytes, prover time, verifier time, prepared cache bytes, and setup preparation time.
- Add profiler output for every prepared NTT slot: dimension, natural prefix length, padded prefix length if any, and cache bytes.

### Performance Policy

The planner should not use proof bytes alone.
A candidate should be scored with at least these values:

- estimated proof bytes;
- estimated next witness length;
- A side flat output size `n_a * d_a`;
- `num_digits_fold`;
- extra prepared setup cache bytes;
- setup preparation time, if measured data exists;
- expected kernel speedup from the larger A dimension, if measured data exists.

The first policy is a hard rejection in planner experiments.
Production code may begin with diagnostics only, but planner experiments must reject candidates that exceed the cap.
This prevents a proof-size-only model from selecting schedules that are not useful in memory.

The first implementation should not have a hidden fallback that silently ignores the cache cap.
If a schedule exceeds the cap, the planner report must show the rejected candidate and its cache bytes.
This makes it clear whether the cap is killing higher lifts, early lifts above `D_floor`, or all mixed D for that profile.

The policy should allow larger A dimensions in later folds where the active setup prefix is small.
This matches the expected memory shape.
An early fold touches a large setup prefix.
A later fold touches only a small envelope of A rows.

## Design

### Planner Profile Model

Mixed D is profile-relative, not fp128 D64 specific.

A planner profile is the plain-value input the DP already uses through `PlannerPolicy` and `policy_of::<Cfg>()`:

- `PlannerPolicy.ring_dimension` is the homogeneous floor dimension `D_floor`
- `sis_modulus_family` selects the SIS floor table
- `ring_challenge_config(d)` selects the fold-challenge family at runtime dimension `d`
- decomposition, onehot chunk size, and tiered flags remain profile-local

At one mixed fold level:

- `d_bd = D_floor` for B and D rows
- `d_a` is chosen from the profile A-lift set
- `D_gen` is the maximum runtime dimension used by that profile schedule envelope

Representative shipped profiles:

| Field family | Example preset | `D_floor` | Typical A-lift set |
|--------------|----------------|-----------|--------------------|
| fp128 | `D64Full`, `D64OneHot` | 64 | `{64, 128, 256}` |
| fp128 | `D128Full`, `D128OneHot` | 128 | `{128, 256}` |
| fp64 | `D128Full`, `D128OneHot` | 128 | `{128, 256}` |
| fp64 | `D256OneHot` | 256 | `{256}` |
| fp32 | `D128OneHot` | 128 | `{128, 256}` |
| fp32 | `D256OneHot` | 256 | `{256}` |

**D=32 floor (not in v1 shipping set, but keep the design open).**
The repo already has fp128 `D32Full` and `D32OneHot` presets with `D_floor = 32`.
Homogeneous D=32 does not clearly win today under current planner pricing and SIS tables.
Mixed A may change that calculus: keeping B and D at `d_bd = 32` while lifting A to `{64, 128, 256}` could recover security and kernel benefits without paying full homogeneous D64/D128 proof-size cost on every row group.
Packet 1 does not need to block v1 on D=32 results, but types, preview tooling, and planner APIs must not assume `D_floor >= 64`.
When mixed D is stable, rerun the cache-aware preview on fp128 `D32Full` and decide whether D=32 belongs in the shipping profile matrix.

Supported CRT dimensions remain `{32, 64, 128, 256}` from `proof_optimized_ring_challenge_config`.
The A-lift set for a profile is:

```text
A_lift(D_floor) = { d | d in {32,64,128,256}, d >= D_floor }
```

Non-securable homogeneous floors such as fp32/fp64 D32/D64 under honest SIS pricing are out of scope for shipping mixed D.
The preview may study them, but implementation should not target them.

Sizing rules:

- `n_a`, fold witness infinity cap, and `num_digits_fold` use `d_a` and `ring_challenge_config(d_a)`
- `n_b` and `n_d` use `d_bd` and `ring_challenge_config(d_bd)`
- SIS floor lookup uses `(sis_family, d, norm, width)` separately for each group

Packet 1 must evaluate at least:

- fp128 `D64Full`
- fp128 `D128Full`
- fp64 `D128OneHot`
- fp32 `D128OneHot`

The fp128 D64 table in the Evaluation section is one example output, not the general contract.

### Level Parameter Shape

The current `LevelParams` has one `ring_dimension`.
The mixed design needs a role dimension shape:

```text
RowRingDimensions {
    d_a: usize,
    d_bd: usize,
}
```

The homogeneous case for a profile is:

```text
d_a = d_bd = D_floor
```

where `D_floor` is that profile's `Cfg::D`.

The first implementation should keep B and D together because their rows are more self-contained and because the current relation code fuses them.
The field should be named by role, not by implementation detail.
For example, prefer `row_dimensions.d_a` and `row_dimensions.d_bd` over names such as `large_d` and `small_d`.

Validation rules:

- `d_a` and `d_bd` must be supported CRT NTT dimensions.
- `d_a` and `d_bd` must divide `D_gen`.
- `d_bd` must equal the profile `D_floor` in the first implementation.
- `d_a` must lie in `A_lift(D_floor)` for that profile.
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

The first implementation may use one shared role for the public shared matrix if that is what the existing setup storage exposes.
Even then, the cache key must still include the logical row role at the API boundary.
This prevents A code from accidentally borrowing a B/D cache with the same dimension but a different active prefix.

Cache policy:

- Build caches lazily where possible.
- Reuse exact prefix caches.
- Allow a larger prefix cache to serve a smaller prefix only if the indexing contract is clear and tested.
- Report cache bytes by cache key.
- Enforce a configurable maximum prepared cache byte budget in profile and benchmark paths.
- In planner experiments, use the hard cap from the Required Cache-Aware Preview Gate section.
- In production prover setup, start with diagnostics and explicit byte reporting before adding hard user-facing failures.

This is the most important implementation guardrail.
Without it, mixed D can increase memory by preparing full shared caches at `D_floor` and at every lifted `d_a` used by the schedule.

### Ring Relation Prover

The current fused relation path assumes one const `D`.
Mixed D needs the relation to split work by row group.

For the first implementation:

- keep B and D rows in the `d_bd = D_floor` relation path;
- move A quotient rows into the `d_a` dimension path;
- combine the resulting row values only at the typed proof assembly boundary;
- keep row order the same as today.

The quotient API should express the row group it computes.
A sketch:

```text
compute_a_quotient_rows<d_a>(...)
compute_bd_cyclic_rows<d_bd>(...)
```

This is clearer than one generic `compute_relation_quotient` that hides mixed dimensions internally.
The old homogeneous path can become the `d_a = d_bd = D_floor` specialization once tests show the outputs match.

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

Any use that refers to A rows must use `d_a`.
Any use that refers to B or D rows must use `d_bd`.
Any use that refers to a flat setup prefix must be explicit about whether the unit is field coefficients, generation-dimension ring slots, or runtime-dimension ring slots.

### Setup-Prefix Offload Dimension

Current setup-prefix offload uses `SETUP_OFFLOAD_D_SETUP = 64`.
That is correct for fp128 profiles with `D_floor = 64`.
For profiles with `D_floor = 128` or `256`, setup-prefix offload must eventually follow the profile floor or a dedicated profile hook.
The first protocol slice may keep the existing D64 offload behavior for fp128 only, but the mixed-D types must not hardcode `64` as the only valid offload dimension.

### Verifier

The verifier does not need prepared NTT setup caches, but it must validate the same shape.
Verifier-visible parameters must bind:

- `D_gen`;
- `d_a`;
- `d_bd`;
- profile identity fields needed to recover `D_floor`, SIS family, and challenge policy;
- row layout;
- row counts;
- challenge shapes;
- proof byte shape.

The verifier must reject a proof if any row group length is not consistent with its dimension.
It must also reject a setup whose generation dimension cannot serve the schedule.

### Planner

The planner should search over `d_a in A_lift(D_floor)` and keep `d_bd = D_floor` in this phase.
For every candidate level it should derive:

- `n_a` using `d_a` and `ring_challenge_config(d_a)`;
- `n_b` using `d_bd` and `ring_challenge_config(d_bd)`;
- `n_d` using `d_bd` and `ring_challenge_config(d_bd)`;
- A side challenge norm and fold witness infinity cap using `d_a`;
- `num_digits_fold` using the selected fold challenge shape;
- next witness length using the mixed row dimensions;
- proof bytes using the mixed row dimensions;
- prepared setup cache cost for every new cache required by the level.

The heuristic should not simply prefer the largest dimension that does not increase `n_a * d_a`.
That rule is useful as a local tie-breaker, but it misses recursive effects.
The dynamic program should minimize total cost over the full fold sequence.
The local tie-breaker can still be:

```text
prefer smaller n_a * d_a;
then prefer fewer fold digits;
then prefer smaller next witness length;
then prefer larger d_a only if cache cost and proof bytes tie.
```

### Proof Size

The current proof-size formula assumes one level dimension.
Mixed proof-size accounting must identify which serialized objects are priced with `d_a`, which are priced with `d_bd`, and which are flat field elements.

The first implementation should not rely on estimates alone.
For each supported mixed shape, build a dummy proof body and compare:

```text
formula bytes == AkitaSerialize::serialized_size()
```

The stage-3 setup-product proof also needs attention.
Its round count includes ring-coordinate bits and setup-ring-index bits.
If setup offload stays at the current profile-specific constant (today `SETUP_OFFLOAD_D_SETUP = 64` for fp128 D64-floor profiles), the mixed `d_a` dimension does not directly change that stage for those profiles.
If later work allows setup-product proofs at `d_a` or at `D_floor` for higher-floor profiles, the formula must become group-specific and profile-specific.

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

### Resolved Implementation Decisions

The following decisions are fixed for the first implementation.
Agents should not reopen them unless a test or benchmark shows they are wrong.

- Implement the first protocol slice on one reference profile, likely fp128 `D64Full`, but keep all APIs profile-parameterized.
- Keep `d_bd = D_floor` for B and D rows.
- Keep setup-prefix offload at the current profile-specific constant for the first protocol slice only. Today that is `SETUP_OFFLOAD_D_SETUP = 64` for fp128 D64-floor profiles.
- Use exact prepared cache prefix keys first.
- Add larger-prefix cache reuse only after exact-prefix caching works and has tests.
- Use planner DP first.
- Generate mixed schedule tables only after the protocol shape and cache policy are stable.
- Treat cache cap failure as a planner rejection in experiments.
- Treat cache cap failure as a diagnostic in production until profile data justifies a user-facing hard failure.
- Keep raw setup storage flat and generated at `D_gen = max(runtime dimensions)`.
- Do not add compatibility shims for old mixed-D experiments.

### Agent Task Packets

Each packet below should be small enough for an independent implementation agent.
Agents should complete packets in order unless the previous packet has already landed.

Packet 1 gates all protocol work.
Do not start Packet 2 until Packet 1 says whether the first useful A lift above `D_floor` survives the cache cap for each target profile and where the proof-size delta comes from.
If Packet 1 shows that mixed D only wins in the terminal tail, stop and reassess before changing protocol code.
If Packet 1 shows no mixed D candidate survives the cache cap for any target profile, do not implement mixed D.

General rules for every agent:

- Keep each packet in one focused PR.
- Do not add compatibility aliases or fallback paths for unfinished mixed D shapes.
- Do not change transcript bytes unless the packet explicitly says to change verifier-visible parameters.
- Do not widen the supported dimension set beyond the packet.
- Do not use unchecked indexing, unchecked slicing, `unwrap`, `expect`, or `panic` in verifier-reachable code.
- Report the exact commands run and the result.
- If a packet exposes a larger design problem, stop and write the blocker in the PR description instead of patching around it.

#### Packet 1: Cache-Aware Planner Preview

Goal:
Turn the scratch mixed D preview into a cache-aware report that can decide whether implementation is worth doing.

Files:

- `crates/akita-config/src/bin/mixed_d_preview.rs`, or a checked-in dev tool with a clearer name.
- No protocol files.

Required output:

- Homogeneous baseline at each profile `D_floor`.
- Mixed D schedule with no cache cap per profile.
- Mixed D schedule with the hard cache cap per profile.
- Fold versus tail proof-size attribution per profile.
- Unique cache key count and cache bytes per profile.
- Rejected candidates that exceed the cap.

Acceptance:

- The report prints the delta identity check.
- The report shows whether proof-size savings come from non-terminal folds or the terminal tail.
- The report states whether the first useful A lift above `D_floor` still wins after the cache cap for each target profile.
- The report states whether any higher lift survives after the cap.
- The scratch binary is either committed as a dev tool or removed before the implementation PR.

#### Packet 2: Mixed Parameter Types

Goal:
Represent mixed row dimensions without changing prover behavior.

Files:

- `crates/akita-types/src/layout/params.rs`
- `crates/akita-types/src/proof/ring_relation.rs`
- serialization and validation modules that mention `LevelParams`
- tests near existing `LevelParams` tests

Required behavior:

- Add `RowRingDimensions { d_a, d_bd }`, or an equivalent type with role-based names.
- Preserve homogeneous behavior at `d_a = d_bd = D_floor`.
- Validate supported dimensions and divisibility by `D_gen`.
- Keep B and D tied to `d_bd = D_floor` for now.

Acceptance:

- Existing schedules still validate.
- Homogeneous params map to `d_a = d_bd = D_floor`.
- Invalid `d_a`, invalid `d_bd`, and invalid `D_gen` return errors.

#### Packet 3: Setup Generation Dimension

Goal:
Separate physical setup generation dimension from runtime row dimensions.

Files:

- `crates/akita-types/src/proof/setup.rs`
- `crates/akita-setup/src/lib.rs`
- `crates/akita-config/src/proof_optimized.rs`
- disk persistence code under `feature = "disk-persistence"`

Required behavior:

- Treat `AkitaSetupSeed::gen_ring_dim` as physical `D_gen`.
- Compare setup envelopes only after converting to the same unit.
- Allow a larger `D_gen` generated setup to serve smaller runtime views.
- Reject a generated setup whose `D_gen` is too small for a lifted runtime row.

Acceptance:

- Disk cache keys distinguish generated setup by `D_gen`, not only by one runtime dimension.
- Setup validation has no unchecked arithmetic.
- Existing homogeneous setup generation still works for every shipped profile floor.

#### Packet 4: Prefix-Aware Prepared Setup Caches

Goal:
Stop prepared setup from meaning one full shared cache per `D`.

Files:

- `crates/akita-prover/src/compute/backend.rs`
- `crates/akita-prover/src/compute/cpu.rs`
- `crates/akita-prover/src/kernels/crt_ntt.rs`
- tests under `crates/akita-prover/src/compute/`

Required behavior:

- Add a prepared cache key that includes role, dimension, and natural ring length.
- Build exact prefix caches lazily.
- Report cache bytes per key and total cache bytes.
- Keep the existing homogeneous path working.

Acceptance:

- A small lifted-dimension prefix request does not build a full shared cache at that dimension.
- Cache byte reporting matches the cache contents.
- Existing dense, one-hot, sparse, and recursive witness commit tests pass.

#### Packet 5: Split Ring Relation Work by Role

Goal:
Make the quotient code dimension-safe by construction.

Files:

- `crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs`
- `crates/akita-prover/src/compute/plans.rs`
- `crates/akita-prover/src/compute/backend.rs`
- `crates/akita-prover/src/compute/cpu.rs`
- quotient kernel tests

Required behavior:

- Compute A quotient rows through an A dimension API at `d_a`.
- Compute B and D cyclic rows through the B/D dimension API at `d_bd = D_floor`.
- Keep row assembly order unchanged.
- Make the homogeneous case at `d_a = d_bd = D_floor` match the old output.

Acceptance:

- Schoolbook quotient tests pass at `D_floor` and at least one lifted `d_a`.
- Homogeneous relation tests match old behavior at each profile floor.
- The API no longer lets a caller pass one generic `D` for all row groups in mixed mode.

#### Packet 6: First Mixed Proof For Reference Profile

Goal:
Produce and verify one mixed proof for a reference profile where `d_a > D_floor` and `d_bd = D_floor`.

The first likely reference is fp128 `D64Full` with `d_a = 128` and `d_bd = 64`.
The packet must generalize to other profiles without rewriting the protocol shape.

Files:

- prover protocol files touched by commit and ring switch
- verifier protocol files that validate row dimensions
- `akita-pcs` end-to-end tests
- proof-size tests

Required behavior:

- One dense singleton mixed proof verifies for the reference profile.
- Homogeneous proof through the mixed parameter path still verifies at `D_floor`.
- Proof-size formula matches serialized proof size.

Acceptance:

- End-to-end mixed proof test passes for the reference profile.
- Bad dimension metadata is rejected.
- Bad setup generation dimension is rejected.

#### Packet 7: Planner Integration

Goal:
Move the cache-aware preview logic into the real planner or table-generation path.

Files:

- `crates/akita-planner`
- `crates/akita-config`
- generated schedule tooling if needed

Required behavior:

- Search over `d_a in A_lift(D_floor)`.
- Keep `d_bd = D_floor`.
- Include cache cap in candidate rejection.
- Print proof-size attribution and cache attribution in profile or planner diagnostics.

Acceptance:

- Homogeneous schedules remain available at each profile `D_floor`.
- Lifted `d_a` schedules are selected only when they survive the cache cap.
- Higher lifts remain disabled unless earlier packets justify them.

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

### Remaining Questions

Only measurement questions remain.
They should be answered by Packet 1 before protocol work begins.

- Does the first useful A lift above `D_floor` reduce total proof bytes after the cache cap for each target profile?
- Is the remaining proof-size win mostly in non-terminal folds or mostly in the terminal tail?
- Do higher lifts survive the cache cap for any target profile?
- Does a homogeneous schedule at the next supported dimension show enough kernel speedup to justify mixed D for that profile?
- Does fp128 `D32Full` with mixed A become competitive once the cache-aware preview is rerun after v1 lands?
- Does the cache-aware planner choose later-fold lifts, early-fold lifts, or no lifts?

### Risks

- Early fold mixed D can allocate too much prepared setup memory.
- The relation code can accidentally compute a quotient at the wrong dimension if APIs keep one generic `D`.
- Setup envelope comparisons can mix field coefficients, generation slots, and runtime slots.
- Proof-size formulas can undercount if they assume one dimension for every row.
- Early fold mixed D can allocate too much prepared setup memory for profiles whose `D_floor` is already large.
- Higher lifts can look good in proof bytes for low-floor profiles while losing after cache and kernel costs are measured.

## References

- `crates/akita-types/src/layout/flat_matrix.rs` for the existing flat setup storage and `ring_view` model.
- `crates/akita-prover/src/compute/cpu.rs` for the current full shared `NttSlotCache<D>` prepared setup model.
- `crates/akita-prover/src/kernels/crt_ntt.rs` for NTT cache byte reporting.
- `crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs` for the current homogeneous quotient path.
- `crates/akita-types/src/proof_size.rs` for level proof byte formulas.
- `crates/akita-types/src/sis/norm_bound.rs` for D-specific fold challenge and digit sizing.
- `specs/setup-prefix-ladder.md` for setup-prefix cache and prefix identity context.
- Scratch preview: `crates/akita-config/src/bin/mixed_d_preview.rs` on branch `quang/mixed-row-ring-dimensions`.
