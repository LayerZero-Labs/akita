# Spec: D=256 and D=512 fp128 A-role readiness for future mixed-ring schedules

| Field         | Value                          |
|---------------|--------------------------------|
| Author(s)     |                                |
| Created       | 2026-07-24                     |
| Status        | active                         |
| PR            |                                |
| Supersedes    |                                |
| Superseded-by |                                |
| Book-chapter  |                                |

## Summary

Future mixed-ring-dimension schedules may assign a larger ring dimension to the
inner (A) commitment role at selected levels while retaining smaller dimensions
for other roles and later levels. This spec records two incremental capability
extensions for fp128:

1. `D = 256` inner-role dispatch, already implemented without adding a preset or
   schedule; and
2. `D = 512` inner-role support, to be added under the same mixed-ring-only
   motivation.

The `D = 512` extension is not only another dispatch literal. Its production
challenge, sampler, fold-tail certificate, fp128 NTT dispatch, and q128 CRT/NTT
profile already support `512`, but the audited SIS table and canonical A-role
coverage stop at `256`. This spec extends the estimator's canonical generation
domain and generates q128 Inner/A rows directly at `D = 512` for ranks
`1..=20`. It does not derive D512 values from another dimension. The rollout
preserves existing schedule identities and does not create or regenerate a
uniform `D = 512` schedule.

## Intent

### Goal

Make fp128 inner/A-role protocol code callable at `D ∈ {256, 512}` so later
production mixed-ring features can select either dimension for only the levels
and roles that benefit, without exposing a uniform fp128 `D256` or `D512`
configuration.

### Delivered slice: D=256 dispatch

The `D = 256` slice changed only:

- `crates/akita-types/src/dispatch/policy.rs` — added `256` to the fp128
  `__dispatch_for_field_inner!` arms and `protocol_dispatch_policy!` `inner`
  list.
- `crates/akita-types/src/dispatch/mod.rs` — updated focused role-dispatch
  acceptance/rejection tests.

This was sufficient because the production challenge ladder and audited
A-role SIS table already covered `D = 256`.

### Proposed slice: D=512 certification and dispatch

The `D = 512` slice has two ordered parts.

#### 1. Generate the exact role cell directly

- Extend canonical SIS coverage for exactly
  `(role = Inner, modulus = Q128OffsetA7F7, D = 512)`.
- Keep q128 outer/opening at their current dimensions. Do not make `D = 512`
  reachable for fp32/fp64 or for B/D merely because it is added for fp128 A.
- Add `512` to the estimator's dimension union. Its canonical role/profile
  filter must create work only for q128 Inner/A.
- Run the pinned lattice-estimator-compatible ADPS16 quantum LGSA path directly
  for every existing A-role coefficient bucket and rank `1..=20`.
- Emit the resulting D512 slices through the same generated-table shape used
  by every other dimension.
- Give the expanded table a distinct digest while continuing to recognize the
  existing digest for existing schedules and cells.

The current `A_ROLE_RING_DIMS` list is role-aware but not modulus-profile-aware.
It must not simply become `[64, 128, 256, 512]`, because that would declare
unused q32/q64 A-role `D = 512` cells. The canonical coverage predicate must
express the q128-only extension directly.

#### 2. Add fp128 inner dispatch

- `crates/akita-types/src/dispatch/policy.rs` — add `512` to both sources of
  fp128 inner-role dispatch policy:
  `__dispatch_for_field_inner!` and the `protocol_dispatch_policy!` `inner`
  list.
- `crates/akita-types/src/dispatch/mod.rs` — accept fp128 inner `D = 512` and
  continue rejecting `D = 32` and `D = 1024`.
- `crates/akita-prover/src/compute/poly.rs` and the generic prover entry points
  that use inner-role dispatch — extend their compile-time source/backend
  capability bounds through `D = 512`. The kernel implementations are already
  const-generic; this adds no schedule selection or protocol algorithm.

The dispatch change must not land before the security table can price the new
role cell.

### Existing prerequisites that remain unchanged

- `crates/akita-challenges/src/config.rs` already defines the production
  `D = 512` challenge as
  `SparseChallengeConfig { count_pm1: 19, count_pm2: 0 }`, validates its
  128-bit entropy floor, and certifies it for fold-`L∞` tail sizing.
- The sparse sampler already has a `257..=512` stack tier.
- fp128 NTT dispatch already includes `D = 512`.
- The q128 CRT/NTT profile already sets `Q128_MAX_RING_D = 512`; its i32 primes
  and universal i16 tail prime have the required transform order.

None of those surfaces needs a production change for this spec.

### Invariants

- **Role- and profile-specific capability.** The new security cell and dispatch
  arm apply only to fp128 inner/A at `D = 512`. No B/D role and no fp32/fp64
  protocol role is widened.
- **No uniform configuration becomes selectable.** No preset, planner
  candidate, schedule catalog entry, generated schedule table, or runtime
  fallback is added or changed.
- **Existing dispatch is stable.** Existing fp128 inner `D = 64`, `128`, and
  `256` behavior is unchanged. `D = 32` and `D = 1024` remain rejected.
- **The two dispatch authorities agree.** The concrete macro arms and
  declarative policy list contain the same fp128 inner dimensions:
  `{64, 128, 256, 512}`.
- **Security remains fail-closed.** A `D = 512` lookup is usable only under the
  new table digest and only for a directly generated q128 Inner/A row.
  Unclassified infinite estimates stop generation; unknown digests and
  unsupported cells reject.
- **Existing schedule identities remain stable.** Checked-in schedules retain
  their existing SIS table digest and catalog identities. The expanded table
  digest is additive and is reserved for a later mixed-ring configuration.
- **No proof or transcript change.** Because this spec adds no schedule that
  selects `D = 512`, existing proof bytes, transcript bytes, and instance
  descriptors remain unchanged.
- **One generation and lookup path.** D512 is estimated by the same canonical
  generator and returned by the same `sis_max_widths` slice lookup used by
  every other dimension.

### Non-Goals

- A uniform fp128 `D = 256` or `D = 512` preset or schedule.
- Any generated or runtime-DP-backed `D = 256`/`D = 512` schedule.
- Mixed ring dimensions per level or per role, schedule splicing, role
  compression, or a three-band schedule.
- Deciding which roles or levels a future mixed-ring configuration assigns
  `D = 256` or `D = 512`.
- Adding a config hook that selects the expanded SIS digest; that belongs to
  the first mixed-ring configuration that consumes `D = 512`.
- Verifier, planner, profile-harness, or end-to-end PCS changes.
- Prover algorithm, backend-kernel, or mixed-schedule changes beyond the
  compile-time capability bounds required by the new dispatch arm.
- Changes to challenge parameters, sampling, entropy policy, fold-tail policy,
  NTT primes, CRT profiles, or backend kernels.
- fp32/fp64 `D = 512` protocol-role support.
- fp128 B/D `D = 512` support.
- `D = 1024` or larger fp128 inner-role support.
- Copying or adapting implementation from the mixed-dimension demo branch.

## Evaluation

### D=256 acceptance criteria

- [x] fp128 inner-role dispatch accepts `D ∈ {64, 128, 256}`.
- [x] The fp128 outer, opening, envelope, and NTT policies are unchanged.
- [x] The fp64 and fp32 dispatch policies are unchanged.
- [x] No preset, schedule, generated artifact, planner, prover, verifier, or
      PCS change was required.
- [x] Repository formatting, dependency, unit-test, and Clippy gates passed.

### D=512 acceptance criteria

- [x] Canonical SIS coverage contains q128 inner/A `D = 512`, but rejects q128
      outer/opening `D = 512` and every q32/q64 role at `D = 512`.
- [x] The infinity-width generator's dimension union contains `512`, and its
      canonical role/profile filter emits q128 Inner/A D512 work only.
- [x] Every A-role coefficient bucket has a directly estimated D512 slice for
      ranks `1..=20`; no D256 row or rank conversion is used.
- [x] The expanded coverage has a distinct digest. The existing digest remains
      valid for existing cells but cannot authorize `D = 512`; the expanded
      digest authorizes the new q128 inner cell.
- [x] The generated q128 runtime table contains ordinary 20-entry D512 slices.
- [x] Existing generated schedule sources and catalog identities are
      byte-for-byte unchanged.
- [x] fp128 inner-role dispatch accepts exactly
      `D ∈ {64, 128, 256, 512}` and rejects `D = 32` and `D = 1024`.
- [x] fp128 outer/opening/envelope policies and every fp32/fp64 protocol policy
      are byte-for-byte unchanged.
- [x] Focused tests confirm the existing q128 CRT/NTT selector accepts
      `D = 512`; no NTT parameter change is needed.
- [x] Prover runtime source/backend capability bundles compile through D512;
      no prover algorithm or backend kernel changes.
- [x] No file under `crates/akita-config/`, `crates/akita-planner/`,
      `crates/akita-schedules/`, `crates/akita-verifier/`, or
      `crates/akita-pcs/` changes.
- [ ] Repository formatting, dependency, unit-test, Clippy, and
      documentation gates pass.

### Testing Strategy

The `D = 512` slice needs focused coverage at four boundaries:

1. **Canonical security coverage**
   - q128 Inner/512 is accepted for A-role coefficient buckets;
   - q128 Outer/Open/512 are rejected;
   - q32 and q64 Inner/512 are rejected.
2. **Direct SIS generation**
   - the new digest resolves q128 Inner/512 rows;
   - the old digest rejects 512;
   - both digests resolve their intended existing cells identically;
   - every D512 coefficient bucket has ranks `1..=20`;
   - focused generator tests prove D512 rows are emitted only from direct D512
     estimator requests.
3. **Arithmetic prerequisite**
   - `select_crt_ntt_params::<Prime128OffsetA7F7, 512>()` succeeds using the
     unchanged q128 profile;
   - prime-order tests continue to cover `Q128_MAX_RING_D = 512`.
4. **Protocol dispatch**
   - fp128 Inner accepts `64`, `128`, `256`, and `512`;
   - fp128 Inner rejects `32` and `1024`;
   - all other role/tier lists remain unchanged;
   - all generic prover users of inner-role dispatch compile with D512 source
     and backend capabilities.

Run the direct D512 offline generator with the explicit
`lattice-estimator` profile and its row validation, then run the repository
preflight, focused crate tests, all three CI Clippy graphs, and the CI Nextest
target set. No end-to-end `D = 512` proof is required until a later mixed
schedule selects the arm.

### Performance

There is no proof-time, proof-size, or memory claim because no configuration
selects `D = 512` in this spec. Existing benchmark modes and proof artifacts
must be unaffected.

Offline SIS generation gains 520 direct requests: 26 A-role coefficient
buckets times 20 ranks. The generated q128 table gains 26 D512 slices. Runtime
lookup has the same slice iteration and no per-rank special case.

## Design

### Architecture

The path to a usable future fp128 A-role `D = 512` cell is:

```text
canonical role/profile coverage
        │
        ├──> direct q128 D512 estimator requests
        │                  │
        │                  └──> generated D512 table slices
        │
        └──> runtime SIS role-cell validation
                                   │
fp128 inner dispatch ───────────────┴──> const-generic D=512 code
```

The challenge and NTT layers are already ready. Direct security-table
generation is added before the dispatch edge.

#### Profile-aware SIS coverage

`SisRoleCell` already contains `modulus_profile`, but `sis_role_cell` currently
chooses dimensions from global `A_ROLE_RING_DIMS` / `BD_ROLE_RING_DIMS` lists.
The implementation should make dimension eligibility a predicate of
`(role, modulus_profile, ring_dimension)`. Existing dimensions keep their
current behavior; only `(Inner, Q128OffsetA7F7, 512)` is added.

The estimator's `RING_DIMS` union includes `512`.
`generate_infinity_width_rows` consults the canonical role-cell predicate
before creating work, so this does not generate q32/q64 or B/D D512 cells.

#### Direct wide estimation

Each D512 request is constructed directly with:

```text
n = rank * 512
m = width * 512
length_bound = B
```

The compact LGSA path retains logarithmic observables for wide profiles so
probabilities remain finite without materializing vectors with billions of
coordinates. The production local-minimum profile matches the pinned
lattice-estimator beta/zeta search. Widths below the module rank have fewer
scalar columns than rows and are accepted by the generic prefix predicate
without invoking the estimator. Generic, unclassified `Infinity` remains an
error.

#### Versioned SIS digest

Changing `SisTableDigest::CURRENT` globally would force every checked-in
schedule to regenerate because the digest is embedded in schedule identity.
That would violate this spec's scope.

Instead, retain the existing digest for existing schedules and add a second
named digest for the expanded q128-inner-512 coverage. Runtime lookup
must:

- reject `D = 512` under the old digest;
- accept the old coverage under the old digest;
- accept the expanded coverage under the new digest; and
- return `None` for unknown digests.

The extension digest commits to the base digest, exact q128 modulus, D512
generation parameters, coefficient buckets, and all 520 generated widths
under the domain tag
`akita-sis-table-q128-inner-d512-direct-v1\0`. Its value is
`c2027a80d84b01dbbffae571cb9bf0e9686db6e762c5a4202d5e53a306e6cace`.

#### Dispatch

After certification, add `512` to:

1. the fp128 branch of `__dispatch_for_field_inner!`; and
2. the fp128 `inner` list in `protocol_dispatch_policy!`.

No other dispatch list changes. The q128 NTT list already includes `512`.

The const-generic arm is compiled at every generic call site even though no
current schedule selects it. Consequently the D-free prover capability bundles
and explicit source bounds used by those call sites must include `512`.
Existing sources and CPU/delegating backends implement their kernels for
arbitrary const `D`; the implementation only closes the runtime trait bundles
over the new arm. This is compile-time readiness, not mixed-schedule
integration.

### Alternatives Considered

- **Only add the dispatch literal.** Rejected. Unlike `D = 256`, `D = 512`
  previously had no direct audited A-role row, so dispatch alone would expose a
  const-generic arm that runtime security validation could not price.
- **Derive D512 rows from D256 rows.** Rejected. D512 is a first-class
  dimension and its complete rank range must come directly from the estimator.
- **Add `512` to global `A_ROLE_RING_DIMS`.** Rejected because it would also
  declare unused q32/q64 A-role cells. The requested capability is q128
  inner-only.
- **Replace the existing table digest and regenerate all schedules.** Rejected
  because the existing schedule behavior and identities do not depend on the
  new unreachable cell. Versioning isolates the security-table extension from
  schedule rollout.
- **Add a `D512` preset or generated schedule.** Rejected because a preset
  represents a selectable uniform configuration. The intended consumer is a
  later mixed-ring feature.
- **Modify challenge or NTT policy.** Rejected because both already cover
  fp128 `D = 512`.
- **Import the mixed-dimension demo.** Rejected because production mixed
  schedules and their prover/verifier changes are separate features.

## Documentation

Update this spec as each slice lands. Because SIS coverage is security-relevant,
also update `specs/sis-quantum128-scalar-n-table.md` with the profile-aware
coverage rule and versioned-digest rollout. Update the Akita Book security page
only if its stated supported SIS coverage becomes inaccurate.

No `AGENTS.md` or architecture change is expected. No schedule or configuration
documentation should advertise `D = 512`, because this spec does not make it
selectable.

## Execution

1. Make canonical SIS dimension coverage role- and modulus-profile-aware.
2. Add only q128 Inner/512 with maximum module rank `20`.
3. Add D512 to the estimator domain and generate every q128 Inner/A bucket and
   rank directly.
4. Emit ordinary D512 runtime slices, add the extension digest, and
   preserve the existing digest and schedule identities.
5. Add focused coverage, digest, direct-generation, and q128-NTT tests.
6. Add `512` to the two fp128 inner-dispatch policy sites, close the prover
   source/backend capability bundles over that arm, and update focused tests.
7. Verify that no config, planner, schedule, verifier, PCS, challenge,
   NTT-parameter, prover-algorithm, or backend-kernel file changed.
8. Run the full required repository checks.

## References

- Dispatch policy and tests:
  `crates/akita-types/src/dispatch/policy.rs`,
  `crates/akita-types/src/dispatch/mod.rs`
- Prover compile-time capability closure:
  `crates/akita-prover/src/compute/poly.rs`,
  `crates/akita-prover/src/api/commitment.rs`,
  `crates/akita-prover/src/protocol/core/fold.rs`,
  `crates/akita-prover/src/protocol/fold_grind.rs`,
  `crates/akita-prover/src/protocol/ring_relation.rs`
- Challenge ladder and sampler:
  `crates/akita-challenges/src/config.rs`,
  `crates/akita-challenges/src/sampler/position_sample.rs`
- Canonical SIS coverage and runtime lookup:
  `crates/akita-types/src/sis/ajtai_key.rs`
- Generated SIS table:
  `crates/akita-types/src/sis/generated_sis_table/`
- SIS generator and certification:
  `crates/akita-sis-estimator/src/width_table.rs`,
  `crates/akita-sis-estimator/examples/infinity_width_table.rs`
- q128 NTT capacity:
  `crates/akita-algebra/src/ntt/tables.rs`,
  `crates/akita-types/src/ntt_cache.rs`
- SIS security contract:
  `specs/sis-quantum128-scalar-n-table.md`
- Mixed-ring architectural context only:
  `specs/runtime-ring-cutover.md`
