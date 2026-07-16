# Spec: OpeningClaims API


| Field        | Value                                                                                                       |
| ------------ | ----------------------------------------------------------------------------------------------------------- |
| Author(s)    |                                                                                                             |
| Created      | 2026-06-29                                                                                                  |
| Status       | proposed                                                                                                    |
| PR           |                                                                                                             |
| Supersedes   | `specs/shared-opening-input-api.md`; partial overlap with `specs/single-point-opening-batch.md` (API layer) |
| Book-chapter | book/src/usage/commitment-api.md                                                                            |




## Summary

Replace the parallel prover/verifier opening-batch structs
(`ProverOpeningBatch`, `VerifierOpeningBatch`, `CommitmentGroup`,
`ProverCommitmentGroup`) and **`OpeningBatchShape` / `OpeningGroupShape`** with:

- **`OpeningClaims`** — real public prove/verify input (point, evaluations,
  commitments, routing).
- **`ProverOpeningData`** — `OpeningClaims` plus polynomials and hints.
- **`OpeningClaimsLayout`** — structure **without** field values, for setup,
  planner, config, and layout-only tests.

Prove/verify APIs take **claims only**. Setup and planner take **`OpeningClaimsLayout`**
—not fake claims with zero placeholders.

Input types use **private fields and accessor methods only**.

**Passing convention.** Callers and internal helpers pass the **whole** typed object
(`OpeningClaims`, `OpeningClaimsLayout`, or `ProverOpeningData`) rather than spreading it
into decomposed scalar/slice arguments (`point()` + `group_sizes()`, a bare
`num_total_polynomials()`, etc.). When a helper only needs counts/routing, it takes the
field-free `&OpeningClaimsLayout` object (obtained via `opening_claims.layout()`), not a
loose `usize`.

**No intermediate types.** This refactor introduces exactly five opening-claims types —
the four `akita-types` types above plus `ProverOpeningData`. No bridge, summary, view, or
derived-limits types are added; existing helper bags such as `OpeningBatchLimits` are
**removed** in favor of passing the already-validated `AkitaSetupSeed` envelope.

**Schedule type unification.** `PolynomialGroupLayout` is shared between opening layout
and schedule lookup (replacing `CommitmentGroupScheduleKey`). Frozen precommit metadata
becomes **`PrecommittedGroupParams`** (replacing schedule `CommitmentGroupLayout`).
Generated schedule tables use the same runtime types directly — the three
`GeneratedCommitmentGroup*` / `GeneratedScheduleLookupKey` structs are deleted.



## Intent



### Goal

Introduce a single, layered public input model for batched single-point openings
where verifier input is literally the public subset of prover input, and
**`OpeningBatchShape` is deleted** — replaced by claims (data) + layout (structure).

Primary types live in two crates:

**`akita-types`** — public claims and layout in one file
(`crates/akita-types/src/opening_claims.rs`):

| Type | Role |
|------|------|
| `OpeningClaims` | point, evaluations, commitments, routing |
| `PolynomialGroupClaims` | per-group public claims |
| `OpeningClaimsLayout` | batch structure without field values |
| `PolynomialGroupLayout` | per-group `(nuposition_bits, num_polynomials)` layout; also the schedule lookup group key |

Schedule types in `crates/akita-types/src/schedule.rs` are renamed and unified:

| Type | Role |
|------|------|
| `PolynomialGroupLayout` | per-group opening/schedule dimensions (shared with opening layout) |
| `PrecommittedGroupParams` | frozen standalone precommit metadata (`position_bits`, `block_bits`, basis, row counts) |
| `AkitaScheduleLookupKey` | multi-group-root schedule lookup key |

Wire as `pub mod opening_claims` from `lib.rs`; re-export the public types at the
crate root alongside existing `proof` items during migration.

**`akita-prover`** — prover prove input (`crates/akita-prover/src/types/`):

| File | Types |
|------|-------|
| `mod.rs` | module root; re-exports `ProverOpeningData` |
| `opening_data.rs` (or inline in `mod.rs`) | `ProverOpeningData` |

```rust
// crates/akita-types/src/opening_claims.rs
/// Public opening claims: one point and polynomial groups in transcript order.
pub struct OpeningClaims<'a, F, C> { /* private */ }

pub struct PolynomialGroupClaims<'a, F, C> { /* private */ }

/// Batch structure without point values, evaluations, or commitments.
/// Used by setup, planner, config — not by PCS prove/verify entry points.
pub struct OpeningClaimsLayout {
    groups: Vec<PolynomialGroupLayout>,
}

/// Per-group opening geometry: active point-variable count and polynomial count.
/// Also used as the schedule lookup group key (replaces `CommitmentGroupScheduleKey`).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PolynomialGroupLayout {
    nuposition_bits: usize,
    num_polynomials: usize,
}

// crates/akita-types/src/schedule.rs
/// Frozen metadata from a standalone precommit (params, not layout).
pub struct PrecommittedGroupParams {
    pub group: PolynomialGroupLayout,
    pub position_bits: usize,
    pub block_bits: usize,
    pub log_basis: u32,
    pub n_a: usize,
    pub conservative_n_b: usize,
}

pub struct AkitaScheduleLookupKey {
    pub final_group: PolynomialGroupLayout,
    pub precommitteds: Vec<PrecommittedGroupParams>,
}

// crates/akita-prover/src/types/opening_data.rs
pub struct ProverOpeningData<'a, PointF, P, CommitF, const D: usize> { /* private */ }
```

**All fields are private** on opening-claims types. Callers construct through validated
constructors and read through accessor methods only.

**Removed types (no aliases, no deprecation wrappers):**


| Removed | Replaced by |
| ----------------------- | ----------------------------------------- |
| `VerifierOpeningBatch` | `OpeningClaims` |
| `CommitmentGroup` | `PolynomialGroupClaims` |
| `ProverOpeningBatch` | `ProverOpeningData` |
| `ProverCommitmentGroup` | accessors on `ProverOpeningData` |
| `OpeningBatchShape` | **`OpeningClaimsLayout`** (layout) + claims accessors (data) |
| `OpeningGroupShape` | **`PolynomialGroupLayout`** |
| `OpeningBatchLimits` | removed — `OpeningClaims::validate(&AkitaSetupSeed)` reads `max_nuposition_bits` / `max_num_batched_polys` directly |
| `CommitmentGroupScheduleKey` | **`PolynomialGroupLayout`** |
| schedule `CommitmentGroupLayout` | **`PrecommittedGroupParams`** |
| `GeneratedCommitmentGroupScheduleKey` | **`PolynomialGroupLayout`** (in static tables) |
| `GeneratedCommitmentGroupLayout` | **`PrecommittedGroupParams`** (in static tables) |
| `GeneratedScheduleLookupKey` | fields inlined into **`GeneratedGroupBatchScheduleTableEntry`** |




### Design decisions (detailed)



#### 1. Private fields, method-only access

**Decision:** `OpeningClaims`, `PolynomialGroupClaims`, and
`ProverOpeningData` expose **no public fields**. All construction goes through
named constructors; all reads go through accessor methods. The layout types
(`OpeningClaimsLayout`, `PolynomialGroupLayout`) follow the same rule: private
fields, validated constructors (`OpeningClaimsLayout::new` / `from_group_sizes` /
`from_groups` / `from_setup_seed`, `PolynomialGroupLayout::new`), accessor reads,
and **no struct literals at call sites** (except `const` generated table rows).

**Rules:**


| Rule | Detail |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| No `pub` fields | On all five opening-claims types |
| Construct via constructors | `PolynomialGroupClaims::new(...)`, `OpeningClaims::from_groups(...)`, `ProverOpeningData::new(...)` |
| Read via methods | `OpeningClaims`: `point()`, `group_evaluations(g)`, …; `ProverOpeningData`: `opening_claims()`, `hints()`, `group_polys(g)`, … |
| No struct literals at call sites | Tests use constructors; generated tables use `const` constructors on `Copy` layout types |




#### 2. `OpeningClaims` — shared public batch

**Decision:** The public batch type is `OpeningClaims`. No verifier alias.

**Why:**

- Both prover and verifier bind the **same** public claims (point, evaluations,
commitments, routing). The name reflects that prover and verifier bind the same claim set.
- Prover wraps it in `ProverOpeningData`; verifier takes `OpeningClaims`
at the PCS boundary.
- Call sites: `batched_verify(..., opening_claims: OpeningClaims)`,
`prover.opening_claims()`.



#### 3. `ProverOpeningData` wraps `OpeningClaims`

**Decision:** Prover input owns private `opening_claims: OpeningClaims<...>` plus
parallel `hints` and `polynomials`. Access via `opening_claims()`, `hints()`,
`group_polys(g)`.

**Alignment invariant:**

```text
opening_claims.num_groups() == hints.len() == polynomials.len()
∀ g: group_polys(g).len() == group_evaluations(g).len()
```



#### 4. Hints on `ProverOpeningData` only

Hints stay out of `OpeningClaims` — prover-only, one per commitment group,
accessed via `hints()` / `group_hint(g)`.

#### 5. `PolynomialGroupClaims` per group

One commitment, many evaluations, plus `point_vars` for transcript routing.
Per-group count: `num_evaluations()` only (no batch-level duplicate names).

Layout does **not** carry `PointVariableSelection`; see decision #12.

#### 6. One batch polynomial count: `num_total_polynomials()`

**Decision:** `OpeningClaims` exposes a **single** batch-wide polynomial
count method. Do **not** expose separate `num_claims()` or `num_polynomials()`.

```rust
impl OpeningClaims<'_, F, C> {
    /// Total polynomials opened across all polynomial groups (sum of group sizes).
    pub fn num_total_polynomials(&self) -> usize;
}
```

**Why one name:**

- In the current opening-batch model, “claims”, “openings”, and “polynomials”
are the same count — one claimed evaluation per committed polynomial per group.
- `num_claims()` vs `num_polynomials()` duplicated the same integer under two
protocol synonyms and invited callers to wonder which to use.
- `num_total_polynomials()` is explicit about what is being counted and matches
prover-side `flat_polys().len()` / verifier-side `flat_evaluations().len()`.

**Mapping from old APIs:**


| Old | New |
| -------------------------------------- | -------------------------------- |
| `OpeningBatchShape::num_claims()` | `opening_claims.num_total_polynomials()` |
| `OpeningBatchShape::num_polynomials()` | `opening_claims.num_total_polynomials()` |
| `VerifierOpeningBatch::num_claims()` | `opening_claims.num_total_polynomials()` |


Per-group sizes remain available via `group_sizes()` (evaluations per polynomial
group). `PolynomialGroupClaims::num_evaluations()` remains for one group.

#### 7. Two types: claims (data) vs layout (structure)

**Decision:** Split what `OpeningBatchShape` conflated today:

| Type | Has | Used by |
|------|-----|---------|
| **`OpeningClaims`** | point, evaluations, commitments, `point_vars` | `batched_prove`, `batched_verify`, transcript absorption of **values**, fold/root replay |
| **`OpeningClaimsLayout`** | per-group `nuposition_bits`, `num_polynomials` only | setup sizing, schedule lookup, planner, commit-param preview, layout tests |

**Prove/verify are not changed to accommodate setup.** PCS entry points keep
`OpeningClaims` / `ProverOpeningData`. Setup and planner APIs take
`&OpeningClaimsLayout` (or `AkitaScheduleLookupKey` where schedule alone suffices).

**Bridge from real input:**

```rust
impl OpeningClaims<'_, F, C> {
    /// Structural view used by config/planner (no field values).
    pub fn layout(&self) -> OpeningClaimsLayout;
}
```

Prove orchestration:

```rust
prover.validate::<Cfg::Field>()?;
let opening_claims = prover.opening_claims();
let schedule = Cfg::get_params_for_prove(&opening_claims.layout())?;
```

**Do not** use `OpeningClaims::fixture()` as the planner API. Layout-only
call sites construct `OpeningClaimsLayout` directly.

#### 8. `OpeningClaimsLayout` — replacing `OpeningBatchShape`

**Decision:** Rename and narrow the old shape type to **`OpeningClaimsLayout`**:
counts only, no pretense of being prove input. There is **no batch-level
`nuposition_bits` field** — each group's active variable count lives on
`PolynomialGroupLayout`, and the batch exposes `max_nuposition_bits()` when a single
scalar is needed.

```rust
impl OpeningClaimsLayout {
    /// Single full-point group (production default).
    pub fn new(nuposition_bits: usize, num_total_polynomials: usize) -> Result<Self, AkitaError>;

    /// Multi-group structure (planner / multi-group specs).
    /// Every group receives the same `nuposition_bits` (shared padded point).
    pub fn from_group_sizes(
        nuposition_bits: usize,
        polynomials_per_group: &[usize],
    ) -> Result<Self, AkitaError>;

    /// Custom per-group geometry (multi-group specs, unequal per-group counts).
    pub fn from_groups(groups: Vec<PolynomialGroupLayout>) -> Result<Self, AkitaError>;

    /// Worst-case envelope from prover setup seed (one full-point group).
    /// Fallible like the other constructors so it never panics on a malformed seed
    /// (verifier no-panic contract); a valid `AkitaSetupSeed` always succeeds.
    pub fn from_setup_seed(seed: &AkitaSetupSeed) -> Result<Self, AkitaError>;

    /// Maximum active point-variable count across groups.
    pub fn max_nuposition_bits(&self) -> usize {
        self.groups()
            .iter()
            .map(PolynomialGroupLayout::nuposition_bits)
            .max()
            .unwrap_or(0)
    }

    pub fn groups(&self) -> &[PolynomialGroupLayout] {
        &self.groups
    }

    pub fn num_groups(&self) -> usize;
    pub fn num_total_polynomials(&self) -> usize;
    pub fn group_sizes(&self) -> Vec<usize>;
    pub fn group_layout(&self, g: usize) -> Result<&PolynomialGroupLayout, AkitaError>;

    /// Routing digest (no point/commitment values) — replaces digest_opening_batch.
    pub fn opening_batch_digest(&self) -> DescriptorDigest;
}

impl PolynomialGroupLayout {
    /// Usable in `const` generated schedule tables.
    pub const fn new(nuposition_bits: usize, num_polynomials: usize) -> Self;

    /// Runtime validation (non-const callers).
    pub fn validate(self) -> Result<(), AkitaError>;

    pub const fn nuposition_bits(self) -> usize;
    pub const fn num_polynomials(self) -> usize;

    /// Scalar default: one polynomial at `nuposition_bits`.
    pub const fn singleton(nuposition_bits: usize) -> Self;
}
```

Default per-group geometry: `PolynomialGroupLayout::new(nuposition_bits, k)`.
Custom per-group counts: `OpeningClaimsLayout::from_groups(vec![PolynomialGroupLayout::new(...), ...])`.

**Replacement map (layout consumers):**

| Current (`OpeningBatchShape`) | After |
|-------------------------------|-------|
| `OpeningBatchShape::new(nv, k)` | `OpeningClaimsLayout::new(nv, k)?` |
| `from_commitment_groups(nv, sizes)` | `OpeningClaimsLayout::from_group_sizes(nv, sizes)?` |
| `shape.nuposition_bits()` | `layout.max_nuposition_bits()` |
| `Cfg::get_params_for_prove(&shape)` | `Cfg::get_params_for_prove(&layout)` |
| `Cfg::get_params_for_batched_commitment(&shape)` | `Cfg::get_params_for_batched_commitment(&layout)` |
| `CommitmentGroupScheduleKey::new_from_opening_batch(&shape)` | `AkitaScheduleLookupKey::from_layout(&layout)?` |
| `CallSection::from_opening_batch(&shape, basis)` | `CallSection::from_layout(&layout, basis)` |
| `digest_opening_batch(&shape)` | `layout.opening_batch_digest()` |
| `active_setup_field_len(lp, &shape, D)` | `active_setup_field_len(lp, &layout, D)` |
| `akita-setup` recursion `OpeningBatchShape::new(max_vars, max_polys)` | `OpeningClaimsLayout::from_setup_seed(seed)?` |

**Replacement map (claims consumers — prove/verify only):**

| Concern | API |
|---------|-----|
| Public point / evaluations / commitments | `OpeningClaims` accessors |
| Batch poly count | `opening_claims.num_total_polynomials()` (accessor; do not spread into call args) |
| Transcript (shape header + values) | `opening_claims.append_to_transcript(transcript)` (pass the claims object) |
| Fiat–Shamir gamma rows | `sample_public_row_coefficients(&layout, t)` where `let layout = opening_claims.layout()` |
| Batched eval sum | `batched_eval_target_from_layout(&layout, &row_coefficients, &openings)` |

#### 9. Schedule lookup key unification

**Decision:** Delete `CommitmentGroupScheduleKey`. Use `PolynomialGroupLayout`
everywhere a per-group schedule key was needed. Rename schedule
`CommitmentGroupLayout` → **`PrecommittedGroupParams`** (params frozen at
precommit time, not opening layout).

```rust
// crates/akita-types/src/schedule.rs

pub struct AkitaScheduleLookupKey {
    pub final_group: PolynomialGroupLayout,
    pub precommitteds: Vec<PrecommittedGroupParams>,
}

impl AkitaScheduleLookupKey {
    /// Scalar same-point opening: one final group, no precommitteds.
    pub fn single(final_group: PolynomialGroupLayout) -> Self {
        Self {
            final_group,
            precommitteds: Vec::new(),
        }
    }

    /// Scalar schedule lookup from a single-group opening layout.
    ///
    /// # Errors
    ///
    /// Returns an error if `layout` has more than one group (multi-group roots use
    /// an explicitly constructed key with `precommitteds` populated).
    pub fn from_layout(layout: &OpeningClaimsLayout) -> Result<Self, AkitaError> {
        if layout.num_groups() != 1 {
            return Err(AkitaError::InvalidSetup(
                "scalar schedule lookup cannot collapse a multi-group layout; \
                 build an explicit multi-group key per specs/multi-group-batching.md"
                    .to_string(),
            ));
        }
        Ok(Self::single(layout.groups()[0]))
    }

    pub fn num_commitment_groups(&self) -> usize;
    pub fn num_polynomials(&self) -> Result<usize, AkitaError>;

    pub fn validate(&self) -> Result<(), AkitaError>;
}

impl PrecommittedGroupParams {
    pub fn from_params(group: PolynomialGroupLayout, params: &LevelParams) -> Self;
    pub fn validate(&self) -> Result<(), AkitaError>;
    pub fn validate_root_geometry(&self, ring_dimension: usize) -> Result<(), AkitaError>;
    pub fn validate_frozen_precommit(
        &self,
        ring_dimension: usize,
        min_log_basis: u32,
    ) -> Result<(), AkitaError>;
}
```

**Grouped-root invariant.** Validation is per group and no longer relates
precommitted group variable counts to the final group variable count:

```text
∀ pre in precommitteds:
    pre.group.validate()
    pre.validate()
```

The setup envelope and schedule lookup decide whether a concrete precommitted
group is supported by the selected preset; above-half and larger-than-final
precommitted groups are valid schedule keys.

**Scalar openings** remain the empty-precommitted case:

```rust
AkitaScheduleLookupKey {
    final_group: PolynomialGroupLayout::new(nuposition_bits, num_polynomials)?,
    precommitteds: vec![],
}
```

**Test migration:** `group_batch_key_allows_precommitted_nuposition_bits_above_half_main`
remains an acceptance test, and coverage includes precommitted groups larger
than the final group.

#### 10. Generated schedule tables — shared runtime types

**Decision:** Delete the three generated mirror structs. Static catalog rows use
`PolynomialGroupLayout` and `PrecommittedGroupParams` directly.

```rust
// crates/akita-planner/src/generated/mod.rs (representative)

pub struct GeneratedScheduleTableEntry {
    pub key: PolynomialGroupLayout,
    pub steps: &'static [GeneratedStep],
}

pub struct GeneratedGroupBatchScheduleTableEntry {
    pub final_group: PolynomialGroupLayout,
    pub precommitteds: &'static [PrecommittedGroupParams],
    pub steps: &'static [GeneratedStep],
}
```

**Requirements on shared types for static tables:**

| Trait / API | Why |
|-------------|-----|
| `Copy + Eq` | Rows are plain data in `static` tables |
| `const fn new(nuposition_bits, num_polynomials)` on `PolynomialGroupLayout` | Emitted by offline table generation |
| `Copy` on `PrecommittedGroupParams` | Precommitted rows in multi-group batch tables |

**Deleted:**

- `GeneratedCommitmentGroupScheduleKey`
- `GeneratedCommitmentGroupLayout`
- `GeneratedScheduleLookupKey`

`GeneratedGroupBatchScheduleTableEntry` no longer wraps a nested lookup-key struct;
`final_group` and `precommitteds` are top-level fields alongside `steps`.

Runtime conversion from a generated grouped row builds `AkitaScheduleLookupKey`
by copying `final_group` and mapping each static `PrecommittedGroupParams` into
the runtime `Vec` (same field shapes — no parallel generated type).

#### 11. Where `OpeningBatchShape` is used today — migration categories

Inventory from the current codebase (~60 references):

**A. Prove/verify hot path** → `OpeningClaims` / `ProverOpeningData`

| Location | Today | Next |
|----------|-------|------|
| `batched_prove` / `batched_verify` | `ProverOpeningBatch` / `VerifierOpeningBatch` | `ProverOpeningData` / `OpeningClaims` |
| `fold.rs`, `root_fold.rs`, `verify.rs` | `&OpeningBatchShape` from `to_shape()` | bind `let layout = opening_claims.layout()` once at the boundary and pass `&layout`; pass `&OpeningClaims` where values are needed |
| `RingRelationInstance` | stores `OpeningBatchShape` | stores an `OpeningClaimsLayout` snapshot (keep the whole object — do not flatten into scalars) |

**B. Config / schedule / planner** → `OpeningClaimsLayout` / `PolynomialGroupLayout`

| Location | Today | Next |
|----------|-------|------|
| `CommitmentConfig::get_params_for_prove` | `&OpeningBatchShape` | `&OpeningClaimsLayout` |
| `proof_optimized.rs` planner | `&OpeningBatchShape` | `&OpeningClaimsLayout` |
| `generated_families.rs` catalog build | `OpeningBatchShape::new(nv, polys)` | `OpeningClaimsLayout::new(nv, polys)?` |
| `CommitmentGroupScheduleKey::new_from_opening_batch` | `&OpeningBatchShape` | `AkitaScheduleLookupKey::from_layout(&layout)?` |
| `AkitaScheduleLookupKey::single(key)` | `CommitmentGroupScheduleKey` | `PolynomialGroupLayout` |

**C. Setup / setup-prefix** → `OpeningClaimsLayout` (built from the `AkitaSetupSeed` envelope)

| Location | Today | Next |
|----------|-------|------|
| `akita-setup/src/lib.rs` | `OpeningBatchShape::new(MAX_VARS, 1)` | `OpeningClaimsLayout::from_setup_seed(seed)?` (or `new(...)`) |
| `akita-setup/src/recursion.rs` | root + suffix shapes from max vars / `(0, 1)` | `OpeningClaimsLayout::from_setup_seed(seed)?` + `OpeningClaimsLayout::new(0, 1)?` |
| `setup_prefix.rs` `active_setup_field_len` | `&OpeningBatchShape` for `num_claims` | `&OpeningClaimsLayout` for `num_total_polynomials` |

**D. Commit-before-prove (polynomials known, no openings yet)** → derive layout from polys

| Location | Today | Next |
|----------|-------|------|
| `prepare_batched_commit_inputs` | returns `OpeningBatchShape` | returns `OpeningClaimsLayout` from poly count + padded `max_nuposition_bits` |
| `batched_commit` param sizing | shape from polys | `OpeningClaimsLayout` from polys |

**E. Tests / benches / examples** — split by intent

| Intent | Today | Next |
|--------|-------|------|
| Schedule / commit layout only | `OpeningBatchShape::new(nv, k)` (~35 sites) | `OpeningClaimsLayout::new(nv, k)?` |
| Full prove/verify e2e | builds batch structs | `OpeningClaims` / `ProverOpeningData` |
| Descriptor digest routing tests | `OpeningBatchShape::from_groups` with custom `point_vars` | claims-side `PolynomialGroupClaims` with custom `point_vars`; layout uses derived `nuposition_bits` |
| Catalog / planner unit tests | shape / `CommitmentGroupScheduleKey` | layout / `PolynomialGroupLayout` |
| Grouped schedule validation | allows precommitted above half main | accepts arbitrary per-group `nuposition_bits` supported by setup/schedule |

**F. Instance descriptor** → layout for routing digest, claims for live prove

`CallSection` fields are counts + routing digest — layout-derived. Live prove
binds descriptor via `opening_claims.layout()` at entry.

#### 12. What we explicitly do **not** do

- **No `OpeningClaims::fixture()`** for planner/setup (removed from primary design). Optional test-only helper may build claims from a layout + dummy values, but planner code must not depend on it.
- **No `Cfg::get_params_for_prove(&OpeningClaims)`** — forces layout extraction and keeps setup independent of claim payloads.
- **No stuffing setup with fake zero evaluations** just to call prove-shaped APIs.
- **No backward compatibility** — old names and generated mirror structs are deleted outright.
- **No new wrapper structs** — `PolynomialGroupLayout` is the single per-group key/layout type; do not reintroduce a scalar-only schedule key sibling.

`AkitaSetupSeed` (`max_nuposition_bits`, `max_num_batched_polys`) remains the stored
**capacity envelope**; `OpeningClaimsLayout::from_setup_seed` is the typed view
for schedule/setup code that today synthesizes `OpeningBatchShape::new(seed.max_nuposition_bits, seed.max_num_batched_polys)`.

#### 13. Pass whole typed objects into internal protocol functions

Public APIs take `OpeningClaims` / `ProverOpeningData`. Config/planner take
`OpeningClaimsLayout`. Internal helpers receive the **whole** object, not a spread of
its fields:

- `&OpeningClaims` when the helper needs public **values** (point, evaluations,
  commitments) — e.g. `validate_batched_inputs(&expanded, &opening_claims, true)`,
  `opening_claims.append_to_transcript(transcript)`.
- `&OpeningClaimsLayout` when the helper needs only **counts** — e.g.
  `sample_public_row_coefficients(&layout, t)`,
  `batched_eval_target_from_layout(&layout, …)`,
  `Cfg::get_params_for_prove(&layout)`.

`OpeningClaimsLayout` is itself one of the five first-class objects, so passing
`&layout` is "pass the object," not field decomposition; it is the field-free view that
keeps config/setup independent of claim payloads (decision #12). Bind it once and reuse:

```rust
let opening_claims = prover.opening_claims();
let layout = opening_claims.layout();
let schedule = Cfg::get_params_for_prove(&layout)?;
```

Do **not** spread an object into scalar/slice arguments such as
`f(opening_claims.point(), &opening_claims.group_sizes())` or
`f(opening_claims.num_total_polynomials())`; pass `&opening_claims` or `&layout`
instead. The old monolithic shape type is gone either way.

#### 14. `point_vars` on claims only; layout uses `nuposition_bits`

**Decision:** `PolynomialGroupLayout` stores **`nuposition_bits: usize`**, not
`PointVariableSelection`. Layout describes how many shared point coordinates each
group actively uses (prefix routing in the common case).

`PolynomialGroupClaims` **still carries `point_vars`** for transcript absorption and
descriptor-digest routing tests that need non-default selections. `opening_claims.layout()`
derives each group's `nuposition_bits` from its `point_vars` (active coordinate count).

There is **no** `group_point_vars` on `OpeningClaimsLayout`.

### Invariants

| Invariant | Detail |
| --------------------------- | ----------------------------------------------------------------------------------- |
| Encapsulation | No public fields on any of the five opening-claims types; constructor + accessor API only |
| Single batch poly count | Only `num_total_polynomials()` at batch level — no `num_claims` / `num_polynomials` |
| Batch max vars | `layout.max_nuposition_bits()` is the batch-wide padded point length; equals `opening_claims.nuposition_bits()` for valid claims |
| Group alignment | `opening_claims.num_groups()`; each group nonempty |
| Prover alignment | `group_polys(g).len() == group_evaluations(g).len()`; hints 1:1 with groups |
| No `OpeningBatchShape` | Replaced by `OpeningClaimsLayout` + claims types |
| No intermediate types | Exactly five opening-claims types; no bridge/summary/limits types (`OpeningBatchLimits` deleted) |
| Whole-object passing | Helpers take `&OpeningClaims` / `&OpeningClaimsLayout` / `&ProverOpeningData`, not spread fields |
| Layout/claims consistency | `opening_claims.layout()` matches group sizes and per-group `nuposition_bits` derived from claims routing |
| Prover/verifier consistency | Same `OpeningClaims` transcript binding |
| No verifier panic | Constructors and validation return `AkitaError`; no panic on malformed input |
| Grouped-root sizing | Precommitted `nuposition_bits` are independent of final-group `nuposition_bits`; setup/schedule capacity decides support |
| Shared schedule key | `PolynomialGroupLayout` is the only per-group schedule lookup key type |
| Generated table types | Static rows use `PolynomialGroupLayout` / `PrecommittedGroupParams` directly — no generated mirror structs |




### Non-Goals

- Multi-group folded root proving (still rejected; `MULTI_GROUP_ROOT_*` unchanged).
- Multipoint batches (removed; unchanged).
- Changing proof wire format or transcript labels.
- Separate batch-level `num_claims()` alias for `num_total_polynomials()`.
- Backward-compatible aliases for deleted schedule or generated types.



## Evaluation



### Acceptance Criteria

- [ ] `OpeningClaims`, `PolynomialGroupClaims`, `OpeningClaimsLayout`, and `PolynomialGroupLayout` in `akita-types/src/opening_claims.rs`, all with private fields + constructor/accessor APIs (except `Copy` schedule-visible fields on `PolynomialGroupLayout`).
- [ ] `OpeningClaims` exposes `check()` and `validate(&AkitaSetupSeed)` (returns `()`); structural views come from `layout()`.
- [ ] `ProverOpeningData` in `akita-prover/src/types/` with private fields, `new(...)`, accessors, and `validate::<PolyF>()` (alignment + poly-shape; no limits arg).
- [ ] Exactly five opening-claims types ship; `OpeningBatchLimits` and all other intermediate/bridge types are removed.
- [ ] Batch-level count API is `num_total_polynomials()` **only** — no `num_claims()` or `num_polynomials()` on `OpeningClaims`.
- [ ] `OpeningClaimsLayout` has no batch-level `nuposition_bits`; exposes `max_nuposition_bits()` and `groups()`.
- [ ] `PolynomialGroupLayout` replaces `CommitmentGroupScheduleKey`; `PrecommittedGroupParams` replaces schedule `CommitmentGroupLayout`.
- [ ] `AkitaScheduleLookupKey` uses `PolynomialGroupLayout` / `PrecommittedGroupParams`; grouped validation accepts independent precommitted group sizes.
- [ ] Generated tables use shared types; `GeneratedCommitmentGroupScheduleKey`, `GeneratedCommitmentGroupLayout`, and `GeneratedScheduleLookupKey` are deleted.
- [ ] `OpeningClaimsLayout` / `PolynomialGroupLayout` replace `OpeningBatchShape` / `OpeningGroupShape`.
- [ ] Setup/planner/config use `OpeningClaimsLayout` — not `OpeningClaims::fixture`.
- [ ] PCS traits use `OpeningClaims` / `ProverOpeningData`.
- [ ] All PCS e2e, transcript-hardening, recursion tests pass.



### Testing Strategy

- Port unit tests to `OpeningClaims`; assert `num_total_polynomials()` matches group sum.
- Update schedule tests: precommitted `nuposition_bits` above half of `final_group.nuposition_bits` and larger than the final group are accepted when setup/schedule capacity supports them.
- Grep gate: no `OpeningBatchShape`, `OpeningGroupShape`, `OpeningBatchLimits`, `CommitmentGroupScheduleKey`, schedule `CommitmentGroupLayout`, or generated mirror struct names.
- Run full workspace `cargo test`.



## Design



### Architecture

```text
   ProverOpeningData
   ├─ opening_claims() ─► OpeningClaims  (point, evaluations, commitments, point_vars)
   ├─ hints() / group_hint(g)
   └─ group_polys(g) / flat_polys()
                          │
                          │  opening_claims.layout()   (structure only; no field values)
                          ▼
                 OpeningClaimsLayout
                 └─ groups: [PolynomialGroupLayout { nuposition_bits, num_polynomials }, ...]
                          │
        ┌─────────────────┼───────────────────────────┐
        ▼                 ▼                            ▼
  get_params_for_prove   active_setup_field_len    CallSection / opening_batch_digest
  (config / planner)     (setup-prefix sizing)     (instance descriptor)
                          │
                          ▼
              AkitaScheduleLookupKey
              ├─ final_group: PolynomialGroupLayout
              └─ precommitteds: [PrecommittedGroupParams { group, position_bits, ... }]
                          │
                          ▼
              GeneratedScheduleTableEntry / GeneratedGroupBatchScheduleTableEntry
              (same PolynomialGroupLayout / PrecommittedGroupParams types)
```

The verifier holds an `OpeningClaims` directly (no `ProverOpeningData` wrapper); it
takes the same `opening_claims.layout()` path to the structural consumers below.



### Public API surface



#### `OpeningClaims` / `OpeningClaimsLayout` (`akita-types/src/opening_claims.rs`)

```rust
impl<'a, F, C> PolynomialGroupClaims<'a, F, C> {
    /// One group: ordered point-variable selection, dense evaluations
    /// (one per committed polynomial), and the group commitment. The selection
    /// is validated against the shared point inside `OpeningClaims::from_groups`.
    pub fn new(
        point_vars: PointVariableSelection,
        evaluations: Vec<F>,
        commitment: C,
    ) -> Result<Self, AkitaError>;

    pub fn point_vars(&self) -> &PointVariableSelection;
    pub fn evaluations(&self) -> &[F];
    pub fn commitment(&self) -> &C;
    pub fn num_evaluations(&self) -> usize;
}

impl<'a, F, C> OpeningClaims<'a, F, C> {
    pub fn from_groups(
        point: impl Into<OpeningPoints<'a, F>>,
        groups: Vec<PolynomialGroupClaims<'a, F, C>>,
    ) -> Result<Self, AkitaError>;

    pub fn check(&self) -> Result<(), AkitaError>;

    /// Validate internal consistency plus public capacity against the setup
    /// envelope (`seed.max_nuposition_bits`, `seed.max_num_batched_polys`). Returns
    /// `()`; callers obtain the structural view via `layout()`.
    pub fn validate(&self, seed: &AkitaSetupSeed) -> Result<(), AkitaError>;

    pub fn point(&self) -> &[F];
    pub fn nuposition_bits(&self) -> usize;
    pub fn num_groups(&self) -> usize;
    pub fn num_total_polynomials(&self) -> usize;
    pub fn group_sizes(&self) -> Vec<usize>;

    pub fn group_evaluations(&self, g: usize) -> Result<&[F], AkitaError>;
    pub fn group_point_vars(&self, g: usize) -> Result<&PointVariableSelection, AkitaError>;
    pub fn group_commitment(&self, g: usize) -> Result<&C, AkitaError>;
    pub fn flat_evaluations(&self) -> Vec<F>;

    /// Structural view for config/planner/setup (no field values).
    pub fn layout(&self) -> OpeningClaimsLayout;

    pub fn opening_batch_digest(&self) -> DescriptorDigest; // == self.layout().opening_batch_digest()
    pub fn append_to_transcript<TranscriptF, T>(&self, transcript: &mut T) -> Result<(), AkitaError>;
}

/// Commitment-less, full-point claims used only by the internal extension-opening
/// reduction (EOR) replay. Pads the shared point with zeroes; not a setup/planner API.
impl<'a, F: FieldCore> OpeningClaims<'a, F, ()> {
    pub fn with_padded_point(
        point: &[F],
        nuposition_bits: usize,
        num_total_polynomials: usize,
    ) -> Result<Self, AkitaError>;
}
```

#### `OpeningClaimsLayout` / `PolynomialGroupLayout`

```rust
impl OpeningClaimsLayout {
    pub fn new(nuposition_bits: usize, num_total_polynomials: usize) -> Result<Self, AkitaError>;
    pub fn from_group_sizes(nuposition_bits: usize, polynomials_per_group: &[usize]) -> Result<Self, AkitaError>;
    pub fn from_groups(groups: Vec<PolynomialGroupLayout>) -> Result<Self, AkitaError>;
    pub fn from_setup_seed(seed: &AkitaSetupSeed) -> Result<Self, AkitaError>;

    pub fn max_nuposition_bits(&self) -> usize;
    pub fn groups(&self) -> &[PolynomialGroupLayout];
    pub fn num_groups(&self) -> usize;
    pub fn num_total_polynomials(&self) -> usize;
    pub fn group_sizes(&self) -> Vec<usize>;
    pub fn group_layout(&self, g: usize) -> Result<&PolynomialGroupLayout, AkitaError>;

    pub fn opening_batch_digest(&self) -> DescriptorDigest;
}

impl PolynomialGroupLayout {
    pub const fn new(nuposition_bits: usize, num_polynomials: usize) -> Self;
    pub fn validate(self) -> Result<(), AkitaError>;
    pub const fn nuposition_bits(self) -> usize;
    pub const fn num_polynomials(self) -> usize;
    pub const fn singleton(nuposition_bits: usize) -> Self;
}
```

#### Schedule types (`akita-types/src/schedule.rs`)

```rust
impl AkitaScheduleLookupKey {
    pub fn single(final_group: PolynomialGroupLayout) -> Self;
    pub fn from_layout(layout: &OpeningClaimsLayout) -> Result<Self, AkitaError>;
    pub fn num_commitment_groups(&self) -> usize;
    pub fn num_polynomials(&self) -> Result<usize, AkitaError>;
    pub fn validate(&self) -> Result<(), AkitaError>;
}

impl PrecommittedGroupParams {
    pub fn from_params(group: PolynomialGroupLayout, params: &LevelParams) -> Self;
    pub fn validate(&self) -> Result<(), AkitaError>;
    pub fn validate_root_geometry(&self, ring_dimension: usize) -> Result<(), AkitaError>;
    pub fn validate_frozen_precommit(
        &self,
        ring_dimension: usize,
        min_log_basis: u32,
    ) -> Result<(), AkitaError>;
}
```

#### Generated schedule tables (`akita-planner/src/generated/mod.rs`)

```rust
pub struct GeneratedScheduleTableEntry {
    pub key: PolynomialGroupLayout,
    pub steps: &'static [GeneratedStep],
}

pub struct GeneratedGroupBatchScheduleTableEntry {
    pub final_group: PolynomialGroupLayout,
    pub precommitteds: &'static [PrecommittedGroupParams],
    pub steps: &'static [GeneratedStep],
}
```



#### `ProverOpeningData` (`akita-prover/src/types/`)

```rust
impl<'a, PointF, P, CommitF, const D: usize> ProverOpeningData<'a, PointF, P, CommitF, D> {
    /// Sole constructor: bundle public claims (which already own commitments) with
    /// the parallel prover-only `hints` and `polynomials`, one per polynomial group.
    /// There is intentionally no `from_groups`/per-group input type — grouping lives
    /// in the `OpeningClaims` argument (decision: no intermediate types).
    pub fn new(
        opening_claims: OpeningClaims<'a, PointF, RingCommitment<CommitF, D>>,
        hints: Vec<AkitaCommitmentHint<CommitF, D>>,
        polynomials: Vec<&'a [&'a P]>,
    ) -> Result<Self, AkitaError>;

    pub fn validate<PolyF>(&self) -> Result<(), AkitaError>
    where
        PolyF: FieldCore,
        P: RootPolyShape<PolyF, D> + RootOpeningSource<PolyF, D>;

    pub fn nuposition_bits<PolyF>(&self) -> Result<usize, AkitaError>
    where P: RootPolyShape<PolyF, D>;

    pub fn opening_claims(&self) -> &OpeningClaims<'a, PointF, RingCommitment<CommitF, D>>;
    pub fn hints(&self) -> &[AkitaCommitmentHint<CommitF, D>];
    pub fn group_hint(&self, index: usize) -> Result<&AkitaCommitmentHint<CommitF, D>, AkitaError>;
    pub fn group_polys(&self, index: usize) -> Result<&'a [&'a P], AkitaError>;
    pub fn flat_polys(&self) -> Vec<&'a P>;
    pub fn commitments(&self) -> Vec<&RingCommitment<CommitF, D>>;

    pub fn append_to_transcript<T>(&self, transcript: &mut T) -> Result<(), AkitaError>;
}
```



#### PCS traits

```rust
fn batched_verify<T: Transcript<F>>(
    ...
    opening_claims: OpeningClaims<'_, Self::ExtField, &Self::Commitment>,
    ...
) -> Result<(), AkitaError>;

fn batched_prove<'a, T, P, B>(
    ...
    prover: ProverOpeningData<'a, Self::ExtField, P, F, D>,
    ...
) -> Result<Self::BatchedProof, AkitaError>;
```



### Function migration map (selected)


| Before | After |
| ------------------------------------------- | ------------------------------------------------------------------- |
| `shape.num_claims()` | `opening_claims.num_total_polynomials()` |
| `claims.num_claims()` | `opening_claims.num_total_polynomials()` |
| `sample_public_row_coefficients(&shape, t)` | `sample_public_row_coefficients(&layout, t)` (`let layout = opening_claims.layout()`) |
| `batched_eval_target_from_opening_batch(&shape, …)` | `batched_eval_target_from_layout(&layout, …)` |
| `validate_batched_inputs(setup, point, &group_sizes, p)` | `validate_batched_inputs(setup, &opening_claims, p)` |
| `claims.validate(OpeningBatchLimits { … })` | `opening_claims.validate(setup.expanded.seed())` |
| `OpeningBatchShape::new(nv, k)` | `OpeningClaimsLayout::new(nv, k)?` |
| `CommitmentGroupScheduleKey::new(nv, k)` | `PolynomialGroupLayout::new(nv, k)?` |
| `AkitaScheduleLookupKey::single(key)` | `AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(nv, k)?)` |
| `prove_input` / `verify_input` | `ProverOpeningData` / `OpeningClaims` |




### Before / after samples



#### Construction

```rust
let group = PolynomialGroupClaims::new(
    PointVariableSelection::prefix(point.len(), point.len())?,
    evaluations,
    commitment,
)?;

// Build the claims object once, then pass the *whole* object into the prover input.
let opening_claims = OpeningClaims::from_groups(point, vec![group])?;

let group_polys: &[&P] = &[&poly_a, &poly_b];
let prover = ProverOpeningData::new(opening_claims, vec![hint], vec![group_polys])?;
```



#### Verifier root replay

```rust
let layout = opening_claims.layout();
let openings = opening_claims.flat_evaluations();
opening_claims.append_to_transcript::<F, T>(transcript)?;
let row_coefficients = sample_public_row_coefficients::<F, E, T>(&layout, transcript)?;
let target = batched_eval_target_from_layout(&layout, &row_coefficients, &openings)?;
```



#### Prove orchestration

```rust
pub fn batched_prove(..., prover: ProverOpeningData<'a, ...>, ...) -> ... {
    prover.validate::<Cfg::Field>()?;
    let opening_claims = prover.opening_claims();
    // Pass the whole claims object — not opening_claims.point() + group_sizes().
    validate_batched_inputs(expanded.as_ref(), opening_claims, true)?;
    let layout = opening_claims.layout();
    let schedule = Cfg::get_params_for_prove(&layout)?;
    ...
}
```



#### Schedule-only test (was `OpeningBatchShape::new(4, 1)`)

```rust
let layout = OpeningClaimsLayout::new(4, 1)?;
let schedule = Cfg::get_params_for_prove(&layout)?;
```

#### Scalar schedule lookup (was `CommitmentGroupScheduleKey::new(4, 1)`)

```rust
let key = PolynomialGroupLayout::new(4, 1)?;
let schedule = Cfg::runtime_schedule(AkitaScheduleLookupKey::single(key))?;
```

#### Generated table row (was `GeneratedCommitmentGroupScheduleKey`)

```rust
GeneratedScheduleTableEntry {
    key: PolynomialGroupLayout::new(20, 4),
    steps: &[ /* ... */ ],
}
```

## Documentation

- Update book stub: `OpeningClaims` / `ProverOpeningData`.
- Update `specs/single-point-opening-batch.md` API bullets.
- Remove `OpeningBatchShape`, `OpeningBatchLimits`, `CommitmentGroupScheduleKey`, and generated mirror struct names from architecture docs if mentioned.



## Execution

1. Add `akita-types/src/opening_claims.rs`; implement `OpeningClaims`, `PolynomialGroupClaims`, `OpeningClaimsLayout`, and `PolynomialGroupLayout`.
2. Rename schedule types: `PrecommittedGroupParams`, update `AkitaScheduleLookupKey`; delete `CommitmentGroupScheduleKey`; tighten grouped `/2` validation and tests.
3. Collapse generated table types in `akita-planner/src/generated/`; regenerate static schedule modules.
4. Add `akita-prover/src/types/`; implement `ProverOpeningData`.
5. Migrate `CommitmentConfig`, schedule, descriptor, setup to `&OpeningClaimsLayout` and `PolynomialGroupLayout`.
6. Switch PCS traits; migrate verifier then prover protocol. Replace `claims.validate(OpeningBatchLimits { … })` with `opening_claims.validate(setup.expanded.seed())`, then derive the structural view from `opening_claims.layout()`.
7. Delete `OpeningBatchShape`, `OpeningGroupShape`, `OpeningBatchLimits`, old batch structs, generated mirror structs, and retired items from `proof/opening_batch.rs`.
8. Test/bench migration: layout for schedule-only, claims for e2e; keep above-half and larger-than-final grouped validation as acceptance coverage.



## References

- `specs/single-point-opening-batch.md`
- `specs/multi-group-batching.md`
- `crates/akita-types/src/opening_claims.rs` (claims + layout)
- `crates/akita-types/src/schedule.rs` (`PolynomialGroupLayout`, `PrecommittedGroupParams`, `AkitaScheduleLookupKey`)
- `crates/akita-prover/src/types/` (`ProverOpeningData`)
- `crates/akita-types/src/proof/opening_batch.rs` (legacy; deleted during migration)
