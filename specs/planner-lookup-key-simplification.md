# Spec: Simplify schedule lookup key and opening-batch counts to `(num_vars, num_polynomials)`

| Field         | Value |
|---------------|-------|
| Author(s)     | |
| Created       | 2026-06-25 |
| Status        | proposed |
| PR            | |
| Supersedes    | (partial) `t`/`w`/`z` key fields in [`planner-incidence-generalization.md`](planner-incidence-generalization.md) |
| Superseded-by | |
| Book-chapter  | how/configuration.md |

## Summary

Two related cleanups share one vocabulary: **`num_polynomials`**.

### A. Schedule lookup key

`AkitaScheduleLookupKey` currently carries four usize dimensions:
`num_vars`, `num_t_vectors`, `num_w_vectors`, and `num_z_vectors`. The planner
entry point [`find_schedule`](crates/akita-planner/src/schedule_params.rs) and
every generated schedule table row key off this shape.

For every **supported production opening batch** (scalar, same-point, single
commitment group), those three vector counts are not independent inputs. They
are fixed functions of `OpeningBatchShape`:

| Legacy key field   | Production value                  |
|--------------------|-----------------------------------|
| `num_t_vectors`    | `opening_batch.num_polynomials()` |
| `num_w_vectors`    | `opening_batch.num_polynomials()` |
| `num_z_vectors`    | `1`                               |

The separate `t`/`w`/`z` fields exist for an incidence-generalization design
([`planner-incidence-generalization.md`](planner-incidence-generalization.md))
that planned distinct protocol-vector counts and multi-group `z` sharing. That
design was never the live folded-root path; grouped roots are rejected at
`new_from_opening_batch` and `validate_scalar_root_batch`. Keeping four fields
in the lookup key forces redundant arguments at every call site, complicates
table emission, and invites stale test keys like `(20, 4, 2, 2)` that do not
correspond to any real opening batch.

This spec collapses the public schedule lookup key to two fields that match
[`OpeningBatchShape`](crates/akita-types/src/proof/opening_batch.rs):
`num_vars` and `num_polynomials`. Witness multiplicities needed for proof-size
accounting are derived locally through one helper, not stored in the key.

### B. Opening-batch `num_claims`

[`OpeningBatchShape`](crates/akita-types/src/proof/opening_batch.rs) currently
exposes two names for the same total:

```rust
pub fn num_claims(&self) -> usize {
    self.groups.iter().map(|group| group.num_claims).sum()
}

pub fn num_polynomials(&self) -> usize {
    self.num_claims()  // alias
}
```

Per-group storage uses `OpeningGroupShape.num_claims` even though each slot
holds one committed polynomial opening. The “claim” vocabulary is legacy from
flattened incidence routing; the live same-point protocol opens one evaluation
per polynomial, so **`num_polynomials` is the correct public name**.

This spec deletes the opening-batch `num_claims` surface and makes
`num_polynomials()` the single canonical count on `OpeningBatchShape`,
`OpeningGroupShape`, and `VerifierOpeningBatch`. Call sites that today call
`opening_batch.num_claims()` become `opening_batch.num_polynomials()`.

**Transcript / Fiat-Shamir:** `append_to_transcript` already absorbs the total
polynomial count as a bare `usize`. Switching the Rust call from
`num_claims()` to `num_polynomials()` does not change absorbed bytes.

**Serialized descriptors:** [`CallSection`](crates/akita-types/src/instance_descriptor/mod.rs)
currently stores both `num_polys` and `num_claims` with a consistency check that
they match the per-group sum. Drop the redundant `num_claims` field from
`CallSection` and from `digest_opening_batch` inputs (breaking wire-format
change; acceptable per repo no-compat policy).

Multi-group root batching remains out of scope here; it will use a separate
grouped key type per [`multi-group-batching.md`](multi-group-batching.md).

## Goal

1. Make `AkitaScheduleLookupKey` a two-field value:
   `(num_vars, num_polynomials)`.
2. Mirror the same two-field shape in `GeneratedScheduleKey` and regenerate
   all shipped schedule tables.
3. Replace direct `key.num_*_vectors` reads in planner materialization with a
   single derived-count helper.
4. **Remove opening-batch `num_claims`:** one canonical
   `OpeningBatchShape::num_polynomials()` (and per-group
   `OpeningGroupShape::num_polynomials`); delete `num_claims()` methods and
   redundant descriptor fields.
5. Keep prover/verifier behavior unchanged for supported batches (schedules,
   witness shapes, and transcript absorbs must byte-match before/after for the
   same opening batch).
6. Delete stale API surface (`AkitaScheduleLookupKey::new(_, _, _, _)`,
   `validate_scalar_root_batch`, tiered checks on `num_z_vectors`).

## Non-goals

- Implementing `GroupBatchAkitaScheduleLookupKey` (see
  [`multi-group-batching.md`](multi-group-batching.md)).
- Renaming `num_vars` → `num_variables` in the public API (keep `num_vars` for
  consistency with `OpeningBatchShape::num_vars()` and existing book/spec
  prose).
- Renaming fold/SIS **function parameters** named `num_claims` inside
  `akita-types/src/sis/*`, `akita-challenges`, or fold-grind helpers when they
  denote the per-level fold batch width (root level passes
  `opening_batch.num_polynomials()`; recursive levels pass `1`). Those
  parameters are not part of the opening-batch shape API and may keep their
  local names in a follow-up.
- Changing witness layout formulas in `w_ring_element_count_with_counts*` or
  tail-segment encoding; only **where counts are sourced** changes.
- Backward compatibility with legacy four-tuple keys, old generated table rows,
  or `CallSection` layouts that carried a separate `num_claims` field (repo
  explicitly allows breaking changes).

## Current state

### Opening-batch counts (redundant vocabulary)

[`crates/akita-types/src/proof/opening_batch.rs`](crates/akita-types/src/proof/opening_batch.rs):

```rust
pub struct OpeningGroupShape {
    pub point_vars: PointVariableSelection,
    pub num_claims: usize,  // per-group polynomial count
}

impl OpeningBatchShape {
    pub fn num_claims(&self) -> usize { /* sum group.num_claims */ }
    pub fn num_polynomials(&self) -> usize { self.num_claims() }
}
```

`VerifierOpeningBatch::num_claims()` forwards to the shape; `num_polynomials()`
is a second alias. [`CallSection::from_opening_batch`](crates/akita-types/src/instance_descriptor/mod.rs)
fills both `num_polys` and `num_claims` from the same totals.

Roughly **40+ call sites** across prover, verifier, config, and types read
`opening_batch.num_claims()` (or `shape.num_claims()`) for sizing, validation,
and transcript binding. All of them already assume one opening evaluation per
polynomial at the supported same-point root.

### Key definition

[`crates/akita-types/src/schedule.rs`](crates/akita-types/src/schedule.rs):

```rust
pub struct AkitaScheduleLookupKey {
    pub num_vars: usize,
    pub num_t_vectors: usize,
    pub num_w_vectors: usize,
    pub num_z_vectors: usize,
}
```

Construction paths:

- `singleton(num_vars)` → `(nv, 1, 1, 1)`
- `new(nv, t, w, z)` — general four-tuple
- `new_from_opening_batch(batch)` → `(batch.num_vars(), batch.num_polynomials())`

For `OpeningBatchShape::new(nv, num_polys)` this always yields `(nv, num_polys)`.

### Where the key is consumed

| Area | Role |
|------|------|
| [`akita-planner/src/schedule_params.rs`](crates/akita-planner/src/schedule_params.rs) | DP search: root commit sizing, fold Linf cap, next-`w` length, EOR bytes |
| [`akita-planner/src/resolve.rs`](crates/akita-planner/src/resolve.rs) | Table expansion, `generated_schedule_lookup_key` |
| [`akita-planner/src/emit/mod.rs`](crates/akita-planner/src/emit/mod.rs) | Offline table emission |
| [`akita-planner/src/generated/mod.rs`](crates/akita-planner/src/generated/mod.rs) | `GeneratedScheduleKey` mirror |
| [`akita-config`](crates/akita-config) | `CommitmentConfig::runtime_schedule`, `family_keys`, tests |
| [`akita-schedules/src/generated/*`](crates/akita-schedules/src/generated/) | ~20 regenerated family modules |
| [`akita-types/src/proof/direct_witness.rs`](crates/akita-types/src/proof/direct_witness.rs) | `terminal_fold_segment_counts`, `terminal_direct_witness_shape_for_key` |
| [`akita-setup`](crates/akita-setup/src/lib.rs) | Disk cache fingerprint |
| Tests / examples / profile harness | Synthetic key construction |

Prover and verifier **do not** read `AkitaScheduleLookupKey` at runtime for
witness construction; they consume the expanded `Schedule` and
`CleartextWitnessShape` produced by the planner path.

## Proposed design

### 1. Opening-batch: `num_polynomials` only

[`crates/akita-types/src/proof/opening_batch.rs`](crates/akita-types/src/proof/opening_batch.rs):

```rust
pub struct OpeningGroupShape {
    pub point_vars: PointVariableSelection,
    /// Polynomials opened in this commitment group.
    pub num_polynomials: usize,
}

impl OpeningBatchShape {
    pub fn check(&self) -> Result<(), AkitaError> {
        if self.groups.is_empty() || self.num_polynomials() == 0 {
            return Err(AkitaError::InvalidProof);
        }
        // ...
    }

    /// Total polynomials opened across all commitment groups.
    pub fn num_polynomials(&self) -> usize {
        self.groups.iter().map(|group| group.num_polynomials).sum()
    }

    pub fn num_polys_per_commitment_group(&self) -> Vec<usize> {
        self.groups.iter().map(|group| group.num_polynomials).collect()
    }

    pub fn append_to_transcript<F, T>(&self, transcript: &mut T) -> Result<(), AkitaError> {
        self.check()?;
        transcript.append_serde(ABSORB_BATCH_SHAPE, &self.num_vars());
        transcript.append_serde(ABSORB_BATCH_SHAPE, &self.num_polynomials());
        transcript.append_serde(ABSORB_BATCH_SHAPE, &self.num_commitment_groups());
        for group in self.groups() {
            transcript.append_serde(ABSORB_BATCH_SHAPE, &group.num_polynomials);
            // ...
        }
        Ok(())
    }
}
```

Delete: `OpeningBatchShape::num_claims()`, `VerifierOpeningBatch::num_claims()`.

Update constructors:

```rust
groups.push(OpeningGroupShape {
    point_vars: PointVariableSelection::prefix(num_vars, num_vars)?,
    num_polynomials: group_size,
});
```

[`OpeningBatchLimits`](crates/akita-types/src/proof/opening_batch.rs): rename
`max_num_claims` → `max_num_polynomials`.

[`CallSection`](crates/akita-types/src/instance_descriptor/mod.rs) after refactor:

```rust
pub struct CallSection {
    pub num_polys: u32,
    // num_claims deleted — num_polys is authoritative
    pub num_commitment_groups: u32,
    pub num_polys_per_commitment_group: Vec<u32>,
    // ...
}

impl CallSection {
    pub fn from_opening_batch(...) -> Result<Self, AkitaError> {
        Ok(Self {
            num_polys: usize_to_u32(opening_batch.num_polynomials(), "num_polys")?,
            num_polys_per_commitment_group: opening_batch
                .groups
                .iter()
                .map(|group| usize_to_u32(group.num_polynomials, "num_polys_per_commitment_group"))
                .collect::<Result<_, _>>()?,
            // ...
        })
    }
}
```

Mechanical call-site rule for the opening-batch domain:

```rust
// before
opening_batch.num_claims()
shape.num_claims()
group.num_claims
limits.max_num_claims

// after
opening_batch.num_polynomials()
shape.num_polynomials()
group.num_polynomials
limits.max_num_polynomials
```

Helper functions in the same module (`sample_public_row_coefficients`,
`batched_eval_target_from_opening_batch`, `VerifierOpeningBatch::with_padded_point`)
take `num_polynomials` parameters and compare against
`shape.num_polynomials()`.

### 2. Two-field lookup key

[`crates/akita-types/src/schedule.rs`](crates/akita-types/src/schedule.rs):

```rust
/// Public runtime key that selects a concrete root schedule context.
///
/// Describes the supported scalar same-point opening batch:
/// `num_vars` coordinates in the shared point and `num_polynomials` committed
/// polynomials opened at that point (one claim per polynomial).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AkitaScheduleLookupKey {
    /// Root polynomial variable count (shared opening point arity).
    pub num_vars: usize,
    /// Number of polynomials in the single commitment group.
    pub num_polynomials: usize,
}

impl AkitaScheduleLookupKey {
    pub const fn singleton(num_vars: usize) -> Self {
        Self {
            num_vars,
            num_polynomials: 1,
        }
    }

    pub const fn new(num_vars: usize, num_polynomials: usize) -> Self {
        Self {
            num_vars,
            num_polynomials,
        }
    }

    pub fn new_from_opening_batch(opening_batch: &OpeningBatchShape) -> Result<Self, AkitaError> {
        if opening_batch.num_commitment_groups() != 1 {
            return Err(AkitaError::InvalidSetup(
                "scalar schedule lookup cannot collapse a multi-commitment batch; \
                 see specs/multi-group-batching.md"
                    .to_string(),
            ));
        }
        Ok(Self::new(
            opening_batch.num_vars(),
            opening_batch.num_polynomials(),
        ))
    }

    pub fn validate(self) -> Result<(), AkitaError> {
        if self.num_vars == 0 || self.num_polynomials == 0 {
            return Err(AkitaError::InvalidSetup(
                "schedule lookup key dimensions must be at least 1".to_string(),
            ));
        }
        Ok(())
    }
}
```

Delete: `new(_, _, _, _)`, `validate_scalar_root_batch`.

### 3. Derived witness counts (internal, not part of the key)

Add a small derived struct next to the key (same module):

```rust
/// Root witness multiplicities implied by a scalar same-point lookup key.
///
/// These are the counts the planner and terminal witness materializer pass to
/// `w_ring_element_count_with_counts*` and tail-segment layout helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarRootWitnessCounts {
    pub num_t_vectors: usize,
    pub num_w_vectors: usize,
    pub num_z_vectors: usize,
}

impl AkitaScheduleLookupKey {
    pub const fn scalar_root_witness_counts(self) -> ScalarRootWitnessCounts {
        ScalarRootWitnessCounts {
            num_t_vectors: self.num_polynomials,
            num_w_vectors: self.num_polynomials,
            num_z_vectors: 1,
        }
    }
}
```

Rationale:

- Same-point batching opens one claim per polynomial → `w` multiplicity tracks
  `num_polynomials`, not a separate “opening point count”.
- Single commitment group at one point → `z` multiplicity is always `1`.
- `t` multiplicity is the committed polynomial count.

Replace [`terminal_fold_segment_counts`](crates/akita-types/src/proof/direct_witness.rs):

```rust
pub fn terminal_fold_segment_counts(
    key: AkitaScheduleLookupKey,
    terminal_fold_level: usize,
) -> (usize, usize, usize, usize) {
    if terminal_fold_level == 0 {
        let c = key.scalar_root_witness_counts();
        (c.num_w_vectors, c.num_t_vectors, c.num_z_vectors, 1)
    } else {
        (1, 1, 1, 1)
    }
}
```

All planner reads of `key.num_t_vectors` / `key.num_w_vectors` /
`key.num_z_vectors` become `let c = key.scalar_root_witness_counts();` followed
by `c.num_*_vectors`.

### 4. Generated table key

[`crates/akita-planner/src/generated/mod.rs`](crates/akita-planner/src/generated/mod.rs):

```rust
pub struct GeneratedScheduleKey {
    pub num_vars: usize,
    pub num_polynomials: usize,
}
```

[`generated_schedule_lookup_key`](crates/akita-planner/src/resolve.rs):

```rust
pub const fn generated_schedule_lookup_key(key: AkitaScheduleLookupKey) -> GeneratedScheduleKey {
    GeneratedScheduleKey {
        num_vars: key.num_vars,
        num_polynomials: key.num_polynomials,
    }
}
```

Emitted row shape (from [`emit/mod.rs`](crates/akita-planner/src/emit/mod.rs)):

```rust
GeneratedScheduleKey { num_vars: 20, num_polynomials: 4 }
```

instead of `{ num_vars: 20, num_t_vectors: 4, num_w_vectors: 4, num_z_vectors: 1 }`.

### 5. Planner entry points after refactor

[`find_schedule`](crates/akita-planner/src/schedule_params.rs) signature stays
the same (`key: AkitaScheduleLookupKey`), but the body becomes:

```rust
pub fn find_schedule(
    key: AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<Schedule, AkitaError> {
    key.validate()?;
    let counts = key.scalar_root_witness_counts();

    let t_vectors = counts.num_t_vectors;
    // ... remainder uses `counts.num_*_vectors` instead of `key.num_*_vectors`
}
```

[`resolve_schedule`](crates/akita-planner/src/resolve.rs): replace
`validate_scalar_root_batch()` with `validate()`; delete the matching
`policy.tiered && key.num_z_vectors != 1` guard (stale after key simplification;
grouped-root rejection moves to `new_from_opening_batch` / future grouped key
type). Same `counts` pattern in `schedule_from_entry` and EOR helpers.

[`CommitmentConfig::runtime_schedule`](crates/akita-config/src/lib.rs) and
[`get_params_for_prove`](crates/akita-config/src/lib.rs) are unchanged at the
type level; they already derive the key from `OpeningBatchShape`.

### 6. Table enumeration

[`family_keys`](crates/akita-config/src/generated_families.rs) stays logically
the same but produces two-field keys:

```rust
for &num_polys in family.num_polys {
    for nv in family.min_num_vars..=family.max_num_vars {
        let opening_batch = OpeningBatchShape::new(nv, num_polys)?;
        keys.push(AkitaScheduleLookupKey::new_from_opening_batch(&opening_batch)?);
    }
}
```

No behavioral change to which `(nv, num_polys)` pairs are emitted.

## Files to change

### Tier 1 — opening-batch + schedule key contract (must land together)

| File | Change |
|------|--------|
| [`crates/akita-types/src/proof/opening_batch.rs`](crates/akita-types/src/proof/opening_batch.rs) | `OpeningGroupShape.num_polynomials`; delete `num_claims()`; rename limits; update transcript, helpers, tests |
| [`crates/akita-types/src/instance_descriptor/mod.rs`](crates/akita-types/src/instance_descriptor/mod.rs) | Drop `CallSection.num_claims`; build from `num_polynomials()` only; update (de)serialization + validation |
| [`crates/akita-types/src/instance_descriptor/tests.rs`](crates/akita-types/src/instance_descriptor/tests.rs) | Fixture updates for slim `CallSection` |
| [`crates/akita-types/src/schedule.rs`](crates/akita-types/src/schedule.rs) | Two-field key, `ScalarRootWitnessCounts`, delete legacy constructors/validators, update unit tests |
| [`crates/akita-types/src/proof/direct_witness.rs`](crates/akita-types/src/proof/direct_witness.rs) | `terminal_fold_segment_counts` uses derived counts |
| [`crates/akita-types/src/proof/batch.rs`](crates/akita-types/src/proof/batch.rs) | Setup-capacity checks use `num_polynomials()` wording |
| [`crates/akita-types/src/proof/setup_prefix.rs`](crates/akita-types/src/proof/setup_prefix.rs) | `opening_batch.num_polynomials()` |
| [`crates/akita-planner/src/generated/mod.rs`](crates/akita-planner/src/generated/mod.rs) | Two-field `GeneratedScheduleKey` |
| [`crates/akita-planner/src/resolve.rs`](crates/akita-planner/src/resolve.rs) | Key projection, validation, materialization reads |
| [`crates/akita-planner/src/schedule_params.rs`](crates/akita-planner/src/schedule_params.rs) | DP search reads derived counts |
| [`crates/akita-planner/src/emit/mod.rs`](crates/akita-planner/src/emit/mod.rs) | Emit two-field keys |
| [`crates/akita-planner/src/generated/expand.rs`](crates/akita-planner/src/generated/expand.rs) | Update comments referencing `num_t_vectors` as key field |
| [`crates/akita-planner/src/catalog_identity.rs`](crates/akita-planner/src/catalog_identity.rs) | Any key-field assertions in identity checks |

### Tier 2 — prover / verifier / config (mechanical `num_claims` → `num_polynomials`)

| File | Change |
|------|--------|
| [`crates/akita-prover/src/protocol/core/root_fold.rs`](crates/akita-prover/src/protocol/core/root_fold.rs) | Batch validation sizing |
| [`crates/akita-prover/src/protocol/core/extension_opening_reduction.rs`](crates/akita-prover/src/protocol/core/extension_opening_reduction.rs) | EOR partial counts |
| [`crates/akita-prover/src/protocol/ring_switch/evals.rs`](crates/akita-prover/src/protocol/ring_switch/evals.rs) | Root batch sourced from `num_polynomials()` |
| [`crates/akita-prover/src/backend/tensor_fold.rs`](crates/akita-prover/src/backend/tensor_fold.rs) | Opening batch read |
| [`crates/akita-verifier/src/protocol/core/verify.rs`](crates/akita-verifier/src/protocol/core/verify.rs) | Witness count check |
| [`crates/akita-verifier/src/proof/direct.rs`](crates/akita-verifier/src/proof/direct.rs) | Opening vs witness length check |
| [`crates/akita-verifier/src/protocol/ring_switch.rs`](crates/akita-verifier/src/protocol/ring_switch.rs) | Root relation batch fields (keep internal `num_claims` only where it is fold-local state filled from `num_polynomials()` at construction) |
| [`crates/akita-verifier/src/protocol/slice_mle/*`](crates/akita-verifier/src/protocol/slice_mle/) | Fixtures + structured slice reads |
| [`crates/akita-config/src/proof_optimized.rs`](crates/akita-config/src/proof_optimized.rs) | `worst_case_opening_batch(num_vars, num_polynomials)` naming |
| [`crates/akita-config/src/lib.rs`](crates/akita-config/src/lib.rs) | Update inline tests using four-tuple `new` |
| [`crates/akita-config/src/generated_families.rs`](crates/akita-config/src/generated_families.rs) | Unchanged enumeration logic |
| [`crates/akita-config/src/test_support.rs`](crates/akita-config/src/test_support.rs) | `akita_batched_root_layout(num_vars, num_polynomials)` |
| [`crates/akita-config/src/proof_optimized/tests.rs`](crates/akita-config/src/proof_optimized/tests.rs) | Replace four-tuple keys; `num_polynomials()` in layout asserts |
| [`crates/akita-config/tests/runtime_fallback.rs`](crates/akita-config/tests/runtime_fallback.rs) | Invalid-key cases use two-field API |
| [`crates/akita-config/tests/tiered_planner.rs`](crates/akita-config/tests/tiered_planner.rs) | Two-field keys; drop grouped stub |
| [`crates/akita-config/tests/generated_tables.rs`](crates/akita-config/tests/generated_tables.rs) | Positional key equality against regenerated tables |
| [`crates/akita-config/tests/regen_diff.rs`](crates/akita-config/tests/regen_diff.rs) | Regen byte-diff guard |
| [`crates/akita-config/tests/commitment_group_layout_probe.rs`](crates/akita-config/tests/commitment_group_layout_probe.rs) | `(nv, 1)` keys |
| [`crates/akita-config/tests/schedule_catalog_*.rs`](crates/akita-config/tests/) | Keys from `new_from_opening_batch` only |
| [`crates/akita-pcs/src/scheme/tests/fp32_ext4.rs`](crates/akita-pcs/src/scheme/tests/fp32_ext4.rs) | `max_num_polynomials` / `opening_batch.num_polynomials()` |
| [`crates/akita-setup/src/lib.rs`](crates/akita-setup/src/lib.rs) | Cache fingerprint: two-field lookup key + explicit `planner_v7_` prefix (see [Resolved decisions](#resolved-decisions)) |
| [`crates/akita-schedules/src/generated/*.rs`](crates/akita-schedules/src/generated/) | **Full regen** via `gen_schedule_tables` |

### Tier 3 — tests / examples (mechanical)

| File | Change |
|------|--------|
| [`crates/akita-planner/src/resolve.rs`](crates/akita-planner/src/resolve.rs) (tests) | Update/remove grouped-key rejection test |
| [`crates/akita-prover/src/protocol/core/tests.rs`](crates/akita-prover/src/protocol/core/tests.rs) | Key from opening batch (likely unchanged call path) |
| [`crates/akita-pcs/tests/akita_e2e.rs`](crates/akita-pcs/tests/akita_e2e.rs) | `singleton` call sites unchanged |
| [`crates/akita-pcs/examples/profile/modes.rs`](crates/akita-pcs/examples/profile/modes.rs) | **Drop** the `EXT_DEGREE > 1` Frobenius four-tuple `(nv, 1, width, width)`; report against the same `new_from_opening_batch`-derived key (`singleton(nv)` for `np=1`) the prover actually resolves. This is **not** a mechanical `(protocol_nv, num_polys)` rewrite — see [Resolved decisions](#resolved-decisions) |
| [`crates/akita-pcs/examples/profile/report.rs`](crates/akita-pcs/examples/profile/report.rs) | If it prints key fields |

### Tier 4 — documentation (same PR or immediate follow-up)

| File | Change |
|------|--------|
| [`crates/akita-planner/README.md`](crates/akita-planner/README.md) | Key section lists `num_vars` + `num_polynomials` only |
| [`specs/planner-incidence-generalization.md`](specs/planner-incidence-generalization.md) | Header note: key simplification superseded the four-field interim key |
| [`specs/schedule-catalog-ownership.md`](specs/schedule-catalog-ownership.md) | Update key-shape examples |
| [`specs/multi-group-batching.md`](specs/multi-group-batching.md) | Clarify scalar key is two-field; grouped key remains future |
| [`book/src/how/configuration.md`](book/src/how/configuration.md) | Stub refresh when folded |

**Out of scope for rename (fold/SIS internals):** `akita-types/src/sis/*`,
`akita-challenges/src/fold_draw.rs`, and fold-grind helpers keep function
parameters named `num_claims` where they mean per-level fold batch width, not
opening-batch totals. Root call sites pass `opening_batch.num_polynomials()`.

## Regeneration procedure

After Tier 1 lands:

```bash
cargo run --release -p akita-config --no-default-features --bin gen_schedule_tables -- \
  crates/akita-schedules/src/generated
cargo run --release -p akita-config --no-default-features --features zk --bin gen_schedule_tables -- \
  crates/akita-schedules/src/generated
cargo run --release -p akita-config --no-default-features --bin gen_schedule_tables -- \
  crates/akita-schedules/src/generated --wiring-only
```

Expect large diffs in every `fp*_*.rs` generated module; schedule **steps** for
`(nv, num_polys)` pairs should match the pre-change tables (only key literals
shrink). Verify with:

```bash
cargo test -p akita-config generated_tables regen_diff
cargo test -p akita-planner
cargo test
```

## Behavioral equivalence

For every opening batch constructible via
`OpeningBatchShape::new(nv, num_polys)` or `new_from_opening_batch`:

| Artifact | Before | After | Expected |
|----------|--------|-------|----------|
| DP schedule bytes | key `(nv, p, p, 1)` | key `(nv, p)` | Identical |
| Table hit expansion | same | same | Identical |
| Terminal witness shape | same | same | Identical |
| Root `LevelParams` | same | same | Identical |
| Batch-shape transcript absorbs | total via `num_claims()` | total via `num_polynomials()` | Identical numeric values |
| Fiat-Shamir plan digest | same | same | Identical |

Keys that were **only** reachable via the deleted four-tuple API and do not
correspond to a valid `OpeningBatchShape` (e.g. `(20, 4, 2, 2)`) are
intentionally dropped. The dense-extension profile key
`(nv, 1, [E:F], [E:F])` built in `run_dense_mode_for` is one such key; it is a
**reporting-only proof-size estimate** that the prover never consumes, and it
models a protocol route that no longer exists (see
[Resolved decisions](#resolved-decisions)).

`CallSection` on-disk / serialized layout **will change** when `num_claims` is
removed from the struct; refresh any fixture bytes in descriptor tests in the
same PR.

## Migration checklist

- [ ] Land Tier 1 opening-batch + schedule-key types together.
- [ ] Mechanical `num_claims()` → `num_polynomials()` sweep (Tier 2–3).
- [ ] Update `CallSection` serialization and descriptor test fixtures.
- [ ] Regenerate all schedule tables; refresh catalog identity if needed.
- [ ] Run `./scripts/check-doc-guardrails.sh` after doc touch (Tier 4).
- [ ] Update disk-cache fingerprint in `akita-setup`: bump embedded schedule tag
      to `planner_v7_` (see [Resolved decisions](#resolved-decisions)).
- [ ] Grep guard: no `OpeningBatchShape::num_claims`, `\.num_claims\(\)` on
      opening-batch types, or `OpeningGroupShape { num_claims` in production
      code (tests for removed API excepted during transition).

## Acceptance criteria

1. `AkitaScheduleLookupKey` and `GeneratedScheduleKey` each expose exactly
   `num_vars` and `num_polynomials`.
2. `OpeningBatchShape`, `OpeningGroupShape`, and `VerifierOpeningBatch` expose
   `num_polynomials()` only; no `num_claims()` methods and no
   `OpeningGroupShape.num_claims` field.
3. `CallSection` has no `num_claims` field; `num_polys` is the sole total
   polynomial count in the descriptor.
4. No production Rust code reads `key.num_t_vectors`, `key.num_w_vectors`, or
   `key.num_z_vectors` (grep clean except `ScalarRootWitnessCounts` / layout
   helpers that take explicit count parameters).
5. `OpeningBatchShape` → key → schedule path unchanged for all e2e tests;
   transcript absorbs unchanged for supported batches.
6. `gen_schedule_tables` regen + `generated_tables` / `regen_diff` tests green.
7. Planner README and this spec accurately describe the unified vocabulary.
8. No `policy.tiered && … num_z_vectors` guards remain in planner resolution
   (`find_schedule`, `resolve_schedule`).

## Resolved decisions

### Tiered grouped-root guard — delete (stale)

After the two-field key lands, `scalar_root_witness_counts()` always yields
`num_z_vectors == 1`. The existing checks in [`find_schedule`](crates/akita-planner/src/schedule_params.rs)
and [`resolve_schedule`](crates/akita-planner/src/resolve.rs):

```rust
if policy.tiered && key.num_z_vectors != 1 { ... }
```

are tautological and should be **removed**, not kept as defensive documentation.
Grouped-root rejection for tiered presets remains where it already lives:

- `OpeningBatchShape::num_commitment_groups() != 1` at
  `AkitaScheduleLookupKey::new_from_opening_batch`
- future `GroupBatchAkitaScheduleLookupKey` per
  [`multi-group-batching.md`](multi-group-batching.md)

### Dense-extension profile four-tuple — drop (stale estimate)

`run_dense_mode_for` in [`crates/akita-pcs/examples/profile/modes.rs`](crates/akita-pcs/examples/profile/modes.rs)
is the **only** place that builds a key whose vector counts are genuinely
independent of `OpeningBatchShape`. For `Cfg::EXT_DEGREE > 1` it constructs

```rust
let width = 1usize << Cfg::EXT_DEGREE.trailing_zeros(); // = [E:F] = 2^kappa
AkitaScheduleLookupKey::new(nv, /*t*/ 1, /*w*/ width, /*z*/ width);
```

This `(t=1, w=[E:F], z=[E:F])` shape **cannot** be expressed by the two-field
key (which forces `t == w == num_polynomials`, `z == 1`), so it deserves an
explicit decision rather than a mechanical rewrite.

The decision is to **drop it**, for two reasons:

1. **It is reporting-only.** The harness uses this `plan` solely for
   `report_proof_size_against_planner` / `emit_runtime_schedule_summary` /
   `emit_proof_tail_report`. The actual proof comes from `batched_prove`, which
   takes no schedule argument and resolves its own schedule from the real
   opening batch (`get_params_for_prove → new_from_opening_batch`, hence
   `z == 1`) or the root-direct fallback. `run_prove` already prints a warning
   that "folded planner byte estimates do not apply" for `EXT_DEGREE > 1`.
2. **It models a removed protocol route.** The `w = z = 2^kappa` factor is the
   Phase 5B Frobenius / product-coordinate estimate from
   [`extension-field-opening-batching.md`](extension-field-opening-batching.md)
   (`opening width = 2^t`), where an `[E:F]`-degree opening was carried as
   `2^kappa` conjugate openings. That route has been removed from the live
   prover; Phase 5C reprices the reduced path to **one** carried opening
   (`carried opening width = 1`, `tensor partial count = [E:F]`), with the
   `[E:F]` multiplicity living in `ExtensionOpeningReductionProof` partials, not
   in root `w`/`z` vector counts.

Replacement: report the dense-extension profile against the same
`new_from_opening_batch`-derived key the prover uses (`singleton(nv)` for the
single-polynomial dense case). If an `[E:F]`-aware byte estimate is still
wanted, account for it through the EOR partial-byte path, not through phantom
`w`/`z` vectors. Net effect: the dense-extension profile's *printed* byte
estimate may change (it was already non-representative of the live root-direct /
tensor-reduced path); real prove/verify behavior is unaffected because it never
used this key.

### Disk-cache version bump — do it

When updating [`akita-setup/src/lib.rs`](crates/akita-setup/src/lib.rs)
`cache_file_name`, bump the schedule fingerprint prefix from `planner_v6_` to
`planner_v7_` and drop the legacy `_t/_w/_z` key tags from the filename stem.
Use the two-field lookup key `(max_num_vars, max_num_batched_polys)` plus the
existing `digest_effective_schedule` hex suffix.

Example stem shape after refactor:

```text
planner_v7_nv{max_nv}_batch{max_batch}_{schedule_digest_hex}
```

Old cached setup files under `planner_v6_*` names will not be reused (acceptable
per repo no-compat policy); the version prefix makes the cutover explicit.

## Open questions

1. **Fold/SIS parameter rename:** optional follow-up to rename internal
   `num_claims` parameters in `sis::fold_witness_beta` and friends to
   `num_fold_polys` or similar so “claims” disappears entirely from the codebase.
   Deferred — discuss separately before any rename PR.

## Related specs

- [`planner-incidence-generalization.md`](planner-incidence-generalization.md) — historical; four-field key and separate `num_claims` vocabulary superseded by this spec for scalar paths.
- [`multi-group-batching.md`](multi-group-batching.md) — future grouped lookup key; per-group counts will use `num_polynomials`, not `num_claims`.
- [`schedule-catalog-ownership.md`](schedule-catalog-ownership.md) — table regen workflow unchanged aside from slimmer keys.
- [`extension-field-opening-batching.md`](extension-field-opening-batching.md) — explains why the dense-extension profile's `(nv, 1, [E:F], [E:F])` four-tuple is a stale Phase 5B Frobenius estimate (superseded by the Phase 5C width-1 tensor reduction), justifying its removal here.
