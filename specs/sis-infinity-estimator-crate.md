# Spec: Standalone Rust SIS infinity estimator

| Field | Value |
|-------|-------|
| Author(s) | Quang Dao, Codex draft |
| Created | 2026-06-30 |
| Status | proposed |
| PR | |

## Summary

Akita currently generates SIS security tables through Sage and the pinned Python
`lattice-estimator` checkout. The current generated table path prices SIS with a
Euclidean norm bound and the `BDGL16` reduction cost model. That path is useful,
but it is not the main model we want for the next sizing work.

We want a standalone Rust crate that implements the SIS lattice estimator API
for infinity norm estimates first. The crate must expose a general API that is
close to `lattice-estimator`, not a narrow helper for one Akita table. The first
implementation can focus on the paths Akita needs, but the public design must
cover the full SIS lattice estimator surface.

The key target profile for Akita is:

```text
norm = infinity
red_cost_model = ADPS16
red_shape_model = LGSA
zeta = full optimizer
```

The crate must match `lattice-estimator` exactly where the Python result is
stable and trustworthy. Exact matching must be proved with Sage golden outputs.
For infinity norm cases affected by known numerical fragility, the reference
should come from the upstream stability branch, not from the old pinned commit.

The current upstream stability context is:

```text
Repository: https://github.com/malb/lattice-estimator
Pull request: https://github.com/malb/lattice-estimator/pull/217
Title: fix(sis): preserve tiny infinity probabilities
Head: quangvdao:quang/fix-amplify-tiny-success
Head SHA at spec time: c667a48546f140c3a5454c7503c3ca44a264cce2
```

## Goals

1. Build a standalone Rust crate for SIS lattice estimates.

2. Make infinity norm estimation the first production-quality path.

3. Keep the API generic. It must support the same conceptual inputs as
   `lattice-estimator`, including the norm, reduction cost model, reduction
   shape model, optimizer choices, and fixed-parameter cost calls.

4. Match trusted `lattice-estimator` outputs exactly or within an explicit
   tolerance that is justified by numerical precision.

5. Use Sage goldens as the source of truth during development.

6. Add the crate to Akita in a way that can eventually replace the Sage table
   generator, while keeping current checked-in tables stable until the Rust path
   is proven.

## Non-goals

1. Do not hard-code the Akita table profile into the crate API.

2. Do not only implement the single profile `norm=oo, ADPS16, LGSA`.

3. Do not silently change Akita security tables as part of the first crate
   landing.

4. Do not claim parity for cases where the Python reference is known to be
   numerically fragile.

5. Do not require Sage or the Python submodule in Rust CI.

## Terms

### SIS

SIS means Short Integer Solution. In this setting the estimator prices the cost
of finding a nonzero short vector in the kernel of a random modular matrix.

For Akita table generation, the scalar SIS dimensions are:

```text
n = rank * ring_dimension
m = width * ring_dimension
q = representative modulus for the selected family
```

### Infinity norm

Infinity norm means every scalar coordinate must satisfy:

```text
|x_i| <= length_bound
```

This is the `norm=oo` path in `lattice-estimator`.

For Akita, this should be driven by coefficient infinity norm buckets, such as
`COEFF_LINF_BUCKETS`, rather than only by the Euclidean key
`collision_l2_sq`.

### Euclidean norm

Euclidean norm means the whole vector must satisfy:

```text
sqrt(sum_i x_i^2) <= length_bound
```

This is the current Akita table path. It should remain supported, but it is not
the first priority.

### Shape model

A shape model predicts the Gram-Schmidt lengths of a BKZ-reduced q-ary lattice
basis. The shape affects the probability calculation in the infinity norm path.

The crate must support:

1. `GSA`. This is the geometric series assumption. It models the log basis
   profile as a straight line.

2. `ZGSA`. This is a Z-shaped q-ary profile. It keeps the visible q-vectors and
   unit vectors.

3. `LGSA`. This is an L-shaped profile that models rerandomization. It is the
   main Akita target for the infinity norm path.

4. `CN11`. This is the Chen-Nguyen simulator. The Python implementation goes
   through `fpylll.tools.bkz_simulator`. The Rust crate must include this in the
   full design. It may land behind a feature flag if it needs a native
   dependency or a separate port.

5. `CN11_NQ`. This is the CN11 path with the q-ary structure ignored. It must be
   included for API parity, even if it is not an Akita production profile.

### Reduction cost model

A reduction cost model estimates the cost of BKZ and short-vector generation for
a chosen block size.

The crate must support:

1. `ADPS16`. This is the key Akita target. It prices BKZ as `2^(c * beta)` with
   modes `classical`, `quantum`, and `paranoid`.

2. `BDGL16`. This is the current Akita Euclidean table model.

3. `MATZOV`. This is the current default in `lattice-estimator`.

4. `GJ21`.

5. `Kyber`.

6. Other `ReductionCost` models that are required for the public API to match
   `lattice-estimator`. These can land after the first infinity norm path, but
   the type system must not prevent them.

### Beta

`beta` is the BKZ block size. Larger beta usually means a stronger reduction and
larger estimated cost.

The estimator uses beta in two ways:

1. Fixed-beta cost calls, such as `cost_infinity(beta, params, zeta, config)`.

2. Optimized estimates, where the estimator searches for the beta that gives
   the cheapest attack for a fixed zeta.

### Eta

`eta` is the final short-vector or sieve dimension returned by the short-vector
cost model. It appears in the cost output.

### Zeta

`zeta` is the number of coordinates set to zero. The effective SIS dimension is:

```text
d_effective = d - zeta
```

The Python estimator uses a local minimum search over zeta and then also checks
`zeta=0`. For Akita we want a full zeta optimizer. That means the Rust crate
must provide an exhaustive or proven optimizer, not only the Python local search.
The crate should still include the Python-compatible local search for parity
tests and speed comparisons.

## Public API

The API should be close to `lattice-estimator`, but Rust typed.

### Parameters

```rust
pub struct SisParameters {
    pub n: u32,
    pub q: BigUint,
    pub m: Option<u32>,
    pub length_bound: Bound,
    pub norm: SisNorm,
    pub tag: Option<String>,
}

pub enum SisNorm {
    Euclidean,
    Infinity,
}

pub enum Bound {
    Integer(BigUint),
    Float(f64),
    Rational { numerator: BigUint, denominator: BigUint },
}
```

The first implementation can use `u128` for Akita modulus families where that is
enough, but the public API should not make a 128-bit modulus a permanent limit.

### Configuration

```rust
pub struct EstimateConfig {
    pub red_cost_model: ReductionCostModel,
    pub red_shape_model: ShapeModel,
    pub optimizer: OptimizerConfig,
    pub success_probability: Probability,
    pub numeric: NumericConfig,
}

pub enum ReductionCostModel {
    Adps16 { mode: Adps16Mode },
    Bdgl16,
    Matzov { nearest_neighbor: NearestNeighborModel },
    Gj21 { nearest_neighbor: NearestNeighborModel },
    Kyber { nearest_neighbor: NearestNeighborModel },
}

pub enum ShapeModel {
    Gsa,
    Zgsa,
    Lgsa,
    Cn11,
    Cn11NoQary,
}

pub enum OptimizerConfig {
    Fixed { beta: u32, zeta: u32 },
    OptimizeBeta { zeta: u32, beta: SearchMode },
    OptimizeZeta { beta: SearchMode, zeta: SearchMode },
}

pub enum SearchMode {
    PythonLocalMinimum,
    Exhaustive,
    ExhaustiveParallel,
    ProvenPruned,
}
```

### Output

```rust
pub struct LatticeCost {
    pub rop: CostValue,
    pub red: Option<CostValue>,
    pub sieve: Option<CostValue>,
    pub delta: Option<f64>,
    pub beta: Option<u32>,
    pub eta: Option<u32>,
    pub zeta: Option<u32>,
    pub d: u32,
    pub prob: Option<Probability>,
    pub repetitions: Option<CostValue>,
    pub tag: EstimateTag,
}

pub enum CostValue {
    Finite(LogCost),
    Infinity,
}

pub struct LogCost {
    pub log2: f64,
}
```

Store large costs in log space. This avoids overflow and matches how users read
estimator outputs.

### Main calls

```rust
pub fn estimate(params: &SisParameters, config: &EstimateConfig) -> Result<LatticeCost, Error>;

pub fn cost_infinity(
    beta: u32,
    params: &SisParameters,
    zeta: u32,
    config: &EstimateConfig,
) -> Result<LatticeCost, Error>;

pub fn cost_zeta(
    zeta: u32,
    params: &SisParameters,
    config: &EstimateConfig,
) -> Result<LatticeCost, Error>;

pub fn cost_euclidean(
    params: &SisParameters,
    config: &EstimateConfig,
) -> Result<LatticeCost, Error>;
```

These should map directly to the Python concepts:

```text
SIS.lattice(params)
SISLattice.cost_infinity(...)
SISLattice.cost_zeta(...)
SISLattice.cost_euclidean(...)
```

## Implementation Surface

### Crate layout

```text
crates/akita-sis-estimator/
  Cargo.toml
  src/
    lib.rs
    params.rs
    error.rs
    cost.rs
    numeric.rs
    probability.rs
    optimize.rs
    lattice.rs
    table_search.rs
    reduction/
      mod.rs
      delta.rs
      adps16.rs
      bdgl16.rs
      matzov.rs
      gj21.rs
      kyber.rs
      short_vectors.rs
    simulator/
      mod.rs
      gsa.rs
      zgsa.rs
      lgsa.rs
      cn11.rs
      cn11_nq.rs
    akita/
      mod.rs
      profiles.rs
      width_search.rs
  tests/
    lattice_estimator_goldens.rs
    akita_width_goldens.rs
```

### `params.rs`

This module owns:

1. `SisParameters`.

2. Default `m` derivation for infinity norm when `m` is not set.

3. The `updated` operation from Python as a Rust builder-style method.

4. Helpers for Akita q32, q64, and q128 representative moduli.

### `numeric.rs`

This module owns all numerical choices.

It must support:

1. A fast `f64` path.

2. A checked high-precision path for golden generation.

3. Log-space arithmetic for probabilities and costs.

4. Tolerance policy for comparing against Sage.

5. A way to mark a golden cell as fragile if the upstream Python result is not
   trusted.

The crate should not scatter raw `f64` decisions across estimator code. Every
precision choice should pass through `NumericConfig`.

### `probability.rs`

This module owns:

1. Gaussian CDF.

2. Log Gaussian tail probability.

3. Amplification from one-trial success probability to target success
   probability.

4. The tiny-probability behavior fixed by upstream PR 217.

This module must be tested heavily because infinity norm estimates are sensitive
to underflow.

### `reduction`

This module owns reduction cost models.

Required pieces:

1. `delta(beta)`.

2. `beta(delta)`.

3. ADPS16 cost.

4. BDGL16 cost.

5. MATZOV and GJ21 short-vector output.

6. Kyber short-vector output.

7. Common `short_vectors` return type:

```rust
pub struct ShortVectors {
    pub rho: f64,
    pub cost_red: CostValue,
    pub count: CostValue,
    pub sieve_dim: u32,
}
```

### `simulator`

This module owns reduced basis shape simulation.

Required pieces:

1. `GSA`.

2. `LGSA`.

3. `ZGSA`.

4. `CN11`.

5. `CN11_NQ`.

6. A common shape profile return type.

The profile should be held in log space where possible:

```rust
pub struct ShapeProfile {
    pub log2_squared_gso: Vec<f64>,
}
```

The Python code often stores squared Gram-Schmidt lengths. In Rust, the log
form should be the primary internal representation to avoid overflow.

### `optimize.rs`

This module owns all searches.

Required modes:

1. Python-compatible local minimum search.

2. Full beta scan.

3. Full zeta scan.

4. Parallel full zeta scan when the `parallel` feature is enabled.

5. A future proven-pruned search. This is optional for the first version, but
   the public enum should leave room for it.

The full zeta optimizer is required for the Akita target profile.

### `lattice.rs`

This module owns the SIS lattice estimator.

Required paths:

1. `cost_infinity` for fixed beta and zeta.

2. `cost_zeta` for fixed zeta and optimized beta.

3. `estimate` for infinity norm and optimized zeta.

4. `cost_euclidean` for later Euclidean support.

5. Correct handling of the two infinity-norm regimes.

The two regimes are:

```text
sqrt(d) * length_bound <= q
sqrt(d) * length_bound > q
```

The first regime follows the MATZOV-style independent Gaussian coordinate
analysis. The second regime follows the Dilithium-style q-ary analysis where
q-vectors and unit vectors affect the probability.

### `table_search.rs`

This module owns generic max-width search.

It must support:

1. A monotone binary search for the largest width with security at or above a
   target.

2. A cap value that means the result is a lower bound.

3. Both Euclidean and infinity-norm profiles.

4. Both `collision_l2_sq` and coefficient `L∞` bucket inputs.

### `akita`

This module adapts the generic crate to Akita table generation.

Required profiles:

```rust
pub enum AkitaSisEstimatorProfile {
    EuclideanBdgl16,
    InfinityAdps16LgsaFullZeta,
}
```

The Akita profile must not hide the generic estimator. It should only assemble
the generic inputs that Akita uses.

## Golden Strategy

### Reference sources

There are two reference sources.

1. The current pinned Akita submodule:

```text
third_party/lattice-estimator
Pinned SHA in scripts/gen_sis_table.py:
27a581bb8e9d49f5e9e2db315bd48ac769d5f5f5
```

2. The infinity-norm stability branch:

```text
PR: https://github.com/malb/lattice-estimator/pull/217
Head: quangvdao:quang/fix-amplify-tiny-success
Head SHA at spec time: c667a48546f140c3a5454c7503c3ca44a264cce2
```

The Euclidean path can use the current pinned submodule. The infinity path
should use the stability branch for cells affected by tiny probabilities or
underflow.

### Golden metadata

Add a new metadata file for infinity goldens:

```text
scripts/sis_golden/infinity_metadata.json
```

It should record:

1. Remote URL.

2. Branch name.

3. Commit SHA.

4. Estimator profile.

5. Shape model.

6. Cost model.

7. Norm.

8. Optimizer mode.

9. Numeric tolerance.

10. A list of fragile cells that are excluded from exact parity.

### Golden grid

The first infinity golden grid should include:

1. Small fixed estimator cells from the `sis_lattice.py` doctests.

2. Dilithium MSIS parameters from `schemes.py`.

3. Akita q32, q64, and q128 family cells.

4. Fixed beta and zeta cells.

5. Full beta optimized cells.

6. Full zeta optimized cells.

7. Boundary cells near:

```text
sqrt(d) * length_bound = q
length_bound = 1
length_bound = (q - 1) / 2
probability near zero
probability near one
```

### Exact match policy

The crate should match exactly for integer fields and discrete choices:

1. Chosen beta.

2. Chosen eta.

3. Chosen zeta.

4. Effective dimension.

5. Tag.

For floating outputs, compare in log space:

1. `log2(rop)`.

2. `log2(red)`.

3. `log2(sieve)`.

4. `log2(repetitions)`.

Use a strict tolerance for stable cells. Use a documented wider tolerance or an
excluded fragile marker for unstable cells.

## Akita Integration Plan

### Phase 1. Add crate and fixed infinity cost

Add `crates/akita-sis-estimator` as a workspace member.

Implement:

1. `SisParameters`.

2. `ShapeModel::Lgsa`.

3. `ReductionCostModel::Adps16`.

4. Fixed `cost_infinity(beta, zeta)`.

5. Sage goldens for fixed cells.

No Akita table generation changes in this phase.

### Phase 2. Add beta and zeta optimizers

Implement:

1. Python-compatible local beta search.

2. Full beta scan.

3. Full zeta scan.

4. Parallel zeta scan behind the existing workspace `parallel` feature.

Add goldens for:

1. `cost_zeta`.

2. Full infinity estimate.

3. Akita target profile.

### Phase 3. Add all shape models

Implement:

1. `GSA`.

2. `ZGSA`.

3. `CN11`.

4. `CN11_NQ`.

`CN11` can be behind a feature flag if it needs a native dependency or if we
decide to port the Python simulator first.

### Phase 4. Add full reduction model surface

Implement:

1. `BDGL16`.

2. `MATZOV`.

3. `GJ21`.

4. `Kyber`.

5. Any other models needed for lattice-estimator SIS API parity.

### Phase 5. Add Akita infinity table generation

Add a Rust table generator path that can produce max-width tables for:

```text
InfinityAdps16LgsaFullZeta
```

Do not delete the Sage generator yet. The Rust generator should first write a
comparison artifact and pass golden checks.

### Phase 6. Switch offline table generation

Once parity is stable, switch `scripts/stitch_generated_sis_table.py` or its
successor to use the Rust estimator for the new infinity tables.

Keep the Python replay path for audit until the new tables have been reviewed.

### Phase 7. Add Euclidean support

Port Euclidean `norm=2` support after the infinity path is stable.

The Euclidean path should support the current table profile:

```text
norm = Euclidean
red_cost_model = BDGL16
length_bound = sqrt(width * collision_l2_sq)
```

## Table Design for Akita

The current table key is:

```text
(family, ring_dimension, collision_l2_sq) -> max widths by rank
```

For infinity norm, the natural key should be:

```text
(family, ring_dimension, coeff_linf_bound) -> max widths by rank
```

Akita already has coefficient infinity norm buckets in
`crates/akita-types/src/sis/ajtai_key.rs`.

The implementation should support both key types:

```rust
pub enum SisTableKey {
    EuclideanCollisionL2Sq { collision_l2_sq: u128 },
    CoeffInfinityBound { coeff_linf: u128 },
}
```

This avoids forcing infinity estimates through a Euclidean collision bucket.

## Trust and Review Rules

1. The crate must not make a security table looser without a golden update and
   explicit review.

2. If the Rust estimator and Sage disagree, treat Rust as wrong until proven
   otherwise.

3. If the pinned Sage reference is known to be unstable, use PR 217 or its
   merged successor as the reference.

4. Every table must record its estimator profile in generated comments.

5. Generated tables must record the reference source SHA.

6. Akita verifier-facing code must not call the estimator at runtime. The
   estimator is for offline generation and tests.

## Test Plan

1. Unit tests for every numerical helper.

2. Unit tests for every shape model.

3. Unit tests for every reduction cost model.

4. Fixed beta and fixed zeta parity tests.

5. Full beta optimizer parity tests.

6. Full zeta optimizer parity tests.

7. Akita width search parity tests.

8. Monotonicity tests for width by rank and bound.

9. Fragile-cell tests that prove the crate reports instability instead of
   pretending exact parity.

10. Feature tests for `parallel` and optional `cn11`.

## Open Questions

1. Should `CN11` be ported directly from `fpylll.tools.bkz_simulator`, or should
   the Rust crate call native fplll through a feature flag?

2. Should Akita keep separate Euclidean and infinity generated tables, or should
   one table enum carry both profiles?

3. What tolerance should we use for stable floating outputs?

4. Which exact commit should become the long-term infinity golden reference if
   PR 217 is revised before merge?

5. Should the full zeta optimizer be the default for all infinity estimates, or
   only for Akita table generation?

## Acceptance Criteria

The first complete implementation is done when all of the following hold:

1. `akita-sis-estimator` exposes the generic SIS API described above.

2. The infinity path supports ADPS16 and LGSA.

3. The infinity path supports full zeta optimization.

4. The crate passes Sage parity tests against trusted goldens.

5. The crate can generate Akita max-width rows for the infinity profile.

6. The generated output records the profile and reference SHA.

7. Existing Akita tables remain unchanged until a separate table update PR.

8. Rust CI does not require Sage.

9. Sage replay remains available as an offline audit workflow.
