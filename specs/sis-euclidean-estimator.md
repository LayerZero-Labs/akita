# Spec: Rust Euclidean SIS Estimator (`akita-sis-estimator`)

| Field       | Value |
|-------------|-------|
| Author(s)   | Quang Dao, Cursor agent draft |
| Created     | 2026-06-05 |
| Status      | proposed |
| PR          | (branch `quang/s3-s5-sis-estimator-spec`) |

## Summary

Akita's L2 MSIS cutover ([`l2-msis-opnorm-folded-witness.md`](l2-msis-opnorm-folded-witness.md))
needs regenerated `generated_sis_table.rs` rows: for each representative modulus family,
ring dimension, squared-Euclidean collision bucket, and module rank, the maximum commitment
width that still yields at least 128-bit security under a fixed lattice-reduction cost model.

Today that table is produced offline by `scripts/gen_sis_table.py`, which calls
[lattice-estimator](https://github.com/malb/lattice-estimator) through SageMath.
That path is fragile (Sage/cysignals hangs on degenerate instances, parallel regen crashes,
no CI regen, schema drift between the Python bucket ladder and the checked-in Rust table).

This spec defines an in-repo Rust estimator crate and table-generation binary that
reproduce the **narrow** surface Akita actually uses (Euclidean SIS + BDGL16 only),
with golden validation against lattice-estimator and deterministic, parallel-friendly regen.

The crate is an **offline build tool**, not a runtime prover/verifier dependency.

## Intent

### Goal

Add workspace crate `akita-sis-estimator` and binary `gen_sis_table` (name may match the
existing script for drop-in replacement) that:

1. Implements `sis_euclidean_security_bits(n, q, m, length_bound) -> f64` matching
   lattice-estimator `SIS.lattice(..., norm=2, red_cost_model=BDGL16)` on the
   `cost_euclidean` code path (see Design).
2. Exposes `max_secure_width(rank, d, collision_l2_sq, q, target_bits, search_cap) -> u64`
   via the same monotone width search the Python script uses today.
3. Emits Rust match arms (or CSV) for `crates/akita-types/src/sis/generated_sis_table.rs`,
   using power-of-two buckets `2^MIN_LOG_BUCKET .. 2^MAX_LOG_BUCKET` in lockstep with
   `akita-types::sis::ajtai_key::{MIN_LOG_BUCKET, MAX_LOG_BUCKET}`.
4. Records provenance in generated file headers (estimator git commit or crate version,
   target bits, search caps, family moduli).

Parent spec slices **S5a** (this spec) unblocks **S5b** (table stitch + `collision_l2_sq`
rename cutover) and **S11** (schedule table regen under L2 pricing).

### Invariants

- **Reference equivalence.** On a pinned golden grid of `(n, q, m, length_bound)` instances,
  `log2(rop)` from the Rust core matches lattice-estimator within a documented tolerance
  (exact where both sides use the same closed forms; at most ±0.01 bits on borderline
  `beta(δ)` knees unless a deliberate conservative timeout applies).
- **Conservative timeout.** If the `beta(δ)` inversion exceeds a fixed iteration or wall
  budget, return `-inf` security bits (treat as insecure). This must never **over**-report
  secure width (may under-report by at most one width at degenerate knees, same policy as
  the Python script intent).
- **Monotone search.** For fixed `(n, q, collision_l2_sq, rank)`, security bits are
  non-increasing in commitment width `w`; width search may assume this predicate.
- **No runtime coupling.** Prover, verifier, and planner continue to read only
  `generated_sis_table.rs`; they do not invoke the estimator at proof time.
- **Single cost model.** Euclidean (`norm = 2`) + `BDGL16` asymptotic reduction cost only.
  The `lgsa` shape model is **not** part of this estimator (it is unused on the Euclidean
  path in lattice-estimator today).
- **Parameter mapping unchanged** from the L2 parent spec and the updated Python script:

  | Akita | Estimator |
  |-------|-----------|
  | module rank `r` | `n = r · d` |
  | ring dimension `d` | folded into `n`, `m` |
  | width `w` (ring elements) | `m = w · d` |
  | per-row `collision_l2_sq` | `length_bound = sqrt(w · collision_l2_sq)` |
  | `SisModulusFamily` | representative `q` (Q32/Q64/Q128) |

- **Bucket ladder lockstep.** Rust table keys are exact powers of two; raw collisions round
  up via `u128::next_power_of_two` in `ceil_supported_collision` (parent S5b).

### Non-Goals

- Porting all of lattice-estimator (LWE, NTRU, infinity-norm SIS, LGSA/CN11 simulators,
  gaussian sampling attacks, batch drivers).
- Changing the 128-bit security target, BDGL16 model, or representative moduli without an
  explicit spec amendment.
- Runtime on-demand SIS estimation inside the planner DP (shipped tables remain static;
  DP fallback uses the same precomputed table).
- Replacing the parent L2 protocol spec or implementing S6–S13 (folded-witness certificate,
  sumchecks, e2e) in this crate.

## Evaluation

### Acceptance Criteria

- [ ] `akita-sis-estimator` crate lands with unit tests and a `gen_sis_table` binary.
- [ ] Golden file (CSV or JSON) checked in under `crates/akita-sis-estimator/tests/data/`
  (or generated once and committed) with ≥50 representative cells spanning all three
  families, all `d ∈ {32,64,128,256}`, ranks `{1,5,20}`, and collision buckets including
  known degenerate knees (`q32`, `d=32`, `r=1`, `collision_l2_sq=16384`, widths `3..6`).
- [ ] `cargo test -p akita-sis-estimator` passes golden comparison against lattice-estimator
  export (documented command in crate README).
- [ ] Full regen completes for all `(family, d)` arms in <2h wall time on a 16-core laptop
  (parallelism via `rayon`; no Sage dependency).
- [ ] Stitched `generated_sis_table.rs` uses only power-of-two keys and updated provenance
  header; `cargo test -p akita-types` and `assert_schedule_stays_within_audited_sis_widths`
  pass after S5b lands.
- [ ] `scripts/gen_sis_table.py` is either removed or reduced to a thin wrapper that
  delegates to the Rust binary (decision at implementation time; must not be the only
  regen path after S5b).

### Testing Strategy

- **Golden tests:** `sis_euclidean_security_bits` vs lattice-estimator CSV rows.
- **Property tests (optional):** security bits monotone decreasing in `m` for fixed
  `(n, q, length_bound)` on random small instances.
- **Regression:** pin the `d=32, r=1, c=16384` knee so `max_secure_width` completes in
  bounded time and matches reference width ±1.
- **Parent tests unchanged** until S5b: `akita-types` tests on `main` keep passing on this
  branch when only the estimator spec is present (no partial table renames).

### Performance

- Per `estimate_bits` call: target **<1 ms** median on reference hardware (vs Sage path
  which is fast except at hangs).
- Full table regen: embarrassingly parallel over `(family, d)`; default concurrency capped
  at `min(6, num_cpus)` to avoid memory spikes.
- No proof-size or prover runtime change from this crate alone (offline only).

## Design

### Architecture

```text
akita-sis-estimator (lib)
  ├── params.rs          # n, q, m, length_bound; trivial-easy check
  ├── euclidean.rs       # cost_euclidean port (delta, beta, predicate, rop)
  ├── reduction.rs       # BDGL16 asymptotic + delta(beta) piecewise table
  └── search.rs          # max_secure_width monotone search

gen_sis_table (bin)      # CLI mirrors scripts/gen_sis_table.py flags
  └── emit rust | csv

akita-types/generated_sis_table.rs   # consumer (S5b, not this PR)
```

**Lattice-estimator surface actually used** (everything else ignored):

```text
SIS.lattice(SIS.Parameters(..., norm=2), red_cost_model=BDGL16)
  → SISLattice.cost_euclidean
      → _opt_sis_d, _solve_for_delta_euclidean
      → ReductionCost.beta(delta)   # can hang: _beta_find_root / _beta_simple
      → log-space length-bound predicate
      → BDGL16._asymptotic(beta, d)
  → log2(rop)
```

Reference commit for golden generation must be recorded in test data README
(e.g. `malb/lattice-estimator` at tag or SHA used during spec approval).

### Alternatives Considered

| Alternative | Why not |
|-------------|---------|
| Fix Python + `cysignals.alarm` only | Still requires Sage in dev/CI; hangs remain possible; poor parallelism. |
| Subprocess one Sage call per cell | Isolation helps crashes, still slow and operationally heavy. |
| Full lattice-estimator Rust port | Months of scope; Akita needs one function family. |
| Different cost model (e.g. ADPS16) | Breaks continuity with existing audited table and parent spec. |

**Chosen:** narrow Rust port with golden parity to lattice-estimator on the Euclidean path.

## Documentation

- `crates/akita-sis-estimator/README.md`: regen commands, golden refresh procedure,
  equivalence statement, pinned lattice-estimator revision.
- Update `specs/l2-msis-opnorm-folded-witness.md` Open Question 1 → this spec (done in
  the companion commit on the same branch).
- `AGENTS.md` or `CONTRIBUTING.md`: one line pointing offline SIS regen to `cargo run -p
  akita-sis-estimator --bin gen_sis_table` (implementation PR).

## Execution

### Slice ordering (this branch family)

| Slice | Deliverable | Depends on |
|-------|-------------|------------|
| **Spec** (this PR) | `sis-euclidean-estimator.md` + parent cross-links | — |
| **S5a** | `akita-sis-estimator` + golden tests + `gen_sis_table` | Spec approved |
| **S5b** | L2 collision rename, regen table, wire `norm_bound` A-role | S5a |
| **S3** | `operator_norm_threshold`, rejection, descriptor | S1 (done); **not** `(31,11),T=16` until S2 |
| **S11** | `gen_schedule_tables` regen + drift | S5b, S6 parameterization |

### Implementation notes (S5a)

- Port `cost_euclidean` and `BDGL16._asymptotic` using `f64` / `log2`; keep the log-space
  `min(A², B²)` branch for overflow safety.
- Replace `ReductionCost.beta` with a capped `_beta_simple`-style loop (hard max β, max
  iterations) to eliminate Sage hangs.
- `@cached_function` behavior is unnecessary; optional memoization per process only.
- Binary flags: `--family`, `--d`, `--collision`, `--max-rank`, `--target-bits`,
  `--search-cap`, `--format {rust,csv}`, `--jobs`.

### Risks

- Bit-level mismatch at `beta(δ)` knees: mitigate with golden grid + conservative timeout policy.
- Full regen wall time: mitigate with parallelism and ladder truncation (all-zero bucket stops
  per-`d` sweep, same as Python).

## References

- Parent: [`specs/l2-msis-opnorm-folded-witness.md`](l2-msis-opnorm-folded-witness.md)
- [`scripts/gen_sis_table.py`](../scripts/gen_sis_table.py) (current generator)
- [`crates/akita-types/src/sis/ajtai_key.rs`](../crates/akita-types/src/sis/ajtai_key.rs)
- lattice-estimator: `estimator/sis_lattice.py` (`cost_euclidean`),
  `estimator/reduction.py` (`BDGL16`, `beta`, `delta`)
- [`specs/SPEC_REVIEW.md`](SPEC_REVIEW.md) (review workflow)
