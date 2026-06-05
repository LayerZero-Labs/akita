# Spec: Rust Euclidean SIS Estimator (`akita-sis-estimator`)

| Field       | Value |
|-------------|-------|
| Author(s)   | Quang Dao, Cursor agent draft |
| Created     | 2026-06-05 |
| Status      | proposed |
| PR          | [#155](https://github.com/LayerZero-Labs/akita/pull/155) (branch `quang/s3-s5-sis-estimator-spec`) |

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
   `cost_euclidean` code path (see Design for the exact contract).
2. Exposes `max_secure_width(rank, d, collision_l2_sq, q, target_bits, search_cap) -> u64`
   via the same monotone width search the Python script uses today.
3. **Emits** Rust match arms (or CSV) to stdout or an artifact file under the crate
   (`crates/akita-sis-estimator/`); it does **not** write `generated_sis_table.rs` in this
   slice. Stitching the emitted rows into `crates/akita-types/src/sis/generated_sis_table.rs`
   (power-of-two buckets `2^MIN_LOG_BUCKET .. 2^MAX_LOG_BUCKET`) is the S5b deliverable
   (see "Bucket ladder lockstep").
4. Records provenance in generated headers (pinned estimator revision + patch, crate version,
   target bits, search caps, family moduli).

Parent spec slices **S5a** (this spec) unblocks **S5b** (table stitch + `collision_l2_sq`
rename cutover) and **S11** (schedule table regen under L2 pricing).

### Invariants

- **Reference equivalence.** On a pinned golden grid of `(n, q, m, length_bound)` instances,
  `log2(rop)` from the Rust core matches the pinned lattice-estimator reference (see
  References) as follows: the block size `β(δ)` is reproduced as an **exact integer** on every
  non-degenerate cell (same monotone root, same `ceil(root − 1e-8)` convention as
  `ReductionCost._beta_find_root`), so the only residual difference is `f64` rounding inside
  `BDGL16._asymptotic` / `log2`, bounded at **±0.01 bits**. Degenerate cells (see
  "Deterministic conservative ceiling") are reported as `-inf` on both sides and compared for
  exact equality, not within the ±0.01 band.
- **Deterministic conservative ceiling.** The `β(δ)` inversion is bounded to lattice-estimator's
  own bracket `β ∈ [40, β_max]`, `β_max = 2^16`. Its default `ReductionCost.beta` is
  `_beta_find_root`, which on a bracket failure falls back to the **unbounded** `_beta_simple`
  loop; that loop diverges precisely when the required root lies below the bracket, i.e. when
  `δ < _delta(β_max) ≈ 1.00006` (the degenerate `δ → 1` regime; e.g. the `d=32, r=1,
  length_bound=256` knee where `_solve_for_delta_euclidean` returns `δ = 1.0`). The Rust core
  detects this condition directly and returns `-inf` security bits — deterministically, with
  **no wall-clock budget**. This fires on exactly the cells the Sage generator currently guards
  with its `SIGALRM` timeout, so it reproduces the generator's `-inf` policy machine-
  independently. It can only **under**-report secure width, never over-report it (a one-width
  effect at the observed knees), and is the single deliberate deviation from a hypothetical
  never-diverging estimator; on every cell lattice-estimator can actually evaluate, parity is
  exact (see "Reference equivalence").
- **Monotone search.** For fixed `(n, q, collision_l2_sq, rank)`, security bits are
  non-increasing in commitment width `w` outside a possible degenerate `-inf` prefix at very
  small `w`; the width search assumes this and Testing Strategy verifies it (a violation is a
  blocker).
- **No runtime coupling.** Prover, verifier, and planner continue to read only
  `generated_sis_table.rs`; they do not invoke the estimator at proof time.
- **Single cost model.** Euclidean (`norm = 2`) + `BDGL16` asymptotic reduction cost only.
  The `lgsa` shape model is **not** part of this estimator: verified against the pinned
  reference, `SIS.lattice` calls `cost_euclidean(params, red_cost_model, log_level)` for
  `norm == 2` and never forwards `red_shape_model` (`estimator/sis_lattice.py`, the
  `tag == "euclidean"` branch), and `cost_euclidean` takes no shape-model parameter. The
  `red_shape_model="lgsa"` argument in `scripts/gen_sis_table.py` and the "Shape model: lgsa"
  lines in the current `generated_sis_table.rs` header are therefore **inert** on this path;
  S5b should correct those comments when it regenerates the table.
- **Parameter mapping unchanged** from the L2 parent spec and the updated Python script:

  | Akita | Estimator |
  |-------|-----------|
  | module rank `r` | `n = r · d` |
  | ring dimension `d` | folded into `n`, `m` |
  | width `w` (ring elements) | `m = w · d` |
  | per-row `collision_l2_sq` | `length_bound = sqrt(w · collision_l2_sq)` |
  | `SisModulusFamily` | representative `q` (Q32/Q64/Q128) |

- **Bucket ladder lockstep.** The power-of-two squared-collision ladder and the
  `next_power_of_two` rounding currently live **only** in `scripts/gen_sis_table.py`
  (`MIN_LOG_BUCKET = 1`, `MAX_LOG_BUCKET = 84`). On `main`, `crates/akita-types/src/sis/ajtai_key.rs`
  still uses the legacy `2^k − 1` buckets and a `collision_inf` field, so this estimator only
  **emits** power-of-two-keyed rows. Introducing `MIN_LOG_BUCKET` / `MAX_LOG_BUCKET` +
  `u128::next_power_of_two` rounding into `ceil_supported_collision`, and keeping the two
  ladders in lockstep, is the **S5b** deliverable (see Goal 3).

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
  known degenerate knees (`q32`, `d=32`, `r=1`, `collision_l2_sq=16384`, sampling widths
  `3..6`, where small widths hit the degenerate `δ → 1` regime and must come back as
  deterministic `-inf`).
- [ ] The golden export is regenerated against the **pinned** lattice-estimator revision
  recorded in the spec/README (upstream SHA **+** the log-space `lb` patch; see References),
  with `ReductionCost.beta` bounded to `β_max = 2^16` (no `SIGALRM`), so it is byte-
  reproducible on any machine. The pin is a hard gate: a golden generated against an
  unrecorded revision is not accepted.
- [ ] `cargo test -p akita-sis-estimator` passes golden comparison against the pinned
  lattice-estimator export (documented command in crate README).
- [ ] Full regen completes for all `(family, d)` arms in <2h wall time on a 16-core laptop
  (parallelism via `rayon`; no Sage dependency).
- [ ] Stitched `generated_sis_table.rs` uses only power-of-two keys and updated provenance
  header; `cargo test -p akita-types` and `assert_schedule_stays_within_audited_sis_widths`
  pass after S5b lands.
- [ ] `scripts/gen_sis_table.py` is either removed or reduced to a thin wrapper that
  delegates to the Rust binary (decision at implementation time; must not be the only
  regen path after S5b).

### Testing Strategy

- **Golden tests:** `sis_euclidean_security_bits` vs the pinned lattice-estimator export
  (exact `-inf` on degenerate cells; ≤ ±0.01 bits elsewhere).
- **Monotonicity (required, not optional):** `max_secure_width` assumes security bits are
  monotone non-increasing in `w` for fixed `(n, q, collision_l2_sq, rank)`. Because
  `d = min(floor(_opt_sis_d), m)` introduces a kink (and the degenerate `-inf` ceiling can
  create a non-monotone hole at very small `w`), assert that the hybrid width search agrees
  with a brute-force linear scan on the full golden grid, and that no cell violates
  monotonicity outside a documented degenerate prefix. A counterexample is a blocker.
- **Regression:** pin the `d=32, r=1, c=16384` knee so `max_secure_width` completes in
  deterministic bounded time (no wall-clock budget) and matches the reference width ±1.
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
      → ReductionCost.beta(delta) = _beta_find_root  # _beta_simple fallback diverges at δ→1
      → log-space length-bound predicate
      → BDGL16._asymptotic(beta, d)
  → log2(rop)
```

### `cost_euclidean` contract (what the Rust core reproduces)

For `params = (n = rank·d, q, m = width·d, length_bound, norm = 2)`:

1. **Trivial-easy guard.** If `length_bound ≥ (q − 1) / 2`, lattice-estimator raises
   `ValueError("SIS trivially easy")`; the core returns `-inf` (insecure), matching the
   generator's `except ValueError` branch.
2. **SIS dimension.** `d_lat = min(floor(_opt_sis_d(params)), m)` — `m` enters the estimate
   only as a cap on the optimized lattice dimension.
3. **Root-Hermite factor.** `δ = _solve_for_delta_euclidean(params, d_lat)` (closed form).
4. **Block size.** If `δ ≥ 1` and `β(δ) ≤ d_lat`, set `β = β(δ)`, `reduction_possible = true`;
   else `β = d_lat`, `reduction_possible = false`. `β(δ)` reproduces `_beta_find_root` as a
   bounded, deterministic monotone bisection on `_delta` over `[40, 2^16]` returning
   `ceil(root − 1e-8)`, with the `_delta(40) < δ → 40` short-circuit. If `δ < _delta(2^16)`
   the root is unbracketable (`_beta_simple` would diverge) → return `-inf` (see
   "Deterministic conservative ceiling").
5. **Length-bound predicate.** Compute `lb` via the log-space `min(A², B²)` branch
   (`A² = n·ln q`, `B² = d_lat·q^(2n/d_lat)`) for overflow safety; the attack succeeds iff
   `length_bound > lb AND reduction_possible`. When the predicate is **false**,
   `cost(..., predicate=False)` sets `rop = ∞`, i.e. the instance is **secure** (`+inf` bits)
   — distinct from the degenerate `-inf` ceiling in step 4.
6. **Cost.** `rop = BDGL16._asymptotic(β, d_lat) = LLL(d_lat) + 2^(0.292·β + 16.4 +
   log2 svp_repeat(β, d_lat))`; return `log2(rop)`.

`_delta(β)` is the standard root-Hermite factor: a fixed small-β table for `β ≤ 40` and the
closed form `(β/(2πe)·(πβ)^{1/β})^{1/(2(β−1))}` for `β > 40`.

The pinned reference is **`malb/lattice-estimator` at `2bfb768`
(`2bfb7682e73e814f31720d7c1d71f4367fa80712`) plus the local log-space `lb` reformulation
patch** to `cost_euclidean` (committed as a patch file or vendored alongside the golden data;
the unrelated Gaussian-sampling additions are not part of the Euclidean path). Pinning the
upstream SHA alone is insufficient — the actual reference is the patched estimator. Record
both in the crate README and the generated-table header.

### Alternatives Considered

| Alternative | Why not |
|-------------|---------|
| Fix Python + `cysignals.alarm` only | Still requires Sage in dev/CI; hangs remain possible; poor parallelism. |
| Subprocess one Sage call per cell | Isolation helps crashes, still slow and operationally heavy. |
| Full lattice-estimator Rust port | Months of scope; Akita needs one function family. |
| Different cost model (e.g. ADPS16) | Breaks continuity with existing audited table and parent spec. |

**Chosen:** narrow Rust port with golden parity to lattice-estimator on the Euclidean path.

## Documentation

- `crates/akita-sis-estimator/README.md`: regen commands, golden refresh procedure (including
  the `RC.beta` → bounded monkeypatch used to make the Sage export deterministic), equivalence
  statement, and the pinned lattice-estimator revision **plus** the log-space `lb` patch.
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
  `min(A², B²)` branch for overflow safety (see the contract above).
- Reproduce `ReductionCost.beta` = `_beta_find_root` (the lattice-estimator default), **not**
  `_beta_simple` (the two disagree by up to 1: `_beta_find_root(δ(500)) = 500` vs
  `_beta_simple = 501`, ≈ 0.292 bits through `_asymptotic`). Implement it as a deterministic,
  bounded monotone bisection on `_delta` over `[40, 2^16]` returning `ceil(root − 1e-8)`; a
  root below the bracket (`δ < _delta(2^16)`) returns `-inf`. This matches the reference
  integer-exactly and eliminates the unbounded `_beta_simple` divergence with no wall-clock
  budget.
- `@cached_function` behavior is unnecessary; optional memoization per process only.
- Binary flags: `--family`, `--d`, `--collision`, `--max-rank`, `--target-bits`,
  `--search-cap`, `--format {rust,csv}`, `--jobs`.

### Risks

- Bit-level mismatch at `β(δ)` knees: resolved by reproducing `_beta_find_root` to integer
  exactness (not `_beta_simple`) plus the deterministic `β_max = 2^16` ceiling; the golden grid
  pins it, and degenerate `δ → 1` cells are exact `-inf` on both sides.
- Reference drift: the estimator tracks a *patched* lattice-estimator; pin the upstream SHA +
  patch and fail the golden gate on an unrecorded revision.
- Full regen wall time: mitigate with parallelism and ladder truncation (all-zero bucket stops
  per-`d` sweep, same as Python).

## References

- Parent: [`specs/l2-msis-opnorm-folded-witness.md`](l2-msis-opnorm-folded-witness.md)
- [`scripts/gen_sis_table.py`](../scripts/gen_sis_table.py) (current generator)
- [`crates/akita-types/src/sis/ajtai_key.rs`](../crates/akita-types/src/sis/ajtai_key.rs)
- lattice-estimator pinned reference: `malb/lattice-estimator` `2bfb768` + local log-space
  `lb` patch — `estimator/sis_lattice.py` (`cost_euclidean`, `_opt_sis_d`,
  `_solve_for_delta_euclidean`), `estimator/reduction.py` (`BDGL16._asymptotic`, `beta` =
  `_beta_find_root`, `_delta`)
- [`specs/SPEC_REVIEW.md`](SPEC_REVIEW.md) (review workflow)
