# Spec: Euclidean SIS Table Regen via lattice-estimator (S5a + S5b)

| Field       | Value |
|-------------|-------|
| Author(s)   | Quang Dao, Cursor agent draft |
| Created     | 2026-06-05 |
| Revised     | 2026-06-08 (S5a + S5b done in #155; derived `d·B²` keys) |
| Status      | proposed |
| PR          | [#155](https://github.com/LayerZero-Labs/akita/pull/155) (branch `quang/s3-s5-sis-estimator-spec`) |

## Summary

Akita's L2 MSIS table regen (archived parent:
[`specs/archive/2026-Q2/l2-msis-opnorm-folded-witness.md`](archive/2026-Q2/l2-msis-opnorm-folded-witness.md);
the folded-witness L2 certificate was cancelled and removed in #247 on main)
needs regenerated `generated_sis_table/` rows: for each representative modulus family,
ring dimension, squared-Euclidean collision bucket, and module rank, the maximum commitment
width that still yields at least 128-bit security under a fixed lattice-reduction cost model.

Today that table is produced offline by `scripts/gen_sis_table.py`, which calls
[lattice-estimator](https://github.com/malb/lattice-estimator) through SageMath from an
implicit external checkout (`--estimator-path`, `LATTICE_ESTIMATOR_PATH`, or a sibling
`../lattice-estimator` directory).
That path is fragile: the checkout is unpinned, degenerate Euclidean instances can hang,
parallel sweeps can crash under Sage + `fork`, and the bucket ladder in the generator can
drift from the checked-in Rust table.

**Strategy (revised):** land general reliability fixes in lattice-estimator upstream,
**vendor the open upstream PR branch directly** as `third_party/lattice-estimator` until
`malb/lattice-estimator` merges the fix, then repoint the submodule URL to `malb` and bump
the SHA. Keep `scripts/gen_sis_table.py` as the canonical offline regen driver.
No in-repo Rust reimplementation of `cost_euclidean`.
Akita-specific golden data and width-search regression live **only in this repo**.

The estimator remains an **offline build tool**, not a runtime prover/verifier dependency.

## Intent

### Goal

1. **Upstream (lattice-estimator):** land a small number of general-purpose fixes so
   Euclidean SIS estimation is reliable, bounded-time, and safe to run in parallel sweeps.
   Upstream PRs must stand on their own merit (correctness, robustness, docs); they must
   not mention Akita, carry Akita-only tests, or exist solely to unblock our table regen.
2. **Akita (S5a, #155):** pin `third_party/lattice-estimator` at the open
   lattice-estimator reliability PR branch commit, harden `scripts/gen_sis_table.py`
   (provenance header, `--jobs`, no `SIGALRM` hang guard), and check in Akita-local golden
   CSV plus a regen/check script. *(Done.)*
3. **Akita (S5b, same PR #155):** stitch emitted rows into
   `crates/akita-types/src/sis/generated_sis_table/`, rename `collision_inf` →
   `collision_l2_sq` (`u128`), and wire L2 A-role / B/D pricing from `norm_bound.rs`
   (Lemma 7 on fold response `z`; see parent spec). The stitched table carries two
   complementary key families: **derived** keys `K = d · B²` for coefficient-`L∞`
   buckets `B` (`COEFF_LINF_BUCKETS`, default for norm-bound envelopes via
   `collision_l2_sq_for_linf_envelope`), plus the **power-of-two** squared-collision
   ladder (`ceil_supported_collision` fallback). Drop only the old L∞ **estimator**
   table keys, not the `8·ω·fold_witness_beta·ν` collision formula.

**S5a + S5b** in #155 deliver the offline estimator pin and the runtime L2 SIS floor cutover.
**S11** (shipped schedule regen + drift under L2 pricing) remains a follow-up after S6.

### Invariants

- **Pinned reference, not ambient checkout.** Regen uses `third_party/lattice-estimator` at
  a recorded submodule commit on the vendored lattice-estimator PR branch until upstream
  merges; then the submodule URL moves to `malb/lattice-estimator` at the merge commit.
  Generated table headers and golden metadata record remote URL + SHA.
  Normal Rust CI and prover/verifier builds do **not** require Sage or an initialized
  submodule.
- **Single source of truth.** Security bits come from lattice-estimator
  `SIS.lattice(..., norm=2, red_cost_model=BDGL16)` on the `cost_euclidean` path.
  Akita does not maintain a parallel numeric core.
- **Bounded-time estimation.** The vendored lattice-estimator fix completes Euclidean SIS
  calls in bounded time (including degenerate `δ → 1` knees). `scripts/gen_sis_table.py`
  maps `rop = ∞` to `+∞` security bits and does not use a wall-clock `SIGALRM` guard.
- **Monotone width search.** For fixed `(n, q, collision_l2_sq, rank)`, security bits are
  non-increasing in commitment width `w` outside a possible degenerate `-inf` prefix at very
  small `w`. The hybrid search in `gen_sis_table.py` assumes this; Akita golden checks
  verify it on the committed grid.
- **Ladder truncation ordering.** Per-`d` all-zero bucket truncation applies only to the
  monotonic power-of-two family sweep (default `family_entries`). Custom `--collisions`
  lists (derived-key supplement and one-off regen) skip dimension-wide truncation so a
  single failed cell does not drop later rows. Pow2 sweeps still sort by `(d, collision)`
  ascending before truncating.
- **Derived-key lockstep.** `COEFF_LINF_BUCKETS` in `ajtai_key.rs` and
  `scripts/gen_sis_table.py` must match. Derived rows are merged by
  `scripts/stitch_generated_sis_table.py` (`--supplement-derived-only` or full stitch).
- **No runtime coupling.** Prover, verifier, and planner read only `generated_sis_table/`.
- **Single cost model.** Euclidean (`norm = 2`) + `BDGL16` only. The `lgsa` shape model is
  **inert** on this path: `SIS.lattice` with `norm == 2` calls `cost_euclidean`, which
  does not take `red_shape_model`. S5b should correct misleading "lgsa" comments in table
  headers when regenerating.
- **Parameter mapping** (unchanged from L2 parent spec):

  | Akita | Estimator |
  |-------|-----------|
  | module rank `r` | `n = r · d` |
  | ring dimension `d` | folded into `n`, `m` |
  | width `w` (ring elements) | `m = w · d` |
  | per-row `collision_l2_sq` | `length_bound = sqrt(w · collision_l2_sq)` |
  | `SisModulusFamily` | representative `q` (Q32/Q64/Q128) |

- **Bucket ladder lockstep (S5b).** Power-of-two squared-collision buckets
  (`MIN_LOG_BUCKET = 1`, `MAX_LOG_BUCKET = 84`) and `COEFF_LINF_BUCKETS` live in
  `scripts/gen_sis_table.py` and `ajtai_key.rs`. Runtime lookup prefers derived
  `d · ceil(linf)²` when tabulated, else `ceil_supported_collision` on raw `d · linf²`.

### Non-Goals

- In-repo Rust port of `cost_euclidean` / `BDGL16` (`akita-sis-estimator` crate).
- Akita-specific tests, golden grids, or documentation inside lattice-estimator.
- Porting all of lattice-estimator (LWE, NTRU, infinity-norm SIS, full simulators).
- Changing the 128-bit target, BDGL16 model, or representative moduli without a spec
  amendment.
- Runtime on-demand SIS estimation in the planner DP.
- Replacing the parent L2 protocol spec or implementing S6–S12 in this slice.
- **S11** shipped-schedule regen (depends on S5b tables plus S6 proof shape).

## Evaluation

### Acceptance Criteria (S5a)

- [x] lattice-estimator reliability PR merged upstream
  ([malb/lattice-estimator#213](https://github.com/malb/lattice-estimator/pull/213));
  Akita submodule repointed to `malb/lattice-estimator@main` at merge commit `27a581b`.
- [x] `.gitmodules` records `third_party/lattice-estimator` at the vendored PR-branch
  commit; SHA also recorded in golden metadata and generated table provenance headers.
- [x] `scripts/gen_sis_table.py` hardened: deterministic output order, `--jobs`
  subprocess shards, provenance header with submodule SHA, `SIGALRM` removed.
- [x] Akita golden CSV checked in under `scripts/sis_golden/` (≥50 cells: three families,
  `d ∈ {32,64,128,256}`, ranks `{1,5,20}`, buckets including known degenerate knees).
  Metadata records submodule SHA used to produce it.
- [x] `scripts/sis_golden/check.py` (or equivalent) replays golden cells through the pinned
  submodule + generator and fails on drift. Manual/Sage workflow documented; **not** on the
  normal Rust CI path unless we add a hash-only gate on committed golden bytes.
- [x] Full canonical regen completes in bounded wall time on a 16-core machine using
  `--jobs` (family × `d` shards). Target: overnight acceptable; no hung cells.

### Acceptance Criteria (S5b, in #155)

- [x] Stitched `generated_sis_table/` unions power-of-two `collision_l2_sq` keys
  (`2^MIN_LOG_BUCKET .. 2^MAX_LOG_BUCKET`) with derived keys `K = d · B²` for each
  `B ∈ COEFF_LINF_BUCKETS` and `d ∈ {32,64,128,256}`; provenance notes BDGL16 Euclidean
  (no `lgsa`); ranks `1..=20` via `scripts/stitch_generated_sis_table.py`.
- [x] `collision_inf` renamed to `collision_l2_sq` on `AjtaiKeyParams`, `min_secure_rank`,
  `ceil_supported_collision`, and descriptor bytes (`u128` LE).
- [x] A/B/D norm-bound pricing routes through `collision_l2_sq_for_linf_envelope`
  (derived `d · ceil(linf)²` when tabulated, else pow2 ceil on `d · linf²`). A-role
  Lemma 7 input is `8·ω·fold_witness_beta·ν`; B/D use `2^lb − 1`. No production path
  drops the `8` or `ν` factors.
- [x] Derived keys restore main-equivalent SIS widths at norm-bound collisions (e.g.
  `fp32_d128_onehot` nv=28: rank-10 width 32768 at `(128, 128·1048575²)` vs 32767 on the
  adjacent pow2 bucket that `ceil_supported_collision` alone would select).
- [x] `cargo test -p akita-types` passes; shipped planner tables regened via
  `gen_schedule_tables` so `assert_schedule_stays_within_audited_sis_widths` and
  `generated_schedule_tables_match_find_schedule` pass under L2 floors.

### ZK vs non-ZK schedule tables (S5b)

Shipped schedules are emitted in **two** passes: plain (`gen_schedule_tables`) and
`--features zk` (`*_zk.rs`). `PlannerPolicy` and SIS collision lookup are identical;
the DP objective differs because `level_proof_bytes` and recursive witness widths
include ZK mask/blinding bytes under the `zk` feature.

Structural divergence at the same `(preset, num_vars, incidence)` key is therefore
**expected**, not table corruption. Example (`fp128_d128_full`, nv=30 singleton): non-zk
tail `…(m=9,r=3)→(m=8,r=3)→(m=8,r=3)`; zk tail `…(m=9,r=3)→(m=9,r=3)→(m=8,r=3)` (same
fold count, different byte-optimal geometry).

Guards:
- `generated_schedule_tables_match_find_schedule` runs in **both** CI passes (non-zk
  and all-features) and compares each shipped table to `find_schedule` compiled under the
  same features.
- `generated_families_stay_within_audited_sis_widths` (zk only) spot-checks every shipped
  family at `num_vars ∈ {8,16,28,30}` against `min_secure_rank` using stored
  `collision_l2_sq` keys (no pow2 re-rounding).

Regen both passes after SIS or proof-size changes; never expect `*_zk.rs` to mirror the
plain table row-for-row.

### Testing Strategy

- **Akita golden (committed):** CSV of `(q, d, collision_l2_sq, rank, width, log2_rop,
  max_width)` for the representative grid. Refresh after submodule bump; review diff in PR.
- **Monotonicity:** On golden refresh, assert width-search monotonicity (hybrid search vs
  brute force on golden cells, plus a bounded window around each returned `max_width`).
- **Degenerate knees:** Golden includes cells where `δ` is just above 1; every cell must
  finish in bounded time and match committed bits ±0.01 (exact `-inf` where documented).
- **S5b is in #155:** the branch intentionally breaks the old L∞ table and renames
  collision fields; golden CSV stays on the offline grid, not the full stitched table.

### Performance

- Per-cell `SIS.lattice` call: typically ≪1 s once upstream hang is fixed; large-`m` probes
  dominate width search.
- Parallelism: `--jobs` subprocess shards over `(family, d)`; cap default
  `min(6, num_cpus)` to limit memory.
- No proof-size or prover runtime change from S5a alone.

## Design

### Architecture

```text
third_party/lattice-estimator/     pinned submodule (LE PR branch until malb merge)

scripts/gen_sis_table.py           canonical regen driver (Sage)
scripts/sis_golden/
  golden.csv                       committed reference grid (Akita-only)
  check.py                         replay + drift gate (manual / optional CI hash)

crates/akita-types/src/sis/
  generated_sis_table/             split consumer modules (S5b stitch)
```

**Estimator surface used** (everything else ignored):

```text
SIS.lattice(SIS.Parameters(..., norm=2), red_cost_model=BDGL16)
  → SISLattice.cost_euclidean
      → _opt_sis_d, _solve_for_delta_euclidean
      → ReductionCost.beta(delta)   # _beta_find_root today; see upstream fix
      → length-bound predicate (lb)
      → BDGL16._asymptotic(beta, d)
  → log2(rop)
```

### Known upstream defects (motivation for general fixes)

These are bugs or robustness gaps in lattice-estimator itself, observed while running
large-parameter Euclidean SIS sweeps. Fixes should be justified in those terms.

1. **`_beta_find_root` bracket failure → `_beta_simple` divergence.**
   When `find_root` on `[40, 2^16]` fails because `δ < _delta(2^16)` (root-Hermite factor
   in the `δ → 1` regime), the fallback `_beta_simple` loop does not terminate.
   Reproducible on small Euclidean instances with moderate `length_bound`, not specific to
   any one consumer.

2. **`cost_euclidean` length-bound overflow.**
   Computing `sqrt(d) * q^(n/d)` overflows for large `q` and moderate `n/d`. The mathematically
   equivalent `min(A, B)` comparison should be done in log space (same class of failures as
   large-`q` MPFR overflows elsewhere in the estimator).

3. **Sage + `fork` multiprocessing.**
   `batch_estimate(..., jobs>1)` uses default `fork` on Linux; combined with Sage/cysignals
   this is unsafe and can crash parallel sweeps. Using `spawn` (or documenting subprocess
   isolation) is a general fix for parallel estimation.

4. **Misleading `red_shape_model` on Euclidean path.**
   Docs and call sites pass `red_shape_model="lgsa"` even though `norm=2` ignores it.
   Clarify in upstream docs; optional runtime warning is upstream-only.

### `cost_euclidean` contract (reference for golden refresh)

For `params = (n, q, m, length_bound, norm=2)`:

1. Trivial-easy: `length_bound ≥ (q−1)/2` → `ValueError`; generator maps to `-inf` bits.
2. `d_lat = min(floor(_opt_sis_d(params)), m)`.
3. `δ = _solve_for_delta_euclidean(params, d_lat)`.
4. If `δ ≥ 1` and `β(δ) ≤ d_lat`: reduction possible; else `β = d_lat`, not possible.
5. Length-bound predicate: attack succeeds iff `length_bound > lb` and reduction possible.
6. `rop = BDGL16._asymptotic(β, d_lat)` when predicate true; else `rop = ∞`.

After upstream fix (1), step 4 must complete in bounded time for all `δ` in the model range.

### Alternatives Considered

| Alternative | Why not |
|-------------|---------|
| Rust port (`akita-sis-estimator`) | Duplicates upstream logic; two sources of drift; large review surface. |
| Ambient checkout only | Unpinned; silent drift. |
| `SIGALRM` timeout forever | Machine-dependent; masks upstream bug; no-op without `SIGALRM`. |
| Akita-only patches in submodule | Fork maintenance; upstream fixes benefit all users. |
| Akita-specific upstream PRs | Review burden on malb for non-general changes; rejected by policy. |

**Chosen:** general upstream fixes + pinned submodule + Akita golden in `scripts/sis_golden/`.

## Documentation

- Regen runbook: submodule init, `sage -python scripts/gen_sis_table.py` flags, `--jobs`,
  golden refresh command, submodule bump procedure.
- Update parent spec cross-links (S5a description, Open Question 1).
- `AGENTS.md` / `CONTRIBUTING.md`: one line pointing offline SIS regen to
  `scripts/gen_sis_table.py` with pinned submodule (implementation PR).

## Execution

### PR plan (minimal count)

**Principle:** At most **two** upstream PRs to `malb/lattice-estimator`, each a general
improvement with upstream-native tests and release notes. **Zero** upstream PRs that mention
Akita or exist only for our table. All Akita-specific verification stays in this repo.

#### Upstream PR 1 — Euclidean SIS reliability (single PR)

**Title (example):** `sis: fix cost_euclidean hangs and large-q overflow`

**Scope (all general):**

| File | Change |
|------|--------|
| `estimator/reduction.py` | On `_beta_find_root` bracket failure, do not call unbounded `_beta_simple`. Return a documented sentinel or raise a typed error; bounded time guaranteed. |
| `estimator/sis_lattice.py` | Log-space `min(A², B²)` for the length-bound predicate; handle degenerate `β` from PR logic without hanging. |
| `estimator/sis_lattice.py` + docs | Document Euclidean path ignores `red_shape_model`; optional one-time warning if passed. |
| `estimator/tests/` | **General** regressions only: e.g. bracket-failure instance completes in <1 s; large-`q` Falcon-class parameter yields finite `rop`; existing doctests preserved. |

**Not in scope:** Akita moduli, Akita ranks, Akita bucket ladder, filenames referencing Akita.

**Review narrative:** Fixes infinite loop in reduction cost inversion; fixes float overflow
in Euclidean SIS length bound (cf. existing large-`q` overflow issues elsewhere).

#### Upstream PR 2 — Parallel estimation safety (optional follow-up, or fold into PR 1 if small)

**Title (example):** `util: use spawn context for multiprocessing pool`

**Scope:** `estimator/util.py` (`batch_estimate`), optionally `param_sweep.py`: default
`get_context("spawn")` when `jobs > 1`; README note on Sage + fork.

Defer if PR 1 is already large; Akita `--jobs` uses subprocess sharding and does not depend
on this for correctness.

#### Akita PR #155 — spec + S3 + S5a + S5b (single PR)

- Spec revision (this file) and parent cross-links.
- S3 op-norm rejection: **removed** in `remove/op-norm-rejection` (was #207).
- S5a *(done):* vendored LE PR branch, hardened `gen_sis_table.py`, golden grid.
- S5b *(done):* L2 `generated_sis_table/` stitch (pow2 ladder + derived `d·B²` keys),
  `collision_l2_sq` rename, `collision_l2_sq_for_linf_envelope` rank pricing, planner
  schedule regen. Submodule pinned to `malb/lattice-estimator` @ `27a581b`; re-run
  `scripts/sis_golden/check.py` in a Sage env when bumping the pin. Parallel regen shards
  pass `--dims`/`--collisions` per work item so derived rows are not silently dropped.

#### Follow-up — S11 (separate PR)

Shipped schedule regen + drift guards under L2 tables (needs S6 proof shape).

### Slice ordering

| Slice | Deliverable | Depends on |
|-------|-------------|------------|
| **PR #155** | spec + S3 + S5a + S5b | LE PR 1 branch exists |
| **LE PR 1** | upstream reliability ([malb#213](https://github.com/malb/lattice-estimator/pull/213)) | — |
| **LE PR 2** | spawn pool (optional) | — |
| **S11** | schedule regen + drift | S5b, S6 |

### Submodule pin

**Current (malb main, post-#213 merge):**

| Field | Value |
|-------|-------|
| Remote | `https://github.com/malb/lattice-estimator.git` |
| Branch | `main` |
| Commit | `27a581bb8e9d49f5e9e2db315bd48ac769d5f5f5` |
| Upstream PR | [malb/lattice-estimator#213](https://github.com/malb/lattice-estimator/pull/213) (merged 2026-06-06) |

Historical vendored PR branch: `quangvdao/lattice-estimator@fix/sis-euclidean-reliability`
at `85110c8010aaace222e4c57ff5bd9c611bdb36c1` (ancestor of the merge commit).

Historical baseline before reliability work: `2bfb768`.

### Risks

- **Submodule repoint on merge:** When malb#213 lands, bump URL + SHA and refresh golden.
- **Numeric drift at knees:** Golden CSV catches drift on bump; review diffs when moving
  from vendored branch to malb `main`.
- **Regen wall time:** Subprocess `--jobs` + ladder truncation; no Rust speedup.

## References

- Parent (archived): [`specs/archive/2026-Q2/l2-msis-opnorm-folded-witness.md`](archive/2026-Q2/l2-msis-opnorm-folded-witness.md)
- [`scripts/gen_sis_table.py`](../scripts/gen_sis_table.py)
- [`crates/akita-types/src/sis/ajtai_key.rs`](../crates/akita-types/src/sis/ajtai_key.rs)
- lattice-estimator: `malb/lattice-estimator` — `estimator/sis_lattice.py`
  (`cost_euclidean`), `estimator/reduction.py` (`BDGL16`, `beta`, `_delta`)
- [`specs/SPEC_REVIEW.md`](SPEC_REVIEW.md)
