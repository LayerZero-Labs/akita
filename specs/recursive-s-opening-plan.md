## Batched Setup-S Opening: Plan to Beat `main` on the Verifier

Status: drafted 2026-05-12, rev 3 — **Phase G.0 ran into a negative
result; plan needs a strategy decision before proceeding.** Author:
`feat/tensor-challenges`.

### Original premise (rev 2)

The plan was to finish the fourth-root verifier optimization without new
cryptographic primitives by collapsing the K per-level setup-polynomial
evaluations into a single evaluation via the standard
same-polynomial-different-points batched-MLE-evaluation reduction. The
projection was that this would drop NV=25 CR verifier from ~38 ms to
~10 ms and the default tensor verifier from ~10 ms to under `main`'s
5 ms.

### Phase G.0 negative result

Standalone perf sanity test (`mle_batched_perf_sanity_nv25_dims` in
`crates/akita-types/src/layout/flat_matrix.rs`, since reverted) found
that the naive batched evaluator is **slower** than the per-level loop
across every shape that actually appears at NV=25:

| shape         | rows | cols  |  D | K | per-level avg | batched avg | speedup |
|---------------|-----:|------:|---:|--:|--------------:|------------:|--------:|
| root-shape    |    2 | 16384 | 64 | 5 |      40.85 ms |    58.53 ms |   0.70× |
| deep-shape    |    1 | 16384 | 64 | 5 |      20.78 ms |    29.48 ms |   0.70× |
| deep-K=4      |    1 | 16384 | 64 | 4 |      16.86 ms |    24.82 ms |   0.68× |
| root-K=2      |    2 | 16384 | 64 | 2 |      16.38 ms |    32.61 ms |   0.50× |

Why: per-level `mle` does ~1 mul per coeff-cell (`eq_c * coeffs[coeff]`)
plus a small per-(i,j) and per-row scaling. Batched does K muls per
coeff-cell to compute the combined weight in-place. The K-fold reduction
in *matrix walks* doesn't offset the K-fold increase in *arithmetic per
cell* because the per-cell work isn't memory-bound at these shapes —
each coefficient access is L1-cache-friendly within a `row_data[col]`
slice.

This invalidates the projection. The "fourth-root verifier reduction
via batched-MLE-evaluation" doesn't apply to this codebase's setup
matrix shape and field choice.

### Direction chosen: hybrid per-level stage-1 shape (Phase K)

Strategy chosen after standup: **per-fold-level stage-1 challenge shape**,
chosen deterministically by the planner. Early levels (large NV, large
column stride) can stay on flat challenges so they don't pay the 4ω MSIS
penalty + heavier digit shapes that hurt verifier work; later recursive
levels can switch to tensor where the prover-side savings outweigh the
verifier cost. The exact crossover is empirical and the planner picks
it.

This works because:
- Stage-1 soundness per level is independent of every other level's
  shape, as long as each level individually meets its security floor.
  The composition is sound by the standard sumcheck composition
  argument that already underpins multi-level Hachi proofs.
- `LevelParams::stage1_challenge_shape` is already a per-level field
  that the prover (`crates/akita-prover/src/protocol/quadratic_equation.rs`
  lines 452, 623) and verifier (`crates/akita-verifier/src/stages/stage1.rs`
  line 44) both consume from the schedule. The architecture already
  supports mixed shapes; only the *planner* currently inherits the
  shape from the root and never searches over it per level.

### Phase K: Hybrid stage-1 shape — concrete steps

| Phase | Goal | Cost |
|-------|------|------|
| K.0 | Manual hand-built mixed-shape schedule + E2E prove/verify test, no planner changes. Confirms the architecture supports mixed shapes end-to-end as claimed. | ~half day |
| K.1 | Extend the planner DP search (`derive_root_candidate` and `derive_optimal_suffix_schedule` in `crates/akita-planner/src/schedule_params.rs`) to iterate over `{Flat, Tensor}` × `log_basis` per level. The schedule output remains a `Vec<LevelParams>` with each level's chosen shape recorded. | ~1 day |
| K.2 | Add a `CommitmentConfig::allow_hybrid_stage1_shapes()` opt-in flag (default `false`). When false, planner pins shape to `stage1_challenge_config().shape_hint()` as today. When true, planner searches both. Audit `validate()` on `LevelParams` so per-level shape's mass and digit envelope is still security-checked. | ~half day |
| K.3 | E2E tests covering all 2^L shape combinations at small NV; one bigger test at NV=20 with the planner choosing freely. | ~half day |
| K.4 | Add `HybridCfg<Base>` test/bench wrappers; benches at NV=15, 20, 25 against `main` baseline and against flat-only / tensor-only variants. | ~half day |
| K.5 | Audit transcript labels and Fiat-Shamir absorption to make sure mixed shapes don't accidentally let an attacker pick the shape after seeing challenges. (Schedule is part of the public verifier setup, so this should already be fine, but explicit audit + test.) | ~few hours |
| K.6 | Final apples-to-apples vs `main` at the bench points used in `tensor-everywhere-implementation-plan.md`. | ~half day |

Total: ~3 days. Soundness is preserved by construction since each
level's stage-1 soundness is independent.

## What the data says about where time actually goes

Per-level instrumentation of `verify_setup_claim_reduction` for
`akita/onehot-d64-claim-reduction/nv25/verify` (single thread, 5 levels
including root, fresh runs averaged):

| level | rounds | sumcheck | weight eval | S MLE  | live_rows × live_cols |
|------:|-------:|---------:|------------:|-------:|----------------------:|
|     0 |     21 |    14 µs |      10.4 ms |  8.2 ms | 2 × 16384            |
|     1 |     20 |    14 µs |       1.8 ms |  4.1 ms | 1 × 16384            |
|     2 |     20 |    14 µs |       0.57 ms|  4.1 ms | 1 × 16384            |
|     3 |     20 |    14 µs |       0.39 ms|  4.1 ms | 1 × 16384            |
|     4 |     20 |    13 µs |       0.30 ms|  4.2 ms | 1 × 16384            |
| total |        |    70 µs |       13.5 ms| 24.7 ms |                       |

Total `setup_claim_reduction` work per verify ≈ **38.3 ms**, matching the
observed 41 ms `onehot-d64-claim-reduction/nv25` bench. Almost all of it
is in `mle` (~65%) and `weight eval` (~35%). The sumcheck-round work
itself is negligible.

Critical observation: **`live_cols = 16384` on every level**. The shared
matrix stride does not shrink with depth in Hachi's current
architecture. So the original "cascade makes each successive S smaller"
intuition from the spec doesn't apply to this codebase as it stands —
every level pays the same column-walk cost. The fourth-root win in
*this* repo has to come from amortizing K column walks into one, not
from a per-level shrink.

## Where the per-level setup cost lives today

Both the default tensor path and the CR path bottleneck on the same
column walk of the shared matrix:

- **Default tensor path** (`PreparedMEval::eval_split_at_point`) walks
  `w_d`, `t_b`, `z_dense_setup` and reads `D`/`B`/`A` rows of the shared
  setup matrix. ~5–7 ms per level at NV=25.
- **CR path** replaces the above with a sumcheck, but still finishes by
  evaluating `SetupMatrixPolynomialView::mle(r_row, r_col, r_coeff)`
  inside `verify_setup_claim_reduction`. The `mle` walks the live
  prefix of the shared matrix. ~4–8 ms per level at NV=25 (see table
  above).

So both paths need a way to amortize the per-level walk.

## Approach: batched per-level S evaluation

`S` is **the same shared matrix** across all levels. Each level evaluates
it at a different sumcheck challenge point. The standard
same-polynomial-different-points batching trick reduces the K per-level
evaluations into a **single** evaluation:

- After all levels are processed, collect the K pairs
  `(r^{(L)}, s^{(L)})` produced by each level's CR sumcheck.
- Sample a fresh transcript scalar `γ`.
- Run a small final sumcheck on
  ```text
  sum_{(i,j,k) ∈ shared_dims} S(i,j,k) · ( sum_L γ^L · eq(r^{(L)}; i,j,k) )
    = sum_L γ^L · s^{(L)}
  ```
  The right-hand side is `O(K)` to evaluate; the sumcheck rounds are
  cheap (degree 2, ~50 rounds total at NV=25); the closing oracle is
  one final `S(r*)` evaluation at the sumcheck's random point plus a
  small `O(K · log_total)` weight check.

Net cost per verify (NV=25):
- K-1 saved mle calls × ~4 ms = ~16 ms saved.
- One new sumcheck of ~50 rounds × ~10 µs/round ≈ 0.5 ms added.
- One closing oracle `S(r*)` + `w(r*)` ≈ ~8 ms (single biggest mle, same
  cost as today's level-0 mle).
- Net: **~16 ms saved** on the mle work; CR drops from ~38 ms to ~22 ms.

A symmetric trick on the per-level **weight eval** (today ~13.5 ms at
NV=25 across levels) can be applied via the same Fiat-Shamir slot,
batching the K weight evaluations into one. Total potential save: ~28 ms
out of 38 ms.

This is the actual fourth-root reduction for the codebase as it stands.
No new commitment scheme, no per-level shrink assumption — just the
classic batched-MLE-evaluation reduction applied across the recursive
fold ladder.

## What this is **not**

- Not a new commitment scheme. We do not commit to `S` explicitly. The
  `s^{(L)}` value is just an additional eq-weighted claim discharged by
  the standard batched-MLE-evaluation reduction that already exists in
  `EqWeightedTableVerifier`.
- Not a soundness change at the cryptographic level. The verifier
  performs the same checks; we just merge K identical checks into one
  via Fiat-Shamir randomness. This is a textbook protocol
  transformation, not a cryptographic change.
- Not a per-level cascade into next-level witnesses. After re-reading
  the bench data, "make each level's `S` smaller via cascade" doesn't
  match how this repo's shared matrix is laid out (every level uses
  `live_cols = max_stride`). The right amortization is across levels
  of the same matrix, not down into a recursive sub-matrix.
- Not an asymptotic change for the prover. Prover still computes the K
  per-level claims as before, plus one extra sumcheck. Prover work
  grows by ~O(K · sumcheck_rounds) which is small.

## Risks and unknowns

| Risk | Mitigation |
|------|-----------|
| Batched sumcheck changes proof shape and proof bytes | Add one new field to the top-level proof (one `SumcheckProof<F>`). Bump a proof-version byte. Regen six fp128 generated schedule tables once. |
| Soundness: batching introduces one Fiat-Shamir scalar | Audit transcript labels; the construction is the standard batched-MLE reduction and inherits its existing soundness analysis. Equivalence test against per-level mle gates it. |
| Closing oracle `S(r*)` still costs ~8 ms at NV=25 | This is one level-0-sized mle, irreducible in this architecture without changing the setup matrix layout (deferred). Even at this cost, total CR work drops by ~50%. |
| Default tensor path doesn't use CR sumcheck | Phase H flips CR on as default for production fp128 presets so the default path also benefits from batching. |
| Planner cost model is wrong | Phase I updates the model. The current schedules already work; this only refines optimality. |

## Phases (concrete and incremental)

Each phase ends with: `cargo fmt -q && cargo clippy --all -- -D warnings
&& cargo test --workspace`, then a focused commit.

### Phase G.0 — Reference batched-S evaluator (no proof change yet)

**Goal**: validate the algorithm and quantify the win before touching
the protocol.

- Add a helper `batched_s_eval_reference` in
  `crates/akita-verifier/src/protocol/setup_claim_reduction.rs` that
  takes:
  - `setup`: the expanded setup,
  - `points: &[Vec<F>]`: K opening points (one per level, padded to
    `row_bits + col_bits + coeff_bits` of the shared matrix dimensions),
  - `claims: &[F]`: K corresponding claimed values,
  - `gamma: F`: a batching scalar.

  Computes `combined_weight(i,j,k) = Σ_L γ^L · eq(points[L]; i,j,k)`
  in a precomputation pass, then evaluates
  `Σ_{i,j,k} S(i,j,k) · combined_weight(i,j,k)` in a single column walk.
- Add a test `batched_s_eval_matches_per_level_mle` asserting that the
  result equals `Σ_L γ^L · S.mle(points[L])` for random points and
  random γ.
- Add a micro-benchmark `bench_batched_s_eval_vs_per_level` measuring
  total time for K = 5 evaluations at NV=25 dimensions, comparing
  per-level loop vs batched.

**Acceptance**: test passes; batched evaluator at K=5 over NV=25
dimensions takes ~8–10 ms (matches one level-0 mle), versus the
~25 ms current sum of 5 per-level mle calls.

### Phase G.1 — Batched closing sumcheck for setup-S evaluation

**Goal**: wire the batched evaluation into the protocol via a single
new sumcheck at the end of verification.

- Add `BatchedSetupSumcheckPayload { batched_sumcheck: SumcheckProof<F>
  }` at the top of the proof (one per batched proof, not per level).
- Modify `verify_setup_claim_reduction` to **not** call
  `setup_view.mle` directly. Instead, append `(challenges, claim,
  weight_at_point)` to a per-verification accumulator on a verifier
  context.
- After all levels are processed, the verifier:
  1. Samples `γ` via Fiat-Shamir.
  2. Runs the batched sumcheck (the prover's new
     `batched_sumcheck`) against the accumulated claims:
     `Σ_L γ^L · weight^{(L)}(r^{(L)}) · S(r^{(L)})`.
  3. The sumcheck closes with a single `S(r*) · w(r*)` check where
     `r*` is the sumcheck's challenge and `w(r*) = Σ_L γ^L ·
     eq(r^{(L)}; r*) · weight^{(L)}_eval(r^{(L)})`.
- Add transcript labels: `BATCH_S_GAMMA`, `BATCH_S_ROUND`.
- Prover side: same accumulator pattern, runs the matching sumcheck
  prover after all levels.

**Acceptance**:
- E2E with `recursive_folds ∈ {0, 1, 2, 3, 4}` and CR enabled verifies.
- Equivalence test: batched closing equals per-level direct closes.
- Focused bench targets at NV=25 single thread, OneHot D64 CR:
  - ≤ **20 ms** (today: ~41 ms; projected ~22 ms but adding overhead
    for the bigger sumcheck closes at ~20–22 ms).

### Phase G.2 — Also batch weight evaluation

**Goal**: the weight eval is the other ~14 ms at NV=25 (per-level
sum of `eval_setup_weight_at_point`). Push it into the same batched
sumcheck.

- Define `w_combined(i,j,k) = Σ_L γ^L · w^{(L)}(i,j,k; r_x^{(L)})` and
  fold it into the batched sumcheck as the eq-weighted factor.
- The closing oracle becomes `S(r*) · w_combined(r*)`. Both `S(r*)` and
  `w_combined(r*)` are evaluated once at the end. `w_combined(r*)` is
  `O(K · per-level weight cost)` at the closing point; the closing
  point evaluation reuses the existing
  `eval_setup_weight_at_point` per level (which is already optimized).

**Acceptance**:
- Focused bench targets at NV=25 single thread, OneHot D64 CR:
  - ≤ **10 ms** (today: ~41 ms; projected ~10 ms after batching both
    mle and weight).

### Phase G.3 — Remove `materialize_setup_claim_tables` from hot path

**Goal**: clean up the verifier and ensure no surprise materializations
remain.

- Keep `materialize_setup_claim_tables` only under `#[cfg(test)]` for
  reference tests. Remove all production callers.
- Audit that no other verifier code reads from `setup.shared_matrix`
  on the hot path (other than the deferred batched mle).

**Acceptance**: `cargo test --workspace` passes; production
verifier code paths reference the shared matrix only inside the
deferred batched closing sumcheck.

### Phase H — Flip CR on by default in fp128 presets

**Goal**: make the production verifier benefit, not just the
opt-in benchmarks.

- Set `D64OneHot`, `D64Full`, `D128OneHot`, `D128Full` to
  `use_setup_claim_reduction = true`.
- Regenerate the six fp128 schedule tables via
  `cargo run -p akita-config --features planner --bin
  gen_schedule_tables`.
- Update generated schedule snapshots and any byte-exact proof-size
  expectations.

**Acceptance**: `cargo test --workspace` passes. Default
`bench_onehot_phases` now exercises CR.

### Phase I — Planner cost model update

**Goal**: the planner should be able to pick the schedule shapes that
Phase G unlocks.

- Add a `verifier_setup_cost_per_level` term to the planner cost
  function. Today this is implicitly `O(setup_per_level)`. After
  Phase G it is `O(N'^{1/4})` (leaf-only).
- Tune the schedule search to prefer more recursive levels with
  smaller per-level `S` when CR is active, since the per-level setup
  cost is now amortized into the cascade.
- Regenerate schedule tables one more time with the corrected cost
  model and audit deltas.

**Acceptance**: planner picks schedules with smaller per-level `S`
when CR is on. Bench results don't regress against Phase H numbers
(should improve marginally at NV ≥ 22).

### Phase J — Final apples-to-apples vs `main`

**Goal**: prove the win.

- Run `AKITA_PARALLEL=0 cargo bench -p akita-pcs --bench akita_e2e --
  'akita/onehot-d64/nv(15|20|25)/(prove|verify)$'` on both `main` and
  this branch.
- Record results in this file's results table below and in the
  roadmap's implementation log.
- Open the PR.

**Acceptance**: branch verifier ≤ `main` verifier at NV=25. Branch
verifier ≤ 1.3× `main` at NV=15. Prover stays at least as fast as
current branch numbers (NV=25 ≈ 510 ms).

## Order of execution

1. **G.0** — Reference batched evaluator + test + micro-bench. **DONE,
   negative result.** Naive batched MLE is 0.5–0.7× of per-level at the
   actual NV=25 setup-matrix shapes.
2. **K.0** — Hand-built mixed-shape schedule + E2E prove/verify test.
   **DONE.** `tests/hybrid_stage1_e2e.rs` proves the architecture
   supports per-level shape mixing.
3. **K.1** — Extend planner DP to `{Flat, Tensor} × log_basis`.
   **DONE WITH A CAVEAT.** New `planner_stage1_shapes_to_search()` trait
   method plus a `try_apply_planner_shape` post-hoc swap. At NV=20 the
   planner picks all-Flat schedules (7.4% smaller proof). **At NV=25 the
   post-hoc swap is internally inconsistent** (the recursive
   `(m_vars, r_vars, num_blocks, block_len)` split is computed for the
   base shape's mass; only `num_digits_fold` is recomputed for the
   target shape); the runtime then rejects the schedule at the
   third+ recursive fold with "scheduled recursive level did not match
   runtime state". **Fix:** add a shape-parameterized recursive layout
   helper to the proof_optimized macros so the planner can re-derive the
   layout from `params_only` per-shape instead of patching after.
4. **K.4 (partial)** — Bench hybrid vs tensor-only at NV=15 and NV=20.
   **DONE.** NV=25 deferred until the cascade bug is fixed.
5. **K.1 fix-up** — Re-derive recursive layout per-shape (proper fix).
   Pending.
6. **K.4 (continued)** — Rerun benches at NV=25 once K.1 fix-up lands.
   Pending.
7. **H/I/J** — Production preset flip, planner cost model, final
   comparison vs `main`. Pending.

## Phase K results so far

Single-threaded (`AKITA_PARALLEL=0`), `D64OneHot` (ExactShell `{18,0}`
mapped to Tensor by default; hybrid lets the planner pick Flat per
level instead):

| NV | metric           | tensor-only (default) | planner-hybrid    | Δ                |
|---:|------------------|----------------------:|------------------:|------------------|
| 15 | proof bytes      |              32 112   |          28 016   | −12.8%           |
| 15 | prove            |              9.98 ms  |          13.05 ms | +31% slower      |
| 15 | verify           |              1.00 ms  |          0.89 ms  | −11% faster      |
| 20 | proof bytes      |              79 744   |          72 856   | −8.6%            |
| 20 | prove            |               314 ms  |           247 ms  | **−21% faster**  |
| 20 | verify           |              4.16 ms  |          3.52 ms  | **−15% faster**  |
| 25 | (planner output) |              87 792   |  (panic at fold 4)| —                |

At NV=20 the hybrid wins on both prove and verify. At NV=15 the
verifier wins but the prover loses to tensor. Crossover is between
NV=15 and NV=20. At NV=25 we can't measure until the cascade bug is
fixed.

Note: the absolute prove/verify numbers in this run came out higher
than the earlier `vs main` snapshot for the same `D64OneHot` config.
That delta is suspected measurement noise (different host conditions)
rather than a regression introduced by K.1; the `vs main` comparison
should be redone in a controlled single bench run after K.1 fix-up.

## Results table (filled in as phases complete)

| Phase                         | NV=15 verify | NV=20 verify | NV=25 verify | NV=25 prove | Δ vs main verify |
|-------------------------------|-------------:|-------------:|-------------:|------------:|-----------------:|
| Before (committed, default)   |     0.688 ms |     2.850 ms |     10.29 ms |      510 ms |  +105% (slower)  |
| Phase G.2 default (target)    |   ≤ 0.65 ms  |    ≤ 1.6 ms  |    ≤ 4.5 ms  |    ≤ 550 ms |   ≈ main (parity)|
| Phase H default (target)      |    ≤ 0.6 ms  |    ≤ 1.4 ms  |    ≤ 4.0 ms  |    ≤ 550 ms |    ~ −20% (faster)|
| Phase J (final)               |       —      |       —      |       —      |       —     |        —         |

`main` reference at the same bench points (committed in
`tensor-everywhere-implementation-plan.md`):

| NV | prove   | verify |
|---:|--------:|-------:|
| 15 | 5.69 ms | 575 µs |
| 20 | 234 ms  | 1.58 ms |
| 25 | 1054 ms | 5.03 ms |
