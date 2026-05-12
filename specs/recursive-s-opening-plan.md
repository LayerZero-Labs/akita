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

### Strategy options (pick one before resuming)

| # | Option | What it gives | Cost | Risk |
|---|--------|---------------|------|------|
| A | Accept verifier regression; ship branch as a prover-perf branch | Prover 2.1× faster at NV=25 already committed | Done | Low |
| B | Revert tensor stage-1 + CR scaffolding; match `main` verifier | Recovers verifier parity, loses prover wins | ~1 day rebases | Low |
| C | Add a smarter batched-S evaluator (precompute combined eq tables across all three dims) | Maybe 1.5–2× over per-level | A few days; might still lose | Medium |
| D | Parallelize `mle` with rayon; re-bench with parallel feature on | ≤ Ncore× verifier speedup, *only with parallelism on* | Half day | Low |
| E | Tiered chunked commitments on `S` (true fourth-root from spec) | Asymptotic verifier win at large NV | Multi-week | High |
| F | Restructure setup matrix so per-level `max_stride` shrinks with depth | Asymptotic verifier win, no new commitments | Multi-week, planner work | Medium-High |

Recommendation in standup: A is the honest default if there isn't budget
for E/F. D is worth trying as a cheap sanity check — `mle` is trivially
row-parallel and rayon coverage of the verifier hot path has not been
audited. C is a credible "few more days" attempt before committing to E
or F.

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

1. **G.0** — Reference batched evaluator + test + micro-bench (no
   protocol changes; just a perf-validated helper). Aim: ~half day.
2. **G.1** — Wire batched mle into the protocol (defer + close once).
   Aim: 1 day. Regen schedule tables once for the bigger proof.
3. **G.2** — Also batch weight evaluation. Aim: half day. No proof-byte
   delta beyond G.1.
4. **G.3** — Clean up: remove `materialize_setup_claim_tables` from hot
   path. Aim: ~1 hour.
5. **H** — Flip CR on by default in fp128 production presets and regen
   tables. Aim: 2–3 hours including table regen and snapshot updates.
6. **I** — Planner cost model update + table regen. Aim: half day.
7. **J** — Final comparison vs `main`; PR. Aim: ~1 hour.

Steps 1–4 are done in this branch sequentially. Steps 5–7 are gated on
a green G.3.

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
