## Recursive-S Opening: Plan to Beat `main` on the Verifier

Status: drafted 2026-05-12. Author: `feat/tensor-challenges`.

This plan finishes the fourth-root verifier optimization. It does **not**
introduce new cryptographic primitives. It restructures how the verifier
evaluates the shared setup polynomial `S` so that the per-level cost drops
from `O(num_rows · num_cols · D)` to `O(N'^{1/4})`, which is the missing
piece that the spec describes and that today's branch lacks.

Once this lands and CR is enabled by default for production fp128 presets,
projected NV=25 verifier ≤ 3 ms (vs `main` 5 ms and current branch 10 ms);
the gap should widen at NV ≥ 28.

## Where the per-level setup cost lives today

Both the default tensor path and the CR path bottleneck on the same
`O(num_rows · num_cols · D)` work per level:

- **Default tensor path** (`PreparedMEval::eval_split_at_point`) walks
  `w_d`, `t_b`, `z_dense_setup` and reads `D`/`B`/`A` rows of the shared
  setup matrix.
- **CR path** replaces the above with a sumcheck, but still finishes by
  evaluating `SetupMatrixPolynomialView::mle(r_row, r_col, r_coeff)`
  inside `verify_setup_claim_reduction`. The `mle` walks the whole live
  prefix of the shared matrix.

So both paths need a fast `S` evaluator to actually scale.

## Approach: recursive `S` opening via cascade

`S` is **public**: it is deterministically expanded from
`PublicMatrixSeed` and shared across all proofs from the same setup. So
the verifier could in principle evaluate `S(r)` itself; it just can't
afford to.

The construction:

- Treat `S_level` (the setup matrix used at level L) as a polynomial we
  open at the verifier's challenge point `r_setup^{(L)}`.
- The prover sends the claimed value `s^{(L)} = S_level(r_setup^{(L)})`
  with the level-L proof.
- The pair `(r_setup^{(L)}, s^{(L)})` is **cascaded into level L+1's
  witness** as an additional polynomial-evaluation claim. Level L+1
  already batches its own witness opening with arbitrary public-point
  claims via the existing batched-sumcheck infrastructure, so adding one
  more eq-weighted claim is structurally the same shape.
- At the deepest level (the smallest `S`), the verifier just evaluates
  `S_deepest(r_setup^{(deepest)})` directly — that one is cheap because
  the deepest setup is small.

The total per-level setup cost becomes: one extra eq-weighted summand in
each level's stage-2 sumcheck (negligible), plus one direct evaluation
of the smallest `S` at the bottom. Concretely with the existing
recursive schedule the bottom `S` lives at the last fold level, sized
`O(N'^{1/4})`, so the total verifier setup work across the entire fold
ladder is `O(N'^{1/4})` instead of `Σ_L O(setup_per_level_L)`.

This is exactly the "tiered commitment / cascade control" item the
roadmap and the spec call out as the missing fourth-root piece.

## What this is **not**

- Not a new commitment scheme. We do not commit to `S` explicitly. The
  `s^{(L)}` value is just an additional public claim discharged by the
  existing recursive-fold sumcheck, the way every other public
  polynomial-evaluation claim already is.
- Not a soundness change at the cryptographic level. The cascade is a
  pure protocol rewrite: the closing equality at level L moves from
  "verifier evaluates `S_level` itself" to "verifier accepts `s^{(L)}`
  if and only if the cascaded claim is upheld at level L+1". This is
  the standard reduction trick — same one that already powers the
  recursive witness fold.
- Not an asymptotic change for the prover. Prover work stays
  `O(setup_per_level)` per level (still has to read `S` to produce
  `s^{(L)}`), but that's already paid today. Prover may even speed up
  slightly because we drop a few materializations on the verifier hot
  path.

## Risks and unknowns

| Risk | Mitigation |
|------|-----------|
| Cascade breaks at the deepest level if final-level `S` is still large | Cap final-level `S` size via planner; fall back to direct mle when below a threshold. |
| Cascade changes proof shape and proof bytes | Bump `LevelProofShape` to carry the new `s^{(L)}` field, add a version byte, regen six fp128 generated schedule tables for the new size. Existing snapshot tests will be updated, not preserved. |
| Cascade interferes with batched openings | Add the cascade claim as a sibling of the witness opening in the level-L+1 sumcheck input claim, not a replacement. Existing batched-opening code already accepts multiple input claims. |
| Soundness: cascade adds one round per level | Audit transcript labels for the new `s^{(L)}` absorption. Add equivalence tests for "direct mle" vs "cascade closes at depth k". |
| Final-level direct mle still costs something | Acceptable — that's the leaf of the recursion, `O(N'^{1/4})` by construction. |
| Planner cost model is wrong | Phase I updates the model. Until then, the current schedules may pick non-optimal `recursive_folds`; tensor + CR remains a net win even on sub-optimal schedules. |

## Phases (concrete and incremental)

Each phase ends with: `cargo fmt -q && cargo clippy --all -- -D warnings
&& cargo test --workspace`, then a focused commit.

### Phase G.0 — Reference recursive evaluator (no proof change yet)

**Goal**: prove the algorithmic trick works before touching protocol.

- Add `setup_polynomial_view_at_level` helper: given a level index and a
  child-level `S_child`, return the row/col/coeff prefix of `S_parent`
  used at that level.
- Add a debug helper `eval_s_recursive` on `SetupMatrixPolynomialView`
  that takes a sequence of `(r_setup^{(L)},)` opening points plus the
  deepest-level direct eval and returns the parent-level result. This
  matches the cascade math but is computed eagerly from the public
  matrix.
- Add a test `recursive_s_eval_matches_direct_mle` that, for random
  challenge points across 0–4 cascade levels, asserts the recursive
  evaluator equals the current `SetupMatrixPolynomialView::mle` result.
- Add a benchmark `bench_setup_s_recursive_vs_direct` measuring the
  evaluator standalone (no protocol) at NV=15/20/25.

**Acceptance**: tests pass, recursive evaluator at NV=25 takes
< 0.5 ms vs direct ~7 ms.

### Phase G.1 — Add `s^{(L)}` to the level-L proof payload

**Goal**: extend the proof shape without changing the verifier
algorithm.

- Add `SetupClaimReductionPayload::s_opening_claim: F` field beside
  `m_setup_eval` (`Some(value)` at every level when CR is on, `None`
  otherwise).
- Extend `LevelProofShape::stage2_setup_claim_reduction` to size
  the new field.
- Update `AkitaDeserialize` / `Valid` / `serialized_size`. Bump the
  proof version byte.
- Prover writes `s^{(L)}` by directly evaluating `S_level(r_setup^{(L)})`
  (the current code path) — same value the verifier computes today.
- Verifier reads it but still cross-checks against direct mle. Behavior
  is identical to today; this phase only adds a slot for the cascade.

**Acceptance**: existing E2E + snapshot tests pass after regenerating
generated tables. Proof bytes grow by `sizeof(F)` per CR level.

### Phase G.2 — Cascade `s^{(L)}` into level L+1 as a public claim

**Goal**: make level L stop evaluating `S` directly.

- Extend the level-L+1 stage-2 sumcheck input claim to accept an
  additional eq-weighted summand:

  ```text
  input_claim_{L+1} = witness_claim + λ_s · s^{(L)}
  ```

  where `λ_s` is a transcript-derived scalar and the new summand is
  evaluated against the level-L+1 setup matrix view restricted to the
  `r_setup^{(L)}` opening point.

- Add transcript labels: `BATCH_S_CASCADE_LAMBDA`, `BATCH_S_CASCADE_ROW`.

- At level L, the verifier now accepts `s^{(L)}` without direct mle
  evaluation; instead, the equivalence is enforced when level L+1's
  sumcheck closes.

- At the **deepest** level (call it `D`), the cascaded claim still has
  no L+1 to push into, so the verifier evaluates
  `S_deepest(r_setup^{(D)})` directly. The deepest `S` is bounded by
  the schedule so this is the leaf cost.

- Prover side: thread the cascade lambda into stage-2 sumcheck input
  construction. Reuse the existing batched-claim infrastructure.

**Acceptance**: E2E proofs with `recursive_folds ∈ {0, 1, 2, 3, 4}` and
CR enabled verify correctly; the verifier never calls
`SetupMatrixPolynomialView::mle` at any non-leaf level (assert via
debug counter).

### Phase G.3 — Drop the `mle` path from the verifier hot loop

**Goal**: actually realize the speedup.

- Remove the `setup_view.mle` call in `verify_setup_claim_reduction`
  for non-leaf levels.
- Keep `mle` as a fallback for the deepest level only.
- Remove `materialize_setup_claim_tables` from the verifier prelude;
  keep it under `#[cfg(test)]` for sanity tests.

**Acceptance**: focused verifier benches for tensor + CR drop
substantially. Targets at NV=25 (single thread, fp128 D64 OneHot):
- tensor + CR: ≤ 4 ms (today: 41 ms, projected post-cascade: ~3 ms).
- tensor (no CR, default): unchanged from current branch (~10 ms).

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

1. G.0 (no-op observable; just a perf-validated evaluator)
2. G.1 (proof shape bump; regen tables once)
3. G.2 (cascade wiring; existing CR tests cover correctness)
4. G.3 (drop `mle`; bench verifier wins)
5. H (flip CR default; regen tables; absorb byte-size deltas)
6. I (planner cost model; regen tables one more time if needed)
7. J (final comparison; PR)

Steps 1–4 can be done in this branch sequentially. Steps 5–7 can be
gated on a green G.3 before regenerating production tables.

## Results table (filled in as phases complete)

| Phase | NV=15 verify | NV=20 verify | NV=25 verify | NV=25 prove | Δ vs main verify |
|-------|-------------:|-------------:|-------------:|------------:|-----------------:|
| Before (committed) | 0.688 ms | 2.850 ms | 10.29 ms | 510 ms | +105% (slower) |
| Phase G.3 (target) | ≤ 0.7 ms | ≤ 1.5 ms | ≤ 4 ms | ≤ 550 ms | ≤ −20% (faster) |
| Phase H (target)   | ≤ 0.7 ms | ≤ 1.5 ms | ≤ 4 ms | ≤ 550 ms | ≤ −20% (faster) |
| Phase J (final)    | —        | —        | —        | —        | —                |

`main` reference at the same bench points (committed in
`tensor-everywhere-implementation-plan.md`):

| NV | prove   | verify |
|---:|--------:|-------:|
| 15 | 5.69 ms | 575 µs |
| 20 | 234 ms  | 1.58 ms |
| 25 | 1054 ms | 5.03 ms |
