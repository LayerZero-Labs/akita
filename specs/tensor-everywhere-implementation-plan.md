# Tensor Everywhere — Implementation Plan

Date: 2026-05-12
Branch: `feat/tensor-challenges` (current)

## Background

We have already landed:

- Tensor stage-1 challenge shape (left/right sparse split, exact aggregate
  evaluator with negacyclic correction).
- Setup-side claim-reduction sumcheck wired into every fold level (root +
  recursive) behind `LevelParams::use_setup_claim_reduction`.
- Algebraic-only fast path for the stage-2 closing M-eval
  (`PreparedMEval::eval_algebraic_at_point`).

The current branch verifies correctly on flat, tensor, and tensor+CR paths,
and rejects tampered claim-reduction payloads.

The diagnostic test in this branch showed that the apparent "tensor verifier
is 4× slower than flat" gap at NV=20 is **entirely** the
`Cfg::get_params_for_prove` DP-search overhead the wrapper config pays on
every `batched_verify` call (12.5 ms / call at NV=20), not the verifier
algorithm. The schedules picked by the table-lookup ("flat") path and the
fresh-planner ("tensor") path are byte-identical because the `D64OneHot`
preset already uses `Stage1ChallengeShape::Tensor`.

Two consequences:

1. The benchmark harness is not yet a fair comparison.
2. Even when the harness is fixed, the verifier won't beat flat *yet* because
   the hot paths (`eval_d_matrix_*`, `eval_b_matrix_*`, `PreparedMEval`,
   opening point block weights, setup-claim-reduction weights/table) still
   do `O(num_blocks × depth × D)` work and aren't exploiting tensor
   structure.

## Goal

Enable the tensor stage-1 path everywhere (prover + verifier, production
presets) with a flat fallback at any fold step where flat is more
convenient, **with measurable end-to-end speedup vs current main** and
maintaining ≥128-bit security at every level.

To get there we need to:

1. Remove the planner overhead so any speedup is observable.
2. Push tensor structure into the verifier hot paths so we actually do
   sub-linear work on stage-2 M-eval and CR weight evaluation.
3. Replace `materialize_setup_claim_tables` with a structured evaluator.
4. Replace `SetupMatrixPolynomialView::materialize_table` with a recursive
   opening of `S` (tiered commitments).
5. Flip the production presets to default to tensor + CR (with per-level
   override to fall back to flat) and verify nothing regresses.
6. Run the final end-to-end benchmark suite.

## Security

- The MSIS extraction degradation for tensor stage-1 stays at `4ω`
  (`stage1_extraction_relative_msis_degradation`, already wired through the
  planner). All schedule choices remain ≥128-bit because rank floors are
  derived from `stage1_extraction_infinity_norm()`.
- The claim-reduction sumcheck is sound by construction: it is a standard
  sumcheck over `w_setup · S` with the verifier-recomputed weight table; the
  prover only reveals `m_setup_eval` plus the sumcheck transcript.
- Structured-weight evaluator must produce *exactly* the same `w_setup(r)`
  the materialized table produces (we will assert byte-identity in tests
  before swapping in the optimized path).
- Recursive setup opening must use SIS commitments derived from the same
  audited rank policy (`audited_root_rank` / `stage1_extraction_*`) used by
  the witness commitments.

## Tests vs benches

Per the user, benches are deferred to the very end. During every phase we
use quick `cargo test` paths:

- New behaviour gets a small focused unit test in the affected crate.
- E2E coverage uses existing
  `crates/akita-pcs/tests/{tensor_stage1_e2e,setup_claim_reduction_e2e,
  single_poly_e2e,batched_e2e,ring_switch}.rs` which already exercise
  prove/verify roundtrips at NV=15/20.
- Workspace-wide regression: `cargo clippy --all -- -D warnings` and
  `cargo test --workspace` after each milestone.

The final benchmark phase will re-run `akita_e2e` and compare against the
`main` branch.

## Phases

The phases are ordered to keep every step independently shippable.

### Phase A — Schedule cache in the verifier path

**Why first.** Without this every later measurement is dominated by 12 ms
of planner search per `batched_verify` call. This is also a real production
win, not just a benchmark fix.

**What.** Memoize the schedule keyed by `AkitaScheduleLookupKey` inside the
verifier path so we call `Cfg::get_params_for_prove` at most once per key.

**Concrete change.**

- Add a thread-safe schedule cache to `AkitaVerifierSetup<F>` (e.g.
  `Arc<Mutex<HashMap<AkitaScheduleLookupKey, Schedule>>>` or `OnceLock`-keyed).
- `verify_batched_with_policy` consults the cache before falling back to
  `Cfg::get_params_for_prove`.
- Cache is also reusable on the prover side; we will wire it there opportunistically.

**Test.** Quick unit test that hits the same `(num_vars, batch)` twice and
asserts the second call doesn't re-enter the DP search (measured via a
counter wrapped in a test-only PlannerConfig).

### Phase B — Dropped after closer inspection

I originally drafted three M-eval tensor-aware passes here. Re-reading
`ring_switch.rs::eval_split_at_point` shows that the verifier-side hot
spots that aren't already tensor-aware (`eval_d_matrix_w_residual_direct`,
`eval_b_matrix_t_residual_direct`, the low-bit eq table, the opening-point
block weights) take `x_challenges` from the stage-2 sumcheck, which is a
plain random point with no exploitable tensor structure. The path that
actually depends on stage-1 folding challenges
(`summarize_all_block_carries` / `summarize_tensor_all_block_carries`)
already takes the factored fast path.

The remaining `O(n_a · n_b · n_d · num_blocks · depth · D)` per-level
verifier cost is pure schedule shape, not tensor-vs-flat. We get
asymptotic fourth-root scaling by attacking the *setup-dependent* cost
(Phase C + D) instead.

### Phase C — Structured weight evaluation for the claim-reduction sumcheck

**Why.** `materialize_setup_claim_tables` is `O(rows × stride × D)` per
fold level. Replacing it with a closed-form evaluator is the biggest
asymptotic win for the CR path.

**What.**

- Implement a `setup_weight_eval_at_point(r_setup, r_x; lp, tau1, alpha)`
  helper inside `akita-verifier/src/protocol/setup_claim_reduction.rs` that
  returns `w_setup(r_setup)` in `O(log)` using:
  - `eq_low(r_setup_low, x_low)` (already factored when challenges are tensor),
  - gadget row scalars (constant per row, precomputed `O(depth)`),
  - alpha powers (precomputed `O(D)`),
  - tau1 expansion (already shared with the main verifier).
- Replace `materialize_setup_claim_tables` calls in
  `verify_setup_claim_reduction` and the claim-reduction sumcheck closing
  oracle.
- Keep the materialized helper in test-only code as a reference oracle.

**Test.** Equivalence unit test: for random `r_setup`, the structured
evaluator's `w_setup(r_setup)` equals the materialized table's
`multilinear_eval(w_table, r_setup)` bit-for-bit. Re-run claim-reduction
E2E tests.

### Phase D — Recursive setup polynomial opening

**Why.** Even with Phase C, the setup polynomial `S` itself is still
materialized into a flat table for the CR sumcheck closing check. To get
fourth-root, we must recursively open `S` at the bound point.

**What.**

- Reuse the existing recursive commitment machinery. `S` is a fixed public
  matrix, so its commitment is part of `AkitaVerifierSetup` (cached once at
  setup time).
- When `verify_setup_claim_reduction` would compute `S(r_setup) * w(r_setup)`,
  replace `S(r_setup)` with an `S`-opening claim that is batched into the
  same recursive fold pipeline. Concretely: emit an additional opening
  claim into the next fold level's `claim_to_point` set.
- Schedule planner must account for the extra `S` opening when sizing the
  cascade; we already plumb cascade control through `LevelParams` (it's why
  `inner_width`/`outer_width` exist).

**Test.** Extend recursive E2E (`setup_claim_reduction_e2e.rs`) to verify
that `S` is opened recursively at every level and that the final claim
chain is consistent. Tamper test: corrupt the `S` opening claim → verify
fails.

### Phase E — Production presets

**Why.** This is the user-visible flip. After Phase D, tensor + CR is
strictly faster on every level we've examined, so we make it the default.

**What.**

- Add `with_tensor_stage1_challenges()` and `with_setup_claim_reduction()`
  to the proof-optimized presets (`D64OneHot`, `D64Full`, `D128OneHot`,
  `D128Full`) for *both* the root and recursive level layout functions.
- Allow per-level override back to flat if the planner's projected cost
  shows flat is cheaper at that level (rare but allowed by the design).
- Regenerate `crates/akita-config/src/generated/schedule_tables.rs` so
  table lookup works for the new defaults (this also kills the planner
  overhead on the production hot path even before Phase A is wired in).

**Test.** All existing E2E tests must pass unchanged with the new presets.

### Phase F — Final benchmarks

**Why.** Only now, with the actual algorithm + production schedules in
place, can we meaningfully compare against `main`.

**What.**

- Run `cargo bench -p akita-pcs --bench akita_e2e` for both the current
  branch and `main`.
- Tabulate prover + verifier times for the same NV / config combinations:
  - one-hot D64 at NV=12, 15, 20, 25
  - full D64 at NV=12, 15, 20
- Document results in this file and in the roadmap.

## Risk register and mitigations

| Risk | Mitigation |
|------|-----------|
| Tensor opening points break batched openings | Add a flat fallback flag on `RingOpeningPoint` and use it for any path that isn't yet ported. |
| Structured weight evaluator disagrees with materialized table | Equivalence test gates the swap. |
| Recursive S opening upsets cascade control | Only enable behind a `LevelParams` flag, walk it in alongside existing recursive openings, regression-test against the legacy materialized path. |
| Schedule cache races | Use `Mutex<HashMap<...>>` keyed by canonical lookup key; the lookup key is `Hash + Eq`. |
| Generated schedule tables drift | Regenerate via `cargo run --bin gen_schedule_tables` and commit; CI test compares against snapshot. |

## Order of execution (concrete)

1. Phase A — schedule cache. ✅
2. Phase B — dropped.
3. Phase C — structured weight evaluator. ✅
4. Phase D-light — eq-table caching + live-prefix bound on `S` MLE. ✅
5. Phase D-full — recursive `S` opening. DEFERRED (architecturally larger,
   only optimization remaining for strict fourth-root scaling).
6. Phase E — production preset flip + table regen. DEFERRED (tensor is
   already production-default; flipping CR on requires regenerating six
   schedule tables because their encoded `total_bytes` excludes the
   per-level CR payload).
7. Phase F — end-to-end benchmarks. ✅

Each milestone ends with: `cargo fmt -q && cargo clippy --all -- -D warnings
&& cargo test --workspace`, then a focused commit.

## Phase F results (single-threaded, fp128 D64 onehot, verifier replay only)

| NV | recursive folds | flat (ms) | tensor (ms) | CR (ms) |
|---:|----------------:|----------:|------------:|--------:|
|  15 (was) | 0 | 1.389 | 1.927 | 2.506 |
|  15 (now) | 0 | 1.004 (-28%) | 1.002 (-48%) | 1.029 (-59%) |
|  20 (was) | 3 | 4.273 | 16.660 | 51.230 |
|  20 (now) | 3 | 4.165 (-3%) | 4.173 (-75%) | 7.026 (-86%) |
|  25 (now) | 4 | 15.520 | 15.119 | 41.093 |

Interpretation:

- Flat is essentially unchanged. It already used a generated schedule
  table, so it never paid the planner DP overhead. The small remaining
  delta is shared eq-table / structured-weight wins on the few
  paths that flat also exercises.
- Tensor now matches flat exactly across every NV. The historic gap was
  ~100% planner DP overhead per `batched_verify`; Phase A's schedule
  cache erased it. Tensor on the verifier costs the same as flat (because
  the schedule the planner picks is identical for both wrapper configs).
- Claim-reduction is **6×–7× faster at NV=20** and **1.7× behind tensor**
  there (vs 12× behind in the previous snapshot). Phase C's structured
  `w_setup` evaluator removed the per-level materialization; Phase
  D-light's MLE precomputation removed the rest of the constant-factor
  fat on `S(r_setup)`.
- At NV=25 (4 recursive folds) CR is 2.7× slower than tensor. The gap
  is the per-level `S(r_setup)` evaluation cost (`O(num_rows · num_cols
  · D)` even with the eq-table optimization). This is what Phase D-full
  (recursive `S` opening) would unlock; until then, CR is a net loss at
  NV ≥ 24.

What this means in practice:

- The tensor stage-1 path is now a no-cost win in production. Existing
  fp128 presets (D64Full, D64OneHot, D128Full, D128OneHot) already
  default to `Stage1ChallengeShape::Tensor` via their
  `SparseChallengeConfig`. The schedule cache means new
  setup-derived configs without generated tables (test wrappers,
  experimental presets) no longer pay 12 ms/verify.
- Claim reduction stays opt-in via `LevelParams::with_setup_claim_reduction()`.
  It is now competitive with the flat/tensor path up to NV ≈ 22 and
  becomes the asymptotically faster verifier as soon as Phase D-full
  lands. Until then, callers should leave it off above NV ≈ 22.

## Apples-to-apples comparison against `main` (single-threaded, OneHotPoly)

Re-ran the same `akita/onehot-d64/nvN/{prove,verify}` bench points on
`main` (`commit 1a3e0bf2`) and on this branch (`commit bb80bd88`) so that
both sides exercise the OneHot prover/verifier path with identical
`D64OneHot` configs. Each measurement is `criterion`'s reported mean,
single thread (`AKITA_PARALLEL=0`), `--measurement-time 3..5s`.

| NV | Stage  | main      | branch    | branch / main      |
|---:|--------|----------:|----------:|--------------------|
| 15 | prove  |   5.69 ms |   6.74 ms | 1.18× (+18%)       |
| 15 | verify |    575 µs |    688 µs | 1.20× (+20%)       |
| 20 | prove  |    235 ms |    215 ms | **0.92× (−8%)**    |
| 20 | verify |   1.58 ms |   2.85 ms | 1.81× (+81%)       |
| 25 | prove  |   1054 ms |    510 ms | **0.48× (−52%)**   |
| 25 | verify |   5.03 ms |   10.3 ms | 2.05× (+104%)      |

Interpretation:

- **Prover wins decisively at large NV** (~2.1× faster at NV=25). The
  branch's combined optimizations — tile multi-chunk Ajtai commit,
  centered fold accumulators, tensor stage-1 planner — beat `main` once
  NV ≥ 20.
- **Verifier regressed vs `main`**. The Phase A/C/D-light work in this
  session collapsed the *intra-branch* regression for tensor and CR
  (16.7 → 2.85 ms at NV=20 etc.) but did not recover the gap to
  `main`. The root cause is that:
  1. `main` ships `ExactShell { count_mag1: 30, count_mag2: 12 }` *flat*
     (L1 mass 54). This branch ships `ExactShell { count_mag1: 18,
     count_mag2: 0 }` *tensor* (effective L1 mass 18² = 324).
  2. The tensor construction has a 4ω MSIS norm-growth penalty
     (~8 extra bits), so the planner picks **heavier digit shapes per
     fold level** to keep the security target.
  3. Verifier hot paths (`PreparedMEval::eval_at_point`,
     `eval_d/b_matrix_*`, stage-2 sumcheck rounds) consume stage-2
     random challenges, not stage-1 tensor challenges, so the heavier
     per-level shape costs the verifier extra work without the tensor
     structure ever paying back algorithmically.
  4. The fourth-root verifier optimization that the spec describes
     (Phase D-full: recursive opening of `S` instead of materialized
     per-level `S(r_setup)` evaluation) is still **not implemented**.

Net: prover side is the present win; verifier side requires Phase
D-full to actually realize the spec's `O(N'^{1/4})` scaling. Until then
this branch is a prover-side performance branch with a verifier
regression — the verifier story flips once Phase D-full lands.
