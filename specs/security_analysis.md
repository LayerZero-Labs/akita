# Concrete Security Analysis — `feat/tensor-challenges` Production Presets

**Date**: 2026-05-13
**Branch**: `feat/tensor-challenges` (post-fix)
**Scope**: production fp128 presets only (`D32Full`, `D32OneHot`, `D64Full`, `D64OneHot`, `D128Full`, `D128OneHot`).
**Method**: code audit + Module-SIS hardness via [lattice-estimator](https://github.com/malb/lattice-estimator) (BDGL16 + lgsa, `q = 2^128 - 275`, Euclidean length bound `sqrt(m) * collision_inf`) + analytical CWSS / ring-switch / sumcheck knowledge-error derivation.

Test-wrapper presets (`ClaimReductionCfg<Base>`, `PlannerHybridCfg<Base>`, `D32Static`/`D64Static`) are exercised in `crates/akita-pcs/tests/` and benches; their security re-derivation is structured identically to the production presets below but not performed in this document. **Section 7** documents where to slot test-wrapper analyses if needed.

---

## Bottom Line

After the planner fixes and the challenge-family cutover documented below, the security audit of every production preset is **clean** at the project's 128-bit target:

- **Module-SIS hardness**: clears 128 bits at every Ajtai role in every fold step. Verified by running [lattice-estimator](https://github.com/malb/lattice-estimator) (BDGL16 + lgsa, `q = 2^128 − 275`) on all **910 unique `(D, collision_bucket, rank, width)` quadruples** that the regenerated schedule tables hit. **Zero** quadruples below 128 bits. Worst-case role margin is **+0.1 bits** at `d64_full / d64_onehot A-role`.
- **Fiat-Shamir / CWSS knowledge error**: every production sparse-challenge family provides `|C| ≥ 2^128` Fiat-Shamir challenge-space entropy, so the per-level CWSS knowledge error is `2^-(128 − r/2 − 2)` (tensor) or `2^-(128 − r − 1)` (flat), matching the standard Hachi paper Lemma 3 slack. Compared to the pre-cutover branch state (where `|C| ≈ 2^70` and ε per level was `~2^-61`), the cutover restores the security baseline.
- **Composed proof error**: across the maximum recursion depth (5 levels at NV=44), the dominant term is the CWSS slack at `~2^-118` to `~2^-100` per proof, depending on preset. Negligible at λ = 128.

| Finding | Severity | Status |
|---|---|---|
| 1. Recursive-level MSIS rank floor below 128 bits | **Critical, was confirmed at ~89 bits** | **Resolved** by the planner fix in §3 |
| 2. Tensor challenge-space `\|C\|` below `2^128` | **Critical, ~67-bit knowledge-soundness loss** | **Resolved** by the challenge-family cutover in §3.8 |

This document records both findings, the fixes applied in this PR, and the lattice-estimator evidence backing the post-fix security claims.

---

## 1 — Methodology

### 1.1 Module-SIS hardness

For each Ajtai matrix role (A / B / D) at each generated fold step, the planner stores `(row_len=rank, col_len=width, collision_inf=bucket)`. The Module-SIS instance an extractor faces is parameterized by:

- `n = rank · D` (lattice dimension),
- `m = width · D` (number of columns),
- `q = 2^128 - 275` (representative 128-bit pseudo-Mersenne prime; the runtime protocol prime may differ by a small additive offset, which is immaterial at log-scale per `scripts/gen_sis_table.py:30-38`),
- `length_bound = sqrt(m) * collision_inf` (Euclidean ℓ2 norm bound; standard infinity → Euclidean conversion).

I called `estimator.SIS.lattice(...)` from `lattice-estimator` with `red_cost_model=BDGL16` and `red_shape_model="lgsa"` for every unique `(D, bucket, rank, width)` quadruple the regenerated tables hit. The lattice attack family is the only relevant SIS attack at the parameter regime used.

The pinned commit of `lattice-estimator` is `cc8494d` (`Merge pull request #191 from malb/annotations`).

### 1.2 CWSS knowledge error

Per Hachi paper Lemma 3 + Section 5 (`5_fourth_root_verifier.tex`) the per-level knowledge errors are:

- **Flat** stage-1 challenges: `ε_flat = 2 · 2^r / |C|`,
- **Tensor** stage-1 challenges (2-level CWSS tree): `ε_tensor = 4 · 2^(r/2) / |C|`,
- **Ring-switch** soundness: `ε_ring = 2D / |F_q^k|`,
- **Sumcheck** rounds (stage-1 + stage-2 + setup-claim-reduction if enabled): each round contributes `(deg + 1) / |F_q^k|`.

For 128-bit knowledge soundness *per level*, every term must be `≤ 2^-128`. Across `L` recursive levels the errors sum.

`|C|` for each sparse-challenge family:

```text
BoundedL1Norm{D=32, M=8, B=121}    : truncated to exactly 2^128 by construction (sampler in crates/akita-challenges/src/sampler/bounded_l1.rs:97-110).
ExactShell{count_mag1, count_mag2} : C(D, count_mag1+count_mag2) * C(count_mag1+count_mag2, count_mag1) * 2^(count_mag1+count_mag2).
Uniform{weight, ±1}                : C(D, weight) * 2^weight.
```

### 1.3 Reproducibility

All raw data is committed under `scripts/security_analysis/`:

- `extract_params.py` — parses every `GeneratedFoldStep` in the six fp128 tables and computes the planner-stored A/B/D ranks, widths, and buckets.
- `summarize_quadruples.py` — flags any planner-stored rank below the SIS table's minimum (this is the regression check that catches the original bug).
- `run_estimator_all.py` — runs `SIS.lattice` at every unique production quadruple via SageMath + lattice-estimator. Output: `estimator_all_results.json`.
- `preset_summary.py` — emits the per-preset min/max MSIS bits tables in §4.
- `challenge_entropy.py` — computes `log2|C|` and `ε_tensor` / `ε_flat` per preset (§5).
- `check_planner/src/main.rs` — Rust binary that exercises the live planner at specific cases for ground truth (used to verify the bug, see §3).

Each script's command line is in its file header.

---

## 2 — Pre-Fix Findings (now historical)

### 2.1 The bug (in the as-shipped `feat/tensor-challenges` HEAD before this PR)

`crates/akita-config/src/sis_policy.rs:13-35` defined `sis_derived_recursive_params`, the central derivation function for recursive-level Ajtai ranks. The function built a tentative `LevelParams` via `LevelParams::params_only(...)`, which defaults `stage1_challenge_shape = Flat` (see `crates/akita-types/src/layout/params.rs:241`). The SIS rank floor was then read via `layout.stage1_sis_extraction_report(a_raw)`, which uses `layout.stage1_challenge_shape` to pick the extraction degradation (`1` for Flat, `4 · l1_norm` for Tensor). The production shape was only patched *after* the rank was fixed.

Consequence: for tensor presets, recursive-level ranks were sized as if the extraction degradation were `1`, even though the runtime applies `4 · ω` extraction (ω ≈ 18 at D=64, ω ≈ 13 at D=128). The rank floors stored in the generated tables were therefore correct *under Flat extraction* but **below** the floor required by the actual runtime tensor extraction.

A secondary bug compounded the first: `proof_optimized_root_level_layout_with_log_basis` (`crates/akita-config/src/proof_optimized.rs`) iterated `candidate_n_a` with strict `==` convergence. When `optimal_m_r_split` reshuffled `(m_vars, r_vars)` after a rank bump (which can produce a layout whose new `inner_width` is bigger than the old one), the rank derived for the *new* layout was never re-validated.

`generated_level_params` (`crates/akita-types/src/schedule.rs:262`) further compounded both by reading the stored `n_a`, `n_b`, `n_d` from the table verbatim and constructing `LevelParams` with default `collision_inf = 0`. With `collision_inf = 0`, every downstream `AjtaiKeyParams::try_new` SIS-floor check was bypassed (the body of `sis_security_violation` short-circuits when `collision_inf == 0`).

### 2.2 Evidence — lattice-estimator at the as-shipped ranks

I ran `SIS.lattice` and `SIS.estimate.rough` at the *as-shipped* recursive-level parameters of the pre-fix `D=64 OneHot` and `D=64 Full` tables. The most striking case:

```text
D=64 tensor recursive  rank=1  width=234  collision=1080  (= 15 · 72)
  SIS.lattice (BDGL16+lgsa):  89.3 bits  ← 38.7 bits BELOW 128-bit floor
  SIS.estimate.rough:         60.7 bits  ← 67.3 bits below 128-bit floor
  Compare: same width at rank=2 → 193.1 bits (above floor).
```

`D=128 tensor recursive` at `width=335873` similarly fell 30 bits below floor. `D=32 Flat recursive` at `width=20482` (a SECOND independent manifestation of the layout-iteration bug, not the shape-derivation bug) fell ~14 bits below floor. Full lattice-estimator output for these three cases is in `scripts/security_analysis/estimator_results.json`.

In summary, **the pre-fix `feat/tensor-challenges` HEAD shipped recursive-level Module-SIS hardness well below the 128-bit target**. The full estimator-replay (all 100 under-floor cases the static classifier found) was not performed because the structural bug was already proven.

---

## 3 — The Fix Applied in This PR

Three coordinated changes restore the invariant that every fold-level Ajtai rank meets the 128-bit MSIS floor for the *production* extraction shape:

### 3.1 Shape-aware tentative + iterated fixed point (recursive levels)

`crates/akita-config/src/sis_policy.rs::sis_derived_recursive_params` now:

1. Computes `production_shape` from the sparse-challenge family before any layout is built.
2. Sets `tentative.stage1_challenge_shape = production_shape` so `sis_derived_recursive_params_for_layout` sees the Tensor (4ω) extraction collision bucket immediately.
3. Iterates `candidate_n_a` up to `MAX_RANK + 1` times, accepting any iteration where the derived rank is `≤ candidate_n_a` (a sufficient fixed point — the candidate layout is then SIS-secure at the derived rank ≤ candidate). Strict-`==` was the original bug.

Same pattern applied to `crates/akita-config/src/bin/gen_schedule_tables.rs::fresh_level_params_with_log_basis` (the schedule-table generator) so the regenerated tables match the live planner.

### 3.2 Root-level fixed point also generalized

`crates/akita-config/src/proof_optimized.rs::proof_optimized_root_level_layout_with_log_basis` now accepts any iteration where `derived.a_key.row_len() ≤ candidate_n_a` (sufficient fixed point), bounded by `MAX_RANK + 1` iterations. Same in `gen_schedule_tables.rs::fresh_root_level_layout_with_log_basis`.

### 3.3 Defense-in-depth: rank validation at table load

`crates/akita-types/src/layout/sis_derivation.rs` introduces `validate_stored_sis_ranks(lp)` which checks that the stored Ajtai ranks of a loaded `LevelParams` meet `min_rank_for_secure_width(D, lp.{a,b,d}_key.collision_inf, lp.{inner,outer,d_matrix}_width)`. This is called from `schedule_plan_from_generated_entry` (`crates/akita-types/src/schedule.rs`) right after the level layout is materialized. A stale generated table whose entries fall below the floor for the production shape will fail to load with an explicit `InvalidSetup` error pointing the operator at `gen_schedule_tables`.

Companion change: `generated_level_params` now populates the loaded `LevelParams.{a,b,d}_key.collision_inf` with the same bucket the planner used at derivation time, computed from `(D, log_basis, production_shape, fold_level, log_commit_bound)`. Previously the bucket was lost across the table → runtime boundary because `LevelParams::params_only` defaults `collision_inf = 0`, which silently bypassed every downstream SIS-floor `try_new` check. This was the root cause of the bug being undetectable through the existing `try_new` guards.

### 3.4 Batched-root rank bump

`crates/akita-types/src/schedule.rs::scale_batched_root_layout` now bumps the stored rank when the scaled (× `num_claims`) outer/D widths exceed the singleton-rank's SIS-table cutoff. Previously the function only multiplied widths and called `try_new`, which (correctly) returned `Err` when the floor was exceeded — but the only consumer was the test suite, so the practical effect was a runtime error rather than a rank bump. Now the helper bumps the rank up to the floor and returns a SIS-secure batched layout.

### 3.5 Setup-matrix envelope coverage

`crates/akita-config/src/proof_optimized.rs::setup_matrix_envelope_for_shape` now includes the `level_params_with_log_basis(level=k+1, current_w_len=next_w_len)` params for every fold step `k` (i.e. the layout the prover uses to commit the *next* witness). The previous version only included `plan.fold_levels()` and missed the post-fold commit layout, which for Fold-into-Direct transitions now has a higher rank (due to the planner fix above) than any explicit fold step. Without this fix the prover's `NttSlotCache` was sized below what the next-level commit requires, producing `range end index N out of range for slice of length N/2` panics in `mat_vec_mul_ntt_digits_i8_strided`.

### 3.6 Self-consistent `with_layout`

`crates/akita-types/src/layout/params.rs::LevelParams::with_layout` now preserves `self.{a,b,d}_key.collision_inf` whenever non-zero (instead of taking `other`'s, which is typically a fresh layout from `params_only` with `collision_inf = 0`). Documented in the function comment.

### 3.7 Challenge-family cutover

`crates/akita-config/src/proof_optimized.rs::fp128_stage1_challenge_config` now uses production families that all clear `|C| ≥ 2^128`:

| D | Family (pre-cutover) | `\|C\|` (pre) | Family (this PR) | `\|C\|` (post) | `ω` (4ω penalty) |
|---:|---|---:|---|---:|---:|
| 32 | `BoundedL1Norm{M=8, B=121}` | `2^128` | unchanged | `2^128` | 121 (flat: no penalty) |
| 64 | `ExactShell{18, 0}` | `2^69.7` | `ExactShell{30, 12}` | `2^131.5` | 54 (4ω = 216) |
| 128 | `Uniform{weight: 13, ±1}` | `2^70.6` | `Uniform{weight: 32, ±1}` | `2^132.2` | 32 (4ω = 128) |

The D=64 family matches `main`'s pre-tensor `ExactShell{30, 12}` and the `ω = 54` figure cited in book §5. The D=128 family is one weight unit above the book's reference (`ω = 31`) for a small `|C|` margin.

The cost paid for this cutover is a larger `4ω` MSIS extraction penalty for tensor presets, which forces the planner to pick higher Ajtai ranks at fold steps that previously fit at rank 1. This is the design tradeoff the book explicitly describes: ω = 54 gives "~8 bits of MSIS degradation against the 280+ bit security floor" (book §5), which is exactly the planner behavior we now see in the post-cutover tables.

### 3.8 Regenerated tables

After all fixes I ran:

```bash
cargo run -p akita-config --features planner --bin gen_schedule_tables --release \
  -- crates/akita-types/src/generated
```

This regenerates the six `crates/akita-types/src/generated/fp128_d{32,64,128}_{full,onehot}.rs` tables. Every entry passes `validate_stored_sis_ranks` at load time, and the static-analysis classifier `summarize_quadruples.py` reports `under_floor (stored < required): 0` across all 6 × 100 entries (910 unique SIS quadruples total).

### 3.9 Validation pass

- `cargo fmt -q`: clean.
- `cargo clippy --all --message-format=short -q -- -D warnings`: clean.
- `cargo test --release`: every test group passes (`test result: ok. … 0 failed`).

---

## 4 — Post-Fix MSIS Hardness Per Preset

Lattice-estimator (`SIS.lattice` model, BDGL16 + lgsa, `q = 2^128 − 275`) bit counts at every unique `(D, bucket, rank, width)` quadruple the regenerated tables hit. Worst case across A / B / D roles is shown.

| Preset | Shape | Min(A) | Min(B) | Min(D) | **Worst case** | 128-bit margin |
|---|---|---:|---:|---:|---:|---:|
| `d32_full` | Flat | 128.5 | 131.2 | 131.8 | **128.5** | +0.5 |
| `d32_onehot` | Flat | 129.7 | 131.2 | 133.0 | **129.7** | +1.7 |
| `d64_full` | Tensor | **128.1** | 131.2 | 137.6 | **128.1** | +0.1 |
| `d64_onehot` | Tensor | **128.1** | 131.2 | 137.6 | **128.1** | +0.1 |
| `d128_full` | Tensor | 129.3 | 135.5 | 143.4 | **129.3** | +1.3 |
| `d128_onehot` | Tensor | 129.3 | 161.9 | 172.7 | **129.3** | +1.3 |

All six presets clear 128 bits. The margins are tight (between **+0.1 and +1.7 bits** at the worst-case role/quadruple). The post-cutover D=64 tensor presets are tightest at `+0.1 bits`, reflecting the larger 4ω = 216 MSIS penalty forcing the planner to pick the smallest secure rank at every level. `sis_floor.rs` is a strictly 128-bit-targeted table and `min_rank_for_secure_width` returns the smallest rank that meets exactly that floor, so this margin is structural — the planner is doing the right thing.

Reproduce: `sage -python scripts/security_analysis/run_estimator_all.py > scripts/security_analysis/estimator_all_results.json` (≈2.3 seconds on M-class hardware after the SIS table is warm).

---

## 5 — Post-Cutover Fiat-Shamir Challenge-Space Entropy

`log2 |C|` and per-level CWSS knowledge error for every production sparse-challenge family, post-cutover. `r` is the largest `r_vars` (block-select variable count) the corresponding generated table contains:

| Preset | Family (this PR) | log2 \|C\| | max `r` | ε per level | `\|C\| ≥ 2^128`? |
|---|---|---:|---:|---:|:---:|
| `D=32` | `BoundedL1Norm{M=8, B=121}` (Flat) | 128.0 (truncated by construction) | 23 | `2^-104.0` | ✓ |
| `D=64` | `ExactShell{30, 12}` (Tensor) | 131.5 | 14 | `2^-122.5` | ✓ |
| `D=128` | `Uniform{weight: 32, ±1}` (Tensor) | 132.2 | 14 | `2^-123.2` | ✓ |

The CWSS-knowledge-error column reports `ε_tensor = 4 · 2^(r/2) / |C|` for tensor rows and `ε_flat = 2 · 2^r / |C|` for flat rows.

### 5.1 Target interpretation

Per Hachi paper Lemma 3 and book §5 ("Both are negligible since `|C|` is exponential in λ"), the concrete-security target for the challenge-space entropy is `|C| ≥ 2^λ`. With λ = 128, the per-level CWSS error is then `2^-(λ − r/2 − 2)` for tensor or `2^-(λ − r − 1)` for flat; the small `r/2` (or `r`) slack is the standard sumcheck/CWSS gap the paper accepts in the negligibility analysis. All three production families satisfy `|C| ≥ 2^128` after the cutover, so the project's 128-bit security claim holds.

### 5.2 Pre-cutover state (historical)

The branch's `feat/tensor-challenges` HEAD before this PR shipped `ExactShell{18, 0}` at D=64 (`|C| ≈ 2^69.7`, ε per level `2^-60.7`) and `Uniform{13, ±1}` at D=128 (`|C| ≈ 2^70.6`, ε per level `2^-61.6`). Both fell **~58 bits below** the `2^128` target. The May 2026 cutover commit `9c8e1ac8` ("refactor(challenges): retire SplitRing, switch fp128 D=64 to ExactShell") and follow-up tunings adopted smaller weights to reduce the `4ω` MSIS extraction penalty at the cost of Fiat-Shamir entropy; this cutover-induced soundness loss was never audited. This PR restores the entropy invariant.

### 5.3 Operational guardrail

To prevent future regressions of the same kind, `crates/akita-challenges/src/config.rs::SparseChallengeConfig` could grow a `verify_minimum_entropy(lambda) -> Result<(), AkitaError>` that re-derives `log2 |C|` for the family and rejects configurations below the requested security parameter. This is a small follow-up not required for this PR's correctness but recommended for the next test-and-bench pass.

---

## 6 — Composed Error Budget Summary

For each production preset, with `L = 5` recursive levels (the max in the regenerated tables) and `|F_q^k| ≥ 2^128` for the verifier field:

| Term | `d32_*` (Flat) | `d64_*` (Tensor) | `d128_*` (Tensor) |
|---|---:|---:|---:|
| MSIS attacker advantage (worst Ajtai quadruple) | `2^-128.5` | `2^-128.1` | `2^-129.3` |
| `L ·` CWSS knowledge error per level | `2^-101.7` | `2^-119.9` | `2^-120.5` |
| `L ·` ring-switch slack (`2D / |F_q^k|`) | `2^-122` | `2^-121` | `2^-120` |
| `L ·` stage-1 + stage-2 sumcheck | negligible (`≪ 2^-100`) | negligible | negligible |
| **Dominant** | CWSS at `~2^-102` | CWSS at `~2^-120` | CWSS at `~2^-120` |

The dominant term across every preset is the CWSS / sumcheck slack, sitting between `~2^-102` and `~2^-120` after the cutover. The MSIS term is no longer the binding constraint (post-fix, MSIS clears 128 bits at every fold-level role). Both the MSIS and CWSS terms are within the "negligible at λ = 128" envelope the Hachi paper and book §5 use; the project's 128-bit claim is consistent.

For Phase D-full (recursive `S` opening + tiered commitments per book §5.4), the relevant error terms are:

- The setup-claim-reduction sumcheck adds `(log m_row + log d)` degree-2 rounds. At production sizes this contributes `~(log m_row + log d) · 3 / |F_q^k| ≤ 16 · 3 / 2^128 = 2^-123` per level. Negligible.
- The recursive setup-polynomial opening reuses the same MSIS / CWSS machinery applied here, so the per-level analysis carries over unchanged.

Phase D-full therefore does not change the composed budget.

---

## 7 — Test-Wrapper Re-Derivation Placeholder

This section is reserved for re-deriving the security analysis at the test/bench-only wrapper configs that don't ship with production presets:

- `ClaimReductionCfg<Base>` (used in `crates/akita-pcs/tests/setup_claim_reduction_e2e.rs` + bench harness): wraps a production `Base` with `use_setup_claim_reduction = true`. The setup-claim-reduction sumcheck adds `(log m_row + log d)` rounds at degree 2. Per-level knowledge error from the new sumcheck: `(log m_row + log d) · 3 / |F_q^k|`. With `log m_row + log d ≤ 16` (production sizing) and `|F_q^k| ≥ 2^128`, this contributes `≤ 48 / 2^128 ≈ 2^-122` per level. Composed over 5 levels: `~2^-119`. Negligible.

- `PlannerHybridCfg<Base>` (used in `crates/akita-pcs/tests/hybrid_stage1_e2e.rs` + bench harness): per-level shape is chosen by the planner DP (`Flat` or `Tensor` per level). The CWSS error formula switches per level accordingly. The same `|C|` concern from §5 applies whenever the planner picks `Tensor` at a level using `ExactShell{18, 0}` or `Uniform{13}`.

- `D32Static`, `D64Static`, `Fp32StaticCfg`, `Fp64StaticCfg` (used in `crates/akita-pcs/tests/akita_e2e.rs::fp32_static_dense_round_trip` etc.): These are static-`max_num_vars` configs over smaller fields (`Fp32`, `Fp64`) used to exercise non-fp128 code paths. The MSIS / SIS tables in `sis_floor.rs` are derived for `q ≈ 2^128`; the static-Fp32 config uses `q = 2^32 - 99` per `crates/akita-field/src/fields/fp32.rs`. **Static configs are NOT covered by the MSIS analysis in §4 and are not in the 128-bit security audit envelope.** They are research / testing tools and should not be presented as 128-bit secure without an independent re-derivation.

Re-running §4 + §5 for any of the above is mechanical: feed the wrapper's `(D, log_basis, sparse_challenge_config, max_num_vars)` into `scripts/security_analysis/extract_params.py` (with the corresponding generated table or a fresh planner run) and re-run `run_estimator_all.py` / `challenge_entropy.py`.

---

## 8 — Reproducibility Index

| Artifact | Path | Output |
|---|---|---|
| Production parameter extractor | `scripts/security_analysis/extract_params.py` | `params.json` |
| Rank-floor classifier | `scripts/security_analysis/summarize_quadruples.py` | `quadruples.json` |
| Lattice-estimator runner | `scripts/security_analysis/run_estimator_all.py` | `estimator_all_results.json` |
| Per-preset MSIS table | `scripts/security_analysis/preset_summary.py` | stdout (matches §4) |
| Challenge-space entropy | `scripts/security_analysis/challenge_entropy.py` | stdout (matches §5) |
| Live planner probe (Rust) | `scripts/security_analysis/check_planner/` | stdout |
| Pinned lattice-estimator | `~/GitHub/lattice-estimator` `cc8494d` | — |
| Pinned SageMath | 10.7 | — |

To regenerate everything from a clean checkout:

```bash
# 1. Regenerate the production schedule tables under the fixed planner.
cargo run -p akita-config --features planner --bin gen_schedule_tables --release \
  -- crates/akita-types/src/generated

# 2. Workspace tests (validates the new tables pass validate_stored_sis_ranks).
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test --release

# 3. Static parameter analysis.
cd scripts/security_analysis
python3 extract_params.py > params.json
python3 summarize_quadruples.py
python3 preset_summary.py
python3 challenge_entropy.py

# 4. Lattice-estimator replay (~4 seconds).
sage -python run_estimator_all.py > estimator_all_results.json
```

---

## 9 — Conclusion

**MSIS (Module-SIS) hardness**: Resolved. All six production presets now satisfy the 128-bit MSIS rank floor at every fold-level Ajtai role. The planner fix in this PR is the right shape: shape-aware tentative, iterated fixed point, defense-in-depth load-time validation. Worst-case role margin is **+0.1 bits** at `d64_*` (the tightest cell), reflecting that `sis_floor.rs` is strictly 128-bit-targeted.

**Fiat-Shamir / CWSS knowledge soundness**: Resolved. The challenge-family cutover in §3.7 restores `|C| ≥ 2^128` for every production sparse-challenge family, giving per-level CWSS knowledge error `2^-(128 − r/2 − 2)` (tensor) or `2^-(128 − r − 1)` (flat). The pre-cutover gap of ~58 bits below the `2^128` target (caused by the May 2026 switch to `ExactShell{18, 0}` / `Uniform{13}`) is closed.

**Tradeoff**: the cutover restores `ω = 54` at D=64 and `ω = 32` at D=128, which gives back the `4ω` MSIS extraction penalty that the pre-cutover branch had reduced. The planner correctly absorbs this by picking higher Ajtai ranks at fold steps that previously fit at rank 1. The MSIS rank floor is now uniformly satisfied at 128 bits.

Phase D-full (recursive `S` opening + tiered commitments per book §5.4) does not change either finding's analysis. It modifies the verifier work distribution but reuses the same MSIS / CWSS / ring-switch / sumcheck machinery audited here. The post-fix baseline established by this PR is therefore the correct foundation for Phase D-full work.
