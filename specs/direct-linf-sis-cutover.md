# Spec: Direct coefficient-L∞ SIS cutover (full revert of L2 MSIS pricing)

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao, Cursor agent draft |
| Created       | 2026-06-26 |
| Status        | active |
| PR            | [#229](https://github.com/LayerZero-Labs/akita/pull/229) (groundwork); follow-up PRs TBD |
| Supersedes    | [`l2-msis-opnorm-folded-witness.md`](l2-msis-opnorm-folded-witness.md) (L2 table + certificate path); [`sis-euclidean-estimator.md`](sis-euclidean-estimator.md) (L2 table regen) |
| Book-chapter  | |

## Summary

[`l2-msis-opnorm-folded-witness.md`](l2-msis-opnorm-folded-witness.md) cut Akita over to an **L2 / Euclidean MSIS floor table** (#155) and planned an in-protocol **folded-witness L2 certificate** (slices S6–S10).
Investigation (2026-06-26) shows:

- The **L2 certificate path was never implemented** in Rust (`fold_l2`, `B_l2`, `ell_hat`, `carry_hat` do not exist in the tree). Only the spec and the dormant [`four_square.rs`](../crates/akita-types/src/sis/four_square.rs) helper remain.
- The **L2 MSIS table path is shipped** (`generated_sis_table/`, `min_secure_rank`, `collision_l2_sq` on `AjtaiKeyParams`) and is the current authority for A/B/D rank lookup via `collision_l2_sq_for_linf_envelope`.
- That L2 surrogate can **under-price A-role rank** relative to direct coefficient-`L∞` SIS (concrete counterexample below).

**Decision:** revert security pricing to **coefficient-`L∞` only**. Delete all L2-certificate planned work and retire the L2 MSIS table path after generated direct-`L∞` floors land. Do **not** implement the L2 certificate.

PR [#229](https://github.com/LayerZero-Labs/akita/pull/229) already fixes an independent bug (A-role must use the verifier-public folded-witness cap) and adds probe tooling. This spec is the durable cutover plan.

**Orthogonal shipped features (stay):**

- **Folded-witness `‖z‖_∞` tail bound + grind** ([`fold-linf-rejection.md`](fold-linf-rejection.md), #189): sizes `δ_fold` from `min(β_inf, t*)` with Fiat–Shamir rerolls. This is digit/width tightening, not SIS table pricing.
- **Operator-norm challenge rejection** (#207): prices A-role with `Γ` instead of `ω` when beneficial. Stays; only the **floor lookup norm** changes to direct `L∞`.

## Intent

### Goal

1. Make **direct coefficient-`L∞` SIS** the sole security model for **A-, B-, and D-role** module rank on every shipped `(SisModulusFamily, d)` preset.
2. **Delete** the L2 MSIS table stack, L2-certificate planned slices, and all hybrid `min(L2, direct-L∞)` migration code.
3. Keep **fold-linf grind** as a first-class **planner optimization lever** (tighter `δ_fold` when tail-bound policy applies; tunable grind acceptance with bounded prover rerolls).
4. Generate tables offline from a **pinned estimator checkout on the `quangvdao` fork** until a single targeted upstream PR lands later.

### Invariants

- **Verifier-public cap for fold collision.** A-role weak binding and `num_digits_fold` size against `folded_witness_public_linf_cap` (includes `min(β_inf, t*)` under [`FoldWitnessLinfCapPolicy::TailBoundWithGrind`](crates/akita-types/src/sis/norm_bound.rs)). Shipped in [#229](https://github.com/LayerZero-Labs/akita/pull/229).
- **Direct `L∞` authority for all SIS ranks.** `min_secure_rank` (or successor) consults only generated direct-`L∞` floors keyed by coefficient collision `B` and ring-column width. No `‖v‖₂² ≤ d·‖v‖∞²` conversion into an L2 table.
- **No L2 certificate.** No proof fields, sumchecks, or planner budgets for `B_l2`, slack limbs, or carry chains.
- **Fold-linf grind consistency.** Planner scoring, stored schedule expansion, prover reroll, and verifier nonce validation share the same `FoldWitnessLinfCapConfig` (policy, `t*`, `p_grind`).
- **Schedule/table consistency.** Stored `n_a` / `n_b` / `n_d` match recomputed direct-`L∞` floors at expansion; `generated_tables` and drift guards stay clean.
- **Verifier no-panic contract** on all lookup paths.

Protected by: `akita-types::sis` tests, `generated_tables`, `fold-linf-rejection` e2e/grind tests, profile-bench fp128 D64 matrix.

### Non-Goals

- In-protocol Euclidean norm certificates for the folded witness.
- Parallel L2 and `L∞` floor tables or feature-flagged dual schedules.
- Changing operator-norm predicate math (#207) or fold challenge sampling families.
- Runtime Sage inside planner or verifier.

## What is actually in the tree today

### Shipped: L2 MSIS table (to remove)

| Piece | Location |
|-------|----------|
| Euclidean floor modules | `crates/akita-types/src/sis/generated_sis_table/` |
| L2 table generator | `scripts/gen_sis_table.py` (`norm=l2`, BDGL16) |
| Rank lookup | `min_secure_rank`, `sis_max_widths` |
| L2 bucket derivation | `collision_l2_sq_for_linf_envelope`, `derived_collision_l2_sq_key`, `ceil_supported_collision` |
| A-role L2 collision | `committed_fold_collision_l2_sq` |
| Stored key field | `AjtaiKeyParams.collision_l2_sq` |
| Hybrid migration | `min(L2, direct-L∞)` in `op_norm_pricing.rs` |
| fp32 stopgap rectangles | `min_secure_rank_linf_direct` |

### Never shipped: L2 certificate (delete from specs + code)

| Piece | Status |
|-------|--------|
| S6–S10, S13 in [`l2-msis-opnorm-folded-witness.md`](l2-msis-opnorm-folded-witness.md) | Spec-only; **cancelled** |
| [`four_square.rs`](../crates/akita-types/src/sis/four_square.rs) | Dead helper for Lagrange slack; **delete on cutover** |
| `l2_sq_from_linf` | Used only for L2 table bridge; **delete** |

### Shipped: fold `‖z‖_∞` grind (keep + extend in planner)

From [`fold-linf-rejection.md`](fold-linf-rejection.md) and `norm_bound.rs`:

| Mechanism | Behavior |
|-----------|----------|
| `FoldWitnessLinfCapPolicy::TailBoundWithGrind` | `cap = min(β_inf, t*)`; grind nonce allowed |
| `FoldWitnessLinfCapPolicy::WorstCaseBetaOnly` | `cap = β_inf`; nonce must be 0 |
| Families today | TailBound: `ExactShell` @ D=64, `Uniform` @ D=128/256; others worst-case |
| `p_grind` | `FoldLinfProtocolBinding::grind_target_accept_prob()` → **1/8** shipped |
| Prover | Reroll fold challenge until realized `‖z‖_∞ ≤ t*` (hard cap) |
| Verifier | Range-check against `balanced_digit_max(lb, K)`; nonce replays FS only |
| Planner DP | `FoldWitnessLinfCapConfig::for_fold_level_scoring` threads the same `p_grind` into `num_digits_fold` and A-role collision via `folded_witness_public_linf_cap` |
| Schedule expand | `LevelParams::with_fold_linf_cap_config` recomputes policy at runtime |

**Planner optimization lever (correctness-preserving):**

- Tighter `t*` (via higher `p_grind` target in the union bound) → smaller `δ_fold` → narrower next-level width → lower proof bytes, at the cost of more expected grind rerolls.
- Shipped `p_grind = 1/8` gives ≤ 8 expected rerolls per fold level; production targets ~zero observed retries while keeping a sound margin.
- Cutover work should **document** grind expectations in planner diagnostics and allow **what-if scoring** over `p_grind` (env override already exists for op-norm sparse cap via `AKITA_OP_NORM_MAX_SPARSE_SAMPLES`; mirror for grind target in planner probes).
- Optional follow-up: DP search over per-level `TailBoundWithGrind` vs `WorstCaseBetaOnly` when the family certificate exists (today policy is family-determined, not per-geometry).

**Note:** `challenge_l2_sq_max` in the tail-bound formula is `max ‖c‖₂²` per block (concentration inequality input). It is **not** L2 MSIS pricing and **stays** after cutover.

### Counterexample: why L2 table must go (fp128 D64 dense nv24, L1)

Probe: [`scripts/probe_linf_sis_table.py`](../scripts/probe_linf_sis_table.py), `zeta=0`.

| Geometry | `B_A` | width | `n_A` (L2 table) | direct `L∞` bits @ rank | rank for 128b |
|----------|------:|------:|-----------------:|------------------------:|--------------:|
| dense L1 | 997,248 | 3,790 | 4 | 124.976 | 5 (170.236 bits) |

Pure direct-`L∞` DP: **+392 B** total on this CI shape; one-hot fp128 D64 unchanged.

## Gaps fixed in PR #229 (groundwork)

| Gap | Fix |
|-----|-----|
| A-role sized against raw `β_inf` | `folded_witness_public_linf_cap` |
| No audit helper for raw `B_A` | `committed_fold_collision_linf` |
| No offline `L∞` probe | `scripts/probe_linf_sis_table.py` |
| fp32 D256 dense root misses | interim `min_secure_rank_linf_direct` rectangles |
| Hybrid migration | `min(L2, direct-L∞)` (**delete in cutover**) |
| Schedule drift | regen under public cap |

## Evaluation

### Acceptance Criteria

**Delete L2 certificate + L2 table path**

- [ ] Remove [`four_square.rs`](../crates/akita-types/src/sis/four_square.rs) and all exports/tests.
- [ ] Remove `committed_fold_collision_l2_sq`, `l2_sq_from_linf`, `collision_l2_sq_for_linf_envelope`, `derived_collision_l2_sq_key`, and L2-only `ceil_supported_collision` ladder tied to `generated_sis_table/`.
- [ ] Remove `crates/akita-types/src/sis/generated_sis_table/` and `scripts/gen_sis_table.py` L2 path (or replace script with `L∞`-only generator).
- [ ] Rename `AjtaiKeyParams.collision_l2_sq` → `collision_linf` (full cutover, no alias).
- [ ] Delete `min_secure_rank_linf_direct`, `exact_linf_from_l2_sq` fallback, hybrid `match (l2, direct_linf)`.
- [ ] Mark [`l2-msis-opnorm-folded-witness.md`](l2-msis-opnorm-folded-witness.md) certificate slices S6–S10/S13 **cancelled**; set spec `Superseded-by:` this file.
- [ ] Mark [`sis-euclidean-estimator.md`](sis-euclidean-estimator.md) `Superseded-by:` this file.

**Direct `L∞` floors**

- [ ] `scripts/gen_sis_linf_table.py` (+ stitcher) generates max-width tables for all `(family, d)` in `supports_family_dimension`.
- [ ] Checked-in `generated_sis_linf_table/`; `min_secure_rank` looks up coefficient-`L∞` collision `B` directly.
- [ ] Golden cells in `scripts/sis_golden/` (or sibling) for `norm=oo` replay.
- [ ] Estimator pinned to **`quangvdao/lattice-estimator`** standalone branch; metadata records fork URL + SHA. One targeted upstream PR later; malb/lattice-estimator [#215](https://github.com/malb/lattice-estimator/pull/215) and [#216](https://github.com/malb/lattice-estimator/pull/216) are **closed without merge** (confirmed 2026-06-26).

**Planner + schedules**

- [ ] All roles (A/B/D) price through direct `L∞` floors.
- [ ] Regenerate `akita-schedules`; `generated_tables` clean.
- [ ] `fp32_dense_planner_diag` un-ignored or promoted to golden subset.
- [ ] Fold-linf grind documented in planner diagnostics; what-if `p_grind` scoring hook for optimization studies.
- [ ] fp128 D64 profile-bench rows match planner `total_bytes` within report tolerances.

### Testing Strategy

- Unit tests: `min_secure_rank` monotonicity, fp128 D64 dense L1 → rank 5, table miss → `None`.
- Delete/update tests that assert L2 bucket derivation or `four_squares`.
- Extend `scripts/sis_golden/check.py` for `L∞` cells.
- Keep fold-grind e2e and nonce validation tests unchanged.
- CI: clippy, doc build, `generated_tables`, profile-bench.

### Performance

- Small proof-size **increases** where L2 under-ranked (dense fp128 D64 nv24: +392 B).
- Fold-linf grind tuning trades prover reroll latency for smaller `δ_fold`; default `p_grind = 1/8` should keep retries rare.

## Design

### Architecture (target)

```
fold challenge + witness norms
        │
        ▼
fold_witness_linf_cap_policy ──► TailBoundWithGrind | WorstCaseBetaOnly
        │
        ▼
folded_witness_public_linf_cap ──► δ_fold, Golomb cap, A-role B_A input
        │                              (grind lever: t* from p_grind)
        ▼
role collision B (coefficient L∞) ──► min_secure_rank ──► generated_sis_linf_table/
        ▲
        ├─ A: committed_fold_collision_linf (Lemma 7 + public cap)
        ├─ B: 2^lb − 1 opening digit diff
        └─ D: 2^lb − 1 opening digit diff
```

### Table generation

| Parameter | Value |
|-----------|-------|
| Estimator remote | `quangvdao/lattice-estimator` (standalone branch; not malb until targeted PR lands) |
| `norm` | `oo` |
| `red_cost_model` | ADPS16 |
| `red_shape_model` | lgsa |
| `zeta_candidates` | `(0,)` (or explicit search once fork fix lands) |
| Collision key | coefficient `B` per ring row; `m = width · d` |
| `target_bits` | 128 |
| Grid | `COEFF_LINF_BUCKETS` + planner anchor points |

```bash
sage -python scripts/stitch_generated_sis_linf_table.py --jobs 6 \
  --estimator-path <quangvdao-lattice-estimator-checkout>
```

### Cutover phases

| Phase | Deliverable |
|-------|-------------|
| **0 — Groundwork** | #229 merged (public cap, probe, interim hybrid) |
| **1 — Estimator fork** | `quangvdao/lattice-estimator` branch with targeted `L∞` fixes; pin submodule |
| **2 — Generator + golden** | `gen_sis_linf_table.py`, stitcher, golden cells |
| **3 — Runtime** | `generated_sis_linf_table/`, unified `min_secure_rank` on `B` |
| **4 — Delete L2** | Remove L2 table, certificate artifacts, hybrid/stopgaps, supersede old specs |
| **5 — Planner** | Wire all roles; fold-grind planner diagnostics / `p_grind` what-if |
| **6 — Schedules** | Regen tables, profile validation, un-ignore diagnostics |

### Alternatives considered

| Alternative | Why rejected |
|-------------|--------------|
| Keep L2 table for B/D only | User mandate: `L∞` only; conversion is still wrong estimator instance |
| Implement L2 certificate | Cancelled; `L∞` digit cap + grind already bound witness |
| `min(L2, direct-L∞)` hybrid | Under-ranks A-role; interim only |
| malb/lattice-estimator #215/#216 as-is | Closed without merge; use `quangvdao` fork |

## Documentation

- This spec owns the cutover until a book security chapter exists.
- Supersede [`l2-msis-opnorm-folded-witness.md`](l2-msis-opnorm-folded-witness.md) and [`sis-euclidean-estimator.md`](sis-euclidean-estimator.md) headers (`Superseded-by:` this file; certificate slices marked cancelled).
- Keep [`fold-linf-rejection.md`](fold-linf-rejection.md) as the grind/tail-bound reference; cross-link planner lever section here.
- Update `scripts/sis_golden/README.md` for `L∞` regen.

## Execution

### Next PRs (after #229)

1. Open / pin `quangvdao/lattice-estimator` branch; one focused fix PR to malb when ready.
2. Land `gen_sis_linf_table.py` + `generated_sis_linf_table/`.
3. Switch all rank lookup to direct `B`; delete L2 table modules and `four_square.rs`.
4. Regenerate schedules; profile fp128 D64; extend planner grind diagnostics.

## References

- [#229](https://github.com/LayerZero-Labs/akita/pull/229) — groundwork
- [`fold-linf-rejection.md`](fold-linf-rejection.md) — shipped grind/tail bound (#189)
- [`weak-binding-norm-fix.md`](weak-binding-norm-fix.md) — Lemma 7 / public cap
- [`scripts/probe_linf_sis_table.py`](../scripts/probe_linf_sis_table.py)
- malb/lattice-estimator [#215](https://github.com/malb/lattice-estimator/pull/215), [#216](https://github.com/malb/lattice-estimator/pull/216) — **closed** (not merged)
- Local probe report (never commit): `FP128-LINF-PLANNER-REPORT-NEVER-COMMIT.md`
