# Spec: Root-Level Tiered B Commitment

| Field      | Value                                    |
|------------|------------------------------------------|
| Author(s)  |                                          |
| Created    | 2026-05-19                               |
| Status     | proposed                                 |
| PR         |                                          |

## Summary

The Akita verifier's dominant per-level cost is `compute_setup_contribution`'s α-evaluation of cells in the shared SIS commitment matrix, and at the root that cost is governed by the **B-side** rectangle of width `outer_width = max_group_polys · n_a · num_digits_open · num_blocks` (see §7 of [`specs/optimized_verifier.md`](specs/optimized_verifier.md) and the captured numbers in [`docs/onehot-d32-nv32-matrix-sizes.md`](docs/onehot-d32-nv32-matrix-sizes.md)). This spec adds a **two-tier** outer commitment at the root only: for a splitting factor `f > 1`, the prover splits `t_hat^(g)` into `f` equal contiguous column chunks against the *same* smaller matrix `B'`, gadget-decomposes each `u_i = B' · t_i` into a balanced i8 witness `uhat_i`, and binds them via a new SIS matrix `F` so the public commitment becomes `u_final = F · (uhat_1 ‖ … ‖ uhat_f)`. The proof carries only `u_final`; the verifier's B-side α-eval rectangle shrinks from `n_b × outer_width` to `n_b' × (outer_width / f)`.

`f = 1` keeps the **current legacy protocol unchanged** (no tiering, no extra rows, no extra witness, byte-identical transcript). Tiering only activates when the planner explicitly selects `split_factor > 1`. Scope is **root level only**; recursive levels are untouched.

## Intent

### Goal

For the root level, support two protocol paths chosen by `LevelParams.split_factor`:

- **Legacy path (`split_factor == 1`).** Identical to today: `u = B · t_hat`, M-row layout `consistency | public | D | B | A`, `RingCommitment.u.len() == lp.b_key.row_len()`, no `uhat` witness, no F matrix. This is the *only* path the runtime must use when `split_factor == 1`. No algebraic collapse, no shared code branch from the tiered path — explicit dispatch.
- **Tiered path (`split_factor > 1`).** Per opening point `g` and chunk `i ∈ {1, …, f}`:

  ```text
  u_i^(g)      = B' · t_i^(g)                     (R_q^{n_b'})
  uhat_i^(g)   = balanced_decompose_i8(u_i^(g))   (depth δ_outer, basis 2^{outer_log_basis})
  uhat_concat  = uhat_1 ‖ … ‖ uhat_f              (R_q^{n_b' · f · δ_outer})
  u_final^(g)  = F · uhat_concat^(g)              (R_q^{n_F})  ← only public artifact
  ```

  Each tier-1 row in M encodes the **single** relation

  ```text
  B' · t_i^(g) − G · uhat_i^(g) = 0
  ```

  i.e. the `B' · t_i` setup-matrix half and the `−G · uhat_i` structured-gadget half **share the same row index and same row weight**. They are not separate M row blocks.

  Each F row in M encodes `F · uhat_concat^(g) − u_final^(g) = 0`.

The verifier's B-side α-eval rectangle in `compute_setup_contribution` drops from `n_b × outer_width` to `n_b' × (outer_width / f)`, with the same physical `B'` α-eval shared across all `f` chunk patterns.

Key abstractions / APIs touched (full surface listed in [Execution](#execution)):

- `LevelParams` gains `split_factor`, `outer_log_basis`, `num_digits_outer`, `f_key`. Helpers `is_tiered_root()`, `outer_commitment_rows()`, `b_prime_rows()`, `b_prime_width()`, `full_outer_width()` keep the full `t_hat` width reachable even when `b_key.col_len()` is the per-chunk width.
- `AkitaExpandedSetup.shared_matrix` is extended with an `F`-row block alongside A/B/D. `AkitaSetupSeed.max_stride` envelope grows; **`AkitaSetupSeed` has no `max_rows` field and this spec does not introduce one** (allocation capacity is computed during setup generation, not stored on the seed).
- `AkitaCommitmentHint` gains `outer_digits: Vec<FlatDigitBlocks<D>>` for the tiered path (one entry per opening point). For `split_factor == 1` this stays empty so legacy commits remain unchanged.
- Prover commit (`commit_with_params`), prover relation builder (`compute_r_split_eq`, `repeated_b_commitment_rows`), prover M-witness builder (`ring_switch_build_w` / `build_w_coeffs`), verifier replay (`prepare_ring_switch_row_eval`, `compute_setup_contribution`), planner (`crates/akita-planner/src/search.rs`), and config (`crates/akita-config/src/proof_optimized.rs`) all branch on `lp.is_tiered_root()`.

### Invariants

- **Explicit legacy dispatch.** For `split_factor == 1` the runtime must use today's code paths and produce byte-identical transcripts, schedule cache entries, and proofs relative to pre-tiering main. Protected by a snapshot test in `crates/akita-planner` that compares the existing preset schedules at `f = 1` against the checked-in golden bytes, plus a serialization test that re-runs an existing E2E proof at `f = 1` and asserts byte-equality.
- **Tiered M-row count is single-counted.**

  ```text
  m_row_count(tiered) =
      1
    + num_public_rows
    + n_d
    + f · n_b' · num_points          // tier-1 rows (B' setup + −G structured share these rows)
    + n_F · num_points               // F rows
    + n_a
  ```

  No separate "gadget consistency" row block. Protected by a unit test on `LevelParams::m_row_count` that asserts the formula above for several `(f, n_b', n_F, num_points)` shapes.
- **Tiered `y` layout.**

  ```text
  consistency: 0
  public:      y_ring public rows         (unchanged)
  D:           v rows                     (unchanged)
  tier1:       all zero
  F:           u_final rows               (RingCommitment.u flattened into this slice)
  A:           zero
  ```

  `RingCommitment.u` is *not* placed in the tier1 slice. Protected by an equality test inside `generate_y` (`crates/akita-prover/src/protocol/quadratic_equation.rs`) and a verifier-side replay test.
- **`uhat` is a real M-column witness segment, not just a hint.** Length `num_points · n_b' · f · δ_outer` ring elements per root call. Lives in the folded witness so all witness-length computations, sumcheck domain, ZK offsets, and `planned_w_ring_element_count` must update. Protected by the materialised-M reference test in `crates/akita-verifier`.
- **Outer digits are balanced i8.** `outer_log_basis ∈ {2, 3, 4, 5, 6}` only (current `FlatDigitBlocks<D>` / `[[i8; D]]` storage and `decompose_rows_i8_into` support only `log_basis ≤ 6`). A larger basis requires a new digit-storage type first; this spec forbids it.
- **`num_digits_outer` is derived from a proven bound on `u_i = B' · t_i`.** Default and safest first implementation: full-field decomposition via `compute_num_digits_full_field(field_bits, outer_log_basis)` from [`crates/akita-types/src/layout/digit_math.rs`](crates/akita-types/src/layout/digit_math.rs). An optimised choice may use `num_digits_for_bound(log_bound, field_bits, outer_log_basis)` only if it proves a tighter bound that covers every reachable `u_i` value, including any ZK blinding contribution that flows through `B' · t_i`.
- **Full `t_hat` width stays reachable.** `lp.b_key.col_len()` in the tiered path equals `outer_width / f`, not `outer_width`. Any code that needs the full `t_hat` length must call `lp.full_outer_width()` (or equivalent), not `lp.b_key.col_len()`. Protected by static code review and by a regression test that runs the legacy path with the new helpers in place.
- **Soundness via two-case binding.** A binding break of `(t_hat, u_final)` reduces to either an `F` SIS collision (when the two openings disagree on `uhat_concat`) or a `B'` SIS collision (when they agree on `uhat_concat` but disagree on some `t_i`). See [Design §6](#6-soundness-two-case-two-tier-binding-argument). `B'` is **not** sized as the augmented matrix `[B' | −G]` unless a cryptographer explicitly decides otherwise.
- **No change to W/Z/r-tail/structured contributions or recursive levels.** §4–§6, §8, §9 of [`specs/optimized_verifier.md`](specs/optimized_verifier.md) and every `level ≥ 1` proof are byte-identical to today. Protected by existing per-block evaluator tests and recursive-level snapshot tests.

### Non-Goals

- **Tiering the D matrix.** Same construction will apply once we tackle D; explicitly deferred.
- **Recursive-level tiering.** Recursive levels have non-pow2 `block_len` and small absolute rectangles; deferred.
- **A new digit-storage type for `outer_log_basis > 6`.** This spec assumes current i8 balanced decomposition.
- **Multi-tier (>2) commitments.** Two tiers only.
- **Changes to W, Z, r-tail, or ZK blinding evaluators.** Today's structured-slice evaluators stay byte-identical. (ZK blinding *offsets* shift to account for the new `uhat` segment — see [Design §7](#7-zk-interaction).)
- **Changes to the sumcheck or recursion protocol.** Stages 1 and 2 are untouched in their logic; their sumcheck domain grows because M grows, but that follows automatically from the new witness length.
- **Proof-size reductions.** `u_final.len() = n_F` may be smaller than `u.len() = n_b`, but no proof-size *guarantee* is required. Optimization target is verifier time.

## Evaluation

### Acceptance Criteria

- [ ] `LevelParams` carries `split_factor`, `outer_log_basis`, `num_digits_outer`, `f_key`, plus the helper methods listed in [Design §4](#4-levelparams-and-helper-apis). `is_tiered_root()` returns `self.split_factor > 1`.
- [ ] All runtime protocol code paths dispatch on `lp.is_tiered_root()`. With `split_factor == 1` no tiered code runs and the proof is byte-identical to the pre-tiering baseline on the existing presets.
- [ ] `commit_with_params` returns a `RingCommitment` whose `u.len() == lp.outer_commitment_rows()` (i.e. `n_b` for legacy, `n_F` for tiered) and an `AkitaCommitmentHint` whose `outer_digits` is empty for legacy and contains `uhat_concat^(g)` per point for tiered.
- [ ] `compute_setup_contribution` in the tiered path matches a materialised-M fixture bit-for-bit (W / Z / r-tail / ZK halves still byte-identical to today, plus the new tier-1 and F halves).
- [ ] Planner only emits `split_factor > 1` candidates that satisfy `split_factor | full_outer_width`; for each emitted candidate, `n_b' ≥ min_rank_for_secure_width(family, D, t_inf_bound, outer_width / split_factor)` and `n_F ≥ min_rank_for_secure_width(family, D, uhat_inf_bound, n_b' · split_factor · num_digits_outer)`.
- [ ] `outer_log_basis ≤ 6` and `num_digits_outer ≥ compute_num_digits_full_field(field_bits, outer_log_basis)` (or matches a separately-proven tighter bound). Rejected at planner-emit time and at `LevelParams` validation.
- [ ] Recursive-level proofs (`level ≥ 1`) are byte-identical before and after this change.
- [ ] `cargo fmt -q && cargo clippy --all --message-format=short -q -- -D warnings && cargo test` is green with default features; `cargo test -p akita-pcs --release --features zk` is green.

### Testing Strategy

Existing tests that must continue to pass:

- [`crates/akita-pcs/tests/multipoint_batched_e2e.rs`](crates/akita-pcs/tests/multipoint_batched_e2e.rs) — root and recursive multipoint paths, both with default features and `--features zk`.
- `setup_contribution_handles_nonidentity_multigroup_routing` in [`crates/akita-verifier/src/protocol/slice_mle/setup_contribution.rs`](crates/akita-verifier/src/protocol/slice_mle/setup_contribution.rs) — extended to also cover the tiered path.
- Recursive-level proofs in `crates/akita-pcs/examples/recursion.rs`.

New tests to add:

- **`f = 1` byte-identical legacy test.** In `crates/akita-pcs/tests/`, run an existing E2E (e.g. the `multipoint_batched_e2e` workload) twice — once on pre-tiering main, once on this branch with `split_factor == 1` everywhere — and assert the serialized proof bytes match. Cheaper alternative: a snapshot bytes file checked in alongside the test.
- **Tiered `commit` unit test.** In `crates/akita-prover/src/api/commitment.rs`, assert `u_final = F · concat_i(balanced_decompose_i8(B' · t_i))` for `f = 2, 4, 8`, with `outer_log_basis ∈ {4, 5, 6}` and `num_digits_outer = compute_num_digits_full_field(field_bits, outer_log_basis)`.
- **Materialised-M tiered fixture.** In `crates/akita-verifier/src/protocol/slice_mle/setup_contribution.rs`, build a small `(f, n_b', n_F, δ_outer, num_points)` shape, materialise the full M with the tiered row layout from [Design §3](#3-corrected-m-row-and-y-layout-for-tiered-path), and compare against the fused `compute_setup_contribution` output.
- **Wrong `u.len()` rejection.** Verifier rejects a proof where `commitment.u.len() != lp.outer_commitment_rows()`.
- **Multipoint and grouped-root E2E at `f > 1`.** Extend `multipoint_batched_e2e.rs` with explicit `f = 2, 4, 8` cases at `num_points = 2`, `group_size = 2`, both with and without `zk`.
- **Planner schedule encoding.** Snapshot test that the planner schedule key (used as a cache key in `crates/akita-config/`) encodes `split_factor`, `outer_log_basis`, `num_digits_outer`, `n_F`, `n_b'` so two distinct tierings cache-separately.
- **SIS-floor regression.** Negative test asserting the planner rejects a candidate where `n_b'` or `n_F` would slip below `min_rank_for_secure_width`.

### Performance

Primary metric: verifier α-eval count per call to `compute_setup_contribution` at the root level for the canonical `onehot_d32_nv32` singleton commit. The cost model below is exactly the per-row dispatch in `slice_inner_sum` (the inner loop of `compute_setup_contribution`); it is **not** a full verifier wall-clock model and should not be quoted as such.

Baseline parameters from [`docs/onehot-d32-nv32-matrix-sizes.md`](docs/onehot-d32-nv32-matrix-sizes.md) at level 0, `num_points = num_claims = 1`, `max_group_polys = 1`:

- `n_a = 3`, `n_b = 2`, `n_d = 2`, `D = 32` (`field_bits = 128`, Q128 family).
- `num_blocks = 2048`, `δopen = 64`, `block_len = 65 536`, `δcommit = 1`.
- `n_cols_w = num_claims · num_blocks · δopen = 131 072`.
- `n_cols_t = max_group_polys · n_a · num_blocks · δopen = 393 216`  (baseline B-side width).
- `z_range = block_len · δcommit = 65 536`.

Tiering parameters (Q128, balanced i8 decomposition, full-field bound):

- `n_b' = n_b = 2` (illustrative; planner may pick smaller via SIS floor on the smaller chunk width).
- `n_F = 2` (illustrative).
- `outer_log_basis = 6` (largest value the current i8 storage admits — see [Design §5](#5-outer-digit-decomposition)).
- `num_digits_outer = compute_num_digits_full_field(128, 6) = ⌈128 / 6⌉ = 22`.
- `F_width = n_b' · f · num_digits_outer = 2 · f · 22 = 44 · f`.
- F α-eval extra cells per call: `n_F · F_width = 2 · 44 · f = 88 · f`.

Per-row α-eval cell count is `e_max(row) = max(e_w, e_t, e_z)` (see `slice_inner_sum`), gated by `row < n_d` / `n_b'` / `n_a`. Total cells × `D` gives total α-eval ops; F α-evals add on top.

| f  | `e_max` rows 0–1 (`<n_b'`) | `e_max` row 2 (Z only) | F extra cells | total α-eval ops (× D) | speedup vs legacy |
|---:|---:|---:|---:|---:|---:|
| 1 (legacy) | `max(131 072, 393 216, 65 536) = 393 216` | 65 536 | 0 | `(2·393 216 + 65 536) · 32 = 27 262 976` | 1.00× |
| 2  | `max(131 072, 196 608, 65 536) = 196 608` | 65 536 | `2·2·2·22 = 176` | `(2·196 608 + 65 536 + 176) · 32 = 14 685 696` | 1.86× |
| 4  | `max(131 072, 98 304, 65 536) = 131 072` | 65 536 | `2·2·4·22 = 352` | `(2·131 072 + 65 536 + 352) · 32 = 10 497 024` | 2.60× |
| 8  | `max(131 072, 49 152, 65 536) = 131 072` | 65 536 | `2·2·8·22 = 704` | `(2·131 072 + 65 536 + 704) · 32 = 10 508 288` | 2.59× |
| 16 | `max(131 072, 24 576, 65 536) = 131 072` | 65 536 | `2·2·16·22 = 1 408` | `(2·131 072 + 65 536 + 1 408) · 32 = 10 530 816` | 2.59× |

Reading this micro-model:

- The B-side α-eval rectangle drops linearly in `f` until the **W-side** (fixed at `n_cols_w = 131 072` per row times `n_b' = 2` rows) becomes the new bottleneck. That happens at `f = 4` for this preset, giving an upper-bound speedup near **2.6×** on this single span. Larger `f` adds prover and witness cost (see below) for negligible verifier savings.
- The numbers above are for the singleton onehot_d32_nv32 root only. **Recompute** for `AKITA_NUM_POLYS=4 AKITA_GROUP_SIZE=2` (multi-group) and multipoint cases before quoting wall-clock claims. Use [`docs/onehot-d32-nv32-g2-matrix-sizes.md`](docs/onehot-d32-nv32-g2-matrix-sizes.md) as the multi-group baseline.
- **Caveats explicitly out of this micro-model.**
  - Stage-2 sumcheck domain grows because M grows by `uhat_len = num_points · n_b' · f · δ_outer` columns and by `f · n_b' · num_points + n_F · num_points − n_b · num_points` rows. Some verifier work scales with these.
  - Tier-1 structured `−G · uhat_i` aggregation adds O(`f · n_b' · num_points · δ_outer`) verifier ops, separate from `compute_setup_contribution`.
  - Prover commit grows by `f` extra `mat_vec_mul_ntt_single_i8` calls against `B'` plus one against `F`, plus `f · n_b'` balanced decompositions.
  - Hint size grows by `uhat_concat` (per point: `n_b' · f · δ_outer` ring elements of i8 digits).
- For `f = 4` at this preset, total proof bytes shrink slightly if `n_F < n_b`; otherwise unchanged. Setup matrix grows by `n_F · F_width = 2 · 88 = 176` ring cells — negligible.

Benchmarks to track this:

- `AKITA_MODE=onehot_d32 AKITA_NUM_VARS=32 cargo run --release --example profile` — verifier wall-clock; record the `setup_contribution` tracing span at `f ∈ {1, 2, 4, 8}`.
- New microbench in `crates/akita-pcs/benches/` that times `compute_setup_contribution` directly across `f ∈ {1, 2, 4, 8, 16}` against this table.
- Repeat both with `AKITA_NUM_POLYS=4 AKITA_GROUP_SIZE=2` to validate batched savings.

## Design

### Architecture

#### 1. Two protocol paths, explicit dispatch

```text
if lp.split_factor == 1:
    use today's legacy implementation exactly
else:
    use tiered implementation (this spec)
```

Every protocol surface that previously assumed `B · t_hat = u` adds an explicit branch on `lp.is_tiered_root()`. The legacy path is *not* re-implemented as a special case of the tiered path: it is the same code paths in use today, untouched, gated on `split_factor == 1`. This preserves the schedule cache, the transcript bytes, the M-row count, the witness layout, and the proof bytes of every existing configuration that ends up at `f = 1`.

The planner may emit `split_factor == 1` candidates (e.g. as the baseline against which it scores `f ∈ {2, 4, 8, …}`), but the runtime protocol code reads `lp.split_factor` and chooses one of two distinct branches.

#### 2. Tiered commitment construction (`split_factor > 1`)

Per opening point `g`:

1. **Inner commit** (unchanged from today). `t_hat^(g)` is the gadget decomposition of `t^(g) = A_inner · poly^(g)`, stored in the existing `FlatDigitBlocks<D>` shape. Length `outer_width = max_group_polys · n_a · num_digits_open · num_blocks` ring elements. Built by `poly.commit_inner_witness` ([`crates/akita-prover/src/backend/dense.rs`](crates/akita-prover/src/backend/dense.rs), `commit_inner_witness`).

2. **Chunk.** Define `t_i^(g) = t_hat^(g)[ (i−1) · W' : i · W' ]` for `i ∈ {1, …, f}`, `W' = outer_width / f`. This is a *column-window view* into the same `FlatDigitBlocks`; no copying. Planner enforces `f | outer_width`.

3. **B' multiply.** `u_i^(g) := B' · t_i^(g) ∈ R_q^{n_b'}` using the existing `mat_vec_mul_ntt_single_i8` kernel ([`crates/akita-prover/src/kernels/linear.rs`](crates/akita-prover/src/kernels/linear.rs)). The same physical `B'` (a column-window view of the shared SIS matrix's B-row block) is reused for every `i`.

   **Risk to resolve at implementation time:** the existing `mat_vec_mul_ntt_single_i8` kernel may assume a prefix view with fixed stride from offset zero. A chunk multiply that takes the `i`-th column window of `t_hat` against the leading `W'` columns of `B` may need either (a) a new entry point that accepts an input-side offset and length, or (b) a thin wrapper that copies the chunk into a contiguous buffer. The handoff calls this out explicitly; the implementing agent must verify the kernel surface before settling on (a) vs (b).

4. **Balanced i8 decompose.** `uhat_i^(g) := balanced_decompose_i8(u_i^(g))` with depth `δ_outer = num_digits_outer` and basis `2^{outer_log_basis}`. `outer_log_basis ≤ 6` (see [§5](#5-outer-digit-decomposition)). Uses the existing `decompose_rows_i8_into` path. The gadget identity `u_i^(g) = G · uhat_i^(g)` holds exactly, where `G = (1, 2^{outer_log_basis}, 2^{2 · outer_log_basis}, …)`.

5. **F multiply.** `uhat_concat^(g) := uhat_1^(g) ‖ … ‖ uhat_f^(g)` is concatenated (no per-chunk gaps) into a single `FlatDigitBlocks<D>` of length `n_b' · f · δ_outer` ring elements; the prover then runs a second `mat_vec_mul_ntt_single_i8` against `F` to get

   ```text
   u_final^(g) := F · uhat_concat^(g) ∈ R_q^{n_F}.
   ```

6. **Proof artifacts.** `RingCommitment { u: u_final }` is the public commitment (`u.len() = n_F`). `AkitaCommitmentHint.outer_digits[g] = uhat_concat^(g)` is added to the prover hint.

The verifier-checked relations per opening point are:

```text
B' · t_i^(g) − G · uhat_i^(g) = 0      for i ∈ {1, …, f}
F · uhat_concat^(g) − u_final^(g) = 0
```

`u_i^(g)` is **virtual**: never materialised; recovered implicitly as `G · uhat_i^(g)` when needed.

#### 3. Corrected M row and y layout for tiered path

For `split_factor > 1`, the M row layout is:

```text
consistency (1) | public (num_public_rows) | D (n_d) | tier1 (f · n_b' · num_points) | F (n_F · num_points) | A (n_a)
```

Each tier-1 row encodes the single relation `B' · t_i − G · uhat_i = 0`. It has two physical contributions:

- A setup-matrix contribution from the `B' · t_i` half (α-evaluated against the `B'` row block of `shared_matrix`).
- A structured contribution from the `−G · uhat_i` half (no SIS scan; depth-`δ_outer` gadget weights against the `uhat` witness segment).

These two contributions **share the same row index and the same row weight `eq_tau1[tier1_start + (g · f · n_b') + (i · n_b') + r]`**. They are not separate logical M rows. Implementations may use separate evaluator helpers for clarity, but the row weights and row count are single-counted.

Each F row encodes `F · uhat_concat − u_final = 0` and is α-evaluated against `F`'s row block of `shared_matrix`.

`m_row_count` becomes:

```text
m_row_count(tiered) =
    1
  + num_public_rows
  + n_d
  + f · n_b' · num_points
  + n_F · num_points
  + n_a
```

`y` (the relation RHS) layout for the tiered path:

```text
consistency: 0
public:      y_ring public rows         (unchanged)
D:           v rows                     (unchanged)
tier1:       all zero
F:           u_final rows               (RingCommitment.u flattened into this slice, n_F · num_points entries)
A:           zero
```

`RingCommitment.u` is placed in the **F slice**, not the tier1 slice. `generate_y` must branch on `lp.is_tiered_root()` to produce this layout.

For `split_factor == 1`, the M row layout, `m_row_count`, and `y` layout are exactly today's.

#### 4. `LevelParams` and helper APIs

New fields on `LevelParams` (root-only meaning; zero-meaning at recursive levels):

```rust
pub struct LevelParams {
    // existing fields ...
    pub a_key: AjtaiKeyParams,
    pub b_key: AjtaiKeyParams,
    pub d_key: AjtaiKeyParams,

    // new root-tiering fields
    pub split_factor: usize,        // 1 = legacy, >1 = tiered
    pub outer_log_basis: u32,       // must be in {2..=6} when split_factor > 1
    pub num_digits_outer: usize,    // derived from bound; full-field default
    pub f_key: AjtaiKeyParams,      // F = n_F × (n_b' · split_factor · num_digits_outer)
}
```

Required helpers (names indicative, not contractual):

```rust
fn is_tiered_root(&self) -> bool {
    self.split_factor > 1
}

fn outer_commitment_rows(&self) -> usize {
    if self.is_tiered_root() {
        self.f_key.row_len()        // = n_F
    } else {
        self.b_key.row_len()        // = n_b (legacy)
    }
}

fn b_prime_rows(&self) -> usize { self.b_key.row_len() }

fn b_prime_width(&self) -> usize {
    // Tiered: b_key.col_len() already shrunk to outer_width / f.
    self.b_key.col_len()
}

fn full_outer_width(&self) -> usize {
    // Full t_hat column count; equal to b_prime_width() * split_factor when
    // tiered, equal to b_key.col_len() when legacy.
    self.b_prime_width() * self.split_factor
}
```

**Critical naming pitfall.** Today `outer_width()` is equivalent to `b_key.col_len()`. If the tiered path makes `b_key.col_len()` equal to `outer_width / f`, every existing call site that meant the *full* `t_hat` width must be migrated to `full_outer_width()`. The implementing agent must audit every call to `lp.outer_width()` / `lp.b_key.col_len()` and decide per call site whether the legacy or chunked width is intended. Failing this audit will silently break the legacy path because `b_key.col_len()` will collapse on tiered LevelParams that the planner emits even when subsequent rounds care about the full `t_hat`.

#### 5. Outer digit decomposition

The current digit-storage and balanced-decompose path (`FlatDigitBlocks<D>`, `[[i8; D]]`, `decompose_rows_i8_into`) supports `1 ≤ log_basis ≤ 6` only. Therefore:

```text
outer_log_basis ∈ {2, 3, 4, 5, 6}
```

A larger basis requires a new non-i8 digit-storage type, which is out of scope for this spec.

`num_digits_outer` must be derived from a *proven* bound on every reachable `u_i = B' · t_i`. The safest first implementation is full-field decomposition:

```text
num_digits_outer = compute_num_digits_full_field(field_bits, outer_log_basis)
```

(`compute_num_digits_full_field` lives in [`crates/akita-types/src/layout/digit_math.rs`](crates/akita-types/src/layout/digit_math.rs)). For Q128:

| `outer_log_basis` | `num_digits_outer` (full-field, Q128) |
|---:|---:|
| 6 | 22 |
| 5 | 26 |
| 4 | 32 |
| 3 | 43 |
| 2 | 64 |

An optimised implementation may use `num_digits_for_bound(log_bound, field_bits, outer_log_basis)` *only* if it independently proves a smaller bound on `B' · t_i` (including any ZK blinding contribution that flows through the B' multiply). Such optimisations are deferred to a follow-up spec; the first implementation should use the full-field default.

The bound on `uhat_i` entries — needed for sizing `F`'s SIS rank — is `‖uhat_i‖_∞ ≤ ⌊2^{outer_log_basis} / 2⌋` (balanced i8 convention). The implementer must decide carefully whether the SIS floor table uses `2^{outer_log_basis}`, `2^{outer_log_basis} − 1`, or `2^{outer_log_basis − 1}` as the collision-inf input. Pick one convention, document it, and apply it consistently in [`crates/akita-planner/src/sis_security.rs`](crates/akita-planner/src/sis_security.rs).

#### 6. Soundness: two-case two-tier binding argument

A binding break of the tiered public commitment exhibits two valid witnesses for the same `u_final`:

```text
B' · t_i      = G · uhat_i        for all i
F  · uhat     = u_final

B' · t_i'     = G · uhat_i'       for all i
F  · uhat'    = u_final
```

(`uhat` denotes `uhat_concat` for brevity.) Subtracting yields

```text
B' · Δt_i     = G · Δuhat_i       for all i
F  · Δuhat    = 0
```

Two cases:

1. **`Δuhat ≠ 0`.** Then `F · Δuhat = 0` is a non-trivial SIS collision for `F` with input bound `‖Δuhat‖_∞ ≤ 2 · ⌊2^{outer_log_basis}/2⌋` and width `n_b' · f · δ_outer`. Sized by `min_rank_for_secure_width(family, D, uhat_inf_bound, n_b' · f · δ_outer)`.
2. **`Δuhat = 0`.** Then `G · Δuhat_i = 0`, so `B' · Δt_i = 0` for every `i`. Since the prover claims two distinct openings, some `i` has `Δt_i ≠ 0`, giving a non-trivial SIS collision for `B'` with input bound `‖Δt_i‖_∞ ≤ 2 · t_inf_bound` (where `t_inf_bound` is the existing inner gadget bound on `t_hat`) and width `outer_width / f`. Sized by `min_rank_for_secure_width(family, D, 2 · t_inf_bound, outer_width / f)`.

Under this reduction:

- `B'` is sized for its **own** input width `outer_width / f` and the bound on `Δt_i` — **not** for an augmented `[B' | −G]` width. The handoff explicitly rejects using `[B' | −G]` sizing unless a cryptographer subsequently decides the augmented relation is the correct SIS reduction, which is not the assumption of this spec.
- `F` is sized for its own input width and the bound on `Δuhat`.

**Sizing recipe (per-candidate, planner):**

```text
B':
  width  = outer_width / split_factor
  bound  = 2 · t_inf_bound        (≈ inner gadget bound for t_hat)
  n_b'   = min_rank_for_secure_width(family, D, bound, width)

F:
  width  = n_b' · split_factor · num_digits_outer
  bound  = 2 · floor(2^{outer_log_basis} / 2)  (balanced i8 max difference)
  n_F    = min_rank_for_secure_width(family, D, bound, width)
```

#### 7. ZK interaction

The existing ZK path adds blinding to `t_hat` entries before the B multiply (`b_blinding_digits` in `AkitaCommitmentHint`, `add_blinding_cyclic_rows` in [`crates/akita-prover/src/protocol/quadratic_equation.rs`](crates/akita-prover/src/protocol/quadratic_equation.rs)). For the tiered path:

1. The blinding still happens at the `t_hat` layer, so the `B' · t_i` multiply consumes blinded `t_hat` chunks. `u_i` and therefore `uhat_i` inherit the blinding mass.
2. **Decision point this spec leaves open:** does `uhat` itself need its own blinding tier (analogous to the existing `b_blinding`)? Since `uhat` becomes part of the folded witness, its privacy status must be argued by the implementer before enabling tiering under `--features zk`. The default in this spec is "no new blinding tier on `uhat`, but the entire tiered path is feature-gated off when `zk` is enabled until a follow-up resolves this." The implementing agent must either (a) keep tiering disabled with `zk`, or (b) add the blinding tier and prove the resulting privacy claim.
3. The existing ZK blinding segment offsets (§9 of [`specs/optimized_verifier.md`](specs/optimized_verifier.md)) shift to account for the new `uhat` segment between `t_hat` and `b_blind`. Existing tests under `--features zk` will catch off-by-one offset errors.
4. Bound on `Δuhat` in [Design §6](#6-soundness-two-case-two-tier-binding-argument) must include any blinding contribution.

This is explicitly *not* a "ZK works unchanged" claim. ZK + tiering needs a dedicated review pass; this spec does not promise byte-stability of the ZK blinding evaluators under `f > 1`.

#### 8. Root-direct path

The current root-direct verification path recomputes `u = B · t_hat` from the direct witnesses. For tiering, choose at implementation time:

1. **Disable tiering for root-direct.** Simplest: planner forbids `split_factor > 1` when the schedule picks root-direct. This caps the tiering surface to folded roots, which is where the dominant verifier cost lives anyway.
2. **Implement tiered root-direct recompute.** The verifier runs `t_hat → B' chunks → balanced_decompose_i8 → F → u_final` and checks equality against the proof's `u_final`. More code; more uniform semantics.

This spec recommends option (1) for the first landing and option (2) as a follow-up.

#### 9. Witness column layout in M

§3 of [`specs/optimized_verifier.md`](specs/optimized_verifier.md) defines the witness segment layout. The tiered path adds one new segment, `uhat`:

| segment | length (ring elements) | innermost → outermost axis order | introduced by |
|---|---|---|---|
| `uhat` | `num_points · n_b' · split_factor · num_digits_outer` | `dig → row → chunk → point` | this spec (tiered path only) |

Full root M-column layout (tiered):

```text
z_first = true :   M = [ z_hat ‖ w_hat ‖ t_hat ‖ uhat ‖ b_blind ‖ d_blind ‖ r_tail ]
z_first = false:   M = [ w_hat ‖ t_hat ‖ uhat ‖ b_blind ‖ d_blind ‖ z_hat ‖ r_tail ]
```

`uhat` is placed immediately after `t_hat` so the block-axis peel-eq trick for `t_hat` is unaffected. `uhat` has no block axis (chunks are not blocks); it is laid out densely along `(dig, row, chunk, point)`.

For `split_factor == 1`, no `uhat` segment appears (legacy layout).

The implementer must update every offset and length computation that touches the post-`t_hat` region:

- `ring_switch_build_w`, `build_w_coeffs` (prover M-witness assembly).
- Verifier offset arithmetic in `prepare_ring_switch_row_eval` and `compute_setup_contribution`.
- Prover debug `compute_m_evals_x` materialised path (if present).
- `planned_w_ring_element_count` (proof-size planner).
- Root expected next-w length.
- Proof-size estimates ([`crates/akita-types/src/layout/proof_size.rs`](crates/akita-types/src/layout/proof_size.rs)).
- `r_tail` offset and length.
- ZK blinding offsets (§9 of `specs/optimized_verifier.md`).

#### 10. Verifier evaluation model (and a failure mode to avoid)

The verifier savings depend on **reusing the physical `B'` α-eval rectangle across all `f` chunks.** The intended loop shape inside `compute_setup_contribution` (or its tiered equivalent) is:

```text
for each physical B' row r ∈ [0, n_b'):
    α-evaluate B'[r, 0..W'] once     // r_eval[c] for c ∈ [0, W')
    for each chunk i ∈ [1, f]:
        combine r_eval with chunk-i column pattern T'_col^{(i)}[c]
        add to row weight eq_tau1[tier1_start + g·f·n_b' + (i-1)·n_b' + r]
```

If the implementation instead performs `f · n_b' · num_points` independent physical SIS scans (one full α-eval per chunk per row), the savings disappear. The dispatch must share `r_eval[c]` across chunk patterns.

Required new column patterns for `compute_setup_contribution` (or a sibling tiered evaluator):

- `T'_col^{(i)}[c]` over the `i`-th chunk window of `t_hat`'s M-layout, for `i ∈ {1, …, f}`. All `f` patterns share the same `B'` α-eval but pair with different `t_hat` column windows.
- `U_col[c]` over the `uhat` segment for the `−G · uhat_i` half of tier-1 rows (structured, no SIS scan).
- `F_col[c]` over `uhat_concat` for the F rows.
- `eq_hi_uhat` (or equivalent) high-index table for the `uhat` segment.

The `−G · uhat_i` structured contribution may live in a separate evaluator (parallel to `WStructuredSlicesEvaluator` / `TStructuredSlicesEvaluator` in [`crates/akita-verifier/src/protocol/slice_mle/structured_slice.rs`](crates/akita-verifier/src/protocol/slice_mle/structured_slice.rs)), but it **must use the tier1 row weights**, not separate weights, so the two halves of each tier-1 row sum into a single row in M.

#### 11. Setup matrix and F key

The existing setup uses one shared flat matrix backing A, B, and D prefixes. Tiering adds an F role:

- `B'` is a deterministic shared key with `n_b'` rows and `outer_width / f` columns. Today's B-row block of the shared matrix is reused; the only change is that the **column window** `B[:, 0 : outer_width/f]` is the active part.
- `F` is a deterministic shared key with `n_F` rows and `n_b' · f · num_digits_outer` columns. New row block appended to the shared matrix. **Domain-separated** from A, B, D — do not accidentally alias rows or columns with B' unless this is explicitly part of the design.
- Setup envelope updates:

  ```text
  max_stride >= max(A_width, B_prime_width, D_width, F_width)
  ```

  Allocation row capacity is computed during setup generation (e.g. in `crates/akita-setup/`); the existing flow already passes a computed row capacity to allocation, so the only change is to include `n_F` in that computation. **`AkitaSetupSeed` has no `max_rows` field** ([`crates/akita-types/src/proof/setup.rs`](crates/akita-types/src/proof/setup.rs) confirms only `max_num_vars`, `max_num_batched_polys`, `max_num_points`, `max_stride`, `public_matrix_seed`). This spec does not add one; the implementing agent must either keep using the existing row-capacity flow or add a new dedicated field with its own justification.

#### 12. Planner

For root only, planner candidate space:

```text
split_factor    ∈ {1, 2, 4, 8, 16, ...}     (must divide full_outer_width)
outer_log_basis ∈ {2, 3, 4, 5, 6}
num_digits_outer = compute_num_digits_full_field(field_bits, outer_log_basis)
                   (or a separately-proven tighter bound)
```

Rules:

- `split_factor == 1` → use legacy path. Planner may emit this as a baseline candidate but it does *not* trigger any tiering code.
- `split_factor > 1` requires `split_factor | full_outer_width`.
- Reject any candidate where the SIS floor lookup fails or returns a rank that the rest of the config cannot support.
- Size `B'` and `F` from the recipe in [Design §6](#6-soundness-two-case-two-tier-binding-argument).
- Include `uhat` witness growth and tiered row-count growth in proof-size and witness-size planning ([`crates/akita-types/src/layout/proof_size.rs`](crates/akita-types/src/layout/proof_size.rs)).
- Score verifier cost honestly: it is not enough to count `compute_setup_contribution` α-evals. At minimum, include:

  ```text
  score = setup_contribution_alpha_evals(f, n_b', n_F, δ_outer)
        + tier1_gadget_structured_cost(f, n_b', δ_outer, num_points)
        + F_setup_alpha_evals(n_F, n_b' · f · δ_outer)
        + r_tail_growth(new_m_row_count)
        + sumcheck_domain_growth(new_witness_length)
  ```

- For **equal verifier scores, prefer smaller `f`**. Larger `f` increases prover work and witness/hint size; tie-break toward less prover impact unless profiling later proves a larger `f` is strictly better.
- Schedule cache key in [`crates/akita-planner/src/schedule_params.rs`](crates/akita-planner/src/schedule_params.rs) must encode `(split_factor, outer_log_basis, num_digits_outer, n_F, n_b')`. Otherwise distinct tierings collide on the same cache slot.

### Alternatives Considered

- **Split along the digit, `a_row`, `t_vector`, or `block` axis of `t_hat` instead of arbitrary contiguous chunks.** Splitting along the digit axis gives the largest theoretical savings but each chunk would have a different M-layout high index, complicating the column pattern in `compute_setup_contribution`. Splitting along `block` breaks the peeled-block fast path (§4–§5 of `optimized_verifier.md`). **Arbitrary contiguous chunks** keeps `t_hat` storage and M-axis order identical, with the chunk index becoming a small extra outer axis on the tier-1 column patterns. User-selected.
- **Raw `u_i` (no gadget decomposition) before F.** Forces `F`'s SIS rank or collision bound to absorb the full `u_i` norm, wiping out the savings. Gadget-decomposing `u_i → uhat_i` lets `F` operate on small balanced i8 inputs, matching how A/B/D are sized today. User-selected.
- **`f = 1` via the generic tiered path.** Rejected by the handoff. Even with `n_F = n_b` and `num_digits_outer = 1`, the generic path adds at least the F relation and the `uhat` witness segment, changing M rows, tau challenges, and transcript bytes. The legacy path must remain the only code path for `split_factor == 1`.
- **Augmented `[B' | −G]` SIS sizing for tier-1.** Rejected by the handoff in favour of the two-case binding proof in [Design §6](#6-soundness-two-case-two-tier-binding-argument). A future cryptographer review may reinstate it, but the default in this spec is that `B'` is sized for its own input width only.
- **Apply tiering at every level.** Recursive levels (~10% of total verifier α-eval work per [`docs/onehot-d32-nv32-matrix-sizes.md`](docs/onehot-d32-nv32-matrix-sizes.md)), have non-pow2 `block_len`, and interact with recursive-witness machinery. Out of scope.
- **More than two tiers.** Implementation cost grows non-linearly per added tier. Two tiers cover the gap up to the W-side bottleneck identified in [Performance](#performance); deeper savings need a W-side change.
- **Tier D instead of B.** D is the second-largest matrix at level 0. User explicitly deferred D; same tiering construction will apply.
- **Implicit gadget consistency rows as a separate M block.** Rejected: the handoff calls out that the original spec double-counted `f · n_b' · num_points` rows by treating the gadget half as a distinct row block. The corrected layout (see [Design §3](#3-corrected-m-row-and-y-layout-for-tiered-path)) folds the `−G · uhat_i` contribution onto the same tier-1 row as `B' · t_i`.

## Documentation

To update when implementation lands (this spec by itself does not write to these):

- [`specs/optimized_verifier.md`](specs/optimized_verifier.md):
  - Extend §2 row-block table with one new entry — "tier-1 (`B' · t_i − G · uhat_i = 0`)" — explicitly stating the row is single-counted across the setup and structured halves.
  - Add a second entry for the F rows.
  - Add a §7.X subsection on `compute_setup_contribution`'s tiered dispatch, the shared-`r_eval` requirement, and the chunked column patterns `T'_col^{(i)}`.
  - Extend §3 witness-layout table with the `uhat` segment and update the M-column layout diagram.
- [`docs/onehot-d32-nv32-matrix-sizes.md`](docs/onehot-d32-nv32-matrix-sizes.md) and [`docs/onehot-d32-nv32-g2-matrix-sizes.md`](docs/onehot-d32-nv32-g2-matrix-sizes.md): add per-level columns for `split_factor`, `n_b'`, `n_F`, `outer_log_basis`, `num_digits_outer`, plus an "optimized B'" rectangle column.
- [`/.cursor/skills/hachi-protocol/SKILL.md`](.cursor/skills/hachi-protocol/SKILL.md) and [`/.cursor/skills/hachi-batching/SKILL.md`](.cursor/skills/hachi-batching/SKILL.md): one paragraph each describing the tiered root commitment and the explicit `split_factor` dispatch, so future agents do not assume `u = B · t_hat` universally.

## Execution

Recommended order. Earlier steps unblock later ones.

1. **Rewrite this spec** (done; this document).
2. **`LevelParams` types and helpers.** [`crates/akita-types/src/layout/params.rs`](crates/akita-types/src/layout/params.rs):
   - Add `split_factor`, `outer_log_basis`, `num_digits_outer`, `f_key`.
   - Add `is_tiered_root`, `outer_commitment_rows`, `b_prime_rows`, `b_prime_width`, `full_outer_width`.
   - Update `with_decomp`, `with_layout`, `m_row_count`, `params_only` constructors.
   - **Audit every existing call to `lp.outer_width()` / `lp.b_key.col_len()`** and migrate to `full_outer_width()` where the full `t_hat` is meant. This is the highest-risk migration in the spec.
3. **Digit math and SIS sizing wrappers.** [`crates/akita-types/src/layout/digit_math.rs`](crates/akita-types/src/layout/digit_math.rs), [`crates/akita-types/src/layout/sis_derivation.rs`](crates/akita-types/src/layout/sis_derivation.rs), [`crates/akita-planner/src/sis_security.rs`](crates/akita-planner/src/sis_security.rs):
   - Helper that returns `num_digits_outer` from `(field_bits, outer_log_basis)`.
   - SIS sizing wrappers for `B'` and `F` using the two-case proof bounds.
4. **`AkitaCommitmentHint.outer_digits`.** [`crates/akita-types/src/proof/mod.rs`](crates/akita-types/src/proof/mod.rs): add the `Vec<FlatDigitBlocks<D>>` field (empty for legacy). Update `into_flat_parts` and `with_recomposed_inner_rows`.
5. **Setup envelope + F-row block.** [`crates/akita-types/src/proof/setup.rs`](crates/akita-types/src/proof/setup.rs) (do **not** add `max_rows` to `AkitaSetupSeed`), `crates/akita-setup/`:
   - Compute envelope including `F_width`.
   - Derive deterministic `F` rows from the existing `public_matrix_seed`, domain-separated from A/B/D row blocks.
6. **Tiered `commit_with_params`.** [`crates/akita-prover/src/api/commitment.rs`](crates/akita-prover/src/api/commitment.rs):
   - Branch on `lp.is_tiered_root()`. Legacy branch unchanged.
   - Tiered branch: per-chunk `B'` multiply, balanced decompose, concat, F multiply.
   - **Verify the kernel surface for restricted column window** in [`crates/akita-prover/src/kernels/linear.rs`](crates/akita-prover/src/kernels/linear.rs) and add a wrapper or a new entry point as needed.
7. **`generate_y` and row construction.** [`crates/akita-prover/src/protocol/quadratic_equation.rs`](crates/akita-prover/src/protocol/quadratic_equation.rs):
   - Branch on `lp.is_tiered_root()`. Legacy branch unchanged.
   - Tiered branch: produce the corrected `y` layout from [Design §3](#3-corrected-m-row-and-y-layout-for-tiered-path).
   - Replace the legacy `B · t_hat − u` block with the tier-1 + F row blocks; emit each tier-1 row with both setup and structured contributions sharing the row weight.
   - Update `m_row_count` consumers downstream.
8. **`uhat` in the root witness.** [`crates/akita-prover/src/protocol/ring_switch.rs`](crates/akita-prover/src/protocol/ring_switch.rs):
   - Thread `outer_digits` through `ring_switch_build_w` / `build_w_coeffs` so the M-column witness for the tiered path includes `uhat` between `t_hat` and the existing blinding segments.
   - Update every offset and length computation downstream (per [Design §9](#9-witness-column-layout-in-m)).
9. **Verifier evaluation.** [`crates/akita-verifier/src/protocol/ring_switch.rs`](crates/akita-verifier/src/protocol/ring_switch.rs), [`crates/akita-verifier/src/protocol/slice_mle/setup_contribution.rs`](crates/akita-verifier/src/protocol/slice_mle/setup_contribution.rs), [`crates/akita-verifier/src/protocol/slice_mle/structured_slice.rs`](crates/akita-verifier/src/protocol/slice_mle/structured_slice.rs):
   - Extend `RingSwitchDeferredRowEval` with `split_factor`, `outer_log_basis`, `num_digits_outer`, `n_F`, `n_b_prime`.
   - Compute `uhat` segment offset.
   - Add tier-1 dispatch in `compute_setup_contribution` per [Design §10](#10-verifier-evaluation-model-and-a-failure-mode-to-avoid). Share `r_eval[c]` across the `f` chunk patterns.
   - Add a structured evaluator for the `−G · uhat_i` half (or fold inline) using the same tier-1 row weights.
   - Add the F setup contribution using a new column pattern over `uhat_concat`.
   - Validate `commitment.u.len() == lp.outer_commitment_rows()` in [`crates/akita-verifier/src/protocol/batched.rs`](crates/akita-verifier/src/protocol/batched.rs) before any further work.
10. **Root-direct decision.** [`crates/akita-verifier/src/protocol/levels.rs`](crates/akita-verifier/src/protocol/levels.rs): either reject `split_factor > 1` on root-direct schedules, or implement the tiered recompute. Spec recommends the former for first landing.
11. **Planner integration.** [`crates/akita-planner/src/search.rs`](crates/akita-planner/src/search.rs), [`crates/akita-planner/src/schedule_params.rs`](crates/akita-planner/src/schedule_params.rs), [`crates/akita-config/src/proof_optimized.rs`](crates/akita-config/src/proof_optimized.rs):
   - Enumerate `(split_factor, outer_log_basis, num_digits_outer)` per [Design §12](#12-planner).
   - Score with the verifier cost model including all components called out in §12.
   - Tie-break toward smaller `f`.
   - Encode the tiering parameters in the schedule cache key.
12. **Tests.**
    - `f = 1` byte-identical legacy snapshot test.
    - Tiered `commit` unit test.
    - Materialised-M tiered fixture in [`crates/akita-verifier/src/protocol/slice_mle/setup_contribution.rs`](crates/akita-verifier/src/protocol/slice_mle/setup_contribution.rs).
    - Wrong `u.len()` verifier rejection.
    - Multipoint and grouped-root E2E at `f > 1` in [`crates/akita-pcs/tests/multipoint_batched_e2e.rs`](crates/akita-pcs/tests/multipoint_batched_e2e.rs).
    - ZK E2E if the ZK decision from [Design §7](#7-zk-interaction) enables tiering under `--features zk`.
13. **Profile and microbench.**
    - `AKITA_MODE=onehot_d32 AKITA_NUM_VARS=32 cargo run --release --example profile` at `split_factor ∈ {1, 2, 4, 8}`. Verify the `setup_contribution` span shrinks roughly per the table in [Performance](#performance).
    - Repeat with `AKITA_NUM_POLYS=4 AKITA_GROUP_SIZE=2`.
    - New microbench in `crates/akita-pcs/benches/` for `compute_setup_contribution` across `f`.
14. **Workspace gates.**

    ```bash
    cargo fmt -q
    cargo clippy --all --message-format=short -q -- -D warnings
    cargo test
    ```

Risks to resolve before / during implementation:

- **Kernel surface for restricted column window** (Step 6). Verify whether `mat_vec_mul_ntt_single_i8` accepts an offset or whether a new wrapper is needed.
- **SIS floor at small widths** (Steps 3, 11). Confirm that `F`'s width `n_b' · f · δ_outer` and `B'`'s width `outer_width / f` produce small `n_F` / `n_b'`. If they force large ranks, the savings shrink.
- **Bound convention for `2^{outer_log_basis}` vs `2^{outer_log_basis} − 1` vs `2^{outer_log_basis − 1}`** ([Design §5](#5-outer-digit-decomposition)). Pick one and apply consistently.
- **Audit of `lp.outer_width()` / `lp.b_key.col_len()` call sites** (Step 2). Highest silent-regression risk.
- **ZK status** ([Design §7](#7-zk-interaction)). Resolve before flipping tiering on under `--features zk`.
- **Root-direct status** ([Design §8](#8-root-direct-path)). Resolve before tiering interacts with any root-direct schedule.

## References

### Internal

- [`specs/tiered_commit_agent_handoff.md`](tiered_commit_agent_handoff.md) — review conclusions and corrections that produced this revision.
- [`specs/optimized_verifier.md`](optimized_verifier.md) — canonical verifier-cost model that this spec extends.
- [`docs/onehot-d32-nv32-matrix-sizes.md`](../docs/onehot-d32-nv32-matrix-sizes.md), [`docs/onehot-d32-nv32-g2-matrix-sizes.md`](../docs/onehot-d32-nv32-g2-matrix-sizes.md) — baseline matrix sizes the cost table compares against.
- [`/.cursor/skills/hachi-protocol/SKILL.md`](../.cursor/skills/hachi-protocol/SKILL.md), [`/.cursor/skills/hachi-batching/SKILL.md`](../.cursor/skills/hachi-batching/SKILL.md) — protocol context.
- Implementation surfaces:
  - [`crates/akita-types/src/layout/params.rs`](../crates/akita-types/src/layout/params.rs), [`crates/akita-types/src/layout/digit_math.rs`](../crates/akita-types/src/layout/digit_math.rs), [`crates/akita-types/src/layout/sis_derivation.rs`](../crates/akita-types/src/layout/sis_derivation.rs), [`crates/akita-types/src/layout/proof_size.rs`](../crates/akita-types/src/layout/proof_size.rs), [`crates/akita-types/src/proof/mod.rs`](../crates/akita-types/src/proof/mod.rs), [`crates/akita-types/src/proof/commitment.rs`](../crates/akita-types/src/proof/commitment.rs), [`crates/akita-types/src/proof/setup.rs`](../crates/akita-types/src/proof/setup.rs).
  - [`crates/akita-prover/src/api/commitment.rs`](../crates/akita-prover/src/api/commitment.rs), [`crates/akita-prover/src/protocol/quadratic_equation.rs`](../crates/akita-prover/src/protocol/quadratic_equation.rs), [`crates/akita-prover/src/protocol/ring_switch.rs`](../crates/akita-prover/src/protocol/ring_switch.rs), [`crates/akita-prover/src/protocol/flow.rs`](../crates/akita-prover/src/protocol/flow.rs), [`crates/akita-prover/src/kernels/linear.rs`](../crates/akita-prover/src/kernels/linear.rs).
  - [`crates/akita-verifier/src/protocol/ring_switch.rs`](../crates/akita-verifier/src/protocol/ring_switch.rs), [`crates/akita-verifier/src/protocol/slice_mle/setup_contribution.rs`](../crates/akita-verifier/src/protocol/slice_mle/setup_contribution.rs), [`crates/akita-verifier/src/protocol/slice_mle/structured_slice.rs`](../crates/akita-verifier/src/protocol/slice_mle/structured_slice.rs), [`crates/akita-verifier/src/protocol/slice_mle/zk_blinding.rs`](../crates/akita-verifier/src/protocol/slice_mle/zk_blinding.rs), [`crates/akita-verifier/src/protocol/levels.rs`](../crates/akita-verifier/src/protocol/levels.rs), [`crates/akita-verifier/src/protocol/batched.rs`](../crates/akita-verifier/src/protocol/batched.rs).
  - [`crates/akita-planner/src/search.rs`](../crates/akita-planner/src/search.rs), [`crates/akita-planner/src/schedule_params.rs`](../crates/akita-planner/src/schedule_params.rs), [`crates/akita-planner/src/sis_security.rs`](../crates/akita-planner/src/sis_security.rs), [`crates/akita-config/src/proof_optimized.rs`](../crates/akita-config/src/proof_optimized.rs).

### External

- Greyhound (BCS24), §3 on tree-of-Ajtai-commitments: the standard two-tier construction this spec instantiates.
- LaBRADOR (BS24): the same tiering pattern in a non-cyclotomic setting; same security argument.
- Ajtai's SIS construction: underpins the gadget identity `u_i = G · uhat_i`.

### Profiling commands

```bash
AKITA_MODE=onehot_d32 AKITA_NUM_VARS=32 cargo run --release --example profile
AKITA_MODE=onehot_d32 AKITA_NUM_VARS=32 AKITA_NUM_POLYS=4 AKITA_GROUP_SIZE=2 cargo run --release --example profile
cargo bench -p akita-pcs --bench setup_contribution -- f_sweep
```
