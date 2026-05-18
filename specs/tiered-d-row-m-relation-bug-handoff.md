# Tiered Chunks A-Row Bug — RESOLVED (2026-05-18)

**Status**: `tiered_onehot_prove_verify_small` and
`tiered_dense_prove_verify_small` PASS under the explicit 10-group
row layout. The bug was **CRT capacity overflow** in
`fused_split_eq_quotients` for the tier-marked chunks A row, not a
shape mismatch in `MRowLayout`. Fix shipped in
`crates/akita-prover/src/protocol/quadratic_equation.rs`'s
`compute_r_split_eq` heterogeneous A-row Z-quotient path.

## 1 — Root cause

Per the book's protocol decision (combined Ajtai binding, `n_A` rows
per tier — see §3 below), the prover computes
`r[A_row] = high_half(Σ c · t_rec) − high_half(A · z_pre)` per A row,
using the NTT-cached `fused_split_eq_quotients` kernel for the
`high_half(A · z_pre)` half. The kernel internally:

1. Converts each `z_pre[j]` cell (centered `i64` in
   `[−z_pre_max_abs, z_pre_max_abs]`) into negacyclic + cyclic NTT
   representations.
2. Accumulates pointwise products with the pre-converted A row NTTs
   over all `inner_width_g` columns.
3. Inverse NTTs and CRT-reconstructs each result row, then computes
   `(cyclic − negacyclic) / 2 = high_half(A·z_pre)`.

The CRT product `P` for the Q128 dispatch (5 × 30-bit primes) is
≈ 2^150. The kernel's reconstruction returns the integer polynomial
coefficient mod `P`. When the **integer** polynomial-product
coefficient `inner_width_g · D · |A| · |z_pre|` exceeds `P`, the
mod-`P` reduction silently wraps and the result (after the final mod
`p_F` conversion) differs from the true field-arithmetic value.

For the chunks group in `tiered_onehot_prove_verify_small` at NV=28
recursive level 1 (the production diagnostic captured this exactly):

```
chunks_inner_width = 13312       (spec.block_len=512 × num_digits_commit=26)
outer_inner_width  = 1539        (lp.inner_width — what A is officially sized for)
a_col_len          = 1539
n_a                = 3
z_pre_centered_inf_norm = 57393  (≈ 2^16)
```

Per-coefficient bound: `13312 × 64 × 2^127 × 2^16 ≈ 2^163 > P ≈ 2^150`.

The verifier evaluates `−chunks_z = LIFTED(A·z_pre)(α)` via scalar
field arithmetic at `α` (no CRT), so it gets the correct value.
Prover-verifier therefore disagree at the chunks A row only, exactly
the symptom the prior handoff captured. The disagreement starts at
chunks A row 0 (row 22 at production parameters) because that is the
first row whose `M·W` identity depends on the wraparound kernel
output.

The same kernel is correct for:

- D-rows (`mat_vec_mul_ntt_single_i8_cyclic` with `i8` digit input,
  per-coefficient bound `inner_width · D · 2^127 · 8` stays under
  `P`);
- W-group and meta-group A rows (small `inner_width`, small `|z_pre|`,
  bound stays under `P` in practice for the production schedules
  shipped today).

## 2 — Fix

Inside `compute_r_split_eq`'s heterogeneous A-row Z-quotient setup
(file `crates/akita-prover/src/protocol/quadratic_equation.rs`),
dispatch on `has_tiered_group && is_tiered`:

- **Untiered groups (W, meta)**: keep the existing NTT-based
  `fused_split_eq_quotients(...z_slice...)` path. CRT capacity is
  fine for these per-group `inner_width_g` values.
- **Tier-marked groups (chunks)**: compute `high_half(A · z_pre)`
  row-by-row via a direct **field-domain** polynomial-product
  reduction:

  ```text
  for each col in 0..inner_width_g:
      a = A[row][col]   (CyclotomicRing<F, D>)
      z = z_pre[slot + col]    (CenteredCoeff = i64 vec of D)
      for (i, j) with i + j >= D:
          high_half[i + j − D] += a.coefficients()[i] · F::from_i64(z[j])
  ```

  All arithmetic is mod `p_F` from the start, so no CRT capacity is
  consumed. The same formula is what
  `direct_high_half(A·z)` produced under the diagnostic in §3 and is
  the (algebraically obvious) identity
  `high_half(p) = Σ_{(i,j) | i + j ≥ D} a[i] z[j] · X^{i + j − D}`.

The performance cost is `n_A · inner_width_g · D²` field multiplies
per A row family per recursive level for tier-marked groups; for
production NV=28 onehot this is ≈ 3 × 13312 × 64² ≈ 1.6 × 10⁸ mults
per level, exercised on the rare tiered code path.

The W and meta paths retain the fast NTT kernel; only the chunks
path takes the field-domain quotient.

## 3 — Book reading (closed)

Book §5.4 line 728–729 specifies **one** combined Ajtai binding
`A · z_pre = c` of `n_A` rows per tier, not `k · n_A`:

- Items 1–2 explicitly say "Per-chunk D-checks (`k × n_{D,chunk}`
  rows)" / "Per-chunk B-checks (`k × n_{B,chunk}` rows)" with
  per-chunk `ê_j` / `t̂_j` subscripts; items 3–5 drop both the
  `k ×` multiplier and the `_j` subscripts.
- Line 727 defines `z_pre = Σ_j c_j · block_j` — the binding's
  `z_pre` is the chunk-summed fold by definition.
- §5.3 line 654 already states "Ajtai binding: covers the combined
  `z_pre`"; the tiered design inherits this.
- §5.4 line 751–754 calls out only rows 1–2 as the per-chunk
  block-diagonal family.
- `fig:fourthroot-protocol` Round 6 lifts a single `M w = h +
  (X^d + 1) r` — singular relation.
- Line 698: "The proof contains `(c, c_meta, v_meta, u_meta)` —
  independent of `k`" — would be violated by `k · n_A` per-chunk A
  rows.

Verifier cost also favours combined: per-chunk would `k×` blow up the
A-side z-segment MLE work, grow `m_row` by `(k − 1) · n_A`, and add
sumcheck rounds proportional to `log_2 (1 + (k − 1) n_A / m_row)`.

The current `MRowLayout.original_a = cursor..(cursor + n_a)` shape
and `compute_r_split_eq`'s "one `a_quotients` slot per group" wiring
match this — no row-layout change was needed.

## 4 — Acceptance criteria

| Test                                                                                                                          | Status   |
|-------------------------------------------------------------------------------------------------------------------------------|----------|
| `cargo test -p akita-types layout::params::tests::m_row_layout_round_trip_tiered --lib`                                       | passes   |
| `cargo test -p akita-prover protocol::quadratic_equation::tests::tiered_grouped_m_rows_match_committed_witness_locally --lib` | passes   |
| `cargo test -p akita-prover protocol::quadratic_equation::tests::tiered_grouped_m_rows_match_committed_witness_multi_a --lib` | passes   |
| `cargo test -p akita-prover protocol::flow::tests::tiered_handle_material_matches_verifier_derivation --lib`                  | passes   |
| `cargo test -p akita-pcs --test multi_group_commit tiered_prepare_m_eval_setup_weight_matches_eval_split`                     | passes   |
| `cargo test --release -p akita-pcs --test tiered_setup_e2e tiered_onehot_prove_verify_small`                                  | passes   |
| `cargo test --release -p akita-pcs --test tiered_setup_e2e tiered_dense_prove_verify_small` (un-ignored this session)         | passes   |
| `cargo fmt -q`                                                                                                                | clean    |
| `cargo clippy --all --message-format=short -q -- -D warnings`                                                                 | clean    |

Bisect playbook from earlier handoff revisions executed and consumed:
§4.1 (chunks_inner_width vs A column budget) was the right suspect;
§4.2 (tensor-stage1 challenge indexing) and §4.3 (r-tail row order)
were ruled out by the in-place diagnostic before the fix.

## 5 — Files changed this session

- `crates/akita-prover/src/protocol/quadratic_equation.rs`: dispatch
  on `is_tiered` in `compute_r_split_eq`'s heterogeneous A-row
  Z-quotient setup; field-domain high-half computation for chunks.
  Reverts of speculative changes from prior session not preserved (the
  raw-vs-decomposed-z path remained for un-tiered heterogeneous
  groups). Added `tiered_grouped_m_rows_match_committed_witness_multi_a`
  unit test covering every D/B row of the explicit 10-group layout
  under `n_a = n_b = n_d = 2`.
- `crates/akita-pcs/tests/tiered_setup_e2e.rs`: un-ignored
  `tiered_dense_prove_verify_small` now that the relation closes.
- `crates/akita-prover/src/protocol/flow.rs`,
  `crates/akita-verifier/src/protocol/setup_claim_reduction.rs`,
  `crates/akita-verifier/src/protocol/levels.rs`: converted noisy
  `eprintln!` diagnostics from prior debugging sessions to
  `tracing::debug!` so they no longer spam test output but stay
  available under `RUST_LOG=debug`.
- `crates/akita-types/src/layout/params.rs`,
  `crates/akita-types/src/layout/proof_size.rs`: added missing
  `# Panics` / `# Errors` doc sections to satisfy clippy.

## 6 — Out-of-scope follow-ups

These tests are NOT part of this session's acceptance criteria but
were observed to fail and are tracked for separate work:

- **`tiered_production_prove_verify`** (NV=32 dense, f=8): gets
  SIGKILL (likely OOM at production scale). Pre-existing behaviour is
  unclear since the test was failing pre-fix at the prover stage too;
  post-fix the prover gets further into the pipeline. Performance
  characteristics of the field-domain chunks A quotient at f=8 (k=64
  chunks, n_A=3) need profiling.
- **`tiered_rejects_tampered_meta_material`** and
  **`tiered_rejects_tampered_s_opening_value`**: pre-fix these failed
  at the prover stage (line 261 = `batched_prove`) — the bug masked
  any verifier-side cache-tamper check. Post-fix the prover succeeds
  for `tiered_rejects_tampered_s_opening_value` (verifier correctly
  rejects the tampered `s_opening_value`). The
  `tiered_rejects_tampered_meta_material` test now reaches the verify
  step but the verifier does NOT reject a tampered
  `tiered_s_cache.chunk_b_commitments` — `expand_tiered_setup_claims`
  reads the cache without cross-checking against the deterministic
  derivation from the public matrix. This is a separate verifier
  hardening task (cache-validation), not part of the row-relation
  bug.

## 7 — Reproducibility notes

The fix is deterministic and platform-independent. The CRT capacity
boundary depends only on:

- Field width (`Fp128`),
- Q-variant dispatch (`Q128`, K=5 30-bit primes),
- Per-group `inner_width_g · D · |A| · |z_pre|` integer bound.

For any future tiered group whose
`spec.num_digits_commit = full_digits` makes `inner_width_g` exceed
the W/meta scale, the same field-domain quotient must be used to
avoid silent wraparound. The dispatch is already keyed on
`spec.tier.is_some_and(|t| t.is_tiered())`, so adding new tier-marked
group shapes (e.g. higher `f`) inherits the fix.

If a future schedule pushes the W or meta A rows over the CRT
capacity, the same field-domain dispatch will need to be widened —
in that case generalize the predicate from `is_tiered` to a per-group
overflow check.
