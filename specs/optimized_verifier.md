# Spec: Optimized Verifier (Matrix-`M`-at-a-Point)

| Field      | Value                                    |
|------------|------------------------------------------|
| Author(s)  |                                          |
| Created    | 2026-05-13                               |
| Status     | proposed                                 |
| PR         |                                          |

## Summary

The Akita verifier's per-level cost is dominated by one job: evaluating
the multilinear extension `M̃(r_row, r_col)` of the *virtual* row-combined
M-table at a random point. The M-table is never materialized; instead the
verifier walks each row block of `M` independently and adds the
contributions. This spec is the canonical reference for **what each
contribution is, where its rows live in `M`, and how it is evaluated
without materialising the table**.

## Intent

### Goal

Decrease verifier time (primary) and verifier memory (secondary) with no
protocol-visible change — the transcript layout, the M-table semantics,
and the scalar the verifier replays at the end of each level all stay
exactly as today. The optimization is a verifier-internal arithmetic
reorganization confined to `crates/akita-verifier`.

### Non-Goals

- No change to the prover, the M-table layout in the prover's
  ring-switch path, or the witness format.
- No change to the SIS commitment, the gadget basis, digit depths, or
  any soundness-relevant parameter.
- No change to the schedule, planner, or recursion structure.

---

## 1. The total formula

The verifier computes

$$
\widetilde{M}(r_\text{row}, r_\text{col})
\;=\; \sum_{i, j} \mathrm{eq}(r_\text{row}, i) \cdot \mathrm{eq}(r_\text{col}, j) \cdot M[i, j].
$$

`M` decomposes into disjoint row blocks; the verifier therefore writes
`M̃` as a sum of one scalar per row block:

$$
\widetilde{M}(r_\text{row}, r_\text{col})
\;=\; w_\text{structured} \;+\; t_\text{structured} \;+\; z_\text{structured}
\;+\; \text{setup\_contribution}
\;+\; r_\text{contribution}
\;+\; \underbrace{b_\text{blinding} + d_\text{blinding}}_{\text{only under }\texttt{feature = "zk"}}.
$$

Each contribution comes from a distinct set of `M`-rows (§4–§9). Every
contribution shares the same `r_col` randomness, but each chooses its own
*evaluation technique* depending on the row block's algebraic structure.

This is exactly the body of `RingSwitchDeferredRowEval::eval_at_point`
in `crates/akita-verifier/src/protocol/ring_switch.rs`.

### Notation Glossary

This spec uses short dimension names for fields on
`RingSwitchDeferredRowEval`:

| notation | `RingSwitchDeferredRowEval` field | meaning |
|---|---|---|
| `B` | `num_blocks` | witness-side block count |
| `C` | `num_claims` | number of batched evaluation claims |
| `L` | `depth_open` | open-side digit depth |
| `P` | `num_points` | number of distinct opening points |
| `DC` | `depth_commit` | commit-side digit depth |
| `DF` | `depth_fold` | fold-side digit depth |
| `n_A` | `n_a` | number of `A` rows |
| `G` | `num_commitment_groups` | number of commitment groups |

## 2. The five row-block categories

| category | rows of `M` | what makes it fast | section |
|---|---|---|---|
| **structured (tensor) rows** over `ŵ`, `t̂`, `ẑ` | rows of the form `vᵀ ⊗ G` for known small public vectors `v` and gadget vectors `G` | every entry factors as a product of public scalars → no SIS-matrix scan | §4, §5, §6 |
| **setup-matrix rows** `D·ŵ + B·t̂ + A·ẑ` | rows of the shared SIS commitment matrix evaluated at `α` | one column-pattern build + a single SIS-row scan, with `r_eval` shared across all three halves | §7 |
| **r-tail** | the `rows × levels` `r`-tail planes | pow2: multi-factor `eval_offset_eq_tensor`; non-pow2: materialise + single-factor | §8 |
| **ZK B-blinding** *(feature-gated)* | per-group `B`-side blinding planes | dedicated single-factor `eval_offset_eq_tensor` on a materialised segment | §9.1 |
| **ZK D-blinding** *(feature-gated)* | global `D`-side blinding planes | dedicated single-factor `eval_offset_eq_tensor` on a materialised segment | §9.2 |

The first row of the table — structured rows — uses the same algorithmic
idea three times with different "shapes": one row block per witness
segment. The second row — setup-matrix rows — is where most of the
work goes (one ring α-evaluation per `(SIS row, column)` cell, fused
across W/T/Z), so it has the longest section.

## 3. Where each segment lives in `M`

The M-table's column axis is one flat coordinate in `[0, 2^N)` for `N =
witness-size bits`. The verifier writes the witness segments to disjoint
column ranges, in one of two orderings (`z_first := m_vars ≥ r_vars`):

```text
z_first = true :   M = [ ẑ ‖ ŵ ‖ t̂ ‖ b_blind ‖ d_blind ‖ r-tail ]
z_first = false:   M = [ ŵ ‖ t̂ ‖ b_blind ‖ d_blind ‖ ẑ ‖ r-tail ]
```

Segment lengths and axes:

| segment | length (ring elements) | nested axis order (innermost → outermost) | controls |
|---|---|---|---|
| `ŵ` | `L · C · B` | `block → claim → dig` | open-side digit depth `L`, claims `C`, blocks `B` |
| `t̂` | `L · n_A · C · B` | `block → claim → dig → a_row` | adds the `A`-row axis `n_A` |
| `ẑ` | `DF · DC · P · block_len` | `blk → pt → df → dc` | commit-side `DC`, fold-side `DF`, points `P`, in-block `block_len` |
| `b_blind` | `prepared.b_blinding_segment_len` (0 without zk) | per-group t̂-tail (§9.1) | per-group append after `t̂` |
| `d_blind` | `prepared.d_blinding_segment_len` (0 without zk) | global D-side tail (§9.2) | per-row append after `b_blind` |
| `r-tail` | `rows · levels` | `level → row` | rows-of-`M` × gadget levels |

**The block axis is always innermost.** This is by design: the verifier
factorises `eq(r_col, ·)` along the block-bit window, so the innermost
axis must align with a contiguous bit range. For `ŵ` and `t̂` the block
axis size is `B = num_blocks` (a power of two by construction); for `ẑ`
it is `block_len` (power of two at the root level, sometimes not at
recursive levels — see §6 + §7.4).

The witness-side coordinate of cell `c` is `offset_segment + idx_M(c)`,
where `idx_M` is the bijection given by the nested order above. Sections
§4–§9 give each segment's `idx_M` formula explicitly.

---

## 4. `w_structured_contribution` — structured rows over `ŵ`

### 4.1 What it covers

Two row blocks of `M` act on `ŵ` with **fully separable** (tensor-product)
structure — every entry is a product of small public scalars:

- **Row block A** (`bᵀ ⊗ G_{2^r}`): `P` rows, one per opening point. Reads
  off the public claim value `y_p` from the digit-decomposed `ŵ`.
- **Row block B** (`cᵀ ⊗ G_1`): one row. Ties `ŵ` to the stage-1
  sparse-challenge linear combination after ring-switch evaluation at `α`.

Both rows share `ŵ`'s axis order `[dig, claim, block]` with block
innermost, so the W-segment M-layout flat index is

$$
j_M^W(\text{claim}, \text{dig}, \text{block}) \;=\; \text{block} \;+\; B \cdot (\text{dig} \cdot C \;+\; \text{claim}).
$$

Row weights:

- Row block A row `pt` carries weight `y_w[pt] = eq_τ₁[1 + pt]`.
- Row block B carries weight `consistency_weight = eq_τ₁[0]`.

### 4.2 Entry formulas

For row block A, fixed point `pt`:

$$
A_{pt}[c] \;=\; \gamma_\text{claim} \cdot b_p[\text{block}] \cdot g_1[\text{dig}]
\cdot \mathbb{1}[\text{claim\_to\_point}(\text{claim}) = pt],
$$

with tensor view `A_{pt}[\cdot] = g_1 \otimes (\gamma \circ \mathbb{1}[\cdot \to pt]) \otimes b_p`.

For row block B:

$$
B[c] \;=\; c_\alpha[\text{claim}, \text{block}] \cdot g_1[\text{dig}],
\qquad B[\cdot] = g_1 \otimes c_\alpha.
$$

Notation:
- `g_1[dig] = b^{dig}` (gadget weight, basis `b = 2^r`)
- `b_p[block]` — outer-block weight of opening point `p`
- `c_α[claim, block]` — α-evaluation of the stage-1 sparse challenge for
  `(claim, block)`
- `γ_claim` — batching coefficient

Both entry formulas are products of one per-`dig`, one per-`claim`, and
one per-`block` factor.

### 4.3 Evaluation: peeled-block + claim summaries

Peel the low `log₂(B)` bits of `r_col` off into an `eq_low` table (size
`B`); the rest becomes `eq_high`. For any column `c` in `ŵ`'s range,

$$
\mathrm{eq}(r_\text{col}, \text{offset}_w + c)
\;=\; \text{eq\_low}[(\text{block\_offset\_low} + \text{block}) \bmod B] \cdot \text{eq\_hi\_w}[q + \text{carry}],
$$

where `q = dig · C + claim`, `block_offset_low = offset_w mod B`, and
`carry ∈ {0, 1}`.

Precompute, **once per verifier**:

- `eq_low` — size `B` (shared with `t_structured`, `setup_contribution`).
- `eq_hi_w[k]` for `k ∈ [0, C · L]`.
- Per-opening-point block summary
  $\text{BLOCK\_SUMMARY\_A}[pt] = \bigl[\sum_{b:\text{carry}=0} \text{eq\_low}[\dots] \cdot b_p[b], \;\sum_{b:\text{carry}=1} \dots\bigr]$.
- Per-claim block summary
  $\text{BLOCK\_SUMMARY\_B}[\text{claim}] = \bigl[\sum_{b:\text{carry}=0} \text{eq\_low}[\dots] \cdot c_\alpha[\text{claim}, b], \;\sum_{b:\text{carry}=1} \dots\bigr]$
  (reused by `t_structured` — see §5.3).

The per-row inner sum then collapses to a two-term lookup per
`(dig, claim)`:

```text
A_pt row contribution at (dig, claim, carry)
   = y_w[pt] · γ_claim · g1[dig] · BLOCK_SUMMARY_A[pt][carry]
                                  · 1[claim_to_point(claim) = pt]
B row contribution at (dig, claim, carry)
   = consistency_weight · g1[dig] · BLOCK_SUMMARY_B[claim][carry]
```

multiplied by `eq_hi_w[q + carry]` and summed over `(dig, claim, carry)`.

### 4.4 Cost

| step | cost | done once per |
|---|---|---|
| `eq_low` (size `B`) | `O(B)` | verifier |
| `eq_hi_w` (size `C · L + 1`) | `O((C · L) · log)` | verifier |
| `BLOCK_SUMMARY_A[pt]` for `pt ∈ [0, P)` | `O(P · B)` | verifier |
| `BLOCK_SUMMARY_B[claim]` for `claim ∈ [0, C)` | `O(C · B)` | verifier (shared with `t_structured`) |
| evaluate one row over its `(dig, claim, carry)` outer | `O(C · L)` | per row |
| sum over `P + 1` rows | `O((P + 1) · C · L)` | total |

**No SIS-matrix read.** Implementation: `WStructuredSlicesEvaluator` via
the `StructuredSliceMleEvaluator` trait;
`WStructuredSlicesEvaluator { ... }.evaluate()`.

---

## 5. `t_structured_contribution` — structured rows over `t̂`

### 5.1 What it covers

One row block of `M` acts on `t̂` with separable structure:

- **Row block C** (`cᵀ ⊗ G_{n_A}`): `n_A` rows, one per row of the
  consistency-check matrix `A`. Same role as row block B for `ŵ`, but
  per `a_row`.

### 5.2 Format

`t̂` has length `L · n_A · C · B` and the nested order
`[block, claim, dig, a_row]`, so

$$
j_M^T(\text{flat\_claim}, \text{dig}, \text{a\_row}, \text{block})
\;=\; \text{block} \;+\; B \cdot \bigl(\text{flat\_claim} + C \cdot \text{dig} + C \cdot L \cdot \text{a\_row}\bigr).
$$

`flat_claim` is the **global** flat-claim index `∈ [0, C)`. Multi-group:
the witness layout doesn't split per group — see §5.5.

Row weights: row `a_row` has weight `eq_τ₁[a_start + a_row]`, where
`a_start = 1 + num_public_eval_rows + n_d + n_b · G`. The evaluator
receives exactly the `eq_τ₁[a_start..rows]` slice, whose length is `n_A`.

### 5.3 Entry formula

$$
C[r=(a_\text{row}), c=(\text{claim}, \text{dig}, \text{block})]
\;=\; a_w[a_\text{row}] \cdot c_\alpha[\text{claim}, \text{block}] \cdot g_1[\text{dig}],
$$

with tensor view `C = a_w \otimes g_1 \otimes c_\alpha`, identical to row
block B's shape with an extra `a_row` axis tensored on.

### 5.4 Evaluation: reuse `BLOCK_SUMMARY_B`

Apply the same peeled-block decomposition as W (§4.3). The block axis is
the same `B = num_blocks`, with the same `block_offset_low`, so the same
`eq_low` table works.

For the high side, build `eq_hi_t[k]` for `k ∈ [0, C · L · n_A]`.

Then for each `(a_row, dig, claim, carry)`:

```text
C row contribution = a_w[a_row] · g1[dig] · BLOCK_SUMMARY_B[claim][carry]
                                  · eq_hi_t[q_t + carry]
```

with `q_t = flat_claim + C · dig + C · L · a_row` (M-layout high index).

`BLOCK_SUMMARY_B` is the **same table** built in §4.3 for W's row block B.
The only extra precompute is `eq_hi_t`.

### 5.5 Multi-group remark

The witness `t̂` is flat-across-groups (one global `flat_claim` axis).
Multi-group affects the fused setup-matrix contribution (§7), where
the `B · t̂` rows are scanned once per commitment group. It does **not**
add a group axis to the structured T contribution: structured T folds
one `n_A`-row weight slice, `eq_τ₁[a_start..rows]`, against the shared
`c_α`-driven column shape.

### 5.6 Cost

| step | cost | done once per |
|---|---|---|
| `eq_hi_t` (size `C · L · n_A + 1`) | `O(C · L · n_A · log)` | verifier |
| `BLOCK_SUMMARY_B` (shared) | already paid in §4 | — |
| evaluate one row | `O(C · L)` | per row |
| sum over `n_A` rows | `O(n_A · C · L)` | total |

**No SIS-matrix read.** Implementation: `TStructuredSlicesEvaluator` via
the `StructuredSliceMleEvaluator` trait;
`TStructuredSlicesEvaluator { ... }.evaluate()`.

---

## 6. `z_structured_contribution` — structured rows over `ẑ`

### 6.1 What it covers

The `z` segment carries one structured row block — the consistency-check
contribution that ties `ẑ` to the openings' in-block weights:

$$
Z_\text{sep}[r, c]
\;=\; -\,\text{consistency\_weight} \cdot g_1^\text{commit}[\text{dc}] \cdot \text{fold\_gadget}[\text{df}] \cdot a_p[\text{blk}],
$$

with `a_p = opening_points[pt].a` the in-block weight vector of point `p`.

### 6.2 Format

`ẑ` has length `DF · DC · P · block_len` and the nested order
`[blk, pt, df, dc]`, so

$$
j_M^Z(\text{dc}, \text{df}, \text{pt}, \text{blk})
\;=\; \text{blk} \;+\; \text{block\_len} \cdot \bigl(\text{pt} + P \cdot \text{df} + P \cdot DF \cdot \text{dc}\bigr).
$$

Each cell carries a distinct base-field value — the `n_A` axes of `t̂`
don't appear here, and the four nested axes (`dc, df, pt, blk`) each
encode an independent dimension of the gadget × point × in-block
decomposition.

The entry formula is a tensor product:

$$
Z_\text{sep}[\cdot] \;=\; -\text{consistency\_weight} \cdot (g_1^\text{commit} \otimes \text{fold\_gadget} \otimes a_p).
$$

### 6.3 Evaluation: peeled-block over `block_len` (pow2 path)

When `block_len.is_power_of_two()`, peel the low `log₂(block_len)` bits
of `r_col` off into `eq_low_z` (size `block_len`), and the rest becomes
`eq_high_z`. Precompute the per-opening-point block summary:

$$
\text{A\_BLOCK\_SUMMARY}[pt] \;=\; \bigl[\,
\sum_{\text{blk}:\text{carry}=0} \text{eq\_low\_z}[\text{low\_idx}(\text{blk})] \cdot a_p[\text{blk}],
\quad \sum_{\text{blk}:\text{carry}=1} \dots
\,\bigr].
$$

Then for each `(pt, df, dc, carry)`:

```text
Z_sep contribution = -consistency_weight · g1_commit[dc] · fold_gadget[df]
                      · A_BLOCK_SUMMARY[pt][carry]
                      · eq_hi_z[q_z + carry]
```

with `q_z = pt + P · df + P · DF · dc`.

### 6.4 Non-power-of-two `block_len` (dense fallback)

When `block_len` is not a power of two (some recursive levels) the
peeled-block carry algebra is not defined cleanly. The verifier falls
back to materialising the structured `z` segment of length
`DF · DC · P · block_len` and calling single-factor `eval_offset_eq_tensor`
over it.

This dispatch happens at the `eval_at_point` call site: pow2 levels use
`ZStructuredPow2SlicesEvaluator`, while non-pow2 levels use
`ZDenseSlicesEvaluator`.

### 6.5 Cost

| step | cost | done once per |
|---|---|---|
| `eq_low_z` (size `block_len`) | `O(block_len)` | verifier |
| `eq_hi_z` (size `P · DF · DC + 1`) | `O(P · DF · DC · log)` | verifier |
| `A_BLOCK_SUMMARY[pt]` for `pt ∈ [0, P)` | `O(P · block_len)` | verifier (pow2 only) |
| pow2 evaluate | `O(P · DF · DC)` | total |
| non-pow2 fallback | `O(DF · DC · P · block_len)` materialisation + single-factor MLE | total |

**No SIS-matrix read.** Implementation: `ZStructuredPow2SlicesEvaluator`
via the `StructuredSliceMleEvaluator` trait for pow2 `block_len`, and
`ZDenseSlicesEvaluator` for the materialized non-pow2 fallback.

---

## 7. `setup_contribution` — fused `D · ŵ + B · t̂ + A · ẑ`

This section is self-contained: it describes both the logical setup-row
formula and the concrete column-pattern translation used by the verifier.

### 7.1 What it covers

Three row blocks of `M` are rows of the **shared SIS commitment matrix**,
evaluated at the challenge `α` to turn each cyclotomic ring entry into a
single extension-field scalar. Each row block reads a *prefix* of one
witness segment:

- **`D · ŵ`** — `n_d` rows of `D` over `ŵ`'s column range. Row weights
  `d_weights[r] = eq_τ₁[d_start + r]` for `r ∈ [0, n_d)`.
- **`B · t̂`** — `n_b` rows of `B` over `t̂`'s column range, per
  commitment group. Row weights `eq_τ₁[b_start + g · n_b + r]`.
- **`A · ẑ`** — `n_A` rows of `A` over `ẑ`'s column range. Row weights
  `a_weights[r] = eq_τ₁[a_start + r]`.

All three row blocks read different rows of the **same backing SIS
matrix**, just with different sub-views and different per-row eq weights.

### 7.2 The per-row formula

For each SIS-matrix row `r ∈ [0, r_max)` (with
`r_max = max(n_d, n_b, n_A)`):

$$
m\_eval[r]
\;=\; \sum_{c} r\_eval[c] \cdot \Bigl(\, d_w(r) \cdot W_\text{col}[c]
\;+\; \sum_g b_w^{(g)}(r) \cdot T_\text{col}^{(g)}[c]
\;+\; a_w(r) \cdot Z_\text{col}[c] \,\Bigr).
$$

- `r_eval[c] = eval_ring_at_pows(SIS_row_r[c], α)` is the SIS-matrix entry
  at row `r`, column `c`, α-evaluated to a single extension-field scalar.
  Computed *once per `(r, c)`* and shared across all three half-products.
- `W_col[c]`, `T_col^{(g)}[c]`, `Z_col[c]` are column-only eq patterns,
  precomputed once per verifier (not per row).
- `d_w(r)`, `b_w^{(g)}(r)`, `a_w(r)` are the per-row eq weights above.

`setup_contribution = Σ_r m_eval[r]`.

### 7.3 The two representations of `c`

The witness segments are stored in **M-layout** (block innermost, defines
the MLE bijection). The SIS commitment matrix is stored in **D-physical
layout** (digit innermost, the prover's natural commit-loop order). The
two are different bijections of the same logical `(claim, block, dig)`
cells.

The verifier walks `c` in D-physical order (so the dominant SIS-row scan
is contiguous) and translates each `c` to its M-layout address with a
handful of integer ops for the `eq` lookup. The column patterns bake the
translation in, so the per-row inner product is layout-agnostic.

The key invariant is that each logical cell has two coordinates: its
SIS-matrix column in D-physical order, and its M-layout address used for
the equality lookup. The translation above converts D-physical `c` into
the M-layout address for each witness segment before multiplying by the
shared SIS entry `r_eval[c]`.

### 7.4 The three column patterns

**`W_col[c]`** — eq weight at the witness M-layout address that
corresponds to D-physical column `c`. Built in `O(1)` per cell:

```text
let dig_w   = c mod L;
let b_w     = (c / L) mod B;
let claim_w = c / (B · L);
let q_w     = dig_w · C + claim_w;
let sum     = block_offset_low + b_w;
W_col[c]    = eq_low[sum mod B] · eq_hi_w[q_w + (sum / B)];
```

**`T_col^{(g)}[c]`** — same shape with extra `a_row` axis and group
sparsity. The group-local polynomial slot `poly_idx` decodes from `c`,
then `flat_claim_for_group[g][poly_idx]` resolves the global
`flat_claim` (or returns `None` if that polynomial slot is not opened).

**`Z_col[c]`** — pow2 mode uses peeled-block with a precomputed
`S_per_dc_per_carry[dc][carry]` table that absorbs the `(pt, df)`
aggregation:

```text
Z_col[c] = eq_low_z[low_idx] · S_per_dc_per_carry[dc][carry]
S_per_dc_per_carry[dc][carry]
    = -Σ_{pt, df} fold_gadget[df] · eq_hi_z[z_offset_high + (pt + P·df + P·DF·dc) + carry]
```

Non-pow2 mode builds `Z_col` densely with a one-shot peeled eq cache
(`EqPolynomial::evals` over the low `log₂(z_len)` bits + a tiny high-bit
factor table), giving the same `Vec<E>` shape with O(P · DF) per-cell
cost. The per-row inner-product loop is identical in both modes.

### 7.5 No post-loop dense fallback

The non-pow2 `block_len` case used to write its `z_a` contribution via a
post-loop dense path (`matrix_a` + `z_segment_matrix` + a single-factor
`eval_offset_eq_tensor`). That path has been folded into the main loop:
the unified `Z_col[c]` build above replaces it, so every α-eval of an
A-matrix row happens exactly once (inside the per-row loop's Z-only
branch for `r ∈ [max(n_d, n_b), n_A)`).

### 7.6 Cost

| step | cost | done once per |
|---|---|---|
| `eq_hi_w` (size `C · L + 1`) | `O(C · L · log)` | verifier |
| `eq_hi_t` (size `C · L · n_A + 1`) | `O(C · L · n_A · log)` | verifier |
| `eq_hi_z` (size `P · DF · DC + 1`) | `O(P · DF · DC · log)` | verifier |
| `S_per_dc_per_carry` (pow2 only) | `O(DC · DF · P)` | verifier |
| `W_col`, `T_col^{(g)}`, `Z_col` builds | `O(n_cols_total · (1 + G))` | verifier |
| `r_eval[c]` for `c ∈ row_range` | `O(n_cols_total · D)` extension-field ops per row | per row |
| per-row inner product | `O(n_cols_total · (1 + G))` | per row |
| **dominant** | `O(r_max · n_cols_total · D)` α-evals | total |

`r_max = max(n_d, n_b, n_A)`, `n_cols_total = max(C · B · L, max_k_g · n_A · B · L, block_len · DC)`,
`D` = ring degree.

The α-eval cost is shared between the W, T, and Z halves — that's the
"free" W + T + Z fusion. Implementation: `compute_setup_contribution`.

---

## 8. `r_contribution` — `r`-tail rows

### 8.1 What it covers

The `r`-tail is the part of `M` that captures the gadget-decomposed
"residue" rows — one row per `(row_idx, level)` pair, with row weight
`-eq_τ₁[row_idx] · denom · r_gadget[level]`.

Length of the `r`-tail in `M`'s column space: `rows · levels`, located at
column offset

```text
offset_r = w_len + d_blinding_segment_len + t_len + b_blinding_segment_len + z_len.
```

There is no separate witness for the `r`-tail; the M-table entries are
deterministically derived from `(eq_τ₁, denom, r_gadget)`.

### 8.2 Two evaluation paths

The verifier dispatches on `r_tail_dims_pow2 := levels.is_power_of_two()`:

**Pow2 path** — multi-factor `eval_offset_eq_tensor`:

```text
r_contribution = eval_offset_eq_tensor(
    full_vec_randomness,
    offset_r,
    -denom,
    &[r_gadget_ext, &eq_τ₁[..rows]],
)
```

Cost: `O(L · rows)` field ops (multi-factor MLE evaluation).

**Non-pow2 path** — materialise the r-tail vector and call single-factor
`eval_offset_eq_tensor`:

```text
r_tail[idx = level + L · row]
    = -eq_τ₁[row] · denom · r_gadget[level]                  for idx ∈ [0, r_tail_len)
r_contribution = eval_offset_eq_tensor(
    full_vec_randomness,
    offset_r,
    E::one(),
    &[r_tail],
)
```

Cost: `O(L · rows + r_tail_len)` field ops.

### 8.3 Cost summary

| `levels.is_power_of_two()` | cost |
|---|---|
| `true` | `O(L · rows)` field ops |
| `false` | `O(L · rows)` build + `O(r_tail_len)` MLE eval |

Implementation: `compute_r_contribution`. No SIS-matrix read.

---

## 9. ZK blinding contributions

These contributions exist only when the `zk` feature is enabled and are
compiled out otherwise.

### 9.1 B-blinding segment

A small per-commitment-group `t̂`-tail is appended after each group's
witness rows in the M-layout. The verifier reads it via the `B` SIS
matrix view, weighted by the per-group eq weights, and combines with
single-factor `eval_offset_eq_tensor`.

Format:

- Located at witness offset `b_blinding_segment_offset = offset_t + t_len`.
- Length `prepared.b_blinding_segment_len` (a multiple of the per-group
  blinding plane count `b_blinding_digit_planes_per_group`).
- Per-cell entry at index `idx`:

  ```text
  group_idx = idx / group_stride;
  local     = idx % group_stride;
  local_col = group_poly_counts[group_idx] · t_cols_per_claim + local;
  entry     = Σ_{row_idx ∈ [b_start + g·n_b, b_start + (g+1)·n_b)} eq_τ₁[row_idx]
                                  · eval_ring_at_pows(B[row_idx, local_col], α)
  ```

Evaluation: build the materialised segment, then call
`eval_offset_eq_tensor(full_vec_randomness, b_blinding_segment_offset,
E::one(), &[segment])`.

Cost: `O(n_b · b_blinding_segment_len · D)` α-evals + `O(b_blinding_segment_len)` MLE.

Implementation: `compute_b_blinding_part` (self-contained — derives the
`B` view, `b_start`, and `b_blinding_segment_offset` directly from
`prepared` and `setup`).

### 9.2 D-blinding segment

A global D-side tail appended after the B-blinding segment, weighted by
the `d_weights` slice of `eq_τ₁`. Same shape as §9.1 but using the `D`
view.

Format:

- Located at `d_blinding_segment_offset = b_blinding_segment_offset + b_blinding_segment_len`.
- Length `prepared.d_blinding_segment_len`.
- Per-cell entry at `idx`:

  ```text
  local_col = w_len + idx;
  entry     = Σ_{row_idx ∈ [d_start, d_start + n_d)} eq_τ₁[row_idx]
                                  · eval_ring_at_pows(D[row_idx, local_col], α)
  ```

Evaluation: same shape as §9.1.

Cost: `O(n_d · d_blinding_segment_len · D)` α-evals + `O(d_blinding_segment_len)` MLE.

Implementation: `compute_d_blinding_part` (self-contained).

### 9.3 Why these aren't fused into `setup_contribution`

The `B`- and `D`-blinding segments are *appended* to the witness, so
their column range is disjoint from the W / T / Z column ranges that the
`setup_contribution` fuses. They are also small (per-level overhead), so
the marginal cost of running a dedicated single-factor MLE evaluator on
each is negligible compared to fusing them in.

If the blinding segments ever grow large enough to dominate verifier
time, they can be folded into `compute_setup_contribution` via the same
column-pattern recipe used for W / T / Z.

---

## 10. Putting it together

`RingSwitchDeferredRowEval::eval_at_point` is the canonical
implementation. With the row-block formulas and column-pattern
translations in this spec, the function reads as the literal application
of the contributions above:

```text
fn eval_at_point(...) -> Result<E, AkitaError> {
    // Precomputes shared by multiple contributions:
    //   alpha_pows, g1_open, g1_commit, fold_gadget, r_gadget,
    //   eq_low, opening_point_block_summaries (BLOCK_SUMMARY_A),
    //   challenge_block_summaries (BLOCK_SUMMARY_B), eq_low_z,
    //   a_block_summary (A_BLOCK_SUMMARY), denom, ...

    let w_structured_contribution = WStructuredSlicesEvaluator { ... }.evaluate();   // §4
    let t_structured_contribution = TStructuredSlicesEvaluator { ... }.evaluate();   // §5
    let setup_contribution        = compute_setup_contribution(...);                  // §7
    let z_structured_contribution = if block_len.is_power_of_two() {                 // §6
        ZStructuredPow2SlicesEvaluator { ... }.evaluate()
    } else {
        ZDenseSlicesEvaluator { ... }.evaluate()
    };
    let r_contribution            = compute_r_contribution(...);                      // §8

    let mut total = w_structured_contribution + t_structured_contribution
                  + z_structured_contribution + setup_contribution
                  + r_contribution;

    #[cfg(feature = "zk")] {
        total += compute_b_blinding_part(...);  // §9.1
        total += compute_d_blinding_part(...);  // §9.2
    }

    Ok(total)
}
```

Each contribution is independently testable (the per-block evaluators
have their own correctness tests against materialised reference rows), and
each follows the same internal pattern: derive the row block's algebraic
structure, choose the matching MLE technique, pay for `eq_low` / `eq_hi`
tables once, and emit a single scalar.

---

## References

### Code

- `crates/akita-verifier/src/protocol/ring_switch.rs` — top-level
  `RingSwitchDeferredRowEval::eval_at_point` orchestrates every
  contribution below.
- `crates/akita-verifier/src/protocol/slice_mle/` — per-contribution
  helpers:
  - `WStructuredSlicesEvaluator`, `TStructuredSlicesEvaluator`,
    `ZStructuredPow2SlicesEvaluator`, `ZDenseSlicesEvaluator` — structured
    slice evaluators.
  - `compute_setup_contribution` — fused setup-matrix contribution.
  - `compute_r_contribution` — `r`-tail.
  - `compute_b_blinding_part`, `compute_d_blinding_part` — ZK blinding.
- `crates/akita-algebra/src/offset_eq.rs` — `eval_offset_eq_tensor`,
  peeled-block primitives.

### Protocol

- Hachi paper, equation (20) — the original block-matrix view that this
  spec expands.
