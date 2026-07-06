# Spec: Two-Factor Sparse/Pruned Equality Binding (pow2 `r`-tail)

| Field | Value |
|-------|-------|
| Author(s) | Omid Bodaghi, Cursor |
| Created | 2026-07-03 |
| Status | theory / design note |
| Related code | `crates/akita-algebra/src/offset_eq.rs`, `crates/akita-verifier/src/protocol/slice_mle/structured_slice.rs` |
| Related drafts | `specs/sparse-pruned-eq-binding.md`, `specs/eval-eq-simplification.md` |

## Summary

`specs/sparse-pruned-eq-binding.md` showed that a **single** materialized
factor over a contiguous global interval can be evaluated by pruned multilinear
binding (`eval_offset_eq_interval`) at ~6–9× the throughput of the carry DP on
the `offset != 0` case.

One production call site is not single-factor: the pow2 branch of
`compute_r_contribution` passes **two** factors to `eval_offset_eq_tensor`.

```313:321:crates/akita-verifier/src/protocol/slice_mle/structured_slice.rs
    if levels.is_power_of_two() {
        let _span = tracing::info_span!("r_structured").entered();
        let r_gadget_ext: Vec<E> = r_gadget.iter().copied().map(E::lift_base).collect();
        eval_offset_eq_tensor(
            full_vec_randomness,
            offset_r,
            -denom,
            &[&r_gadget_ext, &prepared.eq_tau1[..prepared.rows]],
        )
```

Because `offset_r != 0` in production, this currently runs the general 2×2
carry DP (`eval_offset_eq_tensor_carry`) — the same machinery the single-factor
work already replaced.

This note shows that the two-factor pow2 case **can** be supported with the same
efficiency as the single-factor sparse path (and the same ~8× constant-factor
win over the carry DP), **without materializing** the `levels × rows` tensor.
The tensor decomposes exactly into:

1. a one-shot **inner** two-bucket summary of the pow2 low factor, and
2. **two** shifted single-factor `eval_offset_eq_interval` calls over the small
   high factor.

No new algorithm is required beyond `eval_offset_eq_interval` plus the existing
peel primitives; the whole thing is `O(levels + rows)` scalar multiplications.
This document is theory only — no implementation.

## The computation

Let

```text
f0 = r_gadget          (length levels = 2^m0, power of two in this branch)
f1 = eq_tau1[..rows]   (length rows, small: ~3..10)
scale = -denom
n  = full_vec_randomness.len()   (= col_bits)
```

`eval_offset_eq_tensor` orders factors least-significant first, so `f0` occupies
the low `m0` bits and `f1` the next bits. The flattened index is

```text
z = level + levels * row,     level in [0, levels),  row in [0, rows)
```

matching the non-pow2 branch's materialization (`row = idx / levels`,
`level = idx % levels`). The target is

```text
S = scale * sum_{row=0}^{rows-1} sum_{level=0}^{levels-1}
        eq(r, offset_r + level + levels*row) * f0[level] * f1[row].
```

The value is a **rank-1 tensor**: `value(z) = f0[level] * f1[row]`.

## Why single-factor sparse binding is not enough by itself

`eval_offset_eq_interval` takes one materialized vector. To use it directly on
this case we would have to materialize the product

```text
seg[level + levels*row] = f0[level] * f1[row]
```

which is `levels * rows` entries — exactly the dense fallback the non-pow2
branch already uses, and `O(levels * rows)` work. That defeats the purpose. We
need to exploit the tensor structure so the cost stays `O(levels + rows)`.

## The decomposition

Split the offset against the low (pow2) factor window:

```text
m0        = log2(levels)
offset_lo = offset_r mod levels          (= offset_r & (levels - 1))
offset_hi = floor(offset_r / levels)     (= offset_r >> m0)
```

Because `levels` is a power of two, `level` occupies exactly the low `m0` bits of
`z`, and `row` the high bits. The inner sum `s = offset_lo + level` lies in
`[0, 2*levels - 2]`, so it produces **at most one carry bit**. The equality
polynomial is fully multiplicative across bits, so it factors:

```text
eq(r, offset_r + level + levels*row)
  = eq_low( (offset_lo + level) mod levels )
  * eq_high( offset_hi + row + carry ),
    carry = floor( (offset_lo + level) / levels ) in {0, 1}
```

where `eq_low` uses the low `m0` challenges `r[..m0]` and `eq_high` uses the high
challenges `r[m0..]`.

### Step 1 — inner two-bucket summary (once, `O(levels)`)

The low factor `f0` does not depend on `row`, so its equality-weighted sum over
`level`, split by carry, is computed **once**:

```text
A_c = sum_{level : carry(level) = c} f0[level] * eq_low( (offset_lo + level) mod levels )
```

giving two field elements `[A0, A1]`. This is exactly

```text
eq_low       = EqPolynomial::evals(r[..m0])            // O(levels)
[A0, A1]     = summarize_pow2_block_carries(eq_low, offset_lo, f0)   // O(levels)
```

(or the base×ext variant `summarize_pow2_block_carries_base` to avoid lifting
`r_gadget` into `E`; see `specs/eval-eq-simplification.md`).

### Step 2 — outer sum as two shifted single-factor interval evals

Substituting the factorization and pulling the `row`-independent buckets out:

```text
S / scale
  = sum_row f1[row] * ( A0 * eq_high(offset_hi + row) + A1 * eq_high(offset_hi + 1 + row) )
  = A0 * [ sum_row f1[row] * eq_high(offset_hi     + row) ]
  + A1 * [ sum_row f1[row] * eq_high(offset_hi + 1 + row) ].
```

Each bracket is a **single-factor offset-equality over the outer dimension** —
precisely what `eval_offset_eq_interval` computes over the high challenges:

```text
Bracket0 = eval_offset_eq_interval(r[m0..], offset_hi,     1, f1)
Bracket1 = eval_offset_eq_interval(r[m0..], offset_hi + 1, 1, f1)
```

so the entire two-factor result is

```text
S = scale * ( A0 * Bracket0 + A1 * Bracket1 ).
```

That is the whole method: one inner pow2 summary and two outer sparse interval
evaluations of the small factor, combined with the two carry buckets.

### Relationship to the existing peel primitives

This is algebraically identical to the peel already in the codebase. Setting

```text
carry_terms[row] = [ f1[row] * A0, f1[row] * A1 ]
```

and calling

```text
eval_offset_eq_peeled_carry_terms(full_vec_randomness, offset_r, m0, carry_terms)
```

computes `sum_row carry_terms[row][0]*eq_high(offset_hi+row) +
carry_terms[row][1]*eq_high(offset_hi+row+1)`, which equals `A0*Bracket0 +
A1*Bracket1`. So two equivalent framings exist:

- **Peel framing (matches `specs/eval-eq-simplification.md`):**
  `summarize_pow2_block_carries` + `eval_offset_eq_peeled_carry_terms`.
- **Sparse-binding framing (this note):** `A0 * interval(offset_hi) +
  A1 * interval(offset_hi + 1)`.

Both are `O(levels + rows)` with scalar (not 2×2) multiplications. The peel
framing reuses primitives already tested and living in `offset_eq.rs`; the
sparse framing reuses `eval_offset_eq_interval` from
`specs/sparse-pruned-eq-binding.md`. They compute the same value.

## Correctness

The only nontrivial step is the equality factorization, which is exact because:

1. `eq` is a product over bit positions.
2. `levels = 2^m0` means `level` is exactly the low `m0` bits and `row` the high
   bits of `z`; there is no bit sharing between the factors.
3. `s = offset_lo + level in [0, 2*levels - 2]` carries at most one bit, so the
   coupling between the low and high halves is a single carry `c in {0,1}`, and
   `A0/A1` capture both cases.

The outer step is pure distribution of the sum over the two carry buckets, then
recognizing each bucket's `row`-sum as a single-factor interval evaluation with
high offset `offset_hi` and `offset_hi + 1`. Indices `offset_hi + 1 + row` that
reach `2^{n-m0}` fall outside the equality domain and contribute zero;
`eval_offset_eq_interval` already clamps out-of-domain indices to zero, matching
`eval_offset_eq_tensor` semantics, so no special handling is needed at the top of
the window.

## Cost

| Stage | Work |
|-------|------|
| `eq_low = evals(r[..m0])` | `O(levels)` |
| `summarize_pow2_block_carries` | `O(levels)` |
| `Bracket0`, `Bracket1` (two interval evals over `f1`) | `2 * O(rows + (n - m0))` |
| final combine `A0*Bracket0 + A1*Bracket1` | `O(1)` |
| **total** | **`O(levels + rows + n)`**, all scalar mults |

This matches the asymptotics of today's carry-DP fast path, but replaces the
2×2 carry-matrix arithmetic with scalar binding. Based on the single-factor
measurements in `specs/sparse-pruned-eq-binding.md` (carry DP does ~8 mults per
fold pair vs 1 for sparse), the expected constant-factor improvement on the
`r_structured` span is the same ~6–9× seen for the single-factor carry case —
though `r_structured` is already a tiny share of verify (≈2–8 µs per fold,
per `specs/eval-eq-simplification.md`), so the wall-clock impact is negligible.
The value here is **uniformity**: the pow2 `r` case stops needing the carry DP,
letting the tensor/carry machinery be deleted (the stated goal of
`specs/eval-eq-simplification.md`).

## Scope and limits

- **This handles exactly the pow2 two-factor shape** in the `r` branch: the low
  factor's width is a power of two (`levels.is_power_of_two()` — the branch
  condition), so the single-carry peel is well defined. This is guaranteed here.
- **Non-pow2 low factor** (the `else` branch, e.g. `log_basis` 3/5) has no clean
  low-bit window; the low factor does not occupy a contiguous power-of-two bit
  range, so the two-bucket split does not apply. That branch stays on the
  on-the-fly `eq_eval_at_index` loop (already the plan in
  `specs/eval-eq-simplification.md`), or the dense materialization fallback.
- **More than two factors:** each internal factor boundary that is pow2-aligned
  introduces one carry bit. Two factors have a single boundary → two buckets,
  handled cleanly. `k` pow2 factors have `k-1` boundaries whose carries chain;
  supporting them in the sparse framing means threading a small carry state
  between successive interval folds (equivalently, the peel applied
  recursively). This is exactly the bookkeeping the 2×2 carry DP generalizes.
  Out of scope here — the `r`-tail is always two factors.

## Proposed shape (not implemented)

Two equivalent options; pick one during implementation review.

**Option A — reuse peel primitives (smallest new surface).** Rewrite the pow2
branch of `compute_r_contribution` to:

```text
m0        = levels.trailing_zeros()
eq_low    = EqPolynomial::evals(&r[..m0])
offset_lo = offset_r & (levels - 1)
[A0, A1]  = summarize_pow2_block_carries_base(&eq_low, offset_lo, r_gadget)
carry_terms[row] = [ eq_tau1[row] * A0, eq_tau1[row] * A1 ]   for row in 0..rows
combined  = eval_offset_eq_peeled_carry_terms(r, offset_r, m0, &carry_terms)
result    = -denom * combined
```

This is the `specs/eval-eq-simplification.md` plan verbatim; it needs no new
public function.

**Option B — sparse-binding vocabulary (reuses `eval_offset_eq_interval`).**

```text
m0        = levels.trailing_zeros()
eq_low    = EqPolynomial::evals(&r[..m0])
offset_lo = offset_r & (levels - 1)
offset_hi = offset_r >> m0
[A0, A1]  = summarize_pow2_block_carries_base(&eq_low, offset_lo, r_gadget)
b0        = eval_offset_eq_interval(&r[m0..], offset_hi,     E::one(), eq_tau1[..rows])
b1        = eval_offset_eq_interval(&r[m0..], offset_hi + 1, E::one(), eq_tau1[..rows])
result    = -denom * (A0 * b0 + A1 * b1)
```

Both are `O(levels + rows)`, both avoid the `levels × rows` materialization, and
both eliminate the carry DP for this call site.

## Recommendation

The two-factor pow2 `r`-tail is supportable at the same efficiency as the
single-factor sparse path. Prefer **Option A** if the goal is to delete the
carry DP with minimal new API (it is already the `eval-eq-simplification.md`
plan). Prefer **Option B** if we want one uniform sparse-binding entry point
(`eval_offset_eq_interval`) across single- and two-factor cases. Either way, no
new asymptotics and no tensor materialization are introduced.
