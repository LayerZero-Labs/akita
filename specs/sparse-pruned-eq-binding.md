# Spec: Sparse/Pruned Equality Binding for Partial MLE Intervals

| Field | Value |
|-------|-------|
| Author(s) | Omid Bodaghi, Cursor |
| Created | 2026-07-03 |
| Status | theory / design note |
| Related code | `crates/akita-algebra/src/offset_eq.rs` |
| Related draft | `specs/eval-eq-simplification.md` |

## Summary

This note compares two ways to compute a partial equality-weighted multilinear
evaluation over a contiguous global interval. The concrete motivating case is:

```text
N = 32 = 2^5
r = (r_0, r_1, r_2, r_3, r_4)
offset = 11
len = 21
global indices = 11..31
```

Given local values `V[0..21)`, where `V[z]` corresponds to global index
`i = 11 + z`, the target is:

```text
S = sum_{z=0}^{20} eq(r, 11 + z) * V[z]
```

Equivalently, define a global length-32 vector `M'`:

```text
M'[i] = 0                 for i < 11
M'[i] = V[i - 11]         for 11 <= i <= 31
```

Then:

```text
S = sum_{i=0}^{31} eq(r, i) * M'[i]
  = sum_{i=11}^{31} eq(r, i) * V[i - 11]
```

The current `eval_offset_eq_tensor_carry` implementation computes this by
keeping the vector in local coordinates and accounting for `offset + z` via a
2x2 carry dynamic program. The proposed sparse/pruned binding computes the same
quantity by placing the values in global coordinates first, treating entries
outside `[11,31]` as zero, and running the standard multilinear binding fold
only over live branches.

For this concrete one-factor interval:

```text
current carry-DP, as implemented: 257 field multiplications
current carry-DP, core summary only: 248 field multiplications
full length-32 binding: 31 field multiplications
sparse/pruned binding: 23 field multiplications
```

Therefore, for a single contiguous interval such as `[11,31]`, sparse/pruned
binding is theoretically much cheaper than the current generic carry-DP.

## Background: Equality Basis and Standard Binding

Akita uses little-endian bit order. For an index

```text
i = i_0 + 2 i_1 + 4 i_2 + 8 i_3 + 16 i_4,
```

the equality basis weight is:

```text
eq(r, i)
  = product_t eq_bit(r_t, i_t)

where:

eq_bit(r_t, 0) = 1 - r_t
eq_bit(r_t, 1) = r_t
```

For a full length-`2^n` vector `M`, its multilinear evaluation at `r` is:

```text
M~(r) = sum_{i=0}^{2^n-1} eq(r, i) * M[i]
```

The usual in-place binding algorithm folds one bit at a time:

```text
bind(x0, x1, r_k) = (1 - r_k) * x0 + r_k * x1
                  = x0 + r_k * (x1 - x0)
```

Each live parent pair costs one field multiplication by `r_k`. A full
length-32 fold costs:

```text
16 + 8 + 4 + 2 + 1 = 31 multiplications
```

## Current Approach: Offset Carry-DP

### What It Computes

The current helper is designed for the more general tensor expression:

```text
sum_z eq(r, offset + z) * scale * product_j factors[j][idx_j(z)]
```

For the one-factor case, this becomes:

```text
sum_z eq(r, offset + z) * scale * factor[z]
```

The factor is internally padded to the next power of two. For `len = 21`:

```text
padded_len = 32
m = log2(32) = 5
```

So the current path conceptually evaluates over local indices `z = 0..31`,
where entries `z = 21..31` are zero.

### Why Carries Appear

If `offset = 0`, the global index is just `z`, so each bit of `z` independently
selects either `r_t` or `1 - r_t`. That is the aligned case and can be computed
using ordinary MLE binding.

With `offset = 11`, the global index is:

```text
global = 11 + z
```

Binary addition can carry from low bits into high bits. The bit at position `t`
of `11 + z` is not only a function of `z_t`; it also depends on the incoming
carry from lower positions.

At each bit:

```text
sum_t = offset_bit_t + local_bit_t + carry_in
global_bit_t = sum_t mod 2
carry_out = floor(sum_t / 2)
```

The equality weight for that bit is:

```text
global_bit_t = 0 -> multiply by 1 - r_t
global_bit_t = 1 -> multiply by r_t
```

Because `carry_in` is either 0 or 1, the implementation summarizes partial
work with a 2x2 matrix:

```text
matrix[carry_in][carry_out]
```

Each entry stores the total weighted contribution that starts with one carry
state and exits with another.

### Current Cost for `len = 21`, `offset = 11`

The key primitive is `CarryMatrix::mul_transition`. It right-multiplies a 2x2
carry matrix by one sparse bit transition and performs 4 field multiplications.

During `factor_summary`, every pair in the padded binary fold is processed by
two transitions:

```text
local bit 0 transition: 4 multiplications
local bit 1 transition: 4 multiplications
total per pair: 8 multiplications
```

For a padded length-32 factor, the fold has:

```text
round 0: 16 pairs
round 1:  8 pairs
round 2:  4 pairs
round 3:  2 pairs
round 4:  1 pair
total:   31 pairs
```

So the factor summary costs:

```text
31 * 8 = 248 field multiplications
```

Then `eval_offset_eq_tensor_carry` composes this summary into the accumulated
matrix:

```text
composed = composed.mul_matrix(summary)
```

A dense 2x2 matrix multiply costs 8 field multiplications.

Finally:

```text
scale * composed[0][0]
```

costs 1 more multiplication.

Assuming exactly 5 challenge bits and no tail bits:

```text
factor_summary:       248
compose summary:        8
final scale:            1
-------------------------
implemented total:    257 field multiplications
```

If `r.len() > 5`, the tail loop processes the remaining bits with `local_bit =
0`, adding one `mul_transition` per tail bit:

```text
additional tail cost = 4 * (r.len() - 5)
```

So the current implemented count is:

```text
257 + 4 * (r.len() - 5)
```

for `r.len() >= 5`.

> **Caveat (what 257 does and does not measure).** This is the count of `*`
> operations the current code executes, not the intrinsic cost of a carry-based
> method. Several of those products are structurally trivial:
>
> - In fold round 0 the accumulator matrices are diagonal (`scaled(val)`), so
>   half of each `mul_transition`'s 4 products are `x * 0`.
> - The `mul_matrix` compose (8 mults) multiplies by the identity, since
>   `composed` starts as `identity` for a single factor.
> - The final `scale` multiply is `x * 1` when `scale = 1` (the single-factor `z`
>   and the `ZDenseSlicesEvaluator` call site both pass `E::one()`).
>
> A carry method that exploited this sparsity would run below 257, but still far
> above 23. The qualitative conclusion (sparse/pruned binding is dramatically
> cheaper for one contiguous interval) is therefore robust; treat 257 as "cost
> of the current general implementation on this shape," and note that even naive
> full-length binding (31) already beats it.

### Strengths of the Current Approach

The carry-DP is general:

- It handles arbitrary nonzero offsets.
- It handles multiple tensor factors without materializing the full tensor.
- It composes factor summaries through carry states.
- It keeps the local tensor structure intact.

This generality matters for expressions like:

```text
factor0[idx0] * factor1[idx1] * factor2[idx2]
```

where materializing the full tensor may be too expensive.

### Weakness for the One-Factor Interval Case

For a single contiguous global interval, the carry-DP is overkill. It pays a
matrix cost for each binary fold pair even though the desired computation can
be seen as an ordinary partial MLE over a global vector with leading zeros.

The carry-DP is solving:

```text
local index z -> shifted global index offset + z
```

Sparse/pruned binding instead removes the shift from the evaluation by moving
the vector into global coordinates before binding.

## Proposed Approach: Sparse/Pruned Binding

### Main Idea

Define:

```text
M'[i] = 0                 for i < 11
M'[i] = V[i - 11]         for 11 <= i <= 31
```

Then compute:

```text
S = M'~(r)
```

using the standard multilinear binding algorithm, but skip all parent nodes
whose entire subtree is zero.

This avoids the expression `offset + z` during the fold. Once the values are
viewed as a global vector, the equality weight is aligned with the vector index:

```text
eq(r, i) * M'[i]
```

No carry machinery is needed.

### Sparse Pair Rules

At each binding round, each touched parent pair has one of four cases.

For pair `(x0, x1)`:

1. Both children live:

```text
parent = x0 + r_k * (x1 - x0)
```

2. Only the right child lives:

```text
parent = r_k * x1
```

3. Only the left child lives:

```text
parent = (1 - r_k) * x0
       = x0 - r_k * x0
```

4. Neither child lives:

```text
skip
```

Every live parent costs one multiplication. Some cases also need additions or
subtractions, but the multiplication count is the main comparison because field
multiplications dominate.

### Concrete Walkthrough for `[11,31]`

The active interval starts as:

```text
11, 12, 13, ..., 31
```

#### Round 0: Bind by `r_0`

Touched parent pairs:

```text
(10,11), (12,13), (14,15), ..., (30,31)
```

The pair `(10,11)` has only the right child live. All others have both children
live.

Definitions:

```text
T_5 = r_0 * M'[11]

for j = 6..15:
    T_j = M'[2j] + r_0 * (M'[2j + 1] - M'[2j])
```

Cost:

```text
11 multiplications
```

The new active interval is:

```text
5..15
```

#### Round 1: Bind by `r_1`

Touched parent pairs:

```text
(4,5), (6,7), (8,9), (10,11), (12,13), (14,15)
```

The pair `(4,5)` has only the right child live. The rest have both children
live.

Definitions:

```text
U_2 = r_1 * T_5

for j = 3..7:
    U_j = T_{2j} + r_1 * (T_{2j + 1} - T_{2j})
```

Cost:

```text
6 multiplications
```

The new active interval is:

```text
2..7
```

#### Round 2: Bind by `r_2`

Touched parent pairs:

```text
(2,3), (4,5), (6,7)
```

All touched pairs have both children live.

Definitions:

```text
for j = 1..3:
    V_j = U_{2j} + r_2 * (U_{2j + 1} - U_{2j})
```

Cost:

```text
3 multiplications
```

The new active interval is:

```text
1..3
```

#### Round 3: Bind by `r_3`

Touched parent pairs:

```text
(0,1), (2,3)
```

The pair `(0,1)` has only the right child live. The pair `(2,3)` has both
children live.

Definitions:

```text
W_0 = r_3 * V_1
W_1 = V_2 + r_3 * (V_3 - V_2)
```

Cost:

```text
2 multiplications
```

The new active interval is:

```text
0..1
```

#### Round 4: Bind by `r_4`

Touched parent pair:

```text
(0,1)
```

Both children live.

Definition:

```text
S = W_0 + r_4 * (W_1 - W_0)
```

Cost:

```text
1 multiplication
```

### Total Proposed Cost

```text
round 0: 11
round 1:  6
round 2:  3
round 3:  2
round 4:  1
---------------
total:   23 field multiplications
```

This is less than the full length-32 fold:

```text
31 field multiplications
```

and much less than the current implemented carry-DP:

```text
257 field multiplications
```

## General Sparse Binding Algorithm

For a contiguous interval `[a,b]` inside `[0, 2^n - 1]`, maintain the current
active index range at each round.

Pseudo-code (all index ranges are **inclusive** on both ends):

```text
lo = a
hi = b
A[i] = M[i] for i in a..=b

for k in 0, 1, ..., n-1:
    B = empty

    new_lo = floor(lo / 2)
    new_hi = floor(hi / 2)

    for p in new_lo..=new_hi:
        left_index  = 2p
        right_index = 2p + 1

        has_left  = lo <= left_index  <= hi
        has_right = lo <= right_index <= hi

        if has_left and has_right:
            B[p] = A[left_index] + r[k] * (A[right_index] - A[left_index])

        else if has_left:
            B[p] = A[left_index] - r[k] * A[left_index]

        else if has_right:
            B[p] = r[k] * A[right_index]

        else:
            skip

    A = B
    lo = new_lo
    hi = new_hi

return A[0]
```

The `else: skip` case should not occur if the loop only ranges over
`new_lo..new_hi`, but it is useful for expressing the invariant.

### General Multiplication Count

At round `k`, where `k = 0` binds the least-significant bit, the number of live
parent nodes is:

```text
floor(b / 2^{k+1}) - floor(a / 2^{k+1}) + 1
```

Therefore the total multiplication count is:

```text
sum_{k=0}^{n-1}
  (floor(b / 2^{k+1}) - floor(a / 2^{k+1}) + 1)
```

For `[a,b] = [11,31]`, `n = 5`:

```text
k=0: floor(31/2)  - floor(11/2)  + 1 = 15 - 5 + 1 = 11
k=1: floor(31/4)  - floor(11/4)  + 1 =  7 - 2 + 1 =  6
k=2: floor(31/8)  - floor(11/8)  + 1 =  3 - 1 + 1 =  3
k=3: floor(31/16) - floor(11/16) + 1 =  1 - 0 + 1 =  2
k=4: floor(31/32) - floor(11/32) + 1 =  0 - 0 + 1 =  1
```

Total:

```text
11 + 6 + 3 + 2 + 1 = 23
```

## Alternative View: Dyadic Block Decomposition

The interval `[11,31]` can also be decomposed into dyadic blocks:

```text
[11,11] union [12,15] union [16,31]
```

Each dyadic block is a subcube with some fixed bits and some free bits.

For each block:

1. Multiply by equality factors for fixed bits.
2. Run an ordinary MLE fold over free bits.
3. Sum the block contributions.

This is mathematically equivalent to sparse/pruned binding. Sparse/pruned
binding is usually easier to implement because it does not require explicitly
enumerating subcubes or managing fixed-bit masks.

## Efficiency Comparison

### Concrete Case

For `N = 32`, interval `[11,31]`:

| Method | Field multiplications | Notes |
|--------|-----------------------|-------|
| Materialize eq weights + inner product | about `5 * 21 + 21` | `eq_eval_at_index` per entry, then multiply by value; exact count depends on whether weights are reused |
| Full length-32 binding | 31 | Evaluates all 32 entries, including zeros |
| Sparse/pruned binding | 23 | Only folds live parent nodes |
| Current carry-DP core summary | 248 | 31 pairs * 8 matrix-transition multiplications |
| Current carry-DP as implemented | 257 | Summary + 2x2 compose + final scale |

The sparse/pruned approach wins decisively for the one-factor interval.

### Why Sparse Binding Wins Here

Sparse binding is able to use the ordinary scalar binding identity:

```text
x0 + r * (x1 - x0)
```

The current carry-DP has to use matrix transitions because it evaluates:

```text
eq(r, offset + z)
```

while keeping the vector indexed by local `z`. That local representation forces
the algorithm to account for binary carry propagation.

Sparse binding changes representation:

```text
local shifted vector -> global vector with zeros outside the active interval
```

Once the vector is in global coordinates, the problem becomes an aligned MLE
again.

### When the Current Carry-DP May Still Be Better

Sparse/pruned binding is not automatically a replacement for every use of
`eval_offset_eq_tensor_carry`.

The carry-DP handles rank-1 tensor factors without materializing their full
product:

```text
value(z) = factor0[idx0(z)] * factor1[idx1(z)] * ... * factork[idxk(z)]
```

For many factors, sparse binding over global coordinates may require
materializing the entire active tensor interval unless we design a structured
sparse binding algorithm. In such cases, the current carry-DP's matrix overhead
may be worth paying because it preserves tensor factorization.

The proposed sparse binding is most attractive when:

- There is a single materialized factor.
- The active region is a contiguous global interval.
- The ambient domain size `2^n` is known.
- The interval is significantly smaller than the full domain, or starts/end
  boundaries create enough pruning.

It is less obviously suitable when:

- The value is represented as multiple tensor factors.
- The active support is not a simple interval.
- Avoiding materialization is more important than minimizing per-pair
  multiplications.

## Correctness Argument

The current and proposed methods compute the same polynomial evaluation because
they differ only in representation.

Current representation:

```text
S = sum_{z=0}^{20} eq(r, 11 + z) * V[z]
```

Global sparse representation:

```text
M'[i] = V[i - 11] for 11 <= i <= 31
M'[i] = 0 otherwise
```

Then:

```text
M'~(r)
  = sum_{i=0}^{31} eq(r, i) * M'[i]
  = sum_{i=11}^{31} eq(r, i) * V[i - 11]
```

Substitute `i = 11 + z`:

```text
M'~(r)
  = sum_{z=0}^{20} eq(r, 11 + z) * V[z]
  = S
```

Sparse binding is just the standard MLE binding algorithm applied to `M'`, with
zero subtrees skipped. Since skipped subtrees contain only zeros, pruning does
not change the result.

## Open Design Questions

1. API shape:

```text
eval_eq_interval(r, offset, values, domain_bits)
```

or:

```text
sparse_mle_interval(r, lo, values, domain_bits)
```

2. Domain sizing:

The caller must provide or imply the ambient domain size. For the motivating
case, `domain_bits = 5` and `N = 32`. This is different from today's
`eval_offset_eq_tensor`, which infers the local padded factor width but uses
`x_challenges.len()` for the equality domain.

3. Out-of-domain intervals:

If `offset + values.len() > 2^domain_bits`, the implementation must define
whether to reject the input or truncate out-of-domain values. The current
`eq_eval_at_index` returns zero for indices outside the challenge domain.

4. Multi-factor extension:

A future structured sparse algorithm may combine interval pruning with tensor
factorization. That is a separate design from the simple one-factor interval
binding described here.

## Recommendation

For the concrete one-factor case:

```text
offset = 11
len = 21
domain = 32
target = indices 11..31
```

the sparse/pruned binding algorithm is the better theory-level approach. It
computes the exact same partial MLE in 23 field multiplications, compared with
257 field multiplications in the current generic carry-DP implementation.

The current carry-DP should be viewed as a general tensor/offset mechanism.
Sparse/pruned binding should be viewed as a specialized fast path for
materialized one-factor contiguous intervals in global index space.
