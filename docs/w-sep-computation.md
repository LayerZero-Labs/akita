# Computing `w_sep`

This note focuses only on the `w_sep` term in the verifier-side ring-switch
replay:

```rust
let w_sep = {
    let _span = tracing::info_span!("m_eval_w_sep").entered();
    eval_offset_eq_peeled_carry_terms(x_challenges, offset_w, block_bits, &w_carry_terms)
};
```

The goal is to explain the naive computation first, then explain how the current
implementation computes the same value more efficiently.

## 1. What `w_sep` Represents

In the paper matrix, the `\hat w` columns appear in several rows:

```text
D · \hat w                         = v
b^T G_{2^r} · \hat w               = y
(c^T ⊗ G_1) · \hat w - ... \hat z  = 0
```

The verifier samples `tau1`, takes a random linear combination of the rows, and
then evaluates the resulting virtual row-vector `M` at a multilinear point.

The `\hat w` part of that virtual row-vector has two pieces:

```text
w_part = w_sep + w_d
```

Here:

- `w_d` is the contribution from the `D · \hat w = v` row.
- `w_sep` is the structured separable contribution from:
  - `b^T G_{2^r} · \hat w = y`
  - `(c^T ⊗ G_1) · \hat w - ... = 0`

So this note ignores the `D` row and focuses only on:

```text
public/opening-point contribution + consistency/challenge contribution
```

## 2. Shape Of The `\hat w` Segment

Let:

```text
B = num_blocks
C = num_claims
L = depth_open
```

The `\hat w` segment has length:

```text
w_len = L · C · B
```

The local index inside the `\hat w` segment is organized as:

```text
local_index = dig · (C · B) + claim_idx · B + block_idx
```

where:

```text
dig       ∈ [0, L)
claim_idx ∈ [0, C)
block_idx ∈ [0, B)
```

Equivalently:

```text
local_index = block_idx + B · q
```

where:

```text
q = dig · C + claim_idx
```

The full virtual `M` table contains several concatenated segments. The `\hat w`
segment starts at `offset_w`, so the actual global index in `M` is:

```text
global_index = offset_w + local_index
```

Therefore:

```text
global_index = offset_w + block_idx + B · q
```

## 3. The Naive Computation

The verifier wants the multilinear-extension evaluation of this segment at
`x_challenges`.

Before writing the naive loop, define the objects that appear in one term of
the sum.

`claim_idx` identifies which evaluation claim we are processing. In the batched
protocol, several claims can share the same evaluation point. The array
`claim_to_point` tells us which opening point belongs to each claim:

```text
point_idx = claim_to_point[claim_idx]
```

So `point_idx` is not a new protocol challenge. It is just an index into the
list of distinct opening points.

`opening_points[point_idx]` is the ring-level opening point for that claim. In
this implementation, an opening point has two structured pieces:

```text
opening_point.a
opening_point.b
```

For `w_sep`, only the `b` part is used:

```text
opening_points[point_idx].b[block_idx]
```

This corresponds to the paper's `b^T G_{2^r}` row. The value `b[block_idx]`
selects the coefficient attached to this `block_idx`.

`g_open` is the gadget vector for the decomposition of `\hat w`. If a ring or
field element is decomposed into `depth_open` digits in base `2^log_basis`, then
`g_open[dig]` is the scalar weight of digit `dig`.

Conceptually:

```text
G_open = [1, base, base^2, ..., base^{depth_open-1}]
```

up to the exact balanced-digit convention used by the implementation. In the
formula, multiplying by `g_open[dig]` accounts for the gadget matrix `G_1` or
`G_{2^r}` touching digit `dig` of `\hat w`.

`gamma[claim_idx]` is the batching coefficient for this claim. When multiple
evaluation claims are batched together, the verifier combines them using random
linear-combination weights. `gamma[claim_idx]` is the weight for this specific
claim.

With those definitions, the fully naive approach is:

```text
w_sep = 0

for dig in 0..depth_open:
  for claim_idx in 0..num_claims:
    for block_idx in 0..num_blocks:

      point_idx = claim_to_point[claim_idx]

      value =
        g_open[dig] · (
            public_weights[point_idx]
          · gamma[claim_idx]
          · opening_points[point_idx].b[block_idx]
          +
            consistency_weight
          · c_alphas[claim_idx · num_blocks + block_idx]
        )

      local_index =
          dig · (num_claims · num_blocks)
        + claim_idx · num_blocks
        + block_idx

      global_index = offset_w + local_index

      w_sep += value · eq(x_challenges, global_index)
```

This is the definition to keep in mind. Everything in the optimized code is just
a rearrangement of this sum.

In summation form:

```text
w_sep =
Σ_{dig, claim, block}
  g_open[dig]
  · (
      public_weight(point(claim)) · gamma[claim] · b_point[block]
    + consistency_weight · c_alpha[claim, block]
    )
  · eq_x(offset_w + dig·C·B + claim·B + block)
```

Using `q = dig·C + claim`, this becomes:

```text
w_sep =
Σ_{q, block}
  value[q, block]
  · eq_x(offset_w + block + B·q)
```

where `value[q, block]` means the coefficient inside the previous formula.

## 4. Why The Naive Computation Has Structure

The important fact is:

```text
B = num_blocks = 2^block_bits
```

Let:

```text
m = block_bits
B = 2^m
```

Then:

```text
global_index = offset_w + block_idx + 2^m · q
```

The `block_idx` part occupies exactly the low `m` bits. The `q` part occupies
the remaining high bits.

If `offset_w` were aligned to a multiple of `2^m`, then the equality polynomial
would split very simply:

```text
eq_x(offset_w + block_idx + 2^m q)
=
eq_low(block_idx) · eq_high(offset_high + q)
```

But `offset_w` is not always aligned. Its low bits can shift the block index and
cause a carry into the high bits. That is the only subtlety.

## 5. Splitting `offset_w` Into Low And High Pieces

Write:

```text
offset_w = offset_low + 2^m · offset_high
```

where:

```text
offset_low  = offset_w mod 2^m
offset_high = floor(offset_w / 2^m)
```

Then:

```text
offset_w + block_idx + 2^m q
=
offset_low + block_idx + 2^m(offset_high + q)
```

Now split `offset_low + block_idx` into a low part and a carry:

```text
low_idx = (offset_low + block_idx) mod 2^m
carry   = floor((offset_low + block_idx) / 2^m)
```

Since both `offset_low` and `block_idx` are less than `2^m`, the carry can only
be:

```text
carry ∈ {0, 1}
```

Therefore:

```text
global_index
=
low_idx + 2^m · (offset_high + q + carry)
```

This identity is the reason the implementation tracks two carry cases.

## 6. Equality Polynomial Factorization

The multilinear equality polynomial factors over bits.

If an index has low bits `low_idx` and high bits `high_idx`, then:

```text
eq_x(index) = eq_low(low_idx) · eq_high(high_idx)
```

In our case:

```text
high_idx = offset_high + q + carry
```

So:

```text
eq_x(offset_w + block_idx + 2^m q)
=
eq_low((offset_low + block_idx) mod 2^m)
·
eq_high(offset_high + q + carry)
```

Substitute that into the naive sum:

```text
w_sep =
Σ_{q, block}
  value[q, block]
  · eq_low(low_idx)
  · eq_high(offset_high + q + carry)
```

Now group the terms by `q` and by `carry`:

```text
w_sep =
Σ_q [
    (
      Σ_{block with carry 0}
        value[q, block] · eq_low(low_idx)
    )
    · eq_high(offset_high + q)

  +

    (
      Σ_{block with carry 1}
        value[q, block] · eq_low(low_idx)
    )
    · eq_high(offset_high + q + 1)
]
```

The current implementation computes exactly this grouped version.

## 7. First Optimization: Precompute Low-Bit Equality Values

The code first builds:

```rust
let block_low_eq = EqPolynomial::evals(&x_challenges[..block_bits]);
```

Conceptually:

```text
block_low_eq[u] = eq_low(u)
```

for every low-bit index `u ∈ [0, 2^m)`.

This lets the verifier reuse the low-bit equality values for every digit and
claim.

## 8. Second Optimization: Summarize Each Block Vector Into Two Carry Cases

The helper:

```rust
summarize_pow2_block_carries(&block_low_eq, block_offset_low, values)
```

computes:

```text
out[0] = Σ_{block with carry 0} values[block] · eq_low(low_idx)
out[1] = Σ_{block with carry 1} values[block] · eq_low(low_idx)
```

where:

```text
sum     = offset_low + block
carry   = floor(sum / 2^m)
low_idx = sum mod 2^m
```

This is the code:

```rust
for (u, &value) in values.iter().enumerate() {
    let sum = offset_low + u;
    let carry = sum >> inner_bits;
    let low_idx = sum & inner_mask;
    out[carry] += value * eq_low[low_idx];
}
```

The function is used in two ways for `w_sep`.

First, for each opening point:

```rust
summarize_pow2_block_carries(
    &block_low_eq,
    block_offset_low,
    &opening_point.b,
)
```

This produces:

```text
public_low0 = Σ_{block with carry 0} b[block] · eq_low(low_idx)
public_low1 = Σ_{block with carry 1} b[block] · eq_low(low_idx)
```

Second, for each claim's challenge vector:

```rust
summarize_pow2_block_carries(
    &block_low_eq,
    block_offset_low,
    &c_alphas[start..start + num_blocks],
)
```

This produces:

```text
challenge_low0 =
  Σ_{block with carry 0}
    c_alpha[claim, block] · eq_low(low_idx)

challenge_low1 =
  Σ_{block with carry 1}
    c_alpha[claim, block] · eq_low(low_idx)
```

## 9. Building `w_carry_terms`

After summarizing the low block dimension, the code loops over `dig` and
`claim_idx`.

For each pair:

```text
q = dig · num_claims + claim_idx
```

The public scale is:

```text
public_scale =
  public_weights[point_idx] · gamma[claim_idx] · g_open[dig]
```

The consistency scale is:

```text
challenge_scale =
  consistency_weight · g_open[dig]
```

The code then writes:

```rust
w_carry_terms[q][0] += public_scale * public_low0;
w_carry_terms[q][1] += public_scale * public_low1;

w_carry_terms[q][0] += challenge_scale * challenge_low0;
w_carry_terms[q][1] += challenge_scale * challenge_low1;
```

So after this loop:

```text
w_carry_terms[q][0]
=
Σ_{block with carry 0}
  g_open[dig]
  · (
      public_weight(point_idx) · gamma[claim_idx] · b_point[block]
    + consistency_weight · c_alpha[claim_idx, block]
    )
  · eq_low(low_idx)
```

and:

```text
w_carry_terms[q][1]
=
Σ_{block with carry 1}
  g_open[dig]
  · (
      public_weight(point_idx) · gamma[claim_idx] · b_point[block]
    + consistency_weight · c_alpha[claim_idx, block]
    )
  · eq_low(low_idx)
```

These are exactly the inner grouped sums from the algebra above.

## 10. Final High-Bit Evaluation

Now the low-bit block dimension has been consumed. The only remaining dimension
is the high index `q`.

The selected call:

```rust
eval_offset_eq_peeled_carry_terms(
    x_challenges,
    offset_w,
    block_bits,
    &w_carry_terms,
)
```

computes:

```text
Σ_q
  w_carry_terms[q][0] · eq_high(offset_high + q)
+
  w_carry_terms[q][1] · eq_high(offset_high + q + 1)
```

The `+1` is the carry from the low `block_idx` addition.

This matches exactly:

```text
Σ_q [
    (
      Σ_{block with carry 0}
        value[q, block] · eq_low(low_idx)
    )
    · eq_high(offset_high + q)

  +

    (
      Σ_{block with carry 1}
        value[q, block] · eq_low(low_idx)
    )
    · eq_high(offset_high + q + 1)
]
```

which was just the naive formula after factoring the equality polynomial.

## 11. Why This Is Correct

The correctness argument is only a change in summation order.

Start with the naive formula:

```text
Σ_{q, block}
  value[q, block]
  · eq_x(offset_w + block + 2^m q)
```

Use the index identity:

```text
offset_w + block + 2^m q
=
low_idx + 2^m(offset_high + q + carry)
```

Then factor the equality polynomial:

```text
eq_x(offset_w + block + 2^m q)
=
eq_low(low_idx)
·
eq_high(offset_high + q + carry)
```

So the naive formula becomes:

```text
Σ_{q, block}
  value[q, block]
  · eq_low(low_idx)
  · eq_high(offset_high + q + carry)
```

Then group terms by `q` and `carry`:

```text
Σ_q
  grouped_low_sum[q, 0] · eq_high(offset_high + q)
+
Σ_q
  grouped_low_sum[q, 1] · eq_high(offset_high + q + 1)
```

The implementation names these grouped low sums:

```text
grouped_low_sum[q, 0] = w_carry_terms[q][0]
grouped_low_sum[q, 1] = w_carry_terms[q][1]
```

Therefore the optimized code computes the same value as the naive triple loop.

## 12. Intuition

The verifier needs to evaluate a virtual vector, but the vector is too structured
to treat as arbitrary.

The naive computation asks:

```text
For every block, what is its full equality weight?
```

The optimized computation asks:

```text
First, consume the low block bits.
Did the offset addition carry into the high bits?
Then apply the high equality weight.
```

That is why `w_carry_terms[q]` has two entries. They are not two different
protocol terms. They are the two possible arithmetic carry cases created by
placing the `\hat w` segment at `offset_w` inside the larger virtual `M` table.

## 13. Motivation For Splitting The Equality Polynomial

The question is: why not just compute the large equality table once?

That naive strategy would be:

```text
eq_table = EqPolynomial::evals(x_challenges)

w_sep = 0
for every local column in the \hat w segment:
  global_index = offset_w + local_column
  w_sep += value[local_column] · eq_table[global_index]
```

This is simple and correct. The problem is cost.

If `x_challenges` has `n` variables, then the full equality table has size:

```text
2^n
```

For the stage-2 `M` table, `2^n` is the next power-of-two size of the full
virtual `M` row-vector. Even after ring switching has reduced the verifier's
problem size, this table can still be large. In the protocol-level intuition,
this is often the square-root-sized verifier object relative to the original
large statement. That is already much smaller than the original witness, but it
is still too expensive to materialize eagerly whenever we can exploit structure.

So the issue is not that a big equality table is impossible to compute. It is
that computing all of it costs linear time in the full virtual table size:

```text
cost of full eq table = O(2^n)
```

But `w_sep` does not need an arbitrary lookup pattern. Its indices have product
structure:

```text
local_index = block + 2^m · q
```

That means the equality value factors:

```text
eq_x(offset_w + block + 2^m q)
=
eq_low(low_idx) · eq_high(offset_high + q + carry)
```

Instead of building one table of size:

```text
2^n = 2^m · 2^{n-m}
```

the code can work with:

```text
low table size  = 2^m = num_blocks
high evaluations only for the q values that actually occur
```

For `w_sep`, the number of `q` values is:

```text
depth_open · num_claims
```

So the optimized computation is closer to:

```text
cost ≈ O(num_blocks)
     + O(depth_open · num_claims · high_eq_cost)
     + cost of combining the summaries
```

instead of:

```text
O(full M-table width)
```

The second important benefit is reuse. The low block summaries for
`opening_point.b` and for `c_alpha` can be reused across the digit dimension.
The code first consumes the `block` loop using `eq_low`, stores only two carry
cases, and only then applies the high equality weights.

So the motivation is not merely:

```text
small eq table is faster than big eq table
```

It is more specifically:

```text
the \hat w segment is block-structured,
so we can sum over the block dimension before touching the high dimension.
```

There is a small cost. The equality value is now represented as:

```text
eq_low(...) · eq_high(...)
```

instead of one precomputed `eq_full(...)` lookup. That introduces an extra field
multiplication at the point where the low summary is combined with the high
equality value, plus the carry bookkeeping.

But this cost is small compared with avoiding construction of the full
`eq_table` over the whole virtual `M` vector. In short:

```text
full table approach:
  easy, but pays for every position in M

split approach:
  slightly more algebra/bookkeeping,
  but pays mostly for the low block table and the actually-used high positions
```

Your summary is therefore basically right, with one nuance: the saving is not
just from replacing one equality table by two smaller tables. In this code, the
high side is not necessarily fully materialized as a table. The main saving is
that the block sum is performed first, so the verifier never needs the equality
weight for every individual `(q, block)` pair.

## 14. Why Not Just Use Two Smaller Equality Tables?

It is true that equality polynomials factor:

```text
eq(i, r) = eq_lo(i_lo, r_lo) · eq_hi(i_hi, r_hi)
```

So one possible generic strategy is:

```text
build eq_lo table
build eq_hi table

for every needed index i:
  split i into (i_lo, i_hi)
  eq_i = eq_lo[i_lo] · eq_hi[i_hi]
```

This avoids building the full table. If the full table has size:

```text
2^n
```

and we split after `a` bits, then the two smaller tables have total size:

```text
2^a + 2^{n-a}
```

This can be much smaller than `2^n`. For example, a balanced split gives about:

```text
2 · 2^{n/2}
```

instead of:

```text
2^n
```

So yes, this is a real optimization.

But `w_sep` is doing something more specific than computing equality values
one-by-one.

The generic two-table approach would still do:

```text
for q:
  for block:
    eq_i = eq_lo[low_idx] · eq_hi[high_idx]
    w_sep += value[q, block] · eq_i
```

That is better than building the full equality table, but it still treats every
`(q, block)` pair separately.

The current implementation instead uses the structure of `value[q, block]`.
Recall:

```text
value[q, block]
=
g_open[dig]
· (
    public_weight · gamma · b[block]
  + consistency_weight · c_alpha[claim, block]
  )
```

For a fixed `q`, the high equality term is the same for all blocks with the same
carry. Therefore we can do this:

```text
first sum over block using only eq_lo
then multiply the whole block-sum by eq_hi
```

That changes:

```text
Σ_block value[q, block] · eq_lo[low_idx] · eq_hi[high_idx]
```

into:

```text
(
  Σ_block value[q, block] · eq_lo[low_idx]
)
· eq_hi[high_idx]
```

up to the two carry cases.

This is the key difference.

The generic two-table approach uses the factorization like this:

```text
use eq_lo and eq_hi to reconstruct each full eq value
```

The `w_sep` implementation uses the factorization like this:

```text
move eq_hi outside the block sum
```

That is why the code builds `w_carry_terms`. Each entry is already a partially
summed value:

```text
w_carry_terms[q][carry]
=
Σ_block value[q, block] · eq_lo[low_idx]
```

Then the high equality is applied once per `(q, carry)`:

```text
w_carry_terms[q][0] · eq_hi(offset_high + q)
+
w_carry_terms[q][1] · eq_hi(offset_high + q + 1)
```

So the optimized computation avoids needing:

```text
eq_lo[low_idx] · eq_hi[high_idx]
```

for every individual `(q, block)` pair.

It instead pays for:

```text
one low-bit weighted block summary
one high-bit multiplication per q and carry case
```

This is also why the split point is not arbitrary. The split is chosen at:

```text
block_bits = log2(num_blocks)
```

because the column layout is:

```text
local_index = block + 2^block_bits · q
```

That split isolates exactly the `block` dimension. If we split at some unrelated
cutoff, the low bits would mix part of `block` with part of `q`, or the high bits
would still contain part of `block`. Then we could still reconstruct
`eq(i, r)` from two tables, but we would not be able to cleanly sum over the
whole block dimension first.

In short:

```text
two-table generic method:
  useful because eq factors
  still computes each needed eq weight separately

w_sep method:
  uses the same factorization
  chooses the split to match the block layout
  sums over blocks before applying the high equality term
```

That is the real reason for this implementation.

## 15. Why The Carry Cannot Be Avoided

It may seem that we should be able to split exactly as:

```text
local_index = block + 2^m · q
```

and then write:

```text
eq_x(offset_w + local_index)
?=
eq_low(block) · eq_high(offset_high + q)
```

If this were true, we would not need carry terms.

The problem is that the equality polynomial is evaluated on the bits of the
global index:

```text
global_index = offset_w + local_index
```

It is not evaluated on the bits of the local index alone.

So the split has to be a split of:

```text
offset_w + block + 2^m · q
```

not just a split of:

```text
block + 2^m · q
```

If `offset_w` is a multiple of `2^m`, then there is no problem. Suppose:

```text
offset_w = 2^m · offset_high
```

Then:

```text
global_index
=
block + 2^m(offset_high + q)
```

and the clean split works:

```text
eq_x(global_index)
=
eq_low(block) · eq_high(offset_high + q)
```

In that special aligned case, no carry is needed.

But in general `offset_w` is not aligned. It may have nonzero low bits:

```text
offset_w = offset_low + 2^m · offset_high
```

Then:

```text
global_index
=
offset_low + block + 2^m(offset_high + q)
```

Now the low part is not simply `block`. It is:

```text
offset_low + block
```

If this sum exceeds `2^m - 1`, it wraps around in the low bits and adds `1` to
the high bits. That `1` is the carry.

### Tiny Example

Take:

```text
m = 2
2^m = 4
offset_w = 2
```

So:

```text
offset_low = 2
offset_high = 0
```

For `q = 0`, the four block values give:

```text
block = 0:
  global_index = 2 + 0 + 4·0 = 2
  binary low bits = 2
  high bits = 0
  carry = 0

block = 1:
  global_index = 2 + 1 + 4·0 = 3
  binary low bits = 3
  high bits = 0
  carry = 0

block = 2:
  global_index = 2 + 2 + 4·0 = 4
  binary low bits = 0
  high bits = 1
  carry = 1

block = 3:
  global_index = 2 + 3 + 4·0 = 5
  binary low bits = 1
  high bits = 1
  carry = 1
```

So for the same `q = 0`, some blocks need:

```text
eq_high(0)
```

and other blocks need:

```text
eq_high(1)
```

That means there is no single high equality value that applies to all blocks for
this `q`.

The best we can do is split the blocks into two groups:

```text
carry = 0 group
carry = 1 group
```

That is exactly why the implementation stores:

```text
w_carry_terms[q][0]
w_carry_terms[q][1]
```

### What Would Go Wrong Without Carry?

If we ignored carry, we would compute something like:

```text
Σ_block value[q, block] · eq_low(offset_low + block mod 2^m)
```

and then multiply the whole result by:

```text
eq_high(offset_high + q)
```

But in the example above, blocks `2` and `3` actually require:

```text
eq_high(offset_high + q + 1)
```

not:

```text
eq_high(offset_high + q)
```

So ignoring carry would assign the wrong equality weight to every block whose
low-bit addition overflows.

### Can We Choose A Different Split To Avoid Carry?

Sometimes, but not in a way that preserves the useful block summation.

If we choose a split where `offset_w` is aligned, then the carry across that
split disappears. But that split may not match:

```text
local_index = block + 2^block_bits · q
```

The whole optimization relies on the low side being exactly the `block`
dimension, because then we can sum over all `block_idx` values first.

If we split at a different cutoff:

- the low side may contain only part of `block_idx`; or
- the low side may contain all of `block_idx` plus part of `q`; or
- the high side may still depend on `block_idx`.

In those cases, we can still reconstruct individual equality values from two
tables, but we lose the clean block-level grouping that makes `w_sep` efficient.

So the carry is not an artifact of the implementation. It is the exact price of
using a split that matches the block layout while evaluating equality at the
global offset `offset_w`.