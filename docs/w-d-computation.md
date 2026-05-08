# Computing `w_d`

This note focuses only on the `w_d` term in the verifier-side ring-switch
replay:

```rust
let w_d = {
    let _span = tracing::info_span!("m_eval_w_d").entered();
    eval_d_matrix_w_residual_direct(
        x_challenges,
        offset_w,
        num_blocks,
        num_claims,
        depth_open,
        d_weights,
        d_view,
        &alpha_pows,
    )
};
```

The goal is to explain the naive computation first, then explain how the current
implementation computes the same value using the same equality-polynomial split
as `w_sep`.

## 1. What `w_d` Represents

In the paper matrix, the `\hat w` columns appear in several rows:

```text
D · \hat w                         = v
b^T G_{2^r} · \hat w               = y
(c^T ⊗ G_1) · \hat w - ... \hat z  = 0
```

The verifier samples `tau1`, takes a random linear combination of the rows, and
then evaluates the resulting virtual row-vector `M` at a multilinear point.

The `\hat w` part of that virtual row-vector is split in the implementation as:

```text
w_part = w_sep + w_d
```

Here:

- `w_sep` is the structured separable contribution from the public/opening-point
  row and the challenge/consistency row.
- `w_d` is the contribution from the `D · \hat w = v` row.

So this note ignores the `b^T G_{2^r}` and `(c^T ⊗ G_1)` pieces, and focuses
only on:

```text
D · \hat w = v
```

In the implementation, the verifier does not materialize the full row-vector
`M`. Instead, it evaluates the relevant entries of the `D` matrix at the
ring-switch challenge `alpha`, weights the `D` rows by `tau1`, and contributes
that directly to the multilinear evaluation.

## 2. Where `w_d` Appears In The Materialized Formula

The prover-side materialized reference constructs a `w_segment`. For each local
column `x` inside the `\hat w` segment:

```rust
let dig = x / total_blocks;
let blk = x % total_blocks;
let claim_idx = blk / num_blocks;
let block_idx = blk % num_blocks;
let d_phys_col = blk * depth_open + dig;
```

The separable part is:

```rust
(public_weights[point_idx] * gamma[claim_idx] * opening_point.b[block_idx]
    + consistency_weight * c_alphas[blk])
    * g1_open[dig]
```

Then the `D` contribution is added:

```rust
for (di, eq_i) in eq_tau1[d_start..(d_start + n_d)].iter().enumerate() {
    if !eq_i.is_zero() {
        acc += *eq_i * eval_ring_at_pows(&d_view.row(di)[d_phys_col], alpha_pows);
    }
}
```

That final loop is the materialized version of `w_d`.

So for one local `\hat w` column, the `D` contribution is:

```text
D_value[dig, claim_idx, block_idx]
=
Σ_di
  d_weights[di]
  · eval_alpha(D_row_di[d_phys_col])
```

where:

```text
d_weights[di] = eq_tau1[d_start + di]
```

and:

```text
d_phys_col = (claim_idx · num_blocks + block_idx) · depth_open + dig
```

`eval_alpha(...)` means evaluating the cyclotomic-ring element at the sampled
ring-switch point `alpha`.

## 3. Shape Of The `\hat w` Segment

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

The local index inside the `\hat w` segment is:

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
segment starts at `offset_w`, so the global index is:

```text
global_index = offset_w + local_index
```

Therefore:

```text
global_index = offset_w + block_idx + B · q
```

This is the same index shape used by `w_sep`.

## 4. Objects In The `w_d` Formula

Before writing the naive loop, define the objects that appear in one term.

`d_view` is a view of the verifier's shared setup matrix for the `D` rows. A
single row is:

```text
d_view.row(di)
```

and one column is:

```text
d_view.row(di)[d_phys_col]
```

This entry is a cyclotomic-ring element.

`alpha_pows` contains powers of the ring-switch challenge:

```text
alpha_pows = [1, alpha, alpha^2, ..., alpha^{D-1}]
```

The helper:

```text
eval_ring_at_pows(ring_element, alpha_pows)
```

evaluates a ring element at `alpha`. If:

```text
ring_element = a_0 + a_1 X + ... + a_{D-1} X^{D-1}
```

then:

```text
eval_ring_at_pows(ring_element, alpha_pows)
=
a_0 + a_1 alpha + ... + a_{D-1} alpha^{D-1}
```

`d_weights` are the `tau1` equality weights for the `D` rows. If there are
`n_d` rows in the `D` matrix, then:

```text
d_weights.len() = n_d
```

and:

```text
d_weights[di]
```

is the coefficient applied to row `di` after the verifier has randomly combined
the paper matrix rows using `tau1`.

## 5. The Naive Computation

The fully naive approach is:

```text
w_d = 0

for dig in 0..depth_open:
  for claim_idx in 0..num_claims:
    for block_idx in 0..num_blocks:

      blk =
        claim_idx · num_blocks
        + block_idx

      d_phys_col =
        blk · depth_open
        + dig

      value = 0

      for di in 0..n_d:
        value +=
          d_weights[di]
          · eval_alpha(D_row_di[d_phys_col])

      local_index =
          dig · (num_claims · num_blocks)
        + claim_idx · num_blocks
        + block_idx

      global_index = offset_w + local_index

      w_d += value · eq(x_challenges, global_index)
```

In summation form:

```text
w_d =
Σ_{dig, claim, block}
  (
    Σ_di
      d_weights[di]
      · eval_alpha(D_di[(claim · B + block) · L + dig])
  )
  · eq_x(offset_w + dig·C·B + claim·B + block)
```

Using:

```text
q = dig · C + claim
```

this becomes:

```text
w_d =
Σ_{q, block}
  D_value[q, block]
  · eq_x(offset_w + block + B·q)
```

where `D_value[q, block]` is the `tau1`-weighted, `alpha`-evaluated `D` matrix
entry for that `q` and `block`.

## 6. Why This Has The Same Equality Split As `w_sep`

The index shape is:

```text
offset_w + block + B·q
```

and:

```text
B = num_blocks = 2^block_bits
```

Let:

```text
m = block_bits
B = 2^m
```

Split:

```text
offset_w = offset_low + 2^m · offset_high
```

Then:

```text
offset_w + block + 2^m q
=
offset_low + block + 2^m(offset_high + q)
```

Now define:

```text
low_idx = (offset_low + block) mod 2^m
carry   = floor((offset_low + block) / 2^m)
```

Because both `offset_low` and `block` are less than `2^m`, the carry is either
`0` or `1`.

Therefore:

```text
offset_w + block + 2^m q
=
low_idx + 2^m(offset_high + q + carry)
```

The equality polynomial factors over low and high bits:

```text
eq_x(offset_w + block + 2^m q)
=
eq_low(low_idx)
·
eq_high(offset_high + q + carry)
```

So the naive formula:

```text
Σ_{q, block}
  D_value[q, block]
  · eq_x(offset_w + block + 2^m q)
```

becomes:

```text
Σ_{q, block}
  D_value[q, block]
  · eq_low(low_idx)
  · eq_high(offset_high + q + carry)
```

Now group by `q` and by `carry`:

```text
Σ_q [
    (
      Σ_{block with carry 0}
        D_value[q, block] · eq_low(low_idx)
    )
    · eq_high(offset_high + q)

  +

    (
      Σ_{block with carry 1}
        D_value[q, block] · eq_low(low_idx)
    )
    · eq_high(offset_high + q + 1)
]
```

The implementation computes exactly this grouped expression.

## 7. The Strided Access Pattern

The `D` matrix is not stored in the same order as the local `\hat w` segment.
The local `\hat w` segment is digit-major:

```text
local_index = dig · (C · B) + claim · B + block
```

But the physical `D` matrix column is block-major with the digit inside each
block:

```text
d_phys_col = (claim · B + block) · L + dig
```

Expanding:

```text
d_phys_col
=
claim · B · L
+ block · L
+ dig
```

For a fixed `(dig, claim)`, define:

```text
lane_offset = claim · B · L + dig
```

Then:

```text
d_phys_col = block · L + lane_offset
```

This is why the helper is called `summarize_strided_pow2_block_carries`. As
`block` changes, the matrix column advances by:

```text
block_stride = depth_open
```

not by `1`.

## 8. Current Implementation, Step By Step

The function is:

```rust
fn eval_d_matrix_w_residual_direct<F: FieldCore, const D: usize>(
    x_challenges: &[F],
    offset_w: usize,
    num_blocks: usize,
    num_claims: usize,
    depth_open: usize,
    d_weights: &[F],
    d_view: RingMatrixView<'_, F, D>,
    alpha_pows: &[F],
) -> F
```

First it computes:

```rust
let block_bits = num_blocks.trailing_zeros() as usize;
let block_low_eq = EqPolynomial::evals(&x_challenges[..block_bits]);
let block_offset_low = offset_w & (num_blocks - 1);
let per_claim_d_width = num_blocks * depth_open;
```

Conceptually:

```text
block_low_eq[u] = eq_low(u)
```

and:

```text
block_offset_low = offset_w mod num_blocks
```

The value:

```text
per_claim_d_width = num_blocks · depth_open
```

is the number of `D` matrix columns occupied by one claim.

Then it builds one carry summary per:

```text
q = dig · num_claims + claim_idx
```

The code iterates over `q`:

```rust
let carry_terms: Vec<[F; 2]> = cfg_into_iter!(0..(num_claims * depth_open))
    .map(|q| {
        let claim_idx = q % num_claims;
        let dig = q / num_claims;
        let lane_offset = claim_idx * per_claim_d_width + dig;
        ...
    })
    .collect();
```

The decoding is:

```text
claim_idx = q mod num_claims
dig       = floor(q / num_claims)
```

This is the inverse of:

```text
q = dig · num_claims + claim_idx
```

Then:

```text
lane_offset = claim_idx · num_blocks · depth_open + dig
```

so for each block:

```text
d_phys_col = block_idx · depth_open + lane_offset
```

which is equal to:

```text
(claim_idx · num_blocks + block_idx) · depth_open + dig
```

That is exactly the materialized formula's `d_phys_col`.

## 9. Summarizing One `D` Row Over Blocks

For each `q`, the implementation loops over the `D` rows:

```rust
for (di, &d_weight) in d_weights.iter().enumerate() {
    if d_weight.is_zero() {
        continue;
    }
    let row = d_view.row(di);
    let [block_low0, block_low1] = summarize_strided_pow2_block_carries(
        &block_low_eq,
        block_offset_low,
        row,
        alpha_pows,
        num_blocks,
        depth_open,
        lane_offset,
    );
    out[0] += d_weight * block_low0;
    out[1] += d_weight * block_low1;
}
```

The helper:

```rust
summarize_strided_pow2_block_carries(...)
```

computes, for one `D` row and one fixed `(dig, claim)`:

```text
block_low0 =
Σ_{block with carry 0}
  eval_alpha(D_row[block · depth_open + lane_offset])
  · eq_low(low_idx)
```

and:

```text
block_low1 =
Σ_{block with carry 1}
  eval_alpha(D_row[block · depth_open + lane_offset])
  · eq_low(low_idx)
```

The implementation is:

```rust
for block_idx in 0..block_count {
    let sum = offset_low + block_idx;
    let carry = sum >> inner_bits;
    let low_idx = sum & inner_mask;
    let col = block_idx * block_stride + lane_offset;
    let value = eval_ring_at_pows(&row[col], alpha_pows);
    out[carry] += value * eq_low[low_idx];
}
```

For `w_d`, the arguments are:

```text
block_count  = num_blocks
block_stride = depth_open
lane_offset  = claim_idx · num_blocks · depth_open + dig
```

So:

```text
col = block_idx · depth_open + claim_idx · num_blocks · depth_open + dig
```

which simplifies to:

```text
col = (claim_idx · num_blocks + block_idx) · depth_open + dig
```

This is exactly the same `d_phys_col` used by the materialized reference.

## 10. Combining The `D` Rows

After summarizing one row, the function multiplies it by that row's `tau1`
weight:

```rust
out[0] += d_weight * block_low0;
out[1] += d_weight * block_low1;
```

So after the loop over `di`, for this fixed `q`:

```text
out[0]
=
Σ_di d_weights[di]
  ·
  Σ_{block with carry 0}
    eval_alpha(D_di[d_phys_col])
    · eq_low(low_idx)
```

and:

```text
out[1]
=
Σ_di d_weights[di]
  ·
  Σ_{block with carry 1}
    eval_alpha(D_di[d_phys_col])
    · eq_low(low_idx)
```

Equivalently:

```text
out[carry]
=
Σ_{block with carry}
  (
    Σ_di
      d_weights[di] · eval_alpha(D_di[d_phys_col])
  )
  · eq_low(low_idx)
```

The inner parentheses are exactly:

```text
D_value[q, block]
```

Therefore:

```text
carry_terms[q][carry]
=
Σ_{block with carry}
  D_value[q, block] · eq_low(low_idx)
```

This is the grouped low-bit sum needed by the equality split.

## 11. Final High-Bit Evaluation

After building all `carry_terms`, the function calls:

```rust
eval_offset_eq_peeled_carry_terms(
    x_challenges,
    offset_w,
    block_bits,
    &carry_terms,
)
```

This computes:

```text
Σ_q
  carry_terms[q][0] · eq_high(offset_high + q)
+
  carry_terms[q][1] · eq_high(offset_high + q + 1)
```

The `+1` is the carry from:

```text
offset_low + block_idx
```

overflowing the low `block_bits`.

Substituting the definition of `carry_terms` gives:

```text
Σ_q [
    (
      Σ_{block with carry 0}
        D_value[q, block] · eq_low(low_idx)
    )
    · eq_high(offset_high + q)

  +

    (
      Σ_{block with carry 1}
        D_value[q, block] · eq_low(low_idx)
    )
    · eq_high(offset_high + q + 1)
]
```

This is exactly the naive formula after factoring:

```text
eq_x(offset_w + block + 2^m q)
=
eq_low(low_idx)
·
eq_high(offset_high + q + carry)
```

## 12. Why This Is Correct

Start with the naive formula:

```text
w_d =
Σ_{q, block}
  D_value[q, block]
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

So:

```text
w_d =
Σ_{q, block}
  D_value[q, block]
  · eq_low(low_idx)
  · eq_high(offset_high + q + carry)
```

Now group by `q` and `carry`:

```text
w_d =
Σ_q
  grouped_low_sum[q, 0] · eq_high(offset_high + q)
+
Σ_q
  grouped_low_sum[q, 1] · eq_high(offset_high + q + 1)
```

where:

```text
grouped_low_sum[q, carry]
=
Σ_{block with carry}
  D_value[q, block] · eq_low(low_idx)
```

The implementation names these grouped sums:

```text
carry_terms[q][0]
carry_terms[q][1]
```

Therefore the optimized/direct code computes the same value as the naive loop.

## 13. Why This Is Called `residual_direct`

The `w_sep` term is separable: it is built from small vectors like
`opening_point.b`, `c_alphas`, and the gadget vector. The verifier can summarize
those vectors very cheaply.

The `w_d` term is different. It comes from the actual setup matrix `D`. For each
block, digit, claim, and `D` row, the verifier needs:

```text
eval_alpha(D_di[d_phys_col])
```

Those values are not represented by a small separable public vector in the same
way as `b[block]` or `c_alpha[block]`.

So the implementation computes this residual contribution directly from the
matrix rows:

```text
read D matrix entry
evaluate it at alpha
multiply by the tau1 row weight
accumulate into the carry summary
```

It is still optimized in the important way: it does not materialize the full
`M` table, and it does not compute the full equality table. It only materializes
the low equality table for the block dimension, then evaluates high equality
values on demand after the block sums have been grouped.

## 14. Cost Shape

Let:

```text
B = num_blocks
C = num_claims
L = depth_open
n_d = number of D rows
```

The number of `q` values is:

```text
C · L
```

For each `q`, the function loops over:

```text
n_d rows
B blocks
```

So the main matrix-evaluation cost is:

```text
O(C · L · n_d · B)
```

This is the cost of reading and evaluating the relevant `D` matrix entries. The
equality-polynomial work is organized as:

```text
eq_low table size = B
high equality evaluations ≤ 2 · C · L
```

The high side is not materialized as a full table. It is evaluated only for the
needed:

```text
offset_high + q
offset_high + q + 1
```

indices.

## 15. Relationship To `w_sep`

Both `w_sep` and `w_d` compute contributions to the same `\hat w` segment:

```text
w_part = w_sep + w_d
```

They use the same global index:

```text
offset_w + block + 2^block_bits · q
```

and therefore the same equality split:

```text
eq_x(global_index)
=
eq_low(low_idx)
·
eq_high(offset_high + q + carry)
```

The difference is the value being weighted by this equality polynomial.

For `w_sep`:

```text
value[q, block]
=
g_open[dig]
· (
    public_weight · gamma · b[block]
  + consistency_weight · c_alpha[claim, block]
  )
```

For `w_d`:

```text
value[q, block]
=
Σ_di
  d_weights[di]
  · eval_alpha(D_di[(claim · num_blocks + block) · depth_open + dig])
```

So `w_sep` is the separable public/challenge part, while `w_d` is the direct
setup-matrix part from the `D · \hat w` row.
