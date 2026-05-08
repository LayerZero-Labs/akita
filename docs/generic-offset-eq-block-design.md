# Generic Offset-EQ Block Evaluation Design

This note sketches a more generic design for the repeated offset-equality logic
currently used by the verifier-side `M` evaluation.

The motivating examples are:

- `w_sep`
- `w_d`
- `t_sep`
- `t_b`

All of these have the same high-level shape:

```text
Σ_{q, block}
  value(q, block)
  · eq_x(offset + block + 2^m q)
```

where:

```text
2^m = num_blocks
```

The repeated logic is:

1. Build the low equality table.
2. Split block indices into no-carry and carry groups.
3. Accumulate low-bit summaries.
4. Multiply by the high equality value.
5. Sum the carry-0 and carry-1 contributions.

The only thing that differs between cases is how `value(q, block)` is computed.

## 1. The Common Mathematical Shape

Assume a segment is laid out as:

```text
local_index = block + 2^m q
```

The segment starts inside the full virtual `M` table at:

```text
offset
```

so the global index is:

```text
global_index = offset + block + 2^m q
```

The target expression is:

```text
Σ_{q, block}
  value(q, block)
  · eq_x(offset + block + 2^m q)
```

Split:

```text
offset = offset_low + 2^m · offset_high
```

For each `block`, define:

```text
sum     = offset_low + block
low_idx = sum mod 2^m
carry   = floor(sum / 2^m)
```

Since:

```text
0 <= offset_low < 2^m
0 <= block      < 2^m
```

the carry is always:

```text
carry ∈ {0, 1}
```

Then:

```text
offset + block + 2^m q
=
low_idx + 2^m(offset_high + q + carry)
```

So:

```text
eq_x(offset + block + 2^m q)
=
eq_low(low_idx)
·
eq_high(offset_high + q + carry)
```

Therefore the target expression becomes:

```text
Σ_q [
    (
      Σ_{block with carry 0}
        value(q, block) · eq_low(low_idx)
    )
    · eq_high(offset_high + q)

  +

    (
      Σ_{block with carry 1}
        value(q, block) · eq_low(low_idx)
    )
    · eq_high(offset_high + q + 1)
]
```

This is the generic algorithm.

## 2. What Is Repeated Today

Today, several functions manually repeat parts of this pattern.

For `w_sep`, the code first builds low summaries for:

```text
opening_point.b
c_alphas
```

then combines them into:

```text
w_carry_terms[q][carry]
```

and finally calls:

```text
eval_offset_eq_peeled_carry_terms(...)
```

For `w_d`, the code builds:

```text
carry_terms[q][carry]
```

by looping over `D` rows and strided block positions, then calls:

```text
eval_offset_eq_peeled_carry_terms(...)
```

For `t_b`, the logic is extremely similar to `w_d`, except the matrix is `B`
instead of `D`, and the lane decoding uses:

```text
a_idx
digit_idx
claim_idx_within_group
```

For `t_sep`, the logic is similar to `w_sep`, except the separable factors come
from:

```text
a_weights
g_open
c_alphas
```

The common skeleton is:

```text
for q:
  out = [0, 0]
  for block:
    sum = offset_low + block
    carry = sum >> block_bits
    low_idx = sum & block_mask
    out[carry] += value(q, block) · eq_low[low_idx]
  carry_terms[q] = out

answer =
  Σ_q carry_terms[q][0] · eq_high(offset_high + q)
    + carry_terms[q][1] · eq_high(offset_high + q + 1)
```

## 3. Proposed Core Abstraction

The core reusable primitive should represent the common operation:

```text
Σ_{q, block}
  value(q, block)
  · eq_x(offset + block + 2^m q)
```

where `value(q, block)` is provided by the caller.

A direct generic API could look like:

```rust
pub fn eval_offset_eq_block_sum<F, V>(
    x_challenges: &[F],
    offset: usize,
    block_count: usize,
    q_count: usize,
    mut value_at: V,
) -> F
where
    F: FieldCore,
    V: FnMut(usize, usize) -> F,
{
    // q, block -> value(q, block)
}
```

with:

```text
value_at(q, block) = value(q, block)
```

This is the simplest abstraction. It centralizes:

- low equality table generation
- offset-low handling
- carry split
- high equality application
- zero handling, if wanted

The generic implementation would be:

```text
block_bits = log2(block_count)
eq_low = EqPolynomial::evals(x_challenges[..block_bits])
offset_low = offset mod block_count
offset_high = offset >> block_bits

for q in 0..q_count:
  carry0 = 0
  carry1 = 0

  for block in 0..block_count:
    sum = offset_low + block
    carry = sum >> block_bits
    low_idx = sum & (block_count - 1)
    contribution = value_at(q, block) · eq_low[low_idx]

    if carry == 0:
      carry0 += contribution
    else:
      carry1 += contribution

  result += carry0 · eq_high(offset_high + q)
  result += carry1 · eq_high(offset_high + q + 1)
```

This is correct for every case matching:

```text
local_index = block + 2^m q
```

## 4. Why This API Is Not Always Enough

The simple `value_at(q, block)` API is maximally generic but may be too slow for
some existing cases.

For example, `w_sep` currently pre-summarizes:

```text
opening_point.b
c_alphas
```

and reuses those summaries across the digit dimension. If `value_at(q, block)`
is called for every `(q, block)`, then the caller may lose that reuse unless the
closure internally caches or the API exposes a more structured accumulation
mode.

Similarly, `w_d` and `t_b` evaluate ring matrix entries:

```text
eval_ring_at_pows(&row[col], alpha_pows)
```

For these residual matrix cases, calling `value_at(q, block)` directly is
reasonable because the work is genuinely per row/block/column. But for separable
cases, pre-aggregation is useful.

So we probably want two layers:

1. A low-level generic carry/equality engine.
2. Specialized adapters for separable vectors and strided matrix rows.

## 5. Layer 1: Generic Carry Plan

First, separate the offset/carry bookkeeping from the value computation.

Introduce a reusable plan:

```rust
pub struct OffsetEqBlockPlan<F> {
    block_bits: usize,
    block_count: usize,
    offset_low: usize,
    offset_high: usize,
    eq_low: Vec<F>,
    eq_high: Option<Vec<F>>,
}
```

Construction:

```rust
impl<F: FieldCore> OffsetEqBlockPlan<F> {
    pub fn new(
        x_challenges: &[F],
        offset: usize,
        block_count: usize,
        high_mode: HighEqMode,
    ) -> Self
}
```

where:

```rust
pub enum HighEqMode {
    OnDemand,
    Materialize,
}
```

The plan validates:

```text
block_count is power of two
block_bits <= x_challenges.len()
```

and computes:

```text
eq_low
offset_low
offset_high
```

If `HighEqMode::Materialize` is selected, it also builds:

```text
eq_high table of size 2^(x_challenges.len() - block_bits)
```

Otherwise, it computes high equality values on demand.

The plan should expose:

```rust
impl<F: FieldCore> OffsetEqBlockPlan<F> {
    pub fn low_weight_and_carry(&self, block: usize) -> (usize, F, usize) {
        // returns (carry, eq_low[low_idx], low_idx)
    }

    pub fn high_weight(&self, q: usize, carry: usize) -> F {
        // eq_high(offset_high + q + carry)
    }

    pub fn finish(&self, carry_terms: &[[F; 2]]) -> F {
        // Σ_q carry_terms[q][0]·high(q,0) + carry_terms[q][1]·high(q,1)
    }
}
```

This makes the carry logic reusable without forcing every caller to use the same
value-generation strategy.

## 6. Layer 2A: Generic Dense Block Sum

For simple cases, we can provide the direct closure-based evaluator:

```rust
pub fn eval_offset_eq_block_sum<F, V>(
    x_challenges: &[F],
    offset: usize,
    block_count: usize,
    q_count: usize,
    high_mode: HighEqMode,
    mut value_at: V,
) -> F
where
    F: FieldCore,
    V: FnMut(usize, usize) -> F,
{
    let plan = OffsetEqBlockPlan::new(x_challenges, offset, block_count, high_mode);
    let mut carry_terms = vec![[F::zero(), F::zero()]; q_count];

    for q in 0..q_count {
        for block in 0..block_count {
            let (carry, low_weight, _) = plan.low_weight_and_carry(block);
            carry_terms[q][carry] += value_at(q, block) * low_weight;
        }
    }

    plan.finish(&carry_terms)
}
```

This is easy to use and good for tests, reference implementations, or cases
where `value_at(q, block)` is cheap.

## 7. Layer 2B: Reusable Low Summaries For Vectors

For `w_sep` and `t_sep`, we want to summarize vectors over the block dimension.

The existing helper:

```rust
summarize_pow2_block_carries(eq_low, offset_low, values)
```

can become a method:

```rust
impl<F: FieldCore> OffsetEqBlockPlan<F> {
    pub fn summarize_values(&self, values: &[F]) -> [F; 2] {
        // Σ_block values[block] · eq_low[low_idx], split by carry
    }
}
```

Then `w_sep` can say:

```text
public_summary[point] = plan.summarize_values(opening_point.b)
challenge_summary[claim] = plan.summarize_values(c_alphas_for_claim)
```

and build:

```text
w_carry_terms[q][carry]
```

from those summaries.

This preserves the current reuse.

## 8. Layer 2C: Reusable Strided Matrix Summaries

For `w_d` and `t_b`, we want to summarize a strided set of ring matrix columns.

The existing helper:

```rust
summarize_strided_pow2_block_carries(...)
```

can become a method or a generic function:

```rust
pub fn summarize_strided_ring_row<F, const D: usize>(
    plan: &OffsetEqBlockPlan<F>,
    row: &[CyclotomicRing<F, D>],
    alpha_pows: &[F],
    block_stride: usize,
    lane_offset: usize,
) -> [F; 2]
where
    F: FieldCore,
{
    // for block:
    //   col = block * block_stride + lane_offset
    //   value = eval_ring_at_pows(&row[col], alpha_pows)
    //   out[carry] += value * low_weight
}
```

This directly covers:

```text
w_d:
  block_stride = depth_open
  lane_offset = claim_idx · num_blocks · depth_open + dig

t_b:
  block_stride = n_a · depth_open
  lane_offset =
    claim_idx_within_group · t_cols_per_claim
    + a_idx · depth_open
    + digit_idx
```

The high-equality finishing step is still:

```rust
plan.finish(&carry_terms)
```

## 9. Pre-Generating `eq_high`

The proposed `HighEqMode` should be configurable because the best choice depends
on the shape.

### On-Demand Mode

Current behavior:

```text
eq_high(index) is computed when needed
```

This is good when the number of high indices needed is small:

```text
needed high evaluations ≤ 2 · q_count
```

For root `w_sep` in one-hot `nv=32`, this is around:

```text
2 · depth_open · num_claims = 128
```

That is much smaller than some full virtual tables.

### Materialized Mode

Alternative behavior:

```text
eq_high_table = EqPolynomial::evals(high_challenges)
```

Then:

```text
high_weight(q, carry) = eq_high_table[offset_high + q + carry]
```

This is good when:

```text
2 · q_count
```

is comparable to or larger than:

```text
2^(num_high_vars)
```

or when multiple components with the same `offset` and `block_bits` reuse the
same high table.

### Heuristic

A simple heuristic:

```text
if 2 · q_count >= high_table_size / 2:
  materialize eq_high
else:
  compute on demand
```

But this should be benchmarked. `eq_eval_at_index` is cheap for small
`num_high_vars`, while table materialization has allocation and memory costs.

The API should support both modes so profiling can choose.

## 10. Handling No-Overflow And Overflow Indices Explicitly

The user's proposed approach is:

1. Generate `eq_low`.
2. Generate or compute `eq_high`.
3. Find indices that do not overflow.
4. Find indices that do overflow.
5. Use `eq_high(q)` for no-overflow and `eq_high(q + 1)` for overflow.

This is exactly the right conceptual split.

The generic plan can expose this directly:

```rust
pub struct BlockCarryPartition<'a, F> {
    pub no_carry: &'a [(usize, usize, F)],
    pub carry: &'a [(usize, usize, F)],
}
```

where each entry could be:

```text
(block, low_idx, eq_low[low_idx])
```

However, storing these triples may not be necessary. Since:

```text
carry = floor((offset_low + block) / block_count)
```

the no-carry and carry ranges are contiguous:

```text
no carry:
  block < block_count - offset_low

carry:
  block >= block_count - offset_low
```

So the plan can expose ranges:

```rust
pub struct CarryRanges {
    pub no_carry: std::ops::Range<usize>,
    pub carry: std::ops::Range<usize>,
}
```

For example:

```rust
impl<F: FieldCore> OffsetEqBlockPlan<F> {
    pub fn carry_ranges(&self) -> CarryRanges {
        let split = self.block_count - self.offset_low;
        CarryRanges {
            no_carry: 0..split,
            carry: split..self.block_count,
        }
    }
}
```

Then callers can write:

```text
for block in no_carry:
  low_idx = offset_low + block
  use eq_high(offset_high + q)

for block in carry:
  low_idx = offset_low + block - block_count
  use eq_high(offset_high + q + 1)
```

This avoids computing `carry` inside the loop and makes the algorithm match the
conceptual explanation.

## 11. Proposed API Surface

A practical API could be:

```rust
pub enum HighEqMode {
    OnDemand,
    Materialize,
}

pub struct OffsetEqBlockPlan<F> {
    block_count: usize,
    block_bits: usize,
    offset_low: usize,
    offset_high: usize,
    eq_low: Vec<F>,
    high_challenges: Vec<F>,
    eq_high: Option<Vec<F>>,
}

impl<F: FieldCore> OffsetEqBlockPlan<F> {
    pub fn new(
        x_challenges: &[F],
        offset: usize,
        block_count: usize,
        high_mode: HighEqMode,
    ) -> Self;

    pub fn carry_ranges(&self) -> CarryRanges;

    pub fn low_weight(&self, block: usize) -> (usize, F);

    pub fn high_weight(&self, q: usize, carry: usize) -> F;

    pub fn finish(&self, carry_terms: &[[F; 2]]) -> F;

    pub fn summarize_values(&self, values: &[F]) -> [F; 2];
}

pub fn eval_offset_eq_block_sum<F, V>(
    plan: &OffsetEqBlockPlan<F>,
    q_count: usize,
    value_at: V,
) -> F
where
    F: FieldCore,
    V: FnMut(usize, usize) -> F;

pub fn summarize_strided_ring_row<F, const D: usize>(
    plan: &OffsetEqBlockPlan<F>,
    row: &[CyclotomicRing<F, D>],
    alpha_pows: &[F],
    block_stride: usize,
    lane_offset: usize,
) -> [F; 2]
where
    F: FieldCore;
```

This keeps the common offset-equality machinery in `akita-algebra`, while
ring-specific helpers may need to live where `CyclotomicRing` and
`eval_ring_at_pows` are available.

## 12. Refactoring Plan

Proceed in small steps.

1. Add `OffsetEqBlockPlan` in `akita-algebra::offset_eq`.

   It should initially reproduce the behavior of:

   ```text
   summarize_pow2_block_carries
   eval_offset_eq_peeled_carry_terms
   ```

   without changing call sites.

2. Add tests comparing:

   ```text
   plan.summarize_values(values)
   ```

   against:

   ```text
   summarize_pow2_block_carries(eq_low, offset_low, values)
   ```

   and:

   ```text
   plan.finish(carry_terms)
   ```

   against:

   ```text
   eval_offset_eq_peeled_carry_terms(...)
   ```

3. Refactor `w_sep` and `t_sep` to use `OffsetEqBlockPlan`.

   These are the easiest because they already work with dense vectors.

4. Add a strided summary helper for ring rows.

   Then refactor:

   ```text
   eval_d_matrix_w_residual_direct
   eval_b_matrix_t_residual_direct
   ```

5. Add `HighEqMode::Materialize`.

   Keep the default as `OnDemand` until benchmark data shows a win.

6. Benchmark with:

   ```bash
   HACHI_MODE=onehot_d32 HACHI_NUM_VARS=32 HACHI_PROFILE_TRACE=0 cargo run --release --example profile
   ```

   Compare spans:

   ```text
   m_eval_w_sep
   m_eval_w_d
   m_eval_t_sep
   m_eval_t_b
   ```

7. If materializing `eq_high` helps for some levels, add a heuristic.

   Otherwise keep it as an explicit mode for experiments.

## 13. Risks And Design Constraints

The main risk is accidentally making the separable cases slower. A too-generic
closure-based API can hide reuse and force per-`(q, block)` work that the current
code avoids.

So the abstraction should not be:

```text
everything must go through value_at(q, block)
```

It should instead separate:

```text
offset/carry/equality planning
```

from:

```text
how the caller builds carry_terms
```

That gives us generic correctness and less duplicated carry logic while
preserving the specialized fast paths for separable and matrix-backed terms.

## 14. Recommended Direction

The best first implementation is:

```text
OffsetEqBlockPlan + finish + summarize_values
```

This directly cleans up the duplicated low/high equality logic without changing
the algorithm.

Then add:

```text
summarize_strided_ring_row
```

to cover `w_d` and `t_b`.

Only after that should we experiment with:

```text
materialized eq_high
```

because the current on-demand high evaluation may already be better for many
levels. The API can support both, but the initial refactor should preserve the
current behavior by default.
