# Matrix evaluation at a point

During Akita's relation-check sum-check, the verifier has to evaluate the
multilinear extension of the relation matrix $M$ at a random row point
$r_{\text{row}}$ and a random column point $r_{\text{col}}$. This chapter
explains how that evaluation is performed **without ever materializing $M$**,
which is what makes verification fast.

This step is the single most dominant component of the verifier. Evaluating it
efficiently therefore directly determines the total Akita commitment
verification time, so it is worth understanding in detail.

## What the verifier needs to compute

By definition, the quantity the verifier needs is

$$
\widetilde{M}(r_{\text{row}}, r_{\text{col}})
\;=\;
\sum_{i, j} \mathrm{eq}(i, r_{\text{row}}) \cdot \mathrm{eq}(j, r_{\text{col}}) \cdot M(i, j),
$$

where $i$ ranges over the rows of $M$, $j$ ranges over its columns, and
$\mathrm{eq}$ is the multilinear equality polynomial.

The number of rows of $M$ is small in Akita — on the order of 10–20. Computing
$\mathrm{eq}(i, r_{\text{row}})$ for every row is therefore not costly at all,
and the verifier does it once, up front, storing one weight per row:

```text
row_weight[i] = eq(i, r_row).
```

After this, the task reduces to: for each row $i$, evaluate that row's column
contribution under $\mathrm{eq}(\cdot, r_{\text{col}})$, and accumulate it
weighted by `row_weight[i]`.

The hard part is the column axis. The number of columns equals the witness size,
which is large. A naive evaluation would instantiate the full equality vector
$\mathrm{eq}(\cdot, r_{\text{col}})$ and the full materialized matrix $M$, then
take an inner product. This is bad on both axes: it costs too much **memory**
(storing $M$ and the dense equality table) and too much **computation** (a scan
over every cell of $M$).

The good news is that $M$ is not an arbitrary matrix. It is assembled from a
small, known set of structured components, and we can exploit that structure to
evaluate $\widetilde{M}$ far more cheaply — touching each component with work
proportional to its description, not to its materialized size. Each *row block*
of $M$ corresponds to one such component. The rest of this chapter explains how
each component is evaluated efficiently.

The canonical implementation lives in
`crates/akita-verifier/src/protocol/ring_switch.rs`, and the precise per-block
formulas are recorded in `specs/optimized_verifier.md`.

## Connection to the witness and witness shape

The matrix $M$ is not defined in isolation: its columns must line up with the
way the witness is laid out, so that $M$ generates the correct equations. In the
sections that follow we use the structure of the witness to locate the
corresponding entries inside $M$. It is worth stressing that the layout choices
in $M$ (which segment goes where, in what order) are made for efficiency and
engineering reasons only — **they do not change the security properties or the
claims that Akita proves.**

### Two column orderings

The witness comes in one of two layouts, selected by a single boolean
`z_first`. The motivation for having this choice is discussed in a later
section; for now we just record the two shapes.

When `z_first = true`:

```text
z_hat ‖ w_hat ‖ t_hat ‖ b_zk ‖ d_zk ‖ r_tail
```

When `z_first = false`:

```text
w_hat ‖ t_hat ‖ b_zk ‖ d_zk ‖ z_hat ‖ r_tail
```

Each segment occupies a disjoint, contiguous range of columns. The `b_zk` and
`d_zk` blinding segments are present only in zero-knowledge builds and are empty
otherwise.

### Segment sizes and layout

The shapes are parameterized by the per-level quantities:

- `B` — the number of blocks (a power of two);
- `block_len` — the length of each block;
- `C` — the number of claims;
- `n_A` — the number of rows of the commitment matrix `A`;
- `dc` — the number of commit-side gadget digits;
- `do` — the number of open-side gadget digits;
- `df` — the number of fold-side gadget digits.

Each segment is a nested (tensor) indexing of these axes. We list the axis order
from **outermost to innermost**, so the last-named axis is the fastest-varying:

| segment | size (ring elements) | axis order (outermost → innermost) | role |
|---|---|---|---|
| `w_hat` | `do · C · B` | `dig → claim → block` | opening witness |
| `t_hat` | `do · n_A · C · B` | `a_row → dig → claim → block` | per-`A`-row opening witness |
| `z_hat` | `dc · df · block_len` | `dc → df → block` | committed fold response |
| `b_zk` | (zk only) | blinding planes | `B`-side blinding |
| `d_zk` | (zk only) | blinding planes | `D`-side blinding |
| `r_tail` | `rows · levels` | `level → row` | gadget residue rows |

For example, `w_hat` has the digit index as its outermost axis, then the claim
index, and finally the block index as the innermost axis. Concretely, the flat
column coordinate of a `w_hat` cell is

$$
\text{block} \;+\; B \cdot \text{claim} \;+\; (B \cdot C) \cdot \text{dig},
$$

so the block index varies fastest, then the claim, then the digit.

### Why the block axis is innermost

In every witness segment the **block axis is the innermost (fastest-varying)
axis.** This is by design. Because `B` (and, at the root level, `block_len`) is
a power of two, putting the block axis in the contiguous low bits of the column
coordinate lets the verifier *factor* the equality polynomial
$\mathrm{eq}(\cdot, r_{\text{col}})$ along the block-bit window. That
factorization is the key that turns the per-component evaluation from a full
column scan into cheap, structured arithmetic — the optimization that the next
sections build on.

## Tensor components

Three of $M$'s row blocks act on the witness with fully separable
(tensor-product) structure: every entry on these rows is a product of small,
public per-axis factors. That structure is what lets the verifier avoid scanning
the rows cell by cell — it peels the block axis off once and combines a few
small tables. These blocks act on `w_hat`, `t_hat`, and `z_hat`. We work through
`w_hat` in full detail; `t_hat` and `z_hat` follow the same recipe.

### w-related tensor components

One row block of $M$ acts on the `w_hat` columns with fully separable
(tensor-product) structure — every entry on the row is a product of small,
public per-axis factors. This is the **consistency component**
($c^{\top} \otimes G$): it ties `w_hat` to the stage-1 challenge combination
after the ring-switch evaluation at $\alpha$.

It occupies a single fixed row $i$ of $M$ — the **consistency row**. (The rows
of $M$ are few, on the order of 10–20, grouped by purpose: the consistency row,
the commitment rows of `A`/`B`/`D`, the `r`-tail rows, and so on. This component
is exactly the consistency row.) Because $i$ is a single fixed row,
`row_weight[i]` is just that row's precomputed equality weight, applied once at
the very end.

#### The cell formula

A `w_hat` cell is indexed by a digit, a claim, and a block, with local column
coordinate

$$
c \;=\; \text{block} \;+\; B \cdot (\text{claim} + C \cdot \text{dig}),
$$

so the global column is $w_{\text{start}} + c$, where $w_{\text{start}}$ is the
start of the `w_hat` segment. The entry on the consistency row is a product of
two public per-axis factors:

$$
M(i, c) \;=\; c_{\alpha}[\text{claim}, \text{block}] \cdot g_{\text{open}}[\text{dig}],
$$

where $g_{\text{open}}[\text{dig}]$ is the open-side gadget weight (e.g.
$\text{base}^{\text{dig}}$) and $c_{\alpha}[\text{claim}, \text{block}]$ is the
$\alpha$-evaluation of the stage-1 sparse challenge for the pair
$(\text{claim}, \text{block})$. The feature that drives the evaluation is that
$c_{\alpha}$ **couples the claim and the block together** — it is *not* a product
of a separate per-claim factor and a per-block factor.

#### The naive evaluation

Written directly, the component's contribution to $\widetilde{M}$ walks every
column of the segment:

```text
acc = 0
for dig   in 0..do:
  for claim in 0..C:
    for block in 0..B:
      c = block + B*(claim + C*dig)
      acc += row_weight[i] * c_alpha[claim][block] * g_open[dig]
                 * eq(w_start + c, r_col)
```

This costs $O(B \cdot C \cdot \text{do})$ multiplications for the triple loop,
plus another $O(B \cdot C \cdot \text{do})$ to precompute the dense table of
$\mathrm{eq}(\cdot, r_{\text{col}})$ over the segment — and $O(B \cdot C \cdot \text{do})$
memory to store it. Both slow and memory-hungry.

#### The optimization: a block summary per claim

In practice the digit count `do` and the claim count `C` are small, while the
block count `B` is the large axis. We want to do the per-block work in an inner
loop and reuse it.

Split the segment offset into a low part (the block window) and a high part:

$$
w_{\text{start}} \;=\; w_{\text{lo}} \;+\; B \cdot w_{\text{hi}},
\qquad w_{\text{lo}} = w_{\text{start}} \bmod B,
\quad w_{\text{hi}} = \lfloor w_{\text{start}} / B \rfloor.
$$

Then the global column is

$$
w_{\text{start}} + c \;=\; (w_{\text{lo}} + \text{block}) \;+\; B \cdot (w_{\text{hi}} + \text{claim} + C \cdot \text{dig}).
$$

The low part $s = w_{\text{lo}} + \text{block}$ ranges over $[0, 2B - 2]$, so it
produces **at most one carry bit**:

$$
w_{\text{start}} + c \;=\; (s \bmod B) \;+\; B \cdot (w_{\text{hi}} + \text{claim} + C \cdot \text{dig} + \text{carry}),
\qquad \text{carry} = \lfloor s / B \rfloor \in \{0, 1\}.
$$

The low part now occupies exactly the low $\log_2 B$ bits and the high part the
rest, so the equality polynomial factors:

$$
\mathrm{eq}(w_{\text{start}} + c, \; r_{\text{col}})
\;=\;
\mathrm{eq}_{\text{low}}(s \bmod B) \cdot \mathrm{eq}_{\text{high}}(w_{\text{hi}} + \text{claim} + C \cdot \text{dig} + \text{carry}),
$$

where $\mathrm{eq}_{\text{low}}$ uses the low $\log_2 B$ challenge bits of
$r_{\text{col}}$ and $\mathrm{eq}_{\text{high}}$ the remaining bits.

The block-dependent factor is $c_{\alpha}[\text{claim}, \text{block}] \cdot \mathrm{eq}_{\text{low}}(\cdot)$,
and because $c_{\alpha}$ depends on the claim, this sum is **not** the same for
every claim. So we cannot collapse the blocks into one summary — we build one
two-bucket block summary **per claim**:

```text
# Per-claim block summaries — O(C * B):
for claim in 0..C:
    S0[claim] = 0      # carry = 0 bucket
    S1[claim] = 0      # carry = 1 bucket
    for block in 0..B:
        s = w_lo + block
        if s >= B:
            S1[claim] += c_alpha[claim][block] * eq_low[s - B]
        else:
            S0[claim] += c_alpha[claim][block] * eq_low[s]
```

The outer combination then runs only over the small axes, reusing the
summaries:

```text
# Outer combine — O(C * do):
acc = 0
for dig   in 0..do:
  for claim in 0..C:
    q = w_hi + claim + C*dig
    acc += g_open[dig] * (S0[claim] * eq_high[q] + S1[claim] * eq_high[q + 1])
acc *= row_weight[i]
```

#### Cost

Building the per-claim summaries costs $O(C \cdot B)$, and the outer combine
costs $O(C \cdot \text{do})$ ($\mathrm{eq}_{\text{high}}$ is cheap to precompute
and ignored). The total overhead is therefore

$$
O(C \cdot B + C \cdot \text{do}) \;=\; O\!\big(C \cdot (B + \text{do})\big),
$$

versus the naive $O(B \cdot C \cdot \text{do})$, and with no dense table to
store. Intuitively, the heavy block scan now runs once **per claim** ($C \cdot B$)
instead of once per (claim, digit) pair ($B \cdot C \cdot \text{do}$): the digit
axis has been removed from the dominant block loop. The claim axis stays on the
block scan because $c_{\alpha}$ couples the claim and the block; if the per-block
weight did not depend on the claim, a single $O(B)$ summary would suffice.

### t-related tensor components

One row block of $M$ acts on the `t_hat` columns. It is the **per-`A`-row
consistency component** ($c^{\top} \otimes G_{n_A}$): the same consistency check
as the `w_hat` component, but applied once per row of the consistency-check
matrix `A`. So instead of a single row it occupies $n_A$ rows of $M$ — the
`A`-rows — each carrying its own `row_weight`.

#### The cell formula

A `t_hat` cell is indexed by an `A`-row, a digit, a claim, and a block (this is
the same `w_hat` shape with one extra `a_row` axis tensored on). Its local
column coordinate is

$$
c \;=\; \text{block} \;+\; B \cdot (\text{claim} + C \cdot \text{dig} + C \cdot \text{do} \cdot \text{a\_row}),
$$

so the global column is $t_{\text{start}} + c$, where $t_{\text{start}}$ is the
start of the `t_hat` segment. The block on `t_hat` is **block-diagonal in the
`a_row` axis**: the $a_{\text{row}}$-th row of $M$ has nonzero entries only on the
column slice with the matching `a_row`, and there the entry is a product of two
public per-axis factors:

$$
M(i = \text{a\_row}, \, c) \;=\; c_{\alpha}[\text{claim}, \text{block}] \cdot g_{\text{open}}[\text{dig}],
$$

with the whole slice weighted by `row_weight[a_row]`, the equality weight of that
`A`-row. As before, $g_{\text{open}}[\text{dig}]$ is the open-side gadget weight
and $c_{\alpha}[\text{claim}, \text{block}]$ is the $\alpha$-evaluation of the
stage-1 sparse challenge for $(\text{claim}, \text{block})$ — the same
claim-and-block-coupled quantity that appears in the `w_hat` consistency
component.

#### The naive evaluation

Written directly, the component walks every column of the segment, for every
`A`-row:

```text
acc = 0
for a_row in 0..n_A:
  for dig   in 0..do:
    for claim in 0..C:
      for block in 0..B:
        c = block + B*(claim + C*dig + C*do*a_row)
        acc += row_weight[a_row] * c_alpha[claim][block] * g_open[dig]
                   * eq(t_start + c, r_col)
```

This costs $O(n_A \cdot B \cdot C \cdot \text{do})$ multiplications for the loop,
plus another $O(n_A \cdot B \cdot C \cdot \text{do})$ to precompute and store the
dense $\mathrm{eq}(\cdot, r_{\text{col}})$ table over the segment.

#### The optimization: reuse the per-claim block summaries

Split the segment offset into a low part (the block window) and a high part:

$$
t_{\text{start}} \;=\; t_{\text{lo}} \;+\; B \cdot t_{\text{hi}},
\qquad t_{\text{lo}} = t_{\text{start}} \bmod B,
\quad t_{\text{hi}} = \lfloor t_{\text{start}} / B \rfloor.
$$

The global column becomes

$$
t_{\text{start}} + c \;=\; (t_{\text{lo}} + \text{block}) \;+\; B \cdot (t_{\text{hi}} + \text{claim} + C \cdot \text{dig} + C \cdot \text{do} \cdot \text{a\_row}).
$$

As in the `w_hat` case, $s = t_{\text{lo}} + \text{block}$ lies in $[0, 2B - 2]$,
giving **at most one carry bit**, and the equality polynomial factors into a low
(block-window) factor and a high factor:

$$
\mathrm{eq}(t_{\text{start}} + c, \; r_{\text{col}})
\;=\;
\mathrm{eq}_{\text{low}}(s \bmod B) \cdot \mathrm{eq}_{\text{high}}(t_{\text{hi}} + \text{claim} + C \cdot \text{dig} + C \cdot \text{do} \cdot \text{a\_row} + \text{carry}).
$$

The block-dependent factor is again $c_{\alpha}[\text{claim}, \text{block}] \cdot \mathrm{eq}_{\text{low}}(\cdot)$
— **identical** to the `w_hat` consistency component. In fact the two segments
share the same $\mathrm{eq}_{\text{low}}$ table: their offsets differ by a
multiple of $B$ (the `w_hat` segment length is a multiple of $B$), so
$t_{\text{lo}} = w_{\text{lo}}$ and the carry split is the same. Therefore the
**per-claim block summaries are exactly the same**, and are computed only once
and reused here:

```text
# Per-claim block summaries — O(C * B), shared with the w_hat component:
for claim in 0..C:
    S0[claim] = 0      # carry = 0 bucket
    S1[claim] = 0      # carry = 1 bucket
    for block in 0..B:
        s = t_lo + block            # = w_lo + block
        if s >= B:
            S1[claim] += c_alpha[claim][block] * eq_low[s - B]
        else:
            S0[claim] += c_alpha[claim][block] * eq_low[s]
```

The only new work is the outer combine, which now also ranges over the `a_row`
axis:

```text
# Outer combine — O(n_A * C * do):
acc = 0
for a_row in 0..n_A:
  for dig   in 0..do:
    for claim in 0..C:
      q = t_hi + claim + C*dig + C*do*a_row
      acc += row_weight[a_row] * g_open[dig]
               * (S0[claim] * eq_high[q] + S1[claim] * eq_high[q + 1])
```

#### Cost

The per-claim summaries cost $O(C \cdot B)$ (and are shared with the `w_hat`
component, so in practice they are free here), and the outer combine costs
$O(n_A \cdot C \cdot \text{do})$ ($\mathrm{eq}_{\text{high}}$ is cheap and
ignored). The total overhead is therefore

$$
O(C \cdot B + n_A \cdot C \cdot \text{do}),
$$

versus the naive $O(n_A \cdot B \cdot C \cdot \text{do})$, with no dense table to
store. As with `w_hat`, the heavy block scan is lifted out of the inner loops:
the only difference from the `w_hat` consistency component is the extra `a_row`
axis on the (cheap) outer combine.

### z-related tensor components

One row block of $M$ acts on the `z_hat` columns: the **in-block consistency
contribution** that ties `z_hat` to the opening point's in-block weights. Like
the `w_hat` consistency component, it sits on the consistency row of $M$, so it
shares that single row's `row_weight` (applied, with the relation's minus sign,
at the end).

Its block axis is `block_len`, and this is where `z_hat` differs from `w_hat`
and `t_hat`: their block axis is `B = num_blocks`, which is **always** a power of
two, but `block_len` is a power of two only at the root level. At recursive
levels `block_len = ceil(num_ring / num_blocks)` can be any integer. Whether the
fast peeled-block path applies therefore depends on `block_len`, and we cover
both cases.

#### The cell formula

A `z_hat` cell is indexed by a commit digit, a fold digit, and an in-block
position, with local column coordinate

$$
c \;=\; \text{blk} \;+\; \text{block\_len} \cdot (\text{df} + \text{DF} \cdot \text{dc}),
$$

so the global column is $z_{\text{start}} + c$, where $z_{\text{start}}$ is the
start of the `z_hat` segment. The entry is a triple tensor product of public
per-axis factors:

$$
M(i, c) \;=\; g_{\text{commit}}[\text{dc}] \cdot g_{\text{fold}}[\text{df}] \cdot a[\text{blk}],
$$

where $g_{\text{commit}}[\text{dc}]$ is the commit-side gadget weight,
$g_{\text{fold}}[\text{df}]$ is the fold-side gadget weight, and $a[\text{blk}]$
is the opening point's in-block weight. The consistency row contributes this
weighted by $-\,\text{consistency\_weight}$ (its `row_weight`, negated by the
relation). Note that, unlike the `w_hat`/`t_hat` consistency weight
$c_{\alpha}$, the block factor $a[\text{blk}]$ depends on the block **only** — it
is not coupled to the digit axes.

#### The naive evaluation

```text
acc = 0
for dc  in 0..DC:
  for df  in 0..DF:
    for blk in 0..block_len:
      c = blk + block_len*(df + DF*dc)
      acc += g_commit[dc] * g_fold[df] * a[blk]
                 * eq(z_start + c, r_col)
acc *= -consistency_weight
```

This costs $O(\text{DF} \cdot \text{DC} \cdot \text{block\_len})$ for the loop
plus the same again to build and store the dense
$\mathrm{eq}(\cdot, r_{\text{col}})$ table over the segment.

#### Case 1: `block_len` is a power of two (peeled fast path)

This is the situation at the root level, and it works exactly like the `w_hat`
component — except the peeled window is `block_len` (not `B`) and the small outer
axes are the gadget digits `(dc, df)` (not the claims).

Split the offset against the block window:

$$
z_{\text{start}} \;=\; z_{\text{lo}} \;+\; \text{block\_len} \cdot z_{\text{hi}},
\qquad z_{\text{lo}} = z_{\text{start}} \bmod \text{block\_len},
\quad z_{\text{hi}} = \lfloor z_{\text{start}} / \text{block\_len} \rfloor.
$$

Since $s = z_{\text{lo}} + \text{blk}$ lies in $[0, 2\,\text{block\_len} - 2]$
there is **at most one carry bit**, and the equality polynomial factors:

$$
\mathrm{eq}(z_{\text{start}} + c, \; r_{\text{col}})
\;=\;
\mathrm{eq}_{\text{low}}(s \bmod \text{block\_len}) \cdot \mathrm{eq}_{\text{high}}(z_{\text{hi}} + \text{df} + \text{DF} \cdot \text{dc} + \text{carry}).
$$

The block-dependent factor $a[\text{blk}] \cdot \mathrm{eq}_{\text{low}}(\cdot)$
depends only on the block, so — exactly like the `w_hat` public weight — we sum
it over all blocks **once** into two carry buckets:

```text
# In-block summary — computed ONCE, O(block_len):
A0 = 0      # carry = 0
A1 = 0      # carry = 1
for blk in 0..block_len:
    s = z_lo + blk
    if s >= block_len:
        A1 += a[blk] * eq_low_z[s - block_len]
    else:
        A0 += a[blk] * eq_low_z[s]
```

The outer combine then runs only over the small gadget axes:

```text
# Outer combine — O(DF * DC):
acc = 0
for dc in 0..DC:
  for df in 0..DF:
    q = z_hi + df + DF*dc
    acc += g_commit[dc] * g_fold[df] * (A0 * eq_high[q] + A1 * eq_high[q + 1])
acc *= -consistency_weight
```

The overhead is $O(\text{block\_len} + \text{DF} \cdot \text{DC})$, instead of
the naive $O(\text{DF} \cdot \text{DC} \cdot \text{block\_len})$, with no dense
table.

#### Case 2: `block_len` is not a power of two (dense fallback)

When `block_len` is not a power of two there is no clean low-bit window to peel:
the block index does not occupy a contiguous power-of-two range of bits, so the
two-bucket carry split above is not defined. The verifier falls back to
materializing the structured `z` segment and evaluating it with a single generic
offset-equality pass:

```text
# Materialize the structured segment (the tensor product) — O(DF * DC * block_len):
seg = []
for dc  in 0..DC:
  for df  in 0..DF:
    for blk in 0..block_len:
      seg.push(g_commit[dc] * g_fold[df] * a[blk])

# One offset-eq evaluation over the materialized vector — O(DF * DC * block_len):
#   eval_offset_eq_tensor(r_col, z_start, seg) = sum_k seg[k] * eq(z_start + k, r_col)
acc = -consistency_weight * eval_offset_eq_tensor(r_col, z_start, seg)
```

This costs $O(\text{DF} \cdot \text{DC} \cdot \text{block\_len})$ to build the
segment plus the same for the evaluation — i.e. work proportional to the full
segment size, the same order as the naive scan. That is acceptable because the
non-power-of-two case only arises at recursive levels, where the witness (and
hence `z_hat`) has already shrunk; the dominant root level always lands in Case
1.

#### Cost

| case | overhead |
|---|---|
| `block_len` a power of two (root) | $O(\text{block\_len} + \text{DF} \cdot \text{DC})$ |
| `block_len` not a power of two (some recursive levels) | $O(\text{DF} \cdot \text{DC} \cdot \text{block\_len})$ |

The verifier dispatches on `block_len.is_power_of_two()`: the power-of-two path
uses the peeled in-block summary, and the fallback materializes the segment and
runs one offset-equality evaluation over it.

## Setup components

This section will explain the contribution of the shared setup (SIS commitment)
matrix to $\widetilde{M}$.

_(Coming soon.)_
