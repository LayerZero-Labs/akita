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
`crates/akita-verifier/src/protocol/ring_switch.rs`.

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
z_hat ‖ e_hat ‖ t_hat ‖ b_zk ‖ d_zk ‖ r_tail
```

When `z_first = false`:

```text
e_hat ‖ t_hat ‖ b_zk ‖ d_zk ‖ z_hat ‖ r_tail
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
| `e_hat` | `do · C · B` | `dig → claim → block` | opening witness |
| `t_hat` | `do · n_A · C · B` | `a_row → dig → claim → block` | per-`A`-row opening witness |
| `z_hat` | `dc · df · block_len` | `dc → df → block` | committed fold response |
| `b_zk` | (zk only) | blinding planes | `B`-side blinding |
| `d_zk` | (zk only) | blinding planes | `D`-side blinding |
| `r_tail` | `rows · levels` | `level → row` | gadget residue rows |

For example, `e_hat` has the digit index as its outermost axis, then the claim
index, and finally the block index as the innermost axis. Concretely, the flat
column coordinate of a `e_hat` cell is

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
small tables. These blocks act on `e_hat`, `t_hat`, and `z_hat`. We work through
`e_hat` in full detail; `t_hat` and `z_hat` follow the same recipe.

### e-related tensor components

One row block of $M$ acts on the `e_hat` columns with fully separable
(tensor-product) structure — every entry on the row is a product of small,
public per-axis factors. This is the **consistency component**
($c^{\top} \otimes G$): it ties `e_hat` to the stage-1 challenge combination
after the ring-switch evaluation at $\alpha$.

It occupies a single fixed row $i$ of $M$ — the **consistency row**. (The rows
of $M$ are few, on the order of 10–20, grouped by purpose: the consistency row,
the commitment rows of `A`/`B`/`D`, the `r`-tail rows, and so on. This component
is exactly the consistency row.) Because $i$ is a single fixed row,
`row_weight[i]` is just that row's precomputed equality weight, applied once at
the very end.

#### The cell formula

A `e_hat` cell is indexed by a digit, a claim, and a block, with local column
coordinate

$$
c \;=\; \text{block} \;+\; B \cdot (\text{claim} + C \cdot \text{dig}),
$$

so the global column is $e_{\text{start}} + c$, where $e_{\text{start}}$ is the
start of the `e_hat` segment. The entry on the consistency row is a product of
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
                 * eq(e_start + c, r_col)
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
e_{\text{start}} \;=\; e_{\text{lo}} \;+\; B \cdot e_{\text{hi}},
\qquad e_{\text{lo}} = e_{\text{start}} \bmod B,
\quad e_{\text{hi}} = \lfloor e_{\text{start}} / B \rfloor.
$$

Then the global column is

$$
e_{\text{start}} + c \;=\; (e_{\text{lo}} + \text{block}) \;+\; B \cdot (e_{\text{hi}} + \text{claim} + C \cdot \text{dig}).
$$

The low part $s = e_{\text{lo}} + \text{block}$ ranges over $[0, 2B - 2]$, so it
produces **at most one carry bit**:

$$
e_{\text{start}} + c \;=\; (s \bmod B) \;+\; B \cdot (e_{\text{hi}} + \text{claim} + C \cdot \text{dig} + \text{carry}),
\qquad \text{carry} = \lfloor s / B \rfloor \in \{0, 1\}.
$$

The low part now occupies exactly the low $\log_2 B$ bits and the high part the
rest, so the equality polynomial factors:

$$
\mathrm{eq}(e_{\text{start}} + c, \; r_{\text{col}})
\;=\;
\mathrm{eq}_{\text{low}}(s \bmod B) \cdot \mathrm{eq}_{\text{high}}(e_{\text{hi}} + \text{claim} + C \cdot \text{dig} + \text{carry}),
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
        s = e_lo + block
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
    q = e_hi + claim + C*dig
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
as the `e_hat` component, but applied once per row of the consistency-check
matrix `A`. So instead of a single row it occupies $n_A$ rows of $M$ — the
`A`-rows — each carrying its own `row_weight`.

#### The cell formula

A `t_hat` cell is indexed by an `A`-row, a digit, a claim, and a block (this is
the same `e_hat` shape with one extra `a_row` axis tensored on). Its local
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
claim-and-block-coupled quantity that appears in the `e_hat` consistency
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

As in the `e_hat` case, $s = t_{\text{lo}} + \text{block}$ lies in $[0, 2B - 2]$,
giving **at most one carry bit**, and the equality polynomial factors into a low
(block-window) factor and a high factor:

$$
\mathrm{eq}(t_{\text{start}} + c, \; r_{\text{col}})
\;=\;
\mathrm{eq}_{\text{low}}(s \bmod B) \cdot \mathrm{eq}_{\text{high}}(t_{\text{hi}} + \text{claim} + C \cdot \text{dig} + C \cdot \text{do} \cdot \text{a\_row} + \text{carry}).
$$

The block-dependent factor is again $c_{\alpha}[\text{claim}, \text{block}] \cdot \mathrm{eq}_{\text{low}}(\cdot)$
— **identical** to the `e_hat` consistency component. In fact the two segments
share the same $\mathrm{eq}_{\text{low}}$ table: their offsets differ by a
multiple of $B$ (the `e_hat` segment length is a multiple of $B$), so
$t_{\text{lo}} = e_{\text{lo}}$ and the carry split is the same. Therefore the
**per-claim block summaries are exactly the same**, and are computed only once
and reused here:

```text
# Per-claim block summaries — O(C * B), shared with the e_hat component:
for claim in 0..C:
    S0[claim] = 0      # carry = 0 bucket
    S1[claim] = 0      # carry = 1 bucket
    for block in 0..B:
        s = t_lo + block            # = e_lo + block
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

The per-claim summaries cost $O(C \cdot B)$ (and are shared with the `e_hat`
component, so in practice they are free here), and the outer combine costs
$O(n_A \cdot C \cdot \text{do})$ ($\mathrm{eq}_{\text{high}}$ is cheap and
ignored). The total overhead is therefore

$$
O(C \cdot B + n_A \cdot C \cdot \text{do}),
$$

versus the naive $O(n_A \cdot B \cdot C \cdot \text{do})$, with no dense table to
store. As with `e_hat`, the heavy block scan is lifted out of the inner loops:
the only difference from the `e_hat` consistency component is the extra `a_row`
axis on the (cheap) outer combine.

### z-related tensor components

One row block of $M$ acts on the `z_hat` columns: the **in-block consistency
contribution** that ties `z_hat` to the opening point's in-block weights. Like
the `e_hat` consistency component, it sits on the consistency row of $M$, so it
shares that single row's `row_weight` (applied, with the relation's minus sign,
at the end).

Its block axis is `block_len`, and this is where `z_hat` differs from `e_hat`
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
relation). Note that, unlike the `e_hat`/`t_hat` consistency weight
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

This is the situation at the root level, and it works exactly like the `e_hat`
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
depends only on the block, so — exactly like the `e_hat` public weight — we sum
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

The remaining row blocks of $M$ are the **rows of the shared SIS commitment
matrix**. These are fundamentally different from the tensor components: a tensor
entry is a product of small public scalars, but a setup entry is an actual
cyclotomic **ring element** of the commitment matrix. Before it can enter the
scalar sum-check, each such ring element must be evaluated at the ring-switch
challenge $\alpha$ — turning a degree-$D$ ring element into a single
extension-field scalar. That $\alpha$-evaluation costs $O(D)$ and there are many
entries, which is exactly why the setup contribution dominates verifier time.
The whole game here is to do the expensive $\alpha$-evaluations **once** and let
each one feed all the products that need it.

### The three setup row blocks

Three row blocks of $M$ are sub-views of the **same** backing SIS matrix:

- **`D · e_hat`** — $n_d$ rows of `D`, over the `e_hat` columns.
- **`B · t_hat`** — $n_b$ rows of `B`, over the `t_hat` columns, one set per
  commitment group $g$.
- **`A · z_hat`** — $n_A$ rows of `A`, over the `z_hat` columns.

`D`, `B`, and `A` are not separate matrices: they are co-located blocks of one
shared matrix, all anchored at its top-left corner. They differ only in how many
rows they use ($n_d$, $n_b$, $n_A$), how wide their column range is, and the
per-row equality weight they carry. As in the tensor sections, those per-row
weights are just `row_weight[i]` for the appropriate rows of $M$: write
`d_w[r]`, `b_w[g][r]`, and `a_w[r]` for the weights of the $r$-th `D`-row,
$g$-th group's $r$-th `B`-row, and $r$-th `A`-row.

### Why fuse: the $\alpha$-evaluation is the cost

Because `D`, `B`, and `A` overlap on the same backing entries, a naive
evaluation that scanned `D`'s block, then `B`'s block, then `A`'s block would
$\alpha$-evaluate the same shared ring elements up to **three times**. Since the
$\alpha$-evaluation is the dominant cost, the verifier instead scans the shared
matrix **once** and reuses each $\alpha$-evaluated entry across all three
products.

### The fused per-entry computation

The verifier treats the shared matrix as one flat list of ring entries. For each
entry at (row $r$, column $c$) it does three things:

1. $\alpha$-evaluate it **once**: $r_{\text{eval}}(r, c) = $
   `eval_ring_at_pows(matrix[r][c], α)`, an $O(D)$ operation.
2. Build the **aggregated column weight** that bundles every product touching
   that entry:
   $$
   \bar{\omega}(r, c) \;=\; d_w[r] \cdot W_{\text{col}}[c]
   \;+\; \sum_g b_w[g][r] \cdot T_{\text{col}}^{(g)}[c]
   \;+\; a_w[r] \cdot Z_{\text{col}}[c].
   $$
3. Accumulate $r_{\text{eval}}(r, c) \cdot \bar{\omega}(r, c)$ into a running sum.

The entire setup contribution is then a single dot product over the shared
matrix:

$$
\text{setup} \;=\; \sum_{r, c} r_{\text{eval}}(r, c) \cdot \bar{\omega}(r, c).
$$

In pseudocode:

```text
acc = 0
for r in 0..r_max:                 # r_max = max(n_d, n_b, n_A)
  for c in 0..n_cols:              # columns of the shared matrix, scanned once
    r_eval = eval_ring_at_pows(matrix[r][c], alpha)     # the ONE O(D) alpha-eval
    bar_omega = d_w[r] * W_col[c]                         # D · e_hat   (0 if r >= n_d)
              + sum_g b_w[g][r] * T_col[g][c]             # B · t_hat   (0 if r >= n_b)
              + a_w[r] * Z_col[c]                          # A · z_hat   (0 if r >= n_A)
    acc += r_eval * bar_omega
```

Each column pattern is zero outside its own segment, and each row weight is zero
past its row count, so a given entry simply contributes to whichever of the
three products actually cover it — but its $\alpha$-evaluation happens exactly
once. This is the "free" `D·e_hat` + `B·t_hat` + `A·z_hat` fusion: the dominant
cost (the $\alpha$-evaluations) is shared across all three.

### The three column patterns

The column patterns convert a shared-matrix column $c$ into the equality weight
of the corresponding witness entry — and they reuse the very same `eq_low` /
`eq_high` tables (and block-window carry machinery) built for the tensor
components. They are precomputed once (not per row).

**`W_col[c]` (for `D · e_hat`).** Decode $c$ and translate it to the `e_hat`
equality weight, in $O(1)$ per cell:

```text
dig   = c mod do
b     = (c / do) mod B
claim = c / (B * do)
q     = dig * C + claim
s     = block_offset_low + b
W_col[c] = eq_low[s mod B] * eq_high_e[q + (s / B)]      # s / B in {0, 1} is the carry
```

This is exactly the peeled-block factorization from the `e_hat` tensor section,
reusing the same `eq_low` table and the `e_hat` high table `eq_high_e`.

**`T_col[g][c]` (for `B · t_hat`).** The same shape with the extra `a_row` axis
and per-group sparsity: a group-local polynomial slot is decoded from $c$, then
mapped to its global flat claim (or skipped if that slot is not opened in group
$g$). It reuses the `t_hat` high table.

**`Z_col[c]` (for `A · z_hat`).** This follows the two `z_hat` cases. When
`block_len` is a power of two it uses the peeled-block form with a small
precomputed table that has already folded in the fold-digit (`df`) sum:

```text
Z_col[c] = eq_low_z[low_idx] * S_per_dc_per_carry[dc][carry]
S_per_dc_per_carry[dc][carry]
    = -sum_{df} g_fold[df] * eq_high_z[z_offset_high + (df + DF*dc) + carry]
```

When `block_len` is not a power of two it builds `Z_col` densely from a one-shot
peeled equality cache. Either way the per-row inner-product loop above is
identical — only the `Z_col` build differs.

### Two coordinate systems

There is one subtlety in the scan. The witness segments are stored in
**M-layout** (block innermost — the layout that defines the MLE and that the
tensor sections used), while the SIS matrix is stored in **D-physical layout**
(digit innermost — the prover's natural commit-loop order). These are two
different orderings of the same logical `(claim, block, dig)` cells.

The verifier walks $c$ in **D-physical order** so the dominant matrix scan stays
contiguous and cache-friendly, and the column-pattern builds above bake in the
translation to the M-layout equality address (that is what the `dig`/`b`/`claim`
decode followed by the `q`/`s` recomposition does). So the hot per-row loop never
has to think about layouts — it is just a dot product of $\alpha$-evaluated
entries against a precomputed weight vector.

### Cost

| step | cost | done once per |
|---|---|---|
| `eq_high_e`, `eq_high_t`, `eq_high_z` tables | $O(C \cdot \text{do} + C \cdot \text{do} \cdot n_A + \text{DF} \cdot \text{DC})$ | verifier |
| `W_col`, `T_col`, `Z_col` builds | $O(n_{\text{cols}} \cdot (1 + G))$ | verifier |
| $\alpha$-evaluations $r_{\text{eval}}(r, c)$ | $O(n_{\text{cols}} \cdot D)$ ring ops | per row |
| per-row inner product | $O(n_{\text{cols}} \cdot (1 + G))$ | per row |
| **dominant** | $O(r_{\max} \cdot n_{\text{cols}} \cdot D)$ $\alpha$-evaluations | total |

Here $r_{\max} = \max(n_d, n_b, n_A)$, $G$ is the number of commitment groups,
and $n_{\text{cols}}$ is the width of the shared scan (the max of the three
blocks' column widths). The decisive point is the bottom row: the $O(D)$
$\alpha$-evaluations — the verifier's true bottleneck — are performed exactly
once per shared-matrix entry and amortized across `D·e_hat`, `B·t_hat`, and
`A·z_hat`, rather than once per product.

## A future optimization: partial peeling for non-power-of-two `block_len`

> **Status: not implemented.** This is a design note for a possible future
> optimization of the `z`-tensor dense fallback (Case 2 in the z-related tensor
> components). It is recorded here so the reasoning and the expected gain do not
> have to be re-derived.

### Context

The power-of-two fast path (Case 1) requires `block_len.is_power_of_two()`. When
it is not (some recursive levels — `block_len = ceil(num_ring / num_blocks)` need
not be a power of two), the dense fallback (Case 2) materializes the structured
`z` segment and runs a single `eval_offset_eq_tensor` over it, at cost
`O(DF · DC · block_len)`. The question is whether a *partial* peel — peeling the
largest power of two that divides `block_len` — can beat that, and by how much.

### Why only the 2-adic factor can be peeled

A peeled window of size `w = 2^k` isolates the block axis (so its `eq` factor is
reusable across the outer `(df, dc)` axes) only if **both**:

1. **the window contains the whole block index** — `w ≥ block_len`, so all of
   `blk ∈ [0, block_len)` lives in the low `k` bits; and
2. **the window divides the outer stride** — `w | block_len`, so incrementing an
   outer index shifts the column by a whole number of windows and leaves the low
   `k` bits undisturbed.

Both at once force `w = block_len` and `w` a power of two — i.e. exactly the
fast-path case. For non-power-of-two `block_len` the only window satisfying (2)
is the 2-adic factor `w = 2^v`, `v = v₂(block_len)`. It satisfies (2) but not (1):
peeling it isolates `blk mod 2^v`, but the **odd cofactor** `odd = block_len / 2^v`
stays entangled with `(df, dc)` and remains dense. (Underlying fact: `eq`
factorises over *binary bits*, so it only splits at power-of-two boundaries;
`x mod block_len` is a fixed low-bit window iff `block_len` is a power of two. The
largest power of two *below* `block_len` does **not** divide it, so it does not
help — it must *divide*, not merely be smaller.)

For odd `block_len`, `2^v = 1` and there is nothing to peel.

### Cost model

Let `2^v = 2-adic factor of block_len` and `odd = block_len / 2^v`. Write
`blk = blk_lo + 2^v · blk_hi`. Build a low summary **per `blk_hi`** (scanning each
`a[blk]` once), then combine over the high index
`q = z_hi + blk_hi + odd · df + odd · DF · dc`:

```text
dense (Case 2):   O(block_len · DF · DC)                 [ = z_len ]
partial peel:     O(block_len + (block_len / 2^v) · DF · DC)
```

The `DF · DC`-multiplied term shrinks by exactly `2^v`; the `O(block_len)`
summary-build floor is unavoidable (every in-block weight is read once). For
one-hot presets `DC = 1`, so `z_len = DF · block_len`.

### Worked example — `nv = 32`, fp128 `D64` one-hot

Per-level schedule (`depth_commit = 1`, `z_first = true` on all 8 levels). The
dense cost equals `z_len`; the partial-peel cost is
`block_len + (block_len / 2^v) · DF`:

| level | block_len | DF | pow2? | `2^v` | odd | dense (`z_len`) | partial peel | speedup |
|---|---|---|---|---|---|---|---|---|
| 0 | 65536 | 7 | yes (2¹⁶) | full | 1 | ~65536 (fast path) | — (already peeled) | — |
| 1 | 6660 | 6 | no | 4 | 1665 | 39960 | 16650 | 2.4× |
| 2 | 2801 | 4 | no | 1 | 2801 | 11204 | 11204 | 1× |
| 3 | 1238 | 4 | no | 2 | 619 | 4952 | 3714 | 1.33× |
| 4 | 1087 | 3 | no | 1 | 1087 | 3261 | 3261 | 1× |
| 5 | 596 | 3 | no | 4 | 149 | 1788 | 1043 | 1.7× |
| 6 | 412 | 3 | no | 4 | 103 | 1236 | 721 | 1.7× |
| 7 | 343 | 3 | no | 1 | 343 | 1029 | 1029 | 1× |

Totals (including L0's fast path ≈ 65536):

- all dense: ≈ **128,966**
- with partial peel: ≈ **103,158**
- reduction: ≈ **20%** of total `z`-tensor work.

### What the numbers say

- **L0 dominates and is untouched.** `block_len = 65536` already runs the full
  fast-path peel (cost ≈ `block_len` ≈ 65536) — roughly half the total `z`-work —
  and partial peeling cannot improve it.
- **Almost all the saving is one level.** Of the ≈ 25.8k saved, ≈ **23.3k (90%)
  is L1 alone** (`DF = 6`, `block_len = 6660`).
- **Odd levels gain nothing.** L2 (2801), L4 (1087), L7 (343) have `2^v = 1`.
- **The factor is tiny.** `2^v ≤ 4` here, and the `O(block_len)` floor remains, so
  the win is a small constant, not asymptotic.

### Verdict

Technically helpful on 4 of the 7 recursive levels, but **not worth it now**:

- It is a sub-20% improvement on the `z`-tensor component, which is **not** the
  verifier bottleneck — the setup-matrix α-evaluation scan (Setup components)
  dominates; the structured tensor blocks are the cheap part.
- The gain is concentrated in a single level and is zero for odd `block_len` (the
  generic non-power-of-two case).
- It requires a third `z` evaluation path (hybrid peel + dense) inside the
  verifier no-panic boundary — more code and more surface for bugs.

It would become attractive only if `depth_fold · depth_commit` were large **and**
recursive `block_len`s were consistently `2^v · (small odd)` with large `2^v`. The
current planner splits do not produce that, so the implementation keeps the plain
`ZDenseSlicesEvaluator` materialisation (Case 2).

### Implementation sketch (if revisited)

- Add a `ZPartialPeelSlicesEvaluator`, dispatched when
  `!block_len.is_power_of_two() && v₂(block_len) >= threshold`; otherwise keep
  `ZDenseSlicesEvaluator`.
- Peel `w = 2^{v₂(block_len)}` low bits: `eq_low_z` over those bits, two carry
  buckets (the carry-≤-1 argument from Case 1 holds for the power-of-two window
  `w`).
- Build per-`blk_hi` summaries
  `A[blk_hi][carry] = Σ_{blk_lo} a[blk_lo + w · blk_hi] · eq_low_z(...)`, cost
  `O(block_len)`.
- Outer combine over `(dc, df, blk_hi, carry)` with
  `q = z_hi + blk_hi + odd · df + odd · DF · dc` and `eq_high(q + carry)`, cost
  `O(odd · DF · DC)`.
- Reuse the `crates/akita-algebra/src/offset_eq.rs` peeled primitives
  (`summarize_pow2_block_carries`, `eval_offset_eq_peeled_carry_terms`) at window
  `2^v` instead of `block_len`.
