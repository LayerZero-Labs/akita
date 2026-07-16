# The distributed relation verifier

> **Status: design note (verifier side only).** This chapter specifies the
> verifier for the relation produced by the [distributed prover](../proving/distributed-prover.md)
> of Akita. It describes *what the verifier must compute and why its cost is
> unchanged*; it is not an implementation.

> **A note on the name.** "Distributed verifier" would be a misnomer: nothing
> about the verification is distributed. There is a *single* verifier, running on
> one machine, doing the ordinary amount of work. What is distributed is the
> **prover** — several machines cooperate to produce the proof, each holding only
> a slice of the witness (see [the distributed prover](../proving/distributed-prover.md)).
> This chapter is about that single verifier checking the *relation that a
> distributed prover produces*. Hence "distributed-relation verifier", not
> "distributed verifier".

In the distributed-prover protocol several machines $P_0, \dots, P_{\mathcal M-1}$
cooperate to prove a single Akita opening, each holding only a slice of the
witness. We assume $\mathcal M = 2^N$ machines (the canonical choice; the
[distributed prover](../proving/distributed-prover.md) fixes this). The relation
matrix the verifier evaluates is therefore no longer the single-machine $M$ but
the **virtually shared** horizontal concatenation

$$
M \;=\; [\,M_0 \mid M_1 \mid \cdots \mid M_{\mathcal M-1}\,],
$$

where each $M_j$ is a single-machine-shaped relation block restricted to the
blocks $\mathcal I_j$ owned by machine $P_j$. The *rows* of $M$ are unchanged and
shared across all $M_j$ — this is a horizontal stacking, same row blocks,
partitioned columns. Consequently **every component of the single-machine $M$ now
appears as $\mathcal M$ column-repetitions**, and the verifier evaluates each
component once per machine and sums.

This chapter is a delta against
[Matrix evaluation at a point](./matrix_evaluation.md): it follows the same
sections in the same order, and for each component explains how its $\mathcal M$
repetitions are formed and combined. One distinction recurs throughout and is
worth fixing up front:

- **Partitioned segments** ($\hat e$, $\hat t$). Machine $P_j$ owns the blocks
  $\mathcal I_j \subset [B]$, so its $\hat e^{(j)}, \hat t^{(j)}$ cover only
  $|\mathcal I_j| = B/\mathcal M$ blocks. The union over machines is the full
  $\hat e$ (resp. $\hat t$); the $\mathcal M$ repetitions *tile the same data*, so
  their combined cost equals the single-machine cost.
- **Replicated segments** ($z$). Machine $P_j$'s partial fold
  $z^{(j)} = \sum_{i \in \mathcal I_j} c_i s_i$ sums over its blocks but lands in
  the *full* ambient fold space (the same dimension as the single global fold).
  The $\mathcal M$ repetitions are $\mathcal M$ full-size copies — the deliberate
  price the protocol pays to avoid an all-reduce of the fold.

The headline, derived component by component below, is that the verifier's
**dominant cost is unchanged**: the expensive $O(D)$ SIS-matrix
$\alpha$-evaluations are still performed exactly once, and all $\mathcal M$-fold
structure lands on the cheap bookkeeping. The canonical single-machine
implementation lives in `crates/akita-verifier/src/protocol/ring_switch.rs`.

Throughout we write $B_{\mathsf{loc}} := B/\mathcal M$ for the number of blocks one
machine owns. Since $\mathcal M = 2^N$ and $B = 2^r$ with $N \le r$,
$B_{\mathsf{loc}}$ is again a power of two — the property that keeps the
per-machine fast paths available.

## What the verifier needs to compute

The target is identical in form to the single-machine verifier:

$$
\widetilde{M}(r_{\text{row}}, r_{\text{col}})
\;=\;
\sum_{i, c} \mathrm{eq}(i, r_{\text{row}}) \cdot \mathrm{eq}(c, r_{\text{col}}) \cdot M(i, c).
$$

The **row axis is unchanged**: $M$ still has only $\sim 10$–$20$ rows (the
consistency row, the `A`/`B`/`D` commitment rows, the `r`-tail rows), and the
verifier still precomputes one `row_weight[i] = eq(i, r_row)` per row, up front.
Each machine writes into these *same* rows — the consistency row carries the
`e`/`z` contributions of every machine, and the $n_A$ `A`-rows carry the `t` and
`A·G_fold·z_hat` contributions of every machine — so each row weight is applied
**once**, after summing across machines.

What changed is the **column axis**: the columns now enumerate the $\mathcal M$
machine witnesses. Splitting the column sum by machine,

$$
\sum_c M(i,c)\,\mathrm{eq}(c, r_{\text{col}})
\;=\;
\sum_{j=0}^{\mathcal M-1}
\underbrace{\sum_{c \,\in\, \mathrm{cols}(j)} M_j(i,c)\,\mathrm{eq}(c, r_{\text{col}})}_{\text{machine } j\text{'s contribution to row } i},
$$

and each machine's inner sum is structurally one of the single-machine
components, restricted to machine $j$'s data and placed at machine $j$'s column
offset. So the whole chapter reduces to: take each single-machine component,
instantiate it $\mathcal M$ times, and sum — with the partitioned/replicated
distinction deciding whether "$\mathcal M$ times" costs more.

## Connection to the witness and witness shape

### The block partition

The machines split $[B]$ into deterministic, public, contiguous subsets

$$
[B] = \mathcal I_0 \sqcup \dots \sqcup \mathcal I_{\mathcal M-1},
\qquad
\mathcal I_j := \bigl\{\, i \in [B] : \lfloor i \cdot \mathcal M / B \rfloor = j \,\bigr\},
$$

so $P_j$ owns the contiguous block range
$[\,jB_{\mathsf{loc}},\,(j{+}1)B_{\mathsf{loc}}\,)$ with $B_{\mathsf{loc}} := B/\mathcal M$
blocks each. The public matrices $A$, $B$, $D$ are column views of one
seed-expanded matrix, so **every machine regenerates the same public columns
locally** — the fact that keeps the verifier's dominant cost flat.

### The witness layout is grouped by machine

The distributed prover assembles the next-level witness **one machine block at a
time**: machine $P_j$ contributes its local witness
$w_j = (\hat e^{(j)}, \hat t^{(j)}, z^{(j)})$ as a single contiguous unit, and the
lone summed quotient $\hat r$ tails the whole thing (see the
[distributed prover](../proving/distributed-prover.md), "Ring-switch lift and
next-level commitment"):

```text
[ e^(0) | t^(0) | z^(0) ][ e^(1) | t^(1) | z^(1) ] … [ e^(M-1) | t^(M-1) | z^(M-1) ][ r ]
└─────── w_0 ───────┘└─────── w_1 ───────┘          └──────── w_{M-1} ────────┘
```

So **the contiguous unit is the machine block $w_j$, not the component.** A single
component — say `e_hat` — is therefore *not* one fused region: its $\mathcal M$
pieces $\hat e^{(0)}, \dots, \hat e^{(\mathcal M-1)}$ are scattered through the
witness, one inside each $w_j$, with that machine's $\hat t^{(j)}, z^{(j)}$ (and
the next machine's block) sitting *between* consecutive e-pieces. That gap — "the
space related to other stuff" between `e^(0)` and `e^(1)` — is the normal shape,
and it does **not** complicate the verifier.

#### Why the gaps between same-component pieces are a non-issue

The single-machine verifier already evaluates `e_hat`, `t_hat`, `z_hat`, and
`r_tail` at **four different offsets** inside the witness; it never relies on a
component being one giant block. Each per-component evaluator from
[Matrix evaluation at a point](./matrix_evaluation.md) is *parameterized by its
segment's start offset* (`e_start`, `t_start`, `z_start`) and peels the block
window relative to that offset, using the at-most-one-carry split to absorb any
misalignment.

The distributed layout changes nothing structural here — it just has **more
offsets**: $\mathcal M$ `e_hat` offsets, $\mathcal M$ `t_hat` offsets, $\mathcal M$
`z_hat` offsets, one per machine block. Writing
$L := |w_j| = |\hat e^{(j)}| + |\hat t^{(j)}| + |z^{(j)}|$ for the size of one
machine block, machine $j$'s pieces start at

$$
e^{(j)}_{\text{start}} = jL, \qquad
t^{(j)}_{\text{start}} = jL + |\hat e^{(j)}|, \qquad
z^{(j)}_{\text{start}} = jL + |\hat e^{(j)}| + |\hat t^{(j)}|
$$

(plus the witness's base offset). The verifier evaluates each piece at its own
offset and sums; the "space between" two e-pieces is just the numeric difference
$e^{(j+1)}_{\text{start}} - e^{(j)}_{\text{start}} = L$, which the offset-equality
factorization handles exactly as it already handles the gap between `e_hat` and
`z_hat` in the single-machine layout.

Each piece keeps its single-machine internal axis order, with the block sub-axis
shrunk to $B_{\mathsf{loc}}$ for the partitioned `e`/`t` pieces:

| piece | size | axis order (outermost → innermost) | kind |
|---|---|---|---|
| `e_hat^(j)` | `do · C · B_loc` | `dig → claim → block_local` | partitioned |
| `t_hat^(j)` | `do · n_A · C · B_loc` | `a_row → dig → claim → block_local` | partitioned |
| `z_hat^(j)` | `dc · df · block_len` | `dc → df → blk` | **replicated** |
| `r_tail` | `rows · levels` | `level → row` (single) | shared |

The machine index is *not* an axis inside a piece — it is the outer grouping that
selects which block $w_j$ (hence which offset) a piece lives in. The `e`/`t` block
sub-axis shrinks from `B` to `B_loc` because each machine holds only its
`B_loc` blocks; the `z` block sub-axis stays the full `block_len` because the fold
is replicated, and `z_hat^(j)` is repeated $\mathcal M$ times.

> The "global block" point now reads cleanly off this layout: inside `e_hat^(j)`
> the block index restarts at `block_local ∈ [0, B_loc)`, but it names the global
> block `blk_g = j·B_loc + block_local`, which is what the folding challenge
> `c_α` is indexed by (see the e-tensor section).

### Why the block axis is still peelable, and why one `eq_low` still serves all

`matrix_evaluation` peels the block window because `B` is a power of two and the
block axis is innermost. Since $\mathcal M = 2^N$ divides $B = 2^r$, each
machine's sub-window $B_{\mathsf{loc}} = B/\mathcal M$ is again a power of two, so
`block_local` occupies a clean $\log_2 B_{\mathsf{loc}}$-bit window and the
per-machine peel (with `block_local` innermost and `(claim, dig)` in the high
bits) works exactly as single-machine.

One `eq_low` table still serves every machine — and this holds *regardless* of the
per-machine offsets, despite the gaps. The `eq_low` table is just the equality
weights over the **low $\log_2 B_{\mathsf{loc}}$ bits of $r_{\text{col}}$**; it
depends only on $r_{\text{col}}$ and the window size, never on where a segment
sits. A segment's offset enters only as two per-machine scalars: the in-window
shift $e^{(j)}_{\text{lo}} = e^{(j)}_{\text{start}} \bmod B_{\mathsf{loc}}$ (which
sets the carry split) and the high index $e^{(j)}_{\text{hi}}$. So a single
`eq_low` (window $B_{\mathsf{loc}}$) is reused by every `e_hat^(j)` and
`t_hat^(j)`, and a single `eq_low_z` (window `block_len`) by every `z_hat^(j)`;
the offsets differing per machine is fine.

Within one machine, `e_hat^(j)` and `t_hat^(j)` are spaced by
$|\hat e^{(j)}| = do \cdot C \cdot B_{\mathsf{loc}}$, a multiple of
$B_{\mathsf{loc}}$, so they share the same in-window shift and carry split — which
is why machine $j$'s `t_hat` reuses machine $j$'s `e_hat` block summaries (the
single-machine `e`/`t` sharing, now per machine). Across machines the shifts may
differ, but each machine covers different blocks (different `c_α` values) and so
needs its own summaries anyway.

What *does* grow with $\mathcal M$ are the cheap high tables: `eq_high_e`,
`eq_high_t`, `eq_high_z` gain a machine axis (size `M·C·do`, `M·n_A·C·do`,
`M·DF·DC`). These are precompute-once field tables, not $\alpha$-evaluations, so
the growth is negligible.

(Because the canonical deployment uses $\mathcal M = 2^N$, $B_{\mathsf{loc}}$ is
always a power of two and the per-machine `e_hat`/`t_hat` peel always applies; a
non-power-of-two machine count would fall back to a dense per-machine evaluation,
exactly as the `z`-tensor does for non-power-of-two `block_len`. See the closing
note.)

## Tensor components

Three row blocks act with separable (tensor-product) structure — the
`e_hat`-consistency component (on the consistency row), the `t_hat`-consistency
component (on the $n_A$ `A`-rows), and the `z_hat` in-block consistency (on the
consistency row). Each now appears $\mathcal M$ times. We work `e_hat` in full and
let `t_hat`/`z_hat` follow.

### e-related tensor components

This is the consistency component $c^{\top} \otimes G$, tying `e_hat` to the
stage-1 challenge combination. It sits on the single consistency row $i$, so
`row_weight[i]` multiplies the *sum* of all $\mathcal M$ machine contributions,
once, at the end.

#### The cell formula (per machine)

Machine $j$'s `e_hat^(j)` cell is indexed by `(dig, claim, block_local)` with
`block_local` $\in [0, B_{\mathsf{loc}})$. Its local column coordinate is

$$
c \;=\; \text{block\_local} \;+\; B_{\mathsf{loc}} \cdot (\text{claim} + C \cdot \text{dig}),
$$

the global column is $e^{(j)}_{\text{start}} + c$, and the entry is

$$
M(i, c) \;=\; c_{\alpha}[\text{claim}, \text{blk}_g] \cdot g_{\text{open}}[\text{dig}],
\qquad
\text{blk}_g := j\,B_{\mathsf{loc}} + \text{block\_local},
$$

where $\text{blk}_g$ is the **global** block index. Three things differ from the
single-machine cell: (i) the block window is $B_{\mathsf{loc}}$, not $B$; (ii) the
consistency challenge $c_{\alpha}$ is read at the global block $\text{blk}_g$, so
the verifier indexes the *same* full $c_{\alpha}$ table at machine $j$'s slice;
(iii) the offset is machine $j$'s $e^{(j)}_{\text{start}}$. As in the
single-machine case, $c_{\alpha}$ **couples claim and block**, so it does not
factor into separate per-claim and per-block scalars.

> **Why the global block, and why $\text{blk}_g = j\,B_{\mathsf{loc}} + \text{block\_local}$.**
> The folding challenges are a *single global vector* $c = (c_1, \dots, c_B)$ that
> the verifier samples **once**, indexed by global block. Machine $P_j$ gets no
> fresh randomness — it uses the slice $c^{(j)} = \{c_i : i \in \mathcal I_j\}$
> (the $(c^{(j)})^{\top}\!\otimes G$ row of $M_j$ in the distributed prover's
> partial root relation). The scalar that multiplies a fold block is the challenge
> *for that block*, and blocks are numbered globally. A column of `e_hat^(j)` is
> addressed by the **local** index $\text{block\_local} \in [0, B_{\mathsf{loc}})$,
> but it represents a **global** block; because $P_j$ owns the contiguous range
> $\mathcal I_j = [jB_{\mathsf{loc}}, (j{+}1)B_{\mathsf{loc}})$, the global index is
> its base offset $jB_{\mathsf{loc}}$ plus the local position:
> $\text{blk}_g = jB_{\mathsf{loc}} + \text{block\_local}$. Writing
> $c_{\alpha}[\text{claim}, \text{block\_local}]$ would be wrong — it would name a
> block on machine $0$. Concretely with $B = 8$, $\mathcal M = 2$
> ($B_{\mathsf{loc}} = 4$): machine $1$'s local block $0$ is global block $4$ and
> carries $c_{\alpha}[\text{claim}, 4]$. Reading $c_{\alpha}$ at the global block
> is also what keeps the block scan equal to the single-machine cost: every global
> block $0..B$ is visited exactly once, only reordered into $\mathcal M$ chunks of
> $B_{\mathsf{loc}}$.

#### The naive evaluation

```text
acc = 0
for j     in 0..M:
  for dig   in 0..do:
    for claim in 0..C:
      for block_local in 0..B_loc:
        blk_g = j*B_loc + block_local
        c     = block_local + B_loc*(claim + C*dig)
        acc += row_weight[i] * c_alpha[claim][blk_g] * g_open[dig]
                   * eq(e_start[j] + c, r_col)
```

The triple-plus-machine loop costs $O(\mathcal M \cdot B_{\mathsf{loc}} \cdot C \cdot \text{do})
= O(B \cdot C \cdot \text{do})$ — the *same total* as the single-machine naive
scan — plus a dense `eq` table.

#### The optimization: per-(machine, claim) block summaries

Peel the $B_{\mathsf{loc}}$ block window exactly as single-machine, now per
machine. Split $e^{(j)}_{\text{start}} = e^{(j)}_{\text{lo}} + B_{\mathsf{loc}}\, e^{(j)}_{\text{hi}}$;
the low part $s = e^{(j)}_{\text{lo}} + \text{block\_local}$ produces at most one
carry, and the equality factors into `eq_low` (the shared low-window table) and
`eq_high_e`. Because $c_{\alpha}$ couples claim and block, build a two-bucket
summary **per (machine, claim)**:

```text
# Per-(machine, claim) block summaries — O(M * C * B_loc) = O(C * B):
for j     in 0..M:
  for claim in 0..C:
    S0[j][claim] = 0      # carry = 0
    S1[j][claim] = 0      # carry = 1
    for block_local in 0..B_loc:
        blk_g = j*B_loc + block_local
        s     = e_lo[j] + block_local       # e_lo[j] = e_start[j] mod B_loc (per machine)
        if s >= B_loc:
            S1[j][claim] += c_alpha[claim][blk_g] * eq_low[s - B_loc]   # eq_low shared (r_col low bits)
        else:
            S0[j][claim] += c_alpha[claim][blk_g] * eq_low[s]
```

The outer combine runs over the small axes, per machine:

```text
# Outer combine — O(M * C * do):
acc = 0
for j     in 0..M:
  for dig   in 0..do:
    for claim in 0..C:
      q = e_hi[j] + claim + C*dig
      acc += g_open[dig] * (S0[j][claim]*eq_high_e[q] + S1[j][claim]*eq_high_e[q+1])
acc *= row_weight[i]
```

#### Cost

The block summaries cost $O(\mathcal M \cdot C \cdot B_{\mathsf{loc}}) = O(C \cdot B)$
— **the same total block scan as the single-machine component**, merely grouped
into $\mathcal M$ buckets. The outer combine costs
$O(\mathcal M \cdot C \cdot \text{do})$, which is $\mathcal M\times$ the
single-machine $O(C \cdot \text{do})$ but on the small digit axis. Total

$$
O\!\big(C \cdot B + \mathcal M \cdot C \cdot \text{do}\big),
$$

versus single-machine $O\!\big(C \cdot (B + \text{do})\big)$: the heavy block term
is unchanged; only the light digit term scales with $\mathcal M$.

### t-related tensor components

The per-`A`-row consistency component $c^{\top} \otimes G_{n_A}$ is the same check
applied once per row of `A`, so it occupies the $n_A$ `A`-rows, each with its own
`row_weight`. Like `e_hat`, it is now $\mathcal M$-fold, and on each machine it is
block-diagonal in the `a_row` axis.

#### The cell formula (per machine)

Machine $j$'s `t_hat^(j)` cell is indexed by `(a_row, dig, claim, block_local)`:

$$
c \;=\; \text{block\_local} \;+\; B_{\mathsf{loc}} \cdot (\text{claim} + C \cdot \text{dig} + C \cdot \text{do} \cdot \text{a\_row}),
$$

with entry, on the matching `a_row` row,

$$
M(\text{a\_row}, c) \;=\; c_{\alpha}[\text{claim}, \text{blk}_g] \cdot g_{\text{open}}[\text{dig}],
\qquad \text{blk}_g := j\,B_{\mathsf{loc}} + \text{block\_local},
$$

weighted by `row_weight[a_row]`. This is the same claim-and-block-coupled
$c_{\alpha}$ as the `e_hat` component, with the extra `a_row` axis tensored on.

#### The optimization: reuse the per-(machine, claim) block summaries

The block-dependent factor $c_{\alpha}[\text{claim}, \text{blk}_g] \cdot
\mathrm{eq}_{\text{low}}(\cdot)$ is **identical** to the `e_hat` component's. As
shown in the layout section, every `e_hat^(j)` and `t_hat^(j)` shares the same
low-window residue, so the carry split matches per machine and the per-(machine,
claim) summaries `S0[j][claim]`, `S1[j][claim]` are **reused unchanged** — this
is the single-machine `e`/`t` sharing, extended across machines. The only new
work is the outer combine, now ranging over `a_row` as well:

```text
# Outer combine — O(M * n_A * C * do):
acc = 0
for j     in 0..M:
  for a_row in 0..n_A:
    for dig   in 0..do:
      for claim in 0..C:
        q = t_hi[j] + claim + C*dig + C*do*a_row
        acc += row_weight[a_row] * g_open[dig]
                 * (S0[j][claim]*eq_high_t[q] + S1[j][claim]*eq_high_t[q+1])
```

#### Cost

The summaries are shared (free here), and the outer combine costs
$O(\mathcal M \cdot n_A \cdot C \cdot \text{do})$. Total

$$
O\!\big(C \cdot B + \mathcal M \cdot n_A \cdot C \cdot \text{do}\big),
$$

versus single-machine $O(C \cdot B + n_A \cdot C \cdot \text{do})$: as with
`e_hat`, the heavy block scan is unchanged and only the cheap `a_row`/digit
combine scales with $\mathcal M$.

### z-related tensor components

The in-block consistency contribution ties `z_hat` to the opening point's
in-block weights, on the consistency row (so it shares that row's `row_weight`,
applied with the relation's minus sign at the end). Here the **replication
asymmetry** bites: each `z_hat^(j)` is a *full-size* copy with the full
`block_len`, not a $1/\mathcal M$ slice, so the $\mathcal M$ repetitions are
$\mathcal M$ full evaluations.

#### The cell formula (per machine)

Machine $j$'s `z_hat^(j)` cell is indexed by `(dc, df, blk)`, `blk` $\in
[0, \text{block\_len})$:

$$
c \;=\; \text{blk} \;+\; \text{block\_len} \cdot (\text{df} + \text{DF} \cdot \text{dc}),
$$

global column $z^{(j)}_{\text{start}} + c$, entry

$$
M(i, c) \;=\; g_{\text{commit}}[\text{dc}] \cdot g_{\text{fold}}[\text{df}] \cdot a[\text{blk}],
$$

contributed with weight $-\,\text{consistency\_weight}$. Crucially the entry **does
not depend on $j$**: the fold rows carry no machine-specific data (the opening
weight $a[\text{blk}]$ and the gadget weights are global). Only the segment offset
$z^{(j)}_{\text{start}}$ differs between machines.

#### Case 1: `block_len` a power of two (root)

Peel the `block_len` window per machine and build a two-bucket in-block summary
per machine (the shared `eq_low_z` table serves all of them):

```text
# Per-machine in-block summaries — O(M * block_len):
for j in 0..M:
    A0[j] = 0      # carry = 0
    A1[j] = 0      # carry = 1
    for blk in 0..block_len:
        s = z_lo[j] + blk                   # z_lo[j] = z_start[j] mod block_len (per machine)
        if s >= block_len:
            A1[j] += a[blk] * eq_low_z[s - block_len]
        else:
            A0[j] += a[blk] * eq_low_z[s]

# Outer combine — O(M * DF * DC):
acc = 0
for j  in 0..M:
  for dc in 0..DC:
    for df in 0..DF:
      q = z_hi[j] + df + DF*dc
      acc += g_commit[dc] * g_fold[df] * (A0[j]*eq_high_z[q] + A1[j]*eq_high_z[q+1])
acc *= -consistency_weight
```

Because $a[\text{blk}]$ is shared, the per-machine summaries differ only by the
offset/carry split; they combine additively into one `acc`.

#### Case 2: `block_len` not a power of two (dense fallback)

Per machine, materialize the structured `z` segment (the tensor product) and run
one offset-equality evaluation over it — $\mathcal M$ times — exactly as the
single-machine fallback, repeated per machine.

#### Cost

| case | overhead |
|---|---|
| `block_len` a power of two (root) | $O(\mathcal M \cdot \text{block\_len} + \mathcal M \cdot \text{DF} \cdot \text{DC})$ |
| `block_len` not a power of two | $O(\mathcal M \cdot \text{DF} \cdot \text{DC} \cdot \text{block\_len})$ |

This is $\mathcal M\times$ the single-machine `z`-tensor — but the `z`-tensor is
already the *cheap* part of the verifier (no $\alpha$-evaluations), so
$\times \mathcal M$ keeps it well below the setup-matrix floor analyzed next.

## Setup components

These row blocks are the rows of the shared SIS commitment matrix, and they
**dominate** verifier time: each entry is a cyclotomic ring element that must be
$\alpha$-evaluated (an $O(D)$ operation) before it can enter the scalar
sum-check. The single-machine game is to perform each $\alpha$-evaluation **once**
and reuse it across every product that needs it. The distributed question is
whether the $\mathcal M$-fold repetition forces more $\alpha$-evaluations. **It
does not** — and showing why is the crux of the equivalence claim.

### The three setup row blocks, now $\mathcal M$-fold

As single-machine, three row blocks are sub-views of the *same* backing SIS
matrix, now each repeated per machine:

- **`D · e_hat`** — $D_j$ on `e_hat^(j)`, for $j \in [\mathcal M]$.
  **Partitioned**: $\bigcup_j D_j = D$ and $\bigcup_j \hat e^{(j)} = \hat e$, so
  the $\mathcal M$ pieces *collectively* are exactly the single-machine
  $D \cdot \hat e$, only re-laid-out by machine.
- **`B · t_hat`** — $B_j$ on `t_hat^(j)`, for $j \in [\mathcal M]$ (per commitment
  group). **Partitioned**: collectively the single-machine $B \cdot \hat t$.
- **`A · G_fold · z_hat`** — the same digit-domain matrix $A$ on each folded
  commit-digit vector $z^{(j)}$, for $j \in [\mathcal M]$, where
  $z^{(j)}[\text{blk}, \text{dc}]
  = \sum_{\text{df}} g_{\text{fold}}[\text{df}]
  \cdot \hat z^{(j)}[\text{blk}, \text{dc}, \text{df}]$.
  **Replicated**: one shared matrix $A$ acts on $\mathcal M$ distinct full-size
  folds.

### Why the $\alpha$-evaluations do not multiply

The shared SIS matrix (backing `A`, `B`, `D`) is regenerated from one public seed
and is identical on every machine. The verifier scans it **once**, exactly as
single-machine, and $\alpha$-evaluates each entry
`r_eval(r,c) = eval_ring_at_pows(matrix[r][c], α)` **once**, because:

- For `D · e_hat` and `B · t_hat`, the partition only *re-routes* which `e`/`t`
  column each scanned matrix column maps to. The set of matrix columns scanned —
  the full $D$, the full $B$ — is unchanged.
- For `A · G_fold · z_hat`, all $\mathcal M$ folded commit-digit vectors
  $z^{(j)}$ are read by the **same** $A$ columns, so the one
  $\alpha$-evaluation of $A(r,c)$ is reused across all $\mathcal M$ copies.

Hence the $\alpha$-evaluation count stays
$O(r_{\max} \cdot n_{\text{cols}} \cdot D)$ with the **same**
$r_{\max} = \max(n_d, n_b, n_A)$ and the **same** $n_{\text{cols}}$ (the SIS
column width is unchanged because `A`, `B`, `D` are the same matrices). The only
thing the verifier must do differently is fold the $\mathcal M$ fold copies into
the column weight *before* the scan.

### The fused per-entry computation

Same single scan as single-machine, with the `A·G_fold·z_hat` column weight
**combined over the $\mathcal M$ folds** in advance:

$$
Z^{\text{comb}}_{\text{col}}[c] \;:=\; \sum_{j=0}^{\mathcal M-1} Z^{(j)}_{\text{col}}[c],
$$

where $Z^{(j)}_{\text{col}}[c]$ is the `z`-equality weight of matrix column $c$
within machine $j$'s `z_hat^(j)` segment (at offset $z^{(j)}_{\text{start}}$). The
matrix column is $c = (\text{blk}, \text{dc})$, and

$$
Z^{(j)}_{\text{col}}[\text{blk}, \text{dc}]
\;=\;
-\sum_{\text{df}} g_{\text{fold}}[\text{df}]
\cdot
\mathrm{eq}\!\left(z^{(j)}_{\text{start}}
  + \text{blk}
  + \text{block\_len}\cdot(\text{df}+\text{DF}\cdot\text{dc}),
  r_{\text{col}}\right).
$$

There is no `g_commit` factor in this setup weight. The `g_commit` factor appears
in the separate opening row over the `z_hat` segment.

The `D·e` and `B·t` patterns ($W_{\text{col}}$, $T_{\text{col}}$) are rebuilt for the
distributed `e`/`t` layout but keep their **single-machine size** — because those
segments are partitioned, not replicated, every matrix column still maps to
exactly one witness column.

```text
# Column patterns (precomputed once, NOT per row):
#   W_col[c]    : matrix col c -> distributed e-layout eq weight    (size unchanged)
#   T_col[g][c] : matrix col c -> distributed t-layout eq weight    (size unchanged)
#   Z_comb[c]   = sum_{j in 0..M} Z_col_machine(j, c)                (build O(M * A_cols))

acc = 0
for r in 0..r_max:                 # r_max = max(n_d, n_b, n_A)        (UNCHANGED)
  for c in 0..n_cols:              # SIS columns scanned once          (UNCHANGED)
    r_eval    = eval_ring_at_pows(matrix[r][c], alpha)   # the ONE O(D) alpha-eval — UNCHANGED count
    setup_index_weight = d_w[r] * W_col[c]                  # D·e   (partitioned, single-size)
                       + sum_g b_w[g][r] * T_col[g][c]      # B·t   (partitioned, single-size)
                       + a_w[r] * Z_comb[c]                 # A·G_fold·z_hat
    acc += r_eval * setup_index_weight
```

The hot per-row loop is **byte-for-byte the single-machine loop** — only the
precomputed weight vector changed (`Z_comb` instead of `Z_col`).

### The three column patterns

- **`W_col[c]` (`D · e_hat`).** Decode $c$ to `(dig, blk_g, claim)` as
  single-machine, then translate to the distributed `e_hat` address: the machine
  is `mac = blk_g / B_loc`, the in-machine block is
  `block_local = blk_g mod B_loc`, and the column lands in machine `mac`'s piece
  via the peeled `eq_low`/`eq_high_e` product. $O(1)$ per cell, same total column
  count.
- **`T_col[g][c]` (`B · t_hat`).** The same with the extra `a_row` axis and
  per-group sparsity, against the distributed `t_hat` layout; reuses
  `eq_high_t`.
- **`Z_comb[c]` (`A · G_fold · z_hat`).** For each `A` column
  $c = (\text{blk}, \text{dc})$, sum over machines:
  $Z^{\text{comb}}_{\text{col}}[c]
  = \sum_j Z^{(j)}_{\text{col}}[c]$.
  Each $Z^{(j)}_{\text{col}}[c]$ is the `G_fold` weighted equality sum over the
  `df` axis of machine $j$'s `z_hat` segment. Build cost is
  $O(\mathcal M \cdot A_{\text{cols}})$; in the non-power-of-two `block_len`
  case it is built densely per machine, as there.

### Two coordinate systems (plus the machine axis)

`matrix_evaluation` reconciles **M-layout** (block innermost, defines the MLE)
with **D-physical layout** (digit innermost, the prover's commit order) by walking
$c$ in D-physical order and baking the M-layout translation into the column
patterns. The distributed layout adds exactly one axis to that translation: the
**machine index sits in the high bits of the block coordinate**
(`blk_g = mac · B_loc + block_local`). The column-pattern builds absorb it; the
hot per-row loop never sees layouts or machines — it is still a dot product of
$\alpha$-evaluated entries against a precomputed weight vector.

### Cost

| step | cost | done once per |
|---|---|---|
| `eq_high_e`, `eq_high_t`, `eq_high_z` tables | $O\big(\mathcal M\,(C\,\text{do} + C\,\text{do}\,n_A + \text{DF}\,\text{DC})\big)$ | verifier |
| `W_col`, `T_col` builds | $O(n_{\text{cols}} \cdot (1 + G))$ | verifier |
| `Z_comb` build | $O(\mathcal M \cdot A_{\text{cols}})$ | verifier |
| $\alpha$-evaluations $r_{\text{eval}}(r, c)$ | $O(n_{\text{cols}} \cdot D)$ ring ops | per row |
| per-row inner product | $O(n_{\text{cols}} \cdot (1 + G))$ | per row |
| **dominant** | $O(r_{\max} \cdot n_{\text{cols}} \cdot D)$ $\alpha$-evaluations — **UNCHANGED** | total |

The bottom row is the whole point: the $O(D)$ $\alpha$-evaluations — the
verifier's true bottleneck — are performed exactly once per shared-matrix entry
and amortized across `D·e_hat`, `B·t_hat`, **and all $\mathcal M$ copies of
`A·G_fold·z_hat`**. The only $\mathcal M$-dependent additions are the
$O(\mathcal M \cdot A_{\text{cols}})$ combined-weight build and the
$\mathcal M\times$ scaling of the cheap high tables — both far below the
$\alpha$-evaluation floor.

## Cost summary: component by component

Putting the components side by side against the single-machine verifier
($B_{\mathsf{loc}} = B/\mathcal M$, $A_{\text{cols}} = $ width of `A` = the `z`
ambient size):

| component | single-machine | distributed (verifier total over all $\mathcal M$) | delta |
|---|---|---|---|
| `e_hat` consistency | $O(C(B + \text{do}))$ | $O(\mathcal M\,C\,B_{\mathsf{loc}} + \mathcal M\,C\,\text{do}) = O(C\,B + \mathcal M\,C\,\text{do})$ | heavy term $\mathcal M\,C\,B_{\mathsf{loc}} = C\,B$ unchanged; light $\times \mathcal M$ |
| `t_hat` consistency | $O(C\,B + n_A C\,\text{do})$ | $O(\mathcal M\,C\,B_{\mathsf{loc}} + \mathcal M\,n_A C\,\text{do}) = O(C\,B + \mathcal M\,n_A C\,\text{do})$ | heavy term $= C\,B$ unchanged; light $\times \mathcal M$ |
| `z_hat` tensor (replicated) | $O(\text{block\_len} + \text{DF}\,\text{DC})$ | $O(\mathcal M\,\text{block\_len} + \mathcal M\,\text{DF}\,\text{DC})$ | cheap term $\times \mathcal M$ (no tiling) |
| `D·e_hat`, `B·t_hat` setup scan | $\alpha$-evals over full $\hat e$, $\hat t$ | identical | none |
| **`A·G_fold·z_hat` setup scan (dominant)** | $O(r_{\max}\,n_{\text{cols}}\,D)$ $\alpha$-evals | **$O(r_{\max}\,n_{\text{cols}}\,D)$ $\alpha$-evals** | **none** |
| combined column-weight build | — | $O(\mathcal M \cdot A_{\text{cols}})$ field ops | additive, cheap |
| `r_tail` rows | one quotient | one summed quotient | none |

The distributed column is the verifier's **total** work (it is one entity that
sums over all $\mathcal M$ machines), not the per-machine work. For the
*partitioned* `e_hat`/`t_hat` block scan the per-machine cost is
$C\,B_{\mathsf{loc}}$, but the $\mathcal M$ pieces *tile* the same $B$ blocks, so
the total is $\mathcal M \cdot C\,B_{\mathsf{loc}} = C\,B$ — written with $B$, not
$B_{\mathsf{loc}}$, precisely because it does not shrink per machine and does not
grow with $\mathcal M$. The *replicated* `z_hat` is the opposite: each of the
$\mathcal M$ copies carries the full `block_len`, the pieces duplicate rather than
tile, so the $\mathcal M$ survives in $\mathcal M\,\text{block\_len}$.

The dominant row — the $O(D)$ shared-matrix $\alpha$-evaluation scan — is
**unchanged**, because `A`/`B`/`D` are shared and seed-regenerated, and the
$\mathcal M$ fold copies enter only through the additively combined,
$\alpha$-free $Z^{\text{comb}}_{\text{col}}$. Every $\mathcal M$-dependent term is
field-arithmetic bookkeeping bounded by $O(\mathcal M \cdot n_{\text{cols}})$,
negligible whenever $\mathcal M \ll r_{\max} \cdot D$ (e.g. $r_{\max} \approx 9$,
$D \ge 64$ leaves headroom for hundreds of machines).

### The one additive cost: a few sum-check rounds

Because `z_hat` is replicated $\mathcal M$ times, the next-level witness grows, so
its variable count rises by at most $\log_2 \mathcal M = N$ (less when `t_hat`
dominates the width, as is typical). The verifier therefore reads up to $N$ extra
sum-check round polynomials — a handful of field operations and a few hundred
proof bytes each. This is additive, logarithmic in the machine count, and dwarfed
by the unchanged $\alpha$-evaluation scan. Challenges, the batched target
$V_\alpha$ (computed from the summed public right-hand side), the recursive
commitment, and the matrix-evaluation bottleneck are all flat in $\mathcal M$.

This is the precise sense in which the distributed-relation verifier has
**equivalent performance** to today's verifier: identical in its dominant term,
with overhead that is logarithmic in the machine count plus a negligible
field-arithmetic pre-pass.

## A note on partition shape (power-of-two machines)

> **Status: design constraint, not yet implemented.** Recorded so the
> peelability reasoning does not have to be re-derived.

The per-machine fast path requires the block window
$B_{\mathsf{loc}} = B/\mathcal M$ to be a power of two, so that `block_local`
occupies a contiguous low-bit window and the two-bucket carry split is defined
(the same condition `block_len` must satisfy in the `z`-tensor Case 1). This holds
**iff $\mathcal M$ is a power of two dividing $B$** — which is exactly the
canonical choice $\mathcal M = 2^N$ used by the distributed prover: since
$B = 2^r$ is itself a power of two, any $\mathcal M = 2^N \le 2^r$ works and gives
equal chunks $|\mathcal I_j| = B_{\mathsf{loc}}$.

For other machine counts (not a power of two, or $\mathcal M \nmid B$) the
contiguous partition still makes sense but $B_{\mathsf{loc}}$ is no longer a clean
low-bit window, so the per-machine `e_hat`/`t_hat` contribution would use a dense
per-machine evaluation — the exact analogue of the `z`-tensor dense fallback (Case
2), at cost proportional to the machine's segment size. The combined work is still
$O(C \cdot B)$ for the block scan, so this affects only the constant, not the
dominant $\alpha$-evaluation cost. The recommended deployment keeps $\mathcal M$ a
power of two so every segment lands on the peeled fast path.

## Relationship to setup-claim offloading

The dominant cost analyzed above — the shared SIS-matrix $\alpha$-evaluation scan
— is exactly the cost that [verifier offloading](../../roadmap/verifier-offloading.md)
removes from the verifier's local work by delegating the setup contribution to a
product sum-check against a preprocessed commitment. The two techniques are
**orthogonal and compose**: the distributed prover keeps the scan *equivalent* to
today's single-machine scan, and offloading (once landed) removes that scan for
the single-machine and distributed-relation verifier alike. Nothing in the
distributed construction changes the offloading interface, because the setup
matrices are shared and the verifier's setup contribution is the same
shared-matrix evaluation in both cases.

## Verifier no-panic obligations

The distributed layout adds public structure the verifier must validate at the
boundary, consistent with the [verifier no-panic contract](../verification.md).
The machine count $\mathcal M$ and the block partition $\{\mathcal I_j\}$ come from
the public schedule/layout, not from prover-controlled proof bytes, so the segment
offsets $z^{(j)}_{\text{start}}$ and the loop bounds are fixed before any proof
data is read. The verifier must reject a layout whose $\mathcal M$ replicated
`z_hat` copies do not fit the validated next-level witness capacity, and confirm
the partition is the canonical contiguous one (so each $c^{(j)}$ is a well-defined
restriction of the global $c$ and the offsets are exact). After those boundary
checks the hot path above performs no $\mathcal M$-dependent allocation or
indexing that has not already been bounded; malformed input is rejected with
`AkitaError` / `SerializationError`, never by panicking.
