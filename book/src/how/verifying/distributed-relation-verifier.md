# The distributed relation verifier

This chapter explains what one verifier does differently when the prover
splits the witness into several chunks. Read these pages first:

- [Matrix evaluation at a point](./matrix_evaluation.md) defines the row
  families, the witness order, and the compact evaluator.
- [Subring-coefficient-packing consistency](../proving/akita-fold.md#subring-coefficient-packing-consistency)
  defines the packed consistency relation, which every chunked fold uses.
- [Method-dependent structured terms](./stage2.md#method-dependent-structured-terms)
  says how the verifier evaluates that relation.

This chapter collects what depends on the chunk axis in those pages.

> **A note on the name.** "Distributed verifier" would be a misnomer: nothing
> about the verification is distributed. There is a *single* verifier, running on
> one machine, doing the ordinary amount of work. What can be distributed is the
> **prover**: each worker holds only a slice of the witness (see
> [the distributed prover](../proving/distributed-prover.md)). This chapter is
> about that single verifier checking the *relation that a chunked prover
> produces*. Hence "distributed-relation verifier", not "distributed verifier".

## What chunking changes

A fold with $C$ witness chunks proves the same relation rows as a fold with
one chunk. [Multiple witness chunks](../proving/advanced-relation-layouts.md#multiple-witness-chunks)
derives this. The verifier still evaluates the relation matrix at one point,

$$
\widetilde M(\tau_1,x)
=
\sum_{i,k}\operatorname{eq}(\tau_1,i)\operatorname{eq}(x,k)M_{i,k},
$$

where $i$ is a relation row and $k$ is a flat witness coefficient.

| Axis | Effect of chunking |
|---|---|
| Rows $i$ | None. No row family is duplicated, and the public right-hand side is unchanged. |
| Row weights $\operatorname{eq}(\tau_1,i)$ | None. Each row weight is applied once to the sum over all chunks. |
| Columns $k$ | The ordinary witness body is laid out once per chunk. |

So every term of $\widetilde M$ keeps its row factor, and its column factor
becomes a sum over chunk units.

In a chunked fold the relation value has two parts. The relation-matrix
evaluator computes the commitment rows and the quotient rows. The packing
terms cover the fold-evaluation consistency row. [Which modes a chunked fold
uses](#which-modes-a-chunked-fold-uses) explains this split.

## Two kinds of segment

Each chunk owns a contiguous range of live blocks. The three ordinary witness
segments follow two different rules.

- **Partitioned segments** (`E` and `T`). A chunk stores the opening digits
  and the inner-commitment digits of the blocks that it owns. Over all chunks,
  these pieces tile the single-chunk segment. Their total size does not grow.
- **Replicated segments** (`Z`). Chunk $j$ stores the digits of its own
  partial response $\mathbf z^{(j)}$, the fold of its own blocks. A partial
  response is a sum over some blocks, so it has the full response width. $C$
  chunks store $C$ full-width `Z` segments. A chunk with an empty block range
  still stores one, and the honest prover writes zero there.

Replication is the price of the design. The prover never has to add the
partial responses of different chunks before it writes the next witness.

## The chunked witness layout

The canonical walk of
[Matrix evaluation at a point](./matrix_evaluation.md#canonical-walk) has this
form for a multi-chunk fold:

```text
chunk 0:  group in relation order: [Z | E | T]   then alignment gap
chunk 1:  group in relation order: [Z | E | T]   then alignment gap
...
shared ordinary quotient rows
compression digits, alignment ranges, and compression quotient rows
```

The chunk index is the outer key, and the commitment group is the inner key.
One `[Z | E | T]` unit exists for every chunk and group pair. After each chunk
of a multi-chunk fold, the layout pads the body to a multiple of the next
fold's source-block coefficient width. Each chunk body then ends on a block
boundary of the next fold, so no source block of the next fold belongs to two
chunks. These alignment gaps are committed coordinates with no relation
equation. The quotient rows and the compression
suffix appear once, after all chunks. They are not replicated.

Every unit records its exact global block range. Inside a unit, the block
index restarts at zero, but each fold challenge belongs to a global block: the
transcript samples one challenge per live block of the group, not one family
per chunk. The verifier therefore reads the group's challenge list at
`global_block_start + local_block`.

Inside a unit, the semantic value indices are:

| Segment | Semantic value index | Then |
|---|---|---|
| `E` | `claim * B_j + local_block` | `[role subcolumn][role digit][native coefficient]` |
| `T` | `n_a * (claim * B_j + local_block) + a_row` | `[role subcolumn][role digit][native coefficient]` |
| `Z` | `(position * depth_witness + witness_digit) * depth_fold + fold_digit` | `[coefficient]` |

Here `B_j` is the number of live blocks that the unit owns. The `Z` index has
no block coordinate and no `B_j`, because the response has already summed
over the blocks. This is the layout of
[Matrix evaluation at a point](./matrix_evaluation.md#setup-roles-and-mixed-rings);
only `B_j` and the unit's start address depend on the chunk.

## Block ranges

The verifier does not derive chunk ranges from the chunk count alone. It reads
them from the schedule through `WitnessLayout`.

- Every source group except an inherited recursive witness uses the
  proportional partition. This covers root sources and precommitted groups.
  Chunk $j$ of $C$ owns
  $[\lfloor jB/C\rfloor,\lfloor (j+1)B/C\rfloor)$ for $B$ live blocks.
- A recursive witness inherits its ranges. Each chunk body that the previous
  fold produced becomes one owner of the source blocks at this fold. If this
  fold uses fewer chunks, adjacent owners merge. The last owner also receives
  the shared tail. Inherited ranges can differ by more than one block.

[Chunks and fold challenges](../proving/opening-points-layout.md#chunks-and-fold-challenges)
states the complete rule. In both cases a range can be empty. The verifier
never pads a short range, and it never substitutes a power-of-two block domain
for the exact live block count.

## Which modes a chunked fold uses

Schedule validation fixes two choices for every multi-chunk fold.

1. The opening method is subring coefficient packing. A multi-chunk fold that
   selects evaluation trace is rejected.
2. The relation mode is quotient lifting. Reduced evaluation is admitted only
   for single-chunk folds.

The first choice moves the fold-evaluation consistency row out of the
relation-matrix evaluator. For a packing group, that evaluator contributes no
`Z` term and no quotient term on the consistency row. The packed `E` and
quotient weights, the packing-`Z` term, and the direct opening term are
evaluated from compact packing factors, as described in
[Method-dependent structured terms](./stage2.md#method-dependent-structured-terms).
Those factors read the same chunk units.

The second choice means that the shared quotient rows are always present in a
chunked witness. Both setup contribution modes remain available.

## Term by term

A setup term pairs setup columns with witness addresses. An **address tensor**
is a compact description of such a set of pairs: one start column, one start
address, and a list of axes. Each axis has a length, a stride on the setup
side, and a stride on the witness side. The verifier contracts an address
tensor against the equality point without listing the pairs.

The table lists each part of the relation value for a chunked fold.

| Term | Rows | What the chunk axis adds |
|---|---|---|
| Setup `D * E` | D | One address tensor per nonempty unit and claim. The setup column of a block follows its global block index, so every setup column maps to one witness address. |
| Setup `B * T` | B | The same, with the `a_row` axis. |
| Setup `A * Z` | A | One address tensor per unit, including empty units. All of them start at setup column zero. |
| Structured `T` challenge term | A | A sum over the nonempty units; each unit uses the challenges of its own global blocks. |
| Packing terms | consistency | Evaluated from the compact packing factors over the same units. |
| Quotient rows and compression | all | Nothing. They are shared once. |

The `A * Z` row is the one place where replication is visible. All $C$
replicated `Z` segments are read by the same columns of the same matrix
$\mathbf A$. The verifier prepares one `Z` column-weight vector per group. The
address tensors of all units add into it, so the weight of one setup column is
the sum of the equality weights of the $C$ witness addresses that the column
reads. The setup scan then visits each $\mathbf A$ coefficient once, exactly as
in a single-chunk fold.

For `D * E` and `B * T` the pieces tile the same columns, so the scan also
visits each coefficient once, and the column weights have the single-chunk
size.

## Cost

The dominant verifier work is reading the public setup coefficients. Chunking
does not add a second scan, and it does not add setup columns. The work that
grows with the chunk count is smaller:

- preparing the address tensors, which grows with the number of units;
- the Stage 2 sumcheck, whose length follows the witness length. The witness
  gains $C-1$ extra `Z` segments and the alignment gaps.

One indirect cost comes from security, not from evaluation. The shared A rows
bind the sum of the partial responses, while the range check bounds each
partial response separately. The A-role collision target is therefore
multiplied by $C$, and the schedule must select an A-role SIS row that covers
it. [Security](../security.md) gives the formula. A chunked schedule can
therefore need a stronger A-role SIS row than a single-chunk schedule. The
chunk count is a schedule parameter for this reason.

The verifier never materializes a dense witness-sized weight table for these
terms. Address tensors describe repeated affine address axes, and the
equality contraction over them is logarithmic in the axis length.

## Bounds the verifier enforces

All chunk geometry is public and comes from the schedule, not from proof
bytes. The verifier fixes it before it reads proof data.

- The chunk count is a power of two and at most `MAX_WITNESS_CHUNKS = 64`.
  The shipped presets use 2, 4, or 8 chunks on the first 1 to 3 fold levels.
- A recursive fold's inherited ranges must equal the ranges derived from its
  producer's layout. Otherwise schedule resolution rejects the schedule.
- The units must follow chunk-major relation order, and each address must be
  aligned to the relation coefficient block.
- All counts, strides, and address products use checked arithmetic.

Malformed geometry returns `AkitaError`, in line with the
[verifier no-panic contract](../verification.md).

## Relationship to setup contribution modes

A schedule may check the setup contribution directly in Stage 2 or defer it to
Stage 3, as described in
[Setup contribution and Stage 3](./setup_contribution.md). Both modes use the
same `SetupContributionPlan` and the same chunk address tensors. Chunking does
not change that interface.

## Implementation map

- `crates/akita-params/src/witness.rs` owns chunk, group, and segment
  addresses.
- `crates/akita-params/src/witness/chunk_partition.rs` owns the proportional
  ranges and the inherited ranges.
- `crates/akita-params/src/schedule.rs` and
  `crates/akita-params/src/ring_relation_mode.rs` restrict the opening method
  and relation mode of a multi-chunk fold.
- `crates/akita-types/src/setup_contribution/` builds the per-unit address
  tensors for the A, B, and D roles.
- `crates/akita-verifier/src/setup_contribution/` runs the setup scan and the
  structured challenge terms.
- `crates/akita-verifier/src/relation/evaluation.rs` combines the structured,
  setup, and quotient contributions.
- `crates/akita-verifier/src/coefficient_packing_relation/` evaluates the
  packing terms.
- `crates/akita-algebra/src/offset_eq/tensor_pair/` owns the paired equality
  contraction and its dense test oracle.
