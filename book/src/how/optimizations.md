# Optimizations

Akita keeps protocol decisions separate from compute decisions. The protocol
chooses the commitment layout, transcript order, and proof shape. A compute
backend receives a validated source view and performs one named operation.
This split lets the CPU implementation change storage and traversal without
changing proof bytes or exposing CPU plans as public API.

The current CPU design has three important rules:

1. The source owns only its logical witness.
2. Derived data belongs to the operation that uses it.
3. Prepared setup state owns only data that is useful across operations.

These rules are most visible in one hot commitment and large ring switch
operations.

Most of this chapter describes the prover. The section
[Paired equality tensors and the carry recurrence](#paired-equality-tensors-and-the-carry-recurrence)
describes the verifier's evaluation kernels.

## One hot source geometry

A one hot polynomial stores an optional hot index `i_j` for each chunk `j`.
The chunk arity is `K`. A present index identifies the global nonzero field
coordinate

```text
g_j = j K + i_j.
```

For ring dimension `D`, this coordinate maps to a ring and coefficient by

```text
ring_j  = floor(g_j / D),
coeff_j = g_j mod D.
```

If one commitment block contains `P` ring positions, the same coordinate maps
to

```text
block_j    = floor(ring_j / P),
position_j = ring_j mod P.
```

This is the only coordinate rule. Single chunk and multi chunk layouts do not
need different stored block types. `OneHotPoly` stores the indices and scalar
shape data. Its D-free commitment representation carries those semantics into
request compilation; `OneHotView` remains the typed opening view.

## One group commitment operation

The protocol always commits a group. A singleton is a group with one source.
Each source first advertises its representations without materializing them;
request compilation intersects those choices with the registered operation's
capabilities. `CommitmentExecutor` then materializes only the selected path and
runs either independent inner/outer operations or one explicit fused operation.
Dense, one hot, and recursive witness sources implement the same D-free
`CommitmentSource` boundary without sharing a CPU representation. Commitment
state remains policy-selected: a portable policy exports the existing hint,
while a resident policy keeps backend state in process.

The CPU one hot path runs this flow:

```text
validated OneHotRepresentation values
        |
        v
derive the active block interval
        |
        v
materialize one operation tile as flat sparse entries
        |
        v
run the selected matrix sweep
        |
        v
place rows in source order and drop the tile
```

The driver partitions the concatenated block range once. Each tile builds only
the entries that overlap that tile. The output rows remain live because they
are the result of the operation. The sparse entries and sweep scratch are
dropped at tile completion.

This ownership also applies to opening and recursive-suffix tensor projection.
Opening builds only the active challenge range. Batched opening keeps only the
current polynomial and its current block tile. Suffix tensor projection
consumes the recursive witness source directly. Cloning a one hot polynomial
does not share mutable derived state.

## Dense sources and shared storage

A dense polynomial stores every coefficient. The optimizations in this section
reduce what the CPU backend stores and copies for dense sources. None of them
changes proof bytes.

### Compact dense prefixes

A polynomial has a logical domain of $2^{n}$ coefficients, but its tail may be
all zero. `DensePoly::from_field_prefix` stores only a prefix and treats the
rest as an implicit zero suffix. The logical polynomial and the commitment
geometry keep the full domain. Storage is rounded up only to the largest
supported commitment ring dimension, so memory and kernel work follow the
stored prefix. The type is in
`crates/akita-cpu-backend/src/sources/dense/poly.rs`, and the test
`zero_suffix_preserves_full_shape_and_fold_values` checks that the compact
source gives the same shape and fold values as the full one.

### Shared setup-prefix coefficients

[Setup offloading](./setup-offloading.md) commits to a prefix of the public
setup. That prefix is already in memory as part of the expanded setup. A
setup-prefix polynomial therefore holds a reference to the backend's single
setup and a length, and it copies no coefficient. Ordinary polynomials still
own their coefficients.

The inner rows of such commitments repeat across prefix sizes. `InnerRowPool`
in `crates/akita-cpu-backend/src/commitment/shared_inner.rs` stores each
distinct row once. A hash only selects candidates; two rows share storage only
when every coefficient is equal. The tests
`setup_prefix_sources_share_the_setup_and_survive_owner_drop` and
`overlapping_rows_share_storage_and_keep_their_order` cover both parts.

### One digit-block interface

Commitment, folding, and coefficient packing all read a source as blocks of
digit planes. Sources store those digits in different ways: cached digits,
ring values that are decomposed on demand, and packed digits. The internal
`BlockDigits` trait in `crates/akita-cpu-backend/src/sources/block_digits.rs`
gives every storage form the same interface. One driver then serves all of
them, and it can choose the exact 16-bit or the batched 8-bit transform from
the digit bound that the source reports. The trait is private to the CPU
backend.

### Lazy challenge embeddings

Under subring coefficient packing, the fold challenges are drawn in a small
challenge subring and then embedded into the A ring. The embedding keeps every
coefficient and moves position $j$ to position $\text{stride}\cdot j$.
`Challenges::embed_subring_positions` in
`crates/akita-challenges/src/challenges.rs` does not rewrite the positions. It
shares the sampled storage and records the stride. The embedded positions are
built only when a caller first reads them as a slice. The test
`checked_embeddings_are_lazy_composable_and_preserve_public_values` checks
that the lazy form gives the same values.

## Streaming suffix tensor inputs

When a recursive evaluation-trace fold needs extension-opening reduction,
column partials consume the compact witness digits through a row iterator.
Each digit is converted to a base-field value when read, and missing padded
entries yield zero. This avoids allocating a complete padded base-field table.
Packing also reads the compact source directly, but produces a full
extension-field table. The reduction retains that packed table and its
sumcheck state, so this input streaming does not give the whole prover constant
memory usage.

The [small-space EOR roadmap](../roadmap/roadmap.md#small-space-extension-opening-prover)
describes a separate, not-yet-implemented approach to reducing those retained
tables.

The implementation and its comparison with a padded-table reference are in
`crates/akita-cpu-backend/src/opaque/recursive/witness/tensor.rs`.

## Tiling and sweep selection

Tile size and arithmetic traversal solve different problems.

The tile size bounds temporary memory. The CPU kernel computes its scratch
budget internally as the larger of a private 8 MiB batching target and the
estimated minimum needed for one block. The target lets small blocks share a
matrix pass; using only the minimum would force one block per tile.
The estimate includes sparse entries, sweep indexes, wide accumulators,
reduced rows, and small offset arrays. In simplified form,

```text
tile = floor((budget - fixed scratch) / scratch per block).
```

The result is capped by the number of live blocks and by the packed local
block index range. All size arithmetic is checked.

The sweep selects how a materialized tile reads the public matrix. The
retained CPU choices are:

* Bucketed sweep. Entries are grouped by active matrix column, then every
  matrix row is scanned once.
* Merge sweep. Sorted block cursors are advanced while a bounded group of
  active columns is loaded once.

Both sweeps load a matrix entry `a` as `NegacyclicShiftWindows`: the canonical
16-bit lanes of `[-a, a]`, so coefficient `j` of `a * X^k` is entry
`D + j - k`. When a ring holds several one-hot rows (`K < D`), one block adds
several shifts of the same entry. The windows read each shift as one
contiguous window and add up to eight windows per pass over the wide
destination, instead of updating the whole destination once per shift. The
kernel runs over the flat 16-bit and 32-bit lane slices that `jolt-field`
exposes, so each vector instruction widens and adds whole lanes rather than
shuffling per-coefficient lane groups. Each shift adds a value
below `2^16` to each signed 32-bit lane, so the field's commit accumulation
budget bounds the shifts between reductions. On x86-64 the kernel dispatches
at runtime to an AVX2 build of the same loop.

Both sweeps consume the same flat sparse entries and produce the same rows.
The private selector uses total block count, active column count, and worker
count. Correctness tests compare both retained sweeps with a direct arithmetic
reference. A release benchmark measures the retained choices over the
production region. The profile report records the selected sweep and tile so
a route change is visible even when total runtime is noisy.

## CPU resource limits

`CpuBackend::with_ring_switch_cache_limit(expanded,
max_cached_ring_switch_elements)` sets the largest ring switch operation that
keeps a complete transformed matrix prefix. `CpuBackend::new` uses `2^21` ring
elements. A zero limit streams every ring switch operation that has a streamed
implementation. `usize::MAX` retains every supported operation.

Commitment scratch sizing and sweep selection are internal kernel decisions,
as described above. Scratch is per worker and excludes the prover's other
resident data, so automatic sizing does not impose a total memory limit.

These policies choose equivalent CPU execution paths. They do not change the
proof schedule, transcript, setup bytes, proof bytes, or verifier behavior.

## Wide accumulation

Commitment kernels accumulate several products before reducing to the base
field. The safe number of additions depends on the field and CRT profile.
`F::MAX_COMMIT_ACCUMULATIONS` is the single contract for this limit.

If a one-hot block contains more terms than the limit, both retained sweeps
reduce the wide accumulator at the same cap and continue from zero. Ring shift
helpers preserve negacyclic wrap. Tests reach the exact addition cap with
maximal canonical limbs and cover wrapped shifts.

## Compact digit range proving

Stage 1 proves the range protocol described in [Sumcheck
stages](./proving/sumcheck-stages.md#stage-1-digit-range-check), but the prover
does not begin by expanding every digit into a field-valued range-image table.
It keeps the balanced digits in packed signed storage and exploits the
involution

$$
w\longmapsto -w-1.
$$

The two digits in each pair have the same range image:

$$
w(w+1)=(-w-1)(-w).
$$

The implementation therefore replaces a balanced digit by its small collision
class

$$
c(w)=
\begin{cases}
w, & w\ge 0,\\
-w-1, & w<0,
\end{cases}
$$

and recovers the range image as `c(w)(c(w)+1)`. Padding uses class zero. A
`CompactDigitSource` stores the packed digits, the exact live-prefix geometry,
and, when a class-indexed stage needs them, compact indices for consecutive
ordered class pairs.

For bases 16, 32, and 64, the product substages compute their first two rounds
from small tables indexed by these class pairs. After the first challenge, the
prover either uses a quartet coefficient table or rescans factorized folded
pairs to prepare the second-round polynomial. Only after the second challenge
does it materialize the exact folded table, now one quarter of the original
Boolean domain. The final range leaf uses the same class-indexed machinery; the
basis-16 leaf has the dedicated two-round quartet path.

On the coefficient-bound route, the direct basis-4 and basis-8 leaves keep the
digits packed for four rounds. The first three rounds bind the position of an
entry inside an aligned octet of the flat table, so until the fourth round every
folded entry depends only on the classes of its octet's eight entries. Before the
first round the prover scans the packed digits once, reading each octet's class
directly from its packed bytes, and adds the fourth round's equality weight of
each live pair of adjacent octets to two tables indexed by octet class: one for
the pair's even octet and one for its odd octet. Each table has at most `2^16`
entries. Blending the two tables with the fourth round's equality coordinate
gives the third round's weight of each octet class. The quad-class marginals of
that blend give a bivariate prefix that reconstructs the first two sumcheck
messages, and the third message is a sum over octet classes.

The fourth message pairs adjacent octets. Its linear sum, the weighted
difference of the two range values across each pair, separates over the two
octet classes, so it is a sum over octet classes of the range value times the
difference of the two tables. The higher coefficients read, per octet class, the
folded value and two Taylor factors of the range polynomial at that value. The
equality weight multiplies the squared difference once, so a basis-8 pair costs
one squaring, two multiplications, and three unreduced multiplications, and a
basis-4 pair one squaring and one unreduced multiplication. The prover
materializes the range image only after the fourth challenge, at one sixteenth
of its original length, and computes the fifth message in the same pass. A Euclidean fold instead uses
the class-indexed leaf described below because the norm term shares its rounds.

These are prover-only representations. The transcript contains the same stage
claims and sumcheck polynomials defined by the protocol, and the verifier does
not need to know which compact kernel produced them.

## Physical Euclidean norm proving

When the schedule selects the Euclidean security route, the prover reconstructs
the complete physical response from its packed digits according to
`PhysicalResponsePlan`. This is an explicit materialization: the direct route
builds one centered-integer response table, while the small-field route builds
the signed limb tables needed for its Gram decomposition. The implementation
then converts these exact prefixes to field tables for sumcheck.

The prover does not run a separate sumcheck over those tables. It batches the
physical norm identity into the final digit-range leaf and uses the same
challenge vector:

- The direct route contributes the degree-two polynomial for
  `sum_x z(x)^2`.
- The limb-Gram route contributes selected degree-three products of signed
  limb pairs. Public block-and-pair subclaims are combined with a transcript
  challenge, and checked integer arithmetic reconstructs the complete squared
  norm outside the field.

The latter route is required when the whole squared norm need not fit below the
base-field modulus. Each individual Gram subclaim has a unique centered lift,
so the verifier can recover the integer norm without accepting a value that
wrapped in the field.

At the final point, the proof exposes one virtual response evaluation for the
direct route or one evaluation per limb for the Gram route. Stage 2 ties those
values to the balanced digit planes of the committed response. The fused Stage
1 proof saves a second round sequence, but it does not pretend that the
physical response tables are free: their materialization is part of the
Euclidean prover's memory cost.

Relevant sources:

- `crates/akita-cpu-backend/src/sources/packed_digits/` owns compact signed-digit
  storage.
- `crates/akita-cpu-backend/src/opaque/sumcheck/digit_range/` owns the direct and
  class-indexed range provers.
- `crates/akita-cpu-backend/src/opaque/sumcheck/physical_l2_norm.rs` fuses the
  physical norm with the final range leaf.
- `crates/akita-params/src/sis/physical_l2.rs` defines the direct and limb-Gram
  plans and reconstructs the integer norm.

## Prepared NTT state

The flat public matrix is stable setup data. A transformed matrix prefix is a
backend optimization. `CpuPreparedSetup` owns transformed prefixes keyed by
ring dimension, transform domain, and exact ring extent.

`NttExecutionRequirements` describes the matrix operations in one proof. It
does not decide which transformed prefixes remain resident. The routed backend
applies one retention rule during planning, prewarming, memory reporting, and
execution:

```text
execution requirements
        |
        +--> retained operation: prewarm or build an exact cache slot
        |
        +--> streamed operation: transform one matrix chunk at a time
```

Large ring relation and quotient operations stream transformed chunks. Their
complete transformed inputs never become prepared cache slots. Smaller reused
operations retain exact prefixes across proofs.

The requirement record retains each operation's routing extent until the
backend makes this decision. This prevents a large streamed operation from
hiding a smaller cached operation on the same matrix route. Requests from one
fused operation share one routing extent across transform domains. Only
retained requests are max-joined into physical cache slots.

Shared prepared caches may remain available across proofs. Applications can
call `backend.trim_caches()` to release built shared-matrix transform slots.

On CPU, release removes built shared-matrix transform slots. Compression
transforms remain resident. Active readers keep released slots alive through
shared references. The reported byte count describes removed cache entries;
their storage can remain allocated until those readers finish. A later request
builds its exact extent instead of reusing a removed covering slot.

Trim when releasing the shared cache will not remove warm state needed by
concurrent work. Trimming does not guarantee an immediate drop in process
memory, and proof completion does not clear the shared cache. The
[setup runtime guide](../usage/setup-runtime.md#reuse-and-release-cpu-caches)
describes this application choice.

Inside `akita-cpu-backend`, `CpuPreparedSetup` counts the bytes of each
namespace separately. These counters are internal to the crate and are not
part of the public API. Planned requirement metrics describe only shared matrix
work because compression entries are created by compression operations.

## Ring relation and quotient streaming

Cached and streamed ring switch execution use the same validated geometry.
The private quotient plan owns:

* active digit and opening role lengths;
* checked matrix extents;
* centered digit bounds;
* CRT safe chunk ranges;
* cached and streamed input shape checks.

The field width dispatch chooses only the concrete CRT arithmetic profile.
It does not rebuild validation or chunk planning in separate `q32`, `q64`, and
`q128` branches.

For a matrix with `R` rows and active width `W`, cached execution consumes a
validated transformed prefix of `R W` ring entries. Streamed execution covers
the same logical range in CRT safe chunks. Both modes return the same relation
or quotient rows. Differential tests exercise each field profile with a
distinct CRT capacity.

## Copy and parallel range reductions

Owned fold output moves directly into the existing proof container through a
consuming constructor. It does not copy between equivalent row layouts.

Exact prefix products use nonoverlapping parallel waves. Each wave reads a
completed prefix and writes the next disjoint range. Dense differential tests
cross the sequential boundary and every doubling boundary through the full
supported domain.

## Paired equality tensors and the carry recurrence

This section describes verifier kernels. Several verifier chapters refer to
the *paired tensor recurrence*; this is its definition.

### The problem

The equality polynomial $\operatorname{eq}(r,i)$ takes a point
$r=(r_0,r_1,\ldots)$ and an index $i$. It is a product with one factor for
each bit of $i$: the factor is $r_k$ when bit $k$ of $i$ is 1 and $1-r_k$
when it is 0.

The verifier must evaluate sums that contain two equality factors in each
term:

```text
scalar * sum over (i_0, ..., i_k) of
    product_j weight_j[i_j]
  * eq(left,  left_offset  + sum_j left_stride_j  * i_j)
  * eq(right, right_offset + sum_j right_stride_j * i_j)
```

One factor addresses a position in the witness. The other addresses a
position in the setup or in the relation. The sum can have millions of terms,
so the verifier cannot add them one by one.

### Plain definitions

- An **address** is the index passed to one equality factor.
- An **axis** is one independent coordinate $i_j$, such as the claim, block,
  digit, or row index. It has a length, a left stride, a right stride, and
  weights.
- A **stride** is the amount by which one step along an axis changes an
  address. A stride may be zero on one side.
- A **family** is one such sum: a pair of offsets, a scalar, and a list of
  axes. The code type is `EqPairTensorFamily`.
- A **carry** is what an addition passes from one bit to the next bit. An
  offset that is not a multiple of the axis size causes carries.

### The recurrence

The recurrence computes the sum one address bit at a time, from the lowest
bit upward. It keeps a small set of states. A state is a pair
(left carry, right carry) with an accumulated weight.

At each bit, every state is combined with every choice of the coordinate bits
that enter at this bit. The choice is added to both carries. The lowest bit
of each result selects the equality factor for that side, and the rest of the
result is the next carry. States with equal carries are then merged by adding
their weights. After the last bit, the answer is the weight of the state
whose carries are both zero.

The cost is about the number of bits times the number of states. It does not
grow with the number of terms.

### A small example

Take one axis of length 4 with unit weights, left stride 1, right stride 1,
left offset 3, and right offset 0. The left point is $(L_0,L_1,L_2)$ and the
right point is $(R_0,R_1)$, lowest bit first. Write $\bar x=1-x$. The sum has
four terms:

$$
\operatorname{eq}(L,3)\operatorname{eq}(R,0)
+\operatorname{eq}(L,4)\operatorname{eq}(R,1)
+\operatorname{eq}(L,5)\operatorname{eq}(R,2)
+\operatorname{eq}(L,6)\operatorname{eq}(R,3).
$$

The recurrence starts from the single state (3, 0) with weight 1. Bit 0 and
bit 1 each add one coordinate bit to both carries. Bit 2 adds nothing.

| After bit | States (left carry, right carry): weight |
| --- | --- |
| start | (3, 0): $1$ |
| 0 | (1, 0): $L_0\bar R_0$; (2, 0): $\bar L_0R_0$ |
| 1 | (0, 0): $L_0\bar R_0\,L_1\bar R_1$; (1, 0): $L_0\bar R_0\,\bar L_1R_1+\bar L_0R_0\,\bar L_1\bar R_1+\bar L_0R_0\,L_1R_1$ |
| 2 | (0, 0): $\bar L_2\cdot L_0\bar R_0L_1\bar R_1+L_2\cdot(\ldots)$ |

At bit 1 there are four combinations of state and choice. Three of them reach
the carry pair (1, 0) and are merged into one state. The final weight
expands to the four terms above. With a longer axis and the same offsets, the
set never holds more than two states, while the number of terms doubles with
every added axis bit.

### Supporting kernels

The entry point is `eval_boolean_pair_tensor_families` in
`crates/akita-algebra/src/offset_eq/tensor_pair/`. It applies these exact
reductions around the recurrence:

- It contracts independent, aligned low axes first, so they never enter the
  carry states.
- It evaluates families with the same recurrence geometry as one batch.
- It handles long unit-stride intervals with a shared binary-addition
  contraction.
- Construction merges adjacent unit-weight axes whose strides are contiguous,
  so a uniform layout becomes one long axis.
- A power-of-two axis whose weight is a product of one factor per bit stays in
  that factored form (`EqPairTensorWeights::BitProduct`). The kernel never
  expands it into a dense table.

One call has one work budget, `MAX_COMPACT_STRIDE_TERMS = 2^28`. A call that
would exceed it returns an error. All address arithmetic is checked, as the
[verifier no-panic contract](./verification.md#the-verifier-no-panic-contract)
requires.

### Where Akita uses it

- The setup-index weight of the setup contribution, in the direct scan and in
  Stage 3: [Setup contribution and Stage 3](./verifying/setup_contribution.md).
- The column weights of the evaluation-trace opening:
  [Evaluation trace](./verifying/evaluation_trace.md).
- The compression relation weights and the distributed relation verifier:
  [The distributed relation verifier](./verifying/distributed-relation-verifier.md).

Only the verifier uses these kernels. The prover materializes the dense
weights instead.

### Protocol rule and implementation choice

The value of each sum is a protocol rule: the relation and the layout define
it, and prover and verifier must agree on it. How the verifier computes the
value is an implementation choice. Every reduction above is exact, so the
choice of kernel cannot change whether a proof verifies. The tests under
`crates/akita-algebra/src/offset_eq/` compare the kernels with dense
reference evaluations.

## Design boundary

CPU storage plans remain private. A new backend implements the source typed
operation for its device and chooses its own storage, tiling, and scheduling.
It must preserve source order, checked geometry, arithmetic limits, and the
protocol output type. It must not absorb transcript state or expose a CPU row
plan as public API.

See [Compute Backends](https://github.com/LayerZero-Labs/akita/blob/main/docs/compute-backends.md) for backend ownership
and NTT lifecycle details. The full PR 375 design record is
[`specs/archive/2026-Q3/pr375-prover-streaming-and-onehot-unification.md`](https://github.com/LayerZero-Labs/akita/blob/main/specs/archive/2026-Q3/pr375-prover-streaming-and-onehot-unification.md).
