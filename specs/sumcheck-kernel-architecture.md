# Sumcheck evaluation tables and CPU kernels

| Field | Value |
|---|---|
| Author(s) | Quang Dao and Codex |
| Created | 2026-08-06 |
| Status | active |
| PR | [#368](https://github.com/LayerZero-Labs/akita/pull/368) |
| Supersedes | [`packed-sumcheck.md`](archive/2026-Q3/packed-sumcheck.md) |
| Superseded-by | |
| Book-chapter | |

## Summary

Akita already has fast SIMD field arithmetic, but its prover stores materialized
sumcheck tables as scalar extension field values. The prover either never calls
the SIMD code or pays most of the possible gain to rearrange scalar values inside
each round. This design changes the stored table shape so scalar and SIMD kernels
can use the same bytes without conversion.

The canonical materialized table stores each base field coefficient in one
contiguous section. Dense multilinear tables store rows in the order in which
sumcheck binds variables. This makes the two children for every fold available as
two contiguous slices. Compact i8 and i16 sources remain compact until a later
round requires a full field table. Sparse one hot witnesses keep their indices and
store their nonzero values in the same coefficient first form.

The transcript, proof, verifier, challenges, and sumcheck relations do not change.
This is a prover storage and kernel change.

## Intent

### Goal

Build one durable sumcheck representation and one set of canonical prover
operations that let fp32, fp64, fp128, i8, and i16 inputs use the fastest safe CPU
implementation available at runtime.

### Terms

`F` is the base field used to store coefficients. `E` is the field used for
sumcheck challenges and round messages. `E::EXT_DEGREE` is the number of `F`
coefficients in one `E` value.

A materialized table stores field values for every live row. A compact table
stores small signed integers or a short implicit description and has not yet
expanded those values into `E`.

Binding order is the row order in which the variable bound by the next sumcheck
round selects between two contiguous halves. Akita binds logical variable zero
first. If `logical` is an `n` bit row index, its binding order row is
`reverse_bits_n(logical)`.

### Measured starting point

The pinned one worker fp32 and D128 one hot proof at 28 variables takes 6.304
seconds to prove on the Ice Lake test machine. All sumcheck protocols account for
about 4.32 seconds, or 68 percent of that time. EOR takes 2.232 seconds. Stage 1
takes 1.400 seconds. Stage 2 takes 1.296 seconds.

The current `FpExt4<Fp32>` packed multiplication processes one logical value in
2.748 ns per lane, compared with 19.422 ns for the scalar operation. This is a
7.07 times throughput gain. The older claim that packed fp4 multiplication only
gains about 1.1 times is no longer true for the current field code.

The table layout determines whether this field gain reaches sumcheck:

| Kernel over 16,384 output rows | Ice Lake median | Relative to scalar |
|---|---:|---:|
| Scalar fp32 extension fold | 466.053 us | 1.00 times |
| SIMD fold over persistent contiguous halves | 56.619 us | 8.23 times |
| Repack adjacent scalar pairs, fold, and unpack | 372.140 us | 1.25 times |
| Scalar root factor pair operation | 2,371.174 us | 1.00 times |
| Root factor pair with all inputs already coefficient first | 414.500 us | 5.72 times |
| Root factor pair with only coefficients rearranged on demand | 980.810 us | 2.42 times |

These measurements rule out a design that keeps `Vec<E>` as the main storage and
rearranges it within every round.

### Invariants

1. The prover emits identical proof bytes for the scalar, NEON, AVX2, and AVX512
   plans.
2. The transcript absorbs the same values in the same order for every CPU plan.
3. The verifier and all verifier reachable types remain unchanged.
4. A materialized value has exactly one stored representation. The prover must
   not keep a scalar `Vec<E>` beside a coefficient first copy. A bounded
   repeated-value palette is a compact sparse state, not a second materialized
   table. While it is active, the table allocation is dormant backing storage
   and cannot be read as current evaluations.
5. The stored representation does not depend on SIMD width or CPU architecture.
6. Production release builds select SIMD at runtime. Compiling without
   `target-cpu=native` must not force the scalar path on a capable CPU.
7. Safe production code cannot request a CPU plan that the host does not support.
8. Compact i8 and i16 sources remain compact until the protocol needs a
   materialized table.
9. Production compact materialization writes directly into the canonical table
   when the source can stream the final coefficient slabs. A phase boundary may
   use one transient row buffer only when a named benchmark shows that source-
   sequential production plus a stored-row conversion is faster. The buffer is
   consumed immediately and never survives as parallel protocol state.
10. Full dense tables use binding order for their entire live lifetime. A fold
    does not change this invariant.
11. Sparse witness indices keep their current logical meaning except while the
    tensor factor owns an active suffix precontraction. During that private
    state, indices and palette tags remain at their documented original rows;
    the tensor factor owns the logical length and the accumulated low-fold
    weights. One contraction restores ordinary current indices and values
    before any noncached operation can read the witness. Row `j` in every
    ordinary sparse value table belongs to sparse index `indices[j]`.
12. SIMD and scalar accumulation obey the same exact modular arithmetic bounds.
    No optimization may rely on integer overflow or a reduction bound that the
    field type does not prove.
13. No hot round loop allocates. Folding shortens the live part of existing
    storage.
14. The payload size of a materialized table is
    `len * E::EXT_DEGREE * size_of::<F>()`. Small fixed metadata is allowed.
15. Large operations must preserve outer table parallelism. A field-specific
    SIMD operation may own the whole table only when one worker is available.
    With multiple workers, an operation either partitions disjoint row ranges
    and runs the same selected inner kernel in every range, or uses the measured
    parallel portable traversal until that chunked SIMD operation exists. Every
    worker owns only bounded private accumulators, and the merge result remains
    field exact.
16. A Rayon fold identity must be small. In particular, a class histogram with
    field accumulators may be allocated once for a one-worker operation, but it
    must not be cloned per Rayon task. Multiple-worker paths use a bounded direct
    traversal or partition one shared output into disjoint writable ranges.

### Non goals

This work does not change the sumcheck protocol, polynomial degrees, proof
encoding, Fiat Shamir labels, planner schedules, commitment format, NTT cache,
CRT representation, or verifier arithmetic.

This work does not add a GPU kernel. The table representation may be useful to a
future GPU backend, but this specification covers CPU execution only.

This work does not force every field family to use SIMD. The measured faster
implementation wins for each field and operation.

## Design

### Crate ownership

`akita-field` owns extension coefficient access and base field arithmetic. It
does not own sumcheck table order or sumcheck protocol operations.

`akita-sumcheck` owns `EvaluationTable<F, E>`. It also owns the scalar reference
operations that apply to any sumcheck table. The generic transcript drivers stay
unchanged.

`akita-prover` owns runtime CPU selection and optimized operations for EOR,
Stage 1, Stage 2, and Stage 3. These operations use `EvaluationTable` directly.
Architecture specific code stays under one sumcheck kernel module in this crate.

`akita-verifier` does not import the new table or CPU plan.

The data flow is:

```text
compact i8 or i16 source       dense or sparse field source
            |                              |
            | direct materialization       | direct construction
            v                              v
                 EvaluationTable<F, E>
                            |
               detected CPU operation
                            |
       compute round, fold, or fold and compute next round
                            |
                   ordinary UniPoly<E>
                            |
                unchanged transcript driver
```

### Extension coefficient access

`ExtField<F>` gains two required primitive methods:

```rust
fn from_base_fn<G>(f: G) -> Self
where
    G: FnMut(usize) -> F;

fn base_coefficient(&self, index: usize) -> F;
```

`from_base_slice` and `to_base_vec` remain convenience methods, but their default
implementations use these two primitives. Implementations for the identity
field, `FpExt2`, `FpExt4`, and `FpExt8` define only the primitives. This makes
coefficient access allocation free and gives it one source of truth.

Production table construction must call `base_coefficient` directly. In
particular, projecting a tensor equality value against the head equality
weights must not call `to_base_vec` for each logical row. The convenience vector
form is reserved for small boundaries and tests; a row loop over it creates one
heap allocation per evaluation and defeats the coefficient-first ownership
model before the SIMD sumcheck begins.

### EvaluationTable

The table has this semantic shape:

```rust
pub struct EvaluationTable<F, E> {
    coefficients: Box<[F]>,
    len: usize,
    stride: usize,
    marker: PhantomData<fn() -> E>,
}
```

The fields are private. `stride` is the number of rows allocated when the table
was built. `len` is the number of live rows. Coefficient `c` occupies
`coefficients[c * stride .. c * stride + len]`.

The following conditions hold after every public operation:

```text
len <= stride
coefficients.len() == E::EXT_DEGREE * stride
coefficient_slice(c).len() == len
c < E::EXT_DEGREE
```

The table uses one allocation. Each coefficient occupies one contiguous section.
The table keeps the full allocation as rounds shorten `len`, matching the current
in place `Vec<E>` capacity behavior.

The initial row preserving API is:

```rust
from_evaluations
from_evaluation_fn
from_coefficient_fn
from_coefficient_storage
evaluation
coefficient_slice
coefficient_slice_mut
len
is_empty
truncate
into_evaluations
```

These constructors and accessors preserve stored row order. They do not reverse
index bits. `from_evaluations` exists for tests, small setup values, sparse value
lists, and API boundaries. `into_evaluations` is also a boundary and test
operation. Hot prover code must not call it.

Dense multilinear ownership boundaries use these separate canonical
constructors:

```text
from_multilinear_evaluations
from_multilinear_evaluation_fn
from_multilinear_evaluation_array_fn
from_multilinear_coefficient_fn
```

They accept or generate logical LSB first rows and write binding order directly.
They require a nonzero power of two row count. Large production producers use
the function constructors so they never hold a second full representation. The
private bit reversal helper is shared by all dense constructors.

The array constructor generates one of a fixed number of independent tables
from `(table_index, logical_row)`. With multiple workers, independent tables and
disjoint stored-row ranges within each table are the parallel units. Every task
writes one contiguous range in every coefficient slab, and the one-worker path
retains one sequential stored-row traversal. Parallelizing logical-row ranges
would scatter bit-reversed writes through every slab and is not the canonical
construction.

`truncate` only changes the live length. It cannot grow the table. Mutable raw
access to `coefficients`, `len`, or `stride` is not exposed.

### Dense row order

All full multilinear tables owned by a sumcheck prover use binding order. For a
table of length `N`, the next round reads the left child from row `i` and the
right child from row `i + N / 2`.

For challenge `r`, the folded value is:

```text
out[i] = left[i] + r * (right[i] - left[i])
```

The operation writes `out` into the first half and sets the live length to
`N / 2`. The first half is already the binding order for the remaining logical
variables. No later permutation is needed.

Code that receives current logical LSB first evaluations converts them once at
the ownership boundary. Code that generates a table uses the stored row to
recover the logical row and writes the final order directly.

### Sparse values

`SparseExtensionOpeningWitness` evolves from a vector of `(index, value)` pairs
to these fields:

```rust
table_len: usize
indices: Vec<usize>
values: EvaluationTable<F, E>
merge_free_rounds_left: usize
merge_free_values: Unchecked | Unavailable | Palette
```

The indices stay sorted according to the sparse algorithm's current logical
order. The value table has `indices.len()` rows. It is coefficient first but its
rows follow the index sidecar rather than dense binding order.

Duplicate normalization, zero removal, and merge detection remain one canonical
constructor operation. A constructor cannot accept independently prepared
indices and values that have not passed that normalization.

The private palette state is attempted only by the tensor operation when at
least two merge-free rounds remain. Detection accepts at most eight distinct
values and at most eight low path bits. Each row then stores one `u16`: the low
byte is its original merge-free path and the high byte is its palette tag. The
folded palette has at most `8 * 2^8` extension values. It reserves its full
bounded capacity when detected and performs no allocation in a round.

During the palette-only state, the coefficient-first allocation is retained
only as the destination for the first merging round. It is not a second current
value representation. When the tensor factor also owns a suffix
precontraction, original indices, tags, and dormant values remain unchanged
until one multi-round contraction restores the ordinary sparse table. The
palette is materialized exactly once before any general pair traversal,
dense-factor fallback, or public final value can read it. Failed detection is
remembered so later rounds do not rescan the rows.

The root one hot EOR is the only path with the fixed support plateau seen in the
profile. Its 2^20 entries remain live for the first six folds of a 2^26 domain.
Recursive balanced digit witnesses do not use this sparse form. They use dense
tables and their work halves each round.

### Sparse tensor rounds

The root one hot witness may keep almost all of its rows for several rounds
without being globally merge free. When several polynomials are combined, one
early adjacent collision is enough to make `merge_free_rounds_left` zero even
though most rows remain isolated. The tensor operation must therefore handle
ordinary sparse pairs directly. It must not use one global collision flag to
select the fast path.

Rows are sorted, so consecutive witness pairs also form consecutive tensor
suffix groups. For each pair, let `w0` and `w1` be its two witness children,
and let `s0[c]` and `s1[c]` be the low tensor states in coefficient column `c`.
The grouped operation accumulates:

```text
constant[c]  += w0 * s0[c]
quadratic[c] += (w1 - w0) * (s1[c] - s0[c])
```

It applies the shared tensor suffix only after every pair in that suffix group
has been accumulated. This computes the same degree two round polynomial as
constructing both factor children for every pair, but it reduces intermediate
factor values once per suffix group rather than once per sparse row.

The lazy factor stores each low pair as `(state_zero, state_delta)`, where
`state_delta = state_one - state_zero`. This has the same payload as storing two
child states, removes the state subtraction from every live sparse pair, and is
the direct input shape of the constant and quadratic formulas above.

After a challenge, the same operation folds and compacts the current witness
pairs while feeding each nonzero output into the next grouped round. This is one
streaming pass. It supports arbitrary pair merges and does not depend on the
merge free plateau. The existing merge free formula remains the fallback for a
dense factor or a field without an exact grouped product accumulator.

When both the current fold and the cached next round are proven merge free, the
grouped operation consumes the stored nonzero child directly. If the repeated
value palette is available, it reads the already-folded value by row code rather
than multiplying and rewriting one full extension value per row. Requiring two
guaranteed rounds is essential: one round proves the current pair is isolated
but does not prove that two outputs will not become siblings in the cached next
round.

`ExtensionOpeningTables` owns the choice of fused operation for its active
representation. A term asks the table to fold and compute the next round, then
caches the scaled result. The term does not duplicate representation matching.

### Sparse tensor suffix precontraction

For a sufficiently large root one hot table, the lazy tensor factor groups its
fixed high suffix columns once by original low index and palette tag. It then
contracts the original palette values into one value-weighted suffix vector per
low index. A challenge scales that low index by `1 - r` or `r`. Linearity makes
this valid after sparse rows begin to collide; a collision changes the ordinary
row representation but not the sum over original rows.

The cache owns both the changing suffix vectors used for round messages and one
fold weight per original low index. While it is active, the factor length is the
term's logical length and the witness retains its original sorted rows. On the
penultimate lazy-factor fold, one sorted witness pass applies all accumulated
weights, merges equal shifted indices, removes zeros, and restores the ordinary
sparse representation. The final low fold then uses the existing sparse and
dense-factor transition. No cached state reaches final evaluation or a public
witness accessor.

The cache is private to `TensorEqualityFactor`, which owns the suffix tables it
summarizes. Construction is gated by a row-count crossover and a detected
bounded value palette. Unsupported shapes continue through the canonical
grouped sparse traversal. The differential oracle must force cache creation,
cross the first collision, cross cache materialization, and compare every round
and final evaluation with a fully materialized tensor factor.

### Compact i8 and i16 states

Stage 1 and the compact prefix of Stage 2 keep their existing small integer
storage. A compact kernel may perform all of these operations without
materializing `E` values:

1. Load and widen signed i8 or i16 values.
2. Compute the round polynomial with extension field weights.
3. Apply one or two early challenges while the exact compact formula permits it.
4. Build the first required `EvaluationTable` in binding order, directly when
   the source traversal can write its coefficient slabs efficiently.
5. Compute the next round while writing that table when a fused pass saves a
   second read.

For a basis-4 or basis-8 Stage 1 leaf, three bound low variables determine each
remaining row from one byte-sized or word-sized octet class. The prover keeps
that fact explicit for one additional sparse low-variable round:

```text
FoldedOctets {
    class_codes: one u16 per live row,
    class_values: one E per possible octet class,
    class_taylor_coefficients: Q and its first three normalized derivatives,
}
```

Basis 4 has 256 classes and basis 8 has 65,536. The third challenge evaluates
each possible class once. The next round reads class codes and computes its
quartic coefficients from the cached Taylor row. Fp32 uses the detected packed
operation for this class lookup and polynomial arithmetic without changing the
compact representation. The following challenge folds the coded rows and
materializes only the already halved field table. Fp32 performs that transition
with the same packed traversal, replacing only its source read with a class-code
lookup. Later sparse low-variable challenges read materialized values directly.
Both forms write the two folded values and compute the following quadratic or
quartic round in the same pass. The compact state cannot survive into the
x-variable phase or the final evaluation.

The low-basis materialized suffix is a measured exception to the general dense
table rule. It remains one row-oriented `Vec<E>` because its sparse-y and
prefix-x operations consume rows in that order. A coefficient-first binding-
order prototype made root Stage 1 108 to 111 ms instead of about 100 ms and
slowed the later geometrically shrinking folds. This is one live
representation, not a scalar copy beside a SIMD table. High-basis leaves and
all dense product sumchecks still use `EvaluationTable` once materialized.

The canonical quartic entry formula is the Taylor expansion at the left child:

```text
Q(left + delta X) = t0 + t1 delta X + t2 delta^2 X^2
                    + t3 delta^3 X^3 + delta^4 X^4
```

where `t0 = Q(left)`, `t1 = Q'(left)`, `t2 = Q''(left)/2`, and
`t3 = Q'''(left)/6`. Compact class tables and ordinary materialized rounds use
the same Taylor formula. The protocol owns one sparse low-variable traversal
over either class-coded or materialized values. The field operation plan may
handle a complete storage shape in SIMD; returning no implementation keeps the
same traversal on its exact blocked scalar accumulator. This retains one
protocol state machine rather than parallel scalar and packed sumcheck paths.

AVX512 byte and word operations require AVX512BW. The fp32 fold uses AVX512F,
AVX512DQ, and AVX512IFMA. Without IFMA, the current plan falls back to AVX2
instead of assuming that a wider register wins. An operation that uses 128 or
256 bit AVX512 forms must also check AVX512VL. Each target function declares and
checks its exact feature set. The runtime plan selects these operations
independently. A machine may therefore use AVX512 IFMA for fp32 table folds and
AVX2 for a compact i8 or i16 operation if measurement shows that choice is
faster.

NEON uses its widening operations for compact values. A compact path may remain
scalar on a target when its measured SIMD version does not win.

### Runtime CPU selection

`akita-prover` owns one opaque `SumcheckKernelPlan`. Production code obtains it
only through detection. Each sumcheck prover detects and stores one copy during
construction. The plan is not serialized, and a round does not repeat CPU
feature checks. The plan contains private function choices for each operation
rather than one public CPU tier.

Generic protocol code reaches those choices through
`SumcheckTableOperations<F>`. Its default methods call the canonical scalar
operations. A field family overrides only operations with an accepted runtime
implementation. `FpExt4<Fp32>` overrides all three dense operations; fp64,
fp128, and other extension shapes currently keep the scalar defaults. This is a
static field capability, not a table wrapper or a second protocol path.

The plan may select different implementations for:

| Operation family | Candidate implementations |
|---|---|
| fp32 extension table arithmetic | scalar, NEON, AVX2, AVX512 IFMA |
| fp64 extension table arithmetic | scalar, NEON, AVX2, AVX512 |
| fp128 table arithmetic | scalar, measured SIMD only if it wins |
| compact i8 operations | scalar, NEON, AVX2, AVX512BW |
| compact i16 operations | scalar, NEON, AVX2, AVX512BW |
| sparse factor reads | scalar reads, batched reads, AVX512 gather |

Runtime feature checks happen before a target specific function is selected.
Dispatch happens once per whole table operation, outside its row loop. A portable
release build includes the target specific functions for its CPU architecture.

Tests may force a supported plan to compare results. The forced constructor is
not part of the production public API and must reject an unsupported host before
calling target specific code.

### Canonical operations

Each concept has one operation. Scalar and architecture specific code are private
implementations selected by that operation. Protocol modules do not contain a
second scalar copy.

The shared dense operations are:

```text
fold_first_variable
compute_product_round
fold_and_compute_product_round
```

`compute_product_round` computes the degree two constant and quadratic sums used
by EOR. The linear coefficient remains derived from the previous claim, as it is
today.

`fold_and_compute_product_round` folds two tables and computes the next round
from the folded rows in one pass. It preserves the existing delayed reduction
contract. It is used only when another round exists and the fused pass reduces
memory traffic.

Stage 1 and Stage 2 keep operations named for the relation they compute. They use
the same table, plan, coefficient access, and fold operation. The design does not
add a generic iterator or callback API that hides a required fused pass from the
compiler.

### Reduction and accumulation

The scalar reference uses the current `HasUnreducedOps` contract. A SIMD
implementation must prove its integer bounds for the exact modulus and maximum
chunk size before using delayed reduction.

`akita-sumcheck` owns the product round accumulator interface and its delayed
and direct reduction implementations. The scalar table operation and the dense,
fused, and sparse EOR traversals use those same implementations. The delayed
implementation rejects a field unless `DELAYED_PRODUCT_SUM_IS_EXACT` is true.
This keeps the reduction policy in one source of truth.

Small signed inputs may use wider integer accumulators and reduce less often.
Full field products may need a different accumulator. The operation plan may
select different accumulation code for compact first rounds and later full field
rounds.

`HasUnreducedOps::SmallProductAccum` is the exact short-batch primitive for a
field value times an unsigned 16-bit value. Its implementation states the
maximum safe number of terms. Compact operations accumulate positive and
negative terms separately, promote each bounded batch into `ProductAccum`, and
perform the ordinary field reduction only at the protocol boundary. For
`FpExt4<Fp32>`, the short accumulator is four `u64` coefficient sums and the
long accumulator is four `u128` coefficient sums. Other fields may use their
ordinary product accumulator for both roles.

The specification does not require a universal packed accumulator trait. Such a
trait would force unrelated fields and protocols into one shape. The canonical
sumcheck operation owns its accumulator and uses the field's existing reduction
primitives.

### Parallel execution

SIMD is the inner row loop. Rayon remains the outer table partition when the
table exceeds a measured threshold. Each partition begins and ends on a row
boundary. Each worker returns scalar field accumulators, and the operation merges
them exactly once.

Operation selection is worker-count aware because a serial whole-table SIMD
call can be slower than a parallel portable traversal even when its inner loop
is much faster. One-worker runs select the measured SIMD operation. A
multiple-worker run selects chunked SIMD only when the operation exposes safe
disjoint ranges; otherwise it selects the existing parallel portable operation.
This is one operation policy, not a second table representation or protocol
implementation.

Parallel mutation borrows all coefficient slabs once, divides every slab at the
same row boundaries, and gives each task its corresponding disjoint slices. It
must not copy table payloads into per-task buffers. A generic extension-degree
operation may allocate a bounded vector of slice references at task setup. The
fixed-degree SIMD accessors remain allocation free.

The implementation must benchmark one worker and the normal parallel setting.
The one worker result shows kernel quality. The parallel result shows application
throughput. Neither result replaces the other.

### Field families

For `FpExt4<Fp32>`, the coefficient first table has four fp32 sections. This is
the main SIMD target and must have scalar, AVX2, AVX512, and NEON coverage where
the host architecture permits it.

For `FpExt2<Fp64>`, the table has two fp64 sections. The implementation must
measure the scalar wide path against SIMD before choosing a default operation.
The choice is per operation. On Apple Silicon, the scalar product round is
faster while NEON wins for the fused fold and next product round. The detected
plan therefore mixes those choices instead of assigning one tier to every fp64
operation.

For the fp128 identity field, the table has one fp128 section. Its memory payload
is the same as the current `Vec<Fp128>`. The initial operation remains scalar
unless a measured SIMD kernel wins end to end.

The table and scalar reference also support `FpExt8` if a configuration needs it.
This does not require an optimized `FpExt8` kernel in the first cutover.

The first dense EOR acceptance measurement used 65,536 rows and one Rayon
worker on Apple Silicon. The old row-major EOR and the accepted coefficient-first
operations measured:

| Field | Old median | Accepted median | Change |
|---|---:|---:|---:|
| fp32 quartic extension | 1.811366 ms | 1.3728 ms | 24.2 percent faster |
| fp64 quadratic extension | 0.800049 ms | 0.80775 ms | 0.96 percent slower, statistically unchanged |
| fp128 identity field | 1.028971 ms | 1.0120 ms | 1.65 percent faster |

The fp64 all-NEON plan measured 0.89332 ms. The accepted mixed plan keeps the
initial product round scalar and uses NEON for fused rounds. A generic scalar
slice experiment was rejected because it measured 4.5623 ms for fp32, 1.2935 ms
for fp64, and 1.0500 ms for fp128. Field-shaped kernels are required; extension
degree alone is not a sufficient hot-loop abstraction.

The first grouped sparse tensor measurement used the fp32 D64 one hot EOR at 26
variables, four polynomials, and one Apple Silicon worker. The last pushed
coefficient-first sparse path measured 305.00 ms. Grouping arbitrary sparse
pairs and fusing compaction with the next round measured 223.21 ms on the clean
final run, which is 26.8 percent faster. A merge-free-only grouped prototype
measured 302.52 ms and was rejected because the real four-polynomial witness has
early collisions. The pair-native zero-and-delta state layout then reduced the
clean median to 200.49 ms, which is 34.3 percent faster than the 305.00 ms
baseline.

The lazy tensor boundary is measured rather than inferred from domain size.
For the same fp32 target shape, 12 lazy rounds measured about 207.2 ms, 10
rounds measured about 200.7 ms, and 11 rounds measured about 200.1 ms in
alternating runs. The production cap is therefore 11 rounds. A prototype that
converted the nearly dense sparse witness and its tensor factor to dense tables
measured 206 to 209 ms depending on the transition point; factor materialization
cost more than the later dense SIMD rounds saved, so the cutover was removed.

Sparse indices are strictly sorted and unique after construction and after
every fold. One adjacent logical pair therefore contains at most its even and
odd child; it never needs a general duplicate-combining loop. Reading that
canonical pair shape once and sharing it across grouped accumulation, fused
folding, and the ordinary sparse fallback reduced the same fp32 benchmark to a
182.94 ms clean median. This is 40.0 percent below the 305.00 ms sparse-table
baseline.

On the pinned one-worker fp32 D128 proof at 28 variables, the root witness has
2^20 rows, four repeated extension values, and six merge-free rounds. Folding a
bounded value palette instead of every row reduced root EOR from 296 ms to 258
to 260 ms. The four middle plateau folds fell from 38 to 39 ms each to 28.5 to
30.0 ms. The complete proof verified, and its second sample measured 1.898
seconds versus a 1.893-second pushed-head sample while unrelated Stage 1 and
Stage 2 spans were slower in the new run. The protocol-local 12 percent EOR gain
is stable; whole-proof attribution requires pinned Ice Lake confirmation.

During that merge-free plateau, the folded witness values and the tensor
factor's low states are both bounded class tables. The product of their class
counts remains roughly constant as one grows and the other shrinks. The tensor
factor therefore precomputes each value-class times low-state product once per
round, then the million-row traversal only widens and adds those reduced
products. The reusable allocation is reserved when the factor is constructed.
For `FpExt4<Fp32>`, embedding a reduced element in the exact product accumulator
is a four-coefficient widening operation. Two verified one-worker samples
reduced complete root EOR from 330 ms to 270 to 272 ms on the current branch.
The four middle plateau rounds fell from 28.5 to 29.2 ms to 16.4 to 17.3 ms,
and complete proofs measured 1.683 and 1.703 seconds.

Palette detection and the same product table are also used for the first round,
before any challenge has folded the palette. The initial grouped contraction
therefore does not fall back to per-row products merely because the cache was
previously initialized by `ingest_challenge`. On the current profile, round zero
fell from 59.5 ms to 40.7 to 40.9 ms and complete root EOR from 268 ms to 250 to
253 ms. Both complete proofs verified.

The dense recursive EOR factor originally projected every tensor equality value
through `to_base_vec`, allocating a four-element vector for each row. Reading
the same canonical coefficients through `base_coefficient` reduced the first
2^21-row recursive EOR construction from 176 ms to 124 to 125 ms and its total
EOR from 233 ms to 179 to 181 ms. The two complete proofs measured 1.644 and
1.648 seconds and verified. The arithmetic and resulting tables are unchanged.

Dense EOR then removes the row-major factor boundary completely. One prepared
`TensorFactorProjection` owns the validated head equality weights. A split tail
equality source streams each projected value directly into the factor's final
binding-order coefficient storage. The witness is converted once afterward.
This reduced the largest recursive construction again to 97.4 ms and its whole
EOR to 153 to 154 ms. Two complete proofs measured 1.606 and 1.616 seconds and
verified. No temporary full factor vector exists on this path.

The prepared projection also follows the field's product-sum policy. The fp32
and fp64 extensions certify that a short sum of unreduced products is exact, so
their four extension-times-base coordinate products reduce once instead of four
times. Other fields retain direct per-product reduction. This reduced the root
sparse term construction from about 82.8 ms to 74.6 to 74.7 ms and the largest
recursive construction from 97.4 ms to 87.2 to 87.7 ms. Its complete EOR is
142 ms. Two complete proofs measured 1.583 and 1.589 seconds and verified.

Production EOR receives its transcript-bound input claim from the independently
derived tensor partials. Term construction must not recompute every
`witness[row] * factor[row]` merely to compare the tables with that same claim;
sumcheck soundness and the final committed-witness opening reject an inconsistent
table. The explicit `recompute_input_claim` helper remains for tests and
standalone callers and makes its linear scan visible in its name. Removing the
eager scans reduced root sparse construction from 74.8 ms to 15.4 to 15.5 ms
and complete root EOR from 250 to 253 ms to 191 to 192 ms. The largest recursive
construction fell from 87.2 to 87.7 ms to 75.8 to 77.5 ms. Complete proofs
measured 1.496 and 1.522 seconds and verified.

Root sparse EOR transposes the fixed high-factor contraction once. The first
round groups each suffix factor by the entry's original low index and bounded
witness-value tag, then contracts the original palette value. Every lazy low
round is evaluated from 2,048 value-weighted suffix groups, including rounds
after ordinary sparse rows would collide. Original rows are folded once just
before lazy-factor materialization instead of once per round. The first accepted
merge-free-only cache reduced complete root EOR from 191 ms to 102 and 111 ms.
Extending the same linear form through collisions reduced it again to 66.9 ms.
Deferring the ordinary sparse folds to one 13.4 ms contraction produced 50.4
and 49.8 ms root EOR and 1.244 and 1.245 second complete proofs. All samples
verified with unchanged proof bytes.

On the same profile, retaining Stage 1 octet classes after the third challenge
and using the canonical Taylor kernel reduced the root Stage 1 sumcheck from
160 to 161 ms to 138 to 139 ms. Its largest next-round polynomial fell from
55.1 ms to 34.5 to 34.6 ms. Building codes, class values, and cached Taylor
rows cost 7.2 to 7.3 ms. The following fused rounds measured 31.0 to 31.1 ms,
17.2 to 17.4 ms, and 10.7 to 10.8 ms. Both complete proofs verified; the second
whole-proof sample measured 1.874 seconds.

The fp32 operation plan now also handles the class-coded polynomial round and
the later materialized sparse folds. The class-coded round keeps its `u16`
codes and gathers the bounded value and Taylor tables into SIMD lanes. Once the
state is materialized, one packed traversal folds adjacent pairs, writes the
halved table, composes the direct-range polynomial, and accumulates the split
equality weights. Partial final register chunks use zero equality lanes, so the
stored shape remains independent of SIMD width. On the same one-worker fp32
D128 Apple profile, the two largest materialized sparse folds fell from about
17 and 10 ms to 11.0 and 5.51 ms. Complete root Stage 1 fell from about 125 ms
after the class-coded round alone to 115 ms. The complete 1.454-second proof
verified and retained the 77,834-byte proof size.

The class-coded-to-materialized transition now uses that same packed fused
operation. Only the source function differs: it maps each `u16` code through
the bounded class-value table before the shared fold, polynomial composition,
and equality accumulation. Two clean samples reduced the largest transition
from 30.4 ms to 21.9 to 22.1 ms and complete root Stage 1 from 115 ms to 104 to
106 ms. Complete proofs measured 1.438 and 1.447 seconds, both verified, with
the same 77,834-byte proof.

High-basis Stage 1 product substages now keep compact class-pair indices through
their second round and use a runtime-selected packed affine-product operation.
The operation gathers folded values from the bounded class table, evaluates the
quadratic or quartic parent products in SIMD lanes, exploits the split equality
table's contiguous inner blocks, and handles the implicit padding suffix once.
After two challenges, every child lane materializes directly into its canonical
coefficient-first table; all later rounds use the same packed operation and fold
as other materialized sumchecks. The shared scalar affine-product coefficient
formula remains the correctness reference.

On the one-worker fp32 D128 Apple profile, the largest basis-64, eight-lane
compact second round fell from 18.9 to 19.0 ms to 10.9 to 11.2 ms. Its complete
product substage fell from 47.5 ms to 37.0 to 38.1 ms, and the enclosing Stage 1
instance fell from 78.0 ms to about 68.2 ms. The other two basis-64 instances
also improved. Two complete verified proofs measured 1.487 and 1.476 seconds.
An earlier-materialization prototype was rejected: expanding after the first
challenge cost 17.2 ms and made the same substage 51.9 ms.

High-basis polynomial leaves use the same materialized representation. The
compact class-indexed first round remains unchanged. After its first or second
challenge, depending on the class count, the range image and remaining equality
weights materialize directly into coefficient-first binding-order tables. One
runtime-selected affine-polynomial operation composes the shared degree-at-most-
four polynomial with every row pair and accumulates all coefficients against
the equality table. Later challenges use the same detected table fold as the
product substages. The old row-oriented exact-prefix table and its separate fold
implementation have no remaining production role and are removed.

On the one-worker fp32 D128 Apple profile, the four polynomial leaves changed
from 26.9, 20.6, 12.7, and 9.44 ms to 21.9 to 23.3, 17.4 to 18.4, 8.87 to 9.50,
and 8.62 to 9.17 ms. Their combined improvement is 13 to 18 percent. The
enclosing Stage 1 instances measured 37.6 to 39.8, 62.6 to 66.9, 33.1 to 35.7,
and 30.6 to 32.8 ms. Complete proofs measured 1.469 and 1.420 seconds and
verified with the unchanged 77,834-byte proof.

The fp32 Stage 2 compact prefix originally widened every digit-derived product
into four `u128` coefficient sums. Accumulating each x-column first in the exact
`u64` short-batch representation and promoting once per output reduced the root
prefix from about 103 to 107 ms to 92 to 93 ms on Apple Silicon. Complete root
Stage 2 fell from about 258 ms to 229 to 230 ms. An explicit NEON multiply was
rejected because it measured 98 to 99 ms; LLVM's scalar source shape generated
the better loop.

The prefix also keeps equality and evaluation-trace factors outside the compact
y scan. For each live x column, the eight norm outputs are accumulated against
the y equality weights first and multiplied by the column's x equality weight
once. Likewise, `PreparedProverEvaluationTrace` visits each source row with its
lane-constant factor still separate. The compact lookup contraction reduces
that source row to eight outputs before applying the factor. It never expands a
full trace row merely to consume it in the same prefix.

On the one-worker fp32 D128 profile, factoring only the x equality weight
reduced the root prefix from about 95 ms to 91.4 ms. Retaining the prepared
trace factor through the same contraction reduced it further to 72.7 to
72.9 ms. Complete root Stage 2 fell from 235 to 236 ms to 209 to 210 ms. The
compact handoff stayed at 53.7 to 53.8 ms and the later coefficient rounds were
unchanged, isolating the gain to the prefix. The second complete proof measured
1.786 seconds and verified.

The compact-to-field handoff preserves the same factors. It contracts the
folded witness separately against the common alpha row and each prepared trace
source, using the exact product-sum accumulator. Only the resulting three
relation coefficients are multiplied by the lane weight or source factor. It
does not build a materialized relation-factor table. This changed the root
handoff from 53.7 to 53.8 ms to 38.7 to 38.8 ms and complete root Stage 2 from
209 to 210 ms to 194 ms. The second complete proof measured 1.768 seconds and
verified.

The Stage 2 handoff is the named transient-buffer exception from invariant 9.
Its lane-parallel scan produces one row-major `Vec<E>`, then consumes it
immediately into the sole binding-order `EvaluationTable`. Writing the four
distant coefficient slabs through generic setters removed that buffer but lost
the lane-parallel traversal; complete root Stage 2 regressed from about 181 ms
to 223 to 327 ms. The accepted conversion costs about 4.9 ms at the root and no
row-major state remains after the handoff.

Later Stage 2 coefficient folds preserve those factors as well. Each round
contracts the folded witness separately against common alpha and each prepared
trace source, then applies the lane-constant factor to the three relation
coefficients. On the same one-worker profile, the four root coefficient rounds
changed from about 34.0, 18.3, 10.6, and 6.7 ms to 27.1, 15.6, 9.8, and 7.0 ms.
Complete root Stage 2 measured 183 to 184 ms in two verified samples. The
coefficient loops read the witness once per retained source instead of
materializing full factor rows that would be immediately consumed.

The compact prefix also distinguishes unsigned norm lookups from signed
relation lookups. Every norm table entry is `w(w + 1)` at a finite point or
`w^2` at infinity, so it accumulates into one exact short-product accumulator
instead of identical positive and permanently zero negative accumulators. The
signed alpha and trace relations retain their positive and negative halves.
Two complete one-worker fp32 D128 profiles reduced the root prefix from 71.9 ms
to 70.1 and 70.8 ms; complete root Stage 2 measured 182 and 184 ms. Both proofs
verified with the unchanged 77,834-byte proof.

Dense EOR terms construct both canonical tables in stored binding order. The
transparent equality point is reversed once so its ordinary little-endian
evaluation table is already in binding order; its coordinate projection and
coefficient-first write are then sequential. Witness conversion uses the same
stored-row traversal. This removes the former cache-hostile bit-reversed output
writes without changing the persistent representation or adding a second table
type. On the largest recursive fp32 D128 EOR, factor construction fell to
32.6 ms, complete dense-term construction to 42.1 ms from 75.5 ms, and complete
EOR to 98.2 ms from 130 ms. The 1.391-second proof verified unchanged.

Dense tensor sources now return the canonical table itself. In particular, a
recursive i8 witness widens directly into binding-order coefficient slabs;
`TensorPackedWitness` does not retain a logical extension-value alternative.
The largest recursive producer costs 6.56 to 6.64 ms, versus 2.56 ms for the
old logical vector plus 6.06 to 7.25 ms for its table conversion. Dense EOR also
factorizes the equality suffix into two small tables and precomputes the two
branch projection maps. One SIMD traversal writes the permanent factor table
and computes the first product round from the same packed values. That fused
operation costs 32.2 to 34.2 ms, versus 31.4 ms for factor construction plus
11.8 ms for the old first-round reread. The largest recursive EOR measures 84.0
to 85.1 ms, down from 96.0 ms, and the next measures 18.9 to 19.5 ms, down from
21.0 ms. Warm complete proofs measure 1.196 to 1.220 seconds and verify.

On Ice Lake, each tensor projection coefficient now uses the fp32 backend's
exact four-product accumulator and performs one final Solinas reduction instead
of reducing four products separately. In the exact pinned 28-variable fp32 D128
profile, total dense factor construction fell from 728.399 ms to 440.307 ms and
the largest factor fell from 530.945 ms to 320.242 ms. Total EOR time fell from
1.092102 seconds to 804.218 ms. The traced complete proof fell from 3.732681
seconds to 3.493352 seconds, verified successfully, and remained 77,819 bytes.
The matched reduction in factor and EOR time isolates the gain to the intended
kernel rather than unrelated stage movement.

Stage 3 now stores its coefficient and setup-index phases in `EvaluationTable`
and uses the same detected fold, product-round, and fused fold-plus-next-round
operations as dense EOR. The linear coefficient is derived from the carried
claim instead of accumulated separately. On the one-worker fp128 D64 recursive
profile, the old Stage 3 measured 94.5 ms and the canonical path measured 95.8
to 95.9 ms, a 1.5 percent difference within the two-percent fp128 gate. The
sumcheck portion improved from 50.0 ms to 49.4 to 49.5 ms. Logical-order direct
construction avoids a rejected temporary-vector transpose that had measured
224 ms for Stage 3.

The first normal-worker benchmark exposed two parallel-architecture bugs that
one-worker SIMD measurements could not show. First, a 65,536-entry Stage 1
class histogram was used as a Rayon fold identity, which cloned several
megabytes per task and raised fp32/fp64 peak RSS to about 1.7 GiB. The histogram
is now a one-worker operation only; the multiple-worker path uses the bounded
direct traversal. Warm local RSS returned to 539--543 MiB, matching main.

Second, several detected fp32 operations consumed the whole table serially and
bypassed the protocol's outer Rayon traversal. Multiple-worker dispatch now
uses the parallel portable operation where a chunked SIMD entry does not yet
exist. Stage 2's coefficient fold has a generic disjoint-slice traversal that
partitions all coefficient slabs together and merges only its six small field
accumulators. On the 16-worker fp32 D128, 28-variable proof, that fold fell from
77.0 ms to 13.7--14.6 ms and complete Stage 2 fell from 147.7 ms to 87--92 ms.
The complete proof measured 0.653--0.678 seconds versus 0.671 seconds on the
same-machine main trace. For fp64, the same coefficient fold fell from 48.5 ms
to 8.6 ms and complete Stage 2 from 99.6 ms to 65.4 ms.

Canonical table construction follows the same policy at the ownership boundary.
Independent product lanes are generated in parallel, and each table may divide
its stored rows into disjoint ranges that write the same range in every
coefficient slab. The one-worker path remains sequential. Parallelizing the
independent tables first reduced measured fp32 two-round product materialization
from 30.1 ms to 10.6 ms. Extending the same partition to stored-row ranges fixed
the one-table dense EOR case: in a loaded fp64 run, complete proving fell from
1.418 to 1.065 seconds, Stage 1 from 305 to 150 ms, and two-round materialization
from 47.5 to 19.3 ms. A rejected logical-row parallel constructor scattered
bit-reversed writes and measured 34.8 ms instead of about 6 ms for the
sequential stored-row traversal.
The exact current source also passes the separate Ice Lake one-worker gate: the
pinned fp32 D128 proof measured 3.395 seconds traced and 3.365 seconds on a warm
untraced repeat, compared with 3.493 seconds for the prior accepted head. The
proof remained 77,819 bytes and verified in both runs. This confirms that the
worker-aware policy restores normal-worker throughput without replacing the
selected one-worker AVX-512 operations.

The next fp32 Stage 1 slice removes the temporary choice between serial SIMD
and parallel scalar arithmetic for materialized affine rounds. The canonical
NEON, AVX2, and AVX-512 product and polynomial kernels already accept
coefficient slices. The prover now partitions SIMD-width-aligned row ranges,
calls that same selected kernel once per Rayon task, and merges only the five
reduced round coefficients. No target-specific arithmetic was copied into the
prover and no table payload is moved. On a busy 16-worker Apple Silicon run,
the targeted materialized product rounds fell from 19.4 ms to 13.2 ms and the
materialized polynomial rounds fell from 13.5 ms to 10.8 ms. Unrelated spans in
that run were system-load contaminated, so complete-proof acceptance is left to
the isolated PR benchmark rather than inferred from that sample.

### Existing state changes

The cutover evolves current owners rather than adding parallel wrappers:

| Current owner | New materialized state |
|---|---|
| `ExtensionOpeningTables::Dense` | witness and factor `EvaluationTable<F, E>` values |
| `SparseExtensionOpeningWitness` | index sidecar plus one `EvaluationTable<F, E>` for materialized values; bounded private palette while repeated merge-free values remain compact |
| `SparseFactor::Dense` | `EvaluationTable<F, E>` |
| `LowBasisRangeImageStorage::FoldedOctets` | one `u16` class code per row plus bounded class value and Taylor tables until the next sparse low-variable fold |
| `LowBasisRangeImageStorage::Materialized` | one row-oriented `Vec<E>` retained by the measured low-basis exception |
| Stage 2 `WitnessState::FoldedSuffix` | `EvaluationTable<F, E>` |
| Stage 2 factors and trace values | factored vectors while factors remain separable; `EvaluationTable<F, E>` only for a full live row table |
| Stage 3 product tables | `EvaluationTable<F, E>` where they are folded multilinear tables |

`ExtensionOpeningReductionTerm<F, E>` and
`ExtensionOpeningReductionProver<F, E>` name both fields directly. Production
tensor sources construct the binding-order witness table directly, and dense
term construction fuses factor materialization with the first product round.
The prover receives its independently derived input claim from the protocol
boundary. It does not convert tables back to row values or repeat the product
sum.

Small coefficient lists, round polynomials, challenge vectors, lookup tables,
and proof values remain `Vec<E>` or fixed arrays. `EvaluationTable` is only for a
row indexed materialized evaluation set.

### Migration rule

A protocol state changes directly from its current representation to the new
one. The implementation must not add `PackedTable`, `FieldPlanes`, `LaneStorage`,
or a scalar and packed enum. Temporary conversion code may exist only in tests,
small public API boundaries, and the measured Stage 2 handoff. It must not
survive in a production round loop or as parallel protocol state.

## Evaluation

### Acceptance criteria

- [x] `ExtField` provides allocation free coefficient construction and access
  for the identity field, `FpExt2`, `FpExt4`, and `FpExt8`.
- [x] `EvaluationTable` enforces its length, stride, coefficient count, and
  bounds invariants through private fields and focused tests.
- [x] Dense binding order conversion and every later fold match the current LSB
  first scalar evaluation for random tables at 1 to 20 variables.
- [x] A production portable x86 release detects AVX2 and AVX512 at runtime.
- [x] A production aarch64 release detects NEON at runtime.
- [x] Safe production callers cannot construct or forge a target specific plan.
- [ ] Scalar and every supported CPU operation produce identical round
  polynomials, folded tables, final evaluations, proof bytes, and transcript
  events.
- [x] Dense EOR uses the canonical table and operation set without round loop
  allocations.
- [x] Root sparse EOR stores coefficients once and either keeps its index
  sidecar current or retains documented original rows behind the private suffix
  precontraction until one exact restoring fold.
- [x] Sparse tensor EOR groups arbitrary merging pairs by shared suffix and
  folds, compacts, and computes the next round in one traversal.
- [x] Repeated sparse EOR values remain compact while the tensor suffix
  precontraction is active and materialize once before the ordinary transition.
- [x] Stage 2 uses the canonical table for its folded witness, retains separable
  factors without materializing full rows, and confines its measured transient
  row buffer to the compact handoff.
- [x] Stage 1 keeps i8 and i16 values compact. High-basis leaves materialize
  directly into the canonical table; low-basis leaves retain the measured
  row-oriented exception documented above.
- [x] Stage 3 materialized multilinear tables use the same representation or a
  benchmark documents why a named small table remains scalar.
- [ ] fp64 keeps the measured faster operation on each supported architecture.
  Apple Silicon is measured. Ice Lake remains pending. The Apple Silicon fp128
  path is 1.65 percent faster than the old path.
- [ ] No proof size, setup size, schedule, security estimate, or verifier timing
  changes beyond benchmark noise.
- [ ] Release assembly for each accepted SIMD operation contains the expected
  vector instructions and has no allocation or per row dispatch.
- [x] The pinned fp32 and D128 one hot proof benchmark records baseline and final
  setup, commit, prove, verify, proof byte, and per protocol sumcheck times.

### Testing strategy

`EvaluationTable` tests compare all constructors and row access against ordinary
extension values. They cover empty sparse value tables, one row, powers of two,
non power of two sparse row counts, truncation, invalid coefficient indices, and
conversion back to evaluations.

Dense fold tests use the current scalar logical order as the oracle. They compare
one round, all rounds, the final multilinear evaluation, and fused next round
coefficients.

Sparse tests cover duplicate normalization, zeros, sorted and unsorted input,
merge free folds, the first merging fold, and final evaluation. They compare the
new index sidecar and value table against the current pair representation before
that representation is removed. A repeated-value differential crosses the exact
`3 -> 2 -> 1 -> merge` boundary and compares every cached round and final value
against a fully materialized dense-factor term.

Forced plan tests run scalar versus every host supported plan. Cross architecture
CI covers x86 portable, x86 AVX2, x86 AVX512 when available, aarch64 NEON, and a
scalar build. End to end tests compare serialized proof bytes and transcript
logs.

The cheap iteration checks are focused `akita-field`, `akita-sumcheck`, and
`akita-prover` tests in the development profile. Before review, run the exact
repository format, dependency, Clippy, and test commands required by `AGENTS.md`
and the current CI workflow.

### Performance method

Use `quang-sumcheck-avx512` for Ice Lake measurements. Pin the benchmark to CPU 0
with `taskset -c 0`. Set Rayon and every linked thread pool to one worker for
kernel comparisons. Keep work off the sibling SMT thread.

Record these rows for each meaningful slice:

1. Scalar reference operation.
2. AVX2 operation.
3. AVX512 IFMA operation.
4. Conversion or materialization cost when the operation introduces one.
5. End to end proof time with the same build profile and proof parameters.

Inspect release assembly for the hot loop. Use hardware counters when two
implementations are close or when AVX512 does not improve over AVX2. Check vector
instruction count, loads, stores, branches, cache misses, and downclock effects.

The first success gate is not an estimated speedup. The exact 16,384 row dense
fp32 fold benchmark must have a median no greater than 75 us on the Ice Lake test
machine, with no conversion inside the measured round. The complete success gate
is a lower pinned end to end prove time with identical proof bytes.

## Alternatives considered

### Keep Vec<E> and pack every round

Rejected. The measured adjacent pair conversion path improves the fold by only
1.25 times, compared with 8.23 times when the table already has the correct
shape.

### Store architecture specific packed words

Rejected. SIMD width would become part of persistent storage. AVX2, AVX512,
NEON, and scalar builds would need different table types or conversion paths.
Coefficient first base field sections give every target the same bytes.

### Keep both scalar and SIMD copies

Rejected. It doubles the largest table payload, creates cache pressure, and
allows the copies to disagree after a fold.

### Add a generic table iterator layer

Rejected. The important operations fuse folding, multiplication, accumulation,
and next round work. A callback or iterator boundary makes those fusions harder
to inspect and can hide allocations.

### Put sumcheck operations in akita-field

Rejected. The field crate should provide arithmetic and coefficient access. It
should not know variable binding order, sparse EOR indices, or Stage 1 relations.

### Make SIMD a compile time feature

Rejected for production. Released binaries must use the host CPU without a
native rebuild. Compile time native builds remain useful for assembly and upper
bound experiments.

## Execution

1. Land the specification and archive the superseded packed sumcheck spec.
2. Land `ExtField` coefficient primitives and `EvaluationTable` with scalar
   tests. Do not migrate protocol states in this slice.
3. Add binding order conversion, scalar fold, product round, and fused fold plus
   next round operations. Treat these as the correctness oracle.
4. Add opaque runtime detection and fp32 AVX2 and AVX512 operations. Add NEON in
   the same operation structure.
5. Move dense EOR to the table and operation set. Measure before continuing.
6. Move root sparse EOR. Keep values coefficient first and indices separate.
   Group arbitrary tensor pairs by suffix and fuse compaction with the next
   round instead of gating the operation on a global merge-free flag. Keep a
   bounded repeated-value palette compact only while two proven merge-free
   rounds remain, then materialize it once.
7. Move Stage 2 dense witness, factor, and trace tables.
8. Move Stage 1 materialized state. Add direct i8 and i16 kernels and direct
   materialization.
9. Move remaining Stage 3 materialized multilinear tables.
10. Measure fp64 scalar versus SIMD and preserve the winner. Confirm fp128
    parity.
11. Run forced plan differential tests, byte identical proofs, release assembly
    checks, and pinned end to end benchmarks.

Each numbered slice ends with a focused commit and push. A slice may be split
when its tests or review surface become too large, but it may not introduce a
temporary public representation or duplicate production operation.

## Documentation

This specification is the source of truth while implementation is active. The
old packed sumcheck specification is archived as superseded. Live references
must point here.

When the implementation lands, set this status to `implemented`, add the PR, and
update `book/src/how/optimizations.md` with the durable table and runtime
selection design. Then set `Book-chapter` and archive this specification under
the policy in `specs/PRUNING.md`.

## References

- [`eor-streamed-prover.md`](eor-streamed-prover.md) describes EOR source and
  factor materialization. Its production tables must use this representation.
- [`relation-range-image-sumcheck.md`](relation-range-image-sumcheck.md)
  describes the Stage 2 relation and evaluation trace.
- [`packed-sumcheck.md`](archive/2026-Q3/packed-sumcheck.md) is the superseded
  proposal and contains older experiments and prior art.
- `crates/akita-field/src/packed/` contains the current packed field arithmetic.
- `crates/akita-prover/src/protocol/extension_opening_reduction/` contains EOR.
- `crates/akita-prover/src/protocol/sumcheck/` contains Stage 1, Stage 2, and
  Stage 3 prover state.
- `crates/akita-sumcheck/src/drivers/` contains the unchanged transcript drivers.
