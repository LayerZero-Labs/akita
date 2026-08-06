# Sumcheck evaluation tables and CPU kernels

| Field | Value |
|---|---|
| Author(s) | Quang Dao and Codex |
| Created | 2026-08-06 |
| Status | active |
| PR | |
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
   not keep a scalar `Vec<E>` beside a coefficient first copy.
5. The stored representation does not depend on SIMD width or CPU architecture.
6. Production release builds select SIMD at runtime. Compiling without
   `target-cpu=native` must not force the scalar path on a capable CPU.
7. Safe production code cannot request a CPU plan that the host does not support.
8. Compact i8 and i16 sources remain compact until the protocol needs a
   materialized table.
9. Production compact materialization writes directly into the canonical table.
   It must not build a full `Vec<E>` and then transpose it.
10. Full dense tables use binding order for their entire live lifetime. A fold
    does not change this invariant.
11. Sparse witness indices keep their current logical meaning. Row `j` in the
    sparse value table belongs to sparse index `indices[j]`.
12. SIMD and scalar accumulation obey the same exact modular arithmetic bounds.
    No optimization may rely on integer overflow or a reduction bound that the
    field type does not prove.
13. No hot round loop allocates. Folding shortens the live part of existing
    storage.
14. The payload size of a materialized table is
    `len * E::EXT_DEGREE * size_of::<F>()`. Small fixed metadata is allowed.
15. Rayon may divide a large table among workers. Each worker uses the same
    selected CPU kernel and a private accumulator. The merge result must remain
    field exact.

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
from_multilinear_coefficient_fn
```

They accept or generate logical LSB first rows and write binding order directly.
They require a nonzero power of two row count. Large production producers use
the function constructors so they never hold a second full representation. The
private bit reversal helper is shared by all three constructors.

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
```

The indices stay sorted according to the sparse algorithm's current logical
order. The value table has `indices.len()` rows. It is coefficient first but its
rows follow the index sidecar rather than dense binding order.

Duplicate normalization, zero removal, and merge detection remain one canonical
constructor operation. A constructor cannot accept independently prepared
indices and values that have not passed that normalization.

The root one hot EOR is the only path with the fixed support plateau seen in the
profile. Its 2^20 entries remain live for the first six folds of a 2^26 domain.
Recursive balanced digit witnesses do not use this sparse form. They use dense
tables and their work halves each round.

### Compact i8 and i16 states

Stage 1 and the compact prefix of Stage 2 keep their existing small integer
storage. A compact kernel may perform all of these operations without
materializing `E` values:

1. Load and widen signed i8 or i16 values.
2. Compute the round polynomial with extension field weights.
3. Apply one or two early challenges while the exact compact formula permits it.
4. Build the first required `EvaluationTable` directly in binding order.
5. Compute the next round while writing that table when a fused pass saves a
   second read.

AVX512 byte and word operations require AVX512BW. The current fp32 arithmetic
uses AVX512F and AVX512DQ. An operation that uses 128 or 256 bit AVX512 forms
must also check AVX512VL. Each target function declares and checks its exact
feature set. The runtime plan selects these operations independently. A machine
may therefore use AVX512 for fp32 table folds and AVX2 for a compact i8 or i16
operation if measurement shows that choice is faster.

NEON uses its widening operations for compact values. A compact path may remain
scalar on a target when its measured SIMD version does not win.

### Runtime CPU selection

`akita-prover` owns one opaque `SumcheckKernelPlan`. Production code obtains it
only through detection. Each sumcheck prover detects and stores one copy during
construction. The plan is not serialized, and a round does not repeat CPU
feature checks. The plan contains private function choices for each operation
rather than one public CPU tier.

The plan may select different implementations for:

| Operation family | Candidate implementations |
|---|---|
| fp32 extension table arithmetic | scalar, NEON, AVX2, AVX512 |
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

Small signed inputs may use wider integer accumulators and reduce less often.
Full field products may need a different accumulator. The operation plan may
select different accumulation code for compact first rounds and later full field
rounds.

The specification does not require a universal packed accumulator trait. Such a
trait would force unrelated fields and protocols into one shape. The canonical
sumcheck operation owns its accumulator and uses the field's existing reduction
primitives.

### Parallel execution

SIMD is the inner row loop. Rayon remains the outer table partition when the
table exceeds a measured threshold. Each partition begins and ends on a row
boundary. Each worker returns scalar field accumulators, and the operation merges
them exactly once.

The implementation must benchmark one worker and the normal parallel setting.
The one worker result shows kernel quality. The parallel result shows application
throughput. Neither result replaces the other.

### Field families

For `FpExt4<Fp32>`, the coefficient first table has four fp32 sections. This is
the main SIMD target and must have scalar, AVX2, AVX512, and NEON coverage where
the host architecture permits it.

For `FpExt2<Fp64>`, the table has two fp64 sections. The implementation must
measure the scalar wide path against SIMD before choosing a default operation.

For the fp128 identity field, the table has one fp128 section. Its memory payload
is the same as the current `Vec<Fp128>`. The initial operation remains scalar
unless a measured SIMD kernel wins end to end.

The table and scalar reference also support `FpExt8` if a configuration needs it.
This does not require an optimized `FpExt8` kernel in the first cutover.

### Existing state changes

The cutover evolves current owners rather than adding parallel wrappers:

| Current owner | New materialized state |
|---|---|
| `ExtensionOpeningTables::Dense` | witness and factor `EvaluationTable<F, E>` values |
| `SparseExtensionOpeningWitness` | index sidecar plus one `EvaluationTable<F, E>` for values |
| `SparseFactor::Dense` | `EvaluationTable<F, E>` |
| `LowBasisRangeImageStorage::Materialized` | `EvaluationTable<F, E>` |
| Stage 2 `WitnessState::FoldedSuffix` | `EvaluationTable<F, E>` |
| Materialized Stage 2 factors and trace values | `EvaluationTable<F, E>` where they are full live row tables |
| Stage 3 product tables | `EvaluationTable<F, E>` where they are folded multilinear tables |

Small coefficient lists, round polynomials, challenge vectors, lookup tables,
and proof values remain `Vec<E>` or fixed arrays. `EvaluationTable` is only for a
row indexed materialized evaluation set.

### Migration rule

A protocol state changes directly from its current representation to the new
one. The implementation must not add `PackedTable`, `FieldPlanes`, `LaneStorage`,
or a scalar and packed enum. Temporary conversion code may exist only in tests or
small public API boundaries and must not survive in a production round loop.

## Evaluation

### Acceptance criteria

- [ ] `ExtField` provides allocation free coefficient construction and access
  for the identity field, `FpExt2`, `FpExt4`, and `FpExt8`.
- [ ] `EvaluationTable` enforces its length, stride, coefficient count, and
  bounds invariants through private fields and focused tests.
- [ ] Dense binding order conversion and every later fold match the current LSB
  first scalar evaluation for random tables at 1 to 20 variables.
- [ ] A production portable x86 release detects AVX2 and AVX512 at runtime.
- [ ] A production aarch64 release detects NEON at runtime.
- [ ] Safe production callers cannot construct or forge a target specific plan.
- [ ] Scalar and every supported CPU operation produce identical round
  polynomials, folded tables, final evaluations, proof bytes, and transcript
  events.
- [ ] Dense EOR uses the canonical table and operation set without round loop
  allocations.
- [ ] Root sparse EOR stores coefficients once and keeps its index sidecar in
  sync through merge free and merging folds.
- [ ] Stage 2 uses the canonical table for its folded witness, factors, and full
  trace values.
- [ ] Stage 1 keeps i8 and i16 values compact and materializes directly into the
  canonical table.
- [ ] Stage 3 materialized multilinear tables use the same representation or a
  benchmark documents why a named small table remains scalar.
- [ ] fp64 keeps the measured faster operation. fp128 has no more than a 2
  percent one worker regression.
- [ ] No proof size, setup size, schedule, security estimate, or verifier timing
  changes beyond benchmark noise.
- [ ] Release assembly for each accepted SIMD operation contains the expected
  vector instructions and has no allocation or per row dispatch.
- [ ] The pinned fp32 and D128 one hot proof benchmark records baseline and final
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
that representation is removed.

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
3. AVX512 operation.
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
