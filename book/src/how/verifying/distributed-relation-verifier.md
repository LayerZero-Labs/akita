# Multi-chunk relation verification

Multi-chunk mode lets several prover workers own different live block ranges.
Verification is still performed by one ordinary verifier. The protocol changes
the witness column layout, not the verifier's deployment model.

This chapter records the differences from
[Relation matrix and witness layout](./matrix_evaluation.md).

## Canonical block ownership

For `B` exact live blocks and `C` chunks, chunk `c` owns

```math
\mathcal I_c
=
\left[
\left\lfloor\frac{cB}{C}\right\rfloor,
\left\lfloor\frac{(c+1)B}{C}\right\rfloor
\right).
```

Supported chunk counts are powers of two no greater than 64. The ranges are
contiguous, nested across finer supported partitions, and cover `[0,B)` once.
Their lengths differ by at most one.

The layout does not round `B` up. If `C > B`, repeated boundaries create empty
chunks.

## Physical unit order

The witness is chunk-major. Within each chunk, groups follow authenticated
relation order:

```text
chunk 0: [group 0 Z|E|T] [group 1 Z|E|T] ...
chunk 1: [group 0 Z|E|T] [group 1 Z|E|T] ...
...
shared quotient and compression suffix
```

Unit offsets are prefix sums because different groups and chunks can have
different widths. Every consumer reads these offsets from `WitnessLayout`.

## Partitioned and replicated data

`E` and `T` are partitioned. A unit contains values only for the blocks in its
exact range. Across all chunks, those segments cover the same semantic blocks
as single-chunk mode.

`Z` is replicated. Each worker produces a partial folded response in the full
folded response space. The verifier sums the contribution of every `Z` segment.
An empty block range still has a physical `Z` segment, whose honest value is
zero.

The quotient and compression suffix are shared once for the complete relation.

## Global and local block indices

Inside a unit, `E` and `T` use a local block index starting at zero. Fold
challenges and opening block weights use the corresponding global block index.
If the unit begins at `S_c`, local block `beta` names global block
`S_c + beta`.

This distinction is checked at layout construction and appears explicitly in
the structured setup and evaluation trace descriptors.

## Verifier work

Partitioned `E` and `T` terms still cover `B` blocks in total. Their arithmetic
does not grow with the chunk count beyond bounded descriptor and address work.

Replicated `Z` work grows with `C` because the witness contains `C` copies of
that segment. This is the intended cost of avoiding a prover-side all-reduce.

The public A, B, and D setup matrices are evaluated once at `alpha`. The
verifier does not rescan or reevaluate the setup matrix for each chunk.

## Compact affine runs

When several units have equal shapes and constant offset differences, the
verifier combines them into one tensor axis. This often applies when `B` is
divisible by `C`.

Unequal dyadic ranges are not padded. Setup contribution tensors keep their
exact unit families when shapes differ. Evaluation trace preparation groups
only equal affine runs and leaves irregular units separate.

The compact and separate representations are two evaluations of the same
polynomial. Dense oracle tests cover uneven and empty ranges.

## Safety

The verifier rejects zero or non-power-of-two chunk counts and counts above the
public cap. It checks unit order, exact range coverage, physical coefficient
ranges, and proof capacity before final point evaluation. Empty units do not
cause unchecked indexing or allocation.

## Code map

- Canonical ranges: `crates/akita-types/src/witness/chunk_partition.rs`.
- Physical layout: `crates/akita-types/src/witness.rs`.
- Structured relation terms:
  `crates/akita-types/src/setup_contribution/plan/structured.rs`.
- Setup tensors:
  `crates/akita-types/src/setup_contribution/plan/setup_index_weight.rs`.
- Evaluation trace:
  `crates/akita-verifier/src/protocol/evaluation_trace.rs`.
- Prover ownership and witness assembly:
  `crates/akita-prover/src/protocol/fold_grind.rs` and
  `crates/akita-prover/src/protocol/ring_switch/coeffs.rs`.
