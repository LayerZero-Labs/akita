# Arithmetic microbenchmarks

Akita's complete profile is the main performance measure. Focused arithmetic
benchmarks answer a narrower question after that profile identifies a hot
kernel.

The `ntt_matvec` Criterion target measures public matrix multiplication without
setup construction, transcript work, planning, or the rest of the proof.

## Run the benchmark groups

```bash
cargo bench -p akita-pcs --bench ntt_matvec -- rank_ring_dim
cargo bench -p akita-pcs --bench ntt_matvec -- width
cargo bench -p akita-pcs --bench ntt_matvec -- equal_output
cargo bench -p akita-pcs --bench ntt_matvec -- equal_io
```

Use a shape filter for one quick comparison:

```bash
cargo bench -p akita-pcs --bench ntt_matvec -- d64_r4_w128
```

Prepared cache construction is outside the timed region. The measurement
includes digit validation, forward transforms, pointwise accumulation, inverse
transforms, CRT reconstruction, and output allocation.

## Compared kernels

Every common shape includes the production i8 commitment path and the unified
i16 path. The i16 cases cover digit bases L2 through L11 where supported. The
label states whether exact reconstruction uses only the base CRT residues or
also needs the optional i16 tail.

On an x86 host with AVX-512IFMA, eligible labels also include `ifma52`. This
path stores canonical 50 bit prime residues in `u64` lanes and uses the IFMA52
instructions for exact arithmetic.

The label applies only to the selected exact i16 cache and kernel. The ordinary
i32 NTT still dispatches to AVX2 on x86. The benchmark names these paths
separately so an IFMA result cannot be mistaken for a global NTT backend
change.

## Ring dimension and rank sweep

The `rank_ring_dim` group tests ring dimensions 64, 128, 256, and 512. At each
dimension it tests output ranks 1, 2, 4, and 8 with ring width 128.

This sweep changes both the scalar input size and the ring structure. Use it to
see which dimensions are competitive for a specific matrix geometry, not to
claim that a larger ring is always faster.

## Width sweep

The `width` group tests widths 128 through 1024 at ring dimension 64 and rank 4.
It shows how the kernel scales as the number of input ring columns grows.

## Equal output and equal input cases

The `equal_output` group compares these shapes:

```text
D64  rank 8
D128 rank 4
D256 rank 2
D512 rank 1
```

Each shape returns 512 field coefficients. Its scalar input size still grows
with the ring dimension.

The `equal_io` group also adjusts the width:

```text
D64  rank 8 width 1024
D128 rank 4 width 512
D256 rank 2 width 256
D512 rank 1 width 128
```

These shapes hold both scalar input and output sizes fixed. This isolates the
trade between ring structure, transform cost, and available parallel work.

## Understand the scaling

Let `n = width * D` be the scalar input length and `m = rank * D` be the scalar
output length. An ordinary dense matrix multiplication costs `O(m * n)`.

Akita stores the matrix as `rank * width` negacyclic ring blocks. With a
prepared matrix, the approximate work for each CRT residue is:

```text
input transforms:  n * log D
pointwise products: m * n / D
output transforms: m * log D
```

The ring structure divides the pointwise term by `D`. Larger rings also reduce
prepared matrix storage for a fixed scalar input and output shape.

Transform work grows with `log D`. Exact CRT bounds also grow with `D`, and a
shape with fewer ring rows may expose less parallel work. The generated planner
therefore chooses dimensions from measured and certified candidates instead of
always choosing the largest ring.

## Return to the complete proof

Criterion throughput counts `rank * width * D` coefficient products. The
complete profile adds setup, commitment assembly, sumcheck, transcript work,
proof encoding, and verification.

After a kernel change, run the matching complete mode from
[Profiling a workload](./profiling.md). The end to end result decides whether
the change improves Akita for its host application.
