# Prover Optimizations

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
shape data. `OneHotView` validates the runtime ring dimension and exposes the
logical source to a backend.

## One group commitment operation

The protocol always commits a group. A singleton is a group with one source.
`RootCommitKernel::commit_inner_group` is therefore the only root commitment
operation. Dense, one hot, sparse ring, tensor projection, and recursive
witness sources implement the same source typed boundary without sharing a
CPU representation.

The CPU one hot path runs this flow:

```text
validated OneHotView values
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

This ownership also applies to opening and tensor projection. Opening builds
only the active challenge range. Batched opening keeps only the current
polynomial and its current block tile. Tensor projection builds one sparse
ring polynomial and moves it into the returned projection object. Cloning a
one hot polynomial does not share mutable derived state.

## Tiling and sweep selection

Tile size and arithmetic traversal solve different problems.

The tile size bounds temporary memory. The default CPU backend uses one 8 MiB
sparse commitment scratch budget per worker. One-hot and signed sparse-ring
commitments both use it. An application may choose another nonzero budget.
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
  active columns is widened once.

Both sweeps consume the same flat sparse entries and produce the same rows.
The private selector uses total block count, active column count, and worker
count. Correctness tests compare both retained sweeps with a direct arithmetic
reference. A release benchmark measures the retained choices over the
production region. The profile report records the selected sweep and tile so
a route change is visible even when total runtime is noisy.

## CPU resource limits

`CpuBackend` owns two deployment limits. The first is the largest ring switch
operation that keeps a complete transformed matrix prefix. The second is the
sparse commitment scratch budget for each worker. `CpuBackend::DEFAULT` uses `2^21` ring
elements and 8 MiB. Applications may use `CpuBackend::with_resource_limits` to
choose other values.

A zero ring switch limit streams every ring switch operation that has a
streamed implementation. `usize::MAX` retains every supported operation. The
commitment scratch budget must be nonzero. Each one-hot or sparse-ring kernel
returns `InvalidSetup` before its tile allocation if even one block cannot fit.

These limits choose equivalent CPU execution paths. They do not change the
proof schedule, transcript, setup bytes, proof bytes, or verifier behavior.
The CPU backend still selects the private one hot arithmetic sweep.

## Wide accumulation

Commitment kernels accumulate several products before reducing to the base
field. The safe number of additions depends on the field and CRT profile.
`F::MAX_COMMIT_ACCUMULATIONS` is the single contract for this limit.

If a one-hot block contains more terms than the limit, both retained sweeps
reduce the wide accumulator at the same cap and continue from zero. Ring shift
helpers preserve negacyclic wrap. Tests reach the exact addition cap with
maximal canonical limbs and cover wrapped shifts.

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

Retention is the default. A caller with an isolated root owner may apply
`ReleaseRootNttAfterFold`. Release removes every built shared matrix key once
per physical owner. Existing readers remain valid through shared ownership. A
later smaller request builds the smaller exact extent instead of reviving an
empty covering slot. Small compression NTT entries remain resident and are
reused after this boundary.

`CpuPreparedSetup::shared_ntt_cache_bytes` and
`compression_ntt_cache_bytes` report each namespace. `ntt_cache_bytes` returns
their checked sum. Planned requirement metrics describe only shared matrix
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

## Design boundary

CPU storage plans remain private. A new backend implements the source typed
operation for its device and chooses its own storage, tiling, and scheduling.
It must preserve source order, checked geometry, arithmetic limits, and the
protocol output type. It must not absorb transcript state or expose a CPU row
plan as public API.

See [Compute Backends](../../../docs/compute-backends.md) for backend ownership
and NTT lifecycle details. The full PR 375 design record is
[`specs/archive/2026-Q3/pr375-prover-streaming-and-onehot-unification.md`](../../../specs/archive/2026-Q3/pr375-prover-streaming-and-onehot-unification.md).
