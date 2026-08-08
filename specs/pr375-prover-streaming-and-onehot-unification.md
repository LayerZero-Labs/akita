# Spec: PR 375 Prover Streaming and One Hot Unification

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-08-08 |
| Status        | active |
| PR            | https://github.com/LayerZero-Labs/akita/pull/375 |
| Supersedes    | |
| Superseded-by | |
| Book-chapter  | |

## Summary

PR 375 reduces prover memory and repeated setup work. Its main changes fuse
one hot root commitments, stream large ring relation and quotient inputs, make
prepared NTT retention explicit, and remove avoidable copies in fold output.

The current one hot implementation achieves the batch memory goal, but it
mixes the logical witness, derived block geometry, storage policy, and CPU
sweep policy. This added several types and moved the fast singleton workload
from the bucketed column sweep to the slower merge sweep. The fixed D64
singleton benchmark changed from 69.19 percent faster than main to parity.

This spec defines the desired end state for PR 375. The one hot source view is
the only source boundary. The root commit kernel is the only commitment
boundary. The CPU implementation derives flat sparse ring entries for one
operation tile at a time and privately chooses the best measured sweep. The
polynomial owns no mutable derived storage. The remaining PR changes follow
the same ownership rule. Derived memory belongs to the operation that builds
it or to an explicit prepared setup owner.

## Intent

### Goal

Complete the source typed prover compute cutover for one hot commitment while
preserving the memory, ring relation, quotient, NTT, and copy reduction gains
of PR 375.

After freezing evidence, the implementation first closes the independent NTT
and unsafe audit gaps found in review. The one hot cutover then remains the
main implementation sequence. The other PR changes stay in scope because the
final branch must have one coherent ownership and validation model, but they
do not need to share one implementation abstraction.

### PR Scope Map

| Area | Problem solved | Desired owner |
|------|----------------|---------------|
| One hot commitment | Repeated matrix passes and retained block data | Source typed group kernel with operation local blocks |
| One hot opening and tensor work | Hidden clone shared derived state | One hot view plus operation local storage |
| Wide accumulation | Repeated reduction and ring shift overhead | Field accumulator contract and ring primitives |
| Ring relation and quotient work | Matrix sized transformed inputs and repeated validation | One validated plan with cached or streamed execution |
| NTT state | Oversized or stale prepared transforms | Prepared setup owner with explicit release policy |
| Exact prefix and relation preparation | Sequential independent work | Existing table and relation functions with safe parallel ranges |
| Fold witness output | Copies between equivalent owned containers | Existing output types with consuming constructors |
| Verifier setup scan | Reading a padded prefix that is not used | Stage 3 required prefix calculation |
| Profile harness | One file owned unrelated scenarios | Scenario modules with unchanged entry points |

### Problems

#### One hot commitment changed arithmetic strategy by accident

At commit `a9f3c29678`, the fixed D64 singleton benchmark used eager blocks and
the bucketed column sweep. It committed in 0.424 seconds while the matching
main revision took 1.377 seconds.

At commit `2d17e55757`, homogeneous one hot groups, including a group with one
polynomial, entered the new multi source path. Eager and lazy storage then
shared the merge sweep. The same benchmark shape took 1.257 seconds while main
took 1.251 seconds.

Lazy tile construction reduced retained memory. It did not require the merge
sweep. The code joined two independent choices and lost the fast traversal.

#### The derived representation has too many axes

The current one hot commit path combines two geometry variants with two
storage variants:

```text
SingleChunkEntry or MultiChunkEntry
    crossed with
Eager or Lazy storage
```

The cross product appears in `OneHotBlocks`, `OneHotCommitBlocks`,
`OneHotBlockSource`, `OneHotBlockRange`, and a boxed `LazyOneHotBlocks`
builder. CPU code then dispatches across the same variants again.

The logical witness does not have these variants. Every nonzero chunk defines
one flat field position. The ring, block, position, and coefficient are derived
from that position by one formula.

#### The public backend boundary exposes an unusable internal plan

`CommitmentComputeBackend::onehot_commit_rows` accepts the public
`OneHotCommitRowsPlan`. A lazy source inside that plan can only be materialized
through crate private methods. An external backend can observe the number of
blocks but cannot read the block data.

The existing `RootCommitKernel<Source, F, D>` design already solves this
problem. A backend should receive a semantic source view and choose its own
storage and execution policy. The one hot row plan duplicates that boundary.

#### Group commitment is optional at the wrong layer

`RootCommitKernel` currently requires the singleton operation and provides a
default group loop. The delegating CPU backend forwards the singleton method
but not the group override. It can therefore erase a fused group
implementation without a compiler error.

The protocol commits a group. A singleton is a group containing one source.
The group operation must be the canonical method.

#### `OneHotPoly` owns mutable derived state

`OneHotPoly` contains clone shared block and tensor caches behind mutexes.
Cloning the polynomial shares cache mutations. Clearing one clone changes the
state seen by another clone. A poisoned lock can make a pure prover operation
fail.

The explicit preparation methods have no production callers in this tree.
Ordinary commitment already bypasses the block cache. The cache API therefore
adds ownership and error behavior without protecting the main workload.

#### Performance policy depends on machine topology

The current sweep threshold divides block count by the Rayon thread count. The
same commitment can select a different arithmetic strategy when the machine
reports a different number of workers. The threshold also controls whether a
whole source or one tile is materialized.

Memory tiling and arithmetic strategy need separate decisions.

#### CI reports performance but does not protect it

The profile workflow posts median comparisons. It does not fail when a
targeted optimization loses its benefit. Correctness tests prove that merge
and bucketed sweeps agree. They do not prove that production selects the right
one.

#### The NTT plan and runtime disagree about memory

The CPU backend streams large relation and quotient transforms, but the
backend independent prewarm plan still requests complete prepared slots for
those operations. A caller that prewarms therefore restores the allocations
that runtime streaming was meant to avoid.

Release has a second mismatch. Clearing a built slot while retaining its
oversized cache key lets a later smaller request select that empty key and
rebuild the larger allocation. Release must remove the ability to rebuild the
released extent by accident.

Preparation, execution, and release need one description of whether an NTT
operation is retained or streamed.

#### Streaming is incomplete before source materialization

The current decomposition paths build complete one hot block tables before
they limit work to active challenges. The batch path builds those tables for
every polynomial at once. Dropping the tables afterward lowers retained
memory, but it does not bound peak memory.

The active range and tile must be chosen before entries are derived. Peak
derived storage must scale with the current tile, not with every available
block or every polynomial in the group.

#### Repeated low level paths can drift

The quotient kernel repeats cached and streamed traversal as well as the
`q32`, `q64`, and `q128` matrix slicing. The one hot column sweep also has
separate production and test scheduling paths. These are boundary sensitive
loops. Shared validation and range planning must have one owner, and tests
must invoke the production scheduler.

### Invariants

#### Protocol invariants

- The verification equations do not change.
- Transcript labels and challenge order do not change.
- Proof bytes and setup bytes do not change.
- Schedule selection and security bounds do not change.
- Every supported dense, one hot, and sparse ring proof that verifies before
  this refactor still verifies afterward.
- Verifier reachable malformed input returns a typed error and never panics.

#### Source and ownership invariants

- `OneHotPoly` is the sole owner of the logical chunk indices.
- `OneHotView<F, D, I>` is the sole borrowed source for one hot kernels at
  dimension `D`.
- `CommitInnerPlan` is the sole owner of scalar root commit parameters.
- One function owns the mapping from a one hot index to a sparse ring block
  entry.
- Derived one hot blocks are local to the operation range that requested them
  and are dropped when that range finishes.
- A clone of `OneHotPoly` does not share mutable derived state.
- A backend receives semantic source data. It does not receive a boxed CPU
  block builder.

#### Commitment invariants

- Group commitment is the only root commitment operation exposed by
  `RootCommitKernel`.
- A singleton uses the group operation with one source.
- Every source in a same shape one hot group multiplies the same matrix view.
- Results remain in source order.
- Mixed dense and one hot groups preserve their current result order and typed
  errors.
- Accumulator reduction occurs before the field specific addition cap.
- Arithmetic strategy does not determine storage lifetime.

#### Other PR 375 invariants

- Cached and streamed ring relation kernels return the same rows.
- Cached and streamed CRT quotient kernels return the same rows and quotient
  witnesses.
- Prepared NTT state is retained by default.
- Explicit release deduplicates physically shared prepared owners.
- Releasing prepared NTT state does not invalidate an active reader.
- A released NTT slot leaves no empty compatible key that can rebuild a larger
  allocation for a smaller request.
- Prewarm and runtime use the same retained or streamed decision. Prewarm does
  not materialize a full slot for an operation that runtime will stream.
- Active challenge bounds are computed before one hot block materialization.
  Batch peak derived storage is bounded by the current tile.
- Every `FlatBlocks` offset conversion is checked before the offset table is
  modified.
- Every `unsafe` block added or changed by this PR has a complete adjacent
  `// SAFETY:` argument for bounds, alignment, and aliasing.
- New public algebra helpers have a production caller and a documented
  invariant. Otherwise they are removed.
- Exact prefix folding agrees with dense folding at every tested boundary.
- Fold witness construction does not copy coefficients merely to change their
  container type.
- The verifier scans only the setup prefix required by Stage 3.
- The profile workload split does not change scenario behavior or public
  profile entry points.

### Non Goals

- This spec does not change the Akita protocol or planner policy.
- This spec does not change transcript or serialization formats.
- This spec does not add a public commitment strategy flag.
- This spec does not add a separate streaming trait.
- This spec does not force dense, one hot, and sparse ring arithmetic into one
  generic kernel when their accumulator rules differ.
- This spec does not add speculative prepared one hot storage.
- This spec does not promise support for new schedule shapes. The canonical
  coordinate formula may support more `K` and `D` relationships, but new public
  support requires its own validation and security review.
- This spec does not restore the planner payload slack changes that were
  removed from PR 375.

## Evaluation

### SumChecker Review Disposition

The review at the current PR head is part of this specification. A finding is
not resolved merely because its present implementation compiles. It is
resolved when the final design below removes the cause or satisfies the stated
acceptance test.

| Finding | Final disposition | Slice |
|---------|-------------------|-------|
| 1. Released NTT keys | Remove released key eligibility and test a large release followed by a smaller request. | 1 |
| 2. Prewarm defeats streaming | Compile prewarm and runtime retention from one execution decision. Streamed work creates no full slot. | 1 |
| 3. Full block materialization | Choose active ranges first and materialize only the current polynomial tile. | 2 |
| 4. Unusable lazy public plan | Delete the lazy plan boundary. External kernels consume `OneHotView`. | 3 |
| 5. Stale PR contract | Use this spec as the contract, then update the PR body with head pinned evidence. | 7 |
| 6. Duplicate decomposition | Delete the single and multi chunk split. One flat entry builder and one range fold own the algorithm. | 2 |
| 7. Duplicate schedulers | Keep one production scheduler and test it directly. Reference code only compares results. | 4 |
| 8. Duplicate quotient indexing | Centralize validation, chunk ranges, and matrix slicing. Width specific code owns arithmetic only. | 6 |
| 9. Unused wide helpers | Remove them unless the canonical production accumulator uses them. | 6 |
| 10. Incomplete NEON safety notes | Add local complete safety arguments or centralize the unsafe operation. | 1 |
| 11. Unchecked flat offsets | Use checked conversion and a boundary test before the unified table is adopted. | 2 |
| 12. Undocumented cache lifecycle | Document ownership, concurrency, prewarm, release, reuse, and cleanup for the remaining NTT cache. One hot caches are deleted. | 1 and 7 |

### Acceptance Criteria

#### One hot source and representation

- [ ] `OneHotPoly` contains logical witness metadata and indices, but no block
      cache or tensor projection cache.
- [ ] `OneHotView` validates the selected ring dimension and exposes the
      semantic indices, chunk size, and variable count needed by an external
      backend.
- [ ] One canonical builder maps a requested block range to
      `FlatBlocks<SparseRingBlockEntry>`.
- [ ] `SingleChunkEntry`, `MultiChunkEntry`, and the `OneHotEntry` trait are
      removed.
- [ ] `OneHotBlocks`, `OneHotCommitBlocks`, `OneHotBlockSource`,
      `OneHotBlockRange`, and `LazyOneHotBlocks` are removed.
- [ ] `SparseRingBlocks` is replaced by the existing generic `FlatBlocks`.
- [ ] The one hot mapping handles every currently supported `K` and `D` shape,
      including several chunks contributing to one ring.
- [ ] No new one hot struct is added unless its review documents the invariant
      it owns and the existing types it replaces.

#### Commitment boundary

- [ ] `RootCommitKernel` exposes one group commitment method. It does not have
      a second singleton commitment method.
- [ ] The parameter based and profile based commitment paths both call the
      group method.
- [ ] Delegating backends must implement or forward the group operation. The
      default singleton loop cannot silently replace it.
- [ ] The one hot `RootCommitKernel` implementation consumes `OneHotView`
      directly and does not lower through `OneHotCommitRowsPlan`.
- [ ] The one hot methods and plans are removed from
      `CommitmentComputeBackend` and the crate root exports.
- [ ] `CommitBackendFor` requires the source typed root kernel and the shared
      digit row capability it actually uses. It does not require unrelated
      representation methods.
- [ ] `CommitmentComputeBackend` is removed. Dense, one hot, sparse ring, root
      projection, and recursive witness commitment use their source typed
      kernels directly.
- [ ] `DenseCommitRowsPlan`, `SparseRingCommitRowsPlan`, and
      `RecursiveWitnessCommitRowsPlan` are removed. A real input sum type such
      as `DenseCommitInput` may remain as an internal representation detail.

#### CPU execution

- [ ] One private CPU module owns one hot commitment. It has one group entry
      function. Its helpers own tiling and the retained arithmetic kernels.
- [ ] Tile size follows an explicit per worker scratch memory budget.
- [ ] Sweep selection uses workload measures such as block count, term count,
      matrix width, output rank, and ring dimension. It does not use blocks per
      Rayon thread as its only policy.
- [ ] Direct, bucketed, and merge sweeps have one common flat entry type.
- [ ] Every retained sweep has a measured workload where it wins. A sweep with
      no winning region is deleted.
- [ ] A private sweep enum is added only when at least two retained strategies
      need a testable selector. If one direct branch is enough, no enum is
      added.
- [ ] Strategy, tile size, hot term count, and estimated matrix passes are
      visible in tracing or test statistics.
- [ ] Tests call the production scheduler. No test only scheduler duplicates
      production traversal or merge decisions.
- [ ] Opening and decomposition compute the active block range before building
      entries. A batch holds at most the current configured polynomial and
      block tile.

#### Performance

- [ ] The fixed D64 singleton benchmark selects the bucketed or another
      measured best strategy. It does not enter the merge sweep merely because
      its source group has length one.
- [ ] On the benchmark runner, the selected singleton strategy is within 10
      percent of the fastest retained strategy for the same shape.
- [ ] The adaptive singleton profile does not regress by more than 5 percent
      against the merge base median.
- [ ] The fixed D64 singleton profile regains a material advantage over the
      merge base. The target is at least 40 percent lower commit time on the
      interleaved CI comparison.
- [ ] The wide one hot batch keeps the PR memory improvement. Peak RSS must not
      regress by more than 10 percent from the best PR 375 measurement for the
      same shape.
- [ ] The multi polynomial one hot profile remains at least 40 percent faster
      than its interleaved merge base result unless a new baseline is approved
      with evidence.
- [ ] The benchmark report records the selected sweep and tile size so a route
      change is visible even when total runtime is noisy.

#### Remaining PR changes

- [ ] Streamed relation and quotient tests compare against their cached
      reference kernels across every supported field profile that exercises a
      distinct CRT capacity.
- [ ] Shared quotient validation, bounds, and chunk planning have one owner.
- [ ] Cached and streamed quotient modes consume the same typed chunk and
      matrix slice plan. The `q32`, `q64`, and `q128` branches contain only the
      arithmetic that must differ by width.
- [ ] NTT retention remains the default. Explicit root release reports errors,
      deduplicates owner identities, and preserves active readers.
- [ ] Releasing a large NTT slot and then requesting a smaller compatible
      extent does not rebuild the large allocation.
- [ ] The requirements compiler marks an operation retained or streamed using
      the same policy as runtime. Prewarming a streamed operation leaves no
      complete relation or quotient slot resident.
- [ ] Exact prefix parallel folding has dense differential tests that cross
      every sequential and parallel wave boundary.
- [ ] Wide shift accumulation tests cover negacyclic wrap and addition limits.
- [ ] Every new public `WideCyclotomicRing` helper is used by production code
      and documents its invariant, or is removed.
- [ ] Every changed NEON `unsafe` block has one adjacent complete safety
      argument. Repeated unchecked access is centralized when that reduces the
      audited surface.
- [ ] `FlatBlocks` uses checked offset conversion and returns a typed error
      before changing its offsets when the entry count is not representable.
- [ ] The profile workload remains below the repository file line cap without
      pass through facade functions.
- [ ] The exact production mixed dimension profile commits, proves, and
      verifies with the intended NTT lifecycle policy.

#### Compatibility and documentation

- [ ] Proof serialization, setup serialization, transcript schedules, and
      verifier acceptance are byte identical to the merge base for fixed test
      fixtures.
- [ ] Removed public prover types are added to the documentation dead symbol
      guards.
- [ ] `specs/akita-polyops-cutover.md` is updated to reflect the completed
      source typed commitment boundary.
- [ ] The cache ownership text in
      `specs/small-field-prover-opening-optimization.md` points to this spec for
      the current one hot ownership rule.
- [ ] The PR description reports the final architecture and current benchmark
      results rather than the superseded eager and lazy plan design.
- [ ] All repository documentation guardrails pass.

### Testing Strategy

#### One hot coordinate tests

Generate random valid one hot indices and compare the canonical flat entry
builder with dense materialization. Cover:

- Empty chunks.
- Partial final block ranges.
- `K < D`, `K = D`, and `K > D`.
- Several hot coefficients in one ring.
- Empty block ranges and ranges at the last live block.
- Maximum packed position and coefficient boundaries.

The test should reconstruct each global hot field position from the generated
entry and compare it with `j * K + i_j`.

#### Kernel differential tests

For each retained sweep, compare its output with a simple dense or direct
reference. Cover:

- One polynomial and several polynomials.
- Sparse and fully populated one hot chunks.
- Accumulator counts immediately below, at, and above the reduction cap.
- Block counts around every strategy threshold.
- Explicit worker counts that do not depend on the global Rayon pool.
- Every runtime supported ring dimension.

#### Group routing tests

Test direct one hot groups, homogeneous multilinear wrapper groups, and mixed
wrapper groups. Verify output order. Use a test statistic or explicit mock
kernel to prove that delegating backends preserve one group call.

Do not infer route selection from cache mutation. The end state has no one hot
cache mutation to observe.

#### Streaming and lifecycle tests

- Build and release a large NTT slot, request a smaller compatible extent,
  and assert that the released large allocation is not rebuilt.
- Compile and prewarm requirements for a streamed relation and quotient, run
  them, and assert that no complete operation slot remains resident.
- Compare prewarmed and cold execution results for every retained and streamed
  mode.
- Use fewer active challenges than available blocks and prove that no block
  beyond the active range is materialized.
- Run several batched polynomials with an instrumented allocator or materialize
  counter and prove that peak derived storage respects the tile bound.
- Exercise cache reuse, explicit release, concurrent readers, aliased physical
  owners, and post proof cleanup.

#### Boundary and audit tests

- Test the checked `usize` to `u32` offset conversion through a small helper so
  the boundary does not require a `u32::MAX` sized allocation.
- Compare every quotient width and cached or streamed mode around empty, exact,
  partial final, and maximum supported chunk boundaries.
- Audit every changed `unsafe` block locally. Tests supplement but do not
  replace its written safety argument.

#### Required repository checks

Run the cheap gates from `AGENTS.md` first. Then run all three required Clippy
feature graphs and the current CI Nextest command from `.github/workflows/ci.yml`.
Run the Jolt compatibility, profile, documentation, and platform specific jobs
whose workflow path filters include the changed files.

### Performance

The benchmark matrix must separate algorithm cost from end to end schedule
changes.

The kernel matrix covers:

```text
polynomials: 1, 2, 4, 8, 29
ring D:      64, 128, 256
geometry:    K < D, K = D, K > D
blocks:      values below and above each strategy crossover
density:     sparse chunks and all chunks populated
```

The end to end matrix covers the existing singleton, wide batch, direct batch,
and multi group profile modes. Head and merge base runs stay interleaved. Each
reported value uses the median of the same number of runs.

The CPU scheduler chooses the largest useful tile that fits an explicit scratch
budget. A useful estimate is:

\[
M_{tile} \approx
H_t M_{entry}
+ M_{sweep\ scratch}
+ B_t D M_{accum}
+ B_t n_a D M_F.
\]

Here, `H_t` is the number of hot terms and `B_t` is the number of blocks in the
tile. Storage policy uses this memory estimate. Sweep policy uses measured work
costs. Neither decision implies the other.

## Design

### Existing Types That Remain

#### `OneHotPoly`

`OneHotPoly` remains the logical compressed witness. It stores the chunk size,
variable count, construction metadata required by `RootPolyMeta`, and
`Vec<Option<I>>` indices.

The index vector remains a vector. This spec does not replace it with `Arc`
without clone profile evidence.

The block and tensor caches are removed. Tensor projection uses operation local
storage. A future retained projection requires a real production caller and a
separate measured design.

#### `OneHotView`

`OneHotView` remains because the library uses validated source views as the
backend extension boundary. Successful construction validates the source at
dimension `D`. Public read only accessors expose semantic indices and shape.

The view also owns the crate private range materialization method. It does not
store blocks, a strategy, or a cache.

#### `CommitInnerPlan`

`CommitInnerPlan` remains the shared scalar operation plan. Checked helpers may
derive active matrix width and live block count. The implementation must not
copy these values into a new one hot layout type.

#### `FlatBlocks<E>`

`FlatBlocks<E>` becomes the one internal owned container for flat sparse block
storage. It moves out of the one hot module if needed so sparse ring code can
use it without a reverse dependency.

#### `SparseRingBlockEntry`

`SparseRingBlockEntry` becomes the common derived monomial entry. One hot
builders always set `value` to one. One hot arithmetic may ignore the value
field after a debug assertion because only its own validated builder can create
the tile.

`SingleChunkEntry` and `SparseRingBlockEntry` both occupy eight bytes on the
current target layout. Reusing the signed entry does not increase the common
one hot term size.

### Canonical One Hot Mapping

Let `j` be a chunk index and let `i_j` be its nonempty hot index. Define:

\[
x = jK + i_j.
\]

For ring dimension `D` and block width `L`:

\[
\begin{aligned}
u &= \lfloor x / D \rfloor, \\
b &= \lfloor u / L \rfloor, \\
p &= u \bmod L, \\
c &= x \bmod D.
\end{aligned}
\]

The builder emits one entry in block `b` with position `p`, coefficient `c`,
and value one.

Chunk order already sorts entries by global field position. The builder does
not need a sort. Repeated positions are valid when several chunks contribute
to one ring.

### Canonical Commitment Boundary

`RootCommitKernel` exposes one group method:

```rust
fn commit_inner_group(
    &self,
    prepared: &Self::PreparedSetup,
    sources: Vec<S>,
    plan: CommitInnerPlan,
) -> Result<Vec<CommitInnerWitness<F>>, AkitaError>;
```

There is no singleton method. A caller with one source passes one source and
extracts one result.

The CPU one hot implementation obtains its matrix through the existing
prepared setup capability. It validates the sources and plan, selects tiles,
materializes the needed ranges, runs the selected sweep, and returns witnesses
in source order.

The source view owns data and traversal. The CPU implementation owns scheduling
and arithmetic. Protocol code owns neither.

### CPU Sweep Policy

The direct, bucketed, and merge sweeps remain separate only while each has a
measured use.

The direct sweep is suitable for very few blocks. The bucketed sweep pays a
count and scatter cost but streams matrix columns in order across many blocks.
The merge sweep avoids bucket scratch but performs cursor work across blocks
and matrix column chunks.

The policy is private. Protocol APIs cannot request a sweep. Source types do
not encode a sweep.

A private enum is justified only if several strategies remain after the
benchmark matrix. The enum then makes policy tests and tracing exact. It is not
a wrapper around data or another operation.

### Cache and Lifetime Policy

One hot block entries are rebuilt for the range an operation consumes. A
commitment builds one tile. Opening code builds or streams the ranges it needs.
The derived storage is dropped when the operation finishes.

Prepared NTT data follows a different rule because setup transforms are shared
across operations and fold levels. The prepared setup owns them. Retention is
the default. A caller may explicitly wrap its stack with a release policy when
it owns the lifecycle boundary.

These policies are intentionally different. The owner and reuse window decide
retention. The protocol does not clear data merely because it finished one
phase.

### Other PR 375 Work

#### Wide commitment accumulation

`HasCommitAccum` remains the field owned contract for unit scale commitment
streams. It states the accumulator type and the safe addition count. One hot
sweeps use this contract for reduction. General sparse signed kernels may keep
their broader `HasWide` contract when they perform different arithmetic.

The two contracts should not be merged unless their supported values and
reduction limits become identical.

#### Ring relation and quotient streaming

Large relation and quotient work may stream transforms instead of retaining a
matrix sized cache. Cached and streamed execution consume one validated plan.
Validation, role bounds, and chunk sizing have one owner.

That plan also owns the logical matrix range for each quotient width. The
`q32`, `q64`, and `q128` implementations receive validated slices and differ
only where their arithmetic differs. They do not recalculate offsets.

The two execution modes remain separate because one reads prepared transforms
and the other computes transform chunks. They must return the same canonical
rows and witnesses.

#### NTT lifecycle

Prepared owners build exact prefixes lazily. Shared owners retain their slots
by default. Explicit release operates on owner identity, reports failures, and
deduplicates aliases across operation clusters.

The requirements compiler records whether each operation is retained or
streamed. Prewarm consumes that same decision and skips streamed slots. Runtime
must not make a second independent retention choice.

Release removes the released key or otherwise makes an empty key ineligible
for compatible reuse. A later request may reuse a populated compatible slot.
If no populated slot exists, it creates the exact required extent. A smaller
request must never recreate a released larger extent.

Public lifecycle documentation defines who owns preparation and release,
whether clearing may overlap active readers, which state may survive across
proofs, and what cleanup occurs after a proof. It includes a cold, prewarmed,
streamed, released, and reused execution sequence.

The root proof may call the lifecycle hook. The default hook does nothing.

#### Exact prefix folding

Exact prefix tables fold in place. Parallel waves may write only ranges whose
input ranges no longer overlap earlier outputs. Dense differential tests are
the correctness oracle.

#### Copy removal

`RingVec::from_coefficient_rows` consumes fixed coefficient arrays directly.
Fold witness construction uses it to avoid constructing ring objects and then
copying their coefficients into another vector.

This is an ownership improvement, not a new proof representation.

#### Profile workload organization

The profile workload stays split by scenario and policy. Module boundaries may
move whole existing functions. They must not add forwarding functions or copy
scenario logic.

### Deletion Ledger

The baseline design adds no new structs. It removes or internalizes these
concepts:

| Concept | End state |
|---------|-----------|
| `SingleChunkEntry` | Removed. Use `SparseRingBlockEntry` with value one. |
| `MultiChunkEntry` | Removed. Emit one flat entry per hot coefficient. |
| `OneHotEntry` | Removed. Kernels consume one concrete entry type. |
| `OneHotBlocks` | Removed. Use `FlatBlocks<SparseRingBlockEntry>`. |
| `OneHotCommitBlocks` | Removed. Source view materializes a range directly. |
| `OneHotBlockSource` | Removed. Range size expresses full or tiled storage. |
| `OneHotBlockRange` | Removed. `FlatBlocks` owns a requested range. |
| `LazyOneHotBlocks` | Removed. No boxed builder. |
| `OneHotCommitRowsPlan` | Removed. Use `OneHotView` and `CommitInnerPlan`. |
| One hot block cache | Removed. Derived blocks are operation local. |
| One hot tensor cache | Removed. Projection is operation local until reuse is proven. |
| `SparseRingBlocks` | Removed if it remains identical to `FlatBlocks`. |
| Singleton root commit method | Removed. Group commit is canonical. |
| One hot row backend methods | Removed. Source typed kernel is canonical. |
| `CommitmentComputeBackend` | Removed. Source typed kernels plus shared digit rows replace it. |
| Representation commit row plans | Removed. Real input sum types may remain internal. |

### Alternatives Considered

#### Keep the current types and restore a singleton branch

This would recover the benchmark quickly. It would preserve the duplicated
source boundary, storage variants, geometry variants, and cache ownership
problems. It is acceptable only as a short lived diagnostic patch.

#### Add new layout, request, executor, and prepared types

These types would copy contracts already owned by `OneHotView`,
`CommitInnerPlan`, `RootCommitKernel`, and `FlatBlocks`. They would make the
design look organized without reducing the number of concepts.

#### Make `OneHotPoly` a `SparseRingPoly`

One hot chunk indices are used by tensor and opening fast paths. Converting the
logical source into a generic sparse ring polynomial would discard useful
structure or retain two logical copies. The sources remain distinct. Their
derived flat block entry can be shared.

#### Use one generic arithmetic kernel for one hot and sparse signed rings

The sources can share entry storage. Their best accumulators and batch
strategies are not yet the same. A generic kernel would need value branches or
more traits in the hottest loop. Share the kernel only after measurement shows
that the same implementation wins for both.

#### Retain hidden caches

Hidden caches attach resource lifetime and lock failure to a logical witness.
The current production tree does not prepare these one hot caches. Operation
local storage is simpler and gives the scheduler exact control over peak
memory.

#### Expose a public sweep or retention flag

Sweep selection depends on CPU and workload costs. Retention depends on the
owner and reuse window. Neither is a protocol choice. Public flags would make
callers maintain backend policy.

## Documentation

During implementation:

- Update this spec after each accepted slice and keep its status `active`.
- Update `specs/akita-polyops-cutover.md` when the legacy commitment row
  boundary is removed.
- Add a supersession note to the one hot cache ownership section of
  `specs/small-field-prover-opening-optimization.md`.
- Refresh `book/src/how/optimizations.md` with the final stable ownership and
  sweep design before this spec becomes `implemented`.
- Refresh `docs/compute-backends.md` when public plans or backend traits are
  removed.
- Update the PR description and benchmark table after the final route is
  measured.

Before PR 375 merges, check every completed criterion and set the final PR
link. Fold the durable design into the book and archive this spec in the same
PR. Archiving preserves the deletion ledger without making removed public
symbols part of live documentation.

## Execution

### Slice 0: Freeze evidence and add route observability

- Record the current fixed D64 singleton, adaptive singleton, wide batch, and
  multi group benchmark medians.
- Add test or tracing statistics for sweep, tile size, hot terms, and matrix
  passes.
- Keep dense and current one hot outputs as byte exact correctness oracles.

This slice changes no production strategy.

### Slice 1: Close NTT and unsafe contract gaps

- Make released NTT keys ineligible for empty compatible reuse.
- Compile retained or streamed requirements from the runtime execution policy.
- Add regression tests for large to small release and streamed prewarming.
- Document NTT ownership and concurrency at the public API boundary.
- Add complete local safety arguments to changed NEON blocks or centralize the
  repeated unsafe operation.

These changes do not depend on the one hot cutover and fix the two high impact
memory contract mismatches before further benchmark conclusions are drawn.

### Slice 2: Unify the derived entry representation and active ranges

- Move `FlatBlocks<E>` to a neutral internal owner.
- Teach `OneHotView` to materialize a requested block range as
  `FlatBlocks<SparseRingBlockEntry>`.
- Replace the two geometry builders with the canonical coordinate formula.
- Compute active challenge ranges before materialization.
- Port one hot fold, accumulation, and commitment code to one range operation
  over the flat entry.
- Bound batched materialization by the current polynomial and block tile.
- Replace flat block offset casts with checked conversion and add the boundary
  test.
- Delete the geometry trait and enums after differential tests pass.

### Slice 3: Finish the source typed commitment boundary

- Make group commitment the only `RootCommitKernel` method.
- Route both top level commitment paths through it.
- Make delegating backends forward the group method.
- Let the CPU one hot kernel consume `OneHotView` directly.
- Remove one hot row plans, boxed builders, backend methods, and public exports.
- Port dense, sparse ring, root projection, and recursive witness commitment to
  their existing source typed kernel boundary.
- Remove `CommitmentComputeBackend` and the representation commit row plans.

Do not leave forwarding wrappers after the cutover.

### Slice 4: Separate tiling from sweep selection

- Choose tile size from an explicit scratch budget.
- Benchmark direct, bucketed, and merge sweeps over the required matrix.
- Delete dominated sweeps.
- Add one private selector only if several strategies remain.
- Delete test only scheduling drivers and test the production selector and
  driver directly.
- Restore the fixed D64 singleton performance without losing wide batch memory.

### Slice 5: Remove one hot mutable derived state

- Remove block cache preparation, clearing, and lookup.
- Stream opening and decompose ranges through the canonical builder.
- Remove tensor projection cache preparation, clearing, and lookup.
- Keep tensor projection operation local.
- Add clone and concurrent operation tests that prove no shared mutation.

### Slice 6: Consolidate the remaining PR work

- Confirm relation and quotient validation has one owner.
- Make cached and streamed quotient modes consume one chunk and matrix slice
  plan. Keep only width specific arithmetic in their branches.
- Confirm NTT release remains explicit and owner aware.
- Confirm exact prefix parallel ranges cannot overlap their inputs.
- Confirm wide accumulation caps come from the field contract.
- Remove unused public wide ring helpers unless the canonical implementation
  consumes them.
- Confirm the profile module split contains no duplicate scenario logic.

This slice should delete any temporary compatibility code left by earlier PR
history.

### Slice 7: Validate and document

- Run all repository gates and path specific workflows.
- Run the benchmark matrix and enforce the acceptance thresholds.
- Update this spec, the relevant older specs, backend documentation, book
  optimization chapter, and PR description.
- Resolve every SumChecker finding against the disposition table and link the
  validating test, deletion, documentation, or benchmark evidence.
- Mark this spec `implemented` only when the desired end state is present.

## References

- [PR 375](https://github.com/LayerZero-Labs/akita/pull/375)
- [PR 375 benchmark history](https://github.com/LayerZero-Labs/akita/pull/375#issuecomment-5222232182)
- [SumChecker review](https://github.com/LayerZero-Labs/akita/pull/375#issuecomment-5227378132)
- [`specs/akita-polyops-cutover.md`](akita-polyops-cutover.md)
- [`specs/small-field-prover-opening-optimization.md`](small-field-prover-opening-optimization.md)
- [`docs/compute-backends.md`](../docs/compute-backends.md)
- [`docs/documentation.md`](../docs/documentation.md)
- [`book/src/how/optimizations.md`](../book/src/how/optimizations.md)
- PR head at spec creation: `165ad16321cabb917166e716f2ce1e03e323a586`
- PR head at review reconciliation: `82b129adcfde2ddad0febce0a90a618687f1b6df`
- Merge base at spec creation: `b5cb55f5c9f91af6e032621464d37880f1c5784f`
