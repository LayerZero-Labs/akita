# Spec: Typed Fold-Schedule Topology and Planner Cutover

| Field         | Value                                       |
|---------------|---------------------------------------------|
| Author(s)     | Quang Dao                                   |
| Created       | 2026-07-21                                  |
| Status        | active                                      |
| PR            | [#317](https://github.com/LayerZero-Labs/akita/pull/317) |
| Supersedes    |                                             |
| Superseded-by |                                             |
| Book-chapter  | book/src/how/configuration.md               |

## Summary

Akita's proof object already distinguishes the root fold, recursive folds, and
terminal fold, but the planner and generated catalogs encode every fold in one
homogeneous array. Root-only behavior is then recovered from `level == 0`,
terminal behavior from array position, setup-offload transitions from redundant
mode flags, and tensor support from a per-level callback. The representation is
therefore less precise than the protocol and makes invalid combinations, such
as a tensor recursive fold or arbitrary multi-group recursive fold,
representable.

This cutover gives schedules the same typed topology as proofs:

```text
root -> recursive_folds[] -> terminal
```

The root type exclusively owns root source structure and arbitrary
precommitted groups. Within it, only the final group may select a tensor
challenge; every standalone precommitted group is flat by protocol definition.
Recursive folds always use flat challenges and may consume at most one incoming
setup-prefix commitment. The terminal type owns the final committed fold and
cleartext witness handoff and cannot carry root-only or recursive-setup
features.

The cutover also replaces protocol-facing `A`, `B`, and `D` matrix names with
descriptive names. Mathematical notation remains in docstrings:

| Descriptive name | Notation | Function |
|------------------|----------|----------|
| `InnerCommitMatrix` | `A` | Commits decomposed source blocks `s` to inner commitments `t`. |
| `OuterCommitMatrix` | `B` | Commits decomposed `t` values to recursive commitments `u`. |
| `OpenCommitMatrix` | `D` | Commits decomposed partial evaluations `e` to opening commitments `v`. |

Every matrix plan owns its ring dimension. There is no schedule-global
`ring_d` in the final representation. The generated schema records every
independent planner decision, including role-specific ring dimensions, exact
final-root-group tensor factorization, decomposition digit widths, block
geometry, per-fold witness partitioning, setup-prefix inputs, and balanced
outer/open matrix slicing. Commitment compression is added only with its exact
protocol types. Derived widths, digit depths, collision bounds, witness lengths,
and protocol byte limits remain validated calculations rather than competing
sources of truth. Non-binding planner byte estimates live outside the schedule.

This work starts from `origin/main` commit
`e131faf48938b975ca63b12b59ac6d86894048e0` (PR #312). It does not wait for or
assume any open PR. Akita has no backward-compatibility requirement, so the
implementation is a direct cutover with no legacy schedule adapter.

## Intent

### Goal

Replace the flattened schedule and overloaded `LevelParams` representation
with role-specific generated and runtime types that make the legal protocol
topology explicit, make tensor challenges structurally final-root-group-only,
expose all matrix-role ring dimensions, and provide stable extension points for
mixed rings, distributed witness partitions, setup offloading, slicing, and
commitment compression.

### Invariants

#### Topology

- Every valid schedule contains exactly one root fold and exactly one terminal
  fold, with zero or more recursive folds between them.
- Root, recursive, and terminal roles are represented by different Rust types.
  No role is inferred from an integer level or array position.
- The runtime schedule and generated schedule have the same topology as
  `AkitaBatchedProof { root, recursive_folds, terminal }`.
- The outgoing binding of a fold is derived from typed adjacency. An edge to a
  recursive fold uses the outer commitment binding; the edge to the terminal
  fold uses the terminal inner-state binding. No stored binding enum may
  disagree with the topology.
- The terminal fold cannot consume or produce a setup prefix, cannot contain
  arbitrary precommitted groups, and cannot use a multi-chunk witness layout.

#### Tensor challenges

- Tensor challenges are supported only by `RootFinalGroupParams` and
  `GeneratedRootFinalGroup`.
- Standalone precommitted root groups are flat by protocol definition. Their
  generated/runtime types contain no challenge-family or tensor-factor field.
- Recursive and terminal parameter types do not contain a challenge-shape
  field. Their fold challenge is flat by protocol definition.
- A generated final-root-group entry stores the exact selected `fold_low_len`.
  A value such as `Tensor { fold_low_len: 2 }` is never used merely as an
  enablement marker.
- Root groups still receive independent, group-index-domain-separated
  challenge draws. The final group uses its selected flat/tensor shape; every
  precommitted group uses the canonical flat draw for its own live geometry.
- A setup prefix consumed by a recursive fold always uses flat challenge
  geometry. Tensor metadata cannot be serialized into a setup-prefix slot.
- The planner config surface exposes a final-root-group challenge-family policy,
  not a callback accepting an arbitrary fold level or group index.

#### Group ownership

- Arbitrary multi-group batching exists only at the root. The root contains one
  final group and zero or more standalone precommitted groups.
- A recursive fold contains exactly one ordinary witness group and zero or one
  incoming setup-prefix group.
- An incoming setup prefix has its own block geometry and its own inner and
  outer commitment matrices. It shares the consuming fold's open commitment
  matrix.
- The presence of `recursive_folds[i].incoming_setup_prefix` is the canonical
  statement that the predecessor offloaded its setup contribution. There is no
  separately stored `SetupContributionMode` that can disagree with adjacency.
- A direct fold may consume an incoming prefix and then stop offloading. It
  simply has a successor with no incoming prefix.
- `SetupPrefixSlotId` remains the canonical runtime identity of a committed
  setup-prefix artifact.

#### Matrix roles and mixed rings

- Protocol-facing code uses `InnerCommitMatrix`, `OuterCommitMatrix`, and
  `OpenCommitMatrix`. The letters A/B/D appear only as notation in docstrings,
  formulas, and paper-facing explanations.
- Every matrix owns its ring dimension. No separate `ring_dimension` field may
  disagree with matrix keys or a parallel role-dimension carrier.
- For each committed group `g`, dimensions satisfy
  `d_open | d_outer[g] | d_inner[g]`.
- The setup generator ring dimension is divisible by every matrix ring
  dimension used by the schedule.
- `d_inner` is supported by the sparse fold-challenge sampler. Smaller ring
  dimensions may be selected independently for outer and open matrices.
- Multi-group root groups may select different inner and outer ring dimensions.
  The fold-shared open matrix uses one `d_open` compatible with every group.
- Matrix storage, verifier work, proof bytes, and SIS rank are calculated from
  each matrix's actual ring dimension rather than a catalog-global dimension.

#### Decomposition and security

- Inner, outer, and opening decomposition digit widths are independent planner
  choices. In particular, the inner source decomposition is not constrained by
  a range check unless a concrete protocol relation requires it. These outer
  and opening choices exist only at non-terminal folds that actually construct
  the corresponding decompositions and commitment relations.
- Generated entries store selected digit widths, ring dimensions, and balanced
  slice counts. They do not author digit depths, input widths, output ranks,
  coefficient bounds, slice boundaries, or SIS table buckets.
- Expansion derives digit depths, input widths, certified coefficient bounds,
  balanced slice boundaries, and the minimum SIS-secure output rank for every
  matrix or slice. Deliberate rank overprovisioning is not a planner variable.
- `AjtaiKeyParams::try_new` or its renamed canonical successor audits every
  expanded `(role, ring dimension, output rank, input width, coefficient
  bound)` tuple.
- A recursive folded response uses the protocol's digit-certification relation
  and its tight difference interval. Its bound retains the clean
  digit-boundary snap required by that relation.
- A terminal folded response has no clean digit-boundary snap. Planning uses
  the exact unsnapped honest-response quantile to choose an inner-matrix SIS bucket and
  minimum output rank. Schedule validation derives the largest A-role
  collision bucket supported by the expanded inner matrix. Verification checks
  the actual decoded response infinity norm against that matrix-derived
  capacity.
- Frozen standalone commitments bind their exact security descriptor. On
  replay, the descriptor is rederived and equality-checked; it is not accepted
  as an unaudited override.
- There is one canonical calculation for each matrix input width, output rank,
  collision bound, witness width, setup prefix length, and planner byte score.

#### Generated catalogs

- A generated entry contains all independent choices required to reproduce the
  effective schedule without rerunning an optimizer. It also emits exact live
  geometry as an auditable checksum; replay rederives it from the statement or
  predecessor and requires equality.
- A catalog identity contains search/security policy identity, not row-local
  decisions. Exact final-root-group tensor factorization and per-level
  partitioning live in the entry.
- Table expansion and dynamic planning produce descriptor-identical runtime
  schedules for the same lookup key and policy.
- Generated lookup order and key digests include the complete root statement:
  final group plus ordered standalone precommitted commitment descriptors.
- Generated catalogs with different source families, final-root-group challenge
  families, chunk policies, setup-offload policies, matrix dimension domains,
  slicing capability, or SIS table digests cannot alias.

#### Transcript, serialization, and safety

- The instance descriptor binds topology tags, ordered groups, exact final-root
  challenge shape, the flat precommitted-group invariant, all matrix dimensions
  and ranks, decomposition digit widths, block geometry, witness partitions,
  setup-prefix identities, balanced slicing plans, witness lengths, and terminal
  response shape.
- Serialization uses explicit root, recursive, and terminal sections. It does
  not serialize a homogeneous fold list and infer roles during decoding.
- Malformed counts, dimensions, slice counts, prefix identities, or derived
  arithmetic overflow return `AkitaError` or `SerializationError`.
  Verifier-reachable code does not panic or allocate from unchecked
  schedule-controlled dimensions.
- Schedule and proof descriptor changes intentionally define a new protocol
  epoch. Old generated rows, setup artifacts, proofs, and descriptors are not
  accepted through compatibility shims.

### Non-Goals

- Choosing new production parameters in the topology cutover itself. The first
  regeneration preserves current-main planner choices except for the explicit
  terminal direct-response correction and the removal of tensor-shaped
  standalone precommitments.
- Changing planner search behavior, objective weighting, tie-breaking outcomes,
  or emitted schedule numbers merely because the new representation can express
  more choices. Cuts 2, 3, and 5 are behavior-preserving cutovers. Any generated
  value change must be required by Cut 1's protocol correction, the typed
  root-only tensor invariant, or another individually identified correctness
  requirement, and must appear in the Cut 0 parity report.
- Treating open PR behavior as landed. Later commits on this branch may add
  features only after their implementation and canonical formulas are present.
- Reimplementing the SIS estimator or maintaining a second security model in
  the emitter.
- Preserving source compatibility for `GeneratedFold`, `LevelParams.a_key`,
  `LevelParams.b_key`, `LevelParams.d_key`, `CommitmentRingDims`, or the
  per-level fold-shape callback.
- Supporting tensor challenges for precommitted-root, recursive, or terminal
  groups, now or later.
- Adding arbitrary precommitted groups to recursive folds.
- Encoding commitment-compression placeholders before its exact protocol types,
  cost model, and validation rules land.

## Current Main Baseline

### Generated representation

Current main defines one homogeneous step and an optional metadata wrapper:

```rust
pub struct GeneratedFoldStep {
    pub ring_d: u32,
    pub log_basis_inner: u32,
    pub log_basis_outer: u32,
    pub log_basis_open: u32,
    pub position_index_bits: u32,
    pub block_index_bits: u32,
    pub num_live_blocks: u32,
    pub n_a: u32,
    pub n_b: u32,
    pub n_d: u32,
}

pub struct GeneratedFoldStepWithSetupMetadata {
    pub fold: GeneratedFoldStep,
    pub setup_prefix_group: Option<GeneratedSetupPrefixGroup>,
    pub setup_contribution_mode: SetupContributionMode,
}

pub enum GeneratedFold {
    Fold(GeneratedFoldStep),
    FoldWithSetupMetadata(GeneratedFoldStepWithSetupMetadata),
}

pub struct GeneratedScheduleTableEntry {
    pub final_group: PolynomialGroupLayout,
    pub precommitteds: &'static [PrecommittedGroupParams],
    pub folds: &'static [GeneratedFold],
}
```

The catalog identity stores one `ring_dimension`, an allowed
`ring_dimensions` slice, a global `ChunkedWitnessCfg`, and a
`root_fold_shape`. Expansion still requires `GeneratedFoldStep.ring_d` to equal
the policy's global ring dimension. Root and terminal roles are inferred from
the fold index. The setup mode is redundantly stored on the producer while the
setup-prefix group is stored on the consumer.

### Runtime representation

Current `LevelParams` combines all of the following:

- one legacy `ring_dimension`;
- three `log_basis` digit-width fields;
- `a_key`, `b_key`, and `d_key`;
- block geometry;
- sparse challenge configuration and flat/tensor shape;
- derived digit depths and folded-response caches;
- root one-hot metadata;
- multi-chunk witness metadata;
- arbitrary root precommitted groups;
- an incoming setup-prefix slot;
- a second `CommitmentRingDims` dimension carrier; and
- an outgoing `SetupContributionMode`.

`Schedule` stores `Vec<FoldStep>` plus a separate cleartext
`TerminalWitnessPlan`. `get_execution_schedule` infers root, terminal,
penultimate binding, and successor behavior by index.

Current `FoldStep.level_bytes`, `TerminalWitnessPlan.terminal_bytes`, and
`Schedule.total_bytes` are planner scores used by suffix DP, table selection,
tests, and profile reporting. They are also included in the current instance
descriptor even though they are not protocol parameters. The aggregate is a
header-stripped direct-mode estimate: terminal Golomb-Rice bytes are
conservative, while recursive setup-product payloads and outer serialization
framing are excluded. It is neither an exact serialized size nor a total proof
upper bound.

### Exact migration fixture

The current generated `fp128_d64_onehot_recursive` row for a 32-variable,
two-polynomial final group and two 16-variable standalone precommitted groups is
the primary topology migration fixture. It contains nine folds:

| Protocol level | Current variant | `log2` digit widths inner/outer/open | Position bits | Block bits | Live blocks | `d_inner/d_outer/d_open` | Inner/outer/open output ranks | Incoming prefix natural length | Outgoing setup mode |
|----------------|-----------------|--------------------------------|---------------|------------|-------------|--------------------------|-----------------------|--------------------------------|---------------------|
| L0 root | metadata | 3/3/3 | 15 | 11 | 2048 | 64/64/64 | 5/2/1 | none | recursive |
| L1 recursive | metadata | 3/3/3 | 13 | 8 | 148 | 64/64/64 | 5/1/1 | 112,721,920 | recursive |
| L2 recursive | metadata | 5/5/5 | 13 | 7 | 106 | 64/64/64 | 6/1/1 | 39,452,672 | direct |
| L3 recursive | plain | 5/5/5 | 12 | 7 | 76 | 64/64/64 | 6/1/1 | none | direct |
| L4 recursive | plain | 5/5/5 | 10 | 5 | 26 | 64/64/64 | 6/1/1 | none | direct |
| L5 recursive | plain | 6/6/6 | 10 | 3 | 8 | 64/64/64 | 5/1/1 | none | direct |
| L6 recursive | plain | 6/6/6 | 9 | 3 | 7 | 64/64/64 | 5/1/1 | none | direct |
| L7 recursive | plain | 6/6/6 | 8 | 4 | 9 | 64/64/64 | 4/1/1 | none | direct |
| L8 terminal | plain | 6/6/6 | 8 | 3 | 7 | 64/64/64 | 4/1/1 | none | direct |

The L8 outer/open digit widths, dimensions, and output ranks in this table
document current-main's homogeneous representation; they are not target
protocol parameters. The cutover retains only L8's inner digit width, inner
ring dimension, inner output rank, geometry, and derived direct-response
bound/shape.

The root final group has exact source geometry:

```text
layout                         = (num_vars=32, num_polynomials=2)
source                         = one-hot, chunk_size=256
d_inner                        = 64
live ring elements per claim   = 67,108,864
positions per block            = 32,768
live blocks                    = 2,048
root challenge                 = Flat
inner digit width / output rank      = 3 / 5
outer digit width / output rank      = 3 / 2
shared open digit width / output rank = 3 / 1
```

Each of the two standalone root groups has:

```text
layout                         = (num_vars=16, num_polynomials=1)
d_inner / d_outer              = 64 / 64
live ring elements per claim   = 1,024
positions per block            = 32
live blocks                    = 32
root challenge                 = Flat
inner digit width / output rank      = 2 / 4
outer digit width / output rank      = 2 / 2
```

The L1 incoming setup prefix has `N=2,097,152`, `M=2,048`, `B=1,024`,
inner/outer/open digit widths `3/3/3`, inner/outer output ranks `7/1`, and uniform
dimension 64. The L2 incoming setup prefix has `N=1,048,576`, `M=2,048`,
`B=512`, digit widths `5/5/5`, output ranks `7/2`, and uniform dimension 64.

The first cutover regeneration must reproduce all non-terminal effective
parameters and costs. Terminal proof bytes and security inputs intentionally
change to the direct-response model and must have an explicit old/new fixture.

## Terminology and Naming Cutover

### Canonical names

The following names are canonical in Rust APIs and prose owned by the codebase:

```rust
pub enum CommitMatrixRole {
    /// Inner commitment matrix, denoted A in the protocol.
    Inner,
    /// Outer commitment matrix, denoted B in the protocol.
    Outer,
    /// Opening commitment matrix, denoted D in the protocol.
    Open,
}

pub struct InnerCommitMatrixParams { /* ... */ }
pub struct OuterCommitMatrixParams { /* ... */ }
pub struct OpenCommitMatrixParams { /* ... */ }
```

Field and method names use the same vocabulary:

```text
a_key                    -> inner_commit_matrix
b_key                    -> outer_commit_matrix
d_key                    -> open_commit_matrix
n_a                      -> inner_commit_output_rank
n_b                      -> outer_commit_output_rank
n_d                      -> open_commit_output_rank
d_a                      -> inner_commit_ring_dimension
d_b                      -> outer_commit_ring_dimension
d_d                      -> open_commit_ring_dimension
log_basis_inner          -> inner_digit_bits
log_basis_outer          -> outer_digit_bits (non-terminal committed groups)
log_basis_open           -> open_digit_bits (folds with an opening matrix)
log_commit_bound         -> commit_bound_bits
log_open_bound           -> open_bound_bits
inner_width              -> inner_commit_input_width
outer_width              -> outer_commit_input_width
d_matrix_width           -> open_commit_input_width
```

The emitter uses descriptive `const` constructors to keep generated source
compact without abbreviating public field or parameter names. It does not
introduce aliases that keep both vocabularies alive. Mathematical helpers use
local variables `a`, `b`, or `d` only when directly transcribing a formula.

The final decomposition vocabulary is equally explicit:

- `digit_bits = k` means radix `b = 2^k` and the balanced digit alphabet
  `[-2^(k-1), 2^(k-1) - 1]`;
- `num_digits` is the decomposition depth, never the digit width;
- `commit_bound_bits` and `open_bound_bits` are coefficient-range bit widths;
  and
- mathematical prose uses “decomposition radix” or `b`, not “basis,” unless it
  genuinely refers to a vector or polynomial basis.

The API does not use `num_digit_bits`, which is confusable with `num_digits`,
or `bits_per_digit`, which can be mistaken for the physical `i8`/`i16` storage
width. `digit_bits` states the power-of-two balanced alphabet while remaining
compact in role-specific matrix types.

The three matrix dimensions have deliberately different nouns:

- `ring_dimension` is the number of base-field coefficients in one ring
  element;
- `input_width` is the number of ring elements consumed by the matrix; and
- `output_rank` is the number of ring elements produced, equivalently the SIS
  output-module rank.

Thus a commitment matrix is a map
`R^input_width -> R^output_rank`. The API does not use bare `rows`, `columns`,
`height`, or `width`, because those names do not say which module or scalar
domain is being counted. Slice ranges use `input_start` and `input_width`.

### Matrix ownership

Each role-specific expanded matrix parameter object owns the full auditable SIS
tuple. The types are separate because their message geometry and protocol use
are different:

```rust
pub struct InnerCommitMatrixParams {
    pub ring_dimension: usize,
    /// Number of output ring elements; the SIS module rank (n_A).
    pub output_rank: usize,
    /// Number of input ring elements accepted by the matrix.
    pub input_width: usize,
    pub coeff_linf_bound: u128,
    pub sis_table_key: SisTableKey,
}

pub struct OuterCommitMatrixParams {
    pub ring_dimension: usize,
    /// Number of input ring elements accepted by the matrix.
    pub input_width: usize,
    pub coeff_linf_bound: u128,
    pub sis_table_key: SisTableKey,
    pub slices: Vec<CommitMatrixSliceParams>,
}

pub struct OpenCommitMatrixParams {
    pub ring_dimension: usize,
    /// Number of input ring elements accepted by the matrix.
    pub input_width: usize,
    pub coeff_linf_bound: u128,
    pub sis_table_key: SisTableKey,
    pub slices: Vec<CommitMatrixSliceParams>,
}

pub struct CommitMatrixSliceParams {
    pub input_start: usize,
    pub input_width: usize,
    pub output_rank: usize,
}
```

These are not thin wrappers around a public generic matrix object. They own
role-specific validation and behavior. Outer/open matrix params own the common
role, ring dimension, coefficient bound, and logical input width; `slices` owns
one derived output rank per physical matrix object. A one-element vector is the
monolithic case. One private SIS
audit function accepts `CommitMatrixRole` plus a slice's complete tuple and
returns its minimum secure output rank; the three constructors call it rather
than duplicating security logic.

## Desired Generated Representation

Generated types record optimizer decisions. They intentionally omit values
that are uniquely derived and security-checked during expansion.

### Common geometry

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedBlockGeometry {
    /// Exact number of source ring elements per claim.
    pub live_ring_elements_per_claim: u64,
    /// Exact power-of-two block width in source ring elements.
    pub positions_per_block: u32,
    /// Exact live block count, including a possibly partial final block.
    pub live_blocks: u32,
}
```

Validation derives and checks:

```text
positions_per_block = 2^position_index_bits
live_blocks         = ceil(live_ring_elements_per_claim / positions_per_block)
block_index_bits    = ceil_log2(live_blocks)
```

Generated source does not treat all three values as independent choices.
`positions_per_block` is selected; the exact live counts are emitted for
auditability and checked against the root statement or predecessor witness.
`position_index_bits()` and `block_index_bits()` are accessors. Exact live
counts remain explicit so partial final blocks are not lost.

### Matrix choices

```rust
pub const MAX_COMMIT_MATRIX_SLICES: u32 = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedInnerCommitMatrix {
    pub ring_dimension: u32,
    /// k for radix 2^k and balanced digits [-2^(k-1), 2^(k-1) - 1].
    pub digit_bits: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedOuterCommitMatrix {
    pub ring_dimension: u32,
    /// k for radix 2^k and balanced digits [-2^(k-1), 2^(k-1) - 1].
    pub digit_bits: u32,
    /// One is monolithic; valid values are at most MAX_COMMIT_MATRIX_SLICES.
    pub slice_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedOpenCommitMatrix {
    pub ring_dimension: u32,
    /// k for radix 2^k and balanced digits [-2^(k-1), 2^(k-1) - 1].
    pub digit_bits: u32,
    /// One is monolithic; valid values are at most MAX_COMMIT_MATRIX_SLICES.
    pub slice_count: u32,
}
```

For a positive derived logical input width `W`, `slice_count = S` requires
`1 <= S <= min(W, MAX_COMMIT_MATRIX_SLICES)`. A zero logical width means that
the matrix role is absent; it does not produce a zero-width matrix. `S = 1` is
the default and is the monolithic case. There is no monolithic/sliced enum and
no second Boolean or layout tag. Expansion sets

```text
q = W / S
r = W mod S
slice[i].input_start = i * q + min(i, r)
slice[i].input_width = q + (i < r ? 1 : 0)
```

for `0 <= i < S`. Thus slices are contiguous, cover the input exactly once,
and differ in width by at most one; earlier slices receive the remainder. Each
slice's output rank is the minimum rank returned by the canonical SIS table for
that slice's derived width, common coefficient bound, and matrix ring
dimension. The generated schedule stores no slice boundary and no slice rank.
The expanded runtime matrix stores only its derived `slices` vector, so
`slices.len()` is the sole runtime slice count; it does not redundantly retain
the generated `slice_count`.

Commitment compression is not represented by a placeholder in this cutover.
The compression implementation must add its exact generated and runtime types,
descriptor fields, witness cost, and validation rules when it lands. Until
then all commitment outputs are raw.

An inner commitment matrix is always monolithic. Slicing exists only for outer
and open matrices and trades their setup envelope against the public slice
commitments retained in the next witness.

For a monolithic matrix, exact flat setup storage is:

```text
matrix_field_elements = output_rank * input_width * ring_dimension
```

For a sliced matrix, total storage is the sum of this quantity over its derived
slices. Each slice is a separate stored matrix object, so the setup envelope
uses the largest individual slice rather than their sum. Total setup storage
reports the sum of all physical matrix objects.

### Group plans

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedCommittedGroup {
    pub geometry: GeneratedBlockGeometry,
    pub inner_commit_matrix: GeneratedInnerCommitMatrix,
    pub outer_commit_matrix: GeneratedOuterCommitMatrix,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratedRootSource {
    Dense {
        coefficient_bits: u32,
    },
    OneHot {
        chunk_size: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratedRootFinalChallenge {
    Flat,
    Tensor {
        /// Exact optimizer-selected power-of-two low factor.
        fold_low_len: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedRootFinalGroup {
    pub layout: PolynomialGroupLayout,
    pub source: GeneratedRootSource,
    pub challenge: GeneratedRootFinalChallenge,
    pub commitment: GeneratedCommittedGroup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedRootPrecommittedGroup {
    /// Frozen standalone commitment identity and certified bounds.
    pub descriptor: PrecommittedGroupDescriptor,
    pub commitment: GeneratedCommittedGroup,
}
```

The precommitted descriptor is part of the schedule lookup key because the
commitment already exists. Expansion rederives its matrix dimensions, input
widths, bounds, and flat root-opening geometry and requires descriptor
equality. The new-epoch descriptor binds the standalone-precommitted role, for
which flat challenge security is invariant; it does not expose a selectable
challenge family. A historical descriptor authorizing tensor use as a
precommitted group is rejected rather than silently reinterpreted.

### Witness partitioning

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratedWitnessPartition {
    Single,
    Distributed {
        num_chunks: u32,
    },
}
```

Partitioning is stored on each eligible root or recursive fold. The generated
row does not store a global `(num_chunks, num_activated_levels)` and infer which
levels are distributed. The terminal type has no partition field and is always
single-chunk.

### Root fold

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedRootFold {
    pub final_group: GeneratedRootFinalGroup,
    pub precommitted_groups: &'static [GeneratedRootPrecommittedGroup],
    /// One fold-shared opening matrix over all ordered group e-hat segments.
    pub open_commit_matrix: GeneratedOpenCommitMatrix,
    pub witness_partition: GeneratedWitnessPartition,
}
```

`GeneratedRootFinalGroup` is the only generated group type that mentions
`GeneratedRootSource` or `GeneratedRootFinalChallenge`. `GeneratedRootFold` is
the only generated fold type that can contain arbitrary precommitted groups.

### Recursive fold and setup-prefix input

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedSetupPrefixInput {
    pub natural_len: u64,
    pub d_setup: u32,
    pub commitment: GeneratedCommittedGroup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedRecursiveFold {
    /// The balanced recursive witness entering this fold.
    pub witness: GeneratedCommittedGroup,
    /// One fold-shared opening matrix for witness and optional prefix groups.
    pub open_commit_matrix: GeneratedOpenCommitMatrix,
    pub incoming_setup_prefix: Option<GeneratedSetupPrefixInput>,
    pub witness_partition: GeneratedWitnessPartition,
}
```

There is deliberately no challenge field: recursive folding is flat. There is
also no outgoing setup mode. If this fold's successor has an incoming setup
prefix, this fold offloads; otherwise its setup contribution is direct.

The setup-prefix input does not repeat an opening digit width or open matrix. It
uses the consuming fold's shared `open_commit_matrix`. Its inner and outer
matrices may use different ring dimensions from the ordinary witness group,
subject to the nesting and setup-generator constraints.

### Terminal fold

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedTerminalFold {
    pub geometry: GeneratedBlockGeometry,
    pub inner_commit_matrix: GeneratedInnerCommitMatrix,
}
```

The terminal relation has no outer or open commitment matrix and no terminal
outer/open decomposition digit width. Its response contains raw field-valued
`t` and `e`, plus centered folded-response coefficients `z` encoded losslessly. The
verifier decodes `z`, computes its actual infinity norm, applies the complete
A-role weak-binding formula to the corresponding response-difference interval,
checks the resulting collision bound against the inner matrix's derived SIS
capacity, and then checks the opening and inner-commitment relations.
It neither receives nor verifies a digit decomposition of terminal `t`, `e`,
or `z`.

The planner computes an exact, unsnapped honest-response quantile from the
configured failure probability, terminal witness structure, flat challenge,
and trace-subfield factors. It applies the same complete A-role formula to that
quantile to choose the inner matrix's SIS bucket and minimum output rank. This
quantile is a completeness and cost-model input, not the verifier's acceptance
bound.

During expansion and schedule validation, the canonical SIS-table helper finds
the largest A-role collision bucket supported by the inner matrix's fixed ring
dimension, input width, and output rank. The maximum accepted terminal response
norm is the largest centered norm whose complete A-role collision price fits
that bucket. The verifier therefore checks the actual decoded response norm
against capacity already certified by the matrix; there is no terminal
clean digit-boundary snap and no online lattice-estimator call.

The predecessor terminal-binding path computes the inner commitment and binds
raw `t`. It must not decompose `t` merely to satisfy a shared non-terminal code
path. Likewise, raw terminal `e` is checked by its trace and consistency
relations and does not acquire an opening digit width for accounting purposes.

### Complete table entry

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedFoldScheduleEntry {
    pub root: GeneratedRootFold,
    pub recursive_folds: &'static [GeneratedRecursiveFold],
    pub terminal: GeneratedTerminalFold,
}
```

The lookup key is derived from `root.final_group.layout` and the ordered
precommitted descriptors. It is not stored a second time on the entry.

## Desired Runtime Representation

Generated types are planner-owned compact choices. Runtime types are
fully-expanded, verifier-audited protocol parameters.

### Expanded group parameters

```rust
pub struct CommittedGroupParams {
    pub geometry: BlockGeometry,
    pub inner_commit_matrix: InnerCommitMatrixParams,
    pub outer_commit_matrix: OuterCommitMatrixParams,
    pub inner_digits: usize,
    pub outer_digits: usize,
    pub open_digits: usize,
    pub folded_response_digits: FoldedResponseDigitPlan,
}
```

`BlockGeometry` contains exact `N`, `M`, live `B`, and checked index-domain
sizes. Matrix params contain exact input width, ring dimension, collision bound,
and SIS identity; each physical matrix object has its derived output rank.
Digit depths are expanded results, never independent generated inputs.

### Typed fold params

```rust
pub struct RootFoldParams {
    pub final_group: RootFinalGroupParams,
    pub precommitted_groups: Vec<RootPrecommittedGroupParams>,
    pub open_commit_matrix: OpenCommitMatrixParams,
    pub sparse_challenge_config: SparseChallengeConfig,
    pub witness_partition: WitnessPartition,
}

pub struct RecursiveFoldParams {
    pub witness: CommittedGroupParams,
    pub open_commit_matrix: OpenCommitMatrixParams,
    pub sparse_challenge_config: SparseChallengeConfig,
    pub incoming_setup_prefix: Option<SetupPrefixSlotId>,
    pub witness_partition: WitnessPartition,
}

pub struct TerminalFoldParams {
    pub witness: TerminalCommittedGroupParams,
    pub sparse_challenge_config: SparseChallengeConfig,
    pub response_shape: TerminalResponseShape,
}
```

Only `RootFinalGroupParams` contains a `RootFinalChallenge` field.
`RootPrecommittedGroupParams`, `RecursiveFoldParams`, and `TerminalFoldParams`
are flat by type. Sparse sampler configuration remains explicit because it
determines challenge distribution and certified norms even for a flat
challenge.

`TerminalCommittedGroupParams` contains the geometry, inner matrix, and inner
source-decomposition depth required by the terminal relation. It contains no
outer/open matrix, no terminal outer/open digit width, and no fabricated digit depth
for raw response values.

`TerminalResponseShape` counts encoded `z` coefficients, raw `e` field
elements, raw `t` field elements, the codec parameter, and the checked
`z`-payload byte limit. The accepted `z` norm is derived from the inner
matrix's certified SIS capacity and is not stored as a second schedule field.
No shape component expresses raw response values as digit-plane equivalents.

### Typed schedule steps

```rust
pub struct RootFoldStep {
    pub params: RootFoldParams,
    pub input_witness_len: usize,
    pub output_witness_len: usize,
}

pub struct RecursiveFoldStep {
    pub params: RecursiveFoldParams,
    pub input_witness_len: usize,
    pub output_witness_len: usize,
}

pub struct TerminalFoldStep {
    pub params: TerminalFoldParams,
    pub input_witness_len: usize,
}

pub struct FoldSchedule {
    pub root: RootFoldStep,
    pub recursive_folds: Vec<RecursiveFoldStep>,
    pub terminal: TerminalFoldStep,
}
```

The terminal cleartext response is part of `TerminalFoldStep`; it is not a
second object adjacent to a homogeneous fold vector. A terminal step has no
`output_witness_len`: recursion ends there, and its raw response shape is not a
fictional decomposed recursive witness.

Proof-size estimates are planner output, not protocol schedule fields. The
planner returns a `PlannedFoldSchedule` containing the `FoldSchedule` and a
separate estimate:

```rust
pub struct PlannedFoldSchedule {
    pub schedule: FoldSchedule,
    pub estimate: FoldScheduleEstimate,
}

pub struct FoldScheduleEstimate {
    pub estimated_root_direct_payload_bytes: usize,
    pub estimated_recursive_direct_payload_bytes: Vec<usize>,
    pub estimated_terminal_direct_payload_bytes: usize,
    pub estimated_terminal_response_payload_bytes: usize,
}
```

`FoldScheduleEstimate::estimated_direct_proof_payload_bytes()` is a checked
derived accessor, not a stored aggregate. It returns the header-stripped
direct-mode payload estimate for the materialized schedule. It includes the
modeled terminal response payload and excludes recursive setup-product payloads
and outer serialization framing. It is the checked sum of the three per-step
fields. `estimated_terminal_response_payload_bytes` is the terminal component
reported separately, not an additional summand. With the planner recursion cap
of 12, this accessor performs at most 14 checked additions and is used only for
retained/final schedule reporting and consistency checks, not inside candidate
search. None of these estimates is transcript-bound or serialized in the
instance descriptor. Profiles report measured serialized proof bytes
separately.

The hot planner path keeps a separate planner-private scalar score:

```rust
struct CandidateCost {
    estimated_direct_proof_payload_bytes: usize,
    // The other active Pareto coordinates.
}
```

Suffix DP and Pareto search update this scalar incrementally with checked
addition when extending a candidate. Candidate comparison is therefore O(1)
and does not walk a partially built schedule. `CandidateCost` is search state,
not a public estimate or protocol parameter. When a retained or winning
candidate is materialized, `FoldScheduleEstimate::derive` constructs the full
component breakdown and requires its recomputed aggregate to equal the cached
candidate score. This equality check makes the cache auditable without storing
duplicate mutable state in `FoldScheduleEstimate`.

The proof wire uses an equally direct type:

```rust
pub struct TerminalResponse<F: LatticeField> {
    /// Lossless centered encoding; every decoded coefficient is range-checked.
    pub folded_response_payload: Vec<u8>,
    /// Raw full-field partial evaluations.
    pub partial_evaluations: RingVec<F>,
    /// Raw full-field inner commitments bound by the predecessor transcript.
    pub inner_commitments: RingVec<F>,
}
```

This replaces the misleading `SegmentTypedWitness` terminology. The transcript
continues to bind terminal `t` at its predecessor, and binds terminal `e` and
`z` in their existing challenge order. Renaming the container does not change
that order.

Execution APIs consume a typed reference:

```rust
pub enum FoldExecution<'a> {
    Root(&'a RootFoldStep),
    Recursive(&'a RecursiveFoldStep),
    Terminal(&'a TerminalFoldStep),
}
```

This enum is an execution dispatch over genuinely different types, not a
wrapper around a shared `LevelParams`. Code that can be statically specialized
takes the concrete type directly.

## Exact Expansion Rules

### Group input widths

For each group with block width `M`, live block count `B`, polynomial count
`C`, inner output rank `n_inner`, and digit depths `delta_inner`, `delta_outer`, and
`delta_open`:

```text
inner_commit_input_width = M * delta_inner
outer_commit_input_width = C * B * n_inner * delta_outer
open_segment_input_width  = C * B * delta_open
```

The fold-shared open matrix input width is the checked sum of
`open_segment_input_width` over the root or recursive groups opened at that
fold. Sliced layouts partition exactly the corresponding logical input
interval;
their ranges must be ordered, non-overlapping, gap-free, and cover it once.

### Ring projections

The formulas above are semantic matrix input widths. Changing a role dimension does
not silently multiply or divide this semantic address space. A native matrix
entry contains `ring_dimension` field coefficients, so role-specific storage
changes with the selected dimension.

After semantic input widths are known, the canonical mixed-ring
`SetupProjectionGeometry` views every native matrix footprint over one common
base ring. With shared open dimension `d_open`, it uses:

```text
inner_projection_ratio[g] = d_inner[g] / d_open
outer_projection_ratio[g] = d_outer[g] / d_open
open_projection_ratio     = 1
```

The corresponding relation/witness subcolumn ratios are
`d_inner[g] / d_outer[g]` and `d_inner[g] / d_open`. These ratios route the
native outer/open lanes inside the inner-ring witness representation; they are
not extra semantic matrix inputs. Stage 3 setup footprint and evaluation work
multiply each native `(output_rank * input_width)` footprint by its projection ratio.

No caller may derive an alternative projected input layout. Group order,
semantic input widths, witness ownership, and mixed-ring projection remain separate
canonical objects.

### Matrix storage

For each raw matrix slice:

```text
storage_ring_elements = output_rank * input_width
storage_field_elements = output_rank * input_width * ring_dimension
storage_bytes = storage_field_elements * canonical_field_element_bytes
```

The audit report records, per fold and group:

- exact inner, outer, and open matrix dimensions
  `(output_rank, input_width, ring_dimension)`;
- raw field elements and bytes;
- raw storage for every balanced slice;
- the largest individual matrix and its role;
- largest-outer-object / inner and largest-open-object / inner envelope ratios;
- total-outer / inner and total-open / inner storage ratios; and
- total setup storage separately from the maximum envelope.

For a multi-group fold, the denominator of each `/ inner` ratio is the largest
physical inner matrix object at that fold; the audit also reports every
group-local inner matrix separately.

### Witness and proof sizes

For every fold, expansion derives exact `z_hat`, `e_hat`, `t_hat`, shared tail,
and slice-commitment segments. Distributed layouts replicate only the segments
defined by the distributed-prover protocol. Commitment slicing retains one
commitment vector per slice. A sliced matrix at ring dimension `d` contributes
`sum_i(slice[i].output_rank)` ring elements, equivalently
`d * sum_i(slice[i].output_rank)` base-field elements, to the next witness.

The schedule audit exposes both field-coordinate and serialized-byte
breakdowns. A scalar count without its ring dimension or digit width is not a
sufficient report.

## Schedule Construction and Validation

Generated entries are not trusted expanded schedules. The one expansion path
performs the following checks with checked arithmetic, in this order:

1. Validate the lookup key, root statement, ordered precommitted descriptors,
   protocol epoch, catalog policy identity, and SIS table digest.
2. Recompute every group's exact live geometry from the statement or incoming
   witness. Emitted live counts are equality-checked audit checksums.
3. Validate selected digit widths, ring dimensions, root-final challenge shape,
   partition counts, setup-prefix identities, and balanced slice counts against
   both `MAX_COMMIT_MATRIX_SLICES` and the catalog's implemented capability
   domains.
4. Derive digit depths, matrix input widths, balanced slice boundaries,
   collision bounds, and the minimum SIS-secure output rank for every physical
   matrix object. A generated output rank or slice boundary is never accepted.
5. Validate role-specific ring nesting, setup-generator divisibility, matrix
   role tags, and every SIS table lookup.
6. Derive every input/output witness length and require exact equality across
   root-to-recursive and recursive-to-terminal edges. The terminal response is
   validated by its own shape and is not treated as an output witness.
7. Derive setup-prefix ownership from successor inputs and reject duplicate,
   missing, reordered, or incompatible prefix slots.
8. Derive the terminal response codec, coordinate counts, SIS capacity, and
   serialized payload limit as specified below.
9. Construct and bind the expanded `FoldSchedule` descriptor. Generated replay
   and dynamic planning must produce identical expanded descriptors.

Planner cost estimates are calculated after this validation and are not inputs
to it. The verifier consumes only a validated expanded `FoldSchedule`; it never
trusts serialized ranks, slice boundaries, witness lengths, byte estimates, or
collision bounds supplied by a proof.

The implementation has one authority for each derived concept:

| Concept | Sole authority after cutover |
|---------|------------------------------|
| Exact live block geometry | `BlockGeometry::derive` |
| Digit depths and collision intervals | `DigitPlan::derive` |
| Matrix logical input widths | `CommittedGroupParams::expand` |
| Balanced slice boundaries | `BalancedSliceLayout::derive` |
| Minimum secure output rank | `min_secure_output_rank` |
| Maximum collision capacity of a fixed matrix | `max_secure_collision_linf` |
| Recursive witness ranges and lengths | `WitnessLayout::new` |
| Terminal response shape and codec limit | `TerminalResponseShape::derive` |
| Planner-only byte estimates | `FoldScheduleEstimate::derive` |
| Protocol descriptor bytes | `FoldSchedule::append_descriptor_bytes` |
| Deterministic selection tie-break bytes | `GeneratedFoldScheduleEntry::append_decision_bytes` |

These names replace their current equivalents during the vocabulary cutover;
there are no `_for_level` forwarding helpers or duplicate estimator-side
formulas.

### Offline SIS generation versus runtime validation

The lattice estimator is an offline artifact generator only. Dynamic planning,
generated-schedule expansion, schedule validation, setup loading, and proof
verification do not call or link the estimator. They accept SIS claims only
through the checked-in table identified by `SisTableDigest`.

The current table already has the required orientation. For each
`(security policy, modulus profile, ring dimension, collision bucket)` it stores
the maximum secure input width at every output rank from 1 through 20. Therefore
`max_secure_collision_linf` scans the ordered A-role collision buckets and, for
each bucket, directly tests

```text
input_width <= max_secure_widths_for_bucket[output_rank - 1]
```

with checked rank indexing. It returns the last contiguous supported bucket,
stopping at the first unsupported bucket, and rejects a missing row or an
out-of-range rank. Offline table generation separately validates monotonicity
across collision buckets. The current A domain has 26 collision buckets, so
validation performs at most 26 table probes per fixed inner matrix.

No rank-specific inverse table is added. Such a table would transpose the same
cutoffs, duplicate the security authority, and still require input width as a
lookup dimension. On base commit `e131faf4`, the complete checked-in infinity
table across Q32/Q64/Q128 contains 255 bucket rows, 20 rank cutoffs per row, and
about 72 KiB of generated Rust source. Future profiles extend this same
digest-bound artifact offline rather than introducing estimator calls at
runtime.

### Terminal folded-response validation

Let the expanded terminal inner matrix have input width `m_A`, output rank
`n_A`, ring dimension `d_A`, security policy `P`, modulus profile `Q`, and SIS
table digest `H`. Let `omega` be the exact L1 norm of the terminal fold's flat
challenge distribution and `nu` its trace-subfield embedding norm. For a
centered response acceptance radius `Z`, the direct-response specialization of
the A-role weak-binding formula is

```text
response_difference_linf(Z) = 2 * Z
terminal_A_collision_linf(Z) =
    4 * omega * nu * response_difference_linf(Z)
                               = 8 * omega * nu * Z
```

The `4 * omega * nu` multiplier is the same canonical A-role weak-binding
multiplier used for a certified response-difference interval; the terminal
specialization changes only how that interval is obtained. It uses the exact
diameter `2 * Z` of the accepted centered interval. All products use checked
integer arithmetic.

Define

```text
supported_A_collision_linf = largest prefix endpoint beta_j in
    A_ROLE_COLLISION_BUCKETS such that, for every beta_i <= beta_j,
    min_secure_output_rank(P, H, Q, d_A, beta_i, m_A) <= n_A

terminal_z_linf_limit = min(
    floor(supported_A_collision_linf / (8 * omega * nu)),
    maximum_unique_centered_field_magnitude
)
```

Schedule validation rejects if no such bucket exists or if the terminal
response shape cannot be represented within checked integer and configured
parser limits. The arbitrary inner source decomposition is not range-checked
and contributes no separate A-role collision interval. The helper implementing
the fixed-matrix inversion is the sole authority for the matrix's supported
collision capacity; it scans the checked-in table and does not invoke the
offline lattice estimator.

The planner computes the unsnapped honest terminal-response quantile and
chooses the minimum A output rank that secures
`terminal_A_collision_linf(honest_response_linf)`. It then derives any larger
collision-bucket prefix supported by that same fixed rank. The honest quantile
controls completeness, matrix selection, and expected proof size. It is not a
second verifier bound and is not serialized in the protocol schedule.

For an untrusted terminal response, the verifier performs these steps:

1. Check the payload length prefix against the derived
   `TerminalResponseShape` before allocating payload storage.
2. Decode the lossless centered-integer codec canonically, rejecting malformed
   unary runs, non-canonical encodings, trailing data, coordinate-count
   mismatches, and integers without a unique centered field representative.
3. Compute the actual `z_linf = max_i |z_i|` during decoding using checked
   integer arithmetic.
4. Compute `terminal_A_collision_linf(z_linf)`, round it upward to its A-role
   SIS bucket, and require that bucket not to exceed
   `supported_A_collision_linf`. This is checked integer arithmetic plus a
   checked-in table lookup, not a lattice-estimator run. The check contains no
   opening digit width, digit decomposition, or clean digit-boundary snap.
5. Convert the accepted centered integers to field elements and check the raw
   `e` consistency/trace relations and the raw `t = A * z` relation.

The response-difference factor two is tight for direct centered responses: for
any two accepted responses `z` and `z'`,
`||z - z'||_infinity <= ||z||_infinity + ||z'||_infinity`. The matrix-derived
limit, after the full A-role weak-binding multiplier, therefore certifies every
pair of accepted responses. Recursive digit-certified responses continue to
use their tighter digit-interval difference formula instead.

The initial `TerminalResponseWirePolicyId` uses the current calibrated Rice
offsets but applies them to the unsnapped honest quantile:

```text
honest_log = floor_log2(max(1, honest_response_linf))
rice_low_bits = honest_log.saturating_sub(2)
payload_budget_bits_per_coordinate = honest_log + 2
```

`TerminalResponseShape::derive` computes a protocol `z`-payload byte limit from
the exact coordinate count and the named response-wire policy. The initial
policy retains the current `honest_log + 2` bits-per-coordinate budget with
checked ceiling-to-bytes arithmetic. This is deliberately an average-case
grinding and denial-of-service limit, not a claim that every vector below
`terminal_z_linf_limit` has that encoded size. The prover accepts a grind
attempt only when both the matrix-derived norm check and the payload-byte limit
pass.

The terminal parser applies the byte limit to its bounded reader before
allocating the payload, and the canonical decoder additionally caps unary runs
using `terminal_z_linf_limit`. The descriptor binds the response-wire policy
and the derived shape. Codec choice and byte budget can narrow the accepted
proof set for efficiency, but they never enlarge the matrix-secure norm set;
the actual A-role collision check is the security rule.

## Final-Root-Group Tensor API

The current config hook:

```rust
fn fold_challenge_shape_at_level(
    inputs: AkitaScheduleInputs,
) -> TensorChallengeShape;
```

is deleted. Its replacement cannot name a recursive level:

```rust
pub struct RootFinalGroupPlanningInputs {
    pub statement: RootStatementLayout,
    pub input_witness_len: usize,
}

pub enum RootFinalChallengeFamily {
    Flat,
    Tensor,
}

fn root_final_challenge_family(
    inputs: RootFinalGroupPlanningInputs,
) -> RootFinalChallengeFamily;
```

The optimizer enumerates legal power-of-two tensor low factors only while
constructing the final root-group candidate. It writes the selected exact
`GeneratedRootFinalChallenge::Tensor { fold_low_len }` into the entry.
Generated replay does not optimize or reinterpret the value.

Precommitted root groups and recursive candidate construction always call the
flat challenge calculation. There is no per-precommitted-group or recursive
shape parameter to thread through planner functions.

## Setup-Offload Transition Encoding

For a schedule:

```text
root -> r0 -> r1 -> terminal
```

the producer's setup strategy is derived as follows:

```text
root offloads  <=> r0.incoming_setup_prefix.is_some()
r0 offloads    <=> r1.incoming_setup_prefix.is_some()
r1 is direct   <=> terminal has no setup-prefix field (always true)
```

This replaces redundant producer-side `SetupContributionMode`. Validation
checks that every prefix ID matches the exact setup contribution produced by
its predecessor, including natural length, padded domain, commitment params,
and descriptor digest.

The planner may internally score a candidate edge as direct or recursive, but
the selected schedule is materialized only through successor input shape.

## Catalog Identity and Descriptor Binding

`GeneratedFoldScheduleCatalogIdentity` contains:

```text
family_name
protocol_epoch
field / modulus profile
SIS security policy and table digest
source family and root norm policy
planner cost-model identity
frontier selection-policy identity
allowed inner/outer/open digit-bit domains
allowed inner/outer/open ring-dimension domains
final-root-group challenge family
setup-offload planning policy identity
distributed planning policy identity
slicing capability/version
terminal response wire policy identity
folded-response norm and A-collision policy identity
ring-challenge configuration digest by supported inner ring dimension
sorted lookup-key count and digest
```

It no longer contains one ambiguous `ring_dimension`, a root tensor marker
standing in for row-local geometry, or a global activated-level count that must
be replayed to discover per-level partitioning.

The protocol instance descriptor binds the fully expanded schedule, not merely
the catalog identity. Catalog identity prevents using the wrong table; schedule
binding prevents prover/verifier disagreement about the selected row.

## Planner Search and Audit Model

### Decision variables

The durable planner searches, subject to implemented capabilities:

- fold count and root/recursive/terminal topology;
- per-group block split and exact live-block count;
- inner, outer, and open decomposition digit widths;
- per-group inner and outer ring dimensions;
- per-fold shared open ring dimension;
- exact final-root-group tensor low factor, only for tensor final-group families;
- per-fold witness partition;
- setup-offload edges;
- outer/open balanced slice counts.

### Derived values

The planner derives through canonical functions:

- digit depths;
- matrix input widths;
- honest and certified folded-response bounds;
- collision-difference bounds;
- SIS coefficient buckets and minimum ranks;
- mixed-ring projection widths;
- next witness shapes;
- per-fold and aggregate direct-payload byte estimates;
- per-matrix and total setup storage; and
- verifier work estimates.

Generated rows store selected decisions, not an unchecked duplicate of these
derived results. The emitter also writes a human-readable audit artifact or
test snapshot containing derived values so parameter changes can be reviewed
without reverse-engineering Rust constants.

### Objectives and Pareto output

The core search does not hard-code one scalar objective. The cutover implements
and requires this initial cost-model identity:

```rust
pub enum PlannerCostModelId {
    PayloadEstimateAndStorage,
}
```

`PayloadEstimateAndStorage` uses exactly this dominance vector:

```text
next recursive witness bytes
estimated direct proof payload bytes
maximum individual setup matrix bytes
total setup matrix bytes
total offloaded setup-prefix contribution to successor witnesses
```

It reports root prover, later-fold prover, and verifier matrix-evaluation
structural counts for audit, but those counts do not affect dominance or preset
selection. There is no unspecified work weight in the cutover. Adding work to
dominance later requires a new `PlannerCostModelId` with reviewed formulas and
units; changing the vector under the existing ID is invalid.

All current-main catalog families explicitly bind the initial selection policy:

```rust
pub enum SelectionPolicyId {
    MinEstimatedDirectPayload,
}
```

`MinEstimatedDirectPayload` selects from the retained frontier using this
total lexicographic order:

1. estimated direct proof payload bytes;
2. next recursive witness bytes;
3. maximum individual setup matrix bytes;
4. total setup matrix bytes;
5. total offloaded setup-prefix contribution to successor witnesses; and
6. canonical encoded generated-schedule decisions, bytewise, as the final
   deterministic tie breaker.

There is no library-default policy and no dependence on candidate enumeration
order. A catalog family must supply and identity-bind its policy, but the
cutover assignment is no longer an open decision: every family shipped on
current main supplies `MinEstimatedDirectPayloadV1`. The selected entry and
every frontier point rejected by the policy are written to the audit report.
Future catalog families can introduce another reviewed policy ID without
changing this one's ordering.

### Explicitly deferred surfaces

The following are the only design selections intentionally not made by this
spec, and neither blocks the current-main cutover:

- Commitment compression has no placeholder type in this spec. It enters the
  generated schema only through its protocol spec and capability cut.
- Enabling each post-main capability domain remains release sequencing. A
  capability is absent until its cut satisfies the gates listed below, then its
  exact supported values are part of the catalog identity and validation
  domain.

Everything else shown in the generated/runtime type definitions and validation
algorithm is normative for this cutover.

## Full-Cutover Implementation Plan

This is one merge cutover, not a compatibility migration. It is implemented as
reviewable, compiling commits on one branch, but the branch is
merged only after the old schema, old names, homogeneous topology, and old
terminal accounting are gone. In particular, do not make the topology change
look artificially small by adding forwarding accessors, conversion wrappers,
or a second schedule model that survives the branch.

One commit will necessarily cross many crates: the typed schedule topology is
a single ownership change. Keeping that commit atomic is safer than merging a
half-old/half-new schedule behind adapters. The surrounding semantic and
mechanical changes are separated so that this atomic commit is still
reviewable.

The target type definitions in this spec use the final `digit_bits` vocabulary.
To avoid unnecessary churn while the protocol topology is moving, Cuts 0–4
retain the current `log_basis`, `log_commit_bound`, and `log_open_bound`
spellings internally. Cut 5 performs their repository-wide atomic rename after
all semantic changes have stabilized. No intermediate compatibility alias is
introduced.

### Cut 0: freeze a neutral current-main ledger

- From the exact base commit, emit test fixtures containing effective geometry,
  digit widths, dimensions, output ranks, input widths, witness lengths, planner byte
  estimates, measured proof bytes,
  matrix storage, and descriptor inputs for every shipped catalog entry.
- Store the fixture in a schema-neutral audit format, not serialized old Rust
  types. It remains useful after those types are deleted.
- Add a terminal transcript-order fixture that identifies predecessor-bound
  `t`, pre-challenge `e`, and response `z` without freezing the old
  `SegmentTypedWitness` container.
- Record the intentional terminal-protocol delta separately. All non-terminal
  values are parity targets; terminal bytes and security inputs are recomputed.

### Cut 1: correct the terminal protocol and accounting

- Replace `SegmentTypedWitness` and `TerminalWitnessPlan` with the direct
  `TerminalResponse` and `TerminalResponseShape`.
- Add the canonical SIS-table inversion that derives the terminal inner
  matrix's maximum supported collision bucket from its fixed width, dimension,
  and output rank.
- Make the verifier compute the actual decoded `z` infinity norm and require
  its complete A-role weak-binding collision price to fit the derived bucket.
  The honest unsnapped quantile is used only to select parameters and model
  expected bytes.
- Keep `e` and `t` as raw field elements. Remove their digit-plane-equivalent
  length accounting from wire size and remove terminal `outer_log_basis` and
  `open_log_basis` from planner and schedule inputs.
- Split the predecessor path before outer decomposition: terminal binding
  computes the inner commitment, binds raw `t`, and never constructs `t_hat`.
- Derive the lossless codec parameter from the honest distribution and its
  allocation-safe payload limit from the named response-wire policy.
- Update prover, verifier, transcript tests, serialization, proof sizing, and
  planner terminal costs together. This cut is a deliberate protocol change.

### Cut 2: perform the vocabulary cutover repository-wide

- Introduce descriptive matrix-role types and rename public fields and methods.
- Replace `SisMatrixRole::A/B/D`-style public terminology with
  `Inner/Outer/Open`; retain A/B/D only in mathematical docstrings.
- Replace row/column count APIs with `output_rank` and `input_width`.
  Specifically, `row_len()` becomes `output_rank()` and `col_len()` becomes
  `input_width()`; there are no forwarding aliases.
- Replace every `w_len` spelling with the precise contextual name:
  `input_witness_len`, `output_witness_len`, or `witness_len` when no direction
  exists. A terminal response size is never called a witness length.
- Move ring dimension ownership into each matrix parameter object and delete
  duplicate `LevelParams.ring_dimension` versus `role_dims` authority.
- Deliberately leave the decomposition `log_basis` and bound spellings alone in
  this cut; their independent mechanical rename is reserved for final Cut 5.
- Update setup, prover, verifier, persistence, profiler, tests, and docs in this
  commit. Use compiler errors plus an `rg` zero-match gate as the burn-down
  list; do not retain compatibility names.

### Cut 3: atomically replace the schedule topology

- Add the generated and runtime root, recursive, terminal, and committed-group
  types in this spec.
- Replace `GeneratedFold`, `GeneratedFoldStepWithSetupMetadata`,
  `GeneratedScheduleTableEntry`, `LevelParams`, `Schedule.folds`, and the
  adjacent terminal plan with `GeneratedFoldScheduleEntry` and `FoldSchedule`
  in one cross-crate commit.
- Remove `level_bytes`, `terminal_bytes`, and `total_bytes` from protocol
  schedule structs and descriptor hashing. Planner estimates move to the
  non-protocol `FoldScheduleEstimate` with accurate estimate names.
- Preserve O(1) hot-path scoring through planner-private `CandidateCost` values
  updated incrementally by suffix DP and Pareto search. Materialization
  recomputes the component sum and rejects disagreement with the cached score.
- Move root standalone precommitted groups into `GeneratedRootFold`; move setup
  prefix metadata onto recursive consumers; emit exact final-root-group tensor
  factorization; and make partitioning explicit on every eligible fold.
- Remove challenge-shape selection from standalone-precommit planning and
  descriptors. It always sizes the inner collision relation for a flat draw and
  never calls the final-group tensor policy.
- In multi-group prover/verifier sampling, dispatch the final group through its
  typed selected shape and every later precommitted group through the flat
  sampler while preserving group-index domain separation and transcript order.
- Update planner expansion, generated lookup/sorting/hashing/emission, setup,
  prover, verifier, PCS orchestration, persistence, profiles, and tests against
  the concrete typed steps.
- Delete `get_execution_schedule` and all index-based root/terminal/penultimate
  inference in the same commit. Exhaustive typed dispatch is the only runtime
  bridge and must not reconstitute a shared `LevelParams`.

### Cut 4: regenerate and audit the protocol cutover

- Regenerate every catalog shipped on current main.
- Compare all non-terminal selected decisions and derived values against the
  Cut 0 ledger. Compare terminal values against the new direct-response model,
  with an explicit old/new byte and security-bound report.
- Record the intentional rejection of any old tensor-shaped standalone
  precommitment descriptor and the independently regenerated flat replacement.
- Require table replay and dynamic planning to produce descriptor-identical
  typed schedules.
- Bump the instance/proof descriptor epoch once for the combined cutover and
  reject old schedule, setup, persistence, and proof encodings.
- Run zero-match checks for the protocol types and identifiers retired in Cuts
  1–3, then validate the regenerated catalogs.

### Cut 5: atomically rename decomposition widths and seal the cutover

- Rename `log_basis` to `digit_bits` throughout live Rust, generated source,
  planner/config APIs, serialization code, profiles, diagnostics, tests, and
  implementation-owned documentation.
- Rename `log_commit_bound` to `commit_bound_bits` and `log_open_bound` to
  `open_bound_bits` in the same commit. Rename local `log_bound` variables to
  `bound_bits` where they denote bit widths.
- Use role-qualified names such as `inner_digit_bits`, `outer_digit_bits`, and
  `open_digit_bits` wherever more than one decomposition is in scope. Generic
  decomposition primitives take `digit_bits`; the derived depth remains
  `num_digits`.
- Do not rename mathematical paper notation mechanically. Prose uses
  decomposition radix `b = 2^k`; “basis” remains only where it means an actual
  vector or polynomial basis.
- Add no deprecated fields, forwarding accessors, serde aliases, or dual-name
  constructors. Compiler errors drive the atomic cutover.
- Re-emit generated catalogs and require semantic parameters, descriptor bytes,
  security buckets, witness sizes, and planner estimates to remain unchanged
  from Cut 4. This slice changes vocabulary only.
- Require zero matches for `log_basis`, `log_commit_bound`, and
  `log_open_bound` in live Rust, then run the full repository validation matrix
  on the final branch head.

### Cut 6: capability additions

As implementations land, enable one capability at a time:

1. large inner decomposition digit widths;
2. independently selected inner/outer/open ring dimensions;
3. outer/open matrix slicing;
4. commitment compression;
5. broader setup-offload search; and
6. distributed first and second folds.

Each capability adds search domains and generated variants only after the
runtime relation, verifier validation, setup generation, proof-size formula,
and SIS contract exist. The schedule topology does not change again.

## Evaluation

### Acceptance Criteria

- [ ] Public and generated matrix APIs use `InnerCommitMatrix`,
      `OuterCommitMatrix`, and `OpenCommitMatrix`; A/B/D remain only notation.
- [ ] Final live Rust uses `digit_bits`, `commit_bound_bits`, and
      `open_bound_bits`; `log_basis`, `log_commit_bound`, and `log_open_bound`
      have zero matches, with no compatibility aliases.
- [ ] `GeneratedFoldScheduleEntry` contains typed `root`,
      `recursive_folds`, and `terminal` fields and no homogeneous fold enum.
- [ ] Runtime `FoldSchedule` mirrors the proof topology and contains no
      homogeneous `Vec<FoldStep>`.
- [ ] Tensor challenge types appear only below `RootFinalGroupParams` and
      `GeneratedRootFinalGroup`.
- [ ] No planner/config API accepts a challenge shape for an arbitrary level or
      precommitted group.
- [ ] Every generated tensor final-group entry stores its exact optimized
      `fold_low_len`.
- [ ] Precommitted-root, recursive, and terminal tensor schedules are
      unrepresentable in safe Rust.
- [ ] Arbitrary standalone precommitted groups are root-only.
- [ ] Recursive folds accept zero or one typed incoming setup prefix.
- [ ] `SetupContributionMode` is absent from selected schedules; producer mode
      is derived from successor input topology.
- [ ] Every generated inner, outer, and open matrix carries its own ring
      dimension; outer/open matrices carry only `slice_count`, with one meaning
      monolithic and 16 the protocol-wide maximum.
- [ ] Generated entries carry no output rank, slice boundary, or slice rank;
      expansion derives the minimum secure rank of each physical matrix.
- [ ] Expanded matrices carry exact output rank, input width, coefficient
      bound, ring dimension, role, SIS policy, and table digest.
- [ ] Mixed dimensions validate `d_open | d_outer | d_inner` per group and
      generator divisibility across the entire schedule.
- [ ] Terminal params contain no outer/open matrix, outer/open digit width, or
      digit-plane accounting for raw `e`, `t`, or `z`.
- [ ] Schedule validation derives the maximum A-role collision bucket supported
      by the terminal inner matrix. Verification computes the actual decoded
      `z` infinity norm and requires its complete A-role weak-binding collision
      price to fit the bucket.
- [ ] Terminal security and encoding contain no clean digit-boundary snap; the
      honest unsnapped quantile affects planner completeness and cost only.
- [ ] The predecessor terminal path binds raw `t` without constructing
      `t_hat`; terminal `e` and `t` remain raw field elements on the wire.
- [ ] Generated rows store independent choices while canonical expansion owns
      digit depths, collision bounds, SIS buckets, widths, and bytes.
- [ ] `FoldSchedule` contains no planner byte estimate. Estimates live in
      `FoldScheduleEstimate` and are absent from the instance descriptor.
- [ ] Every current-main catalog family identity-binds
      `PayloadEstimateAndStorage` and `MinEstimatedDirectPayload`; frontier
      selection follows the specified total order and is invariant under
      candidate enumeration order.
- [ ] Candidate comparison reads an incrementally maintained planner-private
      aggregate in O(1); final estimate derivation equality-checks that cache
      against the per-fold component sum.
- [ ] Current-main generated catalogs retain non-terminal selected parameters
      and costs; terminal deltas match the reviewed direct-response fixture.
- [ ] The exact nine-fold migration fixture in this spec has a regression test.
- [ ] Table replay and dynamic planning produce equal schedule descriptors for
      every emitted lookup key.
- [ ] Descriptor mutation tests cover topology, every role dimension and rank,
      every digit width, final-root-group tensor factor, block geometry,
      partitions, prefix IDs, and balanced slice counts.
- [ ] Verifier-facing malformed schedule and serialization tests return errors
      without panics or unchecked allocations.
- [ ] Generated audit output reports exact inner/outer/open matrix dimensions,
      storage, ratios, setup envelope, witness components, planner byte
      estimates, and measured proof bytes where available.
- [ ] Repository format, line-limit, dependency, documentation, Clippy, and
      relevant nextest gates pass on the final branch head.

### Testing Strategy

#### Compile-time topology tests

Rust type ownership is the primary test: only the final-root-group structs have
a tensor selector; precommitted-root, recursive, and terminal structs do not.
Recursive and terminal structs also have no arbitrary-group fields.
The cutover does not add a new compile-test framework. Ordinary construction
tests and exhaustive matches demonstrate the legal surface.

#### Generated parity tests

For every current catalog family:

1. load the old expected fixture captured from base commit `e131faf4`;
2. resolve the same lookup key through the new generated row;
3. resolve it through dynamic planning;
4. compare effective semantic parameters and proof sizes; and
5. compare generated versus dynamic descriptors.

The old Rust types are not compiled as a compatibility module. Expected
fixtures are neutral snapshots containing semantic fields.

#### Planner-score cache tests

- Extending a suffix updates every cached byte coordinate with checked
  arithmetic and rejects overflow.
- Every retained frontier point and selected schedule has a materialized
  `FoldScheduleEstimate` whose recomputed aggregate equals its `CandidateCost`.
- Mutating any estimate fixture component changes the recomputed aggregate and
  makes the cache-consistency check reject.

#### Final-root-group tensor tests

- Flat presets emit `GeneratedRootFinalChallenge::Flat`.
- Tensor presets emit the exact low factor selected for the final root group.
- Every precommitted root group takes the flat sampling path and has no
  serialized shape selector.
- Multi-group transcript tests cover a tensor final group followed by one or
  more flat precommitted groups in canonical group order.
- Replay never calls the tensor optimizer.
- Changing the exact low factor changes the descriptor and the certified fold
  bound.
- Every precommitted root group, recursive fold, and terminal fold follows the
  flat transcript path.

#### Setup-offload tests

- Prefix presence on a recursive consumer exactly determines predecessor
  offloading.
- A direct consumer may have an incoming prefix but its successor does not.
- Root and terminal cannot carry incoming prefixes.
- Prefix matrix dimensions, digit widths, geometry, natural length, and slot identity
  are descriptor-bound and independently validated.

#### Mixed-ring tests

- Uniform current-main schedules remain unchanged.
- Legal nested combinations, including `64/16/16`, `128/64/16`, and
  group-local mixed root dimensions, expand through canonical projection.
- Non-dividing dimensions, unsupported inner challenge dimensions, generator
  incompatibility, or key/role mismatch reject.
- Matrix storage reports use the matrix owner's dimension.

#### Balanced-slicing tests

- `W mod S = 0` produces `S` equal contiguous widths.
- `W mod S != 0` assigns one extra input to exactly the first `W mod S`
  slices.
- `S = 0`, `S > W`, `S > MAX_COMMIT_MATRIX_SLICES`, and every
  checked-arithmetic overflow reject; `S = 1` expands to one full-width
  physical matrix.
- Each expanded slice output rank equals `min_secure_output_rank` for its own
  derived width; generated entries contain no rank or boundary override.
- The next-witness contribution equals the sum of all slice output ranks times
  the matrix ring dimension.

#### Terminal-response validation tests

- Schedule validation rejects an inner matrix that does not cover its source
  collision bound or has no supported terminal collision capacity.
- The collision-capacity helper accepts `terminal_z_linf_limit` and rejects the
  first larger representable norm. End-to-end response fixtures separately
  satisfy the descriptor-bound payload-byte budget.
- Malformed, non-canonical, truncated, over-budget, trailing-data, and wrong
  coordinate-count encodings reject without allocation from unchecked lengths.
- Changing the SIS policy, table digest, inner width, dimension, or output rank
  changes or invalidates the derived terminal response limit.
- Planner honest-quantile changes affect matrix selection and estimated bytes,
  but the verifier reads no independent quantile-derived or
  digit-boundary-snapped bound.

#### Serialization and no-panic tests

Fuzz or table-driven tests cover excessive group/fold/slice counts, zero
dimensions, arithmetic overflow in derived balanced slices, malformed prefix
IDs, invalid tensor low factors, and inconsistent witness-length transitions.

### Performance

The topology/terminology cutover has a zero-regression target for selected
current-main parameters, planner byte estimates, measured proof bytes, setup
bytes, and prover/verifier work.
Generated table source size is allowed to change. The objective gates are the
repository Rust file-line cap, generated-table validation, release build, and
reported compile/binary-size comparison; there is no subjective "reasonable
size" exception.

Subsequent planner capabilities are evaluated on their Pareto metrics rather
than against a blanket no-regression rule. Any selected production change must
ship an audit comparison showing planner byte estimates, measured proof bytes,
witness components, exact matrix storage and envelope, and estimated
prover/verifier work.

## Alternatives Considered

### Keep a homogeneous fold vector and validate indices

Rejected. It preserves the exact source of the current ambiguity and makes
root-only capabilities representable at other levels.

### Keep tensor on `LevelParams` but reject non-root use

Rejected. The feature is not useful outside the root and should not burden
every recursive schedule, descriptor, test, or verifier branch forever.

### Allow tensor selection independently for every root group

Rejected. Current main can represent this, but intended applications tensor
only the large final group. Per-precommitted-group selection would keep tensor
shape in frozen commitment descriptors, planner lookup keys, transcript shape
validation, norm sizing, generated entries, and verifier dispatch without a
corresponding use case. Precommitted groups retain independent challenge draws,
but those draws are unconditionally flat.

### Keep `A/B/D` as public names

Rejected. The letters are compact in formulas but fail to communicate matrix
ownership in planner, setup, and verifier code. Descriptive names make mixed
ring dimensions and storage reports substantially clearer.

### Store one `CommitmentRingDims` beside three matrix keys

Rejected. It creates two authorities. Each matrix key/plan owns its dimension;
cross-role nesting is a validation over those owners.

### Store every derived value in generated rows

Rejected. Duplicating digit depths, widths, collision bounds, and SIS buckets
would create a split-brain security contract. Emit these values in audit output,
but derive and validate them through canonical runtime functions.

### Store arbitrary slice boundaries and ranks in generated entries

Rejected. The slicing lever needed by the planner is the number of physical
matrices, not an arbitrary partition. Balanced contiguous partitioning is
deterministic, and every slice rank is fixed by its derived width and the SIS
table. Authoring boundaries or ranks would enlarge the search space and create
new validation authority without serving an intended application.

### Keep proof-byte estimates inside the protocol schedule

Rejected. The current aggregate is a planner approximation rather than an
exact serialized size or upper bound, and it excludes recursive-mode payloads.
Binding it in the instance descriptor turns a cost-model revision into an
unnecessary protocol change. `FoldScheduleEstimate` is the non-protocol owner.

### Keep the runtime type named `Schedule`

Rejected. The crate contains setup, opening, transcript, and planner schedules.
`FoldSchedule` states exactly which execution plan the type represents.

### Hide future features behind generic maps or extension bags

Rejected. Protocol features deserve typed variants with descriptor and
validation coverage. Generic metadata would make illegal combinations easy to
construct and hard to audit.

## Documentation

- Keep this spec active through the implementation/cutover branch.
- Update `book/src/how/configuration.md` when the typed schedule and planner
  surfaces stabilize.
- Update `book/src/how/recursion.md` for the typed root/recursive/terminal
  topology.
- Update `book/src/how/proving/root-fold-ring-switch.md` to state that tensor
  challenges are permanently final-root-group-only and precommitted root
  groups are flat.
- Update `book/src/how/verifying/matrix_evaluation.md` to use descriptive matrix
  names with A/B/D notation in parentheses.
- Update `book/src/how/architecture.md` for the runtime parameter types.
- When implementation is complete and durable content is folded into the book,
  mark this spec implemented and archive it according to `specs/PRUNING.md`.

## References

- [`crates/akita-planner/src/generated/mod.rs`](../crates/akita-planner/src/generated/mod.rs)
- [`crates/akita-planner/src/generated/walk.rs`](../crates/akita-planner/src/generated/walk.rs)
- [`crates/akita-types/src/schedule.rs`](../crates/akita-types/src/schedule.rs)
- [`crates/akita-types/src/layout/params.rs`](../crates/akita-types/src/layout/params.rs)
- [`crates/akita-types/src/layout/ring_dims.rs`](../crates/akita-types/src/layout/ring_dims.rs)
- [`crates/akita-types/src/proof/levels.rs`](../crates/akita-types/src/proof/levels.rs)
- [`specs/setup-offloading-planner.md`](setup-offloading-planner.md)
- [`specs/multi-group-batching.md`](multi-group-batching.md)
- [`specs/distributed-planner.md`](distributed-planner.md)
- [`specs/tensor-structured-folding-challenges.md`](tensor-structured-folding-challenges.md)
- [`specs/commitment-compression-cutover.md`](commitment-compression-cutover.md)
- [`specs/terminal-direct-ring-relations-cutover.md`](terminal-direct-ring-relations-cutover.md)
- [`specs/digit-innermost-layout.md`](digit-innermost-layout.md)
