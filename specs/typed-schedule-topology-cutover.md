# Spec: Typed Schedule Topology and Planner Cutover

| Field         | Value                                       |
|---------------|---------------------------------------------|
| Author(s)     | Quang Dao                                   |
| Created       | 2026-07-21                                  |
| Status        | active                                      |
| PR            |                                             |
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

The root type exclusively owns root source structure, tensor challenges, and
arbitrary precommitted groups. Recursive folds always use flat challenges and
may consume at most one incoming setup-prefix commitment. The terminal type
owns the final committed fold and cleartext witness handoff and cannot carry
root-only or recursive-setup features.

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
root tensor factorization, decomposition bases, ranks, block geometry,
per-fold witness partitioning, setup-prefix inputs, and future matrix slicing
or commitment compression choices. Derived widths, digit depths, collision
bounds, witness lengths, and byte counts remain canonical calculations and are
validated during expansion instead of becoming competing sources of truth.

This work starts from `origin/main` commit
`e131faf48938b975ca63b12b59ac6d86894048e0` (PR #312). It does not wait for or
assume any open PR. Akita has no backward-compatibility requirement, so the
implementation is a direct cutover with no legacy schedule adapter.

## Intent

### Goal

Replace the flattened schedule and overloaded `LevelParams` representation
with role-specific generated and runtime types that make the legal protocol
topology explicit, make tensor challenges structurally root-only, expose all
matrix-role ring dimensions, and provide stable extension points for mixed
rings, distributed witness partitions, setup offloading, slicing, and
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

- Tensor challenges are supported only by `RootFoldParams` and
  `GeneratedRootFold`.
- Recursive and terminal parameter types do not contain a challenge-shape
  field. Their fold challenge is flat by protocol definition.
- A generated root row stores the exact selected `fold_low_len`. A value such
  as `Tensor { fold_low_len: 2 }` is never used merely as an enablement marker.
- Each independently committed group participating in the root fold carries
  its own root challenge geometry. This includes the final group and any
  standalone precommitted groups opened in the same root batch.
- A setup prefix consumed by a recursive fold always uses flat challenge
  geometry. Tensor metadata cannot be serialized into a setup-prefix slot.
- The planner config surface exposes a root challenge-family policy, not a
  callback accepting an arbitrary fold level.

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

- Inner, outer, and opening decomposition bases are independent planner
  choices. In particular, the inner source decomposition is not constrained by
  a range check unless a concrete protocol relation requires it. These outer
  and opening choices exist only at non-terminal folds that actually construct
  the corresponding decompositions and commitment relations.
- Generated entries store selected log bases and matrix output ranks. They do not author
  digit depths, coefficient bounds, or SIS table buckets.
- Expansion derives digit depths and certified coefficient bounds using the
  same canonical functions consumed by verifier validation and SIS sizing.
- `AjtaiKeyParams::try_new` or its renamed canonical successor audits every
  expanded `(role, ring dimension, output rank, input width, coefficient
  bound)` tuple.
- A recursive folded response uses the protocol's digit-certification relation
  and its tight difference interval. A terminal folded response instead uses
  the direct accepted interval checked by the verifier. Both incorporate the
  canonical one-hot average-case bound, snap rule, tensor factor, and
  trace-subfield norm factor where applicable. Generated entries cannot
  substitute an optimistic bound.
- Frozen standalone commitments bind their exact security descriptor. On
  replay, the descriptor is rederived and equality-checked; it is not accepted
  as an unaudited override.
- There is one canonical calculation for each matrix column count, collision
  bound, witness width, setup prefix length, and proof byte count.

#### Generated catalogs

- A generated row contains all independent choices required to reproduce the
  effective schedule without rerunning an optimizer. Exact live geometry may
  also be emitted redundantly as an auditable checksum, but replay must
  rederive it from the statement or predecessor and require equality.
- A catalog identity contains search/security policy identity, not row-local
  decisions. Exact tensor factorization and per-level partitioning live in the
  row.
- Table expansion and dynamic planning produce descriptor-identical runtime
  schedules for the same lookup key and policy.
- Generated lookup order and key digests include the complete root statement:
  final group plus ordered standalone precommitted commitment descriptors.
- Generated catalogs with different source families, root challenge families,
  chunk policies, setup-offload policies, matrix dimension domains, slicing,
  compression, or SIS table digests cannot alias.

#### Transcript, serialization, and safety

- The instance descriptor binds topology tags, ordered groups, exact root
  challenge shape, all matrix dimensions and ranks, decomposition bases, block
  geometry, witness partitions, setup-prefix identities, slicing/compression
  plans, witness lengths, and terminal response shape.
- Serialization uses explicit root, recursive, and terminal sections. It does
  not serialize a homogeneous fold list and infer roles during decoding.
- Malformed counts, dimensions, slice ranges, prefix identities, or arithmetic
  overflow return `AkitaError` or `SerializationError`. Verifier-reachable code
  does not panic or allocate from unchecked schedule-controlled dimensions.
- Schedule and proof descriptor changes intentionally define a new protocol
  epoch. Old generated rows, setup artifacts, proofs, and descriptors are not
  accepted through compatibility shims.

### Non-Goals

- Choosing new production parameters in the topology cutover itself. The first
  regeneration must preserve current-main planner choices while changing their
  representation.
- Treating open PR behavior as landed. Later commits on this branch may add
  features only after their implementation and canonical formulas are present.
- Reimplementing the SIS estimator or maintaining a second security model in
  the emitter.
- Preserving source compatibility for `GeneratedFold`, `LevelParams.a_key`,
  `LevelParams.b_key`, `LevelParams.d_key`, `CommitmentRingDims`, or the
  per-level fold-shape callback.
- Supporting tensor challenges at recursive or terminal folds, now or later.
- Adding arbitrary precommitted groups to recursive folds.
- Encoding speculative compression or slicing semantics before their protocol
  relations land. The topology reserves typed extension points, but only
  implemented variants may be emitted.

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
- three log bases;
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

### Exact migration fixture

The current generated `fp128_d64_onehot_recursive` row for a 32-variable,
two-polynomial final group and two 16-variable standalone precommitted groups is
the primary topology migration fixture. It contains nine folds:

| Protocol level | Current variant | `log2` bases inner/outer/open | Position bits | Block bits | Live blocks | `d_inner/d_outer/d_open` | Inner/outer/open output ranks | Incoming prefix natural length | Outgoing setup mode |
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

The L8 outer/open basis, dimensions, and output ranks in this table document
current-main's homogeneous representation; they are not target protocol
parameters. The cutover retains only L8's inner basis, inner ring dimension,
inner output rank, geometry, and derived direct-response bound/shape.

The root final group has exact source geometry:

```text
layout                         = (num_vars=32, num_polynomials=2)
source                         = one-hot, chunk_size=256
d_inner                        = 64
live ring elements per claim   = 67,108,864
positions per block            = 32,768
live blocks                    = 2,048
root challenge                 = Flat
inner basis / output rank      = 3 / 5
outer basis / output rank      = 3 / 2
shared open basis / output rank = 3 / 1
```

Each of the two standalone root groups has:

```text
layout                         = (num_vars=16, num_polynomials=1)
d_inner / d_outer              = 64 / 64
live ring elements per claim   = 1,024
positions per block            = 32
live blocks                    = 32
root challenge                 = Flat
inner basis / output rank      = 2 / 4
outer basis / output rank      = 2 / 2
```

The L1 incoming setup prefix has `N=2,097,152`, `M=2,048`, `B=1,024`,
inner/outer/open bases `3/3/3`, inner/outer output ranks `7/1`, and uniform dimension
64. The L2 incoming setup prefix has `N=1,048,576`, `M=2,048`, `B=512`, bases
`5/5/5`, output ranks `7/2`, and uniform dimension 64.

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
log_basis_inner          -> inner_log_basis
log_basis_outer          -> outer_log_basis (non-terminal committed groups)
log_basis_open           -> open_log_basis (folds with an opening matrix)
inner_width              -> inner_commit_input_width
outer_width              -> outer_commit_input_width
d_matrix_width           -> open_commit_input_width
```

Compact generated source may use constructors to avoid repeating long field
names, but public types and constructor parameters remain descriptive. Do not
introduce aliases that keep both vocabularies alive. Mathematical helpers may
use local variables `a`, `b`, or `d` when directly transcribing a formula.

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
    /// Number of output ring elements; the SIS module rank (n_B).
    pub output_rank: usize,
    /// Number of input ring elements accepted by the matrix.
    pub input_width: usize,
    pub coeff_linf_bound: u128,
    pub sis_table_key: SisTableKey,
    pub layout: CommitMatrixLayout,
    pub output: CommitmentOutput,
}

pub struct OpenCommitMatrixParams {
    pub ring_dimension: usize,
    /// Number of output ring elements; the SIS module rank (n_D).
    pub output_rank: usize,
    /// Number of input ring elements accepted by the matrix.
    pub input_width: usize,
    pub coeff_linf_bound: u128,
    pub sis_table_key: SisTableKey,
    pub layout: CommitMatrixLayout,
    pub output: CommitmentOutput,
}
```

These are not thin wrappers around a public generic matrix object. They own
role-specific validation and behavior. One private canonical SIS audit
function accepts `CommitMatrixRole` plus the complete tuple and returns the
`SisTableKey`; the three constructors call it rather than duplicating security
logic.

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedInnerCommitMatrix {
    pub ring_dimension: u32,
    pub log_basis: u32,
    pub output_rank: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedOuterCommitMatrix {
    pub ring_dimension: u32,
    pub log_basis: u32,
    pub layout: GeneratedCommitMatrixLayout,
    pub output: GeneratedCommitmentOutput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedOpenCommitMatrix {
    pub ring_dimension: u32,
    pub log_basis: u32,
    pub layout: GeneratedCommitMatrixLayout,
    pub output: GeneratedCommitmentOutput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratedCommitMatrixLayout {
    Monolithic {
        output_rank: u32,
    },
    Sliced {
        slices: &'static [GeneratedCommitMatrixSlice],
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedCommitMatrixSlice {
    /// First input coordinate in the unsliced logical matrix.
    pub input_start: u64,
    /// Exact number of logical input coordinates in this slice.
    pub input_width: u64,
    /// SIS-secure output module rank for this slice.
    pub output_rank: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratedCommitmentOutput {
    Raw,
    Compressed(GeneratedCommitmentCompression),
}
```

`GeneratedCommitmentCompression` is added by the compression implementation
and must contain every independent compression-map choice required by
`specs/commitment-compression-cutover.md`. Until then, only `Raw` is legal.
Likewise, only `Monolithic` is emitted until sliced commitment relations are
implemented. Reserving a typed variant is not permission for the planner to
emit it early.

An inner commitment matrix is not sliced by this design. Slicing exists to
trade outer/open setup storage against additional public slice commitments.
If a future design establishes a useful inner slicing relation, it requires a
separate spec and explicit witness-cost model.

For a monolithic matrix, exact flat setup storage is:

```text
matrix_field_elements = output_rank * input_width * ring_dimension
```

For a sliced matrix it is the sum of this quantity over slices. Compression
matrix storage is accounted separately. The setup envelope is the maximum
stored matrix object, not the sum, while total setup storage reports the sum.

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
pub enum GeneratedRootChallenge {
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
    pub challenge: GeneratedRootChallenge,
    pub commitment: GeneratedCommittedGroup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedRootPrecommittedGroup {
    /// Frozen standalone commitment identity and certified bounds.
    pub descriptor: PrecommittedGroupDescriptor,
    /// Exact root-opening plan; must rederive the descriptor.
    pub challenge: GeneratedRootChallenge,
    pub commitment: GeneratedCommittedGroup,
}
```

The precommitted descriptor is part of the schedule lookup key because the
commitment already exists. Expansion rederives its matrix dimensions, input widths,
bounds, and root challenge geometry and requires descriptor equality.

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

This is the only generated type that mentions `GeneratedRootSource`,
`GeneratedRootChallenge`, or arbitrary precommitted groups.

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

The setup-prefix input does not repeat an opening basis or open matrix. It uses
the consuming fold's shared `open_commit_matrix`. Its inner and outer matrices
may use different ring dimensions from the ordinary witness group, subject to
the nesting and setup-generator constraints.

### Terminal fold

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedTerminalFold {
    pub geometry: GeneratedBlockGeometry,
    pub inner_commit_matrix: GeneratedInnerCommitMatrix,
}
```

The terminal relation has no outer or open commitment matrix and no terminal
outer/open decomposition basis. Its response contains raw field-valued `t` and
`e`, plus centered folded-response coefficients `z` encoded losslessly. The
verifier decodes `z`, checks every coefficient directly against the canonical
accepted interval, and then checks the opening and inner-commitment relations.
It neither receives nor verifies a digit decomposition of terminal `t`, `e`,
or `z`.

The terminal folded-response interval is a derived security value, not a
generated choice. Expansion derives it from the honest response distribution,
the configured average-case failure probability, the exact one-hot/tensor/trace
factors where applicable, and the canonical snap rule. The inner-matrix SIS
calculation uses the width of the accepted interval exactly. Lossless response
encoding parameters are derived from that same interval and do not create a
second bound.

The predecessor terminal-binding path computes the inner commitment and binds
raw `t`. It must not decompose `t` merely to satisfy a shared non-terminal code
path. Likewise, raw terminal `e` is checked by its trace and consistency
relations and does not acquire an opening basis for accounting purposes.

### Complete table entry

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedScheduleTableEntry {
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
sizes. Matrix params contain exact output rank, input width, ring dimension,
collision bound, and SIS identity. Digit depths are expanded results, never
independent generated inputs.

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
    pub response_bound: TerminalResponseBound,
    pub response_shape: TerminalResponseShape,
}
```

Only `RootFinalGroupParams` and `RootPrecommittedGroupParams` contain a
`RootChallenge` field. `RecursiveFoldParams` and `TerminalFoldParams` are flat
by type. Sparse sampler configuration remains explicit because it determines
challenge distribution and certified norms even for a flat challenge.

`TerminalCommittedGroupParams` contains the geometry, inner matrix, and inner
source-decomposition depth required by the terminal relation. It contains no
outer/open matrix, no terminal outer/open basis, and no fabricated digit depth
for raw response values.

`TerminalResponseBound` is the verifier's exact accepted centered interval for
each `z` coefficient. `TerminalResponseShape` separately counts encoded `z`
coefficients, raw `e` field elements, raw `t` field elements, and the checked
serialized-byte limit. Neither type expresses these raw values as digit-plane
equivalents.

### Typed schedule steps

```rust
pub struct RootFoldStep {
    pub params: RootFoldParams,
    pub input_witness_len: usize,
    pub output_witness_len: usize,
    pub proof_bytes: usize,
}

pub struct RecursiveFoldStep {
    pub params: RecursiveFoldParams,
    pub input_witness_len: usize,
    pub output_witness_len: usize,
    pub proof_bytes: usize,
}

pub struct TerminalFoldStep {
    pub params: TerminalFoldParams,
    pub input_witness_len: usize,
    pub proof_bytes: usize,
    pub terminal_response_bytes: usize,
}

pub struct Schedule {
    pub root: RootFoldStep,
    pub recursive_folds: Vec<RecursiveFoldStep>,
    pub terminal: TerminalFoldStep,
    pub total_bytes: usize,
}
```

The terminal cleartext response is part of `TerminalFoldStep`; it is not a
second object adjacent to a homogeneous fold vector. A terminal step has no
`output_witness_len`: recursion ends there, and its raw response shape is not a
fictional decomposed recursive witness.

The proof wire uses an equally direct type:

```rust
pub struct TerminalResponse<F: LatticeField> {
    /// Lossless centered encoding; every decoded coefficient is range-checked.
    pub folded_response_payloads: Vec<Vec<u8>>,
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
should take the concrete type directly.

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

No caller may derive an alternative projected column layout. Group order,
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
- slicing and compression map storage;
- the largest individual matrix and its role;
- `outer / inner` and `open / inner` storage ratios; and
- total setup storage separately from the maximum envelope.

### Witness and proof sizes

For every fold, expansion derives exact `z_hat`, `e_hat`, `t_hat`, shared tail,
and slice-commitment segments. Distributed layouts replicate only the segments
defined by the distributed-prover protocol. Commitment slicing adds exactly the
public outputs retained by its relation. Compression replaces only the outputs
specified by its encoding plan.

The schedule audit exposes both field-coordinate and serialized-byte
breakdowns. A scalar count without its ring dimension or digit basis is not a
sufficient report.

## Root-Only Tensor API

The current config hook:

```rust
fn fold_challenge_shape_at_level(
    inputs: AkitaScheduleInputs,
) -> TensorChallengeShape;
```

is deleted. Its replacement cannot name a recursive level:

```rust
pub struct RootFoldPlanningInputs {
    pub statement: RootStatementLayout,
    pub input_witness_len: usize,
}

pub enum RootChallengeFamily {
    Flat,
    Tensor,
}

fn root_challenge_family(
    inputs: RootFoldPlanningInputs,
) -> RootChallengeFamily;
```

The optimizer enumerates legal power-of-two tensor low factors only while
constructing root group candidates. It writes the selected exact
`GeneratedRootChallenge::Tensor { fold_low_len }` into the row. Generated replay
does not optimize or reinterpret the value.

Recursive candidate construction always calls the flat challenge calculation.
There is no recursive shape parameter to thread through planner functions.

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

`GeneratedScheduleCatalogIdentity` is revised to contain:

```text
family_name
protocol_epoch
field / modulus profile
SIS security policy and table digest
source family and root norm policy
allowed inner/outer/open log-basis domains
allowed inner/outer/open ring-dimension domains
root challenge family
setup-offload planning policy identity
distributed planning policy identity
slicing capability/version
compression capability/version
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
- inner, outer, and open decomposition bases;
- per-group inner and outer ring dimensions;
- per-fold shared open ring dimension;
- inner, outer, and open SIS-secure row counts;
- exact root tensor low factor, only for tensor root families;
- per-fold witness partition;
- setup-offload edges;
- outer/open slice layouts; and
- commitment compression plans.

### Derived values

The planner derives through canonical functions:

- digit depths;
- matrix input widths;
- honest and certified folded-response bounds;
- collision-difference bounds;
- SIS coefficient buckets and minimum ranks;
- mixed-ring projection widths;
- next witness shapes;
- proof bytes;
- per-matrix and total setup storage; and
- verifier work estimates.

Generated rows store selected decisions, not an unchecked duplicate of these
derived results. The emitter also writes a human-readable audit artifact or
test snapshot containing derived values so parameter changes can be reviewed
without reverse-engineering Rust constants.

### Objectives and Pareto output

The core search does not hard-code one scalar objective. It emits a Pareto set
over at least:

```text
next recursive witness bytes
total proof bytes
root prover work
later-fold prover work
maximum individual setup matrix bytes
total setup matrix bytes
verifier matrix-evaluation work
offloaded setup-prefix contribution to successor witnesses
```

Preset selection applies a named, digest-bound policy to that frontier. The
selected row and rejected neighboring frontier points are available in the
audit report.

## Full-Cutover Implementation Plan

This is one merge cutover, not a compatibility migration. It should be
implemented as reviewable, compiling commits on one branch, but the branch is
merged only after the old schema, old names, homogeneous topology, and old
terminal accounting are gone. In particular, do not make the topology change
look artificially small by adding forwarding accessors, conversion wrappers,
or a second schedule model that survives the branch.

One commit will necessarily cross many crates: the typed schedule topology is
a single ownership change. Keeping that commit atomic is safer than merging a
half-old/half-new schedule behind adapters. The surrounding semantic and
mechanical changes are separated so that this atomic commit is still
reviewable.

### Cut 0: freeze a neutral current-main ledger

- From the exact base commit, emit test fixtures containing effective geometry,
  bases, dimensions, output ranks, input widths, witness lengths, proof bytes,
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
  `TerminalResponse`, `TerminalResponseBound`, and `TerminalResponseShape`.
- Make the canonical bound function return the accepted centered interval.
  The verifier checks decoded `z` coefficients directly, and inner-matrix SIS
  sizing consumes the exact interval width.
- Keep `e` and `t` as raw field elements. Remove their digit-plane-equivalent
  length accounting from wire size and remove terminal `outer_log_basis` and
  `open_log_basis` from planner and schedule inputs.
- Split the predecessor path before outer decomposition: terminal binding
  computes the inner commitment, binds raw `t`, and never constructs `t_hat`.
- Derive lossless folded-response coding parameters solely from the certified
  interval. Encoding and verification share that one bound.
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
- Update setup, prover, verifier, persistence, profiler, tests, and docs in this
  commit. Use compiler errors plus an `rg` zero-match gate as the burn-down
  list; do not retain compatibility names.

### Cut 3: atomically replace the schedule topology

- Add the generated and runtime root, recursive, terminal, and committed-group
  types in this spec.
- Replace `GeneratedFold`, `GeneratedFoldStepWithSetupMetadata`, `LevelParams`,
  `Schedule.folds`, and the adjacent terminal plan in one cross-crate commit.
- Move root standalone precommitted groups into `GeneratedRootFold`; move setup
  prefix metadata onto recursive consumers; emit exact root tensor
  factorization; and make partitioning explicit on every eligible fold.
- Update planner expansion, generated lookup/sorting/hashing/emission, setup,
  prover, verifier, PCS orchestration, persistence, profiles, and tests against
  the concrete typed steps.
- Delete `get_execution_schedule` and all index-based root/terminal/penultimate
  inference in the same commit. Exhaustive typed dispatch is the only runtime
  bridge and must not reconstitute a shared `LevelParams`.

### Cut 4: regenerate, audit, and seal the epoch

- Regenerate every catalog shipped on current main.
- Compare all non-terminal selected decisions and derived values against the
  Cut 0 ledger. Compare terminal values against the new direct-response model,
  with an explicit old/new byte and security-bound report.
- Require table replay and dynamic planning to produce descriptor-identical
  typed schedules.
- Bump the instance/proof descriptor epoch once for the combined cutover and
  reject old schedule, setup, persistence, and proof encodings.
- Run zero-match checks for every retired type and identifier, then the full
  repository validation matrix.

### Cut 5: capability additions

As implementations land, enable one capability at a time:

1. large inner decomposition bases;
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
- [ ] `GeneratedScheduleTableEntry` contains typed `root`,
      `recursive_folds`, and `terminal` fields and no homogeneous fold enum.
- [ ] Runtime `Schedule` mirrors the proof topology and contains no homogeneous
      `Vec<FoldStep>`.
- [ ] Tensor challenge types appear only below the root schedule/group types.
- [ ] No planner/config API accepts a challenge shape for an arbitrary level.
- [ ] Every generated tensor row stores its exact optimized `fold_low_len`.
- [ ] Recursive and terminal tensor schedules are unrepresentable in safe Rust.
- [ ] Arbitrary standalone precommitted groups are root-only.
- [ ] Recursive folds accept zero or one typed incoming setup prefix.
- [ ] `SetupContributionMode` is absent from selected schedules; producer mode
      is derived from successor input topology.
- [ ] Every generated inner, outer, and open matrix carries its own ring
      dimension and rank or slice ranks.
- [ ] Expanded matrices carry exact output rank, input width, coefficient
      bound, ring dimension, role, SIS policy, and table digest.
- [ ] Mixed dimensions validate `d_open | d_outer | d_inner` per group and
      generator divisibility across the entire schedule.
- [ ] Terminal params contain no outer/open matrix, outer/open basis, or
      digit-plane accounting for raw `e`, `t`, or `z`.
- [ ] Terminal verification directly checks every decoded centered `z`
      coefficient against the canonical interval, and SIS sizing uses that
      interval's exact difference width.
- [ ] The predecessor terminal path binds raw `t` without constructing
      `t_hat`; terminal `e` and `t` remain raw field elements on the wire.
- [ ] Generated rows store independent choices while canonical expansion owns
      digit depths, collision bounds, SIS buckets, widths, and bytes.
- [ ] Current-main generated catalogs retain non-terminal selected parameters
      and costs; terminal deltas match the reviewed direct-response fixture.
- [ ] The exact nine-fold migration fixture in this spec has a regression test.
- [ ] Table replay and dynamic planning produce equal schedule descriptors for
      every emitted lookup key.
- [ ] Descriptor mutation tests cover topology, every role dimension and rank,
      every basis, root tensor factor, block geometry, partitions, prefix IDs,
      and future slicing/compression variants.
- [ ] Verifier-facing malformed schedule and serialization tests return errors
      without panics or unchecked allocations.
- [ ] Generated audit output reports exact inner/outer/open matrix dimensions,
      storage, ratios, setup envelope, witness components, and proof bytes.
- [ ] Repository format, line-limit, dependency, documentation, Clippy, and
      relevant nextest gates pass on the final branch head.

### Testing Strategy

#### Compile-time topology tests

Rust type ownership is the primary test: recursive and terminal structs simply
have no tensor or arbitrary-group fields. UI/compile-fail tests are optional;
ordinary construction and exhaustive-match tests should demonstrate the legal
surface without adding a new compile-test framework solely for this change.

#### Generated parity tests

For every current catalog family:

1. load the old expected fixture captured from base commit `e131faf4`;
2. resolve the same lookup key through the new generated row;
3. resolve it through dynamic planning;
4. compare effective semantic parameters and proof sizes; and
5. compare generated versus dynamic descriptors.

The old Rust types are not compiled as a compatibility module. Expected
fixtures are neutral snapshots containing semantic fields.

#### Root tensor tests

- Flat presets emit `GeneratedRootChallenge::Flat`.
- Tensor presets emit the exact low factor selected for each root group.
- Replay never calls the tensor optimizer.
- Changing the exact low factor changes the descriptor and the certified fold
  bound.
- Every recursive and terminal fold follows the flat transcript path.

#### Setup-offload tests

- Prefix presence on a recursive consumer exactly determines predecessor
  offloading.
- A direct consumer may have an incoming prefix but its successor does not.
- Root and terminal cannot carry incoming prefixes.
- Prefix matrix dimensions, bases, geometry, natural length, and slot identity
  are descriptor-bound and independently validated.

#### Mixed-ring tests

- Uniform current-main schedules remain unchanged.
- Legal nested combinations, including `64/16/16`, `128/64/16`, and
  group-local mixed root dimensions, expand through canonical projection.
- Non-dividing dimensions, unsupported inner challenge dimensions, generator
  incompatibility, or key/role mismatch reject.
- Matrix storage reports use the matrix owner's dimension.

#### Serialization and no-panic tests

Fuzz or table-driven tests cover excessive group/fold/slice counts, zero
dimensions, overflows, unsorted slice ranges, incomplete slice coverage,
malformed prefix IDs, invalid tensor low factors, and inconsistent witness
length transitions.

### Performance

The topology/terminology cutover has a zero-regression target for selected
current-main parameters, proof bytes, setup bytes, and prover/verifier work.
Generated table source size may change because rows become more explicit; this
is acceptable if compile time and binary size remain reasonable and auditability
improves.

Subsequent planner capabilities are evaluated on their Pareto metrics rather
than against a blanket no-regression rule. Any selected production change must
ship an audit comparison showing proof bytes, witness components, exact matrix
storage and envelope, and estimated prover/verifier work.

## Alternatives Considered

### Keep a homogeneous fold vector and validate indices

Rejected. It preserves the exact source of the current ambiguity and makes
root-only capabilities representable at other levels.

### Keep tensor on `LevelParams` but reject non-root use

Rejected. The feature is not useful outside the root and should not burden
every recursive schedule, descriptor, test, or verifier branch forever.

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
  challenges are permanently root-only.
- Update `book/src/how/verifying/matrix_evaluation.md` to use descriptive matrix
  names with A/B/D notation in parentheses.
- Update `book/src/how/architecture.md` for the runtime parameter types.
- Update `AGENTS.md` only if the final verifier or planner maintainer contract
  needs a new concise pointer.
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
