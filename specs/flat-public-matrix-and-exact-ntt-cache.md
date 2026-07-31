# Spec: Flat Public Matrix and Exact NTT Caches

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-07-31 |
| Status        | implemented |
| PR            | #341, stacked on #338 |
| Supersedes    | The setup-generation-dimension and full-envelope NTT-cache contracts in `runtime-ring-cutover.md`, `mixed-ring-dimension-per-level.md`, and `setup-layout-repack.md`; the packed overlapping-prefix matrix layout itself remains authoritative |
| Superseded-by | |
| Book-chapter  | book/src/usage/commitment-api.md |

## Summary

Akita's public commitment matrices are mathematically one deterministic stream
of base-field elements. A schedule gives a finite prefix of that stream a ring
dimension and a matrix shape. The current implementation instead gives setup a
global generation ring dimension, derives the XOF in chunks of that dimension,
rounds setup capacity into those chunks, restricts every scheduled ring
dimension to a divisor of that setup dimension, and prepares full-capacity NTT
caches in both transform domains for every dimension and compute cluster that
is warmed. These are accidental couplings between protocol identity, storage,
schedule geometry, and backend optimization.

This spec makes the public matrix a dimension-free, prefix-compatible field
stream. A materialized setup is an exact cached prefix of that stream. Ring
dimensions exist only on schedule-owned matrix views and arithmetic operations.
Power-of-two zero padding exists only where a protocol object has a Boolean
evaluation domain, especially setup-prefix offloading; it does not enlarge the
random source prefix. NTT caches are derived backend state, independently sized
for each ring dimension, transform domain, operation cluster, and exactness
profile. Their sizes are compiled from the concrete operations that will use
them, not from the total materialized setup capacity.

This is an intentional protocol and cache-format cutover. Public-matrix
coefficients, transcript descriptors, serialized setup bytes, and disk cache
keys change. Akita has no backward-compatibility requirement, so this stacked follow-up must
perform the full cutover and invalidate old setup caches rather than preserve a
generation-D compatibility path.

## Intent

### Goal

Give setup one durable meaning: a deterministic public field stream plus an
exact materialized prefix. Derive all ring-shaped matrix views, setup-offload
objects, and backend transform caches from that object under explicit,
schedule-owned geometry.

### Normative language

The words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative. A
statement about current behavior is descriptive only when it appears under
"Current implementation."

### Conceptual model

The system has four layers. They are deliberately separate.

```text
semantic public parameters
    PublicMatrixId = derivation algorithm + public seed
    S_F,id : nonnegative field index -> F
                         |
                         | exact prefix [0, capacity_fields)
                         v
operational setup materialization
    PublicMatrixPrefix<F> = id + Vec<F>
                         |
                         | schedule-owned reshape of an exact prefix
                         v
protocol matrix use
    MatrixView = (ring dimension, rows, columns)
    fields used = ring dimension * rows * columns
                         |
                         | backend-local derived representations
                         v
execution caches
    NTT prefixes by (cluster, ring dimension, transform domain, profile)
```

Only the first and third layers affect proof meaning. Materialization length
and cache contents are local capacity and performance choices.

### Durable objects and ownership

The exact Rust spelling may follow crate conventions, but the final object
model MUST preserve these responsibilities. It MUST NOT collapse them back into
one setup seed or one carrier dimension.

#### Public matrix identity

`PublicMatrixId` identifies the infinite deterministic stream. Its semantic
fields are:

```rust
pub struct PublicMatrixId {
    pub derivation: PublicMatrixDerivation,
    pub seed: [u8; 32],
}

pub enum PublicMatrixDerivation {
    Shake256PagedV1,
}
```

The enum is protocol identity, not a compatibility shim. It prevents an
implementation detail such as XOF page size or field sampling from changing
silently. The base field is already bound by the algebra section and Rust type,
so it need not be duplicated in this value.

`PublicMatrixId` MUST NOT contain:

- a ring dimension;
- a matrix row count or width;
- `max_num_vars` or a batch limit;
- a materialized prefix length;
- a config type, preset name, or schedule digest.

#### Materialized public matrix

The expanded coefficient object is logically:

```rust
pub struct PublicMatrixPrefix<F> {
    pub id: PublicMatrixId,
    coefficients: Vec<F>,
}
```

Its capacity is exactly `coefficients.len()` base-field elements. It has no
generation dimension and no ring-element length. It MAY expose a checked method
equivalent to:

```rust
fn matrix_view<const D: usize>(
    &self,
    rows: usize,
    columns: usize,
) -> Result<RingMatrixView<'_, F, D>, AkitaError>;
```

The method computes `required = rows * columns * D` with checked arithmetic,
requires `required <= coefficients.len()`, and reshapes
`coefficients[..required]`. The entire stored vector need not be divisible by
`D`. There MUST NOT be a whole-envelope `total_ring_elements_at(D)` contract.

#### Provisioning limits

`max_num_vars` and `max_num_batched_polys` are host admission and provisioning
limits. They determine which catalog schedules and setup-prefix slots a setup
package promises to cover. They MAY remain metadata on the prover/verifier
setup package, but they MUST NOT be part of `PublicMatrixId`, public-matrix
derivation, or transcript identity.

The canonical physical capacity unit is a base-field element count. The setup
API and config API MUST expose that unit directly. A ring-element
`SetupMatrixEnvelope { max_setup_len }` is not part of the target model.

#### Derived setup artifacts

Setup-prefix commitments are preprocessing artifacts derived from a public
matrix ID, a natural source prefix, and commitment parameters. They remain in
prover and verifier registries. They do not change the identity or storage
shape of the base public matrix.

Backend-prepared NTT data is not part of `AkitaProverSetup`,
`AkitaVerifierSetup`, equality, serialization, or transcript identity.

### Public-matrix derivation

#### Flat field stream

For a base field `F` and `PublicMatrixId id`, setup defines an infinite stream

```text
S_F,id[0], S_F,id[1], S_F,id[2], ...
```

The value at field index `i` MUST be independent of:

- any ring dimension used to view it;
- the requested materialization length;
- schedule level, matrix role, rows, or columns;
- config or planner policy;
- thread count and parallel execution order.

All A, B, and D matrices, including matrices used to commit a setup-prefix
object, are overlapping prefix views of this same stream. They are not assigned
disjoint role labels.

#### `Shake256PagedV1`

The first derivation algorithm is a fixed-page SHAKE256 construction. A page is
a derivation unit for deterministic parallelism, not a ring.

```text
PUBLIC_MATRIX_DOMAIN = "akita/commitment/public-field-stream"
DERIVATION_TAG       = "shake256-paged-v1"
PAGE_FIELD_ELEMENTS  = 4096

page(i)   = floor(i / PAGE_FIELD_ELEMENTS)
offset(i) = i mod PAGE_FIELD_ELEMENTS
```

`4096` is a fixed v1 policy constant, not a number derived from ring geometry.
A temporary local microbenchmark compared 512, 1024, 2048, 4096, 8192, and
16384 field elements per page over `2^23` outputs. On the measured machine,
4096 was the best fp128 candidate and remained near the fp32/fp64 throughput
plateau. Those measurements are machine-specific and are not expected to
establish a portable optimum. A fixed page size is required for deterministic
parallel derivation, and 4096 is a reasonable granularity, so v1 simply chooses
it and does not retain a permanent benchmark target. Changing the constant
requires another derivation variant and new golden vectors.

For each page index, initialize SHAKE256 and absorb the following values with
the repository's canonical length-prefixed encoding:

```text
("domain",     PUBLIC_MATRIX_DOMAIN)
("derivation", DERIVATION_TAG)
("page_field_elements", PAGE_FIELD_ELEMENTS as u64 little-endian)
("seed",       id.seed)
("field",      canonical 32-byte big-endian modulus of F)
("page",       page index as u64 little-endian)
```

Read field elements from the page XOF by calling `F::random`, whose
`RandomSampling` contract is strengthened by this cutover to mean the following
canonical exact-uniform sampler. Let
`b = ceil(log2(p))`, where `p` is the base-field modulus, and let
`n = ceil(b / 8)`. Read `n` bytes, interpret them as a little-endian integer,
clear the unused high bits above bit `b - 1`, and accept the integer exactly
when it is smaller than `p`; otherwise read another `n` bytes and retry. Map an
accepted integer to its canonical field representative. Repeat sequentially to
obtain exactly `PAGE_FIELD_ELEMENTS` elements. Then `S_F,id[i]` is the element
at `offset(i)` in that page. The final requested page MAY stop after the last
needed element.

This rejection sampler is both the protocol sampling rule for
`Shake256PagedV1` and the repository-wide meaning of `RandomSampling`.
`Fp32`, `Fp64`, and `Fp128` MUST share one canonical helper or equivalent
cross-checked implementations with identical byte-consumption semantics.
Extension-field implementations remain coefficient-wise calls to the uniform
base-field sampler. A future change to this algorithm requires a new public
matrix derivation variant even if the resulting distribution remains uniform.

The page constant is deliberately independent of every supported ring
dimension. Parallel derivation assigns pages to workers; the output is
concatenated in page-index order. A future change to the absorbed fields, page
size, XOF, or field sampling MUST introduce a new `PublicMatrixDerivation`
variant and descriptor version.

#### Prefix laws

For one `(F, id)`, derivation MUST satisfy:

```text
derive(n) == derive(m)[0..n]                for 0 <= n <= m
view(D1, r1, c1).flat == S[0..D1*r1*c1]
view(D2, r2, c2).flat == S[0..D2*r2*c2]
```

Changing only `D1` to `D2` changes grouping and multiplication semantics, but
it does not change any field coefficient at a shared flat index.

### Matrix capacity

#### One concrete schedule

For each concrete matrix use `u`, define:

```text
fields(u) = rows(u) * columns(u) * ring_dimension(u)
```

The public-matrix capacity required by a schedule is the maximum, not the sum:

```text
schedule_matrix_capacity = max(1, max_u fields(u))
```

The scan includes:

- root final-group A, B, and D matrices;
- every root precommitted group's A and B matrices;
- every recursive level's A, B, and D matrices;
- the terminal A matrix;
- A and B matrices used to commit every required setup-prefix slot.

All these matrices overlap at flat index zero. No role, group, level, chunk, or
ring dimension receives a disjoint allocation.

#### A provisioned setup package

A config-backed setup package may promise all schedules admitted by
`max_num_vars` and `max_num_batched_polys`. Its matrix capacity is the maximum
of `schedule_matrix_capacity` across that admitted schedule set and all
required preprocessing slot commitments.

The scan MUST operate on final, validated schedules. Planner estimates,
generated schedule rows, setup construction, and runtime validation MUST call
the same canonical field-count primitives. No path may convert the result
through a setup generation dimension.

The stored coefficient count SHOULD equal the exact computed maximum. A host
MAY deliberately materialize a larger prefix for future reuse, but that is an
explicit cache policy and MUST NOT change public matrix identity, proof bytes,
or schedule admissibility.

### Setup-prefix offloading

Setup-prefix offloading has two lengths with different meanings:

```text
natural_len = exact number of public stream coefficients used by Stage 3
n_prefix    = next_power_of_two(max(1, natural_len))
```

The committed witness is:

```text
S[0..natural_len] || zero^(n_prefix - natural_len)
```

Therefore:

- base public-matrix capacity MUST cover `natural_len`, not `n_prefix`;
- zero padding MUST be constructed explicitly and MUST NOT be read from later
  random coefficients of `S`;
- the Boolean setup-index domain and setup-prefix commitment layout continue
  to use `n_prefix`;
- A/B matrices that commit the padded object are ordinary matrix uses and their
  exact `rows * columns * D` footprints remain in capacity accounting;
- the setup-prefix group's inner A-matrix ring dimension determines how the
  padded witness is chunked for commitment. It is a planner-owned commitment
  parameter, not a public-matrix generation dimension;
- the setup-prefix group's outer B-matrix dimension is selected independently,
  exactly like every other committed group's B matrix;
- neither `natural_len` nor the whole materialized setup must be divisible by
  the prefix A dimension. Only the padded witness shape consumed by that
  commitment must be divisible by it.

A setup-prefix slot's semantic identity MUST bind the `PublicMatrixId`,
`natural_len`, padded-domain/commitment geometry, and commitment parameters.
The registry may avoid storing the ID redundantly when its enclosing setup
package already supplies and validates it.

This cutover does not reduce the setup-prefix polynomial from `n_prefix` to
`natural_len`. In a recursive suffix opening, the setup prefix remains its own
singleton polynomial group with `log2(n_prefix)` variables, while the recursive
witness remains a separate group with its own variable count. The multi-group
opening machinery MUST preserve those per-group arities; it MUST NOT physically
extend the setup group to the witness group's larger domain merely to batch
them. Therefore no new common-suffix padding is introduced, but the existing
zero padding inside the setup-prefix group still participates in its commitment,
folding, and opening work.

The immediate saving is narrower: deriving and storing the shared random matrix
requires only `natural_len` source coefficients, not `n_prefix`. This reduces
the final setup capacity only when that source prefix is the maximum matrix
footprint; setup-prefix A/B commitment matrices or another protocol matrix may
still dominate the maximum. Exact NTT-cache sizing can remove additional
derived-state waste independently. Eliminating the remaining Boolean-domain
padding would require a separate non-power-of-two-domain protocol change.

The current restriction that recursive setup planning requires the base setup
to have been generated at D64 MUST be removed. Preprocessing a setup-prefix
commitment with planner-selected A/B dimensions over a flat source is valid
even when the same setup later supplies different dimensions to other protocol
matrices.

The prefix A dimension is not the producer's Stage 3 projection dimension and
is not required to equal the consumer witness group's A dimension. Stage 3
produces one flat field vector. The prefix commitment profile independently
chooses how that vector is chunked and committed; the subsequent fold shares
only the opening/ring-switch dimension required by its relation. A slot ID MUST
derive the prefix A dimension from its committed-group profile rather than
store a second `d_setup` field that can disagree with the profile.

### Transcript and protocol identity

The instance descriptor MUST separate algebra, public matrix identity, plan,
and call shape:

```text
AlgebraSection
    base field modulus
    message/protocol extension degrees

SetupSection
    digest(PublicMatrixId)
    decomposition, SIS, feature, and fold-bound policy

PlanSection
    digest(final effective schedule)
    -- includes every matrix's scheduled ring dimension and shape

CallSection
    claims and per-call public shape
```

`AlgebraSection` MUST NOT contain one global `ring_dimension_d`. There is no
single ring dimension for a mixed schedule. The effective schedule digest is
the sole owner of the A/B/D dimensions used by the proof.

The setup-bound digest MUST bind `PublicMatrixId`, not the serialized
materialization package. In particular it MUST NOT bind:

- materialized field-element capacity;
- `max_num_vars` or `max_num_batched_polys`;
- a setup cache path or config type;
- backend NTT cache state.

Consequently, the same proof is valid under any setup materialization that has
the same public matrix ID and covers every prefix required by its schedule.
Prover and verifier may use different covering capacities. They still derive
identical matrix coefficients and transcript bytes.

This cutover MUST bump `AKITA_INSTANCE_DESCRIPTOR_VERSION`. Old proofs and old
serialized setup caches are intentionally invalid.

### Schedule validation

Schedule validation MUST validate every distinct A/B/D ring dimension directly
against the field's protocol dispatch and CRT/NTT support. It MUST NOT validate
a schedule by requiring all dimensions to divide a global carrier dimension.

For each matrix, validation MUST check its own exact field footprint against
the materialized public prefix. A wider stored prefix can cover a smaller
request even when the wider prefix length is not divisible by the request's
ring dimension.

The planner's dimension search domain is an explicit set of supported matrix
dimensions. `setup_generation_dimension` and equivalent policy fields MUST be
removed. A maximum setup budget, when configured, is measured in base-field
elements or bytes.

Planner cost metrics remain separate because they answer different questions:

```text
setup fields       = reusable maximum public-matrix prefix
matrix work fields = additive matrix coefficients scanned across proof levels
NTT cache bytes    = execution-profile-specific resident derived state
proof bytes        = serialized proof payload
```

The planner MUST NOT use a ring-element setup proxy for any of them. A balanced
objective may compare or weight these metrics, but it must name an execution
profile before using NTT bytes because cluster aliasing and backend
representation affect that value. Generated schedule records SHOULD preserve
the protocol metrics needed to replay selection; benchmark reports may add
backend-specific NTT estimates without making them schedule identity.

### Exact NTT cache model

#### NTT caches are derived views

An NTT cache contains transforms of a public-matrix prefix. It is reproducible
from `PublicMatrixPrefix<F>` and does not belong to setup identity or setup
serialization.

The protocol uses at least these transform domains:

- **negacyclic** for A/B commitment products and the negacyclic half of A-side
  quotient work;
- **cyclic** for ring-switch D and B relations and the cyclic half of A
  quotient work;
- **exact negacyclic with optional i16 tail** for the terminal verifier;
- **compression diagnostic negacyclic** under the diagnostic feature, using
  its separate supported-dimension and profile policy.

These are not interchangeable cache modes. A request for one domain MUST NOT
force materialization of another domain.

#### Requirement unit

The canonical requirement unit is an exact prefix of ring elements at one
ring dimension:

```rust
pub struct NttPrefixRequirement {
    pub ring_dimension: usize,
    pub num_ring_elements: usize,
}
```

Transform domain, CRT profile, and operation cluster are keys around that
value, not booleans that force every vector in one cache object to have equal
length. Conceptually, one execution requirement set is:

```text
(cluster, D, profile) -> {
    negacyclic_prefix_rings,
    cyclic_prefix_rings,
    optional_i16_tail_prefix_rings,
}
```

The target implementation MAY use separate slot types or one per-D slot with
independent vectors. It MUST preserve independent prefix lengths.

The associated field footprint is always
`num_ring_elements * ring_dimension`. Every requested prefix is exactly
divisible by its own dimension; the containing public matrix need not be.

#### Requirement derivation

Every backend operation that consumes a public matrix MUST declare its exact
matrix prefix and transform domain from the same rows and active width passed
to the kernel. The implementation MUST have one canonical derivation path used
by prewarming, lazy cache checks, memory reporting, and tests. A prewarm planner
and the actual kernel call MUST NOT compute independent approximations.

The implementation realizes that contract as follows:

- `NttCacheKey::from_matrix_shape` is the sole rows/active-width derivation
  primitive used by both the execution compiler and lazy kernel acquisition;
- `NttExecutionRequirements::from_prove_schedule` compiles only work inside a
  `batched_prove` call. Prior root commitments are source- and backend-shaped
  separate API calls; `add_setup_prefix_commitment` likewise adds the
  independently invoked setup-prefix preprocessing call layout;
- `prewarm_ntt_requirements` routes each level through its selected
  `ProverComputeStack` cluster before transcript binding; and
- `planned_ntt_cache_metrics` routes the same requirements, partitions them by
  process-local physical `NttCacheOwnerId`, and max-joins keys per owner before
  asking the selected backend for bytes. This preserves real cache aliasing
  when several levels or clusters share one prepared setup. Integration and
  stack tests cover one fully shared owner, two partially shared owners, and
  four independent owners, and require planned and resident bytes to agree.

The execution compiler first max-joins one `(level, cluster, D, domain)` route.
After routing, requirements that land on the same physical owner combine again
by `(owner, D, domain)` maximum:

```text
join(prefix_1, prefix_2, ...) = max(prefix_1, prefix_2, ...)
```

They do not sum because every matrix is an overlapping prefix of the same
stream. Different dimensions, domains, or physical owners remain separate.
Levels and clusters remain separate only when routing assigns them different
prepared owners; aliases of one prepared setup share one cache entry.

A requirement set is scoped to one API phase. Planned-versus-resident equality
therefore uses an initially empty owner or measures that phase's resident-state
delta. A cross-phase total requires a source- and backend-aware union; a
schedule alone cannot derive it because root source kernels differ in whether
their A path uses an NTT. `from_prove_schedule` MUST NOT charge a root
commitment that completed before the proof call.

The cache MUST support covering-prefix lookup: a slot of length `m` covers any
request `n <= m` with the same keys. Growth SHOULD transform only the missing
suffix. Replacement by a larger immutable snapshot is acceptable; an existing
borrower may finish on the older covering snapshot. Concurrent first use or
growth MUST be single-flight per cache key.
Failed initialization is not resident cache state. A failed cell MUST be
removed only when it is still the map entry for that key; callers that waited
on a failed covering cell retry their own exact request. Thus an oversized or
otherwise rejected warm cannot poison a later valid smaller prefix.

#### Operation clusters

`ProverComputeStack` permits commit, opening, tensor, and ring-switch clusters
to use different backends and prepared setups. NTT requirements MUST therefore
be routed to the cluster that actually executes the operation.

The implementation MUST NOT warm every role dimension on all four clusters.
A cluster with no NTT consumer has no NTT requirement. A uniform stack may
deduplicate identical requests naturally because its contexts share one
prepared setup; a heterogeneous stack must not pay for another cluster's
cache.

The cluster label is a routing coordinate, not necessarily a physical cache
identity. After routing, requirements whose contexts refer to the same prepared
cache owner are joined again by `(D, profile, domain)`. Requirements whose
contexts refer to different prepared objects remain separate even when their
backend types are equal. Memory reporting MUST account for this runtime
aliasing, so a uniform stack is counted once and four independent prepared
objects are counted four times.

At minimum, the compiler must account for:

- commit-cluster negacyclic prefixes for root, recursive, terminal, and
  setup-prefix A/B commitment products;
- ring-switch-cluster cyclic D/B prefixes and cyclic plus negacyclic A
  prefixes used by relation quotient construction;
- any opening-cluster digit-matrix operation that actually reads the public
  matrix;
- terminal verifier exact-negacyclic A prefixes and their independently sized
  i16 tail when the selected CRT profile requires it.

Tensor-only work does not acquire an NTT cache merely because it shares a
level with an NTT-consuming operation.

#### Base and tail profiles

Ordinary prover i8 and centered quotient kernels use the field/D-selected base
CRT profile. Their negacyclic and cyclic prefixes can have different lengths
while sharing immutable CRT parameters.

The terminal verifier remains an exact-negacyclic consumer. Its base prefix is
the exact terminal A footprint. Its i16 tail prefix exists only when
`ntt_cache_requires_i16_tail(width, log_basis)` requires it and is independently
covering. The current verifier cache already approximates this target behavior
and should be generalized rather than replaced by a full-envelope cache.

Compression diagnostics remain a separate cache namespace. They MUST NOT
widen the ordinary protocol's ring-dimension support, profile, setup capacity,
or eager cache contract.

#### Memory formula

Let `K(F, D)` be the number of i32 CRT primes in the selected base profile. For
one backend-owned `(cluster, D)` cache with `N` negacyclic ring entries, `C`
cyclic ring entries, and `T` i16-tail ring entries, transformed matrix storage
is:

```text
base bytes = (N + C) * D * K(F, D) * sizeof(i32)
tail bytes = T * D * sizeof(i16)
total      = base bytes + tail bytes
```

Metadata and allocator overhead are reported separately if material.

This formula is also the canonical estimator used by profiling. It makes the
current waste visible: preparing both domains over an `M`-field full envelope
costs `2 * M * K * sizeof(i32)` for each warmed dimension, even when that
dimension uses only a much smaller matrix prefix. Repeating the warm-up on
separate operation clusters repeats the allocation.

### Current implementation

The following behavior is the pre-cutover state that this follow-up must remove.

#### Derivation and setup shape

`AkitaSetupSeed` stores `gen_ring_dim` and `max_setup_len` in ring elements.
`derive_public_matrix_flat::<F, D>` starts one indexed XOF reader per ring
element and samples `D` coefficients from it. Thus the same seed at D64 and
D128 gives different coefficients after flat index 63. The function's flat
name does not make its output dimension-independent.

Current `Fp128::random` already uses exact rejection sampling. Current
`Fp32::random` reduces a 64-bit word modulo `p`, and `Fp64::random` reduces a
128-bit word modulo `p`; those distributions have negligible but nonzero bias.
The cutover fixes the trait semantics for all base fields instead of adding a
setup-only sampler with a competing meaning of field randomness.

`FlatMatrix` duplicates `gen_ring_dim`, requires its whole vector length to be
divisible by that dimension, and permits another D only when it divides the
generation D. `setup_matrix_envelope_for_schedule` takes the already correct
field count and converts it to
`ceil(field_count / gen_ring_dim)` ring elements. The materialized vector is
therefore rounded up to a generation-D boundary.

The seed digest in `AkitaInstanceDescriptor` binds the generation dimension,
capacity, and host nv/batch limits. `AlgebraSection` separately carries one
ring dimension even though the effective mixed schedule carries several.

#### Setup-prefix padding

The setup-prefix witness correctly copies `natural_len` coefficients and fills
the rest of `n_prefix` with zero. Capacity accounting nevertheless includes
`n_prefix` as if those zeros had to be random public-matrix coefficients. That
charge is removed. The commitment A/B matrix footprints remain.

Recursive prefix preprocessing also rejects any base setup whose generation D
is not D64. That restriction is an artifact of the carrier model, not a
protocol requirement.

#### Prover NTT caches

`NttCacheKey::from_envelope` reinterprets the entire setup vector at D.
`prepare_setup` eagerly builds one full-envelope slot at `gen_ring_dim`.
Subsequent mixed dimensions lazily build another full-envelope slot each.
Every ordinary slot uses `BothTransforms`, so cyclic and negacyclic arrays have
the same full length.

Before proving, every distinct role D at a level is warmed on commit, opening,
tensor, and ring-switch contexts. This is harmless when every context shares
one prepared CPU object, but wasteful for heterogeneous stacks and obscures
which operation owns each transform.

There is no general next-power-of-two rounding inside ordinary NTT preparation.
The large waste is full-setup coverage per D and equal-length dual domains.
Power-of-two lengths elsewhere belong to Boolean sumcheck, witness, or
setup-prefix commitment domains and must be reviewed individually rather than
removed indiscriminately.

### Feature interactions

#### Mixed ring dimensions

Each dimension gets its own matrix reshape and transform cache. There is no max
carrier, divisibility lattice, or setup-generation D. A schedule using A512,
B64, and D64 reads three role-specific prefixes from the same flat stream. The
D512 cache covers only D512 operations; the D64 cache covers the maximum of the
actual D64 consumers in each domain and cluster.

#### Multiple commitment groups

Root final and precommitted groups may have distinct A/B dimensions and
widths. Base matrix capacity is the maximum physical field footprint across all
their matrices. Cache requirements at an equal `(cluster, D, domain, profile)`
join by maximum. Group count never causes disjoint copies of the public matrix
or an additive NTT prefix.

#### Multiple witness chunks

Chunking changes final witness geometry and can therefore change scheduled
active widths. Requirements are compiled from those final widths. Once a
matrix shape is known, chunks reuse its public coefficients and transforms;
cache size is not multiplied by `num_chunks`.

One-hot polynomial storage also has single-chunk and multi-chunk kernel
representations. That storage choice changes how the input is traversed, not
the public A matrix. Both forms use the same scheduled active A width and the
same exact transform prefix. It must not create separate setup or NTT entries.

#### Extension openings and tensor work

Extension-opening reduction and tensor projection can change the schedule and
the base-field witness presented to later commitment operations. Those final
matrix shapes are counted normally. Tensor work that never reads the public
matrix creates no setup transform requirement, regardless of extension degree.
The public matrix itself is always over the base field.

#### Recursive folds and terminal verification

Every recursive level contributes its concrete A/B/D requirements. Equal keys
join by maximum across levels. The terminal contributes an A commitment
requirement to the prover commit cluster and an exact-negacyclic A requirement
to the verifier cache. These caches need not have the same profile or lifetime.

#### Setup offloading

The natural public prefix, its padded committed witness, and the matrices that
commit that witness are three separate shapes. Setup capacity and cache
compilation use the correct shape for each. Preprocessing caches may be dropped
after all required prefix slots are materialized; they must not silently become
the runtime prove-cache contract.

Direct Stage 3 evaluation views exactly the `required` natural ring rows. It
must not reinterpret the remainder of the materialized setup merely to obtain a
global `setup_eval_len`. An offloaded Stage 3 may separately use the padded
commitment domain because that domain is bound by its setup-prefix slot.

#### Disk persistence

Disk persistence stores a public matrix prefix and setup-prefix registries. It
does not store backend NTT caches.

The base-prefix cache lineage key is:

```text
(field modulus, PublicMatrixId)
```

Each artifact under that lineage records its exact materialized field count. A
validated artifact with a larger count may satisfy a smaller request because
the derivation is prefix-compatible. The derivation version is already inside
`PublicMatrixId`; config type names, schedule ring dimensions, and generation D
are not semantic cache keys. Provisioning limits and a schedule/catalog digest
MAY key or validate the derived setup-prefix registry because that registry
promises a particular set of precomputed slots.

The base-prefix artifact serializes only `PublicMatrixId`, the materialized
field count, and the flat coefficients. It does not serialize host admission
limits. A load reconstructs `max_num_vars` and `max_num_batched_polys` from the
current request while retaining any larger covering coefficient prefix. Thus a
covering cache cannot silently widen the setup package's admission contract.
Concurrent writers take the lineage lock, validate the resident artifact, and
replace it only when the candidate is a strict prefix extension. The resident
base prefix therefore forms a monotone max-join and cannot shrink under racing
small and large requests.

Loading MUST validate all lengths before allocation and verify that serialized
coefficients equal the deterministic flat stream. Validation SHOULD stream by
page or bounded chunk rather than allocate a second full expected matrix.
Writing SHOULD use an atomic temporary-file replacement. The cache namespace
MUST be bumped; old generation-D cache files are rejected and regenerated.

#### Parallel and no-default-feature builds

Parallelism changes only page scheduling and transform construction. It cannot
change coefficients, requirement joins, or cache keys. Builds without
`parallel` or `disk-persistence` use the same protocol identities and layouts.

#### Setup provisioning versus one execution

Two derived plans are intentionally different:

1. A **provisioning requirement** joins matrix capacity and required
   setup-prefix slot IDs across every schedule promised by host limits.
2. An **execution requirement** describes exact backend caches for one resolved
   schedule, call layout, and compute-stack routing.

Provisioning MUST NOT eagerly fill every NTT representation that any admitted
schedule might someday use. Execution prewarming MUST NOT rescan the entire
admitted schedule universe.

### Invariants

- Public coefficients are determined only by `(field, PublicMatrixId, flat
  index)`.
- A larger materialization is a strict prefix extension of a smaller one.
- Matrix capacity, planner setup cost, and runtime coverage use base-field
  elements as their canonical unit.
- A matrix view validates only its own exact prefix; whole-vector divisibility
  is irrelevant.
- A/B/D role matrices overlap at flat index zero across roles, groups, levels,
  chunks, and dimensions.
- Setup-prefix zero padding is protocol data, not public randomness.
- The effective schedule is the sole owner of protocol ring dimensions.
- Materialization capacity and backend cache state do not change proof or
  transcript bytes.
- NTT cache domains, profiles, dimensions, and operation clusters are explicit.
- Equal-key cache requirements join by maximum, never by sum.
- Prover and verifier derive identical coefficients and reject an undersized
  materialization before indexing or allocation.
- Verifier-reachable malformed shapes return `AkitaError` or
  `SerializationError`, never panic.

### Non-goals

- This spec does not change the packed overlapping-prefix A/B/D layout.
- It does not eliminate legitimate power-of-two Boolean domains.
- It does not serialize NTT caches or require a global eager prewarm.
- It does not change CRT prime selection or SIS security pricing, except that
  both consume schedule-local dimensions without a setup carrier.
- It does not require GPU and CPU caches to share an in-memory representation.
  They share requirement semantics and accounting units.
- It does not preserve old setup/proof/cache bytes.
- It does not by itself redesign `OneHotPoly` or every preset type. It does
  require removal of every use of a OneHot/config D as setup envelope identity.

## Evaluation

### Acceptance criteria

This follow-up is not complete until all merge-blocking criteria below are satisfied.

#### Public matrix and setup API

- [x] `AkitaSetupSeed`, expanded setup storage, and public setup APIs contain no
  `gen_ring_dim` or ring-element capacity.
- [x] `FlatMatrix` is replaced or cut over so a stored public prefix has no
  generation D and can create any individually supported exact prefix view.
- [x] The canonical setup capacity type and all planner/generated estimates are
  measured in base-field elements.
- [x] `setup_matrix_envelope_for_schedule`, setup generation dimension policy,
  and dimension-divisibility validation are removed, with all call sites cut
  over in one pass.
- [x] The flat paged XOF derivation is implemented exactly as specified and old
  generation-D derivation is not retained behind an alias or compatibility
  branch.
- [x] `RandomSampling` specifies exact canonical rejection sampling, and fp32,
  fp64, and fp128 implementations pass shared distribution/byte-consumption
  vectors.
- [x] A temporary page-size microbenchmark covered
  512/1024/2048/4096/8192/16384; `Shake256PagedV1` fixes 4096 as a policy
  constant without retaining benchmark-only code.

#### Protocol identity

- [x] The instance descriptor version is bumped.
- [x] The algebra section no longer stores a single ring dimension.
- [x] The setup section binds `PublicMatrixId`, not materialization length or
  host provisioning limits.
- [x] Schedule digest tests demonstrate that changing any role-local D changes
  the plan digest.
- [x] A proof generated with one materialized capacity verifies with a larger
  covering capacity having the same public matrix ID.

#### Setup-prefix offloading

- [x] Capacity accounting charges `natural_len`, not `n_prefix`, for the source
  prefix and separately includes setup-prefix commitment matrices.
- [x] Tests cover a non-power-of-two natural length and prove all padded entries
  are zero rather than later public-stream coefficients.
- [x] Setup-prefix preprocessing works for independently selected prefix A/B
  dimensions while the same setup serves a differently dimensioned mixed
  schedule.
- [x] Validation neither equates the prefix A dimension with the producer
  Stage 3 projection dimension nor with the consumer witness A dimension.
- [x] Slot identity derives the prefix A dimension from the committed-group
  profile without a duplicate `d_setup` field.
- [x] Setup-prefix slot identity and registry validation are bound to the flat
  public matrix identity and exact natural/padded geometry.

#### NTT caches

- [x] Prover NTT requests use exact matrix prefixes; no ordinary path calls a
  full-envelope key constructor.
- [x] Negacyclic, cyclic, and optional i16-tail prefixes can have independent
  lengths.
- [x] Prewarming routes requirements only to clusters that consume them.
- [x] Lazy lookup accepts a covering prefix and rejects an undersized or wrong-D
  slot without panic.
- [x] Equal requirements from levels/groups/chunks join by maximum and tests
  demonstrate that they do not sum or multiply.
- [x] The terminal verifier retains exact-negacyclic and exact-tail behavior.
- [x] Compression diagnostics remain in a separate cache/profile namespace.
- [x] Cache byte reporting uses the normative formula and agrees with actual
  allocated transform vectors.

#### Persistence and integration

- [x] Disk cache format/key namespace is bumped and generation-D caches are
  rejected.
- [x] A larger validated public prefix can cover a smaller provisioning request
  without changing transcript identity.
- [x] `parallel`, no-default-feature, and `disk-persistence` feature graphs use
  identical public stream coefficients.
- [x] The fp128 one-hot nv32 mixed-dimension CI bench runs through production
  schedule resolution, prewarms the exact shared-owner commit-and-prove union
  in its preparation phase, rejects online cache growth, and reports exact
  setup fields plus per-D/domain/cluster NTT bytes.
- [x] No setup code uses a config/preset D or `D512OneHot` as public-matrix
  identity or allocation unit.

#### Documentation and quality

- [x] The setup and caching section of the Akita Book is updated from this spec
  once the cutover is implemented.
- [x] Scoped revision notes in older specs point here for setup derivation and
  NTT-cache semantics while preserving their unrelated decisions.
- [x] No thin wrapper preserves the removed generation-D API.
- [x] `./scripts/check-doc-guardrails.sh` and every repository preflight command
  required by touched paths pass.

The checklist above records the flat setup and cache cutover. The broader
testing matrix below is a durable target for all field widths and protocol
compositions, not a claim that PR341 ships every combination. In this PR the
mixed-dimension catalog coverage is the scalar fp128 nv32 one-hot family;
mixed multi-group replay, mixed multi-chunk planning, and fp32/fp64 mixed
catalogs remain deferred to the mixed-planner follow-up.

### Testing strategy

#### Derivation tests

Use at least fp32, fp64, and fp128 fields. For each field:

- compare serial and parallel derivation byte-for-byte;
- derive lengths before, at, and after a page boundary;
- prove prefix extension for several capacities;
- view overlapping prefixes at D32, D64, D128, D256, and D512 where supported;
- verify that flat coefficients are unchanged across those views;
- mutate seed, derivation variant, field modulus, and page index independently
  and confirm domain separation;
- round-trip and validate serialized materializations without a generation D.

Golden vectors MUST cover page zero, the last element of page zero, and the
first element of page one. They make accidental sampling or encoding changes
review-visible.

#### Capacity tests

Construct schedules where the largest footprints come respectively from root
A, root B, root D, a precommitted group, a recursive level, terminal A, a
setup-prefix natural source, and a setup-prefix commitment matrix. Assert that
the canonical scan returns the exact maximum in fields.

Include a maximum whose field count is not divisible by another scheduled D.
That schedule must remain valid because each requested matrix prefix is itself
well-shaped.

#### NTT requirement tests

Build table-driven expected requirement maps for:

- uniform-D scalar setup;
- A512/B64/D64 mixed root;
- mixed dimensions across levels;
- multi-group root with unequal group dimensions and widths;
- two and four witness chunks;
- one and multiple setup-offloaded levels;
- terminal verifier with and without an i16 tail;
- uniform and heterogeneous compute stacks;
- compression diagnostics enabled and disabled.

For each map, assert exact keys, independent domain lengths, maximum joins, and
reported bytes. Kernel tests must additionally prove that every planned prefix
covers the actual indices read. A test-only recording backend SHOULD compare
declared requirements with observed matrix reads so future kernels cannot
silently bypass the compiler.

#### End-to-end tests

Run commit/prove/verify for the production fp128 nv32 one-hot mixed-dimension
schedule, including setup offloading when selected. Run the same proof with
different covering materialization capacities. Exercise multi-group and
multi-chunk layouts together because their interaction is the easiest place to
accidentally sum overlapping prefixes or duplicate caches.

Malformed verifier tests cover zero/overflowing dimensions, overflowing
row-width products, insufficient field prefixes, wrong public matrix IDs,
wrong setup-prefix geometry, and serialized length bombs.

### Performance

The cutover has three required measurements:

1. coefficient-form setup bytes;
2. peak and resident NTT bytes broken down by operation cluster, D, transform
   domain, and profile;
3. setup derivation, prefix preprocessing, commit, prove, and verify time.

The fp128 one-hot nv32 mixed-dimension CI bench is the primary regression
fixture. Its report MUST show both the provisioned setup requirement and the
concrete execution-cache requirement. A single aggregate `setup_ring_elements`
number is insufficient.

Expected direction:

- setup coefficient bytes do not increase and normally decrease by removing
  generation-D rounding and padded-source charging;
- NTT bytes decrease materially because each D transforms only its largest
  actual prefix and only in required domains/clusters;
- XOF setup generation should remain parallel and within a small constant
  factor of the current implementation; page-level parallelism must avoid one
  SHAKE initialization per field;
- proof size and security parameters do not change solely because of cache
  refactoring, although proof bytes intentionally change with the new public
  matrix and descriptor version.

Any regression in resident NTT bytes for the nv32 mixed fixture is a merge
blocker unless the report identifies a newly covered real operation and the PR
documents the tradeoff.

## Design

### Architecture

The canonical dataflow is:

```text
host limits + config/catalog
        |
        | enumerate promised final schedules
        v
compile provisioning requirements
        |-- exact max public-matrix fields
        `-- required setup-prefix slot identities
        |
        v
derive/load PublicMatrixPrefix<F>
        |
        | exact natural coefficients
        +---------------------> setup-prefix preprocessing
        |                           | explicit zero padding
        |                           `-> persistent slot registries
        |
resolved schedule + call + compute-stack routing
        |
        v
compile execution requirements
        |-- commit cluster: exact neg prefixes
        |-- ring-switch cluster: exact neg/cyc prefixes
        |-- other clusters: only declared consumers
        `-- verifier terminal: exact neg + optional tail
        |
        v
backend-local lazy/prewarmed cache slots
```

The schedule remains the source of matrix ranks, widths, and dimensions. The
setup requirement compiler is a pure consumer of those shapes. The backend is
a consumer of compiled exact prefix requests. Neither setup nor backend may
invent protocol geometry.

### Canonical joins

Provisioning joins and execution joins are small algebraic operations and
SHOULD be represented as such:

```text
matrix capacity: max(field prefixes)
slot registry:   set union(slot IDs)
NTT prefix:      pointwise max by full cache key
```

These joins are associative, commutative, and idempotent. Tests SHOULD assert
those laws. This gives multi-group, multi-chunk, and multi-level composition a
single answer without feature-specific conditionals.

### API boundaries

`akita-types` owns:

- public matrix identity and flat derivation;
- dimension-free prefix storage and checked matrix views;
- canonical setup-capacity and setup-prefix geometry primitives;
- cache requirement value types and schedule-shape joins that are backend
  independent;
- descriptor serialization and validation.

`akita-config` and `akita-planner` own:

- the admitted schedule universe for host limits;
- exact field-based capacity budgets and scoring;
- production generated schedule data;
- no setup generation dimension.

`akita-setup` owns:

- provisioning scans;
- loading/generating an exact public prefix;
- setup-prefix preprocessing and registry persistence;
- cache namespace/version and strict load validation.

`akita-prover` owns:

- routing execution requirements to concrete compute clusters;
- backend-local transform preparation and covering-prefix lookup;
- operation plans whose exact rows/widths agree with requirement derivation.

`akita-verifier` owns:

- exact direct setup scans;
- exact terminal NTT requests;
- no-panic validation before views and allocations.

### Alternatives considered

#### Keep a maximum generation D

Rejected. A max D is neither matrix identity nor useful physical storage
metadata. It changes XOF coefficients, imposes false divisibility constraints,
rounds capacity, and makes mixed schedules look subordinate to one carrier.

#### Generate at D1 or one field element per XOF

Generating one field per indexed XOF is dimension-independent but pays one
SHAKE initialization per coefficient. A fixed field page preserves parallelism
and prefix compatibility without smuggling ring geometry back into derivation.

#### One sequential XOF for the whole stream

This gives clean prefix semantics but prevents deterministic random-access page
derivation and practical parallel setup generation. Fixed indexed pages retain
both properties.

#### Round materialization to the largest supported D

Rejected as a semantic contract. A host may overallocate deliberately, but
rounding is not needed for any exact matrix view and cannot affect identity or
validation.

#### One NTT slot per D with both equal-length domains

Rejected. It is simpler locally but systematically doubles single-domain uses
and promotes the largest domain prefix to every representation. Independent
covering prefixes express the actual kernel contracts.

#### Build cache entries only lazily without a requirement compiler

Rejected as the sole design. Lazy construction is useful, but without a
declarative requirement set setup sizing, prewarming, memory estimates, and
kernel use drift apart. The same canonical derivation must support both lazy
checks and optional prewarm.

#### Precompute every cache admitted by setup limits

Rejected. Provisioning and execution have different lifetimes. A setup may
cover many schedules while one proof uses exactly one. NTT caches belong to
the latter.

## Documentation

During implementation, `book/src/usage/commitment-api.md` becomes the durable
narrative for public matrix identity, setup provisioning, setup-prefix
offloading, persistence, and cache lifetime. It should explain the four-layer
model without reproducing every cache key.

`book/src/how/architecture.md` should state that ring dimensions are
schedule-owned views and that public setup has no ring dimension. Detailed NTT
cache formulas and acceptance history stay in this spec until the cutover is
stable, then the durable parts are folded into the book and this spec is
archived according to `specs/PRUNING.md`.

The scoped revision authority of these live specs must point here:

- `specs/setup-layout-repack.md`: packed overlapping prefixes remain; flat
  field derivation and capacity semantics move here;
- `specs/runtime-ring-cutover.md`: runtime schedule ownership remains;
  generation-D setup and full-envelope NTT phase-1 contracts move here;
- `specs/mixed-ring-dimension-per-level.md`: planner/mixed-D evidence remains;
  setup carrier and cache semantics move here.

## Execution

This follow-up should implement the cutover in dependency order without compatibility
wrappers.

1. Add flat public matrix identity, paged derivation, golden vectors, and exact
   prefix storage in `akita-types`; strengthen `akita-field::RandomSampling`
   and cut all prime fields to the canonical rejection helper.
2. Cut checked matrix views and canonical field-capacity scans to the new
   storage. Remove whole-envelope ring reinterpretation.
3. Change descriptor identity and version, then update prover/verifier
   transcript construction together.
4. Remove setup generation D from config, planner policies, generated rows,
   estimates, examples, profile output, and schedule validation.
5. Change `akita-setup` provisioning, serialization, disk keys, and streamed
   validation. Invalidate old cache files.
6. Correct setup-prefix source capacity to natural length, remove the D64 base
   setup restriction, make prefix A/B dimensions planner-owned commitment
   parameters, remove duplicate `d_setup` identity, and keep explicit zero
   padding.
7. Introduce canonical execution requirements and independent-domain covering
   cache slots. Cut kernels and prewarming to exact requests.
8. Route requirements per compute cluster and preserve the exact terminal
   verifier/compression namespaces.
9. Add cross-feature end-to-end and memory-accounting tests, wire the nv32
   production mixed schedule into the CI bench, and update generated tables.
10. Fold the stable explanation into the book, update spec statuses, and run
    the complete repository preflight.

The implementation should delete obsolete generation-D helpers and tests as
their call sites are migrated. It must not retain `_flat_v2`, `_for_dimension`,
or pass-through aliases that recreate the old API.

## References

- `specs/setup-layout-repack.md`
- `specs/runtime-ring-cutover.md`
- `specs/mixed-ring-dimension-per-level.md`
- `specs/setup-offloading-planner.md`
- `crates/akita-types/src/proof/setup.rs`
- `crates/akita-types/src/proof/setup_envelope.rs`
- `crates/akita-types/src/proof/setup_prefix.rs`
- `crates/akita-types/src/layout/flat_matrix.rs`
- `crates/akita-types/src/ntt_cache.rs`
- `crates/akita-prover/src/compute/backend.rs`
- `crates/akita-prover/src/compute/cpu.rs`
- `crates/akita-prover/src/compute/stack.rs`
- `crates/akita-prover/src/api/setup_prefix.rs`
- `crates/akita-verifier/src/protocol/core/terminal_ntt.rs`
- `crates/akita-setup/src/lib.rs`
- `crates/akita-setup/src/recursive_prefixes.rs`
- `book/src/usage/commitment-api.md`
