# Spec: Heterogeneous group source contracts

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-07-30 |
| Status        | active |
| PR            | |
| Supersedes    | Root-source and commitment-handle decisions in [`multi-group-batching.md`](multi-group-batching.md) |
| Superseded-by | |
| Book-chapter  | how/architecture.md |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**,
**SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **NOT RECOMMENDED**, **MAY**, and
**OPTIONAL** in this document are to be interpreted as described in BCP 14
when, and only when, they appear in all capitals.

## Summary

Akita multi-group roots currently select one preset-wide source policy. Every
group is therefore treated as either dense with the preset's coefficient bound
or one-hot with the preset's chunk size. The final schedule records that policy
only for the final group. Precommitted groups carry exact A/B geometry but do
not carry their source contract, so schedule materialization derives their fold
norms from the final group's global policy.

This change makes the source contract group-local:

```rust
pub enum GroupSource {
    Dense { coefficient_bits: u32 },
    OneHot { chunk_size: usize },
}
```

Every independently committed group freezes this contract together with its
layout and exact commitment parameters. Public opening claims carry
self-describing committed groups, so the prover and verifier derive one
ordered schedule key from the exact descriptors that produced the
commitments. The final group also carries a descriptor produced by
`commit_final_group`; it is not reconstructed later from a bare
`PolynomialGroupLayout`.

Source coefficient bounds are distinct from opening-value bounds. A dense
group's `coefficient_bits` prices its committed source and A/fold relations.
Opening evaluations remain full-field values. The root-selected opening basis
and shared D matrix therefore remain shared across groups.

Generated catalogs contain only explicitly generated ordered contract
combinations. They do not enumerate the Cartesian product of every supported
dense bound and one-hot chunk size. Arbitrary checked combinations use the
offline planner and may be emitted as workload-specific catalog rows.

## Audited current limitation

The audit uses base
`b0880f73236b89896b15efd63ff955922307afbe`.

### Public ownership loses frozen metadata

`commit_group` returns
`(PrecommittedGroupDescriptor, Commitment, AkitaCommitmentHint)`, but
`commit_final_group` accepts only `Vec<PolynomialGroupLayout>` and reconstructs
every earlier descriptor through `PrecommittedCommitmentConfig<Cfg>`.
`ProverOpeningData` and verifier `OpeningClaims` then carry only bare
commitments. The descriptors returned at commit time do not reach final
commitment, proving, or verification.

This is safe only while one config deterministically selects one source
contract for every group. It cannot represent independent contracts and can
silently plan a commitment under metadata different from the metadata that
created it.

### The schedule key is asymmetric

`AkitaScheduleLookupKey` contains exact `PrecommittedGroupDescriptor` values for
earlier groups but only `PolynomialGroupLayout` for the final group. The
materialized schedule adds `RootSource` only to `RootFinalGroupParams`.
Descriptor bytes consequently bind the final source and precommitted A/B
geometry, but not a precommitted group's semantic dense/one-hot contract.

### Planner and runtime materialization reuse a global source

Both `akita-planner/src/group_batch.rs` and
`akita-schedules/src/group_batch.rs` derive every precommitted group's
`onehot_chunk_size`, decomposition, fold witness norms, A collision bound, and
fold digit count from `PlannerPolicy.decomposition` and
`PlannerPolicy.onehot_chunk_size`. The final group uses the same policy fields.

This creates four defects for heterogeneous input:

1. a dense earlier group can be priced as one-hot;
2. one-hot groups with different chunk sizes share one L1 bound;
3. a bounded dense group cannot select its actual source digit depth;
4. A rank and fold-witness sizing can be security-underpriced relative to the
   verifier-enforced source.

### Validation flattens source identity

`batched_prove` validates all root polynomials against the final
`CommittedGroupParams.onehot_chunk_size`. It does not validate each polynomial
group against its own params. The verifier reconstructs descriptors from
`OpeningClaimsLayout` and the config, so it has no independent public source
contract to compare with the commitments.

### Generated identity is preset-wide

Generated entries store one final `GeneratedRootSource`; catalog identity stores
the preset-wide decomposition and one-hot chunk size. Multi-group keys and
precommitted descriptor bytes omit ordered per-group source contracts. A
generated lookup therefore cannot distinguish two otherwise identical group
layouts with different source contracts.

### Setup sizing samples homogeneous layouts

The setup-envelope scan is keyed by `OpeningClaimsLayout` and decides whether
to include multi-group shapes from the preset's global
`log_commit_bound == 1`. It does not scan heterogeneous source combinations.
Exact post-resolution footprint checks prevent out-of-bounds setup access, but
setup generation can omit the envelope needed by a valid mixed-source
workload.

## Terminology and ownership

A **group source contract** is the public, checked coefficient class of one
commitment group:

- `Dense { coefficient_bits }` means every centered source coefficient has
  magnitude representable by the declared bit bound.
- `OneHot { chunk_size }` means every polynomial uses the declared one-hot
  chunk size and satisfies the one-hot representation invariant.

A **committed group descriptor** freezes:

- `PolynomialGroupLayout`;
- `GroupSource`;
- live source/block geometry;
- inner and outer gadget bases;
- A and B ring dimensions, ranks, widths, and SIS coefficient bounds.

A **committed group** owns one descriptor and one commitment. The prover-only
hint remains separate because it is not a public statement.

A **final group request** contains the final layout and source contract needed
before schedule search. The exact final descriptor is created from the selected
root commitment parameters and returned with the final commitment.

The ordered committed-group descriptors are public statement data. They are
not proof-provided hints and are never inferred from commitment bytes.

## Canonical types and public API

`RootSource` is renamed to `GroupSource` and becomes the only semantic source
contract type. There is no parallel config-only or verifier-only source enum.

Conceptually:

```rust
pub enum GroupSource {
    Dense { coefficient_bits: u32 },
    OneHot { chunk_size: usize },
}

pub struct CommittedGroupDescriptor {
    pub group: PolynomialGroupLayout,
    pub source: GroupSource,
    // exact frozen A/B and source geometry
}

pub struct CommittedGroup<F: FieldCore> {
    pub descriptor: CommittedGroupDescriptor,
    pub commitment: Commitment<F>,
}

pub struct GroupCommitmentRequest {
    pub group: PolynomialGroupLayout,
    pub source: GroupSource,
}

pub struct AkitaScheduleLookupKey {
    pub final_group: GroupCommitmentRequest,
    pub precommitteds: Vec<CommittedGroupDescriptor>,
}
```

Exact field names may be adjusted to fit existing type ownership, but these
properties are normative:

1. There MUST be one `GroupSource` type.
2. Every committed-group descriptor MUST contain it.
3. `commit_group` and `commit_final_group` MUST return the descriptor paired
   with the commitment.
4. `commit_final_group` MUST accept actual earlier descriptors, not bare
   layouts.
5. Prover and verifier claims MUST expose the ordered descriptors used to
   derive the schedule.
6. The final descriptor MUST be checked against the selected final root params
   before commitment execution and again at prove/verify schedule resolution.
7. No compatibility overload or forwarding alias MAY preserve the
   layout-only multi-group path.

The scalar API may obtain its default source contract from `Cfg` to preserve
existing scalar behavior. The normalized scalar schedule key still has no
precommitted groups, but its final request includes the derived source.

## Source validation

`GroupSource::validate(field_bits)` MUST reject:

- dense bounds of zero;
- dense bounds larger than the field's supported centered representation;
- one-hot chunk size zero;
- chunk sizes that cannot be represented or used by the selected backend;
- arithmetic that overflows while deriving digit or norm parameters.

The contract is an upper bound, not an unchecked promise. Prover commitment and
opening entry points MUST validate the concrete polynomial group against it.
For dense input, decomposition MUST reject a coefficient outside the declared
bound. For one-hot input, every polynomial MUST report the exact declared chunk
size and satisfy existing one-hot shape validation.

Groups are validated independently. A heterogeneous root MUST NOT flatten all
polynomials and validate them against the final group's params.

## Decomposition and security invariants

Let group `g` have source contract `S_g`.

For `Dense { coefficient_bits = b_g }`:

```text
log_commit_bound_g = b_g
source ||s_g||∞     = checked centered bound for b_g
source nonzeros_g   = D_A,g
```

For `OneHot { chunk_size = K_g }`:

```text
log_commit_bound_g = 1
source ||s_g||∞     = 1
source nonzeros_g   = ceil(D_A,g / K_g)
```

Each group independently derives:

```text
δ_inner,g
A width_g
A collision bound_g
n_a,g
fold witness norms_g
fold L∞ cap config_g
δ_fold,g
B width_g and bound_g for the frozen outer basis
```

Planner acceptance and verifier validation MUST call the same canonical SIS and
fold-norm primitives with these values. A generated row MUST NOT carry a rank
or bound that bypasses recomputation.

The precommitted group's exact A/B geometry remains frozen. Choosing the root
opening basis MAY change its fresh opening and fold digit depths only when the
frozen A/B bounds certify that choice.

### Full-field opening values remain separate

The source contract does not bound arbitrary opening evaluations. For every
group:

```text
log_open_bound_g = field_bits
```

unless a future separately specified public opening-value contract replaces
that rule. Dense `coefficient_bits` MUST NOT reduce D/opening security pricing.

## Root-shared opening and D geometry

The root keeps:

- one maximum-arity EOR domain;
- each group's own complete opening point;
- one root-selected `log_basis_open`;
- one shared D matrix over `concat(w_hat_0, ..., w_hat_{G-1})`.

This is sound because D commits decomposed opening-side values, not raw source
coefficients. Its width is:

```text
width_D = Σ_g decomposed_w_width(
    full_field_opening_depth,
    live_blocks_g,
    num_polynomials_g,
)
```

The root MUST use each group's actual block geometry. Source bounds affect A
and folded-response sizing; they do not create per-group D matrices.

## Schedule identity and ordering

The schedule key preserves transcript order:

```text
precommitted descriptor 0
...
precommitted descriptor G-2
final request
```

Canonical bytes MUST encode `GroupSource` with an explicit discriminant and
fixed field order:

```text
Dense  -> 0 || coefficient_bits:u32
OneHot -> 1 || chunk_size:u64
```

The final request encoding MUST include layout then source. Every committed
descriptor encoding MUST include layout, source, then frozen geometry and
matrix facts. Reordering groups or changing one contract MUST change:

- the schedule lookup key;
- generated table lookup;
- effective schedule descriptor bytes;
- catalog key digest and duplicate detection;
- the transcript-bound instance descriptor.

`OpeningClaimsLayout::opening_batch_digest` remains a geometry digest. Source
contracts are bound by the effective schedule digest derived from public
committed-group descriptors. The implementation MAY additionally include them
in a call-level digest only if it removes, rather than duplicates, an existing
source of truth.

## Generated catalogs and offline planning

Catalog keys MUST compare the full final request and every ordered
precommitted descriptor. A false hit is invalid.

The stock generator MUST NOT enumerate all possible:

```text
(Dense bit bounds ∪ OneHot chunk sizes)^G
```

Instead:

1. existing scalar and homogeneous catalog families keep their explicit
   default contract;
2. selected mixed workloads MAY be listed as exact generation cases;
3. arbitrary checked combinations use `akita-planner` offline;
4. a workload that needs runtime lookup emits the resulting exact rows into
   its catalog;
5. verifier runtime remains planner-free and rejects a missing row as
   `UnsupportedSchedule`.

Planner replay tests MUST prove that an emitted mixed row expands to the same
canonical schedule as offline search. Catalog identity continues to bind
preset-wide field, SIS, challenge, basis-range, setup, and recursion policy;
ordered source contracts belong to row keys rather than a global identity
field.

## Setup envelope and performance

Exact `ensure_schedule_fits_setup` validation remains mandatory on commit,
prove, and verify.

Setup generation MUST cover:

- scalar default-source schedules;
- every enabled generated mixed-source row within public capacity;
- precommit recipes for every enabled descriptor;
- bounded representative offline-planner shapes used by tests or shipped
  presets.

The public setup seed need not add a Cartesian source-contract range. A config
or application that permits arbitrary contracts MUST provide an explicit
bounded setup policy or exact workload catalog. “Arbitrary” never means that
setup allocation is unbounded.

Planning complexity for one exact key SHOULD remain:

```text
O(G * candidate_bases * candidate_root_splits + suffix_DP)
```

Source contracts add O(G) derivation per root candidate. Implementations
SHOULD prevalidate/freeze each descriptor once and SHOULD NOT clone polynomial
data, replan each group inside the split loop, or enumerate unused contract
combinations.

## Verifier boundary and malformed input

Before schedule lookup or allocation, the verifier MUST:

1. validate claim group counts and setup capacity with checked arithmetic;
2. validate every committed-group descriptor and source contract;
3. validate descriptor layout against that claim group's point arity and
   evaluation count;
4. require exactly one descriptor-bearing commitment per group;
5. build the ordered schedule key from those descriptors;
6. resolve and validate the exact generated schedule;
7. compare the selected final and precommitted params with every descriptor;
8. validate each commitment row count against its group-local B params;
9. validate the exact schedule footprint against setup;
10. only then bind the instance descriptor, allocate replay state, or index
    prepared setup.

Malformed contracts, descriptor mismatches, overflows, excessive sizes,
unsupported rows, and commitment-shape mismatches MUST return `AkitaError` or
`SerializationError`. Verifier-reachable code MUST NOT use unchecked indexing,
`unwrap`, `expect`, assertions, or allocation sized from an unvalidated
descriptor.

If committed groups are serialized, deserialization MUST use fixed-width
integer encodings, checked `u64 -> usize` conversion, a known source
discriminant, and `Valid::check` before returning a usable object.

## Mixed examples

The primary acceptance shape is:

```text
group 0: OneHot { chunk_size: 16 }
group 1: Dense  { coefficient_bits: 20 }
group 2: OneHot { chunk_size: 256 }  // final group
```

Each group may have a different arity, polynomial count, ring dimension, and
opening point. The planner derives three A/fold contracts, freezes the first
two descriptors at their original commits, plans the final descriptor from
the complete ordered key, and uses one full-field shared D matrix.

Reordering the first two groups is a different statement and schedule key even
when aggregate counts are unchanged.

## Migration and cutover

This repository has no backward-compatibility guarantee. The implementation
MUST perform one cutover:

1. rename `RootSource` to `GroupSource`;
2. add source to the frozen descriptor;
3. replace tuple commitment outputs with descriptor-bearing committed groups;
4. replace layout-only `commit_final_group` input;
5. replace bare-commitment multi-group claim construction;
6. remove config-based reconstruction from opening claims;
7. update generated keys, emitters, catalog identity digests, and checked
   expansion;
8. regenerate affected schedule artifacts;
9. update book/API documentation.

No legacy wrapper, layout-only overload, `_with_sources` sibling, or
config-reconstruction fallback remains.

## Acceptance tests

### Types and validation

- Dense bounds `0` and `field_bits + 1` reject.
- One-hot chunk `0` rejects.
- Descriptor bytes distinguish source variant, dense bound, chunk size, and
  group order.
- Checked conversion and size-overflow cases reject without panic.
- Descriptor serialization round-trips and rejects unknown discriminants and
  invalid values if serialization is exposed.

### Commit and claims

- `commit_group` returns the exact source and frozen params used by the commit.
- `commit_final_group` consumes those exact descriptors and returns an exact
  final descriptor.
- A descriptor/source/chunk/bound mismatch rejects before matrix work.
- Claim layout mismatch, missing descriptor, duplicate group, and reordered
  group reject.
- Existing scalar dense and one-hot commits retain their schedules and
  behavior under the config-derived default source.

### Planner, generated rows, and setup

- Offline planning accepts one-hot K=16 + dense bounded + one-hot K=256.
- Changing only one group source changes the key and schedule descriptor.
- Group-local A ranks, fold depths, and norm caps match direct canonical
  primitive calls.
- Generated mixed rows match offline planner output exactly.
- Generated lookup does not collide with a homogeneous or reordered key.
- A missing mixed row rejects at runtime without invoking planner search.
- Setup envelope covers every enabled mixed row and rejects an exact schedule
  that exceeds materialized capacity.

### Prove and verify

- Round-trip the primary three-group example.
- Repeat with reordered groups, unequal arities, unequal polynomial counts,
  independent opening points, and mixed A/B/D ring dimensions.
- Cover base-field and extension-field openings.
- Preserve existing recursive suffix and recursive setup-prefix behavior.
- Tamper each descriptor source, dense bound, chunk size, layout, matrix bound,
  commitment, opening point, and evaluation independently; verification
  rejects.
- Prover rejects a dense coefficient beyond its declared bound.
- Prover rejects one-hot polynomial metadata with the wrong chunk size.
- Transcript logs differ when one ordered source contract changes.
- Fuzz malformed descriptor sizes and discriminants through verifier-facing
  deserialization and entry validation.

## Non-goals

- Per-polynomial source contracts inside one commitment group.
- Unbounded dense coefficients or unconstrained one-hot representations.
- Per-group D matrices or per-group opening bases.
- Runtime planner reachability from the verifier.
- Cartesian generation of all source-contract combinations.
- Tiered or immediately-terminal multi-group roots.
- Backward compatibility with layout-only group claims or old descriptor bytes.

## Resolved audit decisions

1. **Self-describing commitments are required.** Config reconstruction cannot
   preserve independently selected source contracts.
2. **Source and opening bounds are separate.** Dense source bits do not reduce
   full-field opening/D sizing.
3. **D stays shared.** The source contract does not justify extra D outputs.
4. **The final group is described too.** It is not a privileged globally
   configured source.
5. **Catalog growth is explicit.** Exact mixed rows are generated by workload,
   not by Cartesian expansion.
6. **Verifier runtime stays planner-free.** “Offline fallback” means explicit
   offline search and optional row emission, not verifier-side DP.
7. **No parallel legacy path remains.** Breaking API and descriptor changes
   are intentional.

