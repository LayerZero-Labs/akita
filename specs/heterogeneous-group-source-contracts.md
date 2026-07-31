# Spec: Open heterogeneous group profiles and source providers

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

Akita multi-group batching currently assumes one preset-wide polynomial-source
family. The in-flight implementation on this branch first generalized that
assumption into a group-local `GroupSource` value and then into an open
registration wrapped around a small closed encoding enum. That cut fixes the
immediate dense/one-hot asymmetry, but it still gives the verifier a taxonomy
of source implementations that it does not need.

This specification makes the narrower protocol boundary explicit:

- a **source provider** is an open-world prover/planner extension;
- a **committed-group profile** is the exact public A/B geometry and accepted
  digit envelope of one commitment;
- an **opening schedule selection** identifies the exact approved schedule row
  used for one batch;
- a **committed group** carries its profile together with its commitment;
- the verifier validates profiles, commitments, the selected schedule, setup
  capacity, proof shape, and transcript replay without recognizing a source
  family.

Dense and one-hot become built-in source providers, not variants of a
verifier-facing enum. A downstream repository can register a new source
implementation without changing `akita-types`, the transcript format, or
verifier code, provided the implementation lowers its witness to a valid Akita
commitment profile and uses an approved schedule row.

This cut does **not** redesign multi-group completeness or grinding. It
preserves the current per-group honest-bound formulas, the current digit
snap-down behavior, one shared grind nonce per fold, and the existing terminal
wire model. It documents those choices precisely so a later completeness
change can be evaluated separately.

## Audit baseline and branch scope

The implementation worktree has this additive ancestry:

```text
b0880f73236b89896b15efd63ff955922307afbe
  -> 6f7fde8658bc77fde8c5d1b0fda732068f11e6e7
  -> b70d810e79c53dfc925d8daa7cf8ee76d33d98c2
```

`b70d810e` is the standalone first version of this specification. The
uncommitted implementation is based on `6f7fde86`, the then-current #334 mirror
head. During this rewrite the #334 mirror was observed at
`d3aa279a01cd36a2e37867bfe11888d96f56ec18`. The intervening commits contain
only an extension-opening-reduction test Clippy cleanup and do not alter this
design. They MUST be reconciled additively before the implementation commit;
this spec does not silently claim that the dirty worktree already contains
them.

Everything semantically introduced by this branch after `6f7fde86` is in scope:

1. group-local source preparation and validation;
2. exact frozen descriptors for precommitted and final groups;
3. self-describing public commitments and checked serialization;
4. exact ordered schedule selection through commit, prove, and verify;
5. group-local A sizing, fold sizing, and source validation;
6. generated schedule identity, replay, and the curated mixed-source row;
7. setup-envelope and standalone precommit resolution;
8. verifier boundary checks and malformed-input rejection;
9. scalar, multi-group, extension-field, recursive, profile, and example API
   migration;
10. documentation updates, including making the CI workflow authoritative for
    final test invocations in `AGENTS.md`.

The branch's current `GroupSourceRegistration`, `GroupSourceEncoding`, and
`GroupSource` types are implementation staging, not the final architecture.
Their useful separation of provider identity from protocol encoding is
preserved, but public source identity and the closed encoding enum are removed
from the verifier-facing design below.

## Intent

### Goal

Allow every commitment group in one Akita opening batch to use an independently
implemented polynomial source while exposing only the exact algebraic profile
and approved schedule that the verifier needs.

### Terminology

**Source provider**
: Prover/planner code that understands one concrete polynomial representation,
  validates its values, prepares commitment/opening operations, proposes valid
  commitment profiles, and supplies honest-prover completeness data.

**Source registration**
: An application-side mapping from a stable provider identifier and parameters
  to provider construction. Registration supports configuration, persistence,
  and dynamic dispatch. It is not a PCS statement and is not interpreted by
  the verifier.

**Commitment profile**
: Exact public metadata that determines how a group is represented in the A
  source relation and B commitment: group shape, live block geometry, gadget
  bases and digit depths, and exact A/B matrix parameters.

**Committed group**
: A commitment profile paired with the B commitment rows produced under that
  profile.

**Opening schedule selection**
: A canonical identifier for an approved generated schedule row. The selected
  row fixes the root-shared D geometry, group-local fold parameters, challenge
  families and shapes, recursive suffix, terminal response policy, setup
  footprint, and transcript descriptor.

**Accepted envelope**
: The values the proof system actually accepts: source digits at commitment,
  recursive folded digits at intermediate levels, and the terminal response
  cap at the transparent tail.

**Honest completeness model**
: Prover/planner-only information used to predict whether honest witnesses fit
  an accepted envelope and to choose among valid schedules. It is not a claim
  about which witnesses the verifier accepts.

### Core ownership rule

No source name or semantic source variant is a verifier primitive.

The verifier MUST know:

- the exact group and block geometry;
- both source/A and outer/B gadget bases and digit depths;
- the exact A and B matrix ring dimensions, input widths, output ranks,
  coefficient bounds, and SIS identities;
- the selected root schedule, including D geometry;
- each group's fold challenge configuration and challenge shape;
- the accepted recursive digit envelopes;
- the grind nonce range and wire encoding;
- the terminal response cap and terminal wire rule.

The verifier MUST NOT need:

- whether a provider calls its source dense, one-hot, lookup, sparse, encoded,
  structured, or application-specific;
- a Rust polynomial type;
- a provider registration ID;
- the provider's coefficient-generation algorithm;
- the provider's honest average-case distribution;
- the prover's grind probe order;
- the planner's target acceptance probability or schedule cost model.

This distinction is the minimal completeness/soundness boundary. Source
semantics are necessary to construct an honest witness and plan an efficient
envelope. Exact accepted geometry is necessary to verify the proof. The
verifier does not need the former to enforce the latter.

### Invariants

1. **Exact profile flow.** The profile returned with a commitment MUST be the
   profile used to construct its A/B witness and B rows. Later APIs MUST carry
   that profile; they MUST NOT reconstruct it from a bare
   `PolynomialGroupLayout` or a global config.
2. **Ordered statement.** Group order is transcript order. Reordering groups
   MUST change the opening statement and challenge labels even when aggregate
   dimensions are unchanged.
3. **Open provider boundary.** Adding a source provider that lowers to existing
   Akita relations MUST NOT require extending a core enum or changing verifier
   code.
4. **Accepted-envelope security.** SIS sizing MUST use the full digit ranges
   accepted by the verifier, not a provider's honest distribution or observed
   witness.
5. **Honest-model isolation.** Provider-specific completeness data MAY select
   among already secure schedules, but it MUST NOT weaken verifier admission
   checks or matrix security bounds.
6. **Shared D geometry.** Opening evaluations remain full-field values. The
   root-selected opening basis and D matrix remain shared across groups.
7. **Distinct group challenges.** Every group receives a separate,
   group-indexed fold challenge draw. Challenge vectors are not reused across
   groups.
8. **Current joint grind.** One fold proof carries one nonce. That nonce is used
   in every group-local challenge draw, and the prover accepts it only when all
   group-local fold witnesses pass.
9. **Planner-free verification.** The verifier MUST resolve an approved row and
   validate it. It MUST NOT execute planner search or provider code.
10. **Canonical identity.** Commitment-profile bytes, schedule-row identity,
    serialization, generated lookup, and transcript binding MUST agree on one
    canonical field order.
11. **No verifier panic.** Malformed profiles, selection IDs, sizes, matrix
    shapes, commitment rows, proofs, and serialized values MUST return
    `AkitaError` or `SerializationError` before unchecked indexing or
    allocation.
12. **No parallel legacy path.** Layout-only grouped APIs, config-based
    descriptor reconstruction, and compatibility wrappers MUST be removed.

### Non-goals

- Changing the current multi-group completeness probability model.
- Reallocating the current `1/8` target across groups.
- Replacing the existing tail formula, digit snap-down, `n_snap` behavior, or
  terminal average-case byte model.
- Adding one grind nonce per group in this cut.
- Proving that a provider's private semantic description is true beyond the
  accepted digit/range relations already checked by Akita.
- Per-polynomial providers inside one committed group.
- Per-group D matrices or per-group opening bases.
- Changing the folded-only topology: multi-group structure remains root-only,
  recursive suffixes remain singleton, precommitted root groups remain flat,
  and tiered or immediately terminal multi-group roots remain unsupported.
- Runtime planner execution in the verifier.
- Cartesian generation of all provider or profile combinations.
- Unbounded setup, schedule, polynomial, or descriptor sizes.
- Backward compatibility with old commitment or schedule descriptor bytes.

## Audited limitation

### The original public boundary loses commitment facts

At the audited base, `commit_group` returns frozen metadata, a commitment, and
a hint, but `commit_final_group` accepts only earlier
`PolynomialGroupLayout`s. It reconstructs those groups through a single config.
Prover and verifier claims carry bare commitments, so the metadata returned at
commitment time does not reach the final root or verification boundary.

This is only coherent while one config determines one representation and one
profile for every group. It cannot safely express independently prepared
groups.

### The original lookup key is asymmetric

Earlier groups have frozen A/B metadata, while the final group is represented
only by its polynomial layout and a preset-wide source choice. The generated
row and transcript descriptor therefore cannot identify an arbitrary ordered
set of exact group profiles.

### Planner and runtime materialization reuse global source facts

The original grouped planner and generated-row expansion reuse
`PlannerPolicy.decomposition` and `PlannerPolicy.onehot_chunk_size` for every
group. Consequently:

- a dense group can inherit one-hot fold norms;
- one-hot groups with different sparsities can inherit the same norm;
- A width and rank can be derived from the wrong source digit depth;
- fold depth can be under- or over-sized for earlier groups;
- generated identity cannot distinguish otherwise equal layouts.

The in-flight branch has already demonstrated this failure in an end-to-end
mixed test: a global final-group source was still used to size an earlier
group's fold witness. The group-local regression caught and corrected that
specific use. The final design removes the semantic source dependency
entirely from verifier-visible params rather than expanding the taxonomy.

### Validation is flattened

The original prover flattens all root polynomials and validates them against
the final group's one-hot configuration. This rejects valid heterogeneous
groups and can fail to enforce an earlier group's actual representation.

Validation MUST be group-local and provider-owned before commitment and
opening. The core then validates the resulting accepted profile independently.

### Generated and setup identity are homogeneous

Generated keys and setup scans are built around preset-wide source policy.
Naively adding an enum value per source does not solve this: an open registry
would turn catalog generation into an unbounded Cartesian product.

The correct unit of generated identity is an exact approved schedule row over
ordered commitment profiles. Providers are not enumeration axes.

## Design

### Architecture

The data flow is:

```text
downstream source provider
  -> prepare and validate one concrete group
  -> propose checked commitment-profile candidates
  -> offline planner or generated-catalog resolver
  -> exact approved schedule row
  -> commit under the row's exact final profile
  -> CommittedGroup { profile, commitment }
  -> grouped opening statement { row selection, ordered committed groups }
  -> prover and verifier resolve the same row
  -> verifier checks profile/row/setup/proof/transcript consistency
```

The source provider disappears at the public statement boundary. Two unrelated
providers that produce the same profile and witness relation can use the same
schedule row. Conversely, two schedules with the same commitment profiles but
different valid fold or terminal choices have distinct schedule-row IDs.

### Exact committed-group profile

The final public shape is conceptually:

```rust
pub struct CommittedGroupProfile {
    pub version: u8,
    pub group: PolynomialGroupLayout,

    pub num_live_ring_elements_per_claim: usize,
    pub num_positions_per_block: usize,
    pub num_live_blocks: usize,

    pub log_basis_inner: u32,
    pub num_digits_inner: usize,
    pub inner_commit_matrix: InnerCommitMatrixParams,

    pub log_basis_outer: u32,
    pub num_digits_outer: usize,
    pub outer_commit_matrix: OuterCommitMatrixParams,
}

pub struct CommittedGroup<F: FieldCore> {
    pub profile: CommittedGroupProfile,
    pub commitment: Commitment<F>,
}
```

This is illustrative naming, but the field ownership is normative.

Both bases are required:

- `log_basis_inner` and `num_digits_inner` define the accepted source/A digit
  layout and the A consistency rows;
- `log_basis_outer` and `num_digits_outer` define the B input encoding and
  therefore the commitment's row relation.

The profile MUST carry exact digit depths. A `coefficient_bits` value is not a
substitute: it is one provider's input to decomposition planning, not the
protocol geometry the verifier checks.

The profile MUST carry the complete canonical A/B matrix parameters rather
than duplicating a partial list of ranks and bounds. Each matrix parameter
object binds at least:

- SIS security-policy and table identity;
- modulus profile;
- matrix role;
- ring dimension;
- input width;
- output rank;
- coefficient infinity-norm bound.

The core MUST cross-check:

```text
A width =
    num_positions_per_block * num_digits_inner

B width =
    A output rank
    * num_digits_outer
    * num_live_blocks
    * group.num_polynomials
    * (A ring dimension / B ring dimension)
```

All multiplication, division, shifts, powers of two, and conversions use
checked arithmetic. The current carrier relation requires nonzero power-of-two
A/B dimensions with the A dimension divisible by the B dimension.

The profile does not contain:

- a provider registration;
- a dense bound or sparse chunk size as semantic data;
- the shared D matrix;
- the consuming root's opening basis;
- a fold challenge configuration;
- a fold digit depth;
- a terminal response policy.

Those last four items belong to the selected opening schedule because an
independently committed group may be consumed by different valid roots.

### Commitment serialization

`CommittedGroup` serialization is a breaking, versioned format:

```text
profile.version:u8
group.num_vars:u64
group.num_polynomials:u64
num_live_ring_elements_per_claim:u64
num_positions_per_block:u64
num_live_blocks:u64
log_basis_inner:u32
num_digits_inner:u64
canonical InnerCommitMatrixParams bytes
log_basis_outer:u32
num_digits_outer:u64
canonical OuterCommitMatrixParams bytes
canonical Commitment bytes
```

Platform sizes MUST serialize through checked `usize <-> u64` conversions.
Deserialization MUST:

1. reject an unknown profile version;
2. read only fixed-width lengths before validation;
3. validate group and block geometry;
4. validate both matrix parameter objects and exact derived widths;
5. compute `n_b * d_b` with checked arithmetic;
6. reject a coefficient count above the repository allocation cap;
7. only then allocate and deserialize commitment rows;
8. run `Valid::check` before returning a usable group.

Provider IDs and provider parameters MUST NOT appear in these bytes.

### Open source-provider interface

Akita exposes an open trait at the prover/planner boundary. Exact Rust names
remain implementation choices because backend associated types and const ring
dimensions need to fit the existing compute traits, but the interface MUST
provide these capabilities:

```rust
pub trait GroupSourceProvider<F> {
    type PreparedGroup;

    fn prepare_group(
        &self,
        input: Self::Input,
        layout: PolynomialGroupLayout,
        policy: &CommitmentPlanningPolicy,
    ) -> Result<Self::PreparedGroup, AkitaError>;

    fn commitment_candidates(
        &self,
        prepared: &Self::PreparedGroup,
    ) -> Result<Vec<CommitmentProfileCandidate>, AkitaError>;

    fn honest_fold_model(
        &self,
        prepared: &Self::PreparedGroup,
    ) -> &dyn HonestFoldModel;
}
```

This snippet describes responsibilities, not a required object-safe signature.
In particular:

- the provider validates every concrete polynomial in the group;
- it prepares source digits and any specialized commitment/opening kernels;
- it proposes bounded profile candidates under the application's setup policy;
- it supplies the current honest fold model for planner and grind preparation;
- it cannot construct unchecked matrix params or bypass core validation;
- the core revalidates every returned candidate using canonical geometry and
  SIS primitives.

The provider trait MUST be implementable outside the Akita repository. There
MUST NOT be a sealed trait or exhaustive match over source families in
`akita-types`, `akita-schedules`, `akita-verifier`, or transcript code.

Built-in providers replace the current semantic variants:

- bounded dense coefficients;
- one-hot/sparse-binary coefficients.

Other providers MAY reuse either provider's preparation helpers, define a new
honest completeness model, or lower to the same accepted digit profile through
different storage and kernels.

### Registration system

Dynamic applications MAY register providers:

```text
ProviderId + canonical application parameters
    -> provider constructor
```

Registration has three purposes:

- deserialize application job configuration;
- select downstream preparation code;
- key provider-local caches and diagnostics.

Registration MUST NOT:

- be required by the verifier;
- determine transcript identity;
- select a schedule row without an exact profile match;
- be trusted as a proof that a witness has some semantic form;
- create a catalog-generation axis.

A static downstream integration MAY use ordinary Rust trait bounds instead of
the dynamic registry. Both paths lower to the same prepared-group and exact
profile boundary.

### Exact schedule selection

The opening statement carries one schedule selection in addition to ordered
committed groups:

```rust
pub struct OpeningScheduleSelection {
    pub catalog_identity: CatalogIdentity,
    pub row_digest: ScheduleRowDigest,
}

pub struct GroupBatchStatement<'a, E, F> {
    pub schedule: OpeningScheduleSelection,
    pub groups: Vec<PolynomialGroupClaims<'a, E, &'a CommittedGroup<F>>>,
}
```

Again, names are illustrative; these properties are normative:

1. The schedule selection is batch-level, not copied into every commitment.
2. The row digest identifies the complete canonical schedule descriptor.
3. The catalog identity binds the preset-wide field, challenge, SIS, setup,
   recursion, terminal-wire, and cost-policy version expected by the config.
4. The selected row contains the exact ordered commitment profiles it accepts.
5. The verifier resolves by identity/digest and never searches.
6. The verifier compares every group profile with the corresponding row
   profile before transcript replay.
7. A row cannot be selected merely because aggregate widths match.

The current `AkitaScheduleLookupKey` SHOULD become an exact ordered-profile key
used for generation, duplicate detection, and diagnostics. It SHOULD NOT carry
a `final_source` or source registration:

```rust
pub struct AkitaScheduleLookupKey {
    pub groups: Vec<CommittedGroupProfile>,
}
```

The vector is nonempty and ordered as:

```text
precommitted group 0
...
precommitted group G-2
final group G-1
```

Before the final commitment exists, the prover-side resolver combines the
actual earlier profiles with the final provider's bounded profile candidates.
It selects an approved row, obtains the exact final profile from that row, and
commits the final group under that profile. The resulting public statement is
therefore not circular: row selection precedes the final B commitment, while
verification later reconstructs the exact row key from all committed groups.

If more than one secure row supports the same commitment profiles, each row
has a different digest. The prover may choose any row allowed by the
application/catalog policy. The verifier checks the chosen row rather than
pretending that source semantics determine a unique schedule.

### Why the verifier accepts schedule choice

Schedule choice is not an unchecked prover degree of freedom. The resolved row
is approved generated data whose full descriptor is:

- canonically bound to the transcript;
- structurally validated;
- rechecked against the SIS tables and accepted digit envelopes;
- matched to exact committed profiles;
- checked against the materialized setup envelope;
- checked against the proof's recursive and terminal shape.

A provider's completeness model may prefer one approved row, but cannot cause
the verifier to accept a row with insufficient ranks, widths, digit ranges, or
terminal capacity.

### Group-local source validation

Concrete source validation occurs before expensive matrix work and again
before proving through the prepared group object.

The built-in dense provider MUST:

- compute the largest centered coefficient magnitude over the live logical
  coefficients;
- derive its checked bit width without scanning zero padding as live input;
- reject a coefficient outside its requested commitment candidate;
- produce balanced source digits that fit the row's
  `(log_basis_inner, num_digits_inner)` envelope.

The built-in one-hot provider MUST:

- require the one-hot representation rather than accepting arbitrary dense
  storage;
- validate the chunk layout and index range;
- require the exact provider-requested chunk size;
- lower the representation to source digits fitting the selected profile.

The core MUST NOT flatten heterogeneous groups and call one provider's
validator over all polynomials.

### Soundness versus completeness

The implementation MUST keep three quantities separate.

#### Source acceptance

At commitment, the verifier-visible source relation accepts balanced digits
under:

```text
log_basis_inner
num_digits_inner
A input width and matrix security
```

Soundness prices the full accepted digit envelope. A provider's statement that
its honest coefficients are smaller does not reduce the A collision bound
unless it selects a smaller public digit profile that the proof actually
enforces.

Provider semantics are not silently promoted into proof constraints. For
example, the built-in one-hot provider may use one-nonzero-per-chunk sparsity
to predict an honest folded response, while the Akita source relation proves
only the public digit/range profile it actually contains. A malicious prover
using a denser accepted digit vector may have worse completeness, but does not
escape the verifier's accepted envelope or its SIS pricing. An application that
needs one-hot sparsity itself to be a verified semantic statement MUST prove
that property in an explicit relation; a provider registration is not such a
proof.

#### Recursive fold acceptance

At a recursive root or fold, the schedule fixes:

```text
log_basis_open
num_digits_fold
group-local fold challenge config and shape
```

The verifier range-checks the folded digits and prices the full representable
envelope. It does not need to know why the planner expected the honest folded
response to fit.

#### Terminal response acceptance

At the transparent terminal fold, the response is revealed. The verifier
therefore needs the explicit terminal response cap and wire rule to:

- reject a coefficient above the admitted range;
- decode the canonical terminal representation;
- validate the terminal relation;
- account for the terminal SIS/security envelope.

The honest completeness model may predict a tighter distribution, but the
terminal cap in the selected row is the public admission rule.

### Honest fold model

An open provider supplies an honest fold model to the prover and offline
planner. The minimal abstraction is a function from a checked group/schedule
context to an honest cap proposal and supporting diagnostics. It MUST NOT
require every future source to pretend that `(L∞, L1)` is its natural semantic
description.

Conceptually:

```rust
pub trait HonestFoldModel {
    fn plan(
        &self,
        context: &FoldCompletenessContext,
    ) -> Result<HonestFoldPlan, AkitaError>;
}

pub struct HonestFoldPlan {
    pub unsnapped_linf_cap: u128,
    pub decomposed_fold_digits: usize,
    pub snapped_linf_cap: u128,
}
```

The core validates that the proposed digit depth is representable and that the
resulting schedule is SIS-secure. Built-in models MUST call the existing
canonical primitives rather than duplicate their arithmetic.

The current dense and one-hot models use `FoldWitnessNorms`:

```text
dense:
  ||s||∞ = balanced_digit_abs_max(log_basis_inner)
  ||s||1 = d_A * ||s||∞

one-hot/sparse-binary:
  ||s||∞ = 1
  ||s||1 = ceil(d_A / chunk_size)
```

Those values are provider-side completeness facts. They are not a universal
verifier-facing source schema.

### Current completeness formulas are frozen in this cut

For the built-in models, the current worst-case negacyclic product bound stays:

```text
product_bound =
  min(
    ||c||∞ * ||s||1,
    ||c||1 * ||s||∞
  )

beta_inf =
  num_claims * num_live_blocks * product_bound
```

For a certified flat challenge family, the current tail proxy stays:

```text
t_star_squared =
  2
  * (num_claims * num_live_blocks)
  * challenge_l2_sq_max
  * witness_linf_squared
  * grind_union_ln

unsnapped_cap = min(beta_inf, ceil_sqrt(t_star_squared))
```

For a certified tensor challenge family, the existing
`rademacher_proxy_variance_tensor_challenges` formula remains authoritative.
Uncertified families continue to use `beta_inf`.

`fold_witness_digit_plan` continues to:

1. derive the unsnapped cap;
2. choose the corresponding balanced digit depth;
3. walk the depth downward while the representable envelope retains the
   existing snap fraction;
4. use `3/4` for fields narrower than 128 bits and the current protocol-bound
   `1/2` for wide fields;
5. return the snapped honest-prover cap used by grind admission.

The current `p_grind = 1/8`, `MAX_FOLD_GRIND_ATTEMPTS = 4096`, snap behavior,
and terminal planner model remain unchanged. This spec intentionally does not
claim that the resulting bound is tight or that the nominal `1/8` predicts
observed rejection. The current snap margin makes acceptance effectively high
for shipped workloads; changing that model requires its own analysis and
measurements.

Multi-group planning continues to apply the current model independently to
each group and search for one nonce accepted jointly. It does not introduce a
batch-level probability correction in this cut.

### Exact multi-group challenge sampling

Groups do **not** reuse one fold challenge vector.

For one root fold, prover and verifier perform this exact sequence:

1. absorb the shared D output
   `v = D * concat(e_hat_0, ..., e_hat_{G-1})` once;
2. iterate groups in `OpeningClaims` order;
3. for group `g`, obtain its native A ring dimension, live-block count,
   polynomial count, challenge config, and challenge shape;
4. construct a sample label binding:
   - the fold-round domain;
   - `group_index = g`;
   - `num_live_blocks_g`;
   - `num_claims_g`;
   - flat or tensor shape and tensor low length;
   - the base challenge label;
5. absorb the label, challenge count, native ring dimension, challenge-family
   domain separator, and the shared grind nonce;
6. squeeze a fresh group-local seed and sample that group's challenges;
7. for a tensor draw, sample the high factor, absorb its digest, then sample
   the low factor.

Because each draw mutates the transcript, later groups are also downstream of
earlier group draws. The explicit group index and geometry prevent equal-shaped
groups from aliasing.

The selected schedule MUST bind each group's challenge config and shape.
Changing group order, challenge family, shape, ring dimension, block count, or
claim count MUST change replay.

This provider cut does not widen the typed schedule topology. Earlier
precommitted root groups continue to use flat challenges; only the final root
group may select the currently supported tensor challenge. After the root, the
recursive suffix is singleton.

### Why the verifier cares about grind policy

The verifier cares only about the part of grinding that changes its accepted
Fiat-Shamir space:

- whether the level permits only nonce zero or a bounded nonce range;
- the exclusive nonce bound;
- the canonical nonce wire width;
- how the nonce enters each group-local challenge seed;
- the resulting grinding entropy charged to soundness.

With one shared nonce in `[0, Q)`, the prover can choose among at most `Q`
joint challenge tuples for the complete group batch. The soundness budget
therefore charges `log2(Q)` at that fold, not once per group.

The verifier does not need the honest tail formula, the `1/8` planner target,
the prover's sequential or shuffled probe order, the first-accepting convention,
or the terminal expected-byte model. It cannot verify which nonces the prover
tried. Those values may remain descriptor-versioned for compatibility during
the cutover, but the implementation SHOULD split them into:

- a verifier/protocol nonce-and-wire binding;
- a prover/planner completeness and cost-model binding.

This ownership refactor MUST preserve current transcript bytes and accepted
nonce ranges unless a later spec explicitly changes them.

### One nonce per group is deferred

A future design may carry `nonce_g` for every group. It could improve honest
multi-group completeness by allowing each group to find an accepting draw
independently. It is not a free wire-format change:

- prover and verifier would replay a nonce vector in group order;
- transcript labels and proof shape would change;
- the verifier would validate every nonce before sampling;
- a range of size `Q` for each of `G` groups exposes up to `Q^G` challenge
  tuples;
- a conservative soundness budget would therefore charge up to
  `G * log2(Q)` unless a tighter joint argument is proved;
- terminal and recursive group interactions would need new completeness
  measurements.

This work belongs in a separate specification.

### Root-shared opening and D geometry

Source coefficients and opening evaluations have different bounds. A
provider's source profile MUST NOT reduce the opening-value range.

The root retains:

- one maximum-arity EOR domain;
- each group's own complete opening point;
- one root-selected `log_basis_open`;
- group-local `num_digits_open` derived for full-field evaluations;
- one shared D matrix over the ordered concatenation of all `e_hat_g`.

For group `g`, its D-segment width remains:

```text
num_digits_open_g
* num_live_blocks_g
* num_polynomials_g
* (d_A,g / d_D)
```

The total D input width is the checked sum of those segment widths. The
selected row MUST validate every A-to-D carrier ratio and the final sum before
setup access or allocation.

### Generalizing `ProverOpeningData`

The audited-base `ProverOpeningData` stored:

- one homogeneous polynomial type `P`;
- raw opening claims;
- a parallel hints vector;
- parallel polynomial slices;
- an optional reconstructed schedule key.

That ownership is unsuitable for open heterogeneous providers. It encourages
flattening and makes index alignment a global invariant.

The replacement MUST preserve `OpeningClaims` as the canonical public grouping
and aggregate the prover-only material for each corresponding group:

```rust
pub struct ProverOpeningData<'a, E, G, F> {
    schedule: OpeningScheduleSelection,
    claims: OpeningClaims<'a, E, Commitment<F>>,
    group_inputs: Vec<ProverGroupInput<G, F>>,
}

pub struct ProverGroupInput<G, F> {
    hint: AkitaCommitmentHint<F>,
    prepared_source: G,
}
```

This avoids duplicating points, evaluations, and commitments already owned by
`OpeningClaims`. Construction checks the one public-claim-group to one
`ProverGroupInput` correspondence once. Thereafter a hint and its prepared
source cannot be independently reordered. The implementation MAY use enums,
arenas, borrows, or generics to avoid heap allocation.

Type erasure SHOULD occur at the whole-group operation boundary, not at every
low-level polynomial read. Existing `RootOpeningSource` and backend kernels use
generic associated types, const ring dimensions, and monomorphized hot loops.
Wrapping those low-level traits directly in `dyn` objects would either be
impossible or impose unnecessary virtual calls.

The prepared group carrier instead owns one prepared homogeneous group and exposes
coarse operations required by the root protocol, such as:

- validate prepared witness against the selected commitment profile;
- produce group-local opening rows;
- prepare or execute group-local fold accumulation for a dispatched ring
  dimension;
- expose checked provider completeness data to the grind planner;
- report the group layout and exact profile.

The root protocol iterates group records. It MUST NOT require one global
polynomial type `P`, build one `flat_polys` source-validation path, or maintain
independent parallel hint and polynomial vectors.

### Public APIs

The scalar and grouped APIs perform one cutover.

Standalone group preparation and commitment:

```text
provider.prepare_group(...)
resolver.select_standalone_profile(...)
commit_group(prepared_group, exact_profile)
    -> (CommittedGroup, AkitaCommitmentHint)
```

Final group commitment:

```text
resolve_group_batch(
    exact earlier CommittedGroupProfile values,
    prepared final provider candidates,
)
    -> ResolvedGroupBatch { selection, schedule, final_profile }

commit_final_group(prepared_final, ResolvedGroupBatch)
    -> (CommittedGroup, AkitaCommitmentHint)
```

Proving:

```text
ProverOpeningData {
    schedule selection,
    ordered group records carrying exact committed groups,
}
```

Verification:

```text
GroupBatchStatement {
    schedule selection,
    ordered claims over self-describing committed groups,
}
```

The repository has no backward-compatibility guarantee. There MUST NOT be:

- a layout-only `commit_final_group`;
- a `_with_profiles` sibling while the old path remains;
- a verifier overload accepting bare commitments for grouped openings;
- a config adapter that reconstructs earlier profiles;
- forwarding aliases for removed source-enum APIs.

Scalar behavior is preserved by having each config select its built-in default
provider and schedule row. Scalar commitments still become self-describing so
one public commitment type serves both scalar and grouped APIs.

### Generated catalogs and offline planning

Generated catalogs contain exact schedule rows. They do not enumerate source
providers.

The stock generator MUST:

- retain existing scalar and homogeneous families through their built-in
  provider preparations;
- emit selected mixed-profile workloads explicitly;
- sort and deduplicate on canonical exact-profile keys and full row
  descriptors;
- reject a duplicate key/digest with different row contents;
- bind exact ordered profiles in catalog identity and row digest;
- replay every emitted row through the runtime expander and compare it with the
  offline planner result.

The acceptance workload remains:

```text
group 0: built-in one-hot provider, K = 16, arity 14, one polynomial
group 1: built-in bounded-dense provider, 32-bit request, arity 15, two polynomials
group 2: built-in one-hot provider, K = 256, arity 16, one polynomial
```

Its source names and parameters are generation inputs. The emitted public row
contains exact profiles and schedule geometry. Its reordered form is a
different ordered key and remains deliberately absent from the stock catalog.

An arbitrary downstream provider uses the offline planner and emits the exact
approved row into its application catalog. Runtime verification rejects an
unknown selection with `UnsupportedSchedule`; it does not invoke the planner.

### Setup envelope

Setup generation MUST cover:

- every enabled generated schedule row;
- every exact profile used for standalone precommit in those rows;
- scalar default-provider rows;
- recursive/setup-prefix rows reachable from those schedules;
- the maximum A, B, D, and setup-prefix dimensions of the enabled catalog.

The envelope is a maximum over exact rows, not a Cartesian product over
provider registrations. Applications that accept arbitrary provider requests
MUST still supply a bounded catalog and setup policy.

`ensure_schedule_fits_setup` remains mandatory at commit, prove, and verify.
The verifier MUST perform exact checked footprint validation before prepared
setup indexing.

### Transcript identity

Canonical instance binding MUST include:

1. active protocol/version bindings;
2. catalog identity and exact schedule row digest;
3. the full expanded schedule descriptor;
4. ordered commitment-profile bytes;
5. ordered group layouts and opening points;
6. commitment rows and claimed evaluations through their existing transcript
   locations;
7. group-local challenge labels and the shared nonce as described above.

Provider registration IDs MUST NOT be transcript inputs. If two providers
lower to identical profiles and relations, the verifier should not distinguish
them.

Changing any accepted geometry, matrix parameter, group order, challenge
configuration, challenge shape, recursive suffix, terminal cap, or wire rule
MUST change the schedule digest or instance descriptor.

### Verifier boundary

Before transcript replay or attacker-sized allocation, verification MUST:

1. validate the number of groups and claims with checked arithmetic;
2. resolve the schedule selection in the configured catalog;
3. validate catalog identity and row digest;
4. validate every committed-group profile;
5. compare profile layout with point arity and evaluation count;
6. validate exact A/B widths, roles, ranks, bounds, ring dimensions, and SIS
   identities;
7. validate commitment row count as `n_b * d_b`;
8. reconstruct the ordered exact-profile key and compare it with the row;
9. validate the expanded schedule structure and group-local consuming params;
10. validate D segment widths and shared D geometry;
11. validate recursive digit envelopes and terminal response policy;
12. validate the shared grind nonce against the selected fold's range;
13. validate the exact setup footprint;
14. validate proof shape;
15. only then allocate replay state, index setup matrices, or draw challenges.

Unsupported rows, unknown versions, overflows, excessive sizes, profile
mismatches, malformed commitments, and source-independent relation failures
return `AkitaError::InvalidProof`, `AkitaError::UnsupportedSchedule`, or
`SerializationError` as appropriate.

### Downstream integration story

A downstream repository that wants multi-group batching:

1. implements or selects one source provider per group;
2. prepares and validates each concrete group;
3. obtains bounded commitment-profile candidates;
4. uses an existing application catalog row or runs the Akita offline planner;
5. emits any new exact row into its application catalog and regenerates setup
   capacity;
6. commits earlier groups and retains their self-describing
   `CommittedGroup`s;
7. resolves the final batch using those exact profiles;
8. commits the final group under the selected row's exact profile;
9. passes one schedule selection and ordered group records to prove/verify.

If a new provider lowers to an existing exact profile and completeness is
acceptable, it can reuse an existing row without changing Akita core. If it
needs a new profile or fold plan, it emits a new row; verifier code still does
not change.

### Affected crate surfaces

`akita-types`
: Replace source-bearing descriptors with exact profiles; define
  self-describing commitments, schedule selection identity, canonical bytes,
  checked serialization, and verifier-visible accepted envelopes. Remove
  `GroupSourceRegistration`, `GroupSourceEncoding`, public `GroupSource`, and
  `LevelParamsLike::source`.

`akita-prover`
: Define the open provider/prepared-group boundary; move dense/one-hot
  validation behind built-in providers; generalize `ProverOpeningData`; execute
  group-local operations and the current joint grind without flattening.

`akita-planner`
: Accept prepared profile candidates and provider-side honest models; validate
  every candidate with canonical SIS primitives; preserve current completeness
  formulas; emit exact row/profile identity.

`akita-schedules`
: Expand and validate rows from exact profiles; remove source enum matches;
  sort, deduplicate, hash, and resolve exact row identities; stay planner-free
  at runtime.

`akita-config`
: Select built-in default providers for scalar presets; resolve catalog rows
  and bounded setup envelopes; remove config-based reconstruction of earlier
  groups.

`akita-pcs`
: Expose the one-shot public API cutover; migrate examples, benches, recursive
  test support, and profile helpers to self-describing commitments and schedule
  selection.

`akita-verifier`
: Resolve the selected row, validate exact profiles and setup capacity, replay
  distinct group-local challenges with the shared nonce, and reject malformed
  input without provider logic.

`akita-setup`
: Size matrices and setup prefixes from enabled exact rows and profiles.

## Alternatives considered

### Public `Dense | OneHot` enum

This fixes the first two built-in cases but makes every future source a
protocol/API change. It also conflates provider completeness semantics with
verifier acceptance geometry. Rejected.

### Open registration plus closed encoding enum

This is the branch's current staging design. It lets downstream storage formats
register against `Bounded` or `SparseBinary`, but a genuinely new honest model
or source encoding still requires extending the closed enum. The verifier also
binds provider identity that it does not use. Rejected as the final boundary;
useful as an intermediate implementation step only.

### Fully opaque provider certificate interpreted by the verifier

An opaque type ID plus bytes merely moves the closed switch into a registry.
Unless verifier code understands the certificate, it cannot derive security
facts; if it does understand it, the verifier taxonomy remains closed.
Rejected.

### Put only source bounds in the public descriptor

Values such as coefficient bits or one-hot chunk size are insufficient. The
verifier needs the exact digit layout and A/B matrix geometry. Different
providers can share those protocol facts. Rejected.

### Infer profiles from commitment bytes

Commitment rows do not uniquely encode the A/B geometry that created them.
Inference would either be ambiguous or reintroduce config-global assumptions.
Rejected.

### Inline arbitrary schedules in every proof

A fully self-describing inline schedule could be validated, but it enlarges the
attacker-controlled deserialization surface and duplicates generated catalog
policy. An approved row digest gives open offline planning while retaining a
small runtime boundary. Not selected for this cut.

### One low-level trait object per polynomial

The existing backend traits rely on static ring dispatch and monomorphized
kernels. Erasing each coefficient access would complicate object safety and
hurt hot loops. Erase at the prepared whole-group boundary instead.

### One nonce per group now

This may improve completeness, but changes the proof, transcript, grinding
entropy, and security accounting. Deferred to a separate spec.

## Evaluation

### Acceptance criteria

- [ ] A downstream test crate defines a source provider without modifying any
      Akita enum or verifier match and completes scalar commit/prove/verify.
- [ ] The same downstream provider participates in a multi-group proof by
      reusing or emitting an exact schedule row.
- [ ] `CommittedGroupProfile` contains no provider/source registration and no
      semantic dense/one-hot variant.
- [ ] Every public commit returns `CommittedGroup`; every grouped final commit
      consumes exact earlier profiles and a resolved schedule.
- [ ] Layout-only descriptor reconstruction and bare-commitment grouped verify
      APIs are absent.
- [ ] Profile serialization round-trips and rejects unknown versions,
      overflows, excessive allocation, invalid matrix roles/widths, and row
      count mismatches.
- [ ] Generated row identity distinguishes every exact profile field, group
      order, challenge config/shape, fold envelope, D geometry, recursive
      suffix, and terminal policy.
- [ ] Provider registration changes do not change transcript bytes when the
      exact profile and schedule row are unchanged.
- [ ] The curated K=16 + dense-32 + K=256 row matches offline planning exactly.
- [ ] The reordered mixed key misses the stock catalog without runtime planner
      execution.
- [ ] Group-local source validation rejects a dense coefficient outside its
      selected profile and rejects malformed one-hot input before matrix work.
- [ ] The mixed end-to-end proof uses different opening points and independently
      checked profiles for all three groups.
- [ ] Base-field and extension-field grouped openings pass.
- [ ] Existing recursive suffix and mixed-ring-dimension grouped cases pass.
- [ ] Tampering any profile field, row selection, group order, commitment row,
      opening point, evaluation, challenge shape, or terminal cap rejects.
- [ ] A logging-transcript test shows distinct group-local fold challenge events
      and exact prover/verifier event equality.
- [ ] Equal-shaped groups receive different challenge seeds because the group
      index and sequential transcript state differ.
- [ ] One shared nonce is replayed for every group and an out-of-range nonce
      rejects before sampling.
- [ ] Current scalar dense/one-hot schedule behavior and current homogeneous
      multi-group behavior remain unchanged apart from intentional descriptor
      version bytes.
- [ ] Current fold completeness formulas, snap thresholds, shared-nonce search,
      and terminal wire behavior have snapshot tests proving no unplanned drift.
- [ ] Every enabled mixed row fits the generated setup envelope; an undersized
      setup rejects without panic.
- [ ] Generated artifacts replay deterministically after regeneration.
- [ ] All verifier-facing fuzz/malformed-input tests remain no-panic.

### Testing strategy

Unit tests:

- exact profile validation and derived A/B widths;
- canonical profile and selection bytes;
- commitment serialization and allocation caps;
- provider registration independence from protocol identity;
- group-local dense and one-hot provider validation;
- current dense/one-hot honest-model formula snapshots;
- current snap and nonce-range snapshots;
- group-indexed flat and tensor challenge labels.

Planner/schedule tests:

- exact mixed-row offline planning and generated replay;
- duplicate/collision detection;
- reordered group miss;
- two providers lowering to one profile reuse one row;
- two valid rows over one profile key remain distinguished by row digest;
- setup-envelope maxima over all enabled exact rows.

End-to-end tests:

- one-hot K=16, bounded dense, one-hot K=256;
- unequal arities and polynomial counts;
- independent points;
- base and extension fields;
- recursive suffix and per-matrix ring transitions;
- every descriptor/selection tamper case;
- deliberate provider/profile mismatch;
- transcript equality and group-local challenge separation.

Feature and repository gates:

- run the cheap preflight from `AGENTS.md` before expensive compilation;
- run all three exact Clippy feature matrices;
- obtain the current CI test invocation from `.github/workflows/ci.yml`,
  including its Cargo profile, targets, features, and sharding semantics;
- poll every live process to a real exit code;
- run documentation guardrails for this spec and superseded records.

### Performance

The design MUST avoid:

- cloning polynomial buffers to cross the provider boundary;
- virtual dispatch inside coefficient-level hot loops;
- replanning the same prepared group for every root split;
- catalog enumeration over provider registrations;
- verifier-side schedule search;
- attacker-controlled unbounded profile or schedule allocation.

Offline planning for one exact candidate set SHOULD remain linear in group
count per evaluated root candidate, plus the existing suffix dynamic program.
The provider prepares each group once and supplies bounded profile candidates.

Measure the curated mixed workload and existing homogeneous baselines with the
repository profile harness. Record:

- setup, commit, prove, and verify time;
- proof bytes and terminal/fold breakdown;
- peak setup A/B/D capacities;
- selected A/B/D ring dimensions, widths, ranks, bases, and digit depths;
- grind attempts and observed acceptance.

This cut makes no claim that the current completeness model or expected
terminal byte model is optimal. It requires only that the open-provider
refactor not introduce an unexplained regression relative to the same selected
schedule.

## Execution

### Required cutover sequence

1. Introduce the exact `CommittedGroupProfile` and validate all derived widths.
2. Remove semantic source identity from committed params, descriptors,
   canonical bytes, serialization, and verifier APIs.
3. Introduce the batch-level schedule selection and resolve rows by exact
   identity.
4. Refactor built-in dense and one-hot logic into source providers.
5. Move honest fold semantics behind the provider/planner interface while
   preserving current formulas.
6. Generalize `ProverOpeningData` to ordered group records and prepared
   whole-group operations.
7. Cut commit, final commit, prove, verify, claims, examples, benches, and test
   support to self-describing commitments.
8. Regenerate generated rows from exact profiles and preserve the curated mixed
   case.
9. Recompute setup envelopes from enabled exact rows.
10. Remove staging types and every legacy reconstruction/forwarding path.
11. Reconcile the latest #334 mirror additively without rewriting the pushed
    spec commit.
12. Run the repository gates and push logical implementation commits.

### Review checkpoints

Before implementation is considered complete, reviewers should separately
confirm:

1. the exact profile contains every A/B fact needed by commitment and
   verification but no provider semantics;
2. the schedule row contains every consuming fold/D/terminal fact not frozen
   by the standalone commitment;
3. source-provider code cannot influence verifier acceptance except by
   selecting an already validated profile and row;
4. distinct per-group challenges and the single shared nonce are preserved
   byte-for-byte;
5. no completeness correction or per-group nonce change entered this cut;
6. the new group-level prover abstraction retains monomorphized backend hot
   paths;
7. setup and deserialization checks precede allocation and indexing.

## Documentation

This spec is the active design record until implementation and review settle
the exact API names. During cutover:

- the source/commitment decisions in `multi-group-batching.md` remain marked
  superseded;
- `shared-opening-claims-api.md`,
  `schedule-catalog-ownership.md`,
  `planner-incidence-generalization.md`, and
  `typed-schedule-topology-cutover.md` MUST stop presenting `GroupSource` as a
  verifier-facing source of truth;
- `book/src/how/architecture.md` MUST document the durable provider/profile
  boundary after implementation;
- `book/src/how/verification.md` and `docs/verifier-contract.md` MUST document
  the exact profile and schedule-selection validation order;
- downstream usage documentation MUST include the provider registration and
  offline row-emission workflow;
- `AGENTS.md` MUST continue to point at the CI workflow rather than duplicate a
  final test command that can drift.

After the implementation ships and the book owns the durable API, this spec
should move to `implemented` and later follow `specs/PRUNING.md`.

## References

- [`multi-group-batching.md`](multi-group-batching.md)
- [`fold-linf-rejection.md`](fold-linf-rejection.md)
- [`tensor-challenge-prover-cutover.md`](tensor-challenge-prover-cutover.md)
- [`tail-wire-encoding.md`](tail-wire-encoding.md)
- [`schedule-catalog-ownership.md`](schedule-catalog-ownership.md)
- [`shared-opening-claims-api.md`](shared-opening-claims-api.md)
- [`typed-schedule-topology-cutover.md`](typed-schedule-topology-cutover.md)
- [`SPEC_REVIEW.md`](SPEC_REVIEW.md)
- [`../crates/akita-verifier/src/stages/stage1.rs`](../crates/akita-verifier/src/stages/stage1.rs)
- [`../crates/akita-challenges/src/fold_draw.rs`](../crates/akita-challenges/src/fold_draw.rs)
- [`../crates/akita-types/src/sis/fold_witness_grind.rs`](../crates/akita-types/src/sis/fold_witness_grind.rs)
- [`../crates/akita-types/src/sis/fold_linf_cap.rs`](../crates/akita-types/src/sis/fold_linf_cap.rs)
- [`../crates/akita-types/src/sis/norm_bound.rs`](../crates/akita-types/src/sis/norm_bound.rs)
