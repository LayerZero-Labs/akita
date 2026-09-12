# Spec: Homogeneous sources within each commitment group

| Field | Value |
|---|---|
| Created | 2026-09-10 |
| Status | archived |
| PR | [#20](https://github.com/LayerZero-Labs/akita/pull/20) |
| Book-chapter | book/src/usage/commitment-api.md |

## Decision

Remove `MultilinearPolynomial`, its views, implementations, exports, and tests
for mixing dense and one-hot polynomials within one commitment group. Each
group uses one source representation contract. Different groups in the same
opening batch may still use different representations, profiles, arities, and
local opening points.

For example, the following remains supported:

```text
group 0: [dense_a, dense_b]       -> commitment 0
group 1: [onehot_a, onehot_b]     -> commitment 1
group 2: [custom_a, custom_b]     -> commitment 2
                                  joint batched proof
```

A single group containing `[dense_a, onehot_a]` is no longer a supported Akita
input. Akita must not automatically split such a group: splitting changes the
commitment identities, ordered profiles, schedule lookup, and opening claims.
The application chooses its group boundaries before commitment.

## What the current implementation actually does

The removal is more than deleting an enum:

| Current component | Verified role | Proposed change |
|---|---|---|
| `backend/multilinear_polynomial/poly.rs` | Owns the dense/one-hot enum, single and batch views, metadata forwarding, and scans that collect homogeneous reference vectors | Delete the module and use concrete source types directly |
| `backend/multilinear_polynomial/ops.rs` | Dispatches opening and coefficient-packing kernels; handles genuinely mixed batches per polynomial | Delete these implementations; retain concrete dense and one-hot kernels |
| `compute/commitment/source.rs` | Implements the enum through a macro for four index widths; independently selects standard representations for each source | Delete the macro; select one standard representation contract for the group |
| `api/prepared_group.rs` | Stores one concrete `P` per prepared group | Keep typed homogeneous storage |
| `types/opening_data.rs` | Stores one generic group carrier `G` for the full batch; its polynomial convenience constructor uses the same `P` across groups | Add a public whole-group erasure boundary so groups may have different concrete source types |
| `protocol/core/root_group.rs` | Already implements operations over complete typed groups, but its traits are crate-private | Reuse this operation boundary for the erased carrier |

The existing `heterogeneous_group_types` test in
`crates/akita-pcs/tests/akita_fp128_e2e/heterogeneous.rs` wraps otherwise
homogeneous groups in the enum to fit the common prover-input type. Deleting
the wrapper without replacing that integration mechanism would regress a
requirement that this proposal explicitly preserves.

The current public `PreparedGroupProveOps` marker has a private supertrait.
It is not, by itself, an externally implementable whole-group dispatch API.
Telling callers to implement their own group enum is therefore insufficient.

## Proposed architecture

### Concrete sources inside a group

Commit ordinary dense groups directly from `DensePoly`, and one-hot groups
directly from `OneHotPoly`. Keep all supported one-hot index widths and the
existing short-norm and external source paths.

A homogeneous group means a common selected source representation contract,
not equal polynomial values or equal storage addresses. Standard one-hot groups
must agree on the selected chunk size and index width; short-norm groups must
agree on the selected representation parameters. Existing shape, arity,
extent, and configured source-bound checks continue to apply to every member.

Custom sources remain supported. A source can expose a standard representation
or provide an external inner operation. Akita does not inspect an external
backend's private storage or prohibit it from using an internal enum. Its
public group must satisfy one coherent advertised contract. We do not add a
global Rust `TypeId` registry or a closed list of application source types.

### Erase complete groups at the prover-input boundary

Provide an opaque public borrowed group carrier constructed from a typed
`PreparedProverGroup<P>`. Internally it erases complete group operations, while
the implementation for that typed group continues to invoke the existing
source-specific kernels. Different typed groups can then enter one ordered
vector of the same carrier type.

The intended use is schematically:

```text
dense_group  = prepare([&dense_a, &dense_b])
onehot_group = prepare([&onehot_a, &onehot_b])
groups       = [erase_group(dense_group), erase_group(onehot_group)]
opening_data = bind(claims, states, groups)
```

These are conceptual names, not additional promised wrapper functions. During
implementation, choose one canonical public constructor for prepared-group
inputs and share its alignment checks with the existing polynomial convenience
constructor. Do not add a parallel opening-data implementation.

The carrier has the following requirements:

- It borrows polynomial storage. Cloning a carrier copies a reference or shared
  ownership handle; it never clones polynomial coefficients or witnesses.
- Dynamic dispatch occurs for a whole-group operation. There is no enum match
  for each polynomial and no scan to rediscover whether a batch is dense.
- The existing generic `G` boundary can remain. An erased carrier is one `G`,
  so uniformly typed users need not pay for erasure.
- Opening and tensor capabilities remain distinct. Basic opening must not
  acquire tensor bounds merely because another proving flow needs them. Use
  capability-specific carrier forms/implementations at this boundary, matching
  the existing opening and tensor trait bounds; do not add panic stubs or a
  catch-all operation trait with unsupported methods.
- Backend context types remain tied to the selected prover stack. This cut
  does not promise unrelated opening backend types within one stack. Existing
  fold-level stack selection remains supported.
- External implementers supply their concrete source and the existing public
  source/kernel implementations. Akita constructs the erased group; callers
  need not implement crate-private protocol traits.
- Claims, source groups, retained states, and committed profiles stay aligned
  by their existing checked ordering. The carrier is prover-local and adds no
  representation tags to commitments, transcripts, or verifier inputs.

This boundary is necessary replacement functionality, not a second general
backend registry. Keep the protocol operation traits and result types private
where possible, and implement the bridge beside their canonical typed-group
implementations.

### Select the commitment representation once per group

Deleting the enum alone does not eliminate mixed source requests:
`compile_commitment_request` publicly accepts trait-object source references
and currently makes independent standard selections.

Change request compilation to the following sequence:

1. Validate every descriptor and the shared geometry without materialization.
2. Discover every source's standard and external capabilities.
3. Select the common external capability when every source advertises the same
   compatible operation identity, preserving current external precedence.
4. Otherwise select one exact standard `PolynomialType` from the intersection
   of all sources' offers and the backend's capabilities. Use the backend's
   existing preference order; for an operation accepting any standard type,
   use the first source's offer order to break ties deterministically.
5. Reject an empty intersection before representation materialization or
   backend arithmetic. Do not fall back to different selections per member.
6. Materialize each source using the shared selection and validate every
   result against its own descriptor and the checked plan.

Store the selected path once in the compiled request, alongside the ordered
source descriptors. Per-source descriptors, ordinals, and prepared external
inputs remain necessary. A uniform standard type does not imply that all
sources share one buffer, dense cache state, or cached digit-plane encoding.

Keep `AvailablePolynomialTypes`, `PolynomialRepresentation`, and the external
capability API: an adapter may still offer multiple ways to represent its data.
Their purpose is backend negotiation, which is independent of supporting mixed
polynomials within a group. A low-level request whose adapters expose a valid
common standard representation can use it; Akita need not inspect the adapters'
hidden original storage types.

Group-level selection allows backend batch dispatch to be hoisted where the
materialized representation supports it. Do not introduce a second batch
hierarchy solely to remove a small checked match inside an existing kernel.

## What becomes simpler, and what remains necessary

The enum and its two views disappear. Metadata forwarding, homogeneous scans,
temporary reference vectors, mixed coefficient-packing branches, and wrapper
kernel implementations disappear with them. Tests and examples use concrete
polynomials without wrapper construction or wrapper-induced cloning.

Commitment source compilation has one selected group path instead of a vector
of independent path decisions. Backends receive a stronger input contract.
The heterogeneous proving boundary becomes explicit at the same granularity
as commitments and public claims.

The following mechanisms must remain:

- `BatchDecomposeFoldOutcome::FallbackPerPoly`: dense kernels, recursive
  witnesses, setup-prefix sources, and unsupported one-hot batching cases use
  it independently of the removed enum.
- Per-polynomial shape and bound validation: a shared Rust type alone does not
  guarantee equal arities, valid one-hot indices, or correct source extents.
- Source-free public profiles and heterogeneous schedule selection: different
  groups still have different geometries and source contracts.
- Independent inner, outer, fused, and compression operations; prepared stage
  resources; backend-owned retained state; explicit exports; and immutable
  per-round routing. None exists solely to support mixed group members.
- Separate concrete dense, one-hot, short-norm, recursive, and external kernels.
  Removing a dispatch wrapper does not make their arithmetic interchangeable.

No change is proposed to verifier logic, mathematical commitment relations,
setup sizing rules, or proof serialization. For already homogeneous inputs,
the same profiles, setup, inputs, ordering, and randomness must produce the
same commitments and proof bytes.

## Implementation order

1. Introduce the public opaque whole-group carrier and canonical checked
   prepared-group input constructor. Prove through a public integration test
   that concrete dense and one-hot groups can be opened together without the
   polynomial enum. Cover the required opening/tensor capability boundaries.
2. Migrate heterogeneous end-to-end tests to that boundary, retaining group
   order, local points, profiles, independent evaluation oracles, and hints.
3. Migrate homogeneous callers directly to their concrete types. This includes
   the coefficient-packing tests and the small-field driver, whose current
   helper signature unnecessarily requires the enum.
4. Remove the wrapper directory, its module declarations and public exports,
   the four commitment-source macro implementations, and mixed-wrapper tests.
   Replace `fp128_mixed_batched_uses_source_free_group_geometry` with coverage
   of the new admission rule while retaining source-free group-profile tests.
5. Implement common group-level request selection. Add rejection and
   deterministic-selection tests before simplifying duplicate backend checks.
6. Update the Book, live backend docs, and the composable commitment spec.
   Run the repository's required validation and review the final symbol search.

## Acceptance and validation

- [x] No enum, view, alias, export, implementation, or production use of
  `MultilinearPolynomial` remains. Search both its Rust name and module name.
- [x] Concrete dense-only and one-hot-only commit/prove paths pass, including
  all supported index widths and the existing small-field coverage.
- [x] Heterogeneous groups pass joint prove/verify through public APIs with
  independent evaluation oracles; no per-polynomial sum type is used to bridge
  their types. Retain the existing heterogeneous tests' source-bound cases.
- [x] A custom source can enter that same public group boundary without
  implementing private traits or materializing its data as a built-in source.
- [x] Low-level dense/one-hot requests with no common representation fail
  before materialization; instrumented sources establish that ordering.
- [x] Common representation selection handles differing offer order, an
  incompatible external capability, external-only groups, and no common path.
- [x] Existing fused/split parity, resident-state ownership, absent-export,
  schedule cutover, and external-backend tests continue to pass.
- [x] Existing coefficient-packing and extension-opening tests pass; optional
  tensor support is not required for unrelated basic opening callers.
- [x] Homogeneous deterministic fixtures preserve commitments and proof bytes;
  group erasure introduces no witness-sized allocation or clone.
- [x] Run the cheap CI preflight, all three mandated Clippy configurations,
  the current CI test-pass commands, and path-triggered workflows from
  `.github/workflows/`. Run documentation guardrails for this spec.

No numerical speedup is claimed. The expected gain is reduced dispatch code
and smaller maintenance surface. Measure any performance claim separately;
the source-specific arithmetic kernels remain the same.

## Documentation and alternatives

Update `book/src/usage/commitment-api.md`, relevant architecture/proving pages,
`docs/compute-backend-inventory.md`, and
`specs/composable-commitment-execution.md`. Clarify the group boundary in
`specs/heterogeneous-group-source-contracts.md` without changing its source-free
protocol contract. Correct the prepared-group documentation that currently
suggests an application polynomial enum.

Historical archived specifications may retain the old name as history; it must
not survive as an API in source, tests, examples, or live usage documentation.
When implementing the deletion, update deleted-symbol guardrails and fold or
archive this removal record consistently with the documentation policy. This
proposal necessarily names the existing symbols to identify the removal.

A replacement dense/one-hot polynomial enum would preserve the complexity being
removed. A closed dense/one-hot group enum would exclude external source types.
Publicly exposing all root protocol traits would expand the external API more
than necessary. An opaque carrier over existing whole-group implementations
preserves extensibility with a small, intentional type-erasure boundary.
