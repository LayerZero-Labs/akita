# Spec: Opaque Chunk-Aware Fold Responses

| Field | Value |
|---|---|
| Author(s) | |
| Created | 2026-09-16 |
| Status | proposed |
| PR | |
| Supersedes | Chunk-aware batched decompose-fold proposal; opaque fold-response proposal |
| Superseded-by | |
| Book-chapter | |

## Summary

Move every intermediate and terminal folded response `z`, including every
chunk response `z_i`, behind the opening-compute boundary. Akita continues to
draw challenges and own the canonical dyadic partition, acceptance policy,
relation layout, and recursive-witness layout. The selected opening backend
consumes one Akita-validated, context-bound plan, computes and checks the
responses, and returns an opaque accepted-fold handle whose immutable manifest
records the accepted geometry and operation-context identity. Later operations
derive the public A-side relation contribution, emit the selected chunk's
recursive Z digits, or encode the terminal Z payload through that same handle
without exposing centered response coefficients to protocol code.

This combines chunk-aware batch dispatch with opaque response ownership. The
chunk-aware plan from the former design remains; its public
`Chunked(global, chunks)` witness result does not. A backend may retain CPU
vectors, shard-local buffers, device allocations, a computation graph, or a
recomputation recipe. Akita protocol code cannot inspect or depend on that
representation.

For recursive setup offloading, the ownership boundary is asymmetric. The
stage-3 sumcheck is setup-only, so Akita may continue to own and fold the public
setup-prefix polynomial. The recursive witness group used by the following
suffix opening remains consumer-owned. When that opening requires an
extension-opening reduction (EOR), the consumer supplies only its public
opening data, per-round witness polynomial contribution, and final opening
claims. Akita combines those messages with its locally computed setup-prefix
contribution and owns the single transcript challenge sequence. The setup and
witness sources never share a raw-source carrier.

## Intent

### Goal

Give consumers one chunk-aware fold operation while making the opening backend
the sole owner of `z` and `z_i` for their complete lifecycle: computation,
admission, A-side relation arithmetic, recursive Z emission, and terminal Z
encoding.

### Ownership boundary

Akita owns:

- challenge sampling and transcript domains;
- the full ordinary-batch challenge order;
- `dyadic_block_ranges` and the exact block-to-chunk assignment;
- digit bounds, optional L2 caps, terminal coefficient and wire caps;
- the expected number of rows, ring dimension, chunks, Z planes, and output
  ranges;
- grind orchestration and the winning nonce;
- combination of public relation outputs with non-Z relation terms;
- the public setup-prefix polynomial and its local stage-3/setup-opening
  arithmetic;
- aggregation of setup-prefix and consumer-provided sumcheck round
  polynomials, and the one transcript challenge sampled from each aggregate;
- recursive witness and proof serialization layouts.

The opening backend or external consumer owns:

- how the full challenge vector is applied to the source;
- how each supplied chunk range is computed;
- storage and representation of `z` and `z_i`;
- independent per-chunk extrema checks;
- checked L2 accumulation when requested;
- construction of the global response from the chunk responses;
- A-side quotient and Z-dependent consistency arithmetic;
- decomposition and emission of each accepted `z_i` into recursive Z digits;
- terminal Z admission and canonical terminal Z encoding;
- all tensor projections, packed witness tables, EOR folding state, and
  per-round sumcheck polynomial contributions derived from a recursive witness;
- ingestion of the exact EOR challenges returned by Akita and production of
  the witness group's final public opening claims.

The opening backend is trusted for honest-prover computation and admission.
Proof verification checks the algebraic artifacts and verifier-visible response
bounds, but does not certify backend diagnostics, prove that every earlier
candidate nonce was rejected, or otherwise audit an opaque implementation's
control flow. Optional diagnostics are never security inputs.

The consumer does not choose a different semantic partition. Akita supplies the
canonical ranges. The consumer chooses only the computation and storage
strategy for that assignment.

### Mathematical contract

For the canonical ranges `I_0, ..., I_(P-1)`, sparse block challenge `c_b`, and
witness block `s_b`, the accepted handle represents

```text
z_i = sum_{b in I_i} c_b s_b
z   = sum_i z_i
```

coefficient-wise. For a typed batch, every range is relative to one claim's
local block coordinates and applies identically to every claim:

```text
blocks_per_claim = challenges.num_live_blocks_per_claim()
claim p, range r =>
    p * blocks_per_claim + r.start
    .. p * blocks_per_claim + r.end
```

Ranges are never offsets into the full concatenated batch. Empty ranges denote
full-width all-zero chunk responses. A backend must admit each `z_i`
independently before aggregation; cancellation in `z` cannot admit an
out-of-range `z_i`.

### Invariants

1. One fold probe makes exactly one chunk-aware backend call per group and
   candidate nonce.
2. Full, unwindowed challenges cross the compute boundary. Protocol code never
   constructs one challenge object per chunk.
3. `Sparse` and `SparseChunked` are distinct plan variants so a backend cannot
   silently ignore chunk geometry.
4. Every in-tree opening backend supports both variants. There is no
   capability, `Unsupported`, or fallback result.
5. `z` and `z_i` never cross back into protocol code as coefficient buffers,
   field rows, concrete witnesses, debug views, or downcastable values.
6. Relation derivation, Z emission, and terminal encoding use the same handle
   that passed admission. They do not recompute from a protocol-owned copy.
7. For chunked folds, the handle represents `z = sum_i z_i` exactly. Checked
   integer accumulation rejects overflow.
8. Empty canonical ranges remain addressable chunks and emit full-width zero Z
   planes.
9. Single-chunk and multi-chunk proof bytes, transcript events, accepted
   nonces, verifier behavior, and schedule geometry remain unchanged.
10. Opening backend `O` owns the accepted handle even when the ring-switch
    backend `R` differs. This design must not require `O == R`.
11. Malformed plans, layout mismatches, missing chunks, arithmetic overflow,
    and sink-range mismatches return `AkitaError`; verifier-reachable code does
    not panic.
12. Backend correctness includes checking every requested chunk and deriving
    every later artifact from the accepted response. Akita cannot independently
    audit opaque coefficients. Algebraically incorrect or verifier-inadmissible
    proof artifacts remain detectable by proof verification. False rejection,
    fabricated diagnostics, nondeterministic admission, and use of a stricter
    policy are honest-prover backend bugs and are not claimed to be detected by
    the verifier.
13. An accepted handle is bound to the backend, prepared setup, field/ring
    dimension, opening family, group geometry, chunk ranges, and acceptance
    policy used by its probe. It cannot be used with another operation context.
14. Z emission is range-confined, complete, non-overlapping, and atomic: an
    error cannot partially modify the recursive witness.
15. Terminal Z bytes are accepted by Akita only after canonical decoding against
    the schedule-owned coordinate count, coefficient policy, coding parameters,
    and maximum payload budget. Validation may decode into a private temporary
    that is immediately discarded; it does not expose coefficients to protocol
    orchestration.
16. Direct setup-contribution mode does not run stage 3 and introduces no
    witness round-contribution calls.
17. Recursive stage 3 remains a setup-only sumcheck. No witness handle, witness
    evaluation table, or witness round polynomial is passed to
    `AkitaStage3Prover`.
18. The setup-prefix polynomial may remain concrete and prover-owned. This
    design does not require a `SetupContributionSumcheckKernel` or an opaque
    setup-prefix handle.
19. In the following recursive suffix opening, setup-prefix and recursive
    witness groups have distinct owners. They are never erased into a common
    carrier that permits Akita to inspect the witness source.
20. For a shared EOR round, Akita absorbs exactly one aggregate polynomial:
    the coefficient-wise sum of its local setup-prefix contribution and every
    consumer-owned witness contribution. It then sends the same sampled
    challenge to every contributor.
21. A consumer-owned EOR session is transcript-linear and single-use. Round
    order, polynomial degree, previous local claim, challenge count, final
    claim count, point geometry, and operation-context identity are validated.
22. Splitting EOR by owner does not change the aggregate polynomial, transcript
    events, EOR proof, stage-3 proof, setup-prefix opening, or verifier logic.

### Non-goals

- allowing consumers to redefine Akita's canonical chunk partition;
- exposing backend capabilities or a protocol-visible fallback path;
- adding a second chunk-specific probe method;
- requiring a native one-pass implementation from every backend;
- requiring one concrete response representation;
- changing challenge sampling, subring embedding, proof layout, serialization,
  planner sizing, or verifier logic;
- moving the public setup-prefix polynomial or setup-only stage-3 arithmetic
  out of `akita-prover`;
- adding a witness term to the current setup-only stage-3 identity;
- forcing the opening and ring-switch backends to be the same type;
- providing `Any`, downcasts, coefficient accessors, centering hooks, or debug
  escape hatches on an accepted handle.

## Design

### Validated probe plans and outcomes

Backends receive a public read-only plan whose fields and constructors are
private to Akita. The plan retains the typed `Challenges` carrier instead of a
raw sparse-challenge slice plus a second, independently supplied polynomial
count. This preserves the already-validated claim-major batch geometry.

```rust
pub struct ValidatedFoldProbePlan<'a> {
    challenges: &'a Challenges,
    geometry: FoldProbeGeometry<'a>,
    num_positions_per_block: usize,
    num_digits: usize,
    log_basis: u32,
    opening_family: OpeningFamily,
    acceptance: ValidatedFoldAcceptancePlan,
}

pub enum FoldProbeGeometry<'a> {
    Sparse,
    SparseChunked {
        /// Exact output of `dyadic_block_ranges` for one claim.
        chunk_ranges: &'a [Range<usize>],
    },
}

pub struct ValidatedFoldAcceptancePlan {
    digit_negative_abs_bound: u128,
    digit_positive_bound: u128,
    response_l2_sq_cap: Option<u128>,
    collect_l2_diagnostic: bool,
}

pub struct FoldProbeDiagnostics {
    pub observed_l2_sq: Option<u128>,
}

pub enum FoldProbeOutcome<H> {
    Rejected,
    Accepted {
        fold: H,
        diagnostics: FoldProbeDiagnostics,
    },
}
```

The plan exposes getters but no public unchecked constructor. A backend may
inspect every policy input, but cannot reinterpret the challenge batch or
mutate the geometry. `FoldProbeDiagnostics` is non-authoritative metadata, not
a response view or security input. Unrequested values must be `None`.
Response coefficient counts are derived by Akita from validated geometry and
are not reported by the backend.

The statically dispatched kernel is:

```rust
pub trait FoldResponseKernel<S, F, const D: usize>:
    ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
{
    /// Owned response state. It must not borrow `source`, `prepared`, or `plan`.
    type AcceptedFold: Send + Sync + 'static;

    fn probe(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        plan: &ValidatedFoldProbePlan<'_>,
    ) -> Result<FoldProbeOutcome<Self::AcceptedFold>, AkitaError>;
}
```

There is no `Global` or `Chunked` result-shape outcome. The validated plan fixes
the handle's semantic shape. The private adapter stores an immutable manifest
beside the backend handle and validates every later operation against it.

Accepted handles are owned. A device implementation may retain an owned
allocation, event, graph, or reference-counted resource lease, but it may not
borrow the source or prepared setup. `Drop` must release or cancel retained
resources, including handles discarded when another group rejects the same
candidate nonce. A recomputation recipe must own all state it needs and be
deterministic for the accepted manifest.

### Plan validation

Akita constructs the plan only after common validation requires:

- a nonempty `Challenges` value with nonzero claim and live-block counts;
- exact agreement between `Challenges::num_claims()` and the source batch;
- exact agreement between `Challenges::num_live_blocks_per_claim()` and the
  group/schedule geometry;
- validation of every contained `SparseChallenge` for `D`;
- nonzero positions, digit counts, and response dimensions;
- checked range and coefficient-count arithmetic;
- digit bounds and L2 policy derived from the same schedule and security
  primitives enforced by the verifier, not arbitrary caller-supplied values.

`SparseChunked` additionally requires:

- at least two ranges;
- a power-of-two range count not exceeding `MAX_WITNESS_CHUNKS`;
- exact equality with
  `dyadic_block_ranges(challenges.num_live_blocks_per_claim(), ranges.len())`.

Empty ranges are valid. Backends consume supplied ranges and must not derive a
different partition from a chunk count. A malformed plan cannot be constructed
through the public API; defensive backend validation may still reject a plan
with `AkitaError` but must not reinterpret it.

For integer response arithmetic, every `z_i` uses the same canonical centered
coefficient domain as the current prover. Global `z` is formed by checked
coefficient-wise integer addition in that domain before conversion to field
rows. Overflow is `Err(AkitaError)`, never `Rejected` and never modular wrap.

The L2 definition is normative:

```text
Sparse:        observed_l2_sq = ||z||_2^2
SparseChunked: observed_l2_sq = sum_i ||z_i||_2^2
```

The chunked coefficient count is `num_chunks * live_response_coefficient_count`;
empty chunks contribute a full-width all-zero vector. Squaring and accumulation
use checked `u128`. Arithmetic overflow returns `AkitaError`. `Rejected` is
reserved for a correctly computed response that exceeds an admission bound or
payload budget. These rules preserve the current winning nonce.

### Opaque handle lifecycle

The handle has two intermediate-fold operations:

```rust
pub trait FoldRelationKernel<H, F, const D: usize>:
    ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
{
    fn a_relation_from_fold(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        fold: &H,
        plan: &ValidatedFoldRelationPlan<'_, F>,
    ) -> Result<FoldRelationOutput<F>, AkitaError>;
}

pub trait FoldWitnessEmitKernel<H, F, const D: usize>:
    ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
{
    fn emit_z_unit(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        fold: &H,
        plan: &ValidatedEmitZUnitPlan,
        out: &mut dyn ScopedWitnessCoefficientSink,
    ) -> Result<(), AkitaError>;
}
```

`ValidatedFoldRelationPlan` contains only new public inputs required for A-side
arithmetic. Geometry already fixed at probe time is not supplied again. Its
private constructor checks the opening family against the accepted manifest.
The output is family-typed rather than using an optional field:

```rust
pub enum FoldRelationOutput<F> {
    EvaluationTrace {
        a_quotients: Vec<RelationQuotientRow<F>>,
        z_consistency_high_half: RelationQuotientRow<F>,
    },
    SubringCoefficientPacking {
        a_quotients: Vec<RelationQuotientRow<F>>,
    },
}
```

The adapter validates the exact row count, row geometry, ring dimension, and
variant before returning it to protocol code. Protocol code combines these
public outputs with the E-side and other non-Z terms.

`ValidatedEmitZUnitPlan` is privately constructed from the accepted manifest
and the selected `WitnessUnitLayout`. It contains:

- chunk index;
- exact Z output range or unit layout;
- live coefficient length and ring dimension;
- positions per block;
- inner and fold digit counts;
- opening log basis.

The backend does not receive the underlying absolute-address sink. Akita wraps
it in a staging sink with the following public interface:

```rust
pub trait ScopedWitnessCoefficientSink {
    fn len(&self) -> usize;
    fn write_coefficients(
        &mut self,
        relative_start: usize,
        coefficients: &[i8],
    ) -> Result<(), AkitaError>;
}
```

The wrapper accepts only relative writes within the exact Z range, rejects
overlap, and records coverage. Backend success is accepted only if every
coefficient was written exactly once. The wrapper stages bytes privately and
commits them to `WitnessCoefficientSink` only after the kernel and coverage
checks succeed, so an error leaves the recursive witness unchanged. Private
packed-writer types do not appear in public traits.

### Heterogeneous group erasure

External source/backend pairs keep a strongly typed associated handle. Akita's
heterogeneous group carrier uses a private object-safe bridge:

```rust
pub(crate) struct AcceptedFold<F, O> {
    manifest: AcceptedFoldManifest,
    context: OperationContextIdentity,
    inner: Box<dyn AcceptedFoldOps<F, O>>,
}

trait AcceptedFoldOps<F, O>: Send + Sync {
    fn a_relation_from_fold(
        &self,
        ctx: &OperationCtx<'_, F, O>,
        plan: &ValidatedFoldRelationPlan<'_, F>,
    ) -> Result<ErasedFoldRelationOutput<F>, AkitaError>;

    fn emit_z_unit(
        &self,
        ctx: &OperationCtx<'_, F, O>,
        plan: &ValidatedEmitZUnitPlan,
        out: &mut dyn ScopedWitnessCoefficientSink,
    ) -> Result<(), AkitaError>;
}
```

A generic adapter owns the source-specific associated handle and invokes the
static kernels. `OperationContextIdentity` is privately derived from the
backend instance, validated prepared setup, expanded setup identity, and field
and ring dispatch. Before every operation the adapter requires an exact context
match and validates the new plan against `AcceptedFoldManifest`. A different
instance of the same backend type is not sufficient. The bridge exposes no
`Any`, downcast, coefficient iterator, centered representation, extrema, or
inf-norm accessor.

The opening backend `O` performs probe, A-side fold relation work, and Z
emission. The ring-switch backend `R` continues to own unrelated D/B/E/T work.
`ring_switch_build_w` therefore receives both contexts where necessary.
This is an intentional capability migration: all opening backends must
implement A-side quotient arithmetic formerly routed through `R`. Shared
low-level quotient primitives may be reused, but the cutover must not duplicate
the arithmetic or introduce pass-through wrappers. Benchmarks must measure the
loss of any currently fused A/D/B path.

### CPU reference handle

The initial CPU implementation may use:

```rust
struct CpuAcceptedFold<F: Field> {
    global: DecomposeFoldWitness<F>,
    chunks: CpuFoldChunkStorage,
}
```

This is private compute state, not a protocol or public-kernel type. CPU probe
moves existing challenge windowing, centered extrema checks, checked L2
accumulation, and global aggregation behind the compute boundary.

Internal backends may share one private helper that windows a flat batch
challenge vector:

```rust
pub(crate) fn window_batch_sparse_challenges(
    challenges: &Challenges,
    range: Range<usize>,
) -> Result<Challenges, AkitaError>;
```

It preserves `Challenges` batch metadata; entries outside the local range
become canonical empty challenges. Prefer a shared higher-level private helper
if it eliminates repeated per-range/per-polynomial loops. Neither helper is a
fallback outcome.

The ordinary batch-polynomial and chunk-global aggregation paths must share one
checked coefficient accumulation primitive. Do not duplicate sum or field-row
construction logic.

### Backend requirements

- Dense: retain the current single-fold path; for chunked probes, compute and
  aggregate each supplied range privately.
- Generic one-hot: retain fused batching where applicable; initially reuse its
  existing one-hot machinery once per range.
- Recursive setup-prefix: Akita may retain the concrete public source and run
  its local opening and EOR contribution.
- Recursive witness: the consumer restricts each polynomial's local challenge
  slice to each supplied range, retains the resulting private chunk and EOR
  state, and exposes only protocol messages through the opaque kernels.
- Delegating CPU backends: forward plans and opaque results without
  interpretation.
- Grouped/external dispatch: forward `SparseChunked` unchanged to the selected
  representation and erase only the accepted handle.
- Optimized consumers: may perform one packed traversal and retain shard or
  device-local state without constructing CPU witness containers.

Every implementation must derive relation rows and emitted digits from the
same admitted handle.

### Stage 3 and setup-prefix opening ownership

There are two different operations around the recursive setup contribution and
they must not be conflated.

In `Direct` setup-contribution mode there is no stage-3 sumcheck. In
`Recursive` mode, `AkitaStage3Prover` proves only a product involving the
public expanded setup prefix and public factors derived from the relation and
earlier transcript challenges. It does not consume `w`, `z_i`, `e_i`,
`t_i_hat`, or another secret witness table. Akita therefore continues to own:

- `RectangularSetupProductTerm` and the concrete setup-prefix polynomial;
- stage-3 round-polynomial computation and challenge ingestion; and
- the carried `(setup_prefix_point, setup_prefix_eval)` public opening claim.

No new setup sumcheck kernel is required under this ownership policy. Moving
stage 3 itself would be an optional compute-placement optimization, not a
witness-confidentiality requirement.

The confidentiality boundary is the *next recursive suffix opening*. Today
that path represents both the setup prefix and recursive witness as
`RecursiveFoldSource`, constructs one `logical_groups` collection, and lets the
same tensor/EOR machinery see both raw sources. Replace that arrangement with
owner-separated contributors:

```text
Akita-owned setup prefix                 consumer-owned recursive witness
        |                                             |
 local opening/EOR contribution          opaque witness EOR session
        |                                (public messages only)
        +---------------------+-----------------------+
                              |
                  Akita sumcheck coordinator
                  - sums round polynomials
                  - absorbs one aggregate polynomial
                  - samples one challenge
                  - returns it to both contributors
                              |
                    unchanged EOR proof and
                    recursive suffix opening
```

The consumer-facing boundary needs an interactive witness EOR kernel. Names
below are illustrative; the implementation should keep one canonical API per
concept and avoid a wrapper for each recursive level:

```rust
pub trait WitnessExtensionOpeningKernel<H, F, E>: ComputeBackendSetup<F> {
    type Session: Send;

    fn prepare_witness_opening(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        witness: &H,
        plan: &ValidatedWitnessOpeningPlan<'_, E>,
    ) -> Result<PreparedWitnessOpening<E>, AkitaError>;

    fn begin_witness_eor(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        witness: &H,
        opening: &PreparedWitnessOpening<E>,
        plan: &ValidatedWitnessEorPlan<'_, E>,
    ) -> Result<Self::Session, AkitaError>;

    fn witness_round_polynomial(
        &self,
        session: &mut Self::Session,
        round: usize,
        previous_local_claim: E,
    ) -> Result<UniPoly<E>, AkitaError>;

    fn bind_witness_eor_challenge(
        &self,
        session: &mut Self::Session,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError>;

    fn finish_witness_eor(
        &self,
        session: Self::Session,
    ) -> Result<WitnessEorFinalClaims<E>, AkitaError>;
}
```

`PreparedWitnessOpening` contains only verifier-visible messages already
present in the proof flow: opening evaluations and serialized EOR partials. It
must not contain full tensor evaluation tables, private row-folding tables,
packed witness values, a `RecursiveWitnessFlat`, a
`RecursiveFoldSource::Witness`, or another borrow of the witness. The compact
EOR partials already serialized into the proof remain public. The session owns
or references all other folding state inside the consumer.

The validated plans bind the session to the backend instance, prepared setup,
witness handle, group index, point, ring dimension, polynomial count, split
geometry, sampled `eta`, claim coefficients, cylindrical padding, number of
rounds, and degree bound. Fields are private and are constructed only after
Akita has validated the schedule and opening layout.

For each round, Akita obtains one degree-two polynomial from the witness
session and computes the setup-prefix group's degree-two polynomial locally.
It validates the contribution shapes and local claim transitions, sums the
polynomials coefficient-wise, checks the aggregate Boolean sum against the
current aggregate claim, absorbs that aggregate using the existing labels,
samples the existing EOR challenge, and binds the challenge into both states.
The witness session's next local claim is its returned polynomial evaluated at
that challenge; the setup local claim is updated identically. Akita never
needs the witness evaluation table to perform this coordination.

Equivalently, for round `j`, local claims `C_setup,j` and `C_witness,j`, and
local round polynomials `p_setup,j` and `p_witness,j`:

```text
C_j = C_setup,j + C_witness,j
p_j(X) = p_setup,j(X) + p_witness,j(X)
p_j(0) + p_j(1) = C_j
C_setup,j+1 = p_setup,j(r_j)
C_witness,j+1 = p_witness,j(r_j)
C_j+1 = p_j(r_j)
```

The transcript sees only `p_j` and samples `r_j` exactly as before. Additional
consumer-owned witness groups extend the sums; they do not add transcript
rounds or challenges.

At completion, the consumer returns only the witness group's final scalar
claims in canonical claim order. Akita computes the transparent factors and
setup-prefix final claims locally, checks that all local final claims sum to
the aggregate final claim, and appends the final claims in the existing order.
The existing proof representation remains unchanged.

The same owner split applies when the suffix opening bypasses EOR: setup-prefix
opening work remains local, while every witness-dependent evaluation,
coefficient-packing operation, fold probe, relation operation, and recursive
witness emission is invoked through the consumer's opaque handle. There is no
fallback that reconstructs a witness source in protocol code.

Concrete recursive sources remain backend-private. Setup prefixes use ordinary
opaque commitment handles after backend preparation or validated artifact
import. Aggregate EOR coordination exposes only its session and scheduled
protocol messages.

### Grind orchestration

For each candidate nonce and group, protocol code:

1. selects point challenges once;
2. creates the opening batch once;
3. derives `dyadic_block_ranges` once when `num_chunks > 1`;
4. constructs one `ValidatedFoldProbePlan` with `Sparse` or `SparseChunked`
   geometry and the full typed challenge batch;
5. calls `probe` once;
6. rejects the nonce on `Rejected`;
7. retains only the opaque handle, original challenges, and diagnostics on
   `Accepted`.

All groups share the nonce and must accept jointly as today. After committing
the winning nonce, live transcript replay redraws and compares the challenges.
It does not probe again. The accepted preview handle is retained for subsequent
operations, so a backend must keep it valid across nonce commitment. This
preserves one fold computation per candidate and the existing transcript
semantics.

When any group rejects or errors, every handle already accepted for that
candidate is dropped before the next candidate begins. Backends must not retain
candidate-global mutable state outside the returned handle.

`FoldProbeOutput` becomes conceptually:

```rust
struct FoldProbeOutput<F, O> {
    fold: AcceptedFold<F, O>,
    challenges: GroupFoldChallenges,
    diagnostics: FoldProbeDiagnostics,
}
```

### Relation construction

`RingRelationGroupWitness` retains the opaque handle instead of
`z_folded_rings` and `z_folded_coefficients`. A-side construction asks the
handle for:

- A quotient rows; and
- the Z-dependent consistency high-half term when the opening family requires
  it.

`RingSwitchRelationView` no longer carries `z_segment` or
`z_folded_centered_inf_norm`. It may remain for D/B inputs or split into
narrower inputs. The protocol never reconstructs a centered Z buffer.

### Recursive Z emission

For every witness unit, protocol code sends the unit's validated chunk index
and exact Z layout to `emit_z_unit`. The handle writes packed signed digits to
the range-confined staging sink. Akita checks exact coverage and commits the
staged range to `WitnessCoefficientSink` only on success. E and T emission
remain unchanged.

An empty assigned range emits the same full-width zero Z segment required by
the existing witness layout. A unit cannot request a nonexistent chunk or write
outside its exact range.

### Terminal folds

The opaque boundary includes terminal folding. Terminal protocol code must not
read `DecomposeFoldWitness::centered_coeffs_flat()`.

Add a privately constructed validated terminal probe plan containing the typed
sampled challenges, exact response shape, coefficient policy, optional L2 cap,
Golomb-Rice parameters, and maximum payload budget. A terminal probe returns
`Rejected` or an opaque
`AcceptedTerminalFold`. A terminal emission operation returns or writes the
candidate encoded Z bytes, not raw centered coefficients:

```rust
pub trait TerminalFoldResponseKernel<S, F, const D: usize>:
    ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
{
    type AcceptedTerminalFold: Send + Sync + 'static;

    fn probe_terminal(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        plan: &ValidatedTerminalFoldProbePlan<'_>,
    ) -> Result<FoldProbeOutcome<Self::AcceptedTerminalFold>, AkitaError>;

    fn encode_terminal_z(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        fold: &Self::AcceptedTerminalFold,
        plan: &ValidatedTerminalZEncodingPlan,
    ) -> Result<Vec<u8>, AkitaError>;
}
```

The private terminal adapter binds the handle to the same operation-context
identity and immutable manifest used by the probe. After `encode_terminal_z`,
Akita constructs a sealed `CanonicalTerminalZPayload` by running the existing
canonical decoder against the manifest's exact coordinate count, coefficient
policy, Rice parameter, and maximum byte budget. Canonical decoding rejects
trailing data and noncanonical padding. Any decoded temporary is discarded
inside the adapter and is never returned to protocol orchestration.

`build_terminal_response_from_payload` is narrowed to accept only
`CanonicalTerminalZPayload` plus the protocol-owned E and T fields. It validates
the exact E/T field lengths and requires `payload.len() <= z_payload_bytes`;
the terminal Z payload is variable length and is not padded to the scheduled
maximum. This keeps raw terminal `z` opaque while preserving the verifier's
existing wire format and admission checks.

Terminal live replay follows the same rule as intermediate replay: compare
redrawn challenges and retain the accepted preview handle without a second
probe.

### Removed protocol-owned carriers

After cutover, remove from protocol-facing code:

- the chunk loop in `fold_probe_witness_kernel`;
- protocol use of `window_sparse_challenges`;
- `CenteredFoldChunk` and `FoldChunkCoefficients`;
- `z_folded_rings`, `z_folded_coefficients`, `z_centered`, and `z_inf`;
- protocol imports of ordinary witness aggregation;
- direct coefficient extrema and L2 iteration;
- `emit_unit_z_segment`'s direct decomposition path;
- direct terminal centered-coefficient inspection;
- `RecursiveFoldSource::Witness` and protocol construction of
  `RecursiveWitnessFlat` for suffix opening/EOR work;
- a shared raw `logical_groups` vector containing both setup-prefix and
  recursive-witness sources; and
- protocol-owned packed-witness and EOR fold tables for consumer-owned groups.

`DecomposeFoldWitness` and `decompose_fold_batch` may remain only as private or
low-level compute implementation details. They are not exported as the
chunk-aware protocol contract. Do not add compatibility wrappers.

## Evaluation

### Acceptance criteria

- [ ] Intermediate and terminal protocol code has no access to raw `z` or
      `z_i` coefficients or field rows.
- [ ] One public `FoldResponseKernel::probe` entry point accepts both `Sparse`
      and `SparseChunked`; no capability or fallback result exists.
- [ ] Probe plans have no public unchecked constructor, retain typed challenge
      geometry, and require exact canonical dyadic ranges.
- [ ] Every in-tree opening backend supports both variants.
- [ ] A multi-chunk candidate sends full challenges and exact
      `dyadic_block_ranges` in one backend call.
- [ ] The backend independently admits every chunk and checked-sums the global
      response.
- [ ] A-side relation rows and consistency terms are derived through the
      accepted handle.
- [ ] Recursive Z digits are emitted from the accepted handle through a scoped,
      exact-coverage staging sink and committed atomically.
- [ ] Terminal Z is admitted and encoded behind the compute boundary, then
      canonically validated by Akita without exposing decoded coefficients to
      protocol orchestration.
- [ ] A handle is rejected when used with a different backend instance,
      prepared setup, ring dispatch, opening family, or response geometry.
- [ ] Heterogeneous erased groups carry distinct source-specific handles
      without `Any` or downcasting.
- [ ] Separate opening and ring-switch backends remain supported.
- [ ] Live replay does not perform a second fold computation.
- [ ] Single- and multi-chunk proofs remain byte-identical to the pre-cutover
      implementation.
- [ ] Verifier code, proof layout, schedule geometry, and serialization format
      do not change.
- [ ] Direct setup-contribution mode makes no stage-3 or witness-EOR calls.
- [ ] Recursive stage 3 remains setup-only and byte-identical; Akita retains
      the concrete setup-prefix polynomial and no setup sumcheck kernel is
      required.
- [ ] Recursive suffix code does not construct
      `RecursiveFoldSource::Witness`, expose `RecursiveWitnessFlat`, or put a
      witness source beside the setup source in `logical_groups`.
- [ ] The consumer prepares recursive-witness opening messages and computes
      every witness-derived EOR round polynomial without returning its packed
      witness, full tensor tables, private row-folding tables, or folded state;
      only the compact partials already present in the public proof may cross.
- [ ] Every EOR round absorbs the same aggregate polynomial as the pre-cutover
      implementation, then sends the exact sampled challenge to the local
      setup contributor and the consumer session.
- [ ] Consumer EOR sessions reject wrong order, repeated or skipped rounds,
      wrong prior local claims, wrong degree, context mismatch, early finish,
      and wrong final-claim cardinality with `AkitaError`.
- [ ] Mixed setup-prefix and recursive-witness suffix proofs, final claims,
      protocol points, transcript events, and verifier results remain
      byte-identical.

### Testing strategy

Plan validation tests cover:

- valid single, two, four, and eight chunk plans;
- uneven and over-partitioned layouts;
- empty ranges;
- gaps, overlaps, unordered ranges, wrong endpoints, and fewer than two
  chunked ranges;
- non-power-of-two or over-cap chunk counts and contiguous but noncanonical
  partitions;
- empty batches/challenges, malformed sparse challenges, source/challenge claim
  count disagreement, and live-block disagreement;
- checked dimension, range, and coefficient-count overflow.

Probe and admission tests cover:

- exact global coefficient and field-row equality to `sum_i z_i` inside the CPU
  reference implementation;
- equal full widths and full-width zeros for empty ranges;
- per-chunk extrema derived from coefficients;
- checked signed accumulation and L2 overflow, with overflow returning
  `AkitaError` rather than `Rejected`;
- `||z||^2` for sparse folds and `sum_i ||z_i||^2` for chunked folds;
- the cancellation regression where an out-of-range `z_0` is cancelled by
  `z_1` but the probe still rejects;
- requested and unrequested diagnostic metadata.

Protocol-dispatch tests use a recording backend to verify:

- one `Sparse` call for a single-chunk candidate;
- one `SparseChunked` call for a multi-chunk candidate;
- unchanged full challenges and exact canonical ranges;
- no protocol challenge windows or additional probe calls;
- one retained handle reused after live replay;
- rejected-candidate handles are dropped before the next candidate;
- handles cannot be reused with a different validated operation context.

Stage-3 and recursive-suffix ownership tests cover:

- direct mode never constructs a stage-3 prover or witness EOR session;
- recursive stage 3 receives only public setup/relation inputs and produces the
  same setup-prefix point, evaluation, sumcheck proof, and transcript events;
- a recording consumer sees the validated witness opening/EOR plans and the
  exact challenge sequence, while Akita never receives its source or private
  folding tables;
- each consumer round polynomial satisfies its local Boolean-sum transition,
  and the sum of setup and witness contributions equals the old monolithic EOR
  polynomial coefficient-for-coefficient;
- the setup and witness local claims evaluate under each sampled challenge to
  the next aggregate claim;
- final setup and witness claims appear in the existing canonical order and
  reproduce the old aggregate final claim;
- mixed setup-prefix/witness groups with unequal native arity preserve current
  cylindrical padding behavior;
- base-field and coefficient-packing paths keep the same owner split even when
  EOR is bypassed; and
- malformed session order, degree, geometry, context, and final-claim outputs
  fail without exposing or reconstructing the witness.

Backend equivalence tests compare every in-tree backend against the old
protocol-windowing reference before deleting it. Through test-only private
inspection they compare chunk coefficients, extrema, global rows, empty ranges,
multiple polynomials, relation outputs, and emitted digits. Test-only access
must not become a production handle API.

Lifecycle tests cover:

- each unit's Z bytes come from its own accepted chunk;
- A relation and consistency outputs match the old path;
- single- and multi-chunk recursive witness and proof bytes match;
- terminal payload and proof bytes match;
- accepted/rejected nonces are unchanged;
- dense, one-hot, recursive setup-prefix, and recursive witness sources;
- heterogeneous erased groups and `O != R` stacks;
- an external-style implementation of all kernels without importing
  `DecomposeFoldWitness`, `FoldChunkCoefficients`, or private writer types;
- malformed sink/layout requests return `AkitaError` without panic;
- out-of-range, overlapping, incomplete, and partial-then-error Z writes do not
  modify the destination witness;
- terminal payload validation rejects wrong coordinate counts, coefficients
  above the scheduled cap, noncanonical padding, trailing bytes, and payloads
  above the maximum budget;
- variable-length canonical terminal payloads remain unpadded and byte-identical
  to the old encoder.

### Performance

The API permits one packed traversal but does not require it. Initial generic
backends may repeat work per supplied range as they do today. The cutover must
not add a second probe after live replay. Benchmark CPU paths against the old
implementation and record probe, relation, and Z-emission time separately.
Proof size and setup size must remain identical.

Splitting EOR adds one consumer round trip per sumcheck round unless the
consumer and coordinator are colocated. Implementations should support a
stateful local session so CPU backends pay only dynamic-dispatch overhead;
remote consumers may batch transport but may not precompute across transcript
challenges. Benchmarks must report witness preparation, each EOR round, local
setup contribution, coordination, and finalization separately.

## Alternatives considered

### Public chunk witness result

Returning `Chunked(global, centered_chunks)` gives Akita the desired one-call
dispatch but preserves the representation coupling that prevents shard-local
or device-local responses. It is superseded by the opaque handle.

### Protocol-visible fallback

Capability or `FallbackPerPoly` results make orchestration branch on backend
strategy. Every backend instead supports the semantic plan and chooses its own
private implementation.

### Arbitrary consumer-selected partitions

Allowing a consumer to remap blocks would split protocol, planner, witness
layout, and verifier geometry. Akita's canonical dyadic ranges remain the
single semantic authority.

### Re-probe after live replay

Re-probing can simplify some transactional backends but doubles work for the
winning candidate and weakens the exactly-once goal. The accepted preview
handle remains valid after challenge replay instead.

### Require `O == R`

This would simplify handle routing but regress heterogeneous compute stacks.
The opening backend owns the handle and cooperates with the separate
ring-switch context.

## Execution

1. Add private constructors and validation for fold probe, acceptance,
   relation, emission, diagnostic, and terminal plans.
2. Add the immutable accepted manifest, operation-context identity, scoped
   staging sink, and sealed canonical terminal payload.
3. Introduce the CPU accepted handles and move existing computation and
   admission behind them without changing behavior.
4. Implement the static kernels for every in-tree source/backend pair.
5. Add the private object-safe adapter for heterogeneous groups and validate
   context identity at every later operation.
6. Split recursive suffix sources by owner: keep the setup-prefix source local,
   replace protocol-owned recursive-witness sources with opaque consumer
   handles, and remove the shared raw `logical_groups` construction.
7. Add the validated recursive-witness opening/EOR plans, public opening
   messages, single-use consumer session, and final-claim result.
8. Refactor EOR into an owner-aware coordinator that sums the local setup
   polynomial and consumer witness polynomial each round before performing the
   existing transcript absorption and challenge sampling.
9. Route EOR challenges back into both contributors and preserve canonical
   final-claim ordering. Do not add a setup sumcheck kernel or change the
   setup-only stage-3 prover.
10. Route grinding through `probe`, drop rejected-candidate handles, and retain
   the winning handle across live replay.
11. Route A-side relation and consistency arithmetic through the handle.
12. Route recursive Z emission through the exact-coverage staging sink.
13. Move terminal admission and Z encoding behind the terminal handle, then
   validate and seal the canonical payload before response construction.
14. Remove protocol-owned response carriers, coefficient inspection, and
   challenge windowing.
15. Run equivalence, byte-identity, mixed setup/witness EOR,
    heterogeneous-stack, terminal, and external implementation tests before
    deleting the old reference path.

Primary Akita areas include:

- `crates/akita-prover/src/backend/plans/mod.rs`;
- `crates/akita-prover/src/backend/traits.rs`;
- `crates/akita-cpu-backend/src/opaque/capabilities.rs`;
- `crates/akita-cpu-backend/src/opaque/backend.rs`;
- dense, one-hot, and recursive opening backends;
- `crates/akita-prover/src/backend/messages.rs`;
- `crates/akita-prover/src/protocol/fold_grind.rs`;
- `crates/akita-prover/src/protocol/prove/suffix.rs`;
- `crates/akita-prover/src/protocol/prove/opening_reduction.rs`;
- `crates/akita-cpu-backend/src/opaque/recursive/opening/`;
- `crates/akita-cpu-backend/src/opaque/sumcheck/stage3/` (ownership and
  byte-identity tests only; no offload kernel);
- ring-relation witness and quotient construction;
- recursive ring-switch coefficient emission;
- terminal suffix and response construction.

No verifier algorithm, proof layout, planner, or schedule change should be
necessary.

## Documentation

When implemented, fold the durable ownership model into
`book/src/how/proving/root-fold-ring-switch.md`, terminal behavior into
`book/src/how/recursion.md`, and backend contracts into
`book/src/how/architecture.md`. Update the live spec index and documentation
blast-radius map when this proposal is accepted. Archive this record after the
book owns the stable contract.

## References

- [`dyadic-chunk-partition.md`](dyadic-chunk-partition.md)
- [`fold-linf-rejection.md`](fold-linf-rejection.md)
- [`heterogeneous-group-source-contracts.md`](heterogeneous-group-source-contracts.md)
- [`selective-l2-fold-security-sizing.md`](selective-l2-fold-security-sizing.md)
- [`subring-coefficient-packing.md`](subring-coefficient-packing.md)
