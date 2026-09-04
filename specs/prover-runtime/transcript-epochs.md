# Transcript epochs and message boundaries

| Field | Value |
|---|---|
| Package | [`README.md`](README.md) |
| Status | proposed |
| Role | normative semantic-boundary catalog |

## Rule

A runtime epoch begins with the challenges and public data available after the
previous Fiat–Shamir squeeze and ends when the complete next canonical prover
message bundle is available. The driver validates and absorbs that bundle, then
performs the next squeeze or consecutive squeeze family.

Equivalently:

```text
driver: absorb prior message → squeeze challenge bundle
runtime: challenge bundle + checked plan + opaque state → next message + state
driver: validate next message → absorb → squeeze
```

The runtime operation MUST span every causally available witness computation
between those synchronization points. Existing module, stage, recursion-level,
kernel, and collection-element boundaries do not create semantic epochs.

Commitment and precommitment occur before the interactive transcript schedule,
but follow the same message-plus-state rule: one semantic operation produces
the complete canonical public commitment and opaque retained state.

## Boundary consequences

- Several ordered fields absorbed without an intervening squeeze form one
  message bundle.
- Several challenges squeezed without an intervening prover message form one
  input challenge bundle. They do not cause empty runtime calls.
- Public or verifier-derived fields absorbed between two witness-derived fields
  remain driver-owned. An epoch response may contain an ordered message bundle
  whose positions the driver interleaves with those deterministic absorbs.
- Per-group, per-polynomial, per-compression-map, per-relation, and per-sum-check
  member results are not semantic outputs when the transcript absorbs only
  their protocol-defined aggregate.
- A value later serialized, absorbed, or checked as a prover message MUST have
  a canonical message type even if current code stores it inside a “hint.”
- A value never visible to the verifier MUST remain backend-private unless an
  explicit checkpoint or diagnostic contract exports it.

## Driver and runtime ownership

The driver MUST own:

- transcript initialization and instance binding;
- every domain separator and transcript label;
- canonical message validation and encoding;
- the order of every absorb and squeeze;
- challenge decoding and public rejection rules;
- schedule-selected setup-prefix slot IDs;
- grinding candidate order, batch-membership checks, preview, and live replay;
- proof construction and serialization.

The runtime MUST own:

- source and witness representation;
- inner and outer commitment intermediates;
- digit planes, compression stages, and quotients;
- folded oracles, prepared opening tables, relation weights, and residues;
- CPU, GPU, or remote object identities;
- recomputation, residency, and internal scheduling policy;
- any value not fixed by the canonical message contract.

No epoch API may accept a transcript, transcript callback, sponge snapshot,
challenge-sampling closure, or unbounded protocol RNG.

## Current live schedule

The following order is pinned to Akita
`988fc11b48d0edd77f181e96e9c23f1470a583c8`. It describes live prover paths,
not every label declared in the transcript module.

1. The driver binds the instance descriptor.
2. It absorbs root batch shape, public commitments, and opening points.
3. The prover produces opening claims.
4. When evaluation opening reduction (EOR) is enabled, the driver absorbs EOR
   openings and partials, squeezes `eta` and claim-batch coefficients, then
   performs one message/challenge transition per EOR sum-check round.
5. It absorbs EOR final claims, remaining protocol points, and scalar openings
   as applicable.
6. It absorbs the terminal opening-compression payload or raw opening vector.
7. It squeezes `CHALLENGE_EVAL_BATCH`.
8. It grinds/selects the fold nonce and replays the selected sparse-challenge
   draws against the live transcript.
9. It absorbs the next-witness binding.
10. It squeezes the ring-switch `alpha`, `tau0`, and `tau1` challenge bundle.
11. It executes Stage 1, including configured product-tree or L2 boundaries.
12. It absorbs the range-image evaluation.
13. It squeezes L2-virtual, compression-binary, and Stage-2 batching challenges
    as configured.
14. It executes Stage 2 and absorbs `next_w_eval`.
15. For recursive setup, it absorbs the deterministic setup-prefix slot ID and
    Stage-3 input claim, then executes Stage 3.
16. The suffix sequence repeats at the next level.
17. At the terminal suffix it absorbs canonical `t`, executes opening/EOR,
    squeezes the evaluation-batch challenge, absorbs `e_hat`, grinds the sparse
    challenge, and absorbs the terminal-response remainder.

Declared labels including `ABSORB_STOP_CONDITION`,
`CHALLENGE_STOP_CONDITION`, `ABSORB_RING_SWITCH_MESSAGE`, and
`CHALLENGE_LINEAR_RELATION` are not active in the pinned main prover path. The
implementation inventory MUST be generated from exercised schedules and
logging-transcript traces rather than inferred from the label declaration
list.

## Proposed epoch catalog

The names below are semantic placeholders. Implementations MAY refine the Rust
type names but MUST preserve the boundary and ownership columns.

| Epoch | Input since prior message | Canonical output | Private retained state | Current split to remove |
|---|---|---|---|---|
| Commit group | validated commit plan and source | `CommittedGroup` | source/resident upload, inner and outer rows, transforms, digits, compression state, opening state | inner commit, host decomposition, digit rows, compression-map calls |
| Opening prefix | commitment handles, public points and claims | opening claims plus EOR openings/partials or the next raw/compressed opening payload available before a squeeze | prepared openings, folded oracles, `e_hat`, compression witnesses | per-group opening/tensor calls and hint reconstruction |
| EOR round | `eta`, batch coefficients, and then the prior round challenge | one already-combined compressed round polynomial | folded EOR tables and reduction state | per-instance polynomials and host combination |
| EOR terminal | final round challenge | final claims and every later opening field available before `CHALLENGE_EVAL_BATCH` | reduced opening/fold state | generic sum-check finalization followed by host fold preparation |
| Fold grind | evaluation-batch challenge plus a driver-derived ordered candidate batch | selected nonce/candidate identity | winning fold witnesses for all groups | per-nonce and per-group `probe_fold` work |
| Next-witness binding | selected live sparse challenges and prior state | `OuterPayload` or `TerminalInnerState { t_state }` | logical/physical next witness and committed state | host witness construction plus a fragmented commitment pipeline |
| Ring-switch/Stage-1 prefix | consecutive `alpha`, `tau0`, `tau1` challenge bundle | first Stage-1 message, normally the round-zero polynomial | relation weights, witness-evaluation tables, digit-range state | multiple host materializations before generic sum-check |
| Sum-check group round | prior round challenge and checked active-group plan | one protocol-combined round polynomial | folded group tables | per-member compute, host aggregation, separate challenge ingestion |
| Digit-range interstage | final product-level challenge | child-claim message bundle | next-level product or leaf state | host child-claim extraction and new prover construction |
| L2 prefix | current Stage-1 state and prior challenge | response norm and configured subclaims | fused L2 state | separate response/subclaim operations |
| L2/Stage-1 terminal | final L2 or Stage-1 round challenge | virtual evaluations and range-image evaluation | completed Stage-1 state and Stage-2 source state | virtual-evaluation and fold code return separately before driver absorption |
| Stage-2 prefix | consecutive L2-virtual, compression-binary, and Stage-2 batch challenge bundle | first Stage-2 round polynomial | Stage-2 relation state | challenge sampling, relation preparation, and generic round-zero setup are separate calls |
| Stage-2 round | prior round challenge | one combined Stage-2 polynomial | folded Stage-2 tables | transcript-owning generic sum-check |
| Stage-2 terminal/Stage-3 prefix | final Stage-2 challenge plus deterministic setup slot and any next challenge family | `next_w_eval`, recursive setup-product claim, and first Stage-3 polynomial when no squeeze separates them | Stage-3 state | stage output structs, public slot absorb, input-claim absorb, round-zero call |
| Stage-3 round | prior challenge | one combined Stage-3 polynomial | setup-product tables | transcript-owning generic sum-check |
| Terminal opening | evaluation-batch challenge | `TerminalEHatMessage` | terminal opening witness | host folded-`e` materialization |
| Terminal grind | driver-derived ordered sparse-challenge candidates | selected nonce/candidate identity | winning terminal-response state | preview work repeated per candidate |
| Terminal response | selected live sparse challenge | canonical response remainder or full response bundle | none required | canonical `t` recovered from a hint and remainder handled separately |

Strict maximality can cross an existing Rust stage or recursion-level boundary.
An ordered message bundle is preferable to an artificial RPC boundary that
exists only because current modules return intermediate structs.

## Explicit terminal inner-state message

The current code already distinguishes `NextWitnessState::TerminalInnerState`
but recovers its bytes from `next_commitment.hint.inner_rows()[0]`. Those bytes:

1. are absorbed as `ABSORB_NEXT_LEVEL_WITNESS_BINDING`;
2. are absorbed again as `ABSORB_COMMITMENT` when the terminal suffix begins;
3. are checked by the verifier against `terminal_response.t_fields`.

They are therefore an explicit protocol message, provisionally:

```rust,ignore
pub enum NextWitnessBindingMessage<F> {
    OuterPayload(OuterPayloadMessage<F>),
    TerminalInnerState { t_state: TerminalInnerState<F> },
}
```

The exact canonical `TerminalInnerState` representation remains the current
protocol representation unless a separate protocol change modifies it. The
runtime must produce it explicitly; the driver validates and absorbs it in both
required locations. General inner rows, compression stages, and compression
quotients remain private state.

The first commitment cutover MUST eliminate all live
`hint.inner_rows()[0]` reads used to construct transcript messages.

## Sum-check boundary

A sum-check epoch is group-level, not prover-instance-level. The request is
conceptually:

```rust,ignore
pub struct SumcheckGroupRoundRequest<F> {
    pub round: usize,
    pub prior_challenge: Option<F>,
    pub prior_claim: F,
    pub active_members: ValidatedActiveMemberPlan<F>,
    pub batch_coefficients: ValidatedBatchCoefficients<F>,
}
```

The runtime binds the prior challenge into all active member state, computes
member contributions, applies the protocol-defined batching, and returns the
single canonical polynomial the driver absorbs. It MUST NOT return one
polynomial per member for host combination.

After the final challenge, one finalization epoch SHOULD perform every
causally available final bind, claim extraction, relation validation, and
residue parking before the next message. This is the generalization of Jolt's
good prior-challenge-to-next-polynomial contract beyond its current per-member
boundary.

## Grinding boundary

Grinding is the one epoch family whose challenge input is a driver-created
candidate set rather than one already-live challenge.

The required sequence is:

1. The driver snapshots the pre-grind transcript state.
2. In protocol order, it derives a bounded candidate batch. Each candidate
   contains the nonce and fully decoded challenge bundle for every affected
   group, together with public method/configuration domain data.
3. The runtime evaluates all candidates in one semantic operation and returns
   the first accepting candidate plus an opaque handle to retained winning
   state.
4. The driver verifies candidate membership and order metadata. It cannot prove
   that every earlier witness-dependent candidate failed from the chosen nonce
   alone.
5. The driver replays only the selected nonce and challenges against the live
   transcript and requires exact equality with the preview.
6. Only after replay succeeds may the driver absorb the resulting binding or
   response message.

Large search spaces MAY be streamed as idempotent, ordered pages within the same
in-flight semantic request, with stable request IDs and state generations. Page
flow control is transport-internal; it MUST NOT return an intermediate protocol
decision to the driver or create another epoch result. Paging changes transport
scheduling, not candidate order or protocol semantics. No live transcript
mutation occurs until the single winner is returned and replayed.

First-accept semantics are enforced by CPU differential and runtime conformance
tests. Proof soundness MUST NOT depend on selecting the first accepting nonce;
the ordering rule exists for deterministic backend-invariant proof bytes. An
independent host minimality proof would require extra acceptance evidence and
verification cost and is not part of this design.

This removes per-candidate and per-group RPCs without giving the backend the
transcript or control over candidate order.

## Message validation

Every epoch response type MUST provide one canonical structural validator and
one transcript append implementation shared with or mechanically tied to
verifier parsing. The driver MUST also enforce every cheap public algebraic
transition predicate before any field is absorbed. In particular, a sum-check
round checks its degree bound and `g(0) + g(1) == prior_claim`; grouped execution
does not move that check into backend-private state. Commitment and relation
messages similarly check canonical form, ordering, checked-plan identity, and
public geometry. Witness-semantic correctness that would require repeating
prover work is not independently recomputed by the driver and remains covered
by final verifier checks and backend conformance testing.

If validation fails:

- the logging transcript remains unchanged;
- proof state does not advance;
- output handles from the failed response are invalidated or never published;
- retry follows the explicit state/transport policy rather than silently
  selecting another backend.

Ordered bundles MUST prevent callers from reordering, omitting, or duplicating
fields. A generic `Vec<Field>` without a canonical message wrapper is not an
acceptable semantic result.

## Remote idempotency

Every remote epoch request SHOULD carry a stable request ID, input state
generation, plan identity, and expected output-message kind. Repeating an
identical request after an uncertain transport result MUST either:

- return the same message and output handle generation; or
- report that the result was committed and provide a recoverable lookup path.

It MUST NOT advance private state twice. A request with a stale generation or a
different payload under the same ID MUST fail before protocol state changes.

Idempotency is an operational property; request IDs and backend generations are
never transcript inputs.

## Conformance tests

The epoch inventory is enforced by:

- exact `LoggingTranscript` event and proof-byte parity against the pinned CPU
  implementation across root/suffix, EOR/no-EOR, raw/compressed opening,
  L-infinity/L2, direct/recursive setup, and terminal paths;
- a fake remote call counter requiring one call per maximal epoch, no
  per-group/member/map calls, and one semantic grinding call per candidate
  batch;
- a representation-independence backend whose retained state is only remote
  object IDs;
- explicit terminal-message tests proving the same `t_state` is used for both
  transcript bindings and the `terminal_response.t_fields` equality;
- malformed-message tests requiring rejection before absorb and an unchanged
  logging transcript;
- wrong-owner, wrong-domain, wrong-kind, and stale-generation tests;
- grind tests for a non-minimal nonce, preview/live mismatch, incorrect group
  ordering, and incorrect nonce binding;
- retry tests proving remote idempotency;
- an API check that semantic runtime traits contain no transcript or protocol
  RNG parameter.

The characterization suite from Phase 0 of [`roadmap.md`](roadmap.md) MUST land
before implementation changes the transcript-owning drivers.
