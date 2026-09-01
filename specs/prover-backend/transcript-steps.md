# Transcript steps and message boundaries

| Field | Value |
|---|---|
| Package | [`README.md`](README.md) |
| Status | proposed |
| Role | normative transcript-step catalog |

## Rule

A transcript step begins with the challenges and public data available after
the previous Fiat–Shamir squeeze. It ends when every prover message needed
before the next squeeze is available. The driver validates and absorbs those
messages, then performs the next squeeze or consecutive set of squeezes.

Equivalently:

```text
driver: absorb prior message → squeeze challenges
backend: challenges + checked plan + state reference → next messages + state
driver: validate next message → absorb → squeeze
```

The backend call MUST perform all witness work made possible by those inputs.
Existing module, stage, recursion-level, kernel, and collection-element
boundaries do not create transcript steps.

Commitment and precommitment occur before the interactive transcript schedule,
but follow the same message-plus-state rule: one backend call produces the
complete public commitment and a reference to private prover state.

## Boundary consequences

- Several ordered fields absorbed without an intervening squeeze are returned
  together, in protocol order.
- Several challenges squeezed without an intervening prover message form one
  input challenge set. They do not cause empty backend calls.
- Public or verifier-derived fields absorbed between two witness-derived fields
  remain driver-owned. A backend output may contain ordered protocol messages
  that the driver interleaves with those deterministic absorbs.
- Per-group, per-polynomial, per-compression-map, per-relation, and per-sum-check
  member results are not protocol outputs when the transcript absorbs only
  their protocol-defined aggregate.
- A value later serialized, absorbed, or checked as a prover message MUST have
  a protocol message type even if current code stores it inside a “hint.”
- A value never visible to the verifier MUST remain backend-private unless an
  explicit checkpoint or diagnostic contract exports it.

## Driver and backend responsibilities

The driver MUST own:

- transcript initialization and instance binding;
- every domain separator and transcript label;
- protocol message validation and encoding;
- the order of every absorb and squeeze;
- challenge decoding and public rejection rules;
- schedule-selected setup-prefix slot IDs;
- grinding candidate order, batch-membership checks, preview, and live replay;
- proof construction and serialization.

The backend MUST own:

- source and witness representation;
- inner and outer commitment intermediates;
- digit planes, compression stages, and quotients;
- folded oracles, prepared opening tables, relation weights, and residues;
- CPU, GPU, or remote object identities;
- recomputation, storage location, and internal scheduling policy;
- any value not fixed by the protocol message contract.

No backend-call API may accept a transcript, transcript callback, sponge snapshot,
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
17. At the terminal suffix it absorbs protocol `t`, executes opening/EOR,
    squeezes the evaluation-batch challenge, absorbs `e_hat`, grinds the sparse
    challenge, and absorbs the terminal-response remainder.

Declared labels including `ABSORB_STOP_CONDITION`,
`CHALLENGE_STOP_CONDITION`, `ABSORB_RING_SWITCH_MESSAGE`, and
`CHALLENGE_LINEAR_RELATION` are not active in the pinned main prover path. The
implementation inventory MUST be generated from exercised schedules and
logging-transcript traces rather than inferred from the label declaration
list.

## Backend call before the transcript starts

Commitment is not a transcript step because no challenge draw precedes it. It
uses the same message-plus-state rule:

| Call | Input | Protocol output | Private prover state | Current split to remove |
|---|---|---|---|---|
| Commit group | checked commitment plan and source | `CommittedGroup` | source data kept by the backend, inner and outer rows, transforms, digits, compression state, opening state | inner commit, host decomposition, digit rows, compression-map calls |

## Proposed transcript steps

The names below are working descriptions. Implementations MAY refine Rust type
names but MUST preserve the input, output, and private-state boundaries.

| Step | Input since prior message | Protocol output | Private prover state | Current split to remove |
|---|---|---|---|---|
| Start opening | bound `CommittedGroupWithState` values, public points, and claims | opening claims plus EOR openings/partials or the next raw/compressed opening payload available before a squeeze | prepared openings, folded oracles, `e_hat`, compression witnesses | per-group opening/tensor calls and hint reconstruction |
| EOR round | `eta`, batch coefficients, and then the prior round challenge | one already-combined compressed round polynomial | folded EOR tables and reduction state | per-instance polynomials and host combination |
| Finish EOR | final round challenge | final claims and every later opening field available before `CHALLENGE_EVAL_BATCH` | reduced opening/fold state | generic sum-check finalization followed by host fold preparation |
| Fold grind | evaluation-batch challenge plus a driver-derived ordered candidate batch | selected nonce/candidate identity | winning fold witnesses for all groups | per-nonce and per-group `probe_fold` work |
| Next-witness binding | selected live sparse challenges and prior state | `OuterPayload` or `TerminalTFieldsMessage` | logical/physical next witness and committed state | host witness construction plus a fragmented commitment pipeline |
| Start ring switch and Stage 1 | consecutive `alpha`, `tau0`, `tau1` challenges | first Stage-1 message, normally the round-zero polynomial | relation weights, witness-evaluation tables, digit-range state | multiple host materializations before generic sum-check |
| Sum-check group round | prior round challenge and checked active-group plan | one protocol-combined round polynomial | folded group tables | per-member compute, host aggregation, separate challenge ingestion |
| Digit-range child claims | final product-level challenge | child claims in protocol order | next-level product or leaf state | host child-claim extraction and new prover construction |
| Start L2 | current Stage-1 state and prior challenge | response norm and configured subclaims | combined L2 state | separate response/subclaim operations |
| Finish L2 and Stage 1 | final L2 or Stage-1 round challenge | virtual evaluations and range-image evaluation | completed Stage-1 state and Stage-2 source state | virtual-evaluation and fold code return separately before driver absorption |
| Start Stage 2 | consecutive L2-virtual, compression-binary, and Stage-2 batch challenges | first Stage-2 round polynomial | Stage-2 relation state | challenge sampling, relation preparation, and generic round-zero setup are separate calls |
| Stage-2 round | prior round challenge | one combined Stage-2 polynomial | folded Stage-2 tables | transcript-owning generic sum-check |
| Finish Stage 2 and start Stage 3 | final Stage-2 challenge plus deterministic setup slot and any next challenges | `next_w_eval`, recursive setup-product claim, and first Stage-3 polynomial when no squeeze separates them | Stage-3 state | stage output structs, public slot absorb, input-claim absorb, round-zero call |
| Stage-3 round | prior challenge | one combined Stage-3 polynomial | setup-product tables | transcript-owning generic sum-check |
| Terminal opening | evaluation-batch challenge | `TerminalEHatMessage` | terminal opening witness | host folded-`e` materialization |
| Terminal grind | driver-derived ordered sparse-challenge candidates | selected nonce/candidate identity | winning terminal-response state | preview work repeated per candidate |
| Terminal response | selected live sparse challenge | protocol response remainder or full response | none required | protocol `t` recovered from a hint and remainder handled separately |

A transcript step may cross an existing Rust stage or recursion-level boundary.
Returning ordered messages together is preferable to adding an RPC boundary
only because current modules return intermediate structs.

## Explicit terminal `t_fields` message

The current code already distinguishes `NextWitnessState::TerminalInnerState`
but recovers its bytes from `next_commitment.hint.inner_rows()[0]`. Those bytes:

1. are absorbed as `ABSORB_NEXT_LEVEL_WITNESS_BINDING`;
2. are absorbed again as `ABSORB_COMMITMENT` when the terminal suffix begins;
3. are checked by the verifier against `terminal_response.t_fields`.

They are therefore an explicit protocol message, provisionally:

```rust,ignore
pub enum NextWitnessBindingMessage<F> {
    OuterPayload(OuterPayload<F>),
    TerminalTFields(TerminalTFieldsMessage<F>),
}
```

The exact `TerminalTFieldsMessage` protocol representation remains the current
protocol representation unless a separate protocol change modifies it. The
backend must produce it explicitly; the driver validates and absorbs it in both
required locations. General inner rows, compression stages, and compression
quotients remain private state.

The first commitment replacement MUST eliminate all live
`hint.inner_rows()[0]` reads used to construct transcript messages.

## Sum-check boundary

A sum-check step is group-level, not prover-instance-level. The request is
conceptually:

```rust,ignore
pub struct SumcheckGroupRoundRequest<F> {
    pub round: usize,
    pub prior_challenge: Option<F>,
    pub prior_claim: F,
    pub active_members: CheckedActiveMemberPlan<F>,
    pub batch_coefficients: CheckedBatchCoefficients<F>,
}
```

The backend binds the prior challenge into all active member state, computes
member contributions, applies the protocol-defined batching, and returns the
single protocol polynomial the driver absorbs. It MUST NOT return one
polynomial per member for host combination.

After the final challenge, one final backend call SHOULD perform all remaining
work possible from the current inputs: bind the final challenge, extract and
validate claims, and save any private values needed later before the next
message. This extends Jolt's good prior-challenge-to-next-polynomial contract
beyond its current per-member boundary.

## Grinding boundary

Grinding is the one step family whose challenge input is a driver-created
candidate set rather than one already-live challenge.

The required sequence is:

1. The driver snapshots the pre-grind transcript state.
2. In protocol order, it derives a bounded candidate batch. Each candidate
   contains the nonce and fully decoded challenge bundle for every affected
   group, together with public method/configuration domain data.
3. The backend evaluates all candidates in one backend call and returns
   the first accepting candidate plus a reference to its private state.
4. The driver verifies candidate membership and order metadata. It cannot prove
   that every earlier witness-dependent candidate failed from the chosen nonce
   alone.
5. The driver replays only the selected nonce and challenges against the live
   transcript and requires exact equality with the preview.
6. Only after replay succeeds may the driver absorb the resulting binding or
   response message.

Large search spaces MAY be streamed as idempotent, ordered pages within the same
in-flight backend input, with stable request IDs and state generations. Page
flow control is transport-internal; it MUST NOT return an intermediate protocol
decision to the driver or create another step output. Paging changes transport
scheduling, not candidate order or protocol semantics. No live transcript
mutation occurs until the single winner is returned and replayed.

First-accept behavior is enforced by CPU differential and backend
tests. Proof soundness MUST NOT depend on selecting the first accepting nonce;
the ordering rule exists for deterministic backend-invariant proof bytes. An
independent host minimality proof would require extra acceptance evidence and
verification cost and is not part of this design.

This removes per-candidate and per-group RPCs without giving the backend the
transcript or control over candidate order.

## Message validation

Every backend output type MUST provide one structural validator for its
protocol representation and one transcript append implementation shared with
or mechanically tied to verifier parsing. The driver MUST also enforce every
cheap public algebraic
transition predicate before any field is absorbed. In particular, a sum-check
round checks its degree bound and `g(0) + g(1) == prior_claim`; grouped execution
does not move that check into backend-private state. Commitment and relation
messages similarly check canonical representation, ordering, checked-plan identity, and
public geometry. Witness correctness that would require repeating
prover work is not independently recomputed by the driver and remains covered
by final verifier checks and backend testing.

If validation fails:

- the logging transcript remains unchanged;
- proof state does not advance;
- output references from the failed response are invalidated or never published;
- retry follows the explicit state/transport policy rather than silently
  selecting another backend.

Ordered bundles MUST prevent callers from reordering, omitting, or duplicating
fields. A generic `Vec<Field>` without a protocol message wrapper is not an
acceptable backend output.

## Safe retries for remote calls

Every remote step input SHOULD carry a stable request ID, input state
generation, plan identity, and expected output-message kind. Repeating an
identical request after an uncertain transport result MUST either:

- return the same message and output reference generation; or
- report that the request completed and provide a recoverable lookup path.

It MUST NOT advance private state twice. A request with a stale generation or a
different payload under the same ID MUST fail before protocol state changes.

This makes retries idempotent: repeating a request does not repeat the state
change. Request IDs and backend generations are never transcript inputs.

## Backend tests

The step inventory is enforced by:

- exact `LoggingTranscript` event and proof-byte equality against the pinned CPU
  backend across root/suffix, EOR/no-EOR, raw/compressed opening,
  L-infinity/L2, direct/recursive setup, and terminal paths;
- a fake remote backend that counts calls, requiring one call for each listed
  transcript step, no
  per-group/member/map calls, and one grinding call per candidate
  batch;
- a fake remote backend whose private prover state consists only of remote
  object IDs;
- explicit terminal-message tests proving the same `t_fields` are used for both
  transcript bindings and the `terminal_response.t_fields` equality;
- malformed-message tests requiring rejection before absorb and an unchanged
  logging transcript;
- wrong-backend, wrong-session, wrong-kind, and stale-generation tests;
- grind tests for a non-minimal nonce, preview/live mismatch, incorrect group
  ordering, and incorrect nonce binding;
- retry tests proving remote idempotency;
- an API check that backend traits contain no transcript or protocol
  RNG parameter.

The characterization suite from Phase 0 of [`roadmap.md`](roadmap.md) MUST land
before implementation changes the transcript-owning drivers.
