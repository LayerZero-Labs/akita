# Spec: Prover orchestration with explicit stages

| Field | Value |
|-------|-------|
| Author(s) | Codex, for maintainer review |
| Created | 2026-09-10 |
| Status | implemented |
| PR | Not opened |
| Supersedes | None |
| Superseded-by | None |
| Book-chapter | book/src/how/proving/proving.md |

## Summary

Refactor proving into a short sequence of named stages coordinated by an internal
`ProverExecutor`. The executor borrows the existing level stack selector. The
transcript, expanded setup, prefix slots, parameters, and intermediate results are
explicit method arguments. Each stage performs a coherent piece of protocol work
and returns the values needed downstream.

The goal is simpler, more readable orchestration with the same computational
efficiency. Small metadata copies are acceptable when they simplify ownership.
Large witnesses and partial tables must move or remain borrowed, and accepted
fold responses must be reused. This is an internal refactor: protocol rules,
proof encoding, and supported configurations remain unchanged.

## Architecture and scope

The design follows three useful patterns in Jolt's prover: a top-level function
that shows protocol order, named stage results that distinguish proof fields from
downstream data, and canonical protocol helpers shared wherever ordering or
geometry must agree. Jolt keeps one transcript local and passes it explicitly to
its stages. Akita will use the same ownership pattern.

The scope includes batched admission, root and recursive preparation, relation
construction, ring switching, sumcheck orchestration, terminal proving, and proof
assembly. Existing arithmetic engines, commitment execution, source views, and
backend interfaces remain in place. Backend support, registries, sessions,
device management, code generation, new proof modes, and kernel optimization are
outside this refactor. The design introduces no general stage framework or
typestate machine.

The reviewed Akita base is `feat/backend-commit-simplified` at
`b22c41e0a6818740c015f6f94d42b181698dfa26`. The architectural reference is Jolt at
`e99577c48fcaee46af0578ec955ffed280ec56c4`, specifically its
`crates/jolt-prover/src/akita/prover.rs`,
`crates/jolt-prover/src/stages/stage1.rs`,
`crates/jolt-prover/src/stages/stage6b.rs`, and
`crates/jolt-prover/src/driver.rs`. These are source references from the reviewed
checkout, not dependencies or implementation requirements for Akita.

## Implemented structure

| Current owner | Responsibility | Proposed change |
|---|---|---|
| [core/prove.rs](../crates/akita-prover/src/protocol/core/prove.rs): `batched_prove` | Admission, prewarming, transcript setup, root/suffix execution, resource lifecycle, proof assembly | Construct the executor at the public entry and express the body as named executor stages |
| [types/opening_data.rs](../crates/akita-prover/src/types/opening_data.rs) | Constructor validation and stored opening layout; scheduled commitment shape checking during claim binding | Preserve these validation boundaries and use the borrowed layout accessor |
| [core/root_fold.rs](../crates/akita-prover/src/protocol/core/root_fold.rs) | Bind root claims, prepare the opening, execute a fold | Keep a short root-specific front feeding shared fold stages |
| [core/fold/mod.rs](../crates/akita-prover/src/protocol/core/fold/mod.rs): `prepare_fold_relation` | Native group opening preparation and `PreparedFold` assembly | Keep the shared root/recursive preparation boundary explicit and pass transcript dependencies directly |
| [ring_relation.rs](../crates/akita-prover/src/protocol/ring_relation.rs): `RingRelationProver::prepare` | Retained material, opening payload, late claim binding, fold grinding, relation instance and witness | Preserve these ordered phases in one ownership-safe operation and return a named result; the former late-batching callback is removed |
| `core/fold/mod.rs`: `prove_fold` | Build and commit next witness, ring switching, stages 1–3, proof and suffix-state assembly | Split into named operations with explicit outputs and required successor inputs |
| [core/suffix.rs](../crates/akita-prover/src/protocol/core/suffix.rs) | Recursive source preparation, schedule loop, terminal proof | Keep the loop and terminal path explicit; pass carried values between levels |
| [core/fold/stages.rs](../crates/akita-prover/src/protocol/core/fold/stages.rs) | Three sumcheck stages | Reuse these meaningful boundaries; move orchestration bodies without adding forwarding layers |

The base already validates claim/source alignment in `ProverOpeningData`
constructors, provides a borrowed `opening_layout()`, and exposes only
`batched_prove` as the public orchestration entry. Preserve those simplifications.
Repeated field bounds can be deduplicated and shared on impl blocks where useful.

## Executor and ownership

The executor has one field:

```rust,ignore
struct ProverExecutor<'a, Stacks> {
    stacks: &'a Stacks,
}
```

This field removes repeated stack-selector arguments from root, recursive, and
commitment orchestration. It does not introduce a new backend abstraction.
Executor methods use `&self`; the executor contains no transcript, setup, prefix
registry, current level, witness, cached stage result, or mutable progress.

Transcript-sensitive stages receive `&mut T` explicitly. Pure preparation and
assembly functions receive no transcript. The same transcript is passed through
the complete proof, with the grinding wrapper created and finished locally in
`prove`. Its plan and the caller's underlying transcript remain local borrows,
so no extra lifetime coupling or interior mutability is needed on the executor.

Expanded setup and prefix slots are passed only where needed, including admission,
ring-switch finalization, setup sumcheck, and recursive setup-prefix preparation.
Do not add them to argument bundles that merely hide repeated dependencies.

Use an executor method when it owns a real stage. Keep mathematical functions and
existing canonical operations callable directly. Move replaced orchestration
bodies and delete the old versions; an additional method that only forwards to
the same function would not simplify the implementation.

## Top-level proving

Keep public `batched_prove` as the single API and orchestration boundary. It
constructs the executor and invokes its stages directly, avoiding a second
pass-through entry point with duplicate generic bounds.

The intended shape is below. All snippets in this spec are control-flow sketches;
generic bounds and detailed stage argument lists are omitted.

```rust,ignore
fn batched_prove(expanded, prefix_slots, schedules, stacks, opening, basis, transcript) {
    let executor = ProverExecutor { stacks };
    let admitted = executor.validate_params(expanded, schedules, opening)?;
    executor.prepare_resources(admitted.schedule)?;
    let plan = bind_transcript_instance_descriptor(
        expanded, admitted.claims.opening_layout(), admitted.selection,
        admitted.schedule, basis, transcript,
    )?;
    let mut transcript = ProverGrindingTranscript::new(transcript, &plan)?;
    let schedule = admitted.schedule;
    let root = executor.prove_root(
        expanded, prefix_slots, admitted.claims, schedule, basis, &mut transcript,
    )?;
    stacks.after_root_fold()?;
    let suffix = executor.prove_suffix(
        expanded, prefix_slots, root.next_state, schedule, &mut transcript,
    )?;
    let nonce_stream = transcript.finish()?;
    Ok(AkitaBatchedProof {
        nonce_stream,
        root: root.level_proof,
        recursive_folds: suffix.recursive_folds,
        terminal: suffix.terminal,
    })
}
```

`validate_params` resolves the trusted catalog selection, validates its agreement
with the stored opening layout, checks execution/setup compatibility, and
preflights retained-state consumers against the exact root commitment plan. Inner
state validation receives the planned inner operation and group source count;
compressed outer state validation receives the group compression plan and ring
relation mode. It returns claims, selection identity, and the selected schedule
reference in a private ordinary input struct. Read the layout from the claims; do
not store a duplicate or a reference into an owned field of that struct.

Claim point arity, evaluation count, and source/layout alignment remain constructor
checks. Scheduled root commitment row shape remains checked by
`ProverOpeningData::append_to_transcript`. Admission must not reimplement these
checks. Setup identity is validated when compute stacks and commitment executors
are constructed. State-consumer and routing preflight still occurs before
prewarming and descriptor binding.

`prepare_resources` calculates the current NTT requirements and prewarms the
supplied resources. Preserve `after_root_fold` at the root/suffix boundary and
preserve all stack-selection indexes. No stage creates a second cache owner.

## Stage results and dependencies

As in Jolt, each stage returns named values that make its consumers visible.
Akita already has useful examples: `ProveLevelOutput` separates `level_proof`
from `next_state`, and `PreparedFold` owns a relation instance and witness along
with opening material. Retain these concepts rather than inventing a generic
result framework.

Replace positional multi-value tuples with named results at meaningful boundaries.
For example, the stage-1 result should name its proof, sumcheck point, range-image
evaluation, and optional physical-L2 replay. Keep the stage-1 proof required;
remove its current `Some` followed by an immediate requirement that it be present.

Proof fields survive until assembly. Downstream data survives only until its last
consumer. Destructure results to move large fields into the next operation and
drop scratch promptly. A result struct must not extend every intermediate's
lifetime to the end of the proof or duplicate a large buffer into both proof and
execution fields. An `Option` represents a real protocol alternative, such as
extension reduction or setup-prefix opening, rather than incomplete execution.

Document each stage's inputs, outputs, transcript position, and heavy allocations
at its implementation. Existing canonical point derivation, claim checking, and
transcript encoders remain the authority. Jolt's shared driver motivates this
single-source rule; introducing its generated driver machinery is unnecessary.

## Fold preparation

Root preparation binds root claims and enters the existing preparation path that
does not perform extension-opening reduction. Recursive preparation binds the
carried commitment, creates witness and optional setup-prefix sources, and runs
extension reduction when the opening method and extension degree require it.
Both feed the same subsequent relation-preparation sequence. The implementation
keeps source borrows and large opening material within the canonical relation
operation, while exposing the surrounding stages explicitly:

```rust,ignore
let prepared = prepare_fold_relation(
    stack, claims, protocol_points, reduction, opening_batch,
    level, level_params, basis, pad_base_evals, transcript,
)?;
```

| Stage | Work and ownership |
|---|---|
| Compute partial openings | Validate protocol-point dimensions where required, dispatch each group at its native ring dimension, and compute partials and scalar openings. Borrow polynomial sources and own the outputs. Preserve recursive point absorbs and scalar-claim binding order |
| Prepare opening payload | Acquire retained inner/outer material, materialize opening digits, compute D-role rows and compression, and bind the complete opening payload. Own material needed for relation assembly while leaving source views available for grinding |
| Bind evaluation claims | Call canonical evaluation batching and row-coefficient construction after payload binding. Return the trace claim and coefficients as named values |
| Sample accepted fold response | Run the existing joint grinding routine, returning accepted challenges, folded witnesses, and chunk coefficients together |
| Assemble fold relation | Consume the payload and accepted responses to build group products, relation RHS and quotients, the witness, and `PreparedFold`; release preparation-only material |

This structure removes `bind_claims_after_payload`, `FinishFoldArgs`, and
`RingRelationProver::new`. `RingRelationProver::prepare` binds the complete
payload before calling the canonical evaluation-claim helper, then reuses the
accepted fold response while assembling the relation. Group geometry derivation
remains canonical. Avoid a `get_params` wrapper that merely forwards to an
existing parameter method.

Challenge sampling and folded-witness computation are coupled: acceptance requires
computing the candidate witness. The current joint grind retains the first jointly
accepted candidate and replays its challenges on the live transcript. Preserve
that behavior, including all rejection bounds. Separating it into “sample
challenges” and “compute witness” must not recompute the accepted witness. Call
the existing grinding function directly unless its actual body is moved.

## Nonterminal fold execution

The common fold driver performs the same stages for root and recursive inputs:

```rust,ignore
let next = self.commit_next_witness(level, successor, expected_len, prepared, transcript)?;
let relation = ring_switch_finalize(expanded, /* next and relation inputs */, transcript)?;
let range = self.prove_range_image(/* relation inputs */, transcript)?;
let inputs = self.prepare_relation_sumcheck(/* relation, range, openings */, transcript)?;
let opening = self.prove_relation_sumcheck(inputs, transcript)?;
let setup = self.prove_setup_sumcheck(
    expanded, prefix_slots, /* successor and opening inputs */, transcript,
)?;
// Move proof fields into FoldLevelProof and downstream values into SuffixProverState.
```

`commit_next_witness` builds the logical witness, checks its scheduled live length,
aligns it for the successor commitment, commits it, and absorbs the binding.
It calls `ring_switch_build_w`, `commit_w`, or `commit_terminal_w` directly.
Require successor parameters and expected output length in this driver; its
current callers always provide them. Derive recursive outer-payload versus
terminal inner-state binding from the successor variant and preserve schedule
consistency checks.

`prove_range_image` owns the current stage-1 body and the following range-image
evaluation absorb. `prepare_relation_sumcheck` owns optional L2 replay batching,
optional compression batching, stage-2 batching, and opening-family-specific
linear-term preparation, in that order. `prove_relation_sumcheck` owns the current
stage-2 body and the following next-w evaluation absorb. `prove_setup_sumcheck`
owns the stage-3 orchestration when required by the recursive successor and setup
contribution mode.

Move these orchestration bodies instead of retaining `prove_stage1/2/3` as
forwarding aliases. Preserve the standalone sumcheck engines, canonical relation
helpers, and their optimized data representations.

## Recursive and terminal execution

The suffix is an explicit schedule loop: check carried witness length, prepare
the level, execute the common nonterminal fold, and move its returned state into
the next iteration. After the loop, check the terminal input length and call the
terminal stage. `SuffixProverState` remains an ordinary value carried between
levels, outside the executor.

Keep recursive source owners local while borrowed source views are used. Preserve
the distinction between logical and committed witnesses and the optional
setup-prefix source. Avoid self-referential result structs or polynomial clones
introduced solely to satisfy a new stage boundary.

Terminal proving has its own sequence: bind terminal inner state, prepare its
opening with extension reduction when required, bind terminal opening rows,
sample the accepted terminal response, and construct and absorb that response.
It uses the same transcript passed through the suffix. It does not run the
ordinary three sumchecks or require a fictitious successor.

## Protocol and performance invariants

For identical deterministic inputs, preserve serialized proof bytes, transcript
events and encodings, labels, challenge draw counts, grinding sites, and nonces.
Each stage documents its entry/exit transcript boundary. In particular:

1. Bind the selected instance descriptor before root proving, retaining root
   claim and recursive commitment order.
2. Preserve extension-opening reduction's internal batching and its distinct
   later application batching.
3. Bind the complete D/compression opening payload before late evaluation
   batching, then sample the accepted fold response.
4. Bind the next witness before ring-switch challenges, preserving the distinct
   recursive outer-payload and terminal inner-state encodings.
5. Absorb the range-image evaluation after stage 1, then perform optional L2
   batching, optional compression batching, and stage-2 batching. Preserve each
   grinding call before its challenge draw.
6. Absorb the stage-2 next-w evaluation before stage 3. Finish the grinding
   transcript only after terminal proving.

Preserve group order, heterogeneous ring dimensions, opening families, both
relation modes, raw/compressed payloads, and setup-prefix semantics. Security
bounds and sizing use canonical geometry and `akita_error::checked` primitives.
Do not weaken malformed-input rejection or introduce verifier-reachable panics,
unchecked indexing, or unchecked allocation.

Clone small parameters, points, or handles when that makes the code clearer.
Inspect vector-bearing types before treating their clone as cheap. Move or borrow
partial tables, digit blocks, witnesses, and NTT tables. Preserve fused batching,
streamed source access, compact sumcheck inputs, accepted candidate reuse, and
the existing optional dual logical/packed witness representation. No extra full
witness pass or eager dense materialization may be introduced by a stage split.

Retain tracing spans and resource-release points so performance remains comparable.
For future changes to the arithmetic or allocation strategy, capture release-mode
baselines with the
[profiling harness](../book/src/usage/profiling.md). Compare repeated runs on the
same machine, inputs, schedules, features, worker count, and cache conditions.
Cover dense and one-hot inputs, multiple groups, a recursive setup-prefix case,
and a large workload exposing peak memory. Record prove time, stage timings, peak
RSS, proof size, and grinding attempts. Investigate reproducible median prove-time
increases above 3% or peak-RSS increases above 5%; these are proposed review
thresholds, not permission to add large copies. Small metadata-copy costs within
measurement noise are acceptable.

## Implementation and validation

The implementation was organized in these reviewable steps:

1. Introduce the executor and move admission/root/suffix orchestration, retaining
   explicit transcript, setup, and prefix-slot arguments.
2. Extract nonterminal stages, named outputs, and required successor inputs.
3. Make payload-before-claim ordering direct in relation preparation; remove the
   callback and obsolete argument bundle.
4. Keep terminal proving as its distinct final stage and complete validation.

Keep related methods in focused protocol modules rather than one large executor
file. Use semantic names such as opening preparation, fold response, and relation
sumcheck. Avoid a stage trait or macro solely to make unlike phases look uniform.

Regression coverage includes the prover's
[orchestration_dim.rs](../crates/akita-prover/tests/orchestration_dim.rs),
[dispatch_dim.rs](../crates/akita-prover/tests/dispatch_dim.rs), and
[external_commitment_backend.rs](../crates/akita-prover/tests/external_commitment_backend.rs),
plus PCS tests for
[fp128](../crates/akita-pcs/tests/akita_fp128_e2e.rs),
[small fields](../crates/akita-pcs/tests/akita_small_field_e2e.rs),
[recursive setup](../crates/akita-pcs/tests/recursive_setup_e2e.rs),
[fold bounds](../crates/akita-pcs/tests/fold_linf.rs),
[soundness](../crates/akita-pcs/tests/protocol_soundness.rs), and
[transcript hardening](../crates/akita-pcs/tests/transcript_hardening.rs).
Preserve fold-grind acceptance tests and the suffix tests
`non_zk_eor_mismatch_is_rejected` and
`late_application_batch_rejects_beta_orthogonal_terminal_error`.

Add deterministic comparisons against the captured base for serialized proof bytes,
logged transcript events, challenges, and nonce streams. Cover direct
root-to-terminal and recursive schedules, degree-one and extension claims,
supported opening/relation/payload modes, heterogeneous groups, setup-prefix use,
and terminal binding. Use admitted combinations rather than unsupported products
of independent feature choices. Preserve rejection at claim construction,
scheduled commitment binding, and state-consumer preflight. Successful verification
or label-only tests alone do not establish full transcript equivalence.

Run cheap repository preflight gates from [AGENTS.md](../AGENTS.md) before expensive
compilation. Use scoped `rtk cargo` and `rtk cargo nextest` during implementation.
Final validation uses the current test-pass command and sharding from
[CI](../.github/workflows/ci.yml), all three mandated Clippy feature graphs, and
applicable path-triggered workflows. This refactor moves orchestration and small
metadata only: arithmetic kernels, accepted witness reuse, NTT ownership, and
large-buffer representations remain unchanged.

## Acceptance criteria

- [x] `batched_prove`, fold preparation, nonterminal execution, and terminal execution
  read as short sequences of meaningful stages.
- [x] `ProverExecutor` stores only the existing stack selector; transcript,
  expanded setup, prefix slots, and all proof progress remain explicit values.
- [x] Executor stage methods use `&self`; transcript-sensitive operations receive
  the same `&mut T`, with local grinding-wrapper creation and finalization.
- [x] Stage results name proof fields and downstream data, without extending
  heavy-buffer lifetimes or introducing deep copies.
- [x] Late batching is visibly after payload binding, and accepted fold witnesses
  are reused without recomputation.
- [x] Replaced functions and argument bundles are deleted; there are no forwarding
  aliases, duplicate protocol formulas, new backend interfaces, or stage frameworks.
- [x] Prover and PCS regression coverage, transcript-hardening tests, all required
  Clippy feature graphs, and repository guardrails pass.
- [ ] The implemented architecture is folded into the owning Book pages before
  the spec is archived according to the documentation policy.

Update the [proving overview](../book/src/how/proving/proving.md),
[root/ring-switch](../book/src/how/proving/root-fold-ring-switch.md),
[fold path](../book/src/how/proving/fold-path.md), and
[sumcheck stages](../book/src/how/proving/sumcheck-stages.md) before this live
implementation record is archived.
