# Spec: Prover orchestration with a transcript-owning executor

| Field | Value |
|-------|-------|
| Author(s) | Codex, for maintainer review |
| Created | 2026-09-10 |
| Status | proposed |
| PR | Not opened |
| Supersedes | None |
| Superseded-by | None |
| Book-chapter | book/src/how/proving/proving.md |

## Summary

Introduce an internal, per-proof `ProverExecutor` that owns the grinding
transcript and borrows the level stack selector while exposing meaningful prover
stages. Keep all other protocol progress in local variables and stage return
values. The executor has no current level, pending witness, optional result slots,
or state machine. Transcript mutation is the one intentional evolving field: it
is the protocol's ordered Fiat-Shamir channel, not a store for stage results.

The largest benefit comes from simplifying fold preparation and `prove_fold`,
not merely wrapping the existing top-level functions. Move orchestration into
canonical stage implementations, remove the functions they replace, and keep
the existing arithmetic and backend kernels. Favor straightforward ownership
and inexpensive metadata copies; preserve fused work and large-buffer lifetimes.

This is a design proposal only. No Rust implementation or performance measurement
is included. The review is based on `feat/backend-commit-simplified` at
`759c60682fbf760afc4a2ac2f3965d0aa6551539`; the working branch is
`refactor/prover`.

## Current code review

The reviewed path runs from public batched proving through root preparation,
relation construction, ring switching, sumchecks, recursive folds, and terminal
response construction. Backend and sumcheck internals were inspected for their
interfaces and ownership constraints; this is not an exhaustive arithmetic or
security audit.

| Current owner | What it does | Refactoring opportunity |
|---|---|---|
| [`core/prove.rs`](../crates/akita-prover/src/protocol/core/prove.rs): `batched_prove` | Resolves the trusted schedule, validates schedule/setup/state consumers, prewarms NTT resources, binds the instance descriptor, selects the root successor, owns the grinding transcript, invokes root and suffix, releases root resources, and assembles the proof | It is now the single top-level orchestration target. Separate admission, resource preparation, transcript setup, and proof execution while retaining their visible order |
| [`types/opening_data.rs`](../crates/akita-prover/src/types/opening_data.rs): `ProverOpeningData` | Constructors validate claim, point, source, state, and stored-layout alignment; private fields preserve it. Root commitment row shape is checked when claims are appended to the transcript against scheduled geometry | Treat construction as the claim-shape boundary. Do not recreate those checks in executor stages; keep schedule-dependent commitment validation coupled to its canonical transcript encoding |
| [`core/root_fold.rs`](../crates/akita-prover/src/protocol/core/root_fold.rs) | Absorbs validated root claims, prepares the root with the canonical single-field fold path, and invokes the common fold | Keep this short flow, remove repeated resource arguments, and make the transition from claim binding to fold preparation explicit |
| [`core/fold/mod.rs`](../crates/akita-prover/src/protocol/core/fold/mod.rs): `finish_prepared_fold` | Validates points, computes group partial openings, binds evaluations, invokes relation construction through a callback, assembles `PreparedFold` | Split at actual transcript and ownership boundaries; eliminate `FinishFoldArgs` and the callback when the stages own this work |
| Same file: `prove_fold` | Builds/aligns the next witness, commits and binds it, finalizes ring switching, runs stages 1–3, assembles proof and suffix state | Main orchestration target: extract coherent stages with explicit results |
| [`ring_relation.rs`](../crates/akita-prover/src/protocol/ring_relation.rs): `RingRelationProver::new` | Recovers retained state, materializes opening digits and payload, calls back into claim batching, grinds folds, constructs relation instance/witness | Constructor name hides an entire protocol; expose payload preparation, accepted-fold sampling, and relation assembly in execution order |
| [`core/suffix.rs`](../crates/akita-prover/src/protocol/core/suffix.rs) | Walks scheduled folds, creates recursive sources, handles optional setup-prefix claims and extension reduction, executes a distinct terminal path | Keep the loop and terminal branch explicit; keep source owners local while their borrowed views are used |
| [`core/fold/stages.rs`](../crates/akita-prover/src/protocol/core/fold/stages.rs) | Implements digit-range, relation/range-image, and setup sumchecks | Existing useful stage boundaries; move their orchestration bodies where helpful without adding forwarding methods |

The updated base already completed several useful simplifications: it removed the
separate public `prove` and `prove_suffix` orchestration exports, deleted the
duplicate root preparation/checking layer, removed the local opening-family
resolver, and made `ProverOpeningData` construction the alignment boundary. This
proposal treats those changes as the starting point and does not recreate the
removed layers inside `ProverExecutor`.

Concrete cleanup opportunities accompany this structural change:

- Repeated field bounds (`Field`, `ExtField`, `FpExtEncoding`) obscure signatures.
  Deduplicate them and put genuinely shared bounds on executor impl blocks.
- `batched_prove` now has the right validation ownership but combines admission,
  resource preparation, transcript creation, root/suffix execution, lifecycle,
  and proof assembly. Preserve the consolidated boundary while separating these
  operations into readable executor stages.
- `ProverOpeningData::opening_layout()` is now a cheap borrowed accessor because
  constructors establish alignment. Read it from the admitted claims as needed
  rather than storing a second copy or revalidating claim/source geometry.
- Schedule geometry and opening-family decisions are still rebuilt within fold
  preparation. Resolve reusable geometry once per fold, while retaining checks at
  the schedule-dependent transcript and backend boundaries that enforce them.
- `prove_fold` accepts optional successor parameters, expected length, and binding
  policy, although its root and suffix callers supply all three. Require a
  successor and expected length. Derive binding from the existing recursive vs
  terminal successor variant, after checking agreement with the schedule contract.
- The stage-1 proof is wrapped in `Some` and later immediately required again.
  Keep it required throughout a nonterminal fold.
- The stage-1 result is a positional tuple with an optional L2 replay. Give it
  named fields so the next stage reads as protocol data flow.

These are simplifications, not evidence that the existing proof is incorrect.

## Proposed executor

Use one internal executor per proof. It holds only the resources that every
transcript-sensitive stage shares. Schematic Rust, with trait bounds omitted:

```rust,ignore
struct ProverExecutor<'transcript, 'plan, 'stacks, T, Stacks> {
    stacks: &'stacks Stacks,
    transcript: ProverGrindingTranscript<'transcript, 'plan, T>,
}
```

`expanded` setup and `prefix_slots` are deliberately not fields. They are needed
by only part of proving and remain explicit arguments to `prove` and to the root,
setup-sumcheck, recursive-preparation, and terminal stages that consume them. This
keeps the executor from becoming a bag of every available dependency. The catalog,
selected schedule, basis, level, witnesses, and stage outputs also remain arguments
or local values.

The executor owns `ProverGrindingTranscript` after the public entry has bound the
instance descriptor and created the grinding wrapper. Stage methods that absorb or
sample take `&mut self` and access `self.transcript`; callers cannot accidentally
thread a different transcript into one stage. Methods that do not touch the
transcript take `&self` or remain free functions. The executor is consumed by
`prove`, which finishes the transcript and returns its nonce stream with the proof.
Backend caches remain owned by the supplied stack selector.

Calling a canonical kernel directly is clearer than adding an executor method that
only forwards. Executor methods must own an orchestration boundary, transcript
event, or nontrivial result transformation.

Keep the executor private initially. Preserve a small public `batched_prove`
entry boundary that constructs the executor and invokes it. The updated base has
already removed the separate public `prove` and `prove_suffix` exports; preserve
that single entry boundary instead of restoring low-level aliases. Do not broaden
this into a redesign of commitment registration or public statement construction.

### Top-level flow

The admitted entry path should visibly perform these steps:

```rust,ignore
let admitted = ProverExecutor::validate_params(stacks, schedules, opening)?;
ProverExecutor::prepare_resources(stacks, &admitted)?;
let grinding_plan = bind_transcript_instance_descriptor(/* admitted inputs */, transcript)?;
let transcript = ProverGrindingTranscript::new(transcript, &grinding_plan)?;
ProverExecutor::new(stacks, transcript).prove(
    expanded,
    prefix_slots,
    admitted,
    basis,
)
```

`validate_params` consumes the already-constructed `ProverOpeningData`, resolves
the catalog selection, validates the stored opening layout against the selected
row, checks schedule/setup compatibility, and preflights retained-state consumers.
It does not repeat point, evaluation, polynomial, state, or layout-alignment checks
owned by `ProverOpeningData` constructors. Schedule-dependent commitment shape
continues to be checked by the canonical claim transcript encoder when the root
stage binds those claims. The result carries the claims, selected schedule
reference, and selection identity; downstream stages obtain the already-validated
layout through `claims.opening_layout()`. It is ordinary validated input, not
executor state; keep its fields private if that prevents accidental bypass of
admission.

`prepare_resources` performs the existing NTT requirement calculation and prewarm.
It stays before descriptor binding as in the current batched path. It does not
create another cache owner or rebuild prepared setup per stage. These pre-executor
operations can be associated functions or canonical free functions: neither needs
the grinding transcript, and inventing a partially initialized executor would make
the lifecycle less clear.

```rust,ignore
fn prove(
    mut self,
    expanded: &Arc<AkitaExpandedSetup<F>>,
    prefix_slots: &SetupPrefixProverRegistry<F>,
    admitted: AdmittedProverInput<'_>,
    basis: BasisMode,
) -> Result<Proof, Error> {
    let root = self.prove_root(expanded, prefix_slots, &admitted, basis)?;
    self.stacks.after_root_fold()?;
    let suffix = self.prove_suffix(
        expanded,
        prefix_slots,
        root.next_state,
        admitted.schedule,
    )?;
    let nonce_stream = self.transcript.finish()?;
    Ok(assemble_proof(root.level_proof, suffix, nonce_stream))
}
```

The snippets describe control flow, not proposed compilable signatures. Proof
assembly can stay an inline struct literal instead of becoming a single-use helper.
Passing setup and prefix slots once to `prove` keeps the public call compact; their
use remains visible in the smaller internal stages instead of being hidden as
executor-wide ambient context.

### Fold preparation: make the hidden ordering visible

Root preparation validates and absorbs scheduled commitment rows through
`ProverOpeningData::append_to_transcript`. Recursive preparation binds the
recursive commitment and builds source views, including the optional setup-prefix
source. Both then use the shared preparation stages below. Recursive
extension-opening reduction remains conditional on opening method and extension
degree; the root currently uses the canonical single-field preparation path even
when the claim field is an extension field.

```rust,ignore
let params = resolve_fold_geometry(level_params, opening_layout)?;
let openings = self.compute_partial_openings(&params, sources, points)?;
let payload = self.prepare_opening_payload(&params, claims, openings)?;
let batching = self.bind_evaluation_claims(/* payload-bound openings */)?;
let accepted = self.sample_accepted_fold(/* sources */)?;
let prepared = self.assemble_fold_relation(params, payload, batching, accepted)?;
```

Use precise names such as `compute_partial_openings` rather than a vague
`compute_partial_params`: deriving a layout is cheap, evaluating polynomial
partials can dominate runtime. `resolve_fold_geometry` denotes a proposed real
aggregation of reused group geometry; implement it only where it eliminates
repeated derivation. It must call canonical parameter/layout methods rather than
reimplement their arithmetic.

| Stage | Work moved into it | Result and ownership |
|---|---|---|
| Compute partial openings | Group point validation, native-dimension dispatch, partial evaluations and scalar openings from `finish_prepared_fold` | Own prepared group openings; borrow polynomial sources. Use the executor transcript for recursive point absorbs and preserve scalar-claim binding order |
| Prepare opening payload | Retained inner/outer material acquisition, opening digit materialization, D-role rows, compression, opening-payload absorb from `RingRelationProver::new` | Own material required for relation assembly; retain metadata/scalar openings for late batching; no copy of source polynomials; mutate the executor transcript for the payload absorb |
| Bind evaluation claims | Existing `prepare_evaluation_trace_claim` and row-coefficient construction | Named small batching result; call after the complete opening payload is bound, removing `bind_claims_after_payload` callback inversion |
| Sample accepted fold | Existing joint fold-grinding implementation | Return the accepted witness, chunk coefficients, and challenges together; consume them in the next stage |
| Assemble fold relation | Group opening products, relation RHS/quotients, witness and instance assembly, `PreparedFold` construction | Move accepted witnesses and payload material into the existing result; release preparation-only scratch |

Do not add a separate `sample_challenges()` followed by recomputing
`compute_folded_witness()`. The acceptance test in
[`fold_grind.rs`](../crates/akita-prover/src/protocol/fold_grind.rs) needs the
candidate witness. It searches the first jointly accepted nonce and retains the
accepted outputs. A stage named `sample_fold_response` would be truthful if the
existing function is renamed, but the implementation must remain canonical and
return both challenges and witness. Keep preview/live replay and all acceptance
bounds intact.

### Nonterminal fold execution

After preparation, the executor should read approximately as follows:

```rust,ignore
let next = self.commit_next_witness(level, successor, expected_len, prepared)?;
let relation = ring_switch_finalize(/* metadata */, &mut self.transcript)?;
let range = self.prove_range_image(/* relation */)?;
let stage2 = self.prepare_stage2(/* relation, range, openings */)?;
let opening = self.prove_relation_sumcheck(stage2)?;
let setup = self.prove_setup_sumcheck(expanded, prefix_slots, /* successor and opening */)?;
// Assemble FoldLevelProof and SuffixProverState by moving these outputs.
```

`commit_next_witness` combines existing witness construction, scheduled length
checking, alignment, recursive/terminal commitment dispatch, and binding absorb.
It returns the logical witness and commitment output along with the remaining
relation metadata needed by finalization. It calls `ring_switch_build_w`,
`commit_w`, and `commit_terminal_w` directly. It does not duplicate their kernels.

`prove_range_image` owns the existing stage-1 body and the immediate range-image
evaluation absorb. `prepare_stage2` owns the current L2 replay batching,
compression batching, stage-2 batching, and opening-family-specific linear-term
preparation. This removes a large inline branch from the fold driver.
`prove_relation_sumcheck` owns the current stage-2 body and next-w evaluation
absorb. `prove_setup_sumcheck` owns the current stage-3 orchestration, conditional
on the recursive successor and setup contribution mode. Every transcript-sensitive
method above mutates `self.transcript`; the transcript is omitted from signatures
to make the single ordered channel structural. Setup and prefix slots appear only
on the stages that need them.

Move these bodies; do not retain `prove_stage1/2/3` as additional forwarding layers.
Keep standalone sumcheck engines and mathematical helpers in their current modules.
Every stage result should answer what the next stage consumes. Prefer a few named
results for existing tuples/multi-value boundaries over a new type per statement.

### Recursive and terminal paths

The suffix remains a simple loop over the schedule: check carried witness length,
prepare this level, execute the common nonterminal fold, move its next state.
Then check terminal length and call a separate terminal method. Preserve stack
selection indexes and the root/suffix cache-release hook exactly.

The terminal method should visibly bind terminal inner state, prepare its opening
(including extension reduction when required), bind terminal opening rows, sample
the accepted terminal response, and build/absorb that response. It does not run
the ordinary three sumchecks or fabricate a successor to reuse the nonterminal
driver. Keep `SuffixProverState`: it represents real data carried between folds,
not mutable progress stored on the executor.

In recursive preparation, keep the `Arc` source owners in the local scope that
invokes the shared stages. Do not return a struct containing both owned sources
and references into those sources. Moving code into a method must not require
self-referential structures, unsafe code, or polynomial clones.

## Protocol invariants

Stage names are not permission to reorder the transcript. Preserve the complete
event stream, encoded bytes, labels, challenge draw counts, grinding sites and
nonces for identical deterministic inputs. In particular:

1. Bind the selected instance descriptor before proving; preserve root claim and
   recursive commitment encoding and ordering.
2. Keep extension-opening reduction's own transcript sequence in its canonical
   implementation. Its earlier internal batching and later application batching
   have different purposes; do not merge them.
3. Bind all opening payload digits through the D/compression payload before late
   evaluation batching, then perform fold-response grinding.
4. Bind the next witness before ring-switch challenges. Terminal inner-state and
   recursive outer-payload bindings remain distinct encodings.
5. After stage 1, absorb the range-image evaluation, then sample optional L2
   virtual batching, optional compression batching, and stage-2 batching in the
   current order. Preserve grinding calls before their challenge draws.
6. Absorb the stage-2 next-w evaluation before any stage-3 work. Finish the
   grinding transcript only after the terminal proof is complete.

Preserve root-group ordering, group-specific ring dimensions and packing geometry,
both relation modes, raw/compressed payload distinctions, setup-prefix openings,
and the distinction between logical and committed witnesses. All sizes and
security bounds continue to come from shared canonical geometry and checked
arithmetic. No new verifier-reachable panics, unchecked indexing, or unchecked
allocation are introduced. Validation must not be removed simply because a stage
now receives an executor reference.

## Efficiency and ownership rules

| Data/work | Rule |
|---|---|
| Scalar parameters, small points, handles | Copy or clone where it simplifies code; no elaborate borrowing solely to avoid a small copy |
| Parameter/layout structures with vectors | Borrow by default or clone deliberately; do not assume an arbitrary `Clone` implementation is constant-cost |
| Partial tables, digit blocks, folded witnesses, NTT tables | Move or borrow; no new deep clones to cross stage boundaries |
| Accepted fold candidate | Reuse its witness and coefficients; never redo accepted decompose/fold work |
| Fused batches, compact sumcheck inputs, streamed sources | Preserve current kernels, source views and execution plans; no eager dense materialization |
| Logical and packed next witness | Preserve the current optional dual representation; do not always retain two witnesses |
| Scratch and caches | Drop preparation-only data promptly; keep `after_root_fold` and existing backend ownership/lifecycle semantics |

Avoid one giant result that retains every intermediate until proof assembly.
Destructure and consume results as soon as their large fields become unnecessary.
Preserve existing tracing spans so measurements remain comparable. Do not change
dispatch, SIMD, Rayon, NTT cache policies, or sumcheck algorithms in this refactor.

Efficiency is an acceptance condition, not a claim established by this document.
Before implementation, record release-mode baselines using the canonical
[profiling harness](../book/src/usage/profiling.md) at the base commit, then repeat
with the same schedules, inputs, machine, feature sets, worker count, and cache
conditions. Include dense and one-hot workloads, multiple groups, a recursive
setup-prefix case, and a large case exposing peak memory. Record prove time,
stage spans, peak RSS, proof bytes, and grinding attempts. Use repeated runs and
investigate a reproducible median prove-time increase above 3% or peak-RSS increase
above 5%; these are proposed review thresholds, not permission to add large copies.
Any extra full-witness pass or allocation requires explanation even below those
thresholds. Small metadata-copy costs within measurement noise are acceptable.

## Implementation sequence and validation

1. Capture deterministic proof/transcript and performance baselines before moving
   code. Use existing end-to-end fixtures; add missing boundary regression cases.
2. Introduce the per-proof transcript-owning executor and move root/suffix
   orchestration. Consolidate validation ownership and repeated bounds without
   changing kernels.
3. Extract nonterminal stages and named results. Make successor requirements
   explicit, remove the stage-1 optional round trip, and check memory lifetimes.
4. Split relation preparation at payload binding and accepted-fold sampling;
   remove the callback and replaced constructor/argument bundle. This is the
   most sensitive change and deserves its own transcript comparison.
5. Simplify terminal orchestration and delete remaining replaced helpers. Run the
   complete validation and repeat performance comparisons.

Existing regression evidence includes:

- [`orchestration_dim.rs`](../crates/akita-prover/tests/orchestration_dim.rs),
  [`dispatch_dim.rs`](../crates/akita-prover/tests/dispatch_dim.rs), and
  [`external_commitment_backend.rs`](../crates/akita-prover/tests/external_commitment_backend.rs)
  for schedule/backend boundaries.
- [`akita_fp128_e2e.rs`](../crates/akita-pcs/tests/akita_fp128_e2e.rs),
  [`akita_small_field_e2e.rs`](../crates/akita-pcs/tests/akita_small_field_e2e.rs), and
  [`recursive_setup_e2e.rs`](../crates/akita-pcs/tests/recursive_setup_e2e.rs)
  for full proof/verification paths.
- [`fold_linf.rs`](../crates/akita-pcs/tests/fold_linf.rs), fold-grind unit tests,
  and selective-L2 cases for acceptance rules and chunk responses.
- [`protocol_soundness.rs`](../crates/akita-pcs/tests/protocol_soundness.rs),
  [`transcript_hardening.rs`](../crates/akita-pcs/tests/transcript_hardening.rs),
  and the suffix tests `non_zk_eor_mismatch_is_rejected` and
  `late_application_batch_rejects_beta_orthogonal_terminal_error` for rejection
  and late-batching behavior.

Existing label tests alone do not establish full prover transcript equivalence.
Add deterministic comparisons of serialized proof bytes and logged transcript
events against the captured baseline, including challenged values and nonce
stream. Cover root-to-terminal directly and a recursive chain; extension and
degree-one claims; supported opening families, payload/relation modes, heterogeneous
groups, setup-prefix use, and terminal binding. Use valid admitted combinations;
do not create an artificial full Cartesian product of unsupported schedules.
Exercise malformed public claim/source alignment at construction, malformed root
commitment rows at scheduled claim binding, and unsupported state consumers before
expensive witness work.

Run repository preflight gates from [AGENTS.md](../AGENTS.md) before compilation.
During implementation use scoped `rtk cargo` / `rtk cargo nextest` runs. Final
validation must use the current test-pass invocation and sharding from
[CI](../.github/workflows/ci.yml), all three mandated Clippy feature graphs, and
any path-triggered portability/Jolt workflows. This documentation-only proposal
requires documentation guardrails; it does not claim those implementation tests
or benchmarks have already run.

### Acceptance criteria

- [ ] `prove` exposes entry/root/suffix/finish ordering in a short body, with
  similarly readable fold preparation and execution drivers.
- [ ] The executor fields are exactly the level stack selector and one grinding
  transcript. Setup, prefix slots, schedule, basis, level, witnesses, and stage
  outputs are not stored on it.
- [ ] Transcript-sensitive stage methods mutate the executor's single transcript;
  no stage accepts an alternative transcript and finishing consumes the executor.
- [ ] Relation construction no longer calls back into its caller for late batching;
  payload-before-batching ordering is explicit and regression-tested.
- [ ] Accepted fold witnesses are reused and heavy kernels remain canonical.
- [ ] Replaced orchestration helpers are deleted; no wrapper-only executor methods,
  duplicate geometry formulas, or compatibility aliases are introduced.
- [ ] Root, recursive, terminal, and independent entry validation remain correct;
  transcript/proof equivalence and existing verification tests pass.
- [ ] Performance and peak-memory comparisons meet the agreed thresholds with
  recorded evidence; there are no unexplained extra witness passes or deep copies.
- [ ] Required CI checks pass and the owning Book pages describe the implemented
  architecture before this spec is marked implemented and later archived.

## Alternatives and review decisions

An executor with optional stage-result fields would shorten call sites but hide
dependencies, permit invalid call order, and retain large buffers too long. A
typestate chain would make ordering explicit but add generic types and ownership
machinery without enough benefit here. Merely wrapping existing functions would
leave the callback, repeated checks, and long fold body intact. Rewriting backend
traits or kernels alongside orchestration would expand the review and obscure
performance attribution. The proposed transcript-owning executor with ordinary
stage results avoids these costs.

The recommendation is to approve the full orchestration scope, including splitting
`RingRelationProver::new`, while keeping kernel optimization separate. Reviewers
should confirm the proposed performance thresholds. The updated base already
removed the low-level prover exports, so this refactor should keep `batched_prove`
as the sole public orchestration boundary.

## Appendix: Jolt prover architecture review

This appendix reviews the local Jolt checkout at
`/Users/omid.bodaghi/Desktop/a16z/jolt`, branch
`codex/stage7-akita-executor-latest`, commit
`e99577c48fcaee46af0578ec955ffed280ec56c4`. The relevant implementation is in
`crates/jolt-prover/src/akita/prover.rs`, `crates/jolt-prover/src/dory/prover.rs`,
`crates/jolt-prover/src/akita/stage0.rs`, `crates/jolt-prover/src/stages/`, and
`crates/jolt-prover/src/driver.rs`. The associated design rationale is in
`specs/prover-stage-drivers.md` and `specs/clean-slate-prover.md` in that checkout.
This records the inspected revision because Jolt is evolving independently.

Jolt's strongest architectural feature is its top-level `prove`: after stage 0,
the function is a literal protocol outline. It invokes `prove_stage1` through
`prove_stage8` in order and assembles the final proof from their outputs. Each
stage has a named output struct. For example, `Stage1ProverOutput` separates the
wire proofs and claims from `Stage1ClearOutput`, the typed carrier later stages
consume. Dependencies are visible as references to earlier carriers rather than
hidden in one mutable protocol-state object.

Akita should borrow that shape with semantic stage names:

| Jolt idea | Application to this proposal |
|---|---|
| A short top-level protocol outline | Keep `ProverExecutor::prove` as root, resource-boundary, suffix, transcript-finish, and proof assembly. Keep the per-fold driver as a similarly direct sequence of preparation, witness commitment, ring switch, and sumcheck stages |
| Named stage output carriers | Replace positional tuples and optional-then-required values with a small number of named results that contain wire material and the exact downstream carry |
| Explicit upstream inputs | Pass prior stage results into the stage that consumes them. Do not place them on `ProverExecutor` or create getters for ambient mutable state |
| One canonical common driver | Keep shared fold and sumcheck mechanics in one implementation, with root/recursive fronts preparing their distinct inputs. Delete old entry points after moving their bodies |
| Shared transcript-sensitive protocol helpers | Continue using canonical transcript labels, encoders, grinding routines, and verifier-shared geometry. Where practical, have prover and verifier call the same ordering helper rather than maintain parallel sequences |
| Proof assembly at the end | Let stage results retain proof fields until final assembly while moving or dropping heavy computation buffers at the earliest safe boundary |
| Stage-local modules | Organize around protocol phases such as `opening`, `fold_response`, `ring_switch`, and `sumcheck`, rather than large catch-all files or numeric stages that are not meaningful in Akita |

Jolt keeps its transcript as a local variable and passes `&mut transcript` to
each stage. This proposal intentionally adapts that choice: Akita's
`ProverExecutor` owns the one grinding transcript, and its stage methods mutate
that field. The architectural lesson retained from Jolt is a single ordered
transcript across a linear stage sequence. Keeping it on the executor makes that
single-channel rule structural and matches the requested Akita API.

Jolt also separates stage-level hand choreography from a common generated
sumcheck driver. Akita does not need Jolt's code generation in this refactor, but
it should apply the underlying rule: stage methods own protocol-specific ordering,
while arithmetic engines and kernels remain canonical lower-level functions. A
stage method should not duplicate challenge derivation, round loops, point
derivation, or claim validation already owned by a shared primitive.

The Jolt backend registry, proof session, kernel preparation traits, recorder
modes, device residency, and backend-specific lifecycle are intentionally not
adopted here. They are relevant to a future backend architecture, but adding them
to this PR would combine orchestration cleanup with a new compute abstraction and
make correctness and performance changes harder to attribute. This PR should only
choose stage boundaries and result types that do not obstruct such a future: pass
sources and canonical values across stages, avoid embedding CPU-specific types in
the executor, and preserve the existing stack/backend interfaces unchanged.

After implementation, update the owning proving page and the relevant
[root/ring-switch](../book/src/how/proving/root-fold-ring-switch.md),
[fold-path](../book/src/how/proving/fold-path.md), and
[sumcheck](../book/src/how/proving/sumcheck-stages.md) explanations. This proposal
does not change Book descriptions of current behavior, wire formats, planner
policy, or the verifier.
