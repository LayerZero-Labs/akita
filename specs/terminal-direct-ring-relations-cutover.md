# Terminal Direct Ring Relations Cutover

| Field | Value |
| --- | --- |
| Base | PR #294 (`refactor/universal-digit-fast-layout`) |
| Branch | `quang/terminal-direct-ring-relations` |
| Status | implemented |
| Supersedes | the implementation plan in PR #141 |

## Goal

Replace the transparent folded terminal's quotient-to-sumcheck relation proof
with deterministic checks of the revealed witness in
`F[X] / (X^D + 1)`.

The cutover removes, at the terminal only:

- every relation quotient in the raw-field `r` tail;
- terminal `CHALLENGE_RING_SWITCH`, `CHALLENGE_TAU1`, and stage-2 challenges;
- the terminal stage-2 sumcheck proof and prover/verifier work.

Ordinary intermediate edges are unchanged: they retain an outer `u`, the
shared quotient tail, and stages 1--3. The final intermediate edge into a
suffix terminal is the one exception: it binds the following terminal
witness's canonical inner `t` state and omits the redundant outer `u`.

Akita has no backward-compatible proof decoder. This is a hard transparent
protocol cutover, not a runtime legacy/direct toggle.

## Historical #294 Baseline

Before this cutover, every terminal fold used
`RelationMatrixRowLayout::WithoutDBlock`. Its physical rows are:

```text
consistency | A | B
```

The cleartext terminal witness was:

```text
SegmentTyped(z, e, t, r)
```

`r` contained one raw ring element for every physical terminal relation row.
The prover nevertheless digit-decomposed those rows into the logical witness
used by stage 2. After binding the terminal witness, prover and verifier sampled
`alpha` and `tau1` and ran a relation-only stage-2 sumcheck.

The public opening was not a physical `y` row. It was bound by the logical
`EvaluationTrace` row fused into stage 2. The current direct checker replaces
that dependency with the explicit trace check below.

## Direct Terminal Statement

The terminal verifier performs two checks over the same transcript-bound
cleartext witness.

### Extension-opening reduction remains independent

When the current fold requires extension-opening reduction (EOR), its partials
and sumcheck remain in the terminal proof. EOR proves that the original
extension-field opening claims reduce to the packed protocol point and final
claim consumed by the ring fold. Revealing the terminal `z`, `e`, and `t`
segments does not reveal the pre-reduction polynomial table, so the verifier
cannot reconstruct this reduction from the terminal witness alone.

EOR is verified before direct terminal relations exactly as it is before an
intermediate fold. The direct trace-opening check uses:

- the EOR final claim as its target and the EOR equality-factor evaluation as
  its scale when EOR is present;
- the ordinary batched opening target and unit scale otherwise.

Only the terminal relation sumcheck is removed. The EOR sumcheck is neither a
relation quotient check nor redundant with revealing the terminal witness.

### 1. Reduced ring relations

For every physical terminal row selected by the schedule, check:

```text
M_terminal * w_terminal == y_terminal  in F[X] / (X^D + 1)
```

The checker must reproduce the current authoritative row semantics from
`compute_multi_group_relation_quotient`, but calculate only reduced products:

- consistency row;
- A rows;
- B commitment rows for a root terminal only;
- no D rows.

A root terminal still receives the external public `u`, uses
`WithoutDBlock = consistency | A | B`, and must retain the B equation. A suffix
terminal receives public inner `t` from its predecessor, uses
`WithoutCommitmentBlocks = consistency | A`, and has no `u` or B equation. The
suffix cutover is sound because `t` replaces `u` as the public state; merely
deleting B while continuing to accept `u` would be unsound.

The checker must use role-local ring dimensions and the canonical row ranges
from `LevelParams`. It must not infer offsets independently or route through a
second relation-layout implementation.

### 2. Explicit trace opening

Since direct mode removes the fused `EvaluationTrace` row, separately check:

```text
witness_trace_eval == trace_eval_target
```

`trace_eval_target` is derived exactly as in the existing terminal stage-2
path:

- ordinary openings use the batched opening target;
- when extension-opening reduction is present, verify it first and use its
  final claim;
- apply the existing recursive/root trace scales.

Segment-typed witnesses use the checked direct trace evaluator over revealed
`e`; they do not reconstruct the removed stage-2 logical digit stream. Do not
use the root-direct `FieldElements`-only opening helper.

## Transcript

Both terminal forms bind raw `e_folded` before the sparse challenge, but their
state/response split is intentionally different.

For a suffix terminal, the predecessor binds canonical `t` bytes under
`ABSORB_NEXT_LEVEL_WITNESS_BINDING` before its ring-switch challenges. The
terminal then rebinds the same `t` as its current state under
`ABSORB_COMMITMENT`, followed by:

```text
terminal t state
terminal e_folded bytes
sparse-challenge context and challenge
fold challenges
terminal z response bytes
```

For a root terminal, the ordinary root commitment prefix already binds the
external `u`. The terminal binds `e_folded` before the sparse challenge and
absorbs `z || t` afterward; the retained B rows connect that `t` to `u`.

After the response, the transparent terminal performs the two deterministic
checks immediately. It does not sample terminal ring-switch aggregation,
`tau1`, sumcheck batching, or sumcheck-round challenges.

The descriptor version must change with this transcript schedule. No separate
mode bit is needed because the old transparent terminal is removed rather than
kept as an alternate policy.

## Proof And Shape Cutover

The terminal proof body becomes:

```rust,ignore
pub struct TerminalLevelProof<F, E> {
    pub extension_opening_reduction: Option<ExtensionOpeningReductionProof<E>>,
    pub fold_grind_nonce: u32,
    pub final_witness: CleartextWitnessProof<F>,
}
```

There is no empty stage-2 proof and no empty sumcheck sentinel. The terminal
shape mirrors the body and contains only the optional reduction shape and the
cleartext-witness shape. Fold-rooted terminal steps and terminal-root proofs
must share this one representation.

Headerless decoding remains shape-driven. Malformed shapes, segment counts,
row counts, ring dimensions, and arithmetic overflow reject with
`SerializationError` or `AkitaError`; verifier-reachable code must not panic.

## Tail And Witness Layout

For the transparent terminal:

The realized tail is structurally `SegmentTyped(z, e, t)`, with no quotient
field or quotient-mode tag. The intermediate recursive
`WitnessLayout` continues to own a mandatory shared `r_range`; do not make all
recursive witness layouts optional merely to express the terminal cutover.

The execution schedule owns the terminal row layout: root terminal keeps B,
suffix terminal drops all commitment rows. The proof variant alone cannot
derive that context. Layout/shape derivation must not infer quotient or
commitment presence from an empty prover payload.

## Prover Ownership

Split terminal construction before quotient materialization:

1. Build and retain the checked `z`, `e`, and `t` terminal artifacts.
2. Build `SegmentTyped(z, e, t)` with the schedule-derived shape.
3. Bind canonical transcript parts according to schedule topology: pre-bound
   `t` plus post-challenge `z` for a suffix, or post-challenge `z || t` under
   the root's retained external `u`/B statement.
4. Do not call `compute_multi_group_relation_quotient` for the terminal.
5. Do not emit quotient digit planes.
6. Do not call `ring_switch_finalize` for the terminal.
7. Do not run terminal stage 2.
8. Emit the direct `TerminalLevelProof`.

Intermediate construction remains on the current `ring_switch_build_w` and
`ring_switch_finalize` path. Prefer a terminal-specific builder over boolean
branches that leave impossible quotient/stage-2 values in common output
structs.

The cut must occur before `compute_multi_group_relation_quotient`, recursive
digit-witness allocation, `ring_switch_finalize`, and stage-2 trace-table
construction. Terminal construction returns only checked group artifacts and
the cleartext witness; it must not manufacture empty recursive-witness,
quotient, evaluator, or sumcheck objects to satisfy the intermediate API.

## Verifier Ownership

Add one canonical checker under `akita-verifier::protocol::core`. It accepts
already-decoded and schedule-validated terminal data and performs both direct
checks.

It must serve:

- a terminal step at the end of a recursive suffix;
- a one-fold terminal-root proof.

The entry points may prepare different opening-incidence inputs, but must call
the same row checker. Reduced cyclotomic helpers may be shared with the
root-direct verifier; row construction and checked segment slicing must not be
duplicated between the two terminal surfaces.

### Arithmetic and cache boundary

The first implementation uses the same checked plain cyclotomic mat-vec kernel
as root-direct commitment verification. This kernel consumes
`FlatMatrix::ring_view` slices from the seed-validated expanded verifier setup;
it does not accept caller-supplied matrix storage or duplicate A/B setup
derivation. The terminal checker always dispatches A at its inner role
dimension and, for root terminals only, dispatches B at its outer role
dimension. It uses canonical schedule-selected row ranges and `LevelParams`
row/column sizes.

Do not make `akita-verifier` depend on `akita-prover` to reuse its compute
backend. The prover NTT cache currently stores both negacyclic and cyclic
transforms of the full shared matrix and is intentionally much larger than the
coefficient setup. Direct verification needs only negacyclic products over the
exact A prefixes and, for root terminals, B prefixes; copying that cache into
`AkitaVerifierSetup` would inflate
memory and verifier code without establishing a measured need.

Benchmark the checked plain kernel first. If it is material, factor a
negacyclic-only prepared-matrix primitive into a shared lower-level crate and
place its derived, non-serialized cache in a separate prepared-verifier
artifact. Its identity must bind the setup envelope, ring dimension, and exact
matrix-view length, and its reported size must count only the warmed A and
root-terminal B slots.
The serialized `AkitaVerifierSetup` remains the canonical seed-derived setup,
not a container for derived acceleration state.

This keeps the trusted arithmetic surface to one direct mat-vec primitive,
already exercised by root-direct verification, plus the terminal-specific row
orchestration and trace check.

## Planner And Proof Size

Remove:

```text
terminal_stage2_sumcheck_bytes
+ terminal_relation_row_count * D * field_bytes
```

Keep the terminal fold-grind nonce and optional extension-opening-reduction
bytes. `direct_witness_bytes` prices `SegmentTyped(z, e, t)`; terminal
`level_proof_bytes` prices only the nonce plus optional reduction payload.

Regenerate every affected shipped schedule and descriptor digest. The exact
runtime proof size must continue to equal the schedule/accounting result.

## Implementation Slices

1. **Types and wire**
   - detach `final_witness` from `AkitaTerminalStage2Proof`;
   - remove terminal sumcheck fields from proof and shape serialization;
   - bump descriptor version and update canonical-byte pins.
2. **Terminal tail**
   - derive zero `r` fields/planes;
   - construct and decode `SegmentTyped(z, e, t)`;
   - keep intermediate quotient layouts unchanged.
3. **Prover cutover**
   - terminal builder stops before quotient and stage 2;
   - preserve transcript prefix byte-for-byte.
4. **Verifier cutover**
   - direct reduced row checker;
   - explicit EOR-aware trace-opening checker;
   - shared suffix-terminal and terminal-root dispatch.
5. **Planning and rollout**
   - exact proof-size accounting;
   - generated schedules and report attribution;
   - docs and full validation.

## Required Tests

- reference direct-row equality against the existing quotient relation for
  every supported ring dimension and row role;
- tamper `z`, `e`, `t`, the public commitment rows, and the trace target;
- ordinary and extension-opening-reduction paths;
- suffix-terminal and terminal-root proofs;
- an exact two-fold `root -> suffix terminal` proof where tampering the
  transcript-bound `t` rejects without a B block;
- scalar, multipoint, mixed-ring-dimension, and multi-chunk configurations
  that can reach the terminal;
- malformed segment/layout fuzz and no-panic coverage;
- transcript event test proving terminal `alpha`, `tau1`, and stage-2 events
  are absent;
- proof-size equality and generated-schedule drift guards.

## Acceptance

```bash
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test
./scripts/check-doc-guardrails.sh
```

The benchmark report must attribute separately:

- raw terminal `r` bytes removed;
- terminal stage-2 bytes removed;
- prover quotient/stage-2 time removed;
- direct verifier row/trace-check time added.
