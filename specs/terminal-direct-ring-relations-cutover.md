# Terminal Direct Ring Relations Cutover

| Field | Value |
| --- | --- |
| Base | PR #294 (`refactor/universal-digit-fast-layout`) |
| Branch | `quang/terminal-direct-ring-relations` |
| Status | implementation scaffold |
| Supersedes | the implementation plan in PR #141 |

## Goal

Replace the transparent folded terminal's quotient-to-sumcheck relation proof
with deterministic checks of the revealed witness in
`F[X] / (X^D + 1)`.

The cutover removes, at the terminal only:

- every relation quotient in the raw-field `r` tail;
- terminal `CHALLENGE_RING_SWITCH`, `CHALLENGE_TAU1`, and stage-2 challenges;
- the terminal stage-2 sumcheck proof and prover/verifier work.

Intermediate folds are unchanged. They retain committed recursive witnesses,
the shared quotient tail, stages 1--3, and their current transcript schedule.
The terminal `t`-state / final-`u` cutover is a follow-on PR.

Akita has no backward-compatible proof decoder. This is a hard transparent
protocol cutover, not a runtime legacy/direct toggle.

## Current #294 Baseline

The terminal fold currently uses
`RelationMatrixRowLayout::WithoutDBlock`. Its physical rows are:

```text
consistency | A | B
```

The cleartext terminal witness is:

```text
SegmentTyped(z, e, t, r)
```

`r` contains one raw ring element for every physical terminal relation row.
The prover nevertheless digit-decomposes those rows into the logical witness
used by stage 2. After binding the terminal witness, prover and verifier sample
`alpha` and `tau1` and run a relation-only stage-2 sumcheck.

The public opening is not a physical `y` row. It is bound by the logical
`EvaluationTrace` row fused into stage 2. Removing stage 2 without replacing
that trace check is unsound.

## Direct Terminal Statement

The terminal verifier performs two checks over the same transcript-bound
cleartext witness.

### 1. Reduced ring relations

For every physical terminal row, check:

```text
M_terminal * w_terminal == y_terminal  in F[X] / (X^D + 1)
```

The checker must reproduce the current authoritative row semantics from
`compute_multi_group_relation_quotient`, but calculate only reduced products:

- consistency row;
- A rows;
- B commitment rows;
- no D rows.

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

Segment-typed witnesses must use the existing checked `logical_i8_digits`
expansion and trace-table primitives. Do not use the root-direct
`FieldElements`-only opening helper.

## Transcript

The common prefix through the terminal witness remains unchanged:

```text
terminal e bytes
sparse-challenge context and challenge
fold challenges
terminal witness remainder bytes
```

After the remainder, the transparent terminal performs the two deterministic
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

```text
r_field_elems = 0
r logical digit planes = 0
```

The realized tail is `SegmentTyped(z, e, t)`. The intermediate recursive
`WitnessLayout` continues to own a mandatory shared `r_range`; do not make all
recursive witness layouts optional merely to express the terminal cutover.

The terminal layout/shape derivation should take an explicit terminal relation
policy at its owning boundary, or be replaced with a direct-only terminal
derivation. It must not infer quotient presence from an empty payload supplied
by the prover.

## Prover Ownership

Split terminal construction before quotient materialization:

1. Build and retain the checked `z`, `e`, and `t` terminal artifacts.
2. Build `SegmentTyped(z, e, t)` with the schedule-derived shape.
3. Bind the canonical terminal transcript parts.
4. Do not call `compute_multi_group_relation_quotient` for the terminal.
5. Do not emit quotient digit planes.
6. Do not call `ring_switch_finalize` for the terminal.
7. Do not run terminal stage 2.
8. Emit the direct `TerminalLevelProof`.

Intermediate construction remains on the current `ring_switch_build_w` and
`ring_switch_finalize` path. Prefer a terminal-specific builder over boolean
branches that leave impossible quotient/stage-2 values in common output
structs.

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
