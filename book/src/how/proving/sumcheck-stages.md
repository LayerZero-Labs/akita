# Sumcheck stages

Each nonterminal fold uses two required sumchecks and, when the schedule
offloads setup evaluation, one optional sumcheck:

1. **Stage 1** proves the digit-range relation for the balanced witness.
2. **Stage 2** proves the fused virtual claim, relation matrix, and evaluation
   trace claim.
3. **Stage 3** proves the setup product and carries the next-witness opening
   when setup contribution is recursive.

The schedule determines whether Stage 3 is present. The proof object and the
successor-owned setup-prefix edge remain the execution-path authority.

## One setup plan for both Stage 2 paths

Once the Stage 2 relation point is known, the prover and verifier prepare a
`SetupContributionPlan`. This immutable semantic plan represents every D, B,
and A setup contribution as a typed affine span over:

- relation rows;
- witness-opening addresses;
- packed setup addresses;
- the role projection from D, B, or A to the base setup ring; and
- the A-role fold-gadget range.

The plan also retains the validated setup domain, equality window, column
challenges, and fold geometry needed to evaluate those spans. It does not
retain dense relation-column weight vectors, copied row weights, direct-scan
segments, setup coefficients, or a scheduling mode.

Both direct and recursive Stage 2 execution first evaluate the structured E, T,
and Z terms from these spans. The paths differ only in how they discharge the
setup term:

```text
prepare semantic spans
        |
        +-- evaluate structured E/T/Z terms
        |
        +-- direct: compile a private fused scan and contract the setup
        |
        +-- recursive: use the transcript-bound setup claim
                       and pass the exact plan to Stage 3
```

Mixed role dimensions use the same spans. Each contraction applies the role's
local alpha lanes while the structured relation slices the common inner-lane
geometry by the span's subcolumn. This keeps D, B, and A projection in one
checked representation.

## Direct setup contraction

The direct path derives scan segments only when
`SetupContributionPlan::evaluate_direct` is called. The backend sorts and fuses
the span-derived work so the expanded setup is scanned at most once for one
relation evaluation. Those segments are disposable execution state, not a
second representation of setup geometry.

## Recursive Stage 3 contraction

Recursive Stage 2 verifies the same structured relation but substitutes the
setup claim absorbed by the transcript. After Stage 2 succeeds, the verifier
moves its exact plan into `SetupSumcheckVerifier`; Stage 3 cannot rebuild it
from `RelationMatrixEvaluator` or the original layouts.

The Stage 3 final check combines:

- the selected setup-prefix multilinear evaluation;
- the plan's setup-index-weight multilinear evaluation;
- the alpha-coordinate evaluation; and
- the equality-weighted next-witness carry term.

The verifier evaluates the setup-index weight directly from the affine spans
with a compact recurrence. It does not materialize the dense weight vector or
scan the active setup for that factor.

The prover needs the same polynomial as a dense term in its Stage 3 product. It
calls `SetupContributionPlan::materialize_setup_index_weights`, which fills one
vector from the canonical spans. Direct contraction, point contraction, and
prover materialization are distinct kernels over the same semantic source.

## Protocol stability

This plan ownership is internal. It does not change proof bytes, transcript
labels, challenge order, setup-prefix selection, or the Stage 2 and Stage 3
equations. The deterministic `fold_protocol_epoch` integration test pins direct
and recursive proof encodings and logging-transcript event streams.

**Sources to fold in**

- `crates/akita-prover/src/protocol/sumcheck/digit_range/`, `akita_stage2/`, and `two_round_prefix/`.
- `crates/akita-verifier/src/stages/`.
- Paper §3.5 (`fig:akita-sumcheck`), §3.5.1 `sec:akita-range-check` (optimized digit range check), §4.3 `sec:claim-reduction` (setup product sumcheck).
- `specs/packed-sumcheck.md`, `specs/setup-product-sumcheck.md`.
- `specs/archive/2026-Q3/setup-contribution-pipeline-unification.md`.
