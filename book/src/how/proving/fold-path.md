# Fold path and field geometry

Akita uses one fold engine for every field tier. The schedule selects the
opening method for each nonterminal fold.

| Opening method | Production use | EOR |
|----------------|----------------|-----|
| `SubringCoefficientPacking` | Required at levels 0 and 1 and every multi-chunk fold; optional at later single-chunk folds | omitted |
| `EvaluationTrace` | Eligible single-chunk folds at level 2 or later and the terminal path | required only when the claim field is a proper extension |

See [Coefficients and evaluation points have different roles](../../foundations/rings-and-fields.md#coefficients-and-evaluation-points-have-different-roles).
For the packing path, read the three obligations in order: the [direct scalar
opening](./field-ring-reduction.md#subring-coefficient-packing-shorter-partials),
the [source/fold consistency
identity](./akita-fold.md#subring-coefficient-packing-consistency), and the
[physical quotient and ring
switch](./root-fold-ring-switch.md#subring-coefficient-packing).

The extension degree fixes the number of field coordinates in one claim. It
does not select the opening method. In particular, an fp128 schedule may use
coefficient packing even though its extension degree is one. That choice can
reduce the D input width even though there is no EOR proof to remove.

## Single-field path

When `EXT_DEGREE == 1`:

1. `prove_root` prepares coefficient packing for every root group.
2. Recursive folds use packing at level 1 and every multi-chunk fold; later
   single-chunk folds may dispatch to packing or evaluation trace.
3. Shared `prove_fold` runs the ring relation, ring switch, and sumchecks.
4. No EOR payload is produced because the extension degree is one.

The verifier mirrors this path. A scalar root is the one-group case of the
grouped root verifier.

## Extension claim path

When `EXT_DEGREE > 1`, the schedule still selects the method. Coefficient
packing opens the original extension valued claim directly and emits no EOR.
Evaluation trace uses EOR to bridge base field coefficients to the extension
valued opening. The verifier accepts an EOR payload if and only if the scheduled
method is evaluation trace and the extension degree is greater than one. See
[Extension-opening reduction](./extension-opening-reduction.md).

Every group consumed by one fold uses the same method family. Packing groups
may use different challenge subring dimensions. A setup prefix inherits the
method selected by the fold that consumes it. The commitment itself does not
record that method.

For coefficient packing, ring-switch finalization prepares one compact batch
of checked group semantics. The batch retains shared sparse challenges, alpha
powers, row weights, gadgets, and witness addresses. It supplies both the
Stage 2 opening terms and the CPU relation events. The CPU calls
`coefficient_packing_relation_events` on those groups when it needs expanded
weights; shared preparation does not evaluate the challenges or allocate events.
This keeps both consumers tied to the same fold inputs while allowing other
backends to consume the compact batch directly. The verifier evaluates its
challenges once per group when building compact factors.

## Implementation map

- `crates/akita-prover/src/protocol/prove/fold/`.
- `crates/akita-verifier/src/fold/` and `crates/akita-verifier/src/stages/`.
- `crates/akita-params/src/layout/params/precommitted.rs`.
- `crates/akita-params/src/layout/subring_packing_geometry.rs`.
- `crates/akita-types/src/subring_coefficient_packing.rs`.
