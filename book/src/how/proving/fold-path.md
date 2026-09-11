# Fold path and field geometry

Akita uses one fold engine for every field tier. The schedule selects the
opening method for each nonterminal fold.

| Opening method | Production use | EOR |
|----------------|----------------|-----|
| `SubringCoefficientPacking` | Required for nonterminal folds at absolute levels 0 and 1 | omitted |
| `EvaluationTrace` | Later nonterminal folds and the terminal path | required only when the claim field is a proper extension |

See [base-field coefficients vs extension evaluation points](../../foundations/rings-and-fields.md#base-field-coefficients-vs-extension-evaluation-points).
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

1. `prove_root` or the recursive suffix prepares the scheduled opening.
2. The schedule dispatches to coefficient packing or evaluation trace.
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

## Implementation map

- `crates/akita-prover/src/protocol/core/fold/`.
- `crates/akita-verifier/src/protocol/core/fold/`.
- `crates/akita-types/src/layout/params/precommitted.rs`.
- `crates/akita-types/src/subring_coefficient_packing.rs`.
