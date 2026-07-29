# Fold path and field geometry

> **Status:** stub. Part of the protocol field-geometry cutover
> ([`specs/protocol-field-geometry-cutover.md`](../../../specs/protocol-field-geometry-cutover.md)).

Akita’s fold engine is shared, but the **opening geometry** depends on whether
the claim field coincides with the coefficient field.

| Geometry | Criterion | Production presets | EOR |
|----------|-----------|--------------------|-----|
| **Single-field** | `CommitmentConfig::EXT_DEGREE == 1` (`Field` plays both roles) | `fp128::*` | never |
| **Extension-claim** | `EXT_DEGREE > 1` (claims in a proper extension of the coefficient field) | `fp32::*`, `fp64::*` | root when enabled; suffix always |

See [base-field coefficients vs extension evaluation points](../../foundations/rings-and-fields.md#base-field-coefficients-vs-extension-evaluation-points).

## Single-field path (auditor walk)

When `EXT_DEGREE == 1`:

1. `prove_root` / recursive suffix → `prepare_single_field_fold`
2. Shared `prove_fold` (ring relation, ring switch, stages 1/2/3)
3. No imports of extension-opening reduction or root tensor projection on that prep path

Verifier mirrors: single-field prefixes prepare opening points without calling
`verify_fold_eor`.

## Extension-claim path

When `EXT_DEGREE > 1`, prep lives in `prepare_extension_claim_fold` /
matching verifier prefixes. Extension-opening reduction and (when required)
root tensor projection bridge \(F\)-coefficient witnesses to \(E\)-valued
openings. Details: [Extension-opening reduction](./extension-opening-reduction.md).

## Sources to fold in

- `crates/akita-prover/src/protocol/core/fold/{single_field,extension_claim,mod}.rs`
- `crates/akita-verifier/src/protocol/core/fold/{single_field,extension_claim,mod}.rs`
- `akita_types::eor_required_at_level` / `FoldOpeningKind`
