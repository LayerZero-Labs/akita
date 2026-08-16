# Terminal verification

The final fold does not create another recursive witness. The verifier decodes
the terminal response and checks the remaining relations directly.

## Terminal relation

The terminal relation contains only:

```text
consistency | A
```

There is no outer commitment value, B block, D block, quotient sumcheck, Stage
1, Stage 2, or Stage 3. The predecessor transcript state and schedule determine
the exact terminal shape.

The response contains folded `Z` coordinates and the opening and inner
commitment values needed by the direct checks.

## Canonical decoding

`Z` coordinates use the schedule-selected Golomb Rice encoding. The decoder
checks:

- the exact coordinate count;
- the maximum quotient implied by the signed coefficient cap;
- canonical zero padding in the final byte;
- absence of trailing bytes; and
- conversion to the required signed 16 bit coefficient class.

Values outside `[-32768,32767]` are rejected before ring arithmetic. The
verifier does not accept an alternate terminal coefficient representation.

## Consistency products

Terminal fold challenges are sparse. The verifier multiplies decoded `Z` rings
by the canonical signed sparse challenges through checked negacyclic shifts.
It does not expand each challenge into a dense ring and run a quadratic
schoolbook product.

Coefficients `1`, `-1`, `2`, and `-2` use ring addition, subtraction, and
doubling. Valid custom coefficients use the exact field scaling fallback.

## A matrix check

The A relation multiplies the decoded terminal witness by the prepared public
matrix. The schedule audit selects an exact CRT and NTT capability before proof
replay. It uses the base prime profile when the signed accumulation bound fits
and adds the signed 16 bit tail only when required.

A schedule whose bound exceeds every supported exact profile is rejected as an
invalid setup. Prepared matrix views are derived from the coefficient setup and
are never serialized.

## Extension opening trace

For extension field openings, fold and position weights stay in canonical
subfield coordinates. The verifier recovers extension values and trace inner
products directly. It does not materialize one dense ring for each weight.

The inverse subfield map validates the complete canonical image. A malformed
ring or coordinate shape returns `AkitaError` instead of being projected into
an accepted value.

## No-panic boundary

The verifier validates payload lengths, ring dimensions, coordinate counts,
sparse support, NTT capability, and matrix ranges before the hot kernels index
prepared state. Malformed terminal bytes return `AkitaError` or
`SerializationError`.

## Code map

- Terminal orchestration:
  `crates/akita-verifier/src/protocol/core/terminal_direct.rs`.
- Exact matrix path:
  `crates/akita-verifier/src/protocol/core/terminal_ntt.rs`.
- Golomb Rice codec: `crates/akita-types/src/golomb_rice.rs`.
- Compact opening points and terminal payloads:
  `crates/akita-types/src/proof/batch.rs` and
  `crates/akita-types/src/proof/tail_segments.rs`.
- Subfield trace arithmetic: `crates/akita-types/src/field_reduction.rs`.
