# Native proof-stream review

Akita's canonical proof is one Spongefish argument string. The validated public
schedule determines every branch, fixed atom count, and allocation bound; proof
bytes never select a layout. Verification requires all algebraic checks,
grinding-plan exhaustion, and Spongefish EOF.

## Grammar

The native protocol uses a descriptor-bound positional grammar:

- base fields are canonical fixed-width little-endian atoms;
- extensions are fixed ordered base-field coordinates;
- sumcheck rounds contain the public degree-bound coefficient count, with
  lower-degree polynomials padded by high zeros;
- proof-of-work and fold-response nonces are canonical unsigned LEB128 values
  checked against their public plan widths;
- the terminal `z` payload is preceded by a native `u32` length that is checked
  against the schedule before allocation;
- canonical Golomb--Rice decoding rejects padding, trailing data, wrong counts,
  and out-of-bound coefficients.

All other vectors have schedule-fixed lengths. The protocol version, field,
schedule, opening layout, basis, feature policy, and grinding plan are committed
by the Spongefish protocol identifier and instance descriptor. Actual public
commitments, claims, points, and derived bindings are absorbed at their
prescribed positions.

`ProtocolContextRecord` is diagnostic metadata. Logging builds record operation
sites, kinds, counts, widths, and challenge candidate widths and require prover
and verifier traces to agree. Production builds do not hash these records; the
fixed positional grammar and versioned descriptor are the cryptographic domain.

## Dependency review

- The complete opening payload precedes application row batching.
- EOR messages precede their own batching and round challenges.
- A fold-response nonce precedes every sparse root derived from it. Preview
  clones public sponge state only and cannot mutate the live proof or plan.
- Successor witness binding precedes ring-switch and successor randomness.
- Stage-3 authenticates the deferred setup claim before the stage-2 terminal
  equation accepts it.
- Terminal `e` precedes response-root replay. The bounded `z` payload is
  canonically decoded once by the direct relation verifier; norm, relation, and
  trace checks remain mandatory.

## Field challenges

Each base-field coordinate is sampled exactly uniformly. Spongefish squeezes
`F::NUM_BYTES`, unused high bits are cleared, and noncanonical candidates are
rejected. Production fp32, fp64, and fp128 fields consume 4, 8, and 16 bytes per
attempt. There is no modular-reduction bias to budget.

The pinned Keccak duplex can forget squeeze length after a later absorb, so the
Keccak backend absorbs the accepted rejection-attempt count after each field
coordinate. This prevents distinct retry paths from reconverging. Blake2b does
not need or perform this additional transition.

## Malformed input and completion

Receipt uses canonical fixed-width field decoders, canonical LEB128 decoding,
and bounded terminal allocation. Semantic failures abort before a later
challenge is used. Truncation, appended bytes, wrong sessions or descriptors,
nonce overflow, plan under/over-consumption, terminal binding mismatch, and
invalid compression reject with an Akita error.

The sole production boundary is `AkitaCommitmentScheme::batched_prove` and
`AkitaCommitmentScheme::batched_verify`. There is no structured-proof fallback
or legacy decoder.

## Size accounting

The planner intentionally retains main's packed aggregate nonce estimate,
`ceil(total_nonce_bits / 8)`, so schedule generation remains byte-for-byte
stable. Native proofs encode nonce messages independently with LEB128. The
planner estimate is therefore a selection objective, not an exact wire-size or
parser-safety bound. Runtime diagnostics report the native maximum and the
actual committed nonce bytes separately.

The canonical design and performance contract are in
[`specs/spongefish-integration.md`](../specs/spongefish-integration.md).
