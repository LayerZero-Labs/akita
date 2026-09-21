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
attempt. The shared support predicate requires 1 through 64 canonical bytes, a
positive modulus bit length that fits that width, and fewer than eight unused
high bits. Descriptor construction and direct generic draws both reject an
unsupported codec instead of reducing biased bytes modulo the field. There is
no modular-reduction bias to budget.

The pinned Keccak duplex can forget squeeze length after a later absorb, so the
Keccak backend absorbs the accepted rejection-attempt count after each field
coordinate. Counter overflow is an error. This prevents distinct retry paths
from reconverging. Blake2b does not need or perform this additional transition.
The declared query budget bounds protocol oracle queries, not local rejection
attempts.

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

Recursive decoders use `expanded_schedule_native_proof_bound`, which replaces
the packed nonce estimate with the sum of canonical LEB128 maxima and uses the
scheduled terminal response cap. The planner estimate and native parser bound
remain separate so wire hardening cannot silently change schedule rows.

## Adversarial coverage

Native tests cover the old verifier obligations at the byte-stream boundary:

| Obligation | Native coverage |
| --- | --- |
| Fixed-shape sumcheck rounds | exact public coefficient counts, truncation, role-labelled challenge order, fuzzed canonical receipt |
| Root and recursive payload binding | logged family ranges, statement/session mutation, truncation, successor binding and EOF |
| Extension-opening reduction | native prefix/final-claim grammar, truncation, and end-to-end fp32 extension proofs |
| Stage-1 range and L2 claims | role-specific native codecs plus end-to-end family mutation and algebraic verification |
| Terminal response | bounded length before allocation, canonical Golomb--Rice decode, family mutation, direct norm/relation/trace checks |
| Grinding | canonical and malformed nonce codecs, 4095/4096 fold boundary, range and predicate checks, preview/live/verifier agreement, plan completion |

Byte mutation checks establish binding and parser rejection. Algebraic unit and
end-to-end tests remain responsible for the corresponding equations; a byte
flip is not used as a substitute for an equation-specific test.

The canonical design and performance contract are in
[`specs/spongefish-integration.md`](../specs/spongefish-integration.md).
