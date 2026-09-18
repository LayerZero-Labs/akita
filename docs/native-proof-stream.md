# Native proof-stream review

This document is the review map for Akita's Spongefish argument stream. The
validated public schedule determines every branch, count, and allocation bound;
proof bytes never select a layout.

## Grammar

Every logical operation first absorbs a fixed `ProtocolContextRecord` as a
Spongefish public message. It contains the native format domain and version,
the eight-coordinate `ProtocolSiteId`, operation kind, atom count, encoded byte
count, and challenge width. Integers are little endian. Base-field proof atoms
are fixed-width canonical encodings and extension atoms are ordered base-field
coordinates. A terminal `z` payload is the only variable-size proof message: a
native `u32` length is received and checked against the scheduled byte budget
before allocation, followed by exactly that many bytes. Its Golomb--Rice
decoder rejects noncanonical padding, trailing data, and values outside the
scheduled cap. Verification ends with both grinding-plan exhaustion and
Spongefish `check_eof`.

| Family | Value and source of count | Kind | First dependent randomness/check |
| --- | --- | --- | --- |
| Root statement | commitment coefficients and opening points; public claims | public | all root reduction and fold challenges |
| Extension-opening reduction | public openings; scheduled partial/final-claim counts | public/proof | EOR batching and sumcheck challenges |
| Fold binding | derived protocol points and scalar openings | public | application row batching |
| Opening payload | schedule-derived relation payload coefficients | proof | row batching, fold-response search |
| Fold challenge | public sparse-draw context and accepted response nonce | public/proof nonce | indexed sparse roots |
| Next witness | recursive outer payload or terminal inner state, selected by schedule | proof | ring-switch `alpha`, `tau0`, and `tau1` |
| Stage 1 | scheduled sumcheck polynomials, child claims, range image | proof | per-round/interstage challenges |
| Physical L2 | integer norm, subclaims, sumcheck, virtual evaluations | proof | merge/batch challenges and final L2 equation |
| Stage 2 | scheduled sumcheck polynomials and late witness evaluation | proof | per-round challenges and final fused equation |
| Stage 3 | public setup slot, deferred claim, sumcheck, prefix evaluation | public/proof | stage-3 challenges and deferred stage-2 equation |
| Terminal | retained public inner state, folded `e`, response nonce, bounded `z` | public/proof | sparse challenges, direct ring and trace checks |

Proof-of-work sites use a native `u32` nonce when the public grinding target is
nonzero. The verifier range-checks it against the plan's nonce width, checks a
separately domain-separated predicate, and only then draws the protected
challenge. Zero-bit sites emit no nonce and make no state transition.
Fold-response sites also use native `u32` atoms but a distinct operation kind
and the schedule's 12-bit domain.

## Dependency review

- The descriptor binds the field tower, setup, selected schedule, opening
  layout, basis, grinding plan, native format version, and protocol features in
  Spongefish's instance domain. Caller context is independently length-framed
  in the session domain.
- Actual commitments, points, and claimed values are public messages before the
  challenges that authenticate them. Derived values use public messages rather
  than occupying proof bytes.
- The complete opening payload precedes application row batching. An EOR has
  its own earlier batching challenges; the two challenge families do not alias.
- The accepted fold-response nonce precedes every indexed sparse root. Preview
  states clone only public sponge state and cannot modify the live proof, RNG,
  or grinding-plan cursor.
- A successor binding precedes ring switching. On the final predecessor edge,
  the transmitted terminal inner state is retained and later rebound publicly;
  the terminal response is reconstructed with that exact state.
- Stage-3's setup claim is deferred because it depends on stage-2 challenges.
  The verifier first replays stage-2 messages, then verifies stage 3, and only
  then accepts the stage-2 final equation using the authenticated deferred
  claim. No challenge is drawn from an unbound look-ahead value.
- Terminal `e` is received before response-root replay; `z` is received after
  the root and checked against its schedule-bound representation and norm.

## Soundness and malformed-input review

The integration changes Fiat--Shamir transport, not Akita's lattice relations,
sumcheck equations, sparse distribution, or security parameters. Each
extension coordinate challenge is derived from 64 random-oracle bytes. For a
base field of modulus `p`, reduction distance per coordinate is at most
`p / 2^512`; the schedule's certified adversarial query count and total
coordinate count are included by union bound in the existing security budget.
The indexed sparse sampler still uses the fixed 40-byte
`root || little_endian_u64(index)` input and rejection sampling for positions.

Verifier input uses only canonical fixed-width field decoders and the bounded
terminal byte decoder. Counts are computed after public setup/schedule
validation; checked products and sums precede allocation. A malformed atom may
be absorbed before a later semantic rejection, but no subsequent challenge is
accepted or used after that rejection. Truncation, extra bytes, wrong sessions,
wrong descriptors, nonce overflow, plan under/over-consumption, and terminal
binding mismatches reject.

The canonical production boundary is `AkitaCommitmentScheme::batched_prove`
and `AkitaCommitmentScheme::batched_verify`; both operate on one Spongefish
argument byte string. Structured legacy methods remain only while old
diagnostic tests and profiling views are being removed and are not a fallback
accepted by the canonical verifier.

