# Native Spongefish proof stream

## Decision

Akita uses Spongefish `ProverState` and `VerifierState` as its only production
Fiat--Shamir state and proof transport. Proof values are emitted with
`prover_message`, public and derived values with `public_message`, and challenges
from the native duplex state. Verification succeeds only after all Akita
relations, the complete grinding plan, and Spongefish `check_eof` succeed.

There is no compatibility promise for earlier proof bytes. The active native
protocol identifier is version 6 and is backend-specific. The pinned Spongefish
revision is `d2d190b1329d35ac9577438d05aed4f17a57b9f9`.

This design preserves proof-of-work grinding, fold-response search, all schedule
modes, and Akita's verifier no-panic contract. Spongefish owns state evolution,
argument I/O, proof-message absorption, challenge extraction, and EOF. Akita
owns the public grammar, canonical codecs, bounded search, schedule validation,
and algebraic checks.

## Security invariants

1. Every prover-controlled value is received and absorbed before any challenge
   that depends on it.
2. The descriptor binds the field, setup, schedule, opening layout, basis,
   grinding policy, protocol features, and codec epoch. Session and descriptor
   bytes are independently length-framed in Spongefish's domain separator.
3. Commitments, points, claims, and derived bindings are public messages at
   their prescribed positions. The descriptor does not replace statement
   binding.
4. The validated public schedule fixes every branch, atom count, and allocation
   bound except the terminal compressed `z` length. Proof input cannot select a
   different grammar.
5. Proof codecs are canonical and injective. Failed receipt or semantic checks
   abort before a subsequent challenge is accepted or used.
6. Verification rejects truncation, extra bytes, wrong sessions/descriptors,
   noncanonical fields or nonces, nonce range failures, incomplete plans, and
   invalid terminal encodings without panic or proof-controlled unbounded
   allocation.
7. The verifier never constructs a prover state and does not require entropy.
   Native prover randomness does not make Akita zero knowledge.

## Positional grammar

The proof is the Spongefish argument string in protocol order. The instance
descriptor commits the complete schedule and protocol version, so the
production sponge uses a positional grammar rather than hashing a large context
record before every operation. This is safe because accepted encodings are
prefix-free under the validated schedule:

- base fields are fixed-width canonical little-endian atoms;
- extensions are fixed ordered sequences of base-field atoms;
- standard and equality-factored sumcheck rounds contain the exact public
  degree-bound coefficient count;
- lower-degree honest round polynomials are padded with high zero coefficients;
- proof-of-work and fold-response nonces are canonical self-delimiting unsigned
  LEB128 atoms and are range-checked against the grinding plan;
- the terminal `z` payload has one native `u32` length, checked against the
  schedule before allocation, followed by exactly that many bytes;
- the Golomb--Rice decoder rejects noncanonical padding, trailing data, wrong
  coordinate counts, and values outside the scheduled cap.

`ProtocolContextRecord` and `ProtocolSiteId` remain diagnostic metadata. With
`logging-transcript`, prover and verifier record and compare the same operation
sequence, atom counts, encoded widths, and challenge candidate widths. These
records are not part of the production sponge state and add no proof bytes or
production hashing work. The versioned protocol identifier and descriptor are
the cryptographic domain boundary.

The protocol order remains:

| Group | Bound values | First dependent operation |
| --- | --- | --- |
| Root statement | commitments, points, claims, basis and call layout | root batching and fold challenges |
| EOR | partial claims, fixed-shape rounds and final claims | EOR challenges and oracle equality |
| Opening payload | scheduled relation payload | row batching and fold-response search |
| Fold response | one bounded LEB128 nonce per fold | every group-local sparse root |
| Stage 1 / L2 | range and norm proof messages | per-round and merge challenges |
| Stage 2 | relation sumcheck messages | per-round challenges and successor binding |
| Stage 3 | setup slot, claim, rounds and prefix evaluation | stage-3 challenges and deferred equality |
| Successor | recursive outer payload or terminal inner state | ring switch and next fold |
| Terminal | retained `t`, folded `e`, response nonce and bounded `z` | direct relation and trace checks |

The terminal decoder performs one canonical Golomb--Rice decode in the direct
relation verifier. It does not decode, re-encode, and compare the same payload
before that check.

## Exact field challenges

Each base-field challenge coordinate uses canonical rejection sampling directly
from Spongefish:

1. squeeze `F::NUM_BYTES`;
2. clear unused high bits above `F::MODULUS_BITS`;
3. accept only a canonical field representative, otherwise repeat.

This distribution is exactly uniform, so there is no modular-reduction bias or
aggregate statistical-distance budget. The admitted native codec supports
fields up to 64 bytes; Akita's production fields use 4, 8, or 16 bytes.
Extension challenges sample their base coordinates independently in fixed limb
order.

The pinned Keccak duplex can forget a prior squeeze length after a later
absorb. Keccak builds therefore absorb the accepted rejection-attempt count
after each coordinate draw, preventing distinct retry histories from
reconverging. Blake2b needs no retry marker and pays no production cost for it.

## Grinding

For every nonzero proof-of-work site, the plan supplies the site, target `g`,
and nonce width `g + 7`. The prover clones only the public sponge state, tries
the bounded nonce range, absorbs the canonical nonce bytes, and squeezes the
32-byte predicate. It accepts when the first `g` little-endian bits are zero.
The live prover emits the winner at the same position; the verifier receives
it, checks the scheduled range and predicate, and only then draws the protected
challenge. Zero-bit sites emit nothing.

Fold-response search uses a distinct plan entry and a 12-bit bounded nonce. One
nonce is shared by all group roots at that fold. Preview cannot mutate the live
state, proof bytes, private RNG, or plan cursor. The verifier reconstructs the
same roots and enforces response representation and norm bounds.

The grinding plan must be consumed exactly once and completely. Any satisfying
in-range nonce is accepted; proof uniqueness is not required. Storage width
does not change work or adversarial-query accounting.

## Planner and schedule contract

Schedule selection intentionally retains main's packed-bit objective:

```text
estimated_nonce_bytes = ceil(sum(semantic_nonce_widths) / 8)
```

`PackedProofCost`, suffix dynamic programming, the unpruned reference search,
parent-alignment dominance, runtime materialization, and generated artifacts
all use that objective. A fresh canonical generation must reproduce the pinned
main schedule artifacts byte-for-byte. Generated files must never be copied or
hand-edited to obtain parity.

The native wire uses individual LEB128 nonce atoms, so the planner objective is
not an exact native-wire estimator. This is accepted accounting debt during
the migration. Parser bounds, nonce ranges, security accounting, and grinding
plan validation continue to use the actual schedule-derived limits and must not
reuse the approximate selection estimate as a safety bound.

## Performance and size contract

The primary parity workload is fp128 one-hot, one polynomial, nv=36, Blake2b,
release mode, 16 prover/multi-verifier threads, and one single-verifier thread.
Compare a pinned `origin/main` binary and candidate binary in alternating warm
runs on the same idle host.

The accepted implementation removes redundant four-byte sumcheck coefficient
counts, uses stack-backed nonce encoding, absorbs fold payloads in batches,
uses exact narrow challenge sampling, and avoids duplicate terminal `z`
decode/re-encode work. Proof-size diagnostics distinguish actual nonce bytes
from the schedule-derived maximum.

The acceptance target is actual proof size no larger than main and no
reproducible verifier slowdown outside interleaved run-to-run variability.
Schedules must remain byte-identical to main. Other fields, dense, grouped,
recursive, Keccak, and malformed-proof paths remain regression coverage even
when nv=36 is the headline comparison.

## Verification and maintenance gates

- Run the repository preflight and all three Clippy feature graphs from
  `AGENTS.md`.
- Regenerate schedules through `scripts/generate-schedule-artifacts.sh`; compare
  all 16 artifacts with the pinned main revision and rerun to check drift.
- Run transcript hardening and mutation tests with logging enabled and disabled
  for Blake2b and Keccak.
- Test canonical field/nonces, fixed-shape sumcheck truncation, bounded terminal
  lengths, Golomb padding, plan under/over-consumption, and EOF rejection.
- Test grinding preview/live/verifier agreement, zero targets, maximum widths,
  exhaustion, fold-response rejection, and unsuccessful-preview immutability.
- Run representative fp32, fp64, fp128, dense, one-hot, grouped, recursive,
  offloaded, persistence, Jolt, and recursion-guest workflows when touched.
- Treat any new raw Spongefish-state access as security-sensitive. Public-state
  cloning is limited to reviewed grinding/fold previews; live state replacement
  or rollback is forbidden.

## Ownership

| Owner | Responsibility |
| --- | --- |
| Spongefish | duplex states, domain/session/instance setup, argument bytes, proof receipt and absorption, challenges, EOF |
| `akita-transcript` | canonical native codecs, protocol epoch, exact field sampling, diagnostic events, narrow public previews |
| `akita-types` | descriptor, validated grammar shapes, compression, grinding plan and completion |
| `akita-config` | trusted schedule and descriptor derivation |
| prover/verifier | protocol equations, message order, bounded searches and semantic checks |
| `akita-challenges` | indexed sparse challenge distribution and rejection policies |

There is one production proof path and no legacy decoder or compatibility
adapter. Any future format change bumps the native protocol/descriptor identity
and updates this specification, tests, measurements, and affected artifacts.
