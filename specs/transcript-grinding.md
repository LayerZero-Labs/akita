# Spec: Transcript Grinding

| Field       | Value                         |
|-------------|-------------------------------|
| Author(s)   | Quang Dao, Cursor assistant   |
| Created     | 2026-05-22                    |
| Status      | proposed                      |
| PR          |                               |

## Summary

Akita should first fix terminal-fold transcript binding, then add native
Plonky3-style transcript grinding at Fiat-Shamir challenge sites whose
soundness error consumes a small number of bits. The prover searches for a
compact nonce that makes a transcript-derived predicate hold, the verifier
checks that nonce before sampling the actual protocol challenge, and the
configured loss bits are restored so every protected challenge site targets 128
bits of security.

## Intent

### Goal

Build Akita-native transcript grinding with a proof-owned fixed-width nonce
stream, shape-aware serialization, verifier replay, and planner/security
accounting for all loss-bearing sumcheck, ring-switch, multilinear-point, and
batching Fiat-Shamir challenge families. As part of the same cutover, make
terminal fold replay a separate transcript path with explicit terminal `w_hat`
binding and no terminal `tau0` squeeze.

The primary new abstractions are:

- A compact grinding proof stream with one little-endian `u16` nonce per
  nonzero grinding invocation.
- Transcript APIs for byte-granular prover grinding and verifier checking.
- A central grinding policy that converts per-site soundness loss into
  required grinding bits and rejects sites too large for the fixed nonce budget.
- A transcript cursor/journal layer that avoids current absorb/squeeze
  footguns and makes grinding a first-class transcript transition.

### Invariants

- Prover and verifier replay order is:
  1. absorb all public/prover messages that precede the challenge,
  2. absorb/check the grinding nonce when the site has nonzero grinding bits,
  3. squeeze and consume the grinding predicate bits,
  4. squeeze the protocol challenge.

- Grinding consumes transcript state. The predicate squeeze is not reused as
  the protocol challenge. This follows the Plonky3 pattern where
  `check_witness` observes the witness and samples proof-of-work bits before
  the next Fiat-Shamir challenge is sampled.

- Production transcript labels remain diagnostics only. The grinding method
  must absorb a fixed, prefix-free Akita grinding payload containing a version
  tag and canonical nonce bytes; it must not rely on semantic labels entering the
  production sponge.

- Any new labels introduced by this spec are logging/test event names only.
  They must not change production challenge bytes. In production,
  `AkitaTranscript` remains positional: only the ordered framed payload bytes
  and spongefish domain/instance binding enter the sponge.

- Zero grinding bits are a no-op: the prover emits no witness, the verifier
  consumes no witness, and the transcript state is unchanged.

- Zero-bit challenge sites must use an explicit ungrounded path: no nonce
  read/write, no grinding payload absorb, no predicate squeeze, and no
  `Grind` logging event. This is the default for challenge families whose
  soundness accounting is already independent of field-size loss, such as the
  current ring sparse / low-norm challenge samplers.

- Nonzero grinding bits require exactly one nonce per grinding invocation.
  The nonce is read as a little-endian `u16` from the proof-level grinding
  stream. Verifier replay rejects missing, extra, malformed, or invalid nonces
  with `AkitaError`, not a panic.

- The first implementation supports `0..=9` grinding bits per invocation.
  Policies that request more than 9 bits must fail during config/shape
  derivation or verifier setup. A 16-bit nonce gives per-site search failure at
  most about `2^-185` for 9 grinding bits, so hundreds of grinding sites still
  remain below a global `2^-128` completeness-error target.

- Proof bodies remain canonical and headerless where they are headerless
  today. The proof carries one canonical `u16` nonce stream, and verifier
  replay derives the exact read schedule from descriptor-bound policy data.
  Shape and policy validation reject proof/policy mismatches.

- Transcript logging tests must treat grinding as a first-class event. Logging
  must show prover/verifier event equality and preserve the existing
  wire-before-squeeze smell checks.

- Direct/root-direct proofs that do not sample a protected challenge carry an
  empty grinding stream and must remain unchanged except for shape enums if a
  shared type requires it.

- Intermediate and terminal folds are distinct protocol paths. Terminal code
  must not call a shared ring-switch helper that unconditionally samples
  `tau0`; terminal has no stage-1 check, so `tau0` has no mathematical role and
  must not be squeezed.

### Non-Goals

- Do not import or depend on `spongefish-pow`. Its current crate is
  experimental and is not integrated with Akita's positional transcript.

- Do not switch Akita to Plonky3's challenger traits or sponge model.

- Do not preserve legacy proof byte compatibility. Akita makes no
  backward-compatibility guarantee, and this is a deliberate proof-format
  cutover.

- Do not add compatibility shims, deprecated aliases, or dual verifier paths.

- Do not hand-tune per-callsite constants inline. Loss-to-grinding-bit
  decisions must flow through the central policy helpers.

- Do not use grinding to compensate unrelated security margins such as SIS
  parameter selection. Grinding only compensates Fiat-Shamir challenge
  soundness loss.

- Do not equate "sampled from the transcript" with "needs grinding." Degree-one
  batching, singleton rows, terminal deterministic coefficients, and sparse
  challenge seeds with independently sufficient min-entropy consume zero
  grinding bits.

## Evaluation

### Acceptance Criteria

- [ ] `akita-transcript` exposes prover and verifier grinding APIs with a
      documented payload format, bit predicate, byte-granular squeeze behavior,
      9-bit max guard for fixed `u16` nonce search, zero-bit no-op, and tests
      for valid, invalid, and mutated nonces.
- [ ] All protected prover challenge sites call grinding before sampling the
      challenge, and all verifier mirrors check the corresponding witness at
      the same transcript position before sampling the same challenge.
- [ ] The proof carries a canonical fixed-width grinding nonce stream, and all
      protected challenge sites read one `u16` nonce from it in verifier replay
      order.
- [ ] The derived grinding read schedule covers all nonzero-loss sites:
      sumcheck rounds, ring-switch `alpha`, grouped non-terminal `tau0`,
      grouped `tau1`, nontrivial public-row batching, nontrivial stage-1
      interstage batching, and extension-opening reduction sites. Zero-loss
      sparse seeds, singleton batching, and deterministic terminal coefficients
      consume no nonces. Terminal folds have no `tau0` site.
- [ ] Ring sparse / low-norm challenge sampling uses the ungrounded path when
      the policy returns zero grinding bits: replay samples the existing
      challenge bytes from the current transcript state without consuming a
      nonce or doing a predicate squeeze.
- [ ] Policy/setup validation performs a WHIR-style cap check before proving or
      verifier replay: every derived nonzero grinding site must have
      `grinding_bits <= MAX_U16_GRINDING_BITS`, and any violation rejects the
      schedule/configuration instead of entering an unbounded search.
- [ ] Sumcheck grinding is per protected round by default, matching Plonky3
      WHIR: absorb the round message, optionally grind/check one nonce for that
      round, then sample the round challenge. No single "sumcheck block" nonce
      should replace per-round grinding unless a later policy explicitly
      defines and proves that grouped site.
- [ ] Shape/policy data serializes and deserializes enough information for the
      verifier to reject grinding stream length mismatches before accepting a
      proof.
- [ ] `AkitaInstanceDescriptor` or the schedule/setup-bound shape data binds
      the grinding policy so prover and verifier cannot silently choose
      different grinding bits.
- [ ] Planner/proof-size output accounts for two bytes per nonzero grinding
      invocation and reports the resulting 128-bit target security for
      protected challenge sites.
- [ ] Mutating any serialized grinding nonce in an end-to-end proof causes
      verification to fail with `AkitaError::InvalidProof` or another explicit
      `AkitaError`.
- [ ] Existing transcript-hardening checks continue to pass, including
      prover/verifier event-stream equality under `logging-transcript`.
- [ ] Terminal fold replay binds the cleartext logical `w_hat` segment before
      the sparse challenge seed is squeezed, and binds the remaining
      final-witness digits before ring-switch `alpha`/`tau1` sampling. The
      `w_hat` segment is not assumed to be a byte prefix of the serialized
      terminal witness; its offset is derived from the descriptor-bound
      terminal witness layout. No terminal sparse challenge may be sampled from
      a transcript that has not yet committed to the `w_hat` digits it folds.
- [ ] Terminal prover and verifier event streams contain no
      `CHALLENGE_TAU0` squeezes. A logging-transcript test must assert this
      directly for both terminal-root and recursive terminal folds where those
      shapes are reachable.

### Testing Strategy

Existing checks that must remain green:

- `cargo fmt -q`
- `cargo clippy --all --message-format=short -q -- -D warnings`
- `cargo test`
- `cargo test -p akita-transcript --features logging-transcript`
- `cargo test --no-default-features --features transcript-keccak`

New tests:

- `akita-transcript`: deterministic `grind`/`check_grinding` round-trip,
  rejection for mutated nonce, rejection for `grinding_bits > 9`, zero-bit
  no-op, and proof that the post-grinding protocol challenge differs from the
  predicate squeeze.

- `akita-transcript` logging: `TranscriptEvent::Grind` or equivalent records
  label, bits, witness digest, and predicate length; prover/verifier event
  streams match for a protected site.

- `akita-sumcheck`: standard and eq-factored sumcheck proofs with nonzero
  round grinding bits verify; truncating, appending, or mutating the grinding
  stream rejects; prefix-round variants consume fixed `u16` nonces for
  reconstructed and serialized rounds in the same order.
  Tests must include `bits == 0` cases where no nonce is present, mirroring
  Plonky3's zero-PoW sumcheck path.

- `akita-challenges`: sparse challenge seed grinding, when enabled, happens
  after all sparse-fold inputs that must be committed at that point. For
  terminal folds this includes the cleartext logical `w_hat` segment before
  the 32-byte seed squeeze. A mutated seed-grinding nonce changes verifier
  behavior to rejection.

- `akita-types`: serialization round-trips for the proof-level `u16` grinding
  stream, descriptor-bound policy data, and all affected proof shapes.

- End-to-end PCS tests: singleton, same-point batched, multipoint batched,
  extension-opening reduction, intermediate fold, terminal fold, and
  root-direct paths. Root-direct should assert zero grinding witnesses when it
  does not sample protected challenges.

- Terminal transcript-order tests: assert the public event stream has the order
  "current commitment/opening context, terminal logical `w_hat` absorb, sparse
  seed squeeze, terminal witness remainder absorb, `alpha`, `tau1`, stage-2
  sumcheck rounds" and contains no `tau0` squeeze.

- Tamper tests: flip one byte in each witness family and assert verifier
  rejection.

### Performance

The cost is intentionally proof-size and prover-time visible:

- Each nonzero grinding invocation adds exactly two proof bytes for one
  little-endian `u16` nonce. Zero-bit sites add no bytes and do not alter the
  transcript.

- Expected prover work per invocation is `2^k` predicate checks, where `k` is
  the grinding-bit requirement at that site. The prover has at most `2^16`
  candidates. For `k = 9`, bounded-search failure is
  `(1 - 2^-9)^(2^16) ~= 2^-185`; for `k = 10`, it degrades to about `2^-92`.
  The central policy must therefore reject values above 9 under the strict
  global `2^-128` completeness-error target.

- Verifier work is one canonical nonce absorb plus one fixed predicate-word
  squeeze per invocation; it must not perform nonce search.

For the current measured fp128 singleton profile with roughly 309 nonzero
grinding invocations, fixed `u16` nonces add about `309 * 2 = 618` proof bytes,
plus any single top-level stream framing if the final serializer needs one.
This is roughly 273 bytes more than the older bit-packed `k + 7` design
estimate, but removes per-site nonce-width scheduling and most bit-level proof
cursor complexity.

Proof-size and planner accounting must be checked with the existing profile
path:

```bash
AKITA_MODE=onehot AKITA_NUM_VARS=32 cargo run --release --example profile
```

The profile output or planner summary should show aggregate grinding nonce
count, byte overhead, the total bounded-search completeness error, and the
effective challenge security after compensation. A first implementation may report
aggregate overhead only; detailed per-site output is allowed as a follow-up if
the aggregate is correct and test-covered.

## Design

### Architecture

#### Transcript Primitive

The transcript transition should match the Plonky3/WHIR pattern:

```text
observe/absorb protected message
if bits > 0:
    absorb nonce witness
    squeeze/check predicate bits
sample protected Fiat-Shamir challenge
```

Plonky3 stores the nonce witness as a base-field element because its common
fields serialize cheaply, e.g. 4 bytes for BabyBear/KoalaBear/Monty31. Akita
uses fixed `u16` nonces instead: this is still the same transcript transition,
but avoids paying a full field-element encoding in a 128-bit field protocol.

Add a grinding module to `akita-transcript`:

```rust
pub const GRINDING_NONCE_BYTES: usize = 2;
pub const MAX_U16_GRINDING_BITS: u32 = 9;
pub const GRINDING_PREDICATE_WORD_BYTES: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrindingSite {
    pub grinding_bits: u32,
    pub uses_nonce: bool,
}
```

Add a canonical fixed-width nonce stream type in `akita-types` or
`akita-transcript`, owned by the proof layer:

```rust
pub struct GrindingProofStream {
    pub nonces: Vec<u16>,
}
```

For each site with `grinding_bits = k > 0`, the read schedule consumes exactly
one little-endian `u16` nonce:

```text
nonce_bytes = 2
```

For `k == 0`, the site consumes zero nonce bytes and performs no transcript
operation.

Extend `Transcript<F>` with low-level byte-granular operations:

```rust
fn absorb_grinding_nonce(
    &mut self,
    label: &[u8],
    bits: u32,
    nonce: u16,
) -> Result<(), AkitaError>;

fn squeeze_grinding_predicate_word(&mut self, label: &[u8]) -> [u8; 8];
```

For prover ergonomics, add helpers that search `nonce = 0..=u16::MAX` and
append the first valid nonce to the `GrindingProofStream`. For verifier
ergonomics, add helpers that read the next `u16` nonce from the stream and
check it before sampling the protocol challenge.

The search failure probability for one site is:

```text
(1 - 2^-k)^(2^16) <= exp(-2^(16-k))
```

For `k = 9`, this is below `2^-184`, so even hundreds or low thousands of
grinding sites remain below `2^-128` total completeness error by a union bound.
For `k = 10`, this is only about `2^-92`, so the fixed-`u16` design must reject
`grinding_bits > 9` unless a future policy explicitly relaxes the completeness
target or increases the nonce width.

The prover search must not mutate the live transcript for rejected candidates.
Because Spongefish `ProverState` intentionally does not implement `Clone`,
`AkitaTranscript` should use a private replay journal or equivalent safe fork
mechanism to evaluate candidates on scratch transcript states, then commit only
the accepted nonce to the live state.

The verifier cost should be comparable to Plonky3's duplex challenger behavior:
checking a nonzero nonce absorbs the witness and consumes one fixed predicate
word before the protected challenge. The implementation must avoid a shape
where nonce absorption forces one sponge operation and predicate squeezing
forces another avoidable operation for every site.

`check_grinding` absorbs:

```text
b"akita/transcript-grinding/v1" || bits_le_u32 || nonce_le_u16
```

as one framed transcript payload, then squeezes exactly
`GRINDING_PREDICATE_WORD_BYTES` bytes under a grinding label and requires the
low `bits` bits of that predicate word to be zero. The method then returns with
the transcript advanced past the predicate squeeze, so the caller can sample
the actual protocol challenge.

Do not implement grinding by calling the existing `challenge_bytes(label, len)`
helper if it forces Akita's current 32-byte internal chunking. The transcript
API should expose a byte-granular squeeze cursor so a low-bit predicate burns
one fixed predicate word, not an accidental 32-byte challenge chunk.

The production sponge remains positional. The semantic `label` is used for
logging and smell checks, but the fixed payload tag above is the production
domain separation.

`LoggingTranscript` must record grinding events and include the grinding labels
in the known-label set. The event should be separate from ordinary `Absorb` and
`Squeeze` if that makes event equality and smell errors clearer.

#### Grinding Policy

Add a central policy type, most likely in `akita-transcript` or
`akita-config`, and re-export through `akita-config` for protocol code:

```rust
pub struct TranscriptGrindingPolicy {
    pub target_security_bits: u32, // default 128
}
```

The policy exposes helpers of this shape:

```rust
fn challenge_entropy_bits<F, E>() -> u32
where
    F: CanonicalField,
    E: ExtField<F>;

fn bits_for_loss(loss_bits: u32, challenge_entropy_bits: u32) -> Result<u32, AkitaError>;

fn bits_for_degree(degree: usize, challenge_entropy_bits: u32) -> Result<u32, AkitaError>;

fn bits_for_batch_degree(degree: usize, challenge_entropy_bits: u32) -> Result<u32, AkitaError>;

fn bits_for_multilinear_point(num_vars: usize, challenge_entropy_bits: u32) -> Result<u32, AkitaError>;

fn validate_grinding_bits_for_u16_nonce(grinding_bits: u32) -> Result<(), AkitaError>;

fn nonce_count_for_grinding_bits(grinding_bits: u32) -> usize;
```

`challenge_entropy_bits` is `F::modulus_bits() * E::EXT_DEGREE` for extension
challenge field `E/F`. The required grinding bits are:

```text
max(0, target_security_bits + loss_bits - challenge_entropy_bits)
```

rounded up through `ceil_log2` for polynomial degree and total multilinear-point
degree. This keeps the policy valid for both the current fp128 production path
and smaller base-field experiments.

After deriving `grinding_bits`, policy validation must enforce
`grinding_bits <= MAX_U16_GRINDING_BITS`. `nonce_count_for_grinding_bits`
returns `0` for zero-bit sites and `1` for every nonzero site.

Like WHIR's `check_pow_bits`, Akita should validate the full schedule of
derived grinding bits before proving and at verifier setup/descriptor replay.
This catches impossible configurations early and keeps verifier code away from
panic-prone indexing or oversized-search assumptions.

Every callsite must pass an explicit loss source:

- sumcheck round: degree bound of the round polynomial;
- eq-factored sumcheck round: degree bound of the inner polynomial;
- ring-switch `alpha`: ring/evaluation relation degree in `alpha`;
- `tau0`: one grouped grind per non-terminal stage-1 random point, with loss
  `ceil_log2(col_bits + ring_bits)` rather than one grind per coordinate;
  terminal folds must not sample `tau0`;
- `tau1`: one grouped grind per ring-switch row-combination random point, with
  multilinear-point loss `ceil_log2(num_i)` for
  `num_i = ceil_log2(m_rows)`, rather than one grind per coordinate;
- sparse and low-norm challenge seeds: configured sparse challenge family
  soundness loss, usually zero for current fp128 ring sparse / low-norm
  samplers because their seed/support/accounting is independent of the
  field-challenge Schwartz-Zippel loss. Zero means "sample normally" rather
  than "sample a zero-width nonce";
- public-row batching: batching polynomial degree, with singleton rows at zero
  bits and independent-vector batching documented separately from powers-of-one
  gamma batching;
- stage-1 interstage batching: powers-of-gamma degree `child_claims - 1`;
- stage-2 batching: degree of the actual linear combination, currently zero
  extra bits for the non-terminal `gamma * s_claim + relation_claim` degree-one
  combination and no challenge in terminal relation-only stage 2;
- extension-opening reduction batching: number of row partials combined.

If a callsite's exact loss is not already exposed as a variable, the
implementation should add a named helper near the protocol relation rather than
embedding a literal in the prover or verifier.

The policy must be bound into the transcript preamble. Preferred route:
include the target security and all schedule-derived grinding counts in
`AkitaInstanceDescriptor` or in descriptor-bound schedule/security data. This
prevents a prover from using one grinding policy and a verifier from replaying
with another.

#### Proof Ownership

Grinding nonces are proof data, not transcript labels. To avoid one nonce field
per protocol object, store them in one canonical fixed-width stream at the
proof level:

```rust
pub struct AkitaBatchedProof<F: FieldCore, L: FieldCore> {
    pub grinding: GrindingProofStream,
    pub root: AkitaBatchedRootProof<F, L>,
    pub steps: Vec<AkitaProofStep<F, L>>,
}
```

Verifier replay owns the read cursor. Whenever replay reaches a protected
challenge site, it asks the descriptor-bound grinding policy for that site's
`grinding_bits`, reads one `u16` nonce for every nonzero site, checks the nonce,
and then samples the normal challenge. Zero-bit sites read no nonce and leave
the transcript unchanged.

Deserializer/shape checks should mirror Plonky3 sumcheck's proof-count
discipline: if a family has nonzero grinding bits, the proof-level stream must
contain exactly one nonce for every replayed site in that family; if the family
has zero bits, it must contain none. Missing and trailing nonces are verifier
errors before any unchecked indexing.

The stream order is verifier transcript order:

1. public-row batching challenges sampled before root relation construction;
2. root-level extension-opening reduction batching and sumcheck rounds;
3. root-level ring-switch `alpha`, then grouped `tau0` and grouped `tau1`
   point checks before squeezing their coordinates;
4. root-level stage-1 sumcheck rounds and nonzero-loss interstage batching
   challenges;
5. root-level nonzero-loss stage-2 batching and stage-2 sumcheck rounds;
6. recursive suffix levels in order, each following the same local ordering;
7. terminal-level ring-switch `alpha`, grouped `tau1`, and stage-2 sites.
   Terminal replay has no `tau0` site and must not squeeze `CHALLENGE_TAU0`.

Direct root proofs carry an empty grinding stream unless the direct-mode replay
samples a protected challenge.

WHIR also grinds query phases after commitments or final cleartext data are
bound and before query indices are sampled. Akita should preserve that
site-level discipline: grinding belongs immediately before the challenge it
protects, after all messages that challenge is supposed to bind. For Akita this
means, for example, after `ABSORB_SUMCHECK_W` before ring-switch `alpha`, after
a sumcheck round message before that round challenge, and after terminal
`w_hat` binding before any future sparse-seed grinding if sparse accounting
ever becomes nonzero.

#### Fold Path Semantics

Intermediate and terminal folds must have separate transcript schedules. The
implementation should expose separate helpers for the two paths instead of a
single ring-switch helper that samples a superset of challenges and discards
unused values.

Intermediate fold:

1. absorb the current recursive commitment/opening context;
2. compute and absorb `v = D * w_hat` under `ABSORB_PROVER_V`;
3. absorb sparse-challenge context and squeeze the sparse seed;
4. compute `z_pre`, build `w = [w_hat, t_hat, optional blinding, z_pre, r]`,
   and commit to the next recursive witness;
5. absorb the next-witness commitment under `ABSORB_SUMCHECK_W`;
6. squeeze ring-switch `alpha`;
7. squeeze grouped `tau0` coordinates for the stage-1 witness-table point;
8. squeeze grouped `tau1` coordinates for the row-combination point;
9. run stage 1 using `tau0`;
10. absorb `s_claim`;
11. sample the stage-2 batching coefficient if the relation actually needs it;
12. run stage 2.

Intermediate stage 2 proves:

```text
gamma * s_claim + relation_claim
  = sum_{x,y} [
      gamma * eq(r_stage1, (x,y)) * W(x,y) * (W(x,y)+1)
      + W(x,y) * a_alpha(y) * m_{tau1,alpha}(x)
    ].
```

Terminal fold:

1. absorb the current recursive commitment/opening context;
2. compute the decomposed terminal segment `w_hat`;
3. absorb the cleartext logical `w_hat` segment as a terminal-witness
   diagnostic transcript event before any sparse seed is squeezed;
4. absorb sparse-challenge context and squeeze the sparse seed;
5. compute `z_pre`, compute `r`, and build the complete cleartext final
   witness;
6. absorb the remaining final-witness digits as a terminal-witness-remainder
   diagnostic transcript event before ring-switch challenges;
7. squeeze ring-switch `alpha`;
8. squeeze grouped `tau1` coordinates for the row-combination point;
9. run relation-only stage 2.

Terminal folds must not squeeze `tau0`. They skip stage 1, so there is no
stage-1 witness-table point, no `s_claim`, and no stage-2 batching coefficient.
Terminal stage 2 proves:

```text
relation_claim
  = sum_{x,y} W(x,y) * a_alpha(y) * m_{tau1,alpha}(x).
```

The terminal direct witness is still serialized as one canonical proof object,
but transcript replay binds it in two phases. The verifier derives the
`w_hat` range from descriptor-bound schedule data and the same terminal segment
layout used by stage-2 direct-witness evaluation:

```text
w_hat_count = num_w_vectors * num_blocks * num_digits_open
w_hat_offset = if z_first { z_pre_count } else { 0 }
z_first = m_vars >= r_vars
```

in ring elements, converted to packed digit count by multiplying by the active
ring dimension. The remainder contains every terminal witness digit outside
that `w_hat` range, in canonical final-witness order. This avoids relying on a
prefix convention: in current layouts `z_pre` may precede `w_hat` when
`m_vars >= r_vars`.

Verifier replay must reject malformed terminal proofs whose packed witness is
too short for the derived `w_hat` range, whose remainder length does not match
the descriptor-bound final-witness shape, whose extracted `w_hat` digits are
not representable in the scheduled `w_hat` digit basis, or whose event stream
contains any terminal `CHALLENGE_TAU0` squeeze.

The diagnostic event labels for these two absorbs are deliberately not domain
separators. A production prover/verifier must derive the same challenges if
those labels are renamed while the ordered payload bytes stay unchanged.

This path separation is independent of grinding. Grinding must be layered on
top of a transcript order that already binds terminal sparse-fold inputs at the
right time.

#### Serialization and Shapes

`GrindingProofStream` serializes as a context-shaped sequence of little-endian
`u16` nonces. The preferred encoding is headerless once the proof shape is
known: descriptor-bound policy data derives the expected nonce count, and
serialization writes exactly `2 * nonce_count` bytes. If a top-level proof
format needs an explicit sequence length before shape data is available, that
length is a single stream-level field, not one field per nonce or per nested
proof object.

Headerless nested proof bodies remain headerless and do not carry local nonce
lengths. The proof-level shape or descriptor-bound policy carries enough data
to derive the stream read schedule.

Validation requirements:

- `nonce_count` must be bounded by `DEFAULT_MAX_SEQUENCE_LEN / 2` or a stricter
  proof-level cap.
- Every nonzero site consumes exactly one `u16`; zero-bit sites consume none.
- Verifier replay recomputes the expected total nonce count from the policy and
  rejects if the stream has missing or trailing nonces.
- Deserialization validates the stream shape, but proof validity of nonce
  predicates is checked during transcript replay.

#### Prover and Verifier Integration

Add helper functions so protocol code does not duplicate witness-cursor logic:

```rust
fn prove_grinding_challenge<F, T, E>(
    transcript: &mut T,
    label: &[u8],
    bits: u32,
    grinding_stream: &mut GrindingProofStreamWriter,
) -> Result<E, AkitaError>;

fn verify_grinding_challenge<F, T, E>(
    transcript: &mut T,
    label: &[u8],
    bits: u32,
    grinding_stream: &mut GrindingProofStreamReader,
) -> Result<E, AkitaError>;
```

Equivalent helpers should exist for base-field scalar challenges and byte
challenge seeds. They call transcript grinding first, then call the existing
`sample_ext_challenge`, `challenge_scalar`, or `challenge_bytes`.

The helpers must treat `bits == 0` as a no-op. For `bits > 0`, prover helpers
search and append one `u16`, while verifier helpers read and check one `u16`.
Both sides reject `bits > MAX_U16_GRINDING_BITS` before touching the transcript
or stream cursor.

For callsites where zero bits are expected in normal operation, such as ring
sparse / low-norm challenge seed sampling, expose an explicit helper shape:

```rust
fn prove_optional_grinding_then_challenge_bytes<F, T>(
    transcript: &mut T,
    label: &[u8],
    bits: u32,
    len: usize,
    grinding_stream: &mut GrindingProofStreamWriter,
) -> Result<Vec<u8>, AkitaError>;
```

When `bits == 0`, this helper must be observationally identical to
`transcript.challenge_bytes(label, len)`: same transcript state transition, no
nonce stream touch, and no grind event. This protects sparse/low-norm samplers
from accidental dummy nonce overhead while still giving them a policy-controlled
grinding hook if future accounting assigns positive loss.

Update these replay surfaces:

- `crates/akita-sumcheck/src/drivers.rs`
- `crates/akita-sumcheck/src/types.rs`
- `crates/akita-challenges/src/sampler/mod.rs`
- `crates/akita-prover/src/protocol/ring_switch.rs`
- `crates/akita-verifier/src/protocol/ring_switch.rs`
- `crates/akita-prover/src/protocol/flow.rs`
- `crates/akita-verifier/src/protocol/levels.rs`
- `crates/akita-prover/src/protocol/sumcheck/akita_stage1_tree.rs`
- `crates/akita-verifier/src/stages/stage1.rs`
- `crates/akita-types/src/proof/incidence.rs`
- `crates/akita-types/src/proof/mod.rs`

### Alternatives Considered

- `spongefish-pow`: rejected because it is experimental, standalone, and works
  over a separate 32-byte challenge/nonce API rather than Akita's positional
  transcript replay.

- Plonky3 challenger traits: rejected because Akita already owns a
  Spongefish-backed transcript with descriptor preamble binding, production
  positional labels, and logging checks.

- Serializing nonce witnesses as Akita field elements: Plonky3/WHIR does this
  because its common proof-of-work witness fields are 32-bit. Akita's target
  path uses 128-bit fields, so field-element witnesses would cost about 16
  bytes per nonce. A fixed `u16` carries enough search space for the current
  `<= 9` bit policy at much lower proof cost.

- Per-object `u64` witness fields: simple and common in prior art, but rejected
  for Akita because many low-bit sumcheck sites would pay 64 proof bits for
  1-2 bits of grinding.

- Bit-packed `k + 7` nonces: rejected for the first implementation because
  they save only about 273 bytes in the measured fp128 profile while requiring
  per-site nonce-width scheduling, bit-level cursor logic, final-byte padding
  checks, and more complicated proof-shape accounting.

- Per-object fixed-width nonce fields: rejected because they scatter proof
  length logic across many nested proof types. A single proof-level `u16`
  stream keeps replay order canonical and lets verifier replay own one cursor.

- Always serializing nonces for zero-bit sites: rejected because it adds
  unnecessary proof bytes and makes zero-bit grinding alter transcript shape
  without a security reason.

- Absorbing semantic labels into the production sponge for grinding: rejected
  because `specs/transcript-hardening.md` intentionally makes production
  labels diagnostic-only.

## Documentation

Update:

- `crates/akita-transcript/src/lib.rs` and `src/sponge.rs` docs with the
  grinding contract and payload format.
- `crates/akita-transcript/src/labels.rs` with grinding labels used for
  diagnostics.
- `specs/transcript-hardening.md` with a short note that grinding is the next
  transcript hardening layer and still preserves positional production labels.
- Planner/profile docs or output comments to explain the aggregate grinding
  proof-size overhead, bounded-search completeness error, and target security.

No paper-note archival is required for this spec, but the implementation PR
should reference Plonky3, Spongefish PoW, OpenVM, Whir, and Stwo prior art.

Add a short implementation note that Akita intentionally matches
Plonky3/WHIR's transcript order and optional-zero-bit behavior, while choosing
`u16` nonce serialization instead of base-field-element nonce serialization for
proof-size reasons over 128-bit fields.

## Execution

Suggested implementation order:

1. Implement byte-granular transcript absorb/squeeze cursor APIs, grinding
   checking, logging events, labels, and unit tests in `akita-transcript`.
2. Fix terminal witness binding: split terminal direct-witness transcript
   absorption into logical-`w_hat`-before-sparse-seed and
   remainder-before-ring-switch phases using descriptor-derived segment
   ranges, remove all terminal `tau0` sampling, and add prover/verifier
   logging tests for both properties.
3. Implement `GrindingProofStream` reader/writer and canonical `u16`
   serialization.
4. Add central policy helpers and bind policy data into
   `AkitaInstanceDescriptor` or descriptor-bound schedule/security data.
5. Thread the grinding stream reader/writer through `SumcheckProof` and
   `EqFactoredSumcheckProof` drivers without adding local witness vectors.
6. Cut over standalone loss-bearing challenge families in ring-switch, grouped
   tau-point sampling, public-row batching, stage-1 interstage batching,
   stage-2 batching, and extension-opening reduction. Keep zero-loss sparse
  seed sampling classified but routed through the explicit ungrounded path
  unless the security accounting later assigns it a positive loss.
7. Update proof shapes and all call sites that construct, serialize,
   deserialize, or derive them.
8. Update planner/profile proof-size and completeness-error accounting.
9. Add tamper tests and end-to-end transcript logging tests.
10. Run the acceptance commands.

Risk areas:

- Spongefish prover states are intentionally non-cloneable. Grinding cannot
  clone the live transcript and try candidates in place; it needs a replay
  journal or another explicit safe forking mechanism.

- Akita's current transcript helper squeezes in 32-byte chunks. Grinding needs
  a byte-granular predicate path, or low-bit sites will burn unnecessary output
  and create confusing transcript behavior.

- The current shared ring-switch helper samples `tau0` unconditionally.
  Terminal replay must move to a terminal-specific helper that never samples
  `tau0`; leaving a dead terminal squeeze in place preserves an avoidable
  transcript footgun.

- A global proof-level nonce stream makes serialization compact but puts more
  responsibility on verifier replay ordering. The descriptor-bound policy and
  final "no trailing nonces" check are mandatory.

- Prefix-round sumcheck variants reconstruct some round messages instead of
  reading them from `proof.round_polys`; they still need to consume grinding
  witnesses in total round order.

- Verifier code is a no-panic boundary. All count mismatches, policy requests
  above 9 bits, and mutated nonces must return `AkitaError`.

- Planner/security formulas should be reviewed before implementation if a
  protocol relation's exact loss is not already clear from local variables.

## References

- `specs/TEMPLATE.md`
- `specs/SPEC_REVIEW.md`
- `specs/transcript-hardening.md`
- `crates/akita-transcript/src/lib.rs`
- `crates/akita-transcript/src/sponge.rs`
- `crates/akita-transcript/src/logging.rs`
- `crates/akita-sumcheck/src/drivers.rs`
- `crates/akita-sumcheck/src/types.rs`
- `crates/akita-challenges/src/sampler/mod.rs`
- `crates/akita-prover/src/protocol/ring_switch.rs`
- `crates/akita-verifier/src/protocol/ring_switch.rs`
- `crates/akita-types/src/proof/mod.rs`
- [Plonky3 `GrindingChallenger`](https://github.com/Plonky3/Plonky3/blob/1b65d2327f2a9272eb72cb84b332f2c19bd73e4e/challenger/src/grinding_challenger.rs)
- [Plonky3 FRI config](https://github.com/Plonky3/Plonky3/blob/1b65d2327f2a9272eb72cb84b332f2c19bd73e4e/fri/src/config.rs)
- [Spongefish PoW implementation](https://github.com/arkworks-rs/spongefish/blob/d2d190b1329d35ac9577438d05aed4f17a57b9f9/pow/src/lib.rs)
- [Spongefish PoW README](https://github.com/arkworks-rs/spongefish/blob/d2d190b1329d35ac9577438d05aed4f17a57b9f9/pow/README.md)
- [OpenVM STARK backend prover PoW reference](https://github.com/openvm-org/stark-backend/blob/d0c1693c90d8aa84f6299d6fd53a60f0cf57f9f4/crates/stark-backend/src/prover/cpu/mod.rs)
- [whir-p3 prover reference](https://github.com/tcoratger/whir-p3/blob/a755a26ae41c52e3a1695ac4c443af627c96fe77/src/whir/prover/mod.rs)
- [Stwo proof-of-work reference](https://github.com/starkware-libs/stwo/blob/aeceb74c58184d7886ebd7f34a7453fee714ca40/crates/stwo/src/core/proof_of_work.rs)
