# Spec: Transcript Grinding

| Field | Value |
|---|---|
| Author(s) | Quang Dao, Codex |
| Created | 2026-05-22 |
| Status | proposed |
| PR | |
| Supersedes | Unmerged transcript grinding design at `5057456` |
| Superseded-by | |
| Book-chapter | |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**,
and **MAY** in this document are to be interpreted as described in
[BCP 14](https://www.rfc-editor.org/info/bcp14) when, and only when, they appear
in all capitals.

## Summary

Akita will add transcript proof-of-work before Fiat-Shamir challenge queries
whose algebraic soundness loss reduces their nominal 128-bit challenge
security. Each such query has a public zero-bit target. The prover supplies a
nonce that makes that many low bits of a separate transcript predicate equal
zero. The verifier checks the same predicate before drawing the actual protocol
challenge. This restores 128 bits of classical random-oracle work per protected
query without changing the challenge distribution.

All transcript search nonces will move into one proof-level bit stream. The
stream also replaces the current fixed `u32` fields used by fold-response
rejection sampling, but the two mechanisms remain distinct. Proof-of-work pays
for algebraic challenge loss. Fold-response rejection searches for an honest
response that fits a scheduled norm bound. The existing sparse fold challenge
families already provide at least 128 bits of challenge support, so they do not
receive an additional proof-of-work check.

The stream is decoded in a canonical order derived from the public schedule,
proof shape, opening layout, field tower, and one descriptor-bound grinding
policy. A nonce with `g` proof-of-work bits uses exactly `g + 7` wire bits. The
current fold-response nonce uses exactly 12 wire bits. Nonces are packed
without byte alignment, so small queries do not pay for a fixed `u16` or
`u32` slot.

## Current state

Akita has one mechanism currently called grinding. It lives in
`crates/akita-prover/src/protocol/fold_grind.rs`. For each fold, the prover
tries sequential `u32` nonces until the sparse challenge produces a folded
response accepted by the scheduled representation and norm checks. The
verifier validates the nonce against the exclusive bound of 4096, absorbs its
four little-endian bytes into the sparse challenge context, and checks the
resulting response. `FoldLinfProtocolBinding` binds the 4096 attempt cap and
the four-byte wire width.

That mechanism is honest-prover rejection sampling. It does not repair a small
Fiat-Shamir challenge space and it does not add 12 bits of soundness. Every
adversarial nonce trial is already another random-oracle query. The current
accounting is described in
[`book/src/foundations/pcs-and-binding.md`](../book/src/foundations/pcs-and-binding.md).

Akita does not yet have transcript proof-of-work. Sumcheck rounds, ring-switch
checks, multilinear points, and power batching draw from a nominal 128-bit
extension challenge field, but some checks have a bad set larger than one
field element. A degree `d` polynomial check, for example, has conditional
error at most `d / |E|`. The goal of this feature is to charge about
`2^ceil(log2(d))` work before that challenge, so finding an accepted bad challenge again
costs about `2^128` classical random-oracle trials.

The old unmerged design proposed one `u16` per nonzero query, a nine-bit cap,
and a new byte-granular transcript squeeze cursor. The current code makes those
choices unnecessary:

1. `AkitaBatchedProofShape` already derives the exact headerless proof layout.
2. `challenge_bytes(label, 32)` already consumes one complete 32-byte sponge
   block.
3. `AkitaTranscript` already has a prover-only preview path that clones the
   sponge state for fold-response search.
4. The proof already carries one fixed `u32` fold nonce in every nonterminal
   and terminal fold object.

This specification uses those current boundaries.

## Intent

### Goal

Add one schedule-derived `GrindingPlan`, one canonical
`TranscriptNonceStream`, and one transcript proof-of-work primitive, then route
every current Fiat-Shamir query through the plan. The same stream will carry
the existing fold-response nonces in 12 bits each.

### Terminology

This document uses these terms precisely:

| Term | Meaning |
|---|---|
| challenge query | One logical Fiat-Shamir draw or consecutive block of draws with no intervening prover message |
| loss factor `L` | Public upper bound such that a query's conditional algebraic error is at most `L / |E|` |
| grind bits `g` | Public proof-of-work target assigned to a query |
| nonce bits `w` | Number of proof bits reserved for the bounded nonce search |
| proof-of-work query | A query that checks a zero-bit predicate before drawing the protocol challenge |
| fold-response query | Existing rejection sampling that changes the sparse challenge until the honest response fits |
| zero-bit query | A catalogued query with `g = 0`; it consumes no nonce and makes no transcript change |

### Invariants

1. **One public plan.** Prover, verifier, proof shape, descriptor, serializer,
   and proof-size accounting MUST consume the same `GrindingPlan`. No callsite
   may reconstruct a competing nonce width or loss bound.
2. **Logical query boundaries.** One extension-field challenge is one query,
   not one query per base-field limb. One multilinear point is one query when
   its coordinates are consecutive and no prover message intervenes. Each
   sumcheck round is a separate query because a prover round message
   intervenes.
3. **Public acceptance.** A proof-of-work query with target `g` accepts exactly
   when the first `g` bits of its public 32-byte predicate block are zero.
4. **Separate predicate and challenge.** The predicate block MUST NOT be reused
   as the protocol challenge. Checking a nonzero proof-of-work query advances
   the transcript by one full predicate squeeze before the actual challenge is
   drawn.
5. **Zero means no operation.** A zero-bit query consumes no stream bits,
   absorbs no grinding payload, emits no predicate squeeze, and leaves the
   transcript unchanged.
6. **Exact packing.** Each stream entry consumes exactly the number of bits in
   its plan entry. Entries have no wire tags, widths, padding, or alignment
   between them.
7. **Canonical tail.** The stream is little-endian at both the bit and integer
   levels. Any unused high bits in the final byte MUST be zero.
8. **Complete consumption.** Verifier replay MUST reject a truncated stream, a
   stream with nonzero tail padding, a plan mismatch, a query-kind mismatch,
   or any bits left after the final scheduled query.
9. **Bounded proving.** Every search has a public finite candidate space.
   Exhaustion returns `AkitaError`; it never loops without a bound.
10. **Verifier safety.** All verifier-reachable decode, plan, cursor, and
    predicate failures return `AkitaError` or `SerializationError`. They MUST
    NOT add `panic!`, `assert!`, `unwrap`, unchecked indexing, or allocation
    from an unvalidated length.
11. **Positional transcript.** Production labels remain diagnostics. The fixed
    proof-of-work payload enters the sponge. A semantic label does not.
12. **Fold behavior is preserved.** Moving a fold-response nonce into the
    stream MUST preserve its current transcript payload, challenge law,
    4096-candidate bound, prover acceptance rule, and verifier response check.
13. **Sparse support is not double charged.** The signed sparse challenge and
    operator-rejected sparse families retain their current certified support.
    They have no proof-of-work query merely because a fold-response nonce is
    present.
14. **No compatibility path.** The proof and descriptor wire formats change in
    place. Akita provides no backward proof compatibility, so there is no
    legacy decoder or duplicate verifier replay.

### Non-goals

This feature does not do any of the following:

1. It does not change the sparse fold challenge distribution or its 128-bit
   support analysis.
2. It does not use proof-of-work to price SIS or MSIS security.
3. It does not treat the fold-response nonce as evidence that a response is
   short. The verifier still checks the scheduled response representation and
   norm bound.
4. It does not reduce conditional statistical soundness. Conditioned on a
   valid proof-of-work predicate, the protocol challenge remains uniform and
   its bad fraction remains `L / |E|`.
5. It does not claim a quantum random-oracle proof. Akita's current
   Fiat-Shamir accounting uses a classical online random-oracle query bound.
   A QROM theorem, including quantum search against the predicate, is separate
   work.
6. It does not close the full-vector forking and multi-fork extraction gap in
   [`specs/subring-coefficient-packing.md`](subring-coefficient-packing.md).
   Grinding the later `alpha` polynomial check only prices that check.
7. It does not add `spongefish-pow`, replace the transcript backend, or switch
   Akita to another challenger interface.
8. It does not add a general transcript journal or a byte-level squeeze cursor.
9. It does not preserve the old unmerged fixed-`u16` proposal.

## Security model

### Nominal challenge capacity

Akita uses the existing nominal challenge-capacity convention

```text
C = F::modulus_bits() * E::EXT_DEGREE.
```

The production fp128, fp64 quadratic-extension, and fp32 quartic-extension
profiles all have `C = 128`. This is the same field-width convention used by
the current protocol. The tiny difference between a pseudo-Mersenne prime and
the adjacent power of two remains part of complete concrete accounting. It
does not force one extra proof-of-work bit at every query.

### Per-query grind bits

For a challenge query with public loss factor `L >= 1`, target `T = 128`, and
nominal capacity `C`, the plan assigns

```text
loss_bits = ceil_log2(L)
g = max(0, T + loss_bits - C).
```

For current production profiles this simplifies to

```text
g = ceil_log2(L).
```

One candidate passes the predicate with probability `2^-g`. A classical
attacker who needs a challenge in a bad set of fraction at most `L / |E|`
therefore expects about

```text
2^g * |E| / L
```

random-oracle work. Under the nominal capacity convention, this reaches the
`2^128` target when `C = 128`. Exact field cardinality remains visible in the
displayed expression and in complete concrete accounting.

This is computational work accounting. It is not a claim that the interactive
soundness error changed from `L / |E|` to `1 / |E|`. The proof-of-work
predicate and the protocol challenge use separate random-oracle outputs so the
challenge remains uniform after predicate acceptance.

### Query composition

The target is applied to each logical query. The implementation MUST NOT add a
blanket `ceil_log2(number_of_queries)` surcharge to every site. Such a rule
would multiply honest prover work even though the existing Fiat-Shamir theorem
already accounts for the adversary's total oracle-query budget `Q` and the
protocol proof accounts for its own sum of challenge errors.

The final security statement still includes both of these terms:

```text
interactive challenge error = sum of the protocol's query errors
Fiat-Shamir knowledge error <= (Q + 1) * interactive challenge error
```

Grinding changes the work required to realize one of those errors. It does not
erase the sum and it does not erase `Q`. Reusing the current online
random-oracle statement avoids counting nonce freedom twice.

### Bounded nonce search

For every proof-of-work query with `g > 0`, set

```text
w = g + 7.
```

The prover tests canonical integers

```text
0, 1, ..., 2^w - 1.
```

The probability that an honest prover finds no passing predicate is

```text
(1 - 2^-g)^(2^w)
  = (1 - 2^-g)^(2^(g+7))
 <= exp(-128)
 < 2^-184.6.
```

This bound is independent of `g`. The plan accepts fewer than `2^32` query
entries and encodes query ordinals as `u32`. A union bound therefore puts total
proof-of-work exhaustion below `2^-152`, which is already
well below the required `2^-128`. No per-proof query-count surcharge is needed.

The first implementation supports `g <= 25`, hence `w <= 32`. This is an
implementation bound, not a security limit. It lets candidate arithmetic use a
checked `u32` while the wire still uses exactly `w` bits. A schedule requesting
more than 25 grind bits fails during plan derivation and verifier setup.

### Current loss rules

The central query catalog uses these rules. `ceil_log2(0)` is never evaluated.
A singleton or absent check has `L = 1` and `g = 0`.

| Query family | Query boundary | Loss factor `L` | Production `g` |
|---|---|---:|---:|
| degree `d` polynomial identity | one scalar challenge | `max(1, d)` | `ceil_log2(max(1, d))` |
| sumcheck round of degree `d` | one round after its prover message | `max(1, d)` | `ceil_log2(max(1, d))` |
| multilinear point with `n` coordinates | one consecutive point draw | `max(1, n)` | `ceil_log2(max(1, n))` |
| powers of one scalar batching `m` values | one scalar | `max(1, m - 1)` | `ceil_log2(max(1, m - 1))` |
| independent random coefficients | one consecutive coefficient vector | `1` | `0` |
| one random linear merge | one scalar | `1` | `0` |
| subring packing consistency at `alpha` | one scalar | `2s - 1` | `ceil_log2(2s - 1)` |
| signed sparse fold challenge | one 32-byte seed | certified support at least 128 bits | `0` |

For an ordinary evaluation-trace ring relation, the alpha loss MUST come from
one canonical degree-bound helper owned by the relation description. For a
subring coefficient-packing relation, that helper MUST return the existing
`2s - 1` bound. The plan, prover, verifier, and security report all call this
same helper.

Independent coefficient vectors deserve explicit treatment. If at least one
batched claim difference is nonzero, all but one independent coefficient may
be fixed and at most one value of the remaining coefficient makes the linear
combination vanish. The loss is therefore `1 / |E|`, regardless of vector
length. It receives zero grind bits. By contrast, batching with powers
`1, gamma, ..., gamma^(m-1)` creates a degree `m - 1` polynomial and receives
`ceil_log2(m - 1)` bits.

### Quantum scope

The PCS is intended for post-quantum use and its SIS tables target 128-bit
quantum attack cost. The current Fiat-Shamir text, however, states a classical
online random-oracle extraction bound. This feature preserves that theorem and
does not silently relabel the proof-of-work result as QROM security.

A future QROM analysis must account for quantum nonce search and for the
Fiat-Shamir extractor. It may change the grind policy. That work is not a
reason to double every current grind target without a theorem.

## Query catalog and order

`GrindingPlan` is an ordered replay program, not just a total bit count. It
contains every logical challenge query, including zero-bit queries, so an audit
can explain why each challenge is protected or exempt. Only entries with a
nonce width consume stream bits.

The current protocol order is:

1. Opening claim and evaluation batching.
2. Optional extension-opening reduction point, claim batching, and per-round
   sumcheck challenges.
3. One fold-response query for the sparse witness-fold challenge.
4. Ring-switch `alpha`, then the consecutive `tau0` and `tau1` point draws that
   exist for that fold shape.
5. Stage 1 tree sumcheck rounds. Each tree stage is followed by its powers-of
   gamma interstage batching query when child claims exist.
6. Optional physical L2 powers batching, linear merge, and sumcheck rounds.
7. Optional virtual-evaluation powers batching and compression query.
8. Stage 2 linear batching and per-round sumcheck challenges.
9. Optional Stage 3 setup-product sumcheck rounds.
10. The same sequence for each recursive fold and the terminal paths selected
    by the schedule.

The current diagnostic labels map to catalog rules as follows. A label may
appear in more than one protocol context, so the plan site and public shape,
not the byte label alone, select the rule.

| Current label | Current context | Catalog rule |
|---|---|---|
| `CHALLENGE_EVAL_BATCH` | independent opening coefficients | independent coefficient vector, `g = 0` |
| `CHALLENGE_SUMCHECK_BATCH` | EOR split point | multilinear point, `g = ceil_log2(max(1, split_bits))` |
| `CHALLENGE_SUMCHECK_BATCH` | Stage 2 relation merge | one linear merge, `g = 0` |
| `CHALLENGE_EOR_CLAIM_BATCH` | independent EOR claim coefficients | independent coefficient vector, `g = 0` |
| `CHALLENGE_SUMCHECK_ROUND` | EOR, Stage 1, Stage 2, and Stage 3 rounds | round degree from the canonical sumcheck shape |
| `CHALLENGE_SPARSE_CHALLENGE` | sparse witness fold seed | fold-response entry of 12 nonce bits, no proof-of-work |
| `CHALLENGE_RING_SWITCH` | ring-relation evaluation at `alpha` | relation degree bound |
| `CHALLENGE_TAU0` | Stage 1 multilinear point | number of point coordinates |
| `CHALLENGE_TAU1` | relation multilinear point | number of point coordinates |
| `CHALLENGE_SUMCHECK_INTERSTAGE_BATCH` | powers batching of child claims | `max(1, child_claims - 1)` |
| `CHALLENGE_L2_NORM_BATCH` | powers batching of norm subclaims | `max(1, subclaims - 1)` |
| `CHALLENGE_L2_NORM_MERGE` | range and norm merge | one linear merge, `g = 0` |
| `CHALLENGE_L2_VIRTUAL_BATCH` | powers batching of virtual evaluations | `max(1, evaluations - 1)` |
| `CHALLENGE_COMPRESSION_BINARY` | support-restricted binary relation merge | one linear merge, `g = 0` |

`CHALLENGE_LINEAR_RELATION` and `CHALLENGE_STOP_CONDITION` remain registered
diagnostic labels but are not reached by the current core replay found in this
audit. They receive no plan entries until a live protocol path draws them. The
challenge coverage test prevents either label from becoming live without a
catalog decision.

The plan MUST mirror the actual branch structure. In particular:

1. A direct or absent protocol component contributes no phantom queries.
2. Terminal replay contributes no `tau0` query because terminal ring switching
   has no Stage 1 point.
3. A consecutive extension-field vector is one catalog entry even though
   `sample_ext_challenge` currently squeezes one base-field limb at a time.
4. A powers batching site is one entry before its scalar `gamma` draw.
5. A sumcheck contributes one entry per round because each round polynomial is
   absorbed before the next challenge.
6. The fold-response entry appears exactly where the existing nonce is
   absorbed into the sparse challenge context.

The first implementation MUST include an audit test that records every
challenge label reached by each production schedule and proves that it matches
the plan entries in the same order. A challenge draw that bypasses the catalog
is a test failure even when its assigned grind bits would be zero.

## Design

### Public policy and plan

`akita-types` will own the schedule-facing types because it already owns
`FoldSchedule`, `OpeningClaimsLayout`, proof shapes, and descriptor sections.
The intended public model is:

```rust
pub const TRANSCRIPT_SECURITY_BITS: u8 = 128;
pub const GRINDING_NONCE_SLACK_BITS: u8 = 7;
pub const MAX_GRINDING_BITS: u8 = 25;
pub const FOLD_RESPONSE_NONCE_BITS: u8 = 12;

pub enum GrindingQueryKind {
    ProofOfWork,
    FoldResponse,
}

pub enum GrindingSite {
    EvaluationBatch,
    ExtensionOpeningPoint,
    ExtensionOpeningClaimBatch,
    SumcheckRound { protocol: SumcheckProtocol, round: usize },
    FoldResponse { level: usize },
    RingSwitchAlpha { level: usize },
    Tau0Point { level: usize },
    Tau1Point { level: usize },
    Stage1InterstageBatch { level: usize, stage: usize },
    L2SubclaimBatch { level: usize },
    L2NormMerge { level: usize },
    L2VirtualBatch { level: usize },
    CompressionBinary { level: usize },
    Stage2Batch { level: usize },
}

pub struct GrindingQuery {
    pub site: GrindingSite,
    pub kind: GrindingQueryKind,
    pub loss_factor: usize,
    pub grind_bits: u8,
    pub nonce_bits: u8,
}

pub struct GrindingPlan {
    pub queries: Vec<GrindingQuery>,
    pub total_nonce_bits: usize,
}
```

The exact Rust layout MAY change to avoid recursive or oversized enums. The
semantic fields and one canonical derivation MUST remain. A plan constructor
takes the public schedule, opening layout, field tower metadata, and proof
shape. It validates every checked addition, query count, loss factor, grind
target, and total stream length before proving or decoding.

Zero-bit proof-of-work entries have `nonce_bits = 0`. Nonzero proof-of-work
entries have `nonce_bits = grind_bits + 7`. Fold-response entries have
`grind_bits = 0` and `nonce_bits = 12`. This makes the shared storage explicit
without confusing the security meanings.

### Descriptor binding

`FoldLinfProtocolBinding` will be replaced by a protocol-wide
`TranscriptGrindingBinding`. The current binding contains at least:

```text
encoding version                 = 1
target security bits             = 128
proof-of-work slack bits         = 7
maximum proof-of-work bits       = 25
predicate bytes                  = 32
predicate bit order              = little-endian
fold-response attempt bits       = 12
fold-response attempts           = 4096
query-policy revision            = 1
```

`AkitaInstanceDescriptor` will bind this policy and the canonical digest of
the complete `GrindingPlan`. A dedicated grinding section is preferable to
placing call-specific plan data inside `PlanSection`, because the plan depends
on both the schedule and `OpeningClaimsLayout`.

The plan itself is not serialized in the proof. Prover and verifier derive it
from public inputs and compare its digest through the descriptor preamble.
Changing a loss rule, query boundary, or encoding rule is therefore a protocol
revision, not an unbound local choice.

### Packed nonce stream

`AkitaBatchedProof` gains one leading field:

```rust
pub nonce_stream: TranscriptNonceStream,
```

The individual `fold_grind_nonce: u32` fields are removed from
`FoldLevelProof` and `TerminalLevelProof`. No replacement field is added to a
level proof.

The headerless proof body serializes the nonce stream first, then the current
root, recursive, and terminal payloads. The stream has no length header. The
canonical `AkitaBatchedProofShape` gains `nonce_stream_bits`, derived from the
same `GrindingPlan`. Its byte length is

```text
nonce_stream_bytes = ceil(nonce_stream_bits / 8).
```

Bits are appended in plan order. For an entry value `x` with width `w`, stream
bit `j` receives

```text
(x >> j) & 1, for j = 0, ..., w - 1.
```

Within each byte, lower stream indices use lower bit positions. Entries begin
immediately after the preceding entry. If the final entry ends inside a byte,
all remaining high bits of that byte are zero.

The codec exposes checked sequential writer and reader operations. It does not
expose random access by site. The plan cursor checks the next expected
`GrindingSite` and `GrindingQueryKind` before reading or writing. The verifier
checks exact completion once protocol replay ends.

The stream stores values, not attempts. A proof-of-work value must fit in its
`w` bits. A fold-response value must be less than 4096. The prover uses a
checked wider loop counter so the `w = 32` endpoint does not overflow a `u32`
range expression.

### Transcript proof-of-work primitive

`akita-transcript` owns the transcript transition and predicate, but not the
schedule-derived plan. For `g > 0`, the canonical absorbed payload is

```text
b"akita/transcript-grinding/v1"
|| g_as_u8
|| w_as_u8
|| nonce_as_le_u32
```

The complete payload is passed once to `append_bytes` under a diagnostic
grinding label. Akita's framed transcript absorb makes this prefix-free. The
production sponge does not absorb the semantic site label.

After the absorb, the transcript squeezes exactly 32 bytes under a diagnostic
predicate label. Predicate bit `i` is

```text
(predicate[i / 8] >> (i % 8)) & 1.
```

The predicate accepts when bits `0..g` are all zero. Because `g <= 25`, this
check needs no variable-length allocation and no partial squeeze. The caller
then invokes the existing scalar, extension, vector, or byte challenge API for
the actual protocol challenge.

For `g = 0`, callers MUST use the explicit no-op path. They do not absorb the
payload or squeeze the predicate.

The prover tests candidates against a scratch copy of the current sponge state
and commits only the accepted candidate to the live transcript. The existing
`FoldChallengeSeedPreview` path already clones the internal duplex sponge for
prover-only preview. It will be generalized into one prover preview primitive
that can execute the canonical absorb and one 32-byte squeeze. A full replay
journal and a public clone of the transcript state are not required.

The verifier decodes one scheduled nonce, applies the same live absorb and
squeeze, and rejects a nonzero predicate. It performs no search.

`LoggingTranscript` records one first-class event for each nonzero
proof-of-work check. The event contains the diagnostic site label, `g`, `w`,
the nonce value or its digest, and predicate length. Prover and verifier event
streams must remain identical. Zero-bit queries emit no grind event.

### Fold-response migration

The current fold-response search remains in
`crates/akita-prover/src/protocol/fold_grind.rs`. It receives the next
`FoldResponse` plan entry, searches candidates `0..4096`, and writes the
accepted value into 12 stream bits. The verifier reads the same entry and
expands it to `u32` for the existing challenge sampler.

The sparse sampler continues to absorb

```text
challenge context || nonce_as_le_u32
```

and then squeeze the current 32-byte sparse challenge seed. There is no
proof-of-work predicate before this seed. A given accepted numeric nonce must
therefore produce the same sparse challenge as it did before this proof-format
cutover.

The new `TranscriptGrindingBinding` owns the 12-bit and 4096-attempt constants.
`FoldLinfProtocolBinding` and its fixed four-byte field are removed. The
fold-l∞ spec and Book text will be updated in the implementation slice so they
describe the shared stream while preserving their current soundness argument.

### Prover and verifier integration

The top-level prover creates one plan cursor and one stream writer before root
replay. The top-level verifier creates one plan cursor and one stream reader
after bounded proof decoding. Protocol functions receive the cursor through
their existing replay path. Type methods may assemble their fields into the
canonical operation, but no second policy or wrapper helper may recompute
grind bits.

The existing sumcheck APIs already accept a challenge closure. Their callers
will capture the shared cursor and call the canonical grind-then-sample
operation in that closure. `akita-sumcheck` therefore does not need to depend
on `akita-types` or learn Akita's schedule. Its round-message ordering remains
unchanged.

Consecutive challenge vectors use one plan entry and one optional predicate
before their first draw. Examples are EOR split points, independent row
coefficients, `tau0`, and `tau1`. The cursor advances once for the logical
vector. The current limb-level `sample_ext_challenge` calls remain ordinary
transcript squeezes inside that block.

Every prover and verifier mirror MUST consume the same site at the same point.
Tests compare both the ordinary transcript event stream and the plan-consume
event stream.

### Proof size and planning

Proof-size accounting adds exactly

```text
ceil(sum(query.nonce_bits) / 8)
```

bytes. It removes four bytes from every fold level and adds 12 packed bits for
that fold. Proof-of-work sites add their individual `g + 7` widths. Because
entries share final bytes, the total is rounded once, not once per query.

Planner and profile reports show:

1. total query count;
2. nonzero proof-of-work query count;
3. fold-response query count;
4. total stream bits and bytes;
5. a histogram of proof-of-work targets;
6. expected predicate trials, summed as `sum(2^g)`;
7. the maximum and union-bounded honest exhaustion probability;
8. every query family whose loss rule is nonzero.

Generated schedule identity includes the grinding-policy revision and the
grinding contribution to proof bytes. Schedule generation MUST fail if a
candidate cannot derive a valid plan. Existing catalog drift tests protect the
generated output.

## Evaluation

### Acceptance criteria

- [ ] `akita-types` exposes one validated `GrindingPlan` whose ordered entries
      cover every logical challenge query for every production schedule and
      opening shape.
- [ ] The plan derives `g = max(0, 128 + ceil_log2(L) - C)`, uses `w = g + 7`
      for nonzero proof-of-work queries, rejects `g > 25`, and proves the stated
      `exp(-128)` per-query exhaustion bound in unit tests.
- [ ] The plan catalog distinguishes degree checks, sumcheck rounds,
      multilinear points, powers batching, independent coefficient vectors,
      sparse challenge draws, and fold-response search using the loss rules in
      this specification.
- [ ] Every current challenge label reached by a production proof has exactly
      one matching logical plan entry. Zero-bit entries remain visible to the
      audit but consume no stream bits and make no transcript change.
- [ ] `TranscriptGrindingBinding` replaces `FoldLinfProtocolBinding` and binds
      the policy constants, query-policy revision, and canonical plan digest in
      `AkitaInstanceDescriptor`.
- [ ] `AkitaBatchedProof` carries one leading `TranscriptNonceStream`.
      `FoldLevelProof` and `TerminalLevelProof` no longer carry individual
      `u32` nonce fields.
- [ ] `AkitaBatchedProofShape` derives the exact stream bit count. The codec is
      headerless, packs entries little-endian without per-entry alignment, and
      rejects truncation, extra bytes, nonzero final padding, over-wide values,
      wrong query kinds, wrong sites, and incomplete replay.
- [ ] Existing fold-response nonces consume exactly 12 stream bits, retain the
      4096-attempt cap, and produce the same sparse challenge for the same
      transcript prefix and numeric nonce as before the wire cutover.
- [ ] `akita-transcript` exposes one canonical 32-byte proof-of-work predicate
      transition with the exact payload and low-bit test specified above.
- [ ] The prover preview tests candidates without mutating the live transcript.
      The accepted candidate replay equals verifier replay for both Blake2b and
      Keccak transcript backends.
- [ ] The predicate squeeze is distinct from the following protocol challenge.
      A regression test fails if the predicate bytes are reused as challenge
      bytes.
- [ ] Every protected sumcheck round grinds after absorbing its round
      polynomial and before sampling that round's challenge.
- [ ] Every protected consecutive challenge vector grinds once before its
      first coordinate. No extension-field limb or point coordinate receives a
      separate stream entry.
- [ ] Independent random coefficient vectors and single linear merges receive
      zero grind bits. Powers-of-gamma batching receives
      `ceil_log2(m - 1)` bits when `m > 2`.
- [ ] Ring-switch alpha uses the canonical relation degree helper. Subring
      coefficient packing uses `L = 2s - 1`. This change does not mark the
      packing extraction blocker resolved.
- [ ] Sparse fold challenges receive no proof-of-work predicate. Their only
      stream entry is the existing 12-bit fold-response search value.
- [ ] Logging tests show identical prover and verifier transcript events and
      identical ordered plan-consumption events. Zero-bit sites emit no grind
      event.
- [ ] Mutating any used nonce bit causes explicit verifier rejection or changes
      the checked fold response so verification rejects. Mutating only unused
      final padding also rejects.
- [ ] Root, recursive, terminal, direct, extension-opening, subring-packing,
      physical-L2, compressed, and setup-prefix paths have end-to-end coverage
      for their selected query schedules.
- [ ] Proof-size and planner output report the exact packed stream contribution
      and expected prover work. Generated schedules and catalog identities are
      regenerated and drift checks pass.
- [ ] The Book transcript and security chapters, the fold-l∞ spec, the crate
      graph if dependencies change, and verifier-contract documentation are
      updated before the implementation is marked complete.
- [ ] All verifier-reachable failures obey the no-panic contract under malformed
      proof, shape, descriptor, and stream inputs.

### Testing strategy

New focused tests are grouped by owner.

`akita-transcript`:

1. valid and mutated predicate checks for every `g` boundary, including 1, 8,
   9, 16, 17, 25, and zero;
2. exact payload bytes and low-bit order;
3. zero-bit no-op state equality;
4. rejected preview candidates do not mutate live state;
5. accepted preview and live replay agree;
6. predicate output and following challenge are separate blocks;
7. Blake2b and Keccak parity at the protocol event level.

`akita-types` and `akita-serialization`:

1. bit-stream round trips across byte boundaries;
2. all-zero and nonzero final padding behavior;
3. truncated, extended, and mismatched stream rejection;
4. 12-bit fold-response bounds and 32-bit proof-of-work endpoint handling;
5. plan derivation snapshots for every production schedule family;
6. descriptor digest changes when any policy or plan entry changes;
7. proof-shape decode budgets reject before allocation.

Protocol crates:

1. per-round sumcheck placement after the absorbed round polynomial;
2. one entry per consecutive vector rather than per limb or coordinate;
3. query coverage for EOR, ring switching, digit range, physical L2, Stage 2,
   Stage 3, compression, root, recursion, and terminal replay;
4. same-nonce sparse challenge regression across the fold stream migration;
5. proof tampering at each query kind;
6. prover and verifier plan exhaustion at the same final cursor position.

Planner and integration:

1. exact bit and byte totals against serialized proofs;
2. expected-work totals against the plan histogram;
3. generated schedule and catalog identity drift;
4. Jolt shape serialization and verifier replay;
5. all production field profiles and transcript backends.

The implementation runs the repository preflight from `AGENTS.md`, including
all four release Clippy feature graphs. Documentation slices run
`./scripts/check-doc-guardrails.sh`. The final test pass uses the current
commands in `.github/workflows/ci.yml`, including its target selection,
features, profile, and sharding.

### Performance

Verifier overhead is one 32-byte transcript squeeze for each nonzero
proof-of-work query. It performs no nonce search. Stream decoding is linear in
the number of stream bits and uses bounded storage derived before proof decode.

Expected prover predicate work is `2^g` per protected query. The intended
production targets are small because they pay only for public algebraic loss.
The implementation MUST report the observed target histogram before this spec
is marked implemented. Any production site above 12 grind bits requires
explicit review, not because it is unsound, but because it implies at least
4096 expected predicate trials at that site.

The packed stream is always no larger than byte-aligning each entry and is
strictly smaller than keeping one `u32` per fold. The exact before and after
proof sizes will be recorded with the existing profile and schedule census
commands after the plan is implemented. No fixed byte estimate is normative
before that measurement because current proof shapes vary by field, schedule,
opening layout, and optional security route.

## Alternatives considered

### One fixed `u16` per proof-of-work query

This was the old design. It is simple, but it caps the safe target at nine bits
under its completeness rule and wastes space for the common one-bit and
two-bit sites. Exact stream packing has one canonical codec and supports larger
targets without choosing a second proof format.

### Choose `u16` or `u32` at each query

Width classes avoid some waste, but each entry still pays byte alignment and
the policy needs a class-selection rule. The bit stream already needs a cursor
for the 12-bit fold values, so exact widths are simpler as a protocol rule.
Implementations may use `u16` internally when `w <= 16` and `u32` otherwise.

### Keep one nonce field inside each level proof

That works for fold-response search but does not scale to sumcheck rounds and
other nested challenge sites. It also ties storage to Rust proof nesting rather
than transcript replay order. One top-level stream makes ordering explicit and
lets adjacent entries share bytes.

### Serialize tags and lengths with each nonce

Tags make isolated decoding easier but duplicate public schedule information.
The verifier already requires the exact schedule and proof shape. Plan-kind
checks give the same safety without per-entry proof bytes.

### Add a byte-granular transcript cursor

The current transcript squeezes in 32-byte chunks. A 32-byte grinding
predicate consumes exactly one current chunk, so another buffering layer would
add state and replay complexity without saving a sponge call.

### Reuse predicate bytes as the protocol challenge

This would make predicate acceptance condition the challenge itself and would
invalidate the simple work calculation. A separate squeeze keeps the actual
challenge uniform.

### Add proof-of-work to sparse fold challenges

The sparse families already have at least 128 bits of certified support. Their
nonce exists to find an acceptable honest response. Adding a zero predicate
would duplicate work without restoring any missing challenge bits.

### Charge the maximum query loss to every challenge

This is easy to state but unnecessarily expensive. The schedule already knows
each query's degree, vector dimension, and batching form. Public per-site loss
rules are small, auditable, and meet the intended work target directly.

### Double grind bits for quantum search

That would be a policy guess, not a QROM proof. The current Fiat-Shamir theorem
is classical and already has explicit query accounting. Quantum policy changes
belong with a complete QROM analysis.

## Execution

Implementation proceeds in the following ordered slices. Each slice leaves the
tree buildable and testable. The same draft pull request carries all slices.

### Slice 0: Revive the normative specification

1. Add this current design record.
2. Add it to the live-spec indexes and dead-symbol guard.
3. Run documentation guardrails.

Exit condition: reviewers can evaluate the security model, wire format,
ownership, query catalog, and later slice boundaries without relying on the old
unmerged branch.

### Slice 1: Canonical policy, loss helpers, and plan

1. Add the grinding policy constants and query types in `akita-types`.
2. Add the one canonical relation alpha degree helper.
3. Derive the complete ordered plan from schedule, opening layout, field tower,
   and proof shape.
4. Add query coverage snapshots for all generated production schedules.
5. Replace `FoldLinfProtocolBinding` with `TranscriptGrindingBinding` and bind
   the plan digest in `AkitaInstanceDescriptor`.

Exit condition: prover-independent code can derive and validate one plan, its
exact stream bit length, and its descriptor digest. No proof wire change has
landed yet.

### Slice 2: Packed stream and fold-response cutover

1. Implement `TranscriptNonceStream` plus checked reader and writer cursors.
2. Add `nonce_stream_bits` to `AkitaBatchedProofShape`.
3. Add the leading stream to `AkitaBatchedProof` serialization.
4. Remove every per-level `fold_grind_nonce: u32`.
5. Route existing fold-response proving and verification through 12-bit plan
   entries.
6. Update exact proof-size accounting and Jolt shape serialization.

Exit condition: all existing proofs use the packed stream for fold-response
nonces, sparse challenge replay is unchanged for equal numeric nonces, and no
proof-of-work query is active yet.

### Slice 3: Transcript predicate and prover preview

1. Add the canonical payload, 32-byte predicate, and low-bit checker in
   `akita-transcript`.
2. Generalize the existing prover sponge preview for this one transition.
3. Add logging events and known labels.
4. Cover zero-bit no-op, search exhaustion, backend parity, and
   predicate-versus-challenge separation.

Exit condition: a focused prover and verifier test can search, serialize,
decode, and check one proof-of-work entry without protocol integration.

### Slice 4: Protocol query integration

Integrate in dependency order so lower-level drivers are exercised before full
fold replay:

1. sumcheck round closures;
2. powers batching and independent coefficient vectors;
3. extension-opening reduction points and rounds;
4. ring-switch `alpha`, `tau0`, and `tau1`;
5. Stage 1 and physical L2 queries;
6. Stage 2 and compression queries;
7. Stage 3 setup-product rounds;
8. root, recursive, and terminal top-level cursor ownership.

At each step, add the prover and verifier mirror together. Do not land a slice
that consumes a stream entry on only one side.

Exit condition: every live challenge draw is catalogued, all nonzero sites
check proof-of-work, every zero site is a transcript no-op, and the final plan
and stream cursors are exactly exhausted.

### Slice 5: Planner, generated schedules, and reporting

1. Add exact stream bytes and expected predicate work to candidate pricing.
2. Add plan revision to schedule and catalog identity.
3. Emit query counts, target histogram, exhaustion bound, and proof bytes in
   planner and profile reports.
4. Regenerate schedule tables and update stable snapshots.

Exit condition: generated catalog drift checks pass and serialized proof sizes
match planner estimates for every production profile fixture.

### Slice 6: End-to-end hardening and documentation

1. Add bit-level tamper tests, malformed shape tests, and no-panic verifier
   tests.
2. Run all transcript backend and feature-graph tests.
3. Update the Book transcript, security, binding, proof-size, and verification
   text.
4. Update `fold-linf-rejection.md` to describe its 12-bit stream entry without
   changing its soundness argument.
5. Update `AGENTS.md`, verifier contract, and crate graph only where the final
   implementation changes their owned contracts.

Exit condition: every acceptance criterion is checked, the final CI-fidelity
test commands pass, and the spec status and PR field are ready for the merge
state.

## File map

The expected primary surfaces are:

| Area | Current owner | Intended change |
|---|---|---|
| transcript trait and sponge | `crates/akita-transcript/src/lib.rs`, `sponge.rs` | predicate transition and preview |
| transcript diagnostics | `crates/akita-transcript/src/labels.rs`, `logging.rs` | grind labels and events |
| proof objects and shapes | `crates/akita-types/src/proof/levels.rs`, `shapes.rs` | top-level stream and exact bit shape |
| plan and policy | new focused module under `crates/akita-types/src/` | single query catalog and derivation |
| descriptor | `crates/akita-types/src/instance_descriptor/` | policy and plan digest binding |
| fold response search | `crates/akita-prover/src/protocol/fold_grind.rs` | 12-bit stream writer |
| sparse replay | `crates/akita-challenges/src/sampler/mod.rs` | preserve numeric nonce transcript input |
| sumcheck drivers | `crates/akita-sumcheck/src/drivers/`, callers in prover and verifier | captured grind cursor in challenge closures |
| ring switching | `crates/akita-prover/src/protocol/ring_switch/`, `crates/akita-verifier/src/protocol/ring_switch.rs` | alpha and point query integration |
| fold stages | `crates/akita-prover/src/protocol/core/`, `crates/akita-verifier/src/protocol/core/` | shared cursor replay |
| proof size | `crates/akita-types/src/proof_size.rs`, layout sizing | exact packed bytes |
| planning | `crates/akita-planner/`, `crates/akita-schedules/` | pricing, reporting, identity, generation |

No new crate dependency is expected. If implementation changes that, update
`docs/crate-graph.md` in the same slice.

## Documentation

While this spec is proposed and active, it is the normative design record.
After implementation stabilizes, durable behavior belongs in:

1. `book/src/how/transcript.md` for the predicate transition, stream replay,
   and descriptor binding;
2. `book/src/foundations/pcs-and-binding.md` for classical random-oracle work
   accounting and its relation to fold-response nonces;
3. `book/src/how/security.md` for query loss rules and the explicit QROM scope;
4. `book/src/how/verification.md` for malformed stream rejection and the
   verifier no-panic boundary;
5. `book/src/usage/profiling.md` for planner and profile output.

When those chapters own the stable result, set `Book-chapter` to the primary
owner, mark this spec implemented, and archive it according to
[`specs/PRUNING.md`](PRUNING.md).

## References

1. [`book/src/foundations/pcs-and-binding.md`](../book/src/foundations/pcs-and-binding.md)
   for current Fiat-Shamir query and fold nonce accounting.
2. [`book/src/how/transcript.md`](../book/src/how/transcript.md) for positional
   transcript and descriptor binding.
3. [`specs/fold-linf-rejection.md`](fold-linf-rejection.md) for current
   fold-response rejection sampling.
4. [`specs/subring-coefficient-packing.md`](subring-coefficient-packing.md) for
   the `2s - 1` alpha bound and the separate extraction blocker.
5. `crates/akita-transcript/src/sponge.rs` for the current 32-byte squeeze and
   prover preview mechanism.
6. `crates/akita-types/src/proof/shapes.rs` for headerless proof shape
   derivation.
7. Historical unmerged design at commit `5057456`, path
   `specs/transcript-grinding.md`.
