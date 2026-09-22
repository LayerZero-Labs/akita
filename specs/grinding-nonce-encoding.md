# Spec: Grinding nonce encoding

| Field | Value |
|---|---|
| Author(s) | Omid Bodaghi, Codex |
| Created | 2026-09-21 |
| Status | active |
| PR | [#37](https://github.com/LayerZero-Labs/akita/pull/37) |
| Supersedes | |
| Superseded-by | |
| Book-chapter | |

## Summary

This document explains Akita's transcript grinding from first principles and
then compares the packed nonce representation on `origin/main` with the inline
unsigned LEB128 representation in PR #37.

The two encodings do not define grinding security. They transport the bounded
integer selected by grinding. The public grinding plan, nonce range, predicate,
challenge order, response equations, and verifier checks define the security
contract.

This is the explanatory companion to the normative contract in
[`transcript-grinding.md`](transcript-grinding.md). That specification remains
authoritative when this explanation and a protocol invariant differ.

## 1. What grinding is

Akita uses the word *grinding* for two bounded searches that have different
purposes:

1. transcript proof-of-work before selected Fiat--Shamir challenges; and
2. fold-response rejection sampling used to find an honestly representable,
   sufficiently short folded response.

Both searches produce a small nonnegative integer called a *nonce*. They share
one public `GrindingPlan`, but they must not be interpreted as the same
security mechanism.

### 1.1 Why Fiat--Shamir challenges sometimes need proof-of-work

In an interactive proof, a verifier samples a random challenge after receiving
the prover's preceding message. Fiat--Shamir replaces the verifier with a hash
or sponge: the transcript derives the challenge from everything fixed so far.

Some algebraic checks have more than one bad challenge. If a nonzero
polynomial has degree `d`, for example, it can vanish at as many as `d` field
points. The conditional bad fraction is bounded by:

```text
d / |E|
```

where `E` is the challenge field. More generally, Akita assigns a public loss
factor `L` to a challenge site whose bad fraction is at most `L / |E|`.

For the production profiles, the nominal challenge capacity is 128 bits. A
loss factor larger than one would reduce the classical work needed to search
for a bad challenge by approximately `log2(L)` bits. Akita restores that work
factor with transcript proof-of-work.

For nominal challenge capacity `C`, Akita computes:

```text
loss_bits = ceil(log2(L))
g = max(0, 128 + loss_bits - C)
```

`g` is the number of *grind bits*. With the production convention `C = 128`,
this simplifies to:

```text
g = ceil(log2(L))
```

Before drawing the protected challenge, the prover searches for a nonce whose
separate 32-byte predicate begins with `g` zero bits, read low bit first. One
candidate passes with probability `2^-g`, so the expected search is about
`2^g` candidates.

This does not add entropy and does not make the protected challenge larger.
It makes each accepted attempt more expensive. Conditioned on a valid
predicate, the actual protocol challenge is drawn separately and retains its
required distribution.

### 1.2 What happens during proof-of-work grinding

For a nonzero target `g`, the public plan assigns a nonce width:

```text
w = g + 7 bits
```

The seven slack bits give the prover `2^(g+7)` candidates, 128 times the
expected `2^g` work. Under the ideal predicate model, bounded-search exhaustion
is at most approximately `exp(-128)`.

The prover performs:

1. Fix the complete transcript history before the protected query.
2. Try nonce candidates in the public range `0 <= nonce < 2^w`.
3. Preview the grinding transition on a clone of the public transcript state.
4. Accept a candidate if the first `g` predicate bits are zero.
5. Commit the accepted nonce once to the live transcript.
6. Draw the protected protocol challenge separately.

The verifier performs:

1. Read the next scheduled nonce.
2. Check that it fits the public `w`-bit range.
3. Reproduce the predicate transition.
4. Reject unless the first `g` predicate bits are zero.
5. Draw the protected challenge from the resulting live state.

The honest prover searches upward from zero and uses the first winner. The
verifier intentionally accepts *any* in-range winner. Requiring the first
winner would make the verifier repeat the prover's search and is unnecessary
for the soundness accounting.

For a zero-bit target, `w = 0`: there is no nonce, no predicate squeeze, and no
grinding-specific transcript transition. The site can remain in the public
plan as an ordering and query-accounting obligation.

### 1.3 Fold-response grinding is different

Each fold also searches for a nonce, but not to compensate for an algebraic
loss factor. A sparse challenge can produce a folded witness whose encoded
representation or norm exceeds the selected schedule's bounds. The prover is
allowed to try another sparse challenge until it finds an acceptable response.

Akita gives each fold-response search a fixed 12-bit domain:

```text
0 <= nonce < 4096
```

For candidate `c`, the prover derives the fold's sparse challenges, computes
the response, and accepts only if every scheduled representation and norm check
passes. One candidate is shared by all commitment groups in that fold. The
verifier reconstructs the challenges for the supplied candidate and still
checks the response equations and bounds directly.

The 12-bit nonce is therefore honest-prover rejection sampling. It does not add
12 bits of soundness, prove that the response is short, or replace the
verifier's response checks. Every adversarial trial is still included in
Akita's random-oracle query accounting.

## 2. Where `origin/main` uses grinding

The grinding plan is derived only from trusted public data:

- the selected fold schedule;
- the normalized opening layout;
- field modulus bits and extension degree; and
- protocol policy and loss bounds.

Its digest is bound into the instance descriptor. Proof bytes do not choose the
number, order, width, or meaning of grinding entries.

There are three kinds of plan entries:

| Kind | Purpose | Nonce bytes? |
| --- | --- | --- |
| `ProofOfWork` | Price a Fiat--Shamir query's algebraic loss | Only when `g > 0` |
| `FoldResponse` | Bounded search for an acceptable folded response | Always one 12-bit nonce |
| `FoldChallengeGroup` | Audit one group root and all indexed coordinates | No nonce |

The last kind is important for query counting and replay order but is not
additional proof-of-work.

### 2.1 Nonterminal fold level

At each root or recursive nonterminal level, main constructs entries in
protocol order.

#### Optional extension-opening reduction

When the opening method and extension degree require a reduction, the plan
contains:

- one complete extension-opening point draw;
- an optional reduction-claim batching challenge; and
- one distinct proof-of-work site for every extension-opening-reduction
  sumcheck round.

#### Opening and fold setup

Depending on the public layout, the plan contains:

- an optional evaluation/row batching challenge;
- exactly one 12-bit fold-response search for the level; and
- one zero-width fold-challenge-group entry per commitment group, with
  multiplicity covering the group root and every indexed sparse coordinate.

The sparse fold challenges themselves do not receive extra proof-of-work. Their
certified challenge support is accounted for separately.

#### Ring switch and evaluation points

The plan contains proof-of-work sites for:

- the ring-switch `alpha` challenge;
- the complete `tau0` multilinear point; and
- the complete `tau1` multilinear point.

A complete multilinear point is one site. It is not split into one grinding
nonce per coordinate because the public security bound applies jointly to the
point.

#### Stage 1 range proof

For every Stage 1 range-proof stage, the plan contains:

- one proof-of-work site for every Stage 1 sumcheck round; and
- when the stage has child claims, one interstage batching site.

#### Optional physical-L2 route

When the selected security route includes physical-L2 checks, the plan can
contain:

- an L2 subclaim batching site;
- an L2 norm-merge site;
- one site for every physical-L2 sumcheck round; and
- an L2 virtual-evaluation batching site.

#### Stage 2 and optional Stage 3

The plan contains:

- an optional compressed-payload binary challenge;
- the Stage 2 batching challenge;
- one site for every Stage 2 sumcheck round; and
- for a recursive successor with a setup prefix, one site for every Stage 3
  sumcheck round.

Some named sites have loss factor one. In a 128-bit-capacity profile they have
`g = 0`, so they consume no nonce bits and do no PoW transition. They remain in
the plan to keep the query catalog and replay order complete.

### 2.2 Terminal level

The terminal level contains:

- the extension-opening-reduction sites when the extension degree requires
  them;
- exactly one 12-bit terminal fold-response nonce; and
- one zero-width terminal fold-challenge-group entry.

The terminal does not replay the ordinary Stage 1 through Stage 3 pipeline.

### 2.3 Does every sumcheck round need grinding?

Every sumcheck round is a distinct *plan site*. This is necessary because the
prover sends a new round polynomial before each round challenge. That polynomial
fixes a new conditional bad set, so two rounds cannot be merged into one
security event merely because they belong to the same sumcheck.

For a sumcheck round with declared polynomial degree `d`, main uses:

```text
L = max(d, 1)
g = max(0, 128 + ceil(log2(L)) - C)
```

For production profiles with `C = 128`:

| Round degree | Loss factor | Grind bits | Nonce width |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 0 | 0 |
| 2 | 2 | 1 | 8 |
| 3 | 3 | 2 | 9 |
| 4 | 4 | 2 | 9 |
| 5–8 | 5–8 | 3 | 10 |

Consequently, yes: ordinary degree-2 or degree-3 sumcheck rounds have their own
nonzero proof-of-work nonce. A degree-1 round still has its own plan entry but
needs no proof bytes.

In the current plan builder:

- extension-opening-reduction rounds use degree 2;
- Stage 2 rounds use degree 3;
- Stage 3 rounds use degree 2;
- Stage 1 and physical-L2 round degrees come from their public proof shapes.

## 3. How main encodes and parses grinding nonces

Main separates nonce *storage* from nonce *use*.

- Storage is one compact proof-level `TranscriptNonceStream`.
- Protocol replay reconstructs each logical integer at its scheduled site.
- PoW then places that integer in a separate fixed transcript payload.
- Fold-response code uses the integer as the candidate for sparse challenges.

The packed storage bytes are not themselves absorbed as one transcript block.

### 3.1 What a nonce width means

For plan entry `i`, the semantic width `b_i` defines its allowed integer
domain:

```text
0 <= nonce_i < 2^b_i
```

It is not the concrete value's significant-bit length, a field width, or a
domain-separation tag.

For example, a 12-bit fold-response entry occupies 12 bits whether its value is
0, 5, 300, or 4095. The value 4096 is not representable and must reject.

### 3.2 Aggregate packed size

For widths `b_1, b_2, ..., b_n`, main allocates exactly:

```text
total_bits  = b_1 + b_2 + ... + b_n
total_bytes = ceil(total_bits / 8)
```

There is no byte alignment between entries. The first entry starts at bit zero;
the next starts immediately after it, even in the middle of a byte.

This matters. Two 12-bit nonces occupy:

```text
ceil((12 + 12) / 8) = 3 bytes
```

Independently rounding each to two bytes would require four bytes.

### 3.3 Concrete bit-packing example

Suppose the public plan has:

```text
entry 1: width 3, value 5
entry 2: width 5, value 17
```

The first domain is `0..7`; the second is `0..31`. The writer stores the low
bits of each integer least-significant bit first at one global cursor.

Byte positions 0 through 7 become:

```text
entry 1       entry 2
1 0 1        1 0 0 0 1
```

In the conventional most-significant-bit-first display, the byte is:

```text
1000_1101 = 0x8d
```

The two logical integers occupy one byte.

A more representative example is:

```text
entry 1: width 8,  value 5
entry 2: width 12, value 300
```

The total is 20 bits, so the stream is three bytes:

```text
05 2c 01
```

Only the low four bits of the last byte are meaningful. If no later entry fills
the remaining high bits, canonical parsing requires them to be zero.

Changing the values does not change the packed length:

```text
values 5 and 300      -> 3 bytes
values 200 and 4000   -> 3 bytes
values 0 and 0        -> 3 bytes
```

Only the public widths determine the section size.

### 3.4 Main's prover path

Main's writer:

1. Derives the complete public plan.
2. Computes `ceil(total_nonce_bits / 8)` with checked arithmetic.
3. Allocates and zero-initializes that exact number of bytes.
4. Maintains a plan cursor and a global bit offset.
5. At each live site, requires the next plan entry to match the expected site
   and kind.
6. Searches for or receives the concrete nonce value.
7. Rejects a value that does not fit the entry's width.
8. ORs its low bits into the stream at the current bit offset.
9. Advances by exactly the scheduled width.
10. At completion, requires exact plan and bit-cursor exhaustion.

The resulting packed nonce bytes are serialized at the beginning of the
structured `AkitaBatchedProof`, followed by root, recursive-fold, and terminal
objects. No nonce count, width, or per-entry tag is carried in the stream; all
of that comes from the trusted proof shape and plan.

### 3.5 Main's verifier path

Main's verifier:

1. Derives the same plan and total bit count from public data.
2. Reads exactly `ceil(total_nonce_bits / 8)` bytes before the structured fold
   objects.
3. Rejects a length mismatch or nonzero unused high bits in the final byte.
4. Creates a sequential reader with the same plan and bit offset zero.
5. At each protocol site, requires the next plan entry to have the expected
   identity and kind.
6. Extracts exactly that entry's low-width bits, reconstructing a `u32`.
7. Advances by the public width.
8. Performs the PoW predicate or fold-response checks.
9. At the end, rejects any unconsumed plan entry or nonce bit.

Because the stream has no internal tags, parsing is canonical only in the
context of the bound public plan. The proof cannot reorder or reinterpret
entries without causing the plan cursor or the later verifier equations to
fail.

### 3.6 How main uses a decoded nonce in the transcript

For nonzero PoW, main does not absorb the packed bits directly. It reconstructs
the logical integer and absorbs a fixed payload:

```text
"akita/transcript-grinding/v1"
grind_bits                         1 byte
nonce_bits                         1 byte
nonce                              4-byte little-endian u32
```

It then squeezes the predicate and, after acceptance, draws the protected
challenge separately.

For fold-response search, the decoded integer selects the candidate sparse
challenge. On main, its four-byte numeric encoding enters each group root
payload. The verifier reconstructs the same challenge sequence and checks the
response.

Thus main deliberately uses one representation for compact proof storage and
another representation for transcript computation.

## 4. Limits of main's packed approach

Main's approach is valid and has an excellent worst-case size guarantee. Its
limitations are architectural rather than a demonstrated soundness defect.

### 4.1 Proof transport and transcript absorption are separate

The packed bytes live at the proof level, away from the protocol sites that use
them. Akita must maintain:

- a structured proof parser;
- a packed nonce reader or writer;
- a grinding-plan cursor;
- a bit cursor;
- a transcript adapter; and
- protocol code that calls all of them in matching order.

Reading a nonce and absorbing its logical transcript representation are two
separate operations. Correctness depends on those operations never drifting.

### 4.2 The representation is not locally streamable

The first packed byte can contain bits from multiple future protocol sites.
Individual nonce boundaries exist only in the public plan and bit cursor. A
generic proof transport cannot parse one nonce without Akita's external plan
logic.

This is acceptable for main's structured proof architecture, but it prevents
Spongefish from owning nonce receipt directly at the point of use.

### 4.3 Main pays for the entire candidate-domain width

The seven PoW slack bits make honest bounded-search failure negligible, but
main stores them for every nonce even when the first winner is small.

For example, a target `g = 9` has width `w = 16`. The honest winner is typically
on the scale of `2^9`, but main always stores all 16 scheduled bits. Its size is
fixed and predictable, but it cannot benefit from a particularly small winner.

### 4.4 How to improve while retaining safety

There are three broad choices:

1. Keep global packing. This preserves the best fixed aggregate bound but also
   preserves the separate nonce transport.
2. Use fixed-width inline messages. This aligns transport with protocol order
   but rounds every nonce independently to bytes.
3. Use canonical variable-width inline messages. This aligns transport with
   protocol order and makes honest small values compact, while requiring a
   separate conservative maximum for parser safety.

PR #37 chooses the third option with unsigned LEB128.

## 5. What unsigned LEB128 is

LEB128 means **Little-Endian Base 128**. It encodes a nonnegative integer as
base-128 digits, least-significant digit first.

Each byte contains:

```text
continuation bit | seven value bits
       1 bit     |      7 bits
```

- High bit `1`: another byte follows.
- High bit `0`: this is the final byte.

Akita uses unsigned LEB128 for `u32` nonces. It does not use signed LEB128 or
ZigZag encoding.

| Integer | Canonical unsigned LEB128 | Bytes |
| ---: | --- | ---: |
| 0 | `00` | 1 |
| 1 | `01` | 1 |
| 127 | `7f` | 1 |
| 128 | `80 01` | 2 |
| 300 | `ac 02` | 2 |
| 16,383 | `ff 7f` | 2 |
| 16,384 | `80 80 01` | 3 |
| `u32::MAX` | `ff ff ff ff 0f` | 5 |

The encoding is self-delimiting because the continuation bit marks the final
byte. It is value-dependent: two nonces with the same scheduled range can have
different encoded lengths.

## 6. How LEB128 works in detail

### 6.1 Encoding algorithm

To encode an unsigned integer:

1. Take the low seven bits as the next payload.
2. Shift the integer right by seven bits.
3. If the remaining value is nonzero, set the payload byte's high bit.
4. Emit the byte.
5. Repeat until the remaining value is zero.

Equivalent pseudocode is:

```text
repeat:
    byte = value & 0x7f
    value = value >> 7
    if value != 0:
        byte = byte | 0x80
    emit(byte)
until value == 0
```

Akita's encoder uses a five-byte stack buffer because unsigned `u32` needs at
most five LEB128 bytes.

### 6.2 Example: 128 becomes `80 01`

Write 128 in base 128:

```text
128 = 0 + 1 * 128
```

The least-significant base-128 digits are therefore `0` and `1`.

First byte:

```text
payload       = 0000000
more follows  = 1
byte          = 1_0000000 = 0x80
```

Final byte:

```text
payload       = 0000001
more follows  = 0
byte          = 0_0000001 = 0x01
```

So:

```text
128 -> 80 01
```

Decoding reconstructs:

```text
(0x80 & 0x7f) * 128^0 = 0
(0x01 & 0x7f) * 128^1 = 128
total                     128
```

### 6.3 Example: 300 becomes `ac 02`

Write 300 in base 128:

```text
300 = 44 + 2 * 128
```

The digits are `44` and `2`.

First byte:

```text
44            = 0x2c = 0101100
continuation  = 0x80
first byte    = 0xac = 10101100
```

Final byte:

```text
2             = 0x02 = 0000010
continuation  = 0
final byte    = 0x02
```

So:

```text
300 -> ac 02
```

Decoding reconstructs:

```text
(0xac & 0x7f) * 128^0 = 44
(0x02 & 0x7f) * 128^1 = 256
total                     300
```

### 6.4 Canonical decoding

Unsigned LEB128 can have redundant mathematical representations unless the
decoder enforces shortest form. For example, both `00` and `80 00` evaluate to
zero, but Akita accepts only `00`.

Akita's native decoder rejects:

- a redundant terminal zero group such as `80 00`;
- an unterminated sequence such as a lone `80`;
- more than five bytes for a `u32`;
- high payload bits that overflow `u32`; and
- a value that later fails the schedule-derived nonce-width check.

On failure, receipt is transactional: the verifier does not advance its proof
cursor past a malformed nonce.

Canonicality means one integer has one accepted byte encoding. It does not mean
the integer is the smallest satisfying grinding nonce.

### 6.5 Size formula

For a positive scheduled width `b`, the largest allowed value requires at most:

```text
ceil(b / 7) LEB128 bytes
```

For a concrete value `v`, actual size is:

```text
1 byte  for 0 <= v < 2^7
2 bytes for 2^7 <= v < 2^14
3 bytes for 2^14 <= v < 2^21
4 bytes for 2^21 <= v < 2^28
5 bytes for 2^28 <= v <= 2^32 - 1
```

A zero-width PoW entry emits no message. A present nonce whose value is zero is
encoded as the one byte `00`.

## 7. How PR #37 uses LEB128 with Spongefish

PR #37 removes the global `TranscriptNonceStream`. At every nonzero PoW or
fold-response site, the nonce is an inline native Spongefish prover message.

For PoW:

1. Clone only the public Spongefish state for preview.
2. Encode the candidate in canonical unsigned LEB128.
3. Absorb those bytes into the preview state.
4. Squeeze the separate predicate.
5. When a candidate passes, emit it once with `prover_message`.
6. Spongefish appends the LEB128 bytes to the argument and absorbs the same
   bytes into the live state.
7. The verifier's matching `prover_message` decodes and absorbs exactly those
   proof bytes before checking the predicate.

Fold-response search similarly previews by absorbing one LEB128 nonce, derives
all group roots in canonical order, and commits the accepted nonce once before
repeating that sequence on the live state.

There is no nonce prefix, count, width field, or separate packed suffix. The
public positional grammar and grinding-plan cursor say when a nonce must occur;
LEB128 says where that individual integer ends.

Because the LEB128 proof bytes are now the absorbed transcript bytes, this is a
protocol change rather than only a storage change. Challenges differ from main,
and the native protocol uses a new version.

## 8. Detailed size comparison

For nonzero semantic widths `b_i`, three quantities must remain distinct:

```text
main packed size
    = ceil(sum(b_i) / 8)

native LEB128 actual size
    = sum(LEB128_length(chosen_nonce_i))

native LEB128 maximum
    = sum(ceil(b_i / 7))
```

Main rounds once after aggregating all bits. LEB128 rounds every message
independently, but rounds according to the actual value rather than the full
domain width.

### 8.1 One nonce

For one 16-bit nonce:

| Value | Main packed | LEB128 |
| ---: | ---: | ---: |
| 3 | 2 bytes | 1 byte |
| 127 | 2 bytes | 1 byte |
| 128 | 2 bytes | 2 bytes |
| 300 | 2 bytes | 2 bytes |
| 16,383 | 2 bytes | 2 bytes |
| 16,384 | 2 bytes | 3 bytes |
| 65,535 | 2 bytes | 3 bytes |

LEB128 wins for small values, ties for the middle range, and loses near the top
of this domain.

### 8.2 Two 12-bit fold-response nonces

Main always uses:

```text
ceil((12 + 12) / 8) = 3 bytes
```

LEB128 uses:

| Values | LEB128 total | Winner |
| --- | ---: | --- |
| Both below 128 | 2 bytes | LEB128 |
| One below 128, one at least 128 | 3 bytes | Tie |
| Both at least 128 | 4 bytes | Main |

This is the clearest example of the tradeoff between value-sensitive encoding
and aggregate bit packing.

### 8.3 Repeated degree-3 sumcheck rounds

With `C = 128`, a degree-3 round has:

```text
g = 2
w = g + 7 = 9 bits
```

Ten such rounds occupy on main:

```text
ceil(10 * 9 / 8) = 12 bytes
```

For each PoW attempt, success probability is `1/4`. An honest first winner is
normally a very small integer, so almost every LEB128 nonce is one byte. Ten
rounds will normally use about ten bytes. A prover deliberately choosing values
at least 128 could make every nonce two bytes, for a 20-byte native maximum.

This explains both why LEB128 performs well for the honest prover and why its
parser bound cannot use the honest measurement.

### 8.4 Why honest PoW tends to favor LEB128

A `g`-bit predicate has expected first winner on the scale of `2^g`, while the
scheduled domain width is `g + 7`. Main always pays for the seven slack bits.
LEB128 usually pays for the significant bits of the actual first winner.

The slack still protects against bounded-search exhaustion, but it usually does
not occupy proof bytes under LEB128.

Fold-response acceptance is determined by response geometry and norms, not the
simple PoW probability, so its LEB128 savings must be measured rather than
inferred from `g`.

### 8.5 Recorded nv=36 result

For the recorded fp128 one-hot, one-polynomial, `nv=36` parity workload:

| Representation | Nonce bytes | Complete proof bytes |
| --- | ---: | ---: |
| Pinned main packed stream | 400 | 69,776 |
| Native LEB128 candidate | 335 | 69,756 |

LEB128 saved 65 nonce bytes for that honest run. The complete proof saved only
20 bytes because PR #37 also changes other framing and transcript-dependent
terminal data. This measurement does not prove that every honest run or every
valid proof is smaller.

## 9. When each approach is better

### Main's aggregate packing is better when

- a fixed proof-size guarantee is more important than typical size;
- valid nonces are often near the top of their allowed domains;
- many adjacent widths can share partial bytes efficiently;
- worst-case recursion input size should be as small and simple as possible;
- storage is already separate from transcript replay; or
- a verifier already has the structured proof shape and global bit cursor.

Main has the stronger nonce-section worst-case guarantee. Once the schedule is
known, honest, malicious, lucky, and unlucky provers all use the same number of
packed nonce bytes.

### LEB128 is better when

- the honest prover usually finds small nonce values;
- protocol messages should be consumed at the exact point where they affect
  the transcript;
- streaming proof receipt is desirable;
- one canonical codec should define both proof bytes and absorbed bytes;
- removing the global nonce side channel and bit cursor is valuable; or
- proof transport should be delegated to Spongefish.

LEB128 has the stronger typical-size behavior for Akita's honest PoW search,
but its worst case can exceed main's packed stream.

### Neither encoding is universally smaller

The exact comparison is:

```text
LEB128 is smaller when
sum(actual LEB128 lengths) < ceil(sum(scheduled widths) / 8)

main is smaller when
sum(actual LEB128 lengths) > ceil(sum(scheduled widths) / 8)
```

They tie when the two sides are equal.

## 10. Why LEB128 aligns with native Spongefish

Spongefish's native model couples proof I/O and transcript state:

```text
prover_message(value):
    encode value
    append encoded bytes to the argument
    absorb encoded bytes

verifier_message(type):
    decode one value from the argument
    absorb the consumed bytes
    return the value
```

LEB128 fits this model because one nonce has a canonical, self-delimiting local
encoding. Spongefish can receive it exactly where the protocol needs it, and
the proof bytes cannot diverge from the absorbed bytes.

Main's aggregate stream does not fit this model naturally. Bits for different
future sites share bytes and live in a proof-level prefix. Retaining it beside
native Spongefish would require:

- a second proof cursor;
- external plan-aware bit parsing;
- explicit conversion from packed bits to a logical nonce;
- manual absorption at the later protocol position; and
- separate exhaustion checks for Spongefish and the nonce stream.

That hybrid can be implemented soundly, but it offloads less work to
Spongefish and recreates the synchronization boundary that native proof
messages are intended to remove.

LEB128 is not required by Spongefish. A fixed-width inline nonce codec would
also align with native messages. LEB128 is the selected compromise because it
combines native local framing with good honest-prover size.

## 11. Safety and accounting rules for the LEB128 design

The native design is safe only if it keeps three different size concepts
separate:

1. **Native-maximum planner objective.** Schedule selection adds the canonical
   per-message LEB128 maxima, `sum(ceil(width_i / 7))`. This aligns the modeled
   nonce cost with the native format, but the complete objective is still a
   model because terminal response pricing is not an exact wire bound.
2. **Native parser maximum.** Recursive and ordinary input boundaries use the
   sum of per-message LEB128 maxima, along with all other native grammar and
   terminal framing bounds.
3. **Actual honest size.** Profiling measures the concrete LEB128 lengths of
   the accepted nonces.

The planner objective is not a parser bound because it does not price every
component with the conservative grammar maximum. The actual honest size is not
a malicious-proof bound. Changing from aggregate packed-bit pricing to native
per-message maxima changes the optimization policy and therefore requires
regenerating every schedule catalog.

Encoding also does not replace verifier security checks. Acceptance still
requires:

- the exact next grinding-plan site;
- a canonical LEB128 integer;
- the scheduled range check;
- a valid PoW predicate or fold response;
- complete plan consumption; and
- Spongefish end-of-argument consumption.

## 12. Summary

Grinding is bounded search around Fiat--Shamir:

- PoW grinding charges work for challenge sites with algebraic loss.
- Fold-response grinding finds an honestly acceptable bounded response.

Main stores all resulting nonce integers in one globally packed bit stream.
That representation is deterministic and has the best aggregate worst-case
size, but it requires separate plan-aware transport and transcript replay.

PR #37 stores each nonzero nonce as an inline canonical unsigned LEB128
Spongefish message. That representation is locally parseable, couples receipt
with absorption, and is usually smaller for the honest PoW search because it
does not pay proof bytes for unused search slack. Its malicious maximum can be
larger, so native parser bounds must use per-message maxima.

For Akita's goal of offloading proof transport and transcript state to
Spongefish, LEB128 is the cleaner architecture. Main remains the stronger
choice if the sole objective is the smallest fixed worst-case nonce section.

## Relevant code and specifications

Current PR:

- `crates/akita-transcript/src/native/nonce.rs` — unsigned LEB128 codec;
- `crates/akita-transcript/src/native.rs` — native preview, commit, receipt,
  absorption, and challenge extraction;
- `crates/akita-types/src/transcript_grinding/native_replay.rs` — public plan
  cursor and verifier checks;
- `crates/akita-types/src/transcript_grinding_plan.rs` — canonical plan
  derivation and the complete component catalog;
- `crates/akita-schedules/src/runtime.rs` — native-maximum planner estimate
  versus complete native proof bound;
- `specs/transcript-grinding.md` — authoritative grinding contract;
- `book/src/how/transcript.md` — current transcript architecture and verifier
  requirements.

`origin/main`:

- `crates/akita-types/src/transcript_grinding/replay.rs` — packed writer,
  reader, bit cursor, and transcript adapter;
- `crates/akita-types/src/transcript_grinding_plan.rs` — public site and width
  derivation;
- `crates/akita-types/src/proof/wire.rs` — structured proof serialization;
- `crates/akita-transcript/src/grinding.rs` — fixed PoW transcript payload;
- `crates/akita-prover/src/protocol/core/prove.rs` — packed stream completion;
- `crates/akita-verifier/src/protocol/core/verify.rs` — packed replay and final
  cursor checks.
