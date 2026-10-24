# Spec: One stream per sparse fold group

| Field         | Value                          |
|---------------|--------------------------------|
| Author(s)     | Quang Dao, Claude              |
| Created       | 2026-10-20                     |
| Status        | implemented (draft PR)         |
| PR            | [#255](https://github.com/LayerZero-Labs/akita/pull/255) |
| Supersedes    | The indexed coordinate sampler formerly specified in [transcript-grinding](transcript-grinding.md) |
| Superseded-by |                                |
| Book-chapter  | how/transcript.md              |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**,
and **MAY** in this document are to be interpreted as described in
[BCP 14](https://www.rfc-editor.org/info/bcp14) when, and only when, they appear
in all capitals.

## Summary

A fold needs one sparse ring challenge per (claim, live block) pair. Today
each commitment group squeezes a 32-byte root from the transcript, and every
pair then gets its own SHAKE256 stream, `SHAKE256(root || index)`. Each
stream costs at least one Keccak permutation, so a fold with 2048 pairs costs
about 2048 permutations for the prover and the same for the verifier, even
though one challenge carries only about 130 bits.

This spec replaces that with one challenge message per group, squeezed from
the main transcript sponge. The message has a byte count fixed by the number
of pairs, the ring dimension and the challenge family. Each challenge is read
from the message as one fixed-width integer and decoded by exact
combinatorial unranking. Families without an operator-norm filter need no
rejection at all. The two operator-norm families use a fixed-length pool of
candidates and fail closed: a message that runs out is rejected, so the pool
size affects only how often an honest prover must retry.

Moving from independent per-pair streams to one message per group changes how
the fold is priced. One message now delivers every pair of the group at once,
so under Akita's rule that each challenge message must cost an attacker about
`2^128` hash calls per unit of success, the fold needs a small proof of work
before its group messages. Its difficulty is the least `g` with
`(sum of pairs in the fold) * 2^128 <= |C| * 2^g`, where `|C|` is the
challenge set size. That is 2 to 10 bits on the standard profile.

On the standard fp128 one-hot recursive profile (`nv = 36`) the verifier's
sparse-challenge hashing drops from 2601 Keccak permutations to about 420. The
prover drew every challenge twice (preview and live), about 5200
permutations; it now pays about 840 for the two message squeezes plus about
1736 expected for the proof of work. Whole proofs: the verifier goes from 3822
to 1642 permutations, and the prover from a mean of 16658 to 13900. The proof
grows by the proof-of-work nonces, about 11 bytes. Details are in
[Measured effect](#measured-effect).

## Intent

### Goal

Draw every sparse fold challenge of a group from one fixed-length transcript
message with the fewest squeezed bits, without weakening the fold's soundness
bound.

### Invariants

- **Single source.** One decoder in `akita-challenges` turns a message into
  challenges. The prover's search preview, the live prover and the verifier
  call it on the same bytes. The indexed SHAKE256 path is deleted.
- **Fixed length.** The byte count of a group message depends only on the
  public pair count, ring dimension, challenge family and operator-norm
  policy. It never depends on squeezed values.
- **Exact law.** Every accepted challenge is uniform on its accepted family:
  exactly, for unfiltered families; conditioned on the pool not running out,
  for the operator-norm families.
- **Fail closed.** A message whose pool runs out decodes to nothing. The
  verifier rejects and poisons its transcript. The prover treats it as a
  failed fold-response candidate and moves to the next nonce.
- **Priced per message.** Each fold's proof of work satisfies
  `L * 2^128 <= |C| * 2^g` with `L` the fold's total pair count, computed with
  Jolt's `grind_bits_for_loss`. No other check is relaxed.
- **Uniform signs.** The `w` sign bits of every challenge are exactly uniform
  and independent of the positions and of each other. The fold-response
  completeness argument ([fold-linf-rejection](fold-linf-rejection.md)) needs
  only this, so a later decoder change MUST keep it.
- **Verifier no-panic.** Message length, pool size, binomial tables and decode
  use checked arithmetic and fallible allocation. Malformed proofs produce
  `AkitaError`, never a panic.

### Non-goals

- Changing challenge families, ring dimensions, norms or any SIS parameter.
- Changing the fold-response search (12-bit nonce, at most 4096 trials).
- Changing any field-challenge site or its pricing.
- Recursion-guest work. The guest stack is off; the decoder choice below
  records how a guest would check it.

## Current sampler and its cost

For each group the transcript absorbs a public payload (group index, live
blocks, claims, pair count, ring dimension, the counts of ±1 and ±2
coefficients, method and policy domains), squeezes a 32-byte root, and then
expands pair `i` from `SHAKE256(root || u64_le(i))`. Positions use a partial
Fisher-Yates shuffle with bitmask rejection, one byte per sign. Operator-norm
families retry within the same stream, up to 4096 candidates.

Measured on 20,000 to 200,000 draws per family (sandbox, current code):

| Family (`d`, ±1, ±2) | Permutations per pair | Operator-norm acceptance |
|---|---|---|
| 64, 31, 10 | 1.000 | none |
| 128, 31, 0 | 1.000 | none |
| 256, 23, 0 | 1.000 | none |
| 512, 19, 0 | 1.000 | none |
| 1024, 16, 0 | 1.000 | none |
| 2048, 14, 0 | 1.000 | none |
| 64, 31, 11, threshold 18 | 1.490 | 0.6623 |
| 128, 31, 0, threshold 13 | 1.205 | 0.7908 |

Each pair uses one rate block (136 bytes) but only 40 to 100 of its bytes.

## Design

### Where the messages sit

Each nonterminal and terminal fold runs, in order:

1. The fold-response nonce, unchanged: the prover sends it, both sides absorb
   it.
2. The fold proof of work. With `g > 0` the prover runs Jolt's `grind(g)`
   (least nonce below `2^(g+7)`, LEB128, one prover message per byte, then
   `ceil(g/8)` check bytes squeezed). The verifier runs `check_grind(g)`. With
   `g = 0` nothing happens.
3. For each commitment group, in canonical order: the diagnostic context
   record and one squeeze of exactly `message_len` bytes. The public draw
   payload this spec first kept is gone: the descriptor and the group's
   position fix every field it held
   ([positional-word-aligned-transcript](positional-word-aligned-transcript.md)).

No prover message lies between the proof of work and the last group message,
so the work protects every group message of the fold (Jolt's `grind`
contract).

The bound schedule names the whole family, so one transcript state cannot
reinterpret a message under another family or method.

### Decoding one challenge

Let the family have ring dimension `d`, `a` coefficients of magnitude 1 and
`b` of magnitude 2 (`w = a + b`). The shell `S` is every such vector:

```text
|S| = binom(d, a) * binom(d - a, b) * 2^w.
```

A challenge occupies `K` bits of a little-endian bit stream (challenge `j`
uses bits `[jK, (j+1)K)` of the message, low bit of byte 0 first). The
decoder reads two fields from it, so no value wider than 128 bits is ever
formed even though `K` reaches 132:

- `signs`, the low `w` bits. The output lists the ±1 positions in ascending
  order, then the ±2 positions in ascending order; bit `t` of `signs`, least
  significant first, is the sign of output coefficient `t` (0 is `+`).
- `r`, the next `K - w` bits, is the position rank. Split `r = r_a + binom(d, a) * r_b` with
  `r_a < binom(d, a)`. `r_a` unranks to the ±1 positions as a subset of
  `0..d` in colexicographic order. `r_b` unranks to the ±2 positions as a
  subset of the `d - a` remaining positions, also colexicographic, mapped
  back through the sorted complement.

Colexicographic unranking of `r` into a `k`-subset: for `k` down to `1`, take
the largest `c` with `binom(c, k) <= r`, emit position `c`, subtract. The
binomial table `binom(n, k)` for `n <= d`, `k <= a` fits in `u128` for every
production family (largest entry about `2^118`).

This map is a bijection from `[0, |S|)` onto `S`.

**Unfiltered families.** `K = floor(log2 |S|)`. Every `K`-bit integer is
accepted, so the accepted family is the first `2^K` shell elements in this
order and `|C| = 2^K` exactly. There is no rejection and no pool:
`message_len = 8 * ceil(W * K / 64)`, a whole number of words.

| `d` | ±1 | ±2 | `log2 |S|` | `K` |
|---|---|---|---|---|
| 64 | 31 | 10 | 128.088 | 128 |
| 128 | 31 | 0 | 129.621 | 129 |
| 256 | 23 | 0 | 131.078 | 131 |
| 512 | 19 | 0 | 132.757 | 132 |
| 1024 | 16 | 0 | 131.580 | 131 |
| 2048 | 14 | 0 | 131.593 | 131 |

A subset of the shell keeps every property the fold argument uses: every
difference of two distinct elements is still a unit (it was for the whole
shell), and every element has the same norms.

**Operator-norm families.** The certificate counts the predicate-accepted
part of the whole shell, so these families decode over the whole shell:
`K = ceil(log2 |S|)`, a candidate with `r >= binom(d,a) * binom(d-a,b)` is
rejected, and an in-range candidate is accepted when the certified
fixed-point predicate accepts it. The first `W` accepted candidates, in
message order, are the challenges.

| Family | `K` | in-range rate | predicate rate (measured) | combined rate | rate used for sizing `p` | certified accepted size |
|---|---|---|---|---|---|---|
| D64 (31,11), threshold 18 | 131 | 0.55565 | 0.6623 | 0.3680 | `0.3611 <= 0.65 * 0.55565` | `>= 2^128.062` |
| D128 (31,0), threshold 13 | 130 | 0.76918 | 0.7908 | 0.6083 | `0.5999 <= 0.78 * 0.76918` | `>= 2^128.563` |

The certificates prove predicate rates of at least 0.1305 (D64) and 0.3694
(D128); the measured rates above come from 20,000 draws and are used only to
size the pool.

### Pool size for the operator-norm families

The message holds `B` candidates, `message_len = 8 * ceil(B * K / 64)`. Each
candidate is accepted independently with probability `p` (the in-range part
is exact; the predicate part is the measured rate, lowered for sizing). Running out means fewer than `W` acceptances
among `B`. `B` is the least value passing either integer test:

- **Union bound.** `P[X <= W-1] <= binom(B, W-1) * (1-p)^(B-W+1)`. With
  `binom(B, W-1) <= 2^((W-1) * bitlen(B))` and a rational `r <= -log2(1-p)`
  (0.64 for D64, 1.32 for D128), require
  `(B - W + 1) * r >= 161 + (W - 1) * bitlen(B)`.
- **Chernoff bound.** With `p = u/v` and `mu = B p`,
  `P[X < W] <= exp(-(mu - W)^2 / (2 mu))`. Require `B u > W v` and
  `(B u - W v)^2 >= 224 * B * u * v` (`224 >= 2 * 161 * ln 2 = 223.2`).

Both are exact integer conditions in `u128`. The union bound wins for small
`W`, the Chernoff bound for large `W`:

| Family | `W` | `B` | message bytes | exact tail `log2 P[run out]` |
|---|---|---|---|---|
| D64 | 4 | 297 | 4864 | -172.4 |
| D64 | 6 | 327 | 5360 | -180.7 |
| D64 | 8 | 357 | 5848 | -189.5 |
| D64 | 64 | 942 | 15432 | -331.3 |
| D128 | 8 | 172 | 2800 | -183.7 |
| D128 | 64 | 567 | 9216 | -431.4 |

The pool size changes only completeness, and the soundness argument charges
nothing for it. Whether a message runs out depends only on which candidates
are accepted, never on the accepted values. So, conditioned on not running
out, the first `W` accepted candidates are independent and exactly uniform on
the accepted family; when it runs out, the verifier rejects. For any bad set
of challenge vectors, `P[accept and bad] <= P_ideal[bad]`, with no bias term.

The sizing rate is measured, not proven, so the `2^-161` column is a
completeness estimate. At the certified lower rates the same pools still run
out with probability at most `2^-32` (D64, `W = 64`), `2^-44` (D64,
`W <= 8`) and `2^-80` (D128). Running out costs the honest prover one more
fold-response candidate, never a failed proof. An exact tail at the certified
rates would need pools 2 to 3 times larger (D64, `W = 8`: 988 candidates)
for no soundness gain.

### Fold proof-of-work pricing

For fold level `l` with groups `g`, `W_l = sum_g W[l,g]` and
`K_l = min_g K*[l,g]`, where `K*` is the guaranteed set-size exponent:
`K` for unfiltered families and 128 for the certified operator-norm families.
Every other proof of work in the plan is priced against the challenge field;
this one is priced against the sparse set, so it is its own run kind,
`FoldChallengeWork`, which records `K_l` and is validated against it. The
plan prices one proof of work per fold:

```text
g_l = grind_bits_for_loss(W_l, ChallengeSetSize::power_of_two(K_l), 128)
    = least g with W_l * 2^128 <= 2^K_l * 2^g.
```

Standard fp128 one-hot recursive profile, `nv = 36`:

| Fold | Groups: pairs (family) | `W` | `K` | `g` | Message bytes |
|---|---|---|---|---|---|
| 0 | 2048 (128, 31, 0; packing subring) | 2048 | 129 | 10 | 33024 |
| 1 | 256 + 150 (64, 31, 10; packing subring) | 406 | 128 | 9 | 4096 + 2400 |
| 2 | 79 (64, 31, 10) | 79 | 128 | 7 | 1264 |
| 3 | 30 (64, 31, 10) | 30 | 128 | 5 | 480 |
| 4 | 9 (64, 31, 10) | 9 | 128 | 4 | 144 |
| 5 | 8 (D64 operator norm) | 8 | 128 | 3 | 5846 |
| 6 | 4 (D64 operator norm) | 4 | 128 | 2 | 4864 |
| terminal | 6 (D64 operator norm) | 6 | 128 | 3 | 5355 |

The expected honest search is `2^g` permutations per fold-response
candidate, 1732 in total for one candidate per fold. The verifier pays one
check per fold.

## Soundness

This section checks the change against Akita's fold argument
([transcript-grinding](transcript-grinding.md#fold-extraction-from-accepting-children)).
That argument has two parts: deterministic extraction from an accepting
transcript tree, and a random-oracle bound on failing to build the tree.

### What extraction needs (unchanged)

At one fold, extraction needs the original accepting run as the central
child, and for every pair one more accepting child that differs from the
central child only in that pair's challenge. Subtracting the two fold
relations isolates that pair's source, because every nonzero difference of
two accepted challenges is a unit. Each child carries its own accepting
subtree for the later checks; later messages need not agree between children.
None of this depends on how challenges are sampled, so the deterministic part
is unchanged. A subset of the shell keeps the unit property.

### Groups are drawn in sequence

Group `g+1`'s message is squeezed after group `g`'s, so it depends on group
`g`'s bytes. Changing a pair in group 0 therefore also changes all of group
1's challenges. This is harmless:

- Each group's fold relation involves only that group's challenges, sources
  and commitments, all fixed before the fold's challenges. Extracting group
  0's sources needs children that differ in one group-0 pair; what group 1
  does in those children is part of their own subtrees, like any later
  challenge.
- Extracting group 1's sources needs children that share the central group-0
  message and differ in one group-1 pair. The extractor reprograms group 1's
  message at the central prefix.

The tree at a fold is therefore one stage per group, in order, with the
central branch continuing to the next group. It still has
`1 + sum_g W[l,g]` children per fold, as before.

### The extractor can program one pair

The extractor needs to replace one challenge in a message while keeping every
other challenge with its correct distribution. In the ideal sponge model, a
squeeze at a given transcript prefix is one random answer. Given a uniform
answer:

- **Unfiltered.** The message is `W` independent uniform `K`-bit integers, and
  the decoder is a bijection onto `C`. Replacing integer `j` by the encoding of
  a fresh uniform element of `C` gives exactly the distribution of a message
  that differs only at pair `j`.
- **Operator norm.** Each candidate is accepted independently with the same
  probability, and an accepted candidate is uniform on the accepted family
  and independent of which candidates were accepted. Replacing the `j`-th
  accepted candidate by the encoding of a fresh uniform element of the
  accepted family keeps the acceptance pattern, every other candidate and the
  unused tail, so it is again exact. A message that runs out is rejected and
  contributes no accepting child.

This is simpler than the indexed case, which had to program a SHAKE tape
conditioned on a first-success position.

### Failure bound per message

Fix a prefix right after the fold proof of work. For one pair `j` in group
`g`, conditioned on every other pair and on everything before, at most one
value of pair `j` makes the extractor's fork at `j` fail (the original
matching-input argument). The answer has `|C[l,g]|` equally likely values, so
the fork at `j` fails with probability at most `1/|C[l,g]|`. A union over the
pairs of the fold gives at most

```text
L_l / 2^K_l,   L_l = sum_g W[l,g],
```

for one prefix. This is the fold's loss per message, `L_l` in the usual
"bad fraction `L/|C|`" form.

### Why the fold now needs a proof of work

Akita prices every challenge message so that one hash call buys at most
`2^-128` of success: `L * 2^128 <= |E| * 2^g`, with every proof-of-work trial
counted as a call. The indexed sampler met this with no work, because each
pair was its own oracle query with loss `1/|C| <= 2^-128`. One stream per
group makes the whole fold one priced unit with loss `L_l / 2^K_l`, which is
up to `2^11` times larger on the profile above. The proof of work restores the
rate: a fresh fold answer now costs `2^g_l` check queries in expectation, and

```text
2^-g_l * L_l / 2^K_l <= 2^-128.
```

The total bound has the same shape as before. With `Q` the adversary's
hash calls and `R` the messages the final verifier completes itself, the fold
term is at most `(Q + R) * 2^-128` summed with the other sites, the same rate
the indexed theorem gave per coordinate. An adversary that reads whole
vectors pays about `W_l * 2^(128 - K_l)` calls per vector either way.

The honest prover pays about the same: the work replaces the per-pair
streams. The saving is on the verifier, which pays one check per fold
instead of one permutation per pair.

A direct consequence: no single-message design can price below
`L_l * 2^(128 - K_l)` calls per fold for the honest prover, because an
attacker can always do what the honest prover does. The indexed sampler
already sat at that bound for `K = 128` families. The stream design matches
it and gains wherever `K > 128`, and on the verifier everywhere.

### Other terms

- **Group-root collisions disappear.** There is no 32-byte root; each message
  is the sponge's answer at its own prefix.
- **Running out** is fail-closed and independent of the accepted values, so it
  adds no soundness term (see the pool section).
- **Blake2b.** Jolt's proof of work is argued per permutation call for Keccak;
  the Blake2b bridge needs its own argument. Every existing Akita
  proof-of-work site already relies on it. With `transcript-blake2b`, fold
  soundness now relies on it too: both the fold proof of work and the group
  messages come from the Blake2b sponge, where the indexed sampler used
  SHAKE256 streams.
- **Packing consistency check.** The packed consistency polynomial check is a
  field check with its own entry; it is unchanged.

### Verdict of this section

The extraction argument is unchanged. The per-message rate requires a
per-fold proof of work of `g_l` bits, specified above; without it the change
would miss the target. With it the change meets the same target as every
other Akita message: about `2^128` hash calls per unit of success. It is not
stronger per call than the indexed sampler everywhere: at a fold whose
smallest challenge set has `2^K_l` elements with `K_l > 128`, the indexed sampler gave
`2^-K_l` per call and this design gives `2^-128`, a margin above the target
of up to `K_l - 128` bits (one bit at fold 0 of the standard profile).
Pricing against `K_l` instead would cost fold 0 another 1024 expected
permutations per fold-response candidate for margin above the target, which
no other site gets.

## Audit

An independent read-only soundness audit (xhigh) of this design returned
**sound with issues**: it meets the `2^128` target and found nothing that
breaks soundness. Its findings and their resolution:

| Finding | Resolution |
|---|---|
| The pool's `2^-161` rests on a measured predicate rate | Soundness now rests on the fail-closed argument, which needs no rate; the measured rate sizes completeness only, with certified-rate failure bounds stated |
| The preview could not call Jolt's search | The preview runs Jolt's own `grind` on a detached `ProverState` built from the copied sponge |
| "At least as strong" was false at folds with `K > 128` | Reworded: same `2^128` target, smaller per-call margin of up to `K - 128` bits |
| Rates rounded up | 0.3611 and 0.5999 |
| `R` can exceed 128 bits | Signs and rank are read as separate fields |
| Sign order ambiguous | Ascending positions within each magnitude, ±1 first, least significant bit first |
| Guest advice checked one subset | Full check list in the decoder section |
| Prover aborted when a pool ran out | It moves to the next fold-response nonce |
| Plan validation priced every proof of work against the field | New run kind that records and validates `K_l` |
| Payload domain not bumped | `akita/fold-challenge-round/v2` |
| Blake2b caveat, doubled prover work, verifier early stop, squeeze limit, sign invariant | Stated in this spec; the length limit is enforced; the live prover reuses the preview's work instead of searching again |

## Prover parallelism

The squeeze is sequential, but cheap: one Keccak permutation per 136 bytes.
Decoding is parallel. For unfiltered families every challenge has a fixed
bit offset, so the decoder splits the pairs across threads (Rayon, above the
existing work threshold). For operator-norm families the predicate runs on all
candidates in parallel, and a sequential prefix pass picks the first `W`
accepted. The verifier, which usually runs without threads, stops at the
`W`-th accepted candidate.

Each fold-response candidate runs the fold proof of work on a detached prover
state built from a copy of the sponge (Spongefish's `ProverState::from`), by
calling Jolt's own `grind`; Akita has no second copy of the search. The
preview keeps the sponge right after the work and the argument bytes the work
sent (`FoldWork`). For the accepted candidate, the live prover sends the
fold-response nonce, sends those argument bytes one prover message each (as
`grind` does), and takes the preview's sponge, which already holds the work's
check. The live prover therefore never searches again, and no Jolt API change
is needed: Spongefish exposes the sponge state under the `yolocrypto` feature
that the preview already uses. Every candidate pays about `2^g` expected
permutations for the work, the accepted one included, and nothing more. The
live prover then draws the challenges through the real channel and compares
them with the preview's, so a divergence is an honest-prover error, never a
different proof.

## Protocol and byte effects

- **Proof bytes.** Each fold with `g > 0` sends a LEB128 nonce below
  `2^(g+7)`: about 11 bytes on the standard profile. Nothing else changes.
- **Plan.** A new `GrindingSite::FoldChallengeWork { level }` run, of the new
  kind `FoldChallengeWork`, follows each `FoldResponse` and records its set
  exponent `K_l`. `FoldChallengeGroup` runs keep their pair count but stand
  for one message. `FOLD_COORDINATE_ORACLE_REVISION` moves to 2 and
  `GRINDING_QUERY_POLICY_REVISION` to 5, so plan digests change.
- **Domains.** The fold payload and its domain are removed, and the session
  tag is the padded protocol name; the version doc lists the message codec
  ([positional-word-aligned-transcript](positional-word-aligned-transcript.md)).
- **Protocol version.** Proof bytes change, so the protocol version moves to
  9. Riding on version 8 would be safe only if this lands together with #253,
  but #253 is complete and audited on its own and can merge first. A distinct
  tag costs nothing because the known answers are regenerated anyway. The
  embedded-transcript work (A4) also plans version 9; whichever lands second
  takes 10.
- **Transcript surface.** `ProofChannel` gains `challenge_message(len,
  decode)`, a squeeze of exactly `len` bytes through Spongefish's
  `verifier_message_as` that poisons the verifier when decoding fails, and
  the fold preview gains the same call and a proof-of-work step. The 32-byte
  `seed` and `fold_root` are deleted. A message longer than Jolt's
  `2^24`-byte squeeze limit is a setup error.

## Decoder choice: fewest squeezed bits

Bits per challenge, by method:

| Family | Information `log2 |S|` | Current indexed (bytes read) | One stream, bitmask rejection (expected, before pool) | One stream, unranking into a power-of-two subset |
|---|---|---|---|---|
| 64, 31, 10 | 128.09 | 762 (one 1088-bit block) | 356 | **128** |
| 128, 31, 0 | 129.62 | 531 | 278 | **129** |
| 256, 23, 0 | 131.08 | 376 | 215 | **131** |
| 512, 19, 0 | 132.76 | 462 | 193 | **132** |
| 1024, 16, 0 | 131.58 | 386 | 177 | **131** |
| 2048, 14, 0 | 131.59 | 337 | 169 | **131** |
| 64, 31, 11 (operator norm) | 130.15 | 781 per try | 364 per try | **131 per try** (rank and predicate rejection) |

Ranked recommendation:

1. **Unranking into a power-of-two subset** (operator-norm families: whole
   shell with rank rejection). It reads exactly `K` bits, the minimum for a set
   of `2^K` elements, needs no pool and has no rejection for every unfiltered
   family, and gives exactly uniform output. Keeping `2^K` elements loses no
   pricing, because Akita already prices with the integer floor
   `2^K <= |S|`. Native cost is a `u128` binomial table of `(d+1)(a+1)`
   entries and `w * log2 d` comparisons per challenge, well under a
   microsecond.
2. **Rank-check advice** for a recursion guest. Same transcript bits as
   option 1: the host supplies the decoded positions as untrusted advice, and
   the guest checks them instead of searching. The guest MUST check all of:
   the ±1 positions are strictly increasing and their colexicographic rank
   `r_a = sum_k binom(p_k, k)` is below `binom(d, a)`; the ±2 positions are
   given as strictly increasing indices into the complement of the ±1 set,
   with rank `r_b` below `binom(d - a, b)`, and are mapped back through that
   complement; and `r_a + binom(d, a) * r_b` equals the squeezed rank. Checking
   one subset alone would let a prover choose the ±2 positions. It is a guest
   implementation of option 1, not a protocol choice. Putting the positions in the proof instead would add about
   `W * w * log2 d` bits (55 KB at fold 0) and is rejected.
3. **Bitmask rejection in one stream.** 1.3 to 2.8 times the bits of option 1,
   plus a pool for the variable bit consumption. Its only advantage is a
   simpler decoder.

## Measured effect

Keccak permutations per proof on `AKITA_MODE=onehot_fp128 AKITA_NUM_VARS=36
AKITA_SETUP_MODE=recursive` with the Keccak sponge, counted by instrumented
copies of the `keccak` crates (every Keccak-f[1600] call, including the old
SHAKE256 sampler). "Before" is `c94abf699`, the base of this change. The
prover count varies from proof to proof because every proof of work in the
plan has a random search length, so each side proves the same witness under
12 different transcript labels.

| | Before | After |
|---|---|---|
| Verifier (exact) | 3822 (sponge 1219, SHAKE256 sampler 2601, other 2) | 1642 (sponge 1640, other 2) |
| Prover, mean of 12 proofs | 16658 (range 14705 to 19941) | 13900 (range 10927 to 16322) |

The prover's mean drop, about 2760 with a standard error of about 700,
matches the prediction: the 5200 sampler permutations go away, and the two
message squeezes (about 840) and the fold proofs of work (about 1736) come in.

Fold-response candidates per fold on the same 12 proofs: every fold accepted
its first candidate except fold 4, which needed two candidates in 3 of 12
proofs (mean 1.25). The expected proof-of-work cost per proof is therefore
`sum over folds of (mean candidates) * 2^g`, about 1736 permutations, with
`g = 10, 9, 7, 5, 4, 3, 2, 3` for folds 0 to 7. The live prover does not
repeat the search for the accepted candidate.

## Code changes

- `akita-challenges`: new `sampler/stream.rs` with the message length, pool
  sizing, binomial table and decoder; `fold_draw.rs` draws one message per
  group; delete `sampler/xof.rs`, `position_sample.rs`, `signed_sparse.rs` and
  the indexed worker. Keep the operator-norm predicate.
- `akita-transcript`: `ProofChannel::challenge_message`; `FoldPreview`
  squeezes messages and runs Jolt's `grind` on a detached prover state; delete
  `seed`, `fold_root` and `FOLD_CHALLENGE_SEED_LEN`; protocol version 9.
- `akita-params`: `FoldChallengeFamily` lookups on group and terminal
  parameters, shared by pricing and draws; the `FoldChallengeWork` site, run
  kind and pricing; revisions.
- `akita-types`, `akita-prover`, `akita-verifier`: replay the new site, run
  the work in the fold-response search, draw messages.
- Known answers, plan digests and proof pins regenerated.

## Book and spec updates

- `book/src/how/transcript.md`: replace "The group root" and "One indexed
  stream per coordinate" with the message layout, decoder and pool.
- `book/src/how/security.md` and `book/src/foundations/pcs-and-binding.md`:
  replace the indexed-stream explanation of coordinate forks with the
  programmable-message argument and the per-fold proof of work.
- `book/src/how/proving/root-fold-ring-switch.md`: update the fork pointer.
- `specs/transcript-grinding.md`: replace "Indexed sparse challenges" through
  "Established result" with this argument, and price the fold work in the
  total accounting.
- `specs/subring-coefficient-packing.md`, `specs/jolt-transcript-adoption.md`:
  pointers.

## Acceptance criteria

- Decoder round trip: every rank in a sampled set decodes to a shell element
  whose colexicographic rank is the input; distinct ranks give distinct
  elements; edge ranks `0` and `2^K - 1` decode.
- Unfiltered messages: `message_len = 8 * ceil(W K / 64)`; all-zero and all-one
  messages decode.
- Operator-norm messages: pool size matches the table; a message built to
  have too few accepted candidates decodes to nothing and the verifier
  rejects; accepted outputs satisfy the predicate.
- Prover preview, live prover and verifier draw identical challenges.
- A tampered proof-of-work nonce and a truncated proof are rejected with
  `AkitaError`.
- Plan: one `FoldChallengeWork` per fold with the bits in the table above.
- Known answers and proof pins regenerated under both sponges.
