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
| Revised       | 2026-11-05: full-shell rank words for every family ([Full-shell revision](#full-shell-revision)) |

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
from the message as one fixed-width integer: a rank word is mapped onto the
position ranks by Lemire's multiply-shift reduction, then decoded by exact
combinatorial unranking. Every family samples its whole shell. A message is a
fixed-length pool of candidates and fails closed: a candidate is rejected when
its rank word falls in the reduction's short rejection band (probability below
`2^-10`, below `2^-16` for every family the standard profile uses) or, for
the two operator-norm families, when the certified predicate rejects it. A
message that runs out is rejected, so the pool size affects only how often an
honest prover must retry.

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
message, uniformly over the family's whole shell (or the certified accepted
part of it), with few squeezed bits and without weakening the fold's soundness
bound.

### Invariants

- **Single source.** One decoder in `akita-challenges` turns a message into
  challenges. The prover's search preview, the live prover and the verifier
  call it on the same bytes. The indexed SHAKE256 path is deleted.
- **Fixed length.** The byte count of a group message depends only on the
  public pair count, ring dimension, challenge family and operator-norm
  policy. It never depends on squeezed values.
- **Exact law.** Conditioned on the pool not running out, the challenges are
  independent and exactly uniform on the accepted family: the whole shell for
  an unfiltered family, the predicate's accepted part of the shell for an
  operator-norm family. No family is truncated to a power-of-two prefix.
- **Fail closed.** A message whose pool runs out decodes to nothing. The
  verifier rejects and poisons its transcript. The prover treats it as a
  failed fold-response candidate and moves to the next nonce.
- **Priced per message.** Each fold's proof of work satisfies
  `L * 2^128 <= |C| * 2^g` with `L` the fold's total pair count, computed with
  Jolt's `grind_bits_for_loss`. No other check is relaxed.
- **Uniform signs.** For an unfiltered family, the `w` sign bits of every
  challenge are exactly uniform and independent of the positions and of each
  other. Rank-word rejection reads only the rank word, so it keeps this. The fold-response completeness argument
  ([fold-linf-rejection](fold-linf-rejection.md)) needs only this, so a later
  decoder change MUST keep it. An operator-norm family filters on the whole
  vector, signs included, so its accepted signs are uniform only jointly with
  the positions on the accepted set. Its completeness rests on the measured
  challenge covariance and the planner's source envelope instead
  ([selective-l2-fold-security-sizing](selective-l2-fold-security-sizing.md#measured-challenge-covariance)).
  Soundness assumes neither: the verifier enforces the norm checks for every
  accepted challenge.
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

Measured on 20,000 to 200,000 draws per family (sandbox, the sampler this
spec replaces); the operator-norm acceptance rates come from 20,000 draws:

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
2. The fold proof of work. With `g > 0` the prover runs Jolt's `grind(g)`:
   the least nonce below `2^(g+7)` whose 8 squeezed check bytes, read as a
   little-endian `u64`, have `g` zero low bits. The nonce is one prover
   message, canonical unsigned LEB128 in the proof string and absorbed as one
   little-endian `u64` word. The verifier runs `check_grind(g)`. With `g = 0`
   nothing happens.
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

Let `R = binom(d, a) * binom(d - a, b)` be the number of position patterns,
so `|S| = R * 2^w`. A candidate occupies `K = w + m` bits of a little-endian
bit stream (candidate `j` uses bits `[jK, (j+1)K)` of the message, low bit of
byte 0 first), with

```text
m = min(ceil(log2 R) + 16, 128).
```

The shell setup MUST reject a family with `m - ceil(log2 R) < 8`; no
production family comes close. The decoder reads two fields, so no field is
wider than 128 bits even though `K` reaches 148:

- `signs`, the low `w` bits. The output lists the ±1 positions in ascending
  order, then the ±2 positions in ascending order; bit `t` of `signs`, least
  significant first, is the sign of output coefficient `t` (0 is `+`).
- `x`, the next `m` bits, is the rank word. Form the exact 256-bit product
  `x * R = hi * 2^m + lo` with `lo < 2^m`. With `t = 2^m mod R`, fixed per
  family, the candidate is rejected when `lo < t`; otherwise its position rank
  is `r = hi`.

This is Lemire's multiply-shift reduction with a precomputed threshold. Each
`r` in `[0, R)` has exactly `floor(2^m / R)` accepted words `x`, so an
accepted rank is exactly uniform, and a uniform `x` is rejected with
probability `q = t / 2^m < R / 2^m <= 2^-(m - ceil(log2 R))`.

Split `r = r_a + binom(d, a) * r_b` with `r_a < binom(d, a)`. `r_a` unranks to
the ±1 positions as a subset of `0..d` in colexicographic order. `r_b` unranks
to the ±2 positions as a subset of the `d - a` remaining positions, also
colexicographic, mapped back through the sorted complement.

Colexicographic unranking of `r` into a `k`-subset: for `k` down to `1`, take
the largest `c` with `binom(c, k) <= r`, emit position `c`, subtract. The
binomial table `binom(n, k)` for `n <= d`, `k <= a` fits in `u128` for every
production family (largest entry about `2^118`).

Signs and rank together are a bijection from `[0, 2^w) x [0, R)` onto `S`,
so an accepted candidate is exactly uniform on the whole shell.

**Unfiltered families.** Every candidate whose rank word is accepted is a
challenge; the first `W`, in message order, are the group's challenges. The
accepted family is the whole shell, `|C| = |S|`. Fold pricing keeps the
integer floor `K* = floor(log2 |S|) <= log2 |C|`
([Fold proof-of-work pricing](#fold-proof-of-work-pricing)).

| `d` | ±1 | ±2 | `log2 |S|` | `K*` | `ceil(log2 R)` | `m` | `K = w + m` | `log2 q` |
|---|---|---|---|---|---|---|---|---|
| 64 | 31 | 10 | 128.088 | 128 | 88 | 104 | 145 | -18.47 |
| 128 | 31 | 0 | 129.621 | 129 | 99 | 115 | 146 | -16.81 |
| 256 | 23 | 0 | 131.078 | 131 | 109 | 125 | 148 | below -16 |
| 512 | 19 | 0 | 132.757 | 132 | 114 | 128 | 147 | below -14 |
| 1024 | 16 | 0 | 131.580 | 131 | 116 | 128 | 144 | below -12 |
| 2048 | 14 | 0 | 131.593 | 131 | 118 | 128 | 142 | below -10 |

The shell has every property the fold argument uses: every difference of two
distinct elements is a unit, and every element has the same norms. Unlike a
power-of-two prefix of the colexicographic order, it is also closed under
negacyclic rotation and the ring automorphisms, and every coefficient
position is used: a `2^K` prefix never set `X^127` at D128 or `X^499` and above
at D512 ([Full-shell revision](#full-shell-revision)).

**Operator-norm families.** The certificate counts the predicate-accepted
part of the whole shell. A candidate whose rank word is accepted is accepted
when the certified fixed-point predicate also accepts it. The first `W`
accepted candidates, in message order, are the challenges. Both families have
the full slack `m = ceil(log2 R) + 16`, and the shell setup MUST check this,
because their sizing rates assume it.

| Family | `K` | rank-word acceptance | predicate rate (measured) | rate used for sizing `p` | certified accepted size |
|---|---|---|---|---|---|
| D64 (31,11), threshold 18 | 147 | `1 - 2^-18.92` | 0.6623 | `0.6499 <= 0.65 * (1 - 2^-16)` | `>= 2^128.062` |
| D128 (31,0), threshold 13 | 146 | `1 - 2^-16.81` | 0.7908 | `0.7799 <= 0.78 * (1 - 2^-16)` | `>= 2^128.563` |

The certificates prove predicate rates of at least 0.2349 (D64) and 0.4802
(D128) on the shell, that is `2^128.062 / |S|` and `2^128.563 / |S|`. The
measured predicate rates come from 20,000 draws and are used only to size the
pool. The sizing rates 0.65 and 0.78 sit about 3.7 and 3.8 standard errors
below them. The accepted distribution is the same as before this revision,
which read a `ceil(log2 |S|)`-bit candidate and rejected out-of-range ranks;
only the encoding and the pool sizes change.

### Pool size

The message holds `B` candidates, `message_len = 8 * ceil(B * K / 64)`. Each
candidate is accepted independently with probability at least `p`: for an
unfiltered family `p = 1 - q` exactly, for an operator-norm family the
rank-word acceptance times the measured predicate rate, lowered for sizing.
Running out means fewer than `W` acceptances among `B`. `B` is the least value
passing either integer test:

- **Union bound.** Running out means at least `k = B - W + 1` rejections, so
  `P[X <= W-1] <= binom(B, k) * (1-p)^k`. With
  `binom(B, k) = binom(B, W-1) <= 2^(min(W-1, k) * bitlen(B))` and a rational
  `r <= -log2(1-p)`, require
  `(B - W + 1) * r >= 161 + min(W - 1, B - W + 1) * bitlen(B)`. For an
  unfiltered family `r = m - bitlen(t)`, an integer, since
  `-log2 q = m - log2 t`; for the operator-norm families `r` is 1.51 (D64) and
  2.18 (D128). The `min` matters only when rejections are rare: it lets an
  unfiltered pool carry a few spare candidates instead of about `W`.
- **Chernoff bound.** With `p = u/v` and `mu = B p`,
  `P[X < W] <= exp(-(mu - W)^2 / (2 mu))`. Require `B u > W v` and
  `(B u - W v)^2 >= 224 * B * u * v` (`224 >= 2 * 161 * ln 2 = 223.2`). For
  an unfiltered family `u / v = 1 - 2^-s` with `s = min(m - bitlen(t), 16)`.

Both are exact integer conditions in `u128`. The union bound wins for small
`W`, the Chernoff bound for large `W`. An unfiltered family with `t = 0` (a
power-of-two `R`; no production family) rejects nothing and has `B = W`.

| Family | `W` | `B` | message bytes | `log2 P[run out]` at the sizing `p`, at most |
|---|---|---|---|---|
| D64 (31,10) | 9 | 20 | 368 | -204.7 |
| D64 (31,10) | 79 | 93 | 1688 | -220.9 |
| D64 (31,10) | 256 | 273 | 4952 | -240.1 |
| D128 (31,0) | 3 | 13 | 240 | -178.6 |
| D128 (31,0) | 2048 | 2088 | 38112 | -402.3 |
| D64 operator norm | 4 | 124 | 2280 | -166.8 |
| D64 operator norm | 6 | 139 | 2560 | -177.3 |
| D64 operator norm | 8 | 151 | 2776 | -184.1 |
| D64 operator norm | 64 | 524 | 9632 | -463.5 |
| D128 operator norm | 8 | 104 | 1904 | -180.0 |
| D128 operator norm | 64 | 397 | 7248 | -505.5 |

Each value is the exact binomial tail `P[Bin(B, p) < W]`, computed with exact
rationals and rounded up. For the unfiltered rows `p = 1 - t / 2^m` is the
exact acceptance rate, so their tails are exact, not estimates; the Chernoff
test uses the rational lower bound `1 - 2^-s` on it. For the operator-norm
rows `p` is the sizing rate `6499/10000` or `7799/10000`.

The pool size changes only completeness, and the soundness argument charges
nothing for it. Whether a message runs out depends only on which candidates
are accepted, never on the accepted values. So, conditioned on not running
out, the first `W` accepted candidates are independent and exactly uniform on
the accepted family; when it runs out, the verifier rejects. For any bad set
of challenge vectors, `P[accept and bad] <= P_ideal[bad]`, with no bias term.

For the operator-norm families the sizing rate is measured, not proven, so
their `2^-161` is a completeness estimate. At the certified predicate rates
the same pools give a bound that depends on the count `W`. The rate is the
certificate's exact `p0` times the rank-word acceptance `1 - t / 2^m`; a range
row is the worst count in the range, and every value is rounded up:

| Family | `W` | `log2 P[run out]` at the certified rate, at most |
|---|---|---|
| D64 | `<= 64` | -28.4 (worst at `W = 22`, `B = 239`) |
| D64 | 128 | -9.6 |
| D64 | `>= 200` | no useful bound (`B * 0.2349` is about `W`) |
| D128 | `<= 64` | -64.2 (worst at `W = 11`, `B = 116`) |
| D128 | 191 | -91.8 |
| D128 | 256 | -70.3 |
| D128 | 512 | -23.6 |

A count past these ranges is complete only under the measured rate. Running
out costs the honest prover one more fold-response candidate, never a failed
proof. The pools are smaller than before this revision (D64, `W = 8`: 151
candidates instead of 357), so the certified-rate bounds are a little weaker
at small `W` (D64 `W <= 32`: -28.4 instead of -43.8); they were never part of
soundness.

### Fold proof-of-work pricing

For fold level `l` with groups `g`, `W_l = sum_g W[l,g]` and
`K_l = min_g K*[l,g]`, where `K*` is the guaranteed set-size exponent:
`floor(log2 |S|)` for unfiltered families and 128 for the certified
operator-norm families. Pricing at the floor is conservative now that the
accepted family is the whole shell, and it leaves every `g` unchanged.
Every other proof of work in the plan is priced against the challenge field;
this one is priced against the sparse set, so it is its own run kind,
`FoldChallengeWork`, which records `K_l` and is validated against it. The
plan prices one proof of work per fold:

```text
g_l = grind_bits_for_loss(W_l, ChallengeSetSize::power_of_two(K_l), 128)
    = least g with W_l * 2^128 <= 2^K_l * 2^g.
```

Standard fp128 one-hot recursive profile, `nv = 36`:

| Fold | Groups: pairs (family) | `W` | `K*` | `g` | Message bytes | Before the full-shell revision |
|---|---|---|---|---|---|---|
| 0 | 2048 (128, 31, 0; packing subring) | 2048 | 129 | 10 | 38112 | 33024 |
| 1 | 256 + 150 (64, 31, 10; packing subring) | 406 | 128 | 9 | 4952 + 3016 | 4096 + 2400 |
| 2 | 79 (64, 31, 10) | 79 | 128 | 7 | 1688 | 1264 |
| 3 | 30 (64, 31, 10) | 30 | 128 | 5 | 784 | 480 |
| 4 | 9 (64, 31, 10) | 9 | 128 | 4 | 368 | 144 |
| 5 | 8 (D64 operator norm) | 8 | 128 | 3 | 2776 | 5848 |
| 6 | 4 (D64 operator norm) | 4 | 128 | 2 | 2280 | 4864 |
| terminal | 6 (D64 operator norm) | 6 | 128 | 3 | 2560 | 5360 |
| **Total** | | | | | **56536** | **57480** |

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

Every family is a pool. Each candidate is accepted independently with the
same probability, and an accepted candidate is uniform on the accepted family
`C` (the shell, or its predicate-accepted part) and independent of which
candidates were accepted. Replacing the `j`-th accepted candidate by an
encoding of a fresh uniform element `c` of `C` keeps the acceptance pattern, every other candidate
and the unused tail, so it is exact. The encoding copies the signs of `c` and
picks its rank word uniformly among the accepted preimages of the position
rank of `c`: each rank has exactly `floor(2^m / R)` of them. The signs are not
resampled independently, because the operator-norm predicate depends on
them. A message that runs out is rejected and
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
  adds no soundness term (see the pool section). This now holds for every
  family, not only the operator-norm ones.
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
Decoding is parallel. Every candidate has a fixed bit offset, so above the
existing work threshold the decoder decodes candidates in parallel (Rayon) in
chunks of `2W`, rank-word check and predicate included, and a sequential
prefix pass picks the first `W` accepted. An unfiltered pool fits in one
chunk. The verifier, which usually runs without threads, stops at the
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
- **Protocol version.** Versions move once per stack, counted from `main`.
  `main` is at version 7 and #253 already moves it to 8, so this change stays
  at version 8. The stack's single tag, `akita-pcs/transcript/v8`, lands with
  the positional transcript, whose version doc lists every codec version 8
  fixes.
- **Transcript surface.** `ProofChannel` gains `challenge_message(len,
  decode)`, a squeeze of exactly `len` bytes through Spongefish's
  `verifier_message_as` that poisons the verifier when decoding fails, and
  the fold preview gains the same call and a proof-of-work step. The 32-byte
  `seed` and `fold_root` are deleted. A message longer than Jolt's
  `2^24`-byte squeeze limit is a setup error.

## Decoder choice

Bits per accepted challenge, by method:

| Family | Information `log2 |S|` | Indexed (bytes read) | One stream, bitmask rejection (expected, before pool) | Unranking into a `2^K` prefix (v8 before this revision) | **Full shell: Lemire rank word** (expected, before pool) |
|---|---|---|---|---|---|
| 64, 31, 10 | 128.09 | 762 (one 1088-bit block) | 356 | 128 | **145** |
| 128, 31, 0 | 129.62 | 531 | 278 | 129 | **146** |
| 256, 23, 0 | 131.08 | 376 | 215 | 131 | **148** |
| 512, 19, 0 | 132.76 | 462 | 193 | 132 | **147** |
| 1024, 16, 0 | 131.58 | 386 | 177 | 131 | **144** |
| 2048, 14, 0 | 131.59 | 337 | 169 | 131 | **142** |
| 64, 31, 11 (operator norm) | 130.15 | 781 per try | 364 per try | 131 per try (range and predicate rejection) | **147 per try** (predicate rejection only) |

Ranked recommendation:

1. **Full shell with a Lemire rank word** (this spec). About 12 to 17 bits per
   challenge above the information bound, plus a pool of a few spare
   candidates, and exactly uniform on the whole shell. For the operator-norm
   families it is cheaper than the range-rejection decoder it replaces,
   because range rejection discarded 44% (D64) and 23% (D128) of candidates;
   on the standard profile the operator-norm messages shrink from 16072 to
   7616 bytes, more than the unfiltered messages grow (41408 to 48920). Native
   cost is one 128 x 128 -> 256-bit multiply per candidate (four 64-bit
   multiplies, about 10 ns) on top of the unranking.
2. **Rank-check advice** for a recursion guest. Same transcript bits as
   option 1: the host supplies the decoded positions as untrusted advice, and
   the guest checks them instead of searching. The guest MUST check all of:
   for every candidate up to and including the last accepted one, recompute
   `x * R = hi * 2^m + lo` and compare `lo` with `t`, so the guest itself
   decides which rank words are rejected and derives each `hi` from the
   transcript word; for each accepted candidate, the ±1
   positions are strictly increasing and their colexicographic rank
   `r_a = sum_k binom(p_k, k)` is below `binom(d, a)`; the ±2 positions are
   given as strictly increasing indices into the complement of the ±1 set,
   with rank `r_b` below `binom(d - a, b)`, and are mapped back through that
   complement; and `r_a + binom(d, a) * r_b` equals `hi`. Checking one subset
   alone would let a prover choose the ±2 positions. An operator-norm guest
   also evaluates the predicate on every candidate it accepts and every
   rank-word-accepted candidate it skips, after checking that candidate's
   advised positions against its `hi` in the same way and reading its signs
   from the transcript; unchecked advice must never justify a skip. It is a guest implementation of option 1, not a
   protocol choice. Putting the positions in the proof instead would add about
   `W * w * log2 d` bits (55 KB at fold 0) and is rejected.
3. **Unranking into a `2^K` prefix.** The fewest bits, and no rejection for
   unfiltered families, but the accepted family is the first `2^K` shell
   elements in colexicographic order rather than the shell. It loses 0.08 to
   0.76 bits of set size, loses the shell's symmetry and never sets some top
   positions. It priced no differently, but the full-shell revision replaced
   it ([Full-shell revision](#full-shell-revision)).
4. **Mixed-radix block reduction** (one big integer per block of `c`
   challenges, accept below `floor(2^n / R^c) R^c`). Near-minimal bits, but
   the decode is a multi-limb multiplication or division per block: measured
   at 116 to 920 ns per challenge against about 10 ns for one Lemire multiply,
   to save about 48 Keccak permutations (about 30 us) on the standard
   profile.
5. **Bitmask rejection in one stream.** 1.3 to 2.8 times the bits of option
   3, plus a pool for the variable bit consumption. Its only advantage is a
   simpler decoder.

## Full-shell revision

Before 2026-11-05 an unfiltered family read `K = floor(log2 |S|)` bits per
challenge with no rejection, so its accepted family was the first `2^K`
elements of the shell in this decoder's order. That prefix keeps every
per-element property the fold uses, and the plan priced `2^K` anyway, so it
cost no soundness. It did cost:

- **Set size.** `log2 |S| - K` is 0.09 (D64), 0.62 (D128), 0.08 (D256), 0.76
  (D512), 0.58 (D1024) and 0.59 (D2048) bits. A downstream ledger that counts
  events over the shell must instead use `2^K`, which put the D64 L-infinity
  atoms exactly at `2^-128` rather than strictly below it.
- **Symmetry.** The shell is closed under negacyclic rotation `X^j`, the
  automorphisms `X -> X^k`, and permutations of positions. A colexicographic
  prefix is not: it favours low positions (D128 never sets `X^127` and sets
  `X^126` at 0.12 instead of 0.24; D512 never sets `X^499` and above). Any
  argument or estimate that assumes exchangeable positions or invariance under
  rotation would need a separate treatment for the prefix.
- **Two decoder paths.** Unfiltered and operator-norm families decoded
  differently, and the operator-norm range rejection wasted 23% to 44% of each
  candidate's bits.

The revision gives every family one candidate layout and one decoder loop:
signs, a Lemire rank word, an optional predicate, a fail-closed pool. An
unfiltered family is the case where the predicate always accepts. The
operator-norm accepted distribution is unchanged. On the standard profile the
squeezed message bytes drop from 57480 to 56536, about 7 Keccak permutations
fewer; the fold proof-of-work difficulties are unchanged. The challenges and
message lengths change, so this is a transcript change within the same
protocol version stack.

## Measured effect

These counts were measured before the [Full-shell revision](#full-shell-revision).
That revision lowers the squeezed bytes on this profile by 944, about 7
permutations for the verifier and 14 for the prover (preview and live); it
has not been re-measured.

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
  the indexed worker. Keep the operator-norm predicate. The full-shell
  revision adds the rank-word threshold to the shell, `lemire_rank` and
  `widening_mul`, the unfiltered pool rates, and the `min` in the union bound,
  and merges the two decode paths.
- `akita-transcript`: `ProofChannel::challenge_message`; `FoldPreview`
  squeezes messages and runs Jolt's `grind` on a detached prover state; delete
  `seed`, `fold_root` and `FOLD_CHALLENGE_SEED_LEN`; the session tag names
  the codec.
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

- Rank words: on small rank spaces, exhaustively, every rank has exactly
  `floor(2^m / R)` accepted words and exactly `t` words are rejected; the
  256-bit product is exact.
- Decoder round trip: every accepted candidate decodes to a shell element
  whose signs and colexicographic rank are its signs and Lemire rank; edge
  ranks `0` and `R - 1` decode; the D128 and D512 top positions are reached.
- Messages: candidate widths and pool sizes match the tables; a message of
  zero rank words decodes to nothing for every family and the verifier
  rejects; operator-norm outputs are exactly the predicate-accepted
  candidates in order.
- Prover preview, live prover and verifier draw identical challenges.
- A tampered proof-of-work nonce and a truncated proof are rejected with
  `AkitaError`.
- Plan: one `FoldChallengeWork` per fold with the bits in the table above.
- Known answers and proof pins regenerated under both sponges.
