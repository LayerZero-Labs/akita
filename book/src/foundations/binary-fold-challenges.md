# Binary fold challenge families

`akita-challenges` provides exact challenge-family and sampling primitives for
binary folds. The primitives define two scalar rings, exact support families,
one deterministic sign rule, canonical identity bytes, and unbiased samplers.
They do not select a schedule or change Akita's proof protocol.

This distinction matters when reviewing the current code. The crate can count
and sample a binary challenge today. No prover, verifier, planner catalog, or
proof encoding consumes that challenge yet.

## Why the support determines the entropy

In characteristic two, `1` and `-1` have the same residue. Giving every
nonzero coefficient an independent sign would therefore create several
integer challenges with the same binary image. Those signs cannot count as
distinct challenges in an extraction argument that requires distinct binary
residues.

The binary families instead assign exactly one signed ternary lift to each
support. A support is the set of coefficient positions where a challenge is
nonzero. If two supports differ, their reductions modulo two differ. The sign
map can improve the shape of integer responses, but it contributes no entropy.

For a small example, a four-coordinate support ball with cap one contains

```text
{}, {0}, {1}, {2}, {3}.
```

Its cardinality is five, including the empty support. Choosing a weight
uniformly from zero and one would give the empty support probability `1/2`,
which is not uniform over the five supports. The sampler instead gives the
weight-zero shell probability `1/5` and the weight-one shell probability `4/5`,
then samples uniformly inside the selected shell.

## Supported scalar rings and families

`BinaryScalarRing` identifies the two supported power-basis rings.

| Variant | Integer ring | Binary residue | Degree |
| --- | --- | --- | ---: |
| `Cyclotomic243` | `Z[x]/(x^162 + x^81 + 1)` | `F_(2^162)` | 162 |
| `Cyclotomic729` | `Z[z]/(z^486 + z^243 + 1)` | `F_(2^486)` | 486 |

`BinaryChallengeFamily` then selects one of two support laws for degree `d`.

| Family | Admitted supports | Exact cardinality |
| --- | --- | --- |
| `FixedWeight` | supports with size exactly `w` | `binom(d, w)` |
| `BoundedWeight` | supports with size at most `w` | `sum_(j=0)^w binom(d, j)` |

The fixed-weight family is the reference shell. The bounded-weight family uses
every support under the same coefficient L1 cap and is therefore larger unless
the cap is zero.

`BinaryChallengeProfile::minimum_for_budget` selects the smallest shell weight
or ball cap satisfying

```text
cardinality >= fold_width * 2^lambda_fold.
```

The comparison uses exact integers. It does not use a floating-point entropy
estimate. Fixed-weight search stops at `floor(d/2)`, where the binomial shell is
largest. Bounded-weight search can reach the full ball of size `2^d`. A zero
fold width is invalid. A request beyond the residue capacity returns
`UnsupportedSchedule` before constructing an oversized shifted integer.

The cached cumulative binomial rows and public cardinalities use `BigUint`.
This is necessary for degree 486: for example, `binom(486, 243)` has a 482-bit
binary representation. The same type keeps the API exact at both supported
degrees.
Counts above 256 bits are combinatorial facts, not a claim that the current
32-byte transcript root supplies more than 256 bits of challenge entropy. The
current target is a 128-bit fold budget, and this branch wires no profile into
a protocol path.

## Profile identity and deterministic signs

`BinaryChallengeProfile::identity_bytes` is the canonical profile identity. It
binds all facts needed to interpret a sampled support:

- encoding version;
- scalar ring and degree;
- fixed-weight or bounded-weight family;
- sign-rule version;
- little-endian support-bitset encoding;
- weight or cap;
- certified multiplication bound; and
- exact cardinality, with its encoded length.

The only current `BinarySignRule` is `Shake256V1`. It hashes the domain
`akita/labinius/binary-sign/v1`, the complete profile identity, and the
canonical support bitset. One output byte is assigned to each support position;
its low bit selects `1` or `-1`. The sampler does not draw separate sign bits
from the transcript, and callers cannot supply a sign nonce.

`BinaryChallenge` stores terms in increasing position order. Its canonical
support encoding contains `ceil(d/8)` bytes, with coefficient position `i` in
bit `i % 8` of byte `i / 8`. Validation checks the profile weight rule, position
range, strict ordering, coefficients in `{-1, 1}`, and exact sign replay.

## Sampling without bias

`BinaryChallengeSampler` owns reusable scratch for its profile. A transcript
draw absorbs the binary sampler domain, a length-prefixed caller label, the
challenge count, and the complete profile identity. It then squeezes one
32-byte root through the existing Akita transcript interface. The current
support-mapping domain is `akita/labinius/binary-challenge/v2`.

Challenge coordinate `i` expands the existing indexed SHAKE256 stream

```text
root || little_endian_u64(i).
```

The two families map that stream differently:

- Fixed weight uses the existing unbiased partial Fisher--Yates sampler, then
  sorts the chosen positions into canonical order.
- Bounded weight masks enough XOF bits to cover the exact ball cardinality and
  rejects integers outside that interval. It locates the accepted integer's
  weight shell in the cached cumulative row, discards the within-shell offset,
  and samples a support of that weight with the same Fisher--Yates primitive.

For cap `w`, write

```text
A_j = sum_(t=0)^j binom(d, t),   N = A_w.
```

An accepted integer `U` selects the unique weight `j` satisfying
`A_(j-1) <= U < A_j`. Therefore the shell probability is
`binom(d, j) / N`. Conditional on that shell, Fisher--Yates gives every
`j`-support probability `1 / binom(d, j)`. Every support in the ball therefore
has probability exactly `1 / N`. Weight zero selects the empty support.

The Fisher--Yates draw consumes previously unused bytes from the same indexed
XOF stream. In the XOF model, this suffix remains independent of the accepted
prefix even when integer rejection consumes a variable number of attempts.
Masked high bits and rejected candidates are discarded rather than reused.

Integer rejection has no attempt cap and therefore adds no sampler-failure
probability. Its expected attempt count is below two. Selecting a shell uses a
binary search over one cached cumulative row instead of a degree-long sequence
of big-integer comparisons. The sampler reuses its XOF cursor,
partial-permutation state, rank limbs, support storage, and sign buffers across
calls. Returned challenges own their terms.

## Coefficient and multiplication bounds

Every nonzero coefficient has absolute value one. A profile with cap `w` has
the following worst-case bounds:

| API | Bound |
| --- | ---: |
| `coefficient_linf_bound()` | `0` when `w=0`, otherwise `1` |
| `coefficient_l1_bound()` | `w` |
| `coefficient_l2_squared_bound()` | `w` |
| `multiplication_linf_operator_bound()` | `2w` |

The last factor of two comes from the trinomial relation. In degree 162,
multiplication by `x^81` maps a coefficient pair `[a,b]` to `[-b,a-b]`.
The degree-486 ring has the same half-degree form. Packing several scalar-ring
components does not multiply this per-component bound.

The profile owns this bound together with its exact cardinality. Security and
SIS code can consume the methods directly instead of maintaining a second
challenge-family table.

## Initialization and hot-path cost

The first profile for a scalar ring initializes one process-wide cumulative
binomial row through `LazyLock`. Later profiles and samplers reuse it. The cold
path now performs `d` exact binomial-recurrence steps and retains `d + 1`
`BigUint` values: 163 entries for degree 162 or 487 for degree 486. It does not
construct the former quadratic Pascal triangle. Keep this one-time
initialization outside hot sampling benchmarks.

The `sparse_challenge` Criterion benchmark constructs profiles before timing
and compares fixed and bounded samplers with the existing ordinary sparse
sampler for context. A 20-sample development run on the same host measured
batch size 4096 as follows:

| Sampler | Batch time | Throughput |
| --- | ---: | ---: |
| degree 162 fixed weight 47 | 3.956--3.985 ms | about 1.03 million/s |
| degree 162 bounded cap 46 | 4.600--4.967 ms | about 0.86 million/s |
| degree 162 bounded cap 47 | 5.125--6.032 ms | about 0.74 million/s |
| degree 486 fixed weight 25 | 3.385--3.524 ms | about 1.18 million/s |
| degree 486 bounded cap 25 | 3.764--4.073 ms | about 1.04 million/s |
| existing ordinary signed-sparse D64 | 1.606--1.734 ms | about 2.47 million/s |

The ordinary D64 line is a runtime baseline, not an equal-security-family
comparison. Reproduce these measurements with

```bash
cargo bench -p akita-challenges --bench sparse_challenge -- \
  labinius_binary_challenge_batch
```

## Current scope and audit paths

This implementation is a challenge foundation only. It does not:

- select binary profiles in the planner or schedule catalog;
- add binary challenges to an instance descriptor or proof encoding;
- connect them to prover or verifier replay;
- derive response intervals or SIS parameters; or
- replace Akita's ordinary sparse challenge families.

The public API is exported from `crates/akita-challenges/src/lib.rs`. Exact
counts and shell selection live in
`crates/akita-challenges/src/binary/combinatorics.rs`; profile identities and
bounds live in `crates/akita-challenges/src/binary/profile.rs`; sampling lives
in `crates/akita-challenges/src/binary/sampler.rs`; and the deterministic sign
map lives in `crates/akita-challenges/src/binary/sign.rs`.

The focused tests exhaustively enumerate small support balls, check the exact
degree-162 and degree-486 budget boundaries, exercise counts above 256 bits,
compare rejection mapping with an independent SHAKE implementation, pin the
canonical identity and sign fixture, include the zero support, and check the
trinomial wraparound behind the factor-two operator bound.
