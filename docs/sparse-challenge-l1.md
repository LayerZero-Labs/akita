# Sparse Challenge L1 Optimization

## Problem

Hachi samples sparse ring challenges for the stage-1 folding step. A challenge is a short ring element

```text
c(X) = sum_i c_i X^i
```

represented by nonzero coefficient positions and small signed integer coefficients.

Throughout this document, norms are coefficient norms:

```text
||c||_1   = sum_i |c_i|
||c||_inf = max_i |c_i|
```

In code, `l1_mass` is a worst-case `L1` bound for a challenge family, and `max_abs_coeff` is its `L_inf` bound.

> Historical note: this document was written when `D=64` used the
> `SplitRing` family. `SplitRing` has since been retired and `D=64` now
> uses `ExactShell { count_mag1: 30, count_mag2: 12 }`, which preserves the
> previous `l1_mass = 54`, `max_hamming_weight = 42`, and
> `max_abs_coeff = 2` while raising support entropy from `≈ 2^128.54` to
> `≈ 2^131.52`. The motivation analysis below is preserved verbatim and
> still uses the `SplitRing` numbers for context.

The current fp128 presets choose challenge families by ring dimension:

```text
D=32:  Uniform, weight=32, coefficients in {-8,...,-1,1,...,8}
D=64:  SplitRing, half_weight=21, max_mag2_per_half=6   (historical; now ExactShell)
D=128: Uniform, weight=31, coefficients in {-1,1}
```

For `D=64`, `max_mag2_per_half=6` means that each parity half may contain up to 6 coefficients with magnitude 2. It is not an `L_inf` bound of 6; the `D=64` challenge still has `||c||_inf <= 2`.

These choices provide at least 128 bits of support-size entropy for Fiat-Shamir challenges. That is, each challenge family has at least `2^128` possible outputs under its configured sampler.

Their worst-case `L1` masses are:

```text
D=32:  32 * 8 = 256
D=64:  2 * (21 + 6) = 54
D=128: 31 * 1 = 31
```

The question is whether we can keep at least 128 bits of challenge support, keep the same `L_inf` bound, and reduce the worst-case `L1` mass.

The important constraints are:

```text
1. Do not increase coefficient L_inf:
   ||c||_inf must stay <= the current value.

2. Minimize worst-case coefficient L1:
   ||c||_1 should be as small as possible.

3. Keep at least 128 bits of Fiat-Shamir challenge support:
   number of possible challenges must be >= 2^128.
```

## Motivation: Why L1 Mass Is Important

The sparse challenge is not just checked coefficient-by-coefficient. It is used to fold witness data in the coefficient embedding of the negacyclic ring. When a sparse challenge multiplies or linearly combines witness coefficients, many shifted witness terms can add into one output coefficient, possibly with signs from the ring wraparound.

If the witness coefficients satisfy

```text
||s||_inf <= B
```

and the challenge is

```text
c(X) = c_1 X^{i_1} + ... + c_t X^{i_t},
```

then each output coefficient of `c * s` is a signed sum of shifted witness coefficients and is bounded by the triangle inequality:

```text
|(c * s)_k| <= (|c_1| + ... + |c_t|) * B
```

So:

```text
||c * s||_inf <= ||c||_1 * ||s||_inf
```

This is why Hachi tracks `l1_mass`, not only `max_abs_coeff`.

`max_abs_coeff` still matters, but it controls a different part of the system: SIS collision sizing for some matrix roles. `l1_mass` controls the folded-witness `L_inf` bound and therefore the number of balanced digits needed for the folded witness.

The planner uses the bound:

```text
beta = challenge_l1_mass * num_claims * 2^(r_vars + log_basis - 1)
```

and then chooses enough decomposition digits for that bound. Smaller `l1_mass` can reduce `delta_fold`, shrink `z_pre`, reduce recursive witness size, and reduce proof size.

Therefore the safe optimization target is narrow: reduce the worst-case challenge `L1` mass, while preserving the current `L_inf` bound and preserving at least `2^128` possible Fiat-Shamir challenges. The later construction must use the true worst-case `L1` mass of the implemented sampler; using an average or typical `L1` would not be a safe replacement for `l1_mass`.

## Solution

Use a bounded-`L1` challenge family.

Instead of choosing a fixed number of nonzero positions and allowing the worst-case sum of magnitudes, sample uniformly from all coefficient vectors whose coefficient `L1` norm is at most a configured bound, while keeping the same coefficient `L_inf` bound.

Conceptually:

```text
BoundedL1Ball {
    max_abs_coeff: M,  // coefficient L_inf bound
    l1_bound: B,       // coefficient L1 bound
}
```

This means sample from:

```text
{ c in Z^D : ||c||_inf <= M and ||c||_1 <= B }
```

Then store only nonzero coefficients in `SparseChallenge`.

This is the most direct way to optimize the stated goal. For fixed `D`, fixed coefficient `L_inf` limit `M`, and fixed coefficient `L1` limit `B`, the bounded-`L1` ball contains every challenge satisfying those two norm constraints. Therefore, the smallest `B` whose ball has at least `2^128` elements is the best possible worst-case `l1_mass` under the same `L_inf` limit and support-size target.

### How The Entropy Numbers Are Computed

For a fixed `D`, coefficient `L_inf` bound `M`, and coefficient `L1` bound `B`, count:

```text
N(D, M, B) =
  # { (c_0,...,c_{D-1}) : ||c||_inf <= M and ||c||_1 <= B }
```

If the sampler is exactly uniform over this set, the support-size entropy and min-entropy are:

```text
log2(N(D, M, B))
```

Each coefficient contributes:

```text
magnitude 0: one choice, coefficient 0
magnitude a > 0: two choices, +a or -a
```

A dynamic program can compute the count:

```text
count(0, b) = 1

count(n, b) =
    count(n-1, b)
  + sum_{a=1..min(M,b)} 2 * count(n-1, b-a)
```

Here `count(n, b)` counts length-`n` vectors with coefficient `L1` norm at most `b`, for the fixed `M`.

Then search for the smallest `B` such that:

```text
log2(count(D, B)) >= 128
```

### Proposed fp128 Parameters

Keep the current coefficient `L_inf` bound for each ring dimension:

```text
D=32:  M=8
D=64:  M=2
D=128: M=1
```

The smallest coefficient `L1` bounds that reach 128-bit support-size entropy are:

```text
D=32, M=8:
  B=120 gives about 127.972 bits
  B=121 gives about 128.133 bits
  proposed l1_mass = 121

D=64, M=2:
  B=46 gives about 126.956 bits
  B=47 gives about 128.171 bits
  proposed l1_mass = 47

D=128, M=1:
  B=30 gives about 127.196 bits
  B=31 gives about 129.868 bits
  proposed l1_mass = 31
```

So the proposed challenge policy is:

```text
D=32:  BoundedL1Ball { max_abs_coeff: 8, l1_bound: 121 }
D=64:  BoundedL1Ball { max_abs_coeff: 2, l1_bound: 47 }
D=128: BoundedL1Ball { max_abs_coeff: 1, l1_bound: 31 }
```

### Security Invariants

The security-relevant invariants are:

```text
1. The sampler is exactly uniform over its configured bounded-L1 ball.
2. The ball has at least 2^128 elements.
3. The configured l1_mass is the true worst-case L1 bound B.
4. The configured max_abs_coeff is the true L_inf bound M and does not increase.
5. The new challenge family has its own transcript/domain-separation encoding.
```

Under these invariants, the Fiat-Shamir challenge keeps at least 128 bits of min-entropy, the folded-witness `L_inf` analysis uses a valid worst-case `L1` bound, and the SIS sizing that depends on `max_abs_coeff` does not get worse. The change is therefore not relying on average-case challenge mass or on typical samples; it is still a worst-case norm bound.

## Expected Results

Compared to the current fp128 challenge families:

```text
D=32:
  current l1_mass:  256
  proposed l1_mass: 121
  improvement:      135 lower
  L_inf bound:      unchanged at 8
  support entropy:  about 128.133 bits

D=64:
  current l1_mass:  54
  proposed l1_mass: 47
  improvement:      7 lower
  L_inf bound:      unchanged at 2
  support entropy:  about 128.171 bits

D=128:
  current l1_mass:  31
  proposed l1_mass: 31
  improvement:      no L1 change
  L_inf bound:      unchanged at 1
  support entropy:  about 129.868 bits
```

The biggest expected benefit is for `D=32`, where current uniform sampling permits all 32 coefficients to have magnitude 8, giving a worst-case `L1` mass of 256. A bounded-`L1` sampler removes those high-`L1` challenges while preserving enough Fiat-Shamir challenge support.

The expected protocol effect is:

```text
smaller l1_mass
  -> smaller folded-witness bound beta
  -> fewer fold decomposition digits in some layouts
  -> smaller recursive witness and possibly smaller proof size
```

This is a proof-size and bound-size argument, not a claim that every prover-time cost automatically improves. Sparse arithmetic can also depend on Hamming weight and on fast paths for particular challenge shapes. For example, the proposed `D=64` ball has worst-case Hamming weight up to `47`, while the current split-ring sampler always has Hamming weight `42`. That is not a security issue, but it should be benchmarked separately from the `L1`/`L_inf` security argument.

This change would require a new sampler, not only new constants. The sampler must draw uniformly from the bounded-`L1` ball so that the entropy calculation matches the implemented distribution.

## Engineering: Efficient Production Sampling

The bounded-`L1` sampler is more complicated than the current fixed-weight samplers, but it can still be efficient in production. The key is that the supported parameter sets are fixed:

```text
(D=32,  M=8, B=121)
(D=64,  M=2, B=47)
(D=128, M=1, B=31)
```

So the counting tables can be computed once, stored as static tables, or initialized lazily and cached. Table construction should not be on the hot path.

### Current Sampler Cost

The current samplers are very lightweight:

```text
Uniform:
  sample fixed number of distinct positions
  sample each nonzero coefficient from a small alphabet

SplitRing:
  sample fixed number of even positions
  sample fixed number of odd positions
  sample signs
  sample how many magnitude-2 entries each half has
```

These paths use small integer rejection sampling and simple shuffles. They are fast because the output shape is fixed: the sampler knows the Hamming weight before it starts.

The bounded-`L1` sampler has to choose from a larger structured set whose Hamming weight and magnitudes are variable. It should not try rejection sampling over all dense vectors in `[-M, M]^D`; for `D=32, M=8`, almost all dense vectors violate `||c||_1 <= 121`, so naive rejection would waste randomness and time.

### Recommended Production Algorithm

Use combinatorial unranking with cached suffix-count tables.

For each supported `(D, M, B)`, precompute:

```text
ways[n][b] =
  number of length-n suffixes with coefficients in [-M, M]
  and coefficient L1 norm at most b
```

The recurrence is:

```text
ways[0][b] = 1

ways[n][b] =
    ways[n-1][b]
  + sum_{a=1..min(M,b)} 2 * ways[n-1][b-a]
```

Then `ways[D][B]` is the total support size. Sampling one challenge is:

```text
1. Draw r uniformly from [0, ways[D][B]).
2. remaining_budget = B.
3. For i = 0..D-1:
     remaining_positions = D - i - 1
     scan candidate x in a fixed order, for example -M..M
     block_size(x) = ways[remaining_positions][remaining_budget - |x|]
     choose the block containing r
     emit x if x != 0
     r = offset inside that block
     remaining_budget -= |x|
```

This is exact uniform sampling because each coefficient choice gets a block whose size is exactly the number of valid suffix completions. It is also cheap: with cached tables, each challenge requires only `D * (2M + 1)` small block checks.

For the proposed fp128 settings, that is:

```text
D=32,  M=8: at most 32  * 17 = 544 block checks
D=64,  M=2: at most 64  * 5  = 320 block checks
D=128, M=1: at most 128 * 3  = 384 block checks
```

These operations are fixed-width integer comparisons and subtractions, not field arithmetic, NTTs, ring multiplications, or transcript hashes.

### Integer Representation

The table entries are slightly larger than `2^128`:

```text
D=32:  log2(ways[D][B]) ~= 128.133
D=64:  log2(ways[D][B]) ~= 128.171
D=128: log2(ways[D][B]) ~= 129.868
```

So `u128` is not enough. A production implementation should use a fixed-width integer type large enough for all counts, for example a 192-bit or 256-bit unsigned integer. A fixed-width type is preferable to heap-allocated big integers on the hot path.

All table entries are bounded by `ways[D][B]`, so the same integer type can be used for:

```text
ways[n][b]
r
block_size
cumulative block offsets
```

Using 256-bit integers is simple and leaves ample margin. The tables are tiny:

```text
D=32:  (33 * 122) entries
D=64:  (65 * 48) entries
D=128: (129 * 32) entries
```

Even with 256-bit entries, each table is small enough to keep in memory permanently.

### Drawing The Uniform Index

The sampler needs a uniform integer in:

```text
[0, ways[D][B])
```

Use bitmask rejection sampling from the transcript-derived XOF:

```text
k = ceil(log2(ways[D][B]))
draw k random bits
if value < ways[D][B], accept
otherwise draw again
```

The acceptance probabilities are good:

```text
D=32:  about 2^128.133 / 2^129 ~= 55%
D=64:  about 2^128.171 / 2^129 ~= 56%
D=128: about 2^129.868 / 2^130 ~= 91%
```

So the expected number of draws is about `1.8`, `1.8`, and `1.1`, respectively. This is not a practical bottleneck, especially because the current sparse challenge path already derives one SHAKE256 stream and consumes XOF bytes incrementally.

The implementation must not reduce a fixed-width random integer modulo `ways[D][B]`; that would bias the distribution unless the modulus divides the integer range exactly. It also must not draw only 128 bits, because all three proposed supports are larger than `2^128`.

### Batch Sampling

The current code already amortizes transcript cost by deriving one PRG seed and expanding a SHAKE256 XOF stream for all challenges in a batch. The bounded-`L1` sampler should keep the same shape:

```text
absorb challenge context once
derive one seed from the transcript
expand SHAKE256 as a stream
decode each bounded-L1 challenge from the stream
```

This means the extra cost is the unranking work per challenge, not one transcript hash per challenge. For large batches, this is important.

The output should still be the existing sparse representation:

```text
SparseChallenge {
    positions: all nonzero positions,
    coeffs:    corresponding nonzero coefficients,
}
```

The dense vector is only a conceptual sampling object. The implementation does not need to allocate a dense `[i16; D]` unless doing so simplifies the code; it can emit nonzero entries directly while walking the coefficients.

### Efficiency Compared To Current Sampling

The bounded-`L1` sampler will be slower than the current fixed-shape samplers if measured in isolation. The current samplers mostly do small shuffles, sign draws, and small-modulus rejection. The bounded-`L1` sampler adds:

```text
one 129- or 130-bit rejection draw
D * (2M + 1) fixed-width count comparisons
variable-length sparse output construction
```

However, the absolute cost should still be small for `D <= 128`. The expensive parts of the protocol are elsewhere: witness folding, decomposition, commitments, ring operations, and sumcheck work. The bounded-`L1` sampler is attractive if the smaller `l1_mass` reduces `delta_fold` or recursive witness size enough to offset the extra integer work.

The expected production outcome is:

```text
single challenge microbenchmark:
  bounded-L1 likely slower than current sampling

full prover/proof-size benchmark:
  bounded-L1 can win if lower l1_mass reduces fold digits or recursive payload

verifier:
  mostly unaffected, except through smaller proof objects and any changed transcript payloads
```

So this should be evaluated at the protocol level, not only as a sampler microbenchmark.

### Engineering Requirements

A production implementation should satisfy these requirements:

```text
1. Use cached/static tables for each supported (D, M, B).
2. Use exact fixed-width integer arithmetic for all counts and indices.
3. Use rejection sampling, not modulo reduction, to draw r.
4. Use a canonical coefficient order so prover and verifier derive identical challenges.
5. Domain-separate the new BoundedL1Ball config, including D, M, and B.
6. Store only nonzero coefficients in SparseChallenge.
7. Treat config hamming_weight as variable or maximum for this family, not exact.
8. Test that every sampled challenge satisfies ||c||_inf <= M and ||c||_1 <= B.
9. Test deterministic transcript replay.
10. Test the small toy sampler exhaustively against exact uniform counts.
```

The most important engineering point is that the optimized sampler must preserve the exact distribution analyzed above. If the sampler is exact and the tables are cached, the production overhead is controlled and predictable.

## Concrete D=32 Sampling Procedure

For `D=32`, keep the current coefficient `L_inf` limit:

```text
max_abs_coeff = M = 8
```

Replace the current full-weight uniform challenge with:

```text
BoundedL1Ball {
    max_abs_coeff: 8,
    l1_bound: 121,
}
```

This samples a vector

```text
c = (c_0, ..., c_31)
```

from the set

```text
S_32 = {
    c in {-8, -7, ..., 0, ..., 7, 8}^32
    such that ||c||_1 <= 121
}
```

After sampling, the sparse representation stores only the nonzero entries:

```text
positions = { i : c_i != 0 }
coeffs    = { c_i : c_i != 0 }
```

So zero coefficients are allowed in the conceptual dense vector, but they are not stored in `SparseChallenge`.

### Counting The Challenge Set

Define `exact[n][s]` as the number of length-`n` vectors with coefficients in `[-8,8]` and exact coefficient `L1` norm `s`.

Base case:

```text
exact[0][0] = 1
exact[0][s > 0] = 0
```

Recurrence:

```text
exact[n][s] =
    exact[n-1][s]                                  // choose coefficient 0
  + 2 * exact[n-1][s-1]                            // choose coefficient +1 or -1
  + 2 * exact[n-1][s-2]                            // choose coefficient +2 or -2
  + ...
  + 2 * exact[n-1][s-8]                            // choose coefficient +8 or -8
```

Terms with negative indices are omitted.

The number of valid `D=32` challenges with coefficient `L1` norm at most `121` is:

```text
T_32 = sum_{s=0}^{121} exact[32][s]
```

The support-size entropy is:

```text
log2(T_32) ~= 128.133 bits
```

The previous bound is just below the target:

```text
sum_{s=0}^{120} exact[32][s]
log2(...) ~= 127.972 bits
```

Therefore `121` is the smallest `L1` bound that reaches 128-bit support-size entropy under the `max_abs_coeff <= 8` constraint.

### Uniform Sampling Algorithm

To sample uniformly from `S_32`, use the DP table above.

1. Build the `exact[n][s]` table for:

```text
n = 0..32
s = 0..121
```

2. Compute:

```text
T_32 = sum_{s=0}^{121} exact[32][s]
```

3. Draw a uniform integer:

```text
r in [0, T_32)
```

from the transcript-derived XOF stream.

This draw must be exact. Since `T_32` is slightly larger than `2^128`, the implementation cannot sample a fixed 128-bit value or reduce a fixed-width value modulo `T_32`. It should use rejection sampling with enough XOF bits, or an equivalent combinatorial sampler, so every index in `[0, T_32)` has the same probability.

4. Walk the 32 coefficient positions from left to right in a fixed canonical order. At position `i`, suppose:

```text
remaining_positions = 31 - i
remaining_L1_budget = b
```

For each candidate coefficient

```text
x in {-8, -7, ..., 0, ..., 7, 8}
```

with `|x| <= b`, the number of completions is:

```text
block_size(x) = sum_{s=0}^{b - |x|} exact[remaining_positions][s]
```

Use `r` to select one of these blocks, set `c_i = x`, replace `r` with its offset inside the selected block, subtract `|x|` from the remaining `L1` budget, and continue.

5. At the end, emit:

```text
SparseChallenge {
    positions: all i where c_i != 0,
    coeffs:    the corresponding c_i values,
}
```

This produces an exactly uniform sample from the bounded-`L1` ball, assuming the integer `r` is sampled uniformly from `[0, T_32)`.

### Small Decoding Example

Use a tiny toy parameter set:

```text
D = 3
M = 2
B = 3
```

So coefficients are in:

```text
{-2, -1, 0, 1, 2}
```

and the total `L1` norm must be at most `3`.

For this toy set, the total number of valid vectors is:

```text
ways[3][3] = 57
```

Here `ways[n][b]` means the number of length-`n` vectors with `L1` norm at most `b`.

So draw:

```text
r in [0, 57)
```

Use the fixed coefficient order:

```text
-2, -1, 0, 1, 2
```

#### Choose `c_0`

If `c_0 = x`, the remaining `L1` budget is `3 - |x|`, and there are two coefficients left.

The block sizes are:

```text
c_0 = -2: ways[2][1] = 5
c_0 = -1: ways[2][2] = 13
c_0 =  0: ways[2][3] = 21
c_0 =  1: ways[2][2] = 13
c_0 =  2: ways[2][1] = 5
```

These blocks partition the global index range:

```text
r =  0..4   => c_0 = -2
r =  5..17  => c_0 = -1
r = 18..38  => c_0 =  0
r = 39..51  => c_0 =  1
r = 52..56  => c_0 =  2
```

Suppose the sampled index is:

```text
r = 25
```

Since `25` lies in `18..38`, choose:

```text
c_0 = 0
```

The offset inside the `c_0 = 0` block is:

```text
offset = 25 - 18 = 7
```

The remaining `L1` budget is still `3`.

#### Choose `c_1`

Now choose `c_1` using `offset = 7`.

If `c_1 = x`, one coefficient remains. The block sizes are:

```text
c_1 = -2: ways[1][1] = 3
c_1 = -1: ways[1][2] = 5
c_1 =  0: ways[1][3] = 5
c_1 =  1: ways[1][2] = 5
c_1 =  2: ways[1][1] = 3
```

So the offset ranges are:

```text
offset =  0..2   => c_1 = -2
offset =  3..7   => c_1 = -1
offset =  8..12  => c_1 =  0
offset = 13..17  => c_1 =  1
offset = 18..20  => c_1 =  2
```

Since `offset = 7`, choose:

```text
c_1 = -1
```

The new offset inside the `c_1 = -1` block is:

```text
new_offset = 7 - 3 = 4
```

The remaining `L1` budget is:

```text
3 - |c_0| - |c_1| = 3 - 0 - 1 = 2
```

#### Choose `c_2`

One coefficient remains, with `L1` budget `2`.

The allowed values are:

```text
-2, -1, 0, 1, 2
```

Each has exactly one completion because there are no coefficients left afterward:

```text
new_offset = 0 => c_2 = -2
new_offset = 1 => c_2 = -1
new_offset = 2 => c_2 =  0
new_offset = 3 => c_2 =  1
new_offset = 4 => c_2 =  2
```

Since `new_offset = 4`, choose:

```text
c_2 = 2
```

The final dense vector is:

```text
[0, -1, 2]
```

Its `L1` norm is:

```text
|0| + |-1| + |2| = 3
```

The sparse representation stores only nonzero entries:

```text
SparseChallenge {
    positions: [1, 2],
    coeffs:    [-1, 2],
}
```

This illustrates the general rule: a single random index selects a block for `c_0`, then an offset inside that block selects a block for `c_1`, and so on until all coefficients are determined.

### Why This Achieves The Goal

Every sampled challenge satisfies:

```text
||c||_inf <= 8
||c||_1 <= 121
```

So:

```text
max_abs_coeff does not increase
l1_mass becomes 121
```

And because:

```text
log2(|S_32|) = log2(T_32) ~= 128.133
```

the challenge set has more than `2^128` possible challenges.

Compared to the current `D=32` uniform challenge:

```text
current:
  coefficients: every position is nonzero in {-8,...,-1,1,...,8}
  support:      16^32 = 2^128 elements
  l1_mass:      32 * 8 = 256

bounded-L1 proposal:
  coefficients: values in {-8,...,0,...,8}, L1 <= 121
  support:      about 2^128.133 elements
  l1_mass:      121
```

The bounded-`L1` sampler gets the same security target with much smaller worst-case folded-witness growth.
