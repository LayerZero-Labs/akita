# Spec: Bounded-L1 Sparse Challenge

| Field     | Value                            |
|-----------|----------------------------------|
| Author(s) | Omid Bodaghi, Cursor agent draft |
| Created   | 2026-05-04                       |
| Status    | implemented                      |
| PR        | #66                              |

## Summary

This spec documents the implemented `SparseChallengeConfig::BoundedL1Norm`
family for stage-1 sparse fold challenges. The first rollout is a fixed
production preset for fp128 `D=32` only:

```text
D = 32
M = 8
B = 121
```

The sampler draws a 128-bit rank from a transcript-derived SHAKE256 XOF and
unranks it into a sparse vector in:

```text
{ c in Z^32 : ||c||_inf <= 8 and ||c||_1 <= 121 }
```

The realized distribution is uniform over a fixed `2^128`-element subset of
that bounded ball, not over the full ball. This keeps exactly 128 bits of
Fiat-Shamir min-entropy, keeps the existing `L_inf` bound used by SIS sizing,
and reduces the worst-case stage-1 challenge `L1` mass for fp128 `D=32` from
`256` to `121`.

The challenge configuration and type live in `akita-challenges`:

```text
crates/akita-challenges/src/config.rs
crates/akita-challenges/src/challenge.rs
crates/akita-challenges/src/sampler/bounded_l1.rs
```

`LevelParams.stage1_config` carries the selected config through the protocol,
and prover/verifier replay both call `sample_sparse_challenges`.

## Motivation

The stage-1 sparse challenge folds witness data in the coefficient embedding of
the negacyclic ring. If the witness coefficients satisfy:

```text
||s||_inf <= A
```

and the challenge is:

```text
c(X) = c_1 X^{i_1} + ... + c_t X^{i_t}
```

then every output coefficient of `c * s` is bounded by the triangle inequality:

```text
||(c * s)||_inf <= ||c||_1 * ||s||_inf
```

This is why the protocol tracks the challenge `L1` mass, not only the maximum
absolute coefficient. The planner uses this mass in the folded-witness bound:

```text
beta = challenge_l1_mass * num_claims * 2^(r_vars + log_basis - 1)
```

A smaller worst-case `L1` mass can reduce the number of fold decomposition
digits, shrink recursive witnesses, and reduce proof size. The optimization is
safe only because the configured `l1_norm()` remains a true worst-case bound;
the analysis does not rely on average or typical challenge mass.

## Current fp128 Policy

The current fp128 default stage-1 policy used by
`akita-config::proof_optimized::fp128_stage1_challenge_config` is:

```text
D=32:
  SparseChallengeConfig::BoundedL1Norm
  l1_norm()       = 121
  infinity_norm() = 8

D=64:
  SparseChallengeConfig::ExactShell {
      count_mag1: 18,
      count_mag2: 0,
  }
  l1_norm()       = 18
  infinity_norm() = 1

D=128:
  SparseChallengeConfig::Uniform {
      weight: 13,
      nonzero_coeffs: [-1, 1],
  }
  l1_norm()       = 13
  infinity_norm() = 1
```

`BoundedL1Norm` still validates only for `D=32`. The May 2026 tensor planner
rerun reduced the `D=64` and `D=128` defaults, but `D=32` remains on
`BoundedL1Norm`: the low-mass exact-shell candidates did not pass the current
audited schedule checks for the shipped `D=32` ranges.

## Invariants

- `BoundedL1Norm` samples from the fixed `(D=32, M=8, B=121)` production preset.
- Every sampled challenge satisfies `||c||_inf <= 8` and `||c||_1 <= 121`.
- The sampler reads exactly one 16-byte little-endian `u128` rank per challenge.
- The rank is not reduced modulo the full ball size and is not rejection-sampled.
- The realized distribution is uniform over exactly `2^128` retained descent paths.
- Outcomes outside the retained subset have probability `0`.
- `SparseChallenge` stores only nonzero terms as `positions: Vec<u32>` and `coeffs: Vec<i8>`.
- Prover and verifier derive identical challenges from the same transcript prefix, label, batch count, ring degree, and config-domain bytes.
- The DP table and descent are deterministic, platform-independent integer computations with no floating point, randomized initialization, or lazy shared mutation.

## Domain Separation

`SparseChallengeConfig::domain_separator_bytes()` uses the following variant
tags:

```text
Uniform       tag = 0
ExactShell    tag = 1
BoundedL1Norm tag = 2
```

For `BoundedL1Norm`, the config-domain bytes are:

```text
02
08 00 00 00 00 00 00 00
79 00 00 00 00 00 00 00
```

That is:

```text
tag: 2u8
M:   8 as u64 little-endian
B:   121 as u64 little-endian
```

The sparse-challenge absorb buffer is:

```text
label
batch_count as u64 little-endian
D as u64 little-endian
cfg.domain_separator_bytes()
```

The transcript flow is:

```text
transcript.append_bytes(ABSORB_SPARSE_CHALLENGE, absorb_buf)
seed = transcript.challenge_bytes(CHALLENGE_SPARSE_CHALLENGE, 32)
cursor = SHAKE256("akita/sparse-challenge-prg" || seed)
```

## Counting Table

For fixed `D = 32`, `M = 8`, and `B = 121`, define:

```text
WAYS[n][b] =
  # { v in [-8, 8]^n : ||v||_1 <= b }
```

Base case:

```text
WAYS[0][b] = 1 for all b >= 0
```

Recurrence:

```text
WAYS[n][b] =
    WAYS[n-1][b]
  + sum_{a=1..min(8,b)} 2 * WAYS[n-1][b-a]
```

The current implementation stores rows `n = 1..=31` in a compile-time
`[[u128; 122]; 31]` suffix table. The full top cell `WAYS[32][121]` is not
stored because it is larger than `u128`; it is reconstructed in tests to prove
that the full ball exceeds `2^128`.

Parameter facts:

```text
WAYS[32][121] ~= 2^128.133
WAYS[32][120] ~= 2^127.972
```

So `B = 121` is the smallest `L1` bound that reaches the 128-bit support target
while keeping `M = 8`.

## Canonical Rank-Unranking Sampler

The canonical sampler draws one top-level rank:

```text
r = read_u128_le(XOF)
```

Then it visits coefficient positions from left to right and chooses the next
coefficient by descending the DP table.

```text
budget = 121
positions = []
coeffs = []

for i in 0..32:
    if budget == 0:
        break

    remaining_coords = 32 - i - 1
    a = find_bucket(remaining_coords, budget, r)

    if a != 0:
        positions.push(i)
        coeffs.push(a)
        budget -= |a|

return SparseChallenge { positions, coeffs }
```

At each step, candidate coefficients are visited in magnitude-first order:

```text
0, -1, +1, -2, +2, ..., -8, +8
```

For a candidate `a`, the bucket size is:

```text
WAYS[remaining_coords][budget - |a|]
```

If the global rank lies in that bucket, the sampler emits `a` and replaces
`r` with the offset inside the selected bucket. This preserves the invariant
that the next descent step receives a local rank for the suffix problem.

Only the first descent step can observe a cumulative bucket sum exceeding
`u128`, because the full top-level ball has more than `2^128` elements. The
implementation treats `checked_add` overflow at that first step as selecting
the current bucket. This is sound because every top-level rank satisfies
`r < 2^128`, so any cumulative sum above `2^128` is necessarily greater than
`r`. After the first selected bucket, all suffix counts used by the descent fit
in `u128`.

## Small Decoding Example

For a toy parameter set:

```text
D = 3
M = 2
B = 3
```

there are:

```text
WAYS[3][3] = 57
```

valid vectors. If the coefficient order were the production magnitude-first
order:

```text
0, -1, +1, -2, +2
```

then the first-coordinate bucket sizes are:

```text
c_0 =  0: WAYS[2][3] = 21
c_0 = -1: WAYS[2][2] = 13
c_0 =  1: WAYS[2][2] = 13
c_0 = -2: WAYS[2][1] = 5
c_0 =  2: WAYS[2][1] = 5
```

These buckets partition the rank range `0..57`:

```text
r =  0..20 => c_0 =  0
r = 21..33 => c_0 = -1
r = 34..46 => c_0 =  1
r = 47..51 => c_0 = -2
r = 52..56 => c_0 =  2
```

The same rule then recurses with the offset inside the selected bucket, the
remaining coordinates, and the reduced `L1` budget.

## Performance Expectations

The bounded-`L1` sampler is slower than fixed-shape sampling in isolation:
fixed-shape sampling mostly does a small shuffle, sign draws, and tiny bounded
integer draws. The bounded-`L1` path performs DP bucket checks and constructs a
variable-size sparse output.

The intended win is protocol-level:

```text
smaller challenge_l1_mass
  -> smaller folded-witness bound beta
  -> fewer fold decomposition digits near digit boundaries
  -> smaller recursive witness or proof components
```

The sampler cost should be evaluated with both microbenchmarks and end-to-end
proof-size/prover/verifier measurements for affected D=32 modes.

## Tests and Review Gates

The implementation should keep these checks pinned:

- `BoundedL1Norm` validates for `D=32` and rejects other ring dimensions.
- `l1_norm() == 121` and `infinity_norm() == 8`.
- Domain bytes are exactly tag `2`, followed by `8` and `121` as little-endian `u64`s.
- Sampling with identical transcript state is deterministic.
- Sampled challenges have matching `positions` and `coeffs` lengths, unique positions, nonzero coefficients, `||c||_inf <= 8`, `||c||_1 <= 121`, and at most `32` stored coefficients.
- A fixed transcript reference vector pins the byte order, bucket order, and canonical XOF stream behavior.
- DP recurrence tests cover the compile-time suffix table.
- A top-cell test proves `WAYS[32][121] > 2^128`.
- Generated fp128 `D=32` schedules pin `challenge_l1_mass = 121` and validate against runtime `stage1_challenge_config(d).l1_norm()`.

## Non-Goals

- Do not adopt bounded-`L1` sampling for fp128 `D=64` in this branch.
- Do not adopt bounded-`L1` sampling for fp128 `D=128` in this branch.
- Do not expose a generic public `BoundedL1Ball { M, B }` config until there are tables, tests, and transcript fixtures for each supported triple.
- Do not serialize `SparseChallengeConfig` into proof objects; challenges are re-derived from `LevelParams.stage1_config` and transcript state.

## References

- `crates/akita-challenges/src/config.rs`
- `crates/akita-challenges/src/challenge.rs`
- `crates/akita-challenges/src/sampler/bounded_l1.rs`
- `crates/akita-challenges/src/sampler/xof.rs`
- `crates/akita-challenges/tests/sparse_challenge.rs`
- `crates/akita-config/src/proof_optimized.rs`
- `crates/akita-types/src/generated/fp128_d32_full.rs`
- `crates/akita-types/src/generated/fp128_d32_onehot.rs`
