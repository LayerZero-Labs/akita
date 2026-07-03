# Akita ZK Commitment Hiding Proof Note

## Goal

This note proves the statistical hiding claim for Akita's direct digit-source
ZK commitment-masking path.

For one wire-visible outer Ajtai commitment, the prover computes

```text
u = B_msg * t_hat + B_blind * rho
```

where:

- `B_msg` and `B_blind` are public uniformly sampled Ajtai columns over
`R_q`.
- `t_hat` is the decomposed message-dependent opening witness.
- `rho` is sampled freshly by the prover as direct blinding digit planes.
Every coefficient is uniform in the balanced base-`b` digit alphabet.
- `u` is revealed in `R_q^kappa`.

The claim is that, for every fixed message witness `t_hat`, the joint
distribution consisting of the public setup and `u` is negligibly statistically
close to the same public setup together with an independent uniform element of
`R_q^kappa`.

The proof has two independent parts:

1. The family `rho |-> B_blind * rho` is two-universal on the sampled
  masking domain.
2. The number of sampled digit planes gives enough min-entropy for the Leftover
  Hash Lemma (LHL).

SIS is not used for this hiding proof. SIS is the separate binding assumption
for a fixed public matrix.

## Setting and Assumptions

Let

```text
R_q = F_q[X] / (X^D + 1)
```

where:

- `q` is prime.
- `D` is a power of two.
- For the short-invertibility argument below, `q = 2l + 1 mod 4l` for a small
factorization parameter `l`.
- `B_blind` is sampled uniformly from `R_q^{kappa x s}`.
- `log_basis >= 1`, `b = 2^log_basis`, and `s` is the number of direct
blinding digit-ring columns.

Let the balanced digit alphabet be

```text
Digit_b = {-2^{log_basis-1}, ..., 2^{log_basis-1} - 1}.
```

The blinding source is

```text
rho <- Digit_b^{s * D},
```

viewed as `s` ring elements in `R_q`, each with `D` independently sampled digit
coefficients. Therefore

```text
H_min(rho) = s * D * log_basis.
```

We use the following ring-specific assumption.

**A1. Short digit differences are units.** For any two source vectors, every
nonzero digit-ring difference has Euclidean norm `< q^{1/l}`.

This is the ring-specific point. For power-of-two cyclotomic rings with
`q = 2l + 1 mod 4l`, the Lyubashevsky-Seiler short-invertibility lemma gives:

```text
0 < ||c||_2 < q^{1/l}  =>  c is a unit in R_q.
```

The case `q = 8k + 5` is the special case `l = 2`, so the threshold is
`sqrt(q)`.

The source differences are tiny. If base `b = 2^log_basis`, then each
coefficient of a digit difference lies in `[-(b - 1), b - 1]`, so

```text
||c||_2 <= sqrt(D) * (b - 1).
```

For Akita's production parameter ranges this is far below `q^{1/l}`. For
example, when `q ~= 2^128`, even the stricter `l = 4` threshold is about
`2^32`, while `sqrt(D) * (b - 1)` is tiny for the supported `D` and digit bases.

## The Hash Family

Define the hash family

```text
H = { h_B : Digit_b^{s * D} -> R_q^kappa }
h_B(rho) = B * rho
```

where the seed `B` is sampled uniformly from `R_q^{kappa x s}`.

This is the LHL hash family. The output `B * rho` is what acts as the random
pad for the commitment. The full commitment adds the fixed offset
`B_msg * t_hat`.

## Two-Universality

We prove that `H` is two-universal on the masking domain:

```text
for all rho != rho',
Pr_B[h_B(rho) = h_B(rho')] <= 1 / |R_q^kappa|.
```

Fix distinct source vectors `rho, rho' in Digit_b^{s * D}`. Let

```text
z = rho - rho' in R_q^s.
```

Since `rho != rho'`, `z != 0`. Therefore some coordinate `z_j` is nonzero. By
A1 and the short-invertibility lemma, this `z_j` is a unit in `R_q`.

Write one row of `B` as

```text
b = (b_1, ..., b_s) in R_q^s.
```

The event that this row collides on `r` and `r'` is

```text
<b, z> = sum_i b_i z_i = 0 in R_q.
```

Condition on all row entries except `b_j`. Since `z_j` is a unit, there is
exactly one value of `b_j` that satisfies the equation:

```text
b_j = -z_j^{-1} * sum_{i != j} b_i z_i.
```

Because `b_j` is uniform in `R_q`, the probability for this row is exactly

```text
Pr_b[<b, z> = 0] = 1 / |R_q|.
```

The `kappa` output rows are sampled independently, so

```text
Pr_B[B * z = 0] = (1 / |R_q|)^kappa
                = 1 / |R_q^kappa|.
```

Since

```text
h_B(rho) = h_B(rho')  <=>  B * (rho - rho') = 0,
```

the family is two-universal.

## LHL Statistical Distance

Let `Rho` be the random digit-source vector sampled uniformly from
`Digit_b^{s * D}`, and let `U` be uniform over `R_q^kappa`.

Because `Rho` has `s * D` independent coefficients, each uniform over an
alphabet of size `2^log_basis`,

```text
H_min(Rho) = s * D * log_basis.
```

For a two-universal hash family from the source domain to `R_q^kappa`, the
Leftover Hash Lemma gives a statement about the hash seed and the hash output
together:

```text
Delta((B, h_B(Rho)), (B, U))
    <= 1/2 * sqrt(|R_q^kappa| / 2^{H_min(Rho)})
    =  1/2 * sqrt(q^{D * kappa} / 2^{s * D * log_basis}).
```

The `B` appears in both tuples because `B` is public. The ideal distribution is
not just a standalone `U`; it is the same public seed `B` together with a value
`U` that is uniform in `R_q^kappa` and independent of `B`. This is the useful
form of LHL for a public-coin hash family: the hash output remains statistically
close to uniform even after the verifier/adversary sees which hash function was
chosen.

If we later ignore the public seed, marginalization can only decrease
statistical distance, so this joint statement also implies

```text
Delta(h_B(Rho), U) <= Delta((B, h_B(Rho)), (B, U)).
```

Equivalently, to make the distance at most `2^{-lambda}`, it is enough to have

```text
H_min(Rho) >= log2(|R_q^kappa|) + 2 * lambda - 2
```

that is,

```text
s * D * log_basis >= kappa * D * log2(q) + 2 * lambda - 2.
```

So it is enough to choose

```text
s >= ceil((kappa * D * log2(q) + 2 * lambda - 2)
          / (D * log_basis)).
```

Akita implements this rule conservatively using the field modulus bit width:

```text
s = ceil((kappa * D * field_bits + 254) / (D * log_basis)).
```

The direct digit-source construction therefore sizes the blinding segment in
digit-ring columns, not in full uniform ring masks. For example, when
`kappa = 1`, `field_bits = 128`, and `log_basis = 5`, the required count is
roughly

```text
s = ceil((D * 128 + 254) / (D * 5)).
```

For production `D`, this is about 26-27 digit-ring columns instead of sampling
two full ring elements and expanding them into `2 * num_digits_open` digit
columns. With the chosen `s`, the LHL target is

```text
Delta((B, B * Rho), (B, U)) <= 2^{-128}.
```

## Adding the Message Offset

The public commitment is not only the mask hash. It is

```text
u = B_msg * t_hat + B_blind * Rho.
```

For a fixed message witness `t_hat` and public `B_msg`, the term

```text
c = B_msg * t_hat
```

is a fixed element of `R_q^kappa`.

Adding a fixed offset is a bijection on `R_q^kappa`, so it preserves statistical
distance from uniform:

```text
Delta(c + B_blind * Rho, U)
  = Delta(B_blind * Rho, U).
```

Therefore, for every fixed message witness, the joint view containing the public
setup and the revealed commitment `u` is within `2^{-128}` statistical distance
of the same public setup and an independent uniform value in `R_q^kappa`.

The same proof applies independently to every fresh commitment group, replacing
`kappa` by that group's output ring length.

## Why SIS Is Not Enough

It is tempting to say that `B * rho` is a good LHL hash because SIS makes
collisions hard. That is not the right implication.

Two-universality says:

```text
for every fixed nonzero difference z,
Pr_B[B * z = 0] <= 1 / |R_q^kappa|.
```

The probability is over the random setup matrix `B`.

SIS says:

```text
given one fixed public B, it is computationally hard to find
a short nonzero z such that B * z = 0.
```

The probability is over an adversary's computation after seeing `B`.

The hiding proof above is information-theoretic and uses only the random linear
map's two-universality on the masked source domain. The SIS assumption is still
needed for binding, but it is not needed for the LHL statistical hiding bound.

## Caveats

The proof relies on the short-invertibility condition for nonzero digit
differences. For `q = 8k + 5`, this follows from the standard
Lyubashevsky-Seiler lemma with `l = 2` and threshold `sqrt(q)`. Other primes can
also work, but the appropriate `l` and the resulting threshold `q^{1/l}` must be
checked against the digit-difference norm bound.

If Akita switches a parameter set to a prime where the corresponding
short-invertibility lemma does not apply, then the two-universality argument
must be revisited. In that case the random linear map may still be
delta-almost-two-universal on the digit-source domain, but the collision
probability must be bounded by a new ring-specific argument rather than by the
unit-coordinate proof above.