# Lowering To `D=64`

## Goal

Figure out what the challenge family should look like if we try to push Hachi
all the way down to ring dimension `D=64`, ideally without increasing module
rank.

The first immediate issue is challenge entropy. At `D=64`, the current
exact-weight `{-1, +1}` family is too small, so we need a different challenge
space before worrying about code changes.

From the earlier local SIS sweeps, direct Hachi commitment binding at
`D=64, rank=1` did not look like the first thing that breaks. The challenge
family does break immediately, so that is the first lever to settle.

This note isolates that question:

1. what exact family size do we get at `D=64`?
2. what is the smallest multiplication blowup we can get while still reaching
   about `2^128` challenges?
3. if we allow a mix of `+-1` and `+-2`, what is the best distribution?

I am not redoing the full end-to-end security proof here. This is just the
challenge-family side.

## What "blowup" means

For a sparse challenge polynomial

```text
c(X) = sum_{i in S1} eps_i X^i + 2 * sum_{j in S2} delta_j X^j,
```

with `eps_i, delta_j in {-1, +1}`, define:

- `n1 := |S1|`, the number of `+-1` slots
- `n2 := |S2|`, the number of `+-2` slots
- total support `w := n1 + n2`

There are three natural cost metrics:

1. Exact family size / entropy:

```text
|F(n1, n2)| = C(64, n1) * C(64 - n1, n2) * 2^(n1 + n2)
bits(n1, n2) = log2 C(64, n1) + log2 C(64 - n1, n2) + (n1 + n2)
```

2. Coefficient-wise worst-case growth under multiplication by `c`:

```text
||c||_1 = n1 + 2 n2
```

If a ring element has coefficient bound `B`, the crude triangle-inequality
bound after multiplying by `c` is `B * (n1 + 2 n2)`.

3. Euclidean / spectral proxy:

```text
||c||_2^2 = n1 + 4 n2
```

This is the right first-order proxy when the proof analysis cares about
operator-norm growth rather than raw coefficient growth. Labrador already uses
an explicit operator-norm check for exactly this reason.

So there is no single "best" distribution without saying which blowup metric we
care about:

- if the objective is smallest `L2` / smallest spectral blowup, optimize
  `n1 + 4 n2`
- if the objective is smallest `L1` / smallest coefficient-wise blowup,
  optimize `n1 + 2 n2`

## Why pure `+-1` is dead at `D=64`

For an exact-weight `+-1` family at `D=64`, the family size is

```text
C(64, w) * 2^w.
```

The maximum over all `w` is only about

```text
max_w log2(C(64, w) * 2^w) = 98.19026859...
```

attained at `w = 43`.

So an exact-weight pure `+-1` family cannot reach 128 bits at `D=64`.

The same is true for any single-magnitude family `{-a, +a}`: you still only get
one sign bit per occupied position, so the entropy ceiling is the same.

Conclusion: at `D=64`, if we want an exact sparse family of size about `2^128`,
we need either:

- mixed magnitudes such as `+-1` and `+-2`,
- or a more structured family such as a split / class-based construction.

## Exact mixed `+-1` / `+-2` family

For fixed counts `(n1, n2)`, the entropy target is

```text
bits(n1, n2) >= 128.
```

I swept all `0 <= n1 + n2 <= 64` and extracted the Pareto frontier.

### Pareto frontier at `D=64`

| `n1` | `n2` | support | raw bits | `L1 = n1 + 2 n2` | `L2^2 = n1 + 4 n2` |
|---:|---:|---:|---:|---:|---:|
| 31 | 10 | 41 | 128.088 | 51 | 71 |
| 28 | 11 | 39 | 128.119 | 50 | 72 |
| 26 | 12 | 38 | 128.396 | 50 | 74 |
| 24 | 13 | 37 | 128.285 | 50 | 76 |
| 21 | 15 | 36 | 128.331 | 51 | 81 |

This gives two clean direct full-ring answers.

### Best direct full-ring distribution for smallest `L2` blowup

The exact optimum is:

```text
n1 = 31, n2 = 10
```

with

```text
bits = 128.088
L1 = 51
L2^2 = 71
L2 = sqrt(71) ~= 8.426
```

So if the relevant notion of blowup is operator norm / Euclidean growth, the
best direct exact 128-bit family is:

```text
31 coefficients in {+-1}, 10 coefficients in {+-2}.
```

### Best direct full-ring distribution for smallest `L1` blowup

The exact optimum is:

```text
n1 = 28, n2 = 11
```

with

```text
bits = 128.119
L1 = 50
L2^2 = 72
L2 = sqrt(72) ~= 8.485
```

So if the relevant notion of blowup is the crude coefficient-wise bound, the
best direct exact 128-bit family is:

```text
28 coefficients in {+-1}, 11 coefficients in {+-2}.
```

This beats the `31/10` family by one unit of `L1`, at the cost of a very small
increase in `L2`.

## Rigorous split family via `k=32`

The direct full-ring frontier above is still only a raw entropy / blowup sweep.
Those `(31,10)` and `(28,11)` shells do **not** yet come with a unit-difference
proof.

There is now a broader rigorous `k=32` split theorem over

```text
p32 = 2^128 - 5823,
R = F_p[X] / (X^64 + 1),
Y = X^2,
S = F_p[Y] / (Y^32 + 1).
```

For integers `0 <= m <= w <= 32`, define the class family

```text
A_{w,<=m} =
{
  sum_{t in T} eps_t lambda_t Y^t
  :
  T subseteq {0,...,31}, |T| = w,
  eps_t in {+-1},
  lambda_t in {1,2},
  #{t in T : lambda_t = 2} <= m
}.
```

Lift it to the full ring by

```text
C_{w,<=m} = { a0(X^2) + X a1(X^2) : a0,a1 in A_{w,<=m} }.
```

The exact class size is

```text
|A_{w,<=m}| = C(32, w) * 2^w * sum_{j=0}^m C(w, j),
|C_{w,<=m}| = |A_{w,<=m}|^2.
```

Every lifted challenge has:

- exact support `2w`
- max `L1 = 2(w + m)`
- max `L2^2 = 2(w + 3m)`

And the plain LS18 argument already proves:

```text
if w + 3m <= 63, then every nonzero pairwise difference in C_{w,<=m} is a unit.
```

Reason: every class element has `||a||_2^2 <= w + 3m`, so every nonzero class
difference satisfies

```text
||a - a'||_2 <= 2 * sqrt(w + 3m) <= 2 * sqrt(63) < p32^(1/32),
```

and `(2 * sqrt(63))^32 = 252^16 < p32`.

Three especially useful rigorous points are:

| Family | Total bits | Support | max `L1` | max `L2^2` | Why it matters |
| --- | ---: | ---: | ---: | ---: | --- |
| `C_{21,<=6}` | `128.538436` | `42` | `54` | `78` | strict Pareto improvement over the old rigorous exact-six-twos split shell |
| `C_{22,<=6}` | `129.381896` | `44` | `56` | `80` | same support / `L1` / `L2^2` budgets as the old rigorous shell, but `+1.12148` bits |
| `C_{18,<=10}` | `128.831818` | `36` | `56` | `96` | exact-support-`18` compromise that still clears `2^128` |

This changes the rigorous `D=64` picture:

- the old rigorous benchmark was the exact-six-twos split shell with total size
  about `2^128.260418`
- `C_{21,<=6}` is strictly better on entropy, support, `L1`, and `L2^2`
- `C_{22,<=6}` is the best drop-in same-budget replacement if you want more
  entropy without changing worst-case support / `L1` / `L2^2`
- `C_{18,<=10}` is the cleanest theorem-level answer if exact support `18` is
  the main sampler-shape objective

For the unrestricted split exact-support target `U_18`, the generic theorem
already proves almost all of the magnitude patterns: `C_{18,<=15}` still has
`129.623077` total bits, only `0.001894` bits below full `U_18`. The remaining
obstruction is concentrated in the last three magnitude shells with `16`, `17`,
or `18` twos, which need a sharper resultant / norm argument.

## Comparison against simpler code paths

### Uniform exact-weight alphabet `{+-1, +-2}`

If we keep the current generic `SparseChallengeConfig` model and just use

```text
nonzero_coeffs = {-2, -1, +1, +2}
```

with exact support size `w`, then the family size is

```text
C(64, w) * 4^w.
```

The smallest `w` reaching 128 bits is `w = 34`, giving

```text
bits = 128.491.
```

But the worst-case blowup is much worse:

```text
L1 <= 68
L2^2 <= 136
```

even though the expected mix is better on average.

So a fixed-count family is strictly better if we care about worst-case proof
bounds.

### Current Labrador-style split `(32, 8)`

The current Labrador sampler uses exactly:

```text
n1 = 32, n2 = 8.
```

At `D=64`, that exact family has size only

```text
log2 |F(32, 8)| = 123.995...
```

So it is already below 128 bits before any operator-norm rejection.

That means we cannot just port the current Labrador split down to `D=64`.

## Interaction with Labrador's current opnorm rejection

Labrador currently rejects challenges whose operator norm exceeds `14.0`.
I sampled the exact opnorm over the 64 odd roots of unity for the best raw
`D=64` candidates.

Monte Carlo estimates:

| `(n1, n2)` | raw bits | `L1` | `L2` | accept prob at bound `14` | effective bits after rejection |
|---|---:|---:|---:|---:|---:|
| `(31, 10)` | 128.088 | 51 | 8.426 | `~5.35%` | `~123.97` |
| `(28, 11)` | 128.119 | 50 | 8.485 | `~4.55%` | `~123.58` |
| `(26, 12)` | 128.396 | 50 | 8.602 | `~3.30%` | `~123.32` |
| `(32, 10)` | 130.796 | 52 | 8.718 | `~4.65%` | `~126.36` |
| `(32, 11)` | 132.027 | 54 | 8.944 | `~2.40%` | `~126.42` |

So the story changes sharply if we insist on the current Labrador rejection
rule:

- the raw exact optimum is no longer enough;
- even families with a few extra raw bits still land below 128 accepted bits;
- reaching 128 accepted bits at `D=64` under the same `opnorm <= 14` filter
  seems to require a much larger family, which destroys most of the blowup win.

So for `D=64`, there are really two different design questions:

1. Hachi exact sparse challenge family with no Labrador-style opnorm rejection.
2. Labrador-style challenge family with a hard opnorm cap.

They do not have the same optimum.

## Deterministic opnorm bound without rejection

There is a deterministic bound better than the crude `L1` bound, and for fixed
`(n1, n2)` it is essentially the exact worst-case over all support/sign choices.

Let

```text
gamma(c) := max_r |c(zeta_r)|,
zeta_r := exp(sqrt(-1) * (2r+1) * pi / D),
```

so `gamma(c)` is the Labrador operator norm.

For a challenge with `n1` coefficients in `+-1` and `n2` coefficients in
`+-2`, write the nonzero magnitudes in descending order as

```text
w_1, ..., w_m,
```

where `m = n1 + n2`, so this list is just

```text
2, ..., 2, 1, ..., 1.
```

Now fix one evaluation root `zeta_i` and one target output direction in the
complex plane.

Each exponent `j` contributes one antipodal pair

```text
{ +/- zeta_i^j }.
```

Because the coefficient sign is free, the best we can do with exponent `j` is
pick whichever member of that antipodal pair lies closer to the target
direction. Its contribution to the projection on that direction is then at most

```text
|a_j| cos(delta_j),
```

where `delta_j` is the smaller angular distance from that pair to the target.

The key simplification is that for odd `2i+1`, multiplication by `2i+1` is a
permutation mod `2D`, so the multiset of available distances does not depend on
`i`. After a global monomial shift, the sorted distance multiset is

```text
0, pi/D, pi/D, 2pi/D, 2pi/D, ..., (D/2) pi / D.
```

So if `delta_1 <= ... <= delta_m` are the `m` smallest entries in that list,
then every challenge in the fixed-count family satisfies

```text
gamma(c) <= T_det(D; n1, n2)
         := 2 * sum_{t=1}^{n2} cos(delta_t)
          +      sum_{t=n2+1}^{n1+n2} cos(delta_t).
```

This is tighter than the crude

```text
gamma(c) <= ||c||_1 = n1 + 2 n2
```

because every `cos(delta_t) < 1` except the very first one.

It is also essentially exact: equality is achieved by taking all signs aligned
and placing the support on the exponents whose directions are closest to a
common target direction, symmetrically so the imaginary parts cancel.

### What this buys us

For the relevant `D=64` families:

| `(n1, n2)` | crude `L1` bound | exact deterministic no-rejection bound |
|---|---:|---:|
| `(31, 10)` | 51.000 | 44.324 |
| `(28, 11)` | 50.000 | 44.183 |
| `(32, 8)` | 48.000 | 41.817 |

So at `D=64`, the crude `L1` bound is not tight. The exact family-level
worst-case is about `12%` to `13%` smaller.

But this is still nowhere near Labrador's current `T = 14`. So dropping
rejection and merely replacing `L1` by the tighter deterministic bound does
not preserve the current Labrador analysis.

### Why this barely helps at current `D=256`

At larger ring dimension, the smallest evaluation angle is `pi / D`, which gets
tiny. Then the cosine losses become tiny too, so the exact deterministic bound
collapses back toward `L1`.

For the current Labrador-style split `(32, 8)`:

| `D` | crude `L1` bound | exact deterministic no-rejection bound |
|---|---:|---:|
| `64` | 48.000 | 41.817 |
| `128` | 48.000 | 46.398 |
| `256` | 48.000 | 47.596 |

So:

- at `D=64`, the improvement is meaningful;
- at `D=256`, the crude bound is already almost best possible.

That is the real answer to "can we derive a tighter bound without rejection?":

- yes, there is a clean deterministic bound sharper than `L1`;
- it matters at small `D`;
- it does not buy much at the current larger `D`;
- and it is still far too large to substitute for Labrador's present
  `opnorm <= 14` rejection rule.

## Recommendation

If the target is a Hachi-side exact sparse challenge family for `D=64`, there
are now two different recommendation regimes:

1. If you only care about the raw direct full-ring frontier and are willing to
   postpone the proof question, use `(n1, n2) = (31, 10)` for the smallest
   direct `L2^2`, or `(n1, n2) = (28, 11)` for the smallest direct `L1`.
2. If you want a theorem-level family today under the plain LS18 machinery, use
   the split `k=32` family `C_{21,<=6}`.
3. If you want the same support / `L1` / `L2^2` budgets as the old rigorous
   exact-six-twos split shell but more entropy, use `C_{22,<=6}`.
4. If exact support `18` is the sampler-shape priority, use `C_{18,<=10}`.

If we instead need compatibility with the current Labrador opnorm rejection
framework, then the low-blowup `D=64` families above are not enough. In that
case the next step is not "pick a split"; it is "decide whether the opnorm
threshold or the rejection-based family construction should change."

## Bottom line

At `D=64`:

- pure exact-weight `+-1` is impossible for 128 bits;
- mixed `+-1` / `+-2` is enough;
- the best raw direct full-ring families are still

```text
(31, 10)  -> smallest direct L2^2
(28, 11)  -> smallest direct L1
```

- but the best rigorously proved split family is now `C_{21,<=6}`, with
  `128.538436` bits, support `42`, max `L1 = 54`, and max `L2^2 = 78`;
- if you want the same worst-case support / `L1` / `L2^2` budgets as the old
  rigorous shell, `C_{22,<=6}` gives `129.381896` bits;
- if exact support `18` matters most, `C_{18,<=10}` gives `128.831818` bits;
- but if you keep the current Labrador `opnorm <= 14` rejection rule, neither
  low-blowup family still gives 128 accepted bits.

## Labrador Ring-Dependent Profile

Labrador now uses a ring-dimension-dependent challenge family for larger ring
dimensions.

The current code change is:

1. keep the existing `opnorm <= 14` rejection sampler,
2. but replace Labrador's fixed challenge split `(tau1, tau2) = (32, 8)` with
   ring-dimension-specific exact sparse families:
   - `D=256 -> (23, 0)`
   - `D=128 -> (31, 0)`
   - `D=64 -> (32, 8)` unchanged
3. thread the corresponding second-moment factor
   `tau1 + 4 * tau2` into Labrador's planner, instead of always using `64`

The motivation was simple: under the same rejection rule, the accepted entropy
for `D=256` and `D=128` still stays above 128 bits, while the challenge
variance factor drops sharply:

| `D` | old `(tau1, tau2)` | current profile | old `tau1 + 4 tau2` | current `tau1 + 4 tau2` |
|---:|---:|---:|---:|---:|
| 256 | `(32, 8)` | `(23, 0)` | `64` | `23` |
| 128 | `(32, 8)` | `(31, 0)` | `64` | `31` |

That made Labrador materially faster, because the fold planner used a smaller
norm-growth proxy.

Measured behavior from this change:

- dense `nv=26`
  - `full`: `376,275 -> 347,603` bytes, verify `6.91s -> 5.82s`
  - `halving_full`: `259,373 -> 263,635` bytes, verify `3.80s -> 2.08s`
  - `d128_full`: `250,379 -> 252,593` bytes, verify `1.63s -> 0.625s`
- onehot `nv=30`
  - `onehot`: `373,411 -> 394,223` bytes, verify `2.29s -> 1.17s`
  - `halving_onehot`: `254,475 -> 256,689` bytes, verify `0.901s -> 0.625s`
  - `d128_onehot`: `250,379 -> 252,593` bytes, verify `1.41s -> 0.661s`

So the experiment traded a small proof-size regression in the aggressive
`D=128` profiles for a large runtime win. It also made the legacy `full` dense
profile smaller.

Why this is the right shape:

- it keeps `D=64` on the legacy C-parity family
- it lowers variance at `D=128` and `D=256` without relaxing the existing
  `opnorm <= 14` rejection rule
- the planner consumes the matching second moment instead of overestimating it

How to re-derive or adjust it later:

1. in `src/protocol/labrador/challenge.rs`, replace fixed `LABRADOR_TAU1` /
   `LABRADOR_TAU2` with ring-dimension-specific helpers returning
   `(32, 8)`, `(31, 0)`, `(23, 0)` for `D=64/128/256`
2. in the same file, size the support scratch arrays to the maximum support
   size (`40`) and add tests checking the expected support counts at `D=128`
   and `D=256`
3. in `src/protocol/labrador/config.rs`, replace the hardcoded variance factor
   `32 + 4 * 8` with the ring-dimension-specific `tau1 + 4 * tau2`
4. optionally update the transcript comment to say the sampler is
   ring-dimension-specific instead of fixed-`TAU1` / `TAU2`

This is orthogonal to the separate Hachi benchmark configs (`halving_*`,
`d128_*`). Those configs remain independent from the Labrador challenge-family
choice.
