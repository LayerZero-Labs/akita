# Exact `k=64` Challenge Family For `D=256`

## 2026-03-18 Update: `D=64` Field Selection

This note originally answered the `D=256, k=64` question. After lowering the
parent ring to `D=64`, the field-selection story changes:

1. the largest prime `< 2^128` with `p \equiv 65 \pmod{128}` (`k=32`) is

$$
p_{32} = 2^{128} - 5823;
$$

2. the largest prime `< 2^128` with `p \equiv 129 \pmod{256}` (`k=64`) is

$$
p_{64} = 2^{128} - 1407.
$$

By LS18 Corollary 1.2, the plain generic `\ell_2` criterion gives:

$$
k=32:\quad \|h\|_2 < p_{32}^{1/32} < 16,
\qquad
k=64:\quad \|h\|_2 < p_{64}^{1/64} < 4.
$$

So at `D=64` there is a sharp split:

- `k=32` is the clean "rigorous today" option if we want to stay inside the
  generic LS18 machinery.
- `k=64` is only attractive if we are willing to prove a new structured
  injectivity / norm argument beyond the generic lemma.

### Generic-safe `D=64` family via `k=32`

Work in

$$
R_{p_{32}} := \mathbb{F}_{p_{32}}[X]/(X^{64}+1),
\qquad
p_{32} = 2^{128} - 5823,
\qquad
Y := X^2,
\qquad
S_{p_{32}} := \mathbb{F}_{p_{32}}[Y]/(Y^{32}+1).
$$

The old exact-six-twos shell

$$
\mathcal{B}_{16,6}
:=
\left\{
\sum_{t \in S_1} \epsilon_t Y^t + 2\sum_{u \in S_2} \delta_u Y^u
\;:\;
S_1,S_2 \subseteq \{0,\dots,31\},\ |S_1|=16,\ |S_2|=6,\ S_1 \cap S_2 = \varnothing,
\epsilon_t,\delta_u \in \{\pm 1\}
\right\}
$$

is not the end of what the plain LS18 `k=32` argument certifies. The same
generic proof already works for a larger two-parameter family.

For integers `0 \le m \le w \le 32`, define

$$
\mathcal{A}_{w,\le m}
:=
\left\{
\sum_{t \in T} \epsilon_t \lambda_t Y^t
\;:\;
T \subseteq \{0,\dots,31\},\ |T|=w,\ \epsilon_t \in \{\pm 1\},\ \lambda_t \in \{1,2\},\ \#\{t \in T : \lambda_t = 2\} \le m
\right\}.
$$

Lift this class family to

$$
\mathcal{C}_{w,\le m}
:=
\left\{
a_0(X^2) + X a_1(X^2)
\;:\;
a_0,a_1 \in \mathcal{A}_{w,\le m}
\right\}.
$$

The class size is

$$
\left|\mathcal{A}_{w,\le m}\right|
=
\binom{32}{w} 2^w \sum_{j=0}^{m} \binom{w}{j},
$$

so

$$
\left|\mathcal{C}_{w,\le m}\right|
=
\left|\mathcal{A}_{w,\le m}\right|^2.
$$

Every lifted challenge in `\mathcal{C}_{w,\le m}` has:

1. exact support `2w`;
2. total `\ell_1` mass at most `2(w+m)`;
3. total `\ell_2^2` at most `2(w+3m)`.

### Theorem

If

$$
w + 3m \le 63,
$$

then `\mathcal{C}_{w,\le m}` is an exact strong-sampling family in
`R_{p_{32}}`: every nonzero pairwise difference is a unit.

### Proof

Take any `a \in \mathcal{A}_{w,\le m}`. If it has exactly `t` coefficients of
magnitude `2`, then

$$
\|a\|_2^2 = (w-t) \cdot 1^2 + t \cdot 2^2 = w + 3t \le w + 3m.
$$

So for any distinct `a,a' \in \mathcal{A}_{w,\le m}`,

$$
\|a-a'\|_2 \le \|a\|_2 + \|a'\|_2 \le 2\sqrt{w+3m} \le 2\sqrt{63}.
$$

Now

$$
(2\sqrt{63})^{32} = 252^{16}
= 264489632093832371229515191803011137536
< p_{32},
$$

so every nonzero class difference satisfies

$$
0 < \|a-a'\|_2 < p_{32}^{1/32}.
$$

By LS18 Corollary 1.2 in the `n=k=32` class ring, every such difference is
invertible in `S_{p_{32}}`.

Now take distinct lifted elements

$$
c(X)=a_0(X^2)+Xa_1(X^2),
\qquad
c'(X)=a_0'(X^2)+Xa_1'(X^2)
$$

in `\mathcal{C}_{w,\le m}`. At least one class component differs, say
`a_s \ne a_s'`. Then `a_s-a_s'` is invertible in `S_{p_{32}}`, so it evaluates
nonzero at every root `r_j` of `Y^{32}+1`. Modulo each irreducible quadratic
`X^2-r_j`, at least one of the two coordinates

$$
a_0(r_j)-a_0'(r_j),
\qquad
a_1(r_j)-a_1'(r_j)
$$

is nonzero, so `c-c'` is nonzero in every factor
`\mathbb{F}_{p_{32}}[X]/(X^2-r_j)`. Each factor is a field, hence `c-c'` is a
unit in every factor and therefore in `R_{p_{32}}`.

Pure split binary and ternary families are still too small at `D=64`: their
best possible sizes are only about `2^{58.33}` and `2^{95.89}`, respectively.
So mixed `\{\pm 1,\pm 2\}` families remain the first place where the generic
LS18 path reaches `2^{128}`.

### Useful `k=32` split families

The old exact-six-twos shell `\mathcal{B}_{16,6}` is just the exact-`6` shell
inside `\mathcal{A}_{22,\le 6}`. The broader theorem gives three especially
useful rigid points:

$$
\log_2 |\mathcal{C}_{21,\le 6}| \approx 128.5384362731,
\qquad
\text{support } 42,
\qquad
\ell_1 \le 54,
\qquad
\ell_2^2 \le 78;
$$

$$
\log_2 |\mathcal{C}_{22,\le 6}| \approx 129.3818956991,
\qquad
\text{support } 44,
\qquad
\ell_1 \le 56,
\qquad
\ell_2^2 \le 80;
$$

$$
\log_2 |\mathcal{C}_{18,\le 10}| \approx 128.8318179504,
\qquad
\text{support } 36,
\qquad
\ell_1 \le 56,
\qquad
\ell_2^2 \le 96.
$$

Here:

1. `\mathcal{C}_{21,\le 6}` is a strict Pareto improvement over the old rigorous
   `\mathcal{B}_{16,6}` lift: it has more entropy, smaller support, smaller
   `\ell_1`, and smaller `\ell_2^2`.
2. `\mathcal{C}_{22,\le 6}` keeps the same support / `\ell_1` / `\ell_2^2`
   budgets as `\mathcal{B}_{16,6}`, but gains about `1.12148` bits of total
   challenge entropy.
3. `\mathcal{C}_{18,\le 10}` is the cleanest exact-support-`18` theorem-level
   substitute I know that still clears `2^{128}`.

For reference:

| Family | Rigorous? | Total bits | Support | max `L1` | max `L2^2` | Comment |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| split uniform `U_15` | yes | `118.151040` | `30` | `60` | `120` | current rigorous uniform point |
| split uniform `U_16` | no | `122.325965` | `32` | `64` | `128` | above plain LS18 cutoff |
| split uniform `U_17` | no | `126.151040` | `34` | `68` | `136` | above plain LS18 cutoff |
| split uniform `U_18` | no | `129.624971` | `36` | `72` | `144` | unrestricted target |
| `\mathcal{C}_{18,\le 10}` | yes | `128.831818` | `36` | `56` | `96` | exact-support `18` compromise |
| `\mathcal{C}_{18,\le 15}` | yes | `129.623077` | `36` | `66` | `126` | only `0.001894` bits below full `U_18` |
| old exact shell `\mathcal{B}_{16,6}` lift | yes | `128.260418` | `44` | `56` | `80` | previous rigorous benchmark |
| `\mathcal{C}_{21,\le 6}` | yes | `128.538436` | `42` | `54` | `78` | strict Pareto improvement |
| `\mathcal{C}_{22,\le 6}` | yes | `129.381896` | `44` | `56` | `80` | same budgets, more entropy |
| direct full-ring `(31,10)` | not yet | `128.088000` | `41` | `51` | `71` | raw frontier, no proof yet |
| direct full-ring `(28,11)` | not yet | `128.119000` | `39` | `50` | `72` | raw frontier, no proof yet |

### How close this gets to full `U_18`

The unrestricted split exact-support family

$$
U_{18}
:=
\left\{
\sum_{t \in T} a_t Y^t
\;:\;
T \subseteq \{0,\dots,31\},\ |T|=18,\ a_t \in \{\pm 1,\pm 2\}
\right\}
$$

still does not have a full proof. But the generic theorem above already proves
almost all of it.

Indeed, `\mathcal{C}_{18,\le 15}` keeps all magnitude patterns with at most `15`
twos per class, so it retains

$$
\frac{\sum_{j=0}^{15} \binom{18}{j}}{2^{18}}
=
\frac{261972}{262144}
=
\frac{65493}{65536}
\approx 0.999344
$$

of the `U_{18}` magnitude choices on each chosen support, and yet still has

$$
\log_2 |\mathcal{C}_{18,\le 15}| \approx 129.6230771246,
$$

which is only

$$
129.6249709310 - 129.6230771246 \approx 0.0018938064
$$

bits below full `U_{18}`.

The unresolved obstruction is now very narrow. Let `U_{18,t}` denote the exact
support-`18` shell with exactly `t` coefficients of magnitude `2`. Then for
`a \in U_{18,t}` and `a' \in U_{18,t'}`,

$$
\max \|a-a'\|_2^2
=
72 + 7\min(t,t') + 5\max(t,t').
$$

This comes from complete support overlap, opposite signs everywhere, and
greedily pairing `2` against `2`, then `2` against `1`, then `1` against `1`.

In particular,

$$
(t,t')=(15,15) \Rightarrow \max \|a-a'\|_2^2 = 252,
\qquad
(t,t')=(15,16) \Rightarrow \max \|a-a'\|_2^2 = 257.
$$

So shell `16` is exactly where the plain `256` threshold breaks. The remaining
work for full `U_{18}` is concentrated in the last three magnitude shells
`t \in \{16,17,18\}`.

There are still primitive large-norm differences there. For example,

$$
h(Y)=4\sum_{i=0}^{16} Y^i + 3Y^{17}
$$

is a valid difference of two `U_{18}` elements, has coefficient gcd `1`, and
satisfies

$$
\|h\|_2^2 = 17 \cdot 16 + 9 = 281.
$$

So a full `U_{18}` proof must beat the plain generic `256` cutoff on primitive
patterns, probably by an exact resultant / algebraic-norm argument.

### What still needs a bespoke proof

The full `D=64` mixed shells from `LOWERING_TO_D64.md`, especially `(31,10)` and
`(28,11)`, are still not covered by the plain generic LS18 criterion. Their
universal triangle-inequality bounds are

$$
2\sqrt{31+4 \cdot 10} = 2\sqrt{71} \approx 16.85,
\qquad
2\sqrt{28+4 \cdot 11} = 2\sqrt{72} \approx 16.97,
$$

so they remain above the `k=32` cutoff `16` and far above the `k=64` cutoff
`4`.

Likewise, at `D=64`, the old `k=64` miracle from the rest of this note does not
transfer automatically: the one-class binary weight-`8` family over `64` slots
has size only

$$
\binom{64}{8} \approx 2^{32.0434102073},
$$

and it reached `\approx 2^{128}` in the `D=256` setting only because there were
four residue classes to tensor together.

So the current best summary is:

1. if we want a rigorous `D=64` family from the generic LS18 lemma alone, use
   `k=32` and one of the split families above, with `\mathcal{C}_{21,\le 6}` as
   the best strict Pareto point and `\mathcal{C}_{22,\le 6}` as the best
   same-budget replacement for the old exact-six-twos shell;
2. if we specifically want full `U_{18}`, only the last three magnitude shells
   still need a bespoke proof;
3. if we want the raw-optimal direct `D=64` families, or a true `k=64` full
   split at `D=64`, we still need an additional derivation comparable in spirit
   to the bespoke argument later in this note.

The rest of this file remains the original `D=256, k=64` construction.

## Goal

Construct a challenge family for

$$
R_p := \mathbb{F}_p[X]/(X^{256}+1),
\qquad
p := 2^{128} - 1407,
$$

such that:

1. the `k=64` partial split is used;
2. the family size is about `2^128`;
3. every nonzero pairwise difference is invertible in `R_p`;
4. the challenges stay genuinely sparse.

The result below gives an **exact** strong-sampling family of size

$$
\binom{64}{8}^4 \approx 2^{128.1736408293},
$$

with coefficient alphabet `{0,1} ⊂ {-1,0,1}` and total Hamming weight exactly `32`.

This is stronger than the usual LS18-style approximate story: there is no heuristic
"hope the difference is a unit" step anywhere in the proof.

## 1. The `k=64` factorization

Since

$$
p = 2^{128} - 1407 \equiv 129 \pmod{256},
$$

we are in the LS18 / ACX19 regime with `k=64` for the `512`-th cyclotomic:

$$
\Phi_{512}(X) = X^{256}+1.
$$

Let

$$
Y := X^4.
$$

Then `Y^64 + 1` splits completely over `\mathbb{F}_p`, while `X^4-r` stays irreducible for
each root `r` of `Y^64+1`. Concretely,

$$
X^{256}+1 = \prod_{j=0}^{63} (X^4-r_j)
$$

for distinct `r_j \in \mathbb{F}_p^\times`, and each factor `X^4-r_j` is irreducible over
`\mathbb{F}_p`.

Equivalently,

$$
R_p \cong \prod_{j=0}^{63} \mathbb{F}_p[X]/(X^4-r_j).
$$

Now write any ring element as

$$
c(X)=c_0(Y)+Xc_1(Y)+X^2c_2(Y)+X^3c_3(Y),
\qquad
c_s(Y)\in \mathbb{F}_p[Y]/(Y^{64}+1).
$$

Modulo `X^4-r_j`, this becomes

$$
c(X) \bmod (X^4-r_j)
= c_0(r_j)+Xc_1(r_j)+X^2c_2(r_j)+X^3c_3(r_j).
$$

Because `1,X,X^2,X^3` is a basis of `\mathbb{F}_p[X]/(X^4-r_j)`, we get the exact unit
criterion:

### Proposition 1

For `c(X)=\sum_{s=0}^3 X^s c_s(Y) \in R_p`,

$$
c \in R_p^\times \iff \forall j \in \{0,\dots,63\},\quad (c_0(r_j),c_1(r_j),c_2(r_j),c_3(r_j)) \neq (0,0,0,0).
$$

So to build an exact strong-sampling family in `R_p`, it is enough to build, for each root
`r_j`, an injective evaluation family in the class ring

$$
S_p := \mathbb{F}_p[Y]/(Y^{64}+1).
$$

## 2. The class family

For each `8`-subset `S \subseteq \{0,\dots,63\}`, define

$$
a_S(Y) := \sum_{t \in S} Y^t \in S_p.
$$

Let

$$
\mathcal{A}_8 := \{ a_S(Y) : S \subseteq \{0,\dots,63\},\ |S|=8 \}.
$$

This family is binary, weight-`8`, and has size

$$
|\mathcal{A}_8| = \binom{64}{8} = 4{,}426{,}165{,}368 \approx 2^{32.0434102073}.
$$

The key theorem is that `\mathcal{A}_8` is already an exact strong-sampling family in
`S_p`.

## 3. Exact one-class injectivity

### Theorem 2

Let `S,T \subseteq \{0,\dots,63\}` be distinct `8`-subsets. Then

$$
a_S(Y)-a_T(Y)
$$

is invertible in `S_p = \mathbb{F}_p[Y]/(Y^{64}+1)`.

Equivalently, for every root `r` of `Y^{64}+1` in `\mathbb{F}_p`,

$$
a_S(r) \neq a_T(r).
$$

### Proof

Set

$$
h(Y) := a_S(Y)-a_T(Y).
$$

Cancel the overlap `S \cap T`. Then there exist disjoint sets `U,V` with

$$
|U|=|V|=:m,\qquad 1 \le m \le 8,
$$

such that

$$
h(Y)=\sum_{u \in U} Y^u - \sum_{v \in V} Y^v.
$$

So the coefficients of `h` lie in `\{-1,0,1\}`, the support size is `2m`, and the
coefficient sum is zero.

#### Case 1: `m <= 7`

Then

$$
\|h\|_2 = \sqrt{2m} \le \sqrt{14}.
$$

Since

$$
14^{32} < p < 4^{64},
$$

we can argue directly. Let

$$
z_j := h(\xi^{2j+1}),
\qquad
a_j := |z_j|^2,
\qquad
0 \le j \le 63.
$$

By Parseval,

$$
\frac{1}{64}\sum_{j=0}^{63} a_j = \|h\|_2^2 = 2m \le 14.
$$

Hence, by AM-GM,

$$
|N(h(\xi))|^2 = \prod_{j=0}^{63} a_j \le (2m)^{64} \le 14^{64}.
$$

Therefore

$$
|N(h(\xi))| \le 14^{32} < p.
$$

As in Case 2 below, if `h(r)=0` for some root `r` of `Y^{64}+1` modulo `p`, then `p` would
divide `N(h(\xi))`, contradiction. So `h` is invertible in `S_p`.

#### Case 2: `m = 8`

This is the knife-edge case not covered by the plain LS18 `\|h\|_2 < p^{1/64}` criterion,
because now `\|h\|_2 = 4`.

We handle it by a sharper norm computation.

Let `\xi` be a primitive complex `128`-th root of unity, and define

$$
z_j := h(\xi^{2j+1}),
\qquad
a_j := |z_j|^2,
\qquad
0 \le j \le 63.
$$

The algebraic norm of `h(\xi)` from `\mathbb{Q}(\xi)` to `\mathbb{Q}` is

$$
N(h(\xi)) = \prod_{j=0}^{63} z_j,
$$

so

$$
|N(h(\xi))|^2 = \prod_{j=0}^{63} a_j.
$$

Also, `h(\xi) \neq 0`: the polynomial `h` is nonzero and has degree at most `63`, while
`\xi` has minimal polynomial `\Phi_{128}(Y)=Y^{64}+1` of degree `64`.

Write

$$
h(Y)=\sum_{t=0}^{63} c_t Y^t,
\qquad
c_t \in \{-1,0,1\},
$$

with exactly eight `+1` coefficients and eight `-1` coefficients.

Now define the negacyclic autocorrelations

$$
D_s :=
\sum_{t=0}^{63-s} c_t c_{t+s}

\;-\;
\sum_{t=64-s}^{63} c_t c_{t+s-64},
\qquad
0 \le s \le 63.
$$

Then:

1. `D_0 = 16`.
2. `D_{64-s} = -D_s` for `1 <= s <= 63`.
3. The usual DFT identities for the sequence `b_t := c_t \xi^t` give

$$
\frac{1}{64}\sum_{j=0}^{63} a_j = 16,
\qquad
\frac{1}{64}\sum_{j=0}^{63} a_j^2 = \sum_{s=0}^{63} D_s^2.
$$

Indeed, `z_j` is the `64`-point Fourier transform of `(b_t)`, so Parseval gives
`64^{-1}\sum a_j = \sum |b_t|^2 = 16`. For the fourth moment, let

$$
B_s := \sum_{t=0}^{63} b_t \overline{b_{t+s \bmod 64}}.
$$

The standard DFT identity gives

$$
\frac{1}{64}\sum_{j=0}^{63} a_j^2 = \sum_{s=0}^{63} |B_s|^2.
$$

Now, using `b_t = c_t \xi^t` and `\xi^{64}=-1`, a direct wraparound calculation gives

$$
B_s = \xi^{-s}\left(\sum_{t=0}^{63-s} c_t c_{t+s} - \sum_{t=64-s}^{63} c_t c_{t+s-64}\right) = \xi^{-s} D_s.
$$

Hence `|B_s|^2 = D_s^2`, which proves the stated fourth-moment formula.

Now split into two subcases.

##### Subcase 2a: all nonzero `D_s` vanish

If `D_s=0` for all `s != 0`, then

$$
\frac{1}{64}\sum a_j^2 = D_0^2 = 256.
$$

Together with `64^{-1}\sum a_j = 16`, this forces all `a_j` to equal `16`. Hence

$$
|N(h(\xi))|^2 = 16^{64} = 2^{256},
\qquad
|N(h(\xi))| = 2^{128}.
$$

Therefore

$$
N(h(\xi)) = \pm 2^{128}.
$$

Since `p = 2^{128}-1407` is odd, `p` does not divide `N(h(\xi))`.

##### Subcase 2b: some nonzero `D_s` is nonzero

By `D_{64-s}=-D_s`, any nonzero shift contributes at least `2` to
`\sum_{s=0}^{63} D_s^2` beyond the `D_0^2=256` term. So

$$
\frac{1}{64}\sum a_j^2 \ge 258.
$$

Hence the variance of the `a_j` is at least

$$
\frac{1}{64}\sum (a_j-16)^2 = \frac{1}{64}\sum a_j^2 - 16^2 \ge 2.
$$

So not all `a_j` lie in the open interval `(16-\sqrt{2},\,16+\sqrt{2})`. Equivalently,
some `a_j` lies in the closed complement

$$
[0,16-\sqrt{2}] \cup [16+\sqrt{2},\infty).
$$

For a fixed outlier value `x`, the product `\prod a_j` is maximized when the other
`63` values are equal, by AM-GM. Thus

$$
\prod_{j=0}^{63} a_j \le x \left(\frac{1024-x}{63}\right)^{63}.
$$

for some `x \in [0,16-\sqrt{2}] \cup [16+\sqrt{2},1024]`.

The one-variable function

$$
f(x):=
x \left(\frac{1024-x}{63}\right)^{63}
$$

has derivative

$$
\frac{f'(x)}{f(x)} = \frac{1}{x} - \frac{63}{1024-x}
= \frac{1024-64x}{x(1024-x)},
$$

so it is increasing on `(0,16)` and decreasing on `(16,1024)`.

To compare the two boundary points, set

$$
g(d):=\log f(16+d)-\log f(16-d)
\qquad (0<d<16).
$$

Then

$$
g'(d) = \frac{126976\,d^2}{(256-d^2)(1016064-d^2)} > 0.
$$

so `g(d)>0` for all `d>0`, hence `f(16+d) > f(16-d)`. Therefore the worst case on the
admissible domain is the boundary value `x = 16+\sqrt{2}`:

$$
\prod a_j \le \left(16+\sqrt{2}\right)\left(\frac{1008-\sqrt{2}}{63}\right)^{63}.
$$

Taking square roots, and using `707/500 < \sqrt{2}` together with the fact that `f` is
decreasing on `(16,1024)`,

$$
|N(h(\xi))| \le \sqrt{\left(16+\sqrt{2}\right)\left(\frac{1008-\sqrt{2}}{63}\right)^{63}} \!<\! \sqrt{\left(16+\frac{707}{500}\right)\left(\frac{1008-\frac{707}{500}}{63}\right)^{63}} \approx 3.3964471702223210393 \cdot 10^{38}.
$$

The final strict comparison with `p` was checked using exact rational arithmetic after the
replacement `\sqrt{2} > 707/500`.

But

$$
p = 340282366920938463463374607431768210049
\approx
3.4028236692093846346 \cdot 10^{38},
$$

so

$$
|N(h(\xi))| < p.
$$

In both subcases, `p` does not divide `N(h(\xi))`.

Finally, because `p \equiv 1 \pmod{128}`, the prime `p` splits completely in
`\mathbb{Q}(\xi)`. So if `h(r)=0` for some root `r \in \mathbb{F}_p` of `Y^{64}+1`,
then for a prime ideal `\mathfrak p` above `p` with `\xi \bmod \mathfrak p = r`, we would
have `h(\xi) \in \mathfrak p`, hence `p | N(h(\xi))`, contradiction.

Therefore `h(r) != 0` for every root `r` of `Y^{64}+1`, so `h` is invertible in `S_p`.

This proves the theorem.

## 4. The full `k=64` family in `R_p`

Define the lifted family

$$
\mathcal{C}_{64}
:=
\left\{
\sum_{s=0}^3 X^s a_{S_s}(X^4)
\;:\;
S_0,S_1,S_2,S_3 \subseteq \{0,\dots,63\},\ |S_s|=8
\right\}.
$$

Equivalently, a challenge in `\mathcal{C}_{64}` has coefficient vector in `{0,1}^{256}`
with exactly `8` ones in each residue class modulo `4`:

$$
\operatorname{supp}(c) \cap \{ s, s+4, s+8, \dots, s+252 \}
\text{ has size } 8
\quad
(\forall s \in \{0,1,2,3\}).
$$

So every challenge has:

1. coefficient alphabet `{0,1}`;
2. total Hamming weight `32`;
3. support size

$$
|\mathcal{C}_{64}| = \binom{64}{8}^4 \approx 2^{128.1736408293}.
$$

### Theorem 3

`\mathcal{C}_{64}` is an exact strong-sampling family in `R_p`:

$$
c \neq c' \in \mathcal{C}_{64} \implies c-c' \in R_p^\times.
$$

### Proof

Write

$$
c(X)=\sum_{s=0}^3 X^s c_s(Y),
\qquad
c'(X)=\sum_{s=0}^3 X^s c'_s(Y),
\qquad
Y=X^4.
$$

If `c != c'`, then for some residue class `s`, we have `c_s != c'_s`.

By Theorem 2, for every root `r_j` of `Y^{64}+1`, the value `c_s(r_j)-c'_s(r_j)` is
nonzero. Hence for every `j`,

$$
(c_0(r_j)-c'_0(r_j),\ c_1(r_j)-c'_1(r_j),\ c_2(r_j)-c'_2(r_j),\ c_3(r_j)-c'_3(r_j))
\neq
(0,0,0,0).
$$

Proposition 1 then implies `c-c'` is invertible in `R_p`.

## 5. Practical parameters

This family has the following concrete parameters:

- Ring: `R_p = F_p[X]/(X^256+1)` with `p = 2^128 - 1407`.
- Partial split: `k=64`, factor degree `4`.
- Per-residue-class support: exact weight `8`.
- Total support: exact weight `32`.
- Challenge space size:

$$
\log_2 |\mathcal{C}_{64}| = 4 \log_2 \binom{64}{8} \approx 128.1736408293.
$$

- Exact unit-difference property: **yes**.
- Heuristic bad-pair term: **none**.

For coefficient-`\ell_\infty` operator norm, the trivial bound is

$$
\gamma_{\mathcal{C}_{64}} \le 32,
$$

because multiplying by a `0/1` polynomial of Hamming weight `32` can increase any output
coefficient by at most the sum of `32` input coefficients.

So this is still a genuinely sparse family: it is denser than the current Hachi
weight-`23` signed family, but only by a factor of `32/23 \approx 1.39`, while upgrading
the difference condition from heuristic / approximate to exact.

## 6. Summary

The exact `k=64` answer is:

1. `X^256+1` factors into `64` irreducible quartics over `F_p` for
   `p = 2^128 - 1407`.
2. The family

$$
\mathcal{C}_{64} = \left\{ \sum_{s=0}^3 \sum_{t \in S_s} X^{4t+s} \;:\; |S_s|=8 \right\}
$$

has size `\binom{64}{8}^4 \approx 2^{128.17}`.
3. Every nonzero difference of two challenges in `\mathcal{C}_{64}` is invertible in
   `R_p`.
4. The proof is fully rigorous and does not use the heuristic "fully split, hope for the
   best" argument from the approximate strong-sampling literature.

## 7. What this does not solve

This note proves exact invertibility of challenge differences for the `k=64` family above.
It does **not** by itself settle the separate implementation question of whether the whole
Hachi prover can delete the current CRT / quotient path; that still depends on the rest of
the multiplication pipeline.
