# Multilinear extensions and sum-check

Sum-check lets Akita prove a sum over an exponentially large table without
putting that table in the proof. It replaces one claim about all $2^n$ table
entries with $n$ short polynomial messages and one final evaluation claim.

This chapter builds the protocol from a four-entry example. It then explains
the compressed messages and unequal-size batching used by the implementation.

## A table is also a polynomial

Consider a table indexed by two bits:

| $x_0$ | $x_1$ | $f(x_0,x_1)$ |
| ---: | ---: | ---: |
| 0 | 0 | 2 |
| 1 | 0 | 5 |
| 0 | 1 | 7 |
| 1 | 1 | 11 |

There is exactly one polynomial that agrees with this table on the four
Boolean points and has degree at most one in each variable. It is the
**multilinear extension** of the table:

$$
\widetilde f(X_0,X_1)
= 2(1-X_0)(1-X_1)
+ 5X_0(1-X_1)
+ 7(1-X_0)X_1
+ 11X_0X_1.
$$

The four terms are selectors. At $(X_0,X_1)=(1,0)$, for example, only the
second selector is one, so the polynomial returns $5$.

The general selector for two Boolean vectors $a,x\in\{0,1\}^n$ is

$$
\operatorname{eq}(a,x)
= \prod_{i=0}^{n-1}
\bigl(a_i x_i + (1-a_i)(1-x_i)\bigr).
$$

It is one when $a=x$ and zero at every other Boolean point. For a table
$f:\{0,1\}^n\rightarrow\mathbb F$, its multilinear extension is therefore

$$
\widetilde f(X)
= \sum_{a\in\{0,1\}^n} f(a)\operatorname{eq}(a,X).
$$

The polynomial also accepts field values that are not bits. This is what makes
it useful in a proof. The verifier can derive a random field point from the
transcript and ask for one value there.

### Table order in Akita

Akita stores Boolean tables in little-endian order. Coordinate $x_0$ is the
least significant index bit, so adjacent entries differ in $x_0$. The example
above is stored as `[2, 5, 7, 11]`.

Binding $x_0=r$ replaces each adjacent pair $(a,b)$ with

$$
(1-r)a+rb=a+r(b-a).
$$

The four-entry table becomes a two-entry table. Binding the next coordinate
leaves one value, which is $\widetilde f(r_0,r_1)$. The implementation calls
this operation a **fold**.

`multilinear_eval` in `akita-algebra/src/poly.rs` evaluates a table this way.
`fold_evals_in_place` provides the mutable fold used throughout prover code.
`EqPolynomial` in `akita-algebra/src/eq_poly.rs` builds the equality selectors
in the same index order.

## What sum-check proves

Suppose the prover claims

$$
T=\sum_{x\in\{0,1\}^n}g(x),
$$

where $g$ has degree at most $d$ in each variable. Directly checking the claim
would take $2^n$ evaluations. Sum-check reduces it one variable at a time.

In round zero, the prover sends the univariate polynomial

$$
s_0(X)=\sum_{x_1,\ldots,x_{n-1}\in\{0,1\}}g(X,x_1,\ldots,x_{n-1}).
$$

The verifier checks

$$
T=s_0(0)+s_0(1).
$$

It then samples a challenge $r_0$ from the transcript and replaces the claim
with $T_1=s_0(r_0)$. Round one does the same for the next variable:

$$
s_1(X)=\sum_{x_2,\ldots,x_{n-1}\in\{0,1\}}
g(r_0,X,x_2,\ldots,x_{n-1}).
$$

After $n$ rounds, every variable has been bound. The verifier checks the final
claim against $g(r_0,\ldots,r_{n-1})$, computed from the protocol's final
oracle values. The large Boolean sum has become one evaluation at a random
point.

## A complete two-round example

Return to the table `[2, 5, 7, 11]`. Its claimed Boolean sum is

$$
T=2+5+7+11=25.
$$

The first round polynomial sums over $x_1$:

$$
\begin{aligned}
s_0(X)
&=\widetilde f(X,0)+\widetilde f(X,1)\\
&=(2+3X)+(7+4X)\\
&=9+7X.
\end{aligned}
$$

The verifier checks $s_0(0)+s_0(1)=9+16=25$. Suppose its first challenge is
$r_0=3$. The new claim is $T_1=s_0(3)=30$.

The second round polynomial is

$$
s_1(X)=\widetilde f(3,X)=11+8X.
$$

Again, $s_1(0)+s_1(1)=11+19=30$. If the next challenge is $r_1=4$, the final
claim is $s_1(4)=43$. Evaluating the original multilinear extension at $(3,4)$
also gives $43$.

Real challenges are unpredictable field elements derived from the transcript.
Small integers are used here only to make every step visible.

## Why a false claim is caught

At the start of a round, the previous claim fixes the sum $s_j(0)+s_j(1)$.
The prover may try to send a different degree-$d$ polynomial that has the same
sum. Two distinct degree-$d$ polynomials agree at no more than $d$ field
points. A fresh random challenge therefore exposes that substitution except
with probability at most $d/|\mathbb F|$ in that round.

Across $n$ rounds, the usual bound is

$$
\Pr[\text{accept a false claim}]\leq \frac{nd}{|\mathbb F|}.
$$

Akita chooses large challenge fields, so this algebraic error is small. The
complete security argument also accounts for transcript challenges, commitment
binding, and every other reduction. Those pieces are assembled in
[Security model and parameters](../how/security.md).

The extraction argument uses a related fact. A degree-$d$ round polynomial is
determined by $d+1$ distinct evaluations. If an extractor can obtain $d+1$
accepting continuations from the same transcript prefix, it can reconstruct
that round polynomial. This is the special-soundness property used when
sum-check is connected to commitment binding.

## Akita sends one fewer coefficient

Write a round polynomial as

$$
s(X)=c_0+c_1X+c_2X^2+\cdots+c_dX^d.
$$

The verifier already knows the current claim $T_j=s(0)+s(1)$. Since

$$
s(0)+s(1)=2c_0+c_1+\sum_{i=2}^{d}c_i,
$$

the linear coefficient is determined by

$$
c_1=T_j-2c_0-\sum_{i=2}^{d}c_i.
$$

The proof sends only `[c0, c2, ..., cd]`. The verifier recovers the missing
linear contribution while evaluating the message at the next challenge. This
saves one field element in every round without changing the protocol.

The types `UniPoly` and `CompressedUniPoly` in
`akita-algebra/src/uni_poly.rs` own this rule. Proof decoding is headerless:
the verifier derives each expected coefficient count from the proof shape
rather than trusting a length supplied by the proof.

## Batching several claims

Akita often needs several sum-checks at the same point in the protocol. Running
them separately would repeat challenges and round messages. Instead, the
transcript derives one random coefficient $\rho_i$ for each instance and the
prover sends the linear combination of their round polynomials.

If all instances have the same number of variables, their initial claims
$T_i$ combine as

$$
T_{\mathrm{batch}}=\sum_i \rho_iT_i.
$$

The same coefficients combine every round polynomial and the final oracle
values. A false component is therefore hidden only if the random linear
combination happens to cancel it.

### Instances with different numbers of variables

The current implementation right-aligns every instance in the longest
challenge vector. If the longest instance has $N$ variables and one shorter
instance has $n$ variables, the shorter instance receives $N-n$ constant dummy
rounds before its first real round.

A dummy polynomial for running claim $C$ is the constant $C/2$. Its two Boolean
evaluations sum to $C$, and evaluating it at any challenge returns $C/2$.
After all $N-n$ dummy rounds, the claim has been divided by $2^{N-n}$.
Therefore the batch starts that instance at

$$
2^{N-n}T_i.
$$

The dummy rounds reduce it back to $T_i$ exactly when the instance becomes
active. Its real variables then use the suffix

$$
(r_{N-n},\ldots,r_{N-1})
$$

of the shared challenge vector. This is the `offset = max_num_rounds - n`
rule in `batched_sumcheck.rs`.

## Where to inspect the implementation

| Rule | Implementation |
| --- | --- |
| Evaluate a multilinear table | `akita-algebra/src/poly.rs` |
| Build equality-polynomial tables | `akita-algebra/src/eq_poly.rs` |
| Represent and compress round polynomials | `akita-algebra/src/uni_poly.rs` |
| Define prover and verifier instance contracts | `akita-sumcheck/src/traits.rs` |
| Encode proofs and verify ordinary rounds | `akita-sumcheck/src/types.rs` |
| Batch claims and right-align shorter instances | `akita-sumcheck/src/batched_sumcheck.rs` |

For a review, check these facts together:

1. Table indexing and challenge order agree.
2. Every round polynomial respects the declared degree bound.
3. The transcript absorbs the claim and round message before sampling the
   challenge that depends on it.
4. Proof decoding derives the round count and message width from trusted
   configuration.
5. The final output claim is compared with the expected oracle evaluation.
6. Batched instances use the same coefficients, offsets, and challenge slices
   on the prover and verifier paths.

The next chapter explains the specialized form used when every summand contains
a known equality polynomial.
