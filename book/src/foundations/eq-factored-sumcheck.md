# Equality-factored sum-check

Many Akita identities give extra weight to one point $\tau$ through the
equality polynomial:

$$
T=\sum_{x\in\{0,1\}^n}\operatorname{eq}(\tau,x)p(x).
$$

Ordinary sum-check can prove this claim. Equality-factored sum-check proves the
same claim with smaller round messages and a smaller prover table. The saving
comes from treating the known equality factor separately from the polynomial
$p$ that carries the protocol data.

Read [Multilinear extensions and sum-check](./multilinear-sumcheck.md) first if
the ordinary round check is not yet familiar.

## The equality factor splits by coordinate

For one coordinate, define

$$
\operatorname{eq}(\tau_i,X)
=(1-\tau_i)(1-X)+\tau_iX.
$$

This is a linear polynomial in $X$. Across all coordinates,

$$
\operatorname{eq}(\tau,x)
=\prod_{i=0}^{n-1}\operatorname{eq}(\tau_i,x_i).
$$

After the first $j$ variables have been bound to challenges
$r_0,\ldots,r_{j-1}$, their equality contribution is one scalar:

$$
\gamma_j
=\prod_{i=0}^{j-1}\operatorname{eq}(\tau_i,r_i).
$$

The current coordinate contributes the linear factor

$$
l_j(X)=\gamma_j\operatorname{eq}(\tau_j,X).
$$

The remaining coordinates stay inside the Boolean sum. The full round
polynomial therefore factors as

$$
s_j(X)=l_j(X)q_j(X),
$$

where $q_j$ is the round polynomial formed from $p$ and the equality factors
for the still unbound coordinates.

## Send the smaller polynomial

Suppose $q_j$ has degree at most $d$. The full polynomial $s_j=l_jq_j$ can
have degree $d+1$ because $l_j$ is linear. The verifier already knows $l_j$,
so Akita sends coefficients of $q_j$ instead of coefficients of $s_j$.

There is one more saving. Write

$$
q_j(X)=q_0+q_1X+q_2X^2+\cdots+q_dX^d
$$

and let

$$
l_0=l_j(0),\qquad l_1=l_j(1).
$$

The current claim obeys

$$
T_j=s_j(0)+s_j(1)=l_0q_0+l_1(q_0+q_1+q_2+\cdots+q_d).
$$

If division by $l_1$ were used, this equation would recover $q_1$. Akita does
not send that coefficient. An equality-factored round message stores

```text
[q0, q2, q3, ..., qd]
```

For the common case where $p$ contributes degree two, ordinary sum-check would
send three coefficients for the degree-three product $s_j$. The factored
message sends two coefficients for the degree-two inner polynomial $q_j$.

`EqFactoredUniPoly` in `akita-sumcheck/src/types.rs` is this exact wire type.

## The verifier avoids division

Directly recovering $q_1$ would require an inverse of $l_1$. It would also need
a special case when $l_1=0$. Akita carries a scaled claim instead.

Let $P_j$ be the current scale and let

$$
C_j=P_jT_j.
$$

Initially $P_0=1$ and $C_0=T$. Let

$$
h=q_2+q_3+\cdots+q_d.
$$

The verifier can compute

$$
U=C_j-P_j(l_0+l_1)q_0-P_jl_1h.
$$

Using the round identity above, this simplifies to

$$
U=P_jl_1q_1.
$$

After challenge $r_j$ is sampled, the known part of the inner polynomial is

$$
q_{\mathrm{known}}(r_j)
=q_0+q_2r_j^2+\cdots+q_dr_j^d.
$$

The verifier updates

$$
\begin{aligned}
P_{j+1}&=P_jl_1,\\
C_{j+1}
&=P_{j+1}l_j(r_j)q_{\mathrm{known}}(r_j)
+l_j(r_j)r_jU.
\end{aligned}
$$

Substituting $U=P_jl_1q_1$ shows that

$$
C_{j+1}=P_{j+1}l_j(r_j)q_j(r_j).
$$

This is the next scaled claim. No inversion appears. The final verifier check
compares $C_n$ with $P_n$ times the expected folded oracle value.

The function `advance_eq_factored_claim` in
`akita-sumcheck/src/drivers/eq_factored.rs` implements these equations directly.
Both prover and verifier call that one function, so the transcript replay and
the generated proof cannot drift onto different update rules.

## Avoiding a full equality table

A direct prover can materialize all
$2^n$ values $\operatorname{eq}(\tau,x)$ and fold that table in every round.
Akita uses `GruenSplitEq` instead.

It divides the unbound coordinates, other than the current one, into two
halves. For each half it caches equality tables for its remaining suffixes.
At a round, the equality weight for table index $k$ is reconstructed as

$$
E_{\mathrm{first}}[k_{\mathrm{low}}]
\cdot E_{\mathrm{second}}[k_{\mathrm{high}}].
$$

The current coordinate is represented by $l_j(X)$, and coordinates already
bound are represented by the scalar $\gamma_j$. Binding a challenge multiplies
that scalar by $\operatorname{eq}(\tau_j,r_j)$ and pops one cached table level.

Each half has about $n/2$ coordinates. The largest cached table therefore has
about $2^{n/2}$ entries rather than $2^n$. Including all smaller cached levels
changes only the constant factor. This is a substantial memory reduction for
large sum-checks.

`GruenSplitEq` lives in `akita-algebra/src/split_eq.rs`. Its tests compare every
reconstructed weight and every fold against a fully materialized equality
table.

## When this form applies

The optimization applies when the verifier knows a common equality factor for
the sum-check instance. It is especially useful for claims built around one
verifier point $\tau$.

Instances that share the same factor can also share the specialized round
logic. A mixed batch with different equality factors cannot omit the same
inner coefficient under one common round equation. Such a batch uses ordinary
compressed sum-check messages instead. The prover may still use split equality
tables internally to save memory.

This distinction matters in review. The equality-table representation is a
prover implementation choice. Omitting the linear coefficient of $q_j$ changes
the proof message and requires the matching verifier equation.

## Where to inspect the implementation

| Rule | Implementation |
| --- | --- |
| Equality polynomial definition and tables | `akita-algebra/src/eq_poly.rs` |
| Split equality state and suffix caches | `akita-algebra/src/split_eq.rs` |
| Factored prover and verifier interfaces | `akita-sumcheck/src/traits.rs` |
| Factored round message and proof encoding | `akita-sumcheck/src/types.rs` |
| Inversion-free claim update and transcript driver | `akita-sumcheck/src/drivers/eq_factored.rs` |

For a review, check these facts together:

1. The challenge order used by $\tau$, the data table, and `GruenSplitEq`
   agrees.
2. The declared degree bounds apply to the inner polynomial $q_j$, not the
   product $l_jq_j$.
3. The proof shape fixes the number of rounds and stored coefficients before
   decoding.
4. Prover and verifier use the same $l_j(0)$, $l_j(1)$, scale, and challenge in
   every update.
5. The final comparison includes the accumulated claim scale.
6. Tests compare the split representation with the full equality table, not
   only with another optimized path.

Equality factoring does not change what the protocol proves. It exposes a
factor the verifier already knows, then removes work and proof data that would
otherwise repeat that structure.
