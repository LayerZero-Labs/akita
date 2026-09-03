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

The corresponding full round polynomial has the linear factor

$$
l_j(X)=\gamma_j\operatorname{eq}(\tau_j,X).
$$

The remaining coordinates stay inside the Boolean sum. The full round
polynomial therefore factors as

$$
s_j(X)=l_j(X)q_j(X),
$$

where $q_j$ is the round polynomial formed from $p$ and the equality factors
for the still unbound coordinates. Equality-factored sum-check normalizes the
already-bound scalar $\gamma_j$ out of the running claim. Its round identity
therefore uses $\operatorname{eq}(\tau_j,X)$ directly rather than carrying
$l_j(X)$ from one round to the next.

## Omit the constant coefficient

Suppose $q_j$ has degree at most $d$. The full polynomial $s_j=l_jq_j$ can
have degree $d+1$ because $l_j$ is linear. The verifier already knows $l_j$,
so Akita sends coefficients of $q_j$ instead of coefficients of $s_j$.

There is one more saving. Write

$$
q_j(X)=q_0+q_1X+q_2X^2+\cdots+q_dX^d
$$

The normalized current claim obeys

$$
T_j=(1-\tau_j)q_j(0)+\tau_jq_j(1)
   =q_0+\tau_j(q_1+q_2+\cdots+q_d).
$$

The coefficient of $q_0$ is exactly one, independently of $\tau_j$. Akita does
not send that coefficient. An equality-factored round message stores

```text
[q1, q2, ..., qd]
```

For the common case where $p$ contributes degree two, ordinary sum-check would
send three coefficients for the degree-three product $s_j$. The factored
message sends two coefficients for the degree-two inner polynomial $q_j$.

`EqFactoredUniPoly` in `akita-sumcheck/src/types.rs` is this exact wire type.
Changing which coefficient is omitted does not change the message width: a
degree-$d$ inner polynomial still contributes exactly $d$ field elements.

## The verifier avoids division

The verifier recovers the omitted coefficient by subtraction:

$$
q_0=T_j-\tau_j(q_1+q_2+\cdots+q_d).
$$

After sampling $r_j$, it advances the normalized claim as

$$
T_{j+1}=q_j(r_j)=q_0+q_1r_j+q_2r_j^2+\cdots+q_dr_j^d.
$$

No inverse or accumulated claim scale appears, and the formula remains valid
when $\tau_j=0$ or when an earlier equality evaluation vanishes. The final
verifier check compares $T_n$ directly with the expected folded oracle value.

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

The specialized driver reads the current coordinate $\tau_j$ and the remaining
tables, then pops one cached table level after binding a challenge.
`GruenSplitEq` also tracks $\gamma_j$ because other, ordinary sum-check paths
use it to reconstruct the full polynomial $l_j(X)q_j(X)$; the normalized
equality-factored verifier does not multiply its claim by this scalar.

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
prover implementation choice. Omitting the constant coefficient of $q_j$ changes
the proof message and requires the matching verifier equation.

## Where to inspect the implementation

| Rule | Implementation |
| --- | --- |
| Equality polynomial definition and tables | `akita-algebra/src/eq_poly.rs` |
| Split equality state and suffix caches | `akita-algebra/src/split_eq.rs` |
| Factored prover and verifier interfaces | `akita-sumcheck/src/traits.rs` |
| Factored round message and proof encoding | `akita-sumcheck/src/types.rs` |
| Normalized claim update and transcript driver | `akita-sumcheck/src/drivers/eq_factored.rs` |

For a review, check these facts together:

1. The challenge order used by $\tau$, the data table, and `GruenSplitEq`
   agrees.
2. The declared degree bounds apply to the inner polynomial $q_j$, not the
   product $l_jq_j$.
3. The proof shape fixes the number of rounds and stored coefficients before
   decoding.
4. Prover and verifier use the same $\tau_j$ and challenge in every update.
5. The final comparison uses the normalized oracle value, without an
   accumulated equality scale.
6. Tests compare the split representation with the full equality table, not
   only with another optimized path.

Equality factoring does not change what the protocol proves. It exposes a
factor the verifier already knows, then removes work and proof data that would
otherwise repeat that structure.
