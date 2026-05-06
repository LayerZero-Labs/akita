# Spec: Tensor Exact Aggregate Evaluator

| Field       | Value                                      |
|-------------|--------------------------------------------|
| Author(s)   | TBD                                        |
| Created     | 2026-05-06                                 |
| Status      | proposed                                   |
| Scope       | Tensor folding challenges, verifier M-eval |

## Summary

Tensor-structured folding challenges reduce the number of Fiat-Shamir challenge
vectors from `2^r` to roughly `2 * 2^(r/2)`. The Lattice Jolt fourth-root
verifier draft states that after ring switching the challenge contribution
factorizes as `c_alpha(p || q) = c_alpha^L(p) * c_alpha^R(q)`. That statement
would be true if evaluation at the ring-switch point were a ring homomorphism
from `R_q = Z_q[X] / (X^D + 1)` to the verifier field. Hachi's ring switch,
however, evaluates at a generic scalar `alpha`, so this factorization is missing
the quotient correction from negacyclic reduction.

This spec proposes an exact aggregate evaluator that preserves the current
ring-switch soundness while recovering the verifier-time benefit of tensor
challenges. Instead of materializing all reduced products
`reduce(L_p * R_q)`, the verifier evaluates weighted aggregates exactly using a
product term plus an explicit `(alpha^D + 1)` correction term.

## Context

At each fold level, Hachi samples one sparse ring challenge per logical root
block:

```text
c_i in C, for i in {0,1}^r
```

The prover folds decomposed block witnesses with these challenges:

```text
z = sum_i c_i * s_i
```

The verifier later performs a ring switch. The root relation holds in:

```text
R_q = Z_q[X] / (X^D + 1)
```

To check it over a field, the protocol lifts the relation back to `Z_q[X]`,
adds quotient rows, and evaluates at a random field point `alpha`:

```text
M * w = h + (X^D + 1) * r
```

After evaluating at `X = alpha`, the quotient contribution appears as:

```text
(alpha^D + 1) * r(alpha)
```

This is intentional. The ring-switch challenge must be a generic random field
point so that the quotient rows are visible to the verifier with high
probability.

## Tensor Challenges

The fourth-root verifier direction splits the block index:

```text
i = p || q
p in {0,1}^{r_left}
q in {0,1}^{r_right}
```

and samples two shorter challenge vectors:

```text
L_p in C
R_q in C
```

The logical block challenge is the ring product:

```text
C_{p,q} = reduce(L_p * R_q) in R_q
```

The paper draft describes the expected verifier win as:

```text
C_{p,q}(alpha) = L_p(alpha) * R_q(alpha)
```

If true, the verifier could evaluate only the two base vectors and multiply
their scalar images lazily.

## The Issue

The factorization above is not valid for Hachi's current ring-switch evaluation
point. Evaluation of coefficient representatives at a generic scalar is not a
ring homomorphism out of `R_q`.

Minimal counterexample:

```text
D = 2
L = X
R = X

In R_q = Z_q[X] / (X^2 + 1):
reduce(L * R) = -1

eval(reduce(L * R), alpha) = -1
eval(L, alpha) * eval(R, alpha) = alpha^2
```

These are equal only when:

```text
alpha^2 = -1
```

More generally, write the unreduced product as:

```text
L_p(X) * R_q(X) = C_{p,q}(X) + (X^D + 1) * Q_{p,q}(X)
```

where:

```text
C_{p,q} = reduce(L_p * R_q)
Q_{p,q} = high-half quotient of L_p * R_q
```

Evaluating at a generic `alpha` gives the exact identity:

```text
C_{p,q}(alpha)
  = L_p(alpha) * R_q(alpha)
    - (alpha^D + 1) * Q_{p,q}(alpha)
```

The missing term is the source of the mismatch.

## How We Found It

During implementation of tensor challenge plumbing, a test compared:

```text
lazy[p,q] = L_p(alpha) * R_q(alpha)
```

against:

```text
expanded[p,q] = eval(reduce(L_p * R_q), alpha)
```

for transcript-derived sparse challenges. The test failed. The failure was not
a sampling issue or an implementation bug in negacyclic multiplication; it was
the algebra above. The current Akita ring evaluation helper evaluates the
degree-`< D` coefficient representative:

```text
eval_ring_at(r, alpha) = sum_{k=0}^{D-1} r_k * alpha^k
```

and `CyclotomicRing` multiplication uses negacyclic reduction with `X^D = -1`.
Those two operations commute only when `alpha^D = -1`, which Hachi must not
require for ring-switch soundness.

## Goals

- Preserve Hachi's current ring-switch soundness model.
- Keep `alpha` a generic random verifier-field element.
- Avoid materializing all `2^r` reduced tensor products in verifier hot paths.
- Evaluate the same verifier polynomial as the expanded tensor-product path.
- Maintain at least 128-bit concrete security for every enabled production
  configuration.

## Non-Goals

- Do not change the ring-switch challenge distribution to roots of
  `X^D + 1`.
- Do not remove quotient rows from the ring-switch relation.
- Do not claim the paper's simplified factorization is valid in the current
  Hachi ring-switch model.
- Do not implement claim-reduction sumcheck in this spec; this covers only the
  tensor-challenge evaluator needed by verifier M-eval.

## Proposed Solution

Use an exact aggregate evaluator.

The verifier often does not need every scalar `C_{p,q}(alpha)` independently.
It needs weighted sums of those scalars, for example in block summaries used by
deferred M-table evaluation:

```text
S = sum_{p,q} weight[p,q] * C_{p,q}(alpha)
```

When the weights factor as:

```text
weight[p,q] = u[p] * v[q]
```

the exact aggregate is:

```text
S
= (sum_p u[p] * L_p(alpha))
  * (sum_q v[q] * R_q(alpha))
  - (alpha^D + 1)
    * sum_t alpha^t
      * sum_{i+j=D+t}
        (sum_p u[p] * L_p[i])
        (sum_q v[q] * R_q[j])
```

Define aggregated coefficient vectors:

```text
Lbar[i] = sum_p u[p] * L_p[i]
Rbar[j] = sum_q v[q] * R_q[j]
```

Then:

```text
product_eval = eval(Lbar, alpha) * eval(Rbar, alpha)

quotient_eval =
  sum_{t=0}^{D-2} alpha^t * sum_{i+j=D+t} Lbar[i] * Rbar[j]

S = product_eval - (alpha^D + 1) * quotient_eval
```

This is exactly:

```text
sum_{p,q} u[p] * v[q] * eval(reduce(L_p * R_q), alpha)
```

without expanding every `(p,q)` challenge.

## Factored Weight Decomposition

The main verifier challenge-dependent hot spot is currently built around
block-carry summaries:

```text
summarize_pow2_block_carries(eq_low, offset_low, c_alphas)
```

For tensor challenges, the logical block index is:

```text
block = p * right_len + q
```

Let:

```text
offset = offset_left * right_len + offset_right
```

Then:

```text
offset + block
= (offset_left + p + carry_q) * right_len + low_q
```

where:

```text
low_q  = (offset_right + q) mod right_len
carry_q = floor((offset_right + q) / right_len)
```

The final carry is:

```text
carry = floor((offset_left + p + carry_q) / left_len)
```

The multilinear equality weight over the low block index factorizes as:

```text
eq_low(low_p, low_q) = eq_left(low_p) * eq_right(low_q)
```

For each fixed `carry_q` and final `carry`, the weight decomposes into one
factored term:

```text
u[p] = eq_left((offset_left + p + carry_q) mod left_len)
       restricted to final carry

v[q] = eq_right((offset_right + q) mod right_len)
       restricted to carry_q
```

Because `carry_q` and final `carry` are each binary, each block summary can be
expressed as a small constant number of factored terms. This is the structural
reason an exact aggregate evaluator can remain fourth-root sized.

## Algorithm Sketch

Add a tensor-aware M-eval preparation mode:

```text
PreparedChallengeEvals:
  Flat(Vec<F>)
  Tensor {
    left: Vec<SparseChallenge>,
    right: Vec<SparseChallenge>,
    left_len: usize,
    right_len: usize,
    num_claims: usize,
  }
```

Replace full `c_alphas` expansion in verifier preparation with tensor-aware
summary helpers:

```text
tensor_weighted_challenge_eval(
  left_challenges,
  right_challenges,
  u_weights,
  v_weights,
  alpha_pows,
  alpha_pow_d_plus_one,
) -> F
```

Implementation steps:

1. Accumulate sparse base challenges into dense aggregate coefficient vectors:

   ```text
   Lbar[0..D] = 0
   Rbar[0..D] = 0
   for p:
     Lbar += u[p] * L_p
   for q:
     Rbar += v[q] * R_q
   ```

2. Compute:

   ```text
   product_eval = eval(Lbar, alpha) * eval(Rbar, alpha)
   ```

3. Compute the high-half quotient evaluation:

   ```text
   quotient_eval = 0
   for i in 0..D:
     for j in max(D - i, 0)..D:
       t = i + j - D
       quotient_eval += Lbar[i] * Rbar[j] * alpha^t
   ```

4. Return:

   ```text
   product_eval - (alpha^D + 1) * quotient_eval
   ```

5. Sum the small number of factored terms produced by the carry decomposition.

## Complexity

Let:

```text
B = 2^r
L = 2^r_left
R = 2^r_right
omega = max challenge L1 mass
D = ring dimension
T = number of factored terms in one block summary
```

Baseline flat verifier challenge evaluation:

```text
O(B * omega)
```

Current conservative tensor implementation:

```text
O(B * omega^2)
```

It samples fewer base challenges, but still expands and evaluates all logical
products.

Paper's simplified tensor expectation:

```text
O((L + R) * omega)
```

This omits the quotient correction and is not exact for generic ring-switch
`alpha`.

Proposed exact aggregate evaluator:

```text
O(T * ((L + R) * omega + D^2))
```

For current production dimensions `D <= 128`, the `D^2` term is small compared
with large block counts. For balanced tensor halves:

```text
L + R ~= 2 * sqrt(B)
```

so the asymptotic challenge-dependent verifier cost becomes fourth-root sized
up to the constant `D^2` correction per factored aggregate.

Memory:

```text
O((L + R) * sparse_challenge_size + D)
```

The verifier no longer needs `O(B)` scalar `c_alphas` in tensor mode.

## Comparison With The Paper

The paper's tensor-challenge section says the challenge contribution
factorizes after ring switching:

```text
c_alpha(p || q) = c_alpha^L(p) * c_alpha^R(q)
```

That statement is correct only if the evaluation map respects the quotient
relation `X^D = -1`, i.e. only if:

```text
alpha^D + 1 = 0
```

Hachi's ring-switch protocol deliberately samples `alpha` generically and
checks the quotient rows through the term:

```text
(alpha^D + 1) * r(alpha)
```

Therefore, the exact Hachi-compatible tensor identity is:

```text
c_alpha(p || q)
= c_alpha^L(p) * c_alpha^R(q)
  - (alpha^D + 1) * q_alpha(p || q)
```

where `q_alpha(p || q)` evaluates the high-half product quotient.

The proposed evaluator preserves the paper's intended tensor savings for the
first term and adds the missing correction in aggregated form. This keeps the
protocol semantics identical to reduced ring products in `R_q`, rather than
changing the ring-switch challenge distribution or the checked relation.

## Security Analysis

### Completeness

Completeness is preserved because the evaluator is an algebraic rewrite of the
same scalar the expanded verifier would compute.

For every pair `(p,q)`:

```text
L_p * R_q = reduce(L_p * R_q) + (X^D + 1) * Q_pq
```

Evaluating both sides at `alpha` gives:

```text
eval(reduce(L_p * R_q), alpha)
= eval(L_p, alpha) * eval(R_q, alpha)
  - (alpha^D + 1) * eval(Q_pq, alpha)
```

Linearity then gives the aggregate formula for any public weights. The verifier
therefore accepts every honest proof accepted by the expanded tensor verifier.

### Soundness

Soundness is preserved because the verifier checks exactly the same polynomial
identity as the expanded implementation.

The proposed evaluator:

- does not alter the Fiat-Shamir transcript;
- does not alter the challenge distribution;
- does not alter the prover's folded witness definition;
- does not alter the root relation;
- does not alter the ring-switch random point distribution;
- does not remove quotient rows;
- does not introduce prover-supplied correction terms.

All correction terms are deterministic functions of public verifier data:

```text
left/right challenges, alpha, public MLE weights
```

No additional witness data is trusted. A malicious prover cannot choose the
correction term independently from the tensor challenges.

The ring-switch soundness proof remains the existing Hachi proof. In
particular, `alpha` remains generic, so the quotient term

```text
(alpha^D + 1) * r(alpha)
```

continues to catch invalid lifts except with the standard `2D / |F|` style
knowledge-error contribution.

### Tensor CWSS Soundness

Tensor challenges change root-layer extraction, not the ring-switch evaluator.
The extractor uses the two-level CWSS tree described in the fourth-root
verifier draft. For each block `(p,q)`, it compares four accepting transcripts:

```text
(L,  R)
(L', R)
(L,  R')
(L', R')
```

The mixed second difference isolates:

```text
(L'_p - L_p) * (R'_q - R_q) * s_{p,q}
```

This denominator is a product of two short ring differences. If:

```text
omega = max_{c in C} ||c||_1
```

then:

```text
||(L'_p - L_p) * (R'_q - R_q)||_1 <= (2 omega)^2
```

The paper's norm analysis reports a relative MSIS norm ratio of `4 * omega`
for the challenge-dependent rows. This spec does not change that extraction
argument. It only changes how the verifier computes the already-defined
ring-switched table.

### 128-Bit Security Requirement

The exact aggregate evaluator adds no new hardness assumption and no new
knowledge-error term. The concrete 128-bit security requirement is therefore:

1. The tensor challenge family must have enough entropy for the two-level CWSS
   knowledge error:

   ```text
   epsilon_tensor = 4 * 2^(r/2) / |C|
   ```

   or the corresponding unbalanced split expression.

2. Every enabled `(D, challenge_family, schedule)` combination must retain at
   least 128-bit Module-SIS security after applying the tensor extraction norm
   increase.

3. The ring-switch field must retain the existing soundness margin:

   ```text
   2D / |F|
   ```

   plus the usual sumcheck knowledge-error terms.

The implementation must not enable tensor mode by default for a production
configuration until the planner/security tables have been audited with the
effective tensor bounds. A conservative implementation should:

- use `Stage1ChallengeShape::Flat` by default;
- require an explicit config flag for tensor mode;
- compute fold digit depths with the effective tensor challenge mass;
- validate that generated or runtime SIS floors still exceed 128-bit security.

The paper draft gives representative margins, for example a `d = 64` family
with `omega = 54`, where the tensor norm ratio is `216` (`~7.8` bits). If the
baseline SIS floor is `280+` bits, this remains comfortably above 128 bits.
This must still be checked against the exact production schedules and challenge
families used in this repository.

### Why Not Set `alpha^D = -1`?

Choosing `alpha` from the roots of `X^D + 1` would make:

```text
alpha^D + 1 = 0
```

and therefore remove the correction term. This is not acceptable in the
current Hachi ring-switch protocol because it also hides the quotient rows:

```text
(X^D + 1) * r
```

would evaluate to zero. The verifier would no longer check that the lifted
relation is valid over `Z_q[X]`, undermining the existing ring-switch
soundness argument.

### Why Not Restrict Challenge Supports?

Another possible workaround is to sample challenges whose degrees never wrap,
so:

```text
deg(L_p) + deg(R_q) < D
```

and the quotient `Q_pq` is always zero. This would make the simplified
factorization true. It is not the preferred path because it changes the
challenge family, likely reduces entropy or alters invertibility properties,
and requires a fresh challenge-family security analysis. The exact aggregate
evaluator preserves the existing ring product definition instead.

## Testing Strategy

Required unit tests:

- Tensor product expansion equals dense negacyclic multiplication.
- For random sparse `L_p`, `R_q`, and `alpha`, the exact formula equals
  `eval(reduce(L_p * R_q), alpha)`.
- Weighted aggregate evaluator equals explicit expansion for random factored
  weights `u[p] * v[q]`.
- Carry-decomposed factored summaries equal current
  `summarize_pow2_block_carries` on expanded tensor challenges.
- `alpha^D + 1 = 0` edge case reduces to the simple product formula in a local
  algebra test, without changing production sampling.

Required integration tests:

- `prepare_m_eval` tensor mode equals expanded tensor mode on small random
  layouts.
- Prover and verifier derive identical tensor challenge vectors.
- E2E prove/verify with tensor disabled.
- E2E prove/verify with tensor enabled on small D64/D128 configurations.
- Tampering a tensor challenge label, split, or coordinate changes verifier
  replay and causes rejection.

Required security checks:

- Planner/SIS audit for every tensor-enabled preset.
- Tests or checked assertions that tensor mode uses effective challenge mass in
  fold digit depth calculations.
- Regression check that tensor mode is not enabled by default without audited
  parameters.

## Benchmark Plan

Benchmarks should separate the following components:

- sparse challenge sampling;
- base tensor challenge evaluation;
- exact aggregate correction evaluation;
- full `prepare_m_eval`;
- end-to-end verifier replay.

Compare four modes:

```text
flat baseline
tensor expanded exact evaluator
tensor exact aggregate evaluator
paper-style product-only evaluator (test-only, expected unsound for generic alpha)
```

The product-only evaluator is useful only as a diagnostic upper bound on the
possible speedup. It must not be used in production verification.

Success criteria:

- Exact aggregate results match expanded exact results bit-for-bit.
- Tensor aggregate `prepare_m_eval` reduces challenge-dependent verifier work
  from `O(2^r)` toward `O(2^(r/2) + D^2)`.
- End-to-end verifier improvements are reported separately from claim-reduction
  improvements, since claim reduction targets a different setup-dependent
  bottleneck.

## Implementation Plan

1. Keep the current tensor challenge transcript and exact expanded fallback.
2. Add a verifier-only `TensorChallengeEvaluator` abstraction in
   `akita-challenges` or `akita-verifier`.
3. Add exact aggregate evaluation for one factored term.
4. Add tensor carry decomposition for block summaries.
5. Replace tensor-mode `c_alphas` expansion in `PreparedMEval` with the
   aggregate evaluator.
6. Keep a debug/test path that expands tensor products and compares results.
7. Add focused benchmarks before enabling tensor mode in any default config.
8. Run planner/SIS audits before production enablement.

## Open Questions

- Should the exact aggregate evaluator live in `akita-challenges` because it is
  challenge-specific, or in `akita-verifier` because it depends on M-eval
  weights and block-carry decomposition?
- How many factored terms are needed for every current `PreparedMEval` use site
  after accounting for offset carries?
- Can the high-half quotient correction use small sparse convolution instead of
  dense `D^2` once aggregate coefficient vectors remain sparse?
- Should generated schedules encode tensor enablement explicitly, or should
  tensor remain a runtime-only experimental flag until security review is
  complete?

## References

- `sections/5_fourth_root_verifier.tex` in the Lattice Jolt draft, especially
  "Tensor-structured folding challenges".
- `sections/3_batched_hachi.tex` in the Lattice Jolt draft, especially the
  Hachi ring-switch lift.
- `specs/fourth-root-verifier-optimizations.md`.
- `crates/akita-algebra/src/ring/eval.rs`.
- `crates/akita-algebra/src/ring/cyclotomic.rs`.
