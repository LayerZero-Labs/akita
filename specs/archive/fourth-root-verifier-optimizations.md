# Spec: Fourth-Root Verifier Optimizations

| Field     | Value                                         |
|-----------|-----------------------------------------------|
| Status    | proposal                                      |
| Scope     | tensor folding challenges, claim reduction    |
| Goal      | reduce verifier work while preserving Hachi soundness |

## Summary

This spec sketches two protocol-level optimizations from the Lattice Jolt
fourth-root verifier direction:

1. **Tensor-structured folding challenges**: replace the flat vector of root
   folding challenges with a Kronecker/tensor product of two shorter challenge
   vectors.
2. **Claim-reduction sumcheck**: avoid materializing the setup-dependent
   `M_alpha` verifier table directly by reducing it to a short sumcheck and a
   single opening of the shared setup matrix polynomial.

They target different verifier bottlenecks and can be implemented
independently, although the full fourth-root verifier story uses both.

Neither optimization changes the underlying Ajtai/Module-SIS assumption. Both
require updated transcripts, proof objects, verifier replay, and new soundness
tests.

## Current Baseline

At each Hachi fold level, the verifier does two expensive things:

- Derive and evaluate one sparse ring challenge per root block:

```text
c_0, ..., c_{B-1} in C
c_i(alpha) = eval_ring(c_i, alpha)
```

- Materialize and evaluate the row-batched ring-switched matrix table:

```text
M_alpha(i, x) = eval_ring(M(i, x), alpha)
m_tau1(x) = sum_i eq(tau1, i) * M_alpha(i, x)
```

For a normal split where `B = 2^r ~= sqrt(N')`, both costs scale with the root
block count. The setup-table side is usually worse because it has an additional
row-family factor (`A`, `B`, `D`, consistency, public rows, etc.).

## Optimization 1: Tensor-Structured Folding Challenges

### Idea

Instead of sampling a flat challenge vector:

```text
c_i in C, for i in {0,1}^r
```

split the block index:

```text
i = p || q
p in {0,1}^{r_left}
q in {0,1}^{r_right}
```

and sample two shorter vectors:

```text
alpha_p in C
beta_q  in C
```

Then define:

```text
c_{p,q} = alpha_p * beta_q
```

The verifier now derives and evaluates:

```text
2^r_left + 2^r_right
```

base challenges instead of:

```text
2^r
```

For balanced halves, this is roughly:

```text
2 * 2^(r/2)
```

instead of:

```text
2^r
```

After ring evaluation:

```text
c_{p,q}(alpha) = alpha_p(alpha) * beta_q(alpha)
```

so the challenge contribution factorizes across the two block dimensions.

### Transcript Shape

The interactive shape should be modeled as:

```text
P -> V: root opening commitment v
V -> P: alpha-vector in C^{2^r_left}
P -> V: empty message
V -> P: beta-vector in C^{2^r_right}
P -> V: root witnesses / recursive commitment
```

In Fiat-Shamir form, the empty message is not serialized as a proof object, but
the transcript separation must make the two challenge rounds distinct:

```text
alpha_vec = H(prefix, "tensor/left")
beta_vec  = H(prefix, alpha_vec, "tensor/right")
```

Do not sample both vectors from one undifferentiated label. The proof needs the
2-level challenge structure.

### Prover Changes

The root fold becomes:

```text
z = sum_{p,q} alpha_p * beta_q * s_{p,q}
```

A practical prover should compute this in two stages:

```text
tmp_p = sum_q beta_q * s_{p,q}
z     = sum_p alpha_p * tmp_p
```

This can reduce challenge multiplication work and may compose well with the
rotated-table accumulation kernels for dense-ish challenge families.

### Verifier Changes

The verifier should avoid expanding `c_{p,q}(alpha)` unless needed. Instead,
carry separate evaluated vectors:

```text
alpha_eval[p] = alpha_p(alpha)
beta_eval[q]  = beta_q(alpha)
```

and evaluate tensor products lazily:

```text
c_eval[p,q] = alpha_eval[p] * beta_eval[q]
```

The root relation and `M_alpha` algebraic rows need tensor-aware indexing.

### Soundness Sketch

The existing flat CWSS extraction varies one coordinate of the flat challenge
vector and isolates one block.

That is not enough for tensor challenges. Varying only `alpha_p` changes every
block `(p, q)` in that row at once:

```text
sum_q beta_q * s_{p,q}
```

So extraction must use a 2-level CWSS tree. For each block `(p, q)`, compare
four accepting transcripts:

```text
(alpha,  beta)
(alpha', beta)
(alpha,  beta')
(alpha', beta')
```

The mixed second difference isolates one block:

```text
(z(alpha', beta') - z(alpha', beta))
- (z(alpha, beta') - z(alpha, beta))
= (alpha'_p - alpha_p) * (beta'_q - beta_q) * s_{p,q}
```

The denominator is a product of two challenge differences. Therefore the MSIS
norm bound worsens by roughly a factor proportional to `omega`, where:

```text
omega = max_{c in C} ||c||_1
```

This is why challenge-family tuning matters: lower `omega` directly reduces the
tensor-challenge soundness cost.

### Expected Impact

Main win: verifier time.

Challenge evaluation changes from:

```text
O(2^r * D)
```

to roughly:

```text
O((2^r_left + 2^r_right) * D)
```

For balanced halves:

```text
O(2^(r/2) * D)
```

Proof size impact should be small or neutral unless the proof object needs to
carry extra shape metadata. The transcript-derived challenge vectors are not
sent.

### Implementation Plan

1. Add a schedule/layout flag for tensor folding at a root/fold level.
2. Add transcript labels for left and right tensor challenge vectors.
3. Add a tensor challenge sampler API that returns `(left, right)` vectors.
4. Update root quadratic-equation construction to compute tensor-folded `z`.
5. Update prover `M`-table construction to use tensor-indexed challenge rows.
6. Update verifier replay to evaluate tensor challenge vectors and use lazy
   products.
7. Add proof shape metadata if required by deserialization/verifier replay.
8. Gate by config first; keep the existing flat path as the default until
   benchmarks and soundness review are done.

### Tests

- Prover and verifier derive identical left/right challenge vectors.
- Tensor fold equals flat fold when the flat challenges are explicitly expanded.
- Root relation passes for tensor and flat-equivalent paths.
- Tampering a tensor challenge coordinate is detected.
- 2-level extractor algebra test for a small toy instance.
- E2E prove/verify for D32/D64/D128 configs with tensor disabled and enabled.
- Bench verifier challenge evaluation before/after.

### Open Questions

- Which levels should use tensor challenges? Root only may capture most verifier
  benefit with less protocol churn.
- What split should be used for odd `r`? Likely `floor(r/2)` and `ceil(r/2)`.
- Does tensoring interact well with batched/multipoint root shapes?
- How much additional SIS margin is consumed for each production `D` family?

## Optimization 2: Claim-Reduction Sumcheck

### Idea

The current verifier materializes the setup-dependent part of the
ring-switched matrix table:

```text
M_alpha(i, x) = eval_ring(M(i, x), alpha)
```

and then evaluates:

```text
m_tau1(r_x) = sum_i eq(tau1, i) * M_alpha(i, r_x)
```

This is expensive because `M` includes row/column prefixes of the public setup
matrices `A`, `B`, and `D`.

The claim-reduction optimization splits:

```text
m_tau1(x) = m_alg(x) + m_setup(x)
```

where:

- `m_alg` is cheap verifier-computable structure: challenge rows, opening point
  weights, gadget scalars, public rows, and consistency rows.
- `m_setup` is the part contributed by the setup matrix coefficients.

Instead of directly evaluating `m_setup(r_x)`, the verifier reduces that claim
with a short sumcheck over setup-matrix variables.

### Baseline Stage Flow

Current recursion has:

1. Stage 1: range-check sumcheck.
2. Stage 2: fused relation/range continuation sumcheck.

The verifier needs the final relation value involving:

```text
w_eval * alpha_eval * m_tau1(r_x)
```

and therefore has to compute `m_tau1(r_x)`.

### New Stage Flow

Run a batched witness-domain sumcheck first:

```text
range claim:
  0 = sum_z eq(tau0, z) * Q(w(z))

relation claim:
  V_alpha = sum_{x,y} w(x,y) * alpha_weight(y) * m_tau1(x)
```

At the end, the verifier learns:

```text
w_eval = w(r_x, r_y)
alpha_eval = alpha_weight(r_y)
lambda = w_eval * alpha_eval
```

The pending setup-side claim is:

```text
lambda * m_tau1(r_x)
```

Subtract the cheap algebraic part:

```text
lambda * m_setup(r_x)
= lambda * (m_tau1(r_x) - m_alg(r_x))
```

Then prove this setup-side claim with a short claim-reduction sumcheck.

### Setup Matrix Polynomial

Model the shared setup matrix as a multilinear polynomial:

```text
S(row, col, coeff)
```

where:

- `row` indexes rows of the shared setup envelope,
- `col` indexes columns,
- `coeff` indexes ring coefficients `0..D-1`.

The setup-side contribution is a structured linear combination of evaluations
of `S`.

The claim-reduction sumcheck reduces:

```text
lambda * m_setup(r_x)
```

to one opening claim:

```text
lambda * S(r_row, r_col, r_coeff) = y_setup
```

This opening must be checked against a commitment to the setup matrix
polynomial.

### Avoiding Division by Zero

Do not divide by:

```text
lambda = w_eval * alpha_eval
```

It can be zero. Instead, carry the scaled claim:

```text
lambda * m_setup(r_x)
```

through the claim-reduction sumcheck. If `lambda = 0`, the claim is still
well-defined and the sumcheck remains sound.

### Prover Changes

The prover must:

1. Produce the usual recursive witness commitment.
2. Run the batched witness-domain sumcheck.
3. Build the setup-side claim polynomial for the selected level.
4. Run the setup claim-reduction sumcheck.
5. Provide/open the required setup polynomial commitment, or link it into an
   existing committed setup object.

### Verifier Changes

The verifier must:

1. Replay the batched witness-domain sumcheck.
2. Compute `m_alg(r_x)` cheaply.
3. Form the scaled setup-side claim.
4. Verify the claim-reduction sumcheck.
5. Verify the final opening of `S`.

### Soundness Sketch

The witness-domain sumcheck remains a standard batched sumcheck over the same
witness polynomial. The setup-side sumcheck is a standard claim reduction for a
public/preprocessed polynomial `S`.

Soundness relies on:

- the setup matrix commitment binding to one fixed `S`,
- the claim-reduction sumcheck soundness over the verifier field,
- preserving the scaled claim to avoid zero-division,
- keeping all transcript challenges ordered so the prover cannot choose witness
  messages after seeing reduction challenges.

The Module-SIS binding assumptions are unchanged. This optimization changes how
the verifier checks the setup-dependent term, not what algebraic relation is
checked.

### Expected Impact

Main win: verifier time.

The current verifier pays to evaluate setup matrix rows across many columns and
ring coefficients. Claim reduction replaces this with:

```text
O(log(row_count) + log(col_count) + log(D))
```

sumcheck rounds plus one setup polynomial opening.

Proof size may increase by the short setup-side sumcheck and setup opening. The
tradeoff is worthwhile only if the verifier-time reduction dominates the extra
proof bytes and prover work.

### Implementation Plan

1. Define a setup polynomial view `S(row, col, coeff)` over the shared setup
   envelope.
2. Add a commitment/opening mechanism for `S`.
   - Option A: precompute and commit to `S` in setup.
   - Option B: batch `S` into the next recursive opening.
3. Split current `M_alpha` construction into `m_alg` and `m_setup`.
4. Add a setup-side sumcheck prover/verifier.
5. Thread the scaled claim `lambda * m_setup` through transcript/proof objects.
6. Keep the old verifier path behind a config flag until benchmarks are stable.

### Tests

- `m_alg + m_setup` equals the current materialized `m_tau1` on random small
  instances.
- Claim-reduction final opening equals direct setup-table evaluation.
- `lambda = 0` case verifies correctly and does not divide.
- Tampering setup rows, setup opening, or claim-reduction messages fails.
- E2E prove/verify with old and new verifier paths.
- Benchmark verifier time and proof bytes across representative D32/D64/D128
  schedules.

### Open Questions

- How should the setup matrix polynomial be committed without bloating the next
  recursive witness?
- Can setup openings be batched across levels?
- Should this be root-only first, or all recursive levels?
- How does this interact with tensor challenges and batched/multipoint roots?

## Combined Roadmap

Recommended order:

1. Implement direct measurement tools around current verifier hot spots.
2. Prototype tensor challenges for a small root-only configuration.
3. Separately prototype claim reduction against a tiny setup matrix.
4. Once both are independently validated, combine them in one D32 root-level
   experiment.
5. Extend to D64 and D128 only after security-margin and benchmark review.

The two optimizations are logically independent:

- Tensor challenges reduce challenge-vector derivation/evaluation cost.
- Claim reduction reduces setup-table evaluation cost.

Together they are the path toward the fourth-root verifier profile, but each
should land behind a feature/config gate with regression tests before becoming
the default.
