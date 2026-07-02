# Spec: Tensor Challenge Prover Cutover

| Field         | Value                          |
|---------------|--------------------------------|
| Author(s)     | Quang Dao                      |
| Created       | 2026-07-02                     |
| Status        | proposed                       |
| PR            |                                |
| Supersedes    |                                |
| Superseded-by |                                |
| Book-chapter  |                                |

## Summary

Akita has a tensor challenge verifier path that keeps the final ring-switch
challenge evaluations structured. The prover side still treats much of the
same tensor challenge as if it were a dense logical challenge vector. This spec
cuts over the prover to tensor-aware folded-witness and quotient kernels, adds a
small CI profile benchmark for the tensor preset, and gives tensor fold
grinding its own tail-bound formula instead of reusing the flat one.

The current public tensor preset is the fp128 D64 one-hot preset. It uses a
tensor challenge at the root fold only. Recursive suffix folds stay flat. This
is the intended shape for the first implementation pass. After the root fold
there is one committed recursive witness, so the verifier-reduction win from a
large tensor challenge is no longer present in the same form.

## Intent

### Goal

Make tensor-shaped fold challenges a first-class prover execution path at the
root fold, without expanding the tensor product when computing the folded
witness, relation quotient, or profile benchmark coverage.

The affected surfaces are:

1. `akita-prover` folded witness kernels for dense, one-hot, sparse-ring, and
   root-projection backends.
2. `akita-prover` ring-relation quotient accumulation for tensor challenges.
3. `akita-types` fold Linf cap and fold grind policy, through the tensor-chaos
   tail-bound formula in this spec.
4. `akita-config::tensor_verifier::fp128::D64OneHotTensor`, which remains a
   root fold tensor preset.
5. `akita-pcs` profile mode and CI profile feature wiring for
   `onehot_fp128_d64_tensor`.
6. The tensor challenge spec and book content, which should stop implying that
   the current prover path is already structurally optimal.

### Invariants

1. The tensor preset uses tensor challenges only at level 0. Every recursive
   level uses the existing flat challenge shape unless a later spec changes the
   recursive protocol and its schedule tables.
2. A tensor logical challenge for claim `c` and block `b = p * R + q` is still
   the exact negacyclic product

   ```text
   C[c,p,q] = L[c,p] * R[c,q] in Z[X] / (X^D + 1).
   ```

3. The folded witness is byte-identical to a direct negacyclic tensor-product
   reference. Only the order of contraction changes.
4. The relation quotient is byte-identical to a direct negacyclic tensor-product
   reference.
5. The verifier path remains factored and exact. In particular, ring-switch
   evaluation after tensoring must include the negacyclic wrap correction. It is
   not the same as evaluating two individually ring-switched factors and
   multiplying them at a generic alpha.
6. Tensor fold grinding must use the tensor-chaos tail formula in this spec, not
   the flat independent-challenge formula. If that formula is not implemented,
   tensor folds stay on the current worst-case beta policy.
7. The tensor profile benchmark must use generated tensor schedules, not a
   runtime override of a flat schedule.
8. Verifier-reachable code must reject malformed tensor shapes with `AkitaError`
   or `SerializationError`, not panic.

### Non-Goals

1. Do not make tensor challenges the default for all presets.
2. Do not add tensor challenges to recursive suffix folds in this PR.
3. Do not add a compatibility shim for old tensor experiments.
4. Do not change proof serialization for Fiat-Shamir-derived challenges.
5. Do not reuse the flat `t_star` formula for tensor folds. Tensor folds may use
   tail-bound grind only through the tensor-chaos formula in this spec.
6. Do not implement generic k-way tensor challenges in this cutover. The design
   should not block them, but the implementation target is the existing two-way
   tensor preset.

## Evaluation

### Acceptance Criteria

- [x] `onehot_fp128_d64_tensor` is included in the CI profile benchmark matrix
      at `nv = 32`, `num_polys = 1`, with the tensor schedule feature enabled
      by `profile-ci`.
- [x] `scripts/check_profile_ci_features.sh` knows that
      `onehot_fp128_d64_tensor` requires the tensor schedule feature.
- [x] The folded-witness prover path for tensor challenges computes the same
      output as a direct negacyclic tensor-product reference without
      materializing one logical challenge per `(claim, block)`.
- [x] The one-hot tensor folded-witness path does not recompute the same rotated
      tensor product for every worker chunk.
- [x] The relation quotient tensor path avoids allocating the full logical
      challenge vector and has a reference test against direct ring
      multiplication.
- [x] Previewed tensor challenges and live sampled tensor challenges match for
      the same transcript state and nonce.
- [x] Tensor fold grind uses a distinct tensor-chaos `t_star` formula, with a
      descriptor-bound formula tag and tests against the current
      `WorstCaseBetaOnly` fallback.
- [x] The old ignored tensor end-to-end tests are either made active at a small
      deterministic size or replaced by active coverage that proves and verifies
      the tensor root path.
- [x] Documentation states that the current tensor preset is root fold only and
      that recursive folds remain flat.
- [x] `ChallengeShape::effective_l2_sq_max` is corrected to the deterministic
      materialized-product L2 envelope, so tensor grind never interprets
      `s2_factor^2` as that deterministic bound.

### Testing Strategy

The minimum test set for the first implementation PR is:

```text
cargo test -p akita-challenges tensor
cargo test -p akita-prover tensor
cargo test -p akita-verifier tensor
cargo test -p akita-pcs --test single_poly_tensor_e2e
cargo test -p akita-types tensor_tail_bound_matches_hand_formula
AKITA_MODE=onehot_fp128_d64_tensor AKITA_NUM_VARS=20 cargo run --release -p akita-pcs --features profile-ci --example profile
./scripts/check_profile_ci_features.sh
./scripts/check-doc-guardrails.sh
```

The CI profile matrix should add one case:

```text
onehot_fp128_d64_tensor:26:1
```

nv=26 (not nv=32): under the 138-bit L-infinity SIS floors the tensor root split
is top-heavy, so the public setup matrix grows ~4x per +2 nv (~1 GiB at nv=26,
~72 GiB at nv=32, which OOM-aborts the runner during setup). nv=26 keeps the
tensor setup footprint on par with the flat `onehot_fp128_d64` nv=32 cell while
still exercising the tensor root fold plus recursive flat folds.

This is intentionally one root singleton case. A recursive setup-mode case can
be added later if the CI budget allows it, but it is not needed to prove that
the root tensor folded-witness path works.

### Performance

Let:

```text
C = number of claims
P = left tensor length
Q = right tensor length
B = P * Q logical fold blocks
J = block length in witness entries
T = number of decomposition digits
D = ring dimension
w = sparse support per factor challenge
W = support of a materialized tensor product challenge
```

The old materialized tensor folded-witness path did this shape of work:

```text
for c in C:
  for p in P:
    for q in Q:
      C_pq = tensor_product(L_cp, R_cq)
      z += C_pq * decompose(witness[c,p,q])
```

For sparse factors, `W` is at most `w^2`, before collisions. The folded witness
therefore costs roughly:

```text
current tensor fold work  = C * P * Q * J * T * D * W
current product material  = C * P * Q * D * W
```

The cutover computes the same inner product by contracting one tensor factor at
a time:

```text
tmp[c,p,j,t] = sum_q R[c,q] * decompose(witness[c,p,q,j,t])
z[j,t]      += sum_c sum_p L[c,p] * tmp[c,p,j,t]
```

This costs roughly:

```text
right contraction = C * P * Q * J * T * D * w
left contraction  = C * P     * J * T * D * w
working memory    = J * T * D coefficients per worker slice
```

The asymptotic read of witness blocks is unchanged. The win is that each update
uses one sparse factor instead of a materialized tensor product. For balanced
two-way tensors where `P ~= Q ~= sqrt(B)`, the logical block loop remains
`B`, but the challenge work changes from `W ~= w^2` to `w`, plus a lower-order
left pass.

For one-hot witnesses, the same formula should be implemented as an entry-driven
contraction. The kernel should scan occupied entries, apply the right factor into
a temporary accumulator for the current `(claim, p)` slice, then apply the left
factor once to the accumulated temporary state. This avoids building a rotated
tensor challenge for each logical block.

For relation quotient accumulation, the witness term is not separable in
general, so the verifier-style factored aggregate formula does not directly
apply. The first target is allocation-free tensor product use, then loop fusion
where the same witness block is otherwise scanned more than once. A direct
reference formula for a block is:

```text
L_p = sum_i a_i X^i
R_q = sum_j b_j X^j
S_pq = sum_s s_s X^s
kappa(i,j) = (i + j) mod D
sign(i,j) =  1 if i + j < D
          = -1 otherwise

HighHalf((L_p * R_q) * S_pq)_t
  = sum_{i,j : kappa(i,j) > t}
      sign(i,j) * a_i * b_j * s_{t + D - kappa(i,j)}
```

The implementation may use a clearer equivalent routine, but it must be tested
against a direct negacyclic tensor-product reference.

## Design

### Current State

The tensor challenge feature already has the right verifier idea, but several
prover and profile paths still erase the structure.

1. Dense tensor folding used to materialize logical tensor products before
   folding.
2. Sparse-ring tensor folding also expands the tensor product before folding.
3. One-hot tensor folding avoids one global logical vector, but it still derives
   per-block tensor products and rotated challenge tables inside worker chunks.
4. Ring-relation quotient accumulation expands tensor challenges before the
   high-half product accumulation.
5. The profile example has `onehot_fp128_d64_tensor`, but the CI profile feature
   set and mode feature guard do not treat it as a CI profile case.
6. Tensor end-to-end coverage exists, but the active CI signal is weaker than
   the feature deserves.
7. Tensor fold grinding is intentionally conservative today. It uses the
   worst-case beta policy, not a tensor tail-bound grind.
8. Any planner or schedule path that estimates root fold cost from flat
   challenge mass must be checked against the tensor `omega^2` mass.

### Architecture

The cutover has three layers.

1. Challenge shape stays protocol-owned.

   `LevelParams::fold_challenge_shape` is still the source of truth. The tensor
   preset continues to set tensor shape only at level 0. Prover kernels must
   dispatch from this shape and must not infer tensor mode from a schedule name.

2. Prover kernels consume tensor factors directly.

   Add backend-owned contraction routines that accept `TensorChallenges` or a
   borrowed factor view. These routines should not materialize a full logical
   tensor-product vector on the hot path. Tests should use direct negacyclic
   multiplication references instead of a prover-facing expansion path.

3. Fold grind policy remains explicit.

   Tensor folds must not enter the flat `TailBoundWithGrind` path. They either
   stay `WorstCaseBetaOnly` or enter a distinct tensor-chaos grind policy whose
   formula tag is bound in the instance descriptor.

### Fold Grinding and `t_star`

The flat fold-grind proof prices one folded-witness coefficient as a linear sum
of independent signed challenge coefficients. A tensor fold is a degree-two
signed sum. It needs a different threshold, but it is still elementary.

Fix one fold challenge call. Let:

```text
n = num_claims
P = left tensor length per claim
Q = right tensor length per claim
B = n * P * Q = num_fold_blocks
N = num_fold_coeffs = inner_width * D
s_inf = witness_linf
s2_L = max ||L[c,p]||_2^2
s2_R = max ||R[c,q]||_2^2
k_L = max number of nonzero coefficients in one left factor
k_R = max number of nonzero coefficients in one right factor
p = p_grind
mu = 1 - p
```

For the current tensor preset, the two factors use the same exact-shell family,
so `s2_L = s2_R = 71` and `k_L = k_R = 41`. The formula below is written for
possibly different factor families so the API does not bake in that equality.

For claim `c`, left index `p0`, right index `q0`, and output coefficient `rho`,
the logical challenge is

```text
C[c,p0,q0] = L[c,p0] * R[c,q0] in Z[X] / (X^D + 1).
```

Condition on all supports and magnitudes. Write the random signs in the left
factor as `xi` and the random signs in the right factor as `eta`. After expanding
the negacyclic products, one output coefficient has the form:

```text
Z_rho = sum_{c,p0,a} xi[c,p0,a] * m[c,p0,a] * U[c,p0,a,rho]

U[c,p0,a,rho]
  = sum_{q0,b} eta[c,q0,b] * nmag[c,q0,b]
      * sign(a,b,rho) * S[c,p0,q0,rho - a - b]
```

The subtraction in `rho - a - b` is the negacyclic index with the usual wrap
sign absorbed into `sign(a,b,rho)`. The witness coefficient satisfies
`|S[...]| <= s_inf`.

For fixed `(c,p0,a,rho)`, `U` is a linear sum of independent right-factor signs.
Its Hoeffding variance proxy is:

```text
sum_{q0,b} (nmag[c,q0,b] * S[...])^2
  <= s_inf^2 * Q * s2_R.
```

Let the inner failure budget be `mu / 2`. There are at most `N * n * P * k_L`
such `U` values in the right-first orientation. Therefore:

```text
lambda_inner_R = ln(4 * N * n * P * k_L / mu)
u_R^2          = 2 * s_inf^2 * Q * s2_R * lambda_inner_R
```

gives:

```text
Pr[exists rho,c,p0,a with |U[c,p0,a,rho]| > u_R] <= mu / 2.
```

On that event, `Z_rho` is a linear sum of independent left-factor signs with
variance proxy:

```text
sum_{c,p0,a} (m[c,p0,a] * U[c,p0,a,rho])^2
  <= u_R^2 * n * P * s2_L.
```

Unioning this final left-sign tail over the `N` folded-witness coefficients with
the other half of the failure budget gives:

```text
lambda_outer = ln(4 * N / mu)
t_tensor_Rfirst^2
  = 2 * u_R^2 * n * P * s2_L * lambda_outer
  = 4 * B * s_inf^2 * s2_L * s2_R
      * lambda_inner_R * lambda_outer.
```

The left-first proof is symmetric:

```text
lambda_inner_L = ln(4 * N * n * Q * k_R / mu)
t_tensor_Lfirst^2
  = 4 * B * s_inf^2 * s2_L * s2_R
      * lambda_inner_L * lambda_outer.
```

The implementation should use the smaller orientation:

```text
lambda_inner =
  min(
    ln(4 * N * n * P * k_L / mu),
    ln(4 * N * n * Q * k_R / mu)
  )

t_tensor^2 =
  4 * B * s_inf^2 * s2_L * s2_R * lambda_inner * lambda_outer

lambda_outer = ln(4 * N / mu)
```

As in the flat policy, integer code uses conservative ceilings for the logarithms
and `isqrt_ceil(t_tensor^2)` at the digit boundary. This proves:

```text
Pr[||z||_inf > t_tensor] <= 1 - p_grind.
```

So the expected number of grind probes is at most `1 / p_grind`. With the current
`p_grind = 1/8`, it is at most `8`.

For the current symmetric factor family:

```text
s2_L = s2_R = challenge_l2_sq_max
k_L  = k_R  = challenge_nonzero_count_max
P    = 2^floor(r_vars / 2)
Q    = 2^ceil(r_vars / 2)
B    = n * 2^r_vars

lambda_inner =
  ln(4 * N * n * min(P, Q) * k_L / (1 - p_grind))

t_tensor^2 =
  4 * B * s_inf^2 * challenge_l2_sq_max^2
      * lambda_inner * ln(4 * N / (1 - p_grind)).
```

Concrete challenge-family constants:

| ring dimension | family | `omega = l1` | `s2 = max ||factor||_2^2` | `k = support max` | deterministic tensor `max ||L*R||_2^2` bound | tensor-chaos multiplier before `B * s_inf^2 * lambdas` |
|----------------|--------|--------------|----------------------------|-------------------|-----------------------------------------------|---------------------------------------------------------|
| D64 | `ExactShell { count_mag1: 31, count_mag2: 10 }` | 51 | 71 | 41 | `51^2 * 71 = 184671` | `4 * 71^2 = 20164` |
| D128 | `Uniform { weight: 31, nonzero_coeffs: [-1, 1] }` | 31 | 31 | 31 | `31^3 = 29791` | `4 * 31^2 = 3844` |
| D256 | `Uniform { weight: 23, nonzero_coeffs: [-1, 1] }` | 23 | 23 | 23 | `23^3 = 12167` | `4 * 23^2 = 2116` |

So D128 and D256 use the same tensor-chaos method, with smaller absolute
threshold constants than D64. Their ratio against the tensor worst-case beta
`B * omega^2 * s_inf` is not uniformly better, because their `omega^2 / s2`
ratio is smaller, but the absolute `t_tensor` for the same `(B, N, s_inf)` is
lower.

For a concrete balanced-root comparison with `N = 2^16`, `n = 1`, `r_vars = 16`,
`p_grind = 1/8`, and `s_inf = 1`:

| ring dimension | `t_tensor` | tensor beta `B * omega^2` | ratio |
|----------------|------------|----------------------------|-------|
| D64 | 603675 | 170459136 | 0.003541 |
| D128 | 261886 | 62980096 | 0.004158 |
| D256 | 192955 | 34668544 | 0.005566 |

The absolute threshold improves as the ring challenge gets lighter. The relative
ratio is worse for D128 and D256 because their tensor beta also falls faster.

This is larger than the flat independent-sign formula by roughly one logarithmic
factor, as it must be. The all-ones rank-one example
`Z = (sum_i xi_i) * (sum_j eta_j)` has product tails of order `sqrt(PQ) * log`,
not `sqrt(PQ * log)`. The tensor formula above has that shape and therefore does
not pretend that the `P * Q` logical products are independent.

The L2 bound for a tensor product also needs care. For ring product
`c = a * b mod (X^D + 1)`, the safe deterministic inequalities are:

```text
||c||_2 <= ||a||_1 * ||b||_2
||c||_2 <= ||a||_2 * ||b||_1
```

For identical factor envelopes, this gives:

```text
||c||_2^2 <= l1_factor^2 * l2_factor^2.
```

This is the true deterministic materialized-product L2 envelope. It is not
`s2_factor^2`. Collisions and negacyclic wrap can make
`||a * b mod (X^D + 1)||_2^2` larger than `||a||_2^2 * ||b||_2^2`.

The tensor `t_star` formula does contain `s2_L * s2_R`, but that quantity is the
second-order sign-chaos scale from the proof above. It must not be documented or
used as the deterministic L2 norm of one materialized product challenge.

Implementation consequence:

```text
flat TailBoundWithGrind:
  t_flat^2 = 2 * B * s_inf^2 * s2_flat * ln(2 * N / mu)

tensor TailBoundWithGrind:
  t_tensor^2 = 4 * B * s_inf^2 * s2_L * s2_R
      * ln(4 * N / mu)
      * min(
          ln(4 * N * n * P * k_L / mu),
          ln(4 * N * n * Q * k_R / mu)
        )
```

The policy enum, descriptor binding, and tests should distinguish these formulae.
If that code is not part of the implementation PR, tensor folds must remain
`WorstCaseBetaOnly`.

### Why Ring-Switching After Tensoring Is Different

The tensor logical challenge is formed in the integer negacyclic ring first:

```text
C[p,q](X) = L[p](X) * R[q](X) mod (X^D + 1).
```

Ring-switch evaluation then evaluates this reduced polynomial at powers of
`alpha`. For a generic `alpha`, this is not equal to:

```text
eval_alpha(L[p]) * eval_alpha(R[q]).
```

The unreduced product differs from the reduced negacyclic product by a multiple
of `X^D + 1`. Evaluating at `alpha` leaves a correction proportional to
`alpha^D + 1`. The product-only formula is exact only at a negacyclic root where
`alpha^D + 1 = 0`, or in cases where the wrap quotient is zero.

The verifier already has a structured formula for this. The prover cutover must
not replace it with a product of individually ring-switched factors.

### General k-Way Tensor Shape

The two-way contraction generalizes algebraically. For a k-way tensor with
factor lengths `N_1, ..., N_k`, the logical challenge is:

```text
C[i_1,...,i_k] = A_1[i_1] * ... * A_k[i_k]
```

The folded witness can be contracted one factor at a time:

```text
tmp_k     = sum_{i_k}     A_k[i_k]     * witness[i_1,...,i_k]
tmp_{k-1} = sum_{i_{k-1}} A_{k-1}[i_{k-1}] * tmp_k
...
z         = sum_{i_1}     A_1[i_1]     * tmp_2
```

The operation count is:

```text
sum_{r=1}^k C * J * T * D * w_r * product_{m=1}^r N_m
```

if the contraction runs from the last factor toward the first and stores the
remaining prefix state. The best order is the one that minimizes temporary
state and factor work for the actual witness layout. This spec only implements
the two-way path, but the backend API should not bake in an expanded logical
challenge vector as the only abstraction.

The same conditioning proof extends to k-way tensor grind. Choose an order of
the `k` factors. Let:

```text
n = num_claims
H = num_fold_coeffs
B = n * product_{i=1}^k N_i
mu = 1 - p_grind
s2_i = max squared L2 norm of factor i
k_i = max support size of factor i
```

Allocate failure budget `mu / k` to each conditioning layer. For the chosen order,
define:

```text
lambda_1 = ln(2 * k * H / mu)

lambda_j =
  ln(2 * k * H * n * product_{i<j} (N_i * k_i) / mu)
  for j = 2..k
```

Then the same iterated Hoeffding argument gives:

```text
t_kway^2 =
  2^k * B * s_inf^2 * product_{i=1}^k s2_i
      * product_{j=1}^k lambda_j.
```

The implementation should choose the factor order that minimizes the product of
the `lambda_j` terms and the temporary memory. For `k = 2`, this collapses to
the two-way tensor formula above. This spec still implements only the existing
two-way tensor preset, but the math does not leave the k-way case undefined.

### Alternatives Considered

1. Keep materializing tensor products in the prover.

   This is simple, but it erases most of the structure that the tensor feature
   introduced. It also makes the one-hot path recompute rotated tensor products
   in places where an iterative contraction is available.

2. Enable tensor grind immediately with the existing flat `t_star` formula.

   This is rejected. Tensor logical challenges are dependent because they share
   factors. Tensor grind must use the tensor-chaos formula in this spec, with a
   distinct formula tag and tests, before it affects security sizing.

3. Add tensor challenges to recursive suffix folds.

   This is out of scope. The current preset intentionally uses tensor shape only
   at the root. Extending recursive levels would need new schedules, new profile
   data, and a fresh analysis of whether verifier work is actually reduced there.

## Documentation

Update `specs/tensor-structured-folding-challenges.md` after implementation so
it no longer overstates prover-side optimization. The Akita Book should state:

1. The public tensor preset is fp128 D64 one-hot.
2. The tensor challenge is used at the root fold only.
3. Recursive suffix folds are flat.
4. The verifier evaluates tensor ring-switch rows with a wrap-corrected factored
   formula.
5. Tensor fold grinding is conservative unless a future tensor `t_star`
   derivation is active.

## Execution

1. Add the CI profile tensor case.

   Add `onehot_fp128_d64_tensor:26:1` to `.github/workflows/profile-bench.yml`.
   Add `akita-config/schedules-fp128-d64-onehot-tensor` to the `profile-ci`
   feature set. Update `scripts/check_profile_ci_features.sh`.

2. Add reference tests.

   Add small tests that compare direct negacyclic tensor-product references to
   the new contraction kernels for dense, one-hot, sparse-ring, and quotient
   accumulation. Add a preview-versus-live tensor challenge test for fold grind.

3. Cut over folded witness kernels.

   Replace materialized tensor-product paths with factor contraction.
   Keep test references independent of the integer expansion path.

4. Cut over relation quotient accumulation.

   Avoid allocating full logical challenge vectors. Start with a direct
   allocation-free tensor product iterator, then fuse scans where it is clear and
   measured.

5. Wire tensor grind deliberately.

   Add a tensor-chaos grind policy only if the implementation also adds the
   integer `t_tensor^2` helper, descriptor formula tag, schedule plumbing, and
   tests from this spec. If that slice is deferred, keep tensor folds on
   `WorstCaseBetaOnly` and document that the blocker is implementation, not
   missing math.

6. Regenerate schedules if tensor grind is enabled.

   Tensor `t_star` can change `num_digits_fold`. Any implementation that enables
   it must regenerate the tensor schedule tables and rerun the generated schedule
   drift tests.

## References

1. `specs/tensor-structured-folding-challenges.md`
2. `crates/akita-config/src/tensor_verifier.rs`
3. `crates/akita-prover/src/backend/dense/tensor_fold.rs`
4. `crates/akita-prover/src/backend/sparse_ring/tensor_fold.rs`
5. `crates/akita-prover/src/backend/onehot/accumulate.rs`
6. `crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs`
7. `crates/akita-prover/src/protocol/fold_grind.rs`
8. `crates/akita-types/src/sis/fold_linf_cap.rs`
9. `crates/akita-types/src/sis/fold_witness_grind.rs`
10. `.github/workflows/profile-bench.yml`
