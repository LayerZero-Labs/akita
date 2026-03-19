# Labrador Linear-Only Protocol

This note explains the linear-equation-only Labrador variant implemented in this repo, but from the vantage point of the protocol after the obvious implementation inefficiencies have already been removed. The algebra, transcript schedule, and recursive statement are the same as in the code; what we elide is wasteful materialization such as building large temporary matrices, reconstructing sparse constraints at recursive levels, or carrying row-by-row objects when only their final aggregate is needed.

Concretely:

- Transcript hashing, challenge derivation, and accepted JL nonce replay stay in-protocol.
- JL matrix derivation stays transcript-bound, but the matrix is conceptually streamed from its seed instead of fully materialized.
- Recursive levels carry the compact reduced state `amortized_phi`, `aggregated_rhs`, `challenges`, `config`, and `setup`, rather than a rebuilt sparse constraint list.
- Tail verification is described directly in decomposed coordinates, since that is the cleanest optimized form.

Main code anchors for the current implementation:

- Prover loop: `src/protocol/labrador/prover.rs:24-115`
- Per-level fold: `src/protocol/labrador/fold.rs:83-200`
- Verifier reduction: `src/protocol/labrador/verifier.rs:129-205`
- Tail verifier: `src/protocol/labrador/verifier.rs:302-407`
- JL aggregation: `src/protocol/labrador/aggregation.rs:546-639`
- Recursive constraint plan: `src/protocol/labrador/types.rs:56-78`, `src/protocol/labrador/constraints.rs:223-245`

## Scope

This is the "linear-only" Labrador used here:

- Explicit constraints are linear ring equations of the form
  $$
  \sum_t \langle \phi_t, s_{\mathrm{row}(t)}[\mathrm{offset}(t)..] \rangle = b.
  $$
  This is exactly the shape of `LabradorConstraint` / `LabradorConstraintTerm` in `src/protocol/labrador/constraints.rs:14-56`.
- The quadratic paper terms are omitted. In repo terms, this is the `quadratic = 0` path described in `src/protocol/labrador/config.rs:33-76`, and the recursive builders explicitly omit the quadratic `a_ij` / `g_ij` terms in `src/protocol/labrador/constraints.rs:278-449`.

So the whole protocol should be read as:

1. Fold many linear constraints into one aggregated linear relation.
2. Add a JL-based linear certificate that the witness is norm-bounded.
3. Compress the witness by replacing many rows with one random linear combination plus committed side information.
4. Recurse on the reduced statement.

## Symbol Table

The repo symbols line up with the following math objects. When useful, the old
repo/paper names appear once in parentheses.

| Symbol | Meaning | Repo anchor |
| --- | --- | --- |
| `s_i` | Original witness row | `src/protocol/labrador/types.rs:9-53` |
| `virtual_row_count` (old `rr`) | Number of virtual rows after reshaping | `src/protocol/labrador/fold.rs:85-90` |
| `virtual_row_len` (old `nn`) | Common virtual row length | `src/protocol/labrador/config.rs:62-76` |
| `row_split_counts` (old `nu`) | Reshaping metadata from original rows to virtual rows | `src/protocol/labrador/config.rs:13-24`, `src/protocol/labrador/fold.rs:204-239` |
| `A` | Inner commitment matrix | `src/protocol/labrador/setup.rs:32-47` |
| `t_i = A s_i` | Inner commitments of virtual witness rows | `src/protocol/labrador/fold.rs:92-100` |
| `inner_opening_digits` (old `t_hat`) | Decomposed packed form of all `t_i` | `src/protocol/labrador/commit.rs:203-248` |
| `B` | Outer commitment matrix for `inner_opening_digits` | `src/protocol/labrador/setup.rs:48-57` |
| `inner_opening_payload` (old `u1`) | Outer commitment to `inner_opening_digits`, or raw `inner_opening_digits` in tail mode | `src/protocol/labrador/fold.rs:92-100`, `src/protocol/labrador/verifier.rs:294-300` |
| `D_out` | Outer commitment matrix for `linear_garbage_digits` (repo `d_mat`, paper `D`) | `src/protocol/labrador/setup.rs:58-67` |
| `linear_garbage_payload` (old `u2`) | Outer commitment to linear garbage `linear_garbage_digits`, or raw `linear_garbage_digits` in tail mode | `src/protocol/labrador/fold.rs:137-146`, `src/protocol/labrador/fold.rs:321-330` |
| `jl_projection` | Public 256-dimensional JL projection vector | `src/protocol/labrador/johnson_lindenstrauss.rs:286-333` |
| `jl_nonce` | Accepted nonce used to derive the JL matrix from the transcript | `src/protocol/labrador/johnson_lindenstrauss.rs:246-273`, `src/protocol/labrador/johnson_lindenstrauss.rs:312-327` |
| `jl_lift_residuals` (old `bb`) | JL lift polynomials with constant term zeroed before transmission | `src/protocol/labrador/aggregation.rs:574-588`, `src/protocol/labrador/johnson_lindenstrauss.rs:357-375` |
| `c_i` | Amortization challenges used to define `z = sum_i c_i s_i` | `src/protocol/labrador/fold.rs:148-155`, `src/protocol/labrador/verifier.rs:182-183` |
| `amortized_phi` (old `combined_phi`) | `sum_i c_i phi_i`, the only coefficient vector the next round really needs | `src/protocol/labrador/constraints.rs:105-136`, `src/protocol/labrador/types.rs:63-78` |
| `aggregated_rhs` (old `b_total`) | Aggregated right-hand side carried into the next round | `src/protocol/labrador/fold.rs:133-135`, `src/protocol/labrador/verifier.rs:178-180` |

The reduction parameters are:

| Parameter | Meaning | Repo anchor |
| --- | --- | --- |
| `witness_digit_parts` (old `f`) | Number of decomposition parts for `z` | `src/protocol/labrador/types.rs:97-113` |
| `witness_digit_bits` (old `b`) | Bit-width of each `z` decomposition digit | `src/protocol/labrador/types.rs:97-113` |
| `aux_digit_parts` (old `fu`) | Number of decomposition parts for committed side data (`inner_opening_digits`, `linear_garbage_digits`) | `src/protocol/labrador/types.rs:97-113` |
| `aux_digit_bits` (old `bu`) | Bit-width of each committed-side digit | `src/protocol/labrador/types.rs:97-113` |
| `inner_commit_rank` (old `kappa`) | Inner commitment rank | `src/protocol/labrador/types.rs:97-113` |
| `outer_commit_rank` (old `kappa1`) | Outer commitment rank | `src/protocol/labrador/types.rs:97-113` |
| `tail` | Whether this is the final tail fold with no outer `B`/`D` commitments | `src/protocol/labrador/types.rs:97-113`, `src/protocol/labrador/setup.rs:48-67` |

## Decomposition Convention

Whenever a ring-vector `x` is decomposed into `m` digits in base `2^B`, the intended reconstruction is

$$
x = \sum_{u=0}^{m-1} 2^{uB} x^{(u)}.
$$

For Labrador there are two such decompositions:

- Witness-side: `z = sum_{t=0}^{witness_digit_parts-1} 2^{t * witness_digit_bits} z^{(t)}`
- Commitment-side: `t_i = sum_{u=0}^{aux_digit_parts-1} 2^{u aux_digit_bits} t_i^{(u)}` and `h_{ij} = sum_{u=0}^{aux_digit_parts-1} 2^{u aux_digit_bits} h_{ij}^{(u)}`

The carry logic is an implementation detail of `decompose_rows_with_carry`; algebraically the protocol only cares that these equalities hold.

## Big Picture

At a high level, one non-tail Labrador round does this:

```text
original statement + witness rows
        |
        | reshape rows using row_split_counts into virtual_row_count virtual rows of length virtual_row_len
        v
  s_1, ..., s_virtual_row_count
        |
        | inner commitment with A
        v
  t_i = A s_i
        |
        | decompose t_i -> inner_opening_digits, outer commit with B
        v
       inner_opening_payload
        |
        | transcript derives JL seed + accepted nonce
        | JL projection p and JL collapse
        v
   phi^JL, b^JL
        |
        | aggregate statement side
        v
  phi^stmt, b^stmt
        |
        | add them
        v
  phi_i, aggregated_rhs
        |
        | build linear garbage h_ij and commit/decompose it
        v
       linear_garbage_payload
        |
        | sample amortization challenges c_i
        v
  z = sum_i c_i s_i
        |
        | decompose z
        v
 next witness = z-parts plus side data
        |
        | next statement keeps only compact reduced state
        v
 (amortized_phi, aggregated_rhs, c_i, config, setup)
```

The verifier follows the same Fiat-Shamir schedule, recomputes the same reduced state, and either:

- passes that state to the next Labrador round, or
- in tail mode, checks the final algebra directly on the decomposed witness.

## What The Statement Says

The initial statement is a set of sparse linear ring constraints. One scalar constraint is

<!--
$$
\sum_{t \in T_\ell}
\left\langle
\phi_{\ell,t},
s_{\rho(\ell,t)}[o(\ell,t) .. o(\ell,t) + |\phi_{\ell,t}|]
\right\rangle
= b_\ell.
$$

-->

$$ \sum_{t \in T_\ell} \left\langle \phi_{\ell,t}, s_{\rho(\ell,t)}[o(\ell,t) .. o(\ell,t) + |\phi_{\ell,t}|] \right\rangle = b_\ell. $$

This is just the math form of `LabradorConstraint` in `src/protocol/labrador/constraints.rs:39-56`.

At recursive levels, the statement is no longer stored as a sparse list. Instead it is stored as the compact object

$$
(\mathrm{virtual\_row\_count}, \mathrm{virtual\_row\_len}, \mathrm{config}, c_i, \mathrm{amortized\_phi}, \mathrm{aggregated\_rhs}, \mathrm{setup}),
$$

namely `LabradorReducedConstraintPlan` in `src/protocol/labrador/types.rs:56-78`.

This is already the key optimization: the next round does not need the whole old constraint list. It only needs the coefficient vector that will interact with the next amortized witness and the one right-hand side ring element it must hit.

## Phase 0: Reshaping

The fold planner chooses:

- a common virtual row length `virtual_row_len`,
- a number of virtual rows `virtual_row_count`,
- and reshaping metadata `row_split_counts`.

In this repo's linear-only mode, the planner treats all original rows as one long concatenated stream and then splits it into `virtual_row_count` chunks of length `virtual_row_len`; see `src/protocol/labrador/config.rs:62-76` and `src/protocol/labrador/config.rs:97-103`.

So conceptually:

$$ (s_1^{orig}, \dots, s_r^{orig}) \longmapsto (\tilde{s}_1, \dots, \tilde{s}_{virtual_row_count}), \qquad \tilde{s}_i \in R_q^{virtual_row_len}. $$

Why reshape at all? Because all the linear algebra after this point wants uniform row sizes.

### Handoff-specific planning profile

The generic recursive planner and the Hachi handoff now share the same search machinery, but they do not start from the same model.

- Ordinary Labrador recursion seeds the planner from the actual witness row lengths and the current squared norm estimate.
- The Hachi handoff seeds it from a richer witness profile: `(row_lengths, norm_sum, coeff_bit_bound)`, where `coeff_bit_bound` is the small balanced coefficient width of the incoming Hachi witness.
- For the canonical `logbasis` path, `coeff_bit_bound = 3`, so the planner caps the initial `var(z)` estimate at the variance of 3-bit balanced digits instead of pretending that the handoff witness already occupies the whole field range.

This is the important modeling change for the direct handoff. Before it, the handoff planner only saw `(row_count, max_row_len)`, so it could not distinguish "same shape, tiny coefficients" from "same shape, near-field-width coefficients."

### Decomposition search and byte objective

The current planner searches `witness_digit_parts` over a wider range instead of stopping at `f <= 2`.

- Standard folds search `witness_digit_parts in {1, ..., 8}`.
- Tail folds still force `witness_digit_parts = 1`.
- Candidate ranks still have to pass the same SIS checks for `inner_commit_rank` and `outer_commit_rank`.

The score is now an estimator for serialized bytes, not a loose ring-element proxy and not a speculative proof run.

- For one fold step, the estimator predicts the exact serialized size of the next `FlatLabradorLevelProof` framing and of the next `FlatLabradorWitness`, using only `(row_lengths, config, D, |F|)`.
- The prover loop uses that estimate to decide whether a non-tail or tail step is worth executing at all.
- The Hachi handoff uses a recursive version of the same estimator to compare:
  1. the packed direct tail (`PackedDigits`), and
  2. the estimated Labrador tail (`FlatLabradorProof` plus `v`, `y_ring`, and the norm bound).

So the decision rule is now: estimate first, execute only the chosen Labrador path, and otherwise keep the direct packed tail.

## Phase 1: Commit The Virtual Witness

Let `A in R_q^{inner_commit_rank x virtual_row_len}` be the inner commitment matrix. The prover computes

$$ t_i = A \tilde{s}_i \qquad \text{for } i = 1, \dots, virtual_row_count. $$

These `t_i` are then decomposed into `aux_digit_parts` base-`2^aux_digit_bits` digits and concatenated into `inner_opening_digits`.

If this is a standard round, the prover also uses `B in R_q^{outer_commit_rank x (virtual_row_count * inner_commit_rank * aux_digit_parts)}` and sends

$$ \mathrm{inner\_opening\_payload} = B \cdot \mathrm{inner\_opening\_digits}. $$

If this is a tail round, then `outer_commit_rank = 0`, the outer matrices are absent, and the payload simply exposes raw `inner_opening_digits`; see `src/protocol/labrador/setup.rs:48-67` and `src/protocol/labrador/verifier.rs:294-300`.

Intuition:

- `t_i` are the "A-committed images" of the witness rows.
- `inner_opening_payload` is a compact binding handle for all those `t_i` digits, so later recursive rounds do not have to reopen every `t_i` directly.

## Phase 2: JL Projection And JL Collapse

This is the subtlest part, and it is the key to the norm bound.

### 2.1 The JL projection

Flatten the centered coefficients of the reshaped witness into an integer vector

$$ \widetilde{s}^{coeff} \in \mathbb{Z}^N, \qquad N = virtual_row_count \cdot virtual_row_len \cdot D_{\mathrm{ring}}. $$

From the transcript, after absorbing level metadata and `inner_opening_payload`, Labrador derives a JL seed. The prover searches for an accepted nonce `jl_nonce`, and the accepted nonce is the only one actually committed to the transcript; see `src/protocol/labrador/johnson_lindenstrauss.rs:286-333`. The verifier replays exactly that accepted nonce once via `replay_nonce_search` in `src/protocol/labrador/johnson_lindenstrauss.rs:246-273`.

The resulting JL matrix is

$$ \Pi \in \{-1, 0, 1\}^{256 \times N}. $$

The prover computes the public projection

$$ p = \Pi \widetilde{s}^{coeff} \in \mathbb{Z}^{256}. $$

which is exactly `jl_projection`.

Intuition:

- `p` is a 256-dimensional sketch of the witness coefficients.
- If the witness norm is large, then with overwhelming probability the sketch norm is also large.
- The verifier wants a proof that this sketch really came from the witness, without checking 256 separate dense coefficient equations one by one.

### 2.2 What JL collapse is

For each lift `k`, the transcript samples 256 field elements

$$ \omega^{(k)} = (\omega_1^{(k)}, \dots, \omega_{256}^{(k)}) \in F^{256}. $$

These are the JL collapse weights, sampled in `src/protocol/labrador/aggregation.rs:82-89`.

They collapse the 256 JL rows into one coefficient vector:

$$ W_c^{(k)}[m] = \sum_{a=1}^{256} \omega_a^{(k)} \Pi_{a,m}, \qquad m = 1, \dots, N. $$

This is exactly what `collapse_jl_weights_*` computes in `src/protocol/labrador/aggregation.rs:245-380`.

So instead of thinking about 256 separate equations, think of one random linear combination of them.

That is the whole point of the collapse:

- The public side becomes one scalar $\sum_{a=1}^{256} \omega_a^{(k)} p_a$.
- The witness side becomes one dense linear form in the witness coefficients.

### 2.3 Turning the collapse into a ring-linear relation

The witness lives as ring elements, not as a flat coefficient vector. So Labrador groups every `D_{\mathrm{ring}}` collapsed weights into one polynomial and applies `sigma_{-1}`:

$$ \phi_j^{JL,(k)} = \sigma_{-1}\!\left(\sum_{d=0}^{D_{\mathrm{ring}}-1} W_c^{(k)}[j D_{\mathrm{ring}} + d] X^d\right), \qquad j = 0, \dots, virtual_row_count \cdot virtual_row_len - 1. $$

This is `CollapseWeights::into_phi()` in `src/protocol/labrador/aggregation.rs:123-143`, and `sigma_m1()` is `X -> X^{-1}` in the cyclotomic ring; see `src/algebra/ring/cyclotomic.rs:122-130`.

Now define

$$ b^{JL,(k)} = \left\langle \phi^{JL,(k)}, \widetilde{s} \right\rangle. $$

Why does this help? Because of the constant-term trick:

$$ \operatorname{ct}\big(\sigma_{-1}(u) \cdot v\big) = \langle \operatorname{coeff}(u), \operatorname{coeff}(v) \rangle. $$

So the constant coefficient of `b^{JL,(k)}` is exactly the scalar JL collapse:

$$ \operatorname{ct}\big(b^{JL,(k)}\big) = \sum_{a=1}^{256} \omega_a^{(k)} p_a. $$

That is the crucial bridge from "256 scalar JL equations over coefficients" to "one ring-linear equation over witness rows."

### 2.4 Why the prover sends `jl_lift_residuals`

The whole polynomial `b^{JL,(k)}` is not public. Only its constant term is public, because that term equals the collapse of the public projection `p`.

So the prover sends

$$ jl_lift_residuals^{(k)} = b^{JL,(k)} - \operatorname{ct}(b^{JL,(k)}). $$

Equivalently: it transmits the full polynomial with its constant term zeroed. This is `zero_constant_term_for_proof` / `restore_constant_term` in `src/protocol/labrador/johnson_lindenstrauss.rs:357-375`.

The verifier restores the full polynomial as

$$ b^{JL,(k)} = jl_lift_residuals^{(k)} + \left(\sum_{a=1}^{256} \omega_a^{(k)} p_a\right). $$

This is exactly the `restore_constant_term(..., collapse_to_field(jl_projection, omega))` step in `src/protocol/labrador/aggregation.rs:627-636`.

### 2.5 Aggregating the JL lifts

After each `jl_lift_residuals^{(k)}` is absorbed, the transcript samples a ring challenge `beta_k`. Labrador then aggregates all JL lifts into one ring-linear system:

$$ \phi_i^{JL} = \sum_k \beta_k \phi_i^{JL,(k)}, \qquad b^{JL} = \sum_k \beta_k b^{JL,(k)}. $$

This is the prover-side routine `aggregate_jl_constraints_prover` in `src/protocol/labrador/aggregation.rs:546-590` and the verifier-side replay in `src/protocol/labrador/aggregation.rs:603-639`.

The number of lifts is

$$ \mathrm{jl\_lifts}(F) = \left\lceil \frac{128}{\log_2 q} \right\rceil. $$

from `src/protocol/labrador/config.rs:357-359`.

For an `Fp128`-like field, `log_2 q = 128`, so this becomes:

$$ \mathrm{jl\_lifts}(F) = 1. $$

That is an extremely important simplification for recursion: in the `Fp128` setting, JL aggregation is just one collapse, one `jl_lift_residuals`, and one `beta`.

## Phase 3: Aggregate The Statement Side

Independently of JL, Labrador folds the statement's existing linear constraints into one aggregated coefficient system:

$$ (\phi^{stmt}, b^{stmt}). $$

There are two cases.

### 3.1 First round or explicit statement

If the statement is still an explicit sparse constraint list, Labrador samples one fresh ring challenge `alpha_l` per scalar constraint and sets

$$ b^{stmt} = \sum_l \alpha_l b_l, \qquad \phi_i^{stmt} = \sum_l \alpha_l \phi_{l,i}. $$

This is `aggregate_statement_constraints` in `src/protocol/labrador/aggregation.rs:808-887`.

### 3.2 Recursive round or reduced statement

If the statement is already reduced, Labrador does not rebuild the whole sparse system. It directly aggregates from the compact plan `LabradorReducedConstraintPlan`; see `src/protocol/labrador/aggregation.rs:667-782`.

This is one of the main recursive design wins of the implementation:

- `inner_opening_payload` and `linear_garbage_payload` targets contribute directly to `b^{stmt}` and auxiliary-row coefficients.
- The opening relation contributes directly to the `z` rows.
- The linear-garbage relation contributes directly via `plan.amortized_phi`.
- The diagonal relation contributes `alpha_diag * plan.aggregated_rhs`.

So the next round never needs the old sparse constraints again.

### 3.3 Why the paper samples full-ring `alpha` and `beta`, and when scalars suffice

The original LaBRADOR paper samples `alpha` and `beta` as **uniform ring elements** in `R_q`, not as base-field scalars. In the paper's concrete regime this is not gratuitous: the base field is only about 32 bits, while the ring

$$
R_q = \mathbb{F}_q[X]/(X^{D_{\mathrm{ring}}} + 1)
$$

splits into two degree-`D_{\mathrm{ring}} / 2` factors. Equivalently,

$$
R_q \cong K_+ \times K_-, \qquad |K_+| = |K_-| = q^{D_{\mathrm{ring}} / 2}.
$$

Appendix B exploits exactly this point: if some aggregated residual is nonzero, then multiplication by a fresh uniform ring challenge is uniformly random in at least one CRT component, so the bad event costs about

$$
q^{-D_{\mathrm{ring}} / 2}.
$$

That is much stronger than `1 / q` when `q` itself is small.

For the `Fp128`-style setting used in this repo, however, **base-field scalar challenges are already enough** for this aggregation role. The reason is that `alpha` and `beta` are only used to form random linear combinations of fixed ring residuals; they are not the short amortization challenges `c_i`, so they do not need bounded operator norm, short coefficients, or invertible differences.

Concretely, suppose after fixing all earlier prover messages the violated constraints contribute residuals

$$
E_1, \dots, E_m \in R_q,
$$

and instead of ring challenges we sample scalars `a_1, ..., a_m ← F`, where `F` is the base field. The verifier then tests

$$
\sum_{i=1}^m a_i E_i = 0 \in R_q.
$$

Now `R_q` is an `F`-vector space of dimension `D_{\mathrm{ring}}`, so this is just one `F`-linear condition on the random vector `(a_i)_i`. If not all `E_i` are zero, then the map

$$
L : F^m \to R_q, \qquad L(a_1, \dots, a_m) = \sum_i a_i E_i
$$

has nonzero image, and therefore

$$
\Pr[L(a_1, \dots, a_m) = 0] = |F|^{- \dim_F(\operatorname{span}\{E_i\})} \le |F|^{-1}.
$$

This bound is tight in the worst case: if all bad residuals are scalar multiples of one nonzero ring element, then the span has dimension `1` and the failure probability is exactly `1 / |F|`.

This is the key caution point:

- For scalar aggregation challenges, the generic worst-case bound is `1 / |F|`, **not** `1 / |F|^{1/2}`.
- The CRT decomposition of the ring does **not** weaken the scalar bound to a square root loss.
- The CRT decomposition only explains why full-ring challenges are even stronger in the paper's small-`q` setting.

So, in an `Fp128`-like regime:

- scalar `alpha` and `beta` already buy about `2^-128` aggregation soundness;
- full-ring `alpha` and `beta` are stronger, but are plausibly overkill for this specific role;
- for JL specifically, `jl_lifts(F) = ceil(128 / log_2 q)` equals `1` when `log_2 q = 128`, so `beta` is only rescaling a single JL lift anyway.

The current code still samples dense ring challenges for these paths; see `src/protocol/transcript/mod.rs:64-74` and the uses in `src/protocol/labrador/aggregation.rs:388-391`, `src/protocol/labrador/aggregation.rs:431-434`, and `src/protocol/labrador/aggregation.rs:684-737`. The point of this note is narrower: for the aggregation argument itself, the paper's full-ring choice is crucial in the small-`q` analysis, but in the repo's `Fp128` setting the same argument already works with full-field scalar challenges and a `1 / |F|` error term.

## Phase 4: Form The Round's Main Linear Relation

Now combine the JL side and the statement side:

$$ \phi_i = \phi_i^{stmt} + \phi_i^{JL}, \qquad \mathrm{aggregated\_rhs} = b^{stmt} + b^{JL}. $$

In code this is the `phi_total` / `aggregated_rhs` step in `src/protocol/labrador/fold.rs:133-135` and `src/protocol/labrador/verifier.rs:178-180`.

At this point, the current round has effectively reduced everything it wants to prove to the claim:

For each row `i`, the inner product $\langle \phi_i, \tilde{s}_i \rangle$ participates in one structured global relation.

## Phase 5: Linear Garbage

The protocol now computes the symmetric cross terms

$$ h_{ii} = \langle \phi_i, \tilde{s}_i \rangle, \qquad h_{ij} = \langle \phi_i, \tilde{s}_j \rangle + \langle \phi_j, \tilde{s}_i \rangle \quad (i < j). $$

This is exactly `compute_linear_garbage` in `src/protocol/labrador/fold.rs:268-299`.

Pack the upper-triangular family `(h_{ij})_{i <= j}` into one vector, decompose it in base `2^aux_digit_bits`, and call the result `linear_garbage_digits`.

Then:

- In a standard round, send $\mathrm{linear\_garbage\_payload} = D_{\mathrm{out}} \cdot \mathrm{linear\_garbage\_digits}$, where `D_out` is the outer commitment matrix from `src/protocol/labrador/setup.rs:58-67`.
- In a tail round, there is no outer `D_out`, so `linear_garbage_payload` is just raw `linear_garbage_digits`.

Intuition:

- `h_{ij}` are the side terms needed so that one amortized witness `z` can stand in for all rows simultaneously.
- `linear_garbage_payload` binds those side terms so they can be carried into the next recursive statement.

## Phase 6: Amortize The Witness

After `linear_garbage_payload` is absorbed, the transcript samples ring challenges

$$ c_1, \dots, c_{virtual_row_count}. $$

The amortized witness is

$$ z = \sum_{i=1}^{virtual_row_count} c_i \tilde{s}_i. $$

This is `amortize_witness` in `src/protocol/labrador/fold.rs:301-319`.

Then decompose:

$$ z = \sum_{t=0}^{\mathrm{witness\_digit\_parts}-1} 2^{t \cdot \mathrm{witness\_digit\_bits}} z^{(t)}. $$

The next witness is:

- Standard round: $w_{next} = \big(z^{(0)}, \dots, z^{(\mathrm{witness\_digit\_parts}-1)},\; \mathrm{inner\_opening\_digits} \Vert \mathrm{linear\_garbage\_digits}\big)$, matching `NextWitnessLayout` in `src/protocol/labrador/constraints.rs:58-103`.
- Tail round: $w_{next} = \big(z^{(0)}, \dots, z^{(\mathrm{witness\_digit\_parts}-1)}\big)$, since the auxiliary row is omitted; see `src/protocol/labrador/fold.rs:352-365`.

## What The Next Statement Keeps

The next recursive statement does not need every `phi_i`. It only needs

$$ \mathrm{amortized\_phi} = \sum_{i=1}^{virtual_row_count} c_i \phi_i. $$

This is `combine_phi` in `src/protocol/labrador/constraints.rs:452-465`.

So the reduced statement stores:

$$ \big(\mathrm{virtual\_row\_count},\; \mathrm{virtual\_row\_len},\; \mathrm{config},\; (c_i),\; \mathrm{amortized\_phi},\; \mathrm{aggregated\_rhs},\; \mathrm{setup}\big). $$

That is exactly `LabradorReducedConstraintPlan` in `src/protocol/labrador/types.rs:56-78`, built by `build_next_constraint_plan` in `src/protocol/labrador/constraints.rs:223-245`.

This is the conceptual heart of Labrador recursion:

- The witness shrinks.
- The statement is re-expressed in the same Labrador language.
- The only coefficient vector the next round needs is the already-amortized one.

## Optimized Prover Flow

Here is the round in the clean "idealized but same algebra" form.

```text
Input:
  witness rows s_i
  statement S
  config = (witness_digit_parts, witness_digit_bits, aux_digit_parts, aux_digit_bits, inner_commit_rank, outer_commit_rank, tail)

1. Reshape original rows into virtual rows \tilde{s}_1, ..., \tilde{s}_{virtual_row_count} of length virtual_row_len.
2. Compute t_i = A \tilde{s}_i for each i.
3. Decompose all t_i into inner_opening_digits.
4. If non-tail, compute inner_opening_payload = B * inner_opening_digits; else set inner_opening_payload = inner_opening_digits.
5. Absorb level context and inner_opening_payload into transcript.
6. Derive JL seed in-transcript, search accepted jl_nonce, and compute public jl_projection = Pi * centered(\tilde{s}).
7. Absorb jl_projection.
8. For each JL lift k:
     - sample omega^(k)
     - collapse 256 JL rows into one coefficient vector W_c^(k)
     - convert W_c^(k) into phi^{JL,(k)} via sigma_{-1}
     - compute b^{JL,(k)} = <phi^{JL,(k)}, \tilde{s}>
     - send jl_lift_residuals^(k) = b^{JL,(k)} with zero constant term
     - absorb jl_lift_residuals^(k)
     - sample beta_k
   Aggregate these into phi^{JL} and b^{JL}.
9. Aggregate the statement side into phi^{stmt} and b^{stmt}.
10. Set phi_i = phi_i^{stmt} + phi_i^{JL}, and aggregated_rhs = b^{stmt} + b^{JL}.
11. Compute h_ii = <phi_i, \tilde{s}_i> and h_ij = <phi_i, \tilde{s}_j> + <phi_j, \tilde{s}_i>.
12. Decompose h into linear_garbage_digits.
13. If non-tail, compute linear_garbage_payload = D_out * linear_garbage_digits; else set linear_garbage_payload = linear_garbage_digits.
14. Absorb linear_garbage_payload.
15. Sample amortization challenges c_i and form z = sum_i c_i \tilde{s}_i.
16. Decompose z into z^(0), ..., z^(witness_digit_parts-1).
17. Output next witness:
     - non-tail: z-parts plus auxiliary row inner_opening_digits || linear_garbage_digits
     - tail: z-parts only
18. Output next statement:
     - non-tail: compact reduced plan with amortized_phi = sum_i c_i phi_i and aggregated_rhs
     - tail: no further Labrador statement; terminal checks remain
```

## Optimized Verifier Flow

The verifier follows the same Fiat-Shamir schedule, but it should be understood in the most compact possible way:

- It must replay all transcript events exactly.
- It need not materialize the full JL matrix.
- It need not materialize the per-row `phi_i` if the only downstream object needed is `amortized_phi`.
- It need not rebuild sparse recursive constraints.

So the ideal verifier logic is:

```text
Input:
  current statement S
  level payload (inner_opening_payload, linear_garbage_payload, jl_projection, jl_nonce, jl_lift_residuals, next_witness_norm_sq, config, virtual_row_len, row_split_counts)

Transcript replay:
1. Absorb level context and inner_opening_payload.
2. Replay accepted jl_nonce and thereby the JL seed / matrix stream.
3. Absorb jl_projection.
4. Replay all JL collapse challenges omega^(k), all JL aggregation challenges beta_k,
   all statement aggregation challenges alpha, then absorb linear_garbage_payload, then replay amortization c_i.

Algebra:
5. Reconstruct the JL contribution b^{JL} by restoring each jl_lift_residuals^(k) constant term from jl_projection.
6. Aggregate the statement contribution into b^{stmt}.
7. Set aggregated_rhs = b^{stmt} + b^{JL}.
8. Compute the next-round coefficient object conceptually as
     amortized_phi = sum_i c_i (phi_i^{stmt} + phi_i^{JL}).
   In an optimized verifier, this is built directly, without storing all per-row phi_i.
9. Form the next compact reduced statement
     (virtual_row_count, virtual_row_len, config, c_i, amortized_phi, aggregated_rhs, setup).
10. Set the next norm bound to next_witness_norm_sq.
```

The current code computes `phi_total` first and then `amortized_phi` in `src/protocol/labrador/verifier.rs:166-195` and `src/protocol/labrador/constraints.rs:452-465`. The optimized viewpoint is that `amortized_phi` is the real recursive payload; `phi_total` is merely a temporary way to obtain it.

## Tail Round: What Remains To Check

The tail round is the same algebra, except there is no next recursive Labrador level. So instead of packaging the constraints into another `LabradorReducedConstraintPlan`, the verifier checks the final relations directly.

Tail mode is characterized by:

- `outer_commit_rank = 0`
- no outer `B` matrix
- no outer `D` matrix
- `inner_opening_payload = inner_opening_digits`
- `linear_garbage_payload = linear_garbage_digits`

See `src/protocol/labrador/setup.rs:48-67` and the verifier-side shape checks in `src/protocol/labrador/verifier.rs:290-300`.

### Decomposed objects

Write:

$$ z = \sum_{t=0}^{\mathrm{witness\_digit\_parts}-1} 2^{t \cdot \mathrm{witness\_digit\_bits}} z^{(t)}, $$

$$ t_i = \sum_{u=0}^{aux_digit_parts-1} 2^{u aux_digit_bits} t_i^{(u)}, $$

$$ h_{ij} = \sum_{u=0}^{aux_digit_parts-1} 2^{u aux_digit_bits} h_{ij}^{(u)}. $$

The final witness supplies the `z^{(t)}` rows. The level payload supplies the decomposed `t_i^{(u)}` and `h_{ij}^{(u)}` blocks.

### Tail verifier equations

Let `amortized_phi = sum_i c_i phi_i` and `aggregated_rhs = b^{stmt} + b^{JL}` be the same round aggregates defined above. The final checks are:

#### 1. Opening equation

$$ A\left(\sum_{t=0}^{\mathrm{witness\_digit\_parts}-1} 2^{t \cdot \mathrm{witness\_digit\_bits}} z^{(t)}\right) = \sum_{i=1}^{\mathrm{virtual\_row\_count}} c_i \left(\sum_{u=0}^{\mathrm{aux\_digit\_parts}-1} 2^{u \cdot \mathrm{aux\_digit\_bits}} t_i^{(u)}\right). $$

This is the decomposed form of `A z = sum_i c_i t_i`, corresponding to `src/protocol/labrador/verifier.rs:365-375`.

#### 2. Linear-garbage equation

$$ \left\langle \mathrm{amortized\_phi}, \sum_{t=0}^{\mathrm{witness\_digit\_parts}-1} 2^{t \cdot \mathrm{witness\_digit\_bits}} z^{(t)} \right\rangle = \sum_{i \le j} c_i c_j \left(\sum_{u=0}^{\mathrm{aux\_digit\_parts}-1} 2^{u \cdot \mathrm{aux\_digit\_bits}} h_{ij}^{(u)}\right). $$

This is the decomposed form of the linear-only garbage relation checked in `src/protocol/labrador/verifier.rs:378-395`.

#### 3. Diagonal equation

$$ \sum_{i=1}^{\mathrm{virtual\_row\_count}} \left(\sum_{u=0}^{\mathrm{aux\_digit\_parts}-1} 2^{u \cdot \mathrm{aux\_digit\_bits}} h_{ii}^{(u)}\right) = \mathrm{aggregated\_rhs}. $$

This is the decomposed form of `sum_i h_ii = aggregated_rhs`, checked in `src/protocol/labrador/verifier.rs:398-404`.

#### 4. Norm checks

The verifier also checks:

$$ \|w_{final}\|^2 \le \mathrm{next\_witness\_norm\_sq}, $$

and

$$ \|\mathrm{jl\_projection}\|^2 \le 256 \cdot \beta_{\mathrm{in}}^2, $$

where $\beta_{\mathrm{in}}^2$ is the incoming statement norm bound, matching `src/protocol/labrador/verifier.rs:355-363`.

In an optimized recursive setting, these are the only terminal algebraic facts left.

## Canonical Validation Flow

The canonical bench for all of the handoff tuning in this repo is `examples/profile.rs`.

For the small-coefficient handoff path, the reference run is:

```bash
HACHI_MODE=logbasis cargo run --release --example profile
```

What to look for in that run:

- The normal proof-size summary from `examples/profile.rs`, which reports the final chosen tail and its byte breakdown.
- The `labrador_handoff estimated tail comparison` log line from `src/protocol/labrador_handoff.rs`, which reports:
  - `packed_direct_bytes`
  - `estimated_labrador_tail_bytes`
  - `selected_tail`
- If the estimate chooses Labrador, the later `labrador prove complete` log line and the Labrador tail breakdown show the realized recursive payload.

That is the intended loop for tuning `witness_digit_parts`, the handoff coefficient model, and the stop rule: make the change, run `profile.rs` in `logbasis`, and compare direct packed-tail bytes against the estimated Labrador tail bytes that drove the choice.

## Why The Protocol Is Sound Intuitively

There are three interacting ideas.

### 1. Random linear aggregation

Both on the statement side and on the JL side, many equations are fused into one by Fiat-Shamir challenges. A cheating prover would have to satisfy a random linear combination of equations it did not know in advance.

### 2. Amortization

The witness rows are compressed into

$$ z = \sum_i c_i \tilde{s}_i. $$

The linear-garbage terms `h_{ij}` are exactly what makes this safe: they remember the cross terms that appear when many row-wise inner products are folded into one.

### 3. Recursive closure

After one round, the new witness and the new statement are again of Labrador form:

- a decomposed `z`,
- committed side data `inner_opening_digits`, `linear_garbage_digits`,
- and a new coefficient vector `amortized_phi`.

So the same protocol can be applied again.

## What "Fully Optimized" Means Here

This note is intentionally not describing every current implementation detail literally. It is describing the same protocol in the shape you would actually want for recursion.

The important "keep" vs "remove" split is:

### Keep

- Transcript binding order
- Accepted `jl_nonce` replay
- In-guest JL seed derivation
- Public `jl_projection`
- Same algebraic objects `inner_opening_payload`, `linear_garbage_payload`, `jl_lift_residuals`, `amortized_phi`, `aggregated_rhs`

### Remove

- Full JL matrix materialization
- Full `phi_total` materialization when only `amortized_phi` is needed
- Sparse constraint re-materialization at recursive levels
- Unnecessary recomposition in the tail verifier if the backend can check decomposed equations directly

So the best mental model is:

> Labrador is a protocol for recursively replacing a big linear system over many witness rows by a smaller witness plus one carried-forward coefficient vector `amortized_phi`, while JL collapse gives a transcript-bound certificate that the witness stayed norm-bounded.

## Repo Mapping By Phase

If you want to jump back into the code after reading the math, these are the best entry points.

| Topic | Primary anchors |
| --- | --- |
| Parameter selection and reshaping | `src/protocol/labrador/config.rs:33-76`, `src/protocol/labrador/config.rs:97-255` |
| Commitment matrices `A`, `B`, `D_out` | `src/protocol/labrador/setup.rs:32-79` |
| Main prover round | `src/protocol/labrador/fold.rs:83-200` |
| JL projection and nonce replay | `src/protocol/labrador/johnson_lindenstrauss.rs:246-333` |
| JL lift aggregation | `src/protocol/labrador/aggregation.rs:546-639` |
| Reduced statement aggregation | `src/protocol/labrador/aggregation.rs:667-799` |
| Recursive plan construction | `src/protocol/labrador/constraints.rs:105-245` |
| Main verifier reduction | `src/protocol/labrador/verifier.rs:129-205` |
| Tail verifier checks | `src/protocol/labrador/verifier.rs:302-407` |
| Core witness / statement / proof types | `src/protocol/labrador/types.rs:9-149` |

## One-Sentence Summary

The linear-only Labrador round takes many witness rows and many linear constraints, binds the witness through `inner_opening_payload`, proves a JL sketch of its coefficient norm, packages the cross terms through `linear_garbage_payload`, compresses the witness to one amortized row `z`, and carries forward only the compact recursive state `(amortized_phi, aggregated_rhs, c_i, config, setup)`.
