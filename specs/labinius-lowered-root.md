# LaBinius lowered root relation

Status: implemented
Book-chapter: book/src/how/architecture.md
Tracking: https://github.com/LayerZero-Labs/akita/issues/45

## Scope

The opt-in `lowered` modules in `akita-labinius-verifier` and
`akita-labinius-prover` express the clear root endpoints as a digit alphabet,
clear integer range checks and one coefficient-field linear relation.
They take explicit field challenges. They define no transcript, sumcheck,
commitment to the digit table, polynomial opening or wire encoding.
The shared-prime clear endpoint and its encodings are unchanged.

This document is the normative definition of the lowered statement. The
[clear opening](labinius-clear-opening.md),
[root admission](labinius-root-admission.md) and
[field switch](labinius-field-switch.md) specifications give its surrounding
contracts. `LabiniusRootEncoding` supplies the enforced ranges and the response
and image table lengths; this document fixes the relation over those tables.

## Geometry and admission

Write `d=162`, `Phi(Z)=Z^162+Z^81+1`, `D=d*k`, `k` in `{1,2,4}`,
and `Phi_D(Y)=Y^D+s*Y^(D/2)+1`, where `s` is the commitment modulus's middle
coefficient. The scalar ring is over the integers, and the commitment ring is
`R=F[Y]/(Phi_D)` for the coefficient prime field F of modulus P.
Both admitted root profiles have `D=648` and P128. The tag-1 foreign-modulus
relation and its integer argument are specified in
[the small-modulus specification](labinius-small-modulus-root.md).

Let `m=setup.m()`, `M_rows=k*m`, `C=setup.columns()` and `n_A=setup.n_a()`.
`LoweredRootLayout::new` requires agreement with `LabiniusRootShape` on the
coefficient prime, commitment and scalar degrees, packing degree, rank, ring
elements per column, columns, scalar rows, challenge profile and accepted
response interval. The shape admission is mandatory; there is no unchecked
constructor. Both m and the response digit count dc are powers of two.

Let `enc=shape.derive_encoding(base)`, `rv=enc.response()`, `e_v=rv.bits()`,
`dc=rv.digit_count()`, `off=rv.offset()` and `b=base.bits()`.
The accepted interval is exactly `rv.interval()=[-off,off-1]` and must equal
the setup interval. Quotient and carry intervals are `enc.quotient().interval()`
and `enc.carry().interval()`. Their actual enforced magnitudes pass the shape's
no-wrap admission.

## Response and image tables

A response row `r=j*k+c` is scalar component c of ring element j.
For `t=s*k+c<D`, let `sigma(t)=+1` when `k=1` or s is even and `-1` otherwise.
Signed packing gives `p[j][t]=sigma(t)*v[j*k+c][s]`.
Let `off(t)=off` at positive signs and `off-1` at negative signs.
The digit table represents `u[j][t]=p[j][t]+off(t)` in `[0,2^e_v)`.
Thus `v=sigma(t)*(u-off(t))` has exactly the same accepted scalar interval at
both signs. At negative signs, each digit is the complement of the corresponding
unflipped digit of `v+off`.

Set `P_c=D.next_power_of_two()`. The canonical
`TrinomialResponseLayout::new(poly_D,P_c,m,dc,0,m*P_c*dc)` owns addresses:

```text
W[l + dc*(t + P_c*j)] = digit_l(u[j][t]).
```

Here `poly_D` is the matching `RelationPolynomial` with plus or minus trinomial
sign. The domain length is exactly `m*P_c*dc`, with digit, coefficient and ring
index blocks aligned to binary bits. Coefficient tails `t>=D` have honest digit
zero and public weight zero. Every table position is alphabet-checked against
`[0,2^b)`. There is no selector or zero-tail acceptance check.

For image entry `e=col*n_A+i`, flatten `commitment.images[e]` as
`Y[t+P_c*e]=images[e].coefficients()[t]` for `t<D`. Coefficient tails and final
padding are honestly zero. The image domain is the next power of two of
`P_c*C*n_A`; its entries are arbitrary elements of F.
All multilinear extensions use little-endian indices: the first point
coordinate binds the lowest index bit.

## Polynomial rows

Let `Ch_col` be the integer polynomial of a validated fold challenge,
`iota=embed_scalar`, `B_r` the 0/1 integer lift of
`eq_F162(claim.point[..row_vars],r)`, and `U_col` the 0/1 lift of `u[col]`.
The left expansion of u against the remaining claim coordinates is checked
separately by the existing clear endpoint helper.

For tag 0, the unreduced A rows over F are

```text
sum_j A_ij(Y)*p_j(Y) - sum_col iota(Ch_col)(Y)*Y_(col,i)(Y)
    = Phi_D(Y)*QA_i(Y).
```

Every side has degree at most `2D-2`, and each clear QA row has exactly `D-1`
coefficients. Tag 1 adds `q0*KA_i(Y)` to the right side, with D range-checked
integers per row; the canonical auxiliary witness contains QA followed by KA.
The unreduced parity row over the integers is

```text
sum_r B_r(Z)*V_r(Z) - sum_col U_col(Z)*Ch_col(Z)
    = Phi(Z)*Q(Z) + 2*K(Z).
```

Q has 161 coefficients and K has 162. Each clear integer is checked against
its enforced interval before it contributes to a public scalar. Monic integer
division gives Q and an even remainder; K is half that remainder.

## Public linear relation

Fix explicit `alpha,xi,gamma` in F. Put `g=gamma^n_A` and
`Abar_j=sum_i gamma^i*A_ij(alpha)`. Integers reduce into F, with negative x
mapped to `-|x|`. Define the public weights and constant once as follows:

```text
K_W(j,l,t) = 2^(b*l) * (Abar_j*alpha^t
                      + g*sigma(t)*xi^s*B_(j*k+c)(xi)), t=s*k+c<D;
K_Y(e,t)   = -gamma^i*iota(Ch_col)(alpha)*alpha^t, e=col*n_A+i, t<D;
c_pub      = sum_(j,t<D) off(t)*(Abar_j*alpha^t
                               + g*sigma(t)*xi^s*B_(j*k+c)(xi))
             + sum_i gamma^i*Phi_D(alpha)*QA_i(alpha)
             + g*(sum_col U_col(xi)*Ch_col(xi)
                  + Phi(xi)*Q(xi) + 2*K(xi)).
```

Weights are zero at all coefficient tails and image padding. For tag 1, c_pub additionally contains `sum_i gamma^i*q0*KA_i(alpha)`;
K_W and K_Y use the same formulas with the reduced A coefficients. Acceptance is
`<W,K_W>+<Y,K_Y>=c_pub`, together with the alphabet and clear integer range checks.
Subtracting c_pub gives precisely the gamma combination of the A-row residuals
plus g times the parity-row residual. The offset has no effect on either public
weight. `RelationPolynomial::evaluate_modulus_at` supplies both modulus values,
using the degree-162 plus trinomial for Phi.

`LoweredPublic` validates lengths and fold challenges, checks the left expansion
and Q/K ranges (and KA for tag 1), and caches the row evaluations and powers. The dense weight path
uses direct Horner evaluations of matrix coefficients, independently of the
canonical setup-weight preparation path.

## Structured multilinear evaluation

For the A part of the witness weight, the canonical `TrinomialASetupView` has
rows n_A, columns m, zero offset and setup domain equal to the next power of two
of `n_A*m*D`. `prepare(rho,alpha,[gamma^i],[2^(b*l)])` contracts the complete
response point. The inner product of `materialize_setup_weights()` and the
matrix coefficients at `setup_address(i,j,t)` gives the desired A contribution.
The prepared modulus value is separate; no modulus factor belongs in this
inner product. Matrix order is `i*m+j`, as in the existing matrix action.
This verifier calculation costs linear time in the padded setup domain. A
later protocol can replace it with a setup-contribution claim.

Split rho into digit, component, scalar-coefficient and ring-element blocks,
with lengths `log2(dc),log2(k),log2(P_c/k),log2(m)`. The parity contribution is

```text
g * (sum_l eq(rho_l,l)*2^(b*l))
  * (sum_(s<162) eq(rho_s,s)*sigma_s*xi^s)
  * Btilde(rho_c,rho_j).
```

Btilde extends the table `r=c+k*j -> B_r(xi)`. Computing that table costs
`O(M_rows*d)` field operations. Restricting s to its natural degree makes the
coefficient tails zero. The image weight factors as

```text
(sum_(t<D) eq(rho_t,t)*alpha^t)
  * MLE_(e padded)(-gamma^i*iota(Ch_col)(alpha)).
```

The production A quotient constructor uses the original and conjugate
trinomial transforms. The tests independently use schoolbook products and
monic division; tag 1 also has an arbitrary-precision integer oracle that
checks the transmitted QA and KA without the production weights.

## Witness binding order

The lowered helpers take explicit challenges; their consumer binds these
messages in the following order. Exact byte grammar is owned by the
[root reduction](labinius-root-reduction.md) and the small-modulus specification.

| Profile | Before alpha | Before xi | Before gamma |
| --- | --- | --- | --- |
| Tag 0 | Y, W, U, folds, QA, Q, K | Same messages and alpha | All evaluated rows |
| Tag 1 | Y, W, U, folds, QA, KA, Q, K | Same messages and alpha | All evaluated rows |

## Conditional soundness and challenge order

These are obligations for the later proof protocol, not proof obligations
discharged by these arithmetic helpers.

The setup and statement, the evaluation claim, the left expansion U, the fold
challenges and the image table Y (or a binding commitment to it) MUST be fixed
and authenticated before alpha and xi are drawn, together with the witness
messages each bullet names. Every one of these values enters the polynomials
the challenges test: an image table chosen after alpha, for example, could be
adjusted to satisfy a false A row. All evaluated rows MUST be fixed before
gamma.

- A false A polynomial row survives uniform alpha with probability at most
  `(2D-2)/|F|`, equal to `1294/|F|` at D648. QA and W must be fixed before alpha; tag 1 also fixes KA before alpha.
  The degree bound is unchanged. Its derivation-bias and extraction ledger delta
  are recorded only in the small-modulus specification.
- A parity polynomial row false modulo P survives uniform xi with probability
  at most `322/|F|`. W, Q and K must be fixed before xi. If it instead holds as
  a polynomial identity modulo P, the alphabet, clear range checks and admitted
  inequality `H+3*2^(e_Q-1)+2*2^(e_K-1)<P` bound every integer residual coefficient
  strictly below P. Therefore it holds over the integers. Reducing modulo 2
  and Phi gives the F162 parity equation.
- Once the evaluated rows are fixed, batching by uniform gamma adds at most
  `n_A/|F|`. The combined union bound is at most
  `(2D-2+322+n_A)/|F|`, before other protocol losses.
- An in-alphabet nonzero digit in a tail or padding position does not change
  acceptance: its weight is zero. Out-of-alphabet tails still reject.

One fixed set of field challenges is a probabilistic polynomial identity test,
not deterministic equivalence for every adversarial input. In particular, zero
or otherwise degenerate explicit challenges can hide false rows. Differential
regression tests draw seeded field challenges after constructing the witnesses;
they do not establish a complete security theorem.

## Source binding and projection

**Tag-0 obligation sketch; not yet a complete theorem.** For tag 1 use the
cancellation lemma and conditional composition obligation in the small-modulus
specification. Two accepting tag-0 transcripts
that differ in one fold-challenge coordinate should yield an extracted
occurrence `(a,s)` satisfying `A*a=s*Y_col mod P` and
`B(a mod 2)=U_col*(s mod 2)`, with `||a||_inf<=Delta`.
Multiplication by s has integer operator norm at most `2*Gamma_inf`.
The challenge difference is nonzero modulo 2 in F162, hence invertible in that
field. It need not be invertible modulo P, so extraction cannot divide by s
in the coefficient field.

Compare two extracted occurrences `(a,s)` and `(a',s')` over the integers using
`s'*a-s*a'`. Its infinity norm is at most
`eta_A=4*Gamma_inf*Delta<P`, and its A image vanishes modulo P. Either this is
a nonzero short kernel vector, excluded under the SIS assumption at the admitted
rank for a matrix derived under the [setup contract](labinius-setup-contract.md),
or it is zero over the integers and the
rational sources agree. Modulo-two projection then authenticates the shared
F162 source. The host claim is recovered by projecting this source onto its
low 128 coordinates, using the binary coefficients of the field-switch
identities. Establishing that these extracted objects satisfy all consumer
semantics remains part of composition.

The [root reduction](labinius-root-reduction.md) binds the admitted-root
identity and fixes the statement, witness and challenge order. Its conditional
soundness ledger composes these arithmetic reductions. Complete frontend
composition, knowledge extraction and its loss, the SIS reduction and
Fiat–Shamir security remain open as recorded there. Arithmetic tests and shape
admission alone do not resolve them.

## Regression evidence

The lowered integration tests exercise all three digit bases at the small
admitted `(log_num_cells,log_fold_width,lambda_fold)=(4,1,128)` geometry, over
both an explicit random matrix and a seed-derived admitted setup.
They compare honest clear endpoints, dense and structured weights, decoded
residual batching, independently computed integer and field quotients,
complemented digits, both interval endpoints, malformed inputs and zero-weight
tails. These checks support implementation agreement with the relation above;
they do not add a proof protocol.
