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

The A rows are reduced in `R=F[Y]/(Phi_D)` before evaluation:

```text
x_i = rem_Phi_D(sum_j A_ij*p_j - sum_col iota(Ch_col)*Y_(col,i));
x_i = 0                  for tag 0;
x_i = q0*KA_i            for tag 1.
```

Each remainder has degree below D. Tag 0 has no auxiliary matrix-row message.
Tag 1 sends exactly D range-checked carry integers per row. The integer lift
and its admission conditions are owned by the small-modulus specification.
The unreduced parity row over the integers is

```text
sum_r B_r(Z)*V_r(Z) - sum_col U_col(Z)*Ch_col(Z)
    = Phi(Z)*Q(Z) + 2*K(Z).
```

Q has 161 coefficients and K has 162. Each clear integer is checked against
its enforced interval before it contributes to a public scalar. Monic integer
division gives Q and an even remainder; K is half that remainder.

## Public linear relation

Fix explicit `alpha,xi,gamma` in F. For both tags set
`u_i[t]=gamma^i*alpha^t` and `g=gamma^n_A`. These are power-product weights,
not independent random field coordinates. Write `<a,b>` for the coefficient
dot product. Integers reduce into F, with negative x mapped to `-|x|`. Define

```text
wA(j,t)      = sum_i <u_i, rem_Phi_D(A_ij*Y^t)>;
wY(col,i,t)  = -<u_i, rem_Phi_D(iota(Ch_col)*Y^t)>;
h(j,t)       = sigma(t)*xi^(t div k)*B_(j*k+t mod k)(xi);
T_par        = sum_col U_col(xi)*Ch_col(xi) + Phi(xi)*Q(xi) + 2*K(xi);
c_A          = 0                                  for tag 0;
c_A          = sum_i gamma^i*q0*KA_i(alpha)        for tag 1;

K_W[l + dc*(t + P_c*j)]       = 2^(b*l)*(wA(j,t) + g*h(j,t)), t<D;
K_Y[t + P_c*(col*n_A+i)]      = wY(col,i,t),                t<D;
c_pub = sum_(j,t<D) off(t)*(wA(j,t) + g*h(j,t)) + c_A + g*T_par.
```

Weights are zero at coefficient tails and image-entry padding. No extra signed
packing factor belongs on wA: p already carries that sign. Acceptance is
`<W,K_W>+<Y,K_Y>=c_pub`, together with the alphabet and clear integer range checks.
Subtracting c_pub gives precisely `sum_i gamma^i*(x_i(alpha)-q0*KA_i(alpha))`
for tag 1, or `sum_i gamma^i*x_i(alpha)` for tag 0, plus g times the unchanged
parity-row residual. The offset has no effect on either public weight.
`RelationPolynomial::evaluate_modulus_at` supplies Phi(xi) from the degree-162
plus trinomial; no Phi_D(alpha) factor enters the matrix-row relation.

### Cached setup offset

The setup fixes `off(t)` through the response interval and packing degree,
independently of digit base. Define in R

```text
S_i = sum_j A_ij;
O   = sum_(t<D) off(t)*Y^t;
H_i = rem_Phi_D(S_i*O).
```

`BinaryClearSetup` constructs and retains H_i in `a_offset_remainders` from
its own matrix and interval.
Its online matrix contribution to c_pub is `sum_i gamma^i*H_i(alpha)`.
`LoweredPublic::new` validates lengths and fold challenges, checks the left
expansion and Q/K ranges (and KA for tag 1), and caches powers, parity row
evaluations and these H evaluations. It performs no matrix scan. A verification
traverses the matrix once, at the response terminal below.

### Dense definition

For `0<=n<=2D-2`, let `rho(n)=(Y^n mod Phi_D)(alpha)`. Write
`Phi_D=Y^D-c*Y^(D/2)+1`, so c=+1 for the minus trinomial and c=-1 for the plus
trinomial. With h=D/2, the O(D) recurrence is

```text
rho(n) = alpha^n                       for n<D;
rho(n) = c*rho(n-h) - rho(n-D)          for D<=n<=2D-2.
<u_i, rem_Phi_D(a*Y^t)> = gamma^i*sum_(s<D) a[s]*rho(s+t).
```

This definition uses neither ring multiplication nor transforms. The dense
response reference, `check_lowered_clear` and the reference root reduction
use it. Dense image weights are tested against the same definition; their
optimized construction uses the shared adjoint implementation below.

### Trace-form adjoint

The verifier crate owns `trace_gram_into`, `trace_gram_inverse` and
`multiplication_adjoint` in `lowered/adjoint.rs` once;
the prover uses that owner. Let `G[s][t]=Tr(Y^(s+t))` in the power basis of R.
For `0<=n<=2D-2`, the nonzero traces are `Tr(Y^0)=2h`, `Tr(Y^h)=c*h`,
`Tr(Y^(2h))=-h`, `Tr(Y^(3h))=-2c*h`. All others are zero. Consequently

```text
(G v)[0]   = 2h*v[0] + c*h*v[h];
(G v)[h]   = c*h*v[0] - h*v[h];
(G v)[r]   = c*h*v[h-r] - h*v[2h-r];
(G v)[h+r] = -h*v[h-r] - 2c*h*v[2h-r],          1<=r<h.
```

Setup rejects with a typed error if 3h is not invertible in F. With
`z=(3h)^-1`, the inverse map is

```text
v[0]     = (u[0] + c*u[h])*z;
v[h]     = (c*u[0] - 2*u[h])*z;
v[h-r]   = (2c*u[r] - u[h+r])*z;
v[2h-r]  = -(u[r] + c*u[h+r])*z,               1<=r<h.
```

Both maps take O(D) operations. Trace associativity gives
`<u, rem_Phi_D(a*y)>=<G(rem_Phi_D(a*G^-1(u))),y>`; equivalently,
`M_a^T(u)=G(rem_Phi_D(a*G^-1(u)))`. Thus
`wA(j,.)=G(sum_i A_ij*G^-1(u_i))` and
`wY(col,i,.)=-G(iota(Ch_col)*G^-1(u_i))`, with products in R.
The prover's transform schedule and compact-table memory are owned by
[the root sumchecks](labinius-root-sumcheck.md#adjoint-weight-construction).

## Structured multilinear evaluation

Split the combined point into digit, coefficient and ring-element blocks
`(rho_l,rho_t,rho_j)`. Set

```text
e_A = sum_(t<D) eq(rho_t,t)*Y^t;
Abar_i = sum_j eq(rho_j,j)*A_ij;
wA~(rho_t,rho_j) = sum_i <u_i, rem_Phi_D(Abar_i*e_A)>;
gadget(rho_l) = sum_l eq(rho_l,l)*2^(b*l).
```

The canonical `TrinomialASetupView` has rows n_A, columns m, zero offset and
setup domain the next power of two of `n_A*m*D`. Its prepared owner takes the
response point's column block rho_j and dense rows
`z_i[s]=<u_i,rem_Phi_D(Y^s*e_A)>`, incorporating the digit gadget in the
prepared scale. Its weight at `setup_address(i,j,s)` is
`gadget(rho_l)*eq(rho_j,j)*z_i[s]`, with padding zero. Matrix order is `i*m+j`.
`materialize_setup_weights` remains the dense reference;
`evaluate_setup_weight_at` evaluates this MLE without scanning the matrix.
The implemented response terminal forms z with one transform adjoint for the
common alpha-power vector, then scales it by each gamma power. The prepared
owner contracts columns in one pass into n_A degree-D accumulators and dots
them with z. It does not materialize the padded setup-weight table for this
contraction. `witness_weight_mle` obtains its matrix contribution through this
same owner. This is the closed form above by commutativity and linearity.

For parity, split rho_t further into component and scalar-coefficient blocks,
with lengths `log2(k),log2(P_c/k)`. Its unchanged terminal contribution is

```text
g * gadget(rho_l)
  * (sum_(s<162) eq(rho_s,s)*sigma_s*xi^s)
  * Btilde(rho_c,rho_j).
```

Btilde extends `r=c+k*j -> B_r(xi)`. Computing that table costs
`O(M_rows*d)` field operations. Restricting s to its natural degree makes
coefficient tails zero.

At the product point `(rho'_t,rho'_e)`, define

```text
e_Y = sum_(t<D) eq(rho'_t,t)*Y^t;
cbar_i = sum_col eq(rho'_e,col*n_A+i)*iota(Ch_col);
K_Y~(rho') = -sum_i <u_i, rem_Phi_D(cbar_i*e_Y)>.
```

This contracts the flat entry index directly, including rank-three padding.
Neither terminal uses the former alpha-power factorization of matrix or image
weights. Dense and terminal routes are tested against schoolbook reduction.
The arbitrary-precision integer oracle checks the transmitted KA for tag 1
and a zero remainder modulo P for tag 0.

## Witness binding order

The lowered helpers take explicit challenges; their consumer binds these
messages in the following order. Exact byte grammar is owned by the
[root reduction](labinius-root-reduction.md) and the small-modulus specification.

| Profile | Before alpha | Before xi | Before gamma |
| --- | --- | --- | --- |
| Tag 0 | Y, W, U, folds, Q, K | Same messages and alpha | All evaluated rows |
| Tag 1 | Y, W, U, folds, KA, Q, K | Same messages and alpha | All evaluated rows |

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

- A false remainder row survives uniform alpha with probability at most
  `(D-1)/|F|`, equal to `647/|F|` at D648. W must be fixed before alpha;
  tag 1 also fixes KA before alpha. Its derivation-bias and extraction ledger
  delta are recorded only in the small-modulus specification.
- A parity polynomial row false modulo P survives uniform xi with probability
  at most `322/|F|`. W, Q and K must be fixed before xi. If it instead holds as
  a polynomial identity modulo P, the alphabet, clear range checks and admitted
  inequality `H+3*2^(e_Q-1)+2*2^(e_K-1)<P` bound every integer residual coefficient
  strictly below P. Therefore it holds over the integers. Reducing modulo 2
  and Phi gives the F162 parity equation.
- Once the evaluated rows are fixed, batching by uniform gamma adds at most
  `n_A/|F|`. The combined union bound is at most
  `(D-1+322+n_A)/|F|`, before other protocol losses.
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
remainder residual batching, independently computed integer carries,
trace-map inverses and schoolbook adjoints for both trinomials, complemented digits, both interval endpoints, malformed inputs and zero-weight
tails. These checks support implementation agreement with the relation above;
they do not add a proof protocol.
