# LaBinius lowered root relation

Status: implemented
Book-chapter: book/src/how/architecture.md
Tracking: https://github.com/LayerZero-Labs/akita/issues/45

## Scope

The opt-in `lowered` modules in `akita-labinius-verifier` and
`akita-labinius-prover` express the clear root endpoints as two digit tables
with one alphabet, clear integer range checks and one linear relation over a
field pair. For a prime claim they add a third table, of challenge-field
elements, and one more row of that relation. They take explicit field
challenges. They define no transcript,
sumcheck, commitment to either digit table, polynomial opening or wire
encoding. The clear endpoint is arithmetic modulo the commitment prime `q`;
the base field F and the challenge field E enter only in this lowered
relation.

This document is the normative definition of the lowered statement. The
[clear opening](labinius-clear-opening.md),
[root admission](labinius-root-admission.md) and
[field switch](labinius-field-switch.md) specifications give its surrounding
contracts. `LabiniusRootEncoding` supplies the enforced ranges and the response
and image table lengths; this document fixes the relation over those tables.

## Geometry and admission

Write `d=162`, `Phi(Z)=Z^162+Z^81+1`, `D=d*k`, `k` in `{1,2,4}`,
and `Phi_D(Y)=Y^D+s*Y^(D/2)+1`, where `s` is the commitment modulus's middle
coefficient. The scalar ring is over the integers. Commitments live in
`Z_q[Y]/(Phi_D)`. The rows below are identities in `R=F[Y]/(Phi_D)` for a
prime base field F of characteristic P, between integers reduced modulo P.
They are tested at challenges from a field E containing F, into which every
public integer and weight is lifted. Neither field needs a root of unity. The
parameter layer names one profile, `D648Q25BoundedW46`, with `D=648` and
`q=33568993`.

Let `m=setup.m()`, `M_rows=k*m`, `C=setup.columns()` and `n_A=setup.n_a()`.
`LoweredRootLayout::new::<F,E,_,_>(setup, shape)` is where the field pair
enters. It calls `shape.derive_encoding(P)`, which admits P by the unit
condition on fold-challenge differences and the two no-wrap inequalities of
[root admission](labinius-root-admission.md#proof-prime-admission-and-clear-integer-ranges);
a base field that fails cannot build a layout. It admits E exactly when
every field-challenge site of the reduction can be priced
([root admission](labinius-root-admission.md#challenge-field-admission)); a
challenge field that is too small cannot build one either. It then requires
agreement with
`LabiniusRootShape` on the commitment modulus, commitment and scalar degrees,
packing degree, rank, ring elements per column, columns, scalar rows,
challenge profile and accepted response interval. The shape admission is
mandatory; there is no unchecked constructor. An explicit clear setup paired
with a shape it does not match rejects here.

Let `enc=layout.encoding()`. The accepted interval is
`enc.response_interval()=[L,U]`, the balanced base-16 range of
`n=enc.response_digit_count()` digits, with diameter `U-L=16^n-1`; it must
equal the setup interval. The digit axis of the table has
`dc=enc.response_digit_slots()` slots, the next power of two of n. At the
admitted geometries `[L,U]=[-2184,1911]`, `n=3` and `dc=4`. The image digit
axis has `n_Y=enc.image_digit_count()` weighted digits in
`dc_Y=enc.image_digit_slots()` slots, with offset `B_T=enc.image_offset()`;
for the profile's q these are 7, 8 and 143165576. Quotient and carry
intervals are `enc.quotient().interval()`, `enc.carry().interval()` and
`enc.a_carry().interval()`, two's-complement ranges whose enforced magnitudes
passed the no-wrap admission.

## Response and image tables

A response row `r=j*k+c` is scalar component c of ring element j.
For `t=s*k+c<D`, let `sigma(t)=+1` when `k=1` or s is even and `-1` otherwise.
Signed packing gives `p[j][t]=sigma(t)*v[j*k+c][s]`.
Let `off(t)=-L` at positive signs and `off(t)=U` at negative signs.
The digit table represents `u[j][t]=p[j][t]+off(t)` in `[0,16^n-1]`.
Thus `v=sigma(t)*(u-off(t))` has exactly the accepted scalar interval `[L,U]`
at both signs. At positive signs each stored digit is the balanced digit of v
plus 8; at negative signs it is the complement `15-x` of the stored digit of
`v-L`.

Set `P_c=D.next_power_of_two()`. The canonical
`TrinomialResponseLayout::new(poly_D,P_c,m,dc,0,m*P_c*dc)` owns addresses:

```text
W[l + dc*(t + P_c*j)] = digit_l(u[j][t])   for l<n;
W[l + dc*(t + P_c*j)] = 0                  for n<=l<dc.
```

Here `poly_D` is the matching `RelationPolynomial` with plus or minus trinomial
sign. The domain length is exactly `m*P_c*dc`, with digit, coefficient and ring
index blocks aligned to binary bits. Coefficient tails `t>=D` and the padding
digit slots `l>=n` have honest digit zero and public weight zero. Every table
position is alphabet-checked against `[0,15]`. There is no selector or
zero-tail acceptance check.

For image entry `e=col*n_A+i` and `t<D`, write `T[e][t]` for the image
coefficient. For an honest prover it is the canonical residue
`commitment.images[e*D+t]` in `[0,q-1]`. The image digit table stores
`T+B_T` with the addressing of the response table, digit innermost, then the
padded coefficient, then the entry:

```text
Y[l + dc_Y*(t + P_c*e)] = digit_l(T[e][t] + B_T)   for l<n_Y;
Y[l + dc_Y*(t + P_c*e)] = 0                        for n_Y<=l<dc_Y.
```

Coefficient tails `t>=D`, padding digit slots and final padding have honest
digit zero and public weight zero. The image domain is the next power of two
of `dc_Y*P_c*C*n_A`. Every position is alphabet-checked against `[0,15]`, so
a table that passes decodes to integers T in `[-B_T, 16^n_Y-1-B_T]`, of
magnitude at most `B_T=8*(16^n_Y-1)/15`. The count `n_Y` is the least whose
positive reach `7*(16^n_Y-1)/15` covers `q-1`. The prover's
`lowered::encode_image` is the one function that builds this table from a
clear commitment.
All multilinear extensions use little-endian indices: the first point
coordinate binds the lowest index bit.

## Polynomial rows

Let `Ch_col` be the integer polynomial of a validated fold challenge,
`iota=embed_scalar`, `B_r` the 0/1 integer lift of
`eq_F162(claim.point[..row_vars],r)`, and `U_col` the 0/1 lift of `u[col]`.
The left expansion of u against the remaining claim coordinates is checked
separately by the existing clear endpoint helper.

The A rows are reduced modulo `Phi_D` before evaluation:

```text
x_i = rem_Phi_D(sum_j A_ij*p_j - sum_col iota(Ch_col)*T_(col,i));
x_i = q*KA_i.
```

Each remainder has degree below D, with A read as residues in `[0,q)`. For an
honest prover the integer remainder is divisible by q coefficientwise, because
the identity modulo `(q,Phi_D)` is the commitment relation. The prover sends
exactly D range-checked carry integers per row; there is no matrix-quotient
message. The honest bound and the no-wrap condition are owned by
[root admission](labinius-root-admission.md#commitment-row-carry).

The prover's `lowered::a_relation_carry` computes the matrix remainder with
limb-prime transforms and signed CRT when D is 648, the middle coefficient is
-1, the reference product limit `D*q*max|p| <= i64::MAX` holds, and the checked
bound `B = 3*m*D*(q-1)*max|p|` permits a product of admitted limb primes greater
than `2B`. The matrix entries remain the integers in `[0,q)`; CRT recovers the
same integer remainder as `matrix_row_remainder`. Other cases use
`matrix_row_remainder`. The image remainder is computed over the integers with
`folded_image_remainder`. It checks divisibility by q and the carry range; a
failure returns `InvalidProof`.

The unreduced parity row over the integers is

```text
sum_r B_r(Z)*V_r(Z) - sum_col U_col(Z)*Ch_col(Z)
    = Phi(Z)*Q(Z) + 2*K(Z).
```

Q has 161 coefficients and K has 162. Each clear integer is checked against
its enforced interval before it contributes to a public scalar. Monic integer
division gives Q and an even remainder; K is half that remainder.

The A rows are part of every relation. The parity row is present exactly
when the statement has a binary claim (`LoweredParity`); without one there
is no U, Q, K or xi.

## Prime row

This section applies exactly when the statement has a prime claim
(`LoweredPrime`).

**Packed source.** Write `src[j][col]` for packed source ring element j of
column col, the integer polynomial `pack_source_column` commits to. For
`t=s*k+c<D` its coefficient t is `sigma(t)` times bit s of the embedded
source word at index `col*M_rows + j*k + c`; coordinates beyond the embedded
width are zero. The response is its fold,
`p_j = sum_col iota(Ch_col)*src[j][col]` in `Z[Y]/(Phi_D)`.

**Prime left opening.** For a public point `r_ring` in `E^(log2 m)` put
`e_j=eq(r_ring,j)`. The table holds one element of `S_E=E[Y]/(Phi_D)` per
column, coefficient innermost and column outermost:

```text
uP[col] = sum_j e_j*src[j][col]                    in S_E;
uP[t + P_c*col] = coefficient t of uP[col]         for t<D;
uP[t + P_c*col] = 0                                for D<=t<P_c.
```

The table has `P_c*C` entries, `enc.prime_table_log_len()` in logarithm, and
no padding columns. The prover's `lowered::prime_left_opening` is the one
function that builds it: every set source bit adds or subtracts one `e_j`,
so the cost is one addition in E per set bit, with columns independent.
Entries are arbitrary elements of E. No alphabet or range is checked on
them, and the coefficient tails carry no weight and are not constrained.

**Row.** Folding commutes with the E-linear map `sum_j e_j*(.)`:

```text
sum_j e_j*p_j - sum_col iota(Ch_col)*uP[col] = 0      in S_E.
```

Both sides are reduced modulo `Phi_D`, so the row has degree below D and is
tested at the same alpha as the A rows. The response side needs no
reduction: each `p_j` already has degree below D. There is no carry and no
quotient, because the row is an identity over the field and is never lifted
to the integers. With

```text
wP(j,t)           = e_j*alpha^t,                              t<D;
K_P[t + P_c*col]  = (iota(Ch_col)*Y^t mod Phi_D)(alpha),      t<D;
y_P               = <uP,K_P>,
```

the row at alpha reads `sum_(j,t<D) wP(j,t)*p[j][t] = y_P`. `K_P` is the
image weight `wY` without its sign and its power of gamma, one block per
column instead of one per image entry. `prime_row_weights` builds it with
one run of the shift recurrence per column.

**Value claim.** The claim
`v = sum_col eq(r_col,col)*<omega,uP[col]>` is the second linear claim on
the same table, with weights `K_v[t + P_c*col] = eq(r_col,col)*omega[t]`
for `t<D` (`PrimeClaim::value_weights`).

## Public linear relation

Fix explicit `alpha,gamma` in E, `xi` in E when the relation has the parity
row and `eta` in E when it has the prime row. Set
`u_i[t]=gamma^i*alpha^t` and `g=gamma^n_A`. These are power-product weights,
not independent random field coordinates. Write `<a,b>` for the coefficient
dot product. Integers reduce into F and so into E, with negative x mapped to
`-|x|`. Define

```text
wA(j,t)      = sum_i <u_i, rem_Phi_D(A_ij*Y^t)>;
wY(col,i,t)  = -<u_i, rem_Phi_D(iota(Ch_col)*Y^t)>;
h(j,t)       = sigma(t)*xi^(t div k)*B_(j*k+t mod k)(xi);
T_par        = sum_col U_col(xi)*Ch_col(xi) + Phi(xi)*Q(xi) + 2*K(xi);
c_A          = sum_i gamma^i*q*KA_i(alpha);
pow(l)       = 16^l for l<n, and 0 for n<=l<dc;
pow_Y(l)     = 16^l for l<n_Y, and 0 for n_Y<=l<dc_Y;

K_W[l + dc*(t + P_c*j)]               = pow(l)*(wA(j,t) + g*h(j,t)),  t<D;
K_Y[l + dc_Y*(t + P_c*(col*n_A+i))]   = pow_Y(l)*wY(col,i,t),         t<D;
c_pub = sum_(j,t<D) off(t)*(wA(j,t) + g*h(j,t)) + c_A + g*T_par
        + B_T*sum_(col,i,t<D) wY(col,i,t).
```

This is the relation with the parity row and without the prime row.
Without the parity row every term with the factor g is absent. With the
prime row,

```text
K_W[l + dc*(t + P_c*j)] += pow(l)*eta*wP(j,t),  t<D;
c_pub                   += eta*sum_(t<D) off(t)*alpha^t,
```

the constant using `sum_j e_j = 1`, and acceptance becomes
`<W,K_W>+<Y,K_Y>-eta*<uP,K_P>=c_pub` together with the value claim
`<uP,K_v>=v`. `LoweredPublic::response_claim(y_Y, y_P)` returns the linear
claim of the response table, `c_pub-y_Y` plus `eta*y_P` with the prime row,
and rejects a `y_P` that does not match the relation's rows.

Weights are zero at coefficient tails, padding digit slots and image-entry
padding. No extra signed
packing factor belongs on wA: p already carries that sign. Without the prime
row acceptance is
`<W,K_W>+<Y,K_Y>=c_pub`, together with the alphabet check on both tables and
the clear integer range checks. Both inner products are over stored digits:
`<Y,K_Y>=sum wY*(T+B_T)`, and the last term of c_pub removes the image offset
as the first removes the response offsets.
Subtracting c_pub gives precisely `sum_i gamma^i*(x_i(alpha)-q*KA_i(alpha))`
plus g times the unchanged parity-row residual, plus eta times the prime-row
residual at alpha when that row is present. Neither offset has an effect
on a public weight.
`RelationPolynomial::evaluate_modulus_at` supplies Phi(xi) from the degree-162
plus trinomial; no Phi_D(alpha) factor enters the matrix-row relation.

`LoweredPublic::new` computes the image-offset sum without the weight table.
By the identity of the structured evaluation below with every equality
weight equal to one, `sum_(t<D) (B*Y^t mod Phi_D)(alpha) = <B,z1>` for
`z1[s]=(O1*Y^s mod Phi_D)(alpha)` and `O1=sum_(t<D) Y^t`, so

```text
sum_(col,i,t<D) wY(col,i,t) = -(sum_(i<n_A) gamma^i)*sum_col <iota(Ch_col), z1>:
```

one shift recurrence and C sparse dot products.

### Cached setup offset

The setup fixes `off(t)` through the response interval and packing degree.
Define over the integers, reading A as its residues in `[0,q)`,

```text
S_i = sum_j A_ij;
O   = sum_(t<D) off(t)*Y^t;
H_i = rem_Phi_D(S_i*O).
```

`BinaryClearSetup` constructs and retains H_i in `a_offset_remainders` from
its own matrix and interval.
Its online matrix contribution to c_pub is `sum_i gamma^i*H_i(alpha)`.
`LoweredPublic::new` takes the challenges alpha and gamma
(`LoweredChallenges`), an optional `LoweredParity` (the frontend's claim, U,
Q, K and xi) and an optional `LoweredPrime` (`r_ring` and eta). It validates
lengths and fold challenges and the KA range; with the parity input it
checks the left expansion and the Q and K ranges; it caches powers, the
parity and prime row data and these H evaluations. It is generic in the
field the challenges live in and is instantiated at E. It performs no matrix
scan. A verification traverses the matrix once, at the response terminal
below.

### Dense definition

For `0<=n<=2D-2`, let `rho(n)=(Y^n mod Phi_D)(alpha)`. Write
`Phi_D=Y^D-c*Y^(D/2)+1`, so c=+1 for the minus trinomial and c=-1 for the plus
trinomial. With h=D/2, the O(D) recurrence is

```text
rho(n) = alpha^n                       for n<D;
rho(n) = c*rho(n-h) - rho(n-D)          for D<=n<=2D-2.
<u_i, rem_Phi_D(a*Y^t)> = gamma^i*sum_(s<D) a[s]*rho(s+t).
```

This definition uses neither ring multiplication nor transforms.

### Shift recurrence

Every weight above is `g_t=(B*Y^t mod Phi_D)(alpha)` for a public polynomial
B of degree below D: `sum_i gamma^i*A_ij` for `wA(j,.)`, and `iota(Ch_col)`
for `wY(col,i,.)` up to the factor `-gamma^i`. For a reduced X,
`Y*X=(Y*X mod Phi_D)+top(X)*Phi_D` with `top(X)` the coefficient of
`Y^(D-1)`, so

```text
top_t   = b[D-1-t] + (c*top_(t-h) if t>=h else 0);
g_0     = B(alpha);
g_(t+1) = alpha*g_t - top_t*Phi_D(alpha).
```

`akita_algebra::ring::shifted_remainder_evaluations` computes all D values in
O(D) operations of E, with no transform and no root of unity, and is
tested against the dense definition. It replaces the trace-form adjoint
(`trace_gram_into`, `trace_gram_inverse`, `multiplication_adjoint`) and the
proof-field transform of the matrix. The compact-table memory is owned by
[the root sumchecks](labinius-root-sumcheck.md#weight-construction).

## Structured multilinear evaluation

Split the combined point into digit, coefficient and ring-element blocks
`(rho_l,rho_t,rho_j)`. Set

```text
e_A = sum_(t<D) eq(rho_t,t)*Y^t;
Abar_i = sum_j eq(rho_j,j)*A_ij;
wA~(rho_t,rho_j) = sum_i <u_i, rem_Phi_D(Abar_i*e_A)>;
gadget(rho_l) = sum_(l<dc) eq(rho_l,l)*pow(l).
```

`witness_weight_mle` forms `z[s]=(e_A*Y^s mod Phi_D)(alpha)` by one run of
the shift recurrence, so that
`<u_i,rem_Phi_D(Abar_i*e_A)>=gamma^i*sum_s Abar_i[s]*z[s]`. It contracts the
residue matrix in one pass, row by row, into D slots of E holding `Abar_i`,
dots them with z and scales by `gamma^i` and the gadget. Matrix order is
`i*m+j`. It does not materialize a padded setup-weight table. The layout
still constructs the canonical `TrinomialASetupView` (rows n_A, columns m,
zero offset, domain the next power of two of `n_A*m*D`) for its work-cap
admission checks.

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

The prime row adds one more product to the response terminal:

```text
eta * gadget(rho_l) * eq(rho_j,r_ring) * sum_(t<D) eq(rho_t,t)*alpha^t.
```

Split the image point into digit, coefficient and entry blocks
`(rho'_l,rho'_t,rho'_e)` and define

```text
e_Y = sum_(t<D) eq(rho'_t,t)*Y^t;
cbar_i = sum_col eq(rho'_e,col*n_A+i)*iota(Ch_col);
gadget_Y(rho'_l) = sum_(l<dc_Y) eq(rho'_l,l)*pow_Y(l);
K_Y~(rho') = -gadget_Y(rho'_l)*sum_i <u_i, rem_Phi_D(cbar_i*e_Y)>.
```

This contracts the flat entry index directly, including entry padding when
`C*n_A` is not a power of two.
Neither terminal uses the former alpha-power factorization of matrix or image
weights. The image terminal uses the same z at `rho'_t`.

Split a point of the prime table into coefficient and column blocks
`(rho''_t,rho''_col)`, of lengths `log2(P_c)` and `log2(C)`. The two weights
on that table have the terminals

```text
e_P = sum_(t<D) eq(rho''_t,t)*Y^t;
K_P~(rho'') = sum_col eq(rho''_col,col)*(iota(Ch_col)*e_P mod Phi_D)(alpha);
K_v~(rho'') = eq(rho''_col,r_col)*sum_(t<D) eq(rho''_t,t)*omega[t].
```

`prime_row_weight_mle` computes the first with the same z at `rho''_t` and
one sparse contraction per column, `PrimeClaim::value_weight_mle` the second
in D multiplications. The root-reduction end-to-end test checks all four
terminals against the dense weights the prover sums over.

## Witness binding order

The lowered helpers take explicit challenges; their consumer binds these
messages in the following order. Exact byte grammar is owned by the
[root reduction](labinius-root-reduction.md).

| Before the folds | Before alpha | Before xi | Before gamma | Before eta |
| --- | --- | --- | --- | --- |
| Y, U, uP | Y, W, U, uP, folds, KA, Q, K | Same messages and alpha | All evaluated rows | Same rows and gamma |

U, Q, K and xi exist only with the parity row; uP and eta only with the
prime row.

## Conditional soundness and challenge order

These are obligations for the later proof protocol, not proof obligations
discharged by these arithmetic helpers.

The setup and statement, the evaluation claims, the left expansion U, the
prime left opening uP, the fold
challenges and the image table Y (or a binding commitment to it) MUST be fixed
and authenticated before alpha and xi are drawn, together with the witness
messages each bullet names. U and uP MUST be fixed before the fold
challenges: each is an opening of the committed columns that the fold then
tests, and [the prime opening](#prime-opening-and-the-unit-condition) compares
children that share uP. Every one of these values enters the polynomials
the challenges test: an image table chosen after alpha, for example, could be
adjusted to satisfy a false A row. All evaluated rows MUST be fixed before
gamma.

- A remainder row false modulo P survives uniform alpha in E with probability
  at most `(D-1)/|E|`, equal to `647/|E|` at D648. W, Y and KA must be fixed
  before alpha. If it instead holds as a polynomial identity modulo P, the two
  alphabet checks, the carry range and the admitted inequality
  `3*m*D*(q-1)*B_z + C*Gamma*B_T + q*2^(e_KA-1) < P`, with
  `B_z=max(|L|,|U|)`, bound every integer residual coefficient strictly below
  P. Therefore it holds over the integers:
  `A*z - sum_col iota(Ch_col)*T_col = q*KA` in `Z[Y]/(Phi_D)`, and so modulo q.
- A parity polynomial row false modulo P survives uniform xi with probability
  at most `322/|E|`. W, Q and K must be fixed before xi. If it instead holds as
  a polynomial identity modulo P, the alphabet, clear range checks and admitted
  inequality `H+3*2^(e_Q-1)+2*2^(e_K-1)<P` bound every integer residual coefficient
  strictly below P. Therefore it holds over the integers. Reducing modulo 2
  and Phi gives the F162 parity equation.
- A prime row false in `S_E` survives uniform alpha with probability at most
  `(D-1)/|E|`: both of its sides are remainders of degree below D. uP, W and
  the fold challenges must be fixed before alpha. It shares alpha with the A
  rows, so it adds nothing to that term. The row is an identity over E
  between a table of field elements and the response; nothing is lifted to
  the integers and no inequality is involved.
- Once the evaluated rows are fixed, batching the A rows and the parity row
  by powers of uniform gamma adds at most `n_A/|E|` with the parity row and
  `max(n_A-1,1)/|E|` without it. A nonzero prime-row residual at alpha is
  then cancelled by at most one eta, which adds `1/|E|`. The combined union
  bound is at most `(D-1+322+n_A+1)/|E|` with all rows, before other
  protocol losses.
- An in-alphabet nonzero digit in a tail or padding position does not change
  acceptance: its weight is zero. Out-of-alphabet tails still reject.

One fixed set of field challenges is a probabilistic polynomial identity test,
not deterministic equivalence for every adversarial input. In particular, zero
or otherwise degenerate explicit challenges can hide false rows. Differential
regression tests draw seeded field challenges after constructing the witnesses;
they do not establish a complete security theorem.

## Source binding and projection

**Obligation sketch; not a complete theorem.** The statements below are
algebra about a fixed table. The security conclusion drawn from them is
conditional and is stated last.

**Integer rows.** Fix one image digit table that passes the alphabet check
and let T be the integer table it decodes to. Nothing assumes T has the form
`A b` or holds canonical residues. By the first bullet of the previous
section, every accepting child that shares T satisfies, with its own
alphabet-valid response z and range-valid carry KA,

```text
A*z - sum_col iota(Ch_col)*T_col = q*KA      in Z[Y]/(Phi_D), for each of n_A rows.
```

**Relaxed opening from two children.** Suppose two accepting children share
T and the matrix and differ only in the fold challenge of column `col`:
`(c,z,KA)` and `(c',z',KA')`. Put `a=z-z'`, `s=c-c'` and `kappa=KA-KA'`.
Subtracting the two integer rows,

```text
A*a = s*T_col + q*kappa        in Z[Y]/(Phi_D);
A*a = s*T_col                  modulo (q,Phi_D);
|a|_inf <= Delta = U-L.
```

The pair `(a,s)` is a relaxed opening modulo q of the committed column, with
a difference of two fold challenges as slack. Two children suffice.

**Binding.** Let `(a,s)` and `(a2,s2)` be relaxed openings of the same column
modulo q. Multiplying the first by `s2`, the second by s and subtracting
cancels the column: `A*(s2*a - s*a2) = 0` modulo `(q,Phi_D)`, with
`|s2*a - s*a2|_inf <= eta_A = 4*Gamma*Delta`, since a challenge difference
has operator norm at most `2*Gamma`. The certified cell has
`eta_A < (q-1)/2`. Either this is a nonzero short kernel vector of A modulo q,
excluded under the SIS assumption at the admitted rank for a matrix derived
under the [setup contract](labinius-setup-contract.md), or `s2*a = s*a2` over
the integers and the rational sources `a/s` agree.

**No further condition on P.** The inequality `4*Gamma*B_nu < P`, with
`B_nu = 3*m*D*(q-1)*Delta + q*(2^e_KA - 1)`, is not needed and is not
checked. It belonged to the argument for an image table of arbitrary field
elements: there the row of one child was an identity modulo P only, three
children were compared, and the inequality lifted the cancelled equation
`s2*nu = s*nu2` from R to the integers. With the image alphabet-checked, the
admitted commitment no-wrap inequality makes each child's row an integer
identity in that child alone; two children then subtract over the integers,
and the binding comparison is arithmetic modulo q in which P plays no part.
At the sample geometry the former left side is `1251724738092505906320`,
about `2^70.1`, which exceeds `2^64 - 59`; the commitment no-wrap total is
`1737202407392206848`, about `2^60.6`, so `2^64 - 59` and `2^128 - 275` are
both admitted.

**Projection.** The challenge difference s is nonzero modulo two in F162,
hence invertible there; by proof-prime admission it is also a unit modulo P,
which the comparison above does not use and
[the prime opening](#prime-opening-and-the-unit-condition) does.
From `s2*a = s*a2` the common source `a/s` over `Q[Z]/(Phi)` has an odd
denominator. The two children's parity rows, integer identities by the second
bullet of the previous section, subtract to `B(a mod 2)=U_col*(s mod 2)`, so
modulo-two projection authenticates the shared F162 source. The host claim is
recovered by projecting this source onto its low 128 coordinates, using the
binary coefficients of the field-switch identities. Establishing that these
extracted objects satisfy all consumer semantics remains part of composition.

**What this does not establish.** The statements above concern a fixed table
and exact polynomial identities. They become a binding theorem only together
with a theorem that the oracle openings and the sumchecks supply, on a
suitable tree of accepting transcripts, one common alphabet-valid Y, one
alphabet-valid W per child, bounded carries and the exact identities, with a stated extraction
and Fiat–Shamir loss. The [root reduction](labinius-root-reduction.md) binds
the admitted-root identity and fixes the statement, witness and challenge
order, and its conditional ledger composes these arithmetic reductions.
Complete frontend composition, knowledge extraction and its loss, the SIS
reduction and Fiat–Shamir security remain open as recorded there. Arithmetic
tests and shape admission alone do not resolve them.

## Prime opening and the unit condition

**Obligation sketch; not a complete theorem**, in the sense of the previous
section, whose relaxed opening and binding this one reuses. Write
`S=Z[Y]/(Phi_D)`, `S_F=F[Y]/(Phi_D)=S/PS` and `S_E=E[Y]/(Phi_D)`.

**The opened object.** Let `(a,s)` be a relaxed opening of column `col`:
`a[j]=z[j]-z'[j]` in S for every ring element j, and `s=c-c'` in the scalar
ring `Z[Z]/(Phi)`. The rational source of the column is
`w*[j]=a[j]/iota(s)`. Phi is irreducible over the rationals, so the norm
`N(s)`, the resultant of s and Phi, is a nonzero integer and `s'=N(s)/s` has
integer coefficients. Therefore `w*[j]=iota(s')*a[j]/N(s)` lies in
`S[1/N(s)]`. Phi is monic, so s is a unit of `F[Z]/(Phi)` exactly when P does
not divide `N(s)`. In that case reduction modulo P is defined on `S[1/N(s)]`
and

```text
w*[j] mod P = iota(s)^(-1)*a[j]        in S_F.
```

By binding, `iota(s2)*a=iota(s)*a2` over the integers for any second relaxed
opening, so `w*` and its reduction do not depend on the pair of children.
The prime claim the reduction proves is a statement about this object:

```text
v = sum_col eq(r_col,col) * sum_j eq(r_ring,j) * <omega, w*[j][col] mod P>.
```

The prime opening is an evaluation of the extracted rational source reduced
modulo P. For an honest prover `w*[j][col]=src[j][col]`.

**The step that needs a unit.** Suppose two accepting children share the
image table, the bound table uP, the claim and every fold challenge except
that of column `col`, and that the prime row holds in `S_E` in both, by the
alpha bullet above. Every other column has the same challenge and the same
`uP`, so subtracting the two rows leaves

```text
sum_j e_j*a[j] = iota(s)*uP[col]        in S_E.          (*)
```

The conclusion wanted is `uP[col]=sum_j e_j*(w*[j] mod P)`: the bound table
is the `r_ring` fold of the reduced source, so the value claim on it is the
prime claim above. It follows from `(*)` by cancelling `iota(s)`. That
cancellation is the one step of the argument that uses s modulo P, and it
needs `iota(s)` to be a non-zero-divisor of `S_E`. The A rows and the parity
row never divide modulo P: they are lifted to the integers first.

**The condition is a statement about P alone.** `Phi(iota(Z))=Phi_D(Y)`,
with `iota(Z)=-Y^k` for `k>1` and `iota(Z)=Y` for `k=1`. So S is free over
the scalar ring with basis `1,Y,...,Y^(k-1)`, and multiplication by `iota(s)`
acts on `S_F` as k copies of multiplication by s on `F[Z]/(Phi)`. `S_E` is
`S_F` with scalars extended from F to E, and extension of scalars preserves
both injectivity and its failure. Hence `iota(s)` is a non-zero-divisor of
`S_E` exactly when it is a unit of `S_E`, exactly when s is a unit of
`F[Z]/(Phi)`. The challenge field does not enter. The same condition is what
makes `w* mod P` defined, so it is used twice: to say what is opened, and to
prove it.

**The lemma used.** Proof-prime admission condition 1 of
[root admission](labinius-root-admission.md#proof-prime-admission-and-clear-integer-ranges):
`P^f > (6w)^(deg/2)`, with f the order of P modulo 243, makes every nonzero
difference of two fold challenges a unit modulo P. Under it any two distinct
challenges of one column give `(*)` with an invertible `iota(s)`. Two
children per column suffice, as for the binary opening, and the fold term of
the ledger is the same `C/|S|`. The prime row adds no admission condition.

**What fails without it.** If a nonzero difference s is a zero divisor
modulo P, then P divides `N(s)`, `w* mod P` is not defined from that pair of
children, and `(*)` determines `uP[col]` only up to the annihilator of
`iota(s)`, which is not zero. The gap is an attack and not only a missing
proof. Let `f=1`, so that Phi has a root zeta in F. Evaluation at zeta
splits a direct factor `F[Y]/(Y^k+zeta)` off `S_F` (for `k>1`), on which
`iota(c)` acts as the scalar `c(zeta)`. Take a nonzero delta in that factor
and two columns a and b. A prover binds `uP[a]+delta` and `uP[b]-delta` in
place of the honest columns, before the fold challenges as the order
requires. The value moves by `(eq(r_col,a)-eq(r_col,b))*<omega,delta>`, and
the residual of the prime row is

```text
iota(c_a - c_b)*delta = (c_a(zeta) - c_b(zeta))*delta.
```

It vanishes as a polynomial, hence at every alpha and under every eta,
exactly when the two challenges agree at zeta, and no other check reads uP.
Two independent challenges agree at zeta with probability
`sum_x Pr[c(zeta)=x]^2 >= 1/P`, whatever their distribution and whatever E.
For general f the same construction above one prime ideal succeeds with
probability at least `P^(-f)`.

- For a 64-bit prime with `f=1`, such as the rejected `2^64 - 23703`, a
  false prime claim is accepted with probability at least `2^-64` per
  attempt. A larger challenge field does not help.
- For a 128-bit prime with `f=1`, such as the rejected
  `2^128 - 2^32 + 22537`, the lower bound `1/P` is at the target and is not
  by itself a break. But unit differences are then impossible, not merely
  unproved: the challenge set has about `2^136.3` members and zeta has fewer
  than `2^128` possible values, so two distinct challenges agree at zeta.
  The fold term would become C times the probability that two challenges
  agree modulo some prime ideal above P. That probability is at most 162
  times the largest probability eps with which a challenge takes one fixed
  residue modulo one such ideal, and this specification has no bound on
  eps: it concerns the distribution of sparse signed sums of powers of zeta.

`2^128 - 275` has `f=6` and passes condition 1, which is why it is the
128-bit proof prime of a root that carries a prime claim. `2^64 - 59` has
`f=162`: `F[Z]/(Phi)` is a field and every nonzero difference is a unit
without a norm bound.

## Exact bits are the consumer's obligation

The binary claim is a statement about `w* mod 2` and the prime claim one
about `w* mod P`. For an honest prover both read the committed bits.
Acceptance does not make them agree. The relaxed opening yields a rational
source with a bounded numerator, not an integral one, and a prover may also
commit a short source that is integral but not binary: a coefficient 2 is
read as 0 by the binary claim and as 2 by the prime claim. A consumer that
reads the same data through both claims, or needs a prime claim to be about
bits at all, must prove it. The reduction enables that proof and does not
perform it.

**Lemma.** Fix a column, a ring element j and a relaxed opening `(a,s)` with
s a unit modulo P. Suppose every one of the D coefficients of `w*[j] mod P`
is 0 or the sign `sigma(t)` of its position, and let b be the integer vector
with those coefficients. Then `iota(s)*b - a[j]` is zero modulo P
coefficientwise, and

```text
|iota(s)*b - a[j]|_inf <= 2*Gamma + Delta = 184 + 4095 = 4279 < P,
```

since a challenge difference has operator norm at most `2*Gamma` and
`|a[j]|_inf <= Delta`. So `iota(s)*b=a[j]` over the integers and `w*[j]=b`:
the source ring element is integral, its coefficients are exact signed bits,
and its reduction modulo two is the same bit vector.

**Whole ring elements.** The hypothesis covers all D coefficients of the ring
element, including positions no source bit occupies. Checking some positions
does not bound the others, and without a bounded integer representative at
every position the comparison above has nothing to compare. In
`Z[X]/(X^2+1)` with `P=101`, `s=1+2X` and `a=53X`, the source
`a/s=(106+53X)/5` has constant coefficient 1 modulo 101 and 0 modulo 2,
while its other coefficient is 51 modulo 101. With a bound B on the
positions that are not required to be bits, the margin becomes
`2*Gamma*B + Delta < P`.

**What the reduction provides.** The predicate is `h_t*(h_t-sigma(t))=0` at
every position t of the coefficient table h of the ring elements both claims
read, with `h_t=0` in place of it wherever the consumer requires an unused
position to be empty. A sumcheck for it over E ends in one evaluation of h
at a point `(rho_t,rho_j,rho_col)`. That evaluation is a prime claim on the
same commitment with `omega[t]=eq(rho_t,t)` for `t<D`, `r_ring=rho_j` and
`r_col=rho_col`. This is why `omega` has one weight per coefficient of a
ring element and is not confined to the positions of source bits. Data that
only the binary claim reads needs no such check.

## Regression evidence

The shift recurrence has one differential test against the dense definition
for both trinomials, in `akita-algebra`. The relation itself is exercised
through `crates/akita-labinius-prover/tests/root_reduction.rs`: the three
opening modes and both host
fields at the small admitted
`(log_num_cells,log_fold_width,lambda_fold)=(4,1,128)` and `(4,0,128)`
geometries, over both field pairs (`2^128 - 275` as its own challenge field,
and `2^64 - 59` with its quadratic extension), with a tamper table per mode
that covers every wire region including KA, the fold-response nonce and every
proof-of-work nonce, a missing proof-of-work nonce and one outside its search
range, a prime claim with a wrong value, weight or point, and rejections of
a base field that fails proof-prime admission and of
a challenge field that is too small. One test evaluates the prime row at
explicit challenges on the honest table and on a table with opposite errors
in two columns that cancel in the value claim: the row holds on the first
and fails on the second. It works on the relation because the prover's
sumcheck refuses a false sum, so no proof of a false row exists to replay.
These checks support implementation
agreement with the relation above;
they do not add a proof protocol.
