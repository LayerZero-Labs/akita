# Modulus Switching Investigation

Status: theory-first design note on April 2, 2026.

This note is about the algebra we would want for a fully rigorous Hachi
modulus-lowering step. It is not constrained by the current implementation.

The motivating target is a genuine recursive chain such as:

```text
128-bit -> 64-bit -> 32-bit -> 19-bit / 16-bit
```

where later recursive proofs really live over smaller fields, rather than only
using tighter serialization.

The local paper anchors used here are:

- `/Users/quang.dao/Documents/Papers/LNP22.pdf`
- `/Users/quang.dao/Documents/Papers/Aggregating_Falcon_Signatures_with_LaBRADOR.pdf`
- `/Users/quang.dao/Documents/Papers/Nguyen_Thesis_LatticeZK_2022.pdf`
- `/Users/quang.dao/Documents/Papers/LatticeFold+.pdf`
- `/Users/quang.dao/Documents/Papers/Symphony.pdf`

## Executive Summary

- A modulus switch should be modeled as a local translator
  `Switch_{hi->lo}` between two native recursive segments.
- The switch theorem must quantify over one shared switch-level witness table
  `w`, not two unrelated witnesses.
- The externally visible theorem must take the **full recursive claim**
  `(commitment, opening point, opening value)` as input and output. What can be
  moved to a lighter commitment-core boundary is only an internal factorization
  of the proof, not the public statement itself.
- The generic Falcon/LaBRADOR trick
  `small relation -> larger proof modulus`
  is useful, but it is only one side of what we need.
- The biggest correction to the earlier note is Hachi-specific:
  if we place the switch at the boundary where recursion has already reduced to
  a claim about a flat small-digit witness table, then the old and new claims
  are both fundamentally **linear** in that same witness.
- That means a Hachi switch is much closer to
  "same small witness satisfies two modular linear systems"
  than to
  "simulate arbitrary large-field arithmetic inside a small field."
- This makes several constructions plausible:
  1. a translator proved while still in the larger field,
  2. a true smaller-field switch using Nguyen-style carry polynomials,
  3. a joint modulus-and-ring-dimension switch,
  4. more speculative universal-integer or composite/CRT bridge variants.
- The most important new concrete result in this revision is that the hard
  `32 -> 19` and `32 -> 16` drops are numerically plausible once the switch is
  stated at the recursive commitment boundary in split/NTT basis. The same
  conservative bound also explains why late hybrid switches are much easier
  than early ones.

## The Right Abstraction

Fix two native recursion segments:

- old segment over field/ring `K_hi`
- new segment over field/ring `K_lo`

with `|K_hi| > |K_lo|`.

The clean theorem shape is:

> A switch proof `Switch_{hi->lo}` proves that there exists one shared
> switch-level witness table `w` in the allowed small alphabet such that the
> incoming old-field recursive claim is true for `w`, and the outgoing
> new-field recursive claim is true for the same `w`.

So the switch does two jobs:

1. finish the incoming old native claim, and
2. seed a fresh outgoing native claim.

After the switch, recursion is native again over `K_lo`.

This is the right conceptual model for a staged chain:

```text
native K_128
switch 128->64
native K_64
switch 64->32
native K_32
switch 32->19
native K_19
```

not

```text
simulate K_128 inside K_19 forever.
```

## Main New Simplification: Hachi Switches Can Be Defined At A Linear Boundary

The most important refinement is Hachi-specific.

At the right recursion boundary, the carried object is a flat small-digit witness
table `w`, together with a commitment-opening claim about that table. If we
define the switch at exactly that boundary, then the old and new statements are
both linear in `w`.

Abstractly:

```text
old claim:
  Commit_hi(w) = C_hi
  Eval_hi(w, r_hi) = y_hi

new claim:
  Commit_lo(w) = C_lo
  Eval_lo(w, r_lo) = y_lo
```

plus the usual small-alphabet condition on `w`.

This matters enormously.

It means a Hachi switch does not have to emulate an arbitrary old verifier
circuit. Instead, it can be defined as a proof that the same flat small-digit
table satisfies two different PCS instances, one old and one new.

The hard part becomes:

- different moduli,
- possibly different ring degrees / chunkings,
- canonical encoding of old public coefficients,
- quotient and carry witnesses,

not arbitrary witness-witness multiplication over the old field.

For the rest of this note, there are actually two nearby boundaries worth
distinguishing:

- the **public recursive-state boundary**, where the carried claim is
  `(C, r, y)` and the switch must certify the old opening claim as well as the
  old commitment claim,
- the lighter **internal commitment-core factorization**, where one isolates the
  commitment rows as a bridge subproblem for modeling or proof engineering, but
  only inside a larger theorem that still transports the opening claim.

The second boundary is a useful internal decomposition, but the first boundary
is the actual theorem statement Hachi needs.

## Switch Theorem

Fix:

- old field/ring `(q_hi, D_hi)`
- new field/ring `(q_lo, D_lo)`
- allowed digit alphabet `A`
- flat switch-level witness `w in A^N`

Let:

- `C_hi` be the incoming old commitment
- `r_hi` be the incoming old opening point
- `y_hi` be the incoming old opening value
- `C_lo` be the outgoing new commitment
- `r_lo` be the outgoing new opening point
- `y_lo` be the outgoing new opening value

Then the ideal switch statement is:

```text
Switch_{hi->lo}(pub_hi, pub_lo) :=
  exists w, aux_hi, aux_lo
such that
  Small_A(w)
  and Stmt_hi(C_hi, r_hi, y_hi; w, aux_hi)
  and Stmt_lo(C_lo, r_lo, y_lo; w, aux_lo).
```

The soundness claim should be:

> If `Switch_{hi->lo}(pub_hi, pub_lo)` verifies, then there exists one shared
> switch-level witness table `w` in `A^N` such that:
>
> - the incoming old recursive claim is true for `w`, and
> - the outgoing new recursive claim is true for the same `w`.

That "same `w`" clause is non-negotiable.

## Public Inputs Must Be Encoded, Not Reinterpreted

There is no honest field embedding

```text
K_hi -> K_lo
```

when the target field is smaller and of different characteristic.

So the old public state must enter the switch as a canonical encoding:

```text
pub_hi = Encode_hi(C_hi, r_hi, y_hi).
```

A switch proof must never treat old-field elements as if they were native
`K_lo` scalars.

The right pattern is:

- old state enters as encoded data,
- switch proof decodes it canonically,
- old relations are checked after integer lift,
- new state is emitted natively over `K_lo`.

For transcript soundness, the new point `r_lo` should be derived freshly from
bytes after binding:

- the encoded old state, and
- the new outgoing commitment `C_lo`.

There should be no attempt to algebraically map old challenges into the new
field.

## Hachi-Specific Algebraic Form

The useful specialization is to define the switch in terms of linear maps on the
flat witness table `w`.

Let:

- `G_hi` be the old commitment linear map from flat witness digits to old
  commitment coefficients,
- `G_lo` be the new commitment linear map,
- `L_hi(r_hi)` be the old opening linear functional,
- `L_lo(r_lo)` be the new opening linear functional.

Then the core old and new relations are:

```text
G_hi w = c_hi mod q_hi
L_hi(r_hi) w = y_hi mod q_hi

G_lo w = c_lo mod q_lo
L_lo(r_lo) w = y_lo mod q_lo
```

where:

- `c_hi` is the coefficient vector of `C_hi`
- `c_lo` is the coefficient vector of `C_lo`

If the ring degree changes, `G_hi` and `G_lo` are simply different public
linear maps from the same flat witness table. This is why changing modulus and
changing ring dimension are not separate conceptual problems at this level.

## First Important Consequence

The old side is linear in the shared witness.

That means the big cost in a Hachi switch is not "general old-field
multiplication on unknown values." It is:

- transporting old public coefficients,
- proving modular linear relations with quotient witnesses,
- handling carries / encodings when the switch proof itself lives over the
  smaller field,
- proving small-alphabet membership of `w`.

This is much better than the generic non-native arithmetic picture from the
earlier note.

## Construction 1: Translator Proved In The Larger Field

This is the most directly supported by LNP22 and the Falcon/LaBRADOR note.

### Basic idea

Keep the switch proof itself over a field/ring at least as large as the old one.
Inside that switch proof:

1. prove the incoming old claim natively,
2. prove that the same `w` also defines a valid outgoing smaller-modulus claim,
3. emit `(C_lo, r_lo, y_lo)`,
4. continue natively over `K_lo` afterwards.

### Integer-lifted form

For the old side:

```text
G_hi w - c_hi = q_hi * u_hi
L_hi(r_hi) w - y_hi = q_hi * v_hi
```

For the new side:

```text
G_lo w - c_lo = q_lo * u_lo
L_lo(r_lo) w - y_lo = q_lo * v_lo
```

all over the integers.

If the old statement is expressed in ring form, the natural lifted shape is:

```text
Lift(old relation) = (X^(D_hi) + 1) * R_X + q_hi * R_q.
```

This is exactly the
"ring switching, but also over q"
pattern.

### Why it is attractive

- It matches the `small relation -> larger proof modulus` direction that LNP22
  explicitly supports.
- It avoids putting the old claim itself inside a smaller proof field.
- It already supports simultaneous modulus and ring-dimension change, because
  both old and new sides are just public linear maps on the same `w`.

### Why it is not the whole story

- The switch proof itself still pays large-field costs.
- The savings start after the switch, not inside it.
- It is therefore a very strong reference construction, but not necessarily the
  final one if we want the switch layer itself to be cheap.

## Construction 2: True Smaller-Field Switch Via Carry Polynomials

This is the construction most aligned with the actual goal of proving the switch
over the smaller field.

The key literature template is Nguyen Chapter 8:

- integer addition is represented by a carry polynomial
- integer multiplication is represented by a carry polynomial
- then one separately proves:
  - no wrap modulo `q`
  - no wrap modulo `X^n + 1`

### Prime-field encoding

For an old-field element `a in F_{q_hi}`, take its canonical lift

```text
lift(a) in [0, q_hi).
```

Choose a radix `B`, and encode:

```text
lift(a) = a_0 + a_1 B + ... + a_(m-1) B^(m-1).
```

Equivalently, represent it by a digit polynomial:

```text
a_hat(X) = a_0 + a_1 X + ... + a_(m-1) X^(m-1)
```

so that

```text
a_hat(B) = lift(a).
```

### Linear relation gadget

This is where Hachi gets a major win.

Because the switch boundary relations are linear in `w`, the old-side equation
for one row looks like:

```text
sum_j a_j * w_j = c mod q_hi
```

with:

- `a_j` public old-field coefficients,
- `w_j` small witness digits.

Encode each public coefficient `a_j` as `a_hat_j(X)` and the target `c` as
`c_hat(X)`. Then prove:

```text
sum_j w_j * a_hat_j(X) - c_hat(X) - q_hat_hi(X) * k_hat(X)
  = (B - X) * f_hat(X).
```

Interpretation:

- `q_hat_hi(X) * k_hat(X)` handles old-field modular reduction,
- `(B - X) * f_hat(X)` handles radix carries.

This is much lighter than generic non-native multiplication because:

- the `a_hat_j(X)` are public,
- the `w_j` are tiny digits,
- the only multiplication involving a secret polynomial is
  `q_hat_hi(X) * k_hat(X)`, where `q_hat_hi(X)` is public and `k_hat(X)` is a
  short quotient polynomial,
- there is no multiplication of two unknown old-field values in the core old
  Hachi opening/commitment relations.

### The old opening claim: three concrete realizations

The previous paragraph is still too compressed if read as a concrete Hachi
instantiation.

At the theorem level, the old recursive claim is indeed the scalar statement

```text
Eval_hi(w, r_hi) = y_hi.
```

But the actual recursive verifier factors this claim through an intermediate
ring element.

Concretely, the incoming recursive verifier state is

```text
(C_hi, r_hi, y_hi),
```

where `r_hi` splits into:

- an inner part of length `log2(D_hi)`,
- an outer part that is converted into ring-opening weights `(a_hi, b_hi)`.

The native verifier then computes:

- `v_hi`, the ring element induced by the inner coordinates,
- `y_ring_hi`, a claimed outer evaluation ring element,

and checks

1. the inner trace identity

```text
Trace(y_ring_hi * sigma_{-1}(v_hi)) = D_hi * y_hi mod q_hi,
```

2. the outer ring-switch relation system on the current flat witness `w`,
   whose public rows include `v_hi`, `C_hi`, and `y_ring_hi`.

So there are really three candidate ways to transport the old opening claim in
the switch proof.

#### Option A: one fully flattened scalar row

The mathematically simplest choice is the direct multilinear opening row

```text
sum_j lambda_hi,j(r_hi) * w_j = y_hi mod q_hi.
```

Pros:

- the cleanest theorem statement,
- no extra secret old-field intermediate besides the quotient/carry witness.

Cons:

- row support is the full flat witness length `N`,
- it throws away the protocol's natural `w -> y_ring_hi -> y_hi` factorization,
- it is the least attractive option for the harder `32 -> 19/16` regime.

For the main late `128 -> 32` target, however, this option now looks much
better than I first assumed.

At the worked onehot boundary:

```text
N   = 199,584
eta = 8
B   = 128
q_lo = 2^32 - 99 = 4,294,967,197.
```

The witness contribution to one radix coefficient of the dense opening row is
bounded by

```text
N * eta * (B/2) = 199,584 * 8 * 64 = 102,187,008,
```

which is still comfortably below the 32-bit target modulus.

And the old-row quotient bound is only

```text
|K_open| <= (N * eta + 1) / 2 = 798,337,
```

so with `B = 128` this row needs only `m_K = 3` quotient digits.

So for `128 -> 32`, the missing old opening transport plausibly costs only
**one additional old row**, which means its proof-size effect should be tiny.

As a rough sanity check, if I replace the current structured old-opening rows
in `scripts/modulus_switch_bridge_model.py` by this one dense row, the modeled
late `128 -> 32` fused overhead moves only from about `5.66 KB` down to about
`5.19 KB`. So the two realizations are in the same proof-size ballpark, with a
small edge to the simpler dense-row statement.

#### Option B: protocol-aligned hybrid `w -> y_ring_hi -> y_hi`

View the current witness as

```text
w : [live_x_cols] x [D_hi] -> A,
```

using the same `x`/`y` split as Hachi's native ring-switch tables. Let

```text
lambda_hi^out(x) = b_hi[block(x)] * a_hi[pos(x)]
```

be the public outer opening weight on x-columns. Introduce one secret old-field
ring element

```text
y_ring_hi = (y_hat_0, ..., y_hat_{D_hi-1}) in R_{q_hi, D_hi},
```

but represent each coefficient canonically by radix-`B` digits inside the
smaller proof field.

Then `OldOpenRows_hi` can be stated as:

1. `D_hi` coefficientwise outer-evaluation rows

```text
for each beta in [D_hi]:
  sum_x lambda_hi^out(x) * w(x, beta) - y_hat_beta = 0 mod q_hi,
```

2. one inner trace row

```text
sum_beta mu_hi(beta) * y_hat_beta - D_hi * y_hi = 0 mod q_hi,
```

where `mu_hi(beta)` are the public coefficients of the linear functional

```text
y_ring_hi -> Trace(y_ring_hi * sigma_{-1}(v_hi)).
```

Each row is then transported by the same Nguyen radix/carry gadget as the old
commitment rows.

Pros:

- it matches the actual recursive opening logic,
- the outer-row support drops from `N` down to `live_x_cols`,
- the only secret old-field intermediate is `y_ring_hi`, which has only `D_hi`
  coefficients.

Cons:

- the switch witness must include the radix encoding of `y_ring_hi`,
- it is slightly more elaborate than the one-row scalar opening equation.

This remains a very good structured fallback, especially once `q_lo` becomes
too small for Option A's dense row to satisfy the no-wrap bound.

#### Option C: fully local `w -> z_hi -> y_ring_hi -> y_hi`

One can refine Option B further by introducing explicit folded block witnesses

```text
z_hi[block] = sum_pos a_hi[pos] * w(block, pos),
```

followed by

```text
y_ring_hi = sum_block b_hi[block] * z_hi[block].
```

This minimizes row support further, but it introduces many more old-field
intermediate values that must themselves be canonically encoded in the smaller
proof. There is also a subtle trap here: after the first local fold, the
intermediates are already full old-field values, so later rows lose the
"small witness digit" advantage that makes Options A and B pleasant.

For a first theorem target, this currently looks like overkill.

#### Recommended old-opening realization

The recommendation now depends on the target drop.

For the default `SwitchFold_{128->32}` target, Option A is actually the most
attractive current theorem object:

- keep the public statement as the scalar old recursive claim `(C_hi, r_hi, y_hi)`,
- add one dense old evaluation row on the shared witness `w`,
- avoid introducing any extra old-field witness beyond the usual quotient/carry
  transport.

For more aggressive drops such as `32 -> 19/16`, Option A is much less likely
to fit the smaller-field no-wrap bound, and Option B becomes the better default:

- witness one encoded old ring element `y_ring_hi`,
- prove `w -> y_ring_hi` by `D_hi` coefficientwise outer-evaluation rows,
- prove `y_ring_hi -> y_hi` by one inner trace row.

So the switch note should treat:

- Option A as the default for `128 -> 32`,
- Option B as the structured fallback for sub-32-bit targets,
- Option C as a more experimental rescue path if even Option B becomes too
  tight.

#### Size intuition for the late `128 -> 32` boundary

At the late onehot boundary used elsewhere in this note:

```text
N           = 199,584
D_hi        = 32
live_x_cols = ceil(N / D_hi) = 6,237
eta         = 8
B           = 128
m_q         = 19.
```

Under Option B, the old opening transport adds:

- `rows_open = D_hi + 1 = 33` old modular rows,
- `D_hi * m_q = 32 * 19 = 608` radix digits for the secret `y_ring_hi`,
- plus the corresponding old-modulus quotient/carry witnesses.

A conservative row-local small-secret bound is

```text
S_open ~= eta * live_x_cols + (B/2) * m_q
       ~= 8 * 6,237 + 64 * 19
       ~= 51,112,
```

so `m_K = ceil(log_B(S_open + 1)) = 3` is still enough at this boundary.
That puts the opening-only auxiliary payload at roughly

```text
y_hat_ring_hi        = 608
k_open               = rows_open * m_K          =   99
f_open               = rows_open * (m_q+m_K-1)  =  693
------------------------------------------------------
opening-only aux     ~= 1,400
```

small-field witness scalars, before any reuse or fusion. This is not free, but
it is also nowhere near the dominant cost in the current `128 -> 32` bridge
picture.

### Commitment rows

Likewise, each old commitment row is another modular linear row:

```text
sum_j G_hi[t,j] * w_j = c_hi[t] mod q_hi.
```

And each new commitment or opening row is the analogous relation modulo `q_lo`.

### Why this is much more promising than the earlier generic note

The earlier note over-emphasized generic non-native multiplication.

For Hachi at the right boundary, the old claim is not a generic old arithmetic
circuit. It is a collection of modular linear equations in a small-digit witness.

So the small-field switch does not need full old-field multiplication gadgets
for witness-witness products. It needs:

- encoded public coefficients,
- small-scalar accumulation against witness digits,
- carry polynomials,
- quotient witnesses,
- range proofs on `w`.

That is a much more structured problem.

### Concrete Nguyen-style realization over general Hachi parameters

Here is the concrete parameterized blueprint that currently looks the cleanest.

This subsection is still the generic scalar-row envelope. It is useful for
fixing the theorem shape and the quotient/carry bookkeeping, but it is not yet
the instantiation I would optimize around for Hachi. The preferred concrete
instantiation comes immediately after this subsection and moves the bridge to
the recursive commitment boundary.

#### Public switch parameters

Fix:

- old modulus / ring degree `(q_hi, D_hi)`
- new modulus / ring degree `(q_lo, D_lo)`
- witness alphabet `A subset Z` with `|w_j| <= eta`
- flat switch-level witness length `N`
- old commitment map `G_hi in Z^(R_hi x N)`
- old opening row `lambda_hi(r_hi) in Z^N`
- new commitment map `G_lo in Z^(R_lo x N)`
- new opening row `lambda_lo(r_lo) in Z^N`

Here:

- `R_hi` is the number of old scalar rows after flattening the old commitment
  and old opening claim,
- `R_lo` is the corresponding number of new rows.

In the standard Hachi recursive placement, this is simply:

```text
R_hi = (# old commitment coefficients) + 1
R_lo = (# new commitment coefficients) + 1
```

where the extra `+1` is the opening-value row.

All old public coefficients are taken in their centered integer lifts in
`[-(q_hi-1)/2, (q_hi-1)/2]`.

This "flatten first" choice is important:

- if the switch is placed at the flat witness boundary, then the old ring
  structure is already absorbed into the public matrix `G_hi`,
- so the switch no longer needs an explicit old `(X^(D_hi) + 1)` quotient.

If one wanted to switch earlier, at a ring-native boundary, then one would
indeed need both quotient types. But for the flat-boundary switch described
here, the old side is just a family of modular linear equations.

#### Row form

Every old row has the form:

```text
sum_j a_(t,j) * w_j = c_t mod q_hi
```

where:

- `t in [R_hi]`,
- `a_(t,j)` is public,
- `c_t` is public.

This includes:

- all old commitment rows,
- the old opening row.

Likewise every new row has the same shape modulo `q_lo`.

#### Shared l1 bound

Let:

```text
S = ||w||_1 <= N * eta.
```

For a sparse row one can replace `S` by the row-local support bound

```text
S_t = sum_j |w_j| * 1[a_(t,j) != 0].
```

The old modular row quotient

```text
K_t = (sum_j a_(t,j) * w_j - c_t) / q_hi
```

then satisfies the generic bound

```text
|K_t| <= (S_t + 1) / 2 <= (S + 1) / 2.
```

The crucial point is that this bound depends on the witness l1 norm, not on
`q_hi`.

#### Radix choice

Choose a radix

```text
B = 2^b
```

with:

- `gcd(B, q_hi) = 1`, which is automatic for odd prime `q_hi`,
- `B < q_lo`.

Let:

```text
m_q = ceil(log_B q_hi)
m_K = ceil(log_B (S + 1))
m = m_q + m_K.
```

Encode:

- each old row coefficient `a_(t,j)` by a balanced radix polynomial
  `a_hat_(t,j)(X)` of degree `< m_q`,
- each old target `c_t` by `c_hat_t(X)`,
- the old modulus `q_hi` by `q_hat_hi(X)`,
- each quotient `K_t` by a balanced radix polynomial `k_hat_t(X)` of degree
  `< m_K`.

Then:

```text
a_hat_(t,j)(B) = a_(t,j)
c_hat_t(B) = c_t
q_hat_hi(B) = q_hi
k_hat_t(B) = K_t.
```

#### Core old-row identity

For each old row `t`, define the polynomial

```text
P_t(X) =
  sum_j w_j * a_hat_(t,j)(X)
  - c_hat_t(X)
  - q_hat_hi(X) * k_hat_t(X).
```

Because `P_t(B) = 0`, there exists a carry polynomial `f_hat_t(X)` of degree
`< m - 1` such that

```text
P_t(X) = (B - X) * f_hat_t(X).
```

This is the direct Nguyen-style analogue of

```text
a_hat(X) + b_hat(X) - c_hat(X) = (B - X) f_hat(X)
```

for integer addition, except that here the left-hand side is a general linear
combination of encoded public coefficients and one extra encoded quotient term.

#### Why no extra `(X^m + 1)` quotient appears here

For these row-wise linear identities, there is no product of two long unknown
encoded values.

So if we choose `m = m_q + m_K` large enough, the polynomial

```text
q_hat_hi(X) * k_hat_t(X)
```

fits without wrap in degree `< m_q + m_K - 1`, and the entire row identity can
be written in ordinary `Z[X]` without a separate polynomial-modulus quotient.

This is another major simplification over the generic multiplication case.

#### Coefficient bounds

Let:

```text
mu = min(m_q, m_K).
```

If we use balanced radix digits in `[-B/2, B/2)`, then each coefficient of
`P_t(X)` is bounded by

```text
C_t <= (B/2) * (S_t + 1) + (mu * B^2) / 4.
```

Reason:

- `sum_j w_j * a_hat_(t,j)(X)` contributes at most `(B/2) * S_t`,
- `c_hat_t(X)` contributes at most `B/2`,
- the convolution `q_hat_hi(X) * k_hat_t(X)` contributes at most
  `mu * B^2 / 4` coefficientwise.

Since

```text
P_t(X) = (B - X) * f_hat_t(X),
```

the carry coefficients satisfy the recurrence

```text
p_(t,0) = B f_(t,0)
p_(t,u) = B f_(t,u) - f_(t,u-1),
```

so a sufficient bound is

```text
|f_(t,u)| <= C_t / (B - 1).
```

Thus the old-side witness objects that must be range-proved are:

- the witness digits `w_j`,
- the quotient digits in `k_hat_t(X)`,
- the carry coefficients in `f_hat_t(X)`.

#### Sufficient no-wrap condition inside the smaller-field proof

A conservative sufficient condition for checking these identities inside a
`q_lo`-native proof system is:

```text
q_lo > 2 * max_t {
  eta,
  B/2,
  (S_t + 1)/2,
  C_t,
  C_t / (B - 1)
}.
```

This is not optimized, but it is the right first parameter inequality.

It shows the main tradeoff clearly:

- larger `B` shortens the digit polynomials,
- but worsens the `mu * B^2 / 4` term.

#### Outgoing native claim

Once the old rows are handled, the switch must also create a fresh native new
claim over `q_lo`.

That means proving, for the same `w`:

```text
G_lo w = c_lo mod q_lo
lambda_lo(r_lo) w = y_lo mod q_lo.
```

This side is native, so no old-modulus quotient is needed.

The switch therefore outputs:

```text
(C_lo, r_lo, y_lo)
```

where:

- `C_lo` is a new commitment to the same flat witness table `w`,
- `r_lo` is freshly sampled after binding the encoded old state and `C_lo`,
- `y_lo` is the native opening value of the same `w` at `r_lo`.

#### Batching strategy

Because the old rows are linear in `w`, they can be batched.

The cleanest first version is to keep the opening row separate and batch old
commitment rows with small signed weights:

```text
rho_t in {-1, 0, 1}
```

or another short-norm challenge set, so that coefficient bounds only grow by a
controlled factor.

After batching,

```text
sum_t rho_t * P_t(X) = (B - X) * sum_t rho_t * f_hat_t(X)
```

and the corresponding quotient polynomials also batch linearly.

This is exactly where Hachi’s current "lots of linear rows, then batch" design
intuition carries over to the switch layer.

### Internal Commitment-Core Factorization

The previous subsection still modeled the bridge at a generic
commitment-opening boundary. For Hachi, there is a useful internal
factorization one step earlier: isolate the commitment core of the old claim,
then transport the old opening claim as its own structured row family inside
the same switchfold proof.

At that internal factorization point:

- the old segment has already produced the flat small-digit witness `w`,
- the old recursive state still includes an opening point/value `(r_hi, y_hi)`,
- the commitment rows and the opening row can be treated separately inside the
  switchfold proof,
- and the new segment still outputs a full recursive state over `q_lo`.

So the useful internal split is:

```text
SwitchFold_{hi->lo}(C_hi, r_hi, y_hi; pub_lo^+) :=
  exists w, aux_commit_hi, aux_open_hi, aux_lo
such that
  Small_A(w)
  and CommitRel_hi(w, aux_commit_hi; C_hi)
  and OpenRel_hi(w, aux_open_hi; r_hi, y_hi)
  and FirstFoldRel_lo(w, aux_lo; pub_lo^+).
```

This factorization matters for two reasons.

First, it lets us reason about the commitment-core transport separately from the
old opening transport, which are quite different algebraically.

Second, it still lines up with the actual Hachi D-boundary: the old segment has
already assembled the flat witness `w`, and the new segment is free to re-chunk
that same `w` under a different `D_lo`.

This is only an internal decomposition. The public switch theorem still has to
consume the full old recursive claim `(C_hi, r_hi, y_hi)` and produce a full
new recursive claim.

#### Why split / NTT basis is the right concrete representation

There is one crucial footnote here.

If we expand the old ring equations all the way down to coefficient basis
before switching, then every old ring multiplication introduces a negacyclic
convolution factor. In the conservative bound, that effectively multiplies the
row support by about `D_hi`, which is exactly the wrong direction for the hard
`32 -> 19` and `32 -> 16` steps.

The viable concrete bridge should instead use the same split / NTT-flavored
representation that Hachi already uses internally for ring mat-vecs. In that
representation, the old public rows become families of scalar slot equations,
and the relevant support parameter is the number of live ring inputs in the
row, not `D_hi` times that number.

So the working interpretation of the bounds below is:

- old public ring data are canonically encoded slotwise,
- the Nguyen-style carry proof is applied slotwise,
- the hard support parameter is the row support in split basis.

That is the version that looks genuinely plausible.

#### Commitment-boundary row supports

For a recursive commitment level with:

- `num_blocks = 2^r`,
- `delta_open`,
- `delta_commit`,
- `inner_width`,
- `n_a`, `n_b`, `n_d`,

the dominant old-row supports at the commitment boundary are:

```text
M_A = inner_width
M_D = num_blocks * delta_open
M_B = n_a * num_blocks * delta_open.
```

Interpretation:

- `M_A` is the support of one old inner-commitment row,
- `M_D` is the support of one old quotient / D-style row,
- `M_B` is the support of one old outer-commitment row on `t_hat`.

In practice `M_B` is usually the dominant old support.

If the witness alphabet satisfies `|w_j| <= eta`, then the conservative
per-row l1 bound for the old quotient estimate is:

```text
S_old = M_old * eta
M_old = max(M_A, M_B, M_D).
```

The old modular quotient for that row then satisfies

```text
|K_t| <= (S_old + 1) / 2.
```

This is the quantity that should be inserted into the Nguyen-style `m_K`,
`C_t`, and no-wrap inequalities.

#### Worked onehot `nv = 32` numbers from the planner

The mixed-boolean planner already gives two especially useful `32`-bit
boundaries:

1. an early post-boolean boundary, after the last boolean `32`-bit level, and
2. a late hybrid boundary, after several cheap `32`-bit balanced levels.

These are the two concrete bridge points that currently matter most.

##### Early `32b-bool` boundary: hardest realistic bridge

From the `32b-bool` onehot `nv = 32` schedule, after `L2`:

```text
D_hi        = 64
r           = 7
num_blocks  = 128
n_a         = 2
delta_open  = 32
delta_fold  = 14
inner_width = 792
N           = 1,512,448
alphabet    = {0,1}  so eta = 1
```

So the old commitment-boundary supports are:

```text
M_A = 792
M_D = 128 * 32 = 4,096
M_B = 2 * 128 * 32 = 8,192
M_old = 8,192
S_old = 8,192.
```

This is the right "stress test" boundary because it keeps the nice boolean
alphabet but still has large old commitment rows.

Switching this boundary into the packed degree-7 field
`q_lo = 319,541` is already feasible under the conservative bound if we choose
radix `B = 32`:

```text
q_hi = 2^32 - 99
q_lo = 319,541
B    = 32
m_q  = 7
m_K  = 3
mu   = 3
C    = 131,856
2C   = 263,712 < q_lo
margin = 55,829.
```

So the hard `32 -> 19` drop is not blocked by the Nguyen-style no-wrap bound
at this boundary.

Switching the same boundary directly into the exploratory `16`-bit field
`q_lo = 65,437` is also feasible, but only with a much smaller radix:

```text
q_hi = 2^32 - 99
q_lo = 65,437
B    = 4
m_q  = 16
m_K  = 7
mu   = 7
C    = 16,414
2C   = 32,828 < q_lo
margin = 32,609.
```

So the early `32 -> 16` bridge is not impossible. It is just digit-heavy. That
is the first concrete argument for preferring staged descent or later switches
even if the algebra is sound.

##### Late `32b-bool` boundary: the hybrid-friendly bridge

From the same `32b-bool` onehot `nv = 32` schedule, after `L5`:

```text
D_hi        = 64
r           = 4
num_blocks  = 16
n_a         = 2
delta_open  = 17
delta_fold  = 6
inner_width = 193
N           = 135,040
alphabet    = {-2,-1,0,1}  so eta = 2
```

Now the old supports are:

```text
M_A = 193
M_D = 16 * 17 = 272
M_B = 2 * 16 * 17 = 544
M_old = 544
S_old = 1,088.
```

This is dramatically easier.

For the packed degree-7 field `q_lo = 319,541`, we can now use radix `B = 64`:

```text
q_hi = 2^32 - 99
q_lo = 319,541
B    = 64
m_q  = 6
m_K  = 2
mu   = 2
C    = 36,896
2C   = 73,792 < q_lo
margin = 245,749.
```

For the exploratory `16`-bit field `q_lo = 65,437`, radix `B = 32` already
suffices:

```text
q_hi = 2^32 - 99
q_lo = 65,437
B    = 32
m_q  = 7
m_K  = 3
mu   = 3
C    = 18,192
2C   = 36,384 < q_lo
margin = 29,053.
```

This is the concrete algebraic justification for the hybrid idea:

- stay in `32`-bit while recursion is still cheap there,
- then switch late into `19`- or `16`-bit when the old-row support has already
  collapsed.

##### Staged descent sanity checks

The same support calculation also explains why staged descent is the right
generic lever.

For the early `64b-bool` onehot `nv = 32` boundary after its last boolean
level, the planner gives:

```text
D_hi        = 64
r           = 7
num_blocks  = 128
n_a         = 1
delta_open  = 64
M_old       = 1 * 128 * 64 = 8,192
eta         = 1.
```

The direct `64 -> 32` step is then extremely comfortable with radix `B = 256`:

```text
q_hi = 2^64 - 59
q_lo = 2^32 - 99
B    = 256
m_q  = 8
m_K  = 2
C    = 1,081,472
2C   = 2,162,944 << q_lo.
```

By contrast, a direct conservative `64 -> 16` bridge at the same early support
level still works only with a tiny radix such as `B = 4`, which gives
`m_q = 32` old-modulus digits. So even when a giant jump is not literally
forbidden, staged descent remains much cleaner:

```text
64 -> 32 -> 16
```

rather than

```text
64 -> 16
```

The `128 -> 64` step is easier still. The hard algebraic bottleneck in the
family is the last drop below `32` bits, and that is exactly the step where the
late hybrid bridge helps most.

### Proof-Size Implication: Bridge Budgets

The right first proof-size metric is the **break-even bridge budget**:

```text
bridge_budget = native_hi_suffix_cost - switched_lo_suffix_cost.
```

Interpretation:

- `native_hi_suffix_cost` is the best cost of simply staying in the current
  field from the chosen switch boundary onward,
- `switched_lo_suffix_cost` is the best native cost of continuing from that
  same witness state in the smaller field,
- the bridge itself is not included.

So:

- if the budget is negative, switching already loses even with a free bridge,
- if the budget is positive, the switch can spend up to that many bytes and
  still break even.

This is already slightly optimistic for switching, because it assumes the
outgoing commitment `C_lo` is fully reused by the first smaller-field native
level rather than charged twice.

#### `32 -> 19/16`: no proof-size win on current planner states

Using the actual `32b-bool` onehot and dense `nv = 32` boundary states:

| switch point | stay in `32` | switch to `k7-pack` | switch to `16` | bridge budget |
| --- | ---: | ---: | ---: | ---: |
| onehot, early boolean boundary | `39,424 B` | `40,616 B` | `41,088 B` | `-1,192 B`, `-1,664 B` |
| onehot, late hybrid boundary | `31,088 B` | `33,616 B` | `31,984 B` | `-2,528 B`, `-896 B` |
| dense, late hybrid boundary | `32,992 B` | `34,384 B` | `34,272 B` | `-1,392 B`, `-1,280 B` |

So under the current planner model, a `32 -> 19` or `32 -> 16` switch does
**not** reduce total proof size by itself. The smaller-field native suffix is
already slightly worse than just staying in `32` from the same boundary.

This is consistent with the global planner results:

- `32b-bool` is still the best whole-proof technology,
- `k7-pack` is better tail technology, but not better whole-proof technology.

So the current hybrid hope for `32 -> 19/16` is **not**:

```text
switch and immediately win on total bytes.
```

It is instead:

```text
find a bridge + suffix design that beats the current native low-field suffix,
or exploit a new lever that the present planner does not model.
```

#### `64 -> 32`: real room for a win

The picture changes once the starting field is `64` bits.

Using the actual `64b-bool` boundary states:

| switch point | stay in `64` | switch to `32` | switch to `16` | bridge budget |
| --- | ---: | ---: | ---: | ---: |
| onehot, early boolean boundary | `48,288 B` | `39,616 B` | `41,280 B` | `8,672 B`, `7,008 B` |
| onehot, late boundary | `39,184 B` | `32,896 B` | `34,080 B` | `6,288 B`, `5,104 B` |
| dense, late boundary | `40,464 B` | `33,936 B` | `35,120 B` | `6,528 B`, `5,344 B` |

That is the first genuinely encouraging result.

A `64 -> 32` switch has a break-even budget of about `6` to `9 KB` on the
current onehot/dense `nv = 32` states. Since the relevant smaller-field native
levels in this regime cost about `2.7` to `3.1 KB` each, a bridge that is
comparable to roughly two extra small-field recursive levels would still win.

So the current evidence says:

- `64 -> 32` is promising,
- `64 -> 16` is possible, but has less room,
- staged descent still looks safer than a giant direct drop.

#### `128 -> 64/32`: very likely worth it

For the corrected 128-bit onehot `nv = 32` study, take a representative late
balanced boundary with:

```text
w = 199,584
prev_bound = 4
```

which is exactly the regime where the native 128-bit suffix is still paying the
large 128-bit tail.

From that boundary:

| stay / switch | suffix cost |
| --- | ---: |
| stay in `128` | `56,560 B` |
| switch to `64` | `47,936 B` |
| switch to `32` | `37,736 B` |
| switch to `16` | `39,920 B` |

So the break-even bridge budgets are:

```text
128 -> 64 :  8,624 B
128 -> 32 : 18,824 B
128 -> 16 : 16,640 B
```

This is a very strong signal.

Even allowing for the fact that a direct `128 -> 16` bridge is algebraically
harder than `128 -> 32`, the budget for `128 -> 32` is large enough that a
reasonably compact commitment-boundary bridge should still pay for itself.

#### Preliminary verdict

The current planner-plus-bridge-budget picture is:

- `32 -> 19/16`: not a win yet
- `64 -> 32`: promising
- `128 -> 64`: promising
- `128 -> 32`: very promising

So modulus lowering does look like a real proof-size lever, but primarily as a
**large-to-medium** field reduction. The present evidence does not support a
standalone `32 -> 19/16` win without some additional new idea beyond the
current planner model.

### Concrete Fused Bridge-Core Model

The bridge-budget section above still treats the bridge as a black box. The next
step is to model one explicit bridge witness.

The simplest commitment-core lower-bound model I currently trust is:

```text
W_bridge =
  [ w
  | t_hat_hi
  | k_old
  | f_old
  | t_hat_lo ].
```

Interpretation:

- `w` is the shared flat witness at the switch boundary,
- `t_hat_hi` is the old inner commitment witness,
- `k_old` are the old modular quotient digits for Nguyen-style `q_hi` lifting,
- `f_old` are the old carry coefficients,
- `t_hat_lo` is the outgoing smaller-field commitment witness.

In this model:

```text
T_hi      = D_hi * n_a_hi * num_blocks_hi * delta_open_hi
rows_old  = D_hi * (n_a_hi + n_b_hi)
W_bridge  = N + T_hi + rows_old * (m_q + 2 m_K - 1) + T_lo.
```

Here:

- `m_q = ceil(log_B q_hi)`,
- `m_K = ceil(log_B (S_old + 1))`,
- `T_lo` is taken from the first native smaller-field suffix level that would
  follow the switch,
- carry coefficients are stored as native smaller-field scalars bounded by
  `F = ceil(C / (B - 1))`.

This model intentionally prices only the commitment-core transport. A fully
rigorous recursive switchfold still has to add the old opening/evaluation
transport on top of it.

The important qualitative correction is now target-dependent:

- for the default `128 -> 32` switch, the missing old opening claim may be
  cheap enough to model as **one dense old evaluation row** on `w`,
- for harder drops such as `32 -> 19/16`, the dense row is much less likely to
  fit the no-wrap bound, and then the more structured
  `v_hi / y_ring_hi / trace / challenge-fold` transport becomes the right
  replacement.

So the current script should be read as a structured lower-bound model, not yet
the final word on the old opening transport.

The right proof-size comparison for this model is the **fused overhead**:

```text
fused_overhead = full_bridge_cost - native_lo_suffix_cost.
```

This treats the outgoing `C_lo` as fully reused by the first smaller-field
native suffix level. So it is still mildly optimistic, but much fairer than
charging the bridge as a totally separate proof from scratch.

I encoded exactly this model in
`scripts/modulus_switch_bridge_model.py`.

#### Results from the fused bridge model

Using the current script:

```text
python3 scripts/modulus_switch_bridge_model.py
```

the best radices and fused outcomes are:

| case | best `B` | bridge witness | bridge full | fused overhead | bridge budget | net |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `64 -> 32` early onehot | `2048` | `2,369,024` | `43,696 B` | `4,080 B` | `8,672 B` | `+4,592 B` |
| `64 -> 32` late onehot | `2048` | `240,960` | `41,840 B` | `8,944 B` | `6,288 B` | `-2,656 B` |
| `128 -> 64` late onehot | `8192` | `295,456` | `53,760 B` | `5,824 B` | `8,624 B` | `+2,800 B` |
| `128 -> 32` late onehot | `128` | `315,296` | `41,856 B` | `4,120 B` | `18,824 B` | `+14,704 B` |

Here:

```text
net = bridge_budget - fused_overhead.
```

So the first concrete bridge model changes the verdict in a useful way:

- `64 -> 32` is **not automatically a win**. It wins at the early onehot
  boundary, but loses at the late onehot boundary.
- `128 -> 64` still looks like a real win, but the margin is not huge.
- `128 -> 32` remains the strongest target by far. Even after charging an
  explicit bridge witness, there is still about `14.7 KB` of room left in this
  first model.

That last line is the most important concrete result of this revision.

It suggests that if we want a first serious modulus-switch implementation, the
best target is likely:

```text
128 -> 32
```

rather than trying to start with the much tighter `32 -> 19/16` step.

#### Worked `128 -> 32` bridge anatomy

The strongest case is now concrete enough to write down almost completely.

Take the corrected 128-bit onehot `nv = 32` schedule at the late boundary after
`L3`:

```text
D_hi        = 32
n_a_hi      = 2
n_b_hi      = 2
num_blocks  = 32
delta_open  = 33
N           = 199,584
alphabet    = {-8,...,7}  so eta = 8.
```

The bridge-core model below isolates the old commitment rows. This is **not**
the full recursive theorem: a rigorous `SwitchFold_{128->32}` must additionally
transport the old opening claim at `(r_hi, y_hi)`. I isolate the commitment
rows here because they dominate the boundary witness anatomy and are the part I
have sized most concretely so far.

Let:

- `t_hat_hi` be the old inner witness digits,
- `Recomp_hi(t_hat_hi)` be their public gadget recomposition,
- `c_hi` be the slotwise old commitment coefficients.

Then the old bridge rows are:

```text
A_hi rows:
  A_hi * w - Recomp_hi(t_hat_hi) = q_hi * K_A

B_hi rows:
  B_hi * t_hat_hi - c_hi = q_hi * K_B.
```

The row counts are:

```text
rows_A = D_hi * n_a_hi = 32 * 2 = 64
rows_B = D_hi * n_b_hi = 32 * 2 = 64
rows_old = 128.
```

So the bridge is proving 128 old modular linear relations on the shared flat
witness `w`, not emulating a generic old verifier.

For the best fused `128 -> 32` bridge, the model chooses:

```text
q_hi = 2^128 - 5823
q_lo = 2^32 - 99
B    = 128
m_q  = 19
m_K  = 3
F    = 8,612.
```

That gives the concrete bridge witness:

```text
W_bridge =
  [ w
  | t_hat_hi
  | k_old
  | f_old
  | t_hat_lo ].
```

with sizes

```text
w         = 199,584
t_hat_hi  = 32 * 2 * 32 * 33 = 67,584
k_old     = rows_old * m_K = 128 * 3 = 384
f_old     = rows_old * (m_q + m_K - 1) = 128 * 21 = 2,688
t_hat_lo  = 128 * 2 * 16 * 11 = 45,056
------------------------------------------
W_bridge  = 315,296.
```

Two things are worth noticing immediately.

First, the Nguyen quotient-and-carry payload is tiny:

```text
k_old + f_old = 3,072 < 1% of W_bridge.
```

So in this concrete `128 -> 32` case, the bridge is mostly not
"non-native arithmetic overhead." It is mostly:

- the shared witness `w`,
- an old commitment witness `t_hat_hi`,
- a new commitment witness `t_hat_lo`,

plus a relatively small old-modulus carry layer.

Second, the outgoing smaller-field commitment is already the same object the
native 32-bit suffix wants to use next. That is exactly why the fused bridge
comparison is the right one here. In particular, `t_hat_lo` is not dead weight:
it is the first native 32-bit commitment witness, which is why the fused model
can legitimately reuse the outgoing `C_lo` instead of charging a separate
commitment proof from scratch.

#### Why the fused overhead is only `4,120 B`

The native 32-bit suffix from the same boundary is:

```text
levels = [4,096 B, 3,472 B, 3,216 B]
tail   = 26,952 B = 768 B commitment + 26,184 B packed digits
total  = 37,736 B.
```

The fused bridge proof is:

```text
levels = [4,720 B, 4,096 B, 3,472 B, 3,216 B]
tail   = 26,352 B = 768 B commitment + 25,584 B packed digits
total  = 41,856 B.
```

So the bridge overhead decomposes exactly as:

```text
extra recursive levels = 15,504 - 10,784 = 4,720 B
tail reduction         = 26,952 - 26,352 =   600 B
fused overhead         = 4,720 - 600     = 4,120 B.
```

This is the key structural explanation:

- the bridge does **not** create a long train of extra small-field levels,
- it effectively inserts one extra first 32-bit recursive level,
- and that extra level is partly paid back by a slightly smaller final tail.

More concretely, the native 32-bit suffix starts with:

```text
D=128 lb=3 m=7 r=4 na=2 nb=2 nd=1  -> 4,096 B
```

whereas the bridge proof starts with:

```text
D=128 lb=3 m=6 r=6 na=1 nb=2 nd=2  -> 4,720 B.
```

So the bridge pushes the first smaller-field level into a different local
optimum because:

- the bridge witness is larger (`315,296` instead of `199,584`),
- the carried bound is larger (`14` bits instead of `4` bits),
- but after that first level, the bridge schedule and the native 32-bit suffix
  almost coincide.

That is why the fused overhead stays so modest.

In short:

- the bridge is mostly "same witness, old commitment + new commitment,"
- the Nguyen carry layer is real but small,
- and the current byte model says the whole price is about one extra 32-bit
  recursive level.

#### End-to-end effect on the corrected 128-bit onehot proof

For the corrected 128-bit onehot `nv = 32` schedule, the prefix before this
late switch point is fixed:

```text
prefix_128 = L0 + L1 + L2 + L3 = 19,072 B.
```

From that point onward, the native 128-bit suffix is:

```text
128-bit suffix:
  levels = [5,072 B, 5,072 B, 5,072 B] = 15,216 B
  tail   = 41,344 B
  total  = 56,560 B.
```

The switched `128 -> 32` suffix is:

```text
switched suffix:
  bridge+32-bit levels = [4,720 B, 4,096 B, 3,472 B, 3,216 B] = 15,504 B
  tail                = 26,352 B
  total               = 41,856 B.
```

So the end-to-end proof sizes are:

```text
native 128-bit proof     = 19,072 + 56,560 = 75,632 B
with 128 -> 32 switch    = 19,072 + 41,856 = 60,928 B
net savings              = 14,704 B.
```

This is the cleanest way to read the current model:

- the switch does **not** magically shrink the shared witness `w` at the switch
  boundary,
- the bridge witness is actually larger than `w`,
- but the remainder of the proof is now charged at 32-bit rates,
- and almost all of the net gain comes from replacing a heavy 128-bit tail by
  a much cheaper 32-bit tail.

In this concrete case, compared to simply staying in 128 bits:

```text
extra recursive proof bytes  = 15,504 - 15,216 =   288 B
tail savings                 = 41,344 - 26,352 = 14,992 B
net savings                  = 14,704 B.
```

So the current evidence is that `128 -> 32` helps primarily because it is a
tail-lowering lever, not because the bridge itself compresses the boundary
witness.

### SwitchFold Reduction: Fuse The Bridge With The First Lower-Field Fold

The stronger reduction is to eliminate the intermediate lower-field commitment
boundary entirely.

The current two-step theory path is:

```text
shared w
  -> SwitchCommit_{128->32}
  -> native 32-bit level on the same w
  -> next 32-bit recursive state.
```

The more aggressive object is:

```text
shared w
  -> SwitchFold_{128->32}
  -> next 32-bit recursive state.
```

So the switch no longer outputs an intermediate commitment `C_lo` to the same
current witness `w`. It directly outputs the **post-first-32-bit-level**
recursive state.

This is the right fusion to study.

#### Statement shape

At the late 128-bit boundary, the incoming object is the full old recursive
state on the current flat witness `w`:

- old commitment `C_hi`,
- old opening point `r_hi`,
- old claimed opening value `y_hi`,
- and the current basis / shape metadata needed to interpret `w`.

The outgoing object should be the same thing an ordinary first 32-bit level
would have produced:

- the commitment to the next 32-bit recursive witness,
- the claimed evaluation for that next witness,
- and the sumcheck challenges that determine the next opening point.

So the `SwitchFold_{hi->lo}` theorem should be:

```text
SwitchFold_{hi->lo}(old public state, new post-fold public state) :=
  exists
    w,
    t_hat_hi, k_old_commit, f_old_commit,
    aux_open_hi,
    t_hat_lo, z_pre_lo, r_ct_lo
such that
  Small_A(w)
  and OldCommitRows_hi(w, t_hat_hi, k_old_commit, f_old_commit; C_hi)
  and OldOpenRows_hi(w, aux_open_hi; r_hi, y_hi)
  and FirstFoldRows_lo(w, t_hat_lo, z_pre_lo, r_ct_lo)
  and NextState_lo(t_hat_lo, z_pre_lo, r_ct_lo) = public outgoing state.
```

Interpretation:

- `OldCommitRows_hi` is the Nguyen-style old-modulus transport for the old
  commitment rows,
- `OldOpenRows_hi` is the corresponding transport for the old opening/evaluation
  claim,
- `FirstFoldRows_lo` is the ordinary first lower-field recursive relation on
  the same current witness `w`,
- `NextState_lo` is exactly the public state the ordinary lower-field verifier
  would carry after one native level.

So the proof is no longer merely
"same witness under two commitments."
It is
"same witness satisfies the full old recursive claim, including its opening
equation, and also one full native lower-field fold."

That is the real fusion target.

#### What `OldOpenRows_hi` should really contain

The cleanest rigorous formulation is that `OldOpenRows_hi` is **not** a single
flattened scalar row. It should package the actual old recursive opening
subsystem:

```text
OldOpenRows_hi(w, aux_open_hi; r_hi, y_hi)
  := exists y_ring_hi, v_hi, q_open_hi, f_open_hi
     such that
       Trace_hi(y_ring_hi, r_inner_hi, y_hi)
       and SplitRows_hi(w, y_ring_hi, v_hi, r_outer_hi; q_open_hi, f_open_hi).
```

Here:

- `Trace_hi` is the small linear check

```text
Tr(y_ring_hi * nu_hi(r_inner_hi)) = D_hi * y_hi,
```

- `SplitRows_hi` is the old split-basis ring-switch system with public outputs

```text
y_hi^* = [ v_hi, C_hi, y_ring_hi, 0, 0, ..., 0 ].
```

For one recursive claim, that split system has row families:

- `n_d_hi` public `v_hi` rows,
- `n_b_hi` commitment rows for `C_hi`,
- `1` public-output row carrying `y_ring_hi`,
- `1` challenge-fold row whose target is zero,
- `n_a_hi` `A` rows whose targets are zero.

This is the concrete Hachi object that should be transported through the
smaller-field Nguyen layer. It preserves the native sparse row structure and is
strictly better than collapsing the whole old opening claim to one dense
functional on the flat witness.

So the current ranking of candidate encodings for the missing old opening side
is:

1. **Best current target:** transport the full old split-basis row system plus
   one trace row.
2. **Acceptable but weaker abstraction:** keep `y_ring_hi` explicit and talk
   about `OldOpenRows_hi = {trace row + public-output row + challenge-fold row}`
   separately from the other old rows.
3. **Worst option:** eliminate `y_ring_hi` and flatten directly to one scalar
   functional `lambda_hi(r_hi) · w = y_hi`, which loses the structure that
   keeps row supports and coefficient bounds under control.

At the coefficient level, the opening-specific part of the old split system is
still very structured. If `rho_pub` is the batching weight of the public
`y_ring_hi` row and `rho_fold` is the batching weight of the challenge-fold
zero row, then the old opening coefficients seen by the witness segments have
the schematic form

```text
on w_hat[block, digit]:
  (rho_pub * b_hi[block] + rho_fold * c_hi_alpha[block]) * G_open[digit]
    + contributions from the old D-rows

on z_pre[k, fold_digit]:
  -(rho_fold * a_hi[block(k)] * G_commit[pos(k)] * G_fold[fold_digit])

on r_tail[row, level]:
  -(rho_row * (alpha_hi^D + 1) * G_r[level]).
```

This is exactly why the structured transport looks plausible:

- the opening point only enters through the explicit weight vectors `a_hi` and
  `b_hi`,
- the challenge dependence only enters through the sparse block scalars
  `c_hi_alpha[block]`,
- and the old opening transport never needs to materialize one giant dense
  coefficient vector of length `N`.

#### Repricing the missing old opening side

We can now plug this missing piece back into the `128 -> 32` late onehot model.

There are three natural cost models:

1. a **conservative native-row replay** model, where the old opening side is
   transported as:
   - old `D` rows,
   - old public `y_ring_hi` row,
   - old challenge-fold row,
   - one trace row,
   together with explicit old proof objects `v_hi` and `y_ring_hi`;
2. a **cleaner theorem-level direct-opening** model, where the old opening side
   is transported only as:
   - `D_hi` outer-evaluation rows producing `y_ring_hi`,
   - `1` trace row linking `y_ring_hi` to `y_hi`,
   - and the radix digits of the secret `y_ring_hi`.
3. a **fully flattened dense scalar-opening** model, where the old opening side
   is transported as one direct modular row

```text
lambda_hi(r_hi) · w = y_hi mod q_hi.
```

For the late `128 -> 32` onehot boundary (`N = 199,584`, `D_hi = 32`,
`n_a_hi = n_b_hi = n_d_hi = 2`, `delta_open_hi = 33`), the conservative script
model gives:

```text
radix B                     = 8192
m_q / m_k                   = 10 / 2
extra old opening rows      = 64 (D) + 32 (y_ring) + 32 (fold) + 1 (trace)
extra old explicit objects  = 1,536 B   (v_hi + y_ring_hi)
bridge witness              = 315,565
switched suffix             = 43,392 B
full proof                  = 19,072 + 43,392 = 62,464 B
net savings vs native 128   = 13,168 B
```

So even after restoring the missing old opening side in a fairly pessimistic
way, the bridge still looks clearly worthwhile.

The cleaner direct-opening model is better. Optimizing the radix under that
factorization gives:

```text
radix B                     = 32,768
m_q                         = 9
m_k_core / m_k_open         = 1 / 2
y_ring_hi digit witness     = 32 * 9 = 288
outer+trace q/carry payload = 33 * (9 + 2*2 - 1) = 396
bridge witness              = 314,188
switched suffix             = 42,448 B
full proof                  = 19,072 + 42,448 = 61,520 B
net savings vs native 128   = 14,112 B
```

So the missing old opening transport seems to cost only about:

```text
61,520 - 60,928 = 592 B
```

above the earlier commitment-core estimate if we use the cleaner direct-opening
factorization, and about:

```text
62,464 - 60,928 = 1,536 B
```

in the more conservative native-row replay model.

The dense scalar-opening model lands in between:

```text
radix B                     = 64
m_q                         = 22
m_k_core / m_k_open         = 3 / 4
bridge witness              = 315,709
switched suffix             = 42,928 B
full proof                  = 19,072 + 42,928 = 62,000 B
net savings vs native 128   = 13,632 B
```

So at this `128 -> 32` late boundary, the ordering is:

```text
direct_y_ring   : 61,520 B   (best)
direct_dense    : 62,000 B
replay_structured: 62,464 B
```

That is encouragingly stable. All three are good, and the best-vs-worst spread
is still under `1 KB`.

#### Preferred formal theorem: direct `w -> y_ring -> y`

For `128 -> 32` and `64 -> 32`, the best current theorem target is the
`direct_y_ring` factorization.

Let the old recursive opening point split as:

```text
r_hi = (r_inner_hi, r_outer_hi),
```

where:

- `r_inner_hi` has `log2(D_hi)` coordinates,
- `r_outer_hi` has the remaining recursive coordinates.

Define:

- `nu_hi = ReduceInner_hi(r_inner_hi) in R_hi`,
- `mu_hi(beta)` to be the coefficient weights induced by
  `sigma_{-1}(nu_hi)`,
- `lambda_hi^out(x)` to be the outer evaluation weights induced by
  `r_outer_hi`.

If the current flat witness is viewed as a table

```text
W_hi : [X_hi] x [D_hi] -> A,
```

then the semantic old opening claim is:

```text
y_ring_hi[beta] = sum_x lambda_hi^out(x) * W_hi(x, beta)          mod q_hi
D_hi * y_hi     = sum_beta mu_hi(beta) * y_ring_hi[beta]          mod q_hi.
```

The preferred old opening subrelation is therefore:

```text
OldOpenRows_hi^dyr(w, r_hi, y_hi) :=
  exists y_ring_hi, q_out, f_out, q_tr, f_tr
  such that
    for every beta in [D_hi]:
      sum_x lambda_hi^out(x) * W_hi(x, beta) - y_ring_hi[beta]
        = q_hi * Q_out_beta
    and
      sum_beta mu_hi(beta) * y_ring_hi[beta] - D_hi * y_hi
        = q_hi * Q_tr
    and
      each transported old row is certified by its radix/carry identity
      over base B.
```

Equivalently, each outer row and the trace row gets its own Nguyen-style
transport polynomial:

```text
P_beta(X) = Lambda_hat_beta(X) - Y_hat_beta(X) - q_hat_hi(X) * k_hat_beta(X)
          = (B - X) * f_hat_beta(X)

P_tr(X)   = Mu_hat(X) * Y_hat(X) - d_hat_hi(X) * y_hat_hi(X)
            - q_hat_hi(X) * k_hat_tr(X)
          = (B - X) * f_hat_tr(X).
```

Here:

- `Y_hat_beta` are radix encodings of the secret coefficients of `y_ring_hi`,
- `y_hat_hi` is the radix encoding of the public scalar `y_hi`,
- `d_hat_hi` is the radix encoding of the public integer `D_hi`,
- `Lambda_hat_beta` and `Mu_hat` are the public radix-encoded coefficient rows
  induced by `r_outer_hi` and `r_inner_hi`.

The full preferred switch theorem is:

```text
SwitchFold_{hi->lo}^{dyr}(C_hi, r_hi, y_hi ; pub_lo^+) :=
  exists
    w,
    aux_commit_hi,
    y_ring_hi, aux_open_hi,
    aux_lo
  such that
    Small_A(w)
    and OldCommitRows_hi(w, aux_commit_hi; C_hi)
    and OldOpenRows_hi^dyr(w, r_hi, y_hi; y_ring_hi, aux_open_hi)
    and FirstFoldRows_lo(w, aux_lo)
    and NextState_lo(aux_lo) = pub_lo^+.
```

#### Completeness sketch for `SwitchFold^{dyr}`

An honest prover can satisfy this relation as follows.

1. Start from the actual current witness `w`.
2. Compute the true old outer evaluation:

```text
y_ring_hi[beta] = sum_x lambda_hi^out(x) * W_hi(x, beta).
```

3. Compute the true old scalar opening:

```text
D_hi * y_hi = sum_beta mu_hi(beta) * y_ring_hi[beta].
```

4. Compute each old-row residual exactly and divide by `q_hi` to obtain the
   honest old quotient witnesses.
5. Radix-decompose those quotients and residual carries to obtain the honest
   Nguyen transport witness.
6. Run the ordinary first lower-field native fold on the same witness `w`,
   obtaining the honest `aux_lo` and outgoing recursive state `pub_lo^+`.

Every transported row is then an exact integer identity, so every smaller-field
modular check also passes.

#### Soundness sketch for `SwitchFold^{dyr}`

Assume the smaller-field verifier accepts and the no-wrap assumptions hold for:

- every old commitment transport row,
- every old direct-`y_ring` outer row,
- the old trace row,
- and the ordinary first lower-field fold rows.

Then soundness proceeds in four steps.

1. **Smallness of the shared witness.**
   Stage 1 / range proof gives one witness table `w in A^N`.

2. **Old commitment correctness.**
   By no-wrap on the transported old commitment rows, those rows hold as honest
   integer equalities, hence as valid old modular equations. Therefore

```text
Commit_hi(w) = C_hi.
```

3. **Old opening correctness via the intermediate `y_ring_hi`.**
   By no-wrap on the transported outer rows, for each `beta`:

```text
y_ring_hi[beta] = sum_x lambda_hi^out(x) * W_hi(x, beta)
```

as an integer identity, hence modulo `q_hi`.

   By no-wrap on the trace row:

```text
D_hi * y_hi = sum_beta mu_hi(beta) * y_ring_hi[beta]
```

   as an integer identity, hence modulo `q_hi`.

   Substituting the first family into the second gives:

```text
D_hi * y_hi
  = sum_beta mu_hi(beta) * sum_x lambda_hi^out(x) * W_hi(x, beta),
```

   which is exactly the semantic old scalar opening relation at `r_hi`. So the
   full old recursive claim `(C_hi, r_hi, y_hi)` is true for that same `w`.

4. **New lower-field fold correctness on the same witness.**
   By soundness of the ordinary lower-field fold rows, the same `w` yields the
   claimed outgoing recursive state `pub_lo^+`.

Because all three row families are checked against the same witness oracle `w`,
the prover cannot splice an old witness and a new witness together. Separate
batching coefficients for:

- old commitment rows,
- old opening rows,
- and new lower-field fold rows,

prevent cancellation between these families.

This is the core reason `direct_y_ring` is sound: `y_ring_hi` is not free
slack. It is a constrained intermediate variable pinned down on both sides.
Once those constraints hold, the scalar old opening claim follows exactly.

This is the most important numerical takeaway of this revision:

- adding the evaluation claim back in does **not** kill the `128 -> 32`
  switchfold story;
- the real uncertainty is no longer "does opening transport dominate?" but
  "which algebraic factorization of the old opening claim is the right theorem
  target?"
- for `32`- and `64`-bit destinations, the `w -> y_ring -> y` direct-opening
  factorization currently looks like the best default;
- for much smaller destinations like `19` bits, the structured replay remains
  the safer model, because the direct variants start running into no-wrap
  pressure or become infeasible.

#### What disappears

This reduction deletes the standalone intermediate lower-field commitment
boundary.

In particular, it removes the need to expose an intermediate public object

```text
C_lo = Commit_lo(w)
```

whose only purpose would have been to seed the first 32-bit native level.

Instead:

- `t_hat_lo` stays as prover witness,
- the first lower-field fold uses it internally,
- and the public output is only the **next** recursive state after that fold.

This is exactly the piece the current fused-overhead model still pays for.

#### Why this is plausible in Hachi

The reason this reduction looks realistic is structural:

- the old switch rows are linear in the shared current witness `w`,
- the first native lower-field fold is also stated over that same current
  witness `w`,
- both sides can share one norm check on `w`,
- and the old modular rows can be batched into the same stage-2 relation sum as
  extra row families.

So the first lower-field level does not need to come *after* the switch. It can
be part of the switch.

At the proof-system level, the right object is one custom lower-field level
whose stage-2 relation claim is a batched sum of:

```text
gamma_old * OldCommitRows_hi
  + gamma_lo * FirstFoldRows_lo.
```

The current balanced gadget still needs the ordinary stage-1 small-digit check
on `w`. But that stage-1 check is also shared once across both halves.

If a future top lower-field step uses the boolean gadget, the stage-1 check can
be collapsed further; that is an independent later optimization.

#### What the fused witness would look like

The prover-side witness for `SwitchFold_{128->32}` should be viewed as

```text
[ w
| t_hat_hi
| k_old
| f_old
| t_hat_lo
| z_pre_lo
| r_ct_lo ].
```

compared to the current bridge witness

```text
[ w | t_hat_hi | k_old | f_old | t_hat_lo ].
```

The key difference is conceptual:

- in the current bridge model, this large witness is recursively proved as an
  opaque object and only *then* the first native 32-bit fold happens,
- in `SwitchFold`, the lower-field fold auxiliaries are part of the same custom
  level, and the proof immediately outputs the post-fold recursive state.

So `SwitchFold` should be thought of as a **custom first 32-bit level with old
rows attached**, not as a bridge followed by an ordinary level.

#### Size model for the worked `128 -> 32` case

For the concrete onehot `nv = 32` example, the current switched suffix is:

```text
bridge + native 32 suffix
  = [4,720 B, 4,096 B, 3,472 B, 3,216 B] + tail 26,352 B
  = 41,856 B.
```

If `SwitchFold` replaces the first two steps by a single custom level of size
`L_sf`, then the switched suffix becomes:

```text
SwitchFold suffix = L_sf + 3,472 B + 3,216 B + tail.
```

Using the ordinary 32-bit continuation tail gives:

```text
SwitchFold suffix ≈ L_sf + 33,640 B.
```

Now the useful concrete anchor is:

- ordinary first 32-bit level from this boundary: `4,096 B`
- current bridge-compression level: `4,720 B`

and their difference is only:

```text
4,720 - 4,096 = 624 B.
```

That `624 B` delta comes from a very small change in the current planner model:

- one extra `v` ring vector (`+512 B`),
- one extra stage-2 round (`+48 B`),
- one extra stage-1 round (`+64 B`).

So the remaining bridge tax already looks like "one slightly fatter 32-bit
level," not a large hidden non-native blob.

This gives a plausible first band for the fused level:

```text
4,096 B <= L_sf <= 4,720 B
```

which implies the switched suffix should land around:

```text
37,736 B <= switched suffix <= 38,360 B.
```

Adding back the fixed 128-bit prefix `19,072 B` gives the end-to-end estimate:

```text
56,808 B <= full proof with SwitchFold <= 57,432 B.
```

Compared to the native corrected 128-bit proof `75,632 B`, this is a net gain
of about:

```text
18.2 KB to 18.8 KB.
```

So `SwitchFold_{128->32}` looks strictly better than the current separate
bridge model, and it appears capable of recovering most or all of the remaining
`4.1 KB` bridge tax.

#### Comparison to native `32` and `32b-bool`

There are really two different lower bars:

- native corrected `32`-bit: `49,568 B`
- native mixed-boolean `32b-bool`: about `46.4 KB`

For the worked `128 -> 32` switchfold estimate

```text
56,808 B <= full proof with SwitchFold <= 57,432 B
```

the remaining gaps are:

```text
vs native corrected 32-bit :  7.2 KB to  7.9 KB
vs native mixed 32b-bool   : 10.4 KB to 11.0 KB.
```

The important structural fact is that at this **late** switch boundary,
the best continuation under the ordinary `32` profile and the `32b-bool`
profile is the same:

```text
best suffix from (w = 199,584, prev_bound = 4):
  PROFILE_32   = 37,736 B
  PROFILE_32b  = 37,736 B.
```

So once the switch has reached this boundary, the boolean gadget has no further
advantage to exploit. The post-switch continuation is already all-balanced.

That means the residual gap to native `32b-bool` is **not** a switchfold gap.
It is an early-prefix gap.

More concretely:

```text
native corrected 32-bit total     = 49,568 B
native mixed 32b-bool total       = about 46.4 KB
difference                        = about 3.2 KB.
```

So the `32b-bool` advantage comes from the very top of the proof, where it gets
to use its cheap boolean levels from the beginning.

The `128 -> 32` switchfold can never recover that part, because by the time the
switch happens the carried witness is already a balanced recursive witness, not
a root boolean table.

So the remaining delta to native `32b-bool` should be read as:

```text
about 7.2 to 7.9 KB  = irreducible 128-bit prefix tax vs native 32
about 3.2 KB         = lost early boolean-root advantage
-------------------------------------------------------
about 10.4 to 11.0 KB total gap vs native 32b-bool.
```

#### Direct `128 -> 19` looks better on raw tail, but not yet end-to-end

The natural next question is whether we should jump directly from `128` bits to
the packed `19`-bit threshold-prime field (`k7-pack`) in order to exploit its
smaller tail.

At the same late boundary `(w = 199,584, prev_bound = 4)`, the native lower
suffixes are:

```text
native 32 suffix     = 37,736 B
native 19-pack suffix = 34,768 B
```

So at the pure destination-suffix level, `19` is indeed strictly better by:

```text
37,736 - 34,768 = 2,968 B.
```

That improvement comes from exactly the expected place:

```text
32 suffix:
  recursive levels = 10,784 B
  tail             = 26,952 B

19-pack suffix:
  recursive levels = 13,392 B
  tail             = 21,376 B
```

So `19` buys about `5.6 KB` of tail savings, but gives back about `2.6 KB` in
more expensive recursive levels, for a net native-suffix gain of only about
`3.0 KB`.

Now compare the direct bridge models:

```text
128 -> 32 separate bridge:
  fused overhead = 4,120 B
  net savings    = 14,704 B

128 -> 19 separate bridge:
  fused overhead = 8,176 B
  net savings    = 13,616 B
```

So under the current separate-bridge model, direct `128 -> 19` is actually
**worse** than direct `128 -> 32` by about:

```text
14,704 - 13,616 = 1,088 B.
```

Equivalently, the end-to-end totals are:

```text
native 128-bit proof      = 75,632 B
with 128 -> 32 bridge     = 60,928 B
with 128 -> 19 bridge     = 62,016 B.
```

This is the cleanest evidence that the extra emulation complexity is not merely
a prover-time issue. It already shows up in proof size.

For this direct `128 -> 19` bridge, the best radix drops all the way to:

```text
B    = 16
m_q  = 32
m_K  = 4
```

compared to the `128 -> 32` bridge:

```text
B    = 128
m_q  = 19
m_K  = 3.
```

So the old-modulus transport layer needs many more digits. Those digits are not
just local prover work. They become part of the recursively proved witness.

Concretely:

```text
128 -> 32 old-modulus aux:
  k_old + f_old = 3,072

128 -> 19 old-modulus aux:
  k_old + f_old = 4,992.
```

And the first lower-field bridge-compression level gets noticeably heavier:

```text
128 -> 32 first bridge level: D=128 ... -> 4,720 B
128 -> 19 first bridge level: D=256 ... -> 6,640 B.
```

So the cost of "more limbs" lands in all three places:

- **prover time**: more quotient/carry digits to generate and prove,
- **proof size**: larger recursive witness, larger first lower-field level, and
  in this case even a larger bridged tail,
- **verifier time**: larger proof objects to read and check, more ring-vector
  openings, and more/larger sumcheck messages in the first bridged level.

In other words, more limbs are not just a hidden prover-side nuisance.

#### Why direct `128 -> 19` may still be interesting later

Even though the current separate-bridge result favors `128 -> 32`, direct
`128 -> 19` is not dead.

The reason is that the native `19`-bit destination really is better than the
native `32`-bit destination by about `3 KB` at this boundary. So if a true
`SwitchFold_{128->19}` can eliminate most of the standalone bridge tax, it
could still beat `SwitchFold_{128->32}`.

So the current read is:

- separate bridge model: `128 -> 32` wins
- native destination suffix: `19` wins
- true switchfold comparison: still open

That makes `128 -> 19` a plausible **second-wave** target, but not the best
default first target.

#### Main soundness obligations

This reduction still has real proof obligations.

The important ones are:

1. one shared-witness theorem:
   the verifier must conclude that the *same* current witness `w` satisfies
   both the old 128-bit commitment rows and the first 32-bit fold relation.
2. transcript binding without intermediate `C_lo`:
   the lower-field transcript must bind the old public state and the outgoing
   post-fold lower-field state directly, rather than deriving challenges from an
   intermediate commitment boundary that no longer exists.
3. old-row batching versus lower-field rows:
   the old Nguyen rows and the ordinary lower-field fold rows need separate
   batching coefficients to avoid cancellation.
4. canonical old public encoding:
   the old 128-bit coefficients must still enter through integer/radix
   encodings, never as fake native 32-bit scalars.

None of these look like conceptual blockers. They are the right theorem work.

#### Default planning target

The current evidence now points to a clear default target:

```text
SwitchFold_{128->32}
```

rather than

```text
SwitchCommit_{128->32} followed by an ordinary 32-bit level.
```

So for note planning and eventual implementation planning, the default cutover
target should be:

- one custom `128 -> 32` switchfold level,
- no intermediate public 32-bit commitment boundary,
- output equal to the post-first-32-bit recursive state,
- and only after that continue with ordinary native 32-bit recursion.

### Planner Design For Sweeping Switching Points

The right planner architecture is a two-layer one.

#### 1. Native per-profile DP stays mostly unchanged

For each native field profile, we already have:

```text
NativeSuffix(profile, w_len, prev_bound)
```

which returns the optimal native suffix from a commitment-boundary state.

That object should remain the primitive building block.

#### 2. Add a switchfold transition graph on top

What is missing is a meta-planner over profile transitions:

```text
128 -> 64 -> 32 -> 19/16
```

At a commitment boundary, the planner should consider:

- continue natively in the same profile, or
- fire a `SwitchFold_{hi->lo}` edge into a smaller profile.

So the top-level recurrence should look like:

```text
Best(profile_hi, boundary_state) =
  min(
    NativeSuffix(profile_hi, boundary_state),
    min_lo SwitchFoldCost(boundary_desc_hi, profile_lo)
           + NativeSuffix(profile_lo, post_switchfold_state_lo)
  ).
```

The key point is that the switch cost depends not only on
`(w_len, prev_bound)`, but also on the **boundary descriptor** of the just
completed high-field level.

That descriptor must include at least:

- `D_hi`
- `n_a_hi`
- `n_b_hi`
- `r_hi`
- `num_blocks_hi = 2^r`
- `delta_open_hi`
- `eta_hi`
- witness kind / alphabet tag

because those are exactly the quantities that determine the Nguyen support
bound and therefore the switch cost.

#### 3. Witness kind must be part of the boundary state

This is the most important design constraint.

The planner cannot just track `(w_len, prev_bound)` if it wants to sweep switch
points across gadget families.

It also needs an explicit tag for the witness semantics, e.g.

```text
witness_kind in {root_boolean, recursive_boolean, balanced(lb), ... }.
```

Reason:

- native `32b-bool` wins by using boolean levels at the very top,
- but after the late `128 -> 32` boundary, the current witness is not boolean,
- and indeed the best `32` and `32b` continuations from that boundary are
  identical.

So boolean transitions must only be legal when the incoming witness kind really
supports them.

Without this tag, a switch planner would overestimate what the lower profile can
do after switching.

#### 4. Good first implementation strategy

The clean incremental plan is:

1. keep the existing native DP untouched,
2. add a boundary extractor that records every realized level boundary along an
   optimal native schedule,
3. run a one-switch sweep over those realized boundaries,
4. once that works, generalize from "chosen native boundaries" to all memoized
   boundary states,
5. then add multiple switch edges and let the meta-planner optimize over
   `128 -> 64 -> 32 -> 19/16`.

So the first useful sweep is not a full giant DP rewrite. It is:

```text
for each boundary on the best native 128-bit schedule:
  estimate SwitchFold_{128->32}
  compare end-to-end total
  pick the best boundary
```

That is already enough to test whether the currently chosen late switch really
is best among the natural boundaries on the optimal 128-bit path.

#### 5. Default planner target

So the default planner cutover should now be:

- native profile DP remains the core engine,
- one-switch sweep over realized commitment boundaries comes next,
- boundary states carry a witness-kind tag,
- switch edges are priced as `SwitchFold`, not `SwitchCommit`.

#### Updated practical read

Combining the bridge budgets with the explicit fused bridge model:

- `32 -> 19/16`: still not promising in the current model
- `64 -> 32`: promising only at the right boundary
- `128 -> 64`: plausible
- `128 -> 32`: clearly the best current target

So the best near-term theory target is no longer just
"modulus switching in general."
It is:

```text
build a good commitment-boundary bridge for 128 -> 32,
and only then decide how aggressively to stage lower.
```

## Construction 3: Joint Modulus-And-Dimension Switch

If the ring degree also changes, the same switch framework still applies.

The shared witness should be the pre-chunk flat coefficient table `w`.
Then:

- the old commitment map `G_hi` uses the old ring degree `D_hi`,
- the new commitment map `G_lo` uses the new ring degree `D_lo`.

If the change of ring degree is implemented via a public linear isomorphism or
subring map, the switch relation simply includes that map in the definition of
`G_hi` or `G_lo`.

So the joint lifted relation has both quotient types:

```text
Lift(old relation) = (X^(D_hi) + 1) * R_X + q_hi * R_q
Lift(new relation) = (X^(D_lo) + 1) * S_X + q_lo * S_q
```

The conceptual point is:

- `X^(D) + 1` quotients handle ring / negacyclic wrap,
- `q` quotients handle coefficient-field reduction.

This is exactly the "it is both" instinct you raised.

## Construction 4: Universal Integer Layer

A more radical option is to define the switch around a universal integer-level
object.

Very roughly:

- commit once to the flat witness table `w` in a large integer-oriented
  commitment language,
- derive old and new PCS commitments as modular views or reductions of that same
  committed object,
- prove that both old and new commitments/openings are consistent with one
  underlying integer witness.

This is attractive because it turns switching into
"different modular views of one integer commitment."

The downside is that it asks for a more global redesign, and the universal layer
may carry costs that erase the modular savings.

Still, as a pure theory direction, it is worth keeping on the table.

## Construction 5: Composite Or CRT Bridge

LNP22 also keeps open two more special-case patterns.

### Composite bridge modulus

Use a one-off bridge proof system over a modulus `Q` that sees the relevant
factors at once, for example:

- a modulus divisible by `q_lo`,
- or even by both `q_hi` and `q_lo`.

Then prove switch consistency there, and resume recursion natively over `K_lo`.

### CRT multi-prime bridge

Instead of one bridge modulus, prove the lifted old relation modulo several
small primes whose product exceeds the required no-wrap bound, then recover the
integer truth by CRT.

These are less elegant than the first three constructions, but they are real
options.

The composite / CRT bridge family is probably best viewed as:

- a boundary gadget,
- not the native recursion language for the rest of the proof.

## Optional Micro-Optimization: Inverse-And-Smallness Instead Of Explicit Quotients

LNP22 also suggests an alternative to materializing every `q_hi * k` quotient.

If the proof modulus is coprime to `q_hi`, one can sometimes prove that:

```text
q_hi^(-1) * F mod p
```

is small, rather than explicitly committing to the full quotient witness
`k = (F / q_hi)`.

For some switch rows, especially if they are numerous, that may save witness
size.

So even within the "same-witness dual linear bridge" family, there are at least
two sub-variants:

- explicit quotient witnesses,
- inverse-and-smallness witnesses.

## Extension-Field States

If a switched segment uses extension-field challenges or evaluations, then the
old public quantities are not single prime-field elements. They must be encoded
in a fixed basis first.

So for `K_hi = F_(p_hi^t)`:

1. choose a canonical basis of `K_hi` over `F_p_hi`,
2. encode the basis coefficients canonically,
3. only then apply the radix / carry machinery.

The important point is that
"about 128 bits"
is not a proof object. A fixed basis encoding is.

## Cost Model

There are two different cost pictures, and it is important not to mix them up.

### Generic non-native arithmetic picture

If we had to emulate arbitrary old-field multiplication on unknown values inside
the smaller field, then the limb base would be controlled by no-wrap:

```text
B = O(sqrt(q_lo)).
```

This leads to the rough arithmetic limb counts:

| switch | rough arithmetic limbs |
|---|---:|
| `128 -> 64` | about `4` |
| `64 -> 32` | about `4` |
| `32 -> 19` | about `4` |
| `32 -> 16` | about `4` |
| `128 -> 32` | about `8` |
| `128 -> 19` | about `14-15` |

Since multiplication cost is roughly quadratic in limb count, staged descent is
far better than one giant jump.

Very roughly at the partial-product layer:

- direct `128 -> 19`: about `15^2 = 225`
- staged `128 -> 64 -> 32 -> 19`: about `3 * 4^2 = 48`

This is not a proof-size formula, but it is the right structural reason staged
switching beats a one-shot switch.

### Hachi-specific switch picture

At the Hachi boundary we care about, the old claim is linear in the small-digit
witness.

So the dominant terms become:

- transporting old public coefficients,
- commitment-row count,
- carry polynomials,
- quotient witnesses,
- range checks on `w`,
- extension-coordinate overhead if present.

This is much better than generic witness-witness multiplication, and it is the
main reason a smaller-field Hachi switch now looks genuinely plausible.

## Best Current Construction Menu

### 1. Larger-field translator over the shared small witness

Pros:

- strongest direct literature support,
- conceptually clean,
- easy to state rigorously,
- naturally supports both modulus and ring-dimension switching.

Cons:

- switch proof itself still pays large-field cost.

### 2. Smaller-field carry-polynomial switch over the shared small witness

Pros:

- most aligned with the actual goal of modulus lowering,
- Nguyen gives a concrete template,
- Hachi boundary linearity makes it much better than generic non-native
  arithmetic.

Cons:

- most delicate theorem,
- needs careful batching and carry accounting.

### 3. Joint `q` and `D` switch

Pros:

- very natural for Hachi,
- same-witness statement still works,
- "both moduli at once" is the right algebraic picture.

Cons:

- more bookkeeping,
- more places to make a same-witness mistake.

### 4. Universal integer layer

Pros:

- elegant conceptual unification,
- could make switching a derived feature rather than a special gadget.

Cons:

- most speculative,
- likely a large global redesign.

### 5. Composite / CRT bridge

Pros:

- directly tied to LNP22 mixed-modulus patterns,
- may offer a clean special-case bridge.

Cons:

- awkward as a native recursion language,
- likely only useful as a boundary trick.

## Main Footguns

- Proving the old claim for one witness and the new claim for another.
- Treating old-field values as if they were native smaller-field scalars.
- Expanding old ring rows all the way to coefficient basis and accidentally
  paying an extra `D_hi` factor in the no-wrap bound when the split / NTT basis
  version would have been viable.
- Forgetting that the switch must output a fresh outgoing native claim, not just
  verify the incoming old one.
- Forgetting one of the two no-wrap layers:
  - modulo `q`
  - modulo `X^n + 1`
- Using raw storage packing as if it were arithmetic-safe packing.
- Forgetting fixed-basis encoding for extension-field states.
- Defining "same witness" after different chunkings instead of at the shared flat
  coefficient table.

## Best Current Theoretical Picture

The strongest updated picture is:

1. Place the switch at the recursive commitment boundary, where the old segment
   has already produced the flat small-digit witness `w`.
2. Define the switch theorem over one shared witness table `w`.
3. Relate the old and new recursive commitments to that same `w`, and let the
   new native segment sample its own fresh `r_lo` afterwards.
4. State the old side in split / NTT basis so the relevant support parameter is
   the number of live ring inputs in a row, not `D_hi` times that number.
5. Choose one of two serious proof styles:
   - prove the translator while still in the larger field, or
   - prove it in the smaller field using carry-polynomial linear relation proofs.
6. Use staged descent
   `128 -> 64 -> 32 -> 19/16`
   rather than a giant jump.
7. Expect the last drop below `32` bits to be the real algebraic bottleneck,
   and expect late hybrid switches to be easier than early ones.

## Open Questions

- Which exact recursive commitment boundary is best in practice:
  immediately after the boolean prefix, or only later near the tail?
- How aggressively can old commitment and evaluation rows be batched before the
  carry witnesses get too large?
- For the smaller-field switch, is explicit quotient witnessing or
  inverse-and-smallness the better row compression strategy?
- For the `32 -> 19` or `32 -> 16` step, does the one-time switch overhead get
  repaid quickly enough by the cheaper native suffix?

## Conclusion

The detailed picture is now much better than the earlier vague note.

The most important updates are:

- a Hachi switch should be defined over one shared flat small-digit witness,
- the old and new claims can be made linear in that witness,
- the recursive commitment boundary is the right concrete cut for the first
  serious instantiation,
- changing modulus and changing ring dimension can be handled in one translator,
- and Nguyen-style carry polynomials give a concrete smaller-field switch
  template rather than a hand-wavy non-native arithmetic story.

The next natural step is to take one specific switch, probably `32 -> 19` or
`32 -> 16`, and write down the exact public variables, witness variables, and
lifted row equations for the commitment-boundary bridge:

- old commitment rows,
- new commitment rows,
- carry / quotient witnesses,
- and the batching strategy.

Only after that should we decide whether a more generic opened-claim bridge is
still worth carrying as a second construction.
