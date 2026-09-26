# Trinomial ring arithmetic

These APIs require the opt-in `labinius-trinomial` feature on `akita-algebra`.
Enabling the feature only makes the arithmetic available; it does not select a
new setup, commitment, or proof protocol.

Akita's algebra crate provides arithmetic for two related quotient rings:

\[
R_D^+ = \mathbb F_q[X]/(X^D+X^{D/2}+1)
\quad\text{and}\quad
R_D^- = \mathbb F_q[X]/(X^D-X^{D/2}+1).
\]

Both rings store exactly `D` coefficients in ascending degree order. The
sealed `PlusTrinomial` and `MinusTrinomial` types fix the sign in the modulus,
so a value or transform cannot silently change between the two shapes.
Construction rejects zero and odd degrees.

These types are arithmetic foundations. No production protocol configuration
selects them yet.

## Multiplication and quotient construction

`TrinomialRing::schoolbook_mul` convolves two coefficient vectors and reduces
the result by the selected monic trinomial. This path is deliberately simple:
tests use it as an independent oracle for the faster transform multiplication.

`reduce_product_with_quotient` also returns the polynomial quotient. For an
input polynomial `P`, it constructs `A` and `Q` such that

\[
P=A+(X^D+sX^{D/2}+1)Q,
\qquad s\in\{1,-1\},
\]

where `A` is the returned ring element. The routine accepts at most the
`2D - 1` coefficients produced by multiplying two degree-`D` ring elements.

## Full-split transform

Write `h = D / 2` and treat a polynomial as

\[
A(X)=A_0(X)+X^h A_1(X).
\]

The trinomial modulus has two root cosets. The implementation first combines
the low and high coefficient halves with each coset value. It then performs
one size-`h` `SmoothDomain` transform per coset and stores the positive-shift
coset before the inverse-shift coset. Multiplication becomes a pointwise
product of all `D` slots.

The plus shape requires a root of order `3h`. The minus shape requires a
root of order `6h`. `TrinomialNttDomain::new` checks that the field's declared
smooth subgroup contains the required root before it builds the transform.
The initial supported shapes are:

| Field profile | Plus degrees | Minus degrees |
| --- | --- | --- |
| `Prime64Offset23703` | 162 | 324, 648 |
| `Prime128OffsetA7F7` | 162, 486 | 324, 648 |

The generic implementation can accept another even degree when the same root
condition holds.

`TrinomialNttDomain::forward`, `inverse`, and `multiply` allocate temporary
scratch for convenience. Repeated operations should allocate one typed
`TrinomialNttWorkspace` with `workspace()` and call the corresponding
`*_with_workspace` methods. The workspace carries the field, degree, and
modulus shape in its type and reuses both coset buffers and the underlying
mixed-radix FFT scratch.

For a cached matrix multiplied by repeated balanced signed-digit vectors,
`prepare_i8_lut(log_basis)` prepares the active range
`[-2^(log_basis - 1), 2^(log_basis - 1))`. The table stores each digit already
scaled for both root cosets at every coefficient position. Its payload is
`2 * D * 2^log_basis * size_of::<F>()` bytes, so callers should prepare only
the basis they use and amortize setup across many transforms. The checked
transform rejects an out-of-range digit before changing its destination.
`forward_i8_with_lut_into_workspace` reuses both that destination and the
typed transform workspace; it performs no hot-path allocation.

Cached matrix entries remain in `TrinomialNtt` form. Pointwise accumulation
uses the field's fused `mul_add` when available. An explicit packed variant
uses the SIMD backend selected by `jolt-field`; the faster choice depends on
the field and target architecture, so benchmarked callers select it rather
than changing arithmetic semantics through runtime dispatch.

## Packing scalar components

Let `d` be the scalar-ring degree and `r = D / d` the packing rank. For a
power-of-two rank greater than one, packing `r` values `a_t` from the plus
scalar ring uses

\[
\sum_{t=0}^{r-1}Y^t a_t(-Y^r)\in R_D^-.
\]

Thus scalar coefficient `a[t, j]` occupies target position `r * j + t`, with a
minus sign when `j` is odd. The substitution is a ring map because `d / 2`
is odd for the supported scalar degrees. `embed_scalar` uses the `t = 0`
coordinate, and `unpack_scalar_components` inverts the same interleaving.

Rank one is a separate boundary case: source and target are both plus rings,
and packing is the identity variable map. It does not negate odd
coefficients. The checked API rejects a plus target at higher rank, a minus
target at rank one, non-power-of-two higher ranks, and inconsistent degrees.

## The 64-bit arithmetic profile

`Prime64Offset23703` names the existing generic `jolt-field` representation
for

\[
q=2^{64}-23703.
\]

Its configured element `15113553820101191969` has exact order 1,944, or
`2^3 * 3^5`, which supplies the roots needed through the degree-648
minus ring. Tests replay a recursive Lucas primality certificate with exact `u128`
arithmetic. They also check the declared root and every supported derived root
independently with `u128` modular arithmetic.

The default quadratic relation `u^2 = 2` is reducible over this prime. The
valid `Prime64Offset23703Ext2` profile instead uses `u^2 = 5`. Tests check that
5 is a quadratic non-residue and cover basis multiplication, inverses, norms,
and Frobenius behavior. This extension profile is available for arithmetic
experiments; it is not registered as a protocol field.

**Code:** `crates/akita-algebra/src/ring/trinomial.rs` and
`crates/akita-algebra/src/fft/fields.rs`.

**Tests:** `crates/akita-algebra/src/ring/trinomial/tests.rs` and
`crates/akita-algebra/src/fft/fields/tests.rs`.

**Benchmarks:** `crates/akita-algebra/benches/trinomial_ntt.rs` compares the
workspace-based transforms and multiplication against schoolbook
multiplication. `crates/akita-algebra/benches/ntt_comparison.rs` compares
conversion, prepared pointwise work, and a checked bounded matrix-vector
workload with the existing CRT NTT backends. Setup time and prepared storage
are reported separately from the hot matrix-vector operation.
