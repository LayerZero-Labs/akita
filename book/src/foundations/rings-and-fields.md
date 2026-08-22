# Cyclotomic rings and extension fields

Akita uses three related kinds of number system. Each one has a different job.

- A prime field stores the coefficients supplied by an application.
- An extension field supplies large random challenges and evaluation points.
- A cyclotomic ring groups many field coefficients into one object for lattice
  commitments and fast polynomial arithmetic.

This chapter builds those objects from the ground up. It then explains how the
current Rust types represent them.

## Prime fields

Choose a prime number $q$. The prime field $\mathbb{F}_q$ contains the integers
from $0$ through $q-1$, with every operation reduced modulo $q$.

For example, in $\mathbb{F}_{17}$,

$$
15 + 5 = 3
\qquad\text{and}\qquad
4 \cdot 5 = 3.
$$

The first equality holds because $20$ and $3$ differ by $17$. The second holds
because $20$ is also congruent to $3$ modulo $17$.

Every nonzero field element has a multiplicative inverse. This is what makes a
field different from a general ring. Division by a nonzero element means
multiplication by its inverse.

Akita ships configurations over three prime sizes:

| Family | Base field size | Evaluation field |
| --- | ---: | --- |
| `fp32` | About 32 bits | Degree 4 extension of the base field |
| `fp64` | About 64 bits | Degree 2 extension of the base field |
| `fp128` | About 128 bits | The base field itself |

The exact prime belongs to the configuration. For example, the current fp128
family uses $q = 2^{128} - 2^{32} + 22537$.

## Extension fields

An extension field adds new elements while keeping the original field as a
subfield. If $E$ is a degree $K$ extension of $\mathbb{F}_q$, then $E$ has
$q^K$ elements. Each element can be written with $K$ base field coordinates.

For a degree 2 extension, an element has the form

$$
a_0 + a_1 u,
$$

where $a_0,a_1 \in \mathbb{F}_q$ and $u$ satisfies a fixed irreducible
quadratic equation. The equation defines how powers of $u$ reduce during
multiplication. A degree 4 element has four coordinates.

Akita uses extension fields because challenge size controls the probability
that a false polynomial identity passes a random evaluation. A 32 bit base
field is useful for fast arithmetic, but a single 32 bit challenge is too small
for Akita's security target. A degree 4 extension gives a challenge field with
about 128 bits. The fp64 family reaches the same size with a degree 2
extension. The fp128 family needs no extension.

The code records this relationship through `ExtField<F>`. Its
`DEGREE` constant gives $K$. `from_base_slice` and `to_base_vec` convert
between an extension element and its canonical base field coordinates.

## Coefficients and evaluation points have different roles

A polynomial can have coefficients in the base field and still be evaluated
at a point in an extension field.

Suppose

$$
f(X) = 3 + 5X
$$

has coefficients in $\mathbb{F}_q$. If $r$ belongs to an extension field $E$,
then $f(r)=3+5r$ also belongs to $E$. The base values are embedded as constant
elements of $E$ before the arithmetic is performed.

This is the common small field case in Akita. The committed table contains
values in $\mathbb{F}_q$, while sum-check challenges and claimed evaluations
live in $E$. Keeping these roles separate gives the prover fast base field
storage and gives the verifier a large challenge space.

The [extension-opening reduction](./extension-opening-reduction.md) explains
how Akita later turns such an extension field claim into the packed base field
claim needed by the folding protocol.

## The cyclotomic ring

For a power of two $D$, Akita uses the ring

$$
R_q = \mathbb{F}_q[X] /(X^D+1).
$$

An element is a polynomial with at most $D$ coefficients:

$$
a_0 + a_1X + \cdots + a_{D-1}X^{D-1}.
$$

The quotient by $X^D+1$ means that $X^D=-1$. Any term whose degree reaches
$D$ wraps around with a minus sign. This is called negacyclic reduction.

Take $D=4$ as a small example. Then

$$
X^3 \cdot X^2 = X^5 = X(X^4) = -X.
$$

The implementation stores the coefficients in ascending degree order. The
ring element $2+3X-X^3$ is stored as `[2, 3, 0, -1]`, with each entry reduced
in the base field.

`CyclotomicRing<F, D>` owns this representation. Its ordinary multiplication
computes negacyclic convolution. Its `sigma` method applies an automorphism
$X \mapsto X^k$ for an odd $k$. Its `sigma_m1` method applies
$X \mapsto X^{-1}$, which appears in inner products and opening reduction.

## Why one ring element holds many coefficients

A matrix over $R_q$ is also a structured matrix over $\mathbb{F}_q$. One ring
multiplication acts on $D$ base field coefficients at once. This structure is
what lets Akita use Module-SIS commitments without storing an unrelated scalar
matrix entry for every pair of field coordinates.

The ring dimension is not an application parameter. Generated schedules choose
it for each matrix role and fold level. Current production schedules begin at
$D=64$ and may use larger powers of two. Different roles in one proof may use
different dimensions.

## Splitting the ring for fast arithmetic

Some primes let $X^D+1$ factor into smaller irreducible polynomials. Suppose
$k$ is a power of two and

$$
q \equiv 2k+1 \pmod{4k}.
$$

Then $X^D+1$ splits into $k$ irreducible factors of degree $D/k$. The Chinese
remainder theorem identifies the ring with a product of $k$ smaller extension
fields:

$$
R_q \cong \prod_{j=1}^{k} \mathbb{F}_{q^{D/k}}.
$$

Arithmetic can move to these factors, perform smaller pointwise operations,
and reconstruct the result. Akita's NTT and CRT kernels use this structure.
The [NTT and CRT chapter](./ntt-crt.md) explains the transform and the exact
integer reconstruction paths.

The amount of splitting also affects security conditions. A larger $k$ can
make arithmetic faster, but it tightens the coefficient bound that guarantees
a short nonzero ring element is invertible. Akita fixes the prime and challenge
families together, then validates them as configuration data.

## Embedding the challenge field in the ring

The extension field used by a small field configuration must interact with the
ring operations used by the folding protocol. Akita uses a genuine subfield of
$R_q$, rather than treating a list of extension coordinates as unrelated ring
coefficients.

For the power of two extension degrees used by Akita, this subfield has a fixed
basis inside the ring. The Rust types `FpExt2`, `FpExt4`, and `FpExt8` implement
the corresponding arithmetic. The degree 4 and degree 8 types use the same
cyclotomic subfield basis used by trace reduction. There is no second quartic
representation that an integrator must choose between.

This alignment provides two operations used by the protocol:

- A base field value embeds into the constant coordinate of the extension.
- An extension field vector can be packed into the ring so that a trace of a
  ring product recovers the intended field inner product.

The implementation keeps one canonical coordinate order for both operations.
Serialization and packed arithmetic use that same order.

## Centered coefficients and norms

Security bounds treat a field residue as a small signed integer when possible.
The centered representative of a value modulo $q$ is the unique integer in
$(-q/2,q/2]$ that represents it.

For a ring element with centered coefficients $(a_0,\ldots,a_{D-1})$, Akita
uses three common norms:

$$
\lVert a\rVert_\infty = \max_i |a_i|,
$$

$$
\lVert a\rVert_1 = \sum_i |a_i|,
$$

$$
\lVert a\rVert_2 = \sqrt{\sum_i a_i^2}.
$$

The infinity norm controls the largest coefficient. The one norm controls a
simple bound on how much multiplication by a sparse challenge can increase a
coefficient. The Euclidean norm gives a tighter whole-vector measure when the
protocol proves the corresponding physical norm.

For example,

$$
\lVert ab\rVert_\infty
\leq
\lVert a\rVert_1\lVert b\rVert_\infty.
$$

This inequality explains why Akita records the one norm of a fold challenge
and the infinity norm of a response. Their product bounds the coefficients of
the resulting collision vector.

## Challenge subrings and coefficient packing

Some folds use a challenge in a smaller ring while the committed data remains
in a larger ring. Let the extension field have degree $K$, and suppose
$D=Khs$. Akita uses

$$
S=\mathbb{F}_q[Y]/(Y^s+1)
$$

as the challenge subring and embeds it into $R_q$ by

$$
Y \mapsto X^{Kh}.
$$

This map is valid because $(X^{Kh})^s=X^D=-1$. It inserts the $s$ challenge
coefficients at fixed positions in the $D$ coefficient ring. The other
coefficient lanes remain explicit.

The embedding preserves the coefficient norms of the challenge. It also
preserves invertibility. These properties let the protocol use a compact
challenge without changing the security bound applied after embedding.

The [root fold and ring switching](../how/proving/root-fold-ring-switch.md#subring-coefficient-packing)
chapter shows the full coefficient grid and the current packed proof relation.

## Implementation and review map

| Question | Primary source |
| --- | --- |
| How is a ring element stored and multiplied? | `crates/akita-algebra/src/ring/cyclotomic.rs` |
| Which extension types exist? | `jolt-field::solinas` (`FpExt2`, `FpExt4`, and `FpExt8`) |
| How are base values lifted into an extension? | `jolt_field::ExtField` and the Solinas extension implementations |
| Which field and extension does a preset use? | `crates/akita-config/src/proof_optimized/` |
| Which ring dimensions may a preset schedule use? | The `A_RING_DIMENSIONS`, `B_RING_DIMENSIONS`, and `D_RING_DIMENSIONS` declarations in each preset |
| How is subring coefficient packing validated? | `crates/akita-types/src/subring_coefficient_packing.rs` |

A review should check that all representations agree on four facts. The prime
must match the selected modulus profile. Extension coordinates must use one
canonical basis and order. A schedule may select only dimensions admitted by
its configuration. Every optimized ring operation must match the scalar
negacyclic result.
