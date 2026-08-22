# Gadget decomposition

Akita often works with coefficients, the individual numbers in a polynomial,
that can be almost as large as the field modulus. The modulus is the prime
number around which field arithmetic wraps. Akita later locks these
coefficients into short commitments. Its security calculations and matrix
sizes need an explicit bound for every committed value.

Gadget decomposition replaces one large coefficient with several small signed
integers called digits. The original coefficient can be recovered from those
digits. Akita can therefore run its matrix commitments with small, known input
bounds without losing the original value.

This chapter first explains the operation for one ordinary integer. It then
shows the general equation, the exact range that a fixed number of digits can
represent, and the places where Akita uses different decomposition depths.

## Ways to read this chapter

The explanation through [the exact range of a digit
list](#the-exact-range-of-a-digit-list) is the common foundation. After that,
different readers may want different details.

| If you want to | Continue with |
| --- | --- |
| Learn the idea without prior knowledge of proof systems | The chapter in order, starting with the base 4 example |
| Supply values to Akita | [Where Akita uses decomposition](#where-akita-uses-decomposition) and [source width and source shape](#source-width-and-source-shape-are-separate) |
| Change or review the implementation | [How digits are stored](#how-digits-are-stored), [the code path](#path-through-the-code), and [the review checklist](#review-checklist) |
| Audit how the protocol rule reaches the shipped code | [From the protocol rule to the code](#from-the-protocol-rule-to-the-code) |
| Study performance | [From one coefficient to a vector](#from-one-coefficient-to-a-vector) and [how digits are stored](#how-digits-are-stored) |

The later sections do not introduce a different version of decomposition. They
show how the same mathematical rule becomes an input contract, a public
schedule choice, a memory layout, and an optimized routine.

## The values being decomposed

Akita performs most arithmetic in a finite field. A field element is stored as
an integer from $0$ to $q-1$, where $q$ is a prime number called the
modulus. Arithmetic wraps around modulo $q$. For example, $q+3$ and $3$
represent the same field element.

A ring element in Akita is a polynomial whose coefficients are field
elements. Decomposing a ring element means decomposing each of its
coefficients. The same coefficient operation is also used on flat vectors of
field elements.

A polynomial commitment is a short value that fixes a polynomial. After the
prover publishes the commitment, the prover should not be able to claim that
it committed to a different polynomial. A witness is the data used to make a
proof. In Akita, several commitment and opening witnesses contain decomposed
coefficients.

Akita's lattice commitment multiplies witness coefficients by a public matrix.
The security calculation needs an upper bound on those coefficients. A larger
bound can require a matrix with more output rows, which makes the commitment or
proof larger. Decomposition gives the matrix small inputs with an exact bound
while preserving the original coefficient modulo $q$.

## A small example

Choose base $b=4$. Ordinary base 4 digits range from 0 to 3. Akita instead
uses the balanced signed digits

$$
\{-2,-1,0,1\}.
$$

The integer 19 has the following balanced base 4 decomposition:

$$
19 = (-1) + 1\cdot 4 + 1\cdot 4^2.
$$

Akita records the digits from the least significant position to the most
significant position. The digit list for 19 is therefore
$(-1,1,1)$.

Negative values work in the same way. For example,

$$
-6 = (-2) + (-1)\cdot 4 + 0\cdot 4^2,
$$

so its digit list is $(-2,-1,0)$.

These equations also hold for field elements because Akita checks the result
modulo $q$. A field element has many integer representatives that differ by
a multiple of $q$. Before decomposition, Akita chooses a signed
representative that fits the available balanced digit range.

## The general decomposition rule

Akita uses bases that are powers of two. It writes

$$
b = 2^\ell,
$$

where $\ell$ is called the basis exponent in the code. The balanced digit
set is

$$
\mathcal{D}_b = \{-b/2, \ldots, b/2-1\}.
$$

The number of digits is written as $\delta$. A coefficient $x$ is
decomposed into digits
$d_0,\ldots,d_{\delta-1}\in\mathcal{D}_b$ that satisfy

$$
x = \sum_{j=0}^{\delta-1} d_j b^j \pmod q.
$$

The symbols have the following meanings.

| Symbol | Meaning |
| --- | --- |
| $q$ | The prime field modulus |
| $b$ | The decomposition base |
| $\ell$ | The base 2 logarithm of $b$, stored as `log_basis` |
| $\delta$ | The number of digits, also called the decomposition depth |
| $d_j$ | The signed digit at position $j$ |

The digit at position zero has weight 1. The next digit has weight $b$.
Each later position multiplies the weight by another factor of $b$.

## From one coefficient to a vector

The gadget matrix records the powers of $b$ used to rebuild every
coefficient. For $n$ coefficients, it is written as

$$
\mathbf{G}_{b,n}
= I_n \otimes (1,b,\ldots,b^{\delta-1}).
$$

Here $I_n$ is the square identity matrix with $n$ rows and columns. It has
ones on its diagonal and zeros elsewhere. The tensor product symbol
$\otimes$ means that each coefficient gets its own copy of the power vector
$(1,b,\ldots,b^{\delta-1})$. Multiplying this matrix by the digits rebuilds
the original coefficients modulo $q$.

The matrix formula groups all digits of one coefficient together. The flat
decomposition function in the code stores the same data in digit major order.
It stores the first digit of every coefficient, then the second digit of every
coefficient, and continues in that order.

For the two coefficients $(19,-6)$ from the example above, the digit planes
are

$$
(-1,-2),\qquad (1,-1),\qquad (1,0).
$$

The flat output is therefore
$(-1,-2,1,-1,1,0)$. This order lets the prover process one digit plane across
many coefficients at once.

## The exact range of a digit list

A fixed number of digits cannot represent every integer. The largest possible
absolute value depends on the base and the number of digits.

Add the weights of all $\delta$ digit positions and call the result
$S_\delta$:

$$
S_\delta = 1+b+\cdots+b^{\delta-1}.
$$

The most negative digit at every position is $-b/2$. The largest positive
digit is only $b/2-1$. Therefore, $\delta$ digits represent every integer in

$$
[-M_\delta,T_\delta],
$$

where

$$
M_\delta = \frac{b}{2}S_\delta
\qquad\text{and}\qquad
T_\delta = \left(\frac{b}{2}-1\right)S_\delta.
$$

The interval extends farther in the negative direction. This is not a rounding
error. The digit set contains $-b/2$, but it does not contain $b/2$.

For base 4 with three digits, $S_3=1+4+16=21$. The negative endpoint is
$-2\cdot21=-42$. The positive endpoint is $1\cdot21=21$. The exact range
is therefore $[-42,21]$.

The code provides two kinds of range helper. The exact helper returns `None`
for an endpoint when the true mathematical value is larger than any `u128`.
In that result, `None` means that no `u128` coefficient can exceed the
endpoint. A separate helper returns a finite value when its calculation would
overflow. That finite value is safe for estimates, but code that accepts or
rejects a coefficient uses the exact helper.

## How Akita chooses the number of digits

The decomposition depth must be large enough for the value being decomposed.
Akita uses two related rules because a full field element and a source with a
smaller declared bound have different ranges.

### Full field elements

Let $F$ be the number of bits needed for field elements. A full field
decomposition uses

$$
\delta = \left\lceil\frac{F}{\ell}\right\rceil.
$$

The ceiling brackets mean that the fraction is rounded up to the next integer.
For example, a field with 128 bits and basis exponent $\ell=3$ uses 43
digits. Forty two digits cover only 126 bit positions, so one more digit is
required.

There is one edge case when $\delta\ell=F$. A symmetric signed interval can
appear to require an extra digit even though every field value already has a
valid representative. Akita uses the smaller of the exact positive endpoint
$T_\delta$ and $q/2$ as its centering threshold in this case. Residues above
that threshold are represented as negative integers by subtracting $q$. This
lets the full field path keep the rounded up formula above.

### Sources with a smaller bound

Some committed sources are known to be much smaller than a general field
element. Akita records their allowed signed width as $B$. A width of $B$
means the interval

$$
[-2^{B-1},2^{B-1}-1].
$$

Akita first tries $\lceil B/\ell\rceil$ digits. It then checks the exact
positive endpoint $T_\delta$. If that endpoint is too small, Akita adds one
digit.

A concrete case shows why this check is needed. Let $B=6$ and $\ell=3$,
so the base is 8. The declared signed interval is $[-32,31]$. Two balanced
base 8 digits represent only $[-36,27]$. The negative side is wide enough,
but the positive side stops at 27. Akita therefore uses three digits.

A nonnegative value that may be as large as $2^m$ needs signed width
$B=m+1$. This extra bit records the sign. A source that may contain any
`u64` value therefore declares width 65, not 64.

Some parts of the protocol have an exact bound on the largest absolute
coefficient instead of a signed bit width. This bound is often written as an
infinity norm. Akita tries increasing depths and selects the first exact digit
range that contains both the positive and negative bound.

## Where Akita uses decomposition

A proof starts with the values supplied by the application. Akita calls these
the root source. A fold is one proof step that reduces the current witness to a
smaller witness for the next step. An opening proves that a committed
polynomial has a claimed value at a chosen point.

These values need different decomposition depths.

| Value | Why it is decomposed | Bound used for its depth |
| --- | --- | --- |
| Root committed source | It is the application data entering the first commitment | `log_commit_bound` |
| Recursive committed source | It is the smaller witness produced by an earlier fold | The source bound stored by the schedule |
| Opening value | It is used to prove a claimed polynomial evaluation | `log_open_bound` or the full field width |
| Setup prefix | It contains public setup field elements used by a later fold | The full field width |
| Folded response | It is the response produced during one fold | The exact response bound stored by the schedule |

The schedule is a public plan chosen before proving. It records the bases,
digit counts, matrix sizes, and other values that the prover and verifier must
use at each fold. The verifier reads the same digit counts from the resolved
schedule. It does not trust a digit count supplied by the proof.

An application using a standard preset does not call the decomposition routine
or choose a digit count directly. It declares its source representation and
uses a preset whose planner resolves these choices into the schedule. A custom
configuration must keep the source class, source bound, basis, and resulting
depth consistent. Akita rejects the source at commitment time if that contract
does not hold.

## Source width and source shape are separate

Akita records both a numeric bound and a source class. The bound says how large
each coefficient may be. The class says what pattern the coefficients must
follow.

`BalancedSignedDigit` is the general bounded signed source class.
`UnitOneHot` is more specific. Its coefficients are 0 or 1, and each configured
chunk contains at most one coefficient equal to 1. A dense vector of zeros and
ones does not satisfy this class merely because each coefficient is small.

For this reason, `log_commit_bound == 1` does not select `UnitOneHot`. The class
must say `UnitOneHot` explicitly. For a bounded `BalancedSignedDigit` source,
the accepted coefficient interval is the intersection of two ranges:

1. The interval declared by `log_commit_bound`.
2. The interval that the scheduled number of digits can represent.

The code that creates the commitment checks both ranges. It rejects an out of
range coefficient instead of silently dropping high digits. The `UnitOneHot`
path separately checks that every value is 0 or 1 and that each chunk has at
most one value equal to 1.

## How digits are stored

The largest digit depends on $\ell$, so Akita chooses a signed integer type
that can hold every digit. In Rust, `i8` is a signed integer with 8 bits and
`i16` is a signed integer with 16 bits.

| Basis exponent $\ell$ | Balanced digit interval | Rust storage |
| --- | --- | --- |
| 1 to 8 | $[-2^{\ell-1},2^{\ell-1}-1]$ | `i8` |
| 9 to 16 | $[-2^{\ell-1},2^{\ell-1}-1]$ | `i16` |

The basis exponent is not the base itself. For example, $\ell=10$ means
base $2^{10}=1024$, with digits from $-512$ to 511. It does not mean base
10.

Production schedule validation keeps opening bases in the `i8` range and
inner commitment bases in the `i16` range. The current schedule search tries
opening exponents from 3 to 6. It tries inner exponents up to 10 for fields
with 32 bits and up to 11 for fields with 64 or 128 bits.

The ring API can return digits as field elements, `i8` values, or `i16` values.
On supported x86 processors, the packed `i8` function can use AVX2. AVX2 is a
set of CPU instructions that performs the same operation on several integers
at once. The fast path processes field values in their standard 32 bit integer
representation, eight values at a time. The ordinary fallback processes one
value at a time and produces the same digit major output.

The `i8` NTT commitment path next represents each signed digit modulo several
smaller primes. It stores those residues in Montgomery form, which is an
internal representation used for fast multiplication. Converting every digit
inside the matrix multiplication loop would repeat the same work many times.
Akita instead builds a lookup table once for each matrix multiplication. The
table covers the active balanced range and returns the prepared residue for a
digit and a small prime. The fixed `i8` table can cover every value from -128
to 127, while each operation initializes only the part needed by its basis.

## Path through the code

The main code path is:

1. `DecompositionParams` in `crates/akita-types/src/config.rs` records
   `log_basis`, `log_commit_bound`, and the optional `log_open_bound`.
2. `crates/akita-types/src/sis/decomposition_digits.rs` computes the exact
   digit count for each protocol role.
3. The generated schedule stores the selected counts in
   `CommittedGroupParams`.
4. `crates/akita-algebra/src/ring/cyclotomic/decomposition.rs` performs the
   coefficient decomposition and rebuilding operation.
5. `crates/akita-prover/src/kernels/linear/decompose.rs` uses the packed `i8`
   kernel. Wider inner decompositions use
   `crates/akita-prover/src/compute/cpu/exact_i16.rs`.
6. `crates/akita-types/src/sis/committed_source.rs` checks the accepted source
   range and source class.

Supporting limits live in these files:

- `crates/akita-types/src/signed_digit.rs` selects `i8` or `i16` storage.
- `crates/akita-algebra/src/ring/crt_ntt_repr/lut.rs` prepares signed `i8`
  digits for the CRT and NTT commitment path.
- `crates/akita-schedules/src/runtime.rs` validates schedule basis ranges.
- `crates/akita-config/src/proof_optimized.rs` defines the production search
  ranges.

## From the protocol rule to the code

The equations above define decomposition, but the implementation must make
more choices. It decides how many digits each protocol value gets. It chooses
the signed integer type and memory order. It also decides which inputs are
allowed. The schedule makes these choices public so that the prover and
verifier use the same values.

| Property to trace | Explanation in this chapter | Current implementation |
| --- | --- | --- |
| A coefficient is rebuilt from balanced powers of the base | [The general decomposition rule](#the-general-decomposition-rule) and [the gadget matrix](#from-one-coefficient-to-a-vector) | `decomposition.rs` extracts the digits, while `digit_math.rs` produces the public powers used to rebuild them |
| The asymmetric range and full field centering need no extra digit | [The exact range](#the-exact-range-of-a-digit-list) and [full field elements](#full-field-elements) | `decompose_centering_threshold` applies the exact threshold, including the 128 bit overflow boundary |
| Every committed digit is in the balanced set | [The balanced digit set](#the-general-decomposition-rule) | `proof/stage1.rs` defines the range polynomial and `akita-verifier/src/stages/stage1.rs` verifies its sumcheck proof |
| Different protocol values can use different bases and depths | [Where Akita uses decomposition](#where-akita-uses-decomposition) | `decomposition_digits.rs` separates root sources, recursive sources, setup prefixes, openings, and folded responses |
| Prover and verifier use the same choice | [The public schedule](#where-akita-uses-decomposition) | The resolved schedule stores each role specific basis and depth in `CommittedGroupParams` |
| A committed source satisfies the assumptions used to size it | [Source width and source shape](#source-width-and-source-shape-are-separate) | `committed_source.rs` defines the contract and `akita-prover/src/api/commitment.rs` enforces its numeric and structural parts |
| Compact extraction and field conversion preserve the same digits | [How digits are stored](#how-digits-are-stored) | `signed_digit.rs` selects `i8` or `i16`, `decomposition.rs` extracts the digits, and `crt_ntt_repr/lut.rs` prepares `i8` digits for multiplication |
| AVX2 extraction agrees with scalar extraction | [The fast path and fallback](#how-digits-are-stored) | `decomposition/x86.rs` implements the fast path and the centering boundary tests compare it with the scalar path |

This table separates the protocol contract from its implementation. The
equations and ranges state what must hold. The schedule and source contract
state which values Akita accepts. The algebra, prover, and verifier code show
how Akita computes and checks those values.

## Review checklist

A code review or security audit can follow the mechanism in this order.

1. Check that extraction produces digits in
   $[-2^{\ell-1},2^{\ell-1}-1]$ and that their weighted sum rebuilds the
   original coefficient modulo $q$.
2. Check the independent verifier boundary. Stage 1 must prove that every
   committed digit is a member of the same balanced set. Correct prover
   extraction alone is not enough because a proof may contain malformed data.
3. Check that the depth selector covers the correct range for each role. Full
   field values and smaller signed sources use different centering rules.
4. Check that source admission applies both parts of the contract. The numeric
   interval and the declared source class are independent.
5. Check that the resolved schedule, proof layout, prover, and verifier all use
   the same role specific depth and basis.
6. Check that the flat, ring, `i8`, `i16`, scalar, and AVX2 paths agree on digit
   order and centering at their boundaries.
7. Check any performance change against the scalar implementation before
   relying on benchmark results. A faster layout is only valid if it preserves
   the same digits and reconstruction.

The focused regression tests are in
`crates/akita-types/src/sis/decomposition_digits.rs`,
`crates/akita-types/src/sis/committed_source.rs`, and
`crates/akita-algebra/src/ring/cyclotomic/tests.rs`. The Stage 1 range topology
tests live in `crates/akita-types/src/proof/stage1.rs`. Together, these tests
cover the asymmetric field boundary, the exact positive reach, the
intersection of declared and representable source ranges, supported range
proof bases, digit major layout, the wider `i16` bases, and agreement between
scalar and AVX2 decomposition. These tests are useful evidence, but they do not
replace checking that every protocol caller uses the right public parameters.

## What to remember

Gadget decomposition is an exact change of representation. It does not round a
coefficient or change the field element. It replaces one coefficient with
small signed digits whose weighted sum gives the original value modulo $q$.

The base controls the size of each digit. The depth controls how many digits
are available. Akita chooses both values as part of the public schedule, and
the prover must reject any source that does not fit the resulting contract.
