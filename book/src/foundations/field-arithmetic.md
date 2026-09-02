# Field arithmetic

Akita spends most of its prover time adding and multiplying field elements.
The representation of those elements therefore affects nearly every higher
level operation: multilinear evaluation, sum-check, gadget decomposition,
ring multiplication, and Reed--Solomon encoding.

The production configurations use three pseudo-Mersenne prime fields. A
pseudo-Mersenne prime has the form

\[
q = 2^k-c
\]

for a small positive integer \(c\). This shape lets the implementation reduce
wide products by multiplication with \(c\), rather than by a general division
or Montgomery reduction.

## The production field families

The base field stores the committed polynomial. The challenge field contains
the random evaluation points used by the protocol.

| Family | Base-field modulus | Challenge field |
| --- | --- | --- |
| `fp32` | \(2^{32}-99\) | degree-4 extension |
| `fp64` | \(2^{64}-59\) | degree-2 extension |
| `fp128` | \(2^{128}-2^{32}+22537\) | base field |

All three challenge fields therefore have about 128 bits. The `fp128` modulus
also has a smooth multiplicative subgroup used by Akita's Reed--Solomon
encoder. [NTT, CRT, and fast ring arithmetic](./ntt-crt.md) describes that
separate transform.

The Solinas module registers additional pseudo-Mersenne fields for auxiliary
algorithms, tests, and benchmarks. A registered arithmetic type is not by
itself a production Akita configuration. The selected `CommitmentConfig` and
its generated schedule determine the fields used by a proof.

## Canonical residues

Akita stores a base-field element as its canonical integer in \([0,q)\). It
does not use Montgomery form. For the 32- and 64-bit fields this is one native
word. The 128-bit field uses two little-endian 64-bit limbs.

This representation makes conversion to centered integers and power-of-two
digits direct. If \(b=2^L\), the low unsigned digit is simply

\[
u=x\mathbin{\&}(b-1).
\]

The balanced digit is \(u\) when \(u<b/2\), and \(u-b\) otherwise. The
[gadget-decomposition chapter](./gadget-decomposition.md) develops the full
carry rule and the asymmetric final digit range.

## Addition and subtraction

There are two machine-level cases.

When \(k\) is smaller than the storage-word width, two canonical residues can
be added without overflowing that word. The implementation conditionally
subtracts \(q\) to return to \([0,q)\). The same one-word structure supports
subtraction with a conditional correction after a borrow.

The production fields instead fill their storage width: \(k=32\), \(64\), or
\(128\). An overflowing sum has lost one copy of \(2^k\). Because

\[
2^k \equiv c \pmod q,
\]

the implementation folds the carry back as \(c\), then performs the remaining
canonical correction. Subtraction similarly uses the borrow flag to restore
the modulus. The 128-bit implementation has portable two-limb routines and
architecture-specific routines behind its `asm` feature; both preserve the
same canonical representation.

## Multiplication by two Solinas folds

Write a nonnegative integer as

\[
x=x_{\mathrm{lo}}+2^k x_{\mathrm{hi}},
\qquad 0\le x_{\mathrm{lo}}<2^k.
\]

One Solinas fold replaces it with

\[
\operatorname{fold}(x)=x_{\mathrm{lo}}+c x_{\mathrm{hi}}.
\]

The two values are congruent modulo \(q\). A product of canonical residues is
smaller than \(2^{2k}\), so the production multiplication path applies this
fold twice. The field types enforce

\[
c(c+1)<q,
\]

which bounds the second folded value tightly enough for one final canonical
correction.

For the 128-bit field, the unreduced product occupies four 64-bit limbs. The
first fold combines the high two limbs with the low two through \(c\); the
second fold handles the remaining overflow and canonicalizes. Because its
Solinas constant is below \(2^{32}\), every multiplication by \(c\) still fits
the two-limb folding design.

The 128-bit field also implements a fused operation

\[
a b+d \pmod q.
\]

It adds the canonical value \(d\) to the wide product before reduction, then
runs one Solinas reduction on the combined value. Polynomial evaluation and
other multiply-accumulate loops can therefore avoid reducing \(ab\) and then
reducing the following addition separately.

## Deferred reduction

Reducing after every product is unnecessary when an entire inner product has
a known integer bound. The field library exposes unreduced product types and
accumulators for this purpose.

A hot loop widens each product, adds it to an accumulator whose lanes cannot
overflow for the admitted number of terms, and reduces once at the end. Some
paths use separate positive and negative accumulators so that signed small
coefficients do not require a field reduction at every step. Extension-field
accumulators apply the same idea coordinate by coordinate.

The admissible number of terms is an arithmetic contract, not a tuning hint.
Commitment kernels use `F::MAX_COMMIT_ACCUMULATIONS`, while CRT matvecs use the
explicit reconstruction bound in the [NTT and CRT chapter](./ntt-crt.md).
When a row is longer, the implementation ends the current accumulation chunk,
reduces it, and continues. It never relies on a wide accumulator being
effectively unbounded.

### Product accumulators

A product accumulator stores each base-$2^64$ limb sum in its own `u128` slot.
The slots use wrapping addition and subtraction. Reduction later reads each
slot as an unsigned integer and propagates carries between limbs. The result is
exact only while the final mathematical value of every slot remains below
`2^128`; this headroom is established separately for each concrete product
formula.

| Product | Accumulator | Proven term headroom |
| --- | --- | ---: |
| fp32 by fp32 | 2 `u128` slots | `2^64` |
| fp64 by fp64 | 2 `u128` slots | `2^64` |
| fp128 by fp128 | 4 `u128` slots | `2^64 - 1` |
| fp128 by `u64` | 3 `u128` slots | `2^64 - 1` |
| fp32 degree-4 extension product | 4 `u128` slots | `2^61` |
| fp64 degree-2 extension product | 4 `u128` slots | more than `2^62` |

The extension accumulators fuse reduction by the extension polynomial into the
per-coordinate formulas. Subtractive coordinates receive a fixed multiple of
the base-field modulus squared before accumulation, preventing unsigned
underflow without changing their residue.

### Small signed linear accumulators

A different representation serves matrix products with small signed
coefficients. It splits each canonical field element into 16-bit pieces stored
in signed `i32` lanes: 2 lanes for fp32, 4 for fp64, and 8 for fp128. Scaling a
field value by a small signed digit scales each lane directly. Fresh lanes have
magnitude below `2^16`, so at least

$$
\left\lfloor\frac{2^{31}-1}{2^{16}-1}\right\rfloor=32768
$$

same-sign additions fit. More generally, `k` terms scaled by magnitude `s` are
admitted only when

$$
k|s|(2^{16}-1)<2^{31}.
$$

Reduction propagates signed carries through the 16-bit lanes and then applies
the field's Solinas reduction. These lanes use ordinary non-wrapping arithmetic;
debug builds also trap an accidental lane overflow.

## Uniform field sampling

`Field::random` uses exact rejection sampling. For a modulus with \(k\)
significant bits, each attempt reads exactly \(\lceil k/8\rceil\) little-endian
bytes, masks unused high bits, and accepts the candidate only when it is below
the modulus.

This rule matters for reproducible setup generation. Reducing a fixed-width
random integer modulo \(q\) would introduce a small bias. Rejection sampling
does not: every field element has the same probability. It also gives a
canonical byte-consumption rule for every page of the public setup stream.

## Extension arithmetic

An extension element is stored as its canonical base-field coordinates. The
degree-2 implementation uses Karatsuba multiplication and a specialized
squaring formula. The degree-4 and degree-8 implementations use the same
cyclotomic subfield basis consumed by Akita's trace maps. They do not maintain
a second wire or storage basis.

The quartic inverse is internally computed through its quadratic subfield,
reducing inversion to one base-field inverse. This is an arithmetic algorithm,
not another representation of the element. See [Cyclotomic rings and
extension fields](./rings-and-fields.md#the-concrete-extension-bases) for the
basis and multiplication relations.

## Implementation and review map

The field code is supplied by Akita's pinned `jolt-field` dependency.

| Property | Primary source |
| --- | --- |
| Registered pseudo-Mersenne types and exact rejection sampling | [`solinas/mod.rs`](https://github.com/a16z/jolt/blob/72dc6451628d8b1dd794147a1f1cc40be0d77963/crates/jolt-field/src/solinas/mod.rs) |
| 32- and 64-bit word arithmetic | [`solinas/word.rs`](https://github.com/a16z/jolt/blob/72dc6451628d8b1dd794147a1f1cc40be0d77963/crates/jolt-field/src/solinas/word.rs) |
| 128-bit two-limb arithmetic and fused multiply-add | [`solinas/fp128.rs`](https://github.com/a16z/jolt/blob/72dc6451628d8b1dd794147a1f1cc40be0d77963/crates/jolt-field/src/solinas/fp128.rs) |
| Unreduced products and accumulators | [`solinas/unreduced.rs`](https://github.com/a16z/jolt/blob/72dc6451628d8b1dd794147a1f1cc40be0d77963/crates/jolt-field/src/solinas/unreduced.rs) |
| Extension arithmetic | [`solinas/ext.rs`](https://github.com/a16z/jolt/blob/72dc6451628d8b1dd794147a1f1cc40be0d77963/crates/jolt-field/src/solinas/ext.rs) |
| Production field selection | `crates/akita-config/src/proof_optimized/` |

A field change must preserve canonical encoding, exact sampling, centered
conversion, extension-coordinate order, and the bounds assumed by every
deferred accumulator. Differential tests compare specialized arithmetic with
the ordinary field operations.
