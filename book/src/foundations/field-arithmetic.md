# Field arithmetic

Akita spends most of its prover time adding and multiplying field elements.
The representation of those elements therefore affects nearly every higher
level operation: multilinear evaluation, sum-check, gadget decomposition,
and ring multiplication. The arithmetic library also supplies a smooth FFT
and Reed--Solomon extension utility for tests and benchmarks.

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
also has a smooth multiplicative subgroup used by the arithmetic library's
Reed--Solomon extension utility. [NTT, CRT, and fast ring
arithmetic](./ntt-crt.md) describes that separate transform.

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

## Binary scalar arithmetic

`akita_algebra::binary::BinaryField162` provides arithmetic over
`F_2[X]/(X^162 + X^81 + 1)`. An element has 162 bit coefficients; addition is
XOR, and multiplication reduces the polynomial product using
`X^162 = X^81 + 1`. The modulus is irreducible because 2 has multiplicative
order 162 modulo 243.

This type is an arithmetic foundation only. It is separate from the prime-field
extensions used by current commitment configurations and does not enable a
binary opening protocol or a new schedule.

The canonical encoding contains 21 little-endian bytes, with the top six bits
zero. Decoding rejects other lengths and nonzero unused bits. Multiplication
uses six carryless word products and two reduction folds. Runtime dispatch
selects PMULL on supported ARM CPUs or PCLMUL on supported x86-64 CPUs; a
portable kernel remains available on other CPUs. Hardware multiplication fuses
the polynomial product and reduction so the six-word intermediate does not
cross a function boundary. Hardware squaring needs three carryless products;
the portable kernel interleaves zero bits instead. Inversion uses an addition
chain with nine multiplications and returns `None` for zero.

`BinaryField162::dot_product` XOR-accumulates unreduced pairwise products and
reduces once. It returns `None` when the input lengths differ. The x86-64
backend processes two pairs per VPCLMUL instruction when AVX2 and VPCLMULQDQ
are available; all backends produce the same canonical field element.

The implementation and its independent bit-convolution tests live in
`crates/akita-algebra/src/binary.rs` and `crates/akita-algebra/src/binary/`.
The `binary162` Criterion target covers scalar multiplication, squaring,
inversion, deferred-reduction dot products, and multiply-then-sum comparisons.

### Keep binary tables packed across rounds

A product sum-check repeatedly computes a polynomial from two tables, then
folds each table at a challenge. Converting every table before every operation
can consume much of the arithmetic speedup. `PackedBinary162` instead stores
matching words from many field elements together in three arrays. The same
storage survives all rounds, and `refill` reuses its capacity for another input.
This layout changes computation storage only; it does not change the scalar
field or its canonical 21-byte encoding.

The kernel pairs adjacent entries. For a pair `(a0, a1)`, define
`da = a0 + a1`, where addition is XOR. The folded value at challenge `r` is
`a0 + r * da`. For paired tables `a` and `b`, the round polynomial is

```text
g(X) = sum_pairs (a0 + X * da) * (b0 + X * db).
```

`round_product` takes the current claim `h = g(0) + g(1)` and returns the constant,
linear, and quadratic coefficients. In characteristic two, `h = c1 + c2`,
so only `c0` and `c2` require table products; `c1 = h + c2`. The claim is a trusted
prover hint, not a value authenticated by this kernel. This avoids a redundant
product per pair, division by two, and integer-point interpolation.
The caller samples the challenge after
binding the message, then calls `fold_in_place` on both tables. These arithmetic
kernels themselves do not run a transcript or establish a valid opening.

The first eliminated variable is the least-significant bit of the table index.
An unmatched final entry is paired with zero. Folding reduces the live length
to its ceiling half without allocating; empty and singleton tables are unchanged.
A round message requires equal lengths of at least two, otherwise it returns
`None`. The caller is responsible for binding logical lengths and padding to
the statement. Retained capacity is private scratch, not extra live elements.

The `binary162_packed` benchmark group measures messages and full fold sequences,
including conversion into reused storage and allocation as separate cases.
Its array-of-structures (AoS) baseline gathers pairs into reused scratch and
uses the existing accelerated `BinaryField162::dot_product` API with deferred
reduction. Prepacked all-round timings exclude the initial copy; conversion-plus-
all-round timings include refilling both buffers. These compare computation
kernels, not complete binary proof generation.

### Switch host evaluation claims to F162

A binary consumer can keep its own witness and challenge fields while reducing
an evaluation claim to F162 arithmetic. The field-switch module supports two
specific coordinate contracts:

| Source words | Host challenge field | Host basis | Live / padded rows |
|---|---|---|---|
| 128-bit polynomial coordinates | `BinaryField128` | `1,x,...,x^127` | 128 / 128 |
| 64-bit polynomial coordinates | `BinaryField192` | `x^b y^t`, index `64*t+b` | 192 / 256 |

`BinaryField128` uses `x^128+x^7+x^2+x+1`. The second profile uses the base
field `K = F2[x]/(x^64+x^4+x^3+x+1)` and the cubic extension
`K[y]/(y^3+y+1)`. Its three words hold the coefficients of `1,y,y^2`.
In both profiles, bit zero means the constant coefficient. The F128 polynomial
matches the GHASH polynomial, but these coordinates are not a reflected GHASH
network encoding. An adapter must convert its actual source representation.
The host arithmetic is currently portable; F162 keeps its accelerated kernels.

To see why switching is possible, write each host equality weight in its binary
basis. Each basis coordinate is either zero or one, so it selects a subset of
source words. XOR those words to form one **partial evaluation** per host basis
coordinate. For example, a row with bits `[1,0,1,0]` has partial `w0 + w2`.
For host point `r`, source index `j`, and host basis element `beta_k`, this is

```text
eq_host(r,j) = sum_k beta_k M[k,j], with M[k,j] in F2
p_k = XOR of w_j for which M[k,j] = 1
host evaluation = sum_k beta_k * embed_source_in_host(p_k).
```

The source map `phi` into F162 simply copies the source's 64 or 128 bits into
low polynomial coordinates. This preserves XOR and is injective. It does not
preserve field multiplication, and the construction needs no F192-to-F162
field embedding. After the partials are fixed, seven or eight F162 batching
coordinates define row weights `lambda_k`. Binary linearity gives

```text
sum_k lambda_k phi(p_k) = sum_j phi(w_j) c_j
c_j = sum_k lambda_k M[k,j].
```

This is an F162 inner product suitable for the packed product-sumcheck kernels.
F192's final 64 rows are fixed zeros; they are not additional prover choices.
The partials are indexed by **host challenge-field coordinates**, rather than
source-bit coordinates. Reversing that orientation changes their meaning even
when both dimensions happen to be 128.

The final coefficient evaluation need not scan the source. Keep a vector of
128 or 192 F162 coefficients representing an element of the tensor algebra
`host tensor_F2 F162`. Start with one and, for each corresponding host/source
point coordinate `r_i,z_i`, multiply by `r_i tensor 1 + 1 tensor (1+z_i)`.
This identity follows by expanding the two Boolean equality factors in
characteristic two. Contract the resulting coefficients with `lambda` to obtain
`c(z)`. Host multiplication acts by a binary matrix, so its application needs
XORs; only scaling by `1+z_i` needs F162 products. Scratch is independent of the
source table length, and work grows with the number of point coordinates.
No division or tensor-field assumption is needed, including when `c(z)=0`.

The arithmetic path in `binary::field_switch` is:

1. `partial_evaluations` checks the exact source size and fills reusable host
   equality scratch while constructing `SwitchPartials`.
2. `SwitchPartials::reconstruct` gives the host evaluation to compare with the
   authenticated host claim. `try_from_values` checks row count and zero padding
   for partials supplied by a caller.
3. `SwitchPartials::batch` gives the F162 claim. `batched_weights` transforms
   equality scratch for that same host point into F162 coefficients, using a
   small lookup table and reusable output storage.
4. `PackedBinary162` computes the round messages and folds. At the terminal
   point, `transparent_weight` independently evaluates the public coefficient
   factor without enumerating the source table.

Point coordinate zero controls the least-significant table-index bit. The
source must already contain exactly `2^point.len()` entries; callers own logical
lengths and explicit source padding. These functions do not serialize a proof,
run a transcript, authenticate a commitment, or admit a production profile.
The eventual adapter must bind source ownership/layout before host challenges,
partials before batching, and each message before its folding challenge. It
must check the terminal product and open its source factor against the original
commitment. The relaxed-source extraction and combined error accounting remain
protocol obligations.

The independent tests check polynomial arithmetic, dense partial matrices,
host reconstruction, batching, structured coefficient evaluation and complete
packed folds for both profiles. The `binary_field_switch` Criterion groups in
the existing `binary162` target separate partial generation, coefficient
batching, prepared rounds, the combined arithmetic path, host reconstruction
and structured verifier work. They do not measure complete PCS proofs.
