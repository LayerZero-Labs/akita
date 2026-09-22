# Binary host field-switch arithmetic

Status: active
Book-chapter: book/src/foundations/field-arithmetic.md
Tracking: https://github.com/LayerZero-Labs/akita/issues/45

## Scope and representations

The arithmetic implementation and scoped acceptance checks are complete on the
topic branch. This record stays active through review and landing, then follows
the Book-fold/archive lifecycle.

This slice supplies portable host arithmetic and a field-switch reference for
both F128-witness/F128-challenge and F64-witness/F192-challenge profiles. It
builds on the packed F162 round/fold kernels. It does not admit a production
profile, define a proof encoding, sample transcript challenges or implement a
consumer adapter. All source vectors have exactly `2^point.len()` entries;
callers explicitly pad and bind any shorter logical source.

- F128 is `F2[x]/(x^128+x^7+x^2+x+1)`. Two low-degree-first 64-bit words encode
  polynomial coefficients. This is a polynomial-coordinate convention, not
  the reflected bit/byte convention of a GHASH network encoding.
- K=F64 is `F2[x]/(x^64+x^4+x^3+x+1)`; a `u64` stores its coefficients.
- F192 is `K[y]/(y^3+y+1)`. Words are `[c0,c1,c2]`. Coordinate `64*t+b`
  means `x^b*y^t`. K embeds in its constant coefficient.
- F162 retains `F2[X]/(X^162+X^81+1)`. The source map `phi` copies the 64 or
  128 source bits into low polynomial coordinates. It is injective and
  F2-linear, not a field homomorphism. No embedding of F192 into F162 is used.

The cubic is irreducible over F64: its roots have degree three over F2 and
`gcd(3,64)=1`. Polynomial irreducibility of the two base moduli is checked with
independent Frobenius/gcd tests. Host types do not implement the odd-prime
`jolt_field::Field` contract.

## Reduction identities

For host field E with binary basis `beta_k`, expand host equality weights:

```text
M[k,j] = bit_k(eq_E(r,j))
p_k = XOR_{j: M[k,j]=1} w_j
v = sum_k beta_k * embed_K_to_E(p_k)
```

The table index's least-significant bit corresponds to the first point
coordinate. Source values are F128 itself or F64 respectively. The partial
vector has 128 entries for F128 or 192 entries for F192. The latter is padded
to 256 with 64 canonical zero rows. Parsing partials rejects any other length
or nonzero padded row.

After binding the partials, a caller chooses 7 or 8 F162 batching coordinates
`s` and obtains `lambda_k=eq_F162(s,k)`. The batched claim and coefficient table
are:

```text
h = sum_k lambda_k * phi(p_k)
c_j = sum_k lambda_k * M[k,j]
h = sum_j phi(w_j) * c_j
```

The padded rows contribute zero. A product sumcheck over F162 can reduce the
last inner product using the existing packed kernels. At its terminal point
`z`, the verifier evaluates the multilinear extension of `c` with a tensor
calculation independent of the witness length:

```text
T = product_i (r_i tensor 1 + 1 tensor (1+z_i)) in E tensor_F2 F162
c(z) = sum_k lambda_k * T_k
```

Indeed `(1+r_i)(1+z_i)+r_i*z_i=1+r_i+z_i` in characteristic two. Tensor
multiplication uses E's binary basis matrix for multiplication by each `r_i`,
and F162 scaling for `1+z_i`. Storage is bounded by the fixed host dimension;
work is O(log N * dim(E)^2) XORs and O(log N * dim(E)) F162 multiplications.
The tensor algebra need not be a field and no tensor inverses are used.

## Protocol boundary and remaining obligations

These functions check algebraic inputs, not proofs. A production adapter must
bind profile/basis, source commitment, logical/padded layout and owner before
host challenges; authenticate the host evaluation; bind all unpadded partials
before batching challenges; then bind round messages before each challenge and
open the terminal source evaluation against the original owner. Padding is
canonical, not additional prover freedom. Reordering basis rows or variables
changes the claimed relation.

For fixed incorrect partials, injectivity of `phi` makes the batched discrepancy
a nonzero multilinear polynomial: random row batching loses at most 7/|F162|
or 8/|F162|. This is conditional on the original source/partials being fixed;
it does not supply the lattice relaxed-witness extraction argument. A degree-two
sumcheck adds its own per-round error; host reductions and Fiat–Shamir losses
must be composed separately in the eventual schedule.

The switch must authenticate the mod-two projection of the decoded relaxed
witness extracted from the Akita opening. Consumer semantics must hold for that
projection, not merely for the honest constructor vector. Equality among source
occurrences uses characteristic-zero cross multiplication and the certified
source-comparison bound `eta < P`, followed by projection modulo two; it does
not use the ordinary unit-mod-P extraction lemma. The F64/F192 generalization
still needs the full reviewed projection argument, completeness/retry accounting
and composition with that no-wrap policy. Arithmetic agreement alone does not
close those obligations. A terminal coefficient of zero does not authorize the
protocol to omit the source opening against its original commitment. Public within-word weighted claims in
Flock/binius are a subsequent contract, not silently replaced by point openings.

## Acceptance

- [x] Independent host multiplication/basis checks, including full-width inputs.
- [x] Dense host MLE and reconstructed partials agree for both profiles.
- [x] Batched partials equal the F162 source/coefficient inner product.
- [x] Structured terminal evaluation agrees with dense F162 evaluation.
- [x] Complete packed round/fold sequences reach the same terminal equation.
- [x] Malformed lengths, row padding, wrong basis/order and modified partials
  are exercised; no claim of transcript authentication by arithmetic helpers.
- [x] Benchmarks separate partial generation, batching, all rounds and verifier
  calculation, including reusable scratch and both host profiles.
- [x] Independent correctness and maintainability review completed.

## External coordinate references

The host definitions were independently implemented from the algebraic
contracts pinned at leanVM `48a904208d682848dac0e18ef8b01ebfc40df9ad`
(`crates/primitives/src/field/gf2_64.rs` and `gf2_64x3.rs`) and the GHASH
polynomial-coordinate definition in LaBinius
`96b57f724fc15c8e40495015c01dc3e6d8dfe595`
(`crates/pcs/src/fields/scalar.rs`). No reference implementation code is copied.
The latter reference's cross-field partials use the transposed witness-coordinate
orientation; this slice deliberately implements the challenge-coordinate
contract above for both profiles.
