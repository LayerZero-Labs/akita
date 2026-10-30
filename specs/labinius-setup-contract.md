# LaBinius admitted root setup

Status: implemented
Book-chapter: book/src/how/architecture.md
Tracking: https://github.com/LayerZero-Labs/akita/issues/45

The key words "MUST", "MUST NOT", "SHOULD", "SHOULD NOT", and "MAY" in this
document are interpreted as described in BCP 14 (RFC 2119 and RFC 8174) when
they appear in all capitals.

## Derivation and admission

`AdmittedRootSetup::derive::<P>` MUST derive `LabiniusRootShape` from the
selected `LabiniusRootProfile`, `log_num_cells`, `log_fold_width`, and
`lambda_fold`. It MUST use the shape's certified rank `n_A`, ring elements per
column `m`, fold width `C` and response interval, the balanced base-16 digit
range of [root admission](labinius-root-admission.md#fold-response), with the
profile's challenge family, commitment modulus `q` and commitment degree.
The matrix MUST use exact q sampling, so this path MUST NOT apply the legacy
P-to-q derivation-bias gate
([root admission](labinius-root-admission.md#derivation-bias)).
`BinaryClearSetup::new` MUST check the degree, trinomial sign and commitment
modulus against those identities and MUST reject a matrix coefficient at
least `q`.
An admitted root setup MUST NOT be constructed from a caller-supplied matrix.
Its setup, shape, seed, and derivation inputs are immutable public views.
Deriving the setup admits no proof field: that is a separate check on the
shape, made where a proof field enters
([lowered relation](labinius-lowered-root.md#geometry-and-admission)).

`derive_trinomial_matrix::<P, M>` MUST derive exactly `rows * columns * D`
coefficients from `derive_centered_q_matrix`. `P` is a validated field
descriptor, not the source of q candidates; `M` selects the trinomial sign.
For this legacy single-root adapter, the typed label MUST use role
`Commitment`, group zero, level zero, the matrix rank, the ring selected by
`M`, degree `D`, and modulus `Q(q)`. Each matrix entry has its own row and
column indices in the label. The caller's 32 seed bytes pass unchanged to the
typed stream. The PCS composition separately derives that seed
([matrix seed and stream field](#matrix-seed-and-stream-field)).

The typed stream samples centered integers in `[-(q-1)/2, (q-1)/2]`. This
adapter MUST convert them to canonical residues in `[0, q)` for
`BinaryClearSetup`; it MUST NOT change the sampled distribution. Coefficient
`t` of `A[i,j]` is stored at

```text
(i * columns + j) * D + t,    0 <= t < D.
```

This is `TrinomialASetupView::setup_address(i,j,t)` with setup offset zero:
coefficients are contiguous, then columns, then rows. The digest of this view
is specified in [the clear opening](labinius-clear-opening.md#setup-admission).
The matrix element at `i * columns + j` is the row-major element consumed by
`apply_matrix`. For an admitted root, the matrix dimensions are `n_A` by `m`;
`C` counts source columns and does not determine the matrix's column count.
Increasing the column count preserves every existing entry's coefficient
stream, although later rows move in the flat storage layout.

The materialized typed API MUST check the coefficient count and byte count
before allocation and use fallible reservation. It rejects zero dimensions,
a rank or degree not representable as u32, an invalid trinomial degree, an
invalid q, and centered storage over 128 MiB. The legacy adapter MUST validate
`P`'s field descriptor and reserve the canonical-residue output fallibly.
Setup failures MUST return `AkitaError::InvalidSetup`.

## Typed setup streams

`akita_setup` exposes the typed stream API from `akita_params::binary_setup`
behind the default-off `labinius` feature. Calling this API explicitly selects
the binary-source setup derivation. Enabling the feature alone MUST NOT change
ordinary setup streams or select the extension.

`TypedSetupStream::new` MUST initialize SHAKE256 with the following fixed byte
layout. All integers use fixed-width little-endian encoding:

```text
b"akita/binary-source/setup/shake256/v1" || seed:[u8;32]
|| role:u8 || group:u64 || level:u32 || rank:u32 || row:u32 || column:u64
|| ring:u8 || degree:u32 || coefficient_domain:u8 || modulus:u128
```

The role tags are `Commitment = 1`, `ImageCommitment = 2`,
`WitnessCommitment = 3`, and `Projection = 4`. Ring tags are
`PlusTrinomial = 1`, `MinusTrinomial = 2`, and `Negacyclic = 3`.
Coefficient-domain tag 1 names `Q(q)` and tag 2 names
`Prime128Offset275`. The full modulus is included even though the domain tag
already separates q from P. The rank is positive, the row is less than the
rank, and the degree is positive; a trinomial degree is even. Coefficient
position is the sequential position within an entry's XOF stream, including
candidates consumed by rejection.

For odd `3 <= q <= u32::MAX`, `sample_q_centered` MUST read little-endian u32
candidates and accept exactly those below
`floor(2^32 / q) * q`. It reduces an accepted candidate modulo q, then
subtracts q exactly when the residue exceeds `(q-1)/2`. Thus every centered
representative has the same number of accepted preimages. q is a parameter;
the implementation supports both q24 = 14,253,409 and q25 = 33,568,993.

The P stream is separate. `sample_p` MUST read little-endian u128 candidates
and accept exactly those less than `P = 2^128 - 275`, returning a canonical P
residue. A P stream MUST reject a q-sampling request and conversely. Both
samplers return `AkitaError::InvalidSetup` after 256 rejected candidates;
conditional on returning a value, their distributions are exact. No P sample
is reduced modulo q. The typed domains distinguish the q commitment matrix
from the P-field image, witness, and projection setup streams, even when they
use the same seed.

## Matrix seed and stream field

`akita_labinius_pcs::derive_root_setup` is the derivation of the PCS
composition, for every proof-field family. Given a geometry and the
`AkitaSetupSeed` of the nested Akita setup, it MUST compute

```text
matrix_seed = Blake2b-256(LP(b"akita/labinius/root-matrix-seed/v1") || LP(nested_seed))
```

with `LP` as defined under [canonical identity](#canonical-identity),
`nested_seed` in its canonical compressed encoding (the derivation tag byte,
then the 32 seed bytes) and the digest of `akita_params::digest_descriptor_bytes`.
It MUST then call `AdmittedRootSetup::derive::<Prime128Offset275>` with a seed
that has the nested seed's derivation algorithm and `matrix_seed` as its bytes.

`Prime128Offset275` remains the validated field descriptor passed by this
legacy composition, whichever proof field is used. It does not supply the q
matrix candidates, so changing a valid descriptor does not change that matrix.
The labelled matrix seed and the typed q domain separate the root matrix from
the nested ordinary Akita setup stream. The root matrix is no longer a reduced
prefix of the nested P-field matrix.

The seed bytes of the admitted identity (item 8 below) are `matrix_seed`. The
nested seed is bound by the nested Akita setup identity, which the PCS
composition absorbs separately.

## Canonical identity

Define `LP(x) = u64_le(byte_length(x)) || x`. All integer encodings below are
fixed-width little-endian; there is no platform-dependent integer encoding.
`AdmittedRootSetup::identity_bytes::<H>()` MUST concatenate, in order:

1. `LP(b"akita/labinius/admitted-root-setup/v1")`.
2. `LP(LabiniusRootProfile::identity_bytes())` for the admitted profile.
3. `log_num_cells` as u32.
4. `log_fold_width` as u32.
5. `lambda_fold` as u32.
6. The shape's admitted `rank_a` as u32.
7. The seed derivation tag as one byte: `1` for `Shake256PagedV1`, the existing
   public-seed wire tag.
8. The seed's 32 raw bytes.
9. `LP(BinaryClearSetup::identity_bytes::<H>())`.

The profile identity binds the parameter choices and certified width-table
digest. The nested clear identity binds `H`, scalar and commitment degrees,
trinomial sign, packing and matrix geometry, interval, fold budget, challenge
identity, and existing matrix-view digest. That digest binds `q` and the tight
row-major matrix coefficients and is transcript-backend specific. This
contract MUST NOT introduce a second matrix digest.

The identity names neither the validated field descriptor nor a proof field.
The existing matrix-view digest binds the residues derived by the typed q
stream. A protocol over a proof field MUST bind that field itself; the root
reduction absorbs the characteristic of its base field and the degree and multiplication table of
its challenge field
([root reduction](labinius-root-reduction.md#transcript-and-wire-grammar)).

## Boundary

This contract binds setup derivation and admission identities. It does not
prove binding of a source, discharge protocol composition obligations, cover
prepared limb-form matrices, or select the extension. The existing clear
protocol consumes `setup()` and binds its existing clear identity; it does not
automatically absorb the enclosing admitted-root identity. The [root reduction](labinius-root-reduction.md) MUST absorb the full admitted
identity before its frontend challenges. Its binding order is defined there;
source extraction, the SIS reduction and Fiat–Shamir security remain open.
Other callers needing that full identity MUST bind it explicitly in their own
setup context.
