# LaBinius admitted root setup

Status: implemented
Book-chapter: book/src/how/architecture.md
Tracking: https://github.com/LayerZero-Labs/akita/issues/45

## Derivation and admission

`AdmittedRootSetup::derive::<P>` MUST derive `LabiniusRootShape` from the
selected `LabiniusRootProfile`, `log_num_cells`, `log_fold_width`, and
`lambda_fold`. It MUST use the shape's certified rank `n_A`, ring elements per
column `m`, fold width `C` and response interval, the balanced base-16 digit
range of [root admission](labinius-root-admission.md#fold-response), with the
profile's challenge family, commitment modulus `q` and commitment degree.
Before expanding, it MUST apply `LabiniusRootShape::check_derivation_bias` to
the characteristic of the stream field `P`
([root admission](labinius-root-admission.md#derivation-bias)).
`BinaryClearSetup::new` MUST check the degree, trinomial sign and commitment
modulus against those identities and MUST reject a matrix coefficient at
least `q`.
An admitted root setup MUST NOT be constructed from a caller-supplied matrix.
Its setup, shape, seed, and derivation inputs are immutable public views.
Deriving the setup admits no proof field: that is a separate check on the
shape, made where a proof field enters
([lowered relation](labinius-lowered-root.md#geometry-and-admission)).

`derive_trinomial_matrix::<P>` MUST expand exactly `rows * columns * D` field
elements with `derive_public_matrix_prefix::<P>`, passing the caller's
`AkitaSetupSeed` unchanged. No domain separation is applied at this layer; the
PCS composition separates the seed before the call
([matrix seed and stream field](#matrix-seed-and-stream-field)). `P`
names the stream field being reduced; it is a method-level parameter of
`AdmittedRootSetup::derive` and not part of the setup's type. The products
MUST be checked before expansion. Coefficient `t` of `A[i,j]` MUST equal the
canonical integer of the public field element at

```text
(i * columns + j) * D + t,    0 <= t < D,
```

reduced modulo `q` and stored as a u32. This is
`TrinomialASetupView::setup_address(i,j,t)` with setup offset zero:
coefficients are contiguous, then columns, then rows. The digest of this view
is specified in [the clear opening](labinius-clear-opening.md#setup-admission).
The matrix element at `i * columns + j` is the row-major element consumed by
`apply_matrix`.
For an admitted root, the matrix dimensions are `n_A` by `m`; `C` counts source
columns and does not determine the matrix's column count.

The matrix is a reduced view of the public field prefix of its seed over `P`.
The reduced view has the reduction bias the derivation-bias check admits.
Expanding a longer prefix MUST preserve every earlier coefficient. Given the
seed and field of an ordinary Akita setup, this derivation returns a reduced
copy of a prefix of that setup's matrix, and nothing may treat the two as
independent. The PCS composition never passes such a seed.

This materialized API rejects zero dimensions, zero or odd degree, unsupported
field descriptors, more than `MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS`
coefficients, a stream field wider than 128 bits, more than 64 MiB for the
u32 matrix backing, or more than 128 MiB for the transient flat prefix. These are
host-resource limits, not limits on the public stream or the numerical shape admission. It MUST perform
fallible reservation for the matrix and a flat-prefix allocation probe before
calling the bounded, infallibly allocating expander. The probe is released
before expansion; it cannot guarantee success against subsequent global
allocator exhaustion. Setup failures MUST return `AkitaError::InvalidSetup`.

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

The stream field is `2^128 - 275` whichever field the proof uses. A 64-bit
stream fails the derivation-bias check at every geometry
([root admission](labinius-root-admission.md#derivation-bias)), so the 64-bit
family cannot reduce its own field's stream. A fixed stream field also gives
one matrix, and therefore one clear image, for both families. The label keeps
the matrix stream apart from the stream that expands the nested setup: with
the nested seed itself, the 128-bit family's root matrix would be a prefix of
its nested Akita setup matrix reduced modulo `q`.

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

The identity names neither the stream field nor a proof field. The stream
field is bound only through the matrix residues it produced. A protocol over a
proof field MUST bind that field itself; the root reduction absorbs the
characteristic of its base field and the degree and multiplication table of
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
