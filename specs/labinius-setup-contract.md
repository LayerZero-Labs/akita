# LaBinius admitted root setup

Status: implemented
Book-chapter: book/src/how/architecture.md
Tracking: https://github.com/LayerZero-Labs/akita/issues/45

## Derivation and admission

`AdmittedRootSetup::derive` MUST derive `LabiniusRootShape` from the selected
`LabiniusRootProfile`, `log_num_cells`, `log_fold_width`, and `lambda_fold`.
It MUST use the shape's certified rank `n_A`, ring elements per column `m`,
and fold width `C`, with the profile's accepted interval, challenge family,
coefficient prime, and commitment degree. `BinaryClearSetup::new` MUST check
the coefficient field, degree, and trinomial sign against those identities.
An admitted root setup MUST NOT be constructed from a caller-supplied matrix.
Its setup, shape, seed, and derivation inputs are immutable public views.

`derive_trinomial_matrix` MUST expand exactly `rows * columns * D` field
elements with `derive_public_matrix_prefix::<F>`, passing the caller's
`AkitaSetupSeed` unchanged. No additional domain separation is applied.
The products MUST be checked before expansion. For tag 0, coefficient `t`
of `A[i,j]` MUST equal the public field element at

```text
(i * columns + j) * D + t,    0 <= t < D.
```

This is `TrinomialASetupView::setup_address(i,j,t)` with setup offset zero:
coefficients are contiguous, then columns, then rows. For tag 1, the canonical
integer of that field element is reduced modulo q0 before storage; its
reduced-view digest domain is specified in
[the small-modulus contract](labinius-small-modulus-root.md#exact-auxiliary-message-grammar-and-matrix-identity). The matrix element at
`i * columns + j` is the row-major element consumed by `apply_matrix`.
For an admitted root, the matrix dimensions are `n_A` by `m`; `C` counts source
columns and does not determine the matrix's column count.

The matrix is one view of a shared public field prefix. A deployment MAY store
the same prefix once for this view and ordinary Akita matrix views over the
same coefficient field and seed. The shared-prime view is uniform under the
seed expander's random-oracle assumption; tag 1 instead has the admitted
reduction bias specified in the small-modulus contract. No independence between
views is assumed. Expanding a longer prefix MUST preserve every earlier
coefficient.

This materialized API rejects zero dimensions, zero or odd degree, unsupported
field descriptors, more than `MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS`
coefficients, more than 64 MiB for either the flat prefix or matrix backing,
or more than 64 KiB per ring element. Tag 1 temporarily raises the prefix
and matrix byte caps to 128 MiB; the small-modulus contract records the planned
u32 storage and paged derivation needed to restore 64 MiB. These are
host-resource limits, not limits on the public stream or the numerical shape admission. It MUST perform
fallible reservation for the matrix and a flat-prefix allocation probe before
calling the bounded, infallibly allocating expander. The probe is released
before expansion; it cannot guarantee success against subsequent global
allocator exhaustion. Setup failures MUST return `AkitaError::InvalidSetup`.

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
digest. The nested clear identity binds `H`, the coefficient prime, scalar and
commitment degrees, trinomial sign, packing and matrix geometry, interval,
fold budget, challenge identity, and existing matrix-view digest. That digest
binds the tight row-major matrix coefficients and is transcript-backend
specific. This contract MUST NOT introduce a second matrix digest.

## Boundary

This contract binds setup derivation and admission identities. It does not
prove binding of a source, discharge protocol composition obligations, cover
prepared or NTT-form matrices, or select the extension. The existing clear
protocol consumes `setup()` and binds its existing clear identity; it does not
automatically absorb the enclosing admitted-root identity. The [root reduction](labinius-root-reduction.md) MUST absorb the full admitted
identity before its frontend challenges. Its binding order is defined there;
source extraction, the SIS reduction and Fiat–Shamir security remain open.
Other callers needing that full identity MUST bind it explicitly in their own
setup context.
