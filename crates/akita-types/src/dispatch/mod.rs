//! Runtime-to-const-generic dispatch shared by prover and verifier.
//!
//! Fold / ring-switch paths use **role × PCS field tier** tables (see
//! `specs/ring-dim-challenge-cutover.md`). NTT cache build uses field tier only.
//!
//! `protocol_dispatch_policy!` is the semantic arm authority. Synchronization
//! tests exhaustively compare [`crate::dispatch_for_field!`] with that policy.

mod policy;

use crate::layout::{CommitmentRingDims, RingRole};
use crate::sis::SisModulusFamily;
use akita_algebra::ntt::tables::{Q32_MODULUS, Q64_MODULUS};
use akita_field::{AkitaError, CanonicalField};

pub use policy::{
    inner_ring_dim_supported_for_tier, ntt_max_ring_d, ntt_min_ring_d,
    opening_ring_dim_supported_for_tier, outer_opening_ring_dim_supported_for_tier,
    outer_ring_dim_supported_for_tier, role_dim_supported_for_tier, slot_dim_supported_for_tier,
    slot_dims_for_tier,
};

/// PCS base-field tier for protocol and NTT dispatch arm tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProtocolRingDispatchTierId {
    /// 128-bit production field (`Prime128OffsetA7F7` and siblings).
    Fp128,
    /// 64-bit small field.
    Fp64,
    /// 32-bit small field.
    Fp32,
}

/// Which const-generic monomorphization bucket to select at a dispatch call site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProtocolDispatchSlot {
    /// A/B/D matrix role (`RingRole`).
    Role(RingRole),
    /// Commitment-compression matrices.
    Compression,
    /// Setup seed `gen_ring_dim`.
    Envelope,
    /// CRT/NTT cache warm and build.
    Ntt,
}

/// Canonical field modulus from the canonical representation of `-1`.
///
/// Uses the identity: the canonical form of `-1` in `Z_q` is `q - 1`.
#[inline]
pub fn field_modulus<F: CanonicalField>() -> u128 {
    (-F::one()).to_canonical_u128() + 1
}

/// Classify `F` into a dispatch tier from its modulus (Q32 / Q64 / Q128 CRT bands).
#[inline]
pub fn protocol_dispatch_tier<F: CanonicalField>() -> ProtocolRingDispatchTierId {
    let modulus = field_modulus::<F>();
    if modulus <= Q32_MODULUS as u128 {
        ProtocolRingDispatchTierId::Fp32
    } else if modulus <= Q64_MODULUS as u128 {
        ProtocolRingDispatchTierId::Fp64
    } else {
        ProtocolRingDispatchTierId::Fp128
    }
}

/// SIS modulus family canonically associated with the concrete PCS field.
#[inline]
pub fn sis_family_for_field<F: CanonicalField>() -> SisModulusFamily {
    match protocol_dispatch_tier::<F>() {
        ProtocolRingDispatchTierId::Fp128 => SisModulusFamily::Q128,
        ProtocolRingDispatchTierId::Fp64 => SisModulusFamily::Q64,
        ProtocolRingDispatchTierId::Fp32 => SisModulusFamily::Q32,
    }
}

/// Whether `d` is a supported NTT ring degree for `tier`.
#[inline]
#[must_use]
pub fn ntt_ring_degree_supported_for_tier(tier: ProtocolRingDispatchTierId, d: usize) -> bool {
    slot_dim_supported_for_tier(tier, ProtocolDispatchSlot::Ntt, d)
}

/// Whether `d` is a supported NTT ring degree for PCS field `F`.
#[inline]
#[must_use]
pub fn ntt_ring_degree_supported_for_field<F: CanonicalField>(d: usize) -> bool {
    ntt_ring_degree_supported_for_tier(protocol_dispatch_tier::<F>(), d)
}

/// Field-tier validation for per-role dimensions after global [`crate::validate_role_dims`].
///
/// Rejects B/D dimensions below the tier floor and any role dimension outside
/// the live protocol dispatch arm tables for this PCS field tier.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when a role dimension is unsupported.
pub fn validate_role_dims_for_field<F: CanonicalField>(
    dims: CommitmentRingDims,
) -> Result<(), AkitaError> {
    let tier = protocol_dispatch_tier::<F>();
    for (role, d) in [
        (RingRole::Outer, dims.outer),
        (RingRole::Opening, dims.opening),
    ] {
        if !role_dim_supported_for_tier(tier, role, d) {
            return Err(AkitaError::InvalidSetup(format!(
                "{role:?} ring dimension {d} is outside the protocol dispatch table for this PCS field tier"
            )));
        }
    }
    if !role_dim_supported_for_tier(tier, RingRole::Inner, dims.inner) {
        return Err(AkitaError::InvalidSetup(format!(
            "A-role ring dimension {} is outside the inner protocol dispatch table for this PCS field tier",
            dims.inner
        )));
    }
    Ok(())
}

/// Validate that a const-generic ring dimension is supported for dispatch.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when `D` is zero or not a power of two.
#[inline]
pub fn validate_ring_dispatch<const D: usize>() -> Result<usize, AkitaError> {
    if D == 0 || !D.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "ring dimension must be a non-zero power of two".to_string(),
        ));
    }
    Ok(D.trailing_zeros() as usize)
}

/// Validate that schedule level params match the dispatched role dimension.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] on dimension mismatch.
#[inline]
pub fn validate_role_dispatch<const D: usize>(
    dims: CommitmentRingDims,
    role: RingRole,
) -> Result<usize, AkitaError> {
    let ring_bits = validate_ring_dispatch::<D>()?;
    if dims.dim_for(role) != D {
        return Err(AkitaError::InvalidSetup(format!(
            "role {:?} ring dimension {} does not match dispatch D={D}",
            role,
            dims.dim_for(role)
        )));
    }
    Ok(ring_bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch_for_field;
    use akita_field::{Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};

    #[test]
    fn protocol_dispatch_tier_classifies_fields() {
        assert_eq!(
            protocol_dispatch_tier::<Prime128OffsetA7F7>(),
            ProtocolRingDispatchTierId::Fp128
        );
        assert_eq!(
            protocol_dispatch_tier::<Prime64Offset59>(),
            ProtocolRingDispatchTierId::Fp64
        );
        assert_eq!(
            protocol_dispatch_tier::<Prime32Offset99>(),
            ProtocolRingDispatchTierId::Fp32
        );
    }

    fn assert_dispatch_matches_policy<F: CanonicalField>(tier: ProtocolRingDispatchTierId) {
        // Exhaust the complete range through the largest production arm. This
        // deliberately does not use another hand-maintained union of arms:
        // omissions, extra arms, and accidental non-power-of-two arms must all
        // disagree with the policy predicate here.
        for d in 0..=4096 {
            macro_rules! assert_slot {
                ($slot:expr, $dispatch:expr) => {
                    assert_eq!(
                        $dispatch.is_ok(),
                        slot_dim_supported_for_tier(tier, $slot, d),
                        "dispatch/policy mismatch for {:?} at d={d}",
                        $slot
                    );
                };
            }
            assert_slot!(
                ProtocolDispatchSlot::Role(RingRole::Inner),
                dispatch_for_field!(ProtocolDispatchSlot::Role(RingRole::Inner), F, d, |D| {
                    Ok::<usize, AkitaError>(D)
                })
            );
            assert_slot!(
                ProtocolDispatchSlot::Role(RingRole::Outer),
                dispatch_for_field!(ProtocolDispatchSlot::Role(RingRole::Outer), F, d, |D| {
                    Ok::<usize, AkitaError>(D)
                })
            );
            assert_slot!(
                ProtocolDispatchSlot::Role(RingRole::Opening),
                dispatch_for_field!(ProtocolDispatchSlot::Role(RingRole::Opening), F, d, |D| {
                    Ok::<usize, AkitaError>(D)
                })
            );
            assert_slot!(
                ProtocolDispatchSlot::Compression,
                dispatch_for_field!(ProtocolDispatchSlot::Compression, F, d, |D| {
                    Ok::<usize, AkitaError>(D)
                })
            );
            assert_slot!(
                ProtocolDispatchSlot::Envelope,
                dispatch_for_field!(ProtocolDispatchSlot::Envelope, F, d, |D| {
                    Ok::<usize, AkitaError>(D)
                })
            );
            assert_slot!(
                ProtocolDispatchSlot::Ntt,
                dispatch_for_field!(ProtocolDispatchSlot::Ntt, F, d, |D| {
                    Ok::<usize, AkitaError>(D)
                })
            );
        }
    }

    #[test]
    fn runtime_dispatch_arms_match_policy_for_every_slot_and_tier() {
        assert_dispatch_matches_policy::<Prime128OffsetA7F7>(ProtocolRingDispatchTierId::Fp128);
        assert_dispatch_matches_policy::<Prime64Offset59>(ProtocolRingDispatchTierId::Fp64);
        assert_dispatch_matches_policy::<Prime32Offset99>(ProtocolRingDispatchTierId::Fp32);
    }

    #[test]
    fn compression_dispatch_uses_tier_specific_caps() {
        for (d, accepted) in [(8, true), (16, true), (32, true), (64, true), (128, false)] {
            assert_eq!(
                dispatch_for_field!(
                    ProtocolDispatchSlot::Compression,
                    Prime128OffsetA7F7,
                    d,
                    |D| Ok::<usize, AkitaError>(D)
                )
                .is_ok(),
                accepted
            );
        }
        for (d, accepted) in [
            (8, false),
            (16, true),
            (32, true),
            (64, true),
            (128, true),
            (256, false),
        ] {
            assert_eq!(
                dispatch_for_field!(ProtocolDispatchSlot::Compression, Prime64Offset59, d, |D| {
                    Ok::<usize, AkitaError>(D)
                })
                .is_ok(),
                accepted
            );
        }
        for (d, accepted) in [
            (16, false),
            (32, true),
            (64, true),
            (128, true),
            (256, true),
            (512, false),
        ] {
            assert_eq!(
                dispatch_for_field!(ProtocolDispatchSlot::Compression, Prime32Offset99, d, |D| {
                    Ok::<usize, AkitaError>(D)
                })
                .is_ok(),
                accepted
            );
        }
    }

    #[test]
    fn existing_matrix_roles_reject_d8() {
        for result in [
            dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Inner),
                Prime128OffsetA7F7,
                8,
                |D| Ok::<usize, AkitaError>(D)
            ),
            dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Outer),
                Prime128OffsetA7F7,
                8,
                |D| Ok::<usize, AkitaError>(D)
            ),
            dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Opening),
                Prime128OffsetA7F7,
                8,
                |D| Ok::<usize, AkitaError>(D)
            ),
        ] {
            assert!(result.is_err());
        }
        assert!(crate::validate_role_dims(CommitmentRingDims {
            inner: 64,
            outer: 8,
            opening: 8,
        })
        .is_err());
    }

    #[test]
    fn inner_dispatch_fp128_rejects_d32_and_d256() {
        assert!(dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            Prime128OffsetA7F7,
            64usize,
            |D| Ok(D)
        )
        .is_ok());
        assert!(dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            Prime128OffsetA7F7,
            128usize,
            |D| Ok(D)
        )
        .is_ok());
        assert!(dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            Prime128OffsetA7F7,
            32usize,
            |D| Ok(D)
        )
        .is_err());
        assert!(dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            Prime128OffsetA7F7,
            256usize,
            |D| Ok(D)
        )
        .is_err());
    }

    #[test]
    fn outer_dispatch_rejects_d16_on_fp128() {
        assert!(dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Outer),
            Prime128OffsetA7F7,
            16usize,
            |D| Ok(D)
        )
        .is_err());
        assert!(dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Outer),
            Prime128OffsetA7F7,
            32usize,
            |D| Ok(D)
        )
        .is_ok());
    }

    #[test]
    fn outer_dispatch_fp32_rejects_d32() {
        assert!(dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Outer),
            Prime32Offset99,
            32usize,
            |D| Ok(D)
        )
        .is_err());
        assert!(dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Outer),
            Prime32Offset99,
            64usize,
            |D| Ok(D)
        )
        .is_ok());
    }

    #[test]
    fn ntt_dispatch_fp128_accepts_d16_and_caps_at_512() {
        assert!(dispatch_for_field!(
            ProtocolDispatchSlot::Ntt,
            Prime128OffsetA7F7,
            16usize,
            |D| Ok(D)
        )
        .is_ok());
        assert!(dispatch_for_field!(
            ProtocolDispatchSlot::Ntt,
            Prime128OffsetA7F7,
            512usize,
            |D| Ok(D)
        )
        .is_ok());
        assert!(dispatch_for_field!(
            ProtocolDispatchSlot::Ntt,
            Prime128OffsetA7F7,
            1024usize,
            |D| Ok(D)
        )
        .is_err());
    }

    #[test]
    fn ntt_dispatch_fp32_accepts_d32() {
        assert!(
            dispatch_for_field!(ProtocolDispatchSlot::Ntt, Prime32Offset99, 32usize, |D| Ok(
                D
            ))
            .is_ok()
        );
    }

    #[test]
    fn ntt_dispatch_fp32_reaches_2048() {
        assert!(
            dispatch_for_field!(ProtocolDispatchSlot::Ntt, Prime32Offset99, 2048usize, |D| {
                Ok(D)
            })
            .is_ok()
        );
    }

    #[test]
    fn tier_ntt_bounds() {
        assert_eq!(ntt_min_ring_d(ProtocolRingDispatchTierId::Fp128), 8);
        assert_eq!(ntt_min_ring_d(ProtocolRingDispatchTierId::Fp64), 16);
        assert_eq!(ntt_min_ring_d(ProtocolRingDispatchTierId::Fp32), 32);
        assert_eq!(ntt_max_ring_d(ProtocolRingDispatchTierId::Fp128), 512);
        assert_eq!(ntt_max_ring_d(ProtocolRingDispatchTierId::Fp64), 1024);
        assert_eq!(ntt_max_ring_d(ProtocolRingDispatchTierId::Fp32), 2048);
    }

    #[test]
    fn compression_dims_do_not_broaden_abd_matrix_validation() {
        assert!(slot_dim_supported_for_tier(
            ProtocolRingDispatchTierId::Fp128,
            ProtocolDispatchSlot::Compression,
            8
        ));
        for d in [8, 16] {
            assert!(crate::validate_role_dims(CommitmentRingDims {
                inner: 64,
                outer: d,
                opening: d,
            })
            .is_err());
        }
        assert!(crate::validate_role_dims(CommitmentRingDims {
            inner: 64,
            outer: 32,
            opening: 32,
        })
        .is_ok());
        assert!(!crate::layout::SUPPORTED_CHALLENGE_RING_DIMS.contains(&32));
    }

    #[test]
    fn validate_role_dims_for_field_rejects_ladder_dims_without_inner_dispatch() {
        let fp32_ok = CommitmentRingDims {
            inner: 64,
            outer: 64,
            opening: 64,
        };
        assert!(validate_role_dims_for_field::<Prime32Offset99>(fp32_ok).is_ok());

        let fp32_high_a = CommitmentRingDims {
            inner: 512,
            outer: 64,
            opening: 64,
        };
        assert!(validate_role_dims_for_field::<Prime32Offset99>(fp32_high_a).is_err());

        let fp128_high_b = CommitmentRingDims {
            inner: 64,
            outer: 512,
            opening: 64,
        };
        assert!(validate_role_dims_for_field::<Prime128OffsetA7F7>(fp128_high_b).is_err());
    }
}
