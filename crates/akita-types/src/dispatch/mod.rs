//! Runtime-to-const-generic dispatch shared by prover and verifier.
//!
//! Fold / ring-switch paths use **role × PCS field tier** tables (see
//! `book/src/foundations/ntt-crt.md`). NTT cache build uses field tier only.
//!
//! Arm lists come from the policy block in `dispatch/policy.rs`; validators and
//! [`crate::dispatch_for_field!`] expand from that single declaration.

mod policy;

use crate::layout::{CommitmentRingDims, RingRole};
use crate::sis::SisModulusProfileId;
use akita_algebra::ntt::tables::{Q32_MODULUS, Q64_MODULUS};
use akita_field::{AkitaError, CanonicalField};

pub(crate) use policy::role_ring_dimensions_for_tier;
pub use policy::{
    compression_ring_dim_supported_for_tier, inner_ring_dim_supported_for_tier, ntt_max_ring_d,
    ntt_min_ring_d, opening_ring_dim_supported_for_tier, outer_opening_min_ring_d,
    outer_opening_ring_dim_supported_for_tier, outer_ring_dim_supported_for_tier,
    role_dim_supported_for_tier, slot_dim_supported_for_tier,
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
    /// CRT/NTT cache warm and build.
    Ntt,
    /// Compression-only F/H matrices under the modulus-profile ladder.
    Compression,
}

/// Dispatch tier selected by one exact SIS modulus profile.
///
/// Planner/runtime policies carry the modulus profile rather than a concrete
/// field type, so policy validation uses this mapping to audit candidate ring
/// dimensions against the same role tables used by prover/verifier dispatch.
#[inline]
#[must_use]
pub const fn protocol_dispatch_tier_for_sis_profile(
    profile: SisModulusProfileId,
) -> ProtocolRingDispatchTierId {
    match profile {
        SisModulusProfileId::Q128OffsetA7F7 => ProtocolRingDispatchTierId::Fp128,
        SisModulusProfileId::Q64Offset59 => ProtocolRingDispatchTierId::Fp64,
        SisModulusProfileId::Q32Offset99 => ProtocolRingDispatchTierId::Fp32,
    }
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
/// Rejects A/B/D dimensions below 64 and any role dimension outside
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
        (RingRole::Inner, dims.inner),
        (RingRole::Outer, dims.outer),
        (RingRole::Opening, dims.opening),
    ] {
        if d < crate::MIN_A_ROLE_FOLD_CHALLENGE_RING_D {
            return Err(AkitaError::InvalidSetup(format!(
                "{role:?} commitment-matrix ring dimension {d} is below the protocol minimum 64"
            )));
        }
        if !role_dim_supported_for_tier(tier, role, d) {
            return Err(AkitaError::InvalidSetup(format!(
                "{role:?} ring dimension {d} is outside the protocol dispatch table for this PCS field tier"
            )));
        }
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

    #[test]
    fn inner_dispatch_fp128_accepts_through_d512() {
        for d in [64usize, 128, 256, 512] {
            assert_eq!(
                dispatch_for_field!(
                    ProtocolDispatchSlot::Role(RingRole::Inner),
                    Prime128OffsetA7F7,
                    d,
                    |D| Ok(D)
                )
                .expect("supported fp128 inner dimension"),
                d
            );
        }
        for d in [32usize, 1024] {
            assert!(
                dispatch_for_field!(
                    ProtocolDispatchSlot::Role(RingRole::Inner),
                    Prime128OffsetA7F7,
                    d,
                    |D| Ok(D)
                )
                .is_err(),
                "unsupported fp128 inner d={d} must be rejected"
            );
        }
    }

    #[test]
    fn small_field_commitment_dispatch_reaches_profile_caps() {
        for (d, expected) in [(512usize, 512), (1024, 1024)] {
            assert_eq!(
                dispatch_for_field!(
                    ProtocolDispatchSlot::Role(RingRole::Inner),
                    Prime32Offset99,
                    d,
                    |D| Ok(D)
                )
                .expect("supported fp32 inner dimension"),
                expected
            );
        }
        assert!(dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Outer),
            Prime64Offset59,
            512usize,
            |D| Ok(D)
        )
        .is_err());
        assert!(dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            Prime64Offset59,
            1024usize,
            |D| Ok(D)
        )
        .is_err());
    }

    #[test]
    fn outer_dispatch_floor_is_d64_on_every_profile() {
        for d in [16usize, 32] {
            assert!(dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Outer),
                Prime128OffsetA7F7,
                d,
                |D| Ok(D)
            )
            .is_err());
        }
        assert!(dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Outer),
            Prime128OffsetA7F7,
            64usize,
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
    fn shared_ntt_dispatch_fp128_includes_first_compression_stage_and_caps_at_d512() {
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
            64usize,
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
    fn ntt_dispatch_fp32_rejects_d32() {
        assert!(
            dispatch_for_field!(ProtocolDispatchSlot::Ntt, Prime32Offset99, 32usize, |D| Ok(
                D
            ))
            .is_err()
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
        assert_eq!(ntt_min_ring_d(ProtocolRingDispatchTierId::Fp128), 16);
        assert_eq!(ntt_min_ring_d(ProtocolRingDispatchTierId::Fp64), 32);
        assert_eq!(ntt_min_ring_d(ProtocolRingDispatchTierId::Fp32), 64);
        assert_eq!(ntt_max_ring_d(ProtocolRingDispatchTierId::Fp128), 512);
        assert_eq!(ntt_max_ring_d(ProtocolRingDispatchTierId::Fp64), 1024);
        assert_eq!(ntt_max_ring_d(ProtocolRingDispatchTierId::Fp32), 2048);
    }

    #[test]
    fn commitment_compression_and_ntt_dimension_domains_are_separate() {
        assert!(!crate::SUPPORTED_COMMITMENT_RING_DIMS.contains(&16));
        assert!(!crate::SUPPORTED_COMMITMENT_RING_DIMS.contains(&32));
        assert!(!crate::layout::SUPPORTED_CHALLENGE_RING_DIMS.contains(&32));
        assert_eq!(
            outer_opening_min_ring_d(ProtocolRingDispatchTierId::Fp128),
            64
        );
        assert_eq!(ntt_min_ring_d(ProtocolRingDispatchTierId::Fp128), 16);
        for (field_tier, dimensions) in [
            (ProtocolRingDispatchTierId::Fp128, [16, 8]),
            (ProtocolRingDispatchTierId::Fp64, [32, 16]),
            (ProtocolRingDispatchTierId::Fp32, [64, 32]),
        ] {
            assert!(dimensions
                .into_iter()
                .all(|d| compression_ring_dim_supported_for_tier(field_tier, d)));
        }
    }

    #[test]
    fn validate_role_dims_for_field_rejects_nonproduction_role_dimensions() {
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
        assert!(validate_role_dims_for_field::<Prime32Offset99>(fp32_high_a).is_ok());

        let fp32_high_b = CommitmentRingDims {
            inner: 512,
            outer: 512,
            opening: 64,
        };
        assert!(validate_role_dims_for_field::<Prime32Offset99>(fp32_high_b).is_err());

        let fp128_high_b = CommitmentRingDims {
            inner: 64,
            outer: 512,
            opening: 64,
        };
        assert!(validate_role_dims_for_field::<Prime128OffsetA7F7>(fp128_high_b).is_err());

        let fp128_high_a = CommitmentRingDims {
            inner: 512,
            outer: 256,
            opening: 64,
        };
        assert!(validate_role_dims_for_field::<Prime128OffsetA7F7>(fp128_high_a).is_ok());
    }
}
