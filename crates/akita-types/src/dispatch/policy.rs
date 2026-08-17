//! Canonical protocol dispatch arm tables (tier × slot).
//!
//! Edit the policy block at the bottom of this file when adding ring degrees or
//! tiers. Validators and dispatch macros are generated from that block.

use super::{ProtocolDispatchSlot, ProtocolRingDispatchTierId, RingRole};

pub(crate) const fn slice_contains(slice: &[usize], d: usize) -> bool {
    let mut i = 0;
    while i < slice.len() {
        if slice[i] == d {
            return true;
        }
        i += 1;
    }
    false
}

#[doc(hidden)]
#[macro_export]
macro_rules! __apply_protocol_dispatch_policy {
    (
        [define]
        Fp128: {
            inner: [$($i128:literal),* $(,)?]
            outer: [$($o128:literal),* $(,)?]
            opening: [$($p128:literal),* $(,)?]
            ntt: [$($n128:literal),* $(,)?]
            compression: [$($c128:literal),* $(,)?]
        }
        Fp64: {
            inner: [$($i64:literal),* $(,)?]
            outer: [$($o64:literal),* $(,)?]
            opening: [$($p64:literal),* $(,)?]
            ntt: [$($n64:literal),* $(,)?]
            compression: [$($c64:literal),* $(,)?]
        }
        Fp32: {
            inner: [$($i32:literal),* $(,)?]
            outer: [$($o32:literal),* $(,)?]
            opening: [$($p32:literal),* $(,)?]
            ntt: [$($n32:literal),* $(,)?]
            compression: [$($c32:literal),* $(,)?]
        }
    ) => {
        #[inline]
        #[must_use]
        pub const fn outer_opening_min_ring_d(tier: ProtocolRingDispatchTierId) -> usize {
            arms_for_slot(tier, ProtocolDispatchSlot::Role(RingRole::Outer))[0]
        }

        /// Minimum ring degree implemented by the NTT layer for `tier`.
        #[inline]
        #[must_use]
        pub const fn ntt_min_ring_d(tier: ProtocolRingDispatchTierId) -> usize {
            arms_for_slot(tier, ProtocolDispatchSlot::Ntt)[0]
        }

        #[inline]
        #[must_use]
        pub const fn ntt_max_ring_d(tier: ProtocolRingDispatchTierId) -> usize {
            let arms = arms_for_slot(tier, ProtocolDispatchSlot::Ntt);
            arms[arms.len() - 1]
        }

        const fn arms_for_slot(tier: ProtocolRingDispatchTierId, slot: ProtocolDispatchSlot) -> &'static [usize] {
            match (tier, slot) {
                (ProtocolRingDispatchTierId::Fp128, ProtocolDispatchSlot::Role(RingRole::Inner)) => {
                    &[$($i128),*]
                }
                (ProtocolRingDispatchTierId::Fp128, ProtocolDispatchSlot::Role(RingRole::Outer)) => {
                    &[$($o128),*]
                }
                (ProtocolRingDispatchTierId::Fp128, ProtocolDispatchSlot::Role(RingRole::Opening)) => {
                    &[$($p128),*]
                }
                (ProtocolRingDispatchTierId::Fp128, ProtocolDispatchSlot::Ntt) => &[$($n128),*],
                (ProtocolRingDispatchTierId::Fp128, ProtocolDispatchSlot::Compression) => &[$($c128),*],
                (ProtocolRingDispatchTierId::Fp64, ProtocolDispatchSlot::Role(RingRole::Inner)) => {
                    &[$($i64),*]
                }
                (ProtocolRingDispatchTierId::Fp64, ProtocolDispatchSlot::Role(RingRole::Outer)) => {
                    &[$($o64),*]
                }
                (ProtocolRingDispatchTierId::Fp64, ProtocolDispatchSlot::Role(RingRole::Opening)) => {
                    &[$($p64),*]
                }
                (ProtocolRingDispatchTierId::Fp64, ProtocolDispatchSlot::Ntt) => &[$($n64),*],
                (ProtocolRingDispatchTierId::Fp64, ProtocolDispatchSlot::Compression) => &[$($c64),*],
                (ProtocolRingDispatchTierId::Fp32, ProtocolDispatchSlot::Role(RingRole::Inner)) => {
                    &[$($i32),*]
                }
                (ProtocolRingDispatchTierId::Fp32, ProtocolDispatchSlot::Role(RingRole::Outer)) => {
                    &[$($o32),*]
                }
                (ProtocolRingDispatchTierId::Fp32, ProtocolDispatchSlot::Role(RingRole::Opening)) => {
                    &[$($p32),*]
                }
                (ProtocolRingDispatchTierId::Fp32, ProtocolDispatchSlot::Ntt) => &[$($n32),*],
                (ProtocolRingDispatchTierId::Fp32, ProtocolDispatchSlot::Compression) => &[$($c32),*],
            }
        }

        pub(crate) fn role_ring_dimensions_for_tier(
            tier: ProtocolRingDispatchTierId,
            role: RingRole,
        ) -> &'static [usize] {
            arms_for_slot(tier, ProtocolDispatchSlot::Role(role))
        }

        /// Whether `d` is a supported ring degree for `tier` and `slot`.
        #[inline]
        #[must_use]
        pub fn slot_dim_supported_for_tier(
            tier: ProtocolRingDispatchTierId,
            slot: ProtocolDispatchSlot,
            d: usize,
        ) -> bool {
            slice_contains(arms_for_slot(tier, slot), d)
        }

        /// Whether `d` is a supported ring degree for matrix `role` on `tier`.
        #[inline]
        #[must_use]
        pub fn role_dim_supported_for_tier(
            tier: ProtocolRingDispatchTierId,
            role: RingRole,
            d: usize,
        ) -> bool {
            slot_dim_supported_for_tier(tier, ProtocolDispatchSlot::Role(role), d)
        }

        /// Whether `d` is supported for production compression on `tier`.
        #[inline]
        #[must_use]
        pub fn compression_ring_dim_supported_for_tier(
            tier: ProtocolRingDispatchTierId,
            d: usize,
        ) -> bool {
            slot_dim_supported_for_tier(tier, ProtocolDispatchSlot::Compression, d)
        }

        /// Whether `d` is a supported inner (A-role) ring degree for `tier`.
        #[inline]
        #[must_use]
        pub fn inner_ring_dim_supported_for_tier(tier: ProtocolRingDispatchTierId, d: usize) -> bool {
            role_dim_supported_for_tier(tier, RingRole::Inner, d)
        }

        /// Whether `d` is a supported outer (B-role) ring degree for `tier`.
        #[inline]
        #[must_use]
        pub fn outer_ring_dim_supported_for_tier(tier: ProtocolRingDispatchTierId, d: usize) -> bool {
            role_dim_supported_for_tier(tier, RingRole::Outer, d)
        }

        /// Whether `d` is a supported opening (D-role) ring degree for `tier`.
        #[inline]
        #[must_use]
        pub fn opening_ring_dim_supported_for_tier(tier: ProtocolRingDispatchTierId, d: usize) -> bool {
            role_dim_supported_for_tier(tier, RingRole::Opening, d)
        }

        /// Whether `d` is supported on outer or opening roles for `tier`.
        #[inline]
        #[must_use]
        pub fn outer_opening_ring_dim_supported_for_tier(
            tier: ProtocolRingDispatchTierId,
            d: usize,
        ) -> bool {
            outer_ring_dim_supported_for_tier(tier, d) || opening_ring_dim_supported_for_tier(tier, d)
        }

    };

    (
        [dispatch $slot:ident, $F:ty, $d:expr, |$D:ident| $body:expr]
        Fp128: {
            inner: [$($i128:literal),* $(,)?]
            outer: [$($o128:literal),* $(,)?]
            opening: [$($p128:literal),* $(,)?]
            ntt: [$($n128:literal),* $(,)?]
            compression: [$($c128:literal),* $(,)?]
        }
        Fp64: {
            inner: [$($i64:literal),* $(,)?]
            outer: [$($o64:literal),* $(,)?]
            opening: [$($p64:literal),* $(,)?]
            ntt: [$($n64:literal),* $(,)?]
            compression: [$($c64:literal),* $(,)?]
        }
        Fp32: {
            inner: [$($i32:literal),* $(,)?]
            outer: [$($o32:literal),* $(,)?]
            opening: [$($p32:literal),* $(,)?]
            ntt: [$($n32:literal),* $(,)?]
            compression: [$($c32:literal),* $(,)?]
        }
    ) => {
        $crate::__dispatch_protocol_policy_slot!(
            $slot, $F, $d, |$D| $body;
            inner: { fp128: [$($i128),*], fp64: [$($i64),*], fp32: [$($i32),*] }
            outer: { fp128: [$($o128),*], fp64: [$($o64),*], fp32: [$($o32),*] }
            opening: { fp128: [$($p128),*], fp64: [$($p64),*], fp32: [$($p32),*] }
            ntt: { fp128: [$($n128),*], fp64: [$($n64),*], fp32: [$($n32),*] }
            compression: { fp128: [$($c128),*], fp64: [$($c64),*], fp32: [$($c32),*] }
        )
    };
}

/// Expand `d` against a fixed arm list for const-generic monomorphization.
#[doc(hidden)]
#[macro_export]
macro_rules! __dispatch_ring_dim_arms {
    ($d:expr, $D:ident, $body:expr, { $($dim:literal),+ $(,)? }) => {{
        let __d = $d;
        match __d {
            $( $dim => {
                const $D: usize = $dim;
                $body
            }, )+
            _ => Err(akita_field::AkitaError::InvalidSetup(format!(
                "unsupported ring dimension {__d} for this role/tier dispatch table"
            ))),
        }
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __dispatch_protocol_policy_tiers {
    (
        $F:ty, $d:expr, |$D:ident| $body:expr;
        fp128: [$($d128:literal),+ $(,)?],
        fp64: [$($d64:literal),+ $(,)?],
        fp32: [$($d32:literal),+ $(,)?]
    ) => {{
        match $crate::protocol_dispatch_tier::<$F>() {
            $crate::ProtocolRingDispatchTierId::Fp128 => {
                $crate::__dispatch_ring_dim_arms!($d, $D, $body, { $($d128),+ })
            }
            $crate::ProtocolRingDispatchTierId::Fp64 => {
                $crate::__dispatch_ring_dim_arms!($d, $D, $body, { $($d64),+ })
            }
            $crate::ProtocolRingDispatchTierId::Fp32 => {
                $crate::__dispatch_ring_dim_arms!($d, $D, $body, { $($d32),+ })
            }
        }
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __dispatch_protocol_policy_slot {
    (
        inner, $F:ty, $d:expr, |$D:ident| $body:expr;
        inner: { fp128: [$($d128:literal),+], fp64: [$($d64:literal),+], fp32: [$($d32:literal),+] }
        $($rest:tt)*
    ) => {
        $crate::__dispatch_protocol_policy_tiers!(
            $F, $d, |$D| $body;
            fp128: [$($d128),+], fp64: [$($d64),+], fp32: [$($d32),+]
        )
    };
    (
        outer, $F:ty, $d:expr, |$D:ident| $body:expr;
        inner: { $($inner:tt)* }
        outer: { fp128: [$($d128:literal),+], fp64: [$($d64:literal),+], fp32: [$($d32:literal),+] }
        $($rest:tt)*
    ) => {
        $crate::__dispatch_protocol_policy_tiers!(
            $F, $d, |$D| $body;
            fp128: [$($d128),+], fp64: [$($d64),+], fp32: [$($d32),+]
        )
    };
    (
        opening, $F:ty, $d:expr, |$D:ident| $body:expr;
        inner: { $($inner:tt)* }
        outer: { $($outer:tt)* }
        opening: { fp128: [$($d128:literal),+], fp64: [$($d64:literal),+], fp32: [$($d32:literal),+] }
        $($rest:tt)*
    ) => {
        $crate::__dispatch_protocol_policy_tiers!(
            $F, $d, |$D| $body;
            fp128: [$($d128),+], fp64: [$($d64),+], fp32: [$($d32),+]
        )
    };
    (
        ntt, $F:ty, $d:expr, |$D:ident| $body:expr;
        inner: { $($inner:tt)* }
        outer: { $($outer:tt)* }
        opening: { $($opening:tt)* }
        ntt: { fp128: [$($d128:literal),+], fp64: [$($d64:literal),+], fp32: [$($d32:literal),+] }
        $($rest:tt)*
    ) => {
        $crate::__dispatch_protocol_policy_tiers!(
            $F, $d, |$D| $body;
            fp128: [$($d128),+], fp64: [$($d64),+], fp32: [$($d32),+]
        )
    };
    (
        compression, $F:ty, $d:expr, |$D:ident| $body:expr;
        inner: { $($inner:tt)* }
        outer: { $($outer:tt)* }
        opening: { $($opening:tt)* }
        ntt: { $($ntt:tt)* }
        compression: { fp128: [$($d128:literal),+], fp64: [$($d64:literal),+], fp32: [$($d32:literal),+] }
    ) => {
        $crate::__dispatch_protocol_policy_tiers!(
            $F, $d, |$D| $body;
            fp128: [$($d128),+], fp64: [$($d64),+], fp32: [$($d32),+]
        )
    };
}

#[macro_export]
macro_rules! dispatch_for_field {
    ($crate::ProtocolDispatchSlot::Role($crate::RingRole::Inner), $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__protocol_dispatch_policy!(dispatch inner, $F, $d, |$D| $body)
    };
    (ProtocolDispatchSlot::Role(RingRole::Inner), $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__protocol_dispatch_policy!(dispatch inner, $F, $d, |$D| $body)
    };
    (akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner), $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__protocol_dispatch_policy!(dispatch inner, $F, $d, |$D| $body)
    };

    ($crate::ProtocolDispatchSlot::Role($crate::RingRole::Outer), $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__protocol_dispatch_policy!(dispatch outer, $F, $d, |$D| $body)
    };
    (ProtocolDispatchSlot::Role(RingRole::Outer), $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__protocol_dispatch_policy!(dispatch outer, $F, $d, |$D| $body)
    };
    (akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Outer), $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__protocol_dispatch_policy!(dispatch outer, $F, $d, |$D| $body)
    };

    ($crate::ProtocolDispatchSlot::Role($crate::RingRole::Opening), $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__protocol_dispatch_policy!(dispatch opening, $F, $d, |$D| $body)
    };
    (ProtocolDispatchSlot::Role(RingRole::Opening), $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__protocol_dispatch_policy!(dispatch opening, $F, $d, |$D| $body)
    };
    (akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Opening), $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__protocol_dispatch_policy!(dispatch opening, $F, $d, |$D| $body)
    };

    ($crate::ProtocolDispatchSlot::Ntt, $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__protocol_dispatch_policy!(dispatch ntt, $F, $d, |$D| $body)
    };
    (ProtocolDispatchSlot::Ntt, $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__protocol_dispatch_policy!(dispatch ntt, $F, $d, |$D| $body)
    };
    (akita_types::ProtocolDispatchSlot::Ntt, $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__protocol_dispatch_policy!(dispatch ntt, $F, $d, |$D| $body)
    };

    ($crate::ProtocolDispatchSlot::Compression, $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__protocol_dispatch_policy!(dispatch compression, $F, $d, |$D| $body)
    };
    (ProtocolDispatchSlot::Compression, $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__protocol_dispatch_policy!(dispatch compression, $F, $d, |$D| $body)
    };
    (akita_types::ProtocolDispatchSlot::Compression, $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__protocol_dispatch_policy!(dispatch compression, $F, $d, |$D| $body)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __protocol_dispatch_policy {
    ($($operation:tt)*) => {
        $crate::__apply_protocol_dispatch_policy! {
            [$($operation)*]
            Fp128: {
                inner: [64, 128, 256, 512]
                outer: [64, 128, 256]
                opening: [64, 128, 256]
                ntt: [16, 32, 64, 128, 256, 512]
                compression: [8, 16]
            }
            Fp64: {
                inner: [64, 128, 256, 512]
                outer: [64, 128, 256]
                opening: [64, 128, 256]
                ntt: [32, 64, 128, 256, 512, 1024]
                compression: [16, 32]
            }
            Fp32: {
                inner: [64, 128, 256, 512, 1024]
                outer: [64, 128, 256]
                opening: [64, 128, 256]
                ntt: [64, 128, 256, 512, 1024, 2048]
                compression: [32, 64]
            }
        }
    };
}

__protocol_dispatch_policy!(define);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compression_ring_dimensions, SisModulusProfileId, SUPPORTED_COMMITMENT_RING_DIMS};

    #[test]
    fn commitment_dimension_domain_matches_role_policy_union() {
        let tiers = [
            ProtocolRingDispatchTierId::Fp128,
            ProtocolRingDispatchTierId::Fp64,
            ProtocolRingDispatchTierId::Fp32,
        ];
        let roles = [RingRole::Inner, RingRole::Outer, RingRole::Opening];

        for tier in tiers {
            for role in roles {
                for &d in arms_for_slot(tier, ProtocolDispatchSlot::Role(role)) {
                    assert!(
                        SUPPORTED_COMMITMENT_RING_DIMS.contains(&d),
                        "{tier:?} {role:?} admits structurally unsupported d={d}"
                    );
                }
            }
        }
        for &d in &SUPPORTED_COMMITMENT_RING_DIMS {
            assert!(
                tiers.iter().any(|&tier| {
                    roles.iter().any(|&role| {
                        slice_contains(arms_for_slot(tier, ProtocolDispatchSlot::Role(role)), d)
                    })
                }),
                "structural commitment d={d} has no executable role dispatch"
            );
        }
    }

    #[test]
    fn outer_and_opening_share_arms_today() {
        for tier in [
            ProtocolRingDispatchTierId::Fp128,
            ProtocolRingDispatchTierId::Fp64,
            ProtocolRingDispatchTierId::Fp32,
        ] {
            assert_eq!(
                arms_for_slot(tier, ProtocolDispatchSlot::Role(RingRole::Outer)),
                arms_for_slot(tier, ProtocolDispatchSlot::Role(RingRole::Opening)),
                "outer/opening diverged for {tier:?}; split policy rows intentionally"
            );
        }
    }

    #[test]
    fn ntt_arms_are_powers_of_two_within_tier_band() {
        for tier in [
            ProtocolRingDispatchTierId::Fp128,
            ProtocolRingDispatchTierId::Fp64,
            ProtocolRingDispatchTierId::Fp32,
        ] {
            let arms = arms_for_slot(tier, ProtocolDispatchSlot::Ntt);
            assert!(!arms.is_empty());
            assert_eq!(arms[0], ntt_min_ring_d(tier));
            assert_eq!(*arms.last().expect("ntt arms"), ntt_max_ring_d(tier));
            for &d in arms {
                assert!(d.is_power_of_two());
            }
            for w in 1..arms.len() {
                assert_eq!(arms[w], arms[w - 1] * 2);
            }
        }
    }

    #[test]
    fn compression_arms_are_exactly_the_two_profile_ladder_dimensions() {
        for (tier, profile) in [
            (
                ProtocolRingDispatchTierId::Fp128,
                SisModulusProfileId::Q128OffsetA7F7,
            ),
            (
                ProtocolRingDispatchTierId::Fp64,
                SisModulusProfileId::Q64Offset59,
            ),
            (
                ProtocolRingDispatchTierId::Fp32,
                SisModulusProfileId::Q32Offset99,
            ),
        ] {
            let mut canonical = compression_ring_dimensions(profile);
            canonical.sort_unstable();
            assert_eq!(
                arms_for_slot(tier, ProtocolDispatchSlot::Compression),
                canonical
            );
        }
    }

    #[test]
    fn slot_support_matches_every_policy_arm() {
        for tier in [
            ProtocolRingDispatchTierId::Fp128,
            ProtocolRingDispatchTierId::Fp64,
            ProtocolRingDispatchTierId::Fp32,
        ] {
            for slot in [
                ProtocolDispatchSlot::Role(RingRole::Inner),
                ProtocolDispatchSlot::Role(RingRole::Outer),
                ProtocolDispatchSlot::Role(RingRole::Opening),
                ProtocolDispatchSlot::Ntt,
                ProtocolDispatchSlot::Compression,
            ] {
                for &d in arms_for_slot(tier, slot) {
                    assert!(
                        slot_dim_supported_for_tier(tier, slot, d),
                        "{tier:?} {slot:?} d={d}"
                    );
                }
                assert!(!slot_dim_supported_for_tier(tier, slot, 0));
                if !slice_contains(arms_for_slot(tier, slot), 48) {
                    assert!(!slot_dim_supported_for_tier(tier, slot, 48));
                }
            }
        }
    }
}
