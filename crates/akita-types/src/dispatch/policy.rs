//! Canonical protocol dispatch arm tables (tier × slot).
//!
//! Edit [`protocol_dispatch_policy!`] below when adding ring degrees or tiers.
//! Validators and dispatch macros are generated from this block.

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

macro_rules! protocol_dispatch_policy {
    (
        Fp128: {
            inner: [$($i128:literal),* $(,)?]
            outer: [$($o128:literal),* $(,)?]
            opening: [$($p128:literal),* $(,)?]
            envelope: [$($e128:literal),* $(,)?]
            ntt: [$($n128:literal),* $(,)?]
            min_bd: $min128:literal
            ntt_max: $max128:literal
        }
        Fp64: {
            inner: [$($i64:literal),* $(,)?]
            outer: [$($o64:literal),* $(,)?]
            opening: [$($p64:literal),* $(,)?]
            envelope: [$($e64:literal),* $(,)?]
            ntt: [$($n64:literal),* $(,)?]
            min_bd: $min64:literal
            ntt_max: $max64:literal
        }
        Fp32: {
            inner: [$($i32:literal),* $(,)?]
            outer: [$($o32:literal),* $(,)?]
            opening: [$($p32:literal),* $(,)?]
            envelope: [$($e32:literal),* $(,)?]
            ntt: [$($n32:literal),* $(,)?]
            min_bd: $min32:literal
            ntt_max: $max32:literal
        }
    ) => {
        /// All protocol dispatch arms across tiers and slots (sorted union for planner sync tests).
        #[allow(dead_code)]
        pub(crate) const ALL_PROTOCOL_DISPATCH_ARMS: &[usize] =
            &[16, 32, 64, 128, 256, 512, 1024, 2048];

        #[inline]
        #[must_use]
        pub const fn outer_opening_min_ring_d(tier: ProtocolRingDispatchTierId) -> usize {
            match tier {
                ProtocolRingDispatchTierId::Fp128 => $min128,
                ProtocolRingDispatchTierId::Fp64 => $min64,
                ProtocolRingDispatchTierId::Fp32 => $min32,
            }
        }

        #[inline]
        #[must_use]
        pub const fn ntt_max_ring_d(tier: ProtocolRingDispatchTierId) -> usize {
            match tier {
                ProtocolRingDispatchTierId::Fp128 => $max128,
                ProtocolRingDispatchTierId::Fp64 => $max64,
                ProtocolRingDispatchTierId::Fp32 => $max32,
            }
        }

        fn arms_for_slot(tier: ProtocolRingDispatchTierId, slot: ProtocolDispatchSlot) -> &'static [usize] {
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
                (ProtocolRingDispatchTierId::Fp128, ProtocolDispatchSlot::Envelope) => &[$($e128),*],
                (ProtocolRingDispatchTierId::Fp128, ProtocolDispatchSlot::Ntt) => &[$($n128),*],
                (ProtocolRingDispatchTierId::Fp64, ProtocolDispatchSlot::Role(RingRole::Inner)) => {
                    &[$($i64),*]
                }
                (ProtocolRingDispatchTierId::Fp64, ProtocolDispatchSlot::Role(RingRole::Outer)) => {
                    &[$($o64),*]
                }
                (ProtocolRingDispatchTierId::Fp64, ProtocolDispatchSlot::Role(RingRole::Opening)) => {
                    &[$($p64),*]
                }
                (ProtocolRingDispatchTierId::Fp64, ProtocolDispatchSlot::Envelope) => &[$($e64),*],
                (ProtocolRingDispatchTierId::Fp64, ProtocolDispatchSlot::Ntt) => &[$($n64),*],
                (ProtocolRingDispatchTierId::Fp32, ProtocolDispatchSlot::Role(RingRole::Inner)) => {
                    &[$($i32),*]
                }
                (ProtocolRingDispatchTierId::Fp32, ProtocolDispatchSlot::Role(RingRole::Outer)) => {
                    &[$($o32),*]
                }
                (ProtocolRingDispatchTierId::Fp32, ProtocolDispatchSlot::Role(RingRole::Opening)) => {
                    &[$($p32),*]
                }
                (ProtocolRingDispatchTierId::Fp32, ProtocolDispatchSlot::Envelope) => &[$($e32),*],
                (ProtocolRingDispatchTierId::Fp32, ProtocolDispatchSlot::Ntt) => &[$($n32),*],
            }
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

/// Runtime `d` → const-generic `D` for a protocol dispatch slot.
///
/// Arm literals must match [`protocol_dispatch_policy!`] in this file.
/// The slot argument must be a compile-time constant path so the closure is
/// only monomorphized over that slot's arms (not every tier × slot combination).
#[doc(hidden)]
#[macro_export]
macro_rules! __dispatch_for_field_inner {
    ($F:ty, $d:expr, |$D:ident| $body:expr) => {{
        match $crate::protocol_dispatch_tier::<$F>() {
            $crate::ProtocolRingDispatchTierId::Fp128 => {
                $crate::__dispatch_ring_dim_arms!($d, $D, $body, { 64, 128 })
            }
            $crate::ProtocolRingDispatchTierId::Fp64 => {
                $crate::__dispatch_ring_dim_arms!($d, $D, $body, { 64, 128, 256 })
            }
            $crate::ProtocolRingDispatchTierId::Fp32 => {
                $crate::__dispatch_ring_dim_arms!($d, $D, $body, { 64, 128, 256 })
            }
        }
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __dispatch_for_field_outer {
    ($F:ty, $d:expr, |$D:ident| $body:expr) => {{
        match $crate::protocol_dispatch_tier::<$F>() {
            $crate::ProtocolRingDispatchTierId::Fp128 => {
                $crate::__dispatch_ring_dim_arms!($d, $D, $body, { 16, 32, 64, 128, 256 })
            }
            $crate::ProtocolRingDispatchTierId::Fp64 => {
                $crate::__dispatch_ring_dim_arms!($d, $D, $body, { 32, 64, 128, 256 })
            }
            $crate::ProtocolRingDispatchTierId::Fp32 => {
                $crate::__dispatch_ring_dim_arms!($d, $D, $body, { 64, 128, 256 })
            }
        }
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __dispatch_for_field_opening {
    ($F:ty, $d:expr, |$D:ident| $body:expr) => {{
        match $crate::protocol_dispatch_tier::<$F>() {
            $crate::ProtocolRingDispatchTierId::Fp128 => {
                $crate::__dispatch_ring_dim_arms!($d, $D, $body, { 16, 32, 64, 128, 256 })
            }
            $crate::ProtocolRingDispatchTierId::Fp64 => {
                $crate::__dispatch_ring_dim_arms!($d, $D, $body, { 32, 64, 128, 256 })
            }
            $crate::ProtocolRingDispatchTierId::Fp32 => {
                $crate::__dispatch_ring_dim_arms!($d, $D, $body, { 64, 128, 256 })
            }
        }
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __dispatch_for_field_envelope {
    ($F:ty, $d:expr, |$D:ident| $body:expr) => {{
        match $crate::protocol_dispatch_tier::<$F>() {
            $crate::ProtocolRingDispatchTierId::Fp128 => {
                $crate::__dispatch_ring_dim_arms!($d, $D, $body, { 64, 128, 256 })
            }
            $crate::ProtocolRingDispatchTierId::Fp64 => {
                $crate::__dispatch_ring_dim_arms!($d, $D, $body, { 32, 64, 128, 256 })
            }
            $crate::ProtocolRingDispatchTierId::Fp32 => {
                $crate::__dispatch_ring_dim_arms!($d, $D, $body, { 64, 128, 256 })
            }
        }
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __dispatch_for_field_ntt {
    ($F:ty, $d:expr, |$D:ident| $body:expr) => {{
        match $crate::protocol_dispatch_tier::<$F>() {
            $crate::ProtocolRingDispatchTierId::Fp128 => {
                $crate::__dispatch_ring_dim_arms!($d, $D, $body, { 16, 32, 64, 128, 256, 512 })
            }
            $crate::ProtocolRingDispatchTierId::Fp64 => {
                $crate::__dispatch_ring_dim_arms!($d, $D, $body, { 32, 64, 128, 256, 512, 1024 })
            }
            $crate::ProtocolRingDispatchTierId::Fp32 => {
                $crate::__dispatch_ring_dim_arms!($d, $D, $body, { 64, 128, 256, 512, 1024, 2048 })
            }
        }
    }};
}

#[macro_export]
macro_rules! dispatch_for_field {
    ($crate::ProtocolDispatchSlot::Role($crate::RingRole::Inner), $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__dispatch_for_field_inner!($F, $d, |$D| $body)
    };
    (ProtocolDispatchSlot::Role(RingRole::Inner), $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__dispatch_for_field_inner!($F, $d, |$D| $body)
    };
    (akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner), $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__dispatch_for_field_inner!($F, $d, |$D| $body)
    };

    ($crate::ProtocolDispatchSlot::Role($crate::RingRole::Outer), $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__dispatch_for_field_outer!($F, $d, |$D| $body)
    };
    (ProtocolDispatchSlot::Role(RingRole::Outer), $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__dispatch_for_field_outer!($F, $d, |$D| $body)
    };
    (akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Outer), $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__dispatch_for_field_outer!($F, $d, |$D| $body)
    };

    ($crate::ProtocolDispatchSlot::Role($crate::RingRole::Opening), $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__dispatch_for_field_opening!($F, $d, |$D| $body)
    };
    (ProtocolDispatchSlot::Role(RingRole::Opening), $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__dispatch_for_field_opening!($F, $d, |$D| $body)
    };
    (akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Opening), $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__dispatch_for_field_opening!($F, $d, |$D| $body)
    };

    ($crate::ProtocolDispatchSlot::Envelope, $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__dispatch_for_field_envelope!($F, $d, |$D| $body)
    };
    (ProtocolDispatchSlot::Envelope, $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__dispatch_for_field_envelope!($F, $d, |$D| $body)
    };
    (akita_types::ProtocolDispatchSlot::Envelope, $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__dispatch_for_field_envelope!($F, $d, |$D| $body)
    };

    ($crate::ProtocolDispatchSlot::Ntt, $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__dispatch_for_field_ntt!($F, $d, |$D| $body)
    };
    (ProtocolDispatchSlot::Ntt, $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__dispatch_for_field_ntt!($F, $d, |$D| $body)
    };
    (akita_types::ProtocolDispatchSlot::Ntt, $F:ty, $d:expr, |$D:ident| $body:expr) => {
        $crate::__dispatch_for_field_ntt!($F, $d, |$D| $body)
    };
}

protocol_dispatch_policy! {
    Fp128: {
        inner: [64, 128]
        outer: [16, 32, 64, 128, 256]
        opening: [16, 32, 64, 128, 256]
        envelope: [64, 128, 256]
        ntt: [16, 32, 64, 128, 256, 512]
        min_bd: 16
        ntt_max: 512
    }
    Fp64: {
        inner: [64, 128, 256]
        outer: [32, 64, 128, 256]
        opening: [32, 64, 128, 256]
        envelope: [32, 64, 128, 256]
        ntt: [32, 64, 128, 256, 512, 1024]
        min_bd: 32
        ntt_max: 1024
    }
    Fp32: {
        inner: [64, 128, 256]
        outer: [64, 128, 256]
        opening: [64, 128, 256]
        envelope: [64, 128, 256]
        ntt: [64, 128, 256, 512, 1024, 2048]
        min_bd: 64
        ntt_max: 2048
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::SUPPORTED_RING_DIMS;

    #[test]
    fn supported_ring_dims_matches_policy_union() {
        assert_eq!(ALL_PROTOCOL_DISPATCH_ARMS, SUPPORTED_RING_DIMS);
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
            assert_eq!(arms[0], outer_opening_min_ring_d(tier));
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
                ProtocolDispatchSlot::Envelope,
                ProtocolDispatchSlot::Ntt,
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
