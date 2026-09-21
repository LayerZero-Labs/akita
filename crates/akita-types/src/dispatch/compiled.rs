//! Executable capabilities, deliberately separate from protocol admissibility.

use super::ProtocolDispatchSlot;
use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, Field};

// This table describes the application build, not the protocol's SIS policy.
// Keep both compression ladder degrees even though commitment roles use 128.
#[cfg(feature = "dispatch-aerie")]
#[doc(hidden)]
#[macro_export]
macro_rules! __aerie_dispatch_dimensions {
    (inner, $apply:ident, $($args:tt)*) => { $crate::$apply!([128], $($args)*) };
    (outer, $apply:ident, $($args:tt)*) => { $crate::$apply!([128], $($args)*) };
    (opening, $apply:ident, $($args:tt)*) => { $crate::$apply!([128], $($args)*) };
    (ntt, $apply:ident, $($args:tt)*) => { $crate::$apply!([128], $($args)*) };
    (compression, $apply:ident, $($args:tt)*) => { $crate::$apply!([16, 32], $($args)*) };
}

#[cfg(feature = "dispatch-aerie")]
#[doc(hidden)]
#[macro_export]
macro_rules! __aerie_dispatch_checked {
    ([$($dim:literal),+], $F:ty, $d:expr, |$D:ident| $body:expr) => {{
        let __d = $d;
        if $crate::protocol_dispatch_tier::<$F>() != $crate::ProtocolRingDispatchTierId::Fp64 {
            Err(akita_error::AkitaError::InvalidSetup(
                "dispatch-aerie executable requires the Fp64 field tier".into(),
            ))
        } else {
            match __d {
                $($dim => { const $D: usize = $dim; $body },)+
                _ => Err(akita_error::AkitaError::InvalidSetup(format!(
                    "ring dimension {__d} is unavailable for this slot in the dispatch-aerie executable",
                ))),
            }
        }
    }};
}

// cfg selects this definition in akita-types, not in the downstream caller.
// The canonical protocol table is consumed but never narrowed or rewritten.
#[cfg(feature = "dispatch-aerie")]
#[doc(hidden)]
#[macro_export]
macro_rules! __dispatch_protocol_policy_slot {
    ($slot:ident, $F:ty, $d:expr, |$D:ident| $body:expr; $($policy:tt)*) => {
        $crate::__aerie_dispatch_dimensions!($slot, __aerie_dispatch_checked, $F, $d, |$D| $body)
    };
}

/// Check whether the selected executable can dispatch this field/role/degree.
///
/// This is additional to protocol and catalog validation, not a substitute for
/// it. It does not validate an entire schedule, source contract, or field
/// modulus. With `dispatch-aerie`, every foreign group and schedule level must
/// fit the same compiled domain as local groups.
///
/// # Errors
/// Returns an error if the request is outside the compiled dispatch domain.
pub fn validate_compiled_dispatch<F: Field + CanonicalEncoding>(
    slot: ProtocolDispatchSlot,
    dimension: usize,
) -> Result<(), AkitaError> {
    #[cfg(not(feature = "dispatch-aerie"))]
    {
        if super::slot_dim_supported_for_tier(super::protocol_dispatch_tier::<F>(), slot, dimension)
        {
            Ok(())
        } else {
            Err(AkitaError::InvalidSetup(
                "request is outside the compiled dispatch domain".into(),
            ))
        }
    }
    #[cfg(feature = "dispatch-aerie")]
    {
        use super::RingRole;
        match slot {
            ProtocolDispatchSlot::Role(RingRole::Inner) => crate::__aerie_dispatch_dimensions!(
                inner,
                __aerie_dispatch_checked,
                F,
                dimension,
                |D| {
                    let _ = D;
                    Ok(())
                }
            ),
            ProtocolDispatchSlot::Role(RingRole::Outer) => crate::__aerie_dispatch_dimensions!(
                outer,
                __aerie_dispatch_checked,
                F,
                dimension,
                |D| {
                    let _ = D;
                    Ok(())
                }
            ),
            ProtocolDispatchSlot::Role(RingRole::Opening) => crate::__aerie_dispatch_dimensions!(
                opening,
                __aerie_dispatch_checked,
                F,
                dimension,
                |D| {
                    let _ = D;
                    Ok(())
                }
            ),
            ProtocolDispatchSlot::Ntt => crate::__aerie_dispatch_dimensions!(
                ntt,
                __aerie_dispatch_checked,
                F,
                dimension,
                |D| {
                    let _ = D;
                    Ok(())
                }
            ),
            ProtocolDispatchSlot::Compression => crate::__aerie_dispatch_dimensions!(
                compression,
                __aerie_dispatch_checked,
                F,
                dimension,
                |D| {
                    let _ = D;
                    Ok(())
                }
            ),
        }
    }
}
