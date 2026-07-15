// AUTO-GENERATED projection of scalar ADPS16 quantum cutoffs into runtime rank tables.
//
// Generator certifies scalar cells (B, n) -> max m. The checked-in runtime artifact
// stores the Module-SIS projection (d, B) -> max ring widths by rank, with
// width[r-1] = cutoff_m(B, n = r * d) / d. Role coverage is validated before lookup.

mod q128;
mod q32;
mod q64;

use super::{SisModulusProfileId, SisSecurityPolicyId};

/// Generated SIS max-width table for the named security policy.
///
/// For each `(d, coeff_linf_bound)` returns the maximum secure ring-element width
/// per module rank (`widths[rank - 1]`), projected from scalar cutoffs.
#[rustfmt::skip]
pub(crate) fn sis_max_widths(
    policy: SisSecurityPolicyId,
    profile: SisModulusProfileId,
    d: u32,
    coeff_linf_bound: u128,
) -> Option<&'static [u64]> {
    if policy != SisSecurityPolicyId::Quantum128BitADPS16 {
        return None;
    }
    match profile {
        SisModulusProfileId::Q32Offset99 => q32::sis_max_widths(d, coeff_linf_bound),
        SisModulusProfileId::Q64Offset59 => q64::sis_max_widths(d, coeff_linf_bound),
        SisModulusProfileId::Q128OffsetA7F7 => q128::sis_max_widths(d, coeff_linf_bound),
    }
}
