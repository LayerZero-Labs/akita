// AUTO-GENERATED projection of scalar ADPS16 quantum cutoffs for diagnostic
// compressed commitments.
//
// Runtime entries are limited to the six F/H cells exercised by the 16 KiB
// negative-binary shadow path. Widths are indexed by `rank - 1`.

use super::{SisModulusProfileId, SisSecurityPolicyId};

#[rustfmt::skip]
pub(super) fn sis_max_widths(
    policy: SisSecurityPolicyId,
    profile: SisModulusProfileId,
    d: u32,
    coeff_linf_bound: u128,
) -> Option<&'static [u64]> {
    if policy != SisSecurityPolicyId::Quantum128BitADPS16 || coeff_linf_bound != 1 {
        return None;
    }
    match (profile, d) {
        (SisModulusProfileId::Q128OffsetA7F7, 8) => Some(&[508, 512]),
        (SisModulusProfileId::Q128OffsetA7F7, 16) => Some(&[7077, 8192]),
        (SisModulusProfileId::Q64Offset59, 16) => Some(&[254, 256]),
        (SisModulusProfileId::Q64Offset59, 32) => Some(&[3538, 4096]),
        (SisModulusProfileId::Q32Offset99, 32) => Some(&[127, 128]),
        (SisModulusProfileId::Q32Offset99, 64) => Some(&[1769, 2048]),
        _ => None,
    }
}
