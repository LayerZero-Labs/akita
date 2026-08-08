//! Host kernel selection for one prepared CRT+NTT parameter set.

use super::prime::PrimeWidth;

/// Host kernels selected once when a CRT+NTT parameter set is prepared.
///
/// AVX2 is the measured x86 production backend for both transforms and
/// pointwise arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NttKernelPlan(NttKernelKind);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)] // Variants are selected on different target architectures.
enum NttKernelKind {
    /// Portable scalar kernels.
    Scalar,
    /// AArch64 NEON transforms and pointwise arithmetic.
    Neon,
    /// AVX2 transforms and pointwise arithmetic.
    Avx2,
}

impl NttKernelPlan {
    pub(crate) const SCALAR: Self = Self(NttKernelKind::Scalar);

    /// Detect the best enabled host plan for residue width `W`.
    pub fn detect<W: PrimeWidth>() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if matches!(core::mem::size_of::<W>(), 2 | 4) && super::avx::use_avx2_transform_ntt() {
                return Self(NttKernelKind::Avx2);
            }
        }

        #[cfg(target_arch = "aarch64")]
        if super::neon::use_neon_ntt() {
            return Self(NttKernelKind::Neon);
        }

        Self::SCALAR
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub(crate) const fn uses_x86_transform(self) -> bool {
        matches!(self.0, NttKernelKind::Avx2)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub(crate) const fn uses_avx2_i32_dot(self) -> bool {
        matches!(self.0, NttKernelKind::Avx2)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub(crate) const fn x86_pointwise_mode(self) -> Option<super::avx::AvxNttMode> {
        match self.0 {
            NttKernelKind::Avx2 => Some(super::avx::AvxNttMode::Avx2),
            NttKernelKind::Scalar | NttKernelKind::Neon => None,
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub(crate) const fn uses_neon(self) -> bool {
        matches!(self.0, NttKernelKind::Neon)
    }
}
