//! Host kernel selection for one prepared CRT+NTT parameter set.

use super::prime::PrimeWidth;

/// Host kernels selected once when a CRT+NTT parameter set is prepared.
///
/// The mixed x86 plan reflects measured Ice Lake behavior: AVX2 wins for the
/// transform stages while AVX-512 wins for pointwise arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NttKernelPlan {
    /// Portable scalar kernels.
    Scalar,
    /// AArch64 NEON transforms and pointwise arithmetic.
    Neon,
    /// AVX2 transforms and pointwise arithmetic.
    Avx2,
    /// AVX2 transforms with AVX-512 pointwise arithmetic.
    Avx2TransformAvx512Pointwise,
    /// AVX-512 transforms and pointwise arithmetic.
    Avx512,
}

impl NttKernelPlan {
    /// Detect the best enabled host plan for residue width `W`.
    pub fn detect<W: PrimeWidth>() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            use super::avx::AvxNttMode;

            if core::mem::size_of::<W>() == core::mem::size_of::<i16>() {
                return if super::avx::use_avx2_transform_ntt() {
                    Self::Avx2
                } else {
                    Self::Scalar
                };
            }
            if core::mem::size_of::<W>() == core::mem::size_of::<i32>() {
                return match super::avx::avx_ntt_mode() {
                    Some(AvxNttMode::Avx512) if super::avx::use_avx512_transform_ntt() => {
                        Self::Avx512
                    }
                    Some(AvxNttMode::Avx512) => Self::Avx2TransformAvx512Pointwise,
                    Some(AvxNttMode::Avx2) => Self::Avx2,
                    None => Self::Scalar,
                };
            }
        }

        #[cfg(target_arch = "aarch64")]
        if super::neon::use_neon_ntt() {
            return Self::Neon;
        }

        Self::Scalar
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub(crate) const fn uses_x86_transform(self) -> bool {
        matches!(
            self,
            Self::Avx2 | Self::Avx2TransformAvx512Pointwise | Self::Avx512
        )
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub(crate) const fn uses_avx512_transform(self) -> bool {
        matches!(self, Self::Avx512)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub(crate) const fn x86_pointwise_mode(self) -> Option<super::avx::AvxNttMode> {
        match self {
            Self::Avx2 => Some(super::avx::AvxNttMode::Avx2),
            Self::Avx2TransformAvx512Pointwise | Self::Avx512 => {
                Some(super::avx::AvxNttMode::Avx512)
            }
            Self::Scalar | Self::Neon => None,
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub(crate) const fn uses_neon(self) -> bool {
        matches!(self, Self::Neon)
    }
}
