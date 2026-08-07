//! Host kernel selection for one prepared CRT+NTT parameter set.

use super::prime::PrimeWidth;

/// Host kernels selected once when a CRT+NTT parameter set is prepared.
///
/// The mixed x86 plan reflects measured Ice Lake behavior: AVX2 wins for the
/// transform stages while AVX-512 wins for pointwise arithmetic.
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
    /// AVX2 transforms with AVX-512 pointwise arithmetic.
    Avx2TransformAvx512Pointwise,
    /// AVX-512 transforms and pointwise arithmetic.
    Avx512,
}

impl NttKernelPlan {
    pub(crate) const SCALAR: Self = Self(NttKernelKind::Scalar);

    /// Detect the best enabled host plan for residue width `W`.
    pub fn detect<W: PrimeWidth>() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            use super::avx::AvxNttMode;

            if core::mem::size_of::<W>() == core::mem::size_of::<i16>() {
                return if super::avx::use_avx512_vnni_pointwise() {
                    Self(NttKernelKind::Avx2TransformAvx512Pointwise)
                } else if super::avx::use_avx2_transform_ntt() {
                    Self(NttKernelKind::Avx2)
                } else {
                    Self::SCALAR
                };
            }
            if core::mem::size_of::<W>() == core::mem::size_of::<i32>() {
                return match super::avx::avx_ntt_mode() {
                    Some(AvxNttMode::Avx512) if super::avx::use_avx512_transform_ntt() => {
                        Self(NttKernelKind::Avx512)
                    }
                    Some(AvxNttMode::Avx512) => Self(NttKernelKind::Avx2TransformAvx512Pointwise),
                    Some(AvxNttMode::Avx2) => Self(NttKernelKind::Avx2),
                    None => Self::SCALAR,
                };
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
        matches!(
            self.0,
            NttKernelKind::Avx2
                | NttKernelKind::Avx2TransformAvx512Pointwise
                | NttKernelKind::Avx512
        )
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub(crate) const fn uses_avx512_transform(self) -> bool {
        matches!(self.0, NttKernelKind::Avx512)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub(crate) const fn uses_avx512_pointwise(self) -> bool {
        matches!(
            self.0,
            NttKernelKind::Avx2TransformAvx512Pointwise | NttKernelKind::Avx512
        )
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub(crate) const fn x86_pointwise_mode(self) -> Option<super::avx::AvxNttMode> {
        match self.0 {
            NttKernelKind::Avx2 => Some(super::avx::AvxNttMode::Avx2),
            NttKernelKind::Avx2TransformAvx512Pointwise | NttKernelKind::Avx512 => {
                Some(super::avx::AvxNttMode::Avx512)
            }
            NttKernelKind::Scalar | NttKernelKind::Neon => None,
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub(crate) const fn uses_neon(self) -> bool {
        matches!(self.0, NttKernelKind::Neon)
    }
}
