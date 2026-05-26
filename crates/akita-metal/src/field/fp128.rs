//! Host-side descriptors for `Fp128` Metal kernels.

use akita_field::fields::Fp128;

use crate::error::{MetalError, MetalResult};

/// Host/device ABI for one canonical `Fp128` value.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Fp128Limb {
    /// Low 64 bits of the canonical representative.
    pub lo: u64,
    /// High 64 bits of the canonical representative.
    pub hi: u64,
}

impl Fp128Limb {
    /// Convert a field element into the host/device ABI limb layout.
    #[must_use]
    pub fn from_field<const P: u128>(value: Fp128<P>) -> Self {
        let [lo, hi] = value.to_limbs();
        Self { lo, hi }
    }

    /// Convert a canonical host/device ABI limb value back into a field element.
    #[must_use]
    pub fn to_field<const P: u128>(self) -> Fp128<P> {
        Fp128::<P>::from_canonical_u128((self.lo as u128) | ((self.hi as u128) << 64))
    }
}

/// Host/device ABI for one `Fp128` vector dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct Fp128KernelParams {
    /// Low limb of `c = 2^128 - p`.
    pub modulus_c: u64,
    /// Number of field elements to process.
    pub len: u32,
    _padding: u32,
}

impl Fp128KernelParams {
    /// Build kernel parameters for `Fp128<P>`.
    pub fn new<const P: u128>(len: usize) -> MetalResult<Self> {
        Fp128VectorPlan::validate_len(len)?;
        Ok(Self {
            modulus_c: Fp128::<P>::C_LO,
            len: len as u32,
            _padding: 0,
        })
    }
}

/// Parameters needed by generic `Fp128<P>` kernels on the Metal side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp128MetalParams {
    /// Modulus `p = 2^128 - c`.
    pub modulus: u128,
    /// Low limb of `c = 2^128 - p`.
    pub modulus_offset: u64,
}

impl Fp128MetalParams {
    /// Materialize the kernel parameters for an Akita `Fp128` modulus.
    #[must_use]
    pub const fn for_modulus<const P: u128>() -> Self {
        Self {
            modulus: P,
            modulus_offset: Fp128::<P>::C_LO,
        }
    }
}

/// Elementwise `Fp128` vector arithmetic kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp128VectorOp {
    /// `out[i] = lhs[i] + rhs[i]`.
    Add,
    /// `out[i] = lhs[i] - rhs[i]`.
    Sub,
    /// `out[i] = lhs[i] * rhs[i]`.
    Mul,
}

#[cfg(target_os = "macos")]
impl Fp128VectorOp {
    pub(crate) const fn metal_function_name(self) -> &'static str {
        match self {
            Self::Add => "fp128_vector_add",
            Self::Sub => "fp128_vector_sub",
            Self::Mul => "fp128_vector_mul",
        }
    }
}

/// A checked host-side plan for one `Fp128` vector dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp128VectorPlan {
    /// Arithmetic operation to dispatch.
    pub op: Fp128VectorOp,
    /// Number of field elements to process.
    pub len: usize,
    /// Field parameters passed to the Metal kernel.
    pub params: Fp128MetalParams,
}

impl Fp128VectorPlan {
    /// Validate the vector length accepted by the current Metal ABI.
    pub fn validate_len(len: usize) -> MetalResult<()> {
        if len == 0 {
            return Err(MetalError::InvalidInput(
                "fp128 vector kernels require a non-empty input",
            ));
        }
        if len > u32::MAX as usize {
            return Err(MetalError::InvalidInput(
                "fp128 vector kernels support at most u32::MAX elements",
            ));
        }
        Ok(())
    }

    /// Build a dispatch plan for `len` elements over `Fp128<P>`.
    pub fn new<const P: u128>(op: Fp128VectorOp, len: usize) -> MetalResult<Self> {
        Self::validate_len(len)?;

        Ok(Self {
            op,
            len,
            params: Fp128MetalParams::for_modulus::<P>(),
        })
    }
}

#[cfg(test)]
mod tests {
    use akita_field::fields::{Prime128Offset2355, Prime128OffsetA7F7};

    use super::*;

    const P_A7F7: u128 = 0xffffffffffffffffffffffff00005809;
    const P_2355: u128 = 0xfffffffffffffffffffffffffffff6cd;

    #[test]
    fn params_match_fp128_modulus_offsets() {
        let default_params = Fp128MetalParams::for_modulus::<P_A7F7>();
        assert_eq!(default_params.modulus, P_A7F7);
        assert_eq!(default_params.modulus_offset, Prime128OffsetA7F7::C_LO);

        let peer_params = Fp128MetalParams::for_modulus::<P_2355>();
        assert_eq!(peer_params.modulus_offset, Prime128Offset2355::C_LO);
    }

    #[test]
    fn vector_plan_rejects_empty_dispatches() {
        let err = Fp128VectorPlan::new::<P_A7F7>(Fp128VectorOp::Add, 0).unwrap_err();
        assert_eq!(
            err,
            MetalError::InvalidInput("fp128 vector kernels require a non-empty input")
        );
    }

    #[test]
    fn vector_plan_carries_operation_length_and_params() {
        let plan = Fp128VectorPlan::new::<P_A7F7>(Fp128VectorOp::Mul, 16).unwrap();

        assert_eq!(plan.op, Fp128VectorOp::Mul);
        assert_eq!(plan.len, 16);
        assert_eq!(plan.params.modulus_offset, Prime128OffsetA7F7::C_LO);
    }

    #[test]
    fn limb_roundtrip_preserves_canonical_value() {
        let value = Prime128OffsetA7F7::from_canonical_u128(0x1234_5678_9abc_def0);
        let limb = Fp128Limb::from_field(value);

        assert_eq!(limb.to_field::<P_A7F7>(), value);
    }

    #[test]
    fn field_and_metal_limb_layouts_match() {
        assert_eq!(
            core::mem::size_of::<Prime128OffsetA7F7>(),
            core::mem::size_of::<Fp128Limb>()
        );
        assert_eq!(
            core::mem::align_of::<Prime128OffsetA7F7>(),
            core::mem::align_of::<Fp128Limb>()
        );
    }

    #[test]
    fn kernel_params_reject_too_many_elements() {
        let err = Fp128KernelParams::new::<P_2355>((u32::MAX as usize) + 1).unwrap_err();
        assert_eq!(
            err,
            MetalError::InvalidInput("fp128 vector kernels support at most u32::MAX elements")
        );
    }
}
