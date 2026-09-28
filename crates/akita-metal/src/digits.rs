//! Validated balanced digit planes in device memory.

use bytemuck::{NoUninit, Zeroable};
use jolt_metal::runtime::DeviceBuffer;

use crate::{AkitaMetal, AkitaMetalError};

mod sealed {
    pub trait Sealed: Copy {
        fn values_in_balanced_range(values: &[Self], log_basis: u32) -> bool;
    }

    impl Sealed for i8 {
        fn values_in_balanced_range(values: &[Self], log_basis: u32) -> bool {
            if log_basis == 8 {
                return true;
            }
            let Some(bound) = 1i16.checked_shl(log_basis.saturating_sub(1)) else {
                return false;
            };
            let Some(lower) = bound.checked_neg() else {
                return false;
            };
            values
                .iter()
                .all(|&value| i16::from(value) >= lower && i16::from(value) < bound)
        }
    }

    impl Sealed for i16 {
        fn values_in_balanced_range(values: &[Self], log_basis: u32) -> bool {
            if log_basis == 16 {
                return true;
            }
            let Some(bound) = 1i16.checked_shl(log_basis.saturating_sub(1)) else {
                return false;
            };
            akita_algebra::i16_values_in_balanced_range(values, bound)
        }
    }
}

/// A signed digit type for digit planes: `i8` for `log_basis <= 8`, `i16`
/// up to 16.
pub trait DigitPlane: sealed::Sealed + NoUninit + Zeroable {
    /// The MSL spelling.
    const MSL_NAME: &'static str;
    /// Host-name suffix of the kernels over this digit type.
    const SUFFIX: &'static str;
    /// The largest `log_basis` whose balanced digits fit.
    const MAX_LOG_BASIS: u32;
}

impl DigitPlane for i8 {
    const MSL_NAME: &'static str = "char";
    const SUFFIX: &'static str = "i8";
    const MAX_LOG_BASIS: u32 = 8;
}

impl DigitPlane for i16 {
    const MSL_NAME: &'static str = "short";
    const SUFFIX: &'static str = "i16";
    const MAX_LOG_BASIS: u32 = 16;
}

/// Balanced base-`2^log_basis` digit planes whose device storage and range
/// contract cannot disagree.
pub struct DeviceDigitPlanes<T> {
    buffer: DeviceBuffer<T>,
    log_basis: u32,
}

impl<T: DigitPlane> DeviceDigitPlanes<T> {
    /// Validates `digits` against the balanced interval
    /// `[-2^(log_basis-1), 2^(log_basis-1))` and uploads them.
    pub fn from_slice(
        metal: &AkitaMetal,
        digits: &[T],
        log_basis: u32,
    ) -> Result<Self, AkitaMetalError> {
        Self::validate_log_basis(log_basis)?;
        if !<T as sealed::Sealed>::values_in_balanced_range(digits, log_basis) {
            let bound = 1u64
                .checked_shl(log_basis.saturating_sub(1))
                .ok_or_else(|| AkitaMetalError::Shape("balanced digit bound overflow".into()))?;
            return Err(AkitaMetalError::Shape(format!(
                "{} digits contain a value outside [-{bound}, {bound}) for log_basis {log_basis}",
                T::SUFFIX
            )));
        }
        Ok(Self {
            buffer: DeviceBuffer::from_slice(metal.device(), digits)?,
            log_basis,
        })
    }

    pub(crate) fn zeroed(
        metal: &AkitaMetal,
        len: usize,
        log_basis: u32,
    ) -> Result<Self, AkitaMetalError> {
        Self::validate_log_basis(log_basis)?;
        Ok(Self {
            buffer: DeviceBuffer::zeroed(metal.device(), len)?,
            log_basis,
        })
    }

    pub(crate) fn validate_log_basis(log_basis: u32) -> Result<(), AkitaMetalError> {
        if !(1..=T::MAX_LOG_BASIS).contains(&log_basis) {
            return Err(AkitaMetalError::Shape(format!(
                "log_basis {log_basis} does not fit {} digits",
                T::SUFFIX
            )));
        }
        Ok(())
    }

    /// Number of stored digits.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether no digits are stored.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// The power-of-two basis validated for every stored digit.
    pub fn log_basis(&self) -> u32 {
        self.log_basis
    }

    pub(crate) fn buffer(&self) -> &DeviceBuffer<T> {
        &self.buffer
    }

    pub(crate) fn buffer_mut(&mut self) -> &mut DeviceBuffer<T> {
        &mut self.buffer
    }

    /// Reads the planes after device work completes.
    pub fn read(&mut self) -> Result<&[T], jolt_metal::MetalError>
    where
        T: bytemuck::CheckedBitPattern,
    {
        self.buffer.read()
    }
}
