//! Field coefficients to balanced digit planes on the device.

use std::time::Duration;

use akita_algebra::ring::cyclotomic::decompose_centering_threshold;
use akita_algebra::CanonicalEncoding;
use akita_error::checked;
use bytemuck::{Pod, Zeroable};
use jolt_field::{Prime128OffsetA7F7, Prime64Offset59};
use jolt_metal::runtime::{Batch, Binding, DeviceBuffer, Grid, MslType};
use jolt_metal::MetalField;

use crate::crt::COEFFICIENT_GROUP;
use crate::digits::{DeviceDigitPlanes, DigitPlane};
use crate::error::AkitaMetalError;
use crate::library::{AkitaMetal, Instance};
use crate::matvec::field_modulus;
use crate::ntt::shape_overflow;

fn decompose_kernel<F: MslType, T: DigitPlane>() -> String {
    format!("akita_decompose_{}_{}", F::HOST_SUFFIX, T::SUFFIX)
}

fn decompose_instance<F: MslType, T: DigitPlane>() -> Instance {
    Instance {
        template: "akita_decompose",
        args: format!("{}, {}", F::MSL_NAME, T::MSL_NAME),
        host_name: decompose_kernel::<F, T>(),
    }
}

pub(crate) fn instances() -> Vec<Instance> {
    vec![
        decompose_instance::<Prime128OffsetA7F7, i8>(),
        decompose_instance::<Prime128OffsetA7F7, i16>(),
        decompose_instance::<Prime64Offset59, i8>(),
        decompose_instance::<Prime64Offset59, i16>(),
    ]
}

/// Layout of `akita::DecomposeShape` in `shaders/akita/decompose.metal`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct DecomposeShape {
    modulus: [u32; 4],
    threshold: [u32; 4],
    coefficients: u32,
    log_degree: u32,
    levels: u32,
    log_basis: u32,
}

/// `value` as four little-endian 32-bit words.
fn words(value: u128) -> [u32; 4] {
    let chunks: [[u8; 4]; 4] = bytemuck::cast(value.to_le_bytes());
    chunks.map(u32::from_le_bytes)
}

/// Decomposes rings of `ring_degree` coefficients into `levels` balanced
/// base-`2^log_basis` digit planes each, as `decompose_rows_i8_into` and
/// `CyclotomicRing::balanced_decompose_pow2_i16_into` do, and returns the GPU
/// time. The returned planes carry their validated basis into matvec.
///
/// The result holds `levels` planes per ring, ring-major: exactly the digit
/// planes [`DeviceNttMatrix::mat_vec`](crate::matvec::DeviceNttMatrix::mat_vec)
/// takes, with `cols = rings * levels` per block. Decomposing a ring of a
/// larger degree as rings of `ring_degree` gives the subcolumn order of
/// `decompose_commit_blocks_into`.
pub fn decompose<F, T>(
    metal: &AkitaMetal,
    coefficients: &DeviceBuffer<F>,
    ring_degree: usize,
    levels: usize,
    log_basis: u32,
) -> Result<(DeviceDigitPlanes<T>, Duration), AkitaMetalError>
where
    F: MetalField + CanonicalEncoding,
    T: DigitPlane,
{
    if !ring_degree.is_power_of_two() {
        return Err(AkitaMetalError::Shape(format!(
            "ring degree {ring_degree} is not a power of two"
        )));
    }
    // The CPU's limit (BalancedDecomposePow2Params::new).
    let digit_bits = checked::product([levels, log_basis as usize]);
    if digit_bits.is_none_or(|bits| bits > checked::sum([128, log_basis as usize]).unwrap_or(0)) {
        return Err(AkitaMetalError::Shape(format!(
            "{levels} digits of {log_basis} bits exceed a 128-bit field"
        )));
    }
    let count = coefficients.len();
    if checked::exact_div(count, ring_degree).is_none() {
        return Err(AkitaMetalError::Shape(format!(
            "{count} coefficients is not a whole number of degree-{ring_degree} rings"
        )));
    }
    let digits = checked::product([count, levels]).ok_or_else(|| shape_overflow(count))?;
    let pipeline = metal.pipeline(&decompose_kernel::<F, T>())?;
    if digits == 0 {
        return Ok((
            DeviceDigitPlanes::zeroed(metal, digits, log_basis)?,
            Duration::ZERO,
        ));
    }
    DeviceDigitPlanes::<T>::validate_log_basis(log_basis)?;
    let modulus = field_modulus::<F>()?;
    let u32_of = |value: usize| u32::try_from(value).map_err(|_| shape_overflow(value));
    u32_of(ring_degree)?;
    let shape = DecomposeShape {
        modulus: words(modulus),
        threshold: words(decompose_centering_threshold(levels, log_basis, modulus)),
        coefficients: u32_of(count)?,
        log_degree: ring_degree.trailing_zeros(),
        levels: u32_of(levels)?,
        log_basis,
    };
    let mut out = DeviceDigitPlanes::zeroed(metal, digits, log_basis)?;
    let mut batch = Batch::new(metal.device())?;
    batch.dispatch(
        pipeline,
        &[
            Binding::buffer(coefficients),
            Binding::buffer(out.buffer_mut()),
            Binding::value(&shape),
        ],
        Grid::linear(count, COEFFICIENT_GROUP),
    )?;
    let duration = batch.commit_and_wait()?;
    Ok((out, duration))
}

#[cfg(test)]
mod tests {
    #[test]
    fn output_address_crosses_u32_at_admitted_shape() {
        let index = (1u64 << 25) as u32;
        let degree = 64u64;
        let levels = 128u64;
        let ring = u64::from(index) / degree;
        let output = ring
            .checked_mul(levels)
            .and_then(|value| value.checked_mul(degree))
            .expect("decomposition output address");
        assert_eq!(output, 1u64 << 32);
        assert!(output > u64::from(u32::MAX));
    }
}
