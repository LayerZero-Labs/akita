//! CRT reconstruction of residue rings into field coefficients on the device.

use std::time::Duration;

use bytemuck::{Pod, Zeroable};
use jolt_metal::runtime::{Batch, Binding, DeviceBuffer, Grid};
use jolt_metal::MetalField;

use crate::error::AkitaMetalError;
use crate::library::{crt_kernel, AkitaMetal};
use crate::ntt::{rows, shape_overflow, DeviceCrtNtt};

/// Threads per threadgroup for one-coefficient-per-thread kernels.
const COEFFICIENT_GROUP: usize = 256;

/// Layout of `akita::CrtBatch` in `shaders/akita/crt.metal`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CrtBatch {
    coefficients: u32,
    log_degree: u32,
}

impl<const K: usize, const D: usize> DeviceCrtNtt<K, D> {
    /// `prod_{j<i} p_j` in `F` for each prime `i`: the mixed-radix weights of
    /// Garner's digits.
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "field multiplication is modular"
    )]
    pub(crate) fn crt_weights<F: MetalField>(&self) -> [F; K] {
        let mut radix = F::one();
        self.moduli.map(|prime| {
            let weight = radix;
            radix *= F::from_u64(prime.p.unsigned_abs().into());
            weight
        })
    }

    /// Reconstructs every coefficient of the residue rings in `residues`
    /// (raw Montgomery words in `(-p, p)`, as the inverse transform leaves
    /// them) into `out`, one field element per coefficient, and returns the
    /// GPU time. Matches `CyclotomicCrtNtt::to_ring` after the inverse NTT.
    pub fn reconstruct<F: MetalField>(
        &self,
        metal: &AkitaMetal,
        residues: &DeviceBuffer<i32>,
        out: &mut DeviceBuffer<F>,
    ) -> Result<Duration, AkitaMetalError> {
        let pipeline = metal.pipeline(&crt_kernel::<F>(K))?;
        let (elements, _) = rows::<K, D>(residues.len())?;
        let coefficients = elements
            .checked_mul(D)
            .ok_or_else(|| shape_overflow(elements))?;
        if out.len() != coefficients {
            return Err(AkitaMetalError::Shape(format!(
                "{coefficients} coefficients need an output of that length, got {}",
                out.len()
            )));
        }
        let shape = CrtBatch {
            coefficients: u32::try_from(coefficients).map_err(|_| shape_overflow(coefficients))?,
            log_degree: D.trailing_zeros(),
        };
        let radix = DeviceBuffer::from_slice(metal.device(), &self.crt_weights::<F>())?;
        let mut batch = Batch::new(metal.device())?;
        batch.dispatch(
            pipeline,
            &[
                Binding::buffer(residues),
                Binding::buffer(out),
                Binding::buffer(&self.primes),
                Binding::buffer(&self.gamma),
                Binding::buffer(&radix),
                Binding::value(&shape),
            ],
            Grid::linear(coefficients, COEFFICIENT_GROUP),
        )?;
        Ok(batch.commit_and_wait()?)
    }
}
