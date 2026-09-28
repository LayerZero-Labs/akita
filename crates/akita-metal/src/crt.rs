//! CRT reconstruction of residue rings into field coefficients on the device.

use std::time::Duration;

use bytemuck::{Pod, Zeroable};
use jolt_metal::runtime::{Batch, Binding, DeviceBuffer, Grid};
use jolt_metal::MetalField;

use crate::error::AkitaMetalError;
use akita_algebra::tables::{Q128_NUM_PRIMES, Q64_NUM_PRIMES};
use jolt_field::{Prime128OffsetA7F7, Prime64Offset59};
use jolt_metal::runtime::MslType;

use crate::library::{AkitaMetal, Instance};
use crate::ntt::{rows, shape_overflow, DeviceCrtNtt};

/// The host name of the CRT reconstruction into `F` from `primes` residues.
pub(crate) fn crt_kernel<F: MslType>(primes: usize) -> String {
    format!("akita_crt_reconstruct_k{primes}_{}", F::HOST_SUFFIX)
}

fn crt_instance<F: MslType>(primes: usize) -> Instance {
    Instance {
        template: "akita_crt_reconstruct",
        args: format!("{}, {primes}", F::MSL_NAME),
        host_name: crt_kernel::<F>(primes),
    }
}

/// Each field preset reconstructs from any prefix of its own CRT profile:
/// limb-split matrices (`DeviceNttMatrix::from_rings`) use fewer primes.
pub(crate) fn instances() -> Vec<Instance> {
    (1..=Q128_NUM_PRIMES)
        .map(crt_instance::<Prime128OffsetA7F7>)
        .chain((1..=Q64_NUM_PRIMES).map(crt_instance::<Prime64Offset59>))
        .collect()
}

/// Threads per threadgroup for one-coefficient-per-thread kernels.
pub(crate) const COEFFICIENT_GROUP: usize = 256;

/// Layout of `akita::CrtBatch` in `shaders/akita/crt.metal`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct CrtBatch {
    pub(crate) coefficients: u32,
    pub(crate) log_degree: u32,
    /// Nonzero: add into the output (a later CRT segment).
    pub(crate) accumulate: u32,
    /// Limb groups to reconstruct and combine with the scales buffer.
    pub(crate) limbs: u32,
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
            accumulate: 0,
            limbs: 1,
        };
        let radix = DeviceBuffer::from_slice(metal.device(), &self.crt_weights::<F>())?;
        let scales = DeviceBuffer::from_slice(metal.device(), &[F::one()])?;
        let pipeline = metal.pipeline(&crt_kernel::<F>(K))?;
        let mut batch = Batch::new(metal.device())?;
        batch.dispatch(
            pipeline,
            &[
                Binding::buffer(residues),
                Binding::buffer(out),
                Binding::buffer(&self.primes),
                Binding::buffer(&self.gamma),
                Binding::buffer(&radix),
                Binding::buffer(&scales),
                Binding::value(&shape),
            ],
            Grid::linear(coefficients, COEFFICIENT_GROUP),
        )?;
        Ok(batch.commit_and_wait()?)
    }
}
