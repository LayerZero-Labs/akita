//! Negacyclic NTTs over CRT residue rings on the device.
//!
//! [`DeviceCrtNtt`] holds one [`CrtNttParamSet`]'s primes and transform tables
//! in device memory. Its transforms run over buffers of ring elements in the
//! [`CyclotomicCrtNtt`](akita_algebra::CyclotomicCrtNtt) layout (element,
//! then prime, then coefficient, as raw Montgomery `i32` words) and match the
//! CPU's scalar transforms word for word.

use std::time::Duration;

use akita_algebra::{CrtNttParamSet, MontCoeff, NttPrime};
use bytemuck::{Pod, Zeroable};
use jolt_metal::runtime::{Batch, Binding, DeviceBuffer, Grid};

use crate::error::AkitaMetalError;
use crate::library::{ring_degree_kernel, AkitaMetal, Instance, RING_DEGREES};

/// Transform templates, instantiated at every ring degree.
const TEMPLATES: [&str; 2] = ["akita_ntt_forward", "akita_ntt_inverse"];

pub(crate) fn instances() -> Vec<Instance> {
    TEMPLATES
        .iter()
        .flat_map(|&template| {
            RING_DEGREES.iter().map(move |&ring_degree| Instance {
                template,
                args: ring_degree.to_string(),
                host_name: ring_degree_kernel(template, ring_degree),
            })
        })
        .collect()
}

/// Layout of `akita::NttPrime` in `shaders/akita/mont.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct DevicePrime {
    p: i32,
    pinv: i32,
    mont: i32,
    montsq: i32,
}

impl From<NttPrime<i32>> for DevicePrime {
    fn from(prime: NttPrime<i32>) -> Self {
        Self {
            p: prime.p,
            pinv: prime.pinv,
            mont: prime.mont,
            montsq: prime.montsq,
        }
    }
}

/// Layout of `akita::NttBatch` in `shaders/akita/ntt.metal`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct NttBatch {
    polys: u32,
    primes: u32,
}

/// The primes and negacyclic transform tables of one `CrtNttParamSet<i32, K, D>`
/// in device memory.
pub struct DeviceCrtNtt<const K: usize, const D: usize> {
    pub(crate) primes: DeviceBuffer<DevicePrime>,
    /// `[K][7][D]` words: per prime the rows of `akita::NttTables`
    /// (`NttTwiddles::negacyclic_tables`, `psi^i R^2`, and the canonical
    /// forward twiddles with their Shoup quotients).
    pub(crate) tables: DeviceBuffer<i32>,
    /// `[K][K]` words: the Garner inverse `p_j^-1 mod p_i` in prime `i`'s
    /// Montgomery form at `[i][j]` for `j < i`, zero elsewhere.
    pub(crate) gamma: DeviceBuffer<i32>,
    /// The primes, for field-side constants.
    pub(crate) moduli: [NttPrime<i32>; K],
}

impl<const K: usize, const D: usize> DeviceCrtNtt<K, D> {
    /// Uploads `params`' primes and tables.
    pub fn new(
        metal: &AkitaMetal,
        params: &CrtNttParamSet<i32, K, D>,
    ) -> Result<Self, AkitaMetalError> {
        if !RING_DEGREES.contains(&D) {
            return Err(AkitaMetalError::Shape(format!(
                "ring degree {D} has no transform kernel"
            )));
        }
        let primes = params.primes.map(DevicePrime::from);
        let tables = params
            .primes
            .iter()
            .zip(params.twiddles.iter())
            .flat_map(|(prime, twiddles)| {
                let [fwd, inv, psi, untwist] = twiddles.negacyclic_tables();
                // psi^i R^2: mont_mul(psi^i R, R^2) = psi^i R^2 (akita::NttTables::PSI_R2).
                let montsq = MontCoeff::from_raw(prime.montsq);
                let psi_r2 = psi.map(|twist| prime.mul(twist, montsq).raw());
                // Harvey's butterflies take canonical twiddles w and their
                // Shoup quotients floor(w 2^32 / p) (NttTables::FWD_*).
                let canonical = fwd.map(|twiddle| prime.to_canonical(twiddle));
                let shoup = canonical.map(|twiddle| {
                    let quotient = (u64::from(twiddle.unsigned_abs()) << 32)
                        .checked_div(u64::from(prime.p.unsigned_abs()))
                        .unwrap_or_default();
                    // The quotient is below 2^32 because w < p; store its bits.
                    u32::try_from(quotient).unwrap_or_default() as i32
                });
                [
                    fwd.map(MontCoeff::raw),
                    inv.map(MontCoeff::raw),
                    psi.map(MontCoeff::raw),
                    untwist.map(MontCoeff::raw),
                    psi_r2,
                    canonical,
                    shoup,
                ]
            })
            .flatten()
            .collect::<Vec<_>>();
        let gamma = params
            .primes
            .iter()
            .zip(&params.garner.gamma)
            .flat_map(|(prime, row)| {
                row.iter().map(|&inverse| {
                    // Garner inverses are canonical residues below p < 2^30.
                    let canonical = i32::try_from(inverse).map_err(|_| {
                        AkitaMetalError::Shape(format!("Garner inverse {inverse} exceeds i32"))
                    })?;
                    Ok(prime.from_canonical(canonical).raw())
                })
            })
            .collect::<Result<Vec<_>, AkitaMetalError>>()?;
        Ok(Self {
            primes: DeviceBuffer::from_slice(metal.device(), &primes)?,
            tables: DeviceBuffer::from_slice(metal.device(), &tables)?,
            gamma: DeviceBuffer::from_slice(metal.device(), &gamma)?,
            moduli: params.primes,
        })
    }

    /// Forward-transforms every ring element of `data` in place and returns
    /// the GPU time.
    pub fn forward(
        &self,
        metal: &AkitaMetal,
        data: &mut DeviceBuffer<i32>,
    ) -> Result<Duration, AkitaMetalError> {
        self.transform(metal, "akita_ntt_forward", data)
    }

    /// Inverse-transforms every ring element of `data` in place and returns
    /// the GPU time.
    pub fn inverse(
        &self,
        metal: &AkitaMetal,
        data: &mut DeviceBuffer<i32>,
    ) -> Result<Duration, AkitaMetalError> {
        self.transform(metal, "akita_ntt_inverse", data)
    }

    fn transform(
        &self,
        metal: &AkitaMetal,
        template: &str,
        data: &mut DeviceBuffer<i32>,
    ) -> Result<Duration, AkitaMetalError> {
        let (elements, rows) = rows::<K, D>(data.len())?;
        let batch_shape = NttBatch {
            polys: u32::try_from(elements).map_err(|_| shape_overflow(elements))?,
            primes: u32::try_from(K).map_err(|_| shape_overflow(K))?,
        };
        let lanes = D / 2;
        let threads = rows
            .checked_mul(lanes)
            .ok_or_else(|| shape_overflow(rows))?;
        let pipeline = metal.pipeline(&ring_degree_kernel(template, D))?;
        let mut batch = Batch::new(metal.device())?;
        batch.dispatch(
            pipeline,
            &[
                Binding::buffer(data),
                Binding::buffer(&self.primes),
                Binding::buffer(&self.tables),
                Binding::value(&batch_shape),
            ],
            Grid::linear(threads, lanes),
        )?;
        Ok(batch.commit_and_wait()?)
    }
}

/// The number of ring elements and of `(element, prime)` rows in a buffer
/// of `len` words.
pub(crate) fn rows<const K: usize, const D: usize>(
    len: usize,
) -> Result<(usize, usize), AkitaMetalError> {
    let element = K.checked_mul(D).filter(|&words| words > 0);
    element
        .and_then(|words| akita_error::checked::exact_div(len, words))
        .and_then(|elements| Some((elements, elements.checked_mul(K)?)))
        .ok_or_else(|| {
            AkitaMetalError::Shape(format!(
                "{len} words is not a whole number of {K} x {D} residue rings"
            ))
        })
}

pub(crate) fn shape_overflow(value: usize) -> AkitaMetalError {
    AkitaMetalError::Shape(format!("{value} exceeds the 32-bit dispatch range"))
}

#[cfg(test)]
mod tests {
    use super::rows;

    #[test]
    fn accepted_shapes_can_cross_the_32_bit_word_boundary() {
        const D: usize = 64;

        let first_wide_row = usize::try_from(u32::MAX).expect("usize holds u32") / D + 1;
        let len = (first_wide_row + 1) * D;
        let (elements, row_count) = rows::<1, D>(len).expect("whole ring buffer");

        assert_eq!(elements, first_wide_row + 1);
        assert_eq!(row_count, first_wide_row + 1);
        assert_eq!(first_wide_row * D, 1usize << 32);
    }

    #[test]
    fn shader_promotes_global_bases_before_multiplication() {
        const SOURCE: &str = include_str!("../shaders/akita/ntt.metal");

        assert_eq!(SOURCE.matches("data + ulong(row) * D").count(), 2);
        assert_eq!(
            SOURCE
                .matches("tables + ulong(prime) * NttTables::COUNT * D")
                .count(),
            2
        );
    }
}
