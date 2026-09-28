//! Negacyclic NTTs over CRT residue rings on the device.
//!
//! [`DeviceCrtNtt`] holds one [`CrtNttParamSet`]'s primes and transform tables
//! in device memory. Its transforms run over buffers of ring elements in the
//! [`CyclotomicCrtNtt`](akita_algebra::CyclotomicCrtNtt) layout (element,
//! then prime, then coefficient, as raw Montgomery `i32` words) and match the
//! CPU's scalar transforms word for word.

use std::time::Duration;

use akita_algebra::{CrtNttParamSet, NttPrime};
use bytemuck::{Pod, Zeroable};
use jolt_metal::runtime::{Batch, Binding, DeviceBuffer, Grid};

use crate::error::AkitaMetalError;
use crate::library::{ring_degree_kernel, AkitaMetal, RING_DEGREES};

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
    /// `[K][4][D]` words: `NttTwiddles::negacyclic_tables` per prime.
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
            .twiddles
            .iter()
            .flat_map(|twiddles| twiddles.negacyclic_tables())
            .flat_map(|table| table.iter().map(|coefficient| coefficient.raw()))
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
