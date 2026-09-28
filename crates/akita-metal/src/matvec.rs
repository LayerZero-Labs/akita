//! Prepared ring matrices times digit-plane vectors on the device.
//!
//! [`DeviceNttMatrix`] holds a CPU-prepared negacyclic NTT matrix
//! ([`PreparedNttBaseView`]) with its CRT profile. [`DeviceNttMatrix::mat_vec`]
//! computes `out[b][r] = sum_c A[r][c] x[b][c]` over `Z_q[X]/(X^D + 1)` for
//! blocks `x[b]` of small signed digit planes, as the CPU's
//! `mat_vec_mul_ntt_digits_i8` and `PreparedNttCache::mat_vec_i16` do, and
//! returns the same field coefficients.

use std::time::Duration;

use akita_algebra::{CanonicalEncoding, CrtCapacity};
use akita_error::checked;
use akita_types::ntt_cache::PreparedNttBaseView;
use bytemuck::{NoUninit, Pod, Zeroable};
use jolt_metal::runtime::{Batch, Binding, DeviceBuffer, Grid};
use jolt_metal::MetalField;

use crate::crt::{crt_kernel, CrtBatch, COEFFICIENT_GROUP};
use crate::error::AkitaMetalError;
use crate::library::{ring_degree_kernel, AkitaMetal, Instance, RING_DEGREES};
use crate::ntt::{shape_overflow, DeviceCrtNtt};

/// Blocks per partials threadgroup: each column's digit planes for these
/// blocks share one set of transform barriers and matrix loads.
const BLOCK_TILE: usize = 4;
/// Matrix rows per partials threadgroup, accumulated in registers.
const ROW_TILE: usize = 4;
/// Threadgroups the partials pass aims for; columns split into chunks until
/// the grid reaches it.
const TARGET_GROUPS: usize = 256;

mod sealed {
    pub trait Sealed {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}
}

/// A signed digit type for digit planes: `i8` for `log_basis <= 8`, `i16`
/// up to 16.
pub trait DigitPlane: sealed::Sealed + NoUninit {
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

fn partials_kernel<T: DigitPlane>(ring_degree: usize) -> String {
    format!(
        "{}_{}",
        ring_degree_kernel("akita_matvec_partials", ring_degree),
        T::SUFFIX
    )
}

fn partials_instance<T: DigitPlane>(ring_degree: usize) -> Instance {
    Instance {
        template: "akita_matvec_partials",
        args: format!("{ring_degree}, {}, {BLOCK_TILE}, {ROW_TILE}", T::MSL_NAME),
        host_name: partials_kernel::<T>(ring_degree),
    }
}

pub(crate) fn instances() -> Vec<Instance> {
    RING_DEGREES
        .iter()
        .flat_map(|&ring_degree| {
            [
                partials_instance::<i8>(ring_degree),
                partials_instance::<i16>(ring_degree),
                Instance {
                    template: "akita_matvec_finish",
                    args: ring_degree.to_string(),
                    host_name: ring_degree_kernel("akita_matvec_finish", ring_degree),
                },
            ]
        })
        .collect()
}

/// Layout of `akita::MatvecShape` in `shaders/akita/matvec.metal`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct MatvecShape {
    blocks: u32,
    rows: u32,
    cols: u32,
    primes: u32,
    col_begin: u32,
    col_end: u32,
    chunk_cols: u32,
    chunks: u32,
    block_tiles: u32,
    row_tiles: u32,
}

/// The modulus of `F`: one more than the canonical value of `-1`.
#[expect(clippy::arithmetic_side_effects, reason = "field negation is modular")]
pub(crate) fn field_modulus<F: MetalField + CanonicalEncoding>() -> Result<u128, AkitaMetalError> {
    (-F::one())
        .to_u128_checked()
        .and_then(|top| top.checked_add(1))
        .ok_or_else(|| AkitaMetalError::Shape("the field modulus exceeds u128".into()))
}

/// A prepared `rows x cols` negacyclic NTT matrix in device memory.
pub struct DeviceNttMatrix<const K: usize, const D: usize> {
    ntt: DeviceCrtNtt<K, D>,
    /// `[rows][cols][K][D]` raw Montgomery words.
    entries: DeviceBuffer<i32>,
    rows: usize,
    cols: usize,
    capacity: CrtCapacity,
}

impl<const K: usize, const D: usize> DeviceNttMatrix<K, D> {
    /// Uploads the leading `rows x cols` entries of a prepared negacyclic
    /// matrix (row-major, as the CPU prepares it) and its CRT profile.
    pub fn new(
        metal: &AkitaMetal,
        view: PreparedNttBaseView<'_, i32, K, D>,
        rows: usize,
        cols: usize,
    ) -> Result<Self, AkitaMetalError> {
        let prepared = view.negacyclic().ok_or_else(|| {
            AkitaMetalError::Shape("the negacyclic NTT domain is not prepared".into())
        })?;
        let count = rows.checked_mul(cols).ok_or_else(|| shape_overflow(rows))?;
        let prepared = prepared.get(..count).ok_or_else(|| {
            AkitaMetalError::Shape(format!(
                "a prepared matrix of {} entries has no {rows} x {cols} prefix",
                prepared.len()
            ))
        })?;
        let words = prepared
            .iter()
            .flat_map(|entry| entry.limbs.iter().flatten().map(|word| word.raw()))
            .collect::<Vec<_>>();
        let params = view.params();
        Ok(Self {
            ntt: DeviceCrtNtt::new(metal, params)?,
            entries: DeviceBuffer::from_slice(metal.device(), &words)?,
            rows,
            cols,
            capacity: params.crt_capacity(),
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Multiplies the matrix by every block of `planes` and writes the
    /// products' field coefficients to `out`, returning the GPU time.
    ///
    /// `planes` holds `blocks x cols` digit planes of `D` digits, block-major,
    /// each digit balanced for `log_basis` (in `[-2^(log_basis-1),
    /// 2^(log_basis-1)]`); `out` receives `blocks x rows` ring elements. The
    /// columns run in CRT segments, each as wide as the primes' product can
    /// hold exactly, and the segments add in the field, so the result is
    /// `A x mod q` exactly at any width. A shape whose single column
    /// overflows the primes is rejected before any work.
    pub fn mat_vec<F, T>(
        &self,
        metal: &AkitaMetal,
        planes: &DeviceBuffer<T>,
        log_basis: u32,
        out: &mut DeviceBuffer<F>,
    ) -> Result<Duration, AkitaMetalError>
    where
        F: MetalField + CanonicalEncoding,
        T: DigitPlane,
    {
        if !(1..=T::MAX_LOG_BASIS).contains(&log_basis) {
            return Err(AkitaMetalError::Shape(format!(
                "log_basis {log_basis} does not fit {} digits",
                T::SUFFIX
            )));
        }
        let plane_words = checked::product([self.cols, D]).ok_or_else(|| shape_overflow(D))?;
        let blocks = checked::exact_div(planes.len(), plane_words).ok_or_else(|| {
            AkitaMetalError::Shape(format!(
                "{} digits is not a whole number of {} x {D} blocks",
                planes.len(),
                self.cols
            ))
        })?;
        let coefficients =
            checked::product([blocks, self.rows, D]).ok_or_else(|| shape_overflow(blocks))?;
        if out.len() != coefficients {
            return Err(AkitaMetalError::Shape(format!(
                "{blocks} blocks of {} rows need {coefficients} output coefficients, got {}",
                self.rows,
                out.len()
            )));
        }
        let modulus = field_modulus::<F>()?;
        let digit_bound = log_basis
            .checked_sub(1)
            .and_then(|shift| 1u64.checked_shl(shift))
            .ok_or_else(|| shape_overflow(D))?;
        // CRT segments: the widest column ranges whose exact products the
        // primes hold. Each is reconstructed into the field and added there,
        // as the CPU does for widths past its capacity.
        let segment_cols = self
            .capacity
            .max_safe_width_for_modulus(D, modulus, digit_bound)
            .filter(|&width| width > 0)
            .ok_or_else(|| {
                AkitaMetalError::Shape(format!(
                    "{K} primes cannot hold one column of degree {D} with digits up to {digit_bound}"
                ))
            })?
            .min(self.cols);
        if coefficients == 0 {
            return Ok(Duration::ZERO);
        }

        let block_tiles =
            checked::div_ceil(blocks, BLOCK_TILE).ok_or_else(|| shape_overflow(blocks))?;
        let row_tiles =
            checked::div_ceil(self.rows, ROW_TILE).ok_or_else(|| shape_overflow(self.rows))?;
        let tile_groups =
            checked::product([K, block_tiles, row_tiles]).ok_or_else(|| shape_overflow(blocks))?;
        let wanted_chunks = checked::div_ceil(TARGET_GROUPS, tile_groups)
            .unwrap_or(1)
            .clamp(1, segment_cols);
        let chunk_cols = checked::div_ceil(segment_cols, wanted_chunks)
            .ok_or_else(|| shape_overflow(segment_cols))?;
        // Chunks per segment; a shorter last segment leaves its trailing
        // chunks empty, and they contribute zero sums.
        let chunks = checked::div_ceil(segment_cols, chunk_cols)
            .ok_or_else(|| shape_overflow(segment_cols))?;
        let partial_groups =
            checked::product([chunks, tile_groups]).ok_or_else(|| shape_overflow(chunks))?;
        let residue_rows =
            checked::product([blocks, self.rows, K]).ok_or_else(|| shape_overflow(blocks))?;
        let lanes = D / 2;

        let u32_of = |value: usize| u32::try_from(value).map_err(|_| shape_overflow(value));
        let mut shapes = Vec::new();
        let mut crt_shapes = Vec::new();
        let mut col_begin = 0;
        while col_begin < self.cols {
            let col_end = col_begin
                .checked_add(segment_cols)
                .ok_or_else(|| shape_overflow(col_begin))?
                .min(self.cols);
            shapes.push(MatvecShape {
                blocks: u32_of(blocks)?,
                rows: u32_of(self.rows)?,
                cols: u32_of(self.cols)?,
                primes: u32_of(K)?,
                col_begin: u32_of(col_begin)?,
                col_end: u32_of(col_end)?,
                chunk_cols: u32_of(chunk_cols)?,
                chunks: u32_of(chunks)?,
                block_tiles: u32_of(block_tiles)?,
                row_tiles: u32_of(row_tiles)?,
            });
            crt_shapes.push(CrtBatch {
                coefficients: u32_of(coefficients)?,
                log_degree: D.trailing_zeros(),
                accumulate: u32::from(col_begin > 0),
            });
            col_begin = col_end;
        }
        let partials = DeviceBuffer::<i32>::zeroed(
            metal.device(),
            checked::product([chunks, residue_rows, D]).ok_or_else(|| shape_overflow(chunks))?,
        )?;
        let residues = DeviceBuffer::<i32>::zeroed(
            metal.device(),
            checked::product([residue_rows, D]).ok_or_else(|| shape_overflow(residue_rows))?,
        )?;
        let radix = DeviceBuffer::from_slice(metal.device(), &self.ntt.crt_weights::<F>())?;
        let partials_pipeline = metal.pipeline(&partials_kernel::<T>(D))?;
        let finish_pipeline = metal.pipeline(&ring_degree_kernel("akita_matvec_finish", D))?;
        let crt_pipeline = metal.pipeline(&crt_kernel::<F>(K))?;
        let partial_threads = checked::product([partial_groups, lanes])
            .ok_or_else(|| shape_overflow(partial_groups))?;
        let finish_threads =
            checked::product([residue_rows, lanes]).ok_or_else(|| shape_overflow(residue_rows))?;

        // Dispatches run in order, so the segments can share the partials
        // and residue buffers.
        let mut batch = Batch::new(metal.device())?;
        for (shape, crt_shape) in shapes.iter().zip(&crt_shapes) {
            batch.dispatch(
                partials_pipeline,
                &[
                    Binding::buffer(planes),
                    Binding::buffer(&self.entries),
                    Binding::buffer(&partials),
                    Binding::buffer(&self.ntt.primes),
                    Binding::buffer(&self.ntt.tables),
                    Binding::value(shape),
                ],
                Grid::linear(partial_threads, lanes),
            )?;
            batch.dispatch(
                finish_pipeline,
                &[
                    Binding::buffer(&partials),
                    Binding::buffer(&residues),
                    Binding::buffer(&self.ntt.primes),
                    Binding::buffer(&self.ntt.tables),
                    Binding::value(shape),
                ],
                Grid::linear(finish_threads, lanes),
            )?;
            batch.dispatch(
                crt_pipeline,
                &[
                    Binding::buffer(&residues),
                    Binding::buffer(out),
                    Binding::buffer(&self.ntt.primes),
                    Binding::buffer(&self.ntt.gamma),
                    Binding::buffer(&radix),
                    Binding::value(crt_shape),
                ],
                Grid::linear(coefficients, COEFFICIENT_GROUP),
            )?;
        }
        Ok(batch.commit_and_wait()?)
    }
}
