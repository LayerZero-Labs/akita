//! The one-hot inner Ajtai commitment on the device.
//!
//! [`commit_onehot`] computes the inner (A) commitment rows of one-hot
//! sources and matches `akita-cpu-backend`'s one-hot column sweep word for
//! word: for every source, block and A row, the canonical coefficients of
//!
//! ```text
//! t[block][row] = sum over hot entries (position, s) of the block
//!                 of A[row][position * num_digits_inner] * X^s
//! ```
//!
//! in `F[X]/(X^D + 1)`. The setup matrix stays resident in a
//! [`DeviceFlatMatrix`]; the sources' hot indices are uploaded once into a
//! [`DeviceOneHotSources`]. The kernel and its threading are described in
//! `shaders/akita/onehot.metal`.

use std::time::Duration;

use akita_cpu_backend::{OneHotIndex, OneHotSource};
use akita_error::checked;
use bytemuck::{Pod, Zeroable};
use jolt_field::{Prime128OffsetA7F7, Prime64Offset59};
use jolt_metal::runtime::{Batch, Binding, DeviceBuffer, Grid, MslType};
use jolt_metal::MetalField;

use crate::error::AkitaMetalError;
use crate::library::{ring_degree_kernel, AkitaMetal, Instance, RING_DEGREES};

/// `(MSL_NAME, HOST_SUFFIX, element bytes)` of the fields with one-hot
/// kernels: the base fields of Akita's fp128 and fp64 presets.
const FIELDS: [(&str, &str, usize); 2] = [
    (
        Prime128OffsetA7F7::MSL_NAME,
        Prime128OffsetA7F7::HOST_SUFFIX,
        size_of::<Prime128OffsetA7F7>(),
    ),
    (
        Prime64Offset59::MSL_NAME,
        Prime64Offset59::HOST_SUFFIX,
        size_of::<Prime64Offset59>(),
    ),
];

/// The one-hot commitment template, instantiated per field and ring degree.
const ONEHOT_COMMIT: &str = "akita_onehot_commit";
/// The segment-sum template, instantiated per field.
const ONEHOT_COMBINE: &str = "akita_onehot_combine";

/// The host name of a field template instance, from the field's
/// `MslType::HOST_SUFFIX`.
fn field_kernel(template: &str, suffix: &str) -> String {
    format!("{template}_{suffix}")
}

pub(crate) fn instances() -> Vec<Instance> {
    FIELDS
        .iter()
        .flat_map(|&(field, suffix, bytes)| {
            RING_DEGREES
                .iter()
                .filter(move |&&ring_degree| has_commit_kernel(bytes, ring_degree))
                .map(move |&ring_degree| Instance {
                    template: ONEHOT_COMMIT,
                    args: format!(
                        "{field}, {ring_degree}, {BLOCKS_PER_LANE}, {COEFFS_PER_THREAD}, \
                         {MAX_LANES}, {TILE_BYTES}, {NO_HOT}u"
                    ),
                    host_name: ring_degree_kernel(
                        &field_kernel(ONEHOT_COMMIT, suffix),
                        ring_degree,
                    ),
                })
                .chain(std::iter::once(Instance {
                    template: ONEHOT_COMBINE,
                    args: field.to_owned(),
                    host_name: field_kernel(ONEHOT_COMBINE, suffix),
                }))
        })
        .collect()
}

/// Blocks accumulated in registers by each lane: the `BLOCKS_PER_LANE`
/// argument of every `akita_onehot_commit` instance.
///
/// With [`COEFFS_PER_THREAD`], chosen on an Apple M4 from (blocks,
/// coefficients) in {(2, 4), (2, 8), (4, 1), (4, 2), (4, 4), (8, 1), (8, 2)}
/// by the geometric mean of GPU time over fp128 at D = 128, 256 (two chunk
/// sizes) and 512 and fp64 at D = 512: (8, 2) was 8% below the next best.
pub(crate) const BLOCKS_PER_LANE: usize = 8;

/// Coefficients of each block accumulated by one thread, so a lane is
/// `D / COEFFS_PER_THREAD` threads: the `COEFFS_PER_THREAD` argument.
pub(crate) const COEFFS_PER_THREAD: usize = 2;

/// Most lanes in a threadgroup: the `MAX_LANES` argument, which sizes the
/// kernel's staging memory.
pub(crate) const MAX_LANES: usize = 8;

/// Bytes of A per tile in threadgroup memory: the `TILE_BYTES` argument.
/// Tiles of 4, 8 and 16 KiB measured within 10% of each other on an Apple
/// M4. A tile holds at least one column of `D` field elements, so this also
/// bounds the ring degrees a field has kernels for; it keeps the kernel's
/// threadgroup memory within half of Metal's 32 KiB, which the shader
/// validation layer needs.
pub(crate) const TILE_BYTES: usize = 8192;

/// Threads of the segment-sum kernel's threadgroups.
const COMBINE_THREADS: usize = 256;

/// Lanes per threadgroup in the default schedule. More lanes share each A
/// tile among more blocks, but on an Apple M4 two and four lanes measured
/// up to 1.2x and 1.6x slower than one.
const DEFAULT_LANES: usize = 1;

/// Threadgroups the default schedule aims to keep in flight.
const TARGET_THREADGROUPS: usize = 128;

/// Fewest positions per segment in the default schedule, so that the
/// segment partial sums stay small next to the work that produces them.
const MIN_SEGMENT_POSITIONS: usize = 256;

/// Hot index of a chunk without a hot entry: the `NO_HOT` argument.
pub(crate) const NO_HOT: u32 = u32::MAX;

/// Layout of `akita::OneHotCommitParams` in `shaders/akita/onehot.metal`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CommitParams {
    chunks: u64,
    hot_offset: u64,
    out_offset: u64,
    segment_stride: u64,
    fold_terms: u64,
    log_chunk: u32,
    positions: u32,
    blocks: u32,
    digits: u32,
    columns: u32,
    rows: u32,
    segments: u32,
    block_groups: u32,
}

/// Layout of `akita::OneHotCombineParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CombineParams {
    len: u64,
    segments: u32,
    pad: u32,
}

/// The flat setup matrix (a `FlatMatrix`'s coefficients, or a prefix of
/// them) in device memory. A commitment views its first
/// `n_a * active_a_cols` ring elements as the `n_a x active_a_cols` A
/// matrix, as `FlatMatrix::ring_view` does.
pub struct DeviceFlatMatrix<F> {
    coefficients: DeviceBuffer<F>,
}

impl<F: MetalField> DeviceFlatMatrix<F> {
    /// Uploads `coefficients`.
    pub fn new(metal: &AkitaMetal, coefficients: &[F]) -> Result<Self, AkitaMetalError> {
        Ok(Self {
            coefficients: DeviceBuffer::from_slice(metal.device(), coefficients)?,
        })
    }
}

/// One uploaded source: where its chunks live and how they are sized.
#[derive(Clone, Copy, Debug)]
struct SourceLayout {
    hot_offset: usize,
    chunks: usize,
    log_chunk: u32,
    num_vars: usize,
}

/// The hot indices of a group of one-hot sources in device memory, one
/// `u32` per chunk (`u32::MAX` for a chunk without a hot entry), sources
/// concatenated in order.
pub struct DeviceOneHotSources {
    hot: DeviceBuffer<u32>,
    sources: Vec<SourceLayout>,
}

impl DeviceOneHotSources {
    /// Uploads `sources`.
    ///
    /// Rejects a source whose chunk size is not a power of two below `2^32`
    /// or that has a hot index outside its chunk. A [`OneHotPoly`] never has
    /// either, since its chunk size times its chunk count is a power of two
    /// and its constructor checks every index.
    ///
    /// [`OneHotPoly`]: akita_cpu_backend::OneHotPoly
    pub fn new<I: OneHotIndex>(
        metal: &AkitaMetal,
        sources: &[OneHotSource<'_, I>],
    ) -> Result<Self, AkitaMetalError> {
        let total = checked::sum(sources.iter().map(|source| source.indices.len()))
            .ok_or_else(|| shape_error("one-hot chunk count overflows usize"))?;
        let mut hot = Vec::new();
        hot.try_reserve_exact(total)
            .map_err(|_| shape_error(format!("cannot stage {total} one-hot chunks")))?;
        let mut layouts = Vec::with_capacity(sources.len());
        for source in sources {
            let chunk_size = source.chunk_size;
            if !chunk_size.is_power_of_two() || u32::try_from(chunk_size).is_err() {
                return Err(shape_error(format!(
                    "one-hot chunk size {chunk_size} is not a power of two below 2^32"
                )));
            }
            layouts.push(SourceLayout {
                hot_offset: hot.len(),
                chunks: source.indices.len(),
                log_chunk: chunk_size.trailing_zeros(),
                num_vars: source.num_vars,
            });
            for index in source.indices {
                hot.push(match index {
                    None => NO_HOT,
                    // A chunk size below 2^32 keeps every valid index below
                    // NO_HOT.
                    Some(index) => u32::try_from(index.as_usize())
                        .ok()
                        .filter(|&index| (index as usize) < chunk_size)
                        .ok_or_else(|| {
                            shape_error(format!(
                                "hot index {} is outside a chunk of {chunk_size}",
                                index.as_usize()
                            ))
                        })?,
                });
            }
        }
        Ok(Self {
            hot: DeviceBuffer::from_slice(metal.device(), &hot)?,
            sources: layouts,
        })
    }
}

/// The A-matrix geometry of a one-hot commitment, as the CPU's
/// `CommitInnerPlan` gives it: `n_a` rows of `active_a_cols` ring elements,
/// `num_digits_inner` columns per position, of which a one-hot unit touches
/// only the first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OneHotCommitShape {
    pub n_a: usize,
    pub active_a_cols: usize,
    pub num_digits_inner: usize,
}

/// How a commitment is spread over threadgroups. Any valid schedule gives
/// the same output; [`OneHotSchedule::new`] picks the default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OneHotSchedule {
    /// Lanes per threadgroup, each of `D / COEFFS_PER_THREAD` threads
    /// accumulating `BLOCKS_PER_LANE` blocks: from 1 to `MAX_LANES`, within
    /// the pipeline's thread limit.
    pub lanes: usize,
    /// Threadgroups sharing each block's positions, each summing an even
    /// share of the position tiles: at least 1. More than one adds a
    /// segment-sum pass.
    pub segments: usize,
    /// Terms each accumulator takes before the kernel folds it into the
    /// output: any value, capped by the kernel at the accumulator's
    /// capacity. The default, `u64::MAX`, folds only at that capacity; small
    /// values exercise the fold.
    pub fold_terms: u64,
}

impl OneHotSchedule {
    /// The most lanes a schedule for `F` at ring degree `D` may use:
    /// [`MAX_LANES`], or fewer when the kernel's registers limit its
    /// threadgroups.
    pub fn max_lanes<F: MetalField, const D: usize>(
        metal: &AkitaMetal,
    ) -> Result<usize, AkitaMetalError> {
        Ok(metal
            .pipeline(&commit_kernel::<F, D>()?)?
            .max_total_threads_per_threadgroup()
            .checked_div(lane_threads::<D>())
            .unwrap_or(0)
            .min(MAX_LANES))
    }

    /// The default schedule for committing `sources` at ring degree `D`:
    /// [`DEFAULT_LANES`] lanes, or fewer when the blocks run out, then
    /// enough segments to keep [`TARGET_THREADGROUPS`] threadgroups in
    /// flight while each segment keeps [`MIN_SEGMENT_POSITIONS`] positions.
    pub fn new<F: MetalField, const D: usize>(
        metal: &AkitaMetal,
        sources: &DeviceOneHotSources,
        shape: OneHotCommitShape,
    ) -> Result<Self, AkitaMetalError> {
        let geometry = Geometry::new::<D>(shape)?;
        let max_lanes = Self::max_lanes::<F, D>(metal)?;
        let blocks = sources
            .sources
            .iter()
            .map(|source| geometry.blocks::<D>(source))
            .try_fold(0, |max, blocks| blocks.map(|blocks| max.max(blocks)))?;
        let lanes = blocks
            .div_ceil(BLOCKS_PER_LANE)
            .min(DEFAULT_LANES)
            .clamp(1, max_lanes.max(1));
        let group_blocks = checked::product([BLOCKS_PER_LANE, lanes])
            .ok_or_else(|| shape_overflow("blocks per threadgroup"))?;
        let groups = checked::product([blocks.div_ceil(group_blocks), shape.n_a])
            .ok_or_else(|| shape_overflow("threadgroups"))?
            .max(1);
        let segments = TARGET_THREADGROUPS
            .div_ceil(groups)
            .min(
                geometry
                    .positions
                    .checked_div(MIN_SEGMENT_POSITIONS)
                    .unwrap_or(0),
            )
            .max(1);
        Ok(Self {
            lanes,
            segments,
            fold_terms: u64::MAX,
        })
    }
}

/// A validated [`OneHotCommitShape`] at ring degree `D`.
#[derive(Clone, Copy, Debug)]
struct Geometry {
    rows: usize,
    columns: usize,
    digits: usize,
    positions: usize,
}

impl Geometry {
    fn new<const D: usize>(shape: OneHotCommitShape) -> Result<Self, AkitaMetalError> {
        let positions = checked::exact_div(shape.active_a_cols, shape.num_digits_inner)
            .ok_or_else(|| {
                shape_error(format!(
                    "A width {} is not a multiple of a nonzero digit count {}",
                    shape.active_a_cols, shape.num_digits_inner
                ))
            })?;
        if !positions.is_power_of_two() {
            return Err(shape_error(format!(
                "{positions} positions per block is not a power of two"
            )));
        }
        Ok(Self {
            rows: shape.n_a,
            columns: shape.active_a_cols,
            digits: shape.num_digits_inner,
            positions,
        })
    }

    /// The source's live blocks: its `2^num_vars / D` ring elements in
    /// blocks of `positions`, as the CPU's `OneHotSource::view_layout`.
    fn blocks<const D: usize>(&self, source: &SourceLayout) -> Result<usize, AkitaMetalError> {
        let rings = u32::try_from(source.num_vars)
            .ok()
            .and_then(|num_vars| 1usize.checked_shl(num_vars))
            .and_then(|fields| checked::exact_div(fields, D))
            .ok_or_else(|| {
                shape_error(format!(
                    "2^{} one-hot entries are not a whole number of D={D} ring elements",
                    source.num_vars
                ))
            })?;
        checked::div_ceil(rings, self.positions).ok_or_else(|| shape_overflow("blocks"))
    }
}

/// Commits `sources` with the A matrix `shape` views in `matrix`, at ring
/// degree `D`, and returns the rows and the GPU time.
///
/// The rows are canonical field elements in the CPU's order: source, block,
/// A row, coefficient. Source `i` contributes its `ceil(2^num_vars / (D P))`
/// blocks of `n_a` ring elements, `P = active_a_cols / num_digits_inner` being
/// the positions per block. Shapes the kernel does not cover are rejected
/// with [`AkitaMetalError::Shape`] before anything is encoded.
pub fn commit_onehot<F: MetalField, const D: usize>(
    metal: &AkitaMetal,
    matrix: &DeviceFlatMatrix<F>,
    sources: &DeviceOneHotSources,
    shape: OneHotCommitShape,
    schedule: OneHotSchedule,
) -> Result<(DeviceBuffer<F>, Duration), AkitaMetalError> {
    let geometry = Geometry::new::<D>(shape)?;
    let pipeline = metal.pipeline(&commit_kernel::<F, D>()?)?;
    let matrix_len = checked::product([geometry.rows, geometry.columns, D])
        .ok_or_else(|| shape_overflow("A matrix"))?;
    if matrix_len > matrix.coefficients.len() {
        return Err(shape_error(format!(
            "a {} x {} A matrix at D={D} needs {matrix_len} coefficients; the setup has {}",
            geometry.rows,
            geometry.columns,
            matrix.coefficients.len()
        )));
    }
    let max_lanes = OneHotSchedule::max_lanes::<F, D>(metal)?;
    if !(1..=max_lanes).contains(&schedule.lanes) {
        return Err(shape_error(format!(
            "{} lanes is outside 1..={max_lanes}",
            schedule.lanes
        )));
    }
    let threads = checked::product([schedule.lanes, lane_threads::<D>()])
        .ok_or_else(|| shape_overflow("threads per threadgroup"))?;
    if schedule.segments == 0 {
        return Err(shape_error("a schedule needs at least one segment"));
    }
    let group_blocks = checked::product([schedule.lanes, BLOCKS_PER_LANE])
        .ok_or_else(|| shape_overflow("blocks per threadgroup"))?;

    // Per-source dispatch parameters, then the output length they share.
    let mut dispatches = Vec::with_capacity(sources.sources.len());
    let mut len = 0usize;
    for source in &sources.sources {
        let blocks = geometry.blocks::<D>(source)?;
        let block_groups = blocks.div_ceil(group_blocks);
        // The kernel numbers blocks up to block_groups * group_blocks in 32
        // bits.
        to_u32(
            checked::product([block_groups, group_blocks])
                .ok_or_else(|| shape_overflow("blocks"))?,
        )?;
        let groups = checked::product([schedule.segments, geometry.rows, block_groups])
            .ok_or_else(|| shape_overflow("threadgroups"))?;
        let params = CommitParams {
            chunks: to_u64(source.chunks)?,
            hot_offset: to_u64(source.hot_offset)?,
            out_offset: to_u64(len)?,
            segment_stride: 0,
            fold_terms: schedule.fold_terms,
            log_chunk: source.log_chunk,
            positions: to_u32(geometry.positions)?,
            blocks: to_u32(blocks)?,
            digits: to_u32(geometry.digits)?,
            columns: to_u32(geometry.columns)?,
            rows: to_u32(geometry.rows)?,
            segments: to_u32(schedule.segments)?,
            block_groups: to_u32(block_groups)?,
        };
        let grid_threads =
            checked::product([groups, threads]).ok_or_else(|| shape_overflow("threads"))?;
        to_u32(grid_threads)?;
        dispatches.push((params, grid_threads));
        len = checked::product([blocks, geometry.rows, D])
            .and_then(|rows| len.checked_add(rows))
            .ok_or_else(|| shape_overflow("output"))?;
    }
    let segment_stride = to_u64(len)?;
    for (params, _) in &mut dispatches {
        params.segment_stride = segment_stride;
    }

    // With several segments the commitment kernel writes one partial sum per
    // segment, and the segment-sum kernel adds them into `out`.
    let device = metal.device();
    let out = DeviceBuffer::<F>::zeroed(device, len)?;
    let partials = if schedule.segments > 1 {
        let partials_len = checked::product([schedule.segments, len])
            .ok_or_else(|| shape_overflow("segment partial sums"))?;
        Some(DeviceBuffer::<F>::zeroed(device, partials_len)?)
    } else {
        None
    };
    let combine_params = CombineParams {
        len: segment_stride,
        segments: to_u32(schedule.segments)?,
        pad: 0,
    };
    let combine_threads = checked::align_up(len, COMBINE_THREADS)
        .ok_or_else(|| shape_overflow("segment-sum threads"))?;
    to_u32(combine_threads)?;

    let mut batch = Batch::new(device)?;
    for (params, grid_threads) in &dispatches {
        batch.dispatch(
            pipeline,
            &[
                Binding::buffer(&matrix.coefficients),
                Binding::buffer(&sources.hot),
                Binding::buffer(partials.as_ref().unwrap_or(&out)),
                Binding::value(params),
            ],
            Grid::linear(*grid_threads, threads),
        )?;
    }
    if let Some(partials) = &partials {
        batch.dispatch(
            metal.pipeline(&field_kernel(ONEHOT_COMBINE, F::HOST_SUFFIX))?,
            &[
                Binding::buffer(partials),
                Binding::buffer(&out),
                Binding::value(&combine_params),
            ],
            Grid::linear(combine_threads, COMBINE_THREADS),
        )?;
    }
    let time = batch.commit_and_wait()?;
    Ok((out, time))
}

/// Threads of one lane at ring degree `D`.
fn lane_threads<const D: usize>() -> usize {
    D.checked_div(COEFFS_PER_THREAD).unwrap_or(0)
}

/// Whether a field of `field_bytes`-byte elements has a commitment kernel at
/// `ring_degree`: a degree with transform kernels whose A column fits a tile.
pub(crate) fn has_commit_kernel(field_bytes: usize, ring_degree: usize) -> bool {
    RING_DEGREES.contains(&ring_degree)
        && ring_degree
            .checked_mul(field_bytes)
            .is_some_and(|bytes| bytes <= TILE_BYTES)
}

/// The host name of `F`'s commitment kernel at ring degree `D`.
fn commit_kernel<F: MetalField, const D: usize>() -> Result<String, AkitaMetalError> {
    if !FIELDS
        .iter()
        .any(|&(_, suffix, _)| suffix == F::HOST_SUFFIX)
        || !has_commit_kernel(size_of::<F>(), D)
    {
        return Err(shape_error(format!(
            "field {} has no one-hot commitment kernel at D={D}",
            F::MSL_NAME
        )));
    }
    Ok(ring_degree_kernel(
        &field_kernel(ONEHOT_COMMIT, F::HOST_SUFFIX),
        D,
    ))
}

fn shape_error(reason: impl Into<String>) -> AkitaMetalError {
    AkitaMetalError::Shape(reason.into())
}

fn shape_overflow(what: &str) -> AkitaMetalError {
    shape_error(format!("the {what} count overflows"))
}

fn to_u32(value: usize) -> Result<u32, AkitaMetalError> {
    u32::try_from(value)
        .map_err(|_| shape_error(format!("{value} exceeds the kernel's 32-bit range")))
}

fn to_u64(value: usize) -> Result<u64, AkitaMetalError> {
    u64::try_from(value)
        .map_err(|_| shape_error(format!("{value} exceeds the kernel's 64-bit range")))
}
