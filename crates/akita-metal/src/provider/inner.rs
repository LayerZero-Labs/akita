//! The inner (A) commitment stage on the device.

use std::marker::PhantomData;
use std::time::Instant;

use akita_cpu_backend::benchmark_support::column_sweep_ajtai_onehot_multi;
use akita_cpu_backend::commitment_backend::{
    BackendStateRef, CommitInnerPlan, CommitmentStateBinding, DenseRepresentation,
    InnerCommitOperation, InnerCommitOutput, InnerImage, InnerImageExportOperation,
    PolynomialRepresentation, ResolvedCommitSource, StateOwnerCapability, UnitPositionSlice,
};
use akita_cpu_backend::{OneHotIndex, OneHotSource};
use akita_error::{checked, AkitaError};
use akita_types::RingVec;
use jolt_metal::runtime::DeviceBuffer;

use super::shared::Shared;
use super::{bump, record, CommitmentField, MetalCommitmentProvider};
use crate::decompose::decompose;
use crate::error::AkitaMetalError;
use crate::matvec::DigitPlane;
use crate::onehot::{
    commit_onehot, has_commit_kernel, DeviceOneHotSources, OneHotCommitShape, OneHotSchedule,
};

/// The resident inner image: the rows `t` read back once for the retained
/// state and a CPU outer stage, and the device copy the outer stage reads.
pub(super) struct MetalInnerImage<F> {
    /// One `RingVec` per source: `num_live_blocks x n_a` rings of the A ring
    /// degree, block-major.
    pub(super) rows: Vec<RingVec<F>>,
    /// The same rows, sources concatenated, in device memory.
    pub(super) device: Option<Shared<DeviceBuffer<F>>>,
}

/// The inner stage operation.
pub(super) struct MetalInnerCommit<'a, F: CommitmentField> {
    provider: &'a MetalCommitmentProvider<F>,
    owner: StateOwnerCapability<InnerImage>,
}

impl<'a, F: CommitmentField> MetalInnerCommit<'a, F> {
    pub(super) fn new(
        provider: &'a MetalCommitmentProvider<F>,
        owner: StateOwnerCapability<InnerImage>,
    ) -> Self {
        Self { provider, owner }
    }
}

impl<F: CommitmentField> InnerCommitOperation<F> for MetalInnerCommit<'_, F> {
    fn supports_plan(&self, plan: &CommitInnerPlan) -> bool {
        super::inner_supported::<F>(plan.ring_dimension, plan.log_basis_inner)
    }

    fn commit_inner(
        &self,
        binding: &CommitmentStateBinding,
        plan: &CommitInnerPlan,
        sources: &[ResolvedCommitSource<'_, F>],
    ) -> Result<InnerCommitOutput, AkitaError> {
        let start = Instant::now();
        for source in sources {
            source.validate_plan(plan)?;
        }
        let image = with_ring_degree!(plan.ring_dimension, |D| self
            .provider
            .commit_inner::<D>(plan, sources))?;
        let counters = &self.provider.counters;
        record(&counters.inner, &counters.inner_nanos, start);
        let bytes = checked::product([
            image.rows.iter().map(RingVec::coeff_len).sum::<usize>(),
            size_of::<F>(),
        ])
        .unwrap_or(usize::MAX);
        Ok(InnerCommitOutput::new(self.owner.bind(
            binding.clone(),
            bytes,
            image,
        )))
    }
}

/// Exports the rows the inner stage already read back.
pub(super) struct MetalInnerExporter<F> {
    owner: StateOwnerCapability<InnerImage>,
    field: PhantomData<fn() -> F>,
}

impl<F> MetalInnerExporter<F> {
    pub(super) fn new(owner: StateOwnerCapability<InnerImage>) -> Self {
        Self {
            owner,
            field: PhantomData,
        }
    }
}

impl<F: CommitmentField> InnerImageExportOperation<F> for MetalInnerExporter<F> {
    fn export_inner_rows(
        &self,
        _plan: &CommitInnerPlan,
        image: &BackendStateRef<InnerImage>,
    ) -> Result<Vec<RingVec<F>>, AkitaError> {
        Ok(self.owner.value::<MetalInnerImage<F>>(image)?.rows.clone())
    }

    fn consume_inner_rows(
        &self,
        plan: &CommitInnerPlan,
        image: BackendStateRef<InnerImage>,
    ) -> Result<Vec<RingVec<F>>, AkitaError> {
        match self.owner.try_unwrap::<MetalInnerImage<F>>(image)? {
            Ok(image) => Ok(image.rows),
            Err(shared) => self.export_inner_rows(plan, &shared),
        }
    }
}

/// Splits the concatenated rows of `sources` equal groups into one `RingVec`
/// per source.
fn split_rows<F: CommitmentField>(
    flat: &[F],
    sources: usize,
    ring_degree: usize,
) -> Result<Vec<RingVec<F>>, AkitaError> {
    let per_source = checked::exact_div(flat.len(), sources)
        .filter(|&len| len > 0)
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "{} inner coefficients do not split into {sources} sources",
                flat.len()
            ))
        })?;
    flat.chunks_exact(per_source)
        .map(|rows| RingVec::from_coeffs_with_ring_dim(rows.to_vec(), ring_degree))
        .collect()
}

fn mismatch(what: &str) -> AkitaError {
    AkitaError::InvalidInput(format!(
        "the Metal inner stage received a group with {what}"
    ))
}

impl<F: CommitmentField> MetalCommitmentProvider<F> {
    fn commit_inner<const D: usize>(
        &self,
        plan: &CommitInnerPlan,
        sources: &[ResolvedCommitSource<'_, F>],
    ) -> Result<MetalInnerImage<F>, AkitaError> {
        let first = sources
            .first()
            .ok_or_else(|| mismatch("no sources"))?
            .representation();
        match first {
            Some(PolynomialRepresentation::Dense(DenseRepresentation::Coefficients(_))) => {
                self.commit_dense::<D>(plan, sources)
            }
            Some(PolynomialRepresentation::OneHot(onehot)) => {
                macro_rules! onehot_group {
                    ($variant:ident) => {{
                        let group = sources
                            .iter()
                            .map(|source| match source.representation() {
                                Some(PolynomialRepresentation::OneHot(onehot)) => {
                                    match onehot.positions {
                                        UnitPositionSlice::$variant(indices) => Ok(OneHotSource {
                                            indices,
                                            chunk_size: onehot.chunk_size,
                                            num_vars: onehot.num_vars,
                                        }),
                                        _ => Err(mismatch("mixed one-hot index widths")),
                                    }
                                }
                                _ => Err(mismatch("mixed representations")),
                            })
                            .collect::<Result<Vec<_>, AkitaError>>()?;
                        self.commit_onehot::<D, _>(plan, &group)
                    }};
                }
                match onehot.positions {
                    UnitPositionSlice::U8(_) => onehot_group!(U8),
                    UnitPositionSlice::U16(_) => onehot_group!(U16),
                    UnitPositionSlice::U32(_) => onehot_group!(U32),
                    UnitPositionSlice::Usize(_) => onehot_group!(Usize),
                }
            }
            _ => Err(mismatch("a representation without a Metal kernel")),
        }
    }

    /// Dense coefficients: upload (zero-padding a short last block), decompose
    /// into digit planes, multiply by A.
    fn commit_dense<const D: usize>(
        &self,
        plan: &CommitInnerPlan,
        sources: &[ResolvedCommitSource<'_, F>],
    ) -> Result<MetalInnerImage<F>, AkitaError> {
        let overflow = || AkitaError::InvalidSetup("dense inner shape overflows".into());
        let mut slices = Vec::with_capacity(sources.len());
        let mut ring_count = None;
        for source in sources {
            let Some(PolynomialRepresentation::Dense(DenseRepresentation::Coefficients(dense))) =
                source.representation()
            else {
                return Err(mismatch("mixed representations"));
            };
            let logical = checked::pow2(source.descriptor().num_vars()).ok_or_else(overflow)?;
            let rings = checked::div_ceil(logical, D).ok_or_else(overflow)?;
            if ring_count
                .replace(rings)
                .is_some_and(|expected| expected != rings)
            {
                return Err(mismatch("unequal source sizes"));
            }
            let physical = checked::product([rings, D]).ok_or_else(overflow)?;
            let coefficients = dense.coefficients();
            slices.push(
                coefficients
                    .get(..physical)
                    .ok_or(AkitaError::InvalidSize {
                        expected: physical,
                        actual: coefficients.len(),
                    })?,
            );
        }
        let rings = ring_count.ok_or_else(|| mismatch("no sources"))?;
        // The CPU's A width is its first block's: a lone short block uses a
        // narrower matrix, a short last block is zero-padded.
        let block_rings = rings.min(plan.num_positions_per_block);
        if checked::div_ceil(rings, plan.num_positions_per_block) != Some(plan.num_live_blocks) {
            return Err(AkitaError::InvalidSetup(
                "dense source block count disagrees with the inner plan".into(),
            ));
        }
        let block_fields = checked::product([block_rings, D]).ok_or_else(overflow)?;
        let cols = checked::product([block_rings, plan.num_digits_inner]).ok_or_else(overflow)?;
        let metal = self.metal.get();
        let matrix = self.matrices.ntt::<D>(
            metal,
            self.expanded.shared_matrix(),
            plan.n_a,
            cols,
            plan.log_basis_inner,
        )?;
        // Blocks of every source in order, in chunks whose digit planes fit
        // the chunk budget, so any source size stays within device memory.
        let total_blocks =
            checked::product([sources.len(), plan.num_live_blocks]).ok_or_else(overflow)?;
        let digit_bytes = if plan.log_basis_inner <= i8::MAX_LOG_BASIS {
            size_of::<i8>()
        } else {
            size_of::<i16>()
        };
        let block_bytes = checked::product([block_fields, plan.num_digits_inner, digit_bytes])
            .ok_or_else(overflow)?;
        let chunk_blocks = self
            .dense_chunk_bytes
            .checked_div(block_bytes)
            .unwrap_or(0)
            .clamp(1, total_blocks);
        let row_fields = checked::product([plan.n_a, D]).ok_or_else(overflow)?;
        let mut rows =
            Vec::with_capacity(checked::product([total_blocks, row_fields]).ok_or_else(overflow)?);
        let mut device_rows = None;
        let mut first = 0;
        while first < total_blocks {
            let count = chunk_blocks.min(total_blocks.saturating_sub(first));
            let coefficients = stage_blocks(
                metal,
                &slices,
                plan.num_live_blocks,
                block_fields,
                first..checked::sum([first, count]).ok_or_else(overflow)?,
            )?;
            let digits = checked::product([coefficients.len(), plan.num_digits_inner])
                .ok_or_else(overflow)?;
            let mut out = DeviceBuffer::<F>::zeroed(
                metal.device(),
                checked::product([count, row_fields]).ok_or_else(overflow)?,
            )
            .map_err(AkitaMetalError::from)?;
            if plan.log_basis_inner <= i8::MAX_LOG_BASIS {
                let mut planes = DeviceBuffer::<i8>::zeroed(metal.device(), digits)
                    .map_err(AkitaMetalError::from)?;
                decompose(
                    metal,
                    &coefficients,
                    D,
                    plan.num_digits_inner,
                    plan.log_basis_inner,
                    &mut planes,
                )?;
                matrix.mat_vec_i8(metal, &planes, plan.log_basis_inner, &mut out)?;
            } else {
                let mut planes = DeviceBuffer::<i16>::zeroed(metal.device(), digits)
                    .map_err(AkitaMetalError::from)?;
                decompose(
                    metal,
                    &coefficients,
                    D,
                    plan.num_digits_inner,
                    plan.log_basis_inner,
                    &mut planes,
                )?;
                matrix.mat_vec_i16(metal, &planes, plan.log_basis_inner, &mut out)?;
            }
            rows.extend_from_slice(out.read().map_err(AkitaMetalError::from)?);
            if count == total_blocks {
                device_rows = Some(Shared::new(out));
            }
            first = checked::sum([first, count]).ok_or_else(overflow)?;
        }
        Ok(MetalInnerImage {
            rows: split_rows(&rows, sources.len(), D)?,
            device: device_rows,
        })
    }

    /// One-hot sources: the one-hot kernel over A's coefficient form, or the
    /// CPU column sweep at a ring degree without a kernel for the field.
    fn commit_onehot<const D: usize, I: OneHotIndex>(
        &self,
        plan: &CommitInnerPlan,
        group: &[OneHotSource<'_, I>],
    ) -> Result<MetalInnerImage<F>, AkitaError> {
        let overflow = || AkitaError::InvalidSetup("one-hot inner shape overflows".into());
        let active_a_cols = checked::product([plan.num_positions_per_block, plan.num_digits_inner])
            .ok_or_else(overflow)?;
        let expected = checked::product([group.len(), plan.num_live_blocks, plan.n_a, D])
            .ok_or_else(overflow)?;
        if !has_commit_kernel(size_of::<F>(), D) {
            bump(&self.counters.onehot_host);
            let view = self
                .expanded
                .shared_matrix()
                .ring_view::<D>(plan.n_a, active_a_cols)?;
            let rows = column_sweep_ajtai_onehot_multi::<F, D, I>(
                &view,
                group,
                plan.n_a,
                active_a_cols,
                plan.num_digits_inner,
            )?
            .into_iter()
            .map(|blocks| {
                RingVec::from_ring_elems(&blocks.into_iter().flatten().collect::<Vec<_>>())
            })
            .collect::<Vec<_>>();
            if rows.iter().map(RingVec::coeff_len).sum::<usize>() != expected {
                return Err(mismatch("a block count the plan does not have"));
            }
            return Ok(MetalInnerImage { rows, device: None });
        }
        let matrix_len = checked::product([plan.n_a, active_a_cols, D]).ok_or_else(overflow)?;
        let matrix =
            self.matrices
                .flat(self.metal.get(), self.expanded.shared_matrix(), matrix_len)?;
        let sources = DeviceOneHotSources::new(self.metal.get(), group)?;
        let shape = OneHotCommitShape {
            n_a: plan.n_a,
            active_a_cols,
            num_digits_inner: plan.num_digits_inner,
        };
        let schedule = OneHotSchedule::new::<F, D>(self.metal.get(), &sources, shape)?;
        let (mut out, _) =
            commit_onehot::<F, D>(self.metal.get(), matrix.get(), &sources, shape, schedule)?;
        if out.len() != expected {
            return Err(mismatch("a block count the plan does not have"));
        }
        let rows = split_rows(out.read().map_err(AkitaMetalError::from)?, group.len(), D)?;
        Ok(MetalInnerImage {
            rows,
            device: Some(Shared::new(out)),
        })
    }
}

/// The coefficients of global blocks `blocks` (block `g` is block
/// `g % blocks_per_source` of source `g / blocks_per_source`) on the device,
/// each `block_fields` long with a short last block zero-padded. Zero
/// coefficients decompose to zero digits, as the CPU pads its last block.
fn stage_blocks<F: CommitmentField>(
    metal: &crate::library::AkitaMetal,
    slices: &[&[F]],
    blocks_per_source: usize,
    block_fields: usize,
    blocks: std::ops::Range<usize>,
) -> Result<DeviceBuffer<F>, AkitaError> {
    let overflow = || AkitaError::InvalidSetup("dense inner shape overflows".into());
    let locate = |block: usize| {
        let source = block.checked_div(blocks_per_source)?;
        let local = block.checked_rem(blocks_per_source)?;
        Some((
            slices.get(source)?,
            checked::product([local, block_fields])?,
        ))
    };
    let count = blocks.len();
    let last = blocks.end.checked_sub(1).ok_or_else(overflow)?;
    let (source, start) = locate(blocks.start).ok_or_else(overflow)?;
    let (last_source, last_start) = locate(last).ok_or_else(overflow)?;
    // One source's whole blocks are already contiguous.
    let end = checked::sum([last_start, block_fields]).ok_or_else(overflow)?;
    if std::ptr::eq(*source, *last_source) {
        if let Some(contiguous) = source.get(start..end) {
            return Ok(DeviceBuffer::from_slice(metal.device(), contiguous)
                .map_err(AkitaMetalError::from)?);
        }
    }
    let mut staged = vec![F::zero(); checked::product([count, block_fields]).ok_or_else(overflow)?];
    for (block, destination) in blocks.zip(staged.chunks_exact_mut(block_fields)) {
        let (source, start) = locate(block).ok_or_else(overflow)?;
        let present = source.get(start..).unwrap_or_default();
        let present = present.get(..block_fields).unwrap_or(present);
        destination
            .get_mut(..present.len())
            .ok_or_else(overflow)?
            .copy_from_slice(present);
    }
    Ok(DeviceBuffer::from_slice(metal.device(), &staged).map_err(AkitaMetalError::from)?)
}
