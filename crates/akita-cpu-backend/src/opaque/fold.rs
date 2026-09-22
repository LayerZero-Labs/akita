use super::balanced_decompose_centered_i32_i8_into;
use super::witness_build as recursive_witness_builder;
use crate::opaque::DecomposeFoldWitness;
use crate::opaque::*;
use crate::opaque::{
    FoldRelationKernel, FoldRelationOutput, OpeningBatchKernel, RelationQuotientRow,
};
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_types::{gadget_row_scalars, RingMultiplierOpeningPoint};
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, Field};
use std::marker::PhantomData;

#[inline]
pub(crate) fn response_model_diagnostics_enabled() -> bool {
    #[cfg(feature = "response-model-diagnostics")]
    {
        tracing::enabled!(
            target: "akita_prover::protocol::fold_response_model",
            tracing::Level::INFO
        )
    }
    #[cfg(not(feature = "response-model-diagnostics"))]
    {
        false
    }
}

/// CPU reference state retained behind an accepted fold handle.
pub struct CpuAcceptedFold<F: Field> {
    global: DecomposeFoldWitness,
    chunks: Option<Vec<Vec<i32>>>,
    challenges: akita_challenges::Challenges,
    manifest: AcceptedFoldManifest,
    binding: crate::opaque::OperationBinding,
    _field: PhantomData<F>,
}

/// CPU reference state retained behind an accepted terminal-fold handle.
pub struct CpuAcceptedTerminalFold<F: Field> {
    witness: DecomposeFoldWitness,
    ring_dimension: usize,
    rice_low_bits: u32,
    zigzag_width: u32,
    payload_bytes: usize,
    binding: crate::opaque::OperationBinding,
    _field: PhantomData<F>,
}

pub(crate) type CpuAcceptedFoldHandle<F> = CpuAcceptedFold<F>;
pub(crate) type CpuAcceptedTerminalFoldHandle<F> = CpuAcceptedTerminalFold<F>;

impl<F> CpuAcceptedTerminalFold<F>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
{
    pub(crate) fn bind(&mut self, binding: crate::opaque::OperationBinding) {
        self.binding = binding;
    }

    pub(crate) fn binding(&self) -> crate::opaque::OperationBinding {
        self.binding
    }

    fn admit<B: ComputeBackendSetup<F>, const D: usize>(
        _backend: &B,
        _prepared: Option<&B::PreparedSetup>,
        witness: DecomposeFoldWitness,
        plan: &ValidatedTerminalFoldProbePlan<'_>,
    ) -> Result<Option<(Self, Option<u128>)>, AkitaError> {
        let centered = witness.centered_coeffs_flat();
        if centered.len() != plan.coordinate_count() {
            return Err(AkitaError::InvalidSize {
                expected: plan.coordinate_count(),
                actual: centered.len(),
            });
        }
        if let Some(cap) = plan.linf_cap() {
            if akita_types::golomb_rice_values_within_cap(centered, cap).is_err() {
                return Ok(None);
            }
        } else if centered.iter().any(|value| i16::try_from(*value).is_err()) {
            return Ok(None);
        }
        let observed_l2_sq = if plan.l2_sq_cap().is_some() || response_model_diagnostics_enabled() {
            let value = akita_types::sis::checked_centered_l2_sq(centered);
            if plan.l2_sq_cap().is_some() && value.is_none() {
                return Err(AkitaError::InvalidInput(
                    "terminal fold response L2 overflow".into(),
                ));
            }
            if value.is_some_and(|value| plan.l2_sq_cap().is_some_and(|cap| value > cap)) {
                return Ok(None);
            }
            value
        } else {
            None
        };
        let zigzag_width =
            akita_types::golomb_rice_zigzag_width(plan.linf_cap().unwrap_or(i16::MAX as u128));
        if akita_types::golomb_rice_total_wire_bits(centered, plan.rice_low_bits(), zigzag_width)?
            > plan.payload_bytes().saturating_mul(8)
        {
            return Ok(None);
        }
        Ok(Some((
            Self {
                witness,
                ring_dimension: D,
                rice_low_bits: plan.rice_low_bits(),
                zigzag_width,
                payload_bytes: plan.payload_bytes(),
                binding: crate::opaque::OperationBinding::legacy_unscoped(),
                _field: PhantomData,
            },
            observed_l2_sq,
        )))
    }

    pub(crate) fn encode<B: ComputeBackendSetup<F>, const D: usize>(
        &self,
        backend: &B,
        prepared: Option<&B::PreparedSetup>,
    ) -> Result<Vec<u8>, AkitaError> {
        let _ = (backend, prepared);
        if D != self.ring_dimension {
            return Err(AkitaError::InvalidInput(
                "accepted terminal fold used with a different operation context".into(),
            ));
        }
        let values = self
            .witness
            .centered_coeffs_flat()
            .iter()
            .map(|value| i64::from(*value))
            .collect::<Vec<_>>();
        let payload =
            akita_types::golomb_rice_encode_vec(&values, self.rice_low_bits, self.zigzag_width)?;
        if payload.len() > self.payload_bytes {
            return Err(AkitaError::InvalidInput(
                "terminal response exceeds its scheduled payload budget".into(),
            ));
        }
        Ok(payload)
    }
}

impl<F: Field> CpuAcceptedFold<F> {
    pub(crate) fn validate_challenges(
        &self,
        challenges: &akita_challenges::Challenges,
    ) -> Result<(), AkitaError> {
        if &self.challenges != challenges {
            return Err(AkitaError::InvalidInput(
                "accepted fold belongs to different public challenges".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn binding(&self) -> super::OperationBinding {
        self.binding
    }
    pub(crate) fn bind(&mut self, binding: crate::opaque::OperationBinding) {
        self.binding = binding;
    }

    pub(crate) fn ensure_ring_dim<const D: usize>(&self) -> Result<(), AkitaError> {
        if self.manifest.ring_dimension != D {
            return Err(AkitaError::InvalidInput(
                "accepted fold used with a different ring dimension".into(),
            ));
        }
        self.global.ensure_ring_dim::<D>()?;
        if self
            .chunks
            .iter()
            .flatten()
            .any(|chunk| !chunk.len().is_multiple_of(D))
        {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self
                    .chunks
                    .iter()
                    .flatten()
                    .find(|chunk| !chunk.len().is_multiple_of(D))
                    .map_or(0, Vec::len),
            });
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_z_planes<const D: usize>(
        &self,
        plan: &recursive_witness_builder::CpuRecursiveWitnessUnitPlan<'_>,
        builder: &mut recursive_witness_builder::CpuRecursiveWitnessBuilder,
    ) -> Result<(), AkitaError> {
        self.ensure_ring_dim::<D>()?;
        let actual_chunks = self.chunks.as_ref().map_or(1, Vec::len);
        if actual_chunks != plan.expected_chunks() {
            return Err(AkitaError::InvalidSize {
                expected: plan.expected_chunks(),
                actual: actual_chunks,
            });
        }
        let centered = match &self.chunks {
            None if plan.chunk_index() == 0 => self.global.centered_coeffs_flat(),
            None => {
                return Err(AkitaError::InvalidSize {
                    expected: 1,
                    actual: plan.chunk_index() + 1,
                })
            }
            Some(chunks) => chunks.get(plan.chunk_index()).map(Vec::as_slice).ok_or(
                AkitaError::InvalidSize {
                    expected: chunks.len(),
                    actual: plan.chunk_index() + 1,
                },
            )?,
        };
        let (rows, remainder) = centered.as_chunks::<D>();
        if !remainder.is_empty() {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: centered.len(),
            });
        }
        let plane_count = rows
            .len()
            .checked_mul(plan.num_digits_fold())
            .ok_or_else(|| AkitaError::InvalidSetup("Z plane count overflow".into()))?;
        let mut planes = vec![[0i8; D]; plane_count];
        // Each row owns a disjoint `num_digits_fold`-wide plane window, so the
        // decomposition is embarrassingly parallel and byte-identical either
        // way. Keep the parallel form this path had before the kernel move.
        let num_digits_fold = plan.num_digits_fold();
        let log_basis_open = plan.log_basis_open();
        cfg_iter!(rows)
            .zip(cfg_chunks_mut!(&mut planes, num_digits_fold))
            .for_each(|(row, row_planes)| {
                balanced_decompose_centered_i32_i8_into(row, row_planes, log_basis_open);
            });
        let expected_planes = plan
            .num_positions_per_block()
            .checked_mul(plan.num_digits_inner())
            .and_then(|count| count.checked_mul(plan.num_digits_fold()))
            .ok_or_else(|| AkitaError::InvalidSetup("witness Z plane count overflow".into()))?;
        let range = plan.range();
        if planes.len() != expected_planes || planes.as_flattened().len() != range.len() {
            return Err(AkitaError::InvalidSize {
                expected: range.len(),
                actual: planes.as_flattened().len(),
            });
        }
        builder.write_at(range.start, planes.as_flattened())
    }

    pub(crate) fn a_relation_quotients<const D: usize>(
        &self,
        backend: &crate::opaque::CpuBackend,
        prepared: &crate::opaque::CpuPreparedSetup<F>,
        n_a: usize,
        log_basis_open: u32,
        log_basis_outer: u32,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
    where
        F: CanonicalEncoding,
    {
        self.ensure_ring_dim::<D>()?;
        let (z, remainder) = self.global.centered_coeffs_flat().as_chunks::<D>();
        if !remainder.is_empty() {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.global.centered_coeffs_flat().len(),
            });
        }
        let rows = crate::arithmetic::ring_switch::relation_b_a_rows(
            backend,
            prepared,
            &[],
            z,
            self.global.centered_inf_norm(),
            RingSwitchRelationPlan {
                n_d: 0,
                n_b: 0,
                n_a,
                log_basis_open,
                log_basis_outer,
            },
        )?;
        if !rows.b_cyclic.is_empty() || rows.a_quotients.len() != n_a {
            return Err(AkitaError::InvalidProof);
        }
        Ok(rows.a_quotients)
    }

    pub(crate) fn z_consistency_high_half<const D: usize>(
        &self,
        point: &RingMultiplierOpeningPoint<F>,
        num_positions_per_block: usize,
        depth_commit: usize,
        log_basis: u32,
    ) -> Result<CyclotomicRing<F, D>, AkitaError>
    where
        F: CanonicalEncoding + akita_serialization::AkitaSerialize + jolt_field::Ring,
    {
        self.ensure_ring_dim::<D>()?;
        let (z, remainder) = self.global.centered_coeffs_flat().as_chunks::<D>();
        let inner_width = num_positions_per_block
            .checked_mul(depth_commit)
            .ok_or_else(|| AkitaError::InvalidSetup("z inner width overflow".into()))?;
        if !remainder.is_empty() || inner_width == 0 || z.len() != inner_width {
            return Err(AkitaError::InvalidInput(
                "ring-multiplier z layout mismatch".into(),
            ));
        }
        if point.position_len() < num_positions_per_block {
            return Err(AkitaError::InvalidInput(
                "ring-multiplier position length mismatch".into(),
            ));
        }
        let gadget = gadget_row_scalars::<F>(depth_commit, log_basis);
        let mut high_half = [F::zero(); D];
        for position in 0..num_positions_per_block {
            let mut block = CyclotomicRing::<F, D>::zero();
            for (digit, scalar) in gadget.iter().enumerate() {
                block += CyclotomicRing::from_coefficients(std::array::from_fn(|coefficient| {
                    F::from_i64(z[position * depth_commit + digit][coefficient] as i64)
                }))
                .scale(scalar);
            }
            point.accumulate_position_product_high_half(position, &block, &mut high_half)?;
        }
        Ok(CyclotomicRing::from_coefficients(high_half))
    }
}

impl<F> crate::opaque::AcceptedFoldHandle for CpuAcceptedFold<F>
where
    F: Field + CanonicalEncoding + Send + Sync + 'static,
{
    fn metadata(&self) -> AcceptedFoldMetadata {
        self.manifest.into()
    }
}

impl<F, const D: usize> FoldRelationKernel<CpuAcceptedFold<F>, F, D> for crate::opaque::CpuBackend
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + jolt_field::Ring,
{
    fn a_relation_from_fold(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        fold: &CpuAcceptedFold<F>,
        plan: &ValidatedFoldRelationPlan<'_, F>,
    ) -> Result<FoldRelationOutput<F>, AkitaError> {
        let prepared = prepared.ok_or_else(|| {
            AkitaError::InvalidInput("fold relation requires prepared setup".into())
        })?;
        let a_quotients = fold
            .a_relation_quotients::<D>(
                self,
                prepared,
                plan.n_a(),
                plan.log_basis_open(),
                plan.log_basis_outer(),
            )?
            .into_iter()
            .map(|row| {
                RelationQuotientRow::new(
                    akita_types::RelationRowGeometry::native(D)?,
                    row.coefficients().to_vec(),
                )
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let Some(trace) = plan.evaluation_trace() else {
            return Ok(FoldRelationOutput::SubringCoefficientPacking { a_quotients });
        };
        let consistency = if trace.multiplier_point().is_constant() {
            CyclotomicRing::<F, D>::zero()
        } else {
            fold.z_consistency_high_half::<D>(
                trace.multiplier_point(),
                trace.num_positions_per_block(),
                trace.depth_commit(),
                trace.log_basis_inner(),
            )?
        };
        Ok(FoldRelationOutput::EvaluationTrace {
            a_quotients,
            z_consistency_high_half: RelationQuotientRow::new(
                akita_types::RelationRowGeometry::native(D)?,
                consistency.coefficients().to_vec(),
            )?,
        })
    }
}

fn admit_fold_response(
    witness: &DecomposeFoldWitness,
    plan: &ValidatedFoldProbePlan<'_>,
    observed_l2_sq: &mut Option<u128>,
) -> Result<bool, AkitaError> {
    let acceptance = plan.acceptance();
    let (min, max) = witness.centered_signed_extrema();
    if (min < 0 && u128::from(min.unsigned_abs()) > acceptance.digit_negative_abs_bound())
        || (max > 0 && u128::from(max.unsigned_abs()) > acceptance.digit_positive_bound())
    {
        return Ok(false);
    }
    let response_l2_sq_cap = acceptance.response_l2_sq_cap();
    if response_l2_sq_cap.is_none() && !response_model_diagnostics_enabled() {
        return Ok(true);
    }
    // `None` after measurement starts means diagnostics overflowed without a
    // security cap. That must not change admission behavior.
    let Some(previous_l2) = *observed_l2_sq else {
        return Ok(response_l2_sq_cap.is_none());
    };
    let Some(chunk_l2) = akita_types::sis::checked_centered_l2_sq(witness.centered_coeffs_flat())
    else {
        if response_l2_sq_cap.is_some() {
            return Err(AkitaError::InvalidInput(
                "fold response L2 norm overflow".into(),
            ));
        }
        *observed_l2_sq = None;
        return Ok(true);
    };
    let Some(total) = previous_l2.checked_add(chunk_l2) else {
        if response_l2_sq_cap.is_some() {
            return Err(AkitaError::InvalidInput(
                "fold response L2 norm accumulation overflow".into(),
            ));
        }
        *observed_l2_sq = None;
        return Ok(true);
    };
    *observed_l2_sq = Some(total);
    Ok(response_l2_sq_cap.is_none_or(|cap| total <= cap))
}

pub(crate) fn aggregate_decompose_fold_witnesses<const D: usize>(
    witnesses: impl IntoIterator<Item = Result<DecomposeFoldWitness, AkitaError>>,
) -> Result<DecomposeFoldWitness, AkitaError> {
    let mut witnesses = witnesses.into_iter();
    let Some(first) = witnesses.next() else {
        return Err(AkitaError::InvalidInput(
            "batched decompose_fold requires at least one witness".to_string(),
        ));
    };
    let first = first?;
    first.ensure_ring_dim::<D>()?;
    let row_count = first.row_count();
    let mut centered_coeffs = first.into_centered_coeffs_flat();

    for witness in witnesses {
        let witness = witness?;
        add_decompose_fold_witness::<D>(&mut centered_coeffs, row_count, &witness)?;
    }

    DecomposeFoldWitness::from_centered_flat::<D>(centered_coeffs)
}

fn add_decompose_fold_witness<const D: usize>(
    centered_coeffs: &mut [i32],
    row_count: usize,
    witness: &DecomposeFoldWitness,
) -> Result<(), AkitaError> {
    witness.ensure_ring_dim::<D>()?;
    if witness.row_count() != row_count {
        return Err(AkitaError::InvalidInput(
            "batched decompose_fold witness length mismatch".into(),
        ));
    }
    for (dst, src) in centered_coeffs
        .iter_mut()
        .zip(witness.centered_coeffs_flat())
    {
        *dst = dst.checked_add(*src).ok_or_else(|| {
            AkitaError::InvalidInput("batched decompose_fold centered coefficient overflow".into())
        })?;
    }
    Ok(())
}

fn aggregate_chunk_responses<const D: usize>(
    chunks: &[DecomposeFoldWitness],
) -> Result<DecomposeFoldWitness, AkitaError> {
    let Some((first, rest)) = chunks.split_first() else {
        return Err(AkitaError::InvalidInput(
            "chunked decompose_fold requires at least one response".into(),
        ));
    };
    first.ensure_ring_dim::<D>()?;
    let row_count = first.row_count();
    let mut centered_coeffs = first.centered_coeffs_flat().to_vec();
    for witness in rest {
        add_decompose_fold_witness::<D>(&mut centered_coeffs, row_count, witness)?;
    }
    DecomposeFoldWitness::from_centered_flat::<D>(centered_coeffs)
}

/// Private CPU responses produced by one batch-fold dispatch.
pub(crate) struct CpuFoldResponses {
    pub(crate) global: DecomposeFoldWitness,
    pub(crate) chunks: Option<Vec<DecomposeFoldWitness>>,
}

impl CpuFoldResponses {
    pub(crate) fn sparse(global: DecomposeFoldWitness) -> Self {
        Self {
            global,
            chunks: None,
        }
    }

    pub(crate) fn chunked<const D: usize>(
        chunks: Vec<DecomposeFoldWitness>,
    ) -> Result<Self, AkitaError> {
        let global = aggregate_chunk_responses::<D>(&chunks)?;
        Ok(Self {
            global,
            chunks: Some(chunks),
        })
    }
}

#[derive(Clone, Copy)]
struct AcceptedFoldManifest {
    ring_dimension: usize,
    response_coefficients: usize,
    opening_method: akita_types::OpeningMethod,
    source_claims: usize,
    live_blocks: usize,
    positions_per_block: usize,
    num_digits: usize,
    log_basis: u32,
    num_chunks: usize,
}

/// Consumer-owned accepted fold transported by protocol orchestration.
///
/// Implementations expose metadata only. Coefficients and native handles stay
/// entirely inside the consumer's concrete type.
impl<F> CpuAcceptedFold<F>
where
    F: Field + CanonicalEncoding + 'static,
{
    fn new<O, const D: usize>(
        _backend: &O,
        _prepared: Option<&O::PreparedSetup>,
        global: DecomposeFoldWitness,
        chunks: Option<Vec<Vec<i32>>>,
        plan: &ValidatedFoldProbePlan<'_>,
    ) -> Result<Self, AkitaError>
    where
        O: ComputeBackendSetup<F>,
    {
        let response_coefficients = plan
            .num_positions_per_block()
            .checked_mul(plan.num_digits())
            .and_then(|rows| rows.checked_mul(D))
            .ok_or_else(|| AkitaError::InvalidInput("fold response size overflow".into()))?;
        Ok(Self {
            global,
            chunks,
            challenges: plan.challenges().clone(),
            manifest: AcceptedFoldManifest {
                ring_dimension: D,
                response_coefficients,
                opening_method: plan.opening_method(),
                source_claims: plan.challenges().num_claims(),
                live_blocks: plan.challenges().num_live_blocks_per_claim(),
                positions_per_block: plan.num_positions_per_block(),
                num_digits: plan.num_digits(),
                log_basis: plan.log_basis(),
                num_chunks: plan.geometry().chunk_ranges().map_or(1, <[_]>::len),
            },
            binding: crate::opaque::OperationBinding::legacy_unscoped(),
            _field: PhantomData,
        })
    }

    #[cfg(test)]
    pub(crate) fn from_cpu_for_test<const D: usize>(
        _ctx: &crate::opaque::OperationCtx<'_, F, crate::opaque::CpuBackend>,
        global: DecomposeFoldWitness,
        params: &akita_types::GroupOpenPhaseParams,
        source_claims: usize,
        num_chunks: usize,
    ) -> Self
    where
        F: akita_serialization::AkitaSerialize + jolt_field::Ring,
    {
        let response_coefficients = global.centered_coeffs_flat().len();
        Self {
            global,
            chunks: None,
            challenges: akita_challenges::Challenges::from_sparse(Vec::new(), 0, 0)
                .expect("test challenges"),
            manifest: AcceptedFoldManifest {
                ring_dimension: D,
                response_coefficients,
                opening_method: params.opening_method(),
                source_claims,
                live_blocks: params.num_live_blocks(),
                positions_per_block: params.num_positions_per_block(),
                num_digits: params.num_digits_inner(),
                log_basis: params.log_basis_inner(),
                num_chunks,
            },
            binding: crate::opaque::OperationBinding::legacy_unscoped(),
            _field: PhantomData,
        }
    }

    pub(crate) fn response_coefficient_len(&self) -> usize {
        self.manifest.response_coefficients
    }

    pub(super) const fn manifest_num_chunks(&self) -> usize {
        self.manifest.num_chunks
    }

    pub(crate) fn validate_for_build(
        &self,
        build_binding: &crate::opaque::OperationBinding,
        params: &akita_types::GroupOpenPhaseParams,
        source_claims: usize,
        expected_chunks: usize,
    ) -> Result<(), AkitaError> {
        self.binding.validate_lineage(build_binding)?;
        if self.manifest.opening_method != params.opening_method()
            || self.manifest.source_claims != source_claims
            || self.manifest.live_blocks != params.num_live_blocks()
            || self.manifest.positions_per_block != params.num_positions_per_block()
            || self.manifest.num_digits != params.num_digits_inner()
            || self.manifest.log_basis != params.log_basis_inner()
            || self.manifest.num_chunks != expected_chunks
        {
            return Err(AkitaError::InvalidInput(
                "accepted fold disagrees with the recursive-witness plan".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn relation<O, const D: usize>(
        &self,
        ctx: &crate::opaque::OperationCtx<'_, F, O>,
        plan: &ValidatedFoldRelationPlan<'_, F>,
    ) -> Result<FoldRelationOutput<F>, AkitaError>
    where
        O: ComputeBackendSetup<F> + FoldRelationKernel<CpuAcceptedFold<F>, F, D>,
    {
        self.ensure_ring_dim::<D>()?;
        let output = ctx
            .backend()
            .a_relation_from_fold(Some(ctx.prepared()), self, plan)?;
        let (rows, consistency) = match &output {
            FoldRelationOutput::EvaluationTrace {
                a_quotients,
                z_consistency_high_half,
            } if matches!(
                self.manifest.opening_method,
                akita_types::OpeningMethod::EvaluationTrace
            ) =>
            {
                (a_quotients, Some(z_consistency_high_half))
            }
            FoldRelationOutput::SubringCoefficientPacking { a_quotients }
                if matches!(
                    self.manifest.opening_method,
                    akita_types::OpeningMethod::SubringCoefficientPacking { .. }
                ) =>
            {
                (a_quotients, None)
            }
            _ => return Err(AkitaError::InvalidProof),
        };
        if rows.len() != plan.n_a()
            || rows.iter().chain(consistency).any(|row| {
                row.geometry().physical_coefficient_width() != self.manifest.ring_dimension
            })
        {
            return Err(AkitaError::InvalidProof);
        }
        Ok(output)
    }
}

pub(crate) fn cpu_terminal_probe<S, F, B, const D: usize>(
    backend: &B,
    prepared: Option<&B::PreparedSetup>,
    source: S,
    plan: &ValidatedTerminalFoldProbePlan<'_>,
) -> Result<FoldProbeOutcome<CpuAcceptedTerminalFold<F>>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + 'static,
    B: OpeningBatchKernel<S, F, D>,
{
    let responses = backend.decompose_fold_batch(
        prepared,
        source,
        DecomposeFoldBatchPlan::Sparse {
            challenges: plan.challenges().as_slice(),
            num_positions_per_block: plan.num_positions_per_block(),
            num_digits: plan.num_digits(),
            log_basis: plan.log_basis(),
        },
    )?;
    if responses.chunks.is_some() {
        return Err(AkitaError::InvalidInput(
            "terminal fold backend returned chunk responses".into(),
        ));
    }
    let Some((fold, observed_l2_sq)) =
        CpuAcceptedTerminalFold::admit::<B, D>(backend, prepared, responses.global, plan)?
    else {
        return Ok(FoldProbeOutcome::Rejected);
    };
    Ok(FoldProbeOutcome::Accepted {
        fold_handle: fold,
        diagnostics: FoldProbeDiagnostics::new(observed_l2_sq),
    })
}

pub(crate) fn cpu_fold_probe<S, F, B, const D: usize>(
    backend: &B,
    prepared: Option<&B::PreparedSetup>,
    source: S,
    plan: &ValidatedFoldProbePlan<'_>,
) -> Result<FoldProbeOutcome<CpuAcceptedFold<F>>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + jolt_field::Ring + 'static,
    B: OpeningBatchKernel<S, F, D>,
{
    let batch_plan = match plan.geometry() {
        FoldProbeGeometry::Sparse => DecomposeFoldBatchPlan::Sparse {
            challenges: plan.challenges().as_slice(),
            num_positions_per_block: plan.num_positions_per_block(),
            num_digits: plan.num_digits(),
            log_basis: plan.log_basis(),
        },
        FoldProbeGeometry::SparseChunked { chunk_ranges } => {
            DecomposeFoldBatchPlan::SparseChunked {
                challenges: plan.challenges(),
                chunk_ranges,
                num_positions_per_block: plan.num_positions_per_block(),
                num_digits: plan.num_digits(),
                log_basis: plan.log_basis(),
            }
        }
    };
    let responses = backend.decompose_fold_batch(prepared, source, batch_plan)?;
    let mut observed_l2_sq = (plan.acceptance().response_l2_sq_cap().is_some()
        || response_model_diagnostics_enabled())
    .then_some(0);
    let chunks = match responses.chunks {
        None => {
            if !admit_fold_response(&responses.global, plan, &mut observed_l2_sq)? {
                return Ok(FoldProbeOutcome::Rejected);
            }
            None
        }
        Some(chunks) => {
            let mut centered = Vec::with_capacity(chunks.len());
            for chunk in chunks {
                if !admit_fold_response(&chunk, plan, &mut observed_l2_sq)? {
                    return Ok(FoldProbeOutcome::Rejected);
                }
                centered.push(chunk.into_centered_coeffs_flat());
            }
            Some(centered)
        }
    };
    Ok(FoldProbeOutcome::Accepted {
        fold_handle: CpuAcceptedFold::new::<B, D>(
            backend,
            prepared,
            responses.global,
            chunks,
            plan,
        )?,
        diagnostics: FoldProbeDiagnostics::new(observed_l2_sq),
    })
}

impl<S, F, const D: usize> TerminalFoldResponseKernel<S, F, D> for crate::opaque::CpuBackend
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + 'static,
    crate::opaque::CpuBackend: OpeningBatchKernel<S, F, D>,
{
    fn probe_terminal(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        plan: &ValidatedTerminalFoldProbePlan<'_>,
    ) -> Result<FoldProbeOutcome<CpuAcceptedTerminalFold<F>>, AkitaError> {
        cpu_terminal_probe::<S, F, Self, D>(self, prepared, source, plan)
    }

    fn encode_terminal_z(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        fold: &CpuAcceptedTerminalFold<F>,
        _plan: &ValidatedTerminalZEncodingPlan,
    ) -> Result<Vec<u8>, AkitaError> {
        fold.encode::<Self, D>(self, prepared)
    }
}

impl<S, F, const D: usize> FoldResponseKernel<S, F, D> for crate::opaque::CpuBackend
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + jolt_field::Ring + 'static,
    crate::opaque::CpuBackend: OpeningBatchKernel<S, F, D>,
{
    fn probe(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        plan: &ValidatedFoldProbePlan<'_>,
    ) -> Result<FoldProbeOutcome<CpuAcceptedFold<F>>, AkitaError> {
        cpu_fold_probe::<S, F, Self, D>(self, prepared, source, plan)
    }
}

impl<F> FoldHandleBackend<F> for crate::opaque::CpuBackend
where
    F: Field + CanonicalEncoding,
{
    type AcceptedFold = CpuAcceptedFold<F>;
    type AcceptedTerminalFold = CpuAcceptedTerminalFold<F>;
}

impl From<AcceptedFoldManifest> for AcceptedFoldMetadata {
    fn from(manifest: AcceptedFoldManifest) -> Self {
        Self::try_new(
            manifest.ring_dimension,
            manifest.response_coefficients,
            manifest.num_chunks,
        )
        .expect("accepted CPU fold has validated public geometry")
    }
}
