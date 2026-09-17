use crate::compute::backend::ComputeBackendSetup;
use crate::compute::operation_plans::{
    DecomposeFoldBatchPlan, DecomposeFoldPlan, FoldProbeDiagnostics, FoldProbeGeometry,
    FoldProbeOutcome, OpeningFoldOutput, OpeningFoldPlan, RingSwitchRelationPlan,
    SubringCoefficientPackingPartials, SubringCoefficientPackingPlan, ValidatedFoldProbePlan,
};
use crate::compute::plans::RingSwitchRelationRows;
use crate::DecomposeFoldWitness;
use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced};

/// CPU reference state retained behind an accepted fold handle.
pub struct CpuAcceptedFold<F: Field> {
    global: DecomposeFoldWitness<F>,
    chunks: Option<Vec<Vec<i32>>>,
    backend_id: usize,
    prepared_id: Option<usize>,
    ring_dimension: usize,
    opening_method: akita_types::OpeningMethod,
}

/// CPU reference state retained behind an accepted terminal-fold handle.
pub struct CpuAcceptedTerminalFold<F: Field> {
    witness: DecomposeFoldWitness<F>,
    backend_id: usize,
    prepared_id: Option<usize>,
}

impl<F> CpuAcceptedTerminalFold<F>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn admit<B: ComputeBackendSetup<F>>(
        backend: &B,
        prepared: Option<&B::PreparedSetup>,
        witness: DecomposeFoldWitness<F>,
        coordinate_count: usize,
        linf_cap: Option<u128>,
        l2_sq_cap: Option<u128>,
        collect_l2_diagnostic: bool,
        rice_low_bits: u32,
        payload_bytes: usize,
    ) -> Result<Option<(Self, Option<u128>)>, AkitaError> {
        let centered = witness.centered_coeffs_flat();
        if centered.len() != coordinate_count {
            return Err(AkitaError::InvalidSize {
                expected: coordinate_count,
                actual: centered.len(),
            });
        }
        if let Some(cap) = linf_cap {
            if akita_types::golomb_rice_values_within_cap(centered, cap).is_err() {
                return Ok(None);
            }
        } else if centered.iter().any(|value| i16::try_from(*value).is_err()) {
            return Ok(None);
        }
        let observed_l2_sq = if l2_sq_cap.is_some() || collect_l2_diagnostic {
            let Some(value) = akita_types::sis::checked_centered_l2_sq(centered) else {
                return Err(AkitaError::InvalidInput(
                    "terminal fold response L2 overflow".into(),
                ));
            };
            if l2_sq_cap.is_some_and(|cap| value > cap) {
                return Ok(None);
            }
            Some(value)
        } else {
            None
        };
        let zigzag_width =
            akita_types::golomb_rice_zigzag_width(linf_cap.unwrap_or(i16::MAX as u128));
        if akita_types::golomb_rice_total_wire_bits(centered, rice_low_bits, zigzag_width)?
            > payload_bytes.saturating_mul(8)
        {
            return Ok(None);
        }
        Ok(Some((
            Self {
                witness,
                backend_id: backend as *const B as *const () as usize,
                prepared_id: prepared
                    .map(|value| value as *const B::PreparedSetup as *const () as usize),
            },
            observed_l2_sq,
        )))
    }

    pub(crate) fn build_response<B: ComputeBackendSetup<F>>(
        self,
        backend: &B,
        prepared: Option<&B::PreparedSetup>,
        params: &akita_types::TerminalFoldParams,
        shape: &akita_types::TerminalResponseShape,
        e_folded: &akita_types::RingVec<F>,
        t_fields: akita_types::RingVec<F>,
    ) -> Result<akita_types::TerminalResponse<F>, AkitaError> {
        let backend_id = backend as *const B as *const () as usize;
        let prepared_id =
            prepared.map(|value| value as *const B::PreparedSetup as *const () as usize);
        if backend_id != self.backend_id || prepared_id != self.prepared_id {
            return Err(AkitaError::InvalidInput(
                "accepted terminal fold used with a different operation context".into(),
            ));
        }
        akita_types::build_terminal_response(
            params,
            shape,
            e_folded,
            t_fields,
            self.witness.centered_coeffs_flat(),
        )
    }
}

impl<F: Field> CpuAcceptedFold<F> {
    #[cfg(test)]
    pub(crate) fn from_global(global: DecomposeFoldWitness<F>) -> Self {
        let ring_dimension = global.ring_dim();
        Self {
            global,
            chunks: None,
            backend_id: 0,
            prepared_id: None,
            ring_dimension,
            opening_method: akita_types::OpeningMethod::EvaluationTrace,
        }
    }

    pub(crate) fn into_parts(self) -> (DecomposeFoldWitness<F>, Option<Vec<Vec<i32>>>) {
        (self.global, self.chunks)
    }

    pub(crate) fn ensure_ring_dim<const D: usize>(&self) -> Result<(), AkitaError> {
        if self.ring_dimension != D {
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

    pub(crate) fn validate_context<B>(
        &self,
        backend: &B,
        prepared: Option<&B::PreparedSetup>,
        opening_method: akita_types::OpeningMethod,
    ) -> Result<(), AkitaError>
    where
        B: ComputeBackendSetup<F>,
        F: CanonicalEncoding,
    {
        let backend_id = backend as *const B as *const () as usize;
        let prepared_id =
            prepared.map(|value| value as *const B::PreparedSetup as *const () as usize);
        if self.backend_id != 0
            && (self.backend_id != backend_id
                || self.prepared_id != prepared_id
                || self.opening_method != opening_method)
        {
            return Err(AkitaError::InvalidInput(
                "accepted fold used with a different operation context".into(),
            ));
        }
        Ok(())
    }

    #[cfg(feature = "response-model-diagnostics")]
    pub(crate) fn num_chunks(&self) -> usize {
        self.chunks.as_ref().map_or(1, Vec::len)
    }

    #[cfg(feature = "response-model-diagnostics")]
    pub(crate) fn response_coefficient_count(&self) -> usize {
        self.chunks.as_ref().map_or_else(
            || self.global.centered_coeffs_flat().len(),
            |chunks| chunks.iter().map(Vec::len).sum(),
        )
    }
}

fn admit_fold_response<F: Field>(
    witness: &DecomposeFoldWitness<F>,
    plan: &ValidatedFoldProbePlan<'_>,
    observed_l2_sq: &mut Option<u128>,
) -> Result<bool, AkitaError> {
    let acceptance = plan.acceptance();
    let (min, max) = witness.centered_signed_extrema();
    if u128::from(min.unsigned_abs()) > acceptance.digit_negative_abs_bound()
        || u128::from(max.unsigned_abs()) > acceptance.digit_positive_bound()
    {
        return Ok(false);
    }
    if acceptance.response_l2_sq_cap().is_none() && !acceptance.collect_l2_diagnostic() {
        return Ok(true);
    }
    let chunk_l2 = witness
        .centered_coeffs_flat()
        .iter()
        .try_fold(0u128, |sum, coefficient| {
            let magnitude = u128::from(coefficient.unsigned_abs());
            magnitude
                .checked_mul(magnitude)
                .and_then(|square| sum.checked_add(square))
                .ok_or_else(|| AkitaError::InvalidInput("fold response L2 norm overflow".into()))
        })?;
    let total = observed_l2_sq
        .unwrap_or(0)
        .checked_add(chunk_l2)
        .ok_or_else(|| {
            AkitaError::InvalidInput("fold response L2 norm accumulation overflow".into())
        })?;
    *observed_l2_sq = Some(total);
    Ok(acceptance
        .response_l2_sq_cap()
        .is_none_or(|cap| total <= cap))
}

pub(crate) fn aggregate_decompose_fold_witnesses<F: Field, const D: usize>(
    witnesses: impl IntoIterator<Item = Result<DecomposeFoldWitness<F>, AkitaError>>,
) -> Result<DecomposeFoldWitness<F>, AkitaError> {
    let mut witnesses = witnesses.into_iter();
    let Some(first) = witnesses.next() else {
        return Err(AkitaError::InvalidInput(
            "batched decompose_fold requires at least one witness".to_string(),
        ));
    };
    let first = first?;
    first.ensure_ring_dim::<D>()?;
    let row_count = first.row_count();
    let (z_folded_rings, mut centered_coeffs) = first.into_owned_flat_parts();
    let mut z_folded_coeffs = z_folded_rings.into_coeffs();

    for witness in witnesses {
        let witness = witness?;
        witness.ensure_ring_dim::<D>()?;
        if witness.row_count() != row_count {
            return Err(AkitaError::InvalidInput(
                "batched decompose_fold witness length mismatch".to_string(),
            ));
        }
        for (dst, src) in z_folded_coeffs
            .iter_mut()
            .zip(witness.z_folded_rings.coeffs())
        {
            *dst += *src;
        }
        for (dst, src) in centered_coeffs
            .iter_mut()
            .zip(witness.centered_coeffs_flat())
        {
            *dst = dst.checked_add(*src).ok_or_else(|| {
                AkitaError::InvalidInput(
                    "batched decompose_fold centered coefficient overflow".to_string(),
                )
            })?;
        }
    }

    DecomposeFoldWitness::from_owned_flat_parts::<D>(
        akita_types::RingVec::from_coeffs_with_ring_dim(z_folded_coeffs, D)?,
        centered_coeffs,
    )
}

/// Private CPU responses produced by one batch-fold dispatch.
pub struct CpuFoldResponses<F: Field> {
    pub(crate) global: DecomposeFoldWitness<F>,
    pub(crate) chunks: Option<Vec<DecomposeFoldWitness<F>>>,
}

impl<F: Field> CpuFoldResponses<F> {
    pub(crate) fn sparse(global: DecomposeFoldWitness<F>) -> Self {
        Self {
            global,
            chunks: None,
        }
    }

    pub(crate) fn chunked<const D: usize>(
        chunks: Vec<DecomposeFoldWitness<F>>,
    ) -> Result<Self, AkitaError> {
        let global = aggregate_decompose_fold_witnesses::<F, D>(chunks.iter().map(|chunk| {
            DecomposeFoldWitness::from_owned_flat_parts::<D>(
                chunk.z_folded_rings.clone(),
                chunk.centered_coeffs_flat().to_vec(),
            )
        }))?;
        Ok(Self {
            global,
            chunks: Some(chunks),
        })
    }
}

/// Fused ring-switch relation-rows kernel over a borrowed relation view `S`.
pub trait RingSwitchRelationKernel<S, F, const D: usize>: ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
{
    /// Fused D rows in both domains, B cyclic rows, and A-side quotient rows.
    fn relation_rows(
        &self,
        prepared: &Self::PreparedSetup,
        source: S,
        plan: RingSwitchRelationPlan,
    ) -> Result<RingSwitchRelationRows<F, D>, AkitaError>;
}

/// Opening fold / decompose-fold kernel over a borrowed opening view `S`.
///
/// `prepared` is optional because some opening folds do not need setup-owned
/// state; setup-dependent work stays explicitly tied to the backend context.
pub trait OpeningFoldKernel<S, F, const D: usize>: ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
{
    /// Fused fold + evaluation in one pass over the source.
    fn evaluate_and_fold(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        plan: OpeningFoldPlan<'_, F>,
    ) -> Result<OpeningFoldOutput<F, D>, AkitaError>;

    /// Decompose + challenge-fold step.
    fn decompose_fold(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<DecomposeFoldWitness<F>, AkitaError>;
}

/// Batched decompose-fold kernel over a borrowed opening-batch view `S`.
///
/// Implementations return the final aggregate witness. A backend without a
/// fused path is responsible for folding its source polynomials individually
/// and aggregating them before returning.
pub trait OpeningBatchKernel<S, F, const D: usize>: ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
{
    /// Batched decompose-fold at one opening point.
    fn decompose_fold_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<CpuFoldResponses<F>, AkitaError>;
}

/// Backend-owned fold computation and admission over one validated plan.
pub trait FoldResponseKernel<S, F, const D: usize>: ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
{
    type AcceptedFold: Send + Sync + 'static;

    fn probe(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        plan: &ValidatedFoldProbePlan<'_>,
    ) -> Result<FoldProbeOutcome<Self::AcceptedFold>, AkitaError>;
}

impl<S, F, B, const D: usize> FoldResponseKernel<S, F, D> for B
where
    F: Field + CanonicalEncoding + 'static,
    B: OpeningBatchKernel<S, F, D>,
{
    type AcceptedFold = CpuAcceptedFold<F>;

    fn probe(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        plan: &ValidatedFoldProbePlan<'_>,
    ) -> Result<FoldProbeOutcome<Self::AcceptedFold>, AkitaError> {
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
        let responses = self.decompose_fold_batch(prepared, source, batch_plan)?;
        let mut observed_l2_sq = None;
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
                    centered.push(chunk.centered_coeffs_flat().to_vec());
                }
                Some(centered)
            }
        };
        Ok(FoldProbeOutcome::Accepted {
            fold: CpuAcceptedFold {
                global: responses.global,
                chunks,
                backend_id: self as *const B as *const () as usize,
                prepared_id: prepared
                    .map(|value| value as *const B::PreparedSetup as *const () as usize),
                ring_dimension: D,
                opening_method: plan.opening_method(),
            },
            diagnostics: FoldProbeDiagnostics {
                observed_l2_sq: plan
                    .acceptance()
                    .collect_l2_diagnostic()
                    .then_some(observed_l2_sq)
                    .flatten(),
            },
        })
    }
}

/// Tensor projection kernel over a borrowed tensor view `S` for opening at an
/// extension-field point of type `E`.
pub trait TensorProjectionKernel<S, F, E, const D: usize>: ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    /// Tensor-column partials at one logical point.
    fn column_partials(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        logical_point: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<F>;

    /// Tensor-packed EOR witness.
    fn packed_witness(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
    ) -> Result<Vec<E>, AkitaError>;
}

/// Batched tensor projection kernel over a borrowed tensor-batch view `S`.
pub trait TensorProjectionBatchKernel<S, F, E, const D: usize>: ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    /// Tensor-column partials for a same-point batch.
    fn column_partials_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: MulBaseUnreduced<F>;
}

/// Coefficient-packing projection over a borrowed same-shape source batch.
pub trait SubringCoefficientPackingBatchKernel<S, F, E, const D: usize>:
    ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    /// Return one canonical base-field partial buffer per claim.
    ///
    /// Every returned buffer uses
    /// `[block][extension coordinate][subring coefficient]` order.
    fn coefficient_packing_partials_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        plan: SubringCoefficientPackingPlan<'_, E>,
    ) -> Result<Vec<SubringCoefficientPackingPartials<F>>, AkitaError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::CpuBackend;
    use akita_challenges::{Challenges, SparseChallenge};
    use std::sync::atomic::{AtomicUsize, Ordering};

    type F = jolt_field::Prime128Offset275;
    const D: usize = 4;
    static BATCH_CALLS: AtomicUsize = AtomicUsize::new(0);
    static SOURCE_TRAVERSALS: AtomicUsize = AtomicUsize::new(0);

    struct RecordingBatch;

    impl OpeningBatchKernel<RecordingBatch, F, D> for CpuBackend {
        fn decompose_fold_batch(
            &self,
            _prepared: Option<&Self::PreparedSetup>,
            _source: RecordingBatch,
            plan: DecomposeFoldBatchPlan<'_>,
        ) -> Result<CpuFoldResponses<F>, AkitaError> {
            BATCH_CALLS.fetch_add(1, Ordering::Relaxed);
            SOURCE_TRAVERSALS.fetch_add(1, Ordering::Relaxed);
            let DecomposeFoldBatchPlan::SparseChunked { chunk_ranges, .. } = plan else {
                return Err(AkitaError::InvalidInput(
                    "expected chunked recording plan".into(),
                ));
            };
            CpuFoldResponses::chunked::<D>(
                chunk_ranges
                    .iter()
                    .map(|_| {
                        DecomposeFoldWitness::from_parts::<D>(
                            vec![akita_algebra::CyclotomicRing::zero()],
                            vec![[0; D]],
                        )
                    })
                    .collect(),
            )
        }
    }

    #[test]
    fn chunked_probe_dispatches_and_traverses_once() {
        for (blocks, chunks) in [(8, 2), (8, 4), (8, 8), (2, 8)] {
            BATCH_CALLS.store(0, Ordering::Relaxed);
            SOURCE_TRAVERSALS.store(0, Ordering::Relaxed);
            let challenges = Challenges::from_sparse(
                vec![
                    SparseChallenge {
                        positions: Vec::new().into(),
                        coeffs: Vec::new().into(),
                    };
                    blocks
                ],
                blocks,
                1,
            )
            .unwrap();
            let ranges = akita_types::dyadic_block_ranges(blocks, chunks).unwrap();
            let plan = ValidatedFoldProbePlan::new::<D>(
                &challenges,
                1,
                blocks,
                FoldProbeGeometry::SparseChunked {
                    chunk_ranges: &ranges,
                },
                1,
                1,
                1,
                akita_types::OpeningMethod::EvaluationTrace,
                crate::compute::ValidatedFoldAcceptancePlan::new(u128::MAX, u128::MAX, None, false),
            )
            .unwrap();
            let outcome =
                FoldResponseKernel::probe(&CpuBackend::DEFAULT, None, RecordingBatch, &plan)
                    .unwrap();
            assert!(matches!(outcome, FoldProbeOutcome::Accepted { .. }));
            assert_eq!(BATCH_CALLS.load(Ordering::Relaxed), 1);
            assert_eq!(SOURCE_TRAVERSALS.load(Ordering::Relaxed), 1);
        }
    }
}
