use std::array::from_fn;
use std::sync::Arc;

use akita_algebra::ring::cyclotomic::decompose_centering_threshold;
use akita_algebra::CyclotomicRing;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, MulBaseUnreduced,
};
use akita_types::{
    AkitaExpandedSetup, CleartextWitnessProof, FpExtEncoding, RingVec, SetupPrefixSlot,
};

use crate::backend::poly_helpers::{
    balanced_ring_decompose_fold_partitioned, build_decompose_fold_witness, DecomposeParams,
};
use crate::backend::{RecursiveWitnessFlat, SuffixWitnessBatchView, SuffixWitnessView};
use crate::compute::{
    BatchDecomposeFoldOutcome, CpuBackend, DecomposeFoldBatchPlan, DecomposeFoldPlan,
    DirectRootWitnessSource, OpeningBatchKernel, OpeningFoldKernel, OpeningFoldOutput,
    OpeningFoldPlan, RootOpeningSource, RootPolyMeta, RootPolyShape, RootTensorSource,
    TensorPackedWitness, TensorProjectionBatchKernel, TensorProjectionKernel,
};
use crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness;
use crate::RootTensorProjectionPoly;

#[doc(hidden)]
#[derive(Clone)]
pub enum RecursiveFoldSource<F: FieldCore> {
    SetupPrefix {
        expanded: Arc<AkitaExpandedSetup<F>>,
        slot: Arc<SetupPrefixSlot<F>>,
    },
    Witness(Arc<RecursiveWitnessFlat>),
}

impl<F: FieldCore> RecursiveFoldSource<F> {
    pub(crate) fn setup_prefix(
        expanded: Arc<AkitaExpandedSetup<F>>,
        slot: Arc<SetupPrefixSlot<F>>,
    ) -> Self {
        Self::SetupPrefix { expanded, slot }
    }

    pub(crate) fn witness(witness: Arc<RecursiveWitnessFlat>) -> Self {
        Self::Witness(witness)
    }
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub enum RecursiveFoldView<'a, F: FieldCore, const D: usize> {
    SetupPrefix {
        expanded: &'a AkitaExpandedSetup<F>,
        slot: &'a SetupPrefixSlot<F>,
    },
    Witness(SuffixWitnessView<'a, F, D>),
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct RecursiveFoldBatchView<'a, F: FieldCore, const D: usize> {
    polys: &'a [&'a RecursiveFoldSource<F>],
}

impl<F: FieldCore> RootPolyMeta<F> for RecursiveFoldSource<F> {
    fn num_ring_elems(&self) -> usize {
        match self {
            Self::SetupPrefix { slot, .. } => slot.id.n_prefix().unwrap_or(1),
            Self::Witness(witness) => RootPolyMeta::<F>::num_ring_elems(witness.as_ref()),
        }
    }

    fn num_vars(&self) -> usize {
        match self {
            Self::SetupPrefix { slot, .. } => {
                slot.id.n_prefix().unwrap_or(1).trailing_zeros() as usize
            }
            Self::Witness(witness) => RootPolyMeta::<F>::num_vars(witness.as_ref()),
        }
    }
}

impl<F: FieldCore, const D: usize> RootPolyShape<F, D> for RecursiveFoldSource<F> {
    fn num_ring_elems(&self) -> usize {
        match self {
            Self::SetupPrefix { slot, .. } => slot.id.n_prefix().map_or(1, |n| n / D),
            Self::Witness(witness) => RootPolyShape::<F, D>::num_ring_elems(witness.as_ref()),
        }
    }

    fn num_vars(&self) -> usize {
        RootPolyMeta::<F>::num_vars(self)
    }
}

impl<F: FieldCore, const D: usize> RootOpeningSource<F, D> for RecursiveFoldSource<F> {
    type OpeningView<'v>
        = RecursiveFoldView<'v, F, D>
    where
        Self: 'v;

    type OpeningBatchView<'v>
        = RecursiveFoldBatchView<'v, F, D>
    where
        Self: 'v;

    fn opening_view(&self) -> Result<Self::OpeningView<'_>, AkitaError> {
        match self {
            Self::SetupPrefix { expanded, slot } => Ok(RecursiveFoldView::SetupPrefix {
                expanded: expanded.as_ref(),
                slot: slot.as_ref(),
            }),
            Self::Witness(witness) => {
                Ok(RecursiveFoldView::Witness(witness.as_ref().view::<F, D>()?))
            }
        }
    }

    fn opening_batch<'v>(polys: &'v [&'v Self]) -> Result<Self::OpeningBatchView<'v>, AkitaError> {
        Ok(RecursiveFoldBatchView { polys })
    }
}

impl<F: FieldCore, const D: usize> RootTensorSource<F, D> for RecursiveFoldSource<F> {
    type TensorView<'v>
        = RecursiveFoldView<'v, F, D>
    where
        Self: 'v;

    type TensorBatchView<'v>
        = RecursiveFoldBatchView<'v, F, D>
    where
        Self: 'v;

    fn tensor_view(&self) -> Result<Self::TensorView<'_>, AkitaError> {
        self.opening_view()
    }

    fn tensor_batch<'v>(polys: &'v [&'v Self]) -> Result<Self::TensorBatchView<'v>, AkitaError> {
        Ok(RecursiveFoldBatchView { polys })
    }
}

impl<F: FieldCore + CanonicalField, const D: usize> DirectRootWitnessSource<F, D>
    for RecursiveFoldSource<F>
{
    fn direct_root_witness(&self) -> Result<CleartextWitnessProof<F>, AkitaError> {
        match self {
            Self::SetupPrefix { expanded, slot } => Ok(CleartextWitnessProof::FieldElements(
                RingVec::from_coeffs(setup_prefix_field_evals(expanded.as_ref(), slot.as_ref())?),
            )),
            Self::Witness(witness) => {
                DirectRootWitnessSource::<F, D>::direct_root_witness(witness.as_ref())
            }
        }
    }
}

fn setup_prefix_field_evals<F: FieldCore>(
    expanded: &AkitaExpandedSetup<F>,
    slot: &SetupPrefixSlot<F>,
) -> Result<Vec<F>, AkitaError> {
    let n_prefix = slot.id.n_prefix()?;
    let fields = expanded.shared_matrix().as_field_slice();
    if slot.natural_len > fields.len() || slot.natural_len > n_prefix {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix slot exceeds shared setup matrix".to_string(),
        ));
    }
    let mut evals = vec![F::zero(); n_prefix];
    evals[..slot.natural_len].copy_from_slice(&fields[..slot.natural_len]);
    Ok(evals)
}

fn setup_prefix_rings<F: FieldCore, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    slot: &SetupPrefixSlot<F>,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
    let evals = setup_prefix_field_evals(expanded, slot)?;
    Ok(evals
        .chunks_exact(D)
        .map(|chunk| CyclotomicRing::from_coefficients(from_fn(|idx| chunk[idx])))
        .collect())
}

fn setup_prefix_fold_geometry<const D: usize>(
    slot: &SetupPrefixSlot<impl FieldCore>,
    source_ring_len: usize,
) -> Result<(usize, usize), AkitaError> {
    let geometry = &slot.id.commitment_params.layout;
    geometry.validate()?;
    if slot.id.d_setup != D
        || geometry.group.num_polynomials() != 1
        || geometry.num_live_ring_elements_per_claim != source_ring_len
        || geometry.num_positions_per_block == 0
        || geometry.num_live_blocks != source_ring_len.div_ceil(geometry.num_positions_per_block)
    {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix source disagrees with frozen block geometry".into(),
        ));
    }
    Ok((geometry.num_positions_per_block, geometry.num_live_blocks))
}

fn fold_setup_prefix_blocks<F: FieldCore, const D: usize>(
    coeffs: &[CyclotomicRing<F, D>],
    scalars: &[F],
    num_positions_per_block: usize,
) -> Vec<CyclotomicRing<F, D>> {
    (0..coeffs.len().div_ceil(num_positions_per_block))
        .map(|block_idx| {
            let start = block_idx * num_positions_per_block;
            let end = (start + num_positions_per_block).min(coeffs.len());
            let mut acc = CyclotomicRing::<F, D>::zero();
            for (ring, scalar) in coeffs[start..end].iter().zip(scalars.iter()) {
                acc += ring.scale(scalar);
            }
            acc
        })
        .collect()
}

fn fold_setup_prefix_blocks_ring<F: FieldCore + CanonicalField, const D: usize>(
    coeffs: &[CyclotomicRing<F, D>],
    scalars: &[CyclotomicRing<F, D>],
    num_positions_per_block: usize,
) -> Vec<CyclotomicRing<F, D>> {
    (0..coeffs.len().div_ceil(num_positions_per_block))
        .map(|block_idx| {
            let start = block_idx * num_positions_per_block;
            let end = (start + num_positions_per_block).min(coeffs.len());
            let mut acc = CyclotomicRing::<F, D>::zero();
            for (ring, scalar) in coeffs[start..end].iter().zip(scalars.iter()) {
                ring.mul_accumulate_sparse_rhs_into(scalar, &mut acc);
            }
            acc
        })
        .collect()
}

fn setup_prefix_evaluate_and_fold<F: FieldCore + CanonicalField, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    slot: &SetupPrefixSlot<F>,
    plan: OpeningFoldPlan<'_, F, D>,
) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
    let coeffs = setup_prefix_rings::<F, D>(expanded, slot)?;
    let num_positions_per_block = plan.num_positions_per_block();
    let (expected_positions, num_live_blocks) =
        setup_prefix_fold_geometry::<D>(slot, coeffs.len())?;
    if num_positions_per_block != expected_positions {
        return Err(AkitaError::InvalidSize {
            expected: expected_positions,
            actual: num_positions_per_block,
        });
    }
    plan.validate(num_live_blocks)?;
    match plan {
        OpeningFoldPlan::Base {
            live_block_weights,
            position_weights,
            num_positions_per_block,
        } => {
            let folded =
                fold_setup_prefix_blocks(&coeffs, position_weights, num_positions_per_block);
            let (eval, folded) = crate::backend::poly_helpers::fused_evaluate_and_fold_base(
                folded,
                live_block_weights,
            );
            Ok(OpeningFoldOutput { eval, folded })
        }
        OpeningFoldPlan::Ring {
            live_block_weights,
            position_weights,
            num_positions_per_block,
        } => {
            let folded =
                fold_setup_prefix_blocks_ring(&coeffs, position_weights, num_positions_per_block);
            let (eval, folded) = crate::backend::poly_helpers::fused_evaluate_and_fold_ring(
                folded,
                live_block_weights,
            );
            Ok(OpeningFoldOutput { eval, folded })
        }
    }
}

fn setup_prefix_decompose_fold<F: CanonicalField, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    slot: &SetupPrefixSlot<F>,
    plan: DecomposeFoldPlan<'_>,
) -> Result<crate::DecomposeFoldWitness<F>, AkitaError> {
    let coeffs = setup_prefix_rings::<F, D>(expanded, slot)?;
    let (num_positions_per_block, num_live_blocks) =
        setup_prefix_fold_geometry::<D>(slot, coeffs.len())?;
    if plan.num_positions_per_block != num_positions_per_block
        || plan.challenges.len() != num_live_blocks
    {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix decompose plan disagrees with frozen block geometry".into(),
        ));
    }
    let q = (-F::one()).to_canonical_u128() + 1;
    let threshold = decompose_centering_threshold(plan.num_digits, plan.log_basis, q);
    let params = DecomposeParams {
        threshold,
        q,
        mask: (1i128 << plan.log_basis) - 1,
        half_b: 1i128 << (plan.log_basis - 1),
        b_val: 1i128 << plan.log_basis,
        log_basis: plan.log_basis,
        overflow_possible: q.saturating_sub(threshold) > i128::MAX as u128,
    };
    let centered = balanced_ring_decompose_fold_partitioned::<F, D>(
        &coeffs,
        plan.challenges,
        plan.num_positions_per_block,
        plan.num_digits,
        &params,
    );
    Ok(build_decompose_fold_witness::<F, D>(centered, q))
}

impl<F, const D: usize> OpeningFoldKernel<RecursiveFoldView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn evaluate_and_fold(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: RecursiveFoldView<'_, F, D>,
        plan: OpeningFoldPlan<'_, F, D>,
    ) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
        match source {
            RecursiveFoldView::SetupPrefix { expanded, slot } => {
                setup_prefix_evaluate_and_fold(expanded, slot, plan)
            }
            RecursiveFoldView::Witness(view) => <CpuBackend as OpeningFoldKernel<
                SuffixWitnessView<'_, F, D>,
                F,
                D,
            >>::evaluate_and_fold(
                self, prepared, view, plan
            ),
        }
    }

    fn decompose_fold(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: RecursiveFoldView<'_, F, D>,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<crate::DecomposeFoldWitness<F>, AkitaError> {
        match source {
            RecursiveFoldView::SetupPrefix { expanded, slot } => {
                setup_prefix_decompose_fold::<F, D>(expanded, slot, plan)
            }
            RecursiveFoldView::Witness(view) => {
                <CpuBackend as OpeningFoldKernel<SuffixWitnessView<'_, F, D>, F, D>>::decompose_fold(
                    self, prepared, view, plan,
                )
            }
        }
    }
}

impl<F, const D: usize> OpeningBatchKernel<RecursiveFoldBatchView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn decompose_fold_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: RecursiveFoldBatchView<'_, F, D>,
        _plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<BatchDecomposeFoldOutcome<F, D>, AkitaError> {
        let _ = source.polys;
        Ok(BatchDecomposeFoldOutcome::FallbackPerPoly)
    }
}

fn setup_prefix_extension_tensor_unsupported<T>() -> Result<T, AkitaError> {
    Err(AkitaError::InvalidSetup(
        "setup-prefix grouped suffix does not support extension tensor projection".to_string(),
    ))
}

fn recursive_fold_batch_witnesses<'a, F: FieldCore, const D: usize>(
    source: RecursiveFoldBatchView<'a, F, D>,
) -> Result<Vec<&'a RecursiveWitnessFlat>, AkitaError> {
    let mut witnesses = Vec::with_capacity(source.polys.len());
    for poly in source.polys {
        match poly {
            RecursiveFoldSource::Witness(witness) => witnesses.push(witness.as_ref()),
            RecursiveFoldSource::SetupPrefix { .. } => {
                return setup_prefix_extension_tensor_unsupported();
            }
        }
    }
    Ok(witnesses)
}

impl<F, E, const D: usize> TensorProjectionKernel<RecursiveFoldView<'_, F, D>, F, E, D>
    for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: ExtField<F>,
{
    fn column_partials(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: RecursiveFoldView<'_, F, D>,
        logical_point: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        match source {
            RecursiveFoldView::SetupPrefix { .. } => setup_prefix_extension_tensor_unsupported(),
            RecursiveFoldView::Witness(view) => <CpuBackend as TensorProjectionKernel<
                SuffixWitnessView<'_, F, D>,
                F,
                E,
                D,
            >>::column_partials(
                self, prepared, view, logical_point
            ),
        }
    }

    fn packed_witness(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: RecursiveFoldView<'_, F, D>,
    ) -> Result<TensorPackedWitness<E>, AkitaError> {
        match source {
            RecursiveFoldView::SetupPrefix { .. } => Err(AkitaError::InvalidSetup(
                "setup-prefix grouped suffix does not support extension tensor packing".to_string(),
            )),
            RecursiveFoldView::Witness(view) => <CpuBackend as TensorProjectionKernel<
                SuffixWitnessView<'_, F, D>,
                F,
                E,
                D,
            >>::packed_witness(self, prepared, view),
        }
    }

    fn root_projection(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: RecursiveFoldView<'_, F, D>,
    ) -> Result<RootTensorProjectionPoly<F>, AkitaError>
    where
        E: FpExtEncoding<F>,
    {
        match source {
            RecursiveFoldView::SetupPrefix { .. } => setup_prefix_extension_tensor_unsupported(),
            RecursiveFoldView::Witness(view) => <CpuBackend as TensorProjectionKernel<
                SuffixWitnessView<'_, F, D>,
                F,
                E,
                D,
            >>::root_projection(
                self, prepared, view
            ),
        }
    }
}

impl<F, E, const D: usize> TensorProjectionBatchKernel<RecursiveFoldBatchView<'_, F, D>, F, E, D>
    for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: ExtField<F>,
{
    fn column_partials_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: RecursiveFoldBatchView<'_, F, D>,
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        let witnesses = recursive_fold_batch_witnesses(source)?;
        let batch = <RecursiveWitnessFlat as RootTensorSource<F, D>>::tensor_batch(&witnesses)?;
        <CpuBackend as TensorProjectionBatchKernel<SuffixWitnessBatchView<'_, F, D>, F, E, D>>::column_partials_batch(
            self, prepared, batch, logical_point,
        )
    }

    fn sparse_linear_combination(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: RecursiveFoldBatchView<'_, F, D>,
        coeffs: &[E],
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError> {
        let witnesses = recursive_fold_batch_witnesses(source)?;
        let batch = <RecursiveWitnessFlat as RootTensorSource<F, D>>::tensor_batch(&witnesses)?;
        <CpuBackend as TensorProjectionBatchKernel<SuffixWitnessBatchView<'_, F, D>, F, E, D>>::sparse_linear_combination(
            self, prepared, batch, coeffs,
        )
    }
}
