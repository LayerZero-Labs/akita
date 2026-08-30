use std::sync::Arc;

use akita_algebra::ring::cyclotomic::decompose_centering_threshold;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_types::{
    tensor_column_partials_from_base_evals, tensor_packed_witness_evals, AkitaExpandedSetup,
    FlatMatrix, SetupPrefixSlot,
};
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};

use crate::backend::poly_helpers::{
    balanced_ring_decompose_fold_partitioned, build_decompose_fold_witness, DecomposeParams,
};
use crate::backend::{RecursiveWitnessFlat, SuffixWitnessBatchView, SuffixWitnessView};
use crate::compute::{
    BatchDecomposeFoldOutcome, CpuBackend, DecomposeFoldBatchPlan, DecomposeFoldPlan,
    OpeningBatchKernel, OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan, RootOpeningSource,
    RootPolyMeta, RootPolyShape, RootTensorSource, SubringCoefficientPackingBatchKernel,
    SubringCoefficientPackingPartials, SubringCoefficientPackingPlan, TensorProjectionBatchKernel,
    TensorProjectionKernel,
};

#[doc(hidden)]
#[derive(Clone)]
pub enum RecursiveFoldSource<F: Field> {
    SetupPrefix {
        expanded: Arc<AkitaExpandedSetup<F>>,
        slot: Arc<SetupPrefixSlot<F>>,
    },
    Witness(Arc<RecursiveWitnessFlat>),
}

impl<F: Field> RecursiveFoldSource<F> {
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
pub enum RecursiveFoldView<'a, F: Field, const D: usize> {
    SetupPrefix {
        expanded: &'a AkitaExpandedSetup<F>,
        slot: &'a SetupPrefixSlot<F>,
    },
    Witness(SuffixWitnessView<'a, F, D>),
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct RecursiveFoldBatchView<'a, F: Field, const D: usize> {
    polys: &'a [&'a RecursiveFoldSource<F>],
}

impl<F: Field> RootPolyMeta<F> for RecursiveFoldSource<F> {
    fn num_vars(&self) -> usize {
        match self {
            Self::SetupPrefix { slot, .. } => {
                slot.id.n_prefix().unwrap_or(1).trailing_zeros() as usize
            }
            Self::Witness(witness) => RootPolyMeta::<F>::num_vars(witness.as_ref()),
        }
    }

    #[cfg(feature = "response-model-diagnostics")]
    fn exact_integer_coeff_l2_sq(&self) -> Option<u128> {
        match self {
            Self::SetupPrefix { .. } => None,
            Self::Witness(witness) => {
                RootPolyMeta::<F>::exact_integer_coeff_l2_sq(witness.as_ref())
            }
        }
    }
}

impl<F: Field, const D: usize> RootPolyShape<F, D> for RecursiveFoldSource<F> {
    fn num_ring_elems(&self) -> usize {
        match self {
            Self::SetupPrefix { slot, .. } => slot.id.n_prefix().map_or(1, |n| n / D),
            Self::Witness(witness) => RootPolyShape::<F, D>::num_ring_elems(witness.as_ref()),
        }
    }

    fn num_vars(&self) -> usize {
        RootPolyMeta::<F>::num_vars(self)
    }

    fn num_live_ring_elems(&self) -> usize {
        match self {
            Self::SetupPrefix { slot, .. } => slot.id.n_prefix().map_or(1, |n| n / D),
            Self::Witness(witness) => RootPolyShape::<F, D>::num_live_ring_elems(witness.as_ref()),
        }
    }
}

impl<F: Field, const D: usize> RootOpeningSource<F, D> for RecursiveFoldSource<F> {
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

impl<F: Field, const D: usize> RootTensorSource<F, D> for RecursiveFoldSource<F> {
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

fn checked_setup_prefix_ring_count<const D: usize>(
    natural_len: usize,
    n_prefix: usize,
) -> Result<usize, AkitaError> {
    if D == 0 || natural_len > n_prefix || !n_prefix.is_multiple_of(D) {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix length is incompatible with its ring layout".to_string(),
        ));
    }
    Ok(n_prefix / D)
}

fn setup_prefix_rings<F: Field, const D: usize>(
    matrix: &FlatMatrix<F>,
    natural_len: usize,
    n_prefix: usize,
) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
    let ring_count = checked_setup_prefix_ring_count::<D>(natural_len, n_prefix)?;
    Ok(matrix.ring_view::<D>(1, ring_count)?.as_slice())
}

fn setup_prefix_fold_geometry<const D: usize>(
    slot: &SetupPrefixSlot<impl Field>,
    source_ring_len: usize,
) -> Result<(usize, usize), AkitaError> {
    let geometry = &slot.id.commitment_profile;
    geometry.validate(
        slot.id
            .commitment_profile
            .inner
            .matrix
            .sis_modulus_profile()
            .field_bits(),
    )?;
    if slot.id.d_setup() != D
        || geometry.group.num_polynomials() != 1
        || geometry.blocks.live_ring_elements_per_claim != source_ring_len
        || geometry.blocks.positions_per_block == 0
        || geometry.blocks.live_blocks
            != source_ring_len.div_ceil(geometry.blocks.positions_per_block)
    {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix source disagrees with frozen block geometry".into(),
        ));
    }
    Ok((
        geometry.blocks.positions_per_block,
        geometry.blocks.live_blocks,
    ))
}

fn fold_setup_prefix_blocks<F: Field, const D: usize>(
    coeffs: &[CyclotomicRing<F, D>],
    scalars: &[F],
    num_positions_per_block: usize,
) -> Vec<CyclotomicRing<F, D>> {
    cfg_into_iter!(0..coeffs.len().div_ceil(num_positions_per_block))
        .map(|block_idx| {
            let start = block_idx * num_positions_per_block;
            let end = (start + num_positions_per_block).min(coeffs.len());
            let mut acc = CyclotomicRing::<F, D>::zero();
            for (ring, scalar) in coeffs[start..end].iter().zip(scalars.iter()) {
                ring.scale_accumulate_into(&mut acc, *scalar);
            }
            acc
        })
        .collect()
}

fn fold_setup_prefix_blocks_ring<F: Field + CanonicalEncoding, const D: usize>(
    coeffs: &[CyclotomicRing<F, D>],
    scalars: &[CyclotomicRing<F, D>],
    num_positions_per_block: usize,
) -> Vec<CyclotomicRing<F, D>> {
    cfg_into_iter!(0..coeffs.len().div_ceil(num_positions_per_block))
        .map(|block_idx| {
            let start = block_idx * num_positions_per_block;
            let end = (start + num_positions_per_block).min(coeffs.len());
            let mut acc = CyclotomicRing::<F, D>::zero();
            for (ring, scalar) in coeffs[start..end].iter().zip(scalars) {
                ring.mul_accumulate_sparse_rhs_into(scalar, &mut acc);
            }
            acc
        })
        .collect()
}

fn setup_prefix_evaluate_and_fold<F: Field + CanonicalEncoding, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    slot: &SetupPrefixSlot<F>,
    plan: OpeningFoldPlan<'_, F>,
) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
    let coeffs = setup_prefix_rings::<F, D>(
        expanded.shared_matrix(),
        slot.id.natural_len,
        slot.id.n_prefix()?,
    )?;
    let num_positions_per_block = plan.num_positions_per_block();
    let (expected_positions, num_live_blocks) =
        setup_prefix_fold_geometry::<D>(slot, coeffs.len())?;
    if num_positions_per_block != expected_positions {
        return Err(AkitaError::InvalidSize {
            expected: expected_positions,
            actual: num_positions_per_block,
        });
    }
    plan.validate::<D>(num_live_blocks)?;
    match plan {
        OpeningFoldPlan::Base {
            live_block_weights,
            position_weights,
            num_positions_per_block,
        } => {
            let folded =
                fold_setup_prefix_blocks(coeffs, position_weights, num_positions_per_block);
            let (eval, folded) = crate::backend::poly_helpers::fused_evaluate_and_fold_base(
                folded,
                live_block_weights,
            );
            Ok(OpeningFoldOutput { eval, folded })
        }
        OpeningFoldPlan::Subfield {
            multipliers,
            num_positions_per_block,
        } => {
            let position_weights = multipliers.materialize_position_rings::<D>()?;
            let live_block_weights = multipliers.materialize_fold_rings::<D>()?;
            let folded =
                fold_setup_prefix_blocks_ring(coeffs, &position_weights, num_positions_per_block);
            let (eval, folded) = crate::backend::poly_helpers::fused_evaluate_and_fold_materialized(
                folded,
                &live_block_weights,
            );
            Ok(OpeningFoldOutput { eval, folded })
        }
    }
}

fn setup_prefix_decompose_fold<F: Field + CanonicalEncoding, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    slot: &SetupPrefixSlot<F>,
    plan: DecomposeFoldPlan<'_>,
) -> Result<crate::DecomposeFoldWitness<F>, AkitaError> {
    let coeffs = setup_prefix_rings::<F, D>(
        expanded.shared_matrix(),
        slot.id.natural_len,
        slot.id.n_prefix()?,
    )?;
    let (num_positions_per_block, num_live_blocks) =
        setup_prefix_fold_geometry::<D>(slot, coeffs.len())?;
    if plan.num_positions_per_block != num_positions_per_block
        || plan.challenges.len() != num_live_blocks
    {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix decompose plan disagrees with frozen block geometry".into(),
        ));
    }
    let q = (-F::one())
        .to_u128_checked()
        .expect("Akita field element must fit in u128")
        + 1;
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
        coeffs,
        plan.challenges,
        plan.num_positions_per_block,
        plan.num_digits,
        &params,
    );
    Ok(build_decompose_fold_witness::<F, D>(centered, q))
}

impl<F, const D: usize> OpeningFoldKernel<RecursiveFoldView<'_, F, D>, F, D> for CpuBackend
where
    F: Field + CanonicalEncoding,
{
    fn evaluate_and_fold(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: RecursiveFoldView<'_, F, D>,
        plan: OpeningFoldPlan<'_, F>,
    ) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
        match source {
            RecursiveFoldView::SetupPrefix { expanded, slot } => {
                let _span = tracing::info_span!(
                    "fold_recursive_source",
                    source_kind = "setup_prefix",
                    ring_dimension = D
                )
                .entered();
                setup_prefix_evaluate_and_fold(expanded, slot, plan)
            }
            RecursiveFoldView::Witness(view) => {
                let _span = tracing::info_span!(
                    "fold_recursive_source",
                    source_kind = "small_balanced_digits",
                    ring_dimension = D
                )
                .entered();
                <CpuBackend as OpeningFoldKernel<SuffixWitnessView<'_, F, D>, F, D>>::evaluate_and_fold(
                    self, prepared, view, plan,
                )
            }
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
    F: Field + CanonicalEncoding,
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

fn setup_prefix_base_evals<'a, F: Field, const D: usize>(
    expanded: &'a AkitaExpandedSetup<F>,
    slot: &SetupPrefixSlot<F>,
) -> Result<&'a [F], AkitaError> {
    let n_prefix = slot.id.n_prefix()?;
    if slot.id.d_setup() != D || !n_prefix.is_multiple_of(D) {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix tensor source disagrees with its frozen ring dimension".into(),
        ));
    }
    expanded
        .shared_matrix()
        .as_field_slice()
        .get(..n_prefix)
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "setup-prefix tensor source exceeds the shared setup matrix".into(),
            )
        })
}

impl<F, E, const D: usize> TensorProjectionKernel<RecursiveFoldView<'_, F, D>, F, E, D>
    for CpuBackend
where
    F: Field + CanonicalEncoding + Ring,
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
            RecursiveFoldView::SetupPrefix { expanded, slot } => {
                let base_evals = setup_prefix_base_evals::<F, D>(expanded, slot)?;
                tensor_column_partials_from_base_evals::<F, E>(
                    slot.id.n_prefix()?.trailing_zeros() as usize,
                    base_evals,
                    logical_point,
                )
            }
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
    ) -> Result<Vec<E>, AkitaError> {
        match source {
            RecursiveFoldView::SetupPrefix { expanded, slot } => {
                let base_evals = setup_prefix_base_evals::<F, D>(expanded, slot)?;
                tensor_packed_witness_evals::<F, E>(
                    slot.id.n_prefix()?.trailing_zeros() as usize,
                    base_evals,
                )
            }
            RecursiveFoldView::Witness(view) => <CpuBackend as TensorProjectionKernel<
                SuffixWitnessView<'_, F, D>,
                F,
                E,
                D,
            >>::packed_witness(self, prepared, view),
        }
    }
}

impl<F, E, const D: usize> TensorProjectionBatchKernel<RecursiveFoldBatchView<'_, F, D>, F, E, D>
    for CpuBackend
where
    F: Field + CanonicalEncoding + Ring,
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
        source
            .polys
            .iter()
            .map(|poly| {
                <CpuBackend as TensorProjectionKernel<
                    RecursiveFoldView<'_, F, D>,
                    F,
                    E,
                    D,
                >>::column_partials(self, prepared, poly.tensor_view()?, logical_point)
            })
            .collect()
    }
}

impl<F, E, const D: usize>
    SubringCoefficientPackingBatchKernel<RecursiveFoldBatchView<'_, F, D>, F, E, D> for CpuBackend
where
    F: Field + CanonicalEncoding + Ring,
    E: ExtField<F> + akita_types::FpExtEncoding<F>,
{
    fn coefficient_packing_partials_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: RecursiveFoldBatchView<'_, F, D>,
        plan: SubringCoefficientPackingPlan<'_, E>,
    ) -> Result<Vec<SubringCoefficientPackingPartials<F>>, AkitaError> {
        let mut outputs = Vec::with_capacity(source.polys.len());
        for poly in source.polys {
            plan.validate::<D>(RootPolyMeta::<F>::num_vars(*poly))?;
            match poly {
                RecursiveFoldSource::SetupPrefix { expanded, slot } => {
                    let rings = setup_prefix_rings::<F, D>(
                        expanded.shared_matrix(),
                        slot.id.natural_len,
                        slot.id.n_prefix()?,
                    )?;
                    let (positions_per_block, live_blocks) =
                        setup_prefix_fold_geometry::<D>(slot, rings.len())?;
                    if rings.len() != plan.point.num_live_positions()
                        || positions_per_block != plan.point.num_positions_per_block()
                        || live_blocks != plan.point.num_live_blocks()
                    {
                        return Err(AkitaError::InvalidSetup(
                            "setup-prefix source disagrees with coefficient-packing point".into(),
                        ));
                    }
                    let coordinates =
                        crate::backend::coefficient_packing::coefficient_packing_partials_from_position_source::<
                            F,
                            E,
                            _,
                            D,
                        >(
                            plan,
                            RootPolyMeta::<F>::num_vars(*poly),
                            |position| {
                                rings
                                    .get(position)
                                    .map(CyclotomicRing::coefficients)
                                    .ok_or(AkitaError::InvalidProof)
                            },
                            |_, coefficient, source| source[coefficient],
                        )?;
                    outputs.push(SubringCoefficientPackingPartials::new(
                        plan.point.geometry(),
                        plan.point.num_live_blocks(),
                        coordinates,
                    )?);
                }
                RecursiveFoldSource::Witness(witness) => {
                    let witnesses = [witness.as_ref()];
                    let batch =
                        <RecursiveWitnessFlat as RootTensorSource<F, D>>::tensor_batch(&witnesses)?;
                    let mut partials = <CpuBackend as SubringCoefficientPackingBatchKernel<
                        SuffixWitnessBatchView<'_, F, D>,
                        F,
                        E,
                        D,
                    >>::coefficient_packing_partials_batch(
                        self, prepared, batch, plan
                    )?;
                    outputs.push(partials.pop().ok_or(AkitaError::InvalidProof)?);
                }
            }
        }
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallenge;
    use jolt_field::{One, Prime128OffsetA7F7, Zero};

    #[test]
    fn setup_prefix_q128_base_fold_matches_separate_oracle() {
        type F = Prime128OffsetA7F7;
        const D: usize = 8;
        let coeffs: Vec<CyclotomicRing<F, D>> = (0..5)
            .map(|row| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|column| {
                    F::from_u64((row * D + column + 1) as u64)
                }))
            })
            .collect::<Vec<_>>();
        let weights = (2..6).map(F::from_u64).collect::<Vec<_>>();
        let expected = coeffs
            .chunks(4)
            .map(|block| {
                block
                    .iter()
                    .zip(&weights)
                    .fold(CyclotomicRing::zero(), |acc, (ring, weight)| {
                        acc + ring.scale(weight)
                    })
            })
            .collect::<Vec<_>>();

        assert_eq!(fold_setup_prefix_blocks(&coeffs, &weights, 4), expected);
    }

    #[test]
    fn setup_prefix_ring_view_is_borrowed_and_checked() {
        type F = Prime128OffsetA7F7;
        const D: usize = 4;
        let fields = (1..=8).map(F::from_u64).collect::<Vec<_>>();
        let matrix = FlatMatrix::from_flat_data(fields);

        let rings = setup_prefix_rings::<F, D>(&matrix, 7, 8).unwrap();
        assert_eq!(rings.len(), 2);
        assert_eq!(rings.as_ptr().cast::<F>(), matrix.as_field_slice().as_ptr());

        let undersized = FlatMatrix::from_flat_data(vec![F::zero(); 7]);
        assert!(setup_prefix_rings::<F, D>(&undersized, 7, 8).is_err());
        assert!(setup_prefix_rings::<F, D>(&matrix, 9, 8).is_err());
        assert!(setup_prefix_rings::<F, D>(&matrix, 7, 7).is_err());
    }

    #[test]
    fn setup_prefix_large_geometry_needs_no_storage() {
        assert_eq!(
            checked_setup_prefix_ring_count::<256>(1 << 24, 1 << 24),
            Ok(1 << 16)
        );
        assert_eq!(
            checked_setup_prefix_ring_count::<256>(1 << 26, 1 << 26),
            Ok(1 << 18)
        );
    }

    #[test]
    fn borrowed_setup_prefix_feeds_both_fold_consumers() {
        type F = Prime128OffsetA7F7;
        const D: usize = 4;
        let fields = (1..=16).map(F::from_u64).collect::<Vec<_>>();
        let matrix = FlatMatrix::from_flat_data(fields);
        let borrowed = setup_prefix_rings::<F, D>(&matrix, 16, 16).unwrap();
        let owned = borrowed.to_vec();
        let weights = [F::from_u64(2), F::from_u64(3)];
        assert_eq!(
            fold_setup_prefix_blocks(borrowed, &weights, 2),
            fold_setup_prefix_blocks(&owned, &weights, 2)
        );

        let challenges = [SparseChallenge {
            positions: vec![0, 3].into(),
            coeffs: vec![1, -1].into(),
        }];
        let q = (-F::one())
            .to_u128_checked()
            .expect("Akita field element must fit in u128")
            + 1;
        let params = DecomposeParams {
            threshold: decompose_centering_threshold(1, 8, q),
            q,
            mask: 255,
            half_b: 128,
            b_val: 256,
            log_basis: 8,
            overflow_possible: false,
        };
        assert_eq!(
            balanced_ring_decompose_fold_partitioned::<F, D>(borrowed, &challenges, 4, 1, &params,),
            balanced_ring_decompose_fold_partitioned::<F, D>(&owned, &challenges, 4, 1, &params,)
        );
    }

    #[test]
    fn setup_prefix_openings_match_copied_dense_oracles() {
        use akita_types::{
            coefficient_packing_partials, sample_akita_setup_seed, AkitaCommitmentHint,
            AkitaSetupDescriptor, BasisMode, CommittedGroupParams, GroupCommitPhaseParams,
            InnerCommitMatrixParams, OuterCommitMatrixParams, PolynomialGroupLayout,
            PreparedSubringCoefficientPackingPoint, SetupPrefixPublicCommitment, SetupPrefixSlotId,
            SisModulusProfileId, SubringCoefficientPackingGeometry,
        };
        use jolt_field::Ext2;

        type F = Prime128OffsetA7F7;
        const D: usize = 128;
        let mut params = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            D,
            2,
            1,
            1,
            1,
            akita_challenges::SparseChallengeConfig::production_for_ring_dim(D).unwrap(),
        )
        .with_decomp(4, 4, 2, 2, 2)
        .unwrap();
        let inner = &params.inner().matrix;
        params.own_group_mut().profile.inner.matrix = InnerCommitMatrixParams::new_unchecked(
            inner.security_policy(),
            inner.sis_table_key().unwrap().table_digest,
            inner.sis_modulus_profile(),
            inner.output_rank(),
            inner.input_width(),
            2,
            D,
        );
        let outer = &params.outer().matrix;
        params.own_group_mut().profile.outer.matrix = OuterCommitMatrixParams::new_unchecked(
            outer.security_policy(),
            outer.sis_table_key().table_digest,
            outer.sis_modulus_profile(),
            outer.output_rank(),
            outer.input_width(),
            3,
            D,
        );
        let profile =
            GroupCommitPhaseParams::try_from_params(PolynomialGroupLayout::singleton(9), &params)
                .expect("valid setup-prefix profile");
        let fields = (0..512)
            .map(|index| F::from_i64((index % 17) as i64 - 8))
            .collect::<Vec<_>>();
        let expanded = Arc::new(
            AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                AkitaSetupDescriptor {
                    max_num_vars: 9,
                    max_num_batched_polys: 1,
                    num_field_elements: fields.len(),
                    setup_seed: sample_akita_setup_seed(),
                },
                FlatMatrix::from_flat_data(fields.clone()),
            ),
        );
        let slot = Arc::new(SetupPrefixSlot {
            id: SetupPrefixSlotId {
                natural_len: 400,
                commitment_profile: profile,
            },
            commitment: SetupPrefixPublicCommitment { rows: Vec::new() },
            hint: AkitaCommitmentHint::new(1, Vec::new()).unwrap(),
        });
        let geometry = SubringCoefficientPackingGeometry::try_new(1, D, 64).unwrap();
        let public_point = (0..9)
            .map(|index| F::from_u64((index + 2) as u64))
            .collect::<Vec<_>>();
        let point = PreparedSubringCoefficientPackingPoint::new(
            geometry,
            BasisMode::Lagrange,
            4,
            4,
            9,
            &public_point,
        )
        .unwrap();
        let source = RecursiveFoldSource::setup_prefix(expanded, slot);
        let sources = [&source];
        let batch =
            <RecursiveFoldSource<F> as RootTensorSource<F, D>>::tensor_batch(&sources).unwrap();
        let got = <CpuBackend as SubringCoefficientPackingBatchKernel<
            RecursiveFoldBatchView<'_, F, D>,
            F,
            F,
            D,
        >>::coefficient_packing_partials_batch(
            &CpuBackend::DEFAULT,
            None,
            batch,
            SubringCoefficientPackingPlan { point: &point },
        )
        .unwrap();
        let expected = coefficient_packing_partials::<F, F>(
            geometry,
            4,
            4,
            &fields,
            point.position_weights(),
            point.packing_weights(),
        )
        .unwrap();
        assert_eq!(got[0].coordinates(), expected);

        type E = Ext2<F>;
        let logical_point = (0..9)
            .map(|index| {
                E::from_base_slice(&[
                    F::from_u64((index + 2) as u64),
                    F::from_u64((2 * index + 3) as u64),
                ])
            })
            .collect::<Vec<_>>();
        let view =
            <RecursiveFoldSource<F> as RootTensorSource<F, D>>::tensor_view(&source).unwrap();
        let got_partials = <CpuBackend as TensorProjectionKernel<
            RecursiveFoldView<'_, F, D>,
            F,
            E,
            D,
        >>::column_partials(
            &CpuBackend::DEFAULT, None, view, &logical_point
        )
        .unwrap();
        let expected_partials =
            tensor_column_partials_from_base_evals::<F, E>(9, &fields, &logical_point).unwrap();
        assert_eq!(got_partials, expected_partials);

        let view =
            <RecursiveFoldSource<F> as RootTensorSource<F, D>>::tensor_view(&source).unwrap();
        let got_packed = <CpuBackend as TensorProjectionKernel<
            RecursiveFoldView<'_, F, D>,
            F,
            E,
            D,
        >>::packed_witness(&CpuBackend::DEFAULT, None, view)
        .unwrap();
        assert_eq!(
            got_packed,
            tensor_packed_witness_evals::<F, E>(9, &fields).unwrap()
        );

        let witness_digits = (0..512)
            .map(|index| (index % 11) as i8 - 5)
            .collect::<Vec<_>>();
        let witness_fields = witness_digits
            .iter()
            .copied()
            .map(F::from_i8)
            .collect::<Vec<_>>();
        let witness = RecursiveFoldSource::witness(Arc::new(RecursiveWitnessFlat::from_i8_digits(
            witness_digits,
        )));
        let mixed_sources = [&source, &witness];
        let mixed_batch =
            <RecursiveFoldSource<F> as RootTensorSource<F, D>>::tensor_batch(&mixed_sources)
                .unwrap();
        let got_mixed = <CpuBackend as TensorProjectionBatchKernel<
            RecursiveFoldBatchView<'_, F, D>,
            F,
            E,
            D,
        >>::column_partials_batch(
            &CpuBackend::DEFAULT, None, mixed_batch, &logical_point
        )
        .unwrap();
        assert_eq!(
            got_mixed,
            vec![
                expected_partials,
                tensor_column_partials_from_base_evals::<F, E>(9, &witness_fields, &logical_point,)
                    .unwrap(),
            ]
        );
    }
}
