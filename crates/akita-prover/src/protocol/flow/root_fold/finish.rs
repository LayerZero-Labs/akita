use super::super::*;

use super::{prove_root_fold_from_ring_relation, prove_terminal_root_fold_from_ring_relation};

#[allow(clippy::too_many_arguments)]
pub(super) fn finish_root_fold_with_prepared_openings<
    'stack,
    F,
    C,
    T,
    Q,
    B,
    const D: usize,
    CommitW,
>(
    expanded: &AkitaExpandedSetup<F>,
    stack: &crate::compute::ProverComputeStack<'stack, F, D, B, B, B, B>,
    transcript: &mut T,
    polys: &[&Q],
    incidence_summary: &ClaimIncidenceSummary,
    commitments: &[RingCommitment<F, D>],
    hints: Vec<AkitaCommitmentHint<F, D>>,
    root_params: &LevelParams,
    expected_w_len: usize,
    next_log_basis: u32,
    commit_w_for_next: CommitW,
    prepared_points: Vec<PreparedRootOpeningPoint<F, D>>,
    e_folded_by_poly: Vec<Vec<CyclotomicRing<F, D>>>,
    y_rings: Vec<CyclotomicRing<F, D>>,
    #[cfg(feature = "zk")] y_rings_masked: Vec<CyclotomicRing<F, D>>,
    row_coefficients: Vec<C>,
    row_coefficient_rings: Vec<CyclotomicRing<F, D>>,
    extension_opening_reduction: Option<ExtensionOpeningReductionProof<C>>,
    #[cfg(feature = "zk")] zk_hiding_commitment: ZkHidingCommitment<F>,
    #[cfg(feature = "zk")] zk_hiding: ZkHidingProverState<F>,
    setup_contribution_mode: SetupContributionMode,
) -> Result<RootLevelRawOutput<F, C, D>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + FromPrimitiveInt
        + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    C: ExtField<F>
        + RingSubfieldEncoding<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
    Q: RootOpeningSource<F, D>,
    B: DigitRowsComputeBackend<F>
        + RingSwitchComputeBackend<F>
        + for<'view> OpeningFoldKernel<Q::OpeningView<'view>, F, D>
        + for<'view> OpeningBatchKernel<Q::OpeningBatchView<'view>, F, D>,
    CommitW: FnOnce(&RecursiveWitnessFlat) -> Result<NextWitnessCommitment<F>, AkitaError>,
{
    let opening = stack.opening();
    let ring_switch = stack.ring_switch();
    let ring_opening_points = incidence_summary
        .public_rows()
        .iter()
        .map(|row| {
            prepared_points
                .get(row.point_idx())
                .map(|prepared_point| prepared_point.ring_opening_point.clone())
                .ok_or_else(|| {
                    AkitaError::InvalidInput("public row point index out of range".to_string())
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let ring_multiplier_points = incidence_summary
        .public_rows()
        .iter()
        .map(|row| {
            prepared_points
                .get(row.point_idx())
                .map(|prepared_point| prepared_point.ring_multiplier_point.clone())
                .ok_or_else(|| {
                    AkitaError::InvalidInput("public row point index out of range".to_string())
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (instance, witness) = RingRelationProver::new::<F, D, _, Q, B>(
        opening.backend(),
        opening.prepared(),
        ring_opening_points,
        ring_multiplier_points,
        incidence_summary.claim_to_point().to_vec(),
        polys,
        e_folded_by_poly,
        incidence_summary,
        root_params.clone(),
        hints,
        transcript,
        commitments,
        &y_rings,
        row_coefficient_rings,
        MRowLayout::WithDBlock,
    )?;

    let commitment_rows_owned: Option<Vec<CyclotomicRing<F, D>>> = if commitments.len() == 1 {
        None
    } else {
        Some(flatten_batched_commitment_rows(commitments))
    };
    let commitment_rows: &[CyclotomicRing<F, D>] = match &commitment_rows_owned {
        Some(v) => v.as_slice(),
        None => commitments[0].u.as_slice(),
    };

    let mut raw = prove_root_fold_from_ring_relation::<F, C, T, B, D, _>(
        expanded,
        ring_switch.backend(),
        ring_switch.prepared(),
        transcript,
        commitment_rows,
        root_params,
        expected_w_len,
        next_log_basis,
        #[cfg(feature = "zk")]
        zk_hiding_commitment,
        #[cfg(feature = "zk")]
        zk_hiding,
        instance,
        witness,
        y_rings,
        #[cfg(feature = "zk")]
        y_rings_masked,
        row_coefficients,
        setup_contribution_mode,
        commit_w_for_next,
    )?;
    raw.extension_opening_reduction = extension_opening_reduction;
    Ok(raw)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn finish_terminal_root_fold_with_prepared_openings<
    'stack,
    F,
    C,
    T,
    Q,
    B,
    const D: usize,
>(
    expanded: &AkitaExpandedSetup<F>,
    stack: &crate::compute::ProverComputeStack<'stack, F, D, B, B, B, B>,
    transcript: &mut T,
    polys: &[&Q],
    incidence_summary: &ClaimIncidenceSummary,
    commitments: &[RingCommitment<F, D>],
    hints: Vec<AkitaCommitmentHint<F, D>>,
    root_params: &LevelParams,
    expected_w_len: usize,
    final_log_basis: u32,
    prepared_points: Vec<PreparedRootOpeningPoint<F, D>>,
    e_folded_by_poly: Vec<Vec<CyclotomicRing<F, D>>>,
    y_rings: Vec<CyclotomicRing<F, D>>,
    #[cfg(feature = "zk")] y_rings_masked: Vec<CyclotomicRing<F, D>>,
    row_coefficients: Vec<C>,
    row_coefficient_rings: Vec<CyclotomicRing<F, D>>,
    extension_opening_reduction: Option<ExtensionOpeningReductionProof<C>>,
    #[cfg(feature = "zk")] zk_hiding: &mut ZkHidingProverState<F>,
) -> Result<TerminalLevelProof<F, C>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + FromPrimitiveInt
        + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    C: ExtField<F>
        + RingSubfieldEncoding<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
    Q: RootOpeningSource<F, D>,
    B: DigitRowsComputeBackend<F>
        + RingSwitchComputeBackend<F>
        + for<'view> OpeningFoldKernel<Q::OpeningView<'view>, F, D>
        + for<'view> OpeningBatchKernel<Q::OpeningBatchView<'view>, F, D>,
{
    let opening = stack.opening();
    let ring_switch = stack.ring_switch();
    let ring_opening_points = incidence_summary
        .public_rows()
        .iter()
        .map(|row| {
            prepared_points
                .get(row.point_idx())
                .map(|prepared_point| prepared_point.ring_opening_point.clone())
                .ok_or_else(|| {
                    AkitaError::InvalidInput("public row point index out of range".to_string())
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let ring_multiplier_points = incidence_summary
        .public_rows()
        .iter()
        .map(|row| {
            prepared_points
                .get(row.point_idx())
                .map(|prepared_point| prepared_point.ring_multiplier_point.clone())
                .ok_or_else(|| {
                    AkitaError::InvalidInput("public row point index out of range".to_string())
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (instance, witness) = RingRelationProver::new::<F, D, _, Q, B>(
        opening.backend(),
        opening.prepared(),
        ring_opening_points,
        ring_multiplier_points,
        incidence_summary.claim_to_point().to_vec(),
        polys,
        e_folded_by_poly,
        incidence_summary,
        root_params.clone(),
        hints,
        transcript,
        commitments,
        &y_rings,
        row_coefficient_rings,
        MRowLayout::WithoutDBlock,
    )?;

    let commitment_rows_owned: Option<Vec<CyclotomicRing<F, D>>> = if commitments.len() == 1 {
        None
    } else {
        Some(flatten_batched_commitment_rows(commitments))
    };
    let commitment_rows: &[CyclotomicRing<F, D>] = match &commitment_rows_owned {
        Some(v) => v.as_slice(),
        None => commitments[0].u.as_slice(),
    };

    let mut terminal = prove_terminal_root_fold_from_ring_relation::<F, C, T, B, D>(
        expanded,
        ring_switch.backend(),
        ring_switch.prepared(),
        transcript,
        commitment_rows,
        root_params,
        expected_w_len,
        final_log_basis,
        instance,
        witness,
        y_rings,
        #[cfg(feature = "zk")]
        y_rings_masked,
        row_coefficients,
        #[cfg(feature = "zk")]
        zk_hiding,
    )?;
    terminal.extension_opening_reduction = extension_opening_reduction;
    Ok(terminal)
}
