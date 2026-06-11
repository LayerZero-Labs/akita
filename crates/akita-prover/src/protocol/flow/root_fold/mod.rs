mod relation;
pub use relation::{
    prove_root_fold_from_ring_relation, prove_terminal_root_fold_from_ring_relation,
};

mod eval;
mod finish;
mod public_phase;

use finish::{
    finish_root_fold_with_prepared_openings, finish_terminal_root_fold_with_prepared_openings,
};
use public_phase::{
    append_root_fold_transcript_prefix, batched_root_commitment_rows,
    flatten_root_commitment_rows_if_needed, maybe_prepare_root_extension_reduction,
    prepare_root_fold_direct_public_phase, prove_root_extension_reduction_public_phase,
    trace_root_fold_entry, validate_root_fold_inputs,
};

pub(in crate::protocol::flow) use eval::evaluate_recursive_witness_at_multiplier_point;

use super::*;

/// Prove the folded root level using already-selected root and next-level
/// parameters.
///
/// The caller owns schedule/config selection and passes the expected next
/// recursive witness length, next digit basis, and commitment policy for that
/// witness. This function owns root polynomial folding, public root transcript
/// setup, root ring-relation construction, and the folded-root prover
/// mechanics.
///
/// # Errors
///
/// Returns an error if root inputs are malformed, polynomial folding or
/// ring-relation construction fails, or the folded-root prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_root_fold_with_params<'stack, F, E, C, T, P, B, Cfg, const D: usize>(
    expanded: &Arc<AkitaExpandedSetup<F>>,
    prefix_slots: &SetupPrefixProverRegistry<F, D>,
    stack: &crate::compute::ProverComputeStack<'stack, F, D, B, B, B, B>,
    transcript: &mut T,
    polys: &[&P],
    incidence_summary: &ClaimIncidenceSummary,
    claim_points: &[&[E]],
    commitments: &[RingCommitment<F, D>],
    hints: Vec<AkitaCommitmentHint<F, D>>,
    root_params: &LevelParams,
    expected_w_len: usize,
    next_level_params: &LevelParams,
    #[cfg(feature = "zk")] zk_hiding_commitment: ZkHidingCommitment<F>,
    #[cfg(feature = "zk")] mut zk_hiding: ZkHidingProverState<F>,
    basis: BasisMode,
    setup_contribution_mode: SetupContributionMode,
) -> Result<RootLevelProverOutput<F, C, D>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + FromPrimitiveInt
        + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: RingSubfieldEncoding<F> + MulBaseUnreduced<F>,
    C: RingSubfieldEncoding<F>
        + ExtField<E>
        + ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
    P: RootProvePoly<F, D>,
    B: RootProveFlowBackend<F, P, E, C, D>,
    Cfg: CommitmentConfig<Field = F, ClaimField = E, ChallengeField = C>,
{
    let opening = stack.opening();
    let opening_backend = opening.backend();
    let ring_switch = stack.ring_switch();
    validate_root_fold_inputs(
        polys.len(),
        incidence_summary,
        claim_points.len(),
        commitments.len(),
        hints.len(),
    )?;
    trace_root_fold_entry(
        "prove_root_fold_with_params",
        incidence_summary.num_claims(),
        claim_points.len(),
    );
    append_root_fold_transcript_prefix(incidence_summary, commitments, claim_points, transcript)?;

    let alpha_bits = root_params.ring_dimension.trailing_zeros() as usize;
    if let Some(prepared_reduction) = maybe_prepare_root_extension_reduction::<F, E, C, P, B, D>(
        opening_backend,
        polys,
        incidence_summary,
        claim_points,
        incidence_summary.num_vars(),
    )? {
        let extension_phase = prove_root_extension_reduction_public_phase::<F, E, C, T, P, B, D>(
            opening_backend,
            polys,
            incidence_summary,
            root_params,
            basis,
            alpha_bits,
            prepared_reduction,
            transcript,
            #[cfg(feature = "zk")]
            &mut zk_hiding,
        )?;
        let transformed_refs = extension_phase
            .post_transform
            .transformed_polys
            .iter()
            .collect::<Vec<_>>();

        return finish_root_fold_with_prepared_openings::<
            F,
            C,
            T,
            RootTensorProjectionPoly<F, D>,
            B,
            Cfg,
            D,
        >(
            expanded,
            prefix_slots,
            stack,
            transcript,
            &transformed_refs,
            incidence_summary,
            commitments,
            hints,
            root_params,
            expected_w_len,
            next_level_params,
            extension_phase.post_transform.prepared_points,
            extension_phase.post_transform.e_folded_by_poly,
            extension_phase.post_transform.y_rings,
            #[cfg(feature = "zk")]
            extension_phase.post_transform.y_rings_masked,
            extension_phase.row_coefficients,
            extension_phase.row_coefficient_rings,
            Some(extension_phase.post_transform.extension_opening_reduction),
            #[cfg(feature = "zk")]
            zk_hiding_commitment,
            #[cfg(feature = "zk")]
            zk_hiding,
            setup_contribution_mode,
        );
    }

    let direct = prepare_root_fold_direct_public_phase::<F, E, C, T, P, B, D>(
        opening_backend,
        opening.prepared(),
        polys,
        incidence_summary,
        claim_points,
        commitments,
        hints,
        root_params,
        basis,
        alpha_bits,
        MRowLayout::WithDBlock,
        transcript,
        #[cfg(feature = "zk")]
        &mut zk_hiding,
    )?;

    let commitment_rows_owned = flatten_root_commitment_rows_if_needed(commitments);
    let commitment_rows = batched_root_commitment_rows(commitments, &commitment_rows_owned);
    let public_phase::RootFoldDirectPublicPhase {
        y_rings,
        #[cfg(feature = "zk")]
        y_rings_masked,
        row_coefficients,
        instance,
        witness,
    } = direct;
    prove_root_fold_from_ring_relation::<F, C, T, B, Cfg, D>(
        expanded,
        prefix_slots,
        ring_switch.backend(),
        ring_switch.prepared(),
        transcript,
        commitment_rows,
        root_params,
        expected_w_len,
        next_level_params,
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
    )
}

/// Terminal-root analogue of [`prove_root_fold_with_params`] used when the
/// schedule has exactly one fold level (the root is itself the terminal).
///
/// Mirrors the intermediate-root path through claim-incidence absorbs,
/// optional extension-opening reduction, and ring-relation setup, then
/// emits a [`TerminalLevelProof`] via
/// [`prove_terminal_root_fold_from_ring_relation`] instead of a
/// [`RootLevelRawOutput`].
///
/// # Errors
///
/// Returns an error if claim-incidence/transcript setup fails, the
/// extension-opening reduction proof construction fails, or the inner
/// terminal-root prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_terminal_root_fold_with_params<'stack, F, E, C, T, P, B, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    stack: &crate::compute::ProverComputeStack<'stack, F, D, B, B, B, B>,
    transcript: &mut T,
    polys: &[&P],
    incidence_summary: &ClaimIncidenceSummary,
    claim_points: &[&[E]],
    commitments: &[RingCommitment<F, D>],
    hints: Vec<AkitaCommitmentHint<F, D>>,
    root_params: &LevelParams,
    expected_w_len: usize,
    final_log_basis: u32,
    basis: BasisMode,
    _setup_contribution_mode: SetupContributionMode,
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
    E: RingSubfieldEncoding<F> + MulBaseUnreduced<F>,
    C: RingSubfieldEncoding<F>
        + ExtField<E>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
    P: RootProvePoly<F, D>,
    B: RootProveFlowBackend<F, P, E, C, D>,
{
    let opening = stack.opening();
    let opening_backend = opening.backend();
    let ring_switch = stack.ring_switch();
    validate_root_fold_inputs(
        polys.len(),
        incidence_summary,
        claim_points.len(),
        commitments.len(),
        hints.len(),
    )?;
    trace_root_fold_entry(
        "prove_terminal_root_fold_with_params",
        incidence_summary.num_claims(),
        claim_points.len(),
    );
    append_root_fold_transcript_prefix(incidence_summary, commitments, claim_points, transcript)?;

    let alpha_bits = root_params.ring_dimension.trailing_zeros() as usize;
    if let Some(prepared_reduction) = maybe_prepare_root_extension_reduction::<F, E, C, P, B, D>(
        opening_backend,
        polys,
        incidence_summary,
        claim_points,
        incidence_summary.num_vars(),
    )? {
        let extension_phase = prove_root_extension_reduction_public_phase::<F, E, C, T, P, B, D>(
            opening_backend,
            polys,
            incidence_summary,
            root_params,
            basis,
            alpha_bits,
            prepared_reduction,
            transcript,
            #[cfg(feature = "zk")]
            zk_hiding,
        )?;
        let transformed_refs = extension_phase
            .post_transform
            .transformed_polys
            .iter()
            .collect::<Vec<_>>();

        return finish_terminal_root_fold_with_prepared_openings::<
            F,
            C,
            T,
            RootTensorProjectionPoly<F, D>,
            B,
            D,
        >(
            expanded,
            stack,
            transcript,
            &transformed_refs,
            incidence_summary,
            commitments,
            hints,
            root_params,
            expected_w_len,
            final_log_basis,
            extension_phase.post_transform.prepared_points,
            extension_phase.post_transform.e_folded_by_poly,
            extension_phase.post_transform.y_rings,
            #[cfg(feature = "zk")]
            extension_phase.post_transform.y_rings_masked,
            extension_phase.row_coefficients,
            extension_phase.row_coefficient_rings,
            Some(extension_phase.post_transform.extension_opening_reduction),
            #[cfg(feature = "zk")]
            zk_hiding,
        );
    }

    let direct = prepare_root_fold_direct_public_phase::<F, E, C, T, P, B, D>(
        opening_backend,
        opening.prepared(),
        polys,
        incidence_summary,
        claim_points,
        commitments,
        hints,
        root_params,
        basis,
        alpha_bits,
        MRowLayout::WithoutDBlock,
        transcript,
        #[cfg(feature = "zk")]
        zk_hiding,
    )?;

    let commitment_rows_owned = flatten_root_commitment_rows_if_needed(commitments);
    let commitment_rows = batched_root_commitment_rows(commitments, &commitment_rows_owned);
    let public_phase::RootFoldDirectPublicPhase {
        y_rings,
        #[cfg(feature = "zk")]
        y_rings_masked,
        row_coefficients,
        instance,
        witness,
    } = direct;
    prove_terminal_root_fold_from_ring_relation::<F, C, T, B, D>(
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
    )
}
