use super::suffix::{verify_suffix, SuffixVerifierState, SuffixWitnessState};
use super::*;
// Top-level batched verifier orchestration once a schedule is selected.

use akita_config::{
    bind_transcript_instance_descriptor, ensure_verifier_schedule_fits_setup, CommitmentConfig,
    TrustedScheduleCatalog,
};
use akita_error::AkitaError;
use akita_serialization::{AkitaSerialize, Valid};
use akita_transcript::Transcript;
use jolt_field::{CanonicalEncoding, ExtField, Field, PseudoMersenne, Ring};

/// Reject malformed proof carriers against the selected schedule before any
/// transcript replay or proof-owned buffer is cloned.
fn validate_proof_against_schedule<F, E>(
    proof: &AkitaBatchedProof<F, E>,
    schedule: &FoldSchedule,
) -> Result<(), AkitaError>
where
    F: Field + Valid,
    E: Field + Valid,
{
    proof.check().map_err(|_| AkitaError::InvalidProof)?;

    let total_fold_levels = schedule.num_fold_levels();
    if proof.num_fold_levels() != total_fold_levels
        || proof.recursive_folds.len()
            != total_fold_levels
                .checked_sub(2)
                .ok_or(AkitaError::InvalidProof)?
    {
        return Err(AkitaError::InvalidProof);
    }

    let validate_nonterminal = |fold: &FoldLevelProof<F, E>,
                                params: &CommittedGroupParams,
                                next_params: Option<&CommittedGroupParams>,
                                binding: akita_types::NextWitnessBindingPolicy|
     -> Result<(), AkitaError> {
        if matches!(
            params.opening_method(),
            akita_types::OpeningMethod::SubringCoefficientPacking { .. }
        ) && fold.extension_opening_reduction().is_some()
        {
            return Err(AkitaError::InvalidProof);
        }
        if fold.opening_payload.coeff_len()
            != params
                .opening_payload_geometry()?
                .transmitted_coefficients()
        {
            return Err(AkitaError::InvalidProof);
        }

        match (binding, &fold.stage2.next_witness_binding) {
            (
                akita_types::NextWitnessBindingPolicy::OuterPayload,
                akita_types::NextWitnessBinding::OuterPayload(commitment),
            ) => {
                let next_params = next_params.ok_or(AkitaError::InvalidProof)?;
                if commitment.coeff_len()
                    != next_params
                        .outer_payload_geometry()?
                        .transmitted_coefficients()
                {
                    return Err(AkitaError::InvalidProof);
                }
            }
            (
                akita_types::NextWitnessBindingPolicy::TerminalInnerState,
                akita_types::NextWitnessBinding::TerminalInnerState,
            ) => {}
            _ => return Err(AkitaError::InvalidProof),
        }
        Ok(())
    };

    let (root_next, root_binding) = schedule.recursive_folds.first().map_or(
        (
            None,
            akita_types::NextWitnessBindingPolicy::TerminalInnerState,
        ),
        |step| {
            (
                Some(&step.params),
                akita_types::NextWitnessBindingPolicy::OuterPayload,
            )
        },
    );
    validate_nonterminal(&proof.root, &schedule.root.params, root_next, root_binding)?;
    for (index, (fold, step)) in proof
        .recursive_folds
        .iter()
        .zip(&schedule.recursive_folds)
        .enumerate()
    {
        let (next, binding) = schedule.recursive_folds.get(index + 1).map_or(
            (
                None,
                akita_types::NextWitnessBindingPolicy::TerminalInnerState,
            ),
            |next| {
                (
                    Some(&next.params),
                    akita_types::NextWitnessBindingPolicy::OuterPayload,
                )
            },
        );
        validate_nonterminal(fold, &step.params, next, binding)?;
    }

    let terminal_shape = &schedule.terminal.response_shape;
    if !terminal_shape
        .layout
        .admits_realized(&proof.terminal.terminal_response().layout)
    {
        return Err(AkitaError::InvalidProof);
    }

    Ok(())
}

/// Verify a prepared folded batched proof once the schedule and transcript
/// descriptor are fixed.
///
/// # Errors
///
/// Returns an error if the schedule and proof shapes disagree or any root or
/// suffix verification step rejects.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(crate) fn verify<F, E, T>(
    proof: &AkitaBatchedProof<F, E>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    claims: OpeningClaims<'_, E, &Commitment<F>>,
    opening_batch: &OpeningClaimsLayout,
    basis: BasisMode,
    schedule: &FoldSchedule,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + PseudoMersenne,
    E: FpExtEncoding<F> + ExtField<F> + Ring + AkitaSerialize + MulBaseUnreduced<F>,
    T: akita_types::VerifierTranscriptGrinding<F>,
{
    let root_step = schedule.root_fold();
    let first_recursive_params = schedule.recursive_folds.first();
    let root_t_state = if first_recursive_params.is_none() {
        let witness = proof.terminal.terminal_response();
        let t_state = raw_field_segment_bytes(&witness.t_fields)?;
        if t_state.is_empty() {
            return Err(AkitaError::InvalidProof);
        }
        Some(t_state)
    } else {
        None
    };
    let (root_challenges, setup_prefix_opening) = verify_root::<F, E, T>(
        &proof.root,
        setup,
        transcript,
        &claims,
        opening_batch,
        basis,
        &root_step.params,
        first_recursive_params,
        first_recursive_params.map_or(schedule.terminal.d_a(), |step| step.params.d_a()),
        root_t_state.as_deref(),
    )
    .map_err(|error| {
        AkitaError::InvalidInput(format!("compressed root replay failed: {error:?}"))
    })?;

    let root_next_commitment = proof.root.next_w_payload();
    let root_witness = match (root_next_commitment, root_t_state) {
        (Some(commitment), None) => SuffixWitnessState::Commitment(commitment),
        (None, Some(t_state)) => SuffixWitnessState::TerminalT(t_state),
        _ => return Err(AkitaError::InvalidProof),
    };
    verify_suffix::<F, E, T>(
        &proof.recursive_folds,
        &proof.terminal,
        setup,
        transcript,
        schedule,
        SuffixVerifierState {
            opening_point: root_challenges,
            opening: proof.root.next_w_eval(),
            witness: root_witness,
            basis: BasisMode::Lagrange,
            witness_len: root_step.output_witness_len,
            setup_prefix_opening,
        },
    )
}

use akita_types::{
    validate_schedule_ring_dims, AkitaBatchedProof, AkitaVerifierSetup, BasisMode, Commitment,
    CommittedGroupBatchProfile, FoldSchedule, FpExtEncoding, GroupBatchStatement, OpeningClaims,
    PolynomialGroupClaims,
};

/// Verify a batched proof under config `Cfg`.
///
/// This is the verifier crate's top-level orchestration entrypoint. It owns
/// public claim normalization, folded schedule selection from the trusted
/// catalog, and
/// transcript instance-descriptor binding before handing off to `verify`.
///
/// # Errors
///
/// Returns an error if public claims are malformed, schedule/layout policy
/// rejects the proof shape or proof replay fails.
pub fn batched_verify<Cfg, T>(
    proof: &AkitaBatchedProof<Cfg::Field, Cfg::ExtField>,
    setup: &AkitaVerifierSetup<Cfg::Field>,
    schedules: &TrustedScheduleCatalog,
    transcript: &mut T,
    statement: GroupBatchStatement<'_, Cfg::ExtField, Cfg::Field>,
    basis: BasisMode,
) -> Result<(), AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: Field
        + CanonicalEncoding
        + akita_serialization::AkitaSerialize
        + PseudoMersenne
        + Field
        + Valid,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field> + ExtField<Cfg::Field> + Ring + AkitaSerialize + Valid,
    T: Transcript<Cfg::Field>,
{
    let selection = statement.selection();
    let claims = statement.into_claims();
    claims
        .validate(setup.expanded.descriptor())
        .map_err(|_| AkitaError::InvalidProof)?;
    let opening_batch = claims
        .committed_layout()
        .map_err(|_| AkitaError::InvalidProof)?;
    let (final_group, precommitteds) = claims
        .groups()
        .split_last()
        .ok_or(AkitaError::InvalidProof)?;
    let final_descriptor = *final_group.commitment().profile();
    if final_descriptor.group.num_vars() != final_group.num_vars()
        || final_descriptor.group.num_polynomials() != final_group.num_evaluations()
        || precommitteds.iter().any(|group| {
            let descriptor = group.commitment().profile();
            descriptor.group.num_vars() != group.num_vars()
                || descriptor.group.num_polynomials() != group.num_evaluations()
        })
    {
        return Err(AkitaError::InvalidProof);
    }
    for group in claims.groups() {
        let committed = group.commitment();
        let descriptor = committed.profile();
        descriptor
            .validate_frozen_precommit(Cfg::decomposition().field_bits())
            .map_err(|_| AkitaError::InvalidProof)?;
        let source_coefficients = descriptor
            .outer_slice_count
            .complete_source_coefficients(
                descriptor.outer.matrix.output_rank(),
                descriptor.outer.matrix.ring_dimension(),
            )
            .map_err(|_| AkitaError::InvalidProof)?;
        let plan = akita_types::CompressionChainPlan::for_complete_source(
            descriptor.outer.matrix.sis_table_key().modulus_profile,
            source_coefficients,
        )?;
        if committed.commitment().rows().coeff_len() != plan.terminal_coefficients() {
            return Err(AkitaError::InvalidProof);
        }
    }
    let batch_profile = CommittedGroupBatchProfile {
        final_group: final_descriptor,
        precommitteds: precommitteds
            .iter()
            .map(|group| *group.commitment().profile())
            .collect(),
    };
    batch_profile
        .validate(Cfg::decomposition().field_bits())
        .map_err(|_| AkitaError::InvalidProof)?;
    let resolved = schedules.resolve_selection(selection)?;
    resolved
        .validate_opening_layout(&opening_batch)
        .map_err(|_| AkitaError::InvalidProof)?;
    if resolved.profiles() != &batch_profile {
        return Err(AkitaError::InvalidProof);
    }
    let schedule = resolved.schedule();
    let root_params = &schedule.root_fold().params;
    let expected_final_descriptor =
        akita_types::GroupCommitPhaseParams::try_from_params(final_descriptor.group, root_params)
            .map_err(|_| AkitaError::InvalidProof)?;
    if final_descriptor != expected_final_descriptor
        || root_params.precommitted_groups().len() != precommitteds.len()
        || root_params
            .precommitted_groups()
            .iter()
            .zip(precommitteds)
            .any(|(params, claims_group)| params.profile != *claims_group.commitment().profile())
    {
        return Err(AkitaError::InvalidProof);
    }
    validate_schedule_ring_dims(schedule)?;
    ensure_verifier_schedule_fits_setup(setup.expanded.as_ref(), schedule, &opening_batch)?;
    schedule
        .validate_nonterminal_opening_execution(Cfg::EXT_DEGREE)
        .map_err(|_| AkitaError::InvalidProof)?;
    validate_proof_against_schedule(proof, schedule).map_err(|error| {
        AkitaError::InvalidInput(format!(
            "proof does not match the selected compressed schedule: {error:?}"
        ))
    })?;

    // Schedule resolution is the earliest point at which the terminal ring
    // dimension, A widths, and exact base-versus-i16-tail capabilities are all
    // known. Warm those derived, non-serialized prefixes before transcript
    // replay so terminal verification performs cache lookup only.
    super::terminal_ntt::warm_for_schedule(setup, schedule)?;

    let grinding_plan = {
        let _span = tracing::info_span!("verifier_transcript_bind_instance").entered();
        bind_transcript_instance_descriptor::<Cfg::Field, T, Cfg>(
            &setup.expanded,
            &opening_batch,
            selection,
            schedule,
            basis,
            transcript,
        )?
    };
    let mut grinding_transcript = akita_types::VerifierGrindingTranscript::<T>::new(
        transcript,
        &proof.nonce_stream,
        &grinding_plan,
    )?;

    let raw_groups = claims
        .groups()
        .iter()
        .map(|group| {
            PolynomialGroupClaims::new(
                group.point().to_vec(),
                group.evaluations().to_vec(),
                group.commitment().commitment(),
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| AkitaError::InvalidProof)?;
    let raw_claims =
        OpeningClaims::from_groups(raw_groups).map_err(|_| AkitaError::InvalidProof)?;
    verify::<Cfg::Field, Cfg::ExtField, _>(
        proof,
        setup,
        &mut grinding_transcript,
        raw_claims,
        &opening_batch,
        basis,
        schedule,
    )
    .and_then(|()| grinding_transcript.finish())
    .map_err(|error| AkitaError::InvalidInput(format!("compressed proof replay failed: {error:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_types::{RingVec, RingView};
    use jolt_field::{Fp32, Zero};

    type F = Fp32<251>;
    const D: usize = 32;

    /// The D-free commitment read path validates the flat coefficient length
    /// against the schedule-derived ring dimension via `RingView::new` and
    /// returns an error (never panics) when the length is not a multiple of the
    /// ring dimension. This is the no-panic gate the verifier relies on before
    /// interpreting any ring-shaped commitment.
    #[test]
    fn flat_commitment_length_not_multiple_of_ring_dim_rejects() {
        // 33 coefficients is not a multiple of D = 32.
        let commitment = RingVec::from_coeffs(vec![F::zero(); D + 1]);
        let err = RingView::new(commitment.coeffs(), D)
            .expect_err("commitment length must be a multiple of the ring dimension");
        assert!(matches!(err, AkitaError::InvalidProof));

        // A well-formed buffer (2 * D) is accepted and yields the expected ring count.
        let well_formed = vec![F::zero(); 2 * D];
        let ok = RingView::new(&well_formed, D).expect("valid flat commitment");
        assert_eq!(ok.num_rings(), 2);
    }
}
