use super::suffix::{verify_suffix, SuffixVerifierState};
use super::*;
// Top-level batched verifier orchestration once a schedule is selected.

use akita_config::{
    bind_transcript_instance_descriptor, effective_batched_schedule, ensure_schedule_fits_setup,
    CommitmentConfig,
};
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FrobeniusExtField, FromPrimitiveInt, HalvingField,
    PseudoMersenneField, RandomSampling,
};
use akita_serialization::AkitaSerialize;
use akita_transcript::Transcript;
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
    basis: BasisMode,
    schedule: &Schedule,
    setup_contribution_mode: SetupContributionMode,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField + HalvingField,
    E: FpExtEncoding<F>
        + ExtField<F>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + AkitaSerialize
        + MulBaseUnreduced<F>,
    T: Transcript<F>,
{
    let root_step = schedule.root_fold().map_err(|_| AkitaError::InvalidProof)?;
    let total_fold_levels = schedule.num_fold_levels();
    if proof.num_fold_levels() != total_fold_levels
        || proof.recursive_folds.len()
            != total_fold_levels
                .checked_sub(2)
                .ok_or(AkitaError::InvalidProof)?
    {
        return Err(AkitaError::InvalidProof);
    }
    let terminal_direct = match schedule.steps.last() {
        Some(Step::Direct(direct)) => direct,
        Some(Step::Fold(_)) | None => return Err(AkitaError::InvalidProof),
    };
    if !terminal_direct
        .witness_shape
        .admits_realized(&proof.terminal.final_witness().shape())
    {
        return Err(AkitaError::InvalidProof);
    }

    let root_execution = schedule.get_execution_schedule(0)?;
    let first_recursive_params =
        scheduled_next_level_params(schedule, 1).map_err(|_| AkitaError::InvalidProof)?;
    let root_t_state = if matches!(
        root_execution.next_witness_binding,
        Some(akita_types::NextWitnessBindingPolicy::TerminalInnerState)
    ) {
        let witness = proof
            .terminal
            .final_witness()
            .as_segment_typed()
            .ok_or(AkitaError::InvalidProof)?;
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
        basis,
        &root_step.params,
        setup_contribution_mode,
        &first_recursive_params,
        root_t_state.as_deref(),
    )?;

    let root_next_commitment = proof.root.next_w_commitment();
    match root_execution.next_witness_binding {
        Some(akita_types::NextWitnessBindingPolicy::OuterCommitment) => {
            if !root_next_commitment
                .ok_or(AkitaError::InvalidProof)?
                .can_decode_vec(first_recursive_params.role_dims().d_b())
                || root_t_state.is_some()
            {
                return Err(AkitaError::InvalidProof);
            }
        }
        Some(akita_types::NextWitnessBindingPolicy::TerminalInnerState) => {
            if root_next_commitment.is_some() || root_t_state.is_none() {
                return Err(AkitaError::InvalidProof);
            }
        }
        None => {
            return Err(AkitaError::InvalidProof);
        }
    }
    let root_next_opening = proof
        .root
        .stage3_sumcheck_proof()
        .map_or_else(|| proof.root.next_w_eval(), |stage3| stage3.next_w_eval);
    verify_suffix::<F, E, T>(
        &proof.recursive_folds,
        &proof.terminal,
        setup,
        transcript,
        schedule,
        SuffixVerifierState {
            opening_point: root_challenges,
            opening: root_next_opening,
            commitment: root_next_commitment,
            terminal_t_state: root_t_state,
            basis: BasisMode::Lagrange,
            w_len: root_step.next_w_len,
            setup_prefix_opening,
        },
    )
}

use akita_types::{
    dispatch_for_field, validate_schedule_ring_dims, AkitaBatchedProof, AkitaVerifierSetup,
    BasisMode, Commitment, FpExtEncoding, OpeningClaims, Schedule, SetupContributionMode, Step,
};

fn validate_schedule_onehot_chunk_size<Cfg: CommitmentConfig>(
    schedule: &Schedule,
) -> Result<(), AkitaError> {
    let expected = Cfg::onehot_chunk_size();
    if Cfg::decomposition().log_commit_bound != 1 || expected <= 1 {
        return Ok(());
    }
    let Some(akita_types::Step::Fold(root)) = schedule.steps.first() else {
        return Err(AkitaError::InvalidProof);
    };
    let root_params = &root.params;
    if root_params.onehot_chunk_size != expected {
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}

/// Verify a batched proof under config `Cfg`.
///
/// This is the verifier crate's top-level orchestration entrypoint. It owns
/// public claim normalization, folded schedule selection (from `Cfg`), and
/// transcript instance-descriptor binding before handing off to `verify`.
///
/// # Errors
///
/// Returns an error if public claims are malformed, schedule/layout policy
/// rejects the proof shape or proof replay fails.
pub fn batched_verify<Cfg, T>(
    proof: &AkitaBatchedProof<Cfg::Field, Cfg::ExtField>,
    setup: &AkitaVerifierSetup<Cfg::Field>,
    transcript: &mut T,
    claims: OpeningClaims<'_, Cfg::ExtField, &Commitment<Cfg::Field>>,
    basis: BasisMode,
    setup_contribution_mode: SetupContributionMode,
) -> Result<(), AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField + HalvingField,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + FrobeniusExtField<Cfg::Field>
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<Cfg::Field>,
{
    claims
        .validate(setup.expanded.seed())
        .map_err(|_| AkitaError::InvalidProof)?;
    let opening_batch = claims.layout().map_err(|_| AkitaError::InvalidProof)?;
    let schedule = effective_batched_schedule::<Cfg>(&opening_batch, claims.point())
        .map_err(|_| AkitaError::InvalidProof)?;
    validate_schedule_ring_dims(&schedule, setup.expanded.seed())?;
    ensure_schedule_fits_setup::<Cfg>(setup.expanded.as_ref(), &schedule, &opening_batch)?;
    schedule
        .validate_structure()
        .map_err(|_| AkitaError::InvalidProof)?;
    validate_schedule_onehot_chunk_size::<Cfg>(&schedule)?;

    // The transcript instance descriptor binds the setup-wide root ring
    // dimension (`gen_ring_dim`), which is byte-identical to the const `Cfg::D`
    // the prover binds for uniform-D presets. Dispatch on the runtime value so
    // the verifier entry stays D-free; the descriptor bytes are unchanged.
    dispatch_for_field!(
        akita_types::ProtocolDispatchSlot::Envelope,
        Cfg::Field,
        setup.expanded.seed().gen_ring_dim,
        |D| {
            bind_transcript_instance_descriptor::<Cfg::Field, T, D, Cfg>(
                &setup.expanded,
                &opening_batch,
                &schedule,
                basis,
                transcript,
            )
        }
    )?;

    verify::<Cfg::Field, Cfg::ExtField, T>(
        proof,
        setup,
        transcript,
        claims,
        basis,
        &schedule,
        setup_contribution_mode,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Fp32;
    use akita_types::{RingVec, RingView};

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
