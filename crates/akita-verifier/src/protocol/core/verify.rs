use super::suffix::{verify_suffix, SuffixVerifierState, SuffixWitnessState};
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
use akita_serialization::{AkitaSerialize, Valid};
use akita_transcript::Transcript;

/// Reject malformed proof carriers against the selected schedule before any
/// transcript replay or proof-owned buffer is cloned.
fn validate_proof_against_schedule<F, E>(
    proof: &AkitaBatchedProof<F, E>,
    schedule: &Schedule,
) -> Result<(), AkitaError>
where
    F: FieldCore + Valid,
    E: FieldCore + Valid,
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

    for (level, fold) in proof.nonterminal_folds().enumerate() {
        let scheduled = schedule
            .get_execution_schedule(level)
            .map_err(|_| AkitaError::InvalidProof)?;
        if scheduled.is_terminal {
            return Err(AkitaError::InvalidProof);
        }
        let expected_v_coeffs = scheduled
            .params
            .d_key
            .row_len()
            .checked_mul(scheduled.params.role_dims().d_d())
            .ok_or(AkitaError::InvalidProof)?;
        if fold.v.coeff_len() != expected_v_coeffs {
            return Err(AkitaError::InvalidProof);
        }

        match (
            scheduled.next_witness_binding,
            &fold.stage2.next_witness_binding,
        ) {
            (
                Some(akita_types::NextWitnessBindingPolicy::OuterCommitment),
                akita_types::NextWitnessBinding::OuterCommitment(commitment),
            ) => {
                let next_params = scheduled
                    .next_params
                    .as_ref()
                    .ok_or(AkitaError::InvalidProof)?;
                let expected_coeffs = next_params
                    .b_key
                    .row_len()
                    .checked_mul(next_params.role_dims().d_b())
                    .ok_or(AkitaError::InvalidProof)?;
                if commitment.coeff_len() != expected_coeffs {
                    return Err(AkitaError::InvalidProof);
                }
            }
            (
                Some(akita_types::NextWitnessBindingPolicy::TerminalInnerState),
                akita_types::NextWitnessBinding::TerminalInnerState,
            ) => {}
            _ => return Err(AkitaError::InvalidProof),
        }
    }

    let terminal_shape = &schedule.terminal.witness_shape;
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
    basis: BasisMode,
    schedule: &Schedule,
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
    let root_execution = schedule.get_execution_schedule(0)?;
    let first_recursive_params = schedule.folds.get(1).ok_or(AkitaError::InvalidProof)?;
    let root_t_state = if matches!(
        root_execution.next_witness_binding,
        Some(akita_types::NextWitnessBindingPolicy::TerminalInnerState)
    ) {
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
        basis,
        &root_step.params,
        &first_recursive_params.params,
        root_t_state.as_deref(),
    )?;

    let root_next_commitment = proof.root.next_w_commitment();
    let root_witness = match (root_next_commitment, root_t_state) {
        (Some(commitment), None) => SuffixWitnessState::Commitment(commitment),
        (None, Some(t_state)) => SuffixWitnessState::TerminalT(t_state),
        _ => return Err(AkitaError::InvalidProof),
    };
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
            witness: root_witness,
            basis: BasisMode::Lagrange,
            w_len: root_step.next_w_len,
            setup_prefix_opening,
        },
    )
}

use akita_types::{
    dispatch_for_field, validate_schedule_ring_dims, AkitaBatchedProof, AkitaVerifierSetup,
    BasisMode, Commitment, FpExtEncoding, OpeningClaims, Schedule,
};

fn validate_schedule_onehot_chunk_size<Cfg: CommitmentConfig>(
    schedule: &Schedule,
) -> Result<(), AkitaError> {
    let expected = Cfg::onehot_chunk_size();
    if Cfg::decomposition().log_commit_bound != 1 || expected <= 1 {
        return Ok(());
    }
    let root = schedule.root_fold().map_err(|_| AkitaError::InvalidProof)?;
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
) -> Result<(), AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field:
        FieldCore + CanonicalField + RandomSampling + PseudoMersenneField + HalvingField + Valid,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + FrobeniusExtField<Cfg::Field>
        + FromPrimitiveInt
        + AkitaSerialize
        + Valid,
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
    validate_proof_against_schedule(proof, &schedule)?;

    // The transcript instance descriptor binds the setup-wide root ring
    // dimension (`gen_ring_dim`), which is byte-identical to the const `Cfg::D`
    // the prover binds for uniform-D presets. Dispatch on the runtime value so
    // the verifier entry stays D-free; the descriptor bytes are unchanged.
    {
        let _span = tracing::info_span!("verifier_transcript_bind_instance").entered();
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
    }

    verify::<Cfg::Field, Cfg::ExtField, T>(proof, setup, transcript, claims, basis, &schedule)
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
