//! Terminal fold dispatch: replay the terminal transcript and hand the revealed
//! response to the direct terminal checks.

use super::suffix::NativeSuffixVerifierState;
use crate::prepared_cache::TerminalNttCache;
use crate::stages::opening_claims::verify_extension_claim_terminal_suffix_native;
use akita_challenges::FoldDraw;
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::NativeGrinding;
use akita_types::{
    prepare_opening_point, FpExtEncoding, OpeningClaimsLayout, RingVec, TerminalFoldParams,
};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, PseudoMersenne, Ring};

pub(super) fn verify_terminal_suffix_native<F, E>(
    terminal_ntt: &TerminalNttCache,
    grinding: &mut akita_types::NativeVerifierGrinding<'_, '_>,
    level: u32,
    current_state: &NativeSuffixVerifierState<F, E>,
    scheduled: &TerminalFoldParams,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + PseudoMersenne,
    E: FpExtEncoding<F> + ExtField<F> + Ring + AkitaSerialize + MulBaseUnreduced<F>,
{
    if current_state.witness_len != scheduled.input_witness_len
        || current_state.setup_prefix_opening.is_some()
    {
        return Err(AkitaError::InvalidProof);
    }
    let group = scheduled
        .response_shape
        .layout
        .groups
        .first()
        .ok_or(AkitaError::InvalidProof)?;
    if scheduled.response_shape.layout.groups.len() != 1
        || current_state.witness.coeff_len() != group.t_field_elems
        || !current_state.witness.can_decode_vec(scheduled.d_a())
    {
        return Err(AkitaError::InvalidProof);
    }
    akita_transcript::public_native_fields_verifier(
        grinding.state_mut(),
        akita_types::NativeFoldSite::TerminalTFields { level }.id()?,
        current_state.witness.coeffs(),
    )?;
    let recursive_num_vars = scheduled.recursive_opening_num_vars()?;
    if current_state.opening_point.len() > recursive_num_vars {
        return Err(AkitaError::InvalidProof);
    }
    let opening_batch = OpeningClaimsLayout::new(current_state.opening_point.len(), 1)?;
    let (prepared_point, protocol_point, final_relation) = if const { <E as ExtField<F>>::DEGREE == 1 }
    {
        let prepared = akita_types::dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            scheduled.d_a(),
            |D| {
                prepare_opening_point::<F, E, D>(
                    &current_state.opening_point,
                    current_state.basis,
                    scheduled.blocks.positions_per_block,
                    scheduled.blocks.live_blocks,
                    scheduled.d_a().trailing_zeros() as usize,
                )
            }
        )?;
        (prepared, current_state.opening_point.clone(), None)
    } else {
        let replay = verify_extension_claim_terminal_suffix_native::<F, E>(
            &current_state.opening_point,
            current_state.opening,
            &opening_batch,
            current_state.basis,
            scheduled,
            grinding,
            level,
        )?;
        let group = replay
            .groups
            .into_iter()
            .next()
            .ok_or(AkitaError::InvalidProof)?;
        (group.prepared, group.protocol, replay.final_relation)
    };
    akita_transcript::public_native_extensions::<F, E, _>(
        grinding.state_mut(),
        akita_types::NativeFoldSite::TerminalPoint { level }.id()?,
        &protocol_point,
    )?;
    if final_relation.is_none() {
        akita_transcript::public_native_extensions::<F, E, _>(
            grinding.state_mut(),
            akita_types::NativeFoldSite::TerminalOpening { level }.id()?,
            std::slice::from_ref(&current_state.opening),
        )?;
    }
    let row_coefficients = akita_types::row_coefficients_native::<F, E, _>(
        &opening_batch,
        akita_types::GrindingSite::EvaluationBatch { level },
        grinding,
    )?;
    if row_coefficients.as_slice() != [E::one()] {
        return Err(AkitaError::InvalidProof);
    }
    let e_fields = akita_transcript::receive_native_field_group::<F>(
        grinding.state_mut(),
        akita_types::NativeFoldSite::TerminalEFields { level }.id()?,
        group.e_field_elems,
    )
    .map(RingVec::from_coeffs)?;
    grinding.read_fold_response(akita_types::GrindingSite::FoldResponse { level })?;
    let operator_rejection = if scheduled.response_l2_sq_cap().is_some() {
        Some(
            akita_challenges::selective_l2_operator_norm_rejection(
                scheduled.d_a(),
                &scheduled.fold_challenge_config,
            )
            .ok_or(AkitaError::InvalidProof)?,
        )
    } else {
        None
    };
    let challenges = {
        let mut draw =
            akita_challenges::NativeVerifierFoldDraw::new(grinding.state_mut(), level, 0);
        draw.draw_folding_challenges_with_rejection(
            akita_challenges::FoldChallengeDrawDomain::EvaluationTrace,
            scheduled.d_a(),
            0,
            scheduled.blocks.live_blocks,
            1,
            &scheduled.fold_challenge_config,
            operator_rejection,
        )?
    };
    grinding.record_fold_challenges(level, 0, scheduled.blocks.live_blocks)?;
    let z_payload = akita_transcript::receive_native_bounded_bytes(
        grinding.state_mut(),
        akita_types::NativeFoldSite::TerminalZPayload { level }.id()?,
        group.z_payload_bytes,
    )?;
    scheduled.validate_terminal_linf_cap(group.z_linf_cap)?;
    let terminal_response = akita_types::TerminalResponse {
        layout: scheduled.response_shape.layout.clone(),
        z_payloads: vec![z_payload],
        e_fields,
        t_fields: current_state.witness.clone(),
    };
    crate::terminal::direct::verify_terminal_ring_relations(
        terminal_ntt,
        &challenges,
        &prepared_point.ring_multiplier_point,
        scheduled,
        &terminal_response,
    )?;
    let (target, scale) = match final_relation {
        Some((claims, factors)) => (
            *claims.first().ok_or(AkitaError::InvalidProof)?,
            *factors.first().ok_or(AkitaError::InvalidProof)?,
        ),
        None => (current_state.opening, E::one()),
    };
    crate::terminal::direct::verify_terminal_trace(
        &prepared_point.ring_multiplier_point,
        scheduled,
        &terminal_response,
        &prepared_point,
        &[E::one()],
        None,
        scale,
        target,
    )
}
