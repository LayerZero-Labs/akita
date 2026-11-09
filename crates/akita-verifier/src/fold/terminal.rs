//! Terminal fold dispatch: replay the terminal transcript and hand the revealed
//! response to the direct terminal checks.

use super::suffix::SuffixVerifierState;
use super::NextWitness;
use crate::prepared_cache::TerminalNttCache;
use crate::stages::opening_claims::verify_extension_claim_terminal_suffix;
use akita_challenges::FoldDraw;
use akita_error::AkitaError;
use akita_params::{OpeningClaimsLayout, TerminalFoldParams};
use akita_serialization::AkitaSerialize;
use akita_transcript::ChallengeCodec;
use akita_types::GrindingReplay;
use akita_types::{prepare_opening_point, FpExtEncoding, RingVec};
use jolt_algebra::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, PseudoMersenne, Ring};

pub(super) fn verify_terminal_suffix<F, E>(
    terminal_ntt: &TerminalNttCache,
    grinding: &mut akita_types::VerifierGrinding<'_, '_>,
    level: u32,
    current_state: &SuffixVerifierState<F, E>,
    scheduled: &TerminalFoldParams,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + PseudoMersenne,
    E: FpExtEncoding<F>
        + ExtField<F>
        + Ring
        + AkitaSerialize
        + MulBaseUnreduced<F>
        + ChallengeCodec,
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
    let NextWitness::TerminalT {
        sent: t_sent,
        last_block_digest: t_last_block_digest,
    } = &current_state.witness
    else {
        return Err(AkitaError::Internal(
            "terminal scheduled after an outer payload witness".into(),
        ));
    };
    if scheduled.response_shape.layout.groups.len() != 1
        || t_sent.coeff_len() != group.t_field_elems
        || !t_sent.can_decode_vec(scheduled.d_a())
    {
        return Err(AkitaError::InvalidProof);
    }
    // The `t` prefix and its last-block digest were absorbed when they were
    // received as the next witness.
    let recursive_num_vars = scheduled.recursive_opening_num_vars()?;
    if current_state.opening_point.len() > recursive_num_vars {
        return Err(AkitaError::InvalidProof);
    }
    let opening_batch = OpeningClaimsLayout::new(current_state.opening_point.len(), 1)?;
    let (prepared_point, protocol_point, final_relation) = if const { <E as ExtField<F>>::DEGREE == 1 }
    {
        let prepared = akita_params::dispatch_for_field!(
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
        let replay = verify_extension_claim_terminal_suffix::<F, E>(
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
    akita_transcript::public_extensions::<F, E, _>(
        grinding.state_mut(),
        akita_types::FoldSite::TerminalPoint { level }.id()?,
        &protocol_point,
    )?;
    if final_relation.is_none() {
        akita_transcript::public_extensions::<F, E, _>(
            grinding.state_mut(),
            akita_types::FoldSite::TerminalOpening { level }.id()?,
            std::slice::from_ref(&current_state.opening),
        )?;
    }
    let row_coefficients = akita_types::row_coefficients::<F, E, _>(
        &opening_batch,
        akita_params::GrindingSite::EvaluationBatch { level },
        grinding,
    )?;
    if row_coefficients.as_slice() != [E::one()] {
        return Err(AkitaError::InvalidProof);
    }
    let e_fields = akita_transcript::receive_field_group::<F>(
        grinding.state_mut(),
        akita_types::FoldSite::TerminalEFields { level }.id()?,
        group.e_field_elems,
    )
    .map(RingVec::from_coeffs)?;
    // Both segments omit their last block. The predecessor fold bound the `t`
    // digest next to the carried state, and the `e` digest arrives here, before
    // the fold challenges.
    let last_block_digests = crate::terminal::direct::LastBlockDigests {
        t: *t_last_block_digest,
        e: akita_transcript::receive_field_digest(
            grinding.state_mut(),
            akita_types::FoldSite::TerminalEDigest { level }.id()?,
        )?,
    };
    grinding.read_fold_response(akita_params::GrindingSite::FoldResponse { level })?;
    let family = scheduled.fold_challenge_family()?;
    let challenges = {
        let mut draw = akita_challenges::VerifierFoldDraw::new(grinding.state_mut(), level, 0);
        draw.draw_fold_challenges(&family, scheduled.blocks.live_blocks, 1)?
            .ok_or(AkitaError::InvalidProof)?
    };
    grinding.record_fold_challenges(level, 0, scheduled.blocks.live_blocks)?;
    let z_payload = akita_transcript::receive_bounded_bytes(
        grinding.state_mut(),
        akita_types::FoldSite::TerminalZPayload { level }.id()?,
        group.z_payload_bytes,
    )?;
    scheduled.validate_terminal_linf_cap(group.z_linf_cap)?;
    let terminal_response = akita_types::TerminalResponse {
        layout: scheduled.response_shape.layout.clone(),
        z_payloads: vec![z_payload],
        e_fields,
        t_fields: t_sent.clone(),
    };
    let e_fields = crate::terminal::direct::verify_terminal_ring_relations(
        terminal_ntt,
        &challenges,
        &prepared_point.ring_multiplier_point,
        scheduled,
        &terminal_response,
        &last_block_digests,
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
        &e_fields,
        &prepared_point,
        &[E::one()],
        None,
        scale,
        target,
    )
}
