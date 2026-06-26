use super::*;
use akita_types::terminal_witness_segment_layout;

/// Verify the folded-root proof payload for either an intermediate root or the
/// 1-fold terminal root.
///
/// This replays the canonical root transcript layout: batch-shape header,
/// commitments, padded opening points, per-claim field openings, row
/// coefficients, EOR if present, y-rings, ring switch, stage-1 when present,
/// stage-2, and stage-3 setup sumcheck when required by the intermediate branch.
///
/// # Errors
///
/// Returns an error if the proof shape is inconsistent, any public trace check
/// fails, ring-switch replay fails, or a sumcheck verifier rejects.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(super) fn verify_root<F, E, T, const D: usize>(
    proof: &AkitaBatchedRootProof<F, E>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    claims: &VerifierOpeningBatch<'_, E, &RingCommitment<F, D>>,
    basis: BasisMode,
    root_lp: &LevelParams,
    setup_contribution_mode: SetupContributionMode,
    next_fold_level_params: Option<&LevelParams>,
    terminal_final_w_len: usize,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HalvingField,
    E: FpExtEncoding<F> + ExtField<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    validate_level_dispatch::<D>(root_lp)?;
    let m_row_layout = proof.fold_m_row_layout().ok_or(AkitaError::InvalidProof)?;
    let extension_opening_reduction = proof.fold_extension_opening_reduction();
    let v_typed = proof.fold_v::<D>()?;
    let next_fold_level_params = match proof {
        AkitaBatchedRootProof::Fold(_) => next_fold_level_params.ok_or(AkitaError::InvalidProof)?,
        AkitaBatchedRootProof::Terminal(_) => root_lp,
        AkitaBatchedRootProof::ZeroFold { .. } => return Err(AkitaError::InvalidProof),
    };
    let stage3_sumcheck_proof = proof.fold_stage3_sumcheck_proof(setup_contribution_mode)?;
    let commitment = claims
        .single_group_commitment()
        .copied()
        .ok_or(AkitaError::InvalidProof)?;
    let openings = claims.claims();
    let opening_batch = claims.to_shape();
    let shared_opening_point = claims.point();
    let num_claims = opening_batch.num_polynomials();
    if openings.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    if commitment.u.len() != root_lp.effective_commit_rows() {
        return Err(AkitaError::InvalidProof);
    }

    claims.append_to_transcript::<F, T>(transcript)?;
    if extension_opening_reduction.is_none() {
        let prepared_point = prepare_opening_point::<F, E, D>(
            shared_opening_point,
            basis,
            root_lp,
            root_lp.ring_dimension.trailing_zeros() as usize,
            BlockOrder::RowMajor,
        )?;
        for pt in &prepared_point.padded_point {
            append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, pt);
        }
    }
    append_claim_values_to_transcript::<F, E, T>(&openings, transcript);
    let row_coefficients = sample_public_row_coefficients::<F, E, T>(&opening_batch, transcript)?;
    let root_eor = verify_fold_eor::<F, E, T, D>(
        extension_opening_reduction,
        &[],
        shared_opening_point,
        &openings,
        &row_coefficients,
        &opening_batch,
        basis,
        root_lp,
        BlockOrder::RowMajor,
        false,
        transcript,
    )?;
    let reduction_check = root_eor.reduction_challenges;
    let prepared_points = root_eor.prepared_points;
    let eor_trace_final = root_eor.final_relation;
    let prepared_point = prepared_points.first().ok_or(AkitaError::InvalidProof)?;
    if extension_opening_reduction.is_some() {
        for pt in &prepared_point.padded_point {
            append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, pt);
        }
    }
    let trace_block_opening: Vec<E> = if let Some(rho) = &reduction_check {
        let protocol_point =
            ring_subfield_packed_extension_opening_point::<F, E, D>(rho.len(), rho)?;
        root_trace_block_opening::<E>(
            &protocol_point,
            root_lp,
            root_lp.ring_dimension.trailing_zeros() as usize,
        )?
    } else {
        root_trace_block_opening::<E>(
            shared_opening_point,
            root_lp,
            root_lp.ring_dimension.trailing_zeros() as usize,
        )?
    };
    let ordinary_trace_eval_target =
        batched_eval_target_from_opening_batch(&opening_batch, &row_coefficients, &openings)?;
    let trace_eval_target = eor_trace_final
        .as_ref()
        .map(|(final_claim, _)| *final_claim)
        .unwrap_or(ordinary_trace_eval_target);
    let trace_claim_scales = eor_trace_final
        .as_ref()
        .map(|(_, factors_by_point)| {
            let shared_factor = *factors_by_point.first().ok_or(AkitaError::InvalidProof)?;
            Ok(vec![shared_factor; opening_batch.num_polynomials()])
        })
        .transpose()?;

    let w_len = match proof {
        AkitaBatchedRootProof::Terminal(_) => terminal_final_w_len,
        AkitaBatchedRootProof::Fold(_) => {
            w_ring_element_count_with_counts::<F>(root_lp, opening_batch.num_polynomials(), 1)?
                .checked_mul(D)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("next witness length overflow".to_string())
                })?
        }
        AkitaBatchedRootProof::ZeroFold { .. } => return Err(AkitaError::InvalidProof),
    };
    let terminal_replay = match proof {
        AkitaBatchedRootProof::Terminal(terminal) => {
            let final_witness = terminal
                .stage2
                .final_witness()
                .ok_or(AkitaError::InvalidProof)?;
            let layout =
                terminal_witness_segment_layout(root_lp, num_claims, 1, F::modulus_bits())?;
            Some(prepare_terminal_witness_replay::<F, T>(
                transcript,
                final_witness,
                w_len,
                layout,
            )?)
        }
        AkitaBatchedRootProof::Fold(_) => None,
        AkitaBatchedRootProof::ZeroFold { .. } => return Err(AkitaError::InvalidProof),
    };

    let stage1_proof = proof.fold_stage1()?;
    let next_w_commitment = proof.fold_next_w_commitment()?;
    let stage2 = proof.fold_stage2()?;
    let fold_grind_nonce = proof.fold_grind_nonce()?;
    let replay_opening_batch = VerifierOpeningBatch::from_shape_and_groups(
        shared_opening_point,
        opening_batch.clone(),
        vec![CommitmentGroup {
            claims: openings,
            commitment: commitment.u.as_slice(),
        }],
    )?;
    let prepared = PreparedFoldReplay {
        lp: root_lp,
        m_row_layout,
        fold_grind_nonce,
        v: proof.fold_v_buf()?,
        opening_batch: replay_opening_batch,
        row_coefficients,
        ring_opening_point: prepared_point.ring_opening_point.clone(),
        ring_multiplier_point: prepared_point.ring_multiplier_point.clone(),
        w_len,
        stage1: stage1_proof,
        stage2,
        next_w_commitment,
        terminal_replay,
        stage3: stage3_sumcheck_proof.map(|proof| (proof, next_fold_level_params)),
        trace_prepared_point: Some(prepared_point.clone()),
        trace_block_opening: Some(trace_block_opening),
        trace_eval_target,
        trace_eval_scale: E::one(),
        trace_claim_scales,
        trace_basis: basis,
    };
    verify_fold::<F, E, T, D>(setup, transcript, prepared)
}
