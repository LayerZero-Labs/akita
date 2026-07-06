use super::*;
use akita_types::{dispatch_for_field, terminal_witness_segment_layout, Commitment, RingView};

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
pub(super) fn verify_root<F, E, T>(
    proof: &AkitaBatchedRootProof<F, E>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    claims: &OpeningClaims<'_, E, &Commitment<F>>,
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
    let m_row_layout = proof.fold_m_row_layout().ok_or(AkitaError::InvalidProof)?;
    let extension_opening_reduction = proof.fold_extension_opening_reduction();
    let next_fold_level_params = match proof {
        AkitaBatchedRootProof::Fold(_) => next_fold_level_params.ok_or(AkitaError::InvalidProof)?,
        AkitaBatchedRootProof::Terminal(_) => root_lp,
        AkitaBatchedRootProof::ZeroFold { .. } => return Err(AkitaError::InvalidProof),
    };
    let stage3_sumcheck_proof = proof.fold_stage3_sumcheck_proof(setup_contribution_mode)?;
    // Read the proof commitment as a D-free flat coefficient buffer.
    let commitment = claims
        .single_group_commitment()
        .copied()
        .ok_or(AkitaError::InvalidProof)?;
    let openings = claims.flat_evaluations();
    let opening_batch = claims.layout().map_err(|_| AkitaError::InvalidProof)?;
    let shared_opening_point = claims.point();
    let num_claims = opening_batch.num_total_polynomials();
    if openings.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    let ring_dim = root_lp.role_dims().d_b();
    let commitment_view = RingView::new(commitment.rows().coeffs(), ring_dim)?;
    if commitment_view.num_rings() != root_lp.b_key.row_len() {
        return Err(AkitaError::InvalidProof);
    }

    // Transcript binding, D-free and byte-identical to the prover's absorb:
    // batch shape header, then the flat commitment coefficients under
    // `ring_dim`, then the shared opening point. This replaces the former
    // `OpeningClaims::append_to_transcript`, whose generic commitment
    // path required a typed `RingCommitment: AppendToTranscript`.
    opening_batch.append_batch_shape_to_transcript::<F, T>(transcript)?;
    commitment_view.append_flat_to_transcript::<T>(ABSORB_COMMITMENT, transcript)?;
    for coord in shared_opening_point {
        append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, coord);
    }

    // D-free root replay: typed kernels dispatch inside `verify_fold` and
    // `verify_fold_eor` on per-role dimensions.
    verify_root_inner::<F, E, T>(
        proof,
        setup,
        transcript,
        commitment.rows(),
        &openings,
        &opening_batch,
        shared_opening_point,
        num_claims,
        m_row_layout,
        extension_opening_reduction,
        stage3_sumcheck_proof,
        next_fold_level_params,
        basis,
        root_lp,
        terminal_final_w_len,
    )
}

/// Root-fold replay orchestrator (D-free).
///
/// Reached from [`verify_root`]; per-role typed kernels dispatch inside
/// [`verify_fold`] and [`verify_fold_eor`].
#[allow(clippy::too_many_arguments)]
fn verify_root_inner<F, E, T>(
    proof: &AkitaBatchedRootProof<F, E>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    commitment: &RingVec<F>,
    openings: &[E],
    opening_batch: &OpeningClaimsLayout,
    shared_opening_point: &[E],
    num_claims: usize,
    m_row_layout: MRowLayout,
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<E>>,
    stage3_sumcheck_proof: Option<&SetupSumcheckProof<E>>,
    next_fold_level_params: &LevelParams,
    basis: BasisMode,
    root_lp: &LevelParams,
    terminal_final_w_len: usize,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HalvingField,
    E: FpExtEncoding<F> + ExtField<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let role_dims = root_lp.role_dims();
    let d_a = role_dims.d_a();
    let v_storage = match proof {
        AkitaBatchedRootProof::Fold(fold) => fold.v.clone(),
        AkitaBatchedRootProof::Terminal(_) => RingVec::from_coeffs(Vec::new()),
        AkitaBatchedRootProof::ZeroFold { .. } => return Err(AkitaError::InvalidProof),
    };

    if extension_opening_reduction.is_none() {
        let prepared_point =
            dispatch_for_field!(ProtocolDispatchSlot::Role(RingRole::Inner), F, d_a, |D| {
                prepare_opening_point::<F, E, D>(
                    shared_opening_point,
                    basis,
                    root_lp.m_vars,
                    root_lp.r_vars,
                    d_a.trailing_zeros() as usize,
                    BlockOrder::RowMajor,
                )
            })?;
        for pt in &prepared_point.padded_point {
            append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, pt);
        }
    }
    append_claim_values_to_transcript::<F, E, T>(openings, transcript);
    let row_coefficients = sample_public_row_coefficients::<F, E, T>(opening_batch, transcript)?;
    let root_eor = verify_fold_eor::<F, E, T>(
        extension_opening_reduction,
        shared_opening_point,
        openings,
        &row_coefficients,
        opening_batch,
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
            dispatch_for_field!(ProtocolDispatchSlot::Role(RingRole::Inner), F, d_a, |D| {
                ring_subfield_packed_extension_opening_point::<F, E, D>(rho.len(), rho)
            })?;
        root_trace_block_opening::<E>(&protocol_point, root_lp, d_a.trailing_zeros() as usize)?
    } else {
        root_trace_block_opening::<E>(shared_opening_point, root_lp, d_a.trailing_zeros() as usize)?
    };
    let ordinary_trace_eval_target =
        opening_batch.batched_eval_target(&row_coefficients, openings)?;
    let trace_eval_target = eor_trace_final
        .as_ref()
        .map(|(final_claim, _)| *final_claim)
        .unwrap_or(ordinary_trace_eval_target);
    let trace_claim_scales = eor_trace_final
        .as_ref()
        .map(|(_, factors_by_point)| {
            let shared_factor = *factors_by_point.first().ok_or(AkitaError::InvalidProof)?;
            Ok(vec![shared_factor; opening_batch.num_total_polynomials()])
        })
        .transpose()?;

    let w_len = match proof {
        AkitaBatchedRootProof::Terminal(_) => terminal_final_w_len,
        AkitaBatchedRootProof::Fold(_) => {
            // Chunked levels commit a wider (replicated-ẑ) next witness; size it
            // with the per-level chunk count (`num_chunks = 1` is unchanged).
            akita_types::w_ring_element_count_for_chunks(
                F::modulus_bits(),
                root_lp,
                opening_batch.num_total_polynomials(),
                akita_types::MRowLayout::WithDBlock,
                root_lp.witness_chunk.num_chunks,
            )?
            .checked_mul(d_a)
            .ok_or_else(|| AkitaError::InvalidSetup("next witness length overflow".to_string()))?
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
    let replay_opening_batch = OpeningClaims::from_groups(
        shared_opening_point,
        vec![PolynomialGroupClaims::new(
            PointVariableSelection::prefix(
                opening_batch.max_num_vars(),
                opening_batch.max_num_vars(),
            )?,
            openings.to_vec(),
            commitment,
        )?],
    )?;
    let prepared = PreparedFoldReplay {
        lp: root_lp,
        m_row_layout,
        fold_grind_nonce,
        v: v_storage,
        opening_batch: replay_opening_batch,
        row_coefficients,
        ring_opening_point: prepared_point.ring_opening_point.clone(),
        ring_multiplier_point: prepared_point.ring_multiplier_point.clone(),
        w_len,
        stage1: stage1_proof,
        stage2,
        next_w_commitment,
        next_ring_dim: matches!(proof, AkitaBatchedRootProof::Fold(_))
            .then_some(next_fold_level_params.role_dims().d_b()),
        terminal_replay,
        stage3: stage3_sumcheck_proof.map(|proof| (proof, next_fold_level_params)),
        trace_prepared_point: Some(prepared_point.clone()),
        trace_block_opening: Some(trace_block_opening),
        trace_eval_target,
        trace_eval_scale: E::one(),
        trace_claim_scales,
        trace_basis: basis,
    };
    verify_fold::<F, E, T>(setup, transcript, prepared)
}
