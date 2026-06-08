use super::super::*;

/// Prove the folded root level after root orchestration has built its
/// ring relation and selected the next recursive commitment policy.
///
/// The root caller owns transcript setup for public openings and gamma
/// batching, schedule selection, and the commitment-row view used by the root
/// relation. It also passes the already-validated challenge sampler used for
/// the remaining base-field stage proofs. This function owns the config-free
/// prover mechanics from `w` construction through the stage proofs and next
/// recursive state.
///
/// # Errors
///
/// Returns an error if ring switching, recursive commitment, or either
/// sumcheck prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_root_fold_from_ring_relation<F, C, T, B, const D: usize, CommitW>(
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    commitment_rows: &[CyclotomicRing<F, D>],
    lp: &akita_types::LevelParams,
    expected_w_len: usize,
    next_log_basis: u32,
    #[cfg(feature = "zk")] zk_hiding_commitment: ZkHidingCommitment<F>,
    #[cfg(feature = "zk")] mut zk_hiding: ZkHidingProverState<F>,
    instance: RingRelationInstance<F, D>,
    witness: RingRelationWitness<F, D>,
    y_rings: Vec<CyclotomicRing<F, D>>,
    #[cfg(feature = "zk")] y_rings_masked: Vec<CyclotomicRing<F, D>>,
    row_coefficients: Vec<C>,
    setup_contribution_mode: SetupContributionMode,
    commit_w_for_next: CommitW,
) -> Result<RootLevelRawOutput<F, C, D>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasWide + HalvingField,
    C: ExtField<F>
        + RingSubfieldEncoding<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
    B: ProverComputeBackend<F>,
    CommitW: FnOnce(&RecursiveWitnessFlat) -> Result<NextWitnessCommitment<F>, AkitaError>,
{
    let logical_w = ring_switch_build_w::<F, B, D>(&instance, witness, backend, prepared, lp)?;
    if logical_w.len() != expected_w_len {
        return Err(AkitaError::InvalidSetup(format!(
            "scheduled root next-w length did not match runtime witness: expected={expected_w_len}, actual={}",
            logical_w.len()
        )));
    }
    let next_commitment = {
        let _span = tracing::info_span!("commit_w_level", level = 0usize).entered();
        commit_w_for_next(&logical_w)?
    };
    let NextWitnessCommitment {
        witness: packed_witness,
        commitment: committed_commitment,
        hint: committed_hint,
    } = next_commitment;
    let w_commitment_proof = committed_commitment.clone();

    let rs = {
        let _span = tracing::info_span!("root_ring_switch_finalize").entered();
        ring_switch_finalize_with_gamma::<F, C, T, D>(
            &instance,
            expanded,
            transcript,
            &logical_w,
            &w_commitment_proof,
            lp,
            &row_coefficients,
            MRowLayout::WithDBlock,
        )?
    };

    let relation_claim = relation_claim_from_rows_extension::<F, C, D>(
        &rs.tau1,
        rs.alpha,
        &instance.v,
        commitment_rows,
        &y_rings,
    )?;
    #[cfg(feature = "zk")]
    let relation_claim_public = relation_claim_from_rows_extension::<F, C, D>(
        &rs.tau1,
        rs.alpha,
        &instance.v,
        commitment_rows,
        &y_rings_masked,
    )?;
    let RingSwitchOutput {
        w_evals_compact,
        live_x_cols,
        m_evals_x,
        alpha_evals_y,
        col_bits,
        ring_bits,
        tau0,
        tau1,
        b,
        alpha,
    } = rs;
    let tau0_reordered = reorder_stage1_coords(&tau0, col_bits, ring_bits);
    #[cfg(feature = "zk")]
    let (stage1_round_pads, stage1_child_claim_masks, stage2_round_pads) =
        zk_hiding.take_current_level_pads::<C>(col_bits + ring_bits, b)?;
    let (stage1_proof, stage1_point, s_claim) = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        let stage1_prover = AkitaStage1Prover::new(
            &w_evals_compact,
            &tau0_reordered,
            b,
            live_x_cols,
            col_bits,
            ring_bits,
        )?;
        #[cfg(feature = "zk")]
        {
            stage1_prover.prove(transcript, stage1_round_pads, stage1_child_claim_masks)?
        }
        #[cfg(not(feature = "zk"))]
        {
            let (stage1_proof, stage1_point) = stage1_prover.prove(transcript)?;
            let s_claim = stage1_proof.s_claim;
            (stage1_proof, stage1_point, s_claim)
        }
    };

    transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &stage1_proof.s_claim);
    let batching_coeff: C = sample_ext_challenge::<F, C, T>(transcript, CHALLENGE_SUMCHECK_BATCH);
    #[cfg(feature = "zk")]
    let (stage2_sumcheck_proof_masked, sumcheck_challenges, w_eval, w_eval_masked) = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        let stage2_prover_result = AkitaStage2Prover::new(
            batching_coeff,
            w_evals_compact,
            &stage1_point,
            s_claim,
            b,
            alpha_evals_y,
            m_evals_x,
            live_x_cols,
            col_bits,
            ring_bits,
            relation_claim,
        );
        let mut stage2_prover = stage2_prover_result?;
        let stage2_public_input = batching_coeff * stage1_proof.s_claim + relation_claim_public;
        let (stage2_sumcheck_proof_masked, sumcheck_challenges) = stage2_prover
            .prove_zk::<F, T, _>(
                stage2_public_input,
                transcript,
                |tr| sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND),
                stage2_round_pads,
            )?;

        let w_eval = {
            let _span = tracing::info_span!("multilinear_eval", level = 0usize).entered();
            stage2_prover.final_w_eval()
        };
        let w_eval_masked = w_eval + zk_hiding.take_next_w_eval_mask::<C>()?;
        (
            stage2_sumcheck_proof_masked,
            sumcheck_challenges,
            w_eval,
            w_eval_masked,
        )
    };
    #[cfg(not(feature = "zk"))]
    let (stage2_sumcheck_proof, sumcheck_challenges, w_eval) = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        let mut stage2_prover = AkitaStage2Prover::new(
            batching_coeff,
            w_evals_compact,
            &stage1_point,
            s_claim,
            b,
            alpha_evals_y,
            m_evals_x,
            live_x_cols,
            col_bits,
            ring_bits,
            relation_claim,
        )?;
        let (stage2_sumcheck_proof, sumcheck_challenges, _) = stage2_prover
            .prove::<F, T, _>(transcript, |tr| {
                sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND)
            })?;

        let w_eval = {
            let _span = tracing::info_span!("multilinear_eval", level = 0usize).entered();
            stage2_prover.final_w_eval()
        };
        (stage2_sumcheck_proof, sumcheck_challenges, w_eval)
    };
    let (committed_witness, logical_w) = match packed_witness {
        Some(packed_witness) => (packed_witness, Some(logical_w)),
        None => (logical_w, None),
    };

    #[cfg(not(feature = "zk"))]
    let proof_w_eval = w_eval;
    #[cfg(feature = "zk")]
    let proof_w_eval = w_eval_masked;
    transcript.append_serde(ABSORB_STAGE2_NEXT_W_EVAL, &proof_w_eval);
    let stage3_sumcheck_proof = match setup_contribution_mode {
        SetupContributionMode::Recursive => {
            let setup_len = expanded.shared_matrix().total_ring_elements_at::<D>()?;
            let setup_view = expanded.shared_matrix().ring_view::<D>(1, setup_len)?;
            let output = SetupSumcheckProver::prove::<F, T, _, D>(
                setup_view.as_slice(),
                lp,
                &instance,
                &tau1,
                alpha,
                &sumcheck_challenges[ring_bits..],
                transcript,
                |tr| sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND),
            )?;
            Some(SetupSumcheckProof {
                claim: output.claim,
                sumcheck: output.sumcheck,
            })
        }
        SetupContributionMode::Direct => None,
    };

    Ok(RootLevelRawOutput {
        #[cfg(feature = "zk")]
        zk_hiding_commitment,
        #[cfg(feature = "zk")]
        y_rings: y_rings_masked,
        #[cfg(not(feature = "zk"))]
        y_rings,
        extension_opening_reduction: None,
        v: instance.v,
        stage1: stage1_proof,
        #[cfg(not(feature = "zk"))]
        stage2_sumcheck_proof,
        #[cfg(feature = "zk")]
        stage2_sumcheck_proof_masked,
        stage3_sumcheck_proof,
        w_commitment_proof,
        w_eval: proof_w_eval,
        next_state: RecursiveProverState {
            w: committed_witness,
            logical_w,
            commitment: committed_commitment,
            hint: committed_hint,
            log_basis: next_log_basis,
            sumcheck_challenges,
            opening: w_eval,
            #[cfg(feature = "zk")]
            zk_hiding,
        },
    })
}

/// Terminal-root analogue of [`prove_root_fold_from_ring_relation`] used when the
/// schedule has exactly one fold level (the root is itself the terminal).
///
/// Produces a [`TerminalLevelProof`] with cleartext `final_witness` instead
/// of a `RootLevelRawOutput`. There is no recursive suffix and no
/// `next_state` to thread.
///
/// # Errors
///
/// Returns an error if witness reconstruction does not match the schedule's
/// expected length, ring-switch replay fails, or the stage-2 sumcheck prover
/// fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_terminal_root_fold_from_ring_relation<F, C, T, B, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    commitment_rows: &[CyclotomicRing<F, D>],
    lp: &akita_types::LevelParams,
    expected_w_len: usize,
    final_log_basis: u32,
    instance: RingRelationInstance<F, D>,
    witness: RingRelationWitness<F, D>,
    y_rings: Vec<CyclotomicRing<F, D>>,
    #[cfg(feature = "zk")] y_rings_masked: Vec<CyclotomicRing<F, D>>,
    row_coefficients: Vec<C>,
    #[cfg(feature = "zk")] zk_hiding: &mut ZkHidingProverState<F>,
) -> Result<TerminalLevelProof<F, C>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasWide + HalvingField,
    C: ExtField<F>
        + RingSubfieldEncoding<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
    B: ProverComputeBackend<F>,
{
    let terminal_layout = terminal_witness_segment_layout(
        lp,
        instance.claim_to_point().len(),
        instance.num_public_rows(),
        F::modulus_bits(),
    )?;
    let logical_w = ring_switch_build_w::<F, B, D>(&instance, witness, backend, prepared, lp)?;
    if logical_w.len() != expected_w_len {
        return Err(AkitaError::InvalidSetup(format!(
            "scheduled root next-w length did not match runtime witness: expected={expected_w_len}, actual={}",
            logical_w.len()
        )));
    }
    let final_witness = CleartextWitnessProof::PackedDigits(
        PackedDigits::from_i8_digits_with_min_bits(logical_w.as_i8_digits(), final_log_basis),
    );

    let rs = ring_switch_finalize_terminal_with_gamma::<F, C, T, D>(
        &instance,
        expanded,
        transcript,
        &logical_w,
        &final_witness,
        terminal_layout,
        lp,
        &row_coefficients,
    )?;

    // Terminal layout: the D-block is omitted, so the relation claim sums no
    // `v` rows. `instance.v` is constructed as an empty vector under
    // `MRowLayout::WithoutDBlock`; pass `&[]` here for symmetry with the verifier.
    let relation_claim = relation_claim_from_rows_extension::<F, C, D>(
        &rs.tau1,
        rs.alpha,
        &[],
        commitment_rows,
        &y_rings,
    )?;
    #[cfg(feature = "zk")]
    let relation_claim_public = relation_claim_from_rows_extension::<F, C, D>(
        &rs.tau1,
        rs.alpha,
        &[],
        commitment_rows,
        &y_rings_masked,
    )?;

    let RingSwitchOutput {
        w_evals_compact,
        live_x_cols,
        m_evals_x,
        alpha_evals_y,
        col_bits,
        ring_bits,
        tau0: _,
        tau1: _,
        b,
        alpha: _,
    } = rs;

    let stage1_point = vec![C::zero(); col_bits + ring_bits];
    #[cfg(feature = "zk")]
    let stage2_round_pads = zk_hiding.take_compressed_rounds::<C>(col_bits + ring_bits, 3)?;
    #[cfg(feature = "zk")]
    let stage2_sumcheck_proof_masked = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck_terminal_root").entered();
        let mut stage2_prover = AkitaStage2Prover::new(
            C::zero(),
            w_evals_compact,
            &stage1_point,
            C::zero(),
            b,
            alpha_evals_y,
            m_evals_x,
            live_x_cols,
            col_bits,
            ring_bits,
            relation_claim,
        )?;
        let (stage2_sumcheck_proof_masked, _sumcheck_challenges) = stage2_prover
            .prove_zk::<F, T, _>(
                relation_claim_public,
                transcript,
                |tr| sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND),
                stage2_round_pads,
            )?;
        stage2_sumcheck_proof_masked
    };
    #[cfg(not(feature = "zk"))]
    let stage2_sumcheck = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck_terminal_root").entered();
        let mut stage2_prover = AkitaStage2Prover::new(
            C::zero(),
            w_evals_compact,
            &stage1_point,
            C::zero(),
            b,
            alpha_evals_y,
            m_evals_x,
            live_x_cols,
            col_bits,
            ring_bits,
            relation_claim,
        )?;
        let (stage2_sumcheck, _sumcheck_challenges, _stage2_final_claim) = stage2_prover
            .prove::<F, T, _>(transcript, |tr| {
                sample_ext_challenge::<F, C, T>(tr, CHALLENGE_SUMCHECK_ROUND)
            })?;
        stage2_sumcheck
    };

    Ok(
        TerminalLevelProof::new_with_extension_opening_reduction::<D>(
            #[cfg(not(feature = "zk"))]
            y_rings,
            #[cfg(feature = "zk")]
            y_rings_masked,
            None,
            #[cfg(not(feature = "zk"))]
            stage2_sumcheck,
            #[cfg(feature = "zk")]
            stage2_sumcheck_proof_masked,
            final_witness,
        ),
    )
}
