use super::*;

mod carried;

struct PreparedRecursiveFold<F: FieldCore, L: FieldCore, const D: usize> {
    /// Concatenated commitment rows across all carried claims (claim order),
    /// fed into the relation claim. For the singleton case this is just the
    /// recursive witness commitment.
    commitment_rows: Vec<CyclotomicRing<F, D>>,
    instance: RingRelationInstance<F, D>,
    witness: RingRelationWitness<F, D>,
    reduction: Option<RecursiveExtensionOpeningReduction<L>>,
    y_rings: Vec<CyclotomicRing<F, D>>,
    /// Extra committed sources carried from the consumed state, re-evaluated and
    /// propagated into the produced level's proof and next state.
    extra_carried_sources: Vec<RecursiveCarriedSource<F>>,
    #[cfg(feature = "zk")]
    y_rings_masked: Vec<CyclotomicRing<F, D>>,
    #[cfg(feature = "zk")]
    zk_hiding: ZkHidingProverState<F>,
}

#[cfg(not(feature = "zk"))]
type TerminalFoldResult<F, L> = TerminalLevelProof<F, L>;
#[cfg(feature = "zk")]
type TerminalFoldResult<F, L> = (TerminalLevelProof<F, L>, ZkHidingProverState<F>);

/// Drive the recursive fold suffix (after the root) under config `Cfg`.
///
/// The selected planner `schedule` is authoritative: it determines the fold
/// count, per-level `LevelParams`, successor params, and the terminal direct
/// witness basis. Earlier suffix levels run intermediate folds; the last
/// suffix level runs the terminal fold which ships the cleartext
/// `final_witness`.
///
/// # Errors
///
/// Returns an error if level proving fails, or an invalid-setup error when the
/// schedule's recursive suffix is empty (root-terminal proofs do not run this
/// helper).
#[allow(clippy::too_many_arguments)]
pub fn prove_recursive_suffix<Cfg, T, B, const D: usize>(
    expanded: &Arc<AkitaExpandedSetup<Cfg::Field>>,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field, D>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    num_vars: usize,
    transcript: &mut T,
    initial_state: RecursiveProverState<Cfg::Field, Cfg::ChallengeField>,
    schedule: &Schedule,
    setup_contribution_mode: SetupContributionMode,
) -> Result<RecursiveSuffixOutcome<Cfg::Field, Cfg::ChallengeField>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    Cfg::ChallengeField: RingSubfieldEncoding<Cfg::Field>
        + FrobeniusExtField<Cfg::Field>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize
        + MulBaseUnreduced<Cfg::Field>,
    T: Transcript<Cfg::Field>,
    B: ProverComputeBackend<Cfg::Field>,
{
    let planned_num_levels = schedule_num_fold_levels(schedule);
    if planned_num_levels < 2 {
        return Err(AkitaError::InvalidSetup(
            "prove_recursive_suffix expects a non-empty recursive suffix".to_string(),
        ));
    }
    let terminal_level = planned_num_levels - 1;

    let mut intermediate_levels = Vec::new();
    let mut current_state = initial_state;
    let mut level = 1usize;

    while level < terminal_level {
        let inputs = AkitaScheduleInputs {
            num_vars,
            level,
            current_w_len: current_state.w.len(),
        };
        let (level_params, next_params) =
            scheduled_fold_execution(schedule, level, inputs, current_state.log_basis)?;
        let level_d = level_params.ring_dimension;
        let out = if level_d == D {
            let prepared_fold = prepare_fold_data::<Cfg::Field, Cfg::ChallengeField, T, B, D>(
                backend,
                prepared,
                transcript,
                current_state,
                level,
                &level_params,
                MRowLayout::WithDBlock,
            )?;
            prove_fold::<Cfg::Field, Cfg::ChallengeField, T, B, Cfg, D>(
                expanded,
                prefix_slots,
                backend,
                prepared,
                transcript,
                level,
                &level_params,
                &next_params,
                prepared_fold,
                setup_contribution_mode,
            )
        } else {
            dispatch_ring_dim_result!(level_d, |D_LEVEL| {
                let level_prepared = backend.prepare_expanded::<D_LEVEL>(expanded.clone())?;
                let level_prefix_slots = SetupPrefixProverRegistry::new();
                let prepared_fold =
                    prepare_fold_data::<Cfg::Field, Cfg::ChallengeField, T, B, { D_LEVEL }>(
                        backend,
                        &level_prepared,
                        transcript,
                        current_state,
                        level,
                        &level_params,
                        MRowLayout::WithDBlock,
                    )?;
                prove_fold::<Cfg::Field, Cfg::ChallengeField, T, B, Cfg, { D_LEVEL }>(
                    expanded,
                    &level_prefix_slots,
                    backend,
                    &level_prepared,
                    transcript,
                    level,
                    &level_params,
                    &next_params,
                    prepared_fold,
                    setup_contribution_mode,
                )
            })
        }?;
        intermediate_levels.push(out.level_proof);
        current_state = out.next_state;
        level += 1;
    }

    debug_assert_eq!(level, terminal_level);
    let inputs = AkitaScheduleInputs {
        num_vars,
        level,
        current_w_len: current_state.w.len(),
    };
    let (level_params, next_params) =
        scheduled_fold_execution(schedule, level, inputs, current_state.log_basis)?;
    let level_d = level_params.ring_dimension;
    let terminal_result = if level_d == D {
        let prepared_fold = prepare_fold_data::<Cfg::Field, Cfg::ChallengeField, T, B, D>(
            backend,
            prepared,
            transcript,
            current_state,
            level,
            &level_params,
            MRowLayout::WithoutDBlock,
        )?;
        prove_terminal_fold::<Cfg::Field, Cfg::ChallengeField, T, B, D>(
            expanded.as_ref(),
            backend,
            prepared,
            transcript,
            level,
            &level_params,
            next_params.log_basis,
            prepared_fold,
        )
    } else {
        dispatch_ring_dim_result!(level_d, |D_LEVEL| {
            let level_prepared = backend.prepare_expanded::<D_LEVEL>(expanded.clone())?;
            let prepared_fold =
                prepare_fold_data::<Cfg::Field, Cfg::ChallengeField, T, B, { D_LEVEL }>(
                    backend,
                    &level_prepared,
                    transcript,
                    current_state,
                    level,
                    &level_params,
                    MRowLayout::WithoutDBlock,
                )?;
            prove_terminal_fold::<Cfg::Field, Cfg::ChallengeField, T, B, { D_LEVEL }>(
                expanded.as_ref(),
                backend,
                &level_prepared,
                transcript,
                level,
                &level_params,
                next_params.log_basis,
                prepared_fold,
            )
        })
    };
    #[cfg(not(feature = "zk"))]
    let terminal = terminal_result?;
    #[cfg(feature = "zk")]
    let (terminal, zk_hiding) = terminal_result?;

    Ok(RecursiveSuffixOutcome {
        intermediate_levels,
        terminal,
        #[cfg(feature = "zk")]
        zk_hiding,
        num_levels: planned_num_levels,
    })
}

/// Prove one recursive fold level after the caller has built its ring-relation
/// equation and selected the commitment policy for the next `w`.
///
/// This function owns prover mechanics: build `w`, commit it, finish ring
/// switching, run stage-1/stage-2 sumchecks, and produce the next recursive
/// state.
///
/// # Errors
///
/// Returns an error if ring switching, recursive commitment, or either
/// sumcheck prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn prove_fold<F, L, T, B, Cfg, const D: usize>(
    expanded: &Arc<AkitaExpandedSetup<F>>,
    prefix_slots: &SetupPrefixProverRegistry<F, D>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    level: usize,
    lp: &LevelParams,
    next_level_params: &LevelParams,
    prepared_fold: PreparedRecursiveFold<F, L, D>,
    setup_contribution_mode: SetupContributionMode,
) -> Result<ProveLevelOutput<F, L>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    L: ExtField<F>
        + RingSubfieldEncoding<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
    B: ProverComputeBackend<F>,
    Cfg: CommitmentConfig<Field = F, ChallengeField = L>,
{
    let PreparedRecursiveFold {
        commitment_rows,
        instance,
        witness,
        reduction,
        y_rings,
        extra_carried_sources: _extra_carried_sources,
        #[cfg(feature = "zk")]
        y_rings_masked,
        #[cfg(feature = "zk")]
        mut zk_hiding,
    } = prepared_fold;
    let commitment_u = commitment_rows.as_slice();
    let extension_opening_reduction = reduction.map(|reduction| reduction.proof);
    let logical_w = ring_switch_build_w::<F, B, D>(&instance, witness, backend, prepared, lp)?;
    let next_commitment = {
        let _span = tracing::info_span!("commit_w_level", level).entered();
        crate::commit_next_w::<Cfg, B, D>(
            next_level_params,
            expanded,
            backend,
            prepared,
            &logical_w,
        )?
    };
    let NextWitnessCommitment {
        witness: packed_witness,
        commitment: committed_commitment,
        hint: committed_hint,
    } = next_commitment;
    let w_commitment_proof = committed_commitment.clone();
    let rs = ring_switch_finalize::<F, L, T, D>(
        &instance,
        expanded.as_ref(),
        transcript,
        &logical_w,
        &w_commitment_proof,
        lp,
        MRowLayout::WithDBlock,
    )?;

    let relation_claim = relation_claim_from_rows_extension::<F, L, D>(
        &rs.tau1,
        rs.alpha,
        &instance.v,
        commitment_u,
        &y_rings,
    )?;
    #[cfg(feature = "zk")]
    let relation_claim_public = relation_claim_from_rows_extension::<F, L, D>(
        &rs.tau1,
        rs.alpha,
        &instance.v,
        commitment_u,
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
        zk_hiding.take_current_level_pads::<L>(col_bits + ring_bits, b)?;
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
    let batching_coeff: L = sample_ext_challenge::<F, L, T>(transcript, CHALLENGE_SUMCHECK_BATCH);
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
                |tr| sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND),
                stage2_round_pads,
            )?;

        let w_eval = {
            let _span = tracing::info_span!("multilinear_eval", level).entered();
            stage2_prover.final_w_eval()
        };
        let w_eval_masked = w_eval + zk_hiding.take_next_w_eval_mask::<L>()?;
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
                sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND)
            })?;

        let w_eval = {
            let _span = tracing::info_span!("multilinear_eval", level).entered();
            stage2_prover.final_w_eval()
        };
        (stage2_sumcheck_proof, sumcheck_challenges, w_eval)
    };
    #[cfg(not(feature = "zk"))]
    let proof_w_eval = w_eval;
    #[cfg(feature = "zk")]
    let proof_w_eval = w_eval_masked;
    transcript.append_serde(ABSORB_STAGE2_NEXT_W_EVAL, &proof_w_eval);
    let stage3_sumcheck_proof = match setup_contribution_mode {
        SetupContributionMode::Recursive => {
            let output = SetupSumcheckProver::prove::<F, T, _, D>(
                expanded.as_ref(),
                prefix_slots,
                lp,
                next_level_params,
                &instance,
                &tau1,
                alpha,
                &sumcheck_challenges[ring_bits..],
                transcript,
                |tr| sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND),
            )?;
            Some(SetupSumcheckProof {
                claim: output.claim,
                sumcheck: output.sumcheck,
            })
        }
        SetupContributionMode::Direct => None,
    };

    #[cfg(not(feature = "zk"))]
    let proof_y_rings = y_rings;
    #[cfg(feature = "zk")]
    let proof_y_rings = y_rings_masked;
    let mut level_proof = AkitaLevelProof::new_two_stage_many_with_extension_opening_reduction::<D>(
        proof_y_rings,
        extension_opening_reduction,
        instance.v,
        stage1_proof,
        #[cfg(not(feature = "zk"))]
        stage2_sumcheck_proof,
        #[cfg(feature = "zk")]
        stage2_sumcheck_proof_masked,
        w_commitment_proof,
        proof_w_eval,
    );
    level_proof.stage3_sumcheck_proof = stage3_sumcheck_proof;

    let (committed_witness, logical_w) = match packed_witness {
        Some(packed_witness) => (packed_witness, Some(logical_w)),
        None => (logical_w, None),
    };

    let next_w_len = logical_w.as_ref().unwrap_or(&committed_witness).len();
    let mut out = ProveLevelOutput {
        level_proof,
        next_state: RecursiveProverState {
            w: committed_witness,
            logical_w,
            commitment: committed_commitment,
            hint: committed_hint,
            log_basis: next_level_params.log_basis,
            sumcheck_challenges: sumcheck_challenges.clone(),
            opening: w_eval,
            carried_openings: vec![RecursiveCarriedOpening::recursive_witness(
                sumcheck_challenges,
                w_eval,
                next_w_len,
            )],
            extra_carried_sources: Vec::new(),
            #[cfg(feature = "zk")]
            zk_hiding,
        },
    };
    if !_extra_carried_sources.is_empty() {
        carried::propagate_extra_carried_sources::<F, L, D>(&mut out, &_extra_carried_sources, lp)?;
    }
    Ok(out)
}

/// Prove the terminal recursive fold level after the caller has built its
/// ring relation.
///
/// At the terminal level the next witness is shipped in cleartext as
/// [`PackedDigits`], so this function:
///
/// * builds `logical_w` via ring switching,
/// * packs it into the terminal [`CleartextWitnessProof`] using
///   `final_log_basis` as the planner-mandated minimum bits per element,
/// * absorbs logical `e_hat` before fold challenge sampling when the
///   ring relation is built, then absorbs the remaining final-witness
///   bytes before sampling any ring-switch challenges,
/// * skips the stage-1 sumcheck entirely (packed-digit range is structurally
///   enforced by the packing), and
/// * runs stage-2 in relation-only mode with `batching_coeff = 0`,
///   `s_claim = 0`, and dummy `stage1_point` zeros — these zero the virtual
///   sumcheck contribution leaving only the relation oracle.
///
/// # Errors
///
/// Returns an error if ring switching or the stage-2 sumcheck prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn prove_terminal_fold<F, L, T, B, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    _level: usize,
    lp: &LevelParams,
    final_log_basis: u32,
    prepared_fold: PreparedRecursiveFold<F, L, D>,
) -> Result<TerminalFoldResult<F, L>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    L: ExtField<F>
        + RingSubfieldEncoding<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
    B: ProverComputeBackend<F>,
{
    let PreparedRecursiveFold {
        commitment_rows,
        instance,
        witness,
        reduction,
        y_rings,
        extra_carried_sources: _extra_carried_sources,
        #[cfg(feature = "zk")]
        y_rings_masked,
        #[cfg(feature = "zk")]
        mut zk_hiding,
    } = prepared_fold;
    let commitment_u = commitment_rows.as_slice();
    let extension_opening_reduction = reduction.map(|reduction| reduction.proof);
    let terminal_layout = terminal_witness_segment_layout(
        lp,
        instance.claim_to_point().len(),
        instance.num_public_rows(),
        F::modulus_bits(),
    )?;
    let logical_w = ring_switch_build_w::<F, B, D>(&instance, witness, backend, prepared, lp)?;
    let final_witness = CleartextWitnessProof::PackedDigits(
        PackedDigits::from_i8_digits_with_min_bits(logical_w.as_i8_digits(), final_log_basis),
    );
    let rs = ring_switch_finalize_terminal::<F, L, T, D>(
        &instance,
        expanded,
        transcript,
        &logical_w,
        &final_witness,
        terminal_layout,
        lp,
    )?;

    // Terminal layout drops the D-block: the relation claim no longer sums
    // any `v` rows, so pass an empty slice for the v parameter.
    let relation_claim = relation_claim_from_rows_extension::<F, L, D>(
        &rs.tau1,
        rs.alpha,
        &[],
        commitment_u,
        &y_rings,
    )?;
    #[cfg(feature = "zk")]
    let relation_claim_public = relation_claim_from_rows_extension::<F, L, D>(
        &rs.tau1,
        rs.alpha,
        &[],
        commitment_u,
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

    // Relation-only stage-2: batching_coeff = 0 zeros the virtual-claim
    // contribution to every round polynomial regardless of `stage1_point`, so
    // dummy zeros for `stage1_point` and `s_claim` are safe.
    let stage1_point = vec![L::zero(); col_bits + ring_bits];
    #[cfg(feature = "zk")]
    let stage2_round_pads = zk_hiding.take_compressed_rounds::<L>(col_bits + ring_bits, 3)?;
    #[cfg(feature = "zk")]
    let (stage2_sumcheck_proof_masked, _sumcheck_challenges) = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck_terminal").entered();
        let mut stage2_prover = AkitaStage2Prover::new(
            L::zero(),
            w_evals_compact,
            &stage1_point,
            L::zero(),
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
                |tr| sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND),
                stage2_round_pads,
            )?;
        (stage2_sumcheck_proof_masked, _sumcheck_challenges)
    };
    #[cfg(not(feature = "zk"))]
    let (stage2_sumcheck, _sumcheck_challenges) = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck_terminal").entered();
        let mut stage2_prover = AkitaStage2Prover::new(
            L::zero(),
            w_evals_compact,
            &stage1_point,
            L::zero(),
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
                sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND)
            })?;
        (stage2_sumcheck, _sumcheck_challenges)
    };
    let proof = TerminalLevelProof::new_with_extension_opening_reduction::<D>(
        #[cfg(not(feature = "zk"))]
        y_rings,
        #[cfg(feature = "zk")]
        y_rings_masked,
        extension_opening_reduction,
        #[cfg(not(feature = "zk"))]
        stage2_sumcheck,
        #[cfg(feature = "zk")]
        stage2_sumcheck_proof_masked,
        final_witness,
    );
    #[cfg(not(feature = "zk"))]
    {
        Ok(proof)
    }
    #[cfg(feature = "zk")]
    {
        Ok((proof, zk_hiding))
    }
}

pub(in crate::protocol::flow) struct RecursiveExtensionOpeningReduction<L: FieldCore> {
    pub(in crate::protocol::flow) proof: ExtensionOpeningReductionProof<L>,
    pub(in crate::protocol::flow) rho: Vec<L>,
    /// EOR final sumcheck claim and transparent-factor evaluation. Retained so
    /// the prepare step can fail-fast cross-check the folded y-ring opening
    /// against the reduction output; the verifier enforces the same relation.
    pub(in crate::protocol::flow) final_claim: L,
    pub(in crate::protocol::flow) final_factor: L,
}

pub(in crate::protocol::flow) fn recursive_witness_base_evals<F>(
    logical_w: &RecursiveWitnessFlat,
) -> Vec<F>
where
    F: FieldCore + FromPrimitiveInt,
{
    // Pure order-preserving map; the indexed parallel collect yields the same
    // ordering as the serial map, so the base-field witness table is identical.
    cfg_iter!(logical_w.as_i8_digits())
        .copied()
        .map(F::from_i8)
        .collect()
}

pub(in crate::protocol::flow) fn prove_extension_opening_reduction<F, L, T>(
    logical_w: &RecursiveWitnessFlat,
    opening_point: &[L],
    expected_opening: L,
    transcript: &mut T,
    #[cfg(feature = "zk")] zk_hiding: &mut ZkHidingProverState<F>,
) -> Result<RecursiveExtensionOpeningReduction<L>, AkitaError>
where
    F: FieldCore + CanonicalField,
    L: ExtField<F> + HasUnreducedOps + HasOptimizedFold + AkitaSerialize + MulBaseUnreduced<F>,
    T: Transcript<F>,
{
    let num_vars = opening_point.len();
    let padded_len = 1usize.checked_shl(num_vars as u32).ok_or_else(|| {
        AkitaError::InvalidInput("recursive opening point is too large".to_string())
    })?;
    let (split_bits, _width) = tensor_opening_split::<F, L>()?;
    if split_bits > num_vars {
        return Err(AkitaError::InvalidPointDimension {
            expected: split_bits,
            actual: opening_point.len(),
        });
    }
    if logical_w.len() > padded_len {
        return Err(AkitaError::InvalidSize {
            expected: padded_len,
            actual: logical_w.len(),
        });
    }
    let _eor_prep_span = tracing::info_span!("recursive_eor_prepare", num_vars).entered();
    let base_evals = {
        let _s = tracing::info_span!("eor_base_evals").entered();
        let mut base_evals = recursive_witness_base_evals::<F>(logical_w);
        base_evals.resize(padded_len, F::zero());
        base_evals
    };
    let tensor = {
        let _s = tracing::info_span!("eor_tensor_partials").entered();
        tensor_partials_from_base_evals::<F, L>(num_vars, &base_evals, opening_point)?
    };
    check_tensor_extension_opening_claim::<F, L>(
        opening_point,
        expected_opening,
        &tensor.column_partials,
    )?;
    #[cfg(feature = "zk")]
    let (partial_masks, sumcheck_pads) = zk_hiding.take_extension_opening_reduction_pads::<L>(
        tensor.column_partials.len(),
        num_vars - split_bits,
    )?;
    #[cfg(feature = "zk")]
    let proof_partials = tensor
        .column_partials
        .iter()
        .copied()
        .zip(partial_masks)
        .map(|(partial, mask)| partial + mask)
        .collect::<Vec<_>>();
    #[cfg(not(feature = "zk"))]
    let proof_partials = tensor.column_partials.clone();
    for partial in &proof_partials {
        append_ext_field::<F, L, T>(transcript, ABSORB_EVALUATION_CLAIMS, partial);
    }

    let eta = (0..split_bits)
        .map(|_| sample_ext_challenge::<F, L, T>(transcript, CHALLENGE_SUMCHECK_BATCH))
        .collect::<Vec<_>>();
    let proof_row_partials = tensor_row_partials_from_columns::<F, L>(&proof_partials)?;
    let input_claim = tensor_reduction_claim_from_rows::<F, L>(&proof_row_partials, &eta)?;
    let true_input_claim = tensor_reduction_claim_from_rows::<F, L>(&tensor.row_partials, &eta)?;
    #[cfg(not(feature = "zk"))]
    debug_assert_eq!(input_claim, true_input_claim);
    let tail_point = &opening_point[split_bits..];
    let packed_witness = {
        let _s = tracing::info_span!("eor_packed_witness").entered();
        tensor_packed_witness_evals::<F, L>(num_vars, &base_evals)?
    };
    let factor_evals = {
        let _s = tracing::info_span!("eor_factor_evals").entered();
        tensor_equality_factor_evals::<F, L>(tail_point, &eta)?
    };
    let prover = ExtensionOpeningReductionProver::from_dense_tables(packed_witness, factor_evals)?;
    if prover.input_claim() != true_input_claim {
        return Err(AkitaError::InvalidInput(
            "extension-opening reduction input claim mismatch".to_string(),
        ));
    }
    let mut prover = prover;
    drop(_eor_prep_span);
    let _eor_sumcheck_span = tracing::info_span!(
        "extension_opening_reduction_sumcheck",
        path = "recursive",
        num_rounds = prover.num_rounds()
    )
    .entered();
    #[cfg(not(feature = "zk"))]
    let (sumcheck, rho, final_claim) = prover.prove::<F, T, _>(transcript, |tr| {
        sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND)
    })?;
    #[cfg(feature = "zk")]
    let (sumcheck_proof_masked, rho) = prover.prove_zk::<F, T, _>(
        input_claim,
        transcript,
        |tr| sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND),
        sumcheck_pads,
    )?;
    let (final_witness, final_factor_from_table) =
        prover.final_witness_and_factor_evals().ok_or_else(|| {
            AkitaError::InvalidInput(
                "extension-opening reduction has not reached a final point".to_string(),
            )
        })?;
    let final_factor = tensor_equality_factor_eval_at_point::<F, L>(tail_point, &eta, &rho)?;
    if final_factor != final_factor_from_table {
        return Err(AkitaError::InvalidInput(
            "extension-opening reduction transparent factor mismatch".to_string(),
        ));
    }
    #[cfg(feature = "zk")]
    let final_claim = final_witness * final_factor;
    check_extension_opening_reduction_output(final_claim, final_witness, final_factor)?;
    Ok(RecursiveExtensionOpeningReduction {
        proof: ExtensionOpeningReductionProof {
            partials: proof_partials,
            #[cfg(not(feature = "zk"))]
            sumcheck,
            #[cfg(feature = "zk")]
            sumcheck_proof_masked,
        },
        rho,
        final_claim,
        final_factor,
    })
}

/// Prove one recursive fold level using already-selected current and next
/// level parameters.
///
/// The caller owns schedule/config selection and passes the next-level
/// commitment params. This function owns recursive opening-point reduction,
/// witness folding, public recursive transcript absorbs, recursive
/// ring-relation construction, and the folded-level prover mechanics.
///
/// # Errors
///
/// Returns an error if the recursive opening point has the wrong dimension,
/// witness folding or ring-relation construction fails, or the folded
/// prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn prepare_fold_data<F, L, T, B, const D: usize>(
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    current_state: RecursiveProverState<F, L>,
    level: usize,
    level_params: &LevelParams,
    m_row_layout: MRowLayout,
) -> Result<PreparedRecursiveFold<F, L, D>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    L: RingSubfieldEncoding<F>
        + FrobeniusExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize
        + MulBaseUnreduced<F>,
    T: Transcript<F>,
    B: ProverComputeBackend<F>,
{
    {
        let x: u8 = 0;
        tracing::trace!(
            stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
            level,
            "prepare_fold_data"
        );
    }

    let RecursiveProverState {
        w,
        logical_w,
        commitment,
        hint,
        sumcheck_challenges: _,
        opening: _,
        log_basis: _,
        carried_openings,
        extra_carried_sources,
        #[cfg(feature = "zk")]
        zk_hiding,
    } = current_state;
    let witness_view = w.view::<F, D>()?;
    let logical_w = logical_w.as_ref().unwrap_or(&w);
    let committed_w_len = logical_w.len();
    let typed_hint = hint.to_typed::<D>()?;
    #[cfg(feature = "zk")]
    let mut zk_hiding = zk_hiding;

    let alpha = level_params.ring_dimension.trailing_zeros() as usize;

    // Bind the carried-opening batch (sources + per-claim metadata) into the
    // transcript before any fold challenge is sampled. Source 0 is the folded
    // recursive witness; extras are proof-visible carried sources.
    let mut carried_sources = Vec::with_capacity(extra_carried_sources.len() + 1);
    carried_sources.push(CarriedOpeningSource {
        commitment: &commitment,
    });
    carried_sources.extend(
        extra_carried_sources
            .iter()
            .map(|source| CarriedOpeningSource {
                commitment: &source.commitment,
            }),
    );
    let carried_claims = carried_openings
        .iter()
        .map(|claim| CarriedOpeningClaim {
            source_idx: claim.source_idx,
            point: &claim.opening_point,
            value: claim.transcript_opening(),
            basis: claim.basis,
            natural_len: claim.natural_len,
            padded_len: claim.padded_len,
            kind: claim.kind,
        })
        .collect::<Vec<_>>();
    append_carried_opening_batch_to_transcript(&carried_sources, &carried_claims, transcript)?;
    let carried_incidence = carried_opening_incidence_summary(&carried_sources, &carried_claims)?;

    let mut source_views = Vec::with_capacity(extra_carried_sources.len() + 1);
    source_views.push(witness_view);
    let mut source_logical_lens = Vec::with_capacity(extra_carried_sources.len() + 1);
    source_logical_lens.push(committed_w_len);
    for source in &extra_carried_sources {
        source_views.push(source.w.view::<F, D>()?);
        source_logical_lens.push(source.logical_w.as_ref().unwrap_or(&source.w).len());
    }
    if carried_openings.is_empty()
        || carried_openings.iter().any(|claim| {
            (matches!(claim.kind, CarriedOpeningKind::RecursiveWitness)
                && claim.natural_len != committed_w_len)
                || claim.natural_len > claim.padded_len
                || source_logical_lens
                    .get(claim.source_idx)
                    .is_none_or(|&source_len| claim.natural_len > source_len)
        })
        || carried_incidence.num_claims() != carried_openings.len()
    {
        return Err(AkitaError::InvalidInput(
            "recursive carried openings must share the current witness domain".to_string(),
        ));
    }
    if <L as ExtField<F>>::EXT_DEGREE != 1 && carried_openings.len() != 1 {
        return Err(AkitaError::InvalidInput(
            "batched recursive extension-opening reduction is not implemented".to_string(),
        ));
    }

    // Claim 0 uses the extension-opening reduction only for proper extension
    // fields; extra claims always open directly at their carried point.
    let reduction = if <L as ExtField<F>>::EXT_DEGREE == 1 {
        None
    } else {
        let claim = &carried_openings[0];
        Some(prove_extension_opening_reduction::<F, L, T>(
            logical_w,
            &claim.opening_point,
            claim.opening,
            transcript,
            #[cfg(feature = "zk")]
            &mut zk_hiding,
        )?)
    };
    let prepared_points = {
        let _span = tracing::info_span!("ring_opening_point", level).entered();
        let mut prepared = Vec::with_capacity(carried_openings.len());
        for (claim_idx, claim) in carried_openings.iter().enumerate() {
            let protocol_point = match (&reduction, claim_idx) {
                (Some(reduction), 0) => ring_subfield_packed_extension_opening_point::<F, L, D>(
                    reduction.rho.len(),
                    &reduction.rho,
                )?,
                (Some(_), _) => {
                    return Err(AkitaError::InvalidInput(
                        "batched recursive extension-opening reduction is not implemented"
                            .to_string(),
                    ))
                }
                (None, _) => claim.opening_point.clone(),
            };
            prepared.push(prepare_recursive_opening_point_ext::<F, L, D>(
                &protocol_point,
                claim.basis,
                level_params,
                alpha,
                BlockOrder::ColumnMajor,
            )?);
        }
        prepared
    };

    let (y_rings, e_folded_by_claim) = {
        let _span = tracing::info_span!(
            "evaluate_and_fold",
            level,
            num_points = prepared_points.len()
        )
        .entered();
        let mut y_rings = Vec::with_capacity(prepared_points.len());
        let mut folded = Vec::with_capacity(prepared_points.len());
        for (claim, prepared_point) in carried_openings.iter().zip(prepared_points.iter()) {
            let source_view = source_views.get(claim.source_idx).ok_or_else(|| {
                AkitaError::InvalidInput("carried source index out of range".to_string())
            })?;
            let (y_ring, e_folded) = evaluate_witness_at_multiplier_point(
                source_view,
                &prepared_point.ring_multiplier_point,
                level_params.block_len,
                level_params.num_blocks,
            )?;
            y_rings.push(y_ring);
            folded.push(e_folded);
        }
        (y_rings, folded)
    };
    #[cfg(feature = "zk")]
    let y_rings_masked = y_rings
        .iter()
        .map(|y_ring| {
            let (_, y_garbage) = zk_hiding.take_ring::<D>()?;
            Ok(*y_ring + y_garbage)
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    for prepared_point in &prepared_points {
        for pt in &prepared_point.padded_point {
            append_ext_field::<F, L, T>(transcript, ABSORB_EVALUATION_CLAIMS, pt);
        }
    }
    #[cfg(feature = "zk")]
    for y_ring in &y_rings_masked {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }
    #[cfg(not(feature = "zk"))]
    for y_ring in &y_rings {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }

    // Fail-fast: the opening recovered from each folded `y_ring` must match the
    // carried claim (degree-one) or the extension-opening reduction's final
    // claim (proper extension). The verifier re-derives the same relation.
    let internal_claims = y_rings
        .iter()
        .zip(prepared_points.iter())
        .map(|(y_ring, prepared_point)| {
            recover_ring_subfield_inner_product::<F, L, D>(y_ring, &prepared_point.inner_reduction)
        })
        .collect::<Result<Vec<_>, _>>()?;
    match &reduction {
        Some(reduction) => {
            check_extension_opening_reduction_output(
                reduction.final_claim,
                internal_claims[0],
                reduction.final_factor,
            )?;
        }
        None => {
            for (claim, &internal_claim) in carried_openings.iter().zip(internal_claims.iter()) {
                if internal_claim != claim.opening {
                    return Err(AkitaError::InvalidInput(
                        "recursive opening does not match carried claim".to_string(),
                    ));
                }
            }
        }
    }

    // Build the quadratic sources and the concatenated commitment rows in claim
    // order. Source 0 is the recursive witness; extras follow.
    let commitment_u = commitment.as_ring_slice::<D>()?;
    let mut quadratic_sources = Vec::with_capacity(extra_carried_sources.len() + 1);
    quadratic_sources.push(RecursiveQuadraticSource {
        witness: witness_view,
        commitment: commitment_u,
        hint: typed_hint,
    });
    for source in &extra_carried_sources {
        quadratic_sources.push(RecursiveQuadraticSource {
            witness: source.w.view::<F, D>()?,
            commitment: source.commitment.as_ring_slice::<D>()?,
            hint: source.hint.to_typed::<D>()?,
        });
    }
    let claim_to_source = carried_openings
        .iter()
        .map(|claim| claim.source_idx)
        .collect::<Vec<_>>();
    let mut commitment_rows = Vec::new();
    for claim in &carried_openings {
        let source = quadratic_sources
            .get(claim.source_idx)
            .ok_or_else(|| AkitaError::InvalidInput("carried source index out of range".into()))?;
        commitment_rows.extend_from_slice(source.commitment);
    }

    let ring_opening_points = prepared_points
        .iter()
        .map(|prepared_point| prepared_point.ring_opening_point.clone())
        .collect::<Vec<_>>();
    let ring_multiplier_points = prepared_points
        .iter()
        .map(|prepared_point| prepared_point.ring_multiplier_point.clone())
        .collect::<Vec<_>>();
    let (instance, witness) = RingRelationProver::new_recursive_multipoint::<F, D, _, _>(
        backend,
        prepared,
        ring_opening_points,
        ring_multiplier_points,
        quadratic_sources,
        claim_to_source,
        e_folded_by_claim,
        level_params.clone(),
        transcript,
        &y_rings,
        m_row_layout,
    )?;

    Ok(PreparedRecursiveFold {
        commitment_rows,
        instance,
        witness,
        reduction,
        y_rings,
        extra_carried_sources,
        #[cfg(feature = "zk")]
        y_rings_masked,
        #[cfg(feature = "zk")]
        zk_hiding,
    })
}
