use super::*;

/// Dispatch one intermediate fold level to the correct ring dimension under
/// config `Cfg`.
///
/// Handles the fast path (`level_d == D`) and the dynamic-D path. The
/// `#[inline(never)]` attribute isolates the monomorphized match arms in their
/// own stack frame.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn dispatch_prove_level<Cfg, T, B, const D: usize>(
    level_d: usize,
    expanded: &Arc<AkitaExpandedSetup<Cfg::Field>>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    current_state: RecursiveProverState<Cfg::Field, Cfg::ChallengeField>,
    transcript: &mut T,
    level: usize,
    level_params: &LevelParams,
    next_params: LevelParams,
    setup_contribution_mode: SetupContributionMode,
) -> Result<ProveLevelOutput<Cfg::Field, Cfg::ChallengeField>, AkitaError>
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
    if level_d == D {
        prove_recursive_level::<Cfg, T, B, D>(
            expanded,
            backend,
            prepared,
            transcript,
            current_state,
            level,
            level_params,
            &next_params,
            setup_contribution_mode,
        )
    } else {
        dispatch_ring_dim_result!(level_d, |D_LEVEL| {
            let level_prepared = backend.prepare_expanded::<D_LEVEL>(expanded.clone())?;
            prove_recursive_level::<Cfg, T, B, { D_LEVEL }>(
                expanded,
                backend,
                &level_prepared,
                transcript,
                current_state,
                level,
                level_params,
                &next_params,
                setup_contribution_mode,
            )
        })
    }
}

/// Dispatch the terminal fold level to the correct ring dimension under config
/// `Cfg`.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn dispatch_prove_terminal_level<Cfg, T, B, const D: usize>(
    level_d: usize,
    expanded: &Arc<AkitaExpandedSetup<Cfg::Field>>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    current_state: &mut RecursiveProverState<Cfg::Field, Cfg::ChallengeField>,
    transcript: &mut T,
    level: usize,
    level_params: &LevelParams,
    final_log_basis: u32,
) -> Result<TerminalLevelProof<Cfg::Field, Cfg::ChallengeField>, AkitaError>
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
    if level_d == D {
        prove_terminal_recursive_level::<Cfg, T, B, D>(
            expanded.as_ref(),
            backend,
            prepared,
            transcript,
            current_state,
            level,
            level_params,
            final_log_basis,
        )
    } else {
        dispatch_ring_dim_result!(level_d, |D_LEVEL| {
            let level_prepared = backend.prepare_expanded::<D_LEVEL>(expanded.clone())?;
            prove_terminal_recursive_level::<Cfg, T, B, { D_LEVEL }>(
                expanded.as_ref(),
                backend,
                &level_prepared,
                transcript,
                current_state,
                level,
                level_params,
                final_log_basis,
            )
        })
    }
}

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
        let out = dispatch_prove_level::<Cfg, T, B, D>(
            level_params.ring_dimension,
            expanded,
            backend,
            prepared,
            current_state,
            transcript,
            level,
            &level_params,
            next_params,
            setup_contribution_mode,
        )?;
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
    let terminal = dispatch_prove_terminal_level::<Cfg, T, B, D>(
        level_params.ring_dimension,
        expanded,
        backend,
        prepared,
        &mut current_state,
        transcript,
        level,
        &level_params,
        next_params.log_basis,
    )?;

    Ok(RecursiveSuffixOutcome {
        intermediate_levels,
        terminal,
        #[cfg(feature = "zk")]
        zk_hiding: current_state.zk_hiding,
        num_levels: planned_num_levels,
    })
}

/// Prove one recursive fold level after the caller has built its ring-relation
/// equation and selected the commitment policy for the next `w`.
///
/// The caller owns config/schedule decisions through `commit_w_for_next`; this
/// function owns the config-free prover mechanics: build `w`, commit it using
/// that closure, finish ring switching, run stage-1/stage-2 sumchecks, and
/// produce the next recursive state.
///
/// # Errors
///
/// Returns an error if ring switching, recursive commitment, or either
/// sumcheck prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_fold_level_from_ring_relation<F, L, T, B, const D: usize, CommitW>(
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    commitment_u: &[CyclotomicRing<F, D>],
    level: usize,
    lp: &LevelParams,
    next_log_basis: u32,
    instance: RingRelationInstance<F, D>,
    witness: RingRelationWitness<F, D>,
    extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
    gamma_tr: L,
    trace_opening: L,
    #[cfg(feature = "zk")] trace_opening_public: L,
    trace_scale: L,
    trace_prepared: Option<&PreparedRecursiveOpeningPoint<F, L, D>>,
    #[cfg(feature = "zk")] mut zk_hiding: ZkHidingProverState<F>,
    setup_contribution_mode: SetupContributionMode,
    commit_w_for_next: CommitW,
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
    CommitW: FnOnce(&RecursiveWitnessFlat) -> Result<NextWitnessCommitment<F>, AkitaError>,
{
    let logical_w = ring_switch_build_w::<F, B, D>(&instance, witness, backend, prepared, lp)?;
    let next_commitment = {
        let _span = tracing::info_span!("commit_w_level", level).entered();
        commit_w_for_next(&logical_w)?
    };
    let NextWitnessCommitment {
        witness: packed_witness,
        commitment: committed_commitment,
        hint: committed_hint,
    } = next_commitment;
    let w_commitment_proof = committed_commitment.clone();
    let rs = ring_switch_finalize::<F, L, T, D>(
        &instance,
        expanded,
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
    )?;
    #[cfg(feature = "zk")]
    let relation_claim_public = relation_claim;
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
    let trace_opening_claim = trace_input_claim(gamma_tr, trace_opening);
    #[cfg(feature = "zk")]
    let trace_opening_public_claim = trace_input_claim(gamma_tr, trace_opening_public);
    let trace_compact =
        if !trace_stage2_enabled(lp, L::EXT_DEGREE, extension_opening_reduction.is_some()) {
            None
        } else if let Some(prepared) = trace_prepared {
            Some(build_recursive_stage2_trace_compact::<F, L, D>(
                lp,
                &instance,
                prepared,
                trace_scale,
                col_bits,
                ring_bits,
                live_x_cols,
            )?)
        } else {
            None
        };
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
            trace_compact.clone(),
            gamma_tr,
            trace_opening_claim,
        );
        let mut stage2_prover = stage2_prover_result?;
        let mut stage2_public_input = batching_coeff * stage1_proof.s_claim + relation_claim_public;
        if trace_compact.is_some() {
            stage2_public_input += trace_opening_public_claim;
        }
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
            trace_compact,
            gamma_tr,
            trace_opening_claim,
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
                |tr| sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND),
            )?;
            Some(SetupSumcheckProof {
                claim: output.claim,
                sumcheck: output.sumcheck,
            })
        }
        SetupContributionMode::Direct => None,
    };

    let mut level_proof = AkitaLevelProof::new_two_stage_many_with_extension_opening_reduction::<D>(
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

    Ok(ProveLevelOutput {
        level_proof,
        next_state: RecursiveProverState {
            w: committed_witness,
            logical_w,
            commitment: committed_commitment,
            hint: committed_hint,
            log_basis: next_log_basis,
            sumcheck_challenges,
            opening: w_eval,
            #[cfg(feature = "zk")]
            opening_public: w_eval_masked,
            #[cfg(feature = "zk")]
            zk_hiding,
        },
    })
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
pub fn prove_terminal_fold_level_from_ring_relation<F, L, T, B, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    commitment_u: &[CyclotomicRing<F, D>],
    _level: usize,
    lp: &LevelParams,
    final_log_basis: u32,
    instance: RingRelationInstance<F, D>,
    witness: RingRelationWitness<F, D>,
    extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
    gamma_tr: L,
    trace_opening: L,
    #[cfg(feature = "zk")] trace_opening_public: L,
    trace_scale: L,
    trace_prepared: Option<&PreparedRecursiveOpeningPoint<F, L, D>>,
    #[cfg(feature = "zk")] zk_hiding: &mut ZkHidingProverState<F>,
) -> Result<TerminalLevelProof<F, L>, AkitaError>
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
    let relation_claim =
        relation_claim_from_rows_extension::<F, L, D>(&rs.tau1, rs.alpha, &[], commitment_u)?;
    #[cfg(feature = "zk")]
    let relation_claim_public = relation_claim;
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
    let trace_opening_claim = trace_input_claim(gamma_tr, trace_opening);
    #[cfg(feature = "zk")]
    let trace_opening_public_claim = trace_input_claim(gamma_tr, trace_opening_public);
    let trace_compact =
        if !trace_stage2_enabled(lp, L::EXT_DEGREE, extension_opening_reduction.is_some()) {
            None
        } else if let Some(prepared) = trace_prepared {
            Some(build_recursive_stage2_trace_compact::<F, L, D>(
                lp,
                &instance,
                prepared,
                trace_scale,
                col_bits,
                ring_bits,
                live_x_cols,
            )?)
        } else {
            None
        };

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
            trace_compact.clone(),
            gamma_tr,
            trace_opening_claim,
        )?;
        let mut stage2_public_input = relation_claim_public;
        if trace_compact.is_some() {
            stage2_public_input += trace_opening_public_claim;
        }
        let (stage2_sumcheck_proof_masked, _sumcheck_challenges) = stage2_prover
            .prove_zk::<F, T, _>(
                stage2_public_input,
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
            trace_compact,
            gamma_tr,
            trace_opening_claim,
        )?;
        let (stage2_sumcheck, _sumcheck_challenges, _stage2_final_claim) = stage2_prover
            .prove::<F, T, _>(transcript, |tr| {
                sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND)
            })?;
        (stage2_sumcheck, _sumcheck_challenges)
    };
    let proof = TerminalLevelProof::new_with_extension_opening_reduction(
        extension_opening_reduction,
        #[cfg(not(feature = "zk"))]
        stage2_sumcheck,
        #[cfg(feature = "zk")]
        stage2_sumcheck_proof_masked,
        final_witness,
    );
    Ok(proof)
}

pub(in crate::protocol::flow) struct RecursiveExtensionOpeningReduction<L: FieldCore> {
    pub(in crate::protocol::flow) proof: ExtensionOpeningReductionProof<L>,
    pub(in crate::protocol::flow) rho: Vec<L>,
    pub(in crate::protocol::flow) final_claim: L,
    #[cfg(feature = "zk")]
    pub(in crate::protocol::flow) final_claim_public: L,
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

pub(in crate::protocol::flow) fn prove_recursive_extension_opening_reduction<F, L, T>(
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
    #[cfg(feature = "zk")]
    let final_claim_public =
        masked_sumcheck_final_claim(input_claim, &sumcheck_proof_masked, &rho)?;
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
        #[cfg(feature = "zk")]
        final_claim_public,
        final_factor,
    })
}

/// Prove one recursive fold level using already-selected current and next
/// level parameters.
///
/// The caller owns schedule/config selection and passes the next-level
/// commitment policy as a closure. This function owns recursive opening-point
/// reduction, witness folding, public recursive transcript absorbs, recursive
/// ring-relation construction, and the folded-level prover mechanics.
///
/// # Errors
///
/// Returns an error if the recursive opening point has the wrong dimension,
/// witness folding or ring-relation construction fails, or the folded
/// prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_recursive_fold_with_params<F, L, T, B, const D: usize, CommitW>(
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    current_state: RecursiveProverState<F, L>,
    level: usize,
    level_params: &LevelParams,
    next_log_basis: u32,
    setup_contribution_mode: SetupContributionMode,
    commit_w_for_next: CommitW,
) -> Result<ProveLevelOutput<F, L>, AkitaError>
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
    CommitW: FnOnce(&RecursiveWitnessFlat) -> Result<NextWitnessCommitment<F>, AkitaError>,
{
    {
        let x: u8 = 0;
        tracing::trace!(
            stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
            level,
            "prove_recursive_fold_with_params"
        );
    }

    let RecursiveProverState {
        w,
        logical_w,
        commitment,
        hint,
        sumcheck_challenges,
        opening: expected_opening,
        #[cfg(feature = "zk")]
        opening_public,
        log_basis: _,
        #[cfg(feature = "zk")]
        zk_hiding,
    } = current_state;
    let witness_view = w.view::<F, D>()?;
    let logical_w = logical_w.as_ref().unwrap_or(&w);
    let typed_hint = hint.to_typed::<D>()?;
    let opening_point = &sumcheck_challenges;
    #[cfg(feature = "zk")]
    let mut zk_hiding = zk_hiding;

    let alpha = level_params.ring_dimension.trailing_zeros() as usize;
    commitment.append_as_ring_commitment::<T, D>(ABSORB_COMMITMENT, transcript)?;

    let reduction = if <L as ExtField<F>>::EXT_DEGREE == 1 {
        None
    } else {
        Some(prove_recursive_extension_opening_reduction::<F, L, T>(
            logical_w,
            opening_point,
            expected_opening,
            transcript,
            #[cfg(feature = "zk")]
            &mut zk_hiding,
        )?)
    };
    let protocol_point = match &reduction {
        Some(reduction) => ring_subfield_packed_extension_opening_point::<F, L, D>(
            reduction.rho.len(),
            &reduction.rho,
        )?,
        None => opening_point.to_vec(),
    };
    let prepared_points = {
        let _span = tracing::info_span!("ring_opening_point", level).entered();
        vec![prepare_recursive_opening_point_ext::<F, L, D>(
            &protocol_point,
            BasisMode::Lagrange,
            level_params,
            alpha,
            BlockOrder::ColumnMajor,
        )?]
    };

    let (y_rings, e_folded_by_claim) = {
        let _span = tracing::info_span!(
            "evaluate_and_fold",
            level,
            num_ring_elems = witness_view.num_ring_elems(),
            num_points = prepared_points.len()
        )
        .entered();
        let mut y_rings = Vec::with_capacity(prepared_points.len());
        let mut folded = Vec::with_capacity(prepared_points.len());
        for prepared_point in &prepared_points {
            let (y_ring, e_folded) = evaluate_recursive_witness_at_multiplier_point(
                &witness_view,
                &prepared_point.ring_multiplier_point,
                level_params.block_len,
                level_params.num_blocks,
            )?;
            y_rings.push(y_ring);
            folded.push(e_folded);
        }
        (y_rings, folded)
    };
    for prepared_point in &prepared_points {
        for pt in &prepared_point.padded_point {
            append_ext_field::<F, L, T>(transcript, ABSORB_EVALUATION_CLAIMS, pt);
        }
    }
    let gamma_tr: L = sample_ext_challenge::<F, L, T>(transcript, CHALLENGE_TRACE_BATCH);
    #[cfg(not(feature = "zk"))]
    let (trace_opening, trace_scale) = match &reduction {
        Some(reduction) => (reduction.final_claim, reduction.final_factor),
        None => {
            let y_ring = y_rings.first().ok_or(AkitaError::InvalidProof)?;
            let prepared_point = prepared_points.first().ok_or(AkitaError::InvalidProof)?;
            let opening = recover_ring_subfield_inner_product::<F, L, D>(
                y_ring,
                &prepared_point.packed_inner_point,
            )?;
            if opening != expected_opening {
                return Err(AkitaError::InvalidInput(
                    "recursive opening does not match carried claim".to_string(),
                ));
            }
            (opening, L::one())
        }
    };
    #[cfg(feature = "zk")]
    let (trace_opening, trace_scale) = {
        let internal_claims = y_rings
            .iter()
            .zip(prepared_points.iter())
            .map(|(y_ring, prepared_point)| {
                recover_ring_subfield_inner_product::<F, L, D>(
                    y_ring,
                    &prepared_point.packed_inner_point,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        match &reduction {
            Some(reduction) => {
                check_extension_opening_reduction_output(
                    reduction.final_claim,
                    internal_claims[0],
                    reduction.final_factor,
                )?;
                (reduction.final_claim, reduction.final_factor)
            }
            None => {
                if internal_claims[0] != expected_opening {
                    return Err(AkitaError::InvalidInput(
                        "recursive opening does not match carried claim".to_string(),
                    ));
                }
                (internal_claims[0], L::one())
            }
        }
    };
    #[cfg(feature = "zk")]
    let trace_opening_public = match &reduction {
        Some(reduction) => reduction.final_claim_public,
        None => opening_public,
    };
    let commitment_u = commitment.as_ring_slice::<D>()?;

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
        &witness_view,
        e_folded_by_claim,
        level_params.clone(),
        typed_hint,
        transcript,
        commitment_u,
        &y_rings,
        MRowLayout::WithDBlock,
    )?;

    let extension_opening_reduction = reduction.map(|reduction| reduction.proof);
    prove_fold_level_from_ring_relation::<F, L, T, B, D, _>(
        expanded,
        backend,
        prepared,
        transcript,
        commitment_u,
        level,
        level_params,
        next_log_basis,
        instance,
        witness,
        extension_opening_reduction,
        gamma_tr,
        trace_opening,
        #[cfg(feature = "zk")]
        trace_opening_public,
        trace_scale,
        Some(&prepared_points[0]),
        #[cfg(feature = "zk")]
        zk_hiding,
        setup_contribution_mode,
        commit_w_for_next,
    )
}

/// Mirror of [`prove_recursive_fold_with_params`] producing a
/// [`TerminalLevelProof`] instead of an intermediate
/// [`AkitaLevelProof`] + next-witness commitment pair.
///
/// All recursive-opening, witness folding, and ring-relation setup is
/// identical to the intermediate path. The two differ only inside the inner
/// fold proof (see [`prove_terminal_fold_level_from_ring_relation`]).
///
/// # Errors
///
/// Returns an error if recursive-opening setup, witness folding, or the
/// inner terminal fold-level prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_terminal_recursive_fold_with_params<F, L, T, B, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    current_state: &mut RecursiveProverState<F, L>,
    level: usize,
    level_params: &LevelParams,
    final_log_basis: u32,
) -> Result<TerminalLevelProof<F, L>, AkitaError>
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
            "prove_terminal_recursive_fold_with_params"
        );
    }

    let typed_hint = current_state.hint.to_typed::<D>()?;
    let opening_point = &current_state.sumcheck_challenges;
    let expected_opening = current_state.opening;
    let commitment = &current_state.commitment;

    let alpha = level_params.ring_dimension.trailing_zeros() as usize;
    let commitment_u = commitment.as_ring_slice::<D>()?;
    commitment.append_as_ring_commitment::<T, D>(ABSORB_COMMITMENT, transcript)?;

    let reduction = if <L as ExtField<F>>::EXT_DEGREE == 1 {
        None
    } else {
        let logical_w = current_state.logical_w.as_ref().unwrap_or(&current_state.w);
        Some(prove_recursive_extension_opening_reduction::<F, L, T>(
            logical_w,
            opening_point,
            expected_opening,
            transcript,
            #[cfg(feature = "zk")]
            &mut current_state.zk_hiding,
        )?)
    };
    let witness_view = current_state.w.view::<F, D>()?;
    let protocol_point = match &reduction {
        Some(reduction) => ring_subfield_packed_extension_opening_point::<F, L, D>(
            reduction.rho.len(),
            &reduction.rho,
        )?,
        None => opening_point.to_vec(),
    };
    let prepared_points = {
        let _span = tracing::info_span!("ring_opening_point", level).entered();
        vec![prepare_recursive_opening_point_ext::<F, L, D>(
            &protocol_point,
            BasisMode::Lagrange,
            level_params,
            alpha,
            BlockOrder::ColumnMajor,
        )?]
    };

    let (y_rings, e_folded_by_claim) = {
        let _span = tracing::info_span!(
            "evaluate_and_fold",
            level,
            num_ring_elems = witness_view.num_ring_elems(),
            num_points = prepared_points.len()
        )
        .entered();
        let mut y_rings = Vec::with_capacity(prepared_points.len());
        let mut folded = Vec::with_capacity(prepared_points.len());
        for prepared_point in &prepared_points {
            let (y_ring, e_folded) = evaluate_recursive_witness_at_multiplier_point(
                &witness_view,
                &prepared_point.ring_multiplier_point,
                level_params.block_len,
                level_params.num_blocks,
            )?;
            y_rings.push(y_ring);
            folded.push(e_folded);
        }
        (y_rings, folded)
    };
    for prepared_point in &prepared_points {
        for pt in &prepared_point.padded_point {
            append_ext_field::<F, L, T>(transcript, ABSORB_EVALUATION_CLAIMS, pt);
        }
    }
    let gamma_tr: L = sample_ext_challenge::<F, L, T>(transcript, CHALLENGE_TRACE_BATCH);
    #[cfg(not(feature = "zk"))]
    let (trace_opening, trace_scale) = match &reduction {
        Some(reduction) => (reduction.final_claim, reduction.final_factor),
        None => {
            let y_ring = y_rings.first().ok_or(AkitaError::InvalidProof)?;
            let prepared_point = prepared_points.first().ok_or(AkitaError::InvalidProof)?;
            let opening = recover_ring_subfield_inner_product::<F, L, D>(
                y_ring,
                &prepared_point.packed_inner_point,
            )?;
            if opening != expected_opening {
                return Err(AkitaError::InvalidInput(
                    "recursive opening does not match carried claim".to_string(),
                ));
            }
            (opening, L::one())
        }
    };
    #[cfg(feature = "zk")]
    let (trace_opening, trace_scale) = {
        let internal_claims = y_rings
            .iter()
            .zip(prepared_points.iter())
            .map(|(y_ring, prepared_point)| {
                recover_ring_subfield_inner_product::<F, L, D>(
                    y_ring,
                    &prepared_point.packed_inner_point,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        match &reduction {
            Some(reduction) => {
                check_extension_opening_reduction_output(
                    reduction.final_claim,
                    internal_claims[0],
                    reduction.final_factor,
                )?;
                (reduction.final_claim, reduction.final_factor)
            }
            None => {
                if internal_claims[0] != expected_opening {
                    return Err(AkitaError::InvalidInput(
                        "recursive opening does not match carried claim".to_string(),
                    ));
                }
                (internal_claims[0], L::one())
            }
        }
    };
    #[cfg(feature = "zk")]
    let trace_opening_public = match &reduction {
        Some(reduction) => reduction.final_claim_public,
        None => current_state.opening_public,
    };

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
        &witness_view,
        e_folded_by_claim,
        level_params.clone(),
        typed_hint,
        transcript,
        commitment_u,
        &y_rings,
        MRowLayout::WithoutDBlock,
    )?;

    let extension_opening_reduction = reduction.map(|reduction| reduction.proof);
    prove_terminal_fold_level_from_ring_relation::<F, L, T, B, D>(
        expanded,
        backend,
        prepared,
        transcript,
        commitment_u,
        level,
        level_params,
        final_log_basis,
        instance,
        witness,
        extension_opening_reduction,
        gamma_tr,
        trace_opening,
        #[cfg(feature = "zk")]
        trace_opening_public,
        trace_scale,
        Some(&prepared_points[0]),
        #[cfg(feature = "zk")]
        &mut current_state.zk_hiding,
    )
}

/// Prove one recursive fold level from D-erased recursive state under config
/// `Cfg`.
///
/// Delegates witness unpacking and fold mechanics to
/// [`prove_recursive_fold_with_params`]; this wrapper only threads the
/// schedule-selected level params and next-witness commitment policy.
///
/// # Errors
///
/// Returns an error if the current witness cannot be viewed at `D`, the hint
/// cannot be typed at `D`, layout selection fails, or recursive proving fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_recursive_level<Cfg, T, B, const D: usize>(
    expanded: &Arc<AkitaExpandedSetup<Cfg::Field>>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    current_state: RecursiveProverState<Cfg::Field, Cfg::ChallengeField>,
    level: usize,
    level_params: &LevelParams,
    next_params: &LevelParams,
    setup_contribution_mode: SetupContributionMode,
) -> Result<ProveLevelOutput<Cfg::Field, Cfg::ChallengeField>, AkitaError>
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
    let _setup_span = tracing::info_span!("inter_level_setup", level).entered();
    drop(_setup_span);

    prove_recursive_fold_with_params::<Cfg::Field, Cfg::ChallengeField, T, B, D, _>(
        expanded.as_ref(),
        backend,
        prepared,
        transcript,
        current_state,
        level,
        level_params,
        next_params.log_basis,
        setup_contribution_mode,
        |w| crate::commit_next_w::<Cfg, B, D>(next_params, expanded, backend, prepared, w),
    )
}

/// Terminal-fold analogue of [`prove_recursive_level`].
///
/// Same input shape minus the next-witness commitment; the terminal fold ships
/// `final_witness` in cleartext (packed digits) instead of committing.
///
/// # Errors
///
/// Returns an error if witness unpacking or the underlying terminal fold
/// prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_terminal_recursive_level<Cfg, T, B, const D: usize>(
    expanded: &AkitaExpandedSetup<Cfg::Field>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    current_state: &mut RecursiveProverState<Cfg::Field, Cfg::ChallengeField>,
    level: usize,
    level_params: &LevelParams,
    final_log_basis: u32,
) -> Result<TerminalLevelProof<Cfg::Field, Cfg::ChallengeField>, AkitaError>
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
    let _setup_span = tracing::info_span!("inter_level_setup_terminal", level).entered();
    drop(_setup_span);

    prove_terminal_recursive_fold_with_params::<Cfg::Field, Cfg::ChallengeField, T, B, D>(
        expanded,
        backend,
        prepared,
        transcript,
        current_state,
        level,
        level_params,
        final_log_basis,
    )
}
