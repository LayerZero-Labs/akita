use super::*;
use cfg_if::cfg_if;

struct PreparedRecursiveFold<F: FieldCore, L: FieldCore, const D: usize> {
    commitment: FlatRingVec<F>,
    instance: RingRelationInstance<F, D>,
    witness: RingRelationWitness<F, D>,
    extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
    trace_eval_target: L,
    trace_scale: L,
    trace_prepared: PreparedRecursiveOpeningPoint<F, L, D>,
    #[cfg(feature = "zk")]
    trace_eval_target_public: L,
    #[cfg(feature = "zk")]
    zk_hiding: ZkHidingProverState<F>,
}

#[cfg(not(feature = "zk"))]
type TerminalFoldResult<F, L> = TerminalLevelProof<F, L>;
#[cfg(feature = "zk")]
type TerminalFoldResult<F, L> = (TerminalLevelProof<F, L>, ZkHidingProverState<F>);

enum FoldProveOutput<F: FieldCore, L: FieldCore> {
    Intermediate(Box<ProveLevelOutput<F, L>>),
    Terminal(Box<TerminalFoldResult<F, L>>),
}

impl<F: FieldCore, L: FieldCore> FoldProveOutput<F, L> {
    fn get_intermediate(self) -> Result<ProveLevelOutput<F, L>, AkitaError> {
        match self {
            Self::Intermediate(out) => Ok(*out),
            Self::Terminal(_) => Err(AkitaError::InvalidInput(
                "intermediate fold unexpectedly returned terminal proof".to_string(),
            )),
        }
    }

    fn get_terminal(self) -> Result<TerminalFoldResult<F, L>, AkitaError> {
        match self {
            Self::Terminal(terminal) => Ok(*terminal),
            Self::Intermediate(_) => Err(AkitaError::InvalidInput(
                "terminal fold unexpectedly returned intermediate proof".to_string(),
            )),
        }
    }
}

type PreparedRecursiveOpenings<F, L, const D: usize> = (
    Option<RecursiveExtensionOpeningReduction<L>>,
    Vec<PreparedRecursiveOpeningPoint<F, L, D>>,
);

type EvaluatedRecursiveWitness<F, const D: usize> =
    (Vec<CyclotomicRing<F, D>>, Vec<Vec<CyclotomicRing<F, D>>>);

type BoundNextWitness<F> = (
    Option<NextWitnessCommitment<F>>,
    Option<CleartextWitnessProof<F>>,
);

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
pub fn prove_suffix<Cfg, T, B, const D: usize>(
    expanded: &Arc<AkitaExpandedSetup<Cfg::Field>>,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field, D>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    num_vars: usize,
    transcript: &mut T,
    starting_state: RecursiveProverState<Cfg::Field, Cfg::ChallengeField>,
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
            "prove_suffix expects a non-empty recursive suffix".to_string(),
        ));
    }
    let terminal_level = planned_num_levels - 1;

    let mut intermediate_levels = Vec::new();
    let mut current_state = starting_state;
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
                false,
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
                    false,
                )
            })
        }?;
        let out = out.get_intermediate()?;
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
            true,
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
                    MRowLayout::WithoutDBlock,
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
                true,
            )
        })
    }?;
    #[cfg(not(feature = "zk"))]
    let terminal = terminal_result.get_terminal()?;
    #[cfg(feature = "zk")]
    let (terminal, zk_hiding) = terminal_result.get_terminal()?;

    let mut steps = intermediate_levels
        .into_iter()
        .map(AkitaProofStep::Intermediate)
        .collect::<Vec<_>>();
    steps.push(AkitaProofStep::Terminal(terminal));

    Ok(RecursiveSuffixOutcome {
        steps,
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
    is_terminal_fold: bool,
) -> Result<FoldProveOutput<F, L>, AkitaError>
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
    #[cfg(feature = "zk")]
    let mut zk_hiding = prepared_fold.zk_hiding;
    let commitment_u = prepared_fold.commitment.as_ring_slice::<D>()?;
    let logical_w = ring_switch_build_w::<F, B, D>(
        &prepared_fold.instance,
        prepared_fold.witness,
        backend,
        prepared,
        lp,
    )?;
    let next_commitment = if is_terminal_fold {
        None
    } else {
        let _span = tracing::info_span!("commit_w_level", level).entered();
        Some(crate::commit_next_w::<Cfg, B, D>(
            next_level_params,
            expanded,
            backend,
            prepared,
            &logical_w,
        )?)
    };
    let (next_commitment, final_witness) = bind_next_witness_for_ring_switch::<F, T, D>(
        transcript,
        is_terminal_fold,
        lp,
        &prepared_fold.instance,
        &logical_w,
        next_commitment,
        if is_terminal_fold {
            Some(next_level_params.log_basis)
        } else {
            None
        },
    )?;
    let m_row_layout = if is_terminal_fold {
        MRowLayout::WithoutDBlock
    } else {
        MRowLayout::WithDBlock
    };
    let rs = ring_switch_finalize::<F, L, T, D>(
        &prepared_fold.instance,
        expanded.as_ref(),
        transcript,
        &logical_w,
        lp,
        None,
        m_row_layout,
    )?;

    let relation_rows = if is_terminal_fold {
        &[][..]
    } else {
        prepared_fold.instance.v.as_slice()
    };
    let relation_claim = relation_claim_from_rows_extension::<F, L, D>(
        &rs.tau1,
        rs.alpha,
        relation_rows,
        commitment_u,
    )?;
    #[cfg(feature = "zk")]
    let relation_claim_public = relation_claim;
    #[cfg(feature = "zk")]
    let stage2_round_pads;
    let (stage1_proof, stage1_point, s_claim) = if is_terminal_fold {
        #[cfg(feature = "zk")]
        {
            stage2_round_pads =
                zk_hiding.take_compressed_rounds::<L>(rs.col_bits + rs.ring_bits, 3)?;
        }
        (None, vec![L::zero(); rs.col_bits + rs.ring_bits], L::zero())
    } else {
        #[cfg(feature = "zk")]
        let (stage1_round_pads, stage1_child_claim_masks, next_stage2_round_pads) =
            zk_hiding.take_current_level_pads::<L>(rs.col_bits + rs.ring_bits, rs.b)?;
        #[cfg(feature = "zk")]
        {
            stage2_round_pads = next_stage2_round_pads;
        }
        let (stage1_proof, stage1_point, s_claim) = prove_stage1::<F, L, T>(
            transcript,
            &rs,
            #[cfg(feature = "zk")]
            stage1_round_pads,
            #[cfg(feature = "zk")]
            stage1_child_claim_masks,
        )?;
        transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &stage1_proof.s_claim);
        (Some(stage1_proof), stage1_point, s_claim)
    };
    let batching_coeff: L = if is_terminal_fold {
        L::zero()
    } else {
        sample_ext_challenge::<F, L, T>(transcript, CHALLENGE_SUMCHECK_BATCH)
    };
    let trace_coeff = if is_terminal_fold {
        let trace_gamma = sample_ext_challenge::<F, L, T>(transcript, CHALLENGE_SUMCHECK_BATCH);
        trace_gamma * trace_gamma
    } else {
        batching_coeff * batching_coeff
    };
    let trace_opening_claim = trace_input_claim(trace_coeff, prepared_fold.trace_eval_target);
    #[cfg(feature = "zk")]
    let trace_eval_target_public_claim =
        trace_input_claim(trace_coeff, prepared_fold.trace_eval_target_public);
    ensure_trace_stage2_supported(L::EXT_DEGREE)?;
    let trace_compact = Some(build_recursive_stage2_trace_table::<F, L, D>(
        lp,
        &prepared_fold.instance,
        &prepared_fold.trace_prepared,
        prepared_fold.trace_scale,
        trace_coeff,
        rs.col_bits,
        rs.ring_bits,
        rs.live_x_cols,
    )?);
    let ring_bits = rs.ring_bits;
    let tau1 = rs.tau1.clone();
    let alpha = rs.alpha;
    #[cfg(feature = "zk")]
    let stage1_s_claim = stage1_proof
        .as_ref()
        .map(|proof| proof.s_claim)
        .unwrap_or_else(L::zero);
    let (stage2_sumcheck_proof, sumcheck_challenges, stage2_prover) = prove_stage2::<F, L, T>(
        transcript,
        batching_coeff,
        rs,
        &stage1_point,
        s_claim,
        relation_claim,
        #[cfg(feature = "zk")]
        relation_claim_public,
        #[cfg(feature = "zk")]
        stage1_s_claim,
        trace_compact,
        trace_opening_claim,
        #[cfg(feature = "zk")]
        trace_eval_target_public_claim,
        #[cfg(feature = "zk")]
        stage2_round_pads,
    )?;
    if is_terminal_fold {
        let final_witness = final_witness.ok_or_else(|| {
            AkitaError::InvalidInput("terminal fold did not bind a final witness".to_string())
        })?;
        let proof = TerminalLevelProof::new_with_extension_opening_reduction(
            prepared_fold.extension_opening_reduction,
            #[cfg(not(feature = "zk"))]
            stage2_sumcheck_proof,
            #[cfg(feature = "zk")]
            stage2_sumcheck_proof,
            final_witness,
        );
        cfg_if! {
            if #[cfg(feature = "zk")] {
                Ok(FoldProveOutput::Terminal(Box::new((proof, zk_hiding))))
            } else {
                Ok(FoldProveOutput::Terminal(Box::new(proof)))
            }
        }
    } else {
        let w_eval = {
            let _span = tracing::info_span!("multilinear_eval", level).entered();
            stage2_prover.final_w_eval()
        };
        #[cfg(feature = "zk")]
        let proof_w_eval = w_eval + zk_hiding.take_next_w_eval_mask::<L>()?;
        #[cfg(not(feature = "zk"))]
        let proof_w_eval = w_eval;
        transcript.append_serde(ABSORB_STAGE2_NEXT_W_EVAL, &proof_w_eval);
        let stage3_sumcheck_proof = prove_stage3::<F, L, T, D>(
            setup_contribution_mode,
            expanded.as_ref(),
            prefix_slots,
            lp,
            next_level_params,
            &prepared_fold.instance,
            &tau1,
            alpha,
            &sumcheck_challenges,
            ring_bits,
            transcript,
        )?;
        let stage1_proof = stage1_proof.ok_or_else(|| {
            AkitaError::InvalidInput("intermediate fold missing stage-1 proof".to_string())
        })?;
        let NextWitnessCommitment {
            witness: packed_witness,
            commitment: committed_commitment,
            hint: committed_hint,
        } = next_commitment.ok_or_else(|| {
            AkitaError::InvalidInput("intermediate fold did not bind a next commitment".to_string())
        })?;
        let w_commitment_proof = committed_commitment.clone();
        let mut level_proof =
            AkitaLevelProof::new_two_stage_many_with_extension_opening_reduction::<D>(
                prepared_fold.extension_opening_reduction,
                prepared_fold.instance.v,
                stage1_proof,
                #[cfg(not(feature = "zk"))]
                stage2_sumcheck_proof,
                #[cfg(feature = "zk")]
                stage2_sumcheck_proof,
                w_commitment_proof,
                proof_w_eval,
            );
        level_proof.stage3_sumcheck_proof = stage3_sumcheck_proof;

        let (committed_witness, logical_w) = match packed_witness {
            Some(packed_witness) => (packed_witness, Some(logical_w)),
            None => (logical_w, None),
        };

        Ok(FoldProveOutput::Intermediate(Box::new(ProveLevelOutput {
            level_proof,
            next_state: RecursiveProverState {
                w: committed_witness,
                logical_w,
                commitment: committed_commitment,
                hint: committed_hint,
                log_basis: next_level_params.log_basis,
                sumcheck_challenges,
                opening: w_eval,
                #[cfg(feature = "zk")]
                opening_public: proof_w_eval,
                #[cfg(feature = "zk")]
                zk_hiding,
            },
        })))
    }
}

#[allow(clippy::too_many_arguments)]
fn bind_next_witness_for_ring_switch<F, T, const D: usize>(
    transcript: &mut T,
    is_terminal_fold: bool,
    lp: &LevelParams,
    instance: &RingRelationInstance<F, D>,
    logical_w: &RecursiveWitnessFlat,
    next_commitment: Option<NextWitnessCommitment<F>>,
    final_log_basis: Option<u32>,
) -> Result<BoundNextWitness<F>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    if is_terminal_fold {
        let final_log_basis = final_log_basis.ok_or_else(|| {
            AkitaError::InvalidInput("terminal fold missing final witness basis".to_string())
        })?;
        let final_witness = CleartextWitnessProof::PackedDigits(
            PackedDigits::from_i8_digits_with_min_bits(logical_w.as_i8_digits(), final_log_basis),
        );
        let terminal_layout = terminal_witness_segment_layout(
            lp,
            instance.claim_to_point().len(),
            instance.num_public_rows(),
            F::modulus_bits(),
        )?;
        let parts = final_witness.terminal_transcript_parts(terminal_layout)?;
        if final_witness.packed_i8_digits()?.as_slice() != logical_w.as_i8_digits() {
            return Err(AkitaError::InvalidInput(
                "terminal final witness does not match ring-switch witness".to_string(),
            ));
        }
        transcript.append_bytes(ABSORB_TERMINAL_W_REMAINDER, &parts.remainder);
        return Ok((None, Some(final_witness)));
    }

    let next_commitment = next_commitment.ok_or_else(|| {
        AkitaError::InvalidInput("intermediate fold missing next commitment".to_string())
    })?;
    transcript.append_serde(
        ABSORB_NEXT_LEVEL_WITNESS_BINDING,
        &next_commitment.commitment,
    );
    Ok((Some(next_commitment), None))
}

#[allow(clippy::too_many_arguments)]
fn prove_stage1<F, L, T>(
    transcript: &mut T,
    rs: &RingSwitchOutput<L>,
    #[cfg(feature = "zk")] stage1_round_pads: Vec<Vec<akita_sumcheck::EqFactoredUniPoly<L>>>,
    #[cfg(feature = "zk")] stage1_child_claim_masks: Vec<Vec<L>>,
) -> Result<(AkitaStage1Proof<L>, Vec<L>, L), AkitaError>
where
    F: FieldCore + CanonicalField,
    L: ExtField<F> + HasUnreducedOps + HasOptimizedFold + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
    let tau0_reordered = reorder_stage1_coords(&rs.tau0, rs.col_bits, rs.ring_bits);
    let stage1_prover = AkitaStage1Prover::new(
        &rs.w_evals_compact,
        &tau0_reordered,
        rs.b,
        rs.live_x_cols,
        rs.col_bits,
        rs.ring_bits,
    )?;
    cfg_if! {
        if #[cfg(feature = "zk")] {
            stage1_prover.prove::<F, T>(transcript, stage1_round_pads, stage1_child_claim_masks)
        } else {
            let (stage1_proof, stage1_point) = stage1_prover.prove::<F, T>(transcript)?;
            let s_claim = stage1_proof.s_claim;
            Ok((stage1_proof, stage1_point, s_claim))
        }
    }
}

#[cfg(not(feature = "zk"))]
type Stage2ProveResult<L> = (SumcheckProof<L>, Vec<L>, AkitaStage2Prover<L>);
#[cfg(feature = "zk")]
type Stage2ProveResult<L> = (SumcheckProofMasked<L>, Vec<L>, AkitaStage2Prover<L>);

#[allow(clippy::too_many_arguments)]
fn prove_stage2<F, L, T>(
    transcript: &mut T,
    batching_coeff: L,
    rs: RingSwitchOutput<L>,
    stage1_point: &[L],
    s_claim: L,
    relation_claim: L,
    #[cfg(feature = "zk")] relation_claim_public: L,
    #[cfg(feature = "zk")] stage1_s_claim: L,
    trace_compact: Option<TraceTable<L>>,
    trace_opening_claim: L,
    #[cfg(feature = "zk")] trace_eval_target_public_claim: L,
    #[cfg(feature = "zk")] stage2_round_pads: Vec<CompressedUniPoly<L>>,
) -> Result<Stage2ProveResult<L>, AkitaError>
where
    F: FieldCore + CanonicalField,
    L: ExtField<F> + HasUnreducedOps + HasOptimizedFold + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
    let mut stage2_prover = AkitaStage2Prover::new(
        batching_coeff,
        rs.w_evals_compact,
        stage1_point,
        s_claim,
        rs.b,
        rs.alpha_evals_y,
        rs.m_evals_x,
        rs.live_x_cols,
        rs.col_bits,
        rs.ring_bits,
        relation_claim,
        trace_compact.clone(),
        trace_opening_claim,
    )?;
    cfg_if! {
        if #[cfg(feature = "zk")] {
            let mut stage2_public_input = batching_coeff * stage1_s_claim + relation_claim_public;
            if trace_compact.is_some() {
                stage2_public_input += trace_eval_target_public_claim;
            }
            let (stage2_sumcheck_proof_masked, sumcheck_challenges) = stage2_prover
                .prove_zk::<F, T, _>(
                    stage2_public_input,
                    transcript,
                    |tr| sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND),
                    stage2_round_pads,
                )?;
            Ok((
                stage2_sumcheck_proof_masked,
                sumcheck_challenges,
                stage2_prover,
            ))
        } else {
            let (stage2_sumcheck_proof, sumcheck_challenges, _) = stage2_prover
                .prove::<F, T, _>(transcript, |tr| {
                    sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND)
                })?;
            Ok((stage2_sumcheck_proof, sumcheck_challenges, stage2_prover))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn prove_stage3<F, L, T, const D: usize>(
    setup_contribution_mode: SetupContributionMode,
    expanded: &AkitaExpandedSetup<F>,
    prefix_slots: &SetupPrefixProverRegistry<F, D>,
    lp: &LevelParams,
    next_level_params: &LevelParams,
    instance: &RingRelationInstance<F, D>,
    tau1: &[L],
    alpha: L,
    sumcheck_challenges: &[L],
    ring_bits: usize,
    transcript: &mut T,
) -> Result<Option<SetupSumcheckProof<L>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    L: RingSubfieldEncoding<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    match setup_contribution_mode {
        SetupContributionMode::Recursive => {
            let output = SetupSumcheckProver::prove::<F, T, _, D>(
                expanded,
                prefix_slots,
                lp,
                next_level_params,
                instance,
                tau1,
                alpha,
                &sumcheck_challenges[ring_bits..],
                transcript,
                |tr| sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND),
            )?;
            Ok(Some(SetupSumcheckProof {
                claim: output.claim,
                sumcheck: output.sumcheck,
            }))
        }
        SetupContributionMode::Direct => Ok(None),
    }
}

pub(in crate::protocol::flow) struct RecursiveExtensionOpeningReduction<L: FieldCore> {
    pub(in crate::protocol::flow) proof: ExtensionOpeningReductionProof<L>,
    pub(in crate::protocol::flow) rho: Vec<L>,
    /// EOR final sumcheck claim and transparent-factor evaluation. Retained so
    /// the prepare step can fail-fast cross-check the folded y-ring opening
    /// against the reduction output; the verifier enforces the same relation.
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
    cfg_if! {
        if #[cfg(feature = "zk")] {
            let (partial_masks, sumcheck_pads) =
                zk_hiding.take_extension_opening_reduction_pads::<L>(
                    tensor.column_partials.len(),
                    num_vars - split_bits,
                )?;
            let proof_partials = tensor
                .column_partials
                .iter()
                .copied()
                .zip(partial_masks)
                .map(|(partial, mask)| partial + mask)
                .collect::<Vec<_>>();
        } else {
            let proof_partials = tensor.column_partials.clone();
        }
    }
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

fn prepare_openings<F, L, T, const D: usize>(
    logical_w: &RecursiveWitnessFlat,
    opening_point: &[L],
    expected_opening: L,
    transcript: &mut T,
    level: usize,
    level_params: &LevelParams,
    #[cfg(feature = "zk")] zk_hiding: &mut ZkHidingProverState<F>,
) -> Result<PreparedRecursiveOpenings<F, L, D>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    L: RingSubfieldEncoding<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + AkitaSerialize
        + MulBaseUnreduced<F>,
    T: Transcript<F>,
{
    let alpha = level_params.ring_dimension.trailing_zeros() as usize;
    let (reduction, protocol_point) = if <L as ExtField<F>>::EXT_DEGREE == 1 {
        (None, opening_point.to_vec())
    } else {
        let reduction = prove_extension_opening_reduction::<F, L, T>(
            logical_w,
            opening_point,
            expected_opening,
            transcript,
            #[cfg(feature = "zk")]
            zk_hiding,
        )?;
        let protocol_point = ring_subfield_packed_extension_opening_point::<F, L, D>(
            reduction.rho.len(),
            &reduction.rho,
        )?;
        (Some(reduction), protocol_point)
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

    Ok((reduction, prepared_points))
}

fn evaluate_witness<F, L, const D: usize>(
    witness_view: &RecursiveWitnessView<'_, F, D>,
    prepared_points: &[PreparedRecursiveOpeningPoint<F, L, D>],
    level: usize,
    level_params: &LevelParams,
) -> Result<EvaluatedRecursiveWitness<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField,
    L: FieldCore,
{
    let _span = tracing::info_span!(
        "evaluate_and_fold",
        level,
        num_ring_elems = witness_view.num_ring_elems(),
        num_points = prepared_points.len()
    )
    .entered();
    let mut y_rings = Vec::with_capacity(prepared_points.len());
    let mut folded = Vec::with_capacity(prepared_points.len());
    for prepared_point in prepared_points {
        let (y_ring, e_folded) = evaluate_recursive_witness_at_multiplier_point(
            witness_view,
            &prepared_point.ring_multiplier_point,
            level_params.block_len,
            level_params.num_blocks,
        )?;
        y_rings.push(y_ring);
        folded.push(e_folded);
    }
    Ok((y_rings, folded))
}

/// Derive the fused-trace evaluation target and EOR tail scale, and fail-fast
/// check that the folded witness ties back to the carried claim.
///
/// On the degree-one (no-EOR) path the target is the recovered subfield inner
/// product, which must equal the carried `expected_opening`. On the EOR path it
/// is the reduction's `final_claim`, cross-checked against the recovered value
/// scaled by the transparent factor. This writes nothing to the transcript: the
/// verifier re-derives the same relation through the fused stage-2 term.
fn compute_trace_target<F, L, const D: usize>(
    reduction: &Option<RecursiveExtensionOpeningReduction<L>>,
    y_rings: &[CyclotomicRing<F, D>],
    prepared_points: &[PreparedRecursiveOpeningPoint<F, L, D>],
    expected_opening: L,
) -> Result<(L, L), AkitaError>
where
    F: FieldCore + FromPrimitiveInt + Invertible,
    L: ExtField<F> + RingSubfieldEncoding<F>,
{
    #[cfg(not(feature = "zk"))]
    {
        match reduction {
            Some(reduction) => Ok((reduction.final_claim, reduction.final_factor)),
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
                Ok((opening, L::one()))
            }
        }
    }
    #[cfg(feature = "zk")]
    {
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
        match reduction {
            Some(reduction) => {
                check_extension_opening_reduction_output(
                    reduction.final_claim,
                    internal_claims[0],
                    reduction.final_factor,
                )?;
                Ok((reduction.final_claim, reduction.final_factor))
            }
            None => {
                if internal_claims[0] != expected_opening {
                    return Err(AkitaError::InvalidInput(
                        "recursive opening does not match carried claim".to_string(),
                    ));
                }
                Ok((internal_claims[0], L::one()))
            }
        }
    }
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

    let witness_view = current_state.w.view::<F, D>()?;
    let logical_w = current_state.logical_w.as_ref().unwrap_or(&current_state.w);
    let typed_hint = current_state.hint.to_typed::<D>()?;
    let opening_point = &current_state.sumcheck_challenges;
    #[cfg(feature = "zk")]
    let mut zk_hiding = current_state.zk_hiding;

    current_state
        .commitment
        .append_as_ring_commitment::<T, D>(ABSORB_COMMITMENT, transcript)?;

    let (reduction, prepared_points) = prepare_openings::<F, L, T, D>(
        logical_w,
        opening_point,
        current_state.opening,
        transcript,
        level,
        level_params,
        #[cfg(feature = "zk")]
        &mut zk_hiding,
    )?;

    let (y_rings, e_folded_by_claim) =
        evaluate_witness(&witness_view, &prepared_points, level, level_params)?;
    for prepared_point in &prepared_points {
        for pt in &prepared_point.padded_point {
            append_ext_field::<F, L, T>(transcript, ABSORB_EVALUATION_CLAIMS, pt);
        }
    }

    let (trace_eval_target, trace_scale) = compute_trace_target::<F, L, D>(
        &reduction,
        &y_rings,
        &prepared_points,
        current_state.opening,
    )?;
    #[cfg(feature = "zk")]
    let trace_eval_target_public = match &reduction {
        Some(reduction) => reduction.final_claim_public,
        None => current_state.opening_public,
    };
    let commitment_u = current_state.commitment.as_ring_slice::<D>()?;

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
        m_row_layout,
    )?;
    let trace_prepared = prepared_points
        .into_iter()
        .next()
        .ok_or(AkitaError::InvalidProof)?;

    Ok(PreparedRecursiveFold {
        commitment: current_state.commitment,
        instance,
        witness,
        extension_opening_reduction: reduction.map(|reduction| reduction.proof),
        trace_eval_target,
        trace_scale,
        trace_prepared,
        #[cfg(feature = "zk")]
        trace_eval_target_public,
        #[cfg(feature = "zk")]
        zk_hiding,
    })
}
