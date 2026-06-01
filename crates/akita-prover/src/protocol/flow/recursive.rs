use super::*;

/// Per-level proving request handed to the suffix prover closure.
pub enum SuffixLevelRequest<'a, F: FieldCore, L: FieldCore> {
    /// Intermediate fold level — caller must commit to the next witness via
    /// the prover's `commit_w_for_next` policy.
    Intermediate {
        /// Suffix level index (1-based; level 0 is the root).
        level: usize,
        /// Current recursive prover state entering the level.
        current_state: Box<RecursiveProverState<F, L>>,
        /// Current level parameters from the schedule.
        level_params: &'a LevelParams,
        /// Successor level parameters from the schedule.
        next_params: LevelParams,
    },
    /// Terminal fold level — caller emits the cleartext `final_witness` and
    /// does not commit to a next witness.
    Terminal {
        /// Suffix level index for the terminal fold.
        level: usize,
        /// Current recursive prover state entering the terminal fold.
        current_state: &'a mut RecursiveProverState<F, L>,
        /// Current level parameters from the schedule.
        level_params: &'a LevelParams,
        /// Bits-per-element used to pack the final witness as
        /// [`PackedDigits`].
        final_log_basis: u32,
    },
}

/// Per-level proving result returned by the suffix prover closure.
///
/// The `Intermediate` variant is intentionally much larger than `Terminal`
/// (it carries the next-level commitment, hint, packed witness, and full
/// `AkitaLevelProof`). This enum is a short-lived stack value passed through
/// a single closure, so the size disparity has no practical cost and the
/// `large_enum_variant` lint is suppressed locally.
#[allow(clippy::large_enum_variant)]
pub enum SuffixLevelOutput<F: FieldCore, L: FieldCore> {
    /// Result of proving an intermediate suffix level.
    Intermediate(ProveLevelOutput<F, L>),
    /// Result of proving the terminal suffix level.
    Terminal(TerminalLevelProof<F, L>),
}

/// Drive the recursive fold suffix using caller-supplied schedule and
/// per-level proving policies.
///
/// The caller supplies a single `prove_level` closure that dispatches on
/// [`SuffixLevelRequest`] (intermediate vs terminal) and produces the
/// matching [`SuffixLevelOutput`]. Earlier suffix levels run intermediate
/// folds; the last suffix level runs the terminal fold which ships the
/// cleartext `final_witness`.
///
/// # Errors
///
/// Returns an error if schedule selection fails, level proving fails, or
/// the closure returns the wrong [`SuffixLevelOutput`] variant for a given
/// [`SuffixLevelRequest`]. Returns an invalid-setup error when the
/// schedule's recursive suffix is empty (root-terminal proofs do not run
/// this helper).
pub fn prove_recursive_suffix_with_policy<F, L, SelectFold, ProveLevel>(
    num_vars: usize,
    initial_state: RecursiveProverState<F, L>,
    schedule: &Schedule,
    mut select_fold_execution: SelectFold,
    mut prove_level: ProveLevel,
) -> Result<RecursiveSuffixOutcome<F, L>, AkitaError>
where
    F: FieldCore,
    L: ExtField<F>,
    SelectFold:
        FnMut(usize, AkitaScheduleInputs, u32) -> Result<(LevelParams, LevelParams), AkitaError>,
    ProveLevel: FnMut(SuffixLevelRequest<'_, F, L>) -> Result<SuffixLevelOutput<F, L>, AkitaError>,
{
    let planned_num_levels = schedule_num_fold_levels(schedule);
    if planned_num_levels < 2 {
        return Err(AkitaError::InvalidSetup(
            "prove_recursive_suffix_with_policy expects a non-empty recursive suffix".to_string(),
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
            current_w_len: current_state.recursive_witness_len()?,
        };
        let (level_params, next_params) =
            select_fold_execution(level, inputs, current_state.log_basis)?;
        let out = prove_level(SuffixLevelRequest::Intermediate {
            level,
            current_state: Box::new(current_state),
            level_params: &level_params,
            next_params,
        })?;
        let SuffixLevelOutput::Intermediate(out) = out else {
            return Err(AkitaError::InvalidSetup(
                "prove_level returned a terminal proof for an intermediate level".to_string(),
            ));
        };
        intermediate_levels.push(out.level_proof);
        current_state = out.next_state;
        level += 1;
    }

    debug_assert_eq!(level, terminal_level);
    let inputs = AkitaScheduleInputs {
        num_vars,
        level,
        current_w_len: current_state.recursive_witness_len()?,
    };
    let (level_params, next_params) =
        select_fold_execution(level, inputs, current_state.log_basis)?;
    let out = prove_level(SuffixLevelRequest::Terminal {
        level,
        current_state: &mut current_state,
        level_params: &level_params,
        final_log_basis: next_params.log_basis,
    })?;
    let SuffixLevelOutput::Terminal(terminal) = out else {
        return Err(AkitaError::InvalidSetup(
            "prove_level returned an intermediate proof for the terminal level".to_string(),
        ));
    };

    Ok(RecursiveSuffixOutcome {
        intermediate_levels,
        terminal,
        #[cfg(feature = "zk")]
        zk_hiding: current_state.zk_hiding,
        num_levels: planned_num_levels,
    })
}

/// Prove one recursive fold level after the caller has built its quadratic
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
pub fn prove_fold_level_from_quadratic<F, L, T, B, const D: usize, CommitW>(
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    commitment_u: &[CyclotomicRing<F, D>],
    level: usize,
    lp: &LevelParams,
    next_log_basis: u32,
    mut quad_eq: Box<QuadraticEquation<F, { D }>>,
    extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
    y_rings: Vec<CyclotomicRing<F, D>>,
    #[cfg(feature = "zk")] proof_y_rings: Vec<CyclotomicRing<F, D>>,
    #[cfg(feature = "zk")] mut zk_hiding: ZkHidingProverState<F>,
    commit_w_for_next: CommitW,
) -> Result<ProveLevelOutput<F, L>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasUnreducedOps
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    L: ExtField<F> + RingSubfieldEncoding<F> + HasUnreducedOps + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
    B: ProverComputeBackend<F>,
    CommitW: FnOnce(&RecursiveWitnessFlat) -> Result<NextWitnessCommitment<F>, AkitaError>,
{
    let logical_w = ring_switch_build_w::<F, B, { D }>(&mut quad_eq, backend, prepared, lp)?;
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
    let rs = ring_switch_finalize::<F, L, T, { D }>(
        &quad_eq,
        expanded,
        transcript,
        &logical_w,
        &w_commitment_proof,
        lp,
        MRowLayout::Intermediate,
    )?;

    let relation_claim = relation_claim_from_rows_extension::<F, L, D>(
        &rs.tau1,
        rs.alpha,
        &quad_eq.v,
        commitment_u,
        &y_rings,
    )?;
    #[cfg(feature = "zk")]
    let relation_claim_public = relation_claim_from_rows_extension::<F, L, D>(
        &rs.tau1,
        rs.alpha,
        &quad_eq.v,
        commitment_u,
        &proof_y_rings,
    )?;
    let RingSwitchOutput {
        w_evals_compact,
        live_x_cols,
        m_evals_x,
        alpha_evals_y,
        col_bits,
        ring_bits,
        tau0,
        tau1: _,
        b,
        alpha: _,
    } = rs;
    let tau0_reordered = reorder_stage1_coords(&tau0, col_bits, ring_bits);
    #[cfg(feature = "zk")]
    let (stage1_round_pads, stage1_child_claim_masks, stage2_round_pads) =
        zk_hiding.take_current_level_pads::<L>(col_bits + ring_bits, b)?;
    let (stage1_proof, r_stage1, s_claim) = {
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
            let (stage1_proof, r_stage1) = stage1_prover.prove(transcript)?;
            let s_claim = stage1_proof.s_claim;
            (stage1_proof, r_stage1, s_claim)
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
            &r_stage1,
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
            &r_stage1,
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
    #[cfg(not(feature = "zk"))]
    let proof_y_rings = y_rings;
    let (level_proof, sumcheck_challenges) = (
        AkitaLevelProof::new_two_stage_many_with_extension_opening_reduction::<D>(
            proof_y_rings,
            extension_opening_reduction,
            quad_eq.v,
            stage1_proof,
            #[cfg(not(feature = "zk"))]
            stage2_sumcheck_proof,
            #[cfg(feature = "zk")]
            stage2_sumcheck_proof_masked,
            w_commitment_proof,
            proof_w_eval,
        ),
        sumcheck_challenges,
    );

    let (committed_witness, logical_w) = match packed_witness {
        Some(packed_witness) => (packed_witness, Some(logical_w)),
        None => (logical_w, None),
    };
    let committed_witness_len = committed_witness.len();

    Ok(ProveLevelOutput {
        level_proof,
        next_state: RecursiveProverState {
            w: committed_witness,
            logical_w,
            commitment: committed_commitment,
            hint: committed_hint,
            log_basis: next_log_basis,
            carried_openings: vec![RecursiveCarriedOpening::recursive_witness(
                sumcheck_challenges,
                w_eval,
                committed_witness_len,
            )],
            extra_carried_sources: Vec::new(),
            #[cfg(feature = "zk")]
            zk_hiding,
        },
    })
}

/// Prove the terminal recursive fold level after the caller has built its
/// quadratic equation.
///
/// At the terminal level the next witness is shipped in cleartext as
/// [`PackedDigits`], so this function:
///
/// * builds `logical_w` via ring switching,
/// * packs it into the terminal [`DirectWitnessProof`] using
///   `final_log_basis` as the planner-mandated minimum bits per element,
/// * absorbs logical `w_hat` before fold challenge sampling when the
///   quadratic equation is built, then absorbs the remaining final-witness
///   bytes before sampling any ring-switch challenges,
/// * skips the stage-1 sumcheck entirely (packed-digit range is structurally
///   enforced by the packing), and
/// * runs stage-2 in relation-only mode with `batching_coeff = 0`,
///   `s_claim = 0`, and dummy `r_stage1` zeros — these zero the virtual
///   sumcheck contribution leaving only the relation oracle.
///
/// # Errors
///
/// Returns an error if ring switching or the stage-2 sumcheck prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_terminal_fold_level_from_quadratic<F, L, T, B, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    commitment_u: &[CyclotomicRing<F, D>],
    _level: usize,
    lp: &LevelParams,
    final_log_basis: u32,
    mut quad_eq: Box<QuadraticEquation<F, { D }>>,
    extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
    y_rings: Vec<CyclotomicRing<F, D>>,
    #[cfg(feature = "zk")] y_rings_masked: Vec<CyclotomicRing<F, D>>,
    #[cfg(feature = "zk")] zk_hiding: &mut ZkHidingProverState<F>,
) -> Result<TerminalLevelProof<F, L>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasUnreducedOps
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    L: ExtField<F> + RingSubfieldEncoding<F> + HasUnreducedOps + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
    B: ProverComputeBackend<F>,
{
    let terminal_layout = terminal_witness_segment_layout(
        lp,
        quad_eq.claim_to_point().len(),
        quad_eq.num_public_rows(),
    )?;
    let logical_w = ring_switch_build_w::<F, B, { D }>(&mut quad_eq, backend, prepared, lp)?;
    let final_witness = DirectWitnessProof::PackedDigits(
        PackedDigits::from_i8_digits_with_min_bits(logical_w.as_i8_digits(), final_log_basis),
    );
    let rs = ring_switch_finalize_terminal::<F, L, T, { D }>(
        &quad_eq,
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
    // contribution to every round polynomial regardless of `r_stage1`, so
    // dummy zeros for `r_stage1` and `s_claim` are safe.
    let r_stage1 = vec![L::zero(); col_bits + ring_bits];
    #[cfg(feature = "zk")]
    let stage2_round_pads = zk_hiding.take_compressed_rounds::<L>(col_bits + ring_bits, 3)?;
    #[cfg(feature = "zk")]
    let stage2_sumcheck_proof_masked = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck_terminal").entered();
        let mut stage2_prover = AkitaStage2Prover::new(
            L::zero(),
            w_evals_compact,
            &r_stage1,
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
        stage2_sumcheck_proof_masked
    };
    #[cfg(not(feature = "zk"))]
    let stage2_sumcheck = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck_terminal").entered();
        let mut stage2_prover = AkitaStage2Prover::new(
            L::zero(),
            w_evals_compact,
            &r_stage1,
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
        stage2_sumcheck
    };

    Ok(
        TerminalLevelProof::new_with_extension_opening_reduction::<D>(
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
        ),
    )
}

pub(in crate::protocol::flow) struct RecursiveExtensionOpeningReduction<L: FieldCore> {
    pub(in crate::protocol::flow) proof: ExtensionOpeningReductionProof<L>,
    pub(in crate::protocol::flow) rho: Vec<L>,
    pub(in crate::protocol::flow) final_claim: L,
    pub(in crate::protocol::flow) final_factor: L,
}

pub(in crate::protocol::flow) fn recursive_witness_base_evals<F>(
    logical_w: &RecursiveWitnessFlat,
) -> Vec<F>
where
    F: FieldCore + FromPrimitiveInt,
{
    logical_w
        .as_i8_digits()
        .iter()
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
    L: ExtField<F> + AkitaSerialize,
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
    let mut base_evals = recursive_witness_base_evals::<F>(logical_w);
    base_evals.resize(padded_len, F::zero());
    let tensor = tensor_partials_from_base_evals::<F, L>(num_vars, &base_evals, opening_point)?;
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
    let packed_witness = tensor_packed_witness_evals::<F, L>(num_vars, &base_evals)?;
    let factor_evals = tensor_equality_factor_evals::<F, L>(tail_point, &eta)?;
    let prover = ExtensionOpeningReductionProver::new(packed_witness, factor_evals)?;
    if prover.input_claim() != true_input_claim {
        return Err(AkitaError::InvalidInput(
            "extension-opening reduction input claim mismatch".to_string(),
        ));
    }
    let mut prover = prover;
    #[cfg(feature = "zk")]
    let reduction_sumcheck =
        ExtensionOpeningReductionSumcheck::new(input_claim, prover.num_rounds());
    #[cfg(not(feature = "zk"))]
    let reduction_sumcheck =
        ExtensionOpeningReductionSumcheck::new(prover.input_claim(), prover.num_rounds());
    #[cfg(not(feature = "zk"))]
    let (sumcheck, result) =
        reduction_sumcheck.prove::<F, _, _>(&mut prover, transcript, |tr| {
            sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND)
        })?;
    #[cfg(feature = "zk")]
    let (sumcheck_proof_masked, result) = reduction_sumcheck.prove_zk::<F, _, _>(
        &mut prover,
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
    let final_factor =
        tensor_equality_factor_eval_at_point::<F, L>(tail_point, &eta, &result.challenges)?;
    if final_factor != final_factor_from_table {
        return Err(AkitaError::InvalidInput(
            "extension-opening reduction transparent factor mismatch".to_string(),
        ));
    }
    check_extension_opening_reduction_output(result.final_claim, final_witness, final_factor)?;
    Ok(RecursiveExtensionOpeningReduction {
        proof: ExtensionOpeningReductionProof {
            partials: proof_partials,
            #[cfg(not(feature = "zk"))]
            sumcheck,
            #[cfg(feature = "zk")]
            sumcheck_proof_masked,
        },
        rho: result.challenges,
        final_claim: result.final_claim,
        final_factor,
    })
}

/// Prove one recursive fold level using already-selected current and next
/// level parameters.
///
/// The caller owns schedule/config selection and passes the next-level
/// commitment policy as a closure. This function owns recursive opening-point
/// reduction, witness folding, public recursive transcript absorbs, recursive
/// quadratic-equation construction, and the folded-level prover mechanics.
///
/// # Errors
///
/// Returns an error if the recursive opening point has the wrong dimension,
/// witness folding or quadratic-equation construction fails, or the folded
/// prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_recursive_fold_with_params<F, L, T, B, const D: usize, CommitW>(
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    witness: &RecursiveWitnessView<'_, F, D>,
    logical_w: &RecursiveWitnessFlat,
    carried_openings: &[RecursiveCarriedOpening<L>],
    extra_carried_sources: &[RecursiveCarriedSource<F>],
    hint: AkitaCommitmentHint<F, D>,
    commitment: &FlatRingVec<F>,
    level: usize,
    level_params: &LevelParams,
    next_log_basis: u32,
    #[cfg(feature = "zk")] mut zk_hiding: ZkHidingProverState<F>,
    commit_w_for_next: CommitW,
) -> Result<ProveLevelOutput<F, L>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasUnreducedOps
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    L: RingSubfieldEncoding<F>
        + FrobeniusExtField<F>
        + HasUnreducedOps
        + FromPrimitiveInt
        + AkitaSerialize,
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

    let alpha = level_params.ring_dimension.trailing_zeros() as usize;
    let mut carried_sources = Vec::with_capacity(extra_carried_sources.len() + 1);
    carried_sources.push(CarriedOpeningSource { commitment });
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
            value: claim.opening,
            basis: claim.basis,
            natural_len: claim.natural_len,
            padded_len: claim.padded_len,
            kind: claim.kind,
        })
        .collect::<Vec<_>>();
    append_carried_opening_batch_to_transcript(&carried_sources, &carried_claims, transcript)?;
    let carried_incidence = carried_opening_incidence_summary(&carried_sources, &carried_claims)?;
    if carried_openings.is_empty()
        || carried_openings.iter().any(|claim| {
            (matches!(claim.kind, CarriedOpeningKind::RecursiveWitness)
                && claim.natural_len != logical_w.len())
                || claim.natural_len > claim.padded_len
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

    let reduction = if <L as ExtField<F>>::EXT_DEGREE == 1 {
        None
    } else {
        let claim = &carried_openings[0];
        Some(prove_recursive_extension_opening_reduction::<F, L, T>(
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

    let (y_rings, w_folded_by_claim) = {
        let _span = tracing::info_span!(
            "evaluate_and_fold",
            level,
            num_ring_elems = witness.num_ring_elems(),
            num_points = prepared_points.len()
        )
        .entered();
        let mut y_rings = Vec::with_capacity(prepared_points.len());
        let mut folded = Vec::with_capacity(prepared_points.len());
        for prepared_point in &prepared_points {
            let (y_ring, w_folded) = evaluate_recursive_witness_at_multiplier_point(
                witness,
                &prepared_point.ring_multiplier_point,
                level_params.block_len,
                level_params.num_blocks,
            )?;
            y_rings.push(y_ring);
            folded.push(w_folded);
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
    let commitment_u = commitment.as_ring_slice::<D>()?;

    let ring_opening_points = prepared_points
        .iter()
        .map(|prepared_point| prepared_point.ring_opening_point.clone())
        .collect::<Vec<_>>();
    let ring_multiplier_points = prepared_points
        .iter()
        .map(|prepared_point| prepared_point.ring_multiplier_point.clone())
        .collect::<Vec<_>>();
    let quad_eq = Box::new(
        QuadraticEquation::<F, { D }>::new_recursive_multipoint_prover(
            backend,
            prepared,
            ring_opening_points,
            ring_multiplier_points,
            witness,
            w_folded_by_claim,
            level_params.clone(),
            hint,
            transcript,
            commitment_u,
            &y_rings,
            MRowLayout::Intermediate,
        )?,
    );

    let extension_opening_reduction = reduction.map(|reduction| reduction.proof);
    prove_fold_level_from_quadratic::<F, L, T, B, D, _>(
        expanded,
        backend,
        prepared,
        transcript,
        commitment_u,
        level,
        level_params,
        next_log_basis,
        quad_eq,
        extension_opening_reduction,
        y_rings,
        #[cfg(feature = "zk")]
        y_rings_masked,
        #[cfg(feature = "zk")]
        zk_hiding,
        commit_w_for_next,
    )
}

/// Mirror of [`prove_recursive_fold_with_params`] producing a
/// [`TerminalLevelProof`] instead of an intermediate
/// [`AkitaLevelProof`] + next-witness commitment pair.
///
/// All recursive-opening, witness folding, and quadratic-equation setup is
/// identical to the intermediate path. The two differ only inside the inner
/// fold proof (see [`prove_terminal_fold_level_from_quadratic`]).
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
    witness: &RecursiveWitnessView<'_, F, D>,
    logical_w: &RecursiveWitnessFlat,
    carried_openings: &[RecursiveCarriedOpening<L>],
    extra_carried_sources: &[RecursiveCarriedSource<F>],
    hint: AkitaCommitmentHint<F, D>,
    commitment: &FlatRingVec<F>,
    level: usize,
    level_params: &LevelParams,
    final_log_basis: u32,
    #[cfg(feature = "zk")] zk_hiding: &mut ZkHidingProverState<F>,
) -> Result<TerminalLevelProof<F, L>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasUnreducedOps
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    L: RingSubfieldEncoding<F>
        + FrobeniusExtField<F>
        + HasUnreducedOps
        + FromPrimitiveInt
        + AkitaSerialize,
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

    let alpha = level_params.ring_dimension.trailing_zeros() as usize;
    let commitment_u = commitment.as_ring_slice::<D>()?;
    let mut carried_sources = Vec::with_capacity(extra_carried_sources.len() + 1);
    carried_sources.push(CarriedOpeningSource { commitment });
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
            value: claim.opening,
            basis: claim.basis,
            natural_len: claim.natural_len,
            padded_len: claim.padded_len,
            kind: claim.kind,
        })
        .collect::<Vec<_>>();
    append_carried_opening_batch_to_transcript(&carried_sources, &carried_claims, transcript)?;
    let carried_incidence = carried_opening_incidence_summary(&carried_sources, &carried_claims)?;
    if carried_openings.is_empty()
        || carried_openings.iter().any(|claim| {
            (matches!(claim.kind, CarriedOpeningKind::RecursiveWitness)
                && claim.natural_len != logical_w.len())
                || claim.natural_len > claim.padded_len
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

    let reduction = if <L as ExtField<F>>::EXT_DEGREE == 1 {
        None
    } else {
        let claim = &carried_openings[0];
        Some(prove_recursive_extension_opening_reduction::<F, L, T>(
            logical_w,
            &claim.opening_point,
            claim.opening,
            transcript,
            #[cfg(feature = "zk")]
            zk_hiding,
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

    let (y_rings, w_folded_by_claim) = {
        let _span = tracing::info_span!(
            "evaluate_and_fold",
            level,
            num_ring_elems = witness.num_ring_elems(),
            num_points = prepared_points.len()
        )
        .entered();
        let mut y_rings = Vec::with_capacity(prepared_points.len());
        let mut folded = Vec::with_capacity(prepared_points.len());
        for prepared_point in &prepared_points {
            let (y_ring, w_folded) = evaluate_recursive_witness_at_multiplier_point(
                witness,
                &prepared_point.ring_multiplier_point,
                level_params.block_len,
                level_params.num_blocks,
            )?;
            y_rings.push(y_ring);
            folded.push(w_folded);
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
    #[cfg(not(feature = "zk"))]
    for y_ring in &y_rings {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }
    #[cfg(feature = "zk")]
    for y_ring in &y_rings_masked {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }
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

    let ring_opening_points = prepared_points
        .iter()
        .map(|prepared_point| prepared_point.ring_opening_point.clone())
        .collect::<Vec<_>>();
    let ring_multiplier_points = prepared_points
        .iter()
        .map(|prepared_point| prepared_point.ring_multiplier_point.clone())
        .collect::<Vec<_>>();
    let quad_eq = Box::new(
        QuadraticEquation::<F, { D }>::new_recursive_multipoint_prover(
            backend,
            prepared,
            ring_opening_points,
            ring_multiplier_points,
            witness,
            w_folded_by_claim,
            level_params.clone(),
            hint,
            transcript,
            commitment_u,
            &y_rings,
            MRowLayout::Terminal,
        )?,
    );

    let extension_opening_reduction = reduction.map(|reduction| reduction.proof);
    prove_terminal_fold_level_from_quadratic::<F, L, T, B, D>(
        expanded,
        backend,
        prepared,
        transcript,
        commitment_u,
        level,
        level_params,
        final_log_basis,
        quad_eq,
        extension_opening_reduction,
        y_rings,
        #[cfg(feature = "zk")]
        y_rings_masked,
        #[cfg(feature = "zk")]
        zk_hiding,
    )
}

/// Prove one recursive fold level from D-erased recursive state using
/// caller-supplied config policy.
///
/// The prover crate owns the state unpacking, typed recursive witness view,
/// typed hint conversion, opening-point handoff, and fold proof mechanics.
/// The caller supplies only the current-witness layout policy and the
/// next-level recursive commitment policy.
///
/// # Errors
///
/// Returns an error if the current witness cannot be viewed at `D`, the hint
/// cannot be typed at `D`, layout selection fails, or recursive proving fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_recursive_level_with_policy<F, L, T, B, const D: usize, CurrentLayout, CommitW>(
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    current_state: RecursiveProverState<F, L>,
    level: usize,
    level_params: &LevelParams,
    next_log_basis: u32,
    current_layout: CurrentLayout,
    commit_w_for_next: CommitW,
) -> Result<ProveLevelOutput<F, L>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasUnreducedOps
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    L: RingSubfieldEncoding<F>
        + FrobeniusExtField<F>
        + HasUnreducedOps
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
    B: ProverComputeBackend<F>,
    CurrentLayout: FnOnce(&LevelParams, usize) -> Result<LevelParams, AkitaError>,
    CommitW: FnOnce(&RecursiveWitnessFlat) -> Result<NextWitnessCommitment<F>, AkitaError>,
{
    let _setup_span = tracing::info_span!("inter_level_setup", level).entered();
    let current_w_len = current_state.recursive_witness_len()?;

    let RecursiveProverState {
        w: current_w,
        logical_w,
        commitment,
        hint,
        log_basis: _,
        carried_openings,
        extra_carried_sources,
        #[cfg(feature = "zk")]
        zk_hiding,
    } = current_state;
    let w_lp = current_layout(level_params, current_w_len)?;
    let w_view = current_w.view::<F, D>()?;
    let logical_w = logical_w.as_ref().unwrap_or(&current_w);
    let typed_hint: AkitaCommitmentHint<F, D> = hint.to_typed::<D>()?;
    drop(_setup_span);

    prove_recursive_fold_with_params::<F, L, T, B, D, _>(
        expanded,
        backend,
        prepared,
        transcript,
        &w_view,
        logical_w,
        &carried_openings,
        &extra_carried_sources,
        typed_hint,
        &commitment,
        level,
        &w_lp,
        next_log_basis,
        #[cfg(feature = "zk")]
        zk_hiding,
        commit_w_for_next,
    )
}

/// Terminal-fold analogue of [`prove_recursive_level_with_policy`].
///
/// Same input shape minus the next-witness commitment policy; the terminal
/// fold ships `final_witness` in cleartext (packed digits) instead of
/// committing.
///
/// # Errors
///
/// Returns an error if the policy callback fails to produce the current
/// level's layout or the underlying terminal fold prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_terminal_recursive_level_with_policy<F, L, T, B, const D: usize, CurrentLayout>(
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    current_state: &mut RecursiveProverState<F, L>,
    level: usize,
    level_params: &LevelParams,
    final_log_basis: u32,
    current_layout: CurrentLayout,
) -> Result<TerminalLevelProof<F, L>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasUnreducedOps
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    L: RingSubfieldEncoding<F>
        + FrobeniusExtField<F>
        + HasUnreducedOps
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
    B: ProverComputeBackend<F>,
    CurrentLayout: FnOnce(&LevelParams, usize) -> Result<LevelParams, AkitaError>,
{
    let _setup_span = tracing::info_span!("inter_level_setup_terminal", level).entered();

    let current_w = &current_state.w;
    let current_w_len = current_state.recursive_witness_len()?;
    let w_lp = current_layout(level_params, current_w_len)?;
    let w_view = current_w.view::<F, D>()?;
    let logical_w = current_state.logical_w.as_ref().unwrap_or(current_w);
    let typed_hint: AkitaCommitmentHint<F, D> = current_state.hint.to_typed::<D>()?;
    drop(_setup_span);

    prove_terminal_recursive_fold_with_params::<F, L, T, B, D>(
        expanded,
        backend,
        prepared,
        transcript,
        &w_view,
        logical_w,
        &current_state.carried_openings,
        &current_state.extra_carried_sources,
        typed_hint,
        &current_state.commitment,
        level,
        &w_lp,
        final_log_basis,
        #[cfg(feature = "zk")]
        &mut current_state.zk_hiding,
    )
}
