use super::*;
use cfg_if::cfg_if;

#[cfg(not(feature = "zk"))]
use crate::protocol::ring_switch::RingSwitchTerminalArtifacts;
#[cfg(not(feature = "zk"))]
use akita_types::build_segment_typed_witness;
#[cfg(not(feature = "zk"))]
use akita_types::pad_segment_typed_z_payload;
#[cfg(not(feature = "zk"))]
use akita_types::schedule_terminal_direct_witness_shape;
#[cfg(not(feature = "zk"))]
use akita_types::CleartextWitnessShape;

/// Prover state carried between suffix fold levels.
pub struct SuffixProverState<F: FieldCore, L: FieldCore> {
    /// Current committed suffix witness representation.
    pub w: RecursiveWitnessFlat,
    /// Logical suffix witness when it differs from the committed representation.
    pub logical_w: Option<RecursiveWitnessFlat>,
    /// Current suffix witness commitment.
    pub commitment: FlatRingVec<F>,
    /// D-erased suffix commitment hint cache.
    pub hint: RecursiveCommitmentHintCache<F>,
    /// Current digit basis, as `log2(b)`.
    pub log_basis: u32,
    /// Sumcheck challenges that become the next suffix opening point.
    pub sumcheck_challenges: Vec<L>,
    /// Claimed logical opening of `logical_w` at `sumcheck_challenges`.
    pub opening: L,
    /// Transcript-visible masked handle for `opening`.
    #[cfg(feature = "zk")]
    pub opening_public: L,
    /// Proof-level ZK hiding material fixed at batched-prove startup.
    #[cfg(feature = "zk")]
    pub zk_hiding: ZkHidingProverState<F>,
}

impl<F: FieldCore, L: FieldCore> SuffixProverState<F, L> {
    /// Logical witness represented by the carried opening claim.
    #[inline]
    pub fn logical_w(&self) -> &RecursiveWitnessFlat {
        self.logical_w.as_ref().unwrap_or(&self.w)
    }
}

pub(in crate::protocol::flow) struct PreparedFold<F: FieldCore, L: FieldCore, const D: usize> {
    pub(in crate::protocol::flow) commitment: FlatRingVec<F>,
    pub(in crate::protocol::flow) instance: RingRelationInstance<F, D>,
    pub(in crate::protocol::flow) witness: RingRelationWitness<F, D>,
    pub(in crate::protocol::flow) extension_opening_reduction:
        Option<ExtensionOpeningReductionProof<L>>,
    pub(in crate::protocol::flow) trace_eval_target: L,
    #[cfg(feature = "zk")]
    pub(in crate::protocol::flow) trace_eval_target_public: L,
    pub(in crate::protocol::flow) trace_prepared_point: Option<PreparedOpeningPoint<F, L, D>>,
    pub(in crate::protocol::flow) trace_claim_scales: Option<Vec<L>>,
    pub(in crate::protocol::flow) trace_scale: L,
    #[cfg(feature = "zk")]
    pub(in crate::protocol::flow) zk_hiding: ZkHidingProverState<F>,
    pub(in crate::protocol::flow) row_coefficients: Option<Vec<L>>,
}

#[cfg(not(feature = "zk"))]
pub(in crate::protocol::flow) type TerminalFoldResult<F, L> = TerminalLevelProof<F, L>;
#[cfg(feature = "zk")]
pub(in crate::protocol::flow) type TerminalFoldResult<F, L> =
    (TerminalLevelProof<F, L>, ZkHidingProverState<F>);

pub(in crate::protocol::flow) enum FoldProveOutput<F: FieldCore, L: FieldCore> {
    Intermediate(Box<ProveLevelOutput<F, L>>),
    Terminal(Box<TerminalFoldResult<F, L>>),
}

impl<F: FieldCore, L: FieldCore> FoldProveOutput<F, L> {
    pub(in crate::protocol::flow) fn get_intermediate(
        self,
    ) -> Result<ProveLevelOutput<F, L>, AkitaError> {
        match self {
            Self::Intermediate(out) => Ok(*out),
            Self::Terminal(_) => Err(AkitaError::InvalidInput(
                "intermediate fold unexpectedly returned terminal proof".to_string(),
            )),
        }
    }

    pub(in crate::protocol::flow) fn get_terminal(
        self,
    ) -> Result<TerminalFoldResult<F, L>, AkitaError> {
        match self {
            Self::Terminal(terminal) => Ok(*terminal),
            Self::Intermediate(_) => Err(AkitaError::InvalidInput(
                "terminal fold unexpectedly returned intermediate proof".to_string(),
            )),
        }
    }
}

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
    transcript: &mut T,
    starting_state: SuffixProverState<Cfg::Field, Cfg::ExtField>,
    schedule: &Schedule,
    setup_contribution_mode: SetupContributionMode,
) -> Result<RecursiveSuffixOutcome<Cfg::Field, Cfg::ExtField>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
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
    let mut intermediate_levels = Vec::new();
    let mut current_state = starting_state;
    let mut level = 1usize;

    #[cfg(not(feature = "zk"))]
    let terminal_direct_witness_shape = schedule_terminal_direct_witness_shape(schedule)?;
    let terminal_result = loop {
        let scheduled = schedule.get_execution_schedule(level)?;
        scheduled.validate_current_w_len(current_state.w.len())?;
        let level_params = &scheduled.params;
        let level_d = level_params.ring_dimension;
        let is_terminal_level = scheduled.is_terminal;
        let m_row_layout = if is_terminal_level {
            MRowLayout::WithoutDBlock
        } else {
            MRowLayout::WithDBlock
        };
        let out = if level_d == D {
            let prepared_fold = prepare_fold_data::<Cfg::Field, Cfg::ExtField, T, B, D>(
                backend,
                prepared,
                transcript,
                current_state,
                level,
                level_params,
                m_row_layout,
            )?;
            prove_fold::<Cfg::Field, Cfg::ExtField, T, B, Cfg, D>(
                expanded,
                prefix_slots,
                backend,
                prepared,
                transcript,
                level,
                &scheduled,
                prepared_fold,
                setup_contribution_mode,
                is_terminal_level,
                #[cfg(not(feature = "zk"))]
                if is_terminal_level {
                    Some(terminal_direct_witness_shape)
                } else {
                    None
                },
            )
        } else {
            dispatch_ring_dim_result!(level_d, |D_LEVEL| {
                let level_prepared = backend.prepare_expanded::<D_LEVEL>(expanded.clone())?;
                let level_prefix_slots = SetupPrefixProverRegistry::new();
                let prepared_fold =
                    prepare_fold_data::<Cfg::Field, Cfg::ExtField, T, B, { D_LEVEL }>(
                        backend,
                        &level_prepared,
                        transcript,
                        current_state,
                        level,
                        level_params,
                        m_row_layout,
                    )?;
                prove_fold::<Cfg::Field, Cfg::ExtField, T, B, Cfg, { D_LEVEL }>(
                    expanded,
                    &level_prefix_slots,
                    backend,
                    &level_prepared,
                    transcript,
                    level,
                    &scheduled,
                    prepared_fold,
                    setup_contribution_mode,
                    is_terminal_level,
                    #[cfg(not(feature = "zk"))]
                    if is_terminal_level {
                        Some(terminal_direct_witness_shape)
                    } else {
                        None
                    },
                )
            })
        }?;
        if is_terminal_level {
            break out.get_terminal()?;
        }

        let out = out.get_intermediate()?;
        intermediate_levels.push(out.level_proof);
        current_state = out.next_state;
        level += 1;
    };
    #[cfg(not(feature = "zk"))]
    let terminal = terminal_result;
    #[cfg(feature = "zk")]
    let (terminal, zk_hiding) = terminal_result;

    let mut steps = intermediate_levels;
    let final_w_len = terminal.final_witness().num_elems();
    steps.push(AkitaLevelProof::Terminal {
        extension_opening_reduction: terminal.extension_opening_reduction,
        stage2: terminal.stage2,
        final_w_len,
    });

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
pub(in crate::protocol::flow) fn prove_fold<F, L, T, B, Cfg, const D: usize>(
    expanded: &Arc<AkitaExpandedSetup<F>>,
    prefix_slots: &SetupPrefixProverRegistry<F, D>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    level: usize,
    scheduled: &ExecutionSchedule,
    prepared_fold: PreparedFold<F, L, D>,
    setup_contribution_mode: SetupContributionMode,
    is_terminal_fold: bool,
    #[cfg(not(feature = "zk"))]
    terminal_direct_witness_shape: Option<&CleartextWitnessShape>,
) -> Result<FoldProveOutput<F, L>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField
        + AkitaSerialize,
    L: ExtField<F>
        + FpExtEncoding<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
    B: ProverComputeBackend<F>,
    Cfg: CommitmentConfig<Field = F, ExtField = L>,
{
    #[cfg(feature = "zk")]
    let mut zk_hiding = prepared_fold.zk_hiding;
    let lp = &scheduled.params;
    let commitment_u = prepared_fold.commitment.as_ring_slice::<D>()?;
    let build_output = ring_switch_build_w::<F, B, D>(
        &prepared_fold.instance,
        prepared_fold.witness,
        backend,
        prepared,
        lp,
        is_terminal_fold,
    )?;
    let logical_w = build_output.w;
    scheduled.validate_next_w_len(logical_w.len())?;
    let next_commitment = if is_terminal_fold {
        None
    } else {
        let _span = tracing::info_span!("commit_w_level", level).entered();
        Some(crate::commit_next_w::<Cfg, B, D>(
            &scheduled.next_params,
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
            Some(scheduled.next_params.log_basis)
        } else {
            None
        },
        #[cfg(not(feature = "zk"))]
        build_output.terminal_artifacts,
        #[cfg(not(feature = "zk"))]
        terminal_direct_witness_shape,
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
        prepared_fold.row_coefficients.as_deref(),
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
    let trace_coeff = {
        let trace_gamma = if is_terminal_fold {
            sample_ext_challenge::<F, L, T>(transcript, CHALLENGE_SUMCHECK_BATCH)
        } else {
            batching_coeff
        };
        stage2_trace_coeff(batching_coeff, trace_gamma, is_terminal_fold)
    };
    let trace_opening_claim = trace_coeff * prepared_fold.trace_eval_target;
    #[cfg(feature = "zk")]
    let trace_eval_target_public_claim = trace_coeff * prepared_fold.trace_eval_target_public;
    ensure_trace_stage2_supported(L::EXT_DEGREE)?;
    let trace_compact = if let Some(row_coefficients) = prepared_fold.row_coefficients.as_ref() {
        Some(build_root_stage2_trace_table::<F, L, D>(
            lp,
            &prepared_fold.instance,
            prepared_fold
                .trace_prepared_point
                .as_ref()
                .ok_or(AkitaError::InvalidProof)?,
            row_coefficients,
            prepared_fold.trace_claim_scales.as_deref(),
            trace_coeff,
            rs.col_bits,
            rs.ring_bits,
            rs.live_x_cols,
        )?)
    } else if let Some(prepared) = prepared_fold.trace_prepared_point.as_ref() {
        Some(build_recursive_stage2_trace_table::<F, L, D>(
            lp,
            &prepared_fold.instance,
            prepared,
            prepared_fold.trace_scale,
            trace_coeff,
            rs.col_bits,
            rs.ring_bits,
            rs.live_x_cols,
        )?)
    } else {
        None
    };
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
            &scheduled.next_params,
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
        let level_proof = AkitaLevelProof::Intermediate {
            extension_opening_reduction: prepared_fold.extension_opening_reduction,
            v: FlatRingVec::from_ring_elems(&prepared_fold.instance.v).into_compact(),
            stage1: stage1_proof,
            stage2: AkitaStage2Proof::Intermediate(AkitaIntermediateStage2Proof {
                #[cfg(not(feature = "zk"))]
                sumcheck_proof: stage2_sumcheck_proof,
                #[cfg(feature = "zk")]
                sumcheck_proof_masked: stage2_sumcheck_proof,
                next_w_commitment: w_commitment_proof.into_compact(),
                #[cfg(not(feature = "zk"))]
                next_w_eval: proof_w_eval,
                #[cfg(feature = "zk")]
                next_w_eval_masked: proof_w_eval,
            }),
            stage3_sumcheck_proof,
        };

        let (committed_witness, logical_w) = match packed_witness {
            Some(packed_witness) => (packed_witness, Some(logical_w)),
            None => (logical_w, None),
        };

        Ok(FoldProveOutput::Intermediate(Box::new(ProveLevelOutput {
            level_proof,
            next_state: SuffixProverState {
                w: committed_witness,
                logical_w,
                commitment: committed_commitment,
                hint: committed_hint,
                log_basis: scheduled.next_params.log_basis,
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
pub(in crate::protocol::flow) fn bind_next_witness_for_ring_switch<F, T, const D: usize>(
    transcript: &mut T,
    is_terminal_fold: bool,
    lp: &LevelParams,
    instance: &RingRelationInstance<F, D>,
    logical_w: &RecursiveWitnessFlat,
    next_commitment: Option<NextWitnessCommitment<F>>,
    final_log_basis: Option<u32>,
    #[cfg(not(feature = "zk"))] terminal_artifacts: Option<RingSwitchTerminalArtifacts<F, D>>,
    #[cfg(not(feature = "zk"))] terminal_direct_witness_shape: Option<&CleartextWitnessShape>,
) -> Result<BoundNextWitness<F>, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField + AkitaSerialize,
    T: Transcript<F>,
{
    if is_terminal_fold {
        #[cfg(feature = "zk")]
        let final_log_basis = final_log_basis.ok_or_else(|| {
            AkitaError::InvalidInput("terminal fold missing final witness basis".to_string())
        })?;
        #[cfg(not(feature = "zk"))]
        final_log_basis.ok_or_else(|| {
            AkitaError::InvalidInput("terminal fold missing final witness basis".to_string())
        })?;
        #[cfg(not(feature = "zk"))]
        {
            if let Some(artifacts) = terminal_artifacts {
                if artifacts.u_concat_planes != 0 {
                    return Err(AkitaError::InvalidInput(
                        "segment-typed terminal witness does not support tiered u_concat"
                            .to_string(),
                    ));
                }
                let num_claims = instance.opening_batch().num_claims();
                let num_commitment_groups = instance
                    .opening_batch()
                    .num_polys_per_commitment_group()
                    .len();
                let mut segment = build_segment_typed_witness::<D, F>(
                    &artifacts.e_folded,
                    &artifacts.recomposed_inner_rows,
                    &artifacts.z_folded_centered,
                    &artifacts.r,
                    lp,
                    num_claims,
                    1,
                    num_claims,
                    num_commitment_groups,
                )?;
                let CleartextWitnessShape::SegmentTyped(scheduled_shape) =
                    terminal_direct_witness_shape.ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "terminal fold missing scheduled segment-typed witness shape"
                                .to_string(),
                        )
                    })?
                else {
                    return Err(AkitaError::InvalidSetup(
                        "terminal fold expected segment-typed witness shape".to_string(),
                    ));
                };
                if segment.layout != scheduled_shape.layout {
                    return Err(AkitaError::InvalidSetup(
                        "segment-typed witness layout does not match schedule".to_string(),
                    ));
                }
                pad_segment_typed_z_payload(&mut segment, scheduled_shape.z_payload_bytes)?;
                let expanded = segment.layout.logical_num_elems;
                let digits = akita_types::expand_segment_typed_to_i8_digits::<D, F>(
                    &segment,
                    lp,
                    num_claims,
                    1,
                    num_claims,
                    num_commitment_groups,
                )?;
                if digits.len() != expanded || digits.as_slice() != logical_w.as_i8_digits() {
                    return Err(AkitaError::InvalidInput(
                        "segment-typed final witness does not match ring-switch witness"
                            .to_string(),
                    ));
                }
                let parts = segment.terminal_transcript_parts()?;
                transcript.append_bytes(ABSORB_TERMINAL_W_REMAINDER, &parts.remainder);
                return Ok((None, Some(CleartextWitnessProof::SegmentTyped(segment))));
            }
            return Err(AkitaError::InvalidSetup(
                "terminal fold missing segment-typed witness artifacts".to_string(),
            ));
        }
        #[cfg(feature = "zk")]
        {
            let final_witness =
                CleartextWitnessProof::PackedDigits(PackedDigits::from_i8_digits_with_min_bits(
                    logical_w.as_i8_digits(),
                    final_log_basis,
                ));
            let terminal_layout = terminal_witness_segment_layout(
                lp,
                instance.opening_batch().num_claims(),
                instance.opening_batch().num_claims(),
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
pub(in crate::protocol::flow) fn prove_stage1<F, L, T>(
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
pub(in crate::protocol::flow) fn prove_stage3<F, L, T, const D: usize>(
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
    L: FpExtEncoding<F> + FromPrimitiveInt + AkitaSerialize,
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
    cfg_if! {
        if #[cfg(feature = "zk")] {
            let (sumcheck_proof, rho) = prover.prove_zk::<F, T, _>(
                input_claim,
                transcript,
                |tr| sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND),
                sumcheck_pads,
            )?;
            let final_claim_public =
                masked_sumcheck_final_claim(input_claim, &sumcheck_proof, &rho)?;
        } else {
            let (sumcheck_proof, rho, final_claim) = prover.prove::<F, T, _>(transcript, |tr| {
                sample_ext_challenge::<F, L, T>(tr, CHALLENGE_SUMCHECK_ROUND)
            })?;
        }
    }
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
            sumcheck: sumcheck_proof,
            #[cfg(feature = "zk")]
            sumcheck_proof_masked: sumcheck_proof,
        },
        rho,
        final_claim,
        #[cfg(feature = "zk")]
        final_claim_public,
        final_factor,
    })
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
    folded_rings: &[CyclotomicRing<F, D>],
    prepared_point: &PreparedOpeningPoint<F, L, D>,
    expected_opening: L,
) -> Result<(L, L), AkitaError>
where
    F: FieldCore + FromPrimitiveInt + Invertible,
    L: ExtField<F> + FpExtEncoding<F>,
{
    #[cfg(not(feature = "zk"))]
    {
        match reduction {
            Some(reduction) => Ok((reduction.final_claim, reduction.final_factor)),
            None => {
                let folded_ring = folded_rings.first().ok_or(AkitaError::InvalidProof)?;
                let opening = recover_ring_subfield_inner_product::<F, L, D>(
                    folded_ring,
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
        let folded_ring = folded_rings.first().ok_or(AkitaError::InvalidProof)?;
        let internal_claim = recover_ring_subfield_inner_product::<F, L, D>(
            folded_ring,
            &prepared_point.packed_inner_point,
        )?;
        match reduction {
            Some(reduction) => {
                check_extension_opening_reduction_output(
                    reduction.final_claim,
                    internal_claim,
                    reduction.final_factor,
                )?;
                Ok((reduction.final_claim, reduction.final_factor))
            }
            None => {
                if internal_claim != expected_opening {
                    return Err(AkitaError::InvalidInput(
                        "recursive opening does not match carried claim".to_string(),
                    ));
                }
                Ok((internal_claim, L::one()))
            }
        }
    }
}

fn validate_recursive_opening_block_count<F, L, const D: usize>(
    prepared_point: &PreparedOpeningPoint<F, L, D>,
    level_params: &LevelParams,
) -> Result<(), AkitaError>
where
    F: FieldCore,
    L: FieldCore,
{
    let actual = prepared_point.ring_opening_point.b.len();
    if actual != level_params.num_blocks {
        return Err(AkitaError::InvalidInput(format!(
            "recursive opening block count {actual} does not match scheduled num_blocks {}",
            level_params.num_blocks
        )));
    }
    Ok(())
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
    current_state: SuffixProverState<F, L>,
    level: usize,
    level_params: &LevelParams,
    m_row_layout: MRowLayout,
) -> Result<PreparedFold<F, L, D>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    L: FpExtEncoding<F>
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

    let alpha = level_params.ring_dimension.trailing_zeros() as usize;
    let (reduction, protocol_point) = if <L as ExtField<F>>::EXT_DEGREE == 1 {
        (None, opening_point.to_vec())
    } else {
        let reduction = prove_extension_opening_reduction::<F, L, T>(
            logical_w,
            opening_point,
            current_state.opening,
            transcript,
            #[cfg(feature = "zk")]
            &mut zk_hiding,
        )?;
        let protocol_point = ring_subfield_packed_extension_opening_point::<F, L, D>(
            reduction.rho.len(),
            &reduction.rho,
        )?;
        (Some(reduction), protocol_point)
    };
    let prepared_point = prepare_opening_point::<F, L, D>(
        &protocol_point,
        BasisMode::Lagrange,
        level_params,
        alpha,
        BlockOrder::ColumnMajor,
    )?;
    validate_recursive_opening_block_count(&prepared_point, level_params)?;
    let recursive_polys = [&witness_view];

    let (folded_rings, e_folded_by_claim) = evaluate_claims_at_prepared_point(
        &recursive_polys,
        &prepared_point,
        level_params.block_len,
    )?;
    for pt in &prepared_point.padded_point {
        append_ext_field::<F, L, T>(transcript, ABSORB_EVALUATION_CLAIMS, pt);
    }

    let (trace_eval_target, trace_scale) = compute_trace_target::<F, L, D>(
        &reduction,
        &folded_rings,
        &prepared_point,
        current_state.opening,
    )?;
    #[cfg(feature = "zk")]
    let trace_eval_target_public = match &reduction {
        Some(reduction) => reduction.final_claim_public,
        None => current_state.opening_public,
    };
    let commitment_u = current_state.commitment.as_ring_slice::<D>()?;

    let recursive_num_vars = level_params.recursive_opening_num_vars()?;
    let opening_batch = OpeningBatch::same_point(recursive_num_vars, 1)?;
    let recursive_commitment = RingCommitment {
        u: commitment_u.to_vec(),
    };
    let row_coefficient_rings = vec![CyclotomicRing::one(); opening_batch.num_claims()];
    let (instance, witness) = RingRelationProver::new::<F, D, _, _, _>(
        backend,
        prepared,
        prepared_point.ring_opening_point.clone(),
        prepared_point.ring_multiplier_point.clone(),
        &recursive_polys,
        e_folded_by_claim,
        opening_batch,
        level_params.clone(),
        vec![typed_hint],
        transcript,
        std::slice::from_ref(&recursive_commitment),
        row_coefficient_rings,
        m_row_layout,
    )?;
    Ok(PreparedFold {
        commitment: current_state.commitment,
        instance,
        witness,
        extension_opening_reduction: reduction.map(|reduction| reduction.proof),
        trace_eval_target,
        trace_scale,
        trace_prepared_point: Some(prepared_point),
        trace_claim_scales: None,
        #[cfg(feature = "zk")]
        trace_eval_target_public,
        #[cfg(feature = "zk")]
        zk_hiding,
        row_coefficients: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Fp32;
    use akita_types::{RingOpeningPoint, SisModulusFamily};

    type TestF = Fp32<251>;
    const D: usize = 4;

    fn level_params_with_four_blocks() -> LevelParams {
        LevelParams::params_only(
            SisModulusFamily::Q32,
            D,
            2,
            1,
            1,
            1,
            SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![1],
            },
        )
        .with_decomp(1, 2, 1, 1, 0)
        .expect("synthetic level params")
    }

    #[test]
    fn recursive_opening_block_count_mismatch_is_rejected() {
        let level_params = level_params_with_four_blocks();
        assert_eq!(level_params.num_blocks, 4);
        let prepared_point: PreparedOpeningPoint<TestF, TestF, D> = PreparedOpeningPoint {
            padded_point: Vec::new(),
            ring_opening_point: RingOpeningPoint {
                a: vec![TestF::one()],
                b: vec![TestF::one(), TestF::zero()],
            },
            ring_multiplier_point: RingMultiplierOpeningPoint::from_base(&RingOpeningPoint {
                a: vec![TestF::one()],
                b: vec![TestF::one(), TestF::zero()],
            }),
            packed_inner_point: CyclotomicRing::<TestF, D>::zero(),
        };

        let err = validate_recursive_opening_block_count(&prepared_point, &level_params)
            .expect_err("mismatched recursive opening block count should reject");
        assert!(
            matches!(err, AkitaError::InvalidInput(message) if message.contains("scheduled num_blocks"))
        );
    }
}
