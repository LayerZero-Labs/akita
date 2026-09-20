use super::*;
use jolt_field::AdditiveGroup;
/// Prover state carried between suffix fold levels.
pub struct SuffixProverState<F: Field, E: Field, MaterialHandle, WitnessHandle> {
    /// Current committed suffix witness representation.
    pub(crate) witness_handle: WitnessHandle,
    /// Transcript-bound public state for the current suffix witness.
    pub(crate) binding: NextWitnessState<F>,
    /// Consumer-owned commitment material transported without interpretation.
    pub(crate) commitment_material: MaterialHandle,
    /// Sumcheck challenges that become the next suffix opening point.
    pub(crate) sumcheck_challenges: Vec<E>,
    /// Claimed opening of the logical witness at `sumcheck_challenges`.
    pub(crate) opening: E,
    /// Optional setup-prefix opening carried from the previous stage-3 proof.
    pub(crate) setup_prefix_opening: Option<(Vec<E>, E)>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prove_suffix<Cfg, T, B>(
    expanded: &akita_types::AkitaSetupDescriptor,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field, B::CommitmentHandle>,
    backend: &B,
    transcript: &mut T,
    mut current_state: SuffixProverState<
        Cfg::Field,
        Cfg::ExtField,
        B::CommitmentMaterialHandle,
        B::WitnessHandle,
    >,
    schedule: &FoldSchedule,
    session: &B::ProofSessionHandle,
) -> Result<RecursiveSuffixOutcome<Cfg::Field, Cfg::ExtField>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: CanonicalEncoding + AkitaSerialize + Ring + Unreduced + PseudoMersenne + 'static,
    <Cfg::Field as Unreduced>::Wide: From<Cfg::Field> + AdditiveGroup,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + ExtField<Cfg::Field>
        + Unreduced
        + Fold
        + Ring
        + MulBaseUnreduced<Cfg::Field>
        + AkitaSerialize
        + 'static,
    T: akita_types::ProverTranscriptGrinding<Cfg::Field>,
    B: ProverBackend<Cfg::Field, Cfg::ExtField>,
{
    let mut recursive_folds = Vec::with_capacity(schedule.recursive_folds.len());
    for (index, step) in schedule.recursive_folds.iter().enumerate() {
        let level = index + 1;
        if current_state.witness_handle.manifest().logical_len() != step.input_witness_len {
            return Err(AkitaError::InvalidProof);
        }
        let (next_params, next_binding) = schedule.recursive_folds.get(index + 1).map_or(
            (
                fold::FoldSuccessorParams::Terminal(&schedule.terminal),
                akita_types::NextWitnessBindingPolicy::TerminalInnerState,
            ),
            |next| {
                (
                    fold::FoldSuccessorParams::Recursive(next),
                    akita_types::NextWitnessBindingPolicy::OuterPayload,
                )
            },
        );
        let prepared = prepare_suffix::<Cfg::Field, Cfg::ExtField, T, B>(
            backend,
            prefix_slots,
            transcript,
            current_state,
            level,
            &step.params,
            session,
            step.output_witness_len,
            next_params.inner_ring_dimension(),
        )?;
        let output = prove_fold::<Cfg::Field, Cfg::ExtField, T, B>(
            expanded,
            prefix_slots,
            backend,
            session,
            transcript,
            level,
            &step.params,
            next_params,
            step.output_witness_len,
            next_binding,
            prepared,
        )?;
        recursive_folds.push(output.level_proof);
        current_state = output.next_state;
    }
    if current_state.witness_handle.manifest().logical_len() != schedule.terminal.input_witness_len
    {
        return Err(AkitaError::InvalidProof);
    }
    let terminal = prove_terminal_suffix::<Cfg::Field, Cfg::ExtField, T, B>(
        backend,
        transcript,
        schedule.recursive_folds.len() + 1,
        current_state,
        &schedule.terminal,
        session,
    )?;
    Ok(RecursiveSuffixOutcome {
        recursive_folds,
        terminal,
        num_levels: schedule.num_fold_levels(),
    })
}

#[allow(clippy::too_many_arguments)]
fn prove_terminal_suffix<F, E, T, B>(
    backend: &B,
    transcript: &mut T,
    level: usize,
    current_state: SuffixProverState<F, E, B::CommitmentMaterialHandle, B::WitnessHandle>,
    scheduled: &TerminalFoldParams,
    session: &B::ProofSessionHandle,
) -> Result<TerminalLevelProof<F, E>, AkitaError>
where
    F: Field + CanonicalEncoding + AkitaSerialize + Ring + Unreduced + PseudoMersenne + 'static,
    <F as Unreduced>::Wide: From<F> + AdditiveGroup,
    E: FpExtEncoding<F>
        + ExtField<F>
        + Unreduced
        + Fold
        + Ring
        + MulBaseUnreduced<F>
        + AkitaSerialize
        + 'static,
    T: akita_types::ProverTranscriptGrinding<F>,
    B: ProverBackend<F, E>,
{
    let SuffixProverState {
        witness_handle,
        binding,
        commitment_material,
        sumcheck_challenges,
        opening,
        setup_prefix_opening,
        ..
    } = current_state;
    if setup_prefix_opening.is_some() {
        return Err(AkitaError::InvalidSetup(
            "terminal fold cannot receive a setup-prefix opening".into(),
        ));
    }
    match binding {
        NextWitnessState::TerminalInnerState => {}
        NextWitnessState::OuterPayload(_) => return Err(AkitaError::InvalidProof),
    }
    let metadata = crate::backend::CommitmentRelationMaterial::metadata(&commitment_material);
    if metadata.ring_dimension() != scheduled.d_a()
        || metadata.source_count() != 1
        || metadata.has_compression()
    {
        return Err(AkitaError::InvalidInput(
            "terminal commitment material disagrees with the terminal plan".into(),
        ));
    }
    let terminal_message = crate::backend::TerminalCommitmentMaterialKernel::terminal_message(
        backend,
        &commitment_material,
    )?;
    transcript.absorb_and_record_bytes(ABSORB_COMMITMENT, terminal_message.as_bytes());
    let t_state = crate::backend::TerminalCommitmentMaterialKernel::consume_terminal_row(
        backend,
        commitment_material,
    )?;

    let fold_level = u32::try_from(level)
        .map_err(|_| AkitaError::InvalidSetup("fold level exceeds u32".into()))?;
    let consumer = backend;
    let context = backend.proof_context(session, fold_level)?;
    let (terminal_response, extension_opening_reduction) = {
        let witness_source = &witness_handle;
        let logical_source = &witness_handle;
        let params = &scheduled;
        let alpha_bits = params.d_a().trailing_zeros() as usize;
        let recursive_num_vars = params.recursive_opening_num_vars()?;
        if sumcheck_challenges.len() > recursive_num_vars {
            return Err(AkitaError::InvalidPointDimension {
                expected: recursive_num_vars,
                actual: sumcheck_challenges.len(),
            });
        }
        let opening_batch = OpeningClaimsLayout::new(sumcheck_challenges.len(), 1)?;
        let needs_reduction = E::DEGREE > 1;
        let (protocol_point, reduction) = if needs_reduction {
            let eor_inputs = vec![crate::backend::EorGroupRequest {
                source: OpeningSource::Witness(logical_source),
                point: &sumcheck_challenges,
                ring_dimension: params.d_a(),
            }];
            let proved = prove_extension_opening_reduction::<F, E, T, B>(
                backend,
                &context,
                &opening_batch,
                &eor_inputs,
                transcript,
                fold_level,
                &[opening],
            )?;
            (
                proved
                    .protocol_points
                    .into_iter()
                    .next()
                    .ok_or(AkitaError::InvalidProof)?,
                Some(proved.reduction),
            )
        } else {
            (sumcheck_challenges, None)
        };
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            params.d_a(),
            |D| {
                let opening_plan = crate::backend::ValidatedRecursiveGroupOpeningPlan::new(
                    &protocol_point,
                    BasisMode::Lagrange,
                    D,
                    params.blocks.positions_per_block,
                    params.blocks.live_blocks,
                    alpha_bits,
                    akita_types::OpeningMethod::EvaluationTrace,
                    witness_source.manifest().logical_len(),
                );
                let prepared =
                    crate::backend::OpaqueWitnessOpeningKernel::prepare_native_witness_opening(
                        consumer,
                        witness_source,
                        &opening_plan,
                    )?;
                for coordinate in &protocol_point {
                    append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, coordinate);
                }
                let (scalar_openings, opening_handle) = prepared.into_parts();
                if scalar_openings.len() != 1 {
                    return Err(AkitaError::InvalidProof);
                }
                let folded_by_claim =
                    crate::backend::OpaqueWitnessOpeningKernel::terminal_native_witness_opening(
                        consumer,
                        opening_handle,
                    )?;
                if reduction.is_none() {
                    append_claim_values_to_transcript::<F, E, T>(&scalar_openings, transcript);
                }
                let trace = crate::protocol::prove::prepare_evaluation_trace_claim::<F, E, T>(
                    &reduction,
                    &scalar_openings,
                    &opening_batch,
                    transcript,
                    u32::try_from(level)
                        .map_err(|_| AkitaError::InvalidSetup("fold level exceeds u32".into()))?,
                )?
                .0;
                // The EOR proof binds the carried extension-field opening to its
                // reduced final claim. Only a degree-one opening can be compared
                // here verbatim.
                if reduction.is_none() && trace.claimed_evaluation != opening {
                    return Err(AkitaError::InvalidInput(
                        "terminal folded opening does not match the carried claim".into(),
                    ));
                }
                if folded_by_claim.len() != 1 {
                    return Err(AkitaError::InvalidProof);
                }
                let e_folded = folded_by_claim
                    .into_iter()
                    .next()
                    .ok_or(AkitaError::InvalidProof)?;
                transcript.absorb_and_record_bytes(
                    ABSORB_TERMINAL_E_HAT,
                    &akita_types::raw_field_segment_bytes(&e_folded)?,
                );
                let output = crate::protocol::fold_grind::sample_terminal_fold_response::<
                    F,
                    E,
                    B::WitnessHandle,
                    _,
                    T,
                    D,
                >(
                    consumer,
                    transcript,
                    u32::try_from(level)
                        .map_err(|_| AkitaError::InvalidSetup("fold level exceeds u32".into()))?,
                    params,
                    &scheduled.fold_challenge_config,
                    witness_source,
                    &scheduled.response_shape,
                )?;
                let terminal_response = akita_types::build_terminal_response_from_payload::<F>(
                    params,
                    &scheduled.response_shape,
                    &e_folded,
                    t_state.clone(),
                    output.encoded_payload,
                )?;
                Ok::<_, AkitaError>((
                    terminal_response,
                    reduction.as_ref().map(|value| value.proof.clone()),
                ))
            }
        )?
    };
    crate::backend::OpaqueResourceReleaseKernel::release_witness_handle(consumer, witness_handle)?;
    let transcript_parts = terminal_response.terminal_transcript_parts()?;
    transcript.absorb_and_record_bytes(ABSORB_TERMINAL_W_REMAINDER, &transcript_parts.response);
    Ok(TerminalLevelProof {
        extension_opening_reduction,
        terminal_response,
    })
}
#[allow(clippy::too_many_arguments)]
fn prepare_suffix<F, E, T, B>(
    backend: &B,
    prefix_slots: &SetupPrefixProverRegistry<F, B::CommitmentHandle>,
    transcript: &mut T,
    current_state: SuffixProverState<F, E, B::CommitmentMaterialHandle, B::WitnessHandle>,
    level: usize,
    level_params: &CommittedGroupParams,
    session: &B::ProofSessionHandle,
    expected_witness_len: usize,
    commitment_ring_dimension: usize,
) -> Result<PreparedFold<F, E, B::WitnessHandle>, AkitaError>
where
    F: Field + CanonicalEncoding + AkitaSerialize + Ring + Unreduced + PseudoMersenne + 'static,
    <F as Unreduced>::Wide: From<F> + AdditiveGroup,
    E: FpExtEncoding<F>
        + ExtField<F>
        + Unreduced
        + Fold
        + Ring
        + MulBaseUnreduced<F>
        + AkitaSerialize
        + 'static,
    T: akita_types::ProverTranscriptGrinding<F>,
    B: ProverBackend<F, E>,
{
    let SuffixProverState {
        witness_handle,
        binding,
        commitment_material: witness_material,
        sumcheck_challenges,
        opening,
        setup_prefix_opening,
    } = current_state;
    let geometry = level_params.outer_payload_geometry()?;
    let commitment = match binding {
        NextWitnessState::OuterPayload(rows)
            if rows.coeff_len() == geometry.transmitted_coefficients() =>
        {
            Commitment::new(rows)
        }
        _ => return Err(AkitaError::InvalidProof),
    };
    let fold_level = u32::try_from(level).map_err(|_| AkitaError::InvalidProof)?;
    let context = backend.proof_context(session, fold_level)?;
    let mut groups = Vec::new();
    let mut claims = Vec::new();
    let mut layouts = Vec::new();
    let mut materials = Vec::new();
    match (
        level_params.setup_prefix().and_then(|p| p.slot_id()),
        setup_prefix_opening,
    ) {
        (Some(id), Some((point, evaluation))) => {
            let slot = prefix_slots
                .get(&id)
                .ok_or_else(|| AkitaError::InvalidSetup("missing setup prefix".into()))?;
            let rows = slot
                .public
                .commitment
                .rows
                .first()
                .cloned()
                .ok_or(AkitaError::InvalidProof)?;
            let commitment = Commitment::new(rows);
            groups.push(OpeningSource::Commitment(&slot.commitment_handle));
            layouts.push(PolynomialGroupLayout::new(point.len(), 1));
            claims.push(akita_types::PolynomialGroupClaims::new(
                point,
                vec![evaluation],
                commitment,
            )?);
        }
        (None, None) => {}
        _ => return Err(AkitaError::InvalidProof),
    }
    let witness_index = groups.len();
    groups.push(OpeningSource::Witness(&witness_handle));
    layouts.push(PolynomialGroupLayout::new(
        level_params.recursive_opening_num_vars()?,
        1,
    ));
    claims.push(akita_types::PolynomialGroupClaims::new(
        sumcheck_challenges,
        vec![opening],
        commitment,
    )?);
    let layout = OpeningClaimsLayout::from_groups(layouts)?;
    let claims = ProverOpeningData::from_parts(
        akita_types::OpeningClaims::from_groups(claims)?,
        layout,
        groups,
    )?;
    if witness_index > 0 {
        let OpeningSource::Commitment(handle) = *claims.group(0)? else {
            return Err(AkitaError::InvalidProof);
        };
        let params = level_params.group_params(claims.opening_layout(), 0)?;
        materials.push(backend.validate_commitment(
            &context.for_group(0),
            handle,
            &params.profile,
            claims.opening_claims().group_commitment(0)?,
        )?);
    }
    materials.push(witness_material);
    claims
        .opening_claims()
        .group_commitment(witness_index)?
        .rows()
        .append_flat_to_transcript(
            ABSORB_COMMITMENT,
            geometry.transcript_ring_dimension(),
            transcript,
        )?;
    let prepared = prepare_fold::<F, E, T, B>(
        backend,
        claims,
        materials,
        true,
        transcript,
        fold_level,
        level_params,
        BasisMode::Lagrange,
        session,
        expected_witness_len,
        commitment_ring_dimension,
        level_params
            .opening_method()
            .requires_extension_opening_reduction(E::DEGREE),
    )?;
    backend.release_witness_handle(witness_handle)?;
    Ok(prepared)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::prove::fold_kernels::prepare_evaluation_trace_claim;
    use akita_transcript::AkitaTranscript;
    use jolt_field::{Fp32, One, Zero};

    type TestF = Fp32<251>;

    fn evaluation_batch_plan() -> akita_types::GrindingPlan {
        akita_types::GrindingPlan::new(
            vec![akita_types::GrindingRun::proof_of_work(
                akita_types::GrindingSite::EvaluationBatch { level: 0 },
                1,
                128,
            )
            .unwrap()],
            128,
        )
        .unwrap()
    }

    #[test]
    fn non_zk_eor_mismatch_is_rejected() {
        let openings = [TestF::zero()];
        let reduction = Some(ExtensionOpeningReduction {
            proof: ExtensionOpeningReductionProof {
                partials: Vec::new(),
                sumcheck: akita_sumcheck::SumcheckProof {
                    round_polys: Vec::new(),
                },
                final_claims: vec![TestF::one()],
            },
            final_factors: vec![TestF::one()],
        });

        let opening_batch = OpeningClaimsLayout::new(0, 1).expect("singleton opening batch");
        let mut transcript = AkitaTranscript::<TestF>::new(b"test/suffix-shared-trace-target");
        let plan = evaluation_batch_plan();
        let mut transcript =
            akita_types::ProverGrindingTranscript::<_>::new(&mut transcript, &plan).unwrap();
        let err = match prepare_evaluation_trace_claim::<TestF, TestF, _>(
            &reduction,
            &openings,
            &opening_batch,
            &mut transcript,
            0,
        ) {
            Ok(_) => panic!("non-zk EOR mismatch should reject"),
            Err(err) => err,
        };

        assert!(
            matches!(err, AkitaError::InvalidProof),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn late_application_batch_rejects_beta_orthogonal_terminal_error() {
        let openings = [TestF::zero(), TestF::zero()];
        // This error vector cancels under the possible early batch (1, 1).
        // The independent application batch must still reject it.
        let reduction = Some(ExtensionOpeningReduction {
            proof: ExtensionOpeningReductionProof {
                partials: Vec::new(),
                sumcheck: akita_sumcheck::SumcheckProof {
                    round_polys: Vec::new(),
                },
                final_claims: vec![TestF::one(), -TestF::one()],
            },
            final_factors: vec![TestF::one()],
        });

        let opening_batch = OpeningClaimsLayout::new(0, 2).expect("two-claim opening batch");
        let mut transcript =
            AkitaTranscript::<TestF>::new(b"test/suffix-independent-late-eor-batch");
        let plan = evaluation_batch_plan();
        let mut transcript =
            akita_types::ProverGrindingTranscript::<_>::new(&mut transcript, &plan).unwrap();
        let result = prepare_evaluation_trace_claim::<TestF, TestF, _>(
            &reduction,
            &openings,
            &opening_batch,
            &mut transcript,
            0,
        );

        assert!(matches!(result, Err(AkitaError::InvalidProof)));
    }
}
