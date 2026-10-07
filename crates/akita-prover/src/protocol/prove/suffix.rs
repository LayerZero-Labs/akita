use super::*;
use akita_types::GrindingReplay;
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
pub(super) fn prove_suffix<Cfg, B>(
    expanded: &akita_types::AkitaSetupDescriptor,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field, B::CommitmentHandle>,
    backend: &B,
    grinding: &mut akita_types::ProverGrinding<'_>,
    mut current_state: SuffixProverState<
        Cfg::Field,
        Cfg::ExtField,
        B::CommitmentMaterialHandle,
        B::WitnessHandle,
    >,
    schedule: &FoldSchedule,
    session: &B::ProofSessionHandle,
) -> Result<RecursiveSuffixOutcome, AkitaError>
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
    B: ProverBackend<Cfg::Field, Cfg::ExtField>,
{
    for (index, step) in schedule.recursive_folds.iter().enumerate() {
        let level = index + 1;
        if current_state.witness_handle.manifest().logical_len() != step.input_witness_len {
            return Err(AkitaError::Internal(
                "recursive suffix witness length differs from scheduled input".into(),
            ));
        }
        let (next_params, next_binding) = schedule.recursive_folds.get(index + 1).map_or(
            (
                fold::FoldSuccessorParams::Terminal(&schedule.terminal),
                akita_params::NextWitnessBindingPolicy::TerminalInnerState,
            ),
            |next| {
                (
                    fold::FoldSuccessorParams::Recursive(next),
                    akita_params::NextWitnessBindingPolicy::OuterPayload,
                )
            },
        );
        let prepared = prepare_suffix::<Cfg::Field, Cfg::ExtField, B>(
            backend,
            prefix_slots,
            grinding,
            current_state,
            level,
            &step.params,
            session,
            step.output_witness_len,
            next_params.inner_ring_dimension(),
        )?;
        let output = prove_fold::<Cfg::Field, Cfg::ExtField, B>(
            expanded,
            prefix_slots,
            backend,
            session,
            grinding,
            level,
            &step.params,
            next_params,
            step.output_witness_len,
            next_binding,
            prepared,
        )?;
        current_state = output.next_state;
    }
    if current_state.witness_handle.manifest().logical_len() != schedule.terminal.input_witness_len
    {
        return Err(AkitaError::Internal(
            "terminal suffix witness length differs from scheduled input".into(),
        ));
    }
    prove_terminal_suffix::<Cfg::Field, Cfg::ExtField, B>(
        backend,
        grinding,
        schedule.recursive_folds.len() + 1,
        current_state,
        &schedule.terminal,
        session,
    )?;
    Ok(RecursiveSuffixOutcome {
        num_levels: schedule.num_fold_levels(),
    })
}

#[allow(clippy::too_many_arguments)]
fn prove_terminal_suffix<F, E, B>(
    backend: &B,
    grinding: &mut akita_types::ProverGrinding<'_>,
    level: usize,
    current_state: SuffixProverState<F, E, B::CommitmentMaterialHandle, B::WitnessHandle>,
    scheduled: &TerminalFoldParams,
    session: &B::ProofSessionHandle,
) -> Result<(), AkitaError>
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
        return Err(AkitaError::Internal(
            "terminal fold cannot receive a setup-prefix opening".into(),
        ));
    }
    match binding {
        NextWitnessState::TerminalInnerState => {}
        NextWitnessState::OuterPayload(_) => {
            return Err(AkitaError::Internal(
                "terminal suffix received an outer-payload binding".into(),
            ))
        }
    }
    let metadata = crate::backend::CommitmentRelationMaterial::metadata(&commitment_material);
    if metadata.ring_dimension() != scheduled.d_a()
        || metadata.source_count() != 1
        || metadata.has_compression()
    {
        return Err(AkitaError::Internal(
            "terminal commitment material disagrees with the terminal plan".into(),
        ));
    }
    let terminal_message = crate::backend::TerminalCommitmentMaterialKernel::terminal_message(
        backend,
        &commitment_material,
    )?;
    let fold_level =
        u32::try_from(level).map_err(|_| AkitaError::Internal("fold level exceeds u32".into()))?;
    akita_transcript::public_fields_prover(
        grinding.state_mut(),
        akita_types::FoldSite::TerminalTFields { level: fold_level }.id()?,
        terminal_message.fields(),
    )?;
    let t_state = crate::backend::TerminalCommitmentMaterialKernel::consume_terminal_row(
        backend,
        commitment_material,
    )?;

    let consumer = backend;
    let context = backend.proof_context(session, fold_level)?;
    let terminal_response = {
        let witness_source = &witness_handle;
        let logical_source = &witness_handle;
        let params = &scheduled;
        let alpha_bits = params.d_a().trailing_zeros() as usize;
        let recursive_num_vars = params.recursive_opening_num_vars()?;
        if sumcheck_challenges.len() > recursive_num_vars {
            return Err(AkitaError::Internal(format!(
                "terminal sumcheck point dimension: expected {recursive_num_vars}, actual {}",
                sumcheck_challenges.len(),
            )));
        }
        let opening_batch = OpeningClaimsLayout::new(sumcheck_challenges.len(), 1)?;
        let needs_reduction = E::DEGREE > 1;
        let (protocol_point, reduction) = if needs_reduction {
            let eor_inputs = vec![crate::backend::EorGroupRequest {
                source: OpeningSource::Witness(logical_source),
                point: &sumcheck_challenges,
                ring_dimension: params.d_a(),
            }];
            let proved = prove_extension_opening_reduction::<F, E, B>(
                backend,
                session,
                &context,
                &opening_batch,
                &eor_inputs,
                grinding,
                fold_level,
                &[opening],
            )?;
            (
                proved.protocol_points.into_iter().next().ok_or_else(|| {
                    AkitaError::Internal("terminal EOR returned no protocol point".into())
                })?,
                Some(proved.reduction),
            )
        } else {
            (sumcheck_challenges, None)
        };
        akita_transcript::public_extensions::<F, E, _>(
            grinding.state_mut(),
            akita_types::FoldSite::TerminalPoint { level: fold_level }.id()?,
            &protocol_point,
        )?;
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
                    akita_params::OpeningMethod::EvaluationTrace,
                    witness_source.manifest().logical_len(),
                );
                let prepared =
                    crate::backend::OpaqueWitnessOpeningKernel::prepare_native_witness_opening(
                        consumer,
                        witness_source,
                        &opening_plan,
                    )?;
                let (scalar_openings, opening_handle) = prepared.into_parts();
                if scalar_openings.len() != 1 {
                    return Err(AkitaError::Internal(
                        "terminal backend scalar opening count differs from one".into(),
                    ));
                }
                let folded_by_claim =
                    crate::backend::OpaqueWitnessOpeningKernel::terminal_native_witness_opening(
                        consumer,
                        opening_handle,
                    )?;
                if reduction.is_none() {
                    akita_transcript::public_extensions::<F, E, _>(
                        grinding.state_mut(),
                        akita_types::FoldSite::TerminalOpening { level: fold_level }.id()?,
                        &scalar_openings,
                    )?;
                }
                let trace = crate::protocol::prove::prepare_evaluation_trace_claim::<F, E>(
                    &reduction,
                    &scalar_openings,
                    &opening_batch,
                    grinding,
                    fold_level,
                )?
                .0;
                // The EOR proof binds the carried extension-field opening to its
                // reduced final claim. Only a degree-one opening can be compared
                // here verbatim.
                if reduction.is_none() && trace.claimed_evaluation != opening {
                    return Err(AkitaError::Internal(
                        "terminal folded opening does not match the carried claim".into(),
                    ));
                }
                if folded_by_claim.len() != 1 {
                    return Err(AkitaError::Internal(
                        "terminal backend folded opening count differs from one".into(),
                    ));
                }
                let e_folded = folded_by_claim.into_iter().next().ok_or_else(|| {
                    AkitaError::Internal("terminal backend returned no folded opening".into())
                })?;
                akita_transcript::send_field_group(
                    grinding.state_mut(),
                    akita_types::FoldSite::TerminalEFields { level: fold_level }.id()?,
                    e_folded.coeffs(),
                )?;
                let output = crate::protocol::fold_grind::sample_terminal_fold_response::<
                    F,
                    E,
                    B::WitnessHandle,
                    _,
                    D,
                >(
                    consumer,
                    grinding,
                    fold_level,
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
                Ok::<_, AkitaError>(terminal_response)
            }
        )?
    };
    crate::backend::OpaqueResourceReleaseKernel::release_witness_handle(consumer, witness_handle)?;
    let group = scheduled
        .response_shape
        .layout
        .groups
        .first()
        .ok_or_else(|| AkitaError::Internal("terminal response shape has no group".into()))?;
    let z_payload = terminal_response
        .z_payloads
        .first()
        .ok_or_else(|| AkitaError::Internal("terminal response has no z payload".into()))?;
    tracing::info!(
        native_terminal_z_bytes = z_payload.len(),
        native_terminal_e_field_elements = terminal_response.e_fields.coeff_len(),
        native_terminal_t_field_elements = terminal_response.t_fields.coeff_len(),
        "native terminal response bytes"
    );
    akita_transcript::send_bounded_bytes(
        grinding.state_mut(),
        akita_types::FoldSite::TerminalZPayload { level: fold_level }.id()?,
        z_payload,
        group.z_payload_bytes,
    )
}
#[allow(clippy::too_many_arguments)]
fn prepare_suffix<F, E, B>(
    backend: &B,
    prefix_slots: &SetupPrefixProverRegistry<F, B::CommitmentHandle>,
    grinding: &mut akita_types::ProverGrinding<'_>,
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
        _ => {
            return Err(AkitaError::Internal(
                "recursive suffix binding differs from scheduled outer payload".into(),
            ))
        }
    };
    let fold_level = u32::try_from(level)
        .map_err(|_| AkitaError::Internal("recursive suffix fold level exceeds u32".into()))?;
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
                .ok_or_else(|| {
                    AkitaError::Internal(
                        "recursive suffix setup-prefix commitment has no row".into(),
                    )
                })?;
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
        _ => {
            return Err(AkitaError::Internal(
                "recursive suffix setup-prefix opening differs from scheduled prefix presence"
                    .into(),
            ))
        }
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
            return Err(AkitaError::Internal(
                "recursive suffix setup-prefix source is not a commitment".into(),
            ));
        };
        let params = level_params.group_params(claims.opening_layout(), 0)?;
        materials.push(backend.validate_commitment(
            session,
            &context.for_group(0),
            handle,
            &params.profile,
            claims.opening_claims().group_commitment(0)?,
        )?);
    }
    materials.push(witness_material);
    akita_transcript::public_fields_prover(
        grinding.state_mut(),
        akita_types::FoldSite::WitnessCommitment {
            level: fold_level,
            ring_dimension: geometry.transcript_ring_dimension(),
        }
        .id()?,
        claims
            .opening_claims()
            .group_commitment(witness_index)?
            .rows()
            .coeffs(),
    )?;
    let prepared = prepare_fold::<F, E, B>(
        backend,
        claims,
        materials,
        true,
        grinding,
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
    use akita_transcript::new_prover_channel;
    use jolt_field::{One, Prime32Offset99, Zero};

    type TestF = Prime32Offset99;

    fn evaluation_batch_plan() -> akita_params::GrindingPlan {
        let challenge_order = akita_params::ChallengeFieldOrder::from_full_capacity(128).unwrap();
        akita_params::GrindingPlan::new(
            vec![akita_params::GrindingRun::proof_of_work(
                akita_params::GrindingSite::EvaluationBatch { level: 0 },
                1,
                challenge_order,
            )
            .unwrap()],
            challenge_order,
        )
        .unwrap()
    }

    #[test]
    fn non_zk_eor_mismatch_is_rejected() {
        let openings = [TestF::zero()];
        let reduction = Some(ExtensionOpeningReduction {
            final_claims: vec![TestF::one()],
            final_factors: vec![TestF::one()],
        });

        let opening_batch = OpeningClaimsLayout::new(0, 1).expect("singleton opening batch");
        let channel = new_prover_channel(b"test/suffix-shared-trace-target", b"test").unwrap();
        let plan = evaluation_batch_plan();
        let mut grinding = akita_types::ProverGrinding::new(channel, &plan);
        let err = match prepare_evaluation_trace_claim::<TestF, TestF>(
            &reduction,
            &openings,
            &opening_batch,
            &mut grinding,
            0,
        ) {
            Ok(_) => panic!("non-zk EOR mismatch should reject"),
            Err(err) => err,
        };

        assert!(
            matches!(err, AkitaError::Internal(_)),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn late_application_batch_rejects_beta_orthogonal_terminal_error() {
        let openings = [TestF::zero(), TestF::zero()];
        // This error vector cancels under the possible early batch (1, 1).
        // The independent application batch must still reject it.
        let reduction = Some(ExtensionOpeningReduction {
            final_claims: vec![TestF::one(), -TestF::one()],
            final_factors: vec![TestF::one()],
        });

        let opening_batch = OpeningClaimsLayout::new(0, 2).expect("two-claim opening batch");
        let channel =
            new_prover_channel(b"test/suffix-independent-late-eor-batch", b"test").unwrap();
        let plan = evaluation_batch_plan();
        let mut grinding = akita_types::ProverGrinding::new(channel, &plan);
        let result = prepare_evaluation_trace_claim::<TestF, TestF>(
            &reduction,
            &openings,
            &opening_batch,
            &mut grinding,
            0,
        );

        assert!(
            matches!(result, Err(AkitaError::Internal(_))),
            "unexpected error: {:?}",
            result.err()
        );
    }
}
