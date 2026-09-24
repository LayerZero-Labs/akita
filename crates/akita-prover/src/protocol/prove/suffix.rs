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
pub(super) fn prove_suffix<Cfg, B>(
    expanded: &akita_types::AkitaSetupDescriptor,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field, B::CommitmentHandle>,
    backend: &B,
    grinding: &mut akita_types::NativeProverGrinding<'_>,
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
        return Err(AkitaError::InvalidProof);
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
    grinding: &mut akita_types::NativeProverGrinding<'_>,
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
    let fold_level = u32::try_from(level)
        .map_err(|_| AkitaError::InvalidSetup("fold level exceeds u32".into()))?;
    akita_transcript::public_native_fields_prover(
        grinding.state_mut(),
        akita_transcript::ProtocolSiteId {
            family: akita_transcript::SITE_FAMILY_TERMINAL,
            level: fold_level,
            stage: 2,
            ..akita_transcript::ProtocolSiteId::default()
        },
        terminal_message.fields(),
    )
    .map_err(|_| AkitaError::InvalidProof)?;
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
        akita_transcript::public_native_extensions_prover::<F, E>(
            grinding.state_mut(),
            akita_transcript::ProtocolSiteId {
                family: akita_transcript::SITE_FAMILY_FOLD_BINDING,
                level: fold_level,
                stage: 5,
                ..akita_transcript::ProtocolSiteId::default()
            },
            &protocol_point,
        )
        .map_err(|_| AkitaError::InvalidProof)?;
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
                    akita_transcript::public_native_extensions_prover::<F, E>(
                        grinding.state_mut(),
                        akita_transcript::ProtocolSiteId {
                            family: akita_transcript::SITE_FAMILY_FOLD_BINDING,
                            level: fold_level,
                            stage: 4,
                            ..akita_transcript::ProtocolSiteId::default()
                        },
                        &scalar_openings,
                    )
                    .map_err(|_| AkitaError::InvalidProof)?;
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
                akita_transcript::send_native_field_group(
                    grinding.state_mut(),
                    akita_transcript::ProtocolSiteId {
                        family: akita_transcript::SITE_FAMILY_TERMINAL,
                        level: fold_level,
                        stage: 1,
                        ..akita_transcript::ProtocolSiteId::default()
                    },
                    e_folded.coeffs(),
                )
                .map_err(|_| AkitaError::InvalidProof)?;
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
        .ok_or(AkitaError::InvalidProof)?;
    let z_payload = terminal_response
        .z_payloads
        .first()
        .ok_or(AkitaError::InvalidProof)?;
    tracing::info!(
        native_terminal_z_bytes = z_payload.len(),
        native_terminal_e_field_elements = terminal_response.e_fields.coeff_len(),
        native_terminal_t_field_elements = terminal_response.t_fields.coeff_len(),
        "native terminal response bytes"
    );
    akita_transcript::send_native_bounded_bytes(
        grinding.state_mut(),
        akita_transcript::ProtocolSiteId {
            family: akita_transcript::SITE_FAMILY_TERMINAL,
            level: fold_level,
            round: 3,
            ..akita_transcript::ProtocolSiteId::default()
        },
        z_payload,
        group.z_payload_bytes,
    )
    .map_err(|_| AkitaError::InvalidProof)
}
#[allow(clippy::too_many_arguments)]
fn prepare_suffix<F, E, B>(
    backend: &B,
    prefix_slots: &SetupPrefixProverRegistry<F, B::CommitmentHandle>,
    grinding: &mut akita_types::NativeProverGrinding<'_>,
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
            session,
            &context.for_group(0),
            handle,
            &params.profile,
            claims.opening_claims().group_commitment(0)?,
        )?);
    }
    materials.push(witness_material);
    akita_transcript::public_native_fields_prover(
        grinding.state_mut(),
        akita_transcript::ProtocolSiteId {
            family: akita_transcript::SITE_FAMILY_FOLD_BINDING,
            level: fold_level,
            stage: 3,
            detail: u32::try_from(geometry.transcript_ring_dimension())
                .map_err(|_| AkitaError::InvalidProof)?,
            ..akita_transcript::ProtocolSiteId::default()
        },
        claims
            .opening_claims()
            .group_commitment(witness_index)?
            .rows()
            .coeffs(),
    )
    .map_err(|_| AkitaError::InvalidProof)?;
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
    use akita_transcript::new_native_prover;
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
            final_claims: vec![TestF::one()],
            final_factors: vec![TestF::one()],
        });

        let opening_batch = OpeningClaimsLayout::new(0, 1).expect("singleton opening batch");
        let native = new_native_prover(b"test/suffix-shared-trace-target", b"test").unwrap();
        let plan = evaluation_batch_plan();
        let mut grinding = akita_types::NativeProverGrinding::new(native, &plan);
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
            final_claims: vec![TestF::one(), -TestF::one()],
            final_factors: vec![TestF::one()],
        });

        let opening_batch = OpeningClaimsLayout::new(0, 2).expect("two-claim opening batch");
        let native = new_native_prover(b"test/suffix-independent-late-eor-batch", b"test").unwrap();
        let plan = evaluation_batch_plan();
        let mut grinding = akita_types::NativeProverGrinding::new(native, &plan);
        let result = prepare_evaluation_trace_claim::<TestF, TestF>(
            &reduction,
            &openings,
            &opening_batch,
            &mut grinding,
            0,
        );

        assert!(matches!(result, Err(AkitaError::InvalidProof)));
    }
}
