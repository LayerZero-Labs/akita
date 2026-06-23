use super::*;
use crate::api::commitment::validate_onehot_chunk_size_for_params;
#[cfg(not(feature = "zk"))]
use akita_types::schedule_terminal_direct_witness_shape;
use akita_types::{
    GROUPED_ROOT_DENSE_UNSUPPORTED, GROUPED_ROOT_RECURSIVE_SETUP_UNSUPPORTED,
    GROUPED_ROOT_TIERED_UNSUPPORTED, GROUPED_ROOT_UNSUPPORTED,
};

fn reject_unsupported_grouped_root<Cfg, F, P, const D: usize>(
    opening_batch: &OpeningBatch,
    polys: &[&P],
    setup_contribution_mode: SetupContributionMode,
) -> Result<(), AkitaError>
where
    Cfg: CommitmentConfig,
    F: FieldCore,
    P: AkitaPolyOps<F, D>,
{
    if opening_batch.num_commitment_groups() <= 1 {
        return Ok(());
    }
    if Cfg::TIERED_COMMITMENT {
        return Err(AkitaError::InvalidSetup(
            GROUPED_ROOT_TIERED_UNSUPPORTED.to_string(),
        ));
    }
    if setup_contribution_mode == SetupContributionMode::Recursive {
        return Err(AkitaError::InvalidSetup(
            GROUPED_ROOT_RECURSIVE_SETUP_UNSUPPORTED.to_string(),
        ));
    }
    if polys.iter().any(|poly| poly.onehot_chunk_size().is_none()) {
        return Err(AkitaError::InvalidInput(
            GROUPED_ROOT_DENSE_UNSUPPORTED.to_string(),
        ));
    }
    Err(AkitaError::InvalidInput(
        GROUPED_ROOT_UNSUPPORTED.to_string(),
    ))
}

/// Build a root-direct batched proof from flattened polynomial references and
/// their commitment hint.
///
/// # Errors
///
/// Returns an error if any polynomial cannot produce a direct root witness.
pub fn prove_root_direct<F, L, const D: usize, P>(
    polys: &[&P],
    hint: &AkitaCommitmentHint<F, D>,
) -> Result<AkitaBatchedProof<F, L>, AkitaError>
where
    F: FieldCore,
    L: ExtField<F>,
    P: AkitaPolyOps<F, D>,
{
    let witnesses = polys
        .iter()
        .map(|poly| poly.direct_root_witness())
        .collect::<Result<Vec<_>, _>>()?;
    #[cfg(feature = "zk")]
    {
        let b_blinding_digits = hint
            .b_blinding_digits()
            .iter()
            .map(|digits| {
                let mut flat_digits = Vec::with_capacity(digits.flat_digits().len() * D);
                for plane in digits.flat_digits() {
                    flat_digits.extend_from_slice(plane);
                }
                flat_digits
            })
            .collect();
        Ok(AkitaBatchedProof {
            zk_hiding: ZkHidingProof::default(),
            root: AkitaBatchedRootProof::new_zero_fold(witnesses, b_blinding_digits),
            steps: Vec::new(),
        })
    }
    #[cfg(not(feature = "zk"))]
    {
        let _ = hint;
        Ok(AkitaBatchedProof {
            root: AkitaBatchedRootProof::new_zero_fold(witnesses),
            steps: Vec::new(),
        })
    }
}

/// Drive batched proving end-to-end under config `Cfg`.
///
/// This owns the full top-level prover work: validate/flatten public prover
/// claims, select the schedule from `Cfg`, apply the root-direct shortcut when
/// the selected schedule says no fold is needed, bind the transcript instance
/// descriptor, and either emit a root-direct proof or run the folded-root
/// prover.
///
/// # Errors
///
/// Returns an error if claim preparation, schedule selection, root-direct
/// witness construction, transcript binding, or folded-root proving fails.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn batched_prove<'a, Cfg, T, P, B, const D: usize>(
    expanded: &Arc<AkitaExpandedSetup<Cfg::Field>>,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field, D>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    claims: ProverOpeningBatch<
        'a,
        Cfg::ExtField,
        P,
        RingCommitment<Cfg::Field, D>,
        AkitaCommitmentHint<Cfg::Field, D>,
    >,
    transcript: &mut T,
    basis: BasisMode,
    setup_contribution_mode: SetupContributionMode,
) -> Result<AkitaBatchedProof<Cfg::Field, Cfg::ExtField>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    Cfg::ExtField: FpExtEncoding<Cfg::Field> + MulBaseUnreduced<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + ExtField<Cfg::Field>
        + FrobeniusExtField<Cfg::Field>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<Cfg::Field> + ProverTranscriptGrind<Cfg::Field>,
    P: AkitaPolyOps<Cfg::Field, D>,
    B: ProverComputeBackend<Cfg::Field>,
{
    backend.validate_prepared_setup::<D>(prepared, expanded.as_ref())?;
    let group_sizes = claims.group_sizes();
    validate_batched_inputs(expanded.as_ref(), claims.point(), &group_sizes, true)?;
    let opening_batch =
        opening_batch_shape_for_prove::<_, Cfg::Field, P, _, _, D>(&claims, "batched_prove")?;
    let flat_polys = claims.flat_polys();
    reject_unsupported_grouped_root::<Cfg, Cfg::Field, P, D>(
        &opening_batch,
        &flat_polys,
        setup_contribution_mode,
    )?;
    let num_vars = opening_batch.num_vars();
    let mut schedule = Cfg::get_params_for_prove(&opening_batch)?;
    if let Some(root_step) = schedule_root_fold_step(&schedule) {
        let alpha_bits = root_step.params.ring_dimension.trailing_zeros() as usize;
        if !folded_root_supports_opening_shape::<Cfg::Field, Cfg::ExtField, Cfg::ExtField, D>(
            std::slice::from_ref(&claims.point()),
            &root_step.params,
            alpha_bits,
        ) && !root_tensor_projection_enabled::<Cfg::Field, Cfg::ExtField, Cfg::ExtField, D>(
            num_vars,
        ) {
            let commit_params = Cfg::get_params_for_batched_commitment(&opening_batch)?;
            schedule = root_direct_schedule(num_vars, commit_params)?;
        }
    }
    let root_commit_params = match schedule.steps.first() {
        Some(Step::Fold(root)) => Some(&root.params),
        Some(Step::Direct(root)) => root.params.as_ref(),
        None => None,
    }
    .ok_or_else(|| AkitaError::InvalidSetup("root schedule is empty".to_string()))?;
    validate_onehot_chunk_size_for_params::<Cfg::Field, D, &P>(&flat_polys, root_commit_params)?;

    bind_transcript_instance_descriptor::<Cfg::Field, T, D, Cfg>(
        expanded.as_ref(),
        &opening_batch,
        &schedule,
        basis,
        transcript,
    )?;

    if schedule_is_root_direct(&schedule) {
        let commitment_hint = claims
            .single_group()
            .map(|group| &group.hint)
            .ok_or_else(|| {
                AkitaError::InvalidInput(
                    "multi-group root-direct proving is not supported yet".to_string(),
                )
            })?;
        return prove_root_direct::<Cfg::Field, Cfg::ExtField, D, P>(&flat_polys, commitment_hint);
    }

    if schedule_root_fold_step(&schedule).is_none() {
        return Err(AkitaError::InvalidSetup(
            "root schedule does not start with a fold".to_string(),
        ));
    }
    prove::<Cfg, T, P, B, D>(
        expanded,
        prefix_slots,
        backend,
        prepared,
        transcript,
        claims,
        &schedule,
        basis,
        setup_contribution_mode,
    )
    .map(|(proof, _total_levels)| proof)
}

/// Prove a folded batched root and assemble the recursive suffix under config
/// `Cfg`.
///
/// The prover crate owns folded-root preparation (root schedule shape checks,
/// opening-point reduction, commitment row shape validation), root fold
/// proving, the next-`w` commitment, recursive suffix proving, and final proof
/// assembly. All policy facts are obtained directly from `Cfg`.
///
/// # Errors
///
/// Returns an error if the schedule is not folded, root inputs are malformed,
/// root proving fails, or suffix construction fails.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[inline(never)]
pub fn prove<'a, Cfg, T, P, B, const D: usize>(
    expanded: &Arc<AkitaExpandedSetup<Cfg::Field>>,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field, D>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    claims: ProverOpeningBatch<
        'a,
        Cfg::ExtField,
        P,
        RingCommitment<Cfg::Field, D>,
        AkitaCommitmentHint<Cfg::Field, D>,
    >,
    schedule: &Schedule,
    basis: BasisMode,
    setup_contribution_mode: SetupContributionMode,
) -> Result<(AkitaBatchedProof<Cfg::Field, Cfg::ExtField>, usize), AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    Cfg::ExtField: FpExtEncoding<Cfg::Field> + MulBaseUnreduced<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + ExtField<Cfg::Field>
        + FrobeniusExtField<Cfg::Field>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<Cfg::Field> + ProverTranscriptGrind<Cfg::Field>,
    P: AkitaPolyOps<Cfg::Field, D>,
    B: ProverComputeBackend<Cfg::Field>,
{
    backend.validate_prepared_setup::<D>(prepared, expanded.as_ref())?;
    #[cfg(feature = "zk")]
    let opening_batch =
        opening_batch_shape_for_prove::<_, Cfg::Field, P, _, _, D>(&claims, "batched_prove")?;

    let root_scheduled = schedule.get_execution_schedule(0)?;

    let root_packed_w_len = root_current_w_len(&root_scheduled.params);
    root_scheduled.validate_current_w_len(root_packed_w_len)?;

    #[cfg(feature = "zk")]
    let (zk_hiding_commitment, mut zk_hiding_state) =
        build_zk_hiding_context::<Cfg::Field, Cfg::ExtField, Cfg::ExtField, B, D>(
            backend,
            prepared,
            schedule,
            &root_scheduled.params,
            opening_batch.num_vars(),
            opening_batch.num_claims(),
            1,
        )?;
    #[cfg(feature = "zk")]
    transcript.append_serde(ABSORB_ZK_HIDING_COMMITMENT, &zk_hiding_commitment.u_blind);

    if root_scheduled.is_terminal {
        // Root is itself the terminal fold: no recursive suffix.
        #[cfg(not(feature = "zk"))]
        let terminal_shape = schedule_terminal_direct_witness_shape(schedule)?;
        let terminal =
            prove_terminal_root_fold_with_params::<Cfg, Cfg::Field, Cfg::ExtField, T, P, B, D>(
                expanded,
                backend,
                prepared,
                transcript,
                claims,
                &root_scheduled,
                #[cfg(not(feature = "zk"))]
                terminal_shape,
                basis,
                setup_contribution_mode,
                #[cfg(feature = "zk")]
                &mut zk_hiding_state,
            )?;
        #[cfg(feature = "zk")]
        let zk_hiding_proof = zk_hiding_state.into_proof(zk_hiding_commitment)?;
        return Ok((
            AkitaBatchedProof {
                #[cfg(feature = "zk")]
                zk_hiding: zk_hiding_proof,
                root: AkitaBatchedRootProof::new_terminal(terminal),
                steps: Vec::new(),
            },
            1,
        ));
    }

    let root = prove_root::<Cfg::Field, Cfg::ExtField, T, P, B, Cfg, D>(
        expanded,
        prefix_slots,
        backend,
        prepared,
        transcript,
        claims,
        &root_scheduled,
        #[cfg(feature = "zk")]
        zk_hiding_state,
        basis,
        setup_contribution_mode,
    )?;
    let next_state = root.next_state;
    let root = AkitaBatchedRootProof::new(root.level_proof);

    let suffix = crate::prove_suffix::<Cfg, T, B, D>(
        expanded,
        prefix_slots,
        backend,
        prepared,
        transcript,
        next_state,
        schedule,
        setup_contribution_mode,
    )?;
    #[cfg(feature = "zk")]
    let zk_hiding = suffix.zk_hiding.into_proof(zk_hiding_commitment)?;
    Ok((
        AkitaBatchedProof {
            #[cfg(feature = "zk")]
            zk_hiding,
            root,
            steps: suffix.steps,
        },
        suffix.num_levels,
    ))
}
