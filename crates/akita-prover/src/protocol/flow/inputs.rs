use super::*;
use crate::api::commitment::validate_onehot_chunk_size_for_params;

struct ProverPreparedIncidence<'a, F: FieldCore, E: FieldCore, P, const D: usize> {
    points: Vec<&'a [E]>,
    point_payloads:
        Vec<CommittedPolynomials<'a, P, RingCommitment<F, D>, AkitaCommitmentHint<F, D>>>,
    summary: ClaimIncidenceSummary,
}

fn prover_claims_to_incidence<'a, F, E, P, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    claims: ProverClaims<'a, E, P, RingCommitment<F, D>, AkitaCommitmentHint<F, D>>,
) -> Result<ProverPreparedIncidence<'a, F, E, P, D>, AkitaError>
where
    F: FieldCore,
    E: FieldCore,
{
    let points: Vec<&'a [E]> = claims.iter().map(|(point, _)| *point).collect();
    let mut point_payloads: Vec<
        CommittedPolynomials<'a, P, RingCommitment<F, D>, AkitaCommitmentHint<F, D>>,
    > = Vec::with_capacity(claims.len());
    let mut incidence_claims = Vec::new();

    for (point_idx, (_, payload)) in claims.into_iter().enumerate() {
        let poly_count = payload.poly_count();
        incidence_claims.extend((0..poly_count).map(|poly_idx| IncidenceClaim {
            point_idx,
            poly_idx,
            // Prover inputs do not contain claimed evaluations. The shared
            // incidence validator ignores this field, so zero is only a
            // structural placeholder.
            claimed_eval: E::zero(),
        }));
        point_payloads.push(payload);
    }

    let incidence = ClaimIncidence {
        points: points.clone(),
        claims: incidence_claims,
    };
    let summary = incidence.validate(ClaimIncidenceLimits {
        max_num_vars: expanded.seed.max_num_vars,
        max_num_points: expanded.seed.max_num_points,
        max_num_claims: expanded.seed.max_num_batched_polys,
    })?;

    Ok(ProverPreparedIncidence {
        points,
        point_payloads,
        summary,
    })
}

/// Validate and flatten batched prover claims into the root proof shape.
///
/// # Errors
///
/// Returns an error if the claim shape exceeds setup capacity, mixes
/// incompatible dimensions, or has malformed batch counts.
pub fn prepare_batched_prove_inputs<'a, F, E, P, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    claims: ProverClaims<'a, E, P, RingCommitment<F, D>, AkitaCommitmentHint<F, D>>,
) -> Result<PreparedBatchedProveInputs<'a, F, E, P, D>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    validate_batched_inputs(expanded, &claims, |payload| payload.polynomials.len(), true)?;

    let prepared_incidence = prover_claims_to_incidence(expanded, claims)?;
    let opening_points = prepared_incidence.points;
    let commitments_by_point: Vec<RingCommitment<F, D>> = prepared_incidence
        .point_payloads
        .iter()
        .map(|payload| payload.commitment.clone())
        .collect();
    let incidence_summary = prepared_incidence.summary;
    let flat_polys: Vec<&P> = incidence_summary
        .claim_to_point()
        .iter()
        .zip(incidence_summary.claim_poly_indices().iter())
        .map(|(&point_idx, &poly_idx)| {
            &prepared_incidence.point_payloads[point_idx].polynomials[poly_idx]
        })
        .collect();
    let group_polys: Vec<&P> = prepared_incidence
        .point_payloads
        .iter()
        .flat_map(|payload| payload.polynomials.iter())
        .collect();
    let flat_hints: Vec<AkitaCommitmentHint<F, D>> = prepared_incidence
        .point_payloads
        .into_iter()
        .map(|payload| payload.hint)
        .collect();

    Ok(PreparedBatchedProveInputs {
        opening_points,
        commitments_by_point,
        incidence_summary,
        flat_polys,
        group_polys,
        flat_hints,
    })
}

/// Build a root-direct batched proof from flattened polynomial references and
/// their commitment-group hints.
///
/// # Errors
///
/// Returns an error if any polynomial cannot produce a direct root witness.
pub fn prove_root_direct<F, L, const D: usize, P>(
    polys: &[&P],
    hints: &[AkitaCommitmentHint<F, D>],
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
        let b_blinding_digits = hints
            .iter()
            .flat_map(|hint| hint.b_blinding_digits())
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
        let _ = hints;
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
pub fn prove_batched<'a, Cfg, T, P, B, const D: usize>(
    expanded: &Arc<AkitaExpandedSetup<Cfg::Field>>,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field, D>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    claims: ProverClaims<
        'a,
        Cfg::ClaimField,
        P,
        RingCommitment<Cfg::Field, D>,
        AkitaCommitmentHint<Cfg::Field, D>,
    >,
    transcript: &mut T,
    basis: BasisMode,
    setup_contribution_mode: SetupContributionMode,
) -> Result<AkitaBatchedProof<Cfg::Field, Cfg::ChallengeField>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    Cfg::ClaimField: RingSubfieldEncoding<Cfg::Field> + MulBaseUnreduced<Cfg::Field>,
    Cfg::ChallengeField: RingSubfieldEncoding<Cfg::Field>
        + ExtField<Cfg::ClaimField>
        + ExtField<Cfg::Field>
        + FrobeniusExtField<Cfg::Field>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<Cfg::Field>,
    P: AkitaPolyOps<Cfg::Field, D>,
    B: ProverComputeBackend<Cfg::Field>,
{
    backend.validate_prepared_setup::<D>(prepared, expanded.as_ref())?;
    let prepared_claims = {
        let _span = tracing::info_span!("prepare_batched_prove_inputs").entered();
        prepare_batched_prove_inputs::<Cfg::Field, Cfg::ClaimField, P, D>(
            expanded.as_ref(),
            claims,
        )?
    };
    let num_vars = prepared_claims.incidence_summary.num_vars();
    let mut schedule = Cfg::get_params_for_prove(&prepared_claims.incidence_summary)?;
    if let Some(root_step) = schedule_root_fold_step(&schedule) {
        let alpha_bits = root_step.params.ring_dimension.trailing_zeros() as usize;
        if !folded_root_supports_opening_shape::<Cfg::Field, Cfg::ClaimField, Cfg::ChallengeField, D>(
            &prepared_claims.opening_points,
            &root_step.params,
            alpha_bits,
        ) && !root_tensor_projection_enabled::<Cfg::Field, Cfg::ClaimField, Cfg::ChallengeField, D>(
            num_vars,
        ) {
            let commit_params =
                Cfg::get_params_for_batched_commitment(&prepared_claims.incidence_summary)?;
            schedule = root_direct_schedule(num_vars, commit_params)?;
        }
    }
    let root_commit_params = match schedule.steps.first() {
        Some(Step::Fold(root)) => Some(&root.params),
        Some(Step::Direct(root)) => root.params.as_ref(),
        None => None,
    }
    .ok_or_else(|| AkitaError::InvalidSetup("root schedule is empty".to_string()))?;
    validate_onehot_chunk_size_for_params::<Cfg::Field, D, &P>(
        &prepared_claims.group_polys,
        root_commit_params,
    )?;

    bind_transcript_instance_descriptor::<Cfg::Field, T, D, Cfg>(
        expanded.as_ref(),
        &prepared_claims.incidence_summary,
        &schedule,
        basis,
        transcript,
    )?;

    if schedule_is_root_direct(&schedule) {
        return prove_root_direct::<Cfg::Field, Cfg::ChallengeField, D, P>(
            &prepared_claims.group_polys,
            &prepared_claims.flat_hints,
        );
    }

    if schedule_root_fold_step(&schedule).is_none() {
        return Err(AkitaError::InvalidSetup(
            "root schedule does not start with a fold".to_string(),
        ));
    }
    let root_next_params = scheduled_next_level_params(&schedule, 1)?;

    prove_folded_batched::<Cfg, T, P, B, D>(
        expanded,
        prefix_slots,
        backend,
        prepared,
        transcript,
        prepared_claims,
        &schedule,
        basis,
        &root_next_params,
        setup_contribution_mode,
    )
    .map(|(proof, _total_levels)| proof)
}

/// Build the recursive suffix from an intermediate-root handoff, then
/// assemble the final folded batched proof.
///
/// The caller owns suffix schedule/config policy inside `build_suffix`; this
/// helper owns the config-free handoff from root prover output into suffix
/// construction and final proof assembly.
///
/// # Errors
///
/// Returns an error if suffix construction fails.
pub fn build_folded_batched_proof_with_suffix<F, L, const D: usize, BuildSuffix>(
    root: RootLevelProverOutput<F, L, D>,
    build_suffix: BuildSuffix,
) -> Result<(AkitaBatchedProof<F, L>, usize), AkitaError>
where
    F: FieldCore,
    L: ExtField<F>,
    BuildSuffix:
        FnOnce(RecursiveProverState<F, L>) -> Result<RecursiveSuffixOutcome<F, L>, AkitaError>,
{
    let RootLevelProverOutput {
        #[cfg(feature = "zk")]
        zk_hiding_commitment,
        raw,
        extra_carried_sources,
        extra_carried_openings,
        next_state,
    } = root;
    let suffix = build_suffix(next_state)?;
    let RecursiveSuffixOutcome {
        steps,
        #[cfg(feature = "zk")]
        zk_hiding,
        num_levels,
    } = suffix;
    #[cfg(feature = "zk")]
    let zk_hiding = zk_hiding.into_proof(zk_hiding_commitment)?;
    let mut root_proof = AkitaBatchedRootProof::new::<D>(raw);
    if !extra_carried_sources.is_empty() || !extra_carried_openings.is_empty() {
        if let Some(fold) = root_proof.as_fold_mut() {
            fold.stage2.extra_carried_sources = extra_carried_sources;
            fold.stage2.extra_carried_openings = extra_carried_openings;
        }
    }
    Ok((
        AkitaBatchedProof {
            #[cfg(feature = "zk")]
            zk_hiding,
            root: root_proof,
            steps,
        },
        num_levels,
    ))
}

/// Assemble the 1-fold batched proof when the root level is itself the
/// terminal fold (no recursive suffix follows).
pub fn build_terminal_root_batched_proof<F, L>(
    #[cfg(feature = "zk")] zk_hiding: ZkHidingProof<F>,
    terminal: TerminalLevelProof<F, L>,
) -> AkitaBatchedProof<F, L>
where
    F: FieldCore,
    L: ExtField<F>,
{
    AkitaBatchedProof {
        #[cfg(feature = "zk")]
        zk_hiding,
        root: AkitaBatchedRootProof::new_terminal(terminal),
        steps: Vec::new(),
    }
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
pub fn prove_folded_batched<'a, Cfg, T, P, B, const D: usize>(
    expanded: &Arc<AkitaExpandedSetup<Cfg::Field>>,
    prefix_slots: &SetupPrefixProverRegistry<Cfg::Field, D>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    prepared_claims: PreparedBatchedProveInputs<'a, Cfg::Field, Cfg::ClaimField, P, D>,
    schedule: &Schedule,
    basis: BasisMode,
    root_next_params: &LevelParams,
    setup_contribution_mode: SetupContributionMode,
) -> Result<(AkitaBatchedProof<Cfg::Field, Cfg::ChallengeField>, usize), AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    Cfg::ClaimField: RingSubfieldEncoding<Cfg::Field> + MulBaseUnreduced<Cfg::Field>,
    Cfg::ChallengeField: RingSubfieldEncoding<Cfg::Field>
        + ExtField<Cfg::ClaimField>
        + ExtField<Cfg::Field>
        + FrobeniusExtField<Cfg::Field>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<Cfg::Field>,
    P: AkitaPolyOps<Cfg::Field, D>,
    B: ProverComputeBackend<Cfg::Field>,
{
    backend.validate_prepared_setup::<D>(prepared, expanded.as_ref())?;

    let Some(root_step) = schedule_root_fold_step(schedule) else {
        return Err(AkitaError::InvalidSetup(
            "root schedule does not start with a fold".to_string(),
        ));
    };

    if prepared_claims
        .commitments_by_point
        .iter()
        .any(|commitment| commitment.u.len() != root_step.params.effective_commit_rows())
    {
        return Err(AkitaError::InvalidInput(
            "batched_prove received a commitment with the wrong length".to_string(),
        ));
    }

    let num_vars = prepared_claims.incidence_summary.num_vars();

    #[cfg(feature = "zk")]
    let (zk_hiding_commitment, mut zk_hiding_state) =
        build_zk_hiding_context::<Cfg::Field, Cfg::ClaimField, Cfg::ChallengeField, B, D>(
            backend,
            prepared,
            schedule,
            &root_step.params,
            prepared_claims.incidence_summary.num_vars(),
            prepared_claims.incidence_summary.num_claims(),
            prepared_claims.incidence_summary.num_points(),
        )?;
    #[cfg(feature = "zk")]
    transcript.append_serde(ABSORB_ZK_HIDING_COMMITMENT, &zk_hiding_commitment.u_blind);

    if schedule_num_fold_levels(schedule) == 1 {
        // Root is itself the terminal fold: no recursive suffix.
        let direct_step = match schedule.steps.get(1) {
            Some(Step::Direct(direct_step)) => direct_step.clone(),
            _ => {
                return Err(AkitaError::InvalidSetup(
                    "1-fold schedule must terminate in a direct step".to_string(),
                ));
            }
        };
        let final_log_basis = match direct_step.witness_shape {
            CleartextWitnessShape::PackedDigits((_, bits)) => bits,
            CleartextWitnessShape::FieldElements(_) => {
                return Err(AkitaError::InvalidSetup(
                    "terminal root requires a packed-digit direct step".to_string(),
                ));
            }
        };
        let _ = root_next_params;
        let terminal = prove_terminal_root_fold_with_params::<
            Cfg::Field,
            Cfg::ClaimField,
            Cfg::ChallengeField,
            T,
            P,
            B,
            D,
        >(
            expanded.as_ref(),
            backend,
            prepared,
            transcript,
            &prepared_claims.flat_polys,
            &prepared_claims.incidence_summary,
            &prepared_claims.opening_points,
            &prepared_claims.commitments_by_point,
            prepared_claims.flat_hints,
            &root_step.params,
            root_step.next_w_len,
            final_log_basis,
            basis,
            setup_contribution_mode,
            #[cfg(feature = "zk")]
            &mut zk_hiding_state,
        )?;
        #[cfg(feature = "zk")]
        let zk_hiding_proof = zk_hiding_state.into_proof(zk_hiding_commitment)?;
        return Ok((
            build_terminal_root_batched_proof::<Cfg::Field, Cfg::ChallengeField>(
                #[cfg(feature = "zk")]
                zk_hiding_proof,
                terminal,
            ),
            1,
        ));
    }

    let root = prove_root_fold_with_params::<
        Cfg::Field,
        Cfg::ClaimField,
        Cfg::ChallengeField,
        T,
        P,
        B,
        Cfg,
        D,
    >(
        expanded,
        prefix_slots,
        backend,
        prepared,
        transcript,
        &prepared_claims.flat_polys,
        &prepared_claims.incidence_summary,
        &prepared_claims.opening_points,
        &prepared_claims.commitments_by_point,
        prepared_claims.flat_hints,
        &root_step.params,
        root_step.next_w_len,
        root_next_params,
        #[cfg(feature = "zk")]
        zk_hiding_commitment,
        #[cfg(feature = "zk")]
        zk_hiding_state,
        basis,
        setup_contribution_mode,
    )?;
    let RootLevelProverOutput {
        #[cfg(feature = "zk")]
        zk_hiding_commitment,
        raw,
        extra_carried_sources,
        extra_carried_openings,
        next_state: starting_state,
    } = root;
    let mut root = AkitaBatchedRootProof::new::<D>(raw);
    if !extra_carried_sources.is_empty() || !extra_carried_openings.is_empty() {
        if let Some(fold) = root.as_fold_mut() {
            fold.stage2.extra_carried_sources = extra_carried_sources;
            fold.stage2.extra_carried_openings = extra_carried_openings;
        }
    }

    let suffix = crate::prove_suffix::<Cfg, T, B, D>(
        expanded,
        prefix_slots,
        backend,
        prepared,
        num_vars,
        transcript,
        starting_state,
        schedule,
        setup_contribution_mode,
    )?;
    let RecursiveSuffixOutcome {
        steps,
        #[cfg(feature = "zk")]
        zk_hiding,
        num_levels,
    } = suffix;
    #[cfg(feature = "zk")]
    let zk_hiding = zk_hiding.into_proof(zk_hiding_commitment)?;
    Ok((
        AkitaBatchedProof {
            #[cfg(feature = "zk")]
            zk_hiding,
            root,
            steps,
        },
        num_levels,
    ))
}
