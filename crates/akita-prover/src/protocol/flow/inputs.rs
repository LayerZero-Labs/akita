use super::*;

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

/// Drive batched proving up to the config-selected folded-root policy.
///
/// This owns the config-free top-level prover work: validate/flatten public
/// prover claims, derive the schedule lookup key, select the schedule through
/// the supplied policy callback, apply the root-direct shortcut when the
/// selected schedule says no fold is needed, and derive the first recursive
/// schedule inputs for folded roots. Folded-root proving still runs in the
/// caller-supplied closure while config-selected recursive commitment layouts
/// remain outside this crate.
///
/// # Errors
///
/// Returns an error if claim preparation, schedule selection, root-direct
/// witness construction, root-next parameter selection, or folded-root proving
/// fails.
#[allow(clippy::too_many_arguments)]
pub fn prove_batched_with_policy<
    'a,
    F,
    E,
    L,
    T,
    P,
    const D: usize,
    SelectSchedule,
    SelectRootDirectParams,
    SelectRootNext,
    BindTranscript,
    ProveFolded,
>(
    expanded: &AkitaExpandedSetup<F>,
    claims: ProverClaims<'a, E, P, RingCommitment<F, D>, AkitaCommitmentHint<F, D>>,
    transcript: &mut T,
    basis: BasisMode,
    select_schedule: SelectSchedule,
    select_root_direct_params: SelectRootDirectParams,
    select_root_next_params: SelectRootNext,
    bind_transcript: BindTranscript,
    prove_folded: ProveFolded,
) -> Result<AkitaBatchedProof<F, L>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
    L: ExtField<F>,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D>,
    SelectSchedule: FnOnce(&ClaimIncidenceSummary) -> Result<Schedule, AkitaError>,
    SelectRootDirectParams: FnOnce(&ClaimIncidenceSummary) -> Result<LevelParams, AkitaError>,
    SelectRootNext: FnOnce(&Schedule, AkitaScheduleInputs) -> Result<LevelParams, AkitaError>,
    BindTranscript:
        FnOnce(&mut T, &ClaimIncidenceSummary, &Schedule, BasisMode) -> Result<(), AkitaError>,
    ProveFolded: FnOnce(
        PreparedBatchedProveInputs<'a, F, E, P, D>,
        Schedule,
        LevelParams,
        &mut T,
        BasisMode,
    ) -> Result<AkitaBatchedProof<F, L>, AkitaError>,
{
    let prepared_claims = prepare_batched_prove_inputs::<F, E, P, D>(expanded, claims)?;
    let num_vars = prepared_claims.incidence_summary.num_vars();
    let mut schedule = select_schedule(&prepared_claims.incidence_summary)?;
    if let Some(root_step) = schedule_root_fold_step(&schedule) {
        let alpha_bits = root_step.params.ring_dimension.trailing_zeros() as usize;
        if !folded_root_supports_opening_shape::<F, E, L, D>(
            &prepared_claims.opening_points,
            &root_step.params,
            alpha_bits,
        ) && !root_tensor_projection_enabled::<F, E, L, D>(num_vars)
        {
            let commit_params = select_root_direct_params(&prepared_claims.incidence_summary)?;
            schedule = root_direct_schedule(num_vars, commit_params)?;
        }
    }

    bind_transcript(
        transcript,
        &prepared_claims.incidence_summary,
        &schedule,
        basis,
    )?;

    if schedule_is_root_direct(&schedule) {
        return prove_root_direct::<F, L, D, P>(
            &prepared_claims.group_polys,
            &prepared_claims.flat_hints,
        );
    }

    let Some(root_step) = schedule_root_fold_step(&schedule) else {
        return Err(AkitaError::InvalidSetup(
            "root schedule does not start with a fold".to_string(),
        ));
    };
    let next_inputs = AkitaScheduleInputs {
        num_vars,
        level: 1,
        current_w_len: root_step.next_w_len,
    };
    let root_next_params = select_root_next_params(&schedule, next_inputs)?;

    prove_folded(
        prepared_claims,
        schedule,
        root_next_params,
        transcript,
        basis,
    )
}

/// Build the recursive suffix from an intermediate-root handoff, then
/// assemble the final folded batched proof.
///
/// The caller owns suffix schedule/config policy inside `build_suffix`; this
/// helper owns the config-free handoff from root raw output into suffix
/// construction and final proof assembly.
///
/// # Errors
///
/// Returns an error if suffix construction fails.
pub fn build_folded_batched_proof_with_suffix<F, L, const D: usize, BuildSuffix>(
    raw: RootLevelRawOutput<F, L, D>,
    build_suffix: BuildSuffix,
) -> Result<(AkitaBatchedProof<F, L>, usize), AkitaError>
where
    F: FieldCore,
    L: ExtField<F>,
    BuildSuffix:
        FnOnce(RecursiveProverState<F, L>) -> Result<RecursiveSuffixOutcome<F, L>, AkitaError>,
{
    let RootLevelRawOutput {
        #[cfg(feature = "zk")]
        zk_hiding_commitment,
        y_rings,
        extension_opening_reduction,
        v,
        stage1,
        #[cfg(not(feature = "zk"))]
        stage2_sumcheck_proof,
        #[cfg(feature = "zk")]
        stage2_sumcheck_proof_masked,
        w_commitment_proof,
        w_eval,
        extra_carried_sources,
        extra_carried_openings,
        next_state,
    } = raw;
    let suffix = build_suffix(next_state)?;
    let RecursiveSuffixOutcome {
        intermediate_levels,
        terminal,
        #[cfg(feature = "zk")]
        zk_hiding,
        num_levels,
    } = suffix;
    #[cfg(feature = "zk")]
    let zk_hiding = zk_hiding.into_proof(zk_hiding_commitment)?;
    let mut root = AkitaBatchedRootProof::new_two_stage_with_extension_opening_reduction::<D>(
        y_rings,
        extension_opening_reduction,
        v,
        stage1,
        #[cfg(not(feature = "zk"))]
        stage2_sumcheck_proof,
        #[cfg(feature = "zk")]
        stage2_sumcheck_proof_masked,
        w_commitment_proof,
        w_eval,
    );
    if let AkitaBatchedRootProof::Fold(fold_root) = &mut root {
        fold_root.stage2.extra_carried_sources = extra_carried_sources;
        fold_root.stage2.extra_carried_openings = extra_carried_openings;
    }
    let steps = build_final_proof_steps::<F, L>(intermediate_levels, terminal);
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

/// Prove a folded batched root and assemble the recursive suffix.
///
/// The prover crate owns config-free folded-root preparation: root schedule
/// shape checks, opening-point reduction, commitment row shape validation,
/// root fold proving, recursive suffix handoff, and final proof assembly. The
/// caller supplies the already-selected first recursive commitment params plus
/// policy callbacks for committing root's next `w` and proving the suffix.
///
/// # Errors
///
/// Returns an error if the schedule is not folded, root inputs are malformed,
/// root proving fails, or suffix construction fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_folded_batched_with_policy<
    'a,
    F,
    E,
    C,
    T,
    P,
    B,
    const D: usize,
    CommitRootNext,
    BuildSuffix,
    AdjustRaw,
>(
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    transcript: &mut T,
    prepared_claims: PreparedBatchedProveInputs<'a, F, E, P, D>,
    schedule: &Schedule,
    basis: BasisMode,
    root_next_params: &LevelParams,
    commit_root_next: CommitRootNext,
    build_suffix: BuildSuffix,
    adjust_raw: AdjustRaw,
) -> Result<(AkitaBatchedProof<F, C>, usize), AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasUnreducedOps
        + HasWide
        + HalvingField
        + Invertible
        + PseudoMersenneField,
    E: RingSubfieldEncoding<F>,
    C: RingSubfieldEncoding<F>
        + ExtField<E>
        + ExtField<F>
        + FrobeniusExtField<F>
        + HasUnreducedOps
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D>,
    B: ProverComputeBackend<F>,
    CommitRootNext: FnOnce(&RecursiveWitnessFlat) -> Result<NextWitnessCommitment<F>, AkitaError>,
    BuildSuffix: FnOnce(
        RecursiveProverState<F, C>,
        &Schedule,
        &mut T,
    ) -> Result<RecursiveSuffixOutcome<F, C>, AkitaError>,
    AdjustRaw: FnOnce(&mut RootLevelRawOutput<F, C, D>) -> Result<(), AkitaError>,
{
    backend.validate_prepared_setup::<D>(prepared, expanded)?;

    let Some(root_step) = schedule_root_fold_step(schedule) else {
        return Err(AkitaError::InvalidSetup(
            "root schedule does not start with a fold".to_string(),
        ));
    };

    if prepared_claims
        .commitments_by_point
        .iter()
        .any(|commitment| commitment.u.len() != root_step.params.b_key.row_len())
    {
        return Err(AkitaError::InvalidInput(
            "batched_prove received a commitment with the wrong length".to_string(),
        ));
    }

    #[cfg(feature = "zk")]
    let (zk_hiding_commitment, mut zk_hiding_state) = build_zk_hiding_context::<F, E, C, B, D>(
        backend,
        prepared,
        schedule,
        &root_step.params,
        prepared_claims.incidence_summary.num_vars(),
        prepared_claims.incidence_summary.num_claims(),
        prepared_claims.incidence_summary.num_public_rows(),
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
        let _ = (commit_root_next, build_suffix, root_next_params, adjust_raw);
        let terminal = prove_terminal_root_fold_with_params::<F, E, C, T, P, B, D>(
            expanded,
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
            #[cfg(feature = "zk")]
            &mut zk_hiding_state,
        )?;
        #[cfg(feature = "zk")]
        let zk_hiding_proof = zk_hiding_state.into_proof(zk_hiding_commitment)?;
        return Ok((
            build_terminal_root_batched_proof::<F, C>(
                #[cfg(feature = "zk")]
                zk_hiding_proof,
                terminal,
            ),
            1,
        ));
    }

    let mut raw = prove_root_fold_with_params::<F, E, C, T, P, B, D, _>(
        expanded,
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
        root_next_params.log_basis,
        #[cfg(feature = "zk")]
        zk_hiding_commitment,
        #[cfg(feature = "zk")]
        zk_hiding_state,
        basis,
        |w| commit_root_next(w),
    )?;
    adjust_raw(&mut raw)?;

    build_folded_batched_proof_with_suffix::<F, C, D, _>(raw, |next_state| {
        build_suffix(next_state, schedule, transcript)
    })
}
