use super::*;
use akita_types::Commitment;

/// Verify the folded root proof payload.
///
/// This replays the canonical root transcript layout: batch-shape header,
/// commitments, padded opening points, per-claim field openings, row
/// EOR if present, the complete opening payload, native public row coefficients,
/// y-rings, ring switch, stage-1 when present, stage-2, and stage-3 setup
/// sumcheck when required by the intermediate branch. Extension-field EOR
/// retains its earlier internally coupled row coefficients.
///
/// # Errors
///
/// Returns an error if the proof shape is inconsistent, any public trace check
/// fails, ring-switch replay fails, or a sumcheck verifier rejects.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(super) fn verify_root<F, E, T>(
    proof: &FoldLevelProof<F, E>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    claims: &OpeningClaims<'_, E, &Commitment<F>>,
    opening_batch: &OpeningClaimsLayout,
    basis: BasisMode,
    root_lp: &CommittedGroupParams,
    next_fold_params: Option<&FoldParams>,
    next_witness_ring_dim: usize,
    next_t_state: Option<&[u8]>,
) -> Result<FoldVerifyOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HalvingField,
    E: FpExtEncoding<F>
        + ExtField<F>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + AkitaSerialize
        + MulBaseUnreduced<F>,
    T: akita_types::VerifierTranscriptGrinding<F>,
{
    if proof.extension_opening_reduction().is_some()
        || root_lp.source_encoding
            != akita_types::CommittedSourceEncoding::CanonicalCoefficientTable
    {
        return Err(AkitaError::InvalidProof);
    }
    root_lp.validate_opening_batch(opening_batch)?;
    for group_index in 0..opening_batch.num_groups() {
        if !matches!(
            root_lp
                .group_params_geometry(opening_batch, group_index)?
                .opening_method(),
            akita_types::OpeningMethod::SubringCoefficientPacking { .. }
        ) {
            return Err(AkitaError::InvalidProof);
        }
    }
    let setup_contribution_mode = next_fold_params
        .map_or(SetupContributionMode::Direct, |params| {
            params.predecessor_setup_contribution_mode()
        });
    let next_fold_level_params = next_fold_params.map(|params| &params.params);
    let stage3_sumcheck_proof = proof
        .stage3_for_mode(setup_contribution_mode, next_fold_level_params)?
        .map(|(proof, _)| proof);
    let next_witness = match (proof.next_w_payload(), next_t_state) {
        (Some(commitment), None) => {
            let next_params = next_fold_level_params.ok_or(AkitaError::InvalidProof)?;
            let ring_dim = next_params
                .outer_payload_geometry()?
                .transcript_ring_dimension();
            PreparedNextWitness::Commitment {
                commitment,
                ring_dim,
            }
        }
        (None, Some(t_state)) if !t_state.is_empty() => PreparedNextWitness::TerminalT(t_state),
        _ => return Err(AkitaError::InvalidProof),
    };
    let openings = claims.flat_evaluations();
    let num_claims = opening_batch.num_total_polynomials();
    if openings.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    // Transcript binding, D-free and byte-identical to the prover's absorb
    // (`ProverOpeningData::append_to_transcript`): batch shape header, then each
    // group commitment's flat coefficients under `ring_dim` in `OpeningClaims`
    // order, then each group's complete opening point. Each group's committed row count is
    // validated against its (final vs frozen-precommit) params before the
    // absorb, so a swapped/truncated group commitment rejects here.
    opening_batch.append_batch_shape_to_transcript::<F, T>(transcript)?;
    let relation_geometry =
        RelationWitnessGeometry::for_level(root_lp, opening_batch, E::EXT_DEGREE)?;
    let relation_layout = relation_geometry.rhs_layout();
    for group_index in 0..opening_batch.num_groups() {
        let commitment = claims.group_commitment(group_index)?;
        let plan = relation_layout.compression_plan_for_group(group_index)?;
        if commitment.rows().coeff_len() != plan.terminal_coefficients() {
            return Err(AkitaError::InvalidProof);
        }
        let ring_dim = plan
            .maps()
            .last()
            .ok_or(AkitaError::InvalidProof)?
            .ring_dimension();
        commitment.append_to_transcript(ABSORB_COMMITMENT, ring_dim, transcript)?;
    }
    for group in claims.groups() {
        for coord in group.point() {
            append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, coord);
        }
    }
    append_claim_values_to_transcript::<F, E, T>(&openings, transcript);

    // D-free root replay: typed kernels dispatch inside `verify_fold` and the
    // geometry prefix modules on per-role dimensions. A scalar root is the
    // one-group case of the same grouped layout; grouped roots (`G > 1`) never
    // collapse into a synthetic single group.
    verify_root_inner::<F, E, T>(
        proof,
        setup,
        transcript,
        claims,
        &openings,
        opening_batch,
        stage3_sumcheck_proof,
        next_fold_level_params,
        next_witness_ring_dim,
        basis,
        root_lp,
        next_witness,
    )
}

/// Root-fold replay orchestrator (D-free).
///
/// Reached from [`verify_root`]; per-role typed kernels dispatch inside
/// [`verify_fold`] and the geometry prefix modules. Geometry forks only the
/// prefix (single-field vs extension-claim), both producing a
/// [`FoldPrefix`]; [`PreparedFoldReplay`] assembly is shared.
///
/// This builds one prepared opening point per group (mirroring the prover's
/// `finish_prepared_fold` loop and its per-group padded-point absorbs),
/// concatenates the group commitment rows in relation-matrix row (final-first)
/// order, sizes the next witness from the grouped witness layout, and hands a
/// per-group `PreparedFoldReplay` to [`verify_fold`]. Extension-field groups
/// share one EOR sumcheck while retaining group-local opening geometry.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidProof`] for a non-fold root or malformed group
/// shape, and propagates layout/replay errors.
#[allow(clippy::too_many_arguments)]
fn verify_root_inner<F, E, T>(
    proof: &FoldLevelProof<F, E>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    claims: &OpeningClaims<'_, E, &Commitment<F>>,
    openings: &[E],
    opening_batch: &OpeningClaimsLayout,
    stage3_sumcheck_proof: Option<&SetupSumcheckProof<E>>,
    next_fold_level_params: Option<&CommittedGroupParams>,
    next_witness_ring_dim: usize,
    basis: BasisMode,
    root_lp: &CommittedGroupParams,
    next_witness: PreparedNextWitness<'_, F>,
) -> Result<FoldVerifyOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HalvingField,
    E: FpExtEncoding<F>
        + ExtField<F>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + AkitaSerialize
        + MulBaseUnreduced<F>,
    T: akita_types::VerifierTranscriptGrinding<F>,
{
    let claim_material = verify_coefficient_packing_root_prefix::<F, E>(
        claims,
        openings,
        opening_batch,
        basis,
        root_lp,
    )?;
    // Concatenate group commitment rows in relation-matrix row (final-first) order, matching
    // the prover's `RingRelationProver` commitment-row concatenation and
    // `RelationWitnessGeometry` block order.
    let order = opening_batch.root_group_order()?;
    let mut commitment_payloads = Vec::with_capacity(order.len());
    for &group_index in &order {
        let commitment = claims.group_commitment(group_index)?;
        commitment_payloads.push(commitment.rows().clone());
    }

    let witness_len = root_lp.output_witness_len::<F>(opening_batch, E::EXT_DEGREE)?;
    let opening_payload = proof.opening_payload.clone();
    let prefix = bind_opening_payload_and_finalize_claims(
        root_lp,
        opening_batch,
        &opening_payload,
        claim_material,
        transcript,
        0,
    )?;
    let committed_witness_len =
        akita_types::witness_commitment_domain_len(witness_len, next_witness_ring_dim)?;
    let prepared = PreparedFoldReplay {
        lp: root_lp,
        level: 0,
        opening_payload,
        opening_shape: opening_batch.clone(),
        commitment_payloads,
        prefix,
        w_len: witness_len,
        payload: PreparedFoldPayload::Recursive {
            stage1: &proof.stage1,
            stage2: &proof.stage2,
            next_witness,
            next_witness_ring_dim,
            next_opening_source_len: committed_witness_len / next_witness_ring_dim,
            stage3: stage3_sumcheck_proof.zip(next_fold_level_params),
        },
        evaluation_trace_basis: basis,
    };
    verify_fold::<F, E, T>(setup, transcript, prepared).map_err(|error| {
        AkitaError::InvalidInput(format!("compressed root fold failed: {error:?}"))
    })
}
