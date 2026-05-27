//! Top-level batched verifier orchestration once a schedule is selected.

use super::{prepare_verifier_claims, validate_level_dispatch, validate_log_basis};
use crate::proof::claims::PreparedVerifierClaims;
use crate::proof::direct::verify_root_direct_openings_with_incidence;
use crate::protocol::levels::verify_fold_batched_proof;
use akita_algebra::CyclotomicRing;
use akita_config::{bind_transcript_instance_descriptor, CommitmentConfig};
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FrobeniusExtField, FromPrimitiveInt,
    PseudoMersenneField, RandomSampling,
};
use akita_serialization::AkitaSerialize;
use akita_transcript::Transcript;
use akita_types::{
    folded_root_supports_opening_shape, root_direct_schedule, root_tensor_projection_enabled,
    schedule_is_root_direct, schedule_root_fold_step, scheduled_next_level_params,
    AkitaBatchedProof, AkitaBatchedRootProof, AkitaProofStep, AkitaVerifierSetup, BasisMode,
    ClaimIncidenceSummary, DirectWitnessProof, LevelParams, RingCommitment, RingSubfieldEncoding,
    Schedule, SetupRoleDimensions, VerifierClaims,
};
use std::array::from_fn;

#[cfg(feature = "zk")]
/// Root-direct commitment blinding payload carried by zk proofs.
pub type RootDirectBlindingPayload<'a> = &'a [Vec<i8>];
#[cfg(not(feature = "zk"))]
/// Typed empty root-direct commitment blinding payload for transparent builds.
pub struct NoRootDirectBlindingPayload;
#[cfg(not(feature = "zk"))]
/// Borrowed transparent-build placeholder for root-direct blinding payloads.
pub type RootDirectBlindingPayload<'a> = &'a NoRootDirectBlindingPayload;

/// Structural slice of `<AkitaBatchedProof as Valid>::check`, inlined to avoid
/// requiring `F: Valid + L: Valid` at the verifier entrypoint.
pub(crate) fn check_batched_proof_step_shape<F, L>(
    proof: &AkitaBatchedProof<F, L>,
) -> Result<(), AkitaError>
where
    F: FieldCore,
    L: FieldCore,
{
    match &proof.root {
        AkitaBatchedRootProof::Fold(_) => {
            let Some((last, rest)) = proof.steps.split_last() else {
                return Err(AkitaError::InvalidProof);
            };
            if !matches!(last, AkitaProofStep::Terminal(_))
                || rest
                    .iter()
                    .any(|step| !matches!(step, AkitaProofStep::Intermediate(_)))
            {
                return Err(AkitaError::InvalidProof);
            }
        }
        AkitaBatchedRootProof::Terminal(_) => {
            if !proof.steps.is_empty() {
                return Err(AkitaError::InvalidProof);
            }
        }
        AkitaBatchedRootProof::Direct { .. } => {
            if !proof.steps.is_empty() {
                return Err(AkitaError::InvalidProof);
            }
        }
    }
    Ok(())
}

fn i8_plane_to_ring<F, const D: usize>(plane: &[i8; D]) -> CyclotomicRing<F, D>
where
    F: FieldCore + FromPrimitiveInt,
{
    CyclotomicRing::from_coefficients(from_fn(|idx| F::from_i64(plane[idx] as i64)))
}

pub(crate) fn field_evals_to_rings<F, const D: usize>(
    evals: &[F],
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore,
{
    if D == 0 || !D.is_power_of_two() || !evals.len().is_power_of_two() {
        return Err(AkitaError::InvalidProof);
    }
    Ok(evals
        .chunks(D)
        .map(|chunk| {
            CyclotomicRing::from_coefficients(from_fn(|idx| {
                chunk.get(idx).copied().unwrap_or_else(F::zero)
            }))
        })
        .collect())
}

fn checked_root_direct_witness_rings<const D: usize>(
    witness_len: usize,
    num_vars: usize,
    params: &LevelParams,
) -> Result<usize, AkitaError> {
    if D == 0 {
        return Err(AkitaError::InvalidSetup(
            "ring dimension must be non-zero".to_string(),
        ));
    }
    if !witness_len.is_power_of_two() {
        return Err(AkitaError::InvalidProof);
    }
    let expected_len = 1usize
        .checked_shl(u32::try_from(num_vars).map_err(|_| AkitaError::InvalidProof)?)
        .ok_or(AkitaError::InvalidProof)?;
    if witness_len != expected_len {
        return Err(AkitaError::InvalidProof);
    }
    let witness_rings = witness_len.div_ceil(D);
    let capacity = params
        .num_blocks
        .checked_mul(params.block_len)
        .ok_or_else(|| AkitaError::InvalidSetup("direct witness capacity overflow".to_string()))?;
    if witness_rings > capacity {
        return Err(AkitaError::InvalidSetup(
            "direct witness exceeds selected verifier layout".to_string(),
        ));
    }
    Ok(witness_rings)
}

fn validate_root_direct_recommitment_shape<F, const D: usize>(
    witnesses: &[DirectWitnessProof<F>],
    setup: &AkitaVerifierSetup<F>,
    incidence_summary: &ClaimIncidenceSummary,
    params: &LevelParams,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    validate_level_dispatch::<D>(params)?;
    validate_log_basis(params.log_basis)?;
    if params.num_blocks == 0 || params.block_len == 0 {
        return Err(AkitaError::InvalidSetup(
            "direct witness layout requires non-zero block geometry".to_string(),
        ));
    }
    if params.num_digits_commit == 0 || params.num_digits_open == 0 {
        return Err(AkitaError::InvalidSetup(
            "direct witness layout requires non-zero digit depths".to_string(),
        ));
    }

    let total_claims =
        incidence_summary
            .num_polys_per_point()
            .iter()
            .try_fold(0usize, |acc, &count| {
                acc.checked_add(count).ok_or_else(|| {
                    AkitaError::InvalidSetup("direct claim count overflow".to_string())
                })
            })?;
    let role_dimensions = SetupRoleDimensions::for_batched_shape(
        params,
        incidence_summary.num_polys_per_point(),
        total_claims,
    )?;
    setup
        .expanded
        .a_setup_view::<D>(role_dimensions)
        .map(|_| ())?;
    setup
        .expanded
        .b_setup_view::<D>(role_dimensions)
        .map(|_| ())?;
    let mut claim_offset = 0usize;
    for &point_size in incidence_summary.num_polys_per_point() {
        let group_end = claim_offset
            .checked_add(point_size)
            .ok_or(AkitaError::InvalidProof)?;
        for witness in &witnesses[claim_offset..group_end] {
            let witness_len = witness
                .as_field_elements()
                .ok_or(AkitaError::InvalidProof)?
                .coeff_len();
            checked_root_direct_witness_rings::<D>(
                witness_len,
                incidence_summary.num_vars(),
                params,
            )?;
        }
        claim_offset = group_end;
    }
    Ok(())
}

pub(crate) fn mat_vec_mul_i8_plain<F, const D: usize>(
    matrix_rows: &[&[CyclotomicRing<F, D>]],
    digits: &[[i8; D]],
) -> Vec<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField,
{
    matrix_rows
        .iter()
        .map(|row| {
            row.iter()
                .zip(digits.iter())
                .fold(CyclotomicRing::<F, D>::zero(), |acc, (entry, digit)| {
                    acc + (*entry * i8_plane_to_ring::<F, D>(digit))
                })
        })
        .collect()
}

fn decompose_rows_i8<F, const D: usize>(
    rows: &[CyclotomicRing<F, D>],
    num_digits: usize,
    log_basis: u32,
) -> Vec<[i8; D]>
where
    F: FieldCore + CanonicalField,
{
    let mut out = vec![[0i8; D]; rows.len() * num_digits];
    for (dst_chunk, row) in out.chunks_mut(num_digits).zip(rows.iter()) {
        row.balanced_decompose_pow2_i8_into(dst_chunk, log_basis);
    }
    out
}

pub(crate) fn direct_decomposed_inner_rows<F, const D: usize>(
    witness_rings: &[CyclotomicRing<F, D>],
    setup: &AkitaVerifierSetup<F>,
    params: &LevelParams,
    role_dimensions: SetupRoleDimensions,
) -> Result<Vec<[i8; D]>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let a_matrix = setup.expanded.a_setup_view::<D>(role_dimensions)?;
    let a_rows: Vec<_> = a_matrix.rows().collect();
    let out_capacity = params
        .num_blocks
        .checked_mul(params.a_key.row_len())
        .and_then(|len| len.checked_mul(params.num_digits_open))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("direct witness row capacity overflow".to_string())
        })?;
    let mut out = Vec::with_capacity(out_capacity);

    for block_idx in 0..params.num_blocks {
        let start = block_idx.checked_mul(params.block_len).ok_or_else(|| {
            AkitaError::InvalidSetup("direct witness block offset overflow".to_string())
        })?;
        let end = start
            .saturating_add(params.block_len)
            .min(witness_rings.len());
        let block = if start < witness_rings.len() {
            &witness_rings[start..end]
        } else {
            &[]
        };
        let block_digits = decompose_rows_i8(block, params.num_digits_commit, params.log_basis);
        let t_rows = mat_vec_mul_i8_plain::<F, D>(&a_rows, &block_digits);
        out.extend(decompose_rows_i8(
            &t_rows,
            params.num_digits_open,
            params.log_basis,
        ));
    }

    Ok(out)
}

#[cfg(feature = "zk")]
pub(crate) fn direct_blinding_digit_planes<F, const D: usize>(
    revealed_b_blinding_digits: &[i8],
    params: &LevelParams,
) -> Result<Vec<[i8; D]>, AkitaError>
where
    F: CanonicalField,
{
    if D == 0 {
        return Err(AkitaError::InvalidProof);
    }
    let expected_planes =
        akita_types::zk::blinding_column_count::<F>(params.b_key.row_len(), D, params.log_basis);
    let expected_digits = expected_planes
        .checked_mul(D)
        .ok_or(AkitaError::InvalidProof)?;
    if revealed_b_blinding_digits.len() != expected_digits {
        return Err(AkitaError::InvalidProof);
    }
    Ok(revealed_b_blinding_digits
        .chunks_exact(D)
        .map(|chunk| {
            let mut plane = [0i8; D];
            plane.copy_from_slice(chunk);
            plane
        })
        .collect())
}

fn recommit_direct_witness_group<F, const D: usize>(
    #[cfg_attr(not(feature = "zk"), allow(unused_variables))] point_idx: usize,
    group_witnesses: &[DirectWitnessProof<F>],
    setup: &AkitaVerifierSetup<F>,
    params: &LevelParams,
    role_dimensions: SetupRoleDimensions,
    #[cfg(feature = "zk")] blinding_digits: &[i8],
) -> Result<RingCommitment<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + RandomSampling,
{
    let mut outer_input = Vec::new();
    for witness in group_witnesses {
        let field_witness = witness
            .as_field_elements()
            .ok_or(AkitaError::InvalidProof)?
            .coeffs();
        let witness_rings = field_evals_to_rings::<F, D>(field_witness)?;
        outer_input.extend(direct_decomposed_inner_rows(
            &witness_rings,
            setup,
            params,
            role_dimensions,
        )?);
    }

    #[cfg(feature = "zk")]
    let blinding_planes = direct_blinding_digit_planes::<F, D>(blinding_digits, params)?;

    let b_matrix = setup.expanded.b_setup_view::<D>(role_dimensions)?;
    let b_rows: Vec<_> = b_matrix.rows().collect();
    #[cfg_attr(not(feature = "zk"), allow(unused_mut))]
    let mut u = mat_vec_mul_i8_plain::<F, D>(&b_rows, &outer_input);
    #[cfg(feature = "zk")]
    {
        let blinding_rows = akita_types::zk::b_blinding_negacyclic_rows::<F, D>(
            &setup.expanded.seed.zk_blinding_seed,
            point_idx,
            params.b_key.row_len(),
            &blinding_planes,
        );
        for (row, blinding_row) in u.iter_mut().zip(blinding_rows) {
            *row += blinding_row;
        }
    }
    Ok(RingCommitment { u })
}

/// Recompute root-direct commitments from direct witnesses and compare them to
/// the proof commitments.
///
/// # Errors
///
/// Returns an error if the direct witness shape does not match the batch shape,
/// if witness reconstruction fails, or if any recomputed commitment differs
/// from the proof commitment.
pub fn verify_root_direct_commitments_with_params<F, const D: usize>(
    witnesses: &[DirectWitnessProof<F>],
    setup: &AkitaVerifierSetup<F>,
    flat_commitments: &[RingCommitment<F, D>],
    incidence_summary: &ClaimIncidenceSummary,
    params: &LevelParams,
    b_blinding_digits: RootDirectBlindingPayload<'_>,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + RandomSampling + PseudoMersenneField,
{
    if flat_commitments.len() != incidence_summary.num_points() {
        return Err(AkitaError::InvalidProof);
    }
    if incidence_summary.num_polys_per_point().len() != incidence_summary.num_points() {
        return Err(AkitaError::InvalidProof);
    }
    #[cfg(feature = "zk")]
    if b_blinding_digits.len() != flat_commitments.len() {
        return Err(AkitaError::InvalidProof);
    }
    #[cfg(not(feature = "zk"))]
    let _ = b_blinding_digits;
    let total_group_polys = incidence_summary
        .num_polys_per_point()
        .iter()
        .try_fold(0usize, |acc, &count| {
            acc.checked_add(count).ok_or(AkitaError::InvalidProof)
        })?;
    if total_group_polys != witnesses.len() {
        return Err(AkitaError::InvalidProof);
    }
    validate_root_direct_recommitment_shape::<F, D>(witnesses, setup, incidence_summary, params)?;
    let total_claims = incidence_summary
        .num_polys_per_point()
        .iter()
        .try_fold(0usize, |acc, &count| {
            acc.checked_add(count).ok_or(AkitaError::InvalidProof)
        })?;
    let role_dimensions = SetupRoleDimensions::for_batched_shape(
        params,
        incidence_summary.num_polys_per_point(),
        total_claims,
    )?;

    let mut claim_offset = 0usize;
    let mut expected_commitments = Vec::with_capacity(incidence_summary.num_points());
    for (group_idx, &group_size) in incidence_summary.num_polys_per_point().iter().enumerate() {
        #[cfg(not(feature = "zk"))]
        let _ = group_idx;
        let group_witnesses = &witnesses[claim_offset..claim_offset + group_size];
        let commitment = recommit_direct_witness_group::<F, D>(
            group_idx,
            group_witnesses,
            setup,
            params,
            role_dimensions,
            #[cfg(feature = "zk")]
            &b_blinding_digits[group_idx],
        )?;
        expected_commitments.push(commitment);
        claim_offset += group_size;
    }

    if expected_commitments != flat_commitments {
        return Err(AkitaError::InvalidProof);
    }

    Ok(())
}

/// Schedule-derived layouts needed by the folded-root verifier branch.
pub(crate) struct FoldVerifierLayouts {
    /// Root verifier layout selected by the folded proof schedule.
    pub(crate) root_lp: LevelParams,
    /// First recursive-level params reached by the root fold.
    pub(crate) next_level_params: LevelParams,
}

/// Schedule context selected by the root scheme/config layer.
pub(crate) enum BatchedVerifierScheduleContext {
    /// The selected schedule uses the root-direct fast path.
    RootDirect,
    /// The selected schedule starts with a folded root.
    Fold(Box<FoldVerifierLayouts>),
}

/// Build the verifier schedule context for an already-selected proof schedule.
///
/// Root config policy supplies the recursive layout callback; this helper owns
/// only the public schedule shape interpretation needed by verifier replay.
///
/// # Errors
///
/// Returns an error if the schedule is empty or the supplied recursive layout
/// callback rejects the selected folded-root schedule.
pub(crate) fn prepare_batched_verifier_schedule_context(
    schedule: &Schedule,
) -> Result<BatchedVerifierScheduleContext, AkitaError> {
    if schedule_is_root_direct(schedule) {
        Ok(BatchedVerifierScheduleContext::RootDirect)
    } else if let Some(root_step) = schedule_root_fold_step(schedule) {
        let next_level_params = scheduled_next_level_params(schedule, 1)?;
        Ok(BatchedVerifierScheduleContext::Fold(Box::new(
            FoldVerifierLayouts {
                root_lp: root_step.params.clone(),
                next_level_params,
            },
        )))
    } else {
        Err(AkitaError::InvalidProof)
    }
}

/// Verify a batched proof after root schedule selection.
///
/// This owns the root-proof variant dispatch, direct witness/opening checks,
/// folded-root replay, and recursive suffix replay. The caller supplies only
/// the config-derived schedule context and a callback for root-direct
/// commitment recomputation.
///
/// # Errors
///
/// Returns an error if the proof shape disagrees with the schedule context,
/// direct openings fail, direct commitment recomputation fails, or folded-root
/// verification rejects.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_batched_proof_with_schedule<
    'a,
    F,
    E,
    C,
    T,
    const D: usize,
    DirectCommitmentCheck,
>(
    proof: &AkitaBatchedProof<F, C>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    prepared_claims: PreparedVerifierClaims<'a, E, RingCommitment<F, D>>,
    basis: BasisMode,
    schedule: &Schedule,
    schedule_context: BatchedVerifierScheduleContext,
    verify_direct_commitments: DirectCommitmentCheck,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField,
    E: RingSubfieldEncoding<F>,
    C: RingSubfieldEncoding<F>
        + ExtField<E>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
    DirectCommitmentCheck: FnOnce(
        &[DirectWitnessProof<F>],
        &[RingCommitment<F, D>],
        &ClaimIncidenceSummary,
        RootDirectBlindingPayload<'_>,
    ) -> Result<(), AkitaError>,
{
    let PreparedVerifierClaims {
        opening_points,
        commitments,
        openings,
        incidence_summary,
    } = prepared_claims;

    match &proof.root {
        AkitaBatchedRootProof::Direct { witnesses, .. } => {
            #[cfg(feature = "zk")]
            if !proof.zk_hiding.is_empty() {
                return Err(AkitaError::InvalidProof);
            }
            if !proof.steps.is_empty() {
                return Err(AkitaError::InvalidProof);
            }
            if !schedule_is_root_direct(schedule)
                || !matches!(schedule_context, BatchedVerifierScheduleContext::RootDirect)
            {
                return Err(AkitaError::InvalidProof);
            }
            verify_root_direct_openings_with_incidence(
                witnesses,
                &opening_points,
                &openings,
                &incidence_summary,
                basis,
            )?;
            #[cfg(feature = "zk")]
            let direct_commitment_payload = proof
                .root
                .direct_b_blinding_digits()
                .ok_or(AkitaError::InvalidProof)?;
            #[cfg(not(feature = "zk"))]
            let direct_commitment_payload = &NoRootDirectBlindingPayload;
            verify_direct_commitments(
                witnesses,
                &commitments,
                &incidence_summary,
                direct_commitment_payload,
            )?;
        }
        AkitaBatchedRootProof::Fold(_) | AkitaBatchedRootProof::Terminal(_) => {
            let BatchedVerifierScheduleContext::Fold(layouts) = schedule_context else {
                return Err(AkitaError::InvalidProof);
            };
            verify_fold_batched_proof::<F, E, C, T, D>(
                proof,
                setup,
                transcript,
                &opening_points,
                &openings,
                &commitments,
                &incidence_summary,
                basis,
                schedule,
                &layouts.root_lp,
                &layouts.next_level_params,
            )?;
        }
    }

    Ok(())
}

/// Verify a batched Akita proof against `claims` and `setup`.
///
/// Closure-free `<Cfg>`-generic verifier entry point: every policy hook is
/// sourced from `Cfg` (schedule selection, recursive successor params,
/// descriptor binding, and root-direct commitment recheck).
///
/// # Errors
///
/// Returns an error when public claims are malformed, when the configured
/// schedule rejects the proof shape, when the descriptor binding fails, or
/// when proof replay rejects the proof.
pub fn verify_batched<F, Cfg, T, const D: usize>(
    proof: &AkitaBatchedProof<F, Cfg::ChallengeField>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    claims: VerifierClaims<'_, Cfg::ClaimField, RingCommitment<F, D>>,
    basis: BasisMode,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField + AkitaSerialize,
    Cfg: CommitmentConfig<Field = F>,
    Cfg::ClaimField: RingSubfieldEncoding<F>,
    Cfg::ChallengeField: RingSubfieldEncoding<F>
        + ExtField<Cfg::ClaimField>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<F>,
{
    // Reject malformed step shapes that the downstream `fold_levels()` filter
    // would silently skip past.
    check_batched_proof_step_shape(proof)?;

    let prepared_claims = prepare_verifier_claims(&setup.expanded, &claims)?;
    let num_vars = prepared_claims.incidence_summary.num_vars();
    let mut schedule = Cfg::get_params_for_prove(&prepared_claims.incidence_summary)
        .map_err(|_| AkitaError::InvalidProof)?;
    if let Some(root_step) = schedule_root_fold_step(&schedule) {
        let alpha_bits = root_step.params.ring_dimension.trailing_zeros() as usize;
        if !folded_root_supports_opening_shape::<F, Cfg::ClaimField, Cfg::ChallengeField, D>(
            &prepared_claims.opening_points,
            &root_step.params,
            alpha_bits,
        ) && !root_tensor_projection_enabled::<F, Cfg::ClaimField, Cfg::ChallengeField, D>(
            num_vars,
        ) {
            let commit_params =
                Cfg::get_params_for_batched_commitment(&prepared_claims.incidence_summary)
                    .map_err(|_| AkitaError::InvalidProof)?;
            schedule = root_direct_schedule(num_vars, commit_params)
                .map_err(|_| AkitaError::InvalidProof)?;
        }
    }

    bind_transcript_instance_descriptor::<F, T, D, Cfg>(
        &setup.expanded,
        &prepared_claims.incidence_summary,
        &schedule,
        basis,
        transcript,
    )?;

    let schedule_context = prepare_batched_verifier_schedule_context(&schedule)
        .map_err(|_| AkitaError::InvalidProof)?;

    verify_batched_proof_with_schedule::<F, Cfg::ClaimField, Cfg::ChallengeField, T, D, _>(
        proof,
        setup,
        transcript,
        prepared_claims,
        basis,
        &schedule,
        schedule_context,
        |witnesses, commitments, incidence_summary, direct_commitment_payload| {
            let params = Cfg::get_params_for_batched_commitment(incidence_summary)
                .map_err(|_| AkitaError::InvalidProof)?;
            verify_root_direct_commitments_with_params::<F, D>(
                witnesses,
                setup,
                commitments,
                incidence_summary,
                &params,
                direct_commitment_payload,
            )
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Fp32;
    use akita_types::{
        AjtaiKeyParams, AkitaExpandedSetup, AkitaSetupSeed, FlatMatrix, FlatRingVec,
        SisModulusFamily,
    };
    use std::sync::Arc;

    type F = Fp32<251>;
    const D: usize = 32;

    fn stage1_config() -> SparseChallengeConfig {
        SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![1],
        }
    }

    fn incidence_summary(num_vars: usize) -> ClaimIncidenceSummary {
        ClaimIncidenceSummary::same_point(num_vars, 1).expect("valid incidence summary")
    }

    fn verifier_setup(seed: AkitaSetupSeed) -> AkitaVerifierSetup<F> {
        let field_len = seed.max_setup_len.checked_mul(D).expect("test setup fits");
        let shared_matrix = FlatMatrix::from_flat_data(vec![F::zero(); field_len], D);
        AkitaVerifierSetup {
            expanded: Arc::new(
                AkitaExpandedSetup::from_parts(seed, shared_matrix)
                    .expect("test setup descriptor digest"),
            ),
        }
    }

    #[test]
    fn root_direct_recommitment_rejects_undersized_setup_prefix() {
        let params =
            LevelParams::params_only(SisModulusFamily::Q32, D, 2, 1, 1, 1, stage1_config())
                .with_decomp(1, 0, 2, 1, 1, 0)
                .expect("valid direct layout");
        let setup = verifier_setup(AkitaSetupSeed {
            max_num_vars: 6,
            max_num_batched_polys: 1,
            max_num_points: 1,
            max_setup_len: 3,
            public_matrix_seed: [0u8; 32],
            zk_blinding_seed: [1u8; 32],
        });
        let witnesses = vec![DirectWitnessProof::FieldElements(FlatRingVec::from_coeffs(
            vec![F::zero(); 64],
        ))];
        let err = validate_root_direct_recommitment_shape::<F, D>(
            &witnesses,
            &setup,
            &incidence_summary(6),
            &params,
        )
        .expect_err("A layout needs four columns but setup prefix has three entries");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn root_direct_recommitment_rejects_wrong_witness_dimension() {
        let mut params =
            LevelParams::params_only(SisModulusFamily::Q32, D, 2, 1, 1, 1, stage1_config())
                .with_decomp(1, 0, 2, 1, 1, 0)
                .expect("valid direct layout");
        params.b_key = AjtaiKeyParams::new_unchecked(SisModulusFamily::Q32, 1, 128, 0, D);
        let setup = verifier_setup(AkitaSetupSeed {
            max_num_vars: 6,
            max_num_batched_polys: 1,
            max_num_points: 1,
            max_setup_len: 128,
            public_matrix_seed: [0u8; 32],
            zk_blinding_seed: [1u8; 32],
        });
        let witnesses = vec![DirectWitnessProof::FieldElements(FlatRingVec::from_coeffs(
            vec![F::zero(); 32],
        ))];
        let err = validate_root_direct_recommitment_shape::<F, D>(
            &witnesses,
            &setup,
            &incidence_summary(6),
            &params,
        )
        .expect_err("num_vars=6 requires 64 direct witness elements");
        assert!(matches!(err, AkitaError::InvalidProof));
    }
}
