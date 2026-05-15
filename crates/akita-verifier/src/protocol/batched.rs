//! Top-level batched verifier orchestration once a schedule is selected.

use crate::proof::claims::{prepare_verifier_claims, PreparedVerifierClaims};
use crate::proof::direct::verify_root_direct_openings_with_incidence;
use crate::protocol::levels::verify_fold_batched_proof;
use akita_algebra::CyclotomicRing;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FrobeniusExtField, FromPrimitiveInt,
    PseudoMersenneField, RandomSampling,
};
use akita_serialization::AkitaSerialize;
use akita_transcript::Transcript;
use akita_types::{
    folded_root_supports_opening_shape, root_tensor_projection_enabled, schedule_is_root_direct,
    AkitaBatchedProof, AkitaBatchedRootProof, AkitaScheduleInputs, AkitaVerifierSetup, BasisMode,
    ClaimIncidenceSummary, DirectStep, DirectWitnessProof, DirectWitnessShape, LevelParams,
    RingCommitment, RingSubfieldEncoding, Schedule, Step, VerifierClaims,
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

fn i8_plane_to_ring<F, const D: usize>(plane: &[i8; D]) -> CyclotomicRing<F, D>
where
    F: FieldCore + FromPrimitiveInt,
{
    CyclotomicRing::from_coefficients(from_fn(|idx| F::from_i64(plane[idx] as i64)))
}

fn field_evals_to_rings<F, const D: usize>(
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

fn mat_vec_mul_i8_plain<F, const D: usize>(
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

fn direct_decomposed_inner_rows<F, const D: usize>(
    witness_rings: &[CyclotomicRing<F, D>],
    setup: &AkitaVerifierSetup<F>,
    params: &LevelParams,
) -> Vec<[i8; D]>
where
    F: FieldCore + CanonicalField,
{
    let a_matrix = setup
        .expanded
        .shared_matrix
        .ring_view::<D>(params.a_key.row_len(), setup.expanded.seed.max_stride);
    let a_rows: Vec<_> = (0..params.a_key.row_len())
        .map(|row| a_matrix.row(row))
        .collect();
    let mut out =
        Vec::with_capacity(params.num_blocks * params.a_key.row_len() * params.num_digits_open);

    for block_idx in 0..params.num_blocks {
        let start = block_idx * params.block_len;
        let end = (start + params.block_len).min(witness_rings.len());
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

    out
}

#[cfg(feature = "zk")]
fn append_direct_blinding<F, const D: usize>(
    input: &mut Vec<[i8; D]>,
    revealed_b_blinding_digits: &[i8],
    params: &LevelParams,
) -> Result<(), AkitaError>
where
    F: CanonicalField,
{
    let expected_planes =
        akita_types::zk::blinding_column_count::<F>(params.b_key.row_len(), D, params.log_basis);
    let expected_digits = expected_planes
        .checked_mul(D)
        .ok_or(AkitaError::InvalidProof)?;
    if revealed_b_blinding_digits.len() != expected_digits {
        return Err(AkitaError::InvalidProof);
    }
    input.extend(revealed_b_blinding_digits.chunks_exact(D).map(|chunk| {
        let mut plane = [0i8; D];
        plane.copy_from_slice(chunk);
        plane
    }));
    Ok(())
}

fn recommit_direct_witness_group<F, const D: usize>(
    group_witnesses: &[DirectWitnessProof<F>],
    setup: &AkitaVerifierSetup<F>,
    params: &LevelParams,
    #[cfg(feature = "zk")] blinding_digits: &[i8],
) -> Result<RingCommitment<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let mut outer_input = Vec::new();
    for witness in group_witnesses {
        let field_witness = witness
            .as_field_elements()
            .ok_or(AkitaError::InvalidProof)?
            .coeffs();
        let witness_rings = field_evals_to_rings::<F, D>(field_witness)?;
        outer_input.extend(direct_decomposed_inner_rows(&witness_rings, setup, params));
    }

    #[cfg(feature = "zk")]
    append_direct_blinding::<F, D>(&mut outer_input, blinding_digits, params)?;

    let b_matrix = setup
        .expanded
        .shared_matrix
        .ring_view::<D>(params.b_key.row_len(), setup.expanded.seed.max_stride);
    let b_rows: Vec<_> = (0..params.b_key.row_len())
        .map(|row| b_matrix.row(row))
        .collect();
    Ok(RingCommitment {
        u: mat_vec_mul_i8_plain::<F, D>(&b_rows, &outer_input),
    })
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
    F: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField,
{
    if flat_commitments.len() != incidence_summary.num_groups {
        return Err(AkitaError::InvalidProof);
    }
    if incidence_summary.group_poly_counts.len() != incidence_summary.num_groups {
        return Err(AkitaError::InvalidProof);
    }
    #[cfg(feature = "zk")]
    if b_blinding_digits.len() != flat_commitments.len() {
        return Err(AkitaError::InvalidProof);
    }
    #[cfg(not(feature = "zk"))]
    let _ = b_blinding_digits;
    let total_group_polys = incidence_summary
        .group_poly_counts
        .iter()
        .try_fold(0usize, |acc, &count| {
            acc.checked_add(count).ok_or(AkitaError::InvalidProof)
        })?;
    if total_group_polys != witnesses.len() {
        return Err(AkitaError::InvalidProof);
    }

    let mut claim_offset = 0usize;
    let mut expected_commitments = Vec::with_capacity(incidence_summary.num_groups);
    for (group_idx, &group_size) in incidence_summary.group_poly_counts.iter().enumerate() {
        #[cfg(not(feature = "zk"))]
        let _ = group_idx;
        let group_witnesses = &witnesses[claim_offset..claim_offset + group_size];
        let commitment = recommit_direct_witness_group::<F, D>(
            group_witnesses,
            setup,
            params,
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

fn root_direct_schedule(num_vars: usize) -> Result<Schedule, AkitaError> {
    let current_w_len = 1usize.checked_shl(num_vars as u32).ok_or_else(|| {
        AkitaError::InvalidSetup("root-direct witness length overflow".to_string())
    })?;
    Ok(Schedule {
        steps: vec![Step::Direct(DirectStep {
            current_w_len,
            witness_shape: DirectWitnessShape::FieldElements(current_w_len),
            direct_bytes: 0,
        })],
        total_bytes: 0,
    })
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
pub(crate) fn prepare_batched_verifier_schedule_context<NextParams>(
    num_vars: usize,
    schedule: &Schedule,
    mut next_params: NextParams,
) -> Result<BatchedVerifierScheduleContext, AkitaError>
where
    NextParams: FnMut(AkitaScheduleInputs) -> Result<LevelParams, AkitaError>,
{
    match schedule.steps.first() {
        Some(Step::Direct(_)) => Ok(BatchedVerifierScheduleContext::RootDirect),
        Some(Step::Fold(root_step)) => {
            let next_inputs = AkitaScheduleInputs {
                num_vars,
                level: 1,
                current_w_len: root_step.next_w_len,
            };
            let next_level_params = next_params(next_inputs)?;
            Ok(BatchedVerifierScheduleContext::Fold(Box::new(
                FoldVerifierLayouts {
                    root_lp: root_step.params.clone(),
                    next_level_params,
                },
            )))
        }
        None => Err(AkitaError::InvalidProof),
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
        AkitaBatchedRootProof::Fold(_) => {
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

/// Verify a batched proof using caller-supplied config/policy callbacks.
///
/// This is the verifier crate's top-level orchestration entrypoint for the
/// current crate split. It owns public claim normalization, schedule-context
/// construction, root-direct and folded-root dispatch, and recursive verifier
/// replay. The root aggregate crate supplies only config-backed schedule/layout
/// selection and the root-direct commitment recomputation callback.
///
/// # Errors
///
/// Returns an error if public claims are malformed, schedule/layout policy
/// rejects the proof shape, root-direct commitment recomputation rejects, or
/// proof replay fails.
#[allow(clippy::too_many_arguments)]
pub fn verify_batched_with_policy<
    'a,
    F,
    E,
    C,
    T,
    const D: usize,
    SelectSchedule,
    NextParams,
    DirectParams,
    DirectCommitmentCheck,
>(
    proof: &AkitaBatchedProof<F, C>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    claims: VerifierClaims<'a, E, RingCommitment<F, D>>,
    basis: BasisMode,
    select_schedule: SelectSchedule,
    next_params: NextParams,
    direct_params: DirectParams,
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
    SelectSchedule: FnOnce(&ClaimIncidenceSummary) -> Result<Schedule, AkitaError>,
    NextParams: FnMut(&Schedule, AkitaScheduleInputs) -> Result<LevelParams, AkitaError>,
    DirectParams: FnOnce(&ClaimIncidenceSummary, usize) -> Result<LevelParams, AkitaError>,
    DirectCommitmentCheck: FnOnce(
        &[DirectWitnessProof<F>],
        &AkitaVerifierSetup<F>,
        &[RingCommitment<F, D>],
        &ClaimIncidenceSummary,
        &LevelParams,
        RootDirectBlindingPayload<'_>,
    ) -> Result<(), AkitaError>,
{
    let prepared_claims = prepare_verifier_claims(&setup.expanded, &claims)?;
    let num_vars = prepared_claims.incidence_summary.num_vars;
    let mut schedule = select_schedule(&prepared_claims.incidence_summary)
        .map_err(|_| AkitaError::InvalidProof)?;
    if let Some(Step::Fold(root_step)) = schedule.steps.first() {
        let alpha_bits = root_step.params.ring_dimension.trailing_zeros() as usize;
        if !folded_root_supports_opening_shape::<F, E, C, D>(
            &prepared_claims.opening_points,
            &root_step.params,
            alpha_bits,
        ) && !root_tensor_projection_enabled::<F, E, C, D>(num_vars)
        {
            schedule = root_direct_schedule(num_vars).map_err(|_| AkitaError::InvalidProof)?;
        }
    }

    let mut next_params = next_params;
    let schedule_context =
        prepare_batched_verifier_schedule_context(num_vars, &schedule, |next_inputs| {
            next_params(&schedule, next_inputs)
        })
        .map_err(|_| AkitaError::InvalidProof)?;

    verify_batched_proof_with_schedule::<F, E, C, T, D, _>(
        proof,
        setup,
        transcript,
        prepared_claims,
        basis,
        &schedule,
        schedule_context,
        |witnesses, commitments, incidence_summary, direct_commitment_payload| {
            let params = direct_params(incidence_summary, setup.expanded.seed.max_num_points)
                .map_err(|_| AkitaError::InvalidProof)?;
            verify_direct_commitments(
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
