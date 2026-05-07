//! Top-level batched verifier orchestration once a schedule is selected.

use crate::{
    prepare_verifier_claims, verify_fold_batched_proof, verify_root_direct_openings,
    PreparedVerifierClaims,
};
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, RandomSampling};
use akita_transcript::Transcript;
use akita_types::{
    checked_total_claims, checked_total_groups, schedule_is_root_direct, AkitaBatchedProof,
    AkitaBatchedRootProof, AkitaRootBatchSummary, AkitaScheduleInputs, AkitaVerifierSetup,
    BasisMode, DirectWitnessProof, LevelParams, MultiPointBatchShape, RingCommitment, Schedule,
    Step, VerifierClaims,
};
use std::array::from_fn;

#[cfg(feature = "zk")]
/// Root-direct commitment blinding payload carried by zk proofs.
pub type DirectCommitmentPayload<'a> = &'a [Vec<i8>];
#[cfg(not(feature = "zk"))]
/// Empty root-direct commitment blinding payload for transparent builds.
pub type DirectCommitmentPayload<'a> = &'a ();

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

fn direct_inner_opening_digits<F, const D: usize>(
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
    proof_digits: &[i8],
    params: &LevelParams,
) -> Result<(), AkitaError>
where
    F: CanonicalField,
{
    let expected_planes =
        akita_types::zk::blind_column_count::<F>(params.b_key.row_len(), D, params.log_basis);
    let expected_digits = expected_planes
        .checked_mul(D)
        .ok_or(AkitaError::InvalidProof)?;
    if proof_digits.len() != expected_digits {
        return Err(AkitaError::InvalidProof);
    }
    input.extend(proof_digits.chunks_exact(D).map(|chunk| {
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
        outer_input.extend(direct_inner_opening_digits(&witness_rings, setup, params));
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
    batch_shape: &MultiPointBatchShape,
    params: &LevelParams,
    outer_blinding_digits: DirectCommitmentPayload<'_>,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
{
    if flat_commitments.len() != batch_shape.claim_group_sizes.len() {
        return Err(AkitaError::InvalidProof);
    }
    #[cfg(feature = "zk")]
    if outer_blinding_digits.len() != flat_commitments.len() {
        return Err(AkitaError::InvalidProof);
    }
    #[cfg(not(feature = "zk"))]
    let _ = outer_blinding_digits;
    let total_groups = checked_total_groups(
        &batch_shape.point_group_sizes,
        "root_direct_commitment_check",
    )?;
    if total_groups != batch_shape.claim_group_sizes.len() {
        return Err(AkitaError::InvalidProof);
    }
    let total_claims = checked_total_claims(
        &batch_shape.claim_group_sizes,
        "root_direct_commitment_check",
    )?;
    if total_claims != witnesses.len() {
        return Err(AkitaError::InvalidProof);
    }

    let mut claim_offset = 0usize;
    let mut expected_commitments = Vec::with_capacity(batch_shape.claim_group_sizes.len());
    for (group_idx, &group_size) in batch_shape.claim_group_sizes.iter().enumerate() {
        #[cfg(not(feature = "zk"))]
        let _ = group_idx;
        let group_witnesses = &witnesses[claim_offset..claim_offset + group_size];
        let commitment = recommit_direct_witness_group::<F, D>(
            group_witnesses,
            setup,
            params,
            #[cfg(feature = "zk")]
            &outer_blinding_digits[group_idx],
        )?;
        expected_commitments.push(commitment);
        claim_offset += group_size;
    }

    if expected_commitments != flat_commitments {
        return Err(AkitaError::InvalidProof);
    }

    Ok(())
}

/// Config-derived layouts needed by the folded-root verifier branch.
pub struct FoldVerifierLayouts {
    /// Root verifier layout derived for the selected folded-root schedule.
    pub root_lp: LevelParams,
    /// First recursive-level params reached by the root fold.
    pub next_level_params: LevelParams,
}

/// Schedule context selected by the root scheme/config layer.
pub enum BatchedVerifierScheduleContext {
    /// The selected schedule uses the root-direct fast path.
    RootDirect,
    /// The selected schedule starts with a folded root.
    Fold(Box<FoldVerifierLayouts>),
}

/// Build the verifier schedule context for an already-selected proof schedule.
///
/// Root config policy supplies the two layout callbacks; this helper owns only
/// the public schedule shape interpretation needed by verifier replay.
///
/// # Errors
///
/// Returns an error if the schedule is empty or either supplied layout callback
/// rejects the selected folded-root schedule.
pub fn prepare_batched_verifier_schedule_context<RootLayout, NextParams>(
    max_num_vars: usize,
    schedule: &Schedule,
    mut root_layout: RootLayout,
    mut next_params: NextParams,
) -> Result<BatchedVerifierScheduleContext, AkitaError>
where
    RootLayout: FnMut(AkitaScheduleInputs, &LevelParams) -> Result<LevelParams, AkitaError>,
    NextParams: FnMut(AkitaScheduleInputs) -> Result<LevelParams, AkitaError>,
{
    match schedule.steps.first() {
        Some(Step::Direct(_)) => Ok(BatchedVerifierScheduleContext::RootDirect),
        Some(Step::Fold(root_step)) => {
            let root_inputs = AkitaScheduleInputs {
                max_num_vars,
                level: 0,
                current_w_len: root_step.current_w_len,
            };
            let root_lp = root_layout(root_inputs, &root_step.params)?;
            let next_inputs = AkitaScheduleInputs {
                max_num_vars,
                level: 1,
                current_w_len: root_step.next_w_len,
            };
            let next_level_params = next_params(next_inputs)?;
            Ok(BatchedVerifierScheduleContext::Fold(Box::new(
                FoldVerifierLayouts {
                    root_lp,
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
pub fn verify_batched_proof_with_schedule<'a, F, T, const D: usize, DirectCommitmentCheck>(
    proof: &AkitaBatchedProof<F>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    prepared_claims: PreparedVerifierClaims<'a, F, RingCommitment<F, D>>,
    basis: BasisMode,
    schedule: &Schedule,
    schedule_context: BatchedVerifierScheduleContext,
    verify_direct_commitments: DirectCommitmentCheck,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    T: Transcript<F>,
    DirectCommitmentCheck: FnOnce(
        &[DirectWitnessProof<F>],
        &[RingCommitment<F, D>],
        &MultiPointBatchShape,
        DirectCommitmentPayload<'_>,
    ) -> Result<(), AkitaError>,
{
    let PreparedVerifierClaims {
        opening_points,
        commitments,
        openings,
        batch_shape,
        num_vars: _,
        layout_num_claims: _,
        batch_summary: _,
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
            verify_root_direct_openings(
                witnesses,
                &opening_points,
                &openings,
                &batch_shape,
                basis,
            )?;
            #[cfg(feature = "zk")]
            let direct_commitment_payload = proof
                .root
                .direct_outer_blinding_digits()
                .ok_or(AkitaError::InvalidProof)?;
            #[cfg(not(feature = "zk"))]
            let direct_commitment_payload = &();
            verify_direct_commitments(
                witnesses,
                &commitments,
                &batch_shape,
                direct_commitment_payload,
            )?;
        }
        AkitaBatchedRootProof::Fold(_) => {
            let BatchedVerifierScheduleContext::Fold(layouts) = schedule_context else {
                return Err(AkitaError::InvalidProof);
            };
            verify_fold_batched_proof::<F, T, D>(
                proof,
                setup,
                transcript,
                &opening_points,
                &openings,
                &commitments,
                &batch_shape,
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
    T,
    const D: usize,
    SelectSchedule,
    RootLayout,
    NextParams,
    DirectParams,
    DirectCommitmentCheck,
>(
    proof: &AkitaBatchedProof<F>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    claims: VerifierClaims<'a, F, RingCommitment<F, D>>,
    basis: BasisMode,
    select_schedule: SelectSchedule,
    root_layout: RootLayout,
    next_params: NextParams,
    direct_params: DirectParams,
    verify_direct_commitments: DirectCommitmentCheck,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    T: Transcript<F>,
    SelectSchedule:
        FnOnce(usize, usize, usize, AkitaRootBatchSummary) -> Result<Schedule, AkitaError>,
    RootLayout: FnMut(AkitaScheduleInputs, &LevelParams) -> Result<LevelParams, AkitaError>,
    NextParams: FnMut(&Schedule, AkitaScheduleInputs) -> Result<LevelParams, AkitaError>,
    DirectParams: FnOnce(usize, usize) -> Result<LevelParams, AkitaError>,
    DirectCommitmentCheck: FnOnce(
        &[DirectWitnessProof<F>],
        &AkitaVerifierSetup<F>,
        &[RingCommitment<F, D>],
        &MultiPointBatchShape,
        &LevelParams,
        DirectCommitmentPayload<'_>,
    ) -> Result<(), AkitaError>,
{
    let prepared_claims = prepare_verifier_claims(&setup.expanded, &claims)?;
    let num_vars = prepared_claims.num_vars;
    let layout_num_claims = prepared_claims.layout_num_claims;
    let batch_summary = prepared_claims.batch_summary;

    let max_num_vars = setup.expanded.seed.max_num_vars;
    let schedule = select_schedule(max_num_vars, num_vars, layout_num_claims, batch_summary)
        .map_err(|_| AkitaError::InvalidProof)?;

    let mut next_params = next_params;
    let schedule_context = prepare_batched_verifier_schedule_context(
        max_num_vars,
        &schedule,
        root_layout,
        |next_inputs| next_params(&schedule, next_inputs),
    )
    .map_err(|_| AkitaError::InvalidProof)?;

    verify_batched_proof_with_schedule::<F, T, D, _>(
        proof,
        setup,
        transcript,
        prepared_claims,
        basis,
        &schedule,
        schedule_context,
        |witnesses, commitments, batch_shape, direct_commitment_payload| {
            let total_claims =
                checked_total_claims(&batch_shape.claim_group_sizes, "root_direct_verify")
                    .map_err(|_| AkitaError::InvalidProof)?;
            let params =
                direct_params(num_vars, total_claims).map_err(|_| AkitaError::InvalidProof)?;
            verify_direct_commitments(
                witnesses,
                setup,
                commitments,
                batch_shape,
                &params,
                direct_commitment_payload,
            )
        },
    )
}
