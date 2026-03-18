//! Labrador verifier/reducer loop.

use crate::algebra::ring::CyclotomicRing;
use crate::algebra::SparseChallenge;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::commitment::utils::linear::mat_vec_mul_crt_ntt_i8_many;
use crate::protocol::labrador::aggregation::{
    aggregate_jl_constraints_verifier, aggregate_statement, safe_to_use_scalar_randomness,
};
use crate::protocol::labrador::comkey::LabradorComKeySeed;
use crate::protocol::labrador::constraints::{
    materialize_reduced_constraints, pair_index, LabradorConstraint, NextWitnessLayout,
};
use crate::protocol::labrador::guardrails::LABRADOR_MAX_LEVELS;
use crate::protocol::labrador::johnson_lindenstrauss::LabradorJlMatrix;
use crate::protocol::labrador::setup::LabradorSetupMatrices;
use crate::protocol::labrador::transcript::{
    absorb_labrador_jl_projection, absorb_labrador_level_context,
    sample_labrador_aggregation_challenge, LabradorLevelTranscriptContext,
};
use crate::protocol::labrador::types::{
    LabradorLevelProof, LabradorProof, LabradorReducedConstraintPlan, LabradorStatement,
    LabradorWitness,
};
use crate::protocol::labrador::utils::{mat_vec_mul, pow2_field, try_centered_i8_rows};
use crate::protocol::transcript::labels;
use crate::protocol::transcript::{
    challenge_ring_element, challenge_sparse_ring_elements_rejection_sampled, Transcript,
};
use crate::{CanonicalField, FieldCore, FieldSampling, FromSmallInt};
use std::sync::Arc;

/// Output of verifier-side Labrador reduction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabradorVerifyResult<F: FieldCore, const D: usize> {
    /// Statement after replaying all reduction levels.
    pub terminal_statement: LabradorStatement<F, D>,
    /// Final clear opening witness from the proof payload.
    pub final_opening_witness: LabradorWitness<F, D>,
}

/// Verify Labrador proof and return terminal reduction state.
///
/// Currently supports a single Labrador level; recursive reduction is
/// intentionally deferred until the folding statement update is implemented.
///
/// # Errors
///
/// Returns [`HachiError::InvalidProof`] on structural inconsistencies,
/// norm bound violations, or constraint failures.
#[tracing::instrument(skip_all, name = "labrador::verify")]
pub fn verify<F, T, const D: usize>(
    initial_statement: &LabradorStatement<F, D>,
    proof: &LabradorProof<F, D>,
    comkey_seed: &LabradorComKeySeed,
    transcript: &mut T,
) -> Result<LabradorVerifyResult<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + FromSmallInt,
    T: Transcript<F>,
{
    if proof.levels.len() > LABRADOR_MAX_LEVELS || proof.final_opening_witness.rows().is_empty() {
        return Err(HachiError::InvalidProof);
    }

    if proof.levels.is_empty() {
        let final_norm = proof.final_opening_witness.norm();
        if final_norm > initial_statement.witness_norm_bound_sq {
            return Err(HachiError::InvalidProof);
        }
        let constraints = explicit_constraints(initial_statement)?;
        verify_constraints(&constraints, &proof.final_opening_witness)?;
        return Ok(LabradorVerifyResult {
            terminal_statement: initial_statement.clone(),
            final_opening_witness: proof.final_opening_witness.clone(),
        });
    }

    let mut statement = initial_statement.clone();
    let last_idx = proof.levels.len() - 1;
    for (idx, level) in proof.levels.iter().enumerate() {
        if level.tail {
            if idx != last_idx {
                return Err(HachiError::InvalidProof);
            }
            verify_tail_level(
                &statement,
                level,
                &proof.final_opening_witness,
                comkey_seed,
                transcript,
                idx,
            )?;
            return Ok(LabradorVerifyResult {
                terminal_statement: statement,
                final_opening_witness: proof.final_opening_witness.clone(),
            });
        }
        statement = reduce_statement(&statement, level, comkey_seed, transcript, idx)?;
    }

    let final_norm = proof.final_opening_witness.norm();
    if final_norm > statement.witness_norm_bound_sq {
        return Err(HachiError::InvalidProof);
    }
    let constraints = explicit_constraints(&statement)?;
    verify_constraints(&constraints, &proof.final_opening_witness)?;

    Ok(LabradorVerifyResult {
        terminal_statement: statement,
        final_opening_witness: proof.final_opening_witness.clone(),
    })
}

#[tracing::instrument(skip_all, name = "labrador::explicit_constraints")]
fn explicit_constraints<F, const D: usize>(
    statement: &LabradorStatement<F, D>,
) -> Result<Vec<LabradorConstraint<F, D>>, HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
{
    if let Some(plan) = statement.reduced_constraints.as_deref() {
        materialize_reduced_constraints(
            plan,
            &statement.inner_opening_payload,
            &statement.linear_garbage_payload,
        )
    } else {
        Ok(statement.constraints.clone())
    }
}

#[tracing::instrument(
    skip_all,
    name = "labrador::reduce_statement",
    fields(level_index, tail = level.tail)
)]
fn reduce_statement<F, T, const D: usize>(
    statement: &LabradorStatement<F, D>,
    level: &LabradorLevelProof<F, D>,
    comkey_seed: &LabradorComKeySeed,
    transcript: &mut T,
    level_index: usize,
) -> Result<LabradorStatement<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + FromSmallInt,
    T: Transcript<F>,
{
    let virtual_row_count = validate_level_shape(level, false)?;
    let virtual_row_len = level.virtual_row_len;
    let virt_row_lengths = vec![virtual_row_len; virtual_row_count];

    absorb_labrador_level_context(
        transcript,
        &LabradorLevelTranscriptContext {
            level_index,
            tail: level.tail,
            input_row_lengths: level.input_row_lengths.clone(),
            witness_digit_parts: level.config.witness_digit_parts,
            witness_digit_bits: level.config.witness_digit_bits,
            aux_digit_parts: level.config.aux_digit_parts,
            aux_digit_bits: level.config.aux_digit_bits,
            inner_commit_rank: level.config.inner_commit_rank,
            outer_commit_rank: level.config.outer_commit_rank,
        },
    )?;
    transcript.append_serde(
        labels::ABSORB_LABRADOR_INNER_OPENING_PAYLOAD,
        &level.inner_opening_payload,
    );

    let total_len: usize = virt_row_lengths.iter().sum();
    let jl_cols = total_len * D;
    let jl_matrix =
        LabradorJlMatrix::replay_nonce_search::<F, T>(transcript, level.jl_nonce, jl_cols)?;
    absorb_labrador_jl_projection(transcript, &level.jl_projection);

    let (phi_jl_flat, b_jl) = aggregate_jl_constraints_verifier(
        &virt_row_lengths,
        &level.jl_projection,
        &jl_matrix,
        &level.jl_lift_residuals,
        transcript,
    )?;
    let explicit_aggregation = if statement.reduced_constraints.is_none() {
        Some(aggregate_statement(
            statement,
            &level.input_row_lengths,
            transcript,
        )?)
    } else {
        None
    };
    let reduced_aggregation = statement
        .reduced_constraints
        .as_deref()
        .map(|plan| prepare_reduced_statement_aggregation(statement, plan, transcript))
        .transpose()?;

    transcript.append_serde(
        labels::ABSORB_LABRADOR_LINEAR_GARBAGE_PAYLOAD,
        &level.linear_garbage_payload,
    );
    let challenges = replay_amortize_challenges::<F, T, D>(transcript, virtual_row_count)?;
    tracing::debug!(
        level_index,
        tail = false,
        ?challenges,
        "labrador verifier amortize challenges"
    );
    let mut amortized_phi =
        if let Some((phi_stmt_orig, _statement_rhs)) = explicit_aggregation.as_ref() {
            let phi_stmt = reshape_phi_verifier::<F, D>(
                phi_stmt_orig,
                &level.input_row_lengths,
                &level.row_split_counts,
                virtual_row_len,
            )?;
            let mut phi_total = phi_stmt;
            add_phi_flat_in_place(&mut phi_total, &phi_jl_flat)?;
            combine_virtual_rows(&phi_total, &challenges, virtual_row_len)?
        } else {
            let plan = statement
                .reduced_constraints
                .as_deref()
                .ok_or(HachiError::InvalidProof)?;
            let aggregation = reduced_aggregation
                .as_ref()
                .ok_or(HachiError::InvalidProof)?;
            let mut amortized_phi = finalize_reduced_statement_aggregation(
                plan,
                aggregation,
                &level.input_row_lengths,
                &level.row_split_counts,
                virtual_row_len,
                &challenges,
            )?;
            let amortized_phi_jl = combine_flat_rows(&phi_jl_flat, &challenges, virtual_row_len)?;
            add_amortized_phi_in_place(&mut amortized_phi, &amortized_phi_jl)?;
            amortized_phi
        };
    let statement_rhs = if let Some((_, statement_rhs)) = explicit_aggregation {
        statement_rhs
    } else {
        reduced_aggregation
            .as_ref()
            .ok_or(HachiError::InvalidProof)?
            .aggregated_rhs
    };
    let aggregated_rhs = statement_rhs + b_jl;

    let setup = Arc::new(LabradorSetupMatrices::new(
        &level.config,
        virtual_row_count,
        virtual_row_len,
        comkey_seed,
    ));
    let reduced_constraints = LabradorReducedConstraintPlan {
        row_count: virt_row_lengths.len(),
        max_len: virtual_row_len,
        config: level.config,
        challenges: challenges.clone(),
        amortized_phi: std::mem::take(&mut amortized_phi),
        aggregated_rhs,
        setup,
    };

    Ok(LabradorStatement {
        inner_opening_payload: level.inner_opening_payload.clone(),
        linear_garbage_payload: level.linear_garbage_payload.clone(),
        challenges,
        constraints: Vec::new(),
        reduced_constraints: Some(Box::new(reduced_constraints)),
        witness_norm_bound_sq: level.next_witness_norm_sq,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ReshapeCombineSegment {
    src_start: usize,
    dst_start: usize,
    len: usize,
    challenge_idx: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReducedStatementAggregationReplay<F: FieldCore, const D: usize> {
    b_alphas: Vec<CyclotomicRing<F, D>>,
    d_alphas: Vec<CyclotomicRing<F, D>>,
    a_alphas: Vec<CyclotomicRing<F, D>>,
    alpha_lg: CyclotomicRing<F, D>,
    alpha_diag: CyclotomicRing<F, D>,
    aggregated_rhs: CyclotomicRing<F, D>,
}

#[inline]
fn scalar_to_ring<F: FieldCore, const D: usize>(scalar: F) -> CyclotomicRing<F, D> {
    let mut coeffs = [F::zero(); D];
    coeffs[0] = scalar;
    CyclotomicRing::from_coefficients(coeffs)
}

#[tracing::instrument(skip_all, name = "labrador::prepare_reduced_statement_aggregation")]
fn prepare_reduced_statement_aggregation<F, T, const D: usize>(
    statement: &LabradorStatement<F, D>,
    plan: &LabradorReducedConstraintPlan<F, D>,
    transcript: &mut T,
) -> Result<ReducedStatementAggregationReplay<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
    T: Transcript<F>,
{
    if plan.setup.a_mat.len() != plan.config.inner_commit_rank
        || plan.setup.b_mat.len() != statement.inner_opening_payload.len()
        || plan.setup.d_mat.len() != statement.linear_garbage_payload.len()
    {
        return Err(HachiError::InvalidProof);
    }

    let mut aggregated_rhs = CyclotomicRing::<F, D>::zero();

    let b_alphas: Vec<CyclotomicRing<F, D>> = statement
        .inner_opening_payload
        .iter()
        .map(|target| {
            if safe_to_use_scalar_randomness::<F>() {
                let alpha_scalar = sample_labrador_aggregation_challenge::<F, _>(transcript);
                aggregated_rhs += target.scale(&alpha_scalar);
                scalar_to_ring::<F, D>(alpha_scalar)
            } else {
                let alpha =
                    challenge_ring_element(transcript, labels::CHALLENGE_LABRADOR_AGGREGATION);
                aggregated_rhs += alpha * *target;
                alpha
            }
        })
        .collect();
    let d_alphas: Vec<CyclotomicRing<F, D>> = statement
        .linear_garbage_payload
        .iter()
        .map(|target| {
            if safe_to_use_scalar_randomness::<F>() {
                let alpha_scalar = sample_labrador_aggregation_challenge::<F, _>(transcript);
                aggregated_rhs += target.scale(&alpha_scalar);
                scalar_to_ring::<F, D>(alpha_scalar)
            } else {
                let alpha =
                    challenge_ring_element(transcript, labels::CHALLENGE_LABRADOR_AGGREGATION);
                aggregated_rhs += alpha * *target;
                alpha
            }
        })
        .collect();
    let a_alphas = (0..plan.config.inner_commit_rank)
        .map(|_| {
            if safe_to_use_scalar_randomness::<F>() {
                let alpha_scalar = sample_labrador_aggregation_challenge::<F, _>(transcript);
                scalar_to_ring::<F, D>(alpha_scalar)
            } else {
                challenge_ring_element(transcript, labels::CHALLENGE_LABRADOR_AGGREGATION)
            }
        })
        .collect();
    let alpha_lg = if safe_to_use_scalar_randomness::<F>() {
        let alpha_scalar = sample_labrador_aggregation_challenge::<F, _>(transcript);
        scalar_to_ring::<F, D>(alpha_scalar)
    } else {
        challenge_ring_element(transcript, labels::CHALLENGE_LABRADOR_AGGREGATION)
    };
    let alpha_diag = if safe_to_use_scalar_randomness::<F>() {
        let alpha_scalar = sample_labrador_aggregation_challenge::<F, _>(transcript);
        aggregated_rhs += plan.aggregated_rhs.scale(&alpha_scalar);
        scalar_to_ring::<F, D>(alpha_scalar)
    } else {
        let alpha_diag = challenge_ring_element(transcript, labels::CHALLENGE_LABRADOR_AGGREGATION);
        aggregated_rhs += alpha_diag * plan.aggregated_rhs;
        alpha_diag
    };

    Ok(ReducedStatementAggregationReplay {
        b_alphas,
        d_alphas,
        a_alphas,
        alpha_lg,
        alpha_diag,
        aggregated_rhs,
    })
}

fn build_reshape_combine_plan(
    row_lengths: &[usize],
    row_split_counts: &[usize],
    virtual_row_len: usize,
    challenges: &[SparseChallenge],
) -> Result<Vec<Vec<ReshapeCombineSegment>>, HachiError> {
    let virtual_row_count =
        validate_reshape_metadata(row_lengths, row_split_counts, virtual_row_len)?;
    if challenges.len() != virtual_row_count {
        return Err(HachiError::InvalidProof);
    }

    let mut row_segments = vec![Vec::new(); row_lengths.len()];
    let mut group_rows = Vec::new();
    let mut challenge_cursor = 0usize;

    for (row_idx, &row_len) in row_lengths.iter().enumerate() {
        group_rows.push((row_idx, row_len));
        let splits = row_split_counts[row_idx];
        if splits == 0 {
            continue;
        }

        let group_len: usize = group_rows.iter().map(|(_, len)| *len).sum();
        if group_len > splits * virtual_row_len {
            return Err(HachiError::InvalidProof);
        }

        let mut group_pos = 0usize;
        for &(group_row_idx, len) in &group_rows {
            let mut row_offset = 0usize;
            while row_offset < len {
                let challenge_idx = challenge_cursor + group_pos / virtual_row_len;
                let dst_start = group_pos % virtual_row_len;
                let take = (virtual_row_len - dst_start).min(len - row_offset);
                row_segments[group_row_idx].push(ReshapeCombineSegment {
                    src_start: row_offset,
                    dst_start,
                    len: take,
                    challenge_idx,
                });
                row_offset += take;
                group_pos += take;
            }
        }

        challenge_cursor += splits;
        group_rows.clear();
    }

    if !group_rows.is_empty() || challenge_cursor != virtual_row_count {
        return Err(HachiError::InvalidProof);
    }

    Ok(row_segments)
}

fn accumulate_row_slice_into_amortized_phi<F: FieldCore + CanonicalField, const D: usize>(
    amortized_phi: &mut [CyclotomicRing<F, D>],
    segments: &[ReshapeCombineSegment],
    challenges: &[SparseChallenge],
    row_offset: usize,
    coeffs: &[CyclotomicRing<F, D>],
    alpha: &CyclotomicRing<F, D>,
) -> Result<(), HachiError> {
    let row_end = row_offset
        .checked_add(coeffs.len())
        .ok_or(HachiError::InvalidProof)?;
    let mut covered = 0usize;

    for segment in segments {
        let seg_start = segment.src_start;
        let seg_end = seg_start + segment.len;
        let start = row_offset.max(seg_start);
        let end = row_end.min(seg_end);
        if start >= end {
            continue;
        }

        let coeff_start = start - row_offset;
        let dst_start = segment.dst_start + (start - seg_start);
        let weight = alpha.mul_by_sparse(&challenges[segment.challenge_idx]);
        cfg_iter_mut!(amortized_phi[dst_start..dst_start + (end - start)])
            .zip(cfg_iter!(coeffs[coeff_start..coeff_start + (end - start)]))
            .for_each(|(dst, src)| weight.mul_accumulate_into(src, dst));
        covered += end - start;
    }

    if covered != coeffs.len() {
        return Err(HachiError::InvalidProof);
    }
    Ok(())
}

fn accumulate_point_into_amortized_phi<F: FieldCore + CanonicalField, const D: usize>(
    amortized_phi: &mut [CyclotomicRing<F, D>],
    segments: &[ReshapeCombineSegment],
    challenges: &[SparseChallenge],
    position: usize,
    value: &CyclotomicRing<F, D>,
) -> Result<(), HachiError> {
    for segment in segments {
        let seg_end = segment.src_start + segment.len;
        if !(segment.src_start..seg_end).contains(&position) {
            continue;
        }

        let dst_idx = segment.dst_start + (position - segment.src_start);
        value.mul_by_sparse_into(
            &challenges[segment.challenge_idx],
            &mut amortized_phi[dst_idx],
        );
        return Ok(());
    }
    Err(HachiError::InvalidProof)
}

fn combine_virtual_rows<F: FieldCore + CanonicalField, const D: usize>(
    rows: &[Vec<CyclotomicRing<F, D>>],
    challenges: &[SparseChallenge],
    virtual_row_len: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
    if rows.len() != challenges.len() {
        return Err(HachiError::InvalidProof);
    }

    let mut combined = vec![CyclotomicRing::<F, D>::zero(); virtual_row_len];
    for (row, challenge) in rows.iter().zip(challenges.iter()) {
        if row.len() != virtual_row_len {
            return Err(HachiError::InvalidProof);
        }
        for (dst, src) in combined.iter_mut().zip(row.iter()) {
            src.mul_by_sparse_into(challenge, dst);
        }
    }
    Ok(combined)
}

fn combine_flat_rows<F: FieldCore + CanonicalField, const D: usize>(
    rows_flat: &[CyclotomicRing<F, D>],
    challenges: &[SparseChallenge],
    virtual_row_len: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
    if rows_flat.len() != challenges.len() * virtual_row_len {
        return Err(HachiError::InvalidProof);
    }

    let mut combined = vec![CyclotomicRing::<F, D>::zero(); virtual_row_len];
    for (row, challenge) in rows_flat.chunks(virtual_row_len).zip(challenges.iter()) {
        for (dst, src) in combined.iter_mut().zip(row.iter()) {
            src.mul_by_sparse_into(challenge, dst);
        }
    }
    Ok(combined)
}

fn add_amortized_phi_in_place<F: FieldCore, const D: usize>(
    dst: &mut [CyclotomicRing<F, D>],
    src: &[CyclotomicRing<F, D>],
) -> Result<(), HachiError> {
    if dst.len() != src.len() {
        return Err(HachiError::InvalidProof);
    }
    cfg_iter_mut!(dst)
        .zip(cfg_iter!(src))
        .for_each(|(dst_elem, src_elem)| *dst_elem += *src_elem);
    Ok(())
}

#[tracing::instrument(skip_all, name = "labrador::finalize_reduced_statement_aggregation")]
fn finalize_reduced_statement_aggregation<F, const D: usize>(
    plan: &LabradorReducedConstraintPlan<F, D>,
    aggregation: &ReducedStatementAggregationReplay<F, D>,
    row_lengths: &[usize],
    row_split_counts: &[usize],
    virtual_row_len: usize,
    challenges: &[SparseChallenge],
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
{
    let layout = NextWitnessLayout::new(plan.row_count, &plan.config);
    if row_lengths.len() != layout.num_rows() {
        return Err(HachiError::InvalidProof);
    }
    if row_lengths
        .iter()
        .take(plan.config.witness_digit_parts)
        .any(|&len| len != plan.max_len)
        || row_lengths[layout.aux_row] != layout.aux_row_len()
    {
        return Err(HachiError::InvalidProof);
    }

    let row_segments =
        build_reshape_combine_plan(row_lengths, row_split_counts, virtual_row_len, challenges)?;
    let aux_segments = row_segments
        .get(layout.aux_row)
        .ok_or(HachiError::InvalidProof)?;
    let mut amortized_phi = vec![CyclotomicRing::<F, D>::zero(); virtual_row_len];
    let pow_witness_bits: Vec<F> = (0..plan.config.witness_digit_parts)
        .map(|idx| pow2_field::<F>(plan.config.witness_digit_bits * idx))
        .collect();
    let pow_aux_bits: Vec<F> = (0..plan.config.aux_digit_parts)
        .map(|idx| pow2_field::<F>(plan.config.aux_digit_bits * idx))
        .collect();
    let inner_opening_start = layout.inner_opening_digits_range().start;
    let linear_garbage_start = layout.linear_garbage_digits_range().start;

    for (alpha, b_row) in aggregation.b_alphas.iter().zip(plan.setup.b_mat.iter()) {
        accumulate_row_slice_into_amortized_phi(
            &mut amortized_phi,
            aux_segments,
            challenges,
            inner_opening_start,
            b_row,
            alpha,
        )?;
    }
    for (alpha, d_row) in aggregation.d_alphas.iter().zip(plan.setup.d_mat.iter()) {
        accumulate_row_slice_into_amortized_phi(
            &mut amortized_phi,
            aux_segments,
            challenges,
            linear_garbage_start,
            d_row,
            alpha,
        )?;
    }

    for (output_idx, alpha) in aggregation.a_alphas.iter().enumerate() {
        let a_row = &plan.setup.a_mat[output_idx];
        for (part_idx, &scale) in pow_witness_bits.iter().enumerate() {
            let scaled_alpha = alpha.scale(&scale);
            accumulate_row_slice_into_amortized_phi(
                &mut amortized_phi,
                &row_segments[part_idx],
                challenges,
                0,
                a_row,
                &scaled_alpha,
            )?;
        }

        for (row_idx, challenge) in plan.challenges.iter().enumerate() {
            let base = alpha.mul_by_sparse(challenge);
            for (part_idx, &scale) in pow_aux_bits.iter().enumerate() {
                let idx = inner_opening_start
                    + row_idx * plan.config.inner_commit_rank * plan.config.aux_digit_parts
                    + output_idx * plan.config.aux_digit_parts
                    + part_idx;
                let value = -(base.scale(&scale));
                accumulate_point_into_amortized_phi(
                    &mut amortized_phi,
                    aux_segments,
                    challenges,
                    idx,
                    &value,
                )?;
            }
        }
    }

    for (part_idx, &scale) in pow_witness_bits.iter().enumerate() {
        let scaled_alpha = aggregation.alpha_lg.scale(&scale);
        accumulate_row_slice_into_amortized_phi(
            &mut amortized_phi,
            &row_segments[part_idx],
            challenges,
            0,
            &plan.amortized_phi,
            &scaled_alpha,
        )?;
    }
    for i in 0..plan.challenges.len() {
        for j in i..plan.challenges.len() {
            let base = aggregation
                .alpha_lg
                .mul_by_sparse(&plan.challenges[i])
                .mul_by_sparse(&plan.challenges[j]);
            let pair = pair_index(i, j, plan.challenges.len());
            for (part_idx, &scale) in pow_aux_bits.iter().enumerate() {
                let idx = linear_garbage_start + pair * plan.config.aux_digit_parts + part_idx;
                let value = -(base.scale(&scale));
                accumulate_point_into_amortized_phi(
                    &mut amortized_phi,
                    aux_segments,
                    challenges,
                    idx,
                    &value,
                )?;
            }
        }
    }

    for i in 0..plan.row_count {
        let pair = pair_index(i, i, plan.row_count);
        for (part_idx, &scale) in pow_aux_bits.iter().enumerate() {
            let idx = linear_garbage_start + pair * plan.config.aux_digit_parts + part_idx;
            let value = aggregation.alpha_diag.scale(&scale);
            accumulate_point_into_amortized_phi(
                &mut amortized_phi,
                aux_segments,
                challenges,
                idx,
                &value,
            )?;
        }
    }

    Ok(amortized_phi)
}

#[tracing::instrument(skip_all, name = "labrador::reshape_phi_verifier")]
fn reshape_phi_verifier<F: FieldCore, const D: usize>(
    phi: &[Vec<CyclotomicRing<F, D>>],
    row_lengths: &[usize],
    row_split_counts: &[usize],
    virtual_row_len: usize,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, HachiError> {
    let virtual_row_count =
        validate_reshape_metadata(row_lengths, row_split_counts, virtual_row_len)?;
    let mut result = Vec::new();
    let mut group: Vec<CyclotomicRing<F, D>> = Vec::new();

    for (i, row) in phi.iter().enumerate() {
        if i >= row_lengths.len() || row.len() != row_lengths[i] {
            return Err(HachiError::InvalidProof);
        }
        group.extend(row.iter().copied());
        let splits = if i < row_split_counts.len() {
            row_split_counts[i]
        } else {
            0
        };
        if splits > 0 {
            if group.len() > splits * virtual_row_len {
                return Err(HachiError::InvalidProof);
            }
            for chunk_idx in 0..splits {
                let start = chunk_idx * virtual_row_len;
                let mut virtual_row = vec![CyclotomicRing::<F, D>::zero(); virtual_row_len];
                for (j, val) in group.iter().enumerate().skip(start).take(virtual_row_len) {
                    virtual_row[j - start] = *val;
                }
                result.push(virtual_row);
            }
            group.clear();
        }
    }
    if !group.is_empty() || result.len() != virtual_row_count {
        return Err(HachiError::InvalidProof);
    }
    Ok(result)
}

#[tracing::instrument(skip_all, name = "labrador::replay_amortize_challenges")]
fn replay_amortize_challenges<F, T, const D: usize>(
    transcript: &mut T,
    rows: usize,
) -> Result<Vec<SparseChallenge>, HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
    T: Transcript<F>,
{
    challenge_sparse_ring_elements_rejection_sampled::<F, T, D>(
        transcript,
        labels::CHALLENGE_LABRADOR_AMORTIZE,
        rows,
    )
}

#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
#[tracing::instrument(
    skip_all,
    name = "labrador::verify_tail_level",
    fields(level_index, tail = level.tail)
)]
fn verify_tail_level<F, T, const D: usize>(
    statement: &LabradorStatement<F, D>,
    level: &LabradorLevelProof<F, D>,
    witness: &LabradorWitness<F, D>,
    comkey_seed: &LabradorComKeySeed,
    transcript: &mut T,
    level_index: usize,
) -> Result<(), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + FromSmallInt,
    T: Transcript<F>,
{
    let virtual_row_len = level.virtual_row_len;
    let virtual_row_count = validate_level_shape(level, true)?;

    if witness.rows().len() != level.config.witness_digit_parts {
        return Err(HachiError::InvalidProof);
    }
    for row in witness.rows().iter() {
        if row.len() != virtual_row_len {
            return Err(HachiError::InvalidProof);
        }
    }

    let inner_opening_digits_len =
        virtual_row_count * level.config.inner_commit_rank * level.config.aux_digit_parts;
    let linear_garbage_digits_len =
        virtual_row_count * (virtual_row_count + 1) / 2 * level.config.aux_digit_parts;
    if level.inner_opening_payload.len() != inner_opening_digits_len
        || level.linear_garbage_payload.len() != linear_garbage_digits_len
    {
        return Err(HachiError::InvalidProof);
    }
    let inner_opening_digits = &level.inner_opening_payload;
    let linear_garbage_digits = &level.linear_garbage_payload;

    absorb_labrador_level_context(
        transcript,
        &LabradorLevelTranscriptContext {
            level_index,
            tail: level.tail,
            input_row_lengths: level.input_row_lengths.clone(),
            witness_digit_parts: level.config.witness_digit_parts,
            witness_digit_bits: level.config.witness_digit_bits,
            aux_digit_parts: level.config.aux_digit_parts,
            aux_digit_bits: level.config.aux_digit_bits,
            inner_commit_rank: level.config.inner_commit_rank,
            outer_commit_rank: level.config.outer_commit_rank,
        },
    )?;

    transcript.append_serde(
        labels::ABSORB_LABRADOR_INNER_OPENING_PAYLOAD,
        &level.inner_opening_payload,
    );

    let virt_total_len = virtual_row_count * virtual_row_len;
    let jl_cols = virt_total_len * D;
    let jl_matrix =
        LabradorJlMatrix::replay_nonce_search::<F, T>(transcript, level.jl_nonce, jl_cols)?;

    absorb_labrador_jl_projection(transcript, &level.jl_projection);

    let virt_row_lengths = vec![virtual_row_len; virtual_row_count];
    let (phi_jl_flat, b_jl) = aggregate_jl_constraints_verifier(
        &virt_row_lengths,
        &level.jl_projection,
        &jl_matrix,
        &level.jl_lift_residuals,
        transcript,
    )?;

    let explicit_aggregation = if statement.reduced_constraints.is_none() {
        Some(aggregate_statement(
            statement,
            &level.input_row_lengths,
            transcript,
        )?)
    } else {
        None
    };
    let reduced_aggregation = statement
        .reduced_constraints
        .as_deref()
        .map(|plan| prepare_reduced_statement_aggregation(statement, plan, transcript))
        .transpose()?;

    transcript.append_serde(
        labels::ABSORB_LABRADOR_LINEAR_GARBAGE_PAYLOAD,
        &level.linear_garbage_payload,
    );
    let challenges = replay_amortize_challenges::<F, T, D>(transcript, virtual_row_count)?;
    tracing::debug!(
        level_index,
        tail = true,
        ?challenges,
        "labrador verifier amortize challenges"
    );
    let amortized_phi = if let Some((phi_stmt_orig, _)) = explicit_aggregation.as_ref() {
        let phi_stmt = reshape_phi_verifier::<F, D>(
            phi_stmt_orig,
            &level.input_row_lengths,
            &level.row_split_counts,
            virtual_row_len,
        )?;
        let mut phi_total = phi_stmt;
        add_phi_flat_in_place(&mut phi_total, &phi_jl_flat)?;
        combine_virtual_rows(&phi_total, &challenges, virtual_row_len)?
    } else {
        let plan = statement
            .reduced_constraints
            .as_deref()
            .ok_or(HachiError::InvalidProof)?;
        let aggregation = reduced_aggregation
            .as_ref()
            .ok_or(HachiError::InvalidProof)?;
        let mut amortized_phi = finalize_reduced_statement_aggregation(
            plan,
            aggregation,
            &level.input_row_lengths,
            &level.row_split_counts,
            virtual_row_len,
            &challenges,
        )?;
        let amortized_phi_jl = combine_flat_rows(&phi_jl_flat, &challenges, virtual_row_len)?;
        add_amortized_phi_in_place(&mut amortized_phi, &amortized_phi_jl)?;
        amortized_phi
    };
    let b_stmt = if let Some((_, b_stmt)) = explicit_aggregation {
        b_stmt
    } else {
        reduced_aggregation
            .as_ref()
            .ok_or(HachiError::InvalidProof)?
            .aggregated_rhs
    };
    let aggregated_rhs = b_stmt + b_jl;

    let (computed_norm, proj_norm) = tracing::info_span!("labrador::verify_tail_norms")
        .in_scope(|| (witness.norm(), projection_norm_sq(&level.jl_projection)));
    if computed_norm > level.next_witness_norm_sq {
        return Err(HachiError::InvalidProof);
    }
    let proj_bound = 256u128.saturating_mul(statement.witness_norm_bound_sq);
    if proj_norm > proj_bound {
        return Err(HachiError::InvalidProof);
    }

    let setup = tracing::info_span!("labrador::verify_tail_setup").in_scope(|| {
        LabradorSetupMatrices::new(
            &level.config,
            virtual_row_count,
            virtual_row_len,
            comkey_seed,
        )
    });
    let witness_i8 = tracing::info_span!("labrador::verify_tail_digit_cache")
        .in_scope(|| try_centered_i8_rows(witness.rows()));
    let (az, rhs) = tracing::info_span!("labrador::verify_tail_linear_check").in_scope(
        || -> Result<_, HachiError> {
            let az = mat_vec_mul_decomposed::<F, D>(
                &setup.a_mat,
                witness.rows(),
                witness_i8.as_deref(),
                level.config.witness_digit_bits,
            )?;
            let rhs = accumulate_decomposed_t_rhs::<F, D>(
                inner_opening_digits,
                virtual_row_count,
                level.config.inner_commit_rank,
                level.config.aux_digit_parts,
                level.config.aux_digit_bits as u32,
                &challenges,
            )?;
            Ok((az, rhs))
        },
    )?;
    if az != rhs {
        return Err(HachiError::InvalidProof);
    }

    let (lhs, rhs, diag_sum) = tracing::info_span!("labrador::verify_tail_quadratic_check")
        .in_scope(|| -> Result<_, HachiError> {
            let lhs = decomposed_dot_product::<F, D>(
                &amortized_phi,
                witness.rows(),
                witness_i8.as_deref(),
                level.config.witness_digit_bits,
            )?;
            let (rhs, diag_sum) = accumulate_decomposed_h_rhs::<F, D>(
                linear_garbage_digits,
                virtual_row_count,
                level.config.aux_digit_parts,
                level.config.aux_digit_bits as u32,
                &challenges,
            )?;
            Ok((lhs, rhs, diag_sum))
        })?;
    if lhs != rhs {
        return Err(HachiError::InvalidProof);
    }

    if diag_sum - aggregated_rhs != CyclotomicRing::<F, D>::zero() {
        return Err(HachiError::InvalidProof);
    }

    Ok(())
}

#[allow(clippy::too_many_lines)]
#[allow(dead_code)]
fn verify_single_level<F, T, const D: usize>(
    statement: &LabradorStatement<F, D>,
    level: &LabradorLevelProof<F, D>,
    witness: &LabradorWitness<F, D>,
    comkey_seed: &LabradorComKeySeed,
    transcript: &mut T,
) -> Result<(), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + FromSmallInt,
    T: Transcript<F>,
{
    let virtual_row_len = level.virtual_row_len;
    let virtual_row_count = validate_level_shape(level, false)?;
    let layout = NextWitnessLayout::new(virtual_row_count, &level.config);
    let expected_rows = layout.num_rows();
    if witness.rows().len() != expected_rows {
        return Err(HachiError::InvalidProof);
    }
    for row in witness.rows().iter().take(level.config.witness_digit_parts) {
        if row.len() != virtual_row_len {
            return Err(HachiError::InvalidProof);
        }
    }

    let aux = &witness.rows()[layout.aux_row];
    if aux.len() != layout.aux_row_len() {
        return Err(HachiError::InvalidProof);
    }
    let (inner_opening_digits, linear_garbage_digits) =
        aux.split_at(layout.inner_opening_digits_len);

    absorb_labrador_level_context(
        transcript,
        &LabradorLevelTranscriptContext {
            level_index: 0,
            tail: level.tail,
            input_row_lengths: level.input_row_lengths.clone(),
            witness_digit_parts: level.config.witness_digit_parts,
            witness_digit_bits: level.config.witness_digit_bits,
            aux_digit_parts: level.config.aux_digit_parts,
            aux_digit_bits: level.config.aux_digit_bits,
            inner_commit_rank: level.config.inner_commit_rank,
            outer_commit_rank: level.config.outer_commit_rank,
        },
    )?;
    transcript.append_serde(
        labels::ABSORB_LABRADOR_INNER_OPENING_PAYLOAD,
        &level.inner_opening_payload,
    );

    let virt_total_len = virtual_row_count * virtual_row_len;
    let jl_cols = virt_total_len * D;
    let jl_matrix =
        LabradorJlMatrix::replay_nonce_search::<F, T>(transcript, level.jl_nonce, jl_cols)?;
    absorb_labrador_jl_projection(transcript, &level.jl_projection);

    let virt_row_lengths = vec![virtual_row_len; virtual_row_count];
    let (phi_jl_flat, b_jl) = aggregate_jl_constraints_verifier(
        &virt_row_lengths,
        &level.jl_projection,
        &jl_matrix,
        &level.jl_lift_residuals,
        transcript,
    )?;
    let (phi_stmt_orig, b_stmt) =
        aggregate_statement(statement, &level.input_row_lengths, transcript)?;
    let phi_stmt = reshape_phi_verifier::<F, D>(
        &phi_stmt_orig,
        &level.input_row_lengths,
        &level.row_split_counts,
        virtual_row_len,
    )?;

    let mut phi_total = phi_stmt;
    add_phi_flat_in_place(&mut phi_total, &phi_jl_flat)?;
    let aggregated_rhs = b_stmt + b_jl;

    transcript.append_serde(
        labels::ABSORB_LABRADOR_LINEAR_GARBAGE_PAYLOAD,
        &level.linear_garbage_payload,
    );
    let challenges = replay_amortize_challenges::<F, T, D>(transcript, virtual_row_count)?;

    let z_parts: Vec<Vec<CyclotomicRing<F, D>>> = witness
        .rows()
        .iter()
        .take(level.config.witness_digit_parts)
        .cloned()
        .collect();
    let z = recompose_from_parts(&z_parts, level.config.witness_digit_bits as u32)?;

    let t_flat = recompose_flat(
        inner_opening_digits,
        level.config.aux_digit_parts,
        level.config.aux_digit_bits as u32,
    )?;
    let h_flat = recompose_flat(
        linear_garbage_digits,
        level.config.aux_digit_parts,
        level.config.aux_digit_bits as u32,
    )?;
    if t_flat.len() != virtual_row_count * level.config.inner_commit_rank {
        return Err(HachiError::InvalidProof);
    }
    if h_flat.len() != virtual_row_count * (virtual_row_count + 1) / 2 {
        return Err(HachiError::InvalidProof);
    }
    let mut t_by_row = Vec::with_capacity(virtual_row_count);
    for chunk in t_flat.chunks(level.config.inner_commit_rank) {
        t_by_row.push(chunk.to_vec());
    }

    if !statement.inner_opening_payload.is_empty()
        && statement.inner_opening_payload != level.inner_opening_payload
    {
        return Err(HachiError::InvalidProof);
    }
    if !statement.linear_garbage_payload.is_empty()
        && statement.linear_garbage_payload != level.linear_garbage_payload
    {
        return Err(HachiError::InvalidProof);
    }

    let setup = LabradorSetupMatrices::new(
        &level.config,
        virtual_row_count,
        virtual_row_len,
        comkey_seed,
    );

    if level.config.outer_commit_rank > 0 {
        let inner_opening_payload_check = mat_vec_mul(&setup.b_mat, inner_opening_digits);
        if inner_opening_payload_check != level.inner_opening_payload {
            return Err(HachiError::InvalidProof);
        }
        let linear_garbage_payload_check = mat_vec_mul(&setup.d_mat, linear_garbage_digits);
        if linear_garbage_payload_check != level.linear_garbage_payload {
            return Err(HachiError::InvalidProof);
        }
    } else {
        if level.inner_opening_payload != inner_opening_digits {
            return Err(HachiError::InvalidProof);
        }
        if level.linear_garbage_payload != linear_garbage_digits {
            return Err(HachiError::InvalidProof);
        }
    }

    let computed_norm = witness.norm();
    if computed_norm > level.next_witness_norm_sq {
        return Err(HachiError::InvalidProof);
    }

    if projection_norm_sq(&level.jl_projection)
        > 256u128.saturating_mul(statement.witness_norm_bound_sq)
    {
        return Err(HachiError::InvalidProof);
    }

    let az = mat_vec_mul(&setup.a_mat, &z);
    let mut rhs = vec![CyclotomicRing::<F, D>::zero(); level.config.inner_commit_rank];
    for (i, t_row) in t_by_row.iter().enumerate() {
        for k in 0..level.config.inner_commit_rank {
            t_row[k].mul_by_sparse_into(&challenges[i], &mut rhs[k]);
        }
    }
    if az != rhs {
        return Err(HachiError::InvalidProof);
    }

    let mut amortized_phi = vec![CyclotomicRing::<F, D>::zero(); virtual_row_len];
    for (i, phi_row) in phi_total.iter().enumerate() {
        for (j, elem) in phi_row.iter().enumerate() {
            elem.mul_by_sparse_into(&challenges[i], &mut amortized_phi[j]);
        }
    }
    let lhs = dot_product(&amortized_phi, &z);
    let mut rhs = CyclotomicRing::<F, D>::zero();
    let mut idx = 0usize;
    for i in 0..virtual_row_count {
        for j in i..virtual_row_count {
            rhs += h_flat[idx]
                .mul_by_sparse(&challenges[i])
                .mul_by_sparse(&challenges[j]);
            idx += 1;
        }
    }
    if lhs != rhs {
        return Err(HachiError::InvalidProof);
    }

    let mut diag_sum = CyclotomicRing::<F, D>::zero();
    for i in 0..virtual_row_count {
        let idx = pair_index(i, i, virtual_row_count);
        diag_sum += h_flat[idx];
    }
    if diag_sum - aggregated_rhs != CyclotomicRing::<F, D>::zero() {
        return Err(HachiError::InvalidProof);
    }

    Ok(())
}
fn projection_norm_sq(projection: &[i64; 256]) -> u128 {
    projection.iter().fold(0u128, |acc, &v| {
        let x = v as i128;
        let sq = x * x;
        acc.saturating_add(sq as u128)
    })
}

#[tracing::instrument(skip_all, name = "labrador::validate_level_shape")]
fn validate_level_shape<F: FieldCore, const D: usize>(
    level: &LabradorLevelProof<F, D>,
    expect_tail: bool,
) -> Result<usize, HachiError> {
    if level.tail != expect_tail || level.config.tail != expect_tail {
        return Err(HachiError::InvalidProof);
    }
    if level.config.witness_digit_parts == 0 || level.config.aux_digit_parts == 0 {
        return Err(HachiError::InvalidProof);
    }
    if expect_tail {
        if level.config.outer_commit_rank != 0 {
            return Err(HachiError::InvalidProof);
        }
    } else if level.config.outer_commit_rank == 0 {
        return Err(HachiError::InvalidProof);
    }
    validate_reshape_metadata(
        &level.input_row_lengths,
        &level.row_split_counts,
        level.virtual_row_len,
    )
}

fn validate_reshape_metadata(
    row_lengths: &[usize],
    row_split_counts: &[usize],
    virtual_row_len: usize,
) -> Result<usize, HachiError> {
    if row_lengths.is_empty() || row_split_counts.len() != row_lengths.len() || virtual_row_len == 0
    {
        return Err(HachiError::InvalidProof);
    }

    let mut virtual_row_count = 0usize;
    let mut grouped_len = 0usize;
    for (&row_len, &splits) in row_lengths.iter().zip(row_split_counts.iter()) {
        grouped_len = grouped_len
            .checked_add(row_len)
            .ok_or(HachiError::InvalidProof)?;
        if splits > 0 {
            let capacity = splits
                .checked_mul(virtual_row_len)
                .ok_or(HachiError::InvalidProof)?;
            if grouped_len > capacity {
                return Err(HachiError::InvalidProof);
            }
            virtual_row_count = virtual_row_count
                .checked_add(splits)
                .ok_or(HachiError::InvalidProof)?;
            grouped_len = 0;
        }
    }

    if grouped_len != 0 || virtual_row_count == 0 {
        return Err(HachiError::InvalidProof);
    }

    Ok(virtual_row_count)
}

#[tracing::instrument(skip_all, name = "labrador::mat_vec_mul_decomposed")]
fn mat_vec_mul_decomposed<F, const D: usize>(
    matrix: &[Vec<CyclotomicRing<F, D>>],
    parts: &[Vec<CyclotomicRing<F, D>>],
    parts_i8: Option<&[Vec<[i8; D]>]>,
    log_basis: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
{
    if parts.is_empty() {
        return Err(HachiError::InvalidProof);
    }

    if let Some(parts_i8) = parts_i8 {
        if let Ok(images) = mat_vec_mul_crt_ntt_i8_many(matrix, parts_i8) {
            let mut acc = vec![CyclotomicRing::<F, D>::zero(); matrix.len()];
            for (part_idx, image) in images.into_iter().enumerate() {
                let scale = pow2_field::<F>(part_idx * log_basis);
                for (dst, src) in acc.iter_mut().zip(image.iter()) {
                    *dst += src.scale(&scale);
                }
            }
            return Ok(acc);
        }
    }

    let mut acc = vec![CyclotomicRing::<F, D>::zero(); matrix.len()];
    for (part_idx, part) in parts.iter().enumerate() {
        let image = mat_vec_mul(matrix, part);
        let scale = pow2_field::<F>(part_idx * log_basis);
        for (dst, src) in acc.iter_mut().zip(image.iter()) {
            *dst += src.scale(&scale);
        }
    }
    Ok(acc)
}

#[tracing::instrument(skip_all, name = "labrador::decomposed_dot_product")]
fn decomposed_dot_product<F, const D: usize>(
    lhs: &[CyclotomicRing<F, D>],
    parts: &[Vec<CyclotomicRing<F, D>>],
    parts_i8: Option<&[Vec<[i8; D]>]>,
    log_basis: usize,
) -> Result<CyclotomicRing<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
{
    if parts.is_empty() {
        return Err(HachiError::InvalidProof);
    }

    if let Some(parts_i8) = parts_i8 {
        if let Ok(images) = mat_vec_mul_crt_ntt_i8_many(&[lhs.to_vec()], parts_i8) {
            let mut acc = CyclotomicRing::<F, D>::zero();
            for (part_idx, image) in images.into_iter().enumerate() {
                let scale = pow2_field::<F>(part_idx * log_basis);
                let value = image.into_iter().next().ok_or(HachiError::InvalidProof)?;
                acc += value.scale(&scale);
            }
            return Ok(acc);
        }
    }

    let mut acc = CyclotomicRing::<F, D>::zero();
    for (part_idx, part) in parts.iter().enumerate() {
        if part.len() != lhs.len() {
            return Err(HachiError::InvalidProof);
        }
        let scale = pow2_field::<F>(part_idx * log_basis);
        acc += dot_product(lhs, part).scale(&scale);
    }
    Ok(acc)
}

fn recompose_digit_chunk<F: FieldCore + CanonicalField, const D: usize>(
    flat: &[CyclotomicRing<F, D>],
    index: usize,
    parts: usize,
    log_basis: u32,
) -> Result<CyclotomicRing<F, D>, HachiError> {
    let start = index.checked_mul(parts).ok_or(HachiError::InvalidProof)?;
    let end = start.checked_add(parts).ok_or(HachiError::InvalidProof)?;
    if end > flat.len() {
        return Err(HachiError::InvalidProof);
    }
    Ok(CyclotomicRing::gadget_recompose_pow2(
        &flat[start..end],
        log_basis,
    ))
}

#[tracing::instrument(skip_all, name = "labrador::accumulate_decomposed_t_rhs")]
fn accumulate_decomposed_t_rhs<F: FieldCore + CanonicalField, const D: usize>(
    inner_opening_digits: &[CyclotomicRing<F, D>],
    virtual_row_count: usize,
    inner_commit_rank: usize,
    parts: usize,
    log_basis: u32,
    challenges: &[SparseChallenge],
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
    if challenges.len() != virtual_row_count
        || inner_opening_digits.len() != virtual_row_count * inner_commit_rank * parts
    {
        return Err(HachiError::InvalidProof);
    }
    let mut rhs = vec![CyclotomicRing::<F, D>::zero(); inner_commit_rank];
    for (row_idx, challenge) in challenges.iter().enumerate() {
        for (k, rhs_k) in rhs.iter_mut().enumerate() {
            let inner_opening = recompose_digit_chunk(
                inner_opening_digits,
                row_idx * inner_commit_rank + k,
                parts,
                log_basis,
            )?;
            inner_opening.mul_by_sparse_into(challenge, rhs_k);
        }
    }
    Ok(rhs)
}

#[tracing::instrument(skip_all, name = "labrador::accumulate_decomposed_h_rhs")]
fn accumulate_decomposed_h_rhs<F: FieldCore + CanonicalField, const D: usize>(
    linear_garbage_digits: &[CyclotomicRing<F, D>],
    virtual_row_count: usize,
    parts: usize,
    log_basis: u32,
    challenges: &[SparseChallenge],
) -> Result<(CyclotomicRing<F, D>, CyclotomicRing<F, D>), HachiError> {
    let pair_count = virtual_row_count
        .checked_mul(virtual_row_count + 1)
        .and_then(|v| v.checked_div(2))
        .ok_or(HachiError::InvalidProof)?;
    if challenges.len() != virtual_row_count || linear_garbage_digits.len() != pair_count * parts {
        return Err(HachiError::InvalidProof);
    }
    let mut rhs = CyclotomicRing::<F, D>::zero();
    let mut diag_sum = CyclotomicRing::<F, D>::zero();
    for i in 0..virtual_row_count {
        for j in i..virtual_row_count {
            let idx = pair_index(i, j, virtual_row_count);
            let linear_garbage =
                recompose_digit_chunk(linear_garbage_digits, idx, parts, log_basis)?;
            rhs += linear_garbage
                .mul_by_sparse(&challenges[i])
                .mul_by_sparse(&challenges[j]);
            if i == j {
                diag_sum += linear_garbage;
            }
        }
    }
    Ok((rhs, diag_sum))
}

#[tracing::instrument(skip_all, name = "labrador::recompose_from_parts")]
fn recompose_from_parts<F: FieldCore + CanonicalField, const D: usize>(
    parts: &[Vec<CyclotomicRing<F, D>>],
    log_basis: u32,
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
    if parts.is_empty() {
        return Err(HachiError::InvalidProof);
    }
    let len = parts[0].len();
    for row in parts.iter().skip(1) {
        if row.len() != len {
            return Err(HachiError::InvalidProof);
        }
    }
    let mut out = Vec::with_capacity(len);
    for idx in 0..len {
        let mut slice = Vec::with_capacity(parts.len());
        for part in parts {
            slice.push(part[idx]);
        }
        out.push(CyclotomicRing::gadget_recompose_pow2(&slice, log_basis));
    }
    Ok(out)
}

#[tracing::instrument(skip_all, name = "labrador::recompose_flat")]
fn recompose_flat<F: FieldCore + CanonicalField, const D: usize>(
    flat: &[CyclotomicRing<F, D>],
    parts: usize,
    log_basis: u32,
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
    if parts == 0 || flat.len() % parts != 0 {
        return Err(HachiError::InvalidProof);
    }
    let mut out = Vec::with_capacity(flat.len() / parts);
    for chunk in flat.chunks(parts) {
        out.push(CyclotomicRing::gadget_recompose_pow2(chunk, log_basis));
    }
    Ok(out)
}

#[tracing::instrument(skip_all, name = "labrador::add_phi_flat_in_place_verifier")]
fn add_phi_flat_in_place<F: FieldCore, const D: usize>(
    acc: &mut [Vec<CyclotomicRing<F, D>>],
    other_flat: &[CyclotomicRing<F, D>],
) -> Result<(), HachiError> {
    let mut cursor = 0usize;
    for row_acc in acc.iter_mut() {
        let end = cursor + row_acc.len();
        if end > other_flat.len() {
            return Err(HachiError::InvalidProof);
        }
        for (a, b) in row_acc.iter_mut().zip(other_flat[cursor..end].iter()) {
            *a += *b;
        }
        cursor = end;
    }
    if cursor != other_flat.len() {
        return Err(HachiError::InvalidProof);
    }
    Ok(())
}

fn dot_product<F: FieldCore, const D: usize>(
    lhs: &[CyclotomicRing<F, D>],
    rhs: &[CyclotomicRing<F, D>],
) -> CyclotomicRing<F, D> {
    let mut acc = CyclotomicRing::<F, D>::zero();
    let len = lhs.len().min(rhs.len());
    for i in 0..len {
        acc += lhs[i] * rhs[i];
    }
    acc
}

#[tracing::instrument(skip_all, name = "labrador::verify_constraints")]
fn verify_constraints<F: FieldCore + CanonicalField + FromSmallInt, const D: usize>(
    constraints: &[LabradorConstraint<F, D>],
    witness: &LabradorWitness<F, D>,
) -> Result<(), HachiError> {
    for (idx, cnst) in constraints.iter().enumerate() {
        let mut lhs = CyclotomicRing::<F, D>::zero();

        for term in &cnst.terms {
            if term.row >= witness.rows().len() {
                return Err(HachiError::InvalidProof);
            }
            let row = &witness.rows()[term.row];
            if term.offset + term.coefficients.len() > row.len() {
                return Err(HachiError::InvalidProof);
            }
            for (j, coeff) in term.coefficients.iter().enumerate() {
                lhs += *coeff * row[term.offset + j];
            }
        }

        if lhs != cnst.target {
            return Err(HachiError::InvalidInput(format!(
                "Labrador constraint {idx} not satisfied"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::Fp64;
    use crate::algebra::ring::CyclotomicRing;
    use crate::protocol::labrador::LabradorConstraintTerm;
    use crate::protocol::transcript::labels::DOMAIN_LABRADOR_RECURSION;
    use crate::protocol::transcript::Blake2bTranscript;
    use crate::FromSmallInt;

    type F = Fp64<4294967197>;
    const D: usize = 64;

    #[test]
    fn verify_accepts_basic_linear_constraint() {
        let row = vec![CyclotomicRing::<F, D>::from_coefficients(
            std::array::from_fn(|i| if i == 0 { F::from_i64(3) } else { F::zero() }),
        )];
        let witness = LabradorWitness::new(vec![row.clone()]);
        let coeff = vec![CyclotomicRing::one()];
        let target = CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|i| {
            if i == 0 {
                F::from_i64(3)
            } else {
                F::zero()
            }
        }));
        let constraint =
            LabradorConstraint::new(vec![LabradorConstraintTerm::new(0, 0, coeff)], target);
        let statement = LabradorStatement {
            inner_opening_payload: Vec::new(),
            linear_garbage_payload: Vec::new(),
            challenges: Vec::new(),
            constraints: vec![constraint],
            reduced_constraints: None,
            witness_norm_bound_sq: 1000,
        };
        let proof = LabradorProof {
            levels: Vec::new(),
            final_opening_witness: witness.clone(),
        };
        let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_RECURSION);
        let out = verify(&statement, &proof, &[1u8; 32], &mut transcript).unwrap();
        assert_eq!(out.final_opening_witness, witness);
    }
}
