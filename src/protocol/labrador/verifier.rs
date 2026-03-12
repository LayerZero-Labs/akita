//! Labrador verifier/reducer loop.

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::protocol::labrador::aggregation::{
    aggregate_jl_constraints_verifier, aggregate_statement,
};
use crate::protocol::labrador::comkey::LabradorComKeySeed;
use crate::protocol::labrador::constraints::{
    build_next_constraint_plan, materialize_reduced_constraints, pair_index, LabradorConstraint,
    NextWitnessLayout,
};
use crate::protocol::labrador::guardrails::LABRADOR_MAX_LEVELS;
use crate::protocol::labrador::johnson_lindenstrauss::LabradorJlMatrix;
use crate::protocol::labrador::setup::LabradorSetup;
use crate::protocol::labrador::transcript::{
    absorb_labrador_jl_projection, absorb_labrador_level_context, LabradorLevelTranscriptContext,
};
use crate::protocol::labrador::types::{
    LabradorLevelProof, LabradorProof, LabradorStatement, LabradorWitness,
};
use crate::protocol::labrador::utils::mat_vec_mul;
use crate::protocol::transcript::labels;
use crate::protocol::transcript::{challenge_ring_element_rejection_sampled, Transcript};
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
        if final_norm > initial_statement.beta_sq {
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
    if final_norm > statement.beta_sq {
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
        materialize_reduced_constraints(plan, &statement.u1, &statement.u2)
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
    let rr = validate_level_shape(level, false)?;
    let nn = level.nn;
    let virt_row_lengths = vec![nn; rr];

    absorb_labrador_level_context(
        transcript,
        &LabradorLevelTranscriptContext {
            level_index,
            tail: level.tail,
            input_row_lengths: level.input_row_lengths.clone(),
            f: level.config.f,
            b: level.config.b,
            fu: level.config.fu,
            bu: level.config.bu,
            kappa: level.config.kappa,
            kappa1: level.config.kappa1,
        },
    )?;
    transcript.append_serde(labels::ABSORB_LABRADOR_U1, &level.u1);

    let total_len: usize = virt_row_lengths.iter().sum();
    let jl_cols = total_len * D;
    let jl_matrix =
        LabradorJlMatrix::replay_nonce_search::<F, T>(transcript, level.jl_nonce, jl_cols)?;
    absorb_labrador_jl_projection(transcript, &level.jl_projection);

    let (phi_jl, b_jl) = aggregate_jl_constraints_verifier(
        &virt_row_lengths,
        &level.jl_projection,
        &jl_matrix,
        &level.bb,
        transcript,
    )?;
    let (phi_stmt_orig, b_stmt) =
        aggregate_statement(statement, &level.input_row_lengths, transcript)?;
    let phi_stmt =
        reshape_phi_verifier::<F, D>(&phi_stmt_orig, &level.input_row_lengths, &level.nu, nn)?;

    let mut phi_total = phi_stmt;
    add_phi_in_place(&mut phi_total, &phi_jl)?;
    let b_total = b_stmt + b_jl;

    transcript.append_serde(labels::ABSORB_LABRADOR_U2, &level.u2);
    let challenges = replay_amortize_challenges(transcript, rr)?;

    let setup = Arc::new(LabradorSetup::new(&level.config, rr, nn, comkey_seed));
    let reduced_constraints = build_next_constraint_plan(
        &phi_total,
        &b_total,
        &challenges,
        &virt_row_lengths,
        nn,
        &level.config,
        Arc::clone(&setup),
    )
    .map_err(|_| HachiError::InvalidProof)?;

    Ok(LabradorStatement {
        u1: level.u1.clone(),
        u2: level.u2.clone(),
        challenges,
        constraints: Vec::new(),
        reduced_constraints: Some(Box::new(reduced_constraints)),
        beta_sq: level.norm_sq,
    })
}

#[tracing::instrument(skip_all, name = "labrador::reshape_phi_verifier")]
fn reshape_phi_verifier<F: FieldCore, const D: usize>(
    phi: &[Vec<CyclotomicRing<F, D>>],
    row_lengths: &[usize],
    nu: &[usize],
    nn: usize,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, HachiError> {
    let rr = validate_reshape_metadata(row_lengths, nu, nn)?;
    let mut result = Vec::new();
    let mut group: Vec<CyclotomicRing<F, D>> = Vec::new();

    for (i, row) in phi.iter().enumerate() {
        if i >= row_lengths.len() || row.len() != row_lengths[i] {
            return Err(HachiError::InvalidProof);
        }
        group.extend(row.iter().copied());
        let splits = if i < nu.len() { nu[i] } else { 0 };
        if splits > 0 {
            if group.len() > splits * nn {
                return Err(HachiError::InvalidProof);
            }
            for chunk_idx in 0..splits {
                let start = chunk_idx * nn;
                let mut virtual_row = vec![CyclotomicRing::<F, D>::zero(); nn];
                for (j, val) in group.iter().enumerate().skip(start).take(nn) {
                    virtual_row[j - start] = *val;
                }
                result.push(virtual_row);
            }
            group.clear();
        }
    }
    if !group.is_empty() || result.len() != rr {
        return Err(HachiError::InvalidProof);
    }
    Ok(result)
}

#[tracing::instrument(skip_all, name = "labrador::replay_amortize_challenges")]
fn replay_amortize_challenges<F, T, const D: usize>(
    transcript: &mut T,
    rows: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
    T: Transcript<F>,
{
    let mut challenges = Vec::with_capacity(rows);
    for _ in 0..rows {
        challenges.push(challenge_ring_element_rejection_sampled(
            transcript,
            labels::CHALLENGE_LABRADOR_AMORTIZE,
        )?);
    }
    Ok(challenges)
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
    let nn = level.nn;
    let rr = validate_level_shape(level, true)?;

    if witness.rows().len() != level.config.f {
        return Err(HachiError::InvalidProof);
    }
    for row in witness.rows().iter() {
        if row.len() != nn {
            return Err(HachiError::InvalidProof);
        }
    }

    let t_hat_len = rr * level.config.kappa * level.config.fu;
    let h_hat_len = rr * (rr + 1) / 2 * level.config.fu;
    if level.u1.len() != t_hat_len || level.u2.len() != h_hat_len {
        return Err(HachiError::InvalidProof);
    }
    let t_hat = &level.u1;
    let h_hat = &level.u2;

    absorb_labrador_level_context(
        transcript,
        &LabradorLevelTranscriptContext {
            level_index,
            tail: level.tail,
            input_row_lengths: level.input_row_lengths.clone(),
            f: level.config.f,
            b: level.config.b,
            fu: level.config.fu,
            bu: level.config.bu,
            kappa: level.config.kappa,
            kappa1: level.config.kappa1,
        },
    )?;

    transcript.append_serde(labels::ABSORB_LABRADOR_U1, &level.u1);

    let virt_total_len = rr * nn;
    let jl_cols = virt_total_len * D;
    let jl_matrix =
        LabradorJlMatrix::replay_nonce_search::<F, T>(transcript, level.jl_nonce, jl_cols)?;

    absorb_labrador_jl_projection(transcript, &level.jl_projection);

    let virt_row_lengths = vec![nn; rr];
    let (phi_jl, b_jl) = aggregate_jl_constraints_verifier(
        &virt_row_lengths,
        &level.jl_projection,
        &jl_matrix,
        &level.bb,
        transcript,
    )?;

    let (phi_stmt_orig, b_stmt) =
        aggregate_statement(statement, &level.input_row_lengths, transcript)?;
    let phi_stmt =
        reshape_phi_verifier::<F, D>(&phi_stmt_orig, &level.input_row_lengths, &level.nu, nn)?;

    let mut phi_total = phi_stmt;
    add_phi_in_place(&mut phi_total, &phi_jl)?;
    let b_total = b_stmt + b_jl;

    transcript.append_serde(labels::ABSORB_LABRADOR_U2, &level.u2);
    let challenges = replay_amortize_challenges(transcript, rr)?;

    let z_parts: Vec<Vec<CyclotomicRing<F, D>>> = witness.rows().to_vec();
    let z = recompose_from_parts(&z_parts, level.config.b as u32)?;
    let t_flat = recompose_flat(t_hat, level.config.fu, level.config.bu as u32)?;
    let h_flat = recompose_flat(h_hat, level.config.fu, level.config.bu as u32)?;
    if t_flat.len() != rr * level.config.kappa || h_flat.len() != rr * (rr + 1) / 2 {
        return Err(HachiError::InvalidProof);
    }

    let computed_norm = witness.norm();
    if computed_norm > level.norm_sq {
        return Err(HachiError::InvalidProof);
    }
    let proj_norm = projection_norm_sq(&level.jl_projection);
    let proj_bound = 256u128.saturating_mul(statement.beta_sq);
    if proj_norm > proj_bound {
        return Err(HachiError::InvalidProof);
    }

    let setup: LabradorSetup<F, D> = LabradorSetup::new(&level.config, rr, nn, comkey_seed);
    let az = mat_vec_mul(&setup.a_mat, &z);
    let mut rhs = vec![CyclotomicRing::<F, D>::zero(); level.config.kappa];
    for (i, t_row) in t_flat.chunks(level.config.kappa).enumerate() {
        let c = challenges[i];
        for k in 0..level.config.kappa {
            rhs[k] += c * t_row[k];
        }
    }
    if az != rhs {
        return Err(HachiError::InvalidProof);
    }

    let mut combined_phi = vec![CyclotomicRing::<F, D>::zero(); nn];
    for (i, phi_row) in phi_total.iter().enumerate() {
        let c = challenges[i];
        for (j, elem) in phi_row.iter().enumerate() {
            combined_phi[j] += c * *elem;
        }
    }
    let lhs = dot_product(&combined_phi, &z);
    let mut rhs = CyclotomicRing::<F, D>::zero();
    let mut idx = 0usize;
    for i in 0..rr {
        for j in i..rr {
            rhs += challenges[i] * challenges[j] * h_flat[idx];
            idx += 1;
        }
    }
    if lhs != rhs {
        return Err(HachiError::InvalidProof);
    }

    let mut diag_sum = CyclotomicRing::<F, D>::zero();
    for i in 0..rr {
        let idx = pair_index(i, i, rr);
        diag_sum += h_flat[idx];
    }
    if diag_sum - b_total != CyclotomicRing::<F, D>::zero() {
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
    let nn = level.nn;
    let rr = validate_level_shape(level, false)?;
    let layout = NextWitnessLayout::new(rr, &level.config);
    let expected_rows = layout.num_rows();
    if witness.rows().len() != expected_rows {
        return Err(HachiError::InvalidProof);
    }
    for row in witness.rows().iter().take(level.config.f) {
        if row.len() != nn {
            return Err(HachiError::InvalidProof);
        }
    }

    let aux = &witness.rows()[layout.aux_row];
    if aux.len() != layout.aux_row_len() {
        return Err(HachiError::InvalidProof);
    }
    let (t_hat, h_hat) = aux.split_at(layout.t_hat_len);

    absorb_labrador_level_context(
        transcript,
        &LabradorLevelTranscriptContext {
            level_index: 0,
            tail: level.tail,
            input_row_lengths: level.input_row_lengths.clone(),
            f: level.config.f,
            b: level.config.b,
            fu: level.config.fu,
            bu: level.config.bu,
            kappa: level.config.kappa,
            kappa1: level.config.kappa1,
        },
    )?;
    transcript.append_serde(labels::ABSORB_LABRADOR_U1, &level.u1);

    let virt_total_len = rr * nn;
    let jl_cols = virt_total_len * D;
    let jl_matrix =
        LabradorJlMatrix::replay_nonce_search::<F, T>(transcript, level.jl_nonce, jl_cols)?;
    absorb_labrador_jl_projection(transcript, &level.jl_projection);

    let virt_row_lengths = vec![nn; rr];
    let (phi_jl, b_jl) = aggregate_jl_constraints_verifier(
        &virt_row_lengths,
        &level.jl_projection,
        &jl_matrix,
        &level.bb,
        transcript,
    )?;
    let (phi_stmt_orig, b_stmt) =
        aggregate_statement(statement, &level.input_row_lengths, transcript)?;
    let phi_stmt =
        reshape_phi_verifier::<F, D>(&phi_stmt_orig, &level.input_row_lengths, &level.nu, nn)?;

    let mut phi_total = phi_stmt;
    add_phi_in_place(&mut phi_total, &phi_jl)?;
    let b_total = b_stmt + b_jl;

    transcript.append_serde(labels::ABSORB_LABRADOR_U2, &level.u2);
    let challenges = replay_amortize_challenges(transcript, rr)?;

    let z_parts: Vec<Vec<CyclotomicRing<F, D>>> = witness
        .rows()
        .iter()
        .take(level.config.f)
        .cloned()
        .collect();
    let z = recompose_from_parts(&z_parts, level.config.b as u32)?;

    let t_flat = recompose_flat(t_hat, level.config.fu, level.config.bu as u32)?;
    let h_flat = recompose_flat(h_hat, level.config.fu, level.config.bu as u32)?;
    if t_flat.len() != rr * level.config.kappa {
        return Err(HachiError::InvalidProof);
    }
    if h_flat.len() != rr * (rr + 1) / 2 {
        return Err(HachiError::InvalidProof);
    }
    let mut t_by_row = Vec::with_capacity(rr);
    for chunk in t_flat.chunks(level.config.kappa) {
        t_by_row.push(chunk.to_vec());
    }

    if !statement.u1.is_empty() && statement.u1 != level.u1 {
        return Err(HachiError::InvalidProof);
    }
    if !statement.u2.is_empty() && statement.u2 != level.u2 {
        return Err(HachiError::InvalidProof);
    }

    let setup = LabradorSetup::new(&level.config, rr, nn, comkey_seed);

    if level.config.kappa1 > 0 {
        let u1_check = mat_vec_mul(&setup.b_mat, t_hat);
        if u1_check != level.u1 {
            return Err(HachiError::InvalidProof);
        }
        let u2_check = mat_vec_mul(&setup.d_mat, h_hat);
        if u2_check != level.u2 {
            return Err(HachiError::InvalidProof);
        }
    } else {
        if level.u1 != t_hat {
            return Err(HachiError::InvalidProof);
        }
        if level.u2 != h_hat {
            return Err(HachiError::InvalidProof);
        }
    }

    let computed_norm = witness.norm();
    if computed_norm > level.norm_sq {
        return Err(HachiError::InvalidProof);
    }

    if projection_norm_sq(&level.jl_projection) > 256u128.saturating_mul(statement.beta_sq) {
        return Err(HachiError::InvalidProof);
    }

    let az = mat_vec_mul(&setup.a_mat, &z);
    let mut rhs = vec![CyclotomicRing::<F, D>::zero(); level.config.kappa];
    for (i, t_row) in t_by_row.iter().enumerate() {
        let c = challenges[i];
        for k in 0..level.config.kappa {
            rhs[k] += c * t_row[k];
        }
    }
    if az != rhs {
        return Err(HachiError::InvalidProof);
    }

    let mut combined_phi = vec![CyclotomicRing::<F, D>::zero(); nn];
    for (i, phi_row) in phi_total.iter().enumerate() {
        let c = challenges[i];
        for (j, elem) in phi_row.iter().enumerate() {
            combined_phi[j] += c * *elem;
        }
    }
    let lhs = dot_product(&combined_phi, &z);
    let mut rhs = CyclotomicRing::<F, D>::zero();
    let mut idx = 0usize;
    for i in 0..rr {
        for j in i..rr {
            rhs += challenges[i] * challenges[j] * h_flat[idx];
            idx += 1;
        }
    }
    if lhs != rhs {
        return Err(HachiError::InvalidProof);
    }

    let mut diag_sum = CyclotomicRing::<F, D>::zero();
    for i in 0..rr {
        let idx = pair_index(i, i, rr);
        diag_sum += h_flat[idx];
    }
    if diag_sum - b_total != CyclotomicRing::<F, D>::zero() {
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
    if level.config.f == 0 || level.config.fu == 0 {
        return Err(HachiError::InvalidProof);
    }
    if expect_tail {
        if level.config.kappa1 != 0 {
            return Err(HachiError::InvalidProof);
        }
    } else if level.config.kappa1 == 0 {
        return Err(HachiError::InvalidProof);
    }
    validate_reshape_metadata(&level.input_row_lengths, &level.nu, level.nn)
}

fn validate_reshape_metadata(
    row_lengths: &[usize],
    nu: &[usize],
    nn: usize,
) -> Result<usize, HachiError> {
    if row_lengths.is_empty() || nu.len() != row_lengths.len() || nn == 0 {
        return Err(HachiError::InvalidProof);
    }

    let mut rr = 0usize;
    let mut grouped_len = 0usize;
    for (&row_len, &splits) in row_lengths.iter().zip(nu.iter()) {
        grouped_len = grouped_len
            .checked_add(row_len)
            .ok_or(HachiError::InvalidProof)?;
        if splits > 0 {
            let capacity = splits.checked_mul(nn).ok_or(HachiError::InvalidProof)?;
            if grouped_len > capacity {
                return Err(HachiError::InvalidProof);
            }
            rr = rr.checked_add(splits).ok_or(HachiError::InvalidProof)?;
            grouped_len = 0;
        }
    }

    if grouped_len != 0 || rr == 0 {
        return Err(HachiError::InvalidProof);
    }

    Ok(rr)
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

#[tracing::instrument(skip_all, name = "labrador::add_phi_in_place_verifier")]
fn add_phi_in_place<F: FieldCore, const D: usize>(
    acc: &mut [Vec<CyclotomicRing<F, D>>],
    other: &[Vec<CyclotomicRing<F, D>>],
) -> Result<(), HachiError> {
    if acc.len() != other.len() {
        return Err(HachiError::InvalidProof);
    }
    for (row_acc, row_other) in acc.iter_mut().zip(other.iter()) {
        if row_acc.len() != row_other.len() {
            return Err(HachiError::InvalidProof);
        }
        for (a, b) in row_acc.iter_mut().zip(row_other.iter()) {
            *a += *b;
        }
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
    use crate::protocol::transcript::labels::DOMAIN_LABRADOR_PROTOCOL;
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
            u1: Vec::new(),
            u2: Vec::new(),
            challenges: Vec::new(),
            constraints: vec![constraint],
            reduced_constraints: None,
            beta_sq: 1000,
        };
        let proof = LabradorProof {
            levels: Vec::new(),
            final_opening_witness: witness.clone(),
        };
        let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        let out = verify(&statement, &proof, &[1u8; 32], &mut transcript).unwrap();
        assert_eq!(out.final_opening_witness, witness);
    }
}
