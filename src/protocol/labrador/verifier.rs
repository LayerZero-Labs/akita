//! Labrador verifier/reducer loop.

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::protocol::labrador::aggregation::{
    aggregate_jl_constraints_verifier, aggregate_statement_constraints,
};
use crate::protocol::labrador::comkey::LabradorComKeySeed;
use crate::protocol::labrador::guardrails::LABRADOR_MAX_LEVELS;
use crate::protocol::labrador::johnson_lindenstrauss::LabradorJlMatrix;
use crate::protocol::labrador::setup::LabradorSetup;
use crate::protocol::labrador::transcript::{
    absorb_labrador_jl_projection, absorb_labrador_level_context, LabradorLevelTranscriptContext,
};
use crate::protocol::labrador::types::{
    LabradorConstraint, LabradorLevelProof, LabradorProof, LabradorReductionConfig,
    LabradorStatement, LabradorWitness,
};
use crate::protocol::labrador::utils::mat_vec_mul;
use crate::protocol::transcript::labels;
use crate::protocol::transcript::{challenge_ring_element_rejection_sampled, Transcript};
use crate::{CanonicalField, FieldCore, FieldSampling, FromSmallInt};

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
        verify_constraints(&initial_statement.constraints, &proof.final_opening_witness)?;
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
    verify_constraints(&statement.constraints, &proof.final_opening_witness)?;

    Ok(LabradorVerifyResult {
        terminal_statement: statement,
        final_opening_witness: proof.final_opening_witness.clone(),
    })
}

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
    if level.tail {
        return Err(HachiError::InvalidProof);
    }
    let r = level.input_row_lengths.len();
    if r == 0 || level.input_row_chunks.len() != r {
        return Err(HachiError::InvalidProof);
    }
    if level.config.f == 0 || level.config.fu == 0 {
        return Err(HachiError::InvalidProof);
    }
    let max_len = level.input_row_lengths.iter().copied().max().unwrap_or(0);

    absorb_labrador_level_context(
        transcript,
        &LabradorLevelTranscriptContext {
            level_index,
            tail: level.tail,
            input_row_lengths: level.input_row_lengths.clone(),
            input_row_chunks: level.input_row_chunks.clone(),
            f: level.config.f,
            b: level.config.b,
            fu: level.config.fu,
            bu: level.config.bu,
            kappa: level.config.kappa,
            kappa1: level.config.kappa1,
        },
    )?;
    transcript.append_serde(labels::ABSORB_LABRADOR_U1, &level.u1);

    let total_len: usize = level.input_row_lengths.iter().sum();
    let jl_cols = total_len * D;
    let jl_matrix =
        LabradorJlMatrix::replay_nonce_search::<F, T>(transcript, level.jl_nonce, jl_cols)?;
    absorb_labrador_jl_projection(transcript, &level.jl_projection);

    let (phi_jl, b_jl) = aggregate_jl_constraints_verifier(
        &level.input_row_lengths,
        &level.jl_projection,
        &jl_matrix,
        &level.bb,
        transcript,
    )?;
    let (phi_stmt, b_stmt) = aggregate_statement_constraints(
        &statement.constraints,
        &level.input_row_lengths,
        transcript,
    )?;

    let mut phi_total = phi_stmt;
    add_phi_in_place(&mut phi_total, &phi_jl)?;
    let b_total = b_stmt + b_jl;

    transcript.append_serde(labels::ABSORB_LABRADOR_U2, &level.u2);
    let mut challenges = Vec::with_capacity(r);
    for _ in 0..r {
        challenges.push(challenge_ring_element_rejection_sampled(
            transcript,
            labels::CHALLENGE_LABRADOR_AMORTIZE,
        )?);
    }

    let setup = LabradorSetup::new(&level.config, r, max_len, comkey_seed);
    let next_constraints = build_next_constraints(
        &phi_total,
        &b_total,
        &challenges,
        &level.input_row_lengths,
        max_len,
        &level.config,
        &level.u1,
        &level.u2,
        &setup,
    )?;

    Ok(LabradorStatement {
        u1: level.u1.clone(),
        u2: level.u2.clone(),
        challenges,
        constraints: next_constraints,
        beta_sq: level.norm_sq,
        hash: [0u8; 16],
    })
}

#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
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
    if !level.tail {
        return Err(HachiError::InvalidProof);
    }
    let r = level.input_row_lengths.len();
    if r == 0 || level.input_row_chunks.len() != r {
        return Err(HachiError::InvalidProof);
    }
    if level.config.f == 0 || level.config.fu == 0 {
        return Err(HachiError::InvalidProof);
    }
    let max_len = level.input_row_lengths.iter().copied().max().unwrap_or(0);
    if witness.rows().len() != level.config.f {
        return Err(HachiError::InvalidProof);
    }
    for row in witness.rows() {
        if row.len() != max_len {
            return Err(HachiError::InvalidProof);
        }
    }

    let t_hat_len = r * level.config.kappa * level.config.fu;
    let h_hat_len = r * (r + 1) / 2 * level.config.fu;
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
            input_row_chunks: level.input_row_chunks.clone(),
            f: level.config.f,
            b: level.config.b,
            fu: level.config.fu,
            bu: level.config.bu,
            kappa: level.config.kappa,
            kappa1: level.config.kappa1,
        },
    )?;
    transcript.append_serde(labels::ABSORB_LABRADOR_U1, &level.u1);

    let total_len: usize = level.input_row_lengths.iter().sum();
    let jl_cols = total_len * D;
    let jl_matrix =
        LabradorJlMatrix::replay_nonce_search::<F, T>(transcript, level.jl_nonce, jl_cols)?;
    absorb_labrador_jl_projection(transcript, &level.jl_projection);

    let (phi_jl, b_jl) = aggregate_jl_constraints_verifier(
        &level.input_row_lengths,
        &level.jl_projection,
        &jl_matrix,
        &level.bb,
        transcript,
    )?;
    let (phi_stmt, b_stmt) = aggregate_statement_constraints(
        &statement.constraints,
        &level.input_row_lengths,
        transcript,
    )?;
    let mut phi_total = phi_stmt;
    add_phi_in_place(&mut phi_total, &phi_jl)?;
    let b_total = b_stmt + b_jl;

    transcript.append_serde(labels::ABSORB_LABRADOR_U2, &level.u2);
    let mut challenges = Vec::with_capacity(r);
    for _ in 0..r {
        challenges.push(challenge_ring_element_rejection_sampled(
            transcript,
            labels::CHALLENGE_LABRADOR_AMORTIZE,
        )?);
    }

    let z_parts: Vec<Vec<CyclotomicRing<F, D>>> = witness.rows().to_vec();
    let z = recompose_from_parts(&z_parts, level.config.b as u32)?;
    let t_flat = recompose_flat(t_hat, level.config.fu, level.config.bu as u32)?;
    let h_flat = recompose_flat(h_hat, level.config.fu, level.config.bu as u32)?;
    if t_flat.len() != r * level.config.kappa || h_flat.len() != r * (r + 1) / 2 {
        return Err(HachiError::InvalidProof);
    }

    let computed_norm = witness.norm();
    if computed_norm > level.norm_sq {
        return Err(HachiError::InvalidProof);
    }
    if projection_norm_sq(&level.jl_projection) > 128u128.saturating_mul(statement.beta_sq) {
        return Err(HachiError::InvalidProof);
    }

    let setup = LabradorSetup::new(&level.config, r, max_len, comkey_seed);
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

    let mut combined_phi = vec![CyclotomicRing::<F, D>::zero(); max_len];
    for (i, phi_row) in phi_total.iter().enumerate() {
        let c = challenges[i];
        for (j, elem) in phi_row.iter().enumerate() {
            combined_phi[j] += c * *elem;
        }
    }
    let lhs = dot_product(&combined_phi, &z);
    let mut rhs = CyclotomicRing::<F, D>::zero();
    let mut idx = 0usize;
    for i in 0..r {
        for j in i..r {
            rhs += challenges[i] * challenges[j] * h_flat[idx];
            idx += 1;
        }
    }
    if lhs != rhs {
        return Err(HachiError::InvalidProof);
    }

    let mut diag_sum = CyclotomicRing::<F, D>::zero();
    for i in 0..r {
        let idx = diag_index(i, r);
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
    if level.tail {
        return Err(HachiError::InvalidProof);
    }
    let r = level.input_row_lengths.len();
    if r == 0 || level.input_row_chunks.len() != r {
        return Err(HachiError::InvalidProof);
    }
    if level.config.f == 0 || level.config.fu == 0 {
        return Err(HachiError::InvalidProof);
    }

    let max_len = level.input_row_lengths.iter().copied().max().unwrap_or(0);
    let expected_rows = level.config.f + 1;
    if witness.rows().len() != expected_rows {
        return Err(HachiError::InvalidProof);
    }
    for row in witness.rows().iter().take(level.config.f) {
        if row.len() != max_len {
            return Err(HachiError::InvalidProof);
        }
    }

    let t_hat_len = r * level.config.kappa * level.config.fu;
    let h_hat_len = r * (r + 1) / 2 * level.config.fu;
    let aux = &witness.rows()[level.config.f];
    if aux.len() != t_hat_len + h_hat_len {
        return Err(HachiError::InvalidProof);
    }
    let (t_hat, h_hat) = aux.split_at(t_hat_len);

    // Transcript: absorb level context, commitments, JL.
    absorb_labrador_level_context(
        transcript,
        &LabradorLevelTranscriptContext {
            level_index: 0,
            tail: level.tail,
            input_row_lengths: level.input_row_lengths.clone(),
            input_row_chunks: level.input_row_chunks.clone(),
            f: level.config.f,
            b: level.config.b,
            fu: level.config.fu,
            bu: level.config.bu,
            kappa: level.config.kappa,
            kappa1: level.config.kappa1,
        },
    )?;
    transcript.append_serde(labels::ABSORB_LABRADOR_U1, &level.u1);

    let total_len: usize = level.input_row_lengths.iter().sum();
    let jl_cols = total_len * D;
    let jl_matrix =
        LabradorJlMatrix::replay_nonce_search::<F, T>(transcript, level.jl_nonce, jl_cols)?;
    absorb_labrador_jl_projection(transcript, &level.jl_projection);

    let (phi_jl, b_jl) = aggregate_jl_constraints_verifier(
        &level.input_row_lengths,
        &level.jl_projection,
        &jl_matrix,
        &level.bb,
        transcript,
    )?;
    let (phi_stmt, b_stmt) = aggregate_statement_constraints(
        &statement.constraints,
        &level.input_row_lengths,
        transcript,
    )?;

    let mut phi_total = phi_stmt;
    add_phi_in_place(&mut phi_total, &phi_jl)?;
    let b_total = b_stmt + b_jl;

    transcript.append_serde(labels::ABSORB_LABRADOR_U2, &level.u2);

    let mut challenges = Vec::with_capacity(r);
    for _ in 0..r {
        challenges.push(challenge_ring_element_rejection_sampled(
            transcript,
            labels::CHALLENGE_LABRADOR_AMORTIZE,
        )?);
    }

    let z_parts: Vec<Vec<CyclotomicRing<F, D>>> = witness
        .rows()
        .iter()
        .take(level.config.f)
        .cloned()
        .collect();
    let z = recompose_from_parts(&z_parts, level.config.b as u32)?;

    let t_flat = recompose_flat(t_hat, level.config.fu, level.config.bu as u32)?;
    let h_flat = recompose_flat(h_hat, level.config.fu, level.config.bu as u32)?;
    if t_flat.len() != r * level.config.kappa {
        return Err(HachiError::InvalidProof);
    }
    if h_flat.len() != r * (r + 1) / 2 {
        return Err(HachiError::InvalidProof);
    }
    let mut t_by_row = Vec::with_capacity(r);
    for chunk in t_flat.chunks(level.config.kappa) {
        t_by_row.push(chunk.to_vec());
    }

    if !statement.u1.is_empty() && statement.u1 != level.u1 {
        return Err(HachiError::InvalidProof);
    }
    if !statement.u2.is_empty() && statement.u2 != level.u2 {
        return Err(HachiError::InvalidProof);
    }

    let setup = LabradorSetup::new(&level.config, r, max_len, comkey_seed);

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

    if projection_norm_sq(&level.jl_projection) > 128u128.saturating_mul(statement.beta_sq) {
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

    let mut combined_phi = vec![CyclotomicRing::<F, D>::zero(); max_len];
    for (i, phi_row) in phi_total.iter().enumerate() {
        let c = challenges[i];
        for (j, elem) in phi_row.iter().enumerate() {
            combined_phi[j] += c * *elem;
        }
    }
    let lhs = dot_product(&combined_phi, &z);
    let mut rhs = CyclotomicRing::<F, D>::zero();
    let mut idx = 0usize;
    for i in 0..r {
        for j in i..r {
            rhs += challenges[i] * challenges[j] * h_flat[idx];
            idx += 1;
        }
    }
    if lhs != rhs {
        return Err(HachiError::InvalidProof);
    }

    let mut diag_sum = CyclotomicRing::<F, D>::zero();
    for i in 0..r {
        let idx = diag_index(i, r);
        diag_sum += h_flat[idx];
    }
    if diag_sum - b_total != CyclotomicRing::<F, D>::zero() {
        return Err(HachiError::InvalidProof);
    }

    Ok(())
}

fn diag_index(i: usize, r: usize) -> usize {
    i * (2 * r - i + 1) / 2
}

fn projection_norm_sq(projection: &[i32; 256]) -> u128 {
    projection.iter().fold(0u128, |acc, &v| {
        let x = v as i128;
        let sq = x * x;
        acc.saturating_add(sq as u128)
    })
}

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

#[allow(clippy::too_many_arguments)]
fn build_next_constraints<
    F: FieldCore + CanonicalField + FieldSampling + FromSmallInt,
    const D: usize,
>(
    phi_total: &[Vec<CyclotomicRing<F, D>>],
    b_total: &CyclotomicRing<F, D>,
    challenges: &[CyclotomicRing<F, D>],
    row_lengths: &[usize],
    max_len: usize,
    config: &LabradorReductionConfig,
    u1: &[CyclotomicRing<F, D>],
    u2: &[CyclotomicRing<F, D>],
    setup: &LabradorSetup<F, D>,
) -> Result<Vec<LabradorConstraint<F, D>>, HachiError> {
    let r = row_lengths.len();
    if r == 0 || challenges.len() != r {
        return Err(HachiError::InvalidProof);
    }
    if config.f == 0 {
        return Err(HachiError::InvalidProof);
    }

    let pow_b: Vec<F> = (0..config.f)
        .map(|idx| pow2_field::<F>(config.b * idx))
        .collect();
    let pow_bu: Vec<F> = (0..config.fu)
        .map(|idx| pow2_field::<F>(config.bu * idx))
        .collect();

    let mut combined_phi = vec![CyclotomicRing::<F, D>::zero(); max_len];
    for (row_idx, row_phi) in phi_total.iter().enumerate() {
        let c = challenges[row_idx];
        for (j, elem) in row_phi.iter().enumerate() {
            combined_phi[j] += c * *elem;
        }
    }

    let mut constraints = Vec::new();
    let t_hat_len = r * config.kappa * config.fu;
    let h_len = r * (r + 1) / 2;
    let h_hat_len = h_len * config.fu;
    let aux_row = config.f;
    let aux_row_len = t_hat_len + h_hat_len;
    let num_rows = config.f + 1;

    if config.kappa1 > 0 {
        if u1.len() != config.kappa1 || u2.len() != config.kappa1 {
            return Err(HachiError::InvalidProof);
        }

        // B · t_hat = u1
        let mut aux_coeffs = vec![CyclotomicRing::<F, D>::zero(); config.kappa1 * aux_row_len];
        for (out_idx, b_row) in setup.b_mat.iter().enumerate() {
            let start = out_idx * aux_row_len;
            for (j, val) in b_row.iter().enumerate() {
                aux_coeffs[start + j] = *val;
            }
        }
        let mut coefficients = vec![vec![]; num_rows];
        coefficients[aux_row] = aux_coeffs;
        constraints.push(LabradorConstraint {
            coefficients,
            target: u1.to_vec(),
        });

        // D · h_hat = u2
        let mut aux_coeffs = vec![CyclotomicRing::<F, D>::zero(); config.kappa1 * aux_row_len];
        for (out_idx, d_row) in setup.d_mat.iter().enumerate() {
            let start = out_idx * aux_row_len + t_hat_len;
            for (j, val) in d_row.iter().enumerate() {
                aux_coeffs[start + j] = *val;
            }
        }
        let mut coefficients = vec![vec![]; num_rows];
        coefficients[aux_row] = aux_coeffs;
        constraints.push(LabradorConstraint {
            coefficients,
            target: u2.to_vec(),
        });
    }

    // A·z - c·t = 0
    let mut az_coefficients = vec![vec![]; num_rows];
    for part_idx in 0..config.f {
        let scale = pow_b[part_idx];
        let mut coeffs = Vec::with_capacity(config.kappa * max_len);
        for a_row in &setup.a_mat {
            for elem in a_row.iter() {
                coeffs.push(elem.scale(&scale));
            }
        }
        az_coefficients[part_idx] = coeffs;
    }

    let mut t_coeffs = vec![CyclotomicRing::<F, D>::zero(); config.kappa * t_hat_len];
    for (row_idx, challenge) in challenges.iter().enumerate() {
        for (part_idx, &scale) in pow_bu.iter().enumerate() {
            let scaled = challenge.scale(&scale);
            for k in 0..config.kappa {
                let idx = row_idx * config.kappa * config.fu + k * config.fu + part_idx;
                let slot = k * t_hat_len + idx;
                t_coeffs[slot] = -scaled;
            }
        }
    }
    let mut aux_az = vec![CyclotomicRing::<F, D>::zero(); config.kappa * aux_row_len];
    for k in 0..config.kappa {
        let src_start = k * t_hat_len;
        let dst_start = k * aux_row_len;
        aux_az[dst_start..dst_start + t_hat_len]
            .copy_from_slice(&t_coeffs[src_start..src_start + t_hat_len]);
    }
    az_coefficients[aux_row] = aux_az;
    constraints.push(LabradorConstraint {
        coefficients: az_coefficients,
        target: vec![CyclotomicRing::<F, D>::zero(); config.kappa],
    });

    // linear garbage constraint
    let mut lg_coefficients = vec![vec![]; num_rows];
    for part_idx in 0..config.f {
        let scale = pow_b[part_idx];
        let coeffs: Vec<CyclotomicRing<F, D>> =
            combined_phi.iter().map(|elem| elem.scale(&scale)).collect();
        lg_coefficients[part_idx] = coeffs;
    }
    let mut h_coeffs = vec![CyclotomicRing::<F, D>::zero(); h_hat_len];
    for i in 0..r {
        for j in i..r {
            let coeff = challenges[i] * challenges[j];
            let pair = pair_index(i, j, r);
            for (part_idx, &scale) in pow_bu.iter().enumerate() {
                let idx = pair * config.fu + part_idx;
                h_coeffs[idx] = -(coeff.scale(&scale));
            }
        }
    }
    let mut aux_lg = vec![CyclotomicRing::<F, D>::zero(); aux_row_len];
    aux_lg[t_hat_len..t_hat_len + h_hat_len].copy_from_slice(&h_coeffs);
    lg_coefficients[aux_row] = aux_lg;
    constraints.push(LabradorConstraint {
        coefficients: lg_coefficients,
        target: vec![CyclotomicRing::<F, D>::zero()],
    });

    // diagonal (norm) constraint
    let mut diag_coeffs = vec![CyclotomicRing::<F, D>::zero(); aux_row_len];
    for i in 0..r {
        let pair = pair_index(i, i, r);
        for (part_idx, &scale) in pow_bu.iter().enumerate() {
            let idx = pair * config.fu + part_idx;
            diag_coeffs[t_hat_len + idx] = constant_poly(scale);
        }
    }
    let mut diag_coefficients = vec![vec![]; num_rows];
    diag_coefficients[aux_row] = diag_coeffs;
    constraints.push(LabradorConstraint {
        coefficients: diag_coefficients,
        target: vec![*b_total],
    });

    Ok(constraints)
}

fn pow2_field<F: FieldCore + FromSmallInt>(exp: usize) -> F {
    let two = F::from_u64(2);
    let mut acc = F::one();
    for _ in 0..exp {
        acc = acc * two;
    }
    acc
}

fn constant_poly<F: FieldCore, const D: usize>(value: F) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(
        |i| {
            if i == 0 {
                value
            } else {
                F::zero()
            }
        },
    ))
}

fn pair_index(i: usize, j: usize, r: usize) -> usize {
    debug_assert!(i <= j && j < r);
    i * (2 * r - i + 1) / 2 + (j - i)
}

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

fn verify_constraints<F: FieldCore + CanonicalField + FromSmallInt, const D: usize>(
    constraints: &[LabradorConstraint<F, D>],
    witness: &LabradorWitness<F, D>,
) -> Result<(), HachiError> {
    for (idx, cnst) in constraints.iter().enumerate() {
        let outputs = cnst.target.len().max(1);
        let mut lhs = vec![CyclotomicRing::<F, D>::zero(); outputs];

        for (row_idx, coeffs) in cnst.coefficients.iter().enumerate() {
            if coeffs.is_empty() {
                continue;
            }
            if row_idx >= witness.rows().len() {
                return Err(HachiError::InvalidProof);
            }
            let row = &witness.rows()[row_idx];
            let row_len = coeffs.len() / outputs;
            for (out_idx, lhs_elem) in lhs.iter_mut().enumerate() {
                let coeff_start = out_idx * row_len;
                let coeff_slice = &coeffs[coeff_start..coeff_start + row_len];
                let mut inner = CyclotomicRing::<F, D>::zero();
                for (j, coeff) in coeff_slice.iter().enumerate() {
                    let w_elem = row
                        .get(j)
                        .copied()
                        .unwrap_or_else(CyclotomicRing::<F, D>::zero);
                    inner += *coeff * w_elem;
                }
                *lhs_elem += inner;
            }
        }

        for (out_idx, lhs_elem) in lhs.iter().enumerate() {
            let target = cnst
                .target
                .get(out_idx)
                .copied()
                .unwrap_or_else(CyclotomicRing::<F, D>::zero);
            if *lhs_elem != target {
                return Err(HachiError::InvalidInput(format!(
                    "Labrador constraint {idx} not satisfied"
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::Fp64;
    use crate::algebra::ring::CyclotomicRing;
    use crate::protocol::labrador::types::LabradorConstraint;
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
        let target = vec![CyclotomicRing::<F, D>::from_coefficients(
            std::array::from_fn(|i| if i == 0 { F::from_i64(3) } else { F::zero() }),
        )];
        let constraint = LabradorConstraint {
            coefficients: vec![coeff],
            target,
        };
        let statement = LabradorStatement {
            u1: Vec::new(),
            u2: Vec::new(),
            challenges: Vec::new(),
            constraints: vec![constraint],
            beta_sq: 1000,
            hash: [0u8; 16],
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
