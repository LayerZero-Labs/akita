//! Labrador amortization transitions (standard and tail levels).

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::commitment::utils::linear::decompose_rows_with_carry;
use crate::protocol::labrador::comkey::LabradorComKeySeed;
use crate::protocol::labrador::johnson_lindenstrauss::{
    collapse, project, zero_constant_term_for_proof, LabradorJlMatrix,
};
use crate::protocol::labrador::setup::LabradorSetup;
use crate::protocol::labrador::transcript::{
    absorb_labrador_jl_projection, absorb_labrador_level_context, LabradorLevelTranscriptContext,
};
use crate::protocol::labrador::types::{
    LabradorConstraint, LabradorLevelProof, LabradorReductionConfig, LabradorStatement,
    LabradorWitness,
};
use crate::protocol::labrador::utils::mat_vec_mul;
use crate::protocol::prg::MatrixPrgBackendChoice;
use crate::protocol::transcript::labels;
use crate::protocol::transcript::{challenge_ring_element_rejection_sampled, Transcript};
use crate::{CanonicalField, FieldCore, FieldSampling, FromSmallInt};

/// Output of one Labrador fold transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabradorFoldResult<F: FieldCore, const D: usize> {
    /// Next witness after amortization.
    pub next_witness: LabradorWitness<F, D>,
    /// Replay-complete level proof record.
    pub level_proof: LabradorLevelProof<F, D>,
    /// Reduced statement consumed by the next verifier step.
    pub statement: LabradorStatement<F, D>,
}

use crate::protocol::labrador::config::jl_lifts;

use crate::protocol::ajtai::ajtai_commit::AjtaiCommitmentScheme;
use crate::protocol::ajtai::coeff::{CoeffAjtai, CoeffAjtaiConfig};

/// Perform one Labrador fold level (standard or tail, determined by `config.tail`).
///
/// Follows the C Labrador protocol phases:
///   1. Commit: inner + outer Ajtai commitment → u1
///   2. Project: JL projection → p\[256\], nonce
///   3. LIFTS × (collapse + lift): build linear constraints from JL
///   4. Amortize: absorb into transcript, sample ring-element challenges,
///      fold z = sum_i c_i * s_i, decompose z → output witness
///
/// # Errors
///
/// Returns `HachiError::InvalidInput` if the witness is empty or `config.f` is zero.
/// Propagates errors from commitment, projection, or hashing.
#[allow(clippy::too_many_arguments)]
pub fn prove_level<F, T, const D: usize>(
    witness: &LabradorWitness<F, D>,
    statement: &LabradorStatement<F, D>,
    config: &LabradorReductionConfig,
    setup: &LabradorSetup<F, D>,
    comkey_seed: &LabradorComKeySeed,
    backend: MatrixPrgBackendChoice,
    level_index: usize,
    transcript: &mut T,
) -> Result<LabradorFoldResult<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + FromSmallInt,
    T: Transcript<F>,
{
    if witness.rows().is_empty() {
        return Err(HachiError::InvalidInput(
            "cannot fold empty Labrador witness".to_string(),
        ));
    }
    if config.f == 0 {
        return Err(HachiError::InvalidInput(
            "Labrador fold requires f > 0".to_string(),
        ));
    }

    // Phase 1: Inner commitments (t_i) and outer commitment u1.
    let coeff_config = CoeffAjtaiConfig {
        inner_rows: config.kappa,
        outer_rows: config.kappa1,
        num_digits: config.fu,
        decompose_modulus: config.bu as u32,
        comkey_seed: *comkey_seed,
    };

    let (t_hat, u1) =
        CoeffAjtai::two_tier_commit(&setup.a_mat, &setup.b_mat, witness.rows(), &coeff_config)?;

    let r = witness.rows().len();
    let row_lengths: Vec<usize> = witness.rows().iter().map(|row| row.len()).collect();
    let max_len = row_lengths.iter().copied().max().unwrap_or(0);

    // Absorb level context and u1 before deriving JL seed.
    absorb_labrador_level_context(
        transcript,
        &LabradorLevelTranscriptContext {
            level_index,
            tail: config.tail,
            input_row_lengths: row_lengths.clone(),
            input_row_chunks: vec![1usize; r],
            f: config.f,
            b: config.b,
            fu: config.fu,
            bu: config.bu,
            kappa: config.kappa,
            kappa1: config.kappa1,
            prg_backend_id: backend as u8,
        },
    )?;
    transcript.append_serde(labels::ABSORB_LABRADOR_U1, &u1);

    // Phase 2: JL Projection — nonce + matrix squeezed from transcript.
    let (jl_projection, jl_nonce, jl_matrix) = project(witness, transcript)?;

    absorb_labrador_jl_projection(transcript, &jl_projection);

    // Phase 3: JL lift constraints and aggregation.
    let (phi_jl, b_jl, bb) =
        aggregate_jl_constraints_prover(witness, &jl_projection, &jl_matrix, transcript)?;

    // Aggregate statement constraints (after JL lifts).
    let (phi_stmt, b_stmt) =
        aggregate_statement_constraints(&statement.constraints, &row_lengths, transcript)?;

    let mut phi_total = phi_stmt;
    add_phi_in_place(&mut phi_total, &phi_jl)?;
    let b_total = b_stmt + b_jl;

    // Linear garbage h_ij from aggregated phi and witness.
    let h = compute_linear_garbage(&phi_total, witness)?;
    let h_hat = decompose_rows_with_carry(&h, config.fu, config.bu as u32);

    let u2 = if !setup.d_mat.is_empty() {
        mat_vec_mul(&setup.d_mat, &h_hat)
    } else {
        h_hat.clone()
    };

    // Absorb u2 before amortization challenges.
    transcript.append_serde(labels::ABSORB_LABRADOR_U2, &u2);

    // Phase 4: Amortize — sample r challenge ring-elements from transcript, fold.
    let mut challenges = Vec::with_capacity(r);
    for _ in 0..r {
        challenges.push(challenge_ring_element_rejection_sampled(
            transcript,
            labels::CHALLENGE_LABRADOR_AMORTIZE,
        )?);
    }

    let z = amortize_witness(witness, &challenges, max_len);
    let decomposed_z = decompose_rows_with_carry(&z, config.f, config.b as u32);
    let z_rows = split_decomposed_rows(&decomposed_z, config.f, z.len())?;

    let mut output_rows: Vec<Vec<CyclotomicRing<F, D>>> = z_rows;

    if !config.tail {
        let mut aux = Vec::with_capacity(t_hat.len() + h_hat.len());
        aux.extend_from_slice(&t_hat);
        aux.extend_from_slice(&h_hat);
        output_rows.push(aux);
    }

    let next_witness = LabradorWitness::new_unchecked(output_rows);
    let out_norm_sq: u128 = next_witness.norm();

    let next_constraints = if config.tail {
        Vec::new()
    } else {
        build_next_constraints(
            &phi_total,
            &b_total,
            &challenges,
            &row_lengths,
            max_len,
            config,
            &u1,
            &u2,
            setup,
        )?
    };

    let level_proof = LabradorLevelProof {
        tail: config.tail,
        input_row_lengths: row_lengths,
        input_row_chunks: vec![1usize; r],
        config: *config,
        u1: u1.clone(),
        u2: u2.clone(),
        jl_projection,
        jl_nonce,
        bb,
        norm_sq: out_norm_sq,
    };

    // NOTE: Recursive statement update is not implemented yet.
    let statement = LabradorStatement {
        u1,
        u2,
        challenges: challenges.clone(),
        constraints: next_constraints,
        beta_sq: out_norm_sq,
        hash: [0u8; 16],
    };

    Ok(LabradorFoldResult {
        next_witness,
        level_proof,
        statement,
    })
}

fn split_decomposed_rows<F: FieldCore, const D: usize>(
    flat: &[CyclotomicRing<F, D>],
    parts: usize,
    len: usize,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, HachiError> {
    if parts == 0 {
        return Err(HachiError::InvalidInput(
            "cannot split decomposition with zero parts".to_string(),
        ));
    }
    if flat.len() != len * parts {
        return Err(HachiError::InvalidInput(format!(
            "decomposition length mismatch: got {}, expected {}",
            flat.len(),
            len * parts
        )));
    }
    let mut rows = vec![Vec::with_capacity(len); parts];
    for idx in 0..len {
        for part in 0..parts {
            rows[part].push(flat[idx * parts + part]);
        }
    }
    Ok(rows)
}

fn add_phi_in_place<F: FieldCore, const D: usize>(
    acc: &mut [Vec<CyclotomicRing<F, D>>],
    other: &[Vec<CyclotomicRing<F, D>>],
) -> Result<(), HachiError> {
    if acc.len() != other.len() {
        return Err(HachiError::InvalidInput(
            "phi row count mismatch".to_string(),
        ));
    }
    for (row_acc, row_other) in acc.iter_mut().zip(other.iter()) {
        if row_acc.len() != row_other.len() {
            return Err(HachiError::InvalidInput(
                "phi row length mismatch".to_string(),
            ));
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

#[allow(clippy::type_complexity)]
fn aggregate_statement_constraints<F, T, const D: usize>(
    constraints: &[LabradorConstraint<F, D>],
    row_lengths: &[usize],
    transcript: &mut T,
) -> Result<(Vec<Vec<CyclotomicRing<F, D>>>, CyclotomicRing<F, D>), HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
    T: Transcript<F>,
{
    let mut phi_total: Vec<Vec<CyclotomicRing<F, D>>> = row_lengths
        .iter()
        .map(|&len| vec![CyclotomicRing::zero(); len])
        .collect();
    let mut b_total = CyclotomicRing::<F, D>::zero();

    if constraints.is_empty() {
        return Ok((phi_total, b_total));
    }

    for cnst in constraints {
        let outputs = cnst.target.len().max(1);
        for out_idx in 0..outputs {
            let alpha = challenge_ring_element_rejection_sampled(
                transcript,
                labels::CHALLENGE_LABRADOR_AGGREGATION,
            )?;
            let target = cnst
                .target
                .get(out_idx)
                .copied()
                .unwrap_or_else(CyclotomicRing::<F, D>::zero);
            b_total += alpha * target;

            for (row_idx, coeffs) in cnst.coefficients.iter().enumerate() {
                if coeffs.is_empty() {
                    continue;
                }
                if row_idx >= phi_total.len() {
                    return Err(HachiError::InvalidInput(
                        "constraint row index out of bounds".to_string(),
                    ));
                }
                let row_len = coeffs.len() / outputs;
                let coeff_start = out_idx * row_len;
                let coeff_slice = &coeffs[coeff_start..coeff_start + row_len];

                for (j, coeff) in coeff_slice.iter().enumerate() {
                    phi_total[row_idx][j] += alpha * *coeff;
                }
            }
        }
    }

    Ok((phi_total, b_total))
}

fn flatten_witness<F: FieldCore, const D: usize>(
    witness: &LabradorWitness<F, D>,
) -> (Vec<CyclotomicRing<F, D>>, Vec<(usize, usize)>) {
    let mut flat = Vec::new();
    let mut ranges = Vec::with_capacity(witness.rows().len());
    let mut cursor = 0usize;
    for row in witness.rows() {
        let start = cursor;
        flat.extend(row.iter().copied());
        cursor += row.len();
        ranges.push((start, cursor));
    }
    (flat, ranges)
}

fn sample_jl_collapse_challenge<F, T>(transcript: &mut T) -> [i64; 256]
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let q = (-F::one()).to_canonical_u128() + 1;
    let half_q = q / 2;
    std::array::from_fn(|_| {
        let s = transcript.challenge_scalar(labels::CHALLENGE_LABRADOR_JL_COLLAPSE);
        let c = s.to_canonical_u128();
        if c > half_q {
            -((q - c) as i64)
        } else {
            c as i64
        }
    })
}

fn jl_collapse_phi_from_weights<F: FieldCore + CanonicalField + FromSmallInt, const D: usize>(
    matrix: &LabradorJlMatrix,
    omega: &[i64; 256],
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
    if matrix.cols() % D != 0 {
        return Err(HachiError::InvalidInput(
            "JL matrix cols not divisible by ring degree".to_string(),
        ));
    }
    let cols = matrix.cols();
    let mut weights = vec![0i64; cols];
    for (row_idx, row) in matrix.signs.iter().enumerate() {
        let alpha = omega[row_idx];
        for (col_idx, &sign) in row.iter().enumerate() {
            weights[col_idx] += alpha * (sign as i64);
        }
    }

    let ring_elems = cols / D;
    let phi: Vec<CyclotomicRing<F, D>> = cfg_into_iter!(0..ring_elems)
        .map(|idx| {
            let coeffs = std::array::from_fn(|k| {
                let w = weights[idx * D + k];
                F::from_i64(w)
            });
            CyclotomicRing::from_coefficients(coeffs).sigma_m1()
        })
        .collect();
    Ok(phi)
}

#[allow(clippy::type_complexity)]
fn aggregate_jl_constraints_prover<F, T, const D: usize>(
    witness: &LabradorWitness<F, D>,
    jl_projection: &[i32; 256],
    matrix: &LabradorJlMatrix,
    transcript: &mut T,
) -> Result<
    (
        Vec<Vec<CyclotomicRing<F, D>>>,
        CyclotomicRing<F, D>,
        Vec<CyclotomicRing<F, D>>,
    ),
    HachiError,
>
where
    F: FieldCore + CanonicalField + FromSmallInt,
    T: Transcript<F>,
{
    let (flat, ranges) = flatten_witness(witness);
    let mut phi_total: Vec<Vec<CyclotomicRing<F, D>>> = witness
        .rows()
        .iter()
        .map(|row| vec![CyclotomicRing::zero(); row.len()])
        .collect();
    let mut b_total = CyclotomicRing::<F, D>::zero();
    let jl_lifts = jl_lifts::<F>();
    let mut bb = Vec::with_capacity(jl_lifts);

    for _ in 0..jl_lifts {
        let omega = sample_jl_collapse_challenge::<F, T>(transcript);
        let phi_flat = jl_collapse_phi_from_weights::<F, D>(matrix, &omega)?;
        let b_full = dot_product(&phi_flat, &flat);
        let target = collapse(jl_projection, &omega);
        let expected_c0 = F::from_i64(target);
        if b_full.coefficients()[0] != expected_c0 {
            return Err(HachiError::InvalidProof);
        }
        let (b_tx, _c0) = zero_constant_term_for_proof(b_full);
        bb.push(b_tx);
        transcript.append_serde(labels::ABSORB_LABRADOR_BB, &b_tx);

        let beta = challenge_ring_element_rejection_sampled(
            transcript,
            labels::CHALLENGE_LABRADOR_AGGREGATION,
        )?;
        b_total += beta * b_full;

        for (row_idx, (start, end)) in ranges.iter().enumerate() {
            let row = &phi_flat[*start..*end];
            for (j, elem) in row.iter().enumerate() {
                phi_total[row_idx][j] += beta * *elem;
            }
        }
    }

    Ok((phi_total, b_total, bb))
}

fn compute_linear_garbage<F: FieldCore + CanonicalField + FromSmallInt, const D: usize>(
    phi: &[Vec<CyclotomicRing<F, D>>],
    witness: &LabradorWitness<F, D>,
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
    let r = witness.rows().len();
    if phi.len() != r {
        return Err(HachiError::InvalidInput(
            "phi row count mismatch".to_string(),
        ));
    }
    for (phi_row, witness_row) in phi.iter().zip(witness.rows().iter()) {
        if phi_row.len() != witness_row.len() {
            return Err(HachiError::InvalidInput(
                "phi row length mismatch".to_string(),
            ));
        }
    }
    let pairs: Vec<(usize, usize)> = (0..r).flat_map(|i| (i..r).map(move |j| (i, j))).collect();
    let out: Vec<CyclotomicRing<F, D>> = cfg_iter!(pairs)
        .map(|&(i, j)| {
            if i == j {
                dot_product(&phi[i], &witness.rows()[i])
            } else {
                let lhs = dot_product(&phi[i], &witness.rows()[j]);
                let rhs = dot_product(&phi[j], &witness.rows()[i]);
                lhs + rhs
            }
        })
        .collect();
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
        return Err(HachiError::InvalidInput(
            "challenge row count mismatch".to_string(),
        ));
    }
    if config.f == 0 {
        return Err(HachiError::InvalidInput(
            "cannot build next constraints with f=0".to_string(),
        ));
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
            return Err(HachiError::InvalidInput(
                "u1/u2 length mismatch for next statement".to_string(),
            ));
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

    // A·z - c·t = 0  (inner commitment relation)
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

/// Compute z = sum_i c_i * s_i (all-row linear combination).
fn amortize_witness<F: FieldCore + CanonicalField, const D: usize>(
    witness: &LabradorWitness<F, D>,
    challenges: &[CyclotomicRing<F, D>],
    max_len: usize,
) -> Vec<CyclotomicRing<F, D>> {
    cfg_into_iter!(0..max_len)
        .map(|j| {
            let mut acc = CyclotomicRing::<F, D>::zero();
            for (row, challenge) in witness.rows().iter().zip(challenges.iter()) {
                if let Some(elem) = row.get(j) {
                    acc += *challenge * *elem;
                }
            }
            acc
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::Fp64;
    use crate::protocol::labrador::types::LabradorReductionConfig;
    use crate::protocol::labrador::{verify, LabradorProof};
    use crate::protocol::transcript::labels::DOMAIN_LABRADOR_PROTOCOL;
    use crate::protocol::transcript::Blake2bTranscript;
    use crate::FromSmallInt;

    type F = Fp64<4294967197>;
    const D: usize = 64;

    fn sample_witness() -> LabradorWitness<F, D> {
        let row = |len: usize| -> Vec<CyclotomicRing<F, D>> {
            (0..len)
                .map(|i| {
                    CyclotomicRing::from_coefficients(std::array::from_fn(|j| {
                        F::from_i64(((i + j) as i64 % 5) - 2)
                    }))
                })
                .collect()
        };
        LabradorWitness::new(vec![row(4), row(4), row(4)])
    }

    #[test]
    fn standard_fold_produces_decomposed_output() {
        let witness = sample_witness();
        let statement = LabradorStatement {
            u1: Vec::new(),
            u2: Vec::new(),
            challenges: Vec::new(),
            constraints: Vec::new(),
            beta_sq: 1 << 20,
            hash: [0u8; 16],
        };
        let cfg = LabradorReductionConfig {
            f: 1,
            b: 8,
            fu: 2,
            bu: 10,
            kappa: 3,
            kappa1: 2,
            tail: false,
        };
        let seed = [1u8; 32];
        let r = witness.rows().len();
        let max_len = witness
            .rows()
            .iter()
            .map(|row| row.len())
            .max()
            .unwrap_or(0);
        let setup = LabradorSetup::new(&cfg, r, max_len, &seed, MatrixPrgBackendChoice::Shake256);
        let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        let out = prove_level(
            &witness,
            &statement,
            &cfg,
            &setup,
            &seed,
            MatrixPrgBackendChoice::Shake256,
            0,
            &mut transcript,
        )
        .unwrap();
        assert!(
            !out.next_witness.rows().is_empty(),
            "fold must produce output witness"
        );
        assert_eq!(out.next_witness.rows().len(), cfg.f + 1);
        assert!(!out.level_proof.u2.is_empty());
    }

    #[test]
    fn tail_fold_produces_decomposed_output() {
        let witness = sample_witness();
        let statement = LabradorStatement {
            u1: Vec::new(),
            u2: Vec::new(),
            challenges: Vec::new(),
            constraints: Vec::new(),
            beta_sq: 1 << 20,
            hash: [0u8; 16],
        };
        let cfg = LabradorReductionConfig {
            f: 1,
            b: 8,
            fu: 1,
            bu: 32,
            kappa: 2,
            kappa1: 0,
            tail: true,
        };
        let seed = [3u8; 32];
        let r = witness.rows().len();
        let max_len = witness
            .rows()
            .iter()
            .map(|row| row.len())
            .max()
            .unwrap_or(0);
        let setup = LabradorSetup::new(&cfg, r, max_len, &seed, MatrixPrgBackendChoice::Shake256);
        let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        let out = prove_level(
            &witness,
            &statement,
            &cfg,
            &setup,
            &seed,
            MatrixPrgBackendChoice::Shake256,
            1,
            &mut transcript,
        )
        .unwrap();
        assert!(
            !out.next_witness.rows().is_empty(),
            "tail fold must produce output"
        );
        assert_eq!(out.next_witness.rows().len(), cfg.f);
        assert!(out.level_proof.tail);
    }

    #[test]
    fn amortize_is_linear_combination() {
        let witness = sample_witness();
        let one = CyclotomicRing::<F, D>::one();
        let challenges = vec![one; witness.rows().len()];
        let max_len = witness.rows().iter().map(|r| r.len()).max().unwrap();
        let z = amortize_witness(&witness, &challenges, max_len);

        for (j, z_elem) in z.iter().enumerate().take(max_len) {
            let expected = witness
                .rows()
                .iter()
                .map(|row| {
                    row.get(j)
                        .copied()
                        .unwrap_or_else(CyclotomicRing::<F, D>::zero)
                })
                .fold(CyclotomicRing::<F, D>::zero(), |a, b| a + b);
            assert_eq!(*z_elem, expected);
        }
    }

    #[test]
    fn standard_fold_roundtrip_verifies() {
        let mk_ring = |c: i64| {
            CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|i| {
                if i == 0 {
                    F::from_i64(c)
                } else {
                    F::zero()
                }
            }))
        };
        let witness = LabradorWitness::new(vec![
            vec![mk_ring(1), mk_ring(2)],
            vec![mk_ring(3), mk_ring(-1)],
        ]);
        let target = witness.rows()[0][0] + witness.rows()[1][1];
        let statement = LabradorStatement {
            u1: Vec::new(),
            u2: Vec::new(),
            challenges: Vec::new(),
            constraints: vec![LabradorConstraint {
                coefficients: vec![vec![mk_ring(1), mk_ring(0)], vec![mk_ring(0), mk_ring(1)]],
                target: vec![target],
            }],
            beta_sq: 1 << 40,
            hash: [0u8; 16],
        };
        let cfg = LabradorReductionConfig {
            f: 4,
            b: 8,
            fu: 4,
            bu: 8,
            kappa: 2,
            kappa1: 2,
            tail: false,
        };
        let comkey_seed = [9u8; 32];
        let r = witness.rows().len();
        let max_len = witness
            .rows()
            .iter()
            .map(|row| row.len())
            .max()
            .unwrap_or(0);
        let setup = LabradorSetup::new(
            &cfg,
            r,
            max_len,
            &comkey_seed,
            MatrixPrgBackendChoice::Shake256,
        );
        let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        let fold = prove_level(
            &witness,
            &statement,
            &cfg,
            &setup,
            &comkey_seed,
            MatrixPrgBackendChoice::Shake256,
            0,
            &mut transcript,
        )
        .unwrap();

        let proof = LabradorProof {
            levels: vec![fold.level_proof],
            final_opening_witness: fold.next_witness,
        };
        let mut verify_transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        verify(
            &statement,
            &proof,
            &comkey_seed,
            MatrixPrgBackendChoice::Shake256,
            &mut verify_transcript,
        )
        .unwrap();

        let base_proof = LabradorProof {
            levels: Vec::new(),
            final_opening_witness: proof.final_opening_witness.clone(),
        };
        let mut base_transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        verify(
            &fold.statement,
            &base_proof,
            &comkey_seed,
            MatrixPrgBackendChoice::Shake256,
            &mut base_transcript,
        )
        .unwrap();
    }

    #[test]
    fn two_level_fold_roundtrip_verifies() {
        let mk_ring = |c: i64| {
            CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|i| {
                if i == 0 {
                    F::from_i64(c)
                } else {
                    F::zero()
                }
            }))
        };
        let witness = LabradorWitness::new(vec![
            vec![mk_ring(1), mk_ring(2)],
            vec![mk_ring(3), mk_ring(-1)],
        ]);
        let target = witness.rows()[0][0] + witness.rows()[1][1];
        let statement = LabradorStatement {
            u1: Vec::new(),
            u2: Vec::new(),
            challenges: Vec::new(),
            constraints: vec![LabradorConstraint {
                coefficients: vec![vec![mk_ring(1), mk_ring(0)], vec![mk_ring(0), mk_ring(1)]],
                target: vec![target],
            }],
            beta_sq: 1 << 40,
            hash: [0u8; 16],
        };
        let cfg = LabradorReductionConfig {
            f: 4,
            b: 8,
            fu: 4,
            bu: 8,
            kappa: 2,
            kappa1: 2,
            tail: false,
        };
        let comkey_seed = [9u8; 32];
        let r1 = witness.rows().len();
        let max_len1 = witness
            .rows()
            .iter()
            .map(|row| row.len())
            .max()
            .unwrap_or(0);
        let setup1 = LabradorSetup::new(
            &cfg,
            r1,
            max_len1,
            &comkey_seed,
            MatrixPrgBackendChoice::Shake256,
        );
        let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        let fold1 = prove_level(
            &witness,
            &statement,
            &cfg,
            &setup1,
            &comkey_seed,
            MatrixPrgBackendChoice::Shake256,
            0,
            &mut transcript,
        )
        .unwrap();
        let r2 = fold1.next_witness.rows().len();
        let max_len2 = fold1
            .next_witness
            .rows()
            .iter()
            .map(|row| row.len())
            .max()
            .unwrap_or(0);
        let setup2 = LabradorSetup::new(
            &cfg,
            r2,
            max_len2,
            &comkey_seed,
            MatrixPrgBackendChoice::Shake256,
        );
        let fold2 = prove_level(
            &fold1.next_witness,
            &fold1.statement,
            &cfg,
            &setup2,
            &comkey_seed,
            MatrixPrgBackendChoice::Shake256,
            1,
            &mut transcript,
        )
        .unwrap();

        let r = fold2.level_proof.input_row_lengths.len();
        let challenges = &fold2.statement.challenges;
        let aux_row = &fold2.next_witness.rows()[cfg.f];
        let t_hat_len = r * cfg.kappa * cfg.fu;
        let t_hat = &aux_row[..t_hat_len];
        let mut t_flat = Vec::with_capacity(r * cfg.kappa);
        for chunk in t_hat.chunks(cfg.fu) {
            t_flat.push(CyclotomicRing::gadget_recompose_pow2(chunk, cfg.bu as u32));
        }
        let z_parts: Vec<Vec<CyclotomicRing<F, D>>> = fold2.next_witness.rows()[..cfg.f].to_vec();
        let mut z = Vec::with_capacity(z_parts[0].len());
        for idx in 0..z_parts[0].len() {
            let mut slice = Vec::with_capacity(cfg.f);
            for part in &z_parts {
                slice.push(part[idx]);
            }
            z.push(CyclotomicRing::gadget_recompose_pow2(&slice, cfg.b as u32));
        }
        let az = mat_vec_mul(&setup2.a_mat, &z);
        let mut rhs = vec![CyclotomicRing::<F, D>::zero(); cfg.kappa];
        for (row_idx, t_row) in t_flat.chunks(cfg.kappa).enumerate() {
            let c = challenges[row_idx];
            for k in 0..cfg.kappa {
                rhs[k] += c * t_row[k];
            }
        }
        assert_eq!(az, rhs);

        let proof = LabradorProof {
            levels: vec![fold1.level_proof, fold2.level_proof],
            final_opening_witness: fold2.next_witness,
        };
        let mut verify_transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        verify(
            &statement,
            &proof,
            &comkey_seed,
            MatrixPrgBackendChoice::Shake256,
            &mut verify_transcript,
        )
        .unwrap();
    }
}
