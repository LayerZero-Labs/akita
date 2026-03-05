//! Greyhound prover-side evaluation reduction.
//!
//! Produces a 4-row witness matching the C reference structure (adapted for
//! multilinear evaluation) and scalar Labrador constraints via `greyhound_reduce`.

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::primitives::poly::multilinear_lagrange_basis;
use crate::protocol::commitment::utils::linear::decompose_rows_with_carry;
use crate::protocol::greyhound::reduce::greyhound_reduce;
use crate::protocol::greyhound::types::GreyhoundEvalProof;
use crate::protocol::labrador::comkey::{derive_extendable_comkey_matrix, LabradorComKeySeed};
use crate::protocol::labrador::select_config;
use crate::protocol::labrador::transcript::{
    absorb_greyhound_eval_claim, absorb_greyhound_eval_context, absorb_greyhound_u2,
    sample_greyhound_fold_challenge, GreyhoundEvalTranscriptContext,
};
use crate::protocol::labrador::types::{LabradorStatement, LabradorWitness};
use crate::protocol::labrador::utils::mat_vec_mul;
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore, FieldSampling};

/// Output of `greyhound_eval`: proof, witness, and reduced statement.
pub type GreyhoundEvalResult<F, const D: usize> = (
    GreyhoundEvalProof<F, D>,
    LabradorWitness<F, D>,
    LabradorStatement<F, D>,
);

/// Build Greyhound evaluation proof and reduced Labrador witness/statement.
///
/// The witness has 4 rows matching the C reference:
///   row0: z_low  (m*f elements) — low part of decomposed amortized z
///   row1: z_high (m*f elements) — high part (z = z_low + 2^bu * z_high)
///   row2: t_hat  (kappa*fu*n elements) — decomposed inner commitments
///   row3: v_hat  (fu*n elements) — decomposed partial evaluations
///
/// # Errors
///
/// Returns an error if reshaping, config selection, or commitment fails.
pub fn greyhound_eval<F, T, const D: usize>(
    witness_coeffs: &[F],
    eval_point: &[F],
    eval_value: F,
    w_commitment_u1: &[CyclotomicRing<F, D>],
    comkey_seed: &LabradorComKeySeed,
    transcript: &mut T,
) -> Result<GreyhoundEvalResult<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
{
    let ring_witness = pack_coefficients_to_ring::<F, D>(witness_coeffs);
    if ring_witness.is_empty() {
        return Err(HachiError::InvalidInput(
            "greyhound_eval requires non-empty witness".to_string(),
        ));
    }
    let (m_rows, n_cols, inner_vars) = choose_dimensions(ring_witness.len());
    if eval_point.len() < inner_vars {
        return Err(HachiError::InvalidPointDimension {
            expected: inner_vars,
            actual: eval_point.len(),
        });
    }

    let inner_point = &eval_point[eval_point.len() - inner_vars..];
    let mut inner_basis = vec![F::zero(); 1usize << inner_vars];
    multilinear_lagrange_basis(&mut inner_basis, inner_point);

    let matrix = reshape_columns(&ring_witness, m_rows, n_cols);
    let partial_evals = partial_evaluate_columns(&matrix, &inner_basis);

    // Select Labrador config from columns (pre-amortization dimensions).
    let column_witness = columns_to_witness(&matrix);
    let cfg = select_config(&column_witness)?;

    // Decompose partial evaluations v → v_hat (group 3).
    let v_hat = decompose_rows_with_carry(&partial_evals, cfg.fu, cfg.bu as u32);

    // Commit v_hat → u2 (outer commitment to evaluation witness).
    let u2 = if cfg.kappa1 > 0 {
        let b_eval = derive_extendable_comkey_matrix::<F, D>(
            cfg.kappa1,
            v_hat.len(),
            comkey_seed,
            b"greyhound/comkey/B_eval",
        );
        mat_vec_mul(&b_eval, &v_hat)
    } else {
        v_hat.clone()
    };

    // Transcript: absorb context, claim, u2.
    absorb_greyhound_eval_context(
        transcript,
        &GreyhoundEvalTranscriptContext {
            m_rows,
            n_cols,
            inner_vars,
            eval_point_len: eval_point.len(),
        },
    )?;
    absorb_greyhound_eval_claim(transcript, eval_point, &eval_value);
    absorb_greyhound_u2(transcript, &u2);

    // Sample n_cols fold challenges from transcript.
    let fold_challenges: Vec<F> = (0..n_cols)
        .map(|_| sample_greyhound_fold_challenge(transcript))
        .collect();

    // Amortize columns: z[j] = sum_col c_col * column[col][j].
    let mut z = vec![CyclotomicRing::<F, D>::zero(); m_rows];
    for (col_idx, column) in matrix.iter().enumerate() {
        let c = fold_challenges[col_idx];
        for (j, elem) in column.iter().enumerate() {
            z[j] += elem.scale(&c);
        }
    }

    // Decompose z → groups 0 (z_low) and 1 (z_high).
    // First: decompose with (f, b), then split each part into low/high with bu.
    let z_first = decompose_rows_with_carry(&z, cfg.f, cfg.b as u32);
    let z_uniform = decompose_rows_with_carry(&z_first, 2, cfg.bu as u32);
    let mut z_low = Vec::with_capacity(z_first.len());
    let mut z_high = Vec::with_capacity(z_first.len());
    for i in 0..z_first.len() {
        z_low.push(z_uniform[2 * i]);
        z_high.push(z_uniform[2 * i + 1]);
    }

    // Compute inner commitments t_j = A * column_j, decompose → t_hat (group 2).
    let mut t_hat_flat = Vec::new();
    for column in &matrix {
        let a = derive_extendable_comkey_matrix::<F, D>(
            cfg.kappa,
            column.len(),
            comkey_seed,
            b"labrador/comkey/A",
        );
        let t_j = mat_vec_mul(&a, column);
        t_hat_flat.extend(decompose_rows_with_carry(&t_j, cfg.fu, cfg.bu as u32));
    }

    let greyhound_witness = LabradorWitness::new_unchecked(vec![z_low, z_high, t_hat_flat, v_hat]);

    let proof = GreyhoundEvalProof {
        u2: u2.clone(),
        m_rows,
        n_cols,
        inner_vars,
        config: cfg,
    };

    let mut statement = greyhound_reduce(
        &proof,
        w_commitment_u1,
        eval_point,
        eval_value,
        &fold_challenges,
        comkey_seed,
    )?;
    statement.beta_sq = greyhound_witness.norm();

    Ok((proof, greyhound_witness, statement))
}

fn pack_coefficients_to_ring<F: FieldCore, const D: usize>(
    coeffs: &[F],
) -> Vec<CyclotomicRing<F, D>> {
    if coeffs.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(coeffs.len().div_ceil(D));
    for chunk in coeffs.chunks(D) {
        let ring = CyclotomicRing::from_coefficients(std::array::from_fn(|i| {
            chunk.get(i).copied().unwrap_or_else(F::zero)
        }));
        out.push(ring);
    }
    out
}

fn choose_dimensions(num_ring_elements: usize) -> (usize, usize, usize) {
    let n = num_ring_elements.max(1).next_power_of_two();
    let k_total = n.trailing_zeros() as usize;
    let inner_vars = k_total / 2;
    let outer_vars = k_total - inner_vars;
    (1usize << inner_vars, 1usize << outer_vars, inner_vars)
}

fn reshape_columns<F: FieldCore, const D: usize>(
    ring_witness: &[CyclotomicRing<F, D>],
    m_rows: usize,
    n_cols: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    (0..n_cols)
        .map(|col| {
            (0..m_rows)
                .map(|row| {
                    let idx = col * m_rows + row;
                    ring_witness
                        .get(idx)
                        .copied()
                        .unwrap_or_else(CyclotomicRing::<F, D>::zero)
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn partial_evaluate_columns<F: FieldCore, const D: usize>(
    columns: &[Vec<CyclotomicRing<F, D>>],
    inner_basis: &[F],
) -> Vec<CyclotomicRing<F, D>> {
    columns
        .iter()
        .map(|col| {
            let mut acc = CyclotomicRing::<F, D>::zero();
            for (elem, &basis) in col.iter().zip(inner_basis.iter()) {
                acc += elem.scale(&basis);
            }
            acc
        })
        .collect()
}

/// Build a temporary witness from columns for config selection.
fn columns_to_witness<F: FieldCore, const D: usize>(
    matrix: &[Vec<CyclotomicRing<F, D>>],
) -> LabradorWitness<F, D> {
    LabradorWitness::new_unchecked(matrix.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::Fp64;
    use crate::protocol::greyhound::greyhound_verify_stage1;
    use crate::protocol::labrador::{prove_level, prove_with_config, verify, LabradorProof};
    use crate::protocol::transcript::labels::DOMAIN_GREYHOUND_EVAL;
    use crate::protocol::transcript::labels::DOMAIN_LABRADOR_PROTOCOL;
    use crate::protocol::transcript::Blake2bTranscript;
    use crate::FromSmallInt;

    type F = Fp64<4294967197>;
    const D: usize = 64;

    #[test]
    fn eval_outputs_four_row_witness_and_scalar_constraints() {
        let coeffs: Vec<F> = (0..256).map(|i| F::from_i64((i as i64 % 13) - 6)).collect();
        let eval_point: Vec<F> = (0..8).map(|i| F::from_i64(i as i64 + 1)).collect();
        let eval_value = F::from_i64(9);
        let u1 = vec![CyclotomicRing::<F, D>::one(), CyclotomicRing::<F, D>::one()];
        let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_GREYHOUND_EVAL);
        let (proof, witness, statement) = greyhound_eval(
            &coeffs,
            &eval_point,
            eval_value,
            &u1,
            &[8u8; 32],
            &mut transcript,
        )
        .unwrap();
        assert_eq!(proof.u2, statement.u2);
        assert_eq!(witness.rows().len(), 4);
        assert_eq!(
            statement.constraints.len(),
            statement.u1.len() + statement.u2.len() + proof.config.kappa + 2
        );
    }

    #[test]
    fn stage1_constraints_verify_with_full_witness() {
        let comkey_seed = [42u8; 32];

        let ring_elems = 16;
        let coeffs = vec![F::zero(); ring_elems * D];

        let ring_witness = pack_coefficients_to_ring::<F, D>(&coeffs);
        let (m_rows, n_cols, inner_vars) = choose_dimensions(ring_witness.len());
        let outer_vars = n_cols.trailing_zeros() as usize;
        let eval_point: Vec<F> = (0..(inner_vars + outer_vars))
            .map(|i| F::from_i64(i as i64 + 2))
            .collect();

        let inner_point = &eval_point[eval_point.len() - inner_vars..];
        let mut inner_basis = vec![F::zero(); 1usize << inner_vars];
        multilinear_lagrange_basis(&mut inner_basis, inner_point);
        let matrix = reshape_columns(&ring_witness, m_rows, n_cols);
        let partial_evals = partial_evaluate_columns(&matrix, &inner_basis);

        let mut outer_basis = vec![F::zero(); 1usize << outer_vars];
        multilinear_lagrange_basis(&mut outer_basis, &eval_point[..outer_vars]);
        let mut eval_ring = CyclotomicRing::<F, D>::zero();
        for (v, basis) in partial_evals.iter().zip(outer_basis.iter()) {
            eval_ring += v.scale(basis);
        }
        assert!(eval_ring.coefficients()[1..].iter().all(|c| c.is_zero()));
        let eval_value = eval_ring.coefficients()[0];

        let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_GREYHOUND_EVAL);
        let (proof, witness, _statement) = greyhound_eval(
            &coeffs,
            &eval_point,
            eval_value,
            &[],
            &comkey_seed,
            &mut transcript,
        )
        .unwrap();

        let cfg = &proof.config;
        assert!(cfg.kappa1 > 0);
        let t_hat = &witness.rows()[2];
        let b_mat = derive_extendable_comkey_matrix::<F, D>(
            cfg.kappa1,
            t_hat.len(),
            &comkey_seed,
            b"labrador/comkey/B",
        );
        let u1 = mat_vec_mul(&b_mat, t_hat);

        let z_norm_sq = witness.rows()[0]
            .iter()
            .chain(witness.rows()[1].iter())
            .map(|ring| ring.coeff_norm_sq())
            .fold(0u128, |acc, v| acc.saturating_add(v));
        let mut verifier_transcript = Blake2bTranscript::<F>::new(DOMAIN_GREYHOUND_EVAL);
        greyhound_verify_stage1(
            &proof,
            &u1,
            &eval_point,
            eval_value,
            &witness,
            z_norm_sq,
            &comkey_seed,
            &mut verifier_transcript,
        )
        .unwrap();
    }

    #[test]
    fn stage2_single_labrador_fold_verifies() {
        let comkey_seed = [42u8; 32];

        let ring_elems = 16;
        let coeffs = vec![F::zero(); ring_elems * D];

        let ring_witness = pack_coefficients_to_ring::<F, D>(&coeffs);
        let (m_rows, n_cols, inner_vars) = choose_dimensions(ring_witness.len());
        let outer_vars = n_cols.trailing_zeros() as usize;
        let eval_point: Vec<F> = (0..(inner_vars + outer_vars))
            .map(|i| F::from_i64(i as i64 + 2))
            .collect();

        let inner_point = &eval_point[eval_point.len() - inner_vars..];
        let mut inner_basis = vec![F::zero(); 1usize << inner_vars];
        multilinear_lagrange_basis(&mut inner_basis, inner_point);
        let matrix = reshape_columns(&ring_witness, m_rows, n_cols);
        let partial_evals = partial_evaluate_columns(&matrix, &inner_basis);

        let mut outer_basis = vec![F::zero(); 1usize << outer_vars];
        multilinear_lagrange_basis(&mut outer_basis, &eval_point[..outer_vars]);
        let mut eval_ring = CyclotomicRing::<F, D>::zero();
        for (v, basis) in partial_evals.iter().zip(outer_basis.iter()) {
            eval_ring += v.scale(basis);
        }
        let eval_value = eval_ring.coefficients()[0];

        let mut gh_transcript = Blake2bTranscript::<F>::new(DOMAIN_GREYHOUND_EVAL);
        let (proof, witness, _statement) = greyhound_eval(
            &coeffs,
            &eval_point,
            eval_value,
            &[],
            &comkey_seed,
            &mut gh_transcript,
        )
        .unwrap();

        let z_norm_sq = witness.rows()[0]
            .iter()
            .chain(witness.rows()[1].iter())
            .map(|ring| ring.coeff_norm_sq())
            .fold(0u128, |acc, v| acc.saturating_add(v));
        let t_hat = &witness.rows()[2];
        let b_mat = derive_extendable_comkey_matrix::<F, D>(
            proof.config.kappa1,
            t_hat.len(),
            &comkey_seed,
            b"labrador/comkey/B",
        );
        let u1 = mat_vec_mul(&b_mat, t_hat);
        let mut gh_verify_transcript = Blake2bTranscript::<F>::new(DOMAIN_GREYHOUND_EVAL);
        greyhound_verify_stage1(
            &proof,
            &u1,
            &eval_point,
            eval_value,
            &witness,
            z_norm_sq,
            &comkey_seed,
            &mut gh_verify_transcript,
        )
        .unwrap();

        let mut transcript_replay = Blake2bTranscript::<F>::new(DOMAIN_GREYHOUND_EVAL);
        absorb_greyhound_eval_context(
            &mut transcript_replay,
            &GreyhoundEvalTranscriptContext {
                m_rows: proof.m_rows,
                n_cols: proof.n_cols,
                inner_vars: proof.inner_vars,
                eval_point_len: eval_point.len(),
            },
        )
        .unwrap();
        absorb_greyhound_eval_claim(&mut transcript_replay, &eval_point, &eval_value);
        absorb_greyhound_u2(&mut transcript_replay, &proof.u2);
        let fold_challenges: Vec<F> = (0..proof.n_cols)
            .map(|_| sample_greyhound_fold_challenge(&mut transcript_replay))
            .collect();
        let mut statement = greyhound_reduce(
            &proof,
            &u1,
            &eval_point,
            eval_value,
            &fold_challenges,
            &comkey_seed,
        )
        .unwrap();
        statement.beta_sq = witness.norm();

        let r = witness.rows().len();
        let max_len = witness
            .rows()
            .iter()
            .map(|row| row.len())
            .max()
            .unwrap_or(0);
        let setup =
            crate::protocol::labrador::LabradorSetup::new(&proof.config, r, max_len, &comkey_seed);
        let mut prover_transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        let fold = prove_level(
            &witness,
            &statement,
            &proof.config,
            &setup,
            0,
            &mut prover_transcript,
        )
        .unwrap();

        let labrador_proof = LabradorProof {
            levels: vec![fold.level_proof.clone()],
            final_opening_witness: fold.next_witness.clone(),
        };
        let mut verify_transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        let verify_result = verify(
            &statement,
            &labrador_proof,
            &comkey_seed,
            &mut verify_transcript,
        )
        .unwrap();
        assert_eq!(verify_result.final_opening_witness, fold.next_witness);
        assert_eq!(verify_result.terminal_statement, fold.statement);
    }

    #[test]
    fn stage3_full_labrador_recursion_verifies() {
        let comkey_seed = [42u8; 32];

        let ring_elems = 16;
        let coeffs = vec![F::zero(); ring_elems * D];

        let ring_witness = pack_coefficients_to_ring::<F, D>(&coeffs);
        let (m_rows, n_cols, inner_vars) = choose_dimensions(ring_witness.len());
        let outer_vars = n_cols.trailing_zeros() as usize;
        let eval_point: Vec<F> = (0..(inner_vars + outer_vars))
            .map(|i| F::from_i64(i as i64 + 3))
            .collect();

        let inner_point = &eval_point[eval_point.len() - inner_vars..];
        let mut inner_basis = vec![F::zero(); 1usize << inner_vars];
        multilinear_lagrange_basis(&mut inner_basis, inner_point);
        let matrix = reshape_columns(&ring_witness, m_rows, n_cols);
        let partial_evals = partial_evaluate_columns(&matrix, &inner_basis);

        let mut outer_basis = vec![F::zero(); 1usize << outer_vars];
        multilinear_lagrange_basis(&mut outer_basis, &eval_point[..outer_vars]);
        let mut eval_ring = CyclotomicRing::<F, D>::zero();
        for (v, basis) in partial_evals.iter().zip(outer_basis.iter()) {
            eval_ring += v.scale(basis);
        }
        let eval_value = eval_ring.coefficients()[0];

        let mut gh_transcript = Blake2bTranscript::<F>::new(DOMAIN_GREYHOUND_EVAL);
        let (proof, witness, _statement) = greyhound_eval(
            &coeffs,
            &eval_point,
            eval_value,
            &[],
            &comkey_seed,
            &mut gh_transcript,
        )
        .unwrap();

        let z_norm_sq = witness.rows()[0]
            .iter()
            .chain(witness.rows()[1].iter())
            .map(|ring| ring.coeff_norm_sq())
            .fold(0u128, |acc, v| acc.saturating_add(v));
        let t_hat = &witness.rows()[2];
        let b_mat = derive_extendable_comkey_matrix::<F, D>(
            proof.config.kappa1,
            t_hat.len(),
            &comkey_seed,
            b"labrador/comkey/B",
        );
        let u1 = mat_vec_mul(&b_mat, t_hat);
        let mut gh_verify_transcript = Blake2bTranscript::<F>::new(DOMAIN_GREYHOUND_EVAL);
        greyhound_verify_stage1(
            &proof,
            &u1,
            &eval_point,
            eval_value,
            &witness,
            z_norm_sq,
            &comkey_seed,
            &mut gh_verify_transcript,
        )
        .unwrap();

        let mut transcript_replay = Blake2bTranscript::<F>::new(DOMAIN_GREYHOUND_EVAL);
        absorb_greyhound_eval_context(
            &mut transcript_replay,
            &GreyhoundEvalTranscriptContext {
                m_rows: proof.m_rows,
                n_cols: proof.n_cols,
                inner_vars: proof.inner_vars,
                eval_point_len: eval_point.len(),
            },
        )
        .unwrap();
        absorb_greyhound_eval_claim(&mut transcript_replay, &eval_point, &eval_value);
        absorb_greyhound_u2(&mut transcript_replay, &proof.u2);
        let fold_challenges: Vec<F> = (0..proof.n_cols)
            .map(|_| sample_greyhound_fold_challenge(&mut transcript_replay))
            .collect();
        let mut statement = greyhound_reduce(
            &proof,
            &u1,
            &eval_point,
            eval_value,
            &fold_challenges,
            &comkey_seed,
        )
        .unwrap();
        statement.beta_sq = witness.norm();

        let mut prover_transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        let labrador_proof = prove_with_config(
            witness,
            &statement,
            &proof.config,
            &comkey_seed,
            &mut prover_transcript,
        )
        .unwrap();
        assert!(
            !labrador_proof.levels.is_empty(),
            "expected Labrador recursion to run at least one level"
        );

        let mut verify_transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        let verify_result = verify(
            &statement,
            &labrador_proof,
            &comkey_seed,
            &mut verify_transcript,
        )
        .unwrap();
        assert_eq!(
            verify_result.final_opening_witness,
            labrador_proof.final_opening_witness
        );
    }
}
