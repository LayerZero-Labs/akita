//! Greyhound prover-side evaluation reduction.
//!
//! Produces a 4-row witness matching the C reference structure (adapted for
//! multilinear evaluation) and 5 constraints via `greyhound_reduce`.

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::primitives::poly::multilinear_lagrange_basis;
use crate::protocol::commitment::utils::linear::decompose_rows;
use crate::protocol::greyhound::reduce::greyhound_reduce;
use crate::protocol::greyhound::types::GreyhoundEvalProof;
use crate::protocol::labrador::comkey::{derive_extendable_comkey_matrix, LabradorComKeySeed};
use crate::protocol::labrador::commit::mat_vec_mul;
use crate::protocol::labrador::select_config;
use crate::protocol::labrador::transcript::{
    absorb_greyhound_eval_claim, absorb_greyhound_eval_context, absorb_greyhound_u2,
    sample_greyhound_fold_challenge, GreyhoundEvalTranscriptContext,
};
use crate::protocol::labrador::types::{LabradorStatement, LabradorWitness, LabradorWitnessRow};
use crate::protocol::prg::MatrixPrgBackendChoice;
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
    backend: MatrixPrgBackendChoice,
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
    let v_hat = decompose_rows(&partial_evals, cfg.fu, cfg.bu as u32);

    // Commit v_hat → u2 (outer commitment to evaluation witness).
    let u2 = if cfg.kappa1 > 0 {
        let b_eval = derive_extendable_comkey_matrix::<F, D>(
            cfg.kappa1,
            v_hat.len(),
            comkey_seed,
            b"greyhound/comkey/B_eval",
            backend,
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
            prg_backend_id: backend as u8,
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
    let z_first = decompose_rows(&z, cfg.f, cfg.b as u32);
    let z_uniform = decompose_rows(&z_first, 2, cfg.bu as u32);
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
            backend,
        );
        let t_j = mat_vec_mul(&a, column);
        t_hat_flat.extend(decompose_rows(&t_j, cfg.fu, cfg.bu as u32));
    }

    let row_norm =
        |row: &[CyclotomicRing<F, D>]| -> u128 { row.iter().map(|x| x.coeff_norm_sq()).sum() };
    let greyhound_witness = LabradorWitness {
        rows: vec![
            LabradorWitnessRow {
                norm_sq: row_norm(&z_low),
                s: z_low,
            },
            LabradorWitnessRow {
                norm_sq: row_norm(&z_high),
                s: z_high,
            },
            LabradorWitnessRow {
                norm_sq: row_norm(&t_hat_flat),
                s: t_hat_flat,
            },
            LabradorWitnessRow {
                norm_sq: row_norm(&v_hat),
                s: v_hat,
            },
        ],
    };

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
        backend,
    )?;
    statement.beta_sq = greyhound_witness.rows.iter().map(|r| r.norm_sq).sum();

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
fn columns_to_witness<F: FieldCore + CanonicalField, const D: usize>(
    matrix: &[Vec<CyclotomicRing<F, D>>],
) -> LabradorWitness<F, D> {
    LabradorWitness {
        rows: matrix
            .iter()
            .map(|col| {
                let norm_sq = col.iter().map(|x| x.coeff_norm_sq()).sum();
                LabradorWitnessRow {
                    s: col.clone(),
                    norm_sq,
                }
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::Fp64;
    use crate::protocol::transcript::labels::DOMAIN_GREYHOUND_EVAL;
    use crate::protocol::transcript::Blake2bTranscript;
    use crate::FromSmallInt;

    type F = Fp64<4294967197>;
    const D: usize = 64;

    #[test]
    fn eval_outputs_four_row_witness_and_five_constraints() {
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
            MatrixPrgBackendChoice::Shake256,
            &mut transcript,
        )
        .unwrap();
        assert_eq!(proof.u2, statement.u2);
        assert_eq!(witness.rows.len(), 4);
        assert_eq!(statement.constraints.len(), 5);
    }
}
