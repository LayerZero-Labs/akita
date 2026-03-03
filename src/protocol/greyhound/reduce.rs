//! Greyhound verifier-side reduction to Labrador statement.
//!
//! Builds 5 constraints matching the C reference, adapted for multilinear
//! evaluation. The fold challenges are passed in (sampled by the caller from
//! the transcript) so this function is transcript-free.

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::primitives::poly::multilinear_lagrange_basis;
use crate::protocol::greyhound::types::GreyhoundEvalProof;
use crate::protocol::labrador::comkey::{derive_extendable_comkey_matrix, LabradorComKeySeed};
use crate::protocol::labrador::types::{LabradorConstraint, LabradorStatement};
use crate::protocol::prg::MatrixPrgBackendChoice;
use crate::{CanonicalField, FieldCore, FieldSampling};

/// Rebuild a Labrador statement from Greyhound proof data and fold challenges.
///
/// The 5 constraints encode (multilinear adaptation of C's `polcom_reduce`):
///   0. Outer commitment: B · group2 = u1
///   1. Eval-witness commitment: B_eval · group3 = u2
///   2. Amortization consistency: <inner_basis, z> = <challenges, v>
///   3. Inner commitment relation: A · z = <challenges, t> (mult=kappa)
///   4. Evaluation check: <outer_basis, v> = eval_value
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
pub fn greyhound_reduce<F, const D: usize>(
    eval_proof: &GreyhoundEvalProof<F, D>,
    w_commitment_u1: &[CyclotomicRing<F, D>],
    eval_point: &[F],
    eval_value: F,
    fold_challenges: &[F],
    comkey_seed: &LabradorComKeySeed,
    backend: MatrixPrgBackendChoice,
) -> Result<LabradorStatement<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
{
    let m_rows = eval_proof.m_rows;
    let n_cols = eval_proof.n_cols;
    if n_cols == 0 || m_rows == 0 {
        return Err(HachiError::InvalidInput(
            "greyhound proof has zero dimensions".to_string(),
        ));
    }
    if eval_point.len() < eval_proof.inner_vars {
        return Err(HachiError::InvalidPointDimension {
            expected: eval_proof.inner_vars,
            actual: eval_point.len(),
        });
    }
    if fold_challenges.len() != n_cols {
        return Err(HachiError::InvalidInput(format!(
            "expected {} fold challenges, got {}",
            n_cols,
            fold_challenges.len()
        )));
    }

    let outer_vars = eval_point.len() - eval_proof.inner_vars;
    let mut outer_basis = vec![F::zero(); 1usize << outer_vars];
    multilinear_lagrange_basis(&mut outer_basis, &eval_point[..outer_vars]);
    let mut inner_basis = vec![F::zero(); 1usize << eval_proof.inner_vars];
    if eval_proof.inner_vars > 0 {
        multilinear_lagrange_basis(
            &mut inner_basis,
            &eval_point[eval_point.len() - eval_proof.inner_vars..],
        );
    } else if !inner_basis.is_empty() {
        inner_basis[0] = F::one();
    }

    let constraints = build_constraints(
        eval_proof,
        w_commitment_u1,
        &outer_basis,
        &inner_basis,
        eval_value,
        fold_challenges,
        comkey_seed,
        backend,
    );

    Ok(LabradorStatement {
        u1: w_commitment_u1.to_vec(),
        u2: eval_proof.u2.clone(),
        challenges: Vec::new(),
        constraints,
        beta_sq: 0,
        hash: [0u8; 16],
    })
}

/// Build the 5 constraints for the 4-row Greyhound witness.
///
/// Witness layout (element-major decomposition ordering):
///   row0: z_low   — m*f elements, z_low[j*f+k] = low part of k-th decomp of z[j]
///   row1: z_high  — m*f elements, z_high[j*f+k] = high part
///   row2: t_hat   — kappa*fu*n elements, per-column decomposed inner commitments
///   row3: v_hat   — fu*n elements, decomposed partial evaluations
///
/// Reconstruction:
///   z[j] = sum_k 2^{kb} * (z_low[j*f+k] + 2^bu * z_high[j*f+k])
///   t_col_i[c] = sum_l 2^{l*bu} * t_hat[i*kappa*fu + c*fu + l]
///   v[i] = sum_l 2^{l*bu} * v_hat[i*fu + l]
#[allow(clippy::too_many_arguments)]
fn build_constraints<F: FieldCore + CanonicalField + FieldSampling, const D: usize>(
    proof: &GreyhoundEvalProof<F, D>,
    u1: &[CyclotomicRing<F, D>],
    outer_basis: &[F],
    inner_basis: &[F],
    eval_value: F,
    fold_challenges: &[F],
    comkey_seed: &LabradorComKeySeed,
    backend: MatrixPrgBackendChoice,
) -> Vec<LabradorConstraint<F, D>> {
    let m = proof.m_rows;
    let n = proof.n_cols;
    let cfg = &proof.config;
    let f = cfg.f;
    let b = cfg.b;
    let fu = cfg.fu;
    let bu = cfg.bu;
    let kappa = cfg.kappa;
    let kappa1 = cfg.kappa1;

    let scalar_ring =
        |s: F| -> CyclotomicRing<F, D> {
            CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                if k == 0 {
                    s
                } else {
                    F::zero()
                }
            }))
        };

    let pow2 = |exp: usize| -> F {
        let mut v = F::one();
        for _ in 0..exp {
            v = v + v;
        }
        v
    };

    // cnst0: B · row2 = u1  (outer commitment of decomposed inner commitments)
    let num_rows = 4; // z_low, z_high, t_hat, v_hat
    let t_hat_len = kappa * fu * n;
    let v_hat_len = fu * n;
    let z_group_len = m * f;

    // cnst0: B · row2 = u1
    let c0 = if kappa1 > 0 {
        let b_mat = derive_extendable_comkey_matrix::<F, D>(
            kappa1,
            t_hat_len,
            comkey_seed,
            b"labrador/comkey/B",
            backend,
        );
        let coeffs: Vec<CyclotomicRing<F, D>> = b_mat.into_iter().flatten().collect();
        let mut coefficients = vec![vec![]; num_rows];
        coefficients[2] = coeffs;
        LabradorConstraint {
            coefficients,
            target: u1.to_vec(),
        }
    } else {
        let one = CyclotomicRing::<F, D>::one();
        let mut coefficients = vec![vec![]; num_rows];
        coefficients[2] = vec![one; u1.len()];
        LabradorConstraint {
            coefficients,
            target: u1.to_vec(),
        }
    };

    // cnst1: B_eval · row3 = u2
    let c1 = if kappa1 > 0 {
        let b_eval = derive_extendable_comkey_matrix::<F, D>(
            kappa1,
            v_hat_len,
            comkey_seed,
            b"greyhound/comkey/B_eval",
            backend,
        );
        let coeffs: Vec<CyclotomicRing<F, D>> = b_eval.into_iter().flatten().collect();
        let mut coefficients = vec![vec![]; num_rows];
        coefficients[3] = coeffs;
        LabradorConstraint {
            coefficients,
            target: proof.u2.clone(),
        }
    } else {
        let one = CyclotomicRing::<F, D>::one();
        let mut coefficients = vec![vec![]; num_rows];
        coefficients[3] = vec![one; proof.u2.len()];
        LabradorConstraint {
            coefficients,
            target: proof.u2.clone(),
        }
    };

    // cnst2: amortization consistency
    let mut phi0 = vec![CyclotomicRing::<F, D>::zero(); z_group_len];
    let mut phi1 = vec![CyclotomicRing::<F, D>::zero(); z_group_len];
    let bu_scale = pow2(bu);
    for j in 0..m {
        for k in 0..f {
            let w = scalar_ring(inner_basis[j] * pow2(k * b));
            phi0[j * f + k] = w;
            phi1[j * f + k] = w.scale(&bu_scale);
        }
    }
    let mut phi_v = vec![CyclotomicRing::<F, D>::zero(); v_hat_len];
    for i in 0..n {
        for l in 0..fu {
            phi_v[i * fu + l] = scalar_ring(-(fold_challenges[i] * pow2(l * bu)));
        }
    }
    let c2 = LabradorConstraint {
        coefficients: vec![phi0, phi1, vec![], phi_v],
        target: vec![CyclotomicRing::<F, D>::zero()],
    };

    // cnst3: inner commitment relation  A·z - c·t = 0
    let a_mat = derive_extendable_comkey_matrix::<F, D>(
        kappa,
        m,
        comkey_seed,
        b"labrador/comkey/A",
        backend,
    );
    let mut phi_z0 = vec![CyclotomicRing::<F, D>::zero(); kappa * z_group_len];
    let mut phi_z1 = vec![CyclotomicRing::<F, D>::zero(); kappa * z_group_len];
    for r in 0..kappa {
        for j in 0..m {
            for k in 0..f {
                let w = a_mat[r][j].scale(&pow2(k * b));
                phi_z0[r * z_group_len + j * f + k] = w;
                phi_z1[r * z_group_len + j * f + k] = w.scale(&bu_scale);
            }
        }
    }
    let t_hat_per_col = kappa * fu;
    let mut phi_t = vec![CyclotomicRing::<F, D>::zero(); kappa * t_hat_len];
    for i in 0..n {
        for l in 0..fu {
            let neg_ci_scale = scalar_ring(-(fold_challenges[i] * pow2(l * bu)));
            for r in 0..kappa {
                phi_t[r * t_hat_len + i * t_hat_per_col + r * fu + l] = neg_ci_scale;
            }
        }
    }
    let c3 = LabradorConstraint {
        coefficients: vec![phi_z0, phi_z1, phi_t, vec![]],
        target: vec![CyclotomicRing::<F, D>::zero(); kappa],
    };

    // cnst4: evaluation check
    let mut phi_eval = vec![CyclotomicRing::<F, D>::zero(); v_hat_len];
    for i in 0..n {
        let ob = outer_basis.get(i).copied().unwrap_or_else(F::zero);
        for l in 0..fu {
            phi_eval[i * fu + l] = scalar_ring(ob * pow2(l * bu));
        }
    }
    let mut coefficients = vec![vec![]; num_rows];
    coefficients[3] = phi_eval;
    let c4 = LabradorConstraint {
        coefficients,
        target: vec![scalar_ring(eval_value)],
    };

    vec![c0, c1, c2, c3, c4]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::Fp64;
    use crate::protocol::labrador::types::LabradorReductionConfig;
    use crate::FromSmallInt;

    type F = Fp64<4294967197>;
    const D: usize = 64;

    #[test]
    fn reduce_builds_five_constraints() {
        let cfg = LabradorReductionConfig {
            f: 1,
            b: 8,
            fu: 2,
            bu: 10,
            kappa: 3,
            kappa1: 2,
            tail: false,
        };
        let proof = GreyhoundEvalProof {
            u2: vec![CyclotomicRing::<F, D>::one(), CyclotomicRing::<F, D>::one()],
            m_rows: 4,
            n_cols: 4,
            inner_vars: 2,
            config: cfg,
        };
        let u1 = vec![CyclotomicRing::<F, D>::one(), CyclotomicRing::<F, D>::one()];
        let eval_point = vec![
            F::from_i64(1),
            F::from_i64(2),
            F::from_i64(3),
            F::from_i64(4),
        ];
        let fold_challenges = vec![
            F::from_i64(1),
            F::from_i64(2),
            F::from_i64(3),
            F::from_i64(4),
        ];
        let st = greyhound_reduce(
            &proof,
            &u1,
            &eval_point,
            F::from_i64(7),
            &fold_challenges,
            &[8u8; 32],
            MatrixPrgBackendChoice::Shake256,
        )
        .unwrap();
        assert_eq!(st.constraints.len(), 5);
    }
}
