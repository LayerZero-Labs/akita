//! Greyhound verifier-side reduction to Labrador statement.
//!
//! Builds scalar Labrador constraints matching the C reference relations,
//! adapted for multilinear evaluation. The fold challenges are passed in
//! (sampled by the caller from the transcript) so this function is transcript-free.

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::primitives::poly::multilinear_lagrange_basis;
use crate::protocol::greyhound::types::GreyhoundEvalProof;
use crate::protocol::labrador::comkey::{derive_extendable_comkey_matrix, LabradorComKeySeed};
use crate::protocol::labrador::types::LabradorStatement;
use crate::protocol::labrador::{LabradorConstraint, LabradorConstraintTerm};
use crate::{CanonicalField, FieldCore, FieldSampling};

/// Rebuild a Labrador statement from Greyhound proof data and fold challenges.
///
/// The scalar constraints encode (multilinear adaptation of C's `polcom_reduce`):
///   - one outer-commitment equation per entry of `u1`
///   - one eval-witness equation per entry of `u2`
///   - one amortization consistency equation
///   - one inner-commitment equation per row of `A`
///   - one evaluation equation
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
    );

    Ok(LabradorStatement {
        u1: w_commitment_u1.to_vec(),
        u2: eval_proof.u2.clone(),
        challenges: Vec::new(),
        constraints,
        beta_sq: 0,
    })
}

/// Build the scalar constraints for the 4-row Greyhound witness.
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
    let t_hat_len = kappa * fu * n;
    let v_hat_len = fu * n;
    let z_group_len = m * f;

    let mut constraints = Vec::new();

    // cnst0: B · row2 = u1
    if kappa1 > 0 {
        let b_mat = derive_extendable_comkey_matrix::<F, D>(
            kappa1,
            t_hat_len,
            comkey_seed,
            b"labrador/comkey/B",
        );
        constraints.extend(
            b_mat
                .into_iter()
                .zip(u1.iter().copied())
                .map(|(coeffs, target)| {
                    LabradorConstraint::new(vec![LabradorConstraintTerm::new(2, 0, coeffs)], target)
                }),
        );
    } else {
        let one = CyclotomicRing::<F, D>::one();
        constraints.extend(u1.iter().copied().enumerate().map(|(out_idx, target)| {
            LabradorConstraint::new(
                vec![LabradorConstraintTerm::new(2, out_idx, vec![one])],
                target,
            )
        }));
    }

    // cnst1: B_eval · row3 = u2
    if kappa1 > 0 {
        let b_eval = derive_extendable_comkey_matrix::<F, D>(
            kappa1,
            v_hat_len,
            comkey_seed,
            b"greyhound/comkey/B_eval",
        );
        constraints.extend(b_eval.into_iter().zip(proof.u2.iter().copied()).map(
            |(coeffs, target)| {
                LabradorConstraint::new(vec![LabradorConstraintTerm::new(3, 0, coeffs)], target)
            },
        ));
    } else {
        let one = CyclotomicRing::<F, D>::one();
        constraints.extend(
            proof
                .u2
                .iter()
                .copied()
                .enumerate()
                .map(|(out_idx, target)| {
                    LabradorConstraint::new(
                        vec![LabradorConstraintTerm::new(3, out_idx, vec![one])],
                        target,
                    )
                }),
        );
    }

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
    constraints.push(LabradorConstraint::new(
        vec![
            LabradorConstraintTerm::new(0, 0, phi0),
            LabradorConstraintTerm::new(1, 0, phi1),
            LabradorConstraintTerm::new(3, 0, phi_v),
        ],
        CyclotomicRing::<F, D>::zero(),
    ));

    // cnst3: inner commitment relation  A·z - c·t = 0
    let a_mat =
        derive_extendable_comkey_matrix::<F, D>(kappa, m, comkey_seed, b"labrador/comkey/A");
    let mut phi_z0 = vec![vec![CyclotomicRing::<F, D>::zero(); z_group_len]; kappa];
    let mut phi_z1 = vec![vec![CyclotomicRing::<F, D>::zero(); z_group_len]; kappa];
    for r in 0..kappa {
        for j in 0..m {
            for k in 0..f {
                let w = a_mat[r][j].scale(&pow2(k * b));
                phi_z0[r][j * f + k] = w;
                phi_z1[r][j * f + k] = w.scale(&bu_scale);
            }
        }
    }
    let t_hat_per_col = kappa * fu;
    let mut phi_t = vec![vec![CyclotomicRing::<F, D>::zero(); t_hat_len]; kappa];
    for (i, &challenge_i) in fold_challenges.iter().enumerate().take(n) {
        for l in 0..fu {
            let neg_ci_scale = scalar_ring(-(challenge_i * pow2(l * bu)));
            for (r, phi_t_row) in phi_t.iter_mut().enumerate().take(kappa) {
                phi_t_row[i * t_hat_per_col + r * fu + l] = neg_ci_scale;
            }
        }
    }
    constraints.extend((0..kappa).map(|row_idx| {
        LabradorConstraint::new(
            vec![
                LabradorConstraintTerm::new(0, 0, phi_z0[row_idx].clone()),
                LabradorConstraintTerm::new(1, 0, phi_z1[row_idx].clone()),
                LabradorConstraintTerm::new(2, 0, phi_t[row_idx].clone()),
            ],
            CyclotomicRing::<F, D>::zero(),
        )
    }));

    // cnst4: evaluation check
    let mut phi_eval = vec![CyclotomicRing::<F, D>::zero(); v_hat_len];
    for i in 0..n {
        let ob = outer_basis.get(i).copied().unwrap_or_else(F::zero);
        for l in 0..fu {
            phi_eval[i * fu + l] = scalar_ring(ob * pow2(l * bu));
        }
    }
    constraints.push(LabradorConstraint::new(
        vec![LabradorConstraintTerm::new(3, 0, phi_eval)],
        scalar_ring(eval_value),
    ));

    constraints
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
    fn reduce_builds_scalar_constraints() {
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
        )
        .unwrap();
        assert_eq!(
            st.constraints.len(),
            st.u1.len() + st.u2.len() + cfg.kappa + 2
        );
    }
}
