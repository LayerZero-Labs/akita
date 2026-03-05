//! Greyhound verifier-side checks (stage 1, no Labrador recursion).

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::primitives::poly::multilinear_lagrange_basis;
use crate::protocol::greyhound::types::GreyhoundEvalProof;
use crate::protocol::labrador::comkey::{derive_extendable_comkey_matrix, LabradorComKeySeed};
use crate::protocol::labrador::transcript::{
    absorb_greyhound_eval_claim, absorb_greyhound_eval_context, absorb_greyhound_u2,
    sample_greyhound_fold_challenge, GreyhoundEvalTranscriptContext,
};
use crate::protocol::labrador::types::LabradorWitness;
use crate::protocol::labrador::utils::mat_vec_mul;
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore, FieldSampling};

/// Verify Greyhound evaluation proof using the full auxiliary witness.
///
/// This stage performs direct checks of the linear system `Pz = h` and a
/// smallness bound on the transmitted `z` witness. Labrador recursion is
/// intentionally skipped.
///
/// # Errors
///
/// Returns [`HachiError::InvalidInput`] on dimension mismatches, norm bound
/// violations, commitment mismatches, or constraint failures.
/// Propagates transcript replay failures from Fiat-Shamir operations.
#[allow(clippy::too_many_arguments)]
pub fn greyhound_verify_stage1<F, T, const D: usize>(
    eval_proof: &GreyhoundEvalProof<F, D>,
    w_commitment_u1: &[CyclotomicRing<F, D>],
    eval_point: &[F],
    eval_value: F,
    witness: &LabradorWitness<F, D>,
    z_beta_sq: u128,
    comkey_seed: &LabradorComKeySeed,
    transcript: &mut T,
) -> Result<(), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
{
    let m = eval_proof.m_rows;
    let n = eval_proof.n_cols;
    if m == 0 || n == 0 {
        return Err(HachiError::InvalidInput(
            "greyhound: zero-dimension proof".to_string(),
        ));
    }
    if eval_point.len() < eval_proof.inner_vars {
        return Err(HachiError::InvalidPointDimension {
            expected: eval_proof.inner_vars,
            actual: eval_point.len(),
        });
    }

    let cfg = &eval_proof.config;
    let z_group_len = m * cfg.f;
    let t_hat_len = cfg.kappa * cfg.fu * n;
    let v_hat_len = cfg.fu * n;

    let rows = witness.rows();
    if rows.len() != 4 {
        return Err(HachiError::InvalidInput(
            "greyhound: expected 4 witness rows".to_string(),
        ));
    }
    let z_low = &rows[0];
    let z_high = &rows[1];
    let t_hat = &rows[2];
    let v_hat = &rows[3];
    if z_low.len() != z_group_len
        || z_high.len() != z_group_len
        || t_hat.len() != t_hat_len
        || v_hat.len() != v_hat_len
    {
        return Err(HachiError::InvalidInput(
            "greyhound: witness row lengths mismatch".to_string(),
        ));
    }

    // Check smallness of the transmitted z witness (rows 0 and 1).
    let z_norm_sq = z_low
        .iter()
        .chain(z_high.iter())
        .map(|ring| ring.coeff_norm_sq())
        .fold(0u128, |acc, v| acc.saturating_add(v));
    if z_norm_sq > z_beta_sq {
        return Err(HachiError::InvalidInput(
            "greyhound: z norm exceeds bound".to_string(),
        ));
    }

    // Commitment checks: u1 (inner commitments) and u2 (evaluation witness).
    let u1_expected = if cfg.kappa1 > 0 {
        let b_mat = derive_extendable_comkey_matrix::<F, D>(
            cfg.kappa1,
            t_hat_len,
            comkey_seed,
            b"labrador/comkey/B",
        );
        mat_vec_mul(&b_mat, t_hat)
    } else {
        t_hat.to_vec()
    };
    if u1_expected != w_commitment_u1 {
        return Err(HachiError::InvalidInput(
            "greyhound: u1 commitment mismatch".to_string(),
        ));
    }

    let u2_expected = if cfg.kappa1 > 0 {
        let b_eval = derive_extendable_comkey_matrix::<F, D>(
            cfg.kappa1,
            v_hat_len,
            comkey_seed,
            b"greyhound/comkey/B_eval",
        );
        mat_vec_mul(&b_eval, v_hat)
    } else {
        v_hat.to_vec()
    };
    if u2_expected != eval_proof.u2 {
        return Err(HachiError::InvalidInput(
            "greyhound: u2 commitment mismatch".to_string(),
        ));
    }

    // Transcript replay to obtain fold challenges.
    absorb_greyhound_eval_context(
        transcript,
        &GreyhoundEvalTranscriptContext {
            m_rows: m,
            n_cols: n,
            inner_vars: eval_proof.inner_vars,
            eval_point_len: eval_point.len(),
        },
    )?;
    absorb_greyhound_eval_claim(transcript, eval_point, &eval_value);
    absorb_greyhound_u2(transcript, &eval_proof.u2);
    let fold_challenges: Vec<F> = (0..n)
        .map(|_| sample_greyhound_fold_challenge(transcript))
        .collect();

    // Basis vectors.
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

    let z = reconstruct_z(z_low, z_high, m, cfg.f, cfg.b, cfg.bu);
    let v = reconstruct_v(v_hat, n, cfg.fu, cfg.bu);
    let t_cols = reconstruct_t_cols(t_hat, n, cfg.kappa, cfg.fu, cfg.bu);

    // Constraint 2: <inner_basis, z> = sum_i c_i * v_i.
    let mut lhs = CyclotomicRing::<F, D>::zero();
    for (j, basis) in inner_basis.iter().enumerate() {
        let z_j = z.get(j).copied().unwrap_or_else(CyclotomicRing::zero);
        lhs += z_j.scale(basis);
    }
    let mut rhs = CyclotomicRing::<F, D>::zero();
    for (i, c_i) in fold_challenges.iter().enumerate() {
        let v_i = v.get(i).copied().unwrap_or_else(CyclotomicRing::zero);
        rhs += v_i.scale(c_i);
    }
    if lhs != rhs {
        return Err(HachiError::InvalidInput(
            "greyhound: amortization constraint failed".to_string(),
        ));
    }

    // Constraint 3: A * z = sum_i c_i * t_i.
    let a_mat =
        derive_extendable_comkey_matrix::<F, D>(cfg.kappa, m, comkey_seed, b"labrador/comkey/A");
    let lhs_vec = mat_vec_mul(&a_mat, &z);
    let mut rhs_vec = vec![CyclotomicRing::<F, D>::zero(); cfg.kappa];
    for (i, c_i) in fold_challenges.iter().enumerate() {
        if let Some(t_i) = t_cols.get(i) {
            for (r, t_ir) in t_i.iter().enumerate() {
                rhs_vec[r] += t_ir.scale(c_i);
            }
        }
    }
    if lhs_vec != rhs_vec {
        return Err(HachiError::InvalidInput(
            "greyhound: inner commitment constraint failed".to_string(),
        ));
    }

    // Constraint 4: <outer_basis, v> = eval_value.
    let mut eval_check = CyclotomicRing::<F, D>::zero();
    for (i, basis) in outer_basis.iter().enumerate() {
        let v_i = v.get(i).copied().unwrap_or_else(CyclotomicRing::zero);
        eval_check += v_i.scale(basis);
    }
    if eval_check != scalar_ring(eval_value) {
        return Err(HachiError::InvalidInput(
            "greyhound: evaluation constraint failed".to_string(),
        ));
    }

    Ok(())
}

fn pow2<F: FieldCore>(exp: usize) -> F {
    let mut v = F::one();
    for _ in 0..exp {
        v = v + v;
    }
    v
}

fn scalar_ring<F: FieldCore, const D: usize>(s: F) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|i| if i == 0 { s } else { F::zero() }))
}

fn reconstruct_z<F: FieldCore, const D: usize>(
    z_low: &[CyclotomicRing<F, D>],
    z_high: &[CyclotomicRing<F, D>],
    m: usize,
    f: usize,
    b: usize,
    bu: usize,
) -> Vec<CyclotomicRing<F, D>> {
    let mut out = vec![CyclotomicRing::<F, D>::zero(); m];
    let bu_scale = pow2::<F>(bu);
    for (j, out_elem) in out.iter_mut().enumerate() {
        let mut acc = CyclotomicRing::<F, D>::zero();
        for k in 0..f {
            let idx = j * f + k;
            let mut digit = z_low[idx];
            digit += z_high[idx].scale(&bu_scale);
            let scale = pow2::<F>(k * b);
            acc += digit.scale(&scale);
        }
        *out_elem = acc;
    }
    out
}

fn reconstruct_v<F: FieldCore, const D: usize>(
    v_hat: &[CyclotomicRing<F, D>],
    n: usize,
    fu: usize,
    bu: usize,
) -> Vec<CyclotomicRing<F, D>> {
    let mut out = vec![CyclotomicRing::<F, D>::zero(); n];
    for (i, out_elem) in out.iter_mut().enumerate() {
        let mut acc = CyclotomicRing::<F, D>::zero();
        for l in 0..fu {
            let idx = i * fu + l;
            let scale = pow2::<F>(l * bu);
            acc += v_hat[idx].scale(&scale);
        }
        *out_elem = acc;
    }
    out
}

fn reconstruct_t_cols<F: FieldCore, const D: usize>(
    t_hat: &[CyclotomicRing<F, D>],
    n: usize,
    kappa: usize,
    fu: usize,
    bu: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    let mut out = vec![vec![CyclotomicRing::<F, D>::zero(); kappa]; n];
    let per_col = kappa * fu;
    for (i, out_row) in out.iter_mut().enumerate() {
        for (r, out_elem) in out_row.iter_mut().enumerate() {
            let mut acc = CyclotomicRing::<F, D>::zero();
            for l in 0..fu {
                let idx = i * per_col + r * fu + l;
                let scale = pow2::<F>(l * bu);
                acc += t_hat[idx].scale(&scale);
            }
            *out_elem = acc;
        }
    }
    out
}
