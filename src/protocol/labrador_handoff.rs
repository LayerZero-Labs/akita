//! Direct Labrador handoff from Hachi's quadratic equation.
//!
//! Instead of computing the quotient `r`, evaluating at a random `alpha`, and
//! running sumcheck, this module converts the ring-level relation `Mz = y`
//! directly into Labrador constraints.  The witness `w` is `[w_hat | t_hat | z_pre]`
//! with no quotient portion.

use crate::algebra::ring::CyclotomicRing;
use crate::algebra::SparseChallenge;
use crate::error::HachiError;
use crate::primitives::poly::multilinear_lagrange_basis;
use crate::primitives::serialization::Valid;
use crate::protocol::commitment::transcript_append::AppendToTranscript;
use crate::protocol::commitment::utils::crt_ntt::build_ntt_slot;
use crate::protocol::commitment::utils::flat_matrix::FlatMatrix;
use crate::protocol::commitment::utils::linear::flatten_i8_blocks;
use crate::protocol::commitment::utils::matrix::derive_public_matrix;
use crate::protocol::commitment::{
    CommitmentConfig, HachiCommitmentLayout, HachiExpandedSetup, RingCommitment,
};
use crate::protocol::hachi_poly_ops::{BalancedDigitPoly, HachiPolyOps};
use crate::protocol::labrador::types::{
    LabradorReductionConfig, LabradorStatement, LabradorWitness,
};
use crate::protocol::labrador::{prove_with_config, LabradorConstraint, LabradorConstraintTerm};
use crate::protocol::opening_point::{BasisMode, RingOpeningPoint};
use crate::protocol::proof::{FlatLabradorProof, FlatRingVec, HachiProofTail, LabradorTail};
use crate::protocol::quadratic_equation::QuadraticEquation;
use crate::protocol::ring_switch::WCommitmentConfig;
use crate::protocol::transcript::labels::{ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS};
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore, FieldSampling, FromSmallInt};

/// Build Labrador constraints that encode the ring-level relation `Mz = y` from
/// the Hachi quadratic equation.
///
/// Witness layout (3 rows):
///   row 0: `w_hat_flat`  — `depth_open * num_blocks` ring elements
///   row 1: `t_hat_flat`  — `depth_open * N_A * num_blocks` ring elements
///   row 2: `z_pre_decomp` — `depth_fold * inner_width` ring elements
///
/// Constraint rows (all ring-level, no alpha evaluation):
///   - N_D constraints: `D_mat * w_hat_flat = v`
///   - N_B constraints: `B_mat * t_hat_flat = u`
///   - 1 constraint:    `b^T * G_open * w_hat = y_eval`
///   - 1 constraint:    `c^T * G_open * w_hat - a^T * G_commit * J * z_pre = 0`
///   - N_A constraints: `c^T * G_open * t_hat_slice - A * J * z_pre = 0`
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_hachi_labrador_constraints<F, const D: usize, Cfg>(
    a_mat: &FlatMatrix<F>,
    b_mat: &FlatMatrix<F>,
    d_mat: &FlatMatrix<F>,
    opening_point: &RingOpeningPoint<F>,
    challenges: &[SparseChallenge],
    v: &[CyclotomicRing<F, D>],
    u: &[CyclotomicRing<F, D>],
    y_eval: &CyclotomicRing<F, D>,
    layout: HachiCommitmentLayout,
) -> Result<Vec<LabradorConstraint<F, D>>, HachiError>
where
    F: FieldCore + CanonicalField,
    Cfg: CommitmentConfig,
{
    let depth_open = layout.num_digits_open;
    let depth_commit = layout.num_digits_commit;
    let depth_fold = layout.num_digits_fold;
    let log_basis = layout.log_basis;
    let num_blocks = opening_point.b.len();
    let block_len = layout.block_len;
    let inner_width = block_len * depth_commit;

    let w_len = depth_open * num_blocks;
    let t_len = depth_open * Cfg::N_A * num_blocks;
    let z_len = depth_fold * inner_width;

    let g_open = gadget_scalars::<F>(depth_open, log_basis);
    let g_commit = gadget_scalars::<F>(depth_commit, log_basis);
    let j_fold = gadget_scalars::<F>(depth_fold, log_basis);

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

    let dense_challenges: Vec<CyclotomicRing<F, D>> = challenges
        .iter()
        .map(|c| c.to_dense::<F, D>().expect("valid challenge"))
        .collect();

    let mut constraints = Vec::with_capacity(Cfg::N_D + Cfg::N_B + 2 + Cfg::N_A);

    // --- D-rows: D_mat * w_hat_flat = v ---
    let d_view = d_mat.view::<D>();
    for (i, &v_i) in v.iter().enumerate().take(Cfg::N_D) {
        let d_row = d_view.row(i);
        let coeffs: Vec<CyclotomicRing<F, D>> = d_row.iter().take(w_len).copied().collect();
        constraints.push(LabradorConstraint::new(
            vec![LabradorConstraintTerm::new(0, 0, coeffs)],
            v_i,
        ));
    }

    // --- B-rows: B_mat * t_hat_flat = u ---
    let b_view = b_mat.view::<D>();
    for (i, &u_i) in u.iter().enumerate().take(Cfg::N_B) {
        let b_row = b_view.row(i);
        let coeffs: Vec<CyclotomicRing<F, D>> = b_row.iter().take(t_len).copied().collect();
        constraints.push(LabradorConstraint::new(
            vec![LabradorConstraintTerm::new(1, 0, coeffs)],
            u_i,
        ));
    }

    // --- bTw row: sum_i b_i * G_open * w_hat_block_i = y_eval ---
    {
        let mut phi_w = vec![CyclotomicRing::<F, D>::zero(); w_len];
        for (i, &b_i) in opening_point.b.iter().enumerate() {
            for (d, &g) in g_open.iter().enumerate() {
                phi_w[i * depth_open + d] = scalar_ring(b_i * g);
            }
        }
        constraints.push(LabradorConstraint::new(
            vec![LabradorConstraintTerm::new(0, 0, phi_w)],
            *y_eval,
        ));
    }

    // --- fold row: c^T * G_open * w_hat - a^T * G_commit * J * z_pre = 0 ---
    {
        let mut phi_w = vec![CyclotomicRing::<F, D>::zero(); w_len];
        for (i, c_i) in dense_challenges.iter().enumerate() {
            for (d, &g) in g_open.iter().enumerate() {
                phi_w[i * depth_open + d] = c_i.scale(&g);
            }
        }

        let mut phi_z = vec![CyclotomicRing::<F, D>::zero(); z_len];
        for (i, &a_i) in opening_point.a.iter().enumerate() {
            for (d, &g) in g_commit.iter().enumerate() {
                let ag = a_i * g;
                for (t, &j) in j_fold.iter().enumerate() {
                    let idx = (i * depth_commit + d) * depth_fold + t;
                    phi_z[idx] = scalar_ring(-(ag * j));
                }
            }
        }
        constraints.push(LabradorConstraint::new(
            vec![
                LabradorConstraintTerm::new(0, 0, phi_w),
                LabradorConstraintTerm::new(2, 0, phi_z),
            ],
            CyclotomicRing::<F, D>::zero(),
        ));
    }

    // --- A-rows: c^T * G_open * t_hat_slice[a_idx] - A[a_idx] * J * z_pre = 0 ---
    let a_view = a_mat.view::<D>();
    for a_idx in 0..Cfg::N_A {
        let mut phi_t = vec![CyclotomicRing::<F, D>::zero(); t_len];
        for (i, c_i) in dense_challenges.iter().enumerate() {
            for (d, &g) in g_open.iter().enumerate() {
                let t_idx = i * (Cfg::N_A * depth_open) + a_idx * depth_open + d;
                phi_t[t_idx] = c_i.scale(&g);
            }
        }

        let mut phi_z = vec![CyclotomicRing::<F, D>::zero(); z_len];
        let a_row = a_view.row(a_idx);
        for (k, &a_ring) in a_row.iter().take(inner_width).enumerate() {
            for (t, &j) in j_fold.iter().enumerate() {
                phi_z[k * depth_fold + t] = -(a_ring.scale(&j));
            }
        }

        constraints.push(LabradorConstraint::new(
            vec![
                LabradorConstraintTerm::new(1, 0, phi_t),
                LabradorConstraintTerm::new(2, 0, phi_z),
            ],
            CyclotomicRing::<F, D>::zero(),
        ));
    }

    Ok(constraints)
}

/// Assemble the Labrador witness from the quad-eq prover state.
///
/// Converts i8-digit planes to ring elements and decomposes `z_pre`.
pub(crate) fn build_labrador_witness<F, const D: usize>(
    w_hat_flat: &[[i8; D]],
    t_hat_flat: &[[i8; D]],
    z_pre: &[CyclotomicRing<F, D>],
    layout: HachiCommitmentLayout,
) -> LabradorWitness<F, D>
where
    F: FieldCore + CanonicalField,
{
    let depth_fold = layout.num_digits_fold;
    let log_basis = layout.log_basis;

    let to_ring = |digits: &[i8; D]| -> CyclotomicRing<F, D> {
        CyclotomicRing::from_coefficients(std::array::from_fn(|k| F::from_i64(digits[k] as i64)))
    };

    let row0: Vec<CyclotomicRing<F, D>> = w_hat_flat.iter().map(to_ring).collect();
    let row1: Vec<CyclotomicRing<F, D>> = t_hat_flat.iter().map(to_ring).collect();

    let mut row2 = Vec::with_capacity(z_pre.len() * depth_fold);
    for z_j in z_pre {
        for plane in z_j.balanced_decompose_pow2_i8(depth_fold, log_basis) {
            row2.push(to_ring(&plane));
        }
    }

    LabradorWitness::new_unchecked(vec![row0, row1, row2])
}

/// Select Labrador reduction config for the Hachi handoff witness.
///
/// Uses the same SIS-security search as `greyhound_select_config` but
/// adapts to the Hachi witness structure (3 rows).
pub(crate) fn hachi_labrador_select_config<F: CanonicalField, const D: usize>(
    witness: &LabradorWitness<F, D>,
) -> Result<LabradorReductionConfig, HachiError> {
    let max_row_len = witness.rows().iter().map(|r| r.len()).max().unwrap_or(0);
    crate::protocol::greyhound::greyhound_select_config::<F, D>(max_row_len, witness.rows().len())
}

/// Execute the Labrador direct handoff from the Hachi folding loop.
///
/// Replaces `greyhound_handoff_prove`: instead of computing the quotient `r`,
/// evaluating at alpha, and running sumcheck, this function runs the quadratic
/// equation at D' and hands the ring-level `Mz = y` directly to Labrador.
///
/// # Errors
///
/// Propagates errors from the quad eq, Labrador config selection, or Labrador proving.
#[allow(clippy::too_many_arguments)]
pub(crate) fn labrador_handoff_prove<F, T, const D_HANDOFF: usize, Cfg>(
    current_w: &[i8],
    current_challenges: &[F],
    current_num_u: usize,
    current_num_l: usize,
    _w_eval: F,
    expanded_setup: &HachiExpandedSetup<F>,
    transcript: &mut T,
) -> Result<HachiProofTail<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + FromSmallInt + Valid,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
{
    use std::time::Instant;

    let t0 = Instant::now();

    let opening_point = super::commitment_scheme::next_level_opening_point(
        current_challenges,
        current_num_u,
        current_num_l,
    );

    let alpha = D_HANDOFF.trailing_zeros() as usize;
    if opening_point.len() < alpha {
        return Err(HachiError::InvalidPointDimension {
            expected: alpha,
            actual: opening_point.len(),
        });
    }

    let w_layout = <WCommitmentConfig<D_HANDOFF, Cfg>>::commitment_layout(opening_point.len())?;
    let target_num_vars = w_layout.m_vars + w_layout.r_vars + alpha;
    let mut padded_point = opening_point.clone();
    padded_point.resize(target_num_vars, F::zero());
    let outer_point = &padded_point[alpha..];

    let ring_opening_point = super::commitment_scheme::ring_opening_point_from_field::<F>(
        outer_point,
        w_layout.r_vars,
        w_layout.m_vars,
        BasisMode::Lagrange,
    )?;

    // Pad current_w to a multiple of D_HANDOFF (Hachi levels may use a different D).
    let padded_w: Vec<i8>;
    let w_digits = if current_w.len() % D_HANDOFF != 0 {
        let padded_len = current_w.len().div_ceil(D_HANDOFF) * D_HANDOFF;
        padded_w = {
            let mut v = current_w.to_vec();
            v.resize(padded_len, 0);
            v
        };
        &padded_w[..]
    } else {
        current_w
    };

    // Derive fresh commitment key matrices at D_HANDOFF from the public seed.
    // The Hachi-level matrices live at a different D and cannot be viewed at D_HANDOFF.
    let public_seed = &expanded_setup.seed.public_matrix_seed;
    let a_matrix =
        derive_public_matrix::<F, D_HANDOFF>(Cfg::N_A, w_layout.inner_width, public_seed, b"A");
    let b_matrix =
        derive_public_matrix::<F, D_HANDOFF>(Cfg::N_B, w_layout.outer_width, public_seed, b"B");
    let d_matrix =
        derive_public_matrix::<F, D_HANDOFF>(Cfg::N_D, w_layout.d_matrix_width, public_seed, b"D");
    let a_flat = FlatMatrix::from_ring_matrix(&a_matrix);
    let b_flat = FlatMatrix::from_ring_matrix(&b_matrix);
    let d_flat = FlatMatrix::from_ring_matrix(&d_matrix);

    let ntt_a = build_ntt_slot(a_flat.view::<D_HANDOFF>())?;
    let ntt_b = build_ntt_slot(b_flat.view::<D_HANDOFF>())?;
    let ntt_d = build_ntt_slot(d_flat.view::<D_HANDOFF>())?;

    // Create a fresh commitment at D_HANDOFF (Hachi levels may use a different D).
    let (w_commitment, typed_hint) = {
        use crate::protocol::ring_switch::commit_w;
        commit_w::<F, D_HANDOFF, WCommitmentConfig<D_HANDOFF, Cfg>>(w_digits, &ntt_a, &ntt_b)?
    };

    let w_poly = BalancedDigitPoly::<F, D_HANDOFF>::from_i8_digits(w_digits)?;

    let (y_ring, w_folded) = w_poly.evaluate_and_fold(
        &ring_opening_point.b,
        &ring_opening_point.a,
        w_layout.block_len,
    );

    w_commitment.append_to_transcript(ABSORB_COMMITMENT, transcript);
    for pt in &padded_point {
        transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
    }
    transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);

    let quad_eq = Box::new(QuadraticEquation::<
        F,
        D_HANDOFF,
        WCommitmentConfig<D_HANDOFF, Cfg>,
    >::new_prover(
        &ntt_d,
        ring_opening_point.clone(),
        &w_poly,
        w_folded,
        typed_hint,
        transcript,
        &w_commitment,
        &y_ring,
        w_layout,
    )?);

    eprintln!(
        "  [labrador_handoff] quad_eq: {:.2}s",
        t0.elapsed().as_secs_f64()
    );

    let t1 = Instant::now();

    let w_hat_flat = quad_eq
        .w_hat_flat()
        .ok_or_else(|| HachiError::InvalidInput("missing w_hat_flat".into()))?;
    let t_hat = &quad_eq
        .hint()
        .ok_or_else(|| HachiError::InvalidInput("missing hint".into()))?
        .t_hat;
    let t_hat_flat = flatten_i8_blocks(t_hat);
    let z_pre = quad_eq
        .z_pre()
        .ok_or_else(|| HachiError::InvalidInput("missing z_pre".into()))?;

    let constraints =
        build_hachi_labrador_constraints::<F, D_HANDOFF, WCommitmentConfig<D_HANDOFF, Cfg>>(
            &a_flat,
            &b_flat,
            &d_flat,
            quad_eq.opening_point(),
            &quad_eq.challenges,
            &quad_eq.v,
            &w_commitment.u,
            &y_ring,
            w_layout,
        )?;

    let witness = build_labrador_witness(w_hat_flat, &t_hat_flat, z_pre, w_layout);
    let beta_sq = witness.norm();

    let cfg = hachi_labrador_select_config::<F, D_HANDOFF>(&witness)?;

    let statement = LabradorStatement {
        u1: Vec::new(),
        u2: Vec::new(),
        challenges: Vec::new(),
        constraints,
        beta_sq,
    };

    let comkey_seed = expanded_setup.labrador_comkey_seed();

    eprintln!(
        "  [labrador_handoff] witness/constraints: {:.2}s (rows={}, constraint_count={})",
        t1.elapsed().as_secs_f64(),
        witness.rows().len(),
        statement.constraints.len(),
    );

    let t2 = Instant::now();
    let labrador_proof =
        prove_with_config::<F, T, D_HANDOFF>(witness, &statement, &cfg, &comkey_seed, transcript)?;

    eprintln!(
        "  [labrador_handoff] labrador prove: {:.2}s (levels={})",
        t2.elapsed().as_secs_f64(),
        labrador_proof.levels.len(),
    );

    // Section 4.5 ring dimension switch: compute eval_ring for the verifier's
    // consistency check (opening_value == sum basis_p * eval_ring.coeff[p]).
    let alpha_prime = D_HANDOFF.trailing_zeros() as usize;
    let ring_point = &opening_point[alpha_prime..];
    let witness_coeffs: Vec<F> = w_digits.iter().map(|&d| F::from_i64(d as i64)).collect();
    let ring_witness: Vec<CyclotomicRing<F, D_HANDOFF>> = {
        let mut out = Vec::with_capacity(witness_coeffs.len().div_ceil(D_HANDOFF));
        for chunk in witness_coeffs.chunks(D_HANDOFF) {
            out.push(CyclotomicRing::from_coefficients(std::array::from_fn(
                |i| chunk.get(i).copied().unwrap_or_else(F::zero),
            )));
        }
        out
    };
    let n_ring = ring_witness.len().next_power_of_two();
    let mut ring_basis = vec![F::zero(); n_ring];
    multilinear_lagrange_basis(&mut ring_basis, ring_point);
    let mut eval_ring = CyclotomicRing::<F, D_HANDOFF>::zero();
    for (elem, &basis) in ring_witness.iter().zip(ring_basis.iter()) {
        eval_ring += elem.scale(&basis);
    }

    Ok(HachiProofTail::Labrador(LabradorTail {
        labrador_proof: FlatLabradorProof::from_typed(&labrador_proof),
        v: FlatRingVec::from_ring_elems(&quad_eq.v),
        u: FlatRingVec::from_ring_elems(&w_commitment.u),
        y_ring: FlatRingVec::from_single(&y_ring),
        eval_ring: FlatRingVec::from_ring_elems(&[eval_ring]),
        config: cfg,
        beta_sq,
    }))
}

/// Verify the direct Labrador tail of a Hachi proof.
///
/// Replays the quadratic equation transcript operations (absorb commitment,
/// evaluation claims, v; derive challenges), rebuilds the ring-level Labrador
/// constraints, and verifies the Labrador recursive proof.
///
/// # Errors
///
/// Propagates errors from constraint reconstruction or Labrador verification.
#[allow(clippy::too_many_arguments)]
pub(crate) fn labrador_handoff_verify<F, T, const D_HANDOFF: usize, Cfg>(
    tail: &LabradorTail<F>,
    opening_point: &[F],
    opening_value: &F,
    expanded_setup: &HachiExpandedSetup<F>,
    transcript: &mut T,
) -> Result<(), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + FromSmallInt + Valid,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
{
    let eval_ring_vec = tail.eval_ring.to_vec::<D_HANDOFF>();
    if eval_ring_vec.len() != 1 {
        return Err(HachiError::InvalidInput(
            "labrador tail: expected exactly one eval_ring element".into(),
        ));
    }
    let eval_ring: CyclotomicRing<F, D_HANDOFF> = eval_ring_vec[0];

    // Consistency check: opening_value == sum basis_p * eval_ring.coeff[p]
    let alpha_prime = D_HANDOFF.trailing_zeros() as usize;
    if opening_point.len() < alpha_prime {
        return Err(HachiError::InvalidPointDimension {
            expected: alpha_prime,
            actual: opening_point.len(),
        });
    }
    let coeff_point = &opening_point[..alpha_prime];
    let mut coeff_basis = vec![F::zero(); D_HANDOFF];
    multilinear_lagrange_basis(&mut coeff_basis, coeff_point);
    let mut reconstructed = F::zero();
    for (p, &basis_p) in coeff_basis.iter().enumerate() {
        reconstructed += basis_p * eval_ring.coefficients()[p];
    }
    if reconstructed != *opening_value {
        return Err(HachiError::InvalidInput(
            "labrador handoff: eval_ring inconsistent with opening_value".into(),
        ));
    }

    let v: Vec<CyclotomicRing<F, D_HANDOFF>> = tail.v.to_vec();
    let y_ring: CyclotomicRing<F, D_HANDOFF> = tail.y_ring.to_single();
    let labrador_proof = tail.labrador_proof.to_typed::<D_HANDOFF>();

    let w_layout = <WCommitmentConfig<D_HANDOFF, Cfg>>::commitment_layout(opening_point.len())?;
    let target_num_vars = w_layout.m_vars + w_layout.r_vars + alpha_prime;
    let mut padded_point = opening_point.to_vec();
    padded_point.resize(target_num_vars, F::zero());
    let outer_point = &padded_point[alpha_prime..];

    let ring_opening_point = super::commitment_scheme::ring_opening_point_from_field::<F>(
        outer_point,
        w_layout.r_vars,
        w_layout.m_vars,
        BasisMode::Lagrange,
    )?;

    // Replay transcript: absorb commitment (from the tail), padded_point, y_ring.
    let typed_commitment: RingCommitment<F, D_HANDOFF> = tail.u.to_ring_commitment();
    typed_commitment.append_to_transcript(ABSORB_COMMITMENT, transcript);
    for pt in &padded_point {
        transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
    }
    transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);

    // Derive challenges via verifier-side quad eq (absorbs v, samples challenges).
    let quad_eq =
        QuadraticEquation::<F, D_HANDOFF, WCommitmentConfig<D_HANDOFF, Cfg>>::new_verifier(
            ring_opening_point.clone(),
            v.clone(),
            transcript,
            &typed_commitment,
            &y_ring,
            w_layout,
        )?;

    // Derive fresh matrices at D_HANDOFF from the public seed (same as prover).
    let public_seed = &expanded_setup.seed.public_matrix_seed;
    let a_flat = FlatMatrix::from_ring_matrix(&derive_public_matrix::<F, D_HANDOFF>(
        Cfg::N_A,
        w_layout.inner_width,
        public_seed,
        b"A",
    ));
    let b_flat = FlatMatrix::from_ring_matrix(&derive_public_matrix::<F, D_HANDOFF>(
        Cfg::N_B,
        w_layout.outer_width,
        public_seed,
        b"B",
    ));
    let d_flat = FlatMatrix::from_ring_matrix(&derive_public_matrix::<F, D_HANDOFF>(
        Cfg::N_D,
        w_layout.d_matrix_width,
        public_seed,
        b"D",
    ));

    // Rebuild constraints from public data.
    let constraints =
        build_hachi_labrador_constraints::<F, D_HANDOFF, WCommitmentConfig<D_HANDOFF, Cfg>>(
            &a_flat,
            &b_flat,
            &d_flat,
            quad_eq.opening_point(),
            &quad_eq.challenges,
            &v,
            &typed_commitment.u,
            &y_ring,
            w_layout,
        )?;

    let statement = LabradorStatement {
        u1: Vec::new(),
        u2: Vec::new(),
        challenges: Vec::new(),
        constraints,
        beta_sq: tail.beta_sq,
    };

    let comkey_seed = expanded_setup.labrador_comkey_seed();

    let result = crate::protocol::labrador::verify::<F, T, D_HANDOFF>(
        &statement,
        &labrador_proof,
        &comkey_seed,
        transcript,
    );
    eprintln!(
        "  [labrador_handoff_verify] labrador verify: {}",
        if result.is_ok() { "OK" } else { "FAIL" }
    );
    result?;

    Ok(())
}

fn gadget_scalars<F: FieldCore + CanonicalField>(levels: usize, log_basis: u32) -> Vec<F> {
    let base = F::from_canonical_u128_reduced(1u128 << log_basis);
    let mut out = Vec::with_capacity(levels);
    let mut power = F::one();
    for _ in 0..levels {
        out.push(power);
        power = power * base;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::commitment::utils::linear::flatten_i8_blocks;
    use crate::protocol::commitment::{HachiCommitmentCore, RingCommitmentScheme};
    use crate::protocol::hachi_poly_ops::{DensePoly, HachiPolyOps};
    use crate::protocol::proof::HachiCommitmentHint;
    use crate::protocol::quadratic_equation::QuadraticEquation;
    use crate::protocol::transcript::Blake2bTranscript;
    use crate::test_utils::*;
    use crate::Transcript;

    const TRANSCRIPT_SEED: &[u8] = b"test/labrador_handoff";

    /// Verify that the Labrador constraints built from the quad eq are
    /// satisfied by the corresponding witness.
    #[test]
    fn constraints_satisfied_by_witness() {
        let (setup, _) =
            <HachiCommitmentCore as RingCommitmentScheme<F, D, TinyConfig>>::setup(16).unwrap();

        let blocks = sample_blocks();
        let w =
            <HachiCommitmentCore as RingCommitmentScheme<F, D, TinyConfig>>::commit_ring_blocks(
                &blocks, &setup,
            )
            .unwrap();

        let point = crate::protocol::opening_point::RingOpeningPoint {
            a: sample_a(),
            b: sample_b(),
        };

        let ring_coeffs: Vec<CyclotomicRing<F, D>> =
            blocks.iter().flat_map(|b| b.iter().copied()).collect();
        let poly = DensePoly::from_ring_coeffs(ring_coeffs);
        let hint = HachiCommitmentHint::new(w.t_hat);
        let mut transcript = Blake2bTranscript::<F>::new(TRANSCRIPT_SEED);
        let layout = setup.layout();
        let (y_ring, w_folded) = poly.evaluate_and_fold(&point.b, &point.a, layout.block_len);

        let quad_eq = QuadraticEquation::<F, D, TinyConfig>::new_prover(
            &setup.ntt_D,
            point.clone(),
            &poly,
            w_folded,
            hint,
            &mut transcript,
            &w.commitment,
            &y_ring,
            layout,
        )
        .unwrap();

        let w_hat_flat = quad_eq.w_hat_flat().unwrap();
        let t_hat = &quad_eq.hint().unwrap().t_hat;
        let t_hat_flat = flatten_i8_blocks(t_hat);
        let z_pre = quad_eq.z_pre().unwrap();

        let constraints = build_hachi_labrador_constraints::<F, D, TinyConfig>(
            &setup.expanded.A,
            &setup.expanded.B,
            &setup.expanded.D_mat,
            quad_eq.opening_point(),
            &quad_eq.challenges,
            &quad_eq.v,
            &w.commitment.u,
            &y_ring,
            layout,
        )
        .unwrap();

        let witness = build_labrador_witness(w_hat_flat, &t_hat_flat, z_pre, layout);

        let rows = witness.rows();
        for (ci, constraint) in constraints.iter().enumerate() {
            let mut lhs = CyclotomicRing::<F, D>::zero();
            for term in &constraint.terms {
                for (j, coeff) in term.coefficients.iter().enumerate() {
                    let idx = term.offset + j;
                    if idx < rows[term.row].len() {
                        lhs += *coeff * rows[term.row][idx];
                    }
                }
            }
            assert_eq!(lhs, constraint.target, "constraint {ci} not satisfied");
        }
    }
}
