//! Direct Labrador handoff from Hachi's quadratic equation.
//!
//! Instead of computing the quotient `r`, evaluating at a random `alpha`, and
//! running sumcheck, this module converts the ring-level relation `Mz = y`
//! directly into Labrador constraints. The witness `w` is
//! `[w_hat | inner_opening_digits | z_pre]`
//! with no quotient portion.

use crate::algebra::ring::CyclotomicRing;
use crate::algebra::SparseChallenge;
use crate::error::HachiError;
use crate::primitives::poly::multilinear_lagrange_basis;
use crate::primitives::serialization::{Compress, Valid};
use crate::protocol::commitment::transcript_append::AppendToTranscript;
use crate::protocol::commitment::utils::crt_ntt::NttSlotCache;
use crate::protocol::commitment::utils::flat_matrix::FlatMatrix;
use crate::protocol::commitment::utils::linear::flatten_i8_blocks;
use crate::protocol::commitment::{
    CommitmentConfig, HachiCommitmentLayout, HachiExpandedSetup, HachiLevelParams, RingCommitment,
};
use crate::protocol::commitment_scheme::next_level_opening_point;
use crate::protocol::hachi_poly_ops::{BalancedDigitPoly, HachiPolyOps};
use crate::protocol::labrador::config::{
    estimate_handoff_recursive_proof, logq_bits, LabradorRecursiveSizeEstimate,
};
use crate::protocol::labrador::types::{LabradorStatement, LabradorWitness};
use crate::protocol::labrador::{
    prove_with_plan, verify as verify_labrador, LabradorConstraint, LabradorConstraintTerm,
};
use crate::protocol::opening_point::{ring_opening_point_from_field, BasisMode, RingOpeningPoint};
use crate::protocol::proof::{
    FlatLabradorProof, FlatLabradorWitness, FlatRingVec, HachiCommitmentHint, HachiProofTail,
    LabradorTail, PackedDigits,
};
use crate::protocol::quadratic_equation::QuadraticEquation;
use crate::protocol::ring_switch::WCommitmentConfig;
use crate::protocol::transcript::labels::{ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS};
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore, FieldSampling, FromSmallInt, HachiSerialize};
use std::time::Instant;

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
#[tracing::instrument(skip_all, name = "labrador::handoff_build_constraints")]
pub(crate) fn build_hachi_labrador_constraints<F, const D: usize>(
    a_mat: &FlatMatrix<F>,
    b_mat: &FlatMatrix<F>,
    d_mat: &FlatMatrix<F>,
    opening_point: &RingOpeningPoint<F>,
    challenges: &[SparseChallenge],
    v: &[CyclotomicRing<F, D>],
    u: &[CyclotomicRing<F, D>],
    y_eval: &CyclotomicRing<F, D>,
    level_params: &crate::protocol::commitment::HachiLevelParams,
    layout: HachiCommitmentLayout,
) -> Result<Vec<LabradorConstraint<F, D>>, HachiError>
where
    F: FieldCore + CanonicalField,
{
    let depth_open = layout.num_digits_open;
    let depth_commit = layout.num_digits_commit;
    let depth_fold = layout.num_digits_fold;
    let log_basis = layout.log_basis;
    let num_blocks = opening_point.b.len();
    let block_len = layout.block_len;
    let inner_width = block_len * depth_commit;

    let w_len = depth_open * num_blocks;
    let t_len = depth_open * level_params.n_a * num_blocks;
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

    let mut constraints =
        Vec::with_capacity(level_params.n_d + level_params.n_b + 2 + level_params.n_a);

    // D rows enforce `D_mat * w_hat_flat = v`.
    let d_view = d_mat.view::<D>();
    for (i, &v_i) in v.iter().enumerate().take(level_params.n_d) {
        let d_row = d_view.row(i);
        let coeffs: Vec<CyclotomicRing<F, D>> = d_row.iter().take(w_len).copied().collect();
        constraints.push(LabradorConstraint::new(
            vec![LabradorConstraintTerm::new(0, 0, coeffs)],
            v_i,
        ));
    }

    // B rows enforce `B_mat * t_hat_flat = u`.
    let b_view = b_mat.view::<D>();
    for (i, &u_i) in u.iter().enumerate().take(level_params.n_b) {
        let b_row = b_view.row(i);
        let coeffs: Vec<CyclotomicRing<F, D>> = b_row.iter().take(t_len).copied().collect();
        constraints.push(LabradorConstraint::new(
            vec![LabradorConstraintTerm::new(1, 0, coeffs)],
            u_i,
        ));
    }

    // This row enforces the opening evaluation claim.
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

    // This row ties the folded witness to the pre-handoff `z` decomposition.
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

    // A rows link the folded inner openings back to the same `z` decomposition.
    let a_view = a_mat.view::<D>();
    for a_idx in 0..level_params.n_a {
        let mut phi_t = vec![CyclotomicRing::<F, D>::zero(); t_len];
        for (i, c_i) in dense_challenges.iter().enumerate() {
            for (d, &g) in g_open.iter().enumerate() {
                let t_idx = i * (level_params.n_a * depth_open) + a_idx * depth_open + d;
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
#[tracing::instrument(skip_all, name = "labrador::handoff_build_witness")]
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

/// Estimate the full Labrador recursive proof for the Hachi handoff witness.
#[tracing::instrument(skip_all, name = "labrador::handoff_estimate")]
pub(crate) fn hachi_labrador_estimate<
    F: FieldCore + CanonicalField + HachiSerialize,
    const D: usize,
>(
    witness: &LabradorWitness<F, D>,
    coeff_bit_bound: usize,
) -> Result<LabradorRecursiveSizeEstimate, HachiError> {
    estimate_handoff_recursive_proof::<F, D>(witness, coeff_bit_bound)
}

/// Execute the Labrador direct handoff from the Hachi folding loop.
///
/// Instead of computing the quotient `r`, evaluating at alpha, and running
/// sumcheck, this function runs the quadratic equation at D' and hands the
/// ring-level `Mz = y` directly to Labrador.
///
/// # Errors
///
/// Propagates errors from the quad eq, Labrador config selection, or Labrador proving.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "labrador::handoff_prove")]
pub(crate) fn labrador_handoff_prove<F, T, const D_HANDOFF: usize, Cfg>(
    current_w: &[i8],
    current_hint: &HachiCommitmentHint<F, D_HANDOFF>,
    current_commitment: &RingCommitment<F, D_HANDOFF>,
    current_challenges: &[F],
    current_num_u: usize,
    current_num_l: usize,
    expanded_setup: &HachiExpandedSetup<F>,
    ntt_d: &NttSlotCache<D_HANDOFF>,
    level_params: &HachiLevelParams,
    w_layout: HachiCommitmentLayout,
    transcript: &mut T,
) -> Result<HachiProofTail<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + FromSmallInt + Valid,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
{
    let t0 = Instant::now();
    let mut handoff_transcript = transcript.clone();

    if level_params.d != D_HANDOFF {
        return Err(HachiError::InvalidInput(format!(
            "handoff params D={} mismatch witness commitment D={D_HANDOFF}",
            level_params.d
        )));
    }

    let opening_point = tracing::info_span!("labrador::handoff_prepare_opening_point")
        .in_scope(|| next_level_opening_point(current_challenges, current_num_u, current_num_l));

    let alpha = D_HANDOFF.trailing_zeros() as usize;
    if opening_point.len() < alpha {
        return Err(HachiError::InvalidPointDimension {
            expected: alpha,
            actual: opening_point.len(),
        });
    }

    let direct_tail = PackedDigits::from_i8_digits_with_min_bits(current_w, w_layout.log_basis);
    let direct_hachi_tail_bytes = direct_tail.serialized_size(Compress::No);
    let target_num_vars = w_layout.m_vars + w_layout.r_vars + alpha;
    let mut padded_point = opening_point.clone();
    padded_point.resize(target_num_vars, F::zero());
    let outer_point = &padded_point[alpha..];

    let ring_opening_point =
        tracing::info_span!("labrador::handoff_ring_opening_point").in_scope(|| {
            ring_opening_point_from_field::<F>(
                outer_point,
                w_layout.r_vars,
                w_layout.m_vars,
                BasisMode::Lagrange,
            )
        })?;

    let a_flat = &expanded_setup.A;
    let b_flat = &expanded_setup.B;
    let d_flat = &expanded_setup.D_mat;

    let (w_poly, y_ring, w_folded) = tracing::info_span!("labrador::handoff_fold_witness")
        .in_scope(|| {
            let w_poly = BalancedDigitPoly::<F, D_HANDOFF>::from_i8_digits(current_w)?;
            let (y_ring, w_folded) = w_poly.evaluate_and_fold(
                &ring_opening_point.b,
                &ring_opening_point.a,
                w_layout.block_len,
            );
            Ok::<_, HachiError>((w_poly, y_ring, w_folded))
        })?;

    tracing::info_span!("labrador::handoff_absorb_claims").in_scope(|| {
        current_commitment.append_to_transcript(ABSORB_COMMITMENT, &mut handoff_transcript);
        for pt in &padded_point {
            handoff_transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        handoff_transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);
    });

    let quad_eq = tracing::info_span!("labrador::handoff_quad_eq").in_scope(|| {
        Ok::<_, HachiError>(Box::new(QuadraticEquation::<
            F,
            D_HANDOFF,
            WCommitmentConfig<D_HANDOFF, Cfg>,
        >::new_prover(
            ntt_d,
            ring_opening_point.clone(),
            &w_poly,
            w_folded,
            level_params.clone(),
            current_hint.clone(),
            &mut handoff_transcript,
            current_commitment,
            &y_ring,
            w_layout,
        )?))
    })?;

    tracing::debug!(
        elapsed_s = t0.elapsed().as_secs_f64(),
        "labrador_handoff quad_eq"
    );

    let t1 = Instant::now();

    let w_hat_flat = quad_eq
        .w_hat_flat()
        .ok_or_else(|| HachiError::InvalidInput("missing w_hat_flat".into()))?;
    let inner_opening_digits = &quad_eq
        .hint()
        .ok_or_else(|| HachiError::InvalidInput("missing hint".into()))?
        .inner_opening_digits;
    let inner_opening_digits_flat = flatten_i8_blocks(inner_opening_digits);
    let z_pre = quad_eq
        .z_pre()
        .ok_or_else(|| HachiError::InvalidInput("missing z_pre".into()))?;

    let constraints = build_hachi_labrador_constraints::<F, D_HANDOFF>(
        a_flat,
        b_flat,
        d_flat,
        quad_eq.opening_point(),
        &quad_eq.challenges,
        &quad_eq.v,
        &current_commitment.u,
        &y_ring,
        level_params,
        w_layout,
    )?;

    let witness = build_labrador_witness(w_hat_flat, &inner_opening_digits_flat, z_pre, w_layout);
    let witness_norm_bound_sq = witness.norm();

    let estimate = hachi_labrador_estimate::<F, D_HANDOFF>(&witness, w_layout.log_basis as usize)?;
    let plan = estimate.initial_plan.clone();
    let cfg = plan.config;
    let handoff_row_lengths: Vec<usize> = witness.rows().iter().map(|row| row.len()).collect();
    let handoff_ring_elems: usize = handoff_row_lengths.iter().sum();
    let handoff_witness_bits = handoff_ring_elems * D_HANDOFF * logq_bits::<F>();
    let handoff_witness_bytes =
        FlatLabradorWitness::from_typed(&witness).serialized_size(Compress::No);

    let statement = LabradorStatement {
        inner_opening_payload: Vec::new(),
        linear_garbage_payload: Vec::new(),
        challenges: Vec::new(),
        constraints,
        reduced_constraints: None,
        witness_norm_bound_sq,
    };

    let comkey_seed = expanded_setup.labrador_comkey_seed();

    tracing::debug!(
        digits = current_w.len(),
        log_basis = w_layout.log_basis,
        raw_i8_bytes = current_w.len(),
        packed_direct_bytes = direct_hachi_tail_bytes,
        row_count = witness.rows().len(),
        ?handoff_row_lengths,
        total_ring_elems = handoff_ring_elems,
        witness_bits = handoff_witness_bits,
        serialized_bytes = handoff_witness_bytes,
        witness_norm_bound_sq = %witness_norm_bound_sq,
        max_row_len = handoff_row_lengths.iter().copied().max().unwrap_or(0),
        virtual_row_len = plan.virtual_row_len,
        row_split_counts = ?plan.row_split_counts,
        witness_digit_parts = cfg.witness_digit_parts,
        witness_digit_bits = cfg.witness_digit_bits,
        aux_digit_parts = cfg.aux_digit_parts,
        aux_digit_bits = cfg.aux_digit_bits,
        inner_commit_rank = cfg.inner_commit_rank,
        outer_commit_rank = cfg.outer_commit_rank,
        tail = cfg.tail,
        estimated_labrador_levels = estimate.level_count,
        estimated_labrador_proof_bytes = estimate.proof_bytes,
        estimated_labrador_final_witness_bytes = estimate.final_witness_bytes,
        elapsed_s = t1.elapsed().as_secs_f64(),
        rows = witness.rows().len(),
        constraint_count = statement.constraints.len(),
        "labrador_handoff witness/constraints"
    );

    let v_bytes = FlatRingVec::from_ring_elems(&quad_eq.v).serialized_size(Compress::No);
    let y_ring_bytes = FlatRingVec::from_single(&y_ring).serialized_size(Compress::No);
    let estimated_labrador_tail_bytes = estimate.proof_bytes
        + v_bytes
        + y_ring_bytes
        + witness_norm_bound_sq.serialized_size(Compress::No);
    tracing::info!(
        packed_direct_bytes = direct_hachi_tail_bytes,
        estimated_labrador_tail_bytes,
        selected_tail = if estimated_labrador_tail_bytes < direct_hachi_tail_bytes {
            "labrador"
        } else {
            "direct"
        },
        estimated_labrador_proof_bytes = estimate.proof_bytes,
        v_bytes,
        y_ring_bytes,
        witness_norm_bound_sq_bytes = witness_norm_bound_sq.serialized_size(Compress::No),
        "labrador_handoff estimated tail comparison"
    );
    if estimated_labrador_tail_bytes >= direct_hachi_tail_bytes {
        return Ok(HachiProofTail::Direct(direct_tail));
    }

    let t2 = Instant::now();
    let labrador_proof = prove_with_plan::<F, T, D_HANDOFF>(
        witness,
        &statement,
        &plan,
        &comkey_seed,
        &mut handoff_transcript,
    )?;
    #[cfg(debug_assertions)]
    {
        let roundtrip = FlatLabradorProof::from_typed(&labrador_proof).to_typed::<D_HANDOFF>();
        assert!(
            roundtrip == labrador_proof,
            "labrador handoff proof roundtrip must preserve the proof"
        );

        let mut self_verify_transcript = handoff_transcript.clone();
        verify_labrador::<F, T, D_HANDOFF>(
            &statement,
            &labrador_proof,
            &comkey_seed,
            &mut self_verify_transcript,
        )
        .expect("freshly generated Labrador handoff proof must verify");
    }
    *transcript = handoff_transcript;

    tracing::info!(
        elapsed_s = t2.elapsed().as_secs_f64(),
        levels = labrador_proof.levels.len(),
        "labrador prove complete"
    );

    Ok(HachiProofTail::Labrador(Box::new(LabradorTail {
        labrador_proof: FlatLabradorProof::from_typed(&labrador_proof),
        v: FlatRingVec::from_ring_elems(&quad_eq.v),
        y_ring: FlatRingVec::from_single(&y_ring),
        witness_norm_bound_sq,
    })))
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
#[tracing::instrument(skip_all, name = "labrador::handoff_verify")]
pub(crate) fn labrador_handoff_verify<F, T, const D_HANDOFF: usize, Cfg>(
    tail: &LabradorTail<F>,
    opening_point: &[F],
    opening_value: &F,
    current_commitment: &RingCommitment<F, D_HANDOFF>,
    expanded_setup: &HachiExpandedSetup<F>,
    level_params: &HachiLevelParams,
    w_layout: HachiCommitmentLayout,
    transcript: &mut T,
) -> Result<(), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + FromSmallInt + Valid,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
{
    let t0 = Instant::now();
    if level_params.d != D_HANDOFF {
        return Err(HachiError::InvalidProof);
    }
    let alpha_prime = D_HANDOFF.trailing_zeros() as usize;
    if opening_point.len() < alpha_prime {
        return Err(HachiError::InvalidPointDimension {
            expected: alpha_prime,
            actual: opening_point.len(),
        });
    }

    let v: Vec<CyclotomicRing<F, D_HANDOFF>> = tail.v.to_vec();
    let y_ring: CyclotomicRing<F, D_HANDOFF> = tail.y_ring.to_single();
    let labrador_proof = tail.labrador_proof.to_typed::<D_HANDOFF>();

    if !tracing::info_span!("labrador::handoff_match_opening_claim")
        .in_scope(|| matches_opening_claim::<F, D_HANDOFF>(&y_ring, opening_point, opening_value))
    {
        return Err(HachiError::InvalidProof);
    }

    let target_num_vars = w_layout.m_vars + w_layout.r_vars + alpha_prime;
    let mut padded_point = opening_point.to_vec();
    padded_point.resize(target_num_vars, F::zero());
    let outer_point = &padded_point[alpha_prime..];

    let ring_opening_point =
        tracing::info_span!("labrador::handoff_ring_opening_point").in_scope(|| {
            ring_opening_point_from_field::<F>(
                outer_point,
                w_layout.r_vars,
                w_layout.m_vars,
                BasisMode::Lagrange,
            )
        })?;

    // Replay transcript against the carried Hachi commitment.
    tracing::info_span!("labrador::handoff_absorb_claims").in_scope(|| {
        current_commitment.append_to_transcript(ABSORB_COMMITMENT, transcript);
        for pt in &padded_point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);
    });

    // Derive challenges via verifier-side quad eq (absorbs v, samples challenges).
    let quad_eq = tracing::info_span!("labrador::handoff_quad_eq").in_scope(|| {
        QuadraticEquation::<F, D_HANDOFF, WCommitmentConfig<D_HANDOFF, Cfg>>::new_verifier(
            ring_opening_point.clone(),
            v.clone(),
            level_params.clone(),
            transcript,
            current_commitment,
            &y_ring,
            w_layout,
        )
    })?;

    let a_flat = &expanded_setup.A;
    let b_flat = &expanded_setup.B;
    let d_flat = &expanded_setup.D_mat;

    // Rebuild constraints from public data.
    let constraints = build_hachi_labrador_constraints::<F, D_HANDOFF>(
        a_flat,
        b_flat,
        d_flat,
        quad_eq.opening_point(),
        &quad_eq.challenges,
        &v,
        &current_commitment.u,
        &y_ring,
        level_params,
        w_layout,
    )?;

    let statement = LabradorStatement {
        inner_opening_payload: Vec::new(),
        linear_garbage_payload: Vec::new(),
        challenges: Vec::new(),
        constraints,
        reduced_constraints: None,
        witness_norm_bound_sq: tail.witness_norm_bound_sq,
    };

    let comkey_seed = expanded_setup.labrador_comkey_seed();

    let result =
        verify_labrador::<F, T, D_HANDOFF>(&statement, &labrador_proof, &comkey_seed, transcript);
    if result.is_ok() {
        tracing::info!(
            elapsed_s = t0.elapsed().as_secs_f64(),
            levels = labrador_proof.levels.len(),
            "labrador verify complete"
        );
    } else {
        tracing::error!(
            elapsed_s = t0.elapsed().as_secs_f64(),
            levels = labrador_proof.levels.len(),
            "labrador verify FAIL"
        );
    }
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

fn matches_opening_claim<F: FieldCore + CanonicalField, const D: usize>(
    y_ring: &CyclotomicRing<F, D>,
    opening_point: &[F],
    opening_value: &F,
) -> bool {
    let alpha = D.trailing_zeros() as usize;
    let coeff_point = &opening_point[..alpha];
    let mut coeff_basis = vec![F::zero(); D];
    multilinear_lagrange_basis(&mut coeff_basis, coeff_point);
    let inner_ring = CyclotomicRing::from_slice(&coeff_basis);
    let d = F::from_u64(D as u64);
    let trace_lhs = (*y_ring * inner_ring.sigma_m1()).coefficients()[0] * d;
    trace_lhs == d * *opening_value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::commitment::utils::linear::flatten_i8_blocks;
    use crate::protocol::commitment::{
        HachiCommitmentCore, HachiScheduleInputs, RingCommitmentScheme,
    };
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

        let point = RingOpeningPoint {
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
        let level_params = TinyConfig::level_params(HachiScheduleInputs {
            max_num_vars: setup.expanded.seed.max_num_vars,
            level: 0,
            current_w_len: layout.num_blocks * layout.block_len * D,
        });

        let quad_eq = QuadraticEquation::<F, D, TinyConfig>::new_prover(
            &setup.ntt_D,
            point.clone(),
            &poly,
            w_folded,
            level_params.clone(),
            hint,
            &mut transcript,
            &w.commitment,
            &y_ring,
            layout,
        )
        .unwrap();

        let w_hat_flat = quad_eq.w_hat_flat().unwrap();
        let inner_opening_digits = &quad_eq.hint().unwrap().inner_opening_digits;
        let inner_opening_digits_flat = flatten_i8_blocks(inner_opening_digits);
        let z_pre = quad_eq.z_pre().unwrap();

        let constraints = build_hachi_labrador_constraints::<F, D>(
            &setup.expanded.A,
            &setup.expanded.B,
            &setup.expanded.D_mat,
            quad_eq.opening_point(),
            &quad_eq.challenges,
            &quad_eq.v,
            &w.commitment.u,
            &y_ring,
            &level_params,
            layout,
        )
        .unwrap();

        let witness = build_labrador_witness(w_hat_flat, &inner_opening_digits_flat, z_pre, layout);

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
