//! Ring-relation prover for the Akita PCS (§4.2).
//!
//! Builds the stage-1 relation instance and witness (`M`, `y`, `z`, `v`) via
//! [`RingRelationProver`].
#[cfg(feature = "zk")]
use crate::protocol::masking::sample_blinding_digits;
use crate::validation::validate_i8_setup_log_basis;
use crate::{
    AkitaPolyOps, CyclicRowsComputeBackend, DecomposeFoldWitness, DigitRowsComputeBackend,
    RecursiveWitnessView, RingSwitchComputeBackend, RingSwitchQuotientRowsPlan,
    RingSwitchRelationRowsPlan,
};
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2I8Params;
use akita_algebra::CyclotomicRing;
use akita_challenges::{
    sample_folding_challenges, stage1_fold_challenge_labels, ChallengeShape, Challenges,
    IntegerChallenge, SparseChallenge,
};
use akita_field::parallel::*;
use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore, FromPrimitiveInt, HalvingField};
use akita_transcript::labels::{ABSORB_PROVER_V, ABSORB_TERMINAL_W_HAT};
use akita_transcript::Transcript;
use akita_types::{
    gadget_row_scalars, terminal_w_hat_bytes_from_blocks, AkitaCommitmentHint, FlatDigitBlocks,
    MRowLayout, RingCommitment, RingSliceSerializer,
};
use akita_types::{ClaimIncidenceSummary, CommitmentRouting, LevelParams, RingRelationInstance};
use akita_types::{RingMultiplierOpeningPoint, RingOpeningPoint};

use super::ring_relation_witness::RingRelationWitness;
use std::iter::repeat_n;
use std::time::Instant;

mod relation_quotient;
mod repeated_b;

pub use relation_quotient::{compute_relation_quotient, generate_y};

fn beta_linf_fold_bound(
    r: usize,
    challenge_l1_mass: usize,
    log_basis: u32,
) -> Result<u128, AkitaError> {
    if !(1..128).contains(&log_basis) {
        return Err(AkitaError::InvalidSetup("invalid LOG_BASIS".to_string()));
    }
    if r >= 128 {
        return Err(AkitaError::InvalidSetup("r_vars must be < 128".to_string()));
    }
    let blocks = 1u128 << r;
    let b = 1u128 << log_basis;
    let half_b = b / 2;
    let term = blocks
        .checked_mul(challenge_l1_mass as u128)
        .ok_or_else(|| AkitaError::InvalidSetup("beta bound overflow".to_string()))?;
    term.checked_mul(half_b)
        .ok_or_else(|| AkitaError::InvalidSetup("beta bound overflow".to_string()))
}
fn beta_linf_fold_bound_with_num_claims(
    r: usize,
    challenge_l1_mass: usize,
    log_basis: u32,
    num_claims: usize,
) -> Result<u128, AkitaError> {
    let beta = beta_linf_fold_bound(r, challenge_l1_mass, log_basis)?;
    beta.checked_mul(num_claims as u128)
        .ok_or_else(|| AkitaError::InvalidSetup("batched beta bound overflow".to_string()))
}

fn validate_decompose_fold<F: FieldCore + CanonicalField, const D: usize>(
    z: DecomposeFoldWitness<F, D>,
    lp: &LevelParams,
    num_claims: usize,
) -> Result<DecomposeFoldWitness<F, D>, AkitaError> {
    let norm = u128::from(z.centered_inf_norm);
    let beta = beta_linf_fold_bound_with_num_claims(
        lp.r_vars,
        lp.challenge_l1_mass(),
        lp.log_basis,
        num_claims,
    )?;
    if norm > beta {
        return Err(AkitaError::InvalidInput(format!(
            "prover abort: ||z||_inf = {norm} > beta = {beta}"
        )));
    }
    Ok(z)
}

fn absorb_terminal_w_hat<F, T, const D: usize>(
    transcript: &mut T,
    w_hat: &FlatDigitBlocks<D>,
    planes_per_block: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let bytes = terminal_w_hat_bytes_from_blocks(w_hat, planes_per_block)?;
    transcript.append_bytes(ABSORB_TERMINAL_W_HAT, &bytes);
    Ok(())
}

fn aggregate_decompose_fold_witnesses<F: FieldCore, const D: usize>(
    witnesses: Vec<DecomposeFoldWitness<F, D>>,
) -> Result<DecomposeFoldWitness<F, D>, AkitaError> {
    let Some((first, rest)) = witnesses.split_first() else {
        return Err(AkitaError::InvalidInput(
            "batched decompose_fold requires at least one witness".to_string(),
        ));
    };
    let z_len = first.z_folded_rings.len();
    let coeff_len = first.centered_coeffs.len();
    let mut z_folded_rings = first.z_folded_rings.clone();
    let mut centered_coeffs = first.centered_coeffs.clone();

    for witness in rest {
        if witness.z_folded_rings.len() != z_len || witness.centered_coeffs.len() != coeff_len {
            return Err(AkitaError::InvalidInput(
                "batched decompose_fold witness length mismatch".to_string(),
            ));
        }
        for (dst, src) in z_folded_rings.iter_mut().zip(witness.z_folded_rings.iter()) {
            *dst += *src;
        }
        for (dst, src) in centered_coeffs
            .iter_mut()
            .zip(witness.centered_coeffs.iter())
        {
            for k in 0..D {
                dst[k] = dst[k].checked_add(src[k]).ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "batched decompose_fold centered coefficient overflow".to_string(),
                    )
                })?;
            }
        }
    }

    let centered_inf_norm = centered_coeffs
        .iter()
        .flat_map(|coeffs| coeffs.iter())
        .map(|coeff| coeff.unsigned_abs())
        .max()
        .unwrap_or(0);

    Ok(DecomposeFoldWitness {
        z_folded_rings,
        centered_coeffs,
        centered_inf_norm,
    })
}

fn build_point_decompose_fold_witness<F, P, const D: usize>(
    challenges: &Challenges,
    point_polys: &[&P],
    point_indices: &[usize],
    lp: &LevelParams,
) -> Result<DecomposeFoldWitness<F, D>, AkitaError>
where
    F: FieldCore,
    P: AkitaPolyOps<F, D>,
{
    match challenges {
        Challenges::Sparse {
            challenges: sparse,
            num_blocks_per_claim,
            ..
        } => {
            let mut point_challenges =
                Vec::with_capacity(point_indices.len() * *num_blocks_per_claim);
            for &claim_idx in point_indices {
                let start = claim_idx
                    .checked_mul(*num_blocks_per_claim)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("batched challenge offset overflow".to_string())
                    })?;
                let end = start.checked_add(*num_blocks_per_claim).ok_or_else(|| {
                    AkitaError::InvalidSetup("batched challenge offset overflow".to_string())
                })?;
                point_challenges.extend_from_slice(sparse.get(start..end).ok_or(
                    AkitaError::InvalidSize {
                        expected: end,
                        actual: sparse.len(),
                    },
                )?);
            }
            if let Some(z_point) = P::decompose_fold_batched(
                point_polys,
                &point_challenges,
                lp.block_len,
                lp.num_digits_commit,
                lp.log_basis,
            ) {
                Ok(z_point)
            } else {
                let witnesses: Vec<DecomposeFoldWitness<F, D>> = point_polys
                    .iter()
                    .zip(point_challenges.chunks(*num_blocks_per_claim))
                    .map(|(poly, poly_challenges)| {
                        poly.decompose_fold(
                            poly_challenges,
                            lp.block_len,
                            lp.num_digits_commit,
                            lp.log_basis,
                        )
                    })
                    .collect();
                aggregate_decompose_fold_witnesses(witnesses)
            }
        }
        Challenges::Tensor { factored: _ } => {
            let selected = challenges.select_claims::<D>(point_indices)?;
            let point_factored = match selected {
                Challenges::Tensor { factored } => factored,
                Challenges::Sparse { .. } => {
                    return Err(AkitaError::InvalidSetup(
                        "tensor claim selection returned sparse challenges".to_string(),
                    ))
                }
            };
            match P::decompose_fold_tensor_batched(
                point_polys,
                &point_factored,
                lp.block_len,
                lp.num_digits_commit,
                lp.log_basis,
            )? {
                Some(witness) => Ok(witness),
                None => Err(AkitaError::InvalidSetup(
                    "polynomial backend has no tensor-shaped fold kernel".to_string(),
                )),
            }
        }
    }
}

/// Compute the D-side relation rows `v = D · w_hat` (plus ZK blinding when enabled).
fn compute_v_rows<F, B, const D: usize>(
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    row_len: usize,
    w_hat: &FlatDigitBlocks<D>,
    log_basis: u32,
    #[cfg(feature = "zk")] d_blinding_digits: &FlatDigitBlocks<D>,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    B: DigitRowsComputeBackend<F>,
{
    #[cfg(feature = "zk")]
    {
        let mut rows =
            backend.digit_rows::<D>(prepared, row_len, w_hat.flat_digits(), log_basis)?;
        let blinding_rows = backend.zk_d_digit_rows::<D>(
            prepared,
            row_len,
            d_blinding_digits.flat_digits().len(),
            d_blinding_digits.flat_digits(),
        )?;
        for (row, blinding) in rows.iter_mut().zip(blinding_rows) {
            *row += blinding;
        }
        if rows.len() != row_len {
            return Err(AkitaError::InvalidProof);
        }
        Ok(rows)
    }
    #[cfg(not(feature = "zk"))]
    {
        let rows = backend.digit_rows::<D>(prepared, row_len, w_hat.flat_digits(), log_basis)?;
        if rows.len() != row_len {
            return Err(AkitaError::InvalidProof);
        }
        Ok(rows)
    }
}

/// Prover-side builder for the ring relation $M(x) \cdot z = y(x) + (X^D + 1) \cdot r(x)$.
pub struct RingRelationProver;

impl RingRelationProver {
    /// Root-level constructor covering single-claim, same-point batching,
    /// multi-point batching, or any mix.
    ///
    /// `opening_points` holds the distinct ring-level opening points used by
    /// the batch; `claim_to_point` maps each flattened claim to its
    /// opening-point index.  The batched CWSS protocol γ-combines all
    /// polynomials opened at the same point into one ring element, so
    /// `y_rings` carries one entry per opening point
    /// (i.e. `y_rings.len() == opening_points.len()`).  For the trivial
    /// single-claim case use `opening_points = vec![pt]`,
    /// `claim_to_point = vec![0]`, `polys = &[poly]`,
    /// `num_polys_per_point = &[1]`, `gamma = vec![F::one()]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the batched hints, folded witnesses, or decomposed
    /// aggregate witness are malformed.
    ///
    /// # Panics
    ///
    /// Panics if the batched `w_hat` decomposition or flattened batched hints
    /// produced by the prover do not preserve the expected block sizes.  These
    /// invariants hold by construction for well-formed inputs accepted by the
    /// error checks above and are therefore treated as internal programming
    /// errors rather than recoverable failures.
    #[allow(clippy::too_many_arguments, clippy::new_ret_no_self)]
    #[tracing::instrument(skip_all, name = "RingRelationProver::new")]
    #[inline(never)]
    pub fn new<F, const D: usize, T, P, B>(
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        opening_points: Vec<RingOpeningPoint<F>>,
        ring_multiplier_points: Vec<RingMultiplierOpeningPoint<F, D>>,
        claim_to_point: Vec<usize>,
        polys: &[&P],
        pre_folded_by_poly: Vec<Vec<CyclotomicRing<F, D>>>,
        incidence_summary: &ClaimIncidenceSummary,
        lp: LevelParams,
        hints: Vec<AkitaCommitmentHint<F, D>>,
        transcript: &mut T,
        commitments: &[RingCommitment<F, D>],
        y_rings: &[CyclotomicRing<F, D>],
        row_coefficient_rings: Vec<CyclotomicRing<F, D>>,
        m_row_layout: MRowLayout,
    ) -> Result<(RingRelationInstance<F, D>, RingRelationWitness<F, D>), AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
        P: AkitaPolyOps<F, D>,
        B: DigitRowsComputeBackend<F>,
    {
        {
            let x: u8 = 0;
            tracing::trace!(
                stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
                "RingRelationProver::new"
            );
        }
        validate_i8_setup_log_basis(lp.log_basis, "for i8 prover decomposition")?;
        if opening_points.is_empty() {
            return Err(AkitaError::InvalidInput(
                "batched prover requires at least one opening point".to_string(),
            ));
        }
        for opening_point in &opening_points {
            if opening_point.a.len() < lp.block_len || opening_point.b.len() != lp.num_blocks {
                return Err(AkitaError::InvalidInput(
                    "batched prover opening-point layout mismatch".to_string(),
                ));
            }
        }
        if ring_multiplier_points.len() != opening_points.len()
            || ring_multiplier_points
                .iter()
                .any(|point| point.a_len() < lp.block_len || point.b_len() != lp.num_blocks)
        {
            return Err(AkitaError::InvalidInput(
                "batched prover ring-multiplier opening-point layout mismatch".to_string(),
            ));
        }
        let num_claims = incidence_summary.num_claims();
        let num_polys_per_point = &incidence_summary.num_polys_per_point();
        if polys.is_empty() || num_polys_per_point.is_empty() {
            return Err(AkitaError::InvalidInput(
                "batched prover requires at least one polynomial".to_string(),
            ));
        }
        if num_polys_per_point.contains(&0) {
            return Err(AkitaError::InvalidInput(
                "batched prover requires at least one polynomial per opening point".to_string(),
            ));
        }
        // The batched protocol emits one public y-row per packaged public row,
        // so `y_rings.len()` must equal `opening_points.len()`.
        if polys.len() != pre_folded_by_poly.len()
            || polys.len() != num_claims
            || y_rings.len() != opening_points.len()
            || claim_to_point.len() != num_claims
            || incidence_summary.claim_to_point().len() != num_claims
            || incidence_summary.claim_poly_indices().len() != num_claims
            || hints.len() != incidence_summary.num_points()
            || commitments.len() != incidence_summary.num_points()
        {
            return Err(AkitaError::InvalidInput(
                "batched prover input lengths do not match".to_string(),
            ));
        }
        if claim_to_point
            .iter()
            .any(|&point_idx| point_idx >= opening_points.len())
        {
            return Err(AkitaError::InvalidInput(
                "batched prover claim-to-point index out of range".to_string(),
            ));
        }
        for claim_idx in 0..num_claims {
            let point_idx = incidence_summary.claim_to_point()[claim_idx];
            if point_idx >= incidence_summary.num_points()
                || incidence_summary.claim_poly_indices()[claim_idx]
                    >= num_polys_per_point[point_idx]
            {
                return Err(AkitaError::InvalidInput(
                    "batched prover claim incidence index out of range".to_string(),
                ));
            }
        }
        for commitment in commitments {
            if commitment.u.len() != lp.b_key.row_len() {
                return Err(AkitaError::InvalidInput(
                    "batched prover received a commitment with the wrong length".to_string(),
                ));
            }
        }
        if row_coefficient_rings.len() != num_claims {
            return Err(AkitaError::InvalidInput(
                "batched prover row coefficient length does not match claim count".to_string(),
            ));
        }
        let gamma = row_coefficient_rings
            .iter()
            .map(|ring| ring.coefficients()[0])
            .collect::<Vec<_>>();

        let w_hat = {
            let _span = tracing::info_span!("decompose_batched_w_hat").entered();
            let depth_open = lp.num_digits_open;
            let log_basis = lp.log_basis;
            let q = (-F::one()).to_canonical_u128() + 1;
            let decompose_params = BalancedDecomposePow2I8Params::new(depth_open, log_basis, q);
            let total_rows: usize = pre_folded_by_poly.iter().map(Vec::len).sum();
            let block_sizes = vec![depth_open; total_rows];
            let mut w_hat = FlatDigitBlocks::zeroed(block_sizes)
                .expect("batched w_hat decomposition preserves block sizes");
            let mut offset = 0usize;
            for folded_rows in &pre_folded_by_poly {
                for w_i in folded_rows {
                    w_i.balanced_decompose_pow2_i8_into_with_params(
                        &mut w_hat.flat_digits_mut()[offset..offset + depth_open],
                        &decompose_params,
                    );
                    offset += depth_open;
                }
            }
            w_hat
        };
        let flattened_hint = {
            let mut decomposed_inner_rows = Vec::new();
            let mut t_rows_by_poly = Vec::new();
            #[cfg(feature = "zk")]
            let mut b_blinding_digits = Vec::new();
            for (mut hint, &group_size) in hints.into_iter().zip(num_polys_per_point.iter()) {
                if hint.decomposed_inner_rows.len() != group_size {
                    return Err(AkitaError::InvalidInput(
                        "batched prover hint group sizes do not match polynomial groups"
                            .to_string(),
                    ));
                }
                hint.ensure_recomposed_inner_rows(lp.num_digits_open, lp.log_basis)?;
                #[cfg(feature = "zk")]
                let (digits_by_poly, rows_by_poly, mut blinding_by_group) = hint.into_parts();
                #[cfg(not(feature = "zk"))]
                let (digits_by_poly, rows_by_poly) = hint.into_parts();
                #[cfg(feature = "zk")]
                if blinding_by_group.len() != 1 {
                    return Err(AkitaError::InvalidInput(
                        "batched prover hint must carry exactly one blinding group per commitment"
                            .to_string(),
                    ));
                }
                decomposed_inner_rows.extend(digits_by_poly);
                let rows_by_poly = rows_by_poly.ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "missing recomposed inner rows in batched prover hint".to_string(),
                    )
                })?;
                t_rows_by_poly.extend(rows_by_poly);
                #[cfg(feature = "zk")]
                b_blinding_digits.append(&mut blinding_by_group);
            }
            #[cfg(feature = "zk")]
            {
                AkitaCommitmentHint::with_recomposed_inner_rows(
                    decomposed_inner_rows,
                    t_rows_by_poly,
                    b_blinding_digits,
                )
            }
            #[cfg(not(feature = "zk"))]
            {
                AkitaCommitmentHint::with_recomposed_inner_rows(
                    decomposed_inner_rows,
                    t_rows_by_poly,
                )
            }
        };

        // Terminal layout drops the D-block from the M-matrix entirely:
        // `v = D · w_hat` never travels on the wire, the verifier never
        // reconstructs it, and downstream prover paths (`ring_switch_build_w`,
        // `relation_claim_from_rows_extension`) consume an empty `v` slice.
        // Skip both the D-side blinding sample and the D-NTT under Terminal.
        #[cfg(feature = "zk")]
        let d_blinding_digits = match m_row_layout {
            MRowLayout::WithDBlock => {
                sample_blinding_digits::<F, D>(lp.d_key.row_len(), lp.log_basis)?
            }
            MRowLayout::WithoutDBlock => FlatDigitBlocks::<D>::empty(),
        };

        let v = match m_row_layout {
            MRowLayout::WithDBlock => {
                let _span = tracing::info_span!(
                    "compute_batched_v",
                    w_hat_planes = w_hat.flat_digits().len()
                )
                .entered();
                let v = compute_v_rows(
                    backend,
                    prepared,
                    lp.d_key.row_len(),
                    &w_hat,
                    lp.log_basis,
                    #[cfg(feature = "zk")]
                    &d_blinding_digits,
                )?;
                transcript.append_serde(ABSORB_PROVER_V, &RingSliceSerializer(&v));
                v
            }
            MRowLayout::WithoutDBlock => Vec::new(),
        };

        if matches!(m_row_layout, MRowLayout::WithoutDBlock) {
            absorb_terminal_w_hat::<F, T, D>(transcript, &w_hat, lp.num_digits_open)?;
        }
        let challenges = sample_folding_challenges::<F, T, D>(
            transcript,
            lp.num_blocks,
            num_claims,
            &lp.stage1_config,
            &lp.fold_challenge_shape,
            stage1_fold_challenge_labels(),
        )?;

        // Per-point chunking keeps each aggregated witness aligned with one
        // opening point's challenge claims.
        let z_folded_rings = {
            let num_points = opening_points.len();
            let _span =
                tracing::info_span!("compute_batched_z_folded", num_points = num_points).entered();
            let mut polys_by_point: Vec<Vec<&P>> = vec![Vec::new(); num_points];
            let mut claim_indices_by_point: Vec<Vec<usize>> = vec![Vec::new(); num_points];
            for (claim_idx, poly) in polys.iter().enumerate() {
                let point_idx = claim_to_point[claim_idx];
                polys_by_point[point_idx].push(*poly);
                claim_indices_by_point[point_idx].push(claim_idx);
            }

            let mut z_folded_rings = Vec::new();
            let mut centered_coeffs = Vec::new();
            let mut centered_inf_norm = 0u32;
            for (point_idx, point_polys) in polys_by_point.iter().enumerate() {
                let point_indices = &claim_indices_by_point[point_idx];
                let point_claim_count = point_polys.len();
                let witness = build_point_decompose_fold_witness::<F, P, D>(
                    &challenges,
                    point_polys,
                    point_indices,
                    &lp,
                )?;
                let witness = validate_decompose_fold(witness, &lp, point_claim_count)?;
                centered_inf_norm = centered_inf_norm.max(witness.centered_inf_norm);
                z_folded_rings.extend(witness.z_folded_rings);
                centered_coeffs.extend(witness.centered_coeffs);
            }

            DecomposeFoldWitness {
                z_folded_rings,
                centered_coeffs,
                centered_inf_norm,
            }
        };

        let commitment_rows: Vec<CyclotomicRing<F, D>> = commitments
            .iter()
            .flat_map(|commitment| commitment.u.iter().copied())
            .collect();
        // Terminal levels drop the D-block from M entirely, so `y` must
        // also drop the D-rows (the `v = D · ŵ` segment). Pass an empty
        // `v` slice with `n_d_active = 0` so `generate_y` emits
        // `[consistency | public_outputs | commitment_rows | A-zeros]`.
        let (y_v_slice, n_d_active) = match m_row_layout {
            MRowLayout::WithDBlock => (v.as_slice(), lp.d_key.row_len()),
            MRowLayout::WithoutDBlock => (&[][..], 0usize),
        };
        let y = generate_y::<F, D>(
            y_v_slice,
            &commitment_rows,
            y_rings,
            n_d_active,
            lp.b_key.row_len(),
            lp.a_key.row_len(),
        )?;
        let w_folded = pre_folded_by_poly.into_iter().flatten().collect();

        let incidence = incidence_summary.clone();
        let commitment_routing = CommitmentRouting::from_root_incidence(incidence_summary)?;
        let instance = RingRelationInstance::new(
            m_row_layout,
            challenges,
            opening_points,
            ring_multiplier_points,
            incidence,
            commitment_routing,
            gamma,
            row_coefficient_rings,
            y,
            v,
        )?;
        instance.check_v_shape_for_level(&lp)?;
        let witness = RingRelationWitness {
            z_folded_rings,
            w_hat,
            w_folded,
            hint: flattened_hint,
            #[cfg(feature = "zk")]
            d_blinding_digits,
        };
        Ok((instance, witness))
    }

    /// Recursive constructor for one committed witness opened at multiple public rows.
    ///
    /// Specialized to a single committed recursive witness. Each claim opens
    /// the same committed vector at a distinct point; row coefficients are the
    /// identity because there is no same-row polynomial batching.
    ///
    /// # Errors
    ///
    /// Returns an error when the per-claim inputs have inconsistent lengths, the
    /// recursive witness cannot be folded into the requested layout, or the
    /// transcript-derived challenge path rejects the supplied commitment shape.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "RingRelationProver::new_recursive_multipoint")]
    #[inline(never)]
    pub fn new_recursive_multipoint<F, const D: usize, T, B>(
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        ring_opening_points: Vec<RingOpeningPoint<F>>,
        ring_multiplier_points: Vec<RingMultiplierOpeningPoint<F, D>>,
        witness: &RecursiveWitnessView<'_, F, D>,
        pre_folded_by_claim: Vec<Vec<CyclotomicRing<F, D>>>,
        lp: LevelParams,
        mut hint: AkitaCommitmentHint<F, D>,
        transcript: &mut T,
        commitment: &[CyclotomicRing<F, D>],
        y_rings: &[CyclotomicRing<F, D>],
        m_row_layout: MRowLayout,
    ) -> Result<(RingRelationInstance<F, D>, RingRelationWitness<F, D>), AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
        B: DigitRowsComputeBackend<F>,
    {
        validate_i8_setup_log_basis(lp.log_basis, "for i8 prover decomposition")?;
        let num_claims = ring_opening_points.len();
        if num_claims == 0
            || ring_multiplier_points.len() != num_claims
            || pre_folded_by_claim.len() != num_claims
            || y_rings.len() != num_claims
        {
            return Err(AkitaError::InvalidInput(
                "recursive multipoint input lengths do not match".to_string(),
            ));
        }
        for opening_point in &ring_opening_points {
            if opening_point.a.len() < lp.block_len || opening_point.b.len() != lp.num_blocks {
                return Err(AkitaError::InvalidInput(
                    "recursive multipoint opening-point layout mismatch".to_string(),
                ));
            }
        }
        if ring_multiplier_points
            .iter()
            .any(|point| point.a_len() < lp.block_len || point.b_len() != lp.num_blocks)
        {
            return Err(AkitaError::InvalidInput(
                "recursive multipoint ring-multiplier layout mismatch".to_string(),
            ));
        }

        let w_hat = {
            let _span = tracing::info_span!("decompose_recursive_multipoint_w_hat").entered();
            let depth_open = lp.num_digits_open;
            let log_basis = lp.log_basis;
            let q = (-F::one()).to_canonical_u128() + 1;
            let decompose_params = BalancedDecomposePow2I8Params::new(depth_open, log_basis, q);
            let total_rows: usize = pre_folded_by_claim.iter().map(Vec::len).sum();
            let mut w_hat = FlatDigitBlocks::zeroed(vec![depth_open; total_rows])?;
            let mut offset = 0usize;
            for folded_rows in &pre_folded_by_claim {
                for w_i in folded_rows {
                    w_i.balanced_decompose_pow2_i8_into_with_params(
                        &mut w_hat.flat_digits_mut()[offset..offset + depth_open],
                        &decompose_params,
                    );
                    offset += depth_open;
                }
            }
            w_hat
        };
        hint.ensure_recomposed_inner_rows(lp.num_digits_open, lp.log_basis)?;

        // See [`Self::new`]: Terminal layout omits `v = D · w_hat`
        // entirely, so skip both the D-side blinding sample and the D-NTT.
        #[cfg(feature = "zk")]
        let d_blinding_digits = match m_row_layout {
            MRowLayout::WithDBlock => {
                sample_blinding_digits::<F, D>(lp.d_key.row_len(), lp.log_basis)?
            }
            MRowLayout::WithoutDBlock => FlatDigitBlocks::<D>::empty(),
        };

        let v = match m_row_layout {
            MRowLayout::WithDBlock => {
                let _span = tracing::info_span!(
                    "compute_recursive_multipoint_v",
                    w_hat_planes = w_hat.flat_digits().len()
                )
                .entered();
                let v = compute_v_rows(
                    backend,
                    prepared,
                    lp.d_key.row_len(),
                    &w_hat,
                    lp.log_basis,
                    #[cfg(feature = "zk")]
                    &d_blinding_digits,
                )?;
                transcript.append_serde(ABSORB_PROVER_V, &RingSliceSerializer(&v));
                v
            }
            MRowLayout::WithoutDBlock => Vec::new(),
        };

        // Recursive witnesses do not implement a tensor batched fold kernel.
        if !matches!(lp.fold_challenge_shape, ChallengeShape::Flat) {
            return Err(AkitaError::InvalidSetup(
                "tensor fold shape is not supported at recursive levels".to_string(),
            ));
        }
        if matches!(m_row_layout, MRowLayout::WithoutDBlock) {
            absorb_terminal_w_hat::<F, T, D>(transcript, &w_hat, lp.num_digits_open)?;
        }
        let challenges = sample_folding_challenges::<F, T, D>(
            transcript,
            lp.num_blocks,
            num_claims,
            &lp.stage1_config,
            &lp.fold_challenge_shape,
            stage1_fold_challenge_labels(),
        )?;

        // Tensor shapes were rejected above, so the recursive path can use the
        // sparse challenge slice directly.
        let Challenges::Sparse {
            challenges: sparse_challenges,
            ..
        } = &challenges
        else {
            return Err(AkitaError::InvalidSetup(
                "recursive fold sampling returned tensor challenges".to_string(),
            ));
        };
        let z_folded_rings = {
            let _span = tracing::info_span!(
                "compute_recursive_multipoint_z_folded",
                num_claims = num_claims
            )
            .entered();
            let mut z_folded_rings = Vec::new();
            let mut centered_coeffs = Vec::new();
            let mut centered_inf_norm = 0u32;
            for claim_idx in 0..num_claims {
                let challenge_offset = claim_idx.checked_mul(lp.num_blocks).ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "recursive multipoint challenge offset overflow".into(),
                    )
                })?;
                let next_offset = challenge_offset.checked_add(lp.num_blocks).ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "recursive multipoint challenge offset overflow".into(),
                    )
                })?;
                let witness_part = witness.decompose_fold(
                    &sparse_challenges[challenge_offset..next_offset],
                    lp.block_len,
                    lp.num_blocks,
                    lp.num_digits_commit,
                    lp.log_basis,
                );
                let witness_part = validate_decompose_fold(witness_part, &lp, 1)?;
                centered_inf_norm = centered_inf_norm.max(witness_part.centered_inf_norm);
                z_folded_rings.extend(witness_part.z_folded_rings);
                centered_coeffs.extend(witness_part.centered_coeffs);
            }
            DecomposeFoldWitness {
                z_folded_rings,
                centered_coeffs,
                centered_inf_norm,
            }
        };

        let (y_v_slice, n_d_active) = match m_row_layout {
            MRowLayout::WithDBlock => (v.as_slice(), lp.d_key.row_len()),
            MRowLayout::WithoutDBlock => (&[][..], 0usize),
        };
        let y = generate_y::<F, D>(
            y_v_slice,
            commitment,
            y_rings,
            n_d_active,
            lp.b_key.row_len(),
            lp.a_key.row_len(),
        )?;
        let w_folded = pre_folded_by_claim.into_iter().flatten().collect();

        // True recursive multipoint (one commitment opened at k > 1 points) is
        // a deferred feature. The routing types and the row-evaluation on both
        // sides can already express split routing, but the contract is kept
        // single-point because that path is unspecified and unproven, and both
        // callers produce exactly one opening point. `RingRelationInstance::new`
        // enforces the same restriction via `check_matches_incidence`; this
        // guard states it earlier with a clearer message. See
        // specs/core-protocol-naming-cleanup.md (Design > Deferred).
        if num_claims != 1 {
            return Err(AkitaError::InvalidInput(
                "recursive split opening/commitment routing is not supported".to_string(),
            ));
        }
        let num_vars = lp.recursive_opening_num_vars()?;
        let incidence = ClaimIncidenceSummary::from_point_polys(num_vars, vec![1; num_claims])?;
        let commitment_routing = CommitmentRouting::from_recursive_multipoint(num_claims)?;
        let instance = RingRelationInstance::new(
            m_row_layout,
            challenges,
            ring_opening_points,
            ring_multiplier_points,
            incidence,
            commitment_routing,
            vec![F::one(); num_claims],
            vec![CyclotomicRing::one(); num_claims],
            y,
            v,
        )?;
        instance.check_v_shape_for_level(&lp)?;
        let witness = RingRelationWitness {
            z_folded_rings,
            w_hat,
            w_folded,
            hint,
            #[cfg(feature = "zk")]
            d_blinding_digits,
        };
        Ok((instance, witness))
    }
}
