use super::*;
use crate::backend::{RingSwitchQuotientView, RingSwitchRelationView};
use crate::compute::{
    OperationCtx, RingSwitchProveBackend, RingSwitchQuotientKernel, RingSwitchQuotientPlan,
    RingSwitchRelationKernel, RingSwitchRelationPlan,
};
use crate::validation::validate_i8_setup_log_basis;

/// Add only the high-half quotient contribution of `challenge * ring`.
///
/// Skips the first `D - pos` coefficients per challenge term that cannot
/// contribute (degree < D), cutting iteration count roughly in half.
#[inline(always)]
fn add_sparse_ring_product_high_half<F: FieldCore + CanonicalField, const D: usize>(
    quotient: &mut [F],
    challenge: &SparseChallenge,
    ring: &CyclotomicRing<F, D>,
) {
    let rc = ring.coefficients();
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let c = F::from_i64(coeff as i64);
        let p = pos as usize;
        for s in (D - p)..D {
            quotient[p + s - D] += c * rc[s];
        }
    }
}

#[inline(always)]
fn add_integer_ring_product_high_half<F: FieldCore + CanonicalField, const D: usize>(
    quotient: &mut [F],
    challenge: &IntegerChallenge,
    ring: &CyclotomicRing<F, D>,
) {
    let rc = ring.coefficients();
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let c = F::from_i64(i64::from(coeff));
        let p = pos as usize;
        for s in (D - p)..D {
            quotient[p + s - D] += c * rc[s];
        }
    }
}

fn parallel_high_half_accumulate<F, R, const D: usize>(
    challenges: &Challenges,
    tensor_products: Option<&[IntegerChallenge]>,
    ring_fn: R,
) -> Result<Vec<F>, AkitaError>
where
    F: FieldCore + CanonicalField + Send + Sync,
    R: Fn(usize) -> Option<CyclotomicRing<F, D>> + Sync,
{
    let tensor_products = match challenges {
        Challenges::Tensor { factored: _ } => Some(tensor_products.ok_or_else(|| {
            AkitaError::InvalidSetup("tensor fold products were not materialized".to_string())
        })?),
        Challenges::Sparse { .. } => None,
    };
    let total = challenges.logical_len();
    let out = cfg_fold_reduce!(
        0..total,
        || vec![F::zero(); D],
        |mut acc: Vec<F>, i: usize| {
            let Some(ring) = ring_fn(i) else {
                return acc;
            };
            match challenges {
                Challenges::Sparse {
                    challenges: sparse, ..
                } => add_sparse_ring_product_high_half::<F, D>(&mut acc, &sparse[i], &ring),
                Challenges::Tensor { factored: _ } => {
                    if let Some(products) = tensor_products {
                        add_integer_ring_product_high_half::<F, D>(&mut acc, &products[i], &ring);
                    }
                }
            }
            acc
        },
        |mut a: Vec<F>, b: Vec<F>| {
            for (ai, bi) in a.iter_mut().zip(b.iter()) {
                *ai += *bi;
            }
            a
        }
    );
    Ok(out)
}

/// Relation quotient `r` returned by [`compute_relation_quotient`].
pub(crate) type RelationQuotientOutput<F, const D: usize> = Vec<CyclotomicRing<F, D>>;

fn quotient_from_cyclic_and_reduced<F: FieldCore + HalvingField, const D: usize>(
    cyclic: &CyclotomicRing<F, D>,
    reduced: &CyclotomicRing<F, D>,
) -> CyclotomicRing<F, D> {
    let cyc_c = cyclic.coefficients();
    let red_c = reduced.coefficients();
    let quotient = std::array::from_fn(|k| (cyc_c[k] - red_c[k]).half());
    CyclotomicRing::from_coefficients(quotient)
}

fn add_cyclic_ring_product<F: FieldCore, const D: usize>(
    acc: &mut [F; D],
    lhs: &CyclotomicRing<F, D>,
    rhs: &CyclotomicRing<F, D>,
) {
    let lhs_coeffs = lhs.coefficients();
    let rhs_coeffs = rhs.coefficients();
    for (i, &a) in lhs_coeffs.iter().enumerate() {
        if a.is_zero() {
            continue;
        }
        for (j, &b) in rhs_coeffs.iter().enumerate() {
            if !b.is_zero() {
                acc[(i + j) % D] += a * b;
            }
        }
    }
}

fn add_cyclic_scalar_ring_product<F: FieldCore, const D: usize>(
    acc: &mut [F; D],
    scalar: F,
    rhs: &CyclotomicRing<F, D>,
) {
    for (idx, &coeff) in rhs.coefficients().iter().enumerate() {
        if !coeff.is_zero() {
            acc[idx] += scalar * coeff;
        }
    }
}

fn centered_i32_ring<F: FieldCore + FromPrimitiveInt, const D: usize>(
    coeffs: &[i32; D],
) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|idx| F::from_i64(coeffs[idx] as i64)))
}

fn cyclic_consistency_z_product<F, const D: usize>(
    ring_multiplier_point: &RingMultiplierOpeningPoint<F, D>,
    z_folded_centered: &[[i32; D]],
    block_len: usize,
    depth_commit: usize,
    log_basis: u32,
) -> Result<(CyclotomicRing<F, D>, CyclotomicRing<F, D>), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
{
    let inner_width = block_len
        .checked_mul(depth_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("z inner width overflow".to_string()))?;
    if inner_width == 0 || z_folded_centered.len() != inner_width {
        return Err(AkitaError::InvalidInput(format!(
            "ring-multiplier z layout mismatch: z_folded_len={} block_len={} depth_commit={} expected={}",
            z_folded_centered.len(),
            block_len,
            depth_commit,
            inner_width
        )));
    }
    let g_commit = gadget_row_scalars::<F>(depth_commit, log_basis);
    let mut cyclic = [F::zero(); D];
    let mut reduced = CyclotomicRing::<F, D>::zero();

    {
        if ring_multiplier_point.a_len() < block_len {
            return Err(AkitaError::InvalidInput(format!(
                "ring-multiplier a length mismatch: actual={} expected_at_least={block_len}",
                ring_multiplier_point.a_len()
            )));
        }
        for block_idx in 0..block_len {
            let mut z_block = CyclotomicRing::<F, D>::zero();
            for (digit_idx, &g) in g_commit.iter().enumerate() {
                let z_idx = block_idx * depth_commit + digit_idx;
                z_block += centered_i32_ring::<F, D>(&z_folded_centered[z_idx]).scale(&g);
            }
            if let Some(scalar) = ring_multiplier_point.a_constant_coeff(block_idx) {
                add_cyclic_scalar_ring_product::<F, D>(&mut cyclic, scalar, &z_block);
                reduced += z_block.scale(&scalar);
            } else {
                let a_rings = ring_multiplier_point
                    .a_rings()
                    .ok_or(AkitaError::InvalidProof)?;
                let multiplier = a_rings.get(block_idx).ok_or(AkitaError::InvalidProof)?;
                add_cyclic_ring_product::<F, D>(&mut cyclic, multiplier, &z_block);
                reduced += *multiplier * z_block;
            }
        }
    }

    Ok((CyclotomicRing::from_coefficients(cyclic), reduced))
}

/// Split-eq replacement for `generate_m` + `compute_r_via_poly_division`.
///
/// Computes `r` such that `M·z = y + (X^D+1)·r` without materializing M or z.
/// Uses split-eq factoring: `kron(left, gadget) · decomposed = left · pre_decomp`.
///
/// # Errors
///
/// Returns an error if the claim grouping, row layout, or split-eq witness
/// dimensions are inconsistent.
#[allow(clippy::too_many_arguments, clippy::needless_borrow)]
#[tracing::instrument(skip_all, name = "compute_relation_quotient")]
pub fn compute_relation_quotient<F, B, const D: usize>(
    ring_switch_ctx: &OperationCtx<'_, F, B, D>,
    lp: &LevelParams,
    challenges: &Challenges,
    e_hat_flat: &[[i8; D]],
    t_hat: &FlatDigitBlocks<D>,
    recomposed_inner_rows: &[Vec<CyclotomicRing<F, D>>],
    e_folded: &[CyclotomicRing<F, D>],
    ring_multiplier_point: &RingMultiplierOpeningPoint<F, D>,
    row_coefficient_rings: &[CyclotomicRing<F, D>],
    z_folded_centered: &[[i32; D]],
    z_folded_centered_inf_norm: u32,
    y: &[CyclotomicRing<F, D>],
    num_polys: usize,
    blocks_per_claim: usize,
    inner_width: usize,
    m_row_layout: MRowLayout,
) -> Result<RelationQuotientOutput<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HalvingField,
    B: RingSwitchProveBackend<F, D>,
{
    let backend = ring_switch_ctx.backend();
    let prepared = ring_switch_ctx.prepared();
    validate_i8_setup_log_basis(lp.log_basis, "for i8 prover decomposition")?;
    if num_polys == 0 {
        return Err(AkitaError::InvalidProof);
    }
    let num_claims = row_coefficient_rings.len();
    if num_polys != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    if num_claims.checked_mul(blocks_per_claim) != Some(e_folded.len()) {
        return Err(AkitaError::InvalidProof);
    }
    let expected_inner_rows = num_polys
        .checked_mul(blocks_per_claim)
        .ok_or(AkitaError::InvalidProof)?;
    if recomposed_inner_rows.len() != expected_inner_rows {
        return Err(AkitaError::InvalidProof);
    }
    let expected_challenges = num_claims
        .checked_mul(blocks_per_claim)
        .ok_or(AkitaError::InvalidProof)?;
    if challenges.logical_len() != expected_challenges {
        return Err(AkitaError::InvalidProof);
    }
    if e_hat_flat.len()
        != expected_challenges
            .checked_mul(lp.num_digits_open)
            .ok_or(AkitaError::InvalidProof)?
    {
        return Err(AkitaError::InvalidProof);
    }
    let n_b = lp.b_key.row_len();
    let n_d = lp.d_key.row_len();
    let n_a = lp.a_key.row_len();
    let expected_t_hat_block_digits = n_a
        .checked_mul(lp.num_digits_open)
        .ok_or(AkitaError::InvalidProof)?;
    let expected_t_hat_flat_digits = expected_inner_rows
        .checked_mul(expected_t_hat_block_digits)
        .ok_or(AkitaError::InvalidProof)?;
    if t_hat.block_count() != expected_inner_rows
        || t_hat
            .block_sizes()
            .iter()
            .any(|&block_size| block_size != expected_t_hat_block_digits)
        || t_hat.flat_digits().len() != expected_t_hat_flat_digits
    {
        return Err(AkitaError::InvalidProof);
    }
    // Terminal layout drops the D-rows from M (and from `y`). All structural
    // offsets must use `n_d_active`, not `n_d`, to match the verifier.
    let n_d_active = match m_row_layout {
        MRowLayout::WithDBlock => n_d,
        MRowLayout::WithoutDBlock => 0,
    };
    let num_rows = lp.m_row_count_for(1, m_row_layout)?;
    if y.len() != num_rows {
        return Err(AkitaError::InvalidProof);
    }
    // Canonical row layout: consistency (1) | A | B | D.
    let a_start = lp.a_start();
    let b_start = lp.b_start()?;
    let d_start = lp.d_start(1)?;

    if inner_width == 0 || z_folded_centered.len() != inner_width {
        return Err(AkitaError::InvalidProof);
    }

    let mut z_segments = z_folded_centered.chunks(inner_width);
    let first_z_segment = z_segments.next().ok_or(AkitaError::InvalidProof)?;

    let relation_rows = RingSwitchRelationKernel::relation_rows(
        backend,
        prepared,
        RingSwitchRelationView {
            e_hat: e_hat_flat,
            t_hat: t_hat.flat_digits(),
            z_segment: first_z_segment,
            z_folded_centered_inf_norm,
        },
        RingSwitchRelationPlan {
            n_d: n_d_active,
            n_b,
            n_a,
            log_basis: lp.log_basis,
        },
    )?;
    if relation_rows.d_cyclic.len() != n_d_active
        || relation_rows.b_cyclic.len() != n_b
        || relation_rows.a_quotients.len() != n_a
    {
        return Err(AkitaError::InvalidProof);
    }
    let mut a_quotients = relation_rows.a_quotients;
    let b_cyclic = relation_rows.b_cyclic;
    let d_cyclic = relation_rows.d_cyclic;
    for z_segment in z_segments {
        let segment_rows = RingSwitchQuotientKernel::quotient_rows(
            backend,
            prepared,
            RingSwitchQuotientView {
                z_segment,
                z_folded_centered_inf_norm,
            },
            RingSwitchQuotientPlan { n_a },
        )?;
        if segment_rows.len() != n_a {
            return Err(AkitaError::InvalidProof);
        }
        for (dst, src) in a_quotients.iter_mut().zip(segment_rows.into_iter()) {
            *dst += src;
        }
    }
    let commitment_cyclic_rows = b_cyclic;
    if commitment_cyclic_rows.len() != n_b {
        return Err(AkitaError::InvalidProof);
    }
    let constant_opening_multipliers = ring_multiplier_point.is_constant();
    let consistency_z_quotient = if constant_opening_multipliers {
        // Degree-one openings embed scalar weights as constant rings. Cyclic
        // and negacyclic multiplication by a constant agree, so the quotient
        // row is identically zero.
        CyclotomicRing::<F, D>::zero()
    } else {
        let (consistency_z_cyclic, consistency_z_reduced) = cyclic_consistency_z_product::<F, D>(
            ring_multiplier_point,
            z_folded_centered,
            lp.block_len,
            lp.num_digits_commit,
            lp.log_basis,
        )?;
        quotient_from_cyclic_and_reduced(&consistency_z_cyclic, &consistency_z_reduced)
    };

    let tensor_products = match challenges {
        Challenges::Tensor { factored } => Some(factored.expand_integer::<D>()?),
        Challenges::Sparse { .. } => None,
    };
    let tensor_products = tensor_products.as_deref();
    let mut result = Vec::with_capacity(num_rows);
    let mut other_time = 0.0f64;

    for row_idx in 0..num_rows {
        if row_idx == 0 {
            let t_row = Instant::now();
            let _span = tracing::info_span!("challenge_fold_row").entered();
            // Consistency row: Σ c_i · e_folded[i] over all (claim, block).
            let quotient =
                parallel_high_half_accumulate::<F, _, D>(challenges, tensor_products, |i| {
                    Some(e_folded[i])
                })?;
            let mut quotient = CyclotomicRing::from_slice(&quotient);
            quotient -= consistency_z_quotient;
            result.push(quotient);
            other_time += t_row.elapsed().as_secs_f64();
        } else if row_idx < b_start {
            let t_row = Instant::now();
            let _span = tracing::info_span!("A_row").entered();
            let a_idx = row_idx - a_start;

            // In a dense single commitment group, claim order is polynomial
            // order. Iterate `(claim, block)` over the challenge space and
            // read the matching committed polynomial block directly.
            let mut quotient =
                parallel_high_half_accumulate::<F, _, D>(challenges, tensor_products, |i| {
                    let claim_idx = i / blocks_per_claim;
                    let block_idx = i % blocks_per_claim;
                    let inner_idx = claim_idx * blocks_per_claim + block_idx;
                    recomposed_inner_rows[inner_idx].get(a_idx).copied()
                })?;

            let a_q = a_quotients[a_idx].coefficients();
            for k in 0..D {
                quotient[k] -= a_q[k];
            }
            result.push(CyclotomicRing::from_slice(&quotient));
            other_time += t_row.elapsed().as_secs_f64();
        } else if row_idx < d_start {
            // B-block: B·t̂; RHS is the sent commitment in `y`.
            let commit_idx = row_idx - b_start;
            let cyclic = commitment_cyclic_rows
                .get(commit_idx)
                .ok_or(AkitaError::InvalidProof)?;
            result.push(quotient_from_cyclic_and_reduced(cyclic, &y[row_idx]));
        } else {
            // D-block: v = D·ê.
            result.push(quotient_from_cyclic_and_reduced(
                &d_cyclic[row_idx - d_start],
                &y[row_idx],
            ));
        }
    }

    tracing::debug!(other_s = other_time, "compute_r breakdown");

    Ok(result)
}
