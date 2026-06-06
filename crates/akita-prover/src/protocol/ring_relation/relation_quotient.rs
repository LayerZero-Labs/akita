use super::repeated_b::repeated_b_commitment_rows;
#[cfg(feature = "zk")]
use super::repeated_b::{add_zk_b_blinding_cyclic_rows, add_zk_d_blinding_cyclic_rows};
use super::*;
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

fn cyclic_public_row_product<F, const D: usize>(
    e_folded: &[CyclotomicRing<F, D>],
    ring_multiplier_points: &[RingMultiplierOpeningPoint<F, D>],
    claim_to_point: &[usize],
    row_coefficient_rings: &[CyclotomicRing<F, D>],
    target_point_idx: usize,
    blocks_per_claim: usize,
) -> Result<CyclotomicRing<F, D>, AkitaError>
where
    F: FieldCore,
{
    let mut cyclic = [F::zero(); D];
    if row_coefficient_rings.len() != claim_to_point.len() {
        return Err(AkitaError::InvalidProof);
    }
    for (claim_idx, &point_idx) in claim_to_point.iter().enumerate() {
        if point_idx != target_point_idx {
            continue;
        }
        let point = ring_multiplier_points
            .get(point_idx)
            .ok_or(AkitaError::InvalidProof)?;
        for block_idx in 0..blocks_per_claim {
            let folded_idx = claim_idx
                .checked_mul(blocks_per_claim)
                .and_then(|idx| idx.checked_add(block_idx))
                .ok_or(AkitaError::InvalidProof)?;
            let folded = e_folded.get(folded_idx).ok_or(AkitaError::InvalidProof)?;
            let weighted_multiplier = if let Some(scalar) = point.b_constant_coeff(block_idx) {
                row_coefficient_rings[claim_idx].scale(&scalar)
            } else {
                let b_rings = point.b_rings().ok_or(AkitaError::InvalidProof)?;
                row_coefficient_rings[claim_idx] * b_rings[block_idx]
            };
            add_cyclic_ring_product::<F, D>(&mut cyclic, &weighted_multiplier, folded);
        }
    }
    Ok(CyclotomicRing::from_coefficients(cyclic))
}

fn ring_is_constant<F: FieldCore, const D: usize>(ring: &CyclotomicRing<F, D>) -> bool {
    ring.coefficients()[1..].iter().all(|coeff| coeff.is_zero())
}

fn centered_i32_ring<F: FieldCore + FromPrimitiveInt, const D: usize>(
    coeffs: &[i32; D],
) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|idx| F::from_i64(coeffs[idx] as i64)))
}

fn cyclic_consistency_z_product<F, const D: usize>(
    ring_multiplier_points: &[RingMultiplierOpeningPoint<F, D>],
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
    if inner_width == 0
        || z_folded_centered.len()
            != ring_multiplier_points
                .len()
                .checked_mul(inner_width)
                .ok_or_else(|| AkitaError::InvalidSetup("z point width overflow".to_string()))?
    {
        return Err(AkitaError::InvalidInput(format!(
            "ring-multiplier z layout mismatch: z_folded_len={} points={} block_len={} depth_commit={} expected={}",
            z_folded_centered.len(),
            ring_multiplier_points.len(),
            block_len,
            depth_commit,
            ring_multiplier_points.len() * inner_width
        )));
    }
    let g_commit = gadget_row_scalars::<F>(depth_commit, log_basis);
    let mut cyclic = [F::zero(); D];
    let mut reduced = CyclotomicRing::<F, D>::zero();

    for (point_idx, opening_point) in ring_multiplier_points.iter().enumerate() {
        if opening_point.a_len() < block_len {
            return Err(AkitaError::InvalidInput(format!(
                "ring-multiplier a length mismatch: actual={} expected_at_least={block_len}",
                opening_point.a_len()
            )));
        }
        for block_idx in 0..block_len {
            let mut z_block = CyclotomicRing::<F, D>::zero();
            for (digit_idx, &g) in g_commit.iter().enumerate() {
                let z_idx = point_idx * inner_width + block_idx * depth_commit + digit_idx;
                z_block += centered_i32_ring::<F, D>(&z_folded_centered[z_idx]).scale(&g);
            }
            if let Some(scalar) = opening_point.a_constant_coeff(block_idx) {
                add_cyclic_scalar_ring_product::<F, D>(&mut cyclic, scalar, &z_block);
                reduced += z_block.scale(&scalar);
            } else {
                let a_rings = opening_point.a_rings().ok_or(AkitaError::InvalidProof)?;
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
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    lp: &LevelParams,
    challenges: &Challenges,
    e_hat_flat: &[[i8; D]],
    #[cfg(feature = "zk")] d_blinding_digits: &FlatDigitBlocks<D>,
    t_hat: &FlatDigitBlocks<D>,
    #[cfg(feature = "zk")] b_blinding_digits: &[FlatDigitBlocks<D>],
    recomposed_inner_rows: &[Vec<CyclotomicRing<F, D>>],
    e_folded: &[CyclotomicRing<F, D>],
    ring_multiplier_points: &[RingMultiplierOpeningPoint<F, D>],
    claim_to_point: &[usize],
    claim_to_commitment_group: &[usize],
    claim_poly_in_commitment_group: &[usize],
    row_coefficient_rings: &[CyclotomicRing<F, D>],
    z_folded_centered: &[[i32; D]],
    z_folded_centered_inf_norm: u32,
    y: &[CyclotomicRing<F, D>],
    num_polys_per_commitment_group: &[usize],
    blocks_per_claim: usize,
    inner_width: usize,
    m_row_layout: MRowLayout,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HalvingField,
    B: RingSwitchComputeBackend<F>,
{
    validate_i8_setup_log_basis(lp.log_basis, "for i8 prover decomposition")?;
    if num_polys_per_commitment_group.is_empty() || num_polys_per_commitment_group.contains(&0) {
        return Err(AkitaError::InvalidProof);
    }
    let num_claims = claim_to_commitment_group.len();
    if claim_poly_in_commitment_group.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }
    // Build a flat (claim → global poly slot) map. `recomposed_inner_rows`
    // is flattened by polynomial slot (then block), so the global poly
    // slot is `Σ_{g < group_idx} num_polys_per_commitment_group[g] + poly_idx`. Validate
    // that every claim references a real `(group, poly)` cell.
    let mut group_offsets = Vec::with_capacity(num_polys_per_commitment_group.len());
    let mut acc = 0usize;
    for &count in num_polys_per_commitment_group {
        group_offsets.push(acc);
        acc = acc.checked_add(count).ok_or(AkitaError::InvalidProof)?;
    }
    let total_poly_slots = acc;
    let mut poly_slot_for_claim = Vec::with_capacity(num_claims);
    for claim_idx in 0..num_claims {
        let group_idx = claim_to_commitment_group[claim_idx];
        if group_idx >= num_polys_per_commitment_group.len() {
            return Err(AkitaError::InvalidProof);
        }
        let poly_idx = claim_poly_in_commitment_group[claim_idx];
        if poly_idx >= num_polys_per_commitment_group[group_idx] {
            return Err(AkitaError::InvalidProof);
        }
        poly_slot_for_claim.push(group_offsets[group_idx] + poly_idx);
    }
    let num_z_segments = ring_multiplier_points.len();
    if num_z_segments == 0
        || claim_to_point.len().checked_mul(blocks_per_claim) != Some(e_folded.len())
        || row_coefficient_rings.len() != claim_to_point.len()
        || claim_to_commitment_group.len() != claim_to_point.len()
        || claim_poly_in_commitment_group.len() != claim_to_point.len()
    {
        return Err(AkitaError::InvalidProof);
    }
    let num_points = num_polys_per_commitment_group.len();
    if claim_to_point
        .iter()
        .any(|&point_idx| point_idx >= num_points)
    {
        return Err(AkitaError::InvalidProof);
    }
    let expected_inner_rows = total_poly_slots
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
    let commitment_row_count = n_b
        .checked_mul(num_points)
        .ok_or(AkitaError::InvalidProof)?;
    // Public-output M rows are enforced by the fused trace term, not M itself.
    const NUM_PUBLIC_M_ROWS: usize = 0;
    let num_rows = lp.m_row_count_for(num_points, NUM_PUBLIC_M_ROWS, m_row_layout)?;
    if y.len() != num_rows {
        return Err(AkitaError::InvalidProof);
    }
    // Row layout: consistency (1) | D (n_d_active) | B (commitment_row_count) | A (n_a)
    let d_start = 1 + NUM_PUBLIC_M_ROWS;
    let b_start = d_start + n_d_active;
    let a_start = b_start + commitment_row_count;

    if inner_width == 0
        || z_folded_centered.len()
            != num_z_segments
                .checked_mul(inner_width)
                .ok_or(AkitaError::InvalidProof)?
    {
        return Err(AkitaError::InvalidProof);
    }

    let mut z_segments = z_folded_centered.chunks(inner_width);
    let first_z_segment = z_segments.next().ok_or(AkitaError::InvalidProof)?;

    let use_relation_b_rows = commitment_row_count == n_b && num_points == 1;
    let relation_n_b = if use_relation_b_rows { n_b } else { 0 };
    let relation_t_hat: &[[i8; D]] = if use_relation_b_rows {
        t_hat.flat_digits()
    } else {
        &[]
    };
    let relation_rows = backend.ring_switch_relation_rows::<D>(
        prepared,
        RingSwitchRelationRowsPlan {
            n_d: n_d_active,
            n_b: relation_n_b,
            n_a,
            e_hat: e_hat_flat,
            t_hat: relation_t_hat,
            z_segment: first_z_segment,
            z_folded_centered_inf_norm,
            log_basis: lp.log_basis,
        },
    )?;
    if relation_rows.d_cyclic.len() != n_d_active
        || relation_rows.b_cyclic.len() != relation_n_b
        || relation_rows.a_quotients.len() != n_a
    {
        return Err(AkitaError::InvalidProof);
    }
    let mut a_quotients = relation_rows.a_quotients;
    let b_cyclic = relation_rows.b_cyclic;
    #[cfg(feature = "zk")]
    let mut d_cyclic = relation_rows.d_cyclic;
    #[cfg(not(feature = "zk"))]
    let d_cyclic = relation_rows.d_cyclic;
    #[cfg(feature = "zk")]
    add_zk_d_blinding_cyclic_rows(
        backend,
        prepared,
        n_d_active,
        d_blinding_digits,
        &mut d_cyclic,
    )?;
    for z_segment in z_segments {
        let segment_rows = backend.ring_switch_quotient_rows::<D>(
            prepared,
            RingSwitchQuotientRowsPlan {
                n_a,
                z_segment,
                z_folded_centered_inf_norm,
            },
        )?;
        if segment_rows.len() != n_a {
            return Err(AkitaError::InvalidProof);
        }
        for (dst, src) in a_quotients.iter_mut().zip(segment_rows.into_iter()) {
            *dst += src;
        }
    }
    let commitment_cyclic_rows = if use_relation_b_rows {
        #[cfg(feature = "zk")]
        let mut rows = b_cyclic;
        #[cfg(not(feature = "zk"))]
        let rows = b_cyclic;
        #[cfg(feature = "zk")]
        {
            let blinding = b_blinding_digits.first().ok_or(AkitaError::InvalidProof)?;
            add_zk_b_blinding_cyclic_rows(
                backend,
                prepared,
                n_b,
                blinding.flat_digits().len(),
                blinding,
                &mut rows,
            )?;
        }
        rows
    } else {
        repeated_b_commitment_rows(
            backend,
            prepared,
            n_b,
            t_hat,
            #[cfg(feature = "zk")]
            b_blinding_digits,
            num_polys_per_commitment_group,
            blocks_per_claim,
            lp.log_basis,
        )?
    };
    if commitment_cyclic_rows.len() != commitment_row_count {
        return Err(AkitaError::InvalidProof);
    }
    let constant_opening_multipliers = ring_multiplier_points
        .iter()
        .all(|point| point.is_constant());
    let constant_public_multipliers =
        constant_opening_multipliers && row_coefficient_rings.iter().all(ring_is_constant);
    let consistency_z_quotient = if constant_opening_multipliers {
        // Degree-one openings embed scalar weights as constant rings. Cyclic
        // and negacyclic multiplication by a constant agree, so the quotient
        // row is identically zero.
        CyclotomicRing::<F, D>::zero()
    } else {
        let (consistency_z_cyclic, consistency_z_reduced) = cyclic_consistency_z_product::<F, D>(
            ring_multiplier_points,
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
        } else if row_idx < d_start {
            let _span = tracing::info_span!("bTw_row").entered();
            if constant_public_multipliers {
                // Constant public multipliers have identical cyclic and
                // negacyclic products, so this row contributes no quotient.
                result.push(CyclotomicRing::<F, D>::zero());
            } else {
                let point_idx = row_idx - 1;
                let cyclic = cyclic_public_row_product::<F, D>(
                    e_folded,
                    ring_multiplier_points,
                    claim_to_point,
                    row_coefficient_rings,
                    point_idx,
                    blocks_per_claim,
                )?;
                result.push(quotient_from_cyclic_and_reduced(&cyclic, &y[row_idx]));
            }
        } else if row_idx < b_start {
            result.push(quotient_from_cyclic_and_reduced(
                &d_cyclic[row_idx - d_start],
                &y[row_idx],
            ));
        } else if row_idx < a_start {
            result.push(quotient_from_cyclic_and_reduced(
                &commitment_cyclic_rows[row_idx - b_start],
                &y[row_idx],
            ));
        } else {
            let t_row = Instant::now();
            let _span = tracing::info_span!("A_row").entered();
            let a_idx = row_idx - a_start;

            // Iterate `(claim, block)` over the challenge space and route
            // each cell to its polynomial-slot in `recomposed_inner_rows`
            // (`poly_slot * num_blocks + block_idx`). Iterating over the
            // raw `recomposed_inner_rows.len()` would conflate poly slots
            // with claims and overrun `challenges` whenever a group has
            // more polynomial slots than opened claims.
            let mut quotient =
                parallel_high_half_accumulate::<F, _, D>(challenges, tensor_products, |i| {
                    let claim_idx = i / blocks_per_claim;
                    let block_idx = i % blocks_per_claim;
                    let poly_slot = poly_slot_for_claim[claim_idx];
                    let inner_idx = poly_slot * blocks_per_claim + block_idx;
                    recomposed_inner_rows[inner_idx].get(a_idx).copied()
                })?;

            let a_q = a_quotients[a_idx].coefficients();
            for k in 0..D {
                quotient[k] -= a_q[k];
            }
            result.push(CyclotomicRing::from_slice(&quotient));
            other_time += t_row.elapsed().as_secs_f64();
        }
    }

    tracing::debug!(other_s = other_time, "compute_r breakdown");

    Ok(result)
}

/// Build the RHS vector `y` matching the M row layout:
/// consistency (zero) | D (`v`) | B (`commitment_rows`) | A (zeros).
///
/// Public-output rows bind through the fused trace term, not `y`.
///
/// # Errors
///
/// Returns an error if the supplied row slices do not match the expected row
/// counts for the level layout.
pub fn generate_y<F, const D: usize>(
    v: &[CyclotomicRing<F, D>],
    commitment_rows: &[CyclotomicRing<F, D>],
    n_d: usize,
    n_b: usize,
    n_a: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore,
{
    if v.len() != n_d {
        return Err(AkitaError::InvalidSize {
            expected: n_d,
            actual: v.len(),
        });
    }
    if commitment_rows.is_empty() || !commitment_rows.len().is_multiple_of(n_b) {
        return Err(AkitaError::InvalidSize {
            expected: n_b,
            actual: commitment_rows.len(),
        });
    }
    let mut out = Vec::with_capacity(1 + n_d + commitment_rows.len() + n_a);
    out.push(CyclotomicRing::<F, D>::zero());
    out.extend_from_slice(v);
    out.extend_from_slice(commitment_rows);
    out.extend(std::iter::repeat_n(CyclotomicRing::<F, D>::zero(), n_a));
    Ok(out)
}
