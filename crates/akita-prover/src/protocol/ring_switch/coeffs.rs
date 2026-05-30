use super::*;

/// Build the witness vector `w` from the quadratic equation state.
///
/// This is the first half of the ring switch: it computes `r` and assembles
/// `w` as a flat recursive witness. The resulting `w` is D-agnostic and can be
/// committed at any supported ring dimension by the recursive commitment path.
///
/// # Errors
///
/// Returns an error if the quadratic equation is missing prover-side data.
///
/// # Panics
///
/// Panics with `feature = "zk"` enabled if the zero-length `FlatDigitBlocks`
/// constructor rejects an empty vector (an invariant of the type).
#[tracing::instrument(skip_all, name = "ring_switch_build_w")]
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn ring_switch_build_w<F, B, const D: usize>(
    quad_eq: &mut QuadraticEquation<F, D>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    lp: &LevelParams,
) -> Result<RecursiveWitnessFlat, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + FromPrimitiveInt + HalvingField,
    B: RingSwitchComputeBackend<F>,
{
    let num_claims = quad_eq.claim_to_point().len();
    {
        let x: u8 = 0;
        tracing::trace!(
            stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
            "ring_switch_build_w"
        );
    }
    let w_hat = quad_eq
        .take_w_hat()
        .ok_or_else(|| AkitaError::InvalidInput("missing w_hat in prover".to_string()))?;
    #[cfg(feature = "zk")]
    let d_blinding_digits = quad_eq.take_d_blinding_digits().ok_or_else(|| {
        AkitaError::InvalidInput("missing D-blinding digits in prover".to_string())
    })?;
    let z_pre = quad_eq
        .take_z_pre()
        .ok_or_else(|| AkitaError::InvalidInput("missing centered z_pre in prover".to_string()))?;
    let mut hint = quad_eq
        .take_hint()
        .ok_or_else(|| AkitaError::InvalidInput("missing hint in prover".to_string()))?;
    hint.ensure_recomposed_inner_rows(lp.num_digits_open, lp.log_basis)?;
    #[cfg(feature = "zk")]
    let (decomposed_inner_rows, recomposed_inner_rows, b_blinding_digits) = hint.into_flat_parts();
    #[cfg(not(feature = "zk"))]
    let (decomposed_inner_rows, recomposed_inner_rows) = hint.into_flat_parts();
    let recomposed_inner_rows = recomposed_inner_rows.ok_or_else(|| {
        AkitaError::InvalidInput("missing recomposed inner rows in prover hint".to_string())
    })?;
    let w_folded = quad_eq
        .take_w_folded()
        .ok_or_else(|| AkitaError::InvalidInput("missing w_folded in prover".to_string()))?;

    let r = compute_r_split_eq::<F, B, D>(
        backend,
        prepared,
        lp,
        &quad_eq.challenges,
        w_hat.flat_digits(),
        #[cfg(feature = "zk")]
        &d_blinding_digits,
        &decomposed_inner_rows,
        #[cfg(feature = "zk")]
        &b_blinding_digits,
        &recomposed_inner_rows,
        &w_folded,
        quad_eq.ring_multiplier_points(),
        quad_eq.claim_to_point(),
        quad_eq.claim_to_point_poly(),
        quad_eq.claim_poly_indices(),
        quad_eq.row_coefficient_rings(),
        &z_pre.centered_coeffs,
        z_pre.centered_inf_norm,
        quad_eq.y(),
        quad_eq.num_polys_per_point(),
        quad_eq.num_public_rows(),
        lp.num_blocks,
        lp.inner_width(),
        quad_eq.m_row_layout(),
    )?;
    // Terminal layout drops the D-block from M and from the witness; the
    // d-blinding column segment must also disappear so the prover witness
    // matches the verifier's column offsets.
    #[cfg(feature = "zk")]
    let d_blinding_for_w: FlatDigitBlocks<D> = match quad_eq.m_row_layout() {
        MRowLayout::Intermediate => d_blinding_digits,
        MRowLayout::Terminal => {
            FlatDigitBlocks::zeroed(Vec::new()).expect("empty FlatDigitBlocks always valid")
        }
    };
    let w = {
        let _span = tracing::info_span!("build_w_coeffs").entered();
        build_w_coeffs::<F, D>(
            &w_hat,
            #[cfg(feature = "zk")]
            &d_blinding_for_w,
            &decomposed_inner_rows,
            #[cfg(feature = "zk")]
            &b_blinding_digits,
            &z_pre.centered_coeffs,
            &r,
            lp,
            num_claims,
        )
    };
    Ok(w)
}

pub(super) fn balanced_decompose_centered_i32_i8_into<const D: usize>(
    centered: &[i32; D],
    out: &mut [[i8; D]],
    log_basis: u32,
) {
    let levels = out.len();
    assert!(
        log_basis > 0 && log_basis <= 6,
        "log_basis must be in 1..=6 for i8 output"
    );
    assert!(
        (levels as u32).saturating_mul(log_basis) <= 128 + log_basis,
        "levels * log_basis must be <= 128 + log_basis"
    );

    let half_b = 1i128 << (log_basis - 1);
    let b = half_b << 1;
    let mask = b - 1;

    for coeff_idx in 0..D {
        let mut c = centered[coeff_idx] as i128;
        for plane in out.iter_mut() {
            let d = c & mask;
            let balanced = if d >= half_b { d - b } else { d };
            c = (c - balanced) >> log_basis;
            plane[coeff_idx] = balanced as i8;
        }
    }
}

/// Transpose block-major digit planes to digit-major order (block index
/// innermost): for each compound digit index, emit all blocks in order.
fn emit_planes_block_inner<const D: usize>(
    out: &mut Vec<i8>,
    flat: &[[i8; D]],
    total_blocks: usize,
    planes_per_block: usize,
) {
    debug_assert_eq!(
        flat.len(),
        total_blocks * planes_per_block,
        "emit_planes_block_inner: flat.len()={} != total_blocks({}) * planes_per_block({})",
        flat.len(),
        total_blocks,
        planes_per_block
    );
    for compound_dig in 0..planes_per_block {
        for blk in 0..total_blocks {
            out.extend_from_slice(&flat[blk * planes_per_block + compound_dig]);
        }
    }
}

#[cfg(feature = "zk")]
fn emit_blinding_planes<const D: usize>(
    out: &mut Vec<i8>,
    blinding_by_group: &[FlatDigitBlocks<D>],
) {
    for blinding in blinding_by_group {
        for plane in blinding.flat_digits() {
            out.extend_from_slice(plane);
        }
    }
}

/// Decompose z_pre elements and emit in digit-major order.
///
/// z_pre has `num_points * block_len * depth_commit` elements indexed as
/// `z[point * inner_width + blk * depth_commit + dc]`. Each decomposes into
/// `num_digits_fold` planes.
///
/// Output order: for each `(dc, df)`, emit all `(point, blk)` pairs with
/// the global block index `point * block_len + blk` innermost.
fn emit_z_pre_block_inner<const D: usize>(
    out: &mut Vec<i8>,
    z_pre_centered: &[[i32; D]],
    block_len: usize,
    depth_commit: usize,
    num_digits_fold: usize,
    log_basis: u32,
) {
    let total_elems = z_pre_centered.len();
    let inner_width = block_len * depth_commit;
    debug_assert_eq!(
        total_elems % inner_width,
        0,
        "z_pre length {total_elems} not divisible by inner_width {inner_width}",
    );
    let num_points = total_elems / inner_width;

    let mut all_planes = vec![[0i8; D]; total_elems * num_digits_fold];
    for (k, z_j) in z_pre_centered.iter().enumerate() {
        balanced_decompose_centered_i32_i8_into(
            z_j,
            &mut all_planes[k * num_digits_fold..(k + 1) * num_digits_fold],
            log_basis,
        );
    }

    for dc in 0..depth_commit {
        for df in 0..num_digits_fold {
            for pt in 0..num_points {
                for blk in 0..block_len {
                    let k = pt * inner_width + blk * depth_commit + dc;
                    out.extend_from_slice(&all_planes[k * num_digits_fold + df]);
                }
            }
        }
    }
}

/// Build the committed witness polynomial from ring-domain digit planes.
///
/// Emits field-domain coefficients in digit-major order (block index innermost)
/// with adaptive segment ordering: the segment whose block dimension is the
/// larger power of two comes first.
///
/// Segment ordering:
/// - If `m_vars >= r_vars`: z-hat (`2^m` blocks), e-hat + t-hat (`2^r` blocks), r-hat
/// - If `m_vars < r_vars`: e-hat + t-hat (`2^r` blocks), z-hat (`2^m` blocks), r-hat
///
/// Within each segment, the power-of-2 block index is the fastest-varying
/// (innermost) dimension.
///
/// `FlatDigitBlocks` stores ring-domain data in block-major order (all digit
/// planes for one block contiguously), which is natural for ring-domain matvec
/// and recomposition. This function transposes opening digits to digit-major at
/// the ring-to-field boundary; ZK blinding streams are already direct
/// digit-plane sources and are emitted in matrix-column order.
///
/// # Panics
///
/// Panics if the caller supplies digit blocks whose plane counts do not match
/// the fold layout in `lp`, or if ZK blinding digit counts do not match the
/// configured blinding columns.
#[allow(clippy::too_many_arguments)]
pub fn build_w_coeffs<F: CanonicalField, const D: usize>(
    w_hat: &FlatDigitBlocks<D>,
    #[cfg(feature = "zk")] d_blinding_digits: &FlatDigitBlocks<D>,
    t_hat: &FlatDigitBlocks<D>,
    #[cfg(feature = "zk")] b_blinding_digits: &[FlatDigitBlocks<D>],
    z_pre_centered: &[[i32; D]],
    r: &[CyclotomicRing<F, D>],
    lp: &LevelParams,
    num_claims: usize,
) -> RecursiveWitnessFlat {
    let log_basis = lp.log_basis;
    let num_digits_fold = lp.num_digits_fold(num_claims, F::modulus_bits());
    let depth_open = lp.num_digits_open;
    let depth_commit = lp.num_digits_commit;
    let block_len = lp.block_len;
    let levels = r_decomp_levels::<F>(log_basis);

    let w_hat_planes = w_hat.flat_digits().len();
    let t_hat_planes = t_hat.flat_digits().len();
    #[cfg(feature = "zk")]
    let d_blinding_planes = d_blinding_digits.flat_digits().len();
    #[cfg(not(feature = "zk"))]
    let d_blinding_planes = 0usize;
    #[cfg(feature = "zk")]
    let b_blinding_planes: usize = b_blinding_digits
        .iter()
        .map(|digits| digits.flat_digits().len())
        .sum();
    #[cfg(not(feature = "zk"))]
    let b_blinding_planes = 0usize;
    let z_count = w_hat_planes
        + d_blinding_planes
        + t_hat_planes
        + b_blinding_planes
        + z_pre_centered.len() * num_digits_fold;
    let r_hat_count = r.len() * levels;
    let z_first = lp.m_vars >= lp.r_vars;
    tracing::debug!(
        w_hat_planes,
        d_blinding_planes,
        t_hat_planes,
        b_blinding_planes,
        z_pre_elems = z_pre_centered.len(),
        z_pre_planes = z_pre_centered.len() * num_digits_fold,
        r_elems = r.len(),
        r_planes = r_hat_count,
        total_ring = z_count + r_hat_count,
        total_field = (z_count + r_hat_count) * D,
        z_first,
        "build_w_coeffs"
    );
    let total_planes = z_count + r_hat_count;
    let total_elems = total_planes * D;

    let mut out = Vec::with_capacity(total_elems);

    let w_block_count = w_hat.block_count();
    assert_eq!(
        w_hat_planes,
        w_block_count * depth_open,
        "build_w_coeffs: w_hat block layout does not match open digit depth"
    );
    let t_block_count = t_hat.block_count();
    let t_planes_per_block = if t_block_count == 0 {
        0
    } else {
        assert_eq!(
            t_hat_planes % t_block_count,
            0,
            "build_w_coeffs: t_hat block layout must be uniform"
        );
        t_hat_planes / t_block_count
    };

    if z_first {
        emit_z_pre_block_inner(
            &mut out,
            z_pre_centered,
            block_len,
            depth_commit,
            num_digits_fold,
            log_basis,
        );
        emit_planes_block_inner(&mut out, w_hat.flat_digits(), w_block_count, depth_open);
        emit_planes_block_inner(
            &mut out,
            t_hat.flat_digits(),
            t_block_count,
            t_planes_per_block,
        );
        #[cfg(feature = "zk")]
        emit_blinding_planes(&mut out, b_blinding_digits);
        #[cfg(feature = "zk")]
        emit_blinding_planes(&mut out, std::slice::from_ref(d_blinding_digits));
    } else {
        emit_planes_block_inner(&mut out, w_hat.flat_digits(), w_block_count, depth_open);
        emit_planes_block_inner(
            &mut out,
            t_hat.flat_digits(),
            t_block_count,
            t_planes_per_block,
        );
        #[cfg(feature = "zk")]
        emit_blinding_planes(&mut out, b_blinding_digits);
        #[cfg(feature = "zk")]
        emit_blinding_planes(&mut out, std::slice::from_ref(d_blinding_digits));
        emit_z_pre_block_inner(
            &mut out,
            z_pre_centered,
            block_len,
            depth_commit,
            num_digits_fold,
            log_basis,
        );
    }

    let mut r_planes = vec![[0i8; D]; levels];
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2I8Params::new(levels, log_basis, q);
    for ri in r {
        r_planes.fill([0i8; D]);
        ri.balanced_decompose_pow2_i8_into_with_params(&mut r_planes, &decompose_params);
        for plane in &r_planes {
            out.extend_from_slice(plane);
        }
    }
    RecursiveWitnessFlat::from_i8_digits(out)
}
