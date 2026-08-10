use super::*;
use std::mem::size_of;
use std::ops::Range;

mod streamed;

use streamed::{
    fused_split_eq_quotients_one_shot_streamed, streamed_centered_quotient_rows_chunked,
    streamed_cyclic_i8_rows_chunked,
};

/// Minimum number of Rayon work-units for the fused one-shot kernel.
const MIN_FUSED_TILES: usize = 30;
#[cfg(target_arch = "aarch64")]
const FUSED_L2_CACHE_BYTES: usize = 4 * 1024 * 1024;
#[cfg(not(target_arch = "aarch64"))]
const FUSED_L2_CACHE_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy)]
struct CenteredRhsBounds {
    capacity: u64,
    lut: u64,
}

#[derive(Clone, Copy)]
enum DigitRole {
    Witness,
    Outer,
}

#[derive(Clone, Copy)]
struct FusedQuotientPlan {
    n_d: usize,
    n_b: usize,
    n_a: usize,
    witness_len: usize,
    t_len: usize,
    z_len: usize,
    max_col: usize,
    w_digit_abs_bound: u64,
    t_digit_abs_bound: u64,
    z_bounds: CenteredRhsBounds,
    w_chunk_width: Option<usize>,
    t_chunk_width: Option<usize>,
    z_chunk_width: Option<usize>,
    matrix_extent: usize,
}

impl FusedQuotientPlan {
    fn is_one_shot(self) -> bool {
        self.w_chunk_width
            .is_some_and(|width| width >= self.witness_len)
            && self.t_chunk_width.is_some_and(|width| width >= self.t_len)
            && self.z_chunk_width.is_some_and(|width| width >= self.z_len)
    }

    #[inline]
    fn d_row(self, row: usize) -> Range<usize> {
        self.row(row, self.witness_len)
    }

    #[inline]
    fn b_row(self, row: usize) -> Range<usize> {
        self.row(row, self.t_len)
    }

    #[inline]
    fn a_row(self, row: usize) -> Range<usize> {
        self.row(row, self.z_len)
    }

    fn digit_shape(self, role: DigitRole) -> (usize, usize, u64) {
        match role {
            DigitRole::Witness => (self.n_d, self.witness_len, self.w_digit_abs_bound),
            DigitRole::Outer => (self.n_b, self.t_len, self.t_digit_abs_bound),
        }
    }

    fn digit_row(self, role: DigitRole, row: usize) -> Range<usize> {
        match role {
            DigitRole::Witness => self.d_row(row),
            DigitRole::Outer => self.b_row(row),
        }
    }

    #[inline]
    fn row(self, row: usize, width: usize) -> Range<usize> {
        let start = row * width;
        start..start + width
    }

    #[inline]
    fn chunk_range(len: usize, width: usize, chunk_index: usize) -> Range<usize> {
        let start = chunk_index * width;
        start..(start + width).min(len)
    }

    fn validate_streamed<F: FieldCore, const D: usize>(
        self,
        source: &[CyclotomicRing<F, D>],
    ) -> Result<(), AkitaError> {
        if source.len() < self.matrix_extent {
            return Err(AkitaError::InvalidSetup(format!(
                "streamed fused quotients need {} setup ring elements, got {}",
                self.matrix_extent,
                source.len()
            )));
        }
        Ok(())
    }
}

pub(crate) fn fused_quotient_matrix_extent(
    n_d: usize,
    witness_len: usize,
    n_b: usize,
    t_len: usize,
    n_a: usize,
    z_len: usize,
) -> Result<usize, AkitaError> {
    [(n_d, witness_len), (n_b, t_len), (n_a, z_len)]
        .into_iter()
        .try_fold(0, |extent, (rows, width)| {
            rows.checked_mul(width)
                .map(|role_extent| extent.max(role_extent))
        })
        .ok_or_else(|| AkitaError::InvalidSetup("fused quotient matrix extent overflow".into()))
}

fn fused_quotient_digit_bounds(
    log_basis_open: u32,
    log_basis_outer: u32,
) -> Result<(u64, u64), AkitaError> {
    validate_i8_log_basis(log_basis_open)?;
    validate_i8_log_basis(log_basis_outer)?;
    Ok((
        balanced_digit_abs_bound(log_basis_open),
        balanced_digit_abs_bound(log_basis_outer),
    ))
}

#[allow(clippy::too_many_arguments)]
fn plan_fused_quotients<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    e_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    n_d: usize,
    n_b: usize,
    n_a: usize,
    z_folded_max_abs: u32,
    w_digit_abs_bound: u64,
    t_digit_abs_bound: u64,
    params: &CrtNttParamSet<W, K, D>,
) -> Result<FusedQuotientPlan, AkitaError> {
    let witness_len = if n_d != 0 { e_hat.len() } else { 0 };
    let t_len = if n_b != 0 { t_hat.len() } else { 0 };
    let z_len = if n_a != 0 { z_folded_rings.len() } else { 0 };
    if !digit_rows_within_digit_bound::<D>(e_hat, witness_len, w_digit_abs_bound) {
        return Err(AkitaError::InvalidInput(
            "fused quotient e_hat contains digits outside its log_basis range".to_string(),
        ));
    }
    if !digit_rows_within_digit_bound::<D>(t_hat, t_len, t_digit_abs_bound) {
        return Err(AkitaError::InvalidInput(
            "fused quotient t_hat contains digits outside its log_basis range".to_string(),
        ));
    }

    let actual_z_abs_bound = centered_rows_abs_bound(z_folded_rings, z_len);
    let z_bounds = CenteredRhsBounds {
        capacity: u64::from(z_folded_max_abs).max(actual_z_abs_bound),
        lut: actual_z_abs_bound,
    };
    debug_assert!(
        centered_rows_within_bound(z_folded_rings, z_len, z_bounds.capacity),
        "fused quotient centered RHS bound is smaller than the actual max"
    );

    let w_chunk_width = (witness_len == 0)
        .then_some(1)
        .or_else(|| safe_crt_chunk_width::<F, W, K, D>(params, witness_len, w_digit_abs_bound));
    let t_chunk_width = (t_len == 0)
        .then_some(1)
        .or_else(|| safe_crt_chunk_width::<F, W, K, D>(params, t_len, t_digit_abs_bound));
    let z_chunk_width = (z_len == 0 || z_bounds.capacity == 0)
        .then_some(z_len.max(1))
        .or_else(|| safe_crt_chunk_width::<F, W, K, D>(params, z_len, z_bounds.capacity));
    let matrix_extent = fused_quotient_matrix_extent(n_d, witness_len, n_b, t_len, n_a, z_len)?;

    Ok(FusedQuotientPlan {
        n_d,
        n_b,
        n_a,
        witness_len,
        t_len,
        z_len,
        max_col: witness_len.max(t_len).max(z_len),
        w_digit_abs_bound,
        t_digit_abs_bound,
        z_bounds,
        w_chunk_width,
        t_chunk_width,
        z_chunk_width,
        matrix_extent,
    })
}

/// Fused column-tiled kernel for the three split-eq mat-vec products.
///
/// Replaces three separate NTT-cached mat-vec calls (D-cyclic, B-cyclic,
/// A-quotient) with a single pass over the shared NTT cache. Within each
/// column tile, cache entries are loaded once and reused across all three
/// products with their exact row bounds, eliminating redundant DRAM reads.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn fused_split_eq_quotients_with_params<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    d_cyc_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    b_cyc_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    a_cyc_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    neg_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    e_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    plan: FusedQuotientPlan,
    params: &CrtNttParamSet<W, K, D>,
) -> Result<
    (
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
    ),
    AkitaError,
> {
    if plan.max_col == 0 {
        return Ok((
            vec![CyclotomicRing::<F, D>::zero(); plan.n_d],
            vec![CyclotomicRing::<F, D>::zero(); plan.n_b],
            vec![CyclotomicRing::<F, D>::zero(); plan.n_a],
        ));
    }

    if plan.is_one_shot() {
        return Ok(fused_split_eq_quotients_one_shot(
            d_cyc_rows,
            b_cyc_rows,
            a_cyc_rows,
            neg_rows,
            e_hat,
            t_hat,
            z_folded_rings,
            plan,
            params,
        ));
    }

    let w_chunk_width = plan.w_chunk_width.ok_or_else(|| {
        AkitaError::InvalidSetup("CRT parameters cannot represent one e_hat term".to_string())
    })?;
    let t_chunk_width = plan.t_chunk_width.ok_or_else(|| {
        AkitaError::InvalidSetup("CRT parameters cannot represent one t_hat term".to_string())
    })?;
    let d_result = accumulate_cyclic_i8_rows(
        d_cyc_rows,
        plan.n_d,
        e_hat,
        plan.witness_len,
        w_chunk_width,
        plan.w_digit_abs_bound,
        params,
    );
    let b_result = accumulate_cyclic_i8_rows(
        b_cyc_rows,
        plan.n_b,
        t_hat,
        plan.t_len,
        t_chunk_width,
        plan.t_digit_abs_bound,
        params,
    );
    let a_result = accumulate_centered_quotient_rows(
        neg_rows,
        a_cyc_rows,
        plan.n_a,
        z_folded_rings,
        plan,
        params,
    );

    Ok((d_result, b_result, a_result))
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn fused_split_eq_quotients_one_shot<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    d_cyc_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    b_cyc_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    a_cyc_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    neg_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    e_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    plan: FusedQuotientPlan,
    params: &CrtNttParamSet<W, K, D>,
) -> (
    Vec<CyclotomicRing<F, D>>,
    Vec<CyclotomicRing<F, D>>,
    Vec<CyclotomicRing<F, D>>,
) {
    let digit_bound = plan.w_digit_abs_bound.max(plan.t_digit_abs_bound);
    let digit_lut = (plan.witness_len != 0 || plan.t_len != 0)
        .then(|| DigitMontLut::<W, K>::new_with_digit_bound(params, digit_bound));
    let centered_lut = (plan.z_len != 0 && plan.z_bounds.lut <= u64::from(CENTERED_LUT_MAX_ABS))
        .then(|| CenteredMontLut::<W, K>::new(params, plan.z_bounds.lut as i32));
    let base_tw = (FUSED_L2_CACHE_BYTES / (K * D * size_of::<W>())).max(1);
    let tw = base_tw.min(plan.max_col.div_ceil(MIN_FUSED_TILES).max(1));
    let num_tiles = plan.max_col.div_ceil(tw);
    let zero = CyclotomicCrtNtt::<W, K, D>::zero();

    let (d_accs, b_accs, a_neg_accs, a_cyc_accs) = cfg_fold_reduce!(
        0..num_tiles,
        || (
            vec![zero.clone(); plan.n_d],
            vec![zero.clone(); plan.n_b],
            vec![zero.clone(); plan.n_a],
            vec![zero.clone(); plan.n_a],
        ),
        |mut accs: (
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>,
        ),
         tile_idx| {
            let tile_start = tile_idx * tw;
            let tile_end = (tile_start + tw).min(plan.max_col);

            for j in tile_start..tile_end {
                if j < plan.witness_len && !is_zero_plane(&e_hat[j]) {
                    let lut = digit_lut.as_ref().expect("digit LUT exists");
                    let ntt_w = CyclotomicCrtNtt::from_i8_cyclic_with_lut(&e_hat[j], params, lut);
                    for (acc_d, cyc_row) in accs.0.iter_mut().zip(d_cyc_rows.iter()) {
                        accumulate_pointwise_product_into(acc_d, &cyc_row[j], &ntt_w, params);
                    }
                }

                if j < plan.t_len && !is_zero_plane(&t_hat[j]) {
                    let lut = digit_lut.as_ref().expect("digit LUT exists");
                    let ntt_t = CyclotomicCrtNtt::from_i8_cyclic_with_lut(&t_hat[j], params, lut);
                    for (acc_b, cyc_row) in accs.1.iter_mut().zip(b_cyc_rows.iter()) {
                        accumulate_pointwise_product_into(acc_b, &cyc_row[j], &ntt_t, params);
                    }
                }

                if j < plan.z_len && !is_zero_centered_row(&z_folded_rings[j]) {
                    let (ntt_z_neg, ntt_z_cyc) = if let Some(ref lut) = centered_lut {
                        // SAFETY: `plan_fused_quotients` computed
                        // `z_bounds.lut` from these `plan.z_len` rows. This
                        // loop keeps `j < plan.z_len`, and `lut` was built for
                        // that inclusive centered coefficient bound.
                        unsafe {
                            CyclotomicCrtNtt::from_centered_i32_pair_with_lut_unchecked(
                                &z_folded_rings[j],
                                params,
                                lut,
                            )
                        }
                    } else {
                        CyclotomicCrtNtt::from_centered_i32_pair_with_params(
                            &z_folded_rings[j],
                            params,
                        )
                    };
                    for ((acc_neg, acc_cyc), (neg_row, cyc_row)) in accs
                        .2
                        .iter_mut()
                        .zip(accs.3.iter_mut())
                        .zip(neg_rows.iter().zip(a_cyc_rows.iter()))
                    {
                        accumulate_pointwise_product_into(acc_neg, &neg_row[j], &ntt_z_neg, params);
                        accumulate_pointwise_product_into(acc_cyc, &cyc_row[j], &ntt_z_cyc, params);
                    }
                }
            }
            accs
        },
        |mut a: (
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>,
        ),
         b| {
            for r in 0..plan.n_d {
                add_ntt_into(&mut a.0[r], &b.0[r], params);
            }
            for r in 0..plan.n_b {
                add_ntt_into(&mut a.1[r], &b.1[r], params);
            }
            for r in 0..plan.n_a {
                add_ntt_into(&mut a.2[r], &b.2[r], params);
                add_ntt_into(&mut a.3[r], &b.3[r], params);
            }
            a
        }
    );

    let d_result = d_accs
        .into_iter()
        .map(|acc| acc.to_ring_cyclic(params))
        .collect();
    let b_result = b_accs
        .into_iter()
        .map(|acc| acc.to_ring_cyclic(params))
        .collect();
    let a_result = a_neg_accs
        .into_iter()
        .zip(a_cyc_accs)
        .map(|(neg_acc, cyc_acc)| {
            let neg_ring: CyclotomicRing<F, D> = neg_acc.to_ring(params);
            let cyc_ring: CyclotomicRing<F, D> = cyc_acc.to_ring_cyclic(params);
            quotient_from_cyclic_and_negacyclic(&cyc_ring, &neg_ring)
        })
        .collect();

    (d_result, b_result, a_result)
}

/// Streamed counterpart of [`fused_split_eq_quotients_prover_bounds`].
///
/// Entries stream from `flat`, A's field-form prefix covering every product's
/// `rows x width` extent. Roles that exceed one CRT accumulator are reduced in
/// capacity-safe chunks. If the selected protocol CRT profile cannot represent
/// one term, this rejects without allocating a retained NTT cache.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn fused_split_eq_quotients_streamed_prover_bounds<
    F: FieldCore + CanonicalField + HalvingField,
    const D: usize,
>(
    source: &[CyclotomicRing<F, D>],
    n_d: usize,
    n_b: usize,
    n_a: usize,
    e_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    z_folded_max_abs: u32,
    log_basis_open: u32,
    log_basis_outer: u32,
) -> Result<
    (
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
    ),
    AkitaError,
> {
    let (w_digit_abs_bound, t_digit_abs_bound) =
        fused_quotient_digit_bounds(log_basis_open, log_basis_outer)?;
    macro_rules! run {
        ($params:expr) => {{
            let params = $params;
            let plan = plan_fused_quotients::<F, _, _, D>(
                e_hat,
                t_hat,
                z_folded_rings,
                n_d,
                n_b,
                n_a,
                z_folded_max_abs,
                w_digit_abs_bound,
                t_digit_abs_bound,
                &params,
            )?;
            plan.validate_streamed(source)?;
            if plan.is_one_shot() {
                return Ok(fused_split_eq_quotients_one_shot_streamed(
                    source,
                    e_hat,
                    t_hat,
                    z_folded_rings,
                    plan,
                    &params,
                ));
            }

            let w_chunk_width = plan.w_chunk_width.ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "CRT parameters cannot represent one streamed e_hat term".to_string(),
                )
            })?;
            let t_chunk_width = plan.t_chunk_width.ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "CRT parameters cannot represent one streamed t_hat term".to_string(),
                )
            })?;
            let z_chunk_width = plan.z_chunk_width.ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "CRT parameters cannot represent one streamed centered term".to_string(),
                )
            })?;
            tracing::info!(
                witness_len = plan.witness_len,
                w_chunk_width,
                t_len = plan.t_len,
                t_chunk_width,
                z_len = plan.z_len,
                z_chunk_width,
                "streamed fused quotients using CRT-safe chunks"
            );
            let d_rows = streamed_cyclic_i8_rows_chunked(
                source,
                e_hat,
                plan,
                DigitRole::Witness,
                w_chunk_width,
                &params,
            );
            let b_rows = streamed_cyclic_i8_rows_chunked(
                source,
                t_hat,
                plan,
                DigitRole::Outer,
                t_chunk_width,
                &params,
            );
            let a_rows = streamed_centered_quotient_rows_chunked(
                source,
                z_folded_rings,
                plan,
                z_chunk_width,
                &params,
            );
            Ok((d_rows, b_rows, a_rows))
        }};
    }
    match select_crt_ntt_params::<F, D>()? {
        ProtocolCrtNttParams::Q32(params) => run!(params),
        ProtocolCrtNttParams::Q64(params) => run!(params),
        ProtocolCrtNttParams::Q128(params) => run!(params),
    }
}

fn accumulate_cyclic_i8_rows<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    cyc_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    num_rows: usize,
    rhs: &[[i8; D]],
    rhs_len: usize,
    chunk_width: usize,
    rhs_abs_bound: u64,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    if num_rows == 0 {
        return vec![];
    }
    if rhs_len == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); num_rows];
    }

    let num_chunks = rhs_len.div_ceil(chunk_width);
    let lut = DigitMontLut::<W, K>::new_with_digit_bound(params, rhs_abs_bound);

    cfg_fold_reduce!(
        0..num_chunks,
        || vec![CyclotomicRing::<F, D>::zero(); num_rows],
        |mut out: Vec<CyclotomicRing<F, D>>, chunk_idx| {
            let chunk = FusedQuotientPlan::chunk_range(rhs_len, chunk_width, chunk_idx);
            let mut accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); num_rows];

            for j in chunk {
                if is_zero_plane(&rhs[j]) {
                    continue;
                }
                let ntt_rhs = CyclotomicCrtNtt::from_i8_cyclic_with_lut(&rhs[j], params, &lut);
                for (acc, row) in accs.iter_mut().zip(cyc_rows.iter()) {
                    accumulate_pointwise_product_into(acc, &row[j], &ntt_rhs, params);
                }
            }

            for (dst, acc) in out.iter_mut().zip(accs) {
                *dst += acc.to_ring_cyclic(params);
            }
            out
        },
        |mut a: Vec<CyclotomicRing<F, D>>, b| {
            for (dst, src) in a.iter_mut().zip(b) {
                *dst += src;
            }
            a
        }
    )
}

fn centered_rows_within_bound<const D: usize>(rows: &[[i32; D]], len: usize, bound: u64) -> bool {
    rows.iter()
        .take(len)
        .flat_map(|row| row.iter())
        .all(|&coeff| u64::from(coeff.unsigned_abs()) <= bound)
}

fn centered_rows_abs_bound<const D: usize>(rows: &[[i32; D]], len: usize) -> u64 {
    rows.iter()
        .take(len)
        .flat_map(|row| row.iter())
        .map(|&coeff| u64::from(coeff.unsigned_abs()))
        .max()
        .unwrap_or(0)
}

fn centered_i32_ring<F: CanonicalField, const D: usize>(coeffs: &[i32; D]) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(from_fn(|k| F::from_i64(coeffs[k] as i64)))
}

fn accumulate_centered_quotient_rows<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    neg_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    cyc_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    num_rows: usize,
    z_folded_rings: &[[i32; D]],
    plan: FusedQuotientPlan,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    if num_rows == 0 {
        return vec![];
    }
    if plan.z_len == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); num_rows];
    }

    if plan.z_bounds.lut == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); num_rows];
    }

    let Some(chunk_width) = plan.z_chunk_width else {
        return accumulate_centered_quotient_rows_field(
            neg_rows,
            cyc_rows,
            num_rows,
            z_folded_rings,
            plan.z_len,
            params,
        );
    };
    let centered_lut = (plan.z_bounds.lut <= u64::from(CENTERED_LUT_MAX_ABS))
        .then(|| CenteredMontLut::<W, K>::new(params, plan.z_bounds.lut as i32));
    let num_chunks = plan.z_len.div_ceil(chunk_width);

    cfg_fold_reduce!(
        0..num_chunks,
        || vec![CyclotomicRing::<F, D>::zero(); num_rows],
        |mut out: Vec<CyclotomicRing<F, D>>, chunk_idx| {
            let chunk = FusedQuotientPlan::chunk_range(plan.z_len, chunk_width, chunk_idx);
            let mut neg_accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); num_rows];
            let mut cyc_accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); num_rows];

            for j in chunk {
                if is_zero_centered_row(&z_folded_rings[j]) {
                    continue;
                }
                let (ntt_z_neg, ntt_z_cyc) = if let Some(ref lut) = centered_lut {
                    // SAFETY: `plan_fused_quotients` computed
                    // `z_bounds.lut` from these `plan.z_len` rows. This loop
                    // keeps `j < plan.z_len`, and `lut` was built for that
                    // inclusive centered coefficient bound.
                    unsafe {
                        CyclotomicCrtNtt::from_centered_i32_pair_with_lut_unchecked(
                            &z_folded_rings[j],
                            params,
                            lut,
                        )
                    }
                } else {
                    CyclotomicCrtNtt::from_centered_i32_pair_with_params(&z_folded_rings[j], params)
                };
                for ((neg_acc, cyc_acc), (neg_row, cyc_row)) in neg_accs
                    .iter_mut()
                    .zip(cyc_accs.iter_mut())
                    .zip(neg_rows.iter().zip(cyc_rows.iter()))
                {
                    accumulate_pointwise_product_into(neg_acc, &neg_row[j], &ntt_z_neg, params);
                    accumulate_pointwise_product_into(cyc_acc, &cyc_row[j], &ntt_z_cyc, params);
                }
            }

            for ((dst, neg_acc), cyc_acc) in out.iter_mut().zip(neg_accs).zip(cyc_accs) {
                let neg_ring: CyclotomicRing<F, D> = neg_acc.to_ring(params);
                let cyc_ring: CyclotomicRing<F, D> = cyc_acc.to_ring_cyclic(params);
                *dst += quotient_from_cyclic_and_negacyclic(&cyc_ring, &neg_ring);
            }
            out
        },
        |mut a: Vec<CyclotomicRing<F, D>>, b| {
            for (dst, src) in a.iter_mut().zip(b) {
                *dst += src;
            }
            a
        }
    )
}

fn accumulate_centered_quotient_rows_field<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    neg_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    cyc_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    num_rows: usize,
    z_folded_rings: &[[i32; D]],
    z_len: usize,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    cfg_into_iter!(0..num_rows)
        .map(|row_idx| {
            let mut out = CyclotomicRing::<F, D>::zero();
            for j in 0..z_len {
                if is_zero_centered_row(&z_folded_rings[j]) {
                    continue;
                }
                let z = centered_i32_ring::<F, D>(&z_folded_rings[j]);
                let neg_lhs: CyclotomicRing<F, D> = neg_rows[row_idx][j].to_ring(params);
                let cyc_lhs: CyclotomicRing<F, D> = cyc_rows[row_idx][j].to_ring_cyclic(params);
                let neg_product = neg_lhs * z;
                let mut cyc_product = CyclotomicRing::<F, D>::zero();
                add_cyclic_product_into(&mut cyc_product, &cyc_lhs, &z);
                out += quotient_from_cyclic_and_negacyclic(&cyc_product, &neg_product);
            }
            out
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn centered_quotient_rows_with_i16_tail_params<
    F: FieldCore + CanonicalField + HalvingField,
    const K: usize,
    const D: usize,
>(
    neg: &[CyclotomicCrtNtt<i32, K, D>],
    cyc: &[CyclotomicCrtNtt<i32, K, D>],
    tail_neg: &[CyclotomicCrtNtt<i16, 1, D>],
    tail_cyc: &[CyclotomicCrtNtt<i16, 1, D>],
    num_rows: usize,
    z_folded_rings: &[[i32; D]],
    z_folded_max_abs: u32,
    params: &CrtNttParamSet<i32, K, D>,
    tail_params: &CrtNttParamSet<i16, 1, D>,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
    if num_rows == 0 {
        return Ok(Vec::new());
    }
    let width = z_folded_rings.len();
    let required = num_rows
        .checked_mul(width)
        .ok_or_else(|| AkitaError::InvalidSetup("quotient matrix shape overflows".into()))?;
    if width == 0
        || [neg.len(), cyc.len(), tail_neg.len(), tail_cyc.len()]
            .into_iter()
            .any(|length| length < required)
    {
        return Err(AkitaError::InvalidSetup(
            "base-plus-tail quotient cache is shorter than its matrix shape".into(),
        ));
    }
    let actual_bound = centered_rows_abs_bound(z_folded_rings, width);
    if actual_bound == 0 {
        return Ok(vec![CyclotomicRing::<F, D>::zero(); num_rows]);
    }
    let capacity_bound = u64::from(z_folded_max_abs).max(actual_bound);
    let capacity = params
        .crt_capacity()
        .with_prime_modulus(tail_params.primes[0].p as u128);
    let chunk_width = capacity
        .max_safe_width::<F, D>(capacity_bound)
        .map(|safe| safe.min(width))
        .filter(|&safe| safe > 0)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("centered quotient exceeds base plus i16-tail capacity".into())
        })?;
    let mixed_params = I16TailParams::new(params.clone(), tail_params.clone());
    let base_lut = (actual_bound <= u64::from(CENTERED_LUT_MAX_ABS))
        .then(|| CenteredMontLut::<i32, K>::new(params, actual_bound as i32));
    let num_chunks = width.div_ceil(chunk_width);

    Ok(cfg_fold_reduce!(
        0..num_chunks,
        || vec![CyclotomicRing::<F, D>::zero(); num_rows],
        |mut out: Vec<CyclotomicRing<F, D>>, chunk_idx| {
            let start = chunk_idx * chunk_width;
            let end = (start + chunk_width).min(width);
            let mut base_neg_accs = vec![CyclotomicCrtNtt::<i32, K, D>::zero(); num_rows];
            let mut base_cyc_accs = vec![CyclotomicCrtNtt::<i32, K, D>::zero(); num_rows];
            let mut tail_neg_accs = vec![CyclotomicCrtNtt::<i16, 1, D>::zero(); num_rows];
            let mut tail_cyc_accs = vec![CyclotomicCrtNtt::<i16, 1, D>::zero(); num_rows];

            for (offset, z_ring) in z_folded_rings[start..end].iter().enumerate() {
                if is_zero_centered_row(z_ring) {
                    continue;
                }
                let j = start + offset;
                let (z_neg, z_cyc) = if let Some(ref lut) = base_lut {
                    unsafe {
                        CyclotomicCrtNtt::from_centered_i32_pair_with_lut_unchecked(
                            z_ring, params, lut,
                        )
                    }
                } else {
                    CyclotomicCrtNtt::from_centered_i32_pair_with_params(z_ring, params)
                };
                let (z_tail_neg, z_tail_cyc) =
                    CyclotomicCrtNtt::from_centered_i32_pair_with_params(z_ring, tail_params);
                for row in 0..num_rows {
                    let index = row * width + j;
                    accumulate_pointwise_product_into(
                        &mut base_neg_accs[row],
                        &neg[index],
                        &z_neg,
                        params,
                    );
                    accumulate_pointwise_product_into(
                        &mut base_cyc_accs[row],
                        &cyc[index],
                        &z_cyc,
                        params,
                    );
                    accumulate_pointwise_product_into(
                        &mut tail_neg_accs[row],
                        &tail_neg[index],
                        &z_tail_neg,
                        tail_params,
                    );
                    accumulate_pointwise_product_into(
                        &mut tail_cyc_accs[row],
                        &tail_cyc[index],
                        &z_tail_cyc,
                        tail_params,
                    );
                }
            }

            for row in 0..num_rows {
                let neg_ring = ntt_with_i16_tail_to_ring(
                    &base_neg_accs[row],
                    &tail_neg_accs[row],
                    &mixed_params,
                );
                let cyc_ring = cyclic_ntt_with_i16_tail_to_ring(
                    &base_cyc_accs[row],
                    &tail_cyc_accs[row],
                    &mixed_params,
                );
                out[row] += quotient_from_cyclic_and_negacyclic(&cyc_ring, &neg_ring);
            }
            out
        },
        |mut left: Vec<CyclotomicRing<F, D>>, right| {
            for (dst, src) in left.iter_mut().zip(right) {
                *dst += src;
            }
            left
        }
    ))
}

/// Centered A-quotient rows using the protocol CRT prefix plus its 14-bit tail.
pub(crate) fn centered_quotient_rows_with_i16_tail<
    F: FieldCore + CanonicalField + HalvingField,
    const D: usize,
>(
    negacyclic_slot: &PreparedNttCache<D>,
    cyclic_slot: &PreparedNttCache<D>,
    tail_slot: &PreparedNttCache<D>,
    num_rows: usize,
    z_folded_rings: &[[i32; D]],
    z_folded_max_abs: u32,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
    let tail = tail_slot.i16_tail_pair().ok_or_else(|| {
        AkitaError::InvalidSetup("paired i16-tail NTT domain not prepared".into())
    })?;
    macro_rules! dispatch {
        ($neg_base:expr, $cyc_base:expr) => {{
            let (neg_base, cyc_base) = ($neg_base, $cyc_base);
            if neg_base.params() != cyc_base.params() {
                return Err(AkitaError::InvalidSetup(
                    "cyclic and negacyclic NTT profiles do not match".into(),
                ));
            }
            centered_quotient_rows_with_i16_tail_params(
                neg_base.negacyclic().ok_or_else(|| {
                    AkitaError::InvalidSetup("negacyclic NTT domain not prepared".into())
                })?,
                cyc_base.cyclic().ok_or_else(|| {
                    AkitaError::InvalidSetup("cyclic NTT domain not prepared".into())
                })?,
                tail.negacyclic(),
                tail.cyclic(),
                num_rows,
                z_folded_rings,
                z_folded_max_abs,
                neg_base.params(),
                tail.params(),
            )
        }};
    }
    if let (Some(neg), Some(cyc)) = (negacyclic_slot.q32_base(), cyclic_slot.q32_base()) {
        dispatch!(neg, cyc)
    } else if let (Some(neg), Some(cyc)) = (negacyclic_slot.q64_base(), cyclic_slot.q64_base()) {
        dispatch!(neg, cyc)
    } else if let (Some(neg), Some(cyc)) = (negacyclic_slot.q128_base(), cyclic_slot.q128_base()) {
        dispatch!(neg, cyc)
    } else {
        Err(AkitaError::InvalidSetup(
            "cyclic and negacyclic NTT profiles do not match".into(),
        ))
    }
}

/// Fused split-eq quotient kernel dispatching over [`PreparedNttCache`] variants.
///
/// Computes three NTT-cached mat-vec products in a single tiled pass:
/// - D-cyclic: `cyc[0..n_d] · e_hat` (cyclic domain)
/// - B-cyclic: `cyc[0..n_b] · t_hat` (cyclic domain)
/// - A-quotient: `(cyc[0..n_a]·z_cyc − neg[0..n_a]·z_neg) / 2`
///
/// All roles share the same underlying coefficient matrix, but each role uses
/// its own packed row width.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[tracing::instrument(skip_all, name = "fused_split_eq_quotients")]
#[cfg(test)]
pub(crate) fn fused_split_eq_quotients<
    F: FieldCore + CanonicalField + HalvingField,
    const D: usize,
>(
    slot: &PreparedNttCache<D>,
    n_d: usize,
    n_b: usize,
    n_a: usize,
    e_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    z_folded_max_abs: u32,
) -> Result<
    (
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
    ),
    AkitaError,
> {
    fused_split_eq_quotients_with_digit_bound(
        slot,
        slot,
        n_d,
        n_b,
        n_a,
        e_hat,
        t_hat,
        z_folded_rings,
        z_folded_max_abs,
        balanced_digit_abs_bound(6),
        balanced_digit_abs_bound(6),
    )
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn fused_split_eq_quotients_prover_bounds<
    F: FieldCore + CanonicalField + HalvingField,
    const D: usize,
>(
    negacyclic_slot: &PreparedNttCache<D>,
    cyclic_slot: &PreparedNttCache<D>,
    n_d: usize,
    n_b: usize,
    n_a: usize,
    e_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    z_folded_max_abs: u32,
    log_basis_open: u32,
    log_basis_outer: u32,
) -> Result<
    (
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
    ),
    AkitaError,
> {
    let (w_digit_abs_bound, t_digit_abs_bound) =
        fused_quotient_digit_bounds(log_basis_open, log_basis_outer)?;
    fused_split_eq_quotients_with_digit_bound(
        negacyclic_slot,
        cyclic_slot,
        n_d,
        n_b,
        n_a,
        e_hat,
        t_hat,
        z_folded_rings,
        z_folded_max_abs,
        w_digit_abs_bound,
        t_digit_abs_bound,
    )
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn fused_split_eq_quotients_with_digit_bound<
    F: FieldCore + CanonicalField + HalvingField,
    const D: usize,
>(
    negacyclic_slot: &PreparedNttCache<D>,
    cyclic_slot: &PreparedNttCache<D>,
    n_d: usize,
    n_b: usize,
    n_a: usize,
    e_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    z_folded_max_abs: u32,
    w_digit_abs_bound: u64,
    t_digit_abs_bound: u64,
) -> Result<
    (
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
    ),
    AkitaError,
> {
    macro_rules! run {
        ($neg_base:expr, $cyc_base:expr) => {{
            let (neg_base, cyc_base) = ($neg_base, $cyc_base);
            let (params, cyclic_params) = (neg_base.params(), cyc_base.params());
            if params != cyclic_params {
                return Err(AkitaError::InvalidSetup(
                    "cyclic and negacyclic NTT profiles do not match".into(),
                ));
            }
            let neg = match neg_base.negacyclic() {
                Some(neg) => neg,
                None if n_a == 0 => &[],
                None => {
                    return Err(AkitaError::InvalidSetup(
                        "negacyclic NTT domain not prepared".into(),
                    ));
                }
            };
            let cyc = cyc_base
                .cyclic()
                .ok_or_else(|| AkitaError::InvalidSetup("cyclic NTT domain not prepared".into()))?;
            let plan = plan_fused_quotients::<F, _, _, D>(
                e_hat,
                t_hat,
                z_folded_rings,
                n_d,
                n_b,
                n_a,
                z_folded_max_abs,
                w_digit_abs_bound,
                t_digit_abs_bound,
                params,
            )?;
            let neg_rows = matrix_rows(neg, plan.n_a, |row| plan.a_row(row), "negacyclic")?;
            let d_rows = matrix_rows(cyc, plan.n_d, |row| plan.d_row(row), "D cyclic")?;
            let b_rows = matrix_rows(cyc, plan.n_b, |row| plan.b_row(row), "B cyclic")?;
            let a_rows = matrix_rows(cyc, plan.n_a, |row| plan.a_row(row), "A cyclic")?;
            fused_split_eq_quotients_with_params(
                &d_rows,
                &b_rows,
                &a_rows,
                &neg_rows,
                e_hat,
                t_hat,
                z_folded_rings,
                plan,
                params,
            )
        }};
    }
    match (
        negacyclic_slot.q32_base(),
        cyclic_slot.q32_base(),
        negacyclic_slot.q64_base(),
        cyclic_slot.q64_base(),
        negacyclic_slot.q128_base(),
        cyclic_slot.q128_base(),
    ) {
        (Some(neg), Some(cyc), _, _, _, _) => run!(neg, cyc),
        (_, _, Some(neg), Some(cyc), _, _) => run!(neg, cyc),
        (_, _, _, _, Some(neg), Some(cyc)) => run!(neg, cyc),
        _ => Err(AkitaError::InvalidSetup(
            "cyclic and negacyclic NTT profiles do not match".into(),
        )),
    }
}

fn matrix_rows<'a, T>(
    flat: &'a [T],
    num_rows: usize,
    row_range: impl Fn(usize) -> Range<usize>,
    role: &str,
) -> Result<Vec<&'a [T]>, AkitaError> {
    let needed = num_rows.checked_sub(1).map_or(0, |row| row_range(row).end);
    if flat.len() < needed {
        return Err(AkitaError::InvalidSetup(format!(
            "{role} fused quotient matrix needs {needed} elements, got {}",
            flat.len()
        )));
    }
    Ok((0..num_rows).map(|row| &flat[row_range(row)]).collect())
}
