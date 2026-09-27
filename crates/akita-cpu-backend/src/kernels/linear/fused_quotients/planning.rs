use super::*;
use std::mem::size_of;

#[derive(Clone, Copy)]
pub(super) struct CenteredRhsBounds {
    pub(super) capacity: u64,
    pub(super) lut: u64,
}

#[derive(Clone, Copy)]
pub(super) struct FusedQuotientPlan {
    pub(super) n_b: usize,
    pub(super) n_a: usize,
    pub(super) t_len: usize,
    pub(super) z_len: usize,
    pub(super) max_col: usize,
    pub(super) t_digit_abs_bound: u64,
    pub(super) z_bounds: CenteredRhsBounds,
    pub(super) t_chunk_width: Option<usize>,
    pub(super) z_chunk_width: Option<usize>,
    pub(super) matrix_extent: usize,
}

impl FusedQuotientPlan {
    pub(super) fn is_one_shot(self) -> bool {
        self.t_chunk_width.is_some_and(|width| width >= self.t_len)
            && self.z_chunk_width.is_some_and(|width| width >= self.z_len)
    }

    #[inline]
    pub(super) fn chunk_range(len: usize, width: usize, chunk_index: usize) -> Range<usize> {
        let start = chunk_index * width;
        start..(start + width).min(len)
    }
}

pub(crate) fn fused_quotient_matrix_extent(
    n_b: usize,
    t_len: usize,
    n_a: usize,
    z_len: usize,
) -> Result<usize, AkitaError> {
    [(n_b, t_len), (n_a, z_len)]
        .into_iter()
        .try_fold(0, |extent, (rows, width)| {
            rows.checked_mul(width)
                .map(|role_extent| extent.max(role_extent))
        })
        .ok_or_else(|| AkitaError::InvalidSetup("fused quotient matrix extent overflow".into()))
}

pub(super) fn fused_quotient_digit_bound(log_basis_outer: u32) -> Result<u64, AkitaError> {
    validate_i8_log_basis(log_basis_outer)?;
    Ok(balanced_digit_abs_bound(log_basis_outer))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn plan_fused_quotients<
    F: Field + CanonicalEncoding,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    n_b: usize,
    n_a: usize,
    z_folded_max_abs: u32,
    t_digit_abs_bound: u64,
    params: &CrtNttParamSet<W, K, D>,
) -> Result<FusedQuotientPlan, AkitaError> {
    let t_len = if n_b != 0 { t_hat.len() } else { 0 };
    let z_len = if n_a != 0 { z_folded_rings.len() } else { 0 };
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

    let t_chunk_width = (t_len == 0)
        .then_some(1)
        .or_else(|| safe_crt_chunk_width::<F, W, K, D>(params, t_len, t_digit_abs_bound));
    let z_chunk_width = (z_len == 0 || z_bounds.capacity == 0)
        .then_some(z_len.max(1))
        .or_else(|| safe_crt_chunk_width::<F, W, K, D>(params, z_len, z_bounds.capacity));
    let matrix_extent = fused_quotient_matrix_extent(n_b, t_len, n_a, z_len)?;

    Ok(FusedQuotientPlan {
        n_b,
        n_a,
        t_len,
        z_len,
        max_col: t_len.max(z_len),
        t_digit_abs_bound,
        z_bounds,
        t_chunk_width,
        z_chunk_width,
        matrix_extent,
    })
}

pub(super) fn fused_tile_width<W: PrimeWidth, const K: usize, const D: usize>(
    plan: FusedQuotientPlan,
    _params: &CrtNttParamSet<W, K, D>,
) -> usize {
    let base_tw = (FUSED_L2_CACHE_BYTES / (K * D * size_of::<W>())).max(1);
    base_tw.min(plan.max_col.div_ceil(MIN_FUSED_TILES).max(1))
}

#[cfg(test)]
pub(in crate::kernels::linear) fn fused_test_plan_route<
    F: Field + CanonicalEncoding,
    const D: usize,
>(
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    n_b: usize,
    n_a: usize,
    z_folded_max_abs: u32,
    log_basis_outer: u32,
) -> (bool, Option<usize>, Option<usize>, usize, usize) {
    macro_rules! route {
        ($params:expr) => {{
            let params = $params;
            let plan = plan_fused_quotients::<F, _, _, D>(
                t_hat,
                z_folded_rings,
                n_b,
                n_a,
                z_folded_max_abs,
                fused_quotient_digit_bound(log_basis_outer).expect("valid digit bound"),
                &params,
            )
            .expect("valid fused quotient plan");
            let tile_width = fused_tile_width(plan, &params);
            (
                plan.is_one_shot(),
                plan.t_chunk_width,
                plan.z_chunk_width,
                tile_width,
                params.pointwise_dot_batch_size(),
            )
        }};
    }
    match select_crt_ntt_params::<F, D>().expect("supported profile") {
        ProtocolCrtNttParams::Q32(params) => route!(params),
        ProtocolCrtNttParams::Q64(params) => route!(params),
        ProtocolCrtNttParams::Q128(params) => route!(params),
    }
}

pub(super) fn centered_rows_within_bound<const D: usize>(
    rows: &[[i32; D]],
    len: usize,
    bound: u64,
) -> bool {
    rows.iter()
        .take(len)
        .flat_map(|row| row.iter())
        .all(|&coeff| u64::from(coeff.unsigned_abs()) <= bound)
}

pub(super) fn centered_rows_abs_bound<const D: usize>(rows: &[[i32; D]], len: usize) -> u64 {
    rows.iter()
        .take(len)
        .flat_map(|row| row.iter())
        .map(|&coeff| u64::from(coeff.unsigned_abs()))
        .max()
        .unwrap_or(0)
}
