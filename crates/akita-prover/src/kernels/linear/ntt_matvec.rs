use super::*;

macro_rules! dispatch_slot {
    ($slot:expr, $num_rows:expr, $num_cols:expr, $func:ident $(, $arg:expr)*) => {{
        let nr: usize = $num_rows;
        let nc: usize = $num_cols;
        match $slot {
            PreparedNttCache::Q32 { neg, params: p, .. } => {
                let rows: Vec<&[_]> = (0..nr).map(|i| &neg[i * nc..(i + 1) * nc]).collect();
                $func(&rows, $($arg,)* p)
            }
            PreparedNttCache::Q64 { neg, params: p, .. } => {
                let rows: Vec<&[_]> = (0..nr).map(|i| &neg[i * nc..(i + 1) * nc]).collect();
                $func(&rows, $($arg,)* p)
            }
            PreparedNttCache::Q128 { neg, params: p, .. } => {
                let rows: Vec<&[_]> = (0..nr).map(|i| &neg[i * nc..(i + 1) * nc]).collect();
                $func(&rows, $($arg,)* p)
            }
        }
    }};
}

/// Column-tiled A*x across multiple blocks simultaneously.
///
/// Each rayon thread owns one column tile of `ntt_mat` (sized to fit in L2
/// cache) and iterates over all blocks, accumulating partial NTT results.
/// The matrix is loaded from DRAM exactly once. A final reduction sums
/// partial accumulators across tiles for each block.
///
/// Accepts raw ring-coefficient slices per block. Decomposes to i8 digits
/// on-the-fly per tile to avoid materializing all digits at once.
/// Tile width is auto-computed from ring parameters and target L2 cache size.
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_i8")]
pub fn mat_vec_mul_ntt_i8<F: FieldCore + CanonicalField, const D: usize>(
    slot: &PreparedNttCache<D>,
    num_rows: usize,
    num_cols: usize,
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    validate_i8_log_basis(log_basis)?;
    Ok(dispatch_slot!(
        slot,
        num_rows,
        num_cols,
        mat_vec_mul_i8_with_params,
        blocks,
        num_digits,
        log_basis
    ))
}

/// Dense-optimized variant of [`mat_vec_mul_ntt_i8`].
///
/// Skips the full-plane zero scans that are useful for sparse inputs but are
/// almost always wasted work on dense witnesses.
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_i8_dense")]
pub fn mat_vec_mul_ntt_i8_dense<F: FieldCore + CanonicalField, const D: usize>(
    slot: &PreparedNttCache<D>,
    num_rows: usize,
    num_cols: usize,
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    validate_i8_log_basis(log_basis)?;
    Ok(dispatch_slot!(
        slot,
        num_rows,
        num_cols,
        mat_vec_mul_i8_dense_with_params,
        blocks,
        num_digits,
        log_basis
    ))
}

/// Single-row dense variant of [`mat_vec_mul_ntt_i8_dense`].
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_i8_dense_single_row")]
pub fn mat_vec_mul_ntt_i8_dense_single_row<F: FieldCore + CanonicalField, const D: usize>(
    slot: &PreparedNttCache<D>,
    num_cols: usize,
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
    validate_i8_log_basis(log_basis)?;
    Ok(dispatch_slot!(
        slot,
        1usize,
        num_cols,
        mat_vec_mul_i8_dense_single_row_with_params,
        blocks,
        num_digits,
        log_basis
    ))
}

/// Column-tiled A*x across multiple blocks of pre-decomposed i8 digit planes.
///
/// This is the `num_digits_inner = 1` specialization of
/// [`mat_vec_mul_ntt_i8`]. It skips the `CyclotomicRing -> i8 digit plane`
/// decomposition entirely because the caller already holds each coefficient as a
/// balanced digit plane for a validated `log_basis <= 8`.
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_digits_i8")]
pub fn mat_vec_mul_ntt_digits_i8<F: FieldCore + CanonicalField, const D: usize>(
    slot: &PreparedNttCache<D>,
    num_rows: usize,
    num_cols: usize,
    blocks: &[&[[i8; D]]],
    log_basis: u32,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    validate_i8_log_basis(log_basis)?;
    for block in blocks {
        validate_digit_rows_for_log_basis(
            block,
            num_cols.min(block.len()),
            log_basis,
            "for predecomposed digit mat-vec",
        )?;
    }
    Ok(dispatch_slot!(
        slot,
        num_rows,
        num_cols,
        mat_vec_mul_digits_i8_with_params,
        blocks,
        log_basis
    ))
}

/// Dense pre-decomposed digit mat-vec for the backend-owned digit cache.
///
/// The generic pre-decomposed digit kernel skips all-zero planes, which is
/// profitable for sparse witnesses. Dense witnesses pay that scan on almost
/// every plane, so this kernel uses the same math without the zero checks. The
/// cache is produced by Akita's validated decomposer and does not need a second
/// full scan at each commit.
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_dense_digits_i8")]
pub(crate) fn mat_vec_mul_ntt_dense_digits_i8<F: FieldCore + CanonicalField, const D: usize>(
    slot: &PreparedNttCache<D>,
    num_rows: usize,
    num_cols: usize,
    blocks: &[&[[i8; D]]],
    log_basis: u32,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    validate_i8_log_basis(log_basis)?;
    Ok(dispatch_slot!(
        slot,
        num_rows,
        num_cols,
        mat_vec_mul_dense_digits_i8_with_params,
        blocks,
        log_basis
    ))
}

/// Fold-major (block) direct-signed-i8 variant for recursive witnesses.
///
/// The block/column layout and output shape match
/// [`mat_vec_mul_ntt_digits_i8`], but this path does not assume the rows are
/// balanced gadget digits: it is the `num_digits_inner = 1` commit path for a
/// recursive witness whose extension-field tensor base-lift packing can push
/// coefficients past the balanced range. Coefficients too large for the CRT
/// lift are rejected as `AkitaError` rather than panicking.
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_raw_digits_i8")]
pub fn mat_vec_mul_ntt_raw_digits_i8<F: FieldCore + CanonicalField, const D: usize>(
    slot: &PreparedNttCache<D>,
    num_rows: usize,
    num_cols: usize,
    blocks: &[&[[i8; D]]],
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    dispatch_slot!(
        slot,
        num_rows,
        num_cols,
        mat_vec_mul_raw_digits_i8_with_params,
        blocks
    )
}
