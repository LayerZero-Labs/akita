use super::*;

/// Run `$func` on the prepared base of `$slot`. Only the bases field `$field`
/// can select are compiled; see [`FieldCrtBases`].
macro_rules! dispatch_slot {
    ($field:ty, $slot:expr, $num_rows:expr, $num_cols:expr, $func:ident $(, $arg:expr)*) => {{
        let nr: usize = $num_rows;
        let nc: usize = $num_cols;
        macro_rules! run_base {
            ($base:expr) => {{
                let base = $base;
                let neg = base.negacyclic().ok_or_else(|| {
                    AkitaError::InvalidSetup("negacyclic NTT domain not prepared".into())
                })?;
                let rows: Vec<&[_]> = (0..nr).map(|i| &neg[i * nc..(i + 1) * nc]).collect();
                $func(&rows, $($arg,)* base.params())
            }};
        }
        'dispatch: {
            if FieldCrtBases::<$field>::Q32 {
                if let Some(base) = $slot.q32_base() {
                    break 'dispatch run_base!(base);
                }
            }
            if FieldCrtBases::<$field>::Q64 {
                if let Some(base) = $slot.q64_base() {
                    break 'dispatch run_base!(base);
                }
            }
            if FieldCrtBases::<$field>::Q128 {
                if let Some(base) = $slot.q128_base() {
                    break 'dispatch run_base!(base);
                }
            }
            return Err(AkitaError::InvalidSetup(
                "signed-i8 matvec requires a base-profile NTT cache".into(),
            ));
        }
    }};
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn mat_vec_mul_ntt_blocks_i8<
    F: Field + CanonicalEncoding,
    S: I8Rows<D>,
    const D: usize,
>(
    slot: &PreparedNttCache<D>,
    num_rows: usize,
    num_cols: usize,
    source: &S,
    blocks: &[Range<usize>],
    shape: DigitShape,
    check_zero: bool,
    policy: I8CommitPolicy,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    validate_i8_log_basis(shape.log_basis)?;
    if matches!(policy, I8CommitPolicy::Dense) && !check_zero && num_rows == 1 {
        return Ok(dispatch_slot!(
            F,
            slot,
            1usize,
            num_cols,
            mat_vec_mul_blocks_i8_single_row_with_params,
            source,
            blocks,
            shape
        )?
        .into_iter()
        .map(|ring| vec![ring])
        .collect());
    }
    macro_rules! run {
        ($zero:literal) => {{
            fn driver<
                F: Field + CanonicalEncoding,
                S: I8Rows<D>,
                W: PrimeWidth,
                const K: usize,
                const D: usize,
            >(
                matrix: &[&[CyclotomicCrtNtt<W, K, D>]],
                source: &S,
                blocks: &[Range<usize>],
                shape: DigitShape,
                policy: I8CommitPolicy,
                params: &CrtNttParamSet<W, K, D>,
            ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
                mat_vec_mul_blocks_i8_with_params_impl::<F, S, W, K, D, $zero>(
                    matrix,
                    source,
                    blocks,
                    shape,
                    policy,
                    "i8 matvec CRT capacity cannot fit a single term",
                    params,
                )
            }
            dispatch_slot!(F, slot, num_rows, num_cols, driver, source, blocks, shape, policy)
        }};
    }
    if check_zero {
        run!(true)
    } else {
        run!(false)
    }
}

/// Column-tiled A*x across multiple blocks of pre-decomposed i8 digit planes.
///
/// This uses one inner digit per coefficient. It skips the `CyclotomicRing -> i8 digit plane`
/// decomposition entirely because the caller already holds each coefficient as a
/// balanced digit plane for a validated `log_basis <= 8`.
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_digits_i8")]
pub fn mat_vec_mul_ntt_digits_i8<F: Field + CanonicalEncoding, const D: usize>(
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
    dispatch_slot!(
        F,
        slot,
        num_rows,
        num_cols,
        mat_vec_mul_digits_i8_with_params,
        blocks,
        log_basis
    )
}
