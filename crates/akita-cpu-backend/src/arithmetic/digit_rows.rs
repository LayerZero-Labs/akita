use super::prepared::{validate_digit_row_request, CpuPreparedSetup};
use super::CpuBackend;
use crate::kernels::linear::{mat_vec_mul_ntt_digits_i8, mat_vec_mul_ntt_digits_i8_ct};
use crate::opaque::DigitRowsComputeBackend;
use akita_error::AkitaError;
use akita_types::{NttCacheKey, NttTransformDomain};
use jolt_algebra::CyclotomicRing;
use jolt_algebra::{CanonicalEncoding, Field};

impl<F, E> DigitRowsComputeBackend<F> for CpuBackend<F, E>
where
    F: Field + CanonicalEncoding,
{
    fn digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        row_len: usize,
        digit_vectors: &[&[[i8; D]]],
        log_basis: u32,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        digit_rows_with(prepared, row_len, digit_vectors, log_basis, false)
    }

    fn digit_rows_secret<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        row_len: usize,
        digit_vectors: &[&[[i8; D]]],
        log_basis: u32,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        digit_rows_with(prepared, row_len, digit_vectors, log_basis, true)
    }
}

fn digit_rows_with<F, const D: usize>(
    prepared: &CpuPreparedSetup<F>,
    row_len: usize,
    digit_vectors: &[&[[i8; D]]],
    log_basis: u32,
    constant_time: bool,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
where
    F: Field + CanonicalEncoding,
{
    let row_width = digit_vectors
        .first()
        .ok_or_else(|| AkitaError::InvalidInput("digit row batch must be nonempty".into()))?
        .len();
    if digit_vectors.iter().any(|digits| digits.len() != row_width) {
        return Err(AkitaError::InvalidInput(
            "digit row batch inputs must have equal widths".into(),
        ));
    }
    validate_digit_row_request(
        row_len,
        row_width,
        prepared.expanded.shared_matrix.num_field_elements() / D,
    )?;
    prepared.with_shared_ntt::<D, _>(
        NttCacheKey::from_matrix_shape(D, row_len, row_width, NttTransformDomain::Negacyclic)?,
        |ntt| {
            if constant_time {
                mat_vec_mul_ntt_digits_i8_ct(ntt, row_len, row_width, digit_vectors, log_basis)
            } else {
                mat_vec_mul_ntt_digits_i8(ntt, row_len, row_width, digit_vectors, log_basis)
            }
        },
    )
}
