use super::prepared::validate_digit_row_request;
use super::CpuBackend;
use crate::compute::backend::DigitRowsComputeBackend;
use crate::kernels::linear::mat_vec_mul_ntt_digits_i8;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_field::{CanonicalField, FieldCore};
use akita_types::{NttCacheKey, NttTransformDomain};

impl<F> DigitRowsComputeBackend<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        row_len: usize,
        digit_vectors: &[&[[i8; D]]],
        log_basis: u32,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
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
            |ntt| mat_vec_mul_ntt_digits_i8(ntt, row_len, row_width, digit_vectors, log_basis),
        )
    }
}
