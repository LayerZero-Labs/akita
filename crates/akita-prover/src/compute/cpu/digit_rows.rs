use super::prepared::validate_digit_row_request;
use super::CpuBackend;
use crate::compute::backend::DigitRowsComputeBackend;
use crate::kernels::linear::mat_vec_mul_ntt_single_i8;
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_types::{NttCacheKey, NttTransformDomain};

impl<F> DigitRowsComputeBackend<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        row_len: usize,
        digits: &[[i8; D]],
        log_basis: u32,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        validate_digit_row_request(
            row_len,
            digits.len(),
            prepared.expanded.shared_matrix.num_field_elements() / D,
        )?;
        prepared.with_shared_ntt::<D, _>(
            NttCacheKey::from_matrix_shape(
                D,
                row_len,
                digits.len(),
                NttTransformDomain::Negacyclic,
            )?,
            |ntt| mat_vec_mul_ntt_single_i8(ntt, row_len, digits.len(), digits, log_basis),
        )
    }
}
