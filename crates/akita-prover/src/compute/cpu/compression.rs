use super::compression_cache::CompressionNttDomains;
use super::prepared::validate_digit_row_request;
use super::CpuBackend;
use crate::compute::backend::{CompressionComputeBackend, CompressionRowsProducts};
use crate::kernels::linear::{
    mat_vec_mul_ntt_digits_i8, mat_vec_mul_ntt_single_i8_cyclic, validate_compression_batch_shape,
};
use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, Field};

impl<F> CompressionComputeBackend<F> for CpuBackend
where
    F: Field + CanonicalEncoding,
{
    fn compression_cache_bytes(&self, prepared: &Self::PreparedSetup) -> Option<usize> {
        Some(prepared.compression_ntt_cache_bytes())
    }

    fn compression_rows_products<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        digit_vectors: &[&[[i8; D]]],
    ) -> Result<Vec<CompressionRowsProducts<F, D>>, AkitaError> {
        let input_width = validate_compression_batch_shape(digit_vectors)?;
        let total_ring_elements = prepared.expanded.shared_matrix.num_field_elements() / D;
        validate_digit_row_request(1, input_width, total_ring_elements)?;
        prepared.with_compression_ntt::<D, _>(input_width, CompressionNttDomains::Both, |ntt| {
            let negacyclic = mat_vec_mul_ntt_digits_i8(ntt, 1, input_width, digit_vectors, 1)?;
            let cyclic = digit_vectors
                .iter()
                .map(|digits| mat_vec_mul_ntt_single_i8_cyclic(ntt, 1, input_width, digits, 1))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(negacyclic
                .into_iter()
                .zip(cyclic)
                .map(|(negacyclic, cyclic)| CompressionRowsProducts { negacyclic, cyclic })
                .collect())
        })
    }

    fn compression_negacyclic_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        digit_vectors: &[&[[i8; D]]],
    ) -> Result<Vec<Vec<akita_algebra::CyclotomicRing<F, D>>>, AkitaError> {
        let input_width = validate_compression_batch_shape(digit_vectors)?;
        let total_ring_elements = prepared.expanded.shared_matrix.num_field_elements() / D;
        validate_digit_row_request(1, input_width, total_ring_elements)?;
        prepared.with_compression_ntt::<D, _>(
            input_width,
            CompressionNttDomains::Negacyclic,
            |ntt| mat_vec_mul_ntt_digits_i8(ntt, 1, input_width, digit_vectors, 1),
        )
    }
}
