use super::CpuPreparedSetup;
use crate::sources::block_digits::{DigitShape, I16Rows};
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_params::balanced_signed_digit_abs_bound;
use akita_types::{NttCacheKey, NttTransformDomain};
#[allow(unused_imports)]
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, Field};
use std::ops::Range;

#[tracing::instrument(skip_all, name = "dense_commit_rows_exact_i16")]
pub(super) fn commit_rows_i16<F, S, const D: usize>(
    prepared: &CpuPreparedSetup<F>,
    n_a: usize,
    row_width: usize,
    source: &S,
    block_ranges: &[Range<usize>],
    shape: DigitShape,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
where
    F: Field + CanonicalEncoding,
    S: I16Rows<D>,
{
    let rhs_abs_bound = balanced_signed_digit_abs_bound(shape.log_basis)
        .ok_or_else(|| AkitaError::InvalidSetup("invalid signed digit basis".into()))?;
    prepared.with_shared_ntt::<D, _>(
        NttCacheKey::from_matrix_shape(
            D,
            n_a,
            row_width,
            NttTransformDomain::ExactNegacyclicI16 {
                width: row_width,
                rhs_abs_bound,
            },
        )?,
        |ntt| {
            let fill = |scratch: &mut Option<S::I16Scratch>, rings: &Range<usize>| {
                if rings.is_empty() {
                    return Ok(vec![CyclotomicRing::zero(); n_a]);
                }
                let scratch = scratch
                    .get_or_insert_with(|| source.i16_scratch(shape, row_width / shape.num_digits));
                let rhs = source.i16_rows(rings.clone(), shape, scratch)?;
                ntt.mat_vec_i16::<F>(shape.log_basis, n_a, rhs)
            };
            #[cfg(feature = "parallel")]
            {
                cfg_iter!(block_ranges).map_init(|| None, fill).collect()
            }
            #[cfg(not(feature = "parallel"))]
            {
                let mut rhs = None;
                block_ranges
                    .iter()
                    .map(|rings| fill(&mut rhs, rings))
                    .collect()
            }
        },
    )
}
