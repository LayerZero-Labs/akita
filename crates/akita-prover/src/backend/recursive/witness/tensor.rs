//! Tensor extension-opening inputs for compact recursive witnesses.

use super::SuffixWitnessView;
use akita_algebra::SplitEqEvals;
use akita_error::AkitaError;
use akita_types::{tensor_column_partials_split_fold, tensor_opening_split, TensorColumnSource};
#[cfg(feature = "parallel")]
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, ExtField, Field};
use std::marker::PhantomData;

use crate::backend::packed_digits::PackedSignedDigitView;

impl<F, const D: usize> SuffixWitnessView<'_, F, D>
where
    F: Field + CanonicalEncoding,
{
    pub(crate) fn tensor_packed_extension_evals<E>(&self) -> Result<Vec<E>, AkitaError>
    where
        E: ExtField<F>,
    {
        let (split_bits, width) = tensor_opening_split::<F, E>()?;
        let num_vars = self.num_vars();
        if split_bits > num_vars {
            return Err(AkitaError::InvalidInput(
                "extension-opening tensor split exceeds recursive witness arity".to_string(),
            ));
        }
        let table_len = 1usize
            .checked_shl(u32::try_from(num_vars - split_bits).map_err(|_| {
                AkitaError::InvalidInput("recursive tensor table dimension overflow".to_string())
            })?)
            .ok_or_else(|| {
                AkitaError::InvalidInput("recursive tensor table length overflow".to_string())
            })?;
        let pack = |tail: usize| {
            E::from_base_fn(|column| {
                tail.checked_mul(width)
                    .and_then(|start| start.checked_add(column))
                    .and_then(|index| self.digit(index))
                    .map_or_else(F::zero, F::from_i8)
            })
        };
        #[cfg(feature = "parallel")]
        let packed = {
            const PARALLEL_PACK_THRESHOLD: usize = 1 << 14;
            if table_len >= PARALLEL_PACK_THRESHOLD {
                (0..table_len).into_par_iter().map(pack).collect()
            } else {
                (0..table_len).map(pack).collect()
            }
        };
        #[cfg(not(feature = "parallel"))]
        let packed = (0..table_len).map(pack).collect();
        Ok(packed)
    }

    pub(crate) fn tensor_extension_column_partials<E>(
        &self,
        logical_point: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        E: jolt_field::MulBaseUnreduced<F>,
    {
        let (split_bits, width) = tensor_opening_split::<F, E>()?;
        let num_vars = self.num_vars();
        if logical_point.len() != num_vars {
            return Err(AkitaError::InvalidPointDimension {
                expected: num_vars,
                actual: logical_point.len(),
            });
        }
        if split_bits > num_vars {
            return Err(AkitaError::InvalidInput(
                "extension-opening tensor split exceeds recursive witness arity".to_string(),
            ));
        }
        let split = SplitEqEvals::new(&logical_point[split_bits..])?;
        Ok(tensor_column_partials_split_fold::<F, E, _>(
            &split, width, self,
        ))
    }

    pub(crate) fn tensor_extension_column_partials_batch<E>(
        polys: &[&Self],
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: jolt_field::MulBaseUnreduced<F>,
    {
        polys
            .iter()
            .map(|poly| poly.tensor_extension_column_partials(logical_point))
            .collect()
    }
}

#[doc(hidden)]
pub struct SuffixTensorRow<'a, F: Field> {
    digits: PackedSignedDigitView<'a>,
    start: usize,
    width: usize,
    offset: usize,
    _marker: PhantomData<F>,
}

impl<F> Iterator for SuffixTensorRow<'_, F>
where
    F: Field + CanonicalEncoding,
{
    type Item = F;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset == self.width {
            return None;
        }
        let index = self.start.checked_add(self.offset);
        self.offset += 1;
        Some(
            index
                .and_then(|index| self.digits.get(index))
                .map_or_else(F::zero, F::from_i8),
        )
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.width - self.offset;
        (remaining, Some(remaining))
    }
}

impl<F> ExactSizeIterator for SuffixTensorRow<'_, F> where F: Field + CanonicalEncoding {}

impl<F, const D: usize> TensorColumnSource<F> for SuffixWitnessView<'_, F, D>
where
    F: Field + CanonicalEncoding,
{
    type Row<'a>
        = SuffixTensorRow<'a, F>
    where
        Self: 'a;

    #[inline]
    fn row(&self, tail: usize, width: usize) -> Self::Row<'_> {
        SuffixTensorRow {
            digits: self.digits,
            start: tail.saturating_mul(width),
            width,
            offset: 0,
            _marker: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::RecursiveWitnessFlat;
    use jolt_field::{Prime128OffsetA7F7 as F, Ring, Zero};

    #[test]
    fn suffix_tensor_inputs_match_padded_base_table_reference() {
        const D: usize = 16;
        type E = jolt_field::FpExt4<F>;

        let digits = (0..3 * D)
            .map(|index| (index % 7) as i8 - 3)
            .collect::<Vec<_>>();
        let witness = RecursiveWitnessFlat::from_i8_digits(digits.clone());
        let view = witness.view::<F, D>().expect("suffix view");
        let mut base_evals = digits.into_iter().map(F::from_i8).collect::<Vec<_>>();
        base_evals.resize(4 * D, F::zero());

        let packed = view
            .tensor_packed_extension_evals::<E>()
            .expect("direct packed suffix");
        let expected_packed = akita_types::tensor_packed_witness_evals::<F, E>(6, &base_evals)
            .expect("reference packed suffix");
        assert_eq!(packed, expected_packed);

        let point = (0..6)
            .map(|index| {
                E::from_base_fn(|coordinate| F::from_u64((5 * index + coordinate + 2) as u64))
            })
            .collect::<Vec<_>>();
        let partials = view
            .tensor_extension_column_partials::<E>(&point)
            .expect("direct suffix partials");
        let expected_partials =
            akita_types::tensor_column_partials_from_base_evals::<F, E>(6, &base_evals, &point)
                .expect("reference suffix partials");
        assert_eq!(partials, expected_partials);
    }
}
