//! Tensor extension-opening inputs for compact recursive witnesses.

use super::SuffixWitnessView;
use akita_algebra::SplitEqEvals;
use akita_error::AkitaError;
#[cfg(feature = "parallel")]
use akita_field::parallel::*;
use akita_field::{CanonicalField, ExtField, FieldCore};
use akita_types::{tensor_column_partials_split_fold, tensor_opening_split, TensorColumnSource};

impl<F, const D: usize> SuffixWitnessView<'_, F, D>
where
    F: FieldCore + CanonicalField,
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
        let flat = self.coeffs.as_flattened();
        let pack = |tail| {
            E::from_base_fn(|column| {
                flat.get(tail * width + column)
                    .copied()
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
        E: akita_field::MulBaseUnreduced<F>,
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
        E: akita_field::MulBaseUnreduced<F>,
    {
        polys
            .iter()
            .map(|poly| poly.tensor_extension_column_partials(logical_point))
            .collect()
    }
}

type SuffixTensorRow<'a, F> = std::iter::Chain<
    std::iter::Map<std::iter::Copied<std::slice::Iter<'a, i8>>, fn(i8) -> F>,
    std::iter::RepeatN<F>,
>;

impl<F, const D: usize> TensorColumnSource<F> for SuffixWitnessView<'_, F, D>
where
    F: FieldCore + CanonicalField,
{
    type Row<'a>
        = SuffixTensorRow<'a, F>
    where
        Self: 'a;

    #[inline]
    fn row(&self, tail: usize, width: usize) -> Self::Row<'_> {
        let flat = self.coeffs.as_flattened();
        let start = tail * width;
        let end = start.saturating_add(width).min(flat.len());
        let live = flat.get(start..end).unwrap_or_default();
        live.iter()
            .copied()
            .map(F::from_i8 as fn(i8) -> F)
            .chain(std::iter::repeat_n(F::zero(), width - live.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::RecursiveWitnessFlat;
    use akita_field::Prime128OffsetA7F7 as F;

    #[test]
    fn suffix_tensor_inputs_match_padded_base_table_reference() {
        const D: usize = 16;
        type E = akita_field::FpExt4<F>;

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
