//! Setup-linear compression functionals in the existing native-query representation.
use super::*;
use crate::setup_claims::{NativeTerm, SetupQuery};
use akita_algebra::offset_eq::{EqPairTensorAxis, EqPairTensorFamily};
use akita_algebra::ring::ResidueKernelPoint;

#[derive(Clone, Debug)]
pub(super) struct CompressionSetupMap<E: Field> {
    pub(super) span: CompressionWitnessSpan,
    pub(super) row_weight: E,
}

#[derive(Clone, Copy)]
pub(super) enum CompressionSetupMode {
    LiftedPower,
    ReducedEvaluation,
}

/// A shifted physical column meets at most two aligned coefficient blocks.
/// Transpose its low-coordinate functional once for each block.
pub(super) fn coefficient_kernels<E: Field>(
    alpha: E,
    low_point: &[E],
    low_offset: usize,
    mode: CompressionSetupMode,
) -> Result<Vec<Vec<E>>, AkitaError> {
    let equality = EqPolynomial::evals(low_point)?;
    let dimension = equality.len();
    if low_offset >= dimension {
        return Err(AkitaError::InvalidProof);
    }
    let first_len = dimension - low_offset;
    let mut windows = vec![vec![E::zero(); dimension]];
    windows
        .first_mut()
        .and_then(|w| w.get_mut(..first_len))
        .ok_or(AkitaError::InvalidProof)?
        .copy_from_slice(equality.get(low_offset..).ok_or(AkitaError::InvalidProof)?);
    if low_offset != 0 {
        let mut second = vec![E::zero(); dimension];
        second
            .get_mut(first_len..)
            .ok_or(AkitaError::InvalidProof)?
            .copy_from_slice(equality.get(..low_offset).ok_or(AkitaError::InvalidProof)?);
        windows.push(second);
    }
    match mode {
        CompressionSetupMode::ReducedEvaluation => {
            let residue = ResidueKernelPoint::new(alpha, dimension)?;
            windows
                .iter()
                .map(|window| residue.field_kernel(window))
                .collect()
        }
        CompressionSetupMode::LiftedPower => {
            let powers = scalar_powers(alpha, dimension);
            Ok(windows
                .iter()
                .map(|window| {
                    let scale = E::dot_product(&powers, window);
                    powers.iter().map(|&power| scale * power).collect()
                })
                .collect())
        }
    }
}

pub(super) fn compression_setup_query<E: Field>(
    alpha: E,
    point: &[E],
    physical_field_len: usize,
    mode: CompressionSetupMode,
    maps: &[CompressionSetupMap<E>],
) -> Result<SetupQuery<E>, AkitaError> {
    if checked::pow2(point.len()) != Some(physical_field_len) || maps.is_empty() {
        return Err(AkitaError::InvalidProof);
    }
    let _scope = akita_error::experiment_scope("compression/setup_query");
    let mut terms = Vec::new();
    for event in maps {
        let map = event.span.map();
        let dimension = map.ring_dimension();
        let span = event.span.range();
        if !dimension.is_power_of_two()
            || map.input_width() == 0
            || map.input_width().checked_mul(dimension) != Some(span.len())
            || span.end > physical_field_len
        {
            return Err(AkitaError::InvalidSetup(
                "compression query geometry is malformed".into(),
            ));
        }
        let bits = dimension.trailing_zeros() as usize;
        let (low, high) = point
            .split_at_checked(bits)
            .ok_or(AkitaError::InvalidProof)?;
        let kernels = coefficient_kernels(alpha, low, span.start % dimension, mode)?;
        terms
            .try_reserve(kernels.len())
            .map_err(|_| AkitaError::InvalidInput("compression query allocation failed".into()))?;
        for (carry, coefficient) in kernels.into_iter().enumerate() {
            let offset = (span.start / dimension)
                .checked_add(carry)
                .ok_or(AkitaError::InvalidProof)?;
            terms.push(NativeTerm {
                coefficient,
                point: high.to_vec(),
                tensors: vec![EqPairTensorFamily::new(
                    0,
                    offset,
                    event.row_weight,
                    vec![EqPairTensorAxis::unit(map.input_width(), 1, 1)],
                )?],
            });
        }
    }
    Ok(SetupQuery::from_native_terms(terms))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup_claims::{SetupBatchStatement, SetupClaim, SetupPolynomialId};
    use crate::{derive_public_matrix_prefix, AkitaSetupDescriptor};
    use akita_algebra::offset_eq::eq_eval_at_index;
    use akita_params::{CompressionMapPlan, SisModulusProfileId};
    use jolt_field::{Ext2, One, Prime128OffsetA7F7 as F, Zero};
    type E = Ext2<F>;

    #[test]
    fn compression_query_is_the_literal_setup_dot_product() {
        let seed = [27; 32].into();
        let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            AkitaSetupDescriptor {
                max_num_vars: 8,
                max_num_batched_polys: 1,
                num_field_elements: 256,
                setup_seed: seed,
            },
            derive_public_matrix_prefix::<F>(256, &[27; 32].into()),
        );
        let polynomial = SetupPolynomialId::new::<F, E>(&setup, 256).unwrap();
        let dimension = 16;
        let map =
            CompressionMapPlan::new(SisModulusProfileId::Q128OffsetA7F7, 1, dimension, 1).unwrap();
        let width = map.input_width();
        let source = setup.shared_matrix().as_field_slice();
        let point = (0..9)
            .map(|i| E::from_base_fn(|c| F::from_u64((i + 3 + 7 * c) as u64)))
            .collect::<Vec<_>>();
        for start in [0, 5] {
            let span = CompressionWitnessSpan::new_for_test(map, start..start + width * dimension);
            let weight = E::from_u64(13);
            for alpha in [
                E::zero(),
                E::one(),
                E::from_base_fn(|c| F::from_u64((19 + 5 * c) as u64)),
            ] {
                let powers = scalar_powers(alpha, dimension);
                for mode in [
                    CompressionSetupMode::LiftedPower,
                    CompressionSetupMode::ReducedEvaluation,
                ] {
                    let query = compression_setup_query(
                        alpha,
                        &point,
                        512,
                        mode,
                        &[CompressionSetupMap {
                            span: span.clone(),
                            row_weight: weight,
                        }],
                    )
                    .unwrap();
                    let mut expected = E::zero();
                    for column in 0..width {
                        for digit in 0..dimension {
                            let physical =
                                eq_eval_at_index(&point, start + column * dimension + digit);
                            for coefficient in 0..dimension {
                                let power = match mode {
                                    CompressionSetupMode::LiftedPower => {
                                        powers[coefficient] * powers[digit]
                                    }
                                    CompressionSetupMode::ReducedEvaluation => {
                                        let degree = coefficient + digit;
                                        if degree < dimension {
                                            powers[degree]
                                        } else {
                                            -powers[degree - dimension]
                                        }
                                    }
                                };
                                expected += weight
                                    * E::lift_base(source[column * dimension + coefficient])
                                    * power
                                    * physical;
                            }
                        }
                    }
                    assert_eq!(
                        query.evaluate_materialized(&polynomial, &setup).unwrap(),
                        expected
                    );
                    for split in [2, 32] {
                        let compiled = query.compile(&polynomial, split, None).unwrap();
                        let compiled_value = compiled.iter().fold(E::zero(), |sum, t| {
                            sum + (0..t.active_index_len()).fold(E::zero(), |v, row| {
                                v + t.index_weight(row)
                                    * E::dot_product(
                                        t.coefficient(),
                                        &source[row * split..row * split + split]
                                            .iter()
                                            .copied()
                                            .map(E::lift_base)
                                            .collect::<Vec<_>>(),
                                    )
                            })
                        });
                        assert_eq!(compiled_value, expected);
                    }
                    let statement = SetupBatchStatement::new(
                        polynomial.clone(),
                        2,
                        vec![SetupClaim::new(polynomial.clone(), query, expected).unwrap()],
                    )
                    .unwrap();
                    let bytes = statement.descriptor_bytes::<F>().unwrap();
                    let decoded =
                        SetupBatchStatement::<E>::from_descriptor_bytes::<F>(&bytes, &polynomial)
                            .unwrap();
                    assert_eq!(
                        decoded.claims()[0]
                            .query()
                            .evaluate_materialized(&polynomial, &setup)
                            .unwrap(),
                        expected
                    );
                }
            }
        }
    }
}
