//! Physical-norm addend for the existing final Stage-1 range leaf.

use super::digit_range::class_indexed_range_leaf::ClassIndexedRangeLeafProver;
use super::digit_range::exact_prefix::ExactPrefixTable;
use jolt_poly::UnivariatePoly;

use akita_error::AkitaError;
use akita_sumcheck::{EqFactoredSumcheckInstanceProver, SumcheckInstanceProver};
use akita_types::{reconstruct_l2_sq_from_gram, PhysicalL2NormProofShape, PhysicalResponsePlan};
use jolt_field::solinas::parallel::*;
use jolt_field::{Field, Ring};
use jolt_field::{Fold, Unreduced};

const RANGE_Q_MAX_DEGREE: usize = 4;
const FUSED_MAX_DEGREE: usize = RANGE_Q_MAX_DEGREE + 1;
const NORM_MAX_DEGREE: usize = 3;

pub(super) enum PhysicalNormTerm<E: Field> {
    Direct {
        response: ExactPrefixTable<E>,
    },
    LimbGram {
        limbs: Vec<ExactPrefixTable<E>>,
        selectors: Vec<ExactPrefixTable<E>>,
        pairs: Vec<(usize, usize)>,
    },
}

impl<E: Field + Fold> PhysicalNormTerm<E> {
    #[inline(always)]
    fn affine_pair(table: &ExactPrefixTable<E>, pair_index: usize) -> (E, E) {
        let left = table.value_or_default(2 * pair_index);
        (left, table.value_or_default(2 * pair_index + 1) - left)
    }

    fn round_coefficients(&self) -> [E; NORM_MAX_DEGREE + 1] {
        match self {
            Self::Direct { response } => cfg_fold_reduce!(
                0..response.explicit_len().div_ceil(2),
                || [E::zero(); NORM_MAX_DEGREE + 1],
                |mut sum, pair_index| {
                    let (value, delta) = Self::affine_pair(response, pair_index);
                    sum[0] += value * value;
                    sum[1] += (value + value) * delta;
                    sum[2] += delta * delta;
                    sum
                },
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        *left += right;
                    }
                    left
                }
            ),
            Self::LimbGram {
                limbs,
                selectors,
                pairs,
            } => {
                let pair_count = limbs
                    .first()
                    .map_or(0, |table| table.explicit_len().div_ceil(2));
                cfg_fold_reduce!(
                    0..pair_count,
                    || [E::zero(); NORM_MAX_DEGREE + 1],
                    |mut sum, pair_index| {
                        for (selector, &(left, right)) in selectors.iter().zip(pairs) {
                            let (selector, selector_delta) =
                                Self::affine_pair(selector, pair_index);
                            let (left, left_delta) = Self::affine_pair(&limbs[left], pair_index);
                            let (right, right_delta) = Self::affine_pair(&limbs[right], pair_index);
                            let product_constant = left * right;
                            let product_linear = left * right_delta + left_delta * right;
                            let product_quadratic = left_delta * right_delta;
                            sum[0] += selector * product_constant;
                            sum[1] += selector * product_linear + selector_delta * product_constant;
                            sum[2] +=
                                selector * product_quadratic + selector_delta * product_linear;
                            sum[3] += selector_delta * product_quadratic;
                        }
                        sum
                    },
                    |mut left, right| {
                        for (left, right) in left.iter_mut().zip(right) {
                            *left += right;
                        }
                        left
                    }
                )
            }
        }
    }

    fn bind(&mut self, challenge: E) -> Result<(), AkitaError> {
        let context = E::precompute(challenge);
        let fold = |left, right| E::fold_one(&context, left, right);
        match self {
            Self::Direct { response } => response.fold_in_place(fold),
            Self::LimbGram {
                limbs, selectors, ..
            } => {
                for table in limbs.iter_mut().chain(selectors) {
                    table.fold_in_place(fold)?;
                }
                Ok(())
            }
        }
    }

    pub(super) fn virtual_evaluations(&self) -> Result<Vec<E>, AkitaError> {
        match self {
            Self::Direct { response } => response
                .final_value()
                .map(|value| vec![value])
                .ok_or(AkitaError::InvalidProof),
            Self::LimbGram { limbs, .. } => limbs
                .iter()
                .map(|table| table.final_value().ok_or(AkitaError::InvalidProof))
                .collect(),
        }
    }

    pub(super) fn final_claim(&self) -> Result<E, AkitaError> {
        match self {
            Self::Direct { response } => {
                let value = response.final_value().ok_or(AkitaError::InvalidProof)?;
                Ok(value * value)
            }
            Self::LimbGram {
                limbs,
                selectors,
                pairs,
            } => selectors.iter().zip(pairs).try_fold(
                E::zero(),
                |sum, (selector, &(left, right))| {
                    let selector = selector.final_value().ok_or(AkitaError::InvalidProof)?;
                    let left = limbs
                        .get(left)
                        .and_then(ExactPrefixTable::final_value)
                        .ok_or(AkitaError::InvalidProof)?;
                    let right = limbs
                        .get(right)
                        .and_then(ExactPrefixTable::final_value)
                        .ok_or(AkitaError::InvalidProof)?;
                    Ok(sum + selector * left * right)
                },
            ),
        }
    }
}

pub(super) struct FusedRangeNormProver<E: Field> {
    pub(super) range: ClassIndexedRangeLeafProver<E>,
    pub(super) norm: PhysicalNormTerm<E>,
    pub(super) norm_merge: E,
    pub(super) input_claim: E,
    pub(super) rounds_completed: usize,
}

impl<E: Field + Ring + Fold + Unreduced> SumcheckInstanceProver<E> for FusedRangeNormProver<E> {
    fn num_rounds(&self) -> usize {
        EqFactoredSumcheckInstanceProver::num_rounds(&self.range)
    }

    fn degree_bound(&self) -> usize {
        EqFactoredSumcheckInstanceProver::degree_bound(&self.range) + 1
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn compute_round_univariate(&mut self, round: usize, _previous_claim: E) -> UnivariatePoly<E> {
        debug_assert_eq!(round, self.rounds_completed);
        let q_coefficients = self.range.round_q_coefficients(round);
        let (factor_at_zero, factor_at_one) = self.range.current_full_eq_factor_evals();
        let factor_delta = factor_at_one - factor_at_zero;
        let mut coefficients = [E::zero(); FUSED_MAX_DEGREE + 1];
        for (degree, coefficient) in q_coefficients
            .into_iter()
            .take(EqFactoredSumcheckInstanceProver::degree_bound(&self.range) + 1)
            .enumerate()
        {
            coefficients[degree] += factor_at_zero * coefficient;
            coefficients[degree + 1] += factor_delta * coefficient;
        }
        for (destination, coefficient) in
            coefficients.iter_mut().zip(self.norm.round_coefficients())
        {
            *destination += self.norm_merge * coefficient;
        }
        UnivariatePoly::new(coefficients[..=self.degree_bound()].to_vec())
    }

    fn ingest_challenge(&mut self, round: usize, challenge: E) {
        debug_assert_eq!(round, self.rounds_completed);
        self.range.ingest_challenge(round, challenge);
        self.norm
            .bind(challenge)
            .expect("validated physical norm prefix can fold");
        self.rounds_completed += 1;
    }
}

pub(super) fn exact_claims<E: Field + Ring>(
    plan: &PhysicalResponsePlan,
    integers: &[Vec<i128>],
) -> Result<(u128, Vec<E>), AkitaError> {
    match plan.shape() {
        PhysicalL2NormProofShape::Direct { .. } => {
            let response = integers.first().ok_or(AkitaError::InvalidProof)?;
            let response_l2_sq = response.iter().try_fold(0u128, |sum, &value| {
                let magnitude = value.unsigned_abs();
                sum.checked_add(magnitude.checked_mul(magnitude).ok_or_else(|| {
                    AkitaError::InvalidInput("physical response square overflow".into())
                })?)
                .ok_or_else(|| AkitaError::InvalidInput("physical response norm overflow".into()))
            })?;
            Ok((response_l2_sq, Vec::new()))
        }
        shape @ PhysicalL2NormProofShape::LimbGram { .. } => {
            let layout = shape.limb_gram_layout()?.ok_or(AkitaError::InvalidProof)?;
            let mut integer_claims = Vec::with_capacity(layout.subclaim_count());
            for block in layout.block_ranges() {
                for (left, right) in layout.limb_pairs() {
                    let left_values = integers.get(left).ok_or(AkitaError::InvalidProof)?;
                    let right_values = integers.get(right).ok_or(AkitaError::InvalidProof)?;
                    let claim = block.clone().try_fold(0i128, |sum, index| {
                        let product = left_values
                            .get(index)
                            .copied()
                            .ok_or(AkitaError::InvalidProof)?
                            .checked_mul(
                                right_values
                                    .get(index)
                                    .copied()
                                    .ok_or(AkitaError::InvalidProof)?,
                            )
                            .ok_or_else(|| {
                                AkitaError::InvalidInput("limb product overflow".into())
                            })?;
                        sum.checked_add(product).ok_or_else(|| {
                            AkitaError::InvalidInput("limb inner product overflow".into())
                        })
                    })?;
                    integer_claims.push(claim);
                }
            }
            let response_l2_sq =
                reconstruct_l2_sq_from_gram(plan.shape(), plan.fold_basis(), &integer_claims)?;
            Ok((
                response_l2_sq,
                integer_claims.into_iter().map(E::from_i128).collect(),
            ))
        }
    }
}

pub(super) fn prepare_norm_term<E: Field + Ring>(
    plan: &PhysicalResponsePlan,
    integers: Vec<Vec<i128>>,
    subclaim_weights: &[E],
) -> Result<PhysicalNormTerm<E>, AkitaError> {
    let domain_len = plan.domain().domain_len();
    let tables = integers
        .into_iter()
        .map(|values| {
            ExactPrefixTable::new(
                domain_len,
                values.into_iter().map(E::from_i128).collect(),
                E::zero(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    match plan.shape() {
        PhysicalL2NormProofShape::Direct { .. } => {
            let mut tables = tables.into_iter();
            let response = tables.next().ok_or(AkitaError::InvalidProof)?;
            if tables.next().is_some() || !subclaim_weights.is_empty() {
                return Err(AkitaError::InvalidProof);
            }
            Ok(PhysicalNormTerm::Direct { response })
        }
        shape @ PhysicalL2NormProofShape::LimbGram { .. } => {
            let layout = shape.limb_gram_layout()?.ok_or(AkitaError::InvalidProof)?;
            if subclaim_weights.len() != layout.subclaim_count() {
                return Err(AkitaError::InvalidSize {
                    expected: layout.subclaim_count(),
                    actual: subclaim_weights.len(),
                });
            }
            let mut selectors = Vec::with_capacity(layout.pair_count());
            for (left, right) in layout.limb_pairs() {
                let mut values = Vec::with_capacity(layout.physical_response_len());
                for (block_index, block) in layout.block_ranges().enumerate() {
                    let weight_index = layout
                        .subclaim_index(block_index, left, right)
                        .ok_or(AkitaError::InvalidProof)?;
                    let weight = *subclaim_weights
                        .get(weight_index)
                        .ok_or(AkitaError::InvalidProof)?;
                    values.resize(block.end, weight);
                }
                selectors.push(ExactPrefixTable::new(domain_len, values, E::zero())?);
            }
            Ok(PhysicalNormTerm::LimbGram {
                limbs: tables,
                selectors,
                pairs: layout.limb_pairs().collect(),
            })
        }
    }
}
