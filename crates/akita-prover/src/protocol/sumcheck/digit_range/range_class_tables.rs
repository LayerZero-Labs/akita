use super::compact_digit_source::RangeImageClass;
use super::MAX_TREE_STAGE_Q_DEGREE;
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt};
use akita_types::DigitRangePlan;

/// Plan-derived child-node values for every range-image class.
pub(super) struct ProductNodeTable<E: FieldCore, const LANES: usize> {
    rows: Vec<[E; LANES]>,
}

impl<E: FieldCore + FromPrimitiveInt, const LANES: usize> ProductNodeTable<E, LANES> {
    pub(super) fn new(
        plan: DigitRangePlan,
        leaf_polynomials: &[Vec<E>],
        stage_index: usize,
    ) -> Result<Self, AkitaError> {
        let arity = plan
            .product_stage_arities()
            .get(stage_index)
            .copied()
            .ok_or(AkitaError::InvalidProof)?;
        let parent_count = plan.product_stage_arities()[..stage_index]
            .iter()
            .copied()
            .product::<usize>();
        let expected_lanes = parent_count.checked_mul(arity).ok_or_else(|| {
            AkitaError::InvalidInput("range-product lane count overflow".to_string())
        })?;
        if LANES != expected_lanes {
            return Err(AkitaError::InvalidSize {
                expected: expected_lanes,
                actual: LANES,
            });
        }
        if !leaf_polynomials.len().is_multiple_of(LANES) {
            return Err(AkitaError::InvalidSize {
                expected: LANES,
                actual: leaf_polynomials.len(),
            });
        }
        let leaves_per_lane = leaf_polynomials.len() / LANES;
        let rows = (0..plan.basis() / 2)
            .map(|class_index| {
                let class = RangeImageClass::from_balanced_digit(
                    i8::try_from(class_index).expect("supported range class fits i8"),
                    plan.basis() / 2,
                );
                let range_image = class.range_image::<E>();
                std::array::from_fn(|lane| {
                    let first_leaf = lane * leaves_per_lane;
                    leaf_polynomials[first_leaf..first_leaf + leaves_per_lane]
                        .iter()
                        .fold(E::one(), |node_value, polynomial| {
                            node_value * plan.evaluate_leaf_polynomial(polynomial, range_image)
                        })
                })
            })
            .collect();
        Ok(Self { rows })
    }

    #[inline(always)]
    pub(super) fn row(&self, class: RangeImageClass) -> [E; LANES] {
        self.rows[class.index()]
    }

    pub(super) fn padding_row(&self) -> [E; LANES] {
        self.rows[RangeImageClass::PADDING.index()]
    }
}

pub(super) fn product_coefficients<E: FieldCore, const LANES: usize>(
    left: [E; LANES],
    right: [E; LANES],
    arity: usize,
    parent_weights: &[E],
) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] {
    debug_assert_eq!(LANES, arity * parent_weights.len());
    debug_assert!(arity <= MAX_TREE_STAGE_Q_DEGREE);
    let mut batched = [E::zero(); MAX_TREE_STAGE_Q_DEGREE + 1];
    for (parent_index, &weight) in parent_weights.iter().enumerate() {
        let mut polynomial = [E::zero(); MAX_TREE_STAGE_Q_DEGREE + 1];
        polynomial[0] = E::one();
        for child_index in 0..arity {
            let lane = parent_index * arity + child_index;
            let offset = left[lane];
            let slope = right[lane] - offset;
            for degree in (0..=child_index).rev() {
                polynomial[degree + 1] += polynomial[degree] * slope;
                polynomial[degree] *= offset;
            }
        }
        for degree in 0..=arity {
            batched[degree] += weight * polynomial[degree];
        }
    }
    batched
}

#[cfg(test)]
pub(super) struct OrderedProductPairCoefficients<E: FieldCore> {
    class_count: usize,
    rows: Vec<[E; MAX_TREE_STAGE_Q_DEGREE + 1]>,
}

#[cfg(test)]
impl<E: FieldCore + FromPrimitiveInt> OrderedProductPairCoefficients<E> {
    pub(super) fn new<const LANES: usize>(
        nodes: &ProductNodeTable<E, LANES>,
        class_count: usize,
        arity: usize,
        parent_weights: &[E],
    ) -> Self {
        let rows = (0..class_count * class_count)
            .map(|pair_index| {
                let left_class = RangeImageClass::from_balanced_digit(
                    i8::try_from(pair_index / class_count).unwrap(),
                    class_count,
                );
                let right_class = RangeImageClass::from_balanced_digit(
                    i8::try_from(pair_index % class_count).unwrap(),
                    class_count,
                );
                product_coefficients(
                    nodes.row(left_class),
                    nodes.row(right_class),
                    arity,
                    parent_weights,
                )
            })
            .collect();
        Self { class_count, rows }
    }

    pub(super) fn coefficients(
        &self,
        left: RangeImageClass,
        right: RangeImageClass,
    ) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] {
        self.rows[left.index() * self.class_count + right.index()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128Offset275;

    type F = Prime128Offset275;

    #[test]
    fn ordered_pairs_match_node_evaluation_for_every_high_basis_substage() {
        for basis in [16, 32, 64] {
            let plan = DigitRangePlan::new(basis).unwrap();
            let leaf_polynomials = plan.leaf_coeffs::<F>();
            for (stage_index, &arity) in plan.product_stage_arities().iter().enumerate() {
                let parent_count = plan.product_stage_arities()[..stage_index]
                    .iter()
                    .product::<usize>();
                let weights = plan.interstage_batch_weights(F::from_u64(7), parent_count);
                macro_rules! check_lanes {
                    ($lanes:literal) => {{
                        let nodes = ProductNodeTable::<F, $lanes>::new(
                            plan,
                            &leaf_polynomials,
                            stage_index,
                        )
                        .unwrap();
                        let pairs =
                            OrderedProductPairCoefficients::new(&nodes, basis / 2, arity, &weights);
                        for left_index in 0..basis / 2 {
                            for right_index in 0..basis / 2 {
                                let left = RangeImageClass::from_balanced_digit(
                                    i8::try_from(left_index).unwrap(),
                                    basis / 2,
                                );
                                let right = RangeImageClass::from_balanced_digit(
                                    i8::try_from(right_index).unwrap(),
                                    basis / 2,
                                );
                                assert_eq!(
                                    pairs.coefficients(left, right),
                                    product_coefficients(
                                        nodes.row(left),
                                        nodes.row(right),
                                        arity,
                                        &weights,
                                    )
                                );
                            }
                        }
                    }};
                }
                match parent_count * arity {
                    2 => check_lanes!(2),
                    4 => check_lanes!(4),
                    8 => check_lanes!(8),
                    lanes => panic!("unexpected test lane count {lanes}"),
                }
            }
        }
    }
}
