use super::class_indexed_product::ProductArity;
use super::compact_digit_source::RangeImageClass;
use super::{compose_small_poly_with_affine, SmallPoly, MAX_TREE_STAGE_Q_DEGREE};
use akita_error::AkitaError;
use akita_params::DigitRangePlan;
use jolt_field::Fold;
use jolt_field::{Field, Ring};

/// Plan-derived child-node values for every range-image class.
pub(super) struct ProductNodeTable<E: Field, const LANES: usize> {
    rows: Vec<[E; LANES]>,
}

impl<E: Field + Ring, const LANES: usize> ProductNodeTable<E, LANES> {
    pub(super) fn new(
        plan: DigitRangePlan,
        leaf_polynomials: &[Vec<E>],
        stage_index: usize,
    ) -> Result<Self, AkitaError> {
        let expected_lanes = plan.product_stage_lane_count(stage_index).ok_or_else(|| {
            AkitaError::Internal("digit-range class table stage has no planned lane count".into())
        })?;
        if LANES != expected_lanes {
            return Err(AkitaError::Internal(format!(
                "range-class implementation lane count: expected {expected_lanes}, actual {LANES}"
            )));
        }
        if leaf_polynomials.len() != plan.leaf_factor_count() {
            return Err(AkitaError::Internal(format!(
                "range-class leaf polynomial count: expected {}, actual {}",
                plan.leaf_factor_count(),
                leaf_polynomials.len(),
            )));
        }
        let leaves_per_lane = leaf_polynomials.len() / LANES;
        // DigitRangePlan admits log bases 2..=6, so the class count is 2..=32.
        let class_count = 1u8 << (plan.log_basis() - 1);
        let rows = (0..class_count)
            .map(|class_index| {
                let class = RangeImageClass(class_index);
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
    #[cfg(test)]
    pub(super) fn row(&self, class: RangeImageClass) -> [E; LANES] {
        self.rows[class.index()]
    }
}

/// Child-node rows folded at the first challenge, indexed by an ordered class pair.
pub(super) struct FoldedProductPairTable<E: Field, const LANES: usize> {
    rows: Vec<[E; LANES]>,
    ordered_pair_count: usize,
}

/// Round-one product coefficients indexed by two first-challenge-folded class pairs.
pub(super) struct SecondRoundProductQuartetCoefficients<E: Field> {
    rows: Vec<[E; MAX_TREE_STAGE_Q_DEGREE + 1]>,
    ordered_pair_count: usize,
}

impl<E: Field + Ring + Fold> SecondRoundProductQuartetCoefficients<E> {
    pub(super) fn new<const LANES: usize>(
        folded_pairs: &FoldedProductPairTable<E, LANES>,
        arity: ProductArity,
        parent_weights: &[E],
    ) -> Self {
        let ordered_pair_count = folded_pairs.ordered_pair_count;
        let rows = (0..ordered_pair_count * ordered_pair_count)
            .map(|quartet_index| {
                product_coefficients(
                    folded_pairs.row_by_pair_index(quartet_index / ordered_pair_count),
                    folded_pairs.row_by_pair_index(quartet_index % ordered_pair_count),
                    arity,
                    parent_weights,
                )
            })
            .collect();
        Self {
            rows,
            ordered_pair_count,
        }
    }

    #[inline(always)]
    pub(super) fn coefficients_by_pair_indices(
        &self,
        left_pair_index: usize,
        right_pair_index: usize,
    ) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] {
        self.rows[left_pair_index * self.ordered_pair_count + right_pair_index]
    }
}

impl<E: Field + Ring + Fold, const LANES: usize> FoldedProductPairTable<E, LANES> {
    pub(super) fn new(nodes: &ProductNodeTable<E, LANES>, challenge: E) -> Self {
        let class_count = nodes.rows.len();
        let fold_context = E::precompute(challenge);
        let mut rows = Vec::with_capacity(class_count * class_count);
        rows.extend(nodes.rows.iter().flat_map(|left| {
            nodes.rows.iter().map(move |right| {
                std::array::from_fn(|lane| E::fold_one(&fold_context, left[lane], right[lane]))
            })
        }));
        Self {
            rows,
            ordered_pair_count: class_count * class_count,
        }
    }

    #[inline(always)]
    pub(super) fn row_by_pair_index(&self, pair_index: usize) -> [E; LANES] {
        self.rows[pair_index]
    }
}

/// Range-image values folded at the first challenge, indexed by an ordered class pair.
pub(super) struct FoldedRangeImagePairTable<E: Field> {
    values: Vec<E>,
    ordered_pair_count: usize,
}

/// Round-one range-polynomial coefficients indexed by two folded class pairs.
pub(super) struct SecondRoundRangeQuartetCoefficients<E: Field> {
    rows: Vec<[E; MAX_TREE_STAGE_Q_DEGREE + 1]>,
    ordered_pair_count: usize,
}

impl<E: Field + Ring + Fold> SecondRoundRangeQuartetCoefficients<E> {
    pub(super) fn new(
        folded_pairs: &FoldedRangeImagePairTable<E>,
        polynomial_coefficients: &SmallPoly<E>,
    ) -> Self {
        let ordered_pair_count = folded_pairs.ordered_pair_count;
        let rows = (0..ordered_pair_count * ordered_pair_count)
            .map(|quartet_index| {
                let left = folded_pairs.value_by_pair_index(quartet_index / ordered_pair_count);
                let right = folded_pairs.value_by_pair_index(quartet_index % ordered_pair_count);
                compose_small_poly_with_affine(polynomial_coefficients, left, right - left)
            })
            .collect();
        Self {
            rows,
            ordered_pair_count,
        }
    }

    #[inline(always)]
    pub(super) fn coefficients_by_pair_indices(
        &self,
        left_pair_index: usize,
        right_pair_index: usize,
    ) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] {
        self.rows[left_pair_index * self.ordered_pair_count + right_pair_index]
    }
}

impl<E: Field + Ring + Fold> FoldedRangeImagePairTable<E> {
    pub(super) fn new(classes: std::ops::Range<u8>, challenge: E) -> Self {
        let class_count = classes.len();
        let fold_context = E::precompute(challenge);
        let mut values = Vec::with_capacity(class_count * class_count);
        values.extend(classes.clone().flat_map(|left| {
            let left = RangeImageClass(left).range_image();
            classes.clone().map(move |right| {
                E::fold_one(&fold_context, left, RangeImageClass(right).range_image())
            })
        }));
        Self {
            values,
            ordered_pair_count: class_count * class_count,
        }
    }

    #[inline(always)]
    pub(super) fn value_by_pair_index(&self, pair_index: usize) -> E {
        self.values[pair_index]
    }
}

/// Batched round coefficients `sum_p w_p prod_c (left_c + X (right_c - left_c))`
/// over the `arity` child lanes of each parent `p`.
///
/// The first interstage weight is always one, so only later parents pay for
/// their weight, and they pay on a quadratic half-product rather than on the
/// full product.
#[inline]
pub(super) fn product_coefficients<E: Field, const LANES: usize>(
    left: [E; LANES],
    right: [E; LANES],
    arity: ProductArity,
    parent_weights: &[E],
) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] {
    debug_assert_eq!(LANES, arity.degree() * parent_weights.len());
    let mut batched = [E::zero(); MAX_TREE_STAGE_Q_DEGREE + 1];
    for (parent_index, &weight) in parent_weights.iter().enumerate() {
        let first_lane = parent_index * arity.degree();
        let mut head = quadratic_affine_product(
            [left[first_lane], left[first_lane + 1]],
            [right[first_lane], right[first_lane + 1]],
        );
        if weight != E::one() {
            head = head.map(|coefficient| weight * coefficient);
        }
        match arity {
            ProductArity::Two => {
                for (destination, source) in batched.iter_mut().zip(head) {
                    *destination += source;
                }
            }
            ProductArity::Four => {
                let tail = quadratic_affine_product(
                    [left[first_lane + 2], left[first_lane + 3]],
                    [right[first_lane + 2], right[first_lane + 3]],
                );
                for (destination, source) in batched.iter_mut().zip(quadratic_product(head, tail)) {
                    *destination += source;
                }
            }
        }
    }
    batched
}

/// Coefficients of `(l0 + X s0)(l1 + X s1)` with `s = right - left`, using
/// `(l0 + s0)(l1 + s1) = right0 * right1` for the middle term.
#[inline(always)]
fn quadratic_affine_product<E: Field>(left: [E; 2], right: [E; 2]) -> [E; 3] {
    let constant = left[0] * left[1];
    let leading = (right[0] - left[0]) * (right[1] - left[1]);
    [constant, right[0] * right[1] - constant - leading, leading]
}

/// Product of two quadratics with six multiplications (Karatsuba).
#[inline(always)]
fn quadratic_product<E: Field>(first: [E; 3], second: [E; 3]) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] {
    let low = first[0] * second[0];
    let middle = first[1] * second[1];
    let high = first[2] * second[2];
    let low_middle = (first[0] + first[1]) * (second[0] + second[1]);
    let middle_high = (first[1] + first[2]) * (second[1] + second[2]);
    let low_high = (first[0] + first[2]) * (second[0] + second[2]);
    [
        low,
        low_middle - low - middle,
        low_high - low - high + middle,
        middle_high - middle - high,
        high,
    ]
}

pub(super) struct OrderedProductPairCoefficients<E: Field> {
    rows: Vec<[E; MAX_TREE_STAGE_Q_DEGREE + 1]>,
}

impl<E: Field + Ring> OrderedProductPairCoefficients<E> {
    pub(super) fn new<const LANES: usize>(
        nodes: &ProductNodeTable<E, LANES>,
        arity: ProductArity,
        parent_weights: &[E],
    ) -> Self {
        let class_count = nodes.rows.len();
        let mut rows = Vec::with_capacity(class_count * class_count);
        rows.extend(nodes.rows.iter().flat_map(|&left| {
            nodes
                .rows
                .iter()
                .map(move |&right| product_coefficients(left, right, arity, parent_weights))
        }));
        Self { rows }
    }

    #[inline(always)]
    pub(super) fn coefficients_by_pair_index(
        &self,
        pair_index: usize,
    ) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] {
        self.rows[pair_index]
    }
}

/// Complete round-zero polynomial coefficients indexed by two range-image classes.
pub(super) struct OrderedRangePairCoefficients<E: Field> {
    rows: Vec<[E; MAX_TREE_STAGE_Q_DEGREE + 1]>,
}

impl<E: Field + Ring> OrderedRangePairCoefficients<E> {
    pub(super) fn new(
        classes: std::ops::Range<u8>,
        polynomial_coefficients: &SmallPoly<E>,
    ) -> Self {
        let class_count = classes.len();
        let mut rows = Vec::with_capacity(class_count * class_count);
        rows.extend(classes.clone().flat_map(|left| {
            let left = RangeImageClass(left).range_image::<E>();
            classes.clone().map(move |right| {
                let right = RangeImageClass(right).range_image::<E>();
                compose_small_poly_with_affine(polynomial_coefficients, left, right - left)
            })
        }));
        Self { rows }
    }

    #[inline(always)]
    pub(super) fn coefficients_by_pair_index(
        &self,
        pair_index: usize,
    ) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] {
        self.rows[pair_index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{One, Prime128Offset275};

    type F = Prime128Offset275;

    fn product_coefficients_reference<E: Field, const LANES: usize>(
        left: [E; LANES],
        right: [E; LANES],
        arity: usize,
        parent_weights: &[E],
    ) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] {
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

    fn compose_affine_reference<E: Field>(coefficients: &[E], offset: E, slope: E) -> [E; 5] {
        let mut output = [E::zero(); 5];
        let mut power = [E::zero(); 5];
        power[0] = E::one();
        for (index, &coefficient) in coefficients.iter().enumerate() {
            if index > 0 {
                for degree in (0..index).rev() {
                    power[degree + 1] += power[degree] * slope;
                    power[degree] *= offset;
                }
            }
            for degree in 0..=index {
                output[degree] += coefficient * power[degree];
            }
        }
        output
    }

    #[test]
    fn ordered_pairs_match_node_evaluation_for_every_high_basis_substage() {
        let challenge = F::from_u64(13);
        let fold_context = F::precompute(challenge);
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
                        let pairs = OrderedProductPairCoefficients::new(
                            &nodes,
                            ProductArity::new(arity).unwrap(),
                            &weights,
                        );
                        let folded_pairs = FoldedProductPairTable::new(&nodes, challenge);
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
                                let pair_index = left_index * (basis / 2) + right_index;
                                assert_eq!(
                                    pairs.coefficients_by_pair_index(pair_index),
                                    product_coefficients_reference(
                                        nodes.row(left),
                                        nodes.row(right),
                                        arity,
                                        &weights,
                                    )
                                );
                                let left_row = nodes.row(left);
                                let right_row = nodes.row(right);
                                assert_eq!(
                                    folded_pairs.row_by_pair_index(pair_index),
                                    std::array::from_fn(|lane| F::fold_one(
                                        &fold_context,
                                        left_row[lane],
                                        right_row[lane],
                                    ))
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

    #[test]
    fn ordered_range_pairs_match_direct_affine_composition() {
        let challenge = F::from_u64(13);
        let fold_context = F::precompute(challenge);
        for basis in [16, 32, 64] {
            let plan = DigitRangePlan::new(basis).unwrap();
            let coefficients = plan
                .batch_leaf_polynomials(
                    &plan.interstage_batch_weights(F::from_u64(7), plan.leaf_factor_count()),
                    &plan.leaf_coeffs::<F>(),
                )
                .unwrap();
            let polynomial = SmallPoly::new(&coefficients).unwrap();
            let pairs = OrderedRangePairCoefficients::new(0..(basis / 2) as u8, &polynomial);
            let folded_pairs = FoldedRangeImagePairTable::<F>::new(0..(basis / 2) as u8, challenge);
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
                    let pair_index = left_index * (basis / 2) + right_index;
                    let left_value = left.range_image::<F>();
                    let right_value = right.range_image::<F>();
                    assert_eq!(
                        pairs.coefficients_by_pair_index(pair_index),
                        compose_affine_reference(
                            &coefficients,
                            left_value,
                            right_value - left_value
                        )
                    );
                    assert_eq!(
                        folded_pairs.value_by_pair_index(pair_index),
                        F::fold_one(&fold_context, left_value, right_value),
                    );
                }
            }
        }
    }

    #[test]
    fn second_round_quartet_tables_match_folded_pair_evaluation() {
        let plan = DigitRangePlan::new(16).unwrap();
        let class_count = plan.basis() / 2;
        let ordered_pair_count = class_count * class_count;
        let challenge = F::from_u64(13);
        let leaf_polynomials = plan.leaf_coeffs::<F>();
        let parent_weights = vec![F::one()];
        let nodes = ProductNodeTable::<F, 2>::new(plan, &leaf_polynomials, 0).unwrap();
        let folded_products = FoldedProductPairTable::new(&nodes, challenge);
        let product_quartets = SecondRoundProductQuartetCoefficients::new(
            &folded_products,
            ProductArity::Two,
            &parent_weights,
        );

        let range_coefficients = plan
            .batch_leaf_polynomials(
                &plan.interstage_batch_weights(F::from_u64(7), plan.leaf_factor_count()),
                &leaf_polynomials,
            )
            .unwrap();
        let folded_ranges = FoldedRangeImagePairTable::<F>::new(0..class_count as u8, challenge);
        let range_quartets = SecondRoundRangeQuartetCoefficients::new(
            &folded_ranges,
            &SmallPoly::new(&range_coefficients).unwrap(),
        );

        for left_pair in 0..ordered_pair_count {
            for right_pair in 0..ordered_pair_count {
                assert_eq!(
                    product_quartets.coefficients_by_pair_indices(left_pair, right_pair),
                    product_coefficients_reference(
                        folded_products.row_by_pair_index(left_pair),
                        folded_products.row_by_pair_index(right_pair),
                        2,
                        &parent_weights,
                    )
                );
                let left = folded_ranges.value_by_pair_index(left_pair);
                let right = folded_ranges.value_by_pair_index(right_pair);
                assert_eq!(
                    range_quartets.coefficients_by_pair_indices(left_pair, right_pair),
                    compose_affine_reference(&range_coefficients, left, right - left),
                );
            }
        }
    }
}
